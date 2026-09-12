import math

import torch
from torch import nn
from typing import Callable


def _step_map(sn: torch.Tensor, m: torch.Tensor, alpha: float,
              tau_fixed: float | None, adaptive: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Single map-recurrence step. Kept as a standalone function so torch.compile
    can fuse it into ~1 kernel; `adaptive=False` when tau_fixed is used."""
    s_rel = (1 - alpha) * m + alpha
    if adaptive:
        diff = sn - s_rel
        tau = diff.abs().mean(dim=(1, 2), keepdim=True).clamp(0.0, 0.999)
    else:
        tau = tau_fixed
    new_map = (1 - tau) * s_rel + tau * sn
    return m, new_map


def _make_compiled_step() -> Callable:
    """torch.compile wrapper. `reduce-overhead` adds CUDA graphs (CUDA-only)."""
    torch._dynamo.config.recompile_limit = 64
    mode = "reduce-overhead" if torch.cuda.is_available() else "default"
    return torch.compile(_step_map, fullgraph=True, mode=mode)


class ShuntingLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        smap_bias_init: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.smap_bias_init = smap_bias_init

        self.shunt_w1 = nn.Linear(in_features, out_features * rank, bias=False)
        self.shunt_w2 = nn.Linear(in_features, in_features * rank, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1 = self.shunt_w1(x).view(*x.shape[:-1], self.out_features, self.rank)
        w2 = self.shunt_w2(x).view(*x.shape[:-1], self.in_features, self.rank)
        pre = w1 @ w2.mT
        return torch.sigmoid(pre + self.smap_bias_init)


class SIMNLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 0.05,
        bias: bool = False,
        tau_fixed: float | None = None,
        smap_bias_init: float = 0.0,
        torch_compile: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.tau_fixed = tau_fixed
        self._torch_compile = torch_compile
        self._step_fn = None

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))

        self.shunting_linear = ShuntingLinear(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            smap_bias_init=smap_bias_init,
        )

    def _compiled_step(self) -> Callable | None:
        """Lazily build & validate the fused per-step map update (torch.compile)."""
        if self._step_fn is None and self._torch_compile:
            fn = _make_compiled_step()
            try:
                dev = next(iter(self.parameters())).device
                sn = torch.randn(2, self.out_features, self.in_features, device=dev)
                m = torch.ones(2, self.out_features, self.in_features, device=dev)
                fn(sn, m, self.alpha, self.tau_fixed, self.tau_fixed is None)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                elif dev.type == "mps":
                    torch.mps.synchronize()
                self._step_fn = fn
            except Exception:
                torch._dynamo.config.suppress_errors = True
                self._step_fn = None
        return self._step_fn

    def step(
        self,
        x: torch.Tensor,
        shunting_map: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert x.dim() == 2, f"expected input (batch, features), got {tuple(x.shape)}"
        assert shunting_map.shape[:1] == x.shape[:1], (x.shape, shunting_map.shape)

        actual_weight = self.weight.unsqueeze(0) * shunting_map
        out = torch.bmm(actual_weight, x.unsqueeze(-1)).squeeze(-1)
        if self.bias is not None:
            out = out + self.bias

        s_relaxed = (1 - self.alpha) * shunting_map + self.alpha
        s_new = self.shunting_linear(x)
        if self.tau_fixed is None:
            tau = s_new.sub(s_relaxed).abs().mean(dim=(1, 2), keepdim=True)
            tau = tau.clamp(min=0.0, max=0.999)
        else:
            tau = torch.as_tensor(self.tau_fixed, dtype=s_new.dtype, device=s_new.device)
        new_map = (1 - tau) * s_relaxed + tau * s_new
        return out, new_map

    def integrate(
        self,
        x_all: torch.Tensor,
        shunting_map: torch.Tensor,
        store_maps: bool = False,
        chunk: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Time-batched pass over a full sequence (exact same recurrence as `step`).

        x_all: (batch, time, in_features).
        Returns (out, last_map, maps_all). `out` is (batch, time, out_features)
        when store_maps=True, otherwise (batch, out_features) for the last step.
        `maps_all` holds the pre-update map of every timestep (None if not stored).
        Mathematically identical to looping `step`; only GEMM reduction order differs.
        """
        assert x_all.dim() == 3, f"expected input (batch, time, features), got {tuple(x_all.shape)}"
        batch, time, _ = x_all.shape
        out_features, in_features = self.out_features, self.in_features
        chunk = time if chunk is None else max(chunk, 1)

        maps_chunks: list[torch.Tensor] = []
        last_map = shunting_map
        out_map = shunting_map
        step_fn = self._compiled_step()
        adaptive = self.tau_fixed is None

        for t0 in range(0, time, chunk):
            t1 = min(t0 + chunk, time)
            s_new = self.shunting_linear(x_all[:, t0:t1])
            if step_fn is not None:
                s_new = s_new.movedim(1, 0)
            m = last_map
            ms = []
            if not adaptive:
                tau = self.tau_fixed
            for j in range(t1 - t0):
                sn = s_new[j] if step_fn is not None else s_new[:, j]
                if step_fn is not None:
                    out_map, m = step_fn(sn, m, self.alpha, self.tau_fixed, adaptive)
                else:
                    out_map = m
                    s_rel = (1 - self.alpha) * m + self.alpha
                    if adaptive:
                        tau = (sn - s_rel).abs().mean(dim=(1, 2), keepdim=True).clamp(0.0, 0.999)
                    m = (1 - tau) * s_rel + tau * sn
                if store_maps:
                    ms.append(out_map)
            last_map = m
            if store_maps:
                maps_chunks.append(torch.stack(ms, dim=1))

        maps_all = torch.cat(maps_chunks, dim=1) if store_maps else None

        if store_maps:
            actual_weight = self.weight.unsqueeze(0).unsqueeze(0) * maps_all
            flat = torch.bmm(
                actual_weight.reshape(-1, out_features, in_features),
                x_all.reshape(-1, in_features).unsqueeze(-1),
            )
            out = flat.reshape(batch, time, out_features)
        else:
            actual_weight = self.weight.unsqueeze(0) * out_map
            out = torch.bmm(actual_weight, x_all[:, -1].unsqueeze(-1)).squeeze(-1)
        if self.bias is not None:
            out = out + self.bias
        return out, last_map, maps_all