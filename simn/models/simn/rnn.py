import torch
from torch import nn

from simn.models.layers import SIMNLinear


class SIMNCell(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 0.05,
        bias: bool = True,
        tau_fixed: float | None = None,
        smap_bias_init: float = 0.0,
        chunk: int | None = None,
        torch_compile: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.chunk = chunk
        self.pre_norm = None
        # fixed pre-norm (RMSNorm); skipped on 1-dim inputs (sign-collapse)
        if in_features > 1:
            self.pre_norm = nn.RMSNorm(in_features)

        self.linear = SIMNLinear(
            in_features,
            out_features,
            rank=rank,
            alpha=alpha,
            bias=bias,
            tau_fixed=tau_fixed,
            smap_bias_init=smap_bias_init,
            torch_compile=torch_compile,
        )

    def reset_shunting_state(self) -> None:
        pass

    def step(
        self,
        x: torch.Tensor,
        shunting_map: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.pre_norm is not None:
            x = self.pre_norm(x)
        out, new_map = self.linear.step(x, shunting_map)
        return torch.tanh(out), new_map

    def forward_batched(
        self,
        x_all: torch.Tensor,
        shunting_map: torch.Tensor,
        store_maps: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if self.pre_norm is not None:
            x_all = self.pre_norm(x_all)
        out, last_map, maps_all = self.linear.integrate(
            x_all, shunting_map, store_maps=store_maps, chunk=self.chunk
        )
        return torch.tanh(out), last_map, maps_all


class SIMNRNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        rank: int = 8,
        alpha: float = 0.05,
        bias: bool = True,
        tau_fixed: float | None = None,
        smap_bias_init: float = 0.0,
        chunk: int | None = None,
        torch_compile: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.chunk = chunk
        self.torch_compile = torch_compile

        self.layers = nn.ModuleList()
        for l in range(num_layers):
            in_features = input_size if l == 0 else hidden_size
            self.layers.append(
                SIMNCell(
                    in_features,
                    hidden_size,
                    rank=rank,
                    alpha=alpha,
                    bias=bias,
                    tau_fixed=tau_fixed,
                    smap_bias_init=smap_bias_init,
                    chunk=chunk,
                    torch_compile=torch_compile,
                )
            )
        self.head = nn.Linear(hidden_size, output_size, bias=bias)

    def reset_shunting_state(self) -> None:
        pass

    def _init_maps(self, batch: int, device, dtype) -> list[torch.Tensor]:
        return [
            torch.ones(
                batch,
                cell.linear.out_features,
                cell.linear.in_features,
                device=device,
                dtype=dtype,
            )
            for cell in self.layers
        ]

    def _step_all(
        self,
        x_t: torch.Tensor,
        maps: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        h = x_t
        for l, cell in enumerate(self.layers):
            h, maps[l] = cell.step(h, maps[l])
        return h, maps

    def _forward_batched(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        batch, _, _ = x.shape
        maps = self._init_maps(batch, x.device, x.dtype)
        last_layer = len(self.layers) - 1
        h = x
        for l, cell in enumerate(self.layers):
            h, maps[l], _ = cell.forward_batched(
                h, maps[l], store_maps=l < last_layer
            )
        return h, maps

    def _forward_sequential(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        batch, time, _ = x.shape
        maps = self._init_maps(batch, x.device, x.dtype)
        h = None
        for t in range(time):
            h, maps = self._step_all(x[:, t], maps)
        return h, maps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3, f"expected input (batch, time, {self.input_size}), got {tuple(x.shape)}"
        h, _ = self._forward_batched(x)
        return self.head(h)

    def init_state(self, context: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        context = context.to(next(self.parameters()).device)
        h, maps = self._forward_batched(context)
        return self.head(h), maps

    def step(
        self,
        x: torch.Tensor,
        maps: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = x.to(next(self.parameters()).device)
        h, maps = self._step_all(x, maps)
        return self.head(h), maps
