from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class MaskingMLP(nn.Module):
    """Simple fully-connected network for masked inverse-kinetics regression."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: Sequence[int] = (30, 30),
        activation: str = "tanh",
    ) -> None:
        super().__init__()

        if isinstance(hidden_sizes, int):
            hidden_sizes = (hidden_sizes,)
        hidden_sizes = tuple(int(h) for h in hidden_sizes)
        if len(hidden_sizes) == 0:
            raise ValueError("hidden_sizes must contain at least one layer size")

        activation_layer: nn.Module
        if activation == "tanh":
            activation_layer = nn.Tanh()
        elif activation == "relu":
            activation_layer = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation '{activation}'. Use 'tanh' or 'relu'.")

        layers: list[nn.Module] = []
        previous_dim = int(input_dim)
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(type(activation_layer)())
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, int(output_dim)))

        self.network = nn.Sequential(*layers)
        self.hidden_sizes = hidden_sizes
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
