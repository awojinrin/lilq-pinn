"""
Neural Network Architectures for LiL-Q
========================================

Multi-layer perceptron (MLP) and parameter-matching utilities
used by NiL-N and NiL-Q solvers.
"""

import math
import numpy as np
import torch
import torch.nn as nn


def calculate_hidden_dim(n_hidden_layers: int, N_x: int, N_y: int) -> int:
    """Calculate NN hidden layer width to match basis coefficient DOF count.

    Solves for the hidden dimension ``h`` such that the total number of
    learnable parameters in an MLP(2 → h → ... → h → 1) with
    ``n_hidden_layers`` hidden layers approximately equals ``N_x * N_y``
    (the number of basis coefficients in the LiL formulation).

    Parameters
    ----------
    n_hidden_layers : int
        Number of hidden layers in the MLP.
    N_x, N_y : int
        Basis function counts per spatial direction.

    Returns
    -------
    int
        Ceiling of the required hidden dimension.
    """
    n_target = N_x * N_y
    nh = n_hidden_layers

    if nh <= 1:
        return math.ceil(n_target / 4)

    a = nh - 1
    b = 3 + nh
    c = -n_target

    discriminant = b ** 2 - 4 * a * c
    nl1 = (-b + np.sqrt(discriminant)) / (2 * a)
    nl2 = (-b - np.sqrt(discriminant)) / (2 * a)

    return math.ceil(abs(max(nl1, nl2)))


class MLP(nn.Module):
    """Standard multi-layer perceptron for PINN / NiL solvers.

    Architecture::

        Linear(input_dim, hidden_dim) → activation
        [Linear(hidden_dim, hidden_dim) → activation] × (num_layers - 1)
        Linear(hidden_dim, output_dim) [→ output_activation]

    Parameters
    ----------
    input_dim : int
        Dimension of the input (default 2 for (x, y) or (x, t)).
    hidden_dim : int
        Width of each hidden layer.
    output_dim : int
        Dimension of the output (default 1 for scalar PDEs).
    num_layers : int
        Number of hidden layers (default 4).
    activation : nn.Module
        Activation function (default ``nn.Tanh()``; use ``nn.SiLU()``
        for Darcy/SPE10 problems).
    output_activation : nn.Module or None
        Optional activation after the final linear layer.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 50,
        output_dim: int = 1,
        num_layers: int = 4,
        activation: nn.Module = None,
        output_activation: nn.Module = None,
    ):
        super().__init__()

        if activation is None:
            activation = nn.Tanh()

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(activation)

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation)

        layers.append(nn.Linear(hidden_dim, output_dim))

        if output_activation is not None:
            layers.append(output_activation)

        self.network = nn.Sequential(*layers)
        self.n_params = sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
