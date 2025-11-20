"""Reusable MLP module for message passing components."""

import torch.nn as nn
from .config import MLPConfig


class MLP(nn.Module):
    """Multi-layer perceptron with configurable depth and hidden size.

    Architecture:
        input_dim → hidden_size (+ ReLU) → ... → hidden_size (+ ReLU) → output_dim

    The number of hidden layers is controlled by `mlp_config.num_layers`:
        - num_layers = 1: input → hidden (+ReLU) → output
        - num_layers = 2: input → hidden (+ReLU) → hidden (+ReLU) → output
        - num_layers = N: (N-1) hidden layers with ReLU, then final output layer

    Args:
        in_dim: Input feature dimension
        out_dim: Output feature dimension
        mlp_config: Configuration containing hidden_size and num_layers
        device: Device to create layers on (CPU or CUDA)

    Example:
        >>> config = MLPConfig(hidden_size=128, num_layers=2)
        >>> mlp = MLP(in_dim=64, out_dim=32, mlp_config=config, device='cpu')
        >>> # Creates: Linear(64→128) → ReLU → Linear(128→128) → ReLU → Linear(128→32)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        mlp_config: MLPConfig,
        device: str
    ):
        super().__init__()

        layers = []

        # First layer: input_dim → hidden_size
        layers.append(nn.Linear(in_dim, mlp_config.hidden_size, device=device))
        layers.append(nn.ReLU())

        # Middle layers: hidden_size → hidden_size (repeat num_layers - 1 times)
        for _ in range(mlp_config.num_layers - 1):
            layers.append(
                nn.Linear(mlp_config.hidden_size, mlp_config.hidden_size, device=device)
            )
            layers.append(nn.ReLU())

        # Final layer: hidden_size → output_dim
        layers.append(
            nn.Linear(mlp_config.hidden_size, out_dim, device=device)
        )

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the MLP.

        Args:
            x: Input tensor of shape (..., in_dim)

        Returns:
            Output tensor of shape (..., out_dim)
        """
        return self.network(x)
