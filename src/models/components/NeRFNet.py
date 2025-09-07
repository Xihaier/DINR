"""
Implementation of NeRF (Neural Radiance Fields)

This module implements the paper:
'NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis'
(Mildenhall et al., 2020)

Reference:
    https://arxiv.org/abs/2003.08934
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np


class PositionalEncoding(nn.Module):
    """Positional encoding module for coordinate embedding.
    
    Transforms input coordinates into a higher dimensional space using
    frequency-based encoding, enabling better representation of high-frequency
    functions.
    
    Attributes:
        input_dim (int): Dimension of input coordinates
        num_frequencies (int): Number of frequency bands
        freq_bands (torch.Tensor): Frequency bands for encoding
        output_dim (int): Dimension of encoded output
    """
    
    def __init__(
        self,
        input_dim: int,
        num_frequencies: int
    ) -> None:
        """Initialize positional encoding.
        
        Args:
            input_dim: Dimension of input coordinates
            num_frequencies: Number of frequency bands
            
        Raises:
            ValueError: If input_dim or num_frequencies is less than 1
        """
        super().__init__()
        
        if input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {input_dim}")
        if num_frequencies < 1:
            raise ValueError(f"num_frequencies must be >= 1, got {num_frequencies}")
            
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.output_dim = input_dim * 2 * num_frequencies
        
        # Generate frequency bands: 2^0, 2^1, ..., 2^(L-1)
        self.register_buffer(
            'freq_bands',
            2.0 ** torch.linspace(0.0, num_frequencies - 1, num_frequencies)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply positional encoding to input coordinates.
        
        Args:
            x: Input coordinates of shape (batch_size, input_dim)
            
        Returns:
            Encoded coordinates of shape (batch_size, input_dim * 2 * num_frequencies)
            
        Raises:
            ValueError: If input shape doesn't match expected dimensions
        """
        if x.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected input dimension {self.input_dim}, got {x.size(-1)}"
            )
            
        # Create high-dimensional embedding
        x = x.unsqueeze(-1)  # [B, D, 1]
        encoded = (x * self.freq_bands).view(-1, self.input_dim * self.num_frequencies)
        return torch.cat([torch.sin(encoded), torch.cos(encoded)], dim=-1)


class NeRF(nn.Module):
    """Neural Radiance Field network with positional encoding.
    
    Combines positional encoding with an MLP to learn continuous scene
    representations through coordinate-based neural networks.
    
    Attributes:
        positional_encoding (PositionalEncoding): Coordinate embedding module
        mlp (nn.Sequential): Multi-layer perceptron
    """
    
    def __init__(
        self,
        input_dim: int,
        num_frequencies: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int
    ) -> None:
        """Initialize NeRF network.
        
        Args:
            input_dim: Dimension of input coordinates
            num_frequencies: Number of frequency bands for encoding
            hidden_dim: Width of hidden layers
            num_layers: Number of hidden layers
            output_dim: Dimension of output features
            
        Raises:
            ValueError: If network parameters are invalid
        """
        super().__init__()
        
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
            
        self.positional_encoding = PositionalEncoding(input_dim, num_frequencies)
        encoded_dim = self.positional_encoding.output_dim

        # Build MLP layers
        layers: List[nn.Module] = []
        layers.append(nn.Linear(encoded_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through NeRF network.
        
        Args:
            x: Input coordinates of shape (batch_size, input_dim)
            
        Returns:
            Output features of shape (batch_size, output_dim)
        """
        x = self.positional_encoding(x)
        return self.mlp(x)
    
    def get_param_count(self) -> Tuple[int, int]:
        """Get number of trainable and total parameters.
        
        Returns:
            Tuple of (trainable_params, total_params)
        """
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total


def _test() -> None:
    """Test NeRF network implementation."""
    # Network parameters
    params = {
        "input_dim": 3,
        "num_frequencies": 10,
        "hidden_dim": 256,
        "num_layers": 5,
        "output_dim": 1
    }
    
    # Create model and test input
    model = NeRF(**params)
    batch_size = 16
    example_input = torch.rand(batch_size, params["input_dim"])
    output = model(example_input)
    
    # Print model information
    trainable_params, total_params = model.get_param_count()
    
    print(f"Model Architecture:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    print(f"\nParameters:")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Total: {total_params:,}")
    print(f"\nShapes:")
    print(f"  Input: {example_input.shape}")
    print(f"  Output: {output.shape}")


if __name__ == "__main__":
    _test()