"""
Implementation of WIRE (Wavelet Implicit Neural Representations)

This module implements the paper:
'WIRE: Wavelet Implicit Neural Representations'
(Sun et al., 2023)

Reference:
    https://arxiv.org/abs/2301.05187
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import math


class ComplexGaborLayer(nn.Module):
    """Complex Gabor nonlinearity layer for implicit representations.
    
    Implements a complex-valued neural network layer with Gabor nonlinearity,
    combining sinusoidal and Gaussian terms.
    
    Attributes:
        omega_0 (nn.Parameter): Frequency parameter for sinusoidal term
        scale_0 (nn.Parameter): Scale parameter for Gaussian term
        linear (nn.Linear): Complex-valued linear transformation
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        omega_0: float = 10.0,
        sigma_0: float = 40.0,
        is_first: bool = False,
        trainable: bool = False,
        bias: bool = True
    ) -> None:
        """Initialize Complex Gabor layer.
        
        Args:
            input_dim: Number of input features
            output_dim: Number of output features
            omega_0: Initial frequency for sinusoidal term
            sigma_0: Initial scale for Gaussian term
            is_first: Whether this is the first layer
            trainable: Whether omega_0 and sigma_0 are trainable
            bias: Whether to include bias in linear layer
            
        Raises:
            ValueError: If dimensions are invalid
        """
        super().__init__()
        
        if input_dim < 1 or output_dim < 1:
            raise ValueError(
                f"Dimensions must be positive, got input_dim={input_dim}, "
                f"output_dim={output_dim}"
            )
            
        self.omega_0 = nn.Parameter(
            torch.tensor(omega_0),
            requires_grad=trainable
        )
        self.scale_0 = nn.Parameter(
            torch.tensor(sigma_0),
            requires_grad=trainable
        )
        
        dtype = torch.float if is_first else torch.cfloat
        self.linear = nn.Linear(input_dim, output_dim, bias=bias, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass applying complex Gabor nonlinearity.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor after Gabor nonlinearity
        """
        # For first layer, convert real input to complex
        if self.linear.weight.dtype == torch.cfloat and x.dtype != torch.cfloat:
            x = torch.complex(x, torch.zeros_like(x))
            
        lin = self.linear(x)
        omega = self.omega_0 * lin
        scale = self.scale_0 * lin
        return torch.exp(1j * omega - scale.abs().square())


class WIRE(nn.Module):
    """Wavelet Implicit Neural Representation (WIRE) Network.
    
    Implements a neural network using complex-valued Gabor nonlinearities
    for implicit representation learning.
    
    Attributes:
        net (nn.Sequential): Sequential container of network layers
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
        scale: float = 10.0
    ) -> None:
        """Initialize WIRE network.
        
        Args:
            input_dim: Dimension of input coordinates
            hidden_dim: Width of hidden layers
            num_layers: Number of hidden layers
            output_dim: Dimension of output
            first_omega_0: Frequency for first layer
            hidden_omega_0: Frequency for hidden layers
            scale: Scale factor for Gabor nonlinearity
            
        Raises:
            ValueError: If network parameters are invalid
        """
        super().__init__()
        
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
            
        # Adjust hidden dimension for complex values
        hidden_dim = int(hidden_dim / math.sqrt(2.0))
        
        # Build network layers
        layers: List[nn.Module] = [
            ComplexGaborLayer(
                input_dim=input_dim,
                output_dim=hidden_dim,
                omega_0=first_omega_0,
                sigma_0=scale,
                is_first=True
            )
        ]
        
        for _ in range(num_layers):
            layers.append(
                ComplexGaborLayer(
                    input_dim=hidden_dim,
                    output_dim=hidden_dim,
                    omega_0=hidden_omega_0,
                    sigma_0=scale
                )
            )
            
        layers.append(nn.Linear(hidden_dim, output_dim, dtype=torch.cfloat))
        self.net = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Forward pass through WIRE network.
        
        Args:
            coords: Input coordinates tensor
            
        Returns:
            Real part of output tensor
        """
        # Ensure input is float tensor
        if not torch.is_tensor(coords):
            coords = torch.tensor(coords, dtype=torch.float32)
        elif coords.dtype not in [torch.float32, torch.float64]:
            coords = coords.to(torch.float32)
        
        # Keep input as real numbers, ComplexGaborLayer will handle conversion
        return self.net(coords).real
    
    def get_param_count(self) -> Tuple[int, int]:
        """Get number of trainable and total parameters.
        
        Returns:
            Tuple of (trainable_params, total_params)
        """
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total


def _test() -> None:
    """Test WIRE network implementation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Network parameters
    params = {
        "input_dim": 3,
        "hidden_dim": 256,
        "num_layers": 5,
        "output_dim": 1,
        "first_omega_0": 30.0,
        "hidden_omega_0": 30.0,
        "scale": 10.0
    }
    
    try:
        # Create model and test input
        model = WIRE(**params).to(device)
        batch_size = 16
        example_input = torch.rand(
            batch_size,
            params["input_dim"],
            dtype=torch.float32,  # Explicitly use float32
            device=device
        )
        
        # Test forward pass
        output = model(example_input)
        
        # Verify output properties
        assert output.shape == (batch_size, params["output_dim"]), \
            f"Unexpected output shape: {output.shape}"
        assert output.dtype == torch.float32, \
            f"Unexpected output dtype: {output.dtype}"
        
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
        print(f"\nTest passed successfully!")
        
    except Exception as e:
        print(f"Test failed with error: {str(e)}")


if __name__ == "__main__":
    _test()