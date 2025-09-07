"""
Implementation of FINER (Flexible spectral-bias tuning in Implicit NEural Representation)

This module implements the paper:
'FINER: Flexible spectral-bias tuning in Implicit NEural Representation 
by Variable-periodic Activation Functions' (2023)

Reference:
    https://arxiv.org/abs/2312.02434
"""

from typing import List, Optional, Union, Tuple
import torch
import torch.nn as nn
import numpy as np


class FinerLayer(nn.Module):
    """FINER layer with variable-periodic activation functions.
    
    Attributes:
        omega_0 (float): Base frequency for sinusoidal activation
        is_first (bool): Whether this is the first layer
        input_dim (int): Input dimension
        linear (nn.Linear): Linear transformation layer
        scale_req_grad (bool): Whether scale requires gradients
        first_bias_scale (float): Scaling factor for first layer bias
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        omega_0: float = 30.0,
        bias: bool = True,
        is_first: bool = False,
        first_bias_scale: float = 0.0,
        scale_req_grad: bool = False
    ) -> None:
        """Initialize FINER layer.
        
        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output features
            omega_0: Base frequency for sinusoidal activation
            bias: Whether to include bias
            is_first: Whether this is the first layer
            first_bias_scale: Scaling factor for first layer bias
            scale_req_grad: Whether scale requires gradients
        """
        super().__init__()
        
        self.omega_0 = omega_0
        self.is_first = is_first
        self.input_dim = input_dim
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        self.scale_req_grad = scale_req_grad
        self.first_bias_scale = first_bias_scale
        
        self.init_weights()
        if self.first_bias_scale:
            self.init_first_bias()
    
    def init_weights(self) -> None:
        """Initialize layer weights using uniform distribution."""
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(
                    -1 / self.input_dim, 
                    1 / self.input_dim
                )      
            else:
                limit = np.sqrt(6 / self.input_dim) / self.omega_0
                self.linear.weight.uniform_(-limit, limit)

    def init_first_bias(self) -> None:
        """Initialize bias for first layer if specified."""
        with torch.no_grad():
            if self.is_first:
                self.linear.bias.uniform_(
                    -self.first_bias_scale, 
                    self.first_bias_scale
                )

    def generate_scale(self, x: torch.Tensor) -> torch.Tensor:
        """Generate scaling factor for variable-periodic activation.
        
        Args:
            x: Input tensor
            
        Returns:
            Scaling factor tensor
        """
        if self.scale_req_grad:
            return torch.abs(x) + 1
        with torch.no_grad():
            return torch.abs(x) + 1
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of FINER layer.
        
        Args:
            input: Input tensor
            
        Returns:
            Output tensor after variable-periodic activation
        """
        x = self.linear(input)
        scale = self.generate_scale(x)
        return torch.sin(self.omega_0 * scale * x)


class Finer(nn.Module):
    """FINER network with stacked variable-periodic layers.
    
    Attributes:
        net (nn.Sequential): Sequential container of FINER layers
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        omega_0: float = 30.0,
        omega_0_hidden: float = 30.0,
        first_bias_scale: float = 0.0,
        scale_req_grad: bool = False
    ) -> None:
        """Initialize FINER network.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layers
            num_layers: Number of hidden layers
            output_dim: Dimension of output
            omega_0: First layer frequency scaling
            omega_0_hidden: Hidden layers frequency scaling
            first_bias_scale: Scaling factor for first layer bias
            scale_req_grad: Whether scale requires gradients
        """
        super().__init__()
        
        layers: List[nn.Module] = []
        
        # First layer
        layers.append(
            FinerLayer(
                input_dim=input_dim,
                output_dim=hidden_dim,
                omega_0=omega_0,
                is_first=True,
                first_bias_scale=first_bias_scale,
                scale_req_grad=scale_req_grad
            )
        )
        
        # Hidden layers
        for _ in range(num_layers):
            layers.append(
                FinerLayer(
                    input_dim=hidden_dim,
                    output_dim=hidden_dim,
                    omega_0=omega_0_hidden,
                    scale_req_grad=scale_req_grad
                )
            )
        
        # Output layer
        final_linear = nn.Linear(hidden_dim, output_dim)
        with torch.no_grad():
            limit = np.sqrt(6 / hidden_dim) / omega_0_hidden
            final_linear.weight.uniform_(-limit, limit)
        layers.append(final_linear)
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Forward pass through FINER network.
        
        Args:
            coords: Input coordinates tensor
            
        Returns:
            Output tensor
        """
        return self.net(coords)
    
    def get_param_count(self) -> Tuple[int, int]:
        """Get number of trainable and total parameters.
        
        Returns:
            Tuple of (trainable_params, total_params)
        """
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total


def _test() -> None:
    """Test FINER network implementation."""
    # Network parameters
    params = {
        "input_dim": 3,
        "hidden_dim": 256,
        "num_layers": 5,
        "output_dim": 1,
        "omega_0": 30.0,
        "omega_0_hidden": 30.0,
        "first_bias_scale": 0.0,
        "scale_req_grad": False
    }
    
    # Create model and test input
    model = Finer(**params)
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