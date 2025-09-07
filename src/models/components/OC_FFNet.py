"""
Implementation of OC-FFNet
"""

from typing import Optional, Tuple, Literal
import torch
import torch.nn as nn
import numpy as np
import math


class FourierFeatureMapping(nn.Module):
    """Fourier feature mapping module for input coordinate lifting.
    
    Maps input coordinates to a higher dimensional space using random Fourier features,
    enabling better learning of high-frequency functions.
    
    Attributes:
        input_dim (int): Dimensionality of input coordinates
        mapping_size (int): Output dimension of the Fourier mapping
        sigma (float): Standard deviation for feature sampling
        B (nn.Parameter): Random Fourier feature matrix
    """
    
    def __init__(
        self, 
        input_dim: int,
        mapping_size: int,
        sigma: float = 1.0
    ) -> None:
        """Initialize Fourier feature mapping.
        
        Args:
            input_dim: Number of input dimensions
            mapping_size: Size of the feature mapping (must be even)
            sigma: Standard deviation for sampling feature matrix
            
        Raises:
            ValueError: If mapping_size is not even
        """
        super().__init__()
        
        if mapping_size % 2 != 0:
            raise ValueError(
                f"mapping_size must be even, got {mapping_size}"
            )
            
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.sigma = sigma
        
        # Initialize random Fourier features
        self.B = nn.Parameter(
            torch.randn(input_dim, mapping_size // 2) * sigma,
            requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Fourier feature mapping to input coordinates.
        
        Args:
            x: Input coordinates of shape (batch_size, input_dim)
            
        Returns:
            Fourier features of shape (batch_size, mapping_size)
            
        Raises:
            ValueError: If input dimensions don't match expected shape
        """
        if x.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected input dimension {self.input_dim}, got {x.size(-1)}"
            )
            
        # Project and apply sinusoidal activation
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class TimeEmbed(nn.Module):
    """Sinusoidal time embeddings for diffusion-style conditioning."""
    
    def __init__(self, dim: int) -> None:
        """Initialize time embedding module.
        
        Args:
            dim: Dimensionality of the time embedding
        """
        super().__init__()
        self.dim = dim
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Compute sinusoidal time embeddings.
        
        Args:
            t: Time tensor of shape (B,) or scalar
            
        Returns:
            Time embeddings of shape (B, dim)
        """
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half_dim, dtype=torch.float32) / half_dim
        ).to(t.device)
        
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if self.dim % 2 != 0:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            
        return embedding


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation (FiLM) layer."""

    def __init__(
        self,
        time_dim: int,
        feature_dim: int,
        activation: nn.Module = nn.SiLU()
    ) -> None:
        """Initialize the FiLM layer.
        
        Args:
            time_dim: Dimensionality of the time input
            feature_dim: Dimensionality of the features to modulate
            activation: Activation function for the MLP
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(time_dim, feature_dim * 2),
            activation
        )
        
    def forward(self, features: torch.Tensor, time_cond: torch.Tensor) -> torch.Tensor:
        """Apply FiLM modulation.
        
        Args:
            features: Input features of shape (B, D)
            time_cond: Time conditioning of shape (B, T_dim)
            
        Returns:
            Modulated features of shape (B, D)
        """
        gamma, beta = self.net(time_cond).chunk(2, dim=-1)
        return (1 + gamma) * features + beta


class MLPBlock(nn.Module):
    """Basic MLP block with FiLM conditioning."""
    
    def __init__(
        self,
        dim: int,
        time_dim: int,
        dropout_rate: float,
        activation: nn.Module,
        use_film: bool
    ) -> None:
        super().__init__()
        self.use_film = use_film
        
        self.norm = nn.LayerNorm(dim, elementwise_affine=not use_film)
        self.activation = activation
        self.linear = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        if use_film:
            self.film = FiLMLayer(time_dim, dim, activation)
            
    def forward(self, x: torch.Tensor, time_cond: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        if self.use_film:
            x_norm = self.film(x_norm, time_cond)
            
        out = self.linear(x_norm)
        out = self.activation(out)
        out = self.dropout(out)
        return out


class ResidualBlock(nn.Module):
    """Residual block with FiLM conditioning."""
    
    def __init__(
        self,
        dim: int,
        time_dim: int,
        dropout_rate: float,
        activation: nn.Module,
        use_film: bool
    ) -> None:
        super().__init__()
        self.use_film = use_film
        
        self.norm = nn.LayerNorm(dim, elementwise_affine=not use_film)
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)
        
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        
        if self.use_film:
            self.film = FiLMLayer(time_dim, dim, activation)

    def forward(self, x: torch.Tensor, time_cond: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.norm(x)
        if self.use_film:
            out = self.film(out, time_cond)
            
        out = self.linear1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        out = self.linear2(out)
        out = self.dropout(out)
        
        return identity + out


class ODEFunc(nn.Module):
    """ODE dynamics function f(z, t)."""
    
    def __init__(
        self,
        dim: int,
        num_layers: int,
        dropout_rate: float,
        activation: nn.Module,
        block_type: Literal["mlp", "residual"],
        fusion_mode: Literal["cat", "film", "film_time_embed"],
        time_embed_dim: int
    ) -> None:
        super().__init__()
        self.fusion_mode = fusion_mode

        is_concat_mode = fusion_mode == "cat"
        use_film_in_block = fusion_mode in ["film", "film_time_embed"]

        block_dim = dim + 1 if is_concat_mode else dim
        self.output_proj = nn.Linear(block_dim, dim) if is_concat_mode else nn.Identity()

        if fusion_mode == "film_time_embed":
            self.time_embed = TimeEmbed(time_embed_dim)
            time_cond_dim = time_embed_dim
        elif fusion_mode == "film":
            self.time_embed = None
            time_cond_dim = 1
        else:  # "cat"
            self.time_embed = None
            time_cond_dim = 0  # Not used

        Block = {"mlp": MLPBlock, "residual": ResidualBlock}[block_type]

        self.layers = nn.ModuleList([
            Block(
                dim=block_dim,
                time_dim=time_cond_dim,
                dropout_rate=dropout_rate,
                activation=activation,
                use_film=use_film_in_block
            ) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, t: float) -> torch.Tensor:
        if self.fusion_mode == "cat":
            t_vec = torch.full(
                (x.shape[0], 1), t, device=x.device, dtype=x.dtype
            )
            x = torch.cat([x, t_vec], dim=1)
            time_cond = None
        else:
            t_tensor = torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype)
            if self.fusion_mode == "film_time_embed":
                time_cond = self.time_embed(t_tensor)
            else:  # "film"
                time_cond = t_tensor.unsqueeze(1)
            
        for layer in self.layers:
            x = layer(x, time_cond)
            
        return self.output_proj(x)


class OCFourierFeatureNetwork(nn.Module):
    """Optimal Control-regularized FourierFeatureNetwork."""
    
    VALID_ACTIVATIONS = {
        "ReLU": nn.ReLU(),
        "GELU": nn.GELU(),
        "SiLU": nn.SiLU(),
        "LeakyReLU": nn.LeakyReLU(),
        "Sigmoid": nn.Sigmoid(),
        "Tanh": nn.Tanh(),
        "ELU": nn.ELU(),
        "SELU": nn.SELU(),
        "Mish": nn.Mish(),
        "Identity": nn.Identity()
    }
    
    def __init__(
        self,
        input_dim: int,
        mapping_size: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout_rate: float,
        activation: str,
        block_type: Literal["mlp", "residual"],
        fusion_mode: Literal["cat", "film", "film_time_embed"],
        time_embed_dim: int,
        num_steps: int,
        total_time: float = 1.0,
        ot_lambda: float = 1.0,
        sigma: float = 1.0,
        final_activation: Optional[str] = None
    ) -> None:
        """Initialize the OC-FourierFeatureNetwork model.
        
        Args:
            input_dim: Number of input dimensions
            mapping_size: Size of Fourier feature mapping
            hidden_dim: Width of hidden layers
            output_dim: Number of output dimensions
            num_steps: Number of discretization steps for the ODE
            total_time: Total integration time T for the ODE
            ot_lambda: Weight for the optimal transport regularization
            sigma: Standard deviation for Fourier features
            final_activation: Optional activation for the output layer
            
        Raises:
            ValueError: If an unsupported activation name is provided
        """
        super().__init__()
        
        if final_activation and final_activation not in self.VALID_ACTIVATIONS:
            raise ValueError(
                f"Unsupported final activation: {final_activation}. "
                f"Choose from {list(self.VALID_ACTIVATIONS.keys())}"
            )
            
        self.total_time = total_time
        self.num_steps = num_steps
        self.ot_lambda = ot_lambda
        
        # Initial embedding: z(0) = phi(x)
        self.fourier_features = torch.jit.script(FourierFeatureMapping(
            input_dim=input_dim,
            mapping_size=mapping_size,
            sigma=sigma
        ))

        # Projection from mapping_size to hidden_dim if needed
        if mapping_size != hidden_dim:
            self.input_proj = nn.Linear(mapping_size, hidden_dim)
        else:
            self.input_proj = nn.Identity()

        self.ode_func = ODEFunc(dim=hidden_dim, num_layers=num_layers, dropout_rate=dropout_rate, activation=self._get_activation(activation), block_type=block_type, fusion_mode=fusion_mode, time_embed_dim=time_embed_dim)
                 
        # Output projection
        output_layers = [nn.Linear(hidden_dim, output_dim)]
        if final_activation:
            output_layers.append(self._get_activation(final_activation))
        self.output_proj = nn.Sequential(*output_layers)
    
    @classmethod
    def _get_activation(cls, activation_name: str) -> nn.Module:
        """Get activation function by name."""
        if activation_name not in cls.VALID_ACTIVATIONS:
            raise ValueError(
                f"Unsupported activation: {activation_name}. "
                f"Choose from {list(cls.VALID_ACTIVATIONS.keys())}"
            )
        return cls.VALID_ACTIVATIONS[activation_name]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the OC-FourierFeatureNetwork.
        
        Args:
            x: Input coordinates of shape (batch_size, input_dim)
            
        Returns:
            Network output of shape (batch_size, output_dim)
        """
        # Set initial state z(0) = phi(x)
        z = self.fourier_features(x)
        z = self.input_proj(z)
        
        # Solve ODE using Euler's method
        ot_accum = 0.0
        dt = self.total_time / self.num_steps
        for i in range(self.num_steps):
            t = i * dt
            v = self.ode_func(z, t)
            # ot_accum = ot_accum + v.pow(2).sum(dim=-1).mean()
            ot_accum = ot_accum + v.pow(2).mean()
            z = z + dt * v

        ot_reg = 0.5 * self.ot_lambda * dt * ot_accum
        
        # Output projection from final state z(T)
        return self.output_proj(z), ot_reg
    
    def get_param_count(self) -> Tuple[int, int]:
        """Get number of trainable and total parameters.
        
        Returns:
            Tuple of (trainable_params, total_params)
        """
        trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in self.parameters())
        return trainable_params, total_params


def _test():
    """Run tests for the OC-FourierFeatureNetwork network."""
    # Network parameters
    input_dim = 3
    mapping_size = 256
    hidden_dim = 256
    num_steps = 10
    output_dim = 1
    sigma = 10.0
    final_activation = None
    num_layers = 3
    dropout_rate = 0.1
    activation = "GELU"
    block_type = "residual"
    fusion_mode = "film_time_embed"
    time_embed_dim = 64
    
    # Base ODE config for sweepable hyperparameters
    base_ode_config = {
        "num_layers": 3,
        "dropout_rate": 0.1,
        "activation": "GELU",
        "block_type": "residual",
        "fusion_mode": "film_time_embed",
        "time_embed_dim": 64
    }

    configs = {
        "Default (film_time_embed, residual)": base_ode_config,
        "FiLM with plain time": {**base_ode_config, "fusion_mode": "film"},
        "MLP Blocks": {**base_ode_config, "block_type": "mlp"},
        "Concat Fallback": {**base_ode_config, "fusion_mode": "cat"}
    }

    for name, ode_config in configs.items():
        print(f"\n--- Testing: {name} ---")
        
        model = OCFourierFeatureNetwork(
            input_dim=input_dim,
            mapping_size=mapping_size,
            hidden_dim=hidden_dim,
            num_steps=num_steps,
            output_dim=output_dim,
            sigma=sigma,
            final_activation=final_activation,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            activation=activation,
            block_type=block_type,
            fusion_mode=fusion_mode,
            time_embed_dim=time_embed_dim
        )

        batch_size = 16
        x = torch.rand(batch_size, input_dim)
        y, ot_reg = model(x)
        
        trainable_params, total_params = model.get_param_count()
        
        print(f"Model Architecture:")
        print(f"  Input dim: {input_dim}")
        print(f"  Mapping size: {mapping_size}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Num steps (M): {num_steps}")
        print(f"  Output dim: {output_dim}")
        print(f"  ODE Config: {ode_config}")
        print(f"\nParameters:")
        print(f"  Trainable: {trainable_params:,}")
        print(f"  Total: {total_params:,}")
        print(f"\nShapes:")
        print(f"  Input: {x.shape}")
        print(f"  Output: {y.shape}")
        assert y.shape == (batch_size, output_dim)
        print(f"  OT Reg value: {ot_reg.item():.4f}")


if __name__ == "__main__":
    _test()