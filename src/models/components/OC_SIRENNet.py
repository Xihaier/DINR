"""
Implementation of OC-SIREN
"""

from typing import Optional, Tuple, Literal
import torch
import torch.nn as nn
import math


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


class SIRENLayer(nn.Module):
    """SIREN layer with sinusoidal activation function and optional FiLM conditioning.
    
    Attributes:
        input_dim (int): Number of input features
        output_dim (int): Number of output features
        omega_0 (float): Angular frequency factor
        is_first (bool): Whether this is the first layer
        linear (nn.Linear): Linear transformation layer
        use_film (bool): Whether to use FiLM conditioning
        film (FiLMLayer): FiLM layer for time conditioning
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        omega_0: float = 30.0,
        is_first: bool = False,
        time_dim: int = 0,
        use_film: bool = False
    ) -> None:
        """Initialize SIREN layer.
        
        Args:
            input_dim: Number of input features
            output_dim: Number of output features
            omega_0: Angular frequency factor
            is_first: Whether this is the first layer
            time_dim: Dimensionality of time conditioning (if using FiLM)
            use_film: Whether to use FiLM conditioning
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.omega_0 = omega_0
        self.is_first = is_first
        self.use_film = use_film
        
        self.linear = nn.Linear(input_dim, output_dim)
        self.init_weights()
        
        if use_film and time_dim > 0:
            self.film = FiLMLayer(time_dim, output_dim)

    def init_weights(self) -> None:
        """Initialize weights using uniform distribution."""
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(
                    -1 / self.input_dim,
                    1 / self.input_dim
                )
            else:
                limit = math.sqrt(6 / self.input_dim) / self.omega_0
                self.linear.weight.uniform_(-limit, limit)

    def forward(self, x: torch.Tensor, time_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with sine activation and optional FiLM conditioning.
        
        Args:
            x: Input tensor
            time_cond: Optional time conditioning tensor for FiLM
            
        Returns:
            Output tensor after sinusoidal activation
        """
        out = self.linear(x)
        
        if self.use_film and time_cond is not None:
            out = self.film(out, time_cond)
            
        return torch.sin(self.omega_0 * out)


class SIRENBlock(nn.Module):
    """SIREN block with optional FiLM conditioning for ODE dynamics."""
    
    def __init__(
        self,
        dim: int,
        time_dim: int,
        omega_0: float,
        use_film: bool,
        dropout_rate: float = 0.0
    ) -> None:
        """Initialize SIREN block.
        
        Args:
            dim: Feature dimension
            time_dim: Time conditioning dimension
            omega_0: Angular frequency factor
            use_film: Whether to use FiLM conditioning
            dropout_rate: Dropout rate (applied after activation)
        """
        super().__init__()
        self.use_film = use_film
        
        self.siren_layer = SIRENLayer(
            input_dim=dim,
            output_dim=dim,
            omega_0=omega_0,
            is_first=False,
            time_dim=time_dim,
            use_film=use_film
        )
        
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
    def forward(self, x: torch.Tensor, time_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through SIREN block.
        
        Args:
            x: Input features
            time_cond: Optional time conditioning
            
        Returns:
            Output features
        """
        out = self.siren_layer(x, time_cond)
        return self.dropout(out)


class SIRENResidualBlock(nn.Module):
    """SIREN residual block with optional FiLM conditioning."""
    
    def __init__(
        self,
        dim: int,
        time_dim: int,
        omega_0: float,
        use_film: bool,
        dropout_rate: float = 0.0
    ) -> None:
        """Initialize SIREN residual block.
        
        Args:
            dim: Feature dimension
            time_dim: Time conditioning dimension
            omega_0: Angular frequency factor
            use_film: Whether to use FiLM conditioning
            dropout_rate: Dropout rate
        """
        super().__init__()
        self.use_film = use_film
        
        self.siren1 = SIRENLayer(
            input_dim=dim,
            output_dim=dim,
            omega_0=omega_0,
            is_first=False,
            time_dim=time_dim,
            use_film=use_film
        )
        
        self.siren2 = SIRENLayer(
            input_dim=dim,
            output_dim=dim,
            omega_0=omega_0,
            is_first=False,
            time_dim=0,  # Only apply FiLM to first layer in residual block
            use_film=False
        )
        
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, time_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through SIREN residual block.
        
        Args:
            x: Input features
            time_cond: Optional time conditioning
            
        Returns:
            Output features with residual connection
        """
        identity = x
        
        out = self.siren1(x, time_cond)
        out = self.dropout(out)
        
        out = self.siren2(out)
        out = self.dropout(out)
        
        return identity + out


class ODEFunc(nn.Module):
    """ODE dynamics function f(z, t) using SIREN blocks."""
    
    def __init__(
        self,
        dim: int,
        num_layers: int,
        omega_0: float,
        omega_0_hidden: float,
        dropout_rate: float,
        block_type: Literal["mlp", "residual"],
        fusion_mode: Literal["cat", "film", "film_time_embed"],
        time_embed_dim: int
    ) -> None:
        """Initialize SIREN-based ODE function.
        
        Args:
            dim: Feature dimension
            num_layers: Number of SIREN layers
            omega_0: First layer frequency factor
            omega_0_hidden: Hidden layer frequency factor
            dropout_rate: Dropout rate
            block_type: Type of block ("mlp" or "residual")
            fusion_mode: How to incorporate time ("cat", "film", "film_time_embed")
            time_embed_dim: Dimension of time embeddings
        """
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

        # Build SIREN layers
        if block_type == "residual":
            Block = SIRENResidualBlock
        else:  # "mlp"
            Block = SIRENBlock

        self.layers = nn.ModuleList([
            Block(
                dim=block_dim,
                time_dim=time_cond_dim,
                omega_0=omega_0_hidden,  # Use hidden omega_0 for all layers
                use_film=use_film_in_block,
                dropout_rate=dropout_rate
            ) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """Forward pass through SIREN ODE function.
        
        Args:
            x: State tensor of shape (B, D)
            t: Time scalar
            
        Returns:
            Time derivative dz/dt of shape (B, D)
        """
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


class OCSIREN(nn.Module):
    """Optimal Control-regularized SIREN."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        omega_0: float = 30.0,
        omega_0_hidden: float = 30.0,
        dropout_rate: float = 0.0,
        block_type: Literal["mlp", "residual"] = "residual",
        fusion_mode: Literal["cat", "film", "film_time_embed"] = "film_time_embed",
        time_embed_dim: int = 64,
        num_steps: int = 10,
        total_time: float = 1.0,
        ot_lambda: float = 1.0,
        final_activation: Optional[str] = None
    ) -> None:
        """Initialize the OC-SIREN model.
        
        Args:
            input_dim: Number of input dimensions
            hidden_dim: Width of hidden layers
            output_dim: Number of output dimensions
            num_layers: Number of hidden layers in ODE function
            omega_0: First layer frequency factor
            omega_0_hidden: Hidden layer frequency factor
            dropout_rate: Dropout rate
            block_type: Type of block ("mlp" or "residual")
            fusion_mode: How to incorporate time conditioning
            time_embed_dim: Dimension of time embeddings
            num_steps: Number of discretization steps for the ODE
            total_time: Total integration time T for the ODE
            ot_lambda: Weight for the optimal transport regularization
            final_activation: Optional activation for the output layer
            
        Raises:
            ValueError: If an unsupported activation name is provided
        """
        super().__init__()
        
        # Validate final activation
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
        
        if final_activation and final_activation not in VALID_ACTIVATIONS:
            raise ValueError(
                f"Unsupported final activation: {final_activation}. "
                f"Choose from {list(VALID_ACTIVATIONS.keys())}"
            )
            
        self.total_time = total_time
        self.num_steps = num_steps
        self.ot_lambda = ot_lambda
        
        # Initial embedding: z(0) = SIREN_first_layer(x)
        # Use SIREN's first layer initialization for input embedding
        self.input_embedding = SIRENLayer(
            input_dim=input_dim,
            output_dim=hidden_dim,
            omega_0=omega_0,
            is_first=True,  # Use first-layer initialization
            use_film=False
        )

        # ODE function with SIREN dynamics
        self.ode_func = ODEFunc(
            dim=hidden_dim,
            num_layers=num_layers,
            omega_0=omega_0,
            omega_0_hidden=omega_0_hidden,
            dropout_rate=dropout_rate,
            block_type=block_type,
            fusion_mode=fusion_mode,
            time_embed_dim=time_embed_dim
        )
                 
        # Output projection matching SIREN's final layer initialization
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        with torch.no_grad():
            limit = math.sqrt(6 / hidden_dim) / omega_0_hidden
            self.output_proj.weight.uniform_(-limit, limit)
            
        # Apply final activation if specified
        if final_activation:
            self.final_activation = VALID_ACTIVATIONS[final_activation]
        else:
            self.final_activation = nn.Identity()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the OC-SIREN.
        
        Args:
            x: Input coordinates of shape (batch_size, input_dim)
            
        Returns:
            Tuple of (network_output, ot_regularization_term)
            - network_output: shape (batch_size, output_dim)  
            - ot_regularization_term: scalar tensor
        """
        # Set initial state z(0) = SIREN_embedding(x)
        z = self.input_embedding(x)
        
        # Solve ODE using Euler's method
        ot_accum = 0.0
        dt = self.total_time / self.num_steps
        for i in range(self.num_steps):
            t = i * dt
            v = self.ode_func(z, t)
            # Accumulate optimal transport regularization
            ot_accum = ot_accum + v.pow(2).mean()
            z = z + dt * v

        ot_reg = 0.5 * self.ot_lambda * dt * ot_accum
        
        # Output projection from final state z(T)
        output = self.output_proj(z)
        output = self.final_activation(output)
        
        return output, ot_reg
    
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
    """Run tests for the OC-SIREN network."""
    # Network parameters matching SIREN defaults
    input_dim = 2
    hidden_dim = 256
    output_dim = 1
    num_layers = 4
    omega_0 = 30.0
    omega_0_hidden = 30.0
    dropout_rate = 0.0
    num_steps = 12
    final_activation = None
    
    # Base ODE config for sweepable hyperparameters
    base_ode_config = {
        "num_layers": 4,
        "dropout_rate": 0.0,
        "block_type": "mlp",
        "fusion_mode": "cat",
        "time_embed_dim": 64
    }

    configs = {
        "Default (cat, residual)": base_ode_config,
        "FiLM with plain time": {**base_ode_config, "fusion_mode": "film"},
        "MLP Blocks": {**base_ode_config, "block_type": "mlp"},
        "Concat Fallback": {**base_ode_config, "fusion_mode": "cat"}
    }

    for name, ode_config in configs.items():
        print(f"\n--- Testing: {name} ---")
        
        model = OCSIREN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=ode_config["num_layers"],
            omega_0=omega_0,
            omega_0_hidden=omega_0_hidden,
            dropout_rate=ode_config["dropout_rate"],
            block_type=ode_config["block_type"],
            fusion_mode=ode_config["fusion_mode"],
            time_embed_dim=ode_config["time_embed_dim"],
            num_steps=num_steps,
            final_activation=final_activation
        )

        batch_size = 16
        x = torch.rand(batch_size, input_dim)
        y, ot_reg = model(x)
        
        trainable_params, total_params = model.get_param_count()
        
        print(f"Model Architecture:")
        print(f"  Input dim: {input_dim}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Output dim: {output_dim}")
        print(f"  Num steps (M): {num_steps}")
        print(f"  Omega_0: {omega_0}")
        print(f"  Omega_0_hidden: {omega_0_hidden}")
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
