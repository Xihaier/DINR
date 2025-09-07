import torch

from torch.func import jacrev, vmap


def no_sampling(x, y, model=None, sample_ratio=None):
    """
    Return the entire dataset without any sampling.
    """
    return x, y

def random_sampling(x, y, model=None, sample_ratio=None, return_indices=False):
    """
    Perform random sampling by selecting a random subset of the dataset.

    Args:
        x (torch.Tensor): Input coordinates of shape (N, d), where N is the number of data points and d is the input dimensionality.
        y (torch.Tensor): Ground truth values of shape (N,).
        model (torch.nn.Module): Not used for random sampling but kept for consistency with other sampling functions.
        sample_ratio (float): Ratio of the dataset to sample.
        return_indices (bool): Whether to return sampled indices.

    Returns:
        torch.Tensor, torch.Tensor:
            - Subsampled x of shape (num_samples, d).
            - Subsampled y of shape (num_samples,).
            - Indices of sampled points (if return_indices is True).
    """
    N, _ = x.shape
    num_samples = int(sample_ratio * N)

    random_indices = torch.randperm(N)[:num_samples]

    if return_indices:
        return x[random_indices], y[random_indices], random_indices
    return x[random_indices], y[random_indices]

def error_based_sampling(x, y, model, sample_ratio, return_indices=False):
    """
    Select data points with the highest prediction errors.

    Args:
        x (torch.Tensor): Input coordinates of shape (N, d), where N is the number of data points, and d is the input dimensionality.
        y (torch.Tensor): Ground truth values of shape (N,).
        model (torch.nn.Module): The trained model for prediction.
        sample_ratio (float): Ratio of the dataset to sample.
        return_indices (bool): Whether to return sampled indices.

    Returns:
        torch.Tensor, torch.Tensor:
            - Subsampled x of shape (num_samples, d), corresponding to the highest error points.
            - Subsampled y of shape (num_samples,), corresponding to the ground truth values of the selected points.
            - Indices of sampled points (if return_indices is True).
    """
    N, _ = x.shape
    num_samples = int(sample_ratio*N)

    training_mode = model.training
    model.eval()
    with torch.no_grad():
        predictions = model(x)
        errors = (predictions - y).abs().squeeze() 

    if training_mode:
        model.train()

    error_indices = torch.argsort(errors, descending=True)[:num_samples]
    subsampled_x = x[error_indices]
    subsampled_y = y[error_indices]

    if return_indices:
        return subsampled_x, subsampled_y, error_indices
    return subsampled_x, subsampled_y

def error_based_sampling_DINER(x, y, model, sample_ratio, return_indices=False):
    """
    Perform error-based sampling to select the top error-prone data points along each dimension.

    Args:
        x (torch.Tensor): Input coordinates of shape (N, d), where N is the number of points per dimension, and d is the dimensionality.
        y (torch.Tensor): Ground truth values of shape (N, N, ..., N) (d times).
        model (torch.nn.Module): The trained model for prediction.
        sample_ratio (float): Ratio of the dataset to sample.
        return_indices (bool): Whether to return sampled indices.

    Returns:
        torch.Tensor, torch.Tensor: 
            - Subsampled x of shape (num_samples, d), where each column corresponds to the most error-prone indices for that dimension.
            - Subsampled y of shape (num_samples, num_samples, ..., num_samples) (d times).
            - Indices of sampled points (if return_indices is True).
    """
    device = x.device
    N, d = x.shape
    num_samples = int(sample_ratio*N)

    training_mode = model.training
    model.eval()
    with torch.no_grad():
        predictions = model(x)
    if training_mode:
        model.train()

    errors = (predictions - y).abs()

    selected_indices = []
    subsampled_x = torch.zeros((num_samples, d), dtype=x.dtype).to(device)

    for dim in range(d):
        error_projection = errors.sum(dim=tuple(i for i in range(d) if i != dim))
        top_indices = torch.argsort(error_projection, descending=True)[:num_samples]
        selected_indices.append(top_indices)
        subsampled_x[:, dim] = x[top_indices, dim]

    meshgrid_indices = torch.meshgrid(*selected_indices, indexing="ij")
    subsampled_y = y[tuple(meshgrid_indices)]

    return subsampled_x, subsampled_y

def gradient_based_sampling(x, y, model, sample_ratio, mode="forward", return_indices=False):
    """
    Sample data points based on the gradient magnitude of the model's predictions.
    Supports both forward and reverse-mode differentiation.

    Args:
        x (torch.Tensor): Input coordinates of shape (N, d), where N is the number of data points, and d is the input dimensionality.
        y (torch.Tensor): Ground truth values of shape (N,).
        model (torch.nn.Module): The trained model for prediction.
        sample_ratio (float): Ratio of the dataset to sample.
        mode (str): Differentiation mode, either "forward" or "reverse".
        return_indices (bool): Whether to return sampled indices.

    Returns:
        torch.Tensor, torch.Tensor:
            - Subsampled x of shape (num_samples, d), corresponding to the points with the largest gradient magnitude.
            - Subsampled y of shape (num_samples,), corresponding to the ground truth values of the selected points.
            - Indices of sampled points (if return_indices is True).
    """
    device = x.device
    N, _ = x.shape
    num_samples = int(sample_ratio*N)

    training_mode = model.training
    model.eval()

    if mode == "forward":
        def model_output(x_point):
            return model(x_point.unsqueeze(0)).squeeze()

        jacobian_func = vmap(jacrev(model_output))  
        gradients = jacobian_func(x)               

    elif mode == "reverse":
        gradients = torch.autograd.functional.jacobian(lambda x_point: model(x_point).squeeze(), x)

    else:
        raise ValueError(f"Invalid mode '{mode}'. Supported modes are 'forward' and 'reverse'.")

    if training_mode:
        model.train()

    gradient_magnitudes = torch.norm(gradients, dim=-1).to(device)
    gradient_indices = torch.argsort(gradient_magnitudes, descending=True)[:num_samples]
    subsampled_x = x[gradient_indices]
    subsampled_y = y[gradient_indices] 
    if return_indices:
        return subsampled_x, subsampled_y, gradient_indices
    return subsampled_x, subsampled_y

def gradient_based_sampling_DINER(x, y, model, sample_ratio, mode="forward"):
    """
    Perform gradient-based sampling to select the most important data points per axis.

    Args:
        x (torch.Tensor): Input coordinates of shape (N, d), where N is the number of points per dimension, and d is the dimensionality.
        y (torch.Tensor): Ground truth values of shape (N, N, ..., N) (d times).
        model (torch.nn.Module): The trained model for prediction.
        num_samples (int): Number of points to select per axis based on gradient magnitude.
        mode (str): Differentiation mode, either "forward" or "reverse".
                    - "forward": Uses forward-mode differentiation with `vmap`.
                    - "reverse": Uses reverse-mode differentiation with `autograd`.

    Returns:
        torch.Tensor, torch.Tensor:
            - Subsampled x of shape (num_samples, d), where each column corresponds to the selected indices for that dimension.
            - Subsampled y of shape (num_samples, num_samples, ..., num_samples) (d times).
    """
    N, d = x.shape
    num_samples = int(sample_ratio*N)

    training_mode = model.training
    model.eval()

    if mode == "forward":
        def model_output(x_point):
            return model(x_point.unsqueeze(0)).squeeze()
        jacobian_func = vmap(jacrev(model_output))
        gradients = jacobian_func(x)
    elif mode == "reverse":
        gradients = torch.autograd.functional.jacobian(lambda x_point: model(x_point).squeeze(), x)
    else:
        raise ValueError(f"Invalid mode '{mode}'. Supported modes are 'forward' and 'reverse'.")

    if training_mode:
        model.train()

    selected_indices = []
    for dim in range(d):
        top_indices = torch.argsort(gradients[:, dim], descending=True)[:num_samples]
        selected_indices.append(top_indices)

    subsampled_x = torch.zeros((num_samples, d), dtype=x.dtype, device=x.device)
    for dim in range(d):
        subsampled_x[:, dim] = x[selected_indices[dim], dim]

    meshgrid_indices = torch.meshgrid(*selected_indices, indexing="ij")
    subsampled_y = y[tuple(meshgrid_indices)]

    return subsampled_x, subsampled_y

def hybrid_sampling(x, y, model, sample_ratio=0.1, error_ratio=0.5, return_indices=False):
    """
    Perform hybrid sampling by combining error-based and gradient-based sampling.

    Args:
        x (torch.Tensor): Input coordinates of shape (N, d), where N is the number of points and d is the input dimensionality.
        y (torch.Tensor): Ground truth values of shape (N,).
        model (torch.nn.Module): The trained model for prediction.
        sample_ratio (float): Ratio of the dataset to sample.
        error_ratio (float): Proportion of samples to select using error-based sampling.
        return_indices (bool): Whether to return sampled indices.

    Returns:
        torch.Tensor, torch.Tensor:
            - Subsampled x of shape (num_samples, d).
            - Subsampled y of shape (num_samples,).
            - Indices of sampled points (if return_indices is True).
    """
    device = x.device
    N, _ = x.shape
    num_samples = int(sample_ratio*N)

    training_mode = model.training
    model.eval()

    with torch.no_grad():
        predictions = model(x)
        errors = (predictions - y).abs().squeeze()
    
    num_error_samples = int(num_samples * error_ratio)
    error_indices = torch.argsort(errors, descending=True)[:num_error_samples]

    num_gradient_samples = num_samples - num_error_samples
    subset_size = int((1 - error_ratio) * x.size(0))  
    subset_indices = torch.randperm(x.size(0), device=device)[:subset_size] 
    subset_x = x[subset_indices]

    def model_output_fn(input_x):
        return model(input_x.unsqueeze(0)).squeeze()

    jacobian_fn = vmap(jacrev(model_output_fn))  
    jacobians = jacobian_fn(subset_x)  
    gradient_magnitudes = torch.norm(jacobians, dim=-1) 

    if training_mode:
        model.train()

    gradient_top_indices = torch.argsort(gradient_magnitudes, descending=True)[:num_gradient_samples]
    gradient_indices = subset_indices[gradient_top_indices]

    selected_indices = torch.unique(torch.cat([error_indices.squeeze(), gradient_indices]))
    subsampled_x = x[selected_indices]
    subsampled_y = y[selected_indices]

    if return_indices:
        return subsampled_x, subsampled_y, selected_indices
    return subsampled_x, subsampled_y

def hybrid_sampling_DINER(x, y, model, sample_ratio=0.1, error_ratio=0.7):
    """
    Perform hybrid sampling by combining error-based and gradient-based sampling.

    Args:
        x (torch.Tensor): Input coordinates of shape (N, d).
        y (torch.Tensor): Ground truth values of shape (N, N, ..., N) (d times).
        model (torch.nn.Module): Trained model for prediction.
        error_ratio (float): Fraction of samples selected using error-based sampling (0 to 1).
        num_samples (int): Total number of samples to select.

    Returns:
        torch.Tensor, torch.Tensor:
            - Subsampled x of shape (num_samples, d).
            - Subsampled y of shape (num_samples, num_samples, ..., num_samples) (d times).
    """
    device = x.device
    N, d = x.shape
    num_samples = int(sample_ratio*N)

    training_mode = model.training
    model.eval()

    with torch.no_grad():
        predictions = model(x)
        errors = (predictions - y).abs()  # Shape: (N, N, ..., N)

    num_error_samples = int(num_samples * error_ratio)
    error_selected_indices = []

    for dim in range(d):
        error_projection = errors.sum(dim=tuple(i for i in range(d) if i != dim))  # Shape: (N,)
        top_error_indices = torch.argsort(error_projection, descending=True)[:num_error_samples]
        error_selected_indices.append(top_error_indices)

    num_gradient_samples = num_samples - num_error_samples

    def model_output_fn(input_x):
        return model(input_x.unsqueeze(0)).squeeze()

    jacobian_fn = vmap(jacrev(model_output_fn))
    gradients = jacobian_fn(x)  # Shape: (N, d)
    
    if training_mode:
        model.train()
    
    gradient_selected_indices = []
    for dim in range(d):
        top_gradient_indices = torch.argsort(gradients[:, dim], descending=True)
        mask = ~torch.isin(top_gradient_indices, error_selected_indices[dim])
        top_gradient_indices = top_gradient_indices[mask][:num_gradient_samples]
        combined_indices = torch.cat((error_selected_indices[dim], top_gradient_indices), dim=0)
        gradient_selected_indices.append(combined_indices)

    subsampled_x = torch.zeros((num_samples, d), dtype=x.dtype, device=device)
    for dim in range(d):
        subsampled_x[:, dim] = x[gradient_selected_indices[dim], dim]

    meshgrid_indices = torch.meshgrid(*gradient_selected_indices, indexing="ij")
    subsampled_y = y[tuple(meshgrid_indices)]  # Shape: (num_samples, ..., num_samples)

    return subsampled_x, subsampled_y

def get_sampling(model_name, method):
    """
    Retrieve the appropriate sampling function based on the model name and method.

    Args:
        model_name (str): The name of the model (e.g., "FFNet", "DINR").
        method (str): The sampling method (e.g., "no_sampling", "error_based", "gradient_based", "hybrid_sampling").

    Returns:
        function: The selected sampling function.
    """
    baselines = {"FFNet", "Finer", "NeRF", "SIREN", "WIRE"}

    if method == "no_sampling":
        return no_sampling

    if model_name in baselines:
        if method == "random_sampling":
            return random_sampling
        elif method == "error_based":
            return error_based_sampling
        elif method == "gradient_based":
            return gradient_based_sampling
        elif method == "hybrid_sampling":
            return hybrid_sampling
        else:
            raise ValueError(f"Invalid sampling method '{method}' for baseline models.")
    elif model_name == "DINER":
        if method == "error_based":
            return error_based_sampling_DINER
        elif method == "gradient_based":
            return gradient_based_sampling_DINER
        elif method == "hybrid_sampling":
            return hybrid_sampling_DINER
        else:
            raise ValueError(f"Invalid sampling method '{method}' for DINER.")
    else:
        raise ValueError(f"Invalid model name '{model_name}'. Supported models are {baselines} and 'DINER'.")
    
def compute_sample_ratio(epoch, total_epochs, lower, upper, cycles=3, waveform="linear", decay_factor=1.0, mode="cyclical"):
    """
    Compute the sample ratio based on the current mode (cyclical or constant).

    Args:
        epoch (int): Current epoch.
        total_epochs (int): Total number of epochs.
        lower (float): Minimum sample_ratio.
        upper (float): Maximum sample_ratio.
        cycles (int): Number of cycles within total_epochs.
        waveform (str): Shape of the cycle ('linear', 'sinusoidal', 'cosine', or 'exponential').
        decay_factor (float): Decay factor for amplitude reduction across cycles.
        mode (str): Mode of sampling ratio calculation ('cyclical' or 'constant').

    Returns:
        float: Sample ratio for the current epoch.
    """
    if mode == "cyclical":
        total_progress = epoch / total_epochs
        current_amplitude = (upper - lower) * (decay_factor ** (total_progress * cycles))
        cycle_progress = (epoch % (total_epochs / cycles)) / (total_epochs / cycles)

        if waveform == "linear":
            if cycle_progress <= 0.5:
                return lower + 2 * current_amplitude * cycle_progress
            else:
                return upper - 2 * current_amplitude * (cycle_progress - 0.5)
        elif waveform == "sinusoidal":
            return lower + current_amplitude * (1 - np.cos(2 * np.pi * cycle_progress)) / 2
        elif waveform == "cosine":
            return lower + current_amplitude * (1 + np.cos(2 * np.pi * total_progress * cycles)) / 2
        elif waveform == "exponential":
            if cycle_progress <= 0.5:
                return lower + current_amplitude * (2 ** (4 * cycle_progress) - 1) / 15.0
            else:
                return upper - current_amplitude * (2 ** (4 * (1 - cycle_progress)) - 1) / 15.0
        else:
            raise ValueError(f"Invalid waveform '{waveform}'. Choose from 'linear', 'sinusoidal', 'cosine', or 'exponential'.")
    
    elif mode == "constant":
        return (lower + upper) / 2
    
    else:
        raise ValueError(f"Invalid mode '{mode}'. Choose 'cyclical' or 'constant'.")













# def gradient_based_sampling(x, y, model, num_samples, mode="forward"):
#     """
#     Sample data points based on gradient magnitude.
#     Supports forward and reverse mode differentiation.
#     """
#     device = x.device
#     if mode == "forward":
#         def model_output(x):
#             return model(x.unsqueeze(0)).squeeze()
#         # Forward-mode with vmap
#         jacobian_func = vmap(jacrev(model_output))  
#         # Jacobian for each input point
#         gradients = jacobian_func(x)                
#         gradient_magnitudes = torch.norm(gradients, dim=-1).to(device)
#         gradient_indices = torch.argsort(gradient_magnitudes, descending=True)[:num_samples]
#     elif mode =="reverse":
#         gradients = torch.autograd.functional.jacobian(lambda x: model(x).squeeze(), x)
#         gradient_magnitudes = torch.norm(gradients, dim=1).to(device)
#         gradient_indices = torch.argsort(gradient_magnitudes, descending=True)[:num_samples]        
#     return x[gradient_indices], y[gradient_indices]

# def hybrid_sampling(x, y, model, error_ratio=0.7, num_samples=1000):
#     """
#     Combine error-based and gradient-based sampling.
#     """
#     device = x.device

#     with torch.no_grad():
#         predictions = model(x)
#         errors = (predictions - y).abs().squeeze()
#     error_indices = torch.argsort(errors, descending=True)[:int(num_samples * error_ratio)]

#     # Define the subset of data for gradient-based sampling
#     gradient_subset_size = int((1-error_ratio) * x.size(0))
#     subset_indices = torch.randperm(x.size(0), device=device)[:gradient_subset_size]
#     subset_x = x[subset_indices]

#     def model_output_fn(input_x):
#         return model(input_x.unsqueeze(0)).squeeze()

#     # Apply jacrev across all inputs in the batch
#     jacobian_fn = vmap(jacrev(model_output_fn))  
#     jacobians = jacobian_fn(subset_x)

#     gradient_magnitudes = torch.norm(jacobians, dim=1).to(device)

#     gradient_indices = subset_indices[torch.argsort(gradient_magnitudes, descending=True)[:int(num_samples * (1 - error_ratio))]]
#     gradient_indices = gradient_indices.view(-1)

#     error_indices = error_indices.view(-1)

#     selected_indices = torch.unique(torch.cat([error_indices, gradient_indices]))
#     return x[selected_indices], y[selected_indices]

# import torch.nn as nn
# import numpy as np


# class DimensionalINR(nn.Module):
#     def __init__(self, d, hidden_size=64, num_layers=3):
#         """
#         Initialize the Dimensional INR model.
        
#         Args:
#             d (int): Number of dimensions (input features).
#             hidden_size (int): Number of hidden neurons in each MLP.
#             num_layers (int): Number of layers in each MLP.
#         """
#         super(DimensionalINR, self).__init__()
        
#         # Create an MLP for each dimension
#         self.mlp_list = nn.ModuleList([
#             self._build_mlp(hidden_size, num_layers) for _ in range(d)
#         ])
    
#     def _build_mlp(self, hidden_size, num_layers):
#         """
#         Build an MLP with specified hidden size and number of layers.
#         """
#         layers = [nn.Linear(1, hidden_size), nn.ReLU()]  # Input size is 1 (coordinate value)
#         for _ in range(num_layers - 2):  # Add hidden layers
#             layers.append(nn.Linear(hidden_size, hidden_size))
#             layers.append(nn.ReLU())
#         layers.append(nn.Linear(hidden_size, hidden_size))  # Output layer
#         return nn.Sequential(*layers)
    
#     def forward(self, x):
#         """
#         Forward pass for the model.

#         Args:
#             x (torch.Tensor): Input tensor of shape (N, d), where N is the
#                             discretization per axis, and d is the number of dimensions.

#         Returns:
#             torch.Tensor: Output tensor of shape (N, N, ..., N) (d times).
#         """
#         # Pass each dimension through its corresponding MLP
#         sub_outputs = []
#         for i in range(x.size(1)):  # Iterate over dimensions
#             sub_output = self.mlp_list[i](x[:, i:i+1])  # Shape: (N, hidden_size)
#             # Reduce the output to a single dimension using mean
#             sub_output_reduced = sub_output.mean(dim=-1)  # Shape: (N,)
#             sub_outputs.append(sub_output_reduced)

#         # Generate the einsum equation dynamically
#         num_dimensions = len(sub_outputs)  # d dimensions
#         subscripts = ''.join(chr(97 + i) for i in range(num_dimensions))  # 'a', 'b', ..., 'z'
#         einsum_eq = f"{','.join(subscripts)}->{subscripts}"  # Example: 'a,b->ab'

#         # Combine outputs using einsum for fusion
#         fused_output = torch.einsum(einsum_eq, *sub_outputs)

#         return fused_output


# class FourierFeatureMapping(nn.Module):
#     def __init__(self, input_dim, mapping_size, sigma=1.0):
#         """
#         Fourier feature mapping module.

#         Parameters:
#         - input_dim (int): Dimensionality of the input (e.g., 2 for 2D coordinates).
#         - mapping_size (int): Size of the Fourier feature mapping output.
#         - sigma (float): Standard deviation of the Gaussian distribution for random weights.
#         """
#         super(FourierFeatureMapping, self).__init__()
#         self.input_dim = input_dim
#         self.mapping_size = mapping_size
#         self.sigma = sigma
        
#         self.B = nn.Parameter(
#             torch.randn(input_dim, mapping_size // 2) * sigma, requires_grad=False
#         )

#     def forward(self, x):
#         """
#         Forward pass for Fourier feature mapping.

#         Parameters:
#         - x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

#         Returns:
#         - torch.Tensor: Fourier-mapped tensor of shape (batch_size, mapping_size).
#         """
#         x_proj = 2 * np.pi * x @ self.B 
#         return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

# class FourierFeatureNetwork(nn.Module):
#     def __init__(self, input_dim, mapping_size, hidden_dim, num_layers, output_dim, sigma=1.0):
#         """
#         Fourier feature network with an MLP.

#         Parameters:
#         - input_dim (int): Dimensionality of the input (e.g., 2 for 2D coordinates).
#         - mapping_size (int): Size of the Fourier feature mapping output.
#         - hidden_dim (int): Number of neurons in each hidden layer.
#         - num_layers (int): Number of layers in the MLP.
#         - output_dim (int): Dimensionality of the output (e.g., scalar or vector output).
#         - sigma (float): Standard deviation of the Gaussian distribution for Fourier features.
#         """
#         super(FourierFeatureNetwork, self).__init__()
        
#         self.fourier_features = FourierFeatureMapping(input_dim, mapping_size, sigma)
#         mlp_input_dim = mapping_size
        
#         layers = []
#         layers.append(nn.Linear(mlp_input_dim, hidden_dim))
#         layers.append(nn.ReLU())
#         for _ in range(num_layers - 1):
#             layers.append(nn.Linear(hidden_dim, hidden_dim))
#             layers.append(nn.ReLU())
#         layers.append(nn.Linear(hidden_dim, output_dim))
        
#         self.mlp = nn.Sequential(*layers)

#     def forward(self, x):
#         """
#         Forward pass for the Fourier feature network.

#         Parameters:
#         - x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

#         Returns:
#         - torch.Tensor: Output tensor of shape (batch_size, output_dim).
#         """
#         x = self.fourier_features(x)
#         return self.mlp(x)






# if __name__ == "__main__":
#     input_dim = 3       # Input dimension
#     mapping_size = 256  # Number of Fourier features
#     hidden_dim = 256    # Hidden layer size
#     num_layers = 5      # Number of layers
#     output_dim = 1      # Output dimension
#     sigma = 10.0        # Standard deviation for Fourier features

#     model = FourierFeatureNetwork(input_dim, mapping_size, hidden_dim, num_layers, output_dim, sigma)

#     batch_size = 1024
#     num_samples = 128
#     error_ratio = 0.7
#     example_input  = torch.rand(batch_size, input_dim)
#     example_output = torch.rand(batch_size, 1)

#     x_sample, y_sample = error_based_sampling_baselines(example_input, example_output, model, num_samples)

#     print("error_based_sampling: x_sample.shape ", x_sample.shape)
#     print("error_based_sampling: y_sample.shape ", y_sample.shape)

#     x_sample, y_sample = gradient_based_sampling_baselines(example_input, example_output, model, num_samples, mode="forward")

#     print("gradient_based_sampling: x_sample.shape ", x_sample.shape)
#     print("gradient_based_sampling: y_sample.shape ", y_sample.shape)

#     x_sample, y_sample = hybrid_sampling_baselines(example_input, example_output, model, error_ratio, num_samples)

#     print("hybrid_sampling: x_sample.shape ", x_sample.shape)
#     print("hybrid_sampling: y_sample.shape ", y_sample.shape)

#     # Define model parameters
#     d = 3  # Number of dimensions
#     N = 64  # Discretization per axis
#     hidden_size = 32  # Hidden size for each MLP
#     del model
#     model = DimensionalINR(d, hidden_size)
#     x = torch.rand(N, d)
#     output = model(x)
#     num_samples = 10
#     example_input = x
#     example_output = output
#     error_ratio = 0.7

#     x_sample, y_sample = error_based_sampling_DINR(example_input, example_output, model, num_samples)

#     print("error_based_sampling: x_sample.shape ", x_sample.shape)
#     print("error_based_sampling: y_sample.shape ", y_sample.shape)

#     x_sample, y_sample = gradient_based_sampling_DINR(example_input, example_output, model, num_samples, mode="forward")

#     print("gradient_based_sampling: x_sample.shape ", x_sample.shape)
#     print("gradient_based_sampling: y_sample.shape ", y_sample.shape)

#     x_sample, y_sample = hybrid_sampling_DINR(example_input, example_output, model, error_ratio, num_samples)
#     print("error_based_sampling: x_sample.shape ", x_sample.shape)
#     print("error_based_sampling: y_sample.shape ", y_sample.shape)







