import torch
import numpy as np
from typing import Any, Dict, Optional, List, Tuple, Set, Union, Callable
from lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.nn import functional as F
from omegaconf import DictConfig, ListConfig
import warnings


class NormalizationError(ValueError):
    """Raised when data normalization fails."""
    pass


def normalize_data(data: torch.Tensor, method: str = "min-max") -> torch.Tensor:
    """Normalize data using the specified method.

    Args:
        data: Input tensor to normalize
        method: Normalization method ('min-max' or 'z-score')

    Returns:
        Normalized data tensor
        
    Raises:
        NormalizationError: If normalization fails
    """
    try:
        if method == "min-max":
            data_min, data_max = data.min(), data.max()
            if data_max - data_min < 1e-7:
                raise NormalizationError(
                    "Data has zero or near-zero range for min-max normalization"
                )
            return (data - data_min) / (data_max - data_min)

        elif method == "z-score":
            data_mean, data_std = data.mean(), data.std()
            if data_std < 1e-7:
                raise NormalizationError(
                    "Data has near-zero standard deviation for z-score normalization"
                )
            return (data - data_mean) / data_std

        else:
            raise ValueError(
                f"Invalid normalization method '{method}'. "
                "Supported options are 'min-max' and 'z-score'"
            )
            
    except Exception as e:
        if isinstance(e, (NormalizationError, ValueError)):
            raise
        raise NormalizationError(f"Normalization failed: {str(e)}")


class DataModule(LightningDataModule):
    """
    DataModule for Implicit Neural Representation.
    """
    def __init__(
        self,
        data_dir: str,
        in_features: int,
        normalization: str = "min-max",
        temporal: bool = False,
        data_shape: Optional[List[int]] = None,
        batch_size: Union[int, List[int]] = 8192,
        shuffle: Union[bool, List[bool]] = True,
        num_workers: Union[int, List[int]] = 4,
        pin_memory: Union[bool, List[bool]] = True,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        
        self.full_volume = None
        self.coord_vectors = None
        self.train_dataset, self.test_dataset = None, None

        # Store path for data loading
        self.data_dir = data_dir

    def setup(self, stage: Optional[str] = None) -> None:
        """Load and process data."""
        if self.train_dataset is not None:
            return

        # Load and normalize the full data volume
        raw_data = torch.from_numpy(np.load(self.data_dir)).float()
        if self.hparams.data_shape:
            raw_data = raw_data.reshape(list(self.hparams.data_shape))
        
        self.full_volume = normalize_data(raw_data, self.hparams.normalization)

        # Create full-resolution coordinate vectors
        dims = self.full_volume.shape
        ranges = [(-1.0, 1.0)] * self.hparams.in_features
        if self.hparams.temporal:
            ranges[-1] = (0.0, 1.0)
        self.coord_vectors = [torch.linspace(r[0], r[1], d) for r, d in zip(ranges, dims)]

        coords_flat = torch.stack(torch.meshgrid(*self.coord_vectors, indexing="ij"), dim=-1).view(-1, self.hparams.in_features)
        targets_flat = self.full_volume.flatten().unsqueeze(-1)
        full_pointwise_dataset = torch.utils.data.TensorDataset(coords_flat, targets_flat)
        self.train_dataset = full_pointwise_dataset
        self.test_dataset = full_pointwise_dataset

    def _create_dataloader(self, dataset: Dataset, dataloader_idx: int) -> DataLoader:
        """Helper function to create a DataLoader."""
        if dataloader_idx not in [0, 1]:
            raise IndexError(f"Invalid dataloader_idx: {dataloader_idx}. Must be 0 or 1.")
        
        # Determine if hyperparameters are specified per dataloader or globally
        is_list_batch_size = isinstance(self.hparams.batch_size, (list, ListConfig))
        is_list_shuffle = isinstance(self.hparams.shuffle, (list, ListConfig))
        is_list_num_workers = isinstance(self.hparams.num_workers, (list, ListConfig))
        is_list_pin_memory = isinstance(self.hparams.pin_memory, (list, ListConfig))

        # Get specific or global values for dataloader parameters
        batch_size = self.hparams.batch_size[dataloader_idx] if is_list_batch_size else self.hparams.batch_size
        shuffle = self.hparams.shuffle[dataloader_idx] if is_list_shuffle else (dataloader_idx == 0 and self.hparams.shuffle)
        num_workers = self.hparams.num_workers[dataloader_idx] if is_list_num_workers else self.hparams.num_workers
        pin_memory = self.hparams.pin_memory[dataloader_idx] if is_list_pin_memory else self.hparams.pin_memory

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0
        )

    def train_dataloader(self) -> DataLoader:
        """Create the training dataloader.
        """
        return self._create_dataloader(self.train_dataset, 0)

    def test_dataloader(self) -> DataLoader:
        """Create the test dataloader.
        """
        return self._create_dataloader(self.test_dataset, 1)

    def teardown(self, stage: Optional[str] = None) -> None:
        """Clean up after training or testing."""
        # Clear memory if needed
        pass

    def state_dict(self) -> Dict[str, Any]:
        """Return the state of the DataModule."""
        return {
            'data_dir': self.hparams.data_dir,
            'normalization': self.hparams.normalization,
            'temporal': self.hparams.temporal,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load the state of the DataModule."""
        # Update hyperparameters if needed
        for key, value in state_dict.items():
            if hasattr(self.hparams, key):
                setattr(self.hparams, key, value)