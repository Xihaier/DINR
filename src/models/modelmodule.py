from typing import Any, Dict, Tuple, Optional

import torch
from lightning import LightningModule
from torchmetrics import MeanMetric, MinMetric
import numpy as np
import warnings
import os

from src.utils.metrics import l2_relative_error
from src.utils.ntk import analyze_model_ntk


class INRTraining(LightningModule):
    """A Lightning Module for implicit neural representation training.
    """

    def __init__(
        self,
        net: torch.nn.Module = None,
        optimizer: Any = None,
        scheduler: Any = None,
        criterion: Any = None,
        compile: bool = False,
        # NTK analysis parameters
        ntk_analysis: bool = False,
        ntk_frequency: int = 10,
        ntk_top_k: int = 10,
        ntk_subset_size: Optional[int] = None,
        ntk_subset_stride: int = 1,
        ntk_normalize: str = "trace",
        **kwargs
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=['net', 'criterion'])
        
        self.model = net
        self.criterion = criterion

        # NTK analysis setup
        self.ntk_analysis = ntk_analysis
        self.ntk_frequency = ntk_frequency
        self.ntk_top_k = ntk_top_k
        self.ntk_subset_size = ntk_subset_size
        self.ntk_subset_stride = ntk_subset_stride
        self.ntk_normalize = ntk_normalize
        self.ntk_training_inputs = None

        # metrics 
        self.train_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.train_rel_error = MeanMetric()
        self.train_rel_error_best = MinMetric()
        
        # NTK metrics
        if self.ntk_analysis:
            self.ntk_effective_rank = MeanMetric()
            self.ntk_condition_number = MeanMetric()
            self.ntk_spectrum_decay = MeanMetric()
            for i in range(self.ntk_top_k):
                setattr(self, f'ntk_eigenvalue_{i+1}', MeanMetric())

        # For storing test outputs
        self.test_predictions = []
        self.test_ground_truth = []

    def _capture_training_inputs(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """Capture training inputs for NTK analysis."""
        if self.ntk_analysis:
            coords, _ = batch
            if self.ntk_training_inputs is None:
                self.ntk_training_inputs = coords.detach().cpu()
            else:
                # Accumulate all training inputs (subsetting handled in _get_ntk_inputs)
                new_inputs = coords.detach().cpu()
                self.ntk_training_inputs = torch.cat([self.ntk_training_inputs, new_inputs], dim=0)

    def _get_ntk_inputs(self) -> torch.Tensor:
        """Get inputs for NTK analysis from captured training data."""
        if self.ntk_training_inputs is None:
            raise ValueError("No training inputs captured for NTK analysis. Ensure training has started.")
        
        # Apply stride-based subsampling
        inputs = self.ntk_training_inputs[::self.ntk_subset_stride]
        
        # Apply size-based truncation if specified
        if self.ntk_subset_size is not None:
            inputs = inputs[:self.ntk_subset_size]
            
        return inputs

    def _setup_ntk_analysis(self) -> None:
        """Setup NTK analysis components."""
        try:
            # Store NTK results for analysis
            self.ntk_results_history = []
            
            print(f"✓ NTK analysis initialized: top-{self.ntk_top_k} eigenvalues, "
                  f"frequency every {self.ntk_frequency} epochs, "
                  f"using training data")
                  
        except Exception as e:
            warnings.warn(f"Failed to setup NTK analysis: {e}. Disabling NTK analysis.")
            self.ntk_analysis = False

    def _perform_ntk_analysis(self) -> Optional[Dict[str, float]]:
        """Perform NTK analysis on the current model state."""
        if not self.ntk_analysis:
            return None
            
        try:
            # Get inputs for NTK analysis
            inputs = self._get_ntk_inputs()
            device = next(self.model.parameters()).device
            inputs = inputs.to(device)
            
            # Perform NTK analysis
            result = analyze_model_ntk(
                self.model,
                inputs,
                normalize=self.ntk_normalize,
                top_k=self.ntk_top_k
            )
            
            # Extract key metrics
            metrics = {
                'effective_rank': float(result.effective_rank),
                'condition_number': float(result.condition_number),
                'spectrum_decay': float(result.spectrum_decay),
                'trace': float(result.trace),
            }
            
            # Add top-k eigenvalues
            for i, eigenval in enumerate(result.top_k_eigenvalues):
                metrics[f'eigenvalue_{i+1}'] = float(eigenval)
            
            # Store for history tracking
            metrics['epoch'] = self.current_epoch
            self.ntk_results_history.append(metrics)
            
            # Update metrics
            self.ntk_effective_rank(metrics['effective_rank'])
            self.ntk_condition_number(metrics['condition_number'])
            self.ntk_spectrum_decay(metrics['spectrum_decay'])
            
            # Update eigenvalue metrics
            for i in range(min(self.ntk_top_k, len(result.top_k_eigenvalues))):
                eigenval_metric = getattr(self, f'ntk_eigenvalue_{i+1}')
                eigenval_metric(float(result.top_k_eigenvalues[i]))
            
            return metrics
            
        except Exception as e:
            warnings.warn(f"NTK analysis failed at epoch {self.current_epoch}: {e}")
            return None

    def on_fit_start(self) -> None:
        if self.ntk_analysis:
            self._setup_ntk_analysis()

    def model_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step (forward pass + loss computation).
        
        Args:
            batch: A tuple of (coordinates, ground_truth).
            
        Returns:
            Tuple of (loss, predictions, ground_truth)
        """
        coords, gt = batch
        pred = self.model(coords)
        loss = self.criterion(pred, gt)
        
        return loss, pred, gt

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        # Capture training inputs for NTK analysis
        self._capture_training_inputs(batch)
        
        loss, pred, gt = self.model_step(batch)
        rel_error = l2_relative_error(pred.flatten(), gt.flatten())
        self.train_loss(loss)
        self.train_rel_error(rel_error)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/rel_error", self.train_rel_error, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self, unused: Optional = None) -> None:
        """Called at the end of the training epoch."""
        rel_error = self.train_rel_error.compute()
        self.train_rel_error_best.update(rel_error)
        self.log("train/rel_error_best", self.train_rel_error_best.compute(), prog_bar=True)

        if self.ntk_analysis and (self.current_epoch % self.ntk_frequency == 0):
            _ = self._perform_ntk_analysis()
            
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Test step."""
        loss, pred, gt = self.model_step(batch)
        rel_error = l2_relative_error(pred.flatten(), gt.flatten())
        self.test_loss(loss)
        self.log("test/rel_error", rel_error, on_step=False, on_epoch=True, prog_bar=True)
        self.test_predictions.append(pred.detach().cpu())
        self.test_ground_truth.append(gt.detach().cpu())

    def on_test_epoch_end(self) -> None:
        """Evaluate the model on the full data grid if supported."""
        preds = torch.cat(self.test_predictions).numpy()
        self.save_predictions(preds, filename="test_preds")
        
        if self.ntk_analysis:
            self.save_ntk_results()
            
    def setup(self, stage: str) -> None:
        """Called at the beginning of fit and test."""
        if self.hparams.compile and stage == "fit":
            self.model = torch.compile(self.model)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and learning rate schedulers.
        
        Returns:
            A dictionary containing the optimizer and scheduler configuration.
        """
        if hasattr(self.hparams, 'optimizer') and self.hparams.optimizer is not None:
            optimizer = self.hparams.optimizer(params=self.parameters())
        else:
            # Default optimizer if none is provided
            warnings.warn("No optimizer specified, using AdamW with default settings.")
            optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)

        if hasattr(self.hparams, 'scheduler') and self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)

            lr_scheduler_config = {"scheduler": scheduler}

            # For ReduceLROnPlateau, we need to specify the metric to monitor
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler_config["monitor"] = "train/loss"

            return {
                "optimizer": optimizer,
                "lr_scheduler": lr_scheduler_config,
            }
            
        return {"optimizer": optimizer}

    def save_predictions(self, predictions: np.ndarray, filename: str = "predictions.npy") -> None:
        """Save predictions to a file."""
        output_dir = self.trainer.log_dir
        np.save(f"{output_dir}/{filename}", predictions)
        
    def save_ntk_results(self, filename: str = "ntk_analysis.npy") -> None:
        """Save NTK analysis results to file."""
        if hasattr(self, 'ntk_analysis') and self.ntk_analysis and hasattr(self, 'ntk_results_history'):
            output_dir = self.trainer.log_dir
            if output_dir is not None:
                np.save(os.path.join(output_dir, filename), self.ntk_results_history)
                print(f"✓ NTK results saved to {output_dir}/{filename}")
            else:
                np.save(filename, self.ntk_results_history)
                print(f"✓ NTK results saved to {filename}")

    def get_model_info(self) -> Dict[str, Any]:
        """Return model information."""    
        info = {} 
        try:
            if hasattr(self.model, 'get_param_count'):
                trainable_params, total_params = self.model.get_param_count()
            else:
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                total_params = sum(p.numel() for p in self.model.parameters())
            info['trainable_params'] = trainable_params
            info['total_params'] = total_params
        except Exception as e:
            warnings.warn(f"Could not get parameter count: {e}")
            info['trainable_params'] = 'N/A'
            info['total_params'] = 'N/A'
            
        return info
    

class OCINRTraining(LightningModule):
    """A Lightning Module for optimal control-regularized implicit neural representation training.
    """

    def __init__(
        self,
        net: torch.nn.Module = None,
        optimizer: Any = None,
        scheduler: Any = None,
        criterion: Any = None,
        compile: bool = False,
        # NTK analysis parameters
        ntk_analysis: bool = True,
        ntk_frequency: int = 10,
        ntk_top_k: int = 10,
        ntk_subset_size: Optional[int] = None,
        ntk_subset_stride: int = 1,
        ntk_normalize: str = "trace",
        **kwargs
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=['net', 'criterion'])
        
        self.model = net
        self.criterion = criterion

        # NTK analysis setup
        self.ntk_analysis = ntk_analysis
        self.ntk_frequency = ntk_frequency
        self.ntk_top_k = ntk_top_k
        self.ntk_subset_size = ntk_subset_size
        self.ntk_subset_stride = ntk_subset_stride
        self.ntk_normalize = ntk_normalize
        self.ntk_training_inputs = None
        
        # metrics 
        self.train_loss = MeanMetric()
        self.train_data_loss = MeanMetric()
        self.train_ot_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.test_data_loss = MeanMetric()
        self.test_ot_loss = MeanMetric()
        self.train_rel_error = MeanMetric()
        self.train_rel_error_best = MinMetric()
        
        # NTK metrics
        if self.ntk_analysis:
            self.ntk_effective_rank = MeanMetric()
            self.ntk_condition_number = MeanMetric()
            self.ntk_spectrum_decay = MeanMetric()
            # Metrics for top-k eigenvalues
            for i in range(self.ntk_top_k):
                setattr(self, f'ntk_eigenvalue_{i+1}', MeanMetric())

        # For storing test outputs
        self.test_predictions = []
        self.test_ground_truth = []

    def _capture_training_inputs(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """Capture training inputs for NTK analysis."""
        if self.ntk_analysis:
            coords, _ = batch
            if self.ntk_training_inputs is None:
                self.ntk_training_inputs = coords.detach().cpu()
            else:
                # Accumulate all training inputs (subsetting handled in _get_ntk_inputs)
                new_inputs = coords.detach().cpu()
                self.ntk_training_inputs = torch.cat([self.ntk_training_inputs, new_inputs], dim=0)

    def _get_ntk_inputs(self) -> torch.Tensor:
        """Get inputs for NTK analysis from captured training data."""
        if self.ntk_training_inputs is None:
            raise ValueError("No training inputs captured for NTK analysis. Ensure training has started.")
        
        # Apply stride-based subsampling
        inputs = self.ntk_training_inputs[::self.ntk_subset_stride]
        
        # Apply size-based truncation if specified
        if self.ntk_subset_size is not None:
            inputs = inputs[:self.ntk_subset_size]
            
        return inputs

    def _setup_ntk_analysis(self) -> None:
        """Setup NTK analysis components."""
        try:
            # Store NTK results for analysis
            self.ntk_results_history = []
            
            print(f"✓ NTK analysis initialized: top-{self.ntk_top_k} eigenvalues, "
                  f"frequency every {self.ntk_frequency} epochs, "
                  f"using training data")
                  
        except Exception as e:
            warnings.warn(f"Failed to setup NTK analysis: {e}. Disabling NTK analysis.")
            self.ntk_analysis = False

    def _perform_ntk_analysis(self) -> Optional[Dict[str, float]]:
        """Perform NTK analysis on the current model state."""
        if not self.ntk_analysis:
            return None
            
        try:
            # Get inputs for NTK analysis
            inputs = self._get_ntk_inputs()
            device = next(self.model.parameters()).device
            inputs = inputs.to(device)
            
            # Perform NTK analysis
            result = analyze_model_ntk(
                self.model,
                inputs,
                normalize=self.ntk_normalize,
                top_k=self.ntk_top_k
            )
            
            # Extract key metrics
            metrics = {
                'effective_rank': float(result.effective_rank),
                'condition_number': float(result.condition_number),
                'spectrum_decay': float(result.spectrum_decay),
                'trace': float(result.trace),
            }
            
            # Add top-k eigenvalues
            for i, eigenval in enumerate(result.top_k_eigenvalues):
                metrics[f'eigenvalue_{i+1}'] = float(eigenval)
            
            # Store for history tracking
            metrics['epoch'] = self.current_epoch
            self.ntk_results_history.append(metrics)
            
            # Update metrics
            self.ntk_effective_rank(metrics['effective_rank'])
            self.ntk_condition_number(metrics['condition_number'])
            self.ntk_spectrum_decay(metrics['spectrum_decay'])
            
            # Update eigenvalue metrics
            for i in range(min(self.ntk_top_k, len(result.top_k_eigenvalues))):
                eigenval_metric = getattr(self, f'ntk_eigenvalue_{i+1}')
                eigenval_metric(float(result.top_k_eigenvalues[i]))
            
            return metrics
            
        except Exception as e:
            warnings.warn(f"NTK analysis failed at epoch {self.current_epoch}: {e}")
            return None

    def on_fit_start(self) -> None:
        if self.ntk_analysis:
            self._setup_ntk_analysis()

    def model_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step (forward pass + loss computation).
        
        Args:
            batch: A tuple of (coordinates, ground_truth).
            
        Returns:
            Tuple of (loss, predictions, ground_truth)
        """
        coords, gt = batch
        pred, ot_loss = self.model(coords)
        data_loss = self.criterion(pred, gt)
        loss = data_loss + ot_loss
        
        return data_loss, ot_loss, loss, pred, gt

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        # Capture training inputs for NTK analysis
        self._capture_training_inputs(batch)
        
        data_loss, ot_loss, loss, pred, gt = self.model_step(batch)
        rel_error = l2_relative_error(pred.flatten(), gt.flatten())
        self.train_data_loss(data_loss)
        self.train_ot_loss(ot_loss)
        self.train_loss(loss)
        self.train_rel_error(rel_error)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/data_loss", self.train_data_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/ot_loss", self.train_ot_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/rel_error", self.train_rel_error, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self, unused: Optional = None) -> None:
        """Called at the end of the training epoch."""
        rel_error = self.train_rel_error.compute()
        self.train_rel_error_best.update(rel_error)
        self.log("train/rel_error_best", self.train_rel_error_best.compute(), prog_bar=True)

        if self.ntk_analysis and (self.current_epoch % self.ntk_frequency == 0):
            _ = self._perform_ntk_analysis()

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Test step."""
        data_loss, ot_loss, loss, pred, gt = self.model_step(batch)
        rel_error = l2_relative_error(pred.flatten(), gt.flatten())
        self.test_loss(loss)
        self.test_data_loss(data_loss)
        self.test_ot_loss(ot_loss)
        self.log("test/rel_error", rel_error, on_step=False, on_epoch=True, prog_bar=True)
        self.test_predictions.append(pred.detach().cpu())
        self.test_ground_truth.append(gt.detach().cpu())

    def on_test_epoch_end(self) -> None:
        """Evaluate the model on the full data grid if supported."""
        preds = torch.cat(self.test_predictions).numpy()
        self.save_predictions(preds, filename="test_preds")
        
        if hasattr(self, 'ntk_analysis') and self.ntk_analysis:
            self.save_ntk_results()
            
    def setup(self, stage: str) -> None:
        """Called at the beginning of fit and test."""
        if self.hparams.compile and stage == "fit":
            self.model = torch.compile(self.model)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and learning rate schedulers.
        
        Returns:
            A dictionary containing the optimizer and scheduler configuration.
        """
        if hasattr(self.hparams, 'optimizer') and self.hparams.optimizer is not None:
            optimizer = self.hparams.optimizer(params=self.parameters())
        else:
            # Default optimizer if none is provided
            warnings.warn("No optimizer specified, using AdamW with default settings.")
            optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)

        if hasattr(self.hparams, 'scheduler') and self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)

            lr_scheduler_config = {"scheduler": scheduler}

            # For ReduceLROnPlateau, we need to specify the metric to monitor
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler_config["monitor"] = "train/loss"

            return {
                "optimizer": optimizer,
                "lr_scheduler": lr_scheduler_config,
            }
            
        return {"optimizer": optimizer}

    def save_predictions(self, predictions: np.ndarray, filename: str = "predictions.npy") -> None:
        """Save predictions to a file."""
        output_dir = self.trainer.log_dir
        np.save(f"{output_dir}/{filename}", predictions)
        
    def save_ntk_results(self, filename: str = "ntk_analysis.npy") -> None:
        """Save NTK analysis results to file."""
        if hasattr(self, 'ntk_analysis') and self.ntk_analysis and hasattr(self, 'ntk_results_history'):
            output_dir = self.trainer.log_dir
            if output_dir is not None:
                np.save(os.path.join(output_dir, filename), self.ntk_results_history)
                print(f"✓ NTK results saved to {output_dir}/{filename}")
            else:
                np.save(filename, self.ntk_results_history)
                print(f"✓ NTK results saved to {filename}")

    def get_model_info(self) -> Dict[str, Any]:
        """Return model information."""    
        info = {} 
        try:
            if hasattr(self.model, 'get_param_count'):
                trainable_params, total_params = self.model.get_param_count()
            else:
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                total_params = sum(p.numel() for p in self.model.parameters())
            info['trainable_params'] = trainable_params
            info['total_params'] = total_params
        except Exception as e:
            warnings.warn(f"Could not get parameter count: {e}")
            info['trainable_params'] = 'N/A'
            info['total_params'] = 'N/A'
            
        return info
