from typing import Any, Dict, Tuple, Optional, List

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, MinMetric
import numpy as np
import warnings
import hydra

from src.utils.metrics import l2_relative_error


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
        **kwargs
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=['net', 'criterion'])
        
        self.model = net
        self.criterion = criterion

        # metrics 
        self.train_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.train_rel_error = MeanMetric()
        self.train_rel_error_best = MinMetric()

        # For storing test outputs
        self.test_predictions = []
        self.test_ground_truth = []

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
        loss, pred, gt = self.model_step(batch)
        rel_error = l2_relative_error(pred.flatten(), gt.flatten())
        self.train_loss(loss)
        self.train_rel_error(rel_error)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/rel_error", self.train_rel_error, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_training_epoch_end(self, unused: Optional = None) -> None:
        """Called at the end of the training epoch."""
        rel_error = self.train_rel_error.compute()
        self.train_rel_error_best.update(rel_error)
        self.log("train/rel_error_best", self.train_rel_error_best.compute(), prog_bar=True)

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
        **kwargs
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=['net', 'criterion'])
        
        self.model = net
        self.criterion = criterion

        # metrics 
        self.train_loss = MeanMetric()
        self.train_data_loss = MeanMetric()
        self.train_ot_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.test_data_loss = MeanMetric()
        self.test_ot_loss = MeanMetric()
        self.train_rel_error = MeanMetric()
        self.train_rel_error_best = MinMetric()

        # For storing test outputs
        self.test_predictions = []
        self.test_ground_truth = []

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

    def on_training_epoch_end(self, unused: Optional = None) -> None:
        """Called at the end of the training epoch."""
        rel_error = self.train_rel_error.compute()
        self.train_rel_error_best.update(rel_error)
        self.log("train/rel_error_best", self.train_rel_error_best.compute(), prog_bar=True)

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