"""
Deep Learning model implementations using PyTorch and PyTorch Lightning for tabular data.
Includes TabNet and simple MLP architectures.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional, Tuple
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, average_precision_score, brier_score_loss
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.models.kd import compute_distillation_loss, compute_consistency_penalty, should_enable_distillation, get_distillation_params
from src.utils import FastTensorDataLoader

# PyTorch Lightning imports
try:
    import lightning as L
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False
    print("Warning: PyTorch Lightning not available. Falling back to standard PyTorch training.")


class TabularMLP(nn.Module):
    """Multi-Layer Perceptron for tabular data."""
    
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int, dropout: float = 0.2):
        """
        Initialize MLP.
        
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            output_dim: Number of output classes
            dropout: Dropout rate
        """
        super(TabularMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class GEGLU(nn.Module):
    """Gated GELU (GEGLU) activation as in 'Pay Attention to MLPs'."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x is expected to have last dim 2 * d, split in half
        a, b = x.chunk(2, dim=-1)
        return nn.functional.gelu(a) * b


if LIGHTNING_AVAILABLE:
    class MLPLightningModule(L.LightningModule):
        """
        PyTorch Lightning wrapper for MLP tabular classifier.
        """
        def __init__(
            self,
            model: nn.Module,
            criterion: nn.Module,
            optimizer_cfg: Dict[str, Any],
            scheduler_cfg: Optional[Dict[str, Any]] = None,
            weight_decay: float = 0.0,
            mixup_cfg: Optional[Dict[str, Any]] = None,
            gaussian_noise_sigma: float = 0.0,
            distillation_cfg: Optional[Dict[str, Any]] = None,
            total_steps: int = 1000
        ):
            super().__init__()
            self.model = model
            self.criterion = criterion
            self.optimizer_cfg = optimizer_cfg
            self.scheduler_cfg = scheduler_cfg or {}  # Ensure it's always a dict
            self.weight_decay = weight_decay
            self.mixup_cfg = mixup_cfg or {'enabled': False}
            self.gaussian_noise_sigma = gaussian_noise_sigma
            self.distillation_cfg = distillation_cfg or {'enabled': False}
            self.total_steps = total_steps
            
            # Get distillation parameters
            self.distill_enabled = self.distillation_cfg.get('enabled', False)
            if self.distill_enabled:
                from kd import get_distillation_params
                self.distill_lambda, self.distill_temperature, self.consistency_penalty_cfg = get_distillation_params(
                    self.distillation_cfg, self.distill_enabled
                )
            
        def forward(self, x):
            return self.model(x)
        
        def training_step(self, batch: Tuple, batch_idx: int):
            """Training step with support for mixup, distillation, and consistency penalty."""
            if self.distill_enabled and len(batch) == 3:
                batch_X, batch_y, batch_teacher_probs = batch
            else:
                batch_X, batch_y = batch
                batch_teacher_probs = None
            
            # Training-only Gaussian noise augmentation
            if self.gaussian_noise_sigma > 0 and self.training:
                noise = torch.randn_like(batch_X) * self.gaussian_noise_sigma
                batch_X = batch_X + noise
            
            # MixUp augmentation (mutually exclusive with distillation)
            if self.mixup_cfg.get('enabled', False) and not self.distill_enabled:
                alpha = float(self.mixup_cfg.get('alpha', 0.2))
                lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
                idx = torch.randperm(batch_X.size(0), device=batch_X.device)
                mixed_X = lam * batch_X + (1 - lam) * batch_X[idx]
                targets_a, targets_b = batch_y, batch_y[idx]
                outputs = self(mixed_X)
                loss = lam * self.criterion(outputs, targets_a) + (1 - lam) * self.criterion(outputs, targets_b)
            else:
                # For consistency penalty, we need gradients w.r.t. inputs
                if self.distill_enabled and self.consistency_penalty_cfg.get('enabled', False):
                    batch_X.requires_grad_(True)
                
                outputs = self(batch_X)
                
                if self.distill_enabled and batch_teacher_probs is not None:
                    # Compute distillation loss
                    loss = compute_distillation_loss(
                        student_logits=outputs,
                        teacher_logits=batch_teacher_probs,
                        labels=batch_y,
                        criterion=self.criterion,
                        temperature=self.distill_temperature,
                        alpha=self.distill_lambda
                    )
                    
                    # Optional consistency penalty
                    if self.consistency_penalty_cfg.get('enabled', False):
                        top_k_features = self.consistency_penalty_cfg.get('top_k_features', None)
                        consistency_weight = float(self.consistency_penalty_cfg.get('weight', 0.01))
                        if top_k_features is not None and len(top_k_features) > 0:
                            jacobian_penalty = compute_consistency_penalty(
                                student_logits=outputs,
                                inputs=batch_X,
                                top_k_features=top_k_features,
                                device=self.device
                            )
                            loss = loss - consistency_weight * jacobian_penalty
                else:
                    loss = self.criterion(outputs, batch_y)
            
            self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
            return loss
        
        def validation_step(self, batch: Tuple, batch_idx: int):
            """Validation step."""
            if len(batch) == 3:
                batch_X, batch_y, _ = batch  # Ignore teacher probs in validation
            else:
                batch_X, batch_y = batch
            
            outputs = self(batch_X)
            loss = self.criterion(outputs, batch_y)
            
            # Calculate accuracy
            preds = torch.argmax(outputs, dim=1)
            acc = (preds == batch_y).float().mean()
            
            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
            return loss
        
        def configure_optimizers(self):
            """Configure optimizer and learning rate scheduler."""
            opt_name = str(self.optimizer_cfg.get('name', 'Adam'))
            lr = float(self.optimizer_cfg.get('lr', 1e-3))
            betas = tuple(self.optimizer_cfg.get('betas', (0.9, 0.999)))
            eps = float(self.optimizer_cfg.get('eps', 1e-8))
            
            if opt_name.lower() in ['adamw', 'adam_w', 'adam-w']:
                optimizer = optim.AdamW(self.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=self.weight_decay)
            else:
                optimizer = optim.Adam(self.parameters(), lr=lr, betas=betas, eps=eps)
            
            # Scheduler configuration
            if self.scheduler_cfg and str(self.scheduler_cfg.get('name', '')).lower() == 'cosine':
                warmup_prop = float(self.scheduler_cfg.get('warmup_proportion', 0.0))
                warmup_steps = int(self.total_steps * warmup_prop)
                min_lr = float(self.scheduler_cfg.get('min_lr', 0.0))
                
                def lr_lambda(current_step: int):
                    if current_step < warmup_steps and warmup_steps > 0:
                        return float(current_step) / float(max(1, warmup_steps))
                    # Cosine decay after warmup
                    progress = float(current_step - warmup_steps) / float(max(1, self.total_steps - warmup_steps))
                    cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
                    return max(min_lr / lr, float(cosine_decay))
                
                scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
                return {
                    'optimizer': optimizer,
                    'lr_scheduler': {
                        'scheduler': scheduler,
                        'interval': 'step',
                    }
                }
            
            return optimizer


    class TransformerLightningModule(L.LightningModule):
        """
        PyTorch Lightning wrapper for Transformer tabular classifier.
        """
        def __init__(
            self,
            model: nn.Module,
            criterion: nn.Module,
            optimizer_cfg: Dict[str, Any],
            scheduler_cfg: Optional[Dict[str, Any]] = None,
            weight_decay: float = 0.0,
            mixup_cfg: Optional[Dict[str, Any]] = None,
            gaussian_noise_cfg: Optional[Dict[str, Any]] = None,
            distillation_cfg: Optional[Dict[str, Any]] = None,
            total_steps: int = 1000
        ):
            super().__init__()
            self.model = model
            self.criterion = criterion
            self.optimizer_cfg = optimizer_cfg
            self.scheduler_cfg = scheduler_cfg or {}  # Ensure it's always a dict
            self.weight_decay = weight_decay
            self.mixup_cfg = mixup_cfg or {'enabled': False}
            self.gaussian_noise_cfg = gaussian_noise_cfg or {'enabled': False}
            self.distillation_cfg = distillation_cfg or {'enabled': False}
            self.total_steps = total_steps
            
            # Get distillation parameters
            self.distill_enabled = self.distillation_cfg.get('enabled', False)
            if self.distill_enabled:
                from kd import get_distillation_params
                self.distill_lambda, self.distill_temperature, self.consistency_penalty_cfg = get_distillation_params(
                    self.distillation_cfg, self.distill_enabled
                )
            
        def forward(self, x):
            return self.model(x)
        
        def training_step(self, batch: Tuple, batch_idx: int):
            """Training step with support for mixup, distillation, and Gaussian noise."""
            if self.distill_enabled and len(batch) == 3:
                batch_X, batch_y, batch_teacher_probs = batch
            else:
                batch_X, batch_y = batch
                batch_teacher_probs = None
            
            # Gaussian noise augmentation
            if self.gaussian_noise_cfg.get('enabled', False) and self.training:
                std = float(self.gaussian_noise_cfg.get('std', 0.05))
                noise = torch.randn_like(batch_X) * std
                batch_X = batch_X + noise
            
            # MixUp augmentation (mutually exclusive with distillation)
            if self.mixup_cfg.get('enabled', False) and not self.distill_enabled:
                alpha = float(self.mixup_cfg.get('alpha', 0.2))
                lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
                idx = torch.randperm(batch_X.size(0), device=batch_X.device)
                mixed_X = lam * batch_X + (1 - lam) * batch_X[idx]
                targets_a, targets_b = batch_y, batch_y[idx]
                outputs = self(mixed_X)
                loss = lam * self.criterion(outputs, targets_a) + (1 - lam) * self.criterion(outputs, targets_b)
            else:
                # For consistency penalty, we need gradients w.r.t. inputs
                if self.distill_enabled and self.consistency_penalty_cfg.get('enabled', False):
                    batch_X.requires_grad_(True)
                
                outputs = self(batch_X)
                
                if self.distill_enabled and batch_teacher_probs is not None:
                    # Compute distillation loss
                    loss = compute_distillation_loss(
                        student_logits=outputs,
                        teacher_logits=batch_teacher_probs,
                        labels=batch_y,
                        criterion=self.criterion,
                        temperature=self.distill_temperature,
                        alpha=self.distill_lambda
                    )
                    
                    # Optional consistency penalty
                    if self.consistency_penalty_cfg.get('enabled', False):
                        top_k_features = self.consistency_penalty_cfg.get('top_k_features', None)
                        consistency_weight = float(self.consistency_penalty_cfg.get('weight', 0.01))
                        if top_k_features is not None and len(top_k_features) > 0:
                            jacobian_penalty = compute_consistency_penalty(
                                student_logits=outputs,
                                inputs=batch_X,
                                top_k_features=top_k_features,
                                device=self.device
                            )
                            loss = loss - consistency_weight * jacobian_penalty
                else:
                    loss = self.criterion(outputs, batch_y)
            
            self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
            return loss
        
        def validation_step(self, batch: Tuple, batch_idx: int):
            """Validation step."""
            if len(batch) == 3:
                batch_X, batch_y, _ = batch  # Ignore teacher probs in validation
            else:
                batch_X, batch_y = batch
            
            outputs = self(batch_X)
            loss = self.criterion(outputs, batch_y)
            
            # Calculate accuracy
            preds = torch.argmax(outputs, dim=1)
            acc = (preds == batch_y).float().mean()
            
            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
            return loss
        
        def configure_optimizers(self):
            """Configure optimizer and learning rate scheduler."""
            opt_name = str(self.optimizer_cfg.get('name', 'Adam'))
            lr = float(self.optimizer_cfg.get('lr', 1e-3))
            betas = tuple(self.optimizer_cfg.get('betas', (0.9, 0.999)))
            eps = float(self.optimizer_cfg.get('eps', 1e-8))
            
            if opt_name.lower() in ['adamw', 'adam_w', 'adam-w']:
                optimizer = optim.AdamW(self.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=self.weight_decay)
            else:
                optimizer = optim.Adam(self.parameters(), lr=lr, betas=betas, eps=eps)
            
            # Scheduler configuration
            if self.scheduler_cfg and str(self.scheduler_cfg.get('name', '')).lower() == 'cosine':
                warmup_prop = float(self.scheduler_cfg.get('warmup_proportion', 0.0))
                warmup_steps = int(self.total_steps * warmup_prop)
                min_lr = float(self.scheduler_cfg.get('min_lr', 0.0))
                
                def lr_lambda(current_step: int):
                    if current_step < warmup_steps and warmup_steps > 0:
                        return float(current_step) / float(max(1, warmup_steps))
                    # Cosine decay after warmup
                    progress = float(current_step - warmup_steps) / float(max(1, self.total_steps - warmup_steps))
                    cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
                    return max(min_lr / lr, float(cosine_decay))
                
                scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
                return {
                    'optimizer': optimizer,
                    'lr_scheduler': {
                        'scheduler': scheduler,
                        'interval': 'step',
                    }
                }
            
            return optimizer


class MLPClassifier:
    """MLP classifier wrapper for tabular data.

    This implementation accepts configuration keys from YAML such as:
    - hidden_dims: list of layer sizes
    - activation: 'geglu', 'relu', or 'silu'
    - layer_norm_per_block: bool (use LayerNorm)
    - batch_norm_per_block: bool (use BatchNorm, overrides layer_norm if both set)
    - dropout, embedding_dropout
    - weight_decay
    - mixup: {enabled: bool, alpha: float}
    - gaussian_noise: {enabled: bool, std: float} for data augmentation
    - label_smoothing: float
    - optimizer: {name, lr, betas, eps}
    - scheduler: {name: 'cosine', warmup_proportion, min_lr}
    - swa: {enabled: bool, final_epochs: int} for Stochastic Weight Averaging
    - training: {batch_size, epochs, early_stopping: {enabled, monitor, patience}, auto_device}
    - distillation: {enabled: bool, teacher_probs: array, lambda: float, temperature: float, 
                     consistency_penalty: {enabled: bool, top_k_features: list, weight: float}}
    """

    def __init__(
        self,
        hidden_dims: list = [128, 64, 32],
        activation: str = 'relu',
        layer_norm_per_block: bool = False,
        batch_norm_per_block: bool = False,
        dropout: float = 0.2,
        embedding_dropout: float = 0.0,
        weight_decay: float = 0.0,
        mixup: Optional[Dict[str, Any]] = None,
        gaussian_noise: Optional[Dict[str, Any]] = None,
        label_smoothing: float = 0.0,
        optimizer: Optional[Dict[str, Any]] = None,
        scheduler: Optional[Dict[str, Any]] = None,
        swa: Optional[Dict[str, Any]] = None,
        training: Optional[Dict[str, Any]] = None,
        random_seed: Optional[int] = None,
        device: Optional[str] = None,
        gaussian_noise_sigma: float = 0.0,
        distillation: Optional[Dict[str, Any]] = None,
        use_lightning: bool = False,
        **kwargs
    ):
        """
        Initialize MLP classifier.
        
        Args:
            hidden_dims: List of hidden layer dimensions
            activation: Activation function ('relu', 'silu', 'geglu')
            layer_norm_per_block: Use LayerNorm after each layer
            batch_norm_per_block: Use BatchNorm after each layer (overrides layer_norm)
            dropout: Dropout rate
            embedding_dropout: Dropout on input features
            weight_decay: L2 regularization
            mixup: MixUp augmentation config
            gaussian_noise: Gaussian noise augmentation config
            label_smoothing: Label smoothing factor
            optimizer: Optimizer configuration
            scheduler: Learning rate scheduler configuration
            swa: Stochastic Weight Averaging configuration
            training: Training configuration
            random_seed: Random seed for reproducibility
            device: Device to use ('cpu' or 'cuda')
            use_lightning: If True, use PyTorch Lightning for training (default: False for backward compatibility)
        """
        self.hidden_dims = hidden_dims
        self.activation = activation.lower() if activation else 'relu'
        self.layer_norm_per_block = bool(layer_norm_per_block)
        self.batch_norm_per_block = bool(batch_norm_per_block)
        self.dropout = float(dropout)
        self.embedding_dropout = float(embedding_dropout)
        self.weight_decay = float(weight_decay)
        self.mixup = mixup or {'enabled': False}
        self.gaussian_noise = gaussian_noise or {'enabled': False}
        self.label_smoothing = float(label_smoothing or 0.0)
        self.optimizer_cfg = optimizer or {'name': 'Adam', 'lr': 0.001}
        self.scheduler_cfg = scheduler or {'name': None}
        self.swa_cfg = swa or {'enabled': False}
        training = training or {}
        self.batch_size = int(training.get('batch_size', 1024))
        self.epochs = int(training.get('epochs', 100))
        self.early_stopping = training.get('early_stopping', {'enabled': False})
        self.auto_device = bool(training.get('auto_device', True))
        self.random_seed = random_seed
        # training-only Gaussian noise standard deviation. Applied per-batch during training.
        self.gaussian_noise_sigma = float(gaussian_noise_sigma or 0.0)
        
        # Distillation configuration
        self.distillation = distillation or {'enabled': False}
        
        # PyTorch Lightning flag
        self.use_lightning = use_lightning and LIGHTNING_AVAILABLE

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(self.device)

        self.model = None
        self.model_name = "MLP"
        self.input_dim = None
        self.output_dim = None
        self.swa_model = None
        self.lightning_module = None

    def _to_numpy(self, X):
        """Convert input X to a numpy array.

        Supports pandas DataFrame/Series, torch.Tensor, numpy arrays, and lists.
        """
        if isinstance(X, (pd.DataFrame, pd.Series)):
            return X.values
        if isinstance(X, torch.Tensor):
            return X.cpu().numpy()
        return np.asarray(X)
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, teacher_probs: Optional[np.ndarray] = None) -> None:
        """Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            teacher_probs: Optional teacher probabilities for distillation (soft labels)
        """
        # Set random seed if provided
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            torch.manual_seed(self.random_seed)
            if LIGHTNING_AVAILABLE:
                L.seed_everything(self.random_seed, workers=True)

        # Convert to numpy / tensors
        X_np = self._to_numpy(X_train)
        y_np = self._to_numpy(y_train).ravel()
        X_tensor = torch.FloatTensor(X_np)
        y_tensor = torch.LongTensor(y_np)
        
        # Handle distillation teacher probabilities
        distill_enabled = should_enable_distillation(self.distillation, teacher_probs)
        if distill_enabled:
            # Use provided teacher_probs or get from distillation config
            if teacher_probs is not None:
                teacher_probs_np = teacher_probs
            else:
                teacher_probs_np = self.distillation.get('teacher_probs', None)
            
            if teacher_probs_np is not None:
                teacher_tensor = torch.FloatTensor(teacher_probs_np)
                dataloader = FastTensorDataLoader(X_tensor, y_tensor, teacher_tensor, batch_size=self.batch_size, shuffle=True)
            else:
                distill_enabled = False
                dataloader = FastTensorDataLoader(X_tensor, y_tensor, batch_size=self.batch_size, shuffle=True)
        else:
            dataloader = FastTensorDataLoader(X_tensor, y_tensor, batch_size=self.batch_size, shuffle=True)
        
        # If using PyTorch Lightning, delegate to Lightning training
        if self.use_lightning and LIGHTNING_AVAILABLE:
            self._train_with_lightning(X_tensor, y_tensor, teacher_tensor if distill_enabled else None, dataloader)
            return
        
        # Initialize model using configurable architecture
        self.input_dim = X_train.shape[1]
        self.output_dim = int(len(np.unique(y_train)))

        # Build layers respecting activation and layer_norm_per_block
        layers = []
        prev = self.input_dim

        # optional input (embedding) dropout
        if self.embedding_dropout and self.embedding_dropout > 0:
            layers.append(nn.Dropout(self.embedding_dropout))

        for h in self.hidden_dims:
            # If using GEGLU we need to double the linear width for gated proj
            if self.activation == 'geglu':
                layers.append(nn.Linear(prev, h * 2))
                # BatchNorm takes precedence over LayerNorm if both are set
                if self.batch_norm_per_block:
                    layers.append(nn.BatchNorm1d(h * 2))
                elif self.layer_norm_per_block:
                    layers.append(nn.LayerNorm(h * 2))
                layers.append(GEGLU())
            else:
                layers.append(nn.Linear(prev, h))
                # BatchNorm takes precedence over LayerNorm if both are set
                if self.batch_norm_per_block:
                    layers.append(nn.BatchNorm1d(h))
                elif self.layer_norm_per_block:
                    layers.append(nn.LayerNorm(h))
                
                # Activation function selection
                if self.activation == 'silu':
                    layers.append(nn.SiLU())
                else:  # default to relu
                    layers.append(nn.ReLU())

            layers.append(nn.Dropout(self.dropout))
            prev = h

        layers.append(nn.Linear(prev, self.output_dim))
        model_net = nn.Sequential(*layers)
        self.model = model_net.to(self.device)

        # Loss and optimizer (support label smoothing via CrossEntropyLoss with smoothing when available)
        if self.label_smoothing and self.label_smoothing > 0:
            try:
                criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
            except TypeError:
                # older pytorch doesn't support label_smoothing arg; fall back to standard and we'll handle smoothing manually
                criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.CrossEntropyLoss()

        opt_name = str(self.optimizer_cfg.get('name', 'AdamW'))
        lr = float(self.optimizer_cfg.get('lr', 1e-3))
        betas = tuple(self.optimizer_cfg.get('betas', (0.9, 0.999)))
        eps = float(self.optimizer_cfg.get('eps', 1e-8))

        if opt_name.lower() in ['adamw', 'adam_w', 'adam-w']:
            optimizer = optim.AdamW(self.model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=self.weight_decay)
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=betas, eps=eps)
        
        # Scheduler: cosine with warmup if requested
        scheduler = None
        # Be defensive in case scheduler_cfg is None or not a dict
        sched = self.scheduler_cfg or {}
        if isinstance(sched, dict) and str(sched.get('name', '')).lower() == 'cosine':
            total_steps = max(1, len(dataloader) * self.epochs)
            warmup_prop = float(sched.get('warmup_proportion', 0.0))
            warmup_steps = int(total_steps * warmup_prop)
            min_lr = float(sched.get('min_lr', 0.0))

            def lr_lambda(current_step: int):
                if current_step < warmup_steps and warmup_steps > 0:
                    return float(current_step) / float(max(1, warmup_steps))
                # cosine decay after warmup
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
                return max(min_lr / lr, float(cosine_decay))

            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # Get distillation parameters
        distill_lambda, distill_temperature, consistency_penalty_cfg = get_distillation_params(
            self.distillation, distill_enabled
        )
        use_consistency_penalty = consistency_penalty_cfg.get('enabled', False)
        top_k_features = consistency_penalty_cfg.get('top_k_features', None)
        consistency_weight = float(consistency_penalty_cfg.get('weight', 0.01))
        
        # Training loop with MixUp and optional manual label smoothing fallback
        # SWA setup if enabled
        swa_model = None
        swa_start_epoch = None
        if self.swa_cfg and self.swa_cfg.get('enabled', False):
            try:
                from torch.optim.swa_utils import AveragedModel, SWALR
                swa_final_epochs = int(self.swa_cfg.get('final_epochs', 30))
                swa_start_epoch = max(0, self.epochs - swa_final_epochs)
                swa_model = AveragedModel(self.model)
                # Use SWALR scheduler if we have a base scheduler, otherwise None
                swa_scheduler = SWALR(optimizer, swa_lr=lr * 0.1) if scheduler is None else None
                print(f"SWA enabled: will average weights from epoch {swa_start_epoch + 1} to {self.epochs}")
            except ImportError:
                print("Warning: torch.optim.swa_utils not available, SWA disabled")
                swa_model = None
                swa_start_epoch = None

        # Training loop with MixUp, Gaussian noise, and SWA support
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch_data in dataloader:
                if distill_enabled:
                    batch_X, batch_y, batch_teacher_probs = batch_data
                    batch_teacher_probs = batch_teacher_probs.to(self.device)
                else:
                    batch_X, batch_y = batch_data
                    
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                # Training-only Gaussian noise augmentation (applied to inputs only)
                if self.gaussian_noise_sigma and self.gaussian_noise_sigma > 0 and self.model.training:
                    noise = torch.randn_like(batch_X) * float(self.gaussian_noise_sigma)
                    batch_X = batch_X + noise

                # MixUp augmentation (note: distillation and mixup are mutually exclusive)
                if self.mixup and self.mixup.get('enabled', False) and not distill_enabled:
                    alpha = float(self.mixup.get('alpha', 0.2))
                    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
                    idx = torch.randperm(batch_X.size(0))
                    mixed_X = lam * batch_X + (1 - lam) * batch_X[idx]
                    targets_a, targets_b = batch_y, batch_y[idx]
                    optimizer.zero_grad()
                    outputs = self.model(mixed_X)
                    # compute mixup loss: lam * loss(pred, a) + (1-lam) * loss(pred, b)
                    loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                else:
                    optimizer.zero_grad()
                    
                    # For consistency penalty, we need gradients w.r.t. inputs
                    if use_consistency_penalty and top_k_features is not None and len(top_k_features) > 0:
                        batch_X.requires_grad_(True)
                    
                    outputs = self.model(batch_X)
                    
                    if distill_enabled:
                        # Compute distillation loss using kd module
                        loss = compute_distillation_loss(
                            student_logits=outputs,
                            teacher_logits=batch_teacher_probs,
                            labels=batch_y,
                            criterion=criterion,
                            temperature=distill_temperature,
                            alpha=distill_lambda
                        )
                        
                        # Optional: Add consistency penalty on SHAP top-k features
                        if use_consistency_penalty and top_k_features is not None and len(top_k_features) > 0:
                            jacobian_penalty = compute_consistency_penalty(
                                student_logits=outputs,
                                inputs=batch_X,
                                top_k_features=top_k_features,
                                device=self.device
                            )
                            # Add penalty to loss (negative to encourage larger gradients = more sensitivity)
                            loss = loss - consistency_weight * jacobian_penalty
                    else:
                        loss = criterion(outputs, batch_y)

                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                total_loss += loss.item()

            # Update SWA model if we're in the averaging phase
            if swa_model is not None and epoch >= swa_start_epoch:
                swa_model.update_parameters(self.model)
                if swa_scheduler is not None:
                    swa_scheduler.step()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                avg_loss = total_loss / len(dataloader)
                swa_status = f" [SWA averaging]" if swa_model is not None and epoch >= swa_start_epoch else ""
                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.6f}{swa_status}")

        # Finalize SWA if used
        if swa_model is not None:
            print("Finalizing SWA model...")
            try:
                import torch.optim.swa_utils as swa_utils
                # Update batch norm statistics for the SWA model
                swa_utils.update_bn(dataloader, swa_model, device=self.device)
                self.model = swa_model.module  # Replace model with SWA averaged version
                self.swa_model = swa_model
                print("SWA model finalized and set as primary model")
            except Exception as e:
                print(f"Warning: SWA finalization failed: {e}, using non-SWA model")
    
    def _train_with_lightning(self, X_tensor: torch.Tensor, y_tensor: torch.Tensor, 
                             teacher_tensor: Optional[torch.Tensor], dataloader) -> None:
        """Train model using PyTorch Lightning."""
        # Initialize model using configurable architecture
        self.input_dim = X_tensor.shape[1]
        self.output_dim = int(len(torch.unique(y_tensor)))

        # Build layers respecting activation and layer_norm_per_block
        layers = []
        prev = self.input_dim

        # optional input (embedding) dropout
        if self.embedding_dropout and self.embedding_dropout > 0:
            layers.append(nn.Dropout(self.embedding_dropout))

        for h in self.hidden_dims:
            # If using GEGLU we need to double the linear width for gated proj
            if self.activation == 'geglu':
                layers.append(nn.Linear(prev, h * 2))
                # BatchNorm takes precedence over LayerNorm if both are set
                if self.batch_norm_per_block:
                    layers.append(nn.BatchNorm1d(h * 2))
                elif self.layer_norm_per_block:
                    layers.append(nn.LayerNorm(h * 2))
                layers.append(GEGLU())
            else:
                layers.append(nn.Linear(prev, h))
                # BatchNorm takes precedence over LayerNorm if both are set
                if self.batch_norm_per_block:
                    layers.append(nn.BatchNorm1d(h))
                elif self.layer_norm_per_block:
                    layers.append(nn.LayerNorm(h))
                
                # Activation function selection
                if self.activation == 'silu':
                    layers.append(nn.SiLU())
                else:  # default to relu
                    layers.append(nn.ReLU())

            layers.append(nn.Dropout(self.dropout))
            prev = h

        layers.append(nn.Linear(prev, self.output_dim))
        model_net = nn.Sequential(*layers)
        
        # Loss function
        if self.label_smoothing and self.label_smoothing > 0:
            try:
                criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
            except TypeError:
                criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.CrossEntropyLoss()
        
        # Create Lightning module
        total_steps = len(dataloader) * self.epochs
        self.lightning_module = MLPLightningModule(
            model=model_net,
            criterion=criterion,
            optimizer_cfg=self.optimizer_cfg,
            scheduler_cfg=self.scheduler_cfg,
            weight_decay=self.weight_decay,
            mixup_cfg=self.mixup,
            gaussian_noise_sigma=self.gaussian_noise_sigma,
            distillation_cfg=self.distillation,
            total_steps=total_steps
        )
        
        # Setup callbacks
        callbacks = []
        
        # Early stopping
        if self.early_stopping.get('enabled', False):
            monitor = self.early_stopping.get('monitor', 'val_loss')
            patience = int(self.early_stopping.get('patience', 10))
            callbacks.append(EarlyStopping(monitor=monitor, patience=patience, mode='min'))
        
        # Model checkpointing (optional)
        # callbacks.append(ModelCheckpoint(monitor='val_loss', mode='min'))
        
        # Create Trainer
        trainer = L.Trainer(
            max_epochs=self.epochs,
            callbacks=callbacks,
            accelerator='auto',
            devices=1,
            logger=False,  # Disable logging for simplicity
            enable_progress_bar=True,
            enable_model_summary=False,
            deterministic=self.random_seed is not None
        )
        
        # Convert dataloader to Lightning-compatible format
        # For Lightning, we need to create a proper DataLoader with the dataset
        if teacher_tensor is not None:
            train_dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor, teacher_tensor)
        else:
            train_dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        # Train the model
        trainer.fit(self.lightning_module, train_loader)
        
        # Extract the trained model
        self.model = self.lightning_module.model
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        self.model.eval()
        X_tensor = torch.FloatTensor(self._to_numpy(X)).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
        
        return predicted.cpu().numpy()
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        self.model.eval()
        X_tensor = torch.FloatTensor(self._to_numpy(X)).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probas = torch.softmax(outputs, dim=1)
        
        return probas.cpu().numpy()
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        # Add AUC-ROC for binary classification
        if len(np.unique(y_test)) == 2:
            try:
                probs = y_proba[:, 1] if (hasattr(y_proba, 'ndim') and y_proba.ndim == 2 and y_proba.shape[1] > 1) else y_proba
                metrics['roc_auc'] = roc_auc_score(y_test, probs)
            except Exception:
                metrics['roc_auc'] = None
            try:
                metrics['pr_auc'] = average_precision_score(y_test, probs)
            except Exception:
                metrics['pr_auc'] = None
            try:
                metrics['brier_score'] = brier_score_loss(y_test, probs)
            except Exception:
                metrics['brier_score'] = None
        else:
            try:
                metrics['roc_auc'] = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
            except:
                metrics['roc_auc'] = None
        
        return metrics


class TransformerClassifier:
    """
    Transformer-based classifier for tabular data.
    Uses attention mechanism to learn feature relationships.
    
    Supports configuration keys from YAML such as:
    - d_model, nhead, num_layers, dim_feedforward, dropout
    - weight_decay
    - mixup: {enabled: bool, alpha: float}
    - gaussian_noise: {enabled: bool, std: float} for data augmentation
    - label_smoothing: float
    - optimizer: {name, lr, betas, eps}
    - scheduler: {name: 'cosine', warmup_proportion, min_lr}
    - swa: {enabled: bool, final_epochs: int} for Stochastic Weight Averaging
    - training: {batch_size, epochs, auto_device}
    - distillation: {enabled: bool, teacher_probs: array, lambda: float, temperature: float,
                     consistency_penalty: {enabled: bool, top_k_features: list, weight: float}}
    """
    
    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        weight_decay: float = 0.0,
        mixup: Optional[Dict[str, Any]] = None,
        gaussian_noise: Optional[Dict[str, Any]] = None,
        label_smoothing: float = 0.0,
        optimizer: Optional[Dict[str, Any]] = None,
        scheduler: Optional[Dict[str, Any]] = None,
        swa: Optional[Dict[str, Any]] = None,
        training: Optional[Dict[str, Any]] = None,
        random_seed: Optional[int] = None,
        device: Optional[str] = None,
        distillation: Optional[Dict[str, Any]] = None,
        use_lightning: bool = False,
        # Legacy parameters for backward compatibility
        learning_rate: float = 0.001,
        batch_size: int = 1024,
        epochs: int = 100,
        gaussian_noise_sigma: float = 0.0,
        **kwargs
    ):
        """
        Initialize Transformer classifier.
        
        Args:
            d_model: Dimension of the model
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            weight_decay: Weight decay for optimizer
            mixup: MixUp augmentation config
            gaussian_noise: Gaussian noise augmentation config
            label_smoothing: Label smoothing factor
            optimizer: Optimizer configuration
            scheduler: Learning rate scheduler configuration
            swa: Stochastic Weight Averaging configuration
            training: Training configuration
            random_seed: Random seed for reproducibility
            device: Device to use ('cpu' or 'cuda')
            use_lightning: If True, use PyTorch Lightning for training (default: False for backward compatibility)
        """
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = float(dropout)
        self.weight_decay = float(weight_decay)
        self.mixup = mixup or {'enabled': False}
        self.gaussian_noise = gaussian_noise or {'enabled': False}
        self.label_smoothing = float(label_smoothing or 0.0)
        self.optimizer_cfg = optimizer or {'name': 'Adam', 'lr': learning_rate}
        self.scheduler_cfg = scheduler or {'name': None}
        self.swa_cfg = swa or {'enabled': False}
        training = training or {}
        self.batch_size = int(training.get('batch_size', batch_size))
        self.epochs = int(training.get('epochs', epochs))
        self.auto_device = bool(training.get('auto_device', True))
        self.random_seed = random_seed
        
        # Distillation configuration
        self.distillation = distillation or {'enabled': False}
        
        # PyTorch Lightning flag
        self.use_lightning = use_lightning and LIGHTNING_AVAILABLE
        
        # Device selection

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.model = None
        self.model_name = "Transformer"
        self.input_dim = None
        self.output_dim = None
        self.swa_model = None
        self.lightning_module = None
        # training-only Gaussian noise standard deviation. Applied per-batch during training.
        self.gaussian_noise_sigma = float(gaussian_noise_sigma or 0.0)
    
    def _build_model(self):
        """Build the transformer model."""
        class TabularTransformer(nn.Module):
            def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, output_dim, dropout):
                super(TabularTransformer, self).__init__()
                
                # Input projection
                self.input_projection = nn.Linear(input_dim, d_model)
                
                # Transformer encoder
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    batch_first=True
                )
                self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                
                # Output layer
                self.fc_out = nn.Linear(d_model, output_dim)
                self.dropout = nn.Dropout(dropout)
            
            def forward(self, x):
                # x shape: (batch_size, input_dim)
                # Project to d_model and add sequence dimension
                x = self.input_projection(x).unsqueeze(1)  # (batch_size, 1, d_model)
                
                # Transformer encoding
                x = self.transformer_encoder(x)  # (batch_size, 1, d_model)
                
                # Take the output and project to class scores
                x = x.squeeze(1)  # (batch_size, d_model)
                x = self.dropout(x)
                x = self.fc_out(x)  # (batch_size, output_dim)
                
                return x
        
        return TabularTransformer(
            self.input_dim,
            self.d_model,
            self.nhead,
            self.num_layers,
            self.dim_feedforward,
            self.output_dim,
            self.dropout
        )
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, teacher_probs: Optional[np.ndarray] = None) -> None:
        """Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            teacher_probs: Optional teacher probabilities for distillation (soft labels)
        """
        # Set random seed if provided
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            torch.manual_seed(self.random_seed)
            if LIGHTNING_AVAILABLE:
                L.seed_everything(self.random_seed, workers=True)
        
        # Convert to tensors
        X_np = self._to_numpy(X_train)
        X_tensor = torch.FloatTensor(X_np)
        y_tensor = torch.LongTensor(self._to_numpy(y_train).ravel())
        
        # Handle distillation teacher probabilities
        distill_enabled = should_enable_distillation(self.distillation, teacher_probs)
        if distill_enabled:
            # Use provided teacher_probs or get from distillation config
            if teacher_probs is not None:
                teacher_probs_np = teacher_probs
            else:
                teacher_probs_np = self.distillation.get('teacher_probs', None)
            
            if teacher_probs_np is not None:
                teacher_tensor = torch.FloatTensor(teacher_probs_np)
                dataloader = FastTensorDataLoader(X_tensor, y_tensor, teacher_tensor, batch_size=self.batch_size, shuffle=True)
            else:
                distill_enabled = False
                dataloader = FastTensorDataLoader(X_tensor, y_tensor, batch_size=self.batch_size, shuffle=True)
        else:
            dataloader = FastTensorDataLoader(X_tensor, y_tensor, batch_size=self.batch_size, shuffle=True)
        
        # If using PyTorch Lightning, delegate to Lightning training
        if self.use_lightning and LIGHTNING_AVAILABLE:
            self._train_with_lightning(X_train, X_tensor, y_tensor, teacher_tensor if distill_enabled else None, dataloader)
            return
        
        # Initialize model
        self.input_dim = X_train.shape[1]
        self.output_dim = len(np.unique(y_train))
        self.model = self._build_model().to(self.device)
        
        # Loss with label smoothing support
        if self.label_smoothing and self.label_smoothing > 0:
            try:
                criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
            except TypeError:
                # Older PyTorch doesn't support label_smoothing arg
                criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.CrossEntropyLoss()
        
        # Optimizer configuration
        opt_name = str(self.optimizer_cfg.get('name', 'Adam'))
        lr = float(self.optimizer_cfg.get('lr', 0.001))
        betas = tuple(self.optimizer_cfg.get('betas', (0.9, 0.999)))
        eps = float(self.optimizer_cfg.get('eps', 1e-8))
        
        if opt_name.lower() in ['adamw', 'adam_w', 'adam-w']:
            optimizer = optim.AdamW(self.model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=self.weight_decay)
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=betas, eps=eps)
        
        # Scheduler: cosine with warmup if requested
        scheduler = None
        sched = self.scheduler_cfg or {}
        if isinstance(sched, dict) and str(sched.get('name', '')).lower() == 'cosine':
            total_steps = max(1, len(dataloader) * self.epochs)
            warmup_prop = float(sched.get('warmup_proportion', 0.0))
            warmup_steps = int(total_steps * warmup_prop)
            min_lr = float(sched.get('min_lr', 0.0))
            
            def lr_lambda(current_step: int):
                if current_step < warmup_steps and warmup_steps > 0:
                    return float(current_step) / float(max(1, warmup_steps))
                # Cosine decay after warmup
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
                return max(min_lr / lr, float(cosine_decay))
            
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # Get distillation parameters
        distill_lambda, distill_temperature, consistency_penalty_cfg = get_distillation_params(
            self.distillation, distill_enabled
        )
        use_consistency_penalty = consistency_penalty_cfg.get('enabled', False)
        top_k_features = consistency_penalty_cfg.get('top_k_features', None)
        consistency_weight = float(consistency_penalty_cfg.get('weight', 0.01))
        
        # Training loop with MixUp support
        # SWA setup if enabled
        swa_model = None
        swa_start_epoch = None
        if self.swa_cfg and self.swa_cfg.get('enabled', False):
            try:
                from torch.optim.swa_utils import AveragedModel, SWALR
                swa_final_epochs = int(self.swa_cfg.get('final_epochs', 30))
                swa_start_epoch = max(0, self.epochs - swa_final_epochs)
                swa_model = AveragedModel(self.model)
                # Use SWALR scheduler if we have a base scheduler, otherwise None
                swa_scheduler = SWALR(optimizer, swa_lr=lr * 0.1) if scheduler is None else None
                print(f"SWA enabled: will average weights from epoch {swa_start_epoch + 1} to {self.epochs}")
            except ImportError:
                print("Warning: torch.optim.swa_utils not available, SWA disabled")
                swa_model = None
                swa_start_epoch = None

        # Training loop with MixUp, Gaussian noise, and SWA support
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch_data in dataloader:
                if distill_enabled:
                    batch_X, batch_y, batch_teacher_probs = batch_data
                    batch_teacher_probs = batch_teacher_probs.to(self.device)
                else:
                    batch_X, batch_y = batch_data
                    
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Gaussian noise augmentation (applied during training only)
                if self.gaussian_noise and self.gaussian_noise.get('enabled', False):
                    noise_std = float(self.gaussian_noise.get('std', 0.01))
                    noise = torch.randn_like(batch_X) * noise_std
                    batch_X = batch_X + noise

                # MixUp augmentation
                if self.mixup and self.mixup.get('enabled', False):
                    alpha = float(self.mixup.get('alpha', 0.2))
                    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
                    idx = torch.randperm(batch_X.size(0))
                    mixed_X = lam * batch_X + (1 - lam) * batch_X[idx]
                    targets_a, targets_b = batch_y, batch_y[idx]
                    optimizer.zero_grad()
                    outputs = self.model(mixed_X)
                    # Compute mixup loss: lam * loss(pred, a) + (1-lam) * loss(pred, b)
                    loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                else:
                    optimizer.zero_grad()
                    
                    # For consistency penalty, we need gradients w.r.t. inputs
                    if use_consistency_penalty and top_k_features is not None and len(top_k_features) > 0:
                        batch_X.requires_grad_(True)
                    
                    outputs = self.model(batch_X)
                    
                    if distill_enabled:
                        # Compute distillation loss using kd module
                        loss = compute_distillation_loss(
                            student_logits=outputs,
                            teacher_logits=batch_teacher_probs,
                            labels=batch_y,
                            criterion=criterion,
                            temperature=distill_temperature,
                            alpha=distill_lambda
                        )
                        
                        # Optional: Add consistency penalty on SHAP top-k features
                        if use_consistency_penalty and top_k_features is not None and len(top_k_features) > 0:
                            jacobian_penalty = compute_consistency_penalty(
                                student_logits=outputs,
                                inputs=batch_X,
                                top_k_features=top_k_features,
                                device=self.device
                            )
                            # Add penalty to loss (negative to encourage larger gradients = more sensitivity)
                            loss = loss - consistency_weight * jacobian_penalty
                    else:
                        loss = criterion(outputs, batch_y)
                
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                
                total_loss += loss.item()

            # Update SWA model if we're in the averaging phase
            if swa_model is not None and epoch >= swa_start_epoch:
                swa_model.update_parameters(self.model)
                if swa_scheduler is not None:
                    swa_scheduler.step()
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                avg_loss = total_loss / len(dataloader)
                swa_status = f" [SWA averaging]" if swa_model is not None and epoch >= swa_start_epoch else ""
                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.6f}{swa_status}")

        # Finalize SWA if used
        if swa_model is not None:
            print("Finalizing SWA model...")
            try:
                import torch.optim.swa_utils as swa_utils
                # Update batch norm statistics for the SWA model
                swa_utils.update_bn(dataloader, swa_model, device=self.device)
                self.model = swa_model.module  # Replace model with SWA averaged version
                self.swa_model = swa_model
                print("SWA model finalized and set as primary model")
            except Exception as e:
                print(f"Warning: SWA finalization failed: {e}, using non-SWA model")
    
    def _train_with_lightning(self, X_train: pd.DataFrame, X_tensor: torch.Tensor, 
                             y_tensor: torch.Tensor, teacher_tensor: Optional[torch.Tensor], 
                             dataloader) -> None:
        """Train model using PyTorch Lightning."""
        # Initialize model
        self.input_dim = X_train.shape[1]
        self.output_dim = len(np.unique(y_tensor.numpy()))
        model = self._build_model()
        
        # Loss with label smoothing support
        if self.label_smoothing and self.label_smoothing > 0:
            try:
                criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
            except TypeError:
                # Older PyTorch doesn't support label_smoothing arg
                criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.CrossEntropyLoss()
        
        # Create Lightning module
        total_steps = len(dataloader) * self.epochs
        self.lightning_module = TransformerLightningModule(
            model=model,
            criterion=criterion,
            optimizer_cfg=self.optimizer_cfg,
            scheduler_cfg=self.scheduler_cfg,
            weight_decay=self.weight_decay,
            mixup_cfg=self.mixup,
            gaussian_noise_cfg=self.gaussian_noise,
            distillation_cfg=self.distillation,
            total_steps=total_steps
        )
        
        # Setup callbacks
        callbacks = []
        # Early stopping could be added here if needed
        
        # Create Trainer
        trainer = L.Trainer(
            max_epochs=self.epochs,
            callbacks=callbacks,
            accelerator='auto',
            devices=1,
            logger=False,  # Disable logging for simplicity
            enable_progress_bar=True,
            enable_model_summary=False,
            deterministic=self.random_seed is not None
        )
        
        # Convert dataloader to Lightning-compatible format
        if teacher_tensor is not None:
            train_dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor, teacher_tensor)
        else:
            train_dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        # Train the model
        trainer.fit(self.lightning_module, train_loader)
        
        # Extract the trained model
        self.model = self.lightning_module.model

    def _to_numpy(self, X):
        """Convert input X to a numpy array.

        Supports pandas DataFrame/Series, torch.Tensor, numpy arrays, and lists.
        """
        if isinstance(X, (pd.DataFrame, pd.Series)):
            return X.values
        if isinstance(X, torch.Tensor):
            return X.cpu().numpy()
        return np.asarray(X)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        self.model.eval()
        X_tensor = torch.FloatTensor(self._to_numpy(X)).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
        
        return predicted.cpu().numpy()
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        self.model.eval()
        X_tensor = torch.FloatTensor(self._to_numpy(X)).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probas = torch.softmax(outputs, dim=1)
        
        return probas.cpu().numpy()
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        # Add AUC-ROC for binary classification
        if len(np.unique(y_test)) == 2:
            try:
                probs = y_proba[:, 1] if (hasattr(y_proba, 'ndim') and y_proba.ndim == 2 and y_proba.shape[1] > 1) else y_proba
                metrics['roc_auc'] = roc_auc_score(y_test, probs)
            except Exception:
                metrics['roc_auc'] = None
            try:
                metrics['pr_auc'] = average_precision_score(y_test, probs)
            except Exception:
                metrics['pr_auc'] = None
            try:
                metrics['brier_score'] = brier_score_loss(y_test, probs)
            except Exception:
                metrics['brier_score'] = None
        else:
            try:
                metrics['roc_auc'] = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
            except:
                metrics['roc_auc'] = None
        
        return metrics
