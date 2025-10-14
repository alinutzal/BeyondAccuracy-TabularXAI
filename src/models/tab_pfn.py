"""
Also includes TabPFN, a transformer-based prior-fitted network.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

try:
    from tabpfn import TabPFNClassifier as TabPFNModel
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False
    TabPFNModel = None

# PyTorch Lightning imports (for future extensibility)
try:
    import lightning as L
    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False

class TabPFNClassifier:
    """TabPFN classifier wrapper for Prior-Fitted Networks."""
    
    def __init__(self, use_lightning: bool = False, **kwargs):
        """
        Initialize TabPFN classifier.
        
        Args:
            use_lightning: Whether to use PyTorch Lightning wrapper (for API consistency; 
                          TabPFN is pre-trained and doesn't require training)
            **kwargs: Parameters for TabPFNClassifier
        """
        if not TABPFN_AVAILABLE:
            raise ImportError(
                "TabPFN is not installed. Please install it with: pip install tabpfn"
            )
        
        # Store Lightning flag (for API consistency with other models)
        self.use_lightning = use_lightning and LIGHTNING_AVAILABLE
        if use_lightning and not LIGHTNING_AVAILABLE:
            print("Warning: PyTorch Lightning not available. Proceeding without Lightning.")
        
        # TabPFN doesn't need many hyperparameters as it's a pre-trained model
        # Provide sensible defaults but only pass supported args to TabPFNModel
        default_params = {
            'device': 'cuda',  # Use 'cuda' if GPU is available
            'N_ensemble_configurations': 32
        }
        default_params.update(kwargs)

        # Inspect TabPFNModel constructor to determine accepted parameters
        valid_params = {}
        try:
            import inspect
            sig = inspect.signature(TabPFNModel.__init__)
            accepted = set(sig.parameters.keys())
            # remove 'self' if present
            accepted.discard('self')

            for k, v in default_params.items():
                if k in accepted:
                    valid_params[k] = v
        except Exception:
            # If inspection fails, fall back to only passing 'device' if present
            if 'device' in default_params:
                valid_params['device'] = default_params['device']

        try:
            self.model = TabPFNModel(**valid_params)
        except TypeError as e:
            # Some TabPFN versions may have different signatures; try fallback
            print(f"Warning: Could not initialize TabPFN with filtered parameters. Retrying with defaults. Error: {e}")
            try:
                self.model = TabPFNModel()
            except Exception as e2:
                raise RuntimeError(f"TabPFN initialization failed: {e2}")
        except Exception as e:
            # Other errors
            raise
        
        self.model_name = "TabPFN"
        self.feature_names_ = None

    def _to_numpy(self, X):
        """Convert input X to a numpy array (supports pandas, torch tensors, lists)."""
        try:
            import torch
        except Exception:
            torch = None

        if isinstance(X, (pd.DataFrame, pd.Series)):
            return X.values
        if torch is not None and isinstance(X, torch.Tensor):
            return X.cpu().numpy()
        return np.asarray(X)
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train the model."""
        # Store feature names for later use
        self.feature_names_ = X_train.columns.tolist()
        
        # TabPFN has limitations on dataset size (max 10000 samples, 100 features)
        # We'll handle this by sampling if needed
        max_samples = 10000
        max_features = 100

        X_train_np = self._to_numpy(X_train)
        y_train_np = self._to_numpy(y_train).ravel()
        
        # Limit samples if needed
        if len(X_train_np) > max_samples:
            print(f"Warning: TabPFN supports max {max_samples} samples. Sampling from {len(X_train_np)} samples.")
            indices = np.random.choice(len(X_train_np), max_samples, replace=False)
            X_train_np = X_train_np[indices]
            y_train_np = y_train_np[indices]
        
        # Limit features if needed
        if X_train_np.shape[1] > max_features:
            print(f"Warning: TabPFN supports max {max_features} features. Using first {max_features} features.")
            X_train_np = X_train_np[:, :max_features]
            self.feature_names_ = self.feature_names_[:max_features]
            self.n_features_used = max_features
        else:
            self.n_features_used = X_train_np.shape[1]
        
        self.model.fit(X_train_np, y_train_np)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        X_np = self._to_numpy(X)
        
        # Use only the features that were used during training
        if hasattr(self, 'n_features_used'):
            X_np = X_np[:, :self.n_features_used]
        
        return self.model.predict(X_np)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        X_np = self._to_numpy(X)
        
        # Use only the features that were used during training
        if hasattr(self, 'n_features_used'):
            X_np = X_np[:, :self.n_features_used]
        
        return self.model.predict_proba(X_np)
    
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
            metrics['roc_auc'] = roc_auc_score(y_test, y_proba[:, 1])
        else:
            try:
                metrics['roc_auc'] = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
            except:
                metrics['roc_auc'] = None
        
        return metrics
    
    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Note: TabPFN doesn't provide native feature importance.
        Returns uniform importance as a placeholder.
        """
        n_features = len(feature_names) if feature_names else len(self.feature_names_)
        
        # Return uniform importance since TabPFN doesn't provide this directly
        importance = np.ones(n_features) / n_features
        
        return pd.DataFrame({
            'feature': feature_names[:n_features] if feature_names else self.feature_names_,
            'importance': importance
        }).sort_values('importance', ascending=False)
