"""
SHAP (SHapley Additive exPlanations) explainability implementation.
"""

import numpy as np
import pandas as pd
import shap
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt


class SHAPExplainer:
    """SHAP explainer for model interpretability."""
    
    def __init__(self, model, X_train: pd.DataFrame, model_type: str = 'tree'):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained model
            X_train: Training data for background
            model_type: Type of explainer ('tree', 'kernel', 'deep')
        """
        self.model = model
        self.X_train = X_train
        self.model_type = model_type
        self.explainer = None
        self.shap_values = None
        
    def explain(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Generate SHAP values for test data.
        
        Args:
            X_test: Test data to explain
            
        Returns:
            SHAP values array
        """
        if self.model_type == 'tree':
            # For tree-based models (XGBoost, LightGBM)
            self.explainer = shap.TreeExplainer(self.model.model)
            self.shap_values = self.explainer.shap_values(X_test)
        elif self.model_type == 'deep':
            # For deep learning models
            background = shap.sample(self.X_train, 100)
            self.explainer = shap.KernelExplainer(self.model.predict_proba, background)
            self.shap_values = self.explainer.shap_values(X_test)
        else:
            # Default to KernelExplainer
            background = shap.sample(self.X_train, 100)
            self.explainer = shap.KernelExplainer(self.model.predict_proba, background)
            self.shap_values = self.explainer.shap_values(X_test)
        
        return self.shap_values
    
    def plot_summary(self, X_test: pd.DataFrame, save_path: Optional[str] = None):
        """
        Create SHAP summary plot.
        
        Args:
            X_test: Test data
            save_path: Path to save plot
        """
        if self.shap_values is None:
            self.explain(X_test)
        
        plt.figure(figsize=(10, 8))
        
        # Handle multiclass case
        if isinstance(self.shap_values, list):
            shap.summary_plot(self.shap_values[0], X_test, show=False)
        else:
            shap.summary_plot(self.shap_values, X_test, show=False)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    
    def plot_feature_importance(self, X_test: pd.DataFrame, save_path: Optional[str] = None):
        """
        Create feature importance bar plot based on mean absolute SHAP values.
        
        Args:
            X_test: Test data
            save_path: Path to save plot
        """
        if self.shap_values is None:
            self.explain(X_test)
        
        # Aggregate SHAP values into a 1D importance vector (mean |SHAP| per feature)
        def _mean_abs_per_feature(vals, n_features):
            arr = np.abs(np.asarray(vals))
            # If already 1D and matches features, return
            if arr.ndim == 1 and arr.size == n_features:
                return arr.ravel()

            # Try to find which axis corresponds to features (matches n_features)
            feat_axes = [i for i, s in enumerate(arr.shape) if s == n_features]
            if feat_axes:
                feat_axis = feat_axes[-1]
                # average over all axes except the feature axis
                other_axes = tuple(i for i in range(arr.ndim) if i != feat_axis)
                if other_axes:
                    mean_per_feature = arr.mean(axis=other_axes)
                else:
                    mean_per_feature = arr
                return np.asarray(mean_per_feature).ravel()

            # If no matching axis found, try to reshape by assuming last axis is features
            if arr.ndim >= 2 and arr.shape[-1] == n_features:
                mean_per_feature = arr.mean(axis=tuple(range(arr.ndim - 1)))
                return np.asarray(mean_per_feature).ravel()

            # As a last resort, flatten and try to partition into (k, n_features)
            flat = arr.ravel()
            if flat.size % n_features == 0 and flat.size >= n_features:
                reshaped = flat.reshape(-1, n_features)
                return reshaped.mean(axis=0).ravel()

            # Cannot meaningfully map: return a constant importance vector (safe fallback)
            avg = flat.mean() if flat.size > 0 else 0.0
            return np.full(n_features, avg)

        if isinstance(self.shap_values, list):
            shap_vals = _mean_abs_per_feature(self.shap_values[0], len(X_test.columns))
        else:
            shap_vals = _mean_abs_per_feature(self.shap_values, len(X_test.columns))

        # Ensure shapes align
        if shap_vals.size != len(X_test.columns):
            # Try to reshape or reduce to match features; fall back to flatten and trim/pad
            shap_vals = np.asarray(shap_vals).ravel()
            if shap_vals.size < len(X_test.columns):
                # pad with zeros
                padded = np.zeros(len(X_test.columns), dtype=float)
                padded[:shap_vals.size] = shap_vals
                shap_vals = padded
            else:
                shap_vals = shap_vals[:len(X_test.columns)]

        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': shap_vals
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 8))
        plt.barh(feature_importance['feature'][:20], feature_importance['importance'][:20])
        plt.xlabel('Mean |SHAP value|')
        plt.title('Feature Importance (SHAP)')
        plt.gca().invert_yaxis()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        return feature_importance
    
    def get_feature_importance(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """
        Get feature importance based on mean absolute SHAP values.
        
        Args:
            X_test: Test data
            
        Returns:
            DataFrame with features and their importance scores
        """
        if self.shap_values is None:
            self.explain(X_test)
        
        # Aggregate SHAP values robustly into 1D importance vector
        def _mean_abs_per_feature(vals, n_features):
            arr = np.abs(np.asarray(vals))
            if arr.ndim == 1 and arr.size == n_features:
                return arr.ravel()

            feat_axes = [i for i, s in enumerate(arr.shape) if s == n_features]
            if feat_axes:
                feat_axis = feat_axes[-1]
                other_axes = tuple(i for i in range(arr.ndim) if i != feat_axis)
                if other_axes:
                    mean_per_feature = arr.mean(axis=other_axes)
                else:
                    mean_per_feature = arr
                return np.asarray(mean_per_feature).ravel()

            if arr.ndim >= 2 and arr.shape[-1] == n_features:
                mean_per_feature = arr.mean(axis=tuple(range(arr.ndim - 1)))
                return np.asarray(mean_per_feature).ravel()

            flat = arr.ravel()
            if flat.size % n_features == 0 and flat.size >= n_features:
                reshaped = flat.reshape(-1, n_features)
                return reshaped.mean(axis=0).ravel()

            avg = flat.mean() if flat.size > 0 else 0.0
            return np.full(n_features, avg)

        if isinstance(self.shap_values, list):
            shap_vals = _mean_abs_per_feature(self.shap_values[0], len(X_test.columns))
        else:
            shap_vals = _mean_abs_per_feature(self.shap_values, len(X_test.columns))

        # Align shape with feature names
        if shap_vals.size != len(X_test.columns):
            shap_vals = np.asarray(shap_vals).ravel()
            if shap_vals.size < len(X_test.columns):
                padded = np.zeros(len(X_test.columns), dtype=float)
                padded[:shap_vals.size] = shap_vals
                shap_vals = padded
            else:
                shap_vals = shap_vals[:len(X_test.columns)]

        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': shap_vals
        }).sort_values('importance', ascending=False)

        return feature_importance
    
    def explain_instance(self, instance: pd.Series, class_idx: int = 0) -> Dict[str, float]:
        """
        Explain a single instance.
        
        Args:
            instance: Single instance to explain
            class_idx: Class index for multiclass classification
            
        Returns:
            Dictionary mapping features to SHAP values
        """
        if self.shap_values is None:
            # Need to compute SHAP values for at least this instance
            self.explain(instance.to_frame().T)

        # Extract the SHAP values corresponding to this single instance
        if isinstance(self.shap_values, list):
            vals = self.shap_values
            # vals[class_idx] should be (n_samples, n_features)
            arr = np.asarray(vals[class_idx])
            if arr.ndim == 2:
                instance_shap = arr[0]
            else:
                instance_shap = arr.ravel()
        else:
            arr = np.asarray(self.shap_values)
            if arr.ndim == 2:
                instance_shap = arr[0]
            else:
                instance_shap = arr.ravel()

        explanation = dict(zip(instance.index, instance_shap))
        return explanation
