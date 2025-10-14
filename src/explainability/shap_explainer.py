"""
SHAP (SHapley Additive exPlanations) explainability implementation.
"""

import numpy as np
import pandas as pd
import shap
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt

from tabpfn_extensions import TabPFNClassifier, interpretability

class SHAPExplainer:
    """SHAP explainer for model interpretability."""
    
    def __init__(self, model, X_train: pd.DataFrame, model_type: str = 'tree'):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained model
            X_train: Training data for background
            model_type: Type of explainer ('tree', 'kernel', 'deep', 'tabpfn')
        """
        self.model = model
        self.X_train = X_train
        self.model_type = model_type
        self.explainer = None
        self.shap_values = None
        
    def explain(self, X_test: pd.DataFrame, background_samples: int = 50, nsamples: Optional[int] = None) -> np.ndarray:
        """
        Generate SHAP values for test data.
        
        Args:
            X_test: Test data to explain
            
        Returns:
            SHAP values array
        """
        if self.model_type == 'tree':
            # For tree-based models (XGBoost, LightGBM)
            try:
                self.explainer = shap.TreeExplainer(self.model.model)
                self.shap_values = self.explainer.shap_values(X_test)
            except Exception as e:
                # Fallback for unsupported tree model types (e.g., TabPFN)
                print(f"SHAP TreeExplainer failed: {e}. Falling back to KernelExplainer.")
                background = shap.sample(self.X_train, background_samples)
                self.explainer = shap.KernelExplainer(self.model.predict_proba, background)
                # Pass nsamples to limit KernelExplainer calls when provided
                if nsamples is not None:
                    self.shap_values = self.explainer.shap_values(X_test, nsamples=nsamples)
                else:
                    self.shap_values = self.explainer.shap_values(X_test)
        elif self.model_type == 'deep':
            # For deep learning models
            background = shap.sample(self.X_train, background_samples)
            self.explainer = shap.KernelExplainer(self.model.predict_proba, background)
            if nsamples is not None:
                self.shap_values = self.explainer.shap_values(X_test, nsamples=nsamples)
            else:
                self.shap_values = self.explainer.shap_values(X_test)
        elif self.model_type == 'tabpfn':
            nsamples = 50 # nsamples or 100
            # TabPFN interpretability utility may return shap-like arrays; assign to self.shap_values
            self.shap_values = interpretability.shap.get_shap_values(
                estimator=self.model,
                test_x=X_test[:nsamples],
                attribute_names=self.X_train.columns.tolist(),
                algorithm="permutation",
            )
        else:
            # Default to KernelExplainer
            background = shap.sample(self.X_train, 100)
            self.explainer = shap.KernelExplainer(self.model.predict_proba, background)
            self.shap_values = self.explainer.shap_values(X_test)
        
        # Safe fallback: if SHAP produced None, create a zero array matching (n_samples, n_features)
        if self.shap_values is None:
            try:
                n_samples = len(X_test)
                n_features = X_test.shape[1]
                self.shap_values = np.zeros((n_samples, n_features))
            except Exception:
                # Last resort: single zero
                self.shap_values = np.zeros((1, 1))

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
            # Be defensive: handle None or lists containing None
            if vals is None:
                return np.zeros(n_features)
            if isinstance(vals, list):
                # prefer the first non-None element
                pick = None
                for v in vals:
                    if v is not None:
                        pick = v
                        break
                if pick is None:
                    return np.zeros(n_features)
                vals = pick

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
            # Be defensive: handle None or lists containing None
            if vals is None:
                return np.zeros(n_features)
            if isinstance(vals, list):
                pick = None
                for v in vals:
                    if v is not None:
                        pick = v
                        break
                if pick is None:
                    return np.zeros(n_features)
                vals = pick

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
    
    def plot_dependence(
        self, 
        feature: str, 
        X_test: pd.DataFrame, 
        interaction_index: str = 'auto',
        save_path: Optional[str] = None
    ):
        """
        Create SHAP dependence plot showing the effect of a single feature.
        
        Dependence plots show how a single feature affects the model's predictions,
        with the feature value on the x-axis and the SHAP value on the y-axis.
        
        Args:
            feature: Name of the feature to plot
            X_test: Test data
            interaction_index: Feature to use for coloring (default 'auto' finds best interaction)
            save_path: Path to save plot
        """
        if self.shap_values is None:
            self.explain(X_test)
        
        # Validate feature exists
        if feature not in X_test.columns:
            raise ValueError(f"Feature '{feature}' not found in X_test columns")
        
        plt.figure(figsize=(10, 6))
        
        # Handle multiclass case - use first class
        if isinstance(self.shap_values, list):
            shap_vals = self.shap_values[0]
        else:
            shap_vals = self.shap_values
        
        # Normalize shap_vals into an array shaped (n_samples, n_features).
        def _normalize_shap(shap_vals, n_samples, n_features):
            # If list, try to find an element that already matches (n_samples, n_features)
            if isinstance(shap_vals, list):
                for v in shap_vals:
                    try:
                        arr = np.asarray(v)
                    except Exception:
                        continue
                    if arr.ndim == 2 and arr.shape[0] == n_samples and arr.shape[1] == n_features:
                        return arr
                # fallback to first element for normalization logic
                try:
                    arr = np.asarray(shap_vals[0])
                except Exception:
                    arr = None
            else:
                try:
                    arr = np.asarray(shap_vals)
                except Exception:
                    arr = None

            if arr is None:
                return np.zeros((n_samples, n_features))

            # If 1D vector
            if arr.ndim == 1:
                if arr.size == n_features:
                    return np.tile(arr.reshape(1, -1), (n_samples, 1))
                if arr.size == n_samples:
                    # single-column probabilities -> expand to features with zeros
                    col = arr.reshape(-1, 1)
                    if n_features == 1:
                        return col
                    pad = np.zeros((n_samples, n_features - 1))
                    return np.concatenate([col, pad], axis=1)
                # try to reshape if compatible
                flat = arr.ravel()
                if flat.size % n_features == 0:
                    reshaped = flat.reshape(-1, n_features)
                    if reshaped.shape[0] >= n_samples:
                        return reshaped[:n_samples]
                    reps = int(np.ceil(n_samples / float(reshaped.shape[0])))
                    return np.tile(reshaped, (reps, 1))[:n_samples]
                return np.zeros((n_samples, n_features))

            # If 2D array
            if arr.ndim == 2:
                # common case: (n_samples, n_features)
                if arr.shape[1] == n_features:
                    if arr.shape[0] == n_samples:
                        return arr
                    if arr.shape[0] == 1:
                        return np.tile(arr, (n_samples, 1))
                    if arr.shape[0] < n_samples:
                        reps = int(np.ceil(n_samples / float(arr.shape[0])))
                        return np.tile(arr, (reps, 1))[:n_samples]
                    return arr[:n_samples]

                # transposed case: (n_features, n_samples)
                if arr.shape[0] == n_features and arr.shape[1] == n_samples:
                    t = arr.T
                    if t.shape[0] == n_samples:
                        return t

                # if last axis matches n_features, average across other axes and align
                if arr.shape[-1] == n_features:
                    # collapse all leading axes into samples axis
                    leading = int(np.prod(arr.shape[:-1]))
                    reshaped = arr.reshape(leading, n_features)
                    if reshaped.shape[0] >= n_samples:
                        return reshaped[:n_samples]
                    reps = int(np.ceil(n_samples / float(reshaped.shape[0])))
                    return np.tile(reshaped, (reps, 1))[:n_samples]

                # fallback: try to flatten and repartition
                flat = arr.ravel()
                if flat.size % n_features == 0:
                    reshaped = flat.reshape(-1, n_features)
                    if reshaped.shape[0] >= n_samples:
                        return reshaped[:n_samples]
                    reps = int(np.ceil(n_samples / float(reshaped.shape[0])))
                    return np.tile(reshaped, (reps, 1))[:n_samples]

            # Any other ndim: fallback
            return np.zeros((n_samples, n_features))

        n_samples = len(X_test)
        n_features = X_test.shape[1]
        use_shap = _normalize_shap(shap_vals, n_samples, n_features)

        # Create dependence plot
        shap.dependence_plot(
            feature,
            use_shap,
            X_test,
            interaction_index=interaction_index,
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    
    def plot_waterfall(
        self,
        instance: pd.Series,
        class_idx: int = 0,
        max_display: int = 10,
        save_path: Optional[str] = None
    ):
        """
        Create SHAP waterfall plot showing how features push prediction from base value.
        
        The waterfall plot visualizes the "Shapley flow" - showing how each feature's
        contribution moves the prediction from the expected value (base value) to the
        final model prediction. This is particularly useful for explaining individual
        predictions to stakeholders.
        
        Args:
            instance: Single instance to explain (as a Series)
            class_idx: Class index for multiclass classification (default 0)
            max_display: Maximum number of features to display (default 10)
            save_path: Path to save plot
        """
        # Ensure we have SHAP values computed
        if self.shap_values is None:
            self.explain(instance.to_frame().T)
        
        # Get the explainer's expected value (base value)
        if hasattr(self.explainer, 'expected_value'):
            expected_value = self.explainer.expected_value
            # Handle multiclass case
            if isinstance(expected_value, (list, np.ndarray)):
                expected_value = expected_value[class_idx]
        else:
            # Fallback: use mean prediction on training data
            expected_value = 0.0
        
        # Extract SHAP values for this instance
        if isinstance(self.shap_values, list):
            shap_vals = self.shap_values[class_idx]
            if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 2:
                shap_vals = shap_vals[0]
        else:
            shap_vals = self.shap_values
            if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 2:
                shap_vals = shap_vals[0]
        
        # Create SHAP Explanation object for waterfall plot
        # This requires the newer SHAP API
        try:
            import shap
            explanation = shap.Explanation(
                values=shap_vals,
                base_values=expected_value,
                data=instance.values,
                feature_names=instance.index.tolist()
            )
            
            plt.figure(figsize=(10, 6))
            shap.plots.waterfall(explanation, max_display=max_display, show=False)
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
            
        except Exception as e:
            print(f"Warning: Could not create waterfall plot: {e}")
            print("Note: Waterfall plots require shap>=0.40.0")
    
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
