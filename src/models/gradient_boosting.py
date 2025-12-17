"""
Gradient Boosting model implementations using XGBoost and LightGBM.
Also includes TabPFN, a transformer-based prior-fitted network.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, average_precision_score, brier_score_loss
from sklearn.ensemble import GradientBoostingClassifier as SklearnGBC
import xgboost as xgb
import lightgbm as lgb
# CatBoost is optional; guard imports so repo works if package is missing
try:
    from catboost import CatBoostClassifier as _CatBoostClassifier
    _HAS_CATBOOST = True
except Exception:
    _CatBoostClassifier = None
    _HAS_CATBOOST = False

class XGBoostClassifier:
    """XGBoost classifier wrapper."""
    
    def __init__(self, **kwargs):
        """
        Initialize XGBoost classifier.
        
        Args:
            **kwargs: Parameters for XGBClassifier
        """
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'eval_metric': 'logloss'
        }
        default_params.update(kwargs)
        self.model = xgb.XGBClassifier(**default_params)
        self.model_name = "XGBoost"
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train the model."""
        self.model.fit(X_train, y_train)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        return self.model.predict_proba(X)
    
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
    
    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """Get feature importance scores."""
        importance = self.model.feature_importances_
        return pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

    def tune_hyperparameters(self, X, y, param_grid: Dict[str, Any], scoring: Optional[str] = 'roc_auc',
                             cv: int = 3, n_iter: Optional[int] = None, n_jobs: int = -1,
                             random_state: Optional[int] = 42, refit: bool = True, verbose: int = 1, optuna_n_trials: Optional[int] = None) -> Dict[str, Any]:
        """
        Tune hyperparameters for the underlying XGBoost estimator.

        Uses utils.hyperparameter_tuning.tune_model to run Grid/Randomized search or Optuna when `optuna_n_trials` is set.
        If `refit` is True, the wrapper's `self.model` will be replaced with the best estimator.
        """
        from utils.hyperparameter_tuning import tune_model

        res = tune_model(
            self.model,
            X,
            y,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_iter=n_iter,
            n_jobs=n_jobs,
            random_state=random_state,
            refit=refit,
            verbose=verbose,
            optuna_n_trials=optuna_n_trials,
        )

        if refit and res.get('best_estimator') is not None:
            self.model = res['best_estimator']

        return res


class LightGBMClassifier:
    """LightGBM classifier wrapper."""
    
    def __init__(self, **kwargs):
        """
        Initialize LightGBM classifier.
        
        Args:
            **kwargs: Parameters for LGBMClassifier
        """
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'verbose': -1
        }
        default_params.update(kwargs)
        self.model = lgb.LGBMClassifier(**default_params)
        self.model_name = "LightGBM"
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train the model."""
        self.model.fit(X_train, y_train)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        return self.model.predict_proba(X)
    
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
    
    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """Get feature importance scores."""
        importance = self.model.feature_importances_
        return pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

    def tune_hyperparameters(self, X, y, param_grid: Dict[str, Any], scoring: Optional[str] = 'roc_auc',
                             cv: int = 3, n_iter: Optional[int] = None, n_jobs: int = -1,
                             random_state: Optional[int] = 42, refit: bool = True, verbose: int = 1, optuna_n_trials: Optional[int] = None) -> Dict[str, Any]:
        """
        Tune hyperparameters for the underlying LightGBM estimator.

        Uses utils.hyperparameter_tuning.tune_model to run Grid/Randomized search or Optuna when `optuna_n_trials` is set.
        If `refit` is True, the wrapper's `self.model` will be replaced with the best estimator.
        """
        from utils.hyperparameter_tuning import tune_model

        res = tune_model(
            self.model,
            X,
            y,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_iter=n_iter,
            n_jobs=n_jobs,
            random_state=random_state,
            refit=refit,
            verbose=verbose,
            optuna_n_trials=optuna_n_trials,
        )

        if refit and res.get('best_estimator') is not None:
            self.model = res['best_estimator']

        return res



class GradientBoostingClassifier:
    """Scikit-learn GradientBoostingClassifier wrapper."""
    
    def __init__(self, **kwargs):
        """
        Initialize Scikit-learn GradientBoostingClassifier.
        
        Args:
            **kwargs: Parameters for GradientBoostingClassifier
        """
        default_params = {
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.1,
            'random_state': 42
        }
        default_params.update(kwargs)
        self.model = SklearnGBC(**default_params)
        self.model_name = "GradientBoosting"


class CatBoostModel:
    """CatBoost classifier wrapper (optional dependency)."""

    def __init__(self, **kwargs):
        if not _HAS_CATBOOST:
            raise ImportError('catboost is not installed; install via `pip install catboost` to use CatBoostModel')
        # sensible defaults: use small iterations by default and single thread to avoid nested parallelism
        default_params = {
            'iterations': 100,
            'depth': 6,
            'learning_rate': 0.1,
            'random_seed': 42,
            'verbose': 0,
            'thread_count': 1,
        }
        default_params.update(kwargs)
        self.model = _CatBoostClassifier(**default_params)
        self.model_name = 'CatBoost'

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train CatBoost model."""
        # CatBoost accepts pandas DataFrame directly
        self.model.fit(X_train, y_train)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        try:
            return self.model.predict_proba(X)
        except Exception:
            # CatBoost sometimes returns 1d array for binary; coerce to two-column
            p = self.model.predict_proba(X)
            if p.ndim == 1:
                return np.vstack([1 - p, p]).T
            raise

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0)
        }

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

    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        importance = self.model.get_feature_importance()
        return pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

    def tune_hyperparameters(self, X, y, param_grid: Dict[str, Any], scoring: Optional[str] = 'roc_auc',
                             cv: int = 3, n_iter: Optional[int] = None, n_jobs: int = -1,
                             random_state: Optional[int] = 42, refit: bool = True, verbose: int = 1, optuna_n_trials: Optional[int] = None) -> Dict[str, Any]:
        from utils.hyperparameter_tuning import tune_model

        res = tune_model(
            self.model,
            X,
            y,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_iter=n_iter,
            n_jobs=n_jobs,
            random_state=random_state,
            refit=refit,
            verbose=verbose,
            optuna_n_trials=optuna_n_trials,
        )

        if refit and res.get('best_estimator') is not None:
            self.model = res['best_estimator']

        return res    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train the model."""
        self.model.fit(X_train, y_train)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        return self.model.predict_proba(X)
    
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
    
    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """Get feature importance scores."""
        importance = self.model.feature_importances_
        return pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

    def tune_hyperparameters(self, X, y, param_grid: Dict[str, Any], scoring: Optional[str] = 'roc_auc',
                             cv: int = 3, n_iter: Optional[int] = None, n_jobs: int = -1,
                             random_state: Optional[int] = 42, refit: bool = True, verbose: int = 1, optuna_n_trials: Optional[int] = None) -> Dict[str, Any]:
        """
        Tune hyperparameters for the underlying scikit-learn GradientBoosting estimator.

        Uses utils.hyperparameter_tuning.tune_model to run Grid/Randomized search or Optuna when `optuna_n_trials` is set.
        If `refit` is True, the wrapper's `self.model` will be replaced with the best estimator.
        """
        from utils.hyperparameter_tuning import tune_model

        res = tune_model(
            self.model,
            X,
            y,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_iter=n_iter,
            n_jobs=n_jobs,
            random_state=random_state,
            refit=refit,
            verbose=verbose,
            optuna_n_trials=optuna_n_trials,
        )

        if refit and res.get('best_estimator') is not None:
            self.model = res['best_estimator']

        return res
