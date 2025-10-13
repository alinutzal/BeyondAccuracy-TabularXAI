"""
Gradient Boosting model implementations using XGBoost and LightGBM.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
import xgboost as xgb
import lightgbm as lgb


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
            metrics['roc_auc'] = roc_auc_score(y_test, y_proba[:, 1])
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
            metrics['roc_auc'] = roc_auc_score(y_test, y_proba[:, 1])
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
