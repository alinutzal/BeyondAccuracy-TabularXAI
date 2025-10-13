"""
LIME (Local Interpretable Model-agnostic Explanations) implementation.
"""

import numpy as np
import pandas as pd
from lime import lime_tabular
from typing import Dict, Any, Optional, List
import matplotlib.pyplot as plt


class LIMEExplainer:
    """LIME explainer for model interpretability."""
    
    def __init__(self, model, X_train: pd.DataFrame, feature_names: List[str]):
        """
        Initialize LIME explainer.
        
        Args:
            model: Trained model
            X_train: Training data for background
            feature_names: Names of features
        """
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        
        # Initialize LIME explainer
        self.explainer = lime_tabular.LimeTabularExplainer(
            X_train.values,
            feature_names=feature_names,
            mode='classification',
            discretize_continuous=True
        )
    
    def explain_instance(
        self,
        instance: np.ndarray,
        num_features: int = 10,
        top_labels: int = 1
    ) -> Any:
        """
        Explain a single instance.
        
        Args:
            instance: Single instance to explain
            num_features: Number of features to include in explanation
            top_labels: Number of top labels to explain
            
        Returns:
            LIME explanation object
        """
        explanation = self.explainer.explain_instance(
            instance,
            self.model.predict_proba,
            num_features=num_features,
            top_labels=top_labels
        )
        return explanation
    
    def explain_instances(
        self,
        X_test: pd.DataFrame,
        num_samples: int = 100,
        num_features: int = 10
    ) -> List[Any]:
        """
        Explain multiple instances.
        
        Args:
            X_test: Test data to explain
            num_samples: Number of instances to explain
            num_features: Number of features per explanation
            
        Returns:
            List of LIME explanation objects
        """
        explanations = []
        num_samples = min(num_samples, len(X_test))
        
        for i in range(num_samples):
            instance = X_test.iloc[i].values
            exp = self.explain_instance(instance, num_features=num_features)
            explanations.append(exp)
        
        return explanations
    
    def get_feature_importance(
        self,
        X_test: pd.DataFrame,
        num_samples: int = 100
    ) -> pd.DataFrame:
        """
        Aggregate feature importance across multiple instances.
        
        Args:
            X_test: Test data
            num_samples: Number of instances to sample
            
        Returns:
            DataFrame with feature importance scores
        """
        feature_weights = {feature: [] for feature in self.feature_names}
        num_samples = min(num_samples, len(X_test))
        
        for i in range(num_samples):
            instance = X_test.iloc[i].values
            exp = self.explain_instance(instance, num_features=len(self.feature_names))
            
            # Get weights for each feature
            exp_map = dict(exp.as_list())
            for feature in self.feature_names:
                # LIME returns feature names with conditions, match them
                weight = 0.0
                for key, val in exp_map.items():
                    if feature in key:
                        weight = abs(val)
                        break
                feature_weights[feature].append(weight)
        
        # Calculate mean importance
        importance_scores = {
            feature: np.mean(weights) 
            for feature, weights in feature_weights.items()
        }
        
        feature_importance = pd.DataFrame({
            'feature': list(importance_scores.keys()),
            'importance': list(importance_scores.values())
        }).sort_values('importance', ascending=False)
        
        return feature_importance
    
    def plot_explanation(
        self,
        instance: np.ndarray,
        num_features: int = 10,
        save_path: Optional[str] = None
    ):
        """
        Plot explanation for a single instance.
        
        Args:
            instance: Instance to explain
            num_features: Number of features to show
            save_path: Path to save plot
        """
        exp = self.explain_instance(instance, num_features=num_features)
        
        fig = exp.as_pyplot_figure()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    
    def plot_feature_importance(
        self,
        X_test: pd.DataFrame,
        num_samples: int = 100,
        top_k: int = 20,
        save_path: Optional[str] = None
    ):
        """
        Plot aggregated feature importance.
        
        Args:
            X_test: Test data
            num_samples: Number of instances to sample
            top_k: Number of top features to plot
            save_path: Path to save plot
        """
        feature_importance = self.get_feature_importance(X_test, num_samples)
        
        plt.figure(figsize=(10, 8))
        top_features = feature_importance.head(top_k)
        plt.barh(top_features['feature'], top_features['importance'])
        plt.xlabel('Mean Importance Score')
        plt.title('Feature Importance (LIME)')
        plt.gca().invert_yaxis()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
