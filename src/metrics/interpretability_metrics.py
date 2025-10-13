"""
Quantitative metrics for evaluating interpretability.
Includes metrics for feature importance stability, explanation consistency, and more.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import pairwise_distances


class InterpretabilityMetrics:
    """Quantitative metrics for evaluating model interpretability."""
    
    @staticmethod
    def feature_importance_stability(
        importance_scores: List[pd.DataFrame],
        method: str = 'spearman'
    ) -> float:
        """
        Measure stability of feature importance across different runs or samples.
        
        Args:
            importance_scores: List of DataFrames with 'feature' and 'importance' columns
            method: Correlation method ('spearman', 'kendall', or 'pearson')
            
        Returns:
            Mean pairwise correlation coefficient
        """
        if len(importance_scores) < 2:
            return 1.0
        
        # Align features across all DataFrames
        all_features = importance_scores[0]['feature'].tolist()
        
        # Create matrix of importance scores
        importance_matrix = []
        for scores in importance_scores:
            scores_dict = dict(zip(scores['feature'], scores['importance']))
            aligned_scores = [scores_dict.get(f, 0.0) for f in all_features]
            importance_matrix.append(aligned_scores)
        
        importance_matrix = np.array(importance_matrix)
        
        # Calculate pairwise correlations
        correlations = []
        for i in range(len(importance_matrix)):
            for j in range(i + 1, len(importance_matrix)):
                if method == 'spearman':
                    corr, _ = spearmanr(importance_matrix[i], importance_matrix[j])
                elif method == 'kendall':
                    corr, _ = kendalltau(importance_matrix[i], importance_matrix[j])
                else:  # pearson
                    corr = np.corrcoef(importance_matrix[i], importance_matrix[j])[0, 1]
                correlations.append(corr)
        
        return np.mean(correlations) if correlations else 1.0
    
    @staticmethod
    def explanation_consistency(
        explanations1: List[Dict[str, float]],
        explanations2: List[Dict[str, float]],
        metric: str = 'cosine'
    ) -> float:
        """
        Measure consistency between two sets of explanations.
        
        Args:
            explanations1: First set of explanations (list of dicts)
            explanations2: Second set of explanations (list of dicts)
            metric: Distance metric ('cosine', 'euclidean', 'manhattan')
            
        Returns:
            Mean consistency score (1 - mean distance)
        """
        if len(explanations1) != len(explanations2):
            raise ValueError("Explanation sets must have same length")
        
        distances = []
        for exp1, exp2 in zip(explanations1, explanations2):
            # Get common features
            features = set(exp1.keys()) | set(exp2.keys())
            
            # Create vectors
            vec1 = np.array([exp1.get(f, 0.0) for f in features])
            vec2 = np.array([exp2.get(f, 0.0) for f in features])
            
            # Calculate distance
            if metric == 'cosine':
                dist = pairwise_distances([vec1], [vec2], metric='cosine')[0, 0]
            elif metric == 'euclidean':
                dist = pairwise_distances([vec1], [vec2], metric='euclidean')[0, 0]
            else:  # manhattan
                dist = pairwise_distances([vec1], [vec2], metric='manhattan')[0, 0]
            
            distances.append(dist)
        
        # Return consistency (1 - mean distance)
        # Normalize by maximum possible distance for the metric
        mean_distance = np.mean(distances)
        if metric == 'cosine':
            consistency = 1 - mean_distance
        else:
            # For other metrics, normalize by the range of values
            consistency = 1 - (mean_distance / (np.max(distances) + 1e-10))
        
        return max(0.0, consistency)
    
    @staticmethod
    def explanation_fidelity(
        model,
        X: pd.DataFrame,
        explanations: List[Dict[str, float]],
        top_k: int = 5
    ) -> float:
        """
        Measure how well explanations match model behavior (fidelity).
        Tests if removing top-k important features significantly changes predictions.
        
        Args:
            model: Trained model
            X: Input data
            explanations: List of explanations (dicts mapping features to importance)
            top_k: Number of top features to test
            
        Returns:
            Mean fidelity score
        """
        fidelity_scores = []
        
        # Get original predictions
        original_preds = model.predict_proba(X)
        
        for i, explanation in enumerate(explanations):
            if i >= len(X):
                break
            
            # Get top-k features by absolute importance
            sorted_features = sorted(
                explanation.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:top_k]
            top_features = [f for f, _ in sorted_features]
            
            # Create perturbed instance (zero out top features)
            X_perturbed = X.iloc[[i]].copy()
            for feature in top_features:
                if feature in X_perturbed.columns:
                    X_perturbed[feature] = 0.0
            
            # Get perturbed prediction
            perturbed_pred = model.predict_proba(X_perturbed)
            
            # Calculate change in prediction
            pred_change = np.abs(original_preds[i] - perturbed_pred[0]).max()
            fidelity_scores.append(pred_change)
        
        return np.mean(fidelity_scores) if fidelity_scores else 0.0
    
    @staticmethod
    def feature_agreement(
        importance_df1: pd.DataFrame,
        importance_df2: pd.DataFrame,
        top_k: int = 10
    ) -> float:
        """
        Measure agreement between two feature importance rankings.
        
        Args:
            importance_df1: First importance DataFrame
            importance_df2: Second importance DataFrame
            top_k: Number of top features to consider
            
        Returns:
            Jaccard similarity of top-k features
        """
        top_features1 = set(importance_df1.head(top_k)['feature'].tolist())
        top_features2 = set(importance_df2.head(top_k)['feature'].tolist())
        
        intersection = len(top_features1 & top_features2)
        union = len(top_features1 | top_features2)
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def explanation_complexity(explanations: List[Dict[str, float]], threshold: float = 0.01) -> float:
        """
        Measure average complexity of explanations.
        Complexity is the number of features with non-negligible importance.
        
        Args:
            explanations: List of explanations
            threshold: Threshold for considering a feature important
            
        Returns:
            Mean number of important features
        """
        complexities = []
        for explanation in explanations:
            # Count features above threshold
            important_count = sum(1 for v in explanation.values() if abs(v) > threshold)
            complexities.append(important_count)
        
        return np.mean(complexities) if complexities else 0.0
    
    @staticmethod
    def compute_all_metrics(
        model,
        X_test: pd.DataFrame,
        importance_scores_list: List[pd.DataFrame],
        shap_explanations: List[Dict[str, float]],
        lime_explanations: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Compute all interpretability metrics.
        
        Args:
            model: Trained model
            X_test: Test data
            importance_scores_list: List of importance DataFrames from different runs
            shap_explanations: SHAP explanations
            lime_explanations: LIME explanations
            
        Returns:
            Dictionary of all metrics
        """
        metrics = {}
        
        # Feature importance stability
        if len(importance_scores_list) > 1:
            metrics['importance_stability_spearman'] = InterpretabilityMetrics.feature_importance_stability(
                importance_scores_list, method='spearman'
            )
            metrics['importance_stability_kendall'] = InterpretabilityMetrics.feature_importance_stability(
                importance_scores_list, method='kendall'
            )
        
        # Explanation consistency (SHAP vs LIME)
        if shap_explanations and lime_explanations:
            min_len = min(len(shap_explanations), len(lime_explanations))
            metrics['shap_lime_consistency'] = InterpretabilityMetrics.explanation_consistency(
                shap_explanations[:min_len],
                lime_explanations[:min_len]
            )
        
        # Feature agreement
        if len(importance_scores_list) > 1:
            metrics['feature_agreement_top5'] = InterpretabilityMetrics.feature_agreement(
                importance_scores_list[0], importance_scores_list[1], top_k=5
            )
            metrics['feature_agreement_top10'] = InterpretabilityMetrics.feature_agreement(
                importance_scores_list[0], importance_scores_list[1], top_k=10
            )
        
        # Explanation fidelity
        if shap_explanations:
            metrics['shap_fidelity'] = InterpretabilityMetrics.explanation_fidelity(
                model, X_test, shap_explanations, top_k=5
            )
        
        if lime_explanations:
            metrics['lime_fidelity'] = InterpretabilityMetrics.explanation_fidelity(
                model, X_test, lime_explanations, top_k=5
            )
        
        # Explanation complexity
        if shap_explanations:
            metrics['shap_complexity'] = InterpretabilityMetrics.explanation_complexity(shap_explanations)
        
        if lime_explanations:
            metrics['lime_complexity'] = InterpretabilityMetrics.explanation_complexity(lime_explanations)
        
        return metrics
