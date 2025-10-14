"""
ShapIQ (Shapley Interaction Quantification) explainability implementation.

ShapIQ extends SHAP by quantifying interaction effects between features,
providing deeper insights into how features work together to influence predictions.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import matplotlib.pyplot as plt


class ShapIQExplainer:
    """ShapIQ explainer for quantifying feature interactions."""
    
    def __init__(self, model, X_train: pd.DataFrame, max_order: int = 2):
        """
        Initialize ShapIQ explainer.
        
        Args:
            model: Trained model with predict or predict_proba method
            X_train: Training data for background distribution
            max_order: Maximum interaction order to compute (default 2 for pairwise)
        """
        self.model = model
        self.X_train = X_train
        self.max_order = max_order
        self.explainer = None
        self.interaction_values = None
        self.feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else None
        
        # Try to import shapiq
        try:
            import shapiq
            self.shapiq = shapiq
            self._shapiq_available = True
        except ImportError:
            self._shapiq_available = False
            print("Warning: shapiq library not installed. Install with: pip install shapiq")
    
    def explain(self, X_test: pd.DataFrame, index: str = 'k-SII', **kwargs) -> Dict[str, Any]:
        """
        Compute Shapley interaction values for test data.
        
        Args:
            X_test: Test data to explain
            index: Interaction index to compute. Options include:
                   - 'k-SII': k-Shapley Interaction Index (default)
                   - 'STI': Shapley Taylor Index
                   - 'FSI': Faithful Shapley Interaction
                   - 'STII': Shapley Taylor Interaction Index
            **kwargs: Additional arguments passed to shapiq explainer
            
        Returns:
            Dictionary containing interaction values and metadata
        """
        if not self._shapiq_available:
            raise ImportError("shapiq library is required. Install with: pip install shapiq")
        
        # Initialize explainer if not already done
        if self.explainer is None:
            self._initialize_explainer(index, **kwargs)
        
        # Compute interaction values
        self.interaction_values = self._compute_interactions(X_test)
        
        return {
            'interaction_values': self.interaction_values,
            'index': index,
            'max_order': self.max_order,
            'n_features': X_test.shape[1],
            'n_samples': X_test.shape[0]
        }
    
    def _initialize_explainer(self, index: str, **kwargs):
        """Initialize the shapiq explainer based on model type."""
        if not self._shapiq_available:
            return
        
        try:
            # Create a prediction function wrapper
            if hasattr(self.model, 'predict_proba'):
                predict_fn = lambda x: self.model.predict_proba(x)[:, 1] if len(self.model.predict_proba(x).shape) > 1 else self.model.predict_proba(x)
            elif hasattr(self.model, 'predict'):
                predict_fn = self.model.predict
            else:
                raise ValueError("Model must have predict or predict_proba method")
            
            # Initialize the appropriate explainer
            # Note: shapiq has different explainers for different model types
            # Here we use a general approach with TreeExplainer if available
            self.explainer = self.shapiq.TabularExplainer(
                model=predict_fn,
                data=self.X_train.values if hasattr(self.X_train, 'values') else self.X_train,
                max_order=self.max_order,
                index=index,
                **kwargs
            )
        except Exception as e:
            print(f"Warning: Could not initialize shapiq explainer with TabularExplainer: {e}")
            print("Falling back to basic interaction computation")
            self.explainer = None
    
    def _compute_interactions(self, X_test: pd.DataFrame) -> Dict[Tuple, np.ndarray]:
        """
        Compute interaction values for test data.
        
        Returns:
            Dictionary mapping feature tuples to interaction values
        """
        if self.explainer is None:
            # Fallback: compute approximate interactions using game theory
            return self._approximate_interactions(X_test)
        
        try:
            # Compute interactions using shapiq
            X_values = X_test.values if hasattr(X_test, 'values') else X_test
            try:
                interactions = self.explainer.explain(X_values)
            except TypeError as te:
                # Some shapiq versions or explainer wrappers require a 'budget' kwarg.
                msg = str(te)
                if 'budget' in msg or 'missing' in msg:
                    # choose a reasonable default budget: number of samples clipped to [1, 1000]
                    n_samples = X_values.shape[0] if hasattr(X_values, 'shape') else len(X_values)
                    default_budget = int(max(1, min(1000, n_samples)))
                    try:
                        print(f"Warning: shapiq explainer requires 'budget' argument, retrying with budget={default_budget}")
                        interactions = self.explainer.explain(X_values[:default_budget], budget=default_budget)
                    except Exception as e2:
                        print(f"Warning: Error computing shapiq interactions with budget: {e2}")
                        return self._approximate_interactions(X_test)
                else:
                    # re-raise to be handled by outer except
                    raise

            return interactions
        except Exception as e:
            print(f"Warning: Error computing shapiq interactions: {e}")
            return self._approximate_interactions(X_test)
    
    def _approximate_interactions(self, X_test: pd.DataFrame) -> Dict[Tuple, np.ndarray]:
        """
        Compute approximate feature interactions when shapiq is not available.
        This uses a simple correlation-based approach as a fallback.
        """
        interactions = {}
        n_features = X_test.shape[1]
        n_samples = X_test.shape[0]
        
        # Compute pairwise interactions only (max_order 2)
        if self.max_order >= 2:
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    # Simple interaction: product of feature deviations
                    f1 = X_test.iloc[:, i] if hasattr(X_test, 'iloc') else X_test[:, i]
                    f2 = X_test.iloc[:, j] if hasattr(X_test, 'iloc') else X_test[:, j]
                    
                    interaction = (f1 - f1.mean()) * (f2 - f2.mean())
                    interactions[(i, j)] = np.array(interaction)
        
        return interactions
    
    def get_interaction_strength(self, top_k: int = 10) -> pd.DataFrame:
        """
        Get the strongest feature interactions.
        
        Args:
            top_k: Number of top interactions to return
            
        Returns:
            DataFrame with feature pairs and their interaction strengths
        """
        if self.interaction_values is None:
            raise ValueError("Must call explain() first to compute interactions")
        
        # Compute interaction strengths (mean absolute values)
        interaction_strengths = []
        
        for feature_tuple, values in self.interaction_values.items():
            if isinstance(values, np.ndarray):
                strength = np.abs(values).mean()
            else:
                strength = abs(values)
            
            # Get feature names if available
            if self.feature_names and all(isinstance(idx, int) for idx in feature_tuple):
                feature_names = tuple(self.feature_names[idx] for idx in feature_tuple)
            else:
                feature_names = feature_tuple
            
            interaction_strengths.append({
                'features': feature_names,
                'interaction_strength': strength,
                'order': len(feature_tuple)
            })
        
        # Convert to DataFrame and sort
        df = pd.DataFrame(interaction_strengths)
        df = df.sort_values('interaction_strength', ascending=False)
        
        return df.head(top_k) if len(df) > top_k else df
    
    def plot_interaction_network(
        self,
        top_k: int = 10,
        save_path: Optional[str] = None
    ):
        """
        Create a network plot showing feature interactions.
        
        Args:
            top_k: Number of top interactions to visualize
            save_path: Path to save plot
        """
        if self.interaction_values is None:
            raise ValueError("Must call explain() first to compute interactions")
        
        try:
            import networkx as nx
        except ImportError:
            print("Warning: networkx required for interaction network plots")
            print("Install with: pip install networkx")
            return
        
        # Get top interactions
        top_interactions = self.get_interaction_strength(top_k)
        
        # Create network graph
        G = nx.Graph()
        
        for _, row in top_interactions.iterrows():
            features = row['features']
            strength = row['interaction_strength']
            
            # Add nodes and edges for pairwise interactions
            if len(features) == 2:
                G.add_edge(features[0], features[1], weight=strength)
        
        if len(G.edges()) == 0:
            print("No pairwise interactions to plot")
            return
        
        # Create plot
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw network
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                               node_size=3000, alpha=0.9)
        
        # Draw edges with width based on interaction strength
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        max_weight = max(weights) if weights else 1
        
        nx.draw_networkx_edges(
            G, pos, 
            width=[5 * w / max_weight for w in weights],
            alpha=0.6,
            edge_color='gray'
        )
        
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
        
        plt.title('Feature Interaction Network\n(Edge width = interaction strength)', 
                  fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    
    def plot_interaction_heatmap(
        self,
        top_k: int = 20,
        save_path: Optional[str] = None
    ):
        """
        Create a heatmap of pairwise feature interactions.
        
        Args:
            top_k: Number of top features to include in heatmap
            save_path: Path to save plot
        """
        if self.interaction_values is None:
            raise ValueError("Must call explain() first to compute interactions")
        
        # Get pairwise interactions
        pairwise_interactions = {}
        for feature_tuple, values in self.interaction_values.items():
            if len(feature_tuple) == 2:
                strength = np.abs(values).mean() if isinstance(values, np.ndarray) else abs(values)
                pairwise_interactions[feature_tuple] = strength
        
        if not pairwise_interactions:
            print("No pairwise interactions available for heatmap")
            return
        
        # Create interaction matrix
        n_features = len(self.feature_names) if self.feature_names else max(max(ft) for ft in pairwise_interactions.keys()) + 1
        interaction_matrix = np.zeros((n_features, n_features))
        
        for (i, j), strength in pairwise_interactions.items():
            if isinstance(i, int) and isinstance(j, int):
                interaction_matrix[i, j] = strength
                interaction_matrix[j, i] = strength
        
        # Select top k features with strongest interactions
        row_sums = interaction_matrix.sum(axis=1)
        top_indices = np.argsort(row_sums)[-top_k:]
        
        sub_matrix = interaction_matrix[top_indices][:, top_indices]
        
        # Create feature labels
        if self.feature_names:
            labels = [self.feature_names[i] for i in top_indices]
        else:
            labels = [f"Feature {i}" for i in top_indices]
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        im = plt.imshow(sub_matrix, cmap='RdYlBu_r', aspect='auto')
        
        plt.colorbar(im, label='Interaction Strength')
        plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
        plt.yticks(range(len(labels)), labels)
        plt.title('Feature Interaction Heatmap\n(Top {} features)'.format(top_k), 
                  fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    
    def explain_instance(
        self,
        instance: pd.Series,
        index: str = 'k-SII'
    ) -> Dict[str, Any]:
        """
        Explain a single instance with interaction values.
        
        Args:
            instance: Single instance to explain (as Series or 1D array)
            index: Interaction index to use
            
        Returns:
            Dictionary with interaction explanations
        """
        # Convert to DataFrame if needed
        if isinstance(instance, pd.Series):
            X_instance = instance.to_frame().T
        else:
            X_instance = pd.DataFrame([instance], columns=self.feature_names)
        
        # Compute interactions
        result = self.explain(X_instance, index=index)
        
        return {
            'instance': instance,
            'interactions': self.interaction_values,
            'top_interactions': self.get_interaction_strength(top_k=5)
        }
