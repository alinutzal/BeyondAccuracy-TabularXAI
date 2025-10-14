# ShapIQ (Shapley Interaction Quantification) Feature

## Overview

This PR adds support for **ShapIQ (Shapley Interaction Quantification)** to the BeyondAccuracy-TabularXAI framework. ShapIQ extends traditional SHAP analysis by quantifying interaction effects between features, revealing how features work together synergistically or antagonistically to influence model predictions.

## What is ShapIQ?

ShapIQ is an advanced explainability technique that goes beyond individual feature attributions (like SHAP) to quantify **feature interactions**. While SHAP answers "how much does this feature contribute?", ShapIQ answers "how do these features work together?".

### Key Concepts

- **Individual Effects (SHAP)**: Measures each feature's standalone contribution
- **Interaction Effects (ShapIQ)**: Measures how features amplify or counteract each other
- **Interaction Order**: 
  - Order 2: Pairwise interactions (feature A × feature B)
  - Order 3+: Higher-order interactions (feature A × B × C)

### Use Cases

1. **Feature Engineering**: Discover feature synergies for creating interaction terms
2. **Model Understanding**: Understand complex model behaviors beyond individual features
3. **Feature Selection**: Identify redundant or complementary features
4. **Debugging**: Explain unexpected predictions through feature interactions
5. **Domain Insights**: Reveal domain-specific relationships between variables

## What's New

### 1. New Class: `ShapIQExplainer`

Location: `src/explainability/shapiq_explainer.py`

The `ShapIQExplainer` class provides:
- Computation of Shapley interaction values
- Multiple interaction indices (k-SII, STI, FSI, STII)
- Visualization of interaction networks and heatmaps
- Graceful fallback when shapiq library is not installed

### 2. Key Methods

#### `explain(X_test, index='k-SII')`
Computes Shapley interaction values for test data.

```python
shapiq_explainer = ShapIQExplainer(model, X_train, max_order=2)
result = shapiq_explainer.explain(X_test, index='k-SII')
```

**Parameters:**
- `X_test`: Test data to explain
- `index`: Interaction index to compute:
  - `'k-SII'`: k-Shapley Interaction Index (default, recommended)
  - `'STI'`: Shapley Taylor Index
  - `'FSI'`: Faithful Shapley Interaction
  - `'STII'`: Shapley Taylor Interaction Index

#### `get_interaction_strength(top_k=10)`
Returns the strongest feature interactions ranked by magnitude.

```python
top_interactions = shapiq_explainer.get_interaction_strength(top_k=10)
```

Returns a DataFrame with:
- `features`: Tuple of interacting features
- `interaction_strength`: Magnitude of interaction
- `order`: Interaction order (2 for pairwise)

#### `plot_interaction_network(top_k=10, save_path=None)`
Creates a network visualization showing feature interactions.

```python
shapiq_explainer.plot_interaction_network(
    top_k=10,
    save_path='interaction_network.png'
)
```

#### `plot_interaction_heatmap(top_k=20, save_path=None)`
Creates a heatmap of pairwise feature interactions.

```python
shapiq_explainer.plot_interaction_heatmap(
    top_k=20,
    save_path='interaction_heatmap.png'
)
```

#### `explain_instance(instance, index='k-SII')`
Explains a single prediction with interaction values.

```python
instance_result = shapiq_explainer.explain_instance(X_test.iloc[0])
```

### 3. Example Script: `examples/shapiq_example.py`

A comprehensive example demonstrating:
- Loading data and training a model
- Computing Shapley interactions
- Analyzing top feature interactions
- Creating visualizations (network plots, heatmaps)
- Explaining individual predictions

Run the example:
```bash
cd examples
python shapiq_example.py
```

### 4. Updated Package Exports

The `ShapIQExplainer` is now exported from the `explainability` package:

```python
from explainability import ShapIQExplainer
```

### 5. Requirements Update

Added `shapiq>=1.0.0` to `requirements.txt` in the Explainability section.

## Installation

Install the shapiq library:

```bash
pip install shapiq
```

Optional dependencies for visualizations:
```bash
pip install networkx  # For network plots
```

## Usage Examples

### Basic Usage

```python
from explainability import ShapIQExplainer
from models import XGBoostClassifier

# Train model
model = XGBoostClassifier()
model.train(X_train, y_train)

# Initialize explainer
shapiq_explainer = ShapIQExplainer(
    model, 
    X_train, 
    max_order=2  # Pairwise interactions
)

# Compute interactions
result = shapiq_explainer.explain(X_test.head(50))

# Get top interactions
top_interactions = shapiq_explainer.get_interaction_strength(top_k=10)
print(top_interactions)
```

### Visualizing Interactions

```python
# Network plot - shows connections between interacting features
shapiq_explainer.plot_interaction_network(
    top_k=10,
    save_path='interaction_network.png'
)

# Heatmap - shows all pairwise interactions
shapiq_explainer.plot_interaction_heatmap(
    top_k=20,
    save_path='interaction_heatmap.png'
)
```

### Single Instance Explanation

```python
# Explain why a specific prediction was made
instance = X_test.iloc[0]
instance_result = shapiq_explainer.explain_instance(instance)

print("Top interactions for this prediction:")
print(instance_result['top_interactions'])
```

## Interpreting Results

### Interaction Strength

- **Positive interaction**: Features amplify each other's effects
  - Example: "Age × BMI" with positive strength suggests these features work together to influence predictions
  
- **Negative interaction**: Features counteract each other
  - Example: "Exercise × Diet" with negative strength might indicate compensation effects

- **Magnitude**: Larger absolute values indicate stronger interactions

### Network Plot

- **Nodes**: Individual features
- **Edges**: Interaction relationships
- **Edge Width**: Proportional to interaction strength
- Use this to identify clusters of interacting features

### Heatmap

- **Axes**: Features
- **Color**: Interaction strength (red = strong positive, blue = strong negative)
- **Diagonal**: Should be zero (no self-interaction)
- Use this for a comprehensive overview of all pairwise interactions

## Implementation Details

### Graceful Degradation

The `ShapIQExplainer` is designed to work even when the `shapiq` library is not installed:

1. **With shapiq installed**: Uses the full shapiq library with proper Shapley interaction indices
2. **Without shapiq installed**: Falls back to approximate interaction computation using correlation-based methods

This ensures the code doesn't break if dependencies are missing, while still providing useful information.

### Fallback Interaction Computation

When shapiq is not available, the explainer computes approximate interactions using:
- Product of feature deviations from mean for pairwise interactions
- Simple but interpretable measure of feature co-variation

### Performance Considerations

- Interaction computation can be expensive for large datasets
- Recommend using samples (e.g., 50-100 instances) for initial exploration
- For production use, consider caching results
- Higher-order interactions (order > 2) significantly increase computation time

## Testing

Two test files are provided:

1. **`test_shapiq_simple.py`**: Tests module structure without requiring dependencies
   ```bash
   python test_shapiq_simple.py
   ```

2. **`test_shapiq_integration.py`**: Full integration tests (requires all dependencies)
   ```bash
   python test_shapiq_integration.py
   ```

## Related Files

### Modified Files
- `requirements.txt`: Added shapiq dependency
- `src/explainability/__init__.py`: Exported ShapIQExplainer

### New Files
- `src/explainability/shapiq_explainer.py`: Main implementation
- `examples/shapiq_example.py`: Usage example
- `test_shapiq_integration.py`: Integration tests
- `test_shapiq_simple.py`: Structure tests
- `SHAPIQ_FEATURE.md`: This documentation

## Comparison with SHAP

| Feature | SHAP | ShapIQ |
|---------|------|--------|
| Individual attribution | ✓ | ✓ |
| Pairwise interactions | - | ✓ |
| Higher-order interactions | - | ✓ |
| Visualization | Summary, dependence | Network, heatmap |
| Computational cost | Moderate | Higher |
| Use case | Feature importance | Feature synergies |

## Future Enhancements

Potential future improvements:

1. **Integration with SHAP**:
   - Combine SHAP and ShapIQ results in unified visualizations
   - Show both individual and interaction effects side-by-side

2. **Interaction-based Feature Engineering**:
   - Automatically create interaction features based on ShapIQ findings
   - Evaluate model performance with interaction features

3. **Dynamic Interaction Analysis**:
   - Track how interactions change across different data subsets
   - Identify context-dependent interactions

4. **Performance Optimization**:
   - Implement parallel computation for large datasets
   - Add caching mechanisms for repeated analyses

5. **Additional Visualizations**:
   - Sankey diagrams for interaction flows
   - 3D visualizations for higher-order interactions
   - Interactive dashboards

## References

- ShapIQ GitHub Repository: https://github.com/mmschlk/shapiq
- Original SHAP paper: Lundberg & Lee (2017)
- Shapley Interactions: Grabisch & Roubens (1999)

## Support

For issues or questions:
1. Check the example script: `examples/shapiq_example.py`
2. Review this documentation
3. Open an issue on GitHub

## License

This feature follows the same license as the BeyondAccuracy-TabularXAI project.
