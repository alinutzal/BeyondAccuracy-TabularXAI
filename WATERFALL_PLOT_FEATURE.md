# Shapley Flow (Waterfall Plot) Feature

## Overview

This PR adds support for **SHAP waterfall plots** (also known as "Shapley flow" visualizations) to the `SHAPExplainer` class. Waterfall plots provide an intuitive way to explain individual predictions by showing how each feature contribution "flows" from the base value (expected prediction) to the final model output.

## What's New

### 1. New Method: `plot_waterfall()`

Added a new method to `SHAPExplainer` class:

```python
def plot_waterfall(
    self,
    instance: pd.Series,
    class_idx: int = 0,
    max_display: int = 10,
    save_path: Optional[str] = None
)
```

**Parameters:**
- `instance`: Single instance to explain (as a pandas Series)
- `class_idx`: Class index for multiclass classification (default: 0)
- `max_display`: Maximum number of features to display (default: 10)
- `save_path`: Optional path to save the plot

**What it does:**
- Visualizes how individual features push the prediction from the base value to the final prediction
- Shows both positive (red) and negative (blue) feature contributions
- Features are automatically sorted by absolute impact
- Creates clear, publication-ready visualizations

### 2. New Example: `waterfall_plot_example.py`

A comprehensive example script demonstrating:
- How to generate waterfall plots for multiple instances
- Interpreting waterfall plots for model explanations
- Best practices for using waterfall plots in practice

**Run it:**
```bash
python examples/waterfall_plot_example.py
```

### 3. Updated Documentation

- **README.md**: Added waterfall plot mention in explainability methods and usage examples
- **examples/README.md**: Added waterfall plot example to the examples list

## Why Waterfall Plots?

Waterfall plots are particularly useful for:

1. **Individual Prediction Explanations**: Show exactly which features drove a specific prediction
2. **Stakeholder Communication**: Easy to understand for non-technical audiences
3. **Model Debugging**: Quickly identify unexpected feature impacts
4. **Regulatory Compliance**: Provide transparent explanations for model decisions
5. **Trust Building**: Help users understand and trust model predictions

## How to Use

### Basic Usage

```python
from explainability import SHAPExplainer

# Initialize explainer
shap_explainer = SHAPExplainer(model, X_train, model_type='tree')

# Generate explanations
shap_explainer.explain(X_test)

# Create waterfall plot for a specific instance
instance = X_test.iloc[0]
shap_explainer.plot_waterfall(
    instance,
    max_display=10,
    save_path='waterfall_plot.png'
)
```

### Advanced Usage

```python
# For multiclass classification, specify class
shap_explainer.plot_waterfall(
    instance,
    class_idx=1,  # Explain class 1
    max_display=15,
    save_path='waterfall_class1.png'
)

# Show only top 5 features
shap_explainer.plot_waterfall(
    instance,
    max_display=5,
    save_path='waterfall_top5.png'
)
```

## Implementation Details

The waterfall plot implementation:
- Uses the modern SHAP `Explanation` object API
- Handles both binary and multiclass classification
- Automatically extracts the base value from the explainer
- Supports both TreeExplainer and KernelExplainer
- Includes comprehensive error handling and user-friendly messages
- Follows the same pattern as existing plotting methods in the codebase

## Testing

The feature has been thoroughly tested:
- ✓ Method exists and is callable
- ✓ Accepts correct parameters with proper defaults
- ✓ Generates valid PNG image files
- ✓ Works with different instances
- ✓ Handles custom `max_display` parameter
- ✓ Properly documented with docstrings
- ✓ Consistent with existing API patterns

## Compatibility

- Requires SHAP >= 0.40.0 (already in requirements.txt as >= 0.42.0)
- Compatible with all existing model types (tree, kernel, deep)
- No breaking changes to existing functionality

## Visual Example

Waterfall plots show:
- **Base value** (E[f(X)]): The average model prediction
- **Red bars**: Features that increase the prediction
- **Blue bars**: Features that decrease the prediction  
- **Final value** f(x): The actual prediction for this instance

Features are ordered by their absolute impact, making it easy to identify the most important contributors to the prediction.

## Future Enhancements

Possible future improvements:
- Interactive waterfall plots (using plotly)
- Batch waterfall plot generation
- Customizable color schemes
- Integration with experiment runner for automatic generation

## Related Files

- `src/explainability/shap_explainer.py` - Implementation
- `examples/waterfall_plot_example.py` - Usage example
- `README.md` - Main documentation
- `examples/README.md` - Examples documentation
