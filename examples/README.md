# Examples

This directory contains example scripts demonstrating how to use the BeyondAccuracy-TabularXAI framework.

## Available Examples

### simple_example.py

A complete end-to-end example demonstrating:
1. Loading and preparing data
2. Training a model (XGBoost)
3. Generating SHAP explanations
4. Calculating interpretability metrics

**Run it:**
```bash
python examples/simple_example.py
```

**Expected output:**
- Model performance metrics (accuracy, F1, ROC-AUC)
- Top important features identified by SHAP
- Interpretability metrics (stability, fidelity, complexity)

### dependence_plot_example.py

Demonstrates SHAP dependence plots showing feature effects:
1. Train a model on breast cancer dataset
2. Generate SHAP explanations
3. Create dependence plots for top features
4. Show feature interactions

**Run it:**
```bash
python examples/dependence_plot_example.py
```

**Expected output:**
- Dependence plots saved to `results/dependence_plots/`
- Shows how features affect predictions
- Reveals feature interactions through color coding

### waterfall_plot_example.py

Demonstrates SHAP waterfall plots (Shapley flow) for individual predictions:
1. Train a model on breast cancer dataset
2. Generate SHAP explanations
3. Create waterfall plots for multiple instances
4. Show how features push predictions from base value to final prediction

**Run it:**
```bash
python examples/waterfall_plot_example.py
```

**Expected output:**
- Waterfall plots saved to `results/waterfall_plots/`
- Visualizes feature contributions for individual predictions
- Shows the "flow" of Shapley values from base to final prediction

### shapiq_example.py

Demonstrates ShapIQ (Shapley Interaction Quantification) for feature interactions:
1. Train a model on breast cancer dataset
2. Compute Shapley interaction values
3. Identify top feature interactions
4. Create interaction visualizations (network plots, heatmaps)
5. Explain individual predictions with interaction effects

**Run it:**
```bash
python examples/shapiq_example.py
```

**Expected output:**
- Top feature interactions ranked by strength
- Interaction network plot saved to `results/shapiq_plots/`
- Interaction heatmap showing all pairwise interactions
- Single instance explanation with interaction effects

**Note:** Requires shapiq library: `pip install shapiq`

## Creating Your Own Examples

You can create custom examples by following this pattern:

```python
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.data_loader import DataLoader
from models import XGBoostClassifier, LightGBMClassifier, TransformerClassifier
from explainability import SHAPExplainer, LIMEExplainer, ShapIQExplainer
from metrics import InterpretabilityMetrics

# 1. Load data
loader = DataLoader('dataset_name', random_state=42)
X, y = loader.load_data()
data = loader.prepare_data(X, y, test_size=0.2)

# 2. Train model
model = XGBoostClassifier(n_estimators=100, max_depth=6)
model.train(data['X_train'], data['y_train'])

# 3. Generate explanations
explainer = SHAPExplainer(model, data['X_train'], model_type='tree')
shap_values = explainer.explain(data['X_test'])

# 4. Calculate metrics
importance = explainer.get_feature_importance(data['X_test'])
metrics = model.evaluate(data['X_test'], data['y_test'])
```

## Tips

- Start with `simple_example.py` to understand the basic workflow
- Modify parameters to see how they affect results
- Compare different models and datasets
- Experiment with different explainability methods (SHAP vs LIME)

## Additional Resources

- See [QUICKSTART.md](../QUICKSTART.md) for more usage patterns
- Check [notebooks/](../notebooks/) for interactive examples
- Read [README.md](../README.md) for full documentation
