# Quick Start Guide

This guide will help you get started with BeyondAccuracy-TabularXAI quickly.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/alinutzal/BeyondAccuracy-TabularXAI.git
cd BeyondAccuracy-TabularXAI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Test the installation:
```bash
python test_installation.py
```

If all tests pass, you're ready to go!

## Running Your First Experiment

### Option 1: Run a Single Experiment

Train XGBoost on the Breast Cancer dataset:
```bash
cd src
python run_experiments.py breast_cancer XGBoost
```

This will:
- Load and preprocess the data
- Train an XGBoost classifier
- Generate SHAP and LIME explanations
- Calculate interpretability metrics
- Save results to `results/breast_cancer_XGBoost/`

### Option 2: Run All Experiments

Run all combinations of datasets and models:
```bash
cd src
python run_experiments.py
```

This will run experiments for:
- 3 datasets: breast_cancer, adult_income, bank_marketing
- 4 models: XGBoost, LightGBM, TabPFN, Transformer

Results will be saved in the `results/` directory, with a summary in `results/summary.csv`.

**Note:** If results already exist for a dataset-model combination, the experiment will be skipped automatically to save time. To force rerunning experiments, use the `--rerun` flag:
```bash
cd src
python run_experiments.py --rerun  # Rerun all experiments
python run_experiments.py breast_cancer XGBoost --rerun  # Rerun specific experiment
```

### Option 3: Use Jupyter Notebooks

For interactive exploration, use the provided notebooks:

```bash
jupyter notebook
```

Then open:
1. `notebooks/01_data_exploration.ipynb` - Explore datasets
2. `notebooks/02_model_training.ipynb` - Train and compare models
3. `notebooks/03_explainability_analysis.ipynb` - Analyze interpretability

## Understanding the Results

After running an experiment, you'll find in `results/{dataset}_{model}/`:

- `results.json` - Complete metrics and metadata
- `shap_summary.png` - SHAP summary visualization
- `shap_importance.png` - SHAP feature importance plot
- `shap_dependence_*.png` - SHAP dependence plots showing feature effects
- `lime_importance.png` - LIME feature importance plot (if available)
- `shap_feature_importance.csv` - SHAP feature scores
- `lime_feature_importance.csv` - LIME feature scores (if available)

### Key Metrics

**Model Performance:**
- Accuracy: Overall classification accuracy
- F1 Score: Balance between precision and recall
- ROC-AUC: Area under the ROC curve

**Interpretability:**
- Importance Stability: How consistent feature rankings are
- SHAP-LIME Consistency: Agreement between explanation methods
- Feature Agreement: Overlap in top features identified
- Fidelity: How well explanations match model behavior
- Complexity: Number of important features

## Quick Examples

### Example 1: Compare Models on One Dataset
```bash
cd src
python run_experiments.py breast_cancer XGBoost
python run_experiments.py breast_cancer LightGBM
python run_experiments.py breast_cancer Transformer
```

### Example 2: Test One Model on All Datasets
```bash
cd src
python run_experiments.py breast_cancer XGBoost
python run_experiments.py adult_income XGBoost
python run_experiments.py bank_marketing XGBoost
```

### Example 3: View Results Summary
After running multiple experiments:
```bash
cat results/summary.csv
```

Or in Python:
```python
import pandas as pd
summary = pd.read_csv('results/summary.csv')
print(summary)
```

## Customization

### Change Model Parameters

Edit the model initialization in `src/run_experiments.py`:
```python
model = XGBoostClassifier(
    n_estimators=200,  # Increase trees
    max_depth=8,       # Deeper trees
    learning_rate=0.05 # Lower learning rate
)
```

### Add Your Own Dataset

Add a loading method in `src/utils/data_loader.py`:
```python
def _load_my_dataset(self):
    # Load your CSV or other format
    df = pd.read_csv('path/to/your/data.csv')
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Identify feature types
    self.feature_names = X.columns.tolist()
    self.categorical_features = [...]
    self.numerical_features = [...]
    
    return X, y
```

Then run:
```bash
python run_experiments.py my_dataset XGBoost
```

## Troubleshooting

### Issue: Module not found
**Solution:** Make sure you're in the `src/` directory when running experiments:
```bash
cd src
python run_experiments.py ...
```

### Issue: Out of memory
**Solution:** Reduce the number of samples used for explanations:
- Edit `run_experiments.py`
- Change `X_test.head(100)` to `X_test.head(50)`

### Issue: Experiments take too long
**Solution:** 
- Start with a single dataset/model combination
- Reduce epochs for deep learning models
- Use smaller datasets initially

## Next Steps

1. Explore the notebooks for interactive analysis
2. Customize models and parameters
3. Add your own datasets
4. Implement additional explainability metrics
5. Compare results across different configurations

## Getting Help

- Check the main [README.md](README.md) for detailed documentation
- Review example notebooks in `notebooks/`
- Open an issue on GitHub for bugs or questions

Happy experimenting! 🚀
