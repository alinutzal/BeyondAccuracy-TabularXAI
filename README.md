# BeyondAccuracy-TabularXAI

Beyond Accuracy: A Comprehensive Comparative Study of Gradient Boosting and Explainability Techniques for Mixed-Type Tabular Data Using SHAP, LIME, and Automated Machine Learning Pipelines

## Overview

This repository contains a complete implementation to reproduce and extend experiments comparing gradient boosting methods, deep learning models, and explainability techniques on tabular datasets. The project implements:

- **3 Gradient Boosting & Deep Learning Models**: XGBoost, LightGBM, and Transformer-based architectures
- **3 Diverse Tabular Datasets**: Breast Cancer, Adult Income, and Bank Marketing
- **2 Explainability Methods**: SHAP and LIME
- **Rigorous Interpretability Metrics**: Feature importance stability, explanation consistency, feature agreement, explanation fidelity, and complexity

## Project Structure

```
BeyondAccuracy-TabularXAI/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── gradient_boosting.py      # XGBoost and LightGBM implementations
│   │   └── deep_learning.py          # MLP and Transformer implementations
│   ├── explainability/
│   │   ├── __init__.py
│   │   ├── shap_explainer.py         # SHAP explanations
│   │   └── lime_explainer.py         # LIME explanations
│   ├── metrics/
│   │   ├── __init__.py
│   │   └── interpretability_metrics.py  # Quantitative interpretability metrics
│   ├── utils/
│   │   ├── __init__.py
│   │   └── data_loader.py            # Data loading and preprocessing
│   └── run_experiments.py            # Main experiment runner
├── notebooks/
│   ├── 01_data_exploration.ipynb     # Dataset exploration and visualization
│   ├── 02_model_training.ipynb       # Model training and evaluation
│   └── 03_explainability_analysis.ipynb  # SHAP/LIME analysis and metrics
├── data/                              # Dataset storage (auto-downloaded)
├── results/                           # Experiment results and visualizations
├── requirements.txt                   # Python dependencies
├── LICENSE                            # Apache 2.0 License
└── README.md                          # This file
```

## Features

### Models

1. **XGBoost**: Industry-standard gradient boosting with tree-based learning
2. **LightGBM**: Efficient gradient boosting with histogram-based learning
3. **Transformer**: Attention-based deep learning architecture for tabular data

### Datasets

1. **Breast Cancer**: Medical diagnosis dataset (numerical features)
2. **Adult Income**: Census data with mixed categorical and numerical features
3. **Bank Marketing**: Marketing campaign data with diverse feature types

### Explainability Methods

1. **SHAP (SHapley Additive exPlanations)**: 
   - Model-agnostic explanations based on game theory
   - Feature importance through Shapley values
   - Global and local interpretability

2. **LIME (Local Interpretable Model-agnostic Explanations)**:
   - Local approximations with interpretable models
   - Instance-level explanations
   - Feature contribution analysis

### Interpretability Metrics

1. **Feature Importance Stability**: Measures consistency of feature rankings across runs
2. **Explanation Consistency**: Compares agreement between SHAP and LIME
3. **Feature Agreement**: Jaccard similarity of top-k important features
4. **Explanation Fidelity**: Tests how well explanations match model behavior
5. **Explanation Complexity**: Average number of important features per explanation

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/alinutzal/BeyondAccuracy-TabularXAI.git
cd BeyondAccuracy-TabularXAI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running Experiments

#### Run all experiments (all datasets × all models):
```bash
cd src
python run_experiments.py
```

#### Run a single experiment:
```bash
cd src
python run_experiments.py breast_cancer XGBoost
```

#### Available datasets:
- `breast_cancer`
- `adult_income`
- `bank_marketing`

#### Available models:
- `XGBoost`
- `LightGBM`
- `Transformer`

### Using Jupyter Notebooks

Start Jupyter:
```bash
jupyter notebook
```

Then navigate to the `notebooks/` directory and run:
1. `01_data_exploration.ipynb` - Explore and visualize datasets
2. `02_model_training.ipynb` - Train and compare models
3. `03_explainability_analysis.ipynb` - Analyze interpretability with SHAP and LIME

## Results

Results are saved in the `results/` directory with the following structure:
```
results/
├── {dataset}_{model}/
│   ├── results.json                  # Metrics and metadata
│   ├── shap_summary.png              # SHAP summary plot
│   ├── shap_importance.png           # SHAP feature importance
│   ├── shap_dependence_*.png         # SHAP dependence plots for top features
│   ├── lime_importance.png           # LIME feature importance
│   ├── shap_feature_importance.csv   # SHAP importance scores
│   └── lime_feature_importance.csv   # LIME importance scores
└── summary.csv                        # Aggregated results
```

### Example Metrics

The experiments evaluate models on:
- **Accuracy**: Overall classification accuracy
- **F1 Score**: Harmonic mean of precision and recall
- **Precision**: Positive predictive value
- **Recall**: Sensitivity/true positive rate
- **ROC-AUC**: Area under the ROC curve

And interpretability on:
- **Importance Stability**: Consistency across runs (Spearman/Kendall correlation)
- **SHAP-LIME Consistency**: Agreement between explanation methods
- **Feature Agreement**: Top-k feature overlap
- **Fidelity**: Explanation accuracy in predicting model behavior
- **Complexity**: Number of important features

## Extending the Framework

### Adding New Datasets

Add a new loading method in `src/utils/data_loader.py`:
```python
def _load_new_dataset(self):
    # Load your dataset
    X = ...  # Features
    y = ...  # Target
    return X, y
```

### Adding New Models

Implement a new model class in `src/models/`:
```python
class NewModel:
    def train(self, X_train, y_train):
        pass
    
    def predict(self, X):
        pass
    
    def predict_proba(self, X):
        pass
    
    def evaluate(self, X_test, y_test):
        pass
```

### Adding New Metrics

Add new metrics in `src/metrics/interpretability_metrics.py`:
```python
@staticmethod
def new_metric(explanations, ...):
    # Calculate metric
    return score
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{beyondaccuracy2024,
  title={Beyond Accuracy: A Comprehensive Comparative Study of Gradient Boosting and Explainability Techniques for Mixed-Type Tabular Data},
  journal={FLAIRS},
  year={2024}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

This project implements techniques from:
- SHAP: Lundberg & Lee (2017)
- LIME: Ribeiro et al. (2016)
- XGBoost: Chen & Guestrin (2016)
- LightGBM: Ke et al. (2017)

## Contact

For questions or issues, please open an issue on GitHub.