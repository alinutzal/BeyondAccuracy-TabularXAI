# Datasets

This directory contains the datasets used in the experiments. Datasets are automatically downloaded when running experiments.

## Included Datasets

### 1. Breast Cancer Dataset
- **Source**: UCI Machine Learning Repository (via scikit-learn)
- **Type**: Medical diagnosis
- **Features**: 30 numerical features
- **Samples**: 569
- **Classes**: 2 (malignant, benign)
- **Task**: Binary classification

### 2. Adult Income Dataset
- **Source**: UCI Machine Learning Repository (via OpenML)
- **Type**: Census data
- **Features**: 14 mixed (categorical + numerical)
- **Samples**: ~48,000
- **Classes**: 2 (>50K, <=50K)
- **Task**: Binary classification

### 3. Bank Marketing Dataset
- **Source**: UCI Machine Learning Repository (via OpenML)
- **Type**: Marketing campaign
- **Features**: 16 mixed (categorical + numerical)
- **Samples**: ~45,000
- **Classes**: 2 (yes, no - subscription)
- **Task**: Binary classification

## Dataset Processing

All datasets undergo the following preprocessing:
1. Missing value handling (if any)
2. Categorical feature encoding (Label Encoding with rare-category bucketing for adult_income)
3. Feature scaling:
   - **Adult Income**: QuantileTransformer with normal distribution
   - **Other datasets**: StandardScaler
4. Train-test split (80-20) with stratification

### Adult Income Special Preprocessing
The adult_income dataset uses enhanced preprocessing:
- **Categorical features**: Rare categories (< 1% of data) are bucketed into an 'other' category before encoding
- **Numerical features**: QuantileTransformer with normal distribution (instead of StandardScaler) to handle skewed distributions

## Data Loading

Datasets are loaded using the `DataLoader` class in `src/utils/data_loader.py`:

```python
from utils.data_loader import DataLoader

# Load a dataset
loader = DataLoader('breast_cancer', random_state=42)
X, y = loader.load_data()
data = loader.prepare_data(X, y, test_size=0.2)
```

## Citation

If you use these datasets, please cite the original sources:

### Breast Cancer
```
@misc{breast_cancer_wisconsin,
  author = {Wolberg, William H.},
  title = {Breast Cancer Wisconsin (Diagnostic)},
  year = {1995},
  publisher = {UCI Machine Learning Repository}
}
```

### Adult Income
```
@misc{adult_census,
  author = {Kohavi, Ron},
  title = {Adult Census Income},
  year = {1996},
  publisher = {UCI Machine Learning Repository}
}
```

### Bank Marketing
```
@article{moro2014data,
  title={A data-driven approach to predict the success of bank telemarketing},
  author={Moro, S{\'e}rgio and Cortez, Paulo and Rita, Paulo},
  journal={Decision Support Systems},
  volume={62},
  pages={22--31},
  year={2014}
}
```

## Notes

- Datasets are cached locally after first download
- All datasets are publicly available
- No private or sensitive data is included
- Data is used for research and educational purposes only
