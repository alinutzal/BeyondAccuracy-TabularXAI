# Adult Income Dataset Preprocessing Changes

This document describes the preprocessing changes made to the adult_income dataset.

## Overview

The adult_income dataset now uses enhanced preprocessing techniques specifically designed for census data with mixed features and skewed distributions.

## Changes Implemented

### 1. Rare Category Bucketing for Categorical Features

**What**: Categorical values that appear in less than 1% of the dataset are grouped into an "other" category.

**Why**: 
- Reduces noise from infrequent categorical values
- Improves model generalization
- Prevents overfitting to rare categories
- Reduces the cardinality of categorical features

**Example**:
```python
# Before: workclass = ['Private', 'Self-emp', 'Gov', 'Without-pay', 'Never-worked']
# If 'Never-worked' appears in < 1% of samples:
# After: workclass = ['Private', 'Self-emp', 'Gov', 'Without-pay', 'other']
```

### 2. QuantileTransformer with Normal Distribution for Numerical Features

**What**: Numerical features are transformed to follow a normal (Gaussian) distribution using quantile transformation.

**Why**:
- Census data often contains highly skewed distributions (e.g., capital-gain, fnlwgt)
- StandardScaler assumes normal distribution and doesn't handle skewness well
- QuantileTransformer makes distributions more normal and robust to outliers
- Improves performance of models that assume normally distributed features (e.g., neural networks)

**Example**:
```python
# Before transformation:
# capital-gain: Mean=972.51, Std=972.51, Skewness=1.87

# After QuantileTransformer:
# capital-gain: Mean=0.00, Std=1.02, Skewness=0.00 (approximately normal)
```

## Implementation Details

### Code Changes

1. **Import**: Added `QuantileTransformer` to imports in `src/utils/data_loader.py`

2. **_load_adult_income method**: 
   - Added rare category bucketing logic before label encoding
   - Threshold: < 1% of dataset samples
   
3. **prepare_data method**:
   - Added conditional logic to use `QuantileTransformer` for adult_income
   - Other datasets continue using `StandardScaler`

### Backward Compatibility

- Changes are **specific to adult_income dataset only**
- Other datasets (breast_cancer, bank_marketing) continue using StandardScaler
- No breaking changes to the API
- Existing code using DataLoader will automatically benefit from the new preprocessing

## Testing

Comprehensive tests were added in `test_adult_preprocessing.py`:

1. **Rare Category Bucketing Test**: Verifies that categories below 1% threshold are bucketed
2. **QuantileTransformer Test**: Verifies that transformation produces normal distribution
3. **Mock Adult Preprocessing Test**: Tests end-to-end preprocessing with mock adult-like data
4. **DataLoader Integration Test**: Ensures integration with existing code

All tests pass successfully.

## Usage

No changes required to existing code. Simply use DataLoader as before:

```python
from utils.data_loader import DataLoader

# Load adult_income with new preprocessing
loader = DataLoader('adult_income', random_state=42)
X, y = loader.load_data()
data = loader.prepare_data(X, y, test_size=0.2)

# X_train and X_test are now preprocessed with:
# - Rare category bucketing for categorical features
# - QuantileTransformer for numerical features
```

## Benefits

1. **Better handling of skewed distributions**: Common in real-world census data
2. **Reduced noise**: From rare categorical values
3. **Improved model performance**: Especially for models that assume normal distributions
4. **More stable features**: Robust to outliers in numerical features
5. **Better interpretability**: Normal distributions are easier to understand and explain

## Performance Impact

- **Computational**: Minimal overhead, QuantileTransformer is efficient
- **Memory**: Slightly higher due to storing quantile information
- **Accuracy**: Expected to improve model performance on adult_income dataset

## References

- [sklearn.preprocessing.QuantileTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html)
- [Handling Skewed Data](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html)

## Related Files

- `src/utils/data_loader.py`: Main implementation
- `test_adult_preprocessing.py`: Test suite
- `data/README.md`: Updated documentation
