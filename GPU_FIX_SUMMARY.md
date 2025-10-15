# GPU Device Placement Fix for MLP and Transformer Models

## Issue
The MLP and Transformer models were not running on GPU when using PyTorch Lightning training (`use_lightning=True`).

## Root Cause
In the `_train_with_lightning` methods of both `MLPClassifier` and `TransformerClassifier`, the models were created but NOT moved to the device before being wrapped in the LightningModule.

### Standard PyTorch Training (Correct)
- **MLPClassifier** (line 583): `self.model = model_net.to(self.device)` ✓
- **TransformerClassifier** (line 1117): `self.model = self._build_model().to(self.device)` ✓

### PyTorch Lightning Training (Incorrect - Now Fixed)
- **MLPClassifier._train_with_lightning** (line 788): `model_net = nn.Sequential(*layers)` ✗
- **TransformerClassifier._train_with_lightning** (line 1287): `model = self._build_model()` ✗

## Solution
Added `.to(self.device)` to move models to the correct device before wrapping them in LightningModule:

### Changes Made

#### 1. MLPClassifier._train_with_lightning (line 788)
**Before:**
```python
model_net = nn.Sequential(*layers)
```

**After:**
```python
model_net = nn.Sequential(*layers).to(self.device)
```

#### 2. TransformerClassifier._train_with_lightning (line 1287)
**Before:**
```python
model = self._build_model()
```

**After:**
```python
model = self._build_model().to(self.device)
```

## Testing
Created comprehensive test suite in `tests/test_device_placement.py` to verify:
1. MLP with standard PyTorch training - device placement correct ✓
2. MLP with PyTorch Lightning training - device placement correct ✓
3. Transformer with standard PyTorch training - device placement correct ✓
4. Transformer with PyTorch Lightning training - device placement correct ✓

All tests pass successfully on both CPU and GPU (when available).

## Impact
- **Minimal change**: Only 2 lines modified in `src/models/deep_learning.py`
- **Backward compatible**: No changes to the API or existing functionality
- **Fix verified**: All existing tests continue to pass
- **Performance**: Models will now properly utilize GPU when available with Lightning training

## Files Modified
- `src/models/deep_learning.py` - Added device placement in 2 locations
- `tests/test_device_placement.py` - New comprehensive test suite
