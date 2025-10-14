# Migrating to Hydra Configuration Management

This guide helps you migrate from the traditional YAML config approach to Hydra configuration management.

## Why Migrate to Hydra?

### Benefits:
1. **Easier Parameter Sweeps**: Run multiple configurations without scripts
2. **Command-Line Overrides**: Change parameters without editing files
3. **Better Organization**: Separate configs for different scenarios
4. **Automatic Tracking**: Hydra saves all configurations used
5. **Multirun Support**: Built-in support for hyperparameter searches

## Migration Path

### Option 1: Keep Using Original Method (Recommended for Simple Cases)

If you're happy with the current approach, **no migration is needed**. The original `run_experiments.py` continues to work:

```bash
cd src
python run_experiments.py breast_cancer XGBoost
```

### Option 2: Gradually Adopt Hydra (Recommended)

Use Hydra for new experiments while keeping existing workflows:

**Old approach:**
```bash
python run_experiments.py breast_cancer XGBoost
```

**New approach:**
```bash
python run_experiments_hydra.py model=xgboost_default dataset.name=breast_cancer
```

Both work side by side!

### Option 3: Full Migration (Advanced Users)

Switch entirely to Hydra for all experiments.

## Comparison: Old vs New

### Scenario 1: Basic Experiment

**Old Method:**
```bash
python run_experiments.py breast_cancer XGBoost
```

**Hydra Method:**
```bash
python run_experiments_hydra.py model=xgboost_default dataset.name=breast_cancer
```

### Scenario 2: Custom Parameters

**Old Method:**
Create `configs/breast_cancer_XGBoost.yaml`:
```yaml
n_estimators: 200
max_depth: 8
learning_rate: 0.05
```

Then run:
```bash
python run_experiments.py breast_cancer XGBoost
```

**Hydra Method:**
```bash
# Option A: Command-line override
python run_experiments_hydra.py \
  model=xgboost_default \
  dataset.name=breast_cancer \
  model.params.n_estimators=200 \
  model.params.max_depth=8 \
  model.params.learning_rate=0.05

# Option B: Create custom config (conf/model/xgboost_custom.yaml)
python run_experiments_hydra.py model=xgboost_custom dataset.name=breast_cancer
```

### Scenario 3: Parameter Sweep

**Old Method:**
Write a shell script:
```bash
#!/bin/bash
for lr in 0.01 0.05 0.1; do
  # Modify config file
  # Run experiment
done
```

**Hydra Method:**
```bash
python run_experiments_hydra.py -m \
  model=xgboost_default \
  dataset.name=breast_cancer \
  model.params.learning_rate=0.01,0.05,0.1
```

## Converting Existing Configs

### Example: Converting a Custom XGBoost Config

**Old config** (`configs/adult_income_XGBoost.yaml`):
```yaml
n_estimators: 300
max_depth: 8
learning_rate: 0.03
subsample: 0.8
colsample_bytree: 0.8
```

**New Hydra config** (`conf/model/xgboost_adult_custom.yaml`):
```yaml
# @package _global_
model:
  name: XGBoost
  params:
    n_estimators: 300
    max_depth: 8
    learning_rate: 0.03
    subsample: 0.8
    colsample_bytree: 0.8
    random_state: 42
    eval_metric: logloss
```

## Migration Checklist

### For Simple Users:
- [ ] Try out `run_experiments_hydra.py` with default configs
- [ ] Compare results with original method
- [ ] Decide if Hydra benefits your workflow
- [ ] Keep using original method if preferred

### For Advanced Users:
- [ ] Review current config files in `configs/`
- [ ] Create corresponding Hydra configs in `conf/model/`
- [ ] Test Hydra configs with small experiments
- [ ] Update any automation scripts to use Hydra
- [ ] Migrate gradually, dataset by dataset

### For Researchers Running Parameter Sweeps:
- [ ] Read Hydra multirun documentation
- [ ] Convert parameter sweep scripts to Hydra multirun
- [ ] Test with small parameter ranges
- [ ] Scale up to full experiments
- [ ] Enjoy easier experiment management!

## Common Questions

### Q: Do I have to migrate?
**A:** No! The original method still works perfectly.

### Q: Can I use both methods?
**A:** Yes! They work side by side.

### Q: What happens to my existing results?
**A:** Nothing changes. Existing results remain compatible.

### Q: Which method is better?
**A:** 
- **Original method**: Simpler, good for single experiments
- **Hydra method**: More powerful, better for parameter sweeps and experimentation

### Q: Will the old method be deprecated?
**A:** No plans to deprecate. Both are maintained.

### Q: Can I use my existing config files with Hydra?
**A:** Not directly, but conversion is straightforward (see examples above).

### Q: What if I just want to change one parameter?
**A:** Hydra makes this easier:
```bash
python run_experiments_hydra.py model=xgboost_default model.params.n_estimators=200
```

## Step-by-Step Migration Example

Let's migrate a custom experiment step by step.

### Current Setup:

**File:** `configs/custom_experiment.yaml`
```yaml
n_estimators: 250
max_depth: 7
learning_rate: 0.04
```

**Command:**
```bash
python run_experiments.py adult_income XGBoost
```

### Migration Steps:

#### Step 1: Create Hydra Config

Create `conf/model/xgboost_custom_adult.yaml`:
```yaml
# @package _global_
model:
  name: XGBoost
  params:
    n_estimators: 250
    max_depth: 7
    learning_rate: 0.04
    random_state: 42
    eval_metric: logloss
```

#### Step 2: Test Hydra Config

```bash
python run_experiments_hydra.py \
  model=xgboost_custom_adult \
  dataset.name=adult_income
```

#### Step 3: Compare Results

Check that both methods produce similar results.

#### Step 4: Update Documentation

Document the new config for your team.

#### Step 5: (Optional) Remove Old Config

If you're fully migrated, you can remove the old config file.

## Best Practices for Migration

### 1. Test First
Always test Hydra configs on small datasets before full experiments.

### 2. Version Control
Commit both old and new configs during transition period.

### 3. Document Changes
Update your README or team wiki with new commands.

### 4. Gradual Transition
Migrate one model or dataset at a time.

### 5. Keep Backups
Keep old configs until fully confident with new setup.

## Getting Help

- **Quick questions**: See [HYDRA_QUICKSTART.md](HYDRA_QUICKSTART.md)
- **Detailed docs**: See [HYDRA_USAGE.md](HYDRA_USAGE.md)
- **Hydra docs**: Visit [hydra.cc](https://hydra.cc/)
- **Issues**: Open a GitHub issue

## Summary

| Feature | Original Method | Hydra Method |
|---------|----------------|--------------|
| Simplicity | ✓✓✓ | ✓✓ |
| Parameter Overrides | ✗ | ✓✓✓ |
| Parameter Sweeps | Manual | ✓✓✓ |
| Config Tracking | Manual | ✓✓✓ |
| Learning Curve | Easy | Moderate |
| Flexibility | ✓ | ✓✓✓ |

**Bottom line:** 
- Stick with original method for simple, one-off experiments
- Use Hydra for hyperparameter tuning and experimentation
- Both methods are fully supported!
