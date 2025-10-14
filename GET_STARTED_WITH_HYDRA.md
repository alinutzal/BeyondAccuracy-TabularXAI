# Get Started with Hydra in 5 Minutes

This is the **fastest** way to start using Hydra with BeyondAccuracy-TabularXAI.

## Step 1: Install Dependencies (if needed)

```bash
pip install hydra-core omegaconf
```

## Step 2: Run Your First Hydra Experiment

```bash
cd src
python run_experiments_hydra.py
```

That's it! This runs XGBoost with default settings on the breast_cancer dataset.

## Step 3: Try Different Configurations

### Choose a different model configuration:
```bash
python run_experiments_hydra.py model=xgboost_shallow     # Better interpretability
python run_experiments_hydra.py model=lightgbm_deep       # Better accuracy
python run_experiments_hydra.py model=mlp_default         # Deep learning MLP
python run_experiments_hydra.py model=transformer_default # Deep learning Transformer
```

### Choose a different dataset:
```bash
python run_experiments_hydra.py dataset.name=adult_income
```

### Combine both:
```bash
python run_experiments_hydra.py model=lightgbm_shallow dataset.name=adult_income
```

## Step 4: Override a Single Parameter

Want to try 200 trees instead of 100?

```bash
python run_experiments_hydra.py model.params.n_estimators=200
```

Want to change the learning rate?

```bash
python run_experiments_hydra.py model.params.learning_rate=0.05
```

## Step 5: Run a Parameter Sweep

Try multiple learning rates automatically:

```bash
python run_experiments_hydra.py -m model.params.learning_rate=0.01,0.05,0.1
```

This runs **3 experiments** with different learning rates!

## What Just Happened?

1. Hydra loaded configuration from `conf/config.yaml`
2. It created a model based on your selection
3. Trained and evaluated it
4. Saved results with the configuration used
5. Generated SHAP explanations

## Where Are the Results?

Check the `results/` directory. Each experiment creates a folder with:
- `results.json` - Performance metrics
- `hydra_config.yaml` - Configuration used (for reproducibility!)
- `shap_summary.png` - SHAP visualizations
- And more...

## Quick Reference

### Available Configurations

**XGBoost:**
- `xgboost_default` - Balanced
- `xgboost_shallow` - Interpretable (few deep trees)
- `xgboost_deep` - Accurate (more complex)
- `xgboost_regularized` - Robust (prevents overfitting)

**LightGBM:**
- `lightgbm_default` - Balanced
- `lightgbm_shallow` - Interpretable
- `lightgbm_deep` - Accurate
- `lightgbm_regularized` - Robust

**GradientBoosting (sklearn):**
- `gradient_boosting_default` - Standard
- `gradient_boosting_shallow` - Very interpretable
- `gradient_boosting_deep` - Complex patterns

### Available Datasets
- `breast_cancer`
- `adult_income`
- `bank_marketing`

## Common Tasks

### Compare shallow vs deep trees:
```bash
python run_experiments_hydra.py model=xgboost_shallow
python run_experiments_hydra.py model=xgboost_deep
```

### Find best learning rate:
```bash
python run_experiments_hydra.py -m model.params.learning_rate=0.001,0.01,0.05,0.1,0.2
```

### Test on all datasets:
```bash
python run_experiments_hydra.py -m dataset.name=breast_cancer,adult_income,bank_marketing
```

### Quick grid search:
```bash
python run_experiments_hydra.py -m \
  model.params.n_estimators=50,100,200 \
  model.params.max_depth=3,6,9
```

## Tips

1. **Start simple** - Run with defaults first
2. **One change at a time** - Override one parameter to see its effect  
3. **Use sweeps** - Let Hydra run multiple experiments for you
4. **Check results** - Look at the generated folders to understand output

## What's Next?

- 📖 Read [HYDRA_QUICKSTART.md](HYDRA_QUICKSTART.md) for more examples
- 📚 Check [HYDRA_USAGE.md](HYDRA_USAGE.md) for detailed documentation
- 🔄 See [HYDRA_MIGRATION.md](HYDRA_MIGRATION.md) if migrating from old method
- 💻 Try [examples/hydra_example.py](examples/hydra_example.py) for code examples

## Original Method Still Works!

Don't worry - the original way still works perfectly:

```bash
python run_experiments.py breast_cancer XGBoost
```

Use Hydra when you want easier parameter management and sweeps!

## Questions?

Check the documentation or open an issue on GitHub.

---

**That's it! You're now using Hydra for configuration management. 🎉**

Happy experimenting! 🚀
