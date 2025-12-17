"""Example: run hyperparameter tuning for an XGBoost model on breast_cancer dataset."""
import sys
import os
# Ensure both the repo root and src/ are on sys.path so imports like `src.models` and `models` work
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.join(ROOT, 'src')
sys.path.insert(0, ROOT)
sys.path.insert(0, SRC)

from utils.data_loader import DataLoader
from models import XGBoostClassifier, LightGBMClassifier, CatBoostModel
from sklearn.model_selection import RandomizedSearchCV
import csv
import json
import numpy as np

# Datasets to iterate over
DATASETS = [
    'breast_cancer',
    'diabetes',
    'in-vehicle_coupon',
    'bank_marketing',
    'adult_income',
    'credit_card_fraud'
]

# Set to >0 to use Optuna for XGBoost/LightGBM (will use optuna_n_trials trials)
OPTUNA_TRIALS = 50


def _extract_search_results(res):
    """Return (best_params, best_score) for either a dict result or a fitted scikit-learn searcher."""
    # Dict result from utils.hyperparameter_tuning.tune_model
    if isinstance(res, dict):
        return res.get('best_params'), res.get('best_score')
    # scikit-learn GridSearchCV/RandomizedSearchCV objects
    best_params = getattr(res, 'best_params_', None)
    best_score = getattr(res, 'best_score_', None)
    return best_params, best_score


def run_tuning_for_dataset(dataset_name: str, results_list: list, xgb_iter: int = 50, lgb_iter: int = 10, optuna_iter: int = 0):
    """Run tuning for XGBoost, LightGBM and CatBoost on a single dataset and append results to results_list."""
    print(f"\n=== Dataset: {dataset_name} ===")
    loader = DataLoader(dataset_name, random_state=42)
    X, y = loader.load_data()
    if X is None or y is None:
        print(f"Skipping dataset {dataset_name}: failed to load")
        return
    data = loader.prepare_data(X, y, test_size=0.2)
    X_train = data['X_train']
    y_train = data['y_train']

    # XGBoost
    print('Running XGBoost randomized search')
    xgb_model = XGBoostClassifier(random_state=42)
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [4, 6, 8],
        'min_child_weight': [1, 5, 10],
        'subsample': [0.7, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [1, 5, 10],
        'n_estimators': [500, 1000, 2000]
    }
    optuna_iter = 50
    try:
        # Use Optuna if requested, otherwise fall back to RandomizedSearch
        if optuna_iter > 0:
            res = xgb_model.tune_hyperparameters(
                X_train,
                y_train,
                param_grid,
                scoring="neg_log_loss",
                cv=3,
                n_iter=None,
                n_jobs=1,
                verbose=1,
                optuna_n_trials=optuna_iter,
            )
            best_params, best_score = _extract_search_results(res)
        else:
            searcher = RandomizedSearchCV(
                xgb_model.model,
                param_distributions=param_grid,
                n_iter=xgb_iter,
                cv=3,
                scoring="neg_log_loss",
                n_jobs=-1,
                verbose=1,
            ).fit(X_train, y_train)
            best_params, best_score = _extract_search_results(searcher)
        print('Best params (XGB):', best_params)
        print('Best score (XGB):', best_score)
        results_list.append({'dataset': dataset_name, 'model': 'xgboost', 'best_score': best_score, 'best_params': best_params})
    except Exception as e:
        print(f"XGBoost tuning failed on {dataset_name}: {e}")

    # LightGBM
    print('\nRunning LightGBM tuning')
    lgb_model = LightGBMClassifier(random_state=42)
    lgb_param_dist = {
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [31, 63, 127],
        'min_data_in_leaf': [10, 20, 50],
        'feature_fraction': [0.7, 0.9],
        'bagging_fraction': [0.7, 0.9],
        'bagging_freq': [1],
        'reg_alpha': [0, 0.1],
        'reg_lambda': [1, 5],
        'n_estimators': [500, 1000, 2000]
    }
    try:
        # Use Optuna for LightGBM if requested
        if optuna_iter > 0:
            res2 = lgb_model.tune_hyperparameters(
                X_train,
                y_train,
                lgb_param_dist,
                scoring='roc_auc',
                cv=3,
                n_iter=None,
                n_jobs=1,
                verbose=1,
                optuna_n_trials=optuna_iter,
            )
        else:
            res2 = lgb_model.tune_hyperparameters(
                X_train,
                y_train,
                lgb_param_dist,
                scoring='roc_auc',
                cv=3,
                n_iter=lgb_iter,
                n_jobs=1,
                verbose=1
            )
        best_params_lgb, best_score_lgb = _extract_search_results(res2)
        print('Best params (LGB):', best_params_lgb)
        print('Best score (LGB):', best_score_lgb)
        results_list.append({'dataset': dataset_name, 'model': 'lightgbm', 'best_score': best_score_lgb, 'best_params': best_params_lgb})
    except Exception as e:
        print(f"LightGBM tuning failed on {dataset_name}: {e}")

    # CatBoost (optional)
    print('\nRunning CatBoost tuning')
    try:
        cb_model = CatBoostModel()
        cb_param_grid = {
            "depth": [4, 5, 6, 7, 8, 9, 10],
            "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3],
            "l2_leaf_reg": [1, 3, 5, 10, 20, 50, 100],
            "min_data_in_leaf": [1, 5, 10, 20, 50, 100, 200],
            "rsm": [0.6, 0.8, 1.0],
            "random_strength": [0, 0.5, 1, 2, 5, 10],
            "bagging_temperature": [0, 0.5, 1, 2, 5],  # works with Bayesian bootstrap
        }
        res3 = cb_model.tune_hyperparameters(
            X_train,
            y_train,
            cb_param_grid,
            scoring='roc_auc',
            cv=3,
            n_iter=50,
            n_jobs=1,
            verbose=2
        )
        best_params_cb, best_score_cb = _extract_search_results(res3)
        print('Best params (CatBoost):', best_params_cb)
        print('Best score (CatBoost):', best_score_cb)
        results_list.append({'dataset': dataset_name, 'model': 'catboost', 'best_score': best_score_cb, 'best_params': best_params_cb})
    except ImportError:
        print('CatBoost not installed, skipping')
    except Exception as e:
        print(f"CatBoost tuning failed on {dataset_name}: {e}")


def main():
    results = []
    for ds in DATASETS:
        run_tuning_for_dataset(ds, results, xgb_iter=50, lgb_iter=10, optuna_iter=OPTUNA_TRIALS)

    # Save a simple CSV summary
    out_dir = os.path.join(ROOT, 'results')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'hyperparameter_tuning_summary.csv')
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['dataset', 'model', 'best_score', 'best_params'])
        writer.writeheader()
        for r in results:
            # Convert best_params dict to JSON string for CSV cell
            row = r.copy()
            try:
                row['best_params'] = json.dumps(_jsonify(r.get('best_params')))
            except Exception:
                row['best_params'] = json.dumps(r.get('best_params'), default=str)
            writer.writerow(row)

    print(f"\nSaved tuning summary to: {out_path}")
    # Also write a consolidated JSON (dataset -> model -> best params/score) and per-dataset files
    summary = {}
    for r in results:
        ds = r['dataset']
        model = r['model']
        best_score = r.get('best_score')
        best_params = r.get('best_params')
        if ds not in summary:
            summary[ds] = {}
        summary[ds][model] = {
            'best_score': best_score,
            'best_params': _jsonify(best_params)
        }

    json_out = os.path.join(out_dir, 'hyperparameter_tuning_summary.json')
    with open(json_out, 'w') as f:
        json.dump(summary, f, indent=2)

    # Per-dataset files
    for ds, info in summary.items():
        per_path = os.path.join(out_dir, f"{ds}_{model}_best_params.json")
        with open(per_path, 'w') as pf:
            json.dump(info, pf, indent=2)

    print(f"Saved JSON summary to: {json_out} and per-dataset files in {out_dir}")


def _jsonify(obj):
    """Convert objects (numpy types, arrays) to JSON-serializable Python natives recursively."""
    if obj is None:
        return None
    # numpy scalar
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    # numpy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # dict
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    # list/tuple
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    # other: try to convert
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


if __name__ == '__main__':
    main()
