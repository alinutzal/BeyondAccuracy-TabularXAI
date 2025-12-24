"""Example: run hyperparameter tuning for Gradient Boosting on various datasets."""
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
# Ensure both the repo root and src/ are on sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.join(ROOT, 'src')
sys.path.insert(0, ROOT)
sys.path.insert(0, SRC)

from src.models.gradient_boosting import GradientBoostingClassifier
from src.utils.data_loader import DataLoader
from sklearn.model_selection import RandomizedSearchCV
import csv
import json
import numpy as np

# Lock for thread-safe list operations
results_lock = Lock()

# Datasets to iterate over
DATASETS = [
    'breast_cancer',
    'diabetes',
    'in-vehicle_coupon',
    'bank_marketing',
    'adult_income',
    'credit_card_fraud'
]

# Set to >0 to use Optuna for Gradient Boosting (will use optuna_n_trials trials)
OPTUNA_TRIALS = 50


def _extract_search_results(res):
    """Return (best_params, best_score, cv_results) for either a dict result or a fitted scikit-learn searcher."""
    # Dict result from utils.hyperparameter_tuning.tune_model
    if isinstance(res, dict):
        return res.get('best_params'), res.get('best_score'), res.get('cv_results', {})
    # scikit-learn GridSearchCV/RandomizedSearchCV objects
    best_params = getattr(res, 'best_params_', None)
    best_score = getattr(res, 'best_score_', None)
    cv_results = getattr(res, 'cv_results_', {})
    return best_params, best_score, cv_results


def run_tuning_for_dataset(dataset_name: str, results_list: list, gb_iter: int = 50, optuna_iter: int = 0):
    """Run tuning for Gradient Boosting on a single dataset and append results to results_list."""
    print(f"\n=== Dataset: {dataset_name} ===")
    try:
        loader = DataLoader(dataset_name, random_state=42)
        X, y = loader.load_data()
        if X is None or y is None:
            print(f"Skipping dataset {dataset_name}: failed to load")
            return
        data = loader.prepare_data(X, y, test_size=0.2)
        X_train = data['X_train']
        y_train = data['y_train']

        # Gradient Boosting with multi-metric scoring
        print('Running Gradient Boosting randomized search with multi-metric scoring')
        gb_model = GradientBoostingClassifier(random_state=42)
        param_grid = {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [2, 3, 4],
            "subsample": [0.6, 0.8, 1.0],
            "min_samples_leaf": [1, 10, 20],
            "max_features": ["sqrt", "log2", None],
        }
        
        scoring_opt = "roc_auc"
        # Multi-metric scoring
        scoring = {
            "roc_auc": "roc_auc",
            "pr_auc": "average_precision",
            "f1": "f1",
            "brier": "neg_brier_score",
        }
        
        try:
            # Use Optuna if requested, otherwise fall back to RandomizedSearch
            if optuna_iter > 0:
                res = gb_model.tune_hyperparameters(
                    X_train,
                    y_train,
                    param_grid,
                    scoring=scoring_opt,
                    cv=3,
                    n_iter=None,
                    n_jobs=1,
                    verbose=1,
                    optuna_n_trials=optuna_iter,
                )
                best_params, best_score, cv_results = _extract_search_results(res)
            else:
                searcher = RandomizedSearchCV(
                    gb_model.model,
                    param_distributions=param_grid,
                    n_iter=gb_iter,
                    cv=3,
                    scoring=scoring,
                    refit="pr_auc",  # Refit on PR-AUC score
                    n_jobs=-1,
                    verbose=1,
                ).fit(X_train, y_train)
                best_params, best_score, cv_results = _extract_search_results(searcher)
            
            print('Best params (GB):', best_params)
            print('Best score (PR-AUC refit):', best_score)
            
            # Extract all metric scores
            all_scores = {}
            if cv_results:
                for metric in scoring.keys():
                    mean_key = f"mean_test_{metric}"
                    std_key = f"std_test_{metric}"
                    if mean_key in cv_results:
                        all_scores[metric] = {
                            'mean': cv_results[mean_key][searcher.best_index_] if hasattr(searcher, 'best_index_') else cv_results[mean_key][0],
                            'std': cv_results[std_key][searcher.best_index_] if hasattr(searcher, 'best_index_') else cv_results[std_key][0]
                        }
            
            print('All metric scores:', all_scores)
            
            # Thread-safe append
            with results_lock:
                results_list.append({
                    'dataset': dataset_name, 
                    'model': 'gradient_boosting', 
                    'best_score': best_score,
                    'all_scores': all_scores,
                    'best_params': best_params
                })
        except Exception as e:
            print(f"Gradient Boosting tuning failed on {dataset_name}: {e}")
    except Exception as e:
        print(f"Error processing dataset {dataset_name}: {e}")


def main(num_threads: int = 4):
    """Run tuning in parallel using threads.
    
    Args:
        num_threads: Number of threads to use for parallel dataset tuning (default: 4)
    """
    results = []
    
    # Use ThreadPoolExecutor for parallel dataset tuning
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {
            executor.submit(run_tuning_for_dataset, ds, results, 50, OPTUNA_TRIALS): ds 
            for ds in DATASETS
        }
        
        # Monitor progress
        completed = 0
        for future in as_completed(futures):
            completed += 1
            dataset = futures[future]
            try:
                future.result()
                print(f"\n[{completed}/{len(DATASETS)}] Completed: {dataset}")
            except Exception as e:
                print(f"\n[{completed}/{len(DATASETS)}] Error in {dataset}: {e}")

    # Save results
    out_dir = os.path.join(ROOT, 'results')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'hyperparameter_tuning_gb_summary.csv')
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['dataset', 'model', 'best_score', 'all_scores', 'best_params'])
        writer.writeheader()
        for r in results:
            # Convert best_params dict to JSON string for CSV cell
            row = r.copy()
            try:
                row['best_params'] = json.dumps(_jsonify(r.get('best_params')))
                row['all_scores'] = json.dumps(_jsonify(r.get('all_scores', {})))
            except Exception:
                row['best_params'] = json.dumps(r.get('best_params'), default=str)
                row['all_scores'] = json.dumps(r.get('all_scores', {}), default=str)
            writer.writerow(row)

    print(f"\nSaved tuning summary to: {out_path}")
    # Also write a consolidated JSON (dataset -> model -> best params/score)
    summary = {}
    for r in results:
        ds = r['dataset']
        model = r['model']
        best_score = r.get('best_score')
        all_scores = r.get('all_scores', {})
        best_params = r.get('best_params')
        if ds not in summary:
            summary[ds] = {}
        summary[ds][model] = {
            'best_score': best_score,
            'all_scores': _jsonify(all_scores),
            'best_params': _jsonify(best_params)
        }

    json_out = os.path.join(out_dir, 'hyperparameter_tuning_gb_summary.json')
    with open(json_out, 'w') as f:
        json.dump(summary, f, indent=2)

    # Per-dataset files
    for ds, info in summary.items():
        per_path = os.path.join(out_dir, f"{ds}_gradient_boosting_best_params.json")
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
    import argparse
    
    parser = argparse.ArgumentParser(description='Run hyperparameter tuning for Gradient Boosting')
    parser.add_argument('--threads', type=int, default=16, help='Number of threads (default: 16)')
    args = parser.parse_args()
    
    print(f"Starting hyperparameter tuning with {args.threads} threads...")
    main(num_threads=args.threads)
