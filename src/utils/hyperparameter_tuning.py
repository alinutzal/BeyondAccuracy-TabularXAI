"""Hyperparameter tuning utilities using scikit-learn search strategies."""

from typing import Any, Dict, Optional
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import numpy as np


def _grid_size(param_grid: Dict[str, Any]) -> int:
    """Estimate number of candidate parameter combinations for a parameter grid."""
    size = 1
    for v in param_grid.values():
        if hasattr(v, '__len__'):
            size *= max(1, len(v))
        else:
            size *= 1
    return size


def tune_model(
    estimator,
    X,
    y,
    param_grid: Dict[str, Any],
    scoring: Optional[str] = 'roc_auc',
    cv: int = 3,
    n_iter: Optional[int] = None,
    n_jobs: int = -1,
    random_state: Optional[int] = 42,
    refit: bool = True,
    verbose: int = 1,
    fit_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Tune hyperparameters using GridSearchCV or RandomizedSearchCV.

    Notes:
    - For large parameter grids prefer `n_iter` (RandomizedSearchCV) to reduce runtime.
    - Use `fit_params` to pass estimator-specific params (e.g. early_stopping_rounds via `eval_set`).

    Args:
        estimator: scikit-learn compatible estimator
        X: features (DataFrame or array)
        y: target
        param_grid: dict mapping parameter names to lists of values
        scoring: scoring metric for cross-validation
        cv: number of CV folds
        n_iter: if provided (>0), use RandomizedSearchCV with this many iterations;
                otherwise GridSearchCV is used
        n_jobs: parallel jobs
        random_state: random seed for RandomizedSearchCV sampling
        refit: whether to refit best estimator on whole data
        verbose: verbosity level
        fit_params: additional kwargs to pass to estimator.fit

    Returns:
        dict with keys: 'best_estimator', 'best_params', 'best_score', 'cv_results'
    """
    # Warn if the grid is large
    try:
        size = _grid_size(param_grid)
        if n_iter is None and size > 50:
            print(f"⚠️ Parameter grid contains {size} combinations — consider using `n_iter` (RandomizedSearchCV) or reducing ranges")
    except Exception:
        size = None

    if n_iter is not None and int(n_iter) > 0:
        searcher = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_grid,
            n_iter=int(n_iter),
            scoring=scoring,
            cv=cv,
            random_state=random_state,
            n_jobs=n_jobs,
            refit=refit,
            verbose=verbose,
        )
    else:
        searcher = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            refit=refit,
            verbose=verbose,
        )

    try:
        if fit_params:
            searcher.fit(X, y, **fit_params)
        else:
            searcher.fit(X, y)
    except KeyboardInterrupt:
        # allow graceful exit and return partial results if available
        print("\nInterrupted by user during hyperparameter search — returning partial results if any.")
        best_est = getattr(searcher, 'best_estimator_', None)
        best_params = getattr(searcher, 'best_params_', None)
        best_score = getattr(searcher, 'best_score_', None)
        cv_results = getattr(searcher, 'cv_results_', None)
        return {
            'best_estimator': best_est,
            'best_params': best_params,
            'best_score': best_score,
            'cv_results': cv_results,
            'searcher': searcher,
            'interrupted': True,
        }

    best_est = getattr(searcher, 'best_estimator_', None)
    best_params = getattr(searcher, 'best_params_', None)
    best_score = getattr(searcher, 'best_score_', None)
    cv_results = getattr(searcher, 'cv_results_', None)

    # Ensure numpy types are converted to python natives for JSON-compatibility
    if isinstance(best_score, (np.floating, np.integer)):
        best_score = best_score.item()

    return {
        'best_estimator': best_est,
        'best_params': best_params,
        'best_score': best_score,
        'cv_results': cv_results,
        'searcher': searcher,
        'interrupted': False,
    }