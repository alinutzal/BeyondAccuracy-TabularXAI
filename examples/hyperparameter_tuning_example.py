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

def main():
    loader = DataLoader('breast_cancer', random_state=42)
    X, y = loader.load_data()
    data = loader.prepare_data(X, y, test_size=0.2)

    X_train = data['X_train']
    y_train = data['y_train']

    print('Running hyperparameter tuning for XGBoost (Grid search small grid)')
    xgb_model = XGBoostClassifier(device="cuda",random_state=42)
    # Reduced (safe) grid — the previous grid was huge and caused long-running searches
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

    #res = xgb_model.tune_hyperparameters(X_train, y_train, param_grid, scoring='roc_auc', cv=3, n_iter=None, verbose=0)
    res = RandomizedSearchCV(
        xgb_model.model,
        param_distributions=param_grid,
        n_iter=50,
        cv=3,
        scoring="neg_log_loss",
        n_jobs=10,
        n_jobs=1,
        verbose=2,
    ).fit(X_train, y_train)

    print('Best params:', res['best_params'])
    print('Best score:', res['best_score'])

    print('\nRunning a small randomized search for LightGBM')
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
    # Limit parallel workers to avoid nested parallelism issues and reduce iterations for a quick run
    res2 = lgb_model.tune_hyperparameters(
        X_train,
        y_train,
        lgb_param_dist,
        scoring='roc_auc',
        cv=3,
        n_iter=3,
        n_jobs=1,
        verbose=1
    )
    print('Best params (LGB):', res2['best_params'])
    print('Best score (LGB):', res2['best_score'])


    print('\nRunning a small grid search for CatBoost')
    cb_model = CatBoostModel()
    cb_param_grid = {
        'iterations': [50, 100],
        'depth': [4, 6],
        'learning_rate': [0.01, 0.1]
    }
    res3 = cb_model.tune_hyperparameters(
        X_train,
        y_train,
        cb_param_grid,
        scoring='roc_auc',
        cv=3,
        n_iter=None,
        n_jobs=1,
        verbose=0
    )
    print('Best params (CatBoost):', res3['best_params'])
    print('Best score (CatBoost):', res3['best_score'])



if __name__ == '__main__':
    main()
