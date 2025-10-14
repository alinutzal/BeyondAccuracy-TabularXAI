"""Evaluation helpers: per-dataset evaluation strategy, metrics and calibration.

Provides a small Evaluator class that computes metrics (PR-AUC, ROC-AUC, Brier),
supports isotonic calibration on a holdout, and basic threshold selection helpers.
"""
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    precision_recall_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
import hashlib


def run_stratified_cv_evaluation(X, y, model_factory, eval_cfg=None, n_splits=5, random_state=42, dataset_name=None, preprocess_cfg=None):
    """Run stratified K-fold CV using model_factory() to create fresh model instances.

    Returns a dict with per-fold metrics and aggregate means.
    """
    ev = Evaluator(eval_cfg)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_metrics = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        # per-fold preprocessing: standardize breast_cancer (or when requested)
        pc = preprocess_cfg or {}
        do_standardize = False
        if pc.get('standardize_per_fold', False):
            do_standardize = True
        if not do_standardize and dataset_name == 'breast_cancer':
            do_standardize = True
        if do_standardize:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_tr_vals = scaler.fit_transform(X_tr)
            X_val_vals = scaler.transform(X_val)
            X_tr = pd.DataFrame(X_tr_vals, columns=X_tr.columns, index=X_tr.index)
            X_val = pd.DataFrame(X_val_vals, columns=X_val.columns, index=X_val.index)
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        m = model_factory()
        m.train(X_tr, y_tr)
        metrics = ev.evaluate_model(m, X_val, y_val)
        cv_metrics.append(metrics)

    # aggregate
    agg = {}
    if len(cv_metrics) > 0:
        keys = set().union(*[set(d.keys()) for d in cv_metrics])
        for k in keys:
            vals = [d.get(k) for d in cv_metrics if d.get(k) is not None]
            agg[k] = float(np.mean(vals)) if len(vals) > 0 else None

    return {'per_fold': cv_metrics, 'aggregate': agg}


def run_hash_timeish_split(X, y, model_factory, split_cols, threshold=0.5, eval_cfg=None, dataset_name=None, preprocess_cfg=None):
    """Create a deterministic hash-based split on split_cols and evaluate model on test partition.

    Returns a dict with train/test sizes and metrics.
    """
    # check columns
    missing = [c for c in split_cols if c not in X.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing columns for hash split: {missing}")

    def row_hash_fraction(row_values):
        s = '||'.join([str(v) for v in row_values])
        h = hashlib.md5(s.encode('utf8')).hexdigest()
        return int(h, 16) / float(2**128 - 1)

    frac = X[split_cols].apply(lambda r: row_hash_fraction(r.values), axis=1)
    mask_train = frac < threshold
    mask_test = ~mask_train
    if mask_train.sum() < 10 or mask_test.sum() < 10:
        raise ValueError('Insufficient samples in hash-based partitions')

    X_tr, y_tr = X.loc[mask_train].copy(), y.loc[mask_train].copy()
    X_te, y_te = X.loc[mask_test].copy(), y.loc[mask_test].copy()

    # optional preprocessing per dataset
    pc = preprocess_cfg or {}
    do_standardize = False
    if pc.get('standardize_per_fold', False):
        do_standardize = True
    if not do_standardize and dataset_name == 'breast_cancer':
        do_standardize = True
    if do_standardize:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_tr = pd.DataFrame(scaler.fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index)
        X_te = pd.DataFrame(scaler.transform(X_te), columns=X_te.columns, index=X_te.index)
    m = model_factory()
    m.train(X_tr, y_tr)
    ev = Evaluator(eval_cfg)
    metrics = ev.evaluate_model(m, X_te, y_te)

    return {
        'train_size': int(mask_train.sum()),
        'test_size': int(mask_test.sum()),
        'metrics': metrics
    }


class Evaluator:
    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        cfg = cfg or {}
        self.metrics = cfg.get('metrics', ['accuracy', 'precision', 'recall', 'f1', 'pr_auc', 'roc_auc', 'brier_score'])
        self.monitor = cfg.get('monitor', 'accuracy')
        self.mode = cfg.get('mode', 'max')  # 'max' or 'min'
        self.patience = int(cfg.get('patience', 10))
        self.threshold_rule = cfg.get('threshold_rule', None)

    def compute_metrics(self, y_true: np.ndarray, y_proba: np.ndarray, y_pred: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Compute configured metrics.

        y_proba: array-like of probabilities for the positive class (or hard labels if predict_proba not available)
        y_pred: optional array-like of predicted class labels. When provided, label-based metrics
                (accuracy, precision, recall, f1) will be computed from it. Otherwise they will
                be computed by thresholding y_proba at 0.5.
        """
        res = {}
        y_true = np.asarray(y_true)
        y_proba = np.asarray(y_proba)
        y_pred_arr = None if y_pred is None else np.asarray(y_pred)

        # for label-based metrics prefer y_pred when available, otherwise threshold probabilities
        def get_pred_labels():
            if y_pred_arr is not None:
                return y_pred_arr
            # threshold probabilities at 0.5
            try:
                return (y_proba >= 0.5).astype(int)
            except Exception:
                return None

        if 'accuracy' in self.metrics:
            try:
                y_labels = get_pred_labels()
                res['accuracy'] = float(accuracy_score(y_true, y_labels)) if y_labels is not None else None
            except Exception:
                res['accuracy'] = None

        if 'precision' in self.metrics:
            try:
                y_labels = get_pred_labels()
                res['precision'] = float(precision_score(y_true, y_labels)) if y_labels is not None else None
            except Exception:
                res['precision'] = None

        if 'recall' in self.metrics:
            try:
                y_labels = get_pred_labels()
                res['recall'] = float(recall_score(y_true, y_labels)) if y_labels is not None else None
            except Exception:
                res['recall'] = None

        if 'f1' in self.metrics:
            try:
                y_labels = get_pred_labels()
                res['f1'] = float(f1_score(y_true, y_labels)) if y_labels is not None else None
            except Exception:
                res['f1'] = None

        if 'pr_auc' in self.metrics:
            try:
                res['pr_auc'] = float(average_precision_score(y_true, y_proba))
            except Exception:
                res['pr_auc'] = None

        if 'roc_auc' in self.metrics:
            try:
                res['roc_auc'] = float(roc_auc_score(y_true, y_proba))
            except Exception:
                res['roc_auc'] = None

        if 'brier_score' in self.metrics:
            try:
                res['brier_score'] = float(brier_score_loss(y_true, y_proba))
            except Exception:
                res['brier_score'] = None

        # Add threshold-based metrics if requested
        if self.threshold_rule is not None:
            tr = self.threshold_rule
            try:
                if tr.get('type') == 'precision_at_recall':
                    target_recall = float(tr.get('recall', 0.8))
                    prec, rec, thresh = precision_recall_curve(y_true, y_proba)
                    # find first threshold with recall >= target_recall
                    idx = np.where(rec >= target_recall)[0]
                    if len(idx) > 0:
                        ix = idx[0]
                        res['precision_at_recall'] = float(prec[ix])
                    else:
                        res['precision_at_recall'] = 0.0
                elif tr.get('type') == 'f1_max':
                    prec, rec, thresh = precision_recall_curve(y_true, y_proba)
                    f1s = (2 * prec * rec) / (prec + rec + 1e-12)
                    res['f1_max'] = float(np.nanmax(f1s))
            except Exception:
                # swallow metric computation errors and continue
                pass

        return res

    def evaluate_model(self, model, X, y, proba_transform=None) -> Dict[str, Optional[float]]:
        """Compute configured metrics for a fitted model on X/y.

        proba_transform: optional callable to map predicted probabilities (e.g., calibration)
        """
        # ask model for predict_proba; fall back to predict if not available
        y_pred = None
        try:
            y_proba = model.predict_proba(X)
            # if model returns multi-class probabilities, pick positive class if binary
            if y_proba.ndim == 2 and y_proba.shape[1] > 1:
                # assume binary positive is last column
                y_proba = y_proba[:, 1]
            # also try to get hard predictions if available
            try:
                y_pred = model.predict(X)
            except Exception:
                y_pred = None
        except Exception:
            # fallback to using predict and treating as hard labels
            try:
                y_pred = model.predict(X)
                y_proba = np.asarray(y_pred, dtype=float)
            except Exception:
                raise RuntimeError('Model does not implement predict_proba or predict correctly')

        if proba_transform is not None:
            y_proba = proba_transform(y_proba)

        return self.compute_metrics(y, y_proba, y_pred=y_pred)

    def fit_isotonic(self, y_holdout: np.ndarray, proba_holdout: np.ndarray) -> IsotonicRegression:
        """Fit an isotonic regression mapping p -> calibrated_p on holdout probabilities."""
        ir = IsotonicRegression(out_of_bounds='clip')
        ir.fit(proba_holdout, y_holdout)
        return ir

    def apply_calibration(self, ir: IsotonicRegression, proba: np.ndarray) -> np.ndarray:
        return ir.transform(proba)
