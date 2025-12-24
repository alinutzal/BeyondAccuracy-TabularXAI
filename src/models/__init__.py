"""Model implementations for tabular classification."""

from .gradient_boosting import XGBoostClassifier, LightGBMClassifier, GradientBoostingClassifier, CatBoostModel
from .deep_learning import MLPClassifier, TransformerClassifier, MLPDistillationClassifier, TransformerDistillationClassifier
from .tab_pfn import TabPFNClassifier

# Backwards compatible alias expected elsewhere in repo
CatBoostClassifier = CatBoostModel

__all__ = [
    'XGBoostClassifier',
    'LightGBMClassifier',
    'GradientBoostingClassifier',
    'CatBoostModel',
    'CatBoostClassifier',
    'TabPFNClassifier',
    'MLPClassifier',
    'TransformerClassifier',
    'MLPDistillationClassifier',
    'TransformerDistillationClassifier'
]
