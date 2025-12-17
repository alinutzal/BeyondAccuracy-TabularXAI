"""
Data loading utilities for tabular datasets.
Supports loading from sklearn datasets and custom CSV files.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, QuantileTransformer
from typing import Tuple, Dict, Any
import os


class DataLoader:
    """Load and preprocess tabular datasets for experiments."""
    
    def __init__(self, dataset_name: str, random_state: int = 42):
        """
        Initialize DataLoader.
        
        Args:
            dataset_name: Name of the dataset to load
            random_state: Random seed for reproducibility
        """
        self.dataset_name = dataset_name
        self.random_state = random_state
        self.feature_names = None
        self.target_name = None
        self.categorical_features = []
        self.numerical_features = []
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load dataset based on name.
        
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        if self.dataset_name == "breast_cancer":
            return self._load_breast_cancer()
        elif self.dataset_name == "adult_income":
            return self._load_adult_income()
        elif self.dataset_name == "bank_marketing":
            return self._load_bank_marketing()
        elif self.dataset_name == "diabetes":
            return self._load_diabetes()
        elif self.dataset_name == "in-vehicle_coupon":
            return self._load_in_vehicle_coupon()
        elif self.dataset_name == "credit_card_fraud":
            return self._load_credit_card_fraud()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def _load_breast_cancer(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load breast cancer dataset."""
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name='target')
        
        self.feature_names = list(data.feature_names)
        self.target_name = 'target'
        self.numerical_features = self.feature_names
        
        return X, y
    
    def _load_adult_income(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load adult income dataset from OpenML."""
        try:
            data = fetch_openml('adult', version=2, as_frame=True, parser='auto')
            X = data.data
            
            # Ensure we own the DataFrame to avoid SettingWithCopyWarning when mutating
            X = X.copy()
            
            y = data.target
            
            # Encode target
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y), name='income')
            
            # Identify categorical and numerical features
            self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
            self.numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
            
            # Handle categorical features - apply rare category bucketing and encode
            for col in self.categorical_features:
                # Calculate value counts
                value_counts = X[col].value_counts()
                # Find rare categories (< 1% of data)
                rare_threshold = len(X) * 0.01
                rare_categories = value_counts[value_counts < rare_threshold].index.tolist()
                
                # Replace rare categories with 'other'
                if rare_categories:
                    X[col] = X[col].apply(lambda x: 'other' if x in rare_categories else x)
                
                # Encode categorical features
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            
            self.feature_names = X.columns.tolist()
            self.target_name = 'income'
            
            return X, y
        except Exception as e:
            print(f"Error loading adult dataset: {e}")
            # Fallback to simpler dataset
            return self._load_breast_cancer()
    
    def _load_bank_marketing(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load bank marketing dataset from OpenML."""
        try:
            data = fetch_openml('bank-marketing', version=1, as_frame=True, parser='auto')
            X = data.data
            
            # Make an explicit copy before modifying
            X = X.copy()
            
            y = data.target
            
            # Encode target
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y), name='subscription')
            
            # Identify categorical and numerical features
            self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
            self.numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
            
            # Handle categorical features - encode them
            for col in self.categorical_features:
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            
            self.feature_names = X.columns.tolist()
            self.target_name = 'subscription'
            
            return X, y
        except Exception as e:
            print(f"Error loading bank marketing dataset: {e}")
            
    def _try_local_csv(self, filename_candidates):
        """Try to find and load a local CSV from the repo data/ directory.

        Returns (X, y, target_name) or (None, None, None) if not found.
        """
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
        for fname in filename_candidates:
            p = os.path.join(data_dir, fname)
            if os.path.exists(p):
                try:
                    df = pd.read_csv(p)
                except Exception:
                    continue

                # Try to guess the target column
                for cand in ['target', 'Target', 'class', 'Class', 'y', 'Y', 'label', 'Label', 'fraud', 'Fraud', 'Outcome', 'outcome']:
                    if cand in df.columns:
                        y = df[cand]
                        X = df.drop(columns=[cand])
                        return X, y, cand

                # pick a binary column if available
                for col in df.columns:
                    if df[col].nunique() == 2:
                        y = df[col]
                        X = df.drop(columns=[col])
                        return X, y, col

                # fallback: use last column as target
                y = df.iloc[:, -1]
                X = df.iloc[:, :-1]
                return X, y, df.columns[-1]

        return None, None, None

    def _try_openml(self, name):
        try:
            data = fetch_openml(name, as_frame=True, parser='auto')
            X = data.data
            y = data.target
            # If target is not numeric, label-encode
            if y.dtype == object or y.dtype.name == 'category':
                try:
                    y = pd.Series(LabelEncoder().fit_transform(y), name='target')
                except Exception:
                    y = pd.Series(pd.Categorical(y).codes, name='target')
            else:
                y = pd.Series(y, name='target')
            return X, y
        except Exception:
            return None, None

    def _load_diabetes(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load a diabetes classification dataset.

        Tries local CSVs first, then OpenML, otherwise falls back to breast_cancer.
        """
        # Local filenames to try
        local_candidates = ['diabetes.csv', 'diabetes_data.csv', 'pima-indians-diabetes.csv', 'pima_diabetes.csv']
        X, y, target_name = self._try_local_csv(local_candidates)
        if X is not None:
            self.feature_names = X.columns.tolist()
            self.target_name = target_name
            self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
            self.numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
            return X, y

        # Try OpenML common names
        for name in ['diabetes']: #, 'diabetes_012_patient', 'pima-indians-diabetes']:
            Xo, yo = self._try_openml(name)
            if Xo is not None:
                self.feature_names = Xo.columns.tolist()
                self.target_name = 'target'
                self.categorical_features = Xo.select_dtypes(include=['object', 'category']).columns.tolist()
                self.numerical_features = Xo.select_dtypes(include=[np.number]).columns.tolist()
                return Xo, yo

        # fallback
        print('Warning: could not load diabetes dataset, falling back to breast_cancer')

    def _load_in_vehicle_coupon(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load the in-vehicle coupon dataset (UCI style).

        Tries local CSV first, then OpenML, otherwise falls back to breast_cancer.
        """
        local_candidates = ['in-vehicle-coupon.csv', 'in_vehicle_coupon.csv', 'in-vehicle_coupon.csv']
        X, y, target_name = self._try_local_csv(local_candidates)
        if X is not None:
            self.feature_names = X.columns.tolist()
            self.target_name = target_name
            self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
            self.numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
            return X, y

        for name in ['in-vehicle coupon', 'in-vehicle-coupon', 'invehiclecoupon']:
            Xo, yo = self._try_openml(name)
            if Xo is not None:
                self.feature_names = Xo.columns.tolist()
                self.target_name = 'target'
                self.categorical_features = Xo.select_dtypes(include=['object', 'category']).columns.tolist()
                self.numerical_features = Xo.select_dtypes(include=[np.number]).columns.tolist()
                return Xo, yo

        print('Warning: could not load in-vehicle coupon dataset, falling back to breast_cancer')
        return self._load_breast_cancer()

    def _load_credit_card_fraud(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load credit card fraud dataset (Kaggle-style creditcard.csv).

        Tries local CSV first, then OpenML, otherwise falls back to breast_cancer.
        """
        local_candidates = ['credit_card_fraud.csv', 'creditcard.csv', 'creditcard_fraud.csv', 'credit_card.csv']
        X, y, target_name = self._try_local_csv(local_candidates)
        if X is not None:
            self.feature_names = X.columns.tolist()
            self.target_name = target_name
            self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
            self.numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
            return X, y

        for name in ['creditcard', 'credit_card_fraud', 'creditcardfraud']:
            Xo, yo = self._try_openml(name)
            if Xo is not None:
                self.feature_names = Xo.columns.tolist()
                self.target_name = 'target'
                self.categorical_features = Xo.select_dtypes(include=['object', 'category']).columns.tolist()
                self.numerical_features = Xo.select_dtypes(include=[np.number]).columns.tolist()
                return Xo, yo

        print('Warning: could not load credit card fraud dataset, falling back to breast_cancer')
        return self._load_breast_cancer()
    
    def prepare_data(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        test_size: float = 0.2,
        scale_features: bool = True
    ) -> Dict[str, Any]:
        """
        Prepare data for training: split and optionally scale.
        
        Args:
            X: Features DataFrame
            y: Target Series
            test_size: Proportion of test set
            scale_features: Whether to standardize features
            
        Returns:
            Dictionary containing train/test splits and scaler
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        scaler = None
        if scale_features:
            # Use QuantileTransformer with normal distribution for adult_income dataset
            if self.dataset_name == 'adult_income' and self.numerical_features:
                scaler = QuantileTransformer(output_distribution='normal', random_state=self.random_state)
            else:
                scaler = StandardScaler()
            
            X_train = pd.DataFrame(
                scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            X_test = pd.DataFrame(
                scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'scaler': scaler,
            'feature_names': self.feature_names,
            'target_name': self.target_name
        }
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the loaded dataset."""
        return {
            'dataset_name': self.dataset_name,
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'categorical_features': self.categorical_features,
            'numerical_features': self.numerical_features,
            'num_features': len(self.feature_names) if self.feature_names else 0
        }
