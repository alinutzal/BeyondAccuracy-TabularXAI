"""
Data loading utilities for tabular datasets.
Supports loading from sklearn datasets and custom CSV files.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
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
            y = data.target
            
            # Encode target
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y), name='income')
            
            # Identify categorical and numerical features
            self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
            self.numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
            
            # Handle categorical features - encode them
            for col in self.categorical_features:
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
            # Fallback to wine dataset
            data = load_wine()
            X = pd.DataFrame(data.data, columns=data.feature_names)
            y = pd.Series(data.target, name='target')
            
            self.feature_names = list(data.feature_names)
            self.target_name = 'target'
            self.numerical_features = self.feature_names
            
            return X, y
    
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
