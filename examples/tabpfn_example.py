"""
Example script demonstrating TabPFN usage.

TabPFN (Prior-Fitted Networks) is a transformer-based model that performs
in-context learning on small tabular datasets. It works best with:
- Up to 10,000 training samples
- Up to 100 features
- Binary or multi-class classification

This example shows how to use TabPFN with the BeyondAccuracy-TabularXAI framework.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.data_loader import DataLoader
from models import TabPFNClassifier

def main():
    print("="*80)
    print("TabPFN Example - Small Tabular Dataset Classification")
    print("="*80)
    
    # Check if TabPFN is installed
    print("\nChecking TabPFN installation...")
    try:
        model = TabPFNClassifier()
        print("✓ TabPFN is installed and ready to use!")
    except ImportError as e:
        print(f"✗ TabPFN is not installed: {e}")
        print("\nTo install TabPFN, run:")
        print("  pip install tabpfn")
        return 1
    
    # Load data
    print("\nLoading Breast Cancer dataset...")
    loader = DataLoader('breast_cancer', random_state=42)
    X, y = loader.load_data()
    data = loader.prepare_data(X, y, test_size=0.2, scale_features=True)
    
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    
    print(f"Dataset loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    print(f"Features: {X_train.shape[1]}")
    
    # Check if dataset is suitable for TabPFN
    if X_train.shape[0] > 10000:
        print(f"Warning: TabPFN works best with ≤10,000 samples. Current: {X_train.shape[0]}")
        print("The model will automatically sample 10,000 samples for training.")
    
    if X_train.shape[1] > 100:
        print(f"Warning: TabPFN works best with ≤100 features. Current: {X_train.shape[1]}")
        print("The model will automatically use the first 100 features.")
    
    # Train model
    print("\nTraining TabPFN model...")
    model = TabPFNClassifier(device='cuda', N_ensemble_configurations=32)
    model.train(X_train, y_train)
    print("✓ Model trained successfully!")
    
    # Evaluate
    print("\nEvaluating model...")
    metrics = model.evaluate(X_test, y_test)
    
    print("\nTest Metrics:")
    for metric, value in metrics.items():
        if value is not None:
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: N/A")
    
    # Make predictions
    print("\nMaking predictions on test set...")
    predictions = model.predict(X_test.head(5))
    probabilities = model.predict_proba(X_test.head(5))
    
    print("\nFirst 5 predictions:")
    for i in range(5):
        print(f"  Sample {i+1}: Predicted class = {predictions[i]}, "
              f"Probabilities = {probabilities[i]}")
    
    print("\n" + "="*80)
    print("TabPFN example completed successfully!")
    print("="*80)
    print("\nTo run full experiments with TabPFN:")
    print("  cd src && python run_experiments.py breast_cancer TabPFN")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
