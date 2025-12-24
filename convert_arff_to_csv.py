"""Convert ARFF files to CSV format."""
import os
import pandas as pd
from scipy.io import arff

# Data directory
data_dir = os.path.join(os.path.dirname(__file__), 'data')

# Find all ARFF files
arff_files = [f for f in os.listdir(data_dir) if f.endswith('.arff')]

print(f"Found {len(arff_files)} ARFF files to convert")

for arff_file in arff_files:
    arff_path = os.path.join(data_dir, arff_file)
    csv_path = os.path.join(data_dir, arff_file.replace('.arff', '.csv'))
    
    # Skip if CSV already exists
    if os.path.exists(csv_path):
        print(f"✓ {arff_file} -> {csv_path} (already exists)")
        continue
    
    try:
        # Load ARFF file
        data, meta = arff.loadarff(arff_path)
        df = pd.DataFrame(data)
        
        # Convert byte strings to regular strings if present
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = df[col].str.decode('utf-8')
                except (AttributeError, UnicodeDecodeError):
                    pass
        
        # Save to CSV
        df.to_csv(csv_path, index=False)
        print(f"✓ {arff_file} -> {csv_path}")
    except Exception as e:
        print(f"✗ {arff_file}: {e}")

print("\nConversion complete!")
