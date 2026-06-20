import glob
import os
import pandas as pd

def main():
    print("Searching for CSV files containing BraTS-GLI patient IDs...")
    csv_files = []
    # Recursively look for CSVs
    for root, dirs, files in os.walk("."):
        # Skip standard python cache / env dirs
        if "gpu_env" in root or ".git" in root or "node_modules" in root:
            continue
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))
                
    for path in csv_files:
        try:
            df = pd.read_csv(path)
            # Check if any patient ID contains BraTS-GLI
            matches = df[df['patient_id'].astype(str).str.contains('BraTS-GLI')]
            if not matches.empty:
                print(f"FOUND matches in CSV: {path}")
                print(f" - Columns: {df.columns.tolist()}")
                print(f" - Count: {len(matches)}")
                print(f" - Sample IDs: {matches['patient_id'].head(3).tolist()}")
        except Exception as e:
            pass

if __name__ == "__main__":
    main()
