import pandas as pd

df = pd.read_csv('data/metadata/metadata_brats2021.csv')

def fix_val_path(row):
    if row['split'] == 'val' and 'Validation/BraTS2021_' in str(row['flair_path']):
        for col in ['flair_path', 't1_path', 't1ce_path', 't2_path', 'seg_path']:
            if pd.notna(row[col]) and str(row[col]).strip() != '':
                row[col] = str(row[col]).replace('Validation/', 'Validation/BraTS2024-BraTS-GLI-ValidationData/validation_data/')
    return row

df = df.apply(fix_val_path, axis=1)
df.to_csv('data/metadata/metadata_brats2021.csv', index=False)
print('Fixed Validation paths in CSV!')
