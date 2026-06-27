import pandas as pd
import os

df = pd.read_csv('data/metadata/metadata_brats2021.csv')

def update_row(row):
    pid = row['patient_id']
    pid_num_str = pid.split('_')[-1]
    if not pid_num_str.isdigit():
        return row
    
    pid_num = int(pid_num_str)
    base_dir = 'Training_Data'
    split = 'train'
    
    if 0 <= pid_num <= 150:
        base_dir = 'Validation'
        split = 'val'
    elif 151 <= pid_num <= 367:
        base_dir = 'Testing'
        split = 'test'
        
    row['split'] = split
    
    for col in ['flair_path', 't1_path', 't1ce_path', 't2_path', 'seg_path']:
        if pd.notna(row[col]) and str(row[col]).strip() != '':
            old_path = str(row[col]).replace('\\', '/')
            parts = old_path.split('/')
            if len(parts) >= 2:
                # new path format: base_dir/PatientID/Filename
                new_path = f"{base_dir}/{parts[-2]}/{parts[-1]}"
                row[col] = new_path
                
    return row

df = df.apply(update_row, axis=1)
df.to_csv('data/metadata/metadata_brats2021.csv', index=False)
print('Successfully updated metadata paths and splits for Train, Val, and Test folders!')
