# shared/data_loader.py
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_name="data.csv"):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    file_path = os.path.join(project_root, file_name)
    
    if not os.path.exists(file_path):
        file_path = file_name
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Nepavyko rasti duomenų failo '{file_name}' nei projekto šaknyje, nei nurodytu keliu.")

    df = pd.read_csv(file_path)
    
    if 'rownames' in df.columns:
        df = df.drop(columns=['rownames'])
    elif df.columns[0] == 'Unnamed: 0':
        df = df.drop(columns=[df.columns[0]])

    continuous_cols = [
        'violent', 'murder', 'robbery', 'prisoners', 
        'afam', 'cauc', 'male', 'population', 'income', 'density'
    ]
    
    df_continuous = df[continuous_cols].dropna()
    
    z_scores = np.abs((df_continuous - df_continuous.mean()) / df_continuous.std())
    clean_mask = (z_scores < 3).all(axis=1)
    df_clean = df_continuous[clean_mask].copy()
    
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df_clean)
    df_scaled = pd.DataFrame(scaled_array, columns=continuous_cols)
    
    print(f"[data_loader] Pradiniai įrašai: {len(df)}, po valymo liko: {len(df_clean)}")
    return df_clean, df_scaled