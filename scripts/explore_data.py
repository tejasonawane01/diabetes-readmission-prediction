import pandas as pd
import numpy as np

df = pd.read_csv('data/10k_diabete.csv')

print("=== Missing/? values ===")
for col in df.columns:
    q_count = (df[col] == '?').sum()
    missing = df[col].isna().sum()
    total = q_count + missing
    pct = total / len(df) * 100
    if total > 0:
        print(f"  {col}: {total} ({pct:.1f}%)")

print("\n=== Description columns ===")
print('diag_1_desc unique:', df['diag_1_desc'].nunique())
print('diag_2_desc unique:', df['diag_2_desc'].nunique())
print('diag_3_desc unique:', df['diag_3_desc'].nunique())

print("\n=== Medication columns value distributions ===")
med_cols = ['metformin', 'repaglinide', 'nateglinide', 'glimepiride', 'glipizide', 
            'glyburide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'insulin']
for col in med_cols:
    if col in df.columns:
        print(f"\n{col}:")
        print(df[col].value_counts())

print("\n=== Key categoricals ===")
print('\nchange:', df['change'].value_counts().to_dict())
print('diabetesMed:', df['diabetesMed'].value_counts().to_dict())
print('gender:', df['gender'].value_counts().to_dict())
print('race:', df['race'].value_counts().to_dict())

print("\n=== Numeric columns stats ===")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(df[numeric_cols].describe().to_string())

print("\n=== Target class balance ===")
print(df['readmitted'].value_counts())
print(f"Ratio: {df['readmitted'].value_counts()[False] / df['readmitted'].value_counts()[True]:.2f}:1")

print("\n=== Columns that will be encoded as high-cardinality ===")
for col in df.select_dtypes(include='object').columns:
    nunique = df[col].nunique()
    if nunique > 10:
        print(f"  {col}: {nunique} unique values")
