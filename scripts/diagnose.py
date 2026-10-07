"""Diagnostic script to understand why accuracy is low and identify improvements."""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('data/10k_diabete.csv')

# 1. Check discharge dispositions vs readmission
print("=== Discharge Disposition vs Readmission ===")
ct = pd.crosstab(df['discharge_disposition_id'], df['readmitted'], margins=True)
ct['readmit_rate'] = ct[True] / (ct[True] + ct[False])
print(ct.sort_values('readmit_rate', ascending=False).to_string())

# 2. Key: patients who EXPIRED or went to HOSPICE can't be readmitted
print("\n=== Patients who can't be readmitted ===")
impossible = df[df['discharge_disposition_id'].isin(['Expired', 'Hospice / home', 'Hospice / medical facility'])]
print(f"Expired/Hospice patients: {len(impossible)} ({len(impossible)/len(df)*100:.1f}%)")
print(f"Their readmission rate: {impossible['readmitted'].mean():.4f}")
print("These should be REMOVED - they add noise (dead patients can't be readmitted)")

# 3. Check feature importance with a quick RF
print("\n=== Quick Feature Importance (after basic prep) ===")
df_quick = df.copy()
df_quick.replace('?', np.nan, inplace=True)

# Drop high-missing and text cols
drop = ['weight', 'payer_code', 'medical_specialty', 'diag_1_desc', 'diag_2_desc', 'diag_3_desc',
        'encounter_id', 'patient_nbr', 'max_glu_serum', 'A1Cresult']
df_quick.drop(columns=[c for c in drop if c in df_quick.columns], inplace=True)

# Encode target
df_quick['readmitted'] = df_quick['readmitted'].astype(int)

# Simple label encode everything
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in df_quick.select_dtypes(include='object').columns:
    df_quick[col] = df_quick[col].fillna('missing')
    df_quick[col] = le.fit_transform(df_quick[col])

X = df_quick.drop(columns=['readmitted'])
y = df_quick['readmitted']

rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight='balanced')
rf.fit(X, y)

importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 20 features:")
for feat, imp in importances.head(20).items():
    print(f"  {feat}: {imp:.4f}")

print("\nBottom 10 features (candidates for removal):")
for feat, imp in importances.tail(10).items():
    print(f"  {feat}: {imp:.4f}")

# 4. Check what accuracy we get with just top features
print("\n=== Accuracy with top features only ===")
top_features = importances.head(15).index.tolist()
X_train, X_test, y_train, y_test = train_test_split(X[top_features], y, test_size=0.2, 
                                                      stratify=y, random_state=42)
rf2 = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight='balanced')
rf2.fit(X_train, y_train)
print(f"Accuracy (top 15 features): {accuracy_score(y_test, rf2.predict(X_test)):.4f}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
rf3 = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight='balanced')
rf3.fit(X_train, y_train)
print(f"Accuracy (all features): {accuracy_score(y_test, rf3.predict(X_test)):.4f}")

# 5. Check after removing expired/hospice
print("\n=== After removing expired/hospice patients ===")
df_clean = df[~df['discharge_disposition_id'].isin(['Expired', 'Hospice / home', 'Hospice / medical facility'])].copy()
print(f"Remaining: {len(df_clean)} rows (removed {len(df) - len(df_clean)})")
print(f"New class distribution: {df_clean['readmitted'].value_counts().to_dict()}")
print(f"New ratio: {df_clean['readmitted'].value_counts()[False] / df_clean['readmitted'].value_counts()[True]:.2f}:1")

# 6. Check number of inpatient visits vs readmission
print("\n=== number_inpatient vs readmission ===")
for val in sorted(df['number_inpatient'].unique()):
    subset = df[df['number_inpatient'] == val]
    rate = subset['readmitted'].mean()
    if len(subset) >= 20:
        print(f"  inpatient={val}: readmit_rate={rate:.3f} (n={len(subset)})")

# 7. Check number_diagnoses vs readmission
print("\n=== number_diagnoses vs readmission ===")
for val in sorted(df['number_diagnoses'].unique()):
    subset = df[df['number_diagnoses'] == val]
    rate = subset['readmitted'].mean()
    if len(subset) >= 30:
        print(f"  diagnoses={val}: readmit_rate={rate:.3f} (n={len(subset)})")
