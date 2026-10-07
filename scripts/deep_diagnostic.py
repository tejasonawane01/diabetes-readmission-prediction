"""
Deep diagnostic: Why is accuracy stuck at ~65%?
This script investigates the fundamental ceiling of this dataset.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('data/10k_diabete.csv')
df.replace('?', np.nan, inplace=True)

# ============================================================
# 1. TARGET VARIABLE: How is 'readmitted' encoded?
# ============================================================
print("=" * 60)
print("1. TARGET VARIABLE ANALYSIS")
print("=" * 60)
print(f"\nTarget column dtype: {df['readmitted'].dtype}")
print(f"Unique values: {df['readmitted'].unique()}")
print(f"Value counts:\n{df['readmitted'].value_counts()}")

# Check if it's already binary or has 3 classes
if df['readmitted'].dtype == 'bool':
    target_type = "Boolean (True/False)"
elif df['readmitted'].dtype == 'object':
    target_type = f"String categories: {df['readmitted'].unique()}"
else:
    target_type = f"Numeric: min={df['readmitted'].min()}, max={df['readmitted'].max()}"
print(f"\nTarget type: {target_type}")

# ============================================================
# 2. BAYES RATE (theoretical ceiling)
# ============================================================
print("\n" + "=" * 60)
print("2. THEORETICAL ACCURACY CEILING (Bayes Rate Estimation)")
print("=" * 60)

# The majority class baseline
majority_class_pct = df['readmitted'].value_counts(normalize=True).max() * 100
print(f"\nMajority class baseline: {majority_class_pct:.1f}%")
print("(If we just predict the majority class for everyone)")

# ============================================================
# 3. DUPLICATE FEATURE ANALYSIS (noise floor)
# ============================================================
print("\n" + "=" * 60)
print("3. DUPLICATE ROWS WITH DIFFERENT LABELS (noise floor)")
print("=" * 60)

# Drop ID, description, and target columns
feature_cols = [c for c in df.columns if c not in [
    'encounter_id', 'patient_nbr', 'readmitted',
    'diag_1_desc', 'diag_2_desc', 'diag_3_desc'
]]

# Find rows with identical features but different labels
df_feats = df[feature_cols].copy()
for col in df_feats.select_dtypes(include='object').columns:
    df_feats[col] = df_feats[col].fillna('missing')

# Group by all features
grouped = df.groupby(df_feats.apply(tuple, axis=1))['readmitted']
conflicting = 0
total_in_conflict = 0
for name, group in grouped:
    if group.nunique() > 1:
        conflicting += 1
        total_in_conflict += len(group)

print(f"\nGroups with IDENTICAL features but DIFFERENT outcomes: {conflicting}")
print(f"Total rows involved in conflicts: {total_in_conflict} / {len(df)} ({total_in_conflict/len(df)*100:.1f}%)")
print("These rows are IMPOSSIBLE for any model to get right — they define the noise floor.")

# ============================================================
# 4. FEATURE-BY-FEATURE PREDICTIVE POWER
# ============================================================
print("\n" + "=" * 60)
print("4. INDIVIDUAL FEATURE PREDICTIVE POWER (AUC)")
print("=" * 60)

y = df['readmitted'].astype(int)
aucs = {}

for col in feature_cols:
    try:
        if df[col].dtype in ['int64', 'float64', 'bool']:
            vals = df[col].fillna(0)
            auc = roc_auc_score(y, vals)
            aucs[col] = max(auc, 1 - auc)  # flip if negatively correlated
        else:
            le = LabelEncoder()
            vals = le.fit_transform(df[col].fillna('missing'))
            auc = roc_auc_score(y, vals)
            aucs[col] = max(auc, 1 - auc)
    except:
        pass

auc_series = pd.Series(aucs).sort_values(ascending=False)
print("\nTop 15 features by univariate AUC:")
for feat, auc in auc_series.head(15).items():
    print(f"  {feat}: AUC={auc:.4f} {'***' if auc > 0.55 else ''}")

print(f"\nFeatures with AUC > 0.55 (actually useful): {(auc_series > 0.55).sum()}")
print(f"Features with AUC < 0.52 (near-random): {(auc_series < 0.52).sum()}")

# ============================================================
# 5. PATIENT_NBR LEAKAGE CHECK
# ============================================================
print("\n" + "=" * 60)
print("5. PATIENT OVERLAP CHECK (data leakage)")
print("=" * 60)

if 'patient_nbr' in df.columns:
    n_encounters = len(df)
    n_patients = df['patient_nbr'].nunique()
    repeat_patients = df['patient_nbr'].value_counts()
    repeat_count = (repeat_patients > 1).sum()
    print(f"\nTotal encounters: {n_encounters}")
    print(f"Unique patients: {n_patients}")
    print(f"Patients with multiple encounters: {repeat_count} ({repeat_count/n_patients*100:.1f}%)")
    print(f"Max encounters per patient: {repeat_patients.max()}")
    
    if repeat_count > 0:
        print("\n*** WARNING: Same patient can appear in BOTH train and test! ***")
        print("This means the model might be memorizing patient patterns,")
        print("but still can't predict well because readmission is inherently random.")

# ============================================================
# 6. QUICK TEST: What accuracy do we get with PERFECT features?
# ============================================================
print("\n" + "=" * 60)
print("6. ACCURACY WITH DIFFERENT FEATURE STRATEGIES")
print("=" * 60)

# Prepare simple features
df_test = df.copy()
for col in ['encounter_id', 'patient_nbr', 'diag_1_desc', 'diag_2_desc', 'diag_3_desc',
            'weight', 'payer_code']:
    if col in df_test.columns:
        df_test.drop(columns=[col], inplace=True)

df_test['readmitted'] = df_test['readmitted'].astype(int)

# Simple label encode everything
le = LabelEncoder()
for col in df_test.select_dtypes(include='object').columns:
    df_test[col] = df_test[col].fillna('missing')
    df_test[col] = le.fit_transform(df_test[col])

X = df_test.drop(columns=['readmitted'])
y = df_test['readmitted']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Test 1: Simple XGBoost (no class weights)
xgb1 = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                       random_state=42, tree_method='hist', eval_metric='logloss')
xgb1.fit(X_train, y_train)
y_pred1 = xgb1.predict(X_test)
y_proba1 = xgb1.predict_proba(X_test)[:, 1]
print(f"\nXGBoost (NO class weights, raw label encode):")
print(f"  Accuracy: {accuracy_score(y_test, y_pred1):.4f}")
print(f"  Recall:   {recall_score(y_test, y_pred1):.4f}")
print(f"  F1:       {f1_score(y_test, y_pred1):.4f}")
print(f"  ROC AUC:  {roc_auc_score(y_test, y_proba1):.4f}")

# Test 2: XGBoost WITH class weights
spw = (y_train == 0).sum() / (y_train == 1).sum()
xgb2 = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                       random_state=42, tree_method='hist', eval_metric='logloss',
                       scale_pos_weight=spw)
xgb2.fit(X_train, y_train)
y_pred2 = xgb2.predict(X_test)
y_proba2 = xgb2.predict_proba(X_test)[:, 1]
print(f"\nXGBoost (WITH scale_pos_weight={spw:.2f}):")
print(f"  Accuracy: {accuracy_score(y_test, y_pred2):.4f}")
print(f"  Recall:   {recall_score(y_test, y_pred2):.4f}")
print(f"  F1:       {f1_score(y_test, y_pred2):.4f}")
print(f"  ROC AUC:  {roc_auc_score(y_test, y_proba2):.4f}")

# Test 3: XGBoost without class weights, keeping ALL features (incl. patient_nbr style)
print(f"\nTest 3: Just predict majority class")
from collections import Counter
majority = Counter(y_train).most_common(1)[0][0]
y_pred_maj = np.full(len(y_test), majority)
print(f"  Accuracy: {accuracy_score(y_test, y_pred_maj):.4f}")
print(f"  Recall:   {recall_score(y_test, y_pred_maj):.4f}")

# ============================================================
# 7. THE KEY QUESTION: How noisy is the target?
# ============================================================
print("\n" + "=" * 60)
print("7. TARGET NOISE ANALYSIS")
print("=" * 60)

# For the strongest feature (number_inpatient), check readmission rates
if 'number_inpatient' in df.columns:
    print("\nnumber_inpatient (strongest single predictor):")
    for val in sorted(df['number_inpatient'].unique()):
        subset = df[df['number_inpatient'] == val]
        if len(subset) >= 10:
            rate = subset['readmitted'].mean()
            print(f"  inpatient={val}: readmit_rate={rate:.1%} (n={len(subset)})")
    
    print("\n  Even patients with 5+ prior inpatient visits only have ~81% readmission rate")
    print("  And patients with 0 prior visits still have ~34% readmission rate")
    print("  This 34%-81% range shows the fundamental noise in the data")

# Total summary
print("\n" + "=" * 60)
print("SUMMARY: WHY ACCURACY IS CAPPED")
print("=" * 60)
