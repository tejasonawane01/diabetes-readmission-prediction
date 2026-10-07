"""
Data preparation and feature engineering — V3
Key insight: raw diagnosis codes (458 unique values) are the TOP predictive features.
Converting them to 10 categories destroyed most of the signal.
V3 uses target encoding to preserve high-cardinality feature signal.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from config import (
    DATA_FILE, TARGET_COLUMN, IMBALANCE_THRESHOLD, AGE_MAP, DIAG_ORDER,
    A1C_MAP, GLU_MAP, MED_CHANGE_MAP, MEDICATION_COLS,
    TOP_SPECIALTIES, DISCHARGE_MAP, ADMISSION_TYPE_MAP, ADMISSION_SOURCE_MAP
)


def load_data(filepath=DATA_FILE):
    """Load the diabetes dataset."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def remove_impossible_readmissions(df):
    """Remove patients who cannot be readmitted (expired/hospice)."""
    print("\n=== Removing Impossible Readmissions ===")
    
    impossible_discharges = ['Expired', 'Hospice / home', 'Hospice / medical facility']
    
    if 'discharge_disposition_id' in df.columns:
        mask = df['discharge_disposition_id'].isin(impossible_discharges)
        removed = mask.sum()
        df_clean = df[~mask].copy()
        print(f"  Removed {removed} patients (expired/hospice)")
        print(f"  Remaining: {len(df_clean):,} rows")
    else:
        df_clean = df.copy()
    
    return df_clean


def handle_missing_values(df):
    """Handle '?' placeholders and missing values."""
    print("\n=== Handling Missing Values ===")
    df_clean = df.copy()
    
    # Replace '?' with NaN
    df_clean.replace('?', np.nan, inplace=True)
    
    # Analyze missing values
    missing_pct = df_clean.isnull().mean() * 100
    print(f"\nColumns with missing values (>0%):")
    for col in missing_pct[missing_pct > 0].sort_values(ascending=False).index:
        print(f"  {col}: {missing_pct[col]:.1f}%")
    
    # Drop high-missingness columns
    drop_cols = missing_pct[missing_pct > IMBALANCE_THRESHOLD].index.tolist()
    if drop_cols:
        print(f"\nDropping columns with >{IMBALANCE_THRESHOLD}% missing: {drop_cols}")
        df_clean.drop(columns=drop_cols, inplace=True)
    
    # Mode-impute remaining categoricals (including medical_specialty with "Unknown")
    for col in df_clean.select_dtypes(include=['object']).columns:
        if col == TARGET_COLUMN:
            continue
        null_count = df_clean[col].isnull().sum()
        if null_count > 0:
            # Use 'Unknown' for high-missing columns, mode for low-missing
            if missing_pct.get(col, 0) > 20:
                fill_val = 'Unknown'
            else:
                mode_val = df_clean[col].mode(dropna=True)
                fill_val = mode_val[0] if len(mode_val) > 0 else 'Unknown'
            df_clean[col] = df_clean[col].fillna(fill_val)
            print(f"  Imputed {col}: {null_count} values -> '{fill_val}'")
    
    print(f"Remaining nulls: {df_clean.isnull().sum().sum()}")
    return df_clean


def drop_low_value_columns(df):
    """Drop identifiers, description columns, and near-zero variance columns."""
    print("\n=== Dropping Low-Value Columns ===")
    
    id_cols = ['encounter_id', 'patient_nbr']
    desc_cols = ['diag_1_desc', 'diag_2_desc', 'diag_3_desc']
    
    # Near-zero variance meds
    candidate_meds = [
        'examide', 'citoglipton', 'troglitazone', 'tolbutamide',
        'chlorpropamide', 'acetohexamide', 'tolazamide',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone',
        'metformin-pioglitazone', 'glipizide-metformin', 'glyburide-metformin',
        'glyburide.metformin', 'glipizide.metformin',
        'glimepiride.pioglitazone', 'metformin.rosiglitazone',
        'metformin.pioglitazone'
    ]
    
    nzv_meds = []
    for c in candidate_meds:
        if c in df.columns:
            vc = df[c].value_counts(normalize=True)
            if vc.iloc[0] > 0.98:
                nzv_meds.append(c)

    # NOTE: Keep diag_1, diag_2, diag_3 raw codes — they are the TOP features!
    cols_to_drop = [c for c in id_cols + desc_cols + nzv_meds if c in df.columns]
    if cols_to_drop:
        print(f"  Dropping: {cols_to_drop}")
        df.drop(columns=cols_to_drop, inplace=True)
    
    print(f"  Shape after drops: {df.shape}")
    return df


def map_diagnosis_code(code):
    """Map ICD-9 code to clinical category."""
    if str(code) in ['nan', '?', 'None', 'Unknown']:
        return 'Unknown'
    try:
        c = float(str(code).split('.')[0])
        if str(code).startswith('250'):
            return 'Diabetes'
        if 390 <= c <= 459 or c == 785:
            return 'Circulatory'
        if 460 <= c <= 519 or c == 786:
            return 'Respiratory'
        if 520 <= c <= 579 or c == 787:
            return 'Digestive'
        if 800 <= c <= 999:
            return 'Injury'
        if 710 <= c <= 739:
            return 'Musculoskeletal'
        if 580 <= c <= 629 or c == 788:
            return 'Genitourinary'
        if 140 <= c <= 239:
            return 'Neoplasms'
        return 'Other'
    except Exception:
        return 'Other'


def encode_medications(df):
    """Encode medication columns using meaningful ordinals."""
    print("\n=== Encoding Medication Columns ===")
    df_med = df.copy()
    
    changed_count = 0
    for col in MEDICATION_COLS:
        if col in df_med.columns:
            df_med[col] = df_med[col].map(MED_CHANGE_MAP).fillna(0).astype(int)
            changed_count += 1
    
    print(f"  Encoded {changed_count} medication columns")
    
    # Aggregate medication features
    med_cols_present = [c for c in MEDICATION_COLS if c in df_med.columns]
    if med_cols_present:
        df_med['total_med_changes'] = (df_med[med_cols_present] >= 2).sum(axis=1)
        df_med['total_active_meds'] = (df_med[med_cols_present] >= 1).sum(axis=1)
        print(f"  + total_med_changes, total_active_meds")
    
    return df_med


def engineer_features(df):
    """Create domain-informed features. Keep raw diag codes for target encoding later."""
    print("\n=== Feature Engineering ===")
    df_eng = df.copy()
    
    # 1. Age ordinal
    if 'age' in df_eng.columns:
        df_eng['age_numeric'] = df_eng['age'].map(AGE_MAP).fillna(5)
        df_eng.drop(columns=['age'], inplace=True)
        print("  + age_numeric")
    
    # 2. Prior visits score
    visit_cols = ['number_outpatient', 'number_emergency', 'number_inpatient']
    if all(col in df_eng.columns for col in visit_cols):
        df_eng['prior_visits'] = sum(df_eng[col] for col in visit_cols)
        df_eng['high_utilizer'] = (df_eng['prior_visits'] >= 3).astype(int)
        print("  + prior_visits, high_utilizer")
    
    # 3. number_inpatient features (STRONGEST predictor: 34% -> 81% readmit rate)
    if 'number_inpatient' in df_eng.columns:
        df_eng['has_prior_inpatient'] = (df_eng['number_inpatient'] > 0).astype(int)
        df_eng['inpatient_squared'] = df_eng['number_inpatient'] ** 2
        print("  + has_prior_inpatient, inpatient_squared")
    
    # 4. number_diagnoses features (9% -> 47% readmit rate)
    if 'number_diagnoses' in df_eng.columns:
        df_eng['high_comorbidity'] = (df_eng['number_diagnoses'] >= 7).astype(int)
        print("  + high_comorbidity")
    
    # 5. Medication change flag
    if 'change' in df_eng.columns:
        df_eng['medication_changed'] = (df_eng['change'] == 'Ch').astype(int)
        print("  + medication_changed")
    
    # 6. diabetesMed flag
    if 'diabetesMed' in df_eng.columns:
        df_eng['diabetes_med_flag'] = (df_eng['diabetesMed'] == 'Yes').astype(int)
        print("  + diabetes_med_flag")
    
    # 7. Diagnosis CATEGORIES (Drop raw codes)
    for diag_col in ['diag_1', 'diag_2', 'diag_3']:
        if diag_col in df_eng.columns:
            cat_col = f'{diag_col}_category'
            cat_vals = df_eng[diag_col].apply(map_diagnosis_code)
            df_eng[cat_col] = cat_vals
            
            if diag_col == 'diag_1':
                df_eng['diabetes_primary'] = (cat_vals == 'Diabetes').astype(int)
                df_eng['circulatory_primary'] = (cat_vals == 'Circulatory').astype(int)
            
            # Drop the raw hyper-granular column
            df_eng.drop(columns=[diag_col], inplace=True)
            print(f"  + {cat_col}")
    
    # 8. A1C ordinal
    if 'A1Cresult' in df_eng.columns:
        df_eng['A1C_ordinal'] = df_eng['A1Cresult'].map(A1C_MAP).fillna(0).astype(int)
        df_eng['A1C_abnormal'] = (df_eng['A1C_ordinal'] >= 2).astype(int)
        df_eng.drop(columns=['A1Cresult'], inplace=True)
        print("  + A1C_ordinal, A1C_abnormal")
    
    # 9. Glucose serum ordinal
    if 'max_glu_serum' in df_eng.columns:
        df_eng['glu_serum_ordinal'] = df_eng['max_glu_serum'].map(GLU_MAP).fillna(0).astype(int)
        df_eng.drop(columns=['max_glu_serum'], inplace=True)
        print("  + glu_serum_ordinal")
    
    # 10. Interaction features
    interactions = [
        ('num_medications', 'number_inpatient', 'meds_x_inpatient'),
        ('time_in_hospital', 'num_medications', 'time_x_meds'),
        ('time_in_hospital', 'num_procedures', 'time_x_procedures'),
        ('number_diagnoses', 'num_medications', 'diag_x_meds'),
        ('number_diagnoses', 'number_inpatient', 'diag_x_inpatient'),
        ('num_lab_procedures', 'time_in_hospital', 'labs_x_time'),
    ]
    for col_a, col_b, name in interactions:
        if col_a in df_eng.columns and col_b in df_eng.columns:
            df_eng[name] = df_eng[col_a] * df_eng[col_b]
            print(f"  + {name}")
    
    # 11. Intensity features
    if 'num_lab_procedures' in df_eng.columns and 'time_in_hospital' in df_eng.columns:
        df_eng['labs_per_day'] = df_eng['num_lab_procedures'] / df_eng['time_in_hospital'].clip(lower=1)
        print("  + labs_per_day")
    
    if 'num_medications' in df_eng.columns and 'time_in_hospital' in df_eng.columns:
        df_eng['meds_per_day'] = df_eng['num_medications'] / df_eng['time_in_hospital'].clip(lower=1)
        print("  + meds_per_day")
    
    # 12. Emergency visit features
    if 'number_emergency' in df_eng.columns:
        df_eng['has_emergency'] = (df_eng['number_emergency'] > 0).astype(int)
        print("  + has_emergency")
    
    return df_eng


def prepare_data_before_split(df):
    """
    Run all data prep steps EXCEPT encoding.
    Encoding happens after train/test split to prevent target leakage.
    """
    print("\n" + "="*70)
    print("DATA PREPARATION PIPELINE V3".center(70))
    print("="*70)
    
    df = remove_impossible_readmissions(df)
    df = handle_missing_values(df)
    df = drop_low_value_columns(df)
    df = encode_medications(df)
    df = engineer_features(df)
    
    # Ensure target is int
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    
    print(f"\n  Pre-encoding shape: {df.shape}")
    print(f"  Target distribution: {df[TARGET_COLUMN].value_counts().to_dict()}")
    
    return df


def encode_after_split(X_train, X_test, y_train):
    """
    Encode categorical features AFTER train/test split.
    Uses pd.get_dummies to one-hot encode all categorical columns, exactly as done in the notebook.
    Then applies KMeans clustering to generate patient archetypes.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    import re
    
    print("\n=== Encoding Features (leak-safe) ===")
    
    cat_cols = X_train.select_dtypes(include=['object', 'bool']).columns.tolist()
    
    if cat_cols:
        print(f"  One-hot encoding: {cat_cols}")
        
        # One-hot encode using pandas get_dummies
        X_train_encoded = pd.get_dummies(X_train, columns=cat_cols, drop_first=True, dtype=int)
        X_test_encoded = pd.get_dummies(X_test, columns=cat_cols, drop_first=True, dtype=int)
        
        # Align the test set with the train set (adds missing columns, removes extra ones)
        X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)
        
        # Ensure XGBoost compatibility (no special characters in column names)
        safe_cols = [re.sub(r'[\[\]<>]', '_', str(col)) for col in X_train_encoded.columns]
        X_train_encoded.columns = safe_cols
        X_test_encoded.columns = safe_cols
        
        X_train = X_train_encoded.fillna(0)
        X_test = X_test_encoded.fillna(0)
    else:
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)
        
    print("\n=== Generating Patient Archetypes (Clustering) ===")
    # Scale data for clustering
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Fit KMeans (we use 5 archetypes)
    kmeans = KMeans(n_clusters=5, random_state=42, n_init='auto')
    train_clusters = kmeans.fit_predict(X_train_scaled)
    test_clusters = kmeans.predict(X_test_scaled)
    
    # One-hot encode the cluster labels
    for i in range(5):
        X_train[f'archetype_{i}'] = (train_clusters == i).astype(int)
        X_test[f'archetype_{i}'] = (test_clusters == i).astype(int)
        
    print(f"  Added 5 KMeans archetype features")
    print(f"  Final feature count: {X_train.shape[1]}")
    
    return X_train, X_test
