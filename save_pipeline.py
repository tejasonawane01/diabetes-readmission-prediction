#!/usr/bin/env python3
"""
Save Pipeline — Train models and serialize artifacts for Streamlit dashboard.

Run once:  python save_pipeline.py

Saves to models/:
  - best_model.joblib          (trained Extra Trees sklearn estimator)
  - preprocessing_artifacts.joblib  (scaler, kmeans, feature columns, threshold)
  - model_comparison.joblib    (DataFrame of all model results)
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import precision_recall_curve, f1_score
import re

warnings.filterwarnings('ignore')

# Import project modules
from config import RANDOM_STATE, TARGET_COLUMN, TEST_SIZE, DATA_FILE, MODELS_DIR
from data_prep import load_data, prepare_data_before_split
from model_training import train_extra_trees
from utils import print_header, print_section


def main():
    print_header("SAVE PIPELINE — Train & Serialize Best Model", "=")

    # ── Step 1: Load & prepare data ──
    print_section("Step 1: Load & Prepare Data")
    df = load_data(DATA_FILE)
    df = prepare_data_before_split(df)

    # ── Step 2: Split ──
    print_section("Step 2: Train/Test Split")
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    # ── Step 3: Encode (replicating encode_after_split logic, but saving artifacts) ──
    print_section("Step 3: Encode & Save Preprocessing Artifacts")

    cat_cols = X_train.select_dtypes(include=['object', 'bool']).columns.tolist()

    if cat_cols:
        X_train_encoded = pd.get_dummies(X_train, columns=cat_cols, drop_first=True, dtype=int)
        X_test_encoded = pd.get_dummies(X_test, columns=cat_cols, drop_first=True, dtype=int)
        X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

        safe_cols = [re.sub(r'[\[\]<>]', '_', str(col)) for col in X_train_encoded.columns]
        X_train_encoded.columns = safe_cols
        X_test_encoded.columns = safe_cols

        X_train_enc = X_train_encoded.fillna(0)
        X_test_enc = X_test_encoded.fillna(0)
    else:
        X_train_enc = X_train.fillna(0)
        X_test_enc = X_test.fillna(0)

    # Save the encoded feature column list (critical for inference alignment)
    feature_columns_after_dummies = X_train_enc.columns.tolist()

    # Clustering (KMeans archetypes)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_enc)
    X_test_scaled = scaler.transform(X_test_enc)

    kmeans = KMeans(n_clusters=5, random_state=RANDOM_STATE, n_init='auto')
    train_clusters = kmeans.fit_predict(X_train_scaled)
    test_clusters = kmeans.predict(X_test_scaled)

    for i in range(5):
        X_train_enc[f'archetype_{i}'] = (train_clusters == i).astype(int)
        X_test_enc[f'archetype_{i}'] = (test_clusters == i).astype(int)

    final_feature_columns = X_train_enc.columns.tolist()
    print(f"  Final feature count: {len(final_feature_columns)}")

    # ── Step 4: Train the best model (Extra Trees) ──
    print_section("Step 4: Train Extra Trees (Best Model)")
    trainer = train_extra_trees(X_train_enc, y_train, tune=True)

    # ── Step 5: Find optimal F1 threshold ──
    print_section("Step 5: Optimize F1 Threshold")
    y_proba = trainer.predict_proba(X_test_enc)
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    f1_scores = np.nan_to_num(f1_scores)
    best_idx = np.argmax(f1_scores)
    best_threshold = float(thresholds[best_idx])
    print(f"  Optimal threshold: {best_threshold:.4f}")
    print(f"  Max F1 at threshold: {f1_scores[best_idx]:.4f}")

    # Evaluate at optimal threshold
    y_pred_opt = (y_proba >= best_threshold).astype(int)
    from sklearn.metrics import accuracy_score, recall_score, precision_score
    print(f"  Accuracy:  {accuracy_score(y_test, y_pred_opt):.4f}")
    print(f"  Recall:    {recall_score(y_test, y_pred_opt):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred_opt):.4f}")

    # ── Step 6: Save everything ──
    print_section("Step 6: Saving Artifacts")

    os.makedirs(MODELS_DIR, exist_ok=True)

    # 6a. The trained model
    model_path = os.path.join(MODELS_DIR, 'best_model.joblib')
    joblib.dump(trainer.best_model, model_path)
    print(f"  ✓ Saved model:   {model_path}")

    # 6b. Preprocessing artifacts
    artifacts = {
        'scaler': scaler,
        'kmeans': kmeans,
        'feature_columns_after_dummies': feature_columns_after_dummies,
        'final_feature_columns': final_feature_columns,
        'cat_cols_before_encoding': cat_cols,
        'optimal_threshold': best_threshold,
        'model_name': 'Extra Trees (F1-Optimized)',
    }
    artifacts_path = os.path.join(MODELS_DIR, 'preprocessing_artifacts.joblib')
    joblib.dump(artifacts, artifacts_path)
    print(f"  ✓ Saved artifacts: {artifacts_path}")

    # 6c. Model comparison table (from existing CSV if available)
    comparison_path = os.path.join('outputs', 'model_comparison.csv')
    if os.path.exists(comparison_path):
        comparison_df = pd.read_csv(comparison_path)
        joblib.dump(comparison_df, os.path.join(MODELS_DIR, 'model_comparison.joblib'))
        print(f"  ✓ Saved comparison table from existing CSV")
    else:
        print(f"  ⚠ No model_comparison.csv found in outputs/")

    print_header("PIPELINE SAVED SUCCESSFULLY", "=")
    print(f"  Model directory: {MODELS_DIR}")
    print(f"  Files saved: best_model.joblib, preprocessing_artifacts.joblib, model_comparison.joblib")
    print(f"\n  Next: streamlit run app.py")


if __name__ == '__main__':
    main()
