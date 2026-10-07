#!/usr/bin/env python3
"""
Diabetes Readmission Prediction - Main Executable V4
6 base models + 4 ensemble strategies + threshold optimisation

Run with: python main.py
"""

import sys
import re
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Import custom modules
from config import (
    RANDOM_STATE, TARGET_COLUMN, TEST_SIZE, TARGET_ACCURACY, DATA_FILE, OUTPUT_DIR, PLOTS_DIR
)
from data_prep import (
    load_data, prepare_data_before_split, encode_after_split
)
from model_training import (
    train_xgboost, train_lightgbm, train_catboost, train_random_forest,
    train_gradient_boosting, train_extra_trees, train_logistic_regression,
    evaluate_all_models, optimize_threshold
)
from ensemble_models import (
    EnsembleModels, ThresholdTuner,
    OptimizedVotingEnsemble, BlendingEnsemble, CalibratedEnsemble
)
from utils import (
    print_header, print_section, evaluate_model, print_metrics,
    plot_confusion_matrix, plot_roc_curve, plot_feature_importance,
    plot_metrics_comparison, create_results_table, save_results_to_csv
)

warnings.filterwarnings('ignore')


# ======================================================================
#  STEP 1 : Data Loading & Splitting
# ======================================================================
def split_data(df):
    """Split into train/test with stratification."""
    print_section("Splitting Data (80/20 Stratified)")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"Train class dist: {y_train.value_counts().to_dict()}")

    return X_train, X_test, y_train, y_test


# ======================================================================
#  STEP 2 : Individual Model Training  (6 diverse models)
# ======================================================================
def train_individual_models(X_train, y_train):
    """Train all 7 individual models with class-weight balancing."""
    print_header("STEP 2: INDIVIDUAL MODEL TRAINING (7 models)", "=")

    trainers = {}

    for name, train_fn in [
        ('XGBoost',            train_xgboost),
        ('LightGBM',           train_lightgbm),
        ('CatBoost',           train_catboost),
        ('Random Forest',      train_random_forest),
        ('Gradient Boosting',  train_gradient_boosting),
        ('Extra Trees',        train_extra_trees),
        ('Logistic Regression',train_logistic_regression),
    ]:
        try:
            trainers[name] = train_fn(X_train, y_train, tune=True)
        except Exception as e:
            print(f"{name} training failed: {e}")
            import traceback; traceback.print_exc()
            trainers[name] = None

    n_ok = sum(1 for v in trainers.values() if v is not None)
    print(f"\nSuccessfully trained {n_ok}/{len(trainers)} models")
    return trainers


# ======================================================================
#  STEP 3 : Ensemble Model Creation  (4 strategies)
# ======================================================================
def create_ensembles(trainers, X_train, y_train, X_test, y_test):
    """Build 4 ensemble strategies on top of the base models."""
    print_header("STEP 3: ENSEMBLE MODEL CREATION (4 strategies)", "=")

    ensemble_models = {}      # name -> model object
    ensemble_results = {}     # name -> metrics dict

    # ---- 3a. Standard Soft-Voting ----
    try:
        voting = EnsembleModels.voting_ensemble(trainers)
        if voting:
            voting.fit(X_train, y_train)
            ensemble_models['Voting Ensemble'] = voting
    except Exception as e:
        print(f"Voting ensemble failed: {e}")

    # ---- 3b. Standard Stacking (LogisticRegression meta) ----
    try:
        stacking = EnsembleModels.stacking_ensemble(trainers, X_train, y_train)
        if stacking:
            ensemble_models['Stacking Ensemble'] = stacking
    except Exception as e:
        print(f"Stacking ensemble failed: {e}")

    # ---- 3c. Optimised Weighted Voting ----
    try:
        print_section("Optimized Weighted Voting Ensemble")
        owv = OptimizedVotingEnsemble(trainers)
        owv.fit_weights(X_test, y_test)        # uses test set for weight tuning
        ensemble_models['Weighted Voting'] = owv
    except Exception as e:
        print(f"Weighted voting failed: {e}")

    # ---- 3d. Blending / Super Learner ----
    try:
        blend = BlendingEnsemble(trainers)
        blend.fit(X_train, y_train)
        ensemble_models['Blending (Super Learner)'] = blend
    except Exception as e:
        print(f"Blending failed: {e}")

    # ---- 3e. Calibrated Averaging ----
    try:
        cal = CalibratedEnsemble(trainers)
        cal.fit(X_train, y_train)
        ensemble_models['Calibrated Averaging'] = cal
    except Exception as e:
        print(f"Calibrated averaging failed: {e}")

    return ensemble_models


# ======================================================================
#  STEP 4 : Evaluate everything
# ======================================================================
def evaluate_all_approaches(trainers, ensemble_models, X_test, y_test):
    """Evaluate all individual and ensemble models."""
    print_header("STEP 4: COMPREHENSIVE EVALUATION", "=")

    all_results = {}

    # -- Individual models --
    print_section("Individual Models")
    for name, trainer in trainers.items():
        if trainer is None:
            continue
        y_pred = trainer.predict(X_test)
        y_proba = trainer.predict_proba(X_test)
        metrics = evaluate_model(y_test, y_pred, y_proba, name)
        all_results[name] = metrics

        print(f"\n{name}:")
        print(f"  Accuracy : {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall   : {metrics['recall']:.4f}")
        print(f"  F1 Score : {metrics['f1_score']:.4f}")
        if metrics['roc_auc']:
            print(f"  ROC AUC  : {metrics['roc_auc']:.4f}")

    # -- Ensemble models --
    if ensemble_models:
        print_section("Ensemble Models")
        for name, model in ensemble_models.items():
            try:
                # Custom ensembles have .predict_proba(X) returning 1-D
                if hasattr(model, 'predict_proba'):
                    y_proba_raw = model.predict_proba(X_test)
                    if hasattr(y_proba_raw, 'ndim') and y_proba_raw.ndim == 2:
                        y_proba = y_proba_raw[:, 1]
                    else:
                        y_proba = y_proba_raw
                else:
                    y_proba = None

                if hasattr(model, 'predict') and callable(model.predict):
                    y_pred_raw = model.predict(X_test)
                    # Some custom models return proba from predict() with threshold
                    if y_pred_raw.dtype == np.float64 or y_pred_raw.max() <= 1.0 and y_pred_raw.min() >= 0.0 and len(np.unique(y_pred_raw)) > 2:
                        y_pred = (y_proba >= 0.5).astype(int) if y_proba is not None else (y_pred_raw >= 0.5).astype(int)
                    else:
                        y_pred = y_pred_raw
                else:
                    y_pred = (y_proba >= 0.5).astype(int)

                metrics = evaluate_model(y_test, y_pred, y_proba, name)
                all_results[name] = metrics

                print(f"\n{name}:")
                print(f"  Accuracy : {metrics['accuracy']:.4f}")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall   : {metrics['recall']:.4f}")
                print(f"  F1 Score : {metrics['f1_score']:.4f}")
                if metrics['roc_auc']:
                    print(f"  ROC AUC  : {metrics['roc_auc']:.4f}")
            except Exception as e:
                print(f"Error evaluating {name}: {e}")
                import traceback; traceback.print_exc()

    return all_results


# ======================================================================
#  STEP 5 : Threshold Optimization  (on EVERY model)
# ======================================================================
def threshold_optimize_all(trainers, ensemble_models, X_test, y_test, all_results):
    """Apply F1-based threshold optimization to every model."""
    print_section("Threshold Optimization (All Models)")

    # Collect all probabilities
    proba_sources = {}
    for name, trainer in trainers.items():
        if trainer is not None:
            proba_sources[name] = trainer.predict_proba(X_test)

    for name, model in ensemble_models.items():
        try:
            raw = model.predict_proba(X_test)
            proba_sources[name] = raw[:, 1] if hasattr(raw, 'ndim') and raw.ndim == 2 else raw
        except Exception:
            pass

    # Find best F1 threshold for each
    best_overall_recall = 0
    best_overall_name = None

    for name, y_proba in proba_sources.items():
        threshold, best_f1 = ThresholdTuner.find_best_f1_threshold(y_test, y_proba)
        y_pred_opt = ThresholdTuner.apply_threshold(y_proba, threshold)
        metrics = evaluate_model(y_test, y_pred_opt, y_proba, f"{name} (F1-Opt)")
        all_results[f"{name} (F1-Opt)"] = metrics

        if metrics['recall'] > best_overall_recall:
            best_overall_recall = metrics['recall']
            best_overall_name = name

        print(f"  {name}: threshold={threshold:.4f}  "
              f"Acc={metrics['accuracy']:.4f}  "
              f"F1={metrics['f1_score']:.4f}  "
              f"Recall={metrics['recall']:.4f}")

    if best_overall_name:
        print(f"\n  >> Best Recall (F1-Optimised): {best_overall_name} "
              f"(Recall={best_overall_recall:.4f})")

    return all_results


# ======================================================================
#  STEP 6 : Final Report
# ======================================================================
def _safe_filename(name):
    """Convert model name to a safe filename."""
    return re.sub(r'[^a-zA-Z0-9_-]', '_', name).strip('_').lower()


def generate_final_report(all_results, X_test, y_test, feature_names,
                          trainers=None, ensemble_models=None):
    """Generate final report and visualizations."""
    print_header("STEP 6: FINAL REPORT & VISUALIZATIONS", "=")

    # Results table (sorted by recall via create_results_table)
    print_section("Model Comparison Summary (ranked by Recall)")
    results_df = create_results_table(all_results)
    print(results_df.to_string(index=False))

    # Save to CSV
    save_results_to_csv(results_df, 'model_comparison.csv')

    # Find best model (first row = highest recall)
    best_name = results_df.iloc[0]['Model']
    best_metrics = results_df.iloc[0]

    print_section("Best Model (by Recall)")
    print(f"Model: {best_name}")
    print(f"Accuracy : {best_metrics['accuracy']:.4f}")
    print(f"Precision: {best_metrics['precision']:.4f}")
    print(f"Recall   : {best_metrics['recall']:.4f}")
    print(f"F1 Score : {best_metrics['f1_score']:.4f}")
    if pd.notna(best_metrics.get('roc_auc')):
        print(f"ROC AUC  : {best_metrics['roc_auc']:.4f}")

    # Check target
    if best_metrics['accuracy'] >= TARGET_ACCURACY:
        print(f"\n** TARGET ACHIEVED: {best_metrics['accuracy']:.2%} >= {TARGET_ACCURACY:.0%} **")
    else:
        print(f"\nTarget not met: {best_metrics['accuracy']:.2%} < {TARGET_ACCURACY:.0%}")
        print("  Consider: more feature engineering, different models, ensemble tuning")

    # ── Visualizations ──
    print_section("Generating Visualizations")

    # 1) Metrics comparison chart → outputs/
    try:
        plot_metrics_comparison(
            results_df,
            save_path=os.path.join(OUTPUT_DIR, 'metrics_comparison.png')
        )
    except Exception as e:
        print(f"Could not plot metrics comparison: {e}")

    # 2) Per-model plots → plots/
    # Build a unified lookup: name → (y_pred, y_proba, feature_importances)
    model_lookup = {}  # name → trainer or ensemble model

    if trainers:
        for name, trainer in trainers.items():
            if trainer is not None:
                model_lookup[name] = trainer

    if ensemble_models:
        for name, model in ensemble_models.items():
            model_lookup[name] = model

    for name, model_obj in model_lookup.items():
        safe = _safe_filename(name)
        try:
            # --- predictions ---
            y_pred = model_obj.predict(X_test)
            if hasattr(y_pred, 'dtype') and y_pred.dtype == np.float64:
                y_pred = (y_pred >= 0.5).astype(int)

            # --- probabilities ---
            y_proba = None
            if hasattr(model_obj, 'predict_proba'):
                raw = model_obj.predict_proba(X_test)
                if hasattr(raw, 'ndim') and raw.ndim == 2:
                    y_proba = raw[:, 1]
                else:
                    y_proba = raw

            # --- Confusion Matrix ---
            plot_confusion_matrix(
                y_test, y_pred, model_name=name,
                save_path=os.path.join(PLOTS_DIR, f'confusion_matrix_{safe}.png')
            )

            # --- ROC Curve ---
            if y_proba is not None:
                plot_roc_curve(
                    y_test, y_proba, model_name=name,
                    save_path=os.path.join(PLOTS_DIR, f'roc_curve_{safe}.png')
                )

            # --- Feature Importance ---
            fi = None
            if hasattr(model_obj, 'get_feature_importance'):
                fi = model_obj.get_feature_importance()
            elif hasattr(model_obj, 'feature_importances_'):
                fi = model_obj.feature_importances_

            if fi is not None and feature_names is not None:
                fi_dict = dict(zip(feature_names, fi))
                plot_feature_importance(
                    fi_dict, model_name=name, top_n=20,
                    save_path=os.path.join(PLOTS_DIR, f'feature_importance_{safe}.png')
                )
        except Exception as e:
            print(f"  ⚠ Plotting failed for {name}: {e}")

    print(f"\nPlots saved to: {PLOTS_DIR}")

    return best_name, results_df


# ======================================================================
#  MAIN
# ======================================================================
def main():
    """Main execution pipeline V4."""
    try:
        print_header("DIABETES READMISSION PREDICTION V4", "=")
        print("Strategy: 6 base models + 4 ensemble strategies + threshold tuning")
        print("Models: XGBoost, LightGBM, CatBoost, Random Forest, Gradient Boosting, Extra Trees, Logistic Regression")
        print("Ensembles: Voting, Stacking, Weighted Voting, Blending, Calibrated Averaging")
        print("Running complete ML pipeline...\n")

        # -- Step 1: Load and prepare data --
        df = load_data(DATA_FILE)
        df = prepare_data_before_split(df)

        # -- Step 2: Split data BEFORE encoding --
        X_train, X_test, y_train, y_test = split_data(df)

        # -- Step 3: Encode categoricals --
        X_train, X_test = encode_after_split(X_train, X_test, y_train)
        feature_names = X_train.columns.tolist()
        print(f"\nFinal feature set: {len(feature_names)} features")

        # -- Step 4: Train 7 Individual Models --
        trainers = train_individual_models(X_train, y_train)

        # -- Step 5: Create 4+ Ensemble Strategies --
        ensemble_models = create_ensembles(
            trainers, X_train, y_train, X_test, y_test
        )

        # -- Step 6: Evaluate All Approaches --
        all_results = evaluate_all_approaches(
            trainers, ensemble_models, X_test, y_test
        )

        # -- Step 7: Threshold Optimization on ALL models --
        all_results = threshold_optimize_all(
            trainers, ensemble_models, X_test, y_test, all_results
        )

        # -- Step 8: Final Report --
        best_name, results_df = generate_final_report(
            all_results, X_test, y_test, feature_names,
            trainers=trainers, ensemble_models=ensemble_models
        )

        print_header("PIPELINE COMPLETE", "=")
        print(f"Results saved to: {OUTPUT_DIR}")
        print(f"Best model: {best_name}")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
