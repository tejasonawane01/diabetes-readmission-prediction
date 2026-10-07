# Diabetes Readmission Prediction — Production ML Pipeline

A comprehensive, modular Python project for predicting patient readmission using multiple machine learning models, ensemble techniques, and advanced hyperparameter tuning.

## 📊 Project Structure

```
diabetes_readmission/
├── main.py                  # Main executable orchestrator
├── config.py               # Configuration & constants
├── data_prep.py            # Data loading, cleaning, feature engineering
├── model_training.py       # Single model training & hyperparameter tuning
├── ensemble_models.py      # Ensemble methods (Voting, Stacking, AdaBoost)
├── utils.py                # Utility functions (metrics, plotting, helpers)
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── data/                  # Input data directory
│   └── 10k_diabete.csv
├── outputs/               # Results, models, visualizations
├── models/                # Saved model files (if implemented)
└── plots/                 # Generated plots and visualizations
```

## 🎯 Target Accuracy: 75%

This pipeline aims to achieve **75% accuracy** using:
- **Multiple models:** XGBoost, LightGBM, CatBoost, Random Forest
- **Ensemble methods:** Voting, Stacking, AdaBoost
- **Advanced techniques:** SMOTE balancing, hyperparameter tuning, threshold optimization
- **Data-driven decisions:** All feature engineering and model choices backed by analysis

## ⚡ Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Place `10k_diabete.csv` in the `data/` directory:
```bash
mkdir -p data
cp your_dataset.csv data/10k_diabete.csv
```

### 3. Run the Pipeline

```bash
python main.py
```

This will execute:
1. **Data Preparation:** Loading, cleaning, feature engineering, encoding
2. **Model Training:** XGBoost, LightGBM, CatBoost, Random Forest (with tuning)
3. **Ensemble Creation:** Voting, Stacking, AdaBoost combinations
4. **Evaluation:** Comprehensive metrics for all models
5. **Reporting:** Results table, visualizations, comparison plots

## 📁 Module Breakdown

### `main.py` — Main Orchestrator
Coordinates the entire pipeline:
- Data loading & preparation
- Train/test splitting
- SMOTE balancing
- Model training
- Ensemble creation
- Evaluation & reporting

**Run:** `python main.py`

### `config.py` — Configuration
Central configuration with:
- Paths (data, outputs, models, plots)
- Target metrics (75% accuracy, 80% AUC)
- Class imbalance settings (SMOTE k_neighbors=5)
- Feature engineering mappings (age, diagnosis, A1C, glucose)
- Hyperparameter ranges for all models
- Tuning settings (80 iterations, 5-fold CV)

**Modify:** Update hyperparameters or target metrics here

### `data_prep.py` — Data Preparation
Four-step pipeline:

1. **`load_data()`** — Load CSV
2. **`handle_missing_values()`** — Replace '?', drop >40% missing, mode-impute
3. **`drop_low_value_columns()`** — Remove identifiers, diagnose codes, near-zero variance features
4. **`engineer_features()`** — Create:
   - `age_numeric`: Ordinal age encoding
   - `prior_visits`: Cumulative utilization
   - `medication_changed`: Binary medication change flag
   - `diag_1_code`: ICD-9 → clinical category
   - `A1C_ordinal`: Glycemic control level
   - `glu_serum_ordinal`: Glucose severity

5. **`encode_categoricals()`** — Label-encode remaining object columns

**Functions:**
```python
df = load_data()  # Load CSV
df = handle_missing_values(df)  # Clean
df = drop_low_value_columns(df)  # Drop identifiers
df = engineer_features(df)  # Create new features
df = encode_categoricals(df)  # Encode to numeric
df = prepare_data(df)  # All-in-one wrapper
```

### `model_training.py` — Single Model Training
Train individual models with hyperparameter tuning:

**Supported Models:**
- **XGBoost** (`train_xgboost`)
- **LightGBM** (`train_lightgbm`)
- **CatBoost** (`train_catboost`)
- **Random Forest** (`train_random_forest`)
- **Gradient Boosting** (baseline, no tuning)

**Key Class:** `BaseModelTrainer`
- `tune_hyperparameters()` — RandomizedSearchCV with StratifiedKFold
- `train()` — Train on best hyperparameters
- `predict()` — Make predictions
- `predict_proba()` — Get probabilities
- `get_feature_importance()` — Extract feature importance

**Tuning Strategy:**
- **Search method:** RandomizedSearchCV (60-80 iterations)
- **Cross-validation:** 5-fold StratifiedKFold (preserves class balance)
- **Scoring:** ROC-AUC (more robust than accuracy for imbalanced data)
- **Parameters:** See `config.py` for ranges

**Example:**
```python
trainer = train_xgboost(X_train, y_train, tune=True)
y_pred = trainer.predict(X_test)
y_proba = trainer.predict_proba(X_test)
```

### `ensemble_models.py` — Ensemble & Advanced Techniques
Three ensemble strategies plus threshold optimization:

**EnsembleModels class:**
1. **`voting_ensemble()`** — Soft voting of multiple trained models
2. **`stacking_ensemble()`** — Stacking with LogisticRegression meta-learner
3. **`boosting_ensemble()`** — AdaBoost with XGBoost base estimators

**SMOTETrainer class:**
- Wraps any model
- Applies SMOTE before training
- Reports before/after class balance

**ThresholdTuner class:**
- `find_best_threshold()` — Optimize threshold for any metric
- `apply_threshold()` — Convert probabilities to binary predictions
- `evaluate_thresholds()` — Compare metrics across threshold range

**Hybrid Ensemble:**
- `create_hybrid_ensemble()` — Combines all strategies:
  1. Individual models with SMOTE
  2. Voting ensemble
  3. Stacking ensemble
  4. Threshold-optimized predictions

**Example:**
```python
voting = EnsembleModels.voting_ensemble(trainers_dict)
voting.fit(X_train, y_train)
y_pred = voting.predict(X_test)

# Threshold optimization
threshold, metrics, y_pred_opt = optimize_threshold(trainer, X_test, y_test)
```

### `utils.py` — Utilities
Helper functions for metrics, plotting, reporting:

**Metrics:**
- `evaluate_model()` — Compute accuracy, precision, recall, F1, ROC-AUC
- `print_metrics()` — Print formatted classification report

**Plotting:**
- `plot_confusion_matrix()` — Heatmap of predictions vs. actuals
- `plot_roc_curve()` — ROC curve with multiple models
- `plot_feature_importance()` — Top-20 feature importance chart
- `plot_metrics_comparison()` — Side-by-side metric comparison

**Data:**
- `find_optimal_threshold()` — Search thresholds to maximize a metric
- `create_results_table()` — Convert results to DataFrame
- `save_results_to_csv()` — Save to CSV

**Formatting:**
- `print_header()` — Styled section headers
- `print_section()` — Styled subsection headers

## 🚀 Features & Techniques

### Data Preparation
- ✓ Handle '?' placeholders and missing values
- ✓ Drop high-missingness columns (>40%)
- ✓ Drop identifiers and near-zero variance features
- ✓ ICD-9 diagnosis code mapping to clinical categories
- ✓ Feature engineering (age ordinal, utilization score, medication changes, lab ordinals)
- ✓ Label encoding for categoricals

### Class Imbalance
- ✓ Detect 2.9:1 class imbalance (True:False readmissions)
- ✓ Apply SMOTE on training set only (no test leakage)
- ✓ Report before/after balance statistics

### Model Training
- ✓ XGBoost with histogram acceleration
- ✓ LightGBM for fast training
- ✓ CatBoost for categorical feature handling
- ✓ Random Forest as baseline
- ✓ RandomizedSearchCV hyperparameter tuning (60-80 iterations)
- ✓ StratifiedKFold cross-validation (5 folds)
- ✓ ROC-AUC scoring (robust to imbalance)

### Ensemble Methods
- ✓ Soft Voting Classifier (combine multiple models)
- ✓ Stacking Classifier (meta-learner on top)
- ✓ AdaBoost (adaptive boosting)

### Advanced Techniques
- ✓ Threshold optimization for custom decision boundaries
- ✓ Feature importance extraction
- ✓ Cross-validation with StratifiedKFold
- ✓ Probability calibration (predict_proba)

## 📊 Metrics & Evaluation

All models evaluated on:
- **Accuracy** — % correct predictions
- **Precision** — % predicted readmissions that are correct
- **Recall** — % actual readmissions that are caught (clinically important!)
- **F1 Score** — Harmonic mean of precision & recall
- **ROC AUC** — Probability ranking quality

Confusion matrix, ROC curve, and feature importance visualizations generated for best model.

## 📈 Example Output

```
✓ XGBoost:
  Accuracy : 0.7200
  Precision: 0.3450
  Recall   : 0.1850
  F1 Score : 0.2380
  ROC AUC  : 0.6820

✓ Stacking Ensemble:
  Accuracy : 0.7350
  Precision: 0.3620
  Recall   : 0.2100
  F1 Score : 0.2680
  ROC AUC  : 0.7050

✓ TARGET ACHIEVED: 73.50% >= 75%   ⚠ (or ✓ if target met)
```

## 🔧 Customization

### Adjust Target Accuracy
Edit `config.py`:
```python
TARGET_ACCURACY = 0.75  # Change to 0.80 for 80%, etc.
```

### Modify Hyperparameter Ranges
Edit `config.py`:
```python
XGBOOST_PARAMS = {
    'n_estimators': (200, 600),
    'max_depth': (4, 10),
    # ...
}
```

### Enable/Disable Models
Edit `main.py`'s `train_individual_models()`:
```python
# Comment out to skip:
# trainers['LightGBM'] = train_lightgbm(...)
```

### Change Tuning Iterations
Edit `config.py`:
```python
RANDOMSEARCH_ITERATIONS = 100  # More iterations = longer tuning, potentially better
```

## 🐛 Troubleshooting

**LightGBM or CatBoost not installed?**
```bash
pip install lightgbm catboost
```

**Out of memory?**
- Reduce `RANDOMSEARCH_ITERATIONS` in `config.py`
- Use fewer CV folds (`CV_FOLDS = 3`)
- Reduce `n_iter` in hyperparameter tuning

**Accuracy not improving?**
- Check feature engineering in `data_prep.py` — add domain-specific features
- Increase hyperparameter tuning iterations
- Try ensemble methods (voting, stacking)
- Analyze confusion matrix — which class is being missed?

## 📝 Results Location

All outputs saved to `outputs/`:
- `model_comparison.csv` — Results table
- `metrics_comparison.png` — Visualization of model metrics
- `confusion_matrix_[model].png` — Confusion matrices
- `roc_curve_[model].png` — ROC curves
- `feature_importance_[model].png` — Top features

## 🎓 Notes

- **All decisions data-backed:** Every hyperparameter, feature drop, and model choice is validated against data
- **Production-ready:** Modular, extensible, well-documented code
- **Comprehensive:** Multiple models, ensembles, visualizations, metrics
- **Balanced:** Addresses class imbalance with SMOTE, not just raw accuracy

## 📚 References

- XGBoost: https://xgboost.readthedocs.io
- LightGBM: https://lightgbm.readthedocs.io
- CatBoost: https://catboost.ai
- SMOTE: https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html
- Stacking: https://scikit-learn.org/stable/modules/ensemble.html#stacking

---

**Author:** Tejas Sonawane (SVKM's NMIMS)  
**Dataset:** UCI 130-Hospitals Diabetes Dataset (10K records)  
**Target:** 75% accuracy on diabetes patient readmission prediction
