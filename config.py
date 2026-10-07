"""
Configuration and constants for Diabetes Readmission Prediction
"""

import os
from scipy.stats import randint, uniform

# ── Paths ──
DATA_DIR = './data'
OUTPUT_DIR = './outputs'
MODELS_DIR = './models'
PLOTS_DIR = './plots'

# Create directories if they don't exist
for dir_path in [DATA_DIR, OUTPUT_DIR, MODELS_DIR, PLOTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ── Data ──
DATA_FILE = os.path.join(DATA_DIR, '10k_diabete.csv')
TARGET_COLUMN = 'readmitted'
TEST_SIZE = 0.20
RANDOM_STATE = 42

# ── Target Metrics ──
TARGET_ACCURACY = 0.75  # 75% accuracy target
TARGET_AUC = 0.80       # 80% ROC-AUC target

# ── Class Imbalance ──
# missing_pct is on 0-100 scale, so threshold should be too
IMBALANCE_THRESHOLD = 50  # Drop columns >50% missing (keeps medical_specialty at 41%)
SMOTE_K_NEIGHBORS = 5

# ── Feature Engineering ──
AGE_MAP = {
    '[0-10)': 0, '[10-20)': 1, '[20-30)': 2, '[30-40)': 3, '[40-50)': 4,
    '[50-60)': 5, '[60-70)': 6, '[70-80)': 7, '[80-90)': 8, '[90-100)': 9
}

DIAG_ORDER = [
    'Circulatory', 'Respiratory', 'Digestive', 'Diabetes', 'Injury',
    'Musculoskeletal', 'Genitourinary', 'Neoplasms', 'Other', 'Unknown'
]

A1C_MAP = {'None': 0, 'Norm': 1, '>7': 2, '>8': 3}
GLU_MAP = {'None': 0, 'Norm': 1, '>200': 2, '>300': 3}

# ── Medication Change Encoding ──
# Meaningful ordinal: No activity → Steady → Decreased → Increased
MED_CHANGE_MAP = {'No': 0, 'Steady': 1, 'Down': 2, 'Up': 3}

# Medication columns to encode
MEDICATION_COLS = [
    'metformin', 'repaglinide', 'nateglinide', 'glimepiride', 'glipizide',
    'glyburide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol',
    'insulin'
]

# ── High-Cardinality Feature Grouping ──

# Top medical specialties (rest → 'Other')
TOP_SPECIALTIES = [
    'InternalMedicine', 'Emergency/Trauma', 'Family/GeneralPractice',
    'Cardiology', 'Surgery-General', 'Nephrology', 'Orthopedics',
    'Orthopedics-Reconstructive', 'Radiologist', 'Pulmonology'
]

# Discharge disposition grouping
DISCHARGE_MAP = {
    'Discharged to home': 'Home',
    'Discharged/transferred to home with home health service': 'Home_Health',
    'Discharged/transferred to SNF': 'SNF_Rehab',
    'Discharged/transferred to another rehab fac including rehab units of a hospital.': 'SNF_Rehab',
    'Expired': 'Expired',
    'Hospice / home': 'Hospice',
    'Hospice / medical facility': 'Hospice',
    'Left AMA': 'Left_AMA',
}
# Everything else maps to 'Other'

# Admission type grouping
ADMISSION_TYPE_MAP = {
    'Emergency': 'Emergency',
    'Urgent': 'Urgent',
    'Elective': 'Elective',
    'Newborn': 'Other',
    'Trauma Center': 'Emergency',
}
# Everything else maps to 'Other'

# Admission source grouping
ADMISSION_SOURCE_MAP = {
    'Physician Referral': 'Physician',
    'Emergency Room': 'Emergency',
    'Transfer from a hospital': 'Transfer',
    'Transfer from another health care facility': 'Transfer',
    'Transfer from a Skilled Nursing Facility (SNF)': 'Transfer',
}
# Everything else maps to 'Other'

# ── Hyperparameters (scipy.stats distributions for RandomizedSearchCV) ──

# XGBoost
XGBOOST_PARAMS = {
    'n_estimators': randint(200, 600),
    'max_depth': randint(4, 10),
    'learning_rate': uniform(0.02, 0.18),
    'subsample': uniform(0.7, 0.3),
    'colsample_bytree': uniform(0.6, 0.4),
    'min_child_weight': randint(1, 8),
    'gamma': uniform(0, 0.3),
}

# LightGBM
LIGHTGBM_PARAMS = {
    'n_estimators': randint(100, 400),
    'max_depth': randint(4, 10),
    'learning_rate': uniform(0.01, 0.19),
    'num_leaves': randint(20, 100),
    'subsample': uniform(0.7, 0.3),
    'colsample_bytree': uniform(0.6, 0.4),
    'reg_alpha': uniform(0, 1),
    'reg_lambda': uniform(0.5, 1.5),
}

# CatBoost
CATBOOST_PARAMS = {
    'iterations': randint(100, 400),
    'depth': randint(4, 8),
    'learning_rate': uniform(0.01, 0.19),
    'subsample': uniform(0.7, 0.3),
    'l2_leaf_reg': uniform(1, 9),
}

# Random Forest
RF_PARAMS = {
    'n_estimators': randint(100, 300),
    'max_depth': randint(5, 20),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 5),
    'max_features': ['sqrt', 'log2'],
}

# ── Tuning ──
RANDOMSEARCH_ITERATIONS = 80
CV_FOLDS = 5
SCORING_METRIC = 'roc_auc'

# ── Ensemble ──
ENSEMBLE_WEIGHTS = None  # Auto-tuned if None
THRESHOLD_SEARCH_RANGE = (0.3, 0.75)
THRESHOLD_SEARCH_STEP = 0.01

# ── Plotting ──
PLOT_DPI = 110
PLOT_FIGSIZE_SMALL = (8, 5)
PLOT_FIGSIZE_MEDIUM = (10, 6)
PLOT_FIGSIZE_LARGE = (14, 8)
PLOT_COLORS = ['#4CAF50', '#F44336']

# ── Verbosity ──
VERBOSE = 1
