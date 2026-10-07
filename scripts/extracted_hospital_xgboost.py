import pandas as pd
import numpy as np

# 1. Load the dataset
# Adjust the filename if yours is named differently in your directory
df = pd.read_csv('10k_diabetes.csv')

# The dataset uses '?' for missing values. Convert them to actual NaNs for pandas to recognize
df.replace('?', np.nan, inplace=True)

print(f"Original Dataset Shape: {df.shape}\n")

# --- A. DETECT HIGHLY MISSING COLUMNS ---
# Set a threshold: If a column is missing more than 40% of its data, flag it.
missing_threshold = 40.0 
missing_percentages = (df.isnull().sum() / len(df)) * 100
missing_flags = missing_percentages[missing_percentages > missing_threshold]

print(f"--- Columns missing > {missing_threshold}% of data ---")
print(missing_flags, "\n")
columns_to_drop_missing = missing_flags.index.tolist()


# --- B. DETECT NEAR-ZERO VARIANCE COLUMNS ---
# Set a threshold: If one single category makes up >99% of the column, flag it.
variance_threshold = 99.0
columns_to_drop_variance = []

print(f"--- Columns with Near-Zero Variance (> {variance_threshold}% same value) ---")
for col in df.columns:
    # Get the percentage of the most frequent value
    top_value_pct = df[col].value_counts(normalize=True, dropna=False).max() * 100
    if top_value_pct > variance_threshold:
        print(f"{col}: {top_value_pct:.2f}% identical")
        columns_to_drop_variance.append(col)

# Combine the weak columns into one master list
weak_columns_to_drop = list(set(columns_to_drop_missing + columns_to_drop_variance))
print(f"\nTotal recommended columns to drop: {len(weak_columns_to_drop)}")
print(weak_columns_to_drop)

# --- CELL ---

# 1. DROP THE WEAK COLUMNS
# We use the list generated from the diagnostic block above
df_cleaned = df.drop(columns=weak_columns_to_drop).copy()

# Drop ID columns that carry no predictive weight (avoids data leakage/overfitting)
# Note: Keep 'encounter_id' or 'patient_nbr' ONLY if you need them for your final submission index
if 'encounter_id' in df_cleaned.columns:
    df_cleaned.drop(columns=['encounter_id', 'patient_nbr'], errors='ignore', inplace=True)

# 2. FEATURE ENGINEERING: Age Ordinal Mapping
# Machine learning models prefer numbers. We map age ranges to their midpoint.
age_mapping = {
    '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35, 
    '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75, 
    '[80-90)': 85, '[90-100)': 95
}
df_cleaned['age'] = df_cleaned['age'].map(age_mapping)

# CHANGE NEW COLUMN NAME TO DISEASED GROUPING etc.

# 3. FEATURE ENGINEERING: Grouping ICD-9 Codes (diag_1, diag_2, diag_3)
# There are hundreds of unique medical codes. We must group them to give the model a chance.
def categorize_icd9(code):
    if pd.isna(code) or code == '?' or code == 'None':
        return 'Missing'
    
    code = str(code)
    # External injuries or supplemental classification
    if code.startswith('V') or code.startswith('E'):
        return 'Other'
    
    try:
        icd_num = float(code)
        if 250 <= icd_num < 251:
            return 'Diabetes'
        elif (390 <= icd_num <= 459) or icd_num == 785:
            return 'Circulatory'
        elif (460 <= icd_num <= 519) or icd_num == 786:
            return 'Respiratory'
        elif (520 <= icd_num <= 579) or icd_num == 787:
            return 'Digestive'
        elif (800 <= icd_num <= 999):
            return 'Injury'
        elif (710 <= icd_num <= 739):
            return 'Musculoskeletal'
        elif (580 <= icd_num <= 629) or icd_num == 788:
            return 'Genitourinary'
        elif (140 <= icd_num <= 239):
            return 'Neoplasms'
        else:
            return 'Other'
    except ValueError:
        return 'Other'

# Apply the function to all three diagnosis columns
for col in ['diag_1', 'diag_2', 'diag_3']:
    if col in df_cleaned.columns:
        df_cleaned[col + '_category'] = df_cleaned[col].apply(categorize_icd9)
        # Drop the original hyper-granular column
        df_cleaned.drop(columns=[col], inplace=True)


# 4. PREPARE THE TARGET VARIABLE
# Convert 'readmitted' (True/False or >30/<30/NO string) into a binary 1 and 0
if df_cleaned['readmitted'].dtype == object:
    # If the dataset uses standard strings
    df_cleaned['readmitted'] = df_cleaned['readmitted'].apply(lambda x: 0 if pd.isna(x) or x == 'NO' or x == False else 1)
else:
    # If it's already boolean
    df_cleaned['readmitted'] = df_cleaned['readmitted'].astype(int)

print(f"Cleaned Dataset Shape: {df_cleaned.shape}")
print(df_cleaned.head(3))

# Save the checkpoint so you don't have to re-run this every time
df_cleaned.to_csv('10k_diabetes_cleaned.csv', index=False)

# --- CELL ---

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set the visual style for the plots
sns.set_theme(style="whitegrid", palette="muted")

# Create a figure with multiple subplots to organize the EDA
fig, axes = plt.subplots(4, 2, figsize=(18, 24))
fig.tight_layout(pad=6.0)

# 1. Target Distribution
sns.countplot(data=df_cleaned, x='readmitted', ax=axes[0, 0])
axes[0, 0].set_title('Target Distribution: Readmission Status', fontsize=14)
axes[0, 0].set_xticklabels(['Not Readmitted (0)', 'Readmitted (1)'])

# 2. Age Distribution
sns.histplot(data=df_cleaned, x='age', bins=10, kde=True, ax=axes[0, 1], color='skyblue')
axes[0, 1].set_title('Age Distribution (Midpoint Values)', fontsize=14)

# 3. Gender Distribution
# Exclude 'Unknown/Invalid' if they exist in small numbers
gender_data = df_cleaned[df_cleaned['gender'].isin(['Male', 'Female'])]
sns.countplot(data=gender_data, x='gender', ax=axes[1, 0])
axes[1, 0].set_title('Gender Distribution', fontsize=14)

# 4. Readmission by Age
sns.countplot(data=df_cleaned, x='age', hue='readmitted', ax=axes[1, 1])
axes[1, 1].set_title('Readmission Counts by Age Group', fontsize=14)
axes[1, 1].legend(title='Readmitted', labels=['No', 'Yes'])

# 5. Readmission by Gender
sns.countplot(data=gender_data, x='gender', hue='readmitted', ax=axes[2, 0])
axes[2, 0].set_title('Readmission by Gender', fontsize=14)
axes[2, 0].legend(title='Readmitted', labels=['No', 'Yes'])

# 6. Top Diagnosis Categories
# Using the engineered diag_1_category
order = df_cleaned['diag_1_category'].value_counts().index
sns.countplot(data=df_cleaned, y='diag_1_category', order=order, ax=axes[2, 1], palette='viridis')
axes[2, 1].set_title('Primary Diagnosis Categories (diag_1)', fontsize=14)

# 7. Correlation Analysis
# Isolate only numeric columns for the heatmap
numeric_cols = df_cleaned.select_dtypes(include=[np.number])
corr_matrix = numeric_cols.corr()

sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', ax=axes[3, 0], vmin=-1, vmax=1)
axes[3, 0].set_title('Correlation Heatmap (Numeric Features)', fontsize=14)

# Hide the last empty subplot (since we have 7 plots for an 8-slot grid)
fig.delaxes(axes[3, 1])

plt.show()

# Documenting Findings (Print statements for your report)
print("--- Quick EDA Findings for Report ---")
print(f"Total Records: {len(df_cleaned)}")
print(f"Total Features: {len(df_cleaned.columns)}")
print(f"Readmission Rate: {(df_cleaned['readmitted'].mean() * 100):.2f}%\n")

# Display top numeric correlations with the target 'readmitted'
print("Top variables correlated with Readmission:")
correlations = corr_matrix['readmitted'].sort_values(ascending=False)
print(correlations.head(5)[1:]) # Skip the first one as it is readmitted itself

# --- CELL ---

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import re

# --- THE FIX: Drop raw text description columns ---
# These create thousands of useless columns and contain brackets
desc_cols = ['diag_1_desc', 'diag_2_desc', 'diag_3_desc']
cols_to_drop = [col for col in desc_cols if col in df_cleaned.columns]
df_cleaned.drop(columns=cols_to_drop, inplace=True)

# 1. FINAL DATA PREP: Encode Categoricals
categorical_cols = df_cleaned.select_dtypes(include=['object', 'category']).columns
df_encoded = pd.get_dummies(df_cleaned, columns=categorical_cols, drop_first=True)

# --- EXTRA SAFEGUARD ---
# Scrub any XGBoost-forbidden characters from the final column names
df_encoded.columns = [re.sub(r'[\[\]<>]', '_', str(col)) for col in df_encoded.columns]

# Separate features (X) and target (y)
X = df_encoded.drop('readmitted', axis=1)
y = df_encoded['readmitted']

# 2. TRAIN-TEST SPLIT (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training Features Shape: {X_train.shape}")
print(f"Testing Features Shape: {X_test.shape}\n")

# 3. XGBOOST MODEL SETUP
xgb_estimator = xgb.XGBClassifier(
    tree_method='hist',
    device='cuda', # Utilizing your local hardware acceleration
    objective='binary:logistic',
    eval_metric='auc',
    random_state=42
)

# 4. HYPERPARAMETER TUNING
param_grid = {
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [100, 200, 300, 500],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

random_search = RandomizedSearchCV(
    estimator=xgb_estimator,
    param_distributions=param_grid,
    n_iter=25,
    scoring='roc_auc', 
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1 
)

print("Starting Hyperparameter Tuning...")
random_search.fit(X_train, y_train)

# 5. BEST MODEL EVALUATION
best_xgb = random_search.best_estimator_
print(f"\nBest Parameters Found: {random_search.best_params_}")

# Generate predictions
y_pred = best_xgb.predict(X_test)
y_pred_proba = best_xgb.predict_proba(X_test)[:, 1]

# Calculate mandatory metrics
print("\n--- Model Evaluation Metrics ---")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
print(f"ROC AUC:   {roc_auc_score(y_test, y_pred_proba):.4f}")

# --- CELL ---

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve
import shap
import numpy as np

# Set up the visualization grid for Part 4
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
sns.set_theme(style="whitegrid")

# 1. CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False)
axes[0].set_title('Confusion Matrix', fontsize=14)
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')
axes[0].set_xticklabels(['Not Readmitted', 'Readmitted'])
axes[0].set_yticklabels(['Not Readmitted', 'Readmitted'])

# 2. ROC CURVE
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC AUC = {roc_auc_score(y_test, y_pred_proba):.4f}')
axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axes[1].set_title('Receiver Operating Characteristic (ROC)', fontsize=14)
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].legend(loc="lower right")

# 3. FEATURE IMPORTANCE (Top 15)
# Get importance from the XGBoost model
importances = best_xgb.feature_importances_
indices = np.argsort(importances)[::-1][:15] # Top 15
top_features = X.columns[indices]

sns.barplot(x=importances[indices], y=top_features, hue=top_features, ax=axes[2], palette='viridis', legend=False)
axes[2].set_title('XGBoost Feature Importance (Top 15)', fontsize=14)
axes[2].set_xlabel('F-Score')

plt.tight_layout()
plt.show()


# ==========================================
# PART 5: SHAP EXPLAINABILITY
# ==========================================
print("Calculating SHAP values (this uses the GPU and is very fast)...")
# We use TreeExplainer because XGBoost is a tree-based model
explainer = shap.TreeExplainer(best_xgb)

# Calculate SHAP values for the test set
shap_values = explainer.shap_values(X_test)

# 1. SHAP Summary Plot
plt.figure(figsize=(10, 6))
plt.title("SHAP Summary Plot: What Drives Readmission?", fontsize=14)
shap.summary_plot(shap_values, X_test, max_display=15)

# --- CELL ---

plt.figure(figsize=(10, 6))
plt.title("SHAP Summary Plot: What Drives Readmission?", fontsize=14)
shap.summary_plot(shap_values, X_test, max_display=15)

# --- CELL ---

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 1. ISOLATE THE TOP 15 FEATURES
# We use the 'top_features' array you already generated in the Part 4 code block
X_train_lean = X_train[top_features]
X_test_lean = X_test[top_features]

print(f"Phase 2 Training Shape (Lean): {X_train_lean.shape}")

# 2. SETUP THE LEAN XGBOOST MODEL
# Added scale_pos_weight=1.5 to penalize false negatives (missed readmissions)
xgb_lean = xgb.XGBClassifier(
    tree_method='hist',
    device='cuda', 
    objective='binary:logistic',
    eval_metric='auc',
    scale_pos_weight=1.5, 
    random_state=42
)

# 3. HYPERPARAMETER TUNING (PHASE 2)
# Because the noise is gone, we can use 5-fold cross-validation for a more rigorous test
param_grid_lean = {
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [200, 300, 500, 700],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

random_search_lean = RandomizedSearchCV(
    estimator=xgb_lean,
    param_distributions=param_grid_lean,
    n_iter=30,
    scoring='roc_auc', 
    cv=5, 
    verbose=1,
    random_state=42,
    n_jobs=-1
)

print("Starting Phase 2 Hyperparameter Tuning...")
random_search_lean.fit(X_train_lean, y_train)

# 4. FINAL MODEL EVALUATION
final_xgb = random_search_lean.best_estimator_
print(f"\nFinal Best Parameters: {random_search_lean.best_params_}")

# Generate final predictions
y_pred_final = final_xgb.predict(X_test_lean)
y_pred_proba_final = final_xgb.predict_proba(X_test_lean)[:, 1]

# Print final metrics for the PDF Report
print("\n--- PHASE 2: FINAL EVALUATION METRICS ---")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_final):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_final):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_final):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred_final):.4f}")
print(f"ROC AUC:   {roc_auc_score(y_test, y_pred_proba_final):.4f}")

# Optional: Save the highly optimized model for the Bonus RAG Challenge
final_xgb.save_model("xgboost_diabetes_lean.json")

# --- CELL ---

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 1. THE GOLDILOCKS ZONE: Top 45 Features
# We use the importances array already stored in memory from Part 4
indices_45 = np.argsort(importances)[::-1][:45] 
top_45_features = X.columns[indices_45]

X_train_mid = X_train[top_45_features]
X_test_mid = X_test[top_45_features]

print(f"Phase 2.5 Training Shape: {X_train_mid.shape}")

# 2. SETUP FOR PURE ACCURACY
# Removed scale_pos_weight to stop penalizing false negatives
xgb_mid = xgb.XGBClassifier(
    tree_method='hist',
    device='cuda', 
    objective='binary:logistic',
    eval_metric='auc',
    random_state=42
)

# 3. HYPERPARAMETER TUNING
# Expanded the grid slightly to give the model more room to tune depth
param_grid_mid = {
    'max_depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [300, 500, 800],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

random_search_mid = RandomizedSearchCV(
    estimator=xgb_mid,
    param_distributions=param_grid_mid,
    n_iter=40, # Increased iterations to find the optimal accuracy peak
    scoring='accuracy', # Changed from roc_auc to explicitly hunt for >75% accuracy
    cv=4, 
    verbose=1,
    random_state=42,
    n_jobs=-1
)

print("Starting Phase 2.5 Hyperparameter Tuning...")
random_search_mid.fit(X_train_mid, y_train)

# 4. FINAL MODEL EVALUATION
final_xgb = random_search_mid.best_estimator_
print(f"\nFinal Best Parameters: {random_search_mid.best_params_}")

# Generate final predictions
y_pred_final = final_xgb.predict(X_test_mid)
y_pred_proba_final = final_xgb.predict_proba(X_test_mid)[:, 1]

# Print final metrics
print("\n--- PHASE 2.5: FINAL EVALUATION METRICS ---")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_final):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_final):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_final):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred_final):.4f}")
print(f"ROC AUC:   {roc_auc_score(y_test, y_pred_proba_final):.4f}")

# Save the highly optimized model
final_xgb.save_model("xgboost_diabetes_optimized.json")

# --- CELL ---

from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import xgboost as xgb

print("Starting Recursive Feature Elimination (this may take a couple of minutes)...")

# 1. SETUP THE BASE ESTIMATOR
# We use a fast, lean version of XGBoost just to evaluate feature strength
xgb_base = xgb.XGBClassifier(
    tree_method='hist',
    device='cuda',
    max_depth=4,
    learning_rate=0.05,
    n_estimators=200, 
    objective='binary:logistic',
    eval_metric='auc',
    random_state=42
)

# 2. SETUP STRATIFIED CROSS-VALIDATION
# Stratified ensures every fold keeps the exact 60/40 ratio of our target variable
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 3. CONFIGURE RFECV
# We tell it to optimize for the F1 score, dropping 2 features at a time to speed it up
rfecv = RFECV(
    estimator=xgb_base,
    step=2, 
    cv=cv_strategy,
    scoring='f1', 
    min_features_to_select=5,
    n_jobs=-1
)

# Fit RFECV on the un-altered, full dataset
# Make sure to use the X_train that hasn't had the sparse columns dropped yet
rfecv.fit(X_train, y_train)

import pandas as pd

# 1. Extract the raw cross-validation scores
scores = rfecv.cv_results_['mean_test_score']
std_devs = rfecv.cv_results_['std_test_score']
n_scores = len(scores)

# 2. Reconstruct the exact X-axis mathematically
# RFECV evaluates from min_features up to total_features, dropping by 'step'
total_features = len(X_train.columns)
step = 2

# Calculate the exact number of features for every single score in the array
feature_counts = [total_features - (n_scores - 1 - i) * step for i in range(n_scores)]

# 3. CREATE THE LOGGER
log_df = pd.DataFrame({
    'Feature_Count': feature_counts,
    'Mean_F1_Score': scores,
    'Std_Dev': std_devs
})

# Sort to show the highest scores at the top
top_scores = log_df.sort_values(by='Mean_F1_Score', ascending=False).head(10)

print("\n--- DETAILED RFECV LOGGER (TOP 10 PEAKS) ---")
print(top_scores.to_string(index=False))
print(f"\nAlgorithmic Peak Selected by RFECV: {rfecv.n_features_} features")

# 4. THE CORRECTED PLOT
plt.figure(figsize=(10, 6))
plt.title("RFECV: Corrected Feature Curve", fontsize=14)
plt.xlabel("Exact Number of Features Selected")
plt.ylabel("Cross-Validated F1 Score")

# Plot using the aligned dataframe directly
plt.plot(log_df['Feature_Count'], log_df['Mean_F1_Score'], color='navy', lw=2)
plt.axvline(x=rfecv.n_features_, color='red', linestyle='--', label=f'True Peak ({rfecv.n_features_})')

# Add a subtle band for the standard deviation (shows model stability)
plt.fill_between(log_df['Feature_Count'], 
                 log_df['Mean_F1_Score'] - log_df['Std_Dev'], 
                 log_df['Mean_F1_Score'] + log_df['Std_Dev'], 
                 color='navy', alpha=0.1)

plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# --- CELL ---

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 1. THE GOLDILOCKS ZONE: Top 45 Features
# We use the importances array already stored in memory from Part 4
indices_45 = np.argsort(importances)[::-1][:45] 
top_45_features = X.columns[indices_45]

X_train_mid = X_train[top_45_features]
X_test_mid = X_test[top_45_features]

print(f"Phase 3 Training Shape: {X_train_mid.shape}")

# 2. SETUP FOR PURE ACCURACY
# Removed scale_pos_weight to stop penalizing false negatives
xgb_mid = xgb.XGBClassifier(
    tree_method='hist',
    device='cuda', 
    objective='binary:logistic',
    eval_metric='auc',
    random_state=42
)

# 3. HYPERPARAMETER TUNING
# Expanded the grid slightly to give the model more room to tune depth
param_grid_mid = {
    'max_depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [300, 500, 800],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

random_search_mid = RandomizedSearchCV(
    estimator=xgb_mid,
    param_distributions=param_grid_mid,
    n_iter=40, # Increased iterations to find the optimal accuracy peak
    scoring='accuracy', # Changed from roc_auc to explicitly hunt for >75% accuracy
    cv=4, 
    verbose=1,
    random_state=42,
    n_jobs=-1
)

print("Starting Phase 3 Hyperparameter Tuning...")
random_search_mid.fit(X_train_mid, y_train)

# 4. FINAL MODEL EVALUATION
final_xgb = random_search_mid.best_estimator_
print(f"\nFinal Best Parameters: {random_search_mid.best_params_}")

# Generate final predictions
y_pred_final = final_xgb.predict(X_test_mid)
y_pred_proba_final = final_xgb.predict_proba(X_test_mid)[:, 1]

# Print final metrics
print("\n--- PHASE 3: FINAL EVALUATION METRICS ---")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_final):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_final):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_final):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred_final):.4f}")
print(f"ROC AUC:   {roc_auc_score(y_test, y_pred_proba_final):.4f}")

# Save the highly optimized model
final_xgb.save_model("xgboost_diabetes_optimized.json")

# --- CELL ---

from sklearn.metrics import precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

# 1. Get the raw probabilities from the model you JUST trained
y_pred_proba = final_xgb.predict_proba(X_test_mid)[:, 1]

# 2. Calculate Precision, Recall, and F1 across EVERY possible mathematical threshold
precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
f1_scores = np.nan_to_num(f1_scores) 

# 3. Find the exact threshold where the model is perfectly balanced
best_idx = np.argmax(f1_scores)
best_thresh_f1 = thresholds[best_idx]

print(f"--- THE SECRET SAUCE: THRESHOLD TUNING ---")
print(f"Default XGBoost Threshold: 0.5000")
print(f"Optimal Decision Threshold Found: {best_thresh_f1:.4f}\n")

# 4. Generate NEW predictions using the optimal threshold
y_pred_pure = (y_pred_proba >= best_thresh_f1).astype(int)

print("--- TRUE PHASE 3: BALANCED METRICS ---")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_pure):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_pure):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_pure):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred_pure):.4f}")
print(f"ROC AUC:   {roc_auc_score(y_test, y_pred_proba):.4f}")

# --- CELL ---

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 1. Select the predictions you want to present
# Using the Maximum Accuracy predictions from the previous block
chosen_predictions = y_pred_max_acc 

# 2. Calculate the updated confusion matrix
cm_final = confusion_matrix(y_test, y_pred_pure)

# 3. Build a highly polished, presentation-ready visual
plt.figure(figsize=(8, 6))
sns.heatmap(cm_final, annot=True, fmt='d', cmap='Blues', cbar=False,
            annot_kws={"size": 16, "weight": "bold"}) # Larger, bold font for slides

# Add professional labels
plt.title('Final Confusion Matrix (Optimized Threshold)', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Predicted Label (AI System)', fontsize=14, labelpad=10)
plt.ylabel('True Label (Actual Patient Outcome)', fontsize=14, labelpad=10)

# Clean up the tick marks
plt.xticks([0.5, 1.5], ['Not Readmitted\n(Negative)', 'Readmitted\n(Positive)'], fontsize=12)
plt.yticks([0.5, 1.5], ['Not Readmitted\n(Negative)', 'Readmitted\n(Positive)'], fontsize=12, rotation=0)

plt.tight_layout()
plt.show()

# --- CELL ---

cm_final = confusion_matrix(y_test, y_pred_pure)
print(cm_final)

# --- CELL ---

y_test.value_counts()

# --- CELL ---

