#!/usr/bin/env python3
"""
Diabetes Readmission Prediction — Streamlit Dashboard
======================================================
A premium web interface for uploading patient data, running predictions,
and evaluating model performance.

Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

warnings.filterwarnings('ignore')

# ── Project imports for preprocessing ──
from config import (
    TARGET_COLUMN, AGE_MAP, A1C_MAP, GLU_MAP, MED_CHANGE_MAP,
    MEDICATION_COLS, TOP_SPECIALTIES, DISCHARGE_MAP,
    ADMISSION_TYPE_MAP, ADMISSION_SOURCE_MAP, IMBALANCE_THRESHOLD, MODELS_DIR
)
from data_prep import (
    remove_impossible_readmissions, handle_missing_values,
    drop_low_value_columns, encode_medications, engineer_features
)

# ═══════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Diabetes Readmission Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════
#  CUSTOM CSS — Premium dark theme with glassmorphism
# ═══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global ── */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ── Main header ── */
    .main-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .main-header p {
        color: rgba(255,255,255,0.65);
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
        font-weight: 400;
    }

    /* ── Metric cards ── */
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.25);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 0.3rem;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 500;
        opacity: 0.7;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* ── Color variants ── */
    .metric-green .metric-value { color: #00e676; }
    .metric-red .metric-value { color: #ff5252; }
    .metric-blue .metric-value { color: #448aff; }
    .metric-orange .metric-value { color: #ffab40; }
    .metric-purple .metric-value { color: #b388ff; }
    .metric-cyan .metric-value { color: #18ffff; }

    /* ── Section headers ── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.6rem;
        border-bottom: 2px solid rgba(255,255,255,0.1);
    }
    .section-header h2 {
        font-size: 1.35rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.01em;
    }

    /* ── Glass panel ── */
    .glass-panel {
        background: linear-gradient(135deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }

    /* ── Prediction result cards ── */
    .prediction-result {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }
    .prediction-result h3 {
        font-size: 1.1rem;
        font-weight: 600;
        margin: 0 0 1rem 0;
        opacity: 0.8;
    }

    /* ── Sidebar styling ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 100%);
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #ffffff;
    }

    /* ── Status badges ── */
    .status-badge {
        display: inline-block;
        padding: 0.35rem 0.9rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }
    .badge-success {
        background: rgba(0,230,118,0.15);
        color: #00e676;
        border: 1px solid rgba(0,230,118,0.3);
    }
    .badge-warning {
        background: rgba(255,171,64,0.15);
        color: #ffab40;
        border: 1px solid rgba(255,171,64,0.3);
    }
    .badge-info {
        background: rgba(68,138,255,0.15);
        color: #448aff;
        border: 1px solid rgba(68,138,255,0.3);
    }

    /* ── Table styling ── */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    /* ── Button styling ── */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102,126,234,0.4);
    }

    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* ── Divider ── */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
#  MODEL LOADING (cached)
# ═══════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_model_and_artifacts():
    """Load the saved model and preprocessing artifacts."""
    model_path = os.path.join(MODELS_DIR, 'best_model.joblib')
    artifacts_path = os.path.join(MODELS_DIR, 'preprocessing_artifacts.joblib')
    comparison_path = os.path.join(MODELS_DIR, 'model_comparison.joblib')

    if not os.path.exists(model_path) or not os.path.exists(artifacts_path):
        return None, None, None

    model = joblib.load(model_path)
    artifacts = joblib.load(artifacts_path)

    comparison_df = None
    if os.path.exists(comparison_path):
        comparison_df = joblib.load(comparison_path)

    return model, artifacts, comparison_df


# ═══════════════════════════════════════════════════════════════════════
#  PREPROCESSING PIPELINE (mirrors data_prep.py logic)
# ═══════════════════════════════════════════════════════════════════════
def preprocess_uploaded_data(df_raw, artifacts):
    """
    Apply the full preprocessing pipeline to uploaded data.
    Reuses functions from data_prep.py and aligns features with training set.
    """
    df = df_raw.copy()

    # Check if target column exists (for evaluation)
    has_target = TARGET_COLUMN in df.columns
    y_true = None
    if has_target:
        y_true = df[TARGET_COLUMN]
        # Handle boolean/string target
        if y_true.dtype == 'bool':
            y_true = y_true.astype(int)
        elif y_true.dtype == 'object':
            y_true = y_true.map({'True': 1, 'False': 0, 'Yes': 1, 'No': 0,
                                  True: 1, False: 0}).fillna(0).astype(int)
        else:
            y_true = y_true.astype(int)

    # Apply the same data prep steps
    df = remove_impossible_readmissions(df)
    df = handle_missing_values(df)
    df = drop_low_value_columns(df)
    df = encode_medications(df)
    df = engineer_features(df)

    # Drop target if present
    if TARGET_COLUMN in df.columns:
        df = df.drop(columns=[TARGET_COLUMN])

    # One-hot encode
    cat_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)

    # Sanitize column names
    safe_cols = [re.sub(r'[\[\]<>]', '_', str(col)) for col in df.columns]
    df.columns = safe_cols

    # Align to training feature set
    train_dummies_cols = artifacts['feature_columns_after_dummies']
    df = df.reindex(columns=train_dummies_cols, fill_value=0)

    # KMeans archetypes
    scaler = artifacts['scaler']
    kmeans = artifacts['kmeans']
    X_scaled = scaler.transform(df)
    clusters = kmeans.predict(X_scaled)
    for i in range(5):
        df[f'archetype_{i}'] = (clusters == i).astype(int)

    # Final alignment
    final_cols = artifacts['final_feature_columns']
    df = df.reindex(columns=final_cols, fill_value=0)

    # Re-align y_true to the cleaned df index
    surviving_indices = df.index.tolist()
    if has_target and y_true is not None:
        y_true = y_true.loc[df.index].reset_index(drop=True)

    df = df.reset_index(drop=True)

    return df, y_true, surviving_indices


# ═══════════════════════════════════════════════════════════════════════
#  HELPER: Metric card HTML
# ═══════════════════════════════════════════════════════════════════════
def metric_card(value, label, color_class="metric-blue"):
    return f"""
    <div class="metric-card {color_class}">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """


# ═══════════════════════════════════════════════════════════════════════
#  MAIN APP
# ═══════════════════════════════════════════════════════════════════════
def main():
    # ── Load model ──
    model, artifacts, comparison_df = load_model_and_artifacts()

    # ══════════════════════════════════════════════════════════════════
    #  SIDEBAR
    # ══════════════════════════════════════════════════════════════════
    with st.sidebar:
        st.markdown("## 🏥 Diabetes Readmission")
        st.markdown("##### Prediction Dashboard")
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        # Model status
        if model is not None:
            st.markdown(
                '<span class="status-badge badge-success">● Model Loaded</span>',
                unsafe_allow_html=True
            )
            st.markdown(f"**Model:** {artifacts.get('model_name', 'Unknown')}")
            st.markdown(f"**Threshold:** `{artifacts.get('optimal_threshold', 0.5):.4f}`")
            st.markdown(f"**Features:** `{len(artifacts.get('final_feature_columns', []))}`")
        else:
            st.markdown(
                '<span class="status-badge badge-warning">⚠ No Model Found</span>',
                unsafe_allow_html=True
            )
            st.error("Run `python save_pipeline.py` first to train & save the model.")

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        # File upload
        st.markdown("### 📂 Upload Patient Data")
        uploaded_file = st.file_uploader(
            "Upload a CSV or Excel file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload patient records for prediction. The file should contain the same columns as the training data."
        )

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        # Info panel
        st.markdown("### ℹ️ About")
        st.markdown("""
        This dashboard uses an **F1-Optimized Extra Trees** model trained on 10,000 
        diabetes patient encounters to predict hospital readmission risk.
        
        **Pipeline:** Feature engineering → One-hot encoding → KMeans archetypes → 
        Extra Trees with optimized classification threshold.
        """)

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        st.caption("Built with Streamlit • Diabetes Readmission Prediction V4")

    # ══════════════════════════════════════════════════════════════════
    #  MAIN CONTENT
    # ══════════════════════════════════════════════════════════════════

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Diabetes Readmission Predictor</h1>
        <p>Upload patient data • Analyze distributions • Run predictions • Evaluate performance</p>
    </div>
    """, unsafe_allow_html=True)

    if uploaded_file is None:
        # ── Landing state ──
        st.markdown("""
        <div class="glass-panel" style="text-align: center; padding: 3rem 2rem;">
            <h2 style="font-size: 1.5rem; font-weight: 700; margin-bottom: 0.8rem;">
                👋 Welcome to the Prediction Dashboard
            </h2>
            <p style="opacity: 0.7; font-size: 1.05rem; max-width: 600px; margin: 0 auto;">
                Upload a patient dataset using the sidebar to get started. The system will 
                analyze the data, run predictions through the trained ML pipeline, and display 
                comprehensive results.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Show model comparison if available
        if comparison_df is not None:
            st.markdown("""
            <div class="section-header">
                <h2>🏆 Trained Model Performance</h2>
            </div>
            """, unsafe_allow_html=True)

            # Show top 10 models
            display_df = comparison_df.head(10)[['Model', 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']].copy()
            display_df.columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']

            # Format as percentages
            for col in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "—")

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
            )
        return

    # ══════════════════════════════════════════════════════════════════
    #  DATA LOADING
    # ══════════════════════════════════════════════════════════════════
    try:
        if uploaded_file.name.endswith(('.xlsx', '.xls')):
            df_raw = pd.read_excel(uploaded_file)
        else:
            df_raw = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"❌ Failed to read file: {e}")
        return

    has_target = TARGET_COLUMN in df_raw.columns

    # ── Quick stats row ──
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(metric_card(f"{len(df_raw):,}", "Total Patients", "metric-blue"), unsafe_allow_html=True)
    with col2:
        st.markdown(metric_card(f"{df_raw.shape[1]}", "Features", "metric-purple"), unsafe_allow_html=True)
    with col3:
        if has_target:
            readmit_rate = df_raw[TARGET_COLUMN].astype(int).mean()
            st.markdown(metric_card(f"{readmit_rate:.1%}", "Readmission Rate", "metric-orange"), unsafe_allow_html=True)
        else:
            st.markdown(metric_card("—", "Readmission Rate", "metric-orange"), unsafe_allow_html=True)
    with col4:
        missing_pct = df_raw.isnull().mean().mean()
        st.markdown(metric_card(f"{missing_pct:.1%}", "Missing Values", "metric-cyan"), unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════
    #  SECTION 1: DATA ANALYSIS & VISUALIZATION
    # ══════════════════════════════════════════════════════════════════
    st.markdown("""
    <div class="section-header">
        <h2>📊 Data Analysis & Visualization</h2>
    </div>
    """, unsafe_allow_html=True)

    tab_preview, tab_stats, tab_charts = st.tabs(["📋 Data Preview", "📈 Summary Statistics", "📊 Visual Charts"])

    with tab_preview:
        st.dataframe(df_raw.head(50), use_container_width=True, height=400)

    with tab_stats:
        col_num, col_cat = st.columns(2)
        with col_num:
            st.markdown("##### Numerical Features")
            numeric_df = df_raw.describe().T
            if not numeric_df.empty:
                st.dataframe(numeric_df.style.format("{:.2f}"), use_container_width=True, height=350)
            else:
                st.info("No numerical columns found.")

        with col_cat:
            st.markdown("##### Categorical Features")
            cat_cols_display = df_raw.select_dtypes(include=['object', 'bool']).columns.tolist()
            if cat_cols_display:
                cat_summary = []
                for col in cat_cols_display[:15]:
                    cat_summary.append({
                        'Feature': col,
                        'Unique': df_raw[col].nunique(),
                        'Top Value': str(df_raw[col].mode().iloc[0]) if len(df_raw[col].mode()) > 0 else '—',
                        'Missing': df_raw[col].isnull().sum(),
                    })
                st.dataframe(pd.DataFrame(cat_summary), use_container_width=True, hide_index=True, height=350)
            else:
                st.info("No categorical columns found.")

    with tab_charts:
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            # Age distribution
            if 'age' in df_raw.columns:
                st.markdown("##### Age Distribution")
                fig, ax = plt.subplots(figsize=(8, 4))
                age_order = ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
                             '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)']
                age_counts = df_raw['age'].value_counts().reindex(age_order).fillna(0)
                colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(age_counts)))
                bars = ax.bar(range(len(age_counts)), age_counts.values, color=colors, edgecolor='white', linewidth=0.5)
                ax.set_xticks(range(len(age_counts)))
                ax.set_xticklabels(age_counts.index, rotation=45, ha='right', fontsize=9)
                ax.set_ylabel('Count', fontsize=10)
                ax.set_title('Patient Age Distribution', fontsize=12, fontweight='bold')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close()

            # Admission type
            if 'admission_type_id' in df_raw.columns:
                st.markdown("##### Admission Type Distribution")
                fig, ax = plt.subplots(figsize=(8, 4))
                adm_counts = df_raw['admission_type_id'].value_counts().head(8)
                colors_adm = plt.cm.magma(np.linspace(0.2, 0.85, len(adm_counts)))
                ax.barh(range(len(adm_counts)), adm_counts.values, color=colors_adm, edgecolor='white', linewidth=0.5)
                ax.set_yticks(range(len(adm_counts)))
                ax.set_yticklabels(adm_counts.index, fontsize=9)
                ax.set_xlabel('Count', fontsize=10)
                ax.set_title('Admission Types', fontsize=12, fontweight='bold')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close()

        with chart_col2:
            # Medication usage
            med_cols_in_data = [c for c in MEDICATION_COLS if c in df_raw.columns]
            if med_cols_in_data:
                st.markdown("##### Medication Activity")
                fig, ax = plt.subplots(figsize=(8, 4))
                med_activity = {}
                for col in med_cols_in_data:
                    active = (df_raw[col] != 'No').sum() if df_raw[col].dtype == 'object' else (df_raw[col] > 0).sum()
                    med_activity[col.title()] = active
                med_series = pd.Series(med_activity).sort_values(ascending=True)
                colors_med = plt.cm.cool(np.linspace(0.2, 0.85, len(med_series)))
                ax.barh(range(len(med_series)), med_series.values, color=colors_med, edgecolor='white', linewidth=0.5)
                ax.set_yticks(range(len(med_series)))
                ax.set_yticklabels(med_series.index, fontsize=9)
                ax.set_xlabel('Patients with Active Prescriptions', fontsize=10)
                ax.set_title('Medication Activity Overview', fontsize=12, fontweight='bold')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close()

            # Diagnosis categories (primary)
            if 'diag_1' in df_raw.columns:
                st.markdown("##### Primary Diagnosis Categories")
                fig, ax = plt.subplots(figsize=(8, 4))
                from data_prep import map_diagnosis_code
                diag_cats = df_raw['diag_1'].apply(map_diagnosis_code)
                diag_counts = diag_cats.value_counts().head(10)
                colors_diag = plt.cm.plasma(np.linspace(0.15, 0.9, len(diag_counts)))
                wedges, texts, autotexts = ax.pie(
                    diag_counts.values,
                    labels=diag_counts.index,
                    autopct='%1.1f%%',
                    colors=colors_diag,
                    pctdistance=0.82,
                    startangle=140,
                    textprops={'fontsize': 8}
                )
                for autotext in autotexts:
                    autotext.set_fontsize(7)
                    autotext.set_fontweight('bold')
                ax.set_title('Primary Diagnosis Breakdown', fontsize=12, fontweight='bold')
                fig.tight_layout()
                st.pyplot(fig)
                plt.close()

            elif 'diag_1_desc' in df_raw.columns:
                st.markdown("##### Top Primary Diagnoses")
                diag_desc_counts = df_raw['diag_1_desc'].value_counts().head(10)
                st.dataframe(
                    diag_desc_counts.reset_index().rename(columns={'index': 'Diagnosis', 'diag_1_desc': 'Diagnosis', 'count': 'Count'}),
                    use_container_width=True,
                    hide_index=True
                )

    # ══════════════════════════════════════════════════════════════════
    #  SECTION 2: MODEL INFERENCE
    # ══════════════════════════════════════════════════════════════════
    st.markdown("""
    <div class="section-header">
        <h2>🔮 Model Inference</h2>
    </div>
    """, unsafe_allow_html=True)

    if model is None:
        st.warning("⚠️ No trained model found. Run `python save_pipeline.py` to train and save the model before running predictions.")
        return

    # Run predictions button
    if st.button("🚀 Run Predictions", type="primary", use_container_width=True):
        with st.spinner("Processing data through ML pipeline..."):
            try:
                X_processed, y_true, surviving_indices = preprocess_uploaded_data(df_raw, artifacts)

                # Predict probabilities
                y_proba = model.predict_proba(X_processed)[:, 1]
                threshold = artifacts.get('optimal_threshold', 0.5)
                y_pred = (y_proba >= threshold).astype(int)

                # Store results in session state
                st.session_state['y_pred'] = y_pred
                st.session_state['y_proba'] = y_proba
                st.session_state['surviving_indices'] = surviving_indices
                st.session_state['y_true'] = y_true
                st.session_state['threshold'] = threshold
                st.session_state['predictions_ready'] = True

            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
                import traceback
                st.code(traceback.format_exc())
                return

    # ── Show results if predictions exist ──
    if st.session_state.get('predictions_ready', False):
        y_pred = st.session_state['y_pred']
        y_proba = st.session_state['y_proba']
        y_true = st.session_state['y_true']
        threshold = st.session_state['threshold']

        n_readmitted = int(y_pred.sum())
        n_not_readmitted = int(len(y_pred) - n_readmitted)
        readmit_pct = n_readmitted / len(y_pred) * 100
        avg_risk = float(y_proba.mean())

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        # ── Prediction KPIs ──
        res_col1, res_col2, res_col3, res_col4 = st.columns(4)
        with res_col1:
            st.markdown(metric_card(f"{n_readmitted:,}", "Predicted Readmitted", "metric-red"), unsafe_allow_html=True)
        with res_col2:
            st.markdown(metric_card(f"{n_not_readmitted:,}", "Not Readmitted", "metric-green"), unsafe_allow_html=True)
        with res_col3:
            st.markdown(metric_card(f"{readmit_pct:.1f}%", "Readmission Rate", "metric-orange"), unsafe_allow_html=True)
        with res_col4:
            st.markdown(metric_card(f"{avg_risk:.3f}", "Avg Risk Score", "metric-purple"), unsafe_allow_html=True)

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        # ── Risk distribution ──
        pred_chart_col1, pred_chart_col2 = st.columns(2)

        with pred_chart_col1:
            st.markdown("##### Prediction Distribution")
            fig, ax = plt.subplots(figsize=(6, 4))
            categories = ['Not Readmitted', 'Readmitted']
            values = [n_not_readmitted, n_readmitted]
            colors_pred = ['#00e676', '#ff5252']
            bars = ax.bar(categories, values, color=colors_pred, edgecolor='white', linewidth=1, width=0.5)
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(values)*0.02,
                        f'{val:,}', ha='center', va='bottom', fontweight='bold', fontsize=12)
            ax.set_ylabel('Patients', fontsize=11)
            ax.set_title(f'Prediction Results (threshold={threshold:.3f})', fontsize=12, fontweight='bold')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

        with pred_chart_col2:
            st.markdown("##### Risk Score Distribution")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(y_proba, bins=50, color='#667eea', edgecolor='white', linewidth=0.5, alpha=0.85)
            ax.axvline(x=threshold, color='#ff5252', linestyle='--', linewidth=2,
                       label=f'Threshold ({threshold:.3f})')
            ax.set_xlabel('Predicted Probability', fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title('Risk Score Distribution', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

        # ── Downloadable results ──
        surviving_indices = st.session_state.get('surviving_indices', list(range(len(df_raw))))
        results_df = df_raw.iloc[surviving_indices].copy().reset_index(drop=True)
        results_df['predicted_readmission'] = ['Readmitted' if p == 1 else 'Not Readmitted' for p in y_pred]
        results_df['risk_score'] = y_proba.round(4)

        csv_data = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Predictions as CSV",
            data=csv_data,
            file_name="readmission_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # ══════════════════════════════════════════════════════════════
        #  SECTION 3: EVALUATION DASHBOARD (only if target exists)
        # ══════════════════════════════════════════════════════════════
        if y_true is not None and len(y_true) == len(y_pred):
            st.markdown("""
            <div class="section-header">
                <h2>📈 Evaluation Dashboard</h2>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(
                '<span class="status-badge badge-info">Target column detected — showing evaluation metrics</span>',
                unsafe_allow_html=True
            )
            st.markdown("")

            # Compute metrics
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            try:
                auc = roc_auc_score(y_true, y_proba)
            except Exception:
                auc = None

            # ── Metric cards ──
            eval_cols = st.columns(5)
            metrics_data = [
                (f"{acc:.2%}", "Accuracy", "metric-blue"),
                (f"{prec:.2%}", "Precision", "metric-purple"),
                (f"{rec:.2%}", "Recall", "metric-green"),
                (f"{f1:.2%}", "F1 Score", "metric-orange"),
                (f"{auc:.2%}" if auc else "—", "ROC AUC", "metric-cyan"),
            ]
            for col, (val, label, color) in zip(eval_cols, metrics_data):
                with col:
                    st.markdown(metric_card(val, label, color), unsafe_allow_html=True)

            st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

            # ── Confusion Matrix & Classification Report ──
            eval_chart_col1, eval_chart_col2 = st.columns(2)

            with eval_chart_col1:
                st.markdown("##### Confusion Matrix")
                cm = confusion_matrix(y_true, y_pred)
                labels = ['Not Readmitted', 'Readmitted']
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(
                    cm, annot=True, fmt='d', cmap='RdYlBu_r',
                    xticklabels=labels, yticklabels=labels,
                    linewidths=2, linecolor='white', ax=ax,
                    annot_kws={'size': 16, 'fontweight': 'bold'}
                )
                ax.set_title('Confusion Matrix', fontsize=13, fontweight='bold')
                ax.set_xlabel('Predicted Label', fontsize=11)
                ax.set_ylabel('True Label', fontsize=11)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close()

            with eval_chart_col2:
                st.markdown("##### Classification Report")
                report = classification_report(
                    y_true, y_pred,
                    target_names=['Not Readmitted', 'Readmitted'],
                    output_dict=True
                )
                report_df = pd.DataFrame(report).T
                report_df = report_df.round(4)
                st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True, height=250)

                # ROC Curve
                if auc is not None:
                    st.markdown("##### ROC Curve")
                    from sklearn.metrics import roc_curve
                    fpr, tpr, _ = roc_curve(y_true, y_proba)
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.plot(fpr, tpr, color='#667eea', lw=2.5,
                            label=f'Model (AUC = {auc:.3f})')
                    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random')
                    ax.fill_between(fpr, tpr, alpha=0.1, color='#667eea')
                    ax.set_xlabel('False Positive Rate', fontsize=11)
                    ax.set_ylabel('True Positive Rate', fontsize=11)
                    ax.set_title('ROC Curve', fontsize=12, fontweight='bold')
                    ax.legend(loc='lower right', fontsize=10)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close()


if __name__ == '__main__':
    main()
