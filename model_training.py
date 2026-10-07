"""
Single model training with hyperparameter tuning
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import randint, uniform
import warnings

from config import (
    RANDOM_STATE, CV_FOLDS, SCORING_METRIC, RANDOMSEARCH_ITERATIONS,
    XGBOOST_PARAMS, LIGHTGBM_PARAMS, CATBOOST_PARAMS, RF_PARAMS
)
from utils import evaluate_model, find_optimal_threshold, print_section

warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


class BaseModelTrainer:
    """Base class for model training."""
    
    def __init__(self, model_name, model, params_dist):
        self.model_name = model_name
        self.model = model
        self.params_dist = params_dist
        self.best_model = None
        self.cv_score = None
    
    def tune_hyperparameters(self, X_train, y_train, n_iter=RANDOMSEARCH_ITERATIONS):
        """Tune hyperparameters using RandomizedSearchCV."""
        print_section(f"Tuning {self.model_name} Hyperparameters ({n_iter} iterations)")
        
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        
        search = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=self.params_dist,
            n_iter=n_iter,
            scoring=SCORING_METRIC,
            cv=cv,
            verbose=1,
            n_jobs=-1,
            random_state=RANDOM_STATE
        )
        
        search.fit(X_train, y_train)
        self.best_model = search.best_estimator_
        self.cv_score = search.best_score_
        
        print(f"\n✓ Best CV {SCORING_METRIC.upper()}: {self.cv_score:.4f}")
        print("Best parameters:")
        for k, v in search.best_params_.items():
            print(f"  {k}: {v}")
        
        return self.best_model
    
    def train(self, X_train, y_train):
        """Train the tuned model."""
        if self.best_model is None:
            self.best_model = self.model
        
        self.best_model.fit(X_train, y_train)
        return self.best_model
    
    def predict(self, X_test):
        """Make predictions."""
        return self.best_model.predict(X_test)
    
    def predict_proba(self, X_test):
        """Predict probabilities."""
        if hasattr(self.best_model, 'predict_proba'):
            return self.best_model.predict_proba(X_test)[:, 1]
        else:
            raise ValueError(f"{self.model_name} does not support predict_proba")
    
    def get_feature_importance(self):
        """Get feature importance if available."""
        if hasattr(self.best_model, 'feature_importances_'):
            return self.best_model.feature_importances_
        # For Logistic Regression (in a pipeline)
        elif hasattr(self.best_model, 'named_steps') and hasattr(self.best_model.named_steps.get('logisticregression'), 'coef_'):
            return self.best_model.named_steps['logisticregression'].coef_[0]
        else:
            return None


def train_logistic_regression(X_train, y_train, tune=True):
    """Train Logistic Regression model with scaling."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    
    print_section("Logistic Regression")
    
    # Needs scaling!
    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            random_state=RANDOM_STATE, 
            class_weight='balanced',
            max_iter=1000
        )
    )
    
    # We only tune the classifier part
    lr_params = {
        'logisticregression__C': uniform(0.01, 10.0),
        'logisticregression__penalty': ['l2']
    }
    
    trainer = BaseModelTrainer('Logistic Regression', pipeline, lr_params)
    
    if tune:
        trainer.tune_hyperparameters(X_train, y_train, n_iter=20)
    trainer.train(X_train, y_train)
    
    return trainer


def compute_scale_pos_weight(y_train):
    """Compute scale_pos_weight for XGBoost based on class ratio."""
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    return neg / pos

def train_xgboost(X_train, y_train, tune=True):
    """Train XGBoost model (with class weights for recall)."""
    print_section("XGBoost Classifier")
    
    spw = compute_scale_pos_weight(y_train)
    print(f"  scale_pos_weight: {spw:.2f}")
    
    xgb = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method='hist',
        scale_pos_weight=spw
    )
    
    trainer = BaseModelTrainer('XGBoost', xgb, XGBOOST_PARAMS)
    
    if tune:
        trainer.tune_hyperparameters(X_train, y_train)
    trainer.train(X_train, y_train)
    
    return trainer


def train_lightgbm(X_train, y_train, tune=True):
    """Train LightGBM model."""
    if not HAS_LIGHTGBM:
        print("LightGBM not installed. Install with: pip install lightgbm")
        return None
    
    print_section("LightGBM Classifier")
    
    lgb_model = lgb.LGBMClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
        boosting_type='gbdt',
        metric='auc',
        is_unbalance=True
    )
    
    trainer = BaseModelTrainer('LightGBM', lgb_model, LIGHTGBM_PARAMS)
    
    if tune:
        trainer.tune_hyperparameters(X_train, y_train)
    trainer.train(X_train, y_train)
    
    return trainer


def train_catboost(X_train, y_train, tune=True):
    """Train CatBoost model."""
    if not HAS_CATBOOST:
        print("CatBoost not installed. Install with: pip install catboost")
        return None
    
    print_section("CatBoost Classifier")
    
    cb_model = cb.CatBoostClassifier(
        random_state=RANDOM_STATE,
        verbose=False,
        thread_count=-1,
        auto_class_weights='Balanced'
    )
    
    trainer = BaseModelTrainer('CatBoost', cb_model, CATBOOST_PARAMS)
    
    if tune:
        trainer.tune_hyperparameters(X_train, y_train)
    trainer.train(X_train, y_train)
    
    return trainer


def train_random_forest(X_train, y_train, tune=True):
    """Train Random Forest model."""
    print_section("Random Forest Classifier")
    
    rf = RandomForestClassifier(
        random_state=RANDOM_STATE, 
        n_jobs=-1,
        class_weight='balanced'
    )
    
    trainer = BaseModelTrainer('Random Forest', rf, RF_PARAMS)
    
    if tune:
        trainer.tune_hyperparameters(X_train, y_train, n_iter=40)
    trainer.train(X_train, y_train)
    
    return trainer


def train_gradient_boosting(X_train, y_train, tune=True):
    """Train Gradient Boosting model with tuning."""
    print_section("Gradient Boosting Classifier")
    
    gb = GradientBoostingClassifier(
        random_state=RANDOM_STATE,
    )
    
    gb_params = {
        'n_estimators': randint(100, 400),
        'max_depth': randint(3, 8),
        'learning_rate': uniform(0.01, 0.19),
        'subsample': uniform(0.7, 0.3),
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 5),
    }
    
    trainer = BaseModelTrainer('Gradient Boosting', gb, gb_params)
    
    if tune:
        trainer.tune_hyperparameters(X_train, y_train, n_iter=40)
    trainer.train(X_train, y_train)
    
    return trainer


def train_extra_trees(X_train, y_train, tune=True):
    """Train Extra Trees model (highly randomized trees for ensemble diversity)."""
    from sklearn.ensemble import ExtraTreesClassifier
    
    print_section("Extra Trees Classifier")
    
    et = ExtraTreesClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    et_params = {
        'n_estimators': randint(100, 400),
        'max_depth': randint(5, 20),
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 5),
        'max_features': ['sqrt', 'log2'],
    }
    
    trainer = BaseModelTrainer('Extra Trees', et, et_params)
    
    if tune:
        trainer.tune_hyperparameters(X_train, y_train, n_iter=40)
    trainer.train(X_train, y_train)
    
    return trainer


def evaluate_all_models(models_dict, X_test, y_test):
    """Evaluate all trained models."""
    print_section("Model Evaluation Summary")
    
    all_results = {}
    
    for model_name, trainer in models_dict.items():
        if trainer is None:
            continue
        
        y_pred = trainer.predict(X_test)
        y_proba = trainer.predict_proba(X_test)
        
        metrics = evaluate_model(y_test, y_pred, y_proba, model_name)
        all_results[model_name] = metrics
        
        print(f"\n{model_name}:")
        print(f"  Accuracy : {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall   : {metrics['recall']:.4f}")
        print(f"  F1 Score : {metrics['f1_score']:.4f}")
        if metrics['roc_auc']:
            print(f"  ROC AUC  : {metrics['roc_auc']:.4f}")
    
    return all_results


def optimize_threshold(trainer, X_test, y_test):
    """Find optimal classification threshold to maximize F1 Score (balances Precision/Recall)."""
    from sklearn.metrics import precision_recall_curve
    import numpy as np
    
    y_proba = trainer.predict_proba(X_test)
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    f1_scores = np.nan_to_num(f1_scores)
    
    best_idx = np.argmax(f1_scores)
    threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    print(f"\n✓ Optimal F1 threshold: {threshold:.4f}")
    print(f"  Max F1 Score: {best_f1:.4f}")
    
    y_pred_opt = (y_proba >= threshold).astype(int)
    metrics = evaluate_model(y_test, y_pred_opt, y_proba, f"{trainer.model_name} (Optimized)")
    
    return threshold, metrics, y_pred_opt
