"""
Advanced Ensemble Modeling Techniques V4

Strategies:
1. Soft Voting with optimized weights
2. Stacking with Logistic Regression meta-learner
3. Blending (Super Learner) using out-of-fold predictions
4. Calibrated probability averaging
5. Threshold optimization on all ensembles
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    VotingClassifier, StackingClassifier,
    ExtraTreesClassifier, GradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import minimize

from config import RANDOM_STATE, CV_FOLDS
from utils import evaluate_model, print_section


# ---------------------------------------------------------------------------
#  1.  Weighted Soft-Voting Ensemble
# ---------------------------------------------------------------------------
class OptimizedVotingEnsemble:
    """Soft-voting ensemble with weights optimized on validation data."""

    def __init__(self, trainers_dict):
        self.trainers = {k: v for k, v in trainers_dict.items() if v is not None}
        self.weights = None
        self.names = list(self.trainers.keys())

    # -- internal helpers --
    def _collect_probas(self, X):
        return np.column_stack([
            t.predict_proba(X) for t in self.trainers.values()
        ])

    def _blend(self, probas, weights):
        return probas @ weights

    # -- public API --
    def fit_weights(self, X_val, y_val):
        """Find weights that maximise F1 on validation set."""
        from sklearn.metrics import f1_score as sklearn_f1

        probas = self._collect_probas(X_val)
        n = probas.shape[1]

        def neg_f1(w):
            w = np.abs(w) / np.abs(w).sum()        # normalise
            blended = probas @ w
            preds = (blended >= 0.5).astype(int)
            return -sklearn_f1(y_val, preds)

        # optimise starting from equal weights
        x0 = np.ones(n) / n
        result = minimize(neg_f1, x0, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-5})
        raw = np.abs(result.x)
        self.weights = raw / raw.sum()

        print("  Optimized voting weights:")
        for name, w in zip(self.names, self.weights):
            print(f"    {name}: {w:.4f}")

    def predict_proba(self, X):
        probas = self._collect_probas(X)
        if self.weights is None:
            self.weights = np.ones(probas.shape[1]) / probas.shape[1]
        return self._blend(probas, self.weights)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


# ---------------------------------------------------------------------------
#  2.  Blending / Super Learner
# ---------------------------------------------------------------------------
class BlendingEnsemble:
    """
    Two-layer Super Learner:
        Layer 0  -  base models produce OOF predictions
        Layer 1  -  LogisticRegression trained on OOF outputs
    """

    def __init__(self, trainers_dict):
        self.trainers = {k: v for k, v in trainers_dict.items() if v is not None}
        self.meta = LogisticRegression(
            random_state=RANDOM_STATE, max_iter=1000, C=1.0
        )
        self.names = list(self.trainers.keys())

    def fit(self, X_train, y_train, n_splits=5):
        """Generate OOF predictions and train meta-learner."""
        print_section("Blending / Super Learner Ensemble")

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                              random_state=RANDOM_STATE)
        n = len(X_train)
        oof = np.zeros((n, len(self.trainers)))

        X_arr = X_train.values if hasattr(X_train, 'values') else X_train
        y_arr = y_train.values if hasattr(y_train, 'values') else y_train

        print(f"  Generating OOF predictions ({n_splits}-fold)...")

        for fold_idx, (trn_idx, val_idx) in enumerate(skf.split(X_arr, y_arr)):
            X_trn, X_val = X_arr[trn_idx], X_arr[val_idx]
            y_trn = y_arr[trn_idx]

            for i, (name, trainer) in enumerate(self.trainers.items()):
                # Clone the model and re-train on this fold
                import sklearn.base as skb
                cloned = skb.clone(trainer.best_model)
                cloned.fit(X_trn, y_trn)
                oof[val_idx, i] = cloned.predict_proba(X_val)[:, 1]

        # Train meta-learner on OOF stack
        self.meta.fit(oof, y_arr)
        meta_coefs = self.meta.coef_[0]
        print("  Meta-learner coefficients:")
        for name, c in zip(self.names, meta_coefs):
            print(f"    {name}: {c:.4f}")
        print("  Blending ensemble trained")

    def predict_proba(self, X):
        """Predict using base models -> meta-learner."""
        base_preds = np.column_stack([
            t.predict_proba(X) for t in self.trainers.values()
        ])
        return self.meta.predict_proba(base_preds)[:, 1]

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


# ---------------------------------------------------------------------------
#  3.  Calibrated Probability Averaging
# ---------------------------------------------------------------------------
class CalibratedEnsemble:
    """
    Calibrate each base model's probabilities (Platt scaling / isotonic)
    then average.  Calibration often fixes over-confident tree models.
    """

    def __init__(self, trainers_dict):
        self.trainers = {k: v for k, v in trainers_dict.items() if v is not None}
        self.calibrated_models = {}
        self.names = list(self.trainers.keys())

    def fit(self, X_train, y_train):
        """Calibrate each base model using 3-fold CV."""
        print_section("Calibrated Probability Ensemble")

        for name, trainer in self.trainers.items():
            cal = CalibratedClassifierCV(
                trainer.best_model, cv=3, method='sigmoid'
            )
            cal.fit(X_train, y_train)
            self.calibrated_models[name] = cal
            print(f"  Calibrated: {name}")

        print(f"  {len(self.calibrated_models)} models calibrated")

    def predict_proba(self, X):
        probas = np.column_stack([
            m.predict_proba(X)[:, 1] for m in self.calibrated_models.values()
        ])
        return probas.mean(axis=1)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


# ---------------------------------------------------------------------------
#  4.  Standard Voting / Stacking wrappers (kept for compatibility)
# ---------------------------------------------------------------------------
class EnsembleModels:
    """Standard sklearn ensemble wrappers."""

    @staticmethod
    def voting_ensemble(trainers_dict):
        print_section("Voting Ensemble Classifier")
        estimators = [
            (name, trainer.best_model)
            for name, trainer in trainers_dict.items()
            if trainer is not None
        ]
        if not estimators:
            print("No valid trainers!")
            return None

        voting = VotingClassifier(estimators=estimators,
                                  voting='soft', n_jobs=-1)
        print(f"  Created voting ensemble with {len(estimators)} models:")
        for name, _ in estimators:
            print(f"    - {name}")
        return voting

    @staticmethod
    def stacking_ensemble(base_trainers_dict, X_train, y_train):
        print_section("Stacking Classifier Ensemble")
        estimators = [
            (name, trainer.best_model)
            for name, trainer in base_trainers_dict.items()
            if trainer is not None
        ]
        if not estimators:
            print("No valid trainers!")
            return None

        meta = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
        stacking = StackingClassifier(
            estimators=estimators,
            final_estimator=meta,
            cv=StratifiedKFold(n_splits=5, shuffle=True,
                               random_state=RANDOM_STATE),
            n_jobs=-1,
        )
        print(f"  Created stacking with {len(estimators)} base models")
        print("  Meta-learner: LogisticRegression")
        stacking.fit(X_train, y_train)
        print("  Stacking ensemble trained")
        return stacking


# ---------------------------------------------------------------------------
#  5.  Threshold Tuner (unchanged)
# ---------------------------------------------------------------------------
class ThresholdTuner:
    """Threshold optimisation helpers."""

    @staticmethod
    def find_best_f1_threshold(y_true, y_proba):
        from sklearn.metrics import precision_recall_curve
        prec, rec, thresholds = precision_recall_curve(y_true, y_proba)
        f1 = 2 * prec * rec / (prec + rec)
        f1 = np.nan_to_num(f1)
        best_idx = np.argmax(f1)
        return thresholds[best_idx], f1[best_idx]

    @staticmethod
    def apply_threshold(y_proba, threshold):
        return (y_proba >= threshold).astype(int)
