"""
Lightweight XGBoost calibration layer on top of the rule-based disruption risk.

We train a small gradient-boosted regressor on synthetic NER-shaped data at
startup (< 1 second). It maps raw signals (rule risk, rainfall, wind, terrain,
accessibility, incident density) to a learned "calibrated risk" and returns
real metrics (MAE-based accuracy, R^2, top feature) that the dashboard shows.

Everything is deterministic (fixed seed) so the demo is reproducible.
"""

from __future__ import annotations

import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Terrain passability multipliers used by the synthetic ground truth.
# Mountains are the hardest to keep open when rainfall spikes.
TERRAIN_CODE = {"Plains": 0, "Hills": 1, "Mountains": 2}
TERRAIN_WEIGHT = {0: 0.6, 1: 1.0, 2: 1.35}

FEATURE_NAMES = [
    "rule_risk",
    "rainfall_mm",
    "wind_kmph",
    "terrain",
    "accessibility",
    "incidents",
    "cnn_damage",
]

_STATE = {"model": None, "metrics": None}


def _synthesise(n: int = 900, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Build a synthetic NER dataset shaped like the routes fixtures."""
    rng = np.random.default_rng(seed)
    rule_risk = rng.uniform(10, 90, n)
    rainfall = rng.gamma(2.2, 22.0, n).clip(0, 220)  # monsoon-heavy
    wind = rng.normal(18, 9, n).clip(0, 55)
    terrain = rng.integers(0, 3, n)
    accessibility = rng.uniform(30, 90, n)
    incidents = rng.poisson(1.8, n).clip(0, 8)
    # CNN damage score correlates weakly with rainfall + incidents (a wet, damaged
    # corridor is more likely to yield a strong damage photo) plus its own noise.
    cnn_damage = (0.55 * rainfall / 220.0 * 100
                  + 6.0 * incidents
                  + rng.normal(0, 12, n)).clip(0, 100)
    terrain_w = np.vectorize(TERRAIN_WEIGHT.get)(terrain)
    # Ground truth = physically motivated blend + noise, clipped to [0, 100].
    true_risk = (
        0.50 * rule_risk
        + 0.16 * rainfall * terrain_w / 2.2
        + 0.9 * wind * terrain_w / 5.0
        + 3.5 * incidents * terrain_w
        - 0.30 * (accessibility - 60)
        + 0.18 * cnn_damage * terrain_w
        + rng.normal(0, 4.0, n)
    ).clip(0, 100)
    x = np.column_stack([rule_risk, rainfall, wind, terrain, accessibility, incidents, cnn_damage])
    return x, true_risk


def train() -> dict:
    """Fit the model once and cache metrics for the dashboard."""
    x, y = _synthesise()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=7)
    model = xgb.XGBRegressor(
        n_estimators=140,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.2,
        random_state=7,
        n_jobs=1,
        verbosity=0,
    )
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))
    importances = model.feature_importances_.tolist()
    top_idx = int(np.argmax(importances))
    _STATE["model"] = model
    _STATE["metrics"] = {
        # Accuracy is presented as "1 - MAE/100" so it reads like a percentage
        # against the 0-100 risk scale the operator already understands.
        "accuracy": round((1 - mae / 100.0) * 100, 1),
        "confidence": round(max(r2, 0.0), 2),
        "mae": round(mae, 2),
        "r2": round(r2, 2),
        "top_feature": FEATURE_NAMES[top_idx],
        "feature_importance": {name: round(float(imp), 3) for name, imp in zip(FEATURE_NAMES, importances)},
        "training_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "version": "xgb-1.0.0",
    }
    return _STATE["metrics"]


def calibrate(rule_risk: float, rainfall: float, wind: float, terrain: str, accessibility: float, incidents: int, cnn_damage: float = 0.0) -> float:
    """Return the learned calibrated risk (0-100) for a single route segment."""
    model = _STATE["model"]
    if model is None:
        train()
        model = _STATE["model"]
    features = np.array([[rule_risk, rainfall, wind, TERRAIN_CODE.get(terrain, 1), accessibility, incidents, cnn_damage]])
    pred = float(model.predict(features)[0])
    return round(max(0.0, min(100.0, pred)), 1)


def metrics() -> dict:
    """Expose the cached model metrics for the dashboard."""
    if _STATE["metrics"] is None:
        train()
    return _STATE["metrics"]
