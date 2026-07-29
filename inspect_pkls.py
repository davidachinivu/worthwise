"""
Phase 1 (revised): Inspect all five .pkl files + load impulse_booster.json.
impulse_model_final.pkl is discarded — the Booster is loaded via xgb native format.
"""

import joblib
import pprint
import numpy as np
import pandas as pd
import xgboost as xgb

PKL_DIR = r"c:\Users\david\OneDrive\Desktop\WorthWise"

# ── 1. Model columns (full ordered list) ──────────────────────────────────────
model_columns = joblib.load(f"{PKL_DIR}/impulse_model_columns_final.pkl")
print("=" * 70)
print("impulse_model_columns_final.pkl")
print(f"  Type : {type(model_columns)}")
print(f"  Count: {len(model_columns)}")
print("  Contents:")
pprint.pprint(model_columns)

# ── 2. Numeric columns (subset that needs scaling) ────────────────────────────
numeric_cols = joblib.load(f"{PKL_DIR}/impulse_numeric_cols_final.pkl")
print("\n" + "=" * 70)
print("impulse_numeric_cols_final.pkl")
print(f"  Type : {type(numeric_cols)}")
print(f"  Count: {len(numeric_cols)}")
print("  Contents:")
pprint.pprint(numeric_cols)

# ── 3. Decision threshold ─────────────────────────────────────────────────────
threshold = joblib.load(f"{PKL_DIR}/impulse_decision_threshold_final.pkl")
print("\n" + "=" * 70)
print("impulse_decision_threshold_final.pkl")
print(f"  Type : {type(threshold)}")
print(f"  Value: {threshold}")

# ── 4. Scaler ─────────────────────────────────────────────────────────────────
scaler = joblib.load(f"{PKL_DIR}/impulse_scaler_final.pkl")
print("\n" + "=" * 70)
print("impulse_scaler_final.pkl")
print(f"  Type             : {type(scaler)}")
print(f"  n_features_in_   : {scaler.n_features_in_}")
print(f"  feature_names_in_: {list(scaler.feature_names_in_)}")
print(f"  mean_ (first 5)  : {scaler.mean_[:5]}")
print(f"  scale_(first 5)  : {scaler.scale_[:5]}")

# ── 5. Booster (JSON native format) ──────────────────────────────────────────
print("\n" + "=" * 70)
print("impulse_booster.json  (xgb.Booster native format)")
booster = xgb.Booster()
booster.load_model(f"{PKL_DIR}/impulse_booster.json")
print(f"  Type             : {type(booster)}")
print(f"  num_boosting_rounds: {booster.num_boosted_rounds()}")
attrs = booster.attributes()
print(f"  Attributes       : {attrs}")

# ── 6. Smoke-test prediction with a dummy all-zeros row ───────────────────────
print("\n" + "=" * 70)
print("Smoke-test: predict on a dummy all-zeros row")
dummy = pd.DataFrame(np.zeros((1, len(model_columns))), columns=model_columns)
dmatrix = xgb.DMatrix(dummy)
raw_pred = booster.predict(dmatrix)
print(f"  raw_prediction shape : {raw_pred.shape}")
print(f"  raw_prediction value : {raw_pred}")
print(f"  (probability of positive class = {float(raw_pred[0]):.4f})")
print(f"  threshold            : {threshold}")
print(f"  decision             : {'HIGH RISK' if float(raw_pred[0]) >= threshold else 'LOW RISK'}")
print("=" * 70)
print("ALL FILES LOADED SUCCESSFULLY")
