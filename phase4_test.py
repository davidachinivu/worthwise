"""
Phase 4 — SHAP smoke-test.
Verifies that shap.TreeExplainer works directly on the xgb.Booster,
and prints the full SHAP output so we know what shape/type to expect.
"""

import datetime
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import shap

PKL_DIR = r"c:\Users\david\OneDrive\Desktop\WorthWise"

model_columns = joblib.load(f"{PKL_DIR}/impulse_model_columns_final.pkl")
numeric_cols  = joblib.load(f"{PKL_DIR}/impulse_numeric_cols_final.pkl")
scaler        = joblib.load(f"{PKL_DIR}/impulse_scaler_final.pkl")
threshold     = joblib.load(f"{PKL_DIR}/impulse_decision_threshold_final.pkl")
booster       = xgb.Booster()
booster.load_model(f"{PKL_DIR}/impulse_booster.json")

# ── Re-use the same build/encode helpers from phase3_test.py ──────────────────
def get_month_column(month_int):
    mapping = {
        1: "Month_Feb", 2: "Month_Feb", 3: "Month_Mar", 4: "Month_Mar",
        5: "Month_May", 6: "Month_June", 7: "Month_Jul", 8: None,
        9: "Month_Sep", 10: "Month_Oct", 11: "Month_Nov", 12: "Month_Dec",
    }
    return mapping[month_int]

def build_and_encode(session_secs, interactions):
    today = datetime.date.today()
    month_int  = today.month
    is_weekend = int(today.weekday() >= 5)

    admin = 0; admin_dur = 0.0; info = 0; info_dur = 0.0
    prod  = interactions; prod_dur = float(session_secs)
    bounce = 0.05 if interactions == 0 else 0.01
    exit_r = bounce; special = 0.0

    total_dur   = admin_dur + info_dur + prod_dur
    total_pages = admin + info + prod
    denom       = total_pages if total_pages > 0 else 1
    avg_dur     = total_dur / denom
    prod_ratio  = prod / denom
    engage      = total_dur * (1.0 - bounce)

    row = pd.DataFrame([{
        "Administrative": admin, "Administrative_Duration": admin_dur,
        "Informational": info,   "Informational_Duration": info_dur,
        "ProductRelated": prod,  "ProductRelated_Duration": prod_dur,
        "BounceRates": bounce,   "ExitRates": exit_r, "SpecialDay": special,
        "Total_Duration": total_dur, "Total_Pages": float(total_pages),
        "Avg_Duration_Per_Page": avg_dur, "ProductRelated_Ratio": prod_ratio,
        "Engagement_Score": engage,
        "Weekend": is_weekend,
    }])

    month_cols = [c for c in model_columns if c.startswith("Month_")]
    for col in month_cols:
        row[col] = 0
    mc = get_month_column(month_int)
    if mc:
        row[mc] = 1

    row["VisitorType_Other"]             = 0
    row["VisitorType_Returning_Visitor"] = 1

    for col in model_columns:
        if col not in row.columns:
            row[col] = 0

    row = row[model_columns]
    row[numeric_cols] = scaler.transform(row[numeric_cols])
    return row

# ── Build test rows ───────────────────────────────────────────────────────────
row_low  = build_and_encode(12.0, 0)   # short session, no interactions
row_high = build_and_encode(300.0, 8)  # long session, many interactions

# ── Prediction ────────────────────────────────────────────────────────────────
for label, row in [("LOW-engagement row", row_low), ("HIGH-engagement row", row_high)]:
    dm   = xgb.DMatrix(row)
    pred = booster.predict(dm)
    print(f"\n{label}:")
    print(f"  predict() output type  : {type(pred)}")
    print(f"  predict() output shape : {pred.shape}")
    print(f"  probability            : {pred[0]:.4f} ({pred[0]*100:.1f}%)")
    print(f"  decision               : {'HIGH RISK' if pred[0] >= threshold else 'LOW RISK'}")

# ── SHAP ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SHAP test on HIGH-engagement row:")
explainer   = shap.TreeExplainer(booster)
shap_result = explainer.shap_values(row_high)

print(f"  shap_values type  : {type(shap_result)}")
if isinstance(shap_result, list):
    print(f"  shap_values is a list of {len(shap_result)} arrays")
    sv = shap_result[1][0]   # positive class for binary classification
    print(f"  Using shap_result[1][0]  shape: {sv.shape}")
else:
    sv = shap_result[0]
    print(f"  shap_values shape : {shap_result.shape}")
    print(f"  Using shap_result[0]  shape: {sv.shape}")

print(f"  expected_value    : {explainer.expected_value}")
print(f"\n  Feature SHAP values (sorted by |SHAP|):")
shap_series = pd.Series(sv, index=model_columns).sort_values(key=abs, ascending=False)
for feat, val in shap_series.items():
    marker = ">>>" if abs(val) > 0.05 else "   "
    print(f"  {marker} {feat:<35} {val:+.6f}")

# ── Verify sum ────────────────────────────────────────────────────────────────
import math
expected_val = explainer.expected_value
if isinstance(expected_val, (list, np.ndarray)):
    expected_val = expected_val[0]
log_odds_pred = expected_val + sv.sum()
prob_from_shap = 1 / (1 + math.exp(-log_odds_pred))
dm_high = xgb.DMatrix(row_high)
direct_prob = float(booster.predict(dm_high)[0])
print(f"\n  SHAP sum + expected_value → probability : {prob_from_shap:.4f}")
print(f"  Direct booster.predict() probability   : {direct_prob:.4f}")
print(f"  Difference (should be ~0)              : {abs(prob_from_shap - direct_prob):.6f}")
print("=" * 70)
print("Phase 4 SHAP test complete.")
