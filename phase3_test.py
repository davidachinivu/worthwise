"""
Phase 3 — Feature construction, encoding, and scaling test.
Run this script in isolation BEFORE wiring into app.py.

It prints the fully constructed, encoded, and scaled row so any
shape or column mismatch is visible immediately.

Expected: 26 columns in exact order matching impulse_model_columns_final.pkl.
"""

import datetime
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

PKL_DIR = r"c:\Users\david\OneDrive\Desktop\WorthWise"

# ── Load artifacts ─────────────────────────────────────────────────────────────
model_columns = joblib.load(f"{PKL_DIR}/impulse_model_columns_final.pkl")
numeric_cols  = joblib.load(f"{PKL_DIR}/impulse_numeric_cols_final.pkl")
scaler        = joblib.load(f"{PKL_DIR}/impulse_scaler_final.pkl")
threshold     = joblib.load(f"{PKL_DIR}/impulse_decision_threshold_final.pkl")

booster = xgb.Booster()
booster.load_model(f"{PKL_DIR}/impulse_booster.json")

print("Artifacts loaded.")
print(f"  model_columns count : {len(model_columns)}")
print(f"  numeric_cols  count : {len(numeric_cols)}")
print()


# ══════════════════════════════════════════════════════════════════════════════
# FUNCTION: get_month_column
# ══════════════════════════════════════════════════════════════════════════════

def get_month_column(month_int: int):
    """
    Maps calendar month (1-12) to the correct Month_* dummy column name,
    or None if the month is August (the drop_first=True baseline).

    Rules:
      - Aug (8)  → None  (baseline category; all Month_* columns = 0)
      - Jan (1)  → 'Month_Feb'  (Jan never in training data; nearest valid = Feb)
      - Apr (4)  → 'Month_Mar'  (Apr never in training data; nearest valid = Mar)
      - All others map to their own column as normal.

    Valid Month_* columns present in impulse_model_columns_final.pkl:
      Month_Dec, Month_Feb, Month_Jul, Month_June, Month_Mar,
      Month_May, Month_Nov, Month_Oct, Month_Sep
    (Month_Aug absent because it was the drop_first baseline; Month_Jan and
     Month_Apr absent because those months had zero rows in the training data.)
    """
    mapping = {
        1:  "Month_Feb",   # FALLBACK: Jan not in training data → use Feb
        2:  "Month_Feb",
        3:  "Month_Mar",
        4:  "Month_Mar",   # FALLBACK: Apr not in training data → use Mar
        5:  "Month_May",
        6:  "Month_June",
        7:  "Month_Jul",
        8:  None,          # BASELINE: Aug dropped by drop_first=True → all zeros
        9:  "Month_Sep",
        10: "Month_Oct",
        11: "Month_Nov",
        12: "Month_Dec",
    }
    return mapping[month_int]


# ══════════════════════════════════════════════════════════════════════════════
# FUNCTION: build_input_row
# ══════════════════════════════════════════════════════════════════════════════

def build_input_row(session_duration_seconds: float,
                    interaction_count: int) -> pd.DataFrame:
    """
    Construct a single-row DataFrame with base values before one-hot encoding.

    Feature mapping (verified against impulse_model_columns_final.pkl):
    ─────────────────────────────────────────────────────────────────────
    Base numerics:
      ProductRelated_Duration = session_duration_seconds
          (the reflection session is the closest proxy for time spent on
           product-related pages)
      ProductRelated          = interaction_count
          (each "Tell me more" click ~ visiting another product page)
      Administrative          = 0  (not observable in standalone app)
      Administrative_Duration = 0
      Informational           = 0
      Informational_Duration  = 0
      BounceRates  = 0.05 if interaction_count == 0 else 0.01
          (approximation: 0 interactions ≈ high-bounce single-page visit;
           ≥1 interaction ≈ engaged multi-page-equivalent session)
      ExitRates    = same logic as BounceRates
          (BounceRates and ExitRates are correlated in the training data;
           using identical logic here is intentional)
      SpecialDay   = 0  (cannot be determined without live retail calendar)

    Base categoricals (encoded in encode_and_scale_row):
      Month       = current system month (with Aug/Jan/Apr fallback logic)
      Weekend     = 1 if Saturday or Sunday, else 0
      VisitorType = "Returning_Visitor" (fixed default; app users are assumed
                    returning because the app is not an e-commerce site)

    Engineered features (must exactly match training-time formulas):
      Total_Duration       = Admin_Dur + Info_Dur + ProdRel_Dur
      Total_Pages          = Administrative + Informational + ProductRelated
      Avg_Duration_Per_Page = Total_Duration / max(Total_Pages, 1)
      ProductRelated_Ratio  = ProductRelated / max(Total_Pages, 1)
      Engagement_Score      = Total_Duration * (1 - BounceRates)
    """
    # ── Resolve current date context ──────────────────────────────────────────
    today = datetime.date.today()
    month_int    = today.month
    is_weekend   = int(today.weekday() >= 5)   # 5=Sat, 6=Sun

    # ── Base numeric values ───────────────────────────────────────────────────
    administrative            = 0
    administrative_duration   = 0.0
    informational             = 0
    informational_duration    = 0.0
    product_related           = interaction_count
    product_related_duration  = float(session_duration_seconds)

    # BounceRates / ExitRates approximation (see docstring above)
    bounce_rates = 0.05 if interaction_count == 0 else 0.01
    exit_rates   = 0.05 if interaction_count == 0 else 0.01   # same logic
    special_day  = 0.0

    # ── Engineered features ───────────────────────────────────────────────────
    total_duration = (
        administrative_duration
        + informational_duration
        + product_related_duration
    )
    total_pages = administrative + informational + product_related

    # Zero-division guard: if no pages visited, treat denominator as 1
    denom = total_pages if total_pages > 0 else 1

    avg_duration_per_page  = total_duration / denom
    product_related_ratio  = product_related / denom
    engagement_score       = total_duration * (1.0 - bounce_rates)

    # ── Assemble base row (pre-encoding) ──────────────────────────────────────
    row = {
        # Base numerics
        "Administrative":           administrative,
        "Administrative_Duration":  administrative_duration,
        "Informational":            informational,
        "Informational_Duration":   informational_duration,
        "ProductRelated":           product_related,
        "ProductRelated_Duration":  product_related_duration,
        "BounceRates":              bounce_rates,
        "ExitRates":                exit_rates,
        "SpecialDay":               special_day,
        # Engineered
        "Total_Duration":           total_duration,
        "Total_Pages":              float(total_pages),
        "Avg_Duration_Per_Page":    avg_duration_per_page,
        "ProductRelated_Ratio":     product_related_ratio,
        "Engagement_Score":         engagement_score,
        # Categorical (pre-encoding — kept for encode_and_scale_row to handle)
        "_month_int":               month_int,
        "_is_weekend":              is_weekend,
        "_visitor_type":            "Returning_Visitor",
    }

    return pd.DataFrame([row])


# ══════════════════════════════════════════════════════════════════════════════
# FUNCTION: encode_and_scale_row
# ══════════════════════════════════════════════════════════════════════════════

def encode_and_scale_row(raw_row: pd.DataFrame,
                         model_columns: list,
                         numeric_cols: list,
                         scaler) -> pd.DataFrame:
    """
    Transform raw_row (output of build_input_row) into the fully encoded and
    scaled 26-column DataFrame the model expects.

    Steps:
      1. Extract and remove the three encoding-helper columns (_month_int,
         _is_weekend, _visitor_type) from the raw row.
      2. Set Weekend (integer 0/1) directly on the row.
      3. One-hot encode Month: use get_month_column() to find the correct
         Month_* column name, set it to 1 (or leave all Month_* at 0 for Aug).
      4. One-hot encode VisitorType: "Returning_Visitor" → VisitorType_Returning_Visitor=1.
      5. Ensure every column in model_columns exists (fill missing with 0).
      6. Reorder columns to exactly match model_columns order.
      7. Scale only the 14 columns in numeric_cols using the pre-fitted scaler.
         (Do NOT fit a new scaler — only transform.)
    """
    row = raw_row.copy()

    # Extract the encoding helpers then drop them
    month_int    = int(row["_month_int"].iloc[0])
    is_weekend   = int(row["_is_weekend"].iloc[0])
    visitor_type = str(row["_visitor_type"].iloc[0])
    row = row.drop(columns=["_month_int", "_is_weekend", "_visitor_type"])

    # ── Weekend ───────────────────────────────────────────────────────────────
    row["Weekend"] = is_weekend

    # ── Month one-hot encoding ─────────────────────────────────────────────────
    # Initialise all Month_* columns to 0 first
    month_cols = [c for c in model_columns if c.startswith("Month_")]
    for col in month_cols:
        row[col] = 0

    month_col = get_month_column(month_int)
    if month_col is not None:
        # Normal month or Jan/Apr fallback — set the appropriate column to 1
        row[month_col] = 1
    # else: August (baseline) — all Month_* columns remain 0 (already set above)

    # ── VisitorType one-hot encoding ──────────────────────────────────────────
    # Only two VisitorType columns exist: VisitorType_Other, VisitorType_Returning_Visitor
    # "New_Visitor" was the drop_first baseline (absent from model_columns).
    row["VisitorType_Other"]              = 0
    row["VisitorType_Returning_Visitor"]  = 0
    visitor_col = f"VisitorType_{visitor_type}"
    if visitor_col in model_columns:
        row[visitor_col] = 1
    # If visitor_type maps to something not in model_columns (e.g., "New_Visitor"),
    # both columns stay 0, which correctly represents the baseline.

    # ── Ensure all model columns are present (fill any missing with 0) ─────────
    for col in model_columns:
        if col not in row.columns:
            row[col] = 0

    # ── Reorder to exact model_columns order ──────────────────────────────────
    row = row[model_columns]

    # ── Scale numeric columns only (do not fit — transform only) ──────────────
    row[numeric_cols] = scaler.transform(row[numeric_cols])

    return row


# ══════════════════════════════════════════════════════════════════════════════
# TEST RUN
# ══════════════════════════════════════════════════════════════════════════════

# Simulate: 45-second session, 2 interactions (a realistic moderate-engagement user)
TEST_DURATION     = 45.0
TEST_INTERACTIONS = 2

print("=" * 70)
print(f"TEST INPUT:")
print(f"  session_duration_seconds = {TEST_DURATION}")
print(f"  interaction_count        = {TEST_INTERACTIONS}")
print(f"  current date             = {datetime.date.today()}  "
      f"(weekday={datetime.date.today().weekday()}, "
      f"weekend={datetime.date.today().weekday() >= 5})")
print(f"  current month int        = {datetime.date.today().month}")
print(f"  month column             = {get_month_column(datetime.date.today().month)}")
print()

# Step 1: Build raw row
raw_row = build_input_row(TEST_DURATION, TEST_INTERACTIONS)
print("RAW ROW (pre-encoding, pre-scaling):")
print(raw_row.T.to_string())
print()

# Step 2: Encode and scale
encoded_row = encode_and_scale_row(raw_row, model_columns, numeric_cols, scaler)
print("=" * 70)
print("FULLY ENCODED AND SCALED ROW (ready for DMatrix):")
print(f"  Shape: {encoded_row.shape}")
print(f"  Column count: {len(encoded_row.columns)}  (expected: {len(model_columns)})")
print()
print("  Columns and values:")
for col, val in zip(encoded_row.columns, encoded_row.iloc[0].values):
    marker = "[SCALED]" if col in numeric_cols else "       "
    print(f"    {marker}  {col:<30} = {val:.6f}")

# Confirm column count
assert len(encoded_row.columns) == len(model_columns), (
    f"COLUMN COUNT MISMATCH: got {len(encoded_row.columns)}, "
    f"expected {len(model_columns)}"
)
assert list(encoded_row.columns) == model_columns, (
    "COLUMN ORDER MISMATCH — check encoding logic"
)
print()
print("=" * 70)
print(f"COLUMN COUNT CHECK: {len(encoded_row.columns)} == {len(model_columns)}  PASS")
print("COLUMN ORDER CHECK: exact match  PASS")

# Step 3: Test prediction with booster
dmatrix = xgb.DMatrix(encoded_row)
raw_pred = booster.predict(dmatrix)
prob     = float(raw_pred[0])
decision = "HIGH RISK" if prob >= threshold else "LOW RISK"

print()
print("=" * 70)
print("PREDICTION:")
print(f"  raw_pred shape    : {raw_pred.shape}")
print(f"  probability       : {prob:.4f}  ({prob*100:.1f}%)")
print(f"  threshold         : {threshold}")
print(f"  decision          : {decision}")
print("=" * 70)
print("Phase 3 test PASSED — all checks green.")
