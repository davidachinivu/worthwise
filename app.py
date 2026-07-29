import time
import datetime
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
import shap
from dotenv import load_dotenv

load_dotenv()

# ── Paths (all relative to app.py location) ───────────────────────────────────
_BOOSTER_PATH   = "impulse_booster.json"
_COLS_PATH      = "impulse_model_columns_final.pkl"
_NUMERIC_PATH   = "impulse_numeric_cols_final.pkl"
_SCALER_PATH    = "impulse_scaler_final.pkl"
_THRESH_PATH    = "impulse_decision_threshold_final.pkl"

CATEGORIES = ["Clothing", "Electronics", "Home", "Beauty", "Food", "Other"]
REFLECTION_MIN_SECONDS = 8
GEMINI_REFLECTION_MODEL = "gemini-2.5-flash"
GEMINI_TIMEOUT_MS = 10000


# ═══════════════════════════════════════════════════════════════════════════════
# ARTIFACT LOADING
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_artifacts():
    """
    Load all model artifacts once and cache them for the session lifetime.
    Returns (booster, scaler, model_columns, numeric_cols, threshold).
    """
    booster = xgb.Booster()
    booster.load_model(_BOOSTER_PATH)

    scaler        = joblib.load(_SCALER_PATH)
    model_columns = joblib.load(_COLS_PATH)
    numeric_cols  = joblib.load(_NUMERIC_PATH)
    threshold     = joblib.load(_THRESH_PATH)

    return booster, scaler, model_columns, numeric_cols, threshold


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════

def init_session_state():
    """Initialise every session_state key with its default value if not set."""
    defaults = {
        "screen":                   1,
        "item_name":                "",
        "price":                    0.0,
        "category":                 CATEGORIES[0],
        "product_link":             "",
        "is_returning_visitor":     "Yes",  # radio button for VisitorType
        "session_start_time":       None,   # float (time.time()) set on Screen 1 submit
        "compare_clicks":           False,   # ProductRelated toggle
        "policy_clicks":            False,   # Administrative toggle
        "details_clicks":           False,   # Informational toggle
        "need_want_score":          5,      # slider, display-only context
        "why_text":                 "",     # text area, stored but not fed to model
        "session_duration_seconds": 0.0,   # computed at Screen 2 → Screen 3 transition
        "reflection_question":      None,
        "reflection_question_attempted": False,
        "computing_score":          False,  # loading state flag for Screen 2 → Screen 3 transition
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def reset_to_screen1():
    """Reset all session state back to initial values (for 'Check another item')."""
    keys_to_reset = [
        "screen", "item_name", "price", "category", "product_link",
        "is_returning_visitor", "session_start_time", "compare_clicks",
        "policy_clicks", "details_clicks", "need_want_score", "why_text",
        "session_duration_seconds", "reflection_question",
        "reflection_question_attempted", "computing_score",
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE CONSTRUCTION  (Phase 3 — stubs only)
# ═══════════════════════════════════════════════════════════════════════════════

def _get_month_column(month_int: int):
    """
    Map the current calendar month (1–12) to the correct Month_* dummy column,
    or None if the month is August (the drop_first=True baseline).

    Rules:
      - Aug (8)  → None  (baseline; all Month_* columns = 0)
      - Jan (1)  → 'Month_Feb'  (Jan never in training data; nearest valid = Feb)
      - Apr (4)  → 'Month_Mar'  (Apr never in training data; nearest valid = Mar)
      - All others map to their own column as normal.

    Valid Month_* columns in impulse_model_columns_final.pkl (26 cols total):
      Month_Dec, Month_Feb, Month_Jul, Month_June, Month_Mar,
      Month_May, Month_Nov, Month_Oct, Month_Sep
    (Month_Aug absent — it was the drop_first baseline.
     Month_Jan and Month_Apr absent — zero rows in training data.)
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


def _calculate_special_day(today: datetime.date) -> float:
    """
    Calculate closeness of `today` to major retail/shopping holidays.

    Original Dataset Scale:
      SpecialDay takes values in [0.0, 1.0]. A value of 1.0 indicates today is
      the holiday itself or immediately adjacent (0-1 days). The value decays
      linearly by 0.2 per day down to 0.0 for dates >= 5 days away.

    Reference Shopping Dates (evaluated for current, previous, & next year):
      - New Year's Day (Jan 1)
      - Valentine's Day (Feb 14)
      - Mother's Day (Approx May 10)
      - Father's Day (Approx Jun 17)
      - Halloween (Oct 31)
      - Black Friday / Cyber Monday (Approx Nov 25)
      - Christmas (Dec 25)
      - New Year's Eve (Dec 31)
    """
    year = today.year
    special_events = []
    for y in [year - 1, year, year + 1]:
        special_events.extend([
            datetime.date(y, 1, 1),   # New Year's Day
            datetime.date(y, 2, 14),  # Valentine's Day
            datetime.date(y, 5, 10),  # Mother's Day (approx)
            datetime.date(y, 6, 17),  # Father's Day (approx)
            datetime.date(y, 10, 31), # Halloween
            datetime.date(y, 11, 25), # Black Friday / Cyber Monday (approx)
            datetime.date(y, 12, 25), # Christmas
            datetime.date(y, 12, 31), # New Year's Eve
        ])

    min_diff_days = min(abs((event - today).days) for event in special_events)

    if min_diff_days == 0:
        return 1.0
    elif min_diff_days < 5:
        return round(1.0 - (min_diff_days * 0.2), 2)
    else:
        return 0.0


def build_input_row(session_duration_seconds: float,
                    compare_clicks: int = 0,
                    policy_clicks: int = 0,
                    details_clicks: int = 0,
                    visitor_type: str = "Returning_Visitor") -> pd.DataFrame:
    """
    Construct a single-row DataFrame with base values before one-hot encoding.

    Feature mapping (verified against impulse_model_columns_final.pkl, 26 cols):
    ─────────────────────────────────────────────────────────────────────────────
    Base numerics:
      ProductRelated = 1 if compare_clicks is selected else 0
      Administrative = 1 if policy_clicks is selected else 0
      Informational  = 1 if details_clicks is selected else 0

      Proportional duration split approximation:
      The app cannot observe exact time spent on each activity type, so total
      reflection duration is split proportionally across whichever toggles are
      selected. If no toggles are selected, ProductRelated_Duration gets the full
      session duration and the others get 0.0.

      BounceRates  = 0.05 if total_clicks == 0 else 0.01
          (approximation: 0 interactions ≈ high-bounce single-page visit;
           ≥1 selected toggle ≈ engaged multi-page-equivalent session)
      ExitRates    = same logic as BounceRates (correlated in training data)
      SpecialDay   = real proximity score (0.0 to 1.0) based on calendar date

    Base categoricals (encoded in encode_and_scale_row):
      Month       = current system month (with Aug/Jan/Apr special handling)
      Weekend     = 1 if Saturday or Sunday, else 0
      VisitorType = visitor_type ('Returning_Visitor' or 'New_Visitor' baseline)

    Engineered features (exact training-time formulas — do not alter):
      Total_Duration        = Admin_Dur + Info_Dur + ProdRel_Dur
      Total_Pages           = Administrative + Informational + ProductRelated
      Avg_Duration_Per_Page = Total_Duration / max(Total_Pages, 1)  ← zero-div guard
      ProductRelated_Ratio  = ProductRelated / max(Total_Pages, 1)  ← zero-div guard
      Engagement_Score      = Total_Duration * (1 - BounceRates)
    """
    today      = datetime.date.today()
    month_int  = today.month
    is_weekend = int(today.weekday() >= 5)   # 5=Saturday, 6=Sunday

    # Base numerics from 3 distinct toggle selections
    product_related = int(bool(compare_clicks))
    administrative  = int(bool(policy_clicks))
    informational   = int(bool(details_clicks))

    total_clicks = product_related + administrative + informational

    # Proportional duration split approximation
    if total_clicks > 0:
        product_related_duration = float(session_duration_seconds * (product_related / total_clicks))
        administrative_duration  = float(session_duration_seconds * (administrative / total_clicks))
        informational_duration   = float(session_duration_seconds * (informational / total_clicks))
    else:
        product_related_duration = float(session_duration_seconds)
        administrative_duration  = 0.0
        informational_duration   = 0.0

    # BounceRates / ExitRates approximation
    bounce_rates = 0.05 if total_clicks == 0 else 0.01
    exit_rates   = 0.05 if total_clicks == 0 else 0.01
    special_day  = _calculate_special_day(today)

    # Engineered features (must match training-time formulas exactly)
    total_duration = (
        administrative_duration
        + informational_duration
        + product_related_duration
    )
    total_pages = administrative + informational + product_related
    denom = total_pages if total_pages > 0 else 1   # zero-division guard

    avg_duration_per_page = total_duration / denom
    product_related_ratio = product_related / denom
    engagement_score      = total_duration * (1.0 - bounce_rates)

    row = {
        "Administrative":           administrative,
        "Administrative_Duration":  administrative_duration,
        "Informational":            informational,
        "Informational_Duration":   informational_duration,
        "ProductRelated":           product_related,
        "ProductRelated_Duration":  product_related_duration,
        "BounceRates":              bounce_rates,
        "ExitRates":                exit_rates,
        "SpecialDay":               special_day,
        "Total_Duration":           total_duration,
        "Total_Pages":              float(total_pages),
        "Avg_Duration_Per_Page":    avg_duration_per_page,
        "ProductRelated_Ratio":     product_related_ratio,
        "Engagement_Score":         engagement_score,
        # Encoding helpers — removed in encode_and_scale_row
        "_month_int":               month_int,
        "_is_weekend":              is_weekend,
        "_visitor_type":            visitor_type,
    }
    return pd.DataFrame([row])


def encode_and_scale_row(raw_row: pd.DataFrame,
                         model_columns: list,
                         numeric_cols: list,
                         scaler) -> pd.DataFrame:
    """
    Transform raw_row (from build_input_row) into the fully encoded and scaled
    26-column DataFrame the model expects.

    Steps:
      1. Extract & drop the three encoding-helper columns.
      2. Set Weekend (0/1 integer) directly.
      3. One-hot encode Month via _get_month_column(); all Month_* start at 0.
      4. One-hot encode VisitorType; both VisitorType_* start at 0.
      5. Fill any remaining missing model columns with 0.
      6. Reorder columns to exactly match model_columns.
      7. Scale only numeric_cols with the pre-fitted scaler (transform only).
    """
    row = raw_row.copy()

    # Step 1 — extract encoding helpers
    month_int    = int(row["_month_int"].iloc[0])
    is_weekend   = int(row["_is_weekend"].iloc[0])
    visitor_type = str(row["_visitor_type"].iloc[0])
    row = row.drop(columns=["_month_int", "_is_weekend", "_visitor_type"])

    # Step 2 — Weekend
    row["Weekend"] = is_weekend

    # Step 3 — Month one-hot (initialise all to 0, then set active column)
    month_cols = [c for c in model_columns if c.startswith("Month_")]
    for col in month_cols:
        row[col] = 0
    month_col = _get_month_column(month_int)
    if month_col is not None:
        row[month_col] = 1
    # Aug (month_col is None) → all Month_* remain 0 (correct baseline)

    # Step 4 — VisitorType one-hot (initialise both to 0)
    row["VisitorType_Other"]             = 0
    row["VisitorType_Returning_Visitor"] = 0
    visitor_col = f"VisitorType_{visitor_type}"
    if visitor_col in model_columns:
        row[visitor_col] = 1
    # If visitor_type is 'New_Visitor' (drop_first baseline), both stay 0.

    # Step 5 — Fill any missing columns with 0
    for col in model_columns:
        if col not in row.columns:
            row[col] = 0

    # Step 6 — Reorder to exact model_columns order
    row = row[model_columns]

    # Step 7 — Scale numeric columns only (pre-fitted scaler, transform only)
    row[numeric_cols] = scaler.transform(row[numeric_cols])

    return row


def get_prediction_and_shap(encoded_row: pd.DataFrame,
                             booster: xgb.Booster,
                             threshold: float):
    """
    1. Build DMatrix from encoded_row.
    2. Call booster.predict() — returns shape (1,), the positive-class probability.
    3. Compute SHAP values using shap.TreeExplainer on the Booster.
    4. Return (probability, is_high_risk, shap_values, shap_feature_names).
    """
    # 1. Predict
    dmatrix = xgb.DMatrix(encoded_row)
    raw_pred = booster.predict(dmatrix)
    probability = float(raw_pred[0])
    is_high_risk = probability >= threshold

    # 2. SHAP Explainer
    explainer = shap.TreeExplainer(booster)
    shap_result = explainer.shap_values(encoded_row)
    
    # shap_values from TreeExplainer on a binary xgb.Booster returns a 2D array
    if isinstance(shap_result, list):
        shap_vals = shap_result[1][0]
    else:
        shap_vals = shap_result[0]
        
    shap_feature_names = encoded_row.columns.tolist()

    return probability, is_high_risk, shap_vals, shap_feature_names


# ═══════════════════════════════════════════════════════════════════════════════
# STYLING
# ═══════════════════════════════════════════════════════════════════════════════

def get_reflection_question(item_name: str,
                            price: float,
                            score_percent: float,
                            risk_label: str,
                            why_text: str,
                            need_want_score: int):
    """
    Generate one optional rhetorical reflection question with Gemini.
    Returns None silently when unconfigured or unavailable.
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or not why_text.strip():
        return None

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_MS),
        )
        prompt = f'''The user is considering buying "{item_name}" for ${price:,.2f}. Their score for this purchase is {score_percent:.1f}% ({risk_label}). They said their reason for wanting it is:
"{why_text.strip()}"

Their need/want slider value is {need_want_score}/10.

Ask them exactly one short, warm, non-judgmental question (1-2 sentences max) that helps them reflect on whether this is something they truly want or need. Do not give advice, do not tell them what to do, do not mention their score number or percentage directly, do not moralize or lecture. Just ask a single genuine, curious question based on what they wrote. Return only the question itself, no preamble, no quotation marks around it.

Phrase the question so it invites silent self-reflection, not a typed reply. Avoid conversational openers like "Could you share more about..." or "Tell me more about...". Instead, phrase it more like a direct, thought-provoking question the reader considers on their own, for example "What's really driving the pull toward this right now?" or "Is this about needing it, or wanting something new?"'''

        response = client.models.generate_content(
            model=GEMINI_REFLECTION_MODEL,
            contents=prompt,
        )
        question = getattr(response, "text", "").strip()
        if not question:
            return None
        return question.strip('"').strip("'").strip()
    except Exception as exc:
        print(f"Gemini reflection question skipped: {exc}")
        return None
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    /* ── Global reset & typography ── */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        -webkit-font-smoothing: antialiased;
    }

    h1, h2, h3, h4, .ww-brand h1, .ww-card-title, .ww-score-value {
        font-family: 'Outfit', sans-serif !important;
    }

    /* ── Dark Mesh Gradient Background with Ambient Glow ── */
    .stApp {
        background-color: #070914 !important;
        background-image: 
            radial-gradient(at 12% 15%, rgba(99, 102, 241, 0.18) 0px, transparent 55%),
            radial-gradient(at 88% 82%, rgba(52, 211, 153, 0.12) 0px, transparent 55%),
            radial-gradient(at 50% 50%, rgba(139, 92, 246, 0.08) 0px, transparent 65%) !important;
        background-attachment: fixed !important;
        min-height: 100vh;
    }

    /* ── Hide default Streamlit chrome while keeping sidebar toggle visible ── */
    #MainMenu, footer { visibility: hidden; }
    header { background: transparent !important; }
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarExpandButton"],
    button[aria-label*="sidebar"] {
        visibility: visible !important;
        color: #A5B4FC !important;
        background: rgba(30, 35, 64, 0.5) !important;
        border-radius: 8px !important;
        border: 1px solid rgba(165, 180, 252, 0.2) !important;
    }

    /* ── Main content container ── */
    .block-container {
        max-width: 720px !important;
        padding-top: 2rem !important;
        padding-bottom: 3.5rem !important;
    }

    /* ── Brand header ── */
    .ww-brand {
        text-align: center;
        margin-bottom: 1.5rem;
        position: relative;
    }
    .ww-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(168, 85, 247, 0.15));
        border: 1px solid rgba(165, 180, 252, 0.3);
        color: #C7D2FE;
        padding: 0.28rem 0.9rem;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
        box-shadow: 0 0 16px rgba(99, 102, 241, 0.2);
    }
    .ww-brand h1 {
        font-size: 2.85rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FFFFFF 20%, #A5B4FC 60%, #34D399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }
    .ww-brand p {
        color: #9CA3AF;
        font-size: 1.02rem;
        margin: 0.4rem 0 0;
        font-weight: 400;
    }

    /* ── Step indicator timeline ── */
    .ww-steps {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 0;
        margin: 1.5rem 0 2.2rem;
    }
    .ww-step {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.4rem;
        position: relative;
        z-index: 2;
    }
    .ww-step-circle {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.85rem;
        font-weight: 700;
        border: 2px solid #1E2442;
        color: #6B7280;
        background: #0E1225;
        transition: all 0.3s ease;
    }
    .ww-step-circle.active {
        background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%);
        border-color: #A5B4FC;
        color: #FFFFFF;
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.6);
        transform: scale(1.1);
    }
    .ww-step-circle.done {
        background: #064E3B;
        border-color: #10B981;
        color: #34D399;
    }
    .ww-step-label {
        font-size: 0.72rem;
        font-weight: 600;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .ww-step-label.active { color: #A5B4FC; font-weight: 700; }
    .ww-step-label.done   { color: #34D399; }
    .ww-step-line {
        flex: 1;
        height: 3px;
        background: #1E2442;
        margin: 0 0.5rem;
        margin-bottom: 1.4rem;
        min-width: 70px;
        border-radius: 2px;
        transition: all 0.3s ease;
    }
    .ww-step-line.done {
        background: linear-gradient(90deg, #10B981, #34D399);
        box-shadow: 0 0 8px rgba(52, 211, 153, 0.4);
    }

    /* ── Glass Cards System ── */
    .ww-card {
        background: rgba(14, 18, 38, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 2.2rem 2.4rem;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(16px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.06);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .ww-card:hover {
        border-color: rgba(165, 180, 252, 0.2);
    }
    .ww-card-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #F3F4F6;
        margin: 0 0 1.25rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
        letter-spacing: -0.01em;
    }

    /* ── Item summary pill ── */
    .ww-summary {
        background: rgba(26, 32, 66, 0.5);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 16px;
        padding: 1.1rem 1.5rem;
        margin-bottom: 1.5rem;
        display: flex;
        flex-wrap: wrap;
        gap: 1.5rem;
        backdrop-filter: blur(12px);
    }
    .ww-summary-item { display: flex; flex-direction: column; gap: 0.2rem; }
    .ww-summary-label {
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #6B7280;
    }
    .ww-summary-value {
        font-size: 1.05rem;
        font-weight: 700;
        color: #E0E7FF;
    }

    /* ── Countdown ── */
    .ww-countdown {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(139, 92, 246, 0.1));
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        padding: 1rem 1.4rem;
        text-align: center;
        color: #C7D2FE;
        font-size: 0.95rem;
        font-weight: 600;
        margin-top: 0.5rem;
        box-shadow: inset 0 0 12px rgba(99, 102, 241, 0.1);
    }

    /* ── Streamlit widget overrides ── */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox [data-baseweb="select"] {
        background: #090C1B !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        color: #F3F4F6 !important;
        border-radius: 10px !important;
        padding: 0.6rem 0.9rem !important;
        transition: all 0.2s ease !important;
        box-sizing: border-box !important;
        width: 100% !important;
    }
    .stSelectbox [data-baseweb="select"] * {
        color: #F3F4F6 !important;
    }
    .stSelectbox [data-baseweb="select"] [role="combobox"] {
        min-width: 0 !important;
        flex: 1 1 auto !important;
        overflow: visible !important;
    }
    .stSelectbox [data-baseweb="select"] [role="combobox"] > div,
    .stSelectbox [data-baseweb="select"] [role="combobox"] span {
        overflow: visible !important;
        text-overflow: clip !important;
        white-space: nowrap !important;
    }
    .stSelectbox [data-baseweb="select"] [class*="singleValue"] {
        max-width: none !important;
        overflow: visible !important;
        text-overflow: clip !important;
        white-space: nowrap !important;
    }
    div[data-baseweb="popover"] [role="listbox"] {
        background: #090C1B !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
    }
    div[data-baseweb="popover"] [role="option"] {
        background: #090C1B !important;
        color: #F3F4F6 !important;
    }
    div[data-baseweb="popover"] [role="option"]:hover {
        background: #181C38 !important;
    }
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #6366F1 !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25), 0 0 15px rgba(99, 102, 241, 0.15) !important;
    }

    /* Ensure selectbox inner control aligns with number input sizing */
    .stSelectbox [data-baseweb="select"] {
        min-height: 44px !important;
        display: flex !important;
        align-items: center !important;
        padding: 0 !important;
    }
    .stSelectbox [data-baseweb="select"] [data-testid="stMarkdownContainer"] {
        display: flex !important;
        align-items: center !important;
    }

    /* ── Primary button ── */
    .stButton > button[kind="primary"],
    .stButton > button {
        background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 1.8rem !important;
        font-weight: 700 !important;
        font-size: 0.98rem !important;
        letter-spacing: 0.01em !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4) !important;
    }
    .stButton > button:hover:not(:disabled) {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 25px rgba(99, 102, 241, 0.6) !important;
        background: linear-gradient(135deg, #4F46E5 0%, #4338CA 100%) !important;
    }
    .stButton > button:disabled {
        background: #181C38 !important;
        color: #4B5563 !important;
        box-shadow: none !important;
        cursor: not-allowed !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
    }

    /* ── Labels ── */
    .stTextInput label, .stNumberInput label, .stSelectbox label,
    .stTextArea label, .stSlider label {
        color: #9CA3AF !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
    }

    /* ── Interaction counter badge ── */
    .ww-counter {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(52, 211, 153, 0.12);
        border: 1px solid rgba(52, 211, 153, 0.3);
        border-radius: 20px;
        padding: 0.3rem 0.9rem;
        font-size: 0.82rem;
        font-weight: 700;
        color: #34D399;
        margin-top: 0.5rem;
        box-shadow: 0 0 12px rgba(52, 211, 153, 0.15);
    }

    /* ── Section divider ── */
    .ww-divider {
        border: none;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        margin: 1.5rem 0;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #090B18 0%, #050711 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.75rem !important;
    }
    .ww-sb-header {
        text-align: center;
        padding-bottom: 1.5rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 1.6rem;
    }
    .ww-sb-logo {
        font-size: 2.2rem;
        margin-bottom: 0.2rem;
        filter: drop-shadow(0 0 16px rgba(129,140,248,0.6));
    }
    .ww-sb-title {
        font-family: 'Outfit', sans-serif !important;
        font-size: 1.25rem;
        font-weight: 800;
        background: linear-gradient(90deg, #A5B4FC, #60A5FA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .ww-sb-sub {
        font-size: 0.72rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        margin-top: 0.25rem;
        font-weight: 700;
    }
    .ww-sb-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        padding: 0.35rem 0.9rem;
        border-radius: 20px;
        margin-bottom: 0.9rem;
    }
    .ww-sb-chip.blue {
        background: rgba(99,102,241,0.15);
        color: #A5B4FC;
        border: 1px solid rgba(165,180,252,0.3);
    }
    .ww-sb-chip.green {
        background: rgba(52,211,153,0.12);
        color: #34D399;
        border: 1px solid rgba(52,211,153,0.3);
    }
    .ww-sb-divider {
        border: none;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        margin: 1.6rem 0;
    }
    /* Sidebar prose overrides */
    [data-testid="stSidebar"] p {
        color: #9CA3AF !important;
        font-size: 0.85rem !important;
        line-height: 1.65 !important;
    }
    [data-testid="stSidebar"] li {
        color: #9CA3AF !important;
        font-size: 0.85rem !important;
        line-height: 1.65 !important;
        margin-bottom: 0.35rem;
    }
    [data-testid="stSidebar"] strong {
        color: #E0E7FF !important;
        font-weight: 700 !important;
    }
    [data-testid="stSidebar"] h4 {
        font-family: 'Outfit', sans-serif !important;
        color: #F3F4F6 !important;
        font-size: 0.85rem !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 1rem 0 0.4rem !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.08) !important;
        margin: 1.2rem 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)


def render_brand():
    st.markdown("""
    <div class="ww-brand">
        <div class="ww-badge">AI IMPULSE REFLECTION</div>
        <h1>WorthWise</h1>
        <p>Pause. Reflect. Decide with clarity.</p>
    </div>
    """, unsafe_allow_html=True)


def render_step_indicator(active_screen: int):
    steps = [
        ("1", "Entry"),
        ("2", "Reflect"),
        ("3", "Result"),
    ]

    circles = []
    for i, (num, label) in enumerate(steps, start=1):
        if i < active_screen:
            circle_cls = "done"
            icon = "✓"
            label_cls = "done"
        elif i == active_screen:
            circle_cls = "active"
            icon = num
            label_cls = "active"
        else:
            circle_cls = ""
            icon = num
            label_cls = ""

        circles.append(
            f'<div class="ww-step">\n'
            f'  <div class="ww-step-circle {circle_cls}">{icon}</div>\n'
            f'  <div class="ww-step-label {label_cls}">{label}</div>\n'
            f'</div>'
        )

    line1_cls = "done" if active_screen > 1 else ""
    line2_cls = "done" if active_screen > 2 else ""

    st.markdown(
        f'<div class="ww-steps">\n'
        f'  {circles[0]}\n'
        f'  <div class="ww-step-line {line1_cls}"></div>\n'
        f'  {circles[1]}\n'
        f'  <div class="ww-step-line {line2_cls}"></div>\n'
        f'  {circles[2]}\n'
        f'</div>',
        unsafe_allow_html=True
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    """Render the persistent transparency panel in the Streamlit sidebar."""
    with st.sidebar:

        # ── Branding header ───────────────────────────────────────────────────
        st.markdown("""
        <div class="ww-sb-header">
            <div class="ww-sb-logo">WW</div>
            <div class="ww-sb-title">WorthWise</div>
            <div class="ww-sb-sub">Transparency &amp; How It Works</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Section 1: About This Score ───────────────────────────────────────
        st.markdown('<div class="ww-sb-chip blue">About This Score</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        WorthWise uses a machine-learning model trained on real online shopping
        behaviour to estimate how likely a purchase is impulsive — based on how
        you engage with this reflection session.

        **Transparency note:** A standalone app can't observe the same signals a
        live e-commerce site has — like page analytics or traffic source. Some
        signals are approximated and others are fixed defaults. Treat the score as
        a useful reflection prompt, **not a precise prediction**.
        """)

        st.markdown('<hr class="ww-sb-divider">', unsafe_allow_html=True)

        # ── Section 2: How The Score Is Calculated ────────────────────────────
        st.markdown('<div class="ww-sb-chip green">How The Score Is Calculated</div>',
                    unsafe_allow_html=True)

        st.markdown("#### What goes into your score")
        st.markdown("""
        The model evaluates your reflection session using:

        1. **Your Visitor Type** — selected on Step 1 (*Returning Visitor* vs. *New Visitor*).
        2. **Reflection duration** — time spent between landing on Reflect and scoring.
        3. **Categorized interactions** — your button clicks on:
           - *Compare similar items* (`ProductRelated`)
           - *See return/shipping policy* (`Administrative`)
           - *See more product details* (`Informational`)
        4. **Calendar proximity (`SpecialDay`)** — calculated automatically based on how
           close today is to major shopping holidays.

        Everything else — item name, price, category, the Need vs. Want slider, and
        your "Why" text — **does not affect the score**.
        """)

        st.markdown("#### What the app fills in automatically")
        st.markdown("""
        Three signals come from your device's clock and calendar, not typed by hand:

        - **The current month** — shopping patterns vary by time of year.
        - **Whether today is a weekend** — weekend sessions had different purchase rates.
        - **SpecialDay proximity** — score (0.0 to 1.0) calculated from holiday closeness.
        """)

        st.markdown("#### What gets approximated")
        st.markdown("""
        The model was trained on a live e-commerce site's analytics — things
        like pages visited and how quickly people left. A standalone app can't
        observe those, so they're filled with **fixed defaults, the same for
        every user**. This means **the score is a rough estimate, not a
        precise measurement**.
        """)

        st.markdown("#### What the model was trained on")
        st.markdown("""
        A public dataset of real online shopping sessions — specifically,
        whether a browsing session ended in a purchase. WorthWise reuses that
        pattern by having you go through a short reflection "session", measuring
        similar behavioural signals (time spent, engagement level).
        """)

        st.markdown("#### What the percentage means")
        st.markdown("""
        The number shown represents **how often sessions with similar
        characteristics ended in a purchase** in the training data. It is
        **not** a certainty or a diagnosis — think of it as one data point,
        a prompt to pause, rather than a verdict.
        """)


# ═══════════════════════════════════════════════════════════════════════════════
# SCREEN 1 — Entry
# ═══════════════════════════════════════════════════════════════════════════════

def render_screen1():
    render_brand()
    render_step_indicator(1)

    st.markdown('<div class="ww-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="ww-card-title">What are you thinking of buying?</div>',
        unsafe_allow_html=True,
    )

    item_name = st.text_input(
        "Item name",
        value=st.session_state.item_name,
        placeholder="e.g. Sony WH-1000XM5 Headphones",
        key="s1_item_name",
        help="Just for your own reference, not used in the score.",
    )

    col1, col2 = st.columns([0.9, 1.1])
    with col1:
        price = st.number_input(
            "Price ($)",
            min_value=0.0,
            value=float(st.session_state.price),
            step=1.0,
            format="%.2f",
            key="s1_price",
            help="Just for your own reference, not used in the score.",
        )
    with col2:
        category = st.selectbox(
            "Category",
            options=CATEGORIES,
            key="s1_category",
            help="Just for your own reference, not used in the score.",
        )
    product_link = st.text_input(
        "Product link (optional)",
        value=st.session_state.product_link,
        placeholder="https://...",
        key="s1_product_link",
        help="Displayed only — the app does not read or analyze the link.",
    )

    is_returning_choice = st.radio(
        "Have you shopped from a similar store or brand before?",
        options=["Yes", "No"],
        index=0 if st.session_state.is_returning_visitor == "Yes" else 1,
        horizontal=True,
        key="s1_is_returning",
        help="Selecting Yes treats you as a Returning Visitor (VisitorType_Returning_Visitor = 1). Selecting No treats you as a New Visitor (baseline with both VisitorType columns = 0). Factors directly into the model score.",
    )
    st.session_state.is_returning_visitor = is_returning_choice

    st.markdown('<hr class="ww-divider">', unsafe_allow_html=True)

    start_disabled = not item_name.strip()
    if start_disabled:
        st.caption("Enter an item name to continue.")

    if st.button(
        "Start reflection →",
        disabled=start_disabled,
        use_container_width=True,
        key="s1_start",
    ):
        # Store inputs into session state
        st.session_state.item_name      = item_name.strip()
        st.session_state.price          = price
        st.session_state.category       = category
        st.session_state.product_link   = product_link.strip()
        # Record session start time
        st.session_state.session_start_time = time.time()
        st.session_state.screen = 2
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SCREEN 2 — Consideration Session
# ═══════════════════════════════════════════════════════════════════════════════

def render_screen2():
    render_brand()
    render_step_indicator(2)

    # ── Item summary ─────────────────────────────────────────────────────────
    price_str = f"${st.session_state.price:,.2f}"
    st.markdown(f"""
    <div class="ww-summary">
        <div class="ww-summary-item">
            <span class="ww-summary-label">Item</span>
            <span class="ww-summary-value">{st.session_state.item_name}</span>
        </div>
        <div class="ww-summary-item">
            <span class="ww-summary-label">Price</span>
            <span class="ww-summary-value">{price_str}</span>
        </div>
        <div class="ww-summary-item">
            <span class="ww-summary-label">Category</span>
            <span class="ww-summary-value">{st.session_state.category}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Reflection inputs ─────────────────────────────────────────────────────
    st.markdown('<div class="ww-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="ww-card-title">Take a moment to reflect</div>',
        unsafe_allow_html=True,
    )

    need_want = st.slider(
        "Need it vs. Want it",
        min_value=1,
        max_value=10,
        value=st.session_state.need_want_score,
        help="This is for your own reflection only. It does not affect your score.",
        key="s2_need_want",
    )
    st.session_state.need_want_score = need_want

    col_left, col_right = st.columns(2)
    with col_left:
        st.caption("1 · Pure necessity")
    with col_right:
        st.markdown(
            "<div style='text-align:right; color:#6B7280; font-size:0.78rem;'>"
            "10 · Pure impulse want</div>",
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="ww-divider">', unsafe_allow_html=True)

    why_text = st.text_area(
        "Why do you want this? (optional)",
        value=st.session_state.why_text,
        placeholder="Write freely — this is for your own reflection and won't affect the score.",
        height=110,
        key="s2_why",
        help="This is for your own reflection only. It does not affect your score.",
    )
    st.session_state.why_text = why_text

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Explore / interaction buttons ──────────────────────────────────────────
    st.markdown('<div class="ww-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="ww-card-title">Still curious? Explore further</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "These don't show real information, they're a quick check-in on what you'd actually do "
        "before buying. Tap whichever ones are true for you, it helps make your score reflect how "
        "seriously you're weighing this decision."
    )

    compare_selected = st.checkbox(
        "I'd want to compare this to similar products",
        value=bool(st.session_state.compare_clicks),
        key="s2_compare_toggle",
        help="Select this if you would compare this item to similar products before buying.",
    )
    st.session_state.compare_clicks = compare_selected

    policy_selected = st.checkbox(
        "I'd want to check the return/shipping policy",
        value=bool(st.session_state.policy_clicks),
        key="s2_policy_toggle",
        help="Select this if you would look at the return or shipping policy before buying.",
    )
    st.session_state.policy_clicks = policy_selected

    details_selected = st.checkbox(
        "I'd want to read general reviews or FAQs about this brand",
        value=bool(st.session_state.details_clicks),
        key="s2_details_toggle",
        help="Select this if you would read general reviews or FAQs before buying.",
    )
    st.session_state.details_clicks = details_selected

    selected_labels = []
    if st.session_state.compare_clicks:
        selected_labels.append("compare")
    if st.session_state.policy_clicks:
        selected_labels.append("policy")
    if st.session_state.details_clicks:
        selected_labels.append("reviews/FAQs")

    if selected_labels:
        st.markdown(
            f'<div class="ww-counter">Selected: {", ".join(selected_labels)}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="ww-counter">No choices selected yet</div>',
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Timer gating + "Get my score" button ─────────────────────────────────
    st.markdown('<div class="ww-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="ww-card-title">Ready for your score?</div>',
        unsafe_allow_html=True,
    )

    elapsed = time.time() - st.session_state.session_start_time
    remaining = max(0.0, REFLECTION_MIN_SECONDS - elapsed)

    if remaining > 0:
        secs_left = int(remaining) + 1
        st.markdown(
            f'<div class="ww-countdown">'
            f'Please take a moment to reflect &nbsp;·&nbsp; '
            f'<strong>{secs_left}s</strong> remaining'
            f'</div>',
            unsafe_allow_html=True,
        )
        # Disable button and schedule a rerun after 1 second to update countdown
        st.button(
            "Get my score",
            disabled=True,
            use_container_width=True,
            key="s2_score_disabled",
            help="How long you spend here before scoring is one of the two main things that factors into your score.",
        )
        time.sleep(1)
        st.rerun()
    else:
        st.caption("The reflection window has passed — you can now get your score.")
        if st.button(
            "Get my score →",
            use_container_width=True,
            key="s2_score",
            help="How long you spend here before scoring is one of the two main things that factors into your score.",
        ):
            st.session_state.session_duration_seconds = (
                time.time() - st.session_state.session_start_time
            )
            st.session_state.computing_score = True
            st.session_state.screen = 3
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SCREEN 3 — Result
# ═══════════════════════════════════════════════════════════════════════════════

def render_screen3():
    render_brand()
    render_step_indicator(3)

    booster, scaler, model_columns, numeric_cols, threshold = load_artifacts()

    # ── Loading state: compute everything first, then render ─────────────────────
    if st.session_state.computing_score:
        with st.spinner("Calculating your score..."):
            # Compute prediction and SHAP
            v_type = "Returning_Visitor" if st.session_state.get("is_returning_visitor", "Yes") == "Yes" else "New_Visitor"
            raw_row = build_input_row(
                st.session_state.session_duration_seconds,
                st.session_state.compare_clicks,
                st.session_state.policy_clicks,
                st.session_state.details_clicks,
                visitor_type=v_type,
            )
            print("Constructed raw input row before encoding:")
            print(raw_row.T)
            encoded_row = encode_and_scale_row(raw_row, model_columns, numeric_cols, scaler)
            prob, is_high_risk, shap_vals, shap_names = get_prediction_and_shap(encoded_row, booster, threshold)

            # Store results in session state
            st.session_state.score_prob = prob
            st.session_state.score_is_high_risk = is_high_risk
            st.session_state.score_shap_vals = shap_vals
            st.session_state.score_shap_names = shap_names

            # Compute Gemini reflection question if applicable
            why_text_for_question = st.session_state.why_text.strip()
            if why_text_for_question:
                ai_risk_label = "high risk" if is_high_risk else "low risk"
                st.session_state.reflection_question = get_reflection_question(
                    st.session_state.item_name,
                    st.session_state.price,
                    prob * 100,
                    ai_risk_label,
                    why_text_for_question,
                    st.session_state.need_want_score,
                )
                st.session_state.reflection_question_attempted = True
            else:
                st.session_state.reflection_question = None
                st.session_state.reflection_question_attempted = True

            # Clear loading flag and rerun to render full screen
            st.session_state.computing_score = False
            st.rerun()

    # ── Render from stored results ───────────────────────────────────────────────
    prob = st.session_state.score_prob
    is_high_risk = st.session_state.score_is_high_risk
    shap_vals = st.session_state.score_shap_vals
    shap_names = st.session_state.score_shap_names

    # ── Identify Top SHAP Factors ──────────────────────────────────────────────
    allowed_base_names = {
        "ProductRelated_Duration": "the time you spent considering",
        "ProductRelated": "the amount of interaction you had",
        "Total_Duration": "your total session duration",
        "Engagement_Score": "your overall engagement level",
        "Weekend": "shopping on a weekend",
    }

    impacts = []
    for val, name in zip(shap_vals, shap_names):
        if name in allowed_base_names:
            impacts.append((val, allowed_base_names[name]))
        elif name.startswith("Month_"):
            impacts.append((val, "the time of year"))

    impacts.sort(key=lambda x: abs(x[0]), reverse=True)

    # Deduplicate descriptions
    top_factors = []
    seen_desc = set()
    for val, desc in impacts:
        if desc not in seen_desc:
            top_factors.append((val, desc))
            seen_desc.add(desc)
        if len(top_factors) >= 3:
            break

    # Build plain language sentence
    phrases = []
    for val, desc in top_factors:
        action = "increased" if val > 0 else "lowered"
        phrases.append(f"{desc} (which {action} your score)")

    if phrases:
        if len(phrases) > 1:
            sentence = "Your score was mainly influenced by " + ", ".join(phrases[:-1]) + f", and {phrases[-1]}."
        else:
            sentence = f"Your score was mainly influenced by {phrases[-1]}."
    else:
        sentence = "Your score was influenced by various session factors."

    # ── Render Result UI ───────────────────────────────────────────────────────
    st.markdown('<div class="ww-card">', unsafe_allow_html=True)

    color = "#F87171" if is_high_risk else "#34D399"
    risk_label = "HIGH IMPULSE RISK" if is_high_risk else "LOW IMPULSE RISK"
    glow_shadow = "0 0 35px rgba(248, 113, 113, 0.3)" if is_high_risk else "0 0 35px rgba(52, 211, 153, 0.3)"

    st.markdown(f'''
    <div style="text-align: center; padding: 2rem 1rem; position: relative;">
        <div style="font-size: 0.78rem; font-weight: 700; color: #9CA3AF; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 0.8rem;">
            Impulse Purchase Probability
        </div>
        <div style="font-family: 'Outfit', sans-serif; font-size: 5rem; font-weight: 800; color: {color}; line-height: 1; text-shadow: {glow_shadow}; letter-spacing: -0.02em;">
            {prob * 100:.1f}<span style="font-size: 2.8rem; font-weight: 700; opacity: 0.85;">%</span>
        </div>
        <div style="display: inline-flex; align-items: center; background: {color}18; color: {color};
                    padding: 0.45rem 1.6rem; border-radius: 9999px; font-weight: 800; font-size: 0.82rem;
                    margin-top: 1.4rem; border: 1px solid {color}40; letter-spacing: 0.08em; box-shadow: 0 0 20px {color}25;">
            {risk_label}
        </div>
    </div>
    ''', unsafe_allow_html=True)

    if is_high_risk:
        st.markdown(
            '<div style="background: rgba(248, 113, 113, 0.08); border-left: 4px solid #F87171; padding: 1.1rem 1.3rem; margin-bottom: 1.6rem; border-radius: 12px; border: 1px solid rgba(248, 113, 113, 0.15); border-left-width: 4px;">'
            '<div style="color: #FCA5A5; font-size: 0.95rem; font-weight: 700; margin-bottom: 0.2rem;">Recommendation</div>'
            '<div style="color: #D1D5DB; font-size: 0.9rem; line-height: 1.5;">We suggest waiting 24–48 hours before making this purchase to ensure it\'s something you truly need and want.</div>'
            '</div>', unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div style="background: rgba(52, 211, 153, 0.08); border-left: 4px solid #34D399; padding: 1.1rem 1.3rem; margin-bottom: 1.6rem; border-radius: 12px; border: 1px solid rgba(52, 211, 153, 0.15); border-left-width: 4px;">'
            '<div style="color: #6EE7B7; font-size: 0.95rem; font-weight: 700; margin-bottom: 0.2rem;">Recommendation</div>'
            '<div style="color: #D1D5DB; font-size: 0.9rem; line-height: 1.5;">This looks like a considered purchase. Still, take a second to double-check that it aligns with your budget and priorities.</div>'
            '</div>', unsafe_allow_html=True
        )

    st.markdown('<hr class="ww-divider">', unsafe_allow_html=True)

    st.markdown('<div style="font-size: 0.95rem; color: #F3F4F6; font-weight: 700; margin-bottom: 0.5rem;">What drove this score?</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="color: #9CA3AF; font-size: 0.92rem; line-height: 1.6; background: rgba(255,255,255,0.03); padding: 0.9rem 1.1rem; border-radius: 10px; border: 1px solid rgba(255,255,255,0.05);">{sentence}</div>', unsafe_allow_html=True)

    st.markdown('<hr class="ww-divider">', unsafe_allow_html=True)

    st.markdown('<div style="font-size: 0.95rem; color: #F3F4F6; font-weight: 700; margin-bottom: 0.6rem;">Your Reflection Context</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="color: #9CA3AF; font-size: 0.9rem; margin-bottom: 0.5rem;">Need vs. Want score: <strong style="color: #E0E7FF; font-weight: 700;">{st.session_state.need_want_score} / 10</strong></div>', unsafe_allow_html=True)

    if st.session_state.why_text:
        st.markdown(f'<div style="color: #9CA3AF; font-size: 0.9rem; font-style: italic; background: rgba(255,255,255,0.02); border-left: 3px solid #6366F1; padding: 0.7rem 1rem; border-radius: 0 8px 8px 0; margin-top: 0.8rem;">"{st.session_state.why_text}"</div>', unsafe_allow_html=True)

    st.caption("*(Note: The inputs above were not used in the score calculation; they are for your own reflection only.)*")

    if st.session_state.reflection_question:
        st.markdown('<hr class="ww-divider">', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size: 0.95rem; color: #F3F4F6; font-weight: 700; margin-bottom: 0.5rem;">Something to sit with (AI-generated reflection, powered by Gemini)</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="color: #6B7280; font-size: 0.8rem; margin-bottom: 0.8rem;">This reflection question was generated by Google Gemini based on what you wrote, it\'s separate from your score above.</div>',
            unsafe_allow_html=True,
        )
        st.write(st.session_state.reflection_question)

    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("← Check another item", key="s3_reset", use_container_width=True):
        reset_to_screen1()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="WorthWise — Impulse Purchase Reflection",
        page_icon="💡",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    inject_css()
    init_session_state()
    render_sidebar()

    # Pre-load artifacts so they're cached when Screen 3 needs them
    load_artifacts()

    if st.session_state.screen == 1:
        render_screen1()
    elif st.session_state.screen == 2:
        render_screen2()
    elif st.session_state.screen == 3:
        render_screen3()


if __name__ == "__main__":
    main()
