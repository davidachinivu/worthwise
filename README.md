# Worthwise

**Worthwise** is a reflection tool that helps you pause before an impulse purchase. Instead
of just asking you to fill out a form, it puts you through a short, timed "consideration
session" and uses your actual behavior during that session, how long you spent, how
engaged you were, to estimate how "purchase-like" the session looks, based on patterns
learned from real e-commerce browsing data. It then asks one AI-generated reflective
question to help you sit with your decision before you buy.

**Live app:** https://worthwise.streamlit.app/

**Model on Hugging Face:** https://huggingface.co/davidachinivu/worthwise-impulse-model

---

## How It Works

1. **Enter an item** — name, price, category, optionally a link (for your own reference
   only, not used in scoring)
2. **Sit with it** — a short, timed reflection screen. A "Get my score" button stays
   disabled for the first several seconds, a deliberate forced pause
3. **Get a score** — a trained machine learning model estimates the probability that a
   session like yours, based on real, measured behavior, tends to end in a purchase
4. **Get a reflection question** — if you wrote why you want the item, Google Gemini
   generates one short, rhetorical question to help you think it through, no reply needed

Full transparency about exactly what does and doesn't factor into your score is available
in the app itself, via the sidebar.

---

## Why This Project Exists

This started from a simple goal: reduce impulse spending. Rather than build a generic
budgeting app, Worthwise focuses specifically on the moment of decision, using a model
trained on real browsing behavior to add a small, honest speed bump before a purchase.

A deliberate design principle runs through the whole project: **the app should never
pretend to know more than it actually does.** Several features that would have made the
model meaningfully more accurate (like Google Analytics' `PageValues` metric) simply
aren't observable by a standalone app, so they were excluded from training entirely,
rather than faked or defaulted in a way that overstates the tool's precision.

---

## The Model

- **Type:** XGBoost classifier
- **Trained on:** [UCI Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset),
  12,330 real, anonymized e-commerce sessions
- **Predicts:** the probability that a session's behavior resembles sessions that ended in
  a purchase (not whether a purchase will be regretted, no such labeled data exists
  publicly)
- **Performance:** ROC-AUC 0.777, Recall 0.75, Precision 0.31, on a held-out test set,
  using only features a standalone app can genuinely provide
- **Decision threshold:** 0.55, tuned via a precision/recall/F1 sweep rather than using
  the default 0.5

Full model card, including known limitations and the honest tradeoffs made during
training, is available in the [Hugging Face model repo](#).

---

## What Actually Feeds the Score

Worthwise is upfront that most of what you type in does **not** affect your score, it's
there for your own reference and reflection. Only a handful of real, measured signals do:

**Used in scoring:** time spent reflecting, which "explore" options you engaged with,
whether you've shopped a similar store before, the current date (month, weekend, proximity
to major shopping dates)

**Not used in scoring (shown for your own reflection only):** item name, price, category,
product link, "need vs. want" rating, your written reasoning

This distinction is explained in full inside the app.

---

## Tech Stack

- **Frontend/App:** [Streamlit](https://streamlit.io)
- **Model:** XGBoost, trained in a Colab notebook
- **Explainability:** SHAP (per-prediction feature attribution)
- **Reflection questions:** Google Gemini API
- **Deployment:** Streamlit Community Cloud

---

## Project Structure

```
worthwise/
├── app.py                              # Main Streamlit app
├── requirements.txt                    # Python dependencies
├── impulse_booster.json                # Trained XGBoost model (native format)
├── impulse_scaler_final.pkl            # Fitted StandardScaler
├── impulse_model_columns_final.pkl     # Expected model input columns, in order
├── impulse_numeric_cols_final.pkl      # Which columns need scaling
├── impulse_decision_threshold_final.pkl# Tuned decision threshold (0.55)
├── .gitignore
└── README.md
```

---

## Running Locally

```bash
git clone https://github.com/davidachinivu/worthwise.git
cd worthwise
pip install -r requirements.txt
```

Create a `.env` file in the project root (never committed, already excluded via
`.gitignore`):

```
GEMINI_API_KEY=your_key_here
```

Then run:

```bash
streamlit run app.py
```

---

## Known Limitations

- **Precision is low (0.31).** Roughly two out of three "high risk" flags are false
  positives. This is a real, measured limitation of a model that had to be trained
  without its strongest original feature (`PageValues`), which is only available to real
  e-commerce backends, not standalone apps.
- **No regret label exists anywhere publicly**, so "purchase-likelihood" is used as the
  closest available proxy for "impulsive," not a direct measurement of regret.
- **Several features are approximated**, not measured directly, at inference time (for
  example, bounce/exit rate proxies based on click behavior). The in-app transparency
  section documents exactly which ones.
- **Session history isn't persisted** across visits; each session starts fresh.

---

## Roadmap

- A Chrome extension version, which would allow several currently-defaulted features
  (browser, OS, referrer/traffic source) to become real, observed signals instead of
  fixed constants
- Possible model improvement using real usage data collected over time, rather than
  further tuning on the original static dataset

---

## Acknowledgments

Built on the UCI Online Shoppers Purchasing Intention Dataset (Sakar, C.O., Polat, S.O.,
Katircioglu, M. et al., Neural Comput & Applic, 2018).
