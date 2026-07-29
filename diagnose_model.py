"""
Diagnostic script for impulse_model_final.pkl
Attempts multiple load strategies and prints raw pickle metadata.
"""
import struct
import pickle
import joblib
import xgboost as xgb

PKL_PATH = r"c:\Users\david\OneDrive\Desktop\WorthWise\impulse_model_final.pkl"

# ── 1. Read first 16 bytes to identify pickle protocol ───────────────────────
with open(PKL_PATH, "rb") as f:
    header = f.read(16)
print("First 16 bytes (hex):", header.hex())
print("First byte (pickle opcode):", hex(header[0]))
# Pickle protocol markers: 0x80 = PROTO opcode, next byte = protocol number
if header[0] == 0x80:
    print(f"Pickle protocol: {header[1]}")
else:
    print("Does not start with standard pickle PROTO opcode — may be joblib compressed")

print()

# ── 2. Try loading with standard pickle (no joblib) ──────────────────────────
print("Attempting: pickle.load() ...")
try:
    with open(PKL_PATH, "rb") as f:
        model = pickle.load(f)
    print(f"  pickle.load() succeeded. Type: {type(model)}")
except Exception as e:
    print(f"  ❌ pickle.load() failed: {e}")

print()

# ── 3. Try loading XGBoost model directly (if saved as JSON/binary via save_model) ──
print("Attempting: xgb.Booster().load_model() ...")
try:
    booster = xgb.Booster()
    booster.load_model(PKL_PATH)
    print(f"  xgb.Booster.load_model() succeeded.")
except Exception as e:
    print(f"  ❌ xgb.Booster.load_model() failed: {e}")

print()

# ── 4. Print xgboost version ──────────────────────────────────────────────────
print(f"xgboost version: {xgb.__version__}")
import sklearn; print(f"sklearn version : {sklearn.__version__}")
import sys; print(f"Python version  : {sys.version}")
