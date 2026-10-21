
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

tx = pd.read_csv(DATA / "transactions.csv", low_memory=False)
labels = pd.read_csv(DATA / "labels.csv", low_memory=False)

print("\n=== R010: CREDIT-THEN-DEBIT ===")
credit_candidates = [
    c for c in tx.columns
    if "credit" in c.lower() or "balance" in c.lower()
]
print("Available credit/balance fields:", credit_candidates)
print("Required history fields for current rule:",
      ["recent_credit_amount", "minutes_since_credit"])
print("Can detect from current row alone:", False)

print("\n=== R011: SIM SWAP + NEW DEVICE ===")
print("SIM swap field available:", "sim_swap_hours_ago" in tx.columns)
print("Device field available:", "device_id" in tx.columns)

print("\n=== R012: STRUCTURING ===")
if "target_rule" in labels.columns:
    print("Rows labeled R012:",
          labels["target_rule"].astype(str).str.contains(
              "R012", na=False
          ).sum())

if "scenario_type" in labels.columns:
    print("\nScenario types:")
    print(labels["scenario_type"].value_counts().to_string())

if "designed_outcome" in labels.columns:
    print("\nDesigned outcomes:")
    print(labels["designed_outcome"].value_counts().to_string())

print("\n=== DATASET CHECK ===")
print("Transactions:", len(tx))
print("Labels:", len(labels))
print("Transaction columns:", list(tx.columns))
print("\nValidation inspection complete.")
