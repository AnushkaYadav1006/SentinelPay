
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
tx = pd.read_csv(ROOT / "data" / "transactions.csv", low_memory=False)
labels = pd.read_csv(ROOT / "data" / "labels.csv", low_memory=False)

df = tx.merge(labels, on="txn_id", how="inner")
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

print("=== R012 LABEL DISTRIBUTION ===")
print(labels.loc[
    labels["target_rule"].astype(str).str.contains("R012", na=False),
    ["scenario_type", "designed_outcome"]
].value_counts().to_string())

print("\n=== SCENARIO COMPARISON ===")
types = [
    "structuring",
    "near_miss_structuring",
    "near_miss_window",
    "single_near_threshold",
    "at_threshold",
]

for kind in types:
    group = df[df["scenario_type"].astype(str) == kind].copy()
    if group.empty:
        continue

    print(f"\n--- {kind} ({len(group)} rows) ---")
    print("Amount median:", round(group["amount"].median(), 2))
    print("Amount min/max:",
          round(group["amount"].min(), 2),
          round(group["amount"].max(), 2))
    print("Unique senders:", group["sender_id"].nunique())
    print("Unique payees:", group["payee_id"].nunique())
    print("Target rules:",
          group["target_rule"].value_counts().head(5).to_dict())

    # Show one sender's chronological transaction sequence
    sender = group["sender_id"].value_counts().index[0]
    sample = group[group["sender_id"] == sender].sort_values("timestamp")
    print(sample[
        ["txn_id", "timestamp", "sender_id", "payee_id", "amount"]
    ].head(8).to_string(index=False))

print("\nInspection complete.")
