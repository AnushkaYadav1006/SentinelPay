
from collections import defaultdict, deque
from pathlib import Path
from datetime import timedelta
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# =========================================================
# 1. PATHS AND CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ARTIFACTS = ROOT / "backend" / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# These are relative to the first transaction date.
TRAIN_START, TRAIN_END = 30, 37
VAL_START, VAL_END = 38, 44
TEST_START, TEST_END = 45, 59


# =========================================================
# 2. LOAD AND JOIN THE DATASETS
# =========================================================

def load_data():
    tx = pd.read_csv(DATA / "transactions.csv")
    labels = pd.read_csv(DATA / "labels.csv", low_memory=False)
    customers = pd.read_csv(DATA / "customers.csv")
    payees = pd.read_csv(DATA / "payees.csv")

    tx["timestamp"] = pd.to_datetime(tx["timestamp"], errors="coerce")
    customers["created_at"] = pd.to_datetime(
        customers["created_at"], errors="coerce"
    )
    payees["created_at"] = pd.to_datetime(
        payees["created_at"], errors="coerce"
    )

    tx = tx.dropna(subset=["timestamp", "txn_id"]).copy()

    # Verify join keys are unique in lookup tables.
    if labels["txn_id"].duplicated().any():
        raise ValueError("labels.csv contains duplicate txn_id values.")
    if customers["customer_id"].duplicated().any():
        raise ValueError("customers.csv contains duplicate customer_id values.")
    if payees["payee_id"].duplicated().any():
        raise ValueError("payees.csv contains duplicate payee_id values.")

    # Attach the target separately from transaction features.
    df = tx.merge(
        labels[["txn_id", "is_fraud"]],
        on="txn_id",
        how="inner",
        validate="one_to_one",
    )

    # Customer profile of the sender.
    customer_cols = [
        "customer_id", "segment", "created_at",
        "monthly_income", "has_second_device",
    ]
    df = df.merge(
        customers[customer_cols].rename(
            columns={
                "created_at": "customer_created_at",
            }
        ),
        left_on="sender_id",
        right_on="customer_id",
        how="left",
        validate="many_to_one",
    )

    # Payee profile.
    payee_cols = [
        "payee_id", "payee_type", "is_blacklisted", "created_at"
    ]
    df = df.merge(
        payees[payee_cols].rename(
            columns={
                "created_at": "payee_created_at",
                "is_blacklisted": "payee_blacklisted",
            }
        ),
        on="payee_id",
        how="left",
        validate="many_to_one",
    )

    df = df.sort_values(["timestamp", "txn_id"]).reset_index(drop=True)

    df["day_offset"] = (
        (df["timestamp"] - df["timestamp"].min()).dt.total_seconds()
        / 86400
    ).astype(int)

    if df["is_fraud"].isna().any():
        raise ValueError("Some transactions have missing fraud labels.")

    df["is_fraud"] = df["is_fraud"].astype(int)

    if not set(df["is_fraud"].unique()).issubset({0, 1}):
        raise ValueError("is_fraud must contain only 0 and 1.")

    return df


# =========================================================
# 3. CAUSAL HISTORICAL FEATURES
#    Every historical feature uses earlier events only.
# =========================================================

def build_features(df):
    sender_events = defaultdict(deque)
    sender_amounts = defaultdict(deque)
    payee_events = defaultdict(deque)

    seen_sender_payees = defaultdict(set)
    seen_sender_devices = defaultdict(set)
    previous_location = {}

    features = []

    for r in df.itertuples(index=False):
        t = r.timestamp
        sender = r.sender_id
        payee = r.payee_id
        device = r.device_id

        # Expire old events before calculating current features.
        sw = sender_events[sender]
        while sw and (t - sw[0][0]).total_seconds() > 3600:
            sw.popleft()

        sa = sender_amounts[sender]
        while sa and (t - sa[0][0]).total_seconds() > 30 * 86400:
            sa.popleft()

        pw = payee_events[payee]
        while pw and (t - pw[0][0]).total_seconds() > 86400:
            pw.popleft()

        prior_amounts = [x[1] for x in sa]
        past_mean = (
            float(np.mean(prior_amounts)) if prior_amounts else np.nan
        )
        past_std = (
            float(np.std(prior_amounts)) if len(prior_amounts) > 1
            else np.nan
        )

        balance = getattr(r, "sender_balance_before", np.nan)
        amount = float(r.amount)
        income = getattr(r, "monthly_income", np.nan)

        hour = t.hour
        sim_swap = getattr(r, "sim_swap_hours_ago", np.nan)

        # Geo-velocity from the sender's previous known location.
        lat = getattr(r, "latitude", np.nan)
        lon = getattr(r, "longitude", np.nan)
        geo_speed = np.nan

        if (
            sender in previous_location
            and pd.notna(lat)
            and pd.notna(lon)
        ):
            old_t, old_lat, old_lon = previous_location[sender]
            elapsed = (t - old_t).total_seconds() / 3600

            if elapsed > 0:
                lat1, lat2 = np.radians([old_lat, lat])
                dlat = lat2 - lat1
                dlon = np.radians(lon - old_lon)

                a = (
                    np.sin(dlat / 2) ** 2
                    + np.cos(lat1) * np.cos(lat2)
                    * np.sin(dlon / 2) ** 2
                )
                distance = 6371 * 2 * np.arcsin(
                    min(1.0, np.sqrt(a))
                )
                geo_speed = distance / elapsed

        # Only update the location after deriving the feature.
        if pd.notna(lat) and pd.notna(lon):
            previous_location[sender] = (t, lat, lon)

        cust_created = getattr(r, "customer_created_at", pd.NaT)
        payee_created = getattr(r, "payee_created_at", pd.NaT)

        customer_age = (
            (t - cust_created).total_seconds() / 86400
            if pd.notna(cust_created) else np.nan
        )
        payee_age = (
            (t - payee_created).total_seconds() / 86400
            if pd.notna(payee_created) else np.nan
        )

        features.append({
            # "amount": amount,
            "log_amount": np.log1p(max(0, amount)),
            "balance_before": balance,
            "amount_balance_ratio": (
                amount / balance if pd.notna(balance) and balance > 0
                else np.nan
            ),
            "hour": hour,
            "day_of_week": t.dayofweek,
            "is_night": int(0 <= hour < 5),
            "on_call": getattr(r, "on_call", 0),
            "sim_swap_hours_ago": sim_swap,
            "sim_swap_missing": int(pd.isna(sim_swap)),
            "income": income,
            "amount_income_ratio": (
                amount / income if pd.notna(income) and income > 0
                else np.nan
            ),
            "customer_age_days": customer_age,
            "payee_age_days": payee_age,
            "payee_blacklisted": getattr(
                r, "payee_blacklisted", np.nan
            ),
            "has_second_device": getattr(
                r, "has_second_device", np.nan
            ),
            "prior_sender_txns_1h": len(sw),
            "prior_distinct_payees_1h": len({x[1] for x in sw}),
            "prior_amount_mean_30d": past_mean,
            "amount_vs_past_mean": (
                amount / past_mean
                if pd.notna(past_mean) and past_mean > 0
                else np.nan
            ),
            "amount_zscore_30d": (
                (amount - past_mean) / past_std
                if pd.notna(past_mean) and pd.notna(past_std)
                and past_std > 0 else np.nan
            ),
            "first_time_payee": int(
                payee not in seen_sender_payees[sender]
            ),
            "new_device": int(
                device not in seen_sender_devices[sender]
            ),
            "prior_payee_senders_24h": len(
                {x[1] for x in pw}
            ),
            "geo_speed_kmh": geo_speed,
            "channel": getattr(r, "channel", "unknown"),
            "customer_segment": getattr(r, "segment", "unknown"),
            "payee_type": getattr(r, "payee_type", "unknown"),
        })

        # Current transaction becomes history only AFTER its features
        # have been computed.
        sw.append((t, payee, amount))
        sa.append((t, amount))
        pw.append((t, sender, amount))
        seen_sender_payees[sender].add(payee)
        seen_sender_devices[sender].add(device)

    feature_df = pd.DataFrame(features)

    # Keep original transaction/label data and add only
    # engineered features that are not already present.
    new_columns = [
        col for col in feature_df.columns
        if col not in df.columns
    ]

    result = pd.concat(
        [
            df.reset_index(drop=True),
            feature_df[new_columns].reset_index(drop=True),
        ],
        axis=1,
    )

    # Safety check: sklearn requires unique feature names.
    if not result.columns.is_unique:
        duplicates = result.columns[result.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate columns remain: {duplicates}")

    return result


# =========================================================
# 4. CHRONOLOGICAL SPLIT
# =========================================================

def select_days(df, start, end):
    return df[
        (df["day_offset"] >= start)
        & (df["day_offset"] <= end)
    ].copy()


# =========================================================
# 5. MODEL PIPELINE
# =========================================================

NUMERIC = [
    "amount", "log_amount", "balance_before",
    "amount_balance_ratio", "hour", "day_of_week", "is_night",
    "on_call", "sim_swap_hours_ago", "sim_swap_missing",
    "income", "amount_income_ratio", "customer_age_days",
    "payee_age_days", "payee_blacklisted", "has_second_device",
    "prior_sender_txns_1h", "prior_distinct_payees_1h",
    "prior_amount_mean_30d", "amount_vs_past_mean",
    "amount_zscore_30d", "first_time_payee", "new_device",
    "prior_payee_senders_24h", "geo_speed_kmh",
]

CATEGORICAL = ["channel", "customer_segment", "payee_type"]
FEATURES = NUMERIC + CATEGORICAL


def make_model():
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(
            strategy="most_frequent"
        )),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    prep = ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC),
        ("categorical", categorical_pipe, CATEGORICAL),
    ])

    classifier = RandomForestClassifier(
        n_estimators=250,
        min_samples_leaf=2,
        class_weight={0: 1, 1: 2},
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline([
        ("preprocess", prep),
        ("classifier", classifier),
    ])




def choose_threshold(y_true, probabilities):
    # Aim for precision around 85%, while maximizing F1.
    candidates = np.arange(0.05, 0.96, 0.005)
    results = []

    for threshold in candidates:
        pred = (probabilities >= threshold).astype(int)

        if pred.sum() == 0:
            continue

        precision = precision_score(
            y_true, pred, zero_division=0
        )
        recall = recall_score(
            y_true, pred, zero_division=0
        )
        f1 = f1_score(
            y_true, pred, zero_division=0
        )

        results.append({
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    # Prefer a balanced result with precision near 85%.
    eligible = [
        r for r in results
        if 0.82 <= r["precision"] <= 0.88
    ]

    if eligible:
        best = max(
            eligible,
            key=lambda r: (r["f1"], r["recall"])
        )
    else:
        # If the target range is unavailable, maximize F1.
        best = max(
            results,
            key=lambda r: (r["f1"], r["recall"])
        )

    print(
        f"Selected threshold: {best['threshold']:.3f} | "
        f"Validation precision: {best['precision']:.2%} | "
        f"Validation recall: {best['recall']:.2%} | "
        f"Validation F1: {best['f1']:.4f}"
    )

    return best["threshold"]



def report_metrics(name, y_true, probabilities, threshold):
    pred = (probabilities >= threshold).astype(int)

    print(f"\n===== {name} =====")
    print(f"Threshold: {threshold:.2f}")
    print(f"Rows: {len(y_true)}")
    print(f"Fraud rate: {np.mean(y_true):.4f}")
    print(f"Alert rate: {np.mean(pred):.4f}")
    print(f"Precision: {precision_score(y_true, pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true, pred, zero_division=0):.4f}")
    print(f"F1:        {f1_score(y_true, pred, zero_division=0):.4f}")

    if len(np.unique(y_true)) == 2:
        print(
            "PR-AUC:    "
            f"{average_precision_score(y_true, probabilities):.4f}"
        )
        print(
            "ROC-AUC:   "
            f"{roc_auc_score(y_true, probabilities):.4f}"
        )

    print("Confusion matrix [ [TN, FP], [FN, TP] ]:")
    print(confusion_matrix(y_true, pred, labels=[0, 1]))
    print(classification_report(
        y_true, pred, zero_division=0
    ))


# =========================================================
# 6. TRAIN, VALIDATE, TEST, SAVE
# =========================================================

def main():
    df = load_data()
    df = build_features(df)

    train = select_days(df, TRAIN_START, TRAIN_END)
    validation = select_days(df, VAL_START, VAL_END)
    test = select_days(df, TEST_START, TEST_END)

    if train.empty or validation.empty or test.empty:
        available = sorted(df["day_offset"].unique())
        raise ValueError(
            "The expected day ranges are not present. "
            f"Available day offsets: {available[:5]} ... "
            f"{available[-5:]}. Check the handbook split."
        )

    for name, part in [
        ("train", train), ("validation", validation), ("test", test)
    ]:
        if part["is_fraud"].nunique() < 2:
            raise ValueError(
                f"{name} has only one target class; "
                "metrics and threshold selection would be unreliable."
            )

    print("Total labelled rows:", len(df))
    print("Date range:", df["timestamp"].min(), "to", df["timestamp"].max())
    print("Train rows:", len(train))
    print("Validation rows:", len(validation))
    print("Test rows:", len(test))

    # Fit an initial model on training days only.
    model = make_model()
    model.fit(train[FEATURES], train["is_fraud"])

    val_prob = model.predict_proba(
        validation[FEATURES]
    )[:, 1]
    threshold = choose_threshold(
        validation["is_fraud"].to_numpy(), val_prob
    )
    report_metrics(
        "VALIDATION", validation["is_fraud"].to_numpy(),
        val_prob, threshold
    )

    # Final model: refit on training + validation days.
    
    # Retrain the final model using both training and validation data.
    final_fit = select_days(df, TRAIN_START, VAL_END)

    final_model = make_model()
    final_model.fit(
        final_fit[FEATURES],
        final_fit["is_fraud"]
    )

    # Evaluate once on the untouched test period.
    test_prob = final_model.predict_proba(
        test[FEATURES]
    )[:, 1]

    report_metrics(
        "FINAL TEST",
        test["is_fraud"].to_numpy(),
        test_prob,
        threshold
    )

    bundle = {
        "model": final_model,
        "features": FEATURES,
        "numeric_features": NUMERIC,
        "categorical_features": CATEGORICAL,
        "threshold": float(threshold),
        "model_name": "RandomForestClassifier",
        "feature_version": 1,
    }


    model_path = ARTIFACTS / "sentinelpay_model.joblib"
    joblib.dump(bundle, model_path)

    metadata = {
        "model_name": "RandomForestClassifier",
        "threshold": float(threshold),
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "final_fit_rows": int(len(train)),
        "test_rows": int(len(test)),
        "features": FEATURES,
        "note": (
            "Synthetic dataset evaluation; threshold chosen on validation "
            "data. Test metrics are not evidence of real-world bank performance."
        ),
    }

    with open(ARTIFACTS / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved model to:", model_path)
    print("Saved metadata to:", ARTIFACTS / "model_metadata.json")


if __name__ == "__main__":
    main()
