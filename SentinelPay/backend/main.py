from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import math
import uuid

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
import joblib
import pandas as pd

from train_model import load_data, build_features, FEATURES

from rules_engine import evaluate_rules

# -------------------------------
# 1. SQLite database configuration
# -------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATABASE_URL = f"sqlite:///{BASE_DIR / 'sentinelpay.db'}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()


# -------------------------------
# 2. Database table
# -------------------------------

class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    txn_id = Column(String, unique=True, index=True, nullable=False)
    amount = Column(Float, nullable=False)
    risk_score = Column(Float, nullable=True)
    risk_level = Column(String, nullable=False)
    decision = Column(String, nullable=False)
    explanation = Column(String, nullable=False)
    created_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )


Base.metadata.create_all(bind=engine)


# -------------------------------
# 3. Request validation
# -------------------------------

class TransactionInput(BaseModel):
    txn_id: str = Field(min_length=1, max_length=100)
    amount: float = Field(gt=0)
    sender_id: Optional[str] = None
    payee_id: Optional[str] = None
    device_id: Optional[str] = None
    ip_address: Optional[str] = None
    on_call: bool = False
    sim_swap_hours_ago: Optional[float] = Field(
        default=None, ge=0
    )


class ManualTransactionInput(BaseModel):
    amount: float = Field(gt=0, le=100000000)
    channel: str = "UPI"
    hour: int = Field(default=12, ge=0, le=23)
    sender_balance_before: float = Field(default=0, ge=0)
    sender_balance_after: float = Field(default=0, ge=0)
    new_device: bool = False
    first_time_payee: bool = False
    on_call: bool = False
    sim_swap_hours_ago: float = Field(default=999999, ge=0)
    payee_blacklisted: bool = False


# -------------------------------
# 4. Database session dependency
# -------------------------------

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# -------------------------------
# 5. Model & Dataset Loading Initialization
# -------------------------------

MODEL_PATH = BASE_DIR / "artifacts" / "sentinelpay_model.joblib"
model_bundle = joblib.load(MODEL_PATH)

model = model_bundle["model"]
feature_columns = model_bundle["features"]
threshold = model_bundle["threshold"]

# Prepare dataset replay rows once when the backend starts.
# This is for development/demo replay, not production ingestion.
scoring_data = load_data()
scoring_data = build_features(scoring_data)
scoring_data = scoring_data.set_index("txn_id", drop=False)


# -------------------------------
# 6. FastAPI application
# -------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="SentinelPay Fraud Detection API",
    description="Hybrid rule-based and ML transaction risk analysis",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# -------------------------------
# 7. Health check
# -------------------------------

@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "application": "SentinelPay",
        "database": "SQLite"
    }


# -------------------------------
# 8. Read saved transaction history
# -------------------------------

@app.get("/api/transactions")
def get_transactions(db: Session = Depends(get_db)):
    records = (
        db.query(Transaction)
        .order_by(Transaction.id.desc())
        .limit(100)
        .all()
    )

    return [
        {
            "txn_id": row.txn_id,
            "amount": row.amount,
            "risk_score": row.risk_score,
            "risk_level": row.risk_level,
            "decision": row.decision,
            "explanation": row.explanation,
            "created_at": row.created_at.isoformat()
        }
        for row in records
    ]


# -------------------------------
# 9. Dashboard statistics
# -------------------------------

@app.get("/api/stats")
def get_stats(db: Session = Depends(get_db)):
    total = db.query(Transaction).count()
    blocked = db.query(Transaction).filter(
        Transaction.decision == "BLOCK"
    ).count()
    review = db.query(Transaction).filter(
        Transaction.decision == "REVIEW"
    ).count()
    approved = db.query(Transaction).filter(
        Transaction.decision == "APPROVE"
    ).count()

    return {
        "total_transactions": total,
        "blocked": blocked,
        "under_review": review,
        "approved": approved
    }


# -------------------------------
# 10. Dataset replay / scoring endpoint
# -------------------------------

@app.post("/api/transaction/score/{txn_id}")
def score_transaction(txn_id: str, db: Session = Depends(get_db)):
    import json

    # 1. Find the transaction in the demo dataset.
    if txn_id not in scoring_data.index:
        raise HTTPException(
            status_code=404,
            detail="Transaction ID not found in the demo dataset."
        )

    row = scoring_data.loc[txn_id]

    # 2. Generate the ML fraud probability.
    X = pd.DataFrame([
        {col: row[col] for col in feature_columns}
    ])

    probability = float(model.predict_proba(X)[0, 1])

    # 3. Evaluate explicit fraud rules independently.
    transaction_data = row.to_dict()
    rule_result = evaluate_rules(transaction_data)
    matched_rules = rule_result["matched_rules"]
    rule_severity = rule_result["highest_severity"]

    # 4. Combine rule findings and ML output.
    # A blacklist match takes priority over the ML score.
    if rule_severity == "BLOCK":
        decision = "BLOCK"
    elif rule_severity == "HIGH" or probability >= threshold:
        decision = "REVIEW"
    else:
        decision = "APPROVE"

    severity_rank = {
        "LOW": 1,
        "MEDIUM": 2,
        "HIGH": 3,
        "BLOCK": 4,
    }

    ml_level = (
        "HIGH" if probability >= 0.70
        else "MEDIUM" if probability >= threshold
        else "LOW"
    )

    risk_level = max(
        [ml_level, rule_severity],
        key=lambda level: severity_rank[level]
    )

    # 5. Build an explanation for the reviewer.
    if matched_rules:
        explanation = "; ".join(
            f'{item["rule_id"]}: {item["explanation"]}'
            for item in matched_rules
        )
    else:
        explanation = "No implemented fraud rules matched."

    if probability >= threshold:
        explanation += (
            f" ML fraud probability ({probability:.1%}) "
            "meets or exceeds the configured threshold."
        )
    else:
        explanation += (
            f" ML fraud probability ({probability:.1%}) "
            "is below the configured threshold."
        )

    # 6. Save or update the scoring history in SQLite.
    record = (
        db.query(Transaction)
        .filter(Transaction.txn_id == txn_id)
        .first()
    )

    if record is None:
        record = Transaction(
            txn_id=txn_id,
            amount=float(row["amount"]),
            risk_score=probability,
            risk_level=risk_level,
            decision=decision,
            explanation=explanation,
        )
        db.add(record)
    else:
        record.amount = float(row["amount"])
        record.risk_score = probability
        record.risk_level = risk_level
        record.decision = decision
        record.explanation = explanation

    db.commit()
    db.refresh(record)

    # 7. Return both the model result and the rule findings.
    return {
        "txn_id": txn_id,
        "amount": float(row["amount"]),
        "fraud_probability": round(probability, 4),
        "risk_score_percent": round(probability * 100, 2),
        "threshold": threshold,
        "risk_level": risk_level,
        "decision": decision,
        "matched_rules": matched_rules,
        "rule_count": len(matched_rules),
        "rule_severity": rule_severity,
        "explanation": explanation,
        "mode": "dataset_replay_hybrid",
    }


@app.post("/api/transaction/analyze")
def analyze_manual_transaction(data: ManualTransactionInput):
    db = SessionLocal()
    try:
        # Build a neutral starting feature row.
        feature_columns_list = list(model.feature_names_in_)
        features = {column: 0 for column in feature_columns_list}
        # Map user-provided values to model features.
        values = {
            "amount": data.amount,
            "hour": data.hour,
            "transaction_hour": data.hour,
            "is_night": int(data.hour < 6 or data.hour >= 22),
            "on_call": int(data.on_call),
            "new_device": int(data.new_device),
            "first_time_payee": int(data.first_time_payee),
            "sim_swap_hours_ago": data.sim_swap_hours_ago,
            "sender_balance_before": data.sender_balance_before,
            "sender_balance_after": data.sender_balance_after,
            "payee_blacklisted": int(data.payee_blacklisted),
        }
        features.update(values)

        X = pd.DataFrame([features])
        probability = float(model.predict_proba(X)[0, 1])

        rule_result = evaluate_rules(values)
        matched_rules = rule_result["matched_rules"]
        rule_severity = rule_result["highest_severity"]

        if rule_severity == "BLOCK":
            decision = "BLOCK"
        elif rule_severity == "HIGH" or probability >= threshold:
            decision = "REVIEW"
        else:
            decision = "APPROVE"

        severity_rank = {
            "LOW": 1,
            "MEDIUM": 2,
            "HIGH": 3,
            "BLOCK": 4,
        }

        ml_level = (
            "HIGH" if probability >= 0.70
            else "MEDIUM" if probability >= threshold
            else "LOW"
        )

        risk_level = max(
            [ml_level, rule_severity],
            key=lambda level: severity_rank[level]
        )

        if matched_rules:
            explanation = "; ".join(
                f'{item["rule_id"]}: {item["explanation"]}'
                for item in matched_rules
            )
        else:
            explanation = "No implemented fraud rules matched."

        if probability >= threshold:
            explanation += (
                f" ML fraud probability ({probability:.1%}) "
                "meets or exceeds the configured threshold."
            )
        else:
            explanation += (
                f" ML fraud probability ({probability:.1%}) "
                "is below the configured threshold."
            )

        manual_txn_id = f"MANUAL-{uuid.uuid4().hex[:8].upper()}"

        record = Transaction(
            txn_id=manual_txn_id,
            amount=data.amount,
            risk_score=probability,
            risk_level=risk_level,
            decision=decision,
            explanation=explanation,
        )
        db.add(record)
        db.commit()
        db.refresh(record)

        return {
            "txn_id": manual_txn_id,
            "amount": data.amount,
            "fraud_probability": round(probability, 4),
            "risk_score_percent": round(probability * 100, 2),
            "threshold": threshold,
            "risk_level": risk_level,
            "decision": decision,
            "matched_rules": matched_rules,
            "rule_count": len(matched_rules),
            "rule_severity": rule_severity,
            "explanation": explanation,
            "mode": "manual_hybrid",
        }
    finally:
        db.close()