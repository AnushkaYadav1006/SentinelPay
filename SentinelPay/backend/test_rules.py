from rules_engine import evaluate_rules

sample_transaction = {
    "amount": 65000,
    "prior_sender_txns_1h": 7,
    "prior_distinct_payees_1h": 4,
    "amount_zscore_30d": 3.5,
    "first_time_payee": True,
    "new_device": True,
    "timestamp": "2026-09-15 02:30:00",
    "on_call": False,
    "geo_speed_kmh": 20,
    "prior_payee_senders_24h": 2,
    "payee_age_days": 60,
}

result = evaluate_rules(sample_transaction)

print("Matched rules:")
for rule in result["matched_rules"]:
    print(rule["rule_id"], "-", rule["severity"], "-", rule["explanation"])

print("Highest severity:", result["highest_severity"])
