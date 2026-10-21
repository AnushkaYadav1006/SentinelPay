from typing import Any


SEVERITY_ORDER = {
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
    "BLOCK": 4,
}


def get_value(row: dict[str, Any], *names: str, default=None):
    """Return the first available, non-missing field."""
    for name in names:
        value = row.get(name, default)
        if value is None:
            continue

        try:
            if value != value:  # Handles NaN
                continue
        except (TypeError, ValueError):
            pass

        return value

    return default


def as_number(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return False


def evaluate_rules(row: dict[str, Any]) -> dict[str, Any]:
    """
    Evaluate rules using transaction fields and causal historical features.
    Missing required fields do not automatically trigger a rule.
    """
    matches = []

    amount = as_number(get_value(row, "amount"))
    hour_value = get_value(row, "hour")

    # Derive hour from the transaction timestamp if present.
    if hour_value is None:
        timestamp = get_value(row, "timestamp")
        if timestamp is not None:
            try:
                import pandas as pd
                hour_value = pd.Timestamp(timestamp).hour
            except (ValueError, TypeError):
                hour_value = None

    hour = as_number(hour_value, default=-1)
    on_call = as_bool(get_value(row, "on_call", default=False))
    new_device = as_bool(get_value(row, "new_device", default=False))
    first_payee = as_bool(get_value(row, "first_time_payee", default=False))

    def add(rule_id, severity, explanation):
        matches.append({
            "rule_id": rule_id,
            "severity": severity,
            "explanation": explanation,
        })

    # R001: Known blacklisted payee, device, or IP.
    if any(
        as_bool(get_value(row, field, default=False))
        for field in (
            "payee_blacklisted",
            "is_blacklisted",
            "device_blacklisted",
            "ip_blacklisted",
        )
    ):
        add("R001", "BLOCK", "A payee, device, or IP is blacklisted.")

    # R002: Rapid transaction activity.
    txn_count = as_number(get_value(row, "prior_sender_txns_1h"))
    distinct_payees = as_number(get_value(row, "prior_distinct_payees_1h"))
    if txn_count > 5 and distinct_payees >= 4:
        add("R002", "HIGH", "High transaction velocity across multiple payees.")

    # R003: Amount is more than three standard deviations above history.
    zscore = get_value(row, "amount_zscore_30d")
    if zscore is not None and as_number(zscore, default=0) > 3:
        add("R003", "MEDIUM", "Amount is unusually high compared with the customer's history.")

    # R004: First-time payee and high amount.
    # Senior-citizen adjustment is not applied because customer age is
    # not among the confirmed available fields.
    if first_payee and amount > 50000:
        add("R004", "MEDIUM", "High-value transfer to a first-time payee.")

    # R005: Night-time transfer from a new device.
    if hour >= 0 and 0 <= hour < 6 and new_device and amount > 25000:
        add("R005", "HIGH", "High-value night-time transfer from a new device.")

    # R006: Balance drain above 80% to a first-time payee.
    before = get_value(row, "sender_balance_before")
    after = get_value(row, "sender_balance_after")
    if before is not None and after is not None:
        before = as_number(before, default=0)
        after = as_number(after, default=0)
        if before > 0 and (before - after) / before > 0.80 and first_payee:
            add("R006", "HIGH", "More than 80% of the prior balance was spent on a first-time payee.")

    # R007: On-call activity, first-time payee, and high amount.
    if on_call and first_payee and amount > 20000:
        add("R007", "HIGH", "On-call transaction to a first-time payee above the configured threshold.")

    # R008: Improbable geographic velocity.
    speed = get_value(row, "geo_speed_kmh")
    if speed is not None and as_number(speed) > 500:
        add("R008", "HIGH", "Geographic movement exceeds 500 km/h.")

    # R009: Many senders paying a newly created payee.
    fan_in = as_number(get_value(row, "prior_payee_senders_24h"))
    payee_age = get_value(row, "payee_age_days")
    if payee_age is not None and fan_in >= 10 and as_number(payee_age) < 7:
        add("R009", "HIGH", "New payee is receiving transfers from many distinct senders.")

    # R010: Rapid credit-then-debit with high pass-through
    credit_amount = as_number(
        get_value(row, "recent_credit_amount", "credit_amount"),
        default=0
    )
    minutes_since_credit = as_number(
        get_value(row, "minutes_since_credit"),
        default=999999
    )

    if (
        0 < minutes_since_credit <= 15
        and credit_amount > 0
        and amount >= 0.90 * credit_amount
    ):
        add(
            "R010",
            "HIGH",
            "At least 90% of a recent credit transferred out within 15 minutes."
        )

    # R011: Recent SIM swap combined with a new device
    sim_swap_hours = as_number(
        get_value(row, "sim_swap_hours_ago"),
        default=999999
    )
    if (
        sim_swap_hours < 24
        and new_device
        and amount > 25000
    ):
        add(
            "R011",
            "HIGH",
            "Recent SIM swap, new device, and high-value transaction."
        )

    # R012 needs its exact agreed definition.
    # This is intentionally not guessed here.

    highest_severity = (
        max(matches, key=lambda item: SEVERITY_ORDER[item["severity"]])["severity"]
        if matches
        else "LOW"
    )

    return {
        "matched_rules": matches,
        "rule_count": len(matches),
        "highest_severity": highest_severity,
    }