import { useState } from "react";

const initialForm = {
  amount: "48000",
  channel: "UPI",
  hour: "14",
  sender_balance_before: "60000",
  sender_balance_after: "12000",
  new_device: true,
  first_time_payee: true,
  on_call: false,
  sim_swap_hours_ago: "999999",
  payee_blacklisted: false,
};

export default function ManualAnalyzer({ onAnalyzed }) {
  const [form, setForm] = useState(initialForm);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  function update(event) {
    const { name, value, type, checked } = event.target;
    setForm((current) => ({
      ...current,
      [name]: type === "checkbox" ? checked : value,
    }));
  }

  async function analyze(event) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(
        "http://127.0.0.1:8000/api/transaction/analyze",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            ...form,
            amount: Number(form.amount),
            hour: Number(form.hour),
            sender_balance_before: Number(form.sender_balance_before),
            sender_balance_after: Number(form.sender_balance_after),
            sim_swap_hours_ago: Number(form.sim_swap_hours_ago),
          }),
        }
      );

      const body = await response.json();

      if (!response.ok) {
        throw new Error(body.detail || "Transaction analysis failed.");
      }

      setResult(body);

      setForm({
        amount: "",
        channel: "UPI",
        hour: "12",
        sender_balance_before: "",
        sender_balance_after: "",
        new_device: false,
        first_time_payee: false,
        on_call: false,
        sim_swap_hours_ago: "999999",
        payee_blacklisted: false,
      });

      if (onAnalyzed) await onAnalyzed();

    } catch (err) {
      setError(err.message || "Could not connect to the backend.");
    } finally {
      setLoading(false);
    }
  }

  const fieldStyle = {
    width: "100%",
    padding: "10px",
    marginTop: "6px",
    borderRadius: "8px",
    border: "1px solid #334155",
    background: "#111827",
    color: "#f8fafc",
    boxSizing: "border-box",
  };

  const labelStyle = {
    display: "block",
    color: "#cbd5e1",
    fontSize: "14px",
    marginBottom: "14px",
  };

  return (
    <section style={{
      background: "#111827",
      color: "#f8fafc",
      border: "1px solid #263449",
      borderRadius: "14px",
      padding: "24px",
      marginBottom: "24px",
    }}>
      <h2>Analyze a Transaction</h2>
      <p style={{ color: "#94a3b8" }}>
        Enter transaction details to run the hybrid fraud analysis.
      </p>

      <form onSubmit={analyze}>
        <div style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
          gap: "12px",
        }}>
          <label style={labelStyle}>
            Amount (₹)
            <input style={fieldStyle} name="amount" type="number"
              min="0.01" step="0.01" required value={form.amount}
              onChange={update} />
          </label>

          <label style={labelStyle}>
            Payment channel
            <select style={fieldStyle} name="channel"
              value={form.channel} onChange={update}>
              <option value="UPI">UPI</option>
              <option value="IMPS">IMPS</option>
              <option value="NEFT">NEFT</option>
              <option value="RTGS">RTGS</option>
              <option value="CARD">CARD</option>
            </select>
          </label>

          <label style={labelStyle}>
            Transaction hour (0–23)
            <input style={fieldStyle} name="hour" type="number"
              min="0" max="23" required value={form.hour}
              onChange={update} />
          </label>

          <label style={labelStyle}>
            Sender balance before (₹)
            <input style={fieldStyle} name="sender_balance_before"
              type="number" min="0" value={form.sender_balance_before}
              onChange={update} />
          </label>

          <label style={labelStyle}>
            Sender balance after (₹)
            <input style={fieldStyle} name="sender_balance_after"
              type="number" min="0" value={form.sender_balance_after}
              onChange={update} />
          </label>

          <label style={labelStyle}>
            Hours since SIM swap
            <input style={fieldStyle} name="sim_swap_hours_ago"
              type="number" min="0" value={form.sim_swap_hours_ago}
              onChange={update} />
          </label>
        </div>

        <div style={{ display: "flex", flexWrap: "wrap", gap: "18px", margin: "8px 0 20px" }}>
          {[
            ["new_device", "New device"],
            ["first_time_payee", "First-time payee"],
            ["on_call", "Customer on a call"],
            ["payee_blacklisted", "Blacklisted payee"],
          ].map(([name, label]) => (
            <label key={name} style={{ color: "#cbd5e1" }}>
              <input type="checkbox" name={name}
                checked={form[name]} onChange={update} />{" "}
              {label}
            </label>
          ))}
        </div>

        <button disabled={loading} type="submit" style={{
          background: "#2563eb",
          color: "white",
          border: 0,
          borderRadius: "9px",
          padding: "12px 20px",
          cursor: "pointer",
          fontWeight: 600,
        }}>
          {loading ? "Analyzing..." : "Analyze Transaction"}
        </button>
      </form>

      {error && <p style={{ color: "#f87171" }}>{error}</p>}

      {result && (
        <div style={{
          marginTop: "24px",
          padding: "18px",
          border: "1px solid #334155",
          borderRadius: "10px",
        }}>
          <h3>Analysis Result</h3>
          <p>Transaction ID: <strong>{result.txn_id}</strong></p>
          <p>Fraud probability: <strong>{result.risk_score_percent}%</strong></p>
          <p>Risk level: <strong>{result.risk_level}</strong></p>
          <p>Decision: <strong>{result.decision}</strong></p>
          {/* <p>Matched rules: <strong>{result.rule_count}</strong></p> */}
          <p>{result.explanation}</p>
          {result.matched_rules?.map((rule, index) => (
            <p key={`${rule.rule_id}-${index}`}>
              <strong>{rule.rule_id} — {rule.severity}:</strong>{" "}
              {rule.explanation}
            </p>
          ))}
          <small style={{ color: "#94a3b8" }}>
            Prototype output; not a real banking authorization.
          </small>
        </div>
      )}
    </section>
  );
}