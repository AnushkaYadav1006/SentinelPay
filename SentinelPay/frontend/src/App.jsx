import { useCallback, useEffect, useState } from "react";
import "./App.css";
import ManualAnalyzer from "./ManualAnalyzer";

const API = "http://127.0.0.1:8000";

function formatMoney(value) {
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 2,
  }).format(Number(value) || 0);
}

function RiskBadge({ value }) {
  const risk = String(value || "LOW").toUpperCase();

  return (
    <span className={`badge badge-${risk.toLowerCase()}`}>
      {risk}
    </span>
  );
}

function StatCard({ title, value, subtitle, icon }) {
  return (
    <div className="stat-card">
      <div className="stat-top">
        <span className="stat-label">{title}</span>
        <span className="stat-icon">{icon}</span>
      </div>
      <div className="stat-value">{value ?? 0}</div>
      <div className="stat-subtitle">{subtitle}</div>
    </div>
  );
}

export default function App() {
  const [stats, setStats] = useState({});
  const [transactions, setTransactions] = useState([]);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState("");
  const [apiOnline, setApiOnline] = useState(false);

  const loadDashboard = useCallback(async () => {
    setRefreshing(true);

    try {
      const [healthResponse, statsResponse, txResponse] =
        await Promise.all([
          fetch(`${API}/api/health`),
          fetch(`${API}/api/stats`),
          fetch(`${API}/api/transactions`),
        ]);

      if (!healthResponse.ok || !statsResponse.ok || !txResponse.ok) {
        throw new Error("One or more API endpoints returned an error.");
      }

      const statsData = await statsResponse.json();
      const txData = await txResponse.json();

      const transactionList = Array.isArray(txData)
        ? txData
        : txData.transactions || txData.items || txData.value || [];

      setStats(statsData);
      setTransactions(transactionList);
      setApiOnline(true);
      setError("");
    } catch {
      setApiOnline(false);
      setError(
        "Cannot load dashboard data. Make sure the FastAPI backend is running."
      );
    } finally {
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    loadDashboard();
  }, [loadDashboard]);

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

      if (loadDashboard) await loadDashboard();

    } catch (err) {
      setError(err.message || "Could not connect to the backend.");
    } finally {
      setLoading(false);
    }
  }

  const total =
    stats.total ?? stats.total_transactions ?? transactions.length;

  const approved = stats.approved ?? stats.approved_count ?? 0;
  const review =
    stats.under_review ?? stats.review ?? stats.review_count ?? 0;
  const blocked = stats.blocked ?? stats.blocked_count ?? 0;

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">S</div>
          <div>
            <div className="brand-name">SentinelPay</div>
            <div className="brand-caption">FRAUD INTELLIGENCE</div>
          </div>
        </div>

        <div className="nav-label">WORKSPACE</div>
        <div className="nav-item active">
          <span>▦</span> Overview
        </div>

        <div className="sidebar-bottom">
          <div className="system-indicator">
            <span className={apiOnline ? "online-dot" : "offline-dot"} />
            <div>
              <strong>{apiOnline ? "API Connected" : "API Offline"}</strong>
              <small>FastAPI · SQLite</small>
            </div>
          </div>
          <div className="sidebar-note">
            SentinelPay is a fraud-detection prototype using demo data.
          </div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <div className="eyebrow">RISK OPERATIONS / OVERVIEW</div>
            <h1>Fraud monitoring dashboard</h1>
            <p className="page-subtitle">
              Analyze transactions, understand fraud signals, and review risk decisions.
            </p>
          </div>

          <button
            className="refresh-button"
            onClick={loadDashboard}
            disabled={refreshing}
          >
            ↻ {refreshing ? "Refreshing..." : "Refresh data"}
          </button>
        </header>

        {error && (
          <div className="error-banner" role="alert">
            <strong>Notice:</strong> {error}
          </div>
        )}

        <section className="stats-grid">
          <StatCard
            title="Scored transactions"
            value={total}
            subtitle="Recorded scoring history"
            icon="▤"
          />
          <StatCard
            title="Approved"
            value={approved}
            subtitle="Below review criteria"
            icon="✓"
          />
          <StatCard
            title="Needs review"
            value={review}
            subtitle="Requires analyst attention"
            icon="⌕"
          />
          <StatCard
            title="Blocked"
            value={blocked}
            subtitle="Highest-severity decisions"
            icon="⊘"
          />
        </section>

        <ManualAnalyzer onAnalyzed={loadDashboard} />

        <section className="panel transactions-panel">
          <div className="panel-heading">
            <div>
              <div className="eyebrow">AUDIT HISTORY</div>
              <h2>Recent scored transactions</h2>
            </div>
            <span className="table-count">
              {transactions.length} records
            </span>
          </div>

          {transactions.length === 0 ? (
            <div className="empty-state">
              No scoring history yet. Analyze a transaction to create the first
              record.
            </div>
          ) : (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Transaction ID</th>
                    <th>Amount</th>
                    <th>ML score</th>
                    <th>Risk</th>
                    <th>Decision</th>
                    <th>Explanation</th>
                  </tr>
                </thead>
                <tbody>
                  {transactions.map((tx, index) => (
                    <tr key={tx.txn_id ?? tx.id ?? index}>
                      <td className="transaction-id">
                        {tx.txn_id ?? "—"}
                      </td>
                      <td>{formatMoney(tx.amount)}</td>
                      <td>
                        {tx.risk_score == null
                          ? "—"
                          : `${(Number(tx.risk_score) * 100).toFixed(2)}%`}
                      </td>
                      <td><RiskBadge value={tx.risk_level} /></td>
                      <td>
                        <span
                          className={`decision decision-${String(
                            tx.decision || ""
                          ).toLowerCase()}`}
                        >
                          {tx.decision ?? "—"}
                        </span>
                      </td>
                      <td className="explanation-cell">
                        {tx.explanation ?? "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
        <hr style={{ border: "none", borderTop: "1px solid rgba(119, 120, 120, 0.79)", margin: "2rem 0" }} />
        <section className="content-grid pipeline-only">
          <div className="panel pipeline-panel">
            <div className="eyebrow">DECISION PIPELINE</div>
            <h2>How a transaction is assessed</h2>
            <p className="muted">
              Two independent signals contribute to the prototype's decision.
            </p>

            <div className="pipeline-step">
              <div className="pipeline-icon">01</div>
              <div>
                <strong>Transaction features</strong>
                <p>Amount, history, payee behaviour, device and timing.</p>
              </div>
            </div>
            <div className="pipeline-line" />
            <div className="pipeline-step">
              <div className="pipeline-icon">02</div>
              <div>
                <strong>Explicit fraud rules</strong>
                <p>Evaluate configured conditions and explain matched rules.</p>
              </div>
            </div>
            <div className="pipeline-line" />
            <div className="pipeline-step">
              <div className="pipeline-icon">03</div>
              <div>
                <strong>Random Forest model</strong>
                <p>Estimate fraud probability from engineered features.</p>
              </div>
            </div>
            <div className="pipeline-line" />
            <div className="pipeline-step">
              <div className="pipeline-icon">04</div>
              <div>
                <strong>Decision and audit history</strong>
                <p>Combine signals and persist the result in SQLite.</p>
              </div>
            </div>

            <div className="pipeline-footnote">
              Prototype policy: a BLOCK rule blocks; HIGH severity or an ML
              score above threshold sends the transaction for review. This is
              not a production bank authorization system.
            </div>
          </div>
        </section>

        <footer className="footer">
          SentinelPay · Hybrid fraud detection prototype · Synthetic dataset
        </footer>
      </main>
    </div>
  );
}