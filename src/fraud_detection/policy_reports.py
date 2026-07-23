"""Shared threshold-policy reports for optimization and model promotion."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from fraud_detection.threshold import add_business_costs, select_best_threshold

BUSINESS_SCENARIOS = {
    "aggressive": dict(investigation_cost=5.0, false_positive_cost=5.0, fixed_false_negative_cost=500.0, fraud_amount_multiplier=1.0, minimum_recall=0.90),
    "balanced": dict(investigation_cost=10.0, false_positive_cost=10.0, fixed_false_negative_cost=300.0, fraud_amount_multiplier=1.0, minimum_recall=0.80),
    "customer_friendly": dict(investigation_cost=20.0, false_positive_cost=30.0, fixed_false_negative_cost=200.0, fraud_amount_multiplier=1.0, minimum_recall=0.60),
}


def build_policy_recommendations(
    table: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Build statistical and cost-based threshold recommendations."""
    rows = [
        {"strategy": "max_f1", **table.sort_values(["f1", "precision", "alerts"], ascending=[False, False, True]).iloc[0].to_dict()},
        {"strategy": "minimum_recall_80", **table.loc[table["recall"] >= 0.80].sort_values(["precision", "alerts"], ascending=[False, True]).iloc[0].to_dict()},
        {"strategy": "maximum_500_alerts", **table.loc[table["alerts"] <= 500].sort_values(["recall", "precision"], ascending=False).iloc[0].to_dict()},
    ]
    cost_tables = {}
    for name, assumptions in BUSINESS_SCENARIOS.items():
        cost_table = add_business_costs(
            table,
            **{key: value for key, value in assumptions.items() if key != "minimum_recall"},
        )
        cost_tables[name] = cost_table
        selected = select_best_threshold(cost_table, assumptions["minimum_recall"])
        rows.append({"strategy": f"business_{name}", **selected.to_dict(), **{f"assumption_{key}": value for key, value in assumptions.items()}})
    return pd.DataFrame(rows), cost_tables


def save_policy_figures(
    table: pd.DataFrame,
    cost_tables: dict[str, pd.DataFrame],
    threshold: float,
    figures_dir: Path,
    model_id: str | None = None,
) -> None:
    """Write the two policy charts consumed by the dashboard."""
    suffix = f" — {model_id}" if model_id else ""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for metric in ("precision", "recall", "f1"):
        axes[0].plot(table["threshold"], table[metric], label=metric.title())
    axes[0].axvline(threshold, color="black", linestyle="--", label="Active policy")
    axes[0].set(title=f"Validation Metrics{suffix}", xlabel="Threshold", ylabel="Metric")
    axes[0].legend()
    axes[1].plot(table["threshold"], table["alerts"], color="#c96f3b")
    axes[1].axvline(threshold, color="black", linestyle="--")
    axes[1].set(title="Validation Alert Volume", xlabel="Threshold", ylabel="Alerts")
    fig.tight_layout()
    fig.savefig(figures_dir / "threshold_tradeoff.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name, cost_table in cost_tables.items():
        ax.plot(cost_table["threshold"], cost_table["total_cost"], label=name)
    ax.axvline(threshold, color="black", linestyle="--", label="Active policy")
    ax.set(title=f"Business Cost Sensitivity{suffix}", xlabel="Threshold", ylabel="Estimated total cost")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "business_cost_sensitivity.png", dpi=160)
    plt.close(fig)
