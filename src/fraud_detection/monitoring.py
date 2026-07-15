"""Statistik sederhana untuk memonitor perubahan distribusi data."""

from __future__ import annotations

import numpy as np
import pandas as pd


def population_stability_index(
    reference: pd.Series,
    current: pd.Series,
    bins: int = 10,
) -> float:
    """Hitung PSI menggunakan quantile bins dari reference distribution."""
    reference = pd.to_numeric(reference, errors="coerce").dropna()
    current = pd.to_numeric(current, errors="coerce").dropna()
    if reference.empty or current.empty:
        raise ValueError("Reference dan current tidak boleh kosong")

    edges = np.unique(reference.quantile(np.linspace(0, 1, bins + 1)).to_numpy())
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts = pd.cut(reference, edges, include_lowest=True).value_counts(sort=False)
    cur_counts = pd.cut(current, edges, include_lowest=True).value_counts(sort=False)
    epsilon = 1e-6
    ref_share = np.clip(ref_counts.to_numpy() / len(reference), epsilon, None)
    cur_share = np.clip(cur_counts.to_numpy() / len(current), epsilon, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def drift_level(psi: float) -> str:
    """Petakan PSI ke kategori Stable, Monitor, atau Investigate."""
    if psi < 0.10:
        return "Stable"
    if psi < 0.25:
        return "Monitor"
    return "Investigate"

