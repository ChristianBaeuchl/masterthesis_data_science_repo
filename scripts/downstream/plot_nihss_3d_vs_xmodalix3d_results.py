#!/usr/bin/env python3
"""Plot held-out NIHSS results for standalone Imagix3D vs aligned XModalix3D."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _format_p(p: float) -> str:
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


def _save(fig: plt.Figure, base: Path) -> None:
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _paired_boxplot(
    a: np.ndarray,
    b: np.ndarray,
    labels: tuple[str, str],
    title: str,
    p_value: float,
    output_base: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    positions = [1, 2]
    for av, bv in zip(a, b):
        ax.plot(
            positions,
            [av, bv],
            marker="o",
            markersize=2.4,
            linewidth=0.55,
            alpha=0.13,
        )
    ax.boxplot([a, b], positions=positions, widths=0.5, showfliers=False)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Prediction error across modalities")
    ax.set_title(title)
    ax.text(
        0.5, 0.98, f"{_format_p(p_value)}; n = {len(a)}",
        transform=ax.transAxes, ha="center", va="top"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, output_base)


def _modality_boxplots(
    errors: pd.DataFrame,
    modalities: Sequence[str],
    rep_a_key: str,
    rep_b_key: str,
    rep_a_label: str,
    rep_b_label: str,
    stats: pd.DataFrame,
    title: str,
    output_base: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 5.8))
    data, positions, labels, centers = [], [], [], []
    for i, modality in enumerate(modalities):
        base = 1 + i * 3
        a_pos, b_pos = base, base + 1
        a = errors[f"abs_error_{modality}_{rep_a_key}"].to_numpy(dtype=float)
        b = errors[f"abs_error_{modality}_{rep_b_key}"].to_numpy(dtype=float)
        data.extend([a, b])
        positions.extend([a_pos, b_pos])
        labels.extend([rep_a_label, rep_b_label])
        centers.append((a_pos + b_pos) / 2)
        for av, bv in zip(a, b):
            ax.plot([a_pos, b_pos], [av, bv], marker="o", markersize=1.8,
                    linewidth=0.35, alpha=0.08)

    ax.boxplot(data, positions=positions, widths=0.6, showfliers=False)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Prediction error")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    y_min, y_max = ax.get_ylim()
    yr = y_max - y_min
    ax.set_ylim(y_min, y_max + 0.18 * yr)
    for center, modality in zip(centers, modalities):
        row = stats.loc[stats["modality"] == modality].iloc[0]
        dim = ""
        if "pca_components" in row.index and pd.notna(row["pca_components"]):
            dim = f"\nd = {int(row['pca_components'])}"
        ax.text(
            center, y_max + 0.04 * yr,
            f"{modality}{dim}\nHolm {_format_p(float(row['p_value_Holm']))}",
            ha="center", va="bottom", fontsize=9,
        )
    fig.tight_layout()
    _save(fig, output_base)


def _primary_p_value(results_dir: Path, primary: pd.DataFrame) -> float:
    family_path = results_dir / "primary_thesis_holm_family.csv"
    if family_path.is_file():
        family = pd.read_csv(family_path)
        row = family.loc[family["hypothesis"] == "3D vs aligned XModalix3D"]
        if len(row) == 1:
            return float(row.iloc[0]["p_value_Holm"])
    return float(primary.iloc[0]["p_value_raw"])


def create_plots(results_dir: Path) -> None:
    results_dir = Path(results_dir).expanduser().resolve()
    errors = pd.read_csv(results_dir / "patient_level_absolute_errors.csv", dtype={"patient_id": str})
    primary = pd.read_csv(results_dir / "primary_3d_vs_xmodalix3d_comparison.csv")
    secondary = pd.read_csv(results_dir / "secondary_modality_specific_3d_vs_xmodalix3d.csv")
    modalities = secondary["modality"].astype(str).tolist()

    _paired_boxplot(
        errors["mean_abs_error_3d"].to_numpy(dtype=float),
        errors["mean_abs_error_3d_aligned"].to_numpy(dtype=float),
        ("3D", "3D aligned"),
        "3D (unimodal) vs 3D (aligned))",
        _primary_p_value(results_dir, primary),
        results_dir / "figure_primary_3d_vs_xmodalix3d",
    )
    _modality_boxplots(
        errors, modalities,
        "3d", "3d_aligned", "3D", "3D aligned",
        secondary,
        "Modality-specific 3D (unimodal) vs 3D (aligned)",
        results_dir / "figure_modality_specific_3d_vs_xmodalix3d",
    )

    sens_primary_path = results_dir / "sensitivity_dimension_matched_3d_vs_xmodalix3d_comparison.csv"
    sens_secondary_path = results_dir / "sensitivity_modality_specific_dimension_matched_3d_vs_xmodalix3d.csv"
    if sens_primary_path.is_file() and sens_secondary_path.is_file():
        sens_primary = pd.read_csv(sens_primary_path)
        sens_secondary = pd.read_csv(sens_secondary_path)
        _paired_boxplot(
            errors["mean_abs_error_3d_pca_matched"].to_numpy(dtype=float),
            errors["mean_abs_error_3d_aligned"].to_numpy(dtype=float),
            ("3D PCA-matched", "3D aligned"),
            "Dimension-matched 3D vs aligned 3D sensitivity analysis",
            float(sens_primary.iloc[0]["p_value_raw"]),
            results_dir / "figure_sensitivity_dimension_matched_3d_vs_xmodalix3d",
        )
        _modality_boxplots(
            errors, modalities,
            "3d_pca_matched", "3d_aligned", "3D PCA-matched", "3D aligned",
            sens_secondary,
            "Modality-specific dimension-matched 3D vs aligned 3D",
            results_dir / "figure_sensitivity_modality_specific_3d_vs_xmodalix3d",
        )

    print(f"Figures saved as PNG and PDF in: {results_dir}")
