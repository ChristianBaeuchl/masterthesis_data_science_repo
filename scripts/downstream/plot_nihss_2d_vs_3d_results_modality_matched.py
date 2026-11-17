#!/usr/bin/env python3
"""Create boxplots"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot primary and modality-specific NIHSS absolute-error comparisons."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Output directory produced by the NIHSS 2D-vs-3D A/B/C analysis script.",
    )
    return parser.parse_args()


def _format_p(p: float) -> str:
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


def _save(fig: plt.Figure, base_path: Path) -> None:
    fig.savefig(base_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _paired_boxplot(
    x: np.ndarray,
    y: np.ndarray,
    labels: Sequence[str],
    ylabel: str,
    title: str,
    p_value: float,
    output_base: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.boxplot([x, y], positions=[1, 2], widths=0.5, showfliers=False)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(list(labels))

    # Paired observations are connected to make the within-patient comparison visible.
    for xv, yv in zip(x, y):
        ax.plot([1, 2], [xv, yv], marker="o", markersize=2.5, linewidth=0.5, alpha=0.18)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.text(
        0.5,
        0.98,
        f"{_format_p(p_value)}; n = {len(x)}",
        transform=ax.transAxes,
        ha="center",
        va="top",
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
    fig, ax = plt.subplots(figsize=(10.5, 5.8))

    data = []
    positions = []
    labels = []
    group_centers = []

    for i, modality in enumerate(modalities):
        base = 1 + i * 3
        a_pos, b_pos = base, base + 1
        a = errors[f"abs_error_{modality}_{rep_a_key}"].to_numpy(dtype=float)
        b = errors[f"abs_error_{modality}_{rep_b_key}"].to_numpy(dtype=float)

        data.extend([a, b])
        positions.extend([a_pos, b_pos])
        labels.extend([rep_a_label, rep_b_label])
        group_centers.append((a_pos + b_pos) / 2)

        # A light paired overlay shows the repeated-measures structure.
        for av, bv in zip(a, b):
            ax.plot(
                [a_pos, b_pos],
                [av, bv],
                marker="o",
                markersize=1.8,
                linewidth=0.35,
                alpha=0.07,
            )

    ax.boxplot(data, positions=positions, widths=0.6, showfliers=False)

    # Representation labels below individual boxes.
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Absolute NIHSS prediction error")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Modality labels and Holm-adjusted p-values above groups.
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    ax.set_ylim(y_min, y_max + 0.18 * y_range)
    for center, modality in zip(group_centers, modalities):
        row = stats.loc[stats["modality"] == modality].iloc[0]
        dimension_text = (
            f"\nd = {int(row['pca_components'])}"
            if "pca_components" in row.index and pd.notna(row["pca_components"])
            else ""
        )
        ax.text(
            center,
            y_max + 0.04 * y_range,
            f"{modality}{dimension_text}\nHolm {_format_p(float(row['p_value_Holm']))}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    _save(fig, output_base)


def _select_variant_rows(
    table: pd.DataFrame,
    variant: str,
) -> pd.DataFrame:
    """Select rows for one A/B analysis variant.

    Older result files did not contain an ``analysis_variant`` column. For
    backward compatibility, such files are treated as variant A only.
    """
    if "analysis_variant" not in table.columns:
        if variant == "A_embedding_only":
            return table.copy()
        return table.iloc[0:0].copy()

    return table.loc[table["analysis_variant"] == variant].copy()


def _comparison_p_value(row: pd.Series) -> float:
    """Return the preferred p-value from a comparison-table row."""
    if "p_value_Holm" in row.index and pd.notna(row["p_value_Holm"]):
        return float(row["p_value_Holm"])
    return float(row["p_value_raw"])


def _require_error_columns(errors: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in errors.columns]
    if missing:
        raise ValueError(
            "patient_level_absolute_errors.csv is missing required columns: "
            f"{missing}"
        )


def create_plots(results_dir: Path) -> None:
    results_dir = results_dir.expanduser().resolve()
    errors_path = results_dir / "patient_level_absolute_errors.csv"
    primary_path = results_dir / "primary_2d_vs_3d_comparison.csv"
    secondary_path = results_dir / "secondary_modality_specific_2d_vs_3d.csv"

    for path in (errors_path, primary_path, secondary_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required result file not found: {path}")

    errors = pd.read_csv(errors_path, dtype={"patient_id": str})
    primary = pd.read_csv(primary_path)
    secondary = pd.read_csv(secondary_path)

    # The combined A/B result files contain each modality once per analysis
    # variant. Therefore modalities must be taken from one variant only rather
    # than directly from all rows of the secondary table.
    secondary_a = _select_variant_rows(secondary, "A_embedding_only")
    if secondary_a.empty:
        raise ValueError(
            "No A_embedding_only rows were found in "
            "secondary_modality_specific_2d_vs_3d.csv."
        )
    modalities = list(dict.fromkeys(secondary_a["modality"].astype(str).tolist()))

    # ------------------------------------------------------------------
    # Variant A: embedding only
    # ------------------------------------------------------------------
    primary_a = _select_variant_rows(primary, "A_embedding_only")
    if primary_a.empty:
        raise ValueError(
            "No A_embedding_only row was found in primary_2d_vs_3d_comparison.csv."
        )

    _require_error_columns(
        errors,
        ["mean_abs_error_2d", "mean_abs_error_3d"],
    )
    _paired_boxplot(
        x=errors["mean_abs_error_2d"].to_numpy(dtype=float),
        y=errors["mean_abs_error_3d"].to_numpy(dtype=float),
        labels=("2D", "3D"),
        ylabel="Prediction error across modalities",
        title="Variant A: embedding-only 2D vs 3D comparison",
        p_value=_comparison_p_value(primary_a.iloc[0]),
        output_base=results_dir / "figure_primary_2d_vs_3d",
    )

    _modality_boxplots(
        errors=errors,
        modalities=modalities,
        rep_a_key="2d",
        rep_b_key="3d",
        rep_a_label="2D",
        rep_b_label="3D",
        stats=secondary_a,
        title="Variant A: modality-specific 2D vs 3D comparisons",
        output_base=results_dir / "figure_modality_specific_2d_vs_3d",
    )

    # ------------------------------------------------------------------
    # Variant B: embedding + clinical predictors
    # ------------------------------------------------------------------
    primary_b = _select_variant_rows(primary, "B_embedding_plus_clinical")
    secondary_b = _select_variant_rows(secondary, "B_embedding_plus_clinical")

    if not primary_b.empty:
        _require_error_columns(
            errors,
            ["mean_abs_error_2d_clinical", "mean_abs_error_3d_clinical"],
        )
        _paired_boxplot(
            x=errors["mean_abs_error_2d_clinical"].to_numpy(dtype=float),
            y=errors["mean_abs_error_3d_clinical"].to_numpy(dtype=float),
            labels=("2D", "3D"),
            ylabel="Prediction error across modalities",
            title="2D (concatenated) vs 3D (unimodal)",
            p_value=_comparison_p_value(primary_b.iloc[0]),
            output_base=(
                results_dir / "figure_2d_vs_3d"
            ),
        )

        if secondary_b.empty:
            raise ValueError(
                "Variant B is present in the primary comparison but no "
                "B_embedding_plus_clinical rows were found in the secondary "
                "comparison table."
            )
        _modality_boxplots(
            errors=errors,
            modalities=modalities,
            rep_a_key="2d_clinical",
            rep_b_key="3d_clinical",
            rep_a_label="2D",
            rep_b_label="3D",
            stats=secondary_b,
            title="Modality-specific 2D vs 3D + clinical comparisons",
            output_base=(
                results_dir / "figure_modality_specific_2d_vs_3d_embedding_plus_clinical"
            ),
        )

    # ------------------------------------------------------------------
    # Dimension-matched PCA sensitivity analyses for variants A and B
    # ------------------------------------------------------------------
    sens_primary_path = (
        results_dir / "sensitivity_dimension_matched_2d_vs_3d_comparison.csv"
    )
    sens_secondary_path = (
        results_dir
        / "sensitivity_modality_specific_dimension_matched_2d_vs_3d.csv"
    )

    if sens_primary_path.is_file() and sens_secondary_path.is_file():
        sens_primary = pd.read_csv(sens_primary_path)
        sens_secondary = pd.read_csv(sens_secondary_path)

        # Variant A sensitivity
        sens_primary_a = _select_variant_rows(sens_primary, "A_embedding_only")
        sens_secondary_a = _select_variant_rows(sens_secondary, "A_embedding_only")
        if not sens_primary_a.empty:
            _require_error_columns(
                errors,
                ["mean_abs_error_2d_pca_matched", "mean_abs_error_3d"],
            )
            _paired_boxplot(
                x=errors["mean_abs_error_2d_pca_matched"].to_numpy(dtype=float),
                y=errors["mean_abs_error_3d"].to_numpy(dtype=float),
                labels=("2D dim.-matched", "3D"),
                ylabel="Prediction error across modalities",
                title="Dimension-matched sensitivity analysis",
                p_value=_comparison_p_value(sens_primary_a.iloc[0]),
                output_base=(
                    results_dir / "figure_sensitivity_dimension_matched_2d_vs_3d"
                ),
            )

            if not sens_secondary_a.empty:
                _modality_boxplots(
                    errors=errors,
                    modalities=modalities,
                    rep_a_key="2d_pca_matched",
                    rep_b_key="3d",
                    rep_a_label="PCA-matched",
                    rep_b_label="3D",
                    stats=sens_secondary_a,
                    title=(
                        "Variant A: modality-specific dimension-matched 2D vs 3D "
                        "sensitivity"
                    ),
                    output_base=(
                        results_dir
                        / "figure_sensitivity_modality_specific_dimension_matched_2d_vs_3d"
                    ),
                )

        # Variant B sensitivity: PCA is applied to the 2D embedding block only;
        # the clinical predictors are the same additional variables on both sides.
        sens_primary_b = _select_variant_rows(
            sens_primary, "B_embedding_plus_clinical"
        )
        sens_secondary_b = _select_variant_rows(
            sens_secondary, "B_embedding_plus_clinical"
        )
        if not sens_primary_b.empty:
            _require_error_columns(
                errors,
                [
                    "mean_abs_error_2d_pca_matched_clinical",
                    "mean_abs_error_3d_clinical",
                ],
            )
            _paired_boxplot(
                x=errors[
                    "mean_abs_error_2d_pca_matched_clinical"
                ].to_numpy(dtype=float),
                y=errors["mean_abs_error_3d_clinical"].to_numpy(dtype=float),
                labels=("2D (dimension-matched)", "3D"),
                ylabel="Prediction error across modalities",
                title=(
                    "Dimension-matched sensitivity analysis"
                ),
                p_value=_comparison_p_value(sens_primary_b.iloc[0]),
                output_base=(
                    results_dir
                    / "figure_sensitivity_dimension_matched_2d_vs_3d_embedding_plus_clinical"
                ),
            )

            if not sens_secondary_b.empty:
                _modality_boxplots(
                    errors=errors,
                    modalities=modalities,
                    rep_a_key="2d_pca_matched_clinical",
                    rep_b_key="3d_clinical",
                    rep_a_label="PCA-matched",
                    rep_b_label="3D",
                    stats=sens_secondary_b,
                    title=(
                        "Variant B: modality-specific dimension-matched 2D vs 3D "
                        "+ clinical sensitivity"
                    ),
                    output_base=(
                        results_dir
                        / "figure_sensitivity_modality_specific_dimension_matched_2d_vs_3d_embedding_plus_clinical"
                    ),
                )

    print(f"Figures saved as PNG (300 dpi) and PDF in: {results_dir}")

def main() -> None:
    args = parse_args()
    create_plots(args.results_dir)


if __name__ == "__main__":
    main()
