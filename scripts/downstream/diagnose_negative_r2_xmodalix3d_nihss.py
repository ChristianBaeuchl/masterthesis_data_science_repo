#!/usr/bin/env python3
"""Diagnose why held-out aligned XModalix3D NIHSS predictions produce negative R² values.

This script is intended for the output produced by the held-out
Imagix3D-vs-XModalix3D NIHSS analysis.

By default it analyses the thesis-relevant aligned XModalix3D predictions
(``representation == '3D_ALIGNED'``) from
``test_predictions_all_representations.csv``.  The comparison analysis already
uses the clinical predictors (acute NIHSS and onset-to-CT), so there is no
separate ``analysis_variant`` column in this output format.

For each imaging modality it compares the model with the *test-set mean* only
for the mathematical decomposition of R².  This is NOT a predictive baseline
that is fitted on the test set; it is the reference term that appears in the
standard definition of test-set R²:

    R² = 1 - SSE_model / SSE_test_mean

Outputs
-------
1. r2_diagnostic_summary.csv
   One row per modality with MAE, MedAE, R², prediction spread, SSE values,
   and how often / by how much the model beats the test-mean reference.

2. r2_patient_level_diagnostics.csv
   One row per patient and modality with the model prediction, residuals,
   squared error of the model, squared error of the test-mean reference, and
   their difference. Positive ``excess_squared_error_vs_mean`` values are the
   observations that push R² downward.

3. r2_nihss_bin_diagnostics.csv
   Diagnostic aggregation by observed NIHSS range. These bins are only a
   descriptive aid; they are not used for inference.

4. r2_top_excess_error_contributors.csv
   The patient/modality observations that contribute most strongly to making
   R² worse than the test-mean reference.

5. figure_r2_diagnostics.png / .pdf
   For every modality:
     left  - observed versus predicted NIHSS, with identity line and test mean;
     right - excess squared error versus observed NIHSS. Values above zero are
             worse than predicting the test-set mean for that patient.

The script performs no model fitting and therefore does not alter or rerun the
SVR analysis.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score


DEFAULT_REPRESENTATION = "3D_ALIGNED"
DEFAULT_MODALITY_ORDER = ("CBF", "NCCT")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose negative held-out R² values in NIHSS prediction results."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("test_predictions_all_representations.csv"),
        help="Path to test_predictions_all_representations.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("r2_diagnostics_xmodalix3d_aligned"),
        help="Directory for diagnostic CSVs and figures.",
    )
    parser.add_argument(
        "--representation",
        default=DEFAULT_REPRESENTATION,
        help=f"Representation to inspect (default: {DEFAULT_REPRESENTATION}).",
    )
    parser.add_argument(
        "--modalities",
        nargs="*",
        default=None,
        help=(
            "Optional modality list. If omitted, all available modalities are used "
            "in the conventional CBF, NCCT order."
        ),
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of largest excess-squared-error observations to export.",
    )
    return parser.parse_args()


def _validate_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Predictions file is missing required columns: {missing}")


def _ordered_modalities(df: pd.DataFrame, requested: list[str] | None) -> list[str]:
    available = list(dict.fromkeys(df["modality"].astype(str)))
    if requested:
        missing = sorted(set(requested) - set(available))
        if missing:
            raise ValueError(
                f"Requested modalities are absent from the selected predictions: {missing}. "
                f"Available: {available}"
            )
        return list(requested)

    ordered = [m for m in DEFAULT_MODALITY_ORDER if m in available]
    ordered.extend(m for m in available if m not in ordered)
    return ordered


def build_diagnostics(
    predictions: pd.DataFrame,
    representation: str,
    modalities: list[str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = predictions.loc[
        predictions["representation"].astype(str) == representation
    ].copy()

    if selected.empty:
        available = sorted(predictions["representation"].astype(str).unique().tolist())
        raise ValueError(
            f"No rows match representation={representation!r}. "
            f"Available representations: {available}"
        )

    modality_order = _ordered_modalities(selected, modalities)
    selected = selected.loc[selected["modality"].isin(modality_order)].copy()

    # Patient-modality pairs must be unique.
    duplicated = selected.duplicated(["patient_id", "modality"], keep=False)
    if duplicated.any():
        example = selected.loc[duplicated, ["patient_id", "modality"]].head()
        raise ValueError(
            "Duplicate patient/modality prediction rows found. Examples:\n"
            + example.to_string(index=False)
        )

    patient_rows: list[pd.DataFrame] = []
    summary_rows: list[dict] = []

    for modality in modality_order:
        cur = selected.loc[selected["modality"] == modality].copy()
        if cur.empty:
            continue

        observed = cur["nihss_observed"].to_numpy(dtype=float)
        predicted = cur["nihss_predicted"].to_numpy(dtype=float)
        if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
            raise ValueError(f"Non-finite values found for modality {modality}.")

        test_mean = float(np.mean(observed))
        residual = observed - predicted
        abs_error = np.abs(residual)
        squared_error_model = residual**2
        squared_error_mean = (observed - test_mean) ** 2
        excess = squared_error_model - squared_error_mean

        sse_model = float(np.sum(squared_error_model))
        sse_mean = float(np.sum(squared_error_mean))
        calculated_r2 = 1.0 - (sse_model / sse_mean)
        sklearn_r2 = float(r2_score(observed, predicted))
        if not np.isclose(calculated_r2, sklearn_r2, atol=1e-12, rtol=1e-10):
            raise RuntimeError(
                f"R² decomposition mismatch for {modality}: "
                f"manual={calculated_r2}, sklearn={sklearn_r2}"
            )

        better = excess < 0
        worse = excess > 0
        improvement = float(-np.sum(excess[better]))
        deterioration = float(np.sum(excess[worse]))

        patient_diag = cur[
            ["patient_id", "data_scope", "modality", "representation"]
        ].copy()
        patient_diag["nihss_observed"] = observed
        patient_diag["nihss_predicted"] = predicted
        patient_diag["test_mean_reference"] = test_mean
        patient_diag["residual_observed_minus_predicted"] = residual
        patient_diag["absolute_error_model"] = abs_error
        patient_diag["absolute_error_mean_reference"] = np.abs(observed - test_mean)
        patient_diag["squared_error_model"] = squared_error_model
        patient_diag["squared_error_mean_reference"] = squared_error_mean
        patient_diag["excess_squared_error_vs_mean"] = excess
        patient_diag["model_better_than_mean_by_squared_error"] = better
        patient_diag["share_of_modality_model_sse"] = squared_error_model / sse_model
        patient_diag["r2_change_contribution_vs_mean"] = -excess / sse_mean
        patient_rows.append(patient_diag)

        summary_rows.append(
            {
                "modality": modality,
                "n_patients": len(cur),
                "observed_mean": test_mean,
                "observed_median": float(np.median(observed)),
                "observed_min": float(np.min(observed)),
                "observed_max": float(np.max(observed)),
                "observed_sd": float(np.std(observed, ddof=1)),
                "predicted_mean": float(np.mean(predicted)),
                "predicted_median": float(np.median(predicted)),
                "predicted_min": float(np.min(predicted)),
                "predicted_max": float(np.max(predicted)),
                "predicted_sd": float(np.std(predicted, ddof=1)),
                "prediction_range": float(np.max(predicted) - np.min(predicted)),
                "MAE": float(mean_absolute_error(observed, predicted)),
                "MedAE": float(median_absolute_error(observed, predicted)),
                "RMSE": float(np.sqrt(np.mean(squared_error_model))),
                "R2": sklearn_r2,
                "SSE_model": sse_model,
                "SSE_test_mean_reference": sse_mean,
                "SSE_excess_model_minus_mean": sse_model - sse_mean,
                "fraction_patients_model_better_than_mean_squared_error": float(
                    np.mean(better)
                ),
                "fraction_patients_model_worse_than_mean_squared_error": float(
                    np.mean(worse)
                ),
                "total_squared_error_improvement_where_model_better": improvement,
                "total_squared_error_deterioration_where_model_worse": deterioration,
                "net_excess_squared_error": deterioration - improvement,
                "largest_single_excess_squared_error": float(np.max(excess)),
                "largest_single_model_squared_error": float(np.max(squared_error_model)),
            }
        )

    patient_df = pd.concat(patient_rows, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    # Purely descriptive NIHSS bins to reveal where SSE accumulates.
    # These are not used for statistical testing.
    bins = [-np.inf, 4, 9, 14, np.inf]
    labels = ["0-4", "5-9", "10-14", ">=15"]
    patient_df["observed_nihss_bin"] = pd.cut(
        patient_df["nihss_observed"], bins=bins, labels=labels
    )

    bin_rows: list[dict] = []
    for (modality, nihss_bin), group in patient_df.groupby(
        ["modality", "observed_nihss_bin"], observed=False
    ):
        if group.empty:
            continue
        modality_sse = float(
            summary_df.loc[summary_df["modality"] == modality, "SSE_model"].iloc[0]
        )
        model_sse = float(group["squared_error_model"].sum())
        ref_sse = float(group["squared_error_mean_reference"].sum())
        bin_rows.append(
            {
                "modality": modality,
                "observed_nihss_bin": str(nihss_bin),
                "n_patients": len(group),
                "mean_observed_nihss": float(group["nihss_observed"].mean()),
                "mean_predicted_nihss": float(group["nihss_predicted"].mean()),
                "MAE": float(group["absolute_error_model"].mean()),
                "MedAE": float(group["absolute_error_model"].median()),
                "SSE_model": model_sse,
                "SSE_mean_reference": ref_sse,
                "excess_SSE_model_minus_mean": model_sse - ref_sse,
                "share_of_modality_model_SSE": model_sse / modality_sse,
                "fraction_model_better_than_mean_squared_error": float(
                    group["model_better_than_mean_by_squared_error"].mean()
                ),
            }
        )

    bin_df = pd.DataFrame(bin_rows)
    return summary_df, patient_df, bin_df


def make_figure(
    summary_df: pd.DataFrame,
    patient_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    modalities = summary_df["modality"].tolist()
    n_rows = len(modalities)
    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(11.5, max(3.1 * n_rows, 4.5)),
        squeeze=False,
    )

    global_min = float(
        min(patient_df["nihss_observed"].min(), patient_df["nihss_predicted"].min())
    )
    global_max = float(
        max(patient_df["nihss_observed"].max(), patient_df["nihss_predicted"].max())
    )
    margin = max(1.0, 0.05 * (global_max - global_min))
    plot_min = global_min - margin
    plot_max = global_max + margin

    for row, modality in enumerate(modalities):
        cur = patient_df.loc[patient_df["modality"] == modality].copy()
        summary = summary_df.loc[summary_df["modality"] == modality].iloc[0]
        mean_ref = float(summary["observed_mean"])

        ax = axes[row, 0]
        ax.scatter(cur["nihss_observed"], cur["nihss_predicted"], s=22, alpha=0.65)
        ax.plot([plot_min, plot_max], [plot_min, plot_max], linestyle="--", linewidth=1)
        ax.axhline(mean_ref, linestyle=":", linewidth=1.2)
        ax.set_xlim(plot_min, plot_max)
        ax.set_ylim(plot_min, plot_max)
        ax.set_xlabel("Observed discharge NIHSS")
        ax.set_ylabel("Predicted discharge NIHSS")
        ax.set_title(
            f"{modality}: observed vs predicted\n"
            f"$R^2$={summary['R2']:.3f}, MedAE={summary['MedAE']:.2f}"
        )

        ax = axes[row, 1]
        ax.scatter(
            cur["nihss_observed"],
            cur["excess_squared_error_vs_mean"],
            s=22,
            alpha=0.65,
        )
        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.set_xlabel("Observed discharge NIHSS")
        ax.set_ylabel("Model SE − test-mean SE")
        ax.set_title(
            f"{modality}: contribution relative to mean\n"
            f"{100 * summary['fraction_patients_model_better_than_mean_squared_error']:.1f}% "
            "of patients favor model"
        )

        # Label the three observations that hurt R² most for this modality.
        worst = cur.nlargest(3, "excess_squared_error_vs_mean")
        for _, point in worst.iterrows():
            ax.annotate(
                str(point["patient_id"]),
                (point["nihss_observed"], point["excess_squared_error_vs_mean"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )

    fig.suptitle(
        "Why can aligned XModalix3D NIHSS prediction have negative $R^2$?\n"
        "Left: prediction spread; right: patient-wise squared-error difference versus the test-mean reference",
        y=1.002,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "figure_r2_diagnostics.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "figure_r2_diagnostics.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    predictions_path = args.predictions.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not predictions_path.is_file():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

        # ``sep=None`` lets pandas detect either comma- or semicolon-separated CSVs,
    # which is useful if a file has previously been opened/saved in Excel.
    predictions = pd.read_csv(
        predictions_path, sep=None, engine="python", dtype={"patient_id": str}
    )
    _validate_columns(
        predictions,
        [
            "patient_id",
            "modality",
            "representation",
            "nihss_observed",
            "nihss_predicted",
        ],
    )

    # The XModalix3D held-out predictions file does not necessarily contain a
    # data_scope column. The analysis that creates this file is strictly test-set
    # only, so add the label here for compatibility with the diagnostic output.
    if "data_scope" not in predictions.columns:
        predictions.insert(1, "data_scope", "test")

    summary_df, patient_df, bin_df = build_diagnostics(
        predictions=predictions,
        representation=args.representation,
        modalities=args.modalities,
    )

    summary_df.to_csv(output_dir / "r2_diagnostic_summary.csv", index=False)
    patient_df.to_csv(output_dir / "r2_patient_level_diagnostics.csv", index=False)
    bin_df.to_csv(output_dir / "r2_nihss_bin_diagnostics.csv", index=False)

    top_df = patient_df.nlargest(args.top_n, "excess_squared_error_vs_mean").copy()
    top_df.to_csv(output_dir / "r2_top_excess_error_contributors.csv", index=False)

    make_figure(summary_df, patient_df, output_dir)

    print("\nR² diagnostic summary")
    print(
        summary_df[
            [
                "modality",
                "n_patients",
                "predicted_min",
                "predicted_max",
                "MedAE",
                "R2",
                "SSE_excess_model_minus_mean",
                "fraction_patients_model_better_than_mean_squared_error",
            ]
        ].to_string(index=False)
    )
    print(f"\nOutputs written to: {output_dir}")


if __name__ == "__main__":
    main()
