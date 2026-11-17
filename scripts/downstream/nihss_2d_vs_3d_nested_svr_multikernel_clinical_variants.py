#!/usr/bin/env python3
"""
Nested-CV NIHSS prediction comparing Imagix 2D and Imagix3D embeddings,
with three analysis variants and a dimension-matched PCA sensitivity analysis.

Analysis variants
-----------------
A) EMBEDDING ONLY
   * 2D: full concatenated patient-wise 2D embedding
   * 3D: patient-wise Imagix3D embedding
   * optional 2D_PCA_MATCHED sensitivity representation

B) EMBEDDING + CLINICAL
   * the same embedding representations augmented with clinical predictors
   * by default: nihss_acute and onset_to_ct
   * clinical predictors are standardized leakage-safely inside CV folds
   * for PCA sensitivity, PCA is fitted ONLY to the embedding block; the
     clinical block is standardized separately and concatenated afterwards

C) CLINICAL ONLY
   * SVR using only the clinical predictors

The same global patient cohort and the same outer fold assignments are used for
all variants. Hyperparameters are selected independently for every fitted model
inside nested CV using mean absolute error.

Primary 2D-vs-3D statistical comparisons are produced separately for variants
A and B. Variant C is a single clinical-only reference model. Additional paired
comparisons quantify whether adding 2D or 3D embeddings improves absolute error
relative to the clinical-only model.

Dimension-matched PCA sensitivity
---------------------------------
For each modality, the 2D embedding is reduced to the dimensionality of the
corresponding 3D embedding. In variant B the leakage-safe transformation is:

    embedding -> StandardScaler -> PCA(d_m) --                                             +--> SVR
    clinical  -> StandardScaler -------------/

Thus clinical predictors are never included in the PCA.

Expected 2D file columns:
    patient_id, split, n_slices, latent_dim_per_slice, vector_length,
    feature_0001, feature_0002, ...

Expected 3D file columns:
    sample_id, split, LatDim_0, LatDim_1, ...

Expected clinical file columns (defaults):
    patients_id, nihss_acute, onset_to_ct, nihss_discharge, custom_splits
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


DEFAULT_C_VALUES = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
DEFAULT_EPSILON_VALUES = (0.1, 0.5, 1.0, 2.0)
DEFAULT_GAMMA_VALUES = ("scale",)
DEFAULT_DEGREE_VALUES = (3,)
DEFAULT_COEF0_VALUES = (0.0,)
SUPPORTED_KERNELS = ("linear", "rbf", "poly", "sigmoid")
SPLIT_NAMES = {"train", "valid", "test"}
DEFAULT_CLINICAL_PREDICTOR_COLS = ("nihss_acute", "onset_to_ct")


@dataclass
class EmbeddingTable:
    modality: str
    representation: str
    path: Path
    ids: np.ndarray
    X: np.ndarray
    feature_names: List[str]
    splits: Dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Nested patient-wise SVR comparison of concatenated 2D Imagix "
            "embeddings with Imagix3D embeddings, including PCA sensitivity."
        )
    )
    parser.add_argument(
        "--embedding-2d",
        action="append",
        required=True,
        metavar="MODALITY=PATH",
        help=(
            "2D embedding CSV or its containing directory. Repeat once per modality. "
            "Example: CBF=C:/data/cbf/imagix_cbf_patientwise_z.csv"
        ),
    )
    parser.add_argument(
        "--embedding-3d",
        action="append",
        required=True,
        metavar="MODALITY=PATH",
        help=(
            "3D embedding CSV or its containing directory. Repeat once per modality. "
            "Example: CBF=C:/data/cbf/results/imagix3d_final_z_all_splits.csv"
        ),
    )
    parser.add_argument(
        "--clinical",
        type=Path,
        default=None,
        help=(
            "Path to sample_characteristics.csv. If omitted, the script searches "
            "<script_dir>/nihss_prediction/ and <script_dir>/nihss_prediciton/."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Default: a timestamped directory under "
            "<script_dir>/nihss_prediction/."
        ),
    )
    parser.add_argument("--target-col", default="nihss_discharge")
    parser.add_argument("--clinical-id-col", default="patients_id")
    parser.add_argument("--clinical-split-col", default="custom_splits")
    parser.add_argument(
        "--data-scope",
        choices=("all", "test"),
        default="all",
        help=(
            "Evaluation scope. 'all' preserves the original nested-CV analysis over "
            "the full common cohort. 'test' performs strict held-out evaluation: "
            "SVR hyperparameters are selected using only VAE train+valid patients, "
            "the final model is refit on train+valid, and metrics/statistics/figures "
            "are computed only on patients whose VAE split is 'test'."
        ),
    )
    parser.add_argument(
        "--clinical-predictor-cols",
        nargs="+",
        default=list(DEFAULT_CLINICAL_PREDICTOR_COLS),
        help=(
            "Clinical predictors used in variants B and C. "
            "Default: nihss_acute onset_to_ct."
        ),
    )
    parser.add_argument("--outer-folds", type=int, default=10)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--pca-components",
        type=int,
        default=None,
        help=(
            "Optional common PCA dimensionality override. By default, the PCA "
            "target is inferred separately for each modality from the number of "
            "LatDim_* columns in its 3D embedding file. For a true dimension-matched "
            "analysis, an override must equal every supplied 3D dimensionality."
        ),
    )
    parser.add_argument(
        "--skip-pca-sensitivity",
        action="store_true",
        help="Skip the PCA-reduced 2D sensitivity analysis.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel jobs for GridSearchCV. Use -1 for all available CPU cores.",
    )
    parser.add_argument(
        "--kernel",
        choices=("linear", "rbf", "poly", "polynomial", "sigmoid"),
        default="linear",
        help=(
            "SVR kernel. 'polynomial' is accepted as an alias for sklearn's "
            "'poly'. Default: linear."
        ),
    )
    parser.add_argument(
        "--c-values",
        nargs="+",
        type=float,
        default=list(DEFAULT_C_VALUES),
        help="Grid of SVR C values.",
    )
    parser.add_argument(
        "--epsilon-values",
        nargs="+",
        type=float,
        default=list(DEFAULT_EPSILON_VALUES),
        help="Grid of epsilon values.",
    )
    parser.add_argument(
        "--gamma-values",
        nargs="+",
        default=list(DEFAULT_GAMMA_VALUES),
        help=(
            "Gamma grid for rbf/poly/sigmoid. Values may be 'scale', 'auto', "
            "or positive numbers. Default: scale only."
        ),
    )
    parser.add_argument(
        "--degree-values",
        nargs="+",
        type=int,
        default=list(DEFAULT_DEGREE_VALUES),
        help="Polynomial degrees for kernel=poly. Default: 3.",
    )
    parser.add_argument(
        "--coef0-values",
        nargs="+",
        type=float,
        default=list(DEFAULT_COEF0_VALUES),
        help=(
            "coef0 grid for poly/sigmoid kernels. Default: 0.0. "
            "Ignored for linear/rbf."
        ),
    )
    parser.add_argument(
        "--wilcoxon-alternative",
        choices=("two-sided", "greater", "less"),
        default="two-sided",
        help=(
            "Alternative for wilcoxon(error_2d, error_3d). "
            "'greater' tests whether 2D errors tend to be larger than 3D errors."
        ),
    )
    parser.add_argument(
        "--allow-vae-split-mismatch",
        action="store_true",
        help=(
            "Do not abort if embedding split labels disagree with clinical "
            "custom_splits. Normally this should NOT be used."
        ),
    )
    return parser.parse_args()


def _parse_mapping(values: Sequence[str], argument_name: str) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"{argument_name} must use MODALITY=PATH syntax, got {value!r}."
            )
        modality, raw_path = value.split("=", 1)
        modality = modality.strip().upper()
        raw_path = raw_path.strip()
        if not modality or not raw_path:
            raise ValueError(
                f"{argument_name} must use non-empty MODALITY=PATH syntax."
            )
        if modality in result:
            raise ValueError(
                f"Duplicate modality {modality!r} supplied for {argument_name}."
            )
        result[modality] = Path(raw_path)
    return result


def _resolve_embedding_path(path: Path, modality: str, representation: str) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"Embedding path does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"Embedding path is neither a file nor a directory: {path}")

    if representation == "2D":
        expected = path / f"imagix_{modality.lower()}_patientwise_z.csv"
    elif representation == "3D":
        expected = path / "imagix3d_final_z_all_splits.csv"
    else:
        raise ValueError(f"Unknown representation: {representation}")

    if not expected.is_file():
        raise FileNotFoundError(
            f"Expected {representation} embedding file was not found: {expected}"
        )
    return expected


def _find_default_clinical(script_dir: Path) -> Path:
    candidates = [
        script_dir / "nihss_prediction" / "sample_characteristics.csv",
        script_dir / "nihss_prediciton" / "sample_characteristics.csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find sample_characteristics.csv automatically. "
        "Pass it explicitly with --clinical. Checked:\n  "
        + "\n  ".join(str(x) for x in candidates)
    )


def _normalise_ids(series: pd.Series) -> pd.Series:
    ids = series.astype("string").str.strip()
    if ids.isna().any() or (ids == "").any():
        raise ValueError("Missing or empty patient/sample IDs were found.")
    return ids.astype(str)


def _suffix_number(name: str) -> Tuple[int, str]:
    match = re.search(r"(\d+)$", str(name))
    return (int(match.group(1)) if match else 10**12, str(name))


def _validate_unique_ids(ids: pd.Series, label: str) -> None:
    duplicated = ids[ids.duplicated(keep=False)]
    if not duplicated.empty:
        examples = duplicated.drop_duplicates().tolist()[:10]
        raise ValueError(
            f"{label} contains duplicate patient IDs. Examples: {examples}"
        )


def load_clinical(
    path: Path,
    id_col: str,
    target_col: str,
    split_col: str,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Clinical file does not exist: {path}")

    df = pd.read_csv(path, dtype={id_col: "string"})
    missing = {id_col, target_col} - set(df.columns)
    if missing:
        raise ValueError(
            f"Clinical file is missing required columns: {sorted(missing)}"
        )

    df = df.copy()
    df[id_col] = _normalise_ids(df[id_col])
    _validate_unique_ids(df[id_col], "Clinical table")

    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    n_missing_target = int(df[target_col].isna().sum())
    if n_missing_target:
        print(
            f"Clinical table: excluding {n_missing_target} patient(s) with missing "
            f"{target_col}.",
            flush=True,
        )
        df = df.loc[df[target_col].notna()].copy()

    if split_col in df.columns:
        df[split_col] = df[split_col].astype("string").str.strip().str.lower()
        invalid = set(df[split_col].dropna().unique()) - SPLIT_NAMES
        if invalid:
            raise ValueError(
                f"Clinical split column {split_col!r} has invalid labels: "
                f"{sorted(invalid)}"
            )

    return df.set_index(id_col, drop=False)


def load_embedding_table(
    path: Path,
    modality: str,
    representation: str,
) -> EmbeddingTable:
    if representation == "2D":
        id_col = "patient_id"
        prefix = "feature_"
    elif representation == "3D":
        id_col = "sample_id"
        prefix = "LatDim_"
    else:
        raise ValueError(f"Unknown representation: {representation}")

    df = pd.read_csv(path, dtype={id_col: "string"})
    if id_col not in df.columns:
        raise ValueError(
            f"{representation} {modality}: expected ID column {id_col!r} in {path}"
        )

    df[id_col] = _normalise_ids(df[id_col])
    _validate_unique_ids(df[id_col], f"{representation} {modality}")

    feature_cols = [
        col for col in df.columns
        if str(col).lower().startswith(prefix.lower())
    ]
    feature_cols = sorted(feature_cols, key=_suffix_number)

    if not feature_cols:
        raise ValueError(
            f"{representation} {modality}: no feature columns beginning with "
            f"{prefix!r} found in {path}"
        )

    X_df = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    if X_df.isna().any().any():
        bad_cols = X_df.columns[X_df.isna().any()].tolist()[:10]
        raise ValueError(
            f"{representation} {modality}: NaN/non-numeric embedding values found. "
            f"First affected columns: {bad_cols}"
        )

    X = X_df.to_numpy(dtype=np.float64)
    if not np.isfinite(X).all():
        raise ValueError(
            f"{representation} {modality}: non-finite embedding values found."
        )

    if representation == "2D" and "vector_length" in df.columns:
        vector_length = pd.to_numeric(df["vector_length"], errors="coerce")
        inconsistent = vector_length.notna() & (vector_length != len(feature_cols))
        if inconsistent.any():
            examples = df.loc[inconsistent, id_col].tolist()[:10]
            raise ValueError(
                f"2D {modality}: vector_length does not equal the number of "
                f"feature columns ({len(feature_cols)}) for patients: {examples}"
            )

    splits: Dict[str, str] = {}
    if "split" in df.columns:
        split_values = df["split"].astype("string").str.strip().str.lower()
        invalid = set(split_values.dropna().unique()) - SPLIT_NAMES
        if invalid:
            raise ValueError(
                f"{representation} {modality}: invalid split labels: "
                f"{sorted(invalid)}"
            )
        splits = {
            sid: str(split_value)
            for sid, split_value in zip(df[id_col].astype(str), split_values)
            if pd.notna(split_value)
        }

    return EmbeddingTable(
        modality=modality,
        representation=representation,
        path=path,
        ids=df[id_col].astype(str).to_numpy(),
        X=X,
        feature_names=[str(col) for col in feature_cols],
        splits=splits,
    )


def _check_vae_split_labels(
    clinical: pd.DataFrame,
    clinical_split_col: str,
    tables: Iterable[EmbeddingTable],
    allow_mismatch: bool,
) -> None:
    if clinical_split_col not in clinical.columns:
        print(
            f"WARNING: clinical split column {clinical_split_col!r} is absent; "
            "embedding split labels cannot be cross-checked.",
            flush=True,
        )
        return

    clinical_split = clinical[clinical_split_col].to_dict()
    mismatches: List[Tuple[str, str, str, str]] = []

    for table in tables:
        for sid, observed_split in table.splits.items():
            expected_split = clinical_split.get(sid)
            if expected_split is None or pd.isna(expected_split):
                continue
            expected_split = str(expected_split).strip().lower()
            if observed_split != expected_split:
                mismatches.append(
                    (
                        table.modality,
                        table.representation,
                        sid,
                        f"{observed_split} != {expected_split}",
                    )
                )

    if mismatches:
        message = (
            "Embedding split labels disagree with clinical custom_splits. "
            f"Number of mismatches: {len(mismatches)}. "
            f"First examples: {mismatches[:10]}"
        )
        if allow_mismatch:
            print("WARNING: " + message, flush=True)
        else:
            raise ValueError(message)


def _subset_matrix(table: EmbeddingTable, patient_ids: Sequence[str]) -> np.ndarray:
    id_to_row = {sid: idx for idx, sid in enumerate(table.ids)}
    missing = [sid for sid in patient_ids if sid not in id_to_row]
    if missing:
        raise ValueError(
            f"{table.representation} {table.modality}: cohort IDs unexpectedly "
            f"missing. Examples: {missing[:10]}"
        )
    return np.vstack([table.X[id_to_row[sid]] for sid in patient_ids])


def _benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return np.asarray([], dtype=float)

    order = np.argsort(p)
    ranked = p[order]
    adjusted_ranked = ranked * n / np.arange(1, n + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted_ranked = np.clip(adjusted_ranked, 0.0, 1.0)

    adjusted = np.empty(n, dtype=float)
    adjusted[order] = adjusted_ranked
    return adjusted


def _holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values, preserving input order."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return np.asarray([], dtype=float)

    order = np.argsort(p)
    ranked = p[order]
    multipliers = n - np.arange(n)
    adjusted_ranked = ranked * multipliers
    adjusted_ranked = np.maximum.accumulate(adjusted_ranked)
    adjusted_ranked = np.clip(adjusted_ranked, 0.0, 1.0)

    adjusted = np.empty(n, dtype=float)
    adjusted[order] = adjusted_ranked
    return adjusted


def _safe_wilcoxon(
    x: Sequence[float],
    y: Sequence[float],
    alternative: str,
) -> Tuple[float, float]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if np.allclose(x_arr - y_arr, 0.0):
        return 0.0, 1.0

    result = wilcoxon(
        x_arr,
        y_arr,
        alternative=alternative,
        zero_method="wilcox",
        method="auto",
    )
    return float(result.statistic), float(result.pvalue)


def _normalise_kernel(kernel: str) -> str:
    kernel = str(kernel).strip().lower()
    if kernel == "polynomial":
        kernel = "poly"
    if kernel not in SUPPORTED_KERNELS:
        raise ValueError(
            f"Unsupported kernel {kernel!r}. Choose from {SUPPORTED_KERNELS}."
        )
    return kernel


def _normalise_gamma_values(values: Sequence[object]) -> Tuple[object, ...]:
    result: List[object] = []
    for value in values:
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"scale", "auto"}:
                result.append(text)
                continue
            try:
                value = float(text)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid gamma value {value!r}. Use 'scale', 'auto', or a "
                    "positive number."
                ) from exc
        numeric = float(value)
        if numeric <= 0:
            raise ValueError("Numeric gamma values must be > 0.")
        result.append(numeric)
    if not result:
        raise ValueError("At least one gamma value is required.")
    return tuple(result)


def _validate_svr_grid_values(
    c_values: Sequence[float],
    epsilon_values: Sequence[float],
    degree_values: Sequence[int],
) -> None:
    if not c_values or any(float(x) <= 0 for x in c_values):
        raise ValueError("All C values must be > 0.")
    if not epsilon_values or any(float(x) < 0 for x in epsilon_values):
        raise ValueError("All epsilon values must be >= 0.")
    if not degree_values or any(int(x) < 1 for x in degree_values):
        raise ValueError("All polynomial degree values must be >= 1.")


def _svr_candidate_dicts(
    kernel: str,
    c_values: Sequence[float],
    epsilon_values: Sequence[float],
    gamma_values: Sequence[object],
    degree_values: Sequence[int],
    coef0_values: Sequence[float],
) -> List[dict]:
    """Return kernel-specific candidate dictionaries in deterministic order."""
    kernel = _normalise_kernel(kernel)
    candidates: List[dict] = []

    if kernel == "linear":
        for C, epsilon in itertools.product(c_values, epsilon_values):
            candidates.append({"C": float(C), "epsilon": float(epsilon)})
    elif kernel == "rbf":
        for C, epsilon, gamma in itertools.product(
            c_values, epsilon_values, gamma_values
        ):
            candidates.append(
                {"C": float(C), "epsilon": float(epsilon), "gamma": gamma}
            )
    elif kernel == "poly":
        for C, epsilon, gamma, degree, coef0 in itertools.product(
            c_values,
            epsilon_values,
            gamma_values,
            degree_values,
            coef0_values,
        ):
            candidates.append(
                {
                    "C": float(C),
                    "epsilon": float(epsilon),
                    "gamma": gamma,
                    "degree": int(degree),
                    "coef0": float(coef0),
                }
            )
    elif kernel == "sigmoid":
        for C, epsilon, gamma, coef0 in itertools.product(
            c_values, epsilon_values, gamma_values, coef0_values
        ):
            candidates.append(
                {
                    "C": float(C),
                    "epsilon": float(epsilon),
                    "gamma": gamma,
                    "coef0": float(coef0),
                }
            )
    return candidates


def _gridsearch_param_grid(
    kernel: str,
    c_values: Sequence[float],
    epsilon_values: Sequence[float],
    gamma_values: Sequence[object],
    degree_values: Sequence[int],
    coef0_values: Sequence[float],
) -> Dict[str, List[object]]:
    """Build the kernel-specific GridSearchCV parameter grid."""
    kernel = _normalise_kernel(kernel)
    grid: Dict[str, List[object]] = {
        "svr__C": [float(x) for x in c_values],
        "svr__epsilon": [float(x) for x in epsilon_values],
    }
    if kernel in {"rbf", "poly", "sigmoid"}:
        grid["svr__gamma"] = list(gamma_values)
    if kernel == "poly":
        grid["svr__degree"] = [int(x) for x in degree_values]
        grid["svr__coef0"] = [float(x) for x in coef0_values]
    elif kernel == "sigmoid":
        grid["svr__coef0"] = [float(x) for x in coef0_values]
    return grid


def _make_pipeline(
    pca_components: Optional[int],
    kernel: str,
) -> Pipeline:
    """Create the standard non-PCA SVR pipeline.

    The PCA sensitivity branch is handled explicitly in ``run_nested_svr`` so
    scaler/PCA transforms are computed only once per inner split and reused
    across all kernel-specific SVR hyperparameter candidates.
    """
    steps = [("scaler", StandardScaler())]
    if pca_components is not None:
        steps.append(("pca", PCA(n_components=pca_components, svd_solver="auto")))
    steps.append(("svr", SVR(kernel=_normalise_kernel(kernel))))
    return Pipeline(steps=steps)


def _score_svr_candidate_on_precomputed_pca(
    kernel: str,
    candidate: Dict[str, object],
    transformed_inner_splits: Sequence[
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ],
) -> Tuple[dict, float]:
    """Evaluate one kernel-specific SVR candidate on precomputed PCA folds."""
    fold_maes: List[float] = []

    for X_inner_train, y_inner_train, X_inner_valid, y_inner_valid in transformed_inner_splits:
        model = SVR(kernel=_normalise_kernel(kernel), **candidate)
        model.fit(X_inner_train, y_inner_train)
        predicted = model.predict(X_inner_valid)
        fold_maes.append(float(np.mean(np.abs(y_inner_valid - predicted))))

    return dict(candidate), float(np.mean(fold_maes))


def _run_pca_inner_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    inner_cv: Sequence[Tuple[np.ndarray, np.ndarray]],
    pca_components: int,
    kernel: str,
    c_values: Sequence[float],
    epsilon_values: Sequence[float],
    gamma_values: Sequence[object],
    degree_values: Sequence[int],
    coef0_values: Sequence[float],
    n_jobs: int,
    X_clinical_train: Optional[np.ndarray] = None,
) -> Tuple[dict, float]:
    """Leakage-safe PCA inner CV, optionally with a separate clinical block.

    PCA is fitted only to X_train (the embedding block). If X_clinical_train is
    supplied, its columns are standardized separately within each inner-training
    fold and concatenated to the PCA-transformed embedding afterwards.
    """
    transformed_inner_splits: List[
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = []

    for inner_train_idx, inner_valid_idx in inner_cv:
        X_inner_train_raw = X_train[inner_train_idx]
        X_inner_valid_raw = X_train[inner_valid_idx]
        y_inner_train = y_train[inner_train_idx]
        y_inner_valid = y_train[inner_valid_idx]

        embedding_scaler = StandardScaler()
        X_inner_train_scaled = embedding_scaler.fit_transform(X_inner_train_raw)
        X_inner_valid_scaled = embedding_scaler.transform(X_inner_valid_raw)

        pca = PCA(n_components=pca_components, svd_solver="auto")
        X_inner_train_model = pca.fit_transform(X_inner_train_scaled)
        X_inner_valid_model = pca.transform(X_inner_valid_scaled)

        if X_clinical_train is not None:
            clinical_scaler = StandardScaler()
            C_inner_train = clinical_scaler.fit_transform(
                X_clinical_train[inner_train_idx]
            )
            C_inner_valid = clinical_scaler.transform(
                X_clinical_train[inner_valid_idx]
            )
            X_inner_train_model = np.column_stack(
                [X_inner_train_model, C_inner_train]
            )
            X_inner_valid_model = np.column_stack(
                [X_inner_valid_model, C_inner_valid]
            )

        transformed_inner_splits.append(
            (X_inner_train_model, y_inner_train, X_inner_valid_model, y_inner_valid)
        )

    candidates = _svr_candidate_dicts(
        kernel=kernel,
        c_values=c_values,
        epsilon_values=epsilon_values,
        gamma_values=gamma_values,
        degree_values=degree_values,
        coef0_values=coef0_values,
    )

    scored = Parallel(n_jobs=n_jobs)(
        delayed(_score_svr_candidate_on_precomputed_pca)(
            kernel=kernel,
            candidate=candidate,
            transformed_inner_splits=transformed_inner_splits,
        )
        for candidate in candidates
    )

    # min() is stable; exact ties resolve to the first candidate in grid order.
    best_params, best_mae = min(scored, key=lambda row: row[1])
    return best_params, best_mae


def run_nested_svr(
    X: np.ndarray,
    y: np.ndarray,
    patient_ids: np.ndarray,
    outer_splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    inner_folds: int,
    random_state: int,
    n_jobs: int,
    pca_components: Optional[int] = None,
    kernel: str = "linear",
    c_values: Sequence[float] = DEFAULT_C_VALUES,
    epsilon_values: Sequence[float] = DEFAULT_EPSILON_VALUES,
    gamma_values: Sequence[object] = DEFAULT_GAMMA_VALUES,
    degree_values: Sequence[int] = DEFAULT_DEGREE_VALUES,
    coef0_values: Sequence[float] = DEFAULT_COEF0_VALUES,
    X_clinical: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run leakage-safe nested CV and return OOF predictions and tuning details.

    Parameters
    ----------
    X:
        Main feature block. For embedding models this is the embedding matrix.
        For the clinical-only model this is the clinical matrix itself.
    X_clinical:
        Optional additional clinical block. For non-PCA models it is concatenated
        with X before the leakage-safe StandardScaler pipeline. For PCA models it
        is standardized separately and appended only AFTER PCA of the embedding.
    """

    kernel = _normalise_kernel(kernel)
    gamma_values = _normalise_gamma_values(gamma_values)
    _validate_svr_grid_values(c_values, epsilon_values, degree_values)

    if len(X) != len(y) or len(y) != len(patient_ids):
        raise ValueError("X, y, and patient_ids must have identical row counts.")

    if X_clinical is not None:
        if len(X_clinical) != len(y):
            raise ValueError("X_clinical must have the same row count as y.")
        if X_clinical.ndim != 2:
            raise ValueError("X_clinical must be a two-dimensional matrix.")
        if not np.isfinite(X_clinical).all():
            raise ValueError("X_clinical contains non-finite values.")

    if pca_components is not None:
        if pca_components < 1:
            raise ValueError("pca_components must be >= 1.")
        if pca_components > X.shape[1]:
            raise ValueError(
                f"PCA requests {pca_components} components but X has only "
                f"{X.shape[1]} features."
            )

    # For non-PCA models, concatenation before the pipeline is safe because the
    # StandardScaler is fitted inside each CV training split by GridSearchCV.
    X_non_pca = (
        np.column_stack([X, X_clinical])
        if X_clinical is not None and pca_components is None
        else X
    )

    oof_prediction = np.full(len(y), np.nan, dtype=float)
    oof_fold = np.full(len(y), -1, dtype=int)
    tuning_rows: List[dict] = []

    candidate_count = len(
        _svr_candidate_dicts(
            kernel,
            c_values,
            epsilon_values,
            gamma_values,
            degree_values,
            coef0_values,
        )
    )

    for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(outer_splits):
        y_train = y[outer_train_idx]

        if inner_folds > len(outer_train_idx):
            raise ValueError(
                f"inner_folds={inner_folds} exceeds outer-training sample size "
                f"{len(outer_train_idx)}."
            )

        inner_cv = list(
            KFold(
                n_splits=inner_folds,
                shuffle=True,
                random_state=random_state + outer_fold,
            ).split(np.arange(len(outer_train_idx)))
        )

        if pca_components is not None:
            X_train = X[outer_train_idx]
            X_test = X[outer_test_idx]
            C_train = (
                X_clinical[outer_train_idx] if X_clinical is not None else None
            )
            C_test = (
                X_clinical[outer_test_idx] if X_clinical is not None else None
            )

            smallest_inner_train = min(len(train_idx) for train_idx, _ in inner_cv)
            max_components = min(X_train.shape[1], smallest_inner_train)
            if pca_components > max_components:
                raise ValueError(
                    f"PCA with {pca_components} components is not possible in outer "
                    f"fold {outer_fold}: the smallest inner-training fold allows at "
                    f"most {max_components} components."
                )

            fold_start = time.perf_counter()
            clinical_text = " + clinical" if C_train is not None else ""
            print(
                f"  PCA nested CV outer fold {outer_fold + 1}/{len(outer_splits)}: "
                f"precomputing {inner_folds} leakage-safe PCA{clinical_text} "
                f"transforms, then evaluating {candidate_count} {kernel}-SVR "
                f"candidates...",
                flush=True,
            )

            best_params, best_inner_mae = _run_pca_inner_search(
                X_train=X_train,
                y_train=y_train,
                inner_cv=inner_cv,
                pca_components=int(pca_components),
                kernel=kernel,
                c_values=c_values,
                epsilon_values=epsilon_values,
                gamma_values=gamma_values,
                degree_values=degree_values,
                coef0_values=coef0_values,
                n_jobs=n_jobs,
                X_clinical_train=C_train,
            )

            embedding_scaler = StandardScaler()
            X_train_scaled = embedding_scaler.fit_transform(X_train)
            X_test_scaled = embedding_scaler.transform(X_test)

            outer_pca = PCA(n_components=int(pca_components), svd_solver="auto")
            X_train_model = outer_pca.fit_transform(X_train_scaled)
            X_test_model = outer_pca.transform(X_test_scaled)

            if C_train is not None and C_test is not None:
                clinical_scaler = StandardScaler()
                C_train_scaled = clinical_scaler.fit_transform(C_train)
                C_test_scaled = clinical_scaler.transform(C_test)
                X_train_model = np.column_stack(
                    [X_train_model, C_train_scaled]
                )
                X_test_model = np.column_stack(
                    [X_test_model, C_test_scaled]
                )

            best_model = SVR(kernel=kernel, **best_params)
            best_model.fit(X_train_model, y_train)
            predicted = best_model.predict(X_test_model)

            fold_elapsed = time.perf_counter() - fold_start
            param_text = ", ".join(f"{k}={v}" for k, v in best_params.items())
            print(
                f"  Completed PCA outer fold {outer_fold + 1}/{len(outer_splits)} "
                f"in {fold_elapsed / 60.0:.2f} min "
                f"(best {param_text}, inner MAE={best_inner_mae:.4f})",
                flush=True,
            )
            best_score_mae = float(best_inner_mae)

        else:
            X_train = X_non_pca[outer_train_idx]
            X_test = X_non_pca[outer_test_idx]

            search = GridSearchCV(
                estimator=_make_pipeline(pca_components=None, kernel=kernel),
                param_grid=_gridsearch_param_grid(
                    kernel=kernel,
                    c_values=c_values,
                    epsilon_values=epsilon_values,
                    gamma_values=gamma_values,
                    degree_values=degree_values,
                    coef0_values=coef0_values,
                ),
                scoring="neg_mean_absolute_error",
                cv=inner_cv,
                refit=True,
                n_jobs=n_jobs,
                return_train_score=False,
            )
            search.fit(X_train, y_train)
            predicted = search.predict(X_test)
            best_params = {
                key.removeprefix("svr__"): value
                for key, value in search.best_params_.items()
            }
            best_score_mae = float(-search.best_score_)

        oof_prediction[outer_test_idx] = predicted
        oof_fold[outer_test_idx] = outer_fold

        tuning_rows.append(
            {
                "outer_fold": outer_fold,
                "n_outer_train": len(outer_train_idx),
                "n_outer_test": len(outer_test_idx),
                "pca_components": pca_components,
                "n_clinical_features": (
                    0 if X_clinical is None else int(X_clinical.shape[1])
                ),
                "kernel": kernel,
                "best_C": float(best_params["C"]),
                "best_epsilon": float(best_params["epsilon"]),
                "best_gamma": best_params.get("gamma", None),
                "best_degree": best_params.get("degree", None),
                "best_coef0": best_params.get("coef0", None),
                "best_inner_MAE": best_score_mae,
            }
        )

    if np.isnan(oof_prediction).any() or (oof_fold < 0).any():
        raise RuntimeError(
            "Nested CV failed to generate exactly one OOF prediction per patient."
        )

    prediction_df = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "outer_fold": oof_fold,
            "nihss_observed": y,
            "nihss_predicted": oof_prediction,
            "absolute_error": np.abs(y - oof_prediction),
        }
    )
    return prediction_df, pd.DataFrame(tuning_rows)


def run_oof_dummy_regressor(
    y: np.ndarray,
    patient_ids: np.ndarray,
    outer_splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    strategy: str,
) -> pd.DataFrame:
    """Generate leakage-free OOF predictions from a DummyRegressor."""

    oof_prediction = np.full(len(y), np.nan, dtype=float)
    oof_fold = np.full(len(y), -1, dtype=int)

    # DummyRegressor requires X, even though it ignores the features.
    X_dummy = np.zeros((len(y), 1), dtype=float)

    for outer_fold, (train_idx, test_idx) in enumerate(outer_splits):

        model = DummyRegressor(strategy=strategy)

        model.fit(
            X_dummy[train_idx],
            y[train_idx],
        )

        oof_prediction[test_idx] = model.predict(
            X_dummy[test_idx]
        )

        oof_fold[test_idx] = outer_fold

    if np.isnan(oof_prediction).any():
        raise RuntimeError(
            f"Missing OOF predictions for dummy strategy {strategy!r}."
        )

    return pd.DataFrame(
        {
            "patient_id": patient_ids,
            "outer_fold": oof_fold,
            "nihss_observed": y,
            "nihss_predicted": oof_prediction,
            "absolute_error": np.abs(
                y - oof_prediction
            ),
        }
    )

def _strict_holdout_partition(
    clinical: pd.DataFrame,
    clinical_split_col: str,
    tables: Iterable[EmbeddingTable],
    cohort_ids: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return development and held-out test IDs with strict split validation.

    In strict ``data_scope='test'`` mode, the clinical split column is authoritative,
    but every embedding file must carry the same split label for every common patient.
    The function deliberately ignores ``allow_vae_split_mismatch``: a mismatch aborts
    the analysis because otherwise a non-test patient could enter held-out evaluation.
    """
    if clinical_split_col not in clinical.columns:
        raise ValueError(
            f"Strict test mode requires clinical split column {clinical_split_col!r}."
        )

    cohort_ids = np.asarray([str(x) for x in cohort_ids], dtype=str)
    split_series = clinical.loc[cohort_ids, clinical_split_col]

    if split_series.isna().any():
        missing_ids = split_series.index[split_series.isna()].astype(str).tolist()[:10]
        raise ValueError(
            "Strict test mode requires a split label for every common patient. "
            f"Missing examples: {missing_ids}"
        )

    split_series = split_series.astype(str).str.strip().str.lower()
    invalid = set(split_series.unique()) - SPLIT_NAMES
    if invalid:
        raise ValueError(
            f"Strict test mode found invalid split labels: {sorted(invalid)}."
        )

    for table in tables:
        if not table.splits:
            raise ValueError(
                "Strict test mode requires split labels in every embedding file. "
                f"Missing split information for {table.representation} {table.modality}."
            )
        missing = [sid for sid in cohort_ids if sid not in table.splits]
        if missing:
            raise ValueError(
                "Strict test mode requires split labels for every common patient in "
                f"{table.representation} {table.modality}. Missing examples: {missing[:10]}"
            )
        mismatches = [
            (sid, table.splits[sid], split_series.loc[sid])
            for sid in cohort_ids
            if str(table.splits[sid]).strip().lower() != str(split_series.loc[sid])
        ]
        if mismatches:
            raise ValueError(
                "Strict test mode aborted because embedding and clinical split labels "
                f"do not match for {table.representation} {table.modality}. "
                f"Examples: {mismatches[:10]}"
            )

    test_ids = split_series.index[split_series == "test"].astype(str).to_numpy()
    development_ids = split_series.index[
        split_series.isin(["train", "valid"])
    ].astype(str).to_numpy()

    if len(test_ids) == 0:
        raise ValueError("Strict test mode found zero patients with split='test'.")
    if len(development_ids) == 0:
        raise ValueError(
            "Strict test mode found zero development patients in train/valid splits."
        )
    if set(test_ids) & set(development_ids):
        raise RuntimeError("Development and test patient IDs unexpectedly overlap.")
    if set(test_ids) | set(development_ids) != set(cohort_ids):
        raise RuntimeError(
            "Strict test partition does not cover the complete common cohort."
        )

    return np.asarray(sorted(development_ids), dtype=str), np.asarray(sorted(test_ids), dtype=str)


def run_holdout_dummy_regressor(
    y_development: np.ndarray,
    y_test: np.ndarray,
    test_patient_ids: np.ndarray,
    strategy: str,
) -> pd.DataFrame:
    """Fit a dummy model on development targets and evaluate only held-out test data."""
    X_development = np.zeros((len(y_development), 1), dtype=float)
    X_test = np.zeros((len(y_test), 1), dtype=float)
    model = DummyRegressor(strategy=strategy)
    model.fit(X_development, y_development)
    prediction = model.predict(X_test)
    return pd.DataFrame(
        {
            "patient_id": np.asarray(test_patient_ids, dtype=str),
            "outer_fold": -1,
            "evaluation_set": "test",
            "nihss_observed": y_test,
            "nihss_predicted": prediction,
            "absolute_error": np.abs(y_test - prediction),
        }
    )


def run_holdout_svr(
    X_development: np.ndarray,
    y_development: np.ndarray,
    development_patient_ids: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_patient_ids: np.ndarray,
    inner_folds: int,
    random_state: int,
    n_jobs: int,
    pca_components: Optional[int] = None,
    kernel: str = "linear",
    c_values: Sequence[float] = DEFAULT_C_VALUES,
    epsilon_values: Sequence[float] = DEFAULT_EPSILON_VALUES,
    gamma_values: Sequence[object] = DEFAULT_GAMMA_VALUES,
    degree_values: Sequence[int] = DEFAULT_DEGREE_VALUES,
    coef0_values: Sequence[float] = DEFAULT_COEF0_VALUES,
    X_clinical_development: Optional[np.ndarray] = None,
    X_clinical_test: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Tune on VAE train+valid patients, refit there, and predict only held-out test patients."""
    kernel = _normalise_kernel(kernel)
    gamma_values = _normalise_gamma_values(gamma_values)
    _validate_svr_grid_values(c_values, epsilon_values, degree_values)

    if len(X_development) != len(y_development) or len(y_development) != len(development_patient_ids):
        raise ValueError("Development X, y, and patient IDs must have identical row counts.")
    if len(X_test) != len(y_test) or len(y_test) != len(test_patient_ids):
        raise ValueError("Test X, y, and patient IDs must have identical row counts.")
    if set(map(str, development_patient_ids)) & set(map(str, test_patient_ids)):
        raise ValueError("Development and held-out test patient IDs must be disjoint.")
    if inner_folds > len(y_development):
        raise ValueError(
            f"inner_folds={inner_folds} exceeds development sample size {len(y_development)}."
        )

    if (X_clinical_development is None) != (X_clinical_test is None):
        raise ValueError(
            "Clinical blocks must be supplied for both development and test data, or neither."
        )
    if X_clinical_development is not None:
        if len(X_clinical_development) != len(y_development):
            raise ValueError("Development clinical block has wrong row count.")
        if len(X_clinical_test) != len(y_test):
            raise ValueError("Test clinical block has wrong row count.")
        if not np.isfinite(X_clinical_development).all() or not np.isfinite(X_clinical_test).all():
            raise ValueError("Clinical predictor blocks contain non-finite values.")

    inner_cv = list(
        KFold(
            n_splits=inner_folds,
            shuffle=True,
            random_state=random_state,
        ).split(np.arange(len(y_development)))
    )

    if pca_components is not None:
        if pca_components < 1 or pca_components > X_development.shape[1]:
            raise ValueError(
                f"Invalid PCA component count {pca_components} for development matrix "
                f"with {X_development.shape[1]} features."
            )
        smallest_inner_train = min(len(train_idx) for train_idx, _ in inner_cv)
        max_components = min(X_development.shape[1], smallest_inner_train)
        if pca_components > max_components:
            raise ValueError(
                f"PCA with {pca_components} components is not possible: smallest inner "
                f"training fold allows at most {max_components}."
            )

        best_params, best_inner_mae = _run_pca_inner_search(
            X_train=X_development,
            y_train=y_development,
            inner_cv=inner_cv,
            pca_components=int(pca_components),
            kernel=kernel,
            c_values=c_values,
            epsilon_values=epsilon_values,
            gamma_values=gamma_values,
            degree_values=degree_values,
            coef0_values=coef0_values,
            n_jobs=n_jobs,
            X_clinical_train=X_clinical_development,
        )

        embedding_scaler = StandardScaler()
        X_dev_scaled = embedding_scaler.fit_transform(X_development)
        X_test_scaled = embedding_scaler.transform(X_test)
        pca = PCA(n_components=int(pca_components), svd_solver="auto")
        X_dev_model = pca.fit_transform(X_dev_scaled)
        X_test_model = pca.transform(X_test_scaled)

        if X_clinical_development is not None and X_clinical_test is not None:
            clinical_scaler = StandardScaler()
            C_dev = clinical_scaler.fit_transform(X_clinical_development)
            C_test = clinical_scaler.transform(X_clinical_test)
            X_dev_model = np.column_stack([X_dev_model, C_dev])
            X_test_model = np.column_stack([X_test_model, C_test])

        model = SVR(kernel=kernel, **best_params)
        model.fit(X_dev_model, y_development)
        prediction = model.predict(X_test_model)
        best_score_mae = float(best_inner_mae)
    else:
        X_dev_model = (
            np.column_stack([X_development, X_clinical_development])
            if X_clinical_development is not None
            else X_development
        )
        X_test_model = (
            np.column_stack([X_test, X_clinical_test])
            if X_clinical_test is not None
            else X_test
        )
        search = GridSearchCV(
            estimator=_make_pipeline(pca_components=None, kernel=kernel),
            param_grid=_gridsearch_param_grid(
                kernel=kernel,
                c_values=c_values,
                epsilon_values=epsilon_values,
                gamma_values=gamma_values,
                degree_values=degree_values,
                coef0_values=coef0_values,
            ),
            scoring="neg_mean_absolute_error",
            cv=inner_cv,
            refit=True,
            n_jobs=n_jobs,
            return_train_score=False,
        )
        search.fit(X_dev_model, y_development)
        prediction = search.predict(X_test_model)
        best_params = {
            key.removeprefix("svr__"): value for key, value in search.best_params_.items()
        }
        best_score_mae = float(-search.best_score_)

    prediction_df = pd.DataFrame(
        {
            "patient_id": np.asarray(test_patient_ids, dtype=str),
            "outer_fold": -1,
            "evaluation_set": "test",
            "nihss_observed": y_test,
            "nihss_predicted": prediction,
            "absolute_error": np.abs(y_test - prediction),
        }
    )

    tuning_df = pd.DataFrame(
        [
            {
                "outer_fold": -1,
                "evaluation_mode": "heldout_test",
                "n_development": len(y_development),
                "n_outer_train": len(y_development),
                "n_outer_test": len(y_test),
                "pca_components": pca_components,
                "n_clinical_features": (
                    0 if X_clinical_development is None else int(X_clinical_development.shape[1])
                ),
                "kernel": kernel,
                "best_C": float(best_params["C"]),
                "best_epsilon": float(best_params["epsilon"]),
                "best_gamma": best_params.get("gamma", None),
                "best_degree": best_params.get("degree", None),
                "best_coef0": best_params.get("coef0", None),
                "best_inner_MAE": best_score_mae,
            }
        ]
    )
    return prediction_df, tuning_df


def _assert_exact_test_predictions(
    pred_df: pd.DataFrame,
    expected_test_ids: Sequence[str],
    label: str,
) -> None:
    """Hard safety check that a held-out result contains every and only test patient once."""
    expected = set(map(str, expected_test_ids))
    observed_ids = pred_df["patient_id"].astype(str)
    observed = set(observed_ids)
    if observed != expected:
        extra = sorted(observed - expected)[:10]
        missing = sorted(expected - observed)[:10]
        raise RuntimeError(
            f"STRICT TEST-SCOPE FAILURE for {label}: prediction IDs differ from the "
            f"held-out test set. Extra={extra}, missing={missing}."
        )
    if observed_ids.duplicated().any() or len(pred_df) != len(expected):
        raise RuntimeError(
            f"STRICT TEST-SCOPE FAILURE for {label}: expected exactly one prediction "
            "per held-out test patient."
        )



def _metric_row(
    modality: str,
    representation: str,
    analysis_variant: str,
    pred_df: pd.DataFrame,
    n_input_features: int,
    n_model_features: int,
    n_clinical_features: int = 0,
) -> dict:
    observed = pred_df["nihss_observed"]
    predicted = pred_df["nihss_predicted"]

    return {
        "analysis_variant": analysis_variant,
        "modality": modality,
        "representation": representation,
        "n_patients": len(pred_df),
        "n_input_features": n_input_features,
        "n_model_features": n_model_features,
        "n_clinical_features": n_clinical_features,
        "MAE": float(mean_absolute_error(observed, predicted)),
        "MedAE": float(median_absolute_error(observed, predicted)),
        "R2": float(r2_score(observed, predicted)),
    }


def _representation_key(representation: str) -> str:
    return representation.lower().replace("-", "_")


def _add_modality_specific_tests(
    error_wide: pd.DataFrame,
    modalities: Sequence[str],
    rep_a: str,
    rep_b: str,
    alternative: str,
) -> pd.DataFrame:
    rows: List[dict] = []
    raw_p_values: List[float] = []

    key_a = _representation_key(rep_a)
    key_b = _representation_key(rep_b)

    for modality in modalities:
        err_a = error_wide[f"abs_error_{modality}_{key_a}"]
        err_b = error_wide[f"abs_error_{modality}_{key_b}"]
        statistic, p_value = _safe_wilcoxon(err_a, err_b, alternative)
        raw_p_values.append(p_value)
        rows.append(
            {
                "modality": modality,
                "comparison": f"{rep_a} vs {rep_b}",
                "n_patients": len(error_wide),
                f"median_abs_error_{key_a}": float(np.median(err_a)),
                f"median_abs_error_{key_b}": float(np.median(err_b)),
                f"median_difference_{key_a}_minus_{key_b}": float(
                    np.median(err_a - err_b)
                ),
                "wilcoxon_statistic": statistic,
                "p_value_raw": p_value,
            }
        )

    p_bh = _benjamini_hochberg(raw_p_values)
    p_holm = _holm_adjust(raw_p_values)

    for row, bh, holm in zip(rows, p_bh, p_holm):
        row["p_value_BH"] = float(bh)
        row["p_value_Holm"] = float(holm)
        row["significant_BH_0.05"] = bool(bh < 0.05)
        row["significant_Holm_0.05"] = bool(holm < 0.05)
        row["alternative"] = alternative

    return pd.DataFrame(rows)


def _primary_comparison(
    error_wide: pd.DataFrame,
    modalities: Sequence[str],
    rep_a: str,
    rep_b: str,
    alternative: str,
    comparison_label: str,
) -> pd.DataFrame:
    key_a = _representation_key(rep_a)
    key_b = _representation_key(rep_b)
    cols_a = [f"abs_error_{m}_{key_a}" for m in modalities]
    cols_b = [f"abs_error_{m}_{key_b}" for m in modalities]

    mean_a = error_wide[cols_a].mean(axis=1)
    mean_b = error_wide[cols_b].mean(axis=1)
    diff = mean_a - mean_b

    error_wide[f"mean_abs_error_{key_a}"] = mean_a
    error_wide[f"mean_abs_error_{key_b}"] = mean_b
    error_wide[f"error_difference_{key_a}_minus_{key_b}"] = diff

    statistic, p_value = _safe_wilcoxon(mean_a, mean_b, alternative)

    return pd.DataFrame(
        [
            {
                "comparison": comparison_label,
                "representation_a": rep_a,
                "representation_b": rep_b,
                "n_patients": len(error_wide),
                "n_modalities": len(modalities),
                f"median_patient_mean_abs_error_{key_a}": float(np.median(mean_a)),
                f"median_patient_mean_abs_error_{key_b}": float(np.median(mean_b)),
                f"median_difference_{key_a}_minus_{key_b}": float(np.median(diff)),
                "wilcoxon_statistic": statistic,
                "p_value_raw": p_value,
                "alternative": alternative,
            }
        ]
    )


def _prepare_clinical_predictors(
    clinical: pd.DataFrame,
    predictor_cols: Sequence[str],
) -> pd.DataFrame:
    """Validate clinical predictors and exclude rows with missing/non-finite values."""
    predictor_cols = tuple(str(col) for col in predictor_cols)
    if not predictor_cols:
        raise ValueError("At least one clinical predictor column is required.")

    missing = set(predictor_cols) - set(clinical.columns)
    if missing:
        raise ValueError(
            f"Clinical file is missing predictor columns: {sorted(missing)}"
        )

    clinical = clinical.copy()
    for col in predictor_cols:
        clinical[col] = pd.to_numeric(clinical[col], errors="coerce")

    predictor_matrix = clinical[list(predictor_cols)].to_numpy(dtype=float)
    complete = np.isfinite(predictor_matrix).all(axis=1)
    n_excluded = int((~complete).sum())
    if n_excluded:
        print(
            f"Clinical predictors: excluding {n_excluded} patient(s) with missing "
            f"or non-finite values in {list(predictor_cols)} so variants A/B/C use "
            "the same cohort.",
            flush=True,
        )
        clinical = clinical.loc[complete].copy()

    return clinical


def _mean_prediction_across_modalities(
    predictions: pd.DataFrame,
    representation: str,
    modalities: Sequence[str],
) -> pd.DataFrame:
    subset = predictions.loc[
        (predictions["representation"] == representation)
        & (predictions["modality"].isin(modalities))
    ]
    pred_wide = subset.pivot(
        index="patient_id", columns="modality", values="nihss_predicted"
    )
    obs = subset.drop_duplicates("patient_id").set_index("patient_id")[
        "nihss_observed"
    ]
    return pd.DataFrame(
        {
            "nihss_observed": obs.reindex(pred_wide.index),
            "nihss_predicted_mean_across_modalities": pred_wide[
                list(modalities)
            ].mean(axis=1),
        }
    )


def _save_variant_comparison_figure(
    error_wide: pd.DataFrame,
    predictions: pd.DataFrame,
    modalities: Sequence[str],
    rep_a: str,
    rep_b: str,
    label_a: str,
    label_b: str,
    title: str,
    output_stem: Path,
) -> None:
    """Save a two-panel descriptive figure for a 2D-vs-3D variant."""
    key_a = _representation_key(rep_a)
    key_b = _representation_key(rep_b)
    err_a = error_wide[
        [f"abs_error_{m}_{key_a}" for m in modalities]
    ].mean(axis=1)
    err_b = error_wide[
        [f"abs_error_{m}_{key_b}" for m in modalities]
    ].mean(axis=1)

    pred_a = _mean_prediction_across_modalities(predictions, rep_a, modalities)
    pred_b = _mean_prediction_across_modalities(predictions, rep_b, modalities)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].boxplot([err_a.to_numpy(), err_b.to_numpy()], tick_labels=[label_a, label_b])
    axes[0].set_ylabel("Patient mean absolute error across modalities [NIHSS points]")
    axes[0].set_title("Patient-level error distributions")

    axes[1].scatter(
        pred_a["nihss_observed"],
        pred_a["nihss_predicted_mean_across_modalities"],
        alpha=0.65,
        label=label_a,
    )
    axes[1].scatter(
        pred_b["nihss_observed"],
        pred_b["nihss_predicted_mean_across_modalities"],
        alpha=0.65,
        label=label_b,
    )
    all_values = np.concatenate(
        [
            pred_a["nihss_observed"].to_numpy(),
            pred_a["nihss_predicted_mean_across_modalities"].to_numpy(),
            pred_b["nihss_predicted_mean_across_modalities"].to_numpy(),
        ]
    )
    lo = float(np.nanmin(all_values))
    hi = float(np.nanmax(all_values))
    axes[1].plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
    axes[1].set_xlabel("Observed discharge NIHSS")
    axes[1].set_ylabel("Mean predicted NIHSS across modality-specific models")
    axes[1].set_title("OOF predictions (descriptive mean across modalities)")
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_stem.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_clinical_only_figure(
    pred_df: pd.DataFrame,
    title: str,
    output_stem: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    observed = pred_df["nihss_observed"].to_numpy(dtype=float)
    predicted = pred_df["nihss_predicted"].to_numpy(dtype=float)
    errors = pred_df["absolute_error"].to_numpy(dtype=float)

    axes[0].scatter(observed, predicted, alpha=0.65)
    lo = float(min(observed.min(), predicted.min()))
    hi = float(max(observed.max(), predicted.max()))
    axes[0].plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
    axes[0].set_xlabel("Observed discharge NIHSS")
    axes[0].set_ylabel("Predicted discharge NIHSS")
    axes[0].set_title("Clinical-only OOF predictions")

    axes[1].hist(errors, bins=20)
    axes[1].axvline(float(np.median(errors)), linestyle="--", linewidth=1)
    axes[1].set_xlabel("Absolute error [NIHSS points]")
    axes[1].set_ylabel("Patients")
    axes[1].set_title("Clinical-only absolute errors")

    fig.suptitle(title)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_stem.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _paired_model_vs_reference(
    model_error: pd.Series,
    reference_error: pd.Series,
    model_label: str,
    reference_label: str,
    alternative: str,
) -> dict:
    model_error, reference_error = model_error.align(reference_error, join="inner")
    diff = model_error - reference_error
    statistic, p_value = _safe_wilcoxon(model_error, reference_error, alternative)
    return {
        "model": model_label,
        "reference": reference_label,
        "n_patients": len(diff),
        "median_model_abs_error": float(np.median(model_error)),
        "median_reference_abs_error": float(np.median(reference_error)),
        "median_difference_model_minus_reference": float(np.median(diff)),
        "mean_difference_model_minus_reference": float(np.mean(diff)),
        "fraction_model_better": float(np.mean(diff < 0)),
        "fraction_reference_better": float(np.mean(diff > 0)),
        "wilcoxon_statistic": statistic,
        "p_value_raw": p_value,
        "alternative": alternative,
    }


def run_analysis(args: argparse.Namespace) -> Path:
    script_dir = Path(__file__).resolve().parent

    data_scope = str(getattr(args, "data_scope", "all")).strip().lower()
    if data_scope not in {"all", "test"}:
        raise ValueError("data_scope must be either 'all' or 'test'.")

    kernel = _normalise_kernel(getattr(args, "kernel", "linear"))
    c_values = tuple(getattr(args, "c_values", DEFAULT_C_VALUES))
    epsilon_values = tuple(getattr(args, "epsilon_values", DEFAULT_EPSILON_VALUES))
    gamma_values = _normalise_gamma_values(
        getattr(args, "gamma_values", DEFAULT_GAMMA_VALUES)
    )
    degree_values = tuple(getattr(args, "degree_values", DEFAULT_DEGREE_VALUES))
    coef0_values = tuple(getattr(args, "coef0_values", DEFAULT_COEF0_VALUES))
    clinical_predictor_cols = tuple(
        getattr(args, "clinical_predictor_cols", DEFAULT_CLINICAL_PREDICTOR_COLS)
    )
    _validate_svr_grid_values(c_values, epsilon_values, degree_values)

    clinical_path = (
        args.clinical.expanduser().resolve()
        if args.clinical is not None
        else _find_default_clinical(script_dir)
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.output_dir is None:
        base = script_dir / "nihss_prediction"
        base.mkdir(parents=True, exist_ok=True)
        output_dir = base / f"2d_vs_3d_{data_scope}_svr_{kernel}_ABC_{timestamp}"
    else:
        output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    paths_2d_raw = _parse_mapping(args.embedding_2d, "--embedding-2d")
    paths_3d_raw = _parse_mapping(args.embedding_3d, "--embedding-3d")

    if set(paths_2d_raw) != set(paths_3d_raw):
        raise ValueError(
            "Exactly the same modalities must be supplied for 2D and 3D. "
            f"2D={sorted(paths_2d_raw)}, 3D={sorted(paths_3d_raw)}"
        )

    modalities = sorted(paths_2d_raw)
    if not modalities:
        raise ValueError("No modalities were supplied.")
    if args.outer_folds < 2 or args.inner_folds < 2:
        raise ValueError("outer-folds and inner-folds must both be >= 2.")
    if args.pca_components is not None and args.pca_components < 1:
        raise ValueError("--pca-components must be >= 1 when specified.")

    clinical = load_clinical(
        clinical_path,
        args.clinical_id_col,
        args.target_col,
        args.clinical_split_col,
    )
    clinical = _prepare_clinical_predictors(clinical, clinical_predictor_cols)

    tables: Dict[Tuple[str, str], EmbeddingTable] = {}
    resolved_paths: Dict[str, Dict[str, str]] = {}
    pca_components_by_modality: Dict[str, int] = {}

    for modality in modalities:
        path_2d = _resolve_embedding_path(paths_2d_raw[modality], modality, "2D")
        path_3d = _resolve_embedding_path(paths_3d_raw[modality], modality, "3D")

        table_2d = load_embedding_table(path_2d, modality, "2D")
        table_3d = load_embedding_table(path_3d, modality, "3D")

        if not args.skip_pca_sensitivity:
            inferred_dim = int(table_3d.X.shape[1])
            if args.pca_components is not None:
                if int(args.pca_components) != inferred_dim:
                    raise ValueError(
                        f"{modality}: --pca-components={args.pca_components}, but "
                        f"the corresponding 3D embedding has {inferred_dim} dimensions. "
                        "For modality-wise dimension matching, omit --pca-components "
                        "and let the script infer the target from each 3D file."
                    )
                target_dim = int(args.pca_components)
            else:
                target_dim = inferred_dim
            pca_components_by_modality[modality] = target_dim

        tables[(modality, "2D")] = table_2d
        tables[(modality, "3D")] = table_3d
        resolved_paths[modality] = {"2D": str(path_2d), "3D": str(path_3d)}

        pca_text = (
            f"; PCA-matched 2D target p={pca_components_by_modality[modality]}"
            if not args.skip_pca_sensitivity
            else ""
        )
        print(
            f"{modality}: 2D n={len(table_2d.ids)}, p={table_2d.X.shape[1]}; "
            f"3D n={len(table_3d.ids)}, p={table_3d.X.shape[1]}{pca_text}",
            flush=True,
        )

    _check_vae_split_labels(
        clinical,
        args.clinical_split_col,
        tables.values(),
        args.allow_vae_split_mismatch,
    )

    # One common cohort across A, B and C.
    cohort_ids = set(clinical.index.astype(str))
    for table in tables.values():
        cohort_ids &= set(table.ids)

    if not cohort_ids:
        raise ValueError(
            "No patients are common to the clinical table and all embedding files."
        )

    all_common_ids = np.asarray(sorted(cohort_ids), dtype=str)

    # ------------------------------------------------------------------
    # Evaluation scope
    # ------------------------------------------------------------------
    # all  -> original nested-CV evaluation over the complete common cohort
    # test -> STRICT held-out evaluation. VAE train+valid are used only for
    #         downstream SVR tuning/refitting; VAE test patients are the only
    #         patients that enter predictions, metrics, statistics and figures.
    if data_scope == "test":
        development_ids, evaluation_ids = _strict_holdout_partition(
            clinical=clinical,
            clinical_split_col=args.clinical_split_col,
            tables=tables.values(),
            cohort_ids=all_common_ids,
        )
        if len(development_ids) < args.inner_folds:
            raise ValueError(
                f"Only {len(development_ids)} development patients are available, "
                f"fewer than inner_folds={args.inner_folds}."
            )
        development_cohort = clinical.loc[development_ids].copy()
        evaluation_cohort = clinical.loc[evaluation_ids].copy()

        # Hard guarantee: every evaluation patient is clinically labelled test.
        evaluation_labels = (
            evaluation_cohort[args.clinical_split_col]
            .astype(str)
            .str.strip()
            .str.lower()
        )
        if not (evaluation_labels == "test").all():
            raise RuntimeError(
                "STRICT TEST-SCOPE FAILURE: non-test patient entered evaluation cohort."
            )

        y_development = development_cohort[args.target_col].to_numpy(dtype=float)
        y = evaluation_cohort[args.target_col].to_numpy(dtype=float)
        X_clinical_development = development_cohort[
            list(clinical_predictor_cols)
        ].to_numpy(dtype=float)
        X_clinical = evaluation_cohort[
            list(clinical_predictor_cols)
        ].to_numpy(dtype=float)
        patient_ids = evaluation_ids
        outer_splits = None

        print(
            f"\nSTRICT HELD-OUT TEST MODE: {len(development_ids)} development "
            f"patients (train+valid), {len(patient_ids)} held-out test patients.",
            flush=True,
        )
    else:
        development_ids = None
        development_cohort = None
        y_development = None
        X_clinical_development = None
        patient_ids = all_common_ids
        evaluation_cohort = clinical.loc[patient_ids].copy()
        if len(patient_ids) < args.outer_folds:
            raise ValueError(
                f"Only {len(patient_ids)} common patients are available, fewer than "
                f"outer_folds={args.outer_folds}."
            )
        y = evaluation_cohort[args.target_col].to_numpy(dtype=float)
        X_clinical = evaluation_cohort[
            list(clinical_predictor_cols)
        ].to_numpy(dtype=float)
        outer_cv = KFold(
            n_splits=args.outer_folds,
            shuffle=True,
            random_state=args.random_state,
        )
        outer_splits = list(outer_cv.split(np.arange(len(patient_ids))))
        print(
            f"\nALL-DATA NESTED-CV MODE: {len(patient_ids)} common patients across "
            f"{len(modalities)} modalities.",
            flush=True,
        )

    if not np.isfinite(X_clinical).all():
        raise RuntimeError("Clinical predictor matrix unexpectedly contains non-finite values.")
    if X_clinical_development is not None and not np.isfinite(X_clinical_development).all():
        raise RuntimeError(
            "Development clinical predictor matrix unexpectedly contains non-finite values."
        )

    print(
        f"Clinical predictors for variants B/C: {list(clinical_predictor_cols)}",
        flush=True,
    )

    candidate_count = len(
        _svr_candidate_dicts(
            kernel, c_values, epsilon_values, gamma_values, degree_values, coef0_values
        )
    )
    print(
        f"SVR kernel: {kernel}; hyperparameter candidates per inner search: "
        f"{candidate_count}.",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Dummy baselines
    # ------------------------------------------------------------------
    if data_scope == "test":
        dummy_mean_df = run_holdout_dummy_regressor(
            y_development=y_development,
            y_test=y,
            test_patient_ids=patient_ids,
            strategy="mean",
        )
        dummy_median_df = run_holdout_dummy_regressor(
            y_development=y_development,
            y_test=y,
            test_patient_ids=patient_ids,
            strategy="median",
        )
        _assert_exact_test_predictions(dummy_mean_df, patient_ids, "DummyRegressor mean")
        _assert_exact_test_predictions(dummy_median_df, patient_ids, "DummyRegressor median")

        # Auditable split assignment table for every common patient.
        split_df = clinical.loc[all_common_ids, [
            args.target_col,
            args.clinical_split_col,
            *clinical_predictor_cols,
        ]].copy()
        split_df.insert(0, "patient_id", split_df.index.astype(str))
        split_df["analysis_role"] = np.where(
            split_df[args.clinical_split_col].astype(str).str.lower() == "test",
            "heldout_test_evaluation",
            "development_train_valid",
        )
        split_df["data_scope"] = data_scope
        split_df.to_csv(output_dir / "holdout_split_assignments.csv", index=False)
        # Compatibility/audit alias. No CV fold is assigned in held-out mode.
        fold_df = split_df.copy()
        fold_df.insert(1, "outer_fold", -1)
        fold_df.to_csv(output_dir / "outer_fold_assignments.csv", index=False)
    else:
        dummy_mean_df = run_oof_dummy_regressor(
            y=y,
            patient_ids=patient_ids,
            outer_splits=outer_splits,
            strategy="mean",
        )
        dummy_median_df = run_oof_dummy_regressor(
            y=y,
            patient_ids=patient_ids,
            outer_splits=outer_splits,
            strategy="median",
        )

        fold_assignment = np.full(len(patient_ids), -1, dtype=int)
        for fold, (_, test_idx) in enumerate(outer_splits):
            fold_assignment[test_idx] = fold
        fold_df = pd.DataFrame(
            {
                "patient_id": patient_ids,
                "outer_fold": fold_assignment,
                args.target_col: y,
            }
        )
        for col in clinical_predictor_cols:
            fold_df[col] = evaluation_cohort[col].to_numpy(dtype=float)
        if args.clinical_split_col in evaluation_cohort.columns:
            fold_df["vae_custom_split"] = (
                evaluation_cohort[args.clinical_split_col].astype(str).to_numpy()
            )
        fold_df["data_scope"] = data_scope
        fold_df.to_csv(output_dir / "outer_fold_assignments.csv", index=False)

    dummy_mean_df["data_scope"] = data_scope
    dummy_median_df["data_scope"] = data_scope
    dummy_mean_df.to_csv(output_dir / "oof_dummy_mean_predictions.csv", index=False)
    dummy_median_df.to_csv(output_dir / "oof_dummy_median_predictions.csv", index=False)
    if data_scope == "test":
        dummy_mean_df.to_csv(output_dir / "test_dummy_mean_predictions.csv", index=False)
        dummy_median_df.to_csv(output_dir / "test_dummy_median_predictions.csv", index=False)

    dummy_metric_rows = []
    for strategy, pred_df in [
        ("mean", dummy_mean_df),
        ("median", dummy_median_df),
    ]:
        observed = pred_df["nihss_observed"].to_numpy()
        predicted = pred_df["nihss_predicted"].to_numpy()
        dummy_metric_rows.append(
            {
                "data_scope": data_scope,
                "model": f"DummyRegressor_{strategy}",
                "n_patients": len(pred_df),
                "MAE": mean_absolute_error(observed, predicted),
                "MedAE": median_absolute_error(observed, predicted),
                "R2": r2_score(observed, predicted),
            }
        )
    dummy_metrics = pd.DataFrame(dummy_metric_rows)
    dummy_metrics.to_csv(output_dir / "dummy_regressor_metrics.csv", index=False)
    print("\nDummy-regressor baselines:", flush=True)
    print(dummy_metrics.to_string(index=False), flush=True)

    all_predictions: List[pd.DataFrame] = []
    all_tuning: List[pd.DataFrame] = []
    metric_rows: List[dict] = []

    def _fit_model(
        X_evaluation: np.ndarray,
        X_development: Optional[np.ndarray],
        pca_components: Optional[int],
        clinical_evaluation: Optional[np.ndarray],
        clinical_development: Optional[np.ndarray],
        label: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if data_scope == "test":
            if X_development is None:
                raise RuntimeError("Strict test mode requires development features.")
            pred_df, tuning_df = run_holdout_svr(
                X_development=X_development,
                y_development=y_development,
                development_patient_ids=development_ids,
                X_test=X_evaluation,
                y_test=y,
                test_patient_ids=patient_ids,
                inner_folds=args.inner_folds,
                random_state=args.random_state,
                n_jobs=args.n_jobs,
                pca_components=pca_components,
                kernel=kernel,
                c_values=c_values,
                epsilon_values=epsilon_values,
                gamma_values=gamma_values,
                degree_values=degree_values,
                coef0_values=coef0_values,
                X_clinical_development=clinical_development,
                X_clinical_test=clinical_evaluation,
            )
            _assert_exact_test_predictions(pred_df, patient_ids, label)
            return pred_df, tuning_df

        return run_nested_svr(
            X=X_evaluation,
            y=y,
            patient_ids=patient_ids,
            outer_splits=outer_splits,
            inner_folds=args.inner_folds,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
            pca_components=pca_components,
            kernel=kernel,
            c_values=c_values,
            epsilon_values=epsilon_values,
            gamma_values=gamma_values,
            degree_values=degree_values,
            coef0_values=coef0_values,
            X_clinical=clinical_evaluation,
        )

    # ------------------------------------------------------------------
    # Variant C: clinical only (fit exactly once)
    # ------------------------------------------------------------------
    print(
        f"\nRunning variant C clinical-only: evaluation n={X_clinical.shape[0]}, "
        f"p={X_clinical.shape[1]}",
        flush=True,
    )
    pred_c, tune_c = _fit_model(
        X_evaluation=X_clinical,
        X_development=X_clinical_development,
        pca_components=None,
        clinical_evaluation=None,
        clinical_development=None,
        label="C clinical-only",
    )
    pred_c.insert(1, "data_scope", data_scope)
    pred_c.insert(2, "analysis_variant", "C_clinical_only")
    pred_c.insert(3, "modality", "CLINICAL")
    pred_c.insert(4, "representation", "CLINICAL_ONLY")
    all_predictions.append(pred_c)

    tune_c.insert(0, "data_scope", data_scope)
    tune_c.insert(1, "analysis_variant", "C_clinical_only")
    tune_c.insert(2, "modality", "CLINICAL")
    tune_c.insert(3, "representation", "CLINICAL_ONLY")
    all_tuning.append(tune_c)

    metric_c = _metric_row(
        modality="CLINICAL",
        representation="CLINICAL_ONLY",
        analysis_variant="C_clinical_only",
        pred_df=pred_c,
        n_input_features=X_clinical.shape[1],
        n_model_features=X_clinical.shape[1],
        n_clinical_features=X_clinical.shape[1],
    )
    metric_c["data_scope"] = data_scope
    metric_rows.append(metric_c)
    print(
        f"Completed C clinical-only: MAE={metric_c['MAE']:.4f}, "
        f"MedAE={metric_c['MedAE']:.4f}, R2={metric_c['R2']:.4f}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Variants A and B for each modality
    # ------------------------------------------------------------------
    for modality in modalities:
        table_2d = tables[(modality, "2D")]
        table_3d = tables[(modality, "3D")]
        X_2d_eval = _subset_matrix(table_2d, patient_ids)
        X_3d_eval = _subset_matrix(table_3d, patient_ids)
        X_2d_dev = (
            _subset_matrix(table_2d, development_ids) if data_scope == "test" else None
        )
        X_3d_dev = (
            _subset_matrix(table_3d, development_ids) if data_scope == "test" else None
        )

        analyses = [
            (
                "A_embedding_only", "2D", X_2d_eval, X_2d_dev, None,
                None, None,
            ),
            (
                "A_embedding_only", "3D", X_3d_eval, X_3d_dev, None,
                None, None,
            ),
            (
                "B_embedding_plus_clinical", "2D_CLINICAL", X_2d_eval, X_2d_dev,
                None, X_clinical, X_clinical_development,
            ),
            (
                "B_embedding_plus_clinical", "3D_CLINICAL", X_3d_eval, X_3d_dev,
                None, X_clinical, X_clinical_development,
            ),
        ]

        if not args.skip_pca_sensitivity:
            analyses.insert(
                1,
                (
                    "A_embedding_only", "2D_PCA_MATCHED", X_2d_eval, X_2d_dev,
                    pca_components_by_modality[modality], None, None,
                ),
            )
            analyses.insert(
                4,
                (
                    "B_embedding_plus_clinical", "2D_PCA_MATCHED_CLINICAL",
                    X_2d_eval, X_2d_dev, pca_components_by_modality[modality],
                    X_clinical, X_clinical_development,
                ),
            )

        for (
            analysis_variant,
            representation,
            X_eval,
            X_dev,
            pca_components,
            clinical_eval,
            clinical_dev,
        ) in analyses:
            clinical_text = " + clinical" if clinical_eval is not None else ""
            print(
                f"\nRunning {analysis_variant} | {modality} {representation}: "
                f"evaluation n={X_eval.shape[0]}, embedding p={X_eval.shape[1]}"
                f"{clinical_text}",
                flush=True,
            )

            pred_df, tuning_df = _fit_model(
                X_evaluation=X_eval,
                X_development=X_dev,
                pca_components=pca_components,
                clinical_evaluation=clinical_eval,
                clinical_development=clinical_dev,
                label=f"{analysis_variant} {modality} {representation}",
            )

            pred_df.insert(1, "data_scope", data_scope)
            pred_df.insert(2, "analysis_variant", analysis_variant)
            pred_df.insert(3, "modality", modality)
            pred_df.insert(4, "representation", representation)
            all_predictions.append(pred_df)

            tuning_df.insert(0, "data_scope", data_scope)
            tuning_df.insert(1, "analysis_variant", analysis_variant)
            tuning_df.insert(2, "modality", modality)
            tuning_df.insert(3, "representation", representation)
            all_tuning.append(tuning_df)

            n_clinical = 0 if clinical_eval is None else int(clinical_eval.shape[1])
            n_model_features = (
                int(pca_components) + n_clinical
                if pca_components is not None
                else X_eval.shape[1] + n_clinical
            )
            metric_row = _metric_row(
                modality=modality,
                representation=representation,
                analysis_variant=analysis_variant,
                pred_df=pred_df,
                n_input_features=X_eval.shape[1] + n_clinical,
                n_model_features=n_model_features,
                n_clinical_features=n_clinical,
            )
            metric_row["data_scope"] = data_scope
            metric_rows.append(metric_row)

            print(
                f"Completed {analysis_variant} | {modality} {representation}: "
                f"MAE={metric_row['MAE']:.4f}, MedAE={metric_row['MedAE']:.4f}, "
                f"R2={metric_row['R2']:.4f}",
                flush=True,
            )

    predictions = pd.concat(all_predictions, ignore_index=True)
    tuning = pd.concat(all_tuning, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)

    if data_scope == "test":
        # Final hard guard: every fitted model must contain exactly and only the
        # held-out VAE-test patients. No train/valid patient may appear downstream.
        for (variant, modality, representation), group in predictions.groupby(
            ["analysis_variant", "modality", "representation"], dropna=False
        ):
            _assert_exact_test_predictions(
                group, patient_ids, f"{variant} | {modality} | {representation}"
            )

    # Scope-neutral outputs.
    predictions.to_csv(output_dir / "predictions_all_variants.csv", index=False)
    tuning.to_csv(output_dir / "svr_best_hyperparameters_all_variants.csv", index=False)

    # Historical/OOF-compatible names are retained so existing notebooks keep working.
    # In strict test mode these files contain held-out test predictions, not OOF CV
    # predictions; use the data_scope/evaluation_set columns to distinguish them.
    predictions.to_csv(output_dir / "oof_predictions_all_variants.csv", index=False)
    tuning.to_csv(
        output_dir / "svr_best_hyperparameters_by_outer_fold_all_variants.csv",
        index=False,
    )
    metrics.to_csv(output_dir / "svr_metrics_all_variants.csv", index=False)
    if data_scope == "test":
        predictions.to_csv(output_dir / "test_predictions_all_variants.csv", index=False)
        tuning.to_csv(output_dir / "test_selected_hyperparameters_all_variants.csv", index=False)
        metrics.to_csv(output_dir / "test_metrics_all_variants.csv", index=False)

    # Legacy-compatible aliases so existing diagnostic notebooks can continue
    # to point at the familiar filenames. These files now contain all variants.
    predictions.to_csv(output_dir / "oof_predictions.csv", index=False)
    tuning.to_csv(
        output_dir / "svr_best_hyperparameters_by_outer_fold.csv", index=False
    )
    metrics.to_csv(output_dir / "svr_metrics.csv", index=False)

    # Convenience per-variant outputs.
    for variant, tag in [
        ("A_embedding_only", "A_embedding_only"),
        ("B_embedding_plus_clinical", "B_embedding_plus_clinical"),
        ("C_clinical_only", "C_clinical_only"),
    ]:
        predictions.loc[predictions["analysis_variant"] == variant].to_csv(
            output_dir / f"oof_predictions_{tag}.csv", index=False
        )
        tuning.loc[tuning["analysis_variant"] == variant].to_csv(
            output_dir / f"svr_best_hyperparameters_{tag}.csv", index=False
        )
        metrics.loc[metrics["analysis_variant"] == variant].to_csv(
            output_dir / f"svr_metrics_{tag}.csv", index=False
        )

    error_wide = predictions.loc[
        predictions["analysis_variant"].isin(
            ["A_embedding_only", "B_embedding_plus_clinical"]
        )
    ].pivot(
        index="patient_id",
        columns=["modality", "representation"],
        values="absolute_error",
    ).sort_index()
    error_wide.columns = [
        f"abs_error_{modality}_{_representation_key(representation)}"
        for modality, representation in error_wide.columns
    ]

    clinical_error = pred_c.set_index("patient_id")["absolute_error"].reindex(
        error_wide.index
    )
    error_wide["abs_error_clinical_only"] = clinical_error

    # ------------------------------------------------------------------
    # Primary A and B: 2D vs 3D
    # ------------------------------------------------------------------
    primary_frames: List[pd.DataFrame] = []
    primary_specs = [
        (
            "A_embedding_only",
            "2D",
            "3D",
            "A embedding-only: patient mean absolute error across modalities: 2D vs 3D",
        ),
        (
            "B_embedding_plus_clinical",
            "2D_CLINICAL",
            "3D_CLINICAL",
            "B embedding+clinical: patient mean absolute error across modalities: 2D vs 3D",
        ),
    ]
    for variant, rep_a, rep_b, label in primary_specs:
        frame = _primary_comparison(
            error_wide=error_wide,
            modalities=modalities,
            rep_a=rep_a,
            rep_b=rep_b,
            alternative=args.wilcoxon_alternative,
            comparison_label=label,
        )
        frame.insert(0, "analysis_variant", variant)
        frame.insert(0, "data_scope", data_scope)
        primary_frames.append(frame)
    primary_df = pd.concat(primary_frames, ignore_index=True)
    primary_df.to_csv(output_dir / "primary_2d_vs_3d_comparison.csv", index=False)

    # Secondary modality-specific A and B.
    secondary_frames: List[pd.DataFrame] = []
    for variant, rep_a, rep_b, _ in primary_specs:
        frame = _add_modality_specific_tests(
            error_wide=error_wide,
            modalities=modalities,
            rep_a=rep_a,
            rep_b=rep_b,
            alternative=args.wilcoxon_alternative,
        )
        frame.insert(0, "analysis_variant", variant)
        frame.insert(0, "data_scope", data_scope)
        secondary_frames.append(frame)
    secondary_df = pd.concat(secondary_frames, ignore_index=True)
    secondary_df.to_csv(
        output_dir / "secondary_modality_specific_2d_vs_3d.csv", index=False
    )

    sensitivity_primary_df: Optional[pd.DataFrame] = None
    sensitivity_secondary_df: Optional[pd.DataFrame] = None
    if not args.skip_pca_sensitivity:
        sensitivity_specs = [
            (
                "A_embedding_only",
                "2D_PCA_MATCHED",
                "3D",
                "A embedding-only: modality-wise dimension-matched 2D PCA vs 3D",
            ),
            (
                "B_embedding_plus_clinical",
                "2D_PCA_MATCHED_CLINICAL",
                "3D_CLINICAL",
                "B embedding+clinical: modality-wise dimension-matched 2D PCA vs 3D",
            ),
        ]
        sensitivity_frames = []
        sensitivity_secondary_frames = []
        for variant, rep_a, rep_b, label in sensitivity_specs:
            frame = _primary_comparison(
                error_wide=error_wide,
                modalities=modalities,
                rep_a=rep_a,
                rep_b=rep_b,
                alternative=args.wilcoxon_alternative,
                comparison_label=label,
            )
            frame.insert(0, "analysis_variant", variant)
            frame.insert(0, "data_scope", data_scope)
            sensitivity_frames.append(frame)

            secondary = _add_modality_specific_tests(
                error_wide=error_wide,
                modalities=modalities,
                rep_a=rep_a,
                rep_b=rep_b,
                alternative=args.wilcoxon_alternative,
            )
            secondary.insert(0, "analysis_variant", variant)
            secondary.insert(0, "data_scope", data_scope)
            secondary["pca_components"] = (
                secondary["modality"].map(pca_components_by_modality).astype(int)
            )
            sensitivity_secondary_frames.append(secondary)

        sensitivity_primary_df = pd.concat(sensitivity_frames, ignore_index=True)
        sensitivity_primary_df.to_csv(
            output_dir / "sensitivity_dimension_matched_2d_vs_3d_comparison.csv",
            index=False,
        )
        sensitivity_secondary_df = pd.concat(
            sensitivity_secondary_frames, ignore_index=True
        )
        sensitivity_secondary_df.to_csv(
            output_dir / "sensitivity_modality_specific_dimension_matched_2d_vs_3d.csv",
            index=False,
        )

    # ------------------------------------------------------------------
    # Incremental value of embeddings beyond clinical-only model.
    # ------------------------------------------------------------------
    added_value_rows: List[dict] = []
    clinical_error_series = error_wide["abs_error_clinical_only"]
    added_value_specs = [
        ("2D_CLINICAL", "B: clinical + 2D"),
        ("3D_CLINICAL", "B: clinical + 3D"),
    ]
    if not args.skip_pca_sensitivity:
        added_value_specs.append(
            ("2D_PCA_MATCHED_CLINICAL", "B: clinical + PCA-matched 2D")
        )

    for rep, label in added_value_specs:
        model_error = error_wide[
            [f"abs_error_{m}_{_representation_key(rep)}" for m in modalities]
        ].mean(axis=1)
        added_value_rows.append(
            _paired_model_vs_reference(
                model_error=model_error,
                reference_error=clinical_error_series,
                model_label=label,
                reference_label="C: clinical only",
                alternative=args.wilcoxon_alternative,
            )
        )
    added_value_df = pd.DataFrame(added_value_rows)
    added_value_df.insert(0, "data_scope", data_scope)
    added_value_df["p_value_BH"] = _benjamini_hochberg(
        added_value_df["p_value_raw"].to_numpy()
    )
    added_value_df["p_value_Holm"] = _holm_adjust(
        added_value_df["p_value_raw"].to_numpy()
    )
    added_value_df.to_csv(
        output_dir / "embedding_plus_clinical_vs_clinical_only.csv", index=False
    )

    # Save patient-level errors with A/B/C summary columns.
    patient_error_df = error_wide.reset_index()
    patient_error_df.insert(1, "data_scope", data_scope)
    patient_error_df.to_csv(
        output_dir / "patient_level_absolute_errors_all_variants.csv", index=False
    )
    patient_error_df.to_csv(
        output_dir / "patient_level_absolute_errors.csv", index=False
    )

    # Compare every main variant with the leakage-free median dummy.
    dummy_median_error = (
        dummy_median_df.set_index("patient_id")["absolute_error"]
        .reindex(error_wide.index)
    )
    vs_dummy_specs = [
        (
            error_wide[[f"abs_error_{m}_2d" for m in modalities]].mean(axis=1),
            "A: embedding-only 2D",
        ),
        (
            error_wide[[f"abs_error_{m}_3d" for m in modalities]].mean(axis=1),
            "A: embedding-only 3D",
        ),
        (
            error_wide[[f"abs_error_{m}_2d_clinical" for m in modalities]].mean(axis=1),
            "B: clinical + 2D",
        ),
        (
            error_wide[[f"abs_error_{m}_3d_clinical" for m in modalities]].mean(axis=1),
            "B: clinical + 3D",
        ),
        (clinical_error_series, "C: clinical only"),
    ]
    if not args.skip_pca_sensitivity:
        vs_dummy_specs.extend(
            [
                (
                    error_wide[
                        [f"abs_error_{m}_2d_pca_matched" for m in modalities]
                    ].mean(axis=1),
                    "A: PCA-matched 2D",
                ),
                (
                    error_wide[
                        [
                            f"abs_error_{m}_2d_pca_matched_clinical"
                            for m in modalities
                        ]
                    ].mean(axis=1),
                    "B: clinical + PCA-matched 2D",
                ),
            ]
        )

    vs_dummy_rows = [
        _paired_model_vs_reference(
            model_error=model_error,
            reference_error=dummy_median_error,
            model_label=model_label,
            reference_label="DummyRegressor_median",
            alternative=args.wilcoxon_alternative,
        )
        for model_error, model_label in vs_dummy_specs
    ]
    vs_dummy_df = pd.DataFrame(vs_dummy_rows)
    vs_dummy_df.insert(0, "data_scope", data_scope)
    vs_dummy_df["p_value_BH"] = _benjamini_hochberg(
        vs_dummy_df["p_value_raw"].to_numpy()
    )
    vs_dummy_df["p_value_Holm"] = _holm_adjust(
        vs_dummy_df["p_value_raw"].to_numpy()
    )
    vs_dummy_df.to_csv(
        output_dir / "variant_vs_median_dummy_comparison.csv", index=False
    )

    # ------------------------------------------------------------------
    # Figures: separate figures for A, B and C; plus PCA sensitivity.
    # ------------------------------------------------------------------
    scope_title = (
        "held-out VAE test patients" if data_scope == "test" else "all-data nested CV"
    )
    _save_variant_comparison_figure(
        error_wide=error_wide,
        predictions=predictions,
        modalities=modalities,
        rep_a="2D",
        rep_b="3D",
        label_a="2D",
        label_b="3D",
        title=f"Variant A: embedding-only ({scope_title})",
        output_stem=output_dir / "figure_A_embedding_only",
    )
    _save_variant_comparison_figure(
        error_wide=error_wide,
        predictions=predictions,
        modalities=modalities,
        rep_a="2D_CLINICAL",
        rep_b="3D_CLINICAL",
        label_a="2D + clinical",
        label_b="3D + clinical",
        title=f"Variant B: embedding + clinical predictors ({scope_title})",
        output_stem=output_dir / "figure_B_embedding_plus_clinical",
    )
    _save_clinical_only_figure(
        pred_df=pred_c,
        title=f"Variant C: clinical-only ({scope_title})",
        output_stem=output_dir / "figure_C_clinical_only",
    )

    if not args.skip_pca_sensitivity:
        _save_variant_comparison_figure(
            error_wide=error_wide,
            predictions=predictions,
            modalities=modalities,
            rep_a="2D_PCA_MATCHED",
            rep_b="3D",
            label_a="PCA-matched 2D",
            label_b="3D",
            title=f"Variant A: PCA dimension-matched sensitivity ({scope_title})",
            output_stem=output_dir / "figure_A_PCA_sensitivity",
        )
        _save_variant_comparison_figure(
            error_wide=error_wide,
            predictions=predictions,
            modalities=modalities,
            rep_a="2D_PCA_MATCHED_CLINICAL",
            rep_b="3D_CLINICAL",
            label_a="PCA-matched 2D + clinical",
            label_b="3D + clinical",
            title=f"Variant B: PCA dimension-matched sensitivity ({scope_title})",
            output_stem=output_dir / "figure_B_PCA_sensitivity",
        )

    # Primary Holm family remains the embedding-only A comparison, preserving the
    # thesis-level family definition used previously.
    primary_a_p = float(
        primary_df.loc[
            primary_df["analysis_variant"] == "A_embedding_only", "p_value_raw"
        ].iloc[0]
    )
    primary_family = pd.DataFrame(
        [
            {
                "data_scope": data_scope,
                "hypothesis": "Imagix 2D vs Imagix3D (variant A embedding-only)",
                "p_value_raw": primary_a_p,
                "holm_family_status": (
                    "pending second primary p-value: Imagix3D vs XModalix3D"
                ),
            }
        ]
    )
    primary_family.to_csv(
        output_dir / "primary_hypothesis_pvalue_for_future_holm.csv", index=False
    )

    metadata = {
        "timestamp": timestamp,
        "data_scope": data_scope,
        "scope_definition": (
            "Strict held-out evaluation: VAE train+valid patients are used only for "
            "SVR hyperparameter selection/refitting; only VAE test patients contribute "
            "predictions, metrics, statistics and figures."
            if data_scope == "test"
            else "Nested-CV evaluation over the complete common cohort."
        ),
        "clinical_file": str(clinical_path),
        "target_column": args.target_col,
        "clinical_predictor_columns": list(clinical_predictor_cols),
        "analysis_variants": {
            "A": "embedding only",
            "B": "embedding + clinical predictors",
            "C": "clinical predictors only",
        },
        "modalities": modalities,
        "embedding_files": resolved_paths,
        "global_common_cohort_n": int(len(all_common_ids)),
        "development_n": (
            int(len(development_ids)) if data_scope == "test" else None
        ),
        "evaluation_n": int(len(patient_ids)),
        "evaluation_split": "test" if data_scope == "test" else "all",
        "outer_folds": args.outer_folds if data_scope == "all" else None,
        "inner_folds": args.inner_folds,
        "outer_cv": (
            {
                "class": "KFold",
                "shuffle": True,
                "random_state": args.random_state,
            }
            if data_scope == "all"
            else None
        ),
        "heldout_test_design": (
            {
                "development_splits": ["train", "valid"],
                "evaluation_split": "test",
                "split_column": args.clinical_split_col,
                "embedding_split_match_required": True,
                "allow_vae_split_mismatch_ignored_for_strict_test": True,
            }
            if data_scope == "test"
            else None
        ),
        "inner_cv": {
            "class": "KFold",
            "shuffle": True,
            "random_state_rule": (
                "random_state + outer_fold" if data_scope == "all" else "random_state"
            ),
        },
        "model": f"sklearn.svm.SVR(kernel={kernel!r})",
        "kernel": kernel,
        "variant_A_pipeline": f"StandardScaler -> {kernel} SVR",
        "variant_B_non_pca_pipeline": (
            f"concatenate embedding + clinical -> StandardScaler -> {kernel} SVR"
        ),
        "variant_B_pca_pipeline": (
            f"embedding StandardScaler -> PCA(d_m); clinical StandardScaler; "
            f"concatenate -> {kernel} SVR"
            if not args.skip_pca_sensitivity
            else None
        ),
        "variant_C_pipeline": f"clinical StandardScaler -> {kernel} SVR",
        "pca_sensitivity_enabled": not args.skip_pca_sensitivity,
        "pca_components_by_modality": (
            pca_components_by_modality if not args.skip_pca_sensitivity else None
        ),
        "inner_scoring": "neg_mean_absolute_error",
        "C_values": list(c_values),
        "epsilon_values": list(epsilon_values),
        "gamma_values": (
            list(gamma_values) if kernel in {"rbf", "poly", "sigmoid"} else None
        ),
        "degree_values": list(degree_values) if kernel == "poly" else None,
        "coef0_values": (
            list(coef0_values) if kernel in {"poly", "sigmoid"} else None
        ),
        "wilcoxon_alternative": args.wilcoxon_alternative,
        "primary_thesis_holm_family": (
            "Variant A Imagix 2D vs Imagix3D AND Imagix3D vs XModalix3D; "
            "adjust jointly once both raw p-values are available"
        ),
        "n_jobs": args.n_jobs,
    }
    with open(output_dir / "analysis_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 72)
    print("ANALYSIS COMPLETE")
    print("=" * 72)
    print("\nMetrics across variants:")
    print(metrics.to_string(index=False))
    print("\nPrimary A/B 2D-vs-3D comparisons:")
    print(primary_df.to_string(index=False))
    print("\nEmbedding+clinical versus clinical-only:")
    print(added_value_df.to_string(index=False))
    if sensitivity_primary_df is not None:
        print("\nPCA sensitivity A/B comparisons:")
        print(sensitivity_primary_df.to_string(index=False))
    print(f"\nResults written to: {output_dir}")

    return output_dir

def main() -> None:
    args = parse_args()
    run_analysis(args)


if __name__ == "__main__":
    main()
