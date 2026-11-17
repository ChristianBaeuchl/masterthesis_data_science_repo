"""
Aggregate slice-wise Imagix latent z embeddings into patient-wise vectors.

For each modality, the script:
1. Loads one or more Imagix `imagix_final_z_all_splits.csv` files.
2. Parses patient ID and slice number from sample IDs such as:
       2015_axial_sl_10
3. Sorts slices explicitly from bottom to top by slice number.
4. Concatenates the latent z vectors in this order:
       slice 1 LatDim_0 ... LatDim_D-1,
       slice 2 LatDim_0 ... LatDim_D-1,
       ...
5. Writes ONE patient-wise CSV PER MODALITY.

Different modalities may have different latent dimensions. Because every
modality is saved separately, each output CSV contains exactly the required
number of feature columns for that modality and no padding/NaN columns.

"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# USER SETTINGS
# =============================================================================

# Working directory in which the output CSV files will be written.
wd = r"C:/chris/FU/Msc_Data_Science/Masterarbeit/scripts"
os.chdir(wd)


# Expected number of slices per patient.
#
# Current Imagix runs:
#     10 slices per patient.
#
# If future runs use a different fixed number, simply change this value.
#
# Set to None if you want the script to infer the slice count separately for
# each modality. In that case, all patients within a modality must still have
# the same number of slices.
EXPECTED_SLICES_PER_PATIENT: Optional[int] = 10


# Sample IDs are currently of the form:
#     2015_axial_sl_10
#
# The first capture group is the patient ID and the second is the slice number.
# Change this only if the sample-ID naming scheme changes.
SAMPLE_ID_PATTERN = re.compile(
    r"^(?P<patient_id>.+)_axial_sl_(?P<slice_number>\d+)$"
)

# Output separator.
OUTPUT_SEPARATOR = ","

# Require all five modalities to contain exactly the same patients.
# This matches the intended current dataset and prevents silent mismatches.
REQUIRE_IDENTICAL_PATIENT_SETS = True

# Also require each patient to belong to the same train/valid/test split in
# every modality.
REQUIRE_IDENTICAL_SPLITS_ACROSS_MODALITIES = True


# -----------------------------------------------------------------------------
# Input paths
#
#
# Use exactly one CSV path per modality.
# A list is retained so multiple non-overlapping CSV shards could also be used.
#
# IMPORTANT:
# If two files for the same modality contain the same sample_id, the script
# raises an error.
# -----------------------------------------------------------------------------

# NCCT
imagix_ncct = []

# CTA
imagix_cta = []

# CBF
imagix_cbf = []

# CBV
imagix_cbv = []

# TMAX
imagix_tmax = []


# The insertion order here is used consistently for validation and reporting.
MODALITY_FILES: Dict[str, List[str]] = {
    "NCCT": imagix_ncct,
    "CTA": imagix_cta,
    "CBF": imagix_cbf,
    "CBV": imagix_cbv,
    "TMAX": imagix_tmax,
}

# Output filenames.
OUTPUT_FILENAMES: Dict[str, str] = {
    "NCCT": "imagix_ncct_patientwise_z.csv",
    "CTA": "imagix_cta_patientwise_z.csv",
    "CBF": "imagix_cbf_patientwise_z.csv",
    "CBV": "imagix_cbv_patientwise_z.csv",
    "TMAX": "imagix_tmax_patientwise_z.csv",
}

# HELPERS

_LATENT_COLUMN_PATTERN = re.compile(r"^LatDim_(\d+)$")


def _natural_sort_key(value: str):
    """Natural sorting, so e.g. patient 9 comes before patient 10."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    ]


def _detect_csv_separator(path: Path) -> str:
    """Detect comma- versus semicolon-delimited CSV from the header."""
    with path.open("r", encoding="utf-8-sig") as f:
        header = f.readline()

    n_semicolon = header.count(";")
    n_comma = header.count(",")

    if n_semicolon == 0 and n_comma == 0:
        raise ValueError(
            f"Could not detect CSV separator in header of:\n{path}"
        )

    return ";" if n_semicolon > n_comma else ","


def _latent_columns(df: pd.DataFrame) -> List[str]:
    """Return LatDim_* columns in numeric latent-dimension order."""
    indexed_columns: List[Tuple[int, str]] = []

    for column in df.columns:
        match = _LATENT_COLUMN_PATTERN.fullmatch(str(column))
        if match:
            indexed_columns.append((int(match.group(1)), str(column)))

    if not indexed_columns:
        raise ValueError(
            "No columns named LatDim_0, LatDim_1, ... were found."
        )

    indexed_columns.sort(key=lambda item: item[0])

    observed_indices = [idx for idx, _ in indexed_columns]
    expected_indices = list(range(len(indexed_columns)))

    if observed_indices != expected_indices:
        raise ValueError(
            "Latent columns are not contiguous from LatDim_0 onward.\n"
            f"Observed latent indices: {observed_indices}"
        )

    return [column for _, column in indexed_columns]


def _read_one_latent_csv(path_string: str) -> pd.DataFrame:
    """Read and minimally validate one Imagix z CSV."""
    path = Path(path_string).expanduser()

    if not path.is_file():
        raise FileNotFoundError(
            f"Input CSV does not exist:\n{path}"
        )

    sep = _detect_csv_separator(path)

    df = pd.read_csv(path, sep=sep)

    # Remove completely empty rows.
    df = df.dropna(how="all").copy()

    required_columns = {"sample_id", "split"}

    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required columns: {sorted(missing)}"
        )

    latent_cols = _latent_columns(df)

    # Keep only the fields needed for this aggregation.
    df = df[["sample_id", "split", *latent_cols]].copy()

    if df["sample_id"].isna().any():
        bad_rows = df.index[df["sample_id"].isna()].tolist()[:10]
        raise ValueError(
            f"{path} contains rows without sample_id after empty-row cleanup. "
            f"Example row indices: {bad_rows}"
        )

    df["sample_id"] = df["sample_id"].astype(str).str.strip()
    df["split"] = df["split"].astype(str).str.strip()

    # Convert all latent values to numeric and fail on malformed values.
    for column in latent_cols:
        df[column] = pd.to_numeric(df[column], errors="raise")

    latent_array = df[latent_cols].to_numpy(dtype=float)

    if not np.isfinite(latent_array).all():
        bad = np.argwhere(~np.isfinite(latent_array))[0]
        row_idx, col_idx = int(bad[0]), int(bad[1])
        raise ValueError(
            f"{path} contains a non-finite latent value at "
            f"row {row_idx}, column {latent_cols[col_idx]!r}."
        )

    return df


def _load_modality_files(
    modality: str,
    file_paths: Sequence[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """Load all CSVs supplied for one modality."""
    clean_paths = [
        str(path).strip()
        for path in file_paths
        if str(path).strip()
    ]

    if not clean_paths:
        raise ValueError(
            f"No input CSV path was supplied for modality {modality}."
        )

    frames = [_read_one_latent_csv(path) for path in clean_paths]

    # Require the same latent dimensionality across multiple files belonging
    # to one modality.
    latent_column_sets = [_latent_columns(frame) for frame in frames]
    first_latent_cols = latent_column_sets[0]

    for idx, latent_cols in enumerate(latent_column_sets[1:], start=2):
        if latent_cols != first_latent_cols:
            raise ValueError(
                f"Latent dimensionality differs across CSV files for "
                f"{modality}. File 1 has {len(first_latent_cols)} dimensions, "
                f"file {idx} has {len(latent_cols)}."
            )

    df = pd.concat(frames, ignore_index=True)

    duplicate_mask = df["sample_id"].duplicated(keep=False)
    if duplicate_mask.any():
        duplicates = (
            df.loc[duplicate_mask, "sample_id"]
            .drop_duplicates()
            .tolist()[:10]
        )
        raise ValueError(
            f"Duplicate sample_id values found for {modality}. "
            "The same slice must not occur in more than one input row/file.\n"
            f"Examples: {duplicates}"
        )

    return df, first_latent_cols


def _parse_patient_and_slice(
    df: pd.DataFrame,
    modality: str,
) -> pd.DataFrame:
    """Extract patient_id and numeric slice_number from sample_id."""
    parsed = df["sample_id"].str.extract(SAMPLE_ID_PATTERN)

    if parsed.isna().any(axis=None):
        bad_mask = parsed.isna().any(axis=1)
        bad_ids = df.loc[bad_mask, "sample_id"].tolist()[:10]
        raise ValueError(
            f"Could not parse patient/slice information for {modality}.\n"
            f"Expected IDs like '2015_axial_sl_10'.\n"
            f"Examples that failed: {bad_ids}"
        )

    out = df.copy()
    out["patient_id"] = parsed["patient_id"].astype(str)
    out["slice_number"] = parsed["slice_number"].astype(int)

    return out


def _validate_patient_slices(
    df: pd.DataFrame,
    modality: str,
) -> int:
    """
    Validate slice counts/order and return the slice count for this modality.
    """
    counts = df.groupby("patient_id", sort=False).size()

    if EXPECTED_SLICES_PER_PATIENT is not None:
        n_slices = int(EXPECTED_SLICES_PER_PATIENT)

        wrong = counts[counts != n_slices]
        if not wrong.empty:
            examples = wrong.head(10).to_dict()
            raise ValueError(
                f"{modality}: expected exactly {n_slices} slices per patient, "
                "but some patients differ.\n"
                f"Examples patient_id -> observed slices: {examples}"
            )

    else:
        unique_counts = sorted(counts.unique().tolist())

        if len(unique_counts) != 1:
            raise ValueError(
                f"{modality}: slice count differs between patients: "
                f"{unique_counts}.\n"
                "Every patient within a modality "
                "must have the same vector length."
            )

        n_slices = int(unique_counts[0])

    expected_slice_numbers = set(range(1, n_slices + 1))

    # Validate each patient has exactly slices 1..N, without gaps or duplicates.
    for patient_id, patient_df in df.groupby("patient_id", sort=False):
        observed = patient_df["slice_number"].tolist()
        observed_set = set(observed)

        if len(observed) != len(observed_set):
            raise ValueError(
                f"{modality}, patient {patient_id}: duplicate slice numbers "
                f"found: {sorted(observed)}"
            )

        if observed_set != expected_slice_numbers:
            raise ValueError(
                f"{modality}, patient {patient_id}: expected slice numbers "
                f"{sorted(expected_slice_numbers)}, observed "
                f"{sorted(observed_set)}."
            )

    return n_slices


def _validate_split_consistency_within_modality(
    df: pd.DataFrame,
    modality: str,
) -> Dict[str, str]:
    """Require all slices of one patient to have the same split."""
    split_counts = df.groupby("patient_id")["split"].nunique()

    inconsistent = split_counts[split_counts != 1]

    if not inconsistent.empty:
        patients = inconsistent.index.astype(str).tolist()[:10]
        raise ValueError(
            f"{modality}: some patients have slices assigned to multiple "
            f"splits. Examples: {patients}"
        )

    return (
        df.groupby("patient_id")["split"]
        .first()
        .astype(str)
        .to_dict()
    )


def _aggregate_one_modality(
    modality: str,
    file_paths: Sequence[str],
):
    """
    Create one concatenated z vector per patient for one modality.

    Flattening uses row-major order after sorting by slice number:
        slice 1: LatDim_0 ... LatDim_D-1
        slice 2: LatDim_0 ... LatDim_D-1
        ...
    """
    df, latent_cols = _load_modality_files(modality, file_paths)
    df = _parse_patient_and_slice(df, modality)

    n_slices = _validate_patient_slices(df, modality)
    split_by_patient = _validate_split_consistency_within_modality(
        df,
        modality,
    )

    latent_dim = len(latent_cols)
    vector_length = n_slices * latent_dim

    patient_ids = sorted(
        df["patient_id"].unique().tolist(),
        key=_natural_sort_key,
    )

    vectors: Dict[str, np.ndarray] = {}

    for patient_id in patient_ids:
        patient_df = (
            df.loc[df["patient_id"] == patient_id]
            .sort_values("slice_number")
        )

        # Shape: (n_slices, latent_dim)
        matrix = patient_df[latent_cols].to_numpy(dtype=float)

        # Row-major flattening guarantees:
        # slice 1 dims first, then slice 2 dims, ...
        vector = matrix.reshape(-1, order="C")

        if vector.shape[0] != vector_length:
            raise RuntimeError(
                f"Internal length mismatch for {modality}, patient "
                f"{patient_id}: expected {vector_length}, got "
                f"{vector.shape[0]}."
            )

        vectors[patient_id] = vector

    print(
        f"{modality}: "
        f"{len(patient_ids)} patients, "
        f"{n_slices} slices/patient, "
        f"{latent_dim} latent dims/slice -> "
        f"{vector_length} values/patient"
    )

    return {
        "modality": modality,
        "patient_ids": patient_ids,
        "split_by_patient": split_by_patient,
        "n_slices": n_slices,
        "latent_dim": latent_dim,
        "vector_length": vector_length,
        "vectors": vectors,
    }


def _validate_across_modalities(results: Dict[str, dict]) -> List[str]:
    """Validate patient and split correspondence across modalities."""
    modalities = list(results.keys())
    reference_modality = modalities[0]

    reference_patients = set(results[reference_modality]["patient_ids"])

    if REQUIRE_IDENTICAL_PATIENT_SETS:
        for modality in modalities[1:]:
            patients = set(results[modality]["patient_ids"])

            if patients != reference_patients:
                missing = sorted(
                    reference_patients - patients,
                    key=_natural_sort_key,
                )[:10]

                extra = sorted(
                    patients - reference_patients,
                    key=_natural_sort_key,
                )[:10]

                raise ValueError(
                    f"Patient set mismatch: {reference_modality} vs "
                    f"{modality}.\n"
                    f"Missing from {modality} (up to 10): {missing}\n"
                    f"Extra in {modality} (up to 10): {extra}"
                )

    common_patients = set.intersection(
        *[
            set(results[modality]["patient_ids"])
            for modality in modalities
        ]
    )

    patient_order = sorted(common_patients, key=_natural_sort_key)

    if REQUIRE_IDENTICAL_SPLITS_ACROSS_MODALITIES:
        for patient_id in patient_order:
            split_map = {
                modality: results[modality]["split_by_patient"][patient_id]
                for modality in modalities
            }

            if len(set(split_map.values())) != 1:
                raise ValueError(
                    f"Split mismatch across modalities for patient "
                    f"{patient_id}: {split_map}"
                )

    return patient_order


def _build_modality_output_table(
    result: dict,
    patient_order: Sequence[str],
) -> pd.DataFrame:
    """Build one rectangular patient-wise output DataFrame for one modality."""
    vector_length = result["vector_length"]

    feature_columns = [
        f"feature_{idx:04d}"
        for idx in range(1, vector_length + 1)
    ]

    rows = []

    for patient_id in patient_order:
        vector = result["vectors"][patient_id]

        row = {
            "patient_id": patient_id,
            "split": result["split_by_patient"][patient_id],
            "n_slices": result["n_slices"],
            "latent_dim_per_slice": result["latent_dim"],
            "vector_length": vector_length,
        }

        row.update(
            {
                feature_columns[idx]: value
                for idx, value in enumerate(vector)
            }
        )

        rows.append(row)

    output_df = pd.DataFrame(rows)

    metadata_columns = [
        "patient_id",
        "split",
        "n_slices",
        "latent_dim_per_slice",
        "vector_length",
    ]

    output_df = output_df[
        metadata_columns + feature_columns
    ]

    return output_df


# =============================================================================
# MAIN

def main() -> None:
    print("Aggregating slice-wise Imagix z embeddings...")
    print()

    results: Dict[str, dict] = {}

    for modality, file_paths in MODALITY_FILES.items():
        results[modality] = _aggregate_one_modality(
            modality=modality,
            file_paths=file_paths,
        )

    print()
    print("Validating patient correspondence across modalities...")

    patient_order = _validate_across_modalities(results)

    print(
        f"Common validated patient count: {len(patient_order)}"
    )

    print()
    print("Writing modality-specific CSV files...")

    written_paths = {}

    for modality, result in results.items():
        output_df = _build_modality_output_table(
            result=result,
            patient_order=patient_order,
        )

        output_path = (
            Path.cwd()
            / OUTPUT_FILENAMES[modality]
        )

        output_df.to_csv(
            output_path,
            index=False,
            sep=OUTPUT_SEPARATOR,
        )

        written_paths[modality] = output_path

        print(
            f"  {modality}: {output_path.name} | "
            f"{len(output_df)} patients | "
            f"{result['vector_length']} features"
        )

    print()
    print("=" * 78)
    print("Aggregation complete")
    print("=" * 78)

    for modality in MODALITY_FILES:
        print(
            f"{modality}: {written_paths[modality]}"
        )

    print()
    print("Vector specification by modality:")

    summary_rows = []

    for modality, result in results.items():
        summary_rows.append(
            {
                "modality": modality,
                "patients": len(patient_order),
                "n_slices": result["n_slices"],
                "latent_dim_per_slice": result["latent_dim"],
                "vector_length": result["vector_length"],
            }
        )

    summary = pd.DataFrame(summary_rows)

    print(summary.to_string(index=False))



if __name__ == "__main__":
    main()
