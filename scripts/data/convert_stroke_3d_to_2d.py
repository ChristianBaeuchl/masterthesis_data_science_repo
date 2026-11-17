from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import tifffile


# -----------------------------------------------------------------------------
# USER SETTINGS
# -----------------------------------------------------------------------------
# WARNING: slice_num must evenly divide the size of the selected slicing axis.
# For the current 160 x 192 x 160 volumes:
#   axial     -> 160 slices available
#   sagittal  -> 160 slices available
#   coronal   -> 192 slices available
slice_num = 10

# Choose one of: "axial", "coronal", "sagittal"
slice_plane = "axial"

# Pad every extracted 2D slice to this square size without resampling.
# For the current 160 x 192 x 160 volumes, 192 preserves all original pixels.
target_size_2d = 192


# -----------------------------------------------------------------------------
# PATHS / DATASET SETTINGS
# -----------------------------------------------------------------------------
STROKE_ROOT = Path(
    r""
)

INPUT_ROOT = STROKE_ROOT / "3D"
SOURCE_ROOT = INPUT_ROOT / "CT_Imaging"
OUTPUT_ROOT = STROKE_ROOT / "2D"

MODALITIES = ["cbf", "cbv", "cta", "gt", "ncct", "tmax"]
ANNOTATION_SUFFIXES = ["anno", "anno_dst"]

VALID_PLANES = {"axial", "coronal", "sagittal"}


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def strip_nii_suffix(filename: str) -> str:
    """Remove .nii or .nii.gz from a filename."""
    lower = filename.lower()

    if lower.endswith(".nii.gz"):
        return filename[:-7]

    if lower.endswith(".nii"):
        return filename[:-4]

    return Path(filename).stem


def sample_id_to_str(value) -> str:
    """
    Convert a CSV sample_id to a clean string.

    For example:
        564.0 -> "564"
    """
    if pd.isna(value):
        return ""

    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))

    return str(value).strip()


def prepare_volume_for_tiff(volume: np.ndarray) -> np.ndarray:
    """
    Prepare a 3D volume for TIFF export while preserving original intensities.

    No rescaling is performed.
    The output remains float32.

    NaN / +/-inf values are replaced by 0.0.
    """
    volume = np.asarray(volume, dtype=np.float32)

    volume = np.nan_to_num(
        volume,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return volume


def get_axis_for_plane(plane: str) -> int:
    """
    Return the voxel axis after conversion to canonical RAS+ orientation.

    RAS+ axes:
        axis 0 = left -> right
        axis 1 = posterior -> anterior
        axis 2 = inferior -> superior

    Therefore:
        sagittal -> axis 0
        coronal  -> axis 1
        axial    -> axis 2
    """
    return {
        "sagittal": 0,
        "coronal": 1,
        "axial": 2,
    }[plane]


def choose_slice_indices(
    axis_length: int,
    n_slices: int,
) -> list[int]:
    """
    Divide an axis into equally sized blocks and take the centre slice
    of every block.

    Example:
        axis_length = 160
        n_slices = 10

        block size = 16

        selected indices:
        8, 24, 40, 56, 72, 88, 104, 120, 136, 152

    This samples the whole volume evenly while avoiding selecting only
    boundary slices.
    """
    if n_slices <= 0:
        raise ValueError(
            "slice_num must be greater than 0."
        )

    if axis_length % n_slices != 0:
        raise ValueError(
            f"slice_num={n_slices} does not evenly divide the selected "
            f"axis length {axis_length}. "
            f"Choose a divisor of {axis_length}."
        )

    block_size = axis_length // n_slices

    return [
        i * block_size + block_size // 2
        for i in range(n_slices)
    ]


def extract_slice(
    volume: np.ndarray,
    plane: str,
    index: int,
) -> np.ndarray:
    """
    Extract one 2D anatomical slice from a canonical RAS+ volume.

    Array shape conventions:
        TIFF expects an image array of shape
            (height, width)         for grayscale
        or
            (height, width, 3)      for 3-channel images

    For a canonical volume of shape:
        (160, 192, 160)

    an axial slice initially has:
        (160, 192)

    It is transposed so that the resulting 2D image has:
        width  = 160
        height = 192
    """
    if plane == "axial":
        # volume[:, :, z] -> shape (X, Y)
        return volume[:, :, index].T

    if plane == "coronal":
        # volume[:, y, :] -> shape (X, Z)
        return volume[:, index, :].T

    if plane == "sagittal":
        # volume[x, :, :] -> shape (Y, Z)
        return volume[index, :, :].T

    raise ValueError(
        f"Unknown slice plane: {plane}"
    )


def pad_slice_to_square(
    slice_2d: np.ndarray,
    target_size: int,
    pad_value: float = 0.0,
) -> np.ndarray:
    """
    Symmetrically pad a 2D slice to target_size x target_size.

    No interpolation or rescaling is performed, so the original image geometry
    and intensity values are preserved. Padding is added equally on both sides
    whenever possible; if an odd number of pixels is required, the extra pixel
    is added to the bottom or right.

    For the current 160 x 192 x 160 volumes:
      - axial slices    (192, 160) -> 16 px left + 16 px right
      - coronal slices  (160, 160) -> 16 px on all four sides
      - sagittal slices (160, 192) -> 16 px top + 16 px bottom
    """
    slice_2d = np.asarray(slice_2d, dtype=np.float32)

    if slice_2d.ndim != 2:
        raise ValueError(
            f"Expected a 2D slice, got shape {slice_2d.shape}."
        )

    height, width = slice_2d.shape

    if height > target_size or width > target_size:
        raise ValueError(
            f"Slice shape {slice_2d.shape} exceeds target size "
            f"{target_size} x {target_size}. Cropping is intentionally disabled."
        )

    pad_height = target_size - height
    pad_width = target_size - width

    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    return np.pad(
        slice_2d,
        pad_width=(
            (pad_top, pad_bottom),
            (pad_left, pad_right),
        ),
        mode="constant",
        constant_values=pad_value,
    )


def make_three_channel_float32(slice_2d: np.ndarray) -> np.ndarray:
    """
    Convert a 2D slice to a 3-channel float32 image with identical channels.

    This is done intentionally because the current unmodified Imagix TIFF
    reader expects TIFF images in a shape compatible with 3 channels and then
    converts them to grayscale anyway.

    If R = G = B = x, then the grayscale conversion preserves x.
    """
    slice_2d = np.asarray(slice_2d, dtype=np.float32)

    slice_3ch = np.repeat(
        slice_2d[..., np.newaxis],
        repeats=3,
        axis=2,
    )

    return slice_3ch


def convert_nifti_file(
    nifti_path: Path,
    output_dir: Path,
    plane: str,
    n_slices: int,
) -> list[Path]:
    """
    Convert one 3D NIfTI file into a specified number of TIFF slices.

    Slices are exported as 3-channel float32 TIFF images in order to remain
    compatible with the existing unmodified Imagix TIFF loading code while
    preserving the original intensity values.
    """
    img = nib.load(str(nifti_path))

    # Convert to canonical RAS+ orientation without resampling.
    #
    # For axial slicing this ensures that increasing slice indices
    # correspond to inferior -> superior.
    img = nib.as_closest_canonical(img)

    volume = np.asarray(img.get_fdata())
    volume = np.squeeze(volume)

    if volume.ndim != 3:
        raise ValueError(
            f"Expected a 3D volume for {nifti_path}, "
            f"but got shape {volume.shape}."
        )

    axis = get_axis_for_plane(plane)

    indices = choose_slice_indices(
        axis_length=volume.shape[axis],
        n_slices=n_slices,
    )

    volume_f32 = prepare_volume_for_tiff(volume)

    source_stem = strip_nii_suffix(
        nifti_path.name
    )

    output_paths: list[Path] = []

    for slice_number, voxel_index in enumerate(
        indices,
        start=1,
    ):
        slice_2d = extract_slice(
            volume=volume_f32,
            plane=plane,
            index=voxel_index,
        )

        # Preserve the original image geometry and make the slice compatible
        # with the square-input requirement of the 2D Imagix architecture.
        slice_2d = pad_slice_to_square(
            slice_2d=slice_2d,
            target_size=target_size_2d,
            pad_value=0.0,
        )

        slice_3ch = make_three_channel_float32(
            slice_2d
        )

        output_name = (
            f"{source_stem}_"
            f"{plane}_"
            f"sl_{slice_number:02d}.tiff"
        )

        output_path = (
            output_dir
            / output_name
        )

        tifffile.imwrite(
            output_path,
            slice_3ch,
            photometric="rgb",
        )

        output_paths.append(
            output_path
        )

    return output_paths


def find_source_stem_for_annotation_row(
    row: pd.Series,
    modality: str,
    available_stems: set[str],
) -> str:
    """
    Match one annotation row to the corresponding original NIfTI file.

    Matching order:
      1. Try the filename stored in the filepath column.
      2. Try the sample_id directly.
      3. Try modality_sampleid, e.g. cbf_564.
      4. Try a unique filename ending in _sampleid.
    """
    filepath_value = row.get(
        "filepath",
        None,
    )

    if (
        filepath_value is not None
        and not pd.isna(filepath_value)
    ):
        filepath_text = str(
            filepath_value
        ).strip()

        if filepath_text:
            # Handles both Windows and Unix path separators.
            filename = (
                filepath_text
                .replace("\\", "/")
                .split("/")[-1]
            )

            stem = strip_nii_suffix(
                filename
            )

            if stem in available_stems:
                return stem

    sid = sample_id_to_str(
        row.get(
            "sample_id",
            "",
        )
    )

    candidates = [
        sid,
        f"{modality}_{sid}",
    ]

    for candidate in candidates:
        if candidate in available_stems:
            return candidate

    # Final fallback:
    # search for a unique filename ending in _<sample_id>
    if sid:
        suffix_matches = [
            stem
            for stem in available_stems
            if stem.endswith(
                f"_{sid}"
            )
        ]

        if len(suffix_matches) == 1:
            return suffix_matches[0]

    raise KeyError(
        "Could not match annotation row "
        f"with sample_id={row.get('sample_id')} "
        f"and filepath={row.get('filepath')} "
        f"to a {modality} NIfTI file."
    )


def create_2d_annotation_file(
    input_csv: Path,
    output_csv: Path,
    modality: str,
    slice_paths_by_stem: dict[str, list[Path]],
) -> None:
    """
    Expand one patient-level annotation CSV to one row per 2D slice.

    All columns other than sample_id and filepath are copied unchanged.
    """
    df = pd.read_csv(
        input_csv
    )

    required_columns = {
        "sample_id",
        "filepath",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"{input_csv.name} is missing required columns: "
            f"{sorted(missing)}"
        )

    available_stems = set(
        slice_paths_by_stem.keys()
    )

    expanded_rows: list[pd.Series] = []

    for _, row in df.iterrows():
        source_stem = (
            find_source_stem_for_annotation_row(
                row=row,
                modality=modality,
                available_stems=available_stems,
            )
        )

        slice_paths = slice_paths_by_stem[source_stem]

        for slice_path in slice_paths:
            new_row = row.copy()

            # Every 2D slice becomes a separate sample.
            new_row["sample_id"] = (
                slice_path.stem
            )

            new_row["filepath"] = slice_path.name

            # All other columns remain unchanged.
            expanded_rows.append(
                new_row
            )

    expanded_df = pd.DataFrame(
        expanded_rows,
        columns=df.columns,
    )

    expanded_df.to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig",
    )

    expected_rows = (
        len(df)
        * slice_num
    )

    if len(expanded_df) != expected_rows:
        raise RuntimeError(
            f"Unexpected row count for {output_csv.name}: "
            f"expected {expected_rows}, "
            f"got {len(expanded_df)}."
        )

    print(
        f"  Created {output_csv.name}: "
        f"{len(df)} original rows -> "
        f"{len(expanded_df)} slice rows"
    )


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main() -> None:
    global slice_plane

    slice_plane = (
        slice_plane
        .lower()
        .strip()
    )

    if slice_plane not in VALID_PLANES:
        raise ValueError(
            f"slice_plane must be one of "
            f"{sorted(VALID_PLANES)}, "
            f"got {slice_plane!r}."
        )

    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(
            f"Source directory does not exist: "
            f"{SOURCE_ROOT}"
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Stores generated TIFF paths.
    # This ensures that annotation generation uses exactly the
    # slices that were actually created.
    all_slice_paths: dict[
        str,
        dict[str, list[Path]]
    ] = {}

    print("=" * 80)
    print(
        "3D -> 2D stroke image conversion"
    )
    print(
        f"Source root: {SOURCE_ROOT}"
    )
    print(
        f"Output root: {OUTPUT_ROOT}"
    )
    print(
        f"Plane      : {slice_plane}"
    )
    print(
        f"Slices     : {slice_num} per 3D volume"
    )
    print("=" * 80)

    # -------------------------------------------------------------------------
    # CONVERT NIFTI FILES
    # -------------------------------------------------------------------------
    for modality in MODALITIES:

        output_dir = (
            OUTPUT_ROOT
            / modality
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        nifti_files = sorted(
            SOURCE_ROOT.rglob(
                f"{modality}_*.nii.gz"
            )
        )

        nifti_files += sorted(
            SOURCE_ROOT.rglob(
                f"{modality}_*.nii"
            )
        )

        if not nifti_files:
            raise FileNotFoundError(
                f"No NIfTI files for modality "
                f"{modality!r} found below "
                f"{SOURCE_ROOT}"
            )

        print(
            f"\n[{modality}] "
            f"Converting {len(nifti_files)} volumes ..."
        )

        slice_paths_by_stem: dict[
            str,
            list[Path]
        ] = {}

        for file_number, nifti_path in enumerate(
            nifti_files,
            start=1,
        ):
            source_stem = strip_nii_suffix(
                nifti_path.name
            )

            if source_stem in slice_paths_by_stem:
                raise ValueError(
                    f"Duplicate source stem "
                    f"{source_stem!r} found below "
                    f"{SOURCE_ROOT}. "
                    "Each modality/patient combination "
                    "must occur only once."
                )

            output_paths = convert_nifti_file(
                nifti_path=nifti_path,
                output_dir=output_dir,
                plane=slice_plane,
                n_slices=slice_num,
            )

            slice_paths_by_stem[
                source_stem
            ] = output_paths

            print(
                f"  {file_number:>4}/"
                f"{len(nifti_files)}  "
                f"{nifti_path.name} "
                f"-> {len(output_paths)} TIFFs",
                end="\r",
            )

        print()

        all_slice_paths[
            modality
        ] = slice_paths_by_stem

    # -------------------------------------------------------------------------
    # CREATE NEW 2D ANNOTATION FILES
    # -------------------------------------------------------------------------
    print(
        "\n"
        + "=" * 80
    )

    print(
        "Creating 2D annotation CSV files"
    )

    print("=" * 80)

    for modality in MODALITIES:
        for suffix in ANNOTATION_SUFFIXES:

            input_csv = (
                STROKE_ROOT
                / f"{modality}_{suffix}.csv"
            )

            output_csv = (
                STROKE_ROOT
                / f"{modality}_2D_{suffix}.csv"
            )

            if not input_csv.exists():
                raise FileNotFoundError(
                    "Expected annotation file "
                    f"does not exist: {input_csv}"
                )

            create_2d_annotation_file(
                input_csv=input_csv,
                output_csv=output_csv,
                modality=modality,
                slice_paths_by_stem=(
                    all_slice_paths[
                        modality
                    ]
                ),
            )

    print(
        "\nFinished successfully."
    )


if __name__ == "__main__":
    main()