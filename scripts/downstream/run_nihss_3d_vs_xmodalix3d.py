"""
IDE launcher for held-out NIHSS comparison:
standalone Imagix3D vs aligned XModalix3D embeddings.

Edit ONLY the USER CONFIGURATION section, then run this file.
Keep it in the same directory as:
    nihss_3d_vs_xmodalix3d_svr.py
    plot_nihss_3d_vs_xmodalix3d_results.py
    nihss_2d_vs_3d_nested_svr_multikernel_clinical_variants.py
"""

import os
import time
from argparse import Namespace
from datetime import datetime
from pathlib import Path

# ============================================================================
# USER CONFIG
# ============================================================================

base_path = Path(r"")
out_path = Path(r"")
os.chdir(out_path)

from nihss_3d_vs_xmodalix3d_svr import run_analysis
from plot_nihss_3d_vs_xmodalix3d_results import create_plots

# Standalone Imagix3D embeddings (same files used in the 2D-vs-3D analysis).
EMBEDDING_3D = {
    "NCCT": base_path / "",
    "CBF":  base_path / "",
}

# Aligned XModalix3D embeddings.
EMBEDDING_3D_ALIGNED = {
    "NCCT": base_path / "",
    "CBF":  base_path / "",
}

CLINICAL_FILE = out_path / "sample_characteristics.csv"
OUTPUT_DIR = out_path / "results_3d_vs_xmodalix3d"

# Optional: supplying the previous primary CSV lets this analysis immediately
# perform the planned two-hypothesis thesis-level Holm correction:
#   (1) 2D vs 3D
#   (2) 3D vs aligned XModalix3D
PREVIOUS_2D_VS_3D_PRIMARY_CSV = (
    out_path / "results_2d_vs_3d_test/primary_2d_vs_3d_comparison.csv"
)
# Set to None if you do not want the joint correction yet.
# PREVIOUS_2D_VS_3D_PRIMARY_CSV = None

INNER_FOLDS = 5
RANDOM_STATE = 42
N_JOBS = 16

CLINICAL_PREDICTORS = ["nihss_acute", "onset_to_ct"]
WILCOXON_ALTERNATIVE = "two-sided"
CREATE_FIGURES = True

# If XModalix3D uses fewer latent dimensions than standalone Imagix3D, also
# reduce standalone 3D to the aligned dimensionality inside the leakage-safe
# development pipeline and repeat the comparison.
RUN_DIMENSION_MATCHED_SENSITIVITY = False

# Same final SVR setup/grid as the held-out 2D-vs-3D run.
KERNEL = "linear"
C_VALUES = [
    0.0001,
    0.001,
    0.01,
    0.1,
    1.0,
    10.0
]
EPSILON_VALUES = [
    0.01,
    0.05,
    0.1,
    0.5,
    1.0,
    1.5,
    2.0,
    2.5,
    3.0
]
GAMMA_VALUES = [
    3e-5, 1e-4, 2e-4, 3e-4, 5e-4, 1e-3, 3e-3,
    1e-2, 3e-2, "scale",
]
DEGREE_VALUES = [3]
COEF0_VALUES = [0.0]

# ============================================================================
# RUN
# ============================================================================

args = Namespace(
    embedding_3d=[f"{m}={p}" for m, p in EMBEDDING_3D.items()],
    embedding_3d_aligned=[f"{m}={p}" for m, p in EMBEDDING_3D_ALIGNED.items()],
    clinical=CLINICAL_FILE,
    output_dir=OUTPUT_DIR,
    target_col="nihss_discharge",
    clinical_id_col="patients_id",
    clinical_split_col="custom_splits",
    clinical_predictor_cols=CLINICAL_PREDICTORS,
    inner_folds=INNER_FOLDS,
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS,
    wilcoxon_alternative=WILCOXON_ALTERNATIVE,
    run_dimension_matched_sensitivity=RUN_DIMENSION_MATCHED_SENSITIVITY,
    previous_2d_vs_3d_primary_csv=PREVIOUS_2D_VS_3D_PRIMARY_CSV,
    kernel=KERNEL,
    c_values=C_VALUES,
    epsilon_values=EPSILON_VALUES,
    gamma_values=GAMMA_VALUES,
    degree_values=DEGREE_VALUES,
    coef0_values=COEF0_VALUES,
)

start_datetime = datetime.now()
start_perf = time.perf_counter()
print(f"\nScript started: {start_datetime:%Y-%m-%d %H:%M:%S}")

results_dir = run_analysis(args)
if CREATE_FIGURES:
    create_plots(results_dir)

end_datetime = datetime.now()
elapsed = time.perf_counter() - start_perf
hours, rem = divmod(elapsed, 3600)
minutes, seconds = divmod(rem, 60)

print("\nFinished.")
print(f"Results directory: {results_dir}")
print(f"Script started: {start_datetime:%Y-%m-%d %H:%M:%S}")
print(f"Script ended:   {end_datetime:%Y-%m-%d %H:%M:%S}")
print(f"Total runtime:  {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")
