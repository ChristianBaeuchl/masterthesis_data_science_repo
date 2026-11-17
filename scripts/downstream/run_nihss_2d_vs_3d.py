"""
IDE runner for the NIHSS 2D-vs-3D CV analysis.

Edit ONLY the USER CONFIGURATION section below, then press Run in Spyder.
No command-line arguments are required.

Keep this file in the same directory as:
    nihss_2d_vs_3d_nested_svr_modality_matched.py
    plot_nihss_2d_vs_3d_results_modality_matched.py
"""
import os
from argparse import Namespace
from pathlib import Path
from datetime import datetime
import time



# ============================================================================
# USER CONFIGURATION
# ============================================================================

base_path = Path()
out_path = Path()
os.chdir(out_path)

from nihss_2d_vs_3d_nested_svr_multikernel_clinical_variants import run_analysis
from plot_nihss_2d_vs_3d_results_modality_matched import create_plots

EMBEDDING_2D = {
    "NCCT": base_path / "",
    "CTA":  base_path / "",
    "CBF":  base_path / "",
    "CBV":  base_path / "",
    "TMAX": base_path / "",
}

EMBEDDING_3D = {
    "NCCT": base_path / "",
    "CTA":  base_path / "",
    "CBF":  base_path / "",
    "CBV":  base_path / "",
    "TMAX": base_path / "",
}

CLINICAL_FILE = out_path / "sample_characteristics.csv"

# Evaluation scope:
#   "all"  -> original nested-CV analysis over the complete common cohort.
#   "test" -> strict held-out analysis: VAE train+valid patients are used only
#             for SVR tuning/refitting; ONLY VAE test patients contribute to
#             predictions, metrics, statistical comparisons and figures.
DATA_SCOPE = "test"

# Keep all/test runs in separate directories to avoid accidental overwriting.
OUTPUT_DIR = out_path / f"results_2d_vs_3d_{DATA_SCOPE}"


OUTER_FOLDS = 10
INNER_FOLDS = 5
RANDOM_STATE = 42

N_JOBS = 16

# Leave as None for the intended sensitivity analysis. The script infers the
# PCA target separately for every modality from the corresponding 3D LatDim_* columns
PCA_COMPONENTS_OVERRIDE = None
RUN_PCA_SENSITIVITY = True

WILCOXON_ALTERNATIVE = "two-sided"
ALLOW_VAE_SPLIT_MISMATCH = False
CREATE_FIGURES = True

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
    3e-5,
    1e-4,
    2e-4,
    3e-4,
    5e-4,
    1e-3,
    3e-3,
    1e-2,
    3e-2,
    "scale",
]
#GAMMA_VALUES = ["scale", 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
DEGREE_VALUES = [3]
COEF0_VALUES = [0.0]

# ============================================================================
# RUN
# ============================================================================

args = Namespace(
    embedding_2d=[f"{modality}={path}" for modality, path in EMBEDDING_2D.items()],
    embedding_3d=[f"{modality}={path}" for modality, path in EMBEDDING_3D.items()],
    clinical=CLINICAL_FILE,
    output_dir=OUTPUT_DIR,
    target_col="nihss_discharge",
    clinical_id_col="patients_id",
    clinical_split_col="custom_splits",
    clinical_predictor_cols=["nihss_acute", "onset_to_ct"],
    data_scope=DATA_SCOPE,
    outer_folds=OUTER_FOLDS,
    inner_folds=INNER_FOLDS,
    random_state=RANDOM_STATE,
    pca_components=PCA_COMPONENTS_OVERRIDE,
    skip_pca_sensitivity=not RUN_PCA_SENSITIVITY,
    n_jobs=N_JOBS,
    wilcoxon_alternative=WILCOXON_ALTERNATIVE,
    allow_vae_split_mismatch=ALLOW_VAE_SPLIT_MISMATCH,
    kernel=KERNEL,
    c_values=C_VALUES,
    epsilon_values=EPSILON_VALUES,
    gamma_values=GAMMA_VALUES,
    degree_values=DEGREE_VALUES,
    coef0_values=COEF0_VALUES,
)

# Time tracking - START =======================================================
start_datetime = datetime.now()
start_perf = time.perf_counter()
print(f"\nScript started: {start_datetime:%Y-%m-%d %H:%M:%S}")
# ============================================================================

results_dir = run_analysis(args)

if CREATE_FIGURES:
    create_plots(results_dir)

print("\nFinished.")
print(f"Results directory: {results_dir}")

# Time tracking - END =========================================================
end_datetime = datetime.now()
elapsed_seconds = time.perf_counter() - start_perf

hours, remainder = divmod(elapsed_seconds, 3600)
minutes, seconds = divmod(remainder, 60)

print("\nFinished.")
print(f"Results directory: {results_dir}")
print(f"Script started: {start_datetime:%Y-%m-%d %H:%M:%S}")
print(f"Script ended:   {end_datetime:%Y-%m-%d %H:%M:%S}")
print(
    f"Total runtime:  "
    f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}"
)


