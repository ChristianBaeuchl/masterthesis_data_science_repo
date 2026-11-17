
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import ast
import os


wd = ""
os.chdir(wd)

# NCCT | Imagix3D
best_config_list = []
                    
# NCCT | Imagix
best_config_list = []

# CTA | Imagix3D
best_config_list = []

# CTA | Imagix
best_config_list = []

# CBF | Imagix3D
best_config_list = []

# CBF | Imagix
best_config_list = []

# CBV | Imagix3D
best_config_list = []

# CBV | Imagix
best_config_list = []

# Tmax | Imagix3D
best_config_list = []

# Tmax | Imagix
best_config_list = []


def best_configs_to_df(best_config_paths):
    """
    Read multiple best_config.txt files and combine their contents
    into a single pandas DataFrame.

    Parameters
    ----------
    best_config_paths : list[str | Path]
        Paths to best_config.txt files.

    Returns
    -------
    pd.DataFrame
        One row per config file and one column per dictionary entry.
        The column 'reconstruction_loss' is renamed to
        'valid_reconstruction_loss'.
    """

    rows = []

    for config_path in best_config_paths:
        config_path = Path(config_path)

        with config_path.open("r", encoding="utf-8") as f:
            config = ast.literal_eval(f.read())

        if not isinstance(config, dict):
            raise ValueError(f"{config_path} does not contain a dictionary.")

        # reconstruction_loss corresponds to validation reconstruction loss
        if "reconstruction_loss" in config:
            config["valid_recon_loss"] = config.pop("reconstruction_loss")

        # Useful provenance information
        config["source_file"] = config_path.name
        config["source_path"] = str(config_path)

        rows.append(config)
    
    rows = pd.DataFrame(rows)
    
    list = []
    for i in range(1,6): 
        list.append(f"run #{i}")
    rows["run_number"] = list

    return rows


def select_best_config(
    df,
    performance_col="downstream_performance",
    loss_col="valid_total_loss",
    performance_tolerance=0.01,
):
    """
    Select a configuration that retains near-best downstream
    performance while minimizing validation total loss.

    performance_tolerance=0.01 means:
    retain configs within 1% of the best downstream performance.
    """

    df = df.copy()
    
    # Remove invalid runs
    df = df[df[performance_col].notna() & df[loss_col].notna()]

    # Best observed downstream performance
    best_performance = df[performance_col].max()

    # Relative threshold
    threshold = best_performance * (1.0 - performance_tolerance)

    # Configurations whose downstream performance is effectively
    # competitive with the best configuration
    competitive = df[df[performance_col] >= threshold].copy()

    # Among them, choose the lowest reconstruction loss
    competitive = competitive.sort_values(
        by=[loss_col, performance_col], ascending=[True, False])
        
    return competitive, competitive.iloc[0]




#########
#########

def pareto_front(
    df,
    performance_col="downstream_performance",
    loss_col="valid_total_loss",
):
    """
    Higher performance is better.
    Lower loss is better.
    """

    df = df.copy()

    is_pareto = []

    for idx, row in df.iterrows():

        dominated = (
            (df[performance_col] >= row[performance_col])
            & (df[loss_col] <= row[loss_col])
            & (
                (df[performance_col] > row[performance_col])
                | (df[loss_col] < row[loss_col])
            )
        ).any()

        is_pareto.append(not dominated)

    df["pareto_optimal"] = is_pareto

    return df
    
###############################################################################

df = best_configs_to_df(best_config_list)
#df.to_csv('cbv_best_configs.csv')

competitive, selected = select_best_config(
    df,
    performance_tolerance=0.01,
)

results = pareto_front(df)
pareto = results[results["pareto_optimal"]].sort_values("downstream_performance", ascending=False)
#pareto.to_csv('cbv_pareto.csv')
non_pareto = results[~results["pareto_optimal"]]
#non_pareto.to_csv('cbv_non_pareto.csv')

print("Competitive configurations:")
print(competitive)

print("\nSelected configuration:")
print(selected)

print("\nPareto optimal:")
print(pareto)


###############################################################################
fig, ax = plt.subplots(figsize=(7, 5))

ax.scatter(
    non_pareto["valid_total_loss"],
    non_pareto["downstream_performance"],
    alpha=0.5,
    label="Dominated",
)

ax.scatter(
    pareto["valid_total_loss"],
    pareto["downstream_performance"],
    s=80,
    label="Pareto optimal",
)

# Add run-number annotations for dominated points
for _, row in non_pareto.iterrows():
    ax.annotate(
        row["run_number"],
        (
            row["valid_total_loss"],
            row["downstream_performance"],
        ),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=9,
    )

# Add run-number annotations for Pareto-optimal points
for _, row in pareto.iterrows():
    ax.annotate(
        row["run_number"],
        (
            row["valid_total_loss"],
            row["downstream_performance"],
        ),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=9,
    )

ax.set_xlabel("Validation total loss ↓")
ax.set_ylabel("Downstream performance ↑")
ax.set_title("HPO model selection | Tmax dataset")
ax.legend()

plt.show()

######################
fig.savefig(
    "hpo_model_selection.pdf",
    bbox_inches="tight",
)