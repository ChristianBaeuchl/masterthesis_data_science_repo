# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 20:35:34 2026

@author: chris
"""


from pathlib import Path
import pandas as pd


def load_hpo_results(paths):
    """Read and concatenate several Syne Tune results.csv files."""
    frames = [pd.read_csv(Path(path)) for path in paths]
    df = pd.concat(frames, ignore_index=True)

    # Only use completed trials if this column is present.
    if "st_status" in df.columns:
        df = df[df["st_status"] == "Completed"].copy()

    return df


def summarize_modality_shared_configs(
    paths,
    modality,
    performance_col="downstream_performance",
):
    df = load_hpo_results(paths)

    shared_cols = [
        "config_latent_dim",
        "config_hidden_dim",
    ]

    required = set(shared_cols + [performance_col])
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{modality}: missing required columns: {sorted(missing)}"
        )

    df = df[
        shared_cols + [performance_col]
    ].dropna().copy()

    summary = (
        df.groupby(shared_cols)[performance_col]
        .agg(
            best_performance="max",
            median_performance="median",
            mean_performance="mean",
            n_trials="count",
        )
        .reset_index()
    )

    summary["shared_config"] = list(
        zip(
            summary["config_latent_dim"].astype(int),
            summary["config_hidden_dim"].astype(int),
        )
    )
    summary["modality"] = modality

    return summary

def select_common_shared_config(
    results_paths_by_modality,
    performance_col="downstream_performance",
):
    summaries = []

    for modality, paths in results_paths_by_modality.items():
        summaries.append(
            summarize_modality_shared_configs(
                paths=paths,
                modality=modality,
                performance_col=performance_col,
            )
        )

    all_summary = pd.concat(
        summaries,
        ignore_index=True,
    )

    performance_matrix = all_summary.pivot(
        index="modality",
        columns="shared_config",
        values="best_performance",
    )

    # Only configurations that occurred for every modality
    common_configs = performance_matrix.columns[
        performance_matrix.notna().all(axis=0)
    ]

    if len(common_configs) == 0:
        raise ValueError(
            "No identical shared configuration was evaluated "
            "for every modality."
        )

    performance_matrix = performance_matrix[
        common_configs
    ]

    # Best attainable shared configuration for each modality
    best_per_modality = performance_matrix.max(axis=1)

    regret_matrix = performance_matrix.copy()

    for modality in performance_matrix.index:

        best = best_per_modality.loc[modality]

        if best == 0:
            raise ValueError(
                f"Best performance for {modality} is 0."
            )

        regret_matrix.loc[modality] = (
            best - performance_matrix.loc[modality]
        ) / abs(best)

    mean_regret = regret_matrix.mean(axis=0)

    rank_matrix = performance_matrix.rank(
        axis=1,
        ascending=False,
        method="average",
    )

    mean_rank = rank_matrix.mean(axis=0)

    ranking = pd.DataFrame(
        {
            "shared_config": mean_regret.index,
            "mean_relative_regret": mean_regret.values,
            "mean_rank": mean_rank.loc[
                mean_regret.index
            ].values,
        }
    )

    ranking = ranking.sort_values(["mean_relative_regret", "mean_rank"]).reset_index(drop=True)
    
    ranking[
        [
            "latent_dim",
            "hidden_dim",
        ]
    ] = pd.DataFrame(
        ranking["shared_config"].tolist(),
        index=ranking.index,
    )
    
    ranking = ranking[
        [
            "latent_dim",
            "hidden_dim",
            "mean_relative_regret",
            "mean_rank",
            "shared_config",
        ]
    ]

    selected_config = ranking.iloc[0]["shared_config"]

    return (
        selected_config,
        ranking,
        performance_matrix,
        regret_matrix,
        all_summary,
    )


if __name__ == "__main__":

    # ------------------------------------------------------------------
    # DEFINE PATHS
    # ------------------------------------------------------------------

    results_paths_by_modality = {
        "NCCT": [
        ],

        "CTA": [
        ],

        "CBF": [
        ],

        "CBV": [
        ],

        "TMAX": [
        ],
    }
    
    (
        selected_config,
        ranking,
        performance_matrix,
        regret_matrix,
        modality_summaries,
    ) = select_common_shared_config(results_paths_by_modality)
    
    # Unpack the selected common configuration
    (
        selected_latent_dim,
        selected_hidden_dim,
    ) = selected_config
    
    
    print(
        "\nBest downstream performance attained for each "
        "shared hyperparameter configuration:"
    )
    print(performance_matrix)
    
    print("\nRelative regret:")
    print(regret_matrix)
    
    print("\nCommon hyperparameter configuration ranking:")
    print(ranking)
    
    print("\nSelected common configuration:")
    print(f"  latent_dim:          {selected_latent_dim}")
    print(f"  hidden_dim:          {selected_hidden_dim}")
    
    
    # ------------------------------------------------------------------
    # Optional exports
    # ------------------------------------------------------------------
    
    output_dir = Path("common_config_selection")
    output_dir.mkdir(exist_ok=True)
    
    performance_matrix.to_csv(output_dir / "common_config_performance_matrix.csv") 
    regret_matrix.to_csv(output_dir / "common_config_relative_regret_matrix.csv")    
    ranking.to_csv(output_dir / "common_config_ranking.csv", index=False)   
    modality_summaries.to_csv(output_dir / "common_config_modality_summaries.csv", index=False)
    
    
    # Save the selected configuration separately
    selected_config_df = pd.DataFrame(
        {
            "parameter": [
                "latent_dim",
                "hidden_dim",
            ],
            "selected_value": [
                selected_latent_dim,
                selected_hidden_dim,
            ],
        }
    )
    selected_config_df.to_csv(output_dir / "selected_common_config.csv", index=False)
