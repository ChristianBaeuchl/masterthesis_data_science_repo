# Master Thesis Code Repository

This repository contains experiment, hyperparameter optimization, downstream-analysis, diagnostic, and visualization code used for the master's thesis *Implementation and Benchmarking of Three-Dimensional Variational Autoencoders for Analysis of Stroke Imaging*.

## Relation to AUTOENCODIX

The underlying **Imagix3D** and **X-modalix3D** implementations were developed as extensions of the AUTOENCODIX framework.

AUTOENCODIX repository:  
https://github.com/jan-forest/autoencodix_package

At the time of writing, the Imagix3D implementation is available on the AUTOENCODIX `dev` branch but has not yet been merged into `main`. X-modalix3D has not yet been merged into `dev` or `main`.

The present repository contains the experiment-specific code used around these implementations, including model execution, hyperparameter optimization, downstream analyses, diagnostics, benchmarking, data preparation, and report notebooks.

## Repository structure

```text
masterthesis_repository/
├── configs/                 Final model configurations used for the reported runs
├── scripts/
│   ├── training/            Scripts for running final Imagix, Imagix3D, and X-modalix3D models
│   ├── hpo/                 Hyperparameter-optimization and model-selection scripts
│   ├── embeddings/          Patient-level aggregation of learned representations
│   ├── downstream/          NIHSS prediction, statistical comparisons, plots, and diagnostics
│   ├── benchmark/           Imagix3D runtime / performance benchmarking
│   ├── data/                Data-preparation and annotation-generation utilities
│   └── notebooks/           Report notebook templates used for model evaluation
└── jobs/                    Job scripts used to execute the experiments
```

## Analysis workflow

The main computational workflow of the thesis can be summarized as:

```text
Stroke CT data
    ↓
Data preparation / annotation generation
    ↓
Imagix / Imagix3D / X-modalix3D model training
    ↓
Hyperparameter optimization and final-model selection
    ↓
Latent representation extraction / aggregation
    ↓
Downstream NIHSS prediction
    ↓
Statistical comparison, diagnostics, and visualization
```

The corresponding scripts are organized by these stages in the `scripts/` directory.

## Configurations

The `configs/` directory contains YAML configuration files documenting the final model settings for the reported Imagix, Imagix3D, and X-modalix3D experiments.

Some of these YAML files were reconstructed from the configuration records stored in the generated experiment reports / notebooks. They are therefore intended primarily as documentation of the final run settings and as a practical basis for rerunning the corresponding models with the appropriate AUTOENCODIX version and input data.

## Data availability

The clinical imaging and outcome data used in the thesis cannot be redistributed. The configuration files therefore contain paths referring to the original HPC data locations and are not directly executable without access to the corresponding datasets.

The repository does not contain patient imaging data, clinical outcome data, or patient-level derived representations where redistribution is restricted.

## Reproducibility scope

This repository is intended to make the computational workflow of the thesis **traceable and partially reproducible**, but does not provide a fully automated end-to-end reproduction from raw data.

Given access to the corresponding input data and a compatible AUTOENCODIX version, the repository provides the main code used for:

- final model training;
- hyperparameter optimization and shared-hyperparameter selection;
- construction of patient-level latent representations;
- downstream NIHSS prediction;
- statistical comparison of 2D, 3D, and cross-modally aligned representations;
- diagnostic analyses and generation of thesis figures.

Not every exploratory run, intermediate script version, or development-stage configuration is included. The repository focuses on the code that contributed directly to the analyses reported in the thesis.

## Software environment

The experiments were performed using Python and the AUTOENCODIX framework on HPC systems.

For exact reproduction, the software environment should ideally match the versions used during the thesis, in particular for Python, PyTorch, Syne Tune, scikit-learn, and AUTOENCODIX. Where available, the relevant package versions and AUTOENCODIX commit / branch information should be recorded alongside the repository.

## Notes on execution

Many scripts contain paths and cluster-specific settings originating from the HPC environment used during the thesis. These paths may need to be adapted before execution on another system.

The `jobs/` directory contains the SLURM job scripts used to execute the corresponding experiments. These files document the computational resources and commands used on the HPC systems, but they are not intended as portable launch scripts for arbitrary environments.
