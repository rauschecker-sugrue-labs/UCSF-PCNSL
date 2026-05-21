# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research repository for the **UCSF Primary CNS Lymphoma (PCNSL) MRI Dataset** — a publicly available neuroimaging dataset on AWS S3 (`s3://ucsf-pcnsl`). The repo contains:
- `pcnsl_data_loader.py` — data loading utilities for the pcnsl-dataset_v1.0 dataset (imaging + clinical)
- `figures_for_manuscript.ipynb` — generates all publication tables and figures
- `get-to-know-a-dataset-pcnsl.ipynb` — interactive tutorial for exploring the dataset
- Data dictionaries (`data_dictionary_clinical.csv`, `data_dictionary_imaging.csv`)
- `docs/superpowers/specs/2026-04-13-pcnsl-analysis-module-design.md` — design spec for the planned `pcnsl_analysis.py` module (not yet implemented)

The cohort is 150 subjects (64 UCSF500 genomic + 86 non-UCSF500), each with 4 MRI sequences (FLAIR, T1w, ce-gadolinium T1w, DWI ADC).

**Important:** The dataset contains only processed derivatives — raw MRI NIfTI files are not distributed.

## Setup

This project uses `uv` for dependency management (see `uv.lock`). To install:

```bash
uv sync                        # Install core dependencies
uv sync --extra dev            # Include Jupyter support
```

Key dependencies: `boto3`, `nibabel`, `nilearn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `great-tables`, `tqdm`, `python-pptx`, `lifelines`, `pygam`.

To run tests:
```bash
uv run python -m pytest tests/ -v
```

To run notebooks:
```bash
jupyter notebook figures_for_manuscript.ipynb
jupyter notebook get-to-know-a-dataset-pcnsl.ipynb
```

## Architecture

### Data Source: `pcnsl-dataset_v1.0/`

All code uses the local `pcnsl-dataset_v1.0/` directory as the single canonical data source:
```
pcnsl-dataset_v1.0/
├── pyalfe/                              # Imaging derivatives
│   └── sub-XXXX/ses-YYYY/
│       ├── dicom_headers/               # Raw DICOM tag JSONs
│       ├── dcm2niix_sidecars/           # dcm2niix BIDS JSONs (acquisition params)
│       ├── masks/lesions_seg_comp/      # Connected-component labeled masks
│       ├── skullstripped/
│       │   ├── lesions_FLAIR_space/     # 4 sequences registered to FLAIR
│       │   └── lesions_T1Post_space/    # 4 sequences registered to T1Post
│       └── statistics/
│           ├── lesions_SummaryLesions/  # 2 CSVs (FLAIR + T1Post)
│           ├── lesions_IndividualLesions/
│           └── lesions_radiomics/
└── csvs_for_amazon_anonymized/          # Clinical CSVs
    ├── demographics.csv
    ├── biopsy_and_diagnosis_dates.csv
    ├── diagnosis_history.csv
    ├── medication_list_administered.csv
    ├── medication_list_ordered.csv
    ├── ucsf500_mutations.csv
    ├── data_dictionary_clinical.csv
    └── data_dictionary_imaging.csv
```

### Single Loader: `AWSDataLoader`

`pcnsl_data_loader.py` (~740 lines) contains one unified class:

- **`AWSDataLoader`** — handles all data loading. Accepts two paths:
  - `pyalfe_path` — root of the pyalfe derivatives directory (default: `pcnsl-dataset_v1.0/pyalfe`)
  - `csv_path` — root of the clinical CSVs directory (default: `pcnsl-dataset_v1.0/csvs_for_amazon_anonymized`; pass `None` for imaging-only mode)

The public S3 bucket (`s3://ucsf-pcnsl`) uses no authentication (`UNSIGNED` config via botocore). The loader works against a local copy of the dataset, not directly against S3.

### DICOM Metadata Processing

Two standalone functions handle DICOM sidecar parsing:

- **`parse_dicom_tag_json()`** — parses raw DICOM tag JSON files (in `dicom_headers/`), extracts Matrix, FOV, and voxel geometry from pixel spacing/slice thickness. Uses `DICOM_TAG_MAP` (14 tags) for friendly names.
- **`parse_dcm2niix_sidecar()`** — parses dcm2niix BIDS JSON sidecars (in `dcm2niix_sidecars/`) for acquisition parameters (TR, TE, TI, field strength, manufacturer, model).

GE scanners report field strength in Gauss; values >100 are automatically converted to Tesla (÷10,000).

### Data Model

**Statistics file naming pattern**: `{subject}_{session}_{modality}_{stats_type}.csv`
- `modality`: `FLAIR` or `T1Post`
- `stats_type`: `SummaryLesions`, `IndividualLesions`, or `radiomics`

**Clinical CSV identity keys**: `patientdurablekey` is the patient-level identifier used across all clinical CSVs. BIDS subject IDs (e.g., `sub-0001`) correspond to `accessions` in some CSVs. One patient can have multiple accessions/sessions.

### Module-Level Convenience Functions

Six top-level functions batch-load data across all subjects and return concatenated DataFrames:

- `load_aws_demographics()` — patient demographics with imaging subject mapping
- `load_aws_clinical_imaging_merged()` — merged clinical + imaging data
- `load_aws_mutations()` — UCSF500 mutation records for imaging subjects
- `load_aws_dicom_headers()` — acquisition parameters from dcm2niix sidecars
- `load_aws_dicom_geometry()` — voxel geometry derived from raw DICOM tags
- `load_aws_biopsy_and_diagnosis_dates()` — biopsy/diagnosis timing relative to MRI

All accept optional `pyalfe_path=` and `csv_path=` kwargs (defaulting to the `pcnsl-dataset_v1.0` paths).

### Key Type Aliases

Defined at module top — used as `Literal` constraints throughout the API:
- `StatisticsType`: `SummaryLesions`, `IndividualLesions`, `radiomics`
- `Modality`: `FLAIR`, `T1Post`
- `ProcessingType`: `auto`, `human`
- `ImageSpace`: `FLAIR`, `T1Post`
- `ClinicalDataType`: `demographics`, `biopsy_and_diagnosis_dates`, `diagnosis_history`, `medication_list_administered`, `medication_list_ordered`, `ucsf500_mutations`

### Notebooks

**`figures_for_manuscript.ipynb`** — generates all publication outputs:
- `table1.html` — patient demographics and clinical characteristics
- `table2.html` — top 10 most commonly mutated genes
- `tableS2_acquisition_params.html` — MRI acquisition parameters by sequence
- `tableS3_geometry.html` — MRI voxel geometry by sequence
- `tableS1_scanner_info_per_subject.html` — per-subject scanner info
- `fig1_data_overview` (.png/.eps/.pdf) — multi-panel: directory structure, medication timeline, mutation lollipop
- `fig3_lesion_statistics` (.png/.eps) — lesion statistics across FLAIR/T1ce with bootstrapped CIs
- `figS1_selection_flowchart` (.png/.eps/.pdf) — CONSORT-style patient inclusion/exclusion
- `combined_lesion_data.csv` — aggregated lesion + radiomics data (150 subjects × 238 columns)

**`get-to-know-a-dataset-pcnsl.ipynb`** — interactive tutorial covering dataset structure, image loading, visualization, and DICOM header exploration.

### `__init__.py` Exports

Exports only `AWSDataLoader`. Convenience functions must be imported directly from `pcnsl_data_loader`.

### Planned: `pcnsl_analysis.py`

A design spec exists at `docs/superpowers/specs/2026-04-13-pcnsl-analysis-module-design.md` for a new analysis module. It is **not yet implemented**. When built, it will provide three classes:

- **`GAMModel`** — generalized additive model wrapper (`pyGAM`): linear, logistic, Poisson, Gamma families; spline terms; grid-search lambda selection; partial dependence plots.
- **`MutationImputer`** — predicts binary gene mutation status for non-UCSF500 subjects using imaging/clinical features; one `LogisticGAM` per gene with stratified CV and threshold selection.
- **`SurvivalModel`** — survival analysis suite (`lifelines`): Kaplan-Meier, Cox PH (standard + elastic net penalized), AFT models (Weibull/LogNormal/LogLogistic).

All classes accept plain DataFrames — no internal data loading. `lifelines` and `pygam` are already in `pyproject.toml` as core dependencies.
