# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research repository for the **UCSF Primary CNS Lymphoma (PCNSL) MRI Dataset** — a publicly available neuroimaging dataset on AWS S3 (`s3://ucsf-pcnsl`). The repo contains:
- `pcnsl_data_loader.py` — data loading utilities for both local BIDS data and the anonymized AWS dataset (imaging + clinical)
- `figures_for_manuscript.ipynb` — generates all publication tables and figures
- `get-to-know-a-dataset-pcnsl.ipynb` — interactive tutorial for exploring the dataset
- Data dictionaries (`data_dictionary_clinical.csv`, `data_dictionary_imaging.csv`)
- `docs/superpowers/specs/2026-04-13-pcnsl-analysis-module-design.md` — design spec for the planned `pcnsl_analysis.py` module (not yet implemented)

The cohort is 150 subjects (64 UCSF500 genomic + 86 non-UCSF500), each with 4 MRI sequences (FLAIR, T1w, ce-gadolinium T1w, DWI ADC).

**Important:** The S3 dataset contains only processed derivatives under `derivatives/pyalfe/` — raw MRI NIfTI files are not distributed.

## Setup

This project uses `uv` for dependency management (see `uv.lock`). To install:

```bash
uv sync                        # Install core dependencies
uv sync --extra dev            # Include Jupyter support
```

Key dependencies: `boto3`, `nibabel`, `nilearn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `great-tables`, `tqdm`, `python-pptx`, `lifelines`, `pygam`.

To run notebooks:
```bash
jupyter notebook figures_for_manuscript.ipynb
jupyter notebook get-to-know-a-dataset-pcnsl.ipynb
```

There are no tests or linting configurations defined in this project.

## Architecture

### Two Data Loaders, One Backend

`pcnsl_data_loader.py` (~1500 lines) contains two loader classes with a composition relationship:

- **`PCNSLDataLoader`** — the core loader. Reads BIDS-structured data from a local filesystem path (originally `/working/rauschecker1/pcnsl/Box/`). All imaging I/O goes through this class. Supports `list_subjects_with_processing()` to filter by processing type (auto/human/None).
- **`AWSDataLoader`** — wraps `PCNSLDataLoader` internally (`self._imaging_loader`) and adds clinical CSV handling plus DICOM metadata loading. Uses two local paths pointing to the anonymized AWS-ready dataset (both hardcoded to `mromano`'s Box sync at `~/Library/CloudStorage/Box-Box/Research/pcnsl_radiomics/dataset_manuscript/`).

The public S3 bucket (`s3://ucsf-pcnsl`) uses no authentication (`UNSIGNED` config via botocore). The `AWSDataLoader` works against a local copy of the anonymized data, not directly against S3 at runtime.

### DICOM Metadata Processing

Two standalone functions handle DICOM sidecar parsing:

- **`parse_dicom_tag_json()`** — parses raw DICOM tag JSON files (in `dicom_headers/`), extracts Matrix, FOV, and voxel geometry from pixel spacing/slice thickness. Uses `DICOM_TAG_MAP` (14 tags) for friendly names.
- **`parse_dcm2niix_sidecar()`** — parses dcm2niix BIDS JSON sidecars (in `dcm2niix_sidecars/`) for acquisition parameters (TR, TE, TI, field strength, manufacturer, model).

GE scanners report field strength in Gauss; values >100 are automatically converted to Tesla (÷10,000).

### Data Model

**BIDS directory layout** (used by both loaders):
```
sub-XXXX/ses-YYYY/anat/              # T1w, ce-gadolinium_T1w, FLAIR
sub-XXXX/ses-YYYY/dwi/               # ADC maps (AWSDataLoader only)
derivatives/pyalfe/sub-XXXX/ses-YYYY/
  {auto|human}/statistics/            # CSVs (PCNSLDataLoader)
  {auto|human}/masks/                 # Lesion segmentation NIfTIs
  {auto|human}/skullstripped/         # Brain-extracted images
```

For the AWS dataset, the `{auto|human}` subdirectory level is absent. The AWS derivatives layout under each `ses-YYYY/` is:
```
dicom_headers/                        # Raw DICOM tag JSONs (DCMQ prefix)
dcm2niix_sidecars/                    # dcm2niix BIDS JSONs (acquisition params)
masks/lesions_seg_comp/               # Connected-component labeled masks
skullstripped/
  lesions_FLAIR_space/                # 4 sequences registered to FLAIR
  lesions_T1Post_space/               # 4 sequences registered to T1Post
statistics/
  lesions_SummaryLesions/             # 2 CSVs (FLAIR + T1Post)
  lesions_IndividualLesions/          # 2 CSVs (FLAIR + T1Post)
  lesions_radiomics/                  # 2 CSVs (FLAIR + T1Post)
```

**Statistics file naming pattern**: `{subject}_{session}_{modality}_{stats_type}.csv`
- `modality`: `FLAIR` or `T1Post`
- `stats_type`: `SummaryLesions`, `IndividualLesions`, or `radiomics`

**Clinical CSV identity keys**: `patientdurablekey` is the patient-level identifier used across all clinical CSVs. BIDS subject IDs (e.g., `sub-0001`) correspond to `accessions` in some CSVs. One patient can have multiple accessions/sessions.

### Module-Level Convenience Functions

Nine top-level functions batch-load data across all subjects and return concatenated DataFrames:

**PCNSLDataLoader-based** (original Box path):
- `load_all_summary_statistics()`, `load_all_individual_lesions()`, `load_all_radiomics()`

**AWSDataLoader-based**:
- `load_aws_demographics()` — patient demographics with imaging subject mapping
- `load_aws_clinical_imaging_merged()` — merged clinical + imaging data
- `load_aws_mutations()` — UCSF500 mutation records for imaging subjects
- `load_aws_dicom_headers()` — acquisition parameters from dcm2niix sidecars
- `load_aws_dicom_geometry()` — voxel geometry derived from raw DICOM tags
- `load_aws_biopsy_and_diagnosis_dates()` — biopsy/diagnosis timing relative to MRI

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
- `table_acquisition_params.html` — MRI acquisition parameters by sequence
- `table_geometry.html` — MRI voxel geometry by sequence
- `table_scanner_info_per_subject.html` — per-subject scanner info
- `fig1_data_overview` (.png/.eps) — multi-panel: directory structure, medication timeline, mutation lollipop
- `fig2_lesion_comparison` (.png/.eps) — lesion statistics across FLAIR/T1ce with bootstrapped CIs
- `fig_selection_flowchart` (.png/.eps/.pdf) — CONSORT-style patient inclusion/exclusion
- `combined_lesion_data.csv` — aggregated lesion + radiomics data (150 subjects × 238 columns)

**`get-to-know-a-dataset-pcnsl.ipynb`** — interactive tutorial covering S3 access, image loading, visualization, and DICOM header exploration.

### Planned: `pcnsl_analysis.py`

A design spec exists at `docs/superpowers/specs/2026-04-13-pcnsl-analysis-module-design.md` for a new analysis module. It is **not yet implemented**. When built, it will provide three classes:

- **`GAMModel`** — generalized additive model wrapper (`pyGAM`): linear, logistic, Poisson, Gamma families; spline terms; grid-search lambda selection; partial dependence plots.
- **`MutationImputer`** — predicts binary gene mutation status for non-UCSF500 subjects using imaging/clinical features; one `LogisticGAM` per gene with stratified CV and threshold selection.
- **`SurvivalModel`** — survival analysis suite (`lifelines`): Kaplan-Meier, Cox PH (standard + elastic net penalized), AFT models (Weibull/LogNormal/LogLogistic).

All classes accept plain DataFrames — no internal data loading. `lifelines` and `pygam` are already in `pyproject.toml` as core dependencies.

### `__init__.py` Exports

Currently only exports `PCNSLDataLoader` and the three original convenience functions. `AWSDataLoader` and the AWS convenience functions are **not** exported — they must be imported directly from `pcnsl_data_loader`.
