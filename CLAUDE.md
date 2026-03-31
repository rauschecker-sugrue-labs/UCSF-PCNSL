# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research repository for the **UCSF Primary CNS Lymphoma (PCNSL) MRI Dataset** — a publicly available neuroimaging dataset on AWS S3 (`s3://ucsf-pcnsl`). The repo contains utilities for loading/analyzing this dataset and a Jupyter notebook generating publication figures.

## Setup

This project uses `uv` for dependency management (see `uv.lock`). To install:

```bash
uv sync                        # Install core dependencies
uv sync --extra dev            # Include Jupyter support
```

Alternatively with pip:
```bash
pip install boto3 nibabel nilearn pandas numpy matplotlib
pip install jupyter ipykernel  # for notebooks
```

To run notebooks:
```bash
jupyter notebook get-to-know-a-dataset-pcnsl.ipynb
```

There are no tests or linting configurations defined in this project.

## Architecture

### Two Data Loaders, One Backend

`pcnsl_data_loader.py` contains two loader classes with a composition relationship:

- **`PCNSLDataLoader`** — the core loader. Reads BIDS-structured data from a local filesystem path (originally `/working/rauschecker1/pcnsl/Box/`). All imaging I/O goes through this class.
- **`AWSDataLoader`** — wraps `PCNSLDataLoader` internally (`self._imaging_loader`) and adds clinical CSV handling. Uses two local paths pointing to the anonymized AWS-ready dataset (both hardcoded to `mromano`'s Box sync at `~/Library/CloudStorage/Box-Box/Research/pcnsl_radiomics/dataset_manuscript/`).

The public S3 bucket (`s3://ucsf-pcnsl`) uses no authentication (`UNSIGNED` config via botocore). The `AWSDataLoader` works against a local copy of the anonymized data, not directly against S3 at runtime.

### Data Model

**BIDS directory layout** (used by both loaders):
```
sub-XXXX/ses-YYYY/anat/          # T1w, ce-gadolinium_T1w, FLAIR
sub-XXXX/ses-YYYY/dwi/           # ADC maps (AWSDataLoader only)
derivatives/pyalfe/sub-XXXX/ses-YYYY/
  {auto|human}/statistics/       # CSVs: SummaryLesions, IndividualLesions, radiomics
  {auto|human}/masks/            # Lesion segmentation NIfTIs
  {auto|human}/skullstripped/    # Brain-extracted images
```

For the AWS dataset, the `{auto|human}` subdirectory level is absent — statistics live directly under `ses-YYYY/statistics/`. The AWS dataset also includes a `dicom_headers/` subdirectory under each `ses-YYYY/` with per-sequence JSON sidecar files (e.g. `sub-0001_ses-0001_FLAIR.json`) containing MRI acquisition parameters (TR, TE, field strength, manufacturer, etc.).

**Statistics file naming pattern**: `{subject}_{session}_{modality}_{stats_type}.csv`
- `modality`: `FLAIR` or `T1Post`
- `stats_type`: `SummaryLesions`, `IndividualLesions`, or `radiomics`

**Clinical CSV identity keys**: `patientdurablekey` is the patient-level identifier used across all clinical CSVs. BIDS subject IDs (e.g., `sub-0001`) correspond to `accessions` in some CSVs. One patient can have multiple accessions/sessions.

### Module-Level Convenience Functions

Three top-level functions (`load_all_summary_statistics`, `load_all_individual_lesions`, `load_all_radiomics`) batch-load data across all subjects using `PCNSLDataLoader` and return a concatenated DataFrame.

### Key Type Aliases

Defined at module top: `StatisticsType`, `Modality`, `ProcessingType`, `ImageSpace`, `ClinicalDataType` — used as `Literal` constraints throughout the API.
