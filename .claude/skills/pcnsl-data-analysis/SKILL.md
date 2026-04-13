---
name: pcnsl-data-analysis
description: Use when analyzing the UCSF-PCNSL neuroimaging dataset (s3://ucsf-pcnsl), loading PCNSL clinical/imaging/genomic data, or working with pcnsl_data_loader.py. Covers 150-subject MRI cohort with lesion statistics, radiomics, demographics, UCSF500 mutations, and DICOM metadata.
---

# PCNSL Dataset Analysis

## Overview

The UCSF-PCNSL dataset contains MRI data for 150 PCNSL patients (4 sequences each: FLAIR, T1w, ce-gadolinium T1w, DWI ADC) plus clinical CSVs (demographics, mutations, medications, diagnoses, biopsy dates). All data loading goes through `pcnsl_data_loader.py`.

## Data Access Modes

1. **Convenience functions** (most common) -- return pandas DataFrames. Require local data paths.
2. **S3 direct access** -- public bucket `s3://ucsf-pcnsl`, no auth needed. For individual NIfTI/CSV files.
3. **Pre-aggregated CSV** -- `combined_lesion_data.csv` in repo root (150 subjects x 262 columns, both modalities).

## Function Router

| Research Domain | Function | Returns |
|---|---|---|
| Demographics + clinical merged | `load_aws_clinical_imaging_merged()` | 150 rows |
| Genomic mutations (64 subjects) | `load_aws_mutations()` | UCSF500 panel data |
| Scanner/acquisition params | `load_aws_dicom_headers()` | 600 rows (4 seq x 150) |
| Voxel geometry (FOV, matrix) | `load_aws_dicom_geometry()` | 600 rows |
| Demographics only | `load_aws_demographics()` | 150 rows |
| Biopsy/diagnosis timing | `load_aws_biopsy_and_diagnosis_dates()` | date columns |
| Lesion summary stats | `load_all_summary_statistics(processing=None)` | 150 rows/modality |
| Per-lesion measurements | `load_all_individual_lesions(processing=None)` | variable rows |
| Radiomics features | `load_all_radiomics(processing=None)` | 150 rows/modality |
| NIfTI images + masks | `AWSDataLoader.load_image_with_mask()` | (image, mask) tuple |
| Pre-aggregated all | `pd.read_csv("combined_lesion_data.csv")` | 150 x 262 |

## Key Parameters

- **Modality**: `"FLAIR"` or `"T1Post"`
- **StatisticsType**: `"SummaryLesions"`, `"IndividualLesions"`, `"radiomics"`
- **ClinicalDataType**: `"demographics"`, `"biopsy_and_diagnosis_dates"`, `"diagnosis_history"`, `"medication_list_administered"`, `"medication_list_ordered"`, `"ucsf500_mutations"`
- **processing**: Use `None` for AWS dataset (no auto/human subdirectory)

## Critical Gotchas

- **Import path**: `from pcnsl_data_loader import ...` -- `AWSDataLoader` and `load_aws_*` are NOT in `__init__.py`
- **Date century correction**: 2-digit year dates (M/D/YY) parse as 2000s. Subtract 100 years if > current year.
- **GE field strength**: Values >100 are Gauss, auto-converted to Tesla (/10,000) by loaders.
- **SummaryLesions CSV format**: Vertical (Label, Value columns). `load_statistics_single()` transposes automatically, but direct CSV reads from S3 need manual handling.
- **DICOM unit split**: `load_aws_dicom_headers()` returns TR/TE/TI in **milliseconds**. Raw dcm2niix sidecars use **seconds**.

## Request Classification & Skill Routing

**Before doing anything else, classify the user's request into one of these modes:**

| Mode | Trigger keywords / intent | Action |
|---|---|---|
| **Data loading only** | load, access, read, fetch, import, explore dataset, list subjects, DICOM headers, CSV columns | Handle entirely within this skill |
| **Statistical analysis** | test, compare, correlate, regress, predict, survival, significance, p-value, ANOVA, chi-square, effect size, power analysis, assumption check | Invoke the `statistical-analysis` skill (load data first if needed using this skill's guidance) |
| **Visualization only** | plot, figure, chart, heatmap, boxplot, violin, scatter, bar chart, panel figure, publication figure | Invoke the `scientific-visualization` skill (load data first if needed using this skill's guidance) |
| **Combined** | "compare and plot", "analyze and visualize", requests that clearly need both stats and figures | Load data with this skill, then invoke `statistical-analysis` and/or `scientific-visualization` as needed |

### Routing rules

1. **Statistics-only requests**: Load the relevant data using the Function Router below, then invoke the `statistical-analysis` skill with the loaded DataFrame and the user's analysis goal. Do not perform the statistical analysis yourself.
2. **Visualization-only requests**: Load the relevant data using the Function Router below, then invoke the `scientific-visualization` skill with the loaded DataFrame and the user's figure requirements. Do not create the figure yourself.
3. **Combined requests**: Load data first, then invoke the specialized skills in sequence — typically `statistical-analysis` first (to determine what to visualize), then `scientific-visualization`.
4. **Preferred plotting library**: When the visualization skill delegates back for library choice, use `seaborn` by default. Fall back to `matplotlib` only when seaborn lacks the needed plot type or fine-grained control is required.

## References

- See `references/api-reference.md` for full function signatures and type aliases
- See `references/analysis-patterns.md` for 9 self-contained code examples
- See `references/data-domains.md` for column descriptions, metric groups, and data quirks
- Data dictionaries: `data_dictionary_clinical.csv` and `data_dictionary_imaging.csv` in repo root
