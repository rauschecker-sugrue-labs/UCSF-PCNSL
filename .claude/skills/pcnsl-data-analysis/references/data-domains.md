# Data Domains Reference

## Clinical Data (6 CSV types)

All clinical CSVs use `patientdurablekey` (zero-padded string, e.g. `"0001"`) as patient identifier. In the anonymized dataset, `patientdurablekey` maps directly to BIDS subject number (`sub-0001` -> `"0001"`).

| CSV | Key Columns | Notes |
|-----|-------------|-------|
| `demographics.csv` | Sex, BirthDate, Deathdate, FirstRace, Ethnicity | 150 rows (one per patient) |
| `biopsy_and_diagnosis_dates.csv` | BiopsyDate, DiagnosisDate, ImageDate, BiopsyDateMinusImageDate, DiagnosisDateMinusImageDate | Has `accessions` column linking to BIDS subject ID |
| `diagnosis_history.csv` | DiagnosisName, StartDateKeyValue, EndDateKeyValue | Multiple rows per patient |
| `medication_list_administered.csv` | MedicationGenericName, MedicationTherapeuticClass, AdministrationDateKeyValue, MedicationRoute | ~228k rows total |
| `medication_list_ordered.csv` | MedicationGenericName, MedicationTherapeuticClass, OrderedDateKeyValue, MedicationRoute | ~120k rows total |
| `ucsf500_mutations.csv` | gene, alteration, hgvsp, hgvsc, varianttype, variantcategory, variantaf, tmbscore, microsatellitestatus | 64-subject subset with genomic data; has `accessions` column |

**Date gotcha:** BirthDate/Deathdate use `M/D/YY` format. `pd.to_datetime(..., format='%m/%d/%y')` parses 2-digit years as 2000s, so `1/2/47` becomes 2047 instead of 1947. Apply century correction: subtract 100 years if parsed date > current year.

## Genomic Data (UCSF500 panel)

Only 64 of 150 subjects have genomic data. Key analysis columns:
- **gene**: Gene symbol (e.g., MYD88, PIM1, CD79B)
- **hgvsp**: Protein-level HGVS notation (e.g., L265P)
- **varianttype**: SNV, CNV, or SV
- **variantaf**: Variant allele frequency (0-1)
- **tmbscore**: Tumor mutational burden (mutations/Mb)
- **microsatellitestatus**: MS-Stable or MS-High
- **oncotreecode** / **oncotreedx**: Cancer classification

## Lesion Statistics (SummaryLesions - 59 metrics)

Stored as vertical CSV (Label, Value columns). `load_statistics_single()` transposes automatically.

Key metric groups:
- **Volume**: `total_lesion_volume`, `largest_lesion_volume`, `average_lesion_volume`, `number_of_lesions` (all mm^3)
- **Tissue distribution**: `percentage_volume_in_{TissueClass}` where TissueClass = CSF, Cortical Gray Matter, White Matter, Deep Gray Matter, Brain Stem, Cerebellum
- **Anatomical regions**: `percentage_volume_in_{Region}` where Region = Frontal, Parietal, Occipital, Temporal, CorpusCallosum (+ subdivisions)
- **Signal intensities**: `relative_{sequence}_signal` (ratio to normal-appearing white matter) for T1, T1Post, FLAIR, ADC
- **ADC statistics**: mean, median, min, 5th/95th percentile ADC values
- **Enhancement**: post/pre contrast T1 ratio
- **Distance**: `average_dist_to_ventricles_(voxels)`, `minimum_dist_to_Ventricles_(voxels)`

## Radiomics (PyRadiomics - 54 features)

Also vertical CSV format. Feature categories:
- **Shape** (14): Elongation, Flatness, axis lengths, diameters, Sphericity, SurfaceArea, VoxelVolume
- **First-order** (18): 10th/90th percentiles, Energy, Entropy, Kurtosis, Mean, Median, Skewness, Variance, etc.
- **Diagnostics** (~22): PyRadiomics version, configuration, image hash metadata

## DICOM Metadata (two sidecar sources)

| Source | Directory | Parser | Best For |
|--------|-----------|--------|----------|
| dcm2niix sidecars | `dcm2niix_sidecars/` | `parse_dcm2niix_sidecar()` | Acquisition params: TR, TE, TI (seconds), field strength, manufacturer |
| Raw DICOM tags | `dicom_headers/` | `parse_dicom_tag_json()` | Pixel geometry: Matrix, FOV, PixelSpacing, SliceThickness, VoxelVolume |

**Unit conversions**: `load_aws_dicom_headers()` converts TR/TE/TI from BIDS seconds to milliseconds. GE scanners may report field strength in Gauss (>100); automatically corrected to Tesla (/10,000).

## BIDS Layout (AWS dataset)

```
sub-XXXX/ses-YYYY/anat/              # FLAIR, T1w, ce-gadolinium_T1w
sub-XXXX/ses-YYYY/dwi/               # DWI_ADC
derivatives/pyalfe/sub-XXXX/ses-YYYY/
  dicom_headers/                      # Raw DICOM tag JSONs
  dcm2niix_sidecars/                  # dcm2niix BIDS JSONs
  masks/lesions_seg_comp/             # Integer-labeled lesion masks
  skullstripped/lesions_{FLAIR|T1Post}_space/  # 4 sequences each
  statistics/lesions_{SummaryLesions|IndividualLesions|radiomics}/
```

No `{auto|human}` subdirectory in AWS dataset. Use `processing=None` with loader methods.
