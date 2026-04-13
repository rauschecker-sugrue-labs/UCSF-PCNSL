# API Reference

## Import Pattern

**Important:** `AWSDataLoader` and `load_aws_*` functions are NOT in `__init__.py`. Import directly:

```python
from pcnsl_data_loader import (
    AWSDataLoader, load_aws_clinical_imaging_merged,
    load_aws_mutations, load_aws_dicom_headers,
    load_aws_dicom_geometry, load_aws_demographics,
    load_aws_biopsy_and_diagnosis_dates,
    load_all_summary_statistics, load_all_individual_lesions, load_all_radiomics,
    parse_dicom_tag_json, parse_dcm2niix_sidecar,
)
```

## Type Aliases

```python
StatisticsType = Literal["IndividualLesions", "SummaryLesions", "radiomics"]
Modality = Literal["FLAIR", "T1Post"]
ProcessingType = Literal["auto", "human", None]  # None for AWS dataset
ImageSpace = Literal["FLAIR", "T1Post"]
ClinicalDataType = Literal[
    "demographics", "biopsy_and_diagnosis_dates", "diagnosis_history",
    "medication_list_administered", "medication_list_ordered", "ucsf500_mutations",
]
```

## Convenience Functions

### AWS-based (anonymized dataset)

```python
load_aws_clinical_imaging_merged(
    clinical_types: list[ClinicalDataType] | None = None,  # default: demographics + biopsy_and_diagnosis_dates
    include_imaging_stats: bool = False,
    bids_path=DEFAULT_AWS_BIDS_PATH, csv_path=DEFAULT_AWS_CSV_PATH,
) -> pd.DataFrame  # 150 rows, merged clinical + imaging subject info

load_aws_demographics(
    bids_path=..., csv_path=...,
) -> pd.DataFrame  # 150 rows, demographics + subject mapping

load_aws_mutations(
    bids_path=..., csv_path=...,
) -> pd.DataFrame  # UCSF500 mutations filtered to 64 imaging subjects

load_aws_dicom_headers(
    subjects: list[str] | None = None,  # None = all 150
    session: str = "ses-0001",
    bids_path=..., csv_path=...,
) -> pd.DataFrame  # 600 rows (4 sequences x 150 subjects); TR/TE/TI in ms

load_aws_dicom_geometry(
    subjects: list[str] | None = None,
    session: str = "ses-0001",
    bids_path=...,
) -> pd.DataFrame  # 600 rows; Matrix, FOV, VoxelVolume, SliceGap in mm

load_aws_biopsy_and_diagnosis_dates(
    filter_to_imaging_subjects: bool = False,
    csv_path=...,
) -> pd.DataFrame  # BiopsyDate, DiagnosisDate, ImageDate, day differences
```

### PCNSLDataLoader-based (Box or local BIDS path)

```python
load_all_summary_statistics(
    modality: Modality = "FLAIR",
    processing: ProcessingType = "auto",
    box_path=DEFAULT_BOX_PATH,
) -> pd.DataFrame  # 150 rows per modality, 59+ columns

load_all_individual_lesions(modality=..., processing=..., box_path=...) -> pd.DataFrame
load_all_radiomics(modality=..., processing=..., box_path=...) -> pd.DataFrame
```

## AWSDataLoader Key Methods

```python
loader = AWSDataLoader(bids_path=..., csv_path=...)

# Clinical
loader.load_clinical_data(data_type: ClinicalDataType, filter_to_imaging_subjects=False) -> pd.DataFrame
loader.load_merged_data(
    clinical_types: list[ClinicalDataType] | None = None,
    include_imaging_stats=False, stats_type="SummaryLesions",
    modality="FLAIR", processing=None,
) -> pd.DataFrame
loader.get_patient_accession_mapping() -> pd.DataFrame  # patientdurablekey <-> subject
loader.list_available_clinical_data() -> list[str]

# DICOM
loader.load_dicom_headers(subjects=None, session="ses-0001") -> pd.DataFrame

# Imaging (delegated to PCNSLDataLoader)
loader.load_anatomy_image(subject, session="ses-0001", sequence="FLAIR") -> nib.Nifti1Image
loader.load_lesion_mask(subject, session="ses-0001", processing=None, modality="FLAIR") -> nib.Nifti1Image
loader.load_skullstripped_image(subject, session="ses-0001", processing=None, space="FLAIR", sequence="FLAIR") -> nib.Nifti1Image
loader.load_image_with_mask(subject, session="ses-0001", processing=None, modality="FLAIR") -> tuple[nib.Nifti1Image, nib.Nifti1Image]
```

## PCNSLDataLoader Key Methods

```python
loader = PCNSLDataLoader(box_path=...)

loader.list_subjects() -> list[str]
loader.list_sessions(subject) -> list[str]
loader.list_subjects_with_processing(processing="auto") -> list[str]

loader.load_anatomy_image(subject, session="ses-0001", sequence: Literal["T1w", "ce-gadolinium_T1w", "FLAIR"] = "FLAIR") -> nib.Nifti1Image
loader.load_anatomy_images(subject, session="ses-0001") -> dict[str, nib.Nifti1Image]

loader.load_statistics(
    subjects=None, sessions=None,
    stats_type: StatisticsType = "SummaryLesions",
    modality: Modality = "FLAIR",
    processing: ProcessingType = "auto",
    ignore_missing=True,
) -> pd.DataFrame

loader.load_skullstripped_image(subject, session="ses-0001", processing="auto", space="FLAIR", sequence: Literal["T1", "T1Post", "FLAIR", "ADC"] = "FLAIR") -> nib.Nifti1Image
loader.load_lesion_mask(subject, session="ses-0001", processing="auto", modality="FLAIR") -> nib.Nifti1Image
loader.load_image_with_mask(subject, session="ses-0001", processing="auto", modality="FLAIR") -> tuple[nib.Nifti1Image, nib.Nifti1Image]
```

## DICOM Parsers

```python
parse_dicom_tag_json(path) -> dict
# Raw DICOM tags -> friendly names. Derives: Matrix, FOV_row_mm, FOV_col_mm, VoxelVolume_mm3, NumSlices, SliceGap_mm
# Units: TR/TE/TI in ms, field strength in Tesla (auto-corrects GE Gauss)

parse_dcm2niix_sidecar(path) -> dict
# dcm2niix BIDS JSON -> flat dict. Units: TR/TE/TI in seconds. No pixel geometry.
```
