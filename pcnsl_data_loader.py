"""
PCNSL Data Loader Module

Utilities for loading CNS lymphoma MRI data from the UCSF PCNSL dataset
(pcnsl-dataset_v1.0 structure: pyalfe derivatives + clinical CSVs).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

import nibabel as nib
import numpy as np
import pandas as pd

# Default paths (relative to this file's location)
DEFAULT_DATASET_PATH = Path(__file__).resolve().parent / "pcnsl-dataset_v1.0"
DEFAULT_PYALFE_PATH = DEFAULT_DATASET_PATH / "pyalfe"
DEFAULT_CSV_PATH = DEFAULT_DATASET_PATH / "csvs_for_amazon_anonymized"

# Type aliases
StatisticsType = Literal["IndividualLesions", "SummaryLesions", "radiomics"]
Modality = Literal["FLAIR", "T1Post"]
ProcessingType = Literal["auto", "human", None]
ImageSpace = Literal["FLAIR", "T1Post"]
ClinicalDataType = Literal[
    "demographics",
    "biopsy_and_diagnosis_dates",
    "diagnosis_history",
    "medication_list_administered",
    "medication_list_ordered",
    "ucsf500_mutations",
]

# DICOM tag map: (XXXX,XXXX) -> friendly column name
DICOM_TAG_MAP: dict[str, str] = {
    "(0008,0070)": "Manufacturer",
    "(0008,1090)": "ManufacturerModelName",
    "(0008,1030)": "StudyDescription",
    "(0008,103E)": "SeriesDescription",
    "(0018,0023)": "MRAcquisitionType",
    "(0018,0050)": "SliceThickness",
    "(0018,0088)": "SpacingBetweenSlices",
    "(0018,0080)": "RepetitionTime",
    "(0018,0081)": "EchoTime",
    "(0018,0082)": "InversionTime",
    "(0018,0087)": "MagneticFieldStrength",
    "(0018,1314)": "FlipAngle",
    "(0028,0010)": "Rows",
    "(0028,0011)": "Columns",
    "(0028,0030)": "PixelSpacing",
}


# =============================================================================
# Private Helpers
# =============================================================================


def _load_json_series(
    pyalfe_base: Path,
    subjects: list[str],
    session: str,
    subdir: str,
    parser,
) -> pd.DataFrame:
    """Load JSON files from a per-subject derivatives subdirectory."""
    records = []
    for subject in subjects:
        json_dir = pyalfe_base / subject / session / subdir
        if not json_dir.exists():
            continue
        for json_path in sorted(json_dir.glob("*.json")):
            try:
                row = parser(json_path)
                row["subject"] = subject
                row["session"] = session
                parts = json_path.stem.split("_")
                row["sequence"] = "_".join(parts[2:])
                records.append(row)
            except Exception:
                continue
    return pd.DataFrame(records)


# =============================================================================
# Public Utilities
# =============================================================================


def parse_dicom_tag_json(path: str | Path) -> dict:
    """
    Parse a raw DICOM tag JSON file into a flat dictionary.

    Extracts tag values using DICOM_TAG_MAP and derives Matrix, FOV, NumSlices,
    and VoxelVolume from pixel geometry tags. MagneticFieldStrength >100 is
    converted from Gauss to Tesla.
    """
    with open(path) as f:
        d = json.load(f)

    tags = d.get("tags", {})
    result: dict = {}

    for tag, name in DICOM_TAG_MAP.items():
        entry = tags.get(tag)
        if entry is not None:
            result[name] = entry["value"]

    # GE Gauss → Tesla correction
    fs = result.get("MagneticFieldStrength")
    if fs is not None:
        try:
            fs_float = float(fs)
            if fs_float > 100:
                result["MagneticFieldStrength"] = round(fs_float / 10_000, 2)
        except (TypeError, ValueError):
            pass

    rows = result.get("Rows")
    cols = result.get("Columns")
    spacing = result.get("PixelSpacing")
    st_val = result.get("SliceThickness")

    if rows is not None and cols is not None:
        result["Matrix"] = f"{rows}x{cols}"

        if spacing is not None:
            ps = spacing if isinstance(spacing, list) else [spacing, spacing]
            result["PixelSpacing_mm"] = ps[0]
            result["FOV_row_mm"] = round(rows * ps[0], 1)
            result["FOV_col_mm"] = round(cols * ps[1], 1)
            if st_val is not None:
                try:
                    result["VoxelVolume_mm3"] = round(ps[0] * ps[1] * float(st_val), 4)
                except (TypeError, ValueError):
                    pass

    num_files = d.get("_consolidation_info", {}).get("num_files_in_series")
    if num_files is not None:
        result["NumSlices"] = num_files

    sbs = result.get("SpacingBetweenSlices")
    if sbs is not None and st_val is not None:
        try:
            result["SliceGap_mm"] = round(float(sbs) - float(st_val), 3)
        except (TypeError, ValueError):
            pass

    return result


def parse_dcm2niix_sidecar(path: str | Path) -> dict:
    """Parse a dcm2niix BIDS JSON sidecar into a flat dictionary."""
    with open(path) as f:
        return json.load(f)


# =============================================================================
# AWSDataLoader
# =============================================================================


class AWSDataLoader:
    """
    Unified loader for the UCSF PCNSL dataset (pyalfe derivatives + clinical CSVs).

    The dataset structure (pcnsl-dataset_v1.0):
        pyalfe/sub-XXXX/ses-YYYY/
            dcm2niix_sidecars/    # Acquisition parameters
            dicom_headers/        # Raw DICOM tag JSONs
            masks/lesions_seg_comp/
            skullstripped/lesions_{FLAIR,T1Post}_space/
            statistics/lesions_{SummaryLesions,IndividualLesions,radiomics}/
        csvs_for_amazon_anonymized/
            demographics.csv, ucsf500_mutations.csv, ...

    Can operate in imaging-only mode (csv_path=None) or full mode with clinical data.

    Example:
        >>> loader = AWSDataLoader()
        >>> df = loader.load_merged_data(
        ...     clinical_types=["demographics", "biopsy_and_diagnosis_dates"]
        ... )
    """

    CLINICAL_DATA_CONFIG = {
        "demographics": {
            "filename": "demographics.csv",
            "subject_key": "patientdurablekey",
            "has_accessions": False,
        },
        "biopsy_and_diagnosis_dates": {
            "filename": "biopsy_and_diagnosis_dates.csv",
            "subject_key": "patientdurablekey",
            "has_accessions": True,
        },
        "diagnosis_history": {
            "filename": "diagnosis_history.csv",
            "subject_key": "patientdurablekey",
            "has_accessions": False,
        },
        "medication_list_administered": {
            "filename": "medication_list_administered.csv",
            "subject_key": "patientdurablekey",
            "has_accessions": False,
        },
        "medication_list_ordered": {
            "filename": "medication_list_ordered.csv",
            "subject_key": "patientdurablekey",
            "has_accessions": False,
        },
        "ucsf500_mutations": {
            "filename": "ucsf500_mutations.csv",
            "subject_key": "patientdurablekey",
            "has_accessions": True,
        },
    }

    def __init__(
        self,
        pyalfe_path: str | Path = DEFAULT_PYALFE_PATH,
        csv_path: str | Path | None = DEFAULT_CSV_PATH,
    ):
        self.pyalfe_path = Path(pyalfe_path)
        if not self.pyalfe_path.exists():
            raise FileNotFoundError(f"pyalfe directory not found: {self.pyalfe_path}")

        if csv_path is not None:
            self.csv_path = Path(csv_path)
            if not self.csv_path.exists():
                raise FileNotFoundError(f"CSV directory not found: {self.csv_path}")
        else:
            self.csv_path = None

    def _derivatives_base(
        self, subject: str, session: str = "ses-0001", processing: ProcessingType = None
    ) -> Path:
        base = self.pyalfe_path / subject / session
        if processing is not None:
            base = base / processing
        return base

    # --- Subject/Session Discovery ---

    def list_subjects(self) -> list[str]:
        """List all subject IDs in sorted order."""
        subject_dirs = sorted(self.pyalfe_path.glob("sub-*"))
        return [d.name for d in subject_dirs if d.is_dir()]

    def list_imaging_subjects(self) -> list[str]:
        """List all subjects with imaging data."""
        return self.list_subjects()

    def list_sessions(self, subject: str) -> list[str]:
        """List all sessions for a given subject."""
        subject_path = self.pyalfe_path / subject
        if not subject_path.exists():
            raise FileNotFoundError(f"Subject not found: {subject}")
        session_dirs = sorted(subject_path.glob("ses-*"))
        return [d.name for d in session_dirs if d.is_dir()]

    def list_subjects_with_processing(
        self, processing: ProcessingType = None
    ) -> list[str]:
        """List subjects that have the specified processing type available."""
        subjects = []
        for subject_dir in sorted(self.pyalfe_path.glob("sub-*")):
            for session_dir in subject_dir.glob("ses-*"):
                if processing is None:
                    if (session_dir / "statistics").exists():
                        subjects.append(subject_dir.name)
                        break
                elif (session_dir / processing).exists():
                    subjects.append(subject_dir.name)
                    break
        return subjects

    def get_subject_session_list(self) -> pd.DataFrame:
        """Get DataFrame of all subject/session/accession combinations."""
        records = []
        for subject in self.list_imaging_subjects():
            for session in self.list_sessions(subject):
                accession = subject.replace("sub-", "")
                records.append({"subject": subject, "session": session, "accession": accession})
        return pd.DataFrame(records)

    # --- Statistics Loading ---

    def get_statistics_path(
        self,
        subject: str,
        session: str = "ses-0001",
        processing: ProcessingType = None,
    ) -> Path:
        """Get path to the statistics directory for a subject/session."""
        return self._derivatives_base(subject, session, processing) / "statistics"

    def load_statistics_single(
        self,
        subject: str,
        session: str = "ses-0001",
        stats_type: StatisticsType = "SummaryLesions",
        modality: Modality = "FLAIR",
        processing: ProcessingType = None,
    ) -> pd.DataFrame:
        """Load statistics for a single subject/session."""
        stats_path = self.get_statistics_path(subject, session, processing)
        subdir = f"lesions_{stats_type}"
        filename = f"{subject}_{session}_{modality}_{stats_type}.csv"
        filepath = stats_path / subdir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Statistics file not found: {filepath}")

        df = pd.read_csv(filepath)

        if stats_type == "SummaryLesions":
            if df.columns[0] == "Unnamed: 0":
                df = df.set_index("Unnamed: 0")
                df = df.T
                df.index = [0]

        df["subject"] = subject
        df["session"] = session
        df["modality"] = modality
        df["processing"] = processing

        return df

    def load_statistics(
        self,
        subjects: str | list[str] | None = None,
        sessions: str | list[str] | None = None,
        stats_type: StatisticsType = "SummaryLesions",
        modality: Modality = "FLAIR",
        processing: ProcessingType = None,
        ignore_missing: bool = True,
    ) -> pd.DataFrame:
        """
        Load statistics for one or more subjects.

        Args:
            subjects: Subject ID(s). If None, loads all available subjects.
            sessions: Session ID(s). If None, uses 'ses-0001' for all.
            stats_type: 'IndividualLesions', 'SummaryLesions', or 'radiomics'
            modality: 'FLAIR' or 'T1Post'
            processing: 'auto', 'human', or None
            ignore_missing: If True, skip missing files; if False, raise error
        """
        if subjects is None:
            subjects = self.list_subjects_with_processing(processing)
        elif isinstance(subjects, str):
            subjects = [subjects]

        if sessions is None:
            sessions = ["ses-0001"] * len(subjects)
        elif isinstance(sessions, str):
            sessions = [sessions] * len(subjects)

        dfs = []
        for subject, session in zip(subjects, sessions):
            try:
                df = self.load_statistics_single(
                    subject, session, stats_type, modality, processing
                )
                dfs.append(df)
            except FileNotFoundError as e:
                if ignore_missing:
                    continue
                raise e

        if not dfs:
            raise ValueError("No statistics files found for the given parameters")

        return pd.concat(dfs, ignore_index=True)

    # --- Skullstripped Image Loading ---

    def get_skullstripped_path(
        self,
        subject: str,
        session: str = "ses-0001",
        processing: ProcessingType = None,
        space: ImageSpace = "FLAIR",
    ) -> Path:
        """Get path to the skullstripped images directory."""
        return self._derivatives_base(subject, session, processing) / "skullstripped" / f"lesions_{space}_space"

    def list_skullstripped_images(
        self,
        subject: str,
        session: str = "ses-0001",
        processing: ProcessingType = None,
        space: ImageSpace = "FLAIR",
    ) -> list[Path]:
        """List all skullstripped NIfTI files for a subject in a given space."""
        path = self.get_skullstripped_path(subject, session, processing, space)
        if not path.exists():
            raise FileNotFoundError(f"Skullstripped directory not found: {path}")
        return sorted(path.glob("*.nii.gz"))

    def load_skullstripped_image(
        self,
        subject: str,
        session: str = "ses-0001",
        processing: ProcessingType = None,
        space: ImageSpace = "FLAIR",
        sequence: Literal["T1", "T1Post", "FLAIR", "ADC"] = "FLAIR",
    ) -> nib.Nifti1Image:
        """Load a specific skullstripped image by sequence and space."""
        path = self.get_skullstripped_path(subject, session, processing, space)
        pattern = f"*_{sequence}_to_{space}_skullstripped.nii.gz"
        matches = list(path.glob(pattern))

        if not matches:
            raise FileNotFoundError(
                f"Skullstripped image not found for {subject}/{session} "
                f"sequence={sequence} space={space}"
            )

        return nib.load(matches[0])

    def load_skullstripped_images(
        self,
        subject: str,
        session: str = "ses-0001",
        processing: ProcessingType = None,
        space: ImageSpace = "FLAIR",
    ) -> dict[str, nib.Nifti1Image]:
        """Load all skullstripped images, returning {sequence: image} dict."""
        images = {}
        for filepath in self.list_skullstripped_images(subject, session, processing, space):
            match = re.search(r"_([A-Za-z0-9]+)_to_", filepath.name)
            if match:
                sequence = match.group(1)
                images[sequence] = nib.load(filepath)
        return images

    # --- Lesion Mask Loading ---

    def get_masks_path(
        self,
        subject: str,
        session: str = "ses-0001",
        processing: ProcessingType = None,
    ) -> Path:
        """Get path to the lesion masks directory."""
        return self._derivatives_base(subject, session, processing) / "masks" / "lesions_seg_comp"

    def load_lesion_mask(
        self,
        subject: str,
        session: str = "ses-0001",
        processing: ProcessingType = None,
        modality: Modality = "FLAIR",
    ) -> nib.Nifti1Image:
        """Load the lesion segmentation mask for a subject/session."""
        path = self.get_masks_path(subject, session, processing)
        pattern = f"*_{modality}_abnormal_seg_comp.nii.gz"
        matches = list(path.glob(pattern))

        if not matches:
            raise FileNotFoundError(
                f"Lesion mask not found for {subject}/{session} modality={modality}"
            )

        return nib.load(matches[0])

    def load_image_with_mask(
        self,
        subject: str,
        session: str = "ses-0001",
        processing: ProcessingType = None,
        modality: Modality = "FLAIR",
    ) -> tuple[nib.Nifti1Image, nib.Nifti1Image]:
        """Load a skullstripped image with its corresponding lesion mask."""
        image = self.load_skullstripped_image(
            subject, session, processing, space=modality, sequence=modality
        )
        mask = self.load_lesion_mask(subject, session, processing, modality=modality)
        return image, mask

    # --- Clinical Data Loading ---

    def load_clinical_data(
        self,
        data_type: ClinicalDataType,
        filter_to_imaging_subjects: bool = False,
    ) -> pd.DataFrame:
        """
        Load a clinical CSV file.

        Args:
            data_type: Type of clinical data to load
            filter_to_imaging_subjects: If True, only include rows matching
                imaging subjects (requires accession column)
        """
        if self.csv_path is None:
            raise ValueError("csv_path not configured; clinical data loading requires csv_path.")

        if data_type not in self.CLINICAL_DATA_CONFIG:
            raise ValueError(
                f"Unknown data type: {data_type}. "
                f"Valid options: {list(self.CLINICAL_DATA_CONFIG.keys())}"
            )

        config = self.CLINICAL_DATA_CONFIG[data_type]
        filepath = self.csv_path / config["filename"]

        if not filepath.exists():
            raise FileNotFoundError(f"Clinical data file not found: {filepath}")

        df = pd.read_csv(filepath)

        if filter_to_imaging_subjects:
            mapping = self.get_patient_accession_mapping()
            imaging_patients = set(mapping["patientdurablekey"])
            df[config["subject_key"]] = (
                df[config["subject_key"]].astype(str).str.zfill(4)
            )
            df = df[df[config["subject_key"]].isin(imaging_patients)]

        return df

    def list_available_clinical_data(self) -> list[str]:
        """List clinical data types that have files present."""
        if self.csv_path is None:
            return []
        available = []
        for data_type, config in self.CLINICAL_DATA_CONFIG.items():
            if (self.csv_path / config["filename"]).exists():
                available.append(data_type)
        return available

    def get_patient_accession_mapping(self) -> pd.DataFrame:
        """
        Map patient IDs to imaging subjects.

        In the AWS dataset, patientdurablekey corresponds directly to the
        BIDS subject ID (patientdurablekey=1 -> sub-0001).
        """
        subjects = self.list_imaging_subjects()
        records = []
        for subject in subjects:
            subject_num = subject.replace("sub-", "")
            patient_key = str(int(subject_num)).zfill(4)
            records.append({"patientdurablekey": patient_key, "subject": subject})
        return pd.DataFrame(records)

    # --- Data Merging ---

    def load_merged_data(
        self,
        clinical_types: list[ClinicalDataType] | None = None,
        include_imaging_stats: bool = False,
        stats_type: StatisticsType = "SummaryLesions",
        modality: Modality = "FLAIR",
        processing: ProcessingType = None,
    ) -> pd.DataFrame:
        """
        Load and merge clinical data with imaging subject information.

        Args:
            clinical_types: Clinical data types to include.
                Defaults to demographics + biopsy_and_diagnosis_dates.
            include_imaging_stats: If True, also merge imaging statistics
            stats_type: Type of imaging statistics to include
            modality: Imaging modality for statistics
            processing: Processing type for statistics
        """
        if clinical_types is None:
            clinical_types = ["demographics", "biopsy_and_diagnosis_dates"]

        base_df = self.get_subject_session_list()
        mapping = self.get_patient_accession_mapping()
        merged = base_df.merge(mapping, on="subject", how="left")

        for data_type in clinical_types:
            config = self.CLINICAL_DATA_CONFIG[data_type]
            clinical_df = self.load_clinical_data(data_type)

            clinical_df[config["subject_key"]] = (
                clinical_df[config["subject_key"]].astype(str).str.zfill(4)
            )

            merged = merged.merge(
                clinical_df,
                left_on="patientdurablekey",
                right_on=config["subject_key"],
                how="left",
                suffixes=("", f"_{data_type}"),
            )

            dup_col = f"{config['subject_key']}_{data_type}"
            if dup_col in merged.columns:
                merged = merged.drop(columns=[dup_col])

        if include_imaging_stats:
            try:
                stats_df = self.load_statistics(
                    subjects=None,
                    stats_type=stats_type,
                    modality=modality,
                    processing=processing,
                    ignore_missing=True,
                )
                merged = merged.merge(
                    stats_df, on=["subject", "session"], how="left", suffixes=("", "_imaging")
                )
            except ValueError:
                pass

        return merged

    def load_demographics_with_imaging(self) -> pd.DataFrame:
        """Load demographics merged with imaging subjects."""
        return self.load_merged_data(clinical_types=["demographics"])

    def load_mutations_for_imaging_subjects(self) -> pd.DataFrame:
        """Load mutation data filtered to subjects with imaging."""
        return self.load_clinical_data("ucsf500_mutations", filter_to_imaging_subjects=True)

    # --- DICOM Header Loading ---

    def load_dicom_headers(
        self,
        subjects: list[str] | None = None,
        session: str = "ses-0001",
    ) -> pd.DataFrame:
        """
        Load DICOM header metadata from dcm2niix sidecars.

        Returns DataFrame with TR/TE/TI converted from seconds to milliseconds
        and MagneticFieldStrength corrected from Gauss to Tesla where needed.
        """
        if subjects is None:
            subjects = self.list_imaging_subjects()

        df = _load_json_series(self.pyalfe_path, subjects, session, "dcm2niix_sidecars", parse_dcm2niix_sidecar)

        if df.empty:
            return df

        for col in ("RepetitionTime", "EchoTime", "InversionTime"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce") * 1000

        if "MagneticFieldStrength" in df.columns:
            fs = pd.to_numeric(df["MagneticFieldStrength"], errors="coerce")
            df["MagneticFieldStrength"] = fs.where(fs <= 100, fs / 10_000)

        return df


# =============================================================================
# Module-Level Convenience Functions
# =============================================================================


def load_aws_demographics(
    pyalfe_path: str | Path = DEFAULT_PYALFE_PATH,
    csv_path: str | Path = DEFAULT_CSV_PATH,
) -> pd.DataFrame:
    """Load demographics for imaging subjects."""
    loader = AWSDataLoader(pyalfe_path, csv_path)
    return loader.load_demographics_with_imaging()


def load_aws_clinical_imaging_merged(
    clinical_types: list[ClinicalDataType] | None = None,
    include_imaging_stats: bool = False,
    pyalfe_path: str | Path = DEFAULT_PYALFE_PATH,
    csv_path: str | Path = DEFAULT_CSV_PATH,
) -> pd.DataFrame:
    """Load merged clinical and imaging data."""
    loader = AWSDataLoader(pyalfe_path, csv_path)
    return loader.load_merged_data(
        clinical_types=clinical_types, include_imaging_stats=include_imaging_stats
    )


def load_aws_mutations(
    pyalfe_path: str | Path = DEFAULT_PYALFE_PATH,
    csv_path: str | Path = DEFAULT_CSV_PATH,
) -> pd.DataFrame:
    """Load mutation data for imaging subjects."""
    loader = AWSDataLoader(pyalfe_path, csv_path)
    return loader.load_mutations_for_imaging_subjects()


def load_aws_dicom_headers(
    subjects: list[str] | None = None,
    session: str = "ses-0001",
    pyalfe_path: str | Path = DEFAULT_PYALFE_PATH,
    csv_path: str | Path = DEFAULT_CSV_PATH,
) -> pd.DataFrame:
    """Load DICOM header metadata from dcm2niix sidecars."""
    loader = AWSDataLoader(pyalfe_path, csv_path)
    return loader.load_dicom_headers(subjects=subjects, session=session)


def load_aws_dicom_geometry(
    subjects: list[str] | None = None,
    session: str = "ses-0001",
    pyalfe_path: str | Path = DEFAULT_PYALFE_PATH,
) -> pd.DataFrame:
    """
    Load voxel geometry from raw DICOM tag JSON files.

    Uses dicom_headers/ (not dcm2niix_sidecars/) because pixel geometry
    is absent from dcm2niix output.
    """
    pyalfe_base = Path(pyalfe_path)
    if subjects is None:
        subjects = sorted(d.name for d in pyalfe_base.glob("sub-*") if d.is_dir())

    df = _load_json_series(pyalfe_base, subjects, session, "dicom_headers", parse_dicom_tag_json)
    return df


def load_aws_biopsy_and_diagnosis_dates(
    filter_to_imaging_subjects: bool = False,
    csv_path: str | Path = DEFAULT_CSV_PATH,
    pyalfe_path: str | Path = DEFAULT_PYALFE_PATH,
) -> pd.DataFrame:
    """Load biopsy and diagnosis dates from the dataset."""
    filepath = Path(csv_path) / "biopsy_and_diagnosis_dates.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"Biopsy and diagnosis dates file not found: {filepath}")

    df = pd.read_csv(filepath)

    if filter_to_imaging_subjects:
        loader = AWSDataLoader(pyalfe_path=pyalfe_path, csv_path=csv_path)
        mapping = loader.get_patient_accession_mapping()
        imaging_patients = set(mapping["patientdurablekey"])
        df["patientdurablekey"] = df["patientdurablekey"].astype(str).str.zfill(4)
        df = df[df["patientdurablekey"].isin(imaging_patients)]

    return df
