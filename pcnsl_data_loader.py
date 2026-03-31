"""
PCNSL Data Loader Module

This module provides utilities for loading and working with CNS lymphoma MRI data
from the PCNSL Box directory structure.

Directory Structure:
    /working/rauschecker1/pcnsl/Box/
    ├── sub-XXXX/ses-YYYY/anat/           # Raw anatomy images
    └── derivatives/pyalfe/sub-XXXX/ses-YYYY/
        ├── {auto|human}/
        │   ├── statistics/               # Lesion statistics CSVs
        │   ├── skullstripped/            # Skull-stripped images
        │   └── masks/                    # Segmentation masks

Author: CNS Lymphoma Genetics Project
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

import nibabel as nib
import numpy as np
import pandas as pd

# Default base path for the Box directory
DEFAULT_BOX_PATH = Path("/working/rauschecker1/pcnsl/Box")

# Default paths for AWS anonymized dataset
DEFAULT_AWS_BIDS_PATH = Path(
    "/Users/mromano/Library/CloudStorage/Box-Box/Research/"
    "pcnsl_radiomics/dataset_manuscript/bids_dir_for_aws_anon"
)
DEFAULT_AWS_CSV_PATH = Path(
    "/Users/mromano/Library/CloudStorage/Box-Box/Research/"
    "pcnsl_radiomics/dataset_manuscript/csvs_for_amazon_anonymized"
)

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

# DICOM tag map: (XXXX,XXXX) -> friendly column name (DICOM keyword)
DICOM_TAG_MAP: dict[str, str] = {
    "(0008,0070)": "Manufacturer",
    "(0008,1090)": "ManufacturerModelName",
    "(0008,1030)": "StudyDescription",
    "(0008,103E)": "SeriesDescription",
    "(0018,0023)": "MRAcquisitionType",
    "(0018,0050)": "SliceThickness",
    "(0018,0080)": "RepetitionTime",
    "(0018,0081)": "EchoTime",
    "(0018,0082)": "InversionTime",
    "(0018,0087)": "MagneticFieldStrength",
    "(0018,1314)": "FlipAngle",
    "(0028,0010)": "Rows",
    "(0028,0011)": "Columns",
    "(0028,0030)": "PixelSpacing",
}


def parse_dicom_tag_json(path: str | Path) -> dict:
    """
    Parse a raw DICOM tag JSON file into a flat dictionary.

    Extracts tag values using DICOM_TAG_MAP and derives Matrix, FOV_row_mm,
    FOV_col_mm, and NumSlices from pixel geometry tags.

    Units after extraction:
      - RepetitionTime / EchoTime / InversionTime: milliseconds
        (raw DICOM stores in ms; dcm2niix BIDS sidecars divide by 1000 → seconds)
      - MagneticFieldStrength: Tesla
        (DICOM standard requires Tesla, but some older GE scanners incorrectly
        store Gauss; values > 100 are corrected by dividing by 10,000)
      - SliceThickness / PixelSpacing / FOV_*_mm: millimetres

    Args:
        path: Path to the DICOM tag JSON file.

    Returns:
        Flat dict with friendly key names, plus derived fields:
        - Matrix: "RowsxColumns" string (e.g. "512x512")
        - FOV_row_mm, FOV_col_mm: field of view in mm
        - NumSlices: from _consolidation_info.num_files_in_series
    """
    with open(path) as f:
        d = json.load(f)

    tags = d.get("tags", {})
    result: dict = {}

    for tag, name in DICOM_TAG_MAP.items():
        entry = tags.get(tag)
        if entry is not None:
            result[name] = entry["value"]

    # Some older GE scanners store MagneticFieldStrength in Gauss rather than
    # Tesla (DICOM standard).  1 T = 10,000 G, so values > 100 are in Gauss.
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

    if rows is not None and cols is not None:
        result["Matrix"] = f"{rows}x{cols}"

        if spacing is not None:
            ps = spacing if isinstance(spacing, list) else [spacing, spacing]
            result["FOV_row_mm"] = round(rows * ps[0], 1)
            result["FOV_col_mm"] = round(cols * ps[1], 1)

    num_files = d.get("_consolidation_info", {}).get("num_files_in_series")
    if num_files is not None:
        result["NumSlices"] = num_files

    return result


def parse_dcm2niix_sidecar(path: str | Path) -> dict:
    """
    Parse a dcm2niix BIDS JSON sidecar into a flat dictionary.

    Unlike parse_dicom_tag_json, this reads dcm2niix-processed values which
    follow BIDS unit conventions:
      - RepetitionTime / EchoTime / InversionTime: seconds
      - MagneticFieldStrength: Tesla
      - SliceThickness: millimetres

    Pixel geometry (Rows, Columns, PixelSpacing) is not present in dcm2niix
    output; use parse_dicom_tag_json for FOV and Matrix derivation.

    Args:
        path: Path to the dcm2niix JSON sidecar file.

    Returns:
        Dict of all fields as written by dcm2niix.
    """
    with open(path) as f:
        return json.load(f)


class PCNSLDataLoader:
    """
    A class for loading PCNSL neuroimaging data from the Box directory.

    Attributes:
        box_path: Path to the Box directory root

    Example:
        >>> loader = PCNSLDataLoader()
        >>> # Load anatomy images for a subject
        >>> images = loader.load_anatomy_images("sub-0001", "ses-0001")
        >>> # Load statistics for multiple subjects
        >>> stats = loader.load_statistics(
        ...     subjects=["sub-0001", "sub-0002"],
        ...     stats_type="SummaryLesions",
        ...     modality="FLAIR"
        ... )
    """

    def __init__(self, box_path: str | Path = DEFAULT_BOX_PATH):
        """
        Initialize the data loader.

        Args:
            box_path: Path to the Box directory root
        """
        self.box_path = Path(box_path)
        if not self.box_path.exists():
            raise FileNotFoundError(f"Box directory not found: {self.box_path}")

    # =========================================================================
    # Subject/Session Discovery
    # =========================================================================

    def list_subjects(self) -> list[str]:
        """
        List all available subjects in the dataset.

        Returns:
            Sorted list of subject IDs (e.g., ['sub-0001', 'sub-0002', ...])
        """
        subject_dirs = sorted(self.box_path.glob("sub-*"))
        return [d.name for d in subject_dirs if d.is_dir()]

    def list_sessions(self, subject: str) -> list[str]:
        """
        List all sessions for a given subject.

        Args:
            subject: Subject ID (e.g., 'sub-0001')

        Returns:
            Sorted list of session IDs (e.g., ['ses-0001'])
        """
        subject_path = self.box_path / subject
        if not subject_path.exists():
            raise FileNotFoundError(f"Subject not found: {subject}")

        session_dirs = sorted(subject_path.glob("ses-*"))
        return [d.name for d in session_dirs if d.is_dir()]

    def list_subjects_with_processing(
        self, processing: ProcessingType = "auto"
    ) -> list[str]:
        """
        List subjects that have the specified processing type available.

        Args:
            processing: 'auto', 'human', or None (no processing subdirectory)

        Returns:
            List of subject IDs with the specified processing
        """
        derivatives_path = self.box_path / "derivatives" / "pyalfe"
        subjects = []

        for subject_dir in sorted(derivatives_path.glob("sub-*")):
            for session_dir in subject_dir.glob("ses-*"):
                if processing is None:
                    # No processing subdir — check for derivatives directly
                    if (session_dir / "statistics").exists():
                        subjects.append(subject_dir.name)
                        break
                elif (session_dir / processing).exists():
                    subjects.append(subject_dir.name)
                    break

        return subjects

    # =========================================================================
    # Anatomy Image Loading
    # =========================================================================

    def get_anatomy_path(self, subject: str, session: str = "ses-0001") -> Path:
        """
        Get the path to the anatomy directory for a subject/session.

        Args:
            subject: Subject ID (e.g., 'sub-0001')
            session: Session ID (default: 'ses-0001')

        Returns:
            Path to the anatomy directory
        """
        return self.box_path / subject / session / "anat"

    def list_anatomy_images(
        self, subject: str, session: str = "ses-0001"
    ) -> list[Path]:
        """
        List all anatomy images for a subject/session.

        Args:
            subject: Subject ID (e.g., 'sub-0001')
            session: Session ID (default: 'ses-0001')

        Returns:
            List of paths to NIfTI files
        """
        anat_path = self.get_anatomy_path(subject, session)
        if not anat_path.exists():
            raise FileNotFoundError(f"Anatomy directory not found: {anat_path}")

        return sorted(anat_path.glob("*.nii.gz"))

    def load_anatomy_image(
        self,
        subject: str,
        session: str = "ses-0001",
        sequence: Literal["T1w", "ce-gadolinium_T1w", "FLAIR"] = "FLAIR",
    ) -> nib.Nifti1Image:
        """
        Load a specific anatomy image for a subject/session.

        Args:
            subject: Subject ID (e.g., 'sub-0001')
            session: Session ID (default: 'ses-0001')
            sequence: Image sequence type ('T1w', 'ce-gadolinium_T1w', or 'FLAIR')

        Returns:
            Loaded NIfTI image

        Example:
            >>> loader = PCNSLDataLoader()
            >>> flair = loader.load_anatomy_image("sub-0001", sequence="FLAIR")
            >>> t1_post = loader.load_anatomy_image("sub-0001", sequence="ce-gadolinium_T1w")
        """
        anat_path = self.get_anatomy_path(subject, session)
        filename = f"{subject}_{session}_{sequence}.nii.gz"
        filepath = anat_path / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Image not found: {filepath}")

        return nib.load(filepath)

    def load_anatomy_images(
        self, subject: str, session: str = "ses-0001"
    ) -> dict[str, nib.Nifti1Image]:
        """
        Load all anatomy images for a subject/session.

        Args:
            subject: Subject ID (e.g., 'sub-0001')
            session: Session ID (default: 'ses-0001')

        Returns:
            Dictionary mapping sequence name to loaded NIfTI image

        Example:
            >>> loader = PCNSLDataLoader()
            >>> images = loader.load_anatomy_images("sub-0001")
            >>> flair_data = images['FLAIR'].get_fdata()
        """
        images = {}
        for filepath in self.list_anatomy_images(subject, session):
            # Extract sequence name from filename
            # Pattern: sub-XXXX_ses-YYYY_SEQUENCE.nii.gz
            name = filepath.stem.replace(".nii", "")
            parts = name.split("_")
            sequence = "_".join(parts[2:])  # Everything after subject and session
            images[sequence] = nib.load(filepath)

        return images

    # =========================================================================
    # Statistics Loading
    # =========================================================================

    def get_statistics_path(
        self,
        subject: str,
        session: str = "ses-0001",
        processing: ProcessingType = "auto",
    ) -> Path:
        """
        Get the path to the statistics directory for a subject/session.

        Args:
            subject: Subject ID (e.g., 'sub-0001')
            session: Session ID (default: 'ses-0001')
            processing: 'auto', 'human', or None (no processing subdirectory)

        Returns:
            Path to the statistics directory
        """
        base = self.box_path / "derivatives" / "pyalfe" / subject / session
        if processing is not None:
            base = base / processing
        return base / "statistics"

    def load_statistics_single(
        self,
        subject: str,
        session: str = "ses-0001",
        stats_type: StatisticsType = "SummaryLesions",
        modality: Modality = "FLAIR",
        processing: ProcessingType = "auto",
    ) -> pd.DataFrame:
        """
        Load statistics for a single subject/session.

        Args:
            subject: Subject ID (e.g., 'sub-0001')
            session: Session ID (default: 'ses-0001')
            stats_type: Type of statistics ('IndividualLesions', 'SummaryLesions', 'radiomics')
            modality: Imaging modality ('FLAIR' or 'T1Post')
            processing: 'auto' or 'human' processing

        Returns:
            DataFrame containing the statistics

        Example:
            >>> loader = PCNSLDataLoader()
            >>> stats = loader.load_statistics_single(
            ...     "sub-0001",
            ...     stats_type="SummaryLesions",
            ...     modality="FLAIR"
            ... )
        """
        stats_path = self.get_statistics_path(subject, session, processing)
        subdir = f"lesions_{stats_type}"
        filename = f"{subject}_{session}_{modality}_{stats_type}.csv"
        filepath = stats_path / subdir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Statistics file not found: {filepath}")

        df = pd.read_csv(filepath)

        # For SummaryLesions, the data is transposed (metrics as rows)
        # Convert to a more usable format
        if stats_type == "SummaryLesions":
            # First column is metric name, second is value
            if df.columns[0] == "Unnamed: 0":
                df = df.set_index("Unnamed: 0")
                df = df.T
                df.index = [0]

        # Add subject/session identifiers
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
        processing: ProcessingType = "auto",
        ignore_missing: bool = True,
    ) -> pd.DataFrame:
        """
        Load statistics for one or more subjects.

        Args:
            subjects: Subject ID(s). If None, loads all available subjects.
            sessions: Session ID(s). If None, uses 'ses-0001' for all.
            stats_type: Type of statistics ('IndividualLesions', 'SummaryLesions', 'radiomics')
            modality: Imaging modality ('FLAIR' or 'T1Post')
            processing: 'auto' or 'human' processing
            ignore_missing: If True, skip missing files; if False, raise error

        Returns:
            Combined DataFrame with all statistics

        Example:
            >>> loader = PCNSLDataLoader()
            >>> # Load for specific subjects
            >>> stats = loader.load_statistics(
            ...     subjects=["sub-0001", "sub-0002", "sub-0003"],
            ...     stats_type="IndividualLesions",
            ...     modality="FLAIR"
            ... )
            >>> # Load all subjects
            >>> all_stats = loader.load_statistics(stats_type="SummaryLesions")
        """
        # Handle subject input
        if subjects is None:
            subjects = self.list_subjects_with_processing(processing)
        elif isinstance(subjects, str):
            subjects = [subjects]

        # Handle session input
        if sessions is None:
            sessions = ["ses-0001"] * len(subjects)
        elif isinstance(sessions, str):
            sessions = [sessions] * len(subjects)

        # Load and combine data
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

    # =========================================================================
    # Skullstripped Image Loading
    # =========================================================================

    def get_skullstripped_path(
        self,
        subject: str,
        session: str = "ses-0001",
        processing: ProcessingType = "auto",
        space: ImageSpace = "FLAIR",
    ) -> Path:
        """
        Get the path to the skullstripped images directory.

        Args:
            subject: Subject ID (e.g., 'sub-0001')
            session: Session ID (default: 'ses-0001')
            processing: 'auto', 'human', or None (no processing subdirectory)
            space: Target space ('FLAIR' or 'T1Post')

        Returns:
            Path to the skullstripped directory
        """
        base = self.box_path / "derivatives" / "pyalfe" / subject / session
        if processing is not None:
            base = base / processing
        return base / "skullstripped" / f"lesions_{space}_space"

    def list_skullstripped_images(
        self,
        subject: str,
        session: str = "ses-0001",
        processing: ProcessingType = "auto",
        space: ImageSpace = "FLAIR",
    ) -> list[Path]:
        """
        List all skullstripped images for a subject/session in a given space.

        Args:
            subject: Subject ID (e.g., 'sub-0001')
            session: Session ID (default: 'ses-0001')
            processing: 'auto' or 'human' processing
            space: Target space ('FLAIR' or 'T1Post')

        Returns:
            List of paths to skullstripped NIfTI files
        """
        path = self.get_skullstripped_path(subject, session, processing, space)
        if not path.exists():
            raise FileNotFoundError(f"Skullstripped directory not found: {path}")

        return sorted(path.glob("*.nii.gz"))

    def load_skullstripped_image(
        self,
        subject: str,
        session: str = "ses-0001",
        processing: ProcessingType = "auto",
        space: ImageSpace = "FLAIR",
        sequence: Literal["T1", "T1Post", "FLAIR", "ADC"] = "FLAIR",
    ) -> nib.Nifti1Image:
        """
        Load a specific skullstripped image.

        Args:
            subject: Subject ID (e.g., 'sub-0001')
            session: Session ID (default: 'ses-0001')
            processing: 'auto' or 'human' processing
            space: Target space ('FLAIR' or 'T1Post')
            sequence: Image sequence ('T1', 'T1Post', 'FLAIR', or 'ADC')

        Returns:
            Loaded NIfTI image

        Example:
            >>> loader = PCNSLDataLoader()
            >>> flair_img = loader.load_skullstripped_image(
            ...     "sub-0001",
            ...     space="FLAIR",
            ...     sequence="FLAIR"
            ... )
        """
        path = self.get_skullstripped_path(subject, session, processing, space)

        # Find the file matching the pattern
        # Pattern: sub-XXXXX_ses-XXXXX_{sequence}_to_{space}_skullstripped.nii.gz
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
        processing: ProcessingType = "auto",
        space: ImageSpace = "FLAIR",
    ) -> dict[str, nib.Nifti1Image]:
        """
        Load all skullstripped images for a subject/session in a given space.

        Args:
            subject: Subject ID (e.g., 'sub-0001')
            session: Session ID (default: 'ses-0001')
            processing: 'auto' or 'human' processing
            space: Target space ('FLAIR' or 'T1Post')

        Returns:
            Dictionary mapping sequence name to loaded NIfTI image
        """
        images = {}
        for filepath in self.list_skullstripped_images(
            subject, session, processing, space
        ):
            # Extract sequence from filename
            # Pattern: sub-XXX_ses-XXX_{sequence}_to_{space}_skullstripped.nii.gz
            match = re.search(r"_([A-Za-z0-9]+)_to_", filepath.name)
            if match:
                sequence = match.group(1)
                images[sequence] = nib.load(filepath)

        return images

    # =========================================================================
    # Lesion Mask Loading
    # =========================================================================

    def get_masks_path(
        self,
        subject: str,
        session: str = "ses-0001",
        processing: ProcessingType = "auto",
    ) -> Path:
        """
        Get the path to the masks directory.

        Args:
            subject: Subject ID (e.g., 'sub-0001')
            session: Session ID (default: 'ses-0001')
            processing: 'auto', 'human', or None (no processing subdirectory)

        Returns:
            Path to the masks directory
        """
        base = self.box_path / "derivatives" / "pyalfe" / subject / session
        if processing is not None:
            base = base / processing
        return base / "masks" / "lesions_seg_comp"

    def load_lesion_mask(
        self,
        subject: str,
        session: str = "ses-0001",
        processing: ProcessingType = "auto",
        modality: Modality = "FLAIR",
    ) -> nib.Nifti1Image:
        """
        Load the lesion segmentation mask for a subject/session.

        Args:
            subject: Subject ID (e.g., 'sub-0001')
            session: Session ID (default: 'ses-0001')
            processing: 'auto' or 'human' processing
            modality: Mask modality ('FLAIR' or 'T1Post')

        Returns:
            Loaded NIfTI mask image

        Example:
            >>> loader = PCNSLDataLoader()
            >>> mask = loader.load_lesion_mask("sub-0001", modality="FLAIR")
            >>> mask_data = mask.get_fdata()
            >>> print(f"Number of lesion voxels: {np.sum(mask_data > 0)}")
        """
        path = self.get_masks_path(subject, session, processing)

        # Find the file matching the pattern
        # Pattern: sub-XXX_ses-XXX_{modality}_abnormal_seg_comp.nii.gz
        pattern = f"*_{modality}_abnormal_seg_comp.nii.gz"
        matches = list(path.glob(pattern))

        if not matches:
            raise FileNotFoundError(
                f"Lesion mask not found for {subject}/{session} modality={modality}"
            )

        return nib.load(matches[0])

    # =========================================================================
    # Visualization Utilities
    # =========================================================================

    def load_image_with_mask(
        self,
        subject: str,
        session: str = "ses-0001",
        processing: ProcessingType = "auto",
        modality: Modality = "FLAIR",
    ) -> tuple[nib.Nifti1Image, nib.Nifti1Image]:
        """
        Load a skullstripped image along with its corresponding lesion mask.

        This convenience method loads both the image and mask in the same space,
        ready for overlay visualization.

        Args:
            subject: Subject ID (e.g., 'sub-0001')
            session: Session ID (default: 'ses-0001')
            processing: 'auto' or 'human' processing
            modality: Modality/space ('FLAIR' or 'T1Post')

        Returns:
            Tuple of (image, mask) as NIfTI images

        Example:
            >>> loader = PCNSLDataLoader()
            >>> img, mask = loader.load_image_with_mask("sub-0001", modality="FLAIR")
            >>> # Ready for visualization with nilearn
        """
        image = self.load_skullstripped_image(
            subject, session, processing, space=modality, sequence=modality
        )
        mask = self.load_lesion_mask(subject, session, processing, modality=modality)

        return image, mask


class AWSDataLoader:
    """
    A class for loading PCNSL data from the AWS anonymized dataset.

    This loader handles both the BIDS-formatted imaging data and the clinical
    CSV files, providing utilities to merge them based on subject/patient IDs.

    Directory Structure:
        bids_dir_for_aws_anon/
        ├── sub-XXXX/ses-YYYY/
        │   ├── anat/                    # Raw anatomy images (T1w, FLAIR, ce-gadolinium_T1w)
        │   └── dwi/                     # Diffusion images (ADC)
        └── derivatives/pyalfe/sub-XXXX/ses-YYYY/
            ├── masks/lesions_seg_comp/  # Lesion segmentation masks
            ├── skullstripped/           # Skull-stripped images in FLAIR/T1Post space
            └── statistics/              # Summary, individual, radiomics CSVs

        csvs_for_amazon_anonymized/
        ├── demographics.csv             # Patient demographics
        ├── biopsy_and_diagnosis_dates.csv  # Biopsy/diagnosis info
        ├── diagnosis_history.csv        # Diagnosis history
        ├── medication_list_administered.csv
        ├── medication_list_ordered.csv
        └── ucsf500_mutations.csv        # Genetic mutations

    Key Relationships:
        - BIDS subject ID (e.g., 'sub-0001') corresponds to 'accessions' in some CSVs
        - 'patientdurablekey' is the patient identifier used across CSV files
        - Multiple accessions can map to the same patient

    Attributes:
        bids_path: Path to the BIDS directory
        csv_path: Path to the CSV files directory

    Example:
        >>> loader = AWSDataLoader()
        >>> # Load demographics with imaging subjects
        >>> df = loader.load_clinical_data("demographics")
        >>> # Load and merge with subject mapping
        >>> merged = loader.load_merged_data(
        ...     clinical_types=["demographics", "biopsy_and_diagnosis_dates"]
        ... )
    """

    # Mapping of clinical data types to their filename and key columns
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
        bids_path: str | Path = DEFAULT_AWS_BIDS_PATH,
        csv_path: str | Path = DEFAULT_AWS_CSV_PATH,
    ):
        """
        Initialize the AWS data loader.

        Args:
            bids_path: Path to the BIDS directory
            csv_path: Path to the CSV files directory
        """
        self.bids_path = Path(bids_path)
        self.csv_path = Path(csv_path)

        if not self.bids_path.exists():
            raise FileNotFoundError(f"BIDS directory not found: {self.bids_path}")
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV directory not found: {self.csv_path}")

        # Create a PCNSLDataLoader pointing to the BIDS directory
        self._imaging_loader = PCNSLDataLoader(self.bids_path)

    # =========================================================================
    # Subject/Session Discovery
    # =========================================================================

    def list_imaging_subjects(self) -> list[str]:
        """
        List all subjects with imaging data.

        Returns:
            Sorted list of subject IDs (e.g., ['sub-0001', 'sub-0002', ...])
        """
        return self._imaging_loader.list_subjects()

    def list_sessions(self, subject: str) -> list[str]:
        """
        List all sessions for a given subject.

        Args:
            subject: Subject ID (e.g., 'sub-0001')

        Returns:
            Sorted list of session IDs
        """
        return self._imaging_loader.list_sessions(subject)

    def get_subject_session_list(self) -> pd.DataFrame:
        """
        Get a DataFrame of all subject/session combinations.

        Returns:
            DataFrame with 'subject', 'session', and 'accession' columns.
            'accession' is the numeric part of the subject ID for CSV matching.
        """
        records = []
        for subject in self.list_imaging_subjects():
            for session in self.list_sessions(subject):
                # Extract accession number (numeric part of subject ID)
                accession = subject.replace("sub-", "")
                records.append(
                    {
                        "subject": subject,
                        "session": session,
                        "accession": accession,
                    }
                )
        return pd.DataFrame(records)

    # =========================================================================
    # Clinical Data Loading
    # =========================================================================

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

        Returns:
            DataFrame containing the clinical data

        Example:
            >>> loader = AWSDataLoader()
            >>> demographics = loader.load_clinical_data("demographics")
            >>> biopsy_dates = loader.load_clinical_data(
            ...     "biopsy_and_diagnosis_dates",
            ...     filter_to_imaging_subjects=True
            ... )
        """
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
            # Get set of patientdurablekeys that have imaging data
            mapping = self.get_patient_accession_mapping()
            imaging_patients = set(mapping["patientdurablekey"])

            # Normalize patientdurablekey to zero-padded string for matching
            df[config["subject_key"]] = (
                df[config["subject_key"]].astype(str).str.zfill(4)
            )
            df = df[df[config["subject_key"]].isin(imaging_patients)]

        return df

    def list_available_clinical_data(self) -> list[str]:
        """
        List available clinical data files.

        Returns:
            List of clinical data type names that have files present
        """
        available = []
        for data_type, config in self.CLINICAL_DATA_CONFIG.items():
            filepath = self.csv_path / config["filename"]
            if filepath.exists():
                available.append(data_type)
        return available

    def get_patient_accession_mapping(self) -> pd.DataFrame:
        """
        Get the mapping between patient IDs and imaging subjects.

        In the AWS anonymized dataset, patientdurablekey directly corresponds
        to the BIDS subject ID (e.g., patientdurablekey=1 -> sub-0001).

        Returns:
            DataFrame with 'patientdurablekey', 'subject' columns
        """
        # Get all imaging subjects
        subjects = self.list_imaging_subjects()

        # Create mapping: subject ID number = patientdurablekey
        records = []
        for subject in subjects:
            # Extract number from subject ID (e.g., 'sub-0001' -> '0001')
            subject_num = subject.replace("sub-", "")
            # In this dataset, patientdurablekey = subject number (as int)
            patient_key = str(int(subject_num)).zfill(4)
            records.append(
                {
                    "patientdurablekey": patient_key,
                    "subject": subject,
                }
            )

        return pd.DataFrame(records)

    # =========================================================================
    # Data Merging
    # =========================================================================

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
            clinical_types: List of clinical data types to include.
                If None, includes demographics and biopsy_and_diagnosis_dates.
            include_imaging_stats: If True, also merge imaging statistics
            stats_type: Type of imaging statistics to include
            modality: Imaging modality for statistics
            processing: Processing type for statistics

        Returns:
            Merged DataFrame with clinical and imaging data

        Example:
            >>> loader = AWSDataLoader()
            >>> # Basic merge with demographics
            >>> df = loader.load_merged_data()
            >>> # Include imaging statistics
            >>> df = loader.load_merged_data(
            ...     clinical_types=["demographics", "biopsy_and_diagnosis_dates"],
            ...     include_imaging_stats=True,
            ...     stats_type="SummaryLesions"
            ... )
        """
        if clinical_types is None:
            clinical_types = ["demographics", "biopsy_and_diagnosis_dates"]

        # Start with subject/session list and patient mapping
        base_df = self.get_subject_session_list()
        mapping = self.get_patient_accession_mapping()

        # Merge to get patientdurablekey
        merged = base_df.merge(mapping, on="subject", how="left")

        # Merge each clinical data type
        for data_type in clinical_types:
            config = self.CLINICAL_DATA_CONFIG[data_type]
            clinical_df = self.load_clinical_data(data_type)

            # Normalize patientdurablekey to zero-padded string
            clinical_df[config["subject_key"]] = (
                clinical_df[config["subject_key"]].astype(str).str.zfill(4)
            )

            # Merge on patientdurablekey
            merged = merged.merge(
                clinical_df,
                left_on="patientdurablekey",
                right_on=config["subject_key"],
                how="left",
                suffixes=("", f"_{data_type}"),
            )

            # Remove duplicate patientdurablekey column if created
            dup_col = f"{config['subject_key']}_{data_type}"
            if dup_col in merged.columns:
                merged = merged.drop(columns=[dup_col])

        # Optionally include imaging statistics
        if include_imaging_stats:
            try:
                stats_df = self._imaging_loader.load_statistics(
                    subjects=None,
                    stats_type=stats_type,
                    modality=modality,
                    processing=processing,
                    ignore_missing=True,
                )
                merged = merged.merge(
                    stats_df,
                    on=["subject", "session"],
                    how="left",
                    suffixes=("", "_imaging"),
                )
            except ValueError:
                # No imaging statistics found
                pass

        return merged

    def load_demographics_with_imaging(self) -> pd.DataFrame:
        """
        Convenience method to load demographics merged with imaging subjects.

        Returns:
            DataFrame with demographics for subjects that have imaging data
        """
        return self.load_merged_data(clinical_types=["demographics"])

    def load_mutations_for_imaging_subjects(self) -> pd.DataFrame:
        """
        Load mutation data filtered to subjects with imaging.

        Returns:
            DataFrame with mutation data for imaging subjects
        """
        return self.load_clinical_data(
            "ucsf500_mutations",
            filter_to_imaging_subjects=True,
        )

    # =========================================================================
    # Imaging Data Access (delegated to PCNSLDataLoader)
    # =========================================================================

    def load_anatomy_image(
        self,
        subject: str,
        session: str = "ses-0001",
        sequence: Literal["T1w", "ce-gadolinium_T1w", "FLAIR"] = "FLAIR",
    ) -> nib.Nifti1Image:
        """Load an anatomy image. See PCNSLDataLoader.load_anatomy_image."""
        return self._imaging_loader.load_anatomy_image(subject, session, sequence)

    def load_lesion_mask(
        self,
        subject: str,
        session: str = "ses-0001",
        processing: ProcessingType = None,
        modality: Modality = "FLAIR",
    ) -> nib.Nifti1Image:
        """Load a lesion mask. See PCNSLDataLoader.load_lesion_mask."""
        return self._imaging_loader.load_lesion_mask(
            subject, session, processing, modality
        )

    def load_skullstripped_image(
        self,
        subject: str,
        session: str = "ses-0001",
        processing: ProcessingType = None,
        space: ImageSpace = "FLAIR",
        sequence: Literal["T1", "T1Post", "FLAIR", "ADC"] = "FLAIR",
    ) -> nib.Nifti1Image:
        """Load a skullstripped image. See PCNSLDataLoader.load_skullstripped_image."""
        return self._imaging_loader.load_skullstripped_image(
            subject, session, processing, space, sequence
        )

    def load_image_with_mask(
        self,
        subject: str,
        session: str = "ses-0001",
        processing: ProcessingType = None,
        modality: Modality = "FLAIR",
    ) -> tuple[nib.Nifti1Image, nib.Nifti1Image]:
        """Load image and mask. See PCNSLDataLoader.load_image_with_mask."""
        return self._imaging_loader.load_image_with_mask(
            subject, session, processing, modality
        )

    # =========================================================================
    # DICOM Header Loading
    # =========================================================================

    def load_dicom_headers(
        self,
        subjects: list[str] | None = None,
        session: str = "ses-0001",
    ) -> pd.DataFrame:
        """
        Load DICOM header metadata for all subjects.

        Reads JSON files from derivatives/pyalfe/sub-*/ses-*/dcm2niix_sidecars/
        and returns a DataFrame with one row per series per subject.

        Uses dcm2niix-processed values (BIDS unit conventions):
          - TR/TE/TI in seconds, MagneticFieldStrength in Tesla.
        For FOV/Matrix derivation use parse_dicom_tag_json against dicom_headers/.

        Args:
            subjects: Subject IDs. If None, loads all available.
            session: Session ID (default: 'ses-0001')

        Returns:
            DataFrame with subject, session, sequence, and dcm2niix sidecar
            fields. Units: TR/TE/TI in milliseconds, MagneticFieldStrength in T.
            (dcm2niix writes seconds; this method converts to ms for readability.
            Gauss values >100 are corrected to Tesla.)
        """
        if subjects is None:
            subjects = self.list_imaging_subjects()

        derivatives_base = self.bids_path / "derivatives" / "pyalfe"
        records = []

        for subject in subjects:
            dicom_dir = derivatives_base / subject / session / "dcm2niix_sidecars"
            if not dicom_dir.exists():
                continue
            for json_path in sorted(dicom_dir.glob("*.json")):
                try:
                    row = parse_dcm2niix_sidecar(json_path)
                    row["subject"] = subject
                    row["session"] = session
                    parts = json_path.stem.split("_")
                    row["sequence"] = "_".join(parts[2:])
                    records.append(row)
                except Exception:
                    continue

        df = pd.DataFrame(records)
        if df.empty:
            return df

        # Convert TR/TE/TI from BIDS seconds to milliseconds for clinical readability
        for col in ("RepetitionTime", "EchoTime", "InversionTime"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce") * 1000

        # Some older GE scanners write MagneticFieldStrength in Gauss (>100) in DICOM;
        # dcm2niix copies this value verbatim into the sidecar.  1 T = 10,000 G.
        if "MagneticFieldStrength" in df.columns:
            fs = pd.to_numeric(df["MagneticFieldStrength"], errors="coerce")
            df["MagneticFieldStrength"] = fs.where(fs <= 100, fs / 10_000)

        return df


# =============================================================================
# Convenience Functions
# =============================================================================


def load_all_summary_statistics(
    modality: Modality = "FLAIR",
    processing: ProcessingType = "auto",
    box_path: str | Path = DEFAULT_BOX_PATH,
) -> pd.DataFrame:
    """
    Load summary statistics for all available subjects.

    Args:
        modality: Imaging modality ('FLAIR' or 'T1Post')
        processing: 'auto' or 'human' processing
        box_path: Path to the Box directory

    Returns:
        DataFrame with summary statistics for all subjects

    Example:
        >>> df = load_all_summary_statistics(modality="FLAIR")
        >>> print(f"Loaded data for {len(df)} subjects")
    """
    loader = PCNSLDataLoader(box_path)
    return loader.load_statistics(
        subjects=None,  # Load all
        stats_type="SummaryLesions",
        modality=modality,
        processing=processing,
    )


def load_all_individual_lesions(
    modality: Modality = "FLAIR",
    processing: ProcessingType = "auto",
    box_path: str | Path = DEFAULT_BOX_PATH,
) -> pd.DataFrame:
    """
    Load individual lesion statistics for all available subjects.

    Args:
        modality: Imaging modality ('FLAIR' or 'T1Post')
        processing: 'auto' or 'human' processing
        box_path: Path to the Box directory

    Returns:
        DataFrame with individual lesion statistics for all subjects
    """
    loader = PCNSLDataLoader(box_path)
    return loader.load_statistics(
        subjects=None,
        stats_type="IndividualLesions",
        modality=modality,
        processing=processing,
    )


def load_all_radiomics(
    modality: Modality = "FLAIR",
    processing: ProcessingType = "auto",
    box_path: str | Path = DEFAULT_BOX_PATH,
) -> pd.DataFrame:
    """
    Load radiomics features for all available subjects.

    Args:
        modality: Imaging modality ('FLAIR' or 'T1Post')
        processing: 'auto' or 'human' processing
        box_path: Path to the Box directory

    Returns:
        DataFrame with radiomics features for all subjects
    """
    loader = PCNSLDataLoader(box_path)
    return loader.load_statistics(
        subjects=None, stats_type="radiomics", modality=modality, processing=processing
    )


# =============================================================================
# AWS Data Convenience Functions
# =============================================================================


def load_aws_demographics(
    bids_path: str | Path = DEFAULT_AWS_BIDS_PATH,
    csv_path: str | Path = DEFAULT_AWS_CSV_PATH,
) -> pd.DataFrame:
    """
    Load demographics for AWS anonymized imaging subjects.

    Args:
        bids_path: Path to the BIDS directory
        csv_path: Path to the CSV files directory

    Returns:
        DataFrame with demographics merged with imaging subject info

    Example:
        >>> df = load_aws_demographics()
        >>> print(f"Loaded demographics for {len(df)} imaging sessions")
    """
    loader = AWSDataLoader(bids_path, csv_path)
    return loader.load_demographics_with_imaging()


def load_aws_clinical_imaging_merged(
    clinical_types: list[ClinicalDataType] | None = None,
    include_imaging_stats: bool = False,
    bids_path: str | Path = DEFAULT_AWS_BIDS_PATH,
    csv_path: str | Path = DEFAULT_AWS_CSV_PATH,
) -> pd.DataFrame:
    """
    Load merged clinical and imaging data from AWS anonymized dataset.

    Args:
        clinical_types: List of clinical data types to include
        include_imaging_stats: If True, include imaging statistics
        bids_path: Path to the BIDS directory
        csv_path: Path to the CSV files directory

    Returns:
        Merged DataFrame with clinical and imaging data

    Example:
        >>> # Load demographics and biopsy dates
        >>> df = load_aws_clinical_imaging_merged()
        >>> # Include mutations and imaging stats
        >>> df = load_aws_clinical_imaging_merged(
        ...     clinical_types=["demographics", "ucsf500_mutations"],
        ...     include_imaging_stats=True
        ... )
    """
    loader = AWSDataLoader(bids_path, csv_path)
    return loader.load_merged_data(
        clinical_types=clinical_types,
        include_imaging_stats=include_imaging_stats,
    )


def load_aws_mutations(
    bids_path: str | Path = DEFAULT_AWS_BIDS_PATH,
    csv_path: str | Path = DEFAULT_AWS_CSV_PATH,
) -> pd.DataFrame:
    """
    Load mutation data for AWS anonymized imaging subjects.

    Args:
        bids_path: Path to the BIDS directory
        csv_path: Path to the CSV files directory

    Returns:
        DataFrame with mutation data filtered to imaging subjects
    """
    loader = AWSDataLoader(bids_path, csv_path)
    return loader.load_mutations_for_imaging_subjects()


def load_aws_dicom_headers(
    subjects: list[str] | None = None,
    session: str = "ses-0001",
    bids_path: str | Path = DEFAULT_AWS_BIDS_PATH,
    csv_path: str | Path = DEFAULT_AWS_CSV_PATH,
) -> pd.DataFrame:
    """
    Load DICOM header metadata from the AWS anonymized dataset.

    Args:
        subjects: Subject IDs. If None, loads all available.
        session: Session ID (default: 'ses-0001')
        bids_path: Path to the BIDS directory
        csv_path: Path to the CSV files directory

    Returns:
        DataFrame with subject, session, sequence, and dcm2niix sidecar fields.
        Units: TR/TE/TI in milliseconds, MagneticFieldStrength in T.

    Example:
        >>> df = load_aws_dicom_headers()
        >>> flair = df[df["sequence"] == "FLAIR"]
        >>> print(flair[["subject", "Manufacturer", "RepetitionTime", "InversionTime"]].head())
    """
    loader = AWSDataLoader(bids_path, csv_path)
    return loader.load_dicom_headers(subjects=subjects, session=session)


def load_aws_biopsy_and_diagnosis_dates(
    filter_to_imaging_subjects: bool = False,
    csv_path: str | Path = DEFAULT_AWS_CSV_PATH,
) -> pd.DataFrame:
    """
    Load biopsy and diagnosis dates from AWS anonymized dataset.

    Args:
        filter_to_imaging_subjects: If True, only include rows matching
            imaging subjects
        csv_path: Path to the CSV files directory

    Returns:
        DataFrame with biopsy and diagnosis date information

    Example:
        >>> df = load_aws_biopsy_and_diagnosis_dates()
        >>> print(df.columns.tolist())
    """
    filepath = Path(csv_path) / "biopsy_and_diagnosis_dates.csv"
    if not filepath.exists():
        raise FileNotFoundError(
            f"Biopsy and diagnosis dates file not found: {filepath}"
        )

    df = pd.read_csv(filepath)

    if filter_to_imaging_subjects:
        loader = AWSDataLoader(csv_path=csv_path)
        mapping = loader.get_patient_accession_mapping()
        imaging_patients = set(mapping["patientdurablekey"])
        df["patientdurablekey"] = df["patientdurablekey"].astype(str).str.zfill(4)
        df = df[df["patientdurablekey"].isin(imaging_patients)]

    return df
