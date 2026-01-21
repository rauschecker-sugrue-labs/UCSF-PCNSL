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

import re
from pathlib import Path
from typing import Literal

import nibabel as nib
import numpy as np
import pandas as pd

# Default base path for the Box directory
DEFAULT_BOX_PATH = Path("/working/rauschecker1/pcnsl/Box")

# Type aliases
StatisticsType = Literal["IndividualLesions", "SummaryLesions", "radiomics"]
Modality = Literal["FLAIR", "T1Post"]
ProcessingType = Literal["auto", "human"]
ImageSpace = Literal["FLAIR", "T1Post"]


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
        self,
        processing: ProcessingType = "auto"
    ) -> list[str]:
        """
        List subjects that have the specified processing type available.

        Args:
            processing: 'auto' or 'human' processing

        Returns:
            List of subject IDs with the specified processing
        """
        derivatives_path = self.box_path / "derivatives" / "pyalfe"
        subjects = []

        for subject_dir in sorted(derivatives_path.glob("sub-*")):
            for session_dir in subject_dir.glob("ses-*"):
                if (session_dir / processing).exists():
                    subjects.append(subject_dir.name)
                    break

        return subjects

    # =========================================================================
    # Anatomy Image Loading
    # =========================================================================

    def get_anatomy_path(
        self,
        subject: str,
        session: str = "ses-0001"
    ) -> Path:
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
        self,
        subject: str,
        session: str = "ses-0001"
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
        sequence: Literal["T1w", "ce-gadolinium_T1w", "FLAIR"] = "FLAIR"
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
        self,
        subject: str,
        session: str = "ses-0001"
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
            name = filepath.stem.replace('.nii', '')
            parts = name.split('_')
            sequence = '_'.join(parts[2:])  # Everything after subject and session
            images[sequence] = nib.load(filepath)

        return images

    # =========================================================================
    # Statistics Loading
    # =========================================================================

    def get_statistics_path(
        self,
        subject: str,
        session: str = "ses-0001",
        processing: ProcessingType = "auto"
    ) -> Path:
        """
        Get the path to the statistics directory for a subject/session.

        Args:
            subject: Subject ID (e.g., 'sub-0001')
            session: Session ID (default: 'ses-0001')
            processing: 'auto' or 'human' processing

        Returns:
            Path to the statistics directory
        """
        return (
            self.box_path / "derivatives" / "pyalfe" / subject / session /
            processing / "statistics"
        )

    def load_statistics_single(
        self,
        subject: str,
        session: str = "ses-0001",
        stats_type: StatisticsType = "SummaryLesions",
        modality: Modality = "FLAIR",
        processing: ProcessingType = "auto"
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
            if df.columns[0] == 'Unnamed: 0':
                df = df.set_index('Unnamed: 0')
                df = df.T
                df.index = [0]

        # Add subject/session identifiers
        df['subject'] = subject
        df['session'] = session
        df['modality'] = modality
        df['processing'] = processing

        return df

    def load_statistics(
        self,
        subjects: str | list[str] | None = None,
        sessions: str | list[str] | None = None,
        stats_type: StatisticsType = "SummaryLesions",
        modality: Modality = "FLAIR",
        processing: ProcessingType = "auto",
        ignore_missing: bool = True
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
        space: ImageSpace = "FLAIR"
    ) -> Path:
        """
        Get the path to the skullstripped images directory.

        Args:
            subject: Subject ID (e.g., 'sub-0001')
            session: Session ID (default: 'ses-0001')
            processing: 'auto' or 'human' processing
            space: Target space ('FLAIR' or 'T1Post')

        Returns:
            Path to the skullstripped directory
        """
        return (
            self.box_path / "derivatives" / "pyalfe" / subject / session /
            processing / "skullstripped" / f"lesions_{space}_space"
        )

    def list_skullstripped_images(
        self,
        subject: str,
        session: str = "ses-0001",
        processing: ProcessingType = "auto",
        space: ImageSpace = "FLAIR"
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
        sequence: Literal["T1", "T1Post", "FLAIR", "ADC"] = "FLAIR"
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
        space: ImageSpace = "FLAIR"
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
        for filepath in self.list_skullstripped_images(subject, session, processing, space):
            # Extract sequence from filename
            # Pattern: sub-XXX_ses-XXX_{sequence}_to_{space}_skullstripped.nii.gz
            match = re.search(r'_([A-Za-z0-9]+)_to_', filepath.name)
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
        processing: ProcessingType = "auto"
    ) -> Path:
        """
        Get the path to the masks directory.

        Args:
            subject: Subject ID (e.g., 'sub-0001')
            session: Session ID (default: 'ses-0001')
            processing: 'auto' or 'human' processing

        Returns:
            Path to the masks directory
        """
        return (
            self.box_path / "derivatives" / "pyalfe" / subject / session /
            processing / "masks" / "lesions_seg_comp"
        )

    def load_lesion_mask(
        self,
        subject: str,
        session: str = "ses-0001",
        processing: ProcessingType = "auto",
        modality: Modality = "FLAIR"
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
        modality: Modality = "FLAIR"
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
            subject, session, processing,
            space=modality, sequence=modality
        )
        mask = self.load_lesion_mask(
            subject, session, processing, modality=modality
        )

        return image, mask


# =============================================================================
# Convenience Functions
# =============================================================================

def load_all_summary_statistics(
    modality: Modality = "FLAIR",
    processing: ProcessingType = "auto",
    box_path: str | Path = DEFAULT_BOX_PATH
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
        processing=processing
    )


def load_all_individual_lesions(
    modality: Modality = "FLAIR",
    processing: ProcessingType = "auto",
    box_path: str | Path = DEFAULT_BOX_PATH
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
        processing=processing
    )


def load_all_radiomics(
    modality: Modality = "FLAIR",
    processing: ProcessingType = "auto",
    box_path: str | Path = DEFAULT_BOX_PATH
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
        subjects=None,
        stats_type="radiomics",
        modality=modality,
        processing=processing
    )
