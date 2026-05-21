"""
Tests for pcnsl_data_loader.py

Covers:
- parse_dicom_tag_json / parse_dcm2niix_sidecar standalone functions
- AWSDataLoader: subject discovery, statistics, masks, skullstripped,
  clinical data loading, merging, DICOM headers, geometry
- Module-level convenience functions
"""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest


# =============================================================================
# parse_dicom_tag_json
# =============================================================================


class TestParseDicomTagJson:
    def test_basic_extraction(self, tmp_path):
        from pcnsl_data_loader import parse_dicom_tag_json

        data = {
            "tags": {
                "(0008,0070)": {"value": "GE"},
                "(0018,0050)": {"value": 5.0},
                "(0018,0087)": {"value": 3.0},
                "(0028,0010)": {"value": 256},
                "(0028,0011)": {"value": 256},
                "(0028,0030)": {"value": [1.0, 1.0]},
            },
            "_consolidation_info": {"num_files_in_series": 40},
        }
        path = tmp_path / "dicom.json"
        path.write_text(json.dumps(data))

        result = parse_dicom_tag_json(path)
        assert result["Manufacturer"] == "GE"
        assert result["MagneticFieldStrength"] == 3.0
        assert result["Matrix"] == "256x256"
        assert result["FOV_row_mm"] == 256.0
        assert result["FOV_col_mm"] == 256.0
        assert result["NumSlices"] == 40
        assert result["VoxelVolume_mm3"] == 5.0

    def test_ge_gauss_conversion(self, tmp_path):
        from pcnsl_data_loader import parse_dicom_tag_json

        data = {
            "tags": {
                "(0018,0087)": {"value": 15000},
            },
        }
        path = tmp_path / "dicom_ge.json"
        path.write_text(json.dumps(data))

        result = parse_dicom_tag_json(path)
        assert result["MagneticFieldStrength"] == 1.5

    def test_slice_gap_calculation(self, tmp_path):
        from pcnsl_data_loader import parse_dicom_tag_json

        data = {
            "tags": {
                "(0018,0050)": {"value": 5.0},
                "(0018,0088)": {"value": 6.5},
            },
        }
        path = tmp_path / "dicom_gap.json"
        path.write_text(json.dumps(data))

        result = parse_dicom_tag_json(path)
        assert result["SliceGap_mm"] == 1.5

    def test_missing_tags_handled_gracefully(self, tmp_path):
        from pcnsl_data_loader import parse_dicom_tag_json

        data = {"tags": {}}
        path = tmp_path / "empty.json"
        path.write_text(json.dumps(data))

        result = parse_dicom_tag_json(path)
        assert "Matrix" not in result
        assert "FOV_row_mm" not in result
        assert "NumSlices" not in result

    def test_pixel_spacing_scalar(self, tmp_path):
        from pcnsl_data_loader import parse_dicom_tag_json

        data = {
            "tags": {
                "(0028,0010)": {"value": 512},
                "(0028,0011)": {"value": 512},
                "(0028,0030)": {"value": 0.5},
                "(0018,0050)": {"value": 3.0},
            },
        }
        path = tmp_path / "scalar_ps.json"
        path.write_text(json.dumps(data))

        result = parse_dicom_tag_json(path)
        assert result["Matrix"] == "512x512"
        assert result["FOV_row_mm"] == 256.0
        assert result["PixelSpacing_mm"] == 0.5
        assert result["VoxelVolume_mm3"] == pytest.approx(0.75, rel=1e-3)


class TestParseDcm2niixSidecar:
    def test_returns_full_dict(self, tmp_path):
        from pcnsl_data_loader import parse_dcm2niix_sidecar

        data = {"RepetitionTime": 9.0, "EchoTime": 0.081, "Manufacturer": "Siemens"}
        path = tmp_path / "sidecar.json"
        path.write_text(json.dumps(data))

        result = parse_dcm2niix_sidecar(path)
        assert result == data


# =============================================================================
# AWSDataLoader - Init and Subject Discovery
# =============================================================================


class TestAWSDataLoaderInit:
    def test_nonexistent_pyalfe_path_raises(self, tmp_path):
        from pcnsl_data_loader import AWSDataLoader

        with pytest.raises(FileNotFoundError):
            AWSDataLoader(pyalfe_path=tmp_path / "nonexistent", csv_path=None)

    def test_init_success(self, aws_loader):
        assert aws_loader.pyalfe_path.exists()
        assert aws_loader.csv_path.exists()

    def test_init_missing_csv(self, tmp_pyalfe_dir, tmp_path):
        from pcnsl_data_loader import AWSDataLoader

        with pytest.raises(FileNotFoundError):
            AWSDataLoader(pyalfe_path=tmp_pyalfe_dir, csv_path=tmp_path / "no")

    def test_init_csv_none_is_valid(self, tmp_pyalfe_dir):
        from pcnsl_data_loader import AWSDataLoader

        loader = AWSDataLoader(pyalfe_path=tmp_pyalfe_dir, csv_path=None)
        assert loader.csv_path is None


class TestSubjectDiscovery:
    def test_list_subjects(self, pcnsl_loader):
        subjects = pcnsl_loader.list_subjects()
        assert subjects == ["sub-0001", "sub-0002", "sub-0003"]

    def test_list_sessions(self, pcnsl_loader):
        sessions = pcnsl_loader.list_sessions("sub-0001")
        assert sessions == ["ses-0001"]

    def test_list_sessions_invalid_subject(self, pcnsl_loader):
        with pytest.raises(FileNotFoundError):
            pcnsl_loader.list_sessions("sub-9999")

    def test_list_subjects_with_processing_none(self, pcnsl_loader):
        subjects = pcnsl_loader.list_subjects_with_processing(processing=None)
        assert len(subjects) == 3

    def test_list_imaging_subjects(self, aws_loader):
        subjects = aws_loader.list_imaging_subjects()
        assert len(subjects) == 3
        assert "sub-0001" in subjects

    def test_get_subject_session_list(self, aws_loader):
        df = aws_loader.get_subject_session_list()
        assert len(df) == 3
        assert set(df.columns) == {"subject", "session", "accession"}
        assert df["accession"].iloc[0] == "0001"


# =============================================================================
# Statistics Loading
# =============================================================================


class TestStatisticsLoading:
    def test_get_statistics_path_no_processing(self, pcnsl_loader, tmp_pyalfe_dir):
        path = pcnsl_loader.get_statistics_path("sub-0001", processing=None)
        expected = tmp_pyalfe_dir / "sub-0001" / "ses-0001" / "statistics"
        assert path == expected

    def test_load_statistics_single(self, pcnsl_loader):
        df = pcnsl_loader.load_statistics_single(
            "sub-0001", stats_type="SummaryLesions", modality="FLAIR", processing=None
        )
        assert "subject" in df.columns
        assert df["subject"].iloc[0] == "sub-0001"

    def test_load_statistics_single_missing_file(self, pcnsl_loader):
        with pytest.raises(FileNotFoundError):
            pcnsl_loader.load_statistics_single(
                "sub-0001", stats_type="SummaryLesions", modality="FLAIR", processing="auto"
            )

    def test_load_statistics_multiple(self, pcnsl_loader):
        df = pcnsl_loader.load_statistics(
            subjects=["sub-0001", "sub-0002"],
            stats_type="IndividualLesions",
            modality="FLAIR",
            processing=None,
        )
        assert len(df) == 6  # 3 lesions × 2 subjects

    def test_load_statistics_all_subjects(self, pcnsl_loader):
        df = pcnsl_loader.load_statistics(
            subjects=None, stats_type="IndividualLesions", modality="T1Post", processing=None
        )
        assert df["subject"].nunique() == 3

    def test_load_statistics_ignore_missing_true(self, pcnsl_loader):
        df = pcnsl_loader.load_statistics(
            subjects=["sub-0001", "sub-9999"],
            stats_type="SummaryLesions",
            modality="FLAIR",
            processing=None,
            ignore_missing=True,
        )
        assert df["subject"].nunique() == 1

    def test_load_statistics_ignore_missing_false(self, pcnsl_loader):
        with pytest.raises(FileNotFoundError):
            pcnsl_loader.load_statistics(
                subjects=["sub-9999"],
                stats_type="SummaryLesions",
                modality="FLAIR",
                processing=None,
                ignore_missing=False,
            )


# =============================================================================
# Mask Loading
# =============================================================================


class TestMaskLoading:
    def test_load_lesion_mask(self, pcnsl_loader):
        mask = pcnsl_loader.load_lesion_mask("sub-0001", processing=None, modality="FLAIR")
        assert isinstance(mask, nib.Nifti1Image)
        assert mask.shape == (4, 4, 4)

    def test_load_lesion_mask_missing(self, pcnsl_loader):
        with pytest.raises(FileNotFoundError):
            pcnsl_loader.load_lesion_mask("sub-9999", processing=None)


# =============================================================================
# Skullstripped Loading
# =============================================================================


class TestSkullstrippedLoading:
    def test_list_skullstripped_images(self, pcnsl_loader):
        images = pcnsl_loader.list_skullstripped_images(
            "sub-0001", processing=None, space="FLAIR"
        )
        assert len(images) == 4  # T1, T1Post, FLAIR, ADC

    def test_load_skullstripped_image(self, pcnsl_loader):
        img = pcnsl_loader.load_skullstripped_image(
            "sub-0001", processing=None, space="FLAIR", sequence="ADC"
        )
        assert isinstance(img, nib.Nifti1Image)

    def test_load_skullstripped_images_dict(self, pcnsl_loader):
        images = pcnsl_loader.load_skullstripped_images(
            "sub-0001", processing=None, space="T1Post"
        )
        assert "FLAIR" in images
        assert "ADC" in images

    def test_load_image_with_mask(self, pcnsl_loader):
        img, mask = pcnsl_loader.load_image_with_mask(
            "sub-0001", processing=None, modality="FLAIR"
        )
        assert isinstance(img, nib.Nifti1Image)
        assert isinstance(mask, nib.Nifti1Image)


# =============================================================================
# Clinical Data Loading
# =============================================================================


class TestClinicalDataLoading:
    def test_load_demographics(self, aws_loader):
        df = aws_loader.load_clinical_data("demographics")
        assert len(df) == 3
        assert "Sex" in df.columns

    def test_load_mutations(self, aws_loader):
        df = aws_loader.load_clinical_data("ucsf500_mutations")
        assert len(df) == 5
        assert "gene" in df.columns

    def test_load_invalid_type(self, aws_loader):
        with pytest.raises(ValueError, match="Unknown data type"):
            aws_loader.load_clinical_data("nonexistent")

    def test_load_clinical_without_csv_raises(self, pcnsl_loader):
        with pytest.raises(ValueError, match="csv_path not configured"):
            pcnsl_loader.load_clinical_data("demographics")

    def test_list_available_clinical_data(self, aws_loader):
        available = aws_loader.list_available_clinical_data()
        assert "demographics" in available
        assert "ucsf500_mutations" in available
        assert len(available) == 6

    def test_filter_to_imaging_subjects(self, aws_loader):
        df = aws_loader.load_clinical_data(
            "demographics", filter_to_imaging_subjects=True
        )
        assert len(df) == 3

    def test_get_patient_accession_mapping(self, aws_loader):
        mapping = aws_loader.get_patient_accession_mapping()
        assert "patientdurablekey" in mapping.columns
        assert "subject" in mapping.columns
        assert len(mapping) == 3


# =============================================================================
# Data Merging
# =============================================================================


class TestDataMerging:
    def test_load_merged_data_defaults(self, aws_loader):
        df = aws_loader.load_merged_data()
        assert "Sex" in df.columns
        assert "BiopsyDate" in df.columns
        assert len(df) == 3

    def test_load_merged_data_custom_types(self, aws_loader):
        df = aws_loader.load_merged_data(
            clinical_types=["demographics", "ucsf500_mutations"]
        )
        assert "gene" in df.columns

    def test_load_demographics_with_imaging(self, aws_loader):
        df = aws_loader.load_demographics_with_imaging()
        assert "Sex" in df.columns
        assert len(df) == 3

    def test_load_mutations_for_imaging_subjects(self, aws_loader):
        df = aws_loader.load_mutations_for_imaging_subjects()
        assert "gene" in df.columns
        assert len(df) == 5


# =============================================================================
# DICOM Header Loading
# =============================================================================


class TestDicomHeaderLoading:
    def test_load_dicom_headers(self, aws_loader):
        df = aws_loader.load_dicom_headers()
        assert len(df) == 12  # 3 subjects × 4 sequences
        assert "subject" in df.columns
        assert "sequence" in df.columns
        assert "Manufacturer" in df.columns

    def test_dicom_headers_units_converted(self, aws_loader):
        df = aws_loader.load_dicom_headers()
        # TR was 9.0 seconds → 9000 ms
        assert df["RepetitionTime"].iloc[0] == pytest.approx(9000.0)
        # TE was 0.081 seconds → 81 ms
        assert df["EchoTime"].iloc[0] == pytest.approx(81.0)

    def test_load_dicom_headers_specific_subjects(self, aws_loader):
        df = aws_loader.load_dicom_headers(subjects=["sub-0001"])
        assert df["subject"].nunique() == 1

    def test_gauss_to_tesla_conversion(self, tmp_pyalfe_dir, tmp_csv_dir):
        """Verify that MagneticFieldStrength > 100 is converted from Gauss to T."""
        from pcnsl_data_loader import AWSDataLoader

        # Patch one sidecar to have Gauss value
        sidecar_path = (
            tmp_pyalfe_dir / "sub-0001" / "ses-0001"
            / "dcm2niix_sidecars" / "sub-0001_ses-0001_FLAIR.json"
        )
        data = json.loads(sidecar_path.read_text())
        data["MagneticFieldStrength"] = 15000  # Gauss
        sidecar_path.write_text(json.dumps(data))

        loader = AWSDataLoader(pyalfe_path=tmp_pyalfe_dir, csv_path=tmp_csv_dir)
        df = loader.load_dicom_headers(subjects=["sub-0001"])
        flair_row = df[df["sequence"] == "FLAIR"].iloc[0]
        assert flair_row["MagneticFieldStrength"] == pytest.approx(1.5)


class TestDicomGeometry:
    def test_load_aws_dicom_geometry(self, tmp_pyalfe_dir):
        from pcnsl_data_loader import load_aws_dicom_geometry

        df = load_aws_dicom_geometry(pyalfe_path=tmp_pyalfe_dir)
        assert len(df) == 12  # 3 subjects × 4 sequences
        assert "Matrix" in df.columns
        assert "FOV_row_mm" in df.columns
        assert "SliceGap_mm" in df.columns
        assert "NumSlices" in df.columns
        assert df["Matrix"].iloc[0] == "256x256"
        assert df["NumSlices"].iloc[0] == 30


# =============================================================================
# Module-Level Convenience Functions
# =============================================================================


class TestConvenienceFunctions:
    def test_load_aws_demographics(self, tmp_pyalfe_dir, tmp_csv_dir):
        import pcnsl_data_loader

        df = pcnsl_data_loader.load_aws_demographics(
            pyalfe_path=tmp_pyalfe_dir, csv_path=tmp_csv_dir
        )
        assert "Sex" in df.columns
        assert len(df) == 3

    def test_load_aws_clinical_imaging_merged(self, tmp_pyalfe_dir, tmp_csv_dir):
        import pcnsl_data_loader

        df = pcnsl_data_loader.load_aws_clinical_imaging_merged(
            pyalfe_path=tmp_pyalfe_dir, csv_path=tmp_csv_dir
        )
        assert "Sex" in df.columns
        assert "DiagnosisDate" in df.columns

    def test_load_aws_mutations(self, tmp_pyalfe_dir, tmp_csv_dir):
        import pcnsl_data_loader

        df = pcnsl_data_loader.load_aws_mutations(
            pyalfe_path=tmp_pyalfe_dir, csv_path=tmp_csv_dir
        )
        assert "gene" in df.columns

    def test_load_aws_dicom_headers(self, tmp_pyalfe_dir, tmp_csv_dir):
        import pcnsl_data_loader

        df = pcnsl_data_loader.load_aws_dicom_headers(
            pyalfe_path=tmp_pyalfe_dir, csv_path=tmp_csv_dir
        )
        assert len(df) == 12

    def test_load_aws_biopsy_and_diagnosis_dates(self, tmp_csv_dir, tmp_pyalfe_dir):
        import pcnsl_data_loader

        df = pcnsl_data_loader.load_aws_biopsy_and_diagnosis_dates(
            csv_path=tmp_csv_dir, pyalfe_path=tmp_pyalfe_dir
        )
        assert "BiopsyDate" in df.columns
        assert len(df) == 3
