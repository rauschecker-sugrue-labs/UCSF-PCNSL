"""
Shared fixtures for pcnsl_data_loader and figures_for_manuscript tests.

Creates minimal fake BIDS directory structures and CSV files that exercise
the data-loading and figure-generation code without needing real data.
"""

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def tmp_bids_dir(tmp_path):
    """Create a minimal BIDS directory structure with 3 fake subjects."""
    bids = tmp_path / "bids"
    subjects = ["sub-0001", "sub-0002", "sub-0003"]
    session = "ses-0001"
    sequences = ["T1w", "ce-gadolinium_T1w", "FLAIR"]

    for subj in subjects:
        # Raw anatomy
        anat_dir = bids / subj / session / "anat"
        anat_dir.mkdir(parents=True)
        for seq in sequences:
            img = nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4))
            nib.save(img, anat_dir / f"{subj}_{session}_{seq}.nii.gz")

        # DWI
        dwi_dir = bids / subj / session / "dwi"
        dwi_dir.mkdir(parents=True)
        img = nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4))
        nib.save(img, dwi_dir / f"{subj}_{session}_ADC.nii.gz")

        # Derivatives (no {auto|human} subdirectory for AWS layout)
        deriv_base = bids / "derivatives" / "pyalfe" / subj / session

        # Statistics - SummaryLesions
        for modality in ("FLAIR", "T1Post"):
            stats_dir = deriv_base / "statistics" / "lesions_SummaryLesions"
            stats_dir.mkdir(parents=True, exist_ok=True)
            summary_df = pd.DataFrame({
                "metric": ["total_lesion_volume", "number_of_lesions",
                           "lesion_volume_in_White Matter", "lesion_volume_in_Frontal"],
                "value": [5000.0, 3, 3500.0, 1200.0],
            })
            summary_df.to_csv(
                stats_dir / f"{subj}_{session}_{modality}_SummaryLesions.csv",
                index=False,
            )

            # Statistics - radiomics
            rad_dir = deriv_base / "statistics" / "lesions_radiomics"
            rad_dir.mkdir(parents=True, exist_ok=True)
            radiomics_df = pd.DataFrame({
                "original_firstorder_Kurtosis": [2.5],
                "original_firstorder_Entropy": [4.1],
                "original_firstorder_Energy": [1e6],
            })
            radiomics_df.to_csv(
                rad_dir / f"{subj}_{session}_{modality}_radiomics.csv",
                index=False,
            )

            # Statistics - IndividualLesions
            ind_dir = deriv_base / "statistics" / "lesions_IndividualLesions"
            ind_dir.mkdir(parents=True, exist_ok=True)
            ind_df = pd.DataFrame({
                "lesion_id": [1, 2, 3],
                "volume": [2000.0, 1500.0, 1500.0],
            })
            ind_df.to_csv(
                ind_dir / f"{subj}_{session}_{modality}_IndividualLesions.csv",
                index=False,
            )

        # Masks
        masks_dir = deriv_base / "masks" / "lesions_seg_comp"
        masks_dir.mkdir(parents=True, exist_ok=True)
        for modality in ("FLAIR", "T1Post"):
            mask_img = nib.Nifti1Image(
                np.random.randint(0, 4, (4, 4, 4), dtype=np.int16), np.eye(4)
            )
            nib.save(
                mask_img,
                masks_dir / f"{subj}_{session}_{modality}_abnormal_seg_comp.nii.gz",
            )

        # Skullstripped
        for space in ("FLAIR", "T1Post"):
            ss_dir = deriv_base / "skullstripped" / f"lesions_{space}_space"
            ss_dir.mkdir(parents=True, exist_ok=True)
            for seq in ("T1", "T1Post", "FLAIR", "ADC"):
                img = nib.Nifti1Image(
                    np.random.rand(4, 4, 4).astype(np.float32), np.eye(4)
                )
                nib.save(
                    img,
                    ss_dir / f"{subj}_{session}_{seq}_to_{space}_skullstripped.nii.gz",
                )

        # dcm2niix sidecars
        sidecar_dir = deriv_base / "dcm2niix_sidecars"
        sidecar_dir.mkdir(parents=True, exist_ok=True)
        for seq in ("FLAIR", "T1w", "ce-gadolinium_T1w", "DWI_ADC"):
            sidecar = {
                "Modality": "MR",
                "MagneticFieldStrength": 3.0,
                "Manufacturer": "Siemens",
                "ManufacturersModelName": "Prisma",
                "RepetitionTime": 9.0,
                "EchoTime": 0.081,
                "InversionTime": 2.5,
                "FlipAngle": 150,
                "SliceThickness": 5.0,
                "MRAcquisitionType": "2D",
            }
            with open(sidecar_dir / f"{subj}_{session}_{seq}.json", "w") as f:
                json.dump(sidecar, f)

        # Raw DICOM tag JSONs
        dicom_dir = deriv_base / "dicom_headers"
        dicom_dir.mkdir(parents=True, exist_ok=True)
        for seq in ("FLAIR", "T1w", "ce-gadolinium_T1w", "DWI_ADC"):
            dicom_json = {
                "tags": {
                    "(0008,0070)": {"value": "Siemens"},
                    "(0008,1090)": {"value": "Prisma"},
                    "(0018,0023)": {"value": "2D"},
                    "(0018,0050)": {"value": 5.0},
                    "(0018,0088)": {"value": 5.5},
                    "(0018,0080)": {"value": 9000},
                    "(0018,0081)": {"value": 81},
                    "(0018,0082)": {"value": 2500},
                    "(0018,0087)": {"value": 3.0},
                    "(0018,1314)": {"value": 150},
                    "(0028,0010)": {"value": 256},
                    "(0028,0011)": {"value": 256},
                    "(0028,0030)": {"value": [0.9375, 0.9375]},
                },
                "_consolidation_info": {"num_files_in_series": 30},
            }
            with open(dicom_dir / f"DCMQ_{subj}_{session}_{seq}.json", "w") as f:
                json.dump(dicom_json, f)

    return bids


@pytest.fixture
def tmp_csv_dir(tmp_path):
    """Create minimal clinical CSV files matching expected schema."""
    csv_dir = tmp_path / "csvs"
    csv_dir.mkdir()

    # demographics.csv
    pd.DataFrame({
        "patientdurablekey": [1, 2, 3],
        "Sex": ["Female", "Male", "Female"],
        "BirthDate": ["3/13/73", "5/23/62", "2/13/51"],
        "Deathdate": [np.nan, "6/15/23", np.nan],
        "FirstRace": ["White", "White", "Other"],
        "Ethnicity": ["Not Hispanic or Latino", "Hispanic or Latino", "Not Hispanic or Latino"],
        "PreferredLanguage": ["English", "English", "English"],
        "SexualOrientation": ["Straight", "Straight", "Straight"],
        "SecondRace": [np.nan, np.nan, np.nan],
    }).to_csv(csv_dir / "demographics.csv", index=False)

    # biopsy_and_diagnosis_dates.csv
    pd.DataFrame({
        "patientdurablekey": [1, 2, 3],
        "accession": ["0001", "0002", "0003"],
        "BiopsyDate": [np.nan, "1/7/21", "2/27/22"],
        "DiagnosisDate": ["8/26/21", "1/9/21", "3/20/22"],
        "ImageDate": ["7/21/21", "1/5/21", "5/12/21"],
        "BiopsyDateMinusImageDate": [np.nan, 2.0, 291.0],
        "DiagnosisDateMinusImageDate": [36, 4, 312],
    }).to_csv(csv_dir / "biopsy_and_diagnosis_dates.csv", index=False)

    # diagnosis_history.csv
    pd.DataFrame({
        "patientdurablekey": [1, 1, 2, 3],
        "DiagnosisCode": ["C85.1", "C85.1", "C85.1", "C85.1"],
        "DiagnosisDate": ["8/26/21", "9/1/21", "1/9/21", "3/20/22"],
    }).to_csv(csv_dir / "diagnosis_history.csv", index=False)

    # medication_list_administered.csv
    pd.DataFrame({
        "patientdurablekey": [1, 1, 2, 3],
        "MedicationGenericName": ["dexamethasone", "methotrexate", "rituximab", "dexamethasone"],
        "AdministrationInstant": ["8/26/21 10:00", "9/1/21 14:00", "1/9/21 08:00", "3/20/22 09:00"],
    }).to_csv(csv_dir / "medication_list_administered.csv", index=False)

    # medication_list_ordered.csv
    pd.DataFrame({
        "patientdurablekey": [1, 2, 3],
        "MedicationGenericName": ["dexamethasone", "rituximab", "methotrexate"],
        "OrderInstant": ["8/26/21", "1/9/21", "3/20/22"],
    }).to_csv(csv_dir / "medication_list_ordered.csv", index=False)

    # ucsf500_mutations.csv
    pd.DataFrame({
        "patientdurablekey": [1, 1, 2, 2, 2],
        "gene": ["MYD88", "CD79B", "MYD88", "PIM1", "TBL1XR1"],
        "hgvsp": ["p.L265P", "p.Y196H", "p.L265P", "p.E97K", "p.H381Q"],
        "hgvsc": ["c.794T>C", "c.586T>C", "c.794T>C", "c.289G>A", "c.1143C>A"],
        "tmbscore": [5.5, 5.5, 8.2, 8.2, 8.2],
        "microsatellitestatus": ["MS-Stable", "MS-Stable", "MS-Stable", "MS-Stable", "MS-Stable"],
    }).to_csv(csv_dir / "ucsf500_mutations.csv", index=False)

    # Data dictionaries
    pd.DataFrame({
        "variable": ["patientdurablekey", "Sex", "BirthDate"],
        "description": ["Patient ID", "Sex assigned at birth", "Date of birth"],
    }).to_csv(csv_dir / "data_dictionary_clinical.csv", index=False)

    pd.DataFrame({
        "variable": ["total_lesion_volume", "number_of_lesions"],
        "description": ["Total lesion volume in mm³", "Number of discrete lesions"],
    }).to_csv(csv_dir / "data_dictionary_imaging.csv", index=False)

    return csv_dir


@pytest.fixture
def pcnsl_loader(tmp_bids_dir):
    """Create a PCNSLDataLoader with the fake BIDS directory."""
    from pcnsl_data_loader import PCNSLDataLoader
    return PCNSLDataLoader(tmp_bids_dir)


@pytest.fixture
def aws_loader(tmp_bids_dir, tmp_csv_dir):
    """Create an AWSDataLoader with the fake BIDS + CSV directories."""
    from pcnsl_data_loader import AWSDataLoader
    return AWSDataLoader(bids_path=tmp_bids_dir, csv_path=tmp_csv_dir)
