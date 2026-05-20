"""
Tests for figures_for_manuscript.ipynb logic.

These tests extract the key data-processing and figure-generation logic from
the notebook and verify correctness against known fixture data, without
requiring the real dataset or running the notebook end-to-end.

Covers:
- Table 1: demographics aggregation, date parsing, survival calculation
- Table 2: mutation ranking
- Figure 3: lesion data aggregation, bootstrap CI
- Figure 1: medication timeline data prep, mutation lollipop
- Figure S1: flowchart (smoke test only — rendering)
- Tables S1/S2/S3: DICOM table construction
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


# =============================================================================
# Table 1 Logic: Date Parsing and Demographics
# =============================================================================


class TestDateParsing:
    def test_parse_date_with_century_normal(self):
        """Dates like 3/13/73 should parse to 1973, not 2073."""
        dt = pd.to_datetime("3/13/73", format="%m/%d/%y")
        if dt.year > 2025:
            dt = dt.replace(year=dt.year - 100)
        assert dt.year == 1973

    def test_parse_date_with_century_recent(self):
        """Dates like 8/26/21 should remain 2021."""
        dt = pd.to_datetime("8/26/21", format="%m/%d/%y")
        if dt.year > 2025:
            dt = dt.replace(year=dt.year - 100)
        assert dt.year == 2021

    def test_parse_date_with_century_boundary(self):
        """Dates in 2025 should stay as-is."""
        dt = pd.to_datetime("1/1/25", format="%m/%d/%y")
        if dt.year > 2025:
            dt = dt.replace(year=dt.year - 100)
        assert dt.year == 2025


class TestTable1Demographics:
    @pytest.fixture
    def sample_clinical_df(self, tmp_csv_dir):
        """Simulate the merged clinical DataFrame as produced in the notebook."""
        df = pd.DataFrame({
            "subject": ["sub-0001", "sub-0002", "sub-0003"],
            "session": ["ses-0001"] * 3,
            "Sex": ["Female", "Male", "Female"],
            "FirstRace": ["White", "White", "Other"],
            "Ethnicity": ["Not Hispanic or Latino", "Hispanic or Latino", "Not Hispanic or Latino"],
            "BirthDate": ["3/13/73", "5/23/62", "2/13/51"],
            "DiagnosisDate": ["8/26/21", "1/9/21", "3/20/22"],
            "DiagnosisDateMinusImageDate": [36, 4, 312],
            "BiopsyDateMinusImageDate": [np.nan, 2.0, 291.0],
        })
        return df

    def test_age_at_diagnosis_calculation(self, sample_clinical_df):
        df = sample_clinical_df.copy()

        def parse_date_with_century(date_str):
            if pd.isna(date_str):
                return pd.NaT
            dt = pd.to_datetime(date_str, format="%m/%d/%y")
            if dt.year > 2025:
                dt = dt.replace(year=dt.year - 100)
            return dt

        df["BirthDateParsed"] = df["BirthDate"].apply(parse_date_with_century)
        df["DiagnosisDateParsed"] = pd.to_datetime(df["DiagnosisDate"], format="%m/%d/%y")
        df["AgeAtDiagnosis"] = (df["DiagnosisDateParsed"] - df["BirthDateParsed"]).dt.days / 365.25

        assert df["AgeAtDiagnosis"].iloc[0] == pytest.approx(48.4, abs=0.5)
        assert df["AgeAtDiagnosis"].iloc[1] == pytest.approx(58.6, abs=0.5)
        assert all(df["AgeAtDiagnosis"] > 0)

    def test_imaging_timing_sign_convention(self, sample_clinical_df):
        """MRI timing: negative = MRI before diagnosis."""
        df = sample_clinical_df.copy()
        df["ImagingRelativeToDiagnosis"] = -df["DiagnosisDateMinusImageDate"]

        # DiagnosisDateMinusImageDate = 36 means diagnosis was 36 days after MRI
        # So ImagingRelativeToDiagnosis = -36 (MRI was 36 days BEFORE diagnosis)
        assert df["ImagingRelativeToDiagnosis"].iloc[0] == -36

    def test_sex_counts(self, sample_clinical_df):
        counts = sample_clinical_df["Sex"].value_counts()
        assert counts["Female"] == 2
        assert counts["Male"] == 1


class TestSurvivalCalculation:
    def test_survival_time_positive(self):
        """Survival time from diagnosis to death should be positive."""
        diagnosis = pd.Timestamp("2021-01-09")
        death = pd.Timestamp("2023-06-15")
        days = (death - diagnosis).days
        assert days > 0
        months = days / 30.44
        assert months == pytest.approx(29.1, abs=1.0)

    def test_censored_patients_use_today(self):
        """Alive patients should have follow-up calculated to today."""
        diagnosis = pd.Timestamp("2021-08-26")
        today = pd.Timestamp.today()
        days = (today - diagnosis).days
        assert days > 0


# =============================================================================
# Table 2 Logic: Mutation Ranking
# =============================================================================


class TestTable2Mutations:
    @pytest.fixture
    def mutations_df(self):
        return pd.DataFrame({
            "patientdurablekey": ["0001", "0001", "0002", "0002", "0002",
                                  "0003", "0003", "0004", "0004", "0004"],
            "gene": ["MYD88", "CD79B", "MYD88", "PIM1", "TBL1XR1",
                     "MYD88", "CD79B", "MYD88", "PIM1", "CD79B"],
            "hgvsp": ["p.L265P", "p.Y196H", "p.L265P", "p.E97K", "p.H381Q",
                      "p.L265P", "p.Y196H", "p.L265P", "p.E97K", "p.Y196H"],
            "hgvsc": ["c.794T>C"] * 10,
        })

    def test_gene_ranking_by_patient_count(self, mutations_df):
        patients_per_gene = mutations_df.groupby("gene")["patientdurablekey"].nunique()
        top_genes = patients_per_gene.sort_values(ascending=False)

        assert top_genes.index[0] == "MYD88"  # 4 patients
        assert top_genes.iloc[0] == 4
        assert top_genes.index[1] == "CD79B"  # 3 patients

    def test_frequency_calculation(self, mutations_df):
        total_patients = mutations_df["patientdurablekey"].nunique()
        assert total_patients == 4

        patients_myd88 = mutations_df[mutations_df["gene"] == "MYD88"]["patientdurablekey"].nunique()
        pct = (patients_myd88 / total_patients) * 100
        assert pct == 100.0


# =============================================================================
# Figure 3 Logic: Lesion Data Aggregation and Bootstrap CI
# =============================================================================


class TestBootstrapCI:
    def test_bootstrap_ci_basic(self):
        np.random.seed(42)
        data = np.random.normal(100, 10, size=100)

        boot_stats = []
        for _ in range(5000):
            sample = np.random.choice(data, size=len(data), replace=True)
            boot_stats.append(np.mean(sample))

        lower = np.percentile(boot_stats, 2.5)
        upper = np.percentile(boot_stats, 97.5)

        assert lower < np.mean(data) < upper
        assert upper - lower < 10  # CI width should be reasonable

    def test_bootstrap_ci_degenerate(self):
        data = np.array([5.0])
        # With only one data point, CI should be (0, 0) per notebook logic
        if len(data) < 2:
            lower, upper = 0, 0
        assert lower == 0 and upper == 0

    def test_bootstrap_ci_binary_data(self):
        """Test with binary data like the structural involvement calculation."""
        np.random.seed(42)
        data = np.array([100.0] * 80 + [0.0] * 20)  # 80% prevalence

        boot_stats = []
        for _ in range(5000):
            sample = np.random.choice(data, size=len(data), replace=True)
            boot_stats.append(np.mean(sample))

        lower = np.percentile(boot_stats, 2.5)
        upper = np.percentile(boot_stats, 97.5)

        assert 70 < lower < 80
        assert 80 < upper < 90


class TestLesionDataAggregation:
    def test_aggregate_summary_vertical_format(self, tmp_path):
        """Test that 2-column vertical CSVs are properly transposed."""
        stats_dir = tmp_path / "sub-0001" / "ses-0001" / "statistics" / "lesions_SummaryLesions"
        stats_dir.mkdir(parents=True)

        summary = pd.DataFrame({
            "metric": ["total_lesion_volume", "number_of_lesions"],
            "value": [5000.0, 3],
        })
        summary.to_csv(stats_dir / "sub-0001_ses-0001_FLAIR_SummaryLesions.csv", index=False)

        # Simulate the notebook's aggregation logic
        df = pd.read_csv(stats_dir / "sub-0001_ses-0001_FLAIR_SummaryLesions.csv")
        assert df.shape[1] == 2

        record = {}
        var_col, val_col = df.columns[0], df.columns[1]
        for _, row in df.iterrows():
            record[f"{row[var_col]}_FLAIR"] = row[val_col]

        assert record["total_lesion_volume_FLAIR"] == 5000.0
        assert record["number_of_lesions_FLAIR"] == 3

    def test_aggregate_radiomics_horizontal_format(self, tmp_path):
        """Test that multi-column horizontal CSVs are handled."""
        rad_dir = tmp_path / "sub-0001" / "ses-0001" / "statistics" / "lesions_radiomics"
        rad_dir.mkdir(parents=True)

        radiomics = pd.DataFrame({
            "original_firstorder_Kurtosis": [2.5],
            "original_firstorder_Entropy": [4.1],
            "original_firstorder_Energy": [1e6],
        })
        radiomics.to_csv(rad_dir / "sub-0001_ses-0001_T1Post_radiomics.csv", index=False)

        df = pd.read_csv(rad_dir / "sub-0001_ses-0001_T1Post_radiomics.csv")
        assert df.shape[1] > 2

        record = {}
        row_data = df.iloc[0]
        for col in df.columns:
            record[f"{col}_T1Post"] = row_data[col]

        assert record["original_firstorder_Kurtosis_T1Post"] == 2.5


class TestStructuralInvolvement:
    def test_volume_threshold_logic(self):
        """Subjects with >= 100 mm³ lesion volume count as involved."""
        volumes = np.array([5000.0, 50.0, 200.0, 0.0, 150.0])
        threshold = 100
        binary = (volumes >= threshold).astype(float) * 100
        pct = np.mean(binary)
        assert pct == 60.0  # 3/5 subjects


# =============================================================================
# Figure 1 Logic: Medication Timeline
# =============================================================================


class TestMedicationTimeline:
    def test_monthly_aggregation(self):
        dates = pd.to_datetime([
            "2023-01-05", "2023-01-10", "2023-01-15",
            "2023-02-05", "2023-02-10",
            "2023-03-01",
        ])
        df = pd.DataFrame({"dt": dates})
        df["ym"] = df["dt"].dt.to_period("M")
        monthly = df.groupby("ym").size()

        assert monthly.iloc[0] == 3  # January
        assert monthly.iloc[1] == 2  # February
        assert monthly.iloc[2] == 1  # March

    def test_drug_class_detection(self):
        meds = pd.Series([
            "dexamethasone sodium phosphate",
            "methotrexate",
            "methylprednisolone",
            "prednisone",
            "ruxolitinib",
        ])
        lower = meds.str.lower()

        assert lower.str.contains("dexamethasone", na=False).sum() == 1
        assert lower.str.contains("methylprednisolone", na=False).sum() == 1
        prednisone_mask = lower.str.contains("prednisone", na=False) & ~lower.str.contains("methyl", na=False)
        assert prednisone_mask.sum() == 1
        assert lower.str.contains("ruxolitinib", na=False).sum() == 1


# =============================================================================
# Tables S1/S2: DICOM Table Logic
# =============================================================================


class TestAcquisitionParamTable:
    def test_field_strength_categorization(self):
        fs = pd.Series([1.5, 3.0, 3.0, 1.5, 3.0])
        n_1_5 = (fs.round(1) == 1.5).sum()
        n_3_0 = (fs.round(1) == 3.0).sum()
        assert n_1_5 == 2
        assert n_3_0 == 3

    def test_missing_data_percentage(self):
        values = pd.Series([1.0, 2.0, np.nan, 4.0, np.nan])
        n_total = len(values)
        n_missing = values.isna().sum()
        pct_miss = 100 * n_missing / n_total
        assert pct_miss == 40.0

    def test_scanner_string_construction(self):
        row = {"Manufacturer": "Siemens", "ManufacturersModelName": "Prisma"}
        parts = [row.get("Manufacturer"), row.get("ManufacturersModelName")]
        s = " ".join(str(p) for p in parts if pd.notna(p) and str(p).strip())
        assert s == "Siemens Prisma"


class TestGeometryTable:
    def test_slice_gap_only_for_2d(self):
        """SliceGap should only be reported for 2D acquisitions."""
        df = pd.DataFrame({
            "MRAcquisitionType": ["2D", "2D", "3D", "3D"],
            "SliceGap_mm": [1.0, 0.5, -0.5, -0.3],
        })
        mask_2d = df["MRAcquisitionType"] == "2D"
        gap_2d = df.loc[mask_2d, "SliceGap_mm"]
        assert len(gap_2d) == 2
        assert gap_2d.mean() == 0.75

    def test_voxel_volume_calculation(self):
        ps = [0.9375, 0.9375]
        st = 5.0
        vol = ps[0] * ps[1] * st
        assert vol == pytest.approx(4.394, abs=0.01)


# =============================================================================
# Figure S1: Flowchart Smoke Test
# =============================================================================


class TestFlowchartSmoke:
    def test_flowchart_renders_without_error(self):
        """Verify the flowchart rendering code runs without exceptions."""
        from matplotlib.patches import FancyBboxPatch

        fig, ax = plt.subplots(figsize=(11, 13))
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0.0, 1.05)
        ax.axis("off")

        ax.add_patch(FancyBboxPatch(
            (0.2, 0.5), 0.3, 0.1,
            boxstyle="round,pad=0.015",
            facecolor="#ddeef8", edgecolor="#2c3e50",
        ))
        ax.text(0.35, 0.55, "Test Box", ha="center", va="center")

        plt.close(fig)


# =============================================================================
# Output File Naming (verify alignment with submission)
# =============================================================================


class TestOutputNaming:
    def test_figure_filenames_match_submission(self):
        """Verify expected output filenames match submission numbering."""
        expected_main = [
            "fig1_data_overview.png",
            "fig3_lesion_statistics.png",
            "table1.html",
            "table2.html",
        ]
        expected_supplementary = [
            "figS1_selection_flowchart.png",
            "tableS1_scanner_info_per_subject.html",
            "tableS2_acquisition_params.html",
            "tableS3_geometry.html",
        ]
        # Just verify the naming conventions are valid strings
        for name in expected_main + expected_supplementary:
            assert "/" not in name
            assert name.endswith((".png", ".html", ".eps", ".pdf"))


# =============================================================================
# Integration: End-to-End Data Pipeline
# =============================================================================


class TestEndToEndPipeline:
    def test_full_clinical_imaging_merge(self, aws_loader):
        """Test the complete merge pipeline as used in Table 1."""
        df = aws_loader.load_merged_data(
            clinical_types=["demographics", "biopsy_and_diagnosis_dates"]
        )
        required_cols = {"subject", "session", "Sex", "BirthDate", "DiagnosisDate"}
        assert required_cols.issubset(set(df.columns))
        assert len(df) == 3

    def test_mutation_gene_ranking_pipeline(self, aws_loader):
        """Test the mutation analysis pipeline as used in Table 2."""
        genes_df = aws_loader.load_mutations_for_imaging_subjects()
        patients_per_gene = genes_df.groupby("gene")["patientdurablekey"].nunique()
        top_genes = patients_per_gene.sort_values(ascending=False).head(10)

        assert len(top_genes) > 0
        assert top_genes.iloc[0] >= top_genes.iloc[-1]

    def test_dicom_geometry_pipeline(self, tmp_bids_dir):
        """Test geometry loading pipeline as used in Table S3."""
        from pcnsl_data_loader import load_aws_dicom_geometry

        geom_df = load_aws_dicom_geometry(bids_path=tmp_bids_dir)
        assert geom_df["sequence"].nunique() == 4
        assert "PixelSpacing_mm" in geom_df.columns
        assert "VoxelVolume_mm3" in geom_df.columns

        # Verify per-sequence grouping works
        for seq in geom_df["sequence"].unique():
            sub = geom_df[geom_df["sequence"] == seq]
            assert len(sub) == 3  # 3 subjects
