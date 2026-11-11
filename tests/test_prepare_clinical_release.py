"""Tests for scripts/prepare_clinical_release.py using synthetic data."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_clinical_release.py"
spec = importlib.util.spec_from_file_location("prepare_clinical_release", SCRIPT)
pcr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pcr)


class TestIdMapping:
    def test_removed_maps_to_none_and_later_shift_down(self):
        m = pcr.build_id_mapping(5, remove=3)
        assert m == {1: 1, 2: 2, 3: None, 4: 3, 5: 4}

    def test_rename_plan_is_ascending_and_skips_unchanged(self, tmp_path):
        plan = pcr.plan_imaging_renames(tmp_path, pcr.build_id_mapping(5, remove=3))
        assert [(a.name, b.name) for a, b in plan] == [("sub-0004", "sub-0003"), ("sub-0005", "sub-0004")]


class TestMdyToIso:
    def test_birth_year_in_previous_century(self):
        assert pcr.mdy_to_iso(pd.Series(["1/2/47"])).tolist() == ["1947-01-02"]

    def test_recent_dates_stay_in_this_century(self):
        assert pcr.mdy_to_iso(pd.Series(["11/23/20", "2/17/16"])).tolist() == ["2020-11-23", "2016-02-17"]

    def test_blank_stays_blank(self):
        assert pcr.mdy_to_iso(pd.Series(["", "8/30/12"])).tolist() == ["", "2012-08-30"]

    def test_unparseable_raises(self):
        with pytest.raises(ValueError):
            pcr.mdy_to_iso(pd.Series(["2012-08-30"]))


@pytest.fixture
def demographics():
    return pd.DataFrame(
        {
            "patientdurablekey": ["0001", "0002", "0003"],
            "Sex": ["Male", "Female", "Male"],
            "SexualOrientation": ["Straight", "Gay", ""],
            "BirthDate": ["1/2/47", "3/13/73", "5/5/09"],
            "Deathdate": ["", "8/30/12", ""],
            "PreferredLanguage": ["English", "Farsi", "English"],
        }
    )


class TestTransformClinicalTable:
    def test_removes_subject_and_renumbers(self, demographics):
        out = pcr.transform_clinical_table("demographics.csv", demographics, pcr.build_id_mapping(3, remove=2))
        assert out["patientdurablekey"].tolist() == ["0001", "0002"]
        assert out["participant_id"].tolist() == ["sub-0001", "sub-0002"]
        assert out["BirthDate"].tolist() == ["1947-01-02", "2009-05-05"]  # old 0003 is now 0002

    def test_drops_sensitive_columns_and_puts_participant_id_first(self, demographics):
        out = pcr.transform_clinical_table("demographics.csv", demographics, pcr.build_id_mapping(3, remove=2))
        assert list(out.columns) == ["participant_id", "patientdurablekey", "Sex", "BirthDate", "Deathdate"]

    def test_unknown_subject_key_raises(self, demographics):
        with pytest.raises(ValueError):
            pcr.transform_clinical_table("demographics.csv", demographics, pcr.build_id_mapping(2, remove=2))


def test_integer_date_key_is_authoritative():
    """A future year in the 8-digit key is kept (not silently moved back a century)."""
    df = pd.DataFrame({"patientdurablekey": ["0001", "0001"], "DiagnosisName": ["a", "b"],
                       "StartDateKey": ["20400512", "20201123"], "StartDateKeyValue": ["5/12/40", "11/23/20"],
                       "EndDateKey": ["-1", "20201123"], "EndDateKeyValue": ["", "11/23/20"],
                       "DiagnosisEventKey": ["x", "y"], "DiagnosisKey": ["1", "2"]})
    out = pcr.transform_clinical_table("diagnosis_history.csv", df, pcr.build_id_mapping(1, remove=99))
    assert out["StartDateKeyValue"].tolist() == ["2040-05-12", "2020-11-23"]
    assert out["EndDateKeyValue"].tolist() == ["", "2020-11-23"]
    assert pcr.check_date_consistency("diagnosis_history.csv", out) == []


class TestDateConsistency:
    def test_diagnosis_history_keys_match(self):
        df = pd.DataFrame({"StartDateKeyValue": ["2020-11-23", ""], "StartDateKey": ["20201123", "-1"]})
        assert pcr.check_date_consistency("diagnosis_history.csv", df) == []

    def test_wrong_century_is_flagged(self):
        df = pd.DataFrame({"StartDateKeyValue": ["2047-01-02"], "StartDateKey": ["19470102"]})
        assert pcr.check_date_consistency("diagnosis_history.csv", df)

    def test_day_difference_checked(self):
        df = pd.DataFrame(
            {"ImageDate": ["2016-02-02"], "DiagnosisDate": ["2016-02-17"], "DiagnosisDateMinusImageDate": ["15"],
             "BiopsyDate": [""], "BiopsyDateMinusImageDate": [""]}
        )
        assert pcr.check_date_consistency("biopsy_and_diagnosis_dates.csv", df) == []
        df["DiagnosisDateMinusImageDate"] = ["16"]
        assert pcr.check_date_consistency("biopsy_and_diagnosis_dates.csv", df)


def test_rename_subject_renames_prefixed_files_then_folder(tmp_path):
    old = tmp_path / "sub-0004" / "ses-0001" / "masks"
    old.mkdir(parents=True)
    (old / "sub-0004_ses-0001_FLAIR_seg.nii.gz").write_text("x")
    (old / "README").write_text("unprefixed")
    n = pcr.rename_subject(tmp_path / "sub-0004", tmp_path / "sub-0003")
    assert n == 1
    new = tmp_path / "sub-0003" / "ses-0001" / "masks"
    assert sorted(p.name for p in new.iterdir()) == ["README", "sub-0003_ses-0001_FLAIR_seg.nii.gz"]
    assert not (tmp_path / "sub-0004").exists()
