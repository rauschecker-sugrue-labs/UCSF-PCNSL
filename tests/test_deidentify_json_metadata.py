"""Tests for scripts/deidentify_json_metadata.py using synthetic metadata."""

import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "deidentify_json_metadata.py"
spec = importlib.util.spec_from_file_location("deidentify_json_metadata", SCRIPT)
dj = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dj)


@pytest.fixture
def header_doc():
    """Synthetic dicom_headers document covering every removal rule."""
    return {
        "_consolidation_info": {"num_files_in_series": 3},
        "tags": {
            "(0008,0020)": {"vr": "DA", "value": "20150825", "keyword": "StudyDate"},
            "(0008,0030)": {"vr": "TM", "value": "105114", "keyword": "StudyTime"},
            "(0008,0050)": {"vr": "SH", "value": "F7919F839FF8F", "keyword": "AccessionNumber"},
            "(0010,0020)": {"vr": "LO", "value": "DF0DE40C681F68", "keyword": "PatientID"},
            "(0008,1010)": {"vr": "SH", "value": "LILA", "keyword": "StationName"},
            "(0018,1000)": {"vr": "LO", "value": "000000KAISERWCMR", "keyword": "DeviceSerialNumber"},
            "(0043,0011)": {"vr": "LO", "value": "ORIGINAL_STATIONNAME", "keyword": None},
            "(0043,1101)": {"vr": "LO", "value": "WBDEDM1", "keyword": None},
            "(0008,1011)": {"vr": "LO", "value": "import", "keyword": None},
            "(0018,0081)": {"vr": "DS", "value": 70.2, "keyword": "EchoTime"},
            "(0008,0070)": {"vr": "LO", "value": "GE", "keyword": "Manufacturer"},
            "(0020,000D)": {"vr": "UI", "value": "1.2.720.1", "keyword": "StudyInstanceUID"},
            "(0040,0275)": {
                "vr": "SQ",
                "keyword": "RequestAttributesSequence",
                "value": [
                    {
                        "(0040,0009)": {"vr": "SH", "value": "F7919F839FF8F", "keyword": "ScheduledProcedureStepID"},
                        "(0018,A002)": {"vr": "DT", "value": "20210516152538", "keyword": "ContributionDateTime"},
                        "(0032,1060)": {"vr": "LO", "value": "MR BRAIN", "keyword": "RequestedProcedureDescription"},
                        "(0032,1064)": {
                            "vr": "SQ",
                            "keyword": "RequestedProcedureCodeSequence",
                            "value": [
                                {
                                    "(0008,0100)": {"vr": "SH", "value": "F7919F839FF8F", "keyword": "CodeValue"},
                                    "(0008,0104)": {"vr": "LO", "value": "MR BRAIN", "keyword": "CodeMeaning"},
                                }
                            ],
                        },
                    }
                ],
            },
        },
    }


@pytest.fixture
def sidecar_doc():
    """Synthetic dcm2niix sidecar with clock times, identifiers and timing parameters."""
    return {
        "Modality": "MR",
        "AcquisitionTime": "10:52:46.000000",
        "AcquisitionDateTime": "2015-08-25T10:52:46",
        "StationName": "EXTERNAL_MR",
        "DeviceSerialNumber": "000000000000GEHC",
        "EchoTime": 0.07,
        "RepetitionTime": 10.0,
        "InversionTime": 2.2,
        "AcquisitionDuration": 120.0,
        "SliceTiming": [0.0, 0.5],
        "DeidentificationMethodCodeSequence": [{"CodeValue": "F7919F839FF8F", "CodeMeaning": "x"}, {"CodeValue": "113100"}],
    }


REMOVED_TOP = ["(0008,0020)", "(0008,0030)", "(0008,0050)", "(0010,0020)", "(0008,1010)",
               "(0018,1000)", "(0043,0011)", "(0043,1101)", "(0008,1011)"]
KEPT_TOP = ["(0018,0081)", "(0008,0070)", "(0020,000D)", "(0040,0275)"]


class TestCleanDicomHeader:
    def test_removes_dates_identifiers_and_private_tags(self, header_doc):
        out = dj.clean_dicom_header(header_doc, Counter())
        assert not set(REMOVED_TOP) & set(out["tags"])

    def test_keeps_acquisition_parameters_and_uids(self, header_doc):
        out = dj.clean_dicom_header(header_doc, Counter())
        assert set(KEPT_TOP) <= set(out["tags"])

    def test_cleans_nested_sequence_items(self, header_doc):
        item = dj.clean_dicom_header(header_doc, Counter())["tags"]["(0040,0275)"]["value"][0]
        assert set(item) == {"(0032,1060)", "(0032,1064)"}

    def test_collects_identifier_values_including_nested(self, header_doc):
        assert dj.collect_id_values([header_doc]) == {"F7919F839FF8F", "DF0DE40C681F68"}

    def test_removes_identifier_value_copied_into_other_field(self, header_doc):
        ids = dj.collect_id_values([header_doc])
        removed = Counter()
        out = dj.clean_dicom_header(header_doc, removed, ids)
        code = out["tags"]["(0040,0275)"]["value"][0]["(0032,1064)"]["value"][0]
        assert set(code) == {"(0008,0104)"}
        assert removed["identifier-value:CodeValue"] == 1

    def test_counts_removals_by_reason(self, header_doc):
        removed = Counter()
        dj.clean_dicom_header(header_doc, removed)  # no id_values: name/VR rules only
        assert removed["private/unknown"] == 3
        assert removed["identifier:AccessionNumber"] == 1
        assert removed["date/time:ContributionDateTime"] == 1
        assert sum(removed.values()) == 11

    def test_does_not_mutate_input(self, header_doc):
        original = json.loads(json.dumps(header_doc))
        out = dj.clean_dicom_header(header_doc, Counter())
        assert header_doc == original and out["_consolidation_info"] == original["_consolidation_info"]

    def test_verifier(self, header_doc):
        ids = dj.collect_id_values([header_doc])
        assert dj.find_remaining(header_doc, "dicom_header", ids)
        assert dj.find_remaining(dj.clean_dicom_header(header_doc, Counter()), "dicom_header", ids) == ["id-value:F7919F839FF8F"]
        assert dj.find_remaining(dj.clean_dicom_header(header_doc, Counter(), ids), "dicom_header", ids) == []


class TestCleanSidecar:
    def test_removes_clock_times_and_identifiers(self, sidecar_doc):
        out = dj.clean_sidecar(sidecar_doc, Counter())
        assert not {"AcquisitionTime", "AcquisitionDateTime", "StationName", "DeviceSerialNumber"} & set(out)

    def test_keeps_sequence_timing_parameters(self, sidecar_doc):
        out = dj.clean_sidecar(sidecar_doc, Counter())
        for key in ["Modality", "EchoTime", "RepetitionTime", "InversionTime", "AcquisitionDuration", "SliceTiming"]:
            assert out[key] == sidecar_doc[key]

    def test_removes_identifier_values_nested_in_sidecar(self, sidecar_doc):
        out = dj.clean_sidecar(sidecar_doc, Counter(), frozenset({"F7919F839FF8F"}))
        assert out["DeidentificationMethodCodeSequence"] == [{"CodeMeaning": "x"}, {"CodeValue": "113100"}]

    def test_verifier(self, sidecar_doc):
        ids = frozenset({"F7919F839FF8F"})
        assert dj.find_remaining(dj.clean_sidecar(sidecar_doc, Counter()), "sidecar", ids)
        assert dj.find_remaining(dj.clean_sidecar(sidecar_doc, Counter(), ids), "sidecar", ids) == []


def test_private_tag_detection():
    assert dj.is_private_tag("(0043,1039)") and dj.is_private_tag("(2005,0014)")
    assert not dj.is_private_tag("(0008,0020)") and not dj.is_private_tag("not-a-tag")


def test_clean_dataset_end_to_end_preserves_formatting(tmp_path, header_doc, sidecar_doc):
    """Dry run writes nothing; real run rewrites in place with original formatting."""
    ses = tmp_path / "pyalfe" / "sub-0001" / "ses-0001"
    (ses / "dicom_headers").mkdir(parents=True)
    (ses / "dcm2niix_sidecars").mkdir()
    hdr = ses / "dicom_headers" / "sub-0001_ses-0001_FLAIR.json"
    car = ses / "dcm2niix_sidecars" / "sub-0001_ses-0001_FLAIR.json"
    hdr.write_text(json.dumps(header_doc, indent=2))
    car.write_text(json.dumps(sidecar_doc, indent="\t") + "\n")

    removed, _ = dj.clean_dataset(tmp_path, dry_run=True)
    assert sum(removed.values()) == 17  # 11 header + 1 nested CodeValue + 4 sidecar + 1 sidecar CodeValue
    assert "StudyDate" in hdr.read_text()

    removed, ids = dj.clean_dataset(tmp_path, dry_run=False)
    assert sum(removed.values()) == 17
    assert dj.verify_dataset(tmp_path, ids) == 0
    assert "F7919F839FF8F" not in hdr.read_text() + car.read_text()
    assert hdr.read_text().startswith('{\n  "') and not hdr.read_text().endswith("\n")
    assert car.read_text().startswith('{\n\t"') and car.read_text().endswith("\n")
