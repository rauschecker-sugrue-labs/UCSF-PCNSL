"""Remove dates, times, site identifiers and private tags from released JSON metadata.

The clinical CSVs in pcnsl-dataset_v1.0 carry randomly shifted dates, but the
per-series JSON metadata were exported with original acquisition dates/times and
with scanner/site identifiers. This script cleans them in place:

``dicom_headers/*.json`` — an element is removed (recursing into sequence items) if
  * its value representation (VR) is DA, DT or TM (any date or time), or
  * its keyword is in ``DICOM_IDENTIFIER_KEYWORDS`` (patient/accession hashes,
    station names, AE titles, device serial numbers, institution, staff names,
    procedure-step IDs/locations, free-text comments), or
  * it is a private element (odd group number) or has no standard keyword.
    Private blocks include vendor/PACS fields such as ORIGINAL_STATIONNAME.

Value-based rule — the UCSF de-identification service copied hashed patient /
accession IDs into unrelated fields (e.g. ``CodeValue`` in procedure code
sequences, ``CodingSchemeVersion``, ``StackID``). For each session, all values of
the identifier keywords in ``ID_VALUE_KEYWORDS`` are collected from the four
headers first; any element (header) or key (sidecar, at any nesting depth)
holding one of those values is then removed too.

``dcm2niix_sidecars/*.json`` — a key is removed if it names a calendar date or
clock time (e.g. ``AcquisitionTime``) or is in ``SIDECAR_IDENTIFIER_KEYS``.
Sequence timing parameters (``EchoTime``, ``RepetitionTime``, ...) are kept.

UIDs (Study/Series/SOP/FrameOfReference) were already remapped by the UCSF
de-identification service and are kept for within-dataset consistency checks.

After cleaning, an independent verification pass re-reads every file and exits
non-zero if any targeted field remains.

Usage
-----
    uv run python scripts/deidentify_json_metadata.py pcnsl-dataset_v1.0 --dry-run
    uv run python scripts/deidentify_json_metadata.py pcnsl-dataset_v1.0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

log = logging.getLogger(__name__)

DATE_TIME_VRS = frozenset({"DA", "DT", "TM"})

DICOM_IDENTIFIER_KEYWORDS = frozenset(
    {
        # patient / accession identifiers (pseudonymous hashes) and issuers
        "PatientID", "PatientName", "OtherPatientIDs", "OtherPatientIDsSequence",
        "IssuerOfPatientID", "IssuerOfPatientIDQualifiersSequence", "TypeOfPatientID",
        "AccessionNumber", "IssuerOfAccessionNumberSequence", "StudyID",
        "ScheduledProcedureStepID", "PerformedProcedureStepID", "RequestedProcedureID",
        # site / device identifiers
        "InstitutionName", "InstitutionAddress", "InstitutionalDepartmentName",
        "StationName", "StationAETitle", "PerformedStationName", "PerformedStationAETitle",
        "ScheduledStationName", "ScheduledStationAETitle", "ScheduledProcedureStepLocation",
        "PerformedLocation", "DeviceSerialNumber", "SecondaryCaptureDeviceID",
        # staff
        "ReferringPhysicianName", "PerformingPhysicianName", "RequestingPhysician",
        "OperatorsName", "NameOfPhysiciansReadingStudy", "PhysiciansOfRecord",
        # free text that can carry anything
        "ImageComments", "CommentsOnThePerformedProcedureStep",
        "CommentsOnTheScheduledProcedureStep", "ImagingServiceRequestComments",
        "RequestedProcedureComments", "AdmittingDiagnosesDescription",
    }
)

ID_VALUE_KEYWORDS = frozenset(
    {
        "PatientID", "OtherPatientIDs", "AccessionNumber", "StudyID",
        "ScheduledProcedureStepID", "PerformedProcedureStepID", "RequestedProcedureID",
    }
)

SIDECAR_IDENTIFIER_KEYS = frozenset(
    {
        "DeviceSerialNumber", "StationName", "InstitutionName", "InstitutionAddress",
        "InstitutionalDepartmentName", "PatientID", "PatientName", "AccessionNumber",
        "StudyID", "PatientBirthDate", "PatientSex", "PatientWeight", "PatientAge",
    }
)

# Sidecar keys that encode a calendar date or wall-clock time. dcm2niix/BIDS
# names these *Date, *DateTime, or <Event>Time; acquisition-parameter keys that
# end in "Time" (durations, not clock times) are explicitly excluded.
SIDECAR_DATE_PATTERN = re.compile(r"(Date|DateTime|Time)$")
SIDECAR_KEEP = frozenset(
    {
        "EchoTime", "RepetitionTime", "InversionTime", "TotalReadoutTime",
        "EffectiveEchoSpacing", "DwellTime", "ExposureTime", "TriggerTime",
        "DelayTime", "DelayAfterTrigger", "MixingTime", "RepetitionTimeExcitation",
        "RepetitionTimePreparation", "VolumeTiming", "SliceTiming",
        "AcquisitionDuration", "FrameTimesStart", "InversionTimes",
    }
)


def is_private_tag(tag: str) -> bool:
    """Return True for a '(gggg,eeee)' tag in an odd (private) group."""
    match = re.fullmatch(r"\(([0-9A-Fa-f]{4}),[0-9A-Fa-f]{4}\)", tag)
    return bool(match) and int(match.group(1), 16) % 2 == 1


def iter_elements(elements: dict):
    """Yield (tag, element) for every element, recursing into sequence items."""
    stack = list(elements.items())
    while stack:
        tag, element = stack.pop()
        yield tag, element
        value = element.get("value")
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    stack.extend((t, e) for t, e in item.items() if isinstance(e, dict))


def collect_id_values(docs: list[dict]) -> frozenset[str]:
    """Collect non-empty values of identifier keywords from dicom_headers documents."""
    values = set()
    for doc in docs:
        for _, element in iter_elements(doc["tags"]):
            value = element.get("value")
            if element.get("keyword") in ID_VALUE_KEYWORDS and isinstance(value, str) and value.strip():
                values.add(value.strip())
    return frozenset(values)


def removal_reason(tag: str, element: dict, id_values: frozenset[str] = frozenset()) -> str | None:
    """Return why a DICOM element must be removed, or None to keep it."""
    keyword = element.get("keyword")
    value = element.get("value")
    if isinstance(value, str) and value.strip() in id_values:
        return f"identifier-value:{keyword}"
    if element.get("vr") in DATE_TIME_VRS:
        return f"date/time:{keyword}"
    if keyword in DICOM_IDENTIFIER_KEYWORDS:
        return f"identifier:{keyword}"
    if is_private_tag(tag) or keyword is None:
        return "private/unknown"
    return None


def sidecar_removal_reason(key: str) -> str | None:
    """Return why a sidecar key must be removed, or None to keep it."""
    if key in SIDECAR_IDENTIFIER_KEYS:
        return f"identifier:{key}"
    if key not in SIDECAR_KEEP and SIDECAR_DATE_PATTERN.search(key):
        return f"date/time:{key}"
    return None


def strip_dicom_elements(elements: dict, removed: Counter, id_values: frozenset[str] = frozenset()) -> dict:
    """Return a copy of a tag->element mapping with targeted elements removed.

    Sequence (SQ) items are dicts of the same tag->element shape and are cleaned
    recursively. ``removed`` counts each removal reason.
    """
    cleaned = {}
    for tag, element in elements.items():
        reason = removal_reason(tag, element, id_values)
        if reason:
            removed[reason] += 1
            continue
        value = element.get("value")
        if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
            element = {**element, "value": [strip_dicom_elements(item, removed, id_values) for item in value]}
        cleaned[tag] = element
    return cleaned


def clean_dicom_header(doc: dict, removed: Counter, id_values: frozenset[str] = frozenset()) -> dict:
    """Return a dicom_headers JSON document with targeted elements removed."""
    return {**doc, "tags": strip_dicom_elements(doc["tags"], removed, id_values)}


def _strip_id_values(obj, removed: Counter, id_values: frozenset[str]):
    """Recursively drop dict keys whose string value is a known identifier value."""
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            if isinstance(value, str) and value.strip() in id_values:
                removed[f"identifier-value:{key}"] += 1
            else:
                out[key] = _strip_id_values(value, removed, id_values)
        return out
    if isinstance(obj, list):
        return [_strip_id_values(v, removed, id_values) for v in obj]
    return obj


def clean_sidecar(doc: dict, removed: Counter, id_values: frozenset[str] = frozenset()) -> dict:
    """Return a dcm2niix sidecar document with targeted keys removed."""
    cleaned = {}
    for key, value in doc.items():
        reason = sidecar_removal_reason(key)
        if reason:
            removed[reason] += 1
        else:
            cleaned[key] = value
    return _strip_id_values(cleaned, removed, id_values)


def find_remaining(doc: dict, kind: str, id_values: frozenset[str] = frozenset()) -> list[str]:
    """Independently list any targeted fields or identifier values still present."""
    found = [f"id-value:{m}" for m in re.findall(r'"([^"]+)"', json.dumps(doc)) if m in id_values]
    if kind == "sidecar":
        return found + [k for k in doc if sidecar_removal_reason(k)]
    return found + [
        element.get("keyword") or tag for tag, element in iter_elements(doc["tags"]) if removal_reason(tag, element)
    ]


def write_json_atomic(path: Path, doc: dict, indent: str | int, trailing_newline: bool) -> None:
    """Write JSON to a temp file in the same directory, then atomically replace."""
    text = json.dumps(doc, indent=indent, ensure_ascii=False) + ("\n" if trailing_newline else "")
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.chmod(tmp, path.stat().st_mode)
    os.replace(tmp, path)


def list_sessions(root: Path) -> list[Path]:
    """Find session directories (pyalfe/sub-*/ses-*) under a dataset root."""
    return sorted(p for p in root.glob("pyalfe/sub-*/ses-*") if p.is_dir())


def session_files(session: Path) -> dict[str, list[Path]]:
    """Return a session's dicom_headers and dcm2niix_sidecars JSON files."""
    return {
        "dicom_header": sorted(session.glob("dicom_headers/*.json")),
        "sidecar": sorted(session.glob("dcm2niix_sidecars/*.json")),
    }


def clean_dataset(root: Path, dry_run: bool) -> tuple[Counter, dict[Path, frozenset[str]]]:
    """Clean every metadata JSON under ``root``.

    Returns removal-reason counts and, per session, the identifier values that
    were targeted (needed by ``verify_dataset`` once the source fields are gone).
    """
    removed: Counter = Counter()
    ids_by_session: dict[Path, frozenset[str]] = {}
    n_files, n_changed = Counter(), Counter()
    for session in list_sessions(root):
        files = session_files(session)
        raws = {p: p.read_text(encoding="utf-8") for paths in files.values() for p in paths}
        id_values = collect_id_values([json.loads(raws[p]) for p in files["dicom_header"]])
        ids_by_session[session] = id_values
        for kind, paths in files.items():
            for path in paths:
                n_files[kind] += 1
                raw, before = raws[path], sum(removed.values())
                doc = json.loads(raw)
                cleaner = clean_dicom_header if kind == "dicom_header" else clean_sidecar
                cleaned = cleaner(doc, removed, id_values)
                if sum(removed.values()) > before:
                    n_changed[kind] += 1
                    if not dry_run:
                        indent = "\t" if raw.startswith("{\n\t") else 2
                        write_json_atomic(path, cleaned, indent, raw.endswith("\n"))
    log.info(f"Sessions: {len(ids_by_session)}; identifier values collected: {sum(map(len, ids_by_session.values()))}")
    for kind in n_files:
        log.info(f"{kind}: {n_changed[kind]}/{n_files[kind]} files {'would be ' if dry_run else ''}modified")
    return removed, ids_by_session


def verify_dataset(root: Path, ids_by_session: dict[Path, frozenset[str]]) -> int:
    """Re-read every metadata JSON and count targeted fields or ID values still present.

    Identifier values are checked against *all* sessions' values, not just the
    file's own session, to catch cross-session leakage.
    """
    all_ids = frozenset().union(*ids_by_session.values()) if ids_by_session else frozenset()
    n_remaining, n_checked = 0, Counter()
    for session in list_sessions(root):
        for kind, paths in session_files(session).items():
            for path in paths:
                n_checked[kind] += 1
                remaining = find_remaining(json.loads(path.read_text(encoding="utf-8")), kind, all_ids)
                if remaining:
                    n_remaining += len(remaining)
                    log.error(f"Targeted fields remain in {path}: {sorted(set(remaining))[:10]}")
    for kind, n in n_checked.items():
        log.info(f"Verified {n} {kind} files")
    return n_remaining


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_root", type=Path, help="Path to pcnsl-dataset_v1.0")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be removed without writing")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(Path(__file__).with_suffix(".log"), mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    log.info(f"Dataset root: {args.dataset_root.resolve()} (dry run: {args.dry_run})")
    removed, ids_by_session = clean_dataset(args.dataset_root, args.dry_run)
    log.info(f"Removed {sum(removed.values())} fields ({len(removed)} distinct reasons):")
    for reason, n in sorted(removed.items()):
        log.info(f"  {reason}: {n}")
    if args.dry_run:
        return 0
    n_remaining = verify_dataset(args.dataset_root, ids_by_session)
    if n_remaining:
        log.error(f"VERIFICATION FAILED: {n_remaining} targeted fields remain")
        return 1
    log.info("VERIFICATION PASSED: no date/time, identifier, or private fields remain")
    return 0


if __name__ == "__main__":
    sys.exit(main())
