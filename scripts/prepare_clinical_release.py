"""Prepare pcnsl-dataset_v1.0 clinical CSVs and subject IDs for release.

Steps (run once, on the dataset as exported):

1. Back up the subject to be removed (imaging folder), the original clinical CSV
   folder, and the old->new ID mapping into ``--backup-dir`` (kept OUTSIDE the
   dataset; the mapping links release IDs to the prior export).
2. Remove ``--remove-subject`` (default sub-0037: aged 4 at MRI, no lymphoma
   diagnosis code) from imaging and from every clinical CSV.
3. Renumber later subjects down by one so IDs stay contiguous: imaging folders
   and filename prefixes (``sub-XXXX_``) and ``patientdurablekey`` (the loader
   maps patientdurablekey N <-> sub-N, so both move together).
4. Drop sensitive / internal columns (``DROP_COLUMNS``) and the constant
   ``accessions`` column; add ``participant_id`` (sub-XXXX) as the first column.
5. Convert M/D/YY dates to ISO 8601 (YYYY-MM-DD). Where an unambiguous 8-digit
   integer key exists (diagnosis_history ``StartDateKey``/``EndDateKey``) it is
   used as the source of truth. Otherwise two-digit years are expanded with the
   rule "a date after ``REFERENCE_DATE`` belongs to the previous century", which
   is cross-checked against the precomputed day differences in
   biopsy_and_diagnosis_dates. Dates later than ``REFERENCE_DATE`` that remain
   (source-data errors) are kept as-is and logged as warnings.
6. Update data_dictionary_clinical.csv accordingly and rename the folder
   ``csvs_for_amazon_anonymized/`` -> ``clinical/``.
7. Verify the result by re-reading everything from disk.

Usage
-----
    uv run python scripts/prepare_clinical_release.py pcnsl-dataset_v1.0 \\
        --backup-dir ../pcnsl_release --dry-run
    uv run python scripts/prepare_clinical_release.py pcnsl-dataset_v1.0 \\
        --backup-dir ../pcnsl_release
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import sys
import tarfile
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

OLD_CSV_DIR = "csvs_for_amazon_anonymized"
NEW_CSV_DIR = "clinical"
DICTIONARY = "data_dictionary_clinical.csv"
DATA_CSVS = [
    "biopsy_and_diagnosis_dates.csv",
    "demographics.csv",
    "diagnosis_history.csv",
    "medication_list_administered.csv",
    "medication_list_ordered.csv",
    "ucsf500_mutations.csv",
]
DROP_COLUMNS = {
    "demographics.csv": ["SexualOrientation", "PreferredLanguage"],
    "medication_list_administered.csv": ["AdministeringEmployeeKey"],
    "diagnosis_history.csv": ["DiagnosisEventKey", "DiagnosisKey"],
    "biopsy_and_diagnosis_dates.csv": ["accessions"],
    "ucsf500_mutations.csv": ["accessions"],
}
MDY_DATE_COLUMNS = {
    "biopsy_and_diagnosis_dates.csv": ["BiopsyDate", "DiagnosisDate", "ImageDate"],
    "demographics.csv": ["BirthDate", "Deathdate"],
    "diagnosis_history.csv": ["StartDateKeyValue", "EndDateKeyValue"],
}
DATE_KEY_PAIRS = {"StartDateKeyValue": "StartDateKey", "EndDateKeyValue": "EndDateKey"}
REFERENCE_DATE = pd.Timestamp("2026-11-11")
SUBJECT_KEY = "patientdurablekey"


# --- ID mapping ---------------------------------------------------------------


def build_id_mapping(n_subjects: int, remove: int) -> dict[int, int | None]:
    """Map old subject numbers 1..n to new numbers; the removed one maps to None."""
    return {old: (None if old == remove else old - (old > remove)) for old in range(1, n_subjects + 1)}


def sub_id(n: int) -> str:
    """Format a subject number as a BIDS subject ID (sub-0001)."""
    return f"sub-{n:04d}"


# --- dates --------------------------------------------------------------------


def mdy_to_iso(values: pd.Series, reference: pd.Timestamp = REFERENCE_DATE) -> pd.Series:
    """Convert M/D/YY strings to YYYY-MM-DD; blanks stay blank, bad values raise.

    pandas maps two-digit years 69-99 to 19xx and 00-68 to 20xx; any parsed date
    later than ``reference`` is moved back 100 years (e.g. birth '1/2/47').
    """
    blank = values.str.strip().eq("")
    parsed = pd.to_datetime(values.where(~blank), format="%m/%d/%y", errors="raise")
    future = parsed > reference
    parsed = parsed.where(~future, parsed - pd.DateOffset(years=100))
    return parsed.dt.strftime("%Y-%m-%d").where(~blank, "")


# --- clinical tables ----------------------------------------------------------


def transform_clinical_table(name: str, df: pd.DataFrame, mapping: dict[int, int | None]) -> pd.DataFrame:
    """Apply subject removal/renumbering, column drops, ISO dates, participant_id.

    Parameters
    ----------
    df : all-string DataFrame (read with dtype=str, keep_default_na=False)
        Must contain ``patientdurablekey`` (zero-padded subject number).

    Returns
    -------
    DataFrame with ``participant_id`` first, removed subject's rows dropped,
    ``patientdurablekey`` renumbered, DROP_COLUMNS removed and M/D/YY dates ISO.
    """
    new_num = df[SUBJECT_KEY].astype(int).map(mapping)
    unknown = df.loc[~df[SUBJECT_KEY].astype(int).isin(mapping), SUBJECT_KEY].unique()
    if len(unknown):
        raise ValueError(f"{name}: patientdurablekey values outside 1..{len(mapping)}: {unknown[:5]}")
    n_before = len(df)
    df = df.loc[new_num.notna()].copy()
    new_num = new_num.loc[df.index].astype(int)
    log.info(f"{name}: removed subject rows {n_before} -> {len(df)} ({n_before - len(df)} dropped)")

    df[SUBJECT_KEY] = new_num.map(lambda n: f"{n:04d}")
    df = df.drop(columns=DROP_COLUMNS.get(name, []))
    for col in MDY_DATE_COLUMNS.get(name, []):
        df[col] = mdy_to_iso(df[col])
        key_col = DATE_KEY_PAIRS.get(col)
        if key_col in df.columns:
            has_key = df[key_col].str.fullmatch(r"\d{8}") & df[col].ne("")
            from_key = df[key_col].str.replace(r"^(\d{4})(\d{2})(\d{2})$", r"\1-\2-\3", regex=True)
            differs = has_key & (from_key != df[col])
            for i in df.index[differs]:
                log.warning(f"{name}: {col} {df.at[i, col]} (2-digit-year rule) replaced by {key_col} {from_key[i]}")
            df[col] = df[col].where(~has_key, from_key)
        future = df[col].ne("") & (df[col] > REFERENCE_DATE.strftime("%Y-%m-%d"))
        if future.any():
            log.warning(f"{name}: {future.sum()} {col} values after {REFERENCE_DATE.date()} kept as in source "
                        f"(source-data error): {df.loc[future, col].tolist()[:5]}")
    df.insert(0, "participant_id", new_num.map(sub_id))
    log.info(f"{name}: dropped {DROP_COLUMNS.get(name, [])}; ISO dates {MDY_DATE_COLUMNS.get(name, [])}")
    return df.reset_index(drop=True)


def check_date_consistency(name: str, df: pd.DataFrame) -> list[str]:
    """Cross-check converted dates against integer date keys and day differences."""
    problems = []
    for value_col, key_col in DATE_KEY_PAIRS.items():
        if value_col in df.columns and key_col in df.columns:
            has = df[value_col].ne("") & df[key_col].str.fullmatch(r"\d{8}")
            bad = (df.loc[has, value_col].str.replace("-", "") != df.loc[has, key_col]).sum()
            if bad:
                problems.append(f"{name}: {bad} rows where {value_col} != {key_col}")
    for event in ["Biopsy", "Diagnosis"]:
        diff_col, event_col = f"{event}DateMinusImageDate", f"{event}Date"
        if diff_col in df.columns:
            has = df[diff_col].ne("") & df[event_col].ne("") & df["ImageDate"].ne("")
            days = (pd.to_datetime(df.loc[has, event_col]) - pd.to_datetime(df.loc[has, "ImageDate"])).dt.days
            bad = (days != df.loc[has, diff_col].astype(float).astype(int)).sum()
            if bad:
                problems.append(f"{name}: {bad} rows where {diff_col} disagrees with ISO dates")
    return problems


def update_dictionary(dictionary: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Drop rows for removed columns, add participant_id rows, refresh changed examples."""
    dropped = {(f, c) for f, cols in DROP_COLUMNS.items() for c in cols}
    keep = [(f, c) not in dropped for f, c in zip(dictionary["spreadsheet"], dictionary["column name"])]
    out = dictionary.loc[keep].copy()

    def examples(f: str, c: str) -> str:
        vals = tables[f][c].loc[tables[f][c].ne("")].drop_duplicates().head(3)
        return "; ".join(vals)

    for i, row in out.iterrows():
        f, c = row["spreadsheet"], row["column name"]
        if c in MDY_DATE_COLUMNS.get(f, []):
            out.at[i, "description"] = re.sub(r"\(M/D+/YY\)|in M/D+/YY format", "(YYYY-MM-DD)", row["description"])
            out.at[i, "example values"] = examples(f, c)
        elif c == SUBJECT_KEY:
            out.at[i, "description"] = "Anonymized patient identifier; equals the number in participant_id"
            out.at[i, "example values"] = examples(f, c)

    new_rows = pd.DataFrame(
        {
            "spreadsheet": DATA_CSVS,
            "column name": "participant_id",
            "description": "BIDS subject ID linking to imaging folder pyalfe/<participant_id>/",
            "example values": [examples(f, "participant_id") for f in DATA_CSVS],
        }
    )
    out = pd.concat([out, new_rows], ignore_index=True)
    order = {f: i for i, f in enumerate(DATA_CSVS)}
    out["_f"] = out["spreadsheet"].map(order)
    out["_p"] = out["column name"].ne("participant_id")
    return out.sort_values(["_f", "_p"], kind="stable").drop(columns=["_f", "_p"]).reset_index(drop=True)


# --- imaging ------------------------------------------------------------------


def plan_imaging_renames(pyalfe: Path, mapping: dict[int, int | None]) -> list[tuple[Path, Path]]:
    """List (old_dir, new_dir) renames in ascending order (safe after removal)."""
    plan = []
    for old, new in sorted(mapping.items()):
        if new is not None and new != old:
            plan.append((pyalfe / sub_id(old), pyalfe / sub_id(new)))
    return plan


def rename_subject(old_dir: Path, new_dir: Path) -> int:
    """Rename every 'sub-OLD_' filename prefix inside old_dir, then the folder itself."""
    old_prefix, new_prefix = f"{old_dir.name}_", f"{new_dir.name}_"
    n = 0
    for path in sorted(old_dir.rglob(f"{old_prefix}*")):
        path.rename(path.with_name(new_prefix + path.name[len(old_prefix):]))
        n += 1
    if new_dir.exists():
        raise FileExistsError(new_dir)
    old_dir.rename(new_dir)
    return n


# --- verification ---------------------------------------------------------------


def verify_release(root: Path, n_expected: int) -> list[str]:
    """Independently re-read the released dataset and list any problems."""
    problems = []
    pyalfe, csv_dir = root / "pyalfe", root / NEW_CSV_DIR
    subjects = sorted(p.name for p in pyalfe.glob("sub-*") if p.is_dir())
    if subjects != [sub_id(i) for i in range(1, n_expected + 1)]:
        problems.append(f"imaging subjects are not contiguous sub-0001..{sub_id(n_expected)} (found {len(subjects)})")
    for subject in subjects:
        for f in (pyalfe / subject).rglob("sub-*"):
            if f.is_file() and not f.name.startswith(f"{subject}_"):
                problems.append(f"filename prefix mismatch: {f}")
    if (root / OLD_CSV_DIR).exists() or not csv_dir.exists():
        problems.append("CSV folder not renamed")
        return problems
    forbidden = {c for cols in DROP_COLUMNS.values() for c in cols}
    mdy = re.compile(r"^\d{1,2}/\d{1,2}/\d{2}$")
    for name in DATA_CSVS:
        df = pd.read_csv(csv_dir / name, dtype=str, keep_default_na=False)
        if df.columns[0] != "participant_id":
            problems.append(f"{name}: participant_id is not the first column")
        if forbidden & set(df.columns):
            problems.append(f"{name}: forbidden columns present {sorted(forbidden & set(df.columns))}")
        if (df["participant_id"] != "sub-" + df[SUBJECT_KEY]).any():
            problems.append(f"{name}: participant_id does not match patientdurablekey")
        if not set(df["participant_id"]) <= set(subjects):
            problems.append(f"{name}: participant_id without imaging folder")
        n_mdy = sum(df[c].str.match(mdy).sum() for c in df.columns)
        if n_mdy:
            problems.append(f"{name}: {n_mdy} M/D/YY values remain")
        problems += check_date_consistency(name, df)
    demo = pd.read_csv(csv_dir / "demographics.csv", dtype=str, keep_default_na=False)
    if demo["participant_id"].nunique() != n_expected or len(demo) != n_expected:
        problems.append(f"demographics.csv: expected {n_expected} unique participants")
    return problems


# --- main -----------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--backup-dir", type=Path, required=True)
    parser.add_argument("--remove-subject", default="sub-0037")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(Path(__file__).with_suffix(".log"), mode="w"), logging.StreamHandler(sys.stdout)],
    )
    root, pyalfe, old_csv = args.dataset_root, args.dataset_root / "pyalfe", args.dataset_root / OLD_CSV_DIR
    if not old_csv.exists() or (root / NEW_CSV_DIR).exists():
        log.error("Dataset is not in its original state (clinical/ exists or CSV folder missing); aborting")
        return 1
    n_subjects = len([p for p in pyalfe.glob("sub-*") if p.is_dir()])
    remove = int(args.remove_subject.removeprefix("sub-"))
    mapping = build_id_mapping(n_subjects, remove)
    log.info(f"{n_subjects} subjects; removing {args.remove_subject}; {n_subjects - 1} remain (dry run: {args.dry_run})")

    tables, problems = {}, []
    for name in DATA_CSVS:
        raw = pd.read_csv(old_csv / name, dtype=str, keep_default_na=False)
        log.info(f"Loaded {name}: {raw.shape[0]} rows x {raw.shape[1]} cols")
        tables[name] = transform_clinical_table(name, raw, mapping)
        problems += check_date_consistency(name, tables[name])
    if problems:
        for p in problems:
            log.error(p)
        return 1
    log.info("Date conversion cross-checks passed (integer date keys and day differences)")
    dictionary = update_dictionary(pd.read_csv(old_csv / DICTIONARY, dtype=str, keep_default_na=False), tables)
    plan = plan_imaging_renames(pyalfe, mapping)
    log.info(f"Imaging: delete {args.remove_subject}; rename {len(plan)} subject folders ({plan[0][0].name}->{plan[0][1].name} ... {plan[-1][0].name}->{plan[-1][1].name})")
    if args.dry_run:
        return 0

    args.backup_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.backup_dir / f"BACKUP_removed_{args.remove_subject}_imaging.tar.gz", "w:gz") as tar:
        tar.add(pyalfe / args.remove_subject, arcname=args.remove_subject)
    with tarfile.open(args.backup_dir / f"BACKUP_original_{OLD_CSV_DIR}.tar.gz", "w:gz") as tar:
        tar.add(old_csv, arcname=OLD_CSV_DIR)
    pd.DataFrame(
        [(sub_id(o), sub_id(n) if n else "REMOVED") for o, n in mapping.items()], columns=["old_id", "new_id"]
    ).to_csv(args.backup_dir / "subject_id_mapping_v1.0_export_to_release.csv", index=False)
    log.info(f"Backups written to {args.backup_dir.resolve()}")

    shutil.rmtree(pyalfe / args.remove_subject)
    n_files = sum(rename_subject(old, new) for old, new in plan)
    log.info(f"Deleted {args.remove_subject}; renamed {len(plan)} folders and {n_files} files")
    for name, df in tables.items():
        df.to_csv(old_csv / name, index=False)
        log.info(f"Wrote {name}: {df.shape[0]} rows x {df.shape[1]} cols")
    dictionary.to_csv(old_csv / DICTIONARY, index=False)
    old_csv.rename(root / NEW_CSV_DIR)
    log.info(f"Wrote {DICTIONARY} ({len(dictionary)} rows); renamed {OLD_CSV_DIR}/ -> {NEW_CSV_DIR}/")

    problems = verify_release(root, n_subjects - 1)
    for p in problems:
        log.error(p)
    if problems:
        log.error(f"VERIFICATION FAILED ({len(problems)} problems)")
        return 1
    log.info("VERIFICATION PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
