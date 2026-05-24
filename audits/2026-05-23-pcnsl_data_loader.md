# Code Audit: pcnsl_data_loader.py
Date: 2026-05-23
Auditor: Michael Romano (assisted by Claude Code)

## Summary
- Functions audited: 33
- Critical findings: 0
- Important findings: 17
- Minor findings: 28

---

## Findings by Function

### Module-level

**[IMPORTANT | Design]** L16: `import numpy as np` — numpy is never used anywhere in the file. Dead import.

**[IMPORTANT | Design]** L189–220: `CLINICAL_DATA_CONFIG.has_accessions` — declared for all 6 entries but never read by any method. Dead configuration field that implies filtering behaviour that is not driven by it.

**[MINOR | Design]** L25–27: `ImageSpace` and `Modality` are both `Literal["FLAIR", "T1Post"]` — identical type aliases with different names. Their interchangeability (exploited in `load_image_with_mask`) is undocumented.

---

### `_load_json_series` (L62–85)

**[IMPORTANT | Security]** L62–85: `subject` and `session` strings are joined to a filesystem path without normalization or validation. A caller-supplied `subject` like `../../../etc` traverses outside `pyalfe_base`. No `.resolve()` or `startswith()` containment check.

**[MINOR | Security]** L75–84: All per-file exceptions silently swallowed (`except Exception: continue`). Parse failures — including malformed or malicious JSON — are invisible to the caller.

**[MINOR | Logic]** L80–81: `sequence = "_".join(parts[2:])` where `parts = json_path.stem.split("_")`. If the filename has fewer than 3 underscore-delimited parts, `sequence` becomes `""` with no warning.

---

### `parse_dicom_tag_json` (L93–152)

**[MINOR | Logic]** L107–110: `entry["value"]` accessed without checking `entry` is a dict or that `"value"` key exists. A malformed tag entry raises `TypeError`/`KeyError` that propagates uncaught.

**[MINOR | Logic]** L127–134: `rows` and `cols` are raw JSON values (often strings). Used directly in `round(rows * ps[0], 1)` without `int()`/`float()` conversion. The `try/except` at L136–139 only guards `VoxelVolume`, not the FOV lines — a type error here is unhandled.

**[MINOR | Logic]** L131–132: `ps[0]` and `ps[1]` used raw from JSON without `float()` conversion. If `PixelSpacing` is a list of strings, multiplication fails or produces garbage.

**[MINOR | Security]** L128: `f"{rows}x{cols}"` embeds unvalidated JSON values. If `rows`/`cols` is a nested object, `str()` serializes the entire structure into the output cell.

**[MINOR | Design]** L93–152 (complexity=16): Geometry derivation block (L122–150) is a distinct concern from tag extraction. Extracting it into a private helper would lower complexity and isolate the type-coercion issues.

---

### `parse_dcm2niix_sidecar` (L155–158)

**[MINOR | Security]** L155–158: Entire sidecar JSON returned raw. Any unexpected keys (PHI-like fields, large arrays) propagate into the caller's DataFrame without filtering.

---

### `AWSDataLoader.__init__` (L222–236)

**[MINOR | Security]** L222–236: `pyalfe_path` and `csv_path` stored without `.resolve()`. Symlinks and relative components are accepted and followed silently.

---

### `list_imaging_subjects` (L248–250)

**[MINOR | Design]** L248–250: Identical body to `list_subjects` — pure passthrough wrapper with no added behaviour. Either remove and use `list_subjects` directly, or document the semantic difference.

---

### `list_subjects_with_statistics` (L260–268)

**[IMPORTANT | Performance]** L260–268: Two-level glob with existence check per session on every call. No caching. Called by `load_statistics` on the default path every time.

---

### `get_subject_session_list` (L270–277)

**[IMPORTANT | Performance]** L270–277: Calls `list_imaging_subjects()` (glob) then `list_sessions()` (glob per subject) — 1 + 150 filesystem operations per call. No caching. This is the hot path for notebook startup.

---

### `load_statistics_single` (L285–313)

**[MINOR | Logic]** L303–307: When the `SummaryLesions` CSV first column is not `"Unnamed: 0"`, the transpose branch is silently skipped, leaving the DataFrame in column-per-metric orientation. When concatenated with transposed frames, column misalignment results silently.

---

### `load_statistics` (L315–356)

**[IMPORTANT | Logic]** L338–341: `zip(subjects, sessions)` silently truncates to the shorter list when lengths differ. No length-mismatch guard.

**[IMPORTANT | Logic]** L353–354: When `ignore_missing=True` and *all* subjects are missing files, raises `ValueError`. The parameter name implies missing files should be skipped, not that *all-missing* is fatal — this is surprising behaviour.

**[MINOR | Logic]** L341: `sessions` broadcast from a single string fans out silently to all subjects; asymmetric with single-`subjects` normalisation (which wraps in a list, not broadcasts).

**[MINOR | Performance]** L285–313: `df.T` on every `SummaryLesions` file creates an intermediate copy per subject. Negligible at 150 subjects but avoidable.

---

### `load_skullstripped_image` (L381–399)

**[IMPORTANT | Security]** L381–399: `sequence` is embedded in a glob pattern without sanitization. Glob metacharacters in a caller-supplied value alter the match set. First match loaded unconditionally.

**[MINOR | Logic]** L391–399: Multiple glob matches resolved silently by taking `matches[0]`. No warning about ambiguous matches.

**[MINOR | Performance]** L381–399: `path.glob(pattern)` materializes a full match list to take `matches[0]`. Filename is deterministic given inputs — direct path construction with `Path.exists()` would be faster and unambiguous.

---

### `load_skullstripped_images` (L401–414)

**[MINOR | Performance]** L401–414: Loads all NIfTI volumes unconditionally. `.nii.gz` decompresses into memory on load; callers needing one sequence pay the cost of loading all four.

---

### `load_lesion_mask` (L422–438)

**[IMPORTANT | Security]** L422–438: Same glob injection as `load_skullstripped_image`: `modality` inserted into glob pattern without sanitization. `Literal` type annotation not enforced at runtime.

**[MINOR | Logic]** L430–438: Multiple matches resolved silently by taking `matches[0]`.

**[MINOR | Performance]** L422–438: Same glob-to-take-first pattern as `load_skullstripped_image`; direct path construction would be faster and deterministic.

---

### `load_image_with_mask` (L440–451)

**[MINOR | Design]** L440–451: Passes `modality` (type `Modality`) directly as both `space: ImageSpace` and `sequence`. The coupling is implicit — extending either `Literal` without updating this method silently breaks it.

---

### `load_clinical_data` (L455–493)

**[MINOR | Design]** L485–491: `patientdurablekey` zfill(4) normalisation duplicated verbatim in `load_merged_data` (L550–552) and `load_aws_biopsy_and_diagnosis_dates` (L704). Three-way duplication.

**[IMPORTANT | Performance]** L485–491: When `filter_to_imaging_subjects=True`, calls `get_patient_accession_mapping()` (full filesystem scan). In `load_merged_data`'s loop over `clinical_types`, this fires once per type — up to 6 redundant scans.

---

### `get_patient_accession_mapping` (L505–518)

**[IMPORTANT | Performance]** L505–518: Calls `list_imaging_subjects()` (full glob scan) on every invocation. Called independently from `get_subject_session_list()` in `load_merged_data` — two separate identical scans in the same call.

**[MINOR | Logic]** L515–516: `str(int(subject_num)).zfill(4)` raises `ValueError` for non-numeric subject directory names (e.g. `sub-test`, `sub-backup`). No `try/except` — a single unexpected directory crashes the entire mapping.

**[MINOR | Security]** L515–516: Same unguarded `int()` call — a symlinked directory named `sub-../sensitive` would pass the glob but fail here with an opaque `ValueError`.

**[MINOR | Design]** L515–516: `str(int(subject_num)).zfill(4)` silently truncates leading zeros then re-pads. Subject numbers > 4 digits produce a longer-than-4-digit key, breaking the mapping.

---

### `load_merged_data` (L522–580)

**[IMPORTANT | Logic]** L539–540: When `csv_path=None` (imaging-only mode), `load_merged_data` still defaults `clinical_types=["demographics", "biopsy_and_diagnosis_dates"]` and calls `load_clinical_data`, which raises `ValueError`. No guard to skip clinical merging when `csv_path is None`.

**[IMPORTANT | Logic]** L554–560: Merging `has_accessions=True` types (e.g. `biopsy_and_diagnosis_dates`) on `patientdurablekey` produces a one-to-many join that row-multiplies imaging subjects. `has_accessions` is stored in config but never used to deduplicate or warn — the silent row explosion corrupts all subsequent merges for multi-visit patients.

**[IMPORTANT | Performance]** L522–580: Calls `get_subject_session_list()` and `get_patient_accession_mapping()` as separate methods, each performing an independent filesystem scan for the same subject list. Redundant double scan.

**[MINOR | Logic]** L562–564: Duplicate column cleanup assumes pandas appends a non-empty suffix. When `left_on` and `right_on` are the same column name (`patientdurablekey`), the suffix may be empty, leaving the duplicate column in place silently.

---

### `load_dicom_headers` (L592–619)

**[IMPORTANT | Logic]** L611–613: TR/TE/TI multiply by 1000 (s→ms). dcm2niix BIDS sidecars store these in seconds, so the conversion is correct per spec. However, `parse_dicom_tag_json` emits raw DICOM tag values for the same fields (which are in ms per DICOM standard). No cross-check or documentation makes the unit discrepancy between the two loading paths visible.

---

### `load_aws_dicom_headers` (L658–666)

**[MINOR | Design]** L658–666: Accepts `csv_path` parameter and passes it to `AWSDataLoader`, but `load_dicom_headers` never uses clinical data. A missing CSV directory raises `FileNotFoundError`, blocking DICOM-only use.

---

### `load_aws_dicom_geometry` (L669–685)

**[IMPORTANT | Design]** L669–685: Does not instantiate `AWSDataLoader`; duplicates subject-discovery logic (glob `sub-*` dirs) already in `AWSDataLoader.list_subjects`. Breaks the single-loader abstraction.

**[MINOR | Performance]** L669–685: Performs its own `sorted(pyalfe_base.glob("sub-*"))` rather than reusing `AWSDataLoader.list_subjects()`, causing a duplicate scan when called alongside other loader functions.

---

### `load_aws_biopsy_and_diagnosis_dates` (L688–707)

**[IMPORTANT | Design]** L688–707: Bypasses `AWSDataLoader.load_clinical_data` entirely — manually constructs the filepath, reads the CSV, and duplicates the patient-key zfill filtering logic already in `load_clinical_data` (L488–491) and `load_merged_data` (L550–552). Inconsistent with every other `load_aws_*` convenience function.
