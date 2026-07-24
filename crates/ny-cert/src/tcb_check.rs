// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Drift guard: `proofs/tcb.json` (the machine-readable TCB manifest) versus its
//! human mirror in `SPEC.md`.
//!
//! `tcb.json`'s `_doc` promises the manifest is *"Drift-guarded against `SPEC.md`
//! by `ny_cert::tcb_check`"*, and `SPEC.md` §"TCB manifest" restates the same rows
//! as a `| Row | Status | Retired by |` table. This module makes that promise real:
//! it parses both representations and pins them together so a human edit to one that
//! is not mirrored in the other fails the test suite — the same role [`crate::cite_check`]
//! plays for citation grounding, with the same `src/` module + `#[cfg(test)]`
//! layout and the same `CARGO_MANIFEST_DIR` path convention.
//!
//! The pin is deliberately *formatting-tolerant, word-strict*: statuses are compared
//! under [`normalize_status`], which collapses every run of non-alphanumeric bytes to
//! a single `-`. That absorbs the harmless split between `tcb.json`'s hyphenated
//! `closed-by-construction-for-emitted-certs` and `SPEC.md`'s spaced
//! `closed-by-construction for emitted certs`, while still catching any change to the
//! actual *words*.

use std::path::{Path, PathBuf};

/// Exact `status` strings a manifest row may carry (allow-list — new statuses are
/// added here deliberately, never silently). These are the canonical, hyphenated
/// `tcb.json` spellings.
pub const ALLOWED_STATUSES: &[&str] = &[
    "closed-by-construction-for-emitted-certs",
    "kernel-checked",
    "modulo-cite",
];

/// Clean theorems the `clean_kernel` row asserts are kernel-checked and sorry-free.
/// This is the citation checker's canonical registry, not a second partial copy.
pub use crate::cite_check::CITED_THEOREMS;

/// Absolute path to the machine-readable TCB manifest (`proofs/tcb.json`).
#[must_use]
pub fn tcb_manifest_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("proofs/tcb.json")
}

/// Absolute path to the human specification whose §"TCB manifest" mirrors the JSON.
#[must_use]
pub fn spec_md_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("SPEC.md")
}

/// Absolute path to NY's local Lean overlay, where a row's NY-owned discharge-path
/// `*.lean` artifacts are expected to live. Clean citation theorems resolve through
/// [`crate::cite_check::clean_corpus_root`] instead.
#[must_use]
pub fn ny_overlay_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("proofs/lean/NyProof")
}

/// One data row parsed from the `SPEC.md` §"TCB manifest" mirror table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpecTcbRow {
    /// Manifest row id (the first backtick-quoted token of the `Row` cell).
    pub id: String,
    /// The `Status` cell, verbatim (compared under [`normalize_status`]).
    pub status: String,
    /// The `Retired by` cell, verbatim (scanned for discharge `*.lean` artifacts).
    pub retired_by: String,
}

/// Normalize a status for word-strict, formatting-tolerant comparison: lowercase and
/// collapse every maximal run of non-alphanumeric characters (hyphens, spaces,
/// underscores, backticks, asterisks, …) to a single `-`, with no leading or trailing
/// separator.
///
/// This is the crux of the guard: it makes `tcb.json`'s
/// `closed-by-construction-for-emitted-certs` equal to `SPEC.md`'s
/// `closed-by-construction for emitted certs`, while any change to the words still
/// diverges.
///
/// Deliberately CHAR-based (`char::is_alphanumeric`), not byte-based: a byte-level
/// ASCII scan would treat non-ASCII letters/digits as separators, so a SPEC.md
/// status drifting by a Unicode word character (e.g. a Cyrillic homoglyph or
/// superscript letter) would silently normalize back to the canonical form instead
/// of failing the pin — a weakening of the guard's fail-loud direction. The
/// verifier's outstanding unknown here is a false counterexample from its `chars()`
/// model (it admits values above `char::MAX`); fix that in the model, not by
/// weakening this guard.
#[must_use]
pub fn normalize_status(s: &str) -> String {
    // `String::new()` (not `with_capacity(s.len())`): the capacity hint on an
    // unbounded input carries a hardened allocation obligation the model
    // cannot bound; statuses are a few dozen bytes, growth cost is nil.
    let mut out = String::new();
    let mut pending_sep = false;
    for c in s.chars() {
        if c.is_alphanumeric() {
            if pending_sep && !out.is_empty() {
                out.push('-');
            }
            pending_sep = false;
            out.push(c.to_ascii_lowercase());
        } else {
            pending_sep = true;
        }
    }
    out
}

/// `| a | b | c |` -> `["a", "b", "c"]`: trim the line, strip the bounding
/// pipes, split on `|`, trim each cell.
fn row_cells(line: &str) -> Vec<String> {
    // Explicit Vec::new()+push (not `.collect()`): the split count is
    // input-derived and the verifier cannot bound it, so a bulk `.collect()`
    // raises an UnboundedAllocation obligation. The loop has no bulk-alloc
    // obligation at all — identical elements and order.
    let mut cells: Vec<String> = Vec::new();
    for c in line.trim().trim_matches('|').split('|') {
        cells.push(c.trim().to_owned());
    }
    cells
}

/// The first backtick-quoted token of a cell:
/// `` `float_adequacy` (`R_float ⊑ R_real`) `` -> `Some("float_adequacy")`.
fn first_backtick(cell: &str) -> Option<String> {
    // `find` returned the byte index of a 1-byte '`' match, so `open` is
    // <= cell.len() and on a char boundary — but that is a postcondition of
    // `str::find` the verifier does not model, so use the total `get` forms
    // (the `?`-on-None arms are unreachable) instead of panicking slices.
    let open = cell.find('`')?.checked_add(1)?;
    let rest = cell.get(open..)?;
    let close = rest.find('`')?;
    let token = rest.get(..close)?.trim();
    (!token.is_empty()).then(|| token.to_owned())
}

/// Why the `SPEC.md` §"TCB manifest" mirror table failed to parse. Every
/// variant is drift by definition: the drift-guard callers `?`-propagate the
/// parse error, turning each into the same fail-loud test failure the old
/// panics produced.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SpecTcbError {
    /// `SPEC.md` has no `## TCB manifest` heading.
    #[error("SPEC.md is missing its `## TCB manifest` section")]
    MissingSection,
    /// The `Row | Status | Retired by` header was renamed or reordered.
    #[error("SPEC.md TCB manifest table header changed: {0:?}")]
    HeaderChanged(Vec<String>),
    /// No header row was found before the section ended.
    #[error("SPEC.md TCB manifest table header row not found")]
    HeaderRowNotFound,
}

/// The line beginning at byte `pos` of `spec`, plus the cursor for the line
/// after it — one step of a manual `str::lines()` walk. Returns `None` once
/// `pos` is at or past the end (so a terminating `\n` yields no trailing empty
/// line, per `str::lines`). Splits on `b'\n'` and strips one trailing `b'\r'`
/// from a `\n`-terminated piece only (a lone `\r` on the final, unterminated
/// segment is kept), byte-for-byte the `str::lines` verdict.
///
/// Total: a `get`-checked forward scan. `pos` and every slice boundary sit on
/// ASCII bytes (start of input, a `\n`/`\r`, or the end), which are char
/// boundaries, so the `spec.get(..)` is `Some`; its `?`-on-`None` arm is
/// unreachable, and stopping the walk there is the safe direction (fewer rows
/// parsed can only fail the drift guard louder, never ground a bogus pin).
fn next_line(spec: &str, pos: usize) -> Option<(&str, usize)> {
    if pos >= spec.len() {
        return None;
    }
    let bytes = spec.as_bytes();
    let mut end = pos;
    while let Some(&b) = bytes.get(end) {
        if b == b'\n' {
            break;
        }
        end = end.saturating_add(1);
    }
    let mut line_end = end;
    // Strip the `\r` of a `\r\n` — only when the piece is `\n`-terminated,
    // mirroring `str::lines` (strip `\n`, then strip `\r`). Literal patterns
    // (not `==` on `Option<&u8>`): a match compiles to primitive byte compares.
    // Literal `Some(&b'\n')`/`Some(&b'\r')` patterns (not `==` on `Option<&u8>`):
    // primitive byte compares, keeping the absent `Option<&u8>` `PartialEq` impl
    // off the verified path — identical verdict. Covers the nested `if let` too.
    #[allow(clippy::equatable_if_let)]
    if let Some(&b'\n') = bytes.get(end) {
        if line_end > pos {
            if let Some(&b'\r') = bytes.get(line_end.saturating_sub(1)) {
                line_end = line_end.saturating_sub(1);
            }
        }
    }
    let line = spec.get(pos..line_end)?;
    Some((line, end.saturating_add(1)))
}

/// Parse the mirror table under `SPEC.md` §"TCB manifest" into rows.
///
/// The scan is section-scoped: `SPEC.md` also carries an unrelated
/// `| … | Status | … |` checker-status table, so parsing starts only after the
/// `## TCB manifest` heading and stops at the next `## ` heading. The `Row | Status |
/// Retired by` header must be present with those exact column names.
///
/// # Errors
/// Returns [`SpecTcbError`] when the section, header row, or column names are
/// missing or renamed — each is SPEC drift by definition (previously a panic;
/// the drift-guard callers `?`-propagate the parse error into a test failure,
/// keeping the fail-loud behavior while staying panic-free).
pub fn spec_tcb_rows(spec: &str) -> Result<Vec<SpecTcbRow>, SpecTcbError> {
    // Manual byte-cursor line walk via [`next_line`] (not `spec.lines()` +
    // `by_ref()`): direct calls to the free fn resolve to a bundled, verified
    // body, whereas the `Lines`/`&mut Lines` `Iterator::next` adapters minted
    // absent-callee obligations. `pos` is advanced before each body runs, so
    // after the heading `break` the row loop below resumes on the line after
    // the matched heading — exactly the cursor `by_ref()` preserved.
    let mut pos = 0usize;
    let mut saw_section = false;
    while let Some((l, next)) = next_line(spec, pos) {
        pos = next;
        if l.trim_start().starts_with("## TCB manifest") {
            saw_section = true;
            break;
        }
    }
    if !saw_section {
        return Err(SpecTcbError::MissingSection);
    }

    let mut rows = Vec::new();
    let mut saw_header = false;
    while let Some((line, next)) = next_line(spec, pos) {
        pos = next;
        let t = line.trim_start();
        if t.starts_with("## ") {
            break; // reached the next section: the table is over
        }
        if !t.starts_with('|') {
            continue;
        }
        let cells = row_cells(t);
        if !saw_header {
            if cells.first().is_some_and(|c| c.eq_ignore_ascii_case("Row")) {
                let header_ok = cells
                    .get(1)
                    .is_some_and(|c| c.eq_ignore_ascii_case("Status"))
                    && cells
                        .get(2)
                        .is_some_and(|c| c.eq_ignore_ascii_case("Retired by"));
                if !header_ok {
                    return Err(SpecTcbError::HeaderChanged(cells));
                }
                saw_header = true;
            }
            continue;
        }
        // The `|---|---|` separator row has no backtick token in cell 0: skip it.
        let Some(id) = cells.first().and_then(|c| first_backtick(c)) else {
            continue;
        };
        rows.push(SpecTcbRow {
            id,
            status: cells.get(1).cloned().unwrap_or_else(String::new),
            retired_by: cells.get(2).cloned().unwrap_or_else(String::new),
        });
    }
    if !saw_header {
        return Err(SpecTcbError::HeaderRowNotFound);
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cite_check::contains_token;
    use std::collections::BTreeMap;

    // All fallible helpers/tests below return `Result<_, String>` and `?`-propagate
    // instead of `.expect(..)`-panicking: an unreadable/malformed manifest or SPEC
    // is an `Err`, which the test harness reports as a FAILURE — same fail-loud
    // outcome, no may-panic boundary for the strict verifier.

    fn manifest() -> Result<serde_json::Value, String> {
        let raw = std::fs::read_to_string(tcb_manifest_path())
            .map_err(|e| format!("read tcb.json: {e}"))?;
        serde_json::from_str(&raw).map_err(|e| format!("parse tcb.json: {e}"))
    }

    fn spec() -> Result<String, String> {
        std::fs::read_to_string(spec_md_path()).map_err(|e| format!("read SPEC.md: {e}"))
    }

    /// `id -> status` for every manifest row.
    fn manifest_id_status(v: &serde_json::Value) -> Result<BTreeMap<String, String>, String> {
        v["rows"]
            .as_array()
            .ok_or("rows array")?
            .iter()
            .map(|r| {
                Ok((
                    r["id"].as_str().ok_or("row id")?.to_owned(),
                    r["status"].as_str().ok_or("row status")?.to_owned(),
                ))
            })
            .collect()
    }

    /// The single manifest row with the given id.
    fn manifest_row<'a>(
        v: &'a serde_json::Value,
        id: &str,
    ) -> Result<&'a serde_json::Value, String> {
        v["rows"]
            .as_array()
            .ok_or("rows array")?
            .iter()
            .find(|r| r["id"].as_str() == Some(id))
            .ok_or_else(|| format!("manifest has no row `{id}`"))
    }

    /// I1 — the manifest is well-formed: schema_version 1, at least one row, every row
    /// carries non-empty string values for all required fields, and ids are unique.
    #[test]
    fn i1_manifest_well_formed() -> Result<(), String> {
        let v = manifest()?;
        assert_eq!(
            v["schema_version"].as_i64(),
            Some(1),
            "schema_version must be 1"
        );
        let rows = v["rows"].as_array().ok_or("rows must be an array")?;
        assert!(!rows.is_empty(), "manifest has no rows");
        let mut seen = std::collections::BTreeSet::new();
        for r in rows {
            for key in [
                "id",
                "claim",
                "status",
                "grounding",
                "residual_tcb",
                "discharge_path",
            ] {
                let s = r[key]
                    .as_str()
                    .unwrap_or_else(|| panic!("row missing string field `{key}`: {r}"));
                assert!(!s.trim().is_empty(), "row field `{key}` is empty: {r}");
            }
            let id = r["id"].as_str().unwrap().to_owned();
            assert!(seen.insert(id.clone()), "duplicate manifest row id `{id}`");
        }
        Ok(())
    }

    /// I2 — every manifest status is in the `ALLOWED_STATUSES` allow-list.
    #[test]
    fn i2_statuses_in_allowlist() -> Result<(), String> {
        for (id, status) in manifest_id_status(&manifest()?)? {
            assert!(
                ALLOWED_STATUSES.contains(&status.as_str()),
                "row `{id}` status `{status}` is not in ALLOWED_STATUSES {ALLOWED_STATUSES:?}"
            );
        }
        Ok(())
    }

    /// I3 — the `SPEC.md` §"TCB manifest" table parses: the section, header, and one
    /// row per manifest id are all present, each with a non-empty status cell.
    #[test]
    fn i3_spec_table_parses() -> Result<(), String> {
        let rows = spec_tcb_rows(&spec()?)
            .map_err(|e| format!("SPEC.md TCB manifest table parses: {e}"))?;
        assert!(
            !rows.is_empty(),
            "SPEC.md TCB manifest table parsed to no rows"
        );
        for r in &rows {
            assert!(!r.id.trim().is_empty(), "parsed a row with an empty id");
            assert!(
                !r.status.trim().is_empty(),
                "SPEC.md row `{}` has an empty status cell",
                r.id
            );
        }
        Ok(())
    }

    /// I4 — the `id -> normalized-status` maps agree in both directions. `BTreeMap`
    /// equality checks the key sets (so a row added/removed/renamed on either side
    /// fails) and the values (so a per-row status change fails).
    #[test]
    fn i4_id_status_maps_equal_both_ways() -> Result<(), String> {
        let json_map: BTreeMap<String, String> = manifest_id_status(&manifest()?)?
            .into_iter()
            .map(|(id, st)| (id, normalize_status(&st)))
            .collect();
        let spec_map: BTreeMap<String, String> = spec_tcb_rows(&spec()?)
            .map_err(|e| format!("SPEC.md TCB manifest table parses: {e}"))?
            .into_iter()
            .map(|r| (r.id, normalize_status(&r.status)))
            .collect();
        assert_eq!(
            json_map, spec_map,
            "tcb.json and SPEC.md disagree on the id->status mapping"
        );
        Ok(())
    }

    /// I5 — `normalize_status` collapses non-alphanumeric runs to `-`, so the
    /// hyphenated `tcb.json` spelling equals the spaced `SPEC.md` spelling while a
    /// word change still diverges.
    #[test]
    fn i5_normalize_status_is_formatting_tolerant() {
        assert_eq!(
            normalize_status("closed-by-construction-for-emitted-certs"),
            normalize_status("closed-by-construction for emitted certs"),
            "hyphen vs space spellings must normalize equal"
        );
        assert_eq!(
            normalize_status("  Kernel-Checked  "),
            "kernel-checked",
            "leading/trailing separators trimmed, mixed case lowered"
        );
        assert_eq!(
            normalize_status("`modulo-cite`"),
            "modulo-cite",
            "backticks are separators, not content"
        );
        // Word-level change must NOT normalize equal.
        assert_ne!(
            normalize_status("closed-by-construction-for-emitted-certs"),
            normalize_status("closed-by-assumption-for-emitted-certs"),
            "a changed word must remain distinguishable"
        );
    }

    /// I6 — every `CITED_THEOREMS` name appears, as a boundary-safe token, in the
    /// `clean_kernel` row's `grounding` prose.
    #[test]
    fn i6_cited_theorems_named_in_clean_kernel_grounding() -> Result<(), String> {
        let v = manifest()?;
        let grounding = manifest_row(&v, "clean_kernel")?["grounding"]
            .as_str()
            .ok_or("clean_kernel grounding")?;
        for thm in CITED_THEOREMS {
            assert!(
                contains_token(grounding.as_bytes(), thm.as_bytes()),
                "clean_kernel grounding no longer names cited theorem `{thm}`"
            );
        }
        Ok(())
    }

    /// I7 — every `*.lean` artifact named in a row's `discharge_path` exists in NY's
    /// local overlay and is mirrored in that row's `SPEC.md` "Retired by" cell.
    #[test]
    fn i7_referenced_lean_modules_exist() -> Result<(), String> {
        let v = manifest()?;
        let retired_by: BTreeMap<String, String> = spec_tcb_rows(&spec()?)
            .map_err(|e| format!("SPEC.md TCB manifest table parses: {e}"))?
            .into_iter()
            .map(|r| (r.id, r.retired_by))
            .collect();
        let corpus = ny_overlay_root();
        for r in v["rows"].as_array().ok_or("rows array")? {
            let id = r["id"].as_str().ok_or("row id")?;
            let discharge = r["discharge_path"].as_str().ok_or("discharge_path")?;
            let cell = retired_by.get(id).cloned().unwrap_or_default();
            for tok in discharge.split(|c: char| !(c.is_alphanumeric() || c == '_' || c == '.')) {
                if Path::new(tok)
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("lean"))
                {
                    assert!(
                        corpus.join(tok).is_file(),
                        "row `{id}`: discharge artifact `{tok}` not found under {}",
                        corpus.display()
                    );
                    assert!(
                        contains_token(cell.as_bytes(), tok.as_bytes()),
                        "row `{id}`: discharge artifact `{tok}` missing from SPEC.md \"Retired by\": `{cell}`"
                    );
                }
            }
        }
        Ok(())
    }

    /// I8 — `SPEC.md` still points at the machine-readable manifest by path.
    #[test]
    fn i8_spec_references_manifest_path() -> Result<(), String> {
        assert!(
            spec()?.contains("proofs/tcb.json"),
            "SPEC.md no longer references `proofs/tcb.json`"
        );
        Ok(())
    }
}
