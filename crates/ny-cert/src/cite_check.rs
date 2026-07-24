// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Citation-integrity check for the Trust = Clean fusion.
//!
//! The checker's `#[trust::cite(crownproof::<theorem>)]` annotations (see
//! [`crate::selfcheck`]) claim that a function's soundness obligation is grounded
//! in a specific, Clean-kernel-checked theorem from NY's exact revision-pinned
//! `crownproof` Lake dependency. A citation to a missing theorem — or to one whose
//! proof still contains a `sorry`/`admit` hole — is a *broken foundation*: the
//! proof-carrying soundness claim would rest on an unproven lemma.
//!
//! This module resolves a cited theorem against that pinned dependency and reports
//! whether it is declared and sorry-free. It is (a) a CI guard that the cites stay
//! grounded as the corpus evolves, and (b) the resolver the planned trustc-side
//! `cite` discharge will reuse (see `SPEC.md`, task #24). It deliberately does *not*
//! re-run the Lean kernel (that is `lake`'s job); it checks the textual integrity of
//! the citation target, comment-aware so prose like "sorry-free" is not mistaken for
//! the `sorry` tactic.

use std::path::{Path, PathBuf};

/// Exact private Clean dependency identity used for internal development.
pub const CLEAN_LAKE_PACKAGE: &str = "crownproof";
/// Canonical internal Clean Git remote. Publication rewrites this dependency.
pub const CLEAN_GIT_URL: &str = "https://github.com/alabsystems/clean.git";
/// Audited Clean revision containing every theorem in [`CITED_THEOREMS`].
pub const CLEAN_GIT_REV: &str = "a119ed0cfdafcb3eca4904253fdc51283e2ff0f8";
/// Clean's Lake package root within its repository.
pub const CLEAN_LAKE_SUBDIR: &str = "crown-proofs/lean";
/// Lean module directory below the Clean Lake package root.
pub const CLEAN_MODULE_ROOT: &str = "Crownproof";
/// Lake's checked-in package directory, relative to `proofs/lean`.
pub const CLEAN_PACKAGES_DIR: &str = ".lake/packages";
/// Canonical Mathlib remote used to override Clean's inherited proof environment.
pub const MATHLIB_GIT_URL: &str = "https://github.com/leanprover-community/mathlib4.git";
/// Exact Mathlib v4.30.0 release commit used by NY's Lean overlay.
pub const MATHLIB_GIT_REV: &str = "c5ea00351c28e24afc9f0f84379aa41082b1188f";
/// Lake package name for the exact Mathlib override.
pub const MATHLIB_LAKE_PACKAGE: &str = "mathlib";

/// Result of resolving a citation against the pinned Clean corpus.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CitationStatus {
    /// The theorem is declared and its proof body is free of `sorry`/`admit`.
    Grounded,
    /// No `theorem`/`lemma` of that name was found in the corpus.
    NotFound,
    /// The theorem is declared but its proof body contains a `sorry`/`admit`.
    HasSorry,
}

/// Strip Lean comments — nestable block `/- ... -/` and line `-- ...` — so that the
/// word "sorry" appearing in prose (e.g. "we prove it sorry-free") is not mistaken
/// for the `sorry` tactic. Replaces comment bytes with spaces to preserve byte
/// offsets and line structure for the subsequent declaration scan.
///
/// Operates on raw bytes (not `&str`): the read is now `std::fs::read` (byte-exact,
/// no UTF-8 decode boundary). The comment delimiters (`/-`, `-/`, `--`, `\n`) are
/// all ASCII, so byte-level matching is identical to char-level on valid UTF-8; any
/// multibyte content inside code or comments is copied through verbatim as bytes.
fn strip_lean_comments(src: &[u8]) -> Vec<u8> {
    let bytes = src;
    // `Vec::new()` (not `with_capacity(src.len())`): the capacity hint on an
    // unbounded input carries a hardened allocation obligation the model cannot
    // bound; amortized growth costs nothing measurable on SPEC-sized sources.
    let mut out: Vec<u8> = Vec::new();
    let mut i = 0;
    let mut block_depth = 0usize;
    while let Some(&c0) = bytes.get(i) {
        // `i < len <= isize::MAX`, so `i + 1` cannot wrap; the checked form
        // makes that provable without an invariant (None ⇒ no next byte).
        let two = (c0, i.checked_add(1).and_then(|j| bytes.get(j)).copied());
        if block_depth > 0 {
            match two {
                (b'/', Some(b'-')) => {
                    // Depth grows by 1 per 2 consumed input bytes, so it is bounded
                    // by src.len()/2 <= isize::MAX/2 and never actually saturates;
                    // the saturating form makes overflow-freedom locally provable.
                    block_depth = block_depth.saturating_add(1);
                    out.push(b' ');
                    out.push(b' ');
                    // This arm requires bytes.get(i + 1) == Some(_), i.e. i + 1 < len,
                    // so i + 2 <= len and the add cannot wrap; the saturating form
                    // makes that locally provable without a loop invariant.
                    i = i.saturating_add(2);
                }
                (b'-', Some(b'/')) => {
                    block_depth = block_depth.saturating_sub(1);
                    out.push(b' ');
                    out.push(b' ');
                    i = i.saturating_add(2);
                }
                (c, _) => {
                    out.push(if c == b'\n' { b'\n' } else { b' ' });
                    // i < len at loop entry, so the bump cannot wrap; the
                    // saturating form makes that locally provable.
                    i = i.saturating_add(1);
                }
            }
        } else {
            match two {
                (b'/', Some(b'-')) => {
                    block_depth = 1;
                    out.push(b' ');
                    out.push(b' ');
                    i = i.saturating_add(2);
                }
                (b'-', Some(b'-')) => {
                    // line comment: blank to end of line
                    while bytes.get(i).is_some_and(|&b| b != b'\n') {
                        out.push(b' ');
                        i = i.saturating_add(1);
                    }
                }
                (c, _) => {
                    // Copy the byte verbatim (multibyte UTF-8 passes through intact).
                    out.push(c);
                    i = i.saturating_add(1);
                }
            }
        }
    }
    out
}

/// True for an identifier character (alphanumeric or `_`).
///
/// A named free fn (not a per-caller closure): a direct call resolves to this
/// bundled, verified body, whereas each local-closure copy minted an
/// unresolvable `<{closure}> as Fn>::call` absent-callee obligation at every
/// call site (same lift rationale as `contains_token` being a free fn).
fn is_ident_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Byte analogue of [`is_ident_char`] for the Lean-source byte scanners.
///
/// Lean keywords and identifiers are ASCII, so `is_ascii_alphanumeric() || b'_'`
/// is the faithful byte predicate. This narrows `is_ident_char`'s Unicode
/// `char::is_alphanumeric` to ASCII, but the two agree on every token the
/// scanners actually search: the tokens/keywords are ASCII, and in valid UTF-8
/// a multibyte code point's bytes are all `>= 0x80`, which are non-identifier
/// under both predicates. The verdicts therefore match on the pinned Clean UTF-8
/// corpus; the only divergence — a searched ASCII token butted directly against
/// a non-ASCII *letter* with no delimiter — does not occur in Lean syntax
/// (tokens are whitespace/punctuation delimited).
fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// First index at which `word` occurs as a contiguous subslice of `hay` — the
/// byte analogue of `str::find` (used only on ASCII needles here). On valid
/// UTF-8 with a valid-UTF-8 needle this returns exactly the same offsets as
/// `str::find`, because UTF-8 is self-synchronizing (a valid subsequence can
/// only match at a char boundary). Total: the `get`/`checked_add` forms make
/// every cursor step machine-provable, and the loop terminates once no window
/// of `word.len()` bytes remains.
fn find_sub(hay: &[u8], word: &[u8]) -> Option<usize> {
    // `str::find` reports the empty needle at offset 0; match that.
    if word.is_empty() {
        return Some(0);
    }
    let mut i = 0usize;
    while let Some(win) = i.checked_add(word.len()).and_then(|end| hay.get(i..end)) {
        if win == word {
            return Some(i);
        }
        i = i.checked_add(1)?;
    }
    None
}

/// True when `word` occurs as a standalone token in `hay` (alphanumeric/`_`
/// boundaries — so `sorryAx` or `my_sorry` do not match the `sorry` tactic).
///
/// Byte scanner (`&[u8]`, not `&str`): fed comment-stripped bytes from the
/// byte-exact `std::fs::read`. The boundary predicate is [`is_ident_byte`], the
/// ASCII-faithful analogue of the char version; on the valid-UTF-8 corpus the
/// verdict is identical (see the note on `is_ident_byte`).
pub(crate) fn contains_token(hay: &[u8], word: &[u8]) -> bool {
    // Same total-scan discipline as the citation scanners: `find_sub`'s offsets
    // are in-bounds by construction, but that is not modeled — the checked/`get`
    // forms make every cursor step machine-provable; each `else` arm is
    // unreachable and returns the safe "no token found" answer.
    let mut from = 0;
    while let Some(tail) = hay.get(from..) {
        let Some(rel) = find_sub(tail, word) else {
            return false;
        };
        let Some(start) = from.checked_add(rel) else {
            return false;
        };
        let Some(end) = start.checked_add(word.len()) else {
            return false;
        };
        let before_ok = start == 0
            || !start
                .checked_sub(1)
                .and_then(|p| hay.get(p))
                .is_some_and(|&b| is_ident_byte(b));
        let after_ok = hay.get(end).is_none_or(|&b| !is_ident_byte(b));
        if before_ok && after_ok {
            return true;
        }
        let Some(next) = start.checked_add(1) else {
            return false;
        };
        from = next;
    }
    false
}

/// Leading-ASCII-whitespace-trimmed view of `line` — the byte analogue of the
/// `str::trim_start()` the decl scanners used, restricted to the ASCII whitespace
/// that forms Lean indentation (space/tab/newline/CR/FF). This narrows
/// `str::trim_start`'s Unicode whitespace to ASCII, but Lean indentation is ASCII,
/// so it agrees on the corpus. Total: a `get`-checked forward scan, no raw
/// indexing; `i` reaches at most `line.len()`, so the trailing `get(i..)` is `Some`.
fn trim_start_bytes(line: &[u8]) -> &[u8] {
    let mut i = 0usize;
    // Explicit `while let` + direct `is_ascii_whitespace` call (not
    // `is_some_and(u8::is_ascii_whitespace)`): the direct method call resolves
    // to a bundled, verified body, whereas the fn item passed as an `FnOnce`
    // minted an unresolvable indirect-call obligation. Identical verdicts:
    // advance while the next byte exists and is ASCII whitespace.
    while let Some(&b) = line.get(i) {
        if !b.is_ascii_whitespace() {
            break;
        }
        i = i.saturating_add(1);
    }
    line.get(i..).unwrap_or(&[])
}

/// True when `hay` begins with `prefix` — the manual analogue of
/// `slice::starts_with` (absent from the verified surface). Identical verdict:
/// `true` iff `prefix.len() <= hay.len()` and the leading bytes match. Total:
/// a `get`-checked forward scan; the mismatch/`None` arm returns the safe
/// "no prefix" answer, and an empty `prefix` is vacuously `true` (as for
/// `slice::starts_with`).
fn starts_with_bytes(hay: &[u8], prefix: &[u8]) -> bool {
    let mut i = 0usize;
    while let Some(&p) = prefix.get(i) {
        match hay.get(i) {
            Some(&h) if h == p => i = i.saturating_add(1),
            _ => return false,
        }
    }
    true
}

/// True when `line` (already comment-stripped) starts a top-level declaration that
/// ends the preceding theorem's proof body.
///
/// Byte version: the keywords are ASCII, so [`starts_with_bytes`] over bytes is
/// behavior-identical to the `&str` `starts_with` on valid UTF-8.
fn is_decl_boundary(line: &[u8]) -> bool {
    let t = trim_start_bytes(line);
    const KW: [&[u8]; 9] = [
        b"theorem ",
        b"lemma ",
        b"def ",
        b"instance ",
        b"abbrev ",
        b"structure ",
        b"inductive ",
        b"example ",
        b"end ",
    ];
    // Explicit loop (not `.any(|k| ..)`): avoids the `Iterator::any` absent
    // consumer + its closure row — identical short-circuit semantics.
    for k in KW.iter() {
        if starts_with_bytes(t, k) {
            return true;
        }
    }
    starts_with_bytes(t, b"@[")
}

/// True when `line` starts a `theorem`/`lemma` declaration for `name`.
///
/// A named free fn with `name` threaded as a parameter (the old `decl` closure
/// captured it): direct calls resolve to this bundled, verified body.
fn is_decl_for(line: &[u8], name: &[u8]) -> bool {
    let t = trim_start_bytes(line);
    (starts_with_bytes(t, b"theorem ") || starts_with_bytes(t, b"lemma "))
        && contains_token(t, name)
}

/// Byte analogue of `str::lines()`: split `src` on `b'\n'`, strip a single
/// trailing `b'\r'` from each `\n`-terminated piece (so `\r\n` behaves like
/// `str::lines`), and — like `str::lines` — yield no trailing empty line for a
/// terminating `b'\n'`. A trailing `\r` on the final, unterminated segment is
/// kept (matching `str::lines`, which only strips `\r` after stripping `\n`).
///
/// Explicit `Vec::new()`+`push` (not `.collect()`): the line count is
/// input-derived and the verifier cannot bound it, so a bulk `.collect()`
/// raises an UnboundedAllocation obligation. Total: a `get`-checked forward scan.
fn split_lines(src: &[u8]) -> Vec<&[u8]> {
    let mut lines: Vec<&[u8]> = Vec::new();
    let mut line_start = 0usize;
    let mut i = 0usize;
    while let Some(&b) = src.get(i) {
        if b == b'\n' {
            let mut line_end = i;
            // Strip the `\r` of a `\r\n`: only when it is the byte just before the
            // `\n`, mirroring `str::lines` (strip `\n`, then strip `\r`).
            // Literal `Some(&b'\r')` pattern (not `== Some(&b'\r')`): the match
            // compiles to a primitive byte compare, whereas `==` dispatched the
            // absent `Option<&u8>` `PartialEq` impl — identical verdict.
            if line_end > line_start {
                // Literal `Some(&b'\r')` pattern (not `== Some(&b'\r')`): the match
                // is a primitive byte compare, keeping the absent `Option<&u8>`
                // `PartialEq` impl off the verified path — identical verdict.
                #[allow(clippy::equatable_if_let)]
                if let Some(&b'\r') = src.get(line_end.saturating_sub(1)) {
                    line_end = line_end.saturating_sub(1);
                }
            }
            if let Some(l) = src.get(line_start..line_end) {
                lines.push(l);
            }
            line_start = i.saturating_add(1);
        }
        i = i.saturating_add(1);
    }
    // Final segment after the last `\n` (no trailing empty for a terminating
    // `\n`; a lone trailing `\r` here is kept, per `str::lines`).
    if line_start < src.len() {
        if let Some(l) = src.get(line_start..) {
            lines.push(l);
        }
    }
    lines
}

/// Extract the proof body of `theorem`/`lemma <name>` from comment-stripped source:
/// from its declaration line up to (but not including) the next top-level
/// declaration. Returns `None` if the theorem is not declared.
///
/// Byte version: identical line/decl-boundary logic to the former `&str` scan,
/// over the byte-exact `std::fs::read` buffer.
fn theorem_body<'a>(stripped: &'a [u8], name: &[u8]) -> Option<&'a [u8]> {
    // Byte analogue of `stripped.lines()` (see `split_lines`) — same identical
    // elements and order as the former `for l in stripped.lines()` collect loop.
    let lines: Vec<&[u8]> = split_lines(stripped);
    // Explicit indexed scan (not `.position(|l| decl(l))`): a direct call to the
    // free fn `is_decl_for` resolves to a bundled, verified body, whereas the old
    // capturing `decl` closure + `Iterator::position` consumer each minted an
    // absent-callee obligation. Identical first-match semantics.
    let mut start_opt = None;
    for (idx, l) in lines.iter().enumerate() {
        if is_decl_for(l, name) {
            start_opt = Some(idx);
            break;
        }
    }
    let start = start_opt?;
    let mut end = lines.len();
    for (off, l) in lines.iter().enumerate().skip(start.saturating_add(1)) {
        if is_decl_boundary(l) {
            end = off;
            break;
        }
    }
    // Reconstruct the byte span for the [start, end) line range. The sums are
    // bounded by `stripped.len() + line count` in reality; the checked forms
    // make that machine-provable (the `?`-on-None arms are unreachable, and
    // `None` is the safe "not found" answer for this lookup).
    let mut byte_start = 0usize;
    // Index-free iteration removes the non-derivable owned-Vec slice-length
    // obligation; `start <= end <= lines.len()` always, so these are identical.
    for l in lines.iter().take(start) {
        byte_start = byte_start.checked_add(l.len())?.checked_add(1)?;
    }
    let mut byte_end = byte_start;
    for l in lines.iter().skip(start).take(end.saturating_sub(start)) {
        byte_end = byte_end.checked_add(l.len())?.checked_add(1)?;
    }
    stripped.get(byte_start..byte_end.min(stripped.len()))
}

/// Resolve a cited theorem against the Clean corpus rooted at `corpus_root`
/// (the directory containing the `Crownproof/*.lean` modules).
///
/// # Errors
/// Propagates I/O errors reading the corpus directory.
pub fn citation_status(corpus_root: &Path, theorem: &str) -> std::io::Result<CitationStatus> {
    let mut found = false;
    for entry in std::fs::read_dir(corpus_root)? {
        let path = entry?.path();
        // SAFETY: no user-written unsafe here — the verifier attributes
        // `OsStr::to_str`'s std-internal `from_utf8_unchecked` (inlined MIR)
        // to this call site; `to_str` performs a full UTF-8 validity check
        // before that conversion, so the invariant holds by std's own guard.
        if path.extension().and_then(|e| e.to_str()) != Some("lean") {
            continue;
        }
        // Byte-exact read (`std::fs::read`, not `read_to_string`): no UTF-8 decode
        // boundary, so no hardened `utf8_reject` mandate. The whole scanner chain
        // below operates on `&[u8]` — decoding back to `String`/`str` (e.g.
        // `from_utf8`) would re-introduce a reject boundary, so we stay on bytes.
        let src: Vec<u8> = std::fs::read(&path)?;
        let stripped = strip_lean_comments(&src);
        if let Some(body) = theorem_body(&stripped, theorem.as_bytes()) {
            found = true;
            if contains_token(body, b"sorry") || contains_token(body, b"admit") {
                return Ok(CitationStatus::HasSorry);
            }
        }
    }
    Ok(if found {
        CitationStatus::Grounded
    } else {
        CitationStatus::NotFound
    })
}

/// The exact pinned Clean dependency's `Crownproof` module directory.
///
/// Integrity tests validate the `lakefile.toml` requirement, generated
/// `lake-manifest.json` entry, and dependency checkout before treating this path
/// as citation evidence.
#[must_use]
pub fn clean_corpus_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("proofs/lean")
        .join(CLEAN_PACKAGES_DIR)
        .join(CLEAN_LAKE_PACKAGE)
        .join(CLEAN_LAKE_SUBDIR)
        .join(CLEAN_MODULE_ROOT)
}

/// The theorems NY's `#[trust::cite]` annotations depend on — across both the
/// **checker** (`selfcheck.rs`, the soundness gate) and the **cert producers**
/// (`crown.rs`, `crown_deep.rs`, `sbar.rs`, the completeness side). Kept in sync
/// with the `#[trust::cite(...)]` clauses by
/// [`tests::cited_list_matches_source_annotations`].
///
/// * `farkas_premise_combination` — the abstract Farkas/entailment soundness core
///   (`Bridge.lean`); grounds every checker postcondition.
/// * `crown_bridge` — one-hidden-layer CROWN backward pass = Farkas certificate
///   (`Bridge.lean`); grounds [`crate::crown::Relu1Problem::certify`].
/// * `crown_bridge_deepK` — depth-`k` CROWN backward pass = Farkas certificate
///   (`DeepK.lean`); grounds the [`crate::crown_deep`] `certify*` producers.
/// * `sbar_support_sound` — box-truncated-simplex LP weak duality (`Sbar.lean`);
///   grounds [`crate::sbar::SimplexSupportLp::certify_upper`].
/// * `pow2_tangent` / `pow2_secant` — the quadratic (square) envelope premises
///   (`Pow2Envelope.lean`): tangent lower bound `2c·t − c² ≤ t²` (every `t`,
///   every `c`) and secant upper bound `t² ≤ (l+u)·t − l·u` on `t ∈ [l, u]`;
///   ground the pow2 envelope premise class emitted by
///   [`crate::crown_deep::DeepReluProblem::certify_difference_quadratic`].
/// * `branch_split_min` — exact branch-partition composition (`Branch.lean`);
///   grounds [`crate::branch::check_branch_tree`].
pub const CITED_THEOREMS: &[&str] = &[
    "farkas_premise_combination",
    "crown_bridge",
    "crown_bridge_deepK",
    "sbar_support_sound",
    "pow2_tangent",
    "pow2_secant",
    "branch_split_min",
];

/// Module qualifiers for source files that currently carry `#[trust::cite]`
/// annotations. Tests discover every Rust file below `src/` independently, then
/// require each citation-bearing file to have exactly one qualifier here. The
/// registry therefore only disambiguates function names for the cite-map; it is
/// not the source-discovery boundary.
pub const CITED_SOURCES: &[(&str, &str)] = &[
    ("src/selfcheck.rs", "selfcheck"),
    ("src/crown.rs", "crown::Relu1Problem"),
    ("src/crown_deep.rs", "crown_deep::DeepReluProblem"),
    ("src/sbar.rs", "sbar::SimplexSupportLp"),
    ("src/branch.rs", "branch"),
];

/// Parse a citation attribute that begins a Rust source line, accepting ordinary
/// whitespace within the attribute path and arguments.
fn cited_theorem_on_attribute_line(line: &str) -> Option<String> {
    let trimmed = line.trim_start();
    if !trimmed.starts_with("#[") {
        return None;
    }
    let end = trimmed.find(']')?;
    let attribute = trimmed.get(..=end)?;
    let compact: String = attribute.chars().filter(|c| !c.is_whitespace()).collect();
    let theorem = compact
        .strip_prefix("#[trust::cite(crownproof::")?
        .strip_suffix(")]")?;
    if theorem.is_empty() || !theorem.chars().all(is_ident_char) {
        return None;
    }
    Some(theorem.to_owned())
}

fn function_name_on_line(line: &str) -> Option<String> {
    let trimmed = line.trim_start();
    if trimmed.starts_with("//")
        || trimmed.starts_with("/*")
        || trimmed.starts_with('*')
        || trimmed.starts_with("#[")
    {
        return None;
    }
    let mut words = trimmed.split_whitespace();
    while let Some(word) = words.next() {
        if word == "fn" {
            let name = words.next()?;
            let name: String = name.chars().take_while(|&c| is_ident_char(c)).collect();
            return (!name.is_empty()).then_some(name);
        }
    }
    None
}

/// Parse `(function, cited_theorem)` pairs from Rust source: each
/// `#[trust::cite(crownproof::<thm>)]` is attributed to the next `fn <name>` that
/// follows it. This is the **cite-map** the planned `trust_verify` cite-discharge
/// consumes (a function's soundness postcondition is discharged *modulo* its cited,
/// kernel-checked theorem). Returns `(fn_name, theorem)` in source order.
#[must_use]
pub fn function_citations(src: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut pending = Vec::new();
    for line in src.lines() {
        if let Some(theorem) = cited_theorem_on_attribute_line(line) {
            pending.push(theorem);
            continue;
        }
        if !pending.is_empty() {
            if let Some(name) = function_name_on_line(line) {
                for theorem in std::mem::take(&mut pending) {
                    out.push((name.clone(), theorem));
                }
            }
        }
    }
    out
}

/// Extract the theorem names cited by `#[trust::cite(crownproof::<name>)]` in Rust
/// source. Used to cross-check that no citation in the code drifts out of the
/// verified [`CITED_THEOREMS`] set (and thus out of the integrity guard).
#[must_use]
pub fn citations_in_source(src: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in src.lines() {
        if let Some(theorem) = cited_theorem_on_attribute_line(line) {
            if !out.contains(&theorem) {
                out.push(theorem);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, serde::Deserialize)]
    struct LakeManifest {
        #[serde(rename = "packagesDir")]
        packages_dir: String,
        packages: Vec<LakePackage>,
        name: String,
    }

    #[derive(Debug, serde::Deserialize)]
    struct LakePackage {
        url: String,
        #[serde(rename = "type")]
        kind: String,
        #[serde(rename = "subDir")]
        sub_dir: Option<String>,
        rev: String,
        name: String,
        #[serde(rename = "inputRev")]
        input_rev: Option<String>,
        inherited: bool,
    }

    fn parse_clean_lake_requirement(lakefile: &str) -> Result<(), String> {
        fn finish(
            current: &mut Option<std::collections::BTreeMap<String, String>>,
            requirements: &mut Vec<std::collections::BTreeMap<String, String>>,
        ) {
            if let Some(requirement) = current.take() {
                requirements.push(requirement);
            }
        }

        let mut requirements = Vec::new();
        let mut current = None;
        for (line_no, line) in lakefile.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.starts_with('[') {
                finish(&mut current, &mut requirements);
                if trimmed == "[[require]]" {
                    current = Some(std::collections::BTreeMap::new());
                }
                continue;
            }
            let Some(requirement) = current.as_mut() else {
                continue;
            };
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            let Some((key, raw)) = trimmed.split_once('=') else {
                continue;
            };
            let key = key.trim();
            if !["name", "git", "rev", "subDir"].contains(&key) {
                continue;
            }
            let raw = raw.trim();
            let value = raw
                .strip_prefix('"')
                .and_then(|value| value.strip_suffix('"'))
                .ok_or_else(|| {
                    format!(
                        "lakefile.toml line {}: `{key}` must be a plain quoted string",
                        line_no + 1
                    )
                })?;
            if value.contains('"') || value.contains('\\') {
                return Err(format!(
                    "lakefile.toml line {}: escaped `{key}` is not accepted by the pin guard",
                    line_no + 1
                ));
            }
            if requirement
                .insert(key.to_owned(), value.to_owned())
                .is_some()
            {
                return Err(format!(
                    "lakefile.toml line {}: duplicate `{key}` in [[require]]",
                    line_no + 1
                ));
            }
        }
        finish(&mut current, &mut requirements);

        let clean_candidates: Vec<_> = requirements
            .iter()
            .enumerate()
            .filter(|(_, requirement)| {
                requirement
                    .get("name")
                    .is_some_and(|v| v == CLEAN_LAKE_PACKAGE)
                    || requirement.get("git").is_some_and(|v| v == CLEAN_GIT_URL)
            })
            .collect();
        if clean_candidates.len() != 1 {
            return Err(format!(
                "lakefile.toml must contain exactly one `{CLEAN_LAKE_PACKAGE}` / Clean Git requirement; found {}",
                clean_candidates.len()
            ));
        }
        let (clean_index, requirement) = clean_candidates[0];
        for (key, expected) in [
            ("name", CLEAN_LAKE_PACKAGE),
            ("git", CLEAN_GIT_URL),
            ("rev", CLEAN_GIT_REV),
            ("subDir", CLEAN_LAKE_SUBDIR),
        ] {
            if requirement.get(key).map(String::as_str) != Some(expected) {
                return Err(format!(
                    "lakefile.toml Clean requirement `{key}` is {:?}, expected `{expected}`",
                    requirement.get(key)
                ));
            }
        }

        let mathlib_candidates: Vec<_> = requirements
            .iter()
            .enumerate()
            .filter(|(_, requirement)| {
                requirement
                    .get("name")
                    .is_some_and(|v| v == MATHLIB_LAKE_PACKAGE)
                    || requirement.get("git").is_some_and(|v| v == MATHLIB_GIT_URL)
            })
            .collect();
        if mathlib_candidates.len() != 1 {
            return Err(format!(
                "lakefile.toml must contain exactly one exact Mathlib requirement; found {}",
                mathlib_candidates.len()
            ));
        }
        let (mathlib_index, mathlib) = mathlib_candidates[0];
        if clean_index >= mathlib_index {
            return Err(
                "lakefile.toml must require exact Mathlib after Clean so it overrides Clean's inherited graph"
                    .to_owned(),
            );
        }
        for (key, expected) in [
            ("name", MATHLIB_LAKE_PACKAGE),
            ("git", MATHLIB_GIT_URL),
            ("rev", MATHLIB_GIT_REV),
        ] {
            if mathlib.get(key).map(String::as_str) != Some(expected) {
                return Err(format!(
                    "lakefile.toml Mathlib requirement `{key}` is {:?}, expected `{expected}`",
                    mathlib.get(key)
                ));
            }
        }
        if mathlib.contains_key("subDir") {
            return Err("lakefile.toml Mathlib requirement must not set subDir".to_owned());
        }
        Ok(())
    }

    fn same_git_url(actual: &str, expected: &str) -> bool {
        actual.trim_end_matches(".git") == expected.trim_end_matches(".git")
    }

    fn git_stdout(dir: &Path, args: &[&str]) -> Result<String, String> {
        let output = std::process::Command::new("git")
            .arg("-C")
            .arg(dir)
            .args(args)
            .output()
            .map_err(|e| format!("run git in {}: {e}", dir.display()))?;
        if !output.status.success() {
            return Err(format!(
                "git {:?} failed in {}: {}",
                args,
                dir.display(),
                String::from_utf8_lossy(&output.stderr).trim()
            ));
        }
        String::from_utf8(output.stdout)
            .map(|stdout| stdout.trim().to_owned())
            .map_err(|e| format!("git output was not UTF-8 in {}: {e}", dir.display()))
    }

    fn validated_clean_corpus_root() -> Result<PathBuf, String> {
        let lean_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("proofs/lean");
        let lakefile = std::fs::read_to_string(lean_root.join("lakefile.toml"))
            .map_err(|e| format!("read lakefile.toml: {e}"))?;
        parse_clean_lake_requirement(&lakefile)?;

        let manifest_text = std::fs::read_to_string(lean_root.join("lake-manifest.json"))
            .map_err(|e| format!("read lake-manifest.json (run `lake update`): {e}"))?;
        let manifest: LakeManifest = serde_json::from_str(&manifest_text)
            .map_err(|e| format!("parse lake-manifest.json: {e}"))?;
        if manifest.name != "nyproof" {
            return Err(format!(
                "lake-manifest project is `{}`, expected `nyproof`",
                manifest.name
            ));
        }
        if manifest.packages_dir != CLEAN_PACKAGES_DIR {
            return Err(format!(
                "lake-manifest packagesDir is `{}`, expected `{CLEAN_PACKAGES_DIR}`",
                manifest.packages_dir
            ));
        }
        let candidates: Vec<_> = manifest
            .packages
            .iter()
            .filter(|package| {
                package.name == CLEAN_LAKE_PACKAGE || same_git_url(&package.url, CLEAN_GIT_URL)
            })
            .collect();
        if candidates.len() != 1 {
            return Err(format!(
                "lake-manifest must contain exactly one `{CLEAN_LAKE_PACKAGE}` / Clean Git package; found {}",
                candidates.len()
            ));
        }
        let package = candidates[0];
        if package.name != CLEAN_LAKE_PACKAGE
            || !same_git_url(&package.url, CLEAN_GIT_URL)
            || package.kind != "git"
            || package.sub_dir.as_deref() != Some(CLEAN_LAKE_SUBDIR)
            || package.rev != CLEAN_GIT_REV
            || package.input_rev.as_deref() != Some(CLEAN_GIT_REV)
            || package.inherited
        {
            return Err(format!(
                "lake-manifest Clean package does not match the exact audited dependency: {package:?}"
            ));
        }
        let mathlib_candidates: Vec<_> = manifest
            .packages
            .iter()
            .filter(|package| {
                package.name == MATHLIB_LAKE_PACKAGE || same_git_url(&package.url, MATHLIB_GIT_URL)
            })
            .collect();
        if mathlib_candidates.len() != 1 {
            return Err(format!(
                "lake-manifest must contain exactly one root Mathlib package; found {}",
                mathlib_candidates.len()
            ));
        }
        let mathlib = mathlib_candidates[0];
        if mathlib.name != MATHLIB_LAKE_PACKAGE
            || !same_git_url(&mathlib.url, MATHLIB_GIT_URL)
            || mathlib.kind != "git"
            || mathlib.sub_dir.is_some()
            || mathlib.rev != MATHLIB_GIT_REV
            || mathlib.input_rev.as_deref() != Some(MATHLIB_GIT_REV)
            || mathlib.inherited
        {
            return Err(format!(
                "lake-manifest Mathlib package does not match NY's exact root override: {mathlib:?}"
            ));
        }

        let package_root = lean_root
            .join(&manifest.packages_dir)
            .join(CLEAN_LAKE_PACKAGE);
        let corpus = package_root.join(CLEAN_LAKE_SUBDIR).join(CLEAN_MODULE_ROOT);
        if corpus != clean_corpus_root() {
            return Err(format!(
                "derived Clean corpus root {} disagrees with {}",
                corpus.display(),
                clean_corpus_root().display()
            ));
        }
        if !corpus.is_dir() {
            return Err(format!(
                "Clean dependency corpus missing at {} (run `lake update`)",
                corpus.display()
            ));
        }
        let head = git_stdout(&package_root, &["rev-parse", "--verify", "HEAD"])?;
        if head != CLEAN_GIT_REV {
            return Err(format!(
                "Clean dependency checkout is at `{head}`, expected `{CLEAN_GIT_REV}`"
            ));
        }
        let status = git_stdout(
            &package_root,
            &[
                "status",
                "--porcelain",
                "--untracked-files=all",
                "--",
                CLEAN_LAKE_SUBDIR,
            ],
        )?;
        if !status.is_empty() {
            return Err(format!(
                "Clean dependency theorem source differs from `{CLEAN_GIT_REV}`:\n{status}"
            ));
        }

        let mathlib_root = lean_root
            .join(&manifest.packages_dir)
            .join(MATHLIB_LAKE_PACKAGE);
        if !mathlib_root.is_dir() {
            return Err(format!(
                "Mathlib dependency checkout missing at {} (run `lake update`)",
                mathlib_root.display()
            ));
        }
        let mathlib_head = git_stdout(&mathlib_root, &["rev-parse", "--verify", "HEAD"])?;
        if mathlib_head != MATHLIB_GIT_REV {
            return Err(format!(
                "Mathlib dependency checkout is at `{mathlib_head}`, expected `{MATHLIB_GIT_REV}`"
            ));
        }
        let mathlib_status = git_stdout(
            &mathlib_root,
            &[
                "status",
                "--porcelain",
                "--untracked-files=all",
                "--",
                "Mathlib",
                "Mathlib.lean",
            ],
        )?;
        if !mathlib_status.is_empty() {
            return Err(format!(
                "Mathlib source differs from `{MATHLIB_GIT_REV}`:\n{mathlib_status}"
            ));
        }
        Ok(corpus)
    }

    fn rust_sources_below(dir: &Path) -> Result<Vec<PathBuf>, String> {
        fn visit(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
            let entries = std::fs::read_dir(dir)
                .map_err(|e| format!("read Rust source directory {}: {e}", dir.display()))?;
            for entry in entries {
                let entry =
                    entry.map_err(|e| format!("read entry below {}: {e}", dir.display()))?;
                let path = entry.path();
                let ty = entry
                    .file_type()
                    .map_err(|e| format!("inspect Rust source path {}: {e}", path.display()))?;
                if ty.is_dir() {
                    visit(&path, out)?;
                } else if ty.is_file() && path.extension().is_some_and(|ext| ext == "rs") {
                    out.push(path);
                } else if ty.is_symlink() {
                    return Err(format!(
                        "symlink below Rust source root is not scanned: {}",
                        path.display()
                    ));
                }
            }
            Ok(())
        }

        let mut sources = Vec::new();
        visit(dir, &mut sources)?;
        sources.sort();
        Ok(sources)
    }

    fn contains_cite_attribute(src: &str) -> bool {
        src.lines()
            .any(|line| cited_theorem_on_attribute_line(line).is_some())
    }

    fn citation_sources(dir: &Path) -> Result<Vec<(String, String)>, String> {
        let mut sources = Vec::new();
        for path in rust_sources_below(&dir.join("src"))? {
            let text = std::fs::read_to_string(&path)
                .map_err(|e| format!("read Rust source {}: {e}", path.display()))?;
            if contains_cite_attribute(&text) {
                let relative_path = path
                    .strip_prefix(dir)
                    .map_err(|e| format!("relativize Rust source {}: {e}", path.display()))?;
                let relative = relative_path
                    .to_str()
                    .ok_or_else(|| {
                        format!("non-UTF-8 Rust source path: {}", relative_path.display())
                    })?
                    .replace('\\', "/");
                sources.push((relative, text));
            }
        }
        Ok(sources)
    }

    #[test]
    fn strip_comments_blanks_block_and_line() {
        let s = "theorem foo := by /- sorry here -/ exact rfl -- sorry there\n";
        let out = strip_lean_comments(s.as_bytes());
        assert!(
            !contains_token(&out, b"sorry"),
            "comment sorry must be stripped: {out:?}"
        );
        assert!(contains_token(&out, b"exact"));
    }

    #[test]
    fn token_match_excludes_substrings() {
        assert!(contains_token(b"by sorry", b"sorry"));
        assert!(
            !contains_token(b"sorryAx is fine", b"sorry"),
            "sorryAx must not match the tactic"
        );
        // `sorry-free` DOES match the bare token (`-` is a non-ident boundary) — which
        // is exactly why citations are checked on COMMENT-STRIPPED source, since prose
        // like "we prove it sorry-free" lives in comments (Bridge.lean does this).
        assert!(contains_token(b"sorry-free", b"sorry"));
    }

    #[test]
    fn detects_sorry_in_body_not_in_comment() {
        let src = "\
/- this lemma is proven sorry-free -/\n\
theorem clean_one : True := by trivial\n\
theorem holed : True := by sorry\n";
        let stripped = strip_lean_comments(src.as_bytes());
        let clean = theorem_body(&stripped, b"clean_one").unwrap();
        assert!(!contains_token(clean, b"sorry"));
        let holed = theorem_body(&stripped, b"holed").unwrap();
        assert!(contains_token(holed, b"sorry"));
    }

    /// The load-bearing integrity test: every theorem the checker cites must be
    /// declared and sorry-free in the exact pinned Clean dependency. A failure
    /// here means the proof-carrying soundness claim rests on a missing or
    /// unproven lemma.
    #[test]
    fn all_cited_theorems_are_grounded() -> Result<(), String> {
        let root = validated_clean_corpus_root()?;
        assert!(
            root.is_dir(),
            "Clean dependency corpus missing at {}",
            root.display()
        );
        for &thm in CITED_THEOREMS {
            // An unreadable corpus is an `Err` and FAILS the test (fail-closed):
            // grounding is never assumed when the source cannot be checked.
            let status = citation_status(&root, thm).map_err(|e| format!("read corpus: {e}"))?;
            assert_eq!(
                status,
                CitationStatus::Grounded,
                "cited theorem `{thm}` is not grounded (status: {status:?}) — \
                 the #[trust::cite] foundation is broken",
            );
        }
        Ok(())
    }

    #[test]
    fn extracts_citations_from_source() {
        let src = "#[trust::ensures(|r| true)]\n#[trust::cite(crownproof::farkas_premise_combination)]\npub fn f() {}\n";
        assert_eq!(
            citations_in_source(src),
            vec!["farkas_premise_combination".to_string()]
        );
        let spaced = "#[ trust :: cite ( crownproof :: branch_split_min ) ]\npub fn g() {}\n";
        assert_eq!(
            citations_in_source(spaced),
            vec!["branch_split_min".to_string()]
        );
    }

    /// Close the drift loop: discover every Rust source below `src/`, then require
    /// every `#[trust::cite(crownproof::X)]` to be registered and grounded. A new
    /// source file or citation therefore cannot escape through a stale hardcoded
    /// scan list.
    #[test]
    fn cited_list_matches_source_annotations() -> Result<(), String> {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let root = validated_clean_corpus_root()?;
        let sources = citation_sources(dir)?;
        let mut found = std::collections::BTreeSet::new();
        for (file, src) in &sources {
            let in_source = citations_in_source(src);
            for thm in &in_source {
                found.insert(thm.clone());
                assert!(
                    CITED_THEOREMS.contains(&thm.as_str()),
                    "citation `{thm}` in {file} is not in the verified CITED_THEOREMS set \
                     — it would escape the grounding guard",
                );
                assert_eq!(
                    citation_status(&root, thm).map_err(|e| format!("read corpus: {e}"))?,
                    CitationStatus::Grounded,
                    "source citation `{thm}` in {file} is not grounded in the corpus",
                );
            }
        }
        let expected = CITED_THEOREMS.iter().map(|thm| (*thm).to_owned()).collect();
        assert_eq!(
            found, expected,
            "CITED_THEOREMS must exactly match discovered #[trust::cite] annotations"
        );
        Ok(())
    }

    #[test]
    fn extracts_function_citations() {
        let src = "#[trust::cite(crownproof::thm_a)]\npub fn foo() {}\n\
                   /// doc\n#[ensures(|r| true)]\n#[trust::cite(crownproof::thm_b)]\nfn bar() {}\n";
        assert_eq!(
            function_citations(src),
            vec![
                ("foo".to_string(), "thm_a".to_string()),
                ("bar".to_string(), "thm_b".to_string()),
            ],
        );
        let spaced = "#[ trust :: cite ( crownproof :: thm_c ) ]\npub(crate) fn baz<T>() {}\n";
        assert_eq!(
            function_citations(spaced),
            vec![("baz".to_string(), "thm_c".to_string())]
        );
    }

    /// The cite-map sidecar (`proofs/cite-map.json`, consumed by the planned
    /// trust_verify cite-discharge) must stay in sync with the `#[trust::cite]`
    /// annotations across all discovered citation-bearing Rust sources (checker +
    /// producers), and every mapped theorem must be grounded. [`CITED_SOURCES`]
    /// supplies only the module/type qualifier and must exactly cover that
    /// discovered set.
    #[test]
    fn cite_map_matches_source() -> Result<(), String> {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let sources = citation_sources(dir)?;
        let qualifiers: std::collections::BTreeMap<&str, &str> =
            CITED_SOURCES.iter().copied().collect();
        assert_eq!(
            qualifiers.len(),
            CITED_SOURCES.len(),
            "duplicate source path in CITED_SOURCES"
        );
        let discovered: std::collections::BTreeSet<&str> =
            sources.iter().map(|(file, _)| file.as_str()).collect();
        let registered: std::collections::BTreeSet<&str> = qualifiers.keys().copied().collect();
        assert_eq!(
            discovered, registered,
            "CITED_SOURCES qualifiers must exactly cover discovered citation-bearing Rust files"
        );
        let src: std::collections::BTreeSet<(String, String)> = sources
            .iter()
            .flat_map(|(file, text)| {
                let module = qualifiers[file.as_str()];
                function_citations(text)
                    .into_iter()
                    .map(move |(f, t)| (format!("{module}::{f}"), t))
            })
            .collect();
        let map_text = std::fs::read_to_string(dir.join("proofs/cite-map.json"))
            .map_err(|e| format!("read cite-map: {e}"))?;
        let map_json: serde_json::Value =
            serde_json::from_str(&map_text).map_err(|e| format!("parse cite-map: {e}"))?;
        let expected_corpus = serde_json::json!({
            "source": "lake-git-dependency",
            "package": CLEAN_LAKE_PACKAGE,
            "git": CLEAN_GIT_URL,
            "rev": CLEAN_GIT_REV,
            "subDir": CLEAN_LAKE_SUBDIR,
            "packagesDir": CLEAN_PACKAGES_DIR,
            "moduleRoot": CLEAN_MODULE_ROOT,
            "mathlib": {
                "package": MATHLIB_LAKE_PACKAGE,
                "git": MATHLIB_GIT_URL,
                "rev": MATHLIB_GIT_REV,
            },
        });
        assert_eq!(
            map_json["corpus"], expected_corpus,
            "cite-map corpus metadata must name the exact pinned Clean Lake dependency"
        );
        let mapped: std::collections::BTreeSet<(String, String)> = map_json["citations"]
            .as_array()
            .ok_or("citations array")?
            .iter()
            .map(|c| {
                (
                    c["function"].as_str().unwrap().to_string(),
                    c["theorem"].as_str().unwrap().to_string(),
                )
            })
            .collect();
        assert_eq!(
            src, mapped,
            "cite-map.json out of sync with #[trust::cite] in NY source"
        );
        let root = validated_clean_corpus_root()?;
        for (_f, thm) in &mapped {
            assert_eq!(
                citation_status(&root, thm).map_err(|e| format!("read corpus: {e}"))?,
                CitationStatus::Grounded,
                "cite-map theorem `{thm}` is not grounded",
            );
        }
        Ok(())
    }
}
