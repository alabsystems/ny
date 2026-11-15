// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! CATEGORY-NAME RATCHET (#name-blind-routing): benchmark category names in
//! `ny-cli`'s production source may only ever go DOWN.
//!
//! # Why this exists
//!
//! A verifier that picks its route by the benchmark's NAME is overfit by
//! construction: rename the category, or meet an unseen benchmark with the
//! same structure, and the route silently disappears. The 2026 relational
//! route was exactly that (`matches!(category, "isomorphic_acasxu_2026" |
//! "monotonic_acasxu_2026")` at seven sites); #name-blind-routing replaced it
//! with the property's own structure (how many networks the VNN-LIB declares)
//! and deleted three of the four env-dark, name-keyed experiments (the fourth
//! is explained on `is_safenlp_2024_category`). This file keeps the count from
//! creeping back.
//!
//! # What is counted — a deliberately simple lexical rule
//!
//! Every Rust string-literal token under `crates/ny-cli/src`, outside tests,
//! whose spelling is
//!
//! * EXACTLY a scored category id (the roster: every identifier-shaped literal
//!   of `vnncomp_2025_tracks.rs` and `vnncomp_2026_tracks.rs`, which are
//!   themselves exempt), or
//! * a category-FAMILY prefix — at least four bytes and a proper prefix of a
//!   roster id — passed directly to `.starts_with(` / `.ends_with(` /
//!   `.contains(` / `.strip_prefix(` / `.strip_suffix(`.
//!
//! Those are the raw material of every name comparison, `matches!` arm,
//! allowlist constant or name-keyed default. Prose and log text never match
//! (a message merely MENTIONING a category is not a literal equal to it), and
//! the preset lookup builds its path from the runtime category string, so it
//! contributes nothing. The rule is a SUPERSET of the scored path: it also
//! counts the non-scored `margin_row_bab::research` probes, which keeps the
//! ratchet free of a hand-maintained "is this scored?" list.
//!
//! "Outside tests" means: files that are the target of a test-only
//! `mod NAME;` declaration (`#[cfg(test)]`, `#[cfg(all(test, ..))]`, with or
//! without `#[path]`) and everything below that module's directory; and, in
//! every other file, each item carrying `#[test]` or such a cfg. A cfg that
//! does not REQUIRE `test` (`any(test, ..)`, `not(test)`) is production.
//!
//! # How to move the number
//!
//! Only down. Removing a name-keyed decision lowers its row: delete the row or
//! lower its count in `BASELINE` in the same commit. Adding one fails here: a
//! routing decision must be derived from the instance (the VNN-LIB structure,
//! the model's graph, the budget), never from the name the harness passes.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

/// `(file, literal, occurrences)` at the post-#name-blind-routing frontier.
///
/// * `margin_row_bab.rs` (15): the non-scored `margin_row_bab::research`
///   probe module — `NY_PROBE_CAT`/`NY_SWEEP_CAT` defaults, its probe table,
///   and its `starts_with("tiny")` family switch.
/// * `vnncomp.rs` (1): `is_safenlp_2024_category`, the one scored-path name
///   decision left. It is env-dark (`NY_SAFENLP_SHORT_GRACE`) and was kept
///   because removing it makes the short-grace watchdog protocol dead code,
///   and that protocol shares types with the historical watchdog every scored
///   row runs (see the note on the function). Its removal lowers this row to 0.
const BASELINE: &[(&str, &str, usize)] = &[
    (
        "crates/ny-cli/src/commands/margin_row_bab.rs",
        "cifar100_2024",
        8,
    ),
    ("crates/ny-cli/src/commands/margin_row_bab.rs", "tiny", 5),
    (
        "crates/ny-cli/src/commands/margin_row_bab.rs",
        "tinyimagenet_2024",
        2,
    ),
    ("crates/ny-cli/src/commands/vnncomp.rs", "safenlp_2024", 1),
];

/// Double entry for `BASELINE`: the sum of its counts.
const BASELINE_TOTAL: usize = 16;

const ROSTER_FILES: [&str; 2] = [
    "crates/ny-cli/src/commands/vnncomp_2025_tracks.rs",
    "crates/ny-cli/src/commands/vnncomp_2026_tracks.rs",
];
const SCAN_ROOT: &str = "crates/ny-cli/src";
const FAMILY_PREDICATES: [&str; 5] = [
    "starts_with",
    "ends_with",
    "contains",
    "strip_prefix",
    "strip_suffix",
];
const MIN_FAMILY_PREFIX: usize = 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TokenKind {
    Ident,
    String,
    Number,
    Punct(u8),
}

#[derive(Clone, Copy, Debug)]
struct Token {
    kind: TokenKind,
    start: usize,
    end: usize,
    line: usize,
}

/// The source-frontier tokenizer (`tests/source_frontier.rs::rust_tokens`),
/// reduced to the token kinds this rule reads. Comments vanish; string tokens
/// span their CONTENTS (raw, undecoded spelling).
fn rust_tokens(source: &str) -> Vec<Token> {
    let bytes = source.as_bytes();
    let mut tokens = Vec::new();
    let mut cursor = 0;
    let mut line = 1;

    while cursor < bytes.len() {
        if bytes[cursor] == b'\n' {
            line += 1;
            cursor += 1;
            continue;
        }
        if bytes[cursor..].starts_with(b"//") {
            cursor += 2;
            while cursor < bytes.len() && bytes[cursor] != b'\n' {
                cursor += 1;
            }
            continue;
        }
        if bytes[cursor..].starts_with(b"/*") {
            cursor += 2;
            let mut depth = 1_usize;
            while cursor < bytes.len() && depth > 0 {
                if bytes[cursor..].starts_with(b"/*") {
                    depth += 1;
                    cursor += 2;
                } else if bytes[cursor..].starts_with(b"*/") {
                    depth -= 1;
                    cursor += 2;
                } else {
                    line += usize::from(bytes[cursor] == b'\n');
                    cursor += 1;
                }
            }
            continue;
        }
        if let Some((content_start, hashes)) = raw_string_prefix(bytes, cursor) {
            let literal_line = line;
            cursor = content_start;
            let mut content_end = bytes.len();
            while cursor < bytes.len() {
                line += usize::from(bytes[cursor] == b'\n');
                if bytes[cursor] == b'"'
                    && cursor + 1 + hashes <= bytes.len()
                    && bytes[cursor + 1..cursor + 1 + hashes]
                        .iter()
                        .all(|byte| *byte == b'#')
                {
                    content_end = cursor;
                    cursor += hashes + 1;
                    break;
                }
                cursor += 1;
            }
            tokens.push(Token {
                kind: TokenKind::String,
                start: content_start,
                end: content_end,
                line: literal_line,
            });
            continue;
        }
        if bytes[cursor] == b'"' {
            let literal_line = line;
            cursor += 1;
            let content_start = cursor;
            let mut content_end = bytes.len();
            while cursor < bytes.len() {
                line += usize::from(bytes[cursor] == b'\n');
                match bytes[cursor] {
                    b'\\' => {
                        line += usize::from(bytes.get(cursor + 1) == Some(&b'\n'));
                        cursor = (cursor + 2).min(bytes.len());
                    }
                    b'"' => {
                        content_end = cursor;
                        cursor += 1;
                        break;
                    }
                    _ => cursor += 1,
                }
            }
            tokens.push(Token {
                kind: TokenKind::String,
                start: content_start,
                end: content_end,
                line: literal_line,
            });
            continue;
        }
        if bytes[cursor] == b'\'' {
            let literal_end = if bytes.get(cursor + 1) == Some(&b'\\') {
                (cursor + 3 < bytes.len() && bytes[cursor + 3] == b'\'').then_some(cursor + 4)
            } else {
                (cursor + 2 < bytes.len() && bytes[cursor + 2] == b'\'').then_some(cursor + 3)
            };
            cursor = literal_end.unwrap_or(cursor + 1);
            continue;
        }
        if bytes[cursor].is_ascii_alphabetic() || bytes[cursor] == b'_' {
            let start = cursor;
            cursor += 1;
            while cursor < bytes.len()
                && (bytes[cursor].is_ascii_alphanumeric() || bytes[cursor] == b'_')
            {
                cursor += 1;
            }
            tokens.push(Token {
                kind: TokenKind::Ident,
                start,
                end: cursor,
                line,
            });
            continue;
        }
        if bytes[cursor].is_ascii_digit() {
            let start = cursor;
            cursor += 1;
            while cursor < bytes.len()
                && (bytes[cursor].is_ascii_alphanumeric() || matches!(bytes[cursor], b'_' | b'.'))
            {
                cursor += 1;
            }
            tokens.push(Token {
                kind: TokenKind::Number,
                start,
                end: cursor,
                line,
            });
            continue;
        }
        if matches!(
            bytes[cursor],
            b'#' | b'!'
                | b'['
                | b']'
                | b'('
                | b')'
                | b'{'
                | b'}'
                | b','
                | b'&'
                | b':'
                | b';'
                | b'='
                | b'.'
                | b'<'
                | b'>'
                | b'*'
                | b'$'
        ) {
            tokens.push(Token {
                kind: TokenKind::Punct(bytes[cursor]),
                start: cursor,
                end: cursor + 1,
                line,
            });
        }
        cursor += 1;
    }
    tokens
}

fn raw_string_prefix(bytes: &[u8], cursor: usize) -> Option<(usize, usize)> {
    let mut marker = cursor;
    if matches!(bytes.get(marker), Some(b'b' | b'c')) {
        marker += 1;
    }
    if bytes.get(marker) != Some(&b'r') {
        return None;
    }
    marker += 1;
    let hashes_start = marker;
    while bytes.get(marker) == Some(&b'#') {
        marker += 1;
    }
    (bytes.get(marker) == Some(&b'"')).then_some((marker + 1, marker - hashes_start))
}

struct Lexed<'a> {
    source: &'a str,
    tokens: Vec<Token>,
}

impl<'a> Lexed<'a> {
    fn new(source: &'a str) -> Self {
        Self {
            source,
            tokens: rust_tokens(source),
        }
    }

    fn text(&self, index: usize) -> &'a str {
        let token = self.tokens[index];
        let source: &'a str = self.source;
        &source[token.start..token.end]
    }

    fn is_ident(&self, index: usize, expected: &str) -> bool {
        self.tokens
            .get(index)
            .is_some_and(|token| token.kind == TokenKind::Ident && self.text(index) == expected)
    }

    fn is_any_ident(&self, index: usize) -> bool {
        self.tokens
            .get(index)
            .is_some_and(|token| token.kind == TokenKind::Ident)
    }

    fn is_punct(&self, index: usize, expected: u8) -> bool {
        self.tokens
            .get(index)
            .is_some_and(|token| token.kind == TokenKind::Punct(expected))
    }

    fn is_string(&self, index: usize) -> bool {
        self.tokens
            .get(index)
            .is_some_and(|token| token.kind == TokenKind::String)
    }

    /// Index of the token closing the group opened at `open_index`.
    fn matching(&self, open_index: usize, open: u8, close: u8) -> Option<usize> {
        let mut depth = 0_usize;
        for index in open_index..self.tokens.len() {
            if self.is_punct(index, open) {
                depth += 1;
            } else if self.is_punct(index, close) {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(index);
                }
            }
        }
        None
    }
}

/// `#[test]`, `#[cfg(test)]`, or `#[cfg(all(.., test, ..))]` between the
/// brackets at `open` and `close`. Anything else (including `any(test, ..)`
/// and `not(test)`) is production.
fn attribute_requires_test(lexed: &Lexed<'_>, open: usize, close: usize) -> bool {
    if close == open + 2 && lexed.is_ident(open + 1, "test") {
        return true;
    }
    if !lexed.is_ident(open + 1, "cfg") || !lexed.is_punct(open + 2, b'(') {
        return false;
    }
    let first = open + 3;
    if lexed.is_ident(first, "test") && lexed.is_punct(first + 1, b')') {
        return true;
    }
    if !(lexed.is_ident(first, "all") && lexed.is_punct(first + 1, b'(')) {
        return false;
    }
    let Some(all_close) = lexed.matching(first + 1, b'(', b')') else {
        return false;
    };
    let mut depth = 0_i64;
    for index in first + 2..all_close {
        if lexed.is_punct(index, b'(') {
            depth += 1;
        } else if lexed.is_punct(index, b')') {
            depth -= 1;
        } else if depth == 0
            && lexed.is_ident(index, "test")
            && (lexed.is_punct(index + 1, b',') || index + 1 == all_close)
            && (lexed.is_punct(index - 1, b',') || index - 1 == first + 1)
        {
            return true;
        }
    }
    false
}

/// First token index after the item starting at `start` (which may begin with
/// outer attributes): the `;` ending it at bracket depth zero, or the `}`
/// matching its first top-level `{`.
fn item_end(lexed: &Lexed<'_>, start: usize) -> usize {
    let len = lexed.tokens.len();
    let mut index = start;
    while lexed.is_punct(index, b'#') && lexed.is_punct(index + 1, b'[') {
        match lexed.matching(index + 1, b'[', b']') {
            Some(close) => index = close + 1,
            None => return len,
        }
    }
    let mut depth = 0_i64;
    while index < len {
        if lexed.is_punct(index, b'(') || lexed.is_punct(index, b'[') {
            depth += 1;
        } else if lexed.is_punct(index, b')') || lexed.is_punct(index, b']') {
            depth -= 1;
        } else if lexed.is_punct(index, b';') && depth == 0 {
            return index + 1;
        } else if lexed.is_punct(index, b'{') && depth == 0 {
            return lexed
                .matching(index, b'{', b'}')
                .map_or(len, |close| close + 1);
        }
        index += 1;
    }
    len
}

/// Test-only structure of one file.
#[derive(Default)]
struct TestLayout {
    /// `#![cfg(test)]` (or `#![test]`): the whole file is test-only.
    whole_file: bool,
    /// Half-open token ranges of test-only items.
    regions: Vec<(usize, usize)>,
    /// Test-only `mod NAME;` declarations: `(NAME, #[path] value)`.
    modules: Vec<(String, Option<String>)>,
}

fn test_layout(lexed: &Lexed<'_>) -> TestLayout {
    let mut layout = TestLayout::default();
    let len = lexed.tokens.len();
    let mut index = 0;
    while index < len {
        if lexed.is_punct(index, b'#')
            && lexed.is_punct(index + 1, b'!')
            && lexed.is_punct(index + 2, b'[')
        {
            let Some(close) = lexed.matching(index + 2, b'[', b']') else {
                break;
            };
            if attribute_requires_test(lexed, index + 2, close) {
                layout.whole_file = true;
            }
            index = close + 1;
            continue;
        }
        if lexed.is_punct(index, b'#') && lexed.is_punct(index + 1, b'[') {
            let Some(close) = lexed.matching(index + 1, b'[', b']') else {
                break;
            };
            if !attribute_requires_test(lexed, index + 1, close) {
                index = close + 1;
                continue;
            }
            let end = item_end(lexed, index);
            layout.regions.push((index, end));
            let mut cursor = close + 1;
            let mut path_attribute = None;
            while lexed.is_punct(cursor, b'#') && lexed.is_punct(cursor + 1, b'[') {
                let Some(attribute_close) = lexed.matching(cursor + 1, b'[', b']') else {
                    break;
                };
                if lexed.is_ident(cursor + 2, "path")
                    && lexed.is_punct(cursor + 3, b'=')
                    && lexed.is_string(cursor + 4)
                {
                    path_attribute = Some(lexed.text(cursor + 4).to_owned());
                }
                cursor = attribute_close + 1;
            }
            if lexed.is_ident(cursor, "pub") {
                cursor += 1;
                if lexed.is_punct(cursor, b'(') {
                    cursor = lexed
                        .matching(cursor, b'(', b')')
                        .map_or(len, |close| close + 1);
                }
            }
            if lexed.is_ident(cursor, "mod")
                && lexed.is_any_ident(cursor + 1)
                && lexed.is_punct(cursor + 2, b';')
            {
                layout
                    .modules
                    .push((lexed.text(cursor + 1).to_owned(), path_attribute));
            }
            index = end;
            continue;
        }
        index += 1;
    }
    layout
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("workspace root must resolve")
}

fn collect_rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries =
        fs::read_dir(dir).unwrap_or_else(|error| panic!("read {}: {error}", dir.display()));
    for entry in entries {
        let entry = entry.expect("directory entry");
        let path = entry.path();
        let kind = entry.file_type().expect("file type");
        if kind.is_dir() {
            collect_rust_files(&path, out);
        } else if kind.is_file() && path.extension().is_some_and(|extension| extension == "rs") {
            out.push(path);
        }
    }
}

/// Lexical `a/./b/../c` -> `a/c` over `/`-separated relative paths.
fn normalize(path: &str) -> String {
    let mut parts: Vec<&str> = Vec::new();
    for part in path.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            other => parts.push(other),
        }
    }
    parts.join("/")
}

fn parent_dir(relative: &str) -> &str {
    relative.rsplit_once('/').map_or("", |(dir, _)| dir)
}

fn join(dir: &str, tail: &str) -> String {
    if dir.is_empty() {
        tail.to_owned()
    } else {
        format!("{dir}/{tail}")
    }
}

/// Identifier-shaped literals of the two track files: the scored category ids.
fn category_roster(root: &Path) -> BTreeSet<String> {
    let mut roster = BTreeSet::new();
    for relative in ROSTER_FILES {
        let source = fs::read_to_string(root.join(relative))
            .unwrap_or_else(|error| panic!("read roster {relative}: {error}"));
        let lexed = Lexed::new(&source);
        for index in 0..lexed.tokens.len() {
            if !lexed.is_string(index) {
                continue;
            }
            let value = lexed.text(index);
            let mut bytes = value.bytes();
            if bytes.next().is_some_and(|first| first.is_ascii_lowercase())
                && bytes
                    .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
            {
                roster.insert(value.to_owned());
            }
        }
    }
    roster
}

/// Does the string token at `index` count under the rule in the module docs?
fn counts(lexed: &Lexed<'_>, index: usize, roster: &BTreeSet<String>) -> bool {
    let value = lexed.text(index);
    if roster.contains(value) {
        return true;
    }
    value.len() >= MIN_FAMILY_PREFIX
        && index >= 3
        && lexed.is_punct(index - 1, b'(')
        && lexed.is_punct(index - 3, b'.')
        && FAMILY_PREDICATES
            .iter()
            .any(|method| lexed.is_ident(index - 2, method))
        && roster
            .iter()
            .any(|id| id.len() > value.len() && id.starts_with(value))
}

/// `(file, literal) -> source lines` over the production `ny-cli` tree.
fn scan(root: &Path) -> BTreeMap<(String, String), Vec<usize>> {
    let roster = category_roster(root);
    let mut paths = Vec::new();
    collect_rust_files(&root.join(SCAN_ROOT), &mut paths);
    let mut files = Vec::new();
    for path in paths {
        let relative = path
            .strip_prefix(root)
            .expect("scanned file lies under the workspace root")
            .to_string_lossy()
            .replace('\\', "/");
        let source =
            fs::read_to_string(&path).unwrap_or_else(|error| panic!("read {relative}: {error}"));
        files.push((relative, source));
    }
    files.sort();

    let mut test_files = BTreeSet::new();
    let mut test_dirs = BTreeSet::new();
    let mut layouts = Vec::new();
    for (relative, source) in &files {
        let lexed = Lexed::new(source);
        let layout = test_layout(&lexed);
        let dir = parent_dir(relative);
        let stem = relative
            .rsplit('/')
            .next()
            .and_then(|name| name.strip_suffix(".rs"))
            .unwrap_or_default();
        let base = if matches!(stem, "mod" | "main" | "lib") {
            dir.to_owned()
        } else {
            join(dir, stem)
        };
        for (name, path_attribute) in &layout.modules {
            if let Some(path) = path_attribute {
                test_files.insert(normalize(&join(dir, path)));
                continue;
            }
            test_files.insert(join(&base, &format!("{name}.rs")));
            test_files.insert(join(&base, &format!("{name}/mod.rs")));
            test_dirs.insert(join(&base, &format!("{name}/")));
        }
        layouts.push(layout);
    }

    let mut hits: BTreeMap<(String, String), Vec<usize>> = BTreeMap::new();
    for ((relative, source), layout) in files.iter().zip(&layouts) {
        if ROSTER_FILES.contains(&relative.as_str())
            || test_files.contains(relative)
            || test_dirs
                .iter()
                .any(|dir| relative.starts_with(dir.as_str()))
            || layout.whole_file
        {
            continue;
        }
        let lexed = Lexed::new(source);
        for index in 0..lexed.tokens.len() {
            if !lexed.is_string(index)
                || layout
                    .regions
                    .iter()
                    .any(|(start, end)| (*start..*end).contains(&index))
                || !counts(&lexed, index, &roster)
            {
                continue;
            }
            hits.entry((relative.clone(), lexed.text(index).to_owned()))
                .or_default()
                .push(lexed.tokens[index].line);
        }
    }
    hits
}

#[test]
fn category_name_literals_in_ny_cli_production_source_match_the_shrinking_baseline() {
    let baseline_total: usize = BASELINE.iter().map(|(_, _, count)| count).sum();
    assert_eq!(
        baseline_total, BASELINE_TOTAL,
        "BASELINE rows and BASELINE_TOTAL disagree; edit both together"
    );
    let expected: BTreeMap<(String, String), usize> = BASELINE
        .iter()
        .map(|(file, literal, count)| (((*file).to_owned(), (*literal).to_owned()), *count))
        .collect();
    let hits = scan(&workspace_root());
    let actual: BTreeMap<(String, String), usize> = hits
        .iter()
        .map(|(key, lines)| (key.clone(), lines.len()))
        .collect();
    if actual == expected {
        return;
    }
    let mut report = String::new();
    for (key, lines) in &hits {
        let before = expected.get(key).copied().unwrap_or(0);
        if lines.len() != before {
            report.push_str(&format!(
                "  {} {:?}: {} -> {} (lines {:?})\n",
                key.0,
                key.1,
                before,
                lines.len(),
                lines
            ));
        }
    }
    for (key, count) in &expected {
        if !hits.contains_key(key) {
            report.push_str(&format!("  {} {:?}: {} -> 0\n", key.0, key.1, count));
        }
    }
    let total: usize = actual.values().sum();
    panic!(
        "CATEGORY-NAME RATCHET: {total} category-name literal(s) in ny-cli production source, \
         baseline {BASELINE_TOTAL} (signed delta {:+}).\n{report}\n\
         If a count FELL, lower BASELINE/BASELINE_TOTAL in the same commit so the removed \
         name-keyed decision cannot return unnoticed. If one ROSE, do not raise the baseline: \
         derive the decision from the instance (VNN-LIB structure, model graph, budget) instead \
         of the category name the harness passes.",
        total as i128 - BASELINE_TOTAL as i128
    );
}

#[test]
fn category_roster_is_read_from_the_track_files() {
    let roster = category_roster(&workspace_root());
    assert!(
        roster.len() >= 30,
        "expected the 2025+2026 scored rosters (34 ids when written), read {}: {roster:?}",
        roster.len()
    );
    for id in [
        "isomorphic_acasxu_2026",
        "monotonic_acasxu_2026",
        "safenlp_2024",
        "traffic_signs_recognition_2023",
        "cgan_2023",
        "cgan2026",
    ] {
        assert!(roster.contains(id), "roster lost {id}");
    }
}

#[test]
fn scanner_counts_production_name_decisions_and_skips_tests_and_prose() {
    let roster: BTreeSet<String> = ["cgan2026", "tinyimagenet_2024"]
        .into_iter()
        .map(str::to_owned)
        .collect();
    let source = r####"
const FENCE: &str = "cgan2026";
fn route(category: &str) -> bool {
    eprintln!("route for cgan2026 rows");
    category == "cgan2026" || category.starts_with("tiny") || category.contains("tin")
}
#[cfg(any(test, feature = "x"))]
fn not_only_test() -> &'static str { "cgan2026" }
#[test]
fn a_test() { assert!(route("cgan2026")); }
#[cfg(all(test, feature = "mip"))]
mod inline_tests { const X: &str = "tinyimagenet_2024"; }
#[cfg(not(test))]
const PROD: &str = "tinyimagenet_2024";
// "cgan2026" in a comment
"####;
    let lexed = Lexed::new(source);
    let layout = test_layout(&lexed);
    let counted: Vec<&str> = (0..lexed.tokens.len())
        .filter(|&index| {
            lexed.is_string(index)
                && !layout
                    .regions
                    .iter()
                    .any(|(start, end)| (*start..*end).contains(&index))
                && counts(&lexed, index, &roster)
        })
        .map(|index| lexed.text(index))
        .collect();
    assert_eq!(
        counted,
        [
            "cgan2026",
            "cgan2026",
            "tiny",
            "cgan2026",
            "tinyimagenet_2024"
        ],
        "exact ids and family prefixes in production count; prose, comments, \
         sub-4-byte fragments and test-only items do not"
    );
    assert_eq!(layout.modules, [] as [(String, Option<String>); 0]);

    let declarations = r#"
#[cfg(test)]
mod tests;
#[cfg(all(test, feature = "mip"))]
#[path = "lane_tests.rs"]
mod lane_tests;
#[cfg(any(test, feature = "gpu"))]
mod not_test_only;
"#;
    let lexed = Lexed::new(declarations);
    let layout = test_layout(&lexed);
    assert_eq!(
        layout.modules,
        [
            ("tests".to_owned(), None),
            ("lane_tests".to_owned(), Some("lane_tests.rs".to_owned())),
        ]
    );
}
