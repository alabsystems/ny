// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Exact source frontier for the legacy process-environment surface.
//!
//! The older `ratchet` test intentionally counts two textual spellings. This
//! test has a different job: lex Rust source, discard comments, and compare
//! every executable lever-name literal and raw environment-reader site with a
//! reviewed, checked-in occurrence manifest. Both additions and removals fail.
//!
//! This is a lexical policy, not a Rust name resolver. It resolves same-file
//! and globally-unambiguous string constants used as environment keys, module
//! and function aliases imported from `std::env`, and direct calls through
//! locally or globally-unambiguous helper functions whose parameter flows
//! straight into a raw environment read. The deliberately bounded resolver is
//! documented again in the failure message so the frontier is not mistaken
//! for proof of arbitrary interprocedural data flow.

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ffi::OsStr;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

const LITERAL_MANIFEST: &str = include_str!("data/ny_literal_frontier.tsv");
const READER_MANIFEST: &str = include_str!("data/ny_reader_frontier.tsv");

const LITERAL_CATEGORIES: &[&str] = &["registered", "permanent", "legacy"];
const READER_CATEGORIES: &[&str] = &["canonical", "reviewed", "legacy"];
const CONTEXT_RADIUS: usize = 12;

/// Reviewed parameter-governance debt at the current frontier.
///
/// These are exact ratchets, not budgets. A migration that declares a legacy
/// name, removes a stale reader, or reclassifies an intentional non-lever site
/// must lower the corresponding baseline in the same commit. Raising a value
/// would make newly introduced undeclared configuration look like accepted
/// debt and defeats the source frontier's role as the forcing function for
/// completing parameter governance.
const BASELINE_LEGACY_LITERAL_SITES: usize = 1_807;
const BASELINE_LEGACY_LITERAL_NAMES: usize = 768;
const BASELINE_LEGACY_READER_SITES: usize = 979;
// FNV-1a over the sorted identity sets below. Counts alone would permit a new
// raw master to replace an unrelated retired site one-for-one; exact identity
// fingerprints make that substitution fail until review lowers the debt and
// records the new, strictly smaller frontier in the same commit.
// Re-anchored 2026-08-27, NOT a substitution: legacy sites stayed 1_807 and
// legacy names stayed 768. One physical site
// (target_backward.rs `NY_PATCHES_BUDGET_SECS`) changed structural anchor
// because #frozen-crown-authority deleted the adjacent
// `env.set("NY_CROWN_OBJ_CHUNK", ...)` inside CONTEXT_RADIUS of it, which
// collapsed two token-identical scrub blocks onto one anchor with
// collision ordinals 1 and 2. That deletion also RETIRED a registered
// reader, so the frontier moved strictly in the intended direction.
const BASELINE_LEGACY_LITERAL_SITE_FINGERPRINT: u64 = 0x0674_9915_b503_3cd9;
const BASELINE_LEGACY_LITERAL_NAME_FINGERPRINT: u64 = 0x65b7_5585_8418_fa76;
// 2026-08-26 #disj-demand: the NY_SKIP_DISJ_PGD reader in
// verify/disjunctive.rs kept its name, count, and semantics but its structural
// anchor changed (the binding it feeds was renamed `skip_disj_pgd` →
// `skip_disj_pgd_env` when the demand-zeroed skip composed with it), so the
// SAME site re-fingerprints. Debt count unchanged (979 after the upstream
// removals this landed on top of); the manifest row was re-anchored in the
// same commit (ny_reader_frontier.tsv). Value RE-DERIVED from the ratchet's
// own failure output on the MERGED tree, per the collision rule — neither
// pre-merge side's number described this tree.
const BASELINE_LEGACY_READER_SITE_FINGERPRINT: u64 = 0x4663_d458_4d6e_1088;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TokenKind {
    Ident,
    String { raw: bool },
    Number,
    Punct(u8),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Token {
    kind: TokenKind,
    start: usize,
    end: usize,
    line: usize,
}

#[derive(Debug)]
struct SourceFile {
    relative: String,
    source: String,
    tokens: Vec<Token>,
    aliases: BTreeMap<String, String>,
    env_modules: BTreeSet<String>,
    env_functions: BTreeMap<String, String>,
    functions: Vec<FunctionDef>,
}

#[derive(Clone, Debug)]
struct FunctionDef {
    name: String,
    parameters: Vec<Option<String>>,
    body_open: usize,
    body_close: usize,
}

#[derive(Clone, Debug)]
struct RawCall {
    file: usize,
    token: usize,
    api: String,
    target: String,
    argument_ident: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct HelperDef {
    file: usize,
    name: String,
    function_body_open: usize,
    parameter: usize,
}

#[derive(Debug)]
struct LocatedSite {
    base: String,
    line: usize,
}

#[derive(Debug)]
struct Frontier {
    literals: BTreeMap<String, usize>,
    readers: BTreeMap<String, usize>,
    registered_names: BTreeSet<String>,
    reviewed_macro_literal_bases: BTreeSet<String>,
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("workspace root must resolve")
}

fn is_skipped_dir(name: &OsStr) -> bool {
    let name = name.to_string_lossy();
    name.starts_with('.')
        || name == "target"
        || name.starts_with("target-")
        || name.starts_with("target_")
}

fn collect_rust_sources(dir: &Path, sources: &mut Vec<PathBuf>) -> io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let kind = entry.file_type()?;
        if kind.is_symlink() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "refusing to skip symlinked source tree entry {}",
                    path.display()
                ),
            ));
        }
        if kind.is_dir() {
            if !is_skipped_dir(&entry.file_name()) {
                collect_rust_sources(&path, sources)?;
            }
        } else if kind.is_file()
            && path
                .extension()
                .is_some_and(|extension| extension.eq_ignore_ascii_case("rs"))
        {
            sources.push(path);
        }
    }
    Ok(())
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

/// Tokenize the Rust subset needed by the source policy.
///
/// Nested block comments, line comments, character literals, byte/C strings,
/// raw strings, and multiline calls are handled. String tokens point at their
/// contents, while all other tokens point at their source spelling.
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
                kind: TokenKind::String { raw: true },
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
                kind: TokenKind::String { raw: false },
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

fn text<'a>(file: &'a SourceFile, token: &Token) -> &'a str {
    &file.source[token.start..token.end]
}

fn decoded_string<'a>(file: &'a SourceFile, token: &Token) -> Option<Cow<'a, str>> {
    let TokenKind::String { raw } = token.kind else {
        return None;
    };
    let spelling = text(file, token);
    if raw || !spelling.contains('\\') {
        return Some(Cow::Borrowed(spelling));
    }

    let mut decoded = String::with_capacity(spelling.len());
    let mut characters = spelling.chars().peekable();
    while let Some(character) = characters.next() {
        if character != '\\' {
            decoded.push(character);
            continue;
        }
        match characters.next()? {
            '\\' => decoded.push('\\'),
            '"' => decoded.push('"'),
            'n' => decoded.push('\n'),
            'r' => decoded.push('\r'),
            't' => decoded.push('\t'),
            '0' => decoded.push('\0'),
            'x' => {
                let high = characters.next()?.to_digit(16)?;
                let low = characters.next()?.to_digit(16)?;
                decoded.push(char::from_u32(high * 16 + low)?);
            }
            'u' => {
                if characters.next()? != '{' {
                    return None;
                }
                let mut value = 0_u32;
                let mut digits = 0_usize;
                loop {
                    let next = characters.next()?;
                    if next == '}' {
                        break;
                    }
                    if next == '_' {
                        continue;
                    }
                    value = value.checked_mul(16)?.checked_add(next.to_digit(16)?)?;
                    digits += 1;
                }
                if digits == 0 {
                    return None;
                }
                decoded.push(char::from_u32(value)?);
            }
            '\n' => {
                while characters.peek().is_some_and(|next| next.is_whitespace()) {
                    characters.next();
                }
            }
            '\r' => {
                if characters.next()? != '\n' {
                    return None;
                }
                while characters.peek().is_some_and(|next| next.is_whitespace()) {
                    characters.next();
                }
            }
            _ => return None,
        }
    }
    Some(Cow::Owned(decoded))
}

fn ident(file: &SourceFile, index: usize, expected: &str) -> bool {
    file.tokens
        .get(index)
        .is_some_and(|token| token.kind == TokenKind::Ident && text(file, token) == expected)
}

fn punct(file: &SourceFile, index: usize, expected: u8) -> bool {
    file.tokens
        .get(index)
        .is_some_and(|token| token.kind == TokenKind::Punct(expected))
}

fn find_matching(file: &SourceFile, opening: usize, open: u8, close: u8) -> Option<usize> {
    if !punct(file, opening, open) {
        return None;
    }
    let mut depth = 1_usize;
    for index in opening + 1..file.tokens.len() {
        match file.tokens[index].kind {
            TokenKind::Punct(found) if found == open => depth += 1,
            TokenKind::Punct(found) if found == close => {
                depth -= 1;
                if depth == 0 {
                    return Some(index);
                }
            }
            _ => {}
        }
    }
    None
}

fn split_arguments(file: &SourceFile, opening: usize) -> Option<Vec<(usize, usize)>> {
    let closing = find_matching(file, opening, b'(', b')')?;
    let mut arguments = Vec::new();
    let mut start = opening + 1;
    let mut round = 0_usize;
    let mut square = 0_usize;
    let mut curly = 0_usize;
    for index in opening + 1..closing {
        match file.tokens[index].kind {
            TokenKind::Punct(b'(') => round += 1,
            TokenKind::Punct(b')') => round = round.saturating_sub(1),
            TokenKind::Punct(b'[') => square += 1,
            TokenKind::Punct(b']') => square = square.saturating_sub(1),
            TokenKind::Punct(b'{') => curly += 1,
            TokenKind::Punct(b'}') => curly = curly.saturating_sub(1),
            TokenKind::Punct(b',') if round == 0 && square == 0 && curly == 0 => {
                arguments.push((start, index));
                start = index + 1;
            }
            _ => {}
        }
    }
    if start < closing || !arguments.is_empty() {
        arguments.push((start, closing));
    }
    Some(arguments)
}

fn lever_prefix() -> &'static str {
    concat!("N", "Y_")
}

fn is_lever_name(candidate: &str) -> bool {
    candidate.strip_prefix(lever_prefix()).is_some_and(|rest| {
        !rest.is_empty()
            && rest
                .bytes()
                .all(|byte| byte.is_ascii_uppercase() || byte.is_ascii_digit() || byte == b'_')
    })
}

fn collect_aliases(file: &SourceFile) -> BTreeMap<String, String> {
    let mut aliases = BTreeMap::new();
    let mut ambiguous = BTreeSet::new();
    let mut cursor = 0;
    while cursor < file.tokens.len() {
        if !(ident(file, cursor, "const") || ident(file, cursor, "static")) {
            cursor += 1;
            continue;
        }
        let mut name_index = cursor + 1;
        if ident(file, name_index, "mut") {
            name_index += 1;
        }
        let Some(name_token) = file.tokens.get(name_index) else {
            break;
        };
        if name_token.kind != TokenKind::Ident {
            cursor += 1;
            continue;
        }
        let name = text(file, name_token).to_owned();
        let mut probe = name_index + 1;
        let mut saw_equals = false;
        let mut value = None;
        while probe < file.tokens.len() && !punct(file, probe, b';') {
            saw_equals |= punct(file, probe, b'=');
            if saw_equals && matches!(file.tokens[probe].kind, TokenKind::String { .. }) {
                let candidate = decoded_string(file, &file.tokens[probe]);
                if candidate.as_deref().is_some_and(is_lever_name) {
                    value = candidate.map(Cow::into_owned);
                }
                break;
            }
            probe += 1;
        }
        if let Some(value) = value {
            if ambiguous.contains(&name) {
                cursor = probe.saturating_add(1);
                continue;
            }
            match aliases.get(&name) {
                None => {
                    aliases.insert(name, value);
                }
                Some(previous) if previous == &value => {}
                Some(_) => {
                    aliases.remove(&name);
                    ambiguous.insert(name);
                }
            }
        }
        cursor = probe.saturating_add(1);
    }
    aliases
}

fn collect_env_imports(file: &SourceFile) -> (BTreeSet<String>, BTreeMap<String, String>) {
    let mut modules = BTreeSet::from(["env".to_owned()]);
    let mut functions = BTreeMap::new();
    let apis = ["var", "var_os", "vars", "vars_os"];
    let mut cursor = 0;

    while cursor < file.tokens.len() {
        if !ident(file, cursor, "use") {
            cursor += 1;
            continue;
        }
        let end = (cursor + 1..file.tokens.len())
            .find(|index| punct(file, *index, b';'))
            .unwrap_or(file.tokens.len());
        let statement = cursor + 1..end;
        let has_std = statement.clone().any(|index| ident(file, index, "std"));
        let env_positions: Vec<_> = statement
            .clone()
            .filter(|index| ident(file, *index, "env"))
            .collect();
        if has_std {
            for env_index in env_positions {
                let module_alias = if ident(file, env_index + 1, "as") {
                    file.tokens.get(env_index + 2).and_then(|token| {
                        (token.kind == TokenKind::Ident).then(|| text(file, token).to_owned())
                    })
                } else {
                    Some("env".to_owned())
                };
                if let Some(alias) = module_alias {
                    modules.insert(alias);
                }

                for api_index in env_index + 1..end {
                    let Some(api) = apis.iter().copied().find(|api| ident(file, api_index, api))
                    else {
                        continue;
                    };
                    let alias = if ident(file, api_index + 1, "as") {
                        file.tokens.get(api_index + 2).and_then(|token| {
                            (token.kind == TokenKind::Ident).then(|| text(file, token).to_owned())
                        })
                    } else {
                        Some(api.to_owned())
                    };
                    if let Some(alias) = alias {
                        functions.insert(alias, api.to_owned());
                    }
                }
            }
        }
        cursor = end.saturating_add(1);
    }
    (modules, functions)
}

fn parameter_name(file: &SourceFile, start: usize, end: usize) -> Option<String> {
    (start..end).find_map(|index| {
        let token = file.tokens[index];
        if token.kind != TokenKind::Ident {
            return None;
        }
        let candidate = text(file, &token);
        (!matches!(candidate, "mut" | "ref" | "self" | "Self")).then(|| candidate.to_owned())
    })
}

fn collect_functions(file: &SourceFile) -> Vec<FunctionDef> {
    let mut functions = Vec::new();
    for cursor in 0..file.tokens.len() {
        if !ident(file, cursor, "fn") {
            continue;
        }
        let Some(name_token) = file.tokens.get(cursor + 1) else {
            continue;
        };
        if name_token.kind != TokenKind::Ident {
            continue;
        }
        let Some(opening) = (cursor + 2..file.tokens.len())
            .find(|index| punct(file, *index, b'(') || punct(file, *index, b';'))
        else {
            continue;
        };
        if punct(file, opening, b';') {
            continue;
        }
        let Some(closing) = find_matching(file, opening, b'(', b')') else {
            continue;
        };
        let Some(body_open) = (closing + 1..file.tokens.len())
            .find(|index| punct(file, *index, b'{') || punct(file, *index, b';'))
        else {
            continue;
        };
        if punct(file, body_open, b';') {
            continue;
        }
        let Some(body_close) = find_matching(file, body_open, b'{', b'}') else {
            continue;
        };
        let parameters = split_arguments(file, opening)
            .unwrap_or_default()
            .into_iter()
            .map(|(start, end)| parameter_name(file, start, end))
            .collect();
        functions.push(FunctionDef {
            name: text(file, name_token).to_owned(),
            parameters,
            body_open,
            body_close,
        });
    }
    functions
}

fn load_sources(root: &Path) -> io::Result<Vec<SourceFile>> {
    let mut paths = Vec::new();
    collect_rust_sources(&root.join("crates"), &mut paths)?;
    paths.sort();
    let mut files = Vec::new();
    for path in paths {
        let source = fs::read_to_string(&path)?;
        let relative = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");
        let tokens = rust_tokens(&source);
        let mut file = SourceFile {
            relative,
            source,
            tokens,
            aliases: BTreeMap::new(),
            env_modules: BTreeSet::new(),
            env_functions: BTreeMap::new(),
            functions: Vec::new(),
        };
        file.aliases = collect_aliases(&file);
        (file.env_modules, file.env_functions) = collect_env_imports(&file);
        file.functions = collect_functions(&file);
        files.push(file);
    }
    Ok(files)
}

fn global_aliases(files: &[SourceFile]) -> BTreeMap<String, String> {
    let mut candidates: BTreeMap<String, Option<String>> = BTreeMap::new();
    for file in files {
        for (name, value) in &file.aliases {
            candidates
                .entry(name.clone())
                .and_modify(|candidate| {
                    if candidate.as_ref() != Some(value) {
                        *candidate = None;
                    }
                })
                .or_insert_with(|| Some(value.clone()));
        }
    }
    candidates
        .into_iter()
        .filter_map(|(name, value)| value.map(|value| (name, value)))
        .collect()
}

fn resolve_alias(
    file: &SourceFile,
    global: &BTreeMap<String, String>,
    name: &str,
) -> Option<String> {
    file.aliases.get(name).or_else(|| global.get(name)).cloned()
}

fn argument_target(
    file: &SourceFile,
    global: &BTreeMap<String, String>,
    start: usize,
    end: usize,
) -> (String, Option<String>) {
    let significant: Vec<_> = (start..end)
        .filter(|index| !punct(file, *index, b'&'))
        .collect();
    if significant.len() == 1 {
        let token = file.tokens[significant[0]];
        if matches!(token.kind, TokenKind::String { .. }) {
            return match decoded_string(file, &token) {
                Some(value) if is_lever_name(&value) => (value.into_owned(), None),
                Some(_) => ("<non-ny>".to_owned(), None),
                None => ("<escaped-or-invalid>".to_owned(), None),
            };
        }
        if token.kind == TokenKind::Ident {
            let name = text(file, &token);
            if let Some(value) = resolve_alias(file, global, name) {
                return (format!("alias={value}"), Some(name.to_owned()));
            }
            return ("<dynamic>".to_owned(), Some(name.to_owned()));
        }
    }
    ("<dynamic>".to_owned(), None)
}

fn environment_api_at(file: &SourceFile, index: usize) -> Option<String> {
    let token = *file.tokens.get(index)?;
    if token.kind != TokenKind::Ident || !punct(file, index + 1, b'(') {
        return None;
    }
    let called = text(file, &token);
    if let Some(api) = file.env_functions.get(called) {
        return Some(api.clone());
    }
    if !matches!(called, "var" | "var_os" | "vars" | "vars_os")
        || index < 3
        || !punct(file, index - 1, b':')
        || !punct(file, index - 2, b':')
    {
        return None;
    }
    let module = file.tokens[index - 3];
    (module.kind == TokenKind::Ident && file.env_modules.contains(text(file, &module)))
        .then(|| called.to_owned())
}

fn collect_raw_calls(files: &[SourceFile], global: &BTreeMap<String, String>) -> Vec<RawCall> {
    let mut calls = Vec::new();
    for (file_index, file) in files.iter().enumerate() {
        for token_index in 0..file.tokens.len() {
            let Some(api) = environment_api_at(file, token_index) else {
                continue;
            };
            if matches!(api.as_str(), "vars" | "vars_os") {
                calls.push(RawCall {
                    file: file_index,
                    token: token_index,
                    api,
                    target: "<all>".to_owned(),
                    argument_ident: None,
                });
                continue;
            }
            let arguments = split_arguments(file, token_index + 1).unwrap_or_default();
            let (target, argument_ident) = arguments.first().map_or_else(
                || ("<dynamic>".to_owned(), None),
                |(start, end)| argument_target(file, global, *start, *end),
            );
            calls.push(RawCall {
                file: file_index,
                token: token_index,
                api,
                target,
                argument_ident,
            });
        }
    }
    calls
}

fn enclosing_function(file: &SourceFile, token: usize) -> Option<&FunctionDef> {
    file.functions
        .iter()
        .filter(|function| function.body_open < token && token < function.body_close)
        .min_by_key(|function| function.body_close - function.body_open)
}

fn collect_helper_defs(files: &[SourceFile], raw_calls: &[RawCall]) -> BTreeSet<HelperDef> {
    let mut helpers = BTreeSet::new();
    for call in raw_calls {
        let Some(argument) = call.argument_ident.as_deref() else {
            continue;
        };
        let file = &files[call.file];
        let Some(function) = enclosing_function(file, call.token) else {
            continue;
        };
        for (parameter, name) in function.parameters.iter().enumerate() {
            if name.as_deref() == Some(argument) {
                helpers.insert(HelperDef {
                    file: call.file,
                    name: function.name.clone(),
                    function_body_open: function.body_open,
                    parameter,
                });
            }
        }
    }
    helpers
}

fn helper_parameters_for_call(
    helpers: &BTreeSet<HelperDef>,
    file: usize,
    name: &str,
) -> Vec<usize> {
    let local: Vec<_> = helpers
        .iter()
        .filter(|helper| helper.file == file && helper.name == name)
        .collect();
    let local_functions: BTreeSet<_> = local
        .iter()
        .map(|helper| helper.function_body_open)
        .collect();
    if local_functions.len() == 1 {
        return local.into_iter().map(|helper| helper.parameter).collect();
    }
    let global: Vec<_> = helpers
        .iter()
        .filter(|helper| helper.name == name)
        .collect();
    let global_functions: BTreeSet<_> = global
        .iter()
        .map(|helper| (helper.file, helper.function_body_open))
        .collect();
    if global_functions.len() == 1 {
        global.into_iter().map(|helper| helper.parameter).collect()
    } else {
        Vec::new()
    }
}

fn is_function_declaration(file: &SourceFile, token: usize) -> bool {
    token > 0 && ident(file, token - 1, "fn")
}

fn hash_context(file: &SourceFile, center: usize) -> u64 {
    let start = center.saturating_sub(CONTEXT_RADIUS);
    let end = (center + CONTEXT_RADIUS + 1).min(file.tokens.len());
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for token in &file.tokens[start..end] {
        let tag = match token.kind {
            TokenKind::Ident => b'i',
            TokenKind::String { .. } => b's',
            TokenKind::Number => b'n',
            TokenKind::Punct(value) => value,
        };
        hash ^= u64::from(tag);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        let spelling = text(file, token).as_bytes();
        for (index, byte) in spelling.iter().copied().enumerate() {
            if byte == b'\r' && spelling.get(index + 1) == Some(&b'\n') {
                continue;
            }
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        hash ^= 0xff;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn macro_reader_at(file: &SourceFile, token: usize) -> Option<(&str, usize)> {
    let name = file.tokens.get(token)?;
    if name.kind != TokenKind::Ident
        || !matches!(text(file, name), "env" | "option_env")
        || !punct(file, token + 1, b'!')
        || !punct(file, token + 2, b'(')
    {
        return None;
    }
    let arguments = split_arguments(file, token + 2)?;
    let (start, end) = *arguments.first()?;
    if end != start + 1 || !matches!(file.tokens[start].kind, TokenKind::String { .. }) {
        return None;
    }
    decoded_string(file, &file.tokens[start])
        .is_some_and(|value| is_lever_name(&value))
        .then(|| (text(file, name), start))
}

fn disambiguate_sites(sites: Vec<LocatedSite>) -> BTreeMap<String, usize> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    let mut result = BTreeMap::new();
    for site in sites {
        let ordinal = counts
            .entry(site.base.clone())
            .and_modify(|count| *count += 1)
            .or_insert(1);
        let key = format!("{}\t{}", site.base, *ordinal);
        assert!(
            result.insert(key, site.line).is_none(),
            "structural source-site key unexpectedly collided"
        );
    }
    result
}

fn literal_site_base(file: &SourceFile, token: usize, name: &str) -> String {
    format!(
        "{}\t{:016x}\t{}",
        file.relative,
        hash_context(file, token),
        name
    )
}

fn is_reviewed_build_macro_path(path: &str) -> bool {
    matches!(
        path,
        "crates/ny-cli/src/build_id.rs" | "crates/ny-cli/src/main.rs"
    )
}

fn scan_frontier() -> Frontier {
    let root = workspace_root();
    let files = load_sources(&root).unwrap_or_else(|error| {
        panic!("failed to scan {}: {error}", root.join("crates").display())
    });
    assert!(
        files.len() > 500,
        "frontier must scan the whole crates tree; saw only {} Rust files",
        files.len()
    );
    let global = global_aliases(&files);
    let raw_calls = collect_raw_calls(&files, &global);
    let helpers = collect_helper_defs(&files, &raw_calls);

    let registered_names: BTreeSet<_> = ny_levers::all()
        .all()
        .iter()
        .map(|declaration| declaration.name.to_owned())
        .collect();
    let mut literal_sites = Vec::new();
    let mut reader_sites = Vec::new();
    let mut reviewed_macro_literal_bases = BTreeSet::new();

    for file in &files {
        for (token_index, token) in file.tokens.iter().enumerate() {
            if let Some(name) = decoded_string(file, token).filter(|name| is_lever_name(name)) {
                literal_sites.push(LocatedSite {
                    base: literal_site_base(file, token_index, &name),
                    line: token.line,
                });
            }
            if let Some((api, literal_token)) = macro_reader_at(file, token_index) {
                let target = decoded_string(file, &file.tokens[literal_token])
                    .expect("macro_reader_at returned a decodable lever name");
                if is_reviewed_build_macro_path(&file.relative) {
                    reviewed_macro_literal_bases.insert(literal_site_base(
                        file,
                        literal_token,
                        &target,
                    ));
                }
                reader_sites.push(LocatedSite {
                    base: format!(
                        "{}\t{:016x}\t{}!\t{}",
                        file.relative,
                        hash_context(file, token_index),
                        api,
                        target
                    ),
                    line: token.line,
                });
            }
        }
    }

    for call in &raw_calls {
        let file = &files[call.file];
        reader_sites.push(LocatedSite {
            base: format!(
                "{}\t{:016x}\t{}\t{}",
                file.relative,
                hash_context(file, call.token),
                call.api,
                call.target
            ),
            line: file.tokens[call.token].line,
        });
    }

    for (file_index, file) in files.iter().enumerate() {
        for token_index in 0..file.tokens.len() {
            let token = file.tokens[token_index];
            if token.kind != TokenKind::Ident
                || !punct(file, token_index + 1, b'(')
                || is_function_declaration(file, token_index)
                || punct(file, token_index.saturating_sub(1), b'.')
            {
                continue;
            }
            let called = text(file, &token);
            let helper_parameters = helper_parameters_for_call(&helpers, file_index, called);
            if helper_parameters.is_empty() {
                continue;
            }
            let arguments = split_arguments(file, token_index + 1).unwrap_or_default();
            for parameter in helper_parameters {
                let Some((start, end)) = arguments.get(parameter).copied() else {
                    continue;
                };
                let (target, _) = argument_target(file, &global, start, end);
                let target_name = target.strip_prefix("alias=").unwrap_or(&target);
                if !is_lever_name(target_name) {
                    continue;
                }
                reader_sites.push(LocatedSite {
                    base: format!(
                        "{}\t{:016x}\thelper:{}\t{}",
                        file.relative,
                        hash_context(file, token_index),
                        called,
                        target
                    ),
                    line: token.line,
                });
            }
        }
    }

    Frontier {
        literals: disambiguate_sites(literal_sites),
        readers: disambiguate_sites(reader_sites),
        registered_names,
        reviewed_macro_literal_bases,
    }
}

fn parse_manifest(
    label: &str,
    contents: &str,
    allowed_categories: &[&str],
) -> BTreeMap<String, String> {
    let mut records = BTreeMap::new();
    let mut previous_key: Option<String> = None;
    for (line_index, line) in contents.lines().enumerate() {
        let line = line.trim_end();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((category, key)) = line.split_once('\t') else {
            panic!("{label}:{} must be CATEGORY<TAB>KEY", line_index + 1);
        };
        assert!(
            allowed_categories.contains(&category),
            "{label}:{} has unknown category {category:?}",
            line_index + 1
        );
        if let Some(previous) = previous_key.as_deref() {
            assert!(
                previous < key,
                "{label}:{} is not strictly sorted by site key: {key}",
                line_index + 1
            );
        }
        assert!(
            records
                .insert(key.to_owned(), category.to_owned())
                .is_none(),
            "{label}:{} duplicates {key}",
            line_index + 1
        );
        previous_key = Some(key.to_owned());
    }
    records
}

fn literal_name_from_key(key: &str) -> &str {
    key.split('\t')
        .nth(2)
        .unwrap_or_else(|| panic!("malformed literal site key: {key}"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LegacyDebt {
    literal_sites: usize,
    literal_names: usize,
    reader_sites: usize,
    literal_site_fingerprint: u64,
    literal_name_fingerprint: u64,
    reader_site_fingerprint: u64,
}

fn identity_fingerprint<'a>(identities: impl IntoIterator<Item = &'a str>) -> u64 {
    // A fixed algorithm, not `DefaultHasher`: this value is a checked source
    // baseline and must remain stable across toolchains. The 0xff record
    // separator makes `["ab", "c"]` distinct from `["a", "bc"]`.
    let mut fingerprint = 0xcbf2_9ce4_8422_2325_u64;
    for identity in identities {
        for byte in identity.bytes().chain(std::iter::once(0xff)) {
            fingerprint ^= u64::from(byte);
            fingerprint = fingerprint.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    fingerprint
}

fn legacy_debt(
    literals: &BTreeMap<String, String>,
    readers: &BTreeMap<String, String>,
) -> LegacyDebt {
    let legacy_literals: Vec<_> = literals
        .iter()
        .filter(|(_key, category)| category.as_str() == "legacy")
        .collect();
    let literal_names: BTreeSet<_> = legacy_literals
        .iter()
        .map(|(key, _category)| literal_name_from_key(key))
        .collect();
    LegacyDebt {
        literal_sites: legacy_literals.len(),
        literal_names: literal_names.len(),
        reader_sites: readers
            .values()
            .filter(|category| category.as_str() == "legacy")
            .count(),
        literal_site_fingerprint: identity_fingerprint(
            legacy_literals.iter().map(|(key, _category)| key.as_str()),
        ),
        literal_name_fingerprint: identity_fingerprint(literal_names.iter().copied()),
        reader_site_fingerprint: identity_fingerprint(
            readers
                .iter()
                .filter(|(_key, category)| category.as_str() == "legacy")
                .map(|(key, _category)| key.as_str()),
        ),
    }
}

fn assert_exact_shrinking_ratchet(label: &str, current: usize, baseline: usize) {
    assert!(
        current == baseline,
        "{label} changed from its exact parameter-governance baseline: current={current}, \
         baseline={baseline}, signed_delta={:+}. If the count fell, lower the baseline in \
         this commit so the removed debt cannot return unnoticed. If it rose, do not raise \
         the baseline: declare and route the new runtime lever, or classify an intentional \
         build/test/non-lever site in the reviewed category.",
        current as i128 - baseline as i128,
    );
}

fn assert_exact_frontier_identity(label: &str, current: u64, baseline: u64) {
    assert_eq!(
        current, baseline,
        "{label} identity changed: current=0x{current:016x}, baseline=0x{baseline:016x}. \
         A count-only match is insufficient because it can hide a one-for-one debt \
         substitution. If legacy debt fell, lower its count and replace this fingerprint \
         with the strictly smaller reviewed frontier in the same commit. Otherwise route \
         or explicitly review the new site instead of blessing the substitution.",
    );
}

fn base_from_key(key: &str) -> &str {
    key.rsplit_once('\t')
        .map(|(base, _ordinal)| base)
        .unwrap_or_else(|| panic!("malformed source site key: {key}"))
}

fn reader_key_fields(key: &str) -> (&str, &str, &str) {
    let mut fields = key.split('\t');
    let path = fields
        .next()
        .unwrap_or_else(|| panic!("malformed reader site key: {key}"));
    let _anchor = fields
        .next()
        .unwrap_or_else(|| panic!("malformed reader site key: {key}"));
    let api = fields
        .next()
        .unwrap_or_else(|| panic!("malformed reader site key: {key}"));
    let target = fields
        .next()
        .unwrap_or_else(|| panic!("malformed reader site key: {key}"));
    let ordinal = fields
        .next()
        .unwrap_or_else(|| panic!("malformed reader site key: {key}"));
    assert!(
        fields.next().is_none() && ordinal.parse::<usize>().is_ok(),
        "malformed reader site key: {key}"
    );
    (path, api, target)
}

fn is_canonical_reader(key: &str) -> bool {
    let (path, api, target) = reader_key_fields(key);
    matches!(
        path,
        "crates/ny-levers/src/env.rs" | "crates/ny-levers/src/gate.rs"
    ) && matches!(api, "var" | "var_os")
        && target == "<dynamic>"
}

fn diff_message(
    label: &str,
    actual: &BTreeMap<String, usize>,
    expected: &BTreeMap<String, String>,
) -> String {
    let added: Vec<_> = actual
        .iter()
        .filter(|(key, _)| !expected.contains_key(*key))
        .collect();
    let stale: Vec<_> = expected
        .keys()
        .filter(|key| !actual.contains_key(*key))
        .collect();
    if added.is_empty() && stale.is_empty() {
        return String::new();
    }
    let mut message = format!(
        "{label} frontier differs: {} unreviewed current site(s), {} stale manifest site(s)\n",
        added.len(),
        stale.len()
    );
    if !added.is_empty() {
        message.push_str("unreviewed current sites (source line shown after `@`):\n");
        for (key, line) in added.iter().take(80) {
            message.push_str(&format!("  {key} @{line}\n"));
        }
        if added.len() > 80 {
            message.push_str(&format!("  ... and {} more\n", added.len() - 80));
        }
    }
    if !stale.is_empty() {
        message.push_str("stale checked sites:\n");
        for key in stale.iter().take(80) {
            message.push_str(&format!("  {key}\n"));
        }
        if stale.len() > 80 {
            message.push_str(&format!("  ... and {} more\n", stale.len() - 80));
        }
    }
    message
}

fn category_violations(
    frontier: &Frontier,
    literals: &BTreeMap<String, String>,
    readers: &BTreeMap<String, String>,
) -> Vec<String> {
    let mut violations = Vec::new();
    for (key, category) in literals {
        let name = literal_name_from_key(key);
        let registered = frontier.registered_names.contains(name);
        if registered != (category == "registered") {
            violations.push(format!(
                "literal {key} is category {category:?}, but registry membership is {registered}"
            ));
        }
    }
    for (key, category) in readers {
        let canonical = is_canonical_reader(key);
        if canonical != (category == "canonical") {
            violations.push(format!(
                "reader {key} is category {category:?}, but canonical-reader status is {canonical}"
            ));
        }
    }
    violations
}

fn update_requested() -> bool {
    // This test-only bootstrap switch deliberately uses the already-reviewed
    // whole-environment surface. A direct `var_os("NY_...")` here would add
    // one raw-reader occurrence to the very debt frontier this test protects.
    std::env::vars_os().any(|(name, value)| {
        name.as_os_str() == OsStr::new("UPDATE_NY_SOURCE_FRONTIER")
            && value.as_os_str() == OsStr::new("1")
    })
}

fn render_literal_manifest(frontier: &Frontier, existing: &BTreeMap<String, String>) -> String {
    let mut output = String::from(
        "# Exact executable Rust lever-name literal sites.\n\
         # category<TAB>path<TAB>structural-anchor<TAB>name<TAB>collision-ordinal\n\
         # `registered` is checked against ny_levers::all(); `permanent` is reviewed\n\
         # build/test/non-lever surface; `legacy` is temporary undeclared debt.\n",
    );
    for key in frontier.literals.keys() {
        let name = literal_name_from_key(key);
        let category = if frontier.registered_names.contains(name) {
            "registered"
        } else {
            existing
                .get(key)
                .filter(|category| matches!(category.as_str(), "permanent" | "legacy"))
                .map_or_else(
                    || {
                        if frontier
                            .reviewed_macro_literal_bases
                            .contains(base_from_key(key))
                        {
                            "permanent"
                        } else {
                            "legacy"
                        }
                    },
                    String::as_str,
                )
        };
        output.push_str(category);
        output.push('\t');
        output.push_str(key);
        output.push('\n');
    }
    output
}

fn render_reader_manifest(frontier: &Frontier, existing: &BTreeMap<String, String>) -> String {
    let reviewed_whole_environment_paths = [
        "crates/ny-cli/build.rs",
        "crates/ny-cli/src/commands/vnncomp_sweep.rs",
        "crates/ny-cli/src/flight.rs",
        "crates/ny-levers/tests/source_frontier.rs",
    ];
    let mut output = String::from(
        "# Exact executable Rust process-environment reader sites.\n\
         # category<TAB>path<TAB>structural-anchor<TAB>api<TAB>target<TAB>collision-ordinal\n\
         # `canonical` is the ny-levers chokepoint, `reviewed` is intentional\n\
         # non-lever/build/test capture, and `legacy` is migration debt.\n",
    );
    for key in frontier.readers.keys() {
        let (path, api, target) = reader_key_fields(key);
        let category = if is_canonical_reader(key) {
            "canonical"
        } else {
            existing
                .get(key)
                .filter(|category| matches!(category.as_str(), "reviewed" | "legacy"))
                .map_or_else(
                    || {
                        if (api.ends_with('!') && is_reviewed_build_macro_path(path))
                            || target == "<non-ny>"
                            || (target == "<all>"
                                && reviewed_whole_environment_paths.contains(&path))
                        {
                            "reviewed"
                        } else {
                            "legacy"
                        }
                    },
                    String::as_str,
                )
        };
        output.push_str(category);
        output.push('\t');
        output.push_str(key);
        output.push('\n');
    }
    output
}

fn assert_update_introduces_no_legacy_identity(
    label: &str,
    existing: &BTreeMap<String, String>,
    next: &BTreeMap<String, String>,
) {
    let introduced: Vec<_> = next
        .iter()
        .filter(|(key, category)| {
            category.as_str() == "legacy"
                && existing.get(*key).map(String::as_str) != Some("legacy")
        })
        .map(|(key, _category)| key)
        .collect();
    assert!(
        introduced.is_empty(),
        "refusing to update {label}: the scan introduced {} new legacy identity/identities:\n  \
         {}\nDeclare and route each runtime parameter, or classify an intentional \
         build/test/non-lever site through the reviewed policy. The updater may shrink the \
         legacy frontier but cannot bless new debt, including a one-for-one replacement.",
        introduced.len(),
        introduced
            .iter()
            .take(80)
            .map(|key| key.as_str())
            .collect::<Vec<_>>()
            .join("\n  "),
    );
}

fn update_manifests(
    frontier: &Frontier,
    literals: &BTreeMap<String, String>,
    readers: &BTreeMap<String, String>,
) {
    let rendered_literals = render_literal_manifest(frontier, literals);
    let rendered_readers = render_reader_manifest(frontier, readers);
    let next_literals = parse_manifest(
        "updated ny_literal_frontier.tsv",
        &rendered_literals,
        LITERAL_CATEGORIES,
    );
    let next_readers = parse_manifest(
        "updated ny_reader_frontier.tsv",
        &rendered_readers,
        READER_CATEGORIES,
    );
    assert_update_introduces_no_legacy_identity("literal frontier", literals, &next_literals);
    assert_update_introduces_no_legacy_identity("reader frontier", readers, &next_readers);

    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let data_dir = manifest_dir.join("tests/data");
    fs::create_dir_all(&data_dir).expect("frontier manifest directory must be creatable");
    fs::write(data_dir.join("ny_literal_frontier.tsv"), rendered_literals)
        .expect("literal frontier manifest must be writable");
    fs::write(data_dir.join("ny_reader_frontier.tsv"), rendered_readers)
        .expect("reader frontier manifest must be writable");
    eprintln!(
        "wrote {} literal and {} reader sites; rerun without UPDATE_NY_SOURCE_FRONTIER",
        frontier.literals.len(),
        frontier.readers.len()
    );
}

fn assert_frontier_shape(frontier: &Frontier) {
    assert!(
        frontier.literals.len() > 2_000,
        "expected thousands of executable lever-name literals; saw {}",
        frontier.literals.len()
    );
    assert!(
        frontier.readers.len() > 800,
        "expected hundreds of direct, alias, macro, whole-environment, and resolved helper reader sites; saw {}",
        frontier.readers.len()
    );
    assert!(
        frontier.registered_names.len() > 100,
        "expected the authoritative registry to contain over one hundred names; saw {}",
        frontier.registered_names.len()
    );
    let observed_names: BTreeSet<_> = frontier
        .literals
        .keys()
        .map(|key| literal_name_from_key(key))
        .collect();
    let missing_registered: Vec<_> = frontier
        .registered_names
        .iter()
        .filter(|name| !observed_names.contains(name.as_str()))
        .collect();
    assert!(
        missing_registered.is_empty(),
        "registered names missing from the executable literal inventory: {missing_registered:?}"
    );
}

#[test]
fn executable_ny_source_frontier_matches_reviewed_manifests() {
    let frontier = scan_frontier();
    assert_frontier_shape(&frontier);
    let literals = parse_manifest(
        "ny_literal_frontier.tsv",
        LITERAL_MANIFEST,
        LITERAL_CATEGORIES,
    );
    let readers = parse_manifest("ny_reader_frontier.tsv", READER_MANIFEST, READER_CATEGORIES);
    if update_requested() {
        update_manifests(&frontier, &literals, &readers);
        return;
    }

    let literal_diff = diff_message("literal", &frontier.literals, &literals);
    let reader_diff = diff_message("reader", &frontier.readers, &readers);
    let category_violations = category_violations(&frontier, &literals, &readers);
    assert!(
        literal_diff.is_empty() && reader_diff.is_empty() && category_violations.is_empty(),
        "NY source frontier changed. Do not bless an addition merely to make this test green. \n\
         Declare runtime levers in ny-levers and route their readers through a frozen LeverSet; \n\
         classify only intentional build/test/non-lever sites as permanent/reviewed; remove stale \n\
         manifest entries in the same change that removes their source. Structural anchors ignore \n\
         comments and line-number-only movement. The helper resolver covers a parameter passed \n\
         directly to std::env and literal/constant call arguments when the helper is locally or \n\
         globally unambiguous; aliases through re-exports, constructed names, closures, function \n\
         pointers, methods, and multi-hop helper chains remain explicit lexical limitations.\n\n\
         {literal_diff}{reader_diff}category violations:\n{}",
        category_violations
            .iter()
            .take(80)
            .map(|violation| format!("  {violation}"))
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[test]
fn legacy_parameter_debt_matches_the_shrinking_baseline_exactly() {
    let literals = parse_manifest(
        "ny_literal_frontier.tsv",
        LITERAL_MANIFEST,
        LITERAL_CATEGORIES,
    );
    let readers = parse_manifest("ny_reader_frontier.tsv", READER_MANIFEST, READER_CATEGORIES);
    let debt = legacy_debt(&literals, &readers);

    assert_exact_shrinking_ratchet(
        "undeclared executable lever-literal sites",
        debt.literal_sites,
        BASELINE_LEGACY_LITERAL_SITES,
    );
    assert_exact_shrinking_ratchet(
        "distinct undeclared executable lever names",
        debt.literal_names,
        BASELINE_LEGACY_LITERAL_NAMES,
    );
    assert_exact_shrinking_ratchet(
        "legacy process-environment reader sites",
        debt.reader_sites,
        BASELINE_LEGACY_READER_SITES,
    );
    assert_exact_frontier_identity(
        "undeclared executable lever-literal sites",
        debt.literal_site_fingerprint,
        BASELINE_LEGACY_LITERAL_SITE_FINGERPRINT,
    );
    assert_exact_frontier_identity(
        "distinct undeclared executable lever names",
        debt.literal_name_fingerprint,
        BASELINE_LEGACY_LITERAL_NAME_FINGERPRINT,
    );
    assert_exact_frontier_identity(
        "legacy process-environment reader sites",
        debt.reader_site_fingerprint,
        BASELINE_LEGACY_READER_SITE_FINGERPRINT,
    );
}

#[test]
fn frontier_updater_allows_only_legacy_identity_removal() {
    let existing = BTreeMap::from([
        ("legacy-a".to_owned(), "legacy".to_owned()),
        ("legacy-b".to_owned(), "legacy".to_owned()),
        ("reviewed".to_owned(), "reviewed".to_owned()),
    ]);
    let smaller = BTreeMap::from([
        ("legacy-a".to_owned(), "legacy".to_owned()),
        ("reviewed".to_owned(), "reviewed".to_owned()),
        ("new-canonical".to_owned(), "canonical".to_owned()),
    ]);
    assert_update_introduces_no_legacy_identity("fixture", &existing, &smaller);
}

#[test]
#[should_panic(expected = "cannot bless new debt")]
fn frontier_updater_refuses_one_for_one_legacy_substitution() {
    let existing = BTreeMap::from([
        ("legacy-retired".to_owned(), "legacy".to_owned()),
        ("legacy-kept".to_owned(), "legacy".to_owned()),
    ]);
    let substituted = BTreeMap::from([
        ("legacy-new-master".to_owned(), "legacy".to_owned()),
        ("legacy-kept".to_owned(), "legacy".to_owned()),
    ]);
    assert_update_introduces_no_legacy_identity("fixture", &existing, &substituted);
}

fn analyzed_fixture(source: &str) -> SourceFile {
    let tokens = rust_tokens(source);
    let mut file = SourceFile {
        relative: "fixture.rs".to_owned(),
        source: source.to_owned(),
        tokens,
        aliases: BTreeMap::new(),
        env_modules: BTreeSet::new(),
        env_functions: BTreeMap::new(),
        functions: Vec::new(),
    };
    file.aliases = collect_aliases(&file);
    (file.env_modules, file.env_functions) = collect_env_imports(&file);
    file.functions = collect_functions(&file);
    file
}

#[test]
fn lexical_matcher_covers_multiline_alias_macro_helper_and_whole_environment_reads() {
    let fixture = r####"
use std::env as process_env;
use std::env::var_os as read_os;
const KEY: &str = "NY_ALIAS";

fn env_truthy(key: &str) -> bool {
    process_env::var(key).is_ok()
}

fn exercise() {
    process_env::var(
        "\x4eY_ESCAPED",
    );
    read_os(KEY);
    process_env::vars_os();
    env_truthy("NY_HELPER");
    // process_env :: var("NY_COMMENT") must not become a token.
}

const BUILD: Option<&str> = option_env!("NY_BUILD_FIXTURE");
"####;
    let files = vec![analyzed_fixture(fixture)];
    let global = global_aliases(&files);
    let raw_calls = collect_raw_calls(&files, &global);
    let found: BTreeSet<_> = raw_calls
        .iter()
        .map(|call| (call.api.as_str(), call.target.as_str()))
        .collect();
    assert_eq!(
        found,
        BTreeSet::from([
            ("var", "<dynamic>"),
            ("var", "NY_ESCAPED"),
            ("var_os", "alias=NY_ALIAS"),
            ("vars_os", "<all>"),
        ])
    );

    let helpers = collect_helper_defs(&files, &raw_calls);
    assert!(helpers.iter().any(|helper| {
        helper.file == 0 && helper.name == "env_truthy" && helper.parameter == 0
    }));
    let file = &files[0];
    let helper_call = file
        .tokens
        .iter()
        .enumerate()
        .find(|(index, token)| {
            token.kind == TokenKind::Ident
                && text(file, token) == "env_truthy"
                && !is_function_declaration(file, *index)
        })
        .map(|(index, _)| index)
        .expect("fixture helper call must be tokenized");
    let arguments = split_arguments(file, helper_call + 1).expect("helper call must parse");
    let (start, end) = arguments[0];
    assert_eq!(argument_target(file, &global, start, end).0, "NY_HELPER");

    let macros: Vec<_> = (0..file.tokens.len())
        .filter_map(|index| macro_reader_at(file, index))
        .collect();
    assert_eq!(macros.len(), 1);
    assert_eq!(
        decoded_string(file, &file.tokens[macros[0].1]).as_deref(),
        Some("NY_BUILD_FIXTURE")
    );
}

#[test]
fn manifest_diff_reports_additions_and_stale_entries_independently() {
    let actual = BTreeMap::from([("current".to_owned(), 7_usize)]);
    let expected = BTreeMap::from([("stale".to_owned(), "legacy".to_owned())]);
    let message = diff_message("fixture", &actual, &expected);
    assert!(message.contains("1 unreviewed current site(s)"));
    assert!(message.contains("1 stale manifest site(s)"));
    assert!(message.contains("current @7"));
    assert!(message.contains("stale checked sites:\n  stale"));
}
