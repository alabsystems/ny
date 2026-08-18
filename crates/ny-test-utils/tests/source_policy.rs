// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Repository-wide test-source policy.

use std::ffi::OsStr;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .ancestors()
        .find(|candidate| {
            fs::read_to_string(candidate.join("Cargo.toml"))
                .is_ok_and(|manifest| manifest.lines().any(|line| line.trim() == "[workspace]"))
        })
        .unwrap_or_else(|| {
            panic!(
                "could not find workspace Cargo.toml above {}",
                manifest_dir.display()
            )
        })
        .to_path_buf()
}

fn is_build_or_metadata_dir(name: &OsStr) -> bool {
    let name = name.to_string_lossy();
    name.starts_with('.')
        || name == "target"
        || name.starts_with("target-")
        || name.starts_with("target_")
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TokenKind<'source> {
    Ident(&'source str),
    String(&'source str),
    Punct(u8),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Token<'source> {
    kind: TokenKind<'source>,
    line: usize,
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

/// Tokenize only the Rust syntax needed to recognize attributes.
///
/// Comments and literals are discarded so documentation such as `#[ignore]`
/// cannot trip the policy. Block comments nest, and raw strings are handled
/// explicitly because both commonly contain source snippets.
fn rust_attribute_tokens(source: &str) -> Vec<Token<'_>> {
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
                kind: TokenKind::String(&source[content_start..content_end]),
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
                kind: TokenKind::String(&source[content_start..content_end]),
                line: literal_line,
            });
            continue;
        }
        if bytes[cursor] == b'\'' {
            // A lifetime (`'a`) is not a literal. A one-character or escaped
            // character followed by a closing quote is.
            let literal_end = if bytes.get(cursor + 1) == Some(&b'\\') {
                (cursor + 3 < bytes.len() && bytes[cursor + 3] == b'\'').then_some(cursor + 4)
            } else {
                (cursor + 2 < bytes.len() && bytes[cursor + 2] == b'\'').then_some(cursor + 3)
            };
            if let Some(end) = literal_end {
                cursor = end;
                continue;
            }
            cursor += 1;
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
                kind: TokenKind::Ident(&source[start..cursor]),
                line,
            });
            continue;
        }
        if matches!(
            bytes[cursor],
            b'#' | b'!' | b'[' | b']' | b'(' | b')' | b'{' | b'}' | b',' | b'&'
        ) {
            tokens.push(Token {
                kind: TokenKind::Punct(bytes[cursor]),
                line,
            });
        }
        cursor += 1;
    }

    tokens
}

fn split_top_level_commas<'a>(tokens: &'a [Token<'a>]) -> Vec<&'a [Token<'a>]> {
    let mut parts = Vec::new();
    let mut start = 0;
    let mut round = 0_usize;
    let mut square = 0_usize;
    let mut curly = 0_usize;

    for (index, token) in tokens.iter().enumerate() {
        match token.kind {
            TokenKind::Punct(b'(') => round += 1,
            TokenKind::Punct(b')') => round = round.saturating_sub(1),
            TokenKind::Punct(b'[') => square += 1,
            TokenKind::Punct(b']') => square = square.saturating_sub(1),
            TokenKind::Punct(b'{') => curly += 1,
            TokenKind::Punct(b'}') => curly = curly.saturating_sub(1),
            TokenKind::Punct(b',') if round == 0 && square == 0 && curly == 0 => {
                parts.push(&tokens[start..index]);
                start = index + 1;
            }
            _ => {}
        }
    }
    parts.push(&tokens[start..]);
    parts
}

fn meta_applies_ignore(tokens: &[Token<'_>]) -> bool {
    let Some(Token {
        kind: TokenKind::Ident(name),
        ..
    }) = tokens.first()
    else {
        return false;
    };
    if *name == "ignore" {
        return true;
    }
    if *name != "cfg_attr" || tokens.get(1).map(|token| token.kind) != Some(TokenKind::Punct(b'('))
    {
        return false;
    }

    let Some(close) = tokens
        .iter()
        .rposition(|token| token.kind == TokenKind::Punct(b')'))
    else {
        return false;
    };
    let parts = split_top_level_commas(&tokens[2..close]);
    parts.iter().skip(1).any(|part| meta_applies_ignore(part))
}

fn prohibited_ignore_attribute_lines(source: &str) -> Vec<usize> {
    let tokens = rust_attribute_tokens(source);
    let mut lines = Vec::new();
    let mut cursor = 0;

    while cursor < tokens.len() {
        if tokens[cursor].kind != TokenKind::Punct(b'#') {
            cursor += 1;
            continue;
        }
        let line = tokens[cursor].line;
        let mut opening = cursor + 1;
        if tokens.get(opening).map(|token| token.kind) == Some(TokenKind::Punct(b'!')) {
            opening += 1;
        }
        if tokens.get(opening).map(|token| token.kind) != Some(TokenKind::Punct(b'[')) {
            cursor += 1;
            continue;
        }

        let mut depth = 1_usize;
        let mut closing = opening + 1;
        while closing < tokens.len() && depth > 0 {
            match tokens[closing].kind {
                TokenKind::Punct(b'[') => depth += 1,
                TokenKind::Punct(b']') => depth -= 1,
                _ => {}
            }
            closing += 1;
        }
        if depth == 0 && meta_applies_ignore(&tokens[opening + 1..closing - 1]) {
            lines.push(line);
        }
        cursor = closing.max(cursor + 1);
    }

    lines
}

fn doctest_fence_applies_ignore(documentation: &str) -> bool {
    let documentation = documentation.trim_start();
    let Some(marker) = documentation.as_bytes().first().copied() else {
        return false;
    };
    if !matches!(marker, b'`' | b'~') {
        return false;
    }
    let marker_count = documentation
        .as_bytes()
        .iter()
        .take_while(|byte| **byte == marker)
        .count();
    if marker_count < 3 {
        return false;
    }

    documentation[marker_count..]
        .split(|character: char| character == ',' || character.is_whitespace())
        .any(|tag| tag == "ignore" || tag.starts_with("ignore-"))
}

fn prohibited_doctest_ignore_lines(source: &str) -> Vec<usize> {
    let bytes = source.as_bytes();
    let mut lines = Vec::new();
    let mut cursor = 0;
    let mut line = 1;

    while cursor < bytes.len() {
        if bytes[cursor] == b'\n' {
            line += 1;
            cursor += 1;
            continue;
        }
        if bytes[cursor..].starts_with(b"//") {
            let comment_start = cursor;
            cursor += 2;
            while cursor < bytes.len() && bytes[cursor] != b'\n' {
                cursor += 1;
            }
            let is_outer_doc = bytes[comment_start..].starts_with(b"///")
                && !bytes[comment_start..].starts_with(b"////");
            let is_inner_doc = bytes[comment_start..].starts_with(b"//!");
            if (is_outer_doc || is_inner_doc)
                && doctest_fence_applies_ignore(&source[comment_start + 3..cursor])
            {
                lines.push(line);
            }
            continue;
        }
        if bytes[cursor..].starts_with(b"/*") {
            let comment_line = line;
            let is_outer_doc =
                bytes[cursor..].starts_with(b"/**") && !bytes[cursor..].starts_with(b"/***");
            let is_inner_doc = bytes[cursor..].starts_with(b"/*!");
            cursor += 2;
            let content_start = cursor + usize::from(is_outer_doc || is_inner_doc);
            let mut content_end = cursor;
            let mut depth = 1_usize;
            while cursor < bytes.len() && depth > 0 {
                if bytes[cursor..].starts_with(b"/*") {
                    depth += 1;
                    cursor += 2;
                } else if bytes[cursor..].starts_with(b"*/") {
                    depth -= 1;
                    content_end = cursor;
                    cursor += 2;
                } else {
                    line += usize::from(bytes[cursor] == b'\n');
                    cursor += 1;
                }
            }
            if depth > 0 {
                content_end = cursor;
            }
            if is_outer_doc || is_inner_doc {
                for (offset, documentation) in
                    source[content_start..content_end].lines().enumerate()
                {
                    let documentation = documentation
                        .trim_start()
                        .strip_prefix('*')
                        .unwrap_or(documentation.trim_start());
                    if doctest_fence_applies_ignore(documentation) {
                        lines.push(comment_line + offset);
                    }
                }
            }
            continue;
        }
        if let Some((content_start, hashes)) = raw_string_prefix(bytes, cursor) {
            cursor = content_start;
            while cursor < bytes.len() {
                line += usize::from(bytes[cursor] == b'\n');
                if bytes[cursor] == b'"'
                    && cursor + 1 + hashes <= bytes.len()
                    && bytes[cursor + 1..cursor + 1 + hashes]
                        .iter()
                        .all(|byte| *byte == b'#')
                {
                    cursor += hashes + 1;
                    break;
                }
                cursor += 1;
            }
            continue;
        }
        if bytes[cursor] == b'"' {
            cursor += 1;
            while cursor < bytes.len() {
                line += usize::from(bytes[cursor] == b'\n');
                match bytes[cursor] {
                    b'\\' => {
                        line += usize::from(bytes.get(cursor + 1) == Some(&b'\n'));
                        cursor = (cursor + 2).min(bytes.len());
                    }
                    b'"' => {
                        cursor += 1;
                        break;
                    }
                    _ => cursor += 1,
                }
            }
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
        cursor += 1;
    }

    lines
}

fn inspect_rust_source(
    workspace: &Path,
    source: &Path,
    violations: &mut Vec<String>,
) -> io::Result<()> {
    let contents = fs::read_to_string(source)?;
    let relative = source.strip_prefix(workspace).unwrap_or(source);
    let mut prohibited_lines = prohibited_ignore_attribute_lines(&contents);
    prohibited_lines.extend(prohibited_doctest_ignore_lines(&contents));
    prohibited_lines.sort_unstable();
    prohibited_lines.dedup();
    for line in prohibited_lines {
        violations.push(format!("{}:{line}", relative.display()));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PythonTokenKind<'source> {
    Ident(&'source str),
    Dot,
    Comma,
    OpenParen,
    CloseParen,
    Newline,
    Semicolon,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PythonToken<'source> {
    kind: PythonTokenKind<'source>,
    line: usize,
}

fn python_string_opening(bytes: &[u8], cursor: usize) -> Option<(usize, u8)> {
    if matches!(bytes.get(cursor), Some(b'\'' | b'"')) {
        return Some((cursor, bytes[cursor]));
    }

    // Python accepts case-insensitive combinations such as r, b, f, u, br,
    // rb, fr, and rf. Accepting up to three prefix letters also covers the
    // template-string prefix introduced by newer interpreters. This scanner is
    // deliberately lexical: the Python parser remains the authority on whether
    // a particular prefix combination is legal.
    let mut quote = cursor;
    while quote < bytes.len()
        && quote - cursor < 3
        && matches!(
            bytes[quote].to_ascii_lowercase(),
            b'r' | b'b' | b'f' | b'u' | b't'
        )
    {
        quote += 1;
    }
    (quote > cursor && matches!(bytes.get(quote), Some(b'\'' | b'"')))
        .then_some((quote, bytes[quote]))
}

/// Tokenize the small Python grammar needed by the no-skip policy.
///
/// Comments and string literals (including triple-quoted and prefixed forms)
/// are discarded. Newlines remain tokens so import aliases do not leak across
/// statements, while the matcher may still recognize a dotted expression
/// split across a parenthesized or explicitly continued line.
fn python_policy_tokens(source: &str) -> Vec<PythonToken<'_>> {
    let bytes = source.as_bytes();
    let mut tokens = Vec::new();
    let mut cursor = 0;
    let mut line = 1;

    while cursor < bytes.len() {
        if bytes[cursor] == b'\n' {
            tokens.push(PythonToken {
                kind: PythonTokenKind::Newline,
                line,
            });
            line += 1;
            cursor += 1;
            continue;
        }
        if bytes[cursor] == b'#' {
            while cursor < bytes.len() && bytes[cursor] != b'\n' {
                cursor += 1;
            }
            continue;
        }
        if bytes[cursor] == b'\\' && bytes.get(cursor + 1) == Some(&b'\n') {
            line += 1;
            cursor += 2;
            continue;
        }
        if let Some((quote_cursor, quote)) = python_string_opening(bytes, cursor) {
            let triple = bytes.get(quote_cursor..quote_cursor + 3) == Some(&[quote; 3]);
            cursor = quote_cursor + if triple { 3 } else { 1 };
            while cursor < bytes.len() {
                if bytes[cursor] == b'\\' {
                    line += usize::from(bytes.get(cursor + 1) == Some(&b'\n'));
                    cursor = (cursor + 2).min(bytes.len());
                } else if triple && bytes.get(cursor..cursor + 3) == Some(&[quote; 3]) {
                    cursor += 3;
                    break;
                } else if !triple && bytes[cursor] == quote {
                    cursor += 1;
                    break;
                } else {
                    line += usize::from(bytes[cursor] == b'\n');
                    cursor += 1;
                }
            }
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
            tokens.push(PythonToken {
                kind: PythonTokenKind::Ident(&source[start..cursor]),
                line,
            });
            continue;
        }
        let kind = match bytes[cursor] {
            b'.' => Some(PythonTokenKind::Dot),
            b',' => Some(PythonTokenKind::Comma),
            b'(' | b'[' | b'{' => Some(PythonTokenKind::OpenParen),
            b')' | b']' | b'}' => Some(PythonTokenKind::CloseParen),
            b';' => Some(PythonTokenKind::Semicolon),
            _ => None,
        };
        if let Some(kind) = kind {
            tokens.push(PythonToken { kind, line });
        }
        cursor += 1;
    }

    tokens
}

fn python_ident(token: PythonToken<'_>) -> Option<&str> {
    match token.kind {
        PythonTokenKind::Ident(identifier) => Some(identifier),
        _ => None,
    }
}

fn next_python_non_newline(tokens: &[PythonToken<'_>], mut cursor: usize) -> Option<usize> {
    while tokens
        .get(cursor)
        .is_some_and(|token| token.kind == PythonTokenKind::Newline)
    {
        cursor += 1;
    }
    (cursor < tokens.len()).then_some(cursor)
}

fn push_unique<'source>(items: &mut Vec<&'source str>, item: &'source str) {
    if !items.contains(&item) {
        items.push(item);
    }
}

#[derive(Default)]
struct PythonSkipAliases<'source> {
    pytest_modules: Vec<&'source str>,
    pytest_marks: Vec<&'source str>,
    pytest_skip_symbols: Vec<&'source str>,
    unittest_modules: Vec<&'source str>,
    unittest_skip_symbols: Vec<&'source str>,
}

fn collect_python_skip_aliases<'source>(
    tokens: &[PythonToken<'source>],
) -> PythonSkipAliases<'source> {
    let mut aliases = PythonSkipAliases {
        pytest_modules: vec!["pytest"],
        unittest_modules: vec!["unittest"],
        ..PythonSkipAliases::default()
    };

    for (cursor, token) in tokens.iter().copied().enumerate() {
        match python_ident(token) {
            Some("import") => {
                let mut import_cursor = cursor + 1;
                let mut expects_module = true;
                while let Some(next) = tokens.get(import_cursor).copied() {
                    match next.kind {
                        PythonTokenKind::Newline | PythonTokenKind::Semicolon => break,
                        PythonTokenKind::Comma => {
                            expects_module = true;
                            import_cursor += 1;
                        }
                        PythonTokenKind::Ident(module) if expects_module => {
                            let alias_cursor = next_python_non_newline(tokens, import_cursor + 1)
                                .filter(|candidate| python_ident(tokens[*candidate]) == Some("as"))
                                .and_then(|as_cursor| {
                                    next_python_non_newline(tokens, as_cursor + 1)
                                });
                            let alias = alias_cursor
                                .and_then(|candidate| python_ident(tokens[candidate]))
                                .unwrap_or(module);
                            if module == "pytest" {
                                push_unique(&mut aliases.pytest_modules, alias);
                            } else if module == "unittest" {
                                push_unique(&mut aliases.unittest_modules, alias);
                            }
                            expects_module = false;
                            import_cursor =
                                alias_cursor.map_or(import_cursor + 1, |alias| alias + 1);
                        }
                        _ => {
                            expects_module = false;
                            import_cursor += 1;
                        }
                    }
                }
            }
            Some("from") => {
                let Some(module_cursor) = next_python_non_newline(tokens, cursor + 1) else {
                    continue;
                };
                let Some(module) = python_ident(tokens[module_cursor]) else {
                    continue;
                };
                if !matches!(module, "pytest" | "unittest") {
                    continue;
                }

                let mut import_cursor = module_cursor + 1;
                let mut from_pytest_mark = false;
                while let Some(next) = next_python_non_newline(tokens, import_cursor) {
                    match (tokens[next].kind, python_ident(tokens[next])) {
                        (_, Some("import")) => {
                            import_cursor = next + 1;
                            break;
                        }
                        (PythonTokenKind::Dot, _) if module == "pytest" => {
                            from_pytest_mark = next_python_non_newline(tokens, next + 1)
                                .and_then(|part| python_ident(tokens[part]))
                                == Some("mark");
                        }
                        (PythonTokenKind::Newline | PythonTokenKind::Semicolon, _) => break,
                        _ => {}
                    }
                    import_cursor = next + 1;
                }

                let mut depth = 0_usize;
                while let Some(next) = tokens.get(import_cursor).copied() {
                    match next.kind {
                        PythonTokenKind::OpenParen => {
                            depth += 1;
                            import_cursor += 1;
                        }
                        PythonTokenKind::CloseParen => {
                            depth = depth.saturating_sub(1);
                            import_cursor += 1;
                        }
                        PythonTokenKind::Newline if depth == 0 => break,
                        PythonTokenKind::Semicolon => break,
                        PythonTokenKind::Newline | PythonTokenKind::Comma => {
                            import_cursor += 1;
                        }
                        PythonTokenKind::Ident(imported) if imported != "as" => {
                            let alias_cursor = next_python_non_newline(tokens, import_cursor + 1);
                            let (alias, advance_to) = alias_cursor
                                .filter(|candidate| python_ident(tokens[*candidate]) == Some("as"))
                                .and_then(|as_cursor| {
                                    next_python_non_newline(tokens, as_cursor + 1).map(
                                        |name_cursor| {
                                            (
                                                python_ident(tokens[name_cursor])
                                                    .unwrap_or(imported),
                                                name_cursor + 1,
                                            )
                                        },
                                    )
                                })
                                .unwrap_or((imported, import_cursor + 1));

                            if module == "pytest" {
                                if imported == "mark" && !from_pytest_mark {
                                    push_unique(&mut aliases.pytest_marks, alias);
                                } else if (from_pytest_mark
                                    && matches!(imported, "skip" | "skipif"))
                                    || (!from_pytest_mark
                                        && matches!(imported, "skip" | "skipif" | "importorskip"))
                                {
                                    push_unique(&mut aliases.pytest_skip_symbols, alias);
                                }
                            } else if matches!(
                                imported,
                                "skip" | "skipIf" | "skipUnless" | "skipTest" | "SkipTest"
                            ) {
                                push_unique(&mut aliases.unittest_skip_symbols, alias);
                            }
                            import_cursor = advance_to;
                        }
                        _ => import_cursor += 1,
                    }
                }
            }
            _ => {}
        }
    }

    aliases
}

fn dotted_python_identifier<'source>(
    tokens: &[PythonToken<'source>],
    cursor: usize,
) -> Option<(usize, &'source str)> {
    let dot = next_python_non_newline(tokens, cursor + 1)?;
    if tokens[dot].kind != PythonTokenKind::Dot {
        return None;
    }
    let identifier = next_python_non_newline(tokens, dot + 1)?;
    python_ident(tokens[identifier]).map(|name| (identifier, name))
}

fn prohibited_python_skip_lines(source: &str) -> Vec<usize> {
    let tokens = python_policy_tokens(source);
    let aliases = collect_python_skip_aliases(&tokens);
    let mut lines = Vec::new();

    for (cursor, token) in tokens.iter().copied().enumerate() {
        let Some(identifier) = python_ident(token) else {
            continue;
        };
        if aliases.pytest_skip_symbols.contains(&identifier)
            || aliases.unittest_skip_symbols.contains(&identifier)
        {
            lines.push(token.line);
            continue;
        }

        let Some((attribute_cursor, attribute)) = dotted_python_identifier(&tokens, cursor) else {
            continue;
        };
        let is_pytest_module_skip = aliases.pytest_modules.contains(&identifier)
            && (matches!(attribute, "skip" | "skipif" | "importorskip")
                || attribute == "mark"
                    && dotted_python_identifier(&tokens, attribute_cursor)
                        .is_some_and(|(_, marker)| matches!(marker, "skip" | "skipif")));
        let is_pytest_mark_skip =
            aliases.pytest_marks.contains(&identifier) && matches!(attribute, "skip" | "skipif");
        let is_unittest_module_skip = aliases.unittest_modules.contains(&identifier)
            && (matches!(
                attribute,
                "skip" | "skipIf" | "skipUnless" | "skipTest" | "SkipTest"
            ) || attribute == "case"
                && dotted_python_identifier(&tokens, attribute_cursor).is_some_and(
                    |(_, member)| {
                        matches!(
                            member,
                            "skip" | "skipIf" | "skipUnless" | "skipTest" | "SkipTest"
                        )
                    },
                ));

        if attribute == "skipTest"
            || is_pytest_module_skip
            || is_pytest_mark_skip
            || is_unittest_module_skip
        {
            lines.push(token.line);
        }
    }

    lines.sort_unstable();
    lines.dedup();
    lines
}

fn inspect_python_source(
    workspace: &Path,
    source: &Path,
    violations: &mut Vec<String>,
) -> io::Result<()> {
    let contents = fs::read_to_string(source)?;
    let relative = source.strip_prefix(workspace).unwrap_or(source);
    for line in prohibited_python_skip_lines(&contents) {
        violations.push(format!("{}:{line}", relative.display()));
    }
    Ok(())
}

fn has_environment_mutation_lint_waiver(source: &str) -> bool {
    rust_attribute_tokens(source).windows(4).any(|tokens| {
        matches!(tokens[0].kind, TokenKind::Ident("allow"))
            && matches!(tokens[1].kind, TokenKind::Punct(b'('))
            && matches!(tokens[2].kind, TokenKind::Ident("clippy"))
            && matches!(tokens[3].kind, TokenKind::Ident("disallowed_methods"))
    })
}

fn prohibited_ay_environment_write_lines(source: &str) -> Vec<usize> {
    let tokens = rust_attribute_tokens(source);
    let mut lines = Vec::new();
    for (index, token) in tokens.iter().enumerate() {
        let TokenKind::Ident(function) = token.kind else {
            continue;
        };
        if !matches!(function, "set_var" | "remove_var") {
            continue;
        }
        let mut argument = index + 1;
        if tokens.get(argument).map(|token| token.kind) != Some(TokenKind::Punct(b'(')) {
            continue;
        }
        argument += 1;
        if tokens.get(argument).map(|token| token.kind) == Some(TokenKind::Punct(b'&')) {
            argument += 1;
        }
        if matches!(
            tokens.get(argument).map(|token| token.kind),
            Some(TokenKind::String(key)) if key.starts_with("AY_")
        ) {
            lines.push(token.line);
        }
    }
    lines
}

fn inspect_ay_environment_source(
    workspace: &Path,
    source: &Path,
    violations: &mut Vec<String>,
) -> io::Result<()> {
    let contents = fs::read_to_string(source)?;
    let relative = source.strip_prefix(workspace).unwrap_or(source);
    for line in prohibited_ay_environment_write_lines(&contents) {
        violations.push(format!(
            "{}:{line} writes an AY_* environment variable",
            relative.display()
        ));
    }

    if has_environment_mutation_lint_waiver(&contents)
        && relative != Path::new("crates/ny-test-utils/src/env.rs")
    {
        violations.push(format!(
            "{} waives the workspace environment-mutation lint",
            relative.display()
        ));
    }
    Ok(())
}

type SourceInspector = fn(&Path, &Path, &mut Vec<String>) -> io::Result<()>;

fn inspect_tree(
    workspace: &Path,
    directory: &Path,
    extension: &OsStr,
    inspect_source: SourceInspector,
    violations: &mut Vec<String>,
) -> io::Result<()> {
    // Nested repositories are downloaded dependencies or research inputs, not
    // NY source. Do not impose this repository's policy on them.
    if directory != workspace && directory.join(".git").exists() {
        return Ok(());
    }

    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let path = entry.path();

        if file_type.is_dir() {
            if !is_build_or_metadata_dir(&entry.file_name()) {
                inspect_tree(workspace, &path, extension, inspect_source, violations)?;
            }
        } else if file_type.is_file() && path.extension() == Some(extension) {
            inspect_source(workspace, &path, violations)?;
        }
    }
    Ok(())
}

#[test]
fn rust_tests_cannot_be_ignored() {
    let workspace = workspace_root();
    let mut violations = Vec::new();
    inspect_tree(
        &workspace,
        &workspace,
        OsStr::new("rs"),
        inspect_rust_source,
        &mut violations,
    )
    .unwrap_or_else(|error| panic!("failed to inspect {}: {error}", workspace.display()));
    violations.sort_unstable();

    assert!(
        violations.is_empty(),
        "Rust test ignore attributes are prohibited; use a hermetic test or an explicit \
         fail-fast conformance/measurement lane. Rustdoc `ignore` fences are prohibited too \
         (see TEST_CONFORMANCE.md):\n{}",
        violations.join("\n")
    );
}

#[test]
fn python_tests_and_tools_cannot_skip() {
    let workspace = workspace_root();
    let mut violations = Vec::new();
    inspect_tree(
        &workspace,
        &workspace,
        OsStr::new("py"),
        inspect_python_source,
        &mut violations,
    )
    .unwrap_or_else(|error| panic!("failed to inspect {}: {error}", workspace.display()));
    violations.sort_unstable();

    assert!(
        violations.is_empty(),
        "Python tests and tools must not skip or conditionally disappear. pytest \
         skip/importorskip/skipif and unittest skip/SkipTest constructs are prohibited; use a \
         hermetic fixture or an explicit fail-fast conformance/measurement lane (see \
         TEST_CONFORMANCE.md):\n{}",
        violations.join("\n")
    );
}

#[test]
fn ay_configuration_never_uses_process_environment_writes() {
    let workspace = workspace_root();
    let crates = workspace.join("crates");
    let mut violations = Vec::new();
    inspect_tree(
        &workspace,
        &crates,
        OsStr::new("rs"),
        inspect_ay_environment_source,
        &mut violations,
    )
    .unwrap_or_else(|error| panic!("failed to inspect {}: {error}", crates.display()));
    violations.sort_unstable();
    assert!(
        violations.is_empty(),
        "AY configuration travels through typed per-solve options. Process-global AY_* writes or \
         new clippy environment-mutation waivers reintroduce a setenv/getenv race:\n{}",
        violations.join("\n")
    );

    let blessed = fs::read_to_string(workspace.join("crates/ny-test-utils/src/env.rs"))
        .expect("the serialized test-only environment helper must exist");
    assert!(
        has_environment_mutation_lint_waiver(&blessed),
        "update the source-policy contract if the blessed serialized helper no longer needs its \
         sole environment-mutation waiver"
    );
}

#[test]
fn ay_environment_matcher_handles_multiline_calls_and_ignores_text() {
    let fixture = r####"
std::env::set_var("AY_DIRECT", "1");
std::env::remove_var(
    &"AY_MULTILINE",
);
std::env::set_var(r#"AY_RAW"#, "1");
std::env::set_var("NOT_AY", "1");
// std::env::set_var("AY_COMMENT", "1");
const TEXT: &str = r#"std::env::remove_var("AY_STRING")"#;
set_variant("AY_SIMILAR_NAME", "1");
"####;
    assert_eq!(
        prohibited_ay_environment_write_lines(fixture),
        vec![2, 3, 6]
    );
}

#[test]
fn policy_matcher_distinguishes_attributes_from_comments() {
    let fixture = r####"
#[ignore = "external fixture"]
# [ ignore ]
#[cfg_attr(feature = "slow", ignore)]
#[cfg_attr(
    feature = "external",
    cfg_attr(unix, ignore = "nested and conditional"),
)]
// #[ignore] is forbidden
/* #[cfg_attr(any(), ignore)] is only documentation */
const POLICY_TEXT: &str = "#[ignore]";
const RAW_POLICY_TEXT: &str = r###"#[cfg_attr(all(), ignore)]"###;
#[ignored_but_not_the_attribute]
#[cfg_attr(test, ntest::timeout(1000))]
"####;
    assert_eq!(prohibited_ignore_attribute_lines(fixture), vec![2, 3, 4, 5]);
}

#[test]
fn policy_matcher_rejects_ignored_doctests_only_in_documentation() {
    let fixture = r####"
/// ```ignore
/// skipped example
/// ```
//! ~~~rust,ignore-x86_64
//! ~~~
// ```ignore is an ordinary comment
const POLICY_TEXT: &str = "/// ```ignore";
/**
 * ```rust, ignore
 */
/*! ```ignore */
//// ```ignore is not a Rustdoc comment
"####;
    assert_eq!(prohibited_doctest_ignore_lines(fixture), vec![2, 5, 10, 12]);
}

#[test]
fn python_policy_matcher_handles_multiline_aliases_comments_and_strings() {
    let fixture = r####"
pytest.skip("missing")
pytest \
    .importorskip("onnx")
@pytest.mark.skipif(
    unavailable,
    reason="fixture",
)
@(
    pytest  # a comment inside a valid multiline decorator expression
    .mark
    .skip
)
def hidden(): ...
import pytest as pt
pt.skip("aliased")
from pytest import importorskip as require
require("numpy")
from pytest import mark as pm
@pm.skipif(False, reason="still prohibited")
import unittest as unit
@unit.skipUnless(True, "still prohibited")
class Case(unit.TestCase):
    def test_contract(self):
        self.skipTest("missing")
from unittest import skip as omit
@omit("missing")
def hidden_too(): ...
# pytest.skip("comment")
TEXT = "pytest.importorskip('string')"
RAW = r'''@pytest.mark.skipif(True, reason="string")'''
pytest.raises(ValueError)
@pytest.mark.parametrize("value", [1])
def present(value): ...
"####;
    assert_eq!(
        prohibited_python_skip_lines(fixture),
        vec![2, 3, 5, 10, 16, 17, 18, 20, 22, 25, 26, 27]
    );
}

#[test]
fn python_policy_matcher_tracks_comma_separated_module_aliases() {
    let fixture = r#"
import os, pytest as check, unittest as cases
check.skip("missing")
@cases.skipIf(False, "still prohibited")
raise unittest.case.SkipTest("missing")
from unittest.case import skipUnless as omit_unless
@omit_unless(False, "still prohibited")
"#;
    assert_eq!(prohibited_python_skip_lines(fixture), vec![3, 4, 5, 6, 7]);
}
