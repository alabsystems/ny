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
            // Clamp to `content_start`, never to the raw cursor. For a DOC block
            // comment `content_start` is one past the cursor (it skips the third
            // byte of `/**` / `/*!`), so an EMPTY one — `/**/`, which `/**`
            // matches — closes on the first loop iteration with the delimiter
            // sitting AT the cursor, leaving content_end one BELOW content_start
            // and slicing an inverted range:
            //   byte range starts at 201603 but ends at 201602
            // That is a panic in the checker, not a policy finding, so the whole
            // `rust_tests_cannot_be_ignored` lane aborted rather than reporting.
            // Live trigger in-tree: the `/**/` column separators in
            // ny-trace-bridge's ops_misc.rs vec! literals.
            let mut content_end = content_start;
            let mut depth = 1_usize;
            while cursor < bytes.len() && depth > 0 {
                if bytes[cursor..].starts_with(b"/*") {
                    depth += 1;
                    cursor += 2;
                } else if bytes[cursor..].starts_with(b"*/") {
                    depth -= 1;
                    content_end = cursor.max(content_start);
                    cursor += 2;
                } else {
                    line += usize::from(bytes[cursor] == b'\n');
                    cursor += 1;
                }
            }
            if depth > 0 {
                content_end = cursor.max(content_start);
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

// ---------------------------------------------------------------------------
// Shell portability: the M0 runner must run under bash 3.2.
//
// macOS ships bash 3.2 as /bin/bash and ny's shell tools run there. The M0
// runner (scripts/m0/moat_both_profiles.sh) once used bash 4's `mapfile` and
// `declare -A`; bash 3.2 does not merely reject `declare -A`, it runs the
// assignment on an INDEXED array whose string subscripts evaluate
// arithmetically (docs/DEFECT_REGISTER_2026-07-31.md D1 `#bash32-declare-A`).
// The runner's own self-test catches that only on a host whose shell is bash
// 3.2 (ny-test-utils' `python_tooling_black_box` runs it with
// `--bash /bin/bash`); where the runner is edited, on Linux, /bin/bash is
// bash 5 and every such construct passes. This lexical scan needs no shell
// and no interpreter, so it is a red on every host, and it also covers the
// Linux-only `--launch` path the self-test cannot execute.
//
// The same holds for the `date` spellings GNU and BSD date do not share. The
// runner used `date -Is`: BSD date rejects it (`invalid argument 's' for
// -I`), the command substitution is then empty, and the log line simply loses
// its timestamp. A branch merged on 2026-10-08 brought a new `$(date -Is)`
// into the ported runner that way, and nothing on Linux could have refused it.
//
// What the scan names was measured, construct by construct, under
// /bin/bash 3.2.57 on 2026-10-09: each one is a syntax error, an "invalid
// option", a "bad substitution", an unbound-variable abort under `set -u`, or
// (the dangerous half) runs and does something else: `[ -v X ]` reads false,
// `{1..5..2}` stays literal, `{01..03}` loses its zeros, `declare -g` is
// refused and the script carries on, `$'é'` is six literal bytes.
//
// Its limits. It reads one line at a time and treats quoted text as code, so
// it refuses a construct that only appears inside a string (`echo "see
// declare -A"`, a `sed` program holding `|&`): fail-closed, reword the line.
// It cannot see an associative USE of an array declared elsewhere, a construct
// split across lines, or a non-portable option of any tool but `date`; the
// self-test under /bin/bash on a BSD host is what covers those, and it does
// not reach `--launch`.
// ---------------------------------------------------------------------------

const M0_RUNNER: &str = "scripts/m0/moat_both_profiles.sh";

/// Arrays the M0 runner expands without the `+` guard because they cannot be
/// empty where they are expanded: `ROWS` (`load_rows` dies 97 on a spec with
/// no rows before any use), `props` and `envs` (each initialised with at least
/// one element in `launch`). Bash 3.2 reports an EMPTY array as unbound under
/// `set -u`, so every other plain `"${a[@]}"` is written
/// `${a[@]+"${a[@]}"}`.
const M0_NEVER_EMPTY_ARRAYS: &[&str] = &["ROWS", "props", "envs"];

/// Every `shopt` option name bash 3.2.57 has (`/bin/bash -c shopt`). Any other
/// name is an "invalid shell option name" there, and the script carries on
/// without the option.
const BASH32_SHOPT_NAMES: &[&str] = &[
    "cdable_vars",
    "cdspell",
    "checkhash",
    "checkwinsize",
    "cmdhist",
    "compat31",
    "dotglob",
    "execfail",
    "expand_aliases",
    "extdebug",
    "extglob",
    "extquote",
    "failglob",
    "force_fignore",
    "gnu_errfmt",
    "histappend",
    "histreedit",
    "histverify",
    "hostcomplete",
    "huponexit",
    "interactive_comments",
    "lithist",
    "login_shell",
    "mailwarn",
    "no_empty_cmd_completion",
    "nocaseglob",
    "nocasematch",
    "nullglob",
    "progcomp",
    "promptvars",
    "restricted_shell",
    "shift_verbose",
    "sourcepath",
    "xpg_echo",
];

fn is_shell_name_byte(byte: u8) -> bool {
    byte == b'_' || byte.is_ascii_alphanumeric()
}

/// A code line's simple commands, each as its words: split where the shell
/// starts a new command, then on blanks. An option list then ends with its
/// own command (`shopt -s nullglob; shopt -u extglob` is two).
fn shell_commands(code_line: &str) -> Vec<Vec<&str>> {
    code_line
        .split([';', '(', ')', '|', '&', '`'])
        .map(|command| command.split_whitespace().collect::<Vec<_>>())
        .filter(|words| !words.is_empty())
        .collect()
}

/// Whether `words[index]` is the command word of its simple command: only
/// assignments (`IFS= read ...`) and reserved words (`if test ...`) before it.
fn is_command_word(words: &[&str], index: usize) -> bool {
    words[..index].iter().all(|word| {
        matches!(
            *word,
            "if" | "then" | "else" | "elif" | "while" | "until" | "do" | "!" | "{" | "time"
        ) || word
            .split_once('=')
            .is_some_and(|(name, _)| !name.is_empty() && name.bytes().all(is_shell_name_byte))
    })
}

/// The head of one `${...}` parameter expansion.
struct ParameterExpansion<'line> {
    /// Byte offset of its `$`.
    start: usize,
    /// `!` (indirection, keys) or `#` (length) directly after `${`.
    prefix: Option<u8>,
    /// The parameter: a name, a positional number, `@` or `*`.
    name: &'line str,
    /// The text between `[` and `]` when the name is subscripted.
    subscript: Option<&'line str>,
    /// Everything after the name and subscript, to the end of the line.
    rest: &'line str,
}

fn parameter_expansions(code_line: &str) -> Vec<ParameterExpansion<'_>> {
    let bytes = code_line.as_bytes();
    let mut found = Vec::new();
    let mut search = 0;
    while let Some(offset) = code_line[search..].find("${") {
        let start = search + offset;
        search = start + 2;
        let mut cursor = start + 2;
        let prefix = match bytes.get(cursor) {
            Some(&byte @ (b'!' | b'#'))
                if bytes.get(cursor + 1).is_some_and(|next| {
                    is_shell_name_byte(*next) || matches!(next, b'@' | b'*')
                }) =>
            {
                cursor += 1;
                Some(byte)
            }
            _ => None,
        };
        let name_start = cursor;
        if matches!(bytes.get(cursor), Some(b'@' | b'*')) {
            cursor += 1;
        } else {
            while bytes
                .get(cursor)
                .is_some_and(|byte| is_shell_name_byte(*byte))
            {
                cursor += 1;
            }
        }
        if cursor == name_start {
            continue;
        }
        let name = &code_line[name_start..cursor];
        let subscript = if bytes.get(cursor) == Some(&b'[') {
            let Some(close) = code_line[cursor..].find(']') else {
                continue;
            };
            let inner = &code_line[cursor + 1..cursor + close];
            cursor += close + 1;
            Some(inner)
        } else {
            None
        };
        found.push(ParameterExpansion {
            start,
            prefix,
            name,
            subscript,
            rest: &code_line[cursor..],
        });
    }
    found
}

/// Whether `text` opens with `-` and then a digit, blanks allowed around the
/// sign: a negative array subscript or substring length.
fn opens_with_negative_number(text: &str) -> bool {
    text.trim_start().strip_prefix('-').is_some_and(|after| {
        after
            .trim_start()
            .starts_with(|digit: char| digit.is_ascii_digit())
    })
}

/// The digits of a decimal number with its sign removed, or `None` when
/// `part` is not one.
fn unsigned_digits(part: &str) -> Option<&str> {
    let unsigned = part.strip_prefix('-').unwrap_or(part);
    (!unsigned.is_empty() && unsigned.bytes().all(|byte| byte.is_ascii_digit())).then_some(unsigned)
}

/// In `${v:offset:length}`, the text of `length`: what follows the second
/// colon at the expansion's own nesting level. `after_first_colon` starts
/// right after the first colon.
fn substring_length_text(after_first_colon: &str) -> Option<&str> {
    let (mut braces, mut parens) = (0usize, 0usize);
    for (index, character) in after_first_colon.char_indices() {
        match character {
            '{' => braces += 1,
            '}' if braces == 0 => return None,
            '}' => braces -= 1,
            '(' => parens += 1,
            ')' => parens = parens.saturating_sub(1),
            ':' if braces == 0 && parens == 0 => return Some(&after_first_colon[index + 1..]),
            _ => {}
        }
    }
    None
}

/// Constructs bash 3.2 cannot run or runs differently, each with why. Matched
/// on one code line with its comment removed. `never_empty_arrays` names the
/// arrays whose plain `"${a[@]}"` is allowed without the `+` guard.
// The messages quote shell text (`{name}>`), not Rust format arguments.
#[allow(clippy::literal_string_with_formatting_args)]
fn bash4_only_constructs(code_line: &str, never_empty_arrays: &[&str]) -> Vec<&'static str> {
    let mut found = Vec::new();
    let bytes = code_line.as_bytes();
    for words in shell_commands(code_line) {
        for (index, word) in words.iter().enumerate() {
            let flags = || {
                words[index + 1..]
                    .iter()
                    .take_while(|candidate| candidate.starts_with('-'))
            };
            match *word {
                "mapfile" | "readarray" => found.push("mapfile/readarray (bash 4.0)"),
                "coproc" => found.push("coproc (bash 4.0)"),
                "declare" | "local" | "typeset" | "readonly" => {
                    for flag in flags() {
                        if flag.contains('A') {
                            found.push("associative array (bash 4.0; 3.2 collapses it, D1)");
                        }
                        if *word == "readonly" {
                            continue;
                        }
                        if flag.contains('n') {
                            found.push("nameref (bash 4.3)");
                        }
                        if flag.contains('g') {
                            found.push(
                                "declare -g (bash 4.2; 3.2 refuses the option and carries on)",
                            );
                        }
                        if flag.contains('l') || flag.contains('u') {
                            found.push("declare -l/-u case attribute (bash 4.0)");
                        }
                    }
                }
                // `test` is also an argument of other tools (`targo test -v`),
                // so only as a command word
                "test" if is_command_word(&words, index) => {
                    let operand = words[index + 1..]
                        .iter()
                        .find(|candidate| **candidate != "!");
                    if operand == Some(&"-v") {
                        found.push("test -v (bash 4.2; 3.2 reads `[ -v x ]` as false)");
                    }
                }
                "read" => {
                    let mut arguments = words[index + 1..].iter();
                    while let Some(flag) = arguments.next() {
                        if !flag.starts_with('-') {
                            break;
                        }
                        if flag.contains('N') || flag.contains('i') {
                            found.push("read -N/-i (bash 4.0)");
                        }
                        // -t, -d, -n, -p, -u and -a take a value, the next word
                        if flag.ends_with('t') {
                            if arguments.next().is_some_and(|value| value.contains('.')) {
                                found.push("fractional read -t (bash 4.0)");
                            }
                        } else if flag.ends_with(['d', 'n', 'p', 'u', 'a']) {
                            arguments.next();
                        }
                    }
                }
                "wait" => {
                    if flags().any(|flag| flag.contains(['n', 'f', 'p'])) {
                        found.push("wait -n/-f/-p (bash 4.3 and later)");
                    }
                }
                "unset" => {
                    if flags().any(|flag| flag.contains('n')) {
                        found.push("unset -n (bash 4.3)");
                    }
                }
                "printf" => {
                    if code_line.contains("%(") {
                        found.push("printf %(fmt)T (bash 4.2; 3.2 stops at the format)");
                    }
                }
                // `shopt -o` names `set -o` options, a different list
                "shopt" if !flags().any(|flag| flag.contains('o')) => {
                    let unknown = words[index + 1..]
                        .iter()
                        .skip_while(|candidate| candidate.starts_with('-'))
                        .any(|name| !BASH32_SHOPT_NAMES.contains(name));
                    if unknown {
                        found.push("shopt option bash 3.2 lacks (globstar, lastpipe, ...)");
                    }
                }
                _ => {}
            }
        }
    }
    // `[ -v x ]`, `[[ a && -v x ]]`: `-v` as a unary operator anywhere inside
    // the brackets. Blank-separated words only, so the `-v` of a command
    // substitution (`[ "$(ulimit -v)" = unlimited ]`) is not one, and as the
    // right-hand side of a comparison it is a string.
    let blank_separated: Vec<&str> = code_line.split_whitespace().collect();
    for (index, word) in blank_separated.iter().enumerate() {
        // the bracket may open a command substitution: `x=$([ -v y ] && ...)`
        let opener = word.rsplit(['(', '`']).next().unwrap_or(word);
        if !matches!(opener, "[" | "[[") {
            continue;
        }
        for position in index + 1..blank_separated.len() {
            if blank_separated[position].starts_with(']') {
                break;
            }
            let compared = position > index + 1
                && matches!(
                    blank_separated[position - 1],
                    "=" | "==" | "!=" | "=~" | "<" | ">"
                );
            if blank_separated[position] == "-v" && !compared {
                found.push("test -v (bash 4.2; 3.2 reads `[ -v x ]` as false)");
            }
        }
    }
    for (needle, why) in [
        ("&>>", "&>> (bash 4.0)"),
        ("|&", "|& (bash 4.0)"),
        (";;&", "case continue ;;& (bash 4.0)"),
        ("globstar", "globstar (bash 4.0)"),
        ("lastpipe", "lastpipe (bash 4.2)"),
        ("inherit_errexit", "inherit_errexit (bash 4.4)"),
        ("BASHPID", "BASHPID (bash 4.0)"),
        ("BASH_XTRACEFD", "BASH_XTRACEFD (bash 4.1)"),
        ("BASH_ARGV0", "BASH_ARGV0 (bash 5.0)"),
        ("EPOCHSECONDS", "EPOCHSECONDS (bash 5.0)"),
        ("EPOCHREALTIME", "EPOCHREALTIME (bash 5.0)"),
        ("SRANDOM", "SRANDOM (bash 5.1)"),
        (
            "$'\\u",
            "$'\\uXXXX' (bash 4.2; 3.2 keeps the backslash and the u)",
        ),
        (
            "$'\\U",
            "$'\\UXXXXXXXX' (bash 4.2; 3.2 keeps the backslash and the U)",
        ),
    ] {
        if code_line.contains(needle) {
            found.push(why);
        }
    }
    // `;&`: case fall-through (`;;&` is named above)
    for (position, _) in code_line.match_indices(";&") {
        if position == 0 || bytes[position - 1] != b';' {
            found.push("case fall-through ;& (bash 4.0)");
        }
    }
    for expansion in parameter_expansions(code_line) {
        let rest = expansion.rest;
        // ${v^} ${v^^} ${v,} ${v,,}, with or without a pattern
        if rest.starts_with(['^', ',']) {
            found.push("case modification ${v,,}/${v^^} (bash 4.0)");
        }
        // ${v@Q} and the other one-letter transformations
        let mut after = rest.chars();
        if after.next() == Some('@')
            && after
                .next()
                .is_some_and(|operator| operator.is_ascii_alphabetic())
            && after.next() == Some('}')
        {
            found.push("${v@op} transformation (bash 4.4)");
        }
        if expansion.subscript.is_some_and(opens_with_negative_number) {
            found.push("negative array subscript (bash 4.3)");
        }
        if expansion.prefix.is_none() && rest.starts_with('}') {
            // a plain expansion: nothing between the parameter and the brace
            let before = &code_line[..expansion.start];
            let name = expansion.name;
            if matches!(expansion.subscript, Some("@" | "*")) {
                let guarded = before.ends_with(&format!("${{{name}[@]+\""))
                    || before.ends_with(&format!("${{{name}[*]+\""));
                if !guarded && !never_empty_arrays.contains(&name) {
                    found.push(
                        "unguarded \"${a[@]}\" (bash 3.2: an empty array is unbound under set -u; \
                         write ${a[@]+\"${a[@]}\"})",
                    );
                }
            }
            if expansion.subscript.is_none() && matches!(name, "@" | "*") {
                found.push(
                    "${@} or ${*} (bash 3.2: unbound under set -u with no argument; write \"$@\")",
                );
            }
        }
        // ${v:offset:-n}: a negative LENGTH. `${v:-word}` and its kin are the
        // default-value operators, and a negative OFFSET (`${v: -2}`) is 3.2.
        if let Some(after_colon) = rest.strip_prefix(':') {
            if !after_colon.starts_with(['-', '=', '?', '+'])
                && substring_length_text(after_colon).is_some_and(opens_with_negative_number)
            {
                found.push("negative substring length ${v:o:-n} (bash 4.2)");
            }
        }
    }
    // a negative subscript outside `${...}`: `a[-1]=x`, `$(( a[-1] ))`
    for (position, _) in code_line.match_indices('[') {
        if position > 0
            && is_shell_name_byte(bytes[position - 1])
            && opens_with_negative_number(&code_line[position + 1..])
        {
            found.push("negative array subscript (bash 4.3)");
        }
    }
    for (position, _) in code_line.match_indices('{') {
        // `{name}>file`, `{name}<file`: the shell picks the descriptor
        let mut end = position + 1;
        while bytes.get(end).is_some_and(|byte| is_shell_name_byte(*byte)) {
            end += 1;
        }
        let named = end > position + 1 && !bytes[position + 1].is_ascii_digit();
        let own_word = position == 0 || bytes[position - 1].is_ascii_whitespace();
        if named
            && own_word
            && bytes.get(end) == Some(&b'}')
            && matches!(bytes.get(end + 1), Some(b'>' | b'<'))
        {
            found.push("{name}> descriptor allocation (bash 4.1; 3.2 runs {name} as a command)");
        }
        // `{a..b..step}` and `{01..10}`
        let Some(close) = code_line[position..].find('}') else {
            continue;
        };
        let parts: Vec<&str> = code_line[position + 1..position + close]
            .split("..")
            .collect();
        let letter = |part: &str| part.len() == 1 && part.as_bytes()[0].is_ascii_alphabetic();
        let endpoint = |part: &str| unsigned_digits(part).is_some() || letter(part);
        if parts.len() == 3
            && endpoint(parts[0])
            && endpoint(parts[1])
            && unsigned_digits(parts[2]).is_some()
        {
            found.push("brace sequence with a step {a..b..c} (bash 4.0; 3.2 leaves it literal)");
        }
        let numbers: Vec<&str> = parts
            .iter()
            .filter_map(|part| unsigned_digits(part))
            .collect();
        if parts.len() == 2
            && numbers.len() == 2
            && numbers
                .iter()
                .any(|digits| digits.len() > 1 && digits.starts_with('0'))
        {
            found.push("zero-padded brace sequence {01..10} (bash 4.0; 3.2 drops the zeros)");
        }
    }
    found.sort_unstable();
    found.dedup();
    found
}

/// `date` spellings GNU and BSD date do not share. They agree on `-u` and on
/// `+FORMAT` with the common fields; the runner builds its timestamps in
/// `now_iso` from those.
fn non_portable_date_usage(code_line: &str) -> Vec<&'static str> {
    let words: Vec<&str> = code_line
        .split(|character: char| {
            character.is_whitespace()
                || matches!(character, ';' | '(' | ')' | '|' | '&' | '`' | '"' | '$')
        })
        .filter(|word| !word.is_empty())
        .collect();
    let mut found = Vec::new();
    for (index, word) in words.iter().enumerate() {
        if *word != "date" {
            continue;
        }
        for argument in &words[index + 1..] {
            if let Some(format) = argument.strip_prefix('+') {
                if format.contains("%N") {
                    found.push("date +%N (GNU nanoseconds; an older BSD date prints a literal N)");
                }
                if format.contains("%:") {
                    found.push("date +%:z (GNU only; BSD date prints `:z`)");
                }
                break;
            } else if argument.starts_with("-I") || argument.starts_with("--iso") {
                found.push("date -I... (GNU spelling; BSD date rejects `-Is`: call now_iso)");
            } else if argument.starts_with('-') && argument.len() > 1 {
                if *argument != "-u" {
                    found.push(
                        "date option other than -u (GNU and BSD date disagree on -d, -r, -v, -j, -f \
                         and every long option)",
                    );
                }
            } else {
                // an operand: this `date` word takes no further options
                break;
            }
        }
    }
    found.sort_unstable();
    found.dedup();
    found
}

/// The part of a shell line before its comment: `#` starts a comment only at
/// the start of a word and outside quotes (`${#a[@]}` and `$#` are code).
fn without_comment(line: &str) -> &str {
    let (mut single, mut double, mut escaped) = (false, false, false);
    let mut previous: Option<char> = None;
    for (index, character) in line.char_indices() {
        if escaped {
            escaped = false;
        } else if character == '\\' && !single {
            escaped = true;
        } else if character == '\'' && !double {
            single = !single;
        } else if character == '"' && !single {
            double = !double;
        } else if character == '#'
            && !single
            && !double
            && previous.is_none_or(|before| {
                before.is_whitespace() || matches!(before, ';' | '(' | '&' | '|')
            })
        {
            return &line[..index];
        }
        previous = Some(character);
    }
    line
}

// The snippets are shell text (`${v:-}`), not Rust format arguments.
#[allow(clippy::literal_string_with_formatting_args)]
#[test]
fn m0_bash4_scanner_reports_each_construct_it_names() {
    // Positive control: a scanner that finds nothing in the runner proves
    // nothing until it is shown to find each construct. Every snippet here was
    // run under /bin/bash 3.2.57 on 2026-10-09 and failed or misbehaved there.
    for (snippet, want) in [
        // The two lines the runner carried on main until 2026-10-04, verbatim.
        (
            "  mapfile -t ROWS < <(python3 \"$JUDGE\" rows --spec \"$SPEC\" </dev/null)",
            "mapfile",
        ),
        ("  declare -A done_plan=()", "associative"),
        ("mapfile -t ROWS < <(cmd)", "mapfile"),
        ("readarray -t ROWS", "mapfile"),
        ("  declare -A seen=()", "associative"),
        ("local -rA seen", "associative"),
        ("typeset -A m", "associative"),
        ("readonly -A seen=()", "associative"),
        ("local -n ref=x", "nameref"),
        ("  declare -g PIN_SHA", "declare -g"),
        ("  local -l lower=$C", "case attribute"),
        ("  local -u upper=$C", "case attribute"),
        ("echo \"${v,,}\"", "case modification"),
        ("echo \"${v^^}\"", "case modification"),
        ("x=${C^^[a-z]}", "case modification"),
        ("y=${list[2],}", "case modification"),
        ("echo \"${a[-1]}\"", "negative array subscript"),
        ("echo \"${ROWS[ -1 ]}\"", "negative array subscript"),
        ("arr[-1]=x", "negative array subscript"),
        ("echo \"$(( a[-1] ))\"", "negative array subscript"),
        ("  key=${key::-1}", "negative substring length"),
        ("  short=${PIN_SHA:0:-48}", "negative substring length"),
        ("  short=${PIN_SHA:0: -48}", "negative substring length"),
        ("cmd &>> log", "&>>"),
        ("cmd |& tee log", "|&"),
        ("coproc worker { cat; }", "coproc"),
        ("wait -n", "wait -n"),
        ("wait -f $hb", "wait -n"),
        ("unset -n ref", "unset -n"),
        ("echo \"${v@Q}\"", "transformation"),
        ("echo \"${a[@]@Q}\"", "transformation"),
        ("  [[ -v M0_STOP_FILE ]] && echo stop", "test -v"),
        // This one runs under bash 3.2 and reads FALSE: the stop file would
        // never stop the runner there.
        ("  if [ -v M0_STOP_FILE ]; then echo stop; fi", "test -v"),
        ("  if [ ! -v M0_STOP_FILE ]; then :; fi", "test -v"),
        ("test -v M0_OUT", "test -v"),
        ("if ! test -v M0_OUT; then :; fi", "test -v"),
        ("[[ -n $a && -v M0_OUT ]] || exit 2", "test -v"),
        ("ok=$([ -v M0_OUT ] && echo y)", "test -v"),
        (
            "  printf -v stamp '%(%Y-%m-%dT%H:%M:%S%z)T' -1",
            "printf %(fmt)T",
        ),
        ("  exec {lockfd}>\"$OUT/.lock\"", "descriptor allocation"),
        ("exec {fd}<&-", "descriptor allocation"),
        (
            "  case $R in 75) REFUSEDN=$((REFUSEDN+1)) ;& 0) : ;; esac",
            "case fall-through",
        ),
        ("case a in a) : ;;& b) : ;; esac", "case continue"),
        ("  for attempt in {1..3..1}; do :; done", "with a step"),
        ("  for i in {a..z..2}; do :; done", "with a step"),
        ("  for i in {01..03}; do :; done", "zero-padded"),
        ("  shopt -s lastpipe", "lastpipe"),
        ("shopt -s nullglob globstar", "shopt option"),
        ("shopt -s direxpand", "shopt option"),
        ("  read -r -t 0.5 line", "fractional read -t"),
        ("  read -N 4 magic", "read -N"),
        ("read -e -i default answer", "read -N"),
        ("echo $'\\u00e9'", "uXXXX"),
        ("echo \"pid $BASHPID at $EPOCHSECONDS\"", "BASHPID"),
        ("x=$SRANDOM", "SRANDOM"),
        // The line main ran for every measured process until 2026-10-04, and
        // two more arrays that can be empty.
        (
            "  \"$NY/scripts/ny_box.sh\" measure -- env \"${UNSET[@]}\" \\",
            "unguarded",
        ),
        ("  echo ${UNSET[*]}", "unguarded"),
        ("  for v in \"${EXTRA[@]}\"; do :; done", "unguarded"),
        ("  run_it \"${@}\"", "${@} or ${*}"),
    ] {
        let found = bash4_only_constructs(without_comment(snippet), M0_NEVER_EMPTY_ARRAYS);
        assert!(
            found.iter().any(|why| why.contains(want)),
            "scanner missed {want} in {snippet:?}: {found:?}"
        );
    }
    // The `date` spellings GNU and BSD date do not share, with the line a
    // merge brought in on 2026-10-08 and the one main used for its aside
    // names, both verbatim; and what both dates accept is not flagged.
    for (snippet, want) in [
        (
            "          summary; echo \"M0MOAT_DONE rc=3 WORST=3 STOPPED_AT=$T/$P/r$k reason=stop-file $(date -Is)\"; exit 3",
            "date -I",
        ),
        ("stamp=`date -Iseconds`", "date -I"),
        ("date --iso-8601=seconds", "date -I"),
        ("x=$(date -u -Is)", "date -I"),
        (
            "          if [ -e \"$f\" ]; then mv -- \"$f\" \"$f.stale.$(date +%s%N)\" || { summary; die 98 \"stale-move:$f\"; }; fi",
            "date +%N",
        ),
        ("  d=$(date -d \"@$SECS\" +%F)", "other than -u"),
        ("  stamp=$(date --rfc-3339=seconds)", "other than -u"),
        ("  d=$(date -r \"$f\" +%s)", "other than -u"),
        ("  d=$(date -v-1d +%F)", "other than -u"),
        ("  z=$(date +%:z)", "date +%:z"),
    ] {
        let found = non_portable_date_usage(without_comment(snippet));
        assert!(
            found.len() == 1 && found[0].contains(want),
            "scanner missed {want} in {snippet:?}: {found:?}"
        );
    }
    for snippet in [
        "t=$(date +%Y-%m-%dT%H:%M:%S%z); head=${t%??}",
        "stamp=$(date +%Y%m%d-%H%M%S)",
        "aside=$f.stale.$(date +%s); n=0",
        "now=$(date -u +%s)",
        "now_iso() { # GNU `date -Is` output, built from fields BSD date also has",
        "echo \"update -Is is not a date call\"; validate -I x",
    ] {
        let found = non_portable_date_usage(without_comment(snippet));
        assert!(found.is_empty(), "false positive in {snippet:?}: {found:?}");
    }
    // And it does not flag what bash 3.2 runs (each also run there): comments,
    // `${#a[@]}`, `${!a[@]}`, `$#`, `"$@"`, the guarded expansion, the arrays
    // that cannot be empty, array slices and edits, `+=`, `${v:0:16}`, a
    // negative OFFSET, an arithmetic length, `${!v}`, `$'\t'`, brace lists and
    // plain sequences, whole-second `read -t`, the options 3.2 has.
    for snippet in [
        "# mapfile and declare -A are named in comments only",
        "n=${#ROWS[@]}; [ $# -gt 0 ] && ROWS+=(\"$x\")",
        "env ${UNSET[@]+\"${UNSET[@]}\"} cmd",
        "echo ${UNSET[*]+\"${UNSET[*]}\"}",
        "for line in \"${ROWS[@]}\"; do :; done",
        "systemd-run --user \"${props[@]}\" \"${envs[@]}\" /bin/bash x",
        "echo \"${PIN_SHA:0:16} ${!v:-} $'\\t' ${M0_PIN##*/}\" # declare -A",
        "[[ \"${M0_PIN_SHA16:-}\" =~ ^[0-9a-f]{16}$ ]] || die 2 x",
        "mkdir -p \"$OUT\"/{raw,res,}",
        "cp \"$f\"{,.bak}",
        "x=${v/,/}",
        "echo \"${v:0:${#v}-1} ${v: -2} ${v:(-2)} ${v:-word} ${v:+word}\"",
        "for f in \"${files[@]/#/prefix-}\"; do :; done",
        "echo \"${EXTRA[@]: -1} ${EXTRA[@]:1} ${!EXTRA[@]} ${EXTRA[@]-none}\"",
        "run_it \"$@\"; echo \"$*\" \"${1:-}\" \"${10}\"",
        "for i in {1..3}; do :; done; for c in {a..c}; do :; done",
        "read -r -t 1 line; read -r -d '' blob; read -n 1 key; IFS= read -r row",
        "shopt -s nullglob extglob; shopt -u nocasematch; shopt -o -q pipefail",
        "local -a UNSET=(); local -r x=1; declare -i n=0; declare -f now_iso; readonly -a FIXED",
        "[ -n \"${v:-}\" ] && [[ -z $w ]] && test -f \"$f\" && [ ! -e \"$g\" ]",
        "[ \"$flag\" = -v ] && grep -v x \"$f\"; command -v timeout; targo test -v",
        "lv=$(ulimit -v); [ \"$(ulimit -v)\" = unlimited ] || die 86 x",
        "if [ -n \"$x\" ]; then grep -v y \"$f\"; fi",
        "wait \"$hb\"; unset -f helper; unset -v name; printf -v out '%s' \"$x\"",
        "exec 3>\"$OUT/.lock\"; echo \"${x}>y\"",
        "awk -F'\\t' '{s+=($8+$9)*$5} END{printf \"%.1f\", s/3600}' <(printf '%s\\n' \"${ROWS[@]}\")",
    ] {
        let found = bash4_only_constructs(without_comment(snippet), M0_NEVER_EMPTY_ARRAYS);
        assert!(found.is_empty(), "false positive in {snippet:?}: {found:?}");
    }
    // Quoted text is scanned as code (see the limits above): these two valid
    // bash 3.2 lines are refused, and that is the documented behaviour.
    for (snippet, want) in [
        ("echo \"see declare -A in the docs\"", "associative"),
        ("sed -e 's/a|&/x/' file", "|&"),
    ] {
        let found = bash4_only_constructs(without_comment(snippet), M0_NEVER_EMPTY_ARRAYS);
        assert!(
            found.iter().any(|why| why.contains(want)),
            "the scan stopped treating quoted text as code in {snippet:?}: {found:?}"
        );
    }
    // The allowlist is by name: an array not on it is refused unguarded.
    assert!(
        bash4_only_constructs("for line in \"${ROWS[@]}\"; do :; done", &[])
            .iter()
            .any(|why| why.contains("unguarded")),
        "an array is only exempt while it is on the never-empty list"
    );
}

#[test]
fn m0_runner_has_no_bash4_only_construct() {
    let path = workspace_root().join(M0_RUNNER);
    let source = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("{} unreadable: {error}", path.display()));
    let code_lines = source
        .lines()
        .filter(|line| !without_comment(line).trim().is_empty())
        .count();
    assert!(
        code_lines > 200,
        "{M0_RUNNER} has {code_lines} code lines; the scan would prove nothing"
    );
    // The exemption list must not outlive its arrays: a name no line expands
    // any more is a stale exemption waiting for a new array of that name.
    for name in M0_NEVER_EMPTY_ARRAYS {
        assert!(
            source.contains(&format!("\"${{{name}[@]}}\"")),
            "{M0_RUNNER} no longer expands \"${{{name}[@]}}\"; drop it from M0_NEVER_EMPTY_ARRAYS"
        );
    }
    let violations: Vec<String> = source
        .lines()
        .enumerate()
        .flat_map(|(index, line)| {
            let code = without_comment(line);
            bash4_only_constructs(code, M0_NEVER_EMPTY_ARRAYS)
                .into_iter()
                .chain(non_portable_date_usage(code))
                .map(move |why| format!("{M0_RUNNER}:{}: {why}: {}", index + 1, line.trim()))
        })
        .collect();
    assert!(
        violations.is_empty(),
        "the M0 runner must run under bash 3.2 with BSD date (macOS /bin/bash):\n{}",
        violations.join("\n")
    );
}
