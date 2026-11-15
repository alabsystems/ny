// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The Python distribution's version has ONE source: `[workspace.package]
//! version` in the root Cargo.toml. Nothing in the Python package repeats it.
//!
//! WHY THIS IS A CARGO TEST. ac4cf1483 ("prepare 0.5.0 consolidated source
//! snapshot") moved the workspace to 0.5.0 the way every release bump does —
//! Cargo.toml and Cargo.lock only — and left three hand-written copies at
//! 0.4.0: `crates/ny-python/pyproject.toml`, `ny_pytest.__version__`, and the
//! binding suite's `ny.__version__ == "0.4.0"`. The only check that compared
//! any of them was a pytest in the ny_pytest plugin suite, i.e. gate lane [g]:
//! it is not in the default Cargo lane that defines this repository's
//! correctness contract, and the bump never ran it, so the drift sat on main
//! for a month (and the wheel was stamped 0.4.0 around a module reporting
//! 0.5.0).
//!
//! The fix removes the copies instead of guarding them, because a guard only
//! helps a bump that runs it: the wheel takes its version from Cargo (`dynamic
//! = ["version"]`, which maturin stamps from the ny-python crate, itself
//! `version.workspace = true`); the pure-Python plugin derives it
//! (`ny_pytest/_version.py`: the Cargo manifests in a checkout, the
//! distribution metadata in an installed wheel); the binding suite reads the
//! root Cargo.toml. This test is hermetic (it reads tracked files, no
//! interpreter) and pins that structure in the default lane: a static
//! pyproject version, a plugin literal, or any Python line comparing a
//! `__version__` with a version literal fails here, naming the line.

use std::fs;
use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read(relative: &str) -> String {
    let path = workspace_root().join(relative);
    fs::read_to_string(&path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()))
}

/// The comment-stripped `key = value` lines of one named TOML table.
fn table_entries(text: &str, table: &str) -> Vec<(String, String)> {
    let mut active = String::new();
    let mut out = Vec::new();
    for raw in text.lines() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if let Some(name) = line
            .strip_prefix('[')
            .and_then(|rest| rest.strip_suffix(']'))
        {
            active = name.trim().to_owned();
            continue;
        }
        if active != table {
            continue;
        }
        if let Some((key, value)) = line.split_once('=') {
            out.push((key.trim().to_owned(), value.trim().to_owned()));
        }
    }
    out
}

fn table_value(text: &str, table: &str, key: &str) -> Option<String> {
    table_entries(text, table)
        .into_iter()
        .find(|(name, _)| name == key)
        .map(|(_, value)| value)
}

fn workspace_version() -> String {
    let manifest = read("Cargo.toml");
    let version = table_value(&manifest, "workspace.package", "version")
        .expect("root Cargo.toml must declare [workspace.package] version");
    let version = version.trim_matches('"').to_owned();
    assert!(
        !version.is_empty() && version.split('.').all(|part| part.parse::<u64>().is_ok()),
        "[workspace.package] version must be MAJOR.MINOR.PATCH, got {version:?}"
    );
    version
}

#[test]
fn ny_python_crate_inherits_the_workspace_version() {
    let manifest = read("crates/ny-python/Cargo.toml");
    assert_eq!(
        table_value(&manifest, "package", "version").as_deref(),
        None,
        "crates/ny-python/Cargo.toml must not pin its own `version`"
    );
    assert_eq!(
        table_value(&manifest, "package", "version.workspace").as_deref(),
        Some("true"),
        "crates/ny-python must inherit `version.workspace = true`: maturin stamps the \
         wheel from this crate's version"
    );
}

#[test]
fn wheel_version_is_derived_from_cargo_not_pinned() {
    let pyproject = read("crates/ny-python/pyproject.toml");
    assert_eq!(
        table_value(&pyproject, "project", "version"),
        None,
        "crates/ny-python/pyproject.toml must not pin a static [project] version — it went \
         stale at 0.4.0 when the workspace moved to {} (ac4cf1483). List `version` in \
         [project].dynamic instead.",
        workspace_version()
    );
    let dynamic = table_value(&pyproject, "project", "dynamic")
        .expect("crates/ny-python/pyproject.toml must declare [project].dynamic");
    assert!(
        dynamic.contains("\"version\""),
        "[project].dynamic must contain \"version\" so maturin takes it from Cargo, got {dynamic}"
    );
}

#[test]
fn pytest_plugin_derives_its_version_and_carries_no_literal() {
    let init = read("crates/ny-python/ny_pytest/__init__.py");
    let assignments: Vec<&str> = init
        .lines()
        .filter(|line| line.starts_with("__version__ ="))
        .collect();
    assert_eq!(
        assignments,
        ["__version__ = resolve_version()"],
        "ny_pytest/__init__.py must compute __version__ (ny_pytest/_version.py) and never \
         spell it: a literal here is a copy the next Cargo-only release bump leaves behind, \
         as 0.4.0 was when the workspace moved to {}",
        workspace_version()
    );
    let resolver = read("crates/ny-python/ny_pytest/_version.py");
    for needle in [
        "def resolve_version(",
        "\"workspace.package\", \"version\"",
        "metadata.version(DISTRIBUTION)",
    ] {
        assert!(
            resolver.contains(needle),
            "ny_pytest/_version.py no longer reads the version from Cargo or the stamped \
             distribution metadata (missing `{needle}`)"
        );
    }
}

/// Does `line` hold a string literal that is exactly `MAJOR.MINOR.PATCH`
/// (`"0.4.0"` or `'0.4.0'`)?
fn has_quoted_version_literal(line: &str) -> bool {
    for quote in ['"', '\''] {
        let mut rest = line;
        while let Some(start) = rest.find(quote) {
            let after = &rest[start + 1..];
            let Some(end) = after.find(quote) else {
                break;
            };
            let literal = &after[..end];
            let parts: Vec<&str> = literal.split('.').collect();
            if parts.len() == 3
                && parts
                    .iter()
                    .all(|part| !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit()))
            {
                return true;
            }
            rest = &after[end + 1..];
        }
    }
    false
}

fn python_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries =
        fs::read_dir(dir).unwrap_or_else(|error| panic!("read {}: {error}", dir.display()));
    for entry in entries {
        let path = entry.expect("directory entry").path();
        if path.is_dir() {
            if path.file_name().is_some_and(|name| name != "__pycache__") {
                python_sources(&path, out);
            }
        } else if path.extension().is_some_and(|extension| extension == "py") {
            out.push(path);
        }
    }
}

/// No Python source of the package holds a hand-written copy of the version.
/// Every `__version__` comparison takes its expected value from Cargo (the
/// binding suite and the plugin suite both read the root Cargo.toml), so a
/// release bump has no Python line to move. The scan covers the package's two
/// tracked source trees, not the crate directory, so a developer's virtualenv
/// or build output beside them is never read.
#[test]
fn no_python_source_compares_a_version_literal() {
    let package = workspace_root().join("crates/ny-python");
    let mut sources = Vec::new();
    for tree in ["ny_pytest", "tests"] {
        python_sources(&package.join(tree), &mut sources);
    }
    sources.sort();
    for expected in ["ny_pytest/__init__.py", "tests/test_ny.py"] {
        assert!(
            sources.iter().any(|path| path.ends_with(expected)),
            "the scan must reach {expected}, found {sources:?}"
        );
    }
    let mut offenders = Vec::new();
    for path in &sources {
        let text = fs::read_to_string(path)
            .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
        for (index, line) in text.lines().enumerate() {
            let code = line.split('#').next().unwrap_or("");
            if code.contains("__version__") && has_quoted_version_literal(code) {
                offenders.push(format!("{}:{}: {}", path.display(), index + 1, line.trim()));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "a Python source pins a version literal against `__version__`; read the expected value \
         from the root Cargo.toml instead:\n{}",
        offenders.join("\n")
    );
}

#[test]
fn quoted_version_literal_detector_matches_only_exact_versions() {
    assert!(has_quoted_version_literal(
        r#"assert ny.__version__ == "0.4.0""#
    ));
    assert!(has_quoted_version_literal(
        "assert '10.20.0' == ny.__version__"
    ));
    assert!(!has_quoted_version_literal(
        "assert ny.__version__ == _workspace_version()"
    ));
    assert!(!has_quoted_version_literal(
        r#"x = "went stale at 0.4.0 when the workspace moved""#
    ));
    assert!(!has_quoted_version_literal(r#"key.strip() == "version""#));
    assert!(!has_quoted_version_literal(r#"y = "1.2" + "3.4.x""#));
}
