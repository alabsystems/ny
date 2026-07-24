// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Gates for the [`load_vnnlib`] parse memo (#vnnlib-parse-once): a repeated
//! load of an UNCHANGED file must be value-identical to a fresh parse, and a
//! CHANGED file must never be served stale.

use super::load_vnnlib;
use tempfile::tempdir;

const PROP_A: &str = r#"
(declare-const X_0 Real)
(declare-const X_1 Real)
(declare-const Y_0 Real)
(assert (<= X_0 0.5))
(assert (>= X_0 -0.5))
(assert (<= X_1 0.25))
(assert (>= X_1 -0.25))
(assert (or (and (<= X_0 0.1) (<= Y_0 0.0)) (and (>= X_0 0.2) (>= Y_0 1.0))))
"#;

// Different LENGTH from PROP_A so the memo key (size + mtime) always changes
// even on filesystems with coarse mtime granularity.
const PROP_B: &str = r#"
(declare-const X_0 Real)
(declare-const Y_0 Real)
(declare-const Y_1 Real)
(assert (<= X_0 1.5))
(assert (>= X_0 -1.5))
(assert (<= Y_1 3.0))
"#;

/// Repeat loads of the same unchanged file return a spec value-identical to
/// the first parse (compared via Debug formatting — full structural field
/// coverage without requiring PartialEq on the spec).
#[ntest::timeout(10000)]
#[test]
fn memo_repeat_load_is_value_identical() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("prop.vnnlib");
    std::fs::write(&path, PROP_A).unwrap();

    let first = load_vnnlib(&path).unwrap();
    let second = load_vnnlib(&path).unwrap();
    assert_eq!(
        format!("{first:?}"),
        format!("{second:?}"),
        "memoized reload diverged from the first parse"
    );
    assert_eq!(first.num_inputs, 2);
    assert_eq!(first.output_constraint_clauses.len(), 2);
}

/// Rewriting the file invalidates the memo: the next load reflects the new
/// content, never the cached spec.
#[ntest::timeout(10000)]
#[test]
fn memo_invalidated_when_file_changes() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("prop.vnnlib");

    std::fs::write(&path, PROP_A).unwrap();
    let a = load_vnnlib(&path).unwrap();
    assert_eq!(a.num_inputs, 2);
    assert_eq!(a.num_outputs, 1);

    std::fs::write(&path, PROP_B).unwrap();
    let b = load_vnnlib(&path).unwrap();
    assert_eq!(b.num_inputs, 1, "stale memoized spec served after rewrite");
    assert_eq!(b.num_outputs, 2, "stale memoized spec served after rewrite");
}
