// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Verdict-neutral profiling for captured phase-split MILP families.
//!
//! NY's production phase-split path clones one root [`MilpProblem`] for every
//! assignment of up to four selected binaries. This module validates, bit for
//! bit, that a set of captured children really is that complete fixed-prefix
//! partition. It then reports the exact duplicated serialized-IR payload.
//!
//! This is an offline research instrument. It never constructs a solver
//! session and cannot affect a verification verdict. Its purpose is to give a
//! future one-tree/shared-frontier AY implementation a sealed, model-specific
//! acceptance fixture instead of inferring the current split topology from
//! logs.

use std::collections::BTreeMap;

use serde::Serialize;
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::dump::to_milp_text;
use crate::ir::{ColSpec, MilpProblem, RowSpec};

/// Stable schema emitted by [`profile_phase_split_family`].
pub const SHARED_TREE_PROFILE_SCHEMA: &str = "ny_mip_shared_tree_profile_v1";

/// Shape statistics for one captured root problem.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProblemStats {
    pub columns: usize,
    pub continuous_columns: usize,
    pub integer_columns: usize,
    pub live_binary_columns: usize,
    pub fixed_integer_columns: usize,
    pub rows: usize,
    pub nonzeros: usize,
    pub margin_row: Option<usize>,
}

/// One fixed-prefix child, canonicalized by assignment rather than filename.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ChildProfile {
    /// Integer whose bit `i` is the fixed value for `split_columns[i]`.
    pub assignment: usize,
    pub fixed_values: Vec<u8>,
    /// SHA-256 of [`crate::dump::to_milp_text`] for this parsed child.
    pub canonical_milp_sha256: String,
    pub canonical_serialized_bytes: usize,
}

/// Exact, deterministic description of one complete phase-split family.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SharedTreeProfile {
    pub schema: &'static str,
    pub root: ProblemStats,
    pub root_canonical_milp_sha256: String,
    pub family_sha256: String,
    pub split_columns: Vec<usize>,
    pub split_depth: usize,
    pub isolated_sessions: usize,
    pub complete_binary_partition: bool,
    pub invariant_rows_and_non_split_columns: bool,
    pub root_canonical_serialized_bytes: usize,
    pub children_canonical_serialized_bytes: usize,
    /// Serialized-IR proxy only, not a claim about AY's resident heap.
    ///
    /// This is the exact sum of canonical child dump bytes minus one root
    /// dump. A shared-tree engine may still need per-worker LP state.
    pub serialized_ir_bytes_beyond_one_root: usize,
    pub children: Vec<ChildProfile>,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SharedTreeProfileError {
    #[error("phase-split profile requires at least one child")]
    NoChildren,
    #[error("child {child} has {actual} columns; root has {expected}")]
    ColumnCount {
        child: usize,
        expected: usize,
        actual: usize,
    },
    #[error("child {child} has {actual} rows; root has {expected}")]
    RowCount {
        child: usize,
        expected: usize,
        actual: usize,
    },
    #[error("child {child} changed row {row}")]
    RowDrift { child: usize, row: usize },
    #[error("child {child} changed margin marker from {root:?} to {observed:?}")]
    MarginDrift {
        child: usize,
        root: Option<usize>,
        observed: Option<usize>,
    },
    #[error("child {child} changed integrality/objective metadata for column {column}")]
    ColumnMetadataDrift { child: usize, column: usize },
    #[error("child {child} changed non-split column {column}")]
    NonSplitColumnDrift { child: usize, column: usize },
    #[error("changed root column {column} is not a live [0,1] integer binary")]
    ChangedColumnNotLiveBinary { column: usize },
    #[error("child {child} did not pin split column {column} to exactly 0 or 1")]
    SplitColumnNotPinned { child: usize, column: usize },
    #[error("the first child changes no root binary bounds")]
    NoSplitColumns,
    #[error("split depth {depth} cannot be represented by this profiler")]
    SplitDepthOverflow { depth: usize },
    #[error(
        "incomplete fixed-prefix family: split depth {depth} requires {expected} children, got {actual}"
    )]
    IncompletePartition {
        depth: usize,
        expected: usize,
        actual: usize,
    },
    #[error("duplicate fixed-prefix assignment {assignment}")]
    DuplicateAssignment { assignment: usize },
}

/// Validate and profile a root plus all fixed-prefix child problems.
///
/// Acceptance is deliberately strict:
///
/// - every row, coefficient bit pattern, margin marker, objective coefficient,
///   and integrality flag must be identical;
/// - every child must pin the same set of root `[0,1]` integer columns;
/// - each pin must be exactly zero or one; and
/// - the children must enumerate every assignment exactly once.
///
/// Therefore an accepted family is exactly the partition NY intends its
/// isolated phase-split sessions to cover. Any drift is an error, never a
/// best-effort profile.
pub fn profile_phase_split_family(
    root: &MilpProblem,
    children: &[MilpProblem],
) -> Result<SharedTreeProfile, SharedTreeProfileError> {
    let Some(first) = children.first() else {
        return Err(SharedTreeProfileError::NoChildren);
    };
    validate_shape_and_rows(root, first, 0)?;

    let mut split_columns = Vec::new();
    for (column, (root_col, child_col)) in root.cols().iter().zip(first.cols()).enumerate() {
        validate_column_metadata(root_col, child_col, 0, column)?;
        if !same_bounds(root_col, child_col) {
            validate_live_root_binary(root_col, column)?;
            validate_pinned_child(child_col, 0, column)?;
            split_columns.push(column);
        }
    }
    if split_columns.is_empty() {
        return Err(SharedTreeProfileError::NoSplitColumns);
    }

    let depth = split_columns.len();
    let shift = u32::try_from(depth)
        .ok()
        .and_then(|depth| 1usize.checked_shl(depth))
        .ok_or(SharedTreeProfileError::SplitDepthOverflow { depth })?;
    if children.len() != shift {
        return Err(SharedTreeProfileError::IncompletePartition {
            depth,
            expected: shift,
            actual: children.len(),
        });
    }

    let split_set: std::collections::BTreeSet<usize> = split_columns.iter().copied().collect();
    let mut by_assignment = BTreeMap::<usize, ChildProfile>::new();
    for (child_index, child) in children.iter().enumerate() {
        validate_shape_and_rows(root, child, child_index)?;
        let mut assignment = 0usize;
        let mut fixed_values = Vec::with_capacity(depth);
        let mut split_position = 0usize;
        for (column, (root_col, child_col)) in root.cols().iter().zip(child.cols()).enumerate() {
            validate_column_metadata(root_col, child_col, child_index, column)?;
            if split_set.contains(&column) {
                validate_live_root_binary(root_col, column)?;
                let value = validate_pinned_child(child_col, child_index, column)?;
                if value == 1 {
                    assignment |= 1usize << split_position;
                }
                fixed_values.push(value);
                split_position += 1;
            } else if !same_bounds(root_col, child_col) {
                return Err(SharedTreeProfileError::NonSplitColumnDrift {
                    child: child_index,
                    column,
                });
            }
        }

        let canonical = to_milp_text(child);
        let profile = ChildProfile {
            assignment,
            fixed_values,
            canonical_milp_sha256: sha256_hex(canonical.as_bytes()),
            canonical_serialized_bytes: canonical.len(),
        };
        if by_assignment.insert(assignment, profile).is_some() {
            return Err(SharedTreeProfileError::DuplicateAssignment { assignment });
        }
    }

    // Every assignment lies in [0, 2^depth), so uniqueness plus exact
    // cardinality proves that the hypercube is complete.
    debug_assert_eq!(by_assignment.len(), shift);
    let children: Vec<ChildProfile> = by_assignment.into_values().collect();
    let root_canonical = to_milp_text(root);
    let root_sha = sha256_hex(root_canonical.as_bytes());
    let children_bytes = children
        .iter()
        .map(|child| child.canonical_serialized_bytes)
        .sum();
    let family_sha256 = family_sha256(&root_sha, &split_columns, &children);

    Ok(SharedTreeProfile {
        schema: SHARED_TREE_PROFILE_SCHEMA,
        root: problem_stats(root),
        root_canonical_milp_sha256: root_sha,
        family_sha256,
        split_columns,
        split_depth: depth,
        isolated_sessions: shift,
        complete_binary_partition: true,
        invariant_rows_and_non_split_columns: true,
        root_canonical_serialized_bytes: root_canonical.len(),
        children_canonical_serialized_bytes: children_bytes,
        serialized_ir_bytes_beyond_one_root: children_bytes.saturating_sub(root_canonical.len()),
        children,
    })
}

fn validate_shape_and_rows(
    root: &MilpProblem,
    child: &MilpProblem,
    child_index: usize,
) -> Result<(), SharedTreeProfileError> {
    if child.num_cols() != root.num_cols() {
        return Err(SharedTreeProfileError::ColumnCount {
            child: child_index,
            expected: root.num_cols(),
            actual: child.num_cols(),
        });
    }
    if child.num_rows() != root.num_rows() {
        return Err(SharedTreeProfileError::RowCount {
            child: child_index,
            expected: root.num_rows(),
            actual: child.num_rows(),
        });
    }
    for (row, (root_row, child_row)) in root.rows().iter().zip(child.rows()).enumerate() {
        if !same_row(root_row, child_row) {
            return Err(SharedTreeProfileError::RowDrift {
                child: child_index,
                row,
            });
        }
    }
    let root_margin = root.margin_row().map(|row| row.0);
    let child_margin = child.margin_row().map(|row| row.0);
    if root_margin != child_margin {
        return Err(SharedTreeProfileError::MarginDrift {
            child: child_index,
            root: root_margin,
            observed: child_margin,
        });
    }
    Ok(())
}

fn validate_column_metadata(
    root: &ColSpec,
    child: &ColSpec,
    child_index: usize,
    column: usize,
) -> Result<(), SharedTreeProfileError> {
    if root.integer != child.integer || root.obj.to_bits() != child.obj.to_bits() {
        return Err(SharedTreeProfileError::ColumnMetadataDrift {
            child: child_index,
            column,
        });
    }
    Ok(())
}

fn validate_live_root_binary(root: &ColSpec, column: usize) -> Result<(), SharedTreeProfileError> {
    if !root.integer
        || root.lb.to_bits() != 0.0f64.to_bits()
        || root.ub.to_bits() != 1.0f64.to_bits()
    {
        return Err(SharedTreeProfileError::ChangedColumnNotLiveBinary { column });
    }
    Ok(())
}

fn validate_pinned_child(
    child: &ColSpec,
    child_index: usize,
    column: usize,
) -> Result<u8, SharedTreeProfileError> {
    let value = if child.lb.to_bits() == 0.0f64.to_bits() && child.ub.to_bits() == 0.0f64.to_bits()
    {
        0
    } else if child.lb.to_bits() == 1.0f64.to_bits() && child.ub.to_bits() == 1.0f64.to_bits() {
        1
    } else {
        return Err(SharedTreeProfileError::SplitColumnNotPinned {
            child: child_index,
            column,
        });
    };
    Ok(value)
}

fn same_bounds(left: &ColSpec, right: &ColSpec) -> bool {
    left.lb.to_bits() == right.lb.to_bits() && left.ub.to_bits() == right.ub.to_bits()
}

fn same_row(left: &RowSpec, right: &RowSpec) -> bool {
    left.lb.to_bits() == right.lb.to_bits()
        && left.ub.to_bits() == right.ub.to_bits()
        && left.coeffs.len() == right.coeffs.len()
        && left.coeffs.iter().zip(&right.coeffs).all(
            |(&(left_col, left_coeff), &(right_col, right_coeff))| {
                left_col == right_col && left_coeff.to_bits() == right_coeff.to_bits()
            },
        )
}

fn problem_stats(problem: &MilpProblem) -> ProblemStats {
    let integer_columns = problem.cols().iter().filter(|col| col.integer).count();
    let live_binary_columns = problem
        .cols()
        .iter()
        .filter(|col| {
            col.integer
                && col.lb.to_bits() == 0.0f64.to_bits()
                && col.ub.to_bits() == 1.0f64.to_bits()
        })
        .count();
    let fixed_integer_columns = problem
        .cols()
        .iter()
        .filter(|col| col.integer && col.lb.to_bits() == col.ub.to_bits())
        .count();
    ProblemStats {
        columns: problem.num_cols(),
        continuous_columns: problem.num_cols() - integer_columns,
        integer_columns,
        live_binary_columns,
        fixed_integer_columns,
        rows: problem.num_rows(),
        nonzeros: problem.rows().iter().map(|row| row.coeffs.len()).sum(),
        margin_row: problem.margin_row().map(|row| row.0),
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    format!("{digest:x}")
}

fn family_sha256(root_sha256: &str, split_columns: &[usize], children: &[ChildProfile]) -> String {
    let mut hash = Sha256::new();
    hash.update(SHARED_TREE_PROFILE_SCHEMA.as_bytes());
    hash.update([0]);
    hash.update(root_sha256.as_bytes());
    for &column in split_columns {
        hash.update((column as u64).to_le_bytes());
    }
    for child in children {
        hash.update((child.assignment as u64).to_le_bytes());
        hash.update(child.canonical_milp_sha256.as_bytes());
    }
    format!("{:x}", hash.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Col, Row};

    fn root_problem() -> MilpProblem {
        let mut problem = MilpProblem::new();
        let x = problem.add_col(0.0, -2.0, 3.0);
        let a = problem.add_integer_col(0.0, 0.0, 1.0);
        let b = problem.add_integer_col(0.0, 0.0, 1.0);
        let fixed = problem.add_integer_col(0.0, 1.0, 1.0);
        problem.add_row(
            f64::NEG_INFINITY,
            4.0,
            [(x, 1.0), (a, -2.0), (b, 3.0), (fixed, 0.5)],
        );
        let margin = problem.add_row(f64::NEG_INFINITY, 1.0, [(x, 1.0)]);
        problem.mark_margin_row(margin).expect("valid margin");
        problem
    }

    fn children(root: &MilpProblem) -> Vec<MilpProblem> {
        (0..4)
            .map(|assignment| {
                let mut child = root.clone();
                child.fix_col(Col(1), (assignment & 1) as f64);
                child.fix_col(Col(2), ((assignment >> 1) & 1) as f64);
                child
            })
            .collect()
    }

    #[test]
    fn accepts_complete_exact_fixed_prefix_partition() {
        let root = root_problem();
        let mut family = children(&root);
        family.reverse();
        let profile = profile_phase_split_family(&root, &family).expect("exact family");

        assert_eq!(profile.schema, SHARED_TREE_PROFILE_SCHEMA);
        assert_eq!(profile.split_columns, vec![1, 2]);
        assert_eq!(profile.split_depth, 2);
        assert_eq!(profile.isolated_sessions, 4);
        assert!(profile.complete_binary_partition);
        assert!(profile.invariant_rows_and_non_split_columns);
        assert_eq!(
            profile
                .children
                .iter()
                .map(|child| child.assignment)
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );
        assert_eq!(profile.root.columns, 4);
        assert_eq!(profile.root.continuous_columns, 1);
        assert_eq!(profile.root.integer_columns, 3);
        assert_eq!(profile.root.live_binary_columns, 2);
        assert_eq!(profile.root.fixed_integer_columns, 1);
        assert_eq!(profile.root.rows, 2);
        assert_eq!(profile.root.nonzeros, 5);
        assert_eq!(profile.root.margin_row, Some(1));
    }

    #[test]
    fn rejects_incomplete_partition() {
        let root = root_problem();
        let family = children(&root);
        assert_eq!(
            profile_phase_split_family(&root, &family[..3]),
            Err(SharedTreeProfileError::IncompletePartition {
                depth: 2,
                expected: 4,
                actual: 3,
            })
        );
    }

    #[test]
    fn rejects_duplicate_assignment() {
        let root = root_problem();
        let mut family = children(&root);
        family[3] = family[0].clone();
        assert_eq!(
            profile_phase_split_family(&root, &family),
            Err(SharedTreeProfileError::DuplicateAssignment { assignment: 0 })
        );
    }

    #[test]
    fn rejects_row_drift() {
        let root = root_problem();
        let mut family = children(&root);
        family[2].add_row(0.0, 1.0, [(Col(0), 1.0)]);
        assert_eq!(
            profile_phase_split_family(&root, &family),
            Err(SharedTreeProfileError::RowCount {
                child: 2,
                expected: 2,
                actual: 3,
            })
        );
    }

    #[test]
    fn rejects_non_split_bound_drift() {
        let root = root_problem();
        let mut family = children(&root);
        family[3].fix_col(Col(3), 0.0);
        assert_eq!(
            profile_phase_split_family(&root, &family),
            Err(SharedTreeProfileError::NonSplitColumnDrift {
                child: 3,
                column: 3,
            })
        );
    }

    #[test]
    fn rejects_non_binary_split_column() {
        let root = {
            let mut replacement = MilpProblem::new();
            let x = replacement.add_col(0.0, -2.0, 3.0);
            replacement.add_integer_col(0.0, -1.0, 1.0);
            replacement.add_row(0.0, 0.0, [(x, 1.0)]);
            replacement
        };
        let mut child = root.clone();
        child.fix_col(Col(1), 0.0);
        assert_eq!(
            profile_phase_split_family(&root, &[child]),
            Err(SharedTreeProfileError::ChangedColumnNotLiveBinary { column: 1 })
        );
    }

    #[test]
    fn detects_margin_marker_drift() {
        let root = root_problem();
        let mut family = children(&root);
        let without_marker = to_milp_text(&family[1])
            .lines()
            .filter(|line| !line.starts_with("margin "))
            .collect::<Vec<_>>()
            .join("\n");
        family[1] = crate::dump::from_milp_text(&without_marker).expect("unmarked child");
        assert_eq!(
            profile_phase_split_family(&root, &family),
            Err(SharedTreeProfileError::MarginDrift {
                child: 1,
                root: Some(1),
                observed: None,
            })
        );
    }

    #[test]
    fn canonical_hashes_are_order_independent() {
        let root = root_problem();
        let family = children(&root);
        let forward = profile_phase_split_family(&root, &family).expect("forward");
        let mut reverse_family = family;
        reverse_family.reverse();
        let reverse = profile_phase_split_family(&root, &reverse_family).expect("reverse");
        assert_eq!(forward.family_sha256, reverse.family_sha256);
        assert_eq!(forward.children, reverse.children);
    }

    #[test]
    fn test_fixture_uses_expected_margin_row() {
        assert_eq!(root_problem().margin_row(), Some(Row(1)));
    }
}
