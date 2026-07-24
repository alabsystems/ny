// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0
//
// Solver-neutral MILP intermediate representation — the backend seam.
//
// The encoder builds a `MilpProblem` (plain data: Send + Sync + Clone), and
// backend lowerings translate it into the concrete solver's model right
// before each solve: `ay::to_smtlib` in production, `to_highs` under the
// mip-diff gate (docs/SOLVER_POLICY.md; HiGHS deleted at LG3). Keeping the IR
// solver-free is what makes backend dispatch and parallel phase-split racing
// (one model per thread, built from the shared IR) trivially thread-safe.

/// Column (variable) index into a [`MilpProblem`].
///
/// Replaces `highs::Col` in the encoder/`MipParts` surface. The index is the
/// insertion order of the column, which every lowering preserves, so `Col(i)`
/// addresses the same variable in every backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Col(pub usize);

/// Row (constraint) index into a [`MilpProblem`].
///
/// Like [`Col`], this is an insertion-order identity rather than a bare
/// integer.  Backends preserve row order, so `Row(i)` names the same
/// constraint before and after lowering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Row(pub usize);

/// Why a row cannot be named as the unique decision-margin row.
///
/// The marker is solver advice, but assigning it to the wrong constraint can
/// change solver routing.  Reject ambiguous identities at the IR boundary and
/// let callers decline the optional lane instead of guessing.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum MarginRowError {
    /// The row handle does not belong to this problem.
    #[error("margin row {row} is out of range ({rows} rows)")]
    OutOfRange { row: usize, rows: usize },
    /// The row has no effective nonzero coefficient.
    #[error("margin row {row} has no nonzero coefficients")]
    Empty { row: usize },
    /// Margin reframing requires a finite linear form.
    #[error("margin row {row} has a non-finite coefficient")]
    NonFiniteCoefficient { row: usize },
    /// A decision margin must have exactly one finite bound.
    #[error("margin row {row} is not a single one-sided inequality")]
    NotOneSided { row: usize },
    /// One problem cannot carry two competing decision-row identities.
    #[error(
        "margin row {existing} is already marked; refusing to replace it with row {requested}"
    )]
    AlreadyMarked { existing: usize, requested: usize },
}

/// Column (variable) specification: bounds, objective coefficient, integrality.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColSpec {
    /// Lower bound (may be `f64::NEG_INFINITY`).
    pub lb: f64,
    /// Upper bound (may be `f64::INFINITY`).
    pub ub: f64,
    /// Objective coefficient.
    pub obj: f64,
    /// Whether the variable is integer-constrained.
    pub integer: bool,
}

/// Row (linear constraint) specification: `lb <= sum(coeff_i * x_i) <= ub`.
///
/// One-sided rows use `f64::NEG_INFINITY` / `f64::INFINITY` for the open side;
/// equalities use `lb == ub`.
#[derive(Debug, Clone, PartialEq)]
pub struct RowSpec {
    /// Row lower bound (may be `f64::NEG_INFINITY`).
    pub lb: f64,
    /// Row upper bound (may be `f64::INFINITY`).
    pub ub: f64,
    /// Sparse coefficients as `(column index, weight)` pairs.
    pub coeffs: Vec<(usize, f64)>,
}

/// Solver-neutral MILP problem: plain data, no solver handles.
#[derive(Debug, Clone, Default)]
pub struct MilpProblem {
    cols: Vec<ColSpec>,
    rows: Vec<RowSpec>,
    /// Explicit caller-owned identity of the unique decision/violation row.
    ///
    /// `None` is the default and preserves the historical plain-feasibility
    /// path.  This metadata is cloned with the IR and therefore travels inside
    /// `MipParts`; no environment variable or row-shape inference can opt an
    /// unrelated model into margin reframing.
    margin_row: Option<Row>,
}

impl MilpProblem {
    /// Create an empty problem.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a continuous column with objective coefficient `obj` and bounds `[lb, ub]`.
    pub fn add_col(&mut self, obj: f64, lb: f64, ub: f64) -> Col {
        self.push_col(ColSpec {
            lb,
            ub,
            obj,
            integer: false,
        })
    }

    /// Relax every integer column to continuous (the LP RELAXATION in place).
    /// Bounds are kept, so a fixed binary (`[v, v]`) stays the constant `v`.
    /// Used by the certified phase-enumeration lane: with all binaries FIXED
    /// by enumeration, the relaxed problem is a pure LP whose infeasibility
    /// the exact Farkas lane can certify (#relational-bab edge escalation).
    pub fn relax_integrality(&mut self) {
        for col in &mut self.cols {
            col.integer = false;
        }
    }

    /// Add an integer column with objective coefficient `obj` and bounds `[lb, ub]`.
    pub fn add_integer_col(&mut self, obj: f64, lb: f64, ub: f64) -> Col {
        self.push_col(ColSpec {
            lb,
            ub,
            obj,
            integer: true,
        })
    }

    fn push_col(&mut self, spec: ColSpec) -> Col {
        let idx = self.cols.len();
        self.cols.push(spec);
        Col(idx)
    }

    /// Add a row constraint `lb <= sum(coeff * col) <= ub`.
    ///
    /// Use `f64::NEG_INFINITY` / `f64::INFINITY` for one-sided rows and
    /// `lb == ub` for equalities.
    pub fn add_row<I: IntoIterator<Item = (Col, f64)>>(
        &mut self,
        lb: f64,
        ub: f64,
        coeffs: I,
    ) -> Row {
        let row = Row(self.rows.len());
        self.rows.push(RowSpec {
            lb,
            ub,
            coeffs: coeffs.into_iter().map(|(c, w)| (c.0, w)).collect(),
        });
        row
    }

    /// Atomically add and explicitly mark a decision-margin row.
    ///
    /// On validation failure the appended row is rolled back, so an optional
    /// caller such as Graph-MIP can fail closed without leaving a partially
    /// mutated ordinary constraint behind.
    pub fn add_margin_row<I: IntoIterator<Item = (Col, f64)>>(
        &mut self,
        lb: f64,
        ub: f64,
        coeffs: I,
    ) -> Result<Row, MarginRowError> {
        let row = self.add_row(lb, ub, coeffs);
        if let Err(error) = self.mark_margin_row(row) {
            let removed = self.rows.pop();
            debug_assert!(removed.is_some(), "the just-added margin row must exist");
            return Err(error);
        }
        Ok(row)
    }

    /// Name an existing one-sided row as this problem's decision margin.
    ///
    /// This is deliberately explicit and caller-gated: ordinary one-sided
    /// constraints are never inferred to be margins.  Re-marking the same row
    /// is idempotent, while attempting to replace it with a different row is
    /// rejected so no "last row wins" ambiguity can cross the backend seam.
    pub fn mark_margin_row(&mut self, row: Row) -> Result<(), MarginRowError> {
        if let Some(existing) = self.margin_row {
            if existing == row {
                return Ok(());
            }
            return Err(MarginRowError::AlreadyMarked {
                existing: existing.0,
                requested: row.0,
            });
        }
        let spec = self.rows.get(row.0).ok_or(MarginRowError::OutOfRange {
            row: row.0,
            rows: self.rows.len(),
        })?;
        if !(spec.lb.is_finite() ^ spec.ub.is_finite()) {
            return Err(MarginRowError::NotOneSided { row: row.0 });
        }

        // AY canonicalizes duplicate columns and drops zero coefficients.
        // Mirror that effective-emptiness check here so a row such as
        // `x - x <= t` fails closed before it reaches the backend.
        let mut coeffs = spec.coeffs.clone();
        coeffs.sort_unstable_by_key(|&(col, _)| col);
        coeffs.dedup_by(|later, first| {
            if later.0 == first.0 {
                first.1 += later.1;
                true
            } else {
                false
            }
        });
        if coeffs.iter().any(|&(_, coeff)| !coeff.is_finite()) {
            return Err(MarginRowError::NonFiniteCoefficient { row: row.0 });
        }
        if !coeffs.iter().any(|&(_, coeff)| coeff != 0.0) {
            return Err(MarginRowError::Empty { row: row.0 });
        }
        self.margin_row = Some(row);
        Ok(())
    }

    /// The explicitly marked decision-margin row, if the caller supplied one.
    pub fn margin_row(&self) -> Option<Row> {
        self.margin_row
    }

    /// Fix a column to a single value by shrinking its bounds to `[value, value]`.
    ///
    /// Used by phase-split racing to pin ReLU indicator binaries; sound because
    /// it only restricts the feasible set of this clone.
    pub fn fix_col(&mut self, col: Col, value: f64) {
        let spec = &mut self.cols[col.0];
        spec.lb = value;
        spec.ub = value;
    }

    /// Number of columns (variables).
    pub fn num_cols(&self) -> usize {
        self.cols.len()
    }

    /// Number of rows (constraints).
    pub fn num_rows(&self) -> usize {
        self.rows.len()
    }

    /// Column specifications, in insertion order.
    pub fn cols(&self) -> &[ColSpec] {
        &self.cols
    }

    /// Row specifications, in insertion order.
    pub fn rows(&self) -> &[RowSpec] {
        &self.rows
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ir_is_send_sync_clone() {
        fn assert_send_sync_clone<T: Send + Sync + Clone>() {}
        assert_send_sync_clone::<MilpProblem>();
    }

    #[test]
    fn col_indices_are_insertion_order() {
        let mut p = MilpProblem::new();
        let a = p.add_col(0.0, 0.0, 1.0);
        let b = p.add_integer_col(0.0, 0.0, 1.0);
        assert_eq!(a, Col(0));
        assert_eq!(b, Col(1));
        assert_eq!(p.num_cols(), 2);
        assert!(!p.cols()[0].integer);
        assert!(p.cols()[1].integer);
    }

    #[test]
    fn fix_col_pins_bounds() {
        let mut p = MilpProblem::new();
        let z = p.add_integer_col(0.0, 0.0, 1.0);
        p.fix_col(z, 1.0);
        assert_eq!(p.cols()[0].lb, 1.0);
        assert_eq!(p.cols()[0].ub, 1.0);
    }

    #[test]
    fn margin_identity_is_explicit_unique_and_clone_stable() {
        let mut p = MilpProblem::new();
        let x = p.add_col(0.0, 0.0, 1.0);
        let ordinary = p.add_row(0.25, f64::INFINITY, [(x, 1.0)]);
        let margin = p.add_row(f64::NEG_INFINITY, 0.5, [(x, 1.0)]);

        assert_eq!(p.margin_row(), None, "shape must never imply identity");
        p.mark_margin_row(margin).expect("valid one-sided margin");
        p.mark_margin_row(margin)
            .expect("same-row mark is idempotent");
        assert_eq!(p.margin_row(), Some(margin));
        assert_eq!(p.clone().margin_row(), Some(margin));
        assert!(matches!(
            p.mark_margin_row(ordinary),
            Err(MarginRowError::AlreadyMarked { .. })
        ));
        assert_eq!(p.margin_row(), Some(margin), "failed replacement is inert");
    }

    #[test]
    fn malformed_margin_rows_fail_closed() {
        let mut p = MilpProblem::new();
        let x = p.add_col(0.0, 0.0, 1.0);
        let equality = p.add_row(0.5, 0.5, [(x, 1.0)]);
        let empty = p.add_row(f64::NEG_INFINITY, 0.5, [(x, 0.0)]);
        let cancelled = p.add_row(f64::NEG_INFINITY, 0.5, [(x, 1.0), (x, -1.0)]);
        let not_finite = p.add_row(f64::NEG_INFINITY, 0.5, [(x, f64::NAN)]);
        assert!(matches!(
            p.mark_margin_row(equality),
            Err(MarginRowError::NotOneSided { .. })
        ));
        assert!(matches!(
            p.mark_margin_row(empty),
            Err(MarginRowError::Empty { .. })
        ));
        assert!(matches!(
            p.mark_margin_row(cancelled),
            Err(MarginRowError::Empty { .. })
        ));
        assert!(matches!(
            p.mark_margin_row(not_finite),
            Err(MarginRowError::NonFiniteCoefficient { .. })
        ));
        assert!(matches!(
            p.mark_margin_row(Row(usize::MAX)),
            Err(MarginRowError::OutOfRange { .. })
        ));
        assert_eq!(p.margin_row(), None);
    }

    #[test]
    fn atomic_margin_add_rolls_back_on_rejection() {
        let mut p = MilpProblem::new();
        let x = p.add_col(0.0, 0.0, 1.0);
        let rows_before = p.num_rows();
        assert!(matches!(
            p.add_margin_row(0.5, 0.5, [(x, 1.0)]),
            Err(MarginRowError::NotOneSided { .. })
        ));
        assert_eq!(p.num_rows(), rows_before);
        assert_eq!(p.margin_row(), None);
    }
}
