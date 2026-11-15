// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Solver-free lower-bound checking for a SOURCE-AUTHENTICATED linear relaxation.
//!
//! This module does not authenticate neural networks, row derivations or root
//! coverage, and cannot authorize an UNSAT verdict. Its caller MUST construct
//! every row from the admitted source program and its rigorous source bounds;
//! a candidate certificate supplies only multipliers, never replacement rows,
//! constants, variable boxes, objectives or claimed bound values.
//!
//! Equality rows are `a*x = rhs`, inequality rows are `a*x >= rhs`. Given free
//! equality weights mu and nonnegative inequality weights lambda, every feasible
//! x satisfies
//!
//! ```text
//! c*x >= mu*b_eq + lambda*b_ineq
//!          + (c - mu*A_eq - lambda*A_ineq)*x.
//! ```
//!
//! The residual coefficient vector need NOT vanish: its box minimum is retained
//! in full. This makes arbitrary finite, poorly optimized proposals safe. Every
//! weighted coefficient, weighted RHS, residual and box product is computed as
//! an outward interval, charging the floating dot-product errors rather than
//! treating approximate cancellation as an exact Farkas identity.
//!
//! A runtime check qualifies binary64 round-to-nearest and gradual underflow
//! on the executing thread before any dual is admitted.

#[derive(Clone, Debug)]
pub(super) struct LinearRow {
    /// Sparse ordered exact-f64 coefficients. Duplicate indices are allowed;
    /// their mathematical contributions are summed with outward arithmetic.
    pub(super) terms: Vec<(usize, f64)>,
    pub(super) rhs: f64,
}

#[derive(Clone, Debug)]
pub(super) struct DualMultipliers {
    pub(super) equality_multipliers: Vec<f64>,
    pub(super) inequality_multipliers: Vec<f64>,
}

const MAX_VARIABLES: usize = 8_192;
const MAX_ROWS: usize = 100_000;
const MAX_TERMS: usize = 5_000_000;
const MAX_MAGNITUDE: f64 = 1e100;

#[derive(Clone, Copy)]
struct Interval {
    lo: f64,
    hi: f64,
}

impl Interval {
    fn point(value: f64) -> Self {
        Self {
            lo: value,
            hi: value,
        }
    }

    fn valid(self) -> bool {
        self.lo.is_finite() && self.hi.is_finite() && self.lo <= self.hi
    }

    fn add(self, rhs: Self) -> Option<Self> {
        let result = Self {
            lo: add_down(self.lo, rhs.lo),
            hi: add_up(self.hi, rhs.hi),
        };
        result.valid().then_some(result)
    }

    fn neg(self) -> Self {
        Self {
            lo: -self.hi,
            hi: -self.lo,
        }
    }

    fn sub(self, rhs: Self) -> Option<Self> {
        self.add(rhs.neg())
    }

    fn mul(self, rhs: Self) -> Option<Self> {
        let products = [
            (self.lo, rhs.lo),
            (self.lo, rhs.hi),
            (self.hi, rhs.lo),
            (self.hi, rhs.hi),
        ];
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for (a, b) in products {
            lo = lo.min(mul_down(a, b));
            hi = hi.max(mul_up(a, b));
        }
        let result = Self { lo, hi };
        result.valid().then_some(result)
    }
}

// Exact zero/sign/unit identities avoid gratuitous subnormal charges. Every
// other primitive is widened, including a rounded zero due to underflow.
fn add_down(a: f64, b: f64) -> f64 {
    if a == 0.0 {
        b
    } else if b == 0.0 {
        a
    } else if a == -b {
        0.0
    } else {
        (a + b).next_down()
    }
}
fn add_up(a: f64, b: f64) -> f64 {
    if a == 0.0 {
        b
    } else if b == 0.0 {
        a
    } else if a == -b {
        0.0
    } else {
        (a + b).next_up()
    }
}
fn mul_down(a: f64, b: f64) -> f64 {
    if a == 0.0 || b == 0.0 {
        0.0
    } else if a == 1.0 {
        b
    } else if b == 1.0 {
        a
    } else if a == -1.0 {
        -b
    } else if b == -1.0 {
        -a
    } else {
        (a * b).next_down()
    }
}
fn mul_up(a: f64, b: f64) -> f64 {
    if a == 0.0 || b == 0.0 {
        0.0
    } else if a == 1.0 {
        b
    } else if b == 1.0 {
        a
    } else if a == -1.0 {
        -b
    } else if b == -1.0 {
        -a
    } else {
        (a * b).next_up()
    }
}

fn admissible(value: f64) -> bool {
    value.is_finite() && value.abs() <= MAX_MAGNITUDE
}

pub(super) fn suitable_fp_environment() -> bool {
    use std::hint::black_box;
    let large = black_box(9_007_199_254_740_992.0_f64);
    let one = black_box(1.0_f64);
    let three_halves = black_box(1.5_f64);
    let tiny = black_box(f64::from_bits(1));
    let two = black_box(2.0_f64);
    let normal = black_box(f64::MIN_POSITIVE);
    let half = black_box(0.5_f64);
    black_box(large + one) == large
        && black_box(large + three_halves) == large + 2.0
        && black_box(-large - three_halves) == -large - 2.0
        && black_box(tiny * two).to_bits() == 2
        && black_box(tiny + tiny).to_bits() == 2
        && black_box(normal * half).to_bits() == (1u64 << 51)
}

fn validate_rows(rows: &[LinearRow], variables: usize, total_terms: &mut usize) -> Option<()> {
    for row in rows {
        if !admissible(row.rhs) {
            return None;
        }
        *total_terms = total_terms.checked_add(row.terms.len())?;
        if *total_terms > MAX_TERMS {
            return None;
        }
        if row
            .terms
            .iter()
            .any(|&(j, a)| j >= variables || !admissible(a))
        {
            return None;
        }
    }
    Some(())
}

/// Compute a rigorously repaired dual lower bound for the immutable relaxation.
/// This arithmetic checker is INTERNAL and returns a number, not proof authority.
pub(super) fn certified_lower(
    variables_box: &[(f64, f64)],
    c: &[f64],
    eqrows: &[LinearRow],
    ineqrows: &[LinearRow],
    dual: &DualMultipliers,
) -> Option<f64> {
    let variables = variables_box.len();
    if !suitable_fp_environment()
        || variables == 0
        || variables > MAX_VARIABLES
        || c.len() != variables
        || eqrows.len().checked_add(ineqrows.len())? > MAX_ROWS
        || dual.equality_multipliers.len() != eqrows.len()
        || dual.inequality_multipliers.len() != ineqrows.len()
        || variables_box
            .iter()
            .any(|&(l, h)| !admissible(l) || !admissible(h) || l > h)
        || c.iter().any(|&v| !admissible(v))
        || dual.equality_multipliers.iter().any(|&v| !admissible(v))
        || dual
            .inequality_multipliers
            .iter()
            .any(|&v| !admissible(v) || v < 0.0)
    {
        return None;
    }
    let mut total_terms = 0;
    validate_rows(eqrows, variables, &mut total_terms)?;
    validate_rows(ineqrows, variables, &mut total_terms)?;

    let mut weighted_coefficients = vec![Interval::point(0.0); variables];
    let mut weighted_rhs = Interval::point(0.0);
    let contributions = eqrows
        .iter()
        .zip(&dual.equality_multipliers)
        .map(|(row, &w)| (row, w))
        .chain(
            ineqrows
                .iter()
                .zip(&dual.inequality_multipliers)
                .map(|(row, &w)| (row, w)),
        );
    for (row, weight) in contributions {
        if weight == 0.0 {
            continue;
        }
        let multiplier = Interval::point(weight);
        weighted_rhs = weighted_rhs.add(multiplier.mul(Interval::point(row.rhs))?)?;
        for &(j, a) in &row.terms {
            weighted_coefficients[j] =
                weighted_coefficients[j].add(multiplier.mul(Interval::point(a))?)?;
        }
    }
    let mut bound = weighted_rhs.lo;
    for j in 0..variables {
        let residual = Interval::point(c[j]).sub(weighted_coefficients[j])?;
        let box_interval = Interval {
            lo: variables_box[j].0,
            hi: variables_box[j].1,
        };
        let term = residual.mul(box_interval)?;
        bound = add_down(bound, term.lo);
        if !bound.is_finite() {
            return None;
        }
    }
    bound.is_finite().then_some(bound)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(terms: &[(usize, f64)], rhs: f64) -> LinearRow {
        LinearRow {
            terms: terms.to_vec(),
            rhs,
        }
    }
    fn dual(equality: &[f64], inequality: &[f64]) -> DualMultipliers {
        DualMultipliers {
            equality_multipliers: equality.to_vec(),
            inequality_multipliers: inequality.to_vec(),
        }
    }

    #[test]
    fn dyadic_inequality_and_equality_duals_prove_exact_toy_bounds() {
        // x >= 2 over [0,4]. The exact dual lambda=1 closes min x = 2.
        assert_eq!(
            certified_lower(
                &[(0.0, 4.0)],
                &[1.0],
                &[],
                &[row(&[(0, 1.0)], 2.0)],
                &dual(&[], &[1.0])
            ),
            Some(2.0)
        );
        // x+y=3, x<=1; min y = 2, with mu=1 and lambda=1.
        let lower = certified_lower(
            &[(0.0, 4.0), (0.0, 4.0)],
            &[0.0, 1.0],
            &[row(&[(0, 1.0), (1, 1.0)], 3.0)],
            &[row(&[(0, -1.0)], -1.0)],
            &dual(&[1.0], &[1.0]),
        )
        .unwrap();
        assert!(lower <= 2.0 && lower > 1.99999999999999);
    }

    #[test]
    fn free_negative_equality_weights_are_allowed_but_negative_inequality_weights_refused() {
        assert_eq!(
            certified_lower(
                &[(0.0, 4.0)],
                &[1.0],
                &[row(&[(0, -1.0)], -2.0)],
                &[],
                &dual(&[-1.0], &[])
            ),
            Some(2.0)
        );
        assert!(certified_lower(
            &[(0.0, 4.0)],
            &[1.0],
            &[],
            &[row(&[(0, 1.0)], 2.0)],
            &dual(&[], &[-1.0])
        )
        .is_none());
    }

    #[test]
    fn unmatched_residual_is_not_mistaken_for_an_exact_farkas_cancellation() {
        // A forged/poor multiplier 16 leaves c - weighted_row = -15. The
        // variable's box minimum must be charged; discarding it would claim32.
        let actual = certified_lower(
            &[(0.0, 4.0)],
            &[1.0],
            &[],
            &[row(&[(0, 1.0)], 2.0)],
            &dual(&[], &[16.0]),
        )
        .unwrap();
        assert!(actual <= -28.0);
    }

    #[test]
    fn a_proposal_from_forged_rows_or_bias_cannot_override_the_original_problem() {
        // Candidates carry multipliers ONLY. Whatever false authored model
        // motivated the proposal, source reconstruction fixes x>=2 and keeps
        // its RHS at 2. No multiplier can manufacture a lower bound >2.
        let actual_rows = [row(&[(0, 1.0)], 2.0)];
        for lambda in [0.0, 0.125, 0.5, 1.0, 2.0, 16.0, 1024.0] {
            let checked = certified_lower(
                &[(0.0, 4.0)],
                &[1.0],
                &[],
                &actual_rows,
                &dual(&[], &[lambda]),
            )
            .unwrap();
            assert!(
                checked <= 2.0,
                "false proposal {lambda} manufactured {checked}"
            );
            assert!(
                checked.partial_cmp(&2.5) != Some(std::cmp::Ordering::Greater),
                "feasible unsafe point x=2 remains uncovered"
            );
        }
    }

    #[test]
    fn sound_lower_bound_holds_on_every_feasible_dyadic_grid_point() {
        let eqrows = [row(&[(0, 1.0), (1, 1.0)], 3.0)];
        let ineqrows = [row(&[(0, -1.0)], -1.0), row(&[(1, 1.0)], 1.0)];
        for mu in [-2.0, -0.25, 0.0, 0.5, 1.0, 2.0] {
            for a in [0.0, 0.25, 1.0, 4.0] {
                for b in [0.0, 0.5, 2.0] {
                    let lower = certified_lower(
                        &[(-1.0, 4.0), (-1.0, 4.0)],
                        &[0.25, 1.0],
                        &eqrows,
                        &ineqrows,
                        &dual(&[mu], &[a, b]),
                    )
                    .unwrap();
                    for i in -8..=8 {
                        let x = f64::from(i) / 8.0;
                        let y = 3.0 - x;
                        assert!(lower <= 0.25 * x + y);
                    }
                }
            }
        }
    }

    #[test]
    fn empty_dual_reduces_to_the_box_objective_and_duplicates_are_enclosed() {
        let box_bound = certified_lower(
            &[(-2.0, 3.0), (1.0, 4.0)],
            &[0.5, -1.0],
            &[],
            &[],
            &dual(&[], &[]),
        )
        .unwrap();
        assert!(box_bound <= -5.0 && box_bound > -5.00000000000001);
        let duplicates = [row(&[(0, 0.5), (0, 0.5)], 2.0)];
        let checked =
            certified_lower(&[(0.0, 4.0)], &[1.0], &duplicates, &[], &dual(&[1.0], &[])).unwrap();
        assert!(checked <= 2.0 && checked > 1.99999999999999);
    }

    #[test]
    fn malformed_or_nonfinite_proposals_fail_closed() {
        let rows = [row(&[(0, 1.0)], 2.0)];
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.125] {
            assert!(
                certified_lower(&[(0.0, 4.0)], &[1.0], &[], &rows, &dual(&[], &[value])).is_none()
            );
        }
        assert!(certified_lower(&[(0.0, 4.0)], &[1.0], &[], &rows, &dual(&[], &[])).is_none());
        assert!(certified_lower(&[(4.0, 0.0)], &[1.0], &[], &[], &dual(&[], &[])).is_none());
        assert!(certified_lower(&[(0.0, 4.0)], &[f64::NAN], &[], &[], &dual(&[], &[])).is_none());
        assert!(certified_lower(
            &[(0.0, 4.0)],
            &[1.0],
            &[row(&[(1, 1.0)], 2.0)],
            &[],
            &dual(&[1.0], &[])
        )
        .is_none());
        assert!(certified_lower(
            &[(0.0, 4.0)],
            &[1.0],
            &[row(&[(0, 1.0)], f64::NAN)],
            &[],
            &dual(&[1.0], &[])
        )
        .is_none());
    }
}
