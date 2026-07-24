// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0
//
// MilpProblem -> SMT-LIB2 (QF_LRA) lowering with EXACT f64 -> rational
// literals. Every finite f64 is m * 2^e for integers m, e, so it has an exact
// decimal / rational spelling; we emit that spelling (never a rounded
// display form), keeping the SMT problem the precise image of the IR.

use crate::error::MipError;
use crate::ir::{Col, MilpProblem};
use std::fmt::Write as _;

type Result<T> = std::result::Result<T, MipError>;

/// Optimization directive for [`super::optimize_col`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct ObjectiveSpec {
    /// Column whose value is optimized.
    pub col: Col,
    /// Direction.
    pub sense: ObjSense,
}

/// Optimization direction.
#[derive(Debug, Clone, Copy)]
pub(crate) enum ObjSense {
    Minimize,
    Maximize,
}

/// Lower the IR to a complete SMT-LIB2 script (declarations, assertions,
/// optional objective, `check-sat`, `get-value` over every column).
///
/// P0 scope: integer columns must be ReLU-indicator binaries (bounds within
/// `[0, 1]`) — they are encoded as Real variables constrained to `{0, 1}`,
/// keeping the logic QF_LRA. General-integer columns are rejected (ny's
/// encoder never emits them); NaN or inverted bounds are rejected.
pub(crate) fn to_smtlib(problem: &MilpProblem, objective: Option<ObjectiveSpec>) -> Result<String> {
    let mut s = String::with_capacity(4096);
    s.push_str("(set-logic QF_LRA)\n");

    for (i, spec) in problem.cols().iter().enumerate() {
        if spec.lb.is_nan() || spec.ub.is_nan() {
            return Err(MipError::InvalidBounds(format!("col {i}: NaN bound")));
        }
        if spec.lb > spec.ub {
            return Err(MipError::InvalidBounds(format!(
                "col {i}: inverted bounds [{}, {}]",
                spec.lb, spec.ub
            )));
        }
        let _ = writeln!(s, "(declare-const c{i} Real)");
        if spec.integer {
            if spec.lb < 0.0 || spec.ub > 1.0 {
                return Err(MipError::Encoding(format!(
                    "col {i}: general integer columns are not supported by the ay \
                     backend (P0 handles ReLU binaries only); bounds [{}, {}]",
                    spec.lb, spec.ub
                )));
            }
            // Binary: exactly 0 or 1 within the (possibly fix_col-pinned) bounds.
            let can_be_zero = spec.lb <= 0.0;
            let can_be_one = spec.ub >= 1.0;
            match (can_be_zero, can_be_one) {
                (true, true) => {
                    let _ = writeln!(s, "(assert (or (= c{i} 0.0) (= c{i} 1.0)))");
                }
                (true, false) => {
                    let _ = writeln!(s, "(assert (= c{i} 0.0))");
                }
                (false, true) => {
                    let _ = writeln!(s, "(assert (= c{i} 1.0))");
                }
                (false, false) => {
                    return Err(MipError::InvalidBounds(format!(
                        "col {i}: binary column excludes both 0 and 1 (bounds [{}, {}])",
                        spec.lb, spec.ub
                    )));
                }
            }
        } else {
            if spec.lb.is_finite() {
                let _ = writeln!(s, "(assert (>= c{i} {}))", real_literal(spec.lb)?);
            }
            if spec.ub.is_finite() {
                let _ = writeln!(s, "(assert (<= c{i} {}))", real_literal(spec.ub)?);
            }
        }
    }

    for (r, row) in problem.rows().iter().enumerate() {
        if row.lb.is_nan() || row.ub.is_nan() {
            return Err(MipError::InvalidBounds(format!("row {r}: NaN bound")));
        }
        if row.coeffs.is_empty() {
            // A row with no terms constrains the constant 0; emitting it would
            // be `lb <= 0 <= ub`, which is either trivially true or makes the
            // whole problem infeasible. Encode it faithfully either way.
            if row.lb > 0.0 || row.ub < 0.0 {
                s.push_str("(assert false)\n");
            }
            continue;
        }
        let mut sum = String::new();
        if row.coeffs.len() == 1 {
            let (c, w) = row.coeffs[0];
            let _ = write!(sum, "(* {} c{c})", real_literal(w)?);
        } else {
            sum.push_str("(+");
            for &(c, w) in &row.coeffs {
                let _ = write!(sum, " (* {} c{c})", real_literal(w)?);
            }
            sum.push(')');
        }
        if row.lb == row.ub && row.lb.is_finite() {
            let _ = writeln!(s, "(assert (= {sum} {}))", real_literal(row.lb)?);
        } else {
            if row.lb.is_finite() {
                let _ = writeln!(s, "(assert (>= {sum} {}))", real_literal(row.lb)?);
            }
            if row.ub.is_finite() {
                let _ = writeln!(s, "(assert (<= {sum} {}))", real_literal(row.ub)?);
            }
        }
    }

    if let Some(obj) = objective {
        let verb = match obj.sense {
            ObjSense::Minimize => "minimize",
            ObjSense::Maximize => "maximize",
        };
        // Callers only ever send Minimize here: ay 0.11.0's maximize lane
        // returns wrong optima on equality-defined variables in every
        // spelling probed (bare, expression-form, split inequalities, fully
        // bounded), while its minimize lane is exact. `optimize_col` lowers
        // Maximize to minimizing a negated auxiliary column before reaching
        // this function (see mod.rs). Repro + ledger: ay repo,
        // designs/2026-07-12-gurobi-class-milp-for-ny.md (P0 findings).
        let _ = writeln!(s, "({verb} c{})", obj.col.0);
    }

    s.push_str("(check-sat)\n");
    s.push_str("(get-value (");
    for i in 0..problem.num_cols() {
        if i > 0 {
            s.push(' ');
        }
        let _ = write!(s, "c{i}");
    }
    s.push_str("))\n");
    Ok(s)
}

/// Exact SMT-LIB Real literal for a finite f64.
///
/// Decomposes `x = ±m * 2^e` from the IEEE-754 bits and emits either an
/// integer-valued decimal (`e >= 0`) or the exact fraction
/// `(/ m.0 2^{-e}.0)`. Powers of two beyond u128 range are produced with
/// decimal string doubling, so the conversion is total over finite f64
/// (including subnormals) and never rounds.
fn real_literal(x: f64) -> Result<String> {
    if !x.is_finite() {
        return Err(MipError::Encoding(format!(
            "non-finite value {x} has no SMT real literal"
        )));
    }
    if x == 0.0 {
        return Ok("0.0".to_string());
    }
    let bits = x.to_bits();
    let neg = bits >> 63 == 1;
    let exp_bits = ((bits >> 52) & 0x7ff) as i64;
    let frac = bits & ((1u64 << 52) - 1);
    let (mut mant, mut exp) = if exp_bits == 0 {
        (frac, -1074i64) // subnormal
    } else {
        (frac | (1u64 << 52), exp_bits - 1075)
    };
    while mant & 1 == 0 && exp < 0 {
        mant >>= 1;
        exp += 1;
    }
    let body = if exp >= 0 {
        format!("{}.0", shl_decimal(mant, exp as u32))
    } else {
        format!("(/ {mant}.0 {}.0)", shl_decimal(1, (-exp) as u32))
    };
    Ok(if neg { format!("(- {body})") } else { body })
}

/// Decimal string of `m * 2^k`, via repeated doubling on digit vectors —
/// total for any `k` a finite f64 can produce (max ~1074).
fn shl_decimal(m: u64, k: u32) -> String {
    // Little-endian decimal digits.
    let mut digits: Vec<u8> = {
        let mut v = Vec::new();
        let mut m = m;
        if m == 0 {
            v.push(0);
        }
        while m > 0 {
            v.push((m % 10) as u8);
            m /= 10;
        }
        v
    };
    for _ in 0..k {
        let mut carry = 0u8;
        for d in &mut digits {
            let doubled = *d * 2 + carry;
            *d = doubled % 10;
            carry = doubled / 10;
        }
        if carry > 0 {
            digits.push(carry);
        }
    }
    digits.iter().rev().map(|d| (b'0' + d) as char).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::MilpProblem;

    #[test]
    fn test_real_literal_integers_exact() {
        assert_eq!(real_literal(0.0).expect("0"), "0.0");
        assert_eq!(real_literal(1.0).expect("1"), "1.0");
        assert_eq!(real_literal(-3.0).expect("-3"), "(- 3.0)");
        assert_eq!(real_literal(1024.0).expect("1024"), "1024.0");
    }

    #[test]
    fn test_real_literal_dyadic_fractions_exact() {
        assert_eq!(real_literal(0.5).expect("1/2"), "(/ 1.0 2.0)");
        assert_eq!(real_literal(-0.75).expect("-3/4"), "(- (/ 3.0 4.0))");
        // 0.1 is NOT 1/10 in f64; the literal must be the exact dyadic value.
        let tenth = real_literal(0.1).expect("0.1");
        assert_eq!(
            tenth, "(/ 3602879701896397.0 36028797018963968.0)",
            "0.1 must lower to its exact f64 rational, not a decimal display"
        );
    }

    #[test]
    fn test_real_literal_subnormal_is_total() {
        let tiny = f64::from_bits(1); // smallest positive subnormal, 2^-1074
        let lit = real_literal(tiny).expect("subnormal must lower");
        assert!(lit.starts_with("(/ 1.0 "), "got: {lit}");
    }

    #[test]
    fn test_shl_decimal_matches_u128() {
        for k in [0u32, 1, 7, 63, 100] {
            let expect = (3u128) << k.min(120);
            if k <= 120 {
                assert_eq!(shl_decimal(3, k), expect.to_string(), "k={k}");
            }
        }
        assert_eq!(shl_decimal(1, 10), "1024");
    }

    #[test]
    fn test_to_smtlib_binary_becomes_disjunction() {
        let mut p = MilpProblem::new();
        let x = p.add_col(0.0, 0.0, 1.0);
        let z = p.add_integer_col(0.0, 0.0, 1.0);
        p.add_row(0.0, f64::INFINITY, [(x, 1.0), (z, -1.0)]);
        let s = to_smtlib(&p, None).expect("lowering must succeed");
        assert!(s.contains("(assert (or (= c1 0.0) (= c1 1.0)))"), "{s}");
        assert!(s.contains("(set-logic QF_LRA)"), "{s}");
        assert!(s.contains("(get-value (c0 c1))"), "{s}");
    }

    #[test]
    fn test_to_smtlib_fixed_binary_becomes_equality() {
        let mut p = MilpProblem::new();
        let z = p.add_integer_col(0.0, 0.0, 1.0);
        p.fix_col(z, 1.0);
        let s = to_smtlib(&p, None).expect("lowering must succeed");
        assert!(s.contains("(assert (= c0 1.0))"), "{s}");
    }

    #[test]
    fn test_to_smtlib_general_integer_rejected() {
        let mut p = MilpProblem::new();
        p.add_integer_col(0.0, 0.0, 5.0);
        let err = to_smtlib(&p, None).expect_err("general integers must be rejected in P0");
        assert!(matches!(err, MipError::Encoding(_)), "{err:?}");
    }

    #[test]
    fn test_to_smtlib_objective_emitted() {
        let mut p = MilpProblem::new();
        let x = p.add_col(0.0, 0.0, 1.0);
        let s = to_smtlib(
            &p,
            Some(ObjectiveSpec {
                col: x,
                sense: ObjSense::Minimize,
            }),
        )
        .expect("lowering must succeed");
        assert!(s.contains("(minimize c0)"), "{s}");
    }
}
