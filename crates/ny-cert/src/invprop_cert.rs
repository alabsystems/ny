// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! INVPROP assume-violation certificate.
//!
//! The INVPROP output-seed dual (see `docs/INVPROP_ASSUME_VIOLATION_DESIGN.md`)
//! proves a property HOLDS by showing the VIOLATION region
//! `V = { y : C y <= rhs }` is empty. Its soundness object is a **single
//! non-negative (Farkas) combination** the EXISTING kernel replays — no new
//! trusted linear machinery:
//!
//! - the base CROWN relaxation premises (the network's sound linear bounds over
//!   the input box) with their Farkas multipliers, PLUS
//! - the violation rows `C y <= rhs` (kind [`ConstraintKind::Le`]) each carried
//!   with a non-negative multiplier `gamma >= 0`.
//!
//! When the combination cancels all variables to a strictly-negative constant,
//! `V` is empty (a Farkas contradiction of the relaxation-plus-violation system)
//! and the property holds. Because every violation multiplier is `gamma >= 0`, a
//! wrong or suboptimal `gamma` can only fail to establish the contradiction
//! ([`crate::selfcheck::CheckError::NotEstablished`] / uncancelled variables) —
//! it can never manufacture one. This is exactly the moat property: the kernel
//! re-derives the residual and never trusts NY's claimed bound.
//!
//! This mirrors the "negate the row" reconciliation in the design's §7: under the
//! settled VIOLATION semantics the rows are used DIRECTLY (non-strict `Le`,
//! `gamma >= 0`), so the [`crate::selfcheck::check_farkas`] `StrictRelation`/`Eq`
//! rejects never fire and the refutation lands on a strictly-negative constant.

use crate::rational::Rat;
use crate::schema::{ConstraintKind, FarkasCertificate, LinearConstraint};
use std::collections::BTreeMap;

/// One assume-violation output-constraint row `C_r y <= rhs_r` with its
/// non-negative dual multiplier `gamma_r`.
#[derive(Debug, Clone)]
pub struct OutputDualRow {
    /// The violation row `sum_j C[r,j]*y_j <= rhs_r` (kind [`ConstraintKind::Le`]).
    pub row: LinearConstraint,
    /// Non-negative Lagrange/Farkas multiplier `gamma_r >= 0`.
    pub gamma: Rat,
}

/// The INVPROP-augmented HOLD certificate: base CROWN Farkas premises + the
/// violation rows dualized with `gamma >= 0`.
#[derive(Debug, Clone)]
pub struct InvpropAugmentedCertificate {
    /// Base relaxation premises (network sound linear bounds over the box), each a
    /// [`LinearConstraint`]; sound-CROWN provenance is the same trust boundary as
    /// [`crate::crown_deep::CertifiedDeep`].
    pub base_premises: Vec<LinearConstraint>,
    /// Non-negative Farkas multipliers for the base premises (one per premise).
    pub base_multipliers: Vec<Rat>,
    /// Assume-violation output rows `C y <= rhs`, each with `gamma >= 0`.
    pub output_rows: Vec<OutputDualRow>,
}

impl InvpropAugmentedCertificate {
    /// Lower to a single [`FarkasCertificate`] the existing kernel replays:
    /// base premises ++ violation rows, base multipliers ++ gammas.
    #[must_use]
    pub fn to_farkas(&self) -> FarkasCertificate {
        let mut constraints = self.base_premises.clone();
        let mut multipliers = self.base_multipliers.clone();
        for r in &self.output_rows {
            constraints.push(r.row.clone());
            multipliers.push(r.gamma);
        }
        FarkasCertificate {
            constraints,
            multipliers,
        }
    }

    /// Whether every dual multiplier is non-negative (the Farkas requirement).
    /// `check_farkas` also rejects a negative multiplier, so this is a fast
    /// pre-flight, not the trusted check.
    #[must_use]
    pub fn gammas_nonneg(&self) -> bool {
        // Explicit loop (not `.iter().all(..)`): the `all` adapter is an
        // absent-callee for the panic-freedom checker; the loop is the identical
        // short-circuit (return `false` on the first negative gamma, else `true`,
        // vacuously `true` on no rows).
        for r in self.output_rows.iter() {
            if r.gamma.is_negative() {
                return false;
            }
        }
        true
    }

    /// Number of base premises + violation rows (must equal the multiplier count
    /// in the lowered Farkas certificate).
    #[must_use]
    pub fn len(&self) -> usize {
        // `saturating_add` (not `+`): two `Vec::len()` are each `<= isize::MAX`,
        // so `2 * isize::MAX < usize::MAX` — the sum never saturates and equals
        // `+` exactly, while clearing the Add-overflow obligation.
        self.base_premises
            .len()
            .saturating_add(self.output_rows.len())
    }

    /// Whether the certificate carries no premises at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.base_premises.is_empty() && self.output_rows.is_empty()
    }
}

/// Build a violation row `y_var <= rhs` (kind `Le`), the common single-output
/// assume-violation form produced by `to_output_constraints` (`Y_i <= c`).
#[must_use]
pub fn le_row(var: &str, coeff: Rat, rhs: Rat) -> LinearConstraint {
    let mut coefficients = BTreeMap::new();
    coefficients.insert(var.to_string(), coeff);
    LinearConstraint {
        kind: ConstraintKind::Le,
        coefficients,
        constant: rhs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::selfcheck::{check_farkas, CheckError};

    /// A network lower-bound premise `y >= threshold` as `Ge` (normalize turns it
    /// into `-y <= -threshold`).
    fn ge_row(var: &str, threshold: Rat) -> LinearConstraint {
        let mut coefficients = BTreeMap::new();
        coefficients.insert(var.to_string(), Rat::from_int(1));
        LinearConstraint {
            kind: ConstraintKind::Ge,
            coefficients,
            constant: threshold,
        }
    }

    /// Valid assume-violation HOLD: the network proves `y >= 3`, the violation is
    /// `y <= 1`. With gamma = 1 the combination cancels `y` to the constant
    /// `rhs - L = 1 - 3 = -2 < 0` => V empty => HOLD. The kernel accepts it.
    #[test]
    fn invprop_farkas_accepts_valid_hold() {
        let cert = InvpropAugmentedCertificate {
            base_premises: vec![ge_row("y", Rat::from_int(3))],
            base_multipliers: vec![Rat::from_int(1)],
            output_rows: vec![OutputDualRow {
                row: le_row("y", Rat::from_int(1), Rat::from_int(1)),
                gamma: Rat::from_int(1),
            }],
        };
        assert!(cert.gammas_nonneg());
        let farkas = cert.to_farkas();
        assert_eq!(farkas.constraints.len(), farkas.multipliers.len());
        let residual = check_farkas(&farkas).expect("valid INVPROP HOLD must verify");
        assert!(
            residual.is_negative(),
            "combined constant must be strictly negative, got {residual:?}"
        );
    }

    /// Wrong gamma (0.5): the `y` coefficient (`-1 + 0.5 = -0.5`) does NOT cancel,
    /// so the kernel rejects it with `UncancelledVariables`. A suboptimal multiplier
    /// can never manufacture a contradiction.
    #[test]
    fn invprop_farkas_rejects_uncancelled_gamma() {
        let cert = InvpropAugmentedCertificate {
            base_premises: vec![ge_row("y", Rat::from_int(3))],
            base_multipliers: vec![Rat::from_int(1)],
            output_rows: vec![OutputDualRow {
                row: le_row("y", Rat::from_int(1), Rat::from_int(1)),
                gamma: Rat::new(1, 2).unwrap(),
            }],
        };
        let err = check_farkas(&cert.to_farkas()).unwrap_err();
        assert!(
            matches!(err, CheckError::UncancelledVariables(_)),
            "wrong gamma must leave y uncancelled, got {err:?}"
        );
    }

    /// No infeasibility: the network only proves `y >= 0.5`, violation `y <= 1`.
    /// With gamma = 1 the constant is `1 - 0.5 = 0.5 >= 0` => NOT a contradiction:
    /// the violation region is non-empty, so the kernel does NOT establish HOLD.
    #[test]
    fn invprop_farkas_rejects_non_empty_violation() {
        let cert = InvpropAugmentedCertificate {
            base_premises: vec![ge_row("y", Rat::new(1, 2).unwrap())],
            base_multipliers: vec![Rat::from_int(1)],
            output_rows: vec![OutputDualRow {
                row: le_row("y", Rat::from_int(1), Rat::from_int(1)),
                gamma: Rat::from_int(1),
            }],
        };
        let err = check_farkas(&cert.to_farkas()).unwrap_err();
        assert!(
            matches!(err, CheckError::NotEstablished),
            "non-empty violation must not establish HOLD, got {err:?}"
        );
    }

    /// Negative gamma is rejected outright by the kernel (the `nonnegZ` gate) — a
    /// negative multiplier would flip the dualized inequality (the moat guard).
    #[test]
    fn invprop_farkas_rejects_negative_gamma() {
        let cert = InvpropAugmentedCertificate {
            base_premises: vec![ge_row("y", Rat::from_int(3))],
            base_multipliers: vec![Rat::from_int(1)],
            output_rows: vec![OutputDualRow {
                row: le_row("y", Rat::from_int(1), Rat::from_int(1)),
                gamma: Rat::from_int(-1),
            }],
        };
        assert!(!cert.gammas_nonneg());
        let err = check_farkas(&cert.to_farkas()).unwrap_err();
        assert!(
            matches!(err, CheckError::MultiplierNegative(_)),
            "negative gamma must be rejected, got {err:?}"
        );
    }
}
