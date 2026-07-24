// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Certificate schema mirroring Clean's external-certificate JSON contract.
//!
//! The canonical payloads are documented in Clean's
//! `docs/EXTERNAL_CERTIFICATES.md` and implemented in
//! `clean-elab/src/cert/external/verify.rs`. We emit string-encoded rationals
//! (`"n"` / `"n/d"`), which Clean's `ExternalRational` deserializer accepts.

use crate::rational::{Rat, RatError};
use std::collections::BTreeMap;

/// Constraint relation, serialized in Clean's lowercase form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintKind {
    /// `≤`
    Le,
    /// `<`
    Lt,
    /// `=`
    Eq,
    /// `≥`
    Ge,
    /// `>`
    Gt,
}

impl ConstraintKind {
    const fn as_str(self) -> &'static str {
        match self {
            ConstraintKind::Le => "le",
            ConstraintKind::Lt => "lt",
            ConstraintKind::Eq => "eq",
            ConstraintKind::Ge => "ge",
            ConstraintKind::Gt => "gt",
        }
    }
}

/// A linear constraint `Σ cᵢ·xᵢ  ⋈  constant` over named variables.
#[derive(Debug, Clone)]
pub struct LinearConstraint {
    /// Relation symbol.
    pub kind: ConstraintKind,
    /// Variable coefficients keyed by name (sorted for determinism).
    pub coefficients: BTreeMap<String, Rat>,
    /// Right-hand-side constant.
    pub constant: Rat,
}

impl LinearConstraint {
    /// Build a `≤` constraint from `(name, coeff)` pairs and a constant.
    #[must_use]
    pub fn le(terms: &[(&str, Rat)], constant: Rat) -> Self {
        Self::with_kind(ConstraintKind::Le, terms, constant)
    }

    /// Build a constraint with an explicit relation.
    #[must_use]
    pub fn with_kind(kind: ConstraintKind, terms: &[(&str, Rat)], constant: Rat) -> Self {
        let mut coefficients = BTreeMap::new();
        for (name, coeff) in terms {
            if !coeff.is_zero() {
                coefficients.insert((*name).to_string(), *coeff);
            }
        }
        LinearConstraint {
            kind,
            coefficients,
            constant,
        }
    }

    fn to_json(&self) -> Result<serde_json::Value, RatError> {
        let mut coeffs = serde_json::Map::new();
        for (name, coeff) in &self.coefficients {
            coeffs.insert(
                name.clone(),
                serde_json::Value::String(coeff.to_clean_string()?),
            );
        }
        let mut constraint = serde_json::Map::new();
        constraint.insert(
            "type".to_owned(),
            serde_json::Value::String("linear_constraint".to_owned()),
        );
        constraint.insert(
            "kind".to_owned(),
            serde_json::Value::String(self.kind.as_str().to_owned()),
        );
        constraint.insert("coefficients".to_owned(), serde_json::Value::Object(coeffs));
        constraint.insert(
            "constant".to_owned(),
            serde_json::Value::String(self.constant.to_clean_string()?),
        );
        Ok(serde_json::Value::Object(constraint))
    }
}

/// An entailment certificate: a non-negative combination of `premises` derives
/// a bound at least as strong as `conclusion`.
#[derive(Debug, Clone)]
pub struct EntailmentCertificate {
    /// Premise constraints (the relaxed network + input box).
    pub premises: Vec<LinearConstraint>,
    /// Non-negative multipliers, one per premise (the CROWN dual / Farkas
    /// multipliers).
    pub multipliers: Vec<Rat>,
    /// The conclusion entailed by the weighted premises.
    pub conclusion: LinearConstraint,
}

/// A Farkas infeasibility certificate: a non-negative combination of
/// `constraints` collapses to a contradiction.
#[derive(Debug, Clone)]
pub struct FarkasCertificate {
    /// Constraint system asserted infeasible (e.g. the relaxed network plus the
    /// negated safety property defining the unsafe region).
    pub constraints: Vec<LinearConstraint>,
    /// Non-negative multipliers, one per constraint.
    pub multipliers: Vec<Rat>,
}

/// Filter `(rows, multipliers)` down to the rows whose multiplier is nonzero.
///
/// Shared by the two `minimized` passes. Dropping a zero-multiplier row cannot
/// change the checked combination: `selfcheck`'s `add_scaled` early-returns on
/// a zero factor, so a dead row contributes neither coefficients, nor constant,
/// nor strictness. Any variable that only occurred in dead rows vanishes with
/// them (variables live nowhere else in the certificate).
///
/// Explicit indexed loop (not `.zip(..).filter(..).unzip()`): keeps the row
/// gather in verified code (no absent-adapter closure rows), identical
/// elements and order.
fn nonzero_rows(
    rows: &[LinearConstraint],
    multipliers: &[Rat],
) -> (Vec<LinearConstraint>, Vec<Rat>) {
    let mut kept_rows: Vec<LinearConstraint> = Vec::new();
    let mut kept_mults: Vec<Rat> = Vec::new();
    for (i, m) in multipliers.iter().enumerate() {
        if m.is_zero() {
            continue;
        }
        // `i < multipliers.len() == rows.len()` (callers guard arity), so the
        // skip arm is unreachable — total read, identical result.
        if let Some(row) = rows.get(i) {
            kept_rows.push(row.clone());
            kept_mults.push(*m);
        }
    }
    (kept_rows, kept_mults)
}

impl EntailmentCertificate {
    /// A minimized copy: every dead premise row (multiplier exactly zero) is
    /// dropped, along with any variables that only occurred in dead rows.
    ///
    /// Semantics-preserving by construction — a zero multiplier contributes
    /// nothing to the checked combination — and fail-closed on top: the
    /// minimized certificate is returned only if
    /// [`crate::selfcheck::check_entailment`] accepts BOTH the original and the
    /// minimized form with the identical `(derived, claimed)` bounds; otherwise
    /// the original is returned unchanged (a cert this pass cannot prove
    /// equivalent — including one that never checked — is never rewritten).
    #[must_use]
    pub fn minimized(&self) -> Self {
        // Fail closed on arity mismatch: such a cert never checks; leave it
        // for `check_entailment` to reject in its original form.
        if self.premises.len() != self.multipliers.len() {
            return self.clone();
        }
        let (premises, multipliers) = nonzero_rows(&self.premises, &self.multipliers);
        if premises.len() == self.premises.len() {
            // Nothing to drop; skip the double re-check.
            return self.clone();
        }
        let candidate = EntailmentCertificate {
            premises,
            multipliers,
            conclusion: self.conclusion.clone(),
        };
        let original_check = crate::selfcheck::check_entailment(self);
        let candidate_check = crate::selfcheck::check_entailment(&candidate);
        match (original_check, candidate_check) {
            (Ok(a), Ok(b)) if a == b => candidate,
            _ => self.clone(),
        }
    }
}

impl FarkasCertificate {
    /// A minimized copy: every dead constraint row (multiplier exactly zero)
    /// is dropped, along with any variables that only occurred in dead rows.
    ///
    /// Semantics-preserving by construction — a zero multiplier contributes
    /// nothing to the checked combination — and fail-closed on top: the
    /// minimized certificate is returned only if
    /// [`crate::selfcheck::check_farkas`] accepts BOTH the original and the
    /// minimized form with the identical residual constant; otherwise the
    /// original is returned unchanged (a cert this pass cannot prove
    /// equivalent — including one that never checked — is never rewritten).
    #[must_use]
    pub fn minimized(&self) -> Self {
        if self.constraints.len() != self.multipliers.len() {
            return self.clone();
        }
        let (constraints, multipliers) = nonzero_rows(&self.constraints, &self.multipliers);
        if constraints.len() == self.constraints.len() {
            return self.clone();
        }
        let candidate = FarkasCertificate {
            constraints,
            multipliers,
        };
        let original_check = crate::selfcheck::check_farkas(self);
        let candidate_check = crate::selfcheck::check_farkas(&candidate);
        match (original_check, candidate_check) {
            (Ok(a), Ok(b)) if a == b => candidate,
            _ => self.clone(),
        }
    }
}

/// Serialize an entailment certificate to Clean's canonical JSON.
///
/// # Errors
/// Infallible in practice: rationals are emitted as full bignum `"n"`/`"n/d"`
/// strings (the former `i64` emission wall is gone — see `rational.rs`); the
/// `Result` is kept for source compatibility with the i64-era callers.
pub fn entailment_to_json(cert: &EntailmentCertificate) -> Result<serde_json::Value, RatError> {
    // behavior-identical: explicit Vec::new()+push (not `.collect()`); the
    // premise count is input-derived and unbounded to the verifier, so a bulk
    // `.collect()` raises an UnboundedAllocation obligation. Same elements,
    // same order, same early-return on the first `Err`.
    let mut premises = Vec::new();
    for p in cert.premises.iter() {
        premises.push(p.to_json()?);
    }
    // Explicit loop (not `.map(..).collect::<Result<_,_>>()`): a direct
    // `Value::String(to_clean_string()?)` push avoids the map-closure and
    // `Result::map` absent-callee rows plus the collect bulk-alloc obligation
    // — identical elements, order, and first-`Err` early return.
    let mut multipliers = Vec::new();
    for m in cert.multipliers.iter() {
        multipliers.push(serde_json::Value::String(m.to_clean_string()?));
    }
    let multipliers = multipliers;
    let mut certificate = serde_json::Map::new();
    certificate.insert(
        "type".to_owned(),
        serde_json::Value::String("entailment_certificate".to_owned()),
    );
    certificate.insert(
        "version".to_owned(),
        serde_json::Value::String("1.0".to_owned()),
    );
    certificate.insert("premises".to_owned(), serde_json::Value::Array(premises));
    certificate.insert(
        "multipliers".to_owned(),
        serde_json::Value::Array(multipliers),
    );
    certificate.insert("conclusion".to_owned(), cert.conclusion.to_json()?);
    Ok(serde_json::Value::Object(certificate))
}

/// Serialize one entailment certificate as a bare chain *step* (Clean's
/// `ExternalEntailmentCert` shape: `version`/`premises`/`multipliers`/
/// `conclusion`, with no outer `type` tag — chain steps are not the tagged
/// `ExternalCertificate` enum).
fn entailment_step_json(cert: &EntailmentCertificate) -> Result<serde_json::Value, RatError> {
    // behavior-identical: explicit Vec::new()+push (not `.collect()`); the
    // premise count is input-derived and unbounded to the verifier, so a bulk
    // `.collect()` raises an UnboundedAllocation obligation. Same elements,
    // same order, same early-return on the first `Err`.
    let mut premises = Vec::new();
    for p in cert.premises.iter() {
        premises.push(p.to_json()?);
    }
    // Explicit loop (not `.map(..).collect::<Result<_,_>>()`): a direct
    // `Value::String(to_clean_string()?)` push avoids the map-closure and
    // `Result::map` absent-callee rows plus the collect bulk-alloc obligation
    // — identical elements, order, and first-`Err` early return.
    let mut multipliers = Vec::new();
    for m in cert.multipliers.iter() {
        multipliers.push(serde_json::Value::String(m.to_clean_string()?));
    }
    let multipliers = multipliers;
    let mut step = serde_json::Map::new();
    step.insert(
        "version".to_owned(),
        serde_json::Value::String("1.0".to_owned()),
    );
    step.insert("premises".to_owned(), serde_json::Value::Array(premises));
    step.insert(
        "multipliers".to_owned(),
        serde_json::Value::Array(multipliers),
    );
    step.insert("conclusion".to_owned(), cert.conclusion.to_json()?);
    Ok(serde_json::Value::Object(step))
}

/// Serialize a composed entailment *chain* to NY's legacy linkage JSON
/// (`{ "version": "1.0", "steps": [ … ] }`). Clean did not adopt this
/// schema; its current composition API merges and re-verifies certificates.
///
/// The chain is sound only if each step is individually valid *and* every step
/// after the first restates the previous step's conclusion as one of its
/// premises (the cut rule). NY emits and checks the linkage locally.
///
/// # Errors
/// Infallible in practice (full bignum emission; the `i64` wall is gone); the
/// `Result` is kept for source compatibility with the i64-era callers.
pub fn chain_to_json(steps: &[EntailmentCertificate]) -> Result<serde_json::Value, RatError> {
    // behavior-identical: explicit Vec::new()+push (not `.collect()`); the step
    // count is input-derived and unbounded to the verifier, so a bulk
    // `.collect()` raises an UnboundedAllocation obligation. Same elements,
    // same order, same early-return on the first `Err`.
    let mut steps_json = Vec::new();
    for s in steps {
        steps_json.push(entailment_step_json(s)?);
    }
    let steps = steps_json;
    let mut chain = serde_json::Map::new();
    chain.insert(
        "version".to_owned(),
        serde_json::Value::String("1.0".to_owned()),
    );
    chain.insert("steps".to_owned(), serde_json::Value::Array(steps));
    Ok(serde_json::Value::Object(chain))
}

/// Serialize a Farkas certificate to Clean's canonical JSON.
///
/// # Errors
/// Infallible in practice (full bignum emission; the `i64` wall is gone); the
/// `Result` is kept for source compatibility with the i64-era callers.
pub fn farkas_to_json(cert: &FarkasCertificate) -> Result<serde_json::Value, RatError> {
    // behavior-identical: explicit Vec::new()+push (not `.collect()`); the
    // constraint count is input-derived and unbounded to the verifier, so a bulk
    // `.collect()` raises an UnboundedAllocation obligation. Same elements,
    // same order, same early-return on the first `Err`.
    let mut constraints = Vec::new();
    for c in cert.constraints.iter() {
        constraints.push(c.to_json()?);
    }
    // Explicit loop (not `.map(..).collect::<Result<_,_>>()`): a direct
    // `Value::String(to_clean_string()?)` push avoids the map-closure and
    // `Result::map` absent-callee rows plus the collect bulk-alloc obligation
    // — identical elements, order, and first-`Err` early return.
    let mut multipliers = Vec::new();
    for m in cert.multipliers.iter() {
        multipliers.push(serde_json::Value::String(m.to_clean_string()?));
    }
    let multipliers = multipliers;
    let mut certificate = serde_json::Map::new();
    certificate.insert(
        "type".to_owned(),
        serde_json::Value::String("farkas_certificate".to_owned()),
    );
    certificate.insert(
        "version".to_owned(),
        serde_json::Value::String("1.0".to_owned()),
    );
    certificate.insert(
        "constraints".to_owned(),
        serde_json::Value::Array(constraints),
    );
    certificate.insert(
        "multipliers".to_owned(),
        serde_json::Value::Array(multipliers),
    );
    certificate.insert(
        "conclusion".to_owned(),
        serde_json::Value::String("contradiction".to_owned()),
    );
    Ok(serde_json::Value::Object(certificate))
}

#[cfg(test)]
mod minimization_tests {
    use super::*;
    use crate::selfcheck::{check_entailment, check_farkas, CheckError};

    fn r(numerator: i128, denominator: i128) -> Rat {
        Rat::new(numerator, denominator).expect("valid test rational")
    }

    #[test]
    fn minimized_valid_certificates_drop_only_zero_rows_and_keep_exact_checks() {
        // The zero-weight strict row must contribute neither coefficients nor
        // strictness. The live strict premise still proves the non-strict
        // conclusion at the exact same bound.
        let entailment = EntailmentCertificate {
            premises: vec![
                LinearConstraint::with_kind(ConstraintKind::Lt, &[("dead", Rat::ONE)], Rat::ZERO),
                LinearConstraint::with_kind(ConstraintKind::Gt, &[("x", Rat::ONE)], Rat::ONE),
            ],
            multipliers: vec![Rat::ZERO, Rat::ONE],
            conclusion: LinearConstraint::with_kind(
                ConstraintKind::Ge,
                &[("x", Rat::ONE)],
                Rat::ONE,
            ),
        };
        let entailment_check = check_entailment(&entailment).expect("original entailment checks");
        let minimized_entailment = entailment.minimized();
        assert_eq!(minimized_entailment.premises.len(), 1);
        assert_eq!(minimized_entailment.multipliers, vec![Rat::ONE]);
        assert_eq!(minimized_entailment.premises[0].kind, ConstraintKind::Gt);
        assert!(!minimized_entailment.premises[0]
            .coefficients
            .contains_key("dead"));
        assert_eq!(
            check_entailment(&minimized_entailment),
            Ok(entailment_check),
            "minimization must preserve strict entailment semantics exactly"
        );

        // 1/2 * (2x >= 2) + 1 * (x < 1) gives the strict contradiction
        // 0 < 0. The dead equality is deliberately unsupported by Alethe but
        // is inert here because its multiplier is zero.
        let farkas = FarkasCertificate {
            constraints: vec![
                LinearConstraint::with_kind(ConstraintKind::Eq, &[("dead", Rat::ONE)], Rat::ZERO),
                LinearConstraint::with_kind(ConstraintKind::Ge, &[("x", r(2, 1))], r(2, 1)),
                LinearConstraint::with_kind(ConstraintKind::Lt, &[("x", Rat::ONE)], Rat::ONE),
            ],
            multipliers: vec![Rat::ZERO, r(1, 2), Rat::ONE],
        };
        let farkas_check = check_farkas(&farkas).expect("original Farkas cert checks");
        assert_eq!(farkas_check, Rat::ZERO, "strict contradiction is exact");
        let minimized_farkas = farkas.minimized();
        assert_eq!(minimized_farkas.constraints.len(), 2);
        assert_eq!(minimized_farkas.multipliers, vec![r(1, 2), Rat::ONE]);
        assert_eq!(minimized_farkas.constraints[0].kind, ConstraintKind::Ge);
        assert_eq!(minimized_farkas.constraints[1].kind, ConstraintKind::Lt);
        assert!(minimized_farkas
            .constraints
            .iter()
            .all(|row| !row.coefficients.contains_key("dead")));
        assert_eq!(
            check_farkas(&minimized_farkas),
            Ok(farkas_check),
            "minimization must preserve fractional multipliers and strict residual exactly"
        );
    }

    #[test]
    fn minimized_malformed_or_rejected_certificates_remain_byte_for_byte_json_equivalent() {
        let malformed_entailment = EntailmentCertificate {
            premises: vec![
                LinearConstraint::le(&[("dead", Rat::ONE)], Rat::ZERO),
                LinearConstraint::with_kind(ConstraintKind::Ge, &[("x", Rat::ONE)], Rat::ONE),
            ],
            multipliers: vec![Rat::ZERO],
            conclusion: LinearConstraint::with_kind(
                ConstraintKind::Ge,
                &[("x", Rat::ONE)],
                Rat::ONE,
            ),
        };
        assert_eq!(
            check_entailment(&malformed_entailment),
            Err(CheckError::LengthMismatch(2, 1))
        );
        assert_eq!(
            entailment_to_json(&malformed_entailment.minimized()).unwrap(),
            entailment_to_json(&malformed_entailment).unwrap(),
            "arity-invalid entailment must not be rewritten"
        );

        let rejected_entailment = EntailmentCertificate {
            premises: vec![
                LinearConstraint::le(&[("dead", Rat::ONE)], Rat::ZERO),
                LinearConstraint::with_kind(ConstraintKind::Ge, &[("x", Rat::ONE)], Rat::ONE),
            ],
            multipliers: vec![Rat::ZERO, Rat::ONE],
            conclusion: LinearConstraint::with_kind(
                ConstraintKind::Ge,
                &[("x", Rat::ONE)],
                r(2, 1),
            ),
        };
        assert_eq!(
            check_entailment(&rejected_entailment),
            Err(CheckError::NotEstablished)
        );
        assert_eq!(
            entailment_to_json(&rejected_entailment.minimized()).unwrap(),
            entailment_to_json(&rejected_entailment).unwrap(),
            "semantically rejected entailment must retain its dead row"
        );

        let rejected_farkas = FarkasCertificate {
            constraints: vec![
                LinearConstraint::with_kind(ConstraintKind::Eq, &[("dead", Rat::ONE)], Rat::ZERO),
                LinearConstraint::le(&[("x", Rat::ONE)], Rat::ONE),
            ],
            multipliers: vec![Rat::ZERO, Rat::ONE],
        };
        assert_eq!(
            check_farkas(&rejected_farkas),
            Err(CheckError::UncancelledVariables(vec!["x".to_string()]))
        );
        assert_eq!(
            farkas_to_json(&rejected_farkas.minimized()).unwrap(),
            farkas_to_json(&rejected_farkas).unwrap(),
            "rejected Farkas certificate must retain its original structure"
        );

        let malformed_farkas = FarkasCertificate {
            constraints: vec![
                LinearConstraint::le(&[("dead", Rat::ONE)], Rat::ZERO),
                LinearConstraint::le(&[("x", Rat::ONE)], Rat::ONE),
            ],
            multipliers: vec![Rat::ZERO],
        };
        assert_eq!(
            check_farkas(&malformed_farkas),
            Err(CheckError::LengthMismatch(2, 1))
        );
        assert_eq!(
            farkas_to_json(&malformed_farkas.minimized()).unwrap(),
            farkas_to_json(&malformed_farkas).unwrap(),
            "arity-invalid Farkas certificate must not be rewritten"
        );
    }
}
