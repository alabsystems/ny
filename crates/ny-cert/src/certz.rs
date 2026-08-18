// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Kernel-checkable `CertZ` emission for Clean's pinned Lean checker.
//!
//! This module is an **additive** emission path: it transcribes an
//! [`EntailmentCertificate`] into the integer-pair `CertZ` shape that Clean's
//! *proven* checker decides in the Lean kernel
//! (`Crownproof.CertCheckerZ.checkEntailmentZ` in the exact revision-pinned
//! Clean Lake dependency). It does **not** touch the
//! existing self-checker ([`crate::selfcheck`]) or any verdict/soundness path —
//! the local self-check remains the fast pre-flight; this adds a route to a
//! *Lean-kernel theorem*.
//!
//! ## Why integer pairs?
//!
//! The verified ℚ spec `Crownproof.CertChecker.checkEntailment` cannot be
//! reduced by the Lean kernel on bignum leaf certs: ℚ normalization calls
//! `Nat.gcd`, which is not GMP-accelerated and is astronomically slow on
//! 800-bit denominators. `Int`/`Nat` arithmetic, by contrast, *is* GMP-backed
//! in the kernel and reduces instantly. So Clean's `checkEntailmentZ`
//! works over **unreduced integer pairs** `(num, den)` with `den > 0`, where
//! every comparison is an integer cross-multiplication — fully kernel-reducible
//! by `decide`/`rfl`, with NO `native_decide`. `checkEntailmentZ_sound` then
//! proves: `checkEntailmentZ cz = true` implies the entailment holds for every
//! assignment satisfying the (lifted ℚ) premises.
//!
//! ## What this gives you
//!
//! Emitting an [`EntailmentCertificate`] as a `CertZ` data literal (see
//! [`entailment_to_certz_lean`]) lets `decide` run — in the spirit of Clean's
//! `Crownproof.CertRunZ_node10` module — and turn an ny verdict into a
//! **Lean-kernel theorem** via `Crownproof.CertCheckerZ.checkEntailmentZ_sound`,
//! rather than a result trusted only from the Rust checker.
//!
//! ## Precondition discharge by construction
//!
//! [`Rat`] values are always reduced with a positive denominator, so the
//! checker's `allDenPos` precondition (every denominator `> 0`) holds by
//! construction for everything this module emits. Each pair is extracted with
//! the arena's checked one-read API, so a poisoned arena is rejected instead
//! of serializing a potentially substituted value. Numerators and denominators
//! are emitted as full decimal strings, preserving arbitrary precision (the
//! whole point of the integer-pair backend).
//!
//! The two emitters agree on contents:
//!
//! * [`entailment_to_certz_json`] — a `serde_json::Value` with
//!   `{ "premises": [LinConZ…], "multipliers": [[num,den]…], "conclusion": LinConZ }`,
//!   each `LinConZ` being
//!   `{ "coeffs": [[name,[num,den]]…], "kind": "le"|"ge"|"eq", "const": [num,den] }`.
//! * [`entailment_to_certz_lean`] — a Lean `CertZ` data literal in the
//!   `CertRealZ_node10.lean` style, ready to elaborate against the pinned checker and
//!   discharge with `by decide`.

use crate::rational::{Rat, RatError};
use crate::schema::{ConstraintKind, EntailmentCertificate, LinearConstraint};

/// Why a certificate cannot be emitted as a kernel-checkable `CertZ`.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CertZError {
    /// A constraint used a *strict* relation (`<` / `>`). Clean's proven
    /// checker's `Kind` is `le | ge | eq` only — it has no strict forms — so
    /// strict constraints cannot be transcribed into a `CertZ`.
    #[error("strict relation ({0}) has no CertZ encoding (Lean Kind is le|ge|eq)")]
    StrictRelation(&'static str),
    /// The conclusion relation is `eq`. The checker requires the conclusion be
    /// `le`/`ge` (`normalizeConclusionZ` returns `none` on `eq`, so
    /// `checkEntailmentZ` would reject it).
    #[error("conclusion kind must be le|ge for CertZ (got eq)")]
    EqConclusion,
    /// The requested Lean declaration name was empty or outside the
    /// injection-safe ASCII subset. Valid names are emitted with Lean's escaped
    /// identifier syntax, so keywords are accepted safely.
    #[error("invalid Lean declaration name: {0}")]
    InvalidLeanName(String),
    /// Exact rational extraction failed (including a poisoned arena).
    #[error(transparent)]
    Rat(#[from] RatError),
}

/// Validate the conservative ASCII subset of Lean identifiers accepted by the
/// emitter. This keeps source generation injection-safe while covering ordinary
/// generated declaration names.
fn validate_lean_name(name: &str) -> Result<(), CertZError> {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return Err(CertZError::InvalidLeanName(name.to_owned()));
    };
    if !(first.is_ascii_alphabetic() || first == '_')
        || !chars.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '\'')
    {
        return Err(CertZError::InvalidLeanName(name.to_owned()));
    }

    if name == "_" {
        return Err(CertZError::InvalidLeanName(name.to_owned()));
    }
    Ok(())
}

/// Map an ny [`ConstraintKind`] to the Lean `Kind` token (`"le"`/`"ge"`/`"eq"`).
///
/// Strict relations have no encoding in the proven integer-pair checker.
fn certz_kind(kind: ConstraintKind) -> Result<&'static str, CertZError> {
    match kind {
        ConstraintKind::Le => Ok("le"),
        ConstraintKind::Ge => Ok("ge"),
        ConstraintKind::Eq => Ok("eq"),
        ConstraintKind::Lt => Err(CertZError::StrictRelation("lt")),
        ConstraintKind::Gt => Err(CertZError::StrictRelation("gt")),
    }
}

/// Emit a [`Rat`] as the integer pair `[num, den]` (decimal strings, full
/// precision). `den` is always positive, discharging `allDenPos` by
/// construction.
#[allow(clippy::vec_init_then_push)] // deliberate: avoids the vec! macro (see below)
fn qpair_json(r: Rat) -> Result<serde_json::Value, CertZError> {
    // `Vec::new()` + push (not a `vec![…]` literal): the macro's internal
    // boxed-slice `into_vec` inlines hardened alloc/arith obligations into this
    // fn; the pushes build the identical `[num, den]` pair.
    let (num, den) = r.checked_parts()?;
    let mut pair = Vec::new();
    pair.push(serde_json::Value::String(num.to_string()));
    pair.push(serde_json::Value::String(den.to_string()));
    Ok(serde_json::Value::Array(pair))
}

/// Emit one [`LinearConstraint`] as a `LinConZ` JSON object.
///
/// `coeffs` are emitted in the constraint's [`BTreeMap`](std::collections::BTreeMap)
/// order, which is deterministic (lexicographic by variable name).
#[allow(clippy::vec_init_then_push)] // deliberate: avoids the vec! macro (see qpair_json)
fn lincon_json(c: &LinearConstraint) -> Result<serde_json::Value, CertZError> {
    let mut coeffs = Vec::new();
    for (name, coeff) in &c.coefficients {
        // `Vec::new()` + push (not a `vec![…]` literal): same macro-internal
        // alloc/arith rationale as `qpair_json`; identical `[name, qpair]`.
        let mut pair = Vec::new();
        pair.push(serde_json::Value::String(name.clone()));
        pair.push(qpair_json(*coeff)?);
        coeffs.push(serde_json::Value::Array(pair));
    }
    let mut lincon = serde_json::Map::new();
    lincon.insert("coeffs".to_owned(), serde_json::Value::Array(coeffs));
    lincon.insert(
        "kind".to_owned(),
        serde_json::Value::String(certz_kind(c.kind)?.to_owned()),
    );
    lincon.insert("const".to_owned(), qpair_json(c.constant)?);
    Ok(serde_json::Value::Object(lincon))
}

/// Serialize an [`EntailmentCertificate`] into the **kernel-checkable** `CertZ`
/// JSON shape decided by Clean's proven `checkEntailmentZ`.
///
/// The shape mirrors the Lean `CertZ`/`LinConZ` structures exactly:
///
/// ```json
/// {
///   "premises":    [ { "coeffs": [[name,[num,den]], …], "kind": "le"|"ge"|"eq", "const": [num,den] }, … ],
///   "multipliers": [ [num,den], … ],
///   "conclusion":  { "coeffs": [[name,[num,den]], …], "kind": "le"|"ge", "const": [num,den] }
/// }
/// ```
///
/// All numerators/denominators are full decimal strings (arbitrary precision);
/// every denominator is positive by the [`Rat`] invariant, so the `allDenPos`
/// precondition holds by construction.
///
/// Because this is a faithful transcription, a verdict the Rust
/// [`check_entailment`](crate::selfcheck::check_entailment) accepts is one the
/// Lean kernel's `checkEntailmentZ` accepts too — and
/// `Crownproof.CertCheckerZ.checkEntailmentZ_sound` then makes the entailment a
/// Lean theorem.
///
/// # Errors
/// Returns [`CertZError::StrictRelation`] if any premise/conclusion uses `<`/`>`
/// (the proven `Kind` is `le|ge|eq` only), [`CertZError::EqConclusion`] if the
/// conclusion is `eq` (the checker requires a `le`/`ge` conclusion), or
/// [`CertZError::Rat`] when rational extraction fails.
pub fn entailment_to_certz_json(
    cert: &EntailmentCertificate,
) -> Result<serde_json::Value, CertZError> {
    if cert.conclusion.kind == ConstraintKind::Eq {
        return Err(CertZError::EqConclusion);
    }
    // behavior-identical: explicit Vec::new()+push (not `.collect()`); the
    // element count is the input-derived premise count the verifier cannot
    // bound, so a bulk `.collect()` raises an UnboundedAllocation obligation.
    // Same elements, same order, same early-return on the first `Err`.
    let mut premises = Vec::new();
    for p in cert.premises.iter() {
        premises.push(lincon_json(p)?);
    }
    let mut multipliers = Vec::new();
    for m in &cert.multipliers {
        multipliers.push(qpair_json(*m)?);
    }
    let mut certz = serde_json::Map::new();
    certz.insert("premises".to_owned(), serde_json::Value::Array(premises));
    certz.insert(
        "multipliers".to_owned(),
        serde_json::Value::Array(multipliers),
    );
    certz.insert("conclusion".to_owned(), lincon_json(&cert.conclusion)?);
    Ok(serde_json::Value::Object(certz))
}

/// Emit a [`Rat`] as the Lean `QPair` literal `(num, den)`.
fn qpair_lean(r: Rat) -> Result<String, CertZError> {
    let (num, den) = r.checked_parts()?;
    Ok(format!("({num}, {den})"))
}

/// Emit one [`LinearConstraint`] as a Lean `LinConZ` literal
/// `⟨[("name", (num, den)), …], .le, (num, den)⟩`.
fn lincon_lean(c: &LinearConstraint) -> Result<String, CertZError> {
    // Explicit loop (not `.map(..).collect::<Vec<_>>().join(", ")`): the
    // map/collect/join adapters are absent-callees for the panic-freedom checker;
    // the manual concat produces the byte-identical `", "`-separated string.
    let mut coeffs = String::new();
    for (i, (name, coeff)) in c.coefficients.iter().enumerate() {
        if i > 0 {
            coeffs.push_str(", ");
        }
        let qpair = qpair_lean(*coeff)?;
        coeffs.push_str(&format!("({name:?}, {qpair})"));
    }
    let constant = qpair_lean(c.constant)?;
    Ok(format!(
        "⟨[{coeffs}], .{}, {}⟩",
        certz_kind(c.kind)?,
        constant
    ))
}

/// Serialize an [`EntailmentCertificate`] as a self-contained Lean `CertZ` data
/// literal, in the `CertRealZ_node10.lean` style.
///
/// The emitted source defines `<name>PremisesZ`, `<name>MultsZ`, `<name>ConclZ`,
/// and a `<name> : CertZ` binding them, plus a `theorem <name>_checks` discharged
/// by `decide` (the GMP-backed `Int` cross-multiplication the kernel reduces) and
/// a `theorem <name>_safe` that applies `checkEntailmentZ_sound`. Dropping this
/// against the pinned `Crownproof.CertCheckerZ` and elaborating it makes the ny
/// verdict a **Lean-kernel theorem** (cf.
/// `Crownproof.CertCheckerZ.checkEntailmentZ_sound` and
/// `Crownproof.CertRunZ_node10` in Clean).
///
/// # Errors
/// Returns [`CertZError::StrictRelation`] for any `<`/`>` constraint, or
/// [`CertZError::EqConclusion`] if the conclusion is `eq`. Returns
/// [`CertZError::InvalidLeanName`] when `name` is not an injection-safe Lean
/// identifier, or [`CertZError::Rat`] when rational extraction fails.
pub fn entailment_to_certz_lean(
    cert: &EntailmentCertificate,
    name: &str,
) -> Result<String, CertZError> {
    validate_lean_name(name)?;
    if cert.conclusion.kind == ConstraintKind::Eq {
        return Err(CertZError::EqConclusion);
    }
    // Explicit Vec::new()+push (not `.collect()`): the premise/multiplier counts
    // are input-derived and the verifier cannot bound them, so bulk `.collect()`s
    // raise UnboundedAllocation obligations. The loops carry no bulk-alloc
    // obligation — identical elements, order, and error short-circuit (`?` on the
    // first `lincon_lean` error, exactly as `collect::<Result<Vec<_>, _>>()?`).
    let mut premise_lines: Vec<String> = Vec::new();
    for p in &cert.premises {
        premise_lines.push(format!("  {},", lincon_lean(p)?));
    }
    // Explicit concat (not `.join("\n")`): identical newline-separated string;
    // clears the join absent-callee.
    let mut premises = String::new();
    for (i, line) in premise_lines.iter().enumerate() {
        if i > 0 {
            premises.push('\n');
        }
        premises.push_str(line);
    }
    let mut mult_strs: Vec<String> = Vec::new();
    for m in &cert.multipliers {
        mult_strs.push(qpair_lean(*m)?);
    }
    // Explicit concat (not `.join(", ")`): identical `", "`-separated string;
    // clears the join absent-callee.
    let mut mults = String::new();
    for (i, m) in mult_strs.iter().enumerate() {
        if i > 0 {
            mults.push_str(", ");
        }
        mults.push_str(m);
    }
    let concl = lincon_lean(&cert.conclusion)?;
    // Escape every complete generated identifier. Lean's keyword set evolves,
    // and concatenating an otherwise-valid caller name into raw source would
    // make names such as `by` or `section` produce invalid code. The ASCII
    // validator above excludes the closing guillemet, so these escapes cannot
    // be broken out of.
    let premises_name = format!("«{name}PremisesZ»");
    let mults_name = format!("«{name}MultsZ»");
    let concl_name = format!("«{name}ConclZ»");
    let cert_name = format!("«{name}»");
    let checks_name = format!("«{name}_checks»");
    let safe_name = format!("«{name}_safe»");
    Ok(format!(
        "import Crownproof.CertCheckerZ\n\
         set_option maxHeartbeats 10000000\n\
         set_option maxRecDepth 10000000\n\
         namespace Crownproof.CertCheckerZ\n\
         open Crownproof\n\
         open Crownproof.CertChecker\n\n\
         def {premises_name} : List LinConZ := [\n{premises}\n]\n\n\
         def {mults_name} : List QPair := [{mults}]\n\n\
         def {concl_name} : LinConZ := {concl}\n\n\
         def {cert_name} : CertZ :=\n  \
         {{ premises := {premises_name}, multipliers := {mults_name}, conclusion := {concl_name} }}\n\n\
         theorem {checks_name} : checkEntailmentZ {cert_name} = true := by decide\n\n\
         theorem {safe_name} :\n    \
         ∀ σ : Assignment,\n      \
         (∀ lc ∈ (liftCert {cert_name}).premises, lc.satisfies σ) →\n      \
         (liftCert {cert_name}).conclusion.satisfies σ :=\n  \
         checkEntailmentZ_sound {cert_name} {checks_name}\n\n\
         end Crownproof.CertCheckerZ\n"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crown::Relu1Problem;
    use crate::selfcheck::check_entailment;

    struct PoisonReset;

    impl Drop for PoisonReset {
        fn drop(&mut self) {
            crate::rational::set_poisoned_for_test(false);
        }
    }

    /// Build the small ReLU entailment certificate from the crate doc example.
    fn relu_cert() -> EntailmentCertificate {
        let r = |n: i128, d: i128| Rat::new(n, d).unwrap();
        // z0 = x0 + x1, z1 = x0 − x1; y = a0 − a1 + 5/2; box [−1,1]².
        let problem = Relu1Problem {
            w1: vec![vec![r(1, 1), r(1, 1)], vec![r(1, 1), r(-1, 1)]],
            b1: vec![Rat::ZERO, Rat::ZERO],
            w2: vec![r(1, 1), r(-1, 1)],
            b2: r(5, 2),
            input_lower: vec![r(-1, 1), r(-1, 1)],
            input_upper: vec![r(1, 1), r(1, 1)],
            alpha: Some(vec![r(1, 2), r(1, 2)]),
        };
        problem.certify(Rat::ZERO).unwrap().entailment
    }

    /// Helper: read a `QPair` `[num, den]` array's denominator string.
    fn den_str(qpair: &serde_json::Value) -> &str {
        qpair.as_array().unwrap()[1].as_str().unwrap()
    }

    #[test]
    fn certz_json_is_faithful_and_kernel_ready() -> Result<(), String> {
        let cert = relu_cert();

        // (c) Pre-flight: the existing Rust self-check ACCEPTS the original cert.
        // If this passes and the CertZ is a faithful transcription, the kernel's
        // checkEntailmentZ will accept it too. A rejection is an `Err` and FAILS
        // the test (fail-closed) — never a silently skipped pre-flight.
        check_entailment(&cert).map_err(|e| format!("self-check must accept the cert: {e}"))?;

        let value = entailment_to_certz_json(&cert).map_err(|e| format!("emit CertZ json: {e}"))?;

        // The CertZ has the three top-level fields the Lean `CertZ` expects.
        let premises = value["premises"].as_array().ok_or("premises array")?;
        let multipliers = value["multipliers"].as_array().ok_or("multipliers array")?;
        let conclusion = &value["conclusion"];
        assert!(!premises.is_empty(), "expected at least one premise");

        // (b) premises.len == multipliers.len (the checker's length gate).
        assert_eq!(
            premises.len(),
            multipliers.len(),
            "premises and multipliers must be equinumerous"
        );

        // (a) Every denominator string is positive (allDenPos by construction).
        // A non-integer denominator is an `Err` that fails the test (fail-closed),
        // not a skipped positivity check.
        let parse_pos = |s: &str| -> Result<(), String> {
            let d: i128 = s
                .parse()
                .map_err(|e| format!("denominator parses as integer: {e}"))?;
            assert!(d > 0, "denominator {s} must be positive");
            Ok(())
        };
        for premise in premises {
            let pk = premise["kind"].as_str().unwrap();
            assert!(
                pk == "le" || pk == "ge" || pk == "eq",
                "premise kind {pk} must be le|ge|eq"
            );
            parse_pos(den_str(&premise["const"]))?;
            for coeff in premise["coeffs"].as_array().unwrap() {
                let pair = coeff.as_array().unwrap();
                // Each coeff entry is [name, [num, den]].
                assert!(pair[0].is_string(), "coeff name must be a string");
                parse_pos(den_str(&pair[1]))?;
            }
        }
        for m in multipliers {
            parse_pos(den_str(m))?;
        }
        parse_pos(den_str(&conclusion["const"]))?;
        for coeff in conclusion["coeffs"].as_array().unwrap() {
            parse_pos(den_str(&coeff.as_array().unwrap()[1]))?;
        }

        // Conclusion is le/ge (never eq), matching normalizeConclusionZ.
        let ck = conclusion["kind"].as_str().unwrap();
        assert!(
            ck == "le" || ck == "ge",
            "conclusion kind {ck} must be le|ge"
        );
        Ok(())
    }

    #[test]
    fn certz_lean_literal_has_expected_skeleton() -> Result<(), String> {
        let cert = relu_cert();
        let src = entailment_to_certz_lean(&cert, "nyLeaf")
            .map_err(|e| format!("emit CertZ lean: {e}"))?;
        // Mirrors the CertRunZ_node10.lean structure.
        assert!(src.contains("def «nyLeafPremisesZ» : List LinConZ := ["));
        assert!(src.contains("def «nyLeafMultsZ» : List QPair := ["));
        assert!(src.contains("def «nyLeafConclZ» : LinConZ := "));
        assert!(src.contains("def «nyLeaf» : CertZ :="));
        assert!(src.contains("open Crownproof.CertChecker"));
        assert!(
            src.contains("theorem «nyLeaf_checks» : checkEntailmentZ «nyLeaf» = true := by decide")
        );
        assert!(src.contains("checkEntailmentZ_sound «nyLeaf» «nyLeaf_checks»"));
        // The conclusion is a `ge` on the single output variable `y`.
        assert!(src.contains("def «nyLeafConclZ» : LinConZ := ⟨[(\"y\", (1, 1))], .ge,"));
        Ok(())
    }

    #[test]
    #[cfg(feature = "external-lean")]
    fn generated_certz_lean_elaborates_with_pinned_checker() -> Result<(), String> {
        use std::io::Write as _;
        use std::process::Command;

        let cert = relu_cert();
        let source = entailment_to_certz_lean(&cert, "nyGeneratedSmoke")
            .map_err(|e| format!("emit CertZ Lean: {e}"))?;
        let lean_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("proofs")
            .join("lean");

        let build = Command::new("lake")
            .args(["build", "Crownproof.CertCheckerZ"])
            .current_dir(&lean_root)
            .output()
            .map_err(|e| format!("run pinned Lake build: {e}"))?;
        if !build.status.success() {
            return Err(format!(
                "pinned CertCheckerZ build failed:\n{}{}",
                String::from_utf8_lossy(&build.stdout),
                String::from_utf8_lossy(&build.stderr)
            ));
        }

        let mut file = tempfile::Builder::new()
            .prefix("ny-generated-certz-")
            .suffix(".lean")
            .tempfile()
            .map_err(|e| format!("create generated Lean tempfile: {e}"))?;
        file.write_all(source.as_bytes())
            .map_err(|e| format!("write generated Lean tempfile: {e}"))?;
        file.flush()
            .map_err(|e| format!("flush generated Lean tempfile: {e}"))?;

        let elaboration = Command::new("lake")
            .args(["env", "lean"])
            .arg(file.path())
            .current_dir(&lean_root)
            .output()
            .map_err(|e| format!("run Lean elaborator: {e}"))?;
        if !elaboration.status.success() {
            return Err(format!(
                "generated CertZ Lean failed to elaborate:\n{}{}",
                String::from_utf8_lossy(&elaboration.stdout),
                String::from_utf8_lossy(&elaboration.stderr)
            ));
        }
        Ok(())
    }

    #[test]
    fn certz_lean_rejects_unsafe_or_reserved_declaration_names() {
        let cert = relu_cert();
        for name in ["", "_", "2bad", "bad-name", "bad\naxiom forged : False"] {
            assert!(
                matches!(
                    entailment_to_certz_lean(&cert, name),
                    Err(CertZError::InvalidLeanName(_))
                ),
                "unsafe name was accepted: {name:?}"
            );
        }
        entailment_to_certz_lean(&cert, "nyLeaf_2'")
            .expect("ordinary generated Lean identifier is accepted");
        let keyword = entailment_to_certz_lean(&cert, "by")
            .expect("Lean keywords are safe when emitted as escaped identifiers");
        assert!(keyword.contains("def «by» : CertZ :="));
    }

    #[test]
    fn strict_relation_is_rejected() {
        use crate::schema::LinearConstraint;
        let r = |n: i128, d: i128| Rat::new(n, d).unwrap();
        let cert = EntailmentCertificate {
            premises: vec![LinearConstraint::with_kind(
                ConstraintKind::Lt,
                &[("x", r(1, 1))],
                r(0, 1),
            )],
            multipliers: vec![Rat::ONE],
            conclusion: LinearConstraint::with_kind(ConstraintKind::Ge, &[("x", r(1, 1))], r(0, 1)),
        };
        assert_eq!(
            entailment_to_certz_json(&cert),
            Err(CertZError::StrictRelation("lt"))
        );
    }

    #[test]
    fn eq_conclusion_is_rejected() {
        use crate::schema::LinearConstraint;
        let r = |n: i128, d: i128| Rat::new(n, d).unwrap();
        let cert = EntailmentCertificate {
            premises: vec![LinearConstraint::with_kind(
                ConstraintKind::Ge,
                &[("x", r(1, 1))],
                r(0, 1),
            )],
            multipliers: vec![Rat::ONE],
            conclusion: LinearConstraint::with_kind(ConstraintKind::Eq, &[("x", r(1, 1))], r(0, 1)),
        };
        assert_eq!(
            entailment_to_certz_json(&cert),
            Err(CertZError::EqConclusion)
        );
    }

    #[test]
    fn certz_emitters_refuse_a_poisoned_arena() {
        let cert = relu_cert();
        crate::rational::set_poisoned_for_test(true);
        let _reset = PoisonReset;

        assert_eq!(
            entailment_to_certz_json(&cert),
            Err(CertZError::Rat(RatError::Poisoned))
        );
        assert_eq!(
            entailment_to_certz_lean(&cert, "nyPoison"),
            Err(CertZError::Rat(RatError::Poisoned))
        );
    }
}
