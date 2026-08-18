// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Dead-row minimization (the Marabou proof-minimization analog, audited
//! cheap): certificates carrying premise rows with multiplier exactly zero
//! minimize to the same verdict — `check_entailment` / `check_farkas` return
//! the IDENTICAL bounds/residual on the minimized cert — and the Alethe
//! emission of the dead-row cert is smaller. Those offline assertions run in
//! the default suite. The live Carcara validation is an explicit
//! `external-carcara` conformance lane and hard-requires a reachable checker.

#[cfg(feature = "external-carcara")]
use ny_cert::AletheEmission;
use ny_cert::{
    check_entailment, check_farkas, entailment_to_alethe, farkas_to_alethe, ConstraintKind,
    EntailmentCertificate, FarkasCertificate, LinearConstraint, Rat,
};
#[cfg(feature = "external-carcara")]
use std::path::{Path, PathBuf};
#[cfg(feature = "external-carcara")]
use std::process::Command;

fn r(n: i128, d: i128) -> Rat {
    Rat::new(n, d).unwrap()
}

/// Select `$NY_CARCARA` when explicitly set; otherwise let `Command` resolve
/// `carcara` on `PATH`.
#[cfg(feature = "external-carcara")]
fn selected_carcara() -> PathBuf {
    std::env::var_os("NY_CARCARA")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("carcara"))
}

#[cfg(feature = "external-carcara")]
fn require_carcara() -> PathBuf {
    let carcara = selected_carcara();
    let output = Command::new(&carcara)
        .arg("--version")
        .output()
        .unwrap_or_else(|error| {
            panic!(
                "live Carcara test requires a checker at {} \
                 (set NY_CARCARA or install carcara on PATH): {error}",
                carcara.display()
            )
        });
    assert!(
        output.status.success(),
        "Carcara preflight failed at {} (status={}):\nstdout:\n{}\nstderr:\n{}",
        carcara.display(),
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    carcara
}

/// Write the pair to a scratch dir and require carcara to report `valid`.
#[cfg(feature = "external-carcara")]
fn assert_carcara_valid(carcara: &Path, em: &AletheEmission, name: &str) {
    let dir = tempfile::tempdir().expect("scratch dir");
    let problem = dir.path().join(format!("{name}.smt2"));
    let proof = dir.path().join(format!("{name}.alethe"));
    std::fs::write(&problem, &em.problem).expect("write problem");
    std::fs::write(&proof, &em.proof).expect("write proof");
    let out = Command::new(carcara)
        .arg("check")
        .arg(&proof)
        .arg(&problem)
        .output()
        .expect("run carcara");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success() && stdout.trim() == "valid",
        "carcara rejected {name}:\nstdout: {stdout}\nstderr: {stderr}\nproblem:\n{}\nproof:\n{}",
        em.problem,
        em.proof
    );
}

/// A Farkas cert with two live rows and two dead (multiplier-zero) rows, one
/// of which is the ONLY mention of variable `wdead` (so `wdead` vanishes entirely).
fn farkas_with_dead_rows() -> FarkasCertificate {
    FarkasCertificate {
        constraints: vec![
            LinearConstraint::with_kind(ConstraintKind::Ge, &[("y", Rat::ONE)], r(3, 1)),
            LinearConstraint::with_kind(ConstraintKind::Le, &[("wdead", Rat::ONE)], r(5, 1)),
            LinearConstraint::with_kind(ConstraintKind::Le, &[("y", Rat::ONE)], r(1, 1)),
            LinearConstraint::with_kind(ConstraintKind::Ge, &[("y", r(2, 1))], r(-7, 1)),
        ],
        multipliers: vec![Rat::ONE, Rat::ZERO, Rat::ONE, Rat::ZERO],
    }
}

/// An entailment cert with one live premise and two dead rows (one mentioning
/// an otherwise-absent variable `udead`).
fn entailment_with_dead_rows() -> EntailmentCertificate {
    EntailmentCertificate {
        premises: vec![
            LinearConstraint::with_kind(ConstraintKind::Le, &[("udead", Rat::ONE)], r(9, 1)),
            LinearConstraint::with_kind(ConstraintKind::Ge, &[("x", Rat::ONE)], r(1, 1)),
            LinearConstraint::with_kind(ConstraintKind::Ge, &[("x", r(3, 1))], r(-4, 1)),
        ],
        multipliers: vec![Rat::ZERO, r(2, 1), Rat::ZERO],
        conclusion: LinearConstraint::with_kind(ConstraintKind::Ge, &[("x", r(2, 1))], r(2, 1)),
    }
}

#[test]
fn farkas_minimization_round_trips_with_identical_residual() {
    let cert = farkas_with_dead_rows();
    let original_residual = check_farkas(&cert).expect("dead-row cert checks");
    let min = cert.minimized();
    // Dead rows dropped, live rows (and order) kept, arity parallel.
    assert_eq!(min.constraints.len(), 2, "both zero-multiplier rows drop");
    assert_eq!(min.multipliers.len(), 2);
    assert_eq!(min.multipliers, vec![Rat::ONE, Rat::ONE]);
    // The vanished variable `w` is mentioned nowhere in the minimized cert.
    for c in &min.constraints {
        assert!(
            !c.coefficients.contains_key("wdead"),
            "dead variable wdead must vanish"
        );
    }
    // Round-trip: the minimized cert checks with the IDENTICAL residual.
    let min_residual = check_farkas(&min).expect("minimized cert still checks");
    assert_eq!(min_residual, original_residual);
    // Minimization is idempotent.
    assert_eq!(min.minimized().constraints.len(), min.constraints.len());
}

#[test]
fn entailment_minimization_round_trips_with_identical_bounds() {
    let cert = entailment_with_dead_rows();
    let original_bounds = check_entailment(&cert).expect("dead-row cert checks");
    let min = cert.minimized();
    assert_eq!(min.premises.len(), 1, "both zero-multiplier premises drop");
    assert_eq!(min.multipliers, vec![r(2, 1)]);
    for p in &min.premises {
        assert!(
            !p.coefficients.contains_key("udead"),
            "dead variable udead must vanish"
        );
    }
    let min_bounds = check_entailment(&min).expect("minimized cert still checks");
    assert_eq!(min_bounds, original_bounds);
}

#[test]
fn minimization_is_fail_closed_on_certs_that_do_not_check() {
    // Arity mismatch: minimization must return the cert unchanged (fail closed)
    // rather than "repair" it by dropping rows.
    let broken = FarkasCertificate {
        constraints: vec![LinearConstraint::with_kind(
            ConstraintKind::Ge,
            &[("y", Rat::ONE)],
            r(3, 1),
        )],
        multipliers: vec![Rat::ONE, Rat::ZERO],
    };
    let min = broken.minimized();
    assert_eq!(min.constraints.len(), 1);
    assert_eq!(
        min.multipliers.len(),
        2,
        "fail-closed: arity-broken cert unchanged"
    );
    assert!(check_farkas(&min).is_err(), "still rejected downstream");

    // Exercise the symmetric malformed-arity guard on entailments too.
    let broken = EntailmentCertificate {
        premises: vec![
            LinearConstraint::with_kind(ConstraintKind::Le, &[("dead", Rat::ONE)], Rat::ZERO),
            LinearConstraint::with_kind(ConstraintKind::Le, &[("x", Rat::ONE)], r(1, 1)),
        ],
        multipliers: vec![Rat::ZERO],
        conclusion: LinearConstraint::with_kind(ConstraintKind::Le, &[("x", Rat::ONE)], r(1, 1)),
    };
    let min = broken.minimized();
    assert_eq!(min.premises.len(), 2);
    assert_eq!(
        min.multipliers.len(),
        1,
        "fail-closed: arity-broken entailment unchanged"
    );
    assert!(check_entailment(&min).is_err(), "still rejected downstream");

    // A cert whose combination does not check: unchanged too.
    let not_established = EntailmentCertificate {
        premises: vec![
            LinearConstraint::with_kind(ConstraintKind::Ge, &[("x", Rat::ONE)], r(1, 1)),
            LinearConstraint::with_kind(ConstraintKind::Le, &[("q", Rat::ONE)], r(1, 1)),
        ],
        multipliers: vec![Rat::ONE, Rat::ZERO],
        conclusion: LinearConstraint::with_kind(ConstraintKind::Ge, &[("x", Rat::ONE)], r(2, 1)),
    };
    let min = not_established.minimized();
    assert_eq!(
        min.premises.len(),
        2,
        "fail-closed: non-checking cert unchanged"
    );
    assert!(check_entailment(&min).is_err(), "still rejected downstream");

    // And the symmetric non-checking Farkas case: the dead row is retained
    // because minimization must not rewrite a certificate it cannot validate.
    let not_established = FarkasCertificate {
        constraints: vec![
            LinearConstraint::with_kind(ConstraintKind::Le, &[("dead", Rat::ONE)], Rat::ZERO),
            LinearConstraint::with_kind(ConstraintKind::Ge, &[("x", Rat::ONE)], Rat::ZERO),
            LinearConstraint::with_kind(ConstraintKind::Le, &[("x", Rat::ONE)], r(1, 1)),
        ],
        multipliers: vec![Rat::ZERO, Rat::ONE, Rat::ONE],
    };
    let min = not_established.minimized();
    assert_eq!(
        min.constraints.len(),
        3,
        "fail-closed: non-checking cert unchanged"
    );
    assert_eq!(min.multipliers, vec![Rat::ZERO, Rat::ONE, Rat::ONE]);
    assert!(check_farkas(&min).is_err(), "still rejected downstream");
}

#[test]
fn alethe_emission_of_dead_row_certs_is_minimized() {
    let cert = farkas_with_dead_rows();
    let em = farkas_to_alethe(&cert).expect("dead-row Farkas cert emits");
    // The dead rows' variable and atoms are gone from BOTH problem and proof:
    // only the two live rows are assumed (h0, h1) and resolved.
    assert!(
        !em.problem.contains("wdead"),
        "dead variable wdead must not be declared"
    );
    assert!(!em.proof.contains("wdead"));
    assert!(em.proof.contains("h1"));
    assert!(!em.proof.contains("h2"), "only the live rows are assumed");
    let ent = entailment_with_dead_rows();
    let em = entailment_to_alethe(&ent).expect("dead-row entailment emits");
    assert!(
        !em.problem.contains("udead"),
        "dead variable udead must not be declared"
    );
    assert!(!em.proof.contains("udead"));
    // Live premise + negated conclusion = exactly h0 and h1.
    assert!(em.proof.contains("h1"));
    assert!(
        !em.proof.contains("h2"),
        "only live premise + negated conclusion"
    );
}

#[test]
#[cfg(feature = "external-carcara")]
fn minimized_dead_row_alethe_is_carcara_valid() {
    let carcara = require_carcara();
    let farkas = farkas_to_alethe(&farkas_with_dead_rows()).expect("dead-row Farkas cert emits");
    assert_carcara_valid(&carcara, &farkas, "farkas_minimized");

    let entailment =
        entailment_to_alethe(&entailment_with_dead_rows()).expect("dead-row entailment emits");
    assert_carcara_valid(&carcara, &entailment, "entailment_minimized");
}
