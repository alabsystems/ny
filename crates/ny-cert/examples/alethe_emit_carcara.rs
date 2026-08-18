// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Differential validation of NY's Alethe EMISSION against the standard
//! independent checker [carcara](https://github.com/ufmg-smite/carcara):
//! every emitted (problem, proof) pair must check as `valid` with
//! `carcara check proof.alethe problem.smt2`.
//!
//! Run explicitly with:
//! `cargo run -p ny-cert --example alethe_emit_carcara --release`.
//! Invoking the example without a reachable checker (`$NY_CARCARA` or
//! `carcara` on `PATH`) is a hard failure, never a skip.

use ny_cert::branch::{AxisPartition, BranchLeaf};
use ny_cert::{
    branch_tree_to_alethe, entailment_to_alethe, farkas_to_alethe, AletheEmission,
    BranchTreeCertificate, ConstraintKind, EntailmentCertificate, FarkasCertificate,
    LinearConstraint, Rat, ThreshDir,
};
use std::path::PathBuf;
use std::process::Command;

/// Locate `carcara`: `$NY_CARCARA`, then `carcara` on `PATH`.
fn locate_carcara() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("NY_CARCARA") {
        let p = PathBuf::from(p);
        if p.is_file() {
            return Some(p);
        }
    }
    let out = Command::new("sh")
        .arg("-c")
        .arg("command -v carcara")
        .output()
        .ok()?;
    if out.status.success() {
        let path = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if !path.is_empty() {
            return Some(PathBuf::from(path));
        }
    }
    None
}

/// Write the pair to a scratch dir and return Carcara's result.
fn run_carcara(em: &AletheEmission, name: &str) -> std::process::Output {
    let carcara = locate_carcara().unwrap_or_else(|| {
        panic!("cannot validate {name}: no `carcara` binary (set NY_CARCARA or PATH)")
    });
    let dir = tempfile::tempdir().expect("scratch dir");
    let problem = dir.path().join(format!("{name}.smt2"));
    let proof = dir.path().join(format!("{name}.alethe"));
    std::fs::write(&problem, &em.problem).expect("write problem");
    std::fs::write(&proof, &em.proof).expect("write proof");
    Command::new(&carcara)
        .arg("check")
        .arg(&proof)
        .arg(&problem)
        .output()
        .unwrap_or_else(|err| panic!("failed to run Carcara at {}: {err}", carcara.display()))
}

/// Require Carcara to accept the emitted proof.
fn assert_carcara_valid(em: &AletheEmission, name: &str) {
    let out = run_carcara(em, name);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success() && stdout.trim() == "valid",
        "carcara rejected {name}:\nstdout: {stdout}\nstderr: {stderr}\nproblem:\n{}\nproof:\n{}",
        em.problem,
        em.proof
    );
}

/// Require Carcara not to accept a deliberately corrupted emitted proof.
fn assert_carcara_invalid(em: &AletheEmission, name: &str) {
    let out = run_carcara(em, name);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !out.status.success() && stdout.trim() == "invalid",
        "carcara did not explicitly reject deliberately corrupted proof {name}:\nstatus: {}\nstdout: {stdout}\nstderr: {stderr}\nproblem:\n{}\nproof:\n{}",
        out.status,
        em.problem,
        em.proof
    );
}

fn r(n: i128, d: i128) -> Rat {
    Rat::new(n, d).unwrap()
}

fn invprop_farkas() -> FarkasCertificate {
    // The invprop_cert HOLD case: y >= 3 premise + y <= 1 violation row.
    FarkasCertificate {
        constraints: vec![
            LinearConstraint::with_kind(ConstraintKind::Ge, &[("y", Rat::ONE)], r(3, 1)),
            LinearConstraint::with_kind(ConstraintKind::Le, &[("y", Rat::ONE)], r(1, 1)),
        ],
        multipliers: vec![Rat::ONE, Rat::ONE],
    }
}

fn carcara_validates_farkas_emission() {
    let cert = invprop_farkas();
    let em = farkas_to_alethe(&cert).expect("valid Farkas cert emits");
    assert_carcara_valid(&em, "farkas_invprop");
}

fn carcara_rejects_corrupted_farkas_emission() {
    let mut em = farkas_to_alethe(&invprop_farkas()).expect("valid Farkas cert emits");
    let valid_args = ":args (1.0 1.0)";
    let corrupted_args = ":args (0.0 1.0)";
    assert!(
        em.proof.contains(valid_args),
        "expected Farkas multiplier list is absent from emitted proof"
    );
    em.proof = em.proof.replacen(valid_args, corrupted_args, 1);
    assert_carcara_invalid(&em, "farkas_corrupted_multiplier");
}

fn carcara_validates_fractional_and_multi_variable_farkas() {
    // 2x + 3y <= 5 (mult 1) + x + y >= 4 (mult 3) + -x >= -1 i.e. x <= 1 …
    // exercise negative coefficients, fractions, and strict inequalities:
    // -2x + y < -1 (mult 1) and 2x - y <= 1/2 (mult 1): 0 < -1/2.
    let cert = FarkasCertificate {
        constraints: vec![
            LinearConstraint::with_kind(
                ConstraintKind::Lt,
                &[("x", r(-2, 1)), ("y", Rat::ONE)],
                r(-1, 1),
            ),
            LinearConstraint::with_kind(
                ConstraintKind::Le,
                &[("x", r(2, 1)), ("y", r(-1, 1))],
                r(1, 2),
            ),
        ],
        multipliers: vec![Rat::ONE, Rat::ONE],
    };
    let em = farkas_to_alethe(&cert).expect("valid Farkas cert emits");
    assert_carcara_valid(&em, "farkas_fractional");
}

fn carcara_validates_entailment_emission() {
    // x >= 1 (mult 2) entails 2x >= 2; refuted via the negated strict 2x < 2.
    let cert = EntailmentCertificate {
        premises: vec![LinearConstraint::with_kind(
            ConstraintKind::Ge,
            &[("x", Rat::ONE)],
            r(1, 1),
        )],
        multipliers: vec![r(2, 1)],
        conclusion: LinearConstraint::with_kind(ConstraintKind::Ge, &[("x", r(2, 1))], r(2, 1)),
    };
    let em = entailment_to_alethe(&cert).expect("valid entailment emits");
    assert_carcara_valid(&em, "entailment");
}

/// Face `var (kind) bound` with unit coefficient.
fn face(var: &str, kind: ConstraintKind, bound: Rat) -> LinearConstraint {
    LinearConstraint::with_kind(kind, &[(var, Rat::ONE)], bound)
}

/// Entailment `a0*x0 + a1*x1 >= bound - b` over a cell, from corner faces.
fn ent(a0: Rat, a1: Rat, b: Rat, bound: Rat, lo: &[Rat], hi: &[Rat]) -> EntailmentCertificate {
    let (p0, mu0) = if a0.is_negative() {
        (face("x0", ConstraintKind::Le, hi[0]), a0.neg())
    } else {
        (face("x0", ConstraintKind::Ge, lo[0]), a0)
    };
    let (p1, mu1) = if a1.is_negative() {
        (face("x1", ConstraintKind::Le, hi[1]), a1.neg())
    } else {
        (face("x1", ConstraintKind::Ge, lo[1]), a1)
    };
    EntailmentCertificate {
        premises: vec![p0, p1],
        multipliers: vec![mu0, mu1],
        conclusion: LinearConstraint::with_kind(
            ConstraintKind::Ge,
            &[("x0", a0), ("x1", a1)],
            bound.sub(b).unwrap(),
        ),
    }
}

fn carcara_validates_two_leaf_branch_tree() {
    // [-1,1]² split at x0 = 0 for y = x0; threshold -2 (property y <= -2).
    let lo = r(-1, 1);
    let mid = r(0, 1);
    let hi = r(1, 1);
    let mk = |x0lo: Rat, x0hi: Rat, bound: Rat| BranchLeaf {
        lo: vec![x0lo, lo],
        hi: vec![x0hi, hi],
        bound,
        member_entailments: vec![ent(
            Rat::ONE,
            Rat::ZERO,
            Rat::ZERO,
            bound,
            &[x0lo, lo],
            &[x0hi, hi],
        )],
        member_biases: vec![Rat::ZERO],
    };
    let cert = BranchTreeCertificate {
        axes: vec![
            AxisPartition {
                var: "x0".to_owned(),
                edges: vec![lo, mid, hi],
            },
            AxisPartition {
                var: "x1".to_owned(),
                edges: vec![lo, hi],
            },
        ],
        leaves: vec![mk(lo, mid, r(-1, 1)), mk(mid, hi, Rat::ZERO)],
        threshold: r(-2, 1),
        dir: ThreshDir::Le,
    };
    let em = branch_tree_to_alethe(&cert).expect("valid branch tree emits");
    assert_carcara_valid(&em, "branch_two_leaf");
}

fn carcara_validates_three_by_two_grid_branch_tree() {
    // A 3x2 product grid (interior edges on BOTH axes) for y = x0 + 2*x1 + 1
    // over [-1,1]²; per-cell lower bound = x0lo + 2*x1lo + 1; min = -2;
    // threshold -3 (property y <= -3). Exercises the multi-level resolution
    // fold, split-tautology reuse, and implicit literal deduplication.
    let e0 = r(-1, 1);
    let e1 = r(0, 1);
    let e2 = r(1, 2);
    let e3 = r(1, 1);
    let bias = r(1, 1);
    let a0 = Rat::ONE;
    let a1 = r(2, 1);
    let x0_edges = [e0, e1, e2, e3];
    let x1_edges = [e0, e1, e3];
    let mut leaves = Vec::new();
    for k1 in 0..2usize {
        for k0 in 0..3usize {
            let lo = [x0_edges[k0], x1_edges[k1]];
            let hi = [x0_edges[k0 + 1], x1_edges[k1 + 1]];
            // bound = a0*lo0 + a1*lo1 + bias (corner minimum of the affine).
            let bound = a0
                .mul(lo[0])
                .unwrap()
                .add(a1.mul(lo[1]).unwrap())
                .unwrap()
                .add(bias)
                .unwrap();
            leaves.push(BranchLeaf {
                lo: lo.to_vec(),
                hi: hi.to_vec(),
                bound,
                member_entailments: vec![ent(a0, a1, bias, bound, &lo, &hi)],
                member_biases: vec![bias],
            });
        }
    }
    let cert = BranchTreeCertificate {
        axes: vec![
            AxisPartition {
                var: "x0".to_owned(),
                edges: x0_edges.to_vec(),
            },
            AxisPartition {
                var: "x1".to_owned(),
                edges: x1_edges.to_vec(),
            },
        ],
        leaves,
        threshold: r(-3, 1),
        dir: ThreshDir::Le,
    };
    let em = branch_tree_to_alethe(&cert).expect("valid 3x2 branch tree emits");
    assert_carcara_valid(&em, "branch_3x2");
}

fn main() {
    let carcara = locate_carcara()
        .unwrap_or_else(|| panic!("Carcara is required: set NY_CARCARA or put `carcara` on PATH"));
    eprintln!("validating NY Alethe emission with {}", carcara.display());

    carcara_validates_farkas_emission();
    carcara_rejects_corrupted_farkas_emission();
    carcara_validates_fractional_and_multi_variable_farkas();
    carcara_validates_entailment_emission();
    carcara_validates_two_leaf_branch_tree();
    carcara_validates_three_by_two_grid_branch_tree();

    eprintln!("all six Carcara emission checks passed");
}
