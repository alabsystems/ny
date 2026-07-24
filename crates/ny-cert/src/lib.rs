// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `ny-cert` — proof-carrying neural-network verification.
//!
//! This crate is the constructive core of NY's *Proof-Carrying Verification*
//! program (see `crates/ny-cert/SPEC.md`). It turns a CROWN
//! verification result into an **exact-rational certificate** that is checked
//! by Clean's trusted, kernel-backed external-certificate verifier — making a
//! verified verdict a machine-checkable proof rather than a floating-point
//! claim.
//!
//! The central theorem (the *Certificate Equivalence Theorem*) is that the
//! CROWN/β-CROWN backward pass over a ReLU network is identical to choosing the
//! non-negative multipliers of a Farkas / entailment certificate. [`crown`]
//! makes that identity executable for one ReLU hidden layer.
//!
//! ## Example
//!
//! ```
//! use ny_cert::{Rat, Relu1Problem};
//!
//! let r = |n: i128, d: i128| Rat::new(n, d).unwrap();
//! // z0 = x0 + x1, z1 = x0 − x1; y = a0 − a1 + 5/2; box [−1,1]².
//! let problem = Relu1Problem {
//!     w1: vec![vec![r(1, 1), r(1, 1)], vec![r(1, 1), r(-1, 1)]],
//!     b1: vec![Rat::ZERO, Rat::ZERO],
//!     w2: vec![r(1, 1), r(-1, 1)],
//!     b2: r(5, 2),
//!     input_lower: vec![r(-1, 1), r(-1, 1)],
//!     input_upper: vec![r(1, 1), r(1, 1)],
//!     alpha: Some(vec![r(1, 2), r(1, 2)]),
//! };
//! let certified = problem.certify(Rat::ZERO).unwrap();
//! assert_eq!(certified.lower_bound, r(1, 2)); // y ≥ 1/2, so y ≥ 0 holds.
//! ny_cert::check_entailment(&certified.entailment).unwrap();
//! ```
#![cfg_attr(trust_verify, feature(contracts))]
// Under tRustc contract verification (`--cfg trust_verify`) the first-class
// `#[ensures]` attribute lives in the unstable `core::contracts` module; the feature
// gate above enables it. Stable rustc never sees this (cfg-gated) and falls back to
// the NY-owned `trust` compatibility macros. See `selfcheck.rs` for the dual import.

pub mod alethe_bridge;
pub mod alethe_emit;
pub mod branch;
pub mod budget;
pub mod certz;
pub mod cite_check;
pub mod crown;
pub mod crown_deep;
pub mod eval;
pub mod exact;
pub mod generate;
pub mod invprop_cert;
pub mod proof_carrying;
pub mod rational;
pub mod sbar;
pub mod schema;
pub mod selfcheck;
pub mod tcb_check;

pub use alethe_emit::{
    branch_tree_to_alethe, entailment_to_alethe, farkas_to_alethe, AletheEmission, EmitError,
};
pub use branch::{
    branch_tree_leaf_batch_json, branch_tree_to_json, check_branch_tree, AxisPartition,
    BranchError, BranchLeaf, BranchTreeCertificate, ThreshDir,
};
pub use certz::{entailment_to_certz_json, entailment_to_certz_lean, CertZError};
pub use crown::{CertifiedRelu1, CrownError, Relu1Problem};
pub use invprop_cert::{InvpropAugmentedCertificate, OutputDualRow};
pub use rational::{Rat, RatError};
pub use schema::{
    chain_to_json, entailment_to_json, farkas_to_json, ConstraintKind, EntailmentCertificate,
    FarkasCertificate, LinearConstraint,
};
pub use selfcheck::{check_chain, check_entailment, check_farkas, CheckError};

/// Identity barrier, `#[inline(never)]` so the optimizer builds a FRESH in-body
/// `Err(_)` aggregate at each call site instead of forwarding / const-promoting
/// the whole `Result` (a move the `#[ensures]` return-grounding lane cannot
/// resolve). Generic over the error type so ONE fn serves every ensures-bearing
/// delegator's error (`CrownError`/`DeepCrownError`/`BranchError`/`SbarError`/
/// `CheckError`). Behaviour-identical: returns its argument unchanged.
#[inline(never)]
pub(crate) fn err_barrier<E>(e: E) -> E {
    e
}
