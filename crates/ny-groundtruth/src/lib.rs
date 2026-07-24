// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

#![deny(unsafe_code)]

//! Geometric ground-truth graphs for network verification — milestone M1 of
//! `docs/GEOMETRIC_GROUND_TRUTH_PLAN.md`.
//!
//! A *ground-truth graph* is a [`ny_propagate::GraphNetwork`] we build
//! ourselves that computes an analytic geometric residual (distance-like
//! quantity) in **squared/residual polynomial form — no sqrt, no division**.
//! Because it is an ordinary `GraphNetwork`, the existing difference-network
//! machinery reduces "network `f` respects the geometry `g`" to a property
//! ny verifies today with IBP/CROWN:
//!
//! ```text
//! for all x in R:  f(x) ≥ g(x)      ⇔   h(x) = f(x) − g(x) ∈ [0, +∞)
//! for all x in R:  |f(x) − g(x)| ≤ ε ⇔   h(x) ∈ [−ε, ε]
//! ```
//!
//! # Modules
//!
//! - [`builders`] — the five primitive residuals (plane, sphere, cylinder,
//!   cone, torus) as graphs, with the plan §2.3 exact-constant contract;
//! - [`compose`] — affine pre-transform (pose) and min/max combination of
//!   primitives (CSG-style compound models);
//! - [`sidecar`] — the `.gt.json` sidecar format ([`GroundTruthSpec`]):
//!   portable serialization of a spec, validated through the builders;
//! - [`verify`] — [`verify_against_ground_truth`]: difference network +
//!   CROWN verify path + grid witness search for the falsified direction;
//! - [`wholefield`] — [`verify_whole_field_tolerance`]: whole-field
//!   "no-escape" continuous-domain tolerance semantics over the AbsBound
//!   verify path (`|f − g| ≤ tol` proved over the entire region, not sampled);
//! - [`cert`] — [`certify_dominance`]: exact-rational, self-checked
//!   entailment/Farkas certificates for `f ≥ g` (plane/linear ground truths
//!   AND single-level quadratic ones — sphere / cylinder / cone — via the
//!   kernel-checked pow2 envelope theorems; the nested-square/torus
//!   obstruction is documented there);
//! - [`escalate`] — Route B: [`SmtEscalation`] encodes the same difference
//!   network as one exact query to the AY solver (`unsat` ⇒ proved with an
//!   Alethe certificate; `sat` models are re-validated in exact rationals);
//! - [`reference`] — exact rational oracle evaluation of every residual
//!   (golden-test gold standard, the `ny gt eval` backend);
//! - [`error`] — typed rejection of constants that would have to be rounded.
//!
//! # Example
//!
//! ```rust
//! use ny_core::Bound;
//! use ny_groundtruth::{cylinder_residual, verify_against_ground_truth, Relation};
//! # fn main() -> Result<(), ny_groundtruth::GroundTruthError> {
//! // Ground truth: cylinder along z through (1, -2, 0.5), radius 1.5.
//! let g = cylinder_residual([0.0, 0.0, 1.0], [1.0, -2.0, 0.5], 1.5)?;
//! // `f` would be a learned surrogate loaded as a GraphNetwork; here we
//! // ask a trivial query — g agrees with itself to within 1.0 — to show
//! // the API shape end to end.
//! let region = vec![Bound::new(2.0, 3.0), Bound::new(-2.5, -1.5), Bound::new(-0.5, 0.5)];
//! let outcome = verify_against_ground_truth(&g, &g, Relation::AbsBound(1.0), &region)?;
//! println!("{outcome:?}");
//! # Ok(())
//! # }
//! ```
//!
//! # CLI
//!
//! The `ny gt` subcommands (`ny gt eval`, `ny gt verify`) live in `ny-cli`
//! and are thin wrappers over [`GroundTruthSpec`] +
//! [`verify_against_ground_truth`] (+ [`certify_dominance`] for
//! `--emit-cert`); `examples/cylinder_dominance.rs` shows the same flow
//! programmatically.

pub mod builders;
pub mod cert;
pub mod compose;
pub mod error;
pub mod escalate;
pub mod reference;
pub mod sidecar;
pub mod verify;
pub mod wholefield;

mod exact;

pub use builders::{
    cone_residual, cylinder_residual, signed_plane_distance, sphere_residual, torus_residual,
};
pub use cert::{certify_dominance, DominanceCertError, DominanceCertificate};
pub use compose::{max_of, min_of, with_pose, Pose};
pub use error::{GroundTruthError, Result};
pub use escalate::{EscalateError, EscalateOptions, SmtEscalation, SmtVerdict};
pub use sidecar::{BuilderKind, ComposeOp, ComposeSpec, GroundTruthSpec, PoseSpec, GT_FORMAT};
pub use verify::{
    verify_against_ground_truth, verify_against_ground_truth_with, GroundTruthOutcome, Relation,
    VerifyOptions,
};
pub use wholefield::{
    verify_whole_field_tolerance, verify_whole_field_tolerance_with, WholeFieldCertificate,
    WholeFieldOutcome,
};
