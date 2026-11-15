// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `ny-proofcheck`: the B2.1 offline proof-bundle checker (ENGINE_PROGRAM
//! §1.7 item B2), behind `ny proof-check`.
//!
//! # What it is, and what it can never do
//!
//! A **downgrade-only, offline** checker. It reads a `ny-proof-bundle/v2` (or v2.1)
//! written after a published `unsat`, re-derives the proof with its own code,
//! and writes a report whose verdict is one of `verified`, `unconfirmed`,
//! `invalid`, `refuted`, `malformed`, `unsupported` or `checker_fault`. It
//! constructs no verdict and upgrades none: nothing in ny depends on this
//! crate except the `ny proof-check` subcommand, and no verdict path reads
//! its report (`crates/ny-cli/tests/proof_check_ratchet.rs`). Its one use is
//! the banking rule: an uncorroborated margin-row `unsat` banks only with a
//! `PROOF_CHECK_DONE rc=0` on a Proof-profile bundle of the same inputs.
//!
//! # The certificate (brief §1)
//!
//! A bundle claims a set of NODES. A node is a region
//! `R_n = { x ∈ B_n : σ·z_s(x) >= 0 for each ReLU literal (s, σ) }` (with
//! `B_n` the VNN-LIB box cut by input-threshold literals, `z_s` a Relu
//! input as THIS checker computes it) and a list of disjuncts it discharges.
//! The checker confirms a discharge by its own outward-rounded
//! back-substitution (`L > 0`, strict); the hint only supplies the dual point
//! (α, β), and by weak duality ANY α ∈ [0,1], β >= 0 gives a valid bound, so a
//! wrong or adversarial hint can only fail to confirm. It then proves, per
//! region and disjunct, that the confirmed (or certified-empty) nodes cover
//! the box by the case-split identity, which holds for any site. Cover plus
//! discharge is `unsat`.
//!
//! # Legacy lanes: well-formed, `unconfirmed`
//!
//! A legacy lane's published `unsat` streams no proof. The producer's writer
//! completes its bundle with whole-region `opaque` claims
//! (`src: legacy:<authority>`) or, when it cannot render the property or hash
//! an input, the `ny-proof-bundle/v2.1` `unavailable` forms ([`bundle`]).
//! Such a bundle is well-formed and at most `unconfirmed` (exit 10): an
//! `opaque` claim is never re-derived, and an `unavailable` bundle is never
//! read past its claims. A node-less legacy bundle, or a required clause no
//! whole-region node claims in an `unavailable` one, is `invalid`.
//!
//! # B2.3: frozen-dual prunes ([`fd`])
//!
//! A frozen-dual prune is an ordinary node (its split literals, its clause)
//! whose `hint.fd` names the producer's state and the dual point it was
//! certified at. The `fd` rung rebuilds the checker's OWN generator-space
//! state of that hint state (its own back-substitution over its own lowered
//! network and root bounds, the hint's slopes clamped into `[0, 1]`) and
//! re-evaluates the Lagrangian bound with its own directed rounding; a sampled
//! containment check of every built state is a self-containment fault. The
//! class trees (`class_part` frames), nested epoch subtrees (absolute
//! literals under their parent's prefix) and wave leaves compose through the
//! same per-disjunct cover: every node carries its full literal set.
//!
//! # Independence (brief §5)
//!
//! | item | shared with a producer? |
//! |---|---|
//! | ONNX decoding and lowering | no: [`onnx_wire`] + [`lower`] (BN kept unfolded as interval parameters) |
//! | VNN-LIB → objectives | no: [`vnnlib`], compared against the producer's `clauses` |
//! | bound arithmetic | code: no ([`arith`], [`ibp`], [`backsub`], [`fd`]); method: the same textbook family (α/β/triangle, Higham), so a conceptual error in that family would be shared |
//! | cover check | no: [`cover`], accepts overlaps |
//! | site mapping | no: literals self-interpret on the checker's IR |
//! | JSON | `ny-cert`'s total reader/writer (region numbers are strings parsed by `f64::from_str`, never JSON floats) |
//! | remaining TCB | this crate, rustc/std, IEEE-754 hardware (correctly rounded `+ - * / sqrt`) |
//!
//! `#![forbid(unsafe_code)]`; no fused multiply-add; no environment reads
//! (all ratcheted by `tests/independence_ratchet.rs`).

#![forbid(unsafe_code)]

pub mod arith;
pub mod backsub;
pub mod bundle;
pub mod check;
pub mod cover;
pub mod fd;
pub mod ibp;
pub mod lower;
pub mod onnx_wire;
pub mod report;
pub mod vnnlib;

#[doc(hidden)]
pub mod testkit;

pub use check::{check_bundle, default_report_path, CheckArgs, CheckOutcome, FaultInjection};
pub use report::Verdict;
