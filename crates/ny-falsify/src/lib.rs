// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! # `ny-falsify` — the falsification chassis
//!
//! A strategy proposes a candidate **input vector**. It can never name a
//! verdict. That sentence is currently enforced in ny by roughly thirty
//! comments across four crates; here it is enforced by the crate graph, by the
//! return type, and by two tests.
//!
//! ## What this crate is
//!
//! The chassis described in `docs/FALSIFICATION_PORTFOLIO_ARCHITECTURE_2026-08-18.md`
//! §3–§8, plus the two strategies that the calibration table (§1.5) attributes
//! wins to and that ny ships **no equivalent of**:
//!
//! | strategy | calibration evidence | why ny has no equivalent |
//! |---|---|---|
//! | [`SpecialPoints`](strategies::SpecialPoints) (S1) | **34 of 75** wins, at 2% of budget, `{"special": 8}` points on every one of them | ny's `low_dim_ort_corner_falsify` is `corners_full` capped at `UPFRONT_CORNER_MAX_VARIABLE_DIMS = 5` and emits *only* box vertices. `special` won `collins_rul` at **200** free inputs and `cgan` at 2, and four of its eight patterns (centre, two alternating parities, two bound/centre midpoints) are not vertices at all. |
//! | [`Square`](strategies::Square) (S9) | **2** wins — `soundnessbench` and `traffic_signs_recognition_2023` — the only strategy that took either | block sign-flip hill climbing against a *flat* objective. Its docstring names ny's `#deadlane` refusal set exactly: piecewise-constant / integer-gated nets where every gradient estimate is identically zero. |
//!
//! **The port rests on the calibration, not on fresh rows.** The E1 measurement
//! that this work was gated on returned an honest zero: 42 open-row
//! measurements across `cora_2024` (27), `traffic_signs` (9) and
//! `soundnessbench` (1), at the official per-instance budget and at 10–12× it,
//! produced **0 counterexamples**. On that same session's positive control —
//! five known-SAT `cora_2024` rows through the identical pipeline — three
//! different strategies won, and two of them (`special`, `square`) are the two
//! ported here. So the reach-differs-between-strategies thesis reproduced; it
//! did not convert into a capture. Nothing in this crate is armed by default
//! (§ [`Arming`]).
//!
//! ## The soundness boundary, structurally (design §8)
//!
//! - **M1 — the return type cannot express a verdict.** [`Proposal`] is
//!   `Candidate` / `Exhausted` / `Declined`. There is no `Verified`, no
//!   `Unsat`, no `Sat`. Pinned by `the_crate_cannot_name_a_verdict`.
//! - **M2 — the crate graph forbids the type.** This crate has zero workspace
//!   dependencies, so `VnncompResult` is not nameable here. Pinned by
//!   `the_manifest_has_no_workspace_dependency`.
//! - **M3 — candidates carry no outputs.** [`Candidate`] is inputs-only by
//!   construction, so no search arithmetic can reach a published witness. The
//!   `Y_j` coordinates come from the caller's real ORT forward on the ORIGINAL
//!   graph.
//! - **M4 — one publication seam.** Not in this crate, and deliberately so:
//!   `gate_sat_with_trusted_oracle` in `ny-cli` is unchanged and untouched by
//!   this work.
//!
//! [`Oracle::evaluate_batch`] does return a `holds` flag per point. That is not
//! a verdict and cannot become one: it is the caller's own arithmetic about its
//! own point, it never leaves the search, and the only thing a strategy does
//! with it is stop early and hand back the *input vector*. Every candidate is
//! still dropped unless the caller's unchanged trusted-oracle gate confirms it
//! under a real ORT forward.

#![forbid(non_ascii_idents)]

mod admission;
mod domain;
mod oracle;
mod proposal;
mod registry;
mod rng;
mod stall;
pub mod strategies;

pub use admission::{
    Admission, AdmissionContext, AdmissionProfile, AdmissionReceipt, AdmissionStage, Arming,
    CostClass, Decline, FactLadder, GraphFacts, ObjectiveQuality, ObjectiveRequirement, ParamSpace,
    SpecFacts, SpecShape,
};
pub use domain::{BoxError, SearchBox};
pub use oracle::{Oracle, OracleError, Score};
pub use proposal::{Candidate, Effort, Proposal, StrategyName};
pub use registry::{Budget, Receipt, Registry, SearchState, Slice, Strategy};
pub use rng::Rng;
pub use stall::{StallRule, StallTracker, WorkUnit};
