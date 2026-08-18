// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

#![forbid(unsafe_code)]

//! `ny-levers` — the central declaration registry and the single process-env
//! choke point for the workspace's `NY_*` configuration surface.
//!
//! # What this crate is for
//!
//! The workspace reads ~850 `NY_*` environment variables from ~230 files, each
//! with its own parser, its own default and no record of why that default is
//! defensible. That is a performance problem before it is a hygiene problem:
//! budget is spent by levers nobody can enumerate, per-instance adaptation is
//! impossible while values are latched process-wide, and every A/B is exposed
//! to contamination from a knob the measurer did not know was armed.
//!
//! This crate implements Phase 0 of the fix
//! (`docs/LEVER_DEBT_EXECUTION_PLAN_2026-08-11.md`): the registry and narrow
//! direct-read ratchet (0a/0b), plus a layered receipt that flight-v3 records
//! only after typed preset resolution (0c). That makes declared configuration
//! auditable; it does NOT by itself migrate runtime readers. Those readers
//! still need the same frozen [`LeverSet`] threaded to them before Phase 2 is
//! complete.
//!
//! # The four pieces
//!
//! * [`LeverDecl`] and [`declare_levers`] — one centrally owned declaration
//!   per lever, filed in a module [`Registry`] by the same expansion that
//!   creates it. [`Bucket`] has no `Dead` variant on purpose: a dead lever
//!   cannot be described, only deleted.
//! * [`collect`] — merges module registries with duplicate-name detection that
//!   distinguishes an accidental double export from a *declared* multi-reader
//!   lever such as `NY_EFT_ERR`.
//! * [`read`] and [`RawLeverInputs`] — the one place that touches `std::env`,
//!   with the repo's exact `"1"` arming rule pinned and non-UTF-8 presence
//!   preserved.
//! * [`LeverSet`] — a frozen per-run snapshot (the Phase-2 vehicle) plus
//!   [`LeverSet::receipt`]. The API can produce an exhaustive registry receipt;
//!   the current runtime readers still need to be migrated to consume the set
//!   before Phase 2 is complete.
//!
//! # The ratchet
//!
//! `tests/ratchet.rs` counts occurrences of the two direct-literal
//! `std::env::{var,var_os}` call forms whose first token starts with `NY_`, and
//! requires an exact match to the checked-in baseline. It is deliberately a
//! net-count tripwire, not a parser
//! or completeness proof: aliases, wrappers, multiline calls, and a
//! same-count substitution are outside its claim. The canonical gate executes
//! it; every migration of a counted occurrence lowers the baseline in the
//! same commit.

mod decl;
mod env;
mod macros;
mod registry;
mod set;

pub mod decls;
/// The SEARCH SPACE over levers: which knobs an automated search may move, what
/// values survive the parser, and which combinations are provably inert.
pub mod space;

pub use decl::{
    Bucket, DefaultSpec, LeverDecl, LeverKind, MoatRisk, Provenance, ReaderSite, Scope,
};
pub use env::{
    read, read_over_config, read_over_config_with, read_presence, read_raw, read_with, LeverValue,
    RawLeverInputs, ResolveError, Resolved, Source,
};
pub use registry::{collect, CollectError, LeverRegistry, Registry};
pub use set::LeverSet;

use std::sync::OnceLock;

/// The merged workspace registry: every lever declared anywhere in
/// [`decls`].
///
/// # Panics
///
/// Panics if the declaration modules do not merge — a duplicate name is a
/// programming error that must fail loudly at first use, not degrade into a
/// half-populated registry. `registry_merges` in [`decls`] catches it in CI
/// before it can reach a run.
pub fn all() -> &'static LeverRegistry {
    static ALL: OnceLock<LeverRegistry> = OnceLock::new();
    ALL.get_or_init(|| collect(decls::registries()).expect("declared levers must merge"))
}
