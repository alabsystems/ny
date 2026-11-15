// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

#![forbid(unsafe_code)]

//! `ny-bbe` — batched, UNTRUSTED bound search (ENGINE_PROGRAM §1.3).
//!
//! # What this crate is for
//!
//! The engine program keeps every lane's certified evaluator as the trusted
//! computing base and moves hot loops one at a time onto batched search that
//! is allowed to be fast and wrong: f32, SIMD, FMA, any reduction order, later
//! a GPU. A search bug here costs a lost row (+10), never a false verdict
//! (−150), because nothing this crate computes is published: its outputs are
//! [`ny_core::search::Untrusted`] values that a lane consumes only through an
//! INGEST GATE (clamped, then scored by the lane's unchanged certified
//! evaluator and kept only if strictly better).
//!
//! # The crate DAG (R2; pinned by `tests/crate_dag_ratchet.rs`)
//!
//! `ny-bbe` depends on `ny-core` ONLY. It cannot name `ny-propagate`, so it
//! cannot name — let alone construct — a certified type. `ny-propagate` never
//! depends on it either: it sees a backend as
//! `&dyn ny_core::search::SearchBackend`, constructed and registered by
//! `ny-cli`.
//!
//! # What is built
//!
//! * M5a — [`CpuF32Backend`]'s `root_joint_alpha`: abc's joint-alpha root
//!   search (CROWN init, spec prune, per-start-node alpha with an in-loop
//!   refresh of the unstable rows, early exits) over a lane's
//!   [`ny_core::search::SearchNet`], producing the objective's slopes as a hint
//!   for the margin-row `root_certify_once` ingest gate. See [`joint_root`]'s
//!   module docs for the recipe, the gradient and the cost shape.
//! * M4 — [`CpuF32Backend`]'s `upload_fd` / `fd_level`: the frozen-dual
//!   tier's LEVEL SEARCH (`margin_row.frozen_dual_bab_search: cpu_f32`). Every
//!   node of a level chunk is scored at its four dual points in f32 SIMD
//!   (far-first) and the best point, the clamped and far multipliers and the
//!   children's warm start are proposed; the lane certifies a proposal in CPU
//!   f64 with its own evaluator and splits the node otherwise. See [`fd`]'s
//!   module docs.
//! * V2 — [`CpuF32Backend`]'s `graph_root_alpha`: the vit root α search over a
//!   graph lane's [`ny_core::search::GraphSearchNet`] with the lane's
//!   certified boxes frozen — ReLU slopes, the attention products' McCormick
//!   planes and the fixed-index softmax's MulBinary planes, jointly, against
//!   the margin rows (exact envelope gradient, projected Adam). The lane
//!   ingests the result through its clamp + one certified evaluation + keep
//!   only if strictly better gate. See the `graph_root` module docs.
//! * M7 — [`CpuF32Backend`]'s `graph_bab_batch`: the graph β-CROWN lane's
//!   BATCHED domain search (`bab.graph_search: cpu_f32`) over the same V2
//!   [`ny_core::search::GraphSearchNet`] copy — per domain, the ReLU slopes
//!   and one Lagrange multiplier per split literal, jointly, with the lane's
//!   certified boxes frozen (the root is the domain with no split). The lane
//!   certifies each hinted domain ONCE with its own evaluator and keeps a
//!   row's bound only when strictly better. See the `graph_bab` module docs.
//! * M6 — [`CpuF32Backend`]'s `upload_wave` / `backward_batch` /
//!   `branch_scores`: the margin-row tree's WAVE search
//!   (`margin_row.tree_mode: wave`). Every domain of a wave is bounded by
//!   output-only beta-CROWN (certified boxes frozen, splits pinned, Adam on
//!   its slopes and split multipliers, keep-best) and its unsplit neurons are
//!   ranked by a BaBSR intercept score; the lane certifies a claimed domain in
//!   CPU f64 with its own Outward evaluator on a private pool and splits
//!   everything else. See [`wave`]'s module docs.
//! * M8 — [`CpuF32Backend`]'s `upload_forward_linear` /
//!   `forward_linear_batch`: the lowdim lane's FORWARD-LINEAR wave search
//!   (`bab.branching.input_split.lowdim_batch_engine: bbe`). Every domain of a
//!   wave is bounded by the lane's forward-linear intermediates and a CROWN
//!   objective pass in f32 (blocks of domains as SIMD column groups, no
//!   rounding charged); the lane re-bounds only the domains a hint claims
//!   with its own certified f64 forms and splits every other domain into its
//!   exact halves. See the `fl` module docs.

mod cpu;
pub mod dual_breakpoint;
pub mod fd;
mod fl;
mod graph_bab;
mod graph_root;
pub mod joint_root;
mod program;
mod real;
pub mod wave;

pub use cpu::CpuF32Backend;

#[cfg(test)]
mod fd_tests;
#[cfg(test)]
mod fl_tests;
#[cfg(test)]
mod graph_bab_tests;
#[cfg(test)]
mod graph_root_tests;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod wave_tests;
