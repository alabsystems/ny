// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! GCP-CROWN cutting plane infrastructure.
//!
//! Contains:
//! - `CutTerm`/`CuttingPlane`/`CutPool`: Sequential network cuts
//! - `GraphCutTerm`/`GraphCuttingPlane`/`GraphCutPool`: Graph network cuts

pub(crate) mod c3_probe;
mod cut_fold;
mod cut_pool;
mod cutting_plane;
#[cfg(test)]
mod cutting_plane_tests;
mod graph_cut_pool;
mod graph_cutting_plane;
mod merge_index;
mod multi_relu_cut;
mod multi_relu_cut_gen;
mod types;

pub use cut_fold::CutFoldScope;
pub use cut_pool::CutPool;
pub use cutting_plane::{CuttingPlane, StrengthenedCut};
pub use graph_cut_pool::GraphCutPool;
pub use graph_cutting_plane::GraphCuttingPlane;
pub use multi_relu_cut::{
    derive_cut_bound, derive_cut_bound_root, AffineRow, MultiReluCut, SplitState,
};
pub use multi_relu_cut_gen::{
    generate_l1_cuts, generate_l1_cuts_for_splits, generate_l1_cuts_signed, L1Cut,
    L1SplitGroupDiag, SignedCcMode, SignedCutDiag,
};
pub use types::{CutKind, CutMetadata, CutTerm, GraphCutTerm};
