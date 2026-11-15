// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #vit-margin-alpha-complex (P8 bank gate): the INDEPENDENT-FORWARD oracle
//! seam for adjudicating a ROOT-terminal multi-objective `Verified`.
//!
//! # Why an independent forward
//!
//! The containment fence (`multi_objective/containment_fence_lane.rs`) compares
//! ny's certified bounds against ny's OWN sound f64 point walk. That catches a
//! defect in the bound pipeline (the 2026-08-25 N-ary `Concat` shape), but not
//! one in the GRAPH ny built: a graph-loading or op-semantics bug is shared by
//! the bounds AND the walk, so both agree on the wrong function. vit_2023 is
//! all truth-unsat, so no counterexample can ever expose such a bug — the only
//! independent evidence is to evaluate the ORIGINAL model with a different
//! implementation (ONNX Runtime, the runtime the organizer itself uses) and
//! check that the certified objective lower bound never exceeds the sampled
//! objective value (`vit_margin_alpha_root_bound_below_ort_min`'s shape, now
//! on the scored path).
//!
//! The ONNX Runtime forward lives in ny-onnx/ny-cli, which ny-propagate cannot
//! depend on — hence this trait seam, mirroring the
//! [`GraphMipLeafOracle`](super::graph_mip_leaf::GraphMipLeafOracle) precedent
//! (a runtime-attached `Arc` on `BetaCrownVerifier`, preserved by
//! `with_config_from`, `None` = byte-identical default).
//!
//! # Contract (the implementor's obligations)
//!
//! - [`RootAdjudicationOracle::evaluate`] runs the ORIGINAL model on ONE flat
//!   input point in `X_i` (row-major) order and returns the flat output in
//!   `Y_j` order — the same flattening the VNN-LIB property and the trusted
//!   sat gate use. It must not route through ny's graph loader.
//! - Any failure is an `Err`: the caller counts it and, when the operator
//!   armed the adjudication, fails CLOSED (no sample adjudicated means the
//!   verdict is not published as adjudicated).
//! - The oracle only ever DOWNGRADES: a separation turns `Verified` into
//!   `Unknown`; agreement changes nothing. It never tightens, closes or
//!   proves anything.

/// An independent forward of the original model, used to adjudicate a
/// root-terminal `Verified` (see the module docs).
pub trait RootAdjudicationOracle: Send + Sync {
    /// Short stable token for the observable (`ort=<name>`), e.g. `"ort"`.
    fn name(&self) -> &'static str;

    /// Evaluate the ORIGINAL model at one flat input point (`X_i` order),
    /// returning the flat output (`Y_j` order), or a human-readable error.
    fn evaluate(&self, flat_input: &[f32]) -> Result<Vec<f32>, String>;
}
