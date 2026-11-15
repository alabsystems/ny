// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The shared CERTIFIED binary64 zonotope core (VIBECHECK plan T2).
//!
//! One engine for every zonotope lane: V2 (`#cgan-child-zono`, the first
//! user) builds it; V4 (ml4acopf forward root screen) and vit I1 extend it by
//! adding op transformers and programs — they must not fork it. V4 adds
//! [`elementwise`] (selection, per-element affine, ID-aligned product, square,
//! rigorous affine bands) and [`dag`] (a multi-input DAG program over a
//! `GraphNetwork`), plus the screen entry point [`fwd_root`]. vit I1
//! (`#vit-zono-root`) adds [`vit_ops`] (live-live MatMul, Exp / Reciprocal /
//! Clip bands, ReduceSum / ReduceMean, the direct-Softmax pairwise box) behind
//! [`dag::DagOpSet::Vit`], and the root reference-map lane [`vit_root`].
//!
//! # Representation
//!
//! [`ZonoState`]: centre `c`, generator rows `G` (row-major, one row per noise
//! symbol, each row carrying an explicit [`SymbolId`]), and a per-element
//! interval remainder `r >= 0`. It denotes `{c + G e + rho : e in [-1,1]^K,
//! |rho| <= r}` in EXACT real arithmetic.
//!
//! # The soundness contract every transformer keeps
//!
//! * Arithmetic is plain binary64 round-to-nearest; EVERY rounding error is
//!   charged into `r` with an explicit Higham `gamma_n` envelope (order
//!   independent, so faer's blocked GEMM and FMA contraction are covered),
//!   and every endpoint is rounded outward one ulp. A [`rounding::TINY`]
//!   floor absorbs gradual underflow.
//! * Model errors are charged too: BatchNorm uses the layer's certified
//!   exact-real coefficient radii (never an uncharged fold, plan U10) — the
//!   binary64 enclosure `BatchNormLayer::new` computes of the exact-real
//!   coefficients whenever the layer carries an intact one (#zono-exact-bn,
//!   ~`1e-15` relative), else the binary32 affine's radii (~`6e-8`); see
//!   [`BatchNormCoefficients`].
//! * Symbols are combined BY ID ([`ids::add_aligned`]), never by position
//!   (plan U8, the class of ny's retracted N-ary Concat defect).
//! * A ReLU's pre-activation box may be intersected with any other certified
//!   enclosure of the same values before DeepZ picks its slope; a crossed
//!   intersection keeps the looser side (never collapses, plan U9).
//! * Any NaN/Inf, shape mismatch or deadline REFUSES ([`ZonoRefusal`]); a
//!   refused pass publishes nothing, so the caller keeps its previous map.
//! * No generator cap below the configured budget is ever applied, and the
//!   budget folds (sound) rather than refuses; `0` is never "unlimited".
//!
//! Test-only fault switches (`relu::fault`, `ids::fault`) exist solely so the
//! mutation pins can prove the pointwise form checks catch a wrong offset
//! sign, a dropped remainder and positional alignment. They are compiled only
//! under `cfg(test)`.

pub mod affine;
pub mod cgan;
pub mod concretize;
pub mod dag;
pub mod elementwise;
pub mod form_check;
pub mod fwd_root;
pub mod ids;
pub mod kernels;
pub mod program;
pub mod relu;
pub mod rounding;
pub mod spec_crown;
pub mod state;
pub mod vit_ops;
pub mod vit_root;

#[cfg(test)]
pub(crate) mod tests;

#[cfg(test)]
mod exact_bn_tests;

#[cfg(test)]
mod fwd_root_tests;

#[cfg(test)]
pub(crate) mod vit_root_tests;

#[cfg(test)]
mod batch_tests;

pub use affine::{
    apply_affine, apply_affine_par, apply_channel_affine, apply_channel_affine_par, AffineOp,
    BatchNormCoefficients, ChannelAffine,
};
pub use cgan::{
    cgan_child_zono_plan, reset_cgan_child_zono_receipt, take_cgan_child_zono_receipt,
    CganChildZonoBatchTally, CganChildZonoPlan, CganChildZonoReceipt, ChildZonoEvaluator,
};
pub use concretize::{concretize, count_unstable, intersect_keep_looser, to_f32_outward};
pub use dag::{
    precheck_dag, precheck_dag_opset, DagForwardOptions, DagForwardOutcome, DagForwardStats,
    DagNodeView, DagOp, DagOpSet, DagProgram, DagStep,
};
pub use elementwise::{Band, ElementScale, SelectSource, SmoothKind};
pub use fwd_root::{
    admitted_byte_budget, forward_root_screen_plan, run_forward_root_screen, FwdRootDecline,
    FwdRootPlan, FwdRootScreen,
};
pub use ids::{add_aligned, SymbolAllocator, SymbolId};
pub use kernels::{ConvGeom, LinearKernel, RowPar};
pub use program::{
    ReluSummary, StepOp, ZonoBatchChild, ZonoForwardOptions, ZonoForwardOutcome, ZonoForwardStats,
    ZonoNodeView, ZonoProgram, ZonoStep, ZonoTrace,
};
pub use relu::{apply_relu, apply_relu_par, ReluReport};
pub use spec_crown::{spec_lower_bounds, SpecQuery};
pub use state::{InputSymbol, ZonoState};
pub use vit_ops::MatMulDims;
pub use vit_root::{
    compute_vit_zono_root_map, merge_shrink_only, vit_zono_root_map, vit_zono_root_plan,
    VitZonoDecline, VitZonoMergeStats, VitZonoRootMap, VitZonoRootPlan,
};

/// Why a zonotope computation declined. Every variant is fail-closed: the
/// caller keeps the enclosure it already had.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ZonoRefusal {
    /// The graph is not a program the core can represent (topology).
    UnsupportedGraph(String),
    /// A node's layer is outside the op set.
    UnsupportedLayer {
        /// Node name.
        node: String,
        /// Layer type name.
        layer: &'static str,
    },
    /// A shape did not match.
    Shape(String),
    /// A NaN or infinity appeared.
    NonFinite(&'static str),
    /// The deadline passed.
    Deadline,
    /// Memory admission refused the work.
    Memory {
        /// Estimated bytes of the refused work.
        estimate_bytes: u64,
        /// Budget it had to fit in.
        budget_bytes: u64,
    },
    /// An internal consistency check failed.
    Internal(String),
}

impl ZonoRefusal {
    /// Stable lowercase token for observables.
    #[must_use]
    pub fn token(&self) -> &'static str {
        match self {
            Self::UnsupportedGraph(_) => "unsupported_graph",
            Self::UnsupportedLayer { .. } => "unsupported_layer",
            Self::Shape(_) => "shape",
            Self::NonFinite(_) => "non_finite",
            Self::Deadline => "deadline",
            Self::Memory { .. } => "memory",
            Self::Internal(_) => "internal",
        }
    }
}

impl std::fmt::Display for ZonoRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedGraph(why) => write!(f, "unsupported graph: {why}"),
            Self::UnsupportedLayer { node, layer } => {
                write!(f, "unsupported layer {layer} at node `{node}`")
            }
            Self::Shape(why) => write!(f, "shape: {why}"),
            Self::NonFinite(what) => write!(f, "non-finite {what}"),
            Self::Deadline => write!(f, "deadline"),
            Self::Memory {
                estimate_bytes,
                budget_bytes,
            } => write!(
                f,
                "memory: estimate {estimate_bytes} B exceeds budget {budget_bytes} B"
            ),
            Self::Internal(why) => write!(f, "internal: {why}"),
        }
    }
}

impl std::error::Error for ZonoRefusal {}
