// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! β-CROWN: Branch-and-Bound Neural Network Verification
//!
//! Implements complete verification via branch-and-bound search over ReLU activation
//! space. When CROWN/α-CROWN bounds are inconclusive (lower bound < threshold),
//! β-CROWN splits unstable neurons into "always active" (x ≥ 0) and "always inactive"
//! (x ≤ 0) branches, encoding these constraints as β parameters in the bound computation.
//!
//! ## Key Features
//!
//! - **Joint α-β optimization**: Optimizes both ReLU relaxation slopes (α) and Lagrangian
//!   multipliers (β) together within each domain for tighter bounds.
//! - **Exact β gradients**: Uses augmented chain rule for precise gradient computation.
//! - **Parallel domain processing**: Processes multiple domains concurrently via Rayon.
//! - **Adaptive optimization**: Adam-style optimizer with configurable variants.
//!
//! ## Algorithm
//!
//! 1. Initialize with root domain (no splits)
//! 2. While domains remain and not timed out:
//!    a. Pick domain with worst (highest) lower bound
//!    b. If lower bound > threshold, domain is verified
//!    c. Select an unstable neuron to split
//!    d. Create two child domains (neuron active vs inactive)
//!    e. Compute bounds for children with joint α-β optimization
//!    f. Add children to queue if not verified/infeasible
//! 3. If all domains verified, return Verified; else Unknown/Timeout
//!
//! ## Optimizer Defaults
//!
//! `AdaptiveOptConfig::default()` follows the alpha-beta-CROWN optimizer
//! defaults for β-CROWN's adaptive path:
//!
//! | Setting | Default | Notes |
//! |---------|---------|-------|
//! | Base optimizer | Adam | AMSGrad and Lookahead+Adam remain available overrides |
//! | RAdam | `false` | Avoid warmup-heavy behavior for short per-domain runs |
//! | AdamW (weight_decay) | `0.0` | Regularization is disabled by default |
//! | LR scheduler | `ExponentialDecay { ny: 0.98 }` | Matches `optimized_bounds.py:74,498` |
//! | bias_correction | `true` | Essential for short iteration runs |
//! | grad_clip | `10.0` | Moderate clipping for stability |
//!
//! Use `LRScheduler::Constant` explicitly when a workload benchmarks better
//! without per-iteration decay.

// Submodules
pub(crate) mod bab_cuts;
/// #bab-frontier export channel (docs/BAB_FRONTIER_SEEDING_DESIGN.md):
/// surviving-unverified BaB subboxes at exhaustion, exported as attack seeds
/// for ny-cli's post-BaB falsification lane. Attack-only guidance.
pub mod bab_frontier_export;
pub(crate) mod branching;
pub(crate) mod config;
pub(crate) mod conflict_clauses;
pub(crate) mod conflict_clauses_graph;
pub mod constraint_store;
pub(crate) mod domain;
/// Graph-MIP leaf oracle seam (increment 6): implemented by ny-cli, consumed
/// by the graph ReLU-split BaB requeue (`docs/GRAPH_MIP_LEAF_SOLVER.md`).
pub mod graph_mip_leaf;
pub(crate) mod nonlinear_branching;
pub(crate) mod result;
pub(crate) mod state;
// Exposed at crate scope to support internal test helpers.
pub(crate) mod engine;

// Re-export main types at module level for convenience
pub use crate::batched_domain::BatchedDomains;
pub use crate::pgd_attack::{PgdAttacker, PgdConfig, PgdResult};
pub use bab_cuts::{
    CutKind, CutMetadata, CutPool, CutTerm, CuttingPlane, GraphCutPool, GraphCutTerm,
    GraphCuttingPlane,
};
pub use bab_frontier_export::{
    reset_bab_frontier_export, take_bab_frontier_seeds, BabFrontierSeed, BAB_FRONTIER_CORNER_BOXES,
};
// Certified Cut-CROWN C1/C1.5/C2 surface (docs/CERTIFIED_CUT_CROWN_DESIGN.md):
// cut derivation + L1 generation + the dark NY_CUT_FOLD registry, re-exported
// for the C2 experiment harness (ny-onnx integration test) and later ny-cert
// emission. All default-off; the fold registry is inert unless NY_CUT_FOLD=1.
pub use bab_cuts::{
    clear_cut_fold, cut_fold_applied_count, derive_cut_bound, derive_cut_bound_root,
    generate_l1_cuts, generate_l1_cuts_for_splits, generate_l1_cuts_signed,
    reset_cut_fold_applied_count, set_cut_fold, AffineRow, CutFoldEntry, CutFoldScope, L1Cut,
    L1SplitGroupDiag, MultiReluCut, SignedCcMode, SignedCutDiag, SplitState,
};
pub use branching::{
    BranchingHeuristic, GenBabConstraint, GeneralSplitHistory, GraphConstraint,
    GraphNeuronConstraint, GraphSplitHistory, LayerRef, NeuronConstraint, NeuronSplit,
    SplitHistory,
};
pub use config::KfsbReduceOp;
pub use config::{
    AdaptiveOptConfig, BetaCrownConfig, ConvMode, CutEvictionPolicy, CutScoreWeights,
    InputClipType, LRScheduler, LookaheadConfig, PerLayerLR, PhaseBudgetConfig,
};
pub use constraint_store::{
    ArenaConstraintStore, ConstraintHeader, ConstraintOrigin, ConstraintSense,
    DomainConstraintStore, LinearConstraintRef,
};
pub use domain::{
    BabDomain, DomainProcessingConfig, DomainWithUnstable, GraphBabDomain, GraphCrownContext,
    GraphPrecomputedBounds, IntermediateLinearBounds, MultiObjDomainWithUnstable,
    MultiObjectiveGraphBabDomain, MultiObjectiveTargets,
};
pub use engine::{
    BatchedSpecBackwardResult, BetaCrownVerifier, DenseSpecReboundMode, DenseSpecStageTiming,
    DomainSpecCrownResult, GraphDomainBatchCallerLane, GraphDomainBatchMetricsSink,
    GraphDomainBatchRecord, InputSplitBatchRecord, InputSplitMetricsSink, JointMarginCloser,
};
// Test-only debug surface for the GPU per-domain β machinery
// (#w4-split-tightening): the parity/monotonicity tests live in ny-gpu (they
// need a real device) while the internals are crate-private here.
#[doc(hidden)]
pub use engine::graph::gpu_beta_debug;
pub use nonlinear_branching::{
    BranchingDecision, BranchingPointMethod, NonlinearBranching, NonlinearBranchingConfig,
    NonlinearHeuristicMethod,
};
pub use result::{BabVerificationStatus, BetaCrownResult};
pub use state::{
    compute_arelu_cut_slope_bias, AlphaNeuronState, AreluState, BetaEntry, BetaState,
    DomainAlphaState, GraphAlphaStateByteCensus, GraphAlphaStateRepresentation, GraphBetaEntry,
    GraphBetaState, GraphDomainAlphaState, PACKED_GRAPH_ALPHA_FORMAT_VERSION,
};
