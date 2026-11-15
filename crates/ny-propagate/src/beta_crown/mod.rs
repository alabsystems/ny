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
//! ## Algorithm
//!
//! 1. Initialize with root domain (no splits)
//! 2. While domains remain and not timed out:
//!    a. Pick the domain with highest configured violation priority
//!    b. Verify it if its bound excludes the unsafe region
//!    c. Select an unstable neuron to split
//!    d. Create two child domains (neuron active vs inactive)
//!    e. Compute bounds for children with joint α-β optimization
//!    f. Add children to queue if not verified/infeasible
//! 3. If all domains verified, return Verified; else Unknown/Timeout
//!
//! ## Optimizer Defaults
//!
//! [`AdaptiveOptConfig::default()`] configures the adaptive path:
//!
//! | Setting | Default | Notes |
//! |---------|---------|-------|
//! | Base optimizer | Adam | AMSGrad and Lookahead+Adam remain available overrides |
//! | RAdam | `false` | Avoid warmup-heavy behavior for short per-domain runs |
//! | AdamW (weight_decay) | `0.0` | Regularization is disabled by default |
//! | LR scheduler | `ExponentialDecay { ny: 0.98 }` | Per-iteration decay |
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
mod biccos_q_stage0;
pub(crate) mod branching;
/// Runtime resource custody for one same-source certified root-cut attempt.
pub mod certified_root_cut;
pub(crate) mod config;
mod conflict_clause_replay;
pub(crate) mod conflict_clauses;
pub(crate) mod conflict_clauses_graph;
mod conjunctive_proof_objectives;
pub mod constraint_store;
pub(crate) mod domain;
/// Graph-MIP leaf oracle seam (increment 6): implemented by ny-cli, consumed
/// by the graph ReLU-split BaB requeue (`docs/GRAPH_MIP_LEAF_SOLVER.md`).
pub mod graph_mip_leaf;
pub(crate) mod nonlinear_branching;
mod objective_owner;
pub(crate) mod result;
/// #vit-margin-alpha-complex (P8): independent-forward (ONNX Runtime) oracle
/// seam for adjudicating a root-terminal multi-objective `Verified`;
/// implemented by ny-cli, consumed by the default-off containment fence.
pub mod root_adjudication;
/// #relusplitter-lp-refine (wave-6 C7): root intermediate-bound refiner seam;
/// implemented by ny-cli (rigorous LP over ny-mip), consumed by the
/// multi-objective graph BaB root as a shrink-only intersection.
pub mod root_interm_refine;
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
// Certified Cut-CROWN C1/C1.5 surface (docs/CERTIFIED_CUT_CROWN_DESIGN.md):
// cut derivation + L1 generation, re-exported for the experiment harness and
// later ny-cert emission. All default-off. (The C2 `NY_CUT_FOLD` registry that
// used to live alongside these was deleted — it was statically unreachable and
// its arithmetic was not enclosure-preserving.)
pub use bab_cuts::{
    derive_cut_bound, derive_cut_bound_root, generate_l1_cuts, generate_l1_cuts_for_splits,
    generate_l1_cuts_signed, AffineRow, CutFoldScope, L1Cut, L1SplitGroupDiag, MultiReluCut,
    SignedCcMode, SignedCutDiag, SplitState,
};
pub use branching::{
    BranchingHeuristic, GenBabConstraint, GeneralSplitHistory, GraphConstraint,
    GraphNeuronConstraint, GraphSplitHistory, LayerRef, NeuronConstraint, NeuronSplit,
    SplitHistory,
};
pub use config::KfsbReduceOp;
pub use config::MipRefineIntermediates;
pub use config::SingleHiddenHullMode;
// #graph-lane-search (ENGINE_PROGRAM M7): own line.
pub use config::GraphLaneSearch;
// #vit-v3-bankable: own line.
pub use config::MoCoverCertify;
pub use config::{
    AdaptiveOptConfig, BetaCrownConfig, ChildBoundCalibrationConfig, ConvMode, CutEvictionPolicy,
    CutScoreWeights, DepthTwoBranchLookaheadConfig, DepthTwoBranchLookaheadMode, EdgeWaveCapConfig,
    InputClipType, LRScheduler, LookaheadConfig, PerLayerLR, PhaseBudgetConfig,
    SequentialInitialSchedule, VerificationArtifactAuthority, ATOMIC_ROOT_C_MARGIN_MAX_ITERATIONS,
};
pub use conjunctive_proof_objectives::{
    ConjunctiveProofObjectiveProvenance, ConjunctiveProofObjectives,
};
pub use constraint_store::{
    ArenaConstraintStore, ConstraintHeader, ConstraintOrigin, ConstraintSense,
    DomainConstraintStore, LinearConstraintRef,
};
pub use domain::{
    BabDomain, DomainProcessingConfig, DomainWithUnstable, GraphBabDomain, GraphCrownContext,
    GraphPrecomputedBounds, IntermediateLinearBounds, MultiObjDomainWithUnstable,
    MultiObjectiveGraphBabDomain, MultiObjectiveTargets, NodeBoundsHostAllocationInvalidV1,
    NodeBoundsHostAllocationObservationV1, NodeBoundsHostAllocationReceiptV1,
    NodeBoundsHostAllocationUnsupportedV1, NodeBoundsMap, NodeBoundsMapIter, NodeBoundsView,
    NodeBoundsViewIter, ObjectiveAggregation, SharedGraphBootstrap,
    NODE_BOUNDS_HOST_ALLOCATION_MODEL_V1,
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
pub use objective_owner::{
    OwnedSignNormalizedObjectiveSet, ResidentObjectiveInvalidV1, ResidentObjectiveObservationV1,
    ResidentObjectiveReceiptV1, ResidentObjectiveUnsupportedV1,
};
pub use result::{
    reset_global_timeout_deadline_source, reset_sequential_initial_schedule_receipt,
    take_global_timeout_deadline_source, take_sequential_initial_schedule_receipt,
    BabVerificationStatus, BetaCrownResult, SequentialEarlyQuitDisposition,
    SequentialInitialScheduleReceipt, ViolationWitness,
};
pub use state::{
    AlphaNeuronState, BetaEntry, BetaState, DomainAlphaState, GraphAlphaStateByteCensus,
    GraphAlphaStateRepresentation, GraphBetaEntry, GraphBetaState, GraphDomainAlphaState,
    PACKED_GRAPH_ALPHA_FORMAT_VERSION,
};
