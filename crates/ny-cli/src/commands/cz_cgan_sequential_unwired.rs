// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Cooperative, default-off constrained-zonotope probes for the regular
//! imgSz32 cGAN architecture.
//!
//! The correlated probe remains fixed to `cGAN_imgSz32_nCh_1`. The separate
//! remainder-only interval lane admits the exact nCh1 and nCh3 profiles, whose
//! only structural difference is the authored generated-image channel count.
//! Both profiles use this exact 26-node topology:
//!
//! ```text
//! Gemm -> Reshape -> BN -> ConvT -> BN -> ReLU
//!      -> ConvT -> BN -> ReLU -> ConvT -> BN -> ReLU
//!      -> ConvT -> Conv -> ReLU -> Conv -> ReLU -> BN
//!      -> Conv -> ReLU -> BN -> Conv -> ReLU -> BN -> Reshape -> Gemm
//! ```
//!
//! Raw ONNX FLOAT initializer provenance and the finalized-network snapshot
//! are required. Every normalized graph parameter is compared bit-for-bit
//! against the raw initializer it came from before promotion to binary64.
//! Generator-side ReLUs are followed by bounded order reduction that always
//! retains the five leading latent symbols. The full diagnostic extent keeps
//! every symbol through the first two discriminator ReLUs and retains the 512
//! strongest columns at the last interior ReLU, always within the same explicit
//! transient-alpha and sparse-generator caps. Qualification prefixes retain
//! their original reduction receipts. The full diagnostic bounds the final
//! 512-coordinate ReLU directly with M17 after composing the authored
//! BN/Reshape/Gemm tail; the exact property-bound leaf route portfolios that
//! same M17 certificate with M20 as described below.
//! A separate default-unwired interval lane carries all uncertainty in the CZ
//! box remainder and bridges through a certified Box at each authored ReLU. An
//! exact property-bound leaf authenticates the first six preactivation Boxes
//! against the same sealed profile and streams them through the corresponding
//! correlated ReLUs. It retains the final preactivation Box for the M20
//! intersection with the correlated `Relu_23` tail. Neither lane can authorize
//! a verdict independently.
//! The same prepared leaf also runs one fixed, tightly bounded M24 Box-cut
//! search. Its exact replay is retained as measurement only: published scalar
//! bounds continue to use the historical strict M17/M20 selector.
//!
//! M17's exact rational objective uses the normalized BN's binary32 affine
//! coefficients.  A ny-mip certificate independently brackets the authored
//! BatchNorm square roots and measures exact rational errors from those graph
//! surrogates; those exact errors are folded into one nonnegative correction
//! over the pre-ReLU CZ hull.  The graph's rounded error arrays remain
//! provenance data only and are never trusted as proof inputs.
//!
//! This module cannot publish a verdict. Public probes remain diagnostic while
//! [`CganCzVerdictAuthority`] stays disabled. The private imgSz32 leaf-row
//! report may be consumed only by the property-bound beta-crown attachment,
//! which independently authenticates both planned rows and requires both
//! strict safe inequalities before granting all-row authority.

use std::mem::size_of;
use std::time::{Duration, Instant};

use ndarray::{Array2, Array4, ArrayViewD, Ix1, Ix2, Ix4};
use num_rational::BigRational;
use num_traits::{Signed, ToPrimitive, Zero};
use ny_core::LayerType;
use ny_mip::{
    bisect_constrained_zonotope_protected_alpha_unwired_with_budget,
    bound_relu_tail_triangle_dual_unwired_with_budget,
    certified_box_from_remainder_only_zonotope_unwired_with_budget,
    certified_box_relu_recenter_unwired_with_budget,
    certify_batch_norm_affine_surrogate_unwired_with_budget,
    constrained_zonotope_affine_unwired_with_budget,
    constrained_zonotope_batch_norm_unwired_with_budget,
    constrained_zonotope_conv2d_unwired_with_budget,
    constrained_zonotope_conv_transpose2d_unwired_with_budget,
    constrained_zonotope_order_reduce_unwired_with_budget,
    prepare_relu_tail_triangle_dual_unwired_attempt_with_budget,
    prepare_relu_tail_triangle_dual_unwired_with_budget, transform_relu_unwired_with_budget,
    transform_relu_with_auxiliary_bounds_unwired_with_budget, CertifiedAuxiliaryBounds64,
    CertifiedAuxiliaryBounds64BudgetError, CertifiedBox64BridgeError, CertifiedBox64Limits,
    ConstrainedZonotope64, ConstrainedZonotopeAffineLimits,
    ConstrainedZonotopeAlphaBisectionBudgetError, ConstrainedZonotopeAlphaBisectionLimits,
    ConstrainedZonotopeBatchNormAffineCertificateLimits, ConstrainedZonotopeBatchNormBudgetError,
    ConstrainedZonotopeBatchNormLimits, ConstrainedZonotopeBatchNormMode,
    ConstrainedZonotopeBatchNormSpec, ConstrainedZonotopeCallBudget,
    ConstrainedZonotopeCallBudgetError, ConstrainedZonotopeCallReport,
    ConstrainedZonotopeConv2dLimits, ConstrainedZonotopeConv2dSpec,
    ConstrainedZonotopeConvTranspose2dLimits, ConstrainedZonotopeConvTranspose2dSpec,
    ConstrainedZonotopeOrderReductionLimits, ExactBatchNormAffineSurrogateCertificate,
    ExactReluTailMargin, PreparedReluTailGeometry64, ReluTailBoxCutAdamSchedule,
    ReluTailBoxCutOptimizerConfig, ReluTailBoxCutOptimizerLimits, ReluTailBoxCutOptimizerPlan,
    ReluTailBoxCutOptimizerStatus, ReluTailBoxCutSelection, ReluTailBoxCutStatus,
    ReluTailConvBatchNormPullbackBudgetError, ReluTailConvBatchNormPullbackError,
    ReluTailConvBatchNormPullbackLimits, ReluTailConvBatchNormPullbackM17M20Result,
    ReluTailConvBatchNormPullbackPlan, ReluTailDualBudgetError, ReluTailDualConfig,
    ReluTailDualLimits, ReluTailDualResult, ReluTailDualStatus, ReluTransformLimits,
    CONSTRAINED_ZONOTOPE_MAX_ITEMS_PER_POLL,
};
use ny_onnx::vnnlib::{CertifiedInputBox, CertifiedScalarMoat};
use ny_onnx::{AttributeValue, DataType, LayerSpec, OnnxModel};
use ny_propagate::{GraphNetwork, Layer, NETWORK_INPUT};

const PROTECTED_LATENT_SYMBOLS: usize = 5;
const PROTECTED_LATENT_LEAF_DOMAINS: usize = 1 << PROTECTED_LATENT_SYMBOLS;
const CGAN_AUTHORED_TENSOR_RANK: usize = 4;
const ONNX_BATCH_NORM_INPUT_RANK_ATTR: &str = "__onnx_batch_norm_input_rank";
const EXPECTED_NODE_COUNT: usize = 26;
const MAX_RUNNER_STAGES: usize = 40;
const FULL_RUNNER_COMPLETED_STAGES: usize = 29;
const RUNNER_STAGE_RESERVED_BYTES: usize = 256;
const RUNNER_TELEMETRY_RESERVED_BYTES: usize = MAX_RUNNER_STAGES * RUNNER_STAGE_RESERVED_BYTES;
const TAIL_RATIONAL_LIVE_BYTES: usize = 64 * 1024;
const TAIL_TRANSIENT_RATIONAL_SLOTS: usize = 16;
const TAIL_VALUE_DIM: usize = 512;
const TAIL_CHANNELS: usize = 128;
const TAIL_VALUES_PER_CHANNEL: usize = 4;
const TAIL_DISCRIMINATOR_RETAINED_ALPHA_DIM: usize = 512;
const TAIL_M17_WALL_TIME: Duration = Duration::from_secs(5);
const TAIL_DEPTH_TWO_MEASUREMENT_WALL_TIME: Duration = Duration::from_secs(2);
const TAIL_DEPTH_TWO_PUBLICATION_GUARD: Duration = Duration::from_millis(250);
const TAIL_DEPTH_TWO_INPUT_SHAPE: [usize; 3] = [64, 4, 4];
const TAIL_DEPTH_TWO_OUTPUT_SHAPE: [usize; 3] = [128, 2, 2];
const TAIL_DEPTH_TWO_INPUT_VALUES: usize = 1_024;
const TAIL_DEPTH_TWO_OUTPUT_VALUES: usize = TAIL_VALUE_DIM;
const TAIL_DEPTH_TWO_WEIGHT_SHAPE: [usize; 4] = [128, 64, 3, 3];
const TAIL_DEPTH_TWO_WEIGHT_ELEMENTS: usize = 128 * 64 * 3 * 3;
const TAIL_DEPTH_TWO_KERNEL_VISITS: usize = TAIL_VALUE_DIM * 64 * 3 * 3;
const TAIL_DEPTH_TWO_EXACT_PRODUCTS: usize = TAIL_DEPTH_TWO_KERNEL_VISITS
    + TAIL_DEPTH_TWO_OUTPUT_VALUES
    + 3 * TAIL_DEPTH_TWO_INPUT_VALUES
    + 3 * TAIL_DEPTH_TWO_INPUT_SHAPE[0];
const TAIL_M24_WALL_TIME: Duration = Duration::from_secs(1);
const TAIL_M24_SCHEDULE_ITERATIONS: usize = 4;
const TAIL_M24_MAX_BOX_VARIABLES: usize = 1_024;
const TAIL_M24_MAX_TOTAL_ITERATIONS: usize = 8;
const TAIL_M24_MAX_RESTARTS: usize = 2;
const TAIL_M24_MAX_EXACT_REPLAYS: usize = 2;
const TAIL_M24_MAX_GENERATOR_NONZEROS: usize = 150_000;
const TAIL_M24_MAX_SEARCH_WORK: u64 = 3_104_960;
const RUNNER_HARD_MAX_GRAPH_NODES: usize = 256;
const RUNNER_HARD_MAX_GRAPH_EDGES: usize = 1_024;
const RUNNER_HARD_MAX_TOPOLOGY_WORK: usize = 1 << 20;
const RUNNER_HARD_MAX_PARAMETER_ELEMENTS: usize = 2_000_000;
const CGAN_SEALED_MAX_VALUE_DIM: usize = 28_800;
const CGAN_RELU_INDICES: [usize; 7] = [5, 8, 11, 14, 16, 19, 22];
const CGAN_RELU_COUNT: usize = CGAN_RELU_INDICES.len();
const CGAN_CORRELATED_AUXILIARY_RELU_COUNT: usize = CGAN_RELU_COUNT - 1;

/// Exact authored image-channel profile admitted by the imgSz32 cGAN seal.
///
/// This is deliberately not an arbitrary channel count. The two variants are
/// the complete regular imgSz32 architectures shared by the 2025 and 2026
/// cGAN benchmarks; imgSz64 has a different node chain and is not admitted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CganCzImgSz32Profile {
    Nch1,
    Nch3,
}

impl CganCzImgSz32Profile {
    /// Number of channels in the generated 32x32 image.
    #[must_use]
    pub const fn image_channels(self) -> usize {
        match self {
            Self::Nch1 => 1,
            Self::Nch3 => 3,
        }
    }
}

const EXPECTED_NODES: [(&str, LayerType); EXPECTED_NODE_COUNT] = [
    ("Gemm_0", LayerType::Linear),
    ("Reshape_2", LayerType::Reshape),
    ("BatchNormalization_3", LayerType::BatchNorm),
    ("ConvTranspose_4", LayerType::ConvTranspose2d),
    ("BatchNormalization_5", LayerType::BatchNorm),
    ("Relu_6", LayerType::ReLU),
    ("ConvTranspose_7", LayerType::ConvTranspose2d),
    ("BatchNormalization_8", LayerType::BatchNorm),
    ("Relu_9", LayerType::ReLU),
    ("ConvTranspose_10", LayerType::ConvTranspose2d),
    ("BatchNormalization_11", LayerType::BatchNorm),
    ("Relu_12", LayerType::ReLU),
    ("ConvTranspose_13", LayerType::ConvTranspose2d),
    ("Conv_14", LayerType::Conv2d),
    ("Relu_15", LayerType::ReLU),
    ("Conv_16", LayerType::Conv2d),
    ("Relu_17", LayerType::ReLU),
    ("BatchNormalization_18", LayerType::BatchNorm),
    ("Conv_19", LayerType::Conv2d),
    ("Relu_20", LayerType::ReLU),
    ("BatchNormalization_21", LayerType::BatchNorm),
    ("Conv_22", LayerType::Conv2d),
    ("Relu_23", LayerType::ReLU),
    ("BatchNormalization_24", LayerType::BatchNorm),
    ("Reshape_26", LayerType::Reshape),
    ("Gemm_27", LayerType::Linear),
];

/// The exact nCh1 unbatched output shape after each accepted node.
///
/// Correlated entry points remain deliberately fixed to this legacy profile.
fn expected_output_shape(index: usize) -> &'static [usize] {
    expected_output_shape_for_profile(CganCzImgSz32Profile::Nch1, index)
}

/// The exact unbatched output shape for one sealed imgSz32 profile.
fn expected_output_shape_for_profile(
    profile: CganCzImgSz32Profile,
    index: usize,
) -> &'static [usize] {
    match index {
        0 => &[512],
        1 | 2 => &[128, 2, 2],
        3..=5 => &[128, 6, 6],
        6..=8 => &[64, 14, 14],
        9..=11 => &[32, 30, 30],
        12 => match profile {
            CganCzImgSz32Profile::Nch1 => &[1, 32, 32],
            CganCzImgSz32Profile::Nch3 => &[3, 32, 32],
        },
        13 | 14 => &[16, 16, 16],
        15..=17 => &[32, 8, 8],
        18..=20 => &[64, 4, 4],
        21..=23 => &[128, 2, 2],
        24 => &[512],
        25 => &[1],
        _ => &[],
    }
}

/// Caller-tightenable limits for one sealed imgSz32 cGAN probe.
///
/// There is intentionally no `Default`: a qualification caller must select
/// every count ceiling and the retained correlation width explicitly.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CganCzSequentialLimits {
    pub max_graph_nodes: usize,
    pub max_graph_edges: usize,
    pub max_topology_work_items: usize,
    pub max_parameter_elements: usize,
    pub max_value_dim: usize,
    pub max_transient_alpha_dim: usize,
    pub retained_alpha_dim: usize,
    pub max_generator_nonzeros: usize,
    pub max_interval_products_per_stage: usize,
    pub max_exact_terms_per_relu: usize,
    pub max_m17_iterations: usize,
    pub max_m17_search_work: u64,
}

/// Verdict authority is deliberately unavailable to this probe.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CganCzVerdictAuthority {
    DisabledPendingExactMoatReplay,
}

/// The only authority value this unwired module can publish.
pub const CGAN_CZ_VERDICT_AUTHORITY: CganCzVerdictAuthority =
    CganCzVerdictAuthority::DisabledPendingExactMoatReplay;

/// One auditable runner operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CganCzStageKind {
    Affine,
    Reshape,
    BatchNorm,
    ConvTranspose2d,
    Conv2d,
    Relu,
    OrderReduction,
    OutputTailConstruction,
    M17Lower,
    M17Upper,
}

/// Completed-stage resource and correlation telemetry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CganCzStageTelemetry {
    pub node: &'static str,
    pub kind: CganCzStageKind,
    pub output_shape: Vec<usize>,
    pub input_alpha_dim: usize,
    pub output_alpha_dim: usize,
    pub input_generator_nonzeros: usize,
    pub output_generator_nonzeros: usize,
    pub unstable_coordinates: usize,
    pub discarded_generators: usize,
    pub peak_live_bytes: usize,
    pub charged_items: usize,
    pub deadline_polls: usize,
}

/// Typed, fail-closed reason why the diagnostic probe did not publish bounds.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum CganCzProbeDecline {
    #[error("unsupported cGAN topology: {message}")]
    Topology { message: String },
    #[error("missing or changed cGAN provenance: {message}")]
    Provenance { message: String },
    #[error("invalid cGAN resource limit: {message}")]
    InvalidLimit { message: String },
    #[error("{resource} requires {required}, exceeding limit {limit}")]
    ResourceLimit {
        resource: &'static str,
        required: usize,
        limit: usize,
    },
    #[error("resource size overflow while computing {operation}")]
    ResourceOverflow { operation: &'static str },
    #[error(transparent)]
    Budget(#[from] ConstrainedZonotopeCallBudgetError),
    #[error("{node} {operation} declined: {message}")]
    Transform {
        node: &'static str,
        operation: &'static str,
        message: String,
    },
    #[error("M17/M20/M24 tail construction declined: {message}")]
    OutputTail { message: String },
}

/// Diagnostic bounds when the complete cooperative probe finishes.
#[derive(Clone, Debug, PartialEq)]
pub struct CganCzCompletedBounds {
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub low_unsafe_threshold: f64,
    pub high_unsafe_threshold: f64,
    pub separates_unsafe_moat: bool,
    pub bn_tail_correction_upper: f64,
    pub lower_m17_status: ReluTailDualStatus,
    pub upper_m17_status: ReluTailDualStatus,
    /// Candidate attribution for the certified lower bound on `Y_0`.
    pub lower_m17_candidates: CganCzM17CandidateTelemetry,
    /// Candidate attribution for the certified lower bound on `-Y_0`.
    /// Negating `selected_lower_bound` recovers this record's upper bound.
    pub negated_upper_m17_candidates: CganCzM17CandidateTelemetry,
    /// Optional M20 certificate for `Y_0` over an independently certified
    /// preactivation Box intersected with the correlated tail domain.
    pub lower_m20_lower_bound: Option<f64>,
    /// Optional M20 certificate for `-Y_0`.
    pub negated_upper_m20_lower_bound: Option<f64>,
    /// Whether the lower-row M20 member was absent, completed, or fell back.
    pub lower_m20_status: CganCzM20Status,
    /// Whether the negated-upper M20 member was absent, completed, or fell back.
    pub negated_upper_m20_status: CganCzM20Status,
    /// Verdict-neutral M24 observation for `Y_0`; absent only when no
    /// authenticated auxiliary preactivation bounds were supplied.
    pub lower_m24_measurement: Option<CganCzM24Measurement>,
    /// Verdict-neutral M24 observation for `-Y_0`; absent only when no
    /// authenticated auxiliary preactivation bounds were supplied.
    pub negated_upper_m24_measurement: Option<CganCzM24Measurement>,
}

/// Receipt status for the optional, independently replayed M20 member.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CganCzM20Status {
    /// The caller supplied no authenticated auxiliary preactivation bounds.
    NotRequested,
    /// M20 completed and its scalar certificate is present.
    Completed,
    /// Optional M20 work declined; the fully accounted M17 member remains.
    Fallback,
}

/// Verdict-neutral observation from the fixed cGAN M24 Box-cut experiment.
///
/// `counterfactual_lower_bound` and `counterfactual_selection` describe what
/// the exact M17/M20/M24 portfolio would retain. They never replace the
/// historical strict M17/M20 selector used by any published scalar bound.
#[derive(Clone, Debug, PartialEq)]
pub struct CganCzM24Measurement {
    /// Exact outward-replayed Box-cut certificate, when one completed.
    pub exact_box_cut_lower_bound: Option<f64>,
    /// Best exact certificate in the counterfactual M17/M20/M24 portfolio.
    pub counterfactual_lower_bound: f64,
    /// Exact member attaining `counterfactual_lower_bound` under stable ties.
    pub counterfactual_selection: ReluTailBoxCutSelection,
    /// Exact auxiliary/Box replay status from the core portfolio.
    pub replay_status: ReluTailBoxCutStatus,
    /// Candidate-search status; approximate scores have no proof authority.
    pub search_status: ReluTailBoxCutOptimizerStatus,
    /// Checked search plan, absent when setup or validation declined first.
    pub search_plan: Option<ReluTailBoxCutOptimizerPlan>,
    pub iterations_completed: usize,
    pub restarts_completed: usize,
    pub candidates_scored: usize,
    pub exact_replays: usize,
    /// Optional-lane shared-firewall refusal, retaining earlier certificates.
    pub optional_budget_error: Option<ConstrainedZonotopeCallBudgetError>,
}

/// Coarse, allocation-free attribution for an optional depth-two transform refusal.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CganCzDepthTwoTransformFailure {
    Setup,
    Conv2d,
    BatchNorm,
    ReluTail,
}

/// Compact completed observation from one transactional depth-two replay.
///
/// The historical M17/M20 member remains authoritative.  The counterfactual
/// field is telemetry only and uses a strict ordered maximum, so ties retain
/// the historical member.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CganCzDepthTwoCompletedMeasurement {
    pub(crate) historical_lower_bound: f64,
    pub(crate) downstream_m17_candidates: CganCzM17CandidateTelemetry,
    pub(crate) upstream_m17_candidates: CganCzM17CandidateTelemetry,
    pub(crate) upstream_m20_lower_bound: Option<f64>,
    pub(crate) upstream_m20_status: CganCzM20Status,
    /// Shared-firewall refusal from optional upstream M20. `None` also covers
    /// transform/disjoint auxiliary fallback, whose error is intentionally
    /// not promoted into a proof input.
    pub(crate) upstream_m20_optional_budget_error: Option<ConstrainedZonotopeCallBudgetError>,
    pub(crate) upstream_m17_m20_selection: ReluTailBoxCutSelection,
    pub(crate) counterfactual_lower_bound: f64,
    pub(crate) signed_gain: f64,
    pub(crate) plan: ReluTailConvBatchNormPullbackPlan,
    pub(crate) peak_live_bytes: usize,
    pub(crate) charged_items: usize,
    pub(crate) deadline_polls: usize,
}

/// Verdict-neutral status of one optional exact depth-two row replay.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum CganCzDepthTwoMeasurement {
    /// No authenticated Relu_20 -> BN_21 -> Conv_22 handoff was retained.
    ///
    /// The production leaf route deliberately emits this state: retaining the
    /// handoff through the historical tail would raise its absolute peak-live
    /// threshold. Re-enabling the experiment requires a post-publication
    /// second pass that recomputes the handoff without changing mandatory
    /// finite-cap availability.
    NotRequested,
    /// Too little caller time remained after both historical portfolios.
    NoTime,
    /// The optional sub-call firewall declined without affecting authority.
    BudgetFallback(ConstrainedZonotopeCallBudgetError),
    /// Optional exact setup or transform validation declined.
    TransformFallback(CganCzDepthTwoTransformFailure),
    /// Both linked M17 certificates completed; upstream M20 may have fallen
    /// back without suppressing its mandatory M17 sibling.
    Completed(CganCzDepthTwoCompletedMeasurement),
}

/// Verdict-neutral attribution for one independently replayed M17 portfolio.
///
/// All bounds in this record are already outward-replayed certificate values;
/// projected Adam remains candidate search only.  The telemetry makes a
/// bounded iteration sweep diagnostic: if `optimized_improvement` is zero,
/// deeper slope search cannot be credited for the selected certificate and a
/// caller should prefer earlier-relaxation refinement over more iterations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CganCzM17CandidateTelemetry {
    pub selected_lower_bound: f64,
    pub zero_positive_slope_lower_bound: f64,
    pub upper_endpoint_lower_bound: Option<f64>,
    pub canonical_lower_bound: Option<f64>,
    pub optimized_lower_bound: Option<f64>,
    pub best_nonoptimized_lower_bound: f64,
    pub optimized_improvement: f64,
    pub optimizable_slopes: usize,
    pub candidates_replayed: usize,
    pub iterations_completed: usize,
    pub status: ReluTailDualStatus,
}

/// Auditable state after the requested generator-block prefix.
///
/// Widths are outward-rounded binary64 diagnostics over the unconstrained
/// coordinate hull. They are telemetry only and never verdict authority.
#[derive(Clone, Debug, PartialEq)]
pub struct CganCzPrefixCompletion {
    pub last_node: &'static str,
    pub output_shape: Vec<usize>,
    pub value_dim: usize,
    pub alpha_dim: usize,
    pub generator_nonzeros: usize,
    pub maximum_coordinate_width: f64,
    pub mean_coordinate_width: f64,
    pub maximum_box_remainder: f64,
    pub low_unsafe_threshold: f64,
    pub high_unsafe_threshold: f64,
}

/// Caller-selected limits for a complete protected-alpha dyadic cover.
///
/// There is deliberately no `Default`. This queue is an unwired experiment,
/// and callers must choose both the complete-tree ceilings and every limit of
/// the underlying exact-dyadic bisection primitive.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CganCzProtectedAlphaCoverLimits {
    /// Leading alpha columns that the supplied split plan may address.
    pub protected_alpha_dim: usize,
    /// Maximum number of breadth-first split levels.
    pub max_split_levels: usize,
    /// Maximum nodes in the complete binary cover, including the root.
    pub max_tree_nodes: usize,
    /// Maximum leaf domains published by a completed cover.
    pub max_leaf_domains: usize,
    /// Per-node limits passed to the NY-MIP bisection primitive.
    pub bisection: ConstrainedZonotopeAlphaBisectionLimits,
}

/// Limits for propagating the complete protected-latent cover.
///
/// There is deliberately no `Default`. The cover and every sequential
/// transform retain their own explicit ceilings, while
/// `max_leaf_propagations` separately caps the amount of complete-model work
/// and the number of private completion records retained before publication.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CganCzProtectedAlphaProbeLimits {
    pub sequential: CganCzSequentialLimits,
    pub cover: CganCzProtectedAlphaCoverLimits,
    pub max_leaf_propagations: usize,
}

/// Aggregate accounting published only after the complete cover exists.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CganCzProtectedAlphaCoverReport {
    split_levels: usize,
    tree_nodes: usize,
    split_calls: usize,
    leaf_domains: usize,
    peak_live_bytes: usize,
    charged_items: usize,
    deadline_polls: usize,
}

impl CganCzProtectedAlphaCoverReport {
    #[must_use]
    pub const fn split_levels(self) -> usize {
        self.split_levels
    }

    #[must_use]
    pub const fn tree_nodes(self) -> usize {
        self.tree_nodes
    }

    #[must_use]
    pub const fn split_calls(self) -> usize {
        self.split_calls
    }

    #[must_use]
    pub const fn leaf_domains(self) -> usize {
        self.leaf_domains
    }

    #[must_use]
    pub const fn peak_live_bytes(self) -> usize {
        self.peak_live_bytes
    }

    #[must_use]
    pub const fn charged_items(self) -> usize {
        self.charged_items
    }

    #[must_use]
    pub const fn deadline_polls(self) -> usize {
        self.deadline_polls
    }
}

/// A complete breadth-first dyadic cover of the requested protected symbols.
///
/// Leaf order is deterministic: each level visits the prior frontier in
/// order and appends the negative child before the positive child. Alpha
/// dimension and column order are unchanged in every leaf.
#[derive(Clone, Debug, PartialEq)]
pub struct CganCzProtectedAlphaCover {
    split_axes: Vec<usize>,
    leaves: Vec<ConstrainedZonotope64>,
    report: CganCzProtectedAlphaCoverReport,
}

impl CganCzProtectedAlphaCover {
    #[must_use]
    pub fn split_axes(&self) -> &[usize] {
        &self.split_axes
    }

    #[must_use]
    pub fn leaves(&self) -> &[ConstrainedZonotope64] {
        &self.leaves
    }

    #[must_use]
    pub const fn report(&self) -> CganCzProtectedAlphaCoverReport {
        self.report
    }

    #[must_use]
    pub fn into_leaves(self) -> Vec<ConstrainedZonotope64> {
        self.leaves
    }
}

/// One leaf receipt retained privately until the complete cover finishes.
#[derive(Clone, Debug, PartialEq)]
pub struct CganCzProtectedAlphaLeafCompletion {
    pub leaf_index: usize,
    pub bounds: CganCzCompletedBounds,
    pub completed_stages: usize,
    pub peak_live_bytes: usize,
    pub charged_items: usize,
    pub deadline_polls: usize,
}

/// Diagnostic hull of every completed protected-latent leaf.
///
/// `leaf_completions` is in the cover's deterministic negative-child-first
/// order. This value cannot exist unless every leaf completed the full sealed
/// path and the final publication checkpoint succeeded.
#[derive(Clone, Debug, PartialEq)]
pub struct CganCzProtectedAlphaAggregateBounds {
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub low_unsafe_threshold: f64,
    pub high_unsafe_threshold: f64,
    pub separates_unsafe_moat: bool,
    pub cover: CganCzProtectedAlphaCoverReport,
    pub leaf_completions: Vec<CganCzProtectedAlphaLeafCompletion>,
}

/// All-or-nothing status of the complete protected-latent experiment.
#[derive(Clone, Debug, PartialEq)]
pub enum CganCzProtectedAlphaProbeStatus {
    Completed(CganCzProtectedAlphaAggregateBounds),
    Declined {
        /// `None` means admission, sealing, input construction, or cover
        /// enumeration failed before any leaf was selected.
        leaf_index: Option<usize>,
        node: &'static str,
        reason: CganCzProbeDecline,
    },
}

/// In-process receipt for the unwired complete-cover probe.
#[derive(Clone, Debug, PartialEq)]
pub struct CganCzProtectedAlphaProbeReport {
    pub authority: CganCzVerdictAuthority,
    pub status: CganCzProtectedAlphaProbeStatus,
    pub topology_work_items: usize,
    pub parameter_elements: usize,
    pub protected_latent_symbols: usize,
    pub requested_leaf_domains: usize,
    pub peak_live_bytes: usize,
    pub charged_items: usize,
    pub deadline_polls: usize,
}

/// The protected-prefix plan for the five authored cGAN latent symbols.
///
/// Applying one zero bisection to each leading symbol yields all 32 orthants;
/// repeated entries may be supplied to the generic queue for deeper dyadic
/// refinement without changing alpha order.
pub const CGAN_NCH1_PROTECTED_LATENT_COVER_AXES: [usize; PROTECTED_LATENT_SYMBOLS] =
    [0, 1, 2, 3, 4];

/// Probe outcome. A completed bound is still diagnostic-only.
// Keep transactional receipts inline: boxing would add an unbudgeted,
// fallible allocation at the publication boundary.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq)]
pub enum CganCzProbeStatus {
    /// The requested real-model prefix completed; no output bound was formed.
    PrefixCompleted(CganCzPrefixCompletion),
    Completed(CganCzCompletedBounds),
    Declined {
        node: &'static str,
        reason: CganCzProbeDecline,
    },
}

/// In-process telemetry for one sealed probe invocation.
///
/// This proves what happened to the model/property objects supplied to the
/// current call, but it is not a durable artifact receipt: it intentionally
/// carries no source-file digest. Archival evidence must bind the official
/// model, property, limits, and budget outside this diagnostic-only module.
#[derive(Clone, Debug, PartialEq)]
pub struct CganCzProbeReport {
    pub authority: CganCzVerdictAuthority,
    pub status: CganCzProbeStatus,
    pub stages: Vec<CganCzStageTelemetry>,
    pub topology_work_items: usize,
    pub parameter_elements: usize,
    pub protected_latent_symbols: usize,
    pub peak_live_bytes: usize,
    pub charged_items: usize,
    pub deadline_polls: usize,
}

/// One certified preactivation box captured by the independent interval-CZ
/// lane at an authored ReLU.
///
/// The node and shape are fixed by the selected exact 26-node imgSz32 profile.
/// The bounds are intentionally pre-ReLU. The exact property-bound leaf route
/// consumes the first six entries at their corresponding correlated ReLUs and
/// retains the final `Relu_23` entry for M20, without deriving any auxiliary
/// hull from the correlated domain itself.
#[derive(Clone, Debug, PartialEq)]
pub struct CganCzIndependentIntervalReluBounds {
    node: &'static str,
    output_shape: &'static [usize],
    bounds: CertifiedAuxiliaryBounds64,
}

impl CganCzIndependentIntervalReluBounds {
    /// Authored ReLU node owning these preactivation bounds.
    #[must_use]
    pub const fn node(&self) -> &'static str {
        self.node
    }

    /// Exact unbatched shape sealed for this node.
    #[must_use]
    pub const fn output_shape(&self) -> &'static [usize] {
        self.output_shape
    }

    /// Certified preactivation interval endpoints.
    #[must_use]
    pub const fn bounds(&self) -> &CertifiedAuxiliaryBounds64 {
        &self.bounds
    }
}

/// Complete output of the independent remainder-only interval-CZ lane.
///
/// Publication is all-or-nothing: exactly seven pre-ReLU boxes are present in
/// authored order, followed by the zero-symbol/no-predicate domain obtained by
/// applying and recentering `Relu_23`.
#[derive(Clone, Debug, PartialEq)]
pub struct CganCzIndependentIntervalCompletion {
    relu_bounds: Vec<CganCzIndependentIntervalReluBounds>,
    post_relu_23: ConstrainedZonotope64,
}

impl CganCzIndependentIntervalCompletion {
    /// All seven authored pre-ReLU bounds in execution order.
    #[must_use]
    pub fn relu_bounds(&self) -> &[CganCzIndependentIntervalReluBounds] {
        &self.relu_bounds
    }

    /// Certified auxiliary bounds for the final 512-coordinate `Relu_23`.
    #[must_use]
    pub fn final_relu_23_auxiliary_bounds(&self) -> &CertifiedAuxiliaryBounds64 {
        // Construction checks the exact count and authored order before this
        // type crosses the publication boundary.
        &self.relu_bounds[CGAN_RELU_COUNT - 1].bounds
    }

    /// Independent interval-CZ after applying and recentering `Relu_23`.
    #[must_use]
    pub const fn post_relu_23_domain(&self) -> &ConstrainedZonotope64 {
        &self.post_relu_23
    }
}

/// All-or-nothing status of the independent interval-CZ diagnostic lane.
// Keep transactional receipts inline; see `CganCzProbeStatus`.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq)]
pub enum CganCzIndependentIntervalStatus {
    Completed(CganCzIndependentIntervalCompletion),
    Declined {
        node: &'static str,
        reason: CganCzProbeDecline,
    },
}

/// In-process receipt for the default-unwired independent interval-CZ lane.
///
/// On completion, the counters aggregate every nested transform receipt. On a
/// decline, they cover coordinator work plus nested calls that completed
/// before the refusal. A failing nested call still enforces its own deadline
/// and peak cap, but its error type does not expose a partial receipt, so these
/// diagnostic counters intentionally do not claim attempted-work exactness for
/// declined invocations.
#[derive(Clone, Debug, PartialEq)]
pub struct CganCzIndependentIntervalReport {
    pub authority: CganCzVerdictAuthority,
    /// Exact imgSz32 architecture selected before topology sealing.
    pub profile: CganCzImgSz32Profile,
    pub status: CganCzIndependentIntervalStatus,
    pub topology_work_items: usize,
    pub parameter_elements: usize,
    pub peak_live_bytes: usize,
    pub charged_items: usize,
    pub deadline_polls: usize,
}

impl CganCzIndependentIntervalReport {
    /// Authored generated-image channel count carried by this receipt.
    #[must_use]
    pub const fn image_channels(&self) -> usize {
        self.profile.image_channels()
    }
}

/// Certified scalar-row bounds from one exact binary32 imgSz32 input leaf.
///
/// `lower_y` encloses a lower bound on the original scalar `Y` output and
/// `lower_neg_y` encloses a lower bound on `-Y`.  This value remains
/// verdict-neutral: only a property-bound caller may compare both rows against
/// its own authenticated thresholds.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CganCzLeafRowBounds {
    pub(crate) lower_y: f64,
    pub(crate) lower_neg_y: f64,
    pub(crate) bn_tail_correction_upper: f64,
    pub(crate) lower_m17_status: ReluTailDualStatus,
    pub(crate) negated_upper_m17_status: ReluTailDualStatus,
    pub(crate) lower_m17_candidates: CganCzM17CandidateTelemetry,
    pub(crate) negated_upper_m17_candidates: CganCzM17CandidateTelemetry,
    pub(crate) lower_m20_lower_bound: Option<f64>,
    pub(crate) negated_upper_m20_lower_bound: Option<f64>,
    pub(crate) lower_m20_status: CganCzM20Status,
    pub(crate) negated_upper_m20_status: CganCzM20Status,
    pub(crate) lower_m24_measurement: Option<CganCzM24Measurement>,
    pub(crate) negated_upper_m24_measurement: Option<CganCzM24Measurement>,
    pub(crate) lower_depth_two_measurement: CganCzDepthTwoMeasurement,
    pub(crate) negated_upper_depth_two_measurement: CganCzDepthTwoMeasurement,
}

struct CganCzTailPortfolio {
    selected_lower_bound: f64,
    m17_candidates: CganCzM17CandidateTelemetry,
    m20_lower_bound: Option<f64>,
    m20_status: CganCzM20Status,
    m24_measurement: Option<CganCzM24Measurement>,
}

/// Private tail result keeps experimental depth-two telemetry out of the
/// public diagnostic-bounds construction API.  Only the authenticated leaf
/// attachment consumes these observations.
struct CganCzOutputTailResult {
    completed: CganCzCompletedBounds,
    lower_depth_two_measurement: CganCzDepthTwoMeasurement,
    negated_upper_depth_two_measurement: CganCzDepthTwoMeasurement,
}

/// Private by-construction seal for the exact authored depth-two handoff.
///
/// This dormant value documents the seal required by a future second-pass
/// replay. The production leaf route does not construct it while retaining
/// `Relu_20` across the historical tail would violate cap monotonicity.
#[allow(dead_code)]
#[derive(Clone, Copy)]
struct CganCzDepthTwoContext<'a> {
    profile: CganCzImgSz32Profile,
    upstream: &'a ConstrainedZonotope64,
    upstream_auxiliary: &'a CganCzIndependentIntervalReluBounds,
    downstream: &'a ConstrainedZonotope64,
    batch_norm_21: &'a BatchNormParameters,
    conv_22: &'a Conv2dParameters,
}

/// Transactional outcome of the private imgSz32 leaf-row bound API.
// Keep transactional receipts inline; see `CganCzProbeStatus`.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum CganCzLeafRowStatus {
    Completed(CganCzLeafRowBounds),
    Declined {
        node: &'static str,
        reason: CganCzProbeDecline,
    },
}

/// Typed deadline and memory receipt for one imgSz32 leaf-row attempt.
///
/// The authority field is permanently disabled.  A completed report proves
/// only the two scalar bounds over the supplied input leaf; it cannot itself
/// authorize a verifier verdict.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CganCzLeafRowReport {
    pub(crate) authority: CganCzVerdictAuthority,
    pub(crate) profile: CganCzImgSz32Profile,
    pub(crate) deadline: Instant,
    pub(crate) baseline_live_bytes: usize,
    pub(crate) max_peak_live_bytes: usize,
    pub(crate) status: CganCzLeafRowStatus,
    pub(crate) topology_work_items: usize,
    pub(crate) parameter_elements: usize,
    pub(crate) peak_live_bytes: usize,
    pub(crate) charged_items: usize,
    pub(crate) deadline_polls: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct CganCzLeafInput64 {
    lower: [f64; PROTECTED_LATENT_SYMBOLS],
    upper: [f64; PROTECTED_LATENT_SYMBOLS],
}

struct CganCzIndependentPrefixPrivate {
    sealed: SealedCgan,
    stages: Vec<CganCzStageTelemetry>,
    relu_bounds: Vec<CganCzIndependentIntervalReluBounds>,
    post_relu_23: ConstrainedZonotope64,
    retained_baseline: usize,
}

struct Coordinator<N> {
    budget: ConstrainedZonotopeCallBudget,
    now: N,
    peak_live_bytes: usize,
    charged_items: usize,
    items_since_poll: usize,
    deadline_polls: usize,
    topology_work_items: usize,
}

impl<N> Coordinator<N>
where
    N: FnMut(&'static str) -> Instant,
{
    fn new(budget: ConstrainedZonotopeCallBudget, now: N) -> Self {
        Self {
            budget,
            now,
            peak_live_bytes: budget.baseline_live_bytes(),
            charged_items: 0,
            items_since_poll: 0,
            deadline_polls: 0,
            topology_work_items: 0,
        }
    }

    fn checkpoint_time(&mut self, checkpoint: &'static str) -> Result<Instant, CganCzProbeDecline> {
        self.deadline_polls =
            self.deadline_polls
                .checked_add(1)
                .ok_or(CganCzProbeDecline::ResourceOverflow {
                    operation: "cGAN deadline poll count",
                })?;
        let now = (self.now)(checkpoint);
        if now >= self.budget.deadline() {
            return Err(ConstrainedZonotopeCallBudgetError::DeadlineExpired { checkpoint }.into());
        }
        self.items_since_poll = 0;
        Ok(now)
    }

    fn checkpoint(&mut self, checkpoint: &'static str) -> Result<(), CganCzProbeDecline> {
        self.checkpoint_time(checkpoint).map(|_| ())
    }

    fn charge(
        &mut self,
        mut items: usize,
        checkpoint: &'static str,
    ) -> Result<(), CganCzProbeDecline> {
        self.charged_items =
            self.charged_items
                .checked_add(items)
                .ok_or(CganCzProbeDecline::ResourceOverflow {
                    operation: "cGAN charged work items",
                })?;
        while items != 0 {
            let until_poll = CONSTRAINED_ZONOTOPE_MAX_ITEMS_PER_POLL - self.items_since_poll;
            let consumed = items.min(until_poll);
            self.items_since_poll += consumed;
            items -= consumed;
            if self.items_since_poll == CONSTRAINED_ZONOTOPE_MAX_ITEMS_PER_POLL {
                self.checkpoint(checkpoint)?;
            }
        }
        Ok(())
    }

    fn charge_topology(&mut self, items: usize) -> Result<(), CganCzProbeDecline> {
        self.topology_work_items = self.topology_work_items.checked_add(items).ok_or(
            CganCzProbeDecline::ResourceOverflow {
                operation: "cGAN topology work items",
            },
        )?;
        self.charge(items, "cGAN topology scan")
    }

    fn preflight_absolute_peak(&mut self, required: usize) -> Result<(), CganCzProbeDecline> {
        if required > self.budget.max_peak_live_bytes() {
            return Err(ConstrainedZonotopeCallBudgetError::PeakLiveBytesExceeded {
                required,
                limit: self.budget.max_peak_live_bytes(),
            }
            .into());
        }
        self.peak_live_bytes = self.peak_live_bytes.max(required);
        Ok(())
    }

    fn absorb(&mut self, report: ConstrainedZonotopeCallReport) -> Result<(), CganCzProbeDecline> {
        self.peak_live_bytes = self.peak_live_bytes.max(report.peak_live_bytes());
        self.charged_items = self
            .charged_items
            .checked_add(report.charged_items())
            .ok_or(CganCzProbeDecline::ResourceOverflow {
                operation: "aggregate cGAN charged work items",
            })?;
        self.deadline_polls = self
            .deadline_polls
            .checked_add(report.deadline_polls())
            .ok_or(CganCzProbeDecline::ResourceOverflow {
                operation: "aggregate cGAN deadline polls",
            })?;
        Ok(())
    }
}

/// Enumerate a complete dyadic cover over protected alpha columns.
///
/// Every entry of `split_axes` is applied to every leaf at that breadth-first
/// level. Repeating an axis therefore refines every current interval on that
/// symbol. No child is sampled, pruned, or interpreted as infeasible. The
/// function publishes leaves and accounting only after every requested split
/// and the final deadline checkpoint complete; any error drops the private
/// partial frontier.
pub fn enumerate_cgan_cz_protected_alpha_cover_unwired(
    input: &ConstrainedZonotope64,
    split_axes: &[usize],
    limits: CganCzProtectedAlphaCoverLimits,
    budget: ConstrainedZonotopeCallBudget,
) -> Result<CganCzProtectedAlphaCover, CganCzProbeDecline> {
    enumerate_cgan_cz_protected_alpha_cover_with_clock(input, split_axes, limits, budget, |_| {
        Instant::now()
    })
}

/// Apply the one-level plan for all five protected cGAN latent symbols.
///
/// This is the narrow seam from the authored cGAN prefix convention into the
/// generic queue. It remains in this explicitly unwired diagnostic module and
/// cannot publish or influence a verifier verdict.
pub fn enumerate_cgan_nch1_protected_latent_cover_unwired(
    input: &ConstrainedZonotope64,
    limits: CganCzProtectedAlphaCoverLimits,
    budget: ConstrainedZonotopeCallBudget,
) -> Result<CganCzProtectedAlphaCover, CganCzProbeDecline> {
    if limits.protected_alpha_dim != PROTECTED_LATENT_SYMBOLS {
        return Err(CganCzProbeDecline::InvalidLimit {
            message: format!(
                "cGAN protected-alpha cover requires protected_alpha_dim={}, got {}",
                PROTECTED_LATENT_SYMBOLS, limits.protected_alpha_dim
            ),
        });
    }
    enumerate_cgan_cz_protected_alpha_cover_unwired(
        input,
        &CGAN_NCH1_PROTECTED_LATENT_COVER_AXES,
        limits,
        budget,
    )
}

fn enumerate_cgan_cz_protected_alpha_cover_with_clock<N>(
    input: &ConstrainedZonotope64,
    split_axes: &[usize],
    limits: CganCzProtectedAlphaCoverLimits,
    budget: ConstrainedZonotopeCallBudget,
    now: N,
) -> Result<CganCzProtectedAlphaCover, CganCzProbeDecline>
where
    N: FnMut(&'static str) -> Instant,
{
    let queue_baseline = budget.baseline_live_bytes();
    let mut coordinator = Coordinator::new(budget, now);
    enumerate_cgan_cz_protected_alpha_cover_with_coordinator(
        input,
        split_axes,
        limits,
        queue_baseline,
        &mut coordinator,
    )
}

fn enumerate_cgan_cz_protected_alpha_cover_with_coordinator<N>(
    input: &ConstrainedZonotope64,
    split_axes: &[usize],
    limits: CganCzProtectedAlphaCoverLimits,
    queue_baseline: usize,
    coordinator: &mut Coordinator<N>,
) -> Result<CganCzProtectedAlphaCover, CganCzProbeDecline>
where
    N: FnMut(&'static str) -> Instant,
{
    let budget = coordinator.budget;
    let charged_start = coordinator.charged_items;
    let polls_start = coordinator.deadline_polls;
    validate_protected_alpha_cover(input, split_axes, limits)?;
    coordinator.checkpoint("protected-alpha cover admission")?;

    check_resource(
        "protected-alpha split levels",
        split_axes.len(),
        limits.max_split_levels,
    )?;
    let leaf_domains = checked_power_of_two(split_axes.len(), "protected-alpha leaf count")?;
    let tree_nodes = leaf_domains
        .checked_mul(2)
        .and_then(|nodes| nodes.checked_sub(1))
        .ok_or(CganCzProbeDecline::ResourceOverflow {
            operation: "protected-alpha tree node count",
        })?;
    let split_calls = leaf_domains
        .checked_sub(1)
        .ok_or(CganCzProbeDecline::ResourceOverflow {
            operation: "protected-alpha split call count",
        })?;
    check_resource(
        "protected-alpha tree nodes",
        tree_nodes,
        limits.max_tree_nodes,
    )?;
    check_resource(
        "protected-alpha leaf domains",
        leaf_domains,
        limits.max_leaf_domains,
    )?;

    let domain_body_upper = domain_live_bytes(input)?;
    let final_vector_slots =
        leaf_domains
            .checked_add(leaf_domains / 2)
            .ok_or(CganCzProbeDecline::ResourceOverflow {
                operation: "protected-alpha frontier slots",
            })?;
    // The final frontier is the largest set of retained domain bodies. Three
    // additional root-sized bodies conservatively cover an unpublished split
    // pair plus the selected parent while a primitive call is in flight. The
    // primitive independently performs its more detailed logical-peak check.
    let peak_domain_bodies =
        leaf_domains
            .checked_add(3)
            .ok_or(CganCzProbeDecline::ResourceOverflow {
                operation: "protected-alpha peak domain count",
            })?;
    let split_plan_bytes = split_axes.len().checked_mul(size_of::<usize>()).ok_or(
        CganCzProbeDecline::ResourceOverflow {
            operation: "protected-alpha split-plan storage",
        },
    )?;
    let conservative_peak = cover_queue_absolute_bytes(
        queue_baseline,
        final_vector_slots,
        peak_domain_bodies,
        domain_body_upper,
    )?
    .checked_add(split_plan_bytes)
    .ok_or(CganCzProbeDecline::ResourceOverflow {
        operation: "protected-alpha publication storage",
    })?;
    coordinator.preflight_absolute_peak(conservative_peak)?;

    let mut frontier: Option<Vec<ConstrainedZonotope64>> = None;
    for (level, &alpha_axis) in split_axes.iter().enumerate() {
        coordinator.checkpoint("protected-alpha cover level admission")?;
        let parent_count = checked_power_of_two(level, "protected-alpha parent count")?;
        let child_count =
            parent_count
                .checked_mul(2)
                .ok_or(CganCzProbeDecline::ResourceOverflow {
                    operation: "protected-alpha child count",
                })?;
        let mut next = Vec::new();
        next.try_reserve_exact(child_count)
            .map_err(|_| CganCzProbeDecline::ResourceLimit {
                resource: "protected-alpha child frontier allocation",
                required: child_count,
                limit: child_count.saturating_sub(1),
            })?;

        if level == 0 {
            let vector_slots = next.capacity();
            split_cover_parent(
                input,
                alpha_axis,
                limits.bisection,
                budget,
                queue_baseline,
                domain_body_upper,
                vector_slots,
                1,
                coordinator,
                &mut next,
            )?;
        } else {
            let current = frontier
                .take()
                .ok_or_else(|| CganCzProbeDecline::Transform {
                    node: "protected-alpha cover",
                    operation: "frontier transition",
                    message: "a non-root split level had no complete parent frontier".to_string(),
                })?;
            if current.len() != parent_count {
                return Err(CganCzProbeDecline::Transform {
                    node: "protected-alpha cover",
                    operation: "frontier cardinality",
                    message: format!(
                        "level {level} retained {} parents, expected {parent_count}",
                        current.len()
                    ),
                });
            }
            // Count allocator-returned capacities, not requested lengths:
            // `try_reserve_exact` is allowed to over-allocate either frontier.
            let vector_slots = current.capacity().checked_add(next.capacity()).ok_or(
                CganCzProbeDecline::ResourceOverflow {
                    operation: "protected-alpha simultaneous frontier slots",
                },
            )?;
            let mut parents = current.into_iter();
            while let Some(parent) = parents.next() {
                coordinator.checkpoint("protected-alpha cover node admission")?;
                let retained_domains = parents
                    .len()
                    .checked_add(1)
                    .and_then(|count| count.checked_add(next.len()))
                    .ok_or(CganCzProbeDecline::ResourceOverflow {
                        operation: "protected-alpha retained domain count",
                    })?;
                split_cover_parent(
                    &parent,
                    alpha_axis,
                    limits.bisection,
                    budget,
                    queue_baseline,
                    domain_body_upper,
                    vector_slots,
                    retained_domains,
                    coordinator,
                    &mut next,
                )?;
            }
        }
        if next.len() != child_count {
            return Err(CganCzProbeDecline::Transform {
                node: "protected-alpha cover",
                operation: "complete-cover audit",
                message: format!(
                    "level {level} produced {} children, expected {child_count}",
                    next.len()
                ),
            });
        }
        frontier = Some(next);
    }

    let leaves = frontier.ok_or_else(|| CganCzProbeDecline::InvalidLimit {
        message: "protected-alpha split plan must contain at least one level".to_string(),
    })?;
    if leaves.len() != leaf_domains {
        return Err(CganCzProbeDecline::Transform {
            node: "protected-alpha cover",
            operation: "publication cardinality",
            message: format!(
                "completed frontier contains {} leaves, expected {leaf_domains}",
                leaves.len()
            ),
        });
    }
    for leaf in &leaves {
        coordinator.charge(1, "protected-alpha leaf publication audit")?;
        if leaf.alpha_dim() != input.alpha_dim()
            || leaf.constraint_count() != input.constraint_count()
        {
            return Err(CganCzProbeDecline::Transform {
                node: "protected-alpha cover",
                operation: "alpha-order and predicate audit",
                message: "a leaf changed alpha dimension or predicate-row count".to_string(),
            });
        }
    }
    coordinator.checkpoint("protected-alpha cover publication")?;

    let mut published_axes = Vec::new();
    published_axes
        .try_reserve_exact(split_axes.len())
        .map_err(|_| CganCzProbeDecline::ResourceLimit {
            resource: "protected-alpha split-plan publication",
            required: split_axes.len(),
            limit: split_axes.len().saturating_sub(1),
        })?;
    let actual_split_plan_bytes = published_axes
        .capacity()
        .checked_mul(size_of::<usize>())
        .ok_or(CganCzProbeDecline::ResourceOverflow {
            operation: "protected-alpha actual split-plan storage",
        })?;
    // The final frontier is still retained while the split-plan receipt is
    // allocated. Rebuild this publication peak from both allocator-returned
    // capacities: the earlier conservative admission used requested frontier
    // lengths and cannot bound an arbitrary `try_reserve_exact` over-allocation.
    let actual_publication_peak = cover_queue_absolute_bytes(
        queue_baseline,
        leaves.capacity(),
        peak_domain_bodies,
        domain_body_upper,
    )?
    .checked_add(actual_split_plan_bytes)
    .ok_or(CganCzProbeDecline::ResourceOverflow {
        operation: "protected-alpha actual publication storage",
    })?;
    coordinator.preflight_absolute_peak(actual_publication_peak)?;
    published_axes.extend_from_slice(split_axes);
    coordinator.charge(split_axes.len(), "protected-alpha split-plan publication")?;
    coordinator.checkpoint("protected-alpha cover report publication")?;

    Ok(CganCzProtectedAlphaCover {
        split_axes: published_axes,
        leaves,
        report: CganCzProtectedAlphaCoverReport {
            split_levels: split_axes.len(),
            tree_nodes,
            split_calls,
            leaf_domains,
            peak_live_bytes: coordinator.peak_live_bytes,
            charged_items: coordinator.charged_items.checked_sub(charged_start).ok_or(
                CganCzProbeDecline::ResourceOverflow {
                    operation: "protected-alpha cover charged-item delta",
                },
            )?,
            deadline_polls: coordinator.deadline_polls.checked_sub(polls_start).ok_or(
                CganCzProbeDecline::ResourceOverflow {
                    operation: "protected-alpha cover deadline-poll delta",
                },
            )?,
        },
    })
}

#[allow(clippy::too_many_arguments)]
fn split_cover_parent<N>(
    parent: &ConstrainedZonotope64,
    alpha_axis: usize,
    bisection_limits: ConstrainedZonotopeAlphaBisectionLimits,
    budget: ConstrainedZonotopeCallBudget,
    queue_baseline: usize,
    domain_body_upper: usize,
    vector_slots: usize,
    retained_domains: usize,
    coordinator: &mut Coordinator<N>,
    next: &mut Vec<ConstrainedZonotope64>,
) -> Result<(), CganCzProbeDecline>
where
    N: FnMut(&'static str) -> Instant,
{
    coordinator.checkpoint("protected-alpha bisection dispatch")?;
    // `input` remains borrowed by the queue for the complete enumeration.
    // From the second level onward it is no longer one of the frontier
    // parents, so retain one additional root-sized body in every primitive
    // baseline. At level zero this deliberately double-counts the root rather
    // than making the accounting phase-dependent.
    let retained_with_root =
        retained_domains
            .checked_add(1)
            .ok_or(CganCzProbeDecline::ResourceOverflow {
                operation: "protected-alpha retained domains plus root",
            })?;
    let call_baseline = cover_queue_absolute_bytes(
        queue_baseline,
        vector_slots,
        retained_with_root,
        domain_body_upper,
    )?;
    coordinator.preflight_absolute_peak(call_baseline)?;
    let call_budget = ConstrainedZonotopeCallBudget::new(
        budget.deadline(),
        call_baseline,
        budget.max_peak_live_bytes(),
    );
    let outcome = bisect_constrained_zonotope_protected_alpha_unwired_with_budget(
        parent,
        alpha_axis,
        bisection_limits,
        call_budget,
    )
    .map_err(|error| match error {
        ConstrainedZonotopeAlphaBisectionBudgetError::Budget(error) => {
            CganCzProbeDecline::Budget(error)
        }
        ConstrainedZonotopeAlphaBisectionBudgetError::Transform(error) => {
            CganCzProbeDecline::Transform {
                node: "protected-alpha cover",
                operation: "exact-dyadic bisection",
                message: error.to_string(),
            }
        }
    })?;
    let ((children, plan), report) = outcome.into_parts();
    coordinator.absorb(report)?;
    if plan.alpha_axis != alpha_axis
        || plan.value_dim != parent.value_dim()
        || plan.alpha_dim != parent.alpha_dim()
        || plan.constraint_count != parent.constraint_count()
    {
        return Err(CganCzProbeDecline::Transform {
            node: "protected-alpha cover",
            operation: "bisection plan audit",
            message: "the bisection receipt changed the selected axis or domain geometry"
                .to_string(),
        });
    }
    let (negative, positive) = children.into_children();
    next.push(negative);
    next.push(positive);
    Ok(())
}

fn validate_protected_alpha_cover(
    input: &ConstrainedZonotope64,
    split_axes: &[usize],
    limits: CganCzProtectedAlphaCoverLimits,
) -> Result<(), CganCzProbeDecline> {
    if split_axes.is_empty() {
        return Err(CganCzProbeDecline::InvalidLimit {
            message: "protected-alpha split plan must contain at least one level".to_string(),
        });
    }
    if limits.protected_alpha_dim == 0
        || limits.max_split_levels == 0
        || limits.max_tree_nodes == 0
        || limits.max_leaf_domains == 0
        || limits.bisection.max_value_dim == 0
        || limits.bisection.max_alpha_dim == 0
        || limits.bisection.max_generator_nonzeros == 0
    {
        return Err(CganCzProbeDecline::InvalidLimit {
            message: "protected-alpha cover limits must be nonzero".to_string(),
        });
    }
    if limits.protected_alpha_dim > input.alpha_dim() {
        return Err(CganCzProbeDecline::InvalidLimit {
            message: format!(
                "protected alpha dimension {} exceeds input alpha dimension {}",
                limits.protected_alpha_dim,
                input.alpha_dim()
            ),
        });
    }
    if let Some(&axis) = split_axes
        .iter()
        .find(|&&axis| axis >= limits.protected_alpha_dim)
    {
        return Err(CganCzProbeDecline::InvalidLimit {
            message: format!(
                "split axis {axis} is outside protected prefix 0..{}",
                limits.protected_alpha_dim
            ),
        });
    }
    Ok(())
}

fn checked_power_of_two(
    exponent: usize,
    operation: &'static str,
) -> Result<usize, CganCzProbeDecline> {
    let shift =
        u32::try_from(exponent).map_err(|_| CganCzProbeDecline::ResourceOverflow { operation })?;
    1_usize
        .checked_shl(shift)
        .ok_or(CganCzProbeDecline::ResourceOverflow { operation })
}

fn cover_queue_absolute_bytes(
    baseline: usize,
    vector_slots: usize,
    retained_domain_bodies: usize,
    domain_body_upper: usize,
) -> Result<usize, CganCzProbeDecline> {
    let vector_bytes = vector_slots
        .checked_mul(size_of::<ConstrainedZonotope64>())
        .ok_or(CganCzProbeDecline::ResourceOverflow {
            operation: "protected-alpha frontier storage",
        })?;
    let body_bytes = retained_domain_bodies
        .checked_mul(domain_body_upper)
        .ok_or(CganCzProbeDecline::ResourceOverflow {
            operation: "protected-alpha domain storage",
        })?;
    baseline
        .checked_add(vector_bytes)
        .and_then(|bytes| bytes.checked_add(body_bytes))
        .ok_or(CganCzProbeDecline::ResourceOverflow {
            operation: "protected-alpha aggregate live bytes",
        })
}

#[derive(Clone)]
struct AffineParameters {
    weights: Array2<f64>,
    bias: Vec<f64>,
}

#[derive(Clone)]
struct BatchNormParameters {
    gamma: Vec<f64>,
    beta: Vec<f64>,
    mean: Vec<f64>,
    variance: Vec<f64>,
    epsilon: f64,
    normalized_scale: Vec<f64>,
    normalized_bias: Vec<f64>,
}

#[derive(Clone)]
struct Conv2dParameters {
    weights: Array4<f64>,
    bias: Vec<f64>,
    spec: ConstrainedZonotopeConv2dSpec,
}

#[derive(Clone)]
struct ConvTranspose2dParameters {
    weights: Array4<f64>,
    bias: Vec<f64>,
    spec: ConstrainedZonotopeConvTranspose2dSpec,
}

#[derive(Clone)]
enum SealedLayer {
    Affine(AffineParameters),
    Reshape(Vec<usize>),
    BatchNorm(BatchNormParameters),
    ConvTranspose2d(ConvTranspose2dParameters),
    Conv2d(Conv2dParameters),
    Relu,
}

struct SealedCgan {
    layers: Vec<SealedLayer>,
    parameter_elements: usize,
    live_bytes: usize,
}

#[derive(Clone, Copy)]
enum ProbeExtent {
    FirstGeneratorBlock,
    SecondGeneratorBlock,
    ThirdGeneratorBlock,
    GeneratorDiscriminatorHandoff,
    Full,
}

impl ProbeExtent {
    fn prefix_last_index(self) -> Option<usize> {
        match self {
            Self::FirstGeneratorBlock => Some(5),
            Self::SecondGeneratorBlock => Some(8),
            Self::ThirdGeneratorBlock => Some(11),
            Self::GeneratorDiscriminatorHandoff => Some(14),
            Self::Full => None,
        }
    }

    fn prefix_end_exclusive(self) -> usize {
        self.prefix_last_index().map_or(22, |index| index + 1)
    }
}

// Probe-local publication is also allocation-free and transactional.
#[allow(clippy::large_enum_variant)]
enum ProbeCompletion {
    Prefix(CganCzPrefixCompletion),
    Full(CganCzCompletedBounds),
}

#[derive(Debug)]
struct ProtectedAlphaLeafDecline {
    leaf_index: Option<usize>,
    node: &'static str,
    reason: CganCzProbeDecline,
}

fn protected_alpha_leaf_decline(
    leaf_index: Option<usize>,
    node: &'static str,
    reason: CganCzProbeDecline,
) -> ProtectedAlphaLeafDecline {
    ProtectedAlphaLeafDecline {
        leaf_index,
        node,
        reason,
    }
}

/// Consume and propagate a complete cover without exposing partial results.
///
/// The callback receives a baseline that already accounts for every retained
/// input leaf, the fixed leaf vector allocation, the fully reserved private
/// completion vector, and one leaf's stage telemetry. It must add the current
/// transformed domain through `domain_call_budget`, just like the sealed
/// sequential runner does.
fn propagate_cgan_cz_complete_cover_with<N, F>(
    cover: CganCzProtectedAlphaCover,
    expected_split_axes: &[usize],
    moat: CertifiedScalarMoat,
    max_leaf_propagations: usize,
    retained_baseline: usize,
    coordinator: &mut Coordinator<N>,
    mut propagate: F,
) -> Result<CganCzProtectedAlphaAggregateBounds, ProtectedAlphaLeafDecline>
where
    N: FnMut(&'static str) -> Instant,
    F: FnMut(
        usize,
        ConstrainedZonotope64,
        usize,
        &mut Coordinator<N>,
    ) -> Result<(CganCzCompletedBounds, usize), (&'static str, CganCzProbeDecline)>,
{
    let report = cover.report();
    let split_levels = u32::try_from(expected_split_axes.len()).map_err(|_| {
        protected_alpha_leaf_decline(
            None,
            "protected-alpha cover",
            CganCzProbeDecline::ResourceOverflow {
                operation: "protected-alpha expected split levels",
            },
        )
    })?;
    let expected_leaf_domains = 1_usize.checked_shl(split_levels).ok_or_else(|| {
        protected_alpha_leaf_decline(
            None,
            "protected-alpha cover",
            CganCzProbeDecline::ResourceOverflow {
                operation: "protected-alpha expected leaf domains",
            },
        )
    })?;
    if cover.split_axes() != expected_split_axes {
        return Err(protected_alpha_leaf_decline(
            None,
            "protected-alpha cover",
            CganCzProbeDecline::Transform {
                node: "protected-alpha cover",
                operation: "sealed split-plan audit",
                message: format!(
                    "complete cGAN propagation requires split axes {expected_split_axes:?}, got {:?}",
                    cover.split_axes()
                ),
            },
        ));
    }
    if report.split_levels() != expected_split_axes.len()
        || report.leaf_domains() != expected_leaf_domains
        || cover.leaves().len() != expected_leaf_domains
    {
        return Err(protected_alpha_leaf_decline(
            None,
            "protected-alpha cover",
            CganCzProbeDecline::Transform {
                node: "protected-alpha cover",
                operation: "complete leaf-cardinality audit",
                message: format!(
                    "{} exact split levels require {expected_leaf_domains} leaves; report levels={}, leaves={}, storage={}",
                    expected_split_axes.len(),
                    report.split_levels(),
                    report.leaf_domains(),
                    cover.leaves().len()
                ),
            },
        ));
    }
    check_resource(
        "protected-alpha complete leaf propagations",
        expected_leaf_domains,
        max_leaf_propagations,
    )
    .map_err(|reason| protected_alpha_leaf_decline(None, "protected-alpha cover", reason))?;
    coordinator
        .checkpoint("protected-alpha leaf propagation admission")
        .map_err(|reason| protected_alpha_leaf_decline(None, "protected-alpha cover", reason))?;

    let leaves = cover.into_leaves();
    let leaf_vector_bytes = leaves
        .capacity()
        .checked_mul(size_of::<ConstrainedZonotope64>())
        .ok_or_else(|| {
            protected_alpha_leaf_decline(
                None,
                "protected-alpha cover",
                CganCzProbeDecline::ResourceOverflow {
                    operation: "protected-alpha retained leaf vector",
                },
            )
        })?;
    let mut retained_leaf_body_bytes = 0_usize;
    let mut maximum_leaf_body_bytes = 0_usize;
    for leaf in &leaves {
        let body = domain_live_bytes(leaf).map_err(|reason| {
            protected_alpha_leaf_decline(None, "protected-alpha cover", reason)
        })?;
        retained_leaf_body_bytes = retained_leaf_body_bytes.checked_add(body).ok_or_else(|| {
            protected_alpha_leaf_decline(
                None,
                "protected-alpha cover",
                CganCzProbeDecline::ResourceOverflow {
                    operation: "protected-alpha retained leaf bodies",
                },
            )
        })?;
        maximum_leaf_body_bytes = maximum_leaf_body_bytes.max(body);
        coordinator
            .charge(1, "protected-alpha retained leaf accounting")
            .map_err(|reason| {
                protected_alpha_leaf_decline(None, "protected-alpha cover", reason)
            })?;
    }
    let nominal_completion_bytes = expected_leaf_domains
        .checked_mul(size_of::<CganCzProtectedAlphaLeafCompletion>())
        .ok_or_else(|| {
            protected_alpha_leaf_decline(
                None,
                "protected-alpha cover",
                CganCzProbeDecline::ResourceOverflow {
                    operation: "protected-alpha private completion storage",
                },
            )
        })?;
    let telemetry_bytes = MAX_RUNNER_STAGES
        .checked_mul(RUNNER_STAGE_RESERVED_BYTES)
        .ok_or_else(|| {
            protected_alpha_leaf_decline(
                None,
                "protected-alpha cover",
                CganCzProbeDecline::ResourceOverflow {
                    operation: "protected-alpha per-leaf telemetry storage",
                },
            )
        })?;
    let leaf_retained_baseline = retained_baseline
        .checked_add(leaf_vector_bytes)
        .and_then(|bytes| bytes.checked_add(retained_leaf_body_bytes))
        .and_then(|bytes| bytes.checked_add(nominal_completion_bytes))
        .and_then(|bytes| bytes.checked_add(telemetry_bytes))
        .ok_or_else(|| {
            protected_alpha_leaf_decline(
                None,
                "protected-alpha cover",
                CganCzProbeDecline::ResourceOverflow {
                    operation: "protected-alpha propagation retained baseline",
                },
            )
        })?;
    coordinator
        .preflight_absolute_peak(
            leaf_retained_baseline
                .checked_add(maximum_leaf_body_bytes)
                .ok_or_else(|| {
                    protected_alpha_leaf_decline(
                        None,
                        "protected-alpha cover",
                        CganCzProbeDecline::ResourceOverflow {
                            operation: "protected-alpha propagation admission peak",
                        },
                    )
                })?,
        )
        .map_err(|reason| protected_alpha_leaf_decline(None, "protected-alpha cover", reason))?;

    let mut completions = Vec::new();
    completions
        .try_reserve_exact(expected_leaf_domains)
        .map_err(|_| {
            protected_alpha_leaf_decline(
                None,
                "protected-alpha cover",
                CganCzProbeDecline::ResourceLimit {
                    resource: "protected-alpha private completion allocation",
                    required: expected_leaf_domains,
                    limit: expected_leaf_domains - 1,
                },
            )
        })?;
    // `try_reserve_exact` may legally return more capacity than requested.
    // Recompute the retained baseline from the actual allocation before any
    // leaf work so allocator growth cannot escape the logical peak receipt.
    let actual_completion_bytes = completions
        .capacity()
        .checked_mul(size_of::<CganCzProtectedAlphaLeafCompletion>())
        .ok_or_else(|| {
            protected_alpha_leaf_decline(
                None,
                "protected-alpha cover",
                CganCzProbeDecline::ResourceOverflow {
                    operation: "protected-alpha actual completion storage",
                },
            )
        })?;
    let leaf_retained_baseline = retained_baseline
        .checked_add(leaf_vector_bytes)
        .and_then(|bytes| bytes.checked_add(retained_leaf_body_bytes))
        .and_then(|bytes| bytes.checked_add(actual_completion_bytes))
        .and_then(|bytes| bytes.checked_add(telemetry_bytes))
        .ok_or_else(|| {
            protected_alpha_leaf_decline(
                None,
                "protected-alpha cover",
                CganCzProbeDecline::ResourceOverflow {
                    operation: "protected-alpha actual propagation retained baseline",
                },
            )
        })?;
    coordinator
        .preflight_absolute_peak(
            leaf_retained_baseline
                .checked_add(maximum_leaf_body_bytes)
                .ok_or_else(|| {
                    protected_alpha_leaf_decline(
                        None,
                        "protected-alpha cover",
                        CganCzProbeDecline::ResourceOverflow {
                            operation: "protected-alpha actual propagation admission peak",
                        },
                    )
                })?,
        )
        .map_err(|reason| protected_alpha_leaf_decline(None, "protected-alpha cover", reason))?;

    for (leaf_index, leaf) in leaves.into_iter().enumerate() {
        coordinator
            .checkpoint("protected-alpha leaf propagation dispatch")
            .map_err(|reason| {
                protected_alpha_leaf_decline(Some(leaf_index), "protected-alpha cover", reason)
            })?;
        let charged_start = coordinator.charged_items;
        let polls_start = coordinator.deadline_polls;
        let (bounds, completed_stages) =
            propagate(leaf_index, leaf, leaf_retained_baseline, coordinator).map_err(
                |(node, reason)| protected_alpha_leaf_decline(Some(leaf_index), node, reason),
            )?;
        if completed_stages != FULL_RUNNER_COMPLETED_STAGES {
            return Err(protected_alpha_leaf_decline(
                Some(leaf_index),
                "protected-alpha cover",
                CganCzProbeDecline::Transform {
                    node: "protected-alpha cover",
                    operation: "complete sequential-stage audit",
                    message: format!(
                        "leaf {leaf_index} completed {completed_stages} stages, expected {FULL_RUNNER_COMPLETED_STAGES}"
                    ),
                },
            ));
        }
        validate_protected_alpha_leaf_bounds(leaf_index, &bounds, moat).map_err(|reason| {
            protected_alpha_leaf_decline(Some(leaf_index), "protected-alpha cover", reason)
        })?;
        coordinator
            .charge(1, "protected-alpha leaf completion audit")
            .map_err(|reason| {
                protected_alpha_leaf_decline(Some(leaf_index), "protected-alpha cover", reason)
            })?;
        coordinator
            .checkpoint("protected-alpha private leaf completion")
            .map_err(|reason| {
                protected_alpha_leaf_decline(Some(leaf_index), "protected-alpha cover", reason)
            })?;
        completions.push(CganCzProtectedAlphaLeafCompletion {
            leaf_index,
            bounds,
            completed_stages,
            peak_live_bytes: coordinator.peak_live_bytes,
            charged_items: coordinator
                .charged_items
                .checked_sub(charged_start)
                .ok_or_else(|| {
                    protected_alpha_leaf_decline(
                        Some(leaf_index),
                        "protected-alpha cover",
                        CganCzProbeDecline::ResourceOverflow {
                            operation: "protected-alpha leaf charged-item delta",
                        },
                    )
                })?,
            deadline_polls: coordinator
                .deadline_polls
                .checked_sub(polls_start)
                .ok_or_else(|| {
                    protected_alpha_leaf_decline(
                        Some(leaf_index),
                        "protected-alpha cover",
                        CganCzProbeDecline::ResourceOverflow {
                            operation: "protected-alpha leaf deadline-poll delta",
                        },
                    )
                })?,
        });
    }

    if completions.len() != expected_leaf_domains
        || completions
            .iter()
            .enumerate()
            .any(|(index, completion)| completion.leaf_index != index)
    {
        return Err(protected_alpha_leaf_decline(
            None,
            "protected-alpha cover",
            CganCzProbeDecline::Transform {
                node: "protected-alpha cover",
                operation: "all-leaf completion audit",
                message: "the private completion buffer is incomplete or out of order".to_string(),
            },
        ));
    }
    // Deliberately precedes the first aggregate operation. Expiry here drops
    // every private leaf result without constructing or publishing a hull.
    coordinator
        .checkpoint("protected-alpha all-leaf completion audit")
        .map_err(|reason| protected_alpha_leaf_decline(None, "protected-alpha cover", reason))?;

    let mut lower_bound = f64::INFINITY;
    let mut upper_bound = f64::NEG_INFINITY;
    for completion in &completions {
        coordinator
            .charge(1, "protected-alpha aggregate hull")
            .map_err(|reason| {
                protected_alpha_leaf_decline(None, "protected-alpha cover", reason)
            })?;
        lower_bound = lower_bound.min(completion.bounds.lower_bound);
        upper_bound = upper_bound.max(completion.bounds.upper_bound);
    }
    let low_unsafe_threshold = moat.low_upper();
    let high_unsafe_threshold = moat.high_lower();
    let separates_unsafe_moat = tail_bounds_separate_unsafe_moat(
        lower_bound,
        upper_bound,
        low_unsafe_threshold,
        high_unsafe_threshold,
    );
    coordinator
        .checkpoint("protected-alpha aggregate publication")
        .map_err(|reason| protected_alpha_leaf_decline(None, "protected-alpha cover", reason))?;

    Ok(CganCzProtectedAlphaAggregateBounds {
        lower_bound,
        upper_bound,
        low_unsafe_threshold,
        high_unsafe_threshold,
        separates_unsafe_moat,
        cover: report,
        leaf_completions: completions,
    })
}

fn validate_protected_alpha_leaf_bounds(
    leaf_index: usize,
    bounds: &CganCzCompletedBounds,
    moat: CertifiedScalarMoat,
) -> Result<(), CganCzProbeDecline> {
    let selected_lower = select_m17_m20_lower_bound(
        bounds.lower_m17_candidates.selected_lower_bound,
        bounds.lower_m20_lower_bound,
        bounds.lower_m20_status,
    );
    let selected_negated_upper = select_m17_m20_lower_bound(
        bounds.negated_upper_m17_candidates.selected_lower_bound,
        bounds.negated_upper_m20_lower_bound,
        bounds.negated_upper_m20_status,
    );
    if !bounds.lower_bound.is_finite()
        || !bounds.upper_bound.is_finite()
        || bounds.lower_bound > bounds.upper_bound
        || !bounds.bn_tail_correction_upper.is_finite()
        || bounds.bn_tail_correction_upper < 0.0
        || bounds.low_unsafe_threshold != moat.low_upper()
        || bounds.high_unsafe_threshold != moat.high_lower()
        || selected_lower != Some(bounds.lower_bound)
        || selected_negated_upper != Some(-bounds.upper_bound)
        || bounds.lower_m17_status != bounds.lower_m17_candidates.status
        || bounds.upper_m17_status != bounds.negated_upper_m17_candidates.status
        || bounds.separates_unsafe_moat
            != tail_bounds_separate_unsafe_moat(
                bounds.lower_bound,
                bounds.upper_bound,
                moat.low_upper(),
                moat.high_lower(),
            )
    {
        return Err(CganCzProbeDecline::Transform {
            node: "protected-alpha cover",
            operation: "leaf-bound audit",
            message: format!(
                "leaf {leaf_index} returned malformed or mismatched diagnostic bounds"
            ),
        });
    }
    Ok(())
}

/// Propagate an independent remainder-only interval-CZ through the sealed
/// imgSz32 cGAN prefix and retain certified preactivation bounds at every ReLU.
///
/// This lane is diagnostic-only and deliberately independent of the
/// correlated CZ path. Linear operators reuse the budgeted CZ primitives, but
/// every authored ReLU first bridges the zero-symbol/no-predicate domain to a
/// certified Box, copies that enclosure into the auxiliary-bound proof
/// boundary, applies interval ReLU, and recenters back to a remainder-only CZ.
/// No partial bounds are published when any topology, provenance, allocation,
/// deadline, or aggregate-memory check declines.
pub fn probe_cgan_nch1_independent_interval_unwired(
    model: &OnnxModel,
    graph: &GraphNetwork,
    input: &CertifiedInputBox,
    limits: CganCzSequentialLimits,
    budget: ConstrainedZonotopeCallBudget,
) -> CganCzIndependentIntervalReport {
    probe_cgan_nch1_independent_interval_with_clock(model, graph, input, limits, budget, |_| {
        Instant::now()
    })
}

/// Propagate the independent remainder-only interval-CZ through the exact
/// imgSz32 nCh3 architecture.
///
/// This shares the diagnostic-only implementation with nCh1 but selects a
/// separate typed topology profile before inspecting the model. It does not
/// admit imgSz64 or any arbitrary channel count and cannot authorize a
/// verifier verdict.
pub fn probe_cgan_nch3_independent_interval_unwired(
    model: &OnnxModel,
    graph: &GraphNetwork,
    input: &CertifiedInputBox,
    limits: CganCzSequentialLimits,
    budget: ConstrainedZonotopeCallBudget,
) -> CganCzIndependentIntervalReport {
    probe_cgan_imgsz32_independent_interval_with_clock(
        CganCzImgSz32Profile::Nch3,
        model,
        graph,
        input,
        limits,
        budget,
        |_| Instant::now(),
    )
}

fn probe_cgan_nch1_independent_interval_with_clock<N>(
    model: &OnnxModel,
    graph: &GraphNetwork,
    input: &CertifiedInputBox,
    limits: CganCzSequentialLimits,
    budget: ConstrainedZonotopeCallBudget,
    now: N,
) -> CganCzIndependentIntervalReport
where
    N: FnMut(&'static str) -> Instant,
{
    probe_cgan_imgsz32_independent_interval_with_clock(
        CganCzImgSz32Profile::Nch1,
        model,
        graph,
        input,
        limits,
        budget,
        now,
    )
}

fn probe_cgan_imgsz32_independent_interval_with_clock<N>(
    profile: CganCzImgSz32Profile,
    model: &OnnxModel,
    graph: &GraphNetwork,
    input: &CertifiedInputBox,
    limits: CganCzSequentialLimits,
    budget: ConstrainedZonotopeCallBudget,
    now: N,
) -> CganCzIndependentIntervalReport
where
    N: FnMut(&'static str) -> Instant,
{
    let mut parameter_elements = 0_usize;
    let mut at_node = "admission";
    let mut coordinator = Coordinator::new(budget, now);
    let result = run_cgan_imgsz32_independent_prefix(
        profile,
        model,
        graph,
        input.lower(),
        input.upper(),
        limits,
        budget,
        0,
        &mut parameter_elements,
        &mut at_node,
        &mut coordinator,
    )
    .map(|private| {
        let CganCzIndependentPrefixPrivate {
            relu_bounds,
            post_relu_23,
            ..
        } = private;
        CganCzIndependentIntervalCompletion {
            relu_bounds,
            post_relu_23,
        }
    });

    CganCzIndependentIntervalReport {
        authority: CGAN_CZ_VERDICT_AUTHORITY,
        profile,
        status: match result {
            Ok(completed) => CganCzIndependentIntervalStatus::Completed(completed),
            Err(reason) => CganCzIndependentIntervalStatus::Declined {
                node: at_node,
                reason,
            },
        },
        topology_work_items: coordinator.topology_work_items,
        parameter_elements,
        peak_live_bytes: coordinator.peak_live_bytes,
        charged_items: coordinator.charged_items,
        deadline_polls: coordinator.deadline_polls,
    }
}

#[allow(clippy::too_many_arguments)]
fn run_cgan_imgsz32_independent_prefix<N>(
    profile: CganCzImgSz32Profile,
    model: &OnnxModel,
    graph: &GraphNetwork,
    input_lower: &[f64],
    input_upper: &[f64],
    limits: CganCzSequentialLimits,
    budget: ConstrainedZonotopeCallBudget,
    caller_retained_bytes: usize,
    parameter_elements: &mut usize,
    at_node: &mut &'static str,
    coordinator: &mut Coordinator<N>,
) -> Result<CganCzIndependentPrefixPrivate, CganCzProbeDecline>
where
    N: FnMut(&'static str) -> Instant,
{
    validate_limits(limits)?;
    check_resource(
        "independent interval sealed value dimension",
        CGAN_SEALED_MAX_VALUE_DIM,
        limits.max_value_dim,
    )?;
    coordinator.checkpoint("cGAN independent interval admission")?;
    let caller_baseline = budget
        .baseline_live_bytes()
        .checked_add(caller_retained_bytes)
        .ok_or(CganCzProbeDecline::ResourceOverflow {
            operation: "cGAN independent caller-retained baseline",
        })?;
    coordinator.preflight_absolute_peak(caller_baseline)?;

    let planned_relu_record_bytes = CGAN_RELU_COUNT
        .checked_mul(size_of::<CganCzIndependentIntervalReluBounds>())
        .ok_or(CganCzProbeDecline::ResourceOverflow {
            operation: "cGAN independent planned ReLU-bound record bytes",
        })?;
    let planned_runner_storage_bytes = RUNNER_TELEMETRY_RESERVED_BYTES
        .checked_add(planned_relu_record_bytes)
        .and_then(|bytes| bytes.checked_add(2 * CGAN_AUTHORED_TENSOR_RANK * size_of::<usize>()))
        .ok_or(CganCzProbeDecline::ResourceOverflow {
            operation: "cGAN independent planned runner storage bytes",
        })?;
    coordinator.preflight_absolute_peak(
        caller_baseline
            .checked_add(planned_runner_storage_bytes)
            .ok_or(CganCzProbeDecline::ResourceOverflow {
                operation: "cGAN independent planned runner allocation peak",
            })?,
    )?;

    let mut stages = Vec::new();
    let mut relu_bounds = Vec::new();
    stages
        .try_reserve_exact(MAX_RUNNER_STAGES)
        .map_err(|_| CganCzProbeDecline::ResourceLimit {
            resource: "cGAN independent stage telemetry allocation",
            required: MAX_RUNNER_STAGES,
            limit: MAX_RUNNER_STAGES - 1,
        })?;
    relu_bounds
        .try_reserve_exact(CGAN_RELU_COUNT)
        .map_err(|_| CganCzProbeDecline::ResourceLimit {
            resource: "cGAN independent ReLU-bound record allocation",
            required: CGAN_RELU_COUNT,
            limit: CGAN_RELU_COUNT - 1,
        })?;
    let telemetry_bytes = stages
        .capacity()
        .checked_mul(size_of::<CganCzStageTelemetry>())
        .ok_or(CganCzProbeDecline::ResourceOverflow {
            operation: "cGAN independent stage telemetry bytes",
        })?;
    check_resource(
        "cGAN independent stage telemetry bytes",
        telemetry_bytes,
        RUNNER_TELEMETRY_RESERVED_BYTES,
    )?;
    let relu_record_bytes = relu_bounds
        .capacity()
        .checked_mul(size_of::<CganCzIndependentIntervalReluBounds>())
        .ok_or(CganCzProbeDecline::ResourceOverflow {
            operation: "cGAN independent ReLU-bound record bytes",
        })?;
    // Reserve the established per-stage envelope, not merely the Vec's inline
    // records: each completed stage also owns a small shape Vec.
    let runner_storage_bytes = RUNNER_TELEMETRY_RESERVED_BYTES
        .checked_add(relu_record_bytes)
        .and_then(|bytes| bytes.checked_add(2 * CGAN_AUTHORED_TENSOR_RANK * size_of::<usize>()))
        .ok_or(CganCzProbeDecline::ResourceOverflow {
            operation: "cGAN independent runner storage bytes",
        })?;
    let retained_support_bytes = runner_storage_bytes
        .checked_add(caller_retained_bytes)
        .ok_or(CganCzProbeDecline::ResourceOverflow {
            operation: "cGAN independent retained support bytes",
        })?;
    coordinator.preflight_absolute_peak(
        budget
            .baseline_live_bytes()
            .checked_add(retained_support_bytes)
            .ok_or(CganCzProbeDecline::ResourceOverflow {
                operation: "cGAN independent baseline plus runner storage",
            })?,
    )?;

    *at_node = "topology";
    seal_topology_for_profile(profile, model, graph, limits, coordinator)?;
    *at_node = "parameter provenance";
    let sealed = seal_parameters_for_profile(
        profile,
        model,
        graph,
        limits,
        retained_support_bytes,
        coordinator,
    )?;
    *parameter_elements = sealed.parameter_elements;

    *at_node = "input";
    let mut domain = build_independent_input_domain_from_bounds(
        input_lower,
        input_upper,
        limits,
        &sealed,
        retained_support_bytes,
        coordinator,
    )?;
    let mut shape = Vec::new();
    shape
        .try_reserve_exact(CGAN_AUTHORED_TENSOR_RANK)
        .map_err(|_| CganCzProbeDecline::ResourceLimit {
            resource: "cGAN independent shape storage allocation",
            required: CGAN_AUTHORED_TENSOR_RANK,
            limit: CGAN_AUTHORED_TENSOR_RANK - 1,
        })?;
    shape.push(PROTECTED_LATENT_SYMBOLS);
    let retained_baseline = budget
        .baseline_live_bytes()
        .checked_add(retained_support_bytes)
        .and_then(|bytes| bytes.checked_add(sealed.live_bytes))
        .ok_or(CganCzProbeDecline::ResourceOverflow {
            operation: "cGAN independent retained baseline",
        })?;
    coordinator.preflight_absolute_peak(
        retained_baseline
            .checked_add(domain_live_bytes(&domain)?)
            .ok_or(CganCzProbeDecline::ResourceOverflow {
                operation: "cGAN independent retained input-domain peak",
            })?,
    )?;

    for index in 0..=CGAN_RELU_INDICES[CGAN_RELU_COUNT - 1] {
        *at_node = EXPECTED_NODES[index].0;
        let retained_auxiliary_bytes = independent_auxiliary_live_bytes(&relu_bounds)?;
        let layer_retained_baseline = retained_baseline
            .checked_add(retained_auxiliary_bytes)
            .ok_or(CganCzProbeDecline::ResourceOverflow {
                operation: "cGAN independent retained auxiliary baseline",
            })?;
        if matches!(&sealed.layers[index], SealedLayer::Relu) {
            apply_independent_interval_relu(
                profile,
                index,
                &mut domain,
                &mut shape,
                limits,
                layer_retained_baseline,
                budget,
                coordinator,
                &mut relu_bounds,
            )?;
        } else {
            apply_prefix_layer_for_profile(
                profile,
                index,
                &sealed.layers[index],
                &mut domain,
                &mut shape,
                None,
                None,
                limits,
                layer_retained_baseline,
                budget,
                coordinator,
                &mut stages,
            )?;
            if domain.alpha_dim() != 0 || domain.constraint_count() != 0 {
                return Err(CganCzProbeDecline::Transform {
                    node: EXPECTED_NODES[index].0,
                    operation: "independent interval structural audit",
                    message: format!(
                        "linear stage produced alpha_dim={} and constraint_count={}",
                        domain.alpha_dim(),
                        domain.constraint_count()
                    ),
                });
            }
        }
    }

    if relu_bounds.len() != CGAN_RELU_COUNT
        || relu_bounds
            .iter()
            .zip(CGAN_RELU_INDICES)
            .any(|(record, index)| {
                record.node != EXPECTED_NODES[index].0
                    || record.output_shape != expected_output_shape_for_profile(profile, index)
            })
        || shape.as_slice() != expected_output_shape_for_profile(profile, 22)
        || domain.value_dim() != TAIL_VALUE_DIM
        || domain.alpha_dim() != 0
        || domain.constraint_count() != 0
    {
        return Err(CganCzProbeDecline::Topology {
            message: "independent interval lane did not complete the exact seven-ReLU prefix"
                .to_string(),
        });
    }
    coordinator.checkpoint("cGAN independent interval publication")?;
    Ok(CganCzIndependentPrefixPrivate {
        sealed,
        stages,
        relu_bounds,
        post_relu_23: domain,
        retained_baseline,
    })
}

fn promote_cgan_leaf_input_f32<N>(
    lower: &[f32],
    upper: &[f32],
    coordinator: &mut Coordinator<N>,
) -> Result<CganCzLeafInput64, CganCzProbeDecline>
where
    N: FnMut(&'static str) -> Instant,
{
    coordinator.checkpoint("cGAN leaf-row input admission")?;
    if lower.len() != PROTECTED_LATENT_SYMBOLS || upper.len() != PROTECTED_LATENT_SYMBOLS {
        return Err(CganCzProbeDecline::Topology {
            message: format!(
                "cGAN leaf-row input requires five lower and upper endpoints, got {} and {}",
                lower.len(),
                upper.len()
            ),
        });
    }
    let required = coordinator
        .budget
        .baseline_live_bytes()
        .checked_add(size_of::<CganCzLeafInput64>())
        .ok_or(CganCzProbeDecline::ResourceOverflow {
            operation: "cGAN leaf-row promoted input peak",
        })?;
    coordinator.preflight_absolute_peak(required)?;

    let mut promoted_lower = [0.0; PROTECTED_LATENT_SYMBOLS];
    let mut promoted_upper = [0.0; PROTECTED_LATENT_SYMBOLS];
    for index in 0..PROTECTED_LATENT_SYMBOLS {
        coordinator.charge(1, "cGAN leaf-row input validation")?;
        let lo = lower[index];
        let hi = upper[index];
        if !lo.is_finite() || !hi.is_finite() {
            return Err(CganCzProbeDecline::Topology {
                message: format!("cGAN leaf-row input endpoint {index} must be finite"),
            });
        }
        if lo > hi {
            return Err(CganCzProbeDecline::Topology {
                message: format!("cGAN leaf-row input endpoint {index} is reversed: {lo} > {hi}"),
            });
        }
        // Every finite binary32 value is exactly representable in binary64.
        // This promotion therefore neither narrows nor widens the BaB leaf.
        promoted_lower[index] = f64::from(lo);
        promoted_upper[index] = f64::from(hi);
    }
    coordinator.checkpoint("cGAN leaf-row input publication")?;
    Ok(CganCzLeafInput64 {
        lower: promoted_lower,
        upper: promoted_upper,
    })
}

fn publish_cgan_leaf_rows<N>(
    output: CganCzOutputTailResult,
    coordinator: &mut Coordinator<N>,
) -> Result<CganCzLeafRowBounds, CganCzProbeDecline>
where
    N: FnMut(&'static str) -> Instant,
{
    let CganCzOutputTailResult {
        completed,
        lower_depth_two_measurement,
        negated_upper_depth_two_measurement,
    } = output;
    let Some(lower_y) = select_m17_m20_lower_bound(
        completed.lower_m17_candidates.selected_lower_bound,
        completed.lower_m20_lower_bound,
        completed.lower_m20_status,
    ) else {
        return Err(CganCzProbeDecline::OutputTail {
            message: "leaf-row lower-tail portfolio attribution is malformed".to_string(),
        });
    };
    let Some(lower_neg_y) = select_m17_m20_lower_bound(
        completed.negated_upper_m17_candidates.selected_lower_bound,
        completed.negated_upper_m20_lower_bound,
        completed.negated_upper_m20_status,
    ) else {
        return Err(CganCzProbeDecline::OutputTail {
            message: "leaf-row upper-tail portfolio attribution is malformed".to_string(),
        });
    };
    if completed.lower_bound != lower_y
        || completed.upper_bound != -lower_neg_y
        || !lower_neg_y.is_finite()
    {
        return Err(CganCzProbeDecline::OutputTail {
            message: "leaf-row tail attribution did not match its published scalar bounds"
                .to_string(),
        });
    }
    let bounds = CganCzLeafRowBounds {
        lower_y: completed.lower_bound,
        lower_neg_y,
        bn_tail_correction_upper: completed.bn_tail_correction_upper,
        lower_m17_status: completed.lower_m17_status,
        negated_upper_m17_status: completed.upper_m17_status,
        lower_m17_candidates: completed.lower_m17_candidates,
        negated_upper_m17_candidates: completed.negated_upper_m17_candidates,
        lower_m20_lower_bound: completed.lower_m20_lower_bound,
        negated_upper_m20_lower_bound: completed.negated_upper_m20_lower_bound,
        lower_m20_status: completed.lower_m20_status,
        negated_upper_m20_status: completed.negated_upper_m20_status,
        lower_m24_measurement: completed.lower_m24_measurement,
        negated_upper_m24_measurement: completed.negated_upper_m24_measurement,
        lower_depth_two_measurement,
        negated_upper_depth_two_measurement,
    };
    coordinator.checkpoint("cGAN leaf-row publication")?;
    Ok(bounds)
}

pub(crate) fn select_m17_m20_lower_bound(
    m17_lower_bound: f64,
    m20_lower_bound: Option<f64>,
    m20_status: CganCzM20Status,
) -> Option<f64> {
    if !m17_lower_bound.is_finite() {
        return None;
    }
    match (m20_status, m20_lower_bound) {
        (CganCzM20Status::Completed, Some(m20)) if m20.is_finite() => {
            Some(if m20 > m17_lower_bound {
                m20
            } else {
                m17_lower_bound
            })
        }
        (CganCzM20Status::NotRequested | CganCzM20Status::Fallback, None) => Some(m17_lower_bound),
        _ => None,
    }
}

/// Bound both scalar objective rows over one exact binary32 imgSz32 leaf.
///
/// The selected nCh1/nCh3 profile, raw ONNX parameters, and normalized graph
/// are sealed before propagation. An independent remainder-only lane certifies
/// a preactivation Box at every authored ReLU. A correlated lane then reuses
/// the same authenticated parameters and exact leaf, consuming the first six
/// Boxes at their corresponding ReLUs through `Relu_20`. The final `Relu_23`
/// Box is retained for the M17/M20 tail portfolio. No untyped auxiliary bounds
/// or tail parameters cross this boundary.
///
/// The optional depth-two replay is deliberately not requested here. A sound
/// re-enable must reconstruct its `Relu_20` state after the historical tail
/// completes; retaining that state across the mandatory tail can otherwise
/// turn a finite-cap historical completion into an outer refusal.
///
/// A completed report remains verdict-neutral and carries the permanently
/// disabled cGAN authority marker.  Every nested transform shares `budget`'s
/// one immutable absolute deadline and peak-live ceiling.
#[allow(clippy::too_many_arguments)]
pub(crate) fn bound_cgan_imgsz32_leaf_rows_unwired(
    profile: CganCzImgSz32Profile,
    model: &OnnxModel,
    graph: &GraphNetwork,
    leaf_lower: &[f32],
    leaf_upper: &[f32],
    moat: CertifiedScalarMoat,
    limits: CganCzSequentialLimits,
    budget: ConstrainedZonotopeCallBudget,
) -> CganCzLeafRowReport {
    bound_cgan_imgsz32_leaf_rows_with_clock(
        profile,
        model,
        graph,
        leaf_lower,
        leaf_upper,
        moat,
        limits,
        budget,
        |_| Instant::now(),
    )
}

#[allow(clippy::too_many_arguments)]
fn bound_cgan_imgsz32_leaf_rows_with_clock<N>(
    profile: CganCzImgSz32Profile,
    model: &OnnxModel,
    graph: &GraphNetwork,
    leaf_lower: &[f32],
    leaf_upper: &[f32],
    moat: CertifiedScalarMoat,
    limits: CganCzSequentialLimits,
    budget: ConstrainedZonotopeCallBudget,
    now: N,
) -> CganCzLeafRowReport
where
    N: FnMut(&'static str) -> Instant,
{
    let mut parameter_elements = 0_usize;
    let mut at_node = "leaf input";
    let mut coordinator = Coordinator::new(budget, now);
    let result = (|| {
        let input = promote_cgan_leaf_input_f32(leaf_lower, leaf_upper, &mut coordinator)?;
        let private = run_cgan_imgsz32_independent_prefix(
            profile,
            model,
            graph,
            &input.lower,
            &input.upper,
            limits,
            budget,
            size_of::<CganCzLeafInput64>(),
            &mut parameter_elements,
            &mut at_node,
            &mut coordinator,
        )?;
        let CganCzIndependentPrefixPrivate {
            sealed,
            mut stages,
            mut relu_bounds,
            post_relu_23,
            retained_baseline,
        } = private;
        authenticate_independent_relu_bound_sequence(profile, &relu_bounds)?;
        let all_auxiliary_bytes = independent_auxiliary_live_bytes(&relu_bounds)?;
        let final_relu_23 = relu_bounds
            .pop()
            .ok_or_else(|| CganCzProbeDecline::Topology {
                message: "independent leaf prefix published no Relu_23 auxiliary bounds"
                    .to_string(),
            })?;
        authenticate_independent_relu_bound_record(profile, 22, &final_relu_23)?;
        if relu_bounds.len() != CGAN_CORRELATED_AUXILIARY_RELU_COUNT {
            return Err(CganCzProbeDecline::Topology {
                message: format!(
                    "independent leaf prefix retained {} correlated auxiliary records; expected {CGAN_CORRELATED_AUXILIARY_RELU_COUNT}",
                    relu_bounds.len()
                ),
            });
        }
        // The independent post-ReLU carrier is not used by the correlated
        // replay. All seven preactivation endpoint payloads remain live and
        // charged until the first six have been streamed through their exact
        // correlated ReLUs.
        drop(post_relu_23);
        stages.clear();

        let correlated_retained_baseline = retained_baseline
            .checked_add(all_auxiliary_bytes)
            .ok_or(CganCzProbeDecline::ResourceOverflow {
                operation: "cGAN correlated leaf retained auxiliary baseline",
            })?;
        coordinator.preflight_absolute_peak(correlated_retained_baseline)?;

        at_node = "correlated leaf input";
        let mut correlated = build_correlated_leaf_input_domain_from_bounds(
            &input.lower,
            &input.upper,
            limits,
            correlated_retained_baseline,
            &mut coordinator,
        )?;
        let mut shape = Vec::new();
        shape
            .try_reserve_exact(CGAN_AUTHORED_TENSOR_RANK)
            .map_err(|_| CganCzProbeDecline::ResourceLimit {
                resource: "cGAN correlated leaf shape storage allocation",
                required: CGAN_AUTHORED_TENSOR_RANK,
                limit: CGAN_AUTHORED_TENSOR_RANK - 1,
            })?;
        shape.push(PROTECTED_LATENT_SYMBOLS);
        {
            let mut auxiliary_records = relu_bounds.iter();
            let mut auxiliary_records_consumed = 0_usize;
            for index in 0..ProbeExtent::Full.prefix_end_exclusive() {
                at_node = EXPECTED_NODES[index].0;
                let auxiliary = next_correlated_auxiliary_bounds(
                    profile,
                    index,
                    &mut auxiliary_records,
                    &mut auxiliary_records_consumed,
                )?;
                apply_prefix_layer_for_profile(
                    profile,
                    index,
                    &sealed.layers[index],
                    &mut correlated,
                    &mut shape,
                    relu_reduction_target(ProbeExtent::Full, index, limits),
                    auxiliary,
                    limits,
                    correlated_retained_baseline,
                    budget,
                    &mut coordinator,
                    &mut stages,
                )?;
            }
            if auxiliary_records_consumed != CGAN_CORRELATED_AUXILIARY_RELU_COUNT
                || auxiliary_records.next().is_some()
            {
                return Err(CganCzProbeDecline::Topology {
                    message: format!(
                        "correlated prefix consumed {auxiliary_records_consumed} auxiliary records; expected exactly {CGAN_CORRELATED_AUXILIARY_RELU_COUNT} with no extras"
                    ),
                });
            }
        }
        if shape.as_slice() != expected_output_shape_for_profile(profile, 21)
            || correlated.value_dim() != TAIL_VALUE_DIM
        {
            return Err(CganCzProbeDecline::Topology {
                message: format!(
                    "correlated leaf tail has shape {shape:?} and value_dim={}, expected {:?} and {TAIL_VALUE_DIM}",
                    correlated.value_dim(),
                    expected_output_shape_for_profile(profile, 21)
                ),
            });
        }
        // The first six endpoint payloads no longer share the tail ceiling.
        // `bound_output_tail` independently charges the retained Relu_23
        // endpoints alongside its rational and M17/M20/M24 scratch.
        drop(relu_bounds);

        at_node = "Relu_23 exact M17/M20 plus M24 measurement tail";
        let tail_bn = match &sealed.layers[23] {
            SealedLayer::BatchNorm(parameters) => parameters,
            _ => unreachable!("sealed topology fixes BatchNormalization_24"),
        };
        let tail_affine = match &sealed.layers[25] {
            SealedLayer::Affine(parameters) => parameters,
            _ => unreachable!("sealed topology fixes Gemm_27"),
        };
        let completed = bound_output_tail(
            &correlated,
            Some(&final_relu_23.bounds),
            None,
            tail_bn,
            tail_affine,
            moat,
            limits,
            retained_baseline,
            budget,
            &mut coordinator,
            &mut stages,
        )?;
        publish_cgan_leaf_rows(completed, &mut coordinator)
    })();

    CganCzLeafRowReport {
        authority: CGAN_CZ_VERDICT_AUTHORITY,
        profile,
        deadline: budget.deadline(),
        baseline_live_bytes: budget.baseline_live_bytes(),
        max_peak_live_bytes: budget.max_peak_live_bytes(),
        status: match result {
            Ok(bounds) => CganCzLeafRowStatus::Completed(bounds),
            Err(reason) => CganCzLeafRowStatus::Declined {
                node: at_node,
                reason,
            },
        },
        topology_work_items: coordinator.topology_work_items,
        parameter_elements,
        peak_live_bytes: coordinator.peak_live_bytes,
        charged_items: coordinator.charged_items,
        deadline_polls: coordinator.deadline_polls,
    }
}

/// Run the default-off real-model probe with the system monotonic clock.
pub fn probe_cgan_nch1_sequential_unwired(
    model: &OnnxModel,
    graph: &GraphNetwork,
    input: &CertifiedInputBox,
    moat: CertifiedScalarMoat,
    limits: CganCzSequentialLimits,
    budget: ConstrainedZonotopeCallBudget,
) -> CganCzProbeReport {
    probe_cgan_nch1_sequential_with_clock(
        model,
        graph,
        input,
        moat,
        limits,
        budget,
        ProbeExtent::Full,
        |_| Instant::now(),
    )
}

/// Propagate every protected-latent orthant through the sealed full probe.
///
/// This remains an unwired, diagnostic-only experiment. The topology and raw
/// FLOAT provenance are sealed once, the exact 32-leaf dyadic cover is formed,
/// and each leaf then executes the same prefix and M17 tail used by
/// [`probe_cgan_nch1_sequential_unwired`]. No aggregate or leaf bounds are
/// published unless all 32 executions and the final publication checkpoint
/// complete.
pub fn probe_cgan_nch1_protected_latent_cover_unwired(
    model: &OnnxModel,
    graph: &GraphNetwork,
    input: &CertifiedInputBox,
    moat: CertifiedScalarMoat,
    limits: CganCzProtectedAlphaProbeLimits,
    budget: ConstrainedZonotopeCallBudget,
) -> CganCzProtectedAlphaProbeReport {
    probe_cgan_nch1_protected_alpha_plan_unwired(
        model,
        graph,
        input,
        moat,
        &CGAN_NCH1_PROTECTED_LATENT_COVER_AXES,
        limits,
        budget,
    )
}

/// Propagate an exact complete dyadic split plan through the sealed full probe.
///
/// Repeated axes are admitted: every bisection replaces one complete domain by
/// its two exact children, so any successfully enumerated plan remains a cover
/// of the original five-latent input. This surface is diagnostic-only and
/// retains the same all-or-nothing publication rule as the fixed 32-orthant
/// wrapper.
pub fn probe_cgan_nch1_protected_alpha_plan_unwired(
    model: &OnnxModel,
    graph: &GraphNetwork,
    input: &CertifiedInputBox,
    moat: CertifiedScalarMoat,
    split_axes: &[usize],
    limits: CganCzProtectedAlphaProbeLimits,
    budget: ConstrainedZonotopeCallBudget,
) -> CganCzProtectedAlphaProbeReport {
    probe_cgan_nch1_protected_latent_cover_with_clock(
        model,
        graph,
        input,
        moat,
        split_axes,
        limits,
        budget,
        |_| Instant::now(),
    )
}

fn probe_cgan_nch1_protected_latent_cover_with_clock<N>(
    model: &OnnxModel,
    graph: &GraphNetwork,
    input: &CertifiedInputBox,
    moat: CertifiedScalarMoat,
    split_axes: &[usize],
    limits: CganCzProtectedAlphaProbeLimits,
    budget: ConstrainedZonotopeCallBudget,
    now: N,
) -> CganCzProtectedAlphaProbeReport
where
    N: FnMut(&'static str) -> Instant,
{
    let mut coordinator = Coordinator::new(budget, now);
    let mut parameter_elements = 0_usize;
    let requested_leaf_domains = u32::try_from(split_axes.len())
        .ok()
        .and_then(|levels| 1_usize.checked_shl(levels));
    let result = (|| {
        validate_limits(limits.sequential)
            .map_err(|reason| protected_alpha_leaf_decline(None, "admission", reason))?;
        let requested_leaf_domains = requested_leaf_domains.ok_or_else(|| {
            protected_alpha_leaf_decline(
                None,
                "admission",
                CganCzProbeDecline::ResourceOverflow {
                    operation: "protected-alpha requested leaf domains",
                },
            )
        })?;
        if limits.cover.protected_alpha_dim != PROTECTED_LATENT_SYMBOLS {
            return Err(protected_alpha_leaf_decline(
                None,
                "admission",
                CganCzProbeDecline::InvalidLimit {
                    message: format!(
                        "cGAN protected-alpha propagation requires protected_alpha_dim={}, got {}",
                        PROTECTED_LATENT_SYMBOLS, limits.cover.protected_alpha_dim
                    ),
                },
            ));
        }
        check_resource(
            "protected-alpha complete leaf propagations",
            requested_leaf_domains,
            limits.max_leaf_propagations,
        )
        .map_err(|reason| protected_alpha_leaf_decline(None, "admission", reason))?;
        coordinator
            .checkpoint("cGAN protected-alpha propagation admission")
            .map_err(|reason| protected_alpha_leaf_decline(None, "admission", reason))?;
        coordinator
            .preflight_absolute_peak(budget.baseline_live_bytes())
            .map_err(|reason| protected_alpha_leaf_decline(None, "admission", reason))?;

        seal_topology(model, graph, limits.sequential, &mut coordinator)
            .map_err(|reason| protected_alpha_leaf_decline(None, "topology", reason))?;
        // Per-leaf telemetry is allocated only after the complete cover. The
        // private propagation coordinator accounts for its full reservation.
        let sealed = seal_parameters(model, graph, limits.sequential, 0, &mut coordinator)
            .map_err(|reason| protected_alpha_leaf_decline(None, "parameter provenance", reason))?;
        parameter_elements = sealed.parameter_elements;
        let domain = build_input_domain(input, limits.sequential, &sealed, 0, &mut coordinator)
            .map_err(|reason| protected_alpha_leaf_decline(None, "input", reason))?;
        let retained_baseline = budget
            .baseline_live_bytes()
            .checked_add(sealed.live_bytes)
            .ok_or_else(|| {
                protected_alpha_leaf_decline(
                    None,
                    "input",
                    CganCzProbeDecline::ResourceOverflow {
                        operation: "protected-alpha sealed retained baseline",
                    },
                )
            })?;
        coordinator
            .preflight_absolute_peak(
                retained_baseline
                    .checked_add(
                        domain_live_bytes(&domain).map_err(|reason| {
                            protected_alpha_leaf_decline(None, "input", reason)
                        })?,
                    )
                    .ok_or_else(|| {
                        protected_alpha_leaf_decline(
                            None,
                            "input",
                            CganCzProbeDecline::ResourceOverflow {
                                operation: "protected-alpha retained input-domain peak",
                            },
                        )
                    })?,
            )
            .map_err(|reason| protected_alpha_leaf_decline(None, "input", reason))?;

        let cover = enumerate_cgan_cz_protected_alpha_cover_with_coordinator(
            &domain,
            split_axes,
            limits.cover,
            retained_baseline,
            &mut coordinator,
        )
        .map_err(|reason| protected_alpha_leaf_decline(None, "protected-alpha cover", reason))?;
        drop(domain);

        propagate_cgan_cz_complete_cover_with(
            cover,
            split_axes,
            moat,
            limits.max_leaf_propagations,
            retained_baseline,
            &mut coordinator,
            |_, mut leaf, leaf_retained_baseline, coordinator| {
                if leaf.value_dim() != PROTECTED_LATENT_SYMBOLS
                    || leaf.alpha_dim() != PROTECTED_LATENT_SYMBOLS
                {
                    return Err((
                        "protected-alpha cover",
                        CganCzProbeDecline::Transform {
                            node: "protected-alpha cover",
                            operation: "leaf input geometry audit",
                            message: format!(
                                "leaf has value_dim={} and alpha_dim={}, expected five of each",
                                leaf.value_dim(),
                                leaf.alpha_dim()
                            ),
                        },
                    ));
                }
                let mut shape = vec![PROTECTED_LATENT_SYMBOLS];
                let mut stages = Vec::new();
                stages.try_reserve_exact(MAX_RUNNER_STAGES).map_err(|_| {
                    (
                        "protected-alpha cover",
                        CganCzProbeDecline::ResourceLimit {
                            resource: "protected-alpha per-leaf stage telemetry allocation",
                            required: MAX_RUNNER_STAGES,
                            limit: MAX_RUNNER_STAGES - 1,
                        },
                    )
                })?;
                let actual_stage_bytes = stages
                    .capacity()
                    .checked_mul(size_of::<CganCzStageTelemetry>())
                    .ok_or((
                        "protected-alpha cover",
                        CganCzProbeDecline::ResourceOverflow {
                            operation: "protected-alpha actual stage telemetry storage",
                        },
                    ))?;
                if actual_stage_bytes > RUNNER_TELEMETRY_RESERVED_BYTES {
                    return Err((
                        "protected-alpha cover",
                        CganCzProbeDecline::ResourceLimit {
                            resource: "protected-alpha actual stage telemetry bytes",
                            required: actual_stage_bytes,
                            limit: RUNNER_TELEMETRY_RESERVED_BYTES,
                        },
                    ));
                }
                for index in 0..ProbeExtent::Full.prefix_end_exclusive() {
                    let node = EXPECTED_NODES[index].0;
                    apply_prefix_layer(
                        index,
                        &sealed.layers[index],
                        &mut leaf,
                        &mut shape,
                        relu_reduction_target(ProbeExtent::Full, index, limits.sequential),
                        limits.sequential,
                        leaf_retained_baseline,
                        budget,
                        coordinator,
                        &mut stages,
                    )
                    .map_err(|reason| (node, reason))?;
                }

                let tail_bn = match &sealed.layers[23] {
                    SealedLayer::BatchNorm(parameters) => parameters,
                    _ => unreachable!("sealed topology fixes BatchNormalization_24"),
                };
                let tail_affine = match &sealed.layers[25] {
                    SealedLayer::Affine(parameters) => parameters,
                    _ => unreachable!("sealed topology fixes Gemm_27"),
                };
                let bounds = bound_output_tail(
                    &leaf,
                    None,
                    None,
                    tail_bn,
                    tail_affine,
                    moat,
                    limits.sequential,
                    leaf_retained_baseline,
                    budget,
                    coordinator,
                    &mut stages,
                )
                .map_err(|reason| ("Relu_23", reason))?
                .completed;
                let actual_stage_storage = stages
                    .capacity()
                    .checked_mul(size_of::<CganCzStageTelemetry>())
                    .and_then(|bytes| {
                        stages.iter().try_fold(bytes, |total, stage| {
                            stage
                                .output_shape
                                .capacity()
                                .checked_mul(size_of::<usize>())
                                .and_then(|shape_bytes| total.checked_add(shape_bytes))
                        })
                    })
                    .ok_or((
                        "protected-alpha cover",
                        CganCzProbeDecline::ResourceOverflow {
                            operation: "protected-alpha nested stage telemetry storage",
                        },
                    ))?;
                if actual_stage_storage > RUNNER_TELEMETRY_RESERVED_BYTES {
                    return Err((
                        "protected-alpha cover",
                        CganCzProbeDecline::ResourceLimit {
                            resource: "protected-alpha nested stage telemetry bytes",
                            required: actual_stage_storage,
                            limit: RUNNER_TELEMETRY_RESERVED_BYTES,
                        },
                    ));
                }
                Ok((bounds, stages.len()))
            },
        )
    })();

    let status = match result {
        Ok(completed) => CganCzProtectedAlphaProbeStatus::Completed(completed),
        Err(decline) => CganCzProtectedAlphaProbeStatus::Declined {
            leaf_index: decline.leaf_index,
            node: decline.node,
            reason: decline.reason,
        },
    };
    CganCzProtectedAlphaProbeReport {
        authority: CGAN_CZ_VERDICT_AUTHORITY,
        status,
        topology_work_items: coordinator.topology_work_items,
        parameter_elements,
        protected_latent_symbols: PROTECTED_LATENT_SYMBOLS,
        requested_leaf_domains: requested_leaf_domains.unwrap_or(usize::MAX),
        peak_live_bytes: coordinator.peak_live_bytes,
        charged_items: coordinator.charged_items,
        deadline_polls: coordinator.deadline_polls,
    }
}

/// Run through the first authored ConvTranspose/BatchNorm/ReLU block.
///
/// This is the deliberately narrow qualification milestone: it still seals
/// the complete 26-node topology and every raw FLOAT parameter before
/// executing nodes 0 through 5. The five latent symbols are protected across
/// the post-ReLU order reduction.
pub fn probe_cgan_nch1_first_block_unwired(
    model: &OnnxModel,
    graph: &GraphNetwork,
    input: &CertifiedInputBox,
    moat: CertifiedScalarMoat,
    limits: CganCzSequentialLimits,
    budget: ConstrainedZonotopeCallBudget,
) -> CganCzProbeReport {
    probe_cgan_nch1_sequential_with_clock(
        model,
        graph,
        input,
        moat,
        limits,
        budget,
        ProbeExtent::FirstGeneratorBlock,
        |_| Instant::now(),
    )
}

/// Run through the second authored ConvTranspose/BatchNorm/ReLU block.
///
/// Like the first-block milestone, this qualification-only path seals the
/// complete topology and every raw FLOAT parameter. It executes nodes 0
/// through 8, protects the five latent symbols across both order reductions,
/// and cannot publish a verifier verdict.
pub fn probe_cgan_nch1_second_block_unwired(
    model: &OnnxModel,
    graph: &GraphNetwork,
    input: &CertifiedInputBox,
    moat: CertifiedScalarMoat,
    limits: CganCzSequentialLimits,
    budget: ConstrainedZonotopeCallBudget,
) -> CganCzProbeReport {
    probe_cgan_nch1_sequential_with_clock(
        model,
        graph,
        input,
        moat,
        limits,
        budget,
        ProbeExtent::SecondGeneratorBlock,
        |_| Instant::now(),
    )
}

/// Tight fail-closed limits qualified for the third generator-block probe.
///
/// These limits are deliberately much smaller than the primitive hard
/// ceilings. In particular, at most 2,043 new ReLU symbols can coexist with
/// the five protected latent symbols. Exceeding any cap is a typed diagnostic
/// decline and never grants verdict authority.
#[must_use]
pub const fn cgan_nch1_third_block_qualification_limits() -> CganCzSequentialLimits {
    CganCzSequentialLimits {
        max_graph_nodes: 26,
        max_graph_edges: 128,
        max_topology_work_items: 1 << 20,
        max_parameter_elements: 600_000,
        max_value_dim: 28_800,
        max_transient_alpha_dim: 2_048,
        retained_alpha_dim: 5,
        max_generator_nonzeros: 150_000,
        max_interval_products_per_stage: 33_000_000,
        max_exact_terms_per_relu: 500_000,
        max_m17_iterations: 8,
        max_m17_search_work: 10_000_000,
    }
}

/// Tight fail-closed limits qualified for the generator/discriminator handoff.
///
/// The handoff reuses the unchanged third-block qualification caps. The
/// additional `ConvTranspose_13 -> Conv_14 -> Relu_15` segment fits beneath
/// those caps; widening them would weaken the fail-closed contract without
/// admitting any part of this exact model.
#[must_use]
pub const fn cgan_nch1_generator_discriminator_handoff_qualification_limits(
) -> CganCzSequentialLimits {
    cgan_nch1_third_block_qualification_limits()
}

/// Tight fail-closed limits for the independent interval-CZ prefix.
///
/// Zero alpha symbols make the correlation ceilings inert, while the exact
/// 28,800-coordinate generator maximum and all sealed linear-stage work remain
/// within the already qualified third-block envelope.
#[must_use]
pub const fn cgan_nch1_independent_interval_qualification_limits() -> CganCzSequentialLimits {
    cgan_nch1_third_block_qualification_limits()
}

/// Tight fail-closed limits for the imgSz32 nCh3 independent interval prefix.
///
/// The authored image has three channels only between `ConvTranspose_13` and
/// `Conv_14`; the 28,800-coordinate generator maximum and the established
/// per-stage work ceiling are unchanged.
#[must_use]
pub const fn cgan_nch3_independent_interval_qualification_limits() -> CganCzSequentialLimits {
    cgan_nch1_independent_interval_qualification_limits()
}

/// Run through the third authored ConvTranspose/BatchNorm/ReLU block.
///
/// This qualification-only path seals the complete topology and every raw
/// FLOAT parameter, executes nodes 0 through 11, and protects the five latent
/// symbols across all three order reductions. It remains default-off and
/// cannot publish a verifier verdict.
pub fn probe_cgan_nch1_third_block_unwired(
    model: &OnnxModel,
    graph: &GraphNetwork,
    input: &CertifiedInputBox,
    moat: CertifiedScalarMoat,
    limits: CganCzSequentialLimits,
    budget: ConstrainedZonotopeCallBudget,
) -> CganCzProbeReport {
    probe_cgan_nch1_sequential_with_clock(
        model,
        graph,
        input,
        moat,
        limits,
        budget,
        ProbeExtent::ThirdGeneratorBlock,
        |_| Instant::now(),
    )
}

/// Run through the exact generator/discriminator boundary.
///
/// This qualification-only path seals the complete topology and every raw
/// FLOAT parameter, executes nodes 0 through 14, and ends after
/// `ConvTranspose_13 -> Conv_14 -> Relu_15`. There is no BatchNorm in this
/// authored segment. The five latent symbols remain protected across the
/// fourth order reduction. The probe remains default-off and has no verifier
/// verdict authority.
pub fn probe_cgan_nch1_generator_discriminator_handoff_unwired(
    model: &OnnxModel,
    graph: &GraphNetwork,
    input: &CertifiedInputBox,
    moat: CertifiedScalarMoat,
    limits: CganCzSequentialLimits,
    budget: ConstrainedZonotopeCallBudget,
) -> CganCzProbeReport {
    probe_cgan_nch1_sequential_with_clock(
        model,
        graph,
        input,
        moat,
        limits,
        budget,
        ProbeExtent::GeneratorDiscriminatorHandoff,
        |_| Instant::now(),
    )
}

fn probe_cgan_nch1_sequential_with_clock<N>(
    model: &OnnxModel,
    graph: &GraphNetwork,
    input: &CertifiedInputBox,
    moat: CertifiedScalarMoat,
    limits: CganCzSequentialLimits,
    budget: ConstrainedZonotopeCallBudget,
    extent: ProbeExtent,
    now: N,
) -> CganCzProbeReport
where
    N: FnMut(&'static str) -> Instant,
{
    let mut stages = Vec::new();
    let mut parameter_elements = 0;
    let mut at_node = "admission";
    let mut coordinator = Coordinator::new(budget, now);
    let result = (|| {
        validate_limits(limits)?;
        coordinator.checkpoint("cGAN admission")?;
        coordinator.preflight_absolute_peak(budget.baseline_live_bytes())?;
        stages.try_reserve_exact(MAX_RUNNER_STAGES).map_err(|_| {
            CganCzProbeDecline::ResourceLimit {
                resource: "cGAN stage telemetry allocation",
                required: MAX_RUNNER_STAGES,
                limit: MAX_RUNNER_STAGES - 1,
            }
        })?;
        let telemetry_bytes = MAX_RUNNER_STAGES
            .checked_mul(RUNNER_STAGE_RESERVED_BYTES)
            .ok_or(CganCzProbeDecline::ResourceOverflow {
                operation: "cGAN stage telemetry bytes",
            })?;
        coordinator.preflight_absolute_peak(
            budget
                .baseline_live_bytes()
                .checked_add(telemetry_bytes)
                .ok_or(CganCzProbeDecline::ResourceOverflow {
                    operation: "cGAN baseline plus telemetry",
                })?,
        )?;

        at_node = "topology";
        seal_topology(model, graph, limits, &mut coordinator)?;
        at_node = "parameter provenance";
        let sealed = seal_parameters(model, graph, limits, telemetry_bytes, &mut coordinator)?;
        parameter_elements = sealed.parameter_elements;

        at_node = "input";
        let mut domain =
            build_input_domain(input, limits, &sealed, telemetry_bytes, &mut coordinator)?;
        let mut shape = vec![PROTECTED_LATENT_SYMBOLS];
        let retained_baseline = budget
            .baseline_live_bytes()
            .checked_add(telemetry_bytes)
            .and_then(|bytes| bytes.checked_add(sealed.live_bytes))
            .ok_or(CganCzProbeDecline::ResourceOverflow {
                operation: "cGAN retained baseline",
            })?;
        coordinator.preflight_absolute_peak(
            retained_baseline
                .checked_add(domain_live_bytes(&domain)?)
                .ok_or(CganCzProbeDecline::ResourceOverflow {
                    operation: "cGAN retained domain peak",
                })?,
        )?;

        let prefix_end = extent.prefix_end_exclusive();
        for index in 0..prefix_end {
            at_node = EXPECTED_NODES[index].0;
            apply_prefix_layer(
                index,
                &sealed.layers[index],
                &mut domain,
                &mut shape,
                relu_reduction_target(extent, index, limits),
                limits,
                retained_baseline,
                budget,
                &mut coordinator,
                &mut stages,
            )?;
        }

        if let Some(last_index) = extent.prefix_last_index() {
            at_node = EXPECTED_NODES[last_index].0;
            let completion = summarize_prefix(
                &domain,
                &shape,
                last_index,
                moat,
                retained_baseline,
                &mut coordinator,
            )?;
            return Ok::<_, CganCzProbeDecline>(ProbeCompletion::Prefix(completion));
        }

        at_node = "Relu_23";
        let tail_bn = match &sealed.layers[23] {
            SealedLayer::BatchNorm(parameters) => parameters,
            _ => unreachable!("sealed topology fixes BatchNormalization_24"),
        };
        let tail_affine = match &sealed.layers[25] {
            SealedLayer::Affine(parameters) => parameters,
            _ => unreachable!("sealed topology fixes Gemm_27"),
        };
        let completed = bound_output_tail(
            &domain,
            None,
            None,
            tail_bn,
            tail_affine,
            moat,
            limits,
            retained_baseline,
            budget,
            &mut coordinator,
            &mut stages,
        )?
        .completed;
        Ok::<_, CganCzProbeDecline>(ProbeCompletion::Full(completed))
    })();

    match result {
        Ok(completion) => CganCzProbeReport {
            authority: CGAN_CZ_VERDICT_AUTHORITY,
            status: match completion {
                ProbeCompletion::Prefix(completed) => CganCzProbeStatus::PrefixCompleted(completed),
                ProbeCompletion::Full(completed) => CganCzProbeStatus::Completed(completed),
            },
            stages,
            topology_work_items: coordinator.topology_work_items,
            parameter_elements,
            protected_latent_symbols: PROTECTED_LATENT_SYMBOLS,
            peak_live_bytes: coordinator.peak_live_bytes,
            charged_items: coordinator.charged_items,
            deadline_polls: coordinator.deadline_polls,
        },
        Err(reason) => CganCzProbeReport {
            authority: CGAN_CZ_VERDICT_AUTHORITY,
            status: CganCzProbeStatus::Declined {
                node: at_node,
                reason,
            },
            stages,
            topology_work_items: coordinator.topology_work_items,
            parameter_elements,
            protected_latent_symbols: PROTECTED_LATENT_SYMBOLS,
            peak_live_bytes: coordinator.peak_live_bytes,
            charged_items: coordinator.charged_items,
            deadline_polls: coordinator.deadline_polls,
        },
    }
}

fn validate_limits(limits: CganCzSequentialLimits) -> Result<(), CganCzProbeDecline> {
    for (name, supplied, hard) in [
        (
            "max_graph_nodes",
            limits.max_graph_nodes,
            RUNNER_HARD_MAX_GRAPH_NODES,
        ),
        (
            "max_graph_edges",
            limits.max_graph_edges,
            RUNNER_HARD_MAX_GRAPH_EDGES,
        ),
        (
            "max_topology_work_items",
            limits.max_topology_work_items,
            RUNNER_HARD_MAX_TOPOLOGY_WORK,
        ),
        (
            "max_parameter_elements",
            limits.max_parameter_elements,
            RUNNER_HARD_MAX_PARAMETER_ELEMENTS,
        ),
    ] {
        if supplied > hard {
            return Err(CganCzProbeDecline::InvalidLimit {
                message: format!("{name}={supplied} exceeds hard maximum {hard}"),
            });
        }
    }
    if limits.retained_alpha_dim < PROTECTED_LATENT_SYMBOLS {
        return Err(CganCzProbeDecline::InvalidLimit {
            message: format!(
                "retained alpha dimension {} cannot protect five latent symbols",
                limits.retained_alpha_dim
            ),
        });
    }
    if limits.max_transient_alpha_dim < limits.retained_alpha_dim
        || limits.max_value_dim < 512
        || limits.max_generator_nonzeros == 0
        || limits.max_interval_products_per_stage == 0
        || limits.max_exact_terms_per_relu == 0
        || limits.max_m17_search_work == 0
    {
        return Err(CganCzProbeDecline::InvalidLimit {
            message: "nonzero transform limits must cover retained dimensions".to_string(),
        });
    }
    Ok(())
}

fn seal_topology<N>(
    model: &OnnxModel,
    graph: &GraphNetwork,
    limits: CganCzSequentialLimits,
    coordinator: &mut Coordinator<N>,
) -> Result<(), CganCzProbeDecline>
where
    N: FnMut(&'static str) -> Instant,
{
    seal_topology_for_profile(
        CganCzImgSz32Profile::Nch1,
        model,
        graph,
        limits,
        coordinator,
    )
}

fn seal_topology_for_profile<N>(
    profile: CganCzImgSz32Profile,
    model: &OnnxModel,
    graph: &GraphNetwork,
    limits: CganCzSequentialLimits,
    coordinator: &mut Coordinator<N>,
) -> Result<(), CganCzProbeDecline>
where
    N: FnMut(&'static str) -> Instant,
{
    match model.original_network_topology_matches_current() {
        Some(true) => {}
        None => {
            return Err(CganCzProbeDecline::Provenance {
                message: "model lacks the loader-private finalized-network snapshot".to_string(),
            });
        }
        Some(false) => {
            return Err(CganCzProbeDecline::Provenance {
                message: "public model network changed after provenance capture".to_string(),
            });
        }
    }
    check_resource(
        "model layer count",
        model.network.layers.len(),
        limits.max_graph_nodes,
    )?;
    check_resource(
        "graph node count",
        graph.num_nodes(),
        limits.max_graph_nodes,
    )?;
    if model.network.layers.len() != EXPECTED_NODE_COUNT || graph.num_nodes() != EXPECTED_NODE_COUNT
    {
        return Err(CganCzProbeDecline::Topology {
            message: format!(
                "expected exactly {EXPECTED_NODE_COUNT} nodes, got raw={} graph={}",
                model.network.layers.len(),
                graph.num_nodes()
            ),
        });
    }
    if model.network.inputs.len() != 1
        || model.network.inputs[0].name != "X"
        || model.network.inputs[0].shape != [1, 5]
        || model.network.inputs[0].dtype != DataType::Float32
        || model.network.outputs.len() != 1
        || model.network.outputs[0].name != "Y"
        || model.network.outputs[0].shape != [1, 1]
        || model.network.outputs[0].dtype != DataType::Float32
    {
        return Err(CganCzProbeDecline::Topology {
            message: "network boundary must be Float32 X[1,5] -> Y[1,1]".to_string(),
        });
    }

    let order = graph
        .exec_order()
        .map_err(|error| CganCzProbeDecline::Topology {
            message: format!("normalized graph has no unique execution order: {error}"),
        })?;
    if order.len() != EXPECTED_NODE_COUNT {
        return Err(CganCzProbeDecline::Topology {
            message: format!("execution order has {} nodes", order.len()),
        });
    }

    let mut raw_activation = model.network.inputs[0].name.as_str();
    let mut graph_activation = NETWORK_INPUT;
    let mut edge_count = 0_usize;
    for (index, ((raw, graph_name), (expected_name, expected_type))) in model
        .network
        .layers
        .iter()
        .zip(order)
        .zip(EXPECTED_NODES.iter())
        .enumerate()
    {
        coordinator.charge_topology(
            raw.name
                .len()
                .checked_add(raw.inputs.len())
                .and_then(|work| work.checked_add(raw.outputs.len()))
                .and_then(|work| work.checked_add(raw.attributes.capacity()))
                .ok_or(CganCzProbeDecline::ResourceOverflow {
                    operation: "cGAN raw topology work",
                })?,
        )?;
        edge_count = edge_count.checked_add(raw.inputs.len()).ok_or(
            CganCzProbeDecline::ResourceOverflow {
                operation: "cGAN raw edge count",
            },
        )?;
        check_resource("model/graph edge count", edge_count, limits.max_graph_edges)?;
        if raw.name != *expected_name
            || raw.layer_type != *expected_type
            || raw.inputs.first().map(String::as_str) != Some(raw_activation)
            || raw.outputs.len() != 1
            || raw.outputs[0].is_empty()
        {
            return Err(CganCzProbeDecline::Topology {
                message: format!(
                    "raw node {index} must be {expected_name}/{expected_type} after {raw_activation}; got {} / {} with inputs {:?}, outputs {:?}",
                    raw.name, raw.layer_type, raw.inputs, raw.outputs
                ),
            });
        }

        if graph_name != expected_name {
            return Err(CganCzProbeDecline::Topology {
                message: format!("graph node {index} is {graph_name}, expected {expected_name}"),
            });
        }
        let node = graph
            .node(graph_name)
            .ok_or_else(|| CganCzProbeDecline::Topology {
                message: format!("execution order references missing graph node {graph_name}"),
            })?;
        coordinator.charge_topology(graph_name.len().checked_add(node.inputs().len()).ok_or(
            CganCzProbeDecline::ResourceOverflow {
                operation: "cGAN graph topology work",
            },
        )?)?;
        edge_count = edge_count.checked_add(node.inputs().len()).ok_or(
            CganCzProbeDecline::ResourceOverflow {
                operation: "cGAN graph edge count",
            },
        )?;
        check_resource("model/graph edge count", edge_count, limits.max_graph_edges)?;
        if node.inputs() != [graph_activation]
            || node.layer().layer_type() != expected_type.to_string()
        {
            return Err(CganCzProbeDecline::Topology {
                message: format!(
                    "graph node {graph_name} must be {expected_type} after {graph_activation}; got {} with inputs {:?}",
                    node.layer().layer_type(),
                    node.inputs()
                ),
            });
        }

        let expected_shape = expected_output_shape_for_profile(profile, index);
        let mut batched_shape = Vec::with_capacity(expected_shape.len() + 1);
        batched_shape.push(1_i64);
        for &dimension in expected_shape {
            batched_shape.push(i64::try_from(dimension).map_err(|_| {
                CganCzProbeDecline::ResourceOverflow {
                    operation: "cGAN expected tensor dimension",
                }
            })?);
        }
        if model.tensor_shapes().get(&raw.outputs[0]) != Some(&batched_shape) {
            return Err(CganCzProbeDecline::Topology {
                message: format!(
                    "raw node {expected_name} output '{}' shape {:?} does not match {:?}",
                    raw.outputs[0],
                    model.tensor_shapes().get(&raw.outputs[0]),
                    batched_shape
                ),
            });
        }
        raw_activation = &raw.outputs[0];
        graph_activation = expected_name;
    }
    if raw_activation != "Y" || graph.output_name() != "Gemm_27" {
        return Err(CganCzProbeDecline::Topology {
            message: format!(
                "sealed chain must end at Y/Gemm_27, got {raw_activation}/{}",
                graph.output_name()
            ),
        });
    }
    check_resource(
        "model/graph topology work items",
        coordinator.topology_work_items,
        limits.max_topology_work_items,
    )?;
    coordinator.checkpoint("cGAN topology seal complete")
}

fn check_resource(
    resource: &'static str,
    required: usize,
    limit: usize,
) -> Result<(), CganCzProbeDecline> {
    if required > limit {
        Err(CganCzProbeDecline::ResourceLimit {
            resource,
            required,
            limit,
        })
    } else {
        Ok(())
    }
}

fn seal_parameters<N>(
    model: &OnnxModel,
    graph: &GraphNetwork,
    limits: CganCzSequentialLimits,
    telemetry_bytes: usize,
    coordinator: &mut Coordinator<N>,
) -> Result<SealedCgan, CganCzProbeDecline>
where
    N: FnMut(&'static str) -> Instant,
{
    seal_parameters_for_profile(
        CganCzImgSz32Profile::Nch1,
        model,
        graph,
        limits,
        telemetry_bytes,
        coordinator,
    )
}

fn seal_parameters_for_profile<N>(
    profile: CganCzImgSz32Profile,
    model: &OnnxModel,
    graph: &GraphNetwork,
    limits: CganCzSequentialLimits,
    telemetry_bytes: usize,
    coordinator: &mut Coordinator<N>,
) -> Result<SealedCgan, CganCzProbeDecline>
where
    N: FnMut(&'static str) -> Instant,
{
    let mut parameter_elements = 0_usize;
    let mut normalized_bn_elements = 0_usize;
    for (index, raw) in model.network.layers.iter().enumerate() {
        if matches!(
            EXPECTED_NODES[index].1,
            LayerType::Linear
                | LayerType::BatchNorm
                | LayerType::Conv2d
                | LayerType::ConvTranspose2d
        ) {
            for name in raw.inputs.iter().skip(1) {
                require_float_provenance(model, name)?;
                let values =
                    model
                        .weights
                        .get(name)
                        .ok_or_else(|| CganCzProbeDecline::Provenance {
                            message: format!(
                                "{} parameter '{name}' is not a current FLOAT initializer",
                                raw.name
                            ),
                        })?;
                parameter_elements = parameter_elements.checked_add(values.len()).ok_or(
                    CganCzProbeDecline::ResourceOverflow {
                        operation: "cGAN raw FLOAT parameter elements",
                    },
                )?;
                coordinator.charge(values.len(), "cGAN parameter inventory")?;
            }
            if raw.layer_type == LayerType::BatchNorm {
                let channels = model
                    .weights
                    .get(&raw.inputs[1])
                    .map(ndarray::ArrayBase::len)
                    .unwrap_or(0);
                // Only the graph's normalized scale and bias remain live.
                // Its rounded error arrays are compared for provenance during
                // sealing, then discarded because proof construction uses the
                // independent exact surrogate certificate below.
                normalized_bn_elements = normalized_bn_elements
                    .checked_add(channels.checked_mul(2).ok_or(
                        CganCzProbeDecline::ResourceOverflow {
                            operation: "cGAN normalized BatchNorm elements",
                        },
                    )?)
                    .ok_or(CganCzProbeDecline::ResourceOverflow {
                        operation: "cGAN normalized BatchNorm elements",
                    })?;
            }
        }
    }
    check_resource(
        "raw FLOAT parameter elements",
        parameter_elements,
        limits.max_parameter_elements,
    )?;

    let promoted_elements = parameter_elements
        .checked_add(normalized_bn_elements)
        .ok_or(CganCzProbeDecline::ResourceOverflow {
            operation: "cGAN promoted parameter elements",
        })?;
    let layer_storage = EXPECTED_NODE_COUNT
        .checked_mul(size_of::<SealedLayer>())
        .ok_or(CganCzProbeDecline::ResourceOverflow {
            operation: "cGAN sealed-layer storage",
        })?;
    let live_bytes = promoted_elements
        .checked_mul(size_of::<f64>())
        .and_then(|bytes| bytes.checked_add(layer_storage))
        .ok_or(CganCzProbeDecline::ResourceOverflow {
            operation: "cGAN sealed parameter bytes",
        })?;
    let seal_scratch = 128_usize
        .checked_mul(12)
        .and_then(|items| items.checked_mul(size_of::<f32>()))
        .ok_or(CganCzProbeDecline::ResourceOverflow {
            operation: "cGAN BatchNorm seal scratch",
        })?;
    let required = coordinator
        .budget
        .baseline_live_bytes()
        .checked_add(telemetry_bytes)
        .and_then(|bytes| bytes.checked_add(live_bytes))
        .and_then(|bytes| bytes.checked_add(seal_scratch))
        .ok_or(CganCzProbeDecline::ResourceOverflow {
            operation: "cGAN parameter seal peak",
        })?;
    coordinator.preflight_absolute_peak(required)?;
    coordinator.checkpoint("cGAN parameter seal allocation")?;

    let mut layers = Vec::new();
    layers.try_reserve_exact(EXPECTED_NODE_COUNT).map_err(|_| {
        CganCzProbeDecline::ResourceLimit {
            resource: "sealed cGAN layer allocation",
            required: EXPECTED_NODE_COUNT,
            limit: EXPECTED_NODE_COUNT - 1,
        }
    })?;
    for (index, raw) in model.network.layers.iter().enumerate() {
        let graph_node =
            graph
                .node(EXPECTED_NODES[index].0)
                .ok_or_else(|| CganCzProbeDecline::Topology {
                    message: format!("missing normalized node {}", EXPECTED_NODES[index].0),
                })?;
        let sealed = match EXPECTED_NODES[index].1 {
            LayerType::Linear => {
                SealedLayer::Affine(seal_affine(model, raw, graph_node.layer(), coordinator)?)
            }
            LayerType::BatchNorm => SealedLayer::BatchNorm(seal_batch_norm(
                model,
                raw,
                graph_node.layer(),
                coordinator,
            )?),
            LayerType::Conv2d => {
                SealedLayer::Conv2d(seal_conv2d(model, raw, graph_node.layer(), coordinator)?)
            }
            LayerType::ConvTranspose2d => SealedLayer::ConvTranspose2d(seal_conv_transpose2d(
                model,
                raw,
                graph_node.layer(),
                coordinator,
            )?),
            LayerType::Reshape => {
                let Layer::Reshape(reshape) = graph_node.layer() else {
                    unreachable!("topology seal fixed the layer type")
                };
                if raw.inputs.len() != 2
                    || raw.weights.is_some()
                    || raw.attributes.iter().any(|(name, value)| {
                        name != "allowzero" || value != &AttributeValue::Int(0)
                    })
                {
                    return Err(CganCzProbeDecline::Topology {
                        message: format!("{} is not a plain constant-shape Reshape", raw.name),
                    });
                }
                let input_shape = if index == 0 {
                    &[PROTECTED_LATENT_SYMBOLS][..]
                } else {
                    expected_output_shape_for_profile(profile, index - 1)
                };
                let computed = reshape.compute_output_shape(input_shape).map_err(|error| {
                    CganCzProbeDecline::Topology {
                        message: format!("{} target shape rejected: {error}", raw.name),
                    }
                })?;
                if computed != expected_output_shape_for_profile(profile, index) {
                    return Err(CganCzProbeDecline::Topology {
                        message: format!(
                            "{} normalized target produced {computed:?}, expected {:?}",
                            raw.name,
                            expected_output_shape_for_profile(profile, index)
                        ),
                    });
                }
                SealedLayer::Reshape(computed)
            }
            LayerType::ReLU => {
                if raw.inputs.len() != 1
                    || raw.outputs.len() != 1
                    || raw.weights.is_some()
                    || !raw.attributes.is_empty()
                    || !matches!(graph_node.layer(), Layer::ReLU(_))
                {
                    return Err(CganCzProbeDecline::Topology {
                        message: format!("{} is not a plain unary ReLU", raw.name),
                    });
                }
                SealedLayer::Relu
            }
            _ => unreachable!("the exact cGAN contract contains no other layer type"),
        };
        validate_imgsz32_profile_transition(profile, index, &sealed)?;
        layers.push(sealed);
        coordinator.checkpoint("cGAN parameter layer sealed")?;
    }
    coordinator.checkpoint("cGAN parameter provenance complete")?;
    Ok(SealedCgan {
        layers,
        parameter_elements,
        live_bytes,
    })
}

fn validate_imgsz32_profile_transition(
    profile: CganCzImgSz32Profile,
    index: usize,
    layer: &SealedLayer,
) -> Result<(), CganCzProbeDecline> {
    let channels = profile.image_channels();
    let accepted = match (index, layer) {
        (12, SealedLayer::ConvTranspose2d(parameters)) => {
            parameters.weights.dim() == (32, channels, 3, 3) && parameters.bias.len() == channels
        }
        (13, SealedLayer::Conv2d(parameters)) => {
            parameters.weights.dim() == (16, channels, 3, 3) && parameters.bias.len() == 16
        }
        (12 | 13, _) => false,
        _ => true,
    };
    if accepted {
        Ok(())
    } else {
        Err(CganCzProbeDecline::Topology {
            message: format!(
                "{} does not match the sealed imgSz32 nCh{} channel transition",
                EXPECTED_NODES[index].0, channels
            ),
        })
    }
}

fn relu_reduction_target(
    extent: ProbeExtent,
    index: usize,
    limits: CganCzSequentialLimits,
) -> Option<usize> {
    match (extent, index) {
        (ProbeExtent::Full, 14 | 16) => None,
        (ProbeExtent::Full, 19) => Some(
            limits
                .max_transient_alpha_dim
                .min(TAIL_DISCRIMINATOR_RETAINED_ALPHA_DIM),
        ),
        _ => Some(limits.retained_alpha_dim),
    }
}

fn require_float_provenance(model: &OnnxModel, name: &str) -> Result<(), CganCzProbeDecline> {
    match model.original_float32_initializer_matches_current(name) {
        Some(true) => Ok(()),
        None => Err(CganCzProbeDecline::Provenance {
            message: format!("parameter '{name}' has no raw ONNX FLOAT provenance"),
        }),
        Some(false) => Err(CganCzProbeDecline::Provenance {
            message: format!("parameter '{name}' changed after raw ONNX FLOAT capture"),
        }),
    }
}

fn seal_affine<N>(
    model: &OnnxModel,
    raw: &LayerSpec,
    graph_layer: &Layer,
    coordinator: &mut Coordinator<N>,
) -> Result<AffineParameters, CganCzProbeDecline>
where
    N: FnMut(&'static str) -> Instant,
{
    if raw.inputs.len() != 3
        || raw.weights.is_some()
        || raw.attributes.len() != 3
        || raw.attributes.get("alpha") != Some(&AttributeValue::Float(1.0))
        || raw.attributes.get("beta") != Some(&AttributeValue::Float(1.0))
        || raw.attributes.get("transB") != Some(&AttributeValue::Int(1))
    {
        return Err(CganCzProbeDecline::Topology {
            message: format!("{} must be Gemm(alpha=1,beta=1,transB=1)", raw.name),
        });
    }
    let Layer::Linear(graph) = graph_layer else {
        unreachable!("topology seal fixed the layer type")
    };
    let raw_weights = model
        .weights
        .get(&raw.inputs[1])
        .ok_or_else(|| missing_parameter(raw, &raw.inputs[1]))?;
    let raw_bias = model
        .weights
        .get(&raw.inputs[2])
        .ok_or_else(|| missing_parameter(raw, &raw.inputs[2]))?;
    let raw_weights = raw_weights
        .view()
        .into_dimensionality::<Ix2>()
        .map_err(|_| shape_error(raw, "Gemm weight", raw_weights.shape(), &[2]))?;
    let raw_bias = raw_bias
        .view()
        .into_dimensionality::<Ix1>()
        .map_err(|_| shape_error(raw, "Gemm bias", raw_bias.shape(), &[1]))?;
    if graph.weight().shape() != raw_weights.shape()
        || graph.bias().map(ndarray::ArrayBase::len) != Some(raw_bias.len())
    {
        return Err(CganCzProbeDecline::Topology {
            message: format!(
                "{} normalized Gemm shapes differ from raw initializers",
                raw.name
            ),
        });
    }
    compare_f32_bits(
        raw.name.as_str(),
        "Gemm weight",
        raw_weights.iter().copied(),
        graph.weight().iter().copied(),
        coordinator,
    )?;
    compare_f32_bits(
        raw.name.as_str(),
        "Gemm bias",
        raw_bias.iter().copied(),
        graph.bias().unwrap().iter().copied(),
        coordinator,
    )?;
    let weights = promote_array2(raw_weights, "cGAN Gemm weights", coordinator)?;
    let bias = promote_slice(
        raw_bias.iter().copied(),
        raw_bias.len(),
        "cGAN Gemm bias",
        coordinator,
    )?;
    Ok(AffineParameters { weights, bias })
}

fn seal_batch_norm<N>(
    model: &OnnxModel,
    raw: &LayerSpec,
    graph_layer: &Layer,
    coordinator: &mut Coordinator<N>,
) -> Result<BatchNormParameters, CganCzProbeDecline>
where
    N: FnMut(&'static str) -> Instant,
{
    let malformed_attribute = raw.attributes.iter().find(|(name, value)| {
        !match (name.as_str(), *value) {
            ("epsilon", AttributeValue::Float(value)) => value.is_finite() && *value > 0.0,
            // Momentum is inert for inference but still has to satisfy the
            // authenticated ONNX schema admitted by the loader.
            ("momentum", AttributeValue::Float(value)) => value.is_finite(),
            ("training_mode", AttributeValue::Int(0)) => true,
            (ONNX_BATCH_NORM_INPUT_RANK_ATTR, AttributeValue::Int(rank)) => {
                *rank == CGAN_AUTHORED_TENSOR_RANK as i64
            }
            _ => false,
        }
    });
    if raw.inputs.len() != 5
        || raw.weights.is_some()
        || malformed_attribute.is_some()
        || raw.attributes.get(ONNX_BATCH_NORM_INPUT_RANK_ATTR)
            != Some(&AttributeValue::Int(CGAN_AUTHORED_TENSOR_RANK as i64))
    {
        return Err(CganCzProbeDecline::Topology {
            message: format!(
                "{} must be authenticated rank-{CGAN_AUTHORED_TENSOR_RANK} inference BatchNormalization; got inputs={}, weights={:?}, malformed_attribute={malformed_attribute:?}, attributes={:?}",
                raw.name,
                raw.inputs.len(),
                raw.weights,
                raw.attributes,
            ),
        });
    }
    let epsilon = match raw.attributes.get("epsilon") {
        Some(AttributeValue::Float(value)) => *value,
        None => 1e-5,
        _ => unreachable!("attribute shape checked above"),
    };
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(CganCzProbeDecline::Topology {
            message: format!("{} has invalid epsilon {epsilon}", raw.name),
        });
    }
    let Layer::BatchNorm(graph) = graph_layer else {
        unreachable!("topology seal fixed the layer type")
    };
    if graph.channel_axis_hint
        != Some(ny_propagate::layers::BatchNormChannelAxisHint::OnnxNchw {
            authored_rank: CGAN_AUTHORED_TENSOR_RANK,
        })
    {
        return Err(CganCzProbeDecline::Topology {
            message: format!(
                "{} normalized BatchNorm lost its authenticated rank-{CGAN_AUTHORED_TENSOR_RANK} channel-axis provenance",
                raw.name
            ),
        });
    }
    let raw_input_shape =
        model
            .tensor_shapes()
            .get(&raw.inputs[0])
            .ok_or_else(|| CganCzProbeDecline::Topology {
                message: format!("{} raw input shape is unavailable", raw.name),
            })?;
    if raw_input_shape.len() != CGAN_AUTHORED_TENSOR_RANK
        || raw_input_shape[0] != 1
        || raw_input_shape.iter().any(|&dimension| dimension <= 0)
    {
        return Err(CganCzProbeDecline::Topology {
            message: format!(
                "{} requires a positive singleton-batch [N,C,H,W] input, got {raw_input_shape:?}",
                raw.name
            ),
        });
    }
    let mut raw_values = Vec::new();
    raw_values
        .try_reserve_exact(4)
        .map_err(|_| CganCzProbeDecline::ResourceLimit {
            resource: "BatchNorm raw view storage",
            required: 4,
            limit: 3,
        })?;
    for name in raw.inputs.iter().skip(1) {
        let value = model
            .weights
            .get(name)
            .ok_or_else(|| missing_parameter(raw, name))?;
        let value = value
            .view()
            .into_dimensionality::<Ix1>()
            .map_err(|_| shape_error(raw, "BatchNorm parameter", value.shape(), &[1]))?;
        raw_values.push(value);
    }
    let channels = raw_values[0].len();
    if channels == 0
        || usize::try_from(raw_input_shape[1]).ok() != Some(channels)
        || raw_values.iter().any(|values| values.len() != channels)
    {
        return Err(CganCzProbeDecline::Topology {
            message: format!(
                "{} BatchNorm parameter lengths disagree with input channels {}",
                raw.name, raw_input_shape[1]
            ),
        });
    }
    let reconstructed = ny_propagate::layers::BatchNormLayer::new(
        &raw_values[0].to_owned().into_dyn(),
        &raw_values[1].to_owned().into_dyn(),
        &raw_values[2].to_owned().into_dyn(),
        &raw_values[3].to_owned().into_dyn(),
        epsilon,
    )
    .map_err(|error| CganCzProbeDecline::Topology {
        message: format!(
            "{} normalized BatchNorm reconstruction failed: {error}",
            raw.name
        ),
    })?;
    for (field, expected, actual) in [
        ("scale", reconstructed.scale.view(), graph.scale.view()),
        ("bias", reconstructed.bias.view(), graph.bias.view()),
        (
            "scale error",
            reconstructed.scale_err.view(),
            graph.scale_err.view(),
        ),
        (
            "bias error",
            reconstructed.bias_err.view(),
            graph.bias_err.view(),
        ),
    ] {
        if expected.shape() != actual.shape() {
            return Err(CganCzProbeDecline::Topology {
                message: format!("{} normalized BatchNorm {field} shape drift", raw.name),
            });
        }
        compare_f32_bits(
            raw.name.as_str(),
            field,
            expected.iter().copied(),
            actual.iter().copied(),
            coordinator,
        )?;
    }

    let gamma = promote_slice(
        raw_values[0].iter().copied(),
        channels,
        "cGAN BatchNorm gamma",
        coordinator,
    )?;
    let beta = promote_slice(
        raw_values[1].iter().copied(),
        channels,
        "cGAN BatchNorm beta",
        coordinator,
    )?;
    let mean = promote_slice(
        raw_values[2].iter().copied(),
        channels,
        "cGAN BatchNorm mean",
        coordinator,
    )?;
    let variance = promote_slice(
        raw_values[3].iter().copied(),
        channels,
        "cGAN BatchNorm variance",
        coordinator,
    )?;
    let normalized_scale = promote_dynamic(
        graph.scale.view(),
        "cGAN normalized BatchNorm scale",
        coordinator,
    )?;
    let normalized_bias = promote_dynamic(
        graph.bias.view(),
        "cGAN normalized BatchNorm bias",
        coordinator,
    )?;
    Ok(BatchNormParameters {
        gamma,
        beta,
        mean,
        variance,
        epsilon: f64::from(epsilon),
        normalized_scale,
        normalized_bias,
    })
}

fn seal_conv2d<N>(
    model: &OnnxModel,
    raw: &LayerSpec,
    graph_layer: &Layer,
    coordinator: &mut Coordinator<N>,
) -> Result<Conv2dParameters, CganCzProbeDecline>
where
    N: FnMut(&'static str) -> Instant,
{
    let Layer::Conv2d(graph) = graph_layer else {
        unreachable!("topology seal fixed the layer type")
    };
    let (raw_weights, raw_bias) = raw_conv_arrays(model, raw)?;
    let graph_weights = graph
        .kernel
        .view()
        .into_dimensionality::<Ix4>()
        .map_err(|_| shape_error(raw, "normalized Conv kernel", graph.kernel.shape(), &[4]))?;
    if graph_weights.shape() != raw_weights.shape()
        || graph.bias.as_ref().map(ndarray::ArrayBase::len) != Some(raw_bias.len())
    {
        return Err(CganCzProbeDecline::Topology {
            message: format!(
                "{} normalized Conv shapes differ from raw initializers",
                raw.name
            ),
        });
    }
    let stride = parse_positive_pair(raw, "strides", [1, 1])?;
    let dilation = parse_positive_pair(raw, "dilations", [1, 1])?;
    let padding = parse_padding(raw)?;
    let groups = parse_group(raw)?;
    validate_conv_attributes(raw, raw_weights.shape(), false)?;
    if graph.stride != (stride[0], stride[1])
        || graph.dilation != (dilation[0], dilation[1])
        || graph.padding != (padding[0], padding[1])
        || graph.groups != groups
    {
        return Err(CganCzProbeDecline::Topology {
            message: format!(
                "{} normalized Conv attributes differ from raw ONNX",
                raw.name
            ),
        });
    }
    compare_f32_bits(
        raw.name.as_str(),
        "Conv weight",
        raw_weights.iter().copied(),
        graph_weights.iter().copied(),
        coordinator,
    )?;
    compare_f32_bits(
        raw.name.as_str(),
        "Conv bias",
        raw_bias.iter().copied(),
        graph.bias.as_ref().unwrap().iter().copied(),
        coordinator,
    )?;
    Ok(Conv2dParameters {
        weights: promote_array4(raw_weights, "cGAN Conv weights", coordinator)?,
        bias: promote_slice(
            raw_bias.iter().copied(),
            raw_bias.len(),
            "cGAN Conv bias",
            coordinator,
        )?,
        spec: ConstrainedZonotopeConv2dSpec {
            stride,
            padding,
            dilation,
            groups,
        },
    })
}

fn seal_conv_transpose2d<N>(
    model: &OnnxModel,
    raw: &LayerSpec,
    graph_layer: &Layer,
    coordinator: &mut Coordinator<N>,
) -> Result<ConvTranspose2dParameters, CganCzProbeDecline>
where
    N: FnMut(&'static str) -> Instant,
{
    let Layer::ConvTranspose2d(graph) = graph_layer else {
        unreachable!("topology seal fixed the layer type")
    };
    let (raw_weights, raw_bias) = raw_conv_arrays(model, raw)?;
    let graph_weights = graph
        .kernel
        .view()
        .into_dimensionality::<Ix4>()
        .map_err(|_| {
            shape_error(
                raw,
                "normalized ConvTranspose kernel",
                graph.kernel.shape(),
                &[4],
            )
        })?;
    if graph_weights.shape() != raw_weights.shape()
        || graph.bias.as_ref().map(ndarray::ArrayBase::len) != Some(raw_bias.len())
    {
        return Err(CganCzProbeDecline::Topology {
            message: format!(
                "{} normalized ConvTranspose shapes differ from raw initializers",
                raw.name
            ),
        });
    }
    let stride = parse_positive_pair(raw, "strides", [1, 1])?;
    let dilation = parse_positive_pair(raw, "dilations", [1, 1])?;
    let padding = parse_padding(raw)?;
    let output_padding = parse_nonnegative_pair(raw, "output_padding", [0, 0])?;
    let groups = parse_group(raw)?;
    validate_conv_attributes(raw, raw_weights.shape(), true)?;
    if groups != 1
        || graph.stride != (stride[0], stride[1])
        || graph.dilation != (dilation[0], dilation[1])
        || graph.padding != (padding[0], padding[1])
        || graph.output_padding != (output_padding[0], output_padding[1])
    {
        return Err(CganCzProbeDecline::Topology {
            message: format!(
                "{} normalized ConvTranspose attributes differ from raw ONNX",
                raw.name
            ),
        });
    }
    compare_f32_bits(
        raw.name.as_str(),
        "ConvTranspose weight",
        raw_weights.iter().copied(),
        graph_weights.iter().copied(),
        coordinator,
    )?;
    compare_f32_bits(
        raw.name.as_str(),
        "ConvTranspose bias",
        raw_bias.iter().copied(),
        graph.bias.as_ref().unwrap().iter().copied(),
        coordinator,
    )?;
    Ok(ConvTranspose2dParameters {
        weights: promote_array4(raw_weights, "cGAN ConvTranspose weights", coordinator)?,
        bias: promote_slice(
            raw_bias.iter().copied(),
            raw_bias.len(),
            "cGAN ConvTranspose bias",
            coordinator,
        )?,
        spec: ConstrainedZonotopeConvTranspose2dSpec {
            stride,
            padding,
            dilation,
            output_padding,
            groups,
        },
    })
}

fn raw_conv_arrays<'a>(
    model: &'a OnnxModel,
    raw: &LayerSpec,
) -> Result<(ndarray::ArrayView4<'a, f32>, ndarray::ArrayView1<'a, f32>), CganCzProbeDecline> {
    if raw.inputs.len() != 3 || raw.weights.is_some() {
        return Err(CganCzProbeDecline::Topology {
            message: format!("{} must have raw FLOAT kernel and bias inputs", raw.name),
        });
    }
    let weights = model
        .weights
        .get(&raw.inputs[1])
        .ok_or_else(|| missing_parameter(raw, &raw.inputs[1]))?;
    let bias = model
        .weights
        .get(&raw.inputs[2])
        .ok_or_else(|| missing_parameter(raw, &raw.inputs[2]))?;
    let weights = weights
        .view()
        .into_dimensionality::<Ix4>()
        .map_err(|_| shape_error(raw, "raw convolution kernel", weights.shape(), &[4]))?;
    let bias = bias
        .view()
        .into_dimensionality::<Ix1>()
        .map_err(|_| shape_error(raw, "raw convolution bias", bias.shape(), &[1]))?;
    let expected_bias = if raw.layer_type == LayerType::Conv2d {
        weights.shape()[0]
    } else {
        weights.shape()[1].checked_mul(parse_group(raw)?).ok_or(
            CganCzProbeDecline::ResourceOverflow {
                operation: "ConvTranspose output channels",
            },
        )?
    };
    if bias.len() != expected_bias {
        return Err(CganCzProbeDecline::Topology {
            message: format!(
                "{} bias has {} entries, expected {expected_bias}",
                raw.name,
                bias.len()
            ),
        });
    }
    Ok((weights, bias))
}

fn validate_conv_attributes(
    raw: &LayerSpec,
    weight_shape: &[usize],
    transpose: bool,
) -> Result<(), CganCzProbeDecline> {
    const ALLOWED: [&str; 7] = [
        "auto_pad",
        "dilations",
        "group",
        "kernel_shape",
        "output_padding",
        "pads",
        "strides",
    ];
    if raw
        .attributes
        .keys()
        .any(|name| !ALLOWED.contains(&name.as_str()))
    {
        return Err(CganCzProbeDecline::Topology {
            message: format!("{} has unsupported convolution attributes", raw.name),
        });
    }
    if let Some(value) = raw.attributes.get("auto_pad") {
        if value != &AttributeValue::String("NOTSET".to_string()) {
            return Err(CganCzProbeDecline::Topology {
                message: format!("{} has unsupported auto_pad", raw.name),
            });
        }
    }
    if let Some(AttributeValue::Ints(kernel)) = raw.attributes.get("kernel_shape") {
        if kernel
            != &vec![
                i64::try_from(weight_shape[2]).unwrap_or(i64::MAX),
                i64::try_from(weight_shape[3]).unwrap_or(i64::MAX),
            ]
        {
            return Err(CganCzProbeDecline::Topology {
                message: format!("{} kernel_shape disagrees with initializer", raw.name),
            });
        }
    } else if raw.attributes.contains_key("kernel_shape") {
        return Err(CganCzProbeDecline::Topology {
            message: format!("{} has malformed kernel_shape", raw.name),
        });
    }
    if !transpose && raw.attributes.contains_key("output_padding") {
        return Err(CganCzProbeDecline::Topology {
            message: format!("{} Conv cannot carry output_padding", raw.name),
        });
    }
    Ok(())
}

fn parse_positive_pair(
    raw: &LayerSpec,
    name: &'static str,
    fallback: [usize; 2],
) -> Result<[usize; 2], CganCzProbeDecline> {
    let values = match raw.attributes.get(name) {
        None => return Ok(fallback),
        Some(AttributeValue::Ints(values)) if values.len() == 2 => values,
        _ => {
            return Err(CganCzProbeDecline::Topology {
                message: format!("{} has malformed {name}", raw.name),
            });
        }
    };
    let mut parsed = [0; 2];
    for (slot, value) in parsed.iter_mut().zip(values) {
        *slot = usize::try_from(*value).map_err(|_| CganCzProbeDecline::Topology {
            message: format!("{} has negative {name}", raw.name),
        })?;
        if *slot == 0 {
            return Err(CganCzProbeDecline::Topology {
                message: format!("{} has zero {name}", raw.name),
            });
        }
    }
    Ok(parsed)
}

fn parse_nonnegative_pair(
    raw: &LayerSpec,
    name: &'static str,
    fallback: [usize; 2],
) -> Result<[usize; 2], CganCzProbeDecline> {
    let values = match raw.attributes.get(name) {
        None => return Ok(fallback),
        Some(AttributeValue::Ints(values)) if values.len() == 2 => values,
        _ => {
            return Err(CganCzProbeDecline::Topology {
                message: format!("{} has malformed {name}", raw.name),
            });
        }
    };
    Ok([
        usize::try_from(values[0]).map_err(|_| CganCzProbeDecline::Topology {
            message: format!("{} has negative {name}", raw.name),
        })?,
        usize::try_from(values[1]).map_err(|_| CganCzProbeDecline::Topology {
            message: format!("{} has negative {name}", raw.name),
        })?,
    ])
}

fn parse_padding(raw: &LayerSpec) -> Result<[usize; 4], CganCzProbeDecline> {
    let values = match raw.attributes.get("pads") {
        None => return Ok([0; 4]),
        Some(AttributeValue::Ints(values)) if values.len() == 4 => values,
        _ => {
            return Err(CganCzProbeDecline::Topology {
                message: format!("{} has malformed pads", raw.name),
            });
        }
    };
    let mut parsed = [0; 4];
    for (slot, value) in parsed.iter_mut().zip(values) {
        *slot = usize::try_from(*value).map_err(|_| CganCzProbeDecline::Topology {
            message: format!("{} has negative pads", raw.name),
        })?;
    }
    if parsed[0] != parsed[2] || parsed[1] != parsed[3] {
        return Err(CganCzProbeDecline::Topology {
            message: format!("{} uses asymmetric padding", raw.name),
        });
    }
    Ok(parsed)
}

fn parse_group(raw: &LayerSpec) -> Result<usize, CganCzProbeDecline> {
    match raw.attributes.get("group") {
        None => Ok(1),
        Some(AttributeValue::Int(value)) if *value > 0 => {
            usize::try_from(*value).map_err(|_| CganCzProbeDecline::Topology {
                message: format!("{} group does not fit usize", raw.name),
            })
        }
        _ => Err(CganCzProbeDecline::Topology {
            message: format!("{} has malformed group", raw.name),
        }),
    }
}

fn compare_f32_bits<N>(
    node: &str,
    field: &str,
    expected: impl IntoIterator<Item = f32>,
    actual: impl IntoIterator<Item = f32>,
    coordinator: &mut Coordinator<N>,
) -> Result<(), CganCzProbeDecline>
where
    N: FnMut(&'static str) -> Instant,
{
    let mut expected = expected.into_iter();
    let mut actual = actual.into_iter();
    let mut index = 0_usize;
    loop {
        match (expected.next(), actual.next()) {
            (None, None) => return Ok(()),
            (Some(left), Some(right)) => {
                coordinator.charge(1, "cGAN raw/graph bit comparison")?;
                if left.to_bits() != right.to_bits() {
                    return Err(CganCzProbeDecline::Provenance {
                        message: format!("{node} {field}[{index}] differs between raw and graph"),
                    });
                }
                index += 1;
            }
            _ => {
                return Err(CganCzProbeDecline::Topology {
                    message: format!("{node} {field} lengths disagree"),
                });
            }
        }
    }
}

fn promote_array2<N>(
    values: ndarray::ArrayView2<'_, f32>,
    resource: &'static str,
    coordinator: &mut Coordinator<N>,
) -> Result<Array2<f64>, CganCzProbeDecline>
where
    N: FnMut(&'static str) -> Instant,
{
    let shape = values.dim();
    let promoted = promote_slice(values.iter().copied(), values.len(), resource, coordinator)?;
    Array2::from_shape_vec(shape, promoted).map_err(|_| CganCzProbeDecline::ResourceOverflow {
        operation: "cGAN promoted matrix shape",
    })
}

fn promote_array4<N>(
    values: ndarray::ArrayView4<'_, f32>,
    resource: &'static str,
    coordinator: &mut Coordinator<N>,
) -> Result<Array4<f64>, CganCzProbeDecline>
where
    N: FnMut(&'static str) -> Instant,
{
    let shape = values.dim();
    let promoted = promote_slice(values.iter().copied(), values.len(), resource, coordinator)?;
    Array4::from_shape_vec(shape, promoted).map_err(|_| CganCzProbeDecline::ResourceOverflow {
        operation: "cGAN promoted tensor shape",
    })
}

fn promote_dynamic<N>(
    values: ArrayViewD<'_, f32>,
    resource: &'static str,
    coordinator: &mut Coordinator<N>,
) -> Result<Vec<f64>, CganCzProbeDecline>
where
    N: FnMut(&'static str) -> Instant,
{
    promote_slice(values.iter().copied(), values.len(), resource, coordinator)
}

fn promote_slice<N>(
    values: impl IntoIterator<Item = f32>,
    len: usize,
    resource: &'static str,
    coordinator: &mut Coordinator<N>,
) -> Result<Vec<f64>, CganCzProbeDecline>
where
    N: FnMut(&'static str) -> Instant,
{
    coordinator.checkpoint("cGAN parameter promotion allocation")?;
    let mut promoted = Vec::new();
    promoted
        .try_reserve_exact(len)
        .map_err(|_| CganCzProbeDecline::ResourceLimit {
            resource,
            required: len,
            limit: len.saturating_sub(1),
        })?;
    for value in values {
        coordinator.charge(1, "cGAN parameter promotion")?;
        if !value.is_finite() {
            return Err(CganCzProbeDecline::Provenance {
                message: format!("{resource} contains a non-finite value"),
            });
        }
        promoted.push(f64::from(value));
    }
    Ok(promoted)
}

fn missing_parameter(raw: &LayerSpec, name: &str) -> CganCzProbeDecline {
    CganCzProbeDecline::Provenance {
        message: format!("{} parameter '{name}' is missing", raw.name),
    }
}

fn shape_error(
    raw: &LayerSpec,
    field: &str,
    got: &[usize],
    expected_rank: &[usize],
) -> CganCzProbeDecline {
    CganCzProbeDecline::Topology {
        message: format!(
            "{} {field} shape {got:?} does not have rank {}",
            raw.name, expected_rank[0]
        ),
    }
}

fn build_input_domain<N>(
    input: &CertifiedInputBox,
    limits: CganCzSequentialLimits,
    sealed: &SealedCgan,
    telemetry_bytes: usize,
    coordinator: &mut Coordinator<N>,
) -> Result<ConstrainedZonotope64, CganCzProbeDecline>
where
    N: FnMut(&'static str) -> Instant,
{
    if input.len() != PROTECTED_LATENT_SYMBOLS || input.declared_point().iter().any(|&point| point)
    {
        return Err(CganCzProbeDecline::Topology {
            message: "cGAN probe requires five independently varying latent coordinates"
                .to_string(),
        });
    }
    check_resource("input value dimension", input.len(), limits.max_value_dim)?;
    check_resource(
        "input alpha dimension",
        input.len(),
        limits.max_transient_alpha_dim,
    )?;
    let planned_bytes = input
        .len()
        .checked_mul(2 * size_of::<f64>() + size_of::<Vec<(usize, f64)>>())
        .and_then(|bytes| bytes.checked_add(input.len().saturating_mul(size_of::<(usize, f64)>())))
        .ok_or(CganCzProbeDecline::ResourceOverflow {
            operation: "cGAN input-domain bytes",
        })?;
    let required = coordinator
        .budget
        .baseline_live_bytes()
        .checked_add(telemetry_bytes)
        .and_then(|bytes| bytes.checked_add(sealed.live_bytes))
        .and_then(|bytes| bytes.checked_add(planned_bytes))
        .ok_or(CganCzProbeDecline::ResourceOverflow {
            operation: "cGAN input-domain peak",
        })?;
    coordinator.preflight_absolute_peak(required)?;
    coordinator.checkpoint("cGAN input-domain construction")?;
    coordinator.charge(input.len(), "cGAN input-domain bounds")?;
    let domain = ConstrainedZonotope64::from_certified_bounds(
        input.lower(),
        input.upper(),
        input.declared_point(),
    )
    .map_err(|error| CganCzProbeDecline::Transform {
        node: "input",
        operation: "domain construction",
        message: error.to_string(),
    })?;
    if domain.alpha_dim() != PROTECTED_LATENT_SYMBOLS {
        return Err(CganCzProbeDecline::Topology {
            message: format!(
                "input decomposition produced {} alpha symbols, expected five",
                domain.alpha_dim()
            ),
        });
    }
    check_resource(
        "input generator nonzeros",
        generator_nonzeros(&domain)?,
        limits.max_generator_nonzeros,
    )?;
    coordinator.checkpoint("cGAN input-domain publication")?;
    Ok(domain)
}

fn build_independent_input_domain<N>(
    input: &CertifiedInputBox,
    limits: CganCzSequentialLimits,
    sealed: &SealedCgan,
    runner_storage_bytes: usize,
    coordinator: &mut Coordinator<N>,
) -> Result<ConstrainedZonotope64, CganCzProbeDecline>
where
    N: FnMut(&'static str) -> Instant,
{
    build_independent_input_domain_from_bounds(
        input.lower(),
        input.upper(),
        limits,
        sealed,
        runner_storage_bytes,
        coordinator,
    )
}

fn build_independent_input_domain_from_bounds<N>(
    lower: &[f64],
    upper: &[f64],
    limits: CganCzSequentialLimits,
    sealed: &SealedCgan,
    runner_storage_bytes: usize,
    coordinator: &mut Coordinator<N>,
) -> Result<ConstrainedZonotope64, CganCzProbeDecline>
where
    N: FnMut(&'static str) -> Instant,
{
    if lower.len() != PROTECTED_LATENT_SYMBOLS || upper.len() != PROTECTED_LATENT_SYMBOLS {
        return Err(CganCzProbeDecline::Topology {
            message: "cGAN independent interval lane requires exactly five latent coordinates"
                .to_string(),
        });
    }
    check_resource("input value dimension", lower.len(), limits.max_value_dim)?;
    let stored_bytes = lower.len().checked_mul(2 * size_of::<f64>()).ok_or(
        CganCzProbeDecline::ResourceOverflow {
            operation: "cGAN independent input-domain bytes",
        },
    )?;
    let required = coordinator
        .budget
        .baseline_live_bytes()
        .checked_add(runner_storage_bytes)
        .and_then(|bytes| bytes.checked_add(sealed.live_bytes))
        .and_then(|bytes| bytes.checked_add(stored_bytes))
        .ok_or(CganCzProbeDecline::ResourceOverflow {
            operation: "cGAN independent input-domain peak",
        })?;
    coordinator.preflight_absolute_peak(required)?;
    coordinator.checkpoint("cGAN independent input-domain construction")?;
    coordinator.charge(
        lower
            .len()
            .checked_mul(8)
            .ok_or(CganCzProbeDecline::ResourceOverflow {
                operation: "cGAN independent input-domain work",
            })?,
        "cGAN independent input-domain bounds",
    )?;
    // Marking every coordinate here selects the decomposition, not a semantic
    // point claim: from_certified_bounds retains the complete supplied width
    // in the independent box remainder for marked non-point intervals. Exact
    // source points are admitted as well; their outward endpoint moat remains
    // in the same remainder and is never collapsed by the declaration bit.
    let remainder_only_mask = [true; PROTECTED_LATENT_SYMBOLS];
    let domain = ConstrainedZonotope64::from_certified_bounds(lower, upper, &remainder_only_mask)
        .map_err(|error| CganCzProbeDecline::Transform {
        node: "input",
        operation: "independent remainder-only domain construction",
        message: error.to_string(),
    })?;
    if domain.value_dim() != PROTECTED_LATENT_SYMBOLS
        || domain.alpha_dim() != 0
        || domain.constraint_count() != 0
    {
        return Err(CganCzProbeDecline::Transform {
            node: "input",
            operation: "independent remainder-only structural audit",
            message: format!(
                "constructed value_dim={}, alpha_dim={}, constraint_count={}",
                domain.value_dim(),
                domain.alpha_dim(),
                domain.constraint_count()
            ),
        });
    }
    coordinator.checkpoint("cGAN independent input-domain publication")?;
    Ok(domain)
}

fn build_correlated_leaf_input_domain_from_bounds<N>(
    lower: &[f64],
    upper: &[f64],
    limits: CganCzSequentialLimits,
    retained_baseline: usize,
    coordinator: &mut Coordinator<N>,
) -> Result<ConstrainedZonotope64, CganCzProbeDecline>
where
    N: FnMut(&'static str) -> Instant,
{
    if lower.len() != PROTECTED_LATENT_SYMBOLS || upper.len() != PROTECTED_LATENT_SYMBOLS {
        return Err(CganCzProbeDecline::Topology {
            message: "cGAN correlated leaf requires exactly five latent coordinates".to_string(),
        });
    }
    let mut variable_count = 0_usize;
    for index in 0..PROTECTED_LATENT_SYMBOLS {
        coordinator.charge(1, "cGAN correlated leaf input validation")?;
        let lo = lower[index];
        let hi = upper[index];
        if !lo.is_finite() || !hi.is_finite() || lo > hi {
            return Err(CganCzProbeDecline::Topology {
                message: format!("correlated leaf input endpoint {index} is invalid"),
            });
        }
        variable_count += usize::from(lo != hi);
    }
    check_resource(
        "correlated leaf input alpha dimension",
        PROTECTED_LATENT_SYMBOLS,
        limits.max_transient_alpha_dim,
    )?;
    // Keep one stable alpha slot per authored latent coordinate even when a
    // BaB leaf fixes that coordinate. `from_certified_bounds` represents a
    // fixed slot by an empty generator column, so this adds no uncertainty but
    // preserves the five-symbol prefix required by every later order
    // reduction. It also treats IEEE signed-zero endpoints as the same exact
    // real point without changing their alpha identity.
    let correlated_mask = [false; PROTECTED_LATENT_SYMBOLS];
    let planned_bytes = PROTECTED_LATENT_SYMBOLS
        .checked_mul(2 * size_of::<f64>())
        .and_then(|bytes| {
            bytes.checked_add(
                PROTECTED_LATENT_SYMBOLS.saturating_mul(size_of::<Vec<(usize, f64)>>()),
            )
        })
        .and_then(|bytes| {
            bytes.checked_add(variable_count.saturating_mul(size_of::<(usize, f64)>()))
        })
        .ok_or(CganCzProbeDecline::ResourceOverflow {
            operation: "cGAN correlated leaf input-domain bytes",
        })?;
    coordinator.preflight_absolute_peak(retained_baseline.checked_add(planned_bytes).ok_or(
        CganCzProbeDecline::ResourceOverflow {
            operation: "cGAN correlated leaf input-domain peak",
        },
    )?)?;
    coordinator.checkpoint("cGAN correlated leaf input-domain construction")?;
    let domain = ConstrainedZonotope64::from_certified_bounds(lower, upper, &correlated_mask)
        .map_err(|error| CganCzProbeDecline::Transform {
            node: "input",
            operation: "correlated leaf domain construction",
            message: error.to_string(),
        })?;
    if domain.value_dim() != PROTECTED_LATENT_SYMBOLS
        || domain.alpha_dim() != PROTECTED_LATENT_SYMBOLS
        || domain.constraint_count() != 0
        || generator_nonzeros(&domain)? != variable_count
    {
        return Err(CganCzProbeDecline::Transform {
            node: "input",
            operation: "correlated leaf structural audit",
            message: format!(
                "constructed value_dim={}, alpha_dim={}, constraint_count={}, expected {PROTECTED_LATENT_SYMBOLS}, {PROTECTED_LATENT_SYMBOLS}, 0",
                domain.value_dim(),
                domain.alpha_dim(),
                domain.constraint_count()
            ),
        });
    }
    coordinator.checkpoint("cGAN correlated leaf input-domain publication")?;
    Ok(domain)
}

fn summarize_prefix<N>(
    domain: &ConstrainedZonotope64,
    shape: &[usize],
    last_index: usize,
    moat: CertifiedScalarMoat,
    retained_baseline: usize,
    coordinator: &mut Coordinator<N>,
) -> Result<CganCzPrefixCompletion, CganCzProbeDecline>
where
    N: FnMut(&'static str) -> Instant,
{
    let last_node = EXPECTED_NODES
        .get(last_index)
        .map(|(name, _)| *name)
        .ok_or_else(|| CganCzProbeDecline::InvalidLimit {
            message: format!("prefix endpoint index {last_index} is outside the sealed topology"),
        })?;
    if !matches!(last_index, 5 | 8 | 11 | 14) {
        return Err(CganCzProbeDecline::InvalidLimit {
            message: format!("{last_node} is not a qualified diagnostic endpoint"),
        });
    }
    let expected_shape = expected_output_shape(last_index);
    if shape != expected_shape
        || domain.value_dim() != checked_product(expected_shape, "generator-block prefix")?
    {
        return Err(CganCzProbeDecline::Topology {
            message: format!(
                "{last_node} prefix ended with shape {shape:?} and {} values, expected {expected_shape:?}",
                domain.value_dim()
            ),
        });
    }
    if domain.alpha_dim() < PROTECTED_LATENT_SYMBOLS {
        return Err(CganCzProbeDecline::Transform {
            node: last_node,
            operation: "protected-symbol audit",
            message: format!(
                "only {} generators survived, fewer than five protected latent symbols",
                domain.alpha_dim()
            ),
        });
    }

    let scratch_bytes = domain.value_dim().checked_mul(size_of::<f64>()).ok_or(
        CganCzProbeDecline::ResourceOverflow {
            operation: "prefix width scratch",
        },
    )?;
    coordinator.preflight_absolute_peak(
        retained_baseline
            .checked_add(domain_live_bytes(domain)?)
            .and_then(|bytes| bytes.checked_add(scratch_bytes))
            .ok_or(CganCzProbeDecline::ResourceOverflow {
                operation: "prefix width telemetry peak",
            })?,
    )?;
    coordinator.checkpoint("prefix width telemetry allocation")?;
    let mut radii = Vec::new();
    radii
        .try_reserve_exact(domain.value_dim())
        .map_err(|_| CganCzProbeDecline::ResourceLimit {
            resource: "prefix coordinate radii",
            required: domain.value_dim(),
            limit: domain.value_dim().saturating_sub(1),
        })?;
    radii.extend_from_slice(domain.box_remainder());
    coordinator.charge(domain.value_dim(), "prefix remainder telemetry")?;
    for generator in domain.generators() {
        coordinator.charge(1, "prefix generator telemetry")?;
        for (coordinate, coefficient) in generator.entries() {
            coordinator.charge(1, "prefix generator telemetry")?;
            let sum = radii[coordinate] + coefficient.abs();
            if !sum.is_finite() {
                return Err(CganCzProbeDecline::Transform {
                    node: last_node,
                    operation: "width telemetry",
                    message: format!("coordinate {coordinate} radius is non-finite"),
                });
            }
            radii[coordinate] = if sum == 0.0 { 0.0 } else { sum.next_up() };
        }
    }

    let mut maximum_coordinate_width = 0.0_f64;
    let mut total_width = 0.0_f64;
    for radius in radii {
        coordinator.charge(1, "prefix width telemetry")?;
        let width = radius + radius;
        if !width.is_finite() {
            return Err(CganCzProbeDecline::Transform {
                node: last_node,
                operation: "width telemetry",
                message: "coordinate width is non-finite".to_string(),
            });
        }
        let width = if width == 0.0 { 0.0 } else { width.next_up() };
        maximum_coordinate_width = maximum_coordinate_width.max(width);
        let sum = total_width + width;
        if !sum.is_finite() {
            return Err(CganCzProbeDecline::Transform {
                node: last_node,
                operation: "width telemetry",
                message: "mean-width accumulator is non-finite".to_string(),
            });
        }
        total_width = if sum == 0.0 { 0.0 } else { sum.next_up() };
    }
    let mean_coordinate_width = if domain.value_dim() == 0 {
        0.0
    } else {
        let mean = total_width / domain.value_dim() as f64;
        if mean == 0.0 {
            0.0
        } else {
            mean.next_up()
        }
    };
    let maximum_box_remainder = domain
        .box_remainder()
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    coordinator.checkpoint("prefix width telemetry publication")?;

    Ok(CganCzPrefixCompletion {
        last_node,
        output_shape: shape.to_vec(),
        value_dim: domain.value_dim(),
        alpha_dim: domain.alpha_dim(),
        generator_nonzeros: generator_nonzeros(domain)?,
        maximum_coordinate_width,
        mean_coordinate_width,
        maximum_box_remainder,
        low_unsafe_threshold: moat.low_upper(),
        high_unsafe_threshold: moat.high_lower(),
    })
}

#[allow(clippy::too_many_arguments)]
fn bound_output_tail<N>(
    domain: &ConstrainedZonotope64,
    auxiliary: Option<&CertifiedAuxiliaryBounds64>,
    depth_two: Option<CganCzDepthTwoContext<'_>>,
    tail_bn: &BatchNormParameters,
    tail_affine: &AffineParameters,
    moat: CertifiedScalarMoat,
    limits: CganCzSequentialLimits,
    retained_baseline: usize,
    budget: ConstrainedZonotopeCallBudget,
    coordinator: &mut Coordinator<N>,
    stages: &mut Vec<CganCzStageTelemetry>,
) -> Result<CganCzOutputTailResult, CganCzProbeDecline>
where
    N: FnMut(&'static str) -> Instant,
{
    let construction_charged_start = coordinator.charged_items;
    let construction_polls_start = coordinator.deadline_polls;
    validate_output_tail_contract(domain, tail_bn, tail_affine, coordinator)?;
    let downstream_auxiliary_live_bytes = auxiliary
        .map(|bounds| independent_endpoint_bytes(bounds.value_dim()))
        .transpose()?
        .unwrap_or(0);
    let upstream_auxiliary_live_bytes = depth_two
        .map(|context| independent_endpoint_bytes(context.upstream_auxiliary.bounds.value_dim()))
        .transpose()?
        .unwrap_or(0);
    let tail_support_baseline = retained_baseline
        .checked_add(downstream_auxiliary_live_bytes)
        .and_then(|bytes| bytes.checked_add(upstream_auxiliary_live_bytes))
        .ok_or(CganCzProbeDecline::ResourceOverflow {
            operation: "cGAN output-tail retained auxiliary baseline",
        })?;
    let depth_two_upstream_live_bytes = depth_two
        .map(|context| domain_live_bytes(context.upstream))
        .transpose()?
        .unwrap_or(0);
    let tail_retained_baseline = tail_support_baseline
        .checked_add(depth_two_upstream_live_bytes)
        .ok_or(CganCzProbeDecline::ResourceOverflow {
            operation: "cGAN output-tail retained pre-Relu_20 baseline",
        })?;
    coordinator.preflight_absolute_peak(tail_retained_baseline)?;
    let tail_shape = [TAIL_CHANNELS, 2, 2];
    let certificate_budget =
        domain_call_budget(domain, tail_retained_baseline, budget, coordinator)?;
    let certificate_outcome = certify_batch_norm_affine_surrogate_unwired_with_budget(
        ConstrainedZonotopeBatchNormSpec {
            input_shape: &tail_shape,
            channel_axis: 0,
            gamma: &tail_bn.gamma,
            beta: &tail_bn.beta,
            mean: &tail_bn.mean,
            variance: &tail_bn.variance,
            epsilon: tail_bn.epsilon,
            mode: ConstrainedZonotopeBatchNormMode::Inference,
        },
        &tail_bn.normalized_scale,
        &tail_bn.normalized_bias,
        ConstrainedZonotopeBatchNormAffineCertificateLimits {
            max_rank: tail_shape.len(),
            max_channel_count: TAIL_CHANNELS,
            max_parameter_elements: TAIL_CHANNELS * 6,
        },
        certificate_budget,
    )
    .map_err(|error| match error {
        ConstrainedZonotopeBatchNormBudgetError::Budget(error) => CganCzProbeDecline::Budget(error),
        ConstrainedZonotopeBatchNormBudgetError::Transform(error) => {
            CganCzProbeDecline::OutputTail {
                message: format!(
                    "BatchNormalization_24 declared-surrogate certification declined: {error}"
                ),
            }
        }
    })?;
    let (certificate, certificate_report) = certificate_outcome.into_parts();
    coordinator.absorb(certificate_report)?;
    let (correction, correction_peak_live_bytes) = exact_batch_norm_tail_correction(
        domain,
        &certificate,
        tail_affine,
        tail_retained_baseline,
        coordinator,
    )?;
    drop(certificate);

    let (prepared, preparation_peak_live_bytes) = if auxiliary.is_some() || depth_two.is_some() {
        // The exact BN correction remains live throughout preparation and all
        // exact row portfolios. Preparation itself creates the retained
        // geometry, so its output bytes are added only to subsequent calls.
        let preparation_retained_baseline = tail_retained_baseline
            .checked_add(TAIL_RATIONAL_LIVE_BYTES)
            .ok_or(CganCzProbeDecline::ResourceOverflow {
                operation: "cGAN prepared tail retained correction baseline",
            })?;
        let preparation_budget =
            domain_call_budget(domain, preparation_retained_baseline, budget, coordinator)?;
        let outcome =
            prepare_relu_tail_triangle_dual_unwired_with_budget(domain, preparation_budget)
                .map_err(map_tail_dual_budget_error)?;
        let (prepared, report) = outcome.into_parts();
        let peak = report.peak_live_bytes();
        coordinator.absorb(report)?;
        (Some(prepared), peak)
    } else {
        (None, 0)
    };
    let member_retained_baseline =
        prepared
            .as_ref()
            .map_or(Ok(tail_retained_baseline), |value| {
                tail_retained_baseline
                    .checked_add(value.conservative_live_bytes())
                    .ok_or(CganCzProbeDecline::ResourceOverflow {
                        operation: "cGAN prepared tail member baseline",
                    })
            })?;
    coordinator.checkpoint("cGAN output-tail construction publication")?;
    let construction_charged_items = coordinator
        .charged_items
        .checked_sub(construction_charged_start)
        .ok_or(CganCzProbeDecline::ResourceOverflow {
            operation: "cGAN output-tail construction charged-item delta",
        })?;
    let construction_deadline_polls = coordinator
        .deadline_polls
        .checked_sub(construction_polls_start)
        .ok_or(CganCzProbeDecline::ResourceOverflow {
            operation: "cGAN output-tail construction deadline-poll delta",
        })?;
    record_scalar_tail_stage_accounting(
        stages,
        "Gemm_27",
        CganCzStageKind::OutputTailConstruction,
        domain.alpha_dim(),
        generator_nonzeros(domain)?,
        0,
        certificate_report
            .peak_live_bytes()
            .max(correction_peak_live_bytes)
            .max(preparation_peak_live_bytes),
        construction_charged_items,
        construction_deadline_polls,
    )?;

    let lower_portfolio = {
        let stage_charged_start = coordinator.charged_items;
        let stage_polls_start = coordinator.deadline_polls;
        let (margin, objective_peak_live_bytes) = exact_output_tail_margin(
            domain,
            tail_bn,
            tail_affine,
            &correction,
            TailMarginSense::Lower,
            member_retained_baseline,
            coordinator,
        )?;
        run_output_tail_m17(
            domain,
            auxiliary.zip(prepared.as_ref()),
            &margin,
            CganCzStageKind::M17Lower,
            limits,
            member_retained_baseline,
            budget,
            coordinator,
            stages,
            objective_peak_live_bytes,
            stage_charged_start,
            stage_polls_start,
        )?
    };

    // Preserve the historical disjoint margin live ranges exactly.  Only the
    // already-live final margin may survive into optional work; the lower
    // margin is reconstructed later under the guarded optional budget.
    let upper_stage_charged_start = coordinator.charged_items;
    let upper_stage_polls_start = coordinator.deadline_polls;
    let (negated_upper_margin, upper_objective_peak_live_bytes) = exact_output_tail_margin(
        domain,
        tail_bn,
        tail_affine,
        &correction,
        TailMarginSense::NegatedUpper,
        member_retained_baseline,
        coordinator,
    )?;
    let negated_upper_portfolio = run_output_tail_m17(
        domain,
        auxiliary.zip(prepared.as_ref()),
        &negated_upper_margin,
        CganCzStageKind::M17Upper,
        limits,
        member_retained_baseline,
        budget,
        coordinator,
        stages,
        upper_objective_peak_live_bytes,
        upper_stage_charged_start,
        upper_stage_polls_start,
    )?;
    let retained_negated_upper_margin = if depth_two.is_some() {
        Some(negated_upper_margin)
    } else {
        drop(negated_upper_margin);
        None
    };
    let lower_bound = lower_portfolio.selected_lower_bound;
    let negated_upper_lower_bound = negated_upper_portfolio.selected_lower_bound;
    let upper_bound = -negated_upper_lower_bound;
    if !lower_bound.is_finite() || !upper_bound.is_finite() || lower_bound > upper_bound {
        return Err(CganCzProbeDecline::OutputTail {
            message: format!(
                "M17/M20 returned inconsistent finite output bounds [{lower_bound}, {upper_bound}]"
            ),
        });
    }
    let correction_upper = exact_nonnegative_to_upper_f64(&correction)?;
    let (lower_depth_two_measurement, negated_upper_depth_two_measurement) =
        run_depth_two_measurements(
            domain,
            depth_two,
            prepared.as_ref(),
            retained_negated_upper_margin,
            tail_bn,
            tail_affine,
            &correction,
            lower_bound,
            negated_upper_lower_bound,
            limits,
            tail_support_baseline,
            budget,
            coordinator,
        )?;
    coordinator.checkpoint("cGAN output-tail publication")?;

    Ok(CganCzOutputTailResult {
        completed: CganCzCompletedBounds {
            lower_bound,
            upper_bound,
            low_unsafe_threshold: moat.low_upper(),
            high_unsafe_threshold: moat.high_lower(),
            separates_unsafe_moat: tail_bounds_separate_unsafe_moat(
                lower_bound,
                upper_bound,
                moat.low_upper(),
                moat.high_lower(),
            ),
            bn_tail_correction_upper: correction_upper,
            lower_m17_status: lower_portfolio.m17_candidates.status,
            upper_m17_status: negated_upper_portfolio.m17_candidates.status,
            lower_m17_candidates: lower_portfolio.m17_candidates,
            negated_upper_m17_candidates: negated_upper_portfolio.m17_candidates,
            lower_m20_lower_bound: lower_portfolio.m20_lower_bound,
            negated_upper_m20_lower_bound: negated_upper_portfolio.m20_lower_bound,
            lower_m20_status: lower_portfolio.m20_status,
            negated_upper_m20_status: negated_upper_portfolio.m20_status,
            lower_m24_measurement: lower_portfolio.m24_measurement,
            negated_upper_m24_measurement: negated_upper_portfolio.m24_measurement,
        },
        lower_depth_two_measurement,
        negated_upper_depth_two_measurement,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TailMarginSense {
    Lower,
    NegatedUpper,
}

fn duplicate_depth_two_measurement(
    measurement: CganCzDepthTwoMeasurement,
) -> (CganCzDepthTwoMeasurement, CganCzDepthTwoMeasurement) {
    (measurement.clone(), measurement)
}

fn depth_two_setup_fallback() -> CganCzDepthTwoMeasurement {
    CganCzDepthTwoMeasurement::TransformFallback(CganCzDepthTwoTransformFailure::Setup)
}

fn depth_two_decline_measurement(error: CganCzProbeDecline) -> CganCzDepthTwoMeasurement {
    match error {
        CganCzProbeDecline::Budget(error) => CganCzDepthTwoMeasurement::BudgetFallback(error),
        _ => depth_two_setup_fallback(),
    }
}

fn depth_two_preparation_error(error: ReluTailDualBudgetError) -> CganCzDepthTwoMeasurement {
    match error {
        ReluTailDualBudgetError::Budget(error) => CganCzDepthTwoMeasurement::BudgetFallback(error),
        ReluTailDualBudgetError::Bound(_) => {
            CganCzDepthTwoMeasurement::TransformFallback(CganCzDepthTwoTransformFailure::ReluTail)
        }
    }
}

#[allow(clippy::result_large_err)]
fn depth_two_call_budget<N>(
    deadline: Instant,
    baseline_live_bytes: usize,
    outer_budget: ConstrainedZonotopeCallBudget,
    coordinator: &mut Coordinator<N>,
) -> Result<ConstrainedZonotopeCallBudget, CganCzDepthTwoMeasurement>
where
    N: FnMut(&'static str) -> Instant,
{
    coordinator
        .preflight_absolute_peak(baseline_live_bytes)
        .map_err(depth_two_decline_measurement)?;
    Ok(ConstrainedZonotopeCallBudget::new(
        deadline,
        baseline_live_bytes,
        outer_budget.max_peak_live_bytes(),
    ))
}

fn depth_two_retained_measurement_baseline(
    baseline_live_bytes: usize,
    _measurement: &CganCzDepthTwoMeasurement,
) -> Option<usize> {
    baseline_live_bytes.checked_add(size_of::<CganCzDepthTwoMeasurement>())
}

#[allow(clippy::too_many_arguments)]
fn run_depth_two_transaction<N>(
    downstream: &PreparedReluTailGeometry64<'_>,
    upstream: &PreparedReluTailGeometry64<'_>,
    context: CganCzDepthTwoContext<'_>,
    margin: &ExactReluTailMargin,
    historical_lower_bound: f64,
    downstream_config: ReluTailDualConfig,
    upstream_config: ReluTailDualConfig,
    call_budget: ConstrainedZonotopeCallBudget,
    coordinator: &mut Coordinator<N>,
) -> Result<CganCzDepthTwoMeasurement, CganCzProbeDecline>
where
    N: FnMut(&'static str) -> Instant,
{
    let expected_max_peak_live_bytes = call_budget.max_peak_live_bytes();
    let batch_norm_spec = ConstrainedZonotopeBatchNormSpec {
        input_shape: &TAIL_DEPTH_TWO_INPUT_SHAPE,
        channel_axis: 0,
        gamma: &context.batch_norm_21.gamma,
        beta: &context.batch_norm_21.beta,
        mean: &context.batch_norm_21.mean,
        variance: &context.batch_norm_21.variance,
        epsilon: context.batch_norm_21.epsilon,
        mode: ConstrainedZonotopeBatchNormMode::Inference,
    };
    let attempt = downstream.bound_conv2d_batch_norm_pullback_m17_m20_unwired_attempt_with_budget(
        margin,
        None,
        downstream_config,
        upstream,
        &context.upstream_auxiliary.bounds,
        TAIL_DEPTH_TWO_INPUT_SHAPE,
        context.conv_22.weights.view(),
        &context.conv_22.bias,
        context.conv_22.spec,
        batch_norm_spec,
        &context.batch_norm_21.normalized_scale,
        &context.batch_norm_21.normalized_bias,
        ConstrainedZonotopeBatchNormAffineCertificateLimits {
            max_rank: TAIL_DEPTH_TWO_INPUT_SHAPE.len(),
            max_channel_count: TAIL_DEPTH_TWO_INPUT_SHAPE[0],
            max_parameter_elements: TAIL_DEPTH_TWO_INPUT_SHAPE[0] * 6,
        },
        depth_two_pullback_limits(),
        None,
        upstream_config,
        call_budget,
    );
    let (result, report) = attempt.into_parts();
    #[cfg(test)]
    if let Err(error) = &result {
        eprintln!("NY_CGAN_DEPTH_TWO_ATTEMPT_ERROR error={error:?}");
    }
    coordinator.absorb(report)?;
    Ok(match result {
        Ok(result) => completed_depth_two_measurement(
            historical_lower_bound,
            result,
            report,
            expected_max_peak_live_bytes,
        ),
        Err(error) => depth_two_error_measurement(error),
    })
}

#[allow(clippy::too_many_arguments)]
fn run_depth_two_measurements<N>(
    domain: &ConstrainedZonotope64,
    depth_two: Option<CganCzDepthTwoContext<'_>>,
    downstream_prepared: Option<&PreparedReluTailGeometry64<'_>>,
    retained_negated_upper_margin: Option<ExactReluTailMargin>,
    tail_bn: &BatchNormParameters,
    tail_affine: &AffineParameters,
    correction: &BigRational,
    historical_lower_bound: f64,
    historical_negated_upper_lower_bound: f64,
    limits: CganCzSequentialLimits,
    tail_support_baseline: usize,
    outer_budget: ConstrainedZonotopeCallBudget,
    coordinator: &mut Coordinator<N>,
) -> Result<(CganCzDepthTwoMeasurement, CganCzDepthTwoMeasurement), CganCzProbeDecline>
where
    N: FnMut(&'static str) -> Instant,
{
    let Some(context) = depth_two else {
        return Ok(duplicate_depth_two_measurement(
            CganCzDepthTwoMeasurement::NotRequested,
        ));
    };
    let (Some(downstream_prepared), Some(negated_upper_margin)) =
        (downstream_prepared, retained_negated_upper_margin)
    else {
        return Ok(duplicate_depth_two_measurement(depth_two_setup_fallback()));
    };
    if !depth_two_context_is_sealed(domain, context) {
        return Ok(duplicate_depth_two_measurement(depth_two_setup_fallback()));
    }
    let optional_deadline = match depth_two_optional_deadline(coordinator) {
        Ok(Some(deadline)) => deadline,
        Ok(None)
        | Err(CganCzProbeDecline::Budget(ConstrainedZonotopeCallBudgetError::DeadlineExpired {
            ..
        })) => {
            return Ok(duplicate_depth_two_measurement(
                CganCzDepthTwoMeasurement::NoTime,
            ));
        }
        Err(error) => {
            return Ok(duplicate_depth_two_measurement(
                depth_two_decline_measurement(error),
            ));
        }
    };

    let one_margin_live_bytes = match domain
        .value_dim()
        .checked_add(2)
        .ok_or(CganCzProbeDecline::ResourceOverflow {
            operation: "cGAN depth-two exact margin rational slots",
        })
        .and_then(|slots| tail_rational_bytes(slots, "cGAN depth-two exact margin live bytes"))
    {
        Ok(bytes) => bytes,
        Err(error) => {
            return Ok(duplicate_depth_two_measurement(
                depth_two_decline_measurement(error),
            ));
        }
    };
    let downstream_live_bytes = match domain_live_bytes(domain) {
        Ok(bytes) => bytes,
        Err(error) => {
            return Ok(duplicate_depth_two_measurement(
                depth_two_decline_measurement(error),
            ));
        }
    };
    let upstream_live_bytes = match domain_live_bytes(context.upstream) {
        Ok(bytes) => bytes,
        Err(error) => {
            return Ok(duplicate_depth_two_measurement(
                depth_two_decline_measurement(error),
            ));
        }
    };
    let Some(optional_non_domain_baseline) = tail_support_baseline
        .checked_add(downstream_prepared.conservative_live_bytes())
        .and_then(|bytes| bytes.checked_add(TAIL_RATIONAL_LIVE_BYTES))
    else {
        return Ok(duplicate_depth_two_measurement(depth_two_setup_fallback()));
    };
    let Some(upstream_preparation_baseline) = optional_non_domain_baseline
        .checked_add(downstream_live_bytes)
        .and_then(|bytes| bytes.checked_add(one_margin_live_bytes))
        .and_then(|bytes| bytes.checked_add(upstream_live_bytes))
    else {
        return Ok(duplicate_depth_two_measurement(depth_two_setup_fallback()));
    };
    let upstream_preparation_budget = match depth_two_call_budget(
        optional_deadline,
        upstream_preparation_baseline,
        outer_budget,
        coordinator,
    ) {
        Ok(budget) => budget,
        Err(measurement) => return Ok(duplicate_depth_two_measurement(measurement)),
    };
    let preparation_attempt = prepare_relu_tail_triangle_dual_unwired_attempt_with_budget(
        context.upstream,
        upstream_preparation_budget,
    );
    let (upstream_prepared, preparation_report) = preparation_attempt.into_parts();
    coordinator.absorb(preparation_report)?;
    let upstream_prepared = match upstream_prepared {
        Ok(prepared) => prepared,
        Err(error) => {
            return Ok(duplicate_depth_two_measurement(
                depth_two_preparation_error(error),
            ));
        }
    };

    let downstream_config = match depth_two_m17_config(domain, limits) {
        Ok(config) => config,
        Err(error) => {
            return Ok(duplicate_depth_two_measurement(
                depth_two_decline_measurement(error),
            ));
        }
    };
    let upstream_config = match depth_two_m17_config(context.upstream, limits) {
        Ok(config) => config,
        Err(error) => {
            return Ok(duplicate_depth_two_measurement(
                depth_two_decline_measurement(error),
            ));
        }
    };
    let Some(transaction_baseline) =
        upstream_preparation_baseline.checked_add(upstream_prepared.conservative_live_bytes())
    else {
        return Ok(duplicate_depth_two_measurement(depth_two_setup_fallback()));
    };
    let transaction_budget = match depth_two_call_budget(
        optional_deadline,
        transaction_baseline,
        outer_budget,
        coordinator,
    ) {
        Ok(budget) => budget,
        Err(measurement) => return Ok(duplicate_depth_two_measurement(measurement)),
    };
    let negated_upper_measurement = run_depth_two_transaction(
        downstream_prepared,
        &upstream_prepared,
        context,
        &negated_upper_margin,
        historical_negated_upper_lower_bound,
        downstream_config,
        upstream_config,
        transaction_budget,
        coordinator,
    )?;
    drop(negated_upper_margin);

    // Reconstruct the lower margin only after the historical portfolios and
    // upper depth-two transaction.  Temporarily narrowing the Coordinator's
    // deadline makes every injected-clock poll obey the same optional guard.
    let Some(lower_construction_retained_baseline) = optional_non_domain_baseline
        .checked_add(upstream_live_bytes)
        .and_then(|bytes| bytes.checked_add(upstream_prepared.conservative_live_bytes()))
        .and_then(|bytes| {
            depth_two_retained_measurement_baseline(bytes, &negated_upper_measurement)
        })
    else {
        return Ok((depth_two_setup_fallback(), negated_upper_measurement));
    };
    let saved_budget = coordinator.budget;
    coordinator.budget = ConstrainedZonotopeCallBudget::new(
        optional_deadline,
        saved_budget.baseline_live_bytes(),
        saved_budget.max_peak_live_bytes(),
    );
    let lower_margin = exact_output_tail_margin(
        domain,
        tail_bn,
        tail_affine,
        correction,
        TailMarginSense::Lower,
        lower_construction_retained_baseline,
        coordinator,
    );
    coordinator.budget = saved_budget;
    let lower_margin = match lower_margin {
        Ok((margin, _)) => margin,
        Err(error) => {
            return Ok((
                depth_two_decline_measurement(error),
                negated_upper_measurement,
            ));
        }
    };
    let Some(lower_transaction_baseline) =
        depth_two_retained_measurement_baseline(transaction_baseline, &negated_upper_measurement)
    else {
        return Ok((depth_two_setup_fallback(), negated_upper_measurement));
    };
    let lower_transaction_budget = match depth_two_call_budget(
        optional_deadline,
        lower_transaction_baseline,
        outer_budget,
        coordinator,
    ) {
        Ok(budget) => budget,
        Err(measurement) => return Ok((measurement, negated_upper_measurement)),
    };
    let lower_measurement = run_depth_two_transaction(
        downstream_prepared,
        &upstream_prepared,
        context,
        &lower_margin,
        historical_lower_bound,
        downstream_config,
        upstream_config,
        lower_transaction_budget,
        coordinator,
    )?;
    Ok((lower_measurement, negated_upper_measurement))
}

fn depth_two_pullback_limits() -> ReluTailConvBatchNormPullbackLimits {
    ReluTailConvBatchNormPullbackLimits {
        max_input_value_count: TAIL_DEPTH_TWO_INPUT_VALUES,
        max_output_value_count: TAIL_DEPTH_TWO_OUTPUT_VALUES,
        max_weight_elements: TAIL_DEPTH_TWO_WEIGHT_ELEMENTS,
        max_kernel_visits: TAIL_DEPTH_TWO_KERNEL_VISITS,
        max_pulled_margin_construction_exact_products: TAIL_DEPTH_TWO_EXACT_PRODUCTS,
    }
}

fn depth_two_plan_is_exact(plan: ReluTailConvBatchNormPullbackPlan) -> bool {
    plan.input_shape == TAIL_DEPTH_TWO_INPUT_SHAPE
        && plan.output_shape == TAIL_DEPTH_TWO_OUTPUT_SHAPE
        && plan.weight_shape == TAIL_DEPTH_TWO_WEIGHT_SHAPE
        && plan.weight_elements == TAIL_DEPTH_TWO_WEIGHT_ELEMENTS
        && plan.kernel_visits == TAIL_DEPTH_TWO_KERNEL_VISITS
        && plan.pulled_margin_construction_exact_product_bound == TAIL_DEPTH_TWO_EXACT_PRODUCTS
}

fn depth_two_context_is_sealed(
    domain: &ConstrainedZonotope64,
    context: CganCzDepthTwoContext<'_>,
) -> bool {
    std::ptr::eq(domain, context.downstream)
        && authenticate_independent_relu_bound_record(
            context.profile,
            19,
            context.upstream_auxiliary,
        )
        .is_ok()
        && context.upstream.value_dim() == TAIL_DEPTH_TWO_INPUT_VALUES
        && context.downstream.value_dim() == TAIL_DEPTH_TWO_OUTPUT_VALUES
        && context.batch_norm_21.gamma.len() == TAIL_DEPTH_TWO_INPUT_SHAPE[0]
        && context.batch_norm_21.beta.len() == TAIL_DEPTH_TWO_INPUT_SHAPE[0]
        && context.batch_norm_21.mean.len() == TAIL_DEPTH_TWO_INPUT_SHAPE[0]
        && context.batch_norm_21.variance.len() == TAIL_DEPTH_TWO_INPUT_SHAPE[0]
        && context.batch_norm_21.normalized_scale.len() == TAIL_DEPTH_TWO_INPUT_SHAPE[0]
        && context.batch_norm_21.normalized_bias.len() == TAIL_DEPTH_TWO_INPUT_SHAPE[0]
        && context.batch_norm_21.epsilon.is_finite()
        && context.batch_norm_21.epsilon > 0.0
        && context.conv_22.weights.shape() == TAIL_DEPTH_TWO_WEIGHT_SHAPE.as_slice()
        && context.conv_22.bias.len() == TAIL_DEPTH_TWO_OUTPUT_SHAPE[0]
        && context.conv_22.spec.stride == [2, 2]
        && context.conv_22.spec.padding == [1, 1, 1, 1]
        && context.conv_22.spec.dilation == [1, 1]
        && context.conv_22.spec.groups == 1
}

fn depth_two_optional_deadline<N>(
    coordinator: &mut Coordinator<N>,
) -> Result<Option<Instant>, CganCzProbeDecline>
where
    N: FnMut(&'static str) -> Instant,
{
    let now = coordinator.checkpoint_time("cGAN depth-two optional admission")?;
    let Some(latest) = coordinator
        .budget
        .deadline()
        .checked_sub(TAIL_DEPTH_TWO_PUBLICATION_GUARD)
    else {
        return Ok(None);
    };
    if now >= latest {
        return Ok(None);
    }
    Ok(Some(
        now.checked_add(TAIL_DEPTH_TWO_MEASUREMENT_WALL_TIME)
            .map_or(latest, |candidate| candidate.min(latest)),
    ))
}

fn depth_two_m17_config(
    domain: &ConstrainedZonotope64,
    limits: CganCzSequentialLimits,
) -> Result<ReluTailDualConfig, CganCzProbeDecline> {
    Ok(ReluTailDualConfig {
        iterations: 0,
        learning_rate: 0.1,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
        wall_time: TAIL_M17_WALL_TIME,
        limits: ReluTailDualLimits {
            max_value_dim: domain.value_dim(),
            max_alpha_dim: domain.alpha_dim(),
            max_constraints: domain.constraint_count(),
            max_generator_nonzeros: generator_nonzeros(domain)?,
            max_optimizable_slopes: domain.value_dim(),
            max_iterations: 1,
            max_search_work: limits.max_m17_search_work,
            max_wall_time: TAIL_M17_WALL_TIME,
        },
    })
}

fn depth_two_transform_failure(
    error: &ReluTailConvBatchNormPullbackError,
) -> CganCzDepthTwoTransformFailure {
    match error {
        ReluTailConvBatchNormPullbackError::Conv2d(_) => CganCzDepthTwoTransformFailure::Conv2d,
        ReluTailConvBatchNormPullbackError::BatchNorm(_) => {
            CganCzDepthTwoTransformFailure::BatchNorm
        }
        ReluTailConvBatchNormPullbackError::ReluTail(_) => CganCzDepthTwoTransformFailure::ReluTail,
    }
}

fn depth_two_error_measurement(
    error: ReluTailConvBatchNormPullbackBudgetError,
) -> CganCzDepthTwoMeasurement {
    match error {
        ReluTailConvBatchNormPullbackBudgetError::Budget(error) => {
            CganCzDepthTwoMeasurement::BudgetFallback(error)
        }
        ReluTailConvBatchNormPullbackBudgetError::Transform(error) => {
            CganCzDepthTwoMeasurement::TransformFallback(depth_two_transform_failure(&error))
        }
    }
}

fn depth_two_counterfactual_selection(
    historical_lower_bound: f64,
    upstream_lower_bound: f64,
) -> (f64, f64) {
    let selected = if upstream_lower_bound > historical_lower_bound {
        upstream_lower_bound
    } else {
        historical_lower_bound
    };
    (selected, upstream_lower_bound - historical_lower_bound)
}

fn validate_depth_two_m17_m20_portfolio(
    portfolio: &ny_mip::ReluTailBoxCutDualResult,
    optional_budget_error: Option<&ConstrainedZonotopeCallBudgetError>,
    expected_max_peak_live_bytes: usize,
) -> Option<(Option<f64>, CganCzM20Status, ReluTailBoxCutSelection)> {
    if !portfolio.original.lower_bound.is_finite() || portfolio.box_cut.is_some() {
        return None;
    }
    let mut expected_lower_bound = portfolio.original.lower_bound;
    let mut expected_selection = ReluTailBoxCutSelection::Original;
    let (m20_lower_bound, m20_status, expected_status) = match portfolio.auxiliary.as_ref() {
        Some(auxiliary) if auxiliary.lower_bound.is_finite() => {
            if auxiliary.lower_bound > expected_lower_bound {
                expected_lower_bound = auxiliary.lower_bound;
                expected_selection = ReluTailBoxCutSelection::Auxiliary;
            }
            (
                Some(auxiliary.lower_bound),
                CganCzM20Status::Completed,
                ReluTailBoxCutStatus::Completed,
            )
        }
        Some(_) => return None,
        None => (
            None,
            CganCzM20Status::Fallback,
            ReluTailBoxCutStatus::AuxiliaryFallback,
        ),
    };
    if portfolio.status != expected_status
        || portfolio.selected != expected_selection
        || portfolio.lower_bound.to_bits() != expected_lower_bound.to_bits()
        || !depth_two_m20_optional_budget_error_is_coherent(
            m20_status,
            optional_budget_error,
            expected_max_peak_live_bytes,
        )
    {
        return None;
    }
    Some((m20_lower_bound, m20_status, expected_selection))
}

fn depth_two_m20_optional_budget_error_is_coherent(
    status: CganCzM20Status,
    error: Option<&ConstrainedZonotopeCallBudgetError>,
    expected_max_peak_live_bytes: usize,
) -> bool {
    match (status, error) {
        (CganCzM20Status::Completed, None) | (CganCzM20Status::Fallback, None) => true,
        (
            CganCzM20Status::Fallback,
            Some(
                ConstrainedZonotopeCallBudgetError::DeadlineExpired { .. }
                | ConstrainedZonotopeCallBudgetError::ResourceOverflow { .. },
            ),
        ) => true,
        (
            CganCzM20Status::Fallback,
            Some(ConstrainedZonotopeCallBudgetError::PeakLiveBytesExceeded { required, limit }),
        ) => required > limit && *limit == expected_max_peak_live_bytes,
        _ => false,
    }
}

fn completed_depth_two_measurement(
    historical_lower_bound: f64,
    result: ReluTailConvBatchNormPullbackM17M20Result,
    report: ConstrainedZonotopeCallReport,
    expected_max_peak_live_bytes: usize,
) -> CganCzDepthTwoMeasurement {
    if !historical_lower_bound.is_finite()
        || !result.downstream.lower_bound.is_finite()
        || !depth_two_plan_is_exact(result.plan)
    {
        return depth_two_setup_fallback();
    }
    let Some((upstream_m20_lower_bound, upstream_m20_status, upstream_m17_m20_selection)) =
        validate_depth_two_m17_m20_portfolio(
            &result.upstream,
            result.optional_budget_error.as_ref(),
            expected_max_peak_live_bytes,
        )
    else {
        return depth_two_setup_fallback();
    };
    let downstream_m17_candidates = summarize_m17_candidates(&result.downstream);
    let upstream_m17_candidates = summarize_m17_candidates(&result.upstream.original);
    let upstream_lower_bound = result.upstream.lower_bound;
    let (counterfactual_lower_bound, signed_gain) =
        depth_two_counterfactual_selection(historical_lower_bound, upstream_lower_bound);
    if !counterfactual_lower_bound.is_finite() || !signed_gain.is_finite() {
        return depth_two_setup_fallback();
    }
    CganCzDepthTwoMeasurement::Completed(CganCzDepthTwoCompletedMeasurement {
        historical_lower_bound,
        downstream_m17_candidates,
        upstream_m17_candidates,
        upstream_m20_lower_bound,
        upstream_m20_status,
        upstream_m20_optional_budget_error: result.optional_budget_error,
        upstream_m17_m20_selection,
        counterfactual_lower_bound,
        signed_gain,
        plan: result.plan,
        peak_live_bytes: report.peak_live_bytes(),
        charged_items: report.charged_items(),
        deadline_polls: report.deadline_polls(),
    })
}

fn validate_output_tail_contract<N>(
    domain: &ConstrainedZonotope64,
    tail_bn: &BatchNormParameters,
    tail_affine: &AffineParameters,
    coordinator: &mut Coordinator<N>,
) -> Result<(), CganCzProbeDecline>
where
    N: FnMut(&'static str) -> Instant,
{
    if domain.value_dim() != TAIL_VALUE_DIM {
        return Err(CganCzProbeDecline::OutputTail {
            message: format!(
                "Relu_23 input has {} values; expected {TAIL_VALUE_DIM}",
                domain.value_dim()
            ),
        });
    }
    for (field, values) in [
        ("gamma", tail_bn.gamma.as_slice()),
        ("beta", tail_bn.beta.as_slice()),
        ("mean", tail_bn.mean.as_slice()),
        ("variance", tail_bn.variance.as_slice()),
        ("normalized scale", tail_bn.normalized_scale.as_slice()),
        ("normalized bias", tail_bn.normalized_bias.as_slice()),
    ] {
        if values.len() != TAIL_CHANNELS {
            return Err(CganCzProbeDecline::OutputTail {
                message: format!(
                    "BatchNormalization_24 {field} has {} channels; expected {TAIL_CHANNELS}",
                    values.len()
                ),
            });
        }
    }
    if !tail_bn.epsilon.is_finite() || tail_bn.epsilon <= 0.0 {
        return Err(CganCzProbeDecline::OutputTail {
            message: format!(
                "BatchNormalization_24 epsilon must be finite and positive, got {}",
                tail_bn.epsilon
            ),
        });
    }
    for channel in 0..TAIL_CHANNELS {
        coordinator.charge(1, "cGAN output-tail BatchNorm validation")?;
        for (field, value) in [
            ("gamma", tail_bn.gamma[channel]),
            ("beta", tail_bn.beta[channel]),
            ("mean", tail_bn.mean[channel]),
            ("variance", tail_bn.variance[channel]),
            ("normalized scale", tail_bn.normalized_scale[channel]),
            ("normalized bias", tail_bn.normalized_bias[channel]),
        ] {
            if !value.is_finite() {
                return Err(CganCzProbeDecline::OutputTail {
                    message: format!("BatchNormalization_24 {field}[{channel}] must be finite"),
                });
            }
        }
    }
    if tail_affine.weights.dim() != (1, TAIL_VALUE_DIM) || tail_affine.bias.len() != 1 {
        return Err(CganCzProbeDecline::OutputTail {
            message: format!(
                "Gemm_27 must be 1x{TAIL_VALUE_DIM} with one bias, got {:?} and {} biases",
                tail_affine.weights.dim(),
                tail_affine.bias.len()
            ),
        });
    }
    if !tail_affine.bias[0].is_finite() {
        return Err(CganCzProbeDecline::OutputTail {
            message: "Gemm_27 bias must be finite".to_string(),
        });
    }
    for (coordinate, &weight) in tail_affine.weights.row(0).iter().enumerate() {
        coordinator.charge(1, "cGAN output-tail Gemm validation")?;
        if !weight.is_finite() {
            return Err(CganCzProbeDecline::OutputTail {
                message: format!("Gemm_27 weight[0,{coordinate}] must be finite"),
            });
        }
    }
    coordinator.checkpoint("cGAN output-tail contract validation")?;
    Ok(())
}

fn exact_batch_norm_tail_correction<N>(
    domain: &ConstrainedZonotope64,
    certificate: &ExactBatchNormAffineSurrogateCertificate,
    tail_affine: &AffineParameters,
    retained_baseline: usize,
    coordinator: &mut Coordinator<N>,
) -> Result<(BigRational, usize), CganCzProbeDecline>
where
    N: FnMut(&'static str) -> Instant,
{
    if certificate.channels().len() != TAIL_CHANNELS {
        return Err(CganCzProbeDecline::OutputTail {
            message: format!(
                "BatchNormalization_24 certificate has {} channels; expected {TAIL_CHANNELS}",
                certificate.channels().len()
            ),
        });
    }
    let certificate_baseline = retained_baseline
        .checked_add(certificate.conservative_live_bytes())
        .ok_or(CganCzProbeDecline::ResourceOverflow {
            operation: "cGAN retained BatchNorm certificate bytes",
        })?;
    let rational_slots = domain
        .value_dim()
        .checked_add(TAIL_TRANSIENT_RATIONAL_SLOTS)
        .ok_or(CganCzProbeDecline::ResourceOverflow {
            operation: "cGAN tail-correction rational slots",
        })?;
    let correction_peak_live_bytes = preflight_tail_rationals(
        domain,
        certificate_baseline,
        rational_slots,
        "cGAN tail-correction peak",
        coordinator,
    )?;

    let mut radii = Vec::new();
    radii
        .try_reserve_exact(domain.value_dim())
        .map_err(|_| CganCzProbeDecline::OutputTail {
            message: "unable to reserve exact Relu_23 coordinate radii".to_string(),
        })?;
    for (coordinate, &remainder) in domain.box_remainder().iter().enumerate() {
        coordinator.charge(1, "cGAN exact tail radius initialization")?;
        if remainder < 0.0 {
            return Err(CganCzProbeDecline::OutputTail {
                message: format!("Relu_23 box remainder[{coordinate}] is negative"),
            });
        }
        radii.push(exact_tail_f64(
            remainder,
            "Relu_23 box remainder",
            coordinate,
        )?);
    }
    for generator in domain.generators() {
        coordinator.charge(1, "cGAN exact tail generator scan")?;
        for (coordinate, coefficient) in generator.entries() {
            coordinator.charge(1, "cGAN exact tail radius accumulation")?;
            let Some(radius) = radii.get_mut(coordinate) else {
                return Err(CganCzProbeDecline::OutputTail {
                    message: format!(
                        "Relu_23 generator coordinate {coordinate} exceeds {} values",
                        domain.value_dim()
                    ),
                });
            };
            *radius +=
                exact_tail_f64(coefficient, "Relu_23 generator coefficient", coordinate)?.abs();
        }
    }

    let zero = BigRational::zero();
    let mut correction = BigRational::zero();
    for (coordinate, (&center, radius)) in domain.center().iter().zip(radii).enumerate() {
        coordinator.charge(8, "cGAN exact BatchNorm-tail correction")?;
        let center = exact_tail_f64(center, "Relu_23 center", coordinate)?;
        let relu_upper = (center + radius).max(zero.clone());
        let channel = coordinate / TAIL_VALUES_PER_CHANNEL;
        let weight = exact_tail_f64(
            tail_affine.weights[(0, coordinate)],
            "Gemm_27 weight",
            coordinate,
        )?;
        let channel_certificate = &certificate.channels()[channel];
        correction += weight.abs()
            * (channel_certificate.scale_error().clone() * relu_upper
                + channel_certificate.bias_error());
    }
    if correction.is_negative() {
        return Err(CganCzProbeDecline::OutputTail {
            message: "BatchNormalization_24 exact error correction became negative".to_string(),
        });
    }
    ExactReluTailMargin::try_new(Vec::new(), correction.clone()).map_err(|error| {
        CganCzProbeDecline::OutputTail {
            message: format!("BatchNormalization_24 correction exceeds M17 caps: {error}"),
        }
    })?;
    coordinator.checkpoint("cGAN exact BatchNorm-tail correction publication")?;
    Ok((correction, correction_peak_live_bytes))
}

#[allow(clippy::too_many_arguments)]
fn exact_output_tail_margin<N>(
    domain: &ConstrainedZonotope64,
    tail_bn: &BatchNormParameters,
    tail_affine: &AffineParameters,
    correction: &BigRational,
    sense: TailMarginSense,
    retained_baseline: usize,
    coordinator: &mut Coordinator<N>,
) -> Result<(ExactReluTailMargin, usize), CganCzProbeDecline>
where
    N: FnMut(&'static str) -> Instant,
{
    let rational_slots = domain
        .value_dim()
        .checked_add(TAIL_TRANSIENT_RATIONAL_SLOTS + 2)
        .ok_or(CganCzProbeDecline::ResourceOverflow {
            operation: "cGAN output-margin rational slots",
        })?;
    let objective_peak_live_bytes = preflight_tail_rationals(
        domain,
        retained_baseline,
        rational_slots,
        "cGAN output-margin peak",
        coordinator,
    )?;

    let mut coefficients = Vec::new();
    coefficients
        .try_reserve_exact(domain.value_dim())
        .map_err(|_| CganCzProbeDecline::OutputTail {
            message: "unable to reserve exact Gemm_27 objective".to_string(),
        })?;
    let mut nominal_bias = exact_tail_f64(tail_affine.bias[0], "Gemm_27 bias", 0)?;
    for coordinate in 0..domain.value_dim() {
        coordinator.charge(5, "cGAN exact output-tail objective")?;
        let channel = coordinate / TAIL_VALUES_PER_CHANNEL;
        let weight = exact_tail_f64(
            tail_affine.weights[(0, coordinate)],
            "Gemm_27 weight",
            coordinate,
        )?;
        let scale = exact_tail_f64(
            tail_bn.normalized_scale[channel],
            "BatchNormalization_24 normalized scale",
            channel,
        )?;
        let bias = exact_tail_f64(
            tail_bn.normalized_bias[channel],
            "BatchNormalization_24 normalized bias",
            channel,
        )?;
        let coefficient = &weight * scale;
        nominal_bias += weight * bias;
        coefficients.push(match sense {
            TailMarginSense::Lower => coefficient,
            TailMarginSense::NegatedUpper => -coefficient,
        });
    }
    let bias = match sense {
        TailMarginSense::Lower => nominal_bias - correction,
        TailMarginSense::NegatedUpper => -nominal_bias - correction,
    };
    let margin = ExactReluTailMargin::try_new(coefficients, bias).map_err(|error| {
        CganCzProbeDecline::OutputTail {
            message: format!("exact Gemm_27 objective exceeds M17 caps: {error}"),
        }
    })?;
    coordinator.checkpoint("cGAN exact output-tail objective publication")?;
    Ok((margin, objective_peak_live_bytes))
}

#[allow(clippy::too_many_arguments)]
fn run_output_tail_m17<N>(
    domain: &ConstrainedZonotope64,
    prepared_auxiliary: Option<(&CertifiedAuxiliaryBounds64, &PreparedReluTailGeometry64<'_>)>,
    margin: &ExactReluTailMargin,
    kind: CganCzStageKind,
    limits: CganCzSequentialLimits,
    retained_baseline: usize,
    budget: ConstrainedZonotopeCallBudget,
    coordinator: &mut Coordinator<N>,
    stages: &mut Vec<CganCzStageTelemetry>,
    objective_peak_live_bytes: usize,
    stage_charged_start: usize,
    stage_polls_start: usize,
) -> Result<CganCzTailPortfolio, CganCzProbeDecline>
where
    N: FnMut(&'static str) -> Instant,
{
    let input_generator_nonzeros = generator_nonzeros(domain)?;
    let caller_rational_slots =
        domain
            .value_dim()
            .checked_add(2)
            .ok_or(CganCzProbeDecline::ResourceOverflow {
                operation: "cGAN M17 caller rational slots",
            })?;
    let caller_rational_bytes =
        tail_rational_bytes(caller_rational_slots, "cGAN M17 caller rational live bytes")?;
    let call_retained_baseline = retained_baseline.checked_add(caller_rational_bytes).ok_or(
        CganCzProbeDecline::ResourceOverflow {
            operation: "cGAN M17 retained baseline",
        },
    )?;
    let call_budget = domain_call_budget(domain, call_retained_baseline, budget, coordinator)?;
    let max_iterations = limits.max_m17_iterations.max(1);
    let config = ReluTailDualConfig {
        iterations: limits.max_m17_iterations,
        learning_rate: 0.1,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
        wall_time: TAIL_M17_WALL_TIME,
        limits: ReluTailDualLimits {
            max_value_dim: domain.value_dim(),
            max_alpha_dim: domain.alpha_dim(),
            max_constraints: domain.constraint_count(),
            max_generator_nonzeros: input_generator_nonzeros,
            max_optimizable_slopes: domain.value_dim(),
            max_iterations,
            max_search_work: limits.max_m17_search_work,
            max_wall_time: TAIL_M17_WALL_TIME,
        },
    };
    let (portfolio, report, unstable_coordinates) =
        if let Some((auxiliary, prepared)) = prepared_auxiliary {
            let outcome = prepared
                .bound_margin_m17_m20_m24_unwired_with_budget(
                    auxiliary,
                    margin,
                    None,
                    config,
                    cgan_m24_config(),
                    call_budget,
                )
                .map_err(map_tail_dual_budget_error)?;
            let (result, report) = outcome.into_parts();
            let (historical_lower_bound, m20_status) =
                validate_cgan_m24_core_result(&result, domain, input_generator_nonzeros)?;
            let optimized = result.optimized;
            let optional_budget_error = result.optional_budget_error;
            let exact_box_cut_lower_bound = optimized
                .portfolio
                .box_cut
                .as_ref()
                .map(|certificate| certificate.lower_bound);
            let m20_lower_bound = optimized
                .portfolio
                .auxiliary
                .as_ref()
                .map(|value| value.lower_bound);
            let m17_candidates = summarize_m17_candidates(&optimized.portfolio.original);
            let unstable_coordinates = optimized.portfolio.original.optimizable_slopes;
            let measurement = CganCzM24Measurement {
                exact_box_cut_lower_bound,
                counterfactual_lower_bound: optimized.lower_bound,
                counterfactual_selection: optimized.selected,
                replay_status: optimized.portfolio.status,
                search_status: optimized.search_status,
                search_plan: optimized.search_plan,
                iterations_completed: optimized.iterations_completed,
                restarts_completed: optimized.restarts_completed,
                candidates_scored: optimized.candidates_scored,
                exact_replays: optimized.exact_replays,
                optional_budget_error,
            };
            let portfolio = CganCzTailPortfolio {
                selected_lower_bound: historical_lower_bound,
                m17_candidates,
                m20_lower_bound,
                m20_status,
                m24_measurement: Some(measurement),
            };
            (portfolio, report, unstable_coordinates)
        } else {
            let outcome = bound_relu_tail_triangle_dual_unwired_with_budget(
                domain,
                margin,
                None,
                config,
                call_budget,
            )
            .map_err(map_tail_dual_budget_error)?;
            let (result, report) = outcome.into_parts();
            let portfolio = CganCzTailPortfolio {
                selected_lower_bound: result.lower_bound,
                m17_candidates: summarize_m17_candidates(&result),
                m20_lower_bound: None,
                m20_status: CganCzM20Status::NotRequested,
                m24_measurement: None,
            };
            let unstable_coordinates = result.optimizable_slopes;
            drop(result);
            (portfolio, report, unstable_coordinates)
        };
    coordinator.absorb(report)?;
    coordinator.checkpoint("cGAN M17/M20/M24 stage publication")?;
    let stage_charged_items = coordinator
        .charged_items
        .checked_sub(stage_charged_start)
        .ok_or(CganCzProbeDecline::ResourceOverflow {
            operation: "cGAN M17 stage charged-item delta",
        })?;
    let stage_deadline_polls = coordinator
        .deadline_polls
        .checked_sub(stage_polls_start)
        .ok_or(CganCzProbeDecline::ResourceOverflow {
            operation: "cGAN M17 stage deadline-poll delta",
        })?;
    record_scalar_tail_stage_accounting(
        stages,
        "Gemm_27",
        kind,
        domain.alpha_dim(),
        input_generator_nonzeros,
        unstable_coordinates,
        objective_peak_live_bytes.max(report.peak_live_bytes()),
        stage_charged_items,
        stage_deadline_polls,
    )?;
    Ok(portfolio)
}

fn cgan_m24_config() -> ReluTailBoxCutOptimizerConfig {
    ReluTailBoxCutOptimizerConfig {
        schedules: [
            ReluTailBoxCutAdamSchedule {
                iterations: TAIL_M24_SCHEDULE_ITERATIONS,
                learning_rate: 0.005,
                decay: 0.98,
            },
            ReluTailBoxCutAdamSchedule {
                iterations: TAIL_M24_SCHEDULE_ITERATIONS,
                learning_rate: 0.1,
                decay: 0.98,
            },
        ],
        multiplier_cap: 16.0,
        wall_time: TAIL_M24_WALL_TIME,
        limits: ReluTailBoxCutOptimizerLimits {
            max_value_dim: TAIL_VALUE_DIM,
            max_box_variables: TAIL_M24_MAX_BOX_VARIABLES,
            max_total_iterations: TAIL_M24_MAX_TOTAL_ITERATIONS,
            max_restarts: TAIL_M24_MAX_RESTARTS,
            max_exact_replays: TAIL_M24_MAX_EXACT_REPLAYS,
            max_generator_nonzeros: TAIL_M24_MAX_GENERATOR_NONZEROS,
            max_search_work: TAIL_M24_MAX_SEARCH_WORK,
            max_wall_time: TAIL_M24_WALL_TIME,
        },
    }
}

fn validate_cgan_m24_core_result(
    result: &ny_mip::ReluTailBoxCutBudgetedResult,
    domain: &ConstrainedZonotope64,
    input_generator_nonzeros: usize,
) -> Result<(f64, CganCzM20Status), CganCzProbeDecline> {
    let optimized = &result.optimized;
    let portfolio = &optimized.portfolio;
    let malformed = |message: String| CganCzProbeDecline::OutputTail { message };
    if !portfolio.original.lower_bound.is_finite() {
        return Err(malformed(
            "M24 core returned a non-finite mandatory M17 certificate".to_string(),
        ));
    }
    let (m20_lower_bound, m20_status) = match portfolio.auxiliary.as_ref() {
        Some(auxiliary) if auxiliary.lower_bound.is_finite() => {
            (Some(auxiliary.lower_bound), CganCzM20Status::Completed)
        }
        Some(_) => {
            return Err(malformed(
                "M24 core returned a non-finite M20 certificate".to_string(),
            ));
        }
        None => (None, CganCzM20Status::Fallback),
    };
    let Some(historical_lower_bound) =
        select_m17_m20_lower_bound(portfolio.original.lower_bound, m20_lower_bound, m20_status)
    else {
        return Err(malformed(
            "M24 core returned malformed historical M17/M20 attribution".to_string(),
        ));
    };

    let mut expected_counterfactual = portfolio.original.lower_bound;
    let mut expected_selection = ReluTailBoxCutSelection::Original;
    if let Some(auxiliary) = portfolio.auxiliary.as_ref() {
        if auxiliary.lower_bound > expected_counterfactual {
            expected_counterfactual = auxiliary.lower_bound;
            expected_selection = ReluTailBoxCutSelection::Auxiliary;
        }
    }
    if let Some(box_cut) = portfolio.box_cut.as_ref() {
        if !box_cut.lower_bound.is_finite() {
            return Err(malformed(
                "M24 core returned a non-finite exact Box-cut certificate".to_string(),
            ));
        }
        if box_cut.lower_bound > expected_counterfactual {
            expected_counterfactual = box_cut.lower_bound;
            expected_selection = ReluTailBoxCutSelection::BoxCut;
        }
    }
    let expected_replay_status = match (portfolio.auxiliary.is_some(), portfolio.box_cut.is_some())
    {
        (false, false) => ReluTailBoxCutStatus::AuxiliaryFallback,
        (true, false) => ReluTailBoxCutStatus::CandidateFallback,
        (true, true) => ReluTailBoxCutStatus::Completed,
        (false, true) => {
            return Err(malformed(
                "M24 core returned a Box-cut certificate without M20".to_string(),
            ));
        }
    };
    if portfolio.status != expected_replay_status
        || optimized.selected != portfolio.selected
        || optimized.selected != expected_selection
        || optimized.lower_bound.to_bits() != portfolio.lower_bound.to_bits()
        || optimized.lower_bound.to_bits() != expected_counterfactual.to_bits()
    {
        return Err(malformed(
            "M24 core returned inconsistent replay selection or bounds".to_string(),
        ));
    }

    if let Some(plan) = optimized.search_plan {
        let expected_search_work = cgan_m24_plan_search_work(plan)?;
        let max_candidates = plan.total_iterations.checked_add(plan.restarts).ok_or(
            CganCzProbeDecline::ResourceOverflow {
                operation: "cGAN M24 candidate telemetry ceiling",
            },
        )?;
        if plan.value_dim != domain.value_dim()
            || plan.value_dim != TAIL_VALUE_DIM
            || plan.alpha_dim != domain.alpha_dim()
            || plan.alpha_dim > TAIL_DISCRIMINATOR_RETAINED_ALPHA_DIM
            || plan.generator_nonzeros != input_generator_nonzeros
            || plan.box_variables == 0
            || plan.box_variables > TAIL_M24_MAX_BOX_VARIABLES
            || plan.total_iterations != TAIL_M24_MAX_TOTAL_ITERATIONS
            || plan.restarts != TAIL_M24_MAX_RESTARTS
            || plan.exact_replays != TAIL_M24_MAX_EXACT_REPLAYS
            || plan.generator_nonzeros > TAIL_M24_MAX_GENERATOR_NONZEROS
            || plan.search_work != expected_search_work
            || plan.search_work > TAIL_M24_MAX_SEARCH_WORK
            || optimized.iterations_completed > plan.total_iterations
            || optimized.restarts_completed > plan.restarts
            || optimized.candidates_scored > max_candidates
            || optimized.exact_replays > plan.exact_replays
            || optimized.exact_replays > optimized.candidates_scored
        {
            return Err(malformed(
                "M24 core returned telemetry inconsistent with the fixed cGAN plan".to_string(),
            ));
        }
    } else if optimized.iterations_completed != 0
        || optimized.restarts_completed != 0
        || optimized.candidates_scored != 0
        || optimized.exact_replays != 0
        || portfolio.box_cut.is_some()
    {
        return Err(malformed(
            "M24 core returned work telemetry without a checked search plan".to_string(),
        ));
    }
    if portfolio.box_cut.is_some() && optimized.exact_replays == 0 {
        return Err(malformed(
            "M24 core returned a Box-cut certificate without an exact replay".to_string(),
        ));
    }
    let search_status_matches_m20 =
        if optimized.search_status == ReluTailBoxCutOptimizerStatus::AuxiliaryFallback {
            m20_status == CganCzM20Status::Fallback
        } else {
            m20_status == CganCzM20Status::Completed
        };
    if !search_status_matches_m20 {
        return Err(malformed(
            "M24 core returned search status inconsistent with M20 attribution".to_string(),
        ));
    }
    let budget_error_is_coherent = match result.optional_budget_error.as_ref() {
        Some(ConstrainedZonotopeCallBudgetError::DeadlineExpired { .. }) => matches!(
            optimized.search_status,
            ReluTailBoxCutOptimizerStatus::Deadline
                | ReluTailBoxCutOptimizerStatus::AuxiliaryFallback
        ),
        Some(ConstrainedZonotopeCallBudgetError::ResourceOverflow { .. }) => matches!(
            optimized.search_status,
            ReluTailBoxCutOptimizerStatus::ResourceFallback
                | ReluTailBoxCutOptimizerStatus::AuxiliaryFallback
        ),
        Some(ConstrainedZonotopeCallBudgetError::PeakLiveBytesExceeded { required, limit }) => {
            required > limit
                && matches!(
                    optimized.search_status,
                    ReluTailBoxCutOptimizerStatus::ResourceFallback
                        | ReluTailBoxCutOptimizerStatus::AuxiliaryFallback
                )
        }
        None => true,
    };
    if !budget_error_is_coherent {
        return Err(malformed(
            "M24 core returned search status inconsistent with its budget refusal".to_string(),
        ));
    }
    let state_is_reachable = match (
        optimized.search_plan,
        optimized.search_status,
        result.optional_budget_error.as_ref(),
    ) {
        (None, ReluTailBoxCutOptimizerStatus::AuxiliaryFallback, _) => true,
        (None, ReluTailBoxCutOptimizerStatus::NoTighterAuxiliaryBox, None) => true,
        (None, ReluTailBoxCutOptimizerStatus::ResourceFallback, Some(_)) => true,
        (
            None,
            ReluTailBoxCutOptimizerStatus::Deadline,
            Some(ConstrainedZonotopeCallBudgetError::DeadlineExpired { .. }),
        ) => true,
        (Some(plan), ReluTailBoxCutOptimizerStatus::Completed, None) => {
            portfolio.box_cut.is_some()
                && optimized.iterations_completed == plan.total_iterations
                && optimized.restarts_completed == plan.restarts
                && optimized.candidates_scored == plan.total_iterations + plan.restarts
                && optimized.exact_replays == plan.exact_replays
        }
        (
            Some(_),
            ReluTailBoxCutOptimizerStatus::ResourceFallback,
            Some(
                ConstrainedZonotopeCallBudgetError::ResourceOverflow { .. }
                | ConstrainedZonotopeCallBudgetError::PeakLiveBytesExceeded { .. },
            ),
        ) => true,
        (Some(_), ReluTailBoxCutOptimizerStatus::Deadline, None)
        | (
            Some(_),
            ReluTailBoxCutOptimizerStatus::Deadline,
            Some(ConstrainedZonotopeCallBudgetError::DeadlineExpired { .. }),
        ) => true,
        (Some(_), ReluTailBoxCutOptimizerStatus::NonFiniteCandidate, None)
        | (Some(_), ReluTailBoxCutOptimizerStatus::AllocationFallback, None) => true,
        (Some(_), ReluTailBoxCutOptimizerStatus::ExactReplayFallback, None) => {
            optimized.exact_replays > 0
        }
        _ => false,
    };
    if !state_is_reachable {
        return Err(malformed(
            "M24 core returned an unreachable fixed-policy search receipt".to_string(),
        ));
    }
    Ok((historical_lower_bound, m20_status))
}

fn cgan_m24_plan_search_work(plan: ReluTailBoxCutOptimizerPlan) -> Result<u64, CganCzProbeDecline> {
    let values = plan.value_dim as u128;
    let variables = plan.box_variables as u128;
    let sparse = plan.generator_nonzeros as u128;
    let alpha = plan.alpha_dim as u128;
    let overflow = || CganCzProbeDecline::ResourceOverflow {
        operation: "cGAN M24 exact search-work derivation",
    };
    let score = values
        .checked_mul(3)
        .and_then(|work| work.checked_add(variables.checked_mul(2)?))
        .and_then(|work| work.checked_add(sparse.checked_mul(2)?))
        .and_then(|work| work.checked_add(alpha))
        .ok_or_else(overflow)?;
    let restart_startup = variables
        .checked_mul(6)
        .and_then(|work| work.checked_add(values.checked_mul(2)?))
        .and_then(|work| work.checked_add(score))
        .ok_or_else(overflow)?;
    let per_iteration = variables
        .checked_mul(6)
        .and_then(|work| work.checked_add(score))
        .ok_or_else(overflow)?;
    let work = values
        .checked_add(
            restart_startup
                .checked_mul(plan.restarts as u128)
                .ok_or_else(overflow)?,
        )
        .and_then(|work| {
            work.checked_add(per_iteration.checked_mul(plan.total_iterations as u128)?)
        })
        .ok_or_else(overflow)?;
    u64::try_from(work).map_err(|_| overflow())
}

fn map_tail_dual_budget_error(error: ReluTailDualBudgetError) -> CganCzProbeDecline {
    match error {
        ReluTailDualBudgetError::Budget(error) => CganCzProbeDecline::Budget(error),
        ReluTailDualBudgetError::Bound(error) => CganCzProbeDecline::OutputTail {
            message: format!("M17 exact replay declined: {error}"),
        },
    }
}

fn summarize_m17_candidates(result: &ReluTailDualResult) -> CganCzM17CandidateTelemetry {
    let replays = result.zero_predicate_candidate_replays;
    summarize_m17_candidate_values(
        result.lower_bound,
        replays.zero_positive_slope_lower_bound,
        replays.upper_endpoint_lower_bound,
        replays.canonical_lower_bound,
        replays.optimized_lower_bound,
        result.optimizable_slopes,
        result.candidates_replayed,
        result.iterations_completed,
        result.status,
    )
}

#[allow(clippy::too_many_arguments)]
fn summarize_m17_candidate_values(
    selected_lower_bound: f64,
    zero_positive_slope_lower_bound: f64,
    upper_endpoint_lower_bound: Option<f64>,
    canonical_lower_bound: Option<f64>,
    optimized_lower_bound: Option<f64>,
    optimizable_slopes: usize,
    candidates_replayed: usize,
    iterations_completed: usize,
    status: ReluTailDualStatus,
) -> CganCzM17CandidateTelemetry {
    let mut best_nonoptimized_lower_bound = zero_positive_slope_lower_bound;
    for candidate in [upper_endpoint_lower_bound, canonical_lower_bound]
        .into_iter()
        .flatten()
    {
        best_nonoptimized_lower_bound = best_nonoptimized_lower_bound.max(candidate);
    }
    let optimized_improvement = optimized_lower_bound.map_or(0.0, |optimized| {
        (optimized - best_nonoptimized_lower_bound).max(0.0)
    });
    CganCzM17CandidateTelemetry {
        selected_lower_bound,
        zero_positive_slope_lower_bound,
        upper_endpoint_lower_bound,
        canonical_lower_bound,
        optimized_lower_bound,
        best_nonoptimized_lower_bound,
        optimized_improvement,
        optimizable_slopes,
        candidates_replayed,
        iterations_completed,
        status,
    }
}

fn preflight_tail_rationals<N>(
    domain: &ConstrainedZonotope64,
    retained_baseline: usize,
    rational_slots: usize,
    operation: &'static str,
    coordinator: &mut Coordinator<N>,
) -> Result<usize, CganCzProbeDecline>
where
    N: FnMut(&'static str) -> Instant,
{
    let rational_bytes = tail_rational_bytes(rational_slots, operation)?;
    let required = retained_baseline
        .checked_add(domain_live_bytes(domain)?)
        .and_then(|bytes| bytes.checked_add(rational_bytes))
        .ok_or(CganCzProbeDecline::ResourceOverflow { operation })?;
    coordinator.preflight_absolute_peak(required)?;
    Ok(required)
}

fn tail_rational_bytes(
    rational_slots: usize,
    operation: &'static str,
) -> Result<usize, CganCzProbeDecline> {
    rational_slots
        .checked_mul(TAIL_RATIONAL_LIVE_BYTES)
        .ok_or(CganCzProbeDecline::ResourceOverflow { operation })
}

fn exact_tail_f64(
    value: f64,
    field: &'static str,
    index: usize,
) -> Result<BigRational, CganCzProbeDecline> {
    BigRational::from_float(value).ok_or_else(|| CganCzProbeDecline::OutputTail {
        message: format!("{field}[{index}] must have a finite exact dyadic representation"),
    })
}

fn exact_nonnegative_to_upper_f64(value: &BigRational) -> Result<f64, CganCzProbeDecline> {
    if value.is_negative() {
        return Err(CganCzProbeDecline::OutputTail {
            message: "exact BatchNorm tail correction is negative".to_string(),
        });
    }
    let candidate = value
        .to_f64()
        .filter(|value| value.is_finite())
        .ok_or_else(|| CganCzProbeDecline::OutputTail {
            message: "exact BatchNorm tail correction has no finite f64 enclosure".to_string(),
        })?;
    let candidate_exact =
        BigRational::from_float(candidate).ok_or_else(|| CganCzProbeDecline::OutputTail {
            message: "rounded BatchNorm tail correction is not finite".to_string(),
        })?;
    let outward = if candidate_exact < *value {
        candidate.next_up()
    } else {
        candidate
    };
    if !outward.is_finite() || BigRational::from_float(outward).is_none_or(|exact| exact < *value) {
        return Err(CganCzProbeDecline::OutputTail {
            message: "BatchNorm tail correction cannot be rounded outward to f64".to_string(),
        });
    }
    Ok(outward)
}

fn tail_bounds_separate_unsafe_moat(
    lower_bound: f64,
    upper_bound: f64,
    low_unsafe_threshold: f64,
    high_unsafe_threshold: f64,
) -> bool {
    lower_bound > low_unsafe_threshold && upper_bound < high_unsafe_threshold
}

fn independent_auxiliary_live_bytes(
    bounds: &[CganCzIndependentIntervalReluBounds],
) -> Result<usize, CganCzProbeDecline> {
    bounds.iter().try_fold(0_usize, |bytes, record| {
        record
            .bounds
            .value_dim()
            .checked_mul(2 * size_of::<f64>())
            .and_then(|record_bytes| bytes.checked_add(record_bytes))
            .ok_or(CganCzProbeDecline::ResourceOverflow {
                operation: "cGAN independent retained auxiliary-bound bytes",
            })
    })
}

fn authenticate_independent_relu_bound_sequence(
    profile: CganCzImgSz32Profile,
    records: &[CganCzIndependentIntervalReluBounds],
) -> Result<(), CganCzProbeDecline> {
    if records.len() != CGAN_RELU_COUNT {
        return Err(CganCzProbeDecline::Topology {
            message: format!(
                "independent auxiliary record sequence has {} entries; expected exactly {CGAN_RELU_COUNT}",
                records.len()
            ),
        });
    }
    for (record, index) in records.iter().zip(CGAN_RELU_INDICES) {
        authenticate_independent_relu_bound_record(profile, index, record)?;
    }
    Ok(())
}

fn authenticate_independent_relu_bound_record(
    profile: CganCzImgSz32Profile,
    index: usize,
    record: &CganCzIndependentIntervalReluBounds,
) -> Result<&CertifiedAuxiliaryBounds64, CganCzProbeDecline> {
    let Some((expected_node, expected_kind)) = EXPECTED_NODES.get(index) else {
        return Err(CganCzProbeDecline::Topology {
            message: format!("auxiliary record requested unknown node index {index}"),
        });
    };
    let expected_node = *expected_node;
    if expected_kind != &LayerType::ReLU || !CGAN_RELU_INDICES.contains(&index) {
        return Err(CganCzProbeDecline::Topology {
            message: format!(
                "auxiliary record requested non-authored-ReLU node {expected_node} at index {index}"
            ),
        });
    }
    let expected_shape = expected_output_shape_for_profile(profile, index);
    let expected_values =
        checked_product(expected_shape, "cGAN authenticated auxiliary ReLU shape")?;
    if record.node != expected_node
        || record.output_shape != expected_shape
        || record.bounds.value_dim() != expected_values
    {
        return Err(CganCzProbeDecline::Topology {
            message: format!(
                "auxiliary record at index {index} is node {} with shape {:?} and {} values; expected {expected_node}, {profile:?} shape {expected_shape:?}, and {expected_values} values",
                record.node,
                record.output_shape,
                record.bounds.value_dim()
            ),
        });
    }
    Ok(&record.bounds)
}

fn next_correlated_auxiliary_bounds<'a>(
    profile: CganCzImgSz32Profile,
    index: usize,
    records: &mut std::slice::Iter<'a, CganCzIndependentIntervalReluBounds>,
    consumed: &mut usize,
) -> Result<Option<&'a CertifiedAuxiliaryBounds64>, CganCzProbeDecline> {
    let Some((_, kind)) = EXPECTED_NODES.get(index) else {
        return Err(CganCzProbeDecline::Topology {
            message: format!("correlated prefix requested unknown node index {index}"),
        });
    };
    if kind != &LayerType::ReLU {
        return Ok(None);
    }
    if *consumed >= CGAN_CORRELATED_AUXILIARY_RELU_COUNT {
        return Err(CganCzProbeDecline::Topology {
            message: format!(
                "correlated prefix reached unexpected extra ReLU at node index {index}"
            ),
        });
    }
    let expected_index = CGAN_RELU_INDICES[*consumed];
    if index != expected_index {
        return Err(CganCzProbeDecline::Topology {
            message: format!(
                "correlated auxiliary record {consumed} belongs to ReLU index {expected_index}, but the prefix requested index {index}"
            ),
        });
    }
    let record = records.next().ok_or_else(|| CganCzProbeDecline::Topology {
        message: format!(
            "correlated prefix is missing auxiliary record {consumed} for {}",
            EXPECTED_NODES[index].0
        ),
    })?;
    let bounds = authenticate_independent_relu_bound_record(profile, index, record)?;
    *consumed += 1;
    Ok(Some(bounds))
}

fn independent_endpoint_bytes(value_dim: usize) -> Result<usize, CganCzProbeDecline> {
    value_dim
        .checked_mul(2 * size_of::<f64>())
        .ok_or(CganCzProbeDecline::ResourceOverflow {
            operation: "cGAN independent endpoint bytes",
        })
}

fn independent_box_limits(
    limits: CganCzSequentialLimits,
) -> Result<CertifiedBox64Limits, CganCzProbeDecline> {
    let max_stored_f64 =
        limits
            .max_value_dim
            .checked_mul(2)
            .ok_or(CganCzProbeDecline::ResourceOverflow {
                operation: "cGAN independent Box stored endpoints",
            })?;
    let max_work_items =
        limits
            .max_value_dim
            .checked_mul(4)
            .ok_or(CganCzProbeDecline::ResourceOverflow {
                operation: "cGAN independent Box bridge work items",
            })?;
    Ok(CertifiedBox64Limits {
        max_values: limits.max_value_dim,
        max_stored_f64,
        max_weight_elements: 0,
        max_work_items,
        max_scalar_products: 0,
    })
}

fn map_independent_box_bridge_error(
    node: &'static str,
    operation: &'static str,
    error: CertifiedBox64BridgeError,
) -> CganCzProbeDecline {
    match error {
        CertifiedBox64BridgeError::Budget(error) => error.into(),
        other => CganCzProbeDecline::Transform {
            node,
            operation,
            message: other.to_string(),
        },
    }
}

fn map_independent_auxiliary_error(
    node: &'static str,
    error: CertifiedAuxiliaryBounds64BudgetError,
) -> CganCzProbeDecline {
    match error {
        CertifiedAuxiliaryBounds64BudgetError::Budget(error) => error.into(),
        CertifiedAuxiliaryBounds64BudgetError::Bounds(error) => CganCzProbeDecline::Transform {
            node,
            operation: "certified auxiliary-bound publication",
            message: error.to_string(),
        },
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_independent_interval_relu<N>(
    profile: CganCzImgSz32Profile,
    index: usize,
    domain: &mut ConstrainedZonotope64,
    shape: &mut Vec<usize>,
    limits: CganCzSequentialLimits,
    retained_baseline: usize,
    budget: ConstrainedZonotopeCallBudget,
    coordinator: &mut Coordinator<N>,
    relu_bounds: &mut Vec<CganCzIndependentIntervalReluBounds>,
) -> Result<(), CganCzProbeDecline>
where
    N: FnMut(&'static str) -> Instant,
{
    let Some(&expected_index) = CGAN_RELU_INDICES.get(relu_bounds.len()) else {
        return Err(CganCzProbeDecline::ResourceLimit {
            resource: "cGAN independent ReLU-bound records",
            required: relu_bounds.len().saturating_add(1),
            limit: CGAN_RELU_COUNT,
        });
    };
    if index != expected_index || EXPECTED_NODES[index].1 != LayerType::ReLU {
        return Err(CganCzProbeDecline::Topology {
            message: format!(
                "independent interval ReLU position {} expected node index {expected_index}, got {index}",
                relu_bounds.len()
            ),
        });
    }
    let node = EXPECTED_NODES[index].0;
    let expected_shape = expected_output_shape_for_profile(profile, index);
    let expected_values = checked_product(expected_shape, "cGAN independent ReLU shape")?;
    if shape.as_slice() != expected_shape || domain.value_dim() != expected_values {
        return Err(CganCzProbeDecline::Topology {
            message: format!(
                "{node} independent preactivation has shape {shape:?} and {} values, expected {expected_shape:?}",
                domain.value_dim()
            ),
        });
    }
    check_resource(
        "cGAN independent ReLU value dimension",
        expected_values,
        limits.max_value_dim,
    )?;
    if relu_bounds.len() >= relu_bounds.capacity() {
        return Err(CganCzProbeDecline::ResourceLimit {
            resource: "cGAN independent reserved ReLU-bound records",
            required: relu_bounds.len().saturating_add(1),
            limit: relu_bounds.capacity(),
        });
    }

    let box_limits = independent_box_limits(limits)?;
    let source_domain_bytes = domain_live_bytes(domain)?;
    let cz_to_box_budget = domain_call_budget(domain, retained_baseline, budget, coordinator)?;
    let outcome = certified_box_from_remainder_only_zonotope_unwired_with_budget(
        domain,
        box_limits,
        cz_to_box_budget,
    )
    .map_err(|error| {
        map_independent_box_bridge_error(node, "remainder-only CZ-to-Box bridge", error)
    })?;
    let (preactivation_box, report) = outcome.into_parts();
    coordinator.absorb(report)?;
    if preactivation_box.len() != expected_values {
        return Err(CganCzProbeDecline::Transform {
            node,
            operation: "remainder-only CZ-to-Box bridge",
            message: format!(
                "bridge produced {} values, expected {expected_values}",
                preactivation_box.len()
            ),
        });
    }

    let endpoint_bytes = independent_endpoint_bytes(expected_values)?;
    let auxiliary_copy_baseline = retained_baseline
        .checked_add(source_domain_bytes)
        .and_then(|bytes| bytes.checked_add(endpoint_bytes))
        .ok_or(CganCzProbeDecline::ResourceOverflow {
            operation: "cGAN independent auxiliary-bound copy baseline",
        })?;
    coordinator.preflight_absolute_peak(auxiliary_copy_baseline)?;
    let auxiliary_budget = ConstrainedZonotopeCallBudget::new(
        budget.deadline(),
        auxiliary_copy_baseline,
        budget.max_peak_live_bytes(),
    );
    let outcome = CertifiedAuxiliaryBounds64::try_from_certified_box_with_budget(
        &preactivation_box,
        auxiliary_budget,
    )
    .map_err(|error| map_independent_auxiliary_error(node, error))?;
    let (auxiliary_bounds, report) = outcome.into_parts();
    coordinator.absorb(report)?;

    let recenter_baseline = retained_baseline
        .checked_add(source_domain_bytes)
        .and_then(|bytes| bytes.checked_add(endpoint_bytes))
        .and_then(|bytes| bytes.checked_add(endpoint_bytes))
        .ok_or(CganCzProbeDecline::ResourceOverflow {
            operation: "cGAN independent ReLU recenter baseline",
        })?;
    coordinator.preflight_absolute_peak(recenter_baseline)?;
    let recenter_budget = ConstrainedZonotopeCallBudget::new(
        budget.deadline(),
        recenter_baseline,
        budget.max_peak_live_bytes(),
    );
    let outcome = certified_box_relu_recenter_unwired_with_budget(
        &preactivation_box,
        box_limits,
        recenter_budget,
    )
    .map_err(|error| {
        map_independent_box_bridge_error(node, "certified interval ReLU recenter", error)
    })?;
    let (post_relu, report) = outcome.into_parts();
    coordinator.absorb(report)?;
    if post_relu.value_dim() != expected_values
        || post_relu.alpha_dim() != 0
        || post_relu.constraint_count() != 0
    {
        return Err(CganCzProbeDecline::Transform {
            node,
            operation: "independent interval ReLU structural audit",
            message: format!(
                "result has value_dim={}, alpha_dim={}, constraint_count={}",
                post_relu.value_dim(),
                post_relu.alpha_dim(),
                post_relu.constraint_count()
            ),
        });
    }
    if shape.capacity() < expected_shape.len() {
        return Err(CganCzProbeDecline::ResourceLimit {
            resource: "cGAN independent shape storage",
            required: expected_shape.len(),
            limit: shape.capacity(),
        });
    }

    // This is the final fallible operation. Keeping it before all three state
    // mutations makes the helper itself transactional, not merely its private
    // top-level caller.
    coordinator.checkpoint("cGAN independent ReLU auxiliary publication")?;
    relu_bounds.push(CganCzIndependentIntervalReluBounds {
        node,
        output_shape: expected_shape,
        bounds: auxiliary_bounds,
    });
    *domain = post_relu;
    shape.clear();
    shape.extend_from_slice(expected_shape);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn apply_prefix_layer<N>(
    index: usize,
    layer: &SealedLayer,
    domain: &mut ConstrainedZonotope64,
    shape: &mut Vec<usize>,
    relu_reduction_target: Option<usize>,
    limits: CganCzSequentialLimits,
    retained_baseline: usize,
    budget: ConstrainedZonotopeCallBudget,
    coordinator: &mut Coordinator<N>,
    stages: &mut Vec<CganCzStageTelemetry>,
) -> Result<(), CganCzProbeDecline>
where
    N: FnMut(&'static str) -> Instant,
{
    apply_prefix_layer_for_profile(
        CganCzImgSz32Profile::Nch1,
        index,
        layer,
        domain,
        shape,
        relu_reduction_target,
        None,
        limits,
        retained_baseline,
        budget,
        coordinator,
        stages,
    )
}

#[allow(clippy::too_many_arguments)]
fn apply_prefix_layer_for_profile<N>(
    profile: CganCzImgSz32Profile,
    index: usize,
    layer: &SealedLayer,
    domain: &mut ConstrainedZonotope64,
    shape: &mut Vec<usize>,
    relu_reduction_target: Option<usize>,
    auxiliary_bounds: Option<&CertifiedAuxiliaryBounds64>,
    limits: CganCzSequentialLimits,
    retained_baseline: usize,
    budget: ConstrainedZonotopeCallBudget,
    coordinator: &mut Coordinator<N>,
    stages: &mut Vec<CganCzStageTelemetry>,
) -> Result<(), CganCzProbeDecline>
where
    N: FnMut(&'static str) -> Instant,
{
    let node = EXPECTED_NODES[index].0;
    let expected_shape = expected_output_shape_for_profile(profile, index);
    if auxiliary_bounds.is_some()
        && (EXPECTED_NODES[index].1 != LayerType::ReLU || !matches!(layer, SealedLayer::Relu))
    {
        return Err(CganCzProbeDecline::Topology {
            message: format!("{node} received auxiliary ReLU bounds for a non-ReLU layer"),
        });
    }
    match layer {
        SealedLayer::Affine(parameters) => {
            let before_alpha = domain.alpha_dim();
            let before_nnz = generator_nonzeros(domain)?;
            let call_budget = domain_call_budget(domain, retained_baseline, budget, coordinator)?;
            let outcome = constrained_zonotope_affine_unwired_with_budget(
                domain,
                parameters.weights.view(),
                &parameters.bias,
                ConstrainedZonotopeAffineLimits {
                    max_input_value_count: limits.max_value_dim,
                    max_output_value_count: limits.max_value_dim,
                    max_alpha_dim: limits.max_transient_alpha_dim,
                    max_generator_nonzeros: limits.max_generator_nonzeros,
                    max_weight_elements: parameters.weights.len(),
                    max_matrix_visits: parameters.weights.len(),
                    max_interval_products: limits.max_interval_products_per_stage,
                    max_constraint_count: 0,
                    max_constraint_elements: 0,
                },
                call_budget,
            )
            .map_err(|error| match error {
                ny_mip::ConstrainedZonotopeAffineBudgetError::Budget(error) => error.into(),
                ny_mip::ConstrainedZonotopeAffineBudgetError::Transform(error) => {
                    CganCzProbeDecline::Transform {
                        node,
                        operation: "affine",
                        message: error.to_string(),
                    }
                }
            })?;
            let ((output, _plan), report) = outcome.into_parts();
            coordinator.absorb(report)?;
            *domain = output;
            *shape = expected_shape.to_vec();
            record_stage(
                stages,
                node,
                CganCzStageKind::Affine,
                expected_shape,
                before_alpha,
                before_nnz,
                domain,
                0,
                0,
                report,
            )?;
        }
        SealedLayer::Reshape(output_shape) => {
            let before_polls = coordinator.deadline_polls;
            let before_items = coordinator.charged_items;
            coordinator.checkpoint("cGAN Reshape admission")?;
            coordinator.charge(shape.len() + output_shape.len(), "cGAN Reshape shape seal")?;
            if domain.value_dim() != checked_product(output_shape, "cGAN Reshape output")? {
                return Err(CganCzProbeDecline::Topology {
                    message: format!(
                        "{node} target {output_shape:?} does not preserve {} values",
                        domain.value_dim()
                    ),
                });
            }
            *shape = output_shape.clone();
            coordinator.checkpoint("cGAN Reshape publication")?;
            stages.push(CganCzStageTelemetry {
                node,
                kind: CganCzStageKind::Reshape,
                output_shape: output_shape.clone(),
                input_alpha_dim: domain.alpha_dim(),
                output_alpha_dim: domain.alpha_dim(),
                input_generator_nonzeros: generator_nonzeros(domain)?,
                output_generator_nonzeros: generator_nonzeros(domain)?,
                unstable_coordinates: 0,
                discarded_generators: 0,
                peak_live_bytes: coordinator.peak_live_bytes,
                charged_items: coordinator.charged_items - before_items,
                deadline_polls: coordinator.deadline_polls - before_polls,
            });
        }
        SealedLayer::BatchNorm(parameters) => {
            let before_alpha = domain.alpha_dim();
            let before_nnz = generator_nonzeros(domain)?;
            let value_dim = checked_product(shape, "cGAN BatchNorm input shape")?;
            if value_dim != domain.value_dim() {
                return Err(CganCzProbeDecline::Topology {
                    message: format!(
                        "{node} input shape {shape:?} has {value_dim} values, domain has {}",
                        domain.value_dim()
                    ),
                });
            }
            let call_budget = domain_call_budget(domain, retained_baseline, budget, coordinator)?;
            let outcome = constrained_zonotope_batch_norm_unwired_with_budget(
                domain,
                ConstrainedZonotopeBatchNormSpec {
                    input_shape: shape,
                    channel_axis: 0,
                    gamma: &parameters.gamma,
                    beta: &parameters.beta,
                    mean: &parameters.mean,
                    variance: &parameters.variance,
                    epsilon: parameters.epsilon,
                    mode: ConstrainedZonotopeBatchNormMode::Inference,
                },
                ConstrainedZonotopeBatchNormLimits {
                    max_value_count: limits.max_value_dim,
                    max_rank: 3,
                    max_channel_count: 128,
                    max_alpha_dim: limits.max_transient_alpha_dim,
                    max_generator_nonzeros: limits.max_generator_nonzeros,
                    max_parameter_elements: parameters.gamma.len() * 4,
                    max_coordinate_visits: value_dim * 2,
                    max_generator_visits: limits.max_generator_nonzeros.saturating_mul(2),
                    max_interval_products: limits.max_interval_products_per_stage,
                    max_constraint_count: 0,
                    max_constraint_elements: 0,
                },
                call_budget,
            )
            .map_err(|error| match error {
                ConstrainedZonotopeBatchNormBudgetError::Budget(error) => error.into(),
                ConstrainedZonotopeBatchNormBudgetError::Transform(error) => {
                    CganCzProbeDecline::Transform {
                        node,
                        operation: "BatchNorm",
                        message: error.to_string(),
                    }
                }
            })?;
            let ((output, _plan), report) = outcome.into_parts();
            coordinator.absorb(report)?;
            *domain = output;
            *shape = expected_shape.to_vec();
            record_stage(
                stages,
                node,
                CganCzStageKind::BatchNorm,
                expected_shape,
                before_alpha,
                before_nnz,
                domain,
                0,
                0,
                report,
            )?;
        }
        SealedLayer::ConvTranspose2d(parameters) => {
            let input_shape = shape3(shape, node)?;
            let before_alpha = domain.alpha_dim();
            let before_nnz = generator_nonzeros(domain)?;
            let call_budget = domain_call_budget(domain, retained_baseline, budget, coordinator)?;
            let outcome = constrained_zonotope_conv_transpose2d_unwired_with_budget(
                domain,
                input_shape,
                parameters.weights.view(),
                &parameters.bias,
                parameters.spec,
                ConstrainedZonotopeConvTranspose2dLimits {
                    max_value_count: limits.max_value_dim,
                    max_alpha_dim: limits.max_transient_alpha_dim,
                    max_generator_nonzeros: limits.max_generator_nonzeros,
                    max_weight_elements: parameters.weights.len(),
                    max_kernel_visits: limits.max_interval_products_per_stage,
                    max_interval_products: limits.max_interval_products_per_stage,
                    max_constraint_count: 0,
                    max_constraint_elements: 0,
                },
                call_budget,
            )
            .map_err(|error| match error {
                ny_mip::ConstrainedZonotopeConvTranspose2dBudgetError::Budget(error) => {
                    error.into()
                }
                ny_mip::ConstrainedZonotopeConvTranspose2dBudgetError::Transform(error) => {
                    CganCzProbeDecline::Transform {
                        node,
                        operation: "ConvTranspose2d",
                        message: error.to_string(),
                    }
                }
            })?;
            let ((output, plan), report) = outcome.into_parts();
            if plan.output_shape != shape3(expected_shape, node)? {
                return Err(CganCzProbeDecline::Topology {
                    message: format!(
                        "{node} primitive produced {:?}, expected {expected_shape:?}",
                        plan.output_shape
                    ),
                });
            }
            coordinator.absorb(report)?;
            *domain = output;
            *shape = expected_shape.to_vec();
            record_stage(
                stages,
                node,
                CganCzStageKind::ConvTranspose2d,
                expected_shape,
                before_alpha,
                before_nnz,
                domain,
                0,
                0,
                report,
            )?;
        }
        SealedLayer::Conv2d(parameters) => {
            let input_shape = shape3(shape, node)?;
            let before_alpha = domain.alpha_dim();
            let before_nnz = generator_nonzeros(domain)?;
            let call_budget = domain_call_budget(domain, retained_baseline, budget, coordinator)?;
            let outcome = constrained_zonotope_conv2d_unwired_with_budget(
                domain,
                input_shape,
                parameters.weights.view(),
                &parameters.bias,
                parameters.spec,
                ConstrainedZonotopeConv2dLimits {
                    max_value_count: limits.max_value_dim,
                    max_alpha_dim: limits.max_transient_alpha_dim,
                    max_generator_nonzeros: limits.max_generator_nonzeros,
                    max_weight_elements: parameters.weights.len(),
                    max_kernel_visits: limits.max_interval_products_per_stage,
                    max_interval_products: limits.max_interval_products_per_stage,
                    max_constraint_count: 0,
                    max_constraint_elements: 0,
                },
                call_budget,
            )
            .map_err(|error| match error {
                ny_mip::ConstrainedZonotopeConv2dBudgetError::Budget(error) => error.into(),
                ny_mip::ConstrainedZonotopeConv2dBudgetError::Transform(error) => {
                    CganCzProbeDecline::Transform {
                        node,
                        operation: "Conv2d",
                        message: error.to_string(),
                    }
                }
            })?;
            let ((output, plan), report) = outcome.into_parts();
            if plan.output_shape != shape3(expected_shape, node)? {
                return Err(CganCzProbeDecline::Topology {
                    message: format!(
                        "{node} primitive produced {:?}, expected {expected_shape:?}",
                        plan.output_shape
                    ),
                });
            }
            coordinator.absorb(report)?;
            *domain = output;
            *shape = expected_shape.to_vec();
            record_stage(
                stages,
                node,
                CganCzStageKind::Conv2d,
                expected_shape,
                before_alpha,
                before_nnz,
                domain,
                0,
                0,
                report,
            )?;
        }
        SealedLayer::Relu => {
            let expected_values = checked_product(expected_shape, "cGAN authenticated ReLU shape")?;
            if shape.as_slice() != expected_shape || domain.value_dim() != expected_values {
                return Err(CganCzProbeDecline::Topology {
                    message: format!(
                        "{node} preactivation has shape {shape:?} and {} values; expected {expected_shape:?} and {expected_values}",
                        domain.value_dim()
                    ),
                });
            }
            if auxiliary_bounds.is_some_and(|bounds| bounds.value_dim() != expected_values) {
                return Err(CganCzProbeDecline::Topology {
                    message: format!(
                        "{node} auxiliary bounds have {} values; expected {expected_values} for {profile:?} shape {expected_shape:?}",
                        auxiliary_bounds.map_or(0, CertifiedAuxiliaryBounds64::value_dim)
                    ),
                });
            }
            apply_relu_and_reduce(
                node,
                expected_shape,
                domain,
                relu_reduction_target,
                auxiliary_bounds,
                limits,
                retained_baseline,
                budget,
                coordinator,
                stages,
            )?;
            *shape = expected_shape.to_vec();
        }
    }
    if shape.as_slice() != expected_shape
        || domain.value_dim() != checked_product(expected_shape, "cGAN stage shape")?
    {
        return Err(CganCzProbeDecline::Topology {
            message: format!(
                "{node} publication shape/domain mismatch: shape={shape:?}, values={}",
                domain.value_dim()
            ),
        });
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn apply_relu_and_reduce<N>(
    node: &'static str,
    output_shape: &'static [usize],
    domain: &mut ConstrainedZonotope64,
    reduction_target: Option<usize>,
    auxiliary_bounds: Option<&CertifiedAuxiliaryBounds64>,
    limits: CganCzSequentialLimits,
    retained_baseline: usize,
    budget: ConstrainedZonotopeCallBudget,
    coordinator: &mut Coordinator<N>,
    stages: &mut Vec<CganCzStageTelemetry>,
) -> Result<(), CganCzProbeDecline>
where
    N: FnMut(&'static str) -> Instant,
{
    let relu_output = apply_relu_only(
        node,
        output_shape,
        domain,
        auxiliary_bounds,
        limits,
        retained_baseline,
        budget,
        coordinator,
        stages,
    )?;
    let Some(reduction_target) = reduction_target else {
        validate_retained_discriminator_domain(&relu_output, limits)?;
        coordinator.checkpoint("cGAN discriminator-symbol retention publication")?;
        *domain = relu_output;
        return Ok(());
    };
    // Restore the ordinary runner's original live-range: the predecessor is
    // dropped before mutually exclusive order-reduction work begins.
    *domain = relu_output;
    let reduced = reduce_relu_output(
        node,
        output_shape,
        domain,
        reduction_target,
        limits,
        retained_baseline,
        budget,
        coordinator,
        stages,
    )?;
    *domain = reduced;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn apply_relu_and_reduce_retaining_predecessor<N>(
    node: &'static str,
    output_shape: &'static [usize],
    domain: &ConstrainedZonotope64,
    reduction_target: Option<usize>,
    auxiliary_bounds: Option<&CertifiedAuxiliaryBounds64>,
    limits: CganCzSequentialLimits,
    retained_baseline: usize,
    retained_predecessor_bytes: usize,
    budget: ConstrainedZonotopeCallBudget,
    coordinator: &mut Coordinator<N>,
    stages: &mut Vec<CganCzStageTelemetry>,
) -> Result<ConstrainedZonotope64, CganCzProbeDecline>
where
    N: FnMut(&'static str) -> Instant,
{
    let relu_output = apply_relu_only(
        node,
        output_shape,
        domain,
        auxiliary_bounds,
        limits,
        retained_baseline,
        budget,
        coordinator,
        stages,
    )?;
    let Some(reduction_target) = reduction_target else {
        validate_retained_discriminator_domain(&relu_output, limits)?;
        coordinator.checkpoint("cGAN discriminator-symbol retention publication")?;
        return Ok(relu_output);
    };
    let reduction_retained_baseline = retained_baseline
        .checked_add(retained_predecessor_bytes)
        .ok_or(CganCzProbeDecline::ResourceOverflow {
            operation: "cGAN retained pre-ReLU reduction baseline",
        })?;
    reduce_relu_output(
        node,
        output_shape,
        &relu_output,
        reduction_target,
        limits,
        reduction_retained_baseline,
        budget,
        coordinator,
        stages,
    )
}

#[allow(clippy::too_many_arguments)]
fn apply_relu_only<N>(
    node: &'static str,
    output_shape: &'static [usize],
    domain: &ConstrainedZonotope64,
    auxiliary_bounds: Option<&CertifiedAuxiliaryBounds64>,
    limits: CganCzSequentialLimits,
    retained_baseline: usize,
    budget: ConstrainedZonotopeCallBudget,
    coordinator: &mut Coordinator<N>,
    stages: &mut Vec<CganCzStageTelemetry>,
) -> Result<ConstrainedZonotope64, CganCzProbeDecline>
where
    N: FnMut(&'static str) -> Instant,
{
    let before_alpha = domain.alpha_dim();
    let before_nnz = generator_nonzeros(domain)?;
    let call_budget = domain_call_budget(domain, retained_baseline, budget, coordinator)?;
    let relu_limits = ReluTransformLimits {
        max_value_dim: limits.max_value_dim,
        max_output_alpha_dim: limits.max_transient_alpha_dim,
        max_constraints: 0,
        max_constraint_elements: 0,
        max_generator_nnz: limits.max_generator_nonzeros,
        max_unstable: limits.max_transient_alpha_dim,
        max_exact_terms: limits.max_exact_terms_per_relu,
    };
    // The authenticated auxiliary path is a single fail-closed attempt. A
    // refusal never retries the legacy transform after partially charged work.
    let outcome = match auxiliary_bounds {
        Some(auxiliary) => transform_relu_with_auxiliary_bounds_unwired_with_budget(
            domain,
            auxiliary,
            relu_limits,
            call_budget,
        ),
        None => transform_relu_unwired_with_budget(domain, relu_limits, call_budget),
    }
    .map_err(|error| match error {
        ny_mip::ReluTransformBudgetError::Budget(error) => error.into(),
        ny_mip::ReluTransformBudgetError::Transform(error) => CganCzProbeDecline::Transform {
            node,
            operation: "ReLU",
            message: error.to_string(),
        },
    })?;
    let (relu_output, report) = outcome.into_parts();
    let unstable = relu_output.alpha_dim().checked_sub(before_alpha).ok_or(
        CganCzProbeDecline::ResourceOverflow {
            operation: "cGAN ReLU unstable count",
        },
    )?;
    coordinator.absorb(report)?;
    record_stage(
        stages,
        node,
        CganCzStageKind::Relu,
        output_shape,
        before_alpha,
        before_nnz,
        &relu_output,
        unstable,
        0,
        report,
    )?;
    Ok(relu_output)
}

#[allow(clippy::too_many_arguments)]
fn reduce_relu_output<N>(
    node: &'static str,
    output_shape: &'static [usize],
    relu_output: &ConstrainedZonotope64,
    reduction_target: usize,
    limits: CganCzSequentialLimits,
    retained_baseline: usize,
    budget: ConstrainedZonotopeCallBudget,
    coordinator: &mut Coordinator<N>,
    stages: &mut Vec<CganCzStageTelemetry>,
) -> Result<ConstrainedZonotope64, CganCzProbeDecline>
where
    N: FnMut(&'static str) -> Instant,
{
    let reduce_input_alpha = relu_output.alpha_dim();
    let reduce_input_nnz = generator_nonzeros(relu_output)?;
    let call_budget = domain_call_budget(relu_output, retained_baseline, budget, coordinator)?;
    let outcome = constrained_zonotope_order_reduce_unwired_with_budget(
        relu_output,
        reduction_target,
        PROTECTED_LATENT_SYMBOLS,
        ConstrainedZonotopeOrderReductionLimits {
            max_value_dim: limits.max_value_dim,
            max_input_alpha_dim: limits.max_transient_alpha_dim,
            max_output_alpha_dim: reduction_target,
            max_constraints: 0,
            max_constraint_elements: 0,
            max_generator_nnz: limits.max_generator_nonzeros,
        },
        call_budget,
    )
    .map_err(|error| match error {
        ny_mip::ConstrainedZonotopeOrderReductionBudgetError::Budget(error) => error.into(),
        ny_mip::ConstrainedZonotopeOrderReductionBudgetError::Transform(error) => {
            CganCzProbeDecline::Transform {
                node,
                operation: "order reduction",
                message: error.to_string(),
            }
        }
    })?;
    let ((reduced, plan), report) = outcome.into_parts();
    coordinator.absorb(report)?;
    record_stage(
        stages,
        node,
        CganCzStageKind::OrderReduction,
        output_shape,
        reduce_input_alpha,
        reduce_input_nnz,
        &reduced,
        0,
        plan.input_alpha_dim() - plan.output_alpha_dim(),
        report,
    )?;
    Ok(reduced)
}

fn validate_retained_discriminator_domain(
    domain: &ConstrainedZonotope64,
    limits: CganCzSequentialLimits,
) -> Result<(), CganCzProbeDecline> {
    check_resource(
        "retained discriminator alpha dimension",
        domain.alpha_dim(),
        limits.max_transient_alpha_dim,
    )?;
    check_resource(
        "retained discriminator generator nonzeros",
        generator_nonzeros(domain)?,
        limits.max_generator_nonzeros,
    )
}

#[allow(clippy::too_many_arguments)]
fn record_stage(
    stages: &mut Vec<CganCzStageTelemetry>,
    node: &'static str,
    kind: CganCzStageKind,
    output_shape: &[usize],
    input_alpha_dim: usize,
    input_generator_nonzeros: usize,
    output: &ConstrainedZonotope64,
    unstable_coordinates: usize,
    discarded_generators: usize,
    report: ConstrainedZonotopeCallReport,
) -> Result<(), CganCzProbeDecline> {
    if stages.len() >= MAX_RUNNER_STAGES {
        return Err(CganCzProbeDecline::ResourceLimit {
            resource: "cGAN stage telemetry count",
            required: stages.len() + 1,
            limit: MAX_RUNNER_STAGES,
        });
    }
    stages.push(CganCzStageTelemetry {
        node,
        kind,
        output_shape: output_shape.to_vec(),
        input_alpha_dim,
        output_alpha_dim: output.alpha_dim(),
        input_generator_nonzeros,
        output_generator_nonzeros: generator_nonzeros(output)?,
        unstable_coordinates,
        discarded_generators,
        peak_live_bytes: report.peak_live_bytes(),
        charged_items: report.charged_items(),
        deadline_polls: report.deadline_polls(),
    });
    Ok(())
}

/// Record a terminal scalar computation that consumes a CZ but does not
/// publish another CZ. Zero output correlation fields mean N/A here; copying
/// the pre-ReLU domain into those fields would falsely describe a scalar M17
/// result as a 512-coordinate correlated domain. The caller supplies composite
/// counters so exact objective construction and nested primitive work share
/// one complete receipt without double-counting.
#[allow(clippy::too_many_arguments)]
fn record_scalar_tail_stage_accounting(
    stages: &mut Vec<CganCzStageTelemetry>,
    node: &'static str,
    kind: CganCzStageKind,
    input_alpha_dim: usize,
    input_generator_nonzeros: usize,
    unstable_coordinates: usize,
    peak_live_bytes: usize,
    charged_items: usize,
    deadline_polls: usize,
) -> Result<(), CganCzProbeDecline> {
    if stages.len() >= MAX_RUNNER_STAGES {
        return Err(CganCzProbeDecline::ResourceLimit {
            resource: "cGAN stage telemetry count",
            required: stages.len() + 1,
            limit: MAX_RUNNER_STAGES,
        });
    }
    stages.push(CganCzStageTelemetry {
        node,
        kind,
        output_shape: expected_output_shape(25).to_vec(),
        input_alpha_dim,
        output_alpha_dim: 0,
        input_generator_nonzeros,
        output_generator_nonzeros: 0,
        unstable_coordinates,
        discarded_generators: 0,
        peak_live_bytes,
        charged_items,
        deadline_polls,
    });
    Ok(())
}

fn domain_call_budget<N>(
    domain: &ConstrainedZonotope64,
    retained_baseline: usize,
    budget: ConstrainedZonotopeCallBudget,
    coordinator: &mut Coordinator<N>,
) -> Result<ConstrainedZonotopeCallBudget, CganCzProbeDecline>
where
    N: FnMut(&'static str) -> Instant,
{
    let baseline = retained_baseline
        .checked_add(domain_live_bytes(domain)?)
        .ok_or(CganCzProbeDecline::ResourceOverflow {
            operation: "cGAN transform baseline",
        })?;
    coordinator.preflight_absolute_peak(baseline)?;
    Ok(ConstrainedZonotopeCallBudget::new(
        budget.deadline(),
        baseline,
        budget.max_peak_live_bytes(),
    ))
}

fn domain_live_bytes(domain: &ConstrainedZonotope64) -> Result<usize, CganCzProbeDecline> {
    let generator_entries = generator_nonzeros(domain)?;
    let generator_bytes = domain
        .alpha_dim()
        .checked_mul(size_of::<Vec<(usize, f64)>>())
        .and_then(|bytes| {
            bytes.checked_add(generator_entries.saturating_mul(size_of::<(usize, f64)>()))
        })
        .ok_or(CganCzProbeDecline::ResourceOverflow {
            operation: "cGAN generator live bytes",
        })?;
    let f64_values = domain
        .center()
        .len()
        .checked_add(domain.box_remainder().len())
        .and_then(|count| count.checked_add(domain.constraints().len()))
        .and_then(|count| count.checked_add(domain.rhs().len()))
        .ok_or(CganCzProbeDecline::ResourceOverflow {
            operation: "cGAN domain f64 values",
        })?;
    generator_bytes
        .checked_add(f64_values.checked_mul(size_of::<f64>()).ok_or(
            CganCzProbeDecline::ResourceOverflow {
                operation: "cGAN domain f64 bytes",
            },
        )?)
        .ok_or(CganCzProbeDecline::ResourceOverflow {
            operation: "cGAN domain live bytes",
        })
}

fn generator_nonzeros(domain: &ConstrainedZonotope64) -> Result<usize, CganCzProbeDecline> {
    domain
        .generators()
        .iter()
        .try_fold(0_usize, |count, generator| {
            count
                .checked_add(generator.nnz())
                .ok_or(CganCzProbeDecline::ResourceOverflow {
                    operation: "cGAN generator nonzeros",
                })
        })
}

fn checked_product(values: &[usize], operation: &'static str) -> Result<usize, CganCzProbeDecline> {
    values.iter().try_fold(1_usize, |product, &value| {
        product
            .checked_mul(value)
            .ok_or(CganCzProbeDecline::ResourceOverflow { operation })
    })
}

fn shape3(shape: &[usize], node: &'static str) -> Result<[usize; 3], CganCzProbeDecline> {
    <[usize; 3]>::try_from(shape).map_err(|_| CganCzProbeDecline::Topology {
        message: format!("{node} requires a rank-3 unbatched NCHW shape, got {shape:?}"),
    })
}

#[cfg(test)]
mod tests {
    use std::cell::{Cell, RefCell};
    use std::io::Write;
    #[cfg(feature = "external-vnncomp")]
    use std::path::{Path, PathBuf};

    use super::*;
    #[cfg(feature = "external-vnncomp")]
    use ndarray::Array1;
    use ny_onnx::vnnlib::load_vnnlib_with_certified_scalar_moat;
    // Gated to match its usage: every call site below sits behind
    // `external-vnncomp`, like the sibling imports in this block.
    #[cfg(feature = "external-vnncomp")]
    use ny_onnx::{load_onnx_with_config, BatchNormFoldingPolicy, OnnxLoadConfig};
    #[cfg(feature = "external-vnncomp")]
    use ny_propagate::{types::BoundsProvenance, GraphNode};
    #[cfg(feature = "external-vnncomp")]
    use ny_tensor::{cast_f64_to_f32_down, cast_f64_to_f32_up, BoundedTensor};

    fn qualification_limits() -> CganCzSequentialLimits {
        CganCzSequentialLimits {
            max_graph_nodes: 26,
            max_graph_edges: 128,
            max_topology_work_items: 1 << 20,
            max_parameter_elements: 600_000,
            max_value_dim: 12_544,
            max_transient_alpha_dim: 12_549,
            retained_alpha_dim: 5,
            max_generator_nonzeros: 100_000,
            max_interval_products_per_stage: 32_000_000,
            max_exact_terms_per_relu: 1_000_000,
            max_m17_iterations: 8,
            max_m17_search_work: 10_000_000,
        }
    }

    fn protected_cover_fixture() -> ConstrainedZonotope64 {
        ConstrainedZonotope64::try_new(
            vec![0.0; PROTECTED_LATENT_SYMBOLS],
            (0..PROTECTED_LATENT_SYMBOLS)
                .map(|axis| vec![(axis, 1.0)])
                .collect(),
            Array2::from_shape_vec(
                (2, PROTECTED_LATENT_SYMBOLS),
                vec![1.0, 2.0, 0.0, -1.0, 0.5, -2.0, 0.0, 1.0, 0.0, 3.0],
            )
            .unwrap(),
            vec![1.25, 2.5],
            vec![0.125; PROTECTED_LATENT_SYMBOLS],
        )
        .unwrap()
    }

    fn protected_cover_limits() -> CganCzProtectedAlphaCoverLimits {
        CganCzProtectedAlphaCoverLimits {
            protected_alpha_dim: PROTECTED_LATENT_SYMBOLS,
            max_split_levels: PROTECTED_LATENT_SYMBOLS,
            max_tree_nodes: 63,
            max_leaf_domains: 32,
            bisection: ConstrainedZonotopeAlphaBisectionLimits {
                max_value_dim: PROTECTED_LATENT_SYMBOLS,
                max_alpha_dim: PROTECTED_LATENT_SYMBOLS,
                max_generator_nonzeros: PROTECTED_LATENT_SYMBOLS,
                max_constraint_count: 2,
                max_constraint_elements: 2 * PROTECTED_LATENT_SYMBOLS,
            },
        }
    }

    fn protected_cover_budget(start: Instant) -> ConstrainedZonotopeCallBudget {
        ConstrainedZonotopeCallBudget::new(start + Duration::from_mins(1), 0, 1 << 30)
    }

    fn certified_input_and_moat() -> (CertifiedInputBox, CertifiedScalarMoat) {
        let mut vnnlib = tempfile::NamedTempFile::new().unwrap();
        for axis in 0..PROTECTED_LATENT_SYMBOLS {
            writeln!(vnnlib, "(declare-const X_{axis} Real)").unwrap();
        }
        writeln!(vnnlib, "(declare-const Y_0 Real)").unwrap();
        for axis in 0..PROTECTED_LATENT_SYMBOLS {
            writeln!(vnnlib, "(assert (>= X_{axis} -1))").unwrap();
            writeln!(vnnlib, "(assert (<= X_{axis} 1))").unwrap();
        }
        writeln!(vnnlib, "(assert (or (and (>= Y_0 1)) (and (<= Y_0 -1))))").unwrap();
        let (_, input, moat) = load_vnnlib_with_certified_scalar_moat(vnnlib.path()).unwrap();
        (input, moat)
    }

    fn certified_mixed_point_input() -> CertifiedInputBox {
        let mut vnnlib = tempfile::NamedTempFile::new().unwrap();
        for axis in 0..PROTECTED_LATENT_SYMBOLS {
            writeln!(vnnlib, "(declare-const X_{axis} Real)").unwrap();
        }
        writeln!(vnnlib, "(declare-const Y_0 Real)").unwrap();
        for axis in 0..PROTECTED_LATENT_SYMBOLS {
            if axis == 2 {
                // A non-dyadic declared point retains its outward endpoint
                // moat even though it creates no independent alpha symbol.
                writeln!(vnnlib, "(assert (>= X_{axis} -0.3082364797592163))").unwrap();
                writeln!(vnnlib, "(assert (<= X_{axis} -0.3082364797592163))").unwrap();
            } else {
                writeln!(vnnlib, "(assert (>= X_{axis} -1))").unwrap();
                writeln!(vnnlib, "(assert (<= X_{axis} 1))").unwrap();
            }
        }
        writeln!(vnnlib, "(assert (or (and (>= Y_0 1)) (and (<= Y_0 -1))))").unwrap();
        let (_, input, _) = load_vnnlib_with_certified_scalar_moat(vnnlib.path()).unwrap();
        input
    }

    fn synthetic_completed_leaf_bounds(
        leaf_index: usize,
        moat: CertifiedScalarMoat,
    ) -> CganCzCompletedBounds {
        let lower_bound = -0.75 + leaf_index as f64 / 2_000.0;
        let upper_bound = 0.75 + leaf_index as f64 / 1_000.0;
        let lower_m17_candidates = CganCzM17CandidateTelemetry {
            selected_lower_bound: lower_bound,
            zero_positive_slope_lower_bound: lower_bound,
            upper_endpoint_lower_bound: None,
            canonical_lower_bound: None,
            optimized_lower_bound: None,
            best_nonoptimized_lower_bound: lower_bound,
            optimized_improvement: 0.0,
            optimizable_slopes: 0,
            candidates_replayed: 1,
            iterations_completed: 0,
            status: ReluTailDualStatus::NoOptimizableSlopes,
        };
        let negated_upper_m17_candidates = CganCzM17CandidateTelemetry {
            selected_lower_bound: -upper_bound,
            zero_positive_slope_lower_bound: -upper_bound,
            upper_endpoint_lower_bound: None,
            canonical_lower_bound: None,
            optimized_lower_bound: None,
            best_nonoptimized_lower_bound: -upper_bound,
            optimized_improvement: 0.0,
            optimizable_slopes: 0,
            candidates_replayed: 1,
            iterations_completed: 0,
            status: ReluTailDualStatus::NoOptimizableSlopes,
        };
        CganCzCompletedBounds {
            lower_bound,
            upper_bound,
            low_unsafe_threshold: moat.low_upper(),
            high_unsafe_threshold: moat.high_lower(),
            separates_unsafe_moat: tail_bounds_separate_unsafe_moat(
                lower_bound,
                upper_bound,
                moat.low_upper(),
                moat.high_lower(),
            ),
            bn_tail_correction_upper: leaf_index as f64 * f64::EPSILON,
            lower_m17_status: ReluTailDualStatus::NoOptimizableSlopes,
            upper_m17_status: ReluTailDualStatus::NoOptimizableSlopes,
            lower_m17_candidates,
            negated_upper_m17_candidates,
            lower_m20_lower_bound: None,
            negated_upper_m20_lower_bound: None,
            lower_m20_status: CganCzM20Status::NotRequested,
            negated_upper_m20_status: CganCzM20Status::NotRequested,
            lower_m24_measurement: None,
            negated_upper_m24_measurement: None,
        }
    }

    fn synthetic_output_tail_result(completed: CganCzCompletedBounds) -> CganCzOutputTailResult {
        CganCzOutputTailResult {
            completed,
            lower_depth_two_measurement: CganCzDepthTwoMeasurement::NotRequested,
            negated_upper_depth_two_measurement: CganCzDepthTwoMeasurement::NotRequested,
        }
    }

    fn synthetic_m24_measurement(lower_bound: f64) -> CganCzM24Measurement {
        CganCzM24Measurement {
            exact_box_cut_lower_bound: Some(lower_bound),
            counterfactual_lower_bound: lower_bound,
            counterfactual_selection: ReluTailBoxCutSelection::BoxCut,
            replay_status: ReluTailBoxCutStatus::Completed,
            search_status: ReluTailBoxCutOptimizerStatus::Completed,
            search_plan: Some(ReluTailBoxCutOptimizerPlan {
                value_dim: TAIL_VALUE_DIM,
                alpha_dim: TAIL_DISCRIMINATOR_RETAINED_ALPHA_DIM,
                generator_nonzeros: TAIL_M24_MAX_GENERATOR_NONZEROS,
                box_variables: TAIL_M24_MAX_BOX_VARIABLES,
                restarts: TAIL_M24_MAX_RESTARTS,
                total_iterations: TAIL_M24_MAX_TOTAL_ITERATIONS,
                exact_replays: TAIL_M24_MAX_EXACT_REPLAYS,
                search_work: TAIL_M24_MAX_SEARCH_WORK,
            }),
            iterations_completed: TAIL_M24_MAX_TOTAL_ITERATIONS,
            restarts_completed: TAIL_M24_MAX_RESTARTS,
            candidates_scored: TAIL_M24_MAX_TOTAL_ITERATIONS + TAIL_M24_MAX_RESTARTS,
            exact_replays: TAIL_M24_MAX_EXACT_REPLAYS,
            optional_budget_error: None,
        }
    }

    #[test]
    fn leaf_row_final_publication_is_transactional_at_deadline() {
        let (_, moat) = certified_input_and_moat();
        let start = Instant::now();
        let deadline = start + Duration::from_secs(1);
        let mut on_time = Coordinator::new(
            ConstrainedZonotopeCallBudget::new(deadline, 0, 1 << 20),
            move |_| start,
        );
        let rows = publish_cgan_leaf_rows(
            synthetic_output_tail_result(synthetic_completed_leaf_bounds(0, moat)),
            &mut on_time,
        )
        .unwrap();
        assert_eq!(rows.lower_y, -0.75);
        assert_eq!(rows.lower_neg_y, -0.75);

        let mut late = Coordinator::new(
            ConstrainedZonotopeCallBudget::new(deadline, 0, 1 << 20),
            move |_| deadline,
        );
        assert!(matches!(
            publish_cgan_leaf_rows(
                synthetic_output_tail_result(synthetic_completed_leaf_bounds(0, moat)),
                &mut late,
            ),
            Err(CganCzProbeDecline::Budget(
                ConstrainedZonotopeCallBudgetError::DeadlineExpired {
                    checkpoint: "cGAN leaf-row publication"
                }
            ))
        ));
    }

    #[test]
    fn m24_measurement_does_not_change_historical_leaf_authority() {
        let (_, moat) = certified_input_and_moat();
        let mut completed = synthetic_completed_leaf_bounds(0, moat);
        completed.lower_m24_measurement = Some(synthetic_m24_measurement(100.0));
        completed.negated_upper_m24_measurement = Some(synthetic_m24_measurement(200.0));
        let start = Instant::now();
        let mut coordinator = Coordinator::new(
            ConstrainedZonotopeCallBudget::new(start + Duration::from_secs(1), 0, 1 << 20),
            move |_| start,
        );
        let rows =
            publish_cgan_leaf_rows(synthetic_output_tail_result(completed), &mut coordinator)
                .unwrap();
        assert_eq!(rows.lower_y, -0.75);
        assert_eq!(rows.lower_neg_y, -0.75);
        assert_eq!(
            rows.lower_m24_measurement
                .as_ref()
                .unwrap()
                .counterfactual_lower_bound,
            100.0
        );
        assert_eq!(
            rows.negated_upper_m24_measurement
                .as_ref()
                .unwrap()
                .counterfactual_lower_bound,
            200.0
        );
    }

    #[test]
    fn cgan_m24_fixed_config_seals_worst_case_work() {
        let config = cgan_m24_config();
        assert_eq!(config.schedules[0].iterations, 4);
        assert_eq!(
            config.schedules[0].learning_rate.to_bits(),
            0.005_f64.to_bits()
        );
        assert_eq!(config.schedules[1].iterations, 4);
        assert_eq!(
            config.schedules[1].learning_rate.to_bits(),
            0.1_f64.to_bits()
        );
        assert!(config
            .schedules
            .iter()
            .all(|schedule| schedule.decay.to_bits() == 0.98_f64.to_bits()));
        assert_eq!(config.multiplier_cap.to_bits(), 16.0_f64.to_bits());
        assert_eq!(config.wall_time, Duration::from_secs(1));
        assert_eq!(config.limits.max_value_dim, 512);
        assert_eq!(config.limits.max_box_variables, 1_024);
        assert_eq!(config.limits.max_total_iterations, 8);
        assert_eq!(config.limits.max_restarts, 2);
        assert_eq!(config.limits.max_exact_replays, 2);
        assert_eq!(config.limits.max_generator_nonzeros, 150_000);
        assert_eq!(config.limits.max_search_work, 3_104_960);
        assert_eq!(config.limits.max_wall_time, Duration::from_secs(1));

        let value_dim = 512_u64;
        let alpha_dim = 512_u64;
        let box_variables = 1_024_u64;
        let generator_nonzeros = 150_000_u64;
        let score = 3 * value_dim + 2 * box_variables + 2 * generator_nonzeros + alpha_dim;
        let restart_startup = 6 * box_variables + 2 * value_dim + score;
        let per_iteration = 6 * box_variables + score;
        let exact_worst_case_work = value_dim + 2 * restart_startup + 8 * per_iteration;
        assert_eq!(exact_worst_case_work, TAIL_M24_MAX_SEARCH_WORK);
        let max_plan = synthetic_m24_measurement(0.0).search_plan.unwrap();
        assert_eq!(
            cgan_m24_plan_search_work(max_plan).unwrap(),
            TAIL_M24_MAX_SEARCH_WORK
        );
    }

    #[test]
    fn m17_m20_selector_is_strict_stable_and_rejects_malformed_receipts() {
        let tied = select_m17_m20_lower_bound(-0.0, Some(0.0), CganCzM20Status::Completed).unwrap();
        assert_eq!(tied.to_bits(), (-0.0_f64).to_bits());
        assert_eq!(
            select_m17_m20_lower_bound(0.25, Some(0.5), CganCzM20Status::Completed),
            Some(0.5)
        );
        assert_eq!(
            select_m17_m20_lower_bound(0.25, None, CganCzM20Status::Fallback),
            Some(0.25)
        );
        assert_eq!(
            select_m17_m20_lower_bound(0.25, Some(0.5), CganCzM20Status::Fallback),
            None
        );
        assert_eq!(
            select_m17_m20_lower_bound(0.25, Some(f64::NAN), CganCzM20Status::Completed),
            None
        );
        assert_eq!(
            select_m17_m20_lower_bound(0.25, None, CganCzM20Status::Completed),
            None
        );
        assert_eq!(
            select_m17_m20_lower_bound(0.25, Some(0.5), CganCzM20Status::NotRequested),
            None
        );
        assert_eq!(
            select_m17_m20_lower_bound(f64::INFINITY, None, CganCzM20Status::Fallback),
            None
        );
    }

    #[test]
    fn m17_candidate_telemetry_isolates_replayed_optimizer_gain() {
        let improved = summarize_m17_candidate_values(
            1.25,
            0.5,
            Some(0.75),
            Some(1.0),
            Some(1.25),
            17,
            4,
            24,
            ReluTailDualStatus::Completed,
        );
        assert_eq!(improved.best_nonoptimized_lower_bound, 1.0);
        assert_eq!(improved.optimized_improvement, 0.25);
        assert_eq!(improved.selected_lower_bound, 1.25);
        assert_eq!(improved.iterations_completed, 24);

        let no_gain = summarize_m17_candidate_values(
            1.0,
            0.5,
            Some(1.0),
            Some(0.75),
            Some(0.875),
            17,
            4,
            24,
            ReluTailDualStatus::Completed,
        );
        assert_eq!(no_gain.best_nonoptimized_lower_bound, 1.0);
        assert_eq!(no_gain.optimized_improvement, 0.0);
    }

    #[test]
    fn protected_latent_cover_publishes_all_32_orthants_in_alpha_order() {
        let input = protected_cover_fixture();
        let start = Instant::now();
        let cover = enumerate_cgan_nch1_protected_latent_cover_unwired(
            &input,
            protected_cover_limits(),
            protected_cover_budget(start),
        )
        .unwrap();

        assert_eq!(cover.split_axes(), &CGAN_NCH1_PROTECTED_LATENT_COVER_AXES);
        assert_eq!(cover.leaves().len(), 1 << PROTECTED_LATENT_SYMBOLS);
        assert_eq!(cover.report().split_levels(), 5);
        assert_eq!(cover.report().tree_nodes(), 63);
        assert_eq!(cover.report().split_calls(), 31);
        assert_eq!(cover.report().leaf_domains(), 32);
        assert!(cover.report().peak_live_bytes() > 0);
        assert!(cover.report().charged_items() >= 32);
        assert!(cover.report().deadline_polls() > 0);

        let original_constraints = input.constraints();
        for (leaf_index, leaf) in cover.leaves().iter().enumerate() {
            assert_eq!(leaf.alpha_dim(), PROTECTED_LATENT_SYMBOLS);
            assert_eq!(leaf.constraint_count(), input.constraint_count());
            assert_eq!(leaf.box_remainder(), input.box_remainder());
            for axis in 0..PROTECTED_LATENT_SYMBOLS {
                let positive = ((leaf_index >> (PROTECTED_LATENT_SYMBOLS - axis - 1)) & 1) != 0;
                assert_eq!(leaf.center()[axis], if positive { 0.5 } else { -0.5 });
                assert_eq!(
                    leaf.generators()[axis].entries().collect::<Vec<_>>(),
                    vec![(axis, 0.5)]
                );
                for row in 0..input.constraint_count() {
                    assert_eq!(
                        leaf.constraints()[(row, axis)],
                        original_constraints[(row, axis)] * 0.5
                    );
                }
            }

            for row in 0..input.constraint_count() {
                let shifted = (0..PROTECTED_LATENT_SYMBOLS).fold(input.rhs()[row], |rhs, axis| {
                    let sigma = if ((leaf_index >> (PROTECTED_LATENT_SYMBOLS - axis - 1)) & 1) != 0
                    {
                        1.0
                    } else {
                        -1.0
                    };
                    rhs - sigma * original_constraints[(row, axis)] * 0.5
                });
                assert_eq!(leaf.rhs()[row], shifted);
            }
        }
    }

    #[test]
    fn complete_cover_propagation_visits_all_leaves_before_combining_in_order() {
        let input = protected_cover_fixture();
        let start = Instant::now();
        let cover = enumerate_cgan_nch1_protected_latent_cover_unwired(
            &input,
            protected_cover_limits(),
            protected_cover_budget(start),
        )
        .unwrap();
        let (_, moat) = certified_input_and_moat();
        let budget = protected_cover_budget(start);
        let mut coordinator = Coordinator::new(budget, move |_| start);
        let observed_centers = RefCell::new(Vec::new());
        let aggregate = propagate_cgan_cz_complete_cover_with(
            cover,
            &CGAN_NCH1_PROTECTED_LATENT_COVER_AXES,
            moat,
            PROTECTED_LATENT_LEAF_DOMAINS,
            0,
            &mut coordinator,
            |leaf_index, leaf, _, _| {
                observed_centers.borrow_mut().push(leaf.center().to_vec());
                Ok((
                    synthetic_completed_leaf_bounds(leaf_index, moat),
                    FULL_RUNNER_COMPLETED_STAGES,
                ))
            },
        )
        .unwrap();

        assert_eq!(aggregate.leaf_completions.len(), 32);
        assert_eq!(aggregate.cover.leaf_domains(), 32);
        assert_eq!(aggregate.lower_bound, -0.75);
        assert_eq!(aggregate.upper_bound, 0.781);
        assert_eq!(
            aggregate.separates_unsafe_moat,
            tail_bounds_separate_unsafe_moat(
                aggregate.lower_bound,
                aggregate.upper_bound,
                moat.low_upper(),
                moat.high_lower(),
            )
        );
        assert_eq!(observed_centers.borrow().len(), 32);
        for (leaf_index, completion) in aggregate.leaf_completions.iter().enumerate() {
            assert_eq!(completion.leaf_index, leaf_index);
            assert_eq!(completion.completed_stages, FULL_RUNNER_COMPLETED_STAGES);
            assert!(completion.deadline_polls > 0);
            let center = &observed_centers.borrow()[leaf_index];
            for (axis, &coordinate) in center.iter().enumerate() {
                let positive = ((leaf_index >> (PROTECTED_LATENT_SYMBOLS - axis - 1)) & 1) != 0;
                assert_eq!(coordinate, if positive { 0.5 } else { -0.5 });
            }
        }
    }

    #[test]
    fn complete_cover_propagation_deadline_and_cap_publish_no_partial_aggregate() {
        let input = protected_cover_fixture();
        let start = Instant::now();
        let (_, moat) = certified_input_and_moat();
        let cover = enumerate_cgan_nch1_protected_latent_cover_unwired(
            &input,
            protected_cover_limits(),
            protected_cover_budget(start),
        )
        .unwrap();
        let deadline = start + Duration::from_mins(1);
        let dispatches = Cell::new(0_usize);
        let propagated = Cell::new(0_usize);
        let mut coordinator = Coordinator::new(
            ConstrainedZonotopeCallBudget::new(deadline, 0, 1 << 30),
            |checkpoint| {
                if checkpoint == "protected-alpha leaf propagation dispatch" {
                    let next = dispatches.get() + 1;
                    dispatches.set(next);
                    if next == 8 {
                        return deadline;
                    }
                }
                start
            },
        );
        let result = propagate_cgan_cz_complete_cover_with(
            cover,
            &CGAN_NCH1_PROTECTED_LATENT_COVER_AXES,
            moat,
            PROTECTED_LATENT_LEAF_DOMAINS,
            0,
            &mut coordinator,
            |leaf_index, _, _, _| {
                propagated.set(propagated.get() + 1);
                Ok((
                    synthetic_completed_leaf_bounds(leaf_index, moat),
                    FULL_RUNNER_COMPLETED_STAGES,
                ))
            },
        );
        assert!(matches!(
            result,
            Err(ProtectedAlphaLeafDecline {
                leaf_index: Some(7),
                reason: CganCzProbeDecline::Budget(
                    ConstrainedZonotopeCallBudgetError::DeadlineExpired {
                        checkpoint: "protected-alpha leaf propagation dispatch"
                    }
                ),
                ..
            })
        ));
        assert_eq!(dispatches.get(), 8);
        assert_eq!(propagated.get(), 7);

        let cover = enumerate_cgan_nch1_protected_latent_cover_unwired(
            &input,
            protected_cover_limits(),
            protected_cover_budget(start),
        )
        .unwrap();
        let callback_reached = Cell::new(false);
        let mut coordinator = Coordinator::new(protected_cover_budget(start), move |_| start);
        let capped = propagate_cgan_cz_complete_cover_with(
            cover,
            &CGAN_NCH1_PROTECTED_LATENT_COVER_AXES,
            moat,
            PROTECTED_LATENT_LEAF_DOMAINS - 1,
            0,
            &mut coordinator,
            |leaf_index, _, _, _| {
                callback_reached.set(true);
                Ok((
                    synthetic_completed_leaf_bounds(leaf_index, moat),
                    FULL_RUNNER_COMPLETED_STAGES,
                ))
            },
        );
        assert!(matches!(
            capped,
            Err(ProtectedAlphaLeafDecline {
                leaf_index: None,
                reason: CganCzProbeDecline::ResourceLimit {
                    resource: "protected-alpha complete leaf propagations",
                    required: 32,
                    limit: 31,
                },
                ..
            })
        ));
        assert!(!callback_reached.get());
    }

    #[test]
    fn cgan_input_domain_seam_reaches_complete_protected_cover() {
        let (input, moat) = certified_input_and_moat();

        let sealed = SealedCgan {
            layers: Vec::new(),
            parameter_elements: 0,
            live_bytes: 0,
        };
        let start = Instant::now();
        let budget = protected_cover_budget(start);
        let mut coordinator = Coordinator::new(budget, move |_| start);
        let domain =
            build_input_domain(&input, qualification_limits(), &sealed, 0, &mut coordinator)
                .unwrap();
        assert_eq!(domain.alpha_dim(), PROTECTED_LATENT_SYMBOLS);

        let mut limits = protected_cover_limits();
        limits.bisection.max_constraint_count = 0;
        limits.bisection.max_constraint_elements = 0;
        let cover = enumerate_cgan_nch1_protected_latent_cover_unwired(
            &domain,
            limits,
            protected_cover_budget(Instant::now()),
        )
        .unwrap();
        assert_eq!(cover.split_axes(), &CGAN_NCH1_PROTECTED_LATENT_COVER_AXES);
        assert_eq!(cover.leaves().len(), 32);
        assert!(cover
            .leaves()
            .iter()
            .all(|leaf| leaf.alpha_dim() == PROTECTED_LATENT_SYMBOLS));

        let dispatched = Cell::new(0_usize);
        let start = Instant::now();
        let mut coordinator = Coordinator::new(protected_cover_budget(start), move |_| start);
        let aggregate = propagate_cgan_cz_complete_cover_with(
            cover,
            &CGAN_NCH1_PROTECTED_LATENT_COVER_AXES,
            moat,
            PROTECTED_LATENT_LEAF_DOMAINS,
            0,
            &mut coordinator,
            |leaf_index, leaf, _, _| {
                assert_eq!(leaf.value_dim(), PROTECTED_LATENT_SYMBOLS);
                assert_eq!(leaf.alpha_dim(), PROTECTED_LATENT_SYMBOLS);
                dispatched.set(dispatched.get() + 1);
                Ok((
                    synthetic_completed_leaf_bounds(leaf_index, moat),
                    FULL_RUNNER_COMPLETED_STAGES,
                ))
            },
        )
        .unwrap();
        assert_eq!(dispatched.get(), PROTECTED_LATENT_LEAF_DOMAINS);
        assert_eq!(
            aggregate.leaf_completions.len(),
            PROTECTED_LATENT_LEAF_DOMAINS
        );
    }

    #[test]
    fn protected_alpha_queue_repeated_axis_is_a_complete_dyadic_cover() {
        let input = protected_cover_fixture();
        let mut limits = protected_cover_limits();
        limits.max_split_levels = 3;
        limits.max_tree_nodes = 15;
        limits.max_leaf_domains = 8;
        let start = Instant::now();
        let cover = enumerate_cgan_cz_protected_alpha_cover_unwired(
            &input,
            &[0, 0, 0],
            limits,
            protected_cover_budget(start),
        )
        .unwrap();
        assert_eq!(cover.leaves().len(), 8);

        for (index, leaf) in cover.leaves().iter().enumerate() {
            let expected_lower = -1.0 + f64::from(index as u32) * 0.25;
            let expected_upper = expected_lower + 0.25;
            let coefficient = leaf.generators()[0].entries().next().unwrap().1;
            assert_eq!(coefficient, 0.125);
            assert_eq!(leaf.center()[0] - coefficient, expected_lower);
            assert_eq!(leaf.center()[0] + coefficient, expected_upper);
            for axis in 1..PROTECTED_LATENT_SYMBOLS {
                assert_eq!(leaf.center()[axis], 0.0);
                assert_eq!(
                    leaf.generators()[axis].entries().collect::<Vec<_>>(),
                    vec![(axis, 1.0)]
                );
            }
        }

        let (_, moat) = certified_input_and_moat();
        let propagated = Cell::new(0_usize);
        let mut coordinator = Coordinator::new(protected_cover_budget(start), move |_| start);
        let aggregate = propagate_cgan_cz_complete_cover_with(
            cover,
            &[0, 0, 0],
            moat,
            8,
            0,
            &mut coordinator,
            |leaf_index, _, _, _| {
                propagated.set(propagated.get() + 1);
                Ok((
                    synthetic_completed_leaf_bounds(leaf_index, moat),
                    FULL_RUNNER_COMPLETED_STAGES,
                ))
            },
        )
        .unwrap();
        assert_eq!(propagated.get(), 8);
        assert_eq!(aggregate.cover.split_levels(), 3);
        assert_eq!(aggregate.leaf_completions.len(), 8);
    }

    #[test]
    fn protected_alpha_cover_caps_and_deadline_publish_no_partial_frontier() {
        let input = protected_cover_fixture();
        let start = Instant::now();

        let mut node_limited = protected_cover_limits();
        node_limited.max_tree_nodes = 62;
        assert!(matches!(
            enumerate_cgan_nch1_protected_latent_cover_unwired(
                &input,
                node_limited,
                protected_cover_budget(start),
            ),
            Err(CganCzProbeDecline::ResourceLimit {
                resource: "protected-alpha tree nodes",
                required: 63,
                limit: 62,
            })
        ));

        let body = domain_live_bytes(&input).unwrap();
        let conservative_peak = cover_queue_absolute_bytes(0, 48, 35, body).unwrap()
            + PROTECTED_LATENT_SYMBOLS * size_of::<usize>();
        assert!(matches!(
            enumerate_cgan_nch1_protected_latent_cover_unwired(
                &input,
                protected_cover_limits(),
                ConstrainedZonotopeCallBudget::new(
                    start + Duration::from_mins(1),
                    0,
                    conservative_peak - 1,
                ),
            ),
            Err(CganCzProbeDecline::Budget(
                ConstrainedZonotopeCallBudgetError::PeakLiveBytesExceeded {
                    required,
                    limit,
                }
            )) if required == conservative_peak && limit + 1 == required
        ));

        let deadline = start + Duration::from_mins(1);
        let node_admissions = Cell::new(0_usize);
        let result = enumerate_cgan_cz_protected_alpha_cover_with_clock(
            &input,
            &CGAN_NCH1_PROTECTED_LATENT_COVER_AXES,
            protected_cover_limits(),
            ConstrainedZonotopeCallBudget::new(deadline, 0, 1 << 30),
            |checkpoint| {
                if checkpoint == "protected-alpha cover node admission" {
                    node_admissions.set(node_admissions.get() + 1);
                    deadline
                } else {
                    start
                }
            },
        );
        assert!(matches!(
            result,
            Err(CganCzProbeDecline::Budget(
                ConstrainedZonotopeCallBudgetError::DeadlineExpired {
                    checkpoint: "protected-alpha cover node admission"
                }
            ))
        ));
        assert_eq!(node_admissions.get(), 1);
    }

    #[test]
    fn protected_alpha_cover_rejects_malformed_settings_before_splitting() {
        let input = protected_cover_fixture();
        let start = Instant::now();
        let budget = protected_cover_budget(start);

        assert!(matches!(
            enumerate_cgan_cz_protected_alpha_cover_unwired(
                &input,
                &[],
                protected_cover_limits(),
                budget,
            ),
            Err(CganCzProbeDecline::InvalidLimit { .. })
        ));

        let mut zero = protected_cover_limits();
        zero.max_leaf_domains = 0;
        assert!(matches!(
            enumerate_cgan_nch1_protected_latent_cover_unwired(&input, zero, budget),
            Err(CganCzProbeDecline::InvalidLimit { .. })
        ));

        let mut wrong_prefix = protected_cover_limits();
        wrong_prefix.protected_alpha_dim = 4;
        assert!(matches!(
            enumerate_cgan_nch1_protected_latent_cover_unwired(&input, wrong_prefix, budget),
            Err(CganCzProbeDecline::InvalidLimit { .. })
        ));

        assert!(matches!(
            enumerate_cgan_cz_protected_alpha_cover_unwired(
                &input,
                &[PROTECTED_LATENT_SYMBOLS],
                protected_cover_limits(),
                budget,
            ),
            Err(CganCzProbeDecline::InvalidLimit { .. })
        ));

        let mut level_limited = protected_cover_limits();
        level_limited.max_split_levels = 4;
        assert!(matches!(
            enumerate_cgan_nch1_protected_latent_cover_unwired(&input, level_limited, budget),
            Err(CganCzProbeDecline::ResourceLimit {
                resource: "protected-alpha split levels",
                required: 5,
                limit: 4,
            })
        ));
    }

    #[test]
    fn exact_contract_has_26_chain_nodes_and_expected_shapes() {
        assert_eq!(EXPECTED_NODES.len(), 26);
        assert_eq!(EXPECTED_NODES[0], ("Gemm_0", LayerType::Linear));
        assert_eq!(EXPECTED_NODES[5], ("Relu_6", LayerType::ReLU));
        assert_eq!(EXPECTED_NODES[8], ("Relu_9", LayerType::ReLU));
        assert_eq!(EXPECTED_NODES[11], ("Relu_12", LayerType::ReLU));
        assert_eq!(
            &EXPECTED_NODES[12..=14],
            &[
                ("ConvTranspose_13", LayerType::ConvTranspose2d),
                ("Conv_14", LayerType::Conv2d),
                ("Relu_15", LayerType::ReLU),
            ]
        );
        assert_eq!(EXPECTED_NODES[25], ("Gemm_27", LayerType::Linear));
        assert_eq!(expected_output_shape(0), [512]);
        assert_eq!(expected_output_shape(5), [128, 6, 6]);
        assert_eq!(expected_output_shape(8), [64, 14, 14]);
        assert_eq!(expected_output_shape(11), [32, 30, 30]);
        assert_eq!(expected_output_shape(12), [1, 32, 32]);
        assert_eq!(
            expected_output_shape_for_profile(CganCzImgSz32Profile::Nch3, 12),
            [3, 32, 32]
        );
        assert_eq!(expected_output_shape(14), [16, 16, 16]);
        assert_eq!(expected_output_shape(25), [1]);
        assert_eq!(CganCzImgSz32Profile::Nch1.image_channels(), 1);
        assert_eq!(CganCzImgSz32Profile::Nch3.image_channels(), 3);
        for index in 0..EXPECTED_NODE_COUNT {
            if index != 12 {
                assert_eq!(
                    expected_output_shape_for_profile(CganCzImgSz32Profile::Nch1, index),
                    expected_output_shape_for_profile(CganCzImgSz32Profile::Nch3, index),
                    "imgSz32 profiles unexpectedly differ at node {index}"
                );
            }
        }
        assert!(EXPECTED_NODES.windows(2).all(|pair| pair[0].0 != pair[1].0));
        assert_eq!(
            ProbeExtent::FirstGeneratorBlock.prefix_last_index(),
            Some(5)
        );
        assert_eq!(
            ProbeExtent::SecondGeneratorBlock.prefix_last_index(),
            Some(8)
        );
        assert_eq!(
            ProbeExtent::ThirdGeneratorBlock.prefix_last_index(),
            Some(11)
        );
        assert_eq!(
            ProbeExtent::GeneratorDiscriminatorHandoff.prefix_last_index(),
            Some(14)
        );
        assert_eq!(ProbeExtent::Full.prefix_last_index(), None);
        assert_eq!(ProbeExtent::SecondGeneratorBlock.prefix_end_exclusive(), 9);
        assert_eq!(ProbeExtent::ThirdGeneratorBlock.prefix_end_exclusive(), 12);
        assert_eq!(
            ProbeExtent::GeneratorDiscriminatorHandoff.prefix_end_exclusive(),
            15
        );
        assert_eq!(
            cgan_nch1_generator_discriminator_handoff_qualification_limits(),
            cgan_nch1_third_block_qualification_limits()
        );
        assert_eq!(
            cgan_nch1_independent_interval_qualification_limits(),
            cgan_nch1_third_block_qualification_limits()
        );
        assert_eq!(
            cgan_nch3_independent_interval_qualification_limits(),
            cgan_nch1_independent_interval_qualification_limits()
        );
        let limits = qualification_limits();
        assert_eq!(relu_reduction_target(ProbeExtent::Full, 14, limits), None);
        assert_eq!(relu_reduction_target(ProbeExtent::Full, 16, limits), None);
        assert_eq!(
            relu_reduction_target(ProbeExtent::Full, 19, limits),
            Some(TAIL_DISCRIMINATOR_RETAINED_ALPHA_DIM)
        );
        for index in [5, 8, 11, 22] {
            assert_eq!(
                relu_reduction_target(ProbeExtent::Full, index, limits),
                Some(limits.retained_alpha_dim)
            );
        }
        for index in [14, 16, 19] {
            assert_eq!(
                relu_reduction_target(ProbeExtent::GeneratorDiscriminatorHandoff, index, limits),
                Some(limits.retained_alpha_dim)
            );
        }
    }

    #[test]
    fn independent_input_accepts_mixed_declared_points_without_collapsing_their_moat() {
        let input = certified_mixed_point_input();
        assert_eq!(input.declared_point(), &[false, false, true, false, false]);
        assert!(input.lower()[2] < input.upper()[2]);

        let sealed = SealedCgan {
            layers: Vec::new(),
            parameter_elements: 0,
            live_bytes: 0,
        };
        let limits = qualification_limits();
        let start = Instant::now();
        let budget = ConstrainedZonotopeCallBudget::new(start + Duration::from_mins(1), 0, 1 << 30);
        let mut coordinator = Coordinator::new(budget, move |_| start);
        let domain =
            build_independent_input_domain(&input, limits, &sealed, 0, &mut coordinator).unwrap();
        assert_eq!(domain.value_dim(), PROTECTED_LATENT_SYMBOLS);
        assert_eq!(domain.alpha_dim(), 0);
        assert_eq!(domain.constraint_count(), 0);

        let boxed = certified_box_from_remainder_only_zonotope_unwired_with_budget(
            &domain,
            independent_box_limits(limits).unwrap(),
            ConstrainedZonotopeCallBudget::new(
                start + Duration::from_mins(1),
                domain_live_bytes(&domain).unwrap(),
                1 << 30,
            ),
        )
        .unwrap()
        .into_value();
        for coordinate in 0..input.len() {
            assert!(boxed.lower()[coordinate] <= input.lower()[coordinate]);
            assert!(boxed.upper()[coordinate] >= input.upper()[coordinate]);
        }
    }

    #[test]
    fn correlated_leaf_input_preserves_five_stable_symbols_for_fixed_coordinates() {
        let lower = [-0.0, -1.0, 2.0, 3.0, 4.0];
        let upper = [0.0, 1.0, 2.0, 3.5, 4.0];
        let start = Instant::now();
        let budget =
            ConstrainedZonotopeCallBudget::new(start + Duration::from_mins(1), 17, 1 << 20);
        let mut coordinator = Coordinator::new(budget, move |_| start);
        let domain = build_correlated_leaf_input_domain_from_bounds(
            &lower,
            &upper,
            qualification_limits(),
            budget.baseline_live_bytes(),
            &mut coordinator,
        )
        .unwrap();
        assert_eq!(domain.value_dim(), PROTECTED_LATENT_SYMBOLS);
        assert_eq!(domain.alpha_dim(), PROTECTED_LATENT_SYMBOLS);
        assert_eq!(domain.constraint_count(), 0);
        assert_eq!(generator_nonzeros(&domain).unwrap(), 2);
        assert_eq!(
            domain
                .generators()
                .iter()
                .map(|generator| generator.nnz())
                .collect::<Vec<_>>(),
            [0, 1, 0, 1, 0]
        );
        assert!(coordinator.peak_live_bytes <= budget.max_peak_live_bytes());
    }

    #[test]
    fn leaf_f32_input_promotion_is_exact_finite_ordered_and_budgeted() {
        let lower = [-0.0, -1.5, f32::from_bits(1), -f32::MIN_POSITIVE, 42.25];
        let upper = [0.0, -1.25, f32::from_bits(2), f32::MIN_POSITIVE, 42.25];
        let start = Instant::now();
        let deadline = start + Duration::from_secs(1);
        let baseline = 73_usize;
        let required = baseline + size_of::<CganCzLeafInput64>();
        let mut coordinator = Coordinator::new(
            ConstrainedZonotopeCallBudget::new(deadline, baseline, required),
            move |_| start,
        );
        let promoted = promote_cgan_leaf_input_f32(&lower, &upper, &mut coordinator).unwrap();
        for index in 0..PROTECTED_LATENT_SYMBOLS {
            assert_eq!(
                promoted.lower[index].to_bits(),
                f64::from(lower[index]).to_bits()
            );
            assert_eq!(
                promoted.upper[index].to_bits(),
                f64::from(upper[index]).to_bits()
            );
        }
        assert_eq!(coordinator.peak_live_bytes, required);

        let mut one_byte_low = Coordinator::new(
            ConstrainedZonotopeCallBudget::new(deadline, baseline, required - 1),
            move |_| start,
        );
        assert!(matches!(
            promote_cgan_leaf_input_f32(&lower, &upper, &mut one_byte_low),
            Err(CganCzProbeDecline::Budget(
                ConstrainedZonotopeCallBudgetError::PeakLiveBytesExceeded {
                    required: observed,
                    limit
                }
            )) if observed == required && limit + 1 == observed
        ));

        let mut expired = Coordinator::new(
            ConstrainedZonotopeCallBudget::new(deadline, baseline, required),
            move |_| deadline,
        );
        assert!(matches!(
            promote_cgan_leaf_input_f32(&lower, &upper, &mut expired),
            Err(CganCzProbeDecline::Budget(
                ConstrainedZonotopeCallBudgetError::DeadlineExpired {
                    checkpoint: "cGAN leaf-row input admission"
                }
            ))
        ));

        for (bad_lower, bad_upper) in [
            ([f32::NAN, 0.0, 0.0, 0.0, 0.0], [0.0; 5]),
            ([f32::NEG_INFINITY, 0.0, 0.0, 0.0, 0.0], [0.0; 5]),
            ([1.0, 0.0, 0.0, 0.0, 0.0], [0.0; 5]),
        ] {
            let mut malformed = Coordinator::new(
                ConstrainedZonotopeCallBudget::new(deadline, baseline, required),
                move |_| start,
            );
            assert!(matches!(
                promote_cgan_leaf_input_f32(&bad_lower, &bad_upper, &mut malformed),
                Err(CganCzProbeDecline::Topology { .. })
            ));
        }

        let mut wrong_length = Coordinator::new(
            ConstrainedZonotopeCallBudget::new(deadline, baseline, required),
            move |_| start,
        );
        assert!(matches!(
            promote_cgan_leaf_input_f32(&lower[..4], &upper, &mut wrong_length),
            Err(CganCzProbeDecline::Topology { .. })
        ));
    }

    fn independent_relu_fixture() -> (ConstrainedZonotope64, Vec<usize>) {
        let value_dim = checked_product(expected_output_shape(5), "test ReLU fixture").unwrap();
        let lower = (0..value_dim)
            .map(|coordinate| match coordinate % 3 {
                0 => -2.0,
                1 => -1.0,
                _ => 2.0,
            })
            .collect::<Vec<_>>();
        let upper = (0..value_dim)
            .map(|coordinate| match coordinate % 3 {
                0 => -1.0,
                1 => 3.0,
                _ => 4.0,
            })
            .collect::<Vec<_>>();
        let domain =
            ConstrainedZonotope64::from_certified_bounds(&lower, &upper, &vec![true; value_dim])
                .unwrap();
        (domain, expected_output_shape(5).to_vec())
    }

    fn independent_relu_records() -> Vec<CganCzIndependentIntervalReluBounds> {
        let mut records = Vec::new();
        records.try_reserve_exact(CGAN_RELU_COUNT).unwrap();
        records
    }

    fn authenticated_independent_relu_record(
        profile: CganCzImgSz32Profile,
        index: usize,
    ) -> CganCzIndependentIntervalReluBounds {
        let output_shape = expected_output_shape_for_profile(profile, index);
        let value_dim = checked_product(output_shape, "test auxiliary ReLU shape").unwrap();
        CganCzIndependentIntervalReluBounds {
            node: EXPECTED_NODES[index].0,
            output_shape,
            bounds: CertifiedAuxiliaryBounds64::try_new(
                vec![-1.0; value_dim],
                vec![1.0; value_dim],
            )
            .unwrap(),
        }
    }

    fn authenticated_independent_relu_records(
        profile: CganCzImgSz32Profile,
    ) -> Vec<CganCzIndependentIntervalReluBounds> {
        CGAN_RELU_INDICES
            .into_iter()
            .map(|index| authenticated_independent_relu_record(profile, index))
            .collect()
    }

    fn independent_relu_test_baseline(shape_capacity: usize, record_capacity: usize) -> usize {
        shape_capacity * size_of::<usize>()
            + record_capacity * size_of::<CganCzIndependentIntervalReluBounds>()
    }

    #[test]
    fn correlated_auxiliary_stream_authenticates_and_consumes_exact_relu_sequence() {
        for profile in [CganCzImgSz32Profile::Nch1, CganCzImgSz32Profile::Nch3] {
            let mut records = authenticated_independent_relu_records(profile);
            authenticate_independent_relu_bound_sequence(profile, &records).unwrap();
            let expected_endpoint_bytes = CGAN_RELU_INDICES
                .iter()
                .map(|&index| {
                    checked_product(
                        expected_output_shape_for_profile(profile, index),
                        "test retained auxiliary bytes",
                    )
                    .unwrap()
                        * 2
                        * size_of::<f64>()
                })
                .sum::<usize>();
            assert_eq!(
                independent_auxiliary_live_bytes(&records).unwrap(),
                expected_endpoint_bytes
            );

            let final_relu_23 = records.pop().unwrap();
            assert_eq!(records.len(), CGAN_CORRELATED_AUXILIARY_RELU_COUNT);
            assert_eq!(
                authenticate_independent_relu_bound_record(profile, 22, &final_relu_23)
                    .unwrap()
                    .value_dim(),
                TAIL_VALUE_DIM
            );
            let mut stream = records.iter();
            let mut consumed = 0_usize;
            let mut consumed_indices = Vec::new();
            for index in 0..ProbeExtent::Full.prefix_end_exclusive() {
                if next_correlated_auxiliary_bounds(profile, index, &mut stream, &mut consumed)
                    .unwrap()
                    .is_some()
                {
                    consumed_indices.push(index);
                }
            }
            assert_eq!(consumed, CGAN_CORRELATED_AUXILIARY_RELU_COUNT);
            assert!(stream.next().is_none());
            assert_eq!(
                consumed_indices.as_slice(),
                &CGAN_RELU_INDICES[..CGAN_CORRELATED_AUXILIARY_RELU_COUNT]
            );
        }
    }

    #[test]
    fn correlated_auxiliary_stream_rejects_missing_extra_and_malformed_records() {
        let profile = CganCzImgSz32Profile::Nch1;
        let mut records = authenticated_independent_relu_records(profile);
        let missing = records.pop().unwrap();
        assert!(matches!(
            authenticate_independent_relu_bound_sequence(profile, &records),
            Err(CganCzProbeDecline::Topology { .. })
        ));
        records.push(missing);
        records.push(authenticated_independent_relu_record(profile, 22));
        assert!(matches!(
            authenticate_independent_relu_bound_sequence(profile, &records),
            Err(CganCzProbeDecline::Topology { .. })
        ));
        records.pop();

        let mut out_of_order = records[..CGAN_CORRELATED_AUXILIARY_RELU_COUNT].iter();
        let mut consumed = 0_usize;
        assert!(matches!(
            next_correlated_auxiliary_bounds(profile, 8, &mut out_of_order, &mut consumed),
            Err(CganCzProbeDecline::Topology { .. })
        ));

        records[0].node = EXPECTED_NODES[8].0;
        assert!(matches!(
            authenticate_independent_relu_bound_sequence(profile, &records),
            Err(CganCzProbeDecline::Topology { .. })
        ));
        records[0].node = EXPECTED_NODES[5].0;
        records[0].output_shape = &[1];
        assert!(matches!(
            authenticate_independent_relu_bound_sequence(profile, &records),
            Err(CganCzProbeDecline::Topology { .. })
        ));
        records[0].output_shape = expected_output_shape(5);
        records[0].bounds = CertifiedAuxiliaryBounds64::try_new(vec![-1.0], vec![1.0]).unwrap();
        assert!(matches!(
            authenticate_independent_relu_bound_sequence(profile, &records),
            Err(CganCzProbeDecline::Topology { .. })
        ));
        assert!(matches!(
            authenticate_independent_relu_bound_record(profile, 0, &records[0]),
            Err(CganCzProbeDecline::Topology { .. })
        ));
    }

    #[test]
    fn depth_two_plan_and_strict_counterfactual_selection_are_exact() {
        assert_eq!(TAIL_DEPTH_TWO_WEIGHT_ELEMENTS, 73_728);
        assert_eq!(TAIL_DEPTH_TWO_KERNEL_VISITS, 294_912);
        assert_eq!(TAIL_DEPTH_TWO_EXACT_PRODUCTS, 298_688);
        let plan = ReluTailConvBatchNormPullbackPlan {
            input_shape: TAIL_DEPTH_TWO_INPUT_SHAPE,
            output_shape: TAIL_DEPTH_TWO_OUTPUT_SHAPE,
            weight_shape: TAIL_DEPTH_TWO_WEIGHT_SHAPE,
            weight_elements: TAIL_DEPTH_TWO_WEIGHT_ELEMENTS,
            kernel_visits: TAIL_DEPTH_TWO_KERNEL_VISITS,
            pulled_margin_construction_exact_product_bound: TAIL_DEPTH_TWO_EXACT_PRODUCTS,
        };
        assert!(depth_two_plan_is_exact(plan));
        assert!(!depth_two_plan_is_exact(
            ReluTailConvBatchNormPullbackPlan {
                kernel_visits: plan.kernel_visits - 1,
                ..plan
            }
        ));

        assert_eq!(depth_two_counterfactual_selection(-2.0, -3.0), (-2.0, -1.0));
        assert_eq!(depth_two_counterfactual_selection(-2.0, -2.0), (-2.0, 0.0));
        assert_eq!(depth_two_counterfactual_selection(-2.0, -1.0), (-1.0, 1.0));
    }

    #[test]
    fn depth_two_m17_m20_validation_reconstructs_selection_and_binds_peak_cap() {
        let domain =
            ConstrainedZonotope64::from_certified_bounds(&[-1.0], &[1.0], &[true]).unwrap();
        let margin = ExactReluTailMargin::try_new(
            vec![BigRational::from_integer(1.into())],
            BigRational::zero(),
        )
        .unwrap();
        let cap = 4 << 20;
        let outcome = bound_relu_tail_triangle_dual_unwired_with_budget(
            &domain,
            &margin,
            None,
            depth_two_m17_config(&domain, qualification_limits()).unwrap(),
            ConstrainedZonotopeCallBudget::new(
                Instant::now() + Duration::from_mins(1),
                domain_live_bytes(&domain).unwrap(),
                cap,
            ),
        )
        .unwrap();
        let original = outcome.value().clone();
        let mut auxiliary = original.clone();
        auxiliary.lower_bound = original.lower_bound.next_up();
        let completed = ny_mip::ReluTailBoxCutDualResult {
            lower_bound: auxiliary.lower_bound,
            selected: ReluTailBoxCutSelection::Auxiliary,
            original: original.clone(),
            auxiliary: Some(auxiliary),
            box_cut: None,
            status: ReluTailBoxCutStatus::Completed,
        };
        assert_eq!(
            validate_depth_two_m17_m20_portfolio(&completed, None, cap)
                .map(|(_, _, selected)| selected),
            Some(ReluTailBoxCutSelection::Auxiliary)
        );

        let fallback = ny_mip::ReluTailBoxCutDualResult {
            lower_bound: original.lower_bound,
            selected: ReluTailBoxCutSelection::Original,
            original,
            auxiliary: None,
            box_cut: None,
            status: ReluTailBoxCutStatus::AuxiliaryFallback,
        };
        let matching_peak = ConstrainedZonotopeCallBudgetError::PeakLiveBytesExceeded {
            required: cap + 1,
            limit: cap,
        };
        assert!(
            validate_depth_two_m17_m20_portfolio(&fallback, Some(&matching_peak), cap,).is_some()
        );
        let forged_peak = ConstrainedZonotopeCallBudgetError::PeakLiveBytesExceeded {
            required: cap,
            limit: cap - 1,
        };
        assert!(
            validate_depth_two_m17_m20_portfolio(&fallback, Some(&forged_peak), cap,).is_none()
        );
    }

    #[test]
    fn depth_two_optional_admission_uses_injected_clock_and_guard() {
        let start = Instant::now();
        let roomy_budget =
            ConstrainedZonotopeCallBudget::new(start + Duration::from_secs(10), 0, 1 << 20);
        let mut roomy_coordinator = Coordinator::new(roomy_budget, move |_| start);
        assert_eq!(
            depth_two_optional_deadline(&mut roomy_coordinator).unwrap(),
            Some(start + TAIL_DEPTH_TWO_MEASUREMENT_WALL_TIME)
        );

        let budget = ConstrainedZonotopeCallBudget::new(
            start + TAIL_DEPTH_TWO_PUBLICATION_GUARD,
            0,
            1 << 20,
        );
        let mut coordinator = Coordinator::new(budget, move |_| start);
        assert_eq!(depth_two_optional_deadline(&mut coordinator).unwrap(), None);
        assert_eq!(coordinator.deadline_polls, 1);
    }

    #[test]
    fn depth_two_second_row_accounts_retained_first_row_measurement() {
        let first_row = CganCzDepthTwoMeasurement::NotRequested;
        assert_eq!(
            depth_two_retained_measurement_baseline(17, &first_row),
            Some(17 + size_of::<CganCzDepthTwoMeasurement>())
        );
        assert_eq!(
            depth_two_retained_measurement_baseline(usize::MAX, &first_row),
            None
        );
    }

    #[test]
    fn retained_relu_predecessor_is_charged_only_during_reduction() {
        let value_dim = PROTECTED_LATENT_SYMBOLS + 1;
        let lower = vec![-1.0; value_dim];
        let upper = vec![1.0; value_dim];
        let input =
            ConstrainedZonotope64::from_certified_bounds(&lower, &upper, &vec![false; value_dim])
                .unwrap();
        let predecessor_bytes = domain_live_bytes(&input).unwrap();
        let limits = qualification_limits();
        let start = Instant::now();
        let budget = ConstrainedZonotopeCallBudget::new(start + Duration::from_mins(1), 0, 1 << 30);

        let mut ordinary = input.clone();
        let mut ordinary_stages = Vec::new();
        let mut ordinary_coordinator = Coordinator::new(budget, move |_| start);
        apply_relu_and_reduce(
            "test retained ReLU",
            &[PROTECTED_LATENT_SYMBOLS + 1],
            &mut ordinary,
            Some(PROTECTED_LATENT_SYMBOLS),
            None,
            limits,
            0,
            budget,
            &mut ordinary_coordinator,
            &mut ordinary_stages,
        )
        .unwrap();

        let mut retaining_stages = Vec::new();
        let mut retaining_coordinator = Coordinator::new(budget, move |_| start);
        let retaining = apply_relu_and_reduce_retaining_predecessor(
            "test retained ReLU",
            &[PROTECTED_LATENT_SYMBOLS + 1],
            &input,
            Some(PROTECTED_LATENT_SYMBOLS),
            None,
            limits,
            0,
            predecessor_bytes,
            budget,
            &mut retaining_coordinator,
            &mut retaining_stages,
        )
        .unwrap();

        assert_eq!(retaining, ordinary);
        assert_eq!(ordinary_stages.len(), 2);
        assert_eq!(retaining_stages.len(), 2);
        assert_eq!(
            retaining_stages[0].peak_live_bytes,
            ordinary_stages[0].peak_live_bytes
        );
        assert_eq!(
            retaining_stages[1].peak_live_bytes,
            ordinary_stages[1]
                .peak_live_bytes
                .checked_add(predecessor_bytes)
                .unwrap()
        );
    }

    #[test]
    fn auxiliary_relu_refusal_is_fail_closed_without_legacy_retry() {
        let input = ConstrainedZonotope64::try_new(
            vec![0.0],
            vec![vec![(0, 1.0)]],
            Array2::zeros((0, 1)),
            Vec::new(),
            vec![0.0],
        )
        .unwrap();
        let auxiliary = CertifiedAuxiliaryBounds64::try_new(vec![2.0], vec![3.0]).unwrap();
        let limits = qualification_limits();
        let start = Instant::now();
        let budget = ConstrainedZonotopeCallBudget::new(start + Duration::from_mins(1), 0, 1 << 30);
        let mut domain = input.clone();
        let mut stages = Vec::new();
        let mut coordinator = Coordinator::new(budget, move |_| start);
        assert!(matches!(
            apply_relu_and_reduce(
                "test ReLU",
                &[1],
                &mut domain,
                None,
                Some(&auxiliary),
                limits,
                independent_endpoint_bytes(auxiliary.value_dim()).unwrap(),
                budget,
                &mut coordinator,
                &mut stages,
            ),
            Err(CganCzProbeDecline::Transform {
                operation: "ReLU",
                ..
            })
        ));
        assert_eq!(domain, input);
        assert!(stages.is_empty());

        let mut legacy_domain = input;
        let mut legacy_stages = Vec::new();
        let mut legacy_coordinator = Coordinator::new(budget, move |_| start);
        apply_relu_and_reduce(
            "test ReLU",
            &[1],
            &mut legacy_domain,
            None,
            None,
            limits,
            0,
            budget,
            &mut legacy_coordinator,
            &mut legacy_stages,
        )
        .unwrap();
        assert_eq!(legacy_stages.len(), 1);
    }

    #[test]
    fn independent_relu_bridge_chain_is_sound_and_preserves_remainder_only_shape() {
        let (mut domain, mut shape) = independent_relu_fixture();
        let limits = qualification_limits();
        let start = Instant::now();
        let budget = ConstrainedZonotopeCallBudget::new(start + Duration::from_mins(1), 0, 1 << 30);
        let expected_box = certified_box_from_remainder_only_zonotope_unwired_with_budget(
            &domain,
            independent_box_limits(limits).unwrap(),
            ConstrainedZonotopeCallBudget::new(
                start + Duration::from_mins(1),
                domain_live_bytes(&domain).unwrap(),
                1 << 30,
            ),
        )
        .unwrap()
        .into_value();
        let mut records = independent_relu_records();
        let retained_baseline =
            independent_relu_test_baseline(shape.capacity(), records.capacity());
        let mut coordinator = Coordinator::new(budget, move |_| start);

        apply_independent_interval_relu(
            CganCzImgSz32Profile::Nch1,
            5,
            &mut domain,
            &mut shape,
            limits,
            retained_baseline,
            budget,
            &mut coordinator,
            &mut records,
        )
        .unwrap();

        assert_eq!(records.len(), 1);
        assert_eq!(records[0].node(), "Relu_6");
        assert_eq!(records[0].output_shape(), expected_output_shape(5));
        assert_eq!(records[0].bounds().lower(), expected_box.lower());
        assert_eq!(records[0].bounds().upper(), expected_box.upper());
        assert_eq!(shape, expected_output_shape(5));
        assert_eq!(domain.value_dim(), expected_box.len());
        assert_eq!(domain.alpha_dim(), 0);
        assert_eq!(domain.constraint_count(), 0);

        let post_box = certified_box_from_remainder_only_zonotope_unwired_with_budget(
            &domain,
            independent_box_limits(limits).unwrap(),
            ConstrainedZonotopeCallBudget::new(
                start + Duration::from_mins(1),
                domain_live_bytes(&domain).unwrap(),
                1 << 30,
            ),
        )
        .unwrap()
        .into_value();
        for coordinate in 0..post_box.len() {
            assert!(post_box.lower()[coordinate] <= expected_box.lower()[coordinate].max(0.0));
            assert!(post_box.upper()[coordinate] >= expected_box.upper()[coordinate].max(0.0));
        }
        assert!(coordinator.peak_live_bytes > 0);
        assert!(coordinator.charged_items >= expected_box.len() * 3);
        assert!(coordinator.deadline_polls > 0);
    }

    #[test]
    fn independent_relu_bridge_rejects_shape_alpha_and_predicates() {
        let limits = qualification_limits();
        let start = Instant::now();
        let budget = ConstrainedZonotopeCallBudget::new(start + Duration::from_mins(1), 0, 1 << 30);

        let (mut wrong_shape_domain, mut wrong_shape) = independent_relu_fixture();
        wrong_shape[2] -= 1;
        let mut records = independent_relu_records();
        let retained_baseline =
            independent_relu_test_baseline(wrong_shape.capacity(), records.capacity());
        let mut coordinator = Coordinator::new(budget, move |_| start);
        assert!(matches!(
            apply_independent_interval_relu(
                CganCzImgSz32Profile::Nch1,
                5,
                &mut wrong_shape_domain,
                &mut wrong_shape,
                limits,
                retained_baseline,
                budget,
                &mut coordinator,
                &mut records,
            ),
            Err(CganCzProbeDecline::Topology { .. })
        ));

        let value_dim = checked_product(expected_output_shape(5), "test alpha fixture").unwrap();
        let mut alpha_domain = ConstrainedZonotope64::try_new(
            vec![0.0; value_dim],
            vec![vec![(0, 1.0)]],
            Array2::zeros((0, 1)),
            Vec::new(),
            vec![0.0; value_dim],
        )
        .unwrap();
        let mut shape = expected_output_shape(5).to_vec();
        let mut records = independent_relu_records();
        let retained_baseline =
            independent_relu_test_baseline(shape.capacity(), records.capacity());
        let mut coordinator = Coordinator::new(budget, move |_| start);
        assert!(matches!(
            apply_independent_interval_relu(
                CganCzImgSz32Profile::Nch1,
                5,
                &mut alpha_domain,
                &mut shape,
                limits,
                retained_baseline,
                budget,
                &mut coordinator,
                &mut records,
            ),
            Err(CganCzProbeDecline::Transform {
                operation: "remainder-only CZ-to-Box bridge",
                message,
                ..
            }) if message.contains("alpha_dim == 0")
        ));

        let mut predicate_domain = ConstrainedZonotope64::try_new(
            vec![0.0; value_dim],
            Vec::new(),
            Array2::from_shape_vec((1, 0), Vec::new()).unwrap(),
            vec![0.0],
            vec![0.0; value_dim],
        )
        .unwrap();
        let mut shape = expected_output_shape(5).to_vec();
        let mut records = independent_relu_records();
        let retained_baseline =
            independent_relu_test_baseline(shape.capacity(), records.capacity());
        let mut coordinator = Coordinator::new(budget, move |_| start);
        assert!(matches!(
            apply_independent_interval_relu(
                CganCzImgSz32Profile::Nch1,
                5,
                &mut predicate_domain,
                &mut shape,
                limits,
                retained_baseline,
                budget,
                &mut coordinator,
                &mut records,
            ),
            Err(CganCzProbeDecline::Transform {
                operation: "remainder-only CZ-to-Box bridge",
                message,
                ..
            }) if message.contains("constraint_count == 0")
        ));
    }

    #[test]
    fn independent_relu_bridge_deadline_and_exact_peak_fail_closed() {
        let limits = qualification_limits();
        let start = Instant::now();
        let deadline = start + Duration::from_mins(1);
        let budget = ConstrainedZonotopeCallBudget::new(deadline, 0, 1 << 30);
        let (mut domain, mut shape) = independent_relu_fixture();
        let original = domain.clone();
        let original_shape = shape.clone();
        let mut records = independent_relu_records();
        let retained_baseline =
            independent_relu_test_baseline(shape.capacity(), records.capacity());
        let mut coordinator = Coordinator::new(budget, |checkpoint| {
            if checkpoint == "cGAN independent ReLU auxiliary publication" {
                deadline
            } else {
                start
            }
        });
        assert!(matches!(
            apply_independent_interval_relu(
                CganCzImgSz32Profile::Nch1,
                5,
                &mut domain,
                &mut shape,
                limits,
                retained_baseline,
                budget,
                &mut coordinator,
                &mut records,
            ),
            Err(CganCzProbeDecline::Budget(
                ConstrainedZonotopeCallBudgetError::DeadlineExpired {
                    checkpoint: "cGAN independent ReLU auxiliary publication"
                }
            ))
        ));
        assert_eq!(domain, original);
        assert_eq!(shape, original_shape);
        assert!(records.is_empty());

        let (mut domain, mut shape) = independent_relu_fixture();
        let mut records = independent_relu_records();
        let retained_baseline =
            independent_relu_test_baseline(shape.capacity(), records.capacity());
        let mut coordinator = Coordinator::new(budget, move |_| start);
        apply_independent_interval_relu(
            CganCzImgSz32Profile::Nch1,
            5,
            &mut domain,
            &mut shape,
            limits,
            retained_baseline,
            budget,
            &mut coordinator,
            &mut records,
        )
        .unwrap();
        let exact_peak = coordinator.peak_live_bytes;
        assert!(exact_peak > 0);

        let (mut domain, mut shape) = independent_relu_fixture();
        let mut records = independent_relu_records();
        let retained_baseline =
            independent_relu_test_baseline(shape.capacity(), records.capacity());
        let one_byte_low = ConstrainedZonotopeCallBudget::new(deadline, 0, exact_peak - 1);
        let mut coordinator = Coordinator::new(one_byte_low, move |_| start);
        assert!(matches!(
            apply_independent_interval_relu(
                CganCzImgSz32Profile::Nch1,
                5,
                &mut domain,
                &mut shape,
                limits,
                retained_baseline,
                one_byte_low,
                &mut coordinator,
                &mut records,
            ),
            Err(CganCzProbeDecline::Budget(
                ConstrainedZonotopeCallBudgetError::PeakLiveBytesExceeded { required, limit }
            )) if required == exact_peak && limit + 1 == required
        ));
    }

    #[test]
    fn verdict_authority_is_mechanically_disabled() {
        fn exhaustively_require_disabled(value: CganCzVerdictAuthority) {
            match value {
                CganCzVerdictAuthority::DisabledPendingExactMoatReplay => {}
            }
        }
        assert_eq!(
            CGAN_CZ_VERDICT_AUTHORITY,
            CganCzVerdictAuthority::DisabledPendingExactMoatReplay
        );
        exhaustively_require_disabled(CGAN_CZ_VERDICT_AUTHORITY);
    }

    fn synthetic_tail_fixture() -> (ConstrainedZonotope64, BatchNormParameters, AffineParameters) {
        let mut center = vec![-1.0; TAIL_VALUE_DIM];
        center[0] = 1.0;
        center[4] = 2.0;
        let mut remainder = vec![0.0; TAIL_VALUE_DIM];
        remainder[0] = 0.5;
        remainder[4] = 1.0;
        let domain = ConstrainedZonotope64::try_new(
            center,
            Vec::new(),
            Array2::zeros((0, 0)),
            Vec::new(),
            remainder,
        )
        .unwrap();

        let mut normalized_bias = vec![0.0; TAIL_CHANNELS];
        normalized_bias[0] = 0.5;
        normalized_bias[1] = -0.25;
        let mut normalized_scale = vec![1.0; TAIL_CHANNELS];
        normalized_scale[1] = -1.0;
        let mut gamma = vec![1.0; TAIL_CHANNELS];
        gamma[0] = 0.75;
        gamma[1] = -0.5;
        let mut beta = vec![0.0; TAIL_CHANNELS];
        beta[0] = 0.375;
        let batch_norm = BatchNormParameters {
            gamma,
            beta,
            mean: vec![0.0; TAIL_CHANNELS],
            variance: vec![0.0; TAIL_CHANNELS],
            epsilon: 1.0,
            normalized_scale,
            normalized_bias,
        };

        let mut weights = Array2::zeros((1, TAIL_VALUE_DIM));
        weights[(0, 0)] = -2.0;
        weights[(0, 4)] = 3.0;
        let affine = AffineParameters {
            weights,
            bias: vec![1.0],
        };
        (domain, batch_norm, affine)
    }

    fn synthetic_tail_certificate(
        batch_norm: &BatchNormParameters,
    ) -> ExactBatchNormAffineSurrogateCertificate {
        let shape = [TAIL_CHANNELS, 2, 2];
        ny_mip::certify_batch_norm_affine_surrogate_unwired(
            ConstrainedZonotopeBatchNormSpec {
                input_shape: &shape,
                channel_axis: 0,
                gamma: &batch_norm.gamma,
                beta: &batch_norm.beta,
                mean: &batch_norm.mean,
                variance: &batch_norm.variance,
                epsilon: batch_norm.epsilon,
                mode: ConstrainedZonotopeBatchNormMode::Inference,
            },
            &batch_norm.normalized_scale,
            &batch_norm.normalized_bias,
            ConstrainedZonotopeBatchNormAffineCertificateLimits {
                max_rank: 3,
                max_channel_count: TAIL_CHANNELS,
                max_parameter_elements: TAIL_CHANNELS * 6,
            },
        )
        .unwrap()
    }

    #[test]
    fn exact_tail_correction_and_both_m17_objectives_preserve_signs() {
        let (domain, batch_norm, affine) = synthetic_tail_fixture();
        let start = Instant::now();
        let budget = ConstrainedZonotopeCallBudget::new(start + Duration::from_mins(1), 0, 1 << 30);
        let mut coordinator = Coordinator::new(budget, move |_| start);
        validate_output_tail_contract(&domain, &batch_norm, &affine, &mut coordinator).unwrap();
        let shape = [TAIL_CHANNELS, 2, 2];
        let certificate_budget = domain_call_budget(&domain, 0, budget, &mut coordinator).unwrap();
        let outcome = certify_batch_norm_affine_surrogate_unwired_with_budget(
            ConstrainedZonotopeBatchNormSpec {
                input_shape: &shape,
                channel_axis: 0,
                gamma: &batch_norm.gamma,
                beta: &batch_norm.beta,
                mean: &batch_norm.mean,
                variance: &batch_norm.variance,
                epsilon: batch_norm.epsilon,
                mode: ConstrainedZonotopeBatchNormMode::Inference,
            },
            &batch_norm.normalized_scale,
            &batch_norm.normalized_bias,
            ConstrainedZonotopeBatchNormAffineCertificateLimits {
                max_rank: 3,
                max_channel_count: TAIL_CHANNELS,
                max_parameter_elements: TAIL_CHANNELS * 6,
            },
            certificate_budget,
        )
        .unwrap();
        let (certificate, certificate_report) = outcome.into_parts();
        coordinator.absorb(certificate_report).unwrap();
        let (correction, _) =
            exact_batch_norm_tail_correction(&domain, &certificate, &affine, 0, &mut coordinator)
                .unwrap();
        drop(certificate);
        assert_eq!(correction, BigRational::new(25.into(), 4.into()));

        let mut stages = Vec::new();
        let lower_charged_start = coordinator.charged_items;
        let lower_polls_start = coordinator.deadline_polls;
        let (lower, lower_objective_peak) = exact_output_tail_margin(
            &domain,
            &batch_norm,
            &affine,
            &correction,
            TailMarginSense::Lower,
            0,
            &mut coordinator,
        )
        .unwrap();
        assert_eq!(
            lower.coefficients()[0],
            BigRational::from_integer((-2).into())
        );
        assert_eq!(
            lower.coefficients()[4],
            BigRational::from_integer((-3).into())
        );
        assert_eq!(lower.bias(), &BigRational::from_integer((-7).into()));
        let lower_portfolio = run_output_tail_m17(
            &domain,
            None,
            &lower,
            CganCzStageKind::M17Lower,
            qualification_limits(),
            0,
            budget,
            &mut coordinator,
            &mut stages,
            lower_objective_peak,
            lower_charged_start,
            lower_polls_start,
        )
        .unwrap();

        let upper_charged_start = coordinator.charged_items;
        let upper_polls_start = coordinator.deadline_polls;
        let (upper, upper_objective_peak) = exact_output_tail_margin(
            &domain,
            &batch_norm,
            &affine,
            &correction,
            TailMarginSense::NegatedUpper,
            0,
            &mut coordinator,
        )
        .unwrap();
        assert_eq!(upper.coefficients()[0], BigRational::from_integer(2.into()));
        assert_eq!(upper.coefficients()[4], BigRational::from_integer(3.into()));
        assert_eq!(upper.bias(), &BigRational::new((-11).into(), 2.into()));
        let upper_portfolio = run_output_tail_m17(
            &domain,
            None,
            &upper,
            CganCzStageKind::M17Upper,
            qualification_limits(),
            0,
            budget,
            &mut coordinator,
            &mut stages,
            upper_objective_peak,
            upper_charged_start,
            upper_polls_start,
        )
        .unwrap();
        let lower_bound = lower_portfolio.selected_lower_bound;
        let upper_bound = -upper_portfolio.selected_lower_bound;
        assert!(lower_bound <= -19.0 && lower_bound > -19.000_000_001);
        assert!((1.5..1.500_000_001).contains(&upper_bound));
        assert_eq!(
            lower_portfolio.m17_candidates.status,
            ReluTailDualStatus::NoOptimizableSlopes
        );
        assert_eq!(
            upper_portfolio.m17_candidates.status,
            ReluTailDualStatus::NoOptimizableSlopes
        );
        assert_eq!(lower_portfolio.m20_status, CganCzM20Status::NotRequested);
        assert_eq!(upper_portfolio.m20_status, CganCzM20Status::NotRequested);
        assert_eq!(lower_portfolio.m20_lower_bound, None);
        assert_eq!(upper_portfolio.m20_lower_bound, None);
        assert_eq!(stages.len(), 2);
        assert_eq!(stages[0].kind, CganCzStageKind::M17Lower);
        assert_eq!(stages[1].kind, CganCzStageKind::M17Upper);
        assert_eq!(stages[1].output_shape, [1]);
        for stage in &stages {
            assert_eq!(stage.output_alpha_dim, 0);
            assert_eq!(stage.output_generator_nonzeros, 0);
        }
    }

    #[test]
    fn output_tail_contract_and_firewall_fail_closed() {
        let (domain, mut batch_norm, affine) = synthetic_tail_fixture();
        let start = Instant::now();
        let budget = ConstrainedZonotopeCallBudget::new(start + Duration::from_mins(1), 0, 1 << 30);
        let mut coordinator = Coordinator::new(budget, move |_| start);
        batch_norm.normalized_scale[0] = f64::NAN;
        assert!(matches!(
            validate_output_tail_contract(&domain, &batch_norm, &affine, &mut coordinator),
            Err(CganCzProbeDecline::OutputTail { .. })
        ));

        let (_, batch_norm, mut affine) = synthetic_tail_fixture();
        affine.weights[(0, 17)] = f64::INFINITY;
        assert!(matches!(
            validate_output_tail_contract(&domain, &batch_norm, &affine, &mut coordinator),
            Err(CganCzProbeDecline::OutputTail { .. })
        ));

        let (_, batch_norm, _) = synthetic_tail_fixture();
        let wrong_shape = AffineParameters {
            weights: Array2::zeros((1, TAIL_VALUE_DIM - 1)),
            bias: vec![0.0],
        };
        assert!(matches!(
            validate_output_tail_contract(&domain, &batch_norm, &wrong_shape, &mut coordinator),
            Err(CganCzProbeDecline::OutputTail { .. })
        ));

        let certificate = synthetic_tail_certificate(&batch_norm);
        let rational_slots = TAIL_VALUE_DIM + TAIL_TRANSIENT_RATIONAL_SLOTS;
        let exact_required = domain_live_bytes(&domain).unwrap()
            + certificate.conservative_live_bytes()
            + tail_rational_bytes(rational_slots, "test tail peak").unwrap();
        let mut memory_limited = Coordinator::new(
            ConstrainedZonotopeCallBudget::new(
                start + Duration::from_mins(1),
                0,
                exact_required - 1,
            ),
            move |_| start,
        );
        assert!(matches!(
            exact_batch_norm_tail_correction(
                &domain,
                &certificate,
                &synthetic_tail_fixture().2,
                0,
                &mut memory_limited,
            ),
            Err(CganCzProbeDecline::Budget(
                ConstrainedZonotopeCallBudgetError::PeakLiveBytesExceeded { .. }
            ))
        ));
    }

    #[test]
    fn diagnostic_moat_separation_is_strict_and_correction_rounds_outward() {
        assert!(tail_bounds_separate_unsafe_moat(-0.5, 0.5, -1.0, 1.0));
        assert!(!tail_bounds_separate_unsafe_moat(-1.0, 0.5, -1.0, 1.0));
        assert!(!tail_bounds_separate_unsafe_moat(-0.5, 1.0, -1.0, 1.0));

        let tenth = BigRational::new(1.into(), 10.into());
        let outward = exact_nonnegative_to_upper_f64(&tenth).unwrap();
        assert!(BigRational::from_float(outward).unwrap() >= tenth);
        assert!(BigRational::from_float(outward.next_down()).unwrap() < tenth);
        assert!(matches!(
            exact_nonnegative_to_upper_f64(&BigRational::new((-1).into(), 10.into())),
            Err(CganCzProbeDecline::OutputTail { .. })
        ));
    }

    #[test]
    fn limit_firewall_rejects_unprotected_latents_and_hard_limit_escalation() {
        let mut limits = qualification_limits();
        limits.retained_alpha_dim = 4;
        assert!(matches!(
            validate_limits(limits),
            Err(CganCzProbeDecline::InvalidLimit { .. })
        ));

        let mut limits = qualification_limits();
        limits.max_graph_nodes = RUNNER_HARD_MAX_GRAPH_NODES + 1;
        assert!(matches!(
            validate_limits(limits),
            Err(CganCzProbeDecline::InvalidLimit { .. })
        ));

        let two_symbols = ConstrainedZonotope64::try_new(
            vec![0.0],
            vec![vec![(0, 1.0)], vec![(0, -1.0)]],
            Array2::zeros((0, 2)),
            Vec::new(),
            vec![0.0],
        )
        .unwrap();
        let mut limits = qualification_limits();
        limits.max_transient_alpha_dim = 1;
        assert!(matches!(
            validate_retained_discriminator_domain(&two_symbols, limits),
            Err(CganCzProbeDecline::ResourceLimit {
                resource: "retained discriminator alpha dimension",
                required: 2,
                limit: 1,
            })
        ));
    }

    #[test]
    fn coordinator_preserves_deadline_poll_and_exact_peak_boundary() {
        let start = Instant::now();
        let deadline = start + Duration::from_secs(1);
        let mut expired = Coordinator::new(
            ConstrainedZonotopeCallBudget::new(deadline, 7, 32),
            move |_| deadline,
        );
        assert!(matches!(
            expired.checkpoint("test admission"),
            Err(CganCzProbeDecline::Budget(
                ConstrainedZonotopeCallBudgetError::DeadlineExpired {
                    checkpoint: "test admission"
                }
            ))
        ));
        assert_eq!(expired.deadline_polls, 1);
        assert_eq!(expired.peak_live_bytes, 7);

        let mut bounded = Coordinator::new(
            ConstrainedZonotopeCallBudget::new(deadline, 7, 32),
            move |_| start,
        );
        bounded.checkpoint("test admission").unwrap();
        bounded.preflight_absolute_peak(32).unwrap();
        assert_eq!(bounded.peak_live_bytes, 32);
        assert!(matches!(
            bounded.preflight_absolute_peak(33),
            Err(CganCzProbeDecline::Budget(
                ConstrainedZonotopeCallBudgetError::PeakLiveBytesExceeded {
                    required: 33,
                    limit: 32
                }
            ))
        ));
        assert_eq!(bounded.peak_live_bytes, 32);
    }

    // The default suite covers these algorithms with hermetic fixtures. The
    // explicit external-corpus lane must either exercise the real authored
    // assets or fail loudly; a selected conformance test is never a skip.
    #[cfg(feature = "external-vnncomp")]
    fn real_asset_root() -> PathBuf {
        let root = std::env::var_os("NY_CGAN_NCH1_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../benchmarks/vnncomp2025/benchmarks/cgan_2023")
            });
        let model = root.join("onnx/cGAN_imgSz32_nCh_1.onnx");
        assert!(
            model.is_file(),
            "external-vnncomp requires the cGAN real-model asset at {}; set \
             NY_CGAN_NCH1_ROOT to the cgan_2023 benchmark root",
            model.display()
        );
        root
    }

    #[cfg(feature = "external-vnncomp")]
    fn binary32_point_leaf(input: &CertifiedInputBox) -> [f32; PROTECTED_LATENT_SYMBOLS] {
        assert_eq!(input.len(), PROTECTED_LATENT_SYMBOLS);
        std::array::from_fn(|index| {
            let midpoint = (input.lower()[index] + input.upper()[index]) * 0.5;
            let point = midpoint as f32;
            assert!(f64::from(point) >= input.lower()[index]);
            assert!(f64::from(point) <= input.upper()[index]);
            point
        })
    }

    /// Load and authenticate one official regular imgSz32 generator before any
    /// graph slicing. The official properties are distributed as gzip streams;
    /// `load_vnnlib_with_certified_scalar_moat` is the repository's
    /// extension-aware loader and must exercise that path rather than a private
    /// decompression copy in this test.
    #[cfg(feature = "external-vnncomp")]
    fn authenticated_official_generator_fixture(
        profile: CganCzImgSz32Profile,
    ) -> (GraphNetwork, CertifiedInputBox) {
        let root = real_asset_root();
        let (model_name, property_name, expected_parameter_elements) = match profile {
            CganCzImgSz32Profile::Nch1 => (
                "cGAN_imgSz32_nCh_1.onnx",
                "cGAN_imgSz32_nCh_1_prop_1_input_eps_0.010_output_eps_0.015.vnnlib.gz",
                529_538,
            ),
            CganCzImgSz32Profile::Nch3 => (
                "cGAN_imgSz32_nCh_3.onnx",
                "cGAN_imgSz32_nCh_3_prop_2_input_eps_0.015_output_eps_0.020.vnnlib.gz",
                530_404,
            ),
        };
        let model_path = root.join("onnx").join(model_name);
        let property_path = root.join("vnnlib").join(property_name);
        assert!(
            model_path.is_file() && property_path.is_file(),
            "external-vnncomp requires the official imgSz32 {:?} model/property assets: {} / {}; set NY_CGAN_NCH1_ROOT to the cgan_2023 benchmark root",
            profile,
            model_path.display(),
            property_path.display(),
        );
        assert_eq!(
            property_path
                .extension()
                .and_then(|extension| extension.to_str()),
            Some("gz"),
            "the official-property test must retain the gzip loader boundary",
        );

        let config = OnnxLoadConfig::default()
            .with_batch_norm_folding_policy(BatchNormFoldingPolicy::PreserveRaw)
            .with_raw_float32_initializer_provenance(true);
        let model = load_onnx_with_config(&model_path, &config).unwrap_or_else(|error| {
            panic!(
                "failed to load official imgSz32 {:?} model {} with PreserveRaw/authored-f32 provenance: {error}",
                profile,
                model_path.display(),
            )
        });
        let graph = model.to_graph_network().unwrap_or_else(|error| {
            panic!(
                "failed to normalize official imgSz32 {:?} model {}: {error}",
                profile,
                model_path.display(),
            )
        });
        let (_, input, _) =
            load_vnnlib_with_certified_scalar_moat(&property_path).unwrap_or_else(|error| {
                panic!(
                    "failed to load official gzip property {}: {error}",
                    property_path.display(),
                )
            });

        let limits = match profile {
            CganCzImgSz32Profile::Nch1 => cgan_nch1_independent_interval_qualification_limits(),
            CganCzImgSz32Profile::Nch3 => cgan_nch3_independent_interval_qualification_limits(),
        };
        let mut coordinator = Coordinator::new(
            ConstrainedZonotopeCallBudget::new(
                Instant::now() + Duration::from_mins(2),
                64 << 20,
                2 << 30,
            ),
            |_| Instant::now(),
        );
        seal_topology_for_profile(profile, &model, &graph, limits, &mut coordinator)
            .unwrap_or_else(|error| {
                panic!(
                    "official imgSz32 {:?} topology seal declined for {}: {error}",
                    profile,
                    model_path.display(),
                )
            });
        let sealed = seal_parameters_for_profile(
            profile,
            &model,
            &graph,
            limits,
            RUNNER_TELEMETRY_RESERVED_BYTES,
            &mut coordinator,
        )
        .unwrap_or_else(|error| {
            panic!(
                "official imgSz32 {:?} authored-f32 parameter seal declined for {}: {error}",
                profile,
                model_path.display(),
            )
        });
        assert_eq!(sealed.layers.len(), EXPECTED_NODE_COUNT);
        assert_eq!(sealed.parameter_elements, expected_parameter_elements);
        (graph, input)
    }

    /// Clone an inclusive interval of the already authenticated unary chain.
    /// Only the first selected node is rewired to the slice input; every other
    /// edge and every authored layer object must match the full sealed graph.
    #[cfg(feature = "external-vnncomp")]
    fn authenticated_official_unary_slice(
        graph: &GraphNetwork,
        profile: CganCzImgSz32Profile,
        first: usize,
        last: usize,
    ) -> GraphNetwork {
        assert!(first <= last && last < EXPECTED_NODE_COUNT);
        let mut sliced = GraphNetwork::new();
        for index in first..=last {
            let name = EXPECTED_NODES[index].0;
            let original = graph
                .node(name)
                .unwrap_or_else(|| panic!("sealed graph is missing {name}"));
            let authored_input = if index == 0 {
                NETWORK_INPUT
            } else {
                EXPECTED_NODES[index - 1].0
            };
            assert_eq!(
                original.inputs(),
                [authored_input],
                "sealed graph edge into {name} changed before slicing",
            );
            let sliced_input = if index == first {
                NETWORK_INPUT
            } else {
                EXPECTED_NODES[index - 1].0
            };
            sliced
                .try_add_node(GraphNode::new(
                    name,
                    original.layer().clone(),
                    vec![sliced_input.to_string()],
                ))
                .unwrap_or_else(|error| panic!("failed to clone sealed node {name}: {error}"));
            sliced.set_declared_shape(
                name,
                expected_output_shape_for_profile(profile, index).to_vec(),
            );
        }
        sliced.set_output(EXPECTED_NODES[last].0);
        sliced.set_use_patches_mode(true);
        assert_eq!(sliced.num_nodes(), last - first + 1);
        sliced
    }

    #[cfg(feature = "external-vnncomp")]
    fn certified_input_box_as_f32(input: &CertifiedInputBox) -> BoundedTensor {
        assert_eq!(input.len(), PROTECTED_LATENT_SYMBOLS);
        let lower = input
            .lower()
            .iter()
            .copied()
            .map(cast_f64_to_f32_down)
            .collect::<Vec<_>>();
        let upper = input
            .upper()
            .iter()
            .copied()
            .map(cast_f64_to_f32_up)
            .collect::<Vec<_>>();
        for index in 0..PROTECTED_LATENT_SYMBOLS {
            assert!(lower[index].is_finite() && upper[index].is_finite());
            assert!(lower[index] <= upper[index]);
            assert!(f64::from(lower[index]) <= input.lower()[index]);
            assert!(f64::from(upper[index]) >= input.upper()[index]);
        }
        BoundedTensor::new(
            Array1::from_vec(lower).into_dyn(),
            Array1::from_vec(upper).into_dyn(),
        )
        .expect("the certified five-coordinate input box must remain ordered")
    }

    #[cfg(feature = "external-vnncomp")]
    fn binary32_point_tensor(input: &CertifiedInputBox) -> BoundedTensor {
        let point = Array1::from_vec(binary32_point_leaf(input).to_vec()).into_dyn();
        BoundedTensor::new(point.clone(), point)
            .expect("the certified binary32 midpoint must form a point tensor")
    }

    #[cfg(feature = "external-vnncomp")]
    fn assert_finite_ordered_tensor(
        context: &str,
        bounds: &BoundedTensor,
        expected_shape: &[usize],
    ) {
        assert_eq!(bounds.shape(), expected_shape, "{context} shape");
        for (index, (&lower, &upper)) in
            bounds.lower().iter().zip(bounds.upper().iter()).enumerate()
        {
            assert!(
                lower.is_finite() && upper.is_finite(),
                "{context} coordinate {index} is non-finite: [{lower}, {upper}]",
            );
            assert!(
                lower <= upper,
                "{context} coordinate {index} is reversed: [{lower}, {upper}]",
            );
        }
    }

    #[cfg(feature = "external-vnncomp")]
    fn assert_bitwise_equal_tensor(context: &str, left: &BoundedTensor, right: &BoundedTensor) {
        assert_eq!(left.shape(), right.shape(), "{context} shape");
        for (side, left_values, right_values) in [
            ("lower", left.lower(), right.lower()),
            ("upper", left.upper(), right.upper()),
        ] {
            for (index, (&left_value, &right_value)) in
                left_values.iter().zip(right_values.iter()).enumerate()
            {
                assert_eq!(
                    left_value.to_bits(),
                    right_value.to_bits(),
                    "{context} {side}[{index}]: {left_value} != {right_value}",
                );
            }
        }
    }

    #[cfg(feature = "external-vnncomp")]
    fn assert_official_finite_and_legacy_patches_generator_segment(profile: CganCzImgSz32Profile) {
        let (graph, input) = authenticated_official_generator_fixture(profile);

        // Inclusive authored indices: Gemm_0 -> Reshape_2 supplies the exact
        // segment box, while BatchNormalization_3 -> ConvTranspose_13 contains
        // all four generator transposed convolutions, including the terminal
        // stride-one operator this finite Patches route is meant to cover.
        let head = authenticated_official_unary_slice(&graph, profile, 0, 1);
        let segment = authenticated_official_unary_slice(&graph, profile, 2, 12);
        assert_eq!(head.node_names(), ["Gemm_0", "Reshape_2"]);
        assert_eq!(
            segment.node_names(),
            [
                "BatchNormalization_3",
                "ConvTranspose_4",
                "BatchNormalization_5",
                "Relu_6",
                "ConvTranspose_7",
                "BatchNormalization_8",
                "Relu_9",
                "ConvTranspose_10",
                "BatchNormalization_11",
                "Relu_12",
                "ConvTranspose_13",
            ],
        );

        let input_bounds = certified_input_box_as_f32(&input);
        let head_crown = head
            .propagate_crown_with_engine_and_deadline(&input_bounds, None, None)
            .expect("authenticated affine/reshape head CROWN must complete");
        assert_eq!(head_crown.provenance, BoundsProvenance::Crown);
        assert_finite_ordered_tensor(
            "authenticated generator head CROWN",
            &head_crown.bounds,
            expected_output_shape_for_profile(profile, 1),
        );

        // Supplying the identical finite-collected map to both calls keeps the
        // ReLU enclosures fixed while exercising both supported ConvTranspose
        // carrier implementations. They are not a bit-parity pair: finite
        // authority uses the directed-f64 Anchored planner and its coefficient
        // certificates, whereas no-deadline stride one deliberately preserves
        // the historical Affine equivalent-Conv2d route. The independently
        // evaluated official midpoint below is the shared soundness oracle.
        let node_bounds = segment
            .collect_node_bounds_with_engine_and_deadline(
                &head_crown.bounds,
                None,
                Some(Instant::now() + Duration::from_mins(10)),
            )
            .expect("finite official generator-segment forward bounds must complete");
        // Graph-CROWN grants a node at most 25% of its remaining outer budget.
        // The three-channel terminal transpose is three times as wide as the
        // nCh1 layer, so a 40-minute outer fixture budget gives that heavy node
        // an explicit 10-minute cooperative authority. This changes only the
        // external conformance fixture; the production budgeting policy stays
        // exercised unchanged.
        let finite_crown_outer_budget = match profile {
            CganCzImgSz32Profile::Nch1 => Duration::from_mins(10),
            CganCzImgSz32Profile::Nch3 => Duration::from_mins(40),
        };
        let finite = segment
            .propagate_crown_with_engine_and_deadline_and_node_bounds(
                &head_crown.bounds,
                None,
                Some(Instant::now() + finite_crown_outer_budget),
                Some(&node_bounds),
            )
            .expect("finite official generator-segment Patches CROWN must complete");
        assert_eq!(
            finite.provenance,
            BoundsProvenance::Crown,
            "finite generator-segment authority must not be an IBP fallback",
        );

        let legacy = segment
            .propagate_crown_with_engine_and_deadline_and_node_bounds(
                &head_crown.bounds,
                None,
                None,
                Some(&node_bounds),
            )
            .expect("no-deadline official generator-segment Patches CROWN must complete");
        assert_eq!(legacy.provenance, BoundsProvenance::Crown);

        let output_shape = expected_output_shape_for_profile(profile, 12);
        let point_input = binary32_point_tensor(&input);
        let point_head = head
            .propagate_concrete_point(&point_input, None, None)
            .expect("official midpoint head evaluation must complete");
        assert_finite_ordered_tensor(
            "official midpoint head",
            &point_head,
            expected_output_shape_for_profile(profile, 1),
        );
        let point_output = segment
            .propagate_concrete_point(&point_head, None, None)
            .expect("official midpoint generator-segment evaluation must complete");
        assert_finite_ordered_tensor("official midpoint segment", &point_output, output_shape);
        for (index, (&point_lower, &point_upper)) in point_output
            .lower()
            .iter()
            .zip(point_output.upper().iter())
            .enumerate()
        {
            assert_eq!(
                point_lower.to_bits(),
                point_upper.to_bits(),
                "concrete midpoint output {index} must be a point",
            );
        }
        for (context, bounds) in [
            ("finite Anchored generator-segment CROWN", &finite.bounds),
            ("legacy no-deadline generator-segment CROWN", &legacy.bounds),
        ] {
            assert_finite_ordered_tensor(context, bounds, output_shape);
            assert!(
                bounds
                    .lower()
                    .iter()
                    .zip(bounds.upper().iter())
                    .any(|(&lower, &upper)| lower < upper),
                "{context} must exercise a non-point official property",
            );
            for (index, ((&lower, &upper), (&point_lower, &point_upper))) in bounds
                .lower()
                .iter()
                .zip(bounds.upper().iter())
                .zip(point_output.lower().iter().zip(point_output.upper().iter()))
                .enumerate()
            {
                assert!(
                    lower <= point_lower && point_upper <= upper,
                    "{context} misses the official midpoint at {index}: [{lower}, {upper}] vs {point_lower}",
                );
            }
        }

        // Matrix mode is the negative control. Under finite authority it must
        // publish the precollected forward enclosure rather than silently
        // entering an unpolled Dense identity/backward allocation. The same
        // graph and input succeeding above with Crown provenance establishes
        // that this test really selected the Patches carrier.
        let mut dense_control = segment;
        dense_control.set_use_patches_mode(false);
        let dense = dense_control
            .propagate_crown_with_engine_and_deadline_and_node_bounds(
                &head_crown.bounds,
                None,
                Some(Instant::now() + finite_crown_outer_budget),
                Some(&node_bounds),
            )
            .expect("finite Dense negative control must fail closed to forward bounds");
        assert!(
            dense.is_fallback(),
            "finite Dense negative control unexpectedly acquired CROWN provenance",
        );
        let forward_output = node_bounds
            .get(EXPECTED_NODES[12].0)
            .expect("precollected map must contain the segment output");
        assert_bitwise_equal_tensor(
            "finite Dense negative-control forward fallback",
            &dense.bounds,
            forward_output,
        );
    }

    #[cfg(feature = "external-vnncomp")]
    #[test]
    fn real_property_input_domain_dispatches_every_protected_leaf_before_aggregate() {
        let root = real_asset_root();
        let (_, input, moat) = load_vnnlib_with_certified_scalar_moat(
            root.join("vnnlib/cGAN_imgSz32_nCh_1_prop_1_input_eps_0.010_output_eps_0.015.vnnlib"),
        )
        .unwrap();
        let sealed = SealedCgan {
            layers: Vec::new(),
            parameter_elements: 0,
            live_bytes: 0,
        };
        let start = Instant::now();
        let budget = protected_cover_budget(start);
        let mut input_coordinator = Coordinator::new(budget, move |_| start);
        let domain = build_input_domain(
            &input,
            qualification_limits(),
            &sealed,
            0,
            &mut input_coordinator,
        )
        .unwrap();
        let mut cover_limits = protected_cover_limits();
        cover_limits.bisection.max_constraint_count = 0;
        cover_limits.bisection.max_constraint_elements = 0;
        let cover = enumerate_cgan_nch1_protected_latent_cover_unwired(
            &domain,
            cover_limits,
            protected_cover_budget(Instant::now()),
        )
        .unwrap();

        let seen = RefCell::new(Vec::new());
        let mut coordinator = Coordinator::new(protected_cover_budget(start), move |_| start);
        let aggregate = propagate_cgan_cz_complete_cover_with(
            cover,
            &CGAN_NCH1_PROTECTED_LATENT_COVER_AXES,
            moat,
            PROTECTED_LATENT_LEAF_DOMAINS,
            0,
            &mut coordinator,
            |leaf_index, leaf, _, _| {
                seen.borrow_mut().push((leaf_index, leaf.center().to_vec()));
                Ok((
                    synthetic_completed_leaf_bounds(leaf_index, moat),
                    FULL_RUNNER_COMPLETED_STAGES,
                ))
            },
        )
        .unwrap();

        assert_eq!(seen.borrow().len(), PROTECTED_LATENT_LEAF_DOMAINS);
        assert_eq!(
            aggregate.leaf_completions.len(),
            PROTECTED_LATENT_LEAF_DOMAINS
        );
        assert_eq!(aggregate.cover.split_levels(), PROTECTED_LATENT_SYMBOLS);
        assert_eq!(aggregate.cover.split_calls(), 31);
        assert_eq!(
            aggregate.separates_unsafe_moat,
            tail_bounds_separate_unsafe_moat(
                aggregate.lower_bound,
                aggregate.upper_bound,
                moat.low_upper(),
                moat.high_lower(),
            )
        );
        for (expected, (leaf_index, center)) in seen.borrow().iter().enumerate() {
            assert_eq!(*leaf_index, expected);
            assert_eq!(center.len(), PROTECTED_LATENT_SYMBOLS);
            for (axis, &coordinate) in center.iter().enumerate() {
                let midpoint = (input.lower()[axis] + input.upper()[axis]) * 0.5;
                let quarter_width = (input.upper()[axis] - input.lower()[axis]) * 0.25;
                let positive = ((expected >> (PROTECTED_LATENT_SYMBOLS - axis - 1)) & 1) != 0;
                assert_eq!(
                    coordinate,
                    midpoint
                        + if positive {
                            quarter_width
                        } else {
                            -quarter_width
                        }
                );
            }
        }
    }

    #[cfg(feature = "external-vnncomp")]
    #[test]
    fn real_model_seals_topology_raw_float_provenance_and_detects_mutation() {
        let root = real_asset_root();
        let config = OnnxLoadConfig::default()
            .with_batch_norm_folding_policy(BatchNormFoldingPolicy::PreserveRaw)
            .with_raw_float32_initializer_provenance(true);
        let mut model =
            load_onnx_with_config(root.join("onnx/cGAN_imgSz32_nCh_1.onnx"), &config).unwrap();
        let graph = model.to_graph_network().unwrap();
        let start = Instant::now();
        let budget =
            ConstrainedZonotopeCallBudget::new(start + Duration::from_mins(1), 64 << 20, 2 << 30);
        let mut coordinator = Coordinator::new(budget, |_| Instant::now());
        coordinator.checkpoint("test admission").unwrap();
        let limits = qualification_limits();
        seal_topology(&model, &graph, limits, &mut coordinator).unwrap();
        let sealed = seal_parameters(&model, &graph, limits, 40 * 256, &mut coordinator).unwrap();
        assert_eq!(sealed.parameter_elements, 529_538);
        assert_eq!(sealed.layers.len(), 26);
        let SealedLayer::ConvTranspose2d(second_block_conv) = &sealed.layers[6] else {
            panic!("sealed node 6 must be ConvTranspose_7");
        };
        assert_eq!(second_block_conv.weights.dim(), (128, 64, 4, 4));
        assert_eq!(second_block_conv.bias.len(), 64);
        assert_eq!(second_block_conv.spec.stride, [2, 2]);
        assert_eq!(second_block_conv.spec.padding, [0, 0, 0, 0]);
        assert_eq!(second_block_conv.spec.dilation, [1, 1]);
        assert_eq!(second_block_conv.spec.output_padding, [0, 0]);
        assert_eq!(second_block_conv.spec.groups, 1);
        assert_eq!(12_544 * 128 * 4 * 4, 25_690_112);

        let SealedLayer::ConvTranspose2d(third_block_conv) = &sealed.layers[9] else {
            panic!("sealed node 9 must be ConvTranspose_10");
        };
        assert_eq!(third_block_conv.weights.dim(), (64, 32, 4, 4));
        assert_eq!(third_block_conv.bias.len(), 32);
        assert_eq!(third_block_conv.spec.stride, [2, 2]);
        assert_eq!(third_block_conv.spec.padding, [0, 0, 0, 0]);
        assert_eq!(third_block_conv.spec.dilation, [1, 1]);
        assert_eq!(third_block_conv.spec.output_padding, [0, 0]);
        assert_eq!(third_block_conv.spec.groups, 1);
        assert_eq!(28_800 * 64 * 4 * 4, 29_491_200);
        assert_eq!(12_544 * 32 * 4 * 4, 6_422_528);
        assert_eq!(38_325 * 32 * 4 * 4, 19_622_400);
        assert_eq!(6_422_528 + 6_422_528 + 19_622_400, 32_467_456);

        let SealedLayer::ConvTranspose2d(handoff_conv_transpose) = &sealed.layers[12] else {
            panic!("sealed node 12 must be ConvTranspose_13");
        };
        assert_eq!(handoff_conv_transpose.weights.dim(), (32, 1, 3, 3));
        assert_eq!(handoff_conv_transpose.bias.len(), 1);
        assert_eq!(handoff_conv_transpose.spec.stride, [1, 1]);
        assert_eq!(handoff_conv_transpose.spec.padding, [0, 0, 0, 0]);
        assert_eq!(handoff_conv_transpose.spec.dilation, [1, 1]);
        assert_eq!(handoff_conv_transpose.spec.output_padding, [0, 0]);
        assert_eq!(handoff_conv_transpose.spec.groups, 1);
        assert_eq!(
            handoff_conv_transpose
                .weights
                .iter()
                .filter(|&&weight| weight != 0.0)
                .count(),
            288
        );
        assert_eq!(32 * 32 * 32 * 3 * 3, 294_912);

        let SealedLayer::Conv2d(handoff_conv) = &sealed.layers[13] else {
            panic!("sealed node 13 must be Conv_14");
        };
        assert_eq!(handoff_conv.weights.dim(), (16, 1, 3, 3));
        assert_eq!(handoff_conv.bias.len(), 16);
        assert_eq!(handoff_conv.spec.stride, [2, 2]);
        assert_eq!(handoff_conv.spec.padding, [1, 1, 1, 1]);
        assert_eq!(handoff_conv.spec.dilation, [1, 1]);
        assert_eq!(handoff_conv.spec.groups, 1);
        assert_eq!(
            handoff_conv
                .weights
                .iter()
                .filter(|&&weight| weight != 0.0)
                .count(),
            144
        );
        assert_eq!(16 * 16 * 16 * 3 * 3, 36_864);
        assert!(matches!(sealed.layers[14], SealedLayer::Relu));

        for (index, raw) in model.network.layers.iter().enumerate() {
            if EXPECTED_NODES[index].1 == LayerType::BatchNorm {
                assert_eq!(
                    raw.attributes.get(ONNX_BATCH_NORM_INPUT_RANK_ATTR),
                    Some(&AttributeValue::Int(CGAN_AUTHORED_TENSOR_RANK as i64))
                );
            }
        }
        let original_rank = model.network.layers[2].attributes.insert(
            ONNX_BATCH_NORM_INPUT_RANK_ATTR.to_string(),
            AttributeValue::Int((CGAN_AUTHORED_TENSOR_RANK - 1) as i64),
        );
        assert!(matches!(
            seal_parameters(&model, &graph, limits, 40 * 256, &mut coordinator),
            Err(CganCzProbeDecline::Topology { .. })
        ));
        model.network.layers[2].attributes.insert(
            ONNX_BATCH_NORM_INPUT_RANK_ATTR.to_string(),
            original_rank.expect("the loader must attach BatchNorm rank provenance"),
        );

        model.network.layers[0].outputs[0].push_str("_mutated");
        assert!(matches!(
            seal_topology(&model, &graph, limits, &mut coordinator),
            Err(CganCzProbeDecline::Provenance { .. })
        ));
    }

    #[cfg(feature = "external-vnncomp")]
    #[test]
    fn real_nch1_finite_and_legacy_patches_generator_segment_enclose_midpoint() {
        assert_official_finite_and_legacy_patches_generator_segment(CganCzImgSz32Profile::Nch1);
    }

    #[cfg(feature = "external-vnncomp")]
    #[test]
    fn real_nch3_finite_and_legacy_patches_generator_segment_enclose_midpoint() {
        assert_official_finite_and_legacy_patches_generator_segment(CganCzImgSz32Profile::Nch3);
    }

    #[cfg(feature = "external-vnncomp")]
    #[test]
    fn real_model_independent_interval_lane_is_complete_deterministic_and_fail_closed() {
        let root = real_asset_root();
        let config = OnnxLoadConfig::default()
            .with_batch_norm_folding_policy(BatchNormFoldingPolicy::PreserveRaw)
            .with_raw_float32_initializer_provenance(true);
        let model =
            load_onnx_with_config(root.join("onnx/cGAN_imgSz32_nCh_1.onnx"), &config).unwrap();
        let graph = model.to_graph_network().unwrap();
        let (_, input, moat) = load_vnnlib_with_certified_scalar_moat(
            root.join("vnnlib/cGAN_imgSz32_nCh_1_prop_1_input_eps_0.010_output_eps_0.015.vnnlib"),
        )
        .unwrap();
        let limits = cgan_nch1_independent_interval_qualification_limits();
        let baseline = 64 << 20;
        let run = || {
            probe_cgan_nch1_independent_interval_unwired(
                &model,
                &graph,
                &input,
                limits,
                ConstrainedZonotopeCallBudget::new(
                    Instant::now() + Duration::from_mins(2),
                    baseline,
                    512 << 20,
                ),
            )
        };

        let first = run();
        let second = run();
        assert_eq!(first, second);
        assert_eq!(first.authority, CGAN_CZ_VERDICT_AUTHORITY);
        assert_eq!(first.profile, CganCzImgSz32Profile::Nch1);
        assert_eq!(first.image_channels(), 1);
        assert_eq!(first.topology_work_items, 837);
        assert_eq!(first.parameter_elements, 529_538);
        assert!(first.peak_live_bytes > baseline);
        assert!(first.charged_items > 0);
        assert!(first.deadline_polls > 0);
        let CganCzIndependentIntervalStatus::Completed(completion) = &first.status else {
            panic!("independent interval lane declined: {:?}", first.status);
        };
        assert_eq!(completion.relu_bounds().len(), CGAN_RELU_COUNT);
        for (record, index) in completion.relu_bounds().iter().zip(CGAN_RELU_INDICES) {
            assert_eq!(record.node(), EXPECTED_NODES[index].0);
            assert_eq!(record.output_shape(), expected_output_shape(index));
            assert_eq!(
                record.bounds().value_dim(),
                checked_product(expected_output_shape(index), "test independent ReLU shape")
                    .unwrap()
            );
        }
        assert_eq!(
            completion.final_relu_23_auxiliary_bounds().value_dim(),
            TAIL_VALUE_DIM
        );
        assert_eq!(completion.post_relu_23_domain().value_dim(), TAIL_VALUE_DIM);
        assert_eq!(completion.post_relu_23_domain().alpha_dim(), 0);
        assert_eq!(completion.post_relu_23_domain().constraint_count(), 0);

        let point = binary32_point_leaf(&input);
        let leaf_rows = bound_cgan_imgsz32_leaf_rows_unwired(
            CganCzImgSz32Profile::Nch1,
            &model,
            &graph,
            &point,
            &point,
            moat,
            limits,
            ConstrainedZonotopeCallBudget::new(
                Instant::now() + Duration::from_mins(3),
                baseline,
                768 << 20,
            ),
        );
        assert_eq!(leaf_rows.authority, CGAN_CZ_VERDICT_AUTHORITY);
        assert_eq!(leaf_rows.profile, CganCzImgSz32Profile::Nch1);
        assert_eq!(leaf_rows.baseline_live_bytes, baseline);
        assert_eq!(leaf_rows.max_peak_live_bytes, 768 << 20);
        assert!(leaf_rows.peak_live_bytes > baseline);
        assert!(leaf_rows.deadline_polls > 0);
        let CganCzLeafRowStatus::Completed(rows) = &leaf_rows.status else {
            panic!(
                "nCh1 point-leaf row bounds declined: {:?}",
                leaf_rows.status
            );
        };
        assert!(rows.lower_y.is_finite());
        assert!(rows.lower_neg_y.is_finite());
        assert!(rows.lower_y <= -rows.lower_neg_y);
        assert!(rows.bn_tail_correction_upper.is_finite());
        assert!(rows.bn_tail_correction_upper >= 0.0);
        assert_eq!(rows.lower_m20_status, CganCzM20Status::Completed);
        assert_eq!(rows.negated_upper_m20_status, CganCzM20Status::Completed);
        assert!(rows.lower_m24_measurement.is_some());
        assert!(rows.negated_upper_m24_measurement.is_some());
        assert!(matches!(
            rows.lower_depth_two_measurement,
            CganCzDepthTwoMeasurement::NotRequested
        ));
        assert!(matches!(
            rows.negated_upper_depth_two_measurement,
            CganCzDepthTwoMeasurement::NotRequested
        ));
        assert_eq!(
            Some(rows.lower_y),
            select_m17_m20_lower_bound(
                rows.lower_m17_candidates.selected_lower_bound,
                rows.lower_m20_lower_bound,
                rows.lower_m20_status,
            )
        );
        assert_eq!(
            Some(rows.lower_neg_y),
            select_m17_m20_lower_bound(
                rows.negated_upper_m17_candidates.selected_lower_bound,
                rows.negated_upper_m20_lower_bound,
                rows.negated_upper_m20_status,
            )
        );

        let leaf_one_byte_low = bound_cgan_imgsz32_leaf_rows_unwired(
            CganCzImgSz32Profile::Nch1,
            &model,
            &graph,
            &point,
            &point,
            moat,
            limits,
            ConstrainedZonotopeCallBudget::new(
                Instant::now() + Duration::from_mins(3),
                baseline,
                leaf_rows.peak_live_bytes - 1,
            ),
        );
        let CganCzLeafRowStatus::Completed(fallback_rows) = &leaf_one_byte_low.status else {
            panic!(
                "one-byte-low optional portfolio peak must retain M17 authority: {:?}",
                leaf_one_byte_low.status
            );
        };
        let lower_m24 = fallback_rows.lower_m24_measurement.as_ref().unwrap();
        let negated_upper_m24 = fallback_rows
            .negated_upper_m24_measurement
            .as_ref()
            .unwrap();
        let assert_optional_row_fallback =
            |selected_lower_bound: f64,
             m17_lower_bound: f64,
             m20_lower_bound: Option<f64>,
             m20_status: CganCzM20Status| {
                assert_ne!(m20_status, CganCzM20Status::NotRequested);
                assert_eq!(
                    m20_lower_bound.is_some(),
                    m20_status == CganCzM20Status::Completed
                );
                assert_eq!(
                    Some(selected_lower_bound),
                    select_m17_m20_lower_bound(m17_lower_bound, m20_lower_bound, m20_status)
                );
                assert!(selected_lower_bound.is_finite());
            };
        assert_optional_row_fallback(
            fallback_rows.lower_y,
            fallback_rows.lower_m17_candidates.selected_lower_bound,
            fallback_rows.lower_m20_lower_bound,
            fallback_rows.lower_m20_status,
        );
        assert_optional_row_fallback(
            fallback_rows.lower_neg_y,
            fallback_rows
                .negated_upper_m17_candidates
                .selected_lower_bound,
            fallback_rows.negated_upper_m20_lower_bound,
            fallback_rows.negated_upper_m20_status,
        );
        let m24_peak_fallback = |measurement: &CganCzM24Measurement| {
            matches!(
                measurement.optional_budget_error.as_ref(),
                Some(ConstrainedZonotopeCallBudgetError::PeakLiveBytesExceeded { .. })
            )
        };
        assert!(matches!(
            fallback_rows.lower_depth_two_measurement,
            CganCzDepthTwoMeasurement::NotRequested
        ));
        assert!(matches!(
            fallback_rows.negated_upper_depth_two_measurement,
            CganCzDepthTwoMeasurement::NotRequested
        ));
        assert!(
            m24_peak_fallback(lower_m24) || m24_peak_fallback(negated_upper_m24),
            "one-byte-low peak must decline at least one enabled optional M24 replay: lower_m24={lower_m24:?} negated_upper_m24={negated_upper_m24:?}",
        );
        assert!(leaf_one_byte_low.peak_live_bytes < leaf_rows.peak_live_bytes);

        let one_byte_low = probe_cgan_nch1_independent_interval_unwired(
            &model,
            &graph,
            &input,
            limits,
            ConstrainedZonotopeCallBudget::new(
                Instant::now() + Duration::from_mins(2),
                baseline,
                first.peak_live_bytes - 1,
            ),
        );
        assert!(matches!(
            one_byte_low.status,
            CganCzIndependentIntervalStatus::Declined {
                reason: CganCzProbeDecline::Budget(
                    ConstrainedZonotopeCallBudgetError::PeakLiveBytesExceeded { required, limit }
                ),
                ..
            } if required == first.peak_live_bytes && limit + 1 == required
        ));

        let start = Instant::now();
        let deadline = start + Duration::from_mins(2);
        let late = probe_cgan_nch1_independent_interval_with_clock(
            &model,
            &graph,
            &input,
            limits,
            ConstrainedZonotopeCallBudget::new(deadline, baseline, first.peak_live_bytes),
            |checkpoint| {
                if checkpoint == "cGAN independent interval publication" {
                    deadline
                } else {
                    start
                }
            },
        );
        assert!(matches!(
            late.status,
            CganCzIndependentIntervalStatus::Declined {
                node: "Relu_23",
                reason: CganCzProbeDecline::Budget(
                    ConstrainedZonotopeCallBudgetError::DeadlineExpired {
                        checkpoint: "cGAN independent interval publication"
                    }
                )
            }
        ));
    }

    #[cfg(feature = "external-vnncomp")]
    #[test]
    fn real_nch3_prop2_independent_interval_lane_uses_exact_profile_and_mixed_point_input() {
        let root = real_asset_root();
        let model_path = root.join("onnx/cGAN_imgSz32_nCh_3.onnx");
        let property_path = root
            .join("vnnlib/cGAN_imgSz32_nCh_3_prop_2_input_eps_0.015_output_eps_0.020.vnnlib.gz");
        assert!(
            model_path.is_file() && property_path.is_file(),
            "external-vnncomp requires the imgSz32 nCh3 qualification assets: {} / {}",
            model_path.display(),
            property_path.display()
        );

        let config = OnnxLoadConfig::default()
            .with_batch_norm_folding_policy(BatchNormFoldingPolicy::PreserveRaw)
            .with_raw_float32_initializer_provenance(true);
        let model = load_onnx_with_config(&model_path, &config).unwrap();
        let graph = model.to_graph_network().unwrap();
        let (_, input, moat) = load_vnnlib_with_certified_scalar_moat(&property_path).unwrap();
        assert_eq!(input.len(), PROTECTED_LATENT_SYMBOLS);
        assert!(input.declared_point().iter().any(|&point| point));

        let limits = cgan_nch3_independent_interval_qualification_limits();
        let baseline = 64 << 20;
        let report = probe_cgan_nch3_independent_interval_unwired(
            &model,
            &graph,
            &input,
            limits,
            ConstrainedZonotopeCallBudget::new(
                Instant::now() + Duration::from_mins(3),
                baseline,
                512 << 20,
            ),
        );
        assert_eq!(report.authority, CGAN_CZ_VERDICT_AUTHORITY);
        assert_eq!(report.profile, CganCzImgSz32Profile::Nch3);
        assert_eq!(report.image_channels(), 3);
        assert_eq!(report.topology_work_items, 837);
        assert_eq!(report.parameter_elements, 530_404);
        assert!(report.peak_live_bytes > baseline);
        let CganCzIndependentIntervalStatus::Completed(completion) = &report.status else {
            panic!(
                "nCh3 independent interval lane declined: {:?}",
                report.status
            );
        };
        assert_eq!(completion.relu_bounds().len(), CGAN_RELU_COUNT);
        for (record, index) in completion.relu_bounds().iter().zip(CGAN_RELU_INDICES) {
            let expected_shape =
                expected_output_shape_for_profile(CganCzImgSz32Profile::Nch3, index);
            assert_eq!(record.node(), EXPECTED_NODES[index].0);
            assert_eq!(record.output_shape(), expected_shape);
            assert_eq!(
                record.bounds().value_dim(),
                checked_product(expected_shape, "test nCh3 independent ReLU shape").unwrap()
            );
        }
        assert_eq!(
            completion.final_relu_23_auxiliary_bounds().value_dim(),
            TAIL_VALUE_DIM
        );
        assert_eq!(completion.post_relu_23_domain().value_dim(), TAIL_VALUE_DIM);
        assert_eq!(completion.post_relu_23_domain().alpha_dim(), 0);
        assert_eq!(completion.post_relu_23_domain().constraint_count(), 0);

        let point = binary32_point_leaf(&input);
        let leaf_rows = bound_cgan_imgsz32_leaf_rows_unwired(
            CganCzImgSz32Profile::Nch3,
            &model,
            &graph,
            &point,
            &point,
            moat,
            limits,
            ConstrainedZonotopeCallBudget::new(
                Instant::now() + Duration::from_mins(3),
                baseline,
                768 << 20,
            ),
        );
        assert_eq!(leaf_rows.profile, CganCzImgSz32Profile::Nch3);
        assert_eq!(leaf_rows.parameter_elements, 530_404);
        let CganCzLeafRowStatus::Completed(rows) = &leaf_rows.status else {
            panic!(
                "nCh3 point-leaf row bounds declined: {:?}",
                leaf_rows.status
            );
        };
        assert!(rows.lower_y.is_finite());
        assert!(rows.lower_neg_y.is_finite());
        assert!(rows.lower_y <= -rows.lower_neg_y);

        let wrong_profile = probe_cgan_nch1_independent_interval_unwired(
            &model,
            &graph,
            &input,
            cgan_nch1_independent_interval_qualification_limits(),
            ConstrainedZonotopeCallBudget::new(
                Instant::now() + Duration::from_mins(1),
                baseline,
                512 << 20,
            ),
        );
        assert_eq!(wrong_profile.profile, CganCzImgSz32Profile::Nch1);
        assert!(matches!(
            wrong_profile.status,
            CganCzIndependentIntervalStatus::Declined {
                node: "topology",
                reason: CganCzProbeDecline::Topology { .. },
            }
        ));
    }

    #[cfg(feature = "external-vnncomp")]
    #[test]
    fn real_model_second_block_completes_diagnostic_only_and_declines_small_cap() {
        let root = real_asset_root();
        let config = OnnxLoadConfig::default()
            .with_batch_norm_folding_policy(BatchNormFoldingPolicy::PreserveRaw)
            .with_raw_float32_initializer_provenance(true);
        let model =
            load_onnx_with_config(root.join("onnx/cGAN_imgSz32_nCh_1.onnx"), &config).unwrap();
        let graph = model.to_graph_network().unwrap();
        let (_, input, moat) = load_vnnlib_with_certified_scalar_moat(
            root.join("vnnlib/cGAN_imgSz32_nCh_1_prop_1_input_eps_0.010_output_eps_0.015.vnnlib"),
        )
        .unwrap();
        let budget = ConstrainedZonotopeCallBudget::new(
            Instant::now() + Duration::from_mins(3),
            64 << 20,
            2 << 30,
        );
        let report = probe_cgan_nch1_second_block_unwired(
            &model,
            &graph,
            &input,
            moat,
            qualification_limits(),
            budget,
        );
        assert_eq!(report.authority, CGAN_CZ_VERDICT_AUTHORITY);
        assert_eq!(report.topology_work_items, 837);
        assert_eq!(report.parameter_elements, 529_538);
        assert_eq!(report.peak_live_bytes, 910_980_760);
        assert_eq!(report.charged_items, 63_212_584);
        assert_eq!(report.deadline_polls, 96_648);
        assert_eq!(report.stages.len(), 11);
        assert_eq!(report.stages[7].node, "ConvTranspose_7");
        assert_eq!(report.stages[7].output_shape, [64, 14, 14]);
        assert_eq!(report.stages[7].output_generator_nonzeros, 62_720);
        assert_eq!(report.stages[9].node, "Relu_9");
        assert_eq!(report.stages[9].unstable_coordinates, 245);
        assert_eq!(report.stages[9].output_alpha_dim, 250);
        assert_eq!(report.stages[10].output_alpha_dim, 5);
        assert_eq!(report.stages[10].output_generator_nonzeros, 38_325);
        let CganCzProbeStatus::PrefixCompleted(completion) = &report.status else {
            panic!("second-block qualification must complete only a diagnostic prefix");
        };
        assert_eq!(completion.last_node, "Relu_9");
        assert_eq!(completion.output_shape, [64, 14, 14]);
        assert_eq!(completion.value_dim, 12_544);
        assert_eq!(completion.alpha_dim, PROTECTED_LATENT_SYMBOLS);
        assert_eq!(completion.generator_nonzeros, 38_325);

        let mut too_small = qualification_limits();
        too_small.max_value_dim = 4_608;
        too_small.max_transient_alpha_dim = 4_613;
        let budget = ConstrainedZonotopeCallBudget::new(
            Instant::now() + Duration::from_mins(3),
            64 << 20,
            2 << 30,
        );
        let declined =
            probe_cgan_nch1_second_block_unwired(&model, &graph, &input, moat, too_small, budget);
        assert_eq!(declined.authority, CGAN_CZ_VERDICT_AUTHORITY);
        assert!(matches!(
            declined.status,
            CganCzProbeStatus::Declined {
                node: "ConvTranspose_7",
                reason: CganCzProbeDecline::Transform {
                    operation: "ConvTranspose2d",
                    ..
                }
            }
        ));
    }

    #[cfg(feature = "external-vnncomp")]
    #[test]
    fn real_model_third_block_is_exact_deterministic_and_declines_alpha_cap() {
        let root = real_asset_root();
        let config = OnnxLoadConfig::default()
            .with_batch_norm_folding_policy(BatchNormFoldingPolicy::PreserveRaw)
            .with_raw_float32_initializer_provenance(true);
        let model =
            load_onnx_with_config(root.join("onnx/cGAN_imgSz32_nCh_1.onnx"), &config).unwrap();
        let graph = model.to_graph_network().unwrap();
        let (_, input, moat) = load_vnnlib_with_certified_scalar_moat(
            root.join("vnnlib/cGAN_imgSz32_nCh_1_prop_1_input_eps_0.010_output_eps_0.015.vnnlib"),
        )
        .unwrap();

        let run = || {
            let budget = ConstrainedZonotopeCallBudget::new(
                Instant::now() + Duration::from_mins(1),
                64 << 20,
                2 << 30,
            );
            probe_cgan_nch1_third_block_unwired(
                &model,
                &graph,
                &input,
                moat,
                cgan_nch1_third_block_qualification_limits(),
                budget,
            )
        };
        let first = run();
        let second = run();
        assert_eq!(first, second);
        assert_eq!(first.authority, CGAN_CZ_VERDICT_AUTHORITY);
        assert_eq!(first.topology_work_items, 837);
        assert_eq!(first.parameter_elements, 529_538);
        assert_eq!(first.peak_live_bytes, 243_275_016);
        assert_eq!(first.charged_items, 127_191_192);
        assert_eq!(first.stages.len(), 15);
        assert_eq!(first.stages[11].node, "ConvTranspose_10");
        assert_eq!(first.stages[11].output_shape, [32, 30, 30]);
        assert_eq!(first.stages[11].output_generator_nonzeros, 144_000);
        assert_eq!(first.stages[12].node, "BatchNormalization_11");
        assert_eq!(first.stages[12].output_generator_nonzeros, 144_000);
        assert_eq!(first.stages[13].node, "Relu_12");
        assert_eq!(first.stages[13].unstable_coordinates, 454);
        assert_eq!(first.stages[13].output_alpha_dim, 459);
        assert_eq!(first.stages[13].output_generator_nonzeros, 78_379);
        assert_eq!(first.stages[14].output_alpha_dim, PROTECTED_LATENT_SYMBOLS);
        assert_eq!(first.stages[14].output_generator_nonzeros, 77_925);
        let CganCzProbeStatus::PrefixCompleted(completion) = &first.status else {
            panic!("third-block qualification must complete only a diagnostic prefix");
        };
        assert_eq!(completion.last_node, "Relu_12");
        assert_eq!(completion.output_shape, [32, 30, 30]);
        assert_eq!(completion.value_dim, 28_800);
        assert_eq!(completion.alpha_dim, PROTECTED_LATENT_SYMBOLS);
        assert_eq!(completion.generator_nonzeros, 77_925);

        let mut too_small = cgan_nch1_third_block_qualification_limits();
        too_small.max_transient_alpha_dim = 458;
        let budget = ConstrainedZonotopeCallBudget::new(
            Instant::now() + Duration::from_mins(1),
            64 << 20,
            2 << 30,
        );
        let declined =
            probe_cgan_nch1_third_block_unwired(&model, &graph, &input, moat, too_small, budget);
        assert_eq!(declined.authority, CGAN_CZ_VERDICT_AUTHORITY);
        assert!(matches!(
            declined.status,
            CganCzProbeStatus::Declined {
                node: "Relu_12",
                reason: CganCzProbeDecline::Transform {
                    operation: "ReLU",
                    ..
                }
            }
        ));
    }

    #[cfg(feature = "external-vnncomp")]
    #[test]
    fn real_model_handoff_is_exact_deterministic_and_declines_alpha_cap() {
        let root = real_asset_root();
        let config = OnnxLoadConfig::default()
            .with_batch_norm_folding_policy(BatchNormFoldingPolicy::PreserveRaw)
            .with_raw_float32_initializer_provenance(true);
        let model =
            load_onnx_with_config(root.join("onnx/cGAN_imgSz32_nCh_1.onnx"), &config).unwrap();
        let graph = model.to_graph_network().unwrap();
        let (_, input, moat) = load_vnnlib_with_certified_scalar_moat(
            root.join("vnnlib/cGAN_imgSz32_nCh_1_prop_1_input_eps_0.010_output_eps_0.015.vnnlib"),
        )
        .unwrap();

        let run = || {
            let budget = ConstrainedZonotopeCallBudget::new(
                Instant::now() + Duration::from_mins(1),
                64 << 20,
                2 << 30,
            );
            probe_cgan_nch1_generator_discriminator_handoff_unwired(
                &model,
                &graph,
                &input,
                moat,
                cgan_nch1_generator_discriminator_handoff_qualification_limits(),
                budget,
            )
        };
        let first = run();
        let second = run();
        assert_eq!(first, second);
        assert_eq!(first.authority, CGAN_CZ_VERDICT_AUTHORITY);
        assert_eq!(first.topology_work_items, 837);
        assert_eq!(first.parameter_elements, 529_538);
        assert_eq!(first.peak_live_bytes, 243_275_016);
        assert_eq!(first.charged_items, 129_217_773);
        assert_eq!(first.deadline_polls, 311_838);
        assert_eq!(first.stages.len(), 19);

        let conv_transpose = &first.stages[15];
        assert_eq!(conv_transpose.node, "ConvTranspose_13");
        assert_eq!(conv_transpose.kind, CganCzStageKind::ConvTranspose2d);
        assert_eq!(conv_transpose.output_shape, [1, 32, 32]);
        assert_eq!(conv_transpose.input_alpha_dim, 5);
        assert_eq!(conv_transpose.output_alpha_dim, 5);
        assert_eq!(conv_transpose.input_generator_nonzeros, 77_925);
        assert_eq!(conv_transpose.output_generator_nonzeros, 5_120);
        assert_eq!(conv_transpose.peak_live_bytes, 75_426_776);
        assert_eq!(conv_transpose.charged_items, 1_680_022);

        let conv = &first.stages[16];
        assert_eq!(conv.node, "Conv_14");
        assert_eq!(conv.kind, CganCzStageKind::Conv2d);
        assert_eq!(conv.output_shape, [16, 16, 16]);
        assert_eq!(conv.input_alpha_dim, 5);
        assert_eq!(conv.output_alpha_dim, 5);
        assert_eq!(conv.input_generator_nonzeros, 5_120);
        assert_eq!(conv.output_generator_nonzeros, 20_480);
        assert_eq!(conv.peak_live_bytes, 72_304_440);
        assert_eq!(conv.charged_items, 347_406);

        let relu = &first.stages[17];
        assert_eq!(relu.node, "Relu_15");
        assert_eq!(relu.kind, CganCzStageKind::Relu);
        assert_eq!(relu.output_shape, [16, 16, 16]);
        assert_eq!(relu.input_alpha_dim, 5);
        assert_eq!(relu.output_alpha_dim, 72);
        assert_eq!(relu.input_generator_nonzeros, 20_480);
        assert_eq!(relu.output_generator_nonzeros, 9_032);
        assert_eq!(relu.unstable_coordinates, 67);
        assert_eq!(relu.peak_live_bytes, 210_863_368);
        assert_eq!(relu.charged_items, 78_818);

        let reduction = &first.stages[18];
        assert_eq!(reduction.node, "Relu_15");
        assert_eq!(reduction.kind, CganCzStageKind::OrderReduction);
        assert_eq!(reduction.output_shape, [16, 16, 16]);
        assert_eq!(reduction.input_alpha_dim, 72);
        assert_eq!(reduction.output_alpha_dim, PROTECTED_LATENT_SYMBOLS);
        assert_eq!(reduction.input_generator_nonzeros, 9_032);
        assert_eq!(reduction.output_generator_nonzeros, 8_965);
        assert_eq!(reduction.discarded_generators, 67);
        assert_eq!(reduction.peak_live_bytes, 71_936_904);
        assert_eq!(reduction.charged_items, 38_703);

        let CganCzProbeStatus::PrefixCompleted(completion) = &first.status else {
            panic!("handoff qualification must complete only a diagnostic prefix");
        };
        assert_eq!(completion.last_node, "Relu_15");
        assert_eq!(completion.output_shape, [16, 16, 16]);
        assert_eq!(completion.value_dim, 4_096);
        assert_eq!(completion.alpha_dim, PROTECTED_LATENT_SYMBOLS);
        assert_eq!(completion.generator_nonzeros, 8_965);
        assert_eq!(
            completion.maximum_coordinate_width,
            2.141_469_640_611_119e-1
        );
        assert_eq!(completion.mean_coordinate_width, 1.094_751_738_888_092_4e-2);
        assert_eq!(completion.maximum_box_remainder, 4.310_982_385_593_923e-2);

        // The cap is global and fail-closed. Tightening it below the 459
        // symbols required by the preceding qualified Relu_12 correctly
        // declines before the handoff rather than granting a stage exception.
        let mut too_small = cgan_nch1_generator_discriminator_handoff_qualification_limits();
        too_small.max_transient_alpha_dim = 458;
        let budget = ConstrainedZonotopeCallBudget::new(
            Instant::now() + Duration::from_mins(1),
            64 << 20,
            2 << 30,
        );
        let declined = probe_cgan_nch1_generator_discriminator_handoff_unwired(
            &model, &graph, &input, moat, too_small, budget,
        );
        assert_eq!(declined.authority, CGAN_CZ_VERDICT_AUTHORITY);
        assert!(matches!(
            declined.status,
            CganCzProbeStatus::Declined {
                node: "Relu_12",
                reason: CganCzProbeDecline::Transform {
                    operation: "ReLU",
                    ..
                }
            }
        ));
    }

    #[cfg(feature = "external-vnncomp")]
    #[test]
    fn real_model_prop1_full_tail_is_exact_and_diagnostic_only() {
        let root = real_asset_root();
        let config = OnnxLoadConfig::default()
            .with_batch_norm_folding_policy(BatchNormFoldingPolicy::PreserveRaw)
            .with_raw_float32_initializer_provenance(true);
        let model =
            load_onnx_with_config(root.join("onnx/cGAN_imgSz32_nCh_1.onnx"), &config).unwrap();
        let graph = model.to_graph_network().unwrap();
        let (_, input, moat) = load_vnnlib_with_certified_scalar_moat(
            root.join("vnnlib/cGAN_imgSz32_nCh_1_prop_1_input_eps_0.010_output_eps_0.015.vnnlib"),
        )
        .unwrap();
        let budget = ConstrainedZonotopeCallBudget::new(
            Instant::now() + Duration::from_mins(3),
            64 << 20,
            2 << 30,
        );
        let report = probe_cgan_nch1_sequential_unwired(
            &model,
            &graph,
            &input,
            moat,
            cgan_nch1_generator_discriminator_handoff_qualification_limits(),
            budget,
        );
        assert_eq!(report.authority, CGAN_CZ_VERDICT_AUTHORITY);
        let CganCzProbeStatus::Completed(bounds) = &report.status else {
            panic!("full diagnostic tail declined: {:?}", report.status);
        };
        assert_eq!(bounds.lower_bound, -2.107_576_703_444_927_5);
        assert_eq!(bounds.upper_bound, 4.584_528_029_148_942_5);
        assert_eq!(bounds.bn_tail_correction_upper, 2.484_006_936_361_026e-7);
        assert!(!bounds.separates_unsafe_moat);
        assert_eq!(bounds.lower_m17_status, ReluTailDualStatus::Completed);
        assert_eq!(bounds.upper_m17_status, ReluTailDualStatus::Completed);
        assert_eq!(report.stages.len(), 29);
        let final_reduction = &report.stages[23];
        assert_eq!(final_reduction.node, "Relu_20");
        assert_eq!(final_reduction.kind, CganCzStageKind::OrderReduction);
        assert_eq!(final_reduction.output_alpha_dim, 512);
        assert_eq!(final_reduction.discarded_generators, 229);
        assert_eq!(report.stages[25].node, "Conv_22");
        assert_eq!(report.stages[25].output_generator_nonzeros, 119_424);
        let construction = &report.stages[26];
        assert_eq!(construction.kind, CganCzStageKind::OutputTailConstruction);
        assert_eq!(construction.output_shape, [1]);
        assert_eq!(construction.output_alpha_dim, 0);
        assert_eq!(construction.output_generator_nonzeros, 0);
        assert!(construction.charged_items > 0);
        assert!(construction.deadline_polls > 0);
        assert_eq!(
            report.stages[report.stages.len() - 2].kind,
            CganCzStageKind::M17Lower
        );
        assert_eq!(
            report.stages[report.stages.len() - 1].kind,
            CganCzStageKind::M17Upper
        );
    }

    #[cfg(feature = "external-vnncomp")]
    #[test]
    fn real_model_prop3_retains_every_discriminator_symbol_diagnostic_only() {
        let root = real_asset_root();
        let config = OnnxLoadConfig::default()
            .with_batch_norm_folding_policy(BatchNormFoldingPolicy::PreserveRaw)
            .with_raw_float32_initializer_provenance(true);
        let model =
            load_onnx_with_config(root.join("onnx/cGAN_imgSz32_nCh_1.onnx"), &config).unwrap();
        let graph = model.to_graph_network().unwrap();
        let (_, input, moat) = load_vnnlib_with_certified_scalar_moat(
            root.join("vnnlib/cGAN_imgSz32_nCh_1_prop_3_input_eps_0.010_output_eps_0.015.vnnlib"),
        )
        .unwrap();
        let budget = ConstrainedZonotopeCallBudget::new(
            Instant::now() + Duration::from_mins(3),
            64 << 20,
            2 << 30,
        );
        let report = probe_cgan_nch1_sequential_unwired(
            &model,
            &graph,
            &input,
            moat,
            cgan_nch1_generator_discriminator_handoff_qualification_limits(),
            budget,
        );
        assert_eq!(report.authority, CGAN_CZ_VERDICT_AUTHORITY);
        let CganCzProbeStatus::Completed(bounds) = &report.status else {
            panic!("full prop_3 diagnostic tail declined: {:?}", report.status);
        };
        assert_eq!(bounds.lower_bound, -0.622_751_117_118_960_9);
        assert_eq!(bounds.upper_bound, 2.202_485_932_097_938);
        assert_eq!(bounds.bn_tail_correction_upper, 1.154_694_607_797_815_4e-7);
        assert!(!bounds.separates_unsafe_moat);
        assert_eq!(bounds.lower_m17_status, ReluTailDualStatus::Completed);
        assert_eq!(bounds.upper_m17_status, ReluTailDualStatus::Completed);
        assert_eq!(report.stages.len(), 29);
        let final_reduction = &report.stages[23];
        assert_eq!(final_reduction.node, "Relu_20");
        assert_eq!(final_reduction.kind, CganCzStageKind::OrderReduction);
        assert_eq!(final_reduction.input_alpha_dim, 492);
        assert_eq!(final_reduction.output_alpha_dim, 492);
        assert_eq!(final_reduction.discarded_generators, 0);
        assert_eq!(report.stages[25].node, "Conv_22");
        assert_eq!(report.stages[25].output_generator_nonzeros, 113_920);
        let construction = &report.stages[26];
        assert_eq!(construction.kind, CganCzStageKind::OutputTailConstruction);
        assert_eq!(construction.output_shape, [1]);
        assert_eq!(construction.output_alpha_dim, 0);
        assert_eq!(construction.output_generator_nonzeros, 0);
        assert!(construction.charged_items > 0);
        assert!(construction.deadline_polls > 0);
        assert_eq!(
            report.stages[report.stages.len() - 2].kind,
            CganCzStageKind::M17Lower
        );
        assert_eq!(
            report.stages[report.stages.len() - 1].kind,
            CganCzStageKind::M17Upper
        );
    }
}
