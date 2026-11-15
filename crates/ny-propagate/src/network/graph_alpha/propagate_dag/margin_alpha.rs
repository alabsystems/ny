// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #vit-margin-alpha — the CERTIFIED C-ROW MARGIN ASCENT
//! (`scratchpad/vbs/vit_diagnosis.md` §3, item B).
//!
//! # What was wrong with the root α on vit_2023
//!
//! The legacy DAG ascent optimizes `finite_lower_sum` over the ten RAW logits
//! while the verdict depends on the worst of nine MARGIN rows; its ReLU
//! gradient is the sign-definite local rule (machine-checked,
//! `AlphaGradientDefect.lean`), so it ends at the clamp with
//! `best_impr_last=0`; and the bilinear McCormick parameters of the attention
//! products (vit's var×var MatMuls are lowered to `BilinearCrown` by
//! `ny-build/src/convert/matmul.rs`) are SPSA-steered by that same raw sum and
//! then DROPPED — the collection artifact carries only the shared ReLU α, so
//! the root's spec-guided pass never sees them.
//!
//! # What this lane does instead
//!
//! * **B1 — the loss is the margin rows.** Every iterate is ONE certified
//!   backward seeded with the property's C-rows
//!   ([`GraphNetwork::dag_alpha_margin_rows_pass`]), the same walk the loop
//!   uses for its own folds with only the seed changed. Iterates are ranked by
//!   the hinge `Σ_r min(0, slack_r)` (ties: the worst slack).
//! * **B2 — one α per objective.** With `alpha_spec_slots > 0` every margin
//!   row gets a δ slot (`α_row = clamp01(α_base + δ_row)`) through the
//!   existing #spec-axis-alpha machinery; the seed's rows ARE margin rows, so
//!   the walk applies each row's own α.
//! * **B3 — a gradient with a direction.** Per unverified row `r`, the
//!   envelope (Danskin) rule evaluated at THAT row's own input argmin: ReLU
//!   `Σ [A_ri > 0] A_ri · clamp(ĥ_i(x*_r), l_i, u_i)`, bilinear
//!   `w·(L1 − L2)(x̂, ŷ)` / `w·(U1 − U2)(x̂, ŷ)`. Exact for operands affine in
//!   the input; the documented point-forward surrogate deeper
//!   (`backward/gradients.rs`). Not sign-definite, unlike the local rule.
//! * **B4 — α on the attention McCormick planes** (`matmul_alpha`): the
//!   `[4, m, n, k]` interpolation parameters of every `BilinearCrown` and
//!   var×var `MatMul` node, Adam-stepped through the same contracted clamp.
//!
//! The WHOLE optimizer state — shared ReLU α, per-row δ, bilinear parameters
//! — is checkpointed together at the best-scoring iterate, which is what
//! makes admitting bilinear state (refused by the resident lane,
//! `MarginGradientEligibility::NonReluOptimizerState`) safe.
//!
//! # Frozen MulBinary / Reciprocal α (#vit-margin-alpha-complex, P6)
//!
//! The fixed-index softmax decomposition (`softmax: complex_fixed_index`,
//! `network/core/graph/softmax_fixed_index.rs`) plants one `Reciprocal` and one
//! `MulBinary` per softmax, and the lane refused every such root
//! (`reason=unsupported_alpha_family`): the two ascents were mutually
//! exclusive on vit. Under `margin_alpha_freeze_mul_reciprocal` the lane takes
//! those roots with both families' α FROZEN at their DAG-init values — the
//! MulBinary interpolation map (r = 0.5 everywhere, `init.rs`) is handed
//! unchanged to every fold, and the Reciprocal tangent points ride in the
//! runtime α state, which this lane never updates for that family (its
//! `update_all_alphas` call passes empty Reciprocal/MulBinary gradient maps).
//! This is abc's `disable_optimization` shape. Only ReLU α (+ δ) and the
//! attention McCormick parameters move, and they now optimize the margin rows
//! THROUGH the decomposed softmax. Sigmoid/Tanh/Sqrt α still refuse.
//!
//! # V2: the untrusted joint search (`margin_alpha_search: joint_f32`)
//!
//! `joint_search` replaces the certified ascent below with M5a's pattern:
//! lower the graph into a plain-data search copy (self-checked against the
//! lane's own point forward), let the injected untrusted backend search the
//! ReLU slopes, the attention McCormick planes and the fixed-index softmax's
//! MulBinary planes jointly with the certified boxes frozen, clamp its
//! proposal at an ingest gate, certify ONCE, and keep the state only when it
//! is strictly better. Dark by default (`ascent`).
//!
//! # Certify once
//!
//! After the loop the committed state is restored and ONE fresh certified
//! pass is run with exactly that state. Its bounds — and only its bounds —
//! are published ([`MarginAlphaPublication`]) for the multi-objective root to
//! intersect with its own. `margin_alpha_published_bound_recomputes_bit_
//! identically` pins that a second fresh pass with the committed state
//! reproduces the publication bit-for-bit.
//!
//! # Soundness
//!
//! Every relaxation the lane can select is valid: α is written only through
//! `clamp_alpha_to_envelope_domain` (`#[ensures]` + `crownproof::relu_lower`),
//! δ keeps `α_base + δ ∈ [0,1]`, and every bilinear parameter is clamped into
//! `[0,1]` where `interpolated_mccormick` certifies the plane constant
//! outward. Frozen MulBinary/Reciprocal α are FIXED valid relaxations (the
//! certified interpolated McCormick planes at r = 0.5, clamped into `[0,1]`
//! by the kernel; the Reciprocal init tangent/chord), identical in every fold
//! and in the certify pass. Gradients and scores only choose WHICH valid relaxation is
//! evaluated; they never enter a bound. The published bound is a certified
//! pass (outward-rounded `concretize_sound`), and the root merges it
//! all-or-nothing with an identity check. vit rows are all truth-unsat, so no
//! attack can catch a false unsat here: soundness is by construction, pinned
//! by the tests below and the real-ONNX sampled-containment arm in
//! `crates/ny-onnx/tests/vit_root_bound_containment.rs`.

use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

use ndarray::{Array1, Array2, Array4, ArrayView1};
use ny_core::{GemmEngine, Result};
use ny_tensor::BoundedTensor;
use tracing::debug;

use super::super::reference_bounds::GraphAlphaReferenceBounds;
use super::super::runtime_state::DagAlphaRuntimeState;
use super::{
    alpha_update, emit_alpha_ascent_exit_line, spec_axis, DagAlphaCollectionArtifact,
    DagAlphaLoopContext, MarginGradientEligibility,
};
use crate::bounds::{
    AlphaSpecAscent, AlphaSpecEarlyExit, GraphAlphaCrownIntermediate, LinearBounds,
    MarginAlphaPublication,
};
use crate::layers::{BilinearCrownLayer, Layer};
use crate::network::core::{GraphNetwork, NETWORK_INPUT};
use crate::network::graph_alpha::bounds::AlphaReferenceBoundsSource;

// #vit-root-alpha (ENGINE_PROGRAM V2): `margin_alpha_search: joint_f32` — the
// untrusted joint search over the lane's plain-data graph copy, certified once.
// Dark (`ascent`, the default), neither module is entered.
mod joint_search;
#[cfg(test)]
mod joint_search_tests;
// #graph-lane-search (ENGINE_PROGRAM M7) lowers the graph β-CROWN lane's
// GraphNetwork with this same module (re-exported crate-wide as
// `crate::network::graph_search_lowering`); visibility only.
pub(crate) mod search_lowering;

/// Whether the certified C-row margin ascent takes this root, and why not.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum MarginAscentAdmission {
    Admitted,
    /// `root_alpha_margin` + `root_alpha_margin_gradient` not both armed.
    NotRequested,
    /// The caller supplied no margin rows (`AlphaCrownConfig::spec_ascent`).
    MissingSpecObjective,
    /// A bounds-only caller would discard the state that carries the
    /// publication; the lane's output box is only the pre-loop CROWN bound.
    BoundsOnlyCaller,
    /// ReLU-only optimizer state: the resident joint-adjoint lane owns it,
    /// byte-identically to before this lane existed (unless `matmul_alpha`
    /// is armed on a graph carrying a var×var `MatMul`, which the resident
    /// lane cannot parameterize).
    ResidentLaneEligible,
    /// `NY_MULTIOBJ_JOINT_ALPHA` / `NY_ROOT_ALPHA_TRUE` policies are armed;
    /// they keep the legacy loop exactly, as they always have.
    ConflictingAlphaPolicy,
    /// MulBinary / S-shaped / Sqrt / Reciprocal α state: not checkpointed or
    /// differentiated by this lane, so it refuses rather than mix states.
    /// Under `margin_alpha_freeze_mul_reciprocal` MulBinary and Reciprocal no
    /// longer refuse (their α is held frozen); S-shaped and Sqrt still do.
    UnsupportedAlphaFamily,
    /// An active INVPROP γ optimizer owns the output seed.
    InvpropActive,
    /// `iterations == 0`: nothing to optimize.
    NoIterations,
}

impl MarginAscentAdmission {
    pub(super) fn is_admitted(self) -> bool {
        matches!(self, Self::Admitted)
    }

    /// Stable token for the unconditional `[alpha-margin] armed=0` line.
    pub(super) fn reason(self) -> &'static str {
        match self {
            Self::Admitted => "admitted",
            Self::NotRequested => "not_requested",
            Self::MissingSpecObjective => "missing_spec_objective",
            Self::BoundsOnlyCaller => "bounds_only_caller",
            Self::ResidentLaneEligible => "resident_lane_eligible",
            Self::ConflictingAlphaPolicy => "conflicting_alpha_policy",
            Self::UnsupportedAlphaFamily => "unsupported_alpha_family",
            Self::InvpropActive => "invprop_active",
            Self::NoIterations => "no_iterations",
        }
    }
}

/// The ONE admission decision, kept pure so its unit test exercises exactly
/// the predicate the loop calls.
///
/// `matmul_alpha_on_var_matmul` is `matmul_alpha` armed AND the graph carrying
/// at least one var×var `MatMul`: the resident lane only ever sees such a
/// graph as ReLU-only (it has no `MatMul` parameters to checkpoint), so the
/// request for attention-plane α would otherwise be silently dropped.
///
/// `freeze_mul_reciprocal` is `margin_alpha_freeze_mul_reciprocal`
/// (#vit-margin-alpha-complex, P6): MulBinary and Reciprocal α state then no
/// longer refuse, because the lane holds it frozen. With it `false` the
/// predicate is exactly the historical one.
#[allow(clippy::fn_params_excessive_bools, clippy::too_many_arguments)]
pub(super) fn margin_ascent_admission(
    requested: bool,
    has_spec_objective: bool,
    returns_state: bool,
    resident_eligibility: MarginGradientEligibility,
    matmul_alpha_on_var_matmul: bool,
    has_mul_binary: bool,
    has_s_shaped: bool,
    has_sqrt: bool,
    has_reciprocal: bool,
    freeze_mul_reciprocal: bool,
    invprop_gamma_active: bool,
    iterations: usize,
) -> MarginAscentAdmission {
    if !requested {
        return MarginAscentAdmission::NotRequested;
    }
    if !has_spec_objective {
        return MarginAscentAdmission::MissingSpecObjective;
    }
    if !returns_state {
        return MarginAscentAdmission::BoundsOnlyCaller;
    }
    match resident_eligibility {
        MarginGradientEligibility::Eligible if !matmul_alpha_on_var_matmul => {
            return MarginAscentAdmission::ResidentLaneEligible;
        }
        MarginGradientEligibility::Eligible => {}
        MarginGradientEligibility::ConflictMultiobjJointAlpha
        | MarginGradientEligibility::ConflictRootAlphaTrue
        | MarginGradientEligibility::ConflictBothAlphaPolicies => {
            return MarginAscentAdmission::ConflictingAlphaPolicy;
        }
        MarginGradientEligibility::NotRequested
        | MarginGradientEligibility::MissingSpecObjective
        | MarginGradientEligibility::NonAnalyticChain
        | MarginGradientEligibility::NonReluOptimizerState => {}
    }
    // P6: MulBinary / Reciprocal block only while their α would be left
    // unowned; frozen, the lane owns them (unchanged in every fold).
    let mul_reciprocal_blocks = (has_mul_binary || has_reciprocal) && !freeze_mul_reciprocal;
    if has_s_shaped || has_sqrt || mul_reciprocal_blocks {
        return MarginAscentAdmission::UnsupportedAlphaFamily;
    }
    if invprop_gamma_active {
        return MarginAscentAdmission::InvpropActive;
    }
    if iterations == 0 {
        return MarginAscentAdmission::NoIterations;
    }
    MarginAscentAdmission::Admitted
}

/// The property's margin rows as LOWER objectives.
///
/// Upper-verification rows are sign-flipped (exact in f32) so that raising
/// `lower(ĉ_r^T y)` is uniformly the goal and `lower(ĉ_r^T y) > t̂_r` is every
/// row's verification condition — the same convention the resident lane's
/// `binding_margin_lower_objective` uses.
pub(super) struct MarginRows {
    seed: Array2<f32>,
    lower_thresholds: Vec<f32>,
    flipped: Vec<bool>,
    rows: Vec<AlphaSpecEarlyExit>,
}

impl MarginRows {
    /// `None` on any malformed row (width mismatch, non-finite coefficient or
    /// threshold): the lane refuses rather than optimize a partial property.
    pub(super) fn new(ascent: &AlphaSpecAscent, output_dim: usize) -> Option<Self> {
        let k = ascent.rows.len();
        if k == 0 || output_dim == 0 {
            return None;
        }
        let mut seed = Array2::<f32>::zeros((k, output_dim));
        let mut lower_thresholds = Vec::with_capacity(k);
        let mut flipped = Vec::with_capacity(k);
        for (r, row) in ascent.rows.iter().enumerate() {
            if row.objective.len() != output_dim || !row.threshold.is_finite() {
                return None;
            }
            let flip = row.verify_upper_bound;
            for (c, &value) in row.objective.iter().enumerate() {
                if !value.is_finite() {
                    return None;
                }
                seed[[r, c]] = if flip { -value } else { value };
            }
            lower_thresholds.push(if flip { -row.threshold } else { row.threshold });
            flipped.push(flip);
        }
        Some(Self {
            seed,
            lower_thresholds,
            flipped,
            rows: ascent.rows.clone(),
        })
    }

    pub(super) fn len(&self) -> usize {
        self.lower_thresholds.len()
    }

    /// Bounds on the ORIGINAL objective `c_r^T y` from certified bounds on the
    /// lower objective `ĉ_r^T y`: identity for ordinary rows, `(-hi, -lo)` for
    /// flipped rows (negation is exact).
    pub(super) fn original_bounds(&self, lower: &[f32], upper: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut lo = Vec::with_capacity(self.len());
        let mut hi = Vec::with_capacity(self.len());
        for ((&flip, &l), &u) in self.flipped.iter().zip(lower).zip(upper) {
            if flip {
                lo.push(-u);
                hi.push(-l);
            } else {
                lo.push(l);
                hi.push(u);
            }
        }
        (lo, hi)
    }
}

/// Ranking of one certified margin fold. Higher `hinge` is better; `0` means
/// every row verified.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct MarginScore {
    pub(super) hinge: f64,
    pub(super) min_slack: f64,
    pub(super) verified: usize,
}

/// Score lower-objective bounds against their thresholds. A NaN slack is the
/// worst possible slack, never a free pass.
pub(super) fn score_margins(lower: &[f32], lower_thresholds: &[f32]) -> MarginScore {
    let mut hinge = 0.0f64;
    let mut min_slack = f64::INFINITY;
    let mut verified = 0usize;
    for (&lo, &threshold) in lower.iter().zip(lower_thresholds) {
        let slack = f64::from(lo) - f64::from(threshold);
        let slack = if slack.is_nan() {
            f64::NEG_INFINITY
        } else {
            slack
        };
        if slack > 0.0 {
            verified += 1;
        }
        hinge += slack.min(0.0);
        min_slack = min_slack.min(slack);
    }
    MarginScore {
        hinge,
        min_slack,
        verified,
    }
}

fn score_improves(candidate: MarginScore, incumbent: MarginScore) -> bool {
    candidate.hinge > incumbent.hinge
        || (candidate.hinge == incumbent.hinge && candidate.min_slack > incumbent.min_slack)
}

/// One certified margin fold.
pub(super) struct MarginFold {
    lower: Vec<f32>,
    upper: Vec<f32>,
    intermediate: Option<GraphAlphaCrownIntermediate>,
    secs: f64,
}

/// The bilinear/MatMul interpolation parameters the lane optimizes, with
/// their Adam moments (`[4, m, n, k]`, shared over batch dims).
struct MatmulAlphaState {
    alphas: HashMap<String, Array4<f32>>,
    adam_m: HashMap<String, Array4<f32>>,
    adam_v: HashMap<String, Array4<f32>>,
}

/// The MulBinary / Reciprocal α state the lane holds FIXED
/// (#vit-margin-alpha-complex, P6). `Default` (nothing frozen) is the
/// historical lane: no MulBinary map reaches any fold.
#[derive(Clone, Default)]
struct FrozenFamilies {
    /// MulBinary interpolation parameters exactly as the DAG init built them
    /// (`[2, n]`, r = 0.5), handed UNCHANGED to every fold and to the
    /// certify-once pass. `None` when the key is dark or the graph carries no
    /// MulBinary α, so the walk takes the historical `None` route.
    mul_binary: Option<HashMap<String, Array2<f32>>>,
    mul_binary_nodes: usize,
    /// Reciprocal α nodes in the runtime state. Their tangent points are never
    /// written by this lane (empty Reciprocal gradient map), so they stay at
    /// their init value; this is only the count for the observable.
    reciprocal_nodes: usize,
}

impl FrozenFamilies {
    /// Stable token for `frozen_families=` in the observables.
    fn token(&self) -> &'static str {
        match (self.mul_binary_nodes > 0, self.reciprocal_nodes > 0) {
            (true, true) => "mul_binary,reciprocal",
            (true, false) => "mul_binary",
            (false, true) => "reciprocal",
            (false, false) => "none",
        }
    }
}

/// Everything the lane needs, built BEFORE any state is handed over, so a
/// refusal (malformed rows, a first fold the exact-seed face refuses) leaves
/// the legacy loop to run unchanged.
pub(super) struct MarginLanePrepared {
    rows: MarginRows,
    matmul: Option<MatmulAlphaState>,
    frozen: FrozenFamilies,
    first: MarginFold,
}

/// The whole optimizer state that produced one scored fold.
struct MarginCheckpoint {
    relu: crate::bounds::GraphAlphaState,
    bilinear: Option<HashMap<String, Array4<f32>>>,
    iteration: usize,
    lower: Vec<f32>,
    upper: Vec<f32>,
    score: MarginScore,
}

/// One iteration's steering data.
struct MarginGradients {
    /// Shared ReLU gradient per runtime ReLU index (FULL pre-activation width).
    relu: Vec<Array1<f32>>,
    /// Per margin row (active rows only): its own per-ReLU gradient, for δ.
    row_relu: Vec<Option<Vec<Array1<f32>>>>,
    /// Bilinear/MatMul gradients, keyed like [`MatmulAlphaState::alphas`].
    bilinear: HashMap<String, Array4<f32>>,
    point_failures: usize,
}

/// A bilinear operand's certified box: the network input or a node's
/// reference bounds.
fn operand_bounds<'a>(
    name: &str,
    input: &'a BoundedTensor,
    node_bounds: &'a HashMap<String, BoundedTensor>,
) -> Option<&'a BoundedTensor> {
    if name == NETWORK_INPUT {
        Some(input)
    } else {
        node_bounds.get(name)
    }
}

/// A node's concrete value at a row's input argmin, flattened row-major.
fn flat_point(
    name: &str,
    values: &HashMap<String, BoundedTensor>,
    x_star: &[f32],
) -> Option<Vec<f32>> {
    if name == NETWORK_INPUT {
        return Some(x_star.to_vec());
    }
    values
        .get(name)
        .map(|bounds| bounds.lower().iter().copied().collect())
}

/// The argmin corner of one row's LOWER input-level relation: `lo_k` where
/// the coefficient is positive, `hi_k` otherwise (the corner
/// `concretize_sound` itself selects).
fn row_argmin_corner(
    final_bounds: &LinearBounds,
    row: usize,
    lo: &[f32],
    hi: &[f32],
) -> Option<Vec<f32>> {
    let coefficients = final_bounds.lower_a();
    if row >= coefficients.nrows() || coefficients.ncols() != lo.len() || lo.len() != hi.len() {
        return None;
    }
    let corner: Vec<f32> = (0..lo.len())
        .map(|k| {
            if coefficients[[row, k]] > 0.0 {
                lo[k]
            } else {
                hi[k]
            }
        })
        .collect();
    corner.iter().all(|v| v.is_finite()).then_some(corner)
}

/// One row's ReLU α gradient by the envelope rule: `A_ri · clamp(ĥ_i, l_i,
/// u_i)` over unstable neurons with `A_ri > 0` (the lower relaxation
/// `y ≥ α·x` is the α-dependent branch). With no point, `ĥ_i` falls back to
/// `l_i` — the historical local rule — for this row only.
fn relu_row_gradient(
    a_row: ArrayView1<'_, f32>,
    pre_lower: &Array1<f32>,
    pre_upper: &Array1<f32>,
    point: Option<&[f32]>,
) -> Array1<f32> {
    let width = pre_lower.len();
    let mut gradient = Array1::<f32>::zeros(width);
    if a_row.len() != width || pre_upper.len() != width {
        return gradient;
    }
    let point = point.filter(|values| values.len() == width);
    for i in 0..width {
        let (l, u) = (pre_lower[i], pre_upper[i]);
        if !l.is_finite() || !u.is_finite() || l >= 0.0 || u <= 0.0 {
            continue;
        }
        let a = a_row[i];
        if !a.is_finite() || a <= 0.0 {
            continue;
        }
        let factor = match point {
            Some(values) if values[i].is_finite() => values[i].clamp(l, u),
            _ => l,
        };
        gradient[i] = a * factor;
    }
    gradient
}

fn bits_equal(left: &[f32], right: &[f32]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(a, b)| a.to_bits() == b.to_bits())
}

/// What the lane did, for the unconditional observable.
struct MarginLaneSummary {
    rows: usize,
    spec_slots: usize,
    matmul_alpha_nodes: usize,
    relu_alpha_nodes: usize,
    folds: usize,
    init: MarginScore,
    best: MarginScore,
    committed: usize,
    at_clamp: (usize, usize),
    exit: &'static str,
    published: &'static str,
    recompute_identical: bool,
    point_failures: usize,
    frozen_token: &'static str,
    frozen_mul_binary_nodes: usize,
    frozen_reciprocal_nodes: usize,
    secs: f64,
}

/// `[alpha-margin] armed=1 …` — UNCONDITIONAL once the lane runs (rate
/// limited like `[alpha-ascent]`: the first 16 occurrences, then powers of
/// two). Its ABSENCE with the keys set means the keys never reached this lane.
///
/// `frozen_families=` (#vit-margin-alpha-complex) names the α families the
/// lane held fixed: `none` on every historical (ReLU/bilinear-only) root,
/// `mul_binary,reciprocal` on a fixed-index-softmax root admitted by
/// `margin_alpha_freeze_mul_reciprocal`.
fn emit_margin_alpha_line(summary: &MarginLaneSummary) {
    static OCCURRENCES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let Some(occ) = crate::network::graph_alpha::retained_occurrence(&OCCURRENCES) else {
        return;
    };
    let (unstable, at_clamp) = summary.at_clamp;
    eprintln!(
        "[alpha-margin] armed=1 rows={} spec_slots={} matmul_alpha_nodes={} relu_alpha_nodes={} \
         iters={} margin_min_init={:.6e} margin_min_best={:.6e} hinge_init={:.6e} \
         hinge_best={:.6e} verified_init={}/{} verified_best={}/{} committed={} \
         at_clamp={}/{} exit={} published={} recompute_identical={} grad_rule=envelope_per_row \
         frozen_families={} frozen_mul_binary_nodes={} frozen_reciprocal_nodes={} \
         point_failures={} secs={:.2} occ={occ}",
        summary.rows,
        summary.spec_slots,
        summary.matmul_alpha_nodes,
        summary.relu_alpha_nodes,
        summary.folds,
        summary.init.min_slack,
        summary.best.min_slack,
        summary.init.hinge,
        summary.best.hinge,
        summary.init.verified,
        summary.rows,
        summary.best.verified,
        summary.rows,
        summary.committed,
        at_clamp,
        unstable,
        summary.exit,
        summary.published,
        u8::from(summary.recompute_identical),
        summary.frozen_token,
        summary.frozen_mul_binary_nodes,
        summary.frozen_reciprocal_nodes,
        summary.point_failures,
        summary.secs,
    );
}

/// `[alpha-margin-frozen] armed=1 admitted=<0|1> reason=<token> …`
/// (#vit-margin-alpha-complex, P6) — the freeze key's own engagement line,
/// printed UNCONDITIONALLY (rate limited like the other lane lines) every time
/// the resolved `margin_alpha_freeze_mul_reciprocal` reaches the lane's
/// admission decision, whatever that decision is. So a measurement that armed
/// the key and sees NO such line never reached the lane (VOID), one that sees
/// `admitted=0` was refused for the named reason, and one that sees
/// `admitted=1` must then show `[alpha-margin] armed=1 … frozen_families=…`
/// (or an `armed=0 reason=first_fold_*` refusal) for the same root.
pub(super) fn emit_margin_alpha_frozen_line(
    admission: MarginAscentAdmission,
    mul_binary_nodes: usize,
    reciprocal_nodes: usize,
    has_s_shaped: bool,
    has_sqrt: bool,
) {
    static OCCURRENCES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let Some(occ) = crate::network::graph_alpha::retained_occurrence(&OCCURRENCES) else {
        return;
    };
    eprintln!(
        "[alpha-margin-frozen] armed=1 admitted={} reason={} mul_binary_nodes={} \
         reciprocal_nodes={} s_shaped={} sqrt={} occ={occ}",
        u8::from(admission.is_admitted()),
        admission.reason(),
        mul_binary_nodes,
        reciprocal_nodes,
        u8::from(has_s_shaped),
        u8::from(has_sqrt),
    );
}

/// `[alpha-margin] armed=0 reason=…` — printed whenever any of the lane's keys
/// is set but the lane did not run, so a refusal is never mistaken for a
/// negative result.
pub(super) fn emit_margin_alpha_refusal(reason: &str) {
    static OCCURRENCES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let Some(occ) = crate::network::graph_alpha::retained_occurrence(&OCCURRENCES) else {
        return;
    };
    eprintln!("[alpha-margin] armed=0 reason={reason} occ={occ}");
}

/// `[alpha-margin-error] stage=<first_fold|gradient|fold|certify> error="…"` —
/// UNCONDITIONAL (rate limited like the other lines) whenever the lane loses a
/// step to an error. The `armed=0 reason=first_fold_refused` / `exit=fold_error`
/// / `published=0_refused` tokens say THAT it happened; this line says WHY.
/// Without it the cause reached only `debug!`, which the scored path does not
/// print, so a VOID arm could not be diagnosed from its own log.
fn emit_margin_alpha_error(stage: &str, error: &ny_core::NyError) {
    static OCCURRENCES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let Some(occ) = crate::network::graph_alpha::retained_occurrence(&OCCURRENCES) else {
        return;
    };
    let text: String = error
        .to_string()
        .chars()
        .map(|ch| {
            if ch.is_control() || ch == '"' {
                ' '
            } else {
                ch
            }
        })
        .take(240)
        .collect();
    eprintln!("[alpha-margin-error] stage={stage} error=\"{text}\" occ={occ}");
}

/// Test-only record of the state the lane committed and re-certified, so the
/// certify-once contract can be checked by an independent recomputation.
#[cfg(test)]
pub(super) struct MarginCommitRecord {
    pub(super) relu: crate::bounds::GraphAlphaState,
    pub(super) bilinear: Option<HashMap<String, Array4<f32>>>,
    pub(super) seed: Array2<f32>,
    pub(super) relu_name_to_idx: HashMap<String, usize>,
    pub(super) committed_iteration: usize,
    pub(super) folds_completed: usize,
    pub(super) exit: &'static str,
    pub(super) committed_lower_objective: Vec<f32>,
    pub(super) matmul_alpha_moved: bool,
    /// P6: the frozen MulBinary map every fold (and the certify pass) used.
    pub(super) frozen_mul_binary: Option<HashMap<String, Array2<f32>>>,
    /// P6: Reciprocal α bits when the lane started and in the committed state
    /// (`(node, lower_path_mid bits, upper_path_mid bits)`), so a test can
    /// prove the lane never moved them.
    pub(super) reciprocal_at_start: Vec<(String, Vec<u32>, Vec<u32>)>,
    pub(super) reciprocal_at_commit: Vec<(String, Vec<u32>, Vec<u32>)>,
}

/// Test-only: the Reciprocal α of a state as exact bits, node-sorted (the
/// state's map is a `BTreeMap`, so iteration is already deterministic).
#[cfg(test)]
fn reciprocal_alpha_bits(
    state: &crate::bounds::GraphAlphaState,
) -> Vec<(String, Vec<u32>, Vec<u32>)> {
    state
        .reciprocal_alphas
        .iter()
        .map(|(name, alpha)| {
            (
                name.clone(),
                alpha.lower_path_mid.iter().map(|v| v.to_bits()).collect(),
                alpha.upper_path_mid.iter().map(|v| v.to_bits()).collect(),
            )
        })
        .collect()
}

#[cfg(test)]
thread_local! {
    pub(super) static LAST_MARGIN_COMMIT: std::cell::RefCell<Option<MarginCommitRecord>> =
        const { std::cell::RefCell::new(None) };
}

impl GraphNetwork {
    /// One certified margin fold under the runtime's CURRENT state.
    ///
    /// `frozen_mul` is the P6 frozen MulBinary map (`None` = the historical
    /// fixed-facet route, byte-identical).
    #[allow(clippy::too_many_arguments)]
    fn margin_alpha_fold(
        &self,
        ctx: &DagAlphaLoopContext<'_>,
        node_bounds: &HashMap<String, BoundedTensor>,
        runtime: &DagAlphaRuntimeState,
        bilinear: Option<&HashMap<String, Array4<f32>>>,
        frozen_mul: Option<&HashMap<String, Array2<f32>>>,
        seed: &Array2<f32>,
        capture_intermediates: bool,
    ) -> Result<MarginFold> {
        let started = Instant::now();
        let (bounds, intermediate) = self.dag_alpha_margin_rows_pass(
            ctx.input,
            node_bounds,
            ctx.exec_order,
            ctx.output_dim,
            ctx.input_dim,
            runtime.relu_name_to_idx(),
            runtime.graph(),
            ctx.engine,
            bilinear,
            frozen_mul,
            ctx.config.deadline,
            seed,
            capture_intermediates,
        )?;
        Ok(MarginFold {
            lower: bounds.lower().iter().copied().collect(),
            upper: bounds.upper().iter().copied().collect(),
            intermediate,
            secs: started.elapsed().as_secs_f64(),
        })
    }

    /// Whether the graph carries a var×var `MatMul` (the bounded binary op:
    /// ny-build lowers constant-operand MatMuls to `Linear`, and var×var ones
    /// from ONNX to `BilinearCrown`, so this is the decoder/whisper shape).
    pub(super) fn graph_has_var_matmul(&self) -> bool {
        self.nodes
            .values()
            .any(|node| matches!(node.layer, Layer::MatMul(_)))
    }

    /// B4: interpolation parameters for every `BilinearCrown` (carried over
    /// from DAG init, `r = 1` = the L1/U1 planes, auto_LiRPA's
    /// `torch.ones` start) and every var×var `MatMul` (new, same start).
    ///
    /// A `MatMul` whose parameter shape cannot be formed (a non-finite scale,
    /// operands below 2-D, missing operand bounds) is SKIPPED, not refused:
    /// it keeps the historical midpoint plane, which is exactly what it would
    /// have had without this key.
    fn margin_matmul_alpha_state(
        &self,
        node_bounds: &HashMap<String, BoundedTensor>,
        input: &BoundedTensor,
        bilinear_alphas: &HashMap<String, Array4<f32>>,
        bilinear_adam_m: &HashMap<String, Array4<f32>>,
        bilinear_adam_v: &HashMap<String, Array4<f32>>,
    ) -> MatmulAlphaState {
        let mut alphas = bilinear_alphas.clone();
        let mut adam_m = bilinear_adam_m.clone();
        let mut adam_v = bilinear_adam_v.clone();
        let mut names: Vec<&String> = self.nodes.keys().collect();
        names.sort();
        for name in names {
            let Some(node) = self.nodes.get(name.as_str()) else {
                continue;
            };
            let Layer::MatMul(matmul) = &node.layer else {
                continue;
            };
            if alphas.contains_key(name.as_str()) {
                continue;
            }
            let Ok((a_name, b_name)) = node.require_binary_inputs() else {
                continue;
            };
            let (Some(a_bounds), Some(b_bounds)) = (
                operand_bounds(a_name, input, node_bounds),
                operand_bounds(b_name, input, node_bounds),
            ) else {
                continue;
            };
            let Ok(layer) = BilinearCrownLayer::try_new(matmul.transpose_b, matmul.scale) else {
                continue;
            };
            let Ok((m, n, k)) = layer.alpha_shape(a_bounds.shape(), b_bounds.shape()) else {
                continue;
            };
            alphas.insert(name.clone(), Array4::ones((4, m, n, k)));
            adam_m.insert(name.clone(), Array4::zeros((4, m, n, k)));
            adam_v.insert(name.clone(), Array4::zeros((4, m, n, k)));
        }
        // Every carried BilinearCrown entry must have both Adam moments, or
        // `update_all_alphas` would silently skip it (`adam_update_alpha_map`
        // requires all three maps): seed a missing moment at zero.
        for (name, alpha) in &alphas {
            adam_m
                .entry(name.clone())
                .or_insert_with(|| Array4::zeros(alpha.raw_dim()));
            adam_v
                .entry(name.clone())
                .or_insert_with(|| Array4::zeros(alpha.raw_dim()));
        }
        MatmulAlphaState {
            alphas,
            adam_m,
            adam_v,
        }
    }

    /// Concrete forward at `x*`, via a degenerate box (the envelope rule's
    /// point surrogate; `backward/gradients.rs::envelope_points_at`).
    fn margin_point_values(
        &self,
        input: &BoundedTensor,
        x_star: &[f32],
        engine: Option<&dyn GemmEngine>,
    ) -> Option<HashMap<String, BoundedTensor>> {
        let point =
            ndarray::ArrayD::from_shape_vec(input.lower().raw_dim(), x_star.to_vec()).ok()?;
        let degenerate = BoundedTensor::new(point.clone(), point).ok()?;
        self.collect_node_activations_pointwise(&degenerate, engine)
            .ok()
    }

    /// Everything the lane needs, built with only shared borrows: the margin
    /// rows, the bilinear parameter state (when `matmul_alpha`), the frozen
    /// MulBinary/Reciprocal state (when `margin_alpha_freeze_mul_reciprocal`),
    /// and the FIRST certified fold. Any refusal returns its reason and hands
    /// nothing over, so the legacy loop runs unchanged.
    ///
    /// `mul_binary_alphas` is the DAG init's MulBinary map, untouched (the
    /// legacy loop has not run yet); it is cloned ONCE into the frozen state.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prepare_margin_alpha_lane(
        &self,
        ctx: &DagAlphaLoopContext<'_>,
        node_bounds: &HashMap<String, BoundedTensor>,
        runtime: &DagAlphaRuntimeState,
        bilinear_alphas: &HashMap<String, Array4<f32>>,
        bilinear_adam_m: &HashMap<String, Array4<f32>>,
        bilinear_adam_v: &HashMap<String, Array4<f32>>,
        mul_binary_alphas: &HashMap<String, Array2<f32>>,
        ascent: &AlphaSpecAscent,
    ) -> std::result::Result<MarginLanePrepared, &'static str> {
        let rows = MarginRows::new(ascent, ctx.output_dim).ok_or("malformed_margin_rows")?;
        let matmul = if ctx.config.matmul_alpha {
            let state = self.margin_matmul_alpha_state(
                node_bounds,
                ctx.input,
                bilinear_alphas,
                bilinear_adam_m,
                bilinear_adam_v,
            );
            (!state.alphas.is_empty()).then_some(state)
        } else {
            None
        };
        // P6: only an ARMED freeze key hands a MulBinary map to the folds; dark,
        // `FrozenFamilies::default()` keeps every fold on the historical `None`
        // route (and admission already refused any MulBinary/Reciprocal graph).
        let frozen = if ctx.config.margin_alpha_freeze_mul_reciprocal {
            FrozenFamilies {
                mul_binary: (!mul_binary_alphas.is_empty()).then(|| mul_binary_alphas.clone()),
                mul_binary_nodes: mul_binary_alphas.len(),
                reciprocal_nodes: runtime.graph().reciprocal_alpha_names().count(),
            }
        } else {
            FrozenFamilies::default()
        };
        let first = self
            .margin_alpha_fold(
                ctx,
                node_bounds,
                runtime,
                matmul.as_ref().map(|state| &state.alphas),
                frozen.mul_binary.as_ref(),
                &rows.seed,
                true,
            )
            .map_err(|error| {
                if error.is_deadline_exceeded() {
                    "first_fold_deadline"
                } else {
                    emit_margin_alpha_error("first_fold", &error);
                    "first_fold_refused"
                }
            })?;
        Ok(MarginLanePrepared {
            rows,
            matmul,
            frozen,
            first,
        })
    }

    /// B1–B3 steering data for one fold: per unverified row, the envelope rule
    /// at that row's own input argmin. Steering only — never a bound.
    fn margin_alpha_gradients(
        &self,
        ctx: &DagAlphaLoopContext<'_>,
        node_bounds: &HashMap<String, BoundedTensor>,
        relu_names: &[String],
        matmul: Option<&MatmulAlphaState>,
        fold: &MarginFold,
        rows: &MarginRows,
    ) -> Result<MarginGradients> {
        let Some(intermediate) = fold.intermediate.as_ref() else {
            return Err(ny_core::NyError::InternalError(
                "margin-alpha gradient requested from a fold without intermediates".to_string(),
            ));
        };
        let k = rows.len();
        let mut relu: Vec<Array1<f32>> = Vec::with_capacity(relu_names.len());
        for name in relu_names {
            let width = match intermediate.pre_relu_bounds(name) {
                Some((lower, _)) => lower.len(),
                None => self
                    .relu_preactivation_bounds(
                        name,
                        ctx.input,
                        node_bounds,
                        "margin-alpha-gradient",
                    )?
                    .len(),
            };
            relu.push(Array1::<f32>::zeros(width));
        }
        let mut bilinear: HashMap<String, Array4<f32>> =
            matmul.map_or_else(HashMap::new, |state| {
                state
                    .alphas
                    .iter()
                    .map(|(name, alpha)| (name.clone(), Array4::<f32>::zeros(alpha.raw_dim())))
                    .collect()
            });
        let input_lower: Vec<f32> = ctx.input.lower().iter().copied().collect();
        let input_upper: Vec<f32> = ctx.input.upper().iter().copied().collect();
        let mut row_relu: Vec<Option<Vec<Array1<f32>>>> = vec![None; k];
        let mut point_failures = 0usize;
        for r in 0..k {
            let slack = f64::from(fold.lower[r]) - f64::from(rows.lower_thresholds[r]);
            if slack > 0.0 {
                // A verified row contributes nothing to the hinge.
                continue;
            }
            let point =
                row_argmin_corner(&intermediate.final_bounds, r, &input_lower, &input_upper)
                    .and_then(|x_star| {
                        self.margin_point_values(ctx.input, &x_star, ctx.engine)
                            .map(|values| (x_star, values))
                    });
            if point.is_none() {
                point_failures += 1;
            }
            let mut per_node: Vec<Array1<f32>> = Vec::with_capacity(relu_names.len());
            for (idx, name) in relu_names.iter().enumerate() {
                let gradient = match (
                    intermediate.a_at_relu(name),
                    intermediate.pre_relu_bounds(name),
                ) {
                    (Some(a_matrix), Some((pre_lower, pre_upper))) if r < a_matrix.nrows() => {
                        let pre_point = point.as_ref().and_then(|(x_star, values)| {
                            let pre_name = self.nodes.get(name.as_str())?.inputs.first()?;
                            flat_point(pre_name, values, x_star)
                        });
                        relu_row_gradient(
                            a_matrix.row(r),
                            pre_lower,
                            pre_upper,
                            pre_point.as_deref(),
                        )
                    }
                    _ => Array1::<f32>::zeros(relu[idx].len()),
                };
                if gradient.len() == relu[idx].len() {
                    relu[idx] += &gradient;
                }
                per_node.push(gradient);
            }
            row_relu[r] = Some(per_node);

            let Some((x_star, values)) = point.as_ref() else {
                continue;
            };
            for (name, gradient) in &mut bilinear {
                let Some(record) = intermediate.a_at_bilinear.get(name) else {
                    continue;
                };
                if r >= record.nrows() {
                    continue;
                }
                let Some(node) = self.nodes.get(name.as_str()) else {
                    continue;
                };
                let layer = match &node.layer {
                    Layer::BilinearCrown(bilinear_layer) => bilinear_layer.clone(),
                    Layer::MatMul(matmul_layer) => {
                        match BilinearCrownLayer::try_new(
                            matmul_layer.transpose_b,
                            matmul_layer.scale,
                        ) {
                            Ok(layer) => layer,
                            Err(_) => continue,
                        }
                    }
                    _ => continue,
                };
                let Ok((a_name, b_name)) = node.require_binary_inputs() else {
                    continue;
                };
                let (Some(a_bounds), Some(b_bounds)) = (
                    operand_bounds(a_name, ctx.input, node_bounds),
                    operand_bounds(b_name, ctx.input, node_bounds),
                ) else {
                    continue;
                };
                let (Some(a_point), Some(b_point)) = (
                    flat_point(a_name, values, x_star),
                    flat_point(b_name, values, x_star),
                ) else {
                    continue;
                };
                if let Err(error) = layer.margin_alpha_gradient_at_point(
                    record.row(r),
                    a_bounds,
                    b_bounds,
                    &a_point,
                    &b_point,
                    gradient,
                ) {
                    debug!("#vit-margin-alpha: bilinear gradient skipped at {name}: {error}");
                }
            }
        }
        Ok(MarginGradients {
            relu,
            row_relu,
            bilinear,
            point_failures,
        })
    }

    /// Run the certified C-row margin ascent to completion and return the
    /// collection artifact: the pre-loop CROWN output box (the lane's folds
    /// bound margin rows, not logits), the committed SHARED α with its per-spec
    /// δ stripped, and the certify-once publication attached.
    #[allow(clippy::too_many_lines)]
    pub(super) fn run_margin_alpha_lane(
        &self,
        ctx: &DagAlphaLoopContext<'_>,
        prepared: MarginLanePrepared,
        mut runtime: DagAlphaRuntimeState,
        reference_bounds: GraphAlphaReferenceBounds,
        node_bounds_source: AlphaReferenceBoundsSource,
        output_bounds: BoundedTensor,
    ) -> Result<DagAlphaCollectionArtifact> {
        let started = Instant::now();
        let config = ctx.config;
        let MarginLanePrepared {
            rows,
            mut matmul,
            frozen,
            first,
        } = prepared;
        // #vit-root-alpha (V2): the untrusted joint search, certified once,
        // replaces the certified ascent. Dark (`ascent`), this branch is not
        // taken and everything below runs exactly as before.
        if config.margin_alpha_search == crate::bounds::MarginAlphaSearch::JointF32 {
            return self.run_margin_alpha_joint_search(
                ctx,
                rows,
                matmul,
                frozen,
                first,
                runtime,
                reference_bounds,
                node_bounds_source,
                output_bounds,
                started,
            );
        }
        let k = rows.len();
        let node_bounds = reference_bounds.current();
        let relu_names: Vec<String> = runtime.relu_nodes().to_vec();
        let matmul_alpha_nodes = matmul.as_ref().map_or(0, |state| state.alphas.len());
        // P6: the frozen MulBinary map, identical for every fold and for the
        // certify pass (never updated: `update_all_alphas` below receives an
        // empty MulBinary map with `has_mul_binary = false`).
        let frozen_mul = frozen.mul_binary.as_ref();
        #[cfg(test)]
        let reciprocal_at_start = reciprocal_alpha_bits(runtime.graph());

        // B2: one δ slot per margin row (worst-K when fewer slots than rows),
        // keyed by MARGIN row — the seed's carrier rows are margin rows. δ
        // starts at exactly 0, so the next fold is the parity anchor.
        let spec_slots = if config.alpha_spec_slots > 0 {
            let slacks: Vec<f32> = first
                .lower
                .iter()
                .zip(&rows.lower_thresholds)
                .map(|(&lo, &threshold)| lo - threshold)
                .collect();
            spec_axis::assign_spec_slots(
                runtime.graph_mut(),
                &slacks,
                None,
                config.alpha_spec_slots,
            )
        } else {
            0
        };

        let init_score = score_margins(&first.lower, &rows.lower_thresholds);
        let mut best = MarginCheckpoint {
            relu: runtime.snapshot_graph(),
            bilinear: matmul.as_ref().map(|state| state.alphas.clone()),
            iteration: 0,
            lower: first.lower.clone(),
            upper: first.upper.clone(),
            score: init_score,
        };
        let mut fold = first;
        let mut fold_secs_max = fold.secs;
        let mut iter = 0usize;
        let mut folds_completed = 1usize;
        let mut lr = config.learning_rate;
        let mut no_improve = 0usize;
        let mut last_improvement = f64::NAN;
        let mut point_failures = 0usize;
        let mut total_gradient_skips = 0usize;
        let exit: &'static str = loop {
            let score = score_margins(&fold.lower, &rows.lower_thresholds);
            if iter > 0 {
                last_improvement = score.hinge - best.score.hinge;
                if score_improves(score, best.score) {
                    best = MarginCheckpoint {
                        relu: runtime.snapshot_graph(),
                        bilinear: matmul.as_ref().map(|state| state.alphas.clone()),
                        iteration: iter,
                        lower: fold.lower.clone(),
                        upper: fold.upper.clone(),
                        score,
                    };
                }
                if last_improvement > f64::from(config.tolerance) {
                    no_improve = 0;
                } else {
                    no_improve += 1;
                }
            }
            if score.verified == k {
                break "all_verified";
            }
            if iter + 1 >= config.iterations {
                break "iters_exhausted";
            }
            if iter > 0 && no_improve >= config.early_stop_patience {
                break "converged";
            }
            // Keep room for the certify-once pass: a publication that cannot
            // be certified is not published, so running the ascent into the
            // deadline would throw the whole lane away.
            let remaining_secs = || {
                config.deadline.map(|deadline| {
                    deadline
                        .saturating_duration_since(Instant::now())
                        .as_secs_f64()
                })
            };
            if remaining_secs().is_some_and(|remaining| remaining <= 2.0 * fold_secs_max) {
                break "certify_reserve";
            }

            let gradients = match self.margin_alpha_gradients(
                ctx,
                node_bounds,
                &relu_names,
                matmul.as_ref(),
                &fold,
                &rows,
            ) {
                Ok(gradients) => gradients,
                Err(error) => {
                    debug!("#vit-margin-alpha: gradient refused at iter {iter}: {error}");
                    emit_margin_alpha_error("gradient", &error);
                    break "gradient_error";
                }
            };
            // #vit-margin-alpha review (2026-09-24): the check above reserves
            // two FOLDS, but every iteration first pays for the per-row
            // point forwards inside `margin_alpha_gradients` (up to one
            // pointwise forward per unverified row). When that cost is
            // comparable to a fold, an iteration admitted with `2 × fold` left
            // reached its own fold with less than two folds of budget, and
            // the certify-once pass then died on the deadline
            // (`published=0_deadline`): the whole lane discarded on exactly
            // the budget-bound roots it exists for. Re-check here, where the
            // remaining work is known to be this iteration's fold plus the
            // certify fold, with a quarter-fold of headroom for fold-time
            // variance. Nothing has been updated yet, so breaking here leaves
            // `best` and the runtime exactly as the last scored fold left them.
            if remaining_secs().is_some_and(|remaining| remaining <= 2.25 * fold_secs_max) {
                break "certify_reserve";
            }
            point_failures += gradients.point_failures;
            let MarginGradients {
                relu: relu_gradients,
                row_relu,
                bilinear: mut bilinear_gradients,
                ..
            } = gradients;

            // B3: shared ReLU α (and, under `matmul_alpha`, the bilinear
            // parameters) through the single contracted write site. The
            // upper-path α receives a ZERO gradient: every margin objective is
            // a lower objective, so the upper path is left exactly where it is.
            let zeros_upper: Vec<Array1<f32>> = relu_gradients
                .iter()
                .map(|gradient| Array1::<f32>::zeros(gradient.len()))
                .collect();
            let mut no_bilinear_alphas: HashMap<String, Array4<f32>> = HashMap::new();
            let mut no_bilinear_m: HashMap<String, Array4<f32>> = HashMap::new();
            let mut no_bilinear_v: HashMap<String, Array4<f32>> = HashMap::new();
            let mut no_mul_alphas: HashMap<String, Array2<f32>> = HashMap::new();
            let mut no_mul_grads: HashMap<String, Array2<f32>> = HashMap::new();
            let mut no_mul_m: HashMap<String, Array2<f32>> = HashMap::new();
            let mut no_mul_v: HashMap<String, Array2<f32>> = HashMap::new();
            let has_bilinear_update = matmul.is_some();
            let (bilinear_alphas, bilinear_m, bilinear_v) = match matmul.as_mut() {
                Some(state) => (&mut state.alphas, &mut state.adam_m, &mut state.adam_v),
                None => (
                    &mut no_bilinear_alphas,
                    &mut no_bilinear_m,
                    &mut no_bilinear_v,
                ),
            };
            alpha_update::update_all_alphas(
                &mut runtime,
                config,
                &relu_gradients,
                &zeros_upper,
                &BTreeMap::new(),
                &BTreeMap::new(),
                &BTreeMap::new(),
                bilinear_alphas,
                &mut bilinear_gradients,
                bilinear_m,
                bilinear_v,
                &mut no_mul_alphas,
                &mut no_mul_grads,
                &mut no_mul_m,
                &mut no_mul_v,
                has_bilinear_update,
                false,
                false,
                lr,
                iter,
                &mut total_gradient_skips,
            )?;

            // B2: each slot's δ takes ITS row's own gradient, on top of the
            // shared step (the #spec-axis-alpha slice 2d contract).
            if spec_slots > 0 {
                let adam = config.adam_params(lr, iter + 1);
                let slot_rows: Vec<usize> = runtime.graph().spec_slot_rows.clone();
                for (relu_idx, name) in relu_names.iter().enumerate() {
                    let Some(width) = runtime.graph().relu_len(name) else {
                        continue;
                    };
                    let mut per_slot = Array2::<f32>::zeros((slot_rows.len(), width));
                    let mut touched = false;
                    for (slot, &row) in slot_rows.iter().enumerate() {
                        let Some(Some(per_node)) = row_relu.get(row) else {
                            continue;
                        };
                        let Some(gradient) = per_node.get(relu_idx) else {
                            continue;
                        };
                        let reduced = runtime.graph().reduce_gradient(name, gradient);
                        if reduced.len() == width {
                            per_slot.row_mut(slot).assign(&reduced);
                            touched = true;
                        }
                    }
                    if touched {
                        runtime.graph_mut().update_spec_deltas_adam(
                            name,
                            &per_slot,
                            iter + 1,
                            &adam,
                        );
                    }
                }
            }

            lr *= config.lr_decay;
            iter += 1;
            match self.margin_alpha_fold(
                ctx,
                node_bounds,
                &runtime,
                matmul.as_ref().map(|state| &state.alphas),
                frozen_mul,
                &rows.seed,
                true,
            ) {
                Ok(next) => {
                    fold_secs_max = fold_secs_max.max(next.secs);
                    fold = next;
                    folds_completed += 1;
                }
                Err(error) if error.is_deadline_exceeded() => break "budget",
                Err(error) => {
                    debug!("#vit-margin-alpha: fold refused at iter {iter}: {error}");
                    emit_margin_alpha_error("fold", &error);
                    break "fold_error";
                }
            }
        };
        if total_gradient_skips > 0 {
            debug!(
                "#vit-margin-alpha: {total_gradient_skips} non-finite gradient update(s) skipped"
            );
        }

        // COMMIT: restore the WHOLE optimizer state that produced the
        // best-scoring fold, then CERTIFY ONCE with exactly that state.
        //
        // The certify pass captures intermediates like every ascent fold did.
        // That is not for the gradients (none are taken): a capturing walk
        // never takes the GPU-suffix shortcut, which returns concretized
        // bounds WITHOUT the final linear relation the exact-seed face
        // requires, so a non-capturing pass on a GPU host could refuse the
        // publication outright — and it keeps the certify pass on the same
        // code path as the fold it re-certifies (`recompute_identical`).
        runtime.restore_graph(&best.relu);
        let committed_bilinear = best.bilinear.as_ref();
        let certified = self.margin_alpha_fold(
            ctx,
            node_bounds,
            &runtime,
            committed_bilinear,
            frozen_mul,
            &rows.seed,
            true,
        );
        let (publication, recompute_identical, published) = match certified {
            Ok(fresh) => {
                let identical =
                    bits_equal(&fresh.lower, &best.lower) && bits_equal(&fresh.upper, &best.upper);
                let (lower_c, upper_c) = rows.original_bounds(&fresh.lower, &fresh.upper);
                match MarginAlphaPublication::new(
                    rows.rows,
                    lower_c,
                    upper_c,
                    ctx.input,
                    best.iteration,
                    spec_slots,
                    matmul_alpha_nodes,
                ) {
                    Some(publication) => (Some(publication), identical, "1"),
                    None => (None, identical, "0_malformed"),
                }
            }
            Err(error) if error.is_deadline_exceeded() => (None, false, "0_deadline"),
            Err(error) => {
                debug!("#vit-margin-alpha: certify-once pass refused: {error}");
                emit_margin_alpha_error("certify", &error);
                (None, false, "0_refused")
            }
        };

        #[cfg(test)]
        LAST_MARGIN_COMMIT.with(|cell| {
            let moved = best.bilinear.as_ref().is_some_and(|alphas| {
                alphas.values().any(|alpha| {
                    alpha
                        .iter()
                        .any(|&value| value.to_bits() != 1.0_f32.to_bits())
                })
            });
            *cell.borrow_mut() = Some(MarginCommitRecord {
                relu: best.relu.clone(),
                bilinear: best.bilinear.clone(),
                seed: rows.seed.clone(),
                relu_name_to_idx: runtime.relu_name_to_idx().clone(),
                committed_iteration: best.iteration,
                folds_completed,
                exit,
                committed_lower_objective: best.lower.clone(),
                matmul_alpha_moved: moved,
                frozen_mul_binary: frozen.mul_binary.clone(),
                reciprocal_at_start: reciprocal_at_start.clone(),
                reciprocal_at_commit: reciprocal_alpha_bits(&best.relu),
            });
        });

        let at_clamp = runtime.graph().clamp_saturation();
        let mut alpha_state = runtime.into_graph_alpha_state();
        // Downstream consumers key spec rows by OUTPUT row; the lane's slots
        // are keyed by MARGIN row. The per-spec gain travels in the
        // publication instead.
        alpha_state.clear_spec_slots();
        if let Some(publication) = publication {
            alpha_state.attach_margin_alpha_publication(publication);
        }
        let secs = started.elapsed().as_secs_f64();
        emit_margin_alpha_line(&MarginLaneSummary {
            rows: k,
            spec_slots,
            matmul_alpha_nodes,
            relu_alpha_nodes: relu_names.len(),
            folds: folds_completed,
            init: init_score,
            best: best.score,
            committed: best.iteration,
            at_clamp,
            exit,
            published,
            recompute_identical,
            point_failures,
            frozen_token: frozen.token(),
            frozen_mul_binary_nodes: frozen.mul_binary_nodes,
            frozen_reciprocal_nodes: frozen.reciprocal_nodes,
            secs,
        });
        // Keep the historical `[alpha-ascent]` line so every existing parser
        // still sees one exit record per ascent; `lower_sum` is not a quantity
        // this lane computes, so it is printed as NaN rather than invented.
        emit_alpha_ascent_exit_line(
            config,
            folds_completed,
            "margin_lane",
            last_improvement,
            f32::NAN,
            secs,
            at_clamp,
        );

        Ok(DagAlphaCollectionArtifact {
            output_bounds,
            alpha_state,
            reference_bounds: reference_bounds.into_current(),
            reference_bounds_source: node_bounds_source,
            completed_iterations: folds_completed,
            optimizer_updates_completed: best.iteration,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bounds::AlphaCrownConfig;
    use crate::layers::{LinearLayer, MatMulLayer, ReLULayer, SoftmaxLayer};
    use crate::network::GraphNode;
    use ndarray::{arr1, arr2, ArrayD, IxDyn};

    /// How the attention product `q·kᵀ` is represented.
    #[derive(Clone, Copy, Debug)]
    enum Product {
        /// `BilinearCrown` — what ny-build lowers every var×var ONNX MatMul to
        /// (`ny-build/src/convert/matmul.rs`), i.e. every vit_2023 attention
        /// product. Its parameters come from the DAG init.
        Bilinear,
        /// A var×var `Layer::MatMul` (the decoder/whisper shape). Its
        /// parameters exist only inside the margin ascent, so this variant
        /// exercises the `MatMul` dispatch arm and the resident-lane override.
        MatMul,
    }

    /// Attention-shaped head: two ReLU projections, their bilinear product,
    /// a sign-mixing projection so the next ReLU is genuinely unstable, and a
    /// linear head.
    fn attention_head_graph(product: Product) -> GraphNetwork {
        let mut graph = GraphNetwork::new();
        let q = LinearLayer::new(
            arr2(&[[0.8_f32, -0.3], [0.4, 0.9]]),
            Some(arr1(&[0.1_f32, -0.2])),
        )
        .expect("q projection");
        graph.add_node(GraphNode::from_input("proj_q", Layer::Linear(q)));
        graph.add_node(GraphNode::new(
            "relu_q",
            Layer::ReLU(ReLULayer),
            vec!["proj_q".to_string()],
        ));
        let k = LinearLayer::new(
            arr2(&[[-0.5_f32, 0.7], [0.6, -0.4]]),
            Some(arr1(&[-0.1_f32, 0.15])),
        )
        .expect("k projection");
        graph.add_node(GraphNode::from_input("proj_k", Layer::Linear(k)));
        graph.add_node(GraphNode::new(
            "relu_k",
            Layer::ReLU(ReLULayer),
            vec!["proj_k".to_string()],
        ));
        let qk = match product {
            Product::Bilinear => Layer::BilinearCrown(BilinearCrownLayer::new(true, Some(0.5))),
            Product::MatMul => Layer::MatMul(MatMulLayer::new(true, Some(0.5))),
        };
        graph.add_node(GraphNode::binary("qk", qk, "relu_q", "relu_k"));
        let mix = LinearLayer::new(
            arr2(&[[1.0_f32, -1.0], [-0.5, 1.0]]),
            Some(arr1(&[0.05_f32, -0.1])),
        )
        .expect("mix projection");
        graph.add_node(GraphNode::new(
            "mix",
            Layer::Linear(mix),
            vec!["qk".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "relu_mix",
            Layer::ReLU(ReLULayer),
            vec!["mix".to_string()],
        ));
        let head = LinearLayer::new(
            arr2(&[[1.0_f32, -0.5], [0.3, 0.8], [-0.7, 0.6]]),
            Some(arr1(&[0.05_f32, -0.1, 0.2])),
        )
        .expect("head");
        graph.add_node(GraphNode::new(
            "head",
            Layer::Linear(head),
            vec!["relu_mix".to_string()],
        ));
        graph.set_output("head");
        graph
    }

    const CENTER: [f32; 4] = [0.5, 1.0, 1.0, 0.5];
    const RADIUS: [f32; 4] = [0.5, 0.5, 0.5, 0.5];

    fn input_box() -> BoundedTensor {
        let lower: Vec<f32> = CENTER.iter().zip(&RADIUS).map(|(c, r)| c - r).collect();
        let upper: Vec<f32> = CENTER.iter().zip(&RADIUS).map(|(c, r)| c + r).collect();
        BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2, 2]), lower).expect("lower"),
            ArrayD::from_shape_vec(IxDyn(&[2, 2]), upper).expect("upper"),
        )
        .expect("input box")
    }

    fn margin_rows() -> Vec<AlphaSpecEarlyExit> {
        // Six outputs ([2, 3] head); thresholds far above reach, so every row
        // stays unverified and the ascent runs every iteration.
        vec![
            AlphaSpecEarlyExit {
                objective: vec![1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                threshold: 50.0,
                verify_upper_bound: false,
            },
            AlphaSpecEarlyExit {
                objective: vec![0.0, 0.0, 1.0, 0.0, -1.0, 0.0],
                threshold: 50.0,
                verify_upper_bound: false,
            },
            AlphaSpecEarlyExit {
                objective: vec![0.0, 1.0, 0.0, -1.0, 0.0, 0.5],
                threshold: 50.0,
                verify_upper_bound: false,
            },
        ]
    }

    fn margin_config(iterations: usize, gradient_key: bool) -> AlphaCrownConfig {
        AlphaCrownConfig {
            iterations,
            learning_rate: 0.5,
            early_stop_patience: 1000,
            adaptive_skip: false,
            root_alpha_margin: true,
            root_alpha_margin_gradient: gradient_key,
            matmul_alpha: true,
            alpha_spec_slots: 3,
            spec_ascent: AlphaSpecAscent::new(margin_rows()),
            ..AlphaCrownConfig::default()
        }
    }

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    /// Named inputs for the admission predicate, so each case below states
    /// exactly which condition it flips (twelve positional arguments are a
    /// transposition waiting to happen).
    #[derive(Clone, Copy)]
    struct AdmissionCase {
        requested: bool,
        has_spec_objective: bool,
        returns_state: bool,
        eligibility: MarginGradientEligibility,
        matmul_alpha_on_var_matmul: bool,
        mul_binary: bool,
        s_shaped: bool,
        sqrt: bool,
        reciprocal: bool,
        freeze_mul_reciprocal: bool,
        invprop: bool,
        iterations: usize,
    }

    impl AdmissionCase {
        fn vit_root() -> Self {
            Self {
                requested: true,
                has_spec_objective: true,
                returns_state: true,
                eligibility: MarginGradientEligibility::NonReluOptimizerState,
                matmul_alpha_on_var_matmul: false,
                mul_binary: false,
                s_shaped: false,
                sqrt: false,
                reciprocal: false,
                freeze_mul_reciprocal: false,
                invprop: false,
                iterations: 5,
            }
        }

        fn decide(self) -> MarginAscentAdmission {
            margin_ascent_admission(
                self.requested,
                self.has_spec_objective,
                self.returns_state,
                self.eligibility,
                self.matmul_alpha_on_var_matmul,
                self.mul_binary,
                self.s_shaped,
                self.sqrt,
                self.reciprocal,
                self.freeze_mul_reciprocal,
                self.invprop,
                self.iterations,
            )
        }
    }

    #[test]
    fn margin_ascent_admission_takes_only_non_resident_state_returning_roots() {
        let base = AdmissionCase::vit_root();
        assert_eq!(
            base.decide(),
            MarginAscentAdmission::Admitted,
            "bilinear state is exactly what the resident lane refuses and this lane takes"
        );
        assert_eq!(
            AdmissionCase {
                eligibility: MarginGradientEligibility::NonAnalyticChain,
                ..base
            }
            .decide(),
            MarginAscentAdmission::Admitted,
            "this lane owns its gradient; the configured gradient method does not gate it"
        );
        assert_eq!(
            AdmissionCase {
                eligibility: MarginGradientEligibility::Eligible,
                ..base
            }
            .decide(),
            MarginAscentAdmission::ResidentLaneEligible,
            "a ReLU-only graph must keep the resident lane byte-identically"
        );
        assert_eq!(
            AdmissionCase {
                eligibility: MarginGradientEligibility::Eligible,
                matmul_alpha_on_var_matmul: true,
                ..base
            }
            .decide(),
            MarginAscentAdmission::Admitted,
            "matmul_alpha on a var x var MatMul graph must not be dropped by the resident lane"
        );
        for conflict in [
            MarginGradientEligibility::ConflictMultiobjJointAlpha,
            MarginGradientEligibility::ConflictRootAlphaTrue,
            MarginGradientEligibility::ConflictBothAlphaPolicies,
        ] {
            for matmul_alpha_on_var_matmul in [false, true] {
                assert_eq!(
                    AdmissionCase {
                        eligibility: conflict,
                        matmul_alpha_on_var_matmul,
                        ..base
                    }
                    .decide(),
                    MarginAscentAdmission::ConflictingAlphaPolicy
                );
            }
        }
        assert_eq!(
            AdmissionCase {
                requested: false,
                eligibility: MarginGradientEligibility::NotRequested,
                ..base
            }
            .decide(),
            MarginAscentAdmission::NotRequested
        );
        assert_eq!(
            AdmissionCase {
                has_spec_objective: false,
                eligibility: MarginGradientEligibility::MissingSpecObjective,
                ..base
            }
            .decide(),
            MarginAscentAdmission::MissingSpecObjective
        );
        assert_eq!(
            AdmissionCase {
                returns_state: false,
                ..base
            }
            .decide(),
            MarginAscentAdmission::BoundsOnlyCaller
        );
        for flip in 0..4 {
            let case = AdmissionCase {
                mul_binary: flip == 0,
                s_shaped: flip == 1,
                sqrt: flip == 2,
                reciprocal: flip == 3,
                ..base
            };
            assert_eq!(case.decide(), MarginAscentAdmission::UnsupportedAlphaFamily);
        }
        assert_eq!(
            AdmissionCase {
                invprop: true,
                ..base
            }
            .decide(),
            MarginAscentAdmission::InvpropActive
        );
        assert_eq!(
            AdmissionCase {
                iterations: 0,
                ..base
            }
            .decide(),
            MarginAscentAdmission::NoIterations
        );
    }

    /// P6: the freeze key admits EXACTLY the MulBinary / Reciprocal families
    /// (alone or together, as the fixed-index softmax plants them), never
    /// S-shaped or Sqrt, and never overrides any earlier refusal. Dark, every
    /// MulBinary/Reciprocal root still refuses (the historical predicate).
    #[test]
    fn margin_ascent_admission_freeze_admits_only_mul_binary_and_reciprocal() {
        let base = AdmissionCase::vit_root();
        for (mul_binary, reciprocal) in [(true, false), (false, true), (true, true)] {
            let dark = AdmissionCase {
                mul_binary,
                reciprocal,
                ..base
            };
            assert_eq!(
                dark.decide(),
                MarginAscentAdmission::UnsupportedAlphaFamily,
                "dark: mul_binary={mul_binary} reciprocal={reciprocal} must still refuse"
            );
            assert_eq!(
                AdmissionCase {
                    freeze_mul_reciprocal: true,
                    ..dark
                }
                .decide(),
                MarginAscentAdmission::Admitted,
                "armed: mul_binary={mul_binary} reciprocal={reciprocal} is held frozen"
            );
        }
        for (s_shaped, sqrt) in [(true, false), (false, true)] {
            assert_eq!(
                AdmissionCase {
                    mul_binary: true,
                    reciprocal: true,
                    s_shaped,
                    sqrt,
                    freeze_mul_reciprocal: true,
                    ..base
                }
                .decide(),
                MarginAscentAdmission::UnsupportedAlphaFamily,
                "the freeze key must not admit S-shaped ({s_shaped}) or Sqrt ({sqrt}) state"
            );
        }
        let armed = AdmissionCase {
            mul_binary: true,
            reciprocal: true,
            freeze_mul_reciprocal: true,
            ..base
        };
        assert_eq!(
            AdmissionCase {
                requested: false,
                ..armed
            }
            .decide(),
            MarginAscentAdmission::NotRequested,
            "the freeze key alone never runs the lane"
        );
        assert_eq!(
            AdmissionCase {
                returns_state: false,
                ..armed
            }
            .decide(),
            MarginAscentAdmission::BoundsOnlyCaller
        );
        assert_eq!(
            AdmissionCase {
                invprop: true,
                ..armed
            }
            .decide(),
            MarginAscentAdmission::InvpropActive
        );
        assert_eq!(
            AdmissionCase {
                eligibility: MarginGradientEligibility::ConflictRootAlphaTrue,
                ..armed
            }
            .decide(),
            MarginAscentAdmission::ConflictingAlphaPolicy
        );
        // On a family-free graph the key changes nothing.
        assert_eq!(
            AdmissionCase {
                freeze_mul_reciprocal: true,
                ..base
            }
            .decide(),
            base.decide()
        );
    }

    #[test]
    fn margin_rows_flip_upper_rows_exactly_and_refuse_malformed_rows() {
        let ascent = AlphaSpecAscent::new(vec![
            AlphaSpecEarlyExit {
                objective: vec![1.0, -2.0],
                threshold: 0.5,
                verify_upper_bound: false,
            },
            AlphaSpecEarlyExit {
                objective: vec![-3.0, 0.25],
                threshold: -1.5,
                verify_upper_bound: true,
            },
        ])
        .expect("two rows");
        let rows = MarginRows::new(&ascent, 2).expect("well-formed rows");
        assert_eq!(rows.seed, arr2(&[[1.0_f32, -2.0], [3.0, -0.25]]));
        assert_eq!(rows.lower_thresholds, vec![0.5, 1.5]);
        // Row 1 is flipped: bounds on -c^T y map back to (-hi, -lo).
        let (lo, hi) = rows.original_bounds(&[0.1, -4.0], &[0.9, -2.0]);
        assert_eq!(lo, vec![0.1, 2.0]);
        assert_eq!(hi, vec![0.9, 4.0]);
        assert!(MarginRows::new(&ascent, 3).is_none(), "width mismatch");
        let nan = AlphaSpecAscent::new(vec![AlphaSpecEarlyExit {
            objective: vec![f32::NAN, 1.0],
            threshold: 0.0,
            verify_upper_bound: false,
        }])
        .expect("one row");
        assert!(MarginRows::new(&nan, 2).is_none(), "non-finite coefficient");
    }

    #[test]
    fn margin_score_ranks_by_hinge_and_treats_nan_as_worst() {
        let score = score_margins(&[1.0, -2.0, 0.5], &[0.0, 0.0, 1.0]);
        assert_eq!(score.verified, 1);
        assert_eq!(score.hinge, -2.5);
        assert_eq!(score.min_slack, -2.0);
        let nan = score_margins(&[f32::NAN], &[0.0]);
        assert_eq!(nan.hinge, f64::NEG_INFINITY);
        assert!(score_improves(score, nan));
        assert!(!score_improves(nan, score));
    }

    /// B4's gradient has a DIRECTION: at the input layer (operands affine in
    /// the input, so the envelope point is exact) the per-element gradient
    /// matches a finite difference of the certified bound in that parameter.
    ///
    /// The three product terms are chosen so the argmin lands on the three
    /// informative corner types at r = 0.5: `(x_l, y_l)` twice (L1 tight, L2
    /// not: derivative `+Δx·Δy` = +4.5 and +2.25) and `(x_u, y_u)` once (L2
    /// tight: `−Δx·Δy` = −2.5). A corner where BOTH planes are tight has a
    /// zero derivative and would make the comparison vacuous, so none is used.
    #[test]
    fn bilinear_margin_gradient_matches_finite_difference_at_the_input_layer() {
        // z = sum_l (x_l * y_l): one BilinearCrown on raw operands, no
        // downstream relaxation.
        let layer = BilinearCrownLayer::new(false, None);
        let a = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![-1.0_f32, 0.5, -1.5]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![2.0_f32, 1.5, -0.5]).unwrap(),
        )
        .unwrap();
        let b = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[3, 1]), vec![-0.5_f32, -0.25, -2.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[3, 1]), vec![1.0_f32, 2.0, 0.5]).unwrap(),
        )
        .unwrap();
        let spec = LinearBounds::from_spec_matrix(arr2(&[[1.0_f32]])).unwrap();
        let bound = |alphas: &Array4<f32>| -> (f64, Vec<f32>, Vec<f32>) {
            let (la, lb) = layer
                .propagate_linear_binary_with_alpha(&spec, &a, &b, Some(alphas))
                .unwrap();
            let lo = f64::from(la.concretize_sound(&a).lower()[0])
                + f64::from(lb.concretize_sound(&b).lower()[0]);
            // Argmin corners of each side's lower relation.
            let a_lo: Vec<f32> = a.lower().iter().copied().collect();
            let a_hi: Vec<f32> = a.upper().iter().copied().collect();
            let b_lo: Vec<f32> = b.lower().iter().copied().collect();
            let b_hi: Vec<f32> = b.upper().iter().copied().collect();
            let a_star = row_argmin_corner(&la, 0, &a_lo, &a_hi).unwrap();
            let b_star = row_argmin_corner(&lb, 0, &b_lo, &b_hi).unwrap();
            (lo, a_star, b_star)
        };
        let alphas = Array4::<f32>::from_elem((4, 1, 1, 3), 0.5);
        let (_, a_star, b_star) = bound(&alphas);
        assert_eq!(a_star, vec![-1.0, 0.5, -0.5], "x* corners: l, l, u");
        assert_eq!(b_star, vec![-0.5, -0.25, 0.5], "y* corners: l, l, u");
        let mut gradient = Array4::<f32>::zeros((4, 1, 1, 3));
        let contributions = layer
            .margin_alpha_gradient_at_point(
                spec.lower_a().row(0),
                &a,
                &b,
                &a_star,
                &b_star,
                &mut gradient,
            )
            .unwrap();
        assert_eq!(contributions, 3, "every product term contributes");
        let expected = [4.5_f32, 2.25, -2.5];
        for (l, &value) in expected.iter().enumerate() {
            assert!(
                (gradient[[0, 0, 0, l]] - value).abs() <= 1e-5,
                "element {l}: gradient {} vs the corner identity ±Δx·Δy = {value}",
                gradient[[0, 0, 0, l]]
            );
        }
        // The certified bound is exactly affine in r while the argmin corners
        // stay fixed, so a coarse step has no truncation error and keeps f32
        // noise small; ±0.01 moves no coefficient sign (the smallest is 0.25,
        // the largest shift 0.03).
        let step = 1e-2_f32;
        for l in 0..3 {
            let mut plus = alphas.clone();
            plus[[0, 0, 0, l]] += step;
            let mut minus = alphas.clone();
            minus[[0, 0, 0, l]] -= step;
            let fd = (bound(&plus).0 - bound(&minus).0) / (2.0 * f64::from(step));
            let analytic = f64::from(gradient[[0, 0, 0, l]]);
            assert!(
                (fd - analytic).abs() <= 1e-3 * (1.0 + fd.abs()),
                "element {l}: envelope gradient {analytic} vs finite difference {fd}"
            );
            assert_eq!(
                gradient[[1, 0, 0, l]],
                0.0,
                "upper-computation slot untouched"
            );
            assert_eq!(
                gradient[[3, 0, 0, l]],
                0.0,
                "upper-computation slot untouched"
            );
        }
    }

    /// CERTIFY ONCE. The publication the lane attaches must equal, BIT FOR
    /// BIT, a fresh certified pass run independently with the committed state
    /// (shared α + per-spec δ + interpolation parameters) — and must contain
    /// the network on sampled points of the box. Both product representations
    /// run: `BilinearCrown` (vit's) and a var×var `MatMul` (the dispatch arm
    /// and the resident-lane override).
    #[test]
    fn margin_alpha_published_bound_recomputes_bit_identically() {
        let _env = ny_test_utils::env::lock_env_shared();
        for product in [Product::Bilinear, Product::MatMul] {
            certify_once_round_trip(product);
        }
    }

    fn certify_once_round_trip(product: Product) {
        let graph = attention_head_graph(product);
        let input = input_box();
        let config = margin_config(12, true);
        LAST_MARGIN_COMMIT.with(|cell| cell.borrow_mut().take());

        let artifact = graph
            .propagate_dag_alpha_crown_collect_with_engine(&input, &config, None)
            .expect("margin lane runs")
            .expect("the DAG optimizer path, not an early return");
        let record = LAST_MARGIN_COMMIT
            .with(|cell| cell.borrow_mut().take())
            .unwrap_or_else(|| panic!("{product:?}: the margin lane ran and recorded its commit"));
        let publication = artifact
            .alpha_state
            .margin_alpha_publication()
            .unwrap_or_else(|| panic!("{product:?}: certify-once publication attached"))
            .clone();
        assert!(
            record.folds_completed >= 2,
            "{product:?}: the ascent must actually iterate (folds={}, exit={})",
            record.folds_completed,
            record.exit
        );
        assert!(
            !artifact.alpha_state.has_spec_deltas(),
            "{product:?}: margin-row δ slots must not leak to output-row consumers"
        );
        assert!(
            record.frozen_mul_binary.is_none() && record.reciprocal_at_start.is_empty(),
            "{product:?}: with the freeze key dark (and no MulBinary/Reciprocal node) the \
             lane must keep the historical no-MulBinary-map route"
        );
        assert_eq!(
            publication.spec_slots(),
            3,
            "{product:?}: one δ slot per margin row"
        );
        assert_eq!(
            publication.matmul_alpha_nodes(),
            1,
            "{product:?}: the one attention product carries interpolation parameters"
        );
        assert!(
            record
                .bilinear
                .as_ref()
                .is_some_and(|alphas| alphas.contains_key("qk")),
            "{product:?}: the committed state includes the product's parameters"
        );
        eprintln!(
            "[alpha-margin-test] product={product:?} folds={} committed={} exit={} \
             matmul_alpha_moved={}",
            record.folds_completed,
            record.committed_iteration,
            record.exit,
            record.matmul_alpha_moved
        );

        // The publication is identity-bound to exactly these rows and box.
        let rows = margin_rows();
        let objectives: Vec<Vec<f32>> = rows.iter().map(|row| row.objective.clone()).collect();
        let thresholds: Vec<f32> = rows.iter().map(|row| row.threshold).collect();
        assert_eq!(
            publication.describes(&input, &objectives, &thresholds, false),
            Ok(())
        );

        // Independent recomputation with exactly the committed state, on the
        // same (capturing) path the lane certified with.
        let exec_order: Vec<String> = graph.exec_order().expect("exec order").to_vec();
        let (fresh, _) = graph
            .dag_alpha_margin_rows_pass(
                &input,
                &artifact.reference_bounds,
                &exec_order,
                artifact.output_bounds.len(),
                input.len(),
                &record.relu_name_to_idx,
                &record.relu,
                None,
                record.bilinear.as_ref(),
                record.frozen_mul_binary.as_ref(),
                None,
                &record.seed,
                true,
            )
            .expect("fresh certified pass");
        let fresh_lower: Vec<f32> = fresh.lower().iter().copied().collect();
        let fresh_upper: Vec<f32> = fresh.upper().iter().copied().collect();
        // No row is upper-verification here, so objective == lower objective.
        assert_eq!(bits(&fresh_lower), bits(publication.lower()), "{product:?}");
        assert_eq!(bits(&fresh_upper), bits(publication.upper()), "{product:?}");
        assert_eq!(
            bits(&record.committed_lower_objective),
            bits(publication.lower()),
            "{product:?}: the published bound is the committed iterate's own certified value"
        );
        assert_eq!(
            publication.committed_iteration(),
            record.committed_iteration
        );

        // Sampled containment: 3^4 points (lower/mid/upper per coordinate).
        for sample in 0..81_usize {
            let values: Vec<f32> = (0..4)
                .map(|dim| {
                    let level = (sample / 3_usize.pow(dim as u32)) % 3;
                    let t = [0.0_f32, 0.5, 1.0][level];
                    CENTER[dim] - RADIUS[dim] + 2.0 * RADIUS[dim] * t
                })
                .collect();
            let point = ArrayD::from_shape_vec(IxDyn(&[2, 2]), values).unwrap();
            let exact = graph
                .propagate_ibp(&BoundedTensor::concrete(point).unwrap())
                .unwrap();
            let y_lo: Vec<f64> = exact.lower().iter().map(|v| f64::from(*v)).collect();
            let y_hi: Vec<f64> = exact.upper().iter().map(|v| f64::from(*v)).collect();
            for (r, row) in rows.iter().enumerate() {
                let (mut v_lo, mut v_hi, mut magnitude) = (0.0f64, 0.0f64, 0.0f64);
                for (c, &coefficient) in row.objective.iter().enumerate() {
                    let c64 = f64::from(coefficient);
                    if c64 >= 0.0 {
                        v_lo += c64 * y_lo[c];
                        v_hi += c64 * y_hi[c];
                    } else {
                        v_lo += c64 * y_hi[c];
                        v_hi += c64 * y_lo[c];
                    }
                    magnitude += c64.abs() * y_lo[c].abs().max(y_hi[c].abs());
                }
                // `[y_lo, y_hi]` is the outward IBP image of the concrete
                // point, so `[v_lo, v_hi]` already encloses the true margin up
                // to the f64 rounding of this <= 6-term sum. The slack below
                // covers exactly that and nothing more: an absolute `1e-5`
                // (the WIP's value) would hide an uncharged f32-scale error
                // term, which is the defect class this test exists to catch.
                let tol = 1e-12 * (1.0 + magnitude);
                assert!(
                    f64::from(publication.lower()[r]) <= v_hi + tol,
                    "{product:?} sample {sample} row {r}: published lower {} above the \
                     network's value [{v_lo}, {v_hi}] — a false-proof mechanism",
                    publication.lower()[r]
                );
                assert!(
                    f64::from(publication.upper()[r]) >= v_lo - tol,
                    "{product:?} sample {sample} row {r}: published upper {} below the \
                     network's value [{v_lo}, {v_hi}]",
                    publication.upper()[r]
                );
            }
        }
    }

    /// Without the gradient key the lane is dark: no publication, no commit
    /// record, and the legacy loop answers exactly as before.
    #[test]
    fn margin_alpha_lane_is_dark_without_the_gradient_key() {
        let _env = ny_test_utils::env::lock_env_shared();
        for product in [Product::Bilinear, Product::MatMul] {
            let graph = attention_head_graph(product);
            let input = input_box();
            LAST_MARGIN_COMMIT.with(|cell| cell.borrow_mut().take());
            let artifact = graph
                .propagate_dag_alpha_crown_collect_with_engine(
                    &input,
                    &margin_config(4, false),
                    None,
                )
                .expect("legacy loop runs")
                .expect("the DAG optimizer path");
            assert!(
                artifact.alpha_state.margin_alpha_publication().is_none(),
                "{product:?}"
            );
            assert!(
                LAST_MARGIN_COMMIT.with(|cell| cell.borrow().is_none()),
                "{product:?}"
            );
        }
    }

    /// The parent gate owns the rows: the typed child key alone must not run
    /// the lane (and must not publish), exactly as the env-only child never
    /// could.
    #[test]
    fn margin_alpha_lane_is_dark_without_the_parent_gate() {
        let _env = ny_test_utils::env::lock_env_shared();
        let graph = attention_head_graph(Product::Bilinear);
        let input = input_box();
        LAST_MARGIN_COMMIT.with(|cell| cell.borrow_mut().take());
        let config = AlphaCrownConfig {
            root_alpha_margin: false,
            ..margin_config(4, true)
        };
        let artifact = graph
            .propagate_dag_alpha_crown_collect_with_engine(&input, &config, None)
            .expect("legacy loop runs")
            .expect("the DAG optimizer path");
        assert!(artifact.alpha_state.margin_alpha_publication().is_none());
        assert!(LAST_MARGIN_COMMIT.with(|cell| cell.borrow().is_none()));
    }

    // -----------------------------------------------------------------------
    // #vit-margin-alpha-complex (P6): the lane over the FIXED-INDEX softmax.
    // -----------------------------------------------------------------------

    /// The attention head with a `Softmax` on the product, rewritten by the
    /// fixed-index decomposition (`softmax: complex_fixed_index`) into
    /// `… Exp → ReduceSum → Reciprocal → MulBinary → Clip(0,1)` — exactly the
    /// MulBinary + Reciprocal families that made the lane refuse every vit root
    /// that used the rewrite (`reason=unsupported_alpha_family`).
    fn attention_softmax_head_graph(input: &BoundedTensor) -> GraphNetwork {
        let mut graph = GraphNetwork::new();
        let q = LinearLayer::new(
            arr2(&[[0.8_f32, -0.3], [0.4, 0.9]]),
            Some(arr1(&[0.1_f32, -0.2])),
        )
        .expect("q projection");
        graph.add_node(GraphNode::from_input("proj_q", Layer::Linear(q)));
        graph.add_node(GraphNode::new(
            "relu_q",
            Layer::ReLU(ReLULayer),
            vec!["proj_q".to_string()],
        ));
        let k = LinearLayer::new(
            arr2(&[[-0.5_f32, 0.7], [0.6, -0.4]]),
            Some(arr1(&[-0.1_f32, 0.15])),
        )
        .expect("k projection");
        graph.add_node(GraphNode::from_input("proj_k", Layer::Linear(k)));
        graph.add_node(GraphNode::new(
            "relu_k",
            Layer::ReLU(ReLULayer),
            vec!["proj_k".to_string()],
        ));
        graph.add_node(GraphNode::binary(
            "qk",
            Layer::BilinearCrown(BilinearCrownLayer::new(true, Some(0.5))),
            "relu_q",
            "relu_k",
        ));
        graph.add_node(GraphNode::new(
            "sm",
            Layer::Softmax(SoftmaxLayer::new(-1)),
            vec!["qk".to_string()],
        ));
        let mix = LinearLayer::new(
            arr2(&[[1.0_f32, -1.0], [-0.5, 1.0]]),
            Some(arr1(&[0.05_f32, -0.1])),
        )
        .expect("mix projection");
        graph.add_node(GraphNode::new(
            "mix",
            Layer::Linear(mix),
            vec!["sm".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "relu_mix",
            Layer::ReLU(ReLULayer),
            vec!["mix".to_string()],
        ));
        let head = LinearLayer::new(
            arr2(&[[1.0_f32, -0.5], [0.3, 0.8], [-0.7, 0.6]]),
            Some(arr1(&[0.05_f32, -0.1, 0.2])),
        )
        .expect("head");
        graph.add_node(GraphNode::new(
            "head",
            Layer::Linear(head),
            vec!["relu_mix".to_string()],
        ));
        graph.set_output("head");
        let report = graph
            .decompose_softmax_complex_fixed_index(input)
            .expect("the fixed-index rewrite runs");
        assert_eq!(
            report.decomposed,
            vec!["sm".to_string()],
            "the one softmax must be decomposed (skipped: {:?})",
            report.skipped
        );
        assert!(
            graph
                .nodes
                .values()
                .any(|node| matches!(node.layer, Layer::MulBinary(_)))
                && graph
                    .nodes
                    .values()
                    .any(|node| matches!(node.layer, Layer::Reciprocal(_))),
            "the rewrite plants the MulBinary and Reciprocal this lane must freeze"
        );
        graph
    }

    fn frozen_margin_config(iterations: usize) -> AlphaCrownConfig {
        AlphaCrownConfig {
            margin_alpha_freeze_mul_reciprocal: true,
            ..margin_config(iterations, true)
        }
    }

    /// P6 CERTIFY ONCE + FROZEN + CONTAINMENT, on the fixed-index softmax graph.
    ///
    /// With `margin_alpha_freeze_mul_reciprocal` the lane must (1) actually run
    /// and publish on a graph carrying MulBinary + Reciprocal α state; (2) hold
    /// both families FIXED — the MulBinary map every fold used is the DAG
    /// init's r = 0.5 everywhere, and the Reciprocal tangent points are
    /// bit-identical at lane start, in the committed checkpoint and in the
    /// returned α state; (3) publish EXACTLY what an independent fresh certified
    /// pass with the committed state (frozen map included) computes, bit for
    /// bit; and (4) never publish a margin above the network's own value at any
    /// of 3^4 sampled points (a false-proof mechanism otherwise).
    #[test]
    fn margin_alpha_frozen_complex_softmax_published_bound_recomputes_bit_identically() {
        let _env = ny_test_utils::env::lock_env_shared();
        let input = input_box();
        let graph = attention_softmax_head_graph(&input);
        LAST_MARGIN_COMMIT.with(|cell| cell.borrow_mut().take());

        let artifact = graph
            .propagate_dag_alpha_crown_collect_with_engine(&input, &frozen_margin_config(12), None)
            .expect("margin lane runs over the fixed-index softmax")
            .expect("the DAG optimizer path, not an early return");
        let record = LAST_MARGIN_COMMIT
            .with(|cell| cell.borrow_mut().take())
            .expect(
                "the freeze key must admit the fixed-index softmax root: no commit record means \
                 the lane refused (check `[alpha-margin-frozen] … reason=` / `[alpha-margin] \
                 armed=0 reason=`)",
            );
        let publication = artifact
            .alpha_state
            .margin_alpha_publication()
            .expect("certify-once publication attached on the fixed-index softmax root")
            .clone();
        assert!(
            record.folds_completed >= 2,
            "the ascent must actually iterate (folds={}, exit={})",
            record.folds_completed,
            record.exit
        );
        assert_eq!(publication.spec_slots(), 3, "one δ slot per margin row");
        assert_eq!(
            publication.matmul_alpha_nodes(),
            1,
            "the attention product still carries (moving) interpolation parameters"
        );

        // (2) FROZEN. MulBinary: exactly the DAG init map, r = 0.5 everywhere.
        let frozen = record
            .frozen_mul_binary
            .as_ref()
            .expect("the MulBinary init map must be handed to every fold");
        assert!(
            frozen.contains_key("sm/sm_fx_mul"),
            "the rewrite's MulBinary is frozen: {:?}",
            frozen.keys().collect::<Vec<_>>()
        );
        for (name, alphas) in frozen {
            assert_eq!(alphas.nrows(), 2, "{name}: [2, n] interpolation map");
            assert!(
                alphas
                    .iter()
                    .all(|&value| value.to_bits() == 0.5_f32.to_bits()),
                "{name}: the frozen MulBinary α must be the DAG init (0.5), bit for bit"
            );
        }
        // Reciprocal: never moved, anywhere.
        assert!(
            record
                .reciprocal_at_start
                .iter()
                .any(|(name, _, _)| name == "sm/sm_fx_recip"),
            "the rewrite's Reciprocal carries α state the lane must freeze: {:?}",
            record
                .reciprocal_at_start
                .iter()
                .map(|(name, _, _)| name)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            record.reciprocal_at_start, record.reciprocal_at_commit,
            "the committed checkpoint must carry the Reciprocal α exactly as the lane found it"
        );
        assert_eq!(
            reciprocal_alpha_bits(&artifact.alpha_state),
            record.reciprocal_at_start,
            "the returned α state must carry the Reciprocal α exactly as the lane found it"
        );
        eprintln!(
            "[alpha-margin-test] graph=fixed_index_softmax folds={} committed={} exit={} \
             matmul_alpha_moved={} frozen_mul_binary_nodes={} frozen_reciprocal_nodes={}",
            record.folds_completed,
            record.committed_iteration,
            record.exit,
            record.matmul_alpha_moved,
            frozen.len(),
            record.reciprocal_at_start.len()
        );

        // (3) The publication is identity-bound and recomputes bit for bit.
        let rows = margin_rows();
        let objectives: Vec<Vec<f32>> = rows.iter().map(|row| row.objective.clone()).collect();
        let thresholds: Vec<f32> = rows.iter().map(|row| row.threshold).collect();
        assert_eq!(
            publication.describes(&input, &objectives, &thresholds, false),
            Ok(())
        );
        let exec_order: Vec<String> = graph.exec_order().expect("exec order").to_vec();
        let (fresh, _) = graph
            .dag_alpha_margin_rows_pass(
                &input,
                &artifact.reference_bounds,
                &exec_order,
                artifact.output_bounds.len(),
                input.len(),
                &record.relu_name_to_idx,
                &record.relu,
                None,
                record.bilinear.as_ref(),
                record.frozen_mul_binary.as_ref(),
                None,
                &record.seed,
                true,
            )
            .expect("fresh certified pass with the committed + frozen state");
        let fresh_lower: Vec<f32> = fresh.lower().iter().copied().collect();
        let fresh_upper: Vec<f32> = fresh.upper().iter().copied().collect();
        assert_eq!(bits(&fresh_lower), bits(publication.lower()));
        assert_eq!(bits(&fresh_upper), bits(publication.upper()));
        assert_eq!(
            bits(&record.committed_lower_objective),
            bits(publication.lower()),
            "the published bound is the committed iterate's own certified value"
        );

        // (4) Sampled containment against the REWRITTEN graph's own point
        // evaluation (its exactness is pinned by
        // `softmax_fixed_index_decomposition_exact_pointwise`).
        let mut compared = 0usize;
        for sample in 0..81_usize {
            let values: Vec<f32> = (0..4)
                .map(|dim| {
                    let level = (sample / 3_usize.pow(dim as u32)) % 3;
                    let t = [0.0_f32, 0.5, 1.0][level];
                    CENTER[dim] - RADIUS[dim] + 2.0 * RADIUS[dim] * t
                })
                .collect();
            let point = ArrayD::from_shape_vec(IxDyn(&[2, 2]), values).unwrap();
            let exact = graph
                .propagate_ibp(&BoundedTensor::concrete(point).unwrap())
                .unwrap();
            let y_lo: Vec<f64> = exact.lower().iter().map(|v| f64::from(*v)).collect();
            let y_hi: Vec<f64> = exact.upper().iter().map(|v| f64::from(*v)).collect();
            for (r, row) in rows.iter().enumerate() {
                let (mut v_lo, mut v_hi, mut magnitude) = (0.0f64, 0.0f64, 0.0f64);
                for (c, &coefficient) in row.objective.iter().enumerate() {
                    let c64 = f64::from(coefficient);
                    if c64 >= 0.0 {
                        v_lo += c64 * y_lo[c];
                        v_hi += c64 * y_hi[c];
                    } else {
                        v_lo += c64 * y_hi[c];
                        v_hi += c64 * y_lo[c];
                    }
                    magnitude += c64.abs() * y_lo[c].abs().max(y_hi[c].abs());
                }
                // Same f64-rounding-only slack as the historical round trip.
                let tol = 1e-12 * (1.0 + magnitude);
                assert!(
                    f64::from(publication.lower()[r]) <= v_hi + tol,
                    "sample {sample} row {r}: published lower {} above the network's value \
                     [{v_lo}, {v_hi}] through the frozen softmax — a false-proof mechanism",
                    publication.lower()[r]
                );
                assert!(
                    f64::from(publication.upper()[r]) >= v_lo - tol,
                    "sample {sample} row {r}: published upper {} below the network's value \
                     [{v_lo}, {v_hi}]",
                    publication.upper()[r]
                );
                compared += 1;
            }
        }
        assert_eq!(
            compared,
            81 * rows.len(),
            "every (sample, row) pair was compared"
        );
    }

    /// DARK BY DEFAULT: without the freeze key the fixed-index softmax root is
    /// refused exactly as before — no commit record, no publication — and the
    /// legacy loop answers.
    #[test]
    fn margin_alpha_complex_softmax_is_dark_without_the_freeze_key() {
        let _env = ny_test_utils::env::lock_env_shared();
        assert!(
            !AlphaCrownConfig::default().margin_alpha_freeze_mul_reciprocal,
            "the freeze key must be default-dark"
        );
        let input = input_box();
        let graph = attention_softmax_head_graph(&input);
        LAST_MARGIN_COMMIT.with(|cell| cell.borrow_mut().take());
        let config = margin_config(4, true);
        assert!(!config.margin_alpha_freeze_mul_reciprocal);
        let artifact = graph
            .propagate_dag_alpha_crown_collect_with_engine(&input, &config, None)
            .expect("legacy loop runs over the fixed-index softmax")
            .expect("the DAG optimizer path");
        assert!(
            artifact.alpha_state.margin_alpha_publication().is_none(),
            "a dark freeze key must not publish a margin bound on a MulBinary/Reciprocal root"
        );
        assert!(
            LAST_MARGIN_COMMIT.with(|cell| cell.borrow().is_none()),
            "a dark freeze key must leave the lane refused (unsupported_alpha_family)"
        );
    }

    /// The freeze key WITHOUT the parent gate is inert, exactly like the
    /// other two child keys: no lane, no publication.
    #[test]
    fn margin_alpha_freeze_key_alone_does_not_run_the_lane() {
        let _env = ny_test_utils::env::lock_env_shared();
        let input = input_box();
        let graph = attention_softmax_head_graph(&input);
        LAST_MARGIN_COMMIT.with(|cell| cell.borrow_mut().take());
        let config = AlphaCrownConfig {
            root_alpha_margin: false,
            ..frozen_margin_config(4)
        };
        let artifact = graph
            .propagate_dag_alpha_crown_collect_with_engine(&input, &config, None)
            .expect("legacy loop runs")
            .expect("the DAG optimizer path");
        assert!(artifact.alpha_state.margin_alpha_publication().is_none());
        assert!(LAST_MARGIN_COMMIT.with(|cell| cell.borrow().is_none()));
    }
}
