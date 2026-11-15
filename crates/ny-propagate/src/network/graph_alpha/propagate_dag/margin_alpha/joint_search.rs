// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #vit-root-alpha (ENGINE_PROGRAM §2 V2): the vit ROOT α search, certified
//! once (`solver.alpha-crown.margin_alpha_search: joint_f32`, dark).
//!
//! # What changes against the certified ascent
//!
//! The certified C-row margin lane (the parent module) optimises the ReLU
//! slopes and the attention McCormick planes with an ascent whose EVERY
//! iterate is a certified backward fold — R4's "certify every iterate" cost on
//! the vit root. This phase keeps the lane's admission, its margin rows, its
//! certified boxes and its first certified fold (the INCUMBENT, computed before
//! any state is handed over), and replaces the ascent with M5a's pattern:
//!
//! 1. LOWER the graph into the plain-data search copy
//!    ([`super::search_lowering`]) and SELF-CHECK it against the lane's own
//!    point forward (G1). A lowering refusal or mismatch refuses the phase.
//! 2. SEARCH, untrusted, on the injected backend (`ny-bbe`'s CPU f32 search in
//!    production, registered by ny-cli): ReLU slopes, the attention products'
//!    McCormick planes and — on the fixed-index softmax — the MulBinary
//!    `exp × 1/Σexp` planes that `#vit-margin-alpha-complex` froze, jointly,
//!    against the margin rows with the certified boxes frozen. The device
//!    never certifies; nothing the search returns is a number the lane
//!    publishes.
//! 3. INGEST (the gate): a hinted value reaches the lane's state only when its
//!    site, shape and width match the lane's own; a non-finite value keeps the
//!    incumbent's; every other value is clamped into `[0, 1]`. ReLU slopes are
//!    written only on neurons UNSTABLE in the certified boxes, through the
//!    runtime's clamping writer; product planes `[1]`/`[3]` (the upper
//!    direction, which no margin row reads) are never written.
//! 4. CERTIFY ONCE: one certified fold (`margin_alpha_fold`, the lane's own
//!    backward — V1's batched bilinear kernel when `bab.bilinear_gemm_kernel`
//!    is armed) with exactly the ingested state.
//! 5. KEEP ONLY IF STRICTLY BETTER: the candidate state is kept iff its margin
//!    score (hinge, then worst slack) strictly beats the incumbent's; else the
//!    runtime is restored to the incumbent bit for bit. The PUBLICATION is the
//!    per-row intersection of the incumbent's and the candidate's certified
//!    bounds — both are certified enclosures of the same rows over the same
//!    box — so it is never looser than the incumbent on any row. If some
//!    row's two enclosures are DISJOINT (one certified fold is unsound and
//!    nothing says which), the publication is WITHDRAWN and an unconditional
//!    `[bbe-alarm]` line is printed ([`resolve_publication`]).
//!
//! # Soundness
//!
//! Every published number is a certified fold of a state whose every parameter
//! is finite and inside its valid domain (the kernels clamp again); the search
//! only chooses WHICH valid relaxation is certified. A search bug costs the
//! +10 of a missed improvement, never a false bound: garbage parameters are
//! refused or clamped (`joint_search_fault_injection_*`), and the published
//! bound is re-derived bit for bit by an independent certified pass over the
//! recorded ingested state in the tests. Dark (`ascent`), this module is never
//! entered: the parent lane runs its ascent byte for byte.
//!
//! # Observable
//!
//! Armed, every return path prints exactly one fixed-schema line
//! `[bbe] phase=vit_root backend= refused= kept= incumbent_slack=
//! candidate_slack= published_slack= search_init_slack= search_best_slack= …`
//! plus the lane's `[alpha-margin] armed=1 … exit=joint_f32_…` line. The
//! `search_*` slacks are UNTRUSTED f32 search numbers; `incumbent_slack`,
//! `candidate_slack` and `published_slack` are certified. An armed run without
//! the `[bbe] phase=vit_root` line is VOID.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use ndarray::{Array2, Array4};
use ny_core::search::{GraphAlphaHint, GraphRootRequest, GraphRootStats};
use ny_core::Result;
use ny_tensor::BoundedTensor;

use super::super::super::reference_bounds::GraphAlphaReferenceBounds;
use super::super::super::runtime_state::DagAlphaRuntimeState;
use super::super::{emit_alpha_ascent_exit_line, DagAlphaCollectionArtifact, DagAlphaLoopContext};
use super::search_lowering::{self, GraphLowering, SiteKind};
use super::{
    emit_margin_alpha_error, emit_margin_alpha_line, score_improves, score_margins, FrozenFamilies,
    MarginFold, MarginLaneSummary, MarginRows, MarginScore, MatmulAlphaState,
};
use crate::bounds::MarginAlphaPublication;
use crate::layers::Layer;
use crate::network::core::{GraphNetwork, NETWORK_INPUT};
use crate::network::graph_alpha::bounds::AlphaReferenceBoundsSource;

/// Working set admitted to the search (the vit_2023 copy needs tens of MB).
pub(super) const JOINT_SEARCH_MEM_CAP: u64 = 1 << 30;

/// The certify-once reserve, in multiples of the incumbent fold's wall time
/// (the certified ascent's own reserve).
const CERTIFY_RESERVE_FOLDS: f64 = 2.25;

/// What the ingest gate admitted. Every value is finite and in `[0, 1]`.
#[derive(Default)]
pub(super) struct Ingested {
    /// `(relu index, neuron, lower slope)` writes, unstable neurons only.
    pub(super) relu: Vec<(usize, usize, f32)>,
    /// The candidate product planes (the incumbent map with `[0]`/`[2]`
    /// overwritten where hinted), `None` when the lane has no product state.
    pub(super) bilinear: Option<HashMap<String, Array4<f32>>>,
    /// The candidate MulBinary planes, `None` when the lane has none.
    pub(super) mul: Option<HashMap<String, Array2<f32>>>,
    pub(super) counts: IngestCounts,
}

/// Ingest bookkeeping for the observable: values written / refused per family
/// and hint entries that matched no site of the lane.
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct IngestCounts {
    pub(super) relu_written: usize,
    pub(super) relu_refused: usize,
    pub(super) bilinear_written: usize,
    pub(super) bilinear_refused: usize,
    pub(super) mul_written: usize,
    pub(super) mul_refused: usize,
    pub(super) unmatched: usize,
}

/// THE KEEP RULE: the candidate state replaces the incumbent only when its
/// certified margin score is STRICTLY better (the lane's own ranking).
pub(super) fn keep_candidate(candidate: MarginScore, incumbent: MarginScore) -> bool {
    score_improves(candidate, incumbent)
}

/// The per-row intersection of two certified enclosures of the same rows:
/// never looser than the incumbent, and a non-finite / NaN candidate side is
/// ignored. Returns `None` when some row's intersection is EMPTY (one of two
/// certified enclosures is unsound: a soundness alarm, never resolved by
/// guessing).
pub(super) fn intersect_rows(
    incumbent_lower: &[f32],
    incumbent_upper: &[f32],
    candidate_lower: &[f32],
    candidate_upper: &[f32],
) -> Option<(Vec<f32>, Vec<f32>)> {
    let lower: Vec<f32> = incumbent_lower
        .iter()
        .zip(candidate_lower)
        .map(|(&a, &b)| if b > a { b } else { a })
        .collect();
    let upper: Vec<f32> = incumbent_upper
        .iter()
        .zip(candidate_upper)
        .map(|(&a, &b)| if b < a { b } else { a })
        .collect();
    let disjoint = lower.iter().zip(&upper).any(|(l, u)| l > u);
    (!disjoint).then_some((lower, upper))
}

/// THE PUBLICATION RULE (review 2026-09-29). With no candidate fold (or a
/// candidate whose shape does not match the incumbent's) the incumbent's
/// certified bounds stand. With one, the per-row intersection of the two
/// certified enclosures is published. When some row's two enclosures are
/// DISJOINT, one of two certified folds of the same rows over the same box is
/// unsound and nothing says which — so the publication is WITHDRAWN (`None`):
/// the root keeps its own bounds, which is a downgrade only. Publishing the
/// incumbent there would be a guess that the incumbent is the sound one.
pub(super) fn resolve_publication(
    incumbent_lower: &[f32],
    incumbent_upper: &[f32],
    candidate: Option<(&[f32], &[f32])>,
) -> Option<(Vec<f32>, Vec<f32>)> {
    match candidate {
        Some((lower, upper))
            if lower.len() == incumbent_lower.len() && upper.len() == incumbent_upper.len() =>
        {
            intersect_rows(incumbent_lower, incumbent_upper, lower, upper)
        }
        _ => Some((incumbent_lower.to_vec(), incumbent_upper.to_vec())),
    }
}

/// The flat certified box of a node (the network input or a node's reference
/// bounds).
fn certified_box<'a>(
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

/// THE INGEST GATE. Reads the untrusted hint and returns only finite values
/// clamped into `[0, 1]`, on sites, shapes and widths that match the lane's own
/// state; everything else keeps the incumbent.
pub(super) fn ingest(
    lowering: &GraphLowering,
    hint: &GraphAlphaHint,
    runtime: &DagAlphaRuntimeState,
    input: &BoundedTensor,
    node_bounds: &HashMap<String, BoundedTensor>,
    matmul: Option<&MatmulAlphaState>,
    frozen: &FrozenFamilies,
) -> Ingested {
    let mut out = Ingested {
        bilinear: matmul.map(|state| state.alphas.clone()),
        mul: frozen.mul_binary.clone(),
        ..Ingested::default()
    };
    let site_at = |op: usize| lowering.sites.iter().find(|site| site.op == op);

    for (op, slopes) in &hint.relu {
        let Some(site) = site_at(*op).filter(|site| site.kind == SiteKind::Relu) else {
            out.counts.unmatched += 1;
            continue;
        };
        let Some(&relu_idx) = runtime.relu_name_to_idx().get(&site.node) else {
            out.counts.unmatched += 1;
            continue;
        };
        let Some(pre) = certified_box(&site.input_node, input, node_bounds) else {
            out.counts.unmatched += 1;
            continue;
        };
        if runtime.graph().relu_len(&site.node) != Some(slopes.len()) || pre.len() != slopes.len() {
            out.counts.unmatched += 1;
            continue;
        }
        for (j, ((&l, &u), &v)) in pre
            .lower()
            .iter()
            .zip(pre.upper().iter())
            .zip(slopes)
            .enumerate()
        {
            if !(l < 0.0 && u > 0.0) {
                continue;
            }
            if v.is_finite() {
                out.relu.push((relu_idx, j, v.clamp(0.0, 1.0)));
                out.counts.relu_written += 1;
            } else {
                out.counts.relu_refused += 1;
            }
        }
    }

    for (op, params) in &hint.bilinear {
        let Some(SiteKind::Bilinear { m, n, k }) = site_at(*op).map(|site| site.kind) else {
            out.counts.unmatched += 1;
            continue;
        };
        let Some(site) = site_at(*op) else {
            out.counts.unmatched += 1;
            continue;
        };
        let block = m * n * k;
        let Some(alphas) = out
            .bilinear
            .as_mut()
            .and_then(|map| map.get_mut(&site.node))
        else {
            out.counts.unmatched += 1;
            continue;
        };
        if alphas.shape() != [4, m, n, k] || params.len() != 4 * block {
            out.counts.unmatched += 1;
            continue;
        }
        for plane in [0usize, 2] {
            for e in 0..block {
                let v = params[plane * block + e];
                if v.is_finite() {
                    let (i, j, l) = (e / (n * k), (e / k) % n, e % k);
                    alphas[[plane, i, j, l]] = v.clamp(0.0, 1.0);
                    out.counts.bilinear_written += 1;
                } else {
                    out.counts.bilinear_refused += 1;
                }
            }
        }
    }

    for (op, params) in &hint.mul_binary {
        let Some(SiteKind::MulBinary { n }) = site_at(*op).map(|site| site.kind) else {
            out.counts.unmatched += 1;
            continue;
        };
        let Some(site) = site_at(*op) else {
            out.counts.unmatched += 1;
            continue;
        };
        let Some(alphas) = out.mul.as_mut().and_then(|map| map.get_mut(&site.node)) else {
            out.counts.unmatched += 1;
            continue;
        };
        if alphas.shape() != [2, n] || params.len() != 2 * n {
            out.counts.unmatched += 1;
            continue;
        }
        for plane in 0..2usize {
            for j in 0..n {
                let v = params[plane * n + j];
                if v.is_finite() {
                    alphas[[plane, j]] = v.clamp(0.0, 1.0);
                    out.counts.mul_written += 1;
                } else {
                    out.counts.mul_refused += 1;
                }
            }
        }
    }
    out
}

/// Everything the phase did, for the unconditional `[bbe]` line. Search
/// numbers are kept as FORMATTED text so no untrusted number can be compared
/// with zero by accident.
struct JointReport {
    backend: &'static str,
    refused: Option<String>,
    lowered_ops: usize,
    structural: usize,
    probed: usize,
    selfcheck_compared: usize,
    selfcheck_worst: f64,
    relu_sites: usize,
    bilinear_sites: usize,
    mul_sites: usize,
    iters: usize,
    evals: usize,
    search_s: f64,
    certify_s: f64,
    search_stop: &'static str,
    search_init_slack: String,
    search_best_slack: String,
    counts: IngestCounts,
    alarm: &'static str,
}

impl JointReport {
    fn new() -> Self {
        Self {
            backend: "none",
            refused: None,
            lowered_ops: 0,
            structural: 0,
            probed: 0,
            selfcheck_compared: 0,
            selfcheck_worst: f64::NAN,
            relu_sites: 0,
            bilinear_sites: 0,
            mul_sites: 0,
            iters: 0,
            evals: 0,
            search_s: 0.0,
            certify_s: 0.0,
            search_stop: "not_run",
            search_init_slack: "nan".into(),
            search_best_slack: "nan".into(),
            counts: IngestCounts::default(),
            alarm: "none",
        }
    }

    fn absorb(&mut self, stats: &GraphRootStats) {
        self.iters = stats.iters;
        self.evals = stats.evals;
        self.search_stop = stats.stop;
        self.search_init_slack = format!("{:.6}", stats.init_slack);
        self.search_best_slack = format!("{:.6}", stats.best_slack);
    }
}

/// `[bbe] phase=vit_root …` — UNCONDITIONAL on every armed return path (rate
/// limited like the lane's other lines).
fn emit_joint_line(
    report: &JointReport,
    rows: usize,
    kept: bool,
    incumbent: MarginScore,
    candidate: Option<MarginScore>,
    published: MarginScore,
) {
    static OCCURRENCES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let Some(occ) = crate::network::graph_alpha::retained_occurrence(&OCCURRENCES) else {
        return;
    };
    let c = report.counts;
    let refused: String = report
        .refused
        .as_deref()
        .unwrap_or("none")
        .chars()
        .map(|ch| {
            if ch.is_control() || ch == '"' || ch == ' ' {
                '_'
            } else {
                ch
            }
        })
        .take(160)
        .collect();
    eprintln!(
        "[bbe] phase=vit_root backend={} iters={} evals={} search_s={:.3} certify_s={:.3} \
         search_stop={} search_init_slack={} search_best_slack={} lowered_ops={} structural={} \
         probed={} selfcheck_nodes={} selfcheck_worst={:.3e} relu_sites={} bilinear_sites={} \
         mul_sites={} ingest=relu:{}/{},bilinear:{}/{},mul:{}/{},unmatched:{} \
         incumbent_slack={:.6e} candidate_slack={} published_slack={:.6e} verified={}/{} \
         kept={} bilinear_gemm={} alarm={} refused={refused} occ={occ}",
        report.backend,
        report.iters,
        report.evals,
        report.search_s,
        report.certify_s,
        report.search_stop,
        report.search_init_slack,
        report.search_best_slack,
        report.lowered_ops,
        report.structural,
        report.probed,
        report.selfcheck_compared,
        report.selfcheck_worst,
        report.relu_sites,
        report.bilinear_sites,
        report.mul_sites,
        c.relu_written,
        c.relu_refused,
        c.bilinear_written,
        c.bilinear_refused,
        c.mul_written,
        c.mul_refused,
        c.unmatched,
        incumbent.min_slack,
        candidate.map_or_else(|| "nan".to_string(), |s| format!("{:.6e}", s.min_slack)),
        published.min_slack,
        published.verified,
        rows,
        u8::from(kept),
        u8::from(crate::layers::collector_kernels::bilinear_gemm_armed()),
        report.alarm,
    );
}

/// Test-only record of what the phase ingested, certified and published, so a
/// test can re-derive the publication by an independent certified pass.
#[cfg(test)]
pub(super) struct JointSearchRecord {
    pub(super) refused: Option<String>,
    pub(super) backend_called: bool,
    pub(super) kept: bool,
    pub(super) counts: IngestCounts,
    pub(super) selfcheck_compared: usize,
    pub(super) incumbent_lower: Vec<f32>,
    pub(super) incumbent_upper: Vec<f32>,
    /// The candidate fold (lower-objective bounds), when one was certified.
    pub(super) candidate_lower: Option<Vec<f32>>,
    pub(super) candidate_upper: Option<Vec<f32>>,
    /// The published lower-objective bounds.
    pub(super) published_lower: Option<Vec<f32>>,
    pub(super) published_upper: Option<Vec<f32>>,
    /// The ingested state the candidate fold certified.
    pub(super) candidate_relu: Option<crate::bounds::GraphAlphaState>,
    pub(super) candidate_bilinear: Option<HashMap<String, Array4<f32>>>,
    pub(super) candidate_mul: Option<HashMap<String, Array2<f32>>>,
    pub(super) incumbent_bilinear: Option<HashMap<String, Array4<f32>>>,
    pub(super) seed: Array2<f32>,
    pub(super) relu_name_to_idx: HashMap<String, usize>,
}

#[cfg(test)]
thread_local! {
    pub(super) static LAST_JOINT_SEARCH: std::cell::RefCell<Option<JointSearchRecord>> =
        const { std::cell::RefCell::new(None) };
}

/// What the search half produced: the lowered copy (the ingest gate reads its
/// sites) and the untrusted hint.
struct SearchPlan {
    lowering: GraphLowering,
    hint: GraphAlphaHint,
}

impl GraphNetwork {
    /// Whether the graph carries a var×var product (`BilinearCrown` or
    /// `MatMul`), whose planes the phase can move only through the lane's
    /// `matmul_alpha` state.
    fn graph_has_product(&self) -> bool {
        self.nodes
            .values()
            .any(|node| matches!(node.layer, Layer::BilinearCrown(_) | Layer::MatMul(_)))
    }

    /// The search half: lower, self-check, search. Any `Err` is a refusal
    /// (the incumbent stands). `backend_called` records that the backend was
    /// consulted (test evidence; nothing reads it otherwise).
    #[allow(clippy::too_many_arguments)]
    fn joint_search_plan(
        &self,
        ctx: &DagAlphaLoopContext<'_>,
        rows: &MarginRows,
        matmul: Option<&MatmulAlphaState>,
        frozen: &FrozenFamilies,
        first: &MarginFold,
        runtime: &DagAlphaRuntimeState,
        node_bounds: &HashMap<String, BoundedTensor>,
        report: &mut JointReport,
        backend_called: &mut bool,
    ) -> std::result::Result<SearchPlan, String> {
        let config = ctx.config;
        if matmul.is_none() && self.graph_has_product() {
            return Err("matmul_alpha_dark".into());
        }
        // Review 2026-09-29: the lowering, its probes and the G1 self-check
        // poll no deadline, so a root that no longer has room for the one
        // certified fold must not start them (the ascent's own
        // `certify_reserve` exit, taken before any untrusted work).
        let reserve_secs = CERTIFY_RESERVE_FOLDS * first.secs;
        if config.deadline.is_some_and(|deadline| {
            deadline
                .saturating_duration_since(Instant::now())
                .as_secs_f64()
                <= reserve_secs
        }) {
            return Err("certify_reserve".into());
        }
        let started = Instant::now();
        let lowering = search_lowering::lower_graph(self, ctx.exec_order, ctx.input, node_bounds)
            .map_err(|e| format!("lowering: {e}"))?;
        report.lowered_ops = lowering.net.ops.len();
        report.structural = lowering.structural;
        report.probed = lowering.probed;
        for site in &lowering.sites {
            match site.kind {
                SiteKind::Relu => report.relu_sites += 1,
                SiteKind::Bilinear { .. } => report.bilinear_sites += 1,
                SiteKind::MulBinary { .. } => report.mul_sites += 1,
            }
        }
        let check = search_lowering::self_check(&lowering, self, ctx.input, ctx.engine)
            .map_err(|e| format!("self_check: {e}"))?;
        report.selfcheck_compared = check.compared;
        report.selfcheck_worst = check.worst;

        let backend = crate::network::graph_root_search::graph_root_backend()
            .ok_or_else(|| "no_backend".to_string())?;
        report.backend = backend.name();

        let out_width = lowering
            .net
            .tensor_sizes()
            .map_err(|e| format!("lowering: {e}"))?[lowering.net.output()];
        if out_width != rows.seed.ncols() {
            return Err(format!(
                "lowering: output width {out_width} != margin rows {}",
                rows.seed.ncols()
            ));
        }
        let mut init = GraphAlphaHint::default();
        for site in &lowering.sites {
            match site.kind {
                SiteKind::Relu => {
                    if let Some((lower, _)) = runtime.graph().relu_alpha_pair(&site.node) {
                        init.relu.push((site.op, lower.to_vec()));
                    }
                }
                SiteKind::Bilinear { m, n, k } => {
                    if let Some(alphas) = matmul.and_then(|state| state.alphas.get(&site.node)) {
                        if alphas.shape() == [4, m, n, k] {
                            init.bilinear
                                .push((site.op, alphas.iter().copied().collect()));
                        }
                    }
                }
                SiteKind::MulBinary { n } => {
                    if let Some(alphas) = frozen
                        .mul_binary
                        .as_ref()
                        .and_then(|map| map.get(&site.node))
                    {
                        if alphas.shape() == [2, n] {
                            init.mul_binary
                                .push((site.op, alphas.iter().copied().collect()));
                        }
                    }
                }
            }
        }
        let reserve = if reserve_secs.is_finite() && (0.0..1e6).contains(&reserve_secs) {
            Duration::from_secs_f64(reserve_secs)
        } else {
            Duration::ZERO
        };
        let request = GraphRootRequest {
            lo: ctx.input.lower().iter().map(|&v| f64::from(v)).collect(),
            hi: ctx.input.upper().iter().map(|&v| f64::from(v)).collect(),
            boxes: lowering.boxes.clone(),
            rows: (0..rows.len())
                .map(|r| rows.seed.row(r).iter().map(|&v| f64::from(v)).collect())
                .collect(),
            thresholds: rows
                .lower_thresholds
                .iter()
                .map(|&t| f64::from(t))
                .collect(),
            init,
            iters: config.iterations,
            lr: f64::from(config.learning_rate),
            deadline: config.deadline.map(|d| d.checked_sub(reserve).unwrap_or(d)),
            threads: 0,
            mem_cap_bytes: JOINT_SEARCH_MEM_CAP.saturating_sub(lowering.net.approx_bytes()),
        };
        *backend_called = true;
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            backend.graph_root_alpha(&lowering.net, &request)
        }));
        report.search_s = started.elapsed().as_secs_f64();
        let outcome = match outcome {
            Err(_) => return Err("the search backend panicked".into()),
            Ok(Err(refusal)) => return Err(format!("backend: {refusal}")),
            Ok(Ok(outcome)) => outcome,
        };
        report.absorb(&outcome.stats);
        Ok(SearchPlan {
            lowering,
            hint: outcome.alpha.into_hint().into_ingest(),
        })
    }

    /// V2: the joint search, certified once. Takes the lane's prepared state
    /// exactly where the certified ascent would; see the module docs.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub(super) fn run_margin_alpha_joint_search(
        &self,
        ctx: &DagAlphaLoopContext<'_>,
        rows: MarginRows,
        matmul: Option<MatmulAlphaState>,
        frozen: FrozenFamilies,
        first: MarginFold,
        mut runtime: DagAlphaRuntimeState,
        reference_bounds: GraphAlphaReferenceBounds,
        node_bounds_source: AlphaReferenceBoundsSource,
        output_bounds: BoundedTensor,
        started: Instant,
    ) -> Result<DagAlphaCollectionArtifact> {
        let config = ctx.config;
        let k = rows.len();
        let matmul_alpha_nodes = matmul.as_ref().map_or(0, |state| state.alphas.len());
        let relu_alpha_nodes = runtime.relu_nodes().len();
        let incumbent = score_margins(&first.lower, &rows.lower_thresholds);
        let mut report = JointReport::new();
        let mut backend_called = false;

        let plan = {
            let node_bounds = reference_bounds.current();
            self.joint_search_plan(
                ctx,
                &rows,
                matmul.as_ref(),
                &frozen,
                &first,
                &runtime,
                node_bounds,
                &mut report,
                &mut backend_called,
            )
        };

        let mut kept = false;
        let mut folds = 1usize;
        let mut candidate_score: Option<MarginScore> = None;
        // `None` = WITHDRAWN (the disjoint-enclosure alarm): see
        // [`resolve_publication`].
        let mut published: Option<(Vec<f32>, Vec<f32>)> =
            Some((first.lower.clone(), first.upper.clone()));
        let mut alarm = false;
        #[cfg(test)]
        let mut record_candidate: (
            Option<Vec<f32>>,
            Option<Vec<f32>>,
            Option<crate::bounds::GraphAlphaState>,
            Option<HashMap<String, Array4<f32>>>,
            Option<HashMap<String, Array2<f32>>>,
        ) = (None, None, None, None, None);

        match plan {
            Err(reason) => report.refused = Some(reason),
            Ok(plan) => {
                let node_bounds = reference_bounds.current();
                let ingested = ingest(
                    &plan.lowering,
                    &plan.hint,
                    &runtime,
                    ctx.input,
                    node_bounds,
                    matmul.as_ref(),
                    &frozen,
                );
                report.counts = ingested.counts;
                let snapshot = runtime.snapshot_graph();
                let mut wrote_all = true;
                for &(relu_idx, neuron, slope) in &ingested.relu {
                    let written =
                        runtime
                            .relu_alpha_entry(relu_idx, neuron)
                            .and_then(|(_, upper)| {
                                runtime.set_relu_alpha_entry(relu_idx, neuron, slope, upper)
                            });
                    if written.is_err() {
                        wrote_all = false;
                        break;
                    }
                }
                if wrote_all {
                    let certify_started = Instant::now();
                    let bilinear = ingested
                        .bilinear
                        .as_ref()
                        .or_else(|| matmul.as_ref().map(|state| &state.alphas));
                    let mul = ingested.mul.as_ref().or(frozen.mul_binary.as_ref());
                    let certified = self.margin_alpha_fold(
                        ctx,
                        node_bounds,
                        &runtime,
                        bilinear,
                        mul,
                        &rows.seed,
                        true,
                    );
                    report.certify_s = certify_started.elapsed().as_secs_f64();
                    match certified {
                        Ok(fold) => {
                            folds += 1;
                            let score = score_margins(&fold.lower, &rows.lower_thresholds);
                            candidate_score = Some(score);
                            kept = keep_candidate(score, incumbent);
                            published = resolve_publication(
                                &first.lower,
                                &first.upper,
                                Some((fold.lower.as_slice(), fold.upper.as_slice())),
                            );
                            if published.is_none() {
                                // Two certified enclosures of the same rows
                                // over the same box do not intersect: a TCB
                                // defect. Withdraw, keep nothing, and say so
                                // on EVERY occurrence (not rate limited).
                                alarm = true;
                                kept = false;
                                report.alarm = "DISJOINT_SOUNDNESS_ALARM";
                                eprintln!(
                                    "[bbe-alarm] phase=vit_root DISJOINT_SOUNDNESS_ALARM: the \
                                     incumbent and candidate certified folds of the same {k} \
                                     margin rows over the same box do not intersect; the \
                                     margin publication is WITHDRAWN (the root keeps its own \
                                     bounds). This is a soundness emergency, not a regression."
                                );
                            }
                            #[cfg(test)]
                            {
                                record_candidate = (
                                    Some(fold.lower.clone()),
                                    Some(fold.upper),
                                    Some(runtime.snapshot_graph()),
                                    bilinear.cloned(),
                                    mul.cloned(),
                                );
                            }
                        }
                        Err(error) => {
                            if !error.is_deadline_exceeded() {
                                emit_margin_alpha_error("joint_certify", &error);
                            }
                            report.refused = Some(if error.is_deadline_exceeded() {
                                "certify_deadline".to_string()
                            } else {
                                "certify_error".to_string()
                            });
                        }
                    }
                } else {
                    report.refused = Some("relu_write".to_string());
                }
                if !kept {
                    runtime.restore_graph(&snapshot);
                }
            }
        }

        // A withdrawn publication reports the incumbent's certified score
        // (`alarm=` and `published=0_alarm` say what happened).
        let published_score = published.as_ref().map_or(incumbent, |(lower, _)| {
            score_margins(lower, &rows.lower_thresholds)
        });
        let publication = published.as_ref().and_then(|(lower, upper)| {
            let (lower_c, upper_c) = rows.original_bounds(lower, upper);
            MarginAlphaPublication::new(
                rows.rows.clone(),
                lower_c,
                upper_c,
                ctx.input,
                usize::from(kept),
                0,
                matmul_alpha_nodes,
            )
        });
        let published_token = match (alarm, publication.is_some()) {
            (true, _) => "0_alarm",
            (false, true) => "1",
            (false, false) => "0_malformed",
        };
        crate::network::graph_root_search::record_graph_root_phase(
            report.refused.is_some(),
            candidate_score.is_some(),
            kept,
            alarm,
        );

        #[cfg(test)]
        LAST_JOINT_SEARCH.with(|cell| {
            let (
                candidate_lower,
                candidate_upper,
                candidate_relu,
                candidate_bilinear,
                candidate_mul,
            ) = record_candidate;
            *cell.borrow_mut() = Some(JointSearchRecord {
                refused: report.refused.clone(),
                backend_called,
                kept,
                counts: report.counts,
                selfcheck_compared: report.selfcheck_compared,
                incumbent_lower: first.lower.clone(),
                incumbent_upper: first.upper.clone(),
                candidate_lower,
                candidate_upper,
                published_lower: publication
                    .as_ref()
                    .and(published.as_ref())
                    .map(|(lower, _)| lower.clone()),
                published_upper: publication
                    .as_ref()
                    .and(published.as_ref())
                    .map(|(_, upper)| upper.clone()),
                candidate_relu,
                candidate_bilinear,
                candidate_mul,
                incumbent_bilinear: matmul.as_ref().map(|state| state.alphas.clone()),
                seed: rows.seed.clone(),
                relu_name_to_idx: runtime.relu_name_to_idx().clone(),
            });
        });

        emit_joint_line(
            &report,
            k,
            kept,
            incumbent,
            candidate_score,
            published_score,
        );

        let at_clamp = runtime.graph().clamp_saturation();
        let mut alpha_state = runtime.into_graph_alpha_state();
        alpha_state.clear_spec_slots();
        if let Some(publication) = publication {
            alpha_state.attach_margin_alpha_publication(publication);
        }
        let secs = started.elapsed().as_secs_f64();
        let exit = match (&report.refused, kept, alarm) {
            (_, _, true) => "joint_f32_alarm",
            (Some(_), _, false) => "joint_f32_refused",
            (None, true, false) => "joint_f32_kept",
            (None, false, false) => "joint_f32_incumbent",
        };
        emit_margin_alpha_line(&MarginLaneSummary {
            rows: k,
            spec_slots: 0,
            matmul_alpha_nodes,
            relu_alpha_nodes,
            folds,
            init: incumbent,
            best: published_score,
            committed: usize::from(kept),
            at_clamp,
            exit,
            published: published_token,
            // The publication IS certified folds; there is no separate
            // recompute that could diverge.
            recompute_identical: true,
            point_failures: 0,
            frozen_token: frozen.token(),
            frozen_mul_binary_nodes: frozen.mul_binary_nodes,
            frozen_reciprocal_nodes: frozen.reciprocal_nodes,
            secs,
        });
        emit_alpha_ascent_exit_line(
            config,
            folds,
            "margin_lane_joint",
            f64::NAN,
            f32::NAN,
            secs,
            at_clamp,
        );

        Ok(DagAlphaCollectionArtifact {
            output_bounds,
            alpha_state,
            reference_bounds: reference_bounds.into_current(),
            reference_bounds_source: node_bounds_source,
            completed_iterations: folds,
            optimizer_updates_completed: usize::from(kept),
        })
    }
}
