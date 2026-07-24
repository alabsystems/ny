// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Batched CROWN backward pass for graph-based branch-and-bound.
//!
//! Processes multiple BaB domains simultaneously through a single backward traversal
//! of the graph, amortizing topological sort and layer dispatch overhead. Supports two
//! modes via [`BatchedBackwardMode`]:
//!
//! - **Standard**: seeds all domains at the output layer, no intermediate capture.
//! - **WithLaCapture**: warm-starts from cached linear coefficients (lA) at branch
//!   points and optionally captures intermediate lA for child domain reuse.
//!
//! The core entry point is `propagate_crown_batched_backward_core`, called by the
//! public-facing `propagate_crown_batched_with_context` methods on [`BetaCrownVerifier`].
//!
//! Reference: alpha-beta-CROWN `auto_LiRPA/bound_general.py` (batched backward pass).

use std::borrow::Borrow;
use std::collections::HashMap;
use std::mem::size_of;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use ndarray::{Array1, Array2};
use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;

use crate::batched_domain::CachedLinearBounds;
use crate::beta_crown::branching::GraphSplitHistory;
use crate::beta_crown::engine::graph::{DomainCrownResult, DomainSpecCrownResult};
use crate::beta_crown::state::{GraphBetaState, GraphDomainAlphaState};
use crate::bounds::{
    FacetBank, FacetBankSearchConfig, LowerAffineCertificate, FACET_BANK_DEFAULT_DYADIC_BITS,
    FACET_BANK_MAX_PLANES,
};
#[cfg(test)]
use crate::layers::common::BoundPropagation;
use crate::network::{CrownDispatchPlan, ResnetSegmentSkeleton};
use crate::{GraphNetwork, Layer, LinearBounds, NETWORK_INPUT};

use super::super::super::BetaCrownVerifier;

mod adapters;
mod backward_core;
mod backward_stack;
mod batched_bwd;
mod context;
mod indexed_pending;
pub(in crate::beta_crown::engine::graph) mod interm_refine;
mod spec_adapters;
/// Re-export of the spec-gate test mutex so gate-forcing parity tests OUTSIDE
/// this module (e.g. `input_split::f64_tail::tests`, alpha-tail) serialize
/// with the spec-gate tests here (`tests_soundness.rs`).
#[cfg(test)]
pub(crate) use spec_adapters::SPEC_GATE_TEST_LOCK;
pub(crate) mod wide_alpha_true;
pub use context::{
    BatchedBackwardContext, BatchedBackwardResult, BatchedSpecBackwardResult, BatchedStageTiming,
    DenseSpecStageTiming,
};
use indexed_pending::IndexedPendingLinearBounds;

// WARN-level, one-shot telemetry for explicitly enabled CUDA-wide canaries.
// Default production runs never reach these reporters because the global-wide
// backend is returned only when NY_CUDA_WIDE/NY_HYDRA_CROWN requests it.
static CUDA_WIDE_DISPATCH_ATTEMPT_REPORTED: AtomicBool = AtomicBool::new(false);
static CUDA_WIDE_DISPATCH_SUCCESS_REPORTED: AtomicBool = AtomicBool::new(false);
static CUDA_WIDE_DISPATCH_FALLBACK_REPORTED: AtomicBool = AtomicBool::new(false);

fn report_cuda_wide_dispatch_attempt_once(
    lane: &'static str,
    n_domains: usize,
    num_specs: usize,
    output_dim: usize,
) {
    if !CUDA_WIDE_DISPATCH_ATTEMPT_REPORTED.swap(true, Ordering::SeqCst) {
        tracing::warn!(
            lane,
            n_domains,
            num_specs,
            output_dim,
            "CUDA wide CROWN eligible dispatch attempted"
        );
    }
}

fn report_cuda_wide_dispatch_success_once(lane: &'static str) {
    if !CUDA_WIDE_DISPATCH_SUCCESS_REPORTED.swap(true, Ordering::SeqCst) {
        tracing::warn!(lane, "CUDA wide CROWN proof-forest dispatch succeeded");
    }
}

fn report_cuda_wide_dispatch_fallback_once(lane: &'static str, reason: &'static str) {
    if !CUDA_WIDE_DISPATCH_FALLBACK_REPORTED.swap(true, Ordering::SeqCst) {
        tracing::warn!(
            lane,
            reason,
            "CUDA wide CROWN dispatch fell back to the existing local/CPU path"
        );
    }
}

pub(crate) mod gather_score;
#[cfg(test)]
mod test_support;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_nan_safety;
#[cfg(test)]
mod tests_skeleton;
#[cfg(test)]
mod tests_soundness;

/// Per-domain β-optimization request for the GPU resnet fast-path
/// (#w4-split-tightening).
///
/// When present (the multi-objective GPU single-pass lane), β-eligible domains
/// run `iterations` projected-gradient-ascent steps of the CPU analytic β rule
/// at GPU speed instead of a single inherited-β pass:
/// per iteration, one sound β-folded GPU backward returns the bounds AND the
/// A-values at the domain's split neurons; the critical (worst unverified) spec
/// row's gradient `∂lb/∂β_k = −sign_k·A_lower[crit, k]` drives the same Adam
/// step as `optimize_graph_beta_analytical_multi_objective_with_cache`.
///
/// Row indexing matches the spec matrix rows the caller seeds (the pruned
/// union rows on the multi-objective lane).
pub(in crate::beta_crown::engine::graph) struct GpuBetaOptSpec<'a> {
    /// Per-spec-row verification thresholds.
    pub thresholds: &'a [f32],
    /// Per-domain, per-spec-row verified mask (verified rows are skipped when
    /// selecting the critical row — the per-child verified latch).
    pub row_verified: &'a [Vec<bool>],
    /// Per-domain eligibility (β entries present, depth cap, config on).
    pub eligible: &'a [bool],
    /// Per-domain BaB depths (#hard-six tail-iters): lets the wide ascent
    /// detect a pinned-tail batch (few, deep domains) and scale its iteration
    /// budget there only (`NY_MO_GPU_BETA_ITERS_TAIL`). Empty = unknown
    /// (tail scaling never fires).
    pub depths: &'a [usize],
}

/// Controls the batched backward pass behavior: whether to perform lA warm-start
/// seeding and/or capture intermediate lA matrices during traversal.
///
/// This enum unifies `propagate_crown_batched_backward_internal` (standard) and
/// `propagate_crown_batched_backward_internal_with_la` (lA capture) into a single
/// traversal parameterized by mode, eliminating ~150 LOC of duplicated setup,
/// traversal loop, and concretization logic.
///
/// Part of #1813 (wave 2 batched backward kernel dedup).
pub(super) enum BatchedBackwardMode<'a> {
    /// Standard backward pass: seed all domains at output, no intermediate capture.
    Standard,
    /// lA-aware backward pass: attempt warm-start seeding at branch points using
    /// cached lA from parent domains, and optionally capture intermediate lA for
    /// child domains.
    ///
    /// # Reference
    /// - alpha-beta-CROWN: `backward_bound.py:203-210` (initial_As warm-start)
    /// - Design: `designs/2026-02-07-gpu-bab-la-reuse-closure.md` (Dir 2b)
    /// - Issue: #1564, #1669
    WithLaCapture {
        histories: &'a [&'a GraphSplitHistory],
        cached_la: &'a [Option<&'a CachedLinearBounds>],
        capture_intermediate: bool,
    },
}

/// #refold-guard: pick the guard indices for one wide batch — domain 0 (a
/// deterministic anchor: any whole-batch layout bug shows up on it) and the
/// domain with the LARGEST minimum lower bound (the most verified-looking row
/// — the one a cross-domain row misassignment could wrongly prune), dedup'd.
/// Pure selection over the RAW batched results; NaN rows never win the argmax
/// (`f32::min` propagates the finite operand, and a fully-NaN row scores
/// -inf), which is fine — non-finite rows are rejected downstream anyway.
fn refold_guard_indices(results: &[ny_core::GpuCrownResult]) -> Vec<usize> {
    let mut indices = vec![0usize];
    let mut best: Option<(usize, f32)> = None;
    for (i, r) in results.iter().enumerate() {
        let row_min = r
            .lower_bounds
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .fold(f32::INFINITY, f32::min);
        let score = if row_min.is_finite() {
            row_min
        } else {
            f32::NEG_INFINITY
        };
        if best.is_none_or(|(_, s)| score > s) {
            best = Some((i, score));
        }
    }
    if let Some((i, _)) = best {
        if i != 0 {
            indices.push(i);
        }
    }
    indices
}

/// #refold-guard: row agreement between one domain's WIDE result and its
/// SERIAL re-fold, under the SAME contract the kernel-level differential
/// oracles prove (crown_backward_sound_resident.rs test module): the wide
/// pass reorders f32 GEMM accumulation, so wide↔serial is NOT bitwise — the
/// proven invariant is two-sided relative closeness
/// `|a−b| ≤ 1e-3·(1 + max(|a|,|b|))` per spec row, both bounds. A
/// cross-domain row misassignment (the class-C hole: HOLE-3/4 stacking bugs,
/// wg-limit silent-overtight driver UB) deviates FAR beyond that tolerance by
/// construction (domains carry distinct relaxations/boxes), while genuine
/// reorder noise stays inside it. Internal-stacker fallback results (the
/// batched entry's own serial loop) are bitwise-equal and pass trivially.
/// Non-finite on either side fails the guard (fail-closed).
fn refold_rows_match(wide: &ny_core::GpuCrownResult, serial: &ny_core::GpuCrownResult) -> bool {
    let close = |a: f32, b: f32| -> bool {
        a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-3 * (1.0 + a.abs().max(b.abs()))
    };
    wide.lower_bounds.len() == serial.lower_bounds.len()
        && wide.upper_bounds.len() == serial.upper_bounds.len()
        && wide
            .lower_bounds
            .iter()
            .zip(serial.lower_bounds.iter())
            .all(|(&a, &b)| close(a, b))
        && wide
            .upper_bounds
            .iter()
            .zip(serial.upper_bounds.iter())
            .all(|(&a, &b)| close(a, b))
}

fn build_nodes_by_idx<'a>(
    graph: &'a GraphNetwork,
    plan: &CrownDispatchPlan,
) -> Result<Vec<&'a crate::GraphNode>> {
    plan.exec_order
        .iter()
        .map(|&idx| {
            graph.nodes.get(plan.name_of(idx)).ok_or_else(|| {
                NyError::InvalidSpec(format!("Node not found: {}", plan.name_of(idx)))
            })
        })
        .collect()
}

/// #clip-interm-resnet (dark, `NY_CLIP_INTERM_RESNET=1`): FINITE input-relative
/// linear bounds for one node via a plain SOUND CROWN backward seeded with the
/// identity at that node.
///
/// The forward accumulation in `clip_alpha::compute_forward_linear_bounds` falls
/// back to `LinearBounds::conservative` (±inf) for conv / residual-Add / flatten
/// nodes it cannot model going forward, so on a deep resnet every split neuron's
/// forward bound is ±inf and `build_split_constraints` skips it — the clip
/// no-ops. This routine instead reuses NY's certified node-by-node backward
/// (`backward_core::dispatch_node_backward`, the SAME primitive the verdict path
/// concretizes) to produce a finite input-relative enclosure
/// `z(x) ∈ [lA·x + lb, uA·x + ub]` for the seed node's neurons.
///
/// SOUNDNESS: this is a plain α-CROWN backward — default ReLU slopes, `beta=None`
/// (no split duals) — over the frozen child bounds cache, so the result is a valid
/// UNCONSTRAINED enclosure of the seed node over the input box (exactly what
/// `build_split_constraints` turns into a NECESSARY condition). The affine form is
/// the identical class of `LinearBounds` the forward source already feeds into the
/// same pipeline; no new bound math. Any structural miss (unknown node, plan/build
/// failure, per-node backward error) returns `None`, so the caller keeps the
/// forward/no-op path — sound (the clip only ever fails to tighten, never loosens).
///
/// Row `j` of the returned bounds corresponds to dense flattened neuron `j` of
/// `seed_node`; for the cifar100 lane `seed_node = Gemm_56` is dense (512-d) so
/// the row index equals the split history's `neuron_idx` directly.
#[allow(clippy::too_many_arguments)]
pub(in crate::beta_crown::engine::graph) fn backward_input_relative_bounds_at_node(
    graph: &GraphNetwork,
    seed_node: &str,
    bounds_cache: &HashMap<String, Arc<BoundedTensor>>,
    constrained_input: &BoundedTensor,
    engine: &dyn GemmEngine,
    // Cooperative wall-clock bound for one targeted backward. Checked before
    // allocation and between graph nodes, and threaded into layer dispatch so
    // deadline-aware GPU/conv kernels can refuse. Any expiry returns `None`;
    // callers retain their original sound box (fail closed).
    deadline: Option<std::time::Instant>,
    // #root-crown-interm-probe: optional per-ReLU α slopes to fold into the
    // backward (default `None` = heuristic default CROWN slopes = byte-identical
    // to the historical clip callers). Any α∈[0,1] is a valid ReLU lower slope so
    // the enclosure stays sound; the probe uses it to measure the optimized-α
    // CROWN width vs the heuristic-α width.
    alpha_state: Option<&GraphDomainAlphaState>,
) -> Option<LinearBounds> {
    if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
        return None;
    }
    let plan = CrownDispatchPlan::build(graph).ok()?;
    let seed_idx = plan.index_of(seed_node)?;
    let seed_dim = bounds_cache.get(seed_node)?.len();
    if seed_dim == 0 {
        return None;
    }
    let nodes_by_idx = build_nodes_by_idx(graph, &plan).ok()?;

    // Seed the identity at `seed_node` (obj = I · seed_node), then fold backward
    // through every node below it to `NETWORK_INPUT` (obj = A · input + b).
    let mut pending = IndexedPendingLinearBounds::new(&plan, 1);
    pending
        .seed_idx(seed_idx, 0, LinearBounds::identity(seed_dim))
        .ok()?;

    let net_in_dim = constrained_input.len();
    // #lsnc-shared-fwd: dispatch_node_backward borrows a slice of cache refs.
    let bounds_caches: [&HashMap<String, Arc<BoundedTensor>>; 1] = [bounds_cache];
    let constrained_inputs = std::slice::from_ref(constrained_input);
    // Plain enclosure: no β split duals; ReLU slopes = caller-supplied α (default
    // `None` = heuristic default CROWN slopes).
    let beta_states: [Option<&GraphBetaState>; 1] = [None];
    let alpha_states: [Option<&GraphDomainAlphaState>; 1] = [alpha_state];

    for &idx in &plan.reverse_order {
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return None;
        }
        let node_lbs = match pending.take_idx(idx) {
            Some(lbs) => lbs,
            None => continue,
        };
        if !node_lbs.iter().any(|lb| lb.is_some()) {
            continue;
        }
        backward_core::dispatch_node_backward(
            plan.name_of(idx),
            nodes_by_idx[idx],
            node_lbs,
            constrained_inputs,
            &bounds_caches,
            &beta_states,
            &alpha_states,
            &mut pending,
            1,
            net_in_dim,
            engine,
            deadline,
            None, // no shared MulBinary alphas
            false,
        )
        .ok()?;
    }

    // Do not publish a candidate that completed after its scheduling envelope.
    // The caller keeps the pre-pass sound reference box unchanged.
    if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
        return None;
    }
    let mut lb = pending.take_network_input()?.into_iter().next().flatten()?;

    // SOUNDNESS (#clip-interm-resnet, #vnncomp-aw-soundness): the raw backward
    // `LinearBounds` at NETWORK_INPUT still carries NY's certified per-coefficient
    // error (`lower_a_err`/`upper_a_err`). The clip consumer (`override_node` →
    // `into_parts`) DROPS that error, so we MUST discharge it here first, or the
    // affine "enclosure" it feeds the clip is not a guaranteed enclosure (a too-tight
    // intermediate bound → false UNSAT). `fold_coeff_err_over_box_eager` folds each
    // row's error OUTWARD into the bias over the child input box (`next_down`/`next_up`)
    // and zeros the folded error, so `lA·x + lbias ≤ z(x) ≤ uA·x + ubias` holds for all
    // x in the box. Any row whose penalty is NON-FINITE keeps its error — using such a
    // row after `into_parts` would be unsound, so we REFUSE the whole node's bounds
    // (clip then keeps the inherited frozen bound for these splits: sound, no tightening).
    lb.fold_coeff_err_over_box_eager(constrained_input);
    if lb.has_coeff_err() {
        return None;
    }
    Some(lb)
}

/// #batched-bab: the per-domain PREP (alpha bridge, segment extraction, β build, input
/// box, per-ReLU frontier/node abs-max) shared by the serial/rayon lane, the wide bound
/// lane, and the wide β-opt lane. The spec seed is identical across domains (built once).
/// Hoisted to module scope so `gpu_beta_optimize_wide` can consume a `&[ResnetDomainPrep]`.
pub(in crate::beta_crown::engine::graph) struct ResnetDomainPrep {
    pub segments: Vec<ny_core::GpuResnetSegment>,
    pub relu_names: Vec<String>,
    pub frontier_abs: Vec<Vec<f32>>,
    pub node_abs: Vec<Vec<f32>>,
    pub beta_signed: Vec<Vec<f32>>,
    pub in_lo: Vec<f32>,
    pub in_hi: Vec<f32>,
    /// #root-joint-interm-alpha stop-at-M: `Some(M)` iff the extraction stopped
    /// at a frozen-bounded interior node M (truncated discriminator-style stack)
    /// and `in_lo`/`in_hi` is box(M) — the frozen map's f32 endpoints VERBATIM —
    /// not the network input box. `None` for every legacy caller (walk reached
    /// NETWORK_INPUT): byte-identical semantics.
    pub stop_node: Option<String>,
}

// Hydra-CROWN's active-set trajectory bank keeps only the rows that can close a
// child: the two worst currently-unverified margins per domain.  A full
// domains×specs×input coefficient history is unnecessarily large on CIFAR/Tiny;
// fixing the active set at iterate zero gives every retained row multiple
// trajectory planes while keeping PERSISTENT history O(D·input_dim), not
// O(D·S·input_dim). The one-iterate rectangular carrier is charged separately
// to the same byte cap below.
const WIDE_FACET_ROWS_PER_DOMAIN: usize = 2;
const WIDE_FACET_DEFAULT_PLANES: usize = 4;
const WIDE_FACET_DEFAULT_MAX_BYTES: usize = 64 * 1024 * 1024;
const WIDE_FACET_REFINEMENT_ROUNDS: usize = 2;

fn facet_env_switch(name: &str) -> Option<bool> {
    std::env::var(name)
        .ok()
        .and_then(|raw| match raw.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "on" => Some(true),
            "0" | "false" | "off" => Some(false),
            _ => None,
        })
}

fn wide_facet_enabled() -> bool {
    facet_env_switch("NY_FACET_BANK")
        .unwrap_or_else(|| facet_env_switch("NY_HYDRA_CROWN").unwrap_or(false))
}

/// Sparse, memory-bounded bank of sound input-relative rows emitted by the wide
/// α/β trajectory.  `rows[(d,s)]` always refers to specification row `s` of
/// domain `d`; no certificate is ever shared across domains.
struct WideFacetCollector {
    rows: HashMap<(usize, usize), Vec<LowerAffineCertificate>>,
    selected_rows: Vec<Vec<usize>>,
    n_domains: usize,
    specs_per_domain: usize,
    input_dim: usize,
    max_planes: usize,
    rows_per_domain: usize,
    captures: usize,
}

impl WideFacetCollector {
    fn from_env(n_domains: usize, specs_per_domain: usize, input_dim: usize) -> Option<Self> {
        if !wide_facet_enabled() || n_domains == 0 || specs_per_domain == 0 || input_dim == 0 {
            return None;
        }
        let max_planes = std::env::var("NY_FACET_BANK_PLANES")
            .ok()
            .and_then(|raw| raw.trim().parse::<usize>().ok())
            .unwrap_or(WIDE_FACET_DEFAULT_PLANES)
            .clamp(2, FACET_BANK_MAX_PLANES);
        let max_bytes = std::env::var("NY_FACET_BANK_MAX_BYTES")
            .ok()
            .and_then(|raw| raw.trim().parse::<usize>().ok())
            .unwrap_or(WIDE_FACET_DEFAULT_MAX_BYTES);

        // The backend's public carrier is rectangular and transiently exports
        // lower+upper centers/errors for every domain/spec row, even though the
        // persistent bank keeps only lower active rows. Charge both that carrier
        // and the retained lower planes to one cap so WGPU/CUDA cannot surprise
        // the host with a full-frontier allocation larger than the experiment's
        // declared budget. A future sparse device gather can remove this term.
        let total_rows = n_domains.checked_mul(specs_per_domain)?;
        let values_per_full_row = input_dim.checked_mul(4)?.checked_add(4)?;
        let transient_bytes = total_rows
            .checked_mul(values_per_full_row)?
            .checked_mul(size_of::<f32>())?;
        let available_for_bank = max_bytes.checked_sub(transient_bytes)?;

        // One retained lower plane owns coefficient center+error and bias
        // center+error. Reduce the active-row width until the complete bank is
        // guaranteed to fit; a zero-row result disables the optional pass.
        let bytes_per_plane = input_dim
            .checked_mul(2)?
            .checked_add(2)?
            .checked_mul(size_of::<f32>())?;
        let bytes_per_row = bytes_per_plane.checked_mul(max_planes)?;
        let row_budget = available_for_bank.checked_div(bytes_per_row)?;
        let rows_per_domain = WIDE_FACET_ROWS_PER_DOMAIN
            .min(specs_per_domain)
            .min(row_budget / n_domains);
        if rows_per_domain == 0 {
            tracing::debug!(
                n_domains,
                specs_per_domain,
                input_dim,
                max_bytes,
                "Hydra FacetBank disabled by memory cap"
            );
            return None;
        }
        let capacity = n_domains.checked_mul(rows_per_domain)?;
        Some(Self {
            rows: HashMap::with_capacity(capacity),
            selected_rows: vec![Vec::new(); n_domains],
            n_domains,
            specs_per_domain,
            input_dim,
            max_planes,
            rows_per_domain,
            captures: 0,
        })
    }

    fn validate_coeff(&self, coeff: &ny_core::GpuResidentCoeffBatched) -> bool {
        let total_rows = match self.n_domains.checked_mul(self.specs_per_domain) {
            Some(v) => v,
            None => return false,
        };
        let total_coeff = match total_rows.checked_mul(self.input_dim) {
            Some(v) => v,
            None => return false,
        };
        coeff.dim == self.input_dim
            && coeff.num_specs == total_rows
            && coeff.num_specs_per_dom == self.specs_per_domain
            && coeff.lower_a.len() == total_coeff
            && coeff.upper_a.len() == total_coeff
            && coeff.lower_err.len() == total_coeff
            && coeff.upper_err.len() == total_coeff
            && coeff.lower_b.len() == total_rows
            && coeff.upper_b.len() == total_rows
            && coeff.lower_b_err.len() == total_rows
            && coeff.upper_b_err.len() == total_rows
    }

    /// Select the iterate-zero active rows, then retain those exact domain/row
    /// certificates on every later trajectory iterate. Any malformed carrier is
    /// a fail-safe no-op and leaves the scalar best-iterate bounds untouched.
    fn capture(
        &mut self,
        coeff: &ny_core::GpuResidentCoeffBatched,
        results: &[ny_core::GpuCrownResult],
        thresholds: &[f32],
        row_verified: &[Vec<bool>],
    ) -> bool {
        if !self.validate_coeff(coeff)
            || results.len() != self.n_domains
            || thresholds.len() != self.specs_per_domain
        {
            return false;
        }

        for (domain, result) in results.iter().enumerate() {
            if result.lower_bounds.len() != self.specs_per_domain
                || result.upper_bounds.len() != self.specs_per_domain
            {
                return false;
            }
            if self.selected_rows[domain].is_empty() {
                let mut candidates: Vec<(usize, f32)> = (0..self.specs_per_domain)
                    .filter(|&row| {
                        !row_verified
                            .get(domain)
                            .and_then(|mask| mask.get(row))
                            .copied()
                            .unwrap_or(false)
                    })
                    .map(|row| {
                        let margin = result.lower_bounds[row] - thresholds[row];
                        (
                            row,
                            if margin.is_finite() {
                                margin
                            } else {
                                f32::NEG_INFINITY
                            },
                        )
                    })
                    .collect();
                candidates.sort_by(|(ra, ma), (rb, mb)| ma.total_cmp(mb).then(ra.cmp(rb)));
                self.selected_rows[domain].extend(
                    candidates
                        .into_iter()
                        .take(self.rows_per_domain)
                        .map(|(row, _)| row),
                );
            }

            for &row in &self.selected_rows[domain] {
                let global_row = domain * self.specs_per_domain + row;
                let start = global_row * self.input_dim;
                let end = start + self.input_dim;
                let certificate = LowerAffineCertificate::with_errors(
                    coeff.lower_a[start..end].to_vec(),
                    coeff.lower_b[global_row],
                    coeff.lower_err[start..end].to_vec(),
                    coeff.lower_b_err[global_row],
                );
                let Ok(certificate) = certificate else {
                    continue;
                };
                let retained = self.rows.entry((domain, row)).or_default();
                if retained.len() < self.max_planes {
                    retained.push(certificate);
                } else {
                    retained[self.max_planes - 1] = certificate;
                }
                self.captures += 1;
            }
        }
        true
    }
}

/// Bridge a domain's `GraphDomainAlphaState` → `GraphAlphaState` (the type the
/// resnet decomposition consumes). Without this, `alpha=None` = default CROWN
/// slopes = loose. Per-ReLU pre-activation = the ReLU's input node bounds from
/// `bounds_cache`. Hoisted from the `prep_domain` closure (byte-identical) so
/// the #interm-refine truncated backward can fold the SAME per-domain α.
pub(in crate::beta_crown::engine::graph) fn build_alpha_bridge<V: Borrow<BoundedTensor>>(
    graph: &GraphNetwork,
    bounds_cache: &HashMap<String, V>,
    alpha_state: Option<&GraphDomainAlphaState>,
) -> Option<crate::bounds::GraphAlphaState> {
    alpha_state.and_then(|da| {
        if da.is_empty() || std::env::var("NY_NO_ALPHA_BRIDGE").ok().as_deref() == Some("1") {
            return None;
        }
        let mut ga = crate::bounds::GraphAlphaState::new();
        for (name, node) in graph.nodes.iter() {
            if !matches!(node.layer, Layer::ReLU(_)) {
                continue;
            }
            let Some(input_name) = node.inputs.first() else {
                continue;
            };
            let Some(pre_act) = bounds_cache.get(input_name).map(Borrow::borrow) else {
                continue;
            };
            if ga.add_relu_node(name, pre_act, false).is_err() {
                continue;
            }
            let lower = da.build_alpha_array(name, pre_act);
            let upper = da.build_alpha_upper_array(name, pre_act);
            if let Some((l, u)) = ga.relu_alpha_pair_mut(name) {
                if l.len() == lower.len() {
                    *l = lower;
                }
                if u.len() == upper.len() {
                    *u = upper;
                }
            }
        }
        Some(ga)
    })
}

/// #batched-bab: ONE domain's resnet-lane prep (alpha bridge, segment
/// extraction from `start_node` down to the network input, β build, input box).
/// Hoisted from the `prep_domain` closure in `try_gpu_beta_batched_resnet_opt`
/// (byte-identical for `start_node == output_node`) so the #interm-refine
/// backward can prep the SAME machinery on the TRUNCATED stack seeded at the
/// last ReLU's input node.
// #extract-skeleton x #image-node-crown: the branch's extended entry, kept as
// the no-skeleton face of `prep_resnet_domain_with` (Borrow-genericized to
// match main's #cone-delta maps). The root-joint lane calls this with its
// gate-scoped `allow_bn`/`stop_at_bounded`; the skeleton never participates on
// that lane (`prep_resnet_domain_with` declines the fold whenever either flag
// is set).
#[allow(clippy::too_many_arguments)]
pub(in crate::beta_crown::engine::graph) fn prep_resnet_domain_ext<V: Borrow<BoundedTensor>>(
    graph: &GraphNetwork,
    start_node: &str,
    bounds_cache: &HashMap<String, V>,
    constrained_input: &BoundedTensor,
    beta_state: Option<&GraphBetaState>,
    alpha_state: Option<&GraphDomainAlphaState>,
    // #metaroom-chain-wide / #mo-beta-graft: when true, the extraction also
    // accepts pure-chain suffixes ([Chain(..)]) — metaroom's 6cnn conv chains —
    // so their BaB re-bound runs on the sound GPU suffix lane. Callers pass
    // `crate::network::bab_chain_wide_enabled()` (the historical opt-in) OR'd
    // with their own reason (the graft forces it for the ascent-only pass).
    // `false` is byte-identical to the historical resnet-only extraction.
    allow_pure_chain: bool,
    // #cgan-bn-gpu-extract: accept surviving BatchNorm nodes (root-joint lane
    // only; every legacy caller passes false => byte-identical).
    allow_bn: bool,
    // #root-joint-interm-alpha stop-at-M: allow the extraction walk to STOP at
    // the deepest reachable frozen-bounded node M when it cannot continue below
    // M. `false` = byte-identical.
    stop_at_bounded: bool,
) -> Option<ResnetDomainPrep> {
    prep_resnet_domain_with(
        None,
        graph,
        start_node,
        bounds_cache,
        constrained_input,
        beta_state,
        alpha_state,
        allow_pure_chain,
        allow_bn,
        stop_at_bounded,
    )
}

/// #extract-skeleton increment 2: [`prep_resnet_domain`] with an optional
/// prebuilt static skeleton (build-once-pass-down; see [`build_call_skeleton`]).
///
/// When `skeleton` is `Some`, keyed for THIS `(start_node, allow_pure_chain)`
/// pair, and not stale for `graph` ([`ResnetSegmentSkeleton::matches_graph`]),
/// the segment extraction is `fold_for_domain` — oracle-proven bit-identical to
/// `extract_gpu_segments_with_relu_names_ext` whenever both succeed
/// (`resnet_skeleton` tests + the wiring tests in `tests_skeleton`). On ANY
/// refusal — no skeleton, key mismatch, stale skeleton, fold `None` — THIS
/// domain falls back to the legacy full extraction (the fail-closed spine):
/// behavior is identical by construction, never a divergent segment list.
/// The alpha bridge, β build, and input box are shared verbatim by both routes.
#[allow(clippy::too_many_arguments)]
pub(in crate::beta_crown::engine::graph) fn prep_resnet_domain_with<V: Borrow<BoundedTensor>>(
    skeleton: Option<&ResnetSegmentSkeleton>,
    graph: &GraphNetwork,
    start_node: &str,
    bounds_cache: &HashMap<String, V>,
    constrained_input: &BoundedTensor,
    beta_state: Option<&GraphBetaState>,
    alpha_state: Option<&GraphDomainAlphaState>,
    allow_pure_chain: bool,
    // #cgan-bn-gpu-extract: accept surviving BatchNorm nodes (root-joint lane
    // only; every legacy caller passes false => byte-identical).
    allow_bn: bool,
    // #root-joint-interm-alpha stop-at-M: allow the extraction walk to STOP at
    // the deepest reachable frozen-bounded node M when it cannot continue below
    // M. `false` = byte-identical.
    stop_at_bounded: bool,
) -> Option<ResnetDomainPrep> {
    // TIGHTNESS: fold the domain's (warmup-inherited) OPTIMIZED alpha into the
    // decomposition by bridging GraphDomainAlphaState → GraphAlphaState (the type
    // the decomposition consumes). Without this, alpha=None = default CROWN slopes
    // = loose. Per-ReLU pre-activation = the ReLU's input node bounds.
    let alpha_bridge = build_alpha_bridge(graph, bounds_cache, alpha_state);
    // #extract-skeleton: fold the static skeleton for this domain when present
    // and valid. `cache_key` pins the skeleton to this (start_node,
    // allow_pure_chain) — a mismatched key could otherwise fold a DIFFERENT
    // suffix (or accept a pure chain the legacy gate would refuse);
    // `matches_graph` is the stale-skeleton guard (conv geometry / broadcasts
    // were baked from THIS graph's bounds shapes). Any `None` → legacy below.
    //
    // #extract-skeleton x #image-node-crown reconciliation: the skeleton fold
    // DECLINES OUTRIGHT whenever the branch's evolved extraction semantics
    // could bite — `allow_bn` (BN-as-1x1-conv) and `stop_at_bounded` (frozen
    // stop at interior node M) change the walk's structure/refusal set, which
    // the recorded skeleton cannot represent. Those lanes go through the legacy
    // extraction below (fail-closed: cost-only, never a divergent segment
    // list). With BOTH flags false the legacy call is byte-identical to the
    // historical 7-arg call and `stop_node` is structurally `None`
    // (`stop_eligible` short-circuits on `!frozen_stop` and the loop can only
    // `break None` at NETWORK_INPUT — see resnet_decompose.rs), so the
    // skeleton fold's 4-tuple and the legacy 5-tuple describe the SAME
    // extraction over the SAME network-input concretization box.
    let use_skeleton = !allow_bn && !stop_at_bounded;
    let folded = skeleton
        .filter(|_| use_skeleton)
        .filter(|s| s.cache_key() == (start_node, allow_pure_chain) && s.matches_graph(graph))
        .and_then(|s| {
            s.fold_for_domain(
                graph,
                constrained_input,
                bounds_cache,
                bounds_cache,
                alpha_bridge.as_ref(),
            )
        });
    let (segments, relu_names, frontier_abs, node_abs, stop_node) = match folded {
        // Skeleton path: only reachable with both flags false, where the legacy
        // walk is proven to reach NETWORK_INPUT or refuse — stop_node ≡ None.
        Some((segments, relu_names, frontier_abs, node_abs)) => {
            (segments, relu_names, frontier_abs, node_abs, None)
        }
        None => {
            let (segments, relu_names, frontier_abs, node_abs, stop_node) =
                crate::network::extract_gpu_segments_with_relu_names_ext(
                    graph,
                    constrained_input,
                    start_node,
                    bounds_cache,
                    bounds_cache,
                    alpha_bridge.as_ref(),
                    allow_pure_chain,
                    allow_bn,
                    stop_at_bounded,
                )?;
            debug_assert!(
                stop_at_bounded || stop_node.is_none(),
                "frozen_stop=false can never stop"
            );
            (segments, relu_names, frontier_abs, node_abs, stop_node)
        }
    };
    let mut beta_signed: Vec<Vec<f32>> = Vec::with_capacity(relu_names.len());
    for name in &relu_names {
        let nn = bounds_cache.get(name)?.borrow().lower().len();
        let mut bs = vec![0.0f32; nn];
        if let Some(beta) = beta_state {
            for e in beta.entries_for_node(name) {
                if e.split_point().abs() < 1e-6 {
                    let idx = e.neuron_idx();
                    if idx < nn {
                        bs[idx] = e.signed_value();
                    }
                }
            }
        }
        beta_signed.push(bs);
    }
    // Concretization box for the deepest fold frontier: the network input box
    // (legacy), or box(M) when the walk stopped at frozen-bounded M. box(M) is
    // the frozen map's f32 endpoints VERBATIM — already a sound outward f32
    // enclosure of reachable(M) (no rounding on this path; BoundedTensor is
    // ArrayD<f32>), and reachable(M) ⊆ box(M) makes any CROWN fold of
    // graph[M→L] concretized against box(M) a valid enclosure. FAIL-CLOSED:
    // missing entry, non-finite endpoint, crossed interval, or M ==
    // NETWORK_INPUT ⇒ refuse the whole prep (caller keeps its reference bound).
    let (in_lo, in_hi): (Vec<f32>, Vec<f32>) = match stop_node.as_deref() {
        None => (
            constrained_input.lower().iter().copied().collect(),
            constrained_input.upper().iter().copied().collect(),
        ),
        Some(m) => {
            if m == NETWORK_INPUT {
                return None; // contract violation: stop node must be interior
            }
            let bt = bounds_cache.get(m)?.borrow();
            let lo: Vec<f32> = bt.lower().iter().copied().collect();
            let hi: Vec<f32> = bt.upper().iter().copied().collect();
            if lo.is_empty()
                || lo.len() != hi.len()
                || lo
                    .iter()
                    .zip(&hi)
                    .any(|(&l, &u)| !l.is_finite() || !u.is_finite() || l > u)
            {
                return None;
            }
            (lo, hi)
        }
    };
    Some(ResnetDomainPrep {
        segments,
        relu_names,
        frontier_abs,
        node_abs,
        beta_signed,
        in_lo,
        in_hi,
        stop_node,
    })
}

/// Legacy entry — never stops early; byte-identical for every existing caller:
/// `stop_node` is always `None` and `in_lo`/`in_hi` is always the constrained
/// input box. Post-merge (#extract-skeleton x #image-node-crown) the production
/// lanes call `prep_resnet_domain_with`/`prep_resnet_domain_ext` directly; this
/// 7-arg face is kept as the tests' legacy-reference oracle.
#[cfg_attr(not(test), allow(dead_code))]
pub(in crate::beta_crown::engine::graph) fn prep_resnet_domain<V: Borrow<BoundedTensor>>(
    graph: &GraphNetwork,
    start_node: &str,
    bounds_cache: &HashMap<String, V>,
    constrained_input: &BoundedTensor,
    beta_state: Option<&GraphBetaState>,
    alpha_state: Option<&GraphDomainAlphaState>,
    allow_pure_chain: bool,
) -> Option<ResnetDomainPrep> {
    prep_resnet_domain_ext(
        graph,
        start_node,
        bounds_cache,
        constrained_input,
        beta_state,
        alpha_state,
        allow_pure_chain,
        false,
        false,
    )
}

/// #extract-skeleton increment 2: build the per-call [`ResnetSegmentSkeleton`]
/// ONCE from an exemplar domain's (bounds, alpha) — the build-once-pass-down
/// pattern of `F64WeightCache` (`network/graph_ibp_f64_batch.rs`), per-call
/// local; the cross-batch verifier-level cache is increment 3. Callers pass the
/// result to every per-domain [`prep_resnet_domain_with`] in the same call.
///
/// Returns `None` — and every domain then preps through the legacy extraction,
/// byte-identically — under the `NY_EXTRACT_SKELETON=0` kill-switch (wholesale
/// revert) and on any build refusal (un-extractable exemplar, unclassifiable
/// layer, recorder inconsistency): fail closed, cost-only.
pub(in crate::beta_crown::engine::graph) fn build_call_skeleton<V: Borrow<BoundedTensor>>(
    graph: &GraphNetwork,
    start_node: &str,
    bounds_cache: &HashMap<String, V>,
    constrained_input: &BoundedTensor,
    alpha_state: Option<&GraphDomainAlphaState>,
    allow_pure_chain: bool,
) -> Option<ResnetSegmentSkeleton> {
    if !crate::network::extract_skeleton_enabled() {
        return None;
    }
    // The exemplar's bridge only steers the build-time walk (whose per-domain
    // relaxations are NaN-poisoned out of the skeleton anyway); each domain's
    // fold re-bridges its OWN alpha inside `prep_resnet_domain_with`.
    let alpha_bridge = build_alpha_bridge(graph, bounds_cache, alpha_state);
    crate::network::build_resnet_segment_skeleton(
        graph,
        constrained_input,
        start_node,
        bounds_cache,
        bounds_cache,
        alpha_bridge.as_ref(),
        allow_pure_chain,
    )
}

/// #mo-beta-graft: elementwise-tightest composition of two SOUND spec-row
/// enclosures of the SAME quantity over the SAME subdomain (the dense-spec
/// batched bound and the wide segment-lane ascended bound, both over the same
/// spec matrix, split-clamped caches, and β entries).
///
/// Returns `Some((composed, rows_tightened, rows_total, max_lower_gain))`, or
/// `None` when the lengths mismatch or construction fails (the caller keeps
/// the dense bound unchanged — sound). Per row: non-finite wide values keep
/// the dense row; NaN dense values are kept verbatim so the existing NaN
/// rejection (#2982) still fires; an inverted intersection (possible only from
/// f32 slop between two sound enclosures) keeps the dense row; otherwise
/// `l = max(l_dense, l_wide)`, `u = min(u_dense, u_wide)`.
pub(in crate::beta_crown::engine::graph) fn graft_compose_tightest(
    dense: &BoundedTensor,
    wide: &BoundedTensor,
) -> Option<(BoundedTensor, usize, usize, f32)> {
    let d = dense.flatten();
    let w = wide.flatten();
    let n = d.len();
    if n == 0 || w.len() != n {
        return None;
    }
    let mut lo = Vec::with_capacity(n);
    let mut hi = Vec::with_capacity(n);
    let mut tightened = 0usize;
    let mut max_gain = 0.0f32;
    for i in 0..n {
        let dl = d.lower()[[i]];
        let du = d.upper()[[i]];
        let wl = w.lower()[[i]];
        let wu = w.upper()[[i]];
        let (l, u) = if !wl.is_finite() || !wu.is_finite() || dl.is_nan() || du.is_nan() {
            (dl, du)
        } else {
            let l = dl.max(wl);
            let u = du.min(wu);
            if l <= u {
                (l, u)
            } else {
                (dl, du)
            }
        };
        if l > dl || u < du {
            tightened += 1;
            max_gain = max_gain.max(l - dl);
        }
        lo.push(l);
        hi.push(u);
    }
    let lower = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[n]), lo).ok()?;
    let upper = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[n]), hi).ok()?;
    let composed = BoundedTensor::new(lower, upper).ok()?;
    Some((composed, tightened, n, max_gain))
}

impl BetaCrownVerifier {
    /// #mo-beta-graft master gate: preset/config `mo_beta_graft`
    /// (`bab.beta_graft`), env `NY_MO_BETA_GRAFT` overrides in both directions
    /// ("1" forces on, "0" forces off — the A/B knob).
    fn mo_beta_graft_active(&self) -> bool {
        match std::env::var("NY_MO_BETA_GRAFT").ok().as_deref() {
            Some("1") => true,
            Some("0") => false,
            _ => self.config.mo_beta_graft,
        }
    }

    /// Unified batched backward pass implementation.
    ///
    /// This is the single backward pass core that handles both the standard path
    /// (seed at output, no capture) and the lA-aware path (warm-start at branch
    /// points, optional intermediate capture). The `mode` parameter controls which
    /// behavior is active.
    ///
    /// Both `propagate_crown_batched_with_context` and
    /// `propagate_crown_batched_with_context_capture_la` delegate to this function.
    ///
    /// # Arguments
    /// * `graph` - The network graph
    /// * `n_domains` - Number of domains in the batch
    /// * `plan` - Precompiled dispatch plan for indexed reverse traversal
    /// * `bounds_caches` - Pre-computed IBP bounds per domain
    /// * `constrained_inputs` - Constrained input bounds per domain
    /// * `beta_states` - β parameters per domain for Lagrangian optimization
    /// * `alpha_states` - Per-domain α parameters for ReLU slope optimization
    /// * `objective` - Objective coefficients (same for all domains)
    /// * `engine` - GPU compute engine
    /// * `mode` - Controls warm-start seeding and intermediate lA capture
    ///
    /// # Reference
    /// - alpha-beta-CROWN: `auto_LiRPA/bound_general.py` (backward pass)
    /// - Design: designs/2026-02-09-code-structure-wave2-graph-engine-split.md (Step 4.3)
    #[allow(clippy::too_many_arguments)]
    ///
    /// GPU beta-capable resnet per-domain fast-path for the batched DomainList backward
    /// (#unsat-keystone step 4). The batched backward is node-by-node (dense → slow/OOM on
    /// conv-resnets); this computes EACH domain's bound whole-suffix via the sound GPU
    /// resnet backward with the β-CROWN split dual folded in. `seed_rows` is the
    /// num_specs×output_dim seed (row-major); alpha=None (default ReLU slopes from the
    /// CONSTRAINED bounds reflect the splits, β enforces them). Returns `Some(results)`
    /// (one DomainCrownResult per domain) when applicable, else `None` → the caller runs
    /// the proven node-by-node batched backward. Default ON (opt out
    /// `NY_RESNET_BETA_GPU=0`), sound (β≥0 valid dual + sound GPU enclosure); CPU
    /// fallback preserves the 0-wrong moat.
    #[allow(clippy::too_many_arguments)]
    fn try_gpu_beta_batched_resnet(
        &self,
        graph: &GraphNetwork,
        output_node: &str,
        output_dim: usize,
        seed_rows: &[f32],
        num_specs: usize,
        n_domains: usize,
        bounds_caches: &[&HashMap<String, Arc<BoundedTensor>>],
        constrained_inputs: &[BoundedTensor],
        beta_states: &[Option<&GraphBetaState>],
        alpha_states: &[Option<&GraphDomainAlphaState>],
        engine: &dyn GemmEngine,
        probe_tag: &str,
    ) -> Option<Vec<BoundedTensor>> {
        self.try_gpu_beta_batched_resnet_opt(
            graph,
            output_node,
            output_dim,
            seed_rows,
            num_specs,
            n_domains,
            bounds_caches,
            constrained_inputs,
            beta_states,
            alpha_states,
            engine,
            probe_tag,
            None,
            false, // scalar lane: no graft chain-forcing (historical behavior)
        )
        .map(|(bounds, _betas, _alphas)| bounds)
    }

    /// β-optimizing form of [`try_gpu_beta_batched_resnet`] (#w4-split-tightening).
    /// With `beta_opt = None` (or an ineligible domain) each domain gets exactly the
    /// legacy single-shot inherited-β GPU pass — byte-identical bounds. Eligible
    /// domains run the per-domain analytic β ascent (see [`GpuBetaOptSpec`]); their
    /// returned bounds are the element-wise TIGHTEST across the sound iterates
    /// (iterate 0 = the inherited-β pass, so never looser than the single-shot
    /// lane), and the second return slot carries the optimized β for child
    /// warm-starting. The third slot carries the wide α ascent's per-domain
    /// best-margin α snapshot (`Some` only under `NY_WIDE_ALPHA_UNSHARED=1`;
    /// all-`None` otherwise — callers keep inherited α, byte-identical).
    /// SOUND: every iterate is a valid Lagrangian-dual enclosure for
    /// its β ≥ 0 over the SAME child domain, so the per-row intersection encloses
    /// the true range; every pass runs the same certified-error + explosion-merge
    /// machinery as the single-shot lane.
    #[allow(clippy::too_many_arguments)]
    fn try_gpu_beta_batched_resnet_opt(
        &self,
        graph: &GraphNetwork,
        output_node: &str,
        output_dim: usize,
        seed_rows: &[f32],
        num_specs: usize,
        n_domains: usize,
        bounds_caches: &[&HashMap<String, Arc<BoundedTensor>>],
        constrained_inputs: &[BoundedTensor],
        beta_states: &[Option<&GraphBetaState>],
        alpha_states: &[Option<&GraphDomainAlphaState>],
        engine: &dyn GemmEngine,
        probe_tag: &str,
        beta_opt: Option<&GpuBetaOptSpec<'_>>,
        // #mo-beta-graft: force pure-chain segment extraction for THIS call even
        // when `NY_BAB_CHAIN_WIDE` is unset — the graft runs this lane only to
        // OPTIMIZE β/α (and to contribute a second sound bound to an
        // elementwise-tightest composition), never to replace the dense bound.
        graft_pure_chain: bool,
    ) -> Option<(
        Vec<BoundedTensor>,
        Vec<Option<GraphBetaState>>,
        Vec<Option<GraphDomainAlphaState>>,
    )> {
        if !crate::network::resnet_beta_gpu_enabled() {
            return None;
        }
        // Do not start a potentially large GPU proof-forest after the verifier's
        // budget has already expired.  Returning `None` preserves the caller's
        // existing deadline/fallback semantics.
        if self.config.alpha_config.past_deadline() {
            return None;
        }
        let local_gpu = engine
            .as_gpu_crown_backward()
            .filter(|g| g.provides_sound_gpu_crown());
        // CUDA's implementation stacks sibling domains into one proof forest,
        // whereas the propagation engine is normally WGPU.  Prefer the global
        // CUDA backend for the wide calls, but retain the local backend for an
        // immediate retry and for the historical serial fallback.  The global
        // factory is lazy, so this is also the first point at which a scored
        // multi-domain CROWN workload can pay CUDA initialization.
        let global_wide_gpu = crate::sound_gpu_gate::global_sound_gpu_crown_for_wide();
        let wide_gpu = global_wide_gpu.or(local_gpu)?;
        let local_wide_fallback = local_gpu.filter(|&local| !std::ptr::eq(local, wide_gpu));
        if !graph
            .nodes
            .values()
            .any(|n| matches!(n.layer, Layer::Conv2d(_)))
        {
            return None;
        }
        if output_dim == 0
            || num_specs == 0
            || num_specs > 512
            || num_specs.saturating_mul(output_dim) > (1 << 24)
            || seed_rows.len() != num_specs * output_dim
        {
            return None;
        }
        // #stream-c-batched-bab increment 1: the per-domain resnet-backward work
        // (alpha bridge, segment extraction, β build, the GPU backward call) is
        // INDEPENDENT across domains and feeds disjoint outputs. A default-OFF gate
        // (`NY_BAB_RESNET_PARALLEL=1`) runs the domains on a rayon fan-out — GPU
        // calls still serialize on the device `gpu_serialize` mutex, so the win is
        // the overlapped CPU prep; value-identical to the serial loop (proven by
        // the differential test). First sound slice of batched-BaB, ahead of the
        // multi-week GPU domain-axis kernel fusion.
        // #batched-bab: `ResnetDomainPrep` is now module-scoped (shared with the wide
        // β-opt lane). The per-domain prep + shared seed are built once here.
        let shared_seed = ny_core::GpuCrownSeed {
            lower_a: seed_rows.to_vec().into(),
            upper_a: seed_rows.to_vec().into(),
            lower_b: vec![0.0f32; num_specs].into(),
            upper_b: vec![0.0f32; num_specs].into(),
            num_specs,
            current_dim: output_dim,
        };
        // #batched-bab: the per-domain prep is `prep_resnet_domain` (module scope,
        // shared with the #interm-refine truncated backward), byte-identical to the
        // historical inline closure.
        let allow_pure_chain = crate::network::bab_chain_wide_enabled() || graft_pure_chain;
        // #extract-skeleton increment 3: the static skeleton now comes from
        // the verifier-level cross-batch cache (`self.skeleton_cache`) — a hit
        // re-validates `matches_graph` and serves the SAME `Arc` to every
        // batch over the verifier's lifetime; a miss (or stale entry) rebuilds
        // exactly the increment-2 per-call skeleton from domain 0's
        // (bounds, alpha) exemplar. Every per-domain prep below — the
        // reference stacker, the wide-β lane, and the serial/rayon fallback
        // (up to 3 prep repeats per domain) — folds it instead of
        // re-extracting. `None` (kill-switch, no domains, build refusal) keeps
        // every prep on the legacy extraction, byte-identically (fail closed);
        // `prep_resnet_domain_with` still re-checks `cache_key` +
        // `matches_graph` on every use.
        let skeleton = (n_domains > 0)
            .then(|| {
                self.skeleton_cache
                    .get_or_build(graph, output_node, allow_pure_chain, || {
                        build_call_skeleton(
                            graph,
                            output_node,
                            bounds_caches[0],
                            &constrained_inputs[0],
                            alpha_states[0],
                            allow_pure_chain,
                        )
                    })
            })
            .flatten();
        let prep_domain = |i: usize| -> Option<ResnetDomainPrep> {
            prep_resnet_domain_with(
                skeleton.as_deref(),
                graph,
                output_node,
                bounds_caches[i],
                &constrained_inputs[i],
                beta_states[i],
                alpha_states[i],
                allow_pure_chain,
                // #image-node-crown flags: legacy BaB lane — never BN, never a
                // frozen stop (byte-identical to the historical 7-arg prep).
                false,
                false,
            )
        };
        // #prep-dedupe (boxlift charter Inc 3 slice): the three lanes below —
        // reference stacker, wide-β, and the serial/rayon fallback — each
        // re-ran `prep_domain` for every domain they visited, so a
        // stacker-refuse → wide-β-refuse → serial cascade paid the extraction
        // + relaxation bake up to 3× per domain (extraction ≈ 15% of the
        // per-domain frame). Memoize per domain: every input the prep reads
        // (bounds cache, constrained input, β/α state, skeleton) is an
        // immutable borrow for the whole call, so the memoized value is
        // byte-identical to a recompute by determinism. Per-domain (not
        // whole-batch) laziness keeps the serial path's early-deadline
        // scheduling identical: domains never visited are never prepped.
        // OnceLock: the serial fallback fans out on rayon under
        // NY_BAB_RESNET_PARALLEL.
        let prep_memo: Vec<std::sync::OnceLock<Option<ResnetDomainPrep>>> =
            (0..n_domains).map(|_| std::sync::OnceLock::new()).collect();
        let prep_get = |i: usize| -> Option<&ResnetDomainPrep> {
            prep_memo[i].get_or_init(|| prep_domain(i)).as_ref()
        };

        // #batched-bab increment 1 (reference stacker): when gated on AND no domain
        // is β-opt-eligible, compute all domains in ONE batched GPU call. Byte-
        // identical to the serial single-shot lane (the GPU differential oracle
        // proves it); any prep None / Err / non-finite falls through to the
        // serial/rayon loop below (the 0-wrong moat).
        if crate::network::resnet_beta_gpu_batched_enabled() {
            let any_beta_opt = beta_opt.is_some_and(|o| {
                (0..n_domains).any(|i| {
                    o.eligible.get(i).copied().unwrap_or(false)
                        && beta_states[i].is_some_and(|b| !b.is_empty())
                        && o.row_verified.get(i).is_some()
                        && o.thresholds.len() == num_specs
                })
            });
            if !any_beta_opt {
                let run_batched =
                    |gpu: &dyn ny_core::GpuCrownBackward| -> Option<Vec<BoundedTensor>> {
                        // #prep-dedupe: memoized per-domain preps (shared with
                        // the wide-β and serial lanes below).
                        let preps = (0..n_domains).map(&prep_get).collect::<Option<Vec<_>>>()?;
                        let refs: Vec<ny_core::GpuResnetBatchedDomainRef> = preps
                            .iter()
                            .map(|p| ny_core::GpuResnetBatchedDomainRef {
                                segments: &p.segments,
                                input_lower: &p.in_lo,
                                input_upper: &p.in_hi,
                                beta_signed: &p.beta_signed,
                                frontier_abs: &p.frontier_abs,
                                node_abs: &p.node_abs,
                            })
                            .collect();
                        let results = gpu
                            .crown_backward_gpu_resnet_sound_beta_batched(&refs, &shared_seed)
                            .ok()?;
                        if results.len() != n_domains {
                            return None;
                        }
                        // #refold-guard (class-C → class-B): runtime spot-check of
                        // the wide↔serial contract the test-time differential
                        // oracles prove. Re-fold the anchor domain + the most
                        // verified-looking domain through the SERIAL sound
                        // backward on the SAME backend; any row outside the
                        // oracle's two-sided 1e-3 relative tolerance — or any
                        // serial refusal — abandons the ENTIRE wide result to
                        // the proven serial/rayon loop below (downgrade-only:
                        // the guard can only prevent a verdict, never create
                        // one). Cost: 2 serial folds per batch (~2/N of the
                        // batch). The wide-β lane gets its own guard in a later
                        // increment.
                        if crate::network::resnet_refold_guard_enabled() && n_domains > 1 {
                            for gi in refold_guard_indices(&results) {
                                let p = preps[gi];
                                let serial = gpu
                                    .crown_backward_gpu_resnet_sound_beta_refold_oracle(
                                        &p.segments,
                                        &shared_seed,
                                        &p.in_lo,
                                        &p.in_hi,
                                        &p.beta_signed,
                                        &p.frontier_abs,
                                        &p.node_abs,
                                    );
                                let ok = serial
                                    .as_ref()
                                    .is_ok_and(|s| refold_rows_match(&results[gi], s));
                                if !ok {
                                    // Always audible: this is a soundness alarm,
                                    // not a diagnostic.
                                    eprintln!(
                                        "[refold-guard] wide batch REJECTED ({probe_tag}): domain {gi}/{n_domains} failed serial re-fold (serial_ok={}); falling back to the serial loop",
                                        serial.is_ok(),
                                    );
                                    return None;
                                }
                            }
                        }
                        let mut out = Vec::with_capacity(n_domains);
                        for r in results {
                            if r.lower_bounds.len() != num_specs
                                || r.upper_bounds.len() != num_specs
                                || r.lower_bounds
                                    .iter()
                                    .chain(r.upper_bounds.iter())
                                    .any(|v| !v.is_finite())
                            {
                                return None;
                            }
                            let lower = ndarray::ArrayD::from_shape_vec(
                                ndarray::IxDyn(&[num_specs]),
                                r.lower_bounds,
                            )
                            .ok()?;
                            let upper = ndarray::ArrayD::from_shape_vec(
                                ndarray::IxDyn(&[num_specs]),
                                r.upper_bounds,
                            )
                            .ok()?;
                            out.push(
                                BoundedTensor::new_repaired(
                                    lower,
                                    upper,
                                    ny_tensor::RepairStrategy::Widen,
                                )
                                .ok()?,
                            );
                        }
                        Some(out)
                    };
                let using_global_wide = global_wide_gpu.is_some();
                if using_global_wide {
                    report_cuda_wide_dispatch_attempt_once(
                        "single-pass",
                        n_domains,
                        num_specs,
                        output_dim,
                    );
                }
                let primary_batched = run_batched(wide_gpu);
                if using_global_wide {
                    if primary_batched.is_some() {
                        report_cuda_wide_dispatch_success_once("single-pass");
                    } else {
                        report_cuda_wide_dispatch_fallback_once(
                            "single-pass",
                            "domain preparation, CUDA execution, or result validation declined",
                        );
                    }
                }
                let batched = primary_batched.or_else(|| {
                    if self.config.alpha_config.past_deadline() {
                        None
                    } else {
                        local_wide_fallback.and_then(run_batched)
                    }
                });
                if let Some(results) = batched {
                    if std::env::var("NY_BETA_GPU_PROBE").ok().as_deref() == Some("1") {
                        eprintln!(
                            "[beta-gpu-batched:{probe_tag}] BATCHED n_domains={n_domains} num_specs={num_specs} od={output_dim}"
                        );
                    }
                    return Some((results, vec![None; n_domains], vec![None; n_domains]));
                }
                // else: fall through to the serial/rayon loop (unchanged).
            }
            // #batched-bab part A: when SOME domain IS β-opt-eligible (cifar100's DEFAULT
            // scored path), route the per-domain β ascent through the WIDE batched grad
            // backward (one wide pass/iter) instead of the serial compute_domain loop —
            // applying the shipped ~4× wide-fold throughput to the β-opt path. Dark-gated
            // (NY_BAB_RESNET_WIDE_BETA=1); any prep None / gpu None → fall through to the
            // serial loop (0-wrong moat; β-opt is non-soundness-critical either way).
            if any_beta_opt && crate::network::resnet_beta_gpu_wide_beta_enabled() {
                if let Some(bo) = beta_opt {
                    // #prep-dedupe: memoized per-domain preps (shared with the
                    // stacker lane above and the serial lane below).
                    if let Some(preps) = (0..n_domains).map(&prep_get).collect::<Option<Vec<_>>>() {
                        // ReLU node → its input node (whose cached bounds are the
                        // PRE-activation bounds the α ascent needs; the ReLU's own
                        // cache entry is post-activation, l >= 0 — never "unstable").
                        let relu_pre_name: HashMap<String, String> = graph
                            .nodes
                            .iter()
                            .filter(|(_, node)| matches!(node.layer, Layer::ReLU(_)))
                            .filter_map(|(name, node)| {
                                node.inputs.first().map(|i| (name.clone(), i.clone()))
                            })
                            .collect();
                        let run_wide = |gpu: &dyn ny_core::GpuCrownBackward| {
                            self.gpu_beta_optimize_wide(
                                gpu,
                                &preps,
                                &shared_seed,
                                bounds_caches,
                                beta_states,
                                alpha_states,
                                &relu_pre_name,
                                bo,
                                num_specs,
                            )
                        };
                        let using_global_wide = global_wide_gpu.is_some();
                        if using_global_wide {
                            report_cuda_wide_dispatch_attempt_once(
                                "wide-beta",
                                n_domains,
                                num_specs,
                                output_dim,
                            );
                        }
                        let primary_wide_result = run_wide(wide_gpu);
                        if using_global_wide {
                            if primary_wide_result.is_some() {
                                report_cuda_wide_dispatch_success_once("wide-beta");
                            } else {
                                report_cuda_wide_dispatch_fallback_once(
                                    "wide-beta",
                                    "domain preparation, CUDA execution, or result validation declined",
                                );
                            }
                        }
                        let wide_result = primary_wide_result.or_else(|| {
                            if self.config.alpha_config.past_deadline() {
                                None
                            } else {
                                local_wide_fallback.and_then(run_wide)
                            }
                        });
                        if let Some((results, betas, alphas)) = wide_result {
                            return Some((results, betas, alphas));
                        }
                    }
                }
                // None → fall through to the serial compute_domain loop below.
            }
        }

        // The process-global CUDA backend is registered for domain-wide calls
        // only.  If that attempt missed and the propagation engine has no local
        // sound GPU backend, return to the historical CPU path instead of leaking
        // the wide-only CUDA preference into ordinary per-domain CROWN calls.
        if self.config.alpha_config.past_deadline() {
            return None;
        }
        let serial_gpu = local_gpu?;

        let compute_domain = |i: usize| -> Option<(BoundedTensor, Option<GraphBetaState>)> {
            if self.config.alpha_config.past_deadline() {
                return None;
            }
            // #prep-dedupe: borrow the memoized prep (legacy prep ⇒ stop_node
            // always None; the memo shares the SAME prep with the batched
            // lanes above — byte-identical to the historical per-lane
            // recompute by input-immutability + determinism).
            let p = prep_get(i)?;
            debug_assert!(p.stop_node.is_none());
            let (segments, relu_names, frontier_abs, node_abs, beta_signed, in_lo, in_hi) = (
                &p.segments,
                &p.relu_names,
                &p.frontier_abs,
                &p.node_abs,
                &p.beta_signed,
                &p.in_lo,
                &p.in_hi,
            );
            // Per-domain β optimization (#w4-split-tightening): eligible domains run
            // the analytic β ascent (element-wise tightest over sound iterates;
            // iterate 0 = the inherited-β pass below, so never looser). Ineligible
            // domains take the legacy single-shot call byte-identically.
            let opt_ctx = beta_opt.filter(|o| {
                o.eligible.get(i).copied().unwrap_or(false)
                    && beta_states[i].is_some_and(|b| !b.is_empty())
                    && o.row_verified.get(i).is_some()
                    && o.thresholds.len() == num_specs
            });
            let (lower_v, upper_v, opt_beta) = if let Some(opt) = opt_ctx {
                let (lo, hi, best_beta) = self.gpu_beta_optimize_domain(
                    serial_gpu,
                    segments,
                    relu_names,
                    frontier_abs,
                    node_abs,
                    &shared_seed,
                    in_lo,
                    in_hi,
                    bounds_caches[i],
                    beta_states[i]?,
                    // #prep-dedupe: the ascent's mutable iterate-0 seed is
                    // cloned from the shared memoized prep (the historical
                    // code moved it out of a per-lane recompute).
                    beta_signed.clone(),
                    opt.thresholds,
                    &opt.row_verified[i],
                    num_specs,
                )?;
                (lo, hi, Some(best_beta))
            } else {
                let result = serial_gpu
                    .crown_backward_gpu_resnet_sound_beta(
                        segments,
                        &shared_seed,
                        in_lo,
                        in_hi,
                        beta_signed,
                        frontier_abs,
                        node_abs,
                    )
                    .ok()?;
                (result.lower_bounds, result.upper_bounds, None)
            };
            if lower_v.len() != num_specs
                || upper_v.len() != num_specs
                || lower_v.iter().chain(upper_v.iter()).any(|v| !v.is_finite())
            {
                return None;
            }
            let lower =
                ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[num_specs]), lower_v).ok()?;
            let upper =
                ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[num_specs]), upper_v).ok()?;
            let output_bounds =
                BoundedTensor::new_repaired(lower, upper, ny_tensor::RepairStrategy::Widen).ok()?;
            Some((output_bounds, opt_beta))
        };

        // Default: serial (byte-identical to the historical loop). Gated: rayon
        // fan-out over the independent domains. `collect()` preserves domain
        // order; a `None` from any domain short-circuits the `?` below exactly as
        // the serial loop's `?` returned `None` from the whole function.
        let parallel = std::env::var("NY_BAB_RESNET_PARALLEL").ok().as_deref() == Some("1");
        let outcomes: Vec<Option<(BoundedTensor, Option<GraphBetaState>)>> = if parallel {
            use rayon::iter::{IntoParallelIterator, ParallelIterator};
            (0..n_domains)
                .into_par_iter()
                .map(|i| {
                    let _rayon_task_guard = crate::faer_parallelism::RayonTaskGuard::new();
                    compute_domain(i)
                })
                .collect()
        } else {
            (0..n_domains).map(compute_domain).collect()
        };
        let mut results: Vec<BoundedTensor> = Vec::with_capacity(n_domains);
        let mut optimized_betas: Vec<Option<GraphBetaState>> = Vec::with_capacity(n_domains);
        for outcome in outcomes {
            let (output_bounds, opt_beta) = outcome?;
            results.push(output_bounds);
            optimized_betas.push(opt_beta);
        }

        if std::env::var("NY_BETA_GPU_PROBE").ok().as_deref() == Some("1") {
            let n_opt = optimized_betas.iter().filter(|b| b.is_some()).count();
            eprintln!(
                "[beta-gpu-batched:{probe_tag}] SUCCESS n_domains={n_domains} num_specs={num_specs} od={output_dim} beta_opt={n_opt}"
            );
        }
        // Serial fallback lane never steps α — callers keep inherited α.
        Some((results, optimized_betas, vec![None; n_domains]))
    }

    /// Per-domain analytic β ascent on the GPU resnet backward
    /// (#w4-split-tightening) — the CPU
    /// `optimize_graph_beta_analytical_multi_objective_with_cache` loop at GPU
    /// speed. Per iteration: one sound β-folded GPU backward (same certified
    /// error + explosion-merge machinery as the single-shot lane) returns the
    /// spec-row bounds AND the pre-relaxation lower A-values gathered at the
    /// domain's split neurons; the critical row (min unverified margin, NaN →
    /// −∞) supplies the analytic gradient `∂lb/∂β_k = −sign_k·A_lower[crit,k]`
    /// (`GraphBetaState::compute_gradients_for_spec_rows` rule); the SAME Adam
    /// step (`gradient_step_adam`, β ← max(0, ·)) moves β; convergence when
    /// max|grad| < `beta_tolerance`; the wall-clock deadline breaks the loop
    /// between iterations.
    ///
    /// Returns `(best_lower, best_upper, best_beta)`:
    /// * bounds = element-wise tightest per row across ALL sound iterates —
    ///   iterate 0 runs the caller-supplied inherited β (`beta_signed0`), so the
    ///   result is never looser than the single-shot lane; each iterate is a
    ///   valid dual bound for its β ≥ 0 over the SAME sub-domain, so the per-row
    ///   intersection is sound.
    /// * β = the best-margin iterate's snapshot (CPU parity: the returned state
    ///   warm-starts the children).
    ///
    /// `None` ⇔ the FIRST pass failed (mirrors the single-shot `.ok()?`: the
    /// caller falls back to the CPU per-child path). Later-iteration failures
    /// keep the best-so-far (sound).
    // pub(in engine::graph) so the #[doc(hidden)] gpu_beta_debug test surface can
    // drive the exact production loop (the parity/monotonicity tests live in ny-gpu).
    #[allow(clippy::too_many_arguments)]
    pub(in crate::beta_crown::engine::graph) fn gpu_beta_optimize_domain(
        &self,
        gpu: &dyn ny_core::GpuCrownBackward,
        segments: &[ny_core::GpuResnetSegment],
        relu_names: &[String],
        frontier_abs: &[Vec<f32>],
        node_abs: &[Vec<f32>],
        seed: &ny_core::GpuCrownSeed,
        in_lo: &[f32],
        in_hi: &[f32],
        bounds_cache: &HashMap<String, Arc<BoundedTensor>>,
        inherited_beta: &GraphBetaState,
        beta_signed0: Vec<Vec<f32>>,
        thresholds: &[f32],
        row_verified: &[bool],
        num_specs: usize,
    ) -> Option<(Vec<f32>, Vec<f32>, GraphBetaState)> {
        let probe = std::env::var("NY_BETA_GPU_PROBE").ok().as_deref() == Some("1");
        // Default 3 iterations for the multi-objective GPU lane (#w4-split-
        // tightening, measured): the analytic β ascent converges in ~2-3 steps
        // per domain on the cifar100 resnets; 10 (the CPU-loop default) buys
        // no extra tightening but ~3x the per-domain wall, halving explored
        // domains. `NY_MO_GPU_BETA_ITERS_SERIAL` overrides for A/B on THIS
        // serial per-domain lane only. Deliberately NOT `NY_MO_GPU_BETA_ITERS`:
        // that knob is scoped to the wide β/α ascent (`gpu_beta_optimize_wide`,
        // its documented purpose) — this lane is also the wide lane's
        // fallthrough, and each iteration here is one FULL GPU backward PER
        // DOMAIN (vs one wide pass for the whole batch), so a shared knob at
        // 8-16 iterations silently multiplied the fallback's wall by the same
        // factor times n_domains and could eat the whole phase budget.
        const MO_GPU_BETA_ITERS_DEFAULT: usize = 3;
        let iterations = std::env::var("NY_MO_GPU_BETA_ITERS_SERIAL")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(MO_GPU_BETA_ITERS_DEFAULT)
            .max(1);

        // Per-ReLU neuron count (fold order) for entry-index validation.
        let relu_nn: Vec<usize> = relu_names
            .iter()
            .map(|name| bounds_cache.get(name).map_or(0, |b| b.lower().len()))
            .collect();

        // Gather lists (one per ReLU, fold order) + the β-entry → (relu, column)
        // map. Only ReLU (split_point==0) entries with in-range neuron indices
        // participate — exactly the entries the β fold itself applies.
        let mut gather_idx: Vec<Vec<u32>> = vec![Vec::new(); relu_names.len()];
        let mut entry_map: Vec<Option<(usize, usize)>> =
            Vec::with_capacity(inherited_beta.entries.len());
        let mut beta = inherited_beta.clone();
        for entry in &inherited_beta.entries {
            let mapped = relu_names
                .iter()
                .position(|n| n == entry.node_name())
                .filter(|&r| entry.split_point().abs() < 1e-6 && entry.neuron_idx() < relu_nn[r])
                .map(|r| {
                    let col = entry.neuron_idx() as u32;
                    let pos = gather_idx[r]
                        .iter()
                        .position(|&c| c == col)
                        .unwrap_or_else(|| {
                            gather_idx[r].push(col);
                            gather_idx[r].len() - 1
                        });
                    (r, pos)
                });
            entry_map.push(mapped);
        }
        if entry_map.iter().all(|m| m.is_none()) {
            // Nothing optimizable — signal "no opt" by failing over to the
            // single-shot path in the caller via a plain first pass here.
            let r = gpu
                .crown_backward_gpu_resnet_sound_beta(
                    segments,
                    seed,
                    in_lo,
                    in_hi,
                    &beta_signed0,
                    frontier_abs,
                    node_abs,
                )
                .ok()?;
            return Some((r.lower_bounds, r.upper_bounds, beta));
        }

        // Rebuild the per-ReLU signed-β fold input from the current entry values.
        let build_beta_signed = |b: &GraphBetaState| -> Vec<Vec<f32>> {
            let mut out: Vec<Vec<f32>> = relu_nn.iter().map(|&nn| vec![0.0f32; nn]).collect();
            for (name, bs) in relu_names.iter().zip(out.iter_mut()) {
                for e in b.entries_for_node(name) {
                    if e.split_point().abs() < 1e-6 {
                        let idx = e.neuron_idx();
                        if idx < bs.len() {
                            bs[idx] = e.signed_value();
                        }
                    }
                }
            }
            out
        };

        let mut best_lo: Option<Vec<f32>> = None;
        let mut best_hi: Option<Vec<f32>> = None;
        let mut best_margin = f32::NEG_INFINITY;
        let mut best_beta: Option<GraphBetaState> = None;
        let mut iters_run = 0usize;

        for iter in 0..iterations {
            // Deadline between iterations: each GPU pass is bounded; stopping
            // early just returns the best sound bounds so far (#3109 analog).
            if self.config.alpha_config.past_deadline() {
                if iter == 0 {
                    return None;
                }
                break;
            }
            let beta_signed = if iter == 0 {
                beta_signed0.clone()
            } else {
                build_beta_signed(&beta)
            };
            let pass = gpu.crown_backward_gpu_resnet_sound_beta_grad(
                segments,
                seed,
                in_lo,
                in_hi,
                &beta_signed,
                &gather_idx,
                frontier_abs,
                node_abs,
            );
            let pass = match pass {
                Ok(p) => p,
                Err(_) if iter == 0 => return None,
                Err(_) => break,
            };
            if pass.lower_bounds.len() != num_specs || pass.upper_bounds.len() != num_specs {
                if iter == 0 {
                    return None;
                }
                break;
            }
            if iter == 0
                && pass
                    .lower_bounds
                    .iter()
                    .chain(pass.upper_bounds.iter())
                    .any(|v| !v.is_finite())
            {
                // Mirror the single-shot lane: a non-finite inherited-β pass
                // fails the domain over to the CPU per-child path.
                return None;
            }
            iters_run = iter + 1;

            // Element-wise tightest merge across sound iterates (skip
            // non-finite fresh rows; each row's every iterate is a valid bound).
            match (&mut best_lo, &mut best_hi) {
                (Some(bl), Some(bh)) => {
                    for s in 0..num_specs {
                        if pass.lower_bounds[s].is_finite() && pass.lower_bounds[s] > bl[s] {
                            bl[s] = pass.lower_bounds[s];
                        }
                        if pass.upper_bounds[s].is_finite() && pass.upper_bounds[s] < bh[s] {
                            bh[s] = pass.upper_bounds[s];
                        }
                    }
                }
                _ => {
                    best_lo = Some(pass.lower_bounds.clone());
                    best_hi = Some(pass.upper_bounds.clone());
                }
            }

            // Critical row: min margin over unverified rows (NaN → −∞), the
            // disjunctive rule of `select_critical_disjunctive`.
            let mut critical: Option<usize> = None;
            let mut critical_margin = f32::INFINITY;
            let mut min_margin = f32::INFINITY;
            for s in 0..num_specs {
                if row_verified.get(s).copied().unwrap_or(false) {
                    continue;
                }
                let m = pass.lower_bounds[s] - thresholds[s];
                let m = if m.is_nan() { f32::NEG_INFINITY } else { m };
                min_margin = min_margin.min(m);
                if m < critical_margin {
                    critical_margin = m;
                    critical = Some(s);
                }
            }
            let Some(critical) = critical else {
                break; // every row verified for this child — nothing to optimize
            };
            if min_margin > best_margin {
                best_margin = min_margin;
                best_beta = Some(beta.clone());
            }

            // Analytic gradients for the critical row (CPU
            // `compute_gradients_for_spec_row`: grad = −sign·A_lower[crit, k];
            // missing A ⇒ 0).
            beta.zero_grad();
            let mut max_grad = 0.0f32;
            for (e_idx, mapped) in entry_map.iter().enumerate() {
                let grad = match mapped {
                    Some((r, pos)) => {
                        let g = pass.beta_gather.get(*r);
                        let n_idx = gather_idx[*r].len();
                        match g {
                            Some(vals) if vals.len() == num_specs * n_idx => {
                                let a = vals[critical * n_idx + pos];
                                -beta.entries[e_idx].sign() * a
                            }
                            _ => 0.0,
                        }
                    }
                    None => 0.0,
                };
                beta.entries[e_idx].grad = grad;
                max_grad = ny_core::nan_propagating_max(max_grad, grad.abs());
            }

            // Same Adam step + convergence rule as the CPU loop.
            beta.gradient_step_adam(&self.config.adaptive_config, iter + 1);
            if max_grad < self.config.beta_tolerance {
                break;
            }
        }

        let (lo, hi) = (best_lo?, best_hi?);
        if probe {
            eprintln!(
                "[beta-opt] iters={iters_run}/{iterations} entries={} best_margin={best_margin:.5}",
                entry_map.iter().filter(|m| m.is_some()).count()
            );
        }
        // CPU parity: sync the returned β to the best-margin snapshot so the
        // children inherit the state that produced the best bound.
        if let Some(bb) = best_beta {
            beta = bb;
        }
        Some((lo, hi, beta))
    }

    /// #batched-bab part A: the WIDE batched analog of [`gpu_beta_optimize_domain`] — run
    /// the per-domain analytic β ascent for ALL `n_domains` subdomains at once, ONE wide
    /// grad backward per iteration (over N = n_domains·num_specs stacked rows), then per
    /// domain do the SAME critical-row / analytic-gradient / Adam / best-tracking as the
    /// serial loop. The union of every opt-eligible domain's split columns is gathered
    /// once per ReLU; each domain reads its own columns (remapped to union positions) from
    /// its own rows (block d = [d·nsp, (d+1)·nsp)). Returns per domain `(bound, β, α)`
    /// exactly as the serial `compute_domain` loop assembles the first two: opt-eligible
    /// domains return the element-wise-tightest bound across sound iterates +
    /// `Some(best-margin β)`; frozen / ineligible domains return their single-shot
    /// (inherited-β) bound + `None` (matching serial's else-branch). The third slot is the
    /// per-domain best-margin α snapshot — `Some` only under `NY_WIDE_ALPHA_UNSHARED=1`
    /// for α-participating domains (else all `None` = callers keep inherited α;
    /// byte-identical to the historical behavior). NON-soundness-critical (β/α steer
    /// only; the bound is the sound wide fold). `None` (whole batch) ⇔ iter-0 GPU error /
    /// non-finite ⇒ the caller drops to the serial per-domain path (0-wrong moat).
    #[allow(clippy::too_many_arguments)]
    pub(in crate::beta_crown::engine::graph) fn gpu_beta_optimize_wide(
        &self,
        gpu: &dyn ny_core::GpuCrownBackward,
        // #prep-dedupe: borrowed from the caller's per-domain memo (one prep
        // per domain per batched call, shared across all three lanes).
        preps: &[&ResnetDomainPrep],
        seed: &ny_core::GpuCrownSeed,
        bounds_caches: &[&HashMap<String, Arc<BoundedTensor>>],
        beta_states: &[Option<&GraphBetaState>],
        alpha_states: &[Option<&GraphDomainAlphaState>],
        relu_pre_name: &HashMap<String, String>,
        beta_opt: &GpuBetaOptSpec,
        num_specs: usize,
    ) -> Option<(
        Vec<BoundedTensor>,
        Vec<Option<GraphBetaState>>,
        Vec<Option<GraphDomainAlphaState>>,
    )> {
        let n_domains = preps.len();
        if n_domains == 0 || bounds_caches.len() != n_domains || beta_states.len() != n_domains {
            return None;
        }
        let nsp = num_specs;
        let probe = std::env::var("NY_BETA_GPU_PROBE").ok().as_deref() == Some("1");
        // Default 3 = serial-lane parity. `NY_MO_GPU_BETA_ITERS` scales ONLY
        // this wide β/α ascent (its documented purpose; the serial fallback
        // lane keeps its own default / `NY_MO_GPU_BETA_ITERS_SERIAL`, so
        // iteration-scaling A/Bs here can't multiply the per-domain fallback).
        let iterations = std::env::var("NY_MO_GPU_BETA_ITERS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(3)
            .max(1);
        // #hard-six tail-iters (dark, `NY_MO_GPU_BETA_ITERS_TAIL=k`, default
        // off): PER-DOMAIN extended ascent at the pinned tail. A domain is
        // tail-eligible iff its BaB depth ≥ `NY_TAIL_MIN_DEPTH` (default 8)
        // AND its unverified spec-row count ≤ `NY_TAIL_MAX_ACTIVE` (default
        // 2) — the "n_active ≤ 2, depth ≥ 8" pinned-straggler signature.
        // Non-tail domains stop stepping after the base iteration budget
        // (their best iterate is kept exactly as before); tail domains keep
        // ascending to `k` iterations. The measured global iters=16 collapse
        // was whole-tree depth starvation — this spends the extra budget only
        // on the nearly-closed deep stragglers. SOUND: iteration counts only
        // schedule work; every iterate is the sound wide fold, best-merged.
        let base_iterations = iterations;
        let tail_iters_env = std::env::var("NY_MO_GPU_BETA_ITERS_TAIL")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&k| k >= 1);
        let tail_eligible: Vec<bool> = if let Some(_k) = tail_iters_env {
            let min_depth = std::env::var("NY_TAIL_MIN_DEPTH")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(8);
            let max_active = std::env::var("NY_TAIL_MAX_ACTIVE")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(2);
            (0..n_domains)
                .map(|d| {
                    let deep = beta_opt.depths.get(d).copied().unwrap_or(0) >= min_depth;
                    let nearly_closed = beta_opt
                        .row_verified
                        .get(d)
                        .is_some_and(|rv| rv.iter().filter(|&&v| !v).count() <= max_active);
                    deep && nearly_closed
                })
                .collect()
        } else {
            vec![false; n_domains]
        };
        let n_tail = tail_eligible.iter().filter(|&&t| t).count();
        let iterations = match tail_iters_env {
            Some(k) if n_tail > 0 => {
                if probe {
                    eprintln!(
                        "[wide-tail-iters] n_domains={n_domains} tail_eligible={n_tail} iters {base_iterations}->{k}"
                    );
                }
                k.max(base_iterations)
            }
            _ => base_iterations,
        };
        // auto_LiRPA per-spec α parity (dark, NY_AB_PARITY=1): the round-robin
        // objective row (below) needs enough replay iterations to target every
        // active spec at least once, so floor the ascent budget. Off ⇒ no change.
        let ab_parity = wide_alpha_true::ab_parity_enabled();
        let iterations = if ab_parity {
            iterations.max(wide_alpha_true::ab_parity_iters())
        } else {
            iterations
        };
        // Hydra trajectory capture is useful only with at least two iterates.
        // It retains a small fixed active set rather than every spec row, so the
        // optional coefficient download cannot grow with the full objective set.
        let mut facet_collector = (iterations >= 2)
            .then(|| WideFacetCollector::from_env(n_domains, nsp, preps[0].in_lo.len()))
            .flatten();

        // Per-domain optimization state (mirrors gpu_beta_optimize_domain's locals).
        struct DomOpt {
            opt_eligible: bool,
            relu_nn: Vec<usize>,
            gather_idx: Vec<Vec<u32>>,
            entry_map: Vec<Option<(usize, usize)>>,
            entry_union: Vec<Option<(usize, usize)>>,
            beta: Option<GraphBetaState>,
            best_lo: Option<Vec<f32>>,
            best_hi: Option<Vec<f32>>,
            best_margin: f32,
            best_beta: Option<GraphBetaState>,
            done: bool,
            iters_run: usize,
            /// Critical (min-margin unverified) spec row of the CURRENT iterate —
            /// the objective row the TRUE wide-alpha gradient replays (#task-11).
            crit_row: Option<usize>,
            /// Margin of `crit_row` on the CURRENT iterate — the worst-domain
            /// selector for `NY_WIDE_ALPHA_TRUE_DOMS=worst` (#task-11 lever 1).
            crit_margin: f32,
        }
        let n_relu = preps[0].relu_names.len();
        // Build per-domain state + the per-ReLU UNION of split columns.
        let mut union_cols: Vec<Vec<u32>> = vec![Vec::new(); n_relu];
        let mut dstate: Vec<DomOpt> = Vec::with_capacity(n_domains);
        for d in 0..n_domains {
            if preps[d].relu_names.len() != n_relu {
                return None; // heterogeneous ReLU count — caller falls back to serial
            }
            let inherited = beta_states[d];
            let opt_eligible = beta_opt.eligible.get(d).copied().unwrap_or(false)
                && inherited.is_some_and(|b| !b.is_empty())
                && beta_opt.row_verified.get(d).is_some()
                && beta_opt.thresholds.len() == nsp;
            let relu_nn: Vec<usize> = preps[d]
                .relu_names
                .iter()
                .map(|name| bounds_caches[d].get(name).map_or(0, |b| b.lower().len()))
                .collect();
            let mut gather_idx: Vec<Vec<u32>> = vec![Vec::new(); n_relu];
            let mut entry_map: Vec<Option<(usize, usize)>> = Vec::new();
            let beta = if opt_eligible {
                let inh = inherited.expect("opt_eligible ⇒ Some");
                entry_map.reserve(inh.entries.len());
                for entry in &inh.entries {
                    let mapped = preps[d]
                        .relu_names
                        .iter()
                        .position(|n| n == entry.node_name())
                        .filter(|&r| {
                            entry.split_point().abs() < 1e-6 && entry.neuron_idx() < relu_nn[r]
                        })
                        .map(|r| {
                            let col = entry.neuron_idx() as u32;
                            let pos =
                                gather_idx[r]
                                    .iter()
                                    .position(|&c| c == col)
                                    .unwrap_or_else(|| {
                                        gather_idx[r].push(col);
                                        gather_idx[r].len() - 1
                                    });
                            (r, pos)
                        });
                    entry_map.push(mapped);
                }
                // Only truly-optimizable if at least one entry mapped (serial:625).
                if entry_map.iter().all(|m| m.is_none()) {
                    // All-None eligible domain: serial returns Some(inherited) + its bound.
                    Some(inh.clone())
                } else {
                    for r in 0..n_relu {
                        union_cols[r].extend_from_slice(&gather_idx[r]);
                    }
                    Some(inh.clone())
                }
            } else {
                None
            };
            let optimizable = opt_eligible && entry_map.iter().any(|m| m.is_some());
            dstate.push(DomOpt {
                opt_eligible,
                relu_nn,
                gather_idx,
                entry_map,
                entry_union: Vec::new(), // filled after union_cols is finalized
                // best_beta seeded to the inherited β for opt-eligible domains so a
                // never-improving (or all-None) domain returns serial's inherited β.
                best_beta: if opt_eligible { beta.clone() } else { None },
                beta,
                best_lo: None,
                best_hi: None,
                best_margin: f32::NEG_INFINITY,
                done: !optimizable, // frozen / all-None domains never step
                iters_run: 0,
                crit_row: None,
                crit_margin: f32::INFINITY,
            });
        }
        // Finalize the union columns (sorted-unique) + per-domain pos-in-union remap.
        for r in 0..n_relu {
            union_cols[r].sort_unstable();
            union_cols[r].dedup();
        }
        for ds in dstate.iter_mut() {
            let mut eu: Vec<Option<(usize, usize)>> = Vec::with_capacity(ds.entry_map.len());
            for m in &ds.entry_map {
                eu.push(m.and_then(|(r, pos_local)| {
                    let col = ds.gather_idx[r][pos_local];
                    union_cols[r].iter().position(|&c| c == col).map(|u| (r, u))
                }));
            }
            ds.entry_union = eu;
        }
        let union_refs: Vec<&[u32]> = union_cols.iter().map(|v| v.as_slice()).collect();

        // #w4 wide α+β ascent (NY_BAB_RESNET_WIDE_ALPHA=1 — DARK, and the 2026-07-08
        // cifar100 prop_1498 [converge] A/B REFUTED the local gradient rule: with the
        // mechanism fully validated (wide-vs-serial grad oracle, mapcheck 997/997
        // slope↔state matches, SCALE=0 byte-identical to baseline), stepping α by
        // pre_lower·Σ_rows max(A_lower,0) degraded bounds PROPORTIONALLY TO LR IN
        // BOTH SIGNS (lr 0.001: −0.002/line; lr 0.01: −0.024/line; 9→7 domains
        // verified). The warmup-converged α is at a joint optimum this LOCAL
        // single-layer rule cannot improve post-split: it ignores the β interaction,
        // the input-box concretization coupling, and cross-layer α coupling. The
        // genuine lever is the TRUE joint gradient (autograd through the whole bound
        // computation, as alpha-beta-CROWN does) feeding this same — validated —
        // channel/write-back machinery. Keep dark until that gradient exists.):
        // re-optimize each sub-domain's α inside this loop — the
        // batched lane otherwise folds warmup-inherited ROOT α that never adapts to
        // the sub-domain's tighter post-split bounds (the cifar100 tightness gap).
        // Mechanics per iteration: the wide grad backward also returns per-domain
        // analytic α gradients (∂lb/∂α_i = pre_lower[i]·Σ_rows max(A_lower,0), per
        // domain block); each participating domain Adam-ascends its OWNED clone of
        // the inherited GraphDomainAlphaState ([0,1]-projected by set_alpha) and the
        // new slopes are written into that domain's CLONED segments (Arc-shared
        // weights — clones are cheap; caller's preps stay untouched so any fallback
        // path is byte-identical). SOUND regardless of the gradients: every iterate's
        // bound comes from the sound wide fold with α ∈ [0,1], β ≥ 0, and only the
        // element-wise-tightest sound iterate is kept.
        let alpha_on =
            crate::network::resnet_beta_gpu_wide_alpha_enabled() && alpha_states.len() == n_domains;
        if probe && crate::network::resnet_beta_gpu_wide_alpha_enabled() {
            eprintln!(
                "[wide-alpha] gate0: alpha_on={alpha_on} states_len={} n_domains={n_domains}",
                alpha_states.len()
            );
        }
        // Owned α state clones + per-domain per-ReLU pre-activation lower tables
        // (stable or untracked neurons masked to 0 ⇒ zero gradient ⇒ never stepped).
        let mut alpha_opt: Vec<Option<GraphDomainAlphaState>> = vec![None; n_domains];
        let mut alpha_pl: Vec<Vec<Vec<f32>>> = Vec::new();
        let mut seg_store: Vec<Vec<ny_core::GpuResnetSegment>> = Vec::new();
        if alpha_on {
            alpha_pl = (0..n_domains)
                .map(|d| {
                    let mut tables: Vec<Vec<f32>> = dstate[d]
                        .relu_nn
                        .iter()
                        .map(|&nn| vec![0.0f32; nn])
                        .collect();
                    let Some(st) = alpha_states[d] else {
                        return tables;
                    };
                    if st.is_empty() {
                        return tables;
                    }
                    let mut any = false;
                    for (r, name) in preps[d].relu_names.iter().enumerate() {
                        // PRE-activation bounds live under the ReLU's INPUT node.
                        let Some(b) = relu_pre_name
                            .get(name)
                            .and_then(|p| bounds_caches[d].get(p))
                        else {
                            continue;
                        };
                        let Some(tracked) = st.neurons().get(name) else {
                            continue;
                        };
                        // FLAT iteration: conv pre-activation bounds are 4-D ArrayD —
                        // scalar `lo[i]` indexing panics on IxDyn; iterate the flat
                        // element order (the same order relu_nn/extraction use).
                        let nn = tables[r].len();
                        for (i, (&l, &u)) in b.lower().iter().zip(b.upper().iter()).enumerate() {
                            if i >= nn {
                                break;
                            }
                            // Unstable only: stable neurons' slopes are pinned 1/0
                            // by extraction and MUST NOT be stepped or rewritten.
                            if l < 0.0 && u > 0.0 && tracked.contains_key(&i) {
                                tables[r][i] = l;
                                any = true;
                            }
                        }
                    }
                    if any {
                        alpha_opt[d] = alpha_states[d].cloned();
                    }
                    tables
                })
                .collect();
            if alpha_opt.iter().any(|a| a.is_some()) {
                seg_store = preps.iter().map(|p| p.segments.clone()).collect();
            }
            if probe {
                let n_some = alpha_states.iter().filter(|a| a.is_some()).count();
                let n_nonempty = alpha_states
                    .iter()
                    .filter(|a| a.is_some_and(|s| !s.is_empty()))
                    .count();
                let n_part = alpha_opt.iter().filter(|a| a.is_some()).count();
                let name0 = preps
                    .first()
                    .and_then(|p| p.relu_names.first())
                    .map(String::as_str)
                    .unwrap_or("-");
                let key0 = alpha_states
                    .iter()
                    .flatten()
                    .next()
                    .and_then(|s| s.neurons().keys().next())
                    .map(String::as_str)
                    .unwrap_or("-");
                eprintln!(
                    "[wide-alpha] gate: states_some={n_some}/{n} nonempty={n_nonempty} participants={n_part} relu0={name0} key0={key0}",
                    n = n_domains
                );
            }
        }
        let alpha_active = !seg_store.is_empty();
        // #hard-six per-domain UNSHARED α (dark, NY_WIDE_ALPHA_UNSHARED=1):
        // persist each participating domain's best-margin α snapshot so the
        // evaluated child KEEPS its ascended per-neuron α and its descendants
        // inherit it via `from_parent` — the ascent compounds along the branch
        // exactly like β, instead of restarting from the root α every batch.
        // See `wide_alpha_true::wide_alpha_unshared_enabled` for the recon.
        let unshared = alpha_active && wide_alpha_true::wide_alpha_unshared_enabled();
        // BARRIER-1 sound f64 lineage recovery (dark, default-OFF ⇒ byte-identical).
        // `NY_F64_LINEAGE_RECOVER=1` re-folds each opt-eligible domain's critical row
        // in f64 (SOUND accounting at u=2⁻⁵³) and merges max into best_lo, recovering
        // the ~0.088 CONSERVATIVE f32 sound-fold certified-error tax (measured). By
        // default only at iter 0 (the dominant iterate; β/α buy ≪0.088 after) to bound
        // the CPU f64 fold cost; `NY_F64_LINEAGE_RECOVER_EVERY=1` runs it every iter.
        let f64_recover = std::env::var("NY_F64_LINEAGE_RECOVER").ok().as_deref() == Some("1");
        let f64_recover_every = std::env::var("NY_F64_LINEAGE_RECOVER_EVERY")
            .ok()
            .as_deref()
            == Some("1");
        // #mn-head-facet increment 1 (dark, NY_MN_HEAD_FACET=1; rides f64_recover):
        // resolve the registered HEAD coupling-facet fold β-grid's global
        // `target_act` against THIS lane's actual `relu_names` (the authoritative
        // fold order the f64 recovery walks) by the head ReLU's NAME, and refuse
        // the whole set on any mismatch (folding onto the wrong ReLU would be an
        // invalid Lagrangian). Empty folds are dropped. Resolved ONCE; each fold is
        // then sound `max`-merged into best_lo alongside the baseline f64 recovery.
        // `None` unless the gate is armed AND an entry is registered ⇒ byte-
        // identical to today's recovery.
        let head_folds: Option<Vec<ny_core::head_f64_fold::HeadF64Fold>> = if f64_recover {
            ny_core::head_f64_fold::active_head_f64_folds()
                .and_then(|folds| {
                    let rn = &preps.first()?.relu_names;
                    let name = folds.first()?.relu_name.clone();
                    let ta = rn.iter().position(|n| n == &name)?;
                    Some(
                        folds
                            .iter()
                            .filter(|f| !f.is_empty())
                            .map(|f| {
                                let mut c = f.clone();
                                c.target_act = ta;
                                c
                            })
                            .collect::<Vec<_>>(),
                    )
                })
                .filter(|v| !v.is_empty())
        } else {
            None
        };
        if let Some(folds) = head_folds.as_ref() {
            eprintln!(
                "[mn-head-facet] armed: {} fold(s) at relu={} target_act={} head_width={} (max-intersect into f64 critical-row recovery)",
                folds.len(),
                folds[0].relu_name,
                folds[0].target_act,
                folds[0].head_width,
            );
        }
        let mut best_alpha: Vec<Option<GraphDomainAlphaState>> = vec![None; n_domains];
        // Fold-order slope write-back: walk segments exactly as the extractor orders
        // relu_names (segments in vec order; per segment the branch layer vec in
        // order, ResidualProj F then P) and overwrite ONLY the unstable tracked
        // neurons' lower_slope with the domain's stepped α. Upper slope/intercepts
        // are α-independent (chord); lower intercept ≡ 0 — nothing else changes.
        let write_back_alpha = |segs: &mut [ny_core::GpuResnetSegment],
                                relu_names: &[String],
                                pl: &[Vec<f32>],
                                st: &GraphDomainAlphaState| {
            let mut r = 0usize;
            let mut apply = |layers: &mut [ny_core::GpuCrownLayer]| {
                for l in layers.iter_mut() {
                    if let ny_core::GpuCrownLayer::Activation { lower_slope, .. } = l {
                        if let (Some(name), Some(mask)) = (relu_names.get(r), pl.get(r)) {
                            for (i, s) in lower_slope.iter_mut().enumerate() {
                                if mask.get(i).copied().unwrap_or(0.0) != 0.0 {
                                    *s = st.alpha(name, i);
                                }
                            }
                        }
                        r += 1;
                    }
                }
            };
            for seg in segs.iter_mut() {
                match seg {
                    ny_core::GpuResnetSegment::Chain(l)
                    | ny_core::GpuResnetSegment::Residual(l) => apply(l),
                    ny_core::GpuResnetSegment::ResidualProj(f, p) => {
                        apply(f);
                        apply(p);
                    }
                }
            }
        };
        let alpha_pl_refs: Vec<&[Vec<f32>]> = alpha_pl.iter().map(|v| v.as_slice()).collect();
        // MAPPING SELF-CHECK (probe-only): at iter 0 the extracted lower_slope for
        // every masked (tracked+unstable) neuron MUST already equal st.alpha(name,i)
        // (both were built from the same inherited state). Any mismatch ⇒ the
        // relu_names[r] ↔ Activation-position walk is misaligned.
        if probe && alpha_active {
            for d in 0..n_domains {
                let Some(st) = alpha_opt[d].as_ref() else {
                    continue;
                };
                let mut r = 0usize;
                let mut mism = 0usize;
                let mut checked = 0usize;
                let mut check = |layers: &[ny_core::GpuCrownLayer]| {
                    for l in layers.iter() {
                        if let ny_core::GpuCrownLayer::Activation { lower_slope, .. } = l {
                            if let (Some(name), Some(mask)) =
                                (preps[d].relu_names.get(r), alpha_pl[d].get(r))
                            {
                                for (i, s0) in lower_slope.iter().enumerate() {
                                    if mask.get(i).copied().unwrap_or(0.0) != 0.0 {
                                        checked += 1;
                                        if (s0 - st.alpha(name, i)).abs() > 1e-6 {
                                            mism += 1;
                                        }
                                    }
                                }
                            }
                            r += 1;
                        }
                    }
                };
                for seg in seg_store[d].iter() {
                    match seg {
                        ny_core::GpuResnetSegment::Chain(l)
                        | ny_core::GpuResnetSegment::Residual(l) => check(l),
                        ny_core::GpuResnetSegment::ResidualProj(f, p2) => {
                            check(f);
                            check(p2);
                        }
                    }
                }
                eprintln!("[wide-alpha] mapcheck dom={d} checked={checked} mismatched={mism}");
            }
        }

        // Per-ReLU signed-β rebuild from a domain's current β (serial:643-656).
        let build_beta_signed =
            |relu_names: &[String], relu_nn: &[usize], b: &GraphBetaState| -> Vec<Vec<f32>> {
                let mut out: Vec<Vec<f32>> = relu_nn.iter().map(|&nn| vec![0.0f32; nn]).collect();
                for (name, bs) in relu_names.iter().zip(out.iter_mut()) {
                    for e in b.entries_for_node(name) {
                        if e.split_point().abs() < 1e-6 {
                            let idx = e.neuron_idx();
                            if idx < bs.len() {
                                bs[idx] = e.signed_value();
                            }
                        }
                    }
                }
                out
            };

        // #task-11 replay throttles (dark; only read under NY_WIDE_ALPHA_TRUE=1):
        // replay every k-th ascent iteration / only the worst-margin domain; the
        // cache carries each domain's LAST captured true gradient so Adam keeps
        // stepping between replays (stale steps are sound — α stays in [0,1] and
        // the best-iterate merge keeps only improvements).
        let true_every = wide_alpha_true::wide_alpha_true_every();
        let true_worst_only = wide_alpha_true::wide_alpha_true_worst_only();
        let mut true_grad_cache: Vec<Option<Vec<Vec<f32>>>> = vec![None; n_domains];

        // ONE wide grad backward per iteration.
        for iter in 0..iterations {
            if self.config.alpha_config.past_deadline() {
                if iter == 0 {
                    return None;
                }
                break;
            }
            // Per-domain β_signed for this iterate (inherited on iter 0 / frozen / done).
            let beta_signed_all: Vec<Vec<Vec<f32>>> = (0..n_domains)
                .map(|d| {
                    let ds = &dstate[d];
                    if iter == 0 || ds.done || !ds.opt_eligible {
                        preps[d].beta_signed.clone()
                    } else if let Some(b) = &ds.beta {
                        build_beta_signed(&preps[d].relu_names, &ds.relu_nn, b)
                    } else {
                        preps[d].beta_signed.clone()
                    }
                })
                .collect();
            let refs: Vec<ny_core::GpuResnetBatchedDomainRef> = (0..n_domains)
                .map(|d| ny_core::GpuResnetBatchedDomainRef {
                    // α-active: the domain's cloned segments carry its stepped slopes
                    // (iter 0 clones are byte-identical to preps ⇒ same bound).
                    segments: if alpha_active {
                        &seg_store[d]
                    } else {
                        &preps[d].segments
                    },
                    input_lower: &preps[d].in_lo,
                    input_upper: &preps[d].in_hi,
                    beta_signed: &beta_signed_all[d],
                    frontier_abs: &preps[d].frontier_abs,
                    node_abs: &preps[d].node_abs,
                })
                .collect();
            let mut trajectory_coeff = None;
            let pass = if facet_collector.is_some() {
                match gpu.crown_backward_gpu_resnet_sound_beta_batched_trajectory(
                    &refs,
                    seed,
                    &union_refs,
                    if alpha_active { &alpha_pl_refs } else { &[] },
                ) {
                    Ok(trajectory) => {
                        trajectory_coeff = Some(trajectory.coeff);
                        Ok((
                            trajectory.bounds,
                            trajectory.alpha_grads,
                            trajectory.beta_gather,
                        ))
                    }
                    Err(_) => {
                        // A backend may support wide gradients but not coefficient
                        // capture. Disable only FacetBank and immediately retry the
                        // pre-existing sound pass; optimization still proceeds.
                        facet_collector = None;
                        if self.config.alpha_config.past_deadline() {
                            Err(NyError::UnsupportedOp(
                                "trajectory capture crossed the verification deadline".into(),
                            ))
                        } else {
                            gpu.crown_backward_gpu_resnet_sound_beta_batched_grad(
                                &refs,
                                seed,
                                &union_refs,
                                if alpha_active { &alpha_pl_refs } else { &[] },
                            )
                        }
                    }
                }
            } else {
                gpu.crown_backward_gpu_resnet_sound_beta_batched_grad(
                    &refs,
                    seed,
                    &union_refs,
                    if alpha_active { &alpha_pl_refs } else { &[] },
                )
            };
            let (results, alpha_grads, wide_gathers) = match pass {
                Ok(v) => v,
                Err(_) if iter == 0 => return None,
                Err(_) => break,
            };
            if results.len() != n_domains {
                if iter == 0 {
                    return None;
                }
                break;
            }
            // Whole-batch iter-0 finiteness (serial:696-706, but for the whole batch).
            if iter == 0
                && results.iter().any(|r| {
                    r.lower_bounds.len() != nsp
                        || r.upper_bounds.len() != nsp
                        || r.lower_bounds
                            .iter()
                            .chain(r.upper_bounds.iter())
                            .any(|v| !v.is_finite())
                })
            {
                return None;
            }
            // #refold-guard increment 2 (the WIDE-β lane — cifar100's scored
            // path): iterate-0's wide-grad bounds are documented equal to the
            // serial single-shot bound (refs use preps segments and inherited
            // preps β at iter 0; "iterate-0 = the serial single-shot bound"),
            // so the same anchor + most-verified-looking spot-check as the
            // single-pass lane applies, under the same oracle-proven 1e-3
            // relative contract. A structural stacking/layout bug corrupts
            // every iterate identically, so guarding iterate-0 covers the
            // realistic failure class at 2 serial folds per BATCH (not per
            // iterate). Any mismatch or serial refusal returns None → the
            // caller's proven serial per-domain loop (downgrade-only).
            if iter == 0 && n_domains > 1 && crate::network::resnet_refold_guard_enabled() {
                for gi in refold_guard_indices(&results) {
                    let p = preps[gi];
                    let serial = gpu.crown_backward_gpu_resnet_sound_beta_refold_oracle(
                        &p.segments,
                        seed,
                        &p.in_lo,
                        &p.in_hi,
                        &p.beta_signed,
                        &p.frontier_abs,
                        &p.node_abs,
                    );
                    let ok = serial
                        .as_ref()
                        .is_ok_and(|s| refold_rows_match(&results[gi], s));
                    if probe {
                        // Execution evidence for A/Bs (probe-gated; the alarm
                        // below is always audible regardless).
                        eprintln!(
                            "[refold-guard] wide-beta checked domain {gi}/{n_domains} ok={ok}"
                        );
                    }
                    if !ok {
                        // Always audible: this is a soundness alarm.
                        eprintln!(
                            "[refold-guard] wide-beta batch REJECTED: domain {gi}/{n_domains} failed iterate-0 serial re-fold (serial_ok={}); falling back to the serial loop",
                            serial.is_ok(),
                        );
                        return None;
                    }
                }
            }
            if let (Some(collector), Some(coeff)) =
                (facet_collector.as_mut(), trajectory_coeff.as_ref())
            {
                if !collector.capture(coeff, &results, beta_opt.thresholds, beta_opt.row_verified) {
                    tracing::debug!("Hydra FacetBank rejected malformed trajectory coefficients");
                    facet_collector = None;
                }
            }
            // Per-domain best-merge / critical / gradient / Adam / convergence.
            for d in 0..n_domains {
                let ds = &mut dstate[d];
                if ds.done && ds.best_lo.is_some() {
                    continue;
                }
                let (lo_d, hi_d) = (&results[d].lower_bounds, &results[d].upper_bounds);
                if lo_d.len() != nsp || hi_d.len() != nsp {
                    ds.done = true;
                    continue;
                }
                ds.iters_run = iter + 1;
                match (&mut ds.best_lo, &mut ds.best_hi) {
                    (Some(bl), Some(bh)) => {
                        for s in 0..nsp {
                            if lo_d[s].is_finite() && lo_d[s] > bl[s] {
                                bl[s] = lo_d[s];
                            }
                            if hi_d[s].is_finite() && hi_d[s] < bh[s] {
                                bh[s] = hi_d[s];
                            }
                        }
                    }
                    _ => {
                        ds.best_lo = Some(lo_d.clone());
                        ds.best_hi = Some(hi_d.clone());
                    }
                }
                // #hard-six tail-iters: iterations BEYOND the base budget are
                // reserved for tail-eligible domains; everyone else freezes
                // here (best iterate kept; this extra iterate was merged
                // above — a free sound tightening). Never fires unless
                // NY_MO_GPU_BETA_ITERS_TAIL extended the loop.
                if iter >= base_iterations && !tail_eligible[d] {
                    ds.done = true;
                    continue;
                }
                if !ds.opt_eligible {
                    ds.done = true; // frozen: iterate-0 bound is final
                    continue;
                }
                let row_verified = &beta_opt.row_verified[d];
                let mut critical: Option<usize> = None;
                let mut critical_margin = f32::INFINITY;
                let mut min_margin = f32::INFINITY;
                for s in 0..nsp {
                    if row_verified.get(s).copied().unwrap_or(false) {
                        continue;
                    }
                    let m = lo_d[s] - beta_opt.thresholds[s];
                    let m = if m.is_nan() { f32::NEG_INFINITY } else { m };
                    min_margin = min_margin.min(m);
                    if m < critical_margin {
                        critical_margin = m;
                        critical = Some(s);
                    }
                }
                let Some(critical) = critical else {
                    ds.done = true;
                    continue;
                };
                ds.crit_row = Some(critical);
                ds.crit_margin = critical_margin;
                // BARRIER-1 sound f64 lineage recovery (dark NY_F64_LINEAGE_RECOVER=1).
                // Re-fold the critical row in f64 with the SAME sound certified-error
                // accounting (u=2⁻⁵³ ⇒ penalty ~1e-9) and merge max into best_lo. Both
                // gpu_lb and the f64 fold are valid lower bounds ⇒ max is SOUND. Uses
                // the domain's current segments (α baked in) + this batch's beta.
                if f64_recover && (iter == 0 || f64_recover_every) {
                    let segd: &[ny_core::GpuResnetSegment] = if alpha_active {
                        &seg_store[d]
                    } else {
                        &preps[d].segments
                    };
                    let od = seed.current_dim;
                    if let Some(rowseed) = seed.lower_a.get(critical * od..(critical + 1) * od) {
                        if let Some(sl) = wide_alpha_true::sound_f64_lower_bound(
                            segd,
                            rowseed,
                            &beta_signed_all[d],
                            &preps[d].in_lo,
                            &preps[d].in_hi,
                            None,
                        ) {
                            if let Some(bl) = ds.best_lo.as_mut() {
                                if sl.is_finite() && sl > bl[critical] {
                                    bl[critical] = sl;
                                }
                            }
                        }
                        // #mn-head-facet: additionally re-fold the SAME critical row
                        // through each HEAD coupling-facet variant and `max`-merge
                        // (INTERSECT). Every fold is a valid Lagrangian LB (facet is
                        // a superset half-space, β ≥ 0, certified-outward), so this
                        // can only RAISE best_lo[critical] — never above the true
                        // margin. The baseline (no-fold) bound above is one max arm.
                        if let Some(folds) = head_folds.as_ref() {
                            let ta = folds[0].target_act;
                            // Per-domain fail-closed: only fold when THIS domain's
                            // fold order still places the head ReLU at `target_act`.
                            if preps[d].relu_names.get(ta) == Some(&folds[0].relu_name) {
                                for hf in folds {
                                    if let Some(sl) = wide_alpha_true::sound_f64_lower_bound(
                                        segd,
                                        rowseed,
                                        &beta_signed_all[d],
                                        &preps[d].in_lo,
                                        &preps[d].in_hi,
                                        Some(hf),
                                    ) {
                                        if let Some(bl) = ds.best_lo.as_mut() {
                                            if sl.is_finite() && sl > bl[critical] {
                                                bl[critical] = sl;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                // C3 probe (dark, NY_C3_PROBE=1): the incoming pre-relaxation
                // LOWER A at this domain's FIRST-ReLU split columns (fold-order
                // LAST ReLU = the network's first ReLU), critical row, iter 0 —
                // the per-neuron capture the C3 fold decision needs
                // (docs/CERTIFIED_CUT_CROWN_DESIGN.md §C3). Columns exist only
                // when some eligible domain actually split first-ReLU neurons.
                if iter == 0 && crate::beta_crown::bab_cuts::c3_probe::c3_probe_enabled() {
                    let r_last = n_relu.saturating_sub(1);
                    let u_r = union_cols.get(r_last).map_or(0, Vec::len);
                    if let Some(vals) = wide_gathers
                        .get(r_last)
                        .filter(|v| u_r > 0 && v.len() == n_domains * nsp * u_r)
                    {
                        let base = (d * nsp + critical) * u_r;
                        let parts: Vec<String> = union_cols[r_last]
                            .iter()
                            .zip(&vals[base..base + u_r])
                            .map(|(c, a)| format!("{c}:{a:.4e}"))
                            .collect();
                        eprintln!(
                            "[c3-probe] dom={d} crit={critical} margin={critical_margin:.5} L1_A[{}]: {}",
                            preps[d].relu_names[r_last],
                            parts.join(" ")
                        );
                    }
                }
                if min_margin > ds.best_margin {
                    ds.best_margin = min_margin;
                    ds.best_beta = ds.beta.clone();
                    // #hard-six unshared-α: snapshot the α that PRODUCED this
                    // iterate's bound (alpha_opt[d] was stepped at the END of
                    // the previous iteration and baked into seg_store before
                    // this backward — the same timing as ds.beta). The upper
                    // map is re-synced to the stepped lower values so the
                    // persisted state keeps the plain-`Activation` extraction
                    // invariant (`lower == upper`) — a diverged snapshot turns
                    // the child's ReLUs into `ActivationReluDualAlpha`, which
                    // this very lane rejects as unbatchable, demoting every
                    // later batch containing the child to the serial ascent
                    // (see `GraphDomainAlphaState::sync_upper_from_lower`).
                    if unshared {
                        best_alpha[d] = alpha_opt[d].clone().map(|mut st| {
                            st.sync_upper_from_lower();
                            st
                        });
                    }
                    // #gather-score (boxlift charter Inc 4 — DARK,
                    // NY_MO_GATHER_SCORE=1): harvest |A_lower| at the union
                    // split columns for THIS domain's best iterate — sibling
                    // domains' split neurons are exactly the kFSB-grade
                    // branch candidates this domain has not split, and the
                    // gather already materialized their pre-relaxation A
                    // values at the critical row for the β gradient (zero
                    // added backwards). Advisory-only: the cache can only
                    // reorder future branch choices; overwrite-per-best keeps
                    // the tightest iterate's scores.
                    if gather_score::gather_score_mode().is_some() {
                        let mut rows: Vec<gather_score::GatherScoreRow> = Vec::new();
                        for r in 0..n_relu {
                            let u_r = union_cols[r].len();
                            if u_r == 0 {
                                continue;
                            }
                            if let Some(vals) = wide_gathers
                                .get(r)
                                .filter(|v| v.len() == n_domains * nsp * u_r)
                            {
                                let base = (d * nsp + critical) * u_r;
                                for (u, &col) in union_cols[r].iter().enumerate() {
                                    let a = vals[base + u];
                                    if a.is_finite() {
                                        rows.push((preps[d].relu_names[r].clone(), col, a.abs()));
                                    }
                                }
                            }
                        }
                        if let Some(bs) = beta_states[d] {
                            self.gather_score_cache
                                .insert(gather_score::beta_split_fingerprint(bs), rows);
                        }
                    }
                }
                let Some(beta) = ds.beta.as_mut() else {
                    ds.done = true;
                    continue;
                };
                beta.zero_grad();
                let mut max_grad = 0.0f32;
                for (e_idx, mapped) in ds.entry_union.iter().enumerate() {
                    let grad = match mapped {
                        Some((r, upos)) => {
                            let u_r = union_cols[*r].len();
                            match wide_gathers.get(*r) {
                                Some(vals) if vals.len() == n_domains * nsp * u_r => {
                                    let a = vals[(d * nsp + critical) * u_r + *upos];
                                    -beta.entries[e_idx].sign() * a
                                }
                                _ => 0.0,
                            }
                        }
                        None => 0.0,
                    };
                    beta.entries[e_idx].grad = grad;
                    max_grad = ny_core::nan_propagating_max(max_grad, grad.abs());
                }
                beta.gradient_step_adam(&self.config.adaptive_config, iter + 1);
                if max_grad < self.config.beta_tolerance {
                    ds.done = true;
                }
            }
            // #w4 wide α+β ascent: per-domain α Adam step from the wide pass's
            // domain-blocked gradients, then write the stepped slopes into the
            // domain's cloned segments for the NEXT iteration's wide fold. Runs
            // alongside β for domains still active (the last iteration's step is
            // wasted work but harmless — the merged bound only keeps sound
            // improvements). Skipped entirely unless alpha_active.
            if alpha_active && iter + 1 < iterations {
                // INC2 — the TRUE joint α-gradient (docs/BATCHED_BAB_JOINT_ALPHA_GRADIENT.md).
                // The wide GPU pass still returns the (refuted) LOCAL analytic gradient
                // `alpha_grads` (`l_i·Σ_s max(A,0)` — see the gate comment above); INC1's
                // A/B showed it degrades bounds. Here we instead compute, PER DOMAIN, the
                // exact reverse-mode adjoint of the whole fold from the domain's cloned
                // segments (current stepped α baked in) + the shared spec seed + the
                // domain's input box. Non-soundness-critical: it only proposes the next α
                // (clamped to [0,1]); the verdict bound is the sound wide fold every
                // iteration. `NY_WIDE_ALPHA_LOCAL=1` restores the old local rule for A/B;
                // `NY_WIDE_ALPHA_NOBIAS=1` drops the bias channel (the ~0.7× degradation).
                let use_local = std::env::var("NY_WIDE_ALPHA_LOCAL").ok().as_deref() == Some("1");
                // Task #39: the TRUE joint α-gradient is computed ON-DEVICE by default
                // (the coefficient-channel forward fold + reverse-mode adjoint resident
                // on GPU), removing the per-domain CPU re-fold penalty. `NY_WIDE_ALPHA_CPU=1`
                // forces the proven CPU oracle (fallback / differential reference); the GPU
                // path also transparently falls back to the CPU oracle on any Err
                // (unsupported topology / fault) — still the correct gradient, never unsound.
                let use_cpu_joint = std::env::var("NY_WIDE_ALPHA_CPU").ok().as_deref() == Some("1");
                let joint_cfg = ny_core::joint_alpha_grad::JointGradConfig {
                    bias_channel: std::env::var("NY_WIDE_ALPHA_NOBIAS").ok().as_deref()
                        != Some("1"),
                };
                // Per-domain gradient-cost probe (task #39 throughput gate): with
                // NY_WIDE_ALPHA_TIMING=1, time the GPU on-device adjoint AND a throwaway
                // CPU-oracle adjoint on the SAME domain/data in the SAME run (identical
                // load) — the direct measure of whether moving the adjoint on-device
                // removed the per-domain CPU-refold penalty.
                let time_probe = std::env::var("NY_WIDE_ALPHA_TIMING").ok().as_deref() == Some("1");
                // #cifar100 task 11 (dark, NY_WIDE_ALPHA_TRUE=1): the host-side
                // critical-row replay lane — the FD-oracle-validated reference for the
                // on-device joint adjoint above (true_grad_oracle_tests.rs). With the
                // gate on, a failed replay SKIPS the α step for that domain (fail-closed;
                // never the refuted local rule). SOUND regardless: gradients only steer
                // α; the bound is always the sound wide fold, best-merged element-wise.
                // NY_AB_PARITY forces the FD-oracle-validated host replay lane on
                // so the per-spec round-robin (below) has a per-row gradient to
                // follow; otherwise honor NY_WIDE_ALPHA_TRUE as before.
                let true_mode = wide_alpha_true::wide_alpha_true_enabled() || ab_parity;
                // BLOCKER-1 fix (dark NY_WIDE_ALPHA_TRUE_STEP=1): apply the
                // FD-oracle-validated host-replay crit/obj-row gradient as the α
                // step (the direction the fold then consumes), instead of the
                // all-spec-sum joint adjoint. Only meaningful under true_mode
                // (true_grad_cache is populated); default off ⇒ joint_d as before.
                let step_true = true_mode && wide_alpha_true::wide_alpha_true_step_enabled();
                let od = seed.current_dim;
                // #task-11 throttled replay phase: on replay iterations (iter % k == 0)
                // refresh the gradient cache for the selected domains — all active
                // participants, or only the worst-margin one under
                // NY_WIDE_ALPHA_TRUE_DOMS=worst. Replays are independent ⇒ parallel.
                if true_mode && iter % true_every == 0 {
                    // auto_LiRPA per-spec α parity: the α-optimization OBJECTIVE row.
                    // Baseline steers α only toward the worst (`crit_row`) spec, so a
                    // single shared α lands a compromise slack for the other binding
                    // specs. Under NY_AB_PARITY we ROUND-ROBIN the objective over the
                    // active (unverified) specs — spec `active[(replay#) % n_active]`
                    // this replay — so Adam steps α toward each spec's own optimum in
                    // turn; the element-wise `best_lo` merge then captures every spec's
                    // own-α iterate, realizing auto_LiRPA's per-spec decoupling without
                    // a per-spec α tensor. Off ⇒ `obj_rows[d] == crit_row` (identical).
                    let obj_rows: Vec<Option<usize>> = (0..n_domains)
                        .map(|d| {
                            if !ab_parity {
                                return dstate[d].crit_row;
                            }
                            let rv = &beta_opt.row_verified[d];
                            let active: Vec<usize> = (0..nsp)
                                .filter(|&s| !rv.get(s).copied().unwrap_or(false))
                                .collect();
                            if active.is_empty() {
                                dstate[d].crit_row
                            } else {
                                Some(active[(iter / true_every.max(1)) % active.len()])
                            }
                        })
                        .collect();
                    let eligible = |d: usize| {
                        !dstate[d].done && alpha_opt[d].is_some() && obj_rows[d].is_some()
                    };
                    let worst: Option<usize> = if true_worst_only {
                        (0..n_domains).filter(|&d| eligible(d)).min_by(|&a, &b| {
                            dstate[a]
                                .crit_margin
                                .partial_cmp(&dstate[b].crit_margin)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                    } else {
                        None
                    };
                    let selected: Vec<usize> = (0..n_domains)
                        .filter(|&d| eligible(d) && (!true_worst_only || worst == Some(d)))
                        .collect();
                    use rayon::prelude::*;
                    let fresh: Vec<(usize, Option<Vec<Vec<f32>>>)> = selected
                        .par_iter()
                        .map(|&d| {
                            let tg = obj_rows[d].and_then(|obj| {
                                let row = seed.lower_a.get(obj * od..(obj + 1) * od)?;
                                wide_alpha_true::true_alpha_grads_for_row(
                                    &seg_store[d],
                                    row,
                                    &beta_signed_all[d],
                                    &preps[d].in_lo,
                                    &preps[d].in_hi,
                                    preps[d].relu_names.len(),
                                    results[d].lower_bounds.get(obj).copied()?,
                                    probe,
                                )
                            });
                            (d, tg)
                        })
                        .collect();
                    for (d, tg) in fresh {
                        // Fail-closed: a FAILED replay also drops any stale gradient —
                        // the mismatch means the cached direction is no longer trusted.
                        true_grad_cache[d] = tg;
                    }
                }
                let mut alpha_max_grad = 0.0f32;
                let mut alpha_doms = 0usize;
                let mut joint_used = 0usize;
                let mut joint_gpu_used = 0usize;
                let mut t_gpu_us: u128 = 0;
                let mut t_cpu_us: u128 = 0;
                let mut t_n: usize = 0;
                for d in 0..n_domains {
                    if dstate[d].done {
                        continue;
                    }
                    if alpha_opt[d].is_none() {
                        continue;
                    }
                    let true_grads: Option<&Vec<Vec<f32>>> = if true_mode {
                        match true_grad_cache[d].as_ref() {
                            Some(tg) => Some(tg),
                            // Fail-closed / unselected / not-yet-replayed:
                            // no α step (never the refuted local rule).
                            None => continue,
                        }
                    } else {
                        None
                    };
                    let Some(st) = alpha_opt[d].as_mut() else {
                        continue;
                    };
                    // Per-domain joint adjoint gradient (per-ReLU fold order, len nn each).
                    // Declines (None) only on an unsupported topology → fall back to the
                    // local GPU grad for this domain (still sound).
                    let cpu_joint = || {
                        ny_core::joint_alpha_grad::joint_alpha_gradient(
                            &seg_store[d],
                            &seed.lower_a,
                            &seed.lower_b,
                            nsp,
                            seed.current_dim,
                            &preps[d].in_lo,
                            &preps[d].in_hi,
                            joint_cfg,
                        )
                    };
                    let joint_d = if step_true {
                        // The α step will consume `true_grads` (FD-validated host
                        // crit/obj-row gradient) instead — skip the wasted on-device
                        // joint adjoint entirely for this domain.
                        None
                    } else if use_local {
                        None
                    } else if use_cpu_joint {
                        cpu_joint()
                    } else {
                        // ON-DEVICE joint adjoint (task #39); Err ⇒ CPU oracle fallback.
                        let t0 = std::time::Instant::now();
                        let g = gpu.crown_joint_alpha_gradient_resident(
                            &seg_store[d],
                            &seed.lower_a,
                            nsp,
                            seed.current_dim,
                            &preps[d].in_lo,
                            &preps[d].in_hi,
                        );
                        if time_probe {
                            let gpu_us = t0.elapsed().as_micros();
                            let t1 = std::time::Instant::now();
                            let _cpu = cpu_joint(); // throwaway: same domain, for the ratio
                            t_gpu_us += gpu_us;
                            t_cpu_us += t1.elapsed().as_micros();
                            t_n += 1;
                        }
                        match g {
                            Ok(g) => {
                                joint_gpu_used += 1;
                                Some(g)
                            }
                            Err(_) => cpu_joint(),
                        }
                    };
                    if joint_d.is_some() {
                        joint_used += 1;
                    }
                    st.zero_grad();
                    let mut any = false;
                    for (r, name) in preps[d].relu_names.iter().enumerate() {
                        let nn_r = dstate[d].relu_nn.get(r).copied().unwrap_or(0);
                        // joint grad `joint_d[r][i]` (per-domain, per-ReLU) when available,
                        // else the local GPU grad `alpha_grads[r][d*nn_r+i]`.
                        let local_r = alpha_grads.get(r).filter(|gr| gr.len() == n_domains * nn_r);
                        let mask = &alpha_pl[d][r];
                        for i in 0..nn_r {
                            if mask.get(i).copied().unwrap_or(0.0) != 0.0 {
                                // BLOCKER-1: prefer the FD-validated host-replay
                                // gradient (`true_grads`, crit/obj-row) when the
                                // step gate is on; else the all-spec joint adjoint;
                                // else the local GPU grad (never in true_mode).
                                let g = if step_true {
                                    true_grads
                                        .and_then(|tg| tg.get(r))
                                        .and_then(|v| v.get(i))
                                        .copied()
                                        .unwrap_or(0.0)
                                } else {
                                    match joint_d.as_ref() {
                                        Some(jd) => {
                                            jd.get(r).and_then(|v| v.get(i)).copied().unwrap_or(0.0)
                                        }
                                        None => local_r.map(|gr| gr[d * nn_r + i]).unwrap_or(0.0),
                                    }
                                };
                                // Skip non-finite grads (never poison the Adam moments;
                                // α is [0,1]-clamped regardless, so this is robustness,
                                // not soundness).
                                if g != 0.0 && g.is_finite() {
                                    st.accumulate_grad(name, i, g);
                                    any = true;
                                    alpha_max_grad =
                                        ny_core::nan_propagating_max(alpha_max_grad, g.abs());
                                }
                            }
                        }
                    }
                    if any {
                        // NY_WIDE_ALPHA_LR overrides the ascent lr (Adam is
                        // scale-invariant in the raw gradient, so tuning/probing
                        // must go through the lr, not the gradient).
                        let mut cfg = self.config.adaptive_config.clone();
                        if let Some(lr) = std::env::var("NY_WIDE_ALPHA_LR")
                            .ok()
                            .and_then(|v| v.parse::<f32>().ok())
                        {
                            cfg.alpha_lr = lr;
                        }
                        st.gradient_step_adam(&cfg, iter + 1);
                        write_back_alpha(&mut seg_store[d], &preps[d].relu_names, &alpha_pl[d], st);
                        alpha_doms += 1;
                    }
                }
                if probe && alpha_doms > 0 {
                    eprintln!(
                        "[beta-gpu-batched-wide] alpha iter={iter} doms={alpha_doms} joint_used={joint_used} joint_gpu={joint_gpu_used} max|g|={alpha_max_grad:.3e}"
                    );
                }
                if time_probe && t_n > 0 {
                    eprintln!(
                        "[wide-alpha-timing] iter={iter} n={t_n} gpu_us/dom={} cpu_us/dom={} speedup={:.2}x",
                        t_gpu_us / t_n as u128,
                        t_cpu_us / t_n as u128,
                        (t_cpu_us as f64) / (t_gpu_us.max(1) as f64)
                    );
                }
            }
            if dstate.iter().all(|ds| ds.done) {
                break;
            }
        }

        // Certify convex combinations of the retained SAME-domain/SAME-row
        // trajectory planes.  Each candidate is independently rounded outward by
        // FacetBank, then max-merged with the scalar best-iterate result.  Any bad
        // row or expired deadline is a no-op; upper-bound dominance prevents an
        // inverted enclosure if an upstream anomaly escaped its own firewall.
        if let Some(collector) = facet_collector.take() {
            let captures = collector.captures;
            let retained_rows = collector.rows.len();
            let boxes: Vec<Option<BoundedTensor>> = preps
                .iter()
                .map(|p| {
                    let lo = ndarray::ArrayD::from_shape_vec(
                        ndarray::IxDyn(&[collector.input_dim]),
                        p.in_lo.clone(),
                    )
                    .ok()?;
                    let hi = ndarray::ArrayD::from_shape_vec(
                        ndarray::IxDyn(&[collector.input_dim]),
                        p.in_hi.clone(),
                    )
                    .ok()?;
                    BoundedTensor::new(lo, hi).ok()
                })
                .collect();
            let search = FacetBankSearchConfig {
                dyadic_bits: FACET_BANK_DEFAULT_DYADIC_BITS,
                refinement_rounds: WIDE_FACET_REFINEMENT_ROUNDS,
            };
            let mut searched = 0usize;
            let mut tightened = 0usize;
            let mut max_gain = 0.0f32;
            for ((domain, row), certificates) in collector.rows {
                if self.config.alpha_config.past_deadline() {
                    break;
                }
                if certificates.len() < 2 {
                    continue;
                }
                let Some(input) = boxes.get(domain).and_then(Option::as_ref) else {
                    continue;
                };
                let Ok(bank) = FacetBank::from_certificates(certificates, search) else {
                    continue;
                };
                let Ok(certified) = bank.certify(input) else {
                    continue;
                };
                searched += 1;
                let Some((old, upper)) = dstate.get(domain).and_then(|ds| {
                    ds.best_lo
                        .as_ref()?
                        .get(row)
                        .copied()
                        .zip(ds.best_hi.as_ref()?.get(row).copied())
                }) else {
                    continue;
                };
                let candidate = certified.lower_bound;
                if candidate.is_finite() && candidate > old && candidate <= upper {
                    if let Some(slot) = dstate
                        .get_mut(domain)
                        .and_then(|ds| ds.best_lo.as_mut())
                        .and_then(|lo| lo.get_mut(row))
                    {
                        *slot = candidate;
                        max_gain = max_gain.max(candidate - old);
                        tightened += 1;
                    }
                }
            }
            tracing::info!(
                captures,
                retained_rows,
                searched,
                tightened,
                max_gain,
                "Hydra wide FacetBank trajectory summary"
            );
        }

        // Assemble per-domain (bound, β) exactly as the serial compute_domain loop does.
        let mut out_bounds: Vec<BoundedTensor> = Vec::with_capacity(n_domains);
        let mut out_betas: Vec<Option<GraphBetaState>> = Vec::with_capacity(n_domains);
        let mut n_opt = 0usize;
        for ds in dstate {
            let (lo, hi) = (ds.best_lo?, ds.best_hi?);
            if lo.iter().chain(hi.iter()).any(|v| !v.is_finite()) {
                return None;
            }
            let lower = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[nsp]), lo).ok()?;
            let upper = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[nsp]), hi).ok()?;
            let bt =
                BoundedTensor::new_repaired(lower, upper, ny_tensor::RepairStrategy::Widen).ok()?;
            out_bounds.push(bt);
            // opt-eligible → Some(best-margin β) (serial's opt path); else → None.
            if ds.opt_eligible {
                out_betas.push(ds.best_beta);
                n_opt += 1;
            } else {
                out_betas.push(None);
            }
        }
        if probe {
            let n_alpha_persist = best_alpha.iter().filter(|a| a.is_some()).count();
            // Parity measurement: this batch's WORST achieved margin over all
            // domains×specs (min_{d,s} best_lo[d][s] − threshold[s]). Grep the
            // running max of this across a run for the achieved frontier — the
            // direct OFF/ON signal for the per-spec α round-robin.
            let mut batch_worst = f32::INFINITY;
            for bt in &out_bounds {
                let lo = bt.lower();
                for s in 0..nsp {
                    let m = lo[[s]] - beta_opt.thresholds.get(s).copied().unwrap_or(0.0);
                    if m.is_finite() {
                        batch_worst = batch_worst.min(m);
                    }
                }
            }
            eprintln!(
                "[beta-gpu-batched-wide] SUCCESS n_domains={n_domains} num_specs={nsp} opt={n_opt} alpha_persist={n_alpha_persist} ab_parity={} batch_worst_margin={batch_worst:.5}",
                ab_parity as u8
            );
        }
        Some((out_bounds, out_betas, best_alpha))
    }

    fn propagate_crown_batched_backward_core(
        &self,
        graph: &GraphNetwork,
        n_domains: usize,
        plan: &CrownDispatchPlan,
        bounds_caches: &[HashMap<String, Arc<BoundedTensor>>],
        constrained_inputs: &[BoundedTensor],
        beta_states: &[Option<&GraphBetaState>],
        alpha_states: &[Option<&GraphDomainAlphaState>],
        objective: &[f32],
        engine: &dyn GemmEngine,
        mode: BatchedBackwardMode<'_>,
        mul_binary_alphas: Option<&HashMap<String, Array2<f32>>>,
    ) -> Result<BatchedBackwardResult> {
        // Guard: empty domain batch would panic at bounds_caches[0] (line 124). (#2671)
        // The primary caller guards with ctx.is_empty() but this is a pub(super) fn
        // that could be called by other paths without that guard.
        if n_domains == 0 || bounds_caches.is_empty() {
            return Ok(BatchedBackwardResult {
                results: Vec::new(),
                intermediate_la: None,
                stage_timing: None,
            });
        }

        // Structural invariant: all parallel arrays must match n_domains. (#2824, #2671)
        // Runtime check — debug_assert is compiled out in release builds.
        if bounds_caches.len() != n_domains
            || constrained_inputs.len() != n_domains
            || beta_states.len() != n_domains
            || alpha_states.len() != n_domains
        {
            return Err(NyError::InternalError(format!(
                "propagate_crown_batched_backward_core: parallel array length mismatch \
                 (n_domains={n_domains}): bounds_caches={}, constrained_inputs={}, \
                 beta_states={}, alpha_states={}",
                bounds_caches.len(),
                constrained_inputs.len(),
                beta_states.len(),
                alpha_states.len(),
            )));
        }

        // ===== BACKWARD PASS: Batched CROWN propagation =====
        let output_idx = plan.output_node_idx;
        let output_node = plan.name_of(output_idx);

        // Get output dimension (same for all domains)
        let ibp_output = bounds_caches[0].get(output_node).ok_or_else(|| {
            NyError::InvalidSpec(format!("Output node {} not found", output_node))
        })?;
        let output_dim = ibp_output.len();

        if objective.len() != output_dim {
            return Err(NyError::shape_mismatch(
                vec![objective.len()],
                vec![output_dim],
            ));
        }

        if std::env::var("NY_BETA_GPU_PROBE").ok().as_deref() == Some("1") {
            eprintln!("[driver] core n_domains={n_domains} output_dim={output_dim}");
        }
        // #lsnc-shared-fwd: the shared backward helpers borrow per-domain caches
        // (a slice of refs). This owned-Vec core wraps its maps once.
        let cache_refs: Vec<&HashMap<String, Arc<BoundedTensor>>> = bounds_caches.iter().collect();
        // #unsat-keystone step 4: GPU beta-capable resnet per-domain fast-path (gated,
        // sound, CPU fallback). Replaces the dense node-by-node batched backward — the
        // conv-resnet per-domain wall — with the verified resnet kernel per domain.
        if matches!(mode, BatchedBackwardMode::Standard) {
            if let Some(bounds) = self.try_gpu_beta_batched_resnet(
                graph,
                output_node,
                output_dim,
                objective,
                1,
                n_domains,
                &cache_refs,
                constrained_inputs,
                beta_states,
                alpha_states,
                engine,
                "core",
            ) {
                let results = bounds
                    .into_iter()
                    .zip(bounds_caches.iter())
                    .map(|(output_bounds, cache)| (output_bounds, cache.clone()))
                    .collect();
                return Ok(BatchedBackwardResult {
                    results,
                    intermediate_la: None,
                    stage_timing: None,
                });
            }
        }

        // Initialize LinearBounds for all domains at output node
        let output_shape = vec![1usize];
        let initial_a =
            Array2::from_shape_vec((1, output_dim), objective.to_vec()).map_err(|e| {
                NyError::InvalidSpec(format!("Failed to build objective coefficients: {e}"))
            })?;
        // Phase 4 audit: user-provided objective — validated to catch NaN/Inf specs.
        let initial_lb = LinearBounds::new(
            initial_a.clone(),
            Array1::zeros(1),
            initial_a,
            Array1::zeros(1),
        )?;

        // Track LinearBounds per domain per node
        let mut node_linear_bounds = IndexedPendingLinearBounds::new(plan, n_domains);

        // === Seed initialization: warm-start at branch points or seed at output ===
        //
        // In Standard mode, all domains start at the output node.
        // In WithLaCapture mode, domains with cached lA from their parent's backward
        // pass can skip recomputing layers between output and branch point by seeding
        // at the branch point instead.
        //
        // Reference: alpha-beta-CROWN `initial_As` in backward_bound.py:203-210
        //            Design: designs/2026-02-07-gpu-bab-la-reuse-closure.md (Dir 2b)
        let capture_intermediate = match &mode {
            BatchedBackwardMode::Standard => {
                // Simple case: all domains seed at output
                for domain_idx in 0..n_domains {
                    node_linear_bounds.seed_idx(output_idx, domain_idx, initial_lb.clone())?;
                }
                false
            }
            BatchedBackwardMode::WithLaCapture {
                histories,
                cached_la,
                capture_intermediate,
            } => {
                // Structural invariant: WithLaCapture arrays must match n_domains. (#2824, #2671)
                if histories.len() != n_domains || cached_la.len() != n_domains {
                    return Err(NyError::InternalError(format!(
                        "WithLaCapture: array length mismatch (n_domains={n_domains}): \
                         histories={}, cached_la={}",
                        histories.len(),
                        cached_la.len(),
                    )));
                }

                let mut n_warm_started = 0usize;

                for domain_idx in 0..n_domains {
                    let warm_started = 'warm: {
                        if !self.config.enable_la_warm_start {
                            break 'warm false;
                        }

                        let cache = match cached_la.get(domain_idx).and_then(|c| *c) {
                            Some(c) if !c.is_empty() => c,
                            _ => break 'warm false,
                        };

                        // Find the most recent branch node from the split history.
                        // Uses last_branch_node() which correctly handles mixed ReLU+GenBaB
                        // histories by checking split_count vs genbab_split_ids.
                        let history = histories[domain_idx];
                        let branch_node = match history.last_branch_node() {
                            Some(n) => n,
                            None => break 'warm false, // root domain, no branch point
                        };

                        // Reconstruct full LinearBounds at the branch node
                        let warm_lb = match cache.linear_bounds(branch_node) {
                            Some(lb) => lb,
                            None => break 'warm false, // cache miss at branch point
                        };

                        let branch_idx = match plan.index_of(branch_node) {
                            Some(idx) => idx,
                            None => break 'warm false,
                        };

                        // Seed at the branch node for this domain
                        node_linear_bounds.seed_idx(branch_idx, domain_idx, warm_lb)?;
                        n_warm_started += 1;
                        true
                    };

                    if !warm_started {
                        node_linear_bounds.seed_idx(output_idx, domain_idx, initial_lb.clone())?;
                    }
                }

                // Log warm-start statistics (#1669)
                if n_warm_started > 0 {
                    tracing::debug!(
                        n_warm_started = n_warm_started,
                        n_domains = n_domains,
                        "lA warm-start: {}/{} domains seeded at branch point (skipping layers above)",
                        n_warm_started,
                        n_domains
                    );
                } else {
                    let n_with_cache = cached_la.iter().filter(|c| c.is_some()).count();
                    if n_with_cache > 0 {
                        tracing::debug!(
                            n_with_cache = n_with_cache,
                            n_domains = n_domains,
                            "lA cache present for {}/{} domains but warm-start not applicable \
                             (no branch point match in cache or warm-start disabled)",
                            n_with_cache,
                            n_domains
                        );
                    }
                }

                *capture_intermediate
            }
        };

        // Initialize intermediate lA storage if capturing
        let mut intermediate_per_domain: Option<Vec<HashMap<String, LinearBounds>>> =
            if capture_intermediate {
                Some((0..n_domains).map(|_| HashMap::new()).collect())
            } else {
                None
            };
        let nodes_by_idx = build_nodes_by_idx(graph, plan)?;

        // Backward propagation through nodes in reverse topological order.
        // Dispatches to the shared backward core, with optional intermediate capture.
        for &idx in &plan.reverse_order {
            let node_name = plan.name_of(idx);
            let node = nodes_by_idx[idx];

            let node_lbs = match node_linear_bounds.take_idx(idx) {
                Some(lbs) => lbs,
                None => continue,
            };

            if !node_lbs.iter().any(|lb| lb.is_some()) {
                continue;
            }

            // Capture intermediate lA if enabled (before processing)
            if let Some(ref mut intermediate) = intermediate_per_domain {
                for (domain_idx, lb_opt) in node_lbs.iter().enumerate() {
                    if let Some(lb) = lb_opt {
                        intermediate[domain_idx].insert(node_name.to_string(), lb.clone());
                    }
                }
            }

            backward_core::dispatch_node_backward(
                node_name,
                node,
                node_lbs,
                constrained_inputs,
                &cache_refs,
                beta_states,
                alpha_states,
                &mut node_linear_bounds,
                n_domains,
                constrained_inputs[0].len(),
                engine,
                self.config.alpha_config.deadline, // #3795: thread BaB deadline
                mul_binary_alphas,                 // #4284: thread shared MulBinary alphas
                false, // #cgan-batched-stack: scalar core keeps per-domain loop
            )?;
        }

        let results = Self::concretize_batched_results(
            constrained_inputs,
            bounds_caches,
            &mut node_linear_bounds,
            output_node,
            &output_shape,
            n_domains,
        )?;

        Ok(BatchedBackwardResult {
            results,
            intermediate_la: intermediate_per_domain,
            stage_timing: None, // Timing set by caller (batched_forward_then_backward)
        })
    }

    /// Wildcard CROWN backward dispatch for layer types not handled by
    /// optimized per-type dispatch in `backward_core::dispatch_node_backward`.
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn propagate_wildcard_crown_backward_batched(
        node_name: &str,
        layer: &Layer,
        node_inputs: &[String],
        lb: LinearBounds,
        constrained_input: &BoundedTensor,
        bounds_cache: &HashMap<String, Arc<BoundedTensor>>,
        node_linear_bounds: &mut HashMap<String, Vec<Option<LinearBounds>>>,
        input_accumulated: &mut bool,
        domain_idx: usize,
        n_domains: usize,
    ) -> Result<()> {
        let Some(first_input_name) = node_inputs.first() else {
            return Err(NyError::InvalidSpec(format!(
                "Node {} ({}) has no inputs for CROWN backward propagation",
                node_name,
                layer.layer_type()
            )));
        };

        let requires_pre_activation = layer.requires_pre_activation_bounds();
        if requires_pre_activation && node_inputs.len() != 1 {
            return Err(NyError::InvalidSpec(format!(
                "Node {} ({}) expects exactly 1 input for CROWN backward propagation, got {}",
                node_name,
                layer.layer_type(),
                node_inputs.len()
            )));
        }

        let pre_activation: Option<&BoundedTensor> = if requires_pre_activation {
            if first_input_name == NETWORK_INPUT {
                Some(constrained_input)
            } else {
                Some(
                    bounds_cache
                        .get(first_input_name)
                        .map(|a| a.as_ref())
                        .ok_or_else(|| {
                            NyError::InvalidSpec(format!(
                                "Pre-activation bounds for {} not found",
                                first_input_name
                            ))
                        })?,
                )
            }
        } else {
            None
        };

        match layer.propagate_crown_backward(&lb, pre_activation) {
            Ok(new_lb) => match layer {
                Layer::OpaqueSkip(_) => {
                    for input_name in node_inputs {
                        backward_core::accumulate_crown_bounds_batched(
                            input_name,
                            new_lb.clone(),
                            node_linear_bounds,
                            input_accumulated,
                            domain_idx,
                            n_domains,
                        );
                    }
                    Ok(())
                }
                Layer::SkipMerge(_) => {
                    if node_inputs.len() != 1 {
                        return Err(NyError::InvalidSpec(format!(
                            "SkipMerge node {} expects exactly 1 input, got {}. \
                             Use OpaqueSkip for multi-input skipped ops.",
                            node_name,
                            node_inputs.len()
                        )));
                    }
                    backward_core::accumulate_crown_bounds_batched(
                        first_input_name,
                        new_lb,
                        node_linear_bounds,
                        input_accumulated,
                        domain_idx,
                        n_domains,
                    );
                    Ok(())
                }
                _ => {
                    if node_inputs.len() != 1 {
                        return Err(NyError::InvalidSpec(format!(
                            "Node {} ({}) expects exactly 1 input for CROWN backward propagation, got {}",
                            node_name,
                            layer.layer_type(),
                            node_inputs.len()
                        )));
                    }
                    backward_core::accumulate_crown_bounds_batched(
                        first_input_name,
                        new_lb,
                        node_linear_bounds,
                        input_accumulated,
                        domain_idx,
                        n_domains,
                    );
                    Ok(())
                }
            },
            // #3166: Catch both UnsupportedOp and UnsupportedConfiguration.
            Err(NyError::UnsupportedOp(msg) | NyError::UnsupportedConfiguration(msg)) => {
                // Identity pass-through for UnsupportedOp is unsound: the layer's
                // transformation is skipped entirely, producing incorrect bounds.
                // For multi-input ops, passing same bounds to all inputs also
                // double-counts contributions. Return error to trigger sequential
                // fallback in the BaB loop (#1996, same fix class as #1934).
                Err(NyError::UnsupportedOp(format!(
                    "Batched CROWN backward at node '{}' ({}): {}",
                    node_name,
                    layer.layer_type(),
                    msg,
                )))
            }
            Err(e) => Err(NyError::InvalidSpec(format!(
                "Batched CROWN failed at node '{}' ({}): {}",
                node_name,
                layer.layer_type(),
                e
            ))),
        }
    }

    /// Concretize final linear bounds at network input for each domain.
    ///
    /// Shared finalization step for both standard and lA-capture backward passes.
    fn concretize_batched_results(
        constrained_inputs: &[BoundedTensor],
        bounds_caches: &[HashMap<String, Arc<BoundedTensor>>],
        node_linear_bounds: &mut IndexedPendingLinearBounds,
        output_node: &str,
        output_shape: &[usize],
        n_domains: usize,
    ) -> Result<Vec<DomainCrownResult>> {
        // Structural invariant: parallel arrays must match n_domains. (#2824, #2671)
        if constrained_inputs.len() != n_domains || bounds_caches.len() != n_domains {
            return Err(NyError::InternalError(format!(
                "concretize_batched_results: parallel array length mismatch \
                 (n_domains={n_domains}): constrained_inputs={}, bounds_caches={}",
                constrained_inputs.len(),
                bounds_caches.len(),
            )));
        }

        let mut results: Vec<DomainCrownResult> = Vec::with_capacity(n_domains);

        let input_accumulated = node_linear_bounds.input_accumulated().to_vec();
        let input_bounds_vec = node_linear_bounds.take_network_input();

        for domain_idx in 0..n_domains {
            let cache = &bounds_caches[domain_idx];

            let output_bounds = if input_accumulated[domain_idx] {
                let final_lb = input_bounds_vec
                    .as_ref()
                    .and_then(|v| v[domain_idx].clone())
                    .ok_or_else(|| {
                        NyError::InvalidSpec(format!(
                            "No linear bounds at input for domain {}",
                            domain_idx
                        ))
                    })?;
                // #2239: directed rounding on f64→f32 for soundness.
                let crown_output = final_lb
                    .concretize_sound(&constrained_inputs[domain_idx])
                    .reshape(output_shape)?;

                // #2904: Tighten CROWN backward bounds by intersecting with IBP
                // forward bounds. The single-domain path (propagation.rs:655-712)
                // checks for NaN/inverted CROWN output and falls back to IBP
                // before attempting tightening. The batched path must match.
                // Reference: alpha-beta-CROWN optimized_bounds.py:937-947.
                let crown_has_bad = crown_output
                    .lower()
                    .iter()
                    .chain(crown_output.upper().iter())
                    .any(|v| !v.is_finite());
                let crown_inverted = crown_output
                    .lower()
                    .iter()
                    .zip(crown_output.upper().iter())
                    .any(|(&l, &u)| l > u);

                if let Some(forward_bounds) = cache.get(output_node) {
                    if crown_has_bad || crown_inverted {
                        // CROWN backward produced invalid bounds. Fall back to
                        // forward (IBP) bounds entirely, matching single-domain
                        // path at propagation.rs:655-676.
                        let fwd_has_bad = forward_bounds
                            .lower()
                            .iter()
                            .chain(forward_bounds.upper().iter())
                            .any(|v| !v.is_finite());
                        let fwd_inverted = forward_bounds
                            .lower()
                            .iter()
                            .zip(forward_bounds.upper().iter())
                            .any(|(&l, &u)| l > u);
                        if !fwd_has_bad && !fwd_inverted {
                            tracing::debug!(
                                "Batched CROWN domain {}: falling back to IBP forward bounds \
                                 (crown_non_finite={}, crown_inverted={})",
                                domain_idx,
                                crown_has_bad,
                                crown_inverted
                            );
                            forward_bounds.as_ref().clone()
                        } else {
                            // Both CROWN and forward bounds are bad — return
                            // CROWN output and let downstream NaN guards handle it.
                            crown_output
                        }
                    } else if crown_output.shape() == forward_bounds.shape() {
                        let (tightened, disjoint_count) =
                            crate::network::tighten_crown_with_forward_bounds(
                                &crown_output,
                                forward_bounds,
                            )?;
                        if disjoint_count > 0 {
                            tracing::warn!(
                                "Batched CROWN domain {}: forward-bound tightening has {} disjoint \
                                 intervals (out of {}); used union fallback",
                                domain_idx,
                                disjoint_count,
                                tightened.len()
                            );
                        }
                        tightened
                    } else {
                        crown_output
                    }
                } else {
                    crown_output
                }
            } else {
                // No backward pass reached input - fall back to IBP.
                // (Owned copy of the single output tensor — pre-existing
                // behavior on this fallback; the map clone is Arc-shallow.)
                let output = cache
                    .get(output_node)
                    .ok_or_else(|| {
                        NyError::InvalidSpec(format!("Output node {} not found", output_node))
                    })?
                    .as_ref()
                    .clone();
                results.push((output, cache.clone()));
                continue;
            };

            results.push((output_bounds, cache.clone()));
        }

        Ok(results)
    }

    /// Dense-spec batched backward pass: accepts a multi-row spec matrix instead of
    /// a scalar objective vector.
    ///
    /// The backward traversal is identical to `propagate_crown_batched_backward_core`.
    /// The only differences are:
    /// 1. The initial seed uses the full spec matrix (nrows > 1 allowed).
    /// 2. output_shape is `[spec_matrix.nrows()]` instead of `[1]`.
    /// 3. Concretization preserves per-domain input `LinearBounds`.
    ///
    /// Part of #4116 Packet A: dense-spec result surface.
    ///
    /// # Arguments
    /// * `spec_matrix` - Dense spec matrix with shape `(num_specs, output_dim)`.
    ///   Each row is one objective. A single-row matrix is equivalent to the scalar
    ///   objective path.
    #[allow(clippy::too_many_arguments)]
    fn propagate_crown_batched_backward_core_specs(
        &self,
        graph: &GraphNetwork,
        n_domains: usize,
        plan: &CrownDispatchPlan,
        // #lsnc-shared-fwd: borrowed per-domain caches (slice of refs). The
        // input-split lane aliases ONE shared warmup map across every domain, so
        // there is no per-domain node-bounds deep clone in the forward pass.
        bounds_caches: &[&HashMap<String, Arc<BoundedTensor>>],
        constrained_inputs: &[BoundedTensor],
        beta_states: &[Option<&GraphBetaState>],
        alpha_states: &[Option<&GraphDomainAlphaState>],
        spec_matrix: &Array2<f32>,
        engine: &dyn GemmEngine,
        mode: BatchedBackwardMode<'_>,
        mul_binary_alphas: Option<&HashMap<String, Array2<f32>>>,
        beta_opt: Option<&GpuBetaOptSpec<'_>>,
        // #lsnc-skip-node-bounds (S3b): leave `DomainSpecCrownResult.node_bounds`
        // EMPTY instead of deep-cloning the per-domain forward cache. Only the
        // input-split lane (which drops the field unread) sets this.
        skip_node_bounds: bool,
    ) -> Result<BatchedSpecBackwardResult> {
        use context::BatchedSpecBackwardResult;

        if n_domains == 0 || bounds_caches.is_empty() {
            return Ok(BatchedSpecBackwardResult {
                results: Vec::new(),
                intermediate_la: None,
                stage_timing: None,
                optimized_betas: None,
                optimized_alphas: None,
                infeasible_domains: None,
            });
        }

        if bounds_caches.len() != n_domains
            || constrained_inputs.len() != n_domains
            || beta_states.len() != n_domains
            || alpha_states.len() != n_domains
        {
            return Err(NyError::InternalError(format!(
                "propagate_crown_batched_backward_core_specs: parallel array length mismatch \
                 (n_domains={n_domains}): bounds_caches={}, constrained_inputs={}, \
                 beta_states={}, alpha_states={}",
                bounds_caches.len(),
                constrained_inputs.len(),
                beta_states.len(),
                alpha_states.len(),
            )));
        }

        let output_idx = plan.output_node_idx;
        let output_node = plan.name_of(output_idx);

        let ibp_output = bounds_caches[0].get(output_node).ok_or_else(|| {
            NyError::InvalidSpec(format!("Output node {} not found", output_node))
        })?;
        let output_dim = ibp_output.len();
        let num_specs = spec_matrix.nrows();

        if num_specs == 0 {
            return Err(NyError::InvalidSpec(
                "spec_matrix must have at least one row".to_string(),
            ));
        }
        if spec_matrix.ncols() != output_dim {
            return Err(NyError::shape_mismatch(
                vec![spec_matrix.ncols()],
                vec![output_dim],
            ));
        }

        // #interm-refine (dark, `NY_INTERM_REFINE=1`, default OFF = byte-identical):
        // per-subdomain refinement of the LAST ReLU's pre-activation bounds (and with
        // `NY_INTERM_REFINE_LAYERS=2` the second-to-last ReLU's, deepest-first). One
        // extra sound identity-seeded backward from each seed ReLU's INPUT node per
        // domain (truncated resnet stack, same certified-error resident path as the
        // margin backward), intersected with the inherited bounds (both are sound
        // enclosures of the same subdomain — the split clamps on the seed entry are
        // already in the inherited cache via `apply_pre_constraints`). The refined
        // caches then feed (a) the ReLU relaxation of the last ReLU in this call's
        // margin backward (tighter chord), (b) stability detection (segment extraction
        // pins exact slopes), (c) the true-α pre-activation tables, and (d) the
        // children via `node_bounds` (the shadowed slice below is what concretization
        // clones). Runs ONCE per subdomain: this driver bounds each child exactly
        // once. With `NY_INTERM_REFINE_PRUNE=1` the pass also PROVES some subdomains'
        // split-premise sets empty — flagged through `infeasible_domains` so the
        // dense-spec caller can verify them vacuously (the same Err(true) signal the
        // with_constraint infeasibility path uses).
        let refined: Option<interm_refine::IntermRefineOutcome> =
            if matches!(mode, BatchedBackwardMode::Standard)
                && interm_refine::interm_refine_enabled()
            {
                // #lsnc-shared-fwd: the dark interm-refine lane keeps its owned
                // `&[HashMap]` surface (unchanged); materialize the borrowed caches
                // for it here. Only runs under NY_INTERM_REFINE, so the hot path
                // never pays this clone.
                let owned_caches: Vec<HashMap<String, Arc<BoundedTensor>>> =
                    bounds_caches.iter().map(|c| (**c).clone()).collect();
                self.refine_last_relu_interm_bounds(
                    graph,
                    output_node,
                    n_domains,
                    &owned_caches,
                    constrained_inputs,
                    beta_states,
                    alpha_states,
                    engine,
                    spec_matrix,
                )
            } else {
                None
            };
        let (refined_caches, infeasible_domains): (
            Option<Vec<HashMap<String, Arc<BoundedTensor>>>>,
            Option<Vec<bool>>,
        ) = match refined {
            Some(outcome) => {
                let flags = outcome
                    .infeasible
                    .iter()
                    .any(|&b| b)
                    .then_some(outcome.infeasible);
                (Some(outcome.caches), flags)
            }
            None => (None, None),
        };
        // #lsnc-shared-fwd: when the dark interm-refine lane produced refined
        // owned caches, borrow them here; otherwise keep the caller's shared refs.
        let refined_refs: Option<Vec<&HashMap<String, Arc<BoundedTensor>>>> =
            refined_caches.as_ref().map(|v| v.iter().collect());
        let bounds_caches: &[&HashMap<String, Arc<BoundedTensor>>] =
            refined_refs.as_deref().unwrap_or(bounds_caches);

        // #mo-beta-graft (config `mo_beta_graft` / preset `bab.beta_graft`, env
        // `NY_MO_BETA_GRAFT` overrides): on pure conv chains the wide segment
        // lane's base relaxation is 2.4-2.9x LOOSER than this driver's dense
        // node-by-node backward (metaroom 6cnn, measured), so replacing the
        // dense bound with the wide one (NY_BAB_CHAIN_WIDE=1 early return) is a
        // net LOSS — but the wide lane is the only place the per-domain β/α
        // ASCENT runs at GPU speed. The graft keeps both: run the wide ascent
        // to OPTIMIZE the multipliers (forcing pure-chain extraction for that
        // call only), then run the dense backward WITH the ascended β/α folded
        // in, and take the elementwise-tightest of the two sound bounds.
        let graft_active = matches!(mode, BatchedBackwardMode::Standard)
            && self.mo_beta_graft_active()
            && beta_opt.is_some_and(|o| o.eligible.iter().any(|&e| e));
        let mut graft_wide: Option<(
            Vec<BoundedTensor>,
            Vec<Option<GraphBetaState>>,
            Vec<Option<GraphDomainAlphaState>>,
        )> = None;

        // #unsat-keystone step 4: GPU beta resnet fast-path, BATCHED over the N spec rows
        // (one resnet-kernel call per domain, num_specs=N in a single call — the ~Nx
        // speedup vs the dense node-by-node batched backward). cifar100 hits this driver
        // with num_specs=99. Gated, sound (β≥0 dual + sound GPU enclosure), CPU fallback.
        // input_linear=None on this path → BaB uses its fallback split scoring (#4116).
        // Under the graft the wide result is CAPTURED (not returned): the dense
        // backward below still runs and the two bounds are composed at the end.
        if matches!(mode, BatchedBackwardMode::Standard) {
            let seed_rows: Vec<f32> = spec_matrix.iter().copied().collect();
            if let Some((bounds, optimized_betas, optimized_alphas)) = self
                .try_gpu_beta_batched_resnet_opt(
                    graph,
                    output_node,
                    output_dim,
                    &seed_rows,
                    num_specs,
                    n_domains,
                    bounds_caches,
                    constrained_inputs,
                    beta_states,
                    alpha_states,
                    engine,
                    "specs",
                    beta_opt,
                    graft_active,
                )
            {
                if graft_active {
                    graft_wide = Some((bounds, optimized_betas, optimized_alphas));
                } else {
                    // C3-at-Relu_57 ledger probe (dark, `NY_C3_57_PROBE=1`,
                    // stderr-only, read-only): the refined caches, premises, and
                    // OPTIMIZED per-row bounds are all in scope exactly here.
                    if interm_refine::c3_57_probe_enabled() {
                        // #lsnc-shared-fwd: dark probe keeps its owned surface.
                        let owned_caches: Vec<HashMap<String, Arc<BoundedTensor>>> =
                            bounds_caches.iter().map(|c| (**c).clone()).collect();
                        interm_refine::c3_57_probe_dump(
                            graph,
                            output_node,
                            &owned_caches,
                            beta_states,
                            spec_matrix,
                            &bounds,
                        );
                    }
                    let results = bounds
                        .into_iter()
                        .zip(bounds_caches.iter())
                        .map(|(output_bounds, cache)| DomainSpecCrownResult {
                            output_bounds,
                            // #lsnc-skip-node-bounds S3b: unread by input-split.
                            node_bounds: if skip_node_bounds {
                                HashMap::new()
                            } else {
                                (**cache).clone()
                            },
                            input_linear: None,
                        })
                        .collect();
                    // #hard-six unshared-α: surface the persisted α snapshots only
                    // when at least one domain has one (gate on + participated);
                    // `None` keeps every caller byte-identical.
                    let optimized_alphas = optimized_alphas
                        .iter()
                        .any(Option::is_some)
                        .then_some(optimized_alphas);
                    return Ok(BatchedSpecBackwardResult {
                        results,
                        intermediate_la: None,
                        stage_timing: None,
                        optimized_betas: Some(optimized_betas),
                        optimized_alphas,
                        infeasible_domains,
                    });
                }
            }
        }

        // #mo-beta-graft: the MAIN dense pass below keeps the domains'
        // INHERITED β/α untouched — it IS the baseline bound and must never
        // regress. The ascended β* is evaluated through a SEPARATE dense pass
        // after concretization (see the composition block at the end): the
        // first probe measurement showed that folding the wide-ascended β*
        // directly into the only dense pass can LOOSEN it catastrophically
        // (β* is optimized against the 2.4-2.9x looser wide relaxation and
        // overshoots the tight one by ~27 units on many rows), so β* must be
        // an additional composed candidate, never a replacement.

        // Build initial LinearBounds from the full spec matrix.
        // Shape: lower_a/upper_a = (num_specs, output_dim), bias = zeros(num_specs).
        let output_shape = vec![num_specs];
        let initial_lb = LinearBounds::new(
            spec_matrix.clone(),
            Array1::zeros(num_specs),
            spec_matrix.clone(),
            Array1::zeros(num_specs),
        )?;

        // #lsnc-batched-bwd (S3): SoA batched-tensor backward for the clean
        // input-split class — Standard mode, no β states, no cgan domain
        // stacking, no graft capture. The fast lane stores pending linear
        // bounds in contiguous [B, R, W] batch tensors and runs the
        // accumulate/merge path as flat in-place ops under coarse rayon
        // chunks; it is BIT-IDENTICAL to the reference loop below (see
        // batched_bwd.rs module docs) and DECLINES (falls through to the
        // untouched reference) on anything outside the proven class.
        if matches!(mode, BatchedBackwardMode::Standard)
            && graft_wide.is_none()
            && !self.config.input_split_stacked_rebound
            && beta_states.iter().all(Option::is_none)
            && batched_bwd::input_split_batched_bwd_enabled()
        {
            if let Some((input_accumulated, input_bounds_vec)) = batched_bwd::try_backward_soa(
                graph,
                plan,
                n_domains,
                bounds_caches,
                constrained_inputs,
                alpha_states,
                &initial_lb,
                engine,
                self.config.alpha_config.deadline,
                mul_binary_alphas,
            )? {
                let results = Self::concretize_batched_results_specs(
                    graph,
                    spec_matrix,
                    constrained_inputs,
                    bounds_caches,
                    &input_accumulated,
                    input_bounds_vec,
                    output_node,
                    &output_shape,
                    n_domains,
                    skip_node_bounds,
                )?;
                return Ok(BatchedSpecBackwardResult {
                    results,
                    intermediate_la: None,
                    stage_timing: None,
                    optimized_betas: None,
                    optimized_alphas: None,
                    infeasible_domains,
                });
            }
            // Decline → run the byte-identical reference loop below.
        }

        // From here, the backward traversal is identical to the scalar core.
        let mut node_linear_bounds = IndexedPendingLinearBounds::new(plan, n_domains);

        let capture_intermediate = match &mode {
            BatchedBackwardMode::Standard => {
                for domain_idx in 0..n_domains {
                    node_linear_bounds.seed_idx(output_idx, domain_idx, initial_lb.clone())?;
                }
                false
            }
            BatchedBackwardMode::WithLaCapture {
                histories,
                cached_la,
                capture_intermediate,
            } => {
                if histories.len() != n_domains || cached_la.len() != n_domains {
                    return Err(NyError::InternalError(format!(
                        "WithLaCapture: array length mismatch (n_domains={n_domains}): \
                         histories={}, cached_la={}",
                        histories.len(),
                        cached_la.len(),
                    )));
                }

                // For dense-spec, warm-start cache is from a previous single-objective
                // pass and won't match the multi-row seed shape. Fall back to output
                // seeding for all domains in this path.
                let mut n_warm_started = 0usize;

                for domain_idx in 0..n_domains {
                    let warm_started = 'warm: {
                        if !self.config.enable_la_warm_start {
                            break 'warm false;
                        }
                        let cache = match cached_la.get(domain_idx).and_then(|c| *c) {
                            Some(c) if !c.is_empty() => c,
                            _ => break 'warm false,
                        };
                        let history = histories[domain_idx];
                        let branch_node = match history.last_branch_node() {
                            Some(n) => n,
                            None => break 'warm false,
                        };
                        let warm_lb = match cache.linear_bounds(branch_node) {
                            Some(lb) => lb,
                            None => break 'warm false,
                        };
                        // Only reuse if the cached lA matches the current spec shape.
                        // Check both lower and upper to guard against shape mismatch
                        // from from_parts_unchecked cache entries.
                        if warm_lb.lower_a.nrows() != num_specs
                            || warm_lb.upper_a.nrows() != num_specs
                        {
                            break 'warm false;
                        }
                        let branch_idx = match plan.index_of(branch_node) {
                            Some(idx) => idx,
                            None => break 'warm false,
                        };
                        node_linear_bounds.seed_idx(branch_idx, domain_idx, warm_lb)?;
                        n_warm_started += 1;
                        true
                    };
                    if !warm_started {
                        node_linear_bounds.seed_idx(output_idx, domain_idx, initial_lb.clone())?;
                    }
                }

                if n_warm_started > 0 {
                    tracing::debug!(
                        n_warm_started = n_warm_started,
                        n_domains = n_domains,
                        num_specs = num_specs,
                        "dense-spec lA warm-start: {}/{} domains seeded at branch point",
                        n_warm_started,
                        n_domains
                    );
                }

                *capture_intermediate
            }
        };

        let mut intermediate_per_domain: Option<Vec<HashMap<String, LinearBounds>>> =
            if capture_intermediate {
                Some((0..n_domains).map(|_| HashMap::new()).collect())
            } else {
                None
            };
        let nodes_by_idx = build_nodes_by_idx(graph, plan)?;

        // Backward traversal — identical to scalar core.
        for &idx in &plan.reverse_order {
            let node_name = plan.name_of(idx);
            let node = nodes_by_idx[idx];

            let node_lbs = match node_linear_bounds.take_idx(idx) {
                Some(lbs) => lbs,
                None => continue,
            };

            if !node_lbs.iter().any(|lb| lb.is_some()) {
                continue;
            }

            if let Some(ref mut intermediate) = intermediate_per_domain {
                for (domain_idx, lb_opt) in node_lbs.iter().enumerate() {
                    if let Some(lb) = lb_opt {
                        intermediate[domain_idx].insert(node_name.to_string(), lb.clone());
                    }
                }
            }

            backward_core::dispatch_node_backward(
                node_name,
                node,
                node_lbs,
                constrained_inputs,
                bounds_caches,
                beta_states,
                alpha_states,
                &mut node_linear_bounds,
                n_domains,
                constrained_inputs[0].len(),
                engine,
                self.config.alpha_config.deadline,
                mul_binary_alphas, // #4284: thread shared MulBinary alphas
                // #cgan-batched-stack: domain-stack conv/BN backwards across
                // domains (preset-gated; false = historical per-domain loop).
                self.config.input_split_stacked_rebound,
            )?;
        }

        let input_accumulated = node_linear_bounds.input_accumulated().to_vec();
        let input_bounds_vec = node_linear_bounds.take_network_input();
        let mut results = Self::concretize_batched_results_specs(
            graph,
            spec_matrix,
            constrained_inputs,
            bounds_caches,
            &input_accumulated,
            input_bounds_vec,
            output_node,
            &output_shape,
            n_domains,
            skip_node_bounds,
        )?;

        // #mo-beta-graft: THREE-way elementwise-tightest composition of sound
        // bounds on the same spec rows over the same subdomain:
        //   1. the baseline dense bound (inherited β — `results`, unchanged);
        //   2. the dense bound with the wide-ascended β* folded in (an extra
        //      recursive dense pass; β* >= 0 with entries built by
        //      `with_constraint` from THIS domain's split history, values only
        //      moved under a β >= 0 clamp — a valid Lagrangian dual for the
        //      same split constraints the baseline pass enforces via the
        //      sign-clamped `bounds_caches`);
        //   3. the wide ascended bound (the production wide lane's own sound
        //      elementwise-tightest-across-iterates result over the SAME β
        //      entries and caches).
        // Each is a valid enclosure of the same quantity, so per-row
        // `[max(l), min(u)]` is one too and is never looser than the baseline.
        // Any per-row anomaly (non-finite candidate value, inverted
        // intersection from f32 slop) keeps the baseline row. The ascended β*
        // is deliberately NOT returned for child warm-starting: measured on
        // metaroom 6cnn, β* poisons descendants' dense passes (their baseline
        // would inherit it), so it is used strictly within this node.
        if let Some((wide_bounds, graft_betas, _graft_alphas)) = graft_wide {
            // Extra dense pass with β* folded. beta_opt=None ⇒ the recursive
            // call cannot re-enter the graft (graft_active requires it), and
            // the wide fast-path inside it falls through on the conv chain
            // (no chain forcing), leaving exactly one dense traversal.
            let graft_beta_refs: Vec<Option<&GraphBetaState>> = graft_betas
                .iter()
                .zip(beta_states.iter())
                .map(|(g, inherited)| g.as_ref().or(*inherited))
                .collect();
            let folded = if graft_betas.iter().any(Option::is_some) {
                match self.propagate_crown_batched_backward_core_specs(
                    graph,
                    n_domains,
                    plan,
                    bounds_caches,
                    constrained_inputs,
                    &graft_beta_refs,
                    alpha_states,
                    spec_matrix,
                    engine,
                    BatchedBackwardMode::Standard,
                    mul_binary_alphas,
                    None,
                    // #lsnc-skip-node-bounds S3b: the graft composition reads
                    // only `output_bounds` from the folded pass.
                    skip_node_bounds,
                ) {
                    Ok(f) if f.results.len() == n_domains => Some(f.results),
                    Ok(_) => None,
                    Err(e) => {
                        tracing::debug!("graft β*-folded dense pass failed (skipping): {e}");
                        None
                    }
                }
            } else {
                None
            };

            let mut folded_rows = 0usize;
            let mut wide_rows = 0usize;
            let mut max_gain = 0.0f32;
            for (i, result) in results.iter_mut().enumerate() {
                if let Some(folded_results) = folded.as_ref() {
                    if let Some((composed, tightened, _, gain)) = graft_compose_tightest(
                        &result.output_bounds,
                        &folded_results[i].output_bounds,
                    ) {
                        result.output_bounds = composed;
                        folded_rows += tightened;
                        max_gain = max_gain.max(gain);
                    }
                }
                if let Some(wide) = wide_bounds.get(i) {
                    if let Some((composed, tightened, _, gain)) =
                        graft_compose_tightest(&result.output_bounds, wide)
                    {
                        result.output_bounds = composed;
                        wide_rows += tightened;
                        max_gain = max_gain.max(gain);
                    }
                }
            }
            if std::env::var("NY_BETA_GPU_PROBE").ok().as_deref() == Some("1") {
                eprintln!(
                    "[graft] n_domains={} folded_rows={folded_rows} wide_rows={wide_rows} max_lower_gain={max_gain:.5}",
                    results.len(),
                );
            }
            return Ok(BatchedSpecBackwardResult {
                results,
                intermediate_la: intermediate_per_domain,
                stage_timing: None,
                // Children keep their inherited β/α (see above).
                optimized_betas: None,
                optimized_alphas: None,
                infeasible_domains,
            });
        }

        Ok(BatchedSpecBackwardResult {
            results,
            intermediate_la: intermediate_per_domain,
            stage_timing: None,
            optimized_betas: None,
            optimized_alphas: None,
            infeasible_domains,
        })
    }

    /// Per-domain-objective batched CROWN backward core (#4355).
    ///
    /// Like `propagate_crown_batched_backward_core` but each domain gets its own
    /// 1-row objective vector instead of sharing one objective across all domains.
    /// This is the batching primitive for per-disjunct alpha evaluation: N active
    /// disjuncts become N pseudo-domains with the same input/history/beta but
    /// different alpha and objective.
    ///
    /// No warm-start for now — all domains seed at the output node.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::beta_crown::engine::graph) fn propagate_crown_batched_backward_core_per_domain_obj(
        &self,
        graph: &GraphNetwork,
        n_domains: usize,
        plan: &CrownDispatchPlan,
        bounds_caches: &[HashMap<String, Arc<BoundedTensor>>],
        constrained_inputs: &[BoundedTensor],
        beta_states: &[Option<&GraphBetaState>],
        alpha_states: &[Option<&GraphDomainAlphaState>],
        per_domain_objectives: &[Vec<f32>],
        engine: &dyn GemmEngine,
        capture_intermediate: bool,
        mul_binary_alphas: Option<&HashMap<String, Array2<f32>>>,
    ) -> Result<BatchedBackwardResult> {
        if n_domains == 0 || bounds_caches.is_empty() {
            return Ok(BatchedBackwardResult {
                results: Vec::new(),
                intermediate_la: None,
                stage_timing: None,
            });
        }

        if per_domain_objectives.len() != n_domains {
            return Err(NyError::InternalError(format!(
                "per_domain_obj: objectives.len()={} != n_domains={n_domains}",
                per_domain_objectives.len(),
            )));
        }

        if bounds_caches.len() != n_domains
            || constrained_inputs.len() != n_domains
            || beta_states.len() != n_domains
            || alpha_states.len() != n_domains
        {
            return Err(NyError::InternalError(format!(
                "per_domain_obj: parallel array length mismatch (n_domains={n_domains}): \
                 bounds_caches={}, constrained_inputs={}, beta_states={}, alpha_states={}",
                bounds_caches.len(),
                constrained_inputs.len(),
                beta_states.len(),
                alpha_states.len(),
            )));
        }

        let output_idx = plan.output_node_idx;
        let output_node = plan.name_of(output_idx);

        let ibp_output = bounds_caches[0].get(output_node).ok_or_else(|| {
            NyError::InvalidSpec(format!("Output node {} not found", output_node))
        })?;
        let output_dim = ibp_output.len();
        let output_shape = vec![1usize];
        if std::env::var("NY_BETA_GPU_PROBE").ok().as_deref() == Some("1") {
            eprintln!("[driver] per_domain_obj n_domains={n_domains} output_dim={output_dim}");
        }

        // Build per-domain seeds: each domain gets its own 1-row objective.
        let mut per_domain_seeds: Vec<Option<LinearBounds>> = Vec::with_capacity(n_domains);
        for (idx, obj) in per_domain_objectives.iter().enumerate() {
            if obj.len() != output_dim {
                return Err(NyError::shape_mismatch(vec![obj.len()], vec![output_dim]));
            }
            let initial_a = Array2::from_shape_vec((1, output_dim), obj.clone()).map_err(|e| {
                NyError::InvalidSpec(format!(
                    "per_domain_obj: failed to build seed for domain {idx}: {e}"
                ))
            })?;
            per_domain_seeds.push(Some(LinearBounds::new(
                initial_a.clone(),
                Array1::zeros(1),
                initial_a,
                Array1::zeros(1),
            )?));
        }

        // #lsnc-shared-fwd: shared backward helpers borrow per-domain caches.
        let cache_refs: Vec<&HashMap<String, Arc<BoundedTensor>>> = bounds_caches.iter().collect();
        let mut node_linear_bounds = IndexedPendingLinearBounds::new(plan, n_domains);
        for (domain_idx, seed) in per_domain_seeds.into_iter().enumerate() {
            if let Some(bounds) = seed {
                node_linear_bounds.seed_idx(output_idx, domain_idx, bounds)?;
            }
        }

        let mut intermediate_per_domain: Option<Vec<HashMap<String, LinearBounds>>> =
            if capture_intermediate {
                Some((0..n_domains).map(|_| HashMap::new()).collect())
            } else {
                None
            };
        let nodes_by_idx = build_nodes_by_idx(graph, plan)?;

        // Backward traversal — identical to scalar/spec cores.
        for &idx in &plan.reverse_order {
            let node_name = plan.name_of(idx);
            let node = nodes_by_idx[idx];

            let node_lbs = match node_linear_bounds.take_idx(idx) {
                Some(lbs) => lbs,
                None => continue,
            };

            if !node_lbs.iter().any(|lb| lb.is_some()) {
                continue;
            }

            if let Some(ref mut intermediate) = intermediate_per_domain {
                for (domain_idx, lb_opt) in node_lbs.iter().enumerate() {
                    if let Some(lb) = lb_opt {
                        intermediate[domain_idx].insert(node_name.to_string(), lb.clone());
                    }
                }
            }

            backward_core::dispatch_node_backward(
                node_name,
                node,
                node_lbs,
                constrained_inputs,
                &cache_refs,
                beta_states,
                alpha_states,
                &mut node_linear_bounds,
                n_domains,
                constrained_inputs[0].len(),
                engine,
                self.config.alpha_config.deadline,
                mul_binary_alphas,
                false, // #cgan-batched-stack: per-domain-objective core keeps per-domain loop
            )?;
        }

        let results = Self::concretize_batched_results(
            constrained_inputs,
            bounds_caches,
            &mut node_linear_bounds,
            output_node,
            &output_shape,
            n_domains,
        )?;

        Ok(BatchedBackwardResult {
            results,
            intermediate_la: intermediate_per_domain,
            stage_timing: None, // Timing set by caller
        })
    }

    /// Concretize dense-spec results, preserving per-domain input `LinearBounds`.
    ///
    /// Sibling of `concretize_batched_results`. The only difference is that this
    /// method captures the final input `LinearBounds` for each domain instead of
    /// discarding them, and returns `DomainSpecCrownResult` instead of the scalar
    /// `DomainCrownResult` tuple.
    ///
    /// Part of #4116 Packet A Step 2.
    ///
    /// `skip_node_bounds` (#lsnc-skip-node-bounds S3b): when true, the
    /// per-domain `node_bounds` map is left EMPTY instead of deep-cloned —
    /// legal only for callers that drop the field unread (the input-split
    /// lane, `input_split/shared_specs.rs`). Every bound, mask, and
    /// `input_linear` is computed identically either way.
    /// `input_accumulated` / `input_bounds_vec` are the network-input pending
    /// state extracted by the caller from either pending-storage
    /// implementation (`IndexedPendingLinearBounds` on the reference path,
    /// the SoA `BatchPending` on the #lsnc-batched-bwd fast lane) — both
    /// paths share this ONE concretize epilogue, so the verdict-bearing math
    /// is common by construction.
    #[allow(clippy::too_many_arguments)]
    fn concretize_batched_results_specs(
        graph: &GraphNetwork,
        spec_matrix: &Array2<f32>,
        constrained_inputs: &[BoundedTensor],
        // #lsnc-shared-fwd: borrowed per-domain caches (slice of refs).
        bounds_caches: &[&HashMap<String, Arc<BoundedTensor>>],
        input_accumulated: &[bool],
        input_bounds_vec: Option<Vec<Option<LinearBounds>>>,
        output_node: &str,
        output_shape: &[usize],
        n_domains: usize,
        skip_node_bounds: bool,
    ) -> Result<Vec<DomainSpecCrownResult>> {
        if constrained_inputs.len() != n_domains
            || bounds_caches.len() != n_domains
            || input_accumulated.len() != n_domains
        {
            return Err(NyError::InternalError(format!(
                "concretize_batched_results_specs: parallel array length mismatch \
                 (n_domains={n_domains}): constrained_inputs={}, bounds_caches={}, \
                 input_accumulated={}",
                constrained_inputs.len(),
                bounds_caches.len(),
                input_accumulated.len(),
            )));
        }

        let mut results: Vec<DomainSpecCrownResult> = Vec::with_capacity(n_domains);

        for domain_idx in 0..n_domains {
            let cache = bounds_caches[domain_idx];

            if input_accumulated[domain_idx] {
                let final_lb = input_bounds_vec
                    .as_ref()
                    .and_then(|v| v[domain_idx].clone())
                    .ok_or_else(|| {
                        NyError::InvalidSpec(format!(
                            "No linear bounds at input for domain {}",
                            domain_idx
                        ))
                    })?;

                let crown_output = final_lb
                    .concretize_sound(&constrained_inputs[domain_idx])
                    .reshape(output_shape)?;

                let crown_has_bad = crown_output
                    .lower()
                    .iter()
                    .chain(crown_output.upper().iter())
                    .any(|v| !v.is_finite());
                let crown_inverted = crown_output
                    .lower()
                    .iter()
                    .zip(crown_output.upper().iter())
                    .any(|(&l, &u)| l > u);

                let forward_spec_bounds = graph.propagate_crown_with_specs_fallback_ibp(
                    &constrained_inputs[domain_idx],
                    spec_matrix,
                    cache,
                    output_node,
                )?;

                // Track whether CROWN backward produced numerically bad output.
                // If so, input_linear must be None — the linear coefficients from an
                // unstable backward pass cannot be trusted for split scoring. (#4116)
                let mut crown_is_reliable = true;

                if crown_has_bad || crown_inverted {
                    crown_is_reliable = false;
                }
                let output_bounds = crate::network::tighten_crown_output(
                    crown_output,
                    &forward_spec_bounds,
                    "Batched spec CROWN",
                )?;

                // Capture input_linear only when CROWN backward was reliable.
                // Unreliable linear coefficients from NaN/Inf/inverted backward
                // passes must not flow to split scoring. (#4116, self-audit finding 1)
                results.push(DomainSpecCrownResult {
                    output_bounds,
                    // #lsnc-skip-node-bounds S3b: unread by input-split.
                    node_bounds: if skip_node_bounds {
                        HashMap::new()
                    } else {
                        cache.clone()
                    },
                    input_linear: if crown_is_reliable {
                        Some(final_lb)
                    } else {
                        None
                    },
                });
            } else {
                let output = graph.propagate_crown_with_specs_fallback_ibp(
                    &constrained_inputs[domain_idx],
                    spec_matrix,
                    cache,
                    output_node,
                )?;
                results.push(DomainSpecCrownResult {
                    output_bounds: output,
                    // #lsnc-skip-node-bounds S3b: unread by input-split.
                    node_bounds: if skip_node_bounds {
                        HashMap::new()
                    } else {
                        cache.clone()
                    },
                    input_linear: None,
                });
            }
        }

        Ok(results)
    }
}

#[cfg(test)]
mod graft_compose_tests {
    use super::graft_compose_tightest;
    use ndarray::arr1;
    use ny_tensor::BoundedTensor;

    fn bt(lo: &[f32], hi: &[f32]) -> BoundedTensor {
        BoundedTensor::new(
            arr1(lo).into_dyn().into_owned(),
            arr1(hi).into_dyn().into_owned(),
        )
        .expect("bt")
    }

    /// Core soundness shape (#mo-beta-graft): both inputs enclose the same true
    /// per-row value, so `[max(l), min(u)]` still encloses it and is never
    /// looser than either input. Rows where the wide bound is tighter must
    /// tighten; rows where dense is tighter must stay.
    #[test]
    fn graft_compose_takes_elementwise_tightest_mo_graft() {
        // True (unknown) row values: 0.0, 0.5, -1.0 — inside both enclosures.
        let dense = bt(&[-1.0, -0.5, -2.0], &[1.0, 2.0, 0.0]);
        let wide = bt(&[-0.25, -1.5, -3.0], &[3.0, 1.0, -0.5]);
        let (composed, tightened, total, max_gain) =
            graft_compose_tightest(&dense, &wide).expect("compose");
        let f = composed.flatten();
        // Row 0: wide lower tighter, dense upper tighter.
        assert_eq!((f.lower()[[0]], f.upper()[[0]]), (-0.25, 1.0));
        // Row 1: dense lower tighter, wide upper tighter.
        assert_eq!((f.lower()[[1]], f.upper()[[1]]), (-0.5, 1.0));
        // Row 2: dense lower tighter, wide upper tighter.
        assert_eq!((f.lower()[[2]], f.upper()[[2]]), (-2.0, -0.5));
        // Composition still encloses the true values.
        for (i, truth) in [0.0f32, 0.5, -1.0].iter().enumerate() {
            assert!(f.lower()[[i]] <= *truth && *truth <= f.upper()[[i]]);
        }
        assert_eq!(tightened, 3);
        assert_eq!(total, 3);
        assert!((max_gain - 0.75).abs() < 1e-6, "row-0 lower gain 0.75");
    }

    /// A strictly-looser wide row (the measured metaroom depth-1 case: the
    /// wide base relaxation is 2.4-2.9x looser) leaves the dense row exactly
    /// unchanged — the graft can never regress the dense bound.
    /// (Non-finite wide rows are unrepresentable in `BoundedTensor` — the wide
    /// lane refuses them wholesale — but `graft_compose_tightest` still guards
    /// defensively.)
    #[test]
    fn graft_compose_never_regresses_dense_on_looser_wide_mo_graft() {
        let dense = bt(&[-1.0, -1.0], &[1.0, 1.0]);
        let wide = bt(&[-5.5, -0.5], &[5.5, 0.5]);
        let (composed, tightened, _, _) = graft_compose_tightest(&dense, &wide).expect("compose");
        let f = composed.flatten();
        assert_eq!((f.lower()[[0]], f.upper()[[0]]), (-1.0, 1.0));
        assert_eq!((f.lower()[[1]], f.upper()[[1]]), (-0.5, 0.5));
        assert_eq!(tightened, 1);
    }

    /// An inverted intersection (possible only from f32 slop between two sound
    /// enclosures) keeps the dense row — matching what the dense lane alone
    /// would have reported.
    #[test]
    fn graft_compose_keeps_dense_on_inverted_intersection_mo_graft() {
        let dense = bt(&[0.6], &[0.9]);
        let wide = bt(&[0.1], &[0.4]);
        let (composed, tightened, _, _) = graft_compose_tightest(&dense, &wide).expect("compose");
        let f = composed.flatten();
        assert_eq!((f.lower()[[0]], f.upper()[[0]]), (0.6, 0.9));
        assert_eq!(tightened, 0);
    }

    /// Length mismatch refuses composition (caller keeps dense unchanged).
    #[test]
    fn graft_compose_refuses_length_mismatch_mo_graft() {
        let dense = bt(&[0.0, 0.0], &[1.0, 1.0]);
        let wide = bt(&[0.5], &[0.6]);
        assert!(graft_compose_tightest(&dense, &wide).is_none());
    }
}

#[cfg(test)]
mod hydra_trajectory_tests {
    use super::WideFacetCollector;
    use crate::bounds::{FacetBank, FacetBankSearchConfig};
    use ndarray::arr1;
    use ny_core::{GpuCrownResult, GpuResidentCoeffBatched};
    use ny_tensor::BoundedTensor;
    use std::collections::HashMap;

    fn coeff(a: f32) -> GpuResidentCoeffBatched {
        GpuResidentCoeffBatched {
            lower_a: vec![a],
            upper_a: vec![a],
            lower_err: vec![0.0],
            upper_err: vec![0.0],
            lower_b: vec![0.0],
            upper_b: vec![0.0],
            lower_b_err: vec![0.0],
            upper_b_err: vec![0.0],
            dim: 1,
            num_specs: 1,
            num_specs_per_dom: 1,
        }
    }

    /// End-to-end active-row capture for the concrete network objective
    /// `|x| = ReLU(x) + ReLU(-x)`: two sound trajectory facets `x` and `-x`
    /// each give -1 on [-1,1], while their certified mixture gives 0.
    #[test]
    fn wide_trajectory_bank_closes_absolute_value_margin() {
        for i in -100..=100 {
            let x = i as f32 / 100.0;
            let objective = x.max(0.0) + (-x).max(0.0);
            assert!(x <= objective && -x <= objective);
        }

        let mut collector = WideFacetCollector {
            rows: HashMap::new(),
            selected_rows: vec![Vec::new()],
            n_domains: 1,
            specs_per_domain: 1,
            input_dim: 1,
            max_planes: 4,
            rows_per_domain: 1,
            captures: 0,
        };
        let result = GpuCrownResult {
            lower_bounds: vec![-1.0],
            upper_bounds: vec![1.0],
        };
        assert!(collector.capture(
            &coeff(1.0),
            std::slice::from_ref(&result),
            &[0.0],
            &[vec![false]]
        ));
        assert!(collector.capture(&coeff(-1.0), &[result], &[0.0], &[vec![false]]));

        let certificates = collector.rows.remove(&(0, 0)).expect("captured row");
        let bank = FacetBank::from_certificates(certificates, FacetBankSearchConfig::default())
            .expect("valid bank");
        let input =
            BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).expect("box");
        let certified = bank.certify(&input).expect("certify");
        assert!(certified.best_one_hot <= -1.0);
        assert!(certified.lower_bound > -1.0e-6);
        assert!(certified.lower_bound > certified.best_one_hot);
    }

    #[test]
    fn wide_trajectory_bank_rejects_cross_domain_layout() {
        let mut collector = WideFacetCollector {
            rows: HashMap::new(),
            selected_rows: vec![Vec::new(), Vec::new()],
            n_domains: 2,
            specs_per_domain: 1,
            input_dim: 1,
            max_planes: 2,
            rows_per_domain: 1,
            captures: 0,
        };
        let malformed = coeff(1.0); // advertises one row, collector requires two
        let results = [
            GpuCrownResult {
                lower_bounds: vec![-1.0],
                upper_bounds: vec![1.0],
            },
            GpuCrownResult {
                lower_bounds: vec![-1.0],
                upper_bounds: vec![1.0],
            },
        ];
        assert!(!collector.capture(&malformed, &results, &[0.0], &[vec![false], vec![false]]));
        assert!(collector.rows.is_empty());
    }
}
