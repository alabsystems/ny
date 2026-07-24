// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! IMB partition proposal with fail-closed certificate replay.
//!
//! Mirrors the SIGNATURE of `multineuron::root_inject::tighten_root_objective_bounds`
//! so the disjunctive input-split root can call it right after the multineuron
//! reassignment. The decomposed IMB floor is log/proposal data only. With
//! `NY_IMB_WIRE=1`, terminal input boxes are independently replayed against the
//! original full network before a bound may change; without that gate the caller's
//! baseline remains byte-identical. The whole body only runs when [`super::armed`]
//! holds (`NY_IMB=1`, small free-dim box, ConvTranspose generator prefix, not already
//! recursing).
//!
//! # Pipeline (per the validated numpy certifier)
//!
//! 1. **Seam** — resolve the cut node (`NY_IMB_SEAM`, else auto-pick the ReLU whose
//!    ancestors cover every ConvTranspose and that has exactly `NY_IMB_TAIL_RELUS`
//!    (default 2) ReLU descendants → Relu_17 for cGAN). `h(x)` = the seam output.
//! 2. **Tail (p,q)** — build the tail sub-graph (`seam` → NETWORK_INPUT, output
//!    unchanged), alpha-optimize its ReLUs for the objective, run a spec-guided
//!    backward-CROWN to get an input-linear lower functional `p·y+q ≤ Y_o(y)` over
//!    the seam box, and fold the certified coeff error OUTWARD into `q`.
//! 3. **Prefix floor** — build the prefix sub-graph (output = `seam`, kept EXACT)
//!    and certify `min_x[p·h(x)]` by per-leaf backward-CROWN input-split BaB over
//!    the free input dims. `imb_floor = q + min_leaf`.
//! 4. **Log** `[imb] obj=… crown_root=… imb_root=… band_lo=… verified=…`.
//!
//! # Soundness boundary
//!
//! The ordinary decomposed `(p,q)` path and its finite seam samples are
//! proposal/telemetry only. They may choose a useful input partition, but they
//! never authorize a verdict. Before wiring, ny either (a) reconstructs and
//! validates an exact binary cover and independently re-bounds the ORIGINAL
//! full network objective on every leaf, or (b), under the separate exact
//! `NY_IMB_TAIL_CERT_AY=1` gate, exact-proves the original tail objective under
//! each region's certified affine prefix-reachability fact (or uses the legacy
//! residual composition outside that default regional lane). Only those two
//! authority families may construct [`FullObjectiveCertificate`]; every error,
//! unsupported operation, non-finite bound, incomplete cover, or expired
//! deadline leaves the baseline unchanged.
//!
//! `NY_IMB_REPLAY_ONLY=1` plus a strict decimal
//! `NY_IMB_REPLAY_ONLY_LEAF=<index>` bypasses proposal construction and replays
//! one deterministic uniform `NY_IMB_REGION_K` region for diagnostics. The
//! region index equals the terminal-leaf index when the proposal trace reports
//! exactly one terminal per region (as in the sealed row-5 trace). Exact box bits
//! are logged so the identity can be checked. This mode has no certificate or
//! bound-return channel and always leaves caller bounds unchanged.

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::mem::size_of;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ndarray::{Array2, ArrayD, IxDyn};
use ny_core::GemmEngine;
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

use crate::beta_crown::engine::graph::input_split::grouped_semantics::valid_disjunctive_layout;
use crate::beta_crown::engine::graph::input_split::shared::{
    compute_crown_or_ibp_bounds_with_node_bounds, extract_obj_bounds,
};
use crate::beta_crown::engine::graph::input_split::shared_specs::compute_crown_or_ibp_bounds_batched_specs;
use crate::bounds::{AlphaCrownConfig, GraphAlphaState, LinearBounds};
use crate::layers::Layer;
use crate::network::graph_crown_f64_tail::{
    f64_tail_verify, graph_supports_f64_tail, F64TailOutcome,
};
use crate::network::SpecCrownRequest;
use crate::{GraphNetwork, GraphNode, NETWORK_INPUT};

use super::{armed, env_f64, env_usize};

#[cfg(test)]
#[path = "root_inject_tests.rs"]
mod tests;

/// One exact prefix enclosure prepared for a single multi-objective IMB run.
///
/// Keeping the prefix allocation and its anchor together prevents an anchor from
/// being accidentally paired with a separately rebuilt graph. The map is a sound
/// enclosure, not a deadline-bearing proof token; every consumer still uses its
/// own current IMB deadline.
struct PreparedExactPrefix {
    seam: Box<str>,
    prefix: Arc<GraphNetwork>,
    anchor: Arc<HashMap<String, BoundedTensor>>,
}

/// Owned, call-local handles returned by [`ExactPrefixSession`].
///
/// Arc clones deliberately avoid borrowing the mutable session through the
/// region Rayon work or prefix BaB. No reference derived from the session can
/// escape into an [`ImbCandidate`].
struct ExactPrefixUse {
    prefix: Arc<GraphNetwork>,
    anchor: Option<Arc<HashMap<String, BoundedTensor>>>,
}

/// Run-local exact identity for objective-independent prefix work.
///
/// The live borrows keep the source graph and input allocation alive and
/// immutable for the session's lifetime. Pointer equality is still checked at
/// every use: equal Rust lifetimes do not imply that two references identify the
/// same object. This is intentionally stronger than the historical u64 memo key.
struct ExactPrefixSession<'model> {
    source_graph: &'model GraphNetwork,
    root_input: &'model BoundedTensor,
    prepared: Option<PreparedExactPrefix>,
}

impl<'model> ExactPrefixSession<'model> {
    fn new(source_graph: &'model GraphNetwork, root_input: &'model BoundedTensor) -> Self {
        Self {
            source_graph,
            root_input,
            prepared: None,
        }
    }

    fn prepare(
        &mut self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        seam: &str,
        engine: Option<&dyn GemmEngine>,
        deadline: Instant,
    ) -> Option<ExactPrefixUse> {
        self.prepare_with(
            graph,
            input,
            seam,
            deadline,
            build_prefix,
            |prefix, root_input, build_deadline| {
                build_tight_prefix_anchor(prefix, root_input, engine, build_deadline)
            },
        )
    }

    /// Builder-injected core so identity, retry, deadline, and Arc-reuse behavior
    /// can be tested without running a large CROWN collection.
    fn prepare_with<BuildPrefix, BuildAnchor>(
        &mut self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        seam: &str,
        deadline: Instant,
        build_prefix_fn: BuildPrefix,
        build_anchor_fn: BuildAnchor,
    ) -> Option<ExactPrefixUse>
    where
        BuildPrefix: FnOnce(&GraphNetwork, &str) -> Option<GraphNetwork>,
        BuildAnchor: FnOnce(
            &GraphNetwork,
            &BoundedTensor,
            Instant,
        ) -> Option<HashMap<String, BoundedTensor>>,
    {
        if !std::ptr::eq(graph, self.source_graph) || !std::ptr::eq(input, self.root_input) {
            eprintln!("[imb] exact prefix session identity mismatch; skip");
            return None;
        }
        if Instant::now() >= deadline {
            eprintln!("[imb] exact prefix session deadline already expired; skip");
            return None;
        }

        if let Some(prepared) = self.prepared.as_ref() {
            if prepared.seam.as_ref() != seam
                || prepared.prefix.output_name() != seam
                || !prepared.anchor.contains_key(seam)
            {
                eprintln!("[imb] exact prefix session seam/invariant mismatch; skip");
                return None;
            }
            eprintln!(
                "[imb] exact prefix session: RUN-LOCAL HIT ({} nodes)",
                prepared.anchor.len()
            );
            return Some(ExactPrefixUse {
                prefix: Arc::clone(&prepared.prefix),
                anchor: Some(Arc::clone(&prepared.anchor)),
            });
        }

        let prefix = Arc::new(build_prefix_fn(self.source_graph, seam)?);
        if prefix.output_name() != seam {
            eprintln!("[imb] exact prefix builder returned the wrong output; skip");
            return None;
        }
        let anchor = build_anchor_fn(prefix.as_ref(), self.root_input, deadline).map(Arc::new);
        let prepared_use = ExactPrefixUse {
            prefix: Arc::clone(&prefix),
            anchor: anchor.as_ref().map(Arc::clone),
        };

        // A returned map remains a sound enclosure after the build deadline, but
        // exact authority additionally requires the requested seam to be present.
        // Store sound completed work before the current row's final deadline check
        // so a later objective with a fresh sub-budget can reuse it. Never memoize
        // absence/failure: a later row must be allowed to retry.
        if let Some(anchor) = anchor.filter(|map| map.contains_key(seam)) {
            self.prepared = Some(PreparedExactPrefix {
                seam: seam.into(),
                prefix,
                anchor,
            });
        }

        if Instant::now() >= deadline {
            eprintln!("[imb] exact prefix preparation exhausted the current row deadline; skip");
            return None;
        }
        Some(prepared_use)
    }
}

/// IMB partition-proposal entry point.
///
/// A certificate produced either by independent original-objective replay or
/// by an AY+ny-cert tail proof whose conditional prefix fact is backed by a
/// validated exact cover (or by the legacy exact residual composition).
/// Sampling and an ordinary decomposed IMB `(p,q)` cannot construct this
/// authority token.
#[derive(Clone, Copy)]
struct FullObjectiveCertificate {
    /// Certified lower bound on the original full-network objective.
    lower: f32,
    /// Absolute validity deadline. A token consumed at/after this instant is
    /// non-authoritative even if its numerical lower is otherwise valid.
    valid_until: Instant,
}

/// A qualified IMB root-floor candidate produced by [`run_imb_measurement`].
///
/// `imb_floor = q + prefix_floor` is a decomposed proposal floor. It is useful for
/// deciding whether to replay the discovered partition, but it has no verdict
/// authority: only `full_certificate.lower` may enter the bound vector.
#[derive(Clone)]
struct ImbCandidate {
    /// The objective whose lower bound the floor certifies.
    obj_idx: usize,
    /// Decomposed IMB proposal floor (telemetry/trigger only).
    imb_floor: f32,
    /// The objective's verification threshold (band-`lo`).
    threshold: f32,
    /// True iff the `(p,q)` was LOADED (`NY_IMB_LOAD_PQ=1`), bypassing the sound
    /// derivation — such a candidate must NEVER feed a verdict.
    measurement_only: bool,
    /// Verdict authority. `None` unless either the independent full-network
    /// checker re-bounded the original objective over an exact-cover input
    /// partition, or the separately gated exact AY tail proof consumed a sound
    /// prefix fact backed by that cover.
    full_certificate: Option<FullObjectiveCertificate>,
    /// Exhaustive terminal input boxes proposed by the prefix BaB.  These have
    /// no proof authority until an authority seam validates their exact cover
    /// and either re-bounds the original full network or discharges the exact
    /// conditional tail and prefix obligations.
    terminal_boxes: Vec<BoundedTensor>,
    /// Absolute fail-closed deadline inherited from the IMB sub-budget and the
    /// verifier deadline. The authoritative replay may not outlive it.
    recheck_deadline: Instant,
}

/// Prefix-BaB telemetry plus the exhaustive frontier it discovered.
struct PrefixBabResult {
    floor: f32,
    terminal_boxes: Vec<BoundedTensor>,
}

/// Per-region proposal returned to the region-loop collector.
struct RegionProposal {
    floor: f32,
    sampled_slack: f32,
    /// Region-specific tail input coefficient. This is proposal data until its
    /// independently certified prefix lower is installed as the matching
    /// reachability premise in an exact AY proof over the exact region cover.
    p: Vec<f32>,
    /// Sound lower bound on `p·h(x)` over this region's exact prefix frontier.
    prefix_floor: f32,
    terminal_boxes: Vec<BoundedTensor>,
}

/// Pick a genuinely distinct second support direction from the complete
/// regional proposal bank. The score is `1 - |cosine|`, so a negated copy is
/// treated as the same rank-one direction rather than a useful second row.
/// Ties are resolved by proposal index for deterministic sealed runs.
fn farthest_support_index(proposals: &[RegionProposal], region_idx: usize) -> Option<usize> {
    let base = proposals.get(region_idx)?.p.as_slice();
    let base_norm2 = base.iter().try_fold(0.0_f64, |sum, &value| {
        value
            .is_finite()
            .then_some(sum + f64::from(value) * f64::from(value))
    })?;
    if !base_norm2.is_finite() || base_norm2 <= 0.0 {
        return None;
    }
    let mut best: Option<(usize, f64)> = None;
    for (idx, proposal) in proposals.iter().enumerate() {
        if idx == region_idx || proposal.p.len() != base.len() {
            continue;
        }
        let mut dot = 0.0_f64;
        let mut norm2 = 0.0_f64;
        let mut valid = true;
        for (&a, &b) in base.iter().zip(&proposal.p) {
            if !b.is_finite() {
                valid = false;
                break;
            }
            dot += f64::from(a) * f64::from(b);
            norm2 += f64::from(b) * f64::from(b);
        }
        if !valid || !dot.is_finite() || !norm2.is_finite() || norm2 <= 0.0 {
            continue;
        }
        let cosine_abs = (dot / (base_norm2 * norm2).sqrt()).abs().min(1.0);
        let score = 1.0 - cosine_abs;
        if !score.is_finite() {
            continue;
        }
        if best.is_none_or(|(best_idx, best_score)| {
            score > best_score || (score == best_score && idx < best_idx)
        }) {
            best = Some((idx, score));
        }
    }
    // Reject numerically collinear banks. Adding p and -p would keep the
    // projection rank one and does not justify the relational authority path.
    best.and_then(|(idx, score)| (score > 1.0e-10).then_some(idx))
}

fn k2_support_directions(
    proposals: &[RegionProposal],
    region_idx: usize,
) -> Option<(Array2<f32>, usize)> {
    let second_idx = farthest_support_index(proposals, region_idx)?;
    let first = proposals.get(region_idx)?.p.as_slice();
    let second = proposals.get(second_idx)?.p.as_slice();
    if first.is_empty() || first.len() != second.len() {
        return None;
    }
    let mut rows = Vec::with_capacity(2 * first.len());
    rows.extend_from_slice(first);
    rows.extend_from_slice(second);
    let directions =
        Array2::from_shape_vec((super::AY_TAIL_AFFINE_REACHABILITY_ROWS, first.len()), rows)
            .ok()?;
    Some((directions, second_idx))
}

/// One shared-bank prefix backward may consume at most this much wall time.
///
/// The cap is deliberately immutable: it prevents an opt-in relational canary
/// from consuming the full IMB/full-verifier budget before an exact AY worker
/// is even admitted.
const SHARED_INPUT_BANK_BUILD_CAP: Duration = Duration::from_secs(30);
const SHARED_INPUT_AY_PROOF_CAP: Duration = Duration::from_secs(45);
const SHARED_INPUT_EVIDENCE_CANARY_ROWS: usize = 4;
const SHARED_INPUT_EVIDENCE_REGION_COUNT: usize = 16;

fn checked_shared_input_bank_deadline(overall: Instant, now: Instant) -> Option<Instant> {
    (now < overall).then(|| {
        now.checked_add(SHARED_INPUT_BANK_BUILD_CAP)
            .map_or(overall, |cap| cap.min(overall))
    })
}

fn checked_shared_input_proof_deadline(overall: Instant, now: Instant) -> Option<Instant> {
    (now < overall).then(|| {
        now.checked_add(SHARED_INPUT_AY_PROOF_CAP)
            .map_or(overall, |cap| cap.min(overall))
    })
}

/// Deterministically build one requested closed-width proposal basis.
///
/// Modified Gram-Schmidt residual is the diversity score; ties use the lowest
/// proposal index. A negated/collinear proposal has zero residual and cannot
/// inflate the basis. The selected order is bound into every exact request
/// token.
fn shared_support_basis(
    proposals: &[RegionProposal],
    requested_rows: usize,
) -> Option<(Array2<f32>, Vec<usize>)> {
    let first = proposals.first()?;
    let width = first.p.len();
    if width == 0
        || !super::AY_TAIL_SHARED_INPUT_SUPPORT_ROWS.contains(&requested_rows)
        || proposals.len() < requested_rows
        || proposals
            .iter()
            .any(|proposal| proposal.p.len() != width || proposal.p.iter().any(|v| !v.is_finite()))
    {
        return None;
    }

    let mut selected = Vec::with_capacity(requested_rows);
    let mut basis: Vec<Vec<f64>> = Vec::with_capacity(requested_rows);
    while selected.len() < requested_rows {
        let mut best: Option<(usize, f64, Vec<f64>)> = None;
        for (idx, proposal) in proposals.iter().enumerate() {
            if selected.contains(&idx) {
                continue;
            }
            let original: Vec<f64> = proposal.p.iter().copied().map(f64::from).collect();
            let original_norm2 = original.iter().map(|value| value * value).sum::<f64>();
            if !original_norm2.is_finite() || original_norm2 <= 0.0 {
                continue;
            }
            let mut residual = original;
            for _ in 0..2 {
                for unit in &basis {
                    let projection = residual
                        .iter()
                        .zip(unit)
                        .map(|(value, direction)| value * direction)
                        .sum::<f64>();
                    for (value, direction) in residual.iter_mut().zip(unit) {
                        *value -= projection * direction;
                    }
                }
            }
            let residual_norm2 = residual.iter().map(|value| value * value).sum::<f64>();
            let score = residual_norm2 / original_norm2;
            if !score.is_finite() || score <= 1.0e-10 {
                continue;
            }
            if best.as_ref().is_none_or(|(best_idx, best_score, _)| {
                score > *best_score || (score == *best_score && idx < *best_idx)
            }) {
                best = Some((idx, score, residual));
            }
        }
        let Some((idx, _score, residual)) = best else {
            break;
        };
        let norm2 = residual.iter().map(|value| value * value).sum::<f64>();
        let inverse_norm = norm2.sqrt().recip();
        basis.push(
            residual
                .into_iter()
                .map(|value| value * inverse_norm)
                .collect(),
        );
        selected.push(idx);
    }

    if selected.len() != requested_rows {
        return None;
    }
    let mut values = Vec::with_capacity(requested_rows.checked_mul(width)?);
    for &idx in &selected {
        values.extend_from_slice(&proposals.get(idx)?.p);
    }
    let directions = Array2::from_shape_vec((requested_rows, width), values).ok()?;
    Some((directions, selected))
}

/// Dark larger-basis policy exercised by unit tests but not yet production
/// admitted. Evidence currently qualifies only K4 for the strict build slice.
#[cfg(test)]
fn shared_support_bases(proposals: &[RegionProposal]) -> Vec<(Array2<f32>, Vec<usize>)> {
    super::AY_TAIL_SHARED_INPUT_SUPPORT_ROWS
        .iter()
        .copied()
        .filter_map(|rows| shared_support_basis(proposals, rows))
        .collect()
}

/// Derive one root-valid support bank for one global exact tail proof.
///
/// The r5 evidence admits K4 as the production canary. K8/K16 remain closed
/// type/encoder capabilities but are deliberately dark until a measured build
/// proves they fit this same immutable slice. This prevents an oversized first
/// attempt from starving the evidence-backed K4 construction. CROWN coefficient
/// errors are widened over the exact root box, never a narrower region.
#[allow(clippy::too_many_arguments)]
fn prefix_shared_input_reachability_envelope(
    prefix: &GraphNetwork,
    root: &BoundedTensor,
    regions: &[BoundedTensor],
    proposals: &[RegionProposal],
    shared_root_anchor: &HashMap<String, BoundedTensor>,
    engine: Option<&dyn GemmEngine>,
    overall_deadline: Instant,
) -> Option<super::AyTailSharedInputReachabilityEnvelope> {
    if root.has_l2_constraint()
        || regions.len() != SHARED_INPUT_EVIDENCE_REGION_COUNT
        || regions.len() != proposals.len()
    {
        return None;
    }
    let build_started = Instant::now();
    let build_deadline = checked_shared_input_bank_deadline(overall_deadline, build_started)?;
    let (directions, support_indices) =
        shared_support_basis(proposals, SHARED_INPUT_EVIDENCE_CANARY_ROWS)?;
    if Instant::now() >= build_deadline {
        return None;
    }
    let rows = directions.nrows();
    let (bounds, Some(mut linear)) = prefix
        .propagate_crown_with_specs_and_node_bounds_and_linear_and_deadline(
            root,
            &directions,
            engine,
            shared_root_anchor,
            Some(build_deadline),
        )
        .ok()?
    else {
        return None;
    };
    if Instant::now() >= build_deadline
        || bounds.flatten().len() != rows
        || bounds
            .lower()
            .iter()
            .chain(bounds.upper())
            .any(|value| !value.is_finite())
    {
        return None;
    }
    let flat_root = root.flatten();
    let (Some(lower), Some(upper)) = (flat_root.lower().as_slice(), flat_root.upper().as_slice())
    else {
        return None;
    };
    linear.fold_coeff_err_into_bias(lower, upper);
    if linear.has_coeff_err()
        || linear.num_outputs() != rows
        || linear.num_inputs() != flat_root.len()
        || Instant::now() >= build_deadline
    {
        return None;
    }
    let (lower_a, lower_b, upper_a, upper_b) = linear.into_parts();
    let envelope = super::AyTailSharedInputReachabilityEnvelope::from_prefix_crown(
        prefix.output_node.clone(),
        root.clone(),
        root.clone(),
        support_indices,
        directions,
        lower_a,
        lower_b,
        upper_a,
        upper_b,
    )?;
    if Instant::now() >= build_deadline {
        return None;
    }
    eprintln!(
        "[imb] AY-TAIL-CERT shared root bank accepted: supports={} latent={} \
         regions={} payload={}B build={:.3}s",
        envelope.directions().nrows(),
        flat_root.len(),
        regions.len(),
        envelope.bank_bytes(),
        build_started.elapsed().as_secs_f64(),
    );
    Some(envelope)
}

/// Derive the K=2 input-relational support envelope in one batched prefix
/// backward. Every certified coefficient-error matrix is discharged OUTWARD
/// into its matching bias over the exact regional box before the coefficients
/// cross the ny-propagate -> ny-cli authority boundary.
fn prefix_affine_reachability_envelope(
    prefix: &GraphNetwork,
    region: &BoundedTensor,
    directions: Array2<f32>,
    shared_root_anchor: &HashMap<String, BoundedTensor>,
    free_dims: &[usize],
    engine: Option<&dyn GemmEngine>,
    deadline: Instant,
) -> Option<super::AyTailAffineReachabilityEnvelope> {
    if Instant::now() >= deadline
        || region.has_l2_constraint()
        || directions.nrows() != super::AY_TAIL_AFFINE_REACHABILITY_ROWS
    {
        return None;
    }

    // Match the prefix-floor path's strongest sound regional anchor: intersect
    // the shared root CROWN map with the per-region RAF enclosure when the
    // affine+ReLU chain supports it. Either operand alone remains sound.
    let mut relu_sources = HashSet::new();
    if let Ok(exec) = prefix.exec_order() {
        for name in exec {
            if let Some(node) = prefix.nodes.get(name) {
                if matches!(node.layer, Layer::ReLU(_)) {
                    if let Some(source) = node.inputs.first() {
                        if source != NETWORK_INPUT {
                            relu_sources.insert(source.clone());
                        }
                    }
                }
            }
        }
    }
    let regional_anchor = super::raf::raf_forward(prefix, region, free_dims, &relu_sources)
        .map(|raf| intersect_anchor(shared_root_anchor, &raf));
    let node_bounds = regional_anchor.as_ref().unwrap_or(shared_root_anchor);

    let (bounds, linear) = prefix
        .propagate_crown_with_specs_and_node_bounds_and_linear_and_deadline(
            region,
            &directions,
            engine,
            node_bounds,
            Some(deadline),
        )
        .ok()?;
    if bounds.flatten().len() != super::AY_TAIL_AFFINE_REACHABILITY_ROWS
        || bounds
            .lower()
            .iter()
            .chain(bounds.upper())
            .any(|value| !value.is_finite())
    {
        return None;
    }
    let mut linear = linear?;
    let flat = region.flatten();
    let lower = flat.lower().as_slice()?;
    let upper = flat.upper().as_slice()?;
    linear.fold_coeff_err_into_bias(lower, upper);
    if linear.has_coeff_err()
        || linear.num_outputs() != super::AY_TAIL_AFFINE_REACHABILITY_ROWS
        || linear.num_inputs() != flat.len()
    {
        return None;
    }
    let (lower_a, lower_b, upper_a, upper_b) = linear.into_parts();
    super::AyTailAffineReachabilityEnvelope::from_prefix_crown(
        prefix.output_node.clone(),
        region.clone(),
        directions,
        lower_a,
        lower_b,
        upper_a,
        upper_b,
    )
}

/// Return the only lower bound that may feed a verdict-changing IMB wire.
///
/// The sampled `(p,q)` slack is intentionally absent from this decision.  It is
/// useful diagnostics and may guide partition discovery, but finite samples can
/// never establish a universal inequality.  Likewise, the decomposed
/// `imb_floor` is telemetry only until the independent full-network checker
/// returns the module-private certificate token.
fn authoritative_candidate_lower(candidate: &ImbCandidate) -> Option<f32> {
    if candidate.measurement_only {
        return None;
    }
    let certificate = candidate.full_certificate?;
    if Instant::now() >= certificate.valid_until {
        return None;
    }
    let lower = certificate.lower;
    lower.is_finite().then_some(lower)
}

fn checked_duration_from_secs(seconds: f64) -> Option<Duration> {
    if !seconds.is_finite() || seconds < 0.0 {
        return None;
    }
    Duration::try_from_secs_f64(seconds).ok()
}

fn checked_budget_deadline(
    start: Instant,
    budget_seconds: f64,
    overall_deadline: Option<Instant>,
) -> Option<Instant> {
    if budget_seconds <= 0.0 {
        return None;
    }
    let duration = checked_duration_from_secs(budget_seconds)?;
    let budget_deadline = start.checked_add(duration)?;
    Some(overall_deadline.map_or(budget_deadline, |d| d.min(budget_deadline)))
}

#[inline]
fn replay_deadline_open(deadline: Instant) -> bool {
    Instant::now() < deadline
}

fn evaluate_before_deadline<T>(
    deadline: Instant,
    evaluator: impl FnOnce() -> Option<T>,
) -> Option<T> {
    if !replay_deadline_open(deadline) {
        return None;
    }
    let value = evaluator()?;
    replay_deadline_open(deadline).then_some(value)
}

/// Hard cap for a verdict-authoritative partition replay.  This is deliberately
/// not configurable through the environment: an unexpectedly huge proposal is
/// a resource failure and therefore fails closed.
const MAX_FULL_RECHECK_LEAVES: usize = 16_384;

/// Hard cap for the sum of all per-objective exact-cover memberships admitted
/// to one cross-clause batched replay. A repeated leaf in two covers counts
/// twice here even though its original-network forward is deduplicated below.
///
/// This is deliberately not configurable: the batched lane is an optional
/// acceleration, and an unexpectedly large request must fail closed instead of
/// turning certificate construction into an unbounded allocation.
const MAX_BATCHED_FULL_RECHECK_MEMBERSHIPS: usize = 16_384;

/// Hard cap for the dense `unique leaves × replay objectives` result surface.
///
/// The independent leaf and membership caps do not bound their Cartesian
/// product: 8,192 unique leaves plus 8,192 one-leaf covers fit both caps while
/// requesting roughly 67 million row/domain pairs. Keep this immutable because
/// the optional authority lane must never turn an adversarial proposal into an
/// unbounded evaluator allocation.
const MAX_BATCHED_REPLAY_DENSE_CELLS: usize = 1_048_576;

/// Hard cap for the dense original-output objective matrix.
const MAX_BATCHED_REPLAY_SPEC_ELEMENTS: usize = 1_048_576;

/// Conservative incremental host/device staging budget for one replay.
///
/// This is intentionally stricter than the process-wide CROWN allocator guard:
/// the optional lane should decline early and leave the historical verifier
/// path available.
const MAX_BATCHED_REPLAY_ESTIMATED_BYTES: usize = 256 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BatchedReplayResourceShape {
    unique_leaves: usize,
    replay_objectives: usize,
    total_memberships: usize,
    input_elements_per_leaf: usize,
    output_dim: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BatchedReplayResourceEstimate {
    dense_cells: usize,
    spec_elements: usize,
    linear_cells: usize,
    estimated_bytes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BatchedReplayPrevalidationShape {
    total_memberships: usize,
    input_elements_per_leaf: usize,
}

fn f32_carrier_bytes(cells: usize, carriers: usize) -> Option<usize> {
    cells.checked_mul(carriers)?.checked_mul(size_of::<f32>())
}

fn batched_replay_prevalidation_bytes(shape: BatchedReplayPrevalidationShape) -> Option<usize> {
    if shape.total_memberships == 0
        || shape.total_memberships > MAX_BATCHED_FULL_RECHECK_MEMBERSHIPS
        || shape.input_elements_per_leaf == 0
    {
        return None;
    }

    let endpoint_cells = shape
        .total_memberships
        .checked_mul(shape.input_elements_per_leaf)?;
    // Before the exact unique count is known, budget the full membership upper
    // bound. Ten endpoint carriers cover the original root and current leaf
    // lower/upper arrays, their BoundedTensor::flatten temporaries, retained
    // FlatPartitionBox copies, and the current collection or bit-key copies at
    // their peak. The M=1 case determines this factor; larger covers retain
    // fewer per-membership temporaries.
    let endpoint_bytes = f32_carrier_bytes(endpoint_cells, 10)?;
    let membership_bytes = shape.total_memberships.checked_mul(size_of::<usize>())?;
    endpoint_bytes.checked_add(membership_bytes)
}

/// Run exact-cover validation and bit-key indexing only after a conservative
/// preflight based on the membership upper bound, before either operation can
/// copy an attacker-sized input tensor.
fn validate_batched_replay_structure_if_admitted<T>(
    shape: BatchedReplayPrevalidationShape,
    validate_and_index: impl FnOnce() -> Option<T>,
) -> Option<T> {
    let estimated_bytes = batched_replay_prevalidation_bytes(shape)?;
    if estimated_bytes > MAX_BATCHED_REPLAY_ESTIMATED_BYTES {
        eprintln!(
            "[imb] FULL-RECHECK-BATCH rejected prevalidation estimate: \
             memberships={} input_elements_per_leaf={} estimated_bytes={}/{}",
            shape.total_memberships,
            shape.input_elements_per_leaf,
            estimated_bytes,
            MAX_BATCHED_REPLAY_ESTIMATED_BYTES,
        );
        return None;
    }
    validate_and_index()
}

fn batched_replay_resource_estimate(
    shape: BatchedReplayResourceShape,
) -> Option<BatchedReplayResourceEstimate> {
    if shape.unique_leaves == 0
        || shape.replay_objectives == 0
        || shape.total_memberships == 0
        || shape.input_elements_per_leaf == 0
        || shape.output_dim == 0
        || shape.unique_leaves > MAX_FULL_RECHECK_LEAVES
        || shape.total_memberships > MAX_BATCHED_FULL_RECHECK_MEMBERSHIPS
    {
        return None;
    }

    let dense_cells = shape.unique_leaves.checked_mul(shape.replay_objectives)?;
    let spec_elements = shape.replay_objectives.checked_mul(shape.output_dim)?;
    let caller_input_cells = shape
        .total_memberships
        .checked_mul(shape.input_elements_per_leaf)?;
    let unique_input_cells = shape
        .unique_leaves
        .checked_mul(shape.input_elements_per_leaf)?;
    let linear_cells = dense_cells.checked_mul(shape.input_elements_per_leaf)?;

    // Conservative peak-live carriers in the current dense-spec helper:
    // - every caller terminal box remains live even when duplicate boxes are
    //   deduplicated for execution, so count two endpoint carriers for all
    //   memberships, then four more unique-domain carriers for builder clones,
    //   stacked arrays, and per-domain `input_bounds_at` copies;
    // - the original root endpoints remain live beside those terminal boxes;
    // - eight output carriers cover evaluator lower/upper, retained copied
    //   rows, biases, and the current conversion temporaries;
    // - the borrowed caller spec, materialized spec, initial lower/upper
    //   seeds, and two seed matrices for every domain are simultaneously
    //   representable;
    // - each final input LinearBounds may carry lower/upper coefficient and
    //   error matrices; result conversion clones it while the original vector
    //   remains, for eight `objective × input` carriers per domain.
    //
    // This intentionally overestimates overlapping lifetimes. It does not try
    // to replace graph-wide CROWN allocator guards; it prevents this optional
    // lane from admitting a request whose known dense carriers alone are large.
    let caller_input_bytes = f32_carrier_bytes(caller_input_cells, 2)?;
    let unique_input_bytes = f32_carrier_bytes(unique_input_cells, 4)?;
    let root_input_bytes = f32_carrier_bytes(shape.input_elements_per_leaf, 2)?;
    let output_bytes = f32_carrier_bytes(dense_cells, 8)?;
    let spec_seed_carriers = shape.unique_leaves.checked_mul(2)?.checked_add(4)?;
    let spec_bytes = f32_carrier_bytes(spec_elements, spec_seed_carriers)?;
    let linear_bytes = f32_carrier_bytes(linear_cells, 8)?;
    let membership_bytes = shape.total_memberships.checked_mul(size_of::<usize>())?;
    let caller_box_metadata = shape
        .total_memberships
        .checked_mul(size_of::<BoundedTensor>())?;
    let unique_metadata = shape.unique_leaves.checked_mul(
        size_of::<&BoundedTensor>()
            .checked_add(size_of::<BoundedTensor>())?
            .checked_add(size_of::<LinearBounds>())?
            .checked_add(size_of::<Vec<(f32, f32)>>())?,
    )?;
    let objective_metadata = shape.replay_objectives.checked_mul(
        size_of::<&[f32]>()
            .checked_add(size_of::<Vec<usize>>())?
            .checked_add(size_of::<FullObjectiveCertificate>())?,
    )?;
    let estimated_bytes = caller_input_bytes
        .checked_add(unique_input_bytes)?
        .checked_add(root_input_bytes)?
        .checked_add(output_bytes)?
        .checked_add(spec_bytes)?
        .checked_add(linear_bytes)?
        .checked_add(membership_bytes)?
        .checked_add(caller_box_metadata)?
        .checked_add(unique_metadata)?
        .checked_add(objective_metadata)?;

    Some(BatchedReplayResourceEstimate {
        dense_cells,
        spec_elements,
        linear_cells,
        estimated_bytes,
    })
}

/// Invoke `evaluator` only after the immutable dense-cell/spec/byte admission.
///
/// Keeping the call itself behind this seam makes the ordering testable: an
/// oversized request cannot enter either the CPU fallback or a GPU kernel.
fn evaluate_batched_replay_if_admitted<T>(
    shape: BatchedReplayResourceShape,
    evaluator: impl FnOnce() -> Option<T>,
) -> Option<T> {
    let estimate = batched_replay_resource_estimate(shape)?;
    if estimate.dense_cells > MAX_BATCHED_REPLAY_DENSE_CELLS
        || estimate.spec_elements > MAX_BATCHED_REPLAY_SPEC_ELEMENTS
        || estimate.estimated_bytes > MAX_BATCHED_REPLAY_ESTIMATED_BYTES
    {
        eprintln!(
            "[imb] FULL-RECHECK-BATCH rejected resource estimate: \
             unique_leaves={} replay_objectives={} memberships={} \
             dense_cells={}/{} spec_elements={}/{} linear_cells={} \
             estimated_bytes={}/{}",
            shape.unique_leaves,
            shape.replay_objectives,
            shape.total_memberships,
            estimate.dense_cells,
            MAX_BATCHED_REPLAY_DENSE_CELLS,
            estimate.spec_elements,
            MAX_BATCHED_REPLAY_SPEC_ELEMENTS,
            estimate.linear_cells,
            estimate.estimated_bytes,
            MAX_BATCHED_REPLAY_ESTIMATED_BYTES,
        );
        return None;
    }
    evaluator()
}

/// Immutable cap for the materialized uniform region boxes themselves. Prefix
/// frontiers have their separate leaf cap; this prevents an environment-sized
/// `k^free_dims` grid from aborting before either validator can run.
const MAX_IMB_REGION_BOX_BYTES: usize = 256 * 1024 * 1024;
const MAX_IMB_UNIFORM_REGIONS: usize = 256;
// cGAN's R=16 exact grid has measured clearing regions that need 28, 36, and
// 28 prefix leaves.  Keep the ordinary immutable 64-leaf per-region ceiling
// available at that grid size while still bounding the complete retained
// frontier.  The separate byte guard below prices all 1,024 leaves before
// proposal construction, and the authority seam revalidates the actual
// aggregate exact cover before admitting any AY worker.
const MAX_AY_REGION_TOTAL_LEAVES: usize = 1_024;
const MAX_AY_PREFIX_FRONTIER_BYTES: usize = 64 * 1024 * 1024;

fn checked_ay_prefix_frontier_bytes(input_elements: usize, total_leaves: usize) -> Option<usize> {
    if input_elements == 0 || total_leaves == 0 || total_leaves > MAX_AY_REGION_TOTAL_LEAVES {
        return None;
    }
    // Retained leaf endpoints, a full FlatPartitionBox copy, and conservative
    // allowance for the validator's simultaneous flattened/current-root
    // endpoint carriers (eight T×D f32 arrays total).
    let endpoint_bytes = total_leaves
        .checked_mul(input_elements)?
        .checked_mul(8)?
        .checked_mul(size_of::<f32>())?;
    let header_bytes = total_leaves.checked_mul(size_of::<BoundedTensor>())?;
    let bytes = endpoint_bytes.checked_add(header_bytes)?;
    (bytes <= MAX_AY_PREFIX_FRONTIER_BYTES).then_some(bytes)
}

fn checked_imb_region_box_plan(
    input_elements: usize,
    free_dims: usize,
    k: usize,
) -> Option<(usize, usize)> {
    let exponent = u32::try_from(free_dims).ok()?;
    let total = k.checked_pow(exponent)?;
    if input_elements == 0
        || free_dims == 0
        || k == 0
        || total == 0
        || total > MAX_IMB_UNIFORM_REGIONS
    {
        return None;
    }
    let endpoint_bytes = total
        .checked_mul(input_elements)?
        .checked_mul(2)?
        .checked_mul(size_of::<f32>())?;
    let box_headers = total.checked_mul(size_of::<BoundedTensor>())?;
    let edge_cells = free_dims.checked_mul(k.checked_add(1)?)?;
    let edge_bytes = edge_cells
        .checked_mul(size_of::<f32>())?
        .checked_add(free_dims.checked_mul(size_of::<Vec<f32>>())?)?;
    // Account for flattened-root endpoints plus the current region's two
    // endpoint vectors while it is being materialized.
    let transient_bytes = input_elements
        .checked_mul(4)?
        .checked_mul(size_of::<f32>())?;
    let working_bytes = endpoint_bytes
        .checked_add(box_headers)?
        .checked_add(edge_bytes)?
        .checked_add(transient_bytes)?;
    (working_bytes <= MAX_IMB_REGION_BOX_BYTES).then_some((total, working_bytes))
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum ReplayF64Attempt {
    GraphUnsupported,
    Verified { lower: f32 },
    NotVerified { min_gap_f64: f64 },
    Unsupported,
    Rejected,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReplayLeafRoute {
    F64Verified,
    StandardFallback,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ReplayLeafEvaluation {
    lower: Option<f32>,
    route: ReplayLeafRoute,
    f64_attempt: ReplayF64Attempt,
    standard_lower: Option<f32>,
}

#[derive(Default)]
struct ReplayStageTimings {
    f64: Option<Duration>,
    standard: Option<Duration>,
}

#[derive(Clone)]
struct FlatPartitionBox {
    lower: Vec<f32>,
    upper: Vec<f32>,
}

/// Collision-free identity for deduplicating the same bit-exact input leaf
/// across independently validated objective covers.
///
/// A `HashMap` hash collision cannot merge unequal boxes because `Eq` compares
/// every shape and bound bit. Signed zero and distinct NaN payloads remain
/// distinct; NaNs are rejected before a key can become authoritative.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ExactReplayBoxKey {
    shape: Vec<usize>,
    lower_bits: Vec<u32>,
    upper_bits: Vec<u32>,
}

fn exact_replay_box_key(bx: &BoundedTensor) -> Option<ExactReplayBoxKey> {
    // Equal box endpoints are not sufficient identity when one tensor carries
    // an additional L2-ball constraint: choosing the constrained representative
    // could tighten another cover's unconstrained leaf unsoundly. IMB-created
    // split leaves carry no such annotation, so decline rather than invent a
    // second semantic-key format for this optional acceleration.
    if bx.has_l2_constraint() {
        return None;
    }
    let flat = bx.flatten();
    let lower: Vec<f32> = flat.lower().iter().copied().collect();
    let upper: Vec<f32> = flat.upper().iter().copied().collect();
    if lower.is_empty()
        || lower.len() != upper.len()
        || lower
            .iter()
            .zip(&upper)
            .any(|(&lo, &hi)| !lo.is_finite() || !hi.is_finite() || lo > hi)
    {
        return None;
    }
    Some(ExactReplayBoxKey {
        shape: bx.shape().to_vec(),
        lower_bits: lower.into_iter().map(f32::to_bits).collect(),
        upper_bits: upper.into_iter().map(f32::to_bits).collect(),
    })
}

fn index_exact_replay_leaves<'a>(
    terminal_partitions: &[&'a [BoundedTensor]],
    deadline: Instant,
) -> Option<(Vec<&'a BoundedTensor>, Vec<Vec<usize>>)> {
    let mut unique_index: HashMap<ExactReplayBoxKey, usize> = HashMap::new();
    let mut unique_leaves: Vec<&BoundedTensor> = Vec::new();
    let mut cover_leaf_indices: Vec<Vec<usize>> = Vec::with_capacity(terminal_partitions.len());
    for partition in terminal_partitions {
        let mut membership = Vec::with_capacity(partition.len());
        for leaf in *partition {
            if !replay_deadline_open(deadline) {
                return None;
            }
            let key = exact_replay_box_key(leaf)?;
            let index = match unique_index.get(&key).copied() {
                Some(index) => index,
                None => {
                    if unique_leaves.len() >= MAX_FULL_RECHECK_LEAVES {
                        eprintln!("[imb] FULL-RECHECK-BATCH rejected: unique-leaf cap exceeded");
                        return None;
                    }
                    let index = unique_leaves.len();
                    unique_leaves.push(leaf);
                    unique_index.insert(key, index);
                    index
                }
            };
            membership.push(index);
        }
        cover_leaf_indices.push(membership);
    }
    if unique_leaves.is_empty() || !replay_deadline_open(deadline) {
        return None;
    }
    Some((unique_leaves, cover_leaf_indices))
}

fn flatten_finite_box(
    bx: &BoundedTensor,
    expected_shape: &[usize],
    deadline: Instant,
) -> Result<FlatPartitionBox, &'static str> {
    if !replay_deadline_open(deadline) {
        return Err("partition validation deadline expired");
    }
    if bx.shape() != expected_shape {
        return Err("partition leaf shape mismatch");
    }
    let flat = bx.flatten();
    let lower: Vec<f32> = flat.lower().iter().copied().collect();
    let upper: Vec<f32> = flat.upper().iter().copied().collect();
    if !replay_deadline_open(deadline) {
        return Err("partition validation deadline expired");
    }
    if lower.len() != upper.len() || lower.is_empty() {
        return Err("partition box has invalid dimensionality");
    }
    if lower
        .iter()
        .zip(&upper)
        .any(|(&l, &u)| !l.is_finite() || !u.is_finite() || l > u)
    {
        return Err("partition box has non-finite or inverted bounds");
    }
    Ok(FlatPartitionBox { lower, upper })
}

fn same_flat_box(bx: &FlatPartitionBox, lower: &[f32], upper: &[f32]) -> bool {
    bx.lower == lower && bx.upper == upper
}

/// Reconstruct and validate a guillotine binary split tree from terminal boxes.
///
/// Every IMB frontier is produced by exact axis-aligned binary splits (the outer
/// uniform region grid is also guillotine).  Therefore a valid frontier always
/// has an interior coordinate that no terminal box crosses.  Recursing on that
/// coordinate proves exact coverage without relying on floating-point volumes:
/// each child root shares the identical stored split coordinate, gaps have no
/// matching leaf, and overlaps necessarily cross every candidate separator.
fn validate_partition_subtree(
    region_lower: &[f32],
    region_upper: &[f32],
    boxes: &[FlatPartitionBox],
    indices: &[usize],
    deadline: Instant,
) -> Result<(), &'static str> {
    if !replay_deadline_open(deadline) {
        return Err("partition validation deadline expired");
    }
    if indices.is_empty() {
        return Err("partition coverage gap");
    }
    if indices.len() == 1 {
        return same_flat_box(&boxes[indices[0]], region_lower, region_upper)
            .then_some(())
            .ok_or("terminal box does not equal its reconstructed region");
    }

    for dim in 0..region_lower.len() {
        if !replay_deadline_open(deadline) {
            return Err("partition validation deadline expired");
        }
        if region_lower[dim] >= region_upper[dim] {
            continue;
        }
        let mut cuts = Vec::new();
        for &idx in indices {
            if !replay_deadline_open(deadline) {
                return Err("partition validation deadline expired");
            }
            for cut in [boxes[idx].lower[dim], boxes[idx].upper[dim]] {
                if cut > region_lower[dim] && cut < region_upper[dim] {
                    cuts.push(cut);
                }
            }
        }
        cuts.sort_by(f32::total_cmp);
        if !replay_deadline_open(deadline) {
            return Err("partition validation deadline expired");
        }
        cuts.dedup_by(|a, b| *a == *b);
        if !replay_deadline_open(deadline) {
            return Err("partition validation deadline expired");
        }

        for cut in cuts {
            if !replay_deadline_open(deadline) {
                return Err("partition validation deadline expired");
            }
            let mut left = Vec::new();
            let mut right = Vec::new();
            let mut crossed = false;
            for &idx in indices {
                if !replay_deadline_open(deadline) {
                    return Err("partition validation deadline expired");
                }
                let bx = &boxes[idx];
                if bx.upper[dim] <= cut {
                    left.push(idx);
                } else if bx.lower[dim] >= cut {
                    right.push(idx);
                } else {
                    crossed = true;
                    break;
                }
            }
            if crossed || left.is_empty() || right.is_empty() {
                continue;
            }

            let mut left_upper = region_upper.to_vec();
            left_upper[dim] = cut;
            let mut right_lower = region_lower.to_vec();
            right_lower[dim] = cut;
            validate_partition_subtree(region_lower, &left_upper, boxes, &left, deadline)?;
            return validate_partition_subtree(&right_lower, region_upper, boxes, &right, deadline);
        }
    }
    Err("terminal boxes are not an exact binary split cover")
}

fn validate_binary_partition_cover(
    root: &BoundedTensor,
    terminal_boxes: &[BoundedTensor],
    deadline: Instant,
) -> Result<(), &'static str> {
    if !replay_deadline_open(deadline) {
        return Err("partition validation deadline expired");
    }
    if terminal_boxes.is_empty() {
        return Err("partition has no terminal boxes");
    }
    if terminal_boxes.len() > MAX_FULL_RECHECK_LEAVES {
        return Err("partition exceeds the full-recheck leaf cap");
    }
    let shape = root.shape().to_vec();
    let root = flatten_finite_box(root, &shape, deadline)?;
    let mut boxes = Vec::with_capacity(terminal_boxes.len());
    for leaf in terminal_boxes {
        if !replay_deadline_open(deadline) {
            return Err("partition validation deadline expired");
        }
        let leaf = flatten_finite_box(leaf, &shape, deadline)?;
        if leaf
            .lower
            .iter()
            .zip(&leaf.upper)
            .zip(root.lower.iter().zip(&root.upper))
            .any(|((&l, &u), (&rl, &ru))| l < rl || u > ru)
        {
            return Err("partition leaf escapes the root box");
        }
        boxes.push(leaf);
    }
    let indices: Vec<usize> = (0..boxes.len()).collect();
    validate_partition_subtree(&root.lower, &root.upper, &boxes, &indices, deadline)
}

fn directed_f64_lower_to_f32(lower: f64) -> Option<f32> {
    if !lower.is_finite() {
        return None;
    }
    let cast = if lower > f64::from(f32::MAX) {
        f32::MAX
    } else {
        next_down_f32(lower as f32)
    };
    cast.is_finite().then_some(cast)
}

/// Smallest finite binary32 residual threshold whose directed-down sum with
/// `prefix_floor` is still strictly above the original property threshold.
///
/// Compute against the exact composition operation itself. Algebraically
/// subtracting two f32 values and adding one ULP is insufficient around
/// cancellation, subnormals, and exponent transitions because the authority
/// seam performs a binary64 sum followed by a directed binary32 cast.
fn minimum_q_for_strict_composition(prefix_floor: f32, threshold: f32) -> Option<f32> {
    if !prefix_floor.is_finite() || !threshold.is_finite() {
        return None;
    }
    const SIGN: u32 = 1 << 31;
    const MIN_FINITE_KEY: u32 = !(-f32::MAX).to_bits();
    const MAX_FINITE_KEY: u32 = f32::MAX.to_bits() | SIGN;

    let from_order_key = |key: u32| {
        let bits = if key & SIGN == 0 { !key } else { key & !SIGN };
        f32::from_bits(bits)
    };
    let clears = |q: f32| {
        directed_f64_lower_to_f32(f64::from(q) + f64::from(prefix_floor))
            .is_some_and(|lower| lower > threshold)
    };

    // IEEE binary32 values are contiguous under this sign-adjusted key:
    // [-MAX, ..., -0, +0, ..., +MAX]. The composition predicate is monotone in
    // q, so a fixed 32-step lower-bound search finds the true smallest finite
    // threshold without assumptions about cancellation or ULP ratios.
    if !clears(f32::MAX) {
        return None;
    }
    let mut low = MIN_FINITE_KEY;
    let mut high = MAX_FINITE_KEY;
    while low < high {
        let mid = low + (high - low) / 2;
        if clears(from_order_key(mid)) {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    let q = from_order_key(low);
    q.is_finite().then_some(q)
}

/// Mint verdict authority by composing an independently replayed AY tail
/// inequality with NY's sound prefix lower over an exact-cover frontier.
///
/// Both inputs are binary32 lower bounds. Their exact real sum is therefore
/// representable in binary64; the final cast is directed downward.  The
/// terminal-box validation is repeated at this authority seam so a truncated,
/// duplicated, overlapping, or foreign prefix frontier cannot cross it.
fn certify_ay_tail_prefix_composition(
    root: &BoundedTensor,
    terminal_boxes: &[BoundedTensor],
    certified_q: f32,
    prefix_floor: f32,
    threshold: f32,
    deadline: Instant,
) -> Option<FullObjectiveCertificate> {
    if !replay_deadline_open(deadline)
        || !certified_q.is_finite()
        || !prefix_floor.is_finite()
        || !threshold.is_finite()
    {
        return None;
    }
    if let Err(reason) = validate_binary_partition_cover(root, terminal_boxes, deadline) {
        eprintln!("[imb] AY-TAIL-CERT partition rejected: {reason}");
        return None;
    }
    let lower = directed_f64_lower_to_f32(f64::from(certified_q) + f64::from(prefix_floor))?;
    if !replay_deadline_open(deadline) || lower <= threshold {
        return None;
    }
    eprintln!(
        "[imb] AY-TAIL-CERT COMPOSED: leaves={} q={certified_q:.9} \
         prefix_lower={prefix_floor:.9} original_lower={lower:.9} > \
         threshold={threshold:.9}",
        terminal_boxes.len()
    );
    Some(FullObjectiveCertificate {
        lower,
        valid_until: deadline,
    })
}

/// Prove every region's independently certified AY residual and compose it
/// with that region's own prefix lower.
///
/// The callback is deliberately invoked sequentially. The exact AY bridge has
/// process-global worker admission, and a parallel fan-out would turn a valid
/// all-region proof into an admission race. Region proposal discovery remains
/// independently parallel above this authority seam.
fn certify_ay_region_partition_with<F>(
    root: &BoundedTensor,
    regions: &[BoundedTensor],
    proposals: &[RegionProposal],
    threshold: f32,
    deadline: Instant,
    mut certify_residual: F,
) -> Option<f32>
where
    F: FnMut(usize, &BoundedTensor, &[f32], f32, Instant) -> Option<f32>,
{
    if !replay_deadline_open(deadline)
        || !threshold.is_finite()
        || root.has_l2_constraint()
        || regions.is_empty()
        || regions.len() != proposals.len()
    {
        return None;
    }
    if let Err(reason) = validate_binary_partition_cover(root, regions, deadline) {
        eprintln!("[imb] AY-TAIL-CERT region grid rejected: {reason}");
        return None;
    }

    // Complete every structural/numeric preflight before admitting any exact
    // worker. A bad late region must not waste successful early-region proofs.
    let mut total_leaves = 0usize;
    for (region_idx, (region, proposal)) in regions.iter().zip(proposals).enumerate() {
        if !replay_deadline_open(deadline)
            || region.has_l2_constraint()
            || proposal.p.is_empty()
            || proposal.p.iter().any(|v| !v.is_finite())
            || !proposal.prefix_floor.is_finite()
            || proposal
                .terminal_boxes
                .iter()
                .any(BoundedTensor::has_l2_constraint)
        {
            return None;
        }
        total_leaves = total_leaves.checked_add(proposal.terminal_boxes.len())?;
        if total_leaves > MAX_AY_REGION_TOTAL_LEAVES {
            eprintln!(
                "[imb] AY-TAIL-CERT global region frontier exceeds exact-lane leaf cap \
                 ({total_leaves} > {MAX_AY_REGION_TOTAL_LEAVES}) before solver admission"
            );
            return None;
        }
        if let Err(reason) =
            validate_binary_partition_cover(region, &proposal.terminal_boxes, deadline)
        {
            eprintln!("[imb] AY-TAIL-CERT region {region_idx} prefix cover rejected: {reason}");
            return None;
        }
    }

    let mut global_lower = f32::INFINITY;
    for (region_idx, (region, proposal)) in regions.iter().zip(proposals).enumerate() {
        let required_q = minimum_q_for_strict_composition(proposal.prefix_floor, threshold)?;
        let certified_q = certify_residual(region_idx, region, &proposal.p, required_q, deadline)?;
        if !certified_q.is_finite() || certified_q < required_q {
            return None;
        }
        let local = certify_ay_tail_prefix_composition(
            region,
            &proposal.terminal_boxes,
            certified_q,
            proposal.prefix_floor,
            threshold,
            deadline,
        )?;
        global_lower = global_lower.min(local.lower);
    }

    if !replay_deadline_open(deadline) || !global_lower.is_finite() || global_lower <= threshold {
        None
    } else {
        Some(global_lower)
    }
}

/// Prove every region's original tail objective under that region's certified
/// affine prefix-reachability fact.
///
/// For a region proposal, prefix BaB has established `p · h(x) >=
/// prefix_floor` over an independently validated exact cover. The callback
/// proves the original objective over the root seam box intersected with that
/// one affine premise. This retains the correlation needed by the proposal
/// without introducing 2,048 independent regional seam coordinates or
/// composing two unrelated minima.
///
/// The exact worker callbacks remain sequential and every region/prefix cover
/// is structurally preflighted before the first callback is admitted.
fn certify_ay_region_reachability_partition_with<F>(
    root: &BoundedTensor,
    regions: &[BoundedTensor],
    proposals: &[RegionProposal],
    threshold: f32,
    deadline: Instant,
    mut certify_original: F,
) -> Option<f32>
where
    F: FnMut(usize, &BoundedTensor, &[f32], f32, f32, Instant) -> Option<f32>,
{
    let requested_lower = next_up_f32(threshold);
    if !replay_deadline_open(deadline)
        || !threshold.is_finite()
        || !requested_lower.is_finite()
        || requested_lower <= threshold
        || root.has_l2_constraint()
        || regions.is_empty()
        || regions.len() != proposals.len()
    {
        return None;
    }
    if let Err(reason) = validate_binary_partition_cover(root, regions, deadline) {
        eprintln!("[imb] AY-TAIL-CERT reachability region grid rejected: {reason}");
        return None;
    }

    let mut total_leaves = 0usize;
    for (region_idx, (region, proposal)) in regions.iter().zip(proposals).enumerate() {
        if !replay_deadline_open(deadline)
            || region.has_l2_constraint()
            || proposal.p.is_empty()
            || proposal.p.iter().any(|v| !v.is_finite())
            || !proposal.prefix_floor.is_finite()
            || proposal
                .terminal_boxes
                .iter()
                .any(BoundedTensor::has_l2_constraint)
        {
            return None;
        }
        total_leaves = total_leaves.checked_add(proposal.terminal_boxes.len())?;
        if total_leaves > MAX_AY_REGION_TOTAL_LEAVES {
            eprintln!(
                "[imb] AY-TAIL-CERT reachability frontier exceeds exact-lane leaf cap \
                 ({total_leaves} > {MAX_AY_REGION_TOTAL_LEAVES}) before solver admission"
            );
            return None;
        }
        if let Err(reason) =
            validate_binary_partition_cover(region, &proposal.terminal_boxes, deadline)
        {
            eprintln!(
                "[imb] AY-TAIL-CERT reachability region {region_idx} prefix cover rejected: \
                 {reason}"
            );
            return None;
        }
    }

    let mut global_lower = f32::INFINITY;
    for (region_idx, (region, proposal)) in regions.iter().zip(proposals).enumerate() {
        let certified_lower = certify_original(
            region_idx,
            region,
            &proposal.p,
            proposal.prefix_floor,
            requested_lower,
            deadline,
        )?;
        if certified_lower.to_bits() != requested_lower.to_bits() || !replay_deadline_open(deadline)
        {
            return None;
        }
        global_lower = global_lower.min(certified_lower);
    }

    if !global_lower.is_finite() || global_lower <= threshold {
        None
    } else {
        Some(global_lower)
    }
}

fn same_bounded_tensor_bits(left: &BoundedTensor, right: &BoundedTensor) -> bool {
    left.shape() == right.shape()
        && left.has_l2_constraint() == right.has_l2_constraint()
        && left
            .lower()
            .iter()
            .zip(right.lower())
            .all(|(&a, &b)| a.to_bits() == b.to_bits())
        && left
            .upper()
            .iter()
            .zip(right.upper())
            .all(|(&a, &b)| a.to_bits() == b.to_bits())
}

/// Prove every region's original tail objective under that region's certified
/// K=2 shared-input prefix envelope.
///
/// All envelopes and the exact region cover are preflighted before the first
/// exact worker is admitted. Rows from different regions remain disjunctive:
/// the callback receives exactly one envelope/model at a time.
fn certify_ay_region_affine_reachability_partition_with<F>(
    root: &BoundedTensor,
    regions: &[BoundedTensor],
    envelopes: &[super::AyTailAffineReachabilityEnvelope],
    threshold: f32,
    deadline: Instant,
    mut certify_original: F,
) -> Option<f32>
where
    F: FnMut(usize, &super::AyTailAffineReachabilityEnvelope, f32, Instant) -> Option<f32>,
{
    let requested_lower = next_up_f32(threshold);
    if !replay_deadline_open(deadline)
        || !threshold.is_finite()
        || !requested_lower.is_finite()
        || requested_lower <= threshold
        || root.has_l2_constraint()
        || regions.is_empty()
        || regions.len() != envelopes.len()
    {
        return None;
    }
    if let Err(reason) = validate_binary_partition_cover(root, regions, deadline) {
        eprintln!("[imb] AY-TAIL-CERT affine region grid rejected: {reason}");
        return None;
    }
    for (region_idx, (region, envelope)) in regions.iter().zip(envelopes).enumerate() {
        if !replay_deadline_open(deadline)
            || region.has_l2_constraint()
            || !same_bounded_tensor_bits(region, envelope.region_input())
            || envelope.directions().nrows() != super::AY_TAIL_AFFINE_REACHABILITY_ROWS
            || envelope.directions().iter().any(|value| !value.is_finite())
        {
            eprintln!(
                "[imb] AY-TAIL-CERT affine region {} envelope preflight rejected",
                region_idx + 1
            );
            return None;
        }
    }

    let mut global_lower = f32::INFINITY;
    for (region_idx, envelope) in envelopes.iter().enumerate() {
        let certified_lower = certify_original(region_idx, envelope, requested_lower, deadline)?;
        if certified_lower.to_bits() != requested_lower.to_bits() || !replay_deadline_open(deadline)
        {
            return None;
        }
        global_lower = global_lower.min(certified_lower);
    }
    (global_lower.is_finite() && global_lower > threshold).then_some(global_lower)
}

/// Prove the global original tail objective once under one root-input bank.
///
/// The R-region grid remains exact-checked proposal provenance: its complete
/// cover is checked, and every request-bit-bound support index must name one of
/// those proposals. It is not expanded into R exact tail models. The one
/// envelope instead uses `region_input == certified_root_input == root`, so one
/// exact Graph-MIP/AY call proves the global objective directly. Because every
/// would-be regional model uses this identical root-certified bank and the
/// regions exactly cover root, their feasible-set union is precisely this one
/// root-input model. The entire callback, including encoding, shares one
/// immutable 45-second slice.
fn certify_ay_shared_input_root_with<F>(
    root: &BoundedTensor,
    regions: &[BoundedTensor],
    envelope: &super::AyTailSharedInputReachabilityEnvelope,
    threshold: f32,
    overall_deadline: Instant,
    certify_original: F,
) -> Option<f32>
where
    F: FnOnce(&super::AyTailSharedInputReachabilityEnvelope, f32, Instant) -> Option<f32>,
{
    let requested_lower = next_up_f32(threshold);
    if !replay_deadline_open(overall_deadline)
        || !threshold.is_finite()
        || !requested_lower.is_finite()
        || requested_lower <= threshold
        || root.has_l2_constraint()
        || regions.len() != SHARED_INPUT_EVIDENCE_REGION_COUNT
        || !same_bounded_tensor_bits(root, envelope.certified_root_input())
        || !same_bounded_tensor_bits(root, envelope.region_input())
        || envelope.directions().nrows() != SHARED_INPUT_EVIDENCE_CANARY_ROWS
        || envelope.support_indices().len() != SHARED_INPUT_EVIDENCE_CANARY_ROWS
        || envelope
            .support_indices()
            .iter()
            .any(|&support_idx| support_idx >= regions.len())
        || envelope.bank_bytes() > super::AY_TAIL_SHARED_INPUT_MAX_BANK_BYTES
    {
        return None;
    }
    if let Err(reason) = validate_binary_partition_cover(root, regions, overall_deadline) {
        eprintln!("[imb] AY-TAIL-CERT shared region grid rejected: {reason}");
        return None;
    }
    if regions.iter().any(|region| region.has_l2_constraint()) {
        return None;
    }

    let proof_deadline = checked_shared_input_proof_deadline(overall_deadline, Instant::now())?;
    let certified_lower = certify_original(envelope, requested_lower, proof_deadline)?;
    if certified_lower.to_bits() != requested_lower.to_bits()
        || !replay_deadline_open(proof_deadline)
    {
        return None;
    }
    (certified_lower.is_finite() && certified_lower > threshold).then_some(certified_lower)
}

/// Mint the single all-regions authority token only after the concatenated
/// prefix frontier is independently shown to be an exact cover of the original
/// input. `region_lower` is already the minimum of every local certified
/// original-objective lower (reachability mode), or the directed-down minimum
/// of every residual + prefix composition in the explicit legacy A/B mode, or
/// the one globally certified root lower in shared-input mode.
fn certify_ay_region_global_composition(
    root: &BoundedTensor,
    terminal_boxes: &[BoundedTensor],
    region_lower: f32,
    threshold: f32,
    deadline: Instant,
) -> Option<FullObjectiveCertificate> {
    if !replay_deadline_open(deadline)
        || !region_lower.is_finite()
        || !threshold.is_finite()
        || region_lower <= threshold
    {
        return None;
    }
    if let Err(reason) = validate_binary_partition_cover(root, terminal_boxes, deadline) {
        eprintln!("[imb] AY-TAIL-CERT global region frontier rejected: {reason}");
        return None;
    }
    if !replay_deadline_open(deadline) {
        return None;
    }
    eprintln!(
        "[imb] AY-TAIL-CERT REGION COMPOSED: regions_lower={region_lower:.9} \
         leaves={} > threshold={threshold:.9}",
        terminal_boxes.len()
    );
    Some(FullObjectiveCertificate {
        lower: region_lower,
        valid_until: deadline,
    })
}

fn standard_no_imb_objective_lower(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    spec: &Array2<f32>,
    engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
) -> Option<f32> {
    // This is the low-level standard CROWN/IBP evaluator, not a verifier entry
    // point: it has no IMB hook and mutates no environment state.  Fresh
    // per-leaf IBP references tighten nonlinear relaxations while every failure
    // returns a conservative/non-finite result that is rejected below.
    let (bounds, _linear) = compute_crown_or_ibp_bounds_with_node_bounds(
        graph, input, spec, engine, None, None, None, None, deadline, None, true,
    )
    .ok()?;
    let lower = extract_obj_bounds(&bounds, 1).ok()?.first()?.0;
    lower.is_finite().then_some(lower)
}

/// Evaluate one original-network replay leaf through the exact production route.
///
/// The optional timing sink is used only by the explicitly armed replay-only
/// diagnostic. Production replay passes `None`, so its logging and timing
/// behavior remain unchanged.
fn evaluate_original_objective_leaf(
    graph: &GraphNetwork,
    leaf: &BoundedTensor,
    spec: &Array2<f32>,
    thresholds: &[f32],
    clause_sizes: &[usize],
    engine: Option<&dyn GemmEngine>,
    deadline: Instant,
    supports_f64: bool,
    mut timings: Option<&mut ReplayStageTimings>,
) -> ReplayLeafEvaluation {
    let f64_started = (supports_f64 && timings.is_some()).then(Instant::now);
    let f64_raw = if supports_f64 {
        evaluate_before_deadline(deadline, || {
            Some(f64_tail_verify(
                graph,
                leaf,
                spec,
                thresholds,
                clause_sizes,
                None,
                None,
                engine,
                Some(deadline),
            ))
        })
    } else {
        None
    };
    if let (Some(timings), Some(started)) = (timings.as_mut(), f64_started) {
        timings.f64 = Some(started.elapsed());
    }

    let (f64_attempt, f64_lower) = if !supports_f64 {
        (ReplayF64Attempt::GraphUnsupported, None)
    } else {
        match f64_raw {
            Some(F64TailOutcome::Verified { row_lowers }) => {
                match row_lowers
                    .first()
                    .copied()
                    .and_then(directed_f64_lower_to_f32)
                {
                    Some(lower) => (ReplayF64Attempt::Verified { lower }, Some(lower)),
                    None => (ReplayF64Attempt::Rejected, None),
                }
            }
            Some(F64TailOutcome::NotVerified { min_gap_f64 }) => {
                (ReplayF64Attempt::NotVerified { min_gap_f64 }, None)
            }
            Some(F64TailOutcome::Unsupported) => (ReplayF64Attempt::Unsupported, None),
            None => (ReplayF64Attempt::Rejected, None),
        }
    };

    if let Some(lower) = f64_lower {
        return ReplayLeafEvaluation {
            lower: Some(lower),
            route: ReplayLeafRoute::F64Verified,
            f64_attempt,
            standard_lower: None,
        };
    }

    let standard_started = timings.as_ref().map(|_| Instant::now());
    let standard_lower = evaluate_before_deadline(deadline, || {
        standard_no_imb_objective_lower(graph, leaf, spec, engine, Some(deadline))
    });
    if let (Some(timings), Some(started)) = (timings.as_mut(), standard_started) {
        timings.standard = Some(started.elapsed());
    }
    ReplayLeafEvaluation {
        lower: standard_lower,
        route: ReplayLeafRoute::StandardFallback,
        f64_attempt,
        standard_lower,
    }
}

fn strict_decimal_usize(raw: &str) -> Option<usize> {
    (!raw.is_empty() && raw.bytes().all(|b| b.is_ascii_digit()))
        .then(|| raw.parse::<usize>().ok())
        .flatten()
}

fn replay_only_leaf_request(
    gate: Option<&str>,
    leaf: Option<&str>,
) -> Result<Option<usize>, &'static str> {
    match gate {
        None | Some("0") => return Ok(None),
        Some("1") => {}
        Some(_) => return Err("NY_IMB_REPLAY_ONLY must be exactly '0' or '1'"),
    }
    let raw = leaf.ok_or("NY_IMB_REPLAY_ONLY_LEAF is required")?;
    strict_decimal_usize(raw)
        .map(Some)
        .ok_or("NY_IMB_REPLAY_ONLY_LEAF must be an in-range decimal usize")
}

fn replay_only_objective(raw: Option<&str>, objective_count: usize) -> Result<usize, &'static str> {
    let obj_idx = match raw {
        Some(raw) => strict_decimal_usize(raw)
            .ok_or("NY_IMB_OBJ must be an in-range decimal usize in replay-only mode")?,
        None => 0,
    };
    (obj_idx < objective_count)
        .then_some(obj_idx)
        .ok_or("NY_IMB_OBJ is outside the objective vector")
}

const MAX_REPLAY_ONLY_LOGGED_SCALARS: usize = 64;

#[cfg(test)]
static REPLAY_ONLY_EVALUATIONS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

fn exact_box_bits(bx: &BoundedTensor) -> Option<String> {
    let lower = bx.lower();
    let upper = bx.upper();
    if lower.len() != upper.len() || lower.len() > MAX_REPLAY_ONLY_LOGGED_SCALARS {
        return None;
    }
    Some(
        lower
            .iter()
            .zip(upper.iter())
            .enumerate()
            .map(|(idx, (&lo, &hi))| {
                format!("{idx}:0x{:08x}..0x{:08x}", lo.to_bits(), hi.to_bits())
            })
            .collect::<Vec<_>>()
            .join(","),
    )
}

fn exact_split_bounds_bits(bx: &BoundedTensor, split_dims: &[usize]) -> Option<String> {
    let lower = bx.lower().as_slice()?;
    let upper = bx.upper().as_slice()?;
    split_dims
        .iter()
        .map(|&idx| {
            Some(format!(
                "{idx}:0x{:08x}..0x{:08x}",
                lower.get(idx)?.to_bits(),
                upper.get(idx)?.to_bits()
            ))
        })
        .collect::<Option<Vec<_>>>()
        .map(|parts| parts.join(","))
}

fn stable_box_fingerprint(bx: &BoundedTensor) -> u64 {
    const OFFSET: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;
    let mut hash = OFFSET;
    let mut mix = |word: u64| {
        for byte in word.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(PRIME);
        }
    };
    for &dim in bx.shape() {
        mix(dim as u64);
    }
    for &value in bx.lower() {
        mix(u64::from(value.to_bits()));
    }
    for &value in bx.upper() {
        mix(u64::from(value.to_bits()));
    }
    hash
}

/// Run one original-network uniform-region replay without proposal construction.
///
/// This hook is structurally diagnostic-only: it neither accepts nor returns a
/// baseline, and it cannot construct [`FullObjectiveCertificate`]. Callers that
/// observe `true` return their original baseline/vacuous bounds verbatim.
#[allow(clippy::too_many_arguments)]
fn maybe_run_replay_only_diagnostic(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    objectives: &[Vec<f32>],
    thresholds: &[f32],
    engine: Option<&dyn GemmEngine>,
    overall_deadline: Option<Instant>,
) -> bool {
    let gate = std::env::var("NY_IMB_REPLAY_ONLY").ok();
    let leaf = std::env::var("NY_IMB_REPLAY_ONLY_LEAF").ok();
    let leaf_idx = match replay_only_leaf_request(gate.as_deref(), leaf.as_deref()) {
        Ok(None) => return false,
        Ok(Some(leaf_idx)) => leaf_idx,
        Err(reason) => {
            eprintln!("[imb-replay-only] rejected: {reason}; authority=false");
            return true;
        }
    };
    if super::replay_only_attempted() {
        // The other IMB entry point already ran the one admitted diagnostic.
        // Consume this hook without proposal work and without duplicate telemetry.
        return true;
    }

    if objectives.is_empty() || objectives.len() != thresholds.len() {
        eprintln!(
            "[imb-replay-only] rejected: objective/threshold shape mismatch; authority=false"
        );
        return true;
    }
    let obj_raw = std::env::var("NY_IMB_OBJ").ok();
    let obj_idx = match replay_only_objective(obj_raw.as_deref(), objectives.len()) {
        Ok(obj_idx) => obj_idx,
        Err(reason) => {
            eprintln!("[imb-replay-only] rejected: {reason}; authority=false");
            return true;
        }
    };
    let objective = &objectives[obj_idx];
    let threshold = thresholds[obj_idx];
    if objective.is_empty() || objective.iter().any(|v| !v.is_finite()) || !threshold.is_finite() {
        eprintln!(
            "[imb-replay-only] rejected: non-finite/empty objective or threshold; authority=false"
        );
        return true;
    }

    let total_started = Instant::now();
    let budget_s = match std::env::var("NY_IMB_BUDGET_S") {
        Ok(raw) => match raw.parse::<f64>() {
            Ok(value) => value,
            Err(_) => {
                eprintln!(
                    "[imb-replay-only] rejected: NY_IMB_BUDGET_S is not a decimal f64; authority=false"
                );
                return true;
            }
        },
        Err(_) => 300.0,
    };
    let Some(deadline) = checked_budget_deadline(total_started, budget_s, overall_deadline) else {
        eprintln!("[imb-replay-only] rejected: invalid or exhausted budget; authority=false");
        return true;
    };
    if !replay_deadline_open(deadline) {
        eprintln!("[imb-replay-only] rejected: deadline already expired; authority=false");
        return true;
    }

    let region_k = match std::env::var("NY_IMB_REGION_K") {
        Ok(raw) => match strict_decimal_usize(&raw) {
            Some(value) => value,
            None => {
                eprintln!(
                    "[imb-replay-only] rejected: NY_IMB_REGION_K is not an in-range decimal usize; authority=false"
                );
                return true;
            }
        },
        Err(_) => 1,
    };
    if region_k <= 1 {
        eprintln!(
            "[imb-replay-only] rejected: NY_IMB_REGION_K must be greater than 1; authority=false"
        );
        return true;
    }
    let split_dims = split_free_dims(input);
    let Some(exponent) = u32::try_from(split_dims.len()).ok() else {
        eprintln!("[imb-replay-only] rejected: split dimension count overflow; authority=false");
        return true;
    };
    let Some(expected_regions) = region_k.checked_pow(exponent) else {
        eprintln!("[imb-replay-only] rejected: region count overflow; authority=false");
        return true;
    };
    if split_dims.is_empty() || expected_regions > MAX_FULL_RECHECK_LEAVES {
        eprintln!(
            "[imb-replay-only] rejected: empty/oversized uniform partition ({expected_regions} regions); authority=false"
        );
        return true;
    }

    let validation_started = Instant::now();
    let regions = region_boxes(input, &split_dims, region_k);
    if regions.len() != expected_regions {
        eprintln!(
            "[imb-replay-only] rejected: built {} of {expected_regions} uniform regions; authority=false",
            regions.len()
        );
        return true;
    }
    if let Err(reason) = validate_binary_partition_cover(input, &regions, deadline) {
        eprintln!(
            "[imb-replay-only] rejected: uniform partition failed exact-cover validation ({reason}); authority=false"
        );
        return true;
    }
    let validation_elapsed = validation_started.elapsed();
    let Some(target) = regions.get(leaf_idx) else {
        eprintln!(
            "[imb-replay-only] rejected: leaf {leaf_idx} outside 0..{}; authority=false",
            regions.len()
        );
        return true;
    };

    let split_bits =
        exact_split_bounds_bits(target, &split_dims).unwrap_or_else(|| "unavailable".to_string());
    let full_bits =
        exact_box_bits(target).unwrap_or_else(|| format!("omitted(len={})", target.lower().len()));
    eprintln!(
        "[imb-replay-only] target obj={obj_idx} leaf={leaf_idx}/{} k={region_k} split_dims={split_dims:?} \
         order=region_boxes_mixed_radix_low_dim_first exact_cover=true validation_s={:.6} \
         terminal_index_equivalence=requires_one_terminal_per_region \
         box_fingerprint=fnv1a64:{:016x} split_bounds_bits=[{split_bits}] full_bounds_bits=[{full_bits}] authority=false",
        regions.len(),
        validation_elapsed.as_secs_f64(),
        stable_box_fingerprint(target),
    );

    let Some(spec) = Array2::from_shape_vec((1, objective.len()), objective.clone()).ok() else {
        eprintln!(
            "[imb-replay-only] rejected: objective matrix allocation failed; authority=false"
        );
        return true;
    };
    if !super::begin_replay_only_attempt() {
        return true;
    }
    #[cfg(test)]
    REPLAY_ONLY_EVALUATIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    let one_threshold = [threshold];
    let one_clause = [1_usize];
    let mut timings = ReplayStageTimings::default();
    let evaluation = evaluate_original_objective_leaf(
        graph,
        target,
        &spec,
        &one_threshold,
        &one_clause,
        engine,
        deadline,
        graph_supports_f64_tail(graph),
        Some(&mut timings),
    );
    let f64_seconds = timings.f64.map(|d| d.as_secs_f64());
    let standard_seconds = timings.standard.map(|d| d.as_secs_f64());
    eprintln!(
        "[imb-replay-only] result obj={obj_idx} leaf={leaf_idx} route={:?} f64={:?} \
         standard_lower={:?} diagnostic_lower={:?} threshold={threshold:.9} clears={} \
         f64_s={f64_seconds:?} standard_s={standard_seconds:?} total_s={:.6} authority=false",
        evaluation.route,
        evaluation.f64_attempt,
        evaluation.standard_lower,
        evaluation.lower,
        evaluation
            .lower
            .is_some_and(|lower| lower.is_finite() && lower > threshold),
        total_started.elapsed().as_secs_f64(),
    );
    true
}

/// Independently replay a proposed input partition against the original full
/// network and objective.  The IMB seam, `(p,q)`, sampled slack, tail split and
/// cached IMB floors are deliberately not inputs to this function.
fn independently_recheck_original_objective(
    graph: &GraphNetwork,
    root: &BoundedTensor,
    terminal_boxes: &[BoundedTensor],
    objective: &[f32],
    threshold: f32,
    engine: Option<&dyn GemmEngine>,
    deadline: Instant,
) -> Option<FullObjectiveCertificate> {
    if !replay_deadline_open(deadline) {
        eprintln!("[imb] FULL-RECHECK deadline expired before validation");
        return None;
    }
    if !threshold.is_finite() || objective.is_empty() || objective.iter().any(|v| !v.is_finite()) {
        return None;
    }
    if let Err(reason) = validate_binary_partition_cover(root, terminal_boxes, deadline) {
        eprintln!("[imb] FULL-RECHECK partition rejected: {reason}");
        return None;
    }
    if !replay_deadline_open(deadline) {
        eprintln!("[imb] FULL-RECHECK deadline expired after validation");
        return None;
    }
    let spec = Array2::from_shape_vec((1, objective.len()), objective.to_vec()).ok()?;
    let thresholds = [threshold];
    let clause_sizes = [1_usize];
    let supports_f64 = graph_supports_f64_tail(graph);
    let mut global_lower = f32::INFINITY;

    for (leaf_idx, leaf) in terminal_boxes.iter().enumerate() {
        if !replay_deadline_open(deadline) {
            eprintln!("[imb] FULL-RECHECK deadline expired at leaf {leaf_idx}");
            return None;
        }

        let lower = evaluate_original_objective_leaf(
            graph,
            leaf,
            &spec,
            &thresholds,
            &clause_sizes,
            engine,
            deadline,
            supports_f64,
            None,
        )
        .lower?;

        // This checker is invoked only for a candidate that could change a
        // verdict.  A leaf that does not strictly clear cannot contribute a
        // certificate; abort immediately rather than spend the remaining budget.
        if !replay_deadline_open(deadline) || !lower.is_finite() || lower <= threshold {
            eprintln!(
                "[imb] FULL-RECHECK leaf {leaf_idx} did not clear: lower={lower:.9} threshold={threshold:.9}"
            );
            return None;
        }
        global_lower = global_lower.min(lower);
    }

    if !replay_deadline_open(deadline) || !global_lower.is_finite() || global_lower <= threshold {
        return None;
    }
    eprintln!(
        "[imb] FULL-RECHECK CERTIFIED: leaves={} original_lower={global_lower:.9} > threshold={threshold:.9}",
        terminal_boxes.len()
    );
    if !replay_deadline_open(deadline) {
        return None;
    }
    Some(FullObjectiveCertificate {
        lower: global_lower,
        valid_until: deadline,
    })
}

fn batched_replay_gate(raw: Option<&str>) -> Result<bool, &'static str> {
    match raw {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(_) => Err("NY_IMB_BATCHED_REPLAY must be exactly '0' or '1'"),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SignedReplayProjection {
    representative: usize,
    use_negated_upper: bool,
}

struct SignedReplayObjectivePlan<'a> {
    representatives: Vec<&'a [f32]>,
    projections: Vec<SignedReplayProjection>,
}

fn same_finite_row_bits(left: &[f32], right: &[f32]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(&a, &b)| a.is_finite() && b.is_finite() && a.to_bits() == b.to_bits())
}

/// Whether `left` is the exact mathematical negation of `right`.
///
/// Non-zero coefficients must match bit-for-bit after flipping the IEEE-754
/// sign bit. Signed zero carries no mathematical sign, so either zero encoding
/// matches either encoding on a row whose orientation is established by at
/// least one non-zero coefficient. An all-zero row has no orientation and is
/// deliberately not sign-quotiented.
fn opposite_finite_row_bits(left: &[f32], right: &[f32]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    let mut oriented = false;
    for (&a, &b) in left.iter().zip(right) {
        if !a.is_finite() || !b.is_finite() {
            return false;
        }
        if a == 0.0 && b == 0.0 {
            continue;
        }
        oriented = true;
        if a.to_bits() != (b.to_bits() ^ (1_u32 << 31)) {
            return false;
        }
    }
    oriented
}

/// Quotient dense replay objectives by exact equality up to negation.
///
/// One CROWN spec row already returns both a sound lower and a sound upper.
/// Therefore an objective `-c` does not need a second propagated row when `c`
/// is present: its lower bound is exactly the negation of `c`'s upper bound.
/// Representatives retain first-occurrence order. Rows outside an exact signed
/// equivalence class retain their original row and direct-lower projection.
fn signed_replay_objective_plan<'a>(
    objectives: &[&'a [f32]],
    output_dim: usize,
) -> Option<SignedReplayObjectivePlan<'a>> {
    if objectives.is_empty() || output_dim == 0 {
        return None;
    }

    let mut representatives: Vec<&'a [f32]> = Vec::with_capacity(objectives.len());
    let mut projections = Vec::with_capacity(objectives.len());
    for &objective in objectives {
        if objective.len() != output_dim || objective.iter().any(|value| !value.is_finite()) {
            return None;
        }

        if let Some(representative) = representatives
            .iter()
            .position(|row| same_finite_row_bits(objective, row))
        {
            projections.push(SignedReplayProjection {
                representative,
                use_negated_upper: false,
            });
            continue;
        }
        if let Some(representative) = representatives
            .iter()
            .position(|row| opposite_finite_row_bits(objective, row))
        {
            projections.push(SignedReplayProjection {
                representative,
                use_negated_upper: true,
            });
            continue;
        }

        let representative = representatives.len();
        representatives.push(objective);
        projections.push(SignedReplayProjection {
            representative,
            use_negated_upper: false,
        });
    }

    Some(SignedReplayObjectivePlan {
        representatives,
        projections,
    })
}

fn signed_replay_project_lower(
    rows: &[(f32, f32)],
    projection: SignedReplayProjection,
) -> Option<f32> {
    let &(lower, upper) = rows.get(projection.representative)?;
    if !lower.is_finite() || !upper.is_finite() || lower > upper {
        return None;
    }
    let projected = if projection.use_negated_upper {
        // IEEE-754 negation flips only the sign bit, so this introduces no
        // rounding: upper(c·f) >= c·f implies -upper(c·f) <= (-c)·f.
        -upper
    } else {
        lower
    };
    projected.is_finite().then_some(projected)
}

/// Independently replay several original-network objectives over their own
/// exact-cover partitions in one dense-spec domain batch.
///
/// The covers do not need to be identical. Bit-identical leaves are
/// deduplicated across covers, while `cover_leaf_indices[obj][leaf]` retains
/// the exact membership of each objective's independently validated cover. Exact
/// opposite objective rows share one CROWN representative: the representative's
/// lower serves `c`, and its upper serves `-c` after exact negation. Every other
/// row retains the historical direct-lower path. No IMB seam, `(p,q)`, sample,
/// prefix floor, or proposal cache is an input to this replay.
/// `retained_certified_memberships` prices the endpoint storage of quarantined
/// AY-certified covers that remain live beside this replay but are not
/// redundantly evaluated.
///
/// Soundness boundary:
/// - every objective cover is validated separately against `root`;
/// - the original full graph and original objective rows are evaluated;
/// - all returned row bounds must be finite and finish before the common
///   deadline;
/// - every leaf in every cover must strictly clear its own threshold;
/// - certificates are returned atomically (all objectives or none).
#[allow(clippy::too_many_arguments)]
fn independently_recheck_original_objectives_batched(
    graph: &GraphNetwork,
    root: &BoundedTensor,
    terminal_partitions: &[&[BoundedTensor]],
    objectives: &[&[f32]],
    thresholds: &[f32],
    retained_certified_memberships: usize,
    engine: Option<&dyn GemmEngine>,
    deadline: Instant,
) -> Option<Vec<FullObjectiveCertificate>> {
    if !replay_deadline_open(deadline)
        || objectives.is_empty()
        || objectives.len() != thresholds.len()
        || objectives.len() != terminal_partitions.len()
    {
        return None;
    }

    let output_dim = objectives.first()?.len();
    if output_dim == 0
        || objectives
            .iter()
            .any(|row| row.len() != output_dim || row.iter().any(|v| !v.is_finite()))
        || thresholds.iter().any(|v| !v.is_finite())
    {
        return None;
    }
    let signed_plan = signed_replay_objective_plan(objectives, output_dim)?;
    let replay_specs = signed_plan.representatives.len();

    let mut replay_memberships = 0usize;
    for partition in terminal_partitions {
        replay_memberships = replay_memberships.checked_add(partition.len())?;
        let total_live_memberships =
            replay_memberships.checked_add(retained_certified_memberships)?;
        if partition.is_empty()
            || partition.len() > MAX_FULL_RECHECK_LEAVES
            || total_live_memberships > MAX_BATCHED_FULL_RECHECK_MEMBERSHIPS
        {
            eprintln!(
                "[imb] FULL-RECHECK-BATCH rejected oversized/empty cover: \
                 cover_leaves={} replay_memberships={replay_memberships} \
                 retained_certified_memberships={retained_certified_memberships} \
                 total_live_memberships={total_live_memberships}",
                partition.len(),
            );
            return None;
        }
    }
    let total_live_memberships = replay_memberships.checked_add(retained_certified_memberships)?;
    let prevalidation_shape = BatchedReplayPrevalidationShape {
        total_memberships: total_live_memberships,
        input_elements_per_leaf: root.lower().len(),
    };
    let (unique_leaves, cover_leaf_indices) =
        validate_batched_replay_structure_if_admitted(prevalidation_shape, || {
            for partition in terminal_partitions {
                if let Err(reason) = validate_binary_partition_cover(root, partition, deadline) {
                    eprintln!("[imb] FULL-RECHECK-BATCH partition rejected: {reason}");
                    return None;
                }
            }
            if !replay_deadline_open(deadline) {
                return None;
            }
            index_exact_replay_leaves(terminal_partitions, deadline)
        })?;

    let started = Instant::now();
    let resource_shape = BatchedReplayResourceShape {
        unique_leaves: unique_leaves.len(),
        replay_objectives: objectives.len(),
        total_memberships: total_live_memberships,
        input_elements_per_leaf: root.lower().len(),
        output_dim,
    };
    let batched = evaluate_batched_replay_if_admitted(resource_shape, || {
        let spec_elements = replay_specs.checked_mul(output_dim)?;
        let mut spec_values = Vec::new();
        spec_values.try_reserve_exact(spec_elements).ok()?;
        for row in &signed_plan.representatives {
            spec_values.extend_from_slice(row);
        }
        let spec = Array2::from_shape_vec((replay_specs, output_dim), spec_values).ok()?;

        // Materialization is bounded above, but it can still consume the last
        // usable wall-clock slice. Never enter the evaluator after that work
        // if the common authority deadline has closed.
        if !replay_deadline_open(deadline) {
            return None;
        }

        match compute_crown_or_ibp_bounds_batched_specs(
            graph,
            &unique_leaves,
            &spec,
            engine,
            None,
            None,
            None,
            Some(deadline),
            None,
            true,
            true,
        ) {
            Ok(result) => Some(result),
            Err(error) => {
                eprintln!("[imb] FULL-RECHECK-BATCH evaluator failed closed: {error}");
                None
            }
        }
    })?;
    if !replay_deadline_open(deadline)
        || batched.bounds.len() != unique_leaves.len()
        || batched.rebound_timing.domains != unique_leaves.len()
        || batched.rebound_timing.num_specs != replay_specs
    {
        eprintln!(
            "[imb] FULL-RECHECK-BATCH rejected result shape/deadline: \
             bounds={} unique_leaves={} timing_domains={} timing_specs={} expected_specs={}",
            batched.bounds.len(),
            unique_leaves.len(),
            batched.rebound_timing.domains,
            batched.rebound_timing.num_specs,
            replay_specs
        );
        return None;
    }

    let mut per_domain_rows = Vec::with_capacity(batched.bounds.len());
    for bounds in &batched.bounds {
        if !replay_deadline_open(deadline)
            || bounds.lower().len() != replay_specs
            || bounds.upper().len() != replay_specs
        {
            return None;
        }
        let lower: Vec<f32> = bounds.lower().iter().copied().collect();
        let upper: Vec<f32> = bounds.upper().iter().copied().collect();
        let rows: Vec<(f32, f32)> = lower.into_iter().zip(upper).collect();
        if rows
            .iter()
            .any(|(lower, upper)| !lower.is_finite() || !upper.is_finite() || lower > upper)
        {
            eprintln!("[imb] FULL-RECHECK-BATCH rejected non-finite/inverted row bounds");
            return None;
        }
        per_domain_rows.push(rows);
    }

    let mut certificates = Vec::with_capacity(objectives.len());
    for (obj_idx, (membership, &projection)) in cover_leaf_indices
        .iter()
        .zip(&signed_plan.projections)
        .enumerate()
    {
        let threshold = thresholds[obj_idx];
        let mut global_lower = f32::INFINITY;
        for (leaf_idx, &unique_idx) in membership.iter().enumerate() {
            if !replay_deadline_open(deadline) {
                return None;
            }
            let lower = signed_replay_project_lower(per_domain_rows.get(unique_idx)?, projection)?;
            if !lower.is_finite() || lower <= threshold {
                eprintln!(
                    "[imb] FULL-RECHECK-BATCH obj={obj_idx} leaf={leaf_idx} \
                     unique_leaf={unique_idx} did not clear: \
                     lower={lower:.9} threshold={threshold:.9}"
                );
                return None;
            }
            global_lower = global_lower.min(lower);
        }
        if !global_lower.is_finite() || global_lower <= threshold {
            return None;
        }
        certificates.push(FullObjectiveCertificate {
            lower: global_lower,
            valid_until: deadline,
        });
    }

    if !replay_deadline_open(deadline) {
        return None;
    }
    if replay_specs != objectives.len() {
        eprintln!(
            "[imb] FULL-RECHECK-BATCH signed-spec quotient: objectives={} representatives={replay_specs}",
            objectives.len(),
        );
    }
    eprintln!(
        "[imb] FULL-RECHECK-BATCH CERTIFIED: objectives={} replay_memberships={} \
         retained_certified_memberships={} total_live_memberships={} unique_leaves={} \
         route={} wall_s={:.6} forward_s={:?} backward_s={:?} materialize_s={:?}",
        objectives.len(),
        replay_memberships,
        retained_certified_memberships,
        total_live_memberships,
        unique_leaves.len(),
        batched.rebound_timing.mode.as_str(),
        started.elapsed().as_secs_f64(),
        batched.rebound_timing.forward_elapsed_s,
        batched.rebound_timing.backward_elapsed_s,
        batched.rebound_timing.materialize_elapsed_s,
    );
    for (obj_idx, certificate) in certificates.iter().enumerate() {
        eprintln!(
            "[imb] FULL-RECHECK-BATCH obj={obj_idx} original_lower={:.9} > threshold={:.9} \
             cover_leaves={}",
            certificate.lower,
            thresholds[obj_idx],
            terminal_partitions[obj_idx].len(),
        );
    }
    replay_deadline_open(deadline).then_some(certificates)
}

/// Resolve one all-single-row candidate set as a single authority transaction.
///
/// `run_imb_measurement` may already have minted an AY-tail/prefix certificate.
/// Those tokens are removed from the candidates before any validation so they
/// cannot be consumed accidentally while the batch is incomplete. Each is
/// accepted only when its objective, threshold, exact cover, and deadline match
/// its candidate. Objectives without such a token are replayed together through
/// [`independently_recheck_original_objectives_batched`]. Only after every slot
/// holds a valid certificate are all tokens reinstalled and the lower vector
/// returned. Any failure leaves every candidate certificate-free.
#[allow(clippy::too_many_arguments)]
fn certify_candidates_with_batched_replay(
    graph: &GraphNetwork,
    root: &BoundedTensor,
    candidates: &mut [ImbCandidate],
    objectives: &[Vec<f32>],
    thresholds: &[f32],
    engine: Option<&dyn GemmEngine>,
) -> Option<Vec<f32>> {
    let n = candidates.len();
    if n == 0 || objectives.len() != n || thresholds.len() != n {
        return None;
    }
    let output_dim = objectives.first()?.len();
    if output_dim == 0
        || objectives
            .iter()
            .any(|row| row.len() != output_dim || row.iter().any(|value| !value.is_finite()))
        || thresholds.iter().any(|threshold| !threshold.is_finite())
    {
        return None;
    }
    let common_deadline = candidates.iter().map(|c| c.recheck_deadline).min()?;
    if !replay_deadline_open(common_deadline) {
        return None;
    }

    // Quarantine every pre-existing token up front. On any return below no
    // candidate retains partial authority.
    let incoming: Vec<Option<FullObjectiveCertificate>> = candidates
        .iter_mut()
        .map(|candidate| candidate.full_certificate.take())
        .collect();
    let total_candidate_memberships = candidates.iter().try_fold(0usize, |total, candidate| {
        total.checked_add(candidate.terminal_boxes.len())
    })?;
    validate_batched_replay_structure_if_admitted(
        BatchedReplayPrevalidationShape {
            total_memberships: total_candidate_memberships,
            input_elements_per_leaf: root.lower().len(),
        },
        || Some(()),
    )?;
    let mut authorities = vec![None; n];
    let mut replay_rows = Vec::new();
    let mut retained_certified_memberships = 0usize;

    for (row, (candidate, preexisting)) in candidates.iter().zip(incoming).enumerate() {
        if candidate.measurement_only
            || candidate.obj_idx != row
            || candidate.threshold.to_bits() != thresholds[row].to_bits()
            || !replay_deadline_open(candidate.recheck_deadline)
        {
            eprintln!("[imb] FULL-RECHECK-BATCH candidate {row} failed identity checks");
            return None;
        }

        let Some(mut certificate) = preexisting else {
            if !candidate.imb_floor.is_finite() || candidate.imb_floor <= candidate.threshold {
                eprintln!(
                    "[imb] FULL-RECHECK-BATCH uncertified candidate {row} proposal did not clear"
                );
                return None;
            }
            replay_rows.push(row);
            continue;
        };
        if certificate.valid_until != candidate.recheck_deadline
            || !replay_deadline_open(certificate.valid_until)
            || !certificate.lower.is_finite()
            || certificate.lower <= thresholds[row]
        {
            eprintln!(
                "[imb] FULL-RECHECK-BATCH candidate {row} rejected mismatched \
                 pre-existing certificate"
            );
            return None;
        }
        if let Err(reason) =
            validate_binary_partition_cover(root, &candidate.terminal_boxes, common_deadline)
        {
            eprintln!(
                "[imb] FULL-RECHECK-BATCH candidate {row} rejected pre-certified cover: {reason}"
            );
            return None;
        }
        // Never extend an AY token; shorten it to the grouped transaction's
        // earliest deadline so every installed authority expires atomically.
        certificate.valid_until = common_deadline;
        retained_certified_memberships =
            retained_certified_memberships.checked_add(candidate.terminal_boxes.len())?;
        authorities[row] = Some(certificate);
    }

    if !replay_rows.is_empty() {
        eprintln!("[imb] FULL-RECHECK-BATCH replaying uncertified global rows {replay_rows:?}");
        let terminal_partitions: Vec<&[BoundedTensor]> = replay_rows
            .iter()
            .map(|&row| candidates[row].terminal_boxes.as_slice())
            .collect();
        let objective_rows: Vec<&[f32]> = replay_rows
            .iter()
            .map(|&row| objectives[row].as_slice())
            .collect();
        let replay_thresholds: Vec<f32> = replay_rows.iter().map(|&row| thresholds[row]).collect();
        let replayed = independently_recheck_original_objectives_batched(
            graph,
            root,
            &terminal_partitions,
            &objective_rows,
            &replay_thresholds,
            retained_certified_memberships,
            engine,
            common_deadline,
        )?;
        if replayed.len() != replay_rows.len() {
            return None;
        }
        for (row, certificate) in replay_rows.iter().copied().zip(replayed) {
            authorities[row] = Some(certificate);
        }
    }

    if !replay_deadline_open(common_deadline) {
        return None;
    }
    let mut lowers = Vec::with_capacity(n);
    for (row, certificate) in authorities.iter().enumerate() {
        let certificate = certificate.as_ref()?;
        if certificate.valid_until != common_deadline
            || !certificate.lower.is_finite()
            || certificate.lower <= thresholds[row]
        {
            return None;
        }
        lowers.push(certificate.lower);
    }
    if !replay_deadline_open(common_deadline) {
        return None;
    }

    for (candidate, certificate) in candidates.iter_mut().zip(authorities) {
        candidate.full_certificate = certificate;
    }
    if !replay_deadline_open(common_deadline) {
        for candidate in candidates {
            candidate.full_certificate = None;
        }
        return None;
    }
    eprintln!(
        "[imb] FULL-RECHECK-BATCH authority transaction complete: \
         objectives={n} pre_certified={} replayed={}",
        n - replay_rows.len(),
        replay_rows.len(),
    );
    Some(lowers)
}

/// IMB root-floor injection.
///
/// - **STAGE 1 (default, log-only):** with `NY_IMB=1` but WITHOUT `NY_IMB_WIRE=1`,
///   the measurement runs and LOGS the floor, and this returns the caller's
///   `baseline` VERBATIM — no verdict/timing/byte change.
/// - **STAGE 3 (opt-in wiring):** with `NY_IMB=1 && NY_IMB_WIRE=1`, IMB proposes a
///   terminal input partition. The original full-network objective is either
///   independently re-bounded over every exact-cover leaf or, under the additional
///   exact `NY_IMB_TAIL_CERT_AY=1` gate, proved by exact AY+ny-cert tail
///   obligations under NY's cover-backed prefix facts. Only a certified lower
///   is raised into the baseline. Every other path returns `baseline` unchanged.
#[allow(clippy::too_many_arguments)]
pub fn tighten_root_objective_bounds_imb(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    objectives: &[Vec<f32>],
    thresholds: &[f32],
    engine: Option<&dyn GemmEngine>,
    node_bounds: &HashMap<String, BoundedTensor>,
    _alpha_state: Option<&GraphAlphaState>,
    baseline: &[(f32, f32)],
    deadline: Option<Instant>,
) -> Vec<(f32, f32)> {
    // Re-check the arming predicate (the call site only checked `enabled()`).
    if !armed(graph, input) {
        return baseline.to_vec();
    }
    // Re-entrancy guard: the nested per-leaf prefix-BaB must not re-arm IMB.
    let _scope = super::scope();

    if maybe_run_replay_only_diagnostic(graph, input, objectives, thresholds, engine, deadline) {
        // Replay-only is measurement data, never a bound source. In particular,
        // NY_IMB_WIRE cannot turn its selected-leaf lower into authority.
        return baseline.to_vec();
    }

    let mut cand = run_imb_measurement(
        graph,
        input,
        objectives,
        thresholds,
        engine,
        node_bounds,
        baseline,
        None,
        None,
        deadline,
    );

    // STAGE 3 — only an independent full-network partition replay can construct
    // the authority token consumed below.  The IMB floor merely decides whether
    // spending the replay budget is worthwhile.
    let wire = matches!(std::env::var("NY_IMB_WIRE").ok().as_deref(), Some("1"));
    if wire {
        if let Some(c) = cand.as_mut() {
            if c.full_certificate.is_none()
                && !c.measurement_only
                && c.obj_idx < objectives.len()
                && c.imb_floor.is_finite()
                && c.imb_floor > c.threshold
            {
                c.full_certificate = independently_recheck_original_objective(
                    graph,
                    input,
                    &c.terminal_boxes,
                    &objectives[c.obj_idx],
                    c.threshold,
                    engine,
                    c.recheck_deadline,
                );
            }
        }
    }
    if let Some(c) = cand {
        if wire && c.obj_idx < baseline.len() {
            let Some(certified_lower) = authoritative_candidate_lower(&c) else {
                return baseline.to_vec();
            };
            let mut out = baseline.to_vec();
            let old = out[c.obj_idx].0;
            // SOUNDNESS: only the independent full-network recheck's lower bound
            // feeds the max.  The sampled/decomposed IMB floor remains telemetry.
            let raised = old.max(certified_lower);
            out[c.obj_idx].0 = raised;
            eprintln!(
                "[imb] WIRED: raised obj_idx={} lower baseline={old:.6} -> certified={certified_lower:.6} (imb_floor={:.6}, threshold={:.6} : {})",
                c.obj_idx,
                c.imb_floor,
                c.threshold,
                if raised > c.threshold { "VERIFIED" } else { "not-yet" }
            );
            return out;
        }
    }

    // Log-only default / guard-fail / unavailable: baseline returned unchanged.
    baseline.to_vec()
}

/// STEP 2 — multi-objective early fast-path floors for an OR-of-AND disjunction.
///
/// For UNSAT, EVERY disjunctive clause must be refuted (`disjunctive_domain_verified`
/// = every clause has some row with `lower > threshold`). This certifies a per-objective
/// IMB partition for each clause's binding row(s), independently rechecks the
/// original full-network row over every terminal box, and raises only those
/// replay-certified lowers into a fresh
/// vacuous-`(-inf,+inf)` bound vector, which the caller feeds to
/// `disjunctive_domain_verified`. Each clause is refuted as soon as ONE of its rows
/// clears (`out[row].0 > threshold`); a clause whose rows all fail to clear aborts the
/// loop early (the whole disjunction can't be IMB-refuted → the caller falls through to
/// the standard pipeline, unchanged).
///
/// SHARING: the prefix tight-anchor (`build_tight_prefix_anchor`, the ~35 s Lever-A
/// build) is OBJECTIVE-INDEPENDENT. Exact AY mode binds it to this invocation's live
/// graph/input borrows in [`ExactPrefixSession`], so the first objective builds it and
/// later objectives shallow-clone the same prefix/map Arcs without consulting a
/// process-persistent hash memo. The tail functional `(p,q)` and region tail coeffs ARE
/// objective-dependent (they carry `obj_row`'s sign), so they are recomputed per
/// objective. Each row is verdict-authoritative only after the exact-cover
/// original-network replay succeeds; an incomplete/un-cleared row simply does not
/// raise.
///
/// prop_0 (single clause, single row) certifies exactly one objective — byte-for-byte
/// the prior single-objective behavior.
///
/// `NY_IMB_BATCHED_REPLAY=1` enables a strict default-off specialization for
/// layouts where every clause contains exactly one row. It first constructs
/// every clause's candidate and quarantines any separately minted AY-tail
/// certificate. Missing authorities are independently replayed over validated
/// exact covers in one domain-stacked dense-spec transaction; only a complete
/// AY/replay certificate vector is installed atomically. This is the
/// alpha-beta-CROWN/NeuralSAT batching pattern (domain axis × spec-row axis)
/// applied only at the independent authority boundary. Unsupported layouts
/// retain the historical serial path; malformed gate values fail closed.
#[allow(clippy::too_many_arguments)]
pub fn imb_multi_objective_floors(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    objectives: &[Vec<f32>],
    thresholds: &[f32],
    clause_sizes: &[usize],
    engine: Option<&dyn GemmEngine>,
    node_bounds: &HashMap<String, BoundedTensor>,
    deadline: Option<Instant>,
) -> Vec<(f32, f32)> {
    let n = objectives.len();
    let mut out = vec![(f32::NEG_INFINITY, f32::INFINITY); n];
    if !armed(graph, input) || !valid_disjunctive_layout(n, thresholds.len(), clause_sizes) {
        return out;
    }
    // Re-entrancy guard set ONCE for the whole multi-objective loop (the nested
    // per-leaf prefix-BaB must not re-arm IMB).
    let _scope = super::scope();

    if maybe_run_replay_only_diagnostic(graph, input, objectives, thresholds, engine, deadline) {
        // A single diagnostic leaf cannot refute any clause.
        return out;
    }

    let batched_replay =
        match batched_replay_gate(std::env::var("NY_IMB_BATCHED_REPLAY").ok().as_deref()) {
            Ok(enabled) => enabled,
            Err(reason) => {
                eprintln!("[imb] FULL-RECHECK-BATCH rejected: {reason}");
                return out;
            }
        };

    // Vacuous per-objective baseline — the sound IMB floor stands alone
    // (`max(-inf, imb_floor) = imb_floor`), forcing the objective directly.
    let vac_baseline = vec![(f32::NEG_INFINITY, f32::INFINITY); n];
    // One exact graph/input identity spans both control-flow variants below.
    // Gate-off calls ignore it and retain the historical proposal-only memo path.
    let mut exact_prefix_session = ExactPrefixSession::new(graph, input);

    if batched_replay
        && clause_sizes.len() == n
        && clause_sizes.iter().all(|&size| size == 1)
        && !graph_supports_f64_tail(graph)
    {
        eprintln!(
            "[imb] FULL-RECHECK-BATCH armed for {} independent single-row clauses",
            n
        );
        let mut candidates = Vec::with_capacity(n);
        let mut retained_candidate_memberships = 0usize;
        for row in 0..n {
            eprintln!(
                "[imb] MULTI-OBJ clause {row} row {row}: proposing for batched replay \
                 (threshold={:.6})",
                thresholds[row]
            );
            let Some(candidate) = run_imb_measurement(
                graph,
                input,
                objectives,
                thresholds,
                engine,
                node_bounds,
                &vac_baseline,
                Some(row),
                Some(&mut exact_prefix_session),
                deadline,
            ) else {
                eprintln!(
                    "[imb] MULTI-OBJ clause {row} NOT refuted by IMB — \
                     abandoning batched early fast-path"
                );
                return out;
            };
            retained_candidate_memberships =
                match retained_candidate_memberships.checked_add(candidate.terminal_boxes.len()) {
                    Some(total) => total,
                    None => return out,
                };
            if validate_batched_replay_structure_if_admitted(
                BatchedReplayPrevalidationShape {
                    total_memberships: retained_candidate_memberships,
                    input_elements_per_leaf: input.lower().len(),
                },
                || Some(()),
            )
            .is_none()
            {
                eprintln!(
                    "[imb] FULL-RECHECK-BATCH retained candidate covers exceeded \
                     aggregate membership/byte admission"
                );
                return out;
            }
            candidates.push(candidate);
        }

        let Some(certified_lowers) = certify_candidates_with_batched_replay(
            graph,
            input,
            &mut candidates,
            objectives,
            thresholds,
            engine,
        ) else {
            eprintln!("[imb] FULL-RECHECK-BATCH failed closed — abandoning early fast-path");
            return out;
        };
        if certified_lowers.len() != candidates.len() {
            return out;
        }
        let Some(common_deadline) = candidates
            .first()
            .and_then(|candidate| candidate.full_certificate)
            .map(|certificate| certificate.valid_until)
        else {
            return out;
        };

        let mut certified = out.clone();
        for (row, (candidate, transaction_lower)) in
            candidates.iter().zip(certified_lowers).enumerate()
        {
            let Some(certified_lower) = authoritative_candidate_lower(candidate) else {
                return out;
            };
            if candidate.obj_idx != row
                || certified_lower.to_bits() != transaction_lower.to_bits()
                || !certified_lower.is_finite()
                || certified_lower <= thresholds[row]
            {
                return out;
            }
            certified[row].0 = certified_lower;
        }
        for (row, candidate) in candidates.iter().enumerate() {
            eprintln!(
                "[imb] MULTI-OBJ clause {row} REFUTED via row {row} \
                 (certified={:.6} > thr={:.6}; imb_floor={:.6}; batched=true)",
                certified[row].0, thresholds[row], candidate.imb_floor
            );
        }
        return if replay_deadline_open(common_deadline) {
            certified
        } else {
            out
        };
    }
    if batched_replay {
        eprintln!(
            "[imb] FULL-RECHECK-BATCH declined: requires all-single-row layout and \
             graph-unsupported f64 tail; using historical serial replay"
        );
    }

    let mut offset = 0usize;
    for (clause_i, &size) in clause_sizes.iter().enumerate() {
        let mut cleared = false;
        for row in offset..(offset + size).min(n) {
            eprintln!(
                "[imb] MULTI-OBJ clause {clause_i} row {row}: certifying (threshold={:.6})",
                thresholds[row]
            );
            let mut cand = run_imb_measurement(
                graph,
                input,
                objectives,
                thresholds,
                engine,
                node_bounds,
                &vac_baseline,
                Some(row),
                Some(&mut exact_prefix_session),
                deadline,
            );
            if let Some(c) = cand.as_mut() {
                if c.full_certificate.is_none()
                    && !c.measurement_only
                    && c.obj_idx < objectives.len()
                    && c.imb_floor.is_finite()
                    && c.imb_floor > c.threshold
                {
                    c.full_certificate = independently_recheck_original_objective(
                        graph,
                        input,
                        &c.terminal_boxes,
                        &objectives[c.obj_idx],
                        c.threshold,
                        engine,
                        c.recheck_deadline,
                    );
                }
            }
            if let Some(c) = cand {
                // Same fail-closed authority boundary as the single-objective wire:
                // sampled slack and the decomposed IMB floor are diagnostics only.
                if let Some(certified_lower) = authoritative_candidate_lower(&c) {
                    if c.obj_idx >= n {
                        continue;
                    }
                    out[c.obj_idx].0 = out[c.obj_idx].0.max(certified_lower);
                    if out[c.obj_idx].0 > thresholds[c.obj_idx] {
                        eprintln!(
                            "[imb] MULTI-OBJ clause {clause_i} REFUTED via row {} (certified={:.6} > thr={:.6}; imb_floor={:.6})",
                            c.obj_idx, out[c.obj_idx].0, thresholds[c.obj_idx]
                            , c.imb_floor
                        );
                        cleared = true;
                        break;
                    }
                }
            }
        }
        if !cleared {
            eprintln!(
                "[imb] MULTI-OBJ clause {clause_i} NOT refuted by IMB — abandoning early fast-path"
            );
            break;
        }
        offset += size;
    }
    out
}

/// Compute + LOG the IMB floor, returning the qualified [`ImbCandidate`] (or `None`
/// on any unavailable / gate-failed path, each already diagnosed via `eprintln!`).
/// The candidate is what STAGE 3 wiring consumes; STAGE 1 ignores it.
#[allow(clippy::too_many_arguments)]
fn run_imb_measurement(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    objectives: &[Vec<f32>],
    thresholds: &[f32],
    engine: Option<&dyn GemmEngine>,
    node_bounds: &HashMap<String, BoundedTensor>,
    baseline: &[(f32, f32)],
    forced_obj_idx: Option<usize>,
    exact_prefix_session: Option<&mut ExactPrefixSession<'_>>,
    overall_deadline: Option<Instant>,
) -> Option<ImbCandidate> {
    if objectives.is_empty()
        || objectives.len() != baseline.len()
        || objectives.len() != thresholds.len()
    {
        eprintln!(
            "[imb] shape mismatch objectives={} baseline={} thresholds={} — skip",
            objectives.len(),
            baseline.len(),
            thresholds.len()
        );
        return None;
    }

    if matches!(std::env::var("NY_IMB_DUMP").ok().as_deref(), Some("1")) {
        dump_nodes(graph);
    }

    // IMB has its own sub-budget but may never outlive the verifier's official
    // overall deadline now that its output can trigger an authoritative replay.
    let imb_t0 = Instant::now();
    let budget_s = env_f64("NY_IMB_BUDGET_S", 300.0);
    if !budget_s.is_finite() || budget_s <= 0.0 {
        eprintln!("[imb] invalid NY_IMB_BUDGET_S={budget_s}; skip");
        return None;
    }
    let Some(imb_deadline) = checked_budget_deadline(imb_t0, budget_s, overall_deadline) else {
        eprintln!("[imb] unrepresentable NY_IMB_BUDGET_S={budget_s}; skip");
        return None;
    };
    if imb_t0 >= imb_deadline {
        eprintln!("[imb] overall deadline already expired; skip");
        return None;
    }

    // --- objective selection: STEP-2 multi-objective forces a specific clause row;
    // otherwise the binding (smallest-margin) objective, else NY_IMB_OBJ.
    let obj_idx = match forced_obj_idx {
        Some(i) if i < objectives.len() => i,
        _ => select_objective(objectives, thresholds, baseline),
    };
    let obj_row = &objectives[obj_idx];
    let crown_root = baseline[obj_idx].0;
    let band_lo = thresholds[obj_idx];

    // --- seam.
    let seam = match resolve_seam(graph) {
        Some(s) => s,
        None => {
            eprintln!("[imb] no seam resolved (set NY_IMB_SEAM or check NY_IMB_TAIL_RELUS); skip");
            return None;
        }
    };
    // TIGHT prefix anchor (~180 s once): tight intermediate ReLU-source bounds for
    // the input→seam path. The seam-box CROWN backward relaxes its intermediate
    // ReLUs on these. WITHOUT them — e.g. the EARLY fast-path, whose cheap IBP
    // `node_bounds` has no CROWN-tightened disc nodes — the seam box explodes to IBP
    // width (~278), which collapses the tail affine relaxation (and stalls the tail
    // optimizer). With the full-collection `node_bounds` the disc nodes were already
    // tight; overlaying the prefix anchor is tighter-or-equal, so this helps the early
    // path and never loosens the late path. Exact AY obtains a graph/input-bound
    // run-local pair; legacy proposal-only execution retains its historical memo.
    let exact_ay_requested = super::ay_tail_certificate_enabled();
    let (prefix, prefix_anchor) = if exact_ay_requested {
        let t_anchor = Instant::now();
        let prepared = match exact_prefix_session {
            Some(session) => session.prepare(graph, input, &seam, engine, imb_deadline),
            None => {
                // The single-objective entry has no sharing opportunity, but uses
                // the same raw exact builder and never consults the weak hash memo.
                let mut one_shot = ExactPrefixSession::new(graph, input);
                one_shot.prepare(graph, input, &seam, engine, imb_deadline)
            }
        };
        eprintln!(
            "[imb] PHASE exact-prefix-anchor: {:.1}s (cum {:.1}s)",
            t_anchor.elapsed().as_secs_f64(),
            imb_t0.elapsed().as_secs_f64()
        );
        let Some(prepared) = prepared else {
            eprintln!("[imb] exact prefix preparation unavailable; skip");
            return None;
        };
        (prepared.prefix, prepared.anchor)
    } else {
        // Proposal-only compatibility path: preserve the historical prefix-build
        // placement, timing, and one-slot memo behavior.
        let prefix = match build_prefix(graph, &seam) {
            Some(graph) => Arc::new(graph),
            None => {
                eprintln!("[imb] prefix sub-graph build failed; skip");
                return None;
            }
        };
        let t_anchor = Instant::now();
        let prefix_anchor =
            build_tight_prefix_anchor_cached(prefix.as_ref(), input, engine, imb_deadline, true);
        eprintln!(
            "[imb] PHASE prefix-anchor: {:.1}s (cum {:.1}s)",
            t_anchor.elapsed().as_secs_f64(),
            imb_t0.elapsed().as_secs_f64()
        );
        (prefix, prefix_anchor)
    };
    let exact_ay_anchor = exact_ay_requested
        .then_some(prefix_anchor.as_ref())
        .flatten()
        .filter(|anchor| anchor.contains_key(&seam));
    // Exact reuse borrows the Arc map directly. Only the legacy/fallback overlay
    // allocates a merged map; cloning a 13-node, large-tensor anchor per objective
    // would erase much of the run-local reuse win.
    let merged_seam_nb = exact_ay_anchor.is_none().then(|| {
        let mut bounds = node_bounds.clone();
        if let Some(pa) = prefix_anchor.as_ref() {
            for (k, v) in pa.iter() {
                bounds.insert(k.clone(), v.clone());
            }
        }
        bounds
    });
    let seam_nb: &HashMap<String, BoundedTensor> = exact_ay_anchor
        .map(Arc::as_ref)
        .or(merged_seam_nb.as_ref())?;

    // CROWN-TIGHT seam box (the whole fix): reading `root_node_bounds[seam]` gives
    // the IBP box for the seam — ny only CROWN-tightens its demand-set nodes and
    // the seam ReLU is not one, so its stored box is IBP-wide (width ~100-255 for
    // disc nodes), over which the tail affine relaxation explodes. A CROWN backward
    // concretized at the seam node over the (tiny) input box — anchored on the tight
    // prefix bounds — gives a tight box (~0.02-0.11). Fall back to the IBP box only if
    // that fails (logged).
    let ibp_seam_box = exact_ay_anchor
        .and_then(|anchor| anchor.get(&seam).cloned())
        .or_else(|| {
            (!exact_ay_requested)
                .then(|| node_bounds.get(&seam).cloned())
                .flatten()
        });
    let crown_seam_box = graph.propagate_crown_to_node(
        input,
        &seam,
        seam_nb,
        seam_nb,
        engine,
        Some(imb_deadline),
        None,
        None,
    );
    let (seam_box, ay_seam_box_trusted) = match crown_seam_box {
        Ok(b) => (b, exact_ay_requested && exact_ay_anchor.is_some()),
        Err(e) => {
            eprintln!("[imb] CROWN-tight seam box failed ({e}); falling back to IBP seam box");
            match ibp_seam_box.clone() {
                Some(b) => (b, exact_ay_requested && exact_ay_anchor.is_some()),
                None => {
                    eprintln!("[imb] seam '{seam}' has no root node-bounds box; skip");
                    return None;
                }
            }
        }
    };
    let (mw, aw, ma) = box_width_stats(&seam_box);
    let ibp_desc = ibp_seam_box
        .as_ref()
        .map(|b| {
            let (m, a, x) = box_width_stats(b);
            format!(" | IBP-box max_w={m:.2} mean_w={a:.2} max_abs={x:.2}")
        })
        .unwrap_or_default();
    eprintln!(
        "[imb] ===== ROOT FLOOR PROPOSAL obj={obj_idx} seam='{seam}' crown_root={crown_root:.6} band_lo={band_lo:.6} seam_dim={} free_dims={} =====",
        seam_box.flatten().len(),
        super::free_input_dims(input),
    );
    eprintln!("[imb] seam box CROWN-tight: max_w={mw:.4} mean_w={aw:.4} max_abs={ma:.4}{ibp_desc}");

    // --- tail sub-graph (the prefix was already built up-front, above).
    let tail = match build_tail(graph, &seam) {
        Some(t) => t,
        None => {
            eprintln!("[imb] tail sub-graph build failed (seam not a clean cut?); skip");
            return None;
        }
    };
    // The single-root (k=1) sampling free dims — the NAIVE `lo<hi` predicate, kept
    // BYTE-IDENTICAL to the validated prop_0 run (imb_root=0.639468). FIX 1's
    // negligible-width drop applies ONLY to the region grid (`split_dims` below), so
    // the k=1 path's yseam / alpha selection is unchanged.
    let free_dims_vec: Vec<usize> = {
        let f = input.flatten();
        match (f.lower().as_slice(), f.upper().as_slice()) {
            (Some(lo), Some(hi)) => (0..lo.len()).filter(|&k| lo[k] < hi[k]).collect(),
            _ => Vec::new(),
        }
    };

    // Tail functional (p, q):
    //   NY_IMB_LOAD_PQ=1     → the numpy-validated functional (END-TO-END check),
    //   NY_IMB_TAIL_ALPHA=sample → ny's own MANIFOLD-SAMPLE-optimized functional,
    //   else                 → the box-alpha functional (auto/opt/init).
    // Every path is sound (any tail alpha yields a sound (p,q) via the coeff-err fold).
    let load_pq = matches!(std::env::var("NY_IMB_LOAD_PQ").ok().as_deref(), Some("1"));
    let tail_alpha_policy = std::env::var("NY_IMB_TAIL_ALPHA")
        .ok()
        .map(|s| s.trim().to_ascii_lowercase())
        .unwrap_or_default();
    // Per-region tail extension (STEP 1, single objective): NY_IMB_REGION_K>1 splits
    // the free box into k^free_dims regions and certifies a per-region tail functional
    // + floor, taking the GLOBAL min. Default k=1 short-circuits to the byte-identical
    // single-root path below. LOAD_PQ (measurement-only) stays single-root.
    let region_k = env_usize("NY_IMB_REGION_K", 1).max(1);
    if !load_pq && region_k > 1 {
        // FIX 1: the region grid splits/samples ONLY non-negligible-width dims (drops
        // the fixed cGAN class-label coord X_3 that a disjunctive sub-box may carry with
        // a denormal-thin width) — sound for any subset (regions still cover the full
        // box), and it halves the grid (k^4 vs k^5). SEPARATE from the k=1
        // `free_dims_vec` so the single-root path stays byte-identical (computed here so
        // k=1 never even logs it).
        let split_dims = split_free_dims(input);
        if !split_dims.is_empty() {
            return run_region_loop(
                graph,
                &tail,
                prefix.as_ref(),
                prefix_anchor.as_ref(),
                ay_seam_box_trusted,
                node_bounds,
                input,
                &seam_box,
                &split_dims,
                obj_row,
                obj_idx,
                crown_root,
                band_lo,
                region_k,
                engine,
                imb_deadline,
            );
        }
    }

    let pq = if load_pq {
        load_numpy_pq(seam_box.flatten().len())
    } else if tail_alpha_policy == "sample" {
        tail_functional_sample_opt(
            &tail,
            prefix.as_ref(),
            graph,
            node_bounds,
            &seam_box,
            input,
            &free_dims_vec,
            obj_row,
            None,
            None,
            None,
            None,
            engine,
            imb_deadline,
        )
        .map(|(p, q, _yseam)| (p, q))
    } else {
        tail_lower_functional(&tail, &seam_box, obj_row, engine, imb_deadline)
    };
    let (p, mut q) = match pq {
        Some(pq) => pq,
        None => {
            eprintln!("[imb] tail (p,q) extraction failed (mode load_pq={load_pq} policy={tail_alpha_policy:?}); skip");
            return None;
        }
    };
    eprintln!(
        "[imb] PHASE tail-functional done (cum {:.1}s)",
        imb_t0.elapsed().as_secs_f64()
    );

    // LOAD_PQ bypasses ny's sound derivation (reads (p,q) verbatim), so it is
    // MEASUREMENT-ONLY: it validates the prefix pipeline but must NEVER feed a wired
    // certified floor. The future max-wiring MUST skip `measurement_only` candidates.
    let measurement_only = load_pq;
    if load_pq {
        let p_absmax = p.iter().fold(0.0f32, |m, c| m.max(c.abs()));
        let p_l1: f64 = p.iter().map(|c| c.abs() as f64).sum();
        eprintln!(
            "[imb] LOADED (p,q) [MEASUREMENT-ONLY]: q={q:+.6} p_absmax={p_absmax:.4} |p|1={p_l1:.4}"
        );
    }

    // Optional exact tail authority. The fixed `p` remains proposal data; AY
    // optimizes the residual only to propose `q`, then a separate decision
    // model must be proved infeasible and every proof leaf replayed through
    // ny-cert before this token exists. A missing CLI MIP implementation,
    // timeout, unsupported tail op, oversized model, or proof disagreement
    // simply keeps the ordinary full-network replay path below.
    let ay_tail_certificate = if !measurement_only && exact_ay_requested && ay_seam_box_trusted {
        match super::certify_tail_with_ay(
            &tail,
            &seam_box,
            node_bounds,
            obj_row,
            &p,
            None,
            imb_deadline,
        ) {
            Some(certificate) => {
                q = certificate.q;
                eprintln!(
                    "[imb] AY-TAIL-CERT accepted q={:.9} tree_leaves={} ny_cert_replays={}",
                    certificate.q, certificate.ay_tree_leaves, certificate.ny_cert_farkas_replays
                );
                Some(certificate)
            }
            None => {
                eprintln!(
                    "[imb] AY-TAIL-CERT unavailable/inconclusive; retaining full-network \
                     replay as the only authority path"
                );
                None
            }
        }
    } else {
        if exact_ay_requested && !ay_seam_box_trusted {
            eprintln!("[imb] AY-TAIL-CERT skipped: no exact run-local seam enclosure");
        }
        None
    };

    // Diagnostic/proposal gate: samples can DISPROVE a bad `(p,q)` and save the
    // cost of building its partition, but they can never prove the universal
    // inequality and never feed a verdict.  Authority comes exclusively from
    // the later original-full-network replay over an exact-cover partition.
    let t_selfcheck = Instant::now();
    let worst_pq_slack = tail_pq_self_check(&tail, &seam_box, &p, q, obj_row, engine, imb_deadline);
    eprintln!(
        "[imb] PHASE pq-self-check: {:.1}s (cum {:.1}s)",
        t_selfcheck.elapsed().as_secs_f64(),
        imb_t0.elapsed().as_secs_f64()
    );
    let pq_tol = env_f64("NY_IMB_PQ_TOL", -1e-6) as f32;
    if !worst_pq_slack.is_finite() || worst_pq_slack < pq_tol {
        eprintln!(
            "[imb] pq diagnostic found a sampled violation: worst_slack={worst_pq_slack:.3e} < tol={pq_tol:.1e} \
             — abandoning this partition proposal"
        );
        return None;
    }

    // --- certified floor of min_x[p·h(x)].
    let spec_p = Array2::from_shape_vec((1, p.len()), p.clone()).ok()?;
    let prefix_result = prefix_bab_floor(
        prefix.as_ref(),
        input,
        &spec_p,
        engine,
        imb_deadline,
        band_lo - q,
        prefix_anchor.clone(),
        None,
        false,
    )?;
    let tail_floor = prefix_result.floor;
    let leaves_used = prefix_result.terminal_boxes.len();
    let imb_floor = if ay_tail_certificate.is_some() {
        directed_f64_lower_to_f32(f64::from(q) + f64::from(tail_floor)).unwrap_or(f32::NEG_INFINITY)
    } else {
        q + tail_floor
    };
    let full_certificate = ay_tail_certificate.and_then(|certificate| {
        certify_ay_tail_prefix_composition(
            input,
            &prefix_result.terminal_boxes,
            certificate.q,
            tail_floor,
            band_lo,
            imb_deadline,
        )
    });

    let verified = imb_floor.is_finite() && imb_floor >= band_lo;
    eprintln!(
        "[imb] obj={obj_idx} crown_root={crown_root:.6} imb_root={imb_floor:.6} band_lo={band_lo:.6} verified={verified} measurement_only={measurement_only} \
         | q={q:+.6} tail_floor={tail_floor:+.6} p_dim={} leaves={leaves_used} pq_worst_slack={worst_pq_slack:.2e}",
        p.len(),
    );
    eprintln!(
        "[imb] ===== END ROOT FLOOR: imb_root {imb_floor:.6} {} band_lo {band_lo:.6} (Δ={:+.6}) =====",
        if verified { ">=" } else { "<" },
        imb_floor - band_lo,
    );
    Some(ImbCandidate {
        obj_idx,
        imb_floor,
        threshold: band_lo,
        measurement_only,
        full_certificate,
        terminal_boxes: prefix_result.terminal_boxes,
        recheck_deadline: imb_deadline,
    })
}

/// Objective to certify: `NY_IMB_OBJ` if set + valid, else the binding objective
/// (smallest `baseline.lower − threshold`) — the constraint nearest violation,
/// which is what the numpy certificate targeted (the band-`lo` edge).
fn select_objective(objectives: &[Vec<f32>], thresholds: &[f32], baseline: &[(f32, f32)]) -> usize {
    if let Some(i) = std::env::var("NY_IMB_OBJ")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
    {
        if i < objectives.len() {
            return i;
        }
    }
    (0..baseline.len())
        .min_by(|&a, &b| {
            let ma = baseline[a].0 - thresholds[a];
            let mb = baseline[b].0 - thresholds[b];
            ma.partial_cmp(&mb).unwrap_or(Ordering::Equal)
        })
        .unwrap_or(0)
}

// ===========================================================================
// Per-region tail extension (STEP 1 — single objective)
// ===========================================================================

/// Free input dims to SPLIT / SAMPLE over: dims whose width exceeds a negligible
/// fraction of the widest dim (`NY_IMB_FREE_REL`, default `1e-3`, times `max_width`).
///
/// A truly-fixed dim (`lo==hi`) is excluded by the strict `>` regardless. A
/// denormal-thin dim — the cGAN class-label coord X_3, which numpy fixes but the
/// disjunctive sub-box can carry with a tiny nonzero width — is ALSO excluded, so
/// the region grid is `k^(true-free)` (16 for k=2, 81 for k=3) instead of doubling
/// on near-duplicate point-regions.
///
/// SOUND for region-splitting under ANY choice of split dims: `region_boxes`
/// narrows ONLY the split dims and leaves every other dim at its full `[lo,hi]` in
/// every region, so the regions still cover the whole input box and `min_r floor_r`
/// lower-bounds the true full-box min. (This only affects WHERE we branch, never the
/// per-region soundness — every region's tail anchors are concretized over the full
/// region box, un-split dims included.)
fn split_free_dims(input: &BoundedTensor) -> Vec<usize> {
    let f = input.flatten();
    let (Some(lo), Some(hi)) = (f.lower().as_slice(), f.upper().as_slice()) else {
        return Vec::new();
    };
    let mut max_w = 0.0f32;
    for (l, u) in lo.iter().zip(hi.iter()) {
        let w = u - l;
        if w > max_w {
            max_w = w;
        }
    }
    let rel = env_f64("NY_IMB_FREE_REL", 1e-3) as f32;
    let cutoff = if max_w.is_finite() && max_w > 0.0 {
        max_w * rel
    } else {
        0.0
    };
    let dims: Vec<usize> = (0..lo.len()).filter(|&k| hi[k] - lo[k] > cutoff).collect();
    let widths: Vec<f32> = dims.iter().map(|&k| hi[k] - lo[k]).collect();
    eprintln!(
        "[imb] split free_dims={dims:?} widths={widths:?} (max_w={max_w:.4} cutoff={cutoff:.2e} rel={rel:.1e})"
    );
    dims
}

/// The `k^free_dims` uniform sub-boxes over the free dims (deterministic order): the
/// full input box with the free coords restricted to each grid cell. Fixed dims
/// unchanged. Global floor = min over these regions (order-independent).
fn region_boxes(input: &BoundedTensor, free_dims: &[usize], k: usize) -> Vec<BoundedTensor> {
    let input_elements = input.lower().len();
    let Some((total, working_bytes)) =
        checked_imb_region_box_plan(input_elements, free_dims.len(), k)
    else {
        eprintln!(
            "[imb] REGION grid rejected before allocation: input_elements={input_elements} \
             free_dims={} k={k} region_cap={MAX_IMB_UNIFORM_REGIONS} \
             bytes_cap={MAX_IMB_REGION_BOX_BYTES}",
            free_dims.len()
        );
        return Vec::new();
    };
    if free_dims.iter().any(|&dim| dim >= input_elements) {
        return Vec::new();
    }
    eprintln!("[imb] REGION grid admitted: regions={total} projected_box_bytes={working_bytes}");
    let flat = input.flatten();
    let (Some(lo), Some(hi)) = (flat.lower().as_slice(), flat.upper().as_slice()) else {
        return Vec::new();
    };
    let shape = input.lower().shape().to_vec();
    let edges: Vec<Vec<f32>> = free_dims
        .iter()
        .map(|&d| {
            (0..=k)
                .map(|i| lo[d] + (hi[d] - lo[d]) * (i as f32 / k as f32))
                .collect()
        })
        .collect();
    let mut boxes = Vec::with_capacity(total);
    for idx in 0..total {
        let mut rlo = lo.to_vec();
        let mut rhi = hi.to_vec();
        let mut rem = idx;
        for (kf, &d) in free_dims.iter().enumerate() {
            let cell = rem % k;
            rem /= k;
            rlo[d] = edges[kf][cell];
            rhi[d] = edges[kf][cell + 1];
        }
        if let (Ok(la), Ok(ua)) = (
            ArrayD::from_shape_vec(IxDyn(&shape), rlo),
            ArrayD::from_shape_vec(IxDyn(&shape), rhi),
        ) {
            if let Ok(bt) = BoundedTensor::new(la, ua) {
                boxes.push(bt);
            }
        }
    }
    boxes
}

/// Manifold pq gate: forward the tail at each REGION seam sample `y` and require
/// `p·y+q ≤ obj·Y(y)`. Returns the worst slack `min_i (obj·Y(y_i) − (p·y_i+q))`
/// (must be ≥ tol). This checks over the region MANIFOLD (not the root seam box —
/// the region-tight tail anchors are only valid over the region manifold).
fn pq_check_samples(
    tail: &GraphNetwork,
    yseam: &[Vec<f32>],
    seam_shape: &[usize],
    p: &[f32],
    q: f32,
    obj_row: &[f32],
    engine: Option<&dyn GemmEngine>,
) -> f32 {
    let mut worst = f32::INFINITY;
    for y in yseam {
        if y.len() != p.len() {
            return f32::NAN;
        }
        let mut pq = q as f64;
        for (yi, pi) in y.iter().zip(p.iter()) {
            pq += *yi as f64 * *pi as f64;
        }
        let Ok(arr) = ArrayD::from_shape_vec(IxDyn(seam_shape), y.clone()) else {
            continue;
        };
        let Ok(pt) = BoundedTensor::new(arr.clone(), arr) else {
            continue;
        };
        let Ok(out) = tail.propagate_concrete_point(&pt, engine, None) else {
            continue;
        };
        let of = out.flatten();
        let ov = of.lower();
        let mut yo = 0.0f64;
        for (j, &c) in obj_row.iter().enumerate() {
            yo += c as f64 * ov.get(j).copied().unwrap_or(0.0) as f64;
        }
        let slack = (yo - pq) as f32;
        if slack < worst {
            worst = slack;
        }
    }
    worst
}

/// Certify one region: build the region tail functional (sample-opt over region seam
/// samples + region-tight tail anchors) → manifold pq gate → region prefix BaB over
/// the region box (shared root prefix anchor). Returns `(region_floor, worst_slack)`
/// or `None` (region unsafe / unavailable).
#[allow(clippy::too_many_arguments)]
fn certify_region(
    graph: &GraphNetwork,
    tail: &GraphNetwork,
    prefix: &GraphNetwork,
    node_bounds: &HashMap<String, BoundedTensor>,
    region_box: &BoundedTensor,
    root_seam_box: &BoundedTensor,
    shared_anchor: &Arc<HashMap<String, BoundedTensor>>,
    precomputed_tail: (&HashMap<String, BoundedTensor>, &GraphAlphaState),
    tail_coeffs: &HashMap<String, TailAnchorCoeff>,
    global_pool: &[(Vec<f32>, Vec<f32>)],
    obj_row: &[f32],
    free_dims: &[usize],
    band_lo: f32,
    pq_tol: f32,
    immutable_leaf_cap: Option<usize>,
    engine: Option<&dyn GemmEngine>,
    deadline: Instant,
) -> Option<RegionProposal> {
    // FIX 2: region-tight Conv_19/Conv_22 anchors by concretizing the SHARED
    // input-linear maps over the region box (no per-region crown-to-node). Empty
    // `tail_coeffs` = cascade fallback (Option A): pass `None` so
    // `tail_functional_sample_opt` builds the anchors via per-region crown-to-node.
    let region_anchors = if tail_coeffs.is_empty() {
        None
    } else {
        Some(concretize_region_anchors(tail_coeffs, region_box)?)
    };

    // FIX 3: region seam samples by FILTERING the global root pool (no per-region
    // forward). Top up with fresh region samples only if the filter is too thin.
    let mut region_ys = filter_pool_to_region(global_pool, region_box, free_dims);
    let min_keep = env_usize("NY_IMB_REGION_MIN_SAMPLES", 64);
    if region_ys.len() < min_keep {
        if let Some(mut extra) = sample_seam_points(prefix, region_box, free_dims, engine, deadline)
        {
            region_ys.append(&mut extra);
        }
    }
    if region_ys.is_empty() {
        eprintln!("[imb] region: no seam samples (empty pool + topup); skip");
        return None;
    }

    // Region tail functional (sample path) with the SHARED root prefix anchor, the
    // HOISTED (region-independent) root-seam `tail_ibp`/init alpha, the PRE-CONCRETIZED
    // region tail anchors, and the region-filtered seam pool. Sound: the SpecCrownRequest
    // relaxes on the region-tight tail anchors (valid over the region manifold). It
    // RETURNS the region seam samples so the manifold pq gate reuses them.
    let (p_r, q_r, region_yseam) = tail_functional_sample_opt(
        tail,
        prefix,
        graph,
        node_bounds,
        root_seam_box,
        region_box,
        free_dims,
        obj_row,
        Some(shared_anchor.as_ref()),
        Some(precomputed_tail),
        region_anchors.as_ref(),
        Some(region_ys),
        engine,
        deadline,
    )?;

    // Manifold sample diagnostic (reused from tail-opt). A violation rejects the
    // proposal early; a pass is not proof authority.
    let seam_shape = root_seam_box.lower().shape().to_vec();
    let ys = if region_yseam.is_empty() {
        sample_seam_points(prefix, region_box, free_dims, engine, deadline)?
    } else {
        region_yseam
    };
    let slack = pq_check_samples(tail, &ys, &seam_shape, &p_r, q_r, obj_row, engine);
    if !slack.is_finite() || slack < pq_tol {
        eprintln!(
            "[imb] region pq diagnostic found violation: slack={slack:.3e} < tol={pq_tol:.1e}"
        );
        return None;
    }

    // Region prefix BaB over the region box, reusing the shared root prefix anchor,
    // with FORCED early-exit (stop once the region floor clears band_lo).
    let spec_p = Array2::from_shape_vec((1, p_r.len()), p_r.clone()).ok()?;
    let prefix_result = prefix_bab_floor(
        prefix,
        region_box,
        &spec_p,
        engine,
        deadline,
        band_lo - q_r,
        Some(shared_anchor.clone()),
        immutable_leaf_cap,
        true,
    )?;
    Some(RegionProposal {
        floor: q_r + prefix_result.floor,
        sampled_slack: slack,
        p: p_r,
        prefix_floor: prefix_result.floor,
        terminal_boxes: prefix_result.terminal_boxes,
    })
}

/// STEP 1 region loop: certify each of the `k^free_dims` regions and take the GLOBAL
/// min floor. Any region that fails its pq gate / can't be bounded ⇒ the whole IMB
/// candidate is abandoned (`None`). Regions are independent → optionally parallel
/// (`NY_IMB_REGION_THREADS`, default 1 = sequential/reproducible).
#[allow(clippy::too_many_arguments)]
fn run_region_loop(
    graph: &GraphNetwork,
    tail: &GraphNetwork,
    prefix: &GraphNetwork,
    prepared_prefix_anchor: Option<&Arc<HashMap<String, BoundedTensor>>>,
    ay_seam_box_trusted: bool,
    node_bounds: &HashMap<String, BoundedTensor>,
    input: &BoundedTensor,
    root_seam_box: &BoundedTensor,
    free_dims: &[usize],
    obj_row: &[f32],
    obj_idx: usize,
    crown_root: f32,
    band_lo: f32,
    region_k: usize,
    engine: Option<&dyn GemmEngine>,
    deadline: Instant,
) -> Option<ImbCandidate> {
    use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

    let ay_region_authority = ay_seam_box_trusted && super::ay_tail_certificate_enabled();
    // Region-floor memo: the disjunctive driver reaches this injection twice per
    // instance with the identical (input box, objective, region_k). On the repeat
    // key, serve the whole loop from cache — skips the shared anchor, the hoisted
    // root-seam collection, and every per-region tail-opt + BaB. (Checked BEFORE the
    // anchor build so a hit reclaims all of it.)
    let memo_key = region_floor_cache_key(input, obj_row, region_k);
    if !ay_region_authority {
        if let Some(mut cached) = REGION_FLOOR_MEMO.with(|m| {
            m.borrow()
                .as_ref()
                .and_then(|(k, c)| (*k == memo_key).then(|| c.clone()))
        }) {
            eprintln!(
                "[imb] region-loop: CACHE HIT (global_floor={:.6})",
                cached.imb_floor
            );
            // The partition remains a proposal only and will be fully replayed, so
            // refreshing the resource deadline does not reuse any proof authority.
            cached.recheck_deadline = deadline;
            return Some(cached);
        }
    }

    let pq_tol = env_f64("NY_IMB_PQ_TOL", -1e-6) as f32;
    // Shared root prefix anchor — built ONCE, reused for every region (sound over
    // region ⊆ root; avoids the 160s per-region recompute trap).
    let t0 = Instant::now();
    let shared_anchor = match prepared_prefix_anchor {
        Some(anchor) => Arc::clone(anchor),
        None => build_tight_prefix_anchor_cached(
            prefix,
            input,
            engine,
            deadline,
            prefix_anchor_memo_allowed(ay_region_authority),
        )?,
    };
    eprintln!(
        "[imb] REGION shared root prefix anchor: {} nodes ({:.1}s)",
        shared_anchor.len(),
        t0.elapsed().as_secs_f64()
    );

    // Hoist the REGION-INDEPENDENT root-seam collection (BN tail bounds + init alpha
    // structure) ONCE — each region only OVERRIDES Conv_19/Conv_22 region-tight.
    let acfg = AlphaCrownConfig {
        iterations: env_usize("NY_IMB_TAIL_OPT_INNER", 20),
        deadline: Some(deadline),
        ..Default::default()
    };
    let (tail_ibp_base, init_alpha) = tail
        .collect_alpha_crown_bounds_dag_with_engine(root_seam_box, &acfg, engine)
        .ok()?;

    // FIX 2: extract the tail ReLU-source (Conv_19, Conv_22) INPUT-linear maps ONCE
    // over the ROOT box (shared root prefix anchors). Each region then concretizes
    // these in ms instead of a fresh ~6-min crown-to-node. `NY_IMB_TAIL_ANCHOR=cascade`
    // falls back to Option A (per-region crown-to-node).
    let anchor_mode = std::env::var("NY_IMB_TAIL_ANCHOR")
        .ok()
        .map(|s| s.trim().to_ascii_lowercase())
        .unwrap_or_default();
    let tail_coeffs: HashMap<String, TailAnchorCoeff> = if anchor_mode == "cascade" {
        eprintln!("[imb] NY_IMB_TAIL_ANCHOR=cascade — per-region crown-to-node (Option A)");
        HashMap::new()
    } else {
        let t1 = Instant::now();
        let tail_srcs = tail_relu_source_names(tail);
        // Tight-tail ∩-cap. The region anchors are ∩-capped to whatever caps we put
        // here (build_tail_anchor_coeffs's `run_with_linear` root box is correlation-
        // BLIND — width ~3735/51293 through the amplifying ConvTranspose generator — so
        // the cap is what actually determines tightness). Two valid root enclosures,
        // combined tighter-wins:
        //   1. `tail_ibp_base` — alpha-CROWN over the seam box, ~0.5-wide, INDEPENDENT-
        //      coord (loses the manifold correlation). Sane but not tight; alone it caps
        //      every region at ~0.52 and the region tail-opt PLATEAUS (q≈-0.33, 0.06
        //      short of prop_2's -0.268, identical at 200/800 iters — a fixed point of
        //      the ascent against the loose anchor polytope).
        //   2. `build_tight_tail_anchors` — the SAME input-correlated crown-to-node the
        //      working prop_0 ROOT path uses (Conv_19/Conv_22 ~0.065-tight, matching
        //      numpy). Computed ONCE over the root box (region-INDEPENDENT — no per-
        //      region "160s trap"). Capping at these lifts q to ~-0.268 → prop_2 clears.
        // SOUND: both are valid CROWN over-approximations over the root box; every
        // region ⊆ root; ∩ of valid enclosures only tightens, never false-UNSAT
        // (intersection_per_element falls back to the looser side on NaN/shape-mismatch).
        // Without any cap the region tail q blows to ~-14537 (garbage, prop_2 never
        // refutes).
        let root_tight = build_tight_tail_anchors(
            graph,
            input,
            node_bounds,
            prefix,
            &tail_srcs,
            Some(shared_anchor.as_ref()),
            engine,
            deadline,
        );
        let mut coeff_caps = node_bounds.clone();
        for (k, v) in tail_ibp_base.iter() {
            coeff_caps.insert(k.clone(), v.clone());
        }
        // Override the tail sources with the tighter input-correlated anchors (∩ where
        // both are present so we never loosen).
        for (k, v) in root_tight.iter() {
            let capped = match coeff_caps.get(k) {
                Some(existing) => existing
                    .intersection_per_element(v)
                    .map(|(t, _)| t)
                    .unwrap_or_else(|| v.clone()),
                None => v.clone(),
            };
            coeff_caps.insert(k.clone(), capped);
        }
        match build_tail_anchor_coeffs(
            graph,
            input,
            &coeff_caps,
            shared_anchor.as_ref(),
            &tail_srcs,
            engine,
            deadline,
        ) {
            Some(c) => {
                eprintln!(
                    "[imb] REGION tail-anchor coeffs: {} sources ({:.1}s)",
                    c.len(),
                    t1.elapsed().as_secs_f64()
                );
                c
            }
            None => {
                eprintln!("[imb] coeff extraction failed; abandoning region candidate");
                return None;
            }
        }
    };

    // FIX 3: the GLOBAL seam-sample pool over the ROOT box — sampled ONCE, filtered
    // per region (no per-region forward).
    let global_pool = sample_seam_pool(prefix, input, free_dims, engine, deadline)?;

    let regions = region_boxes(input, free_dims, region_k);
    if regions.is_empty() {
        eprintln!("[imb] REGION loop: no regions built; skip");
        return None;
    }
    let immutable_leaf_cap = if ay_region_authority {
        let per_region = (MAX_AY_REGION_TOTAL_LEAVES / regions.len()).max(1);
        let total = per_region.checked_mul(regions.len())?;
        let bytes = checked_ay_prefix_frontier_bytes(input.lower().len(), total)?;
        eprintln!(
            "[imb] AY-TAIL-CERT prefix frontier admitted: per_region_leaves={per_region} \
             total_leaves={total} projected_bytes={bytes}"
        );
        Some(per_region)
    } else {
        None
    };
    let region_threads = env_usize("NY_IMB_REGION_THREADS", 1).max(1);
    eprintln!(
        "[imb] ===== REGION LOOP obj={obj_idx} crown_root={crown_root:.6} band_lo={band_lo:.6} k={region_k} R={} free_dims={:?} region_threads={region_threads} ====="
        , regions.len(), free_dims
    );

    // Per-region completion diagnostics: a shared atomic sequence number (order regions
    // FINISH, not their index) + the wall since loop start. If the region par_iter runs
    // TRULY N-way, the first `region_threads` completions cluster at ~the same wall; if
    // it is (still) serial, they step by ~Δ each. `leaves` is the region's BaB leaf count.
    let region_t0 = Instant::now();
    let region_done = std::sync::atomic::AtomicUsize::new(0);
    let n_regions = regions.len();
    let certify = |rb: &BoundedTensor| -> Option<RegionProposal> {
        let res = certify_region(
            graph,
            tail,
            prefix,
            node_bounds,
            rb,
            root_seam_box,
            &shared_anchor,
            (&tail_ibp_base, &init_alpha),
            &tail_coeffs,
            &global_pool,
            obj_row,
            free_dims,
            band_lo,
            pq_tol,
            immutable_leaf_cap,
            engine,
            deadline,
        );
        let k = region_done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        let leaves = res.as_ref().map(|r| r.terminal_boxes.len()).unwrap_or(0);
        eprintln!(
            "[imb] region {k}/{n_regions} done wall={:.1}s leaves={leaves}",
            region_t0.elapsed().as_secs_f64()
        );
        res
    };

    let results: Vec<Option<RegionProposal>> = if region_threads > 1 {
        match rayon::ThreadPoolBuilder::new()
            .num_threads(region_threads)
            .build()
        {
            Ok(pool) => {
                // Force every rayon fan-out INSIDE each region (tail-opt, sampling, conv
                // backward, leaf-BaB) to run SERIALLY, so the N region workers are the
                // ONLY parallelism → true N-way concurrency (no nested-fan-out starvation).
                let _seq_inner = super::RegionSeqGuard::enable();
                eprintln!(
                    "[imb] region-loop: pool installed threads={region_threads} (inner rayon seq-guarded)"
                );
                pool.install(|| regions.par_iter().map(&certify).collect())
            }
            Err(e) => {
                eprintln!(
                    "[imb] region-loop: pool build failed ({e}); running regions SEQUENTIALLY"
                );
                regions.iter().map(&certify).collect()
            }
        }
    } else {
        regions.iter().map(certify).collect()
    };
    {
        let wall = region_t0.elapsed().as_secs_f64();
        let per = wall / regions.len().max(1) as f64;
        eprintln!(
            "[imb] region-loop: R={} threads={region_threads} wall={wall:.1}s ({per:.2}s/region)",
            regions.len()
        );
    }

    let mut proposals = Vec::with_capacity(results.len());
    for (i, res) in results.into_iter().enumerate() {
        match res {
            Some(region) => proposals.push(region),
            None => {
                eprintln!("[imb] REGION {i} failed (unsafe/unbounded) — abandoning IMB candidate");
                return None;
            }
        }
    }
    let global_floor = proposals
        .iter()
        .map(|region| region.floor)
        .fold(f32::INFINITY, f32::min);
    let worst_slack = proposals
        .iter()
        .map(|region| region.sampled_slack)
        .fold(f32::INFINITY, f32::min);

    // Exact AY authority is a second, sequential phase. Each region's prefix
    // frontier already proves `p·h(x) >= prefix_floor`; install precisely that
    // fact as one reachability row in the fresh root-seam tail MILP, then prove
    // the ORIGINAL objective. This preserves the proposal's one-dimensional
    // seam correlation instead of independently minimizing a residual and
    // `p·h(x)` at unrelated seam points. One missing proof invalidates the
    // entire authority token while leaving ordinary full-network replay intact.
    // `NY_IMB_AY_REGION_PROOF=residual` retains the previous formulation only
    // for explicit A/B diagnostics. `affine` opts into the existing regional
    // K=2 envelope. `shared` opts into one root-valid K4 support bank and one
    // global exact root model. Any other value, including unset, preserves the
    // scalar reachability default byte-for-byte.
    let region_proof_mode = std::env::var("NY_IMB_AY_REGION_PROOF").ok();
    let legacy_residual_proof = region_proof_mode.as_deref() == Some("residual");
    let affine_reachability_proof = region_proof_mode.as_deref() == Some("affine");
    let shared_input_reachability_proof = region_proof_mode.as_deref() == Some("shared");
    let ay_region_lower = if ay_region_authority {
        let certified = if input.has_l2_constraint() {
            eprintln!(
                "[imb] AY-TAIL-CERT region partition rejects L2-annotated input; \
                 retaining full-network replay authority"
            );
            None
        } else if legacy_residual_proof {
            eprintln!("[imb] AY-TAIL-CERT region proof mode=residual (explicit A/B)");
            certify_ay_region_partition_with(
                input,
                &regions,
                &proposals,
                band_lo,
                deadline,
                |region_idx, _region_box, p, required_q, proof_deadline| {
                    if !replay_deadline_open(proof_deadline) {
                        return None;
                    }
                    let certificate = super::certify_tail_with_ay(
                        tail,
                        root_seam_box,
                        node_bounds,
                        obj_row,
                        p,
                        Some(required_q),
                        proof_deadline,
                    )?;
                    eprintln!(
                        "[imb] AY-TAIL-CERT residual region {}/{} accepted q={:.9} \
                         tree_leaves={} ny_cert_replays={}",
                        region_idx + 1,
                        regions.len(),
                        certificate.q(),
                        certificate.ay_tree_leaves(),
                        certificate.ny_cert_farkas_replays(),
                    );
                    Some(certificate.q())
                },
            )
        } else if shared_input_reachability_proof {
            eprintln!(
                "[imb] AY-TAIL-CERT region proof mode=shared \
                 (root-valid evidence-backed K=4 canary)"
            );
            // Validate the complete region cover before spending any of the
            // strict root-bank construction slice.
            if let Err(reason) = validate_binary_partition_cover(input, &regions, deadline) {
                eprintln!(
                    "[imb] AY-TAIL-CERT shared region grid rejected before bank build: {reason}"
                );
                None
            } else {
                prefix_shared_input_reachability_envelope(
                    prefix,
                    input,
                    &regions,
                    &proposals,
                    shared_anchor.as_ref(),
                    engine,
                    deadline,
                )
                .and_then(|envelope| {
                    certify_ay_shared_input_root_with(
                        input,
                        &regions,
                        &envelope,
                        band_lo,
                        deadline,
                        |envelope, requested_lower, proof_deadline| {
                            if !replay_deadline_open(proof_deadline) {
                                return None;
                            }
                            let certificate =
                                super::certify_tail_with_ay_shared_input_reachability(
                                    tail,
                                    root_seam_box,
                                    obj_row,
                                    envelope,
                                    requested_lower,
                                    proof_deadline,
                                )?;
                            eprintln!(
                                "[imb] AY-TAIL-CERT shared global root accepted \
                                 proposals={} supports={} latent={} original_lower={:.9} \
                                 tree_leaves={} ny_cert_replays={}",
                                regions.len(),
                                envelope.directions().nrows(),
                                envelope.region_input().flatten().len(),
                                certificate.lower(),
                                certificate.ay_tree_leaves(),
                                certificate.ny_cert_farkas_replays(),
                            );
                            Some(certificate.lower())
                        },
                    )
                })
            }
        } else if affine_reachability_proof {
            eprintln!(
                "[imb] AY-TAIL-CERT region proof mode=affine \
                 (K={} shared prefix inputs)",
                super::AY_TAIL_AFFINE_REACHABILITY_ROWS
            );
            // Construct every regional envelope before the first exact worker
            // is admitted. A malformed/expired late region therefore cannot
            // waste or partially authorize successful early-region proofs.
            let envelopes = (|| {
                let mut envelopes = Vec::with_capacity(regions.len());
                for region_idx in 0..regions.len() {
                    let (directions, second_idx) = k2_support_directions(&proposals, region_idx)?;
                    let envelope = prefix_affine_reachability_envelope(
                        prefix,
                        &regions[region_idx],
                        directions,
                        shared_anchor.as_ref(),
                        free_dims,
                        engine,
                        deadline,
                    )?;
                    eprintln!(
                        "[imb] AY-TAIL-CERT affine region {}/{} support rows=[{},{}]",
                        region_idx + 1,
                        regions.len(),
                        region_idx,
                        second_idx,
                    );
                    envelopes.push(envelope);
                }
                Some(envelopes)
            })();
            envelopes.and_then(|envelopes| {
                certify_ay_region_affine_reachability_partition_with(
                    input,
                    &regions,
                    &envelopes,
                    band_lo,
                    deadline,
                    |region_idx, envelope, requested_lower, proof_deadline| {
                        if !replay_deadline_open(proof_deadline) {
                            return None;
                        }
                        let certificate = super::certify_tail_with_ay_affine_reachability(
                            tail,
                            root_seam_box,
                            obj_row,
                            envelope,
                            requested_lower,
                            proof_deadline,
                        )?;
                        eprintln!(
                            "[imb] AY-TAIL-CERT affine region {}/{} accepted \
                             supports={} latent={} original_lower={:.9} \
                             tree_leaves={} ny_cert_replays={}",
                            region_idx + 1,
                            regions.len(),
                            envelope.directions().nrows(),
                            envelope.region_input().flatten().len(),
                            certificate.lower(),
                            certificate.ay_tree_leaves(),
                            certificate.ny_cert_farkas_replays(),
                        );
                        Some(certificate.lower())
                    },
                )
            })
        } else {
            eprintln!("[imb] AY-TAIL-CERT region proof mode=reachability");
            certify_ay_region_reachability_partition_with(
                input,
                &regions,
                &proposals,
                band_lo,
                deadline,
                |region_idx, _region_box, p, prefix_lower, requested_lower, proof_deadline| {
                    if !replay_deadline_open(proof_deadline) {
                        return None;
                    }
                    let certificate = super::certify_tail_with_ay_reachability(
                        tail,
                        root_seam_box,
                        obj_row,
                        p,
                        prefix_lower,
                        requested_lower,
                        proof_deadline,
                    )?;
                    eprintln!(
                        "[imb] AY-TAIL-CERT reachability region {}/{} accepted \
                         prefix_lower={:.9} original_lower={:.9} \
                         tree_leaves={} ny_cert_replays={}",
                        region_idx + 1,
                        regions.len(),
                        certificate.prefix_lower(),
                        certificate.lower(),
                        certificate.ay_tree_leaves(),
                        certificate.ny_cert_farkas_replays(),
                    );
                    Some(certificate.lower())
                },
            )
        };
        if certified.is_none() {
            if shared_input_reachability_proof {
                eprintln!(
                    "[imb] AY-TAIL-CERT shared global-root proof unavailable/inconclusive; \
                     retaining full-network replay as the only authority path"
                );
            } else {
                let proof_mode = if legacy_residual_proof {
                    "residual"
                } else if affine_reachability_proof {
                    "affine"
                } else {
                    "reachability"
                };
                eprintln!(
                    "[imb] AY-TAIL-CERT {proof_mode} region partition unavailable/inconclusive; \
                     retaining full-network replay as the only authority path"
                );
            }
        }
        certified
    } else {
        None
    };

    let terminal_boxes: Vec<BoundedTensor> = proposals
        .into_iter()
        .flat_map(|region| region.terminal_boxes)
        .collect();
    let full_certificate = ay_region_lower.and_then(|lower| {
        certify_ay_region_global_composition(input, &terminal_boxes, lower, band_lo, deadline)
    });
    let verified = global_floor.is_finite() && global_floor >= band_lo;
    eprintln!(
        "[imb] obj={obj_idx} crown_root={crown_root:.6} imb_root={global_floor:.6} band_lo={band_lo:.6} verified={verified} measurement_only=false | R={} regions worst_pq_slack={worst_slack:.2e}",
        regions.len()
    );
    eprintln!(
        "[imb] ===== END REGION LOOP: global imb_root {global_floor:.6} {} band_lo {band_lo:.6} (Δ={:+.6}) =====",
        if verified { ">=" } else { "<" },
        global_floor - band_lo,
    );
    let candidate = ImbCandidate {
        obj_idx,
        imb_floor: global_floor,
        threshold: band_lo,
        measurement_only: false,
        full_certificate,
        terminal_boxes,
        recheck_deadline: deadline,
    };
    // Cache the completed candidate so the driver's 2nd same-key injection is a hit.
    // Exact AY authority is intentionally never cached or deadline-refreshed:
    // every gated invocation must reconstruct its exact proof path atomically.
    if !ay_region_authority {
        REGION_FLOOR_MEMO.with(|m| *m.borrow_mut() = Some((memo_key, candidate.clone())));
    }
    Some(candidate)
}

// ===========================================================================
// Seam resolution
// ===========================================================================

fn is_layer_relu(graph: &GraphNetwork, name: &str) -> bool {
    matches!(
        graph.nodes.get(name).map(|n| &n.layer),
        Some(Layer::ReLU(_))
    )
}

fn is_layer_convtranspose(graph: &GraphNetwork, name: &str) -> bool {
    matches!(
        graph.nodes.get(name).map(|n| &n.layer),
        Some(Layer::ConvTranspose2d(_)) | Some(Layer::ConvTranspose1d(_))
    )
}

/// Resolve the seam node: explicit `NY_IMB_SEAM` (if it names a real node), else
/// the auto-picker.
fn resolve_seam(graph: &GraphNetwork) -> Option<String> {
    if let Ok(s) = std::env::var("NY_IMB_SEAM") {
        let s = s.trim().to_string();
        if !s.is_empty() {
            if graph.nodes.contains_key(&s) {
                return Some(s);
            }
            eprintln!("[imb] NY_IMB_SEAM='{s}' not a graph node; falling back to auto-pick");
        }
    }
    autopick_seam(graph)
}

/// Auto-pick the seam: the ReLU node whose ancestor set covers EVERY ConvTranspose
/// (the whole generator sits in the prefix) and that has exactly
/// `NY_IMB_TAIL_RELUS` (default 2) ReLU descendants. For cGAN this is uniquely
/// Relu_17 (tail ReLUs = {Relu_20, Relu_23}); the earliest such ReLU wins on ties.
fn autopick_seam(graph: &GraphNetwork) -> Option<String> {
    let exec = graph.exec_order().ok()?;
    let ancestors = graph.all_ancestors().ok()?;
    let want_tail = env_usize("NY_IMB_TAIL_RELUS", 2);

    let ct_nodes: Vec<String> = exec
        .iter()
        .filter(|n| is_layer_convtranspose(graph, n))
        .cloned()
        .collect();
    let relu_nodes: Vec<String> = exec
        .iter()
        .filter(|n| is_layer_relu(graph, n))
        .cloned()
        .collect();

    for r in &relu_nodes {
        let anc = match ancestors.get(r) {
            Some(a) => a,
            None => continue,
        };
        let covers_generator = ct_nodes.iter().all(|ct| anc.iter().any(|a| a == ct));
        if !covers_generator {
            continue;
        }
        let desc = match graph.descendants_inclusive(std::slice::from_ref(r)) {
            Ok(d) => d,
            Err(_) => continue,
        };
        let tail_relus = relu_nodes
            .iter()
            .filter(|rr| rr.as_str() != r.as_str() && desc.contains(*rr))
            .count();
        if tail_relus == want_tail {
            eprintln!("[imb] auto-picked seam='{r}' (tail ReLUs={tail_relus}, generator covered)");
            return Some(r.clone());
        }
    }
    None
}

// ===========================================================================
// Sub-graph construction
// ===========================================================================

/// Sub-graph from the network input to `target`: retain ancestors(`target`) ∪
/// {`target`} with their ORIGINAL input references, output = `target`. Every node's
/// inputs stay valid (ancestors are closed under the input relation), so the CROWN
/// backward over this sub-graph is w.r.t. the SAME network input as the full graph
/// — the basis for extracting `target`'s input-linear map (`build_tail_anchor_coeffs`).
fn build_subgraph_to_node(graph: &GraphNetwork, target: &str) -> Option<GraphNetwork> {
    let ancestors = graph.all_ancestors().ok()?;
    let retained = ancestors.get(target)?; // topological order, includes target
    let mut sub = GraphNetwork::new();
    for name in retained {
        let node = graph.nodes.get(name)?;
        sub.try_add_node(GraphNode::new(
            node.name().to_string(),
            node.layer().clone(),
            node.inputs().to_vec(),
        ))
        .ok()?;
    }
    sub.set_output(target);
    // Generous per-node CROWN-IBP budget: the ConvTranspose generator's 28,800-dim
    // nodes exceed the default 12 s cap (cgan's BatchNormalization_11 alone needs
    // ~143 s for a full collection), which would silently fall back to IBP and
    // re-loosen the ReLU relaxations — exactly the failure the `crown`/`crown_root`
    // modes exist to avoid. Raise the cap (`NY_IMB_PREFIX_CAP_S`, default 60 s) so
    // the generator nodes complete CROWN.
    let cap = env_f64("NY_IMB_PREFIX_CAP_S", 60.0);
    if cap > 0.0 {
        sub.set_crown_ibp_per_node_time_budget(crate::types::CrownIbpPerNodeTimeBudget {
            floor_secs: None,
            cap_secs: Some(cap),
        });
    }
    Some(sub)
}

/// Prefix sub-graph: retain ancestors(seam) ∪ {seam}, output = seam. Kept EXACT
/// (no affine collapse); the BaB bounds `p·h(x)` per leaf over it.
fn build_prefix(graph: &GraphNetwork, seam: &str) -> Option<GraphNetwork> {
    build_subgraph_to_node(graph, seam)
}

/// Tail sub-graph: retain descendants(seam) \ {seam}, rewrite every `seam` input
/// reference to NETWORK_INPUT (the seam value becomes the tail's input), output
/// unchanged. `None` if the seam is not a clean cut (a tail node references an
/// un-retained upstream node → `try_add_node` dangling-reference error).
fn build_tail(graph: &GraphNetwork, seam: &str) -> Option<GraphNetwork> {
    let seam_owned = seam.to_string();
    let desc = graph
        .descendants_inclusive(std::slice::from_ref(&seam_owned))
        .ok()?;
    let exec = graph.exec_order().ok()?;
    let output = graph.output_node.clone();
    let mut sub = GraphNetwork::new();
    for name in exec {
        if name.as_str() == seam || !desc.contains(name) {
            continue;
        }
        let node = graph.nodes.get(name)?;
        let inputs: Vec<String> = node
            .inputs()
            .iter()
            .map(|i| {
                if i.as_str() == seam {
                    NETWORK_INPUT.to_string()
                } else {
                    i.clone()
                }
            })
            .collect();
        sub.try_add_node(GraphNode::new(
            node.name().to_string(),
            node.layer().clone(),
            inputs,
        ))
        .ok()?;
    }
    sub.set_output(output);
    Some(sub)
}

// ===========================================================================
// Tail: alpha-optimized affine lower functional (p, q)
// ===========================================================================

/// Build the certified input-linear lower functional `p·y+q ≤ Y_o(y)` over the
/// seam box, with the tail ReLUs alpha-optimized for the objective and the
/// certified coeff error folded OUTWARD into `q`.
fn tail_lower_functional(
    tail: &GraphNetwork,
    seam_box: &BoundedTensor,
    obj_row: &[f32],
    engine: Option<&dyn GemmEngine>,
    deadline: Instant,
) -> Option<(Vec<f32>, f32)> {
    let acfg = AlphaCrownConfig {
        iterations: env_usize("NY_IMB_ALPHA_ITERS", 20),
        deadline: Some(deadline),
        ..Default::default()
    };

    // Reference/IBP intermediate bounds + an initialized+warmed alpha state.
    let (tail_ibp, init_alpha) = tail
        .collect_alpha_crown_bounds_dag_with_engine(seam_box, &acfg, engine)
        .ok()?;
    // Specialize alpha for the single objective row (the single-box root
    // spec_guided_lower target — condition (A): tail ReLU slopes MUST be
    // alpha-optimized). SPSA may UNDER-SHOOT or even regress the thin margin, so
    // we keep BOTH the warmed init alpha and the optimized one and pick whichever
    // gives the tighter box-concretized tail-output lower (sound: both are valid
    // CROWN relaxations; picking the tighter never over-claims).
    let opt_alpha = tail
        .optimize_alpha_for_spec_objective(seam_box, &tail_ibp, &init_alpha, &acfg, obj_row, engine)
        .ok();

    let spec = Array2::from_shape_vec((1, obj_row.len()), obj_row.to_vec()).ok()?;

    let tail_out_init = tail_crown_out_lower(
        tail,
        seam_box,
        &spec,
        engine,
        &tail_ibp,
        &init_alpha,
        deadline,
    );
    let tail_out_opt = opt_alpha
        .as_ref()
        .and_then(|a| tail_crown_out_lower(tail, seam_box, &spec, engine, &tail_ibp, a, deadline));
    // AUTO heuristic: pick the alpha with the higher box-concretized tail-output
    // lower. NOTE: this seam-box criterion is only a proxy — the IMB floor is the
    // MANIFOLD min `q + min_x p·h`, not the independent-coordinate seam-box min, so
    // `NY_IMB_TAIL_ALPHA` ∈ {auto (default), opt, init} force-selects the functional
    // so the two can be measured empirically (numpy validated that alpha-OPT is what
    // clears band_lo). Both alphas are sound; the choice only affects tightness.
    let auto_use_opt = match (tail_out_opt, tail_out_init) {
        (Some(o), Some(i)) => o > i,
        (Some(_), None) => true,
        _ => false,
    };
    let policy = std::env::var("NY_IMB_TAIL_ALPHA")
        .ok()
        .map(|s| s.trim().to_ascii_lowercase())
        .unwrap_or_else(|| "auto".to_string());
    let use_opt = match policy.as_str() {
        "opt" => opt_alpha.is_some(), // force opt when available, else fall back to init
        "init" => false,
        _ => auto_use_opt,
    };
    let chosen_alpha = if use_opt {
        opt_alpha.unwrap_or_else(|| init_alpha.clone())
    } else {
        init_alpha.clone()
    };
    eprintln!(
        "[imb] tail alpha: unstable_relus={} tail_out_lower init={} opt={} policy={policy} chosen={}",
        init_alpha.num_unstable(),
        tail_out_init
            .map(|v| format!("{v:.6}"))
            .unwrap_or_else(|| "n/a".into()),
        tail_out_opt
            .map(|v| format!("{v:.6}"))
            .unwrap_or_else(|| "n/a".into()),
        if use_opt { "opt" } else { "init" },
    );

    let (_bounds, lin_opt) = SpecCrownRequest::new(tail, seam_box, &spec, engine)
        .node_bounds(&tail_ibp)
        .alpha_state_opt(Some(&chosen_alpha))
        .deadline_opt(Some(deadline))
        .run_with_linear()
        .ok()?;
    let mut lin = lin_opt?;
    let q_raw = lin.lower_b().iter().next().copied().unwrap_or(f32::NAN);
    // Fold the certified coeff error outward into the bias over the seam box, so
    // the STORED p is exact and q absorbs the uncertainty (`p·y + q ≤ Y_o`).
    lin.fold_coeff_err_over_box_eager(seam_box);
    let p: Vec<f32> = lin.lower_a().row(0).to_vec();
    let q = *lin.lower_b().iter().next()?;
    let p_absmax = p.iter().fold(0.0f32, |m, c| m.max(c.abs()));
    eprintln!("[imb] tail (p,q): q_raw={q_raw:+.6} q_folded={q:+.6} p_absmax={p_absmax:.4}");
    if !q.is_finite() || p.iter().any(|c| !c.is_finite()) {
        return None;
    }
    Some((p, q))
}

// ===========================================================================
// Manifold-sample-guided tail optimizer (numpy `build_functional`, item A)
// ===========================================================================

/// xorshift64 step + a `[0,1)` f32 draw / a ±1 draw.
fn xs64(rng: &mut u64) -> u64 {
    *rng ^= *rng << 13;
    *rng ^= *rng >> 7;
    *rng ^= *rng << 17;
    *rng
}
fn rand01(rng: &mut u64) -> f32 {
    ((xs64(rng) >> 40) as f32) / ((1u32 << 24) as f32)
}

/// Forward the prefix (input → seam) at a grid + random cloud of inputs over the
/// free dims (fixed dims held) to get the reachable-seam MANIFOLD as flat points.
fn sample_seam_points(
    prefix: &GraphNetwork,
    input: &BoundedTensor,
    free_dims: &[usize],
    engine: Option<&dyn GemmEngine>,
    deadline: Instant,
) -> Option<Vec<Vec<f32>>> {
    let f = input.flatten();
    let lo = f.lower().as_slice()?.to_vec();
    let hi = f.upper().as_slice()?.to_vec();
    let shape = input.lower().shape().to_vec();
    let nfree = free_dims.len();

    let mut inputs: Vec<Vec<f32>> = Vec::new();
    // Coarse grid (cartesian product over the free dims, capped).
    let grid = env_usize("NY_IMB_SAMPLE_GRID", 5).max(1);
    let grid_cap = env_usize("NY_IMB_SAMPLE_GRID_MAX", 4096);
    let grid_total = (0..nfree)
        .fold(1usize, |a, _| a.saturating_mul(grid))
        .min(grid_cap);
    for idx in 0..grid_total {
        let mut x = lo.clone(); // fixed dims already at lo (== hi)
        let mut rem = idx;
        for &d in free_dims {
            let gi = rem % grid;
            rem /= grid;
            let t = if grid > 1 {
                gi as f32 / (grid - 1) as f32
            } else {
                0.5
            };
            x[d] = lo[d] + (hi[d] - lo[d]) * t;
        }
        inputs.push(x);
    }
    // Random cloud.
    let mut rng: u64 = 0x9e37_79b9_7f4a_7c15;
    let n_rand = env_usize("NY_IMB_SAMPLES", 2000);
    for _ in 0..n_rand {
        let mut x = lo.clone();
        for &d in free_dims {
            x[d] = lo[d] + (hi[d] - lo[d]) * rand01(&mut rng);
        }
        inputs.push(x);
    }

    // Forward each input through the prefix (output node = seam), in PARALLEL. The
    // forwards are independent and, at ~0.05 s each through the ConvTranspose
    // generator × several thousand samples, dominate the tail-opt setup (minutes
    // serially — enough to eat the whole EARLY fast-path budget before the leaf-BaB
    // even starts). Rayon over the global pool cuts it to ~cores× faster; faer stays
    // Seq-guarded inside the workers, so no oversubscription.
    use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
    if Instant::now() >= deadline {
        return None;
    }
    let forward = |x: &Vec<f32>| -> Option<Vec<f32>> {
        let arr = ArrayD::from_shape_vec(IxDyn(&shape), x.clone()).ok()?;
        let pt = BoundedTensor::new(arr.clone(), arr).ok()?;
        let out = prefix.propagate_concrete_point(&pt, engine, None).ok()?;
        Some(out.flatten().lower().as_slice()?.to_vec())
    };
    // Region-parallel path: forward SERIALLY (this region is one of the N concurrent
    // workers; a nested par_iter here would fan out on the region pool).
    let yseam: Option<Vec<Vec<f32>>> = if crate::imb::region_seq_inner() {
        inputs.iter().map(forward).collect()
    } else {
        inputs.par_iter().map(forward).collect()
    };
    yseam
}

/// FIX 3 — a GLOBAL seam-sample pool over the ROOT box: forward `NY_IMB_POOL`
/// (default 4000) random inputs (+ the coarse grid) through the prefix ONCE, keeping
/// each `(input coords, seam point)` PAIR. Regions then FILTER this pool to inputs
/// inside their box instead of re-forwarding ~2000 points per region (numpy's trick).
/// Fixed / un-split dims are held at `lo` (identical to `sample_seam_points`), so a
/// region only ever filters on the split dims.
fn sample_seam_pool(
    prefix: &GraphNetwork,
    input: &BoundedTensor,
    free_dims: &[usize],
    engine: Option<&dyn GemmEngine>,
    deadline: Instant,
) -> Option<Vec<(Vec<f32>, Vec<f32>)>> {
    let f = input.flatten();
    let lo = f.lower().as_slice()?.to_vec();
    let hi = f.upper().as_slice()?.to_vec();
    let shape = input.lower().shape().to_vec();
    let nfree = free_dims.len();

    let mut inputs: Vec<Vec<f32>> = Vec::new();
    let grid = env_usize("NY_IMB_SAMPLE_GRID", 5).max(1);
    let grid_cap = env_usize("NY_IMB_SAMPLE_GRID_MAX", 4096);
    let grid_total = (0..nfree)
        .fold(1usize, |a, _| a.saturating_mul(grid))
        .min(grid_cap);
    for idx in 0..grid_total {
        let mut x = lo.clone();
        let mut rem = idx;
        for &d in free_dims {
            let gi = rem % grid;
            rem /= grid;
            let t = if grid > 1 {
                gi as f32 / (grid - 1) as f32
            } else {
                0.5
            };
            x[d] = lo[d] + (hi[d] - lo[d]) * t;
        }
        inputs.push(x);
    }
    let mut rng: u64 = 0x9e37_79b9_7f4a_7c15;
    let n_rand = env_usize("NY_IMB_POOL", 4000);
    for _ in 0..n_rand {
        let mut x = lo.clone();
        for &d in free_dims {
            x[d] = lo[d] + (hi[d] - lo[d]) * rand01(&mut rng);
        }
        inputs.push(x);
    }

    // Forward in PARALLEL (independent; ~0.05 s each through the generator).
    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    if Instant::now() >= deadline {
        return None;
    }
    let pool: Option<Vec<(Vec<f32>, Vec<f32>)>> = inputs
        .into_par_iter()
        .map(|x| {
            let arr = ArrayD::from_shape_vec(IxDyn(&shape), x.clone()).ok()?;
            let pt = BoundedTensor::new(arr.clone(), arr).ok()?;
            let out = prefix.propagate_concrete_point(&pt, engine, None).ok()?;
            let y = out.flatten().lower().as_slice()?.to_vec();
            Some((x, y))
        })
        .collect();
    let pool = pool?;
    eprintln!(
        "[imb] global seam pool: {} points over root box",
        pool.len()
    );
    Some(pool)
}

/// Filter the global seam pool to inputs inside `region_box` (on the split dims) and
/// return their seam points. Purely a subset of a sound pool — no new forwards.
fn filter_pool_to_region(
    pool: &[(Vec<f32>, Vec<f32>)],
    region_box: &BoundedTensor,
    free_dims: &[usize],
) -> Vec<Vec<f32>> {
    let rf = region_box.flatten();
    let (Some(rl), Some(ru)) = (rf.lower().as_slice(), rf.upper().as_slice()) else {
        return Vec::new();
    };
    pool.iter()
        .filter(|(x, _)| {
            free_dims
                .iter()
                .all(|&d| d < x.len() && x[d] >= rl[d] && x[d] <= ru[d])
        })
        .map(|(_, y)| y.clone())
        .collect()
}

/// RAW (unfolded) tail lower functional `(p, q)` for a given alpha — the cheap
/// score-eval kernel (one tail CROWN backward with fixed intermediates).
fn build_pq_raw(
    tail: &GraphNetwork,
    seam_box: &BoundedTensor,
    tail_ibp: &HashMap<String, BoundedTensor>,
    alpha: &GraphAlphaState,
    spec: &Array2<f32>,
    engine: Option<&dyn GemmEngine>,
    deadline: Instant,
) -> Option<(Vec<f32>, f32)> {
    let (_b, lin) = SpecCrownRequest::new(tail, seam_box, spec, engine)
        .node_bounds(tail_ibp)
        .alpha_state_opt(Some(alpha))
        .deadline_opt(Some(deadline))
        .run_with_linear()
        .ok()?;
    let lin = lin?;
    let p = lin.lower_a().row(0).to_vec();
    let q = *lin.lower_b().iter().next()?;
    Some((p, q))
}

/// Item A — the manifold-sample-guided tail functional (`NY_IMB_TAIL_ALPHA=sample`).
///
/// ny's box-alpha objective (seam-BOX min) is pessimistic (`q≈0.6446`); the IMB floor
/// lives on the reachable-seam MANIFOLD, so this maximizes `min_i (p·yseam_i + q)`.
/// It ports numpy's `build_functional` cutting-plane: each round, find the BINDING
/// manifold seam point `ystar = argmin_i (p·yseam_i + q)`, then run ny's per-objective
/// alpha optimizer AT `ystar` — a point seam box with the FULL-box `tail_ibp` as the
/// relaxation anchors, so the tail ReLUs stay UNSTABLE and alpha has a real gradient.
/// This tightens the functional exactly where it binds (numpy's worst-sample ascent),
/// using ny's own per-neuron backward adjoint gradient — NOT the sample-inefficient
/// SPSA-on-1500-alphas. Warm-started from the init alpha; best-by-sample-min kept.
///
/// SOUNDNESS: EVERY alpha in [0,1] yields a sound `(p,q)` (`p·y+q ≤ Y_o(y)` over the
/// seam box) via the same sound tail CROWN + `fold_coeff_err_over_box_eager` — alpha
/// only selects WHICH sound functional. The samples only GUIDE the choice; the
/// certified floor is still the prefix BaB, and the universal pq self-check gates it.
#[allow(clippy::too_many_arguments)]
fn tail_functional_sample_opt(
    tail: &GraphNetwork,
    prefix: &GraphNetwork,
    graph: &GraphNetwork,
    node_bounds: &HashMap<String, BoundedTensor>,
    seam_box: &BoundedTensor,
    input: &BoundedTensor,
    free_dims: &[usize],
    obj_row: &[f32],
    prefix_anchor: Option<&HashMap<String, BoundedTensor>>,
    precomputed_tail: Option<(&HashMap<String, BoundedTensor>, &GraphAlphaState)>,
    region_tail_anchors: Option<&HashMap<String, BoundedTensor>>,
    precomputed_yseam: Option<Vec<Vec<f32>>>,
    engine: Option<&dyn GemmEngine>,
    deadline: Instant,
) -> Option<(Vec<f32>, f32, Vec<Vec<f32>>)> {
    // The single-root (R=1) path serves the 2nd disjunctive-root injection from the
    // TAIL_PQ_MEMO cache. The per-region path (precomputed_tail=Some) bypasses this
    // cache (the region loop is cached at a coarser level; the single-slot memo would
    // just thrash across regions), and it hoists the region-independent
    // `collect_alpha_crown_bounds_dag(root_seam_box)` in from the caller.
    let use_memo = precomputed_tail.is_none();
    let pq_key = tail_pq_cache_key(input, obj_row);
    if use_memo {
        if let Some(pq) = TAIL_PQ_MEMO.with(|m| {
            m.borrow()
                .as_ref()
                .filter(|(k, _)| *k == pq_key)
                .map(|(_, pq)| pq.clone())
        }) {
            eprintln!("[imb] tail-opt: CACHE HIT (q={:+.6})", pq.1);
            return Some((pq.0, pq.1, Vec::new()));
        }
    }

    let acfg = AlphaCrownConfig {
        iterations: env_usize("NY_IMB_TAIL_OPT_INNER", 20),
        deadline: Some(deadline),
        ..Default::default()
    };
    let (mut tail_ibp, init_alpha) = match precomputed_tail {
        Some((base, init)) => (base.clone(), init.clone()),
        None => tail
            .collect_alpha_crown_bounds_dag_with_engine(seam_box, &acfg, engine)
            .ok()?,
    };
    // Override the tail ReLU-source anchors (Conv_19/Conv_22) with CROWN-tight,
    // input-correlated bounds — the seam-box `tail_ibp` is loose (independent-coord).
    // Used by BOTH the bespoke optimizer AND the final sound (p,q) below (all read
    // `tail_ibp`), so the whole tail functional is tight. The per-region path passes
    // `region_tail_anchors` PRE-CONCRETIZED from the shared input-linear maps (FIX 2,
    // no per-region crown-to-node); otherwise they are built here via crown-to-node.
    match region_tail_anchors {
        Some(anchors) => {
            for (k, v) in anchors.iter() {
                tail_ibp.insert(k.clone(), v.clone());
            }
        }
        None => {
            let tail_srcs = tail_relu_source_names(tail);
            let tail_tight = build_tight_tail_anchors(
                graph,
                input,
                node_bounds,
                prefix,
                &tail_srcs,
                prefix_anchor,
                engine,
                deadline,
            );
            for (k, v) in tail_tight {
                tail_ibp.insert(k, v);
            }
        }
    }
    let spec = Array2::from_shape_vec((1, obj_row.len()), obj_row.to_vec()).ok()?;

    // Seam manifold samples: the per-region path passes the ROOT-pool filtered to the
    // region (FIX 3, no per-region forward); otherwise sample over `input` here.
    let t_sample = Instant::now();
    let yseam = match precomputed_yseam {
        Some(ys) => ys,
        None => sample_seam_points(prefix, input, free_dims, engine, deadline)?,
    };
    if yseam.is_empty() {
        eprintln!("[imb] tail-opt: no seam samples; skip");
        return None;
    }
    eprintln!(
        "[imb] tail-opt: {} seam samples (dim {}) — sampled in {:.1}s",
        yseam.len(),
        yseam.first().map(Vec::len).unwrap_or(0),
        t_sample.elapsed().as_secs_f64(),
    );
    let t_opt = Instant::now();

    // Bespoke EXACT reverse-mode gradient ascent over the tail-ReLU lower slopes
    // (numpy disc_affine_alpha + grad_alpha) — ny's own alpha optimizers stalled on
    // the cGAN tail, so this self-contained gradient selects the alpha. It only
    // chooses `best` alpha; the certified functional is built below via the proven
    // sound path.
    //
    // RESERVE the last `NY_IMB_LEAF_RESERVE_S` (default 90 s) of the IMB budget for
    // the prefix leaf-BaB that FOLLOWS: the gradient loop otherwise runs to the full
    // deadline and starves the leaf-BaB (which is what decides the floor), timing the
    // whole certificate out. Stopping the (heuristic) alpha search early just uses the
    // best-so-far alpha — always sound. If the budget is already tighter than the
    // reserve, keep the original deadline (don't skip the search entirely).
    let leaf_reserve_s = env_f64("NY_IMB_LEAF_RESERVE_S", 90.0).max(0.0);
    let leaf_reserve = checked_duration_from_secs(leaf_reserve_s)?;
    let opt_deadline = deadline.checked_sub(leaf_reserve).unwrap_or(deadline);
    let (best, best_score) = super::tail_grad::optimize_tail_alpha_bespoke(
        tail,
        seam_box,
        &tail_ibp,
        &yseam,
        obj_row,
        &init_alpha,
        engine,
        opt_deadline,
    )?;
    eprintln!(
        "[imb] tail-opt(bespoke) ran in {:.1}s",
        t_opt.elapsed().as_secs_f64()
    );

    // Consistency: ny's tail CROWN (p,q) for the SAME init alpha — compare to the
    // bespoke(init) line to confirm the optimizer scores the right functional.
    if let Some((p, q)) = build_pq_raw(
        tail,
        seam_box,
        &tail_ibp,
        &init_alpha,
        &spec,
        engine,
        deadline,
    ) {
        let l1: f64 = p.iter().map(|c| c.abs() as f64).sum();
        eprintln!("[imb] tail-grad CONSISTENCY ny(init): q={q:+.6} |p|1={l1:.4}");
    }

    // Final SOUND (p,q) with the best alpha, coeff error folded outward.
    let (_b, lin) = SpecCrownRequest::new(tail, seam_box, &spec, engine)
        .node_bounds(&tail_ibp)
        .alpha_state_opt(Some(&best))
        .deadline_opt(Some(deadline))
        .run_with_linear()
        .ok()?;
    let mut lin = lin?;
    lin.fold_coeff_err_over_box_eager(seam_box);
    let p: Vec<f32> = lin.lower_a().row(0).to_vec();
    let q = *lin.lower_b().iter().next()?;
    if !q.is_finite() || p.iter().any(|c| !c.is_finite()) {
        return None;
    }
    let p_l1: f64 = p.iter().map(|c| c.abs() as f64).sum();
    eprintln!(
        "[imb] tail-opt(bespoke): best_sample_min={best_score:.6} q={q:+.6} |p|1={p_l1:.4} \
         (numpy targets: q~0.6551 |p|1~41.87 sample_min~0.6397)"
    );
    if use_memo {
        TAIL_PQ_MEMO.with(|m| *m.borrow_mut() = Some((pq_key, (p.clone(), q))));
    }
    Some((p, q, yseam))
}

/// Load the numpy-validated tail functional `(p, q)` for the END-TO-END prefix
/// validation (`NY_IMB_LOAD_PQ=1`):
/// - `p`: `expected_dim` little-endian f32 from `NY_IMB_P_PATH`
///   (default `/tmp/imb_numpy_p.f32`, exactly `4*expected_dim` bytes).
/// - `q`: an f64 parsed from `NY_IMB_Q_PATH` (default `/tmp/imb_numpy_q.txt`),
///   cast to f32 to match the f32 prefix pipeline.
///
/// Returns `None` (with a diagnostic) on any read/shape/parse failure.
fn load_numpy_pq(expected_dim: usize) -> Option<(Vec<f32>, f32)> {
    let p_path =
        std::env::var("NY_IMB_P_PATH").unwrap_or_else(|_| "/tmp/imb_numpy_p.f32".to_string());
    let q_path =
        std::env::var("NY_IMB_Q_PATH").unwrap_or_else(|_| "/tmp/imb_numpy_q.txt".to_string());
    let bytes = match std::fs::read(&p_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("[imb] LOAD_PQ: cannot read p from {p_path}: {e}");
            return None;
        }
    };
    if bytes.len() != expected_dim * 4 {
        eprintln!(
            "[imb] LOAD_PQ: {p_path} has {} bytes, expected {} (dim {expected_dim})",
            bytes.len(),
            expected_dim * 4
        );
        return None;
    }
    let p: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    let q_str = match std::fs::read_to_string(&q_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[imb] LOAD_PQ: cannot read q from {q_path}: {e}");
            return None;
        }
    };
    let q = match q_str.trim().parse::<f64>() {
        Ok(v) => v as f32,
        Err(e) => {
            eprintln!("[imb] LOAD_PQ: cannot parse q from {q_path} ({q_str:?}): {e}");
            return None;
        }
    };
    if !q.is_finite() || p.iter().any(|c| !c.is_finite()) {
        eprintln!("[imb] LOAD_PQ: loaded (p,q) has non-finite entries");
        return None;
    }
    Some((p, q))
}

/// Concretized tail-output lower bound over the seam box under a given alpha
/// state (single spec row). Used to log the alpha gain (init vs opt).
fn tail_crown_out_lower(
    tail: &GraphNetwork,
    seam_box: &BoundedTensor,
    spec: &Array2<f32>,
    engine: Option<&dyn GemmEngine>,
    tail_ibp: &HashMap<String, BoundedTensor>,
    alpha: &GraphAlphaState,
    deadline: Instant,
) -> Option<f32> {
    let out = SpecCrownRequest::new(tail, seam_box, spec, engine)
        .node_bounds(tail_ibp)
        .alpha_state_opt(Some(alpha))
        .deadline_opt(Some(deadline))
        .run()
        .ok()?;
    out.flatten().lower().iter().next().copied()
}

/// (max_width, mean_width, max_abs) of a bounded box — seam-box quality logging.
fn box_width_stats(b: &BoundedTensor) -> (f32, f32, f32) {
    let flat = b.flatten();
    let lo = flat.lower();
    let hi = flat.upper();
    let n = lo.len().max(1);
    let mut max_w = 0.0f32;
    let mut sum_w = 0.0f64;
    let mut max_abs = 0.0f32;
    for (l, u) in lo.iter().zip(hi.iter()) {
        let w = u - l;
        if w > max_w {
            max_w = w;
        }
        sum_w += w as f64;
        let a = l.abs().max(u.abs());
        if a > max_abs {
            max_abs = a;
        }
    }
    (max_w, (sum_w / n as f64) as f32, max_abs)
}

/// Tail ReLU-source (pre-activation) node names — the tail ReLUs' input nodes
/// (Conv_19, Conv_22 for cGAN), whose anchors drive the tail relaxation.
fn tail_relu_source_names(tail: &GraphNetwork) -> Vec<String> {
    let mut v: Vec<String> = Vec::new();
    if let Ok(exec) = tail.exec_order() {
        for name in exec {
            if let Some(node) = tail.nodes.get(name) {
                if matches!(node.layer, Layer::ReLU(_)) {
                    if let Some(src) = node.inputs.first() {
                        if src.as_str() != NETWORK_INPUT && !v.iter().any(|s| s == src) {
                            v.push(src.clone());
                        }
                    }
                }
            }
        }
    }
    v
}

/// CROWN-TIGHT anchors for the TAIL ReLU sources (Conv_19, Conv_22), bounded from
/// the ORIGINAL input over the FULL graph (capturing the manifold correlation the
/// seam-box `tail_ibp` loses). `tail_ibp` treats the 2048 seam coords as
/// independent → ~62 unstable / huge width; this input-correlated pass matches
/// numpy `build_crown_tight` (~12/0.026 for Conv_19, ~6/0.042 for Conv_22).
///
/// Reuses the tight prefix anchor (cached) as the intermediate relaxation anchors,
/// then sequentially `propagate_crown_to_node` each tail source, IBP-capped. SOUND:
/// a tighter valid enclosure of the pre-activation over the input box.
fn build_tight_tail_anchors(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    node_bounds: &HashMap<String, BoundedTensor>,
    prefix: &GraphNetwork,
    tail_srcs: &[String],
    precomputed_prefix_anchor: Option<&HashMap<String, BoundedTensor>>,
    engine: Option<&dyn GemmEngine>,
    deadline: Instant,
) -> HashMap<String, BoundedTensor> {
    let mut out: HashMap<String, BoundedTensor> = HashMap::new();
    // Tight prefix anchors seed the intermediate relaxation anchors so the full-graph
    // crown-to-node for the tail sources (over `input`, which may be a REGION box) is
    // tight through the generator. In the per-region path the SHARED ROOT prefix
    // anchor is passed in (region⊆root ⇒ sound, and NOT recomputed per region — the
    // 160s trap); otherwise the cached root anchor over `input` is used.
    let mut tightened = node_bounds.clone();
    match precomputed_prefix_anchor {
        Some(pa) => {
            for (k, v) in pa.iter() {
                tightened.insert(k.clone(), v.clone());
            }
        }
        None => {
            let Some(prefix_anchor) =
                build_tight_prefix_anchor_cached(prefix, input, engine, deadline, true)
            else {
                return out;
            };
            for (k, v) in prefix_anchor.iter() {
                tightened.insert(k.clone(), v.clone());
            }
        }
    }
    let Ok(exec) = graph.exec_order() else {
        return out;
    };
    // Same Lever-A parallel-chunk backward as the prefix anchor (scoped to this build).
    let _chunk_par = super::AnchorChunkParallelGuard::enable();
    // Tail sources in topological order (Conv_19 before Conv_22).
    let exec_owned: Vec<String> = exec.to_vec();
    for name in &exec_owned {
        if !tail_srcs.iter().any(|s| s == name) {
            continue;
        }
        let name_dim = node_bounds
            .get(name)
            .map(|b| b.flatten().len())
            .unwrap_or(0);
        let crown = match graph.propagate_crown_to_node(
            input,
            name,
            &tightened,
            node_bounds,
            engine,
            Some(deadline),
            imb_anchor_chunk_override(name_dim),
            None,
        ) {
            Ok(b) => b,
            Err(e) => {
                eprintln!(
                    "[imb] tight-tail-anchor {name}: crown-to-node failed ({e}); keep tail_ibp"
                );
                continue;
            }
        };
        let capped = match node_bounds.get(name) {
            Some(ib) if ib.shape() == crown.shape() => crown
                .intersection_per_element(ib)
                .map(|(t, _)| t)
                .unwrap_or(crown),
            _ => crown,
        };
        let (dim, unst, mw) = node_stats(&capped);
        eprintln!("[imb] tight-tail-anchor {name}: dim={dim} unstable={unst} max_w={mw:.4}");
        tightened.insert(name.clone(), capped.clone());
        out.insert(name.clone(), capped);
    }
    out
}

/// A tail ReLU-source node's INPUT-linear map + its ROOT-concretized enclosure — the
/// shared coefficients FIX 2 extracts ONCE so each region is a trivial concretize.
struct TailAnchorCoeff {
    /// CROWN linear bounds of the node w.r.t. the network input, valid over the ROOT
    /// box: `lin.lower_a·x + lin.lower_b ≤ node(x) ≤ lin.upper_a·x + lin.upper_b`
    /// (fixed ReLU relaxations anchored on the ROOT prefix/earlier-tail anchors, so
    /// the same map is valid over every region ⊆ root). Carries the certified
    /// per-coefficient error (`lower_a_err`/`upper_a_err`), discharged OUTWARD when a
    /// region concretizes it.
    lin: LinearBounds,
    /// The ROOT-box enclosure (CROWN concretized over the full input box, IBP-capped)
    /// — the `∩` cap that keeps a region anchor no looser than the root anchor.
    root_box: BoundedTensor,
    /// The node's tensor shape (to rebuild a `BoundedTensor` from a flat region box).
    shape: Vec<usize>,
}

/// FIX 2 (Option B) — extract each tail ReLU-source's INPUT-linear map ONCE over the
/// ROOT box, so per-region anchors are a trivial `concretize_box` matmul instead of a
/// fresh ~6-min `propagate_crown_to_node` through the whole ConvTranspose generator
/// (Option A was ~12 min/region → hours over R regions).
///
/// For each source (Conv_19, Conv_22) in topological order, build the sub-graph
/// `input→…→src` (output = src), seed an IDENTITY spec, and `run_with_linear` with
/// the SHARED ROOT prefix anchors (+ already-extracted earlier-source root boxes) as
/// the fixed ReLU-relaxation `node_bounds`. That single backward returns BOTH the
/// root enclosure (`result.bounds`) AND the input-linear map (`lower_a [dim×in]`,
/// `lower_b`, `upper_a`, `upper_b` + coeff error).
///
/// SOUND: the linear map holds over the whole root box (fixed relaxations anchored on
/// root-valid boxes), so for any region ⊆ root the same map bounds the node — the
/// per-region `concretize_box` (with the coeff error folded outward and directed
/// rounding) is a valid enclosure. Matches numpy's shared-coefficient trick.
fn build_tail_anchor_coeffs(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    node_bounds: &HashMap<String, BoundedTensor>,
    shared_prefix_anchor: &HashMap<String, BoundedTensor>,
    tail_srcs: &[String],
    engine: Option<&dyn GemmEngine>,
    deadline: Instant,
) -> Option<HashMap<String, TailAnchorCoeff>> {
    let mut out: HashMap<String, TailAnchorCoeff> = HashMap::new();
    // Fixed ReLU-relaxation anchors: full-graph IBP overlaid with the SHARED ROOT
    // prefix anchors; each source's root box is added as it is computed (so Conv_22's
    // backward relaxes Relu_20 on Conv_19's tight root box).
    let mut tightened = node_bounds.clone();
    for (k, v) in shared_prefix_anchor.iter() {
        tightened.insert(k.clone(), v.clone());
    }
    let exec = graph.exec_order().ok()?;
    let exec_owned: Vec<String> = exec.to_vec();
    for name in &exec_owned {
        if !tail_srcs.iter().any(|s| s == name) {
            continue;
        }
        let t0 = Instant::now();
        let sub = match build_subgraph_to_node(graph, name) {
            Some(s) => s,
            None => {
                eprintln!("[imb] coeff {name}: sub-graph build failed; skip region-coeff path");
                return None;
            }
        };
        let dim = node_bounds.get(name).map(|b| b.flatten().len())?;
        let identity: Array2<f32> = Array2::eye(dim);
        let (root_bounds, lin_opt) = match SpecCrownRequest::new(&sub, input, &identity, engine)
            .node_bounds(&tightened)
            .deadline_opt(Some(deadline))
            .run_with_linear()
        {
            Ok(v) => v,
            Err(e) => {
                eprintln!(
                    "[imb] coeff {name}: run_with_linear failed ({e}); skip region-coeff path"
                );
                return None;
            }
        };
        let lin = lin_opt?;
        // ∩-cap the root enclosure with the full-graph IBP (numpy parity; keeps the
        // region ∩ cap tight).
        let root_box = match node_bounds.get(name) {
            Some(ib) if ib.shape() == root_bounds.shape() => root_bounds
                .intersection_per_element(ib)
                .map(|(t, _)| t)
                .unwrap_or(root_bounds),
            _ => root_bounds,
        };
        let (d, unst, mw) = node_stats(&root_box);
        eprintln!(
            "[imb] coeff {name}: dim={d} unstable={unst} root_max_w={mw:.4} in_cols={} ({:.1}s)",
            lin.num_inputs(),
            t0.elapsed().as_secs_f64(),
        );
        let shape = root_box.lower().shape().to_vec();
        // Feed this source's tight ROOT box forward as the relaxation anchor for later
        // sources (Conv_19 → Relu_20 → Conv_22).
        tightened.insert(name.clone(), root_box.clone());
        out.insert(
            name.clone(),
            TailAnchorCoeff {
                lin,
                root_box,
                shape,
            },
        );
    }
    if out.is_empty() {
        eprintln!("[imb] coeff extraction produced no tail-source maps; skip region-coeff path");
        return None;
    }
    Some(out)
}

/// Per-region tail anchors by concretizing each pre-extracted input-linear map over
/// the REGION box (numpy `concretize_box`) — the 0-fresh-crown-to-node-per-region
/// step. For each source: `l_r[i] = Σ_j (pos(la)·rl + neg(la)·ru) + lb[i]` minus the
/// outward coeff-error penalty; `u_r[i]` analogously with `ua/ub`; then `∩` the root
/// box. Directed-rounded OUTWARD (accumulate in f64, `next_down`/`next_up` the f32).
///
/// SOUND: for a region ⊆ root, `l_r ≤ min_{x∈region} node(x)` and
/// `u_r ≥ max_{x∈region} node(x)` (interval concretization of a root-valid linear
/// map + outward error fold), and the root `∩` only tightens with a root-valid box.
fn concretize_region_anchors(
    coeffs: &HashMap<String, TailAnchorCoeff>,
    region_box: &BoundedTensor,
) -> Option<HashMap<String, BoundedTensor>> {
    let rf = region_box.flatten();
    let (rl, ru) = (rf.lower().as_slice()?, rf.upper().as_slice()?);
    let n_in = rl.len();
    // Per-column worst-case magnitude for the coeff-error penalty.
    let mag: Vec<f64> = (0..n_in)
        .map(|j| (rl[j] as f64).abs().max((ru[j] as f64).abs()))
        .collect();
    let mut out: HashMap<String, BoundedTensor> = HashMap::new();
    for (name, c) in coeffs.iter() {
        let la = c.lin.lower_a();
        let ua = c.lin.upper_a();
        let lb = c.lin.lower_b();
        let ub = c.lin.upper_b();
        if la.ncols() != n_in || ua.ncols() != n_in {
            eprintln!(
                "[imb] concretize {name}: input-dim mismatch (map cols {} vs region {n_in}); skip region-coeff path",
                la.ncols()
            );
            return None;
        }
        let dim = la.nrows();
        let le = c.lin.lower_a_err();
        let ue = c.lin.upper_a_err();
        let rbf = c.root_box.flatten();
        let (rblo, rbhi) = (rbf.lower().as_slice()?, rbf.upper().as_slice()?);
        let mut lo_v = vec![0.0f32; dim];
        let mut hi_v = vec![0.0f32; dim];
        for i in 0..dim {
            let mut l = lb[i] as f64;
            let mut u = ub[i] as f64;
            for j in 0..n_in {
                let laij = la[[i, j]] as f64;
                let uaij = ua[[i, j]] as f64;
                l += if laij >= 0.0 {
                    laij * rl[j] as f64
                } else {
                    laij * ru[j] as f64
                };
                u += if uaij >= 0.0 {
                    uaij * ru[j] as f64
                } else {
                    uaij * rl[j] as f64
                };
            }
            // Discharge the certified coeff error OUTWARD over the region box.
            if let Some(e) = le {
                let mut p = 0.0f64;
                for j in 0..n_in {
                    p += e[[i, j]] as f64 * mag[j];
                }
                if p.is_finite() {
                    l -= p;
                } else {
                    l = f64::NEG_INFINITY;
                }
            }
            if let Some(e) = ue {
                let mut p = 0.0f64;
                for j in 0..n_in {
                    p += e[[i, j]] as f64 * mag[j];
                }
                if p.is_finite() {
                    u += p;
                } else {
                    u = f64::INFINITY;
                }
            }
            // Outward-round f32, then ∩ the root box (both root-valid enclosures).
            let mut lf = next_down_f32(l as f32);
            let mut uf = next_up_f32(u as f32);
            lf = lf.max(rblo[i]);
            uf = uf.min(rbhi[i]);
            if lf > uf {
                // f32 rounding crossed the (thin) interval — clamp to the root box
                // (sound: root box encloses the region).
                lf = rblo[i];
                uf = rbhi[i];
            }
            lo_v[i] = lf;
            hi_v[i] = uf;
        }
        let la_arr = ArrayD::from_shape_vec(IxDyn(&c.shape), lo_v).ok()?;
        let ua_arr = ArrayD::from_shape_vec(IxDyn(&c.shape), hi_v).ok()?;
        out.insert(name.clone(), BoundedTensor::new(la_arr, ua_arr).ok()?);
    }
    Some(out)
}

/// Log-only self-check: sample seam-box points, forward the tail through the REAL
/// (ONNX-faithful) network, and report the worst `Y_o(y) − (p·y+q)` (must be ≥ 0
/// for a sound lower functional). Returns the worst observed slack (NaN if the
/// tail forward is unavailable).
const TAIL_PQ_SELF_CHECK_MAX_SAMPLES: usize = 4_096;

fn bounded_tail_pq_samples(requested: usize) -> usize {
    requested.min(TAIL_PQ_SELF_CHECK_MAX_SAMPLES)
}

fn tail_pq_self_check(
    tail: &GraphNetwork,
    seam_box: &BoundedTensor,
    p: &[f32],
    q: f32,
    obj_row: &[f32],
    engine: Option<&dyn GemmEngine>,
    deadline: Instant,
) -> f32 {
    if Instant::now() >= deadline {
        return f32::NAN;
    }
    let flat = seam_box.flatten();
    let (Some(lo), Some(hi)) = (flat.lower().as_slice(), flat.upper().as_slice()) else {
        return f32::NAN;
    };
    let n = lo.len();
    if p.len() != n {
        return f32::NAN;
    }
    let shape = seam_box.lower().shape().to_vec();
    // This is proposal-only telemetry, so an environment override must not
    // create an unbounded pre-authority workload.
    let samples = bounded_tail_pq_samples(env_usize("NY_IMB_PQ_SAMPLES", 256));
    let mut rng: u64 = 0x2545_f491_4f6c_dd1d;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut worst = f32::INFINITY;
    for s in 0..samples {
        if Instant::now() >= deadline {
            return f32::NAN;
        }
        let mut y = Vec::with_capacity(n);
        for k in 0..n {
            let v = if s % 2 == 0 {
                if next() < 0.5 {
                    lo[k]
                } else {
                    hi[k]
                }
            } else {
                lo[k] + (hi[k] - lo[k]) * (next() as f32)
            };
            y.push(v);
        }
        // p·y + q
        let mut pq = q as f64;
        for k in 0..n {
            pq += p[k] as f64 * y[k] as f64;
        }
        let Ok(arr) = ArrayD::from_shape_vec(IxDyn(&shape), y) else {
            continue;
        };
        let Ok(pt) = BoundedTensor::new(arr.clone(), arr) else {
            continue;
        };
        let Ok(out) = tail.propagate_concrete_point(&pt, engine, Some(deadline)) else {
            continue;
        };
        if Instant::now() >= deadline {
            return f32::NAN;
        }
        let of = out.flatten();
        let ov = of.lower();
        let mut yo = 0.0f64;
        for (j, &c) in obj_row.iter().enumerate() {
            yo += c as f64 * ov.get(j).copied().unwrap_or(0.0) as f64;
        }
        let slack = (yo - pq) as f32;
        if slack < worst {
            worst = slack;
        }
    }
    worst
}

// ===========================================================================
// Prefix: certified BaB floor of min_x[p·h(x)]
// ===========================================================================

thread_local! {
    /// Proposal-only one-slot memo of the last root tight-anchor.
    ///
    /// The historical u64 key does not bind graph weights/operators, input shape,
    /// or L2 metadata and may collide. It is therefore suitable only for paths
    /// whose final authority independently replays the original full network.
    /// Exact AY never consults this memo; [`ExactPrefixSession`] supplies its
    /// collision-free run-local reuse.
    static ROOT_ANCHOR_MEMO: std::cell::RefCell<Option<(u64, Arc<HashMap<String, BoundedTensor>>)>> =
        const { std::cell::RefCell::new(None) };
}

thread_local! {
    /// One-slot memo of the sample-guided tail `(p,q)` (keyed by input box bits +
    /// objective row). The disjunctive driver reaches the IMB injection twice, and
    /// the bespoke tail-opt is the most expensive step; caching keeps it single-run
    /// (BUG: double tail-opt). Sound: `(p,q)` is a deterministic function of the
    /// (graph, input box, objective), all in the key.
    static TAIL_PQ_MEMO: std::cell::RefCell<Option<(u64, (Vec<f32>, f32))>> =
        const { std::cell::RefCell::new(None) };
}

/// Cache key for the tail `(p,q)`: input box f32 bits + objective row bits.
fn tail_pq_cache_key(input: &BoundedTensor, obj_row: &[f32]) -> u64 {
    use std::hash::Hasher;
    let mut h = std::collections::hash_map::DefaultHasher::new();
    let f = input.flatten();
    if let (Some(lo), Some(hi)) = (f.lower().as_slice(), f.upper().as_slice()) {
        for v in lo {
            h.write_u32(v.to_bits());
        }
        for v in hi {
            h.write_u32(v.to_bits());
        }
    }
    for c in obj_row {
        h.write_u32(c.to_bits());
    }
    h.finish()
}

thread_local! {
    /// One-slot memo of the completed region-loop candidate (keyed by input box bits
    /// + objective row + region_k). The disjunctive driver reaches the IMB injection
    /// TWICE per instance; the whole region loop (shared anchor + per-region tail-opt
    /// + per-region BaB over R = k^free regions) is the single most expensive IMB
    /// step, so serving the second same-key injection from cache reclaims all of it
    /// (BUG: double region-loop). This is a HIGHER-LEVEL cache than ROOT_ANCHOR_MEMO
    /// / TAIL_PQ_MEMO (which cache sub-steps) and composes with them: a cache MISS
    /// here still hits those inner memos as before. Sound: `global_floor` (and the
    /// whole `ImbCandidate`) is a deterministic function of (graph, bit-exact input
    /// box, objective, region_k), every one of which is in the key — a hit returns
    /// the identical candidate that recomputing would.
    static REGION_FLOOR_MEMO: std::cell::RefCell<Option<(u64, ImbCandidate)>> =
        const { std::cell::RefCell::new(None) };
}

/// Cache key for the region-loop candidate: input box f32 bits + objective row
/// bits + region_k. (Reuses [`tail_pq_cache_key`] for the input+objective portion,
/// then mixes in `region_k` so different grid resolutions never collide.)
fn region_floor_cache_key(input: &BoundedTensor, obj_row: &[f32], region_k: usize) -> u64 {
    use std::hash::Hasher;
    let mut h = std::collections::hash_map::DefaultHasher::new();
    h.write_u64(tail_pq_cache_key(input, obj_row));
    h.write_usize(region_k);
    h.finish()
}

/// Historical proposal-memo key: flattened input endpoint bits plus output name
/// and node count. This deliberately remains byte-compatible for non-exact
/// proposal reuse, but is not a graph identity and must never gate exact authority.
fn anchor_cache_key(prefix: &GraphNetwork, input: &BoundedTensor) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    let f = input.flatten();
    if let (Some(lo), Some(hi)) = (f.lower().as_slice(), f.upper().as_slice()) {
        for v in lo {
            h.write_u32(v.to_bits());
        }
        for v in hi {
            h.write_u32(v.to_bits());
        }
    }
    prefix.output_node.hash(&mut h);
    h.write_usize(prefix.num_nodes());
    h.finish()
}

fn prefix_anchor_memo_allowed(exact_ay_authority: bool) -> bool {
    !exact_ay_authority
}

/// Proposal-only cached [`build_tight_prefix_anchor`] over the root box.
///
/// Exact callers use the raw builder through [`ExactPrefixSession`]; `memo_allowed`
/// is retained for legacy fallback sites that cannot mint AY authority.
fn build_tight_prefix_anchor_cached(
    prefix: &GraphNetwork,
    input: &BoundedTensor,
    engine: Option<&dyn GemmEngine>,
    deadline: Instant,
    memo_allowed: bool,
) -> Option<Arc<HashMap<String, BoundedTensor>>> {
    let key = anchor_cache_key(prefix, input);
    if memo_allowed {
        if let Some(cached) = ROOT_ANCHOR_MEMO.with(|m| {
            m.borrow()
                .as_ref()
                .filter(|(k, _)| *k == key)
                .map(|(_, a)| a.clone())
        }) {
            eprintln!(
                "[imb] prefix tight anchor: CACHE HIT ({} nodes)",
                cached.len()
            );
            return Some(cached);
        }
    }
    let arc = Arc::new(build_tight_prefix_anchor(prefix, input, engine, deadline)?);
    if memo_allowed {
        ROOT_ANCHOR_MEMO.with(|m| *m.borrow_mut() = Some((key, arc.clone())));
    }
    Some(arc)
}

/// Per-node objective-row chunk size for the anchor backward (Lever A). Returns the
/// `chunk_override` passed to `propagate_crown_to_node`:
/// - `NY_IMB_ANCHOR_CHUNK` unset → AUTO: `ceil(dim / (cores·4))` (many small chunks
///   for work-stealing + bounded peak memory — each concurrent chunk holds only
///   `[chunk_rows × max_intermediate_dim]`, so `chunk_rows ≈ 500` on BN_11's 28800
///   keeps peak modest even at full core occupancy).
/// - `NY_IMB_ANCHOR_CHUNK=0` → `None` (the old single-pass, no chunking/parallelism).
/// - `NY_IMB_ANCHOR_CHUNK=N` → `Some(N)` (fixed rows/chunk, for tuning).
///
/// `propagate_crown_to_node` only chunks when `chunk_size < target_dim`, so small
/// nodes (`dim ≤ chunk_rows`) transparently stay single-pass.
fn imb_anchor_chunk_override(dim: usize) -> Option<usize> {
    match std::env::var("NY_IMB_ANCHOR_CHUNK")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
    {
        Some(0) => None,
        Some(n) => Some(n.max(1)),
        None => {
            let cores = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(8);
            Some(dim.div_ceil(cores * 4).max(1))
        }
    }
}

/// Build a TIGHT prefix intermediate-bound anchor by SEQUENTIAL CROWN-to-node
/// tightening — the port of the numpy reference `build_crown_tight`.
///
/// For each ReLU-source (pre-activation) node in topological order, run
/// backward-CROWN to that node ([`GraphNetwork::propagate_crown_to_node`] — the
/// SAME path that produced the 0.024-tight seam box), using the ALREADY-TIGHTENED
/// earlier-node bounds as the relaxation anchors, IBP-cap the result, and feed it
/// back for subsequent nodes. `collect_crown_ibp_bounds_dag` leaves the amplifying
/// ConvTranspose generator on IBP (~1000× wider → the −229 root floor); this pass
/// tightens every generator pre-activation to the numpy widths (0.07–0.33).
///
/// SOUND: each per-node result is `max(crown_lo, ibp_lo) .. min(crown_hi, ibp_hi)`
/// — an intersection of two valid enclosures, hence still an enclosure. Reusing it
/// as a fixed relaxation anchor over any leaf `L ⊆ root` stays sound (the true
/// pre-activation range over `L` ⊆ range over root ⊆ the stored box).
fn build_tight_prefix_anchor(
    prefix: &GraphNetwork,
    input: &BoundedTensor,
    engine: Option<&dyn GemmEngine>,
    deadline: Instant,
) -> Option<HashMap<String, BoundedTensor>> {
    // IBP map: base for every node + the per-node cap reference.
    let ibp = prefix
        .collect_node_bounds_with_engine_and_deadline(input, engine, Some(deadline))
        .ok()?;
    let mut tightened = ibp.clone();

    // ReLU-source (pre-activation) nodes, deduped, in topological order.
    let exec = prefix.exec_order().ok()?;
    let mut relu_sources: Vec<String> = Vec::new();
    for name in exec {
        let Some(node) = prefix.nodes.get(name) else {
            continue;
        };
        if matches!(node.layer, Layer::ReLU(_)) {
            if let Some(src) = node.inputs.first() {
                if src != NETWORK_INPUT && !relu_sources.iter().any(|s| s == src) {
                    relu_sources.push(src.clone());
                }
            }
        }
    }

    // Opt the per-node objective-row backward into PARALLEL chunks (Lever A): each
    // node's crown-to-node seeds a `[dim×dim]` identity and its rows are independent,
    // so chunking the rows and bounding the chunks across cores is bound-equivalent
    // (deterministic) and turns the ~2-core single pass (BN_11, dim=28800, dominates
    // the ~180s build) into an ~8-core one. Guard is scoped to THIS build only.
    let _chunk_par = super::AnchorChunkParallelGuard::enable();
    for node in &relu_sources {
        let node_dim = ibp.get(node).map(|b| b.flatten().len()).unwrap_or(0);
        // Backward-CROWN to this pre-activation node, anchoring intermediate ReLU
        // relaxations on the already-tightened earlier nodes.
        let crown_box = match prefix.propagate_crown_to_node(
            input,
            node,
            &tightened,
            &ibp,
            engine,
            Some(deadline),
            imb_anchor_chunk_override(node_dim),
            None,
        ) {
            Ok(b) => b,
            // Keep the IBP box for this node on any failure (sound, just looser).
            Err(_) => continue,
        };
        // IBP cap: max(crown_lo, ibp_lo) .. min(crown_hi, ibp_hi) (numpy parity).
        let capped = match ibp.get(node) {
            Some(ib) if ib.shape() == crown_box.shape() => crown_box
                .intersection_per_element(ib)
                .map(|(t, _)| t)
                .unwrap_or(crown_box),
            _ => crown_box,
        };
        let (dim, unstable, max_w) = node_stats(&capped);
        eprintln!("[imb] tight-anchor {node}: dim={dim} unstable={unstable} max_w={max_w:.4}");
        tightened.insert(node.clone(), capped);
    }
    Some(tightened)
}

/// (dim, unstable_count, max_width) — per-node anchor quality vs the numpy reference.
fn node_stats(bt: &BoundedTensor) -> (usize, usize, f32) {
    let flat = bt.flatten();
    let lo = flat.lower();
    let hi = flat.upper();
    let dim = lo.len();
    let mut unstable = 0usize;
    let mut max_w = 0.0f32;
    for (l, u) in lo.iter().zip(hi.iter()) {
        if *l < 0.0 && *u > 0.0 {
            unstable += 1;
        }
        let w = u - l;
        if w > max_w {
            max_w = w;
        }
    }
    (dim, unstable, max_w)
}

/// Per-leaf intermediate-bound policy for the prefix BaB (`NY_IMB_LEAF_MODE`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum LeafMode {
    /// `ibp` — per-leaf IBP intermediates (cheap; re-anchors relaxations per leaf
    /// but IBP-loose on the deep generator → floor explodes).
    Ibp,
    /// `crown` — per-leaf full CROWN-IBP collection (tight, but ~2 min/leaf on the
    /// ConvTranspose generator → unusable past a few leaves).
    Crown,
    /// `crown_root` — collect the tight prefix CROWN-IBP intermediate map ONCE over
    /// the ROOT input box and REUSE it as the fixed ReLU-relaxation anchor for every
    /// leaf. Cheap per leaf (one backward, no collection); per-leaf tightening comes
    /// from concretizing the backward functional over the smaller leaf box.
    CrownRoot,
}

fn leaf_mode() -> LeafMode {
    match std::env::var("NY_IMB_LEAF_MODE").ok().as_deref() {
        Some("crown") => LeafMode::Crown,
        Some("crown_root") => LeafMode::CrownRoot,
        _ => LeafMode::Ibp,
    }
}

/// Certified lower bound on `min_x[p·h(x)]` over the input box, by per-leaf
/// backward-CROWN input-split BaB over the free dims. Splits the current
/// min-lower leaf's widest free dim until `NY_IMB_LEAVES` leaves (or the budget /
/// an early-exit once every leaf clears `early_exit` = band_lo − q). Returns
/// `(floor, leaves_used)`; the floor = min over leaf lowers (a valid global LB).
///
/// # `crown_root` soundness (root-anchored CROWN-IBP reuse)
///
/// The prefix CROWN-IBP pre-activation enclosures `[l_i, u_i]` are collected over
/// the ROOT input box. For any leaf sub-box `L ⊆ root`, the TRUE pre-activation
/// range over `L` is a subset of the range over `root`, hence a subset of
/// `[l_i, u_i]` — so the ReLU relaxation anchored on `[l_i, u_i]` is a valid
/// over-approximation for EVERY leaf. Per-leaf tightening comes from concretizing
/// the backward linear functional (nonzero input-space slope) over the smaller
/// leaf INPUT box: `min_{x∈L} (A·x + b)` strictly tightens as `L` shrinks. This is
/// standard fixed-relaxation input-split BaB; the "no split gain" caveat applies
/// only to a fully-collapsed (`A = 0`) affine functional, which this is not.
/// A leaf's anchor is always collected over a box that CONTAINS the leaf (the root
/// box, or — under `NY_IMB_REANCHOR_EVERY>0` — an ancestor leaf's box), so reuse
/// stays sound after any re-anchor.
fn prefix_bab_floor(
    prefix: &GraphNetwork,
    input: &BoundedTensor,
    spec: &Array2<f32>,
    engine: Option<&dyn GemmEngine>,
    deadline: Instant,
    early_exit: f32,
    precomputed_root_anchor: Option<Arc<HashMap<String, BoundedTensor>>>,
    immutable_leaf_cap: Option<usize>,
    force_early_exit: bool,
) -> Option<PrefixBabResult> {
    use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

    let mode = leaf_mode();
    let requested_leaves = env_usize("NY_IMB_LEAVES", 64).max(1);
    let target_leaves = immutable_leaf_cap
        .map(|cap| requested_leaves.min(cap.max(1)))
        .unwrap_or(requested_leaves);
    let flat = input.flatten();
    let lo = flat.lower().as_slice()?.to_vec();
    let hi = flat.upper().as_slice()?.to_vec();
    let free_dims: Vec<usize> = (0..lo.len()).filter(|&k| lo[k] < hi[k]).collect();
    // `force_early_exit` (the per-region path) stops the BaB as soon as the min leaf
    // clears the target (region floor ≥ band_lo) regardless of the env flag — the
    // region only needs to VERIFY, not log the full floor (numpy `bab_region` caps
    // similarly). The R=1 measurement path passes `false` (env-gated, full floor).
    // A region that CANNOT clear runs to `NY_IMB_LEAVES` (the cap) and returns its
    // best floor (< band_lo → region, hence global min, doesn't clear → not wired).
    let early_exit_on = force_early_exit
        || matches!(
            std::env::var("NY_IMB_EARLY_EXIT").ok().as_deref(),
            Some("1")
        );

    // `crown_root`: the TIGHT prefix anchor. In the per-region path the SHARED ROOT
    // anchor is passed in (sound over the region ⊆ root; avoids the 160s per-region
    // recompute); otherwise it's built/cached over `input`.
    let root_anchor: Option<Arc<HashMap<String, BoundedTensor>>> =
        if precomputed_root_anchor.is_some() {
            precomputed_root_anchor
        } else if mode == LeafMode::CrownRoot {
            let t0 = Instant::now();
            let arc = build_tight_prefix_anchor_cached(prefix, input, engine, deadline, true)?;
            eprintln!(
                "[imb] prefix tight anchor ready: {} nodes ({:.1}s)",
                arc.len(),
                t0.elapsed().as_secs_f64()
            );
            Some(arc)
        } else {
            None
        };

    // RAF per-leaf anchors: default ON for crown_root (NY_IMB_RAF=0 to disable).
    let raf_on = mode == LeafMode::CrownRoot
        && !matches!(std::env::var("NY_IMB_RAF").ok().as_deref(), Some("0"));
    // Prefix ReLU-source (pre-activation) node names — the RAF anchor targets.
    let relu_sources: HashSet<String> = {
        let mut s = HashSet::new();
        if let Ok(exec) = prefix.exec_order() {
            for name in exec {
                if let Some(node) = prefix.nodes.get(name) {
                    if matches!(node.layer, Layer::ReLU(_)) {
                        if let Some(src) = node.inputs.first() {
                            if src != NETWORK_INPUT {
                                s.insert(src.clone());
                            }
                        }
                    }
                }
            }
        }
        s
    };
    let anchor_ref = root_anchor.as_deref();

    // Parallelism: bound `NY_IMB_LEAF_THREADS` leaves' backward passes at once (each
    // an INDEPENDENT `crown_lower` with its own per-leaf RAF anchor). A dedicated
    // rayon pool caps concurrency for RSS control (each backward is compute-heavy but
    // holds only small coeff rows + a transient anchor clone).
    //
    // DEFAULT = 1 (SEQUENTIAL, BIT-EXACT, machine-INDEPENDENT): the committed default
    // must not depend on the box's core count (K>1 explores a different leaf set and
    // its floor is K-specific). `NY_IMB_LEAF_THREADS=N` opts into ~2× parallelism
    // (a different-but-sound deterministic floor; ny's per-leaf backward already uses
    // ONNX threadpools + faer SIMD, so leaf threads oversubscribe past a point).
    // Region-parallel path: run this region's leaf-BaB SEQUENTIALLY (no leaf pool). The
    // region par_iter already put us on one of N concurrent region workers; a nested
    // leaf pool + its conv `par_chunks_mut` (now serial via `region_seq_inner`) would
    // only fan out and starve the N-way region parallelism. Non-region path: the usual
    // `NY_IMB_LEAF_THREADS` leaf pool.
    let region_seq = crate::imb::region_seq_inner();
    let n_threads = if region_seq {
        1
    } else {
        env_usize("NY_IMB_LEAF_THREADS", 1).max(1)
    };
    let pool: Option<rayon::ThreadPool> = if region_seq {
        None
    } else {
        Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(n_threads)
                .build()
                .ok()?,
        )
    };

    // Sanitize non-finite leaf lowers to NEG_INFINITY (a `Some(NaN)` leaf must NOT be
    // silently dropped by `f32::min` — it would raise the floor above the truth).
    let san = |v: Option<f32>| -> f32 {
        match v {
            Some(x) if x.is_finite() => x,
            _ => f32::NEG_INFINITY,
        }
    };

    // Bound a batch of leaf boxes — parallel on the leaf pool (non-region path), or
    // serially on THIS region worker (region path). Order-preserving; the shared
    // `&prefix` caches are concurrent-read safe (mirrors `shared_specs.rs` par_iter).
    let bound_batch = |boxes: &[BoundedTensor]| -> Vec<f32> {
        let one = |bx: &BoundedTensor| {
            san(bound_leaf(
                prefix,
                bx,
                spec,
                engine,
                deadline,
                mode,
                anchor_ref,
                raf_on,
                &free_dims,
                &relu_sources,
            ))
        };
        match &pool {
            Some(p) => p.install(|| boxes.par_iter().map(one).collect()),
            None => boxes.iter().map(one).collect(),
        }
    };

    let root = san(bound_leaf(
        prefix,
        input,
        spec,
        engine,
        deadline,
        mode,
        anchor_ref,
        raf_on,
        &free_dims,
        &relu_sources,
    ));
    eprintln!(
        "[imb] prefix root p·h lower = {root:+.6} (leaf_mode={mode:?} raf={} threads={n_threads})",
        if raf_on { "on" } else { "off" }
    );

    let mut leaves: Vec<(f32, BoundedTensor)> = vec![(root, input.clone())];

    // Leaf-BaB timing (measurement of the parallel-scaling wall): total + per-batch.
    let bab_t0 = Instant::now();
    let mut batch_no = 0usize;
    let mut leaves_bounded = 0usize;

    while leaves.len() < target_leaves {
        if Instant::now() >= deadline {
            break;
        }
        // DETERMINISTIC selection: order the frontier by (lower asc, index asc). The
        // index tiebreak makes `NY_IMB_LEAF_THREADS=1` reproduce the sequential
        // best-first (split-the-single-min) leaf set BIT-EXACTLY. For K>1 we split
        // the K lowest per round (deterministic; a different but sound leaf set).
        let mut order: Vec<usize> = (0..leaves.len()).collect();
        order.sort_by(|&a, &b| {
            leaves[a]
                .0
                .partial_cmp(&leaves[b].0)
                .unwrap_or(Ordering::Equal)
                .then(a.cmp(&b))
        });
        // Optional early exit: the lowest leaf already clears the target ⇒ decided.
        if early_exit_on && early_exit.is_finite() && leaves[order[0]].0 >= early_exit {
            break;
        }
        let budget = target_leaves.saturating_sub(leaves.len());
        let k = n_threads.min(budget).min(leaves.len()).max(1);

        // Remove the K lowest (descending index so earlier indices are not shifted).
        let mut remove_idx: Vec<usize> = order[0..k].to_vec();
        remove_idx.sort_unstable_by(|a, b| b.cmp(a));
        let mut to_split: Vec<(f32, BoundedTensor)> = Vec::with_capacity(k);
        for i in remove_idx {
            to_split.push(leaves.remove(i));
        }

        // Split each on its widest free dim; unsplittable leaves are kept as-is.
        let mut children: Vec<BoundedTensor> = Vec::with_capacity(2 * k);
        let mut kept: Vec<(f32, BoundedTensor)> = Vec::new();
        for (lwr, bx) in to_split {
            match widest_free_dim(&bx, &free_dims).and_then(|d| split_box(&bx, d)) {
                Some((left, right)) => {
                    children.push(left);
                    children.push(right);
                }
                None => kept.push((lwr, bx)),
            }
        }
        if children.is_empty() {
            leaves.extend(kept);
            break; // no splittable leaf left
        }
        // Bound the children's backward passes IN PARALLEL (order-preserving).
        let bt = Instant::now();
        let lowers = bound_batch(&children);
        let bwall = bt.elapsed().as_secs_f64();
        batch_no += 1;
        leaves_bounded += children.len();
        eprintln!(
            "[imb] leaf-batch #{batch_no}: {} leaves in {bwall:.2}s ({:.2}s/leaf, threads={n_threads})",
            children.len(),
            bwall / children.len().max(1) as f64,
        );
        for (child, l) in children.into_iter().zip(lowers) {
            leaves.push((l, child));
        }
        leaves.extend(kept);
    }

    eprintln!(
        "[imb] ===== leaf-BaB WALL {:.2}s | {leaves_bounded} leaves bounded in {batch_no} batches | threads={n_threads} leaves_total={} =====",
        bab_t0.elapsed().as_secs_f64(),
        leaves.len(),
    );

    // NaN-safe global min = the sound floor (order-independent → deterministic).
    let floor = leaves
        .iter()
        .map(|(l, _)| *l)
        .fold(f32::INFINITY, |acc, l| {
            if l.is_finite() {
                acc.min(l)
            } else {
                f32::NEG_INFINITY
            }
        });
    Some(PrefixBabResult {
        floor,
        terminal_boxes: leaves.into_iter().map(|(_, bx)| bx).collect(),
    })
}

/// Bound `spec·h` over `bx`, choosing the leaf anchor per mode. For crown_root with
/// RAF on, the anchor is `intersect(root_crown, raf_forward(bx))` (per-leaf tightening
/// from the correlation-preserving affine form); otherwise the mode's usual anchor.
#[allow(clippy::too_many_arguments)]
fn bound_leaf(
    prefix: &GraphNetwork,
    bx: &BoundedTensor,
    spec: &Array2<f32>,
    engine: Option<&dyn GemmEngine>,
    deadline: Instant,
    mode: LeafMode,
    root_crown: Option<&HashMap<String, BoundedTensor>>,
    raf_on: bool,
    free_dims: &[usize],
    relu_sources: &HashSet<String>,
) -> Option<f32> {
    if mode == LeafMode::CrownRoot && raf_on {
        if let (Some(rc), Some(raf)) = (
            root_crown,
            super::raf::raf_forward(prefix, bx, free_dims, relu_sources),
        ) {
            let leaf_anchor = intersect_anchor(rc, &raf);
            return crown_lower(prefix, bx, spec, engine, deadline, mode, Some(&leaf_anchor));
        }
        // RAF unavailable (non-affine op / non-chain prefix) — fall back to root-crown.
    }
    crown_lower(prefix, bx, spec, engine, deadline, mode, root_crown)
}

/// Per-node `intersect(root_crown, raf)` (max lower, min upper). Both operands are
/// sound enclosures, so the result is a sound (tighter) enclosure. Nodes present in
/// `raf` but absent/shape-mismatched in `root_crown` keep the root-crown box.
fn intersect_anchor(
    root_crown: &HashMap<String, BoundedTensor>,
    raf: &HashMap<String, BoundedTensor>,
) -> HashMap<String, BoundedTensor> {
    let mut m = root_crown.clone();
    for (node, raf_box) in raf {
        if let Some(cw) = m.get(node) {
            if cw.shape() == raf_box.shape() {
                if let Some((capped, _)) = cw.intersection_per_element(raf_box) {
                    m.insert(node.clone(), capped);
                }
            }
        }
    }
    m
}

/// Per-leaf backward-CROWN lower bound of `spec·h` over `bx` (spec is the single
/// row `p`), with the ReLU-relaxation anchor chosen by `mode`:
/// - [`LeafMode::Ibp`]: collect a cheap per-leaf IBP map and anchor on it.
/// - [`LeafMode::Crown`]: `node_bounds = None` → the CROWN pass recomputes full
///   per-leaf CROWN-IBP intermediates (tight, slow).
/// - [`LeafMode::CrownRoot`]: anchor on the fixed root (`anchor`) map — sound reuse
///   (see [`prefix_bab_floor`] docs); the leaf tightens via input-box concretization.
///
/// All three are sound (each anchor map is a valid enclosure over `bx`, and CROWN
/// is an over-approximation).
fn crown_lower(
    prefix: &GraphNetwork,
    bx: &BoundedTensor,
    spec: &Array2<f32>,
    engine: Option<&dyn GemmEngine>,
    deadline: Instant,
    mode: LeafMode,
    anchor: Option<&HashMap<String, BoundedTensor>>,
) -> Option<f32> {
    // For ibp mode, collect the per-leaf IBP map here so it outlives the borrow.
    let per_leaf_ibp = if mode == LeafMode::Ibp {
        Some(
            prefix
                .collect_node_bounds_with_engine_and_deadline(bx, engine, Some(deadline))
                .ok()?,
        )
    } else {
        None
    };
    let leaf_nb: Option<&HashMap<String, BoundedTensor>> = match mode {
        LeafMode::Crown => None,
        LeafMode::CrownRoot => anchor,
        LeafMode::Ibp => per_leaf_ibp.as_ref(),
    };
    let (bounds, _lin) = compute_crown_or_ibp_bounds_with_node_bounds(
        prefix,
        bx,
        spec,
        engine,
        leaf_nb,        // alpha_node_bounds — fixed anchor / per-leaf IBP / None(=CROWN-IBP)
        None,           // child_node_bounds
        None,           // alpha_state — prefix kept EXACT via adaptive-slope CROWN
        None,           // mul_binary_alphas
        Some(deadline), // deadline
        None,           // crown_backward_layers
        false,          // ibp_enhancement
    )
    .ok()?;
    let obj = extract_obj_bounds(&bounds, 1).ok()?;
    Some(obj[0].0)
}

/// Widest free dim of `bx` (largest `hi−lo` among `free_dims`).
fn widest_free_dim(bx: &BoundedTensor, free_dims: &[usize]) -> Option<usize> {
    let flat = bx.flatten();
    let lo = flat.lower();
    let hi = flat.upper();
    let lo = lo.as_slice()?;
    let hi = hi.as_slice()?;
    let mut best: Option<(usize, f32)> = None;
    for &d in free_dims {
        if d >= lo.len() {
            continue;
        }
        let w = hi[d] - lo[d];
        if best.map_or(true, |(_, bw)| w > bw) {
            best = Some((d, w));
        }
    }
    best.map(|(d, _)| d)
}

/// Midpoint-split `bx` along flat index `dim` into two covering children.
fn split_box(bx: &BoundedTensor, dim: usize) -> Option<(BoundedTensor, BoundedTensor)> {
    let shape = bx.lower().shape().to_vec();
    let flat = bx.flatten();
    let lo = flat.lower().as_slice()?.to_vec();
    let hi = flat.upper().as_slice()?.to_vec();
    if dim >= lo.len() {
        return None;
    }
    let mid = lo[dim] + (hi[dim] - lo[dim]) * 0.5;
    // A rounded midpoint equal to either endpoint is not a binary partition:
    // it would duplicate a parent/point child and could make coverage replay
    // ambiguous.  Treat one-ULP boxes as unsplittable.
    if !mid.is_finite() || mid <= lo[dim] || mid >= hi[dim] {
        return None;
    }
    let mut left_hi = hi.clone();
    left_hi[dim] = mid;
    let mut right_lo = lo.clone();
    right_lo[dim] = mid;
    let left = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&shape), lo).ok()?,
        ArrayD::from_shape_vec(IxDyn(&shape), left_hi).ok()?,
    )
    .ok()?;
    let right = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&shape), right_lo).ok()?,
        ArrayD::from_shape_vec(IxDyn(&shape), hi).ok()?,
    )
    .ok()?;
    Some((left, right))
}

// ===========================================================================
// Diagnostics
// ===========================================================================

/// Dump the exec-order graph (name / layer-type / inputs) for seam discovery
/// (`NY_IMB_DUMP=1`).
fn dump_nodes(graph: &GraphNetwork) {
    if let Ok(exec) = graph.exec_order() {
        eprintln!("[imb] graph has {} nodes (exec order):", exec.len());
        for (i, name) in exec.iter().enumerate() {
            if let Some(node) = graph.nodes.get(name) {
                eprintln!(
                    "[imb]   {i:3} {name} type={} inputs={:?}",
                    node.layer().layer_type(),
                    node.inputs()
                );
            }
        }
    }
}
