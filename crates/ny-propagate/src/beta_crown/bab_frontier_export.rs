// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #bab-frontier: process-global export of the BaB engine's SURVIVING
//! UNVERIFIED frontier at exhaustion (docs/BAB_FRONTIER_SEEDING_DESIGN.md).
//!
//! MOTIVATION (acasxu prop_2 basin-not-found class): when `verify_impl`
//! exhausts (timeout / domain limit / queue-mem cap) its `BinaryHeap<BabDomain>`
//! queue IS the set of subboxes where a counterexample must live if one exists
//! — every other region was verified safe. Today all three exhaust exits
//! silently DROP that frontier; the post-BaB attack then restarts from the box
//! center + uniform random points and never reaches the surviving basins.
//! Exporting the top-K most-violation-likely subboxes (midpoint centers +
//! bounds) gives the attack's restart schedule direct basin contact.
//!
//! SOUNDNESS: attack-only guidance, never a verdict carrier. The consumer
//! (ny-cli `try_postbab_falsify`) only uses the centers as SEARCH SEEDS; any
//! candidate still passes the UNCHANGED trusted-ORT + zero-tolerance
//! `property_violated_f64` acceptance gate. A wrong or stale seed can at worst
//! spend otherwise-dead leftover budget — it can never manufacture a false
//! `sat`.
//!
//! Mirrors the `best_margin_export` channel shape (process-global `Mutex` with
//! reset/record/take + poisoned-lock recovery), but lives in ny-propagate
//! because the writer is `verify_impl`. Gate: `NY_POSTBAB_BAB_SEEDS` (default
//! OFF), checked only in the [`record_bab_frontier_if_enabled`] wrapper so the
//! record/take/reset core stays env-free and unit-testable.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::Mutex;

use ndarray::Axis;
use ny_tensor::BoundedTensor;

use crate::batched_domain::DomainList;
use crate::beta_crown::domain::{BabDomain, GraphBabDomain};
use crate::beta_crown::engine::JointMarginCloser;
use crate::BetaCrownConfig;

/// One surviving-unverified BaB domain exported as an attack seed.
#[derive(Debug, Clone)]
pub struct BabFrontierSeed {
    /// Midpoint of the domain's input subbox, row-major ORIGINAL X-space
    /// (input splits act on the network inputs, so this is directly a valid
    /// attack seed).
    pub center: Vec<f32>,
    /// Subbox lower corner (same space/arity as `center`).
    pub box_lo: Vec<f32>,
    /// Subbox upper corner (same space/arity as `center`).
    pub box_hi: Vec<f32>,
    /// The domain's certified output lower bound. Seeds are stored in
    /// violation-priority order (most violation-likely first — for the
    /// lower-bound direction these are the most-negative-margin domains).
    pub margin: f32,
    /// Input-split depth of the domain (root = 0).
    pub depth: usize,
    /// #bab-frontier v2 (b), `NY_POSTBAB_BAB_SEEDS=2` only: per-row MINIMIZER
    /// corners of the JointMarginCloser's certified affine lower-bound rows
    /// over THIS subbox (`x_d = lo_d` if `a[j,d] > 0` else `hi_d`), computed
    /// for the top [`BAB_FRONTIER_CORNER_BOXES`] seeds when the closer is
    /// attached. Empty in v1 mode, past the top boxes, when no closer is
    /// attached (all graph-lane seeds), or when the corner pass fails — the
    /// consumer then falls back to the subbox's own extreme corners.
    /// Attack-side guidance only.
    pub corners: Vec<Vec<f32>>,
}

/// Upper clamp for the requested K (memory bound together with
/// [`BAB_FRONTIER_MAX_DIMS`]: 4096 × 3 vecs × 1024 dims × 4 B ≈ 48 MB absolute
/// worst case; the default K=256 is ~3 MB worst case and ~15 KB on acasxu).
pub const BAB_FRONTIER_MAX_K: usize = 4096;

/// Nets with more inputs than this export nothing (memory bound; the target
/// class — input-split BaB — is low-dimensional by construction).
pub const BAB_FRONTIER_MAX_DIMS: usize = 1024;

/// Default K when `NY_POSTBAB_BAB_SEEDS_K` is unset/unparsable.
pub const BAB_FRONTIER_DEFAULT_K: usize = 256;

/// #bab-frontier v2 (b): number of top (most violation-likely) subboxes that
/// get corner seeds — the exporter attaches JointMarginCloser per-row
/// minimizer corners to at most this many seeds, and the consumer applies the
/// extreme-corner fallback to the same prefix. Small by design: corners
/// multiply the restart list (k rows per box), and only the best boxes are
/// worth the extra legs.
pub const BAB_FRONTIER_CORNER_BOXES: usize = 16;

/// The exported frontier since the last reset (violation-priority order).
static FRONTIER: Mutex<Vec<BabFrontierSeed>> = Mutex::new(Vec::new());

/// Recover from a poisoned lock: the channel is guidance-only, so a panicked
/// thread must never take the verdict path down with it.
fn lock() -> std::sync::MutexGuard<'static, Vec<BabFrontierSeed>> {
    FRONTIER.lock().unwrap_or_else(|p| p.into_inner())
}

/// Clear the exported frontier (call before starting a fresh verification run).
pub fn reset_bab_frontier_export() {
    lock().clear();
}

/// Take (and clear) the frontier seeds recorded since the last reset, most
/// violation-likely first.
pub fn take_bab_frontier_seeds() -> Vec<BabFrontierSeed> {
    std::mem::take(&mut *lock())
}

/// Record the top-`k` most-violation-likely surviving domains from the BaB
/// queue as attack seeds, replacing any previously recorded frontier.
///
/// Domains with non-finite bounds/priority or without their own input subbox
/// (`input_bounds = None`, e.g. the root before any input split) are skipped.
/// Env-free by design (`k` is explicit); the env gate lives in
/// [`record_bab_frontier_if_enabled`].
pub fn record_bab_frontier_from_queue(
    queue: &BinaryHeap<BabDomain>,
    root_input: &BoundedTensor,
    k: usize,
) {
    *lock() = build_bab_frontier_from_queue(queue, root_input, k);
}

/// Pure builder behind [`record_bab_frontier_from_queue`]: returns the top-`k`
/// seed list instead of storing it, so the env-gated wrapper can attach v2
/// corner seeds before publishing.
fn build_bab_frontier_from_queue(
    queue: &BinaryHeap<BabDomain>,
    root_input: &BoundedTensor,
    k: usize,
) -> Vec<BabFrontierSeed> {
    let k = k.clamp(1, BAB_FRONTIER_MAX_K);
    let dims = root_input.len();
    if dims == 0 || dims > BAB_FRONTIER_MAX_DIMS {
        return Vec::new();
    }

    // (a) Collect refs to the eligible surviving domains.
    let mut candidates: Vec<&BabDomain> = queue
        .iter()
        .filter(|d| {
            d.input_bounds.is_some()
                && d.lower_bound.is_finite()
                && d.upper_bound.is_finite()
                && d.priority.is_finite()
        })
        .collect();

    // (b)+(c) Top-K by violation priority DESC (most violation-likely first).
    let by_priority_desc = |a: &&BabDomain, b: &&BabDomain| -> Ordering {
        b.priority
            .partial_cmp(&a.priority)
            .unwrap_or(Ordering::Equal)
    };
    if candidates.len() > k {
        candidates.select_nth_unstable_by(k - 1, by_priority_desc);
        candidates.truncate(k);
    }
    candidates.sort_by(by_priority_desc);

    // (d) Midpoint centers + subboxes.
    let mut out: Vec<BabFrontierSeed> = Vec::with_capacity(candidates.len());
    for domain in candidates {
        let Some(input_bounds) = domain.input_bounds.as_deref() else {
            continue; // filtered above; defensive
        };
        if input_bounds.len() != dims {
            continue; // arity mismatch vs the root box: never export
        }
        let (lower, upper) = input_bounds.lower_upper();
        let box_lo: Vec<f32> = lower.iter().copied().collect();
        let box_hi: Vec<f32> = upper.iter().copied().collect();
        if box_lo.iter().chain(box_hi.iter()).any(|v| !v.is_finite()) {
            continue;
        }
        // Midpoint, clamped per-dim so f32 rounding can never place the center
        // outside its own subbox.
        let center: Vec<f32> = box_lo
            .iter()
            .zip(box_hi.iter())
            .map(|(&l, &h)| (l + 0.5 * (h - l)).clamp(l, h))
            .collect();
        out.push(BabFrontierSeed {
            center,
            box_lo,
            box_hi,
            margin: domain.lower_bound,
            depth: domain.input_split_count,
            corners: Vec::new(),
        });
    }
    out
}

/// Thin env-gated wrapper for `verify_impl`'s three exhaust exits (timeout /
/// domain limit / queue-mem cap). Gate `NY_POSTBAB_BAB_SEEDS`: unset/anything
/// but `"1"`/`"2"` => fully off (no recording, zero cost — the queue is about
/// to be dropped anyway); `"1"` => v1: record top-K midpoint centers (K =
/// `NY_POSTBAB_BAB_SEEDS_K`, default [`BAB_FRONTIER_DEFAULT_K`], clamped
/// 1..=[`BAB_FRONTIER_MAX_K`]); `"2"` => v2: additionally attach
/// JointMarginCloser per-row minimizer corners to the top
/// [`BAB_FRONTIER_CORNER_BOXES`] seeds when a closer is attached (the
/// consumer also keys its subbox-projected restarts off the same value).
///
/// The corner pass is post-deadline work on the tiny truncated same-LHS net
/// (<= 16 CROWN passes), and like everything in this channel it is guidance
/// only: a wrong corner can never manufacture a `sat`.
pub(crate) fn record_bab_frontier_if_enabled(
    queue: &BinaryHeap<BabDomain>,
    root_input: &BoundedTensor,
    joint_margin_closer: Option<&JointMarginCloser>,
) {
    let Some((mode, k)) = frontier_gate_mode_and_k() else {
        return;
    };
    let mut out = build_bab_frontier_from_queue(queue, root_input, k);
    if mode >= 2 {
        if let Some(closer) = joint_margin_closer {
            for seed in out.iter_mut().take(BAB_FRONTIER_CORNER_BOXES) {
                seed.corners = subbox_tensor(&seed.box_lo, &seed.box_hi)
                    .and_then(|b| closer.per_row_minimizer_corners(&b))
                    .unwrap_or_default();
            }
        }
    }
    *lock() = out;
}

/// Rebuild a [`BoundedTensor`] from an exported subbox (both corners already
/// validated finite by the builder). `None` on the (defensive) invalid-bounds
/// case — the seed then simply carries no closer corners.
fn subbox_tensor(box_lo: &[f32], box_hi: &[f32]) -> Option<BoundedTensor> {
    BoundedTensor::new(
        ndarray::Array1::from(box_lo.to_vec()).into_dyn(),
        ndarray::Array1::from(box_hi.to_vec()).into_dyn(),
    )
    .ok()
}

/// Shared `NY_POSTBAB_BAB_SEEDS` gate for every recording wrapper (sequential
/// heap, graph heap, graph DomainList): `None` => fully off (unset/anything
/// but `"1"`/`"2"`); `Some((mode, k))` otherwise, with K =
/// `NY_POSTBAB_BAB_SEEDS_K` (default [`BAB_FRONTIER_DEFAULT_K`], clamped
/// 1..=[`BAB_FRONTIER_MAX_K`]).
fn frontier_gate_mode_and_k() -> Option<(u8, usize)> {
    let mode: u8 = match std::env::var("NY_POSTBAB_BAB_SEEDS").ok().as_deref() {
        Some("1") => 1,
        Some("2") => 2,
        // #bab-frontier v3: recording is identical to mode 2 (the corner pass
        // keys off `mode >= 2`); the extra behavior is CONSUMER-side only
        // (the attack's leftover random restarts are drawn INSIDE the exported
        // subboxes instead of the global box).
        Some("3") => 3,
        _ => return None,
    };
    let k = std::env::var("NY_POSTBAB_BAB_SEEDS_K")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(BAB_FRONTIER_DEFAULT_K)
        .clamp(1, BAB_FRONTIER_MAX_K);
    Some((mode, k))
}

// ---------------------------------------------------------------------------
// #bab-frontier graph lane: the GRAPH/GPU BaB engines (DomainList GPU BaB and
// the GraphBabDomain heap lanes) exhaust through `check_termination` without
// ever touching the sequential `BinaryHeap<BabDomain>` recorder above. Their
// domains carry their own box types, so each gets a mapper onto the SAME
// channel + env gate.
//
// Subbox rule: a graph domain always carries `input_bounds`, but only a box
// that DIFFERS from the root box is a genuine subbox (input-split children,
// clip-shrunk domains). Domains still covering the whole root box are skipped
// — their midpoint is the root center the post-BaB attack already tries — so
// pure ReLU-split frontiers export nothing, mirroring the sequential lane's
// `input_bounds = None` skip.
// ---------------------------------------------------------------------------

/// Build one exported seed from an owned subbox: any non-finite coordinate
/// voids the seed; the midpoint is clamped per-dim so f32 rounding can never
/// place the center outside its own subbox (same rules as the sequential
/// builder above).
fn seed_from_box(
    box_lo: Vec<f32>,
    box_hi: Vec<f32>,
    margin: f32,
    depth: usize,
) -> Option<BabFrontierSeed> {
    if box_lo.iter().chain(box_hi.iter()).any(|v| !v.is_finite()) {
        return None;
    }
    let center: Vec<f32> = box_lo
        .iter()
        .zip(box_hi.iter())
        .map(|(&l, &h)| (l + 0.5 * (h - l)).clamp(l, h))
        .collect();
    Some(BabFrontierSeed {
        center,
        box_lo,
        box_hi,
        margin,
        depth,
        corners: Vec::new(),
    })
}

/// Keep the top-`k` candidates by priority DESC (most violation-likely first)
/// and leave them sorted. Priorities are pre-filtered finite by the callers,
/// so `partial_cmp` falling back to `Equal` is unreachable in practice.
fn keep_top_k_by_priority_desc<T>(candidates: &mut Vec<T>, k: usize, priority: impl Fn(&T) -> f32) {
    let by_priority_desc = |a: &T, b: &T| -> Ordering {
        priority(b)
            .partial_cmp(&priority(a))
            .unwrap_or(Ordering::Equal)
    };
    if candidates.len() > k {
        candidates.select_nth_unstable_by(k - 1, by_priority_desc);
        candidates.truncate(k);
    }
    candidates.sort_by(by_priority_desc);
}

/// Does this domain's box differ from the root box in any coordinate?
/// (Equality means "no subbox of its own" — see the graph-lane header above.)
fn differs_from_root<'a>(
    lo: impl Iterator<Item = &'a f32>,
    hi: impl Iterator<Item = &'a f32>,
    root_lo: &[f32],
    root_hi: &[f32],
) -> bool {
    lo.zip(root_lo.iter()).any(|(a, b)| a != b) || hi.zip(root_hi.iter()).any(|(a, b)| a != b)
}

/// Record the top-`k` surviving [`GraphBabDomain`]s that own a genuine input
/// subbox, replacing any previously recorded frontier. Env-free by design
/// (`k` is explicit); the env gate lives in
/// [`record_graph_bab_frontier_if_enabled`]. Takes an iterator, not the heap,
/// so callers that already popped the current domain can chain it back in.
pub fn record_graph_bab_frontier_from_domains<'a, I>(
    domains: I,
    root_input: &BoundedTensor,
    k: usize,
) where
    I: IntoIterator<Item = &'a GraphBabDomain>,
{
    let k = k.clamp(1, BAB_FRONTIER_MAX_K);
    let dims = root_input.len();
    if dims == 0 || dims > BAB_FRONTIER_MAX_DIMS {
        return;
    }
    let (root_lo, root_hi) = root_input.lower_upper();
    let root_lo: Vec<f32> = root_lo.iter().copied().collect();
    let root_hi: Vec<f32> = root_hi.iter().copied().collect();

    let mut candidates: Vec<&GraphBabDomain> = domains
        .into_iter()
        .filter(|d| {
            d.lower_bound.is_finite()
                && d.upper_bound.is_finite()
                && d.priority.is_finite()
                && d.input_bounds.len() == dims
                && {
                    let (lo, hi) = d.input_bounds.lower_upper();
                    differs_from_root(lo.iter(), hi.iter(), &root_lo, &root_hi)
                }
        })
        .collect();
    keep_top_k_by_priority_desc(&mut candidates, k, |d| d.priority);

    let out: Vec<BabFrontierSeed> = candidates
        .into_iter()
        .filter_map(|d| {
            let (lo, hi) = d.input_bounds.lower_upper();
            seed_from_box(
                lo.iter().copied().collect(),
                hi.iter().copied().collect(),
                d.lower_bound,
                d.depth,
            )
        })
        .collect();
    *lock() = out;
}

/// Env-gated wrapper for the graph ReLU-split heap lanes' exhaust exits
/// (`check_termination` returning a terminal result). Same gate and channel
/// as the sequential lane; guidance only.
pub(crate) fn record_graph_bab_frontier_if_enabled<'a, I>(domains: I, root_input: &BoundedTensor)
where
    I: IntoIterator<Item = &'a GraphBabDomain>,
{
    let Some((_mode, k)) = frontier_gate_mode_and_k() else {
        return;
    };
    record_graph_bab_frontier_from_domains(domains, root_input, k);
}

/// Record the top-`k` surviving [`DomainList`] domains that own a genuine
/// input subbox (the GPU BaB lane, both input-split and ReLU-split modes),
/// replacing any previously recorded frontier. Priority is derived from the
/// stored per-domain bounds via the SAME contract the DomainList queue sorts
/// by (`BetaCrownConfig::domain_priority_for_mode`), so the exported order
/// matches the frontier ordering the engine itself was exploring. Env-free by
/// design; the env gate lives in [`record_domain_list_frontier_if_enabled`].
pub fn record_domain_list_frontier(
    domain_list: &DomainList,
    root_input: &BoundedTensor,
    verify_upper_bound: bool,
    k: usize,
) {
    let k = k.clamp(1, BAB_FRONTIER_MAX_K);
    let dims = root_input.len();
    if dims == 0 || dims > BAB_FRONTIER_MAX_DIMS {
        return;
    }
    // Guidance-only channel: a wrapped BFS ring buffer cannot produce a
    // contiguous view — skip the export rather than disturb the (already
    // decided) verdict path.
    let (Ok(lo_view), Ok(hi_view)) = (
        domain_list.input_lowers.tensor(),
        domain_list.input_uppers.tensor(),
    ) else {
        return;
    };
    let n = domain_list.metadata.len();
    if lo_view.shape().first() != Some(&n) || hi_view.shape().first() != Some(&n) {
        return;
    }
    let (root_lo, root_hi) = root_input.lower_upper();
    let root_lo: Vec<f32> = root_lo.iter().copied().collect();
    let root_hi: Vec<f32> = root_hi.iter().copied().collect();

    // (index, priority, margin, depth) per eligible domain; boxes are only
    // materialized for the K winners below.
    let mut candidates: Vec<(usize, f32, f32, usize)> = Vec::new();
    for (i, meta) in domain_list.metadata.iter().enumerate() {
        let lower = meta.lower_bound();
        let upper = meta.upper_bound();
        let Ok(priority) =
            BetaCrownConfig::domain_priority_for_mode(verify_upper_bound, lower, upper)
        else {
            continue; // non-finite bounds: never export
        };
        let lo_row = lo_view.index_axis(Axis(0), i);
        let hi_row = hi_view.index_axis(Axis(0), i);
        if lo_row.len() != dims || hi_row.len() != dims {
            continue; // arity mismatch vs the root box: never export
        }
        if !differs_from_root(lo_row.iter(), hi_row.iter(), &root_lo, &root_hi) {
            continue; // whole root box: no subbox of its own
        }
        candidates.push((i, priority, lower, meta.depth()));
    }
    keep_top_k_by_priority_desc(&mut candidates, k, |c| c.1);

    let out: Vec<BabFrontierSeed> = candidates
        .into_iter()
        .filter_map(|(i, _priority, margin, depth)| {
            seed_from_box(
                lo_view.index_axis(Axis(0), i).iter().copied().collect(),
                hi_view.index_axis(Axis(0), i).iter().copied().collect(),
                margin,
                depth,
            )
        })
        .collect();
    *lock() = out;
}

/// Env-gated wrapper for the GPU BaB DomainList lane's exhaust exit
/// (`check_termination` returning a terminal result inside the BaB loop).
/// Same gate and channel as the sequential lane; guidance only.
pub(crate) fn record_domain_list_frontier_if_enabled(
    domain_list: &DomainList,
    root_input: &BoundedTensor,
    verify_upper_bound: bool,
) {
    let Some((_mode, k)) = frontier_gate_mode_and_k() else {
        return;
    };
    record_domain_list_frontier(domain_list, root_input, verify_upper_bound, k);
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use ndarray::arr1;

    use super::*;
    use crate::beta_crown::branching::SplitHistory;
    use crate::beta_crown::domain::IntermediateLinearBounds;
    use crate::beta_crown::state::{BetaState, DomainAlphaState};

    /// A synthetic queue domain over the subbox `[lo, hi]^2`. The export only
    /// relies on the queue convention "higher `priority` = more
    /// violation-likely", so tests set `priority` explicitly.
    fn domain(lower_bound: f32, priority: f32, lo: f32, hi: f32, depth: usize) -> BabDomain {
        let bounds =
            BoundedTensor::new(arr1(&[lo, lo]).into_dyn(), arr1(&[hi, hi]).into_dyn()).unwrap();
        BabDomain {
            history: SplitHistory::new(),
            lower_bound,
            upper_bound: lower_bound + 1.0,
            priority,
            layer_bounds: vec![Arc::new(bounds.clone())],
            alpha_state: None,
            domain_alpha_state: DomainAlphaState::empty(),
            beta_state: BetaState::empty(),
            input_bounds: Some(Arc::new(bounds)),
            input_split_count: depth,
            intermediate_bounds: IntermediateLinearBounds::empty(),
        }
    }

    fn root_box() -> BoundedTensor {
        BoundedTensor::new(
            arr1(&[-1.0f32, -1.0]).into_dyn(),
            arr1(&[1.0f32, 1.0]).into_dyn(),
        )
        .unwrap()
    }

    /// Serialize tests touching the process-global channel.
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn keeps_exactly_top_k_sorted_most_violation_first() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let mut queue = BinaryHeap::new();
        // 7 domains, priorities -3..=3 in shuffled push order; K = 4 must keep
        // priorities [3, 2, 1, 0] in that order.
        for p in [0.0f32, -2.0, 3.0, -1.0, 2.0, -3.0, 1.0] {
            queue.push(domain(p, p, -0.5, 0.5, 1));
        }
        record_bab_frontier_from_queue(&queue, &root_box(), 4);
        let seeds = take_bab_frontier_seeds();
        assert_eq!(seeds.len(), 4, "exactly K survive");
        let margins: Vec<f32> = seeds.iter().map(|s| s.margin).collect();
        assert_eq!(margins, vec![3.0, 2.0, 1.0, 0.0], "priority DESC order");
    }

    #[test]
    fn take_clears_and_reset_clears() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let mut queue = BinaryHeap::new();
        queue.push(domain(-1.0, -1.0, -0.5, 0.5, 1));
        record_bab_frontier_from_queue(&queue, &root_box(), 8);
        assert_eq!(take_bab_frontier_seeds().len(), 1);
        assert!(take_bab_frontier_seeds().is_empty(), "take clears");

        record_bab_frontier_from_queue(&queue, &root_box(), 8);
        reset_bab_frontier_export();
        assert!(take_bab_frontier_seeds().is_empty(), "reset clears");
    }

    #[test]
    fn skips_non_finite_bounds_and_missing_input_bounds() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let mut queue = BinaryHeap::new();
        queue.push(domain(-1.0, -1.0, -0.5, 0.5, 1)); // eligible
        let mut nan_lower = domain(-2.0, -2.0, -0.5, 0.5, 1);
        nan_lower.lower_bound = f32::NAN;
        queue.push(nan_lower);
        let mut inf_priority = domain(-3.0, -3.0, -0.5, 0.5, 1);
        inf_priority.priority = f32::NEG_INFINITY;
        queue.push(inf_priority);
        let mut rootish = domain(-4.0, -4.0, -0.5, 0.5, 0);
        rootish.input_bounds = None; // root: no subbox of its own
        queue.push(rootish);

        record_bab_frontier_from_queue(&queue, &root_box(), 8);
        let seeds = take_bab_frontier_seeds();
        assert_eq!(seeds.len(), 1, "only the finite subbox domain exports");
        assert_eq!(seeds[0].margin, -1.0);
        assert_eq!(seeds[0].depth, 1);
    }

    #[test]
    fn every_exported_center_lies_inside_its_own_box() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let mut queue = BinaryHeap::new();
        for (i, (lo, hi)) in [(-1.0f32, -0.5f32), (-0.25, 0.25), (0.5, 1.0), (0.9, 1.0)]
            .into_iter()
            .enumerate()
        {
            queue.push(domain(-(i as f32), -(i as f32), lo, hi, i + 1));
        }
        record_bab_frontier_from_queue(&queue, &root_box(), 16);
        let seeds = take_bab_frontier_seeds();
        assert_eq!(seeds.len(), 4);
        for seed in &seeds {
            assert_eq!(seed.center.len(), seed.box_lo.len());
            assert_eq!(seed.center.len(), seed.box_hi.len());
            for d in 0..seed.center.len() {
                assert!(
                    seed.box_lo[d] <= seed.center[d] && seed.center[d] <= seed.box_hi[d],
                    "center[{d}]={} outside [{}, {}]",
                    seed.center[d],
                    seed.box_lo[d],
                    seed.box_hi[d]
                );
            }
            assert!(seed.depth > 0);
        }
    }

    /// A synthetic graph-lane heap domain over the box `[lo, hi]^2` with an
    /// explicit priority (graph heap convention: higher = more
    /// violation-likely, same as the sequential lane).
    fn graph_domain(
        lower_bound: f32,
        priority: f32,
        lo: f32,
        hi: f32,
        depth: usize,
    ) -> GraphBabDomain {
        let bounds =
            BoundedTensor::new(arr1(&[lo, lo]).into_dyn(), arr1(&[hi, hi]).into_dyn()).unwrap();
        let mut d = GraphBabDomain::root(
            std::collections::HashMap::new(),
            lower_bound,
            lower_bound + 1.0,
            &bounds,
            false,
        )
        .unwrap();
        d.priority = priority;
        d.depth = depth;
        d
    }

    #[test]
    fn graph_lane_skips_whole_root_box_domains_and_orders_subboxes() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let root = root_box();
        let domains = [
            // Whole root box: a pure ReLU-split domain — no subbox, skipped.
            graph_domain(-9.0, 9.0, -1.0, 1.0, 3),
            // Genuine subboxes (clip-shrunk / input-split): exported.
            graph_domain(-1.0, 1.0, -0.5, 0.5, 1),
            graph_domain(-2.0, 2.0, -0.25, 0.25, 2),
            // Non-finite priority: skipped.
            {
                let mut d = graph_domain(-3.0, 3.0, -0.5, 0.0, 2);
                d.priority = f32::NAN;
                d
            },
        ];
        record_graph_bab_frontier_from_domains(domains.iter(), &root, 8);
        let seeds = take_bab_frontier_seeds();
        assert_eq!(
            seeds.len(),
            2,
            "only the finite genuine-subbox domains export"
        );
        // Priority DESC: 2.0 before 1.0.
        assert_eq!(seeds[0].margin, -2.0);
        assert_eq!(seeds[0].depth, 2);
        assert_eq!(seeds[1].margin, -1.0);
        for seed in &seeds {
            for d in 0..2 {
                assert!(seed.box_lo[d] <= seed.center[d] && seed.center[d] <= seed.box_hi[d]);
            }
        }
    }

    #[test]
    fn oversized_input_dimension_exports_nothing() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        reset_bab_frontier_export();
        let mut queue = BinaryHeap::new();
        queue.push(domain(-1.0, -1.0, -0.5, 0.5, 1));
        let big = BoundedTensor::new(
            ndarray::ArrayD::from_elem(ndarray::IxDyn(&[BAB_FRONTIER_MAX_DIMS + 1]), -1.0f32),
            ndarray::ArrayD::from_elem(ndarray::IxDyn(&[BAB_FRONTIER_MAX_DIMS + 1]), 1.0f32),
        )
        .unwrap();
        record_bab_frontier_from_queue(&queue, &big, 8);
        assert!(take_bab_frontier_seeds().is_empty());
    }
}
