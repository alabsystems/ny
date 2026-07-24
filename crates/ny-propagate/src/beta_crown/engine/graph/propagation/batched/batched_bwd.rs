// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #lsnc-batched-bwd — the BATCHED-TENSOR CROWN backward for the input-split
//! lane (design-doc slice S3, `docs/LSNC_BATCH_TENSOR_DESIGN.md`).
//!
//! # What this replaces
//!
//! The reference dense-spec backward
//! (`propagate_crown_batched_backward_core_specs`) stores pending linear
//! bounds as per-domain `Option<LinearBounds>` objects
//! (`IndexedPendingLinearBounds`) and runs `accumulate_idx` — ~6 heap
//! allocations per call (`safe_add` map_collect ×2, `merge_bias` clone ×2,
//! `merge_coeff_err`, result `LinearBounds`) — `n_domains × n_nodes` times,
//! SERIALLY on the driver thread. MEASURED on real lsnc_relu this backward is
//! ~93% of the rebound wall and ~90% serial.
//!
//! This fast lane keeps pending state in contiguous SoA batch tensors
//! ([`BatchSlot`]: `[B, R, W]` coefficient/error planes + `[B, R]` biases +
//! per-domain presence masks, the design doc's `BatchPendingLinear`), so
//! accumulate/merge/safe_add run as flat, allocation-free, IN-PLACE batched
//! ops over the whole domain batch, parallelized with COARSE rayon chunks
//! over the domain axis (no per-domain fan-out).
//!
//! # Parity class: BIT-IDENTICAL (gate ON vs OFF)
//!
//! Every arithmetic step is either (a) shared reference code invoked
//! per-domain (`relu_backward_domain` / `div_backward_domain` /
//! `generic_backward_domain` in `backward_core.rs`, exactly the bodies the
//! reference loop runs), (b) a shared per-element primitive
//! (`GraphNetwork::safe_add_elem`, `indexed_pending::merge_bias_elem`,
//! `indexed_pending::merged_err_elem`), or (c) the SoA Linear kernel
//! (`crown_batched_soa.rs`), a transcription of the reference multi-domain
//! Linear backward that issues the IDENTICAL stacked engine GEMMs and runs
//! the identical per-domain certified-error block. The batch dimension never
//! reassociates any reduction: per-row f64 bias folds keep their `i = 0..n`
//! order inside the shared kernels, and the `aw_f64_with_abssum` / `γ_n·S`
//! certified-error accounting produces identical per-row results (workers
//! hold a `NestedFaerParGuard` so faer sees the same parallelism policy as
//! the reference's driver-thread calls).
//!
//! # Fail-closed decline discipline (pattern: `propagate_linear_multi_domain_relu`)
//!
//! Anything outside the clean class returns `Ok(None)` and the UNTOUCHED
//! reference loop runs: non-`Standard` mode, active β states, the stacked
//! cgan rebound (`input_split_stacked_rebound`), a `mo-beta-graft` capture,
//! or any structural surprise the SoA layout cannot represent bit-faithfully
//! (cross-domain shape divergence at a pending slot — unreachable for
//! graph-shaped contributions, but guarded). The fast lane owns ALL of its
//! state, so a mid-traversal decline discards it and the reference recomputes
//! from the seed — strictly a performance transform, never a bound change.
//!
//! Parity is pinned by `test_input_split_batched_bwd_*` in
//! `tests_soundness.rs` (raw f32 bit equality of output bounds, coefficient
//! matrices, biases, certified-error matrices — presence AND bits — and
//! `input_linear` masks, gate ON vs OFF, on an lsnc-shaped
//! MulBinary/MatMul/Concat/ReduceSum fixture) and the slot-merge parity test
//! below (`test_batch_slot_merge_matches_accumulate_idx`).

use std::collections::HashMap;
use std::sync::Arc;

use ndarray::Array2;
use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use rayon::prelude::*;

use crate::beta_crown::state::GraphDomainAlphaState;
use crate::layers::linear::crown_batched_soa::{
    propagate_linear_batched_soa, SoaLinearBackwardInput, SoaLinearBackwardOutput,
};
use crate::network::CrownDispatchPlan;
use crate::{GraphNetwork, Layer, LinearBounds};

use super::backward_core::{
    div_backward_domain, generic_backward_domain, relu_backward_domain, resolve_pre_activation,
};
use super::build_nodes_by_idx;
use super::indexed_pending::{merge_bias_elem, merged_err_elem};

/// Runtime gate for the #lsnc-batched-bwd SoA batched backward (slice S3).
///
/// `-1` = uninitialized (read the env once, then cache); `1` = ON; `0` = OFF.
///
/// Default ON. Parity class: BIT-IDENTICAL to the reference per-domain
/// pending loop — kernel parity `test_linear_backward_soa_matches_reference`,
/// slot-merge parity `test_batch_slot_merge_matches_accumulate_idx`,
/// full-pipeline gate-ON/OFF bit parity
/// `test_input_split_batched_bwd_full_pipeline_bit_identical` in
/// `tests_soundness.rs`. The end-to-end lsnc verdict-identity A/B ran green
/// (2026-07-18, uncontended, instances 0/1/3/5/6: no forbidden verdict flip;
/// `[disjunctive-multi-clause]` per-batch verified/clipped/best_gap
/// trajectories identical on every matched batch; matched-batch backward
/// 2.5–2.7x faster, big-batch dps 1.5–2.4x). Set
/// `NY_INPUT_SPLIT_BATCHED_BWD=0` to force the historical reference (the A/B
/// + parity baseline), mirroring `NY_INPUT_SPLIT_BATCHED_INTERM`.
///
/// Gate composition: the bit-parity claim holds under any engine
/// (`NY_BATCHED_NAIVE_ENGINE` included — both legs issue the identical
/// stacked GEMM calls) and assumes `NY_INPUT_SPLIT_NESTED_PAR` is unchanged
/// (default ON; with `=0` the per-domain faer f64 products inside rayon
/// workers run `Par::Seq`, which is bit-identical at lane-typical sizes where
/// faer never splits, and γ_n·S-sound in general — I-D2).
static BATCHED_BWD_MODE: std::sync::atomic::AtomicI8 = std::sync::atomic::AtomicI8::new(-1);

/// Whether the SoA batched backward is enabled (see [`BATCHED_BWD_MODE`]).
pub(super) fn input_split_batched_bwd_enabled() -> bool {
    use std::sync::atomic::Ordering;
    match BATCHED_BWD_MODE.load(Ordering::Relaxed) {
        0 => false,
        1 => true,
        _ => {
            let on = !matches!(
                std::env::var("NY_INPUT_SPLIT_BATCHED_BWD").ok().as_deref(),
                Some("0") | Some("false")
            );
            BATCHED_BWD_MODE.store(i8::from(on), Ordering::Relaxed);
            on
        }
    }
}

/// Test-only runtime override for the batched-bwd gate: `Some(true|false)`
/// forces ON/OFF, `None` restores the env-derived default. Mirrors
/// `force_batched_relu` so parity tests can A/B the exact same pipeline
/// without mutating process-global env. Tests MUST restore `None` afterward
/// and hold `spec_adapters::SPEC_GATE_TEST_LOCK` for their full body.
#[cfg(test)]
pub(crate) fn force_batched_bwd(mode: Option<bool>) {
    use std::sync::atomic::Ordering;
    let v = match mode {
        Some(true) => 1,
        Some(false) => 0,
        None => -1,
    };
    BATCHED_BWD_MODE.store(v, Ordering::Relaxed);
}

// Test-only probe: number of times the SoA batched backward actually ENGAGED
// (returned `Some`, i.e. did not decline) ON THE CURRENT THREAD. Lets the
// parity tests assert which leg ran (checklist Part 3 A, "decline/fallback
// leg exercised"). Thread-local rather than a process-global atomic: the
// forced-gate window is process-global, so a concurrent unlocked test's
// pipeline call that STARTED inside the window can finish (and would bump a
// global counter) after the window closed — a race the S3 parity tests must
// not see. The backward runs synchronously on the caller's thread, so the
// thread-local delta observes exactly the test's own calls.
#[cfg(test)]
thread_local! {
    pub(crate) static BATCHED_BWD_ENGAGED_ON_THREAD: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
}

/// Read the current thread's engagement count (see
/// [`BATCHED_BWD_ENGAGED_ON_THREAD`]).
#[cfg(test)]
pub(crate) fn engaged_on_thread() -> usize {
    BATCHED_BWD_ENGAGED_ON_THREAD.with(std::cell::Cell::get)
}

/// Minimum domains per rayon task: coarse chunks (a task is always many
/// domains of work), NO per-domain fan-out — the refuted lever from the
/// design doc's measured profile.
fn par_min_chunk(n_domains: usize) -> usize {
    let threads = rayon::current_num_threads().max(1);
    (n_domains / (threads * 4)).max(16)
}

/// SoA pending storage for one node: the design doc's `BatchLinearBounds`.
///
/// Planes are domain-major row-major (`[B, R, C]` flat); domain `d` is live
/// iff `active[d]`. Certified-error planes are always allocated (zeros);
/// `lower_err_present[d]` / `upper_err_present[d]` mirror the per-domain
/// `Option` semantics of `LinearBounds::{lower,upper}_a_err` (`None` = exact
/// marker, I-D5) so materialization reconstructs the exact same presence.
struct BatchSlot {
    rows: usize,
    cols: usize,
    active: Vec<bool>,
    lower_a: Vec<f32>,
    upper_a: Vec<f32>,
    lower_b: Vec<f32>,
    upper_b: Vec<f32>,
    lower_a_err: Vec<f32>,
    upper_a_err: Vec<f32>,
    lower_err_present: Vec<bool>,
    upper_err_present: Vec<bool>,
}

/// One domain's mutable region of a [`BatchSlot`] (disjoint across domains,
/// so the per-target merge parallelizes safely over the domain axis).
struct DomainRegion<'a> {
    active: &'a mut bool,
    la: &'a mut [f32],
    ua: &'a mut [f32],
    lb: &'a mut [f32],
    ub: &'a mut [f32],
    le: &'a mut [f32],
    ue: &'a mut [f32],
    lep: &'a mut bool,
    uep: &'a mut bool,
}

impl BatchSlot {
    fn new_empty(n_domains: usize, rows: usize, cols: usize) -> Self {
        let rc = rows * cols;
        Self {
            rows,
            cols,
            active: vec![false; n_domains],
            lower_a: vec![0.0; n_domains * rc],
            upper_a: vec![0.0; n_domains * rc],
            lower_b: vec![0.0; n_domains * rows],
            upper_b: vec![0.0; n_domains * rows],
            lower_a_err: vec![0.0; n_domains * rc],
            upper_a_err: vec![0.0; n_domains * rc],
            lower_err_present: vec![false; n_domains],
            upper_err_present: vec![false; n_domains],
        }
    }

    /// O(1) adoption of a SoA Linear backward result. The reference Linear arm
    /// attaches certified error unconditionally (`set_coeff_err`), so err
    /// presence == active.
    fn from_linear_out(out: SoaLinearBackwardOutput) -> Self {
        let lower_err_present = out.active.clone();
        let upper_err_present = out.active.clone();
        Self {
            rows: out.rows,
            cols: out.cols,
            active: out.active,
            lower_a: out.lower_a,
            upper_a: out.upper_a,
            lower_b: out.lower_b,
            upper_b: out.upper_b,
            lower_a_err: out.lower_a_err,
            upper_a_err: out.upper_a_err,
            lower_err_present,
            upper_err_present,
        }
    }

    fn any_active(&self) -> bool {
        self.active.iter().any(|&a| a)
    }

    /// Parallel iterator over per-domain regions (coarse chunks are applied by
    /// the caller via `with_min_len`).
    fn par_domain_regions(&mut self) -> impl IndexedParallelIterator<Item = DomainRegion<'_>> {
        let rc = self.rows * self.cols;
        let rows = self.rows;
        self.active
            .par_iter_mut()
            .zip(self.lower_a.par_chunks_mut(rc))
            .zip(self.upper_a.par_chunks_mut(rc))
            .zip(self.lower_b.par_chunks_mut(rows))
            .zip(self.upper_b.par_chunks_mut(rows))
            .zip(self.lower_a_err.par_chunks_mut(rc))
            .zip(self.upper_a_err.par_chunks_mut(rc))
            .zip(self.lower_err_present.par_iter_mut())
            .zip(self.upper_err_present.par_iter_mut())
            .map(
                |((((((((active, la), ua), lb), ub), le), ue), lep), uep)| DomainRegion {
                    active,
                    la,
                    ua,
                    lb,
                    ub,
                    le,
                    ue,
                    lep,
                    uep,
                },
            )
    }

    /// Materialize domain `d` as the exact `LinearBounds` the reference
    /// pending storage would hold (bits AND err presence). Uses a struct
    /// literal because accumulated pending bounds legitimately carry ±Inf
    /// coefficients (`safe_add`'s conservative degrade, #3032) that
    /// `LinearBounds::new` would reject.
    fn materialize(&self, d: usize) -> Option<LinearBounds> {
        if !self.active[d] {
            return None;
        }
        let rc = self.rows * self.cols;
        let arr2 = |src: &[f32]| {
            Array2::from_shape_vec((self.rows, self.cols), src[d * rc..(d + 1) * rc].to_vec())
                .expect("BatchSlot region is row-major by construction")
        };
        let arr1 =
            |src: &[f32]| ndarray::Array1::from(src[d * self.rows..(d + 1) * self.rows].to_vec());
        Some(LinearBounds {
            lower_a: arr2(&self.lower_a),
            lower_b: arr1(&self.lower_b),
            upper_a: arr2(&self.upper_a),
            upper_b: arr1(&self.upper_b),
            lower_a_err: self.lower_err_present[d].then(|| arr2(&self.lower_a_err)),
            upper_a_err: self.upper_err_present[d].then(|| arr2(&self.upper_a_err)),
        })
    }
}

/// Verbatim SoA transcription of `IndexedPendingLinearBounds::accumulate_idx`
/// for one domain region, split into insert (empty slot entry: store the
/// contribution's bits verbatim) and merge (NaN-safe coefficient add, 2Sum
/// outward bias merge, certified coefficient-error merge with the exact
/// presence pairing of `set_coeff_err`). Every element goes through the
/// shared per-element primitives, so the stored bits match the reference
/// storage exactly (`test_batch_slot_merge_matches_accumulate_idx`).
fn insert_region(reg: &mut DomainRegion<'_>, contrib: &LinearBounds) {
    for (dst, &src) in reg.la.iter_mut().zip(contrib.lower_a.iter()) {
        *dst = src;
    }
    for (dst, &src) in reg.ua.iter_mut().zip(contrib.upper_a.iter()) {
        *dst = src;
    }
    for (dst, &src) in reg.lb.iter_mut().zip(contrib.lower_b.iter()) {
        *dst = src;
    }
    for (dst, &src) in reg.ub.iter_mut().zip(contrib.upper_b.iter()) {
        *dst = src;
    }
    if let Some(e) = contrib.lower_a_err.as_ref() {
        for (dst, &src) in reg.le.iter_mut().zip(e.iter()) {
            *dst = src;
        }
        *reg.lep = true;
    }
    if let Some(e) = contrib.upper_a_err.as_ref() {
        for (dst, &src) in reg.ue.iter_mut().zip(e.iter()) {
            *dst = src;
        }
        *reg.uep = true;
    }
    *reg.active = true;
}

/// In-place NaN-safe coefficient merge for one side, replicating
/// `GraphNetwork::safe_add`'s shape handling exactly: same-shape → per-element
/// `safe_add_elem`; same-count reshape (via the SAME
/// `into_shape_with_order`) → per-element; otherwise the conservative ±Inf
/// fill.
fn merge_coeff_side(
    dst: &mut [f32],
    contrib: &Array2<f32>,
    rows: usize,
    cols: usize,
    is_lower: bool,
) {
    if contrib.nrows() == rows && contrib.ncols() == cols {
        for (o, &n) in dst.iter_mut().zip(contrib.iter()) {
            *o = GraphNetwork::safe_add_elem(*o, n, is_lower);
        }
        return;
    }
    if contrib.len() == rows * cols {
        if let Ok(reshaped) = contrib.view().into_shape_with_order((rows, cols)) {
            tracing::debug!(
                existing_shape = ?[rows, cols],
                new_shape = ?contrib.shape(),
                "safe_add reshaping same-count contribution before accumulation"
            );
            for (o, &n) in dst.iter_mut().zip(reshaped.iter()) {
                *o = GraphNetwork::safe_add_elem(*o, n, is_lower);
            }
            return;
        }
    }
    tracing::warn!(
        existing_shape = ?[rows, cols],
        new_shape = ?contrib.shape(),
        "safe_add shape mismatch; returning conservative bounds"
    );
    let conservative = if is_lower {
        f32::NEG_INFINITY
    } else {
        f32::INFINITY
    };
    dst.fill(conservative);
}

/// In-place bias merge for one side (`merge_bias` of `accumulate_idx`):
/// length mismatch → conservative fill; else the shared 2Sum element.
fn merge_bias_side(dst: &mut [f32], contrib: &ndarray::Array1<f32>, is_lower: bool) {
    if contrib.len() != dst.len() {
        let conservative = if is_lower {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        };
        dst.fill(conservative);
        return;
    }
    for (o, &bv) in dst.iter_mut().zip(contrib.iter()) {
        *o = merge_bias_elem(*o, bv, is_lower);
    }
}

/// Whether `merge_coeff_err` would return `Some` for one side, mirroring its
/// early-outs exactly: mis-shaped err → `Some` (∞ degrade); both inputs
/// absent AND merged coefficients all-zero → `None`; otherwise `Some`.
fn merged_err_would_be_some(
    existing_present: bool,
    new_err: Option<&Array2<f32>>,
    merged_coeff: &[f32],
    rows: usize,
    cols: usize,
) -> bool {
    if let Some(e) = new_err {
        if e.nrows() != rows || e.ncols() != cols {
            return true; // mis-shaped → conservative ∞ matrix
        }
    }
    if !existing_present && new_err.is_none() {
        return merged_coeff.iter().any(|&v| v != 0.0);
    }
    true
}

/// In-place certified coefficient-error merge for one side, given the ALREADY
/// merged coefficients in `merged_coeff` (the reference computes `new_la`
/// first and feeds it to `merge_coeff_err`). `existing_present == false`
/// means the err region holds zeros and is treated as `None`.
fn merge_err_side(
    dst: &mut [f32],
    existing_present: bool,
    new_err: Option<&Array2<f32>>,
    merged_coeff: &[f32],
    rows: usize,
    cols: usize,
) {
    let mis_shaped = new_err
        .map(|e| e.nrows() != rows || e.ncols() != cols)
        .unwrap_or(false);
    if mis_shaped {
        // If an error matrix is present but mis-shaped, we cannot map it
        // soundly: degrade every coefficient so concretize widens the rows.
        dst.fill(f32::INFINITY);
        return;
    }
    match new_err {
        Some(e) => {
            for ((o, &n), &mc) in dst.iter_mut().zip(e.iter()).zip(merged_coeff.iter()) {
                *o = merged_err_elem(existing_present.then_some(*o), Some(n), mc);
            }
        }
        None => {
            for (o, &mc) in dst.iter_mut().zip(merged_coeff.iter()) {
                *o = merged_err_elem(existing_present.then_some(*o), None, mc);
            }
        }
    }
}

/// Merge a per-domain `LinearBounds` contribution into an ACTIVE domain
/// region — the SoA twin of the `accumulate_idx` merge branch.
fn merge_region(reg: &mut DomainRegion<'_>, contrib: &LinearBounds, rows: usize, cols: usize) {
    // Coefficients first (the err merge consumes the MERGED coefficients).
    merge_coeff_side(reg.la, &contrib.lower_a, rows, cols, true);
    merge_coeff_side(reg.ua, &contrib.upper_a, rows, cols, false);
    merge_bias_side(reg.lb, &contrib.lower_b, true);
    merge_bias_side(reg.ub, &contrib.upper_b, false);

    let lower_some =
        merged_err_would_be_some(*reg.lep, contrib.lower_a_err.as_ref(), reg.la, rows, cols);
    let upper_some =
        merged_err_would_be_some(*reg.uep, contrib.upper_a_err.as_ref(), reg.ua, rows, cols);
    match (lower_some, upper_some) {
        (false, false) => {}
        _ => {
            // `set_coeff_err` pairing: a side whose merge yields `None` while
            // the other carries error gets an explicit ZERO error matrix so
            // the carried side is never silently dropped.
            if lower_some {
                merge_err_side(
                    reg.le,
                    *reg.lep,
                    contrib.lower_a_err.as_ref(),
                    reg.la,
                    rows,
                    cols,
                );
            } else {
                reg.le.fill(0.0);
            }
            if upper_some {
                merge_err_side(
                    reg.ue,
                    *reg.uep,
                    contrib.upper_a_err.as_ref(),
                    reg.ua,
                    rows,
                    cols,
                );
            } else {
                reg.ue.fill(0.0);
            }
            *reg.lep = true;
            *reg.uep = true;
        }
    }
}

/// Merge one domain of a SoA Linear result into an ACTIVE domain region.
/// Shapes are guaranteed equal by the caller (`out.rows/cols == slot dims`);
/// Linear contributions always carry both err planes, so the merged err is
/// always `Some` (mirrors `merged_err_would_be_some` with `new_err` present).
struct SoaContribRegion<'a> {
    la: &'a [f32],
    ua: &'a [f32],
    lb: &'a [f32],
    ub: &'a [f32],
    le: &'a [f32],
    ue: &'a [f32],
}

fn insert_region_from_soa(reg: &mut DomainRegion<'_>, c: &SoaContribRegion<'_>) {
    reg.la.copy_from_slice(c.la);
    reg.ua.copy_from_slice(c.ua);
    reg.lb.copy_from_slice(c.lb);
    reg.ub.copy_from_slice(c.ub);
    reg.le.copy_from_slice(c.le);
    reg.ue.copy_from_slice(c.ue);
    *reg.lep = true;
    *reg.uep = true;
    *reg.active = true;
}

fn merge_region_from_soa(reg: &mut DomainRegion<'_>, c: &SoaContribRegion<'_>) {
    for (o, &n) in reg.la.iter_mut().zip(c.la.iter()) {
        *o = GraphNetwork::safe_add_elem(*o, n, true);
    }
    for (o, &n) in reg.ua.iter_mut().zip(c.ua.iter()) {
        *o = GraphNetwork::safe_add_elem(*o, n, false);
    }
    for (o, &bv) in reg.lb.iter_mut().zip(c.lb.iter()) {
        *o = merge_bias_elem(*o, bv, true);
    }
    for (o, &bv) in reg.ub.iter_mut().zip(c.ub.iter()) {
        *o = merge_bias_elem(*o, bv, false);
    }
    let lep = *reg.lep;
    for ((o, &n), &mc) in reg.le.iter_mut().zip(c.le.iter()).zip(reg.la.iter()) {
        *o = merged_err_elem(lep.then_some(*o), Some(n), mc);
    }
    let uep = *reg.uep;
    for ((o, &n), &mc) in reg.ue.iter_mut().zip(c.ue.iter()).zip(reg.ua.iter()) {
        *o = merged_err_elem(uep.then_some(*o), Some(n), mc);
    }
    *reg.lep = true;
    *reg.uep = true;
}

/// The SoA replacement for `IndexedPendingLinearBounds` (plan-idx keyed).
struct BatchPending {
    slots: Vec<Option<BatchSlot>>,
    n_domains: usize,
    network_input_idx: usize,
    input_accumulated: Vec<bool>,
}

impl BatchPending {
    fn new(plan: &CrownDispatchPlan, n_domains: usize) -> Self {
        Self {
            slots: (0..=plan.node_count()).map(|_| None).collect(),
            n_domains,
            network_input_idx: plan.network_input_idx,
            input_accumulated: vec![false; n_domains],
        }
    }

    /// Seed the output node with the shared spec seed for EVERY domain — the
    /// values of the reference's per-domain `initial_lb.clone()` seeds without
    /// the per-domain allocations.
    fn seed_broadcast(&mut self, idx: usize, seed: &LinearBounds) -> Result<()> {
        let rows = seed.lower_a.nrows();
        let cols = seed.lower_a.ncols();
        if seed.upper_a.nrows() != rows
            || seed.upper_a.ncols() != cols
            || seed.lower_b.len() != rows
            || seed.upper_b.len() != rows
            || seed.lower_a_err.is_some()
            || seed.upper_a_err.is_some()
        {
            return Err(NyError::InvalidSpec(
                "batched-bwd seed must be a plain spec seed".to_string(),
            ));
        }
        let mut slot = BatchSlot::new_empty(self.n_domains, rows, cols);
        let rc = rows * cols;
        let la: Vec<f32> = seed.lower_a.iter().copied().collect();
        let ua: Vec<f32> = seed.upper_a.iter().copied().collect();
        let lb: Vec<f32> = seed.lower_b.iter().copied().collect();
        let ub: Vec<f32> = seed.upper_b.iter().copied().collect();
        for d in 0..self.n_domains {
            slot.lower_a[d * rc..(d + 1) * rc].copy_from_slice(&la);
            slot.upper_a[d * rc..(d + 1) * rc].copy_from_slice(&ua);
            slot.lower_b[d * rows..(d + 1) * rows].copy_from_slice(&lb);
            slot.upper_b[d * rows..(d + 1) * rows].copy_from_slice(&ub);
            slot.active[d] = true;
        }
        if idx == self.network_input_idx {
            self.input_accumulated.fill(true);
        }
        self.slots[idx] = Some(slot);
        Ok(())
    }

    fn take(&mut self, idx: usize) -> Option<BatchSlot> {
        self.slots[idx].take()
    }
}

/// Per-domain contribution list from one node's Phase A dispatch:
/// `(target plan idx, bounds)` in sink-call order (the order matters when one
/// domain contributes twice to the same target, e.g. `x·x`).
type DomainContribs = Vec<(usize, Option<LinearBounds>)>;

/// Outcome of the fast lane: `None` = decline (caller runs the reference
/// loop), `Some((input_accumulated, network-input pending))` = success.
type SoaBackwardOutcome = Option<(Vec<bool>, Option<Vec<Option<LinearBounds>>>)>;

/// Run the full dense-spec backward traversal on SoA pending storage.
/// See module docs for the parity/decline contract.
#[allow(clippy::too_many_arguments)]
pub(super) fn try_backward_soa(
    graph: &GraphNetwork,
    plan: &CrownDispatchPlan,
    n_domains: usize,
    bounds_caches: &[&HashMap<String, Arc<BoundedTensor>>],
    constrained_inputs: &[BoundedTensor],
    alpha_states: &[Option<&GraphDomainAlphaState>],
    initial_lb: &LinearBounds,
    engine: &dyn GemmEngine,
    deadline: Option<std::time::Instant>,
    mul_binary_alphas: Option<&HashMap<String, Array2<f32>>>,
) -> Result<SoaBackwardOutcome> {
    let nodes_by_idx = build_nodes_by_idx(graph, plan)?;
    let network_input_dim = constrained_inputs[0].len();
    let min_chunk = par_min_chunk(n_domains);

    let mut pending = BatchPending::new(plan, n_domains);
    pending.seed_broadcast(plan.output_node_idx, initial_lb)?;

    let resolve_target = |name: &str| -> Result<usize> {
        plan.name_to_idx.get(name).copied().ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "IndexedPendingLinearBounds: unknown input '{}'",
                name
            ))
        })
    };

    for &idx in &plan.reverse_order {
        let Some(slot) = pending.take(idx) else {
            continue;
        };
        if !slot.any_active() {
            continue;
        }
        let node_name = plan.name_of(idx);
        let node = nodes_by_idx[idx];

        let require_inputs = |min: usize| -> Result<()> {
            if node.inputs.len() < min {
                Err(NyError::InvalidSpec(format!(
                    "Node '{}' ({}) requires at least {} input(s), got {}",
                    node_name,
                    node.layer.layer_type(),
                    min,
                    node.inputs.len(),
                )))
            } else {
                Ok(())
            }
        };
        let require_unary_input = || -> Result<&str> {
            node.require_unary_input().map_err(|_| {
                NyError::InvalidSpec(format!(
                    "Node '{}' ({}) has no inputs for CROWN backward propagation",
                    node_name,
                    node.layer.layer_type()
                ))
            })
        };
        let require_binary_input_names = || -> Result<(&str, &str)> {
            node.require_binary_inputs().map_err(|_| {
                NyError::InvalidSpec(format!(
                    "Node '{}' ({}) requires 2 inputs for CROWN backward propagation but has {}",
                    node_name,
                    node.layer.layer_type(),
                    node.inputs.len()
                ))
            })
        };

        if let Layer::Linear(l) = &node.layer {
            require_inputs(1)?;
            let first_input = require_unary_input()?;
            let target = resolve_target(first_input)?;

            let soa_in = SoaLinearBackwardInput {
                n_domains,
                rows: slot.rows,
                cols: slot.cols,
                active: &slot.active,
                lower_a: &slot.lower_a,
                upper_a: &slot.upper_a,
                lower_b: &slot.lower_b,
                upper_b: &slot.upper_b,
                lower_a_err: Some(&slot.lower_a_err),
                upper_a_err: Some(&slot.upper_a_err),
                lower_err_present: &slot.lower_err_present,
                upper_err_present: &slot.upper_err_present,
            };
            let out = propagate_linear_batched_soa(l, &soa_in, engine, min_chunk)?;
            drop(slot);
            if !accumulate_linear_out(&mut pending, target, out, min_chunk) {
                return Ok(None); // structural decline → reference loop
            }
            continue;
        }

        // Non-Linear arms: Phase A runs the SHARED per-domain reference body
        // under coarse rayon chunks; Phase B applies the collected
        // contributions to the SoA slots (parallel over domains per target).
        let is_relu = matches!(&node.layer, Layer::ReLU(_));
        let is_div = matches!(&node.layer, Layer::Div(_));
        if is_relu {
            require_inputs(1)?;
        }
        if is_div {
            require_inputs(2)?;
        }
        let relu_first_input = if is_relu {
            Some(resolve_target(require_unary_input()?)?)
        } else {
            None
        };
        let div_names = if is_div {
            Some(require_binary_input_names()?)
        } else {
            None
        };

        let per_domain: Vec<Option<Result<DomainContribs>>> = (0..n_domains)
            .into_par_iter()
            .with_min_len(min_chunk)
            .map(|d| {
                if !slot.active[d] {
                    return None;
                }
                let _faer_par = crate::faer_parallelism::NestedFaerParGuard::new();
                let run = || -> Result<DomainContribs> {
                    let lb = slot.materialize(d).expect("active domain must materialize");
                    let mut out: DomainContribs = Vec::with_capacity(4);
                    if let Layer::ReLU(r) = &node.layer {
                        let pre = resolve_pre_activation(
                            &node.inputs,
                            &constrained_inputs[d],
                            bounds_caches[d],
                        )?;
                        // β declined upstream (input split carries no β), so
                        // the shared body's β arm is structurally inert.
                        let new_lb =
                            relu_backward_domain(node_name, r, lb, pre, alpha_states[d], None)?;
                        out.push((relu_first_input.expect("resolved above"), Some(new_lb)));
                        return Ok(out);
                    }
                    let mut sink = |input_name: &str, bounds: LinearBounds| -> Result<()> {
                        let target = resolve_target(input_name)?;
                        out.push((target, Some(bounds)));
                        Ok(())
                    };
                    if let Some((a_name, b_name)) = div_names {
                        div_backward_domain(
                            node_name,
                            a_name,
                            b_name,
                            lb,
                            &constrained_inputs[d],
                            bounds_caches[d],
                            network_input_dim,
                            &mut sink,
                        )?;
                    } else {
                        generic_backward_domain(
                            node_name,
                            node,
                            lb,
                            &constrained_inputs[d],
                            bounds_caches[d],
                            network_input_dim,
                            engine,
                            deadline,
                            mul_binary_alphas,
                            &mut sink,
                        )?;
                    }
                    Ok(out)
                };
                Some(run())
            })
            .collect();

        // First error in domain order wins (the reference loop processes
        // domains in ascending order and aborts on the first failure).
        let mut contribs: Vec<Option<DomainContribs>> = Vec::with_capacity(n_domains);
        let mut first_err: Option<NyError> = None;
        for entry in per_domain {
            match entry {
                None => contribs.push(None),
                Some(Ok(list)) => contribs.push(Some(list)),
                Some(Err(e)) => {
                    if first_err.is_none() {
                        first_err = Some(e);
                    }
                    contribs.push(None);
                }
            }
        }
        if let Some(e) = first_err {
            return Err(e);
        }
        drop(slot);

        if !apply_contributions(&mut pending, &mut contribs, min_chunk) {
            return Ok(None); // structural decline → reference loop
        }
    }

    let input_accumulated = pending.input_accumulated.clone();
    let input_bounds_vec = pending.take(pending.network_input_idx).map(|slot| {
        (0..n_domains)
            .map(|d| slot.materialize(d))
            .collect::<Vec<Option<LinearBounds>>>()
    });

    #[cfg(test)]
    BATCHED_BWD_ENGAGED_ON_THREAD.with(|c| c.set(c.get() + 1));
    tracing::debug!(
        n_domains,
        "SoA batched backward engaged (#lsnc-batched-bwd)"
    );

    Ok(Some((input_accumulated, input_bounds_vec)))
}

/// Accumulate a SoA Linear backward result into `target`. Returns `false` on
/// a structural decline (SoA cannot represent per-domain shape divergence at
/// an insert — unreachable for graph-shaped contributions).
fn accumulate_linear_out(
    pending: &mut BatchPending,
    target: usize,
    out: SoaLinearBackwardOutput,
    min_chunk: usize,
) -> bool {
    if target == pending.network_input_idx {
        for d in 0..pending.n_domains {
            if out.active[d] {
                pending.input_accumulated[d] = true;
            }
        }
    }
    let Some(slot) = pending.slots[target].as_mut() else {
        pending.slots[target] = Some(BatchSlot::from_linear_out(out));
        return true;
    };
    if slot.rows != out.rows || slot.cols != out.cols {
        // Insert-side shape divergence between the existing slot and a fresh
        // Linear contribution: the per-domain reference stores heterogeneous
        // shapes; SoA cannot — decline. (Merge-side mismatches degrade
        // per-element exactly like the reference and never take this path.)
        // Only domains INACTIVE in the slot force the decline; active domains
        // would degrade in-merge, but a single slot has a single dim pair, so
        // decline conservatively on any mismatch.
        return false;
    }
    let rc = out.rows * out.cols;
    let rows = out.rows;
    slot.par_domain_regions()
        .zip(out.active.par_iter())
        .enumerate()
        .with_min_len(min_chunk)
        .for_each(|(d, (mut reg, &contrib_active))| {
            if !contrib_active {
                return;
            }
            let c = SoaContribRegion {
                la: &out.lower_a[d * rc..(d + 1) * rc],
                ua: &out.upper_a[d * rc..(d + 1) * rc],
                lb: &out.lower_b[d * rows..(d + 1) * rows],
                ub: &out.upper_b[d * rows..(d + 1) * rows],
                le: &out.lower_a_err[d * rc..(d + 1) * rc],
                ue: &out.upper_a_err[d * rc..(d + 1) * rc],
            };
            if *reg.active {
                merge_region_from_soa(&mut reg, &c);
            } else {
                insert_region_from_soa(&mut reg, &c);
            }
        });
    true
}

/// Apply one node's per-domain contribution lists to the pending slots:
/// serial pre-scan (slot creation, insert-shape decline check,
/// `input_accumulated` flags — matching `accumulate_idx`'s flag-before-merge
/// order), then a parallel per-target apply over coarse domain chunks.
/// Returns `false` on structural decline.
fn apply_contributions(
    pending: &mut BatchPending,
    contribs: &mut [Option<DomainContribs>],
    min_chunk: usize,
) -> bool {
    // Targets in first-appearance order (domain-major, list order).
    let mut targets: Vec<usize> = Vec::new();
    for list in contribs.iter().flatten() {
        for &(t, _) in list.iter() {
            if !targets.contains(&t) {
                targets.push(t);
            }
        }
    }

    for &t in &targets {
        // Pre-scan: dims discovery + insert-compatibility + accumulated flags.
        let mut dims: Option<(usize, usize)> = pending.slots[t].as_ref().map(|s| (s.rows, s.cols));
        let mut sim_active: Vec<bool> = match pending.slots[t].as_ref() {
            Some(s) => s.active.clone(),
            None => vec![false; pending.n_domains],
        };
        for (d, list) in contribs.iter().enumerate() {
            let Some(list) = list else { continue };
            for (tt, lb) in list.iter() {
                if *tt != t {
                    continue;
                }
                let lb = lb.as_ref().expect("contribution not yet consumed");
                if t == pending.network_input_idx {
                    pending.input_accumulated[d] = true;
                }
                let c_rows = lb.lower_a.nrows();
                let c_cols = lb.lower_a.ncols();
                let (rows, cols) = *dims.get_or_insert((c_rows, c_cols));
                if !sim_active[d] {
                    // Insert must store the contribution verbatim: any shape
                    // the slot cannot hold verbatim is a structural decline.
                    if c_rows != rows
                        || c_cols != cols
                        || lb.upper_a.nrows() != rows
                        || lb.upper_a.ncols() != cols
                        || lb.lower_b.len() != rows
                        || lb.upper_b.len() != rows
                        || lb
                            .lower_a_err
                            .as_ref()
                            .is_some_and(|e| e.nrows() != rows || e.ncols() != cols)
                        || lb
                            .upper_a_err
                            .as_ref()
                            .is_some_and(|e| e.nrows() != rows || e.ncols() != cols)
                    {
                        return false;
                    }
                    sim_active[d] = true;
                }
            }
        }
        let (rows, cols) = dims.expect("target has at least one contribution");
        if pending.slots[t].is_none() {
            pending.slots[t] = Some(BatchSlot::new_empty(pending.n_domains, rows, cols));
        }

        // Parallel apply over domains (disjoint regions).
        let slot = pending.slots[t].as_mut().expect("slot created above");
        slot.par_domain_regions()
            .zip(contribs.par_iter_mut())
            .with_min_len(min_chunk)
            .for_each(|(mut reg, list)| {
                let Some(list) = list else { return };
                for (tt, lb_opt) in list.iter_mut() {
                    if *tt != t {
                        continue;
                    }
                    let lb = lb_opt.take().expect("each contribution consumed once");
                    if *reg.active {
                        merge_region(&mut reg, &lb, rows, cols);
                    } else {
                        insert_region(&mut reg, &lb);
                    }
                }
            });
    }
    true
}

#[cfg(test)]
mod tests {
    use ndarray::{arr1, Array1, Array2};

    use super::super::indexed_pending::IndexedPendingLinearBounds;
    use super::*;
    use crate::layers::LinearLayer;
    use crate::GraphNode;

    fn make_plan(node_name: &str) -> CrownDispatchPlan {
        let mut graph = GraphNetwork::new();
        graph
            .try_add_node(GraphNode::from_input(
                node_name,
                Layer::Linear(
                    LinearLayer::new(ndarray::arr2(&[[1.0_f32]]), Some(arr1(&[0.0_f32])))
                        .expect("valid linear layer"),
                ),
            ))
            .expect("node should be added");
        graph.set_output(node_name);
        CrownDispatchPlan::build(&graph).expect("dispatch plan should build")
    }

    fn lcg_stream(seed: u64) -> impl FnMut() -> f32 {
        let mut state = seed;
        move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((state >> 33) as f32) / (u32::MAX as f32);
            match (state >> 20) & 0x1f {
                0 => f32::INFINITY,
                1 => f32::NEG_INFINITY,
                2 => 0.0,
                _ => (u - 0.5) * 1e3,
            }
        }
    }

    fn assert_lb_bits_equal(a: &LinearBounds, b: &LinearBounds, ctx: &str) {
        let cmp2 = |x: &Array2<f32>, y: &Array2<f32>, what: &str| {
            assert_eq!(x.shape(), y.shape(), "{ctx}: {what} shape");
            for (i, (xa, ya)) in x.iter().zip(y.iter()).enumerate() {
                assert_eq!(xa.to_bits(), ya.to_bits(), "{ctx}: {what}[{i}]");
            }
        };
        let cmp1 = |x: &Array1<f32>, y: &Array1<f32>, what: &str| {
            assert_eq!(x.len(), y.len(), "{ctx}: {what} len");
            for (i, (xa, ya)) in x.iter().zip(y.iter()).enumerate() {
                assert_eq!(xa.to_bits(), ya.to_bits(), "{ctx}: {what}[{i}]");
            }
        };
        cmp2(&a.lower_a, &b.lower_a, "lower_a");
        cmp2(&a.upper_a, &b.upper_a, "upper_a");
        cmp1(&a.lower_b, &b.lower_b, "lower_b");
        cmp1(&a.upper_b, &b.upper_b, "upper_b");
        assert_eq!(
            a.lower_a_err.is_some(),
            b.lower_a_err.is_some(),
            "{ctx}: lower_a_err presence (None = exact marker, I-D5)"
        );
        assert_eq!(
            a.upper_a_err.is_some(),
            b.upper_a_err.is_some(),
            "{ctx}: upper_a_err presence"
        );
        if let (Some(x), Some(y)) = (a.lower_a_err.as_ref(), b.lower_a_err.as_ref()) {
            cmp2(x, y, "lower_a_err");
        }
        if let (Some(x), Some(y)) = (a.upper_a_err.as_ref(), b.upper_a_err.as_ref()) {
            cmp2(x, y, "upper_a_err");
        }
    }

    /// The SoA slot insert+merge must store BIT-IDENTICAL state to the
    /// reference `IndexedPendingLinearBounds::accumulate_idx` across: err
    /// present/absent on either side, ±Inf cancellation (NaN-safe add),
    /// all-zero seeds (the `None` err early-out), and repeated merges.
    /// Compares materialized `LinearBounds` (bits AND err presence) after
    /// every accumulation step. #lsnc-batched-bwd.
    #[ntest::timeout(60000)]
    #[test]
    fn test_batch_slot_merge_matches_accumulate_idx() {
        let plan = make_plan("node1");
        let (rows, cols) = (5usize, 6usize);
        let n_domains = 3usize;
        let mut next = lcg_stream(0xC0FFEE1234567890);

        let mk = |next: &mut dyn FnMut() -> f32, with_err: (bool, bool), zero: bool| {
            let f = |next: &mut dyn FnMut() -> f32| {
                if zero {
                    0.0
                } else {
                    next()
                }
            };
            let mut lb = LinearBounds {
                lower_a: Array2::from_shape_fn((rows, cols), |_| f(next)),
                lower_b: Array1::from_shape_fn(rows, |_| {
                    let v = f(next);
                    if v.is_nan() {
                        0.0
                    } else {
                        v
                    }
                }),
                upper_a: Array2::from_shape_fn((rows, cols), |_| f(next)),
                upper_b: Array1::from_shape_fn(rows, |_| {
                    let v = f(next);
                    if v.is_nan() {
                        0.0
                    } else {
                        v
                    }
                }),
                lower_a_err: None,
                upper_a_err: None,
            };
            // One-sided err carries exercise the set_coeff_err zero-pairing.
            if with_err.0 {
                lb.lower_a_err = Some(Array2::from_shape_fn((rows, cols), |_| f(next).abs()));
            }
            if with_err.1 {
                lb.upper_a_err = Some(Array2::from_shape_fn((rows, cols), |_| f(next).abs()));
            }
            lb
        };

        // A sequence of contributions per domain: insert, then merges with
        // every err-presence combination, plus an all-zero merge.
        let sequences: Vec<Vec<LinearBounds>> = (0..n_domains)
            .map(|d| {
                vec![
                    mk(&mut next, (d == 0, d == 0), d == 2),
                    mk(&mut next, (false, false), false),
                    mk(&mut next, (true, false), false),
                    mk(&mut next, (false, true), false),
                    mk(&mut next, (true, true), false),
                ]
            })
            .collect();

        let mut reference = IndexedPendingLinearBounds::new(&plan, n_domains);
        let mut slot: Option<BatchSlot> = None;
        for step in 0..sequences[0].len() {
            for d in 0..n_domains {
                let lb = sequences[d][step].clone();
                reference
                    .accumulate_name("node1", lb.clone(), d)
                    .expect("reference accumulate");
                let slot = slot.get_or_insert_with(|| BatchSlot::new_empty(n_domains, rows, cols));
                // Apply through the same region primitives the fast lane uses.
                let rc = rows * cols;
                let mut reg = DomainRegion {
                    active: &mut slot.active[d],
                    la: &mut slot.lower_a[d * rc..(d + 1) * rc],
                    ua: &mut slot.upper_a[d * rc..(d + 1) * rc],
                    lb: &mut slot.lower_b[d * rows..(d + 1) * rows],
                    ub: &mut slot.upper_b[d * rows..(d + 1) * rows],
                    le: &mut slot.lower_a_err[d * rc..(d + 1) * rc],
                    ue: &mut slot.upper_a_err[d * rc..(d + 1) * rc],
                    lep: &mut slot.lower_err_present[d],
                    uep: &mut slot.upper_err_present[d],
                };
                if *reg.active {
                    merge_region(&mut reg, &lb, rows, cols);
                } else {
                    insert_region(&mut reg, &lb);
                }
            }
            // Compare full materialized state after every step.
            let slot_ref = slot.as_ref().unwrap();
            for d in 0..n_domains {
                let want = reference
                    .get_name("node1")
                    .and_then(|per| per[d].clone())
                    .expect("reference entry");
                let got = slot_ref.materialize(d).expect("slot entry");
                assert_lb_bits_equal(&got, &want, &format!("step {step} domain {d}"));
            }
        }
    }
}
