// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Property-direction attention VJP with checked scalar Taylor envelopes.
//! No output-by-input Jacobian is constructed. Each query retains O(keys)
//! scratch; the six returned f32 coefficient matrices are the CROWN interface.
//! All stored-coefficient deviations are paid in the shared bias on this box.

use ndarray::{Array1, Array2};
use ny_core::{NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};
use std::time::Instant;

use super::crown_ternary::{
    parse_slice_dims, parse_slice_dims_with_key_storage, AttnTernaryResult, KeyStorage, SliceDims,
};
use super::{AttentionMask, SelfAttentionLayer};
use crate::certify::work::{WorkBudget, WorkRefusal, WorkRequest};
use crate::layers::softmax::weighted_support::{
    exp_nonpositive, softmax_tilt_tv_upper, up, Interval as I, WeightedScoreBox,
};
use crate::LinearBounds;

mod demand;
use demand::{visible_range, Demand, Summary as DemandSummary};

const MAX_OUTPUT_BYTES: usize = 256 * 1024 * 1024;
const MAX_CONSTRUCTION_BYTES: usize = 512 * 1024 * 1024;
const MAX_DEFAULT_WORK: u64 = 10_000_000_000;
// A checked exp uses fixed Taylor length and at most thirteen squarings;
// include its cached per-query construction and both per-direction min uses.
const TILT_CHECK_WORK_PER_QUERY: u64 = 4096;
const MIXED_WORK_PER_DIRECTION: u64 = 32;

fn checkpoint(deadline: Option<Instant>, budget: Option<&WorkBudget>) -> Result<()> {
    if deadline.is_some_and(|d| Instant::now() >= d) {
        Err(NyError::DeadlineExceeded("directional attention".into()))
    } else {
        budget.map_or(Ok(()), |b| b.checkpoint().map_err(work_refusal))
    }
}

fn input_endpoints(t: &BoundedTensor) -> Result<(&[f32], &[f32])> {
    let lower = t.lower().as_slice().ok_or_else(|| {
        NyError::UnsupportedOp("SelfAttention CROWN: non-contiguous input".into())
    })?;
    let upper = t.upper().as_slice().ok_or_else(|| {
        NyError::UnsupportedOp("SelfAttention CROWN: non-contiguous input".into())
    })?;
    Ok((lower, upper))
}

fn validate_input_box(
    lower: &[f32],
    upper: &[f32],
    deadline: Option<Instant>,
    budget: Option<&WorkBudget>,
) -> Result<()> {
    for (index, (&l, &u)) in lower.iter().zip(upper).enumerate() {
        if index.is_multiple_of(1024) {
            checkpoint(deadline, budget)?;
        }
        if !l.is_finite() || !u.is_finite() || l > u {
            return Err(NyError::UnsupportedOp(
                "directional attention requires finite input boxes".into(),
            ));
        }
    }
    Ok(())
}

fn fill_centers(
    lower: &[f32],
    upper: &[f32],
    center: &mut [f64],
    radius: &mut [f64],
    range: std::ops::Range<usize>,
    deadline: Option<Instant>,
    budget: Option<&WorkBudget>,
) -> Result<()> {
    for index in range {
        if index.is_multiple_of(1024) {
            checkpoint(deadline, budget)?;
        }
        let (l, u) = (lower[index], upper[index]);
        // The chosen binary64 point is a real expansion point; both endpoint
        // distances are independently outward rather than assuming an exact midpoint.
        let c = l as f64 + (u as f64 - l as f64) * 0.5;
        center[index] = c;
        radius[index] = I::point(l as f64)
            .sub(I::point(c))
            .abs_max()
            .max(I::point(u as f64).sub(I::point(c)).abs_max());
    }
    Ok(())
}

/// Read the original authored Kt endpoints in canonical cache order. The
/// logical iteration/poll order and each scalar operation match fill_centers;
/// only the original endpoint address differs. There is no index vector.
fn fill_authored_key_centers(
    lower: &[f32],
    upper: &[f32],
    center: &mut [f64],
    radius: &mut [f64],
    slice: usize,
    key: usize,
    sk: usize,
    dk: usize,
    mut poll: impl FnMut(usize) -> Result<()>,
) -> Result<()> {
    let base = slice * sk * dk + key * dk;
    for channel in 0..dk {
        let index = base + channel;
        // Poll logical cells, never a strided original address modulo.
        if index.is_multiple_of(1024) {
            poll(index)?;
        }
        let original = KeyStorage::AuthoredKt.column(slice, key, channel, sk, dk);
        let (l, u) = (lower[original], upper[original]);
        let c = l as f64 + (u as f64 - l as f64) * 0.5;
        center[index] = c;
        radius[index] = I::point(l as f64)
            .sub(I::point(c))
            .abs_max()
            .max(I::point(u as f64).sub(I::point(c)).abs_max());
    }
    Ok(())
}

fn oscillation(intervals: &[I]) -> f64 {
    if intervals.len() <= 1 {
        return 0.0;
    }
    let low = intervals.iter().map(|v| v.lo).fold(f64::INFINITY, f64::min);
    let high = intervals
        .iter()
        .map(|v| v.hi)
        .fold(f64::NEG_INFINITY, f64::max);
    I::point(high).sub(I::point(low)).hi.max(0.0)
}

/// The difference of any two independently bounded deviations is at most the
/// sum of their radii. One visible key has exactly zero softmax oscillation.
fn deviation_oscillation(radii: &[f64]) -> f64 {
    let (mut first, mut second) = (0.0_f64, 0.0_f64);
    for &r in radii {
        if r >= first {
            second = first;
            first = r;
        } else {
            second = second.max(r);
        }
    }
    if radii.len() <= 1 {
        0.0
    } else {
        I::point(first).add(I::point(second)).hi
    }
}

/// Update a stored coefficient and return the exact-real interval for its
/// increment. Both nearest coefficient casts and accumulation cancellation
/// are charged against deviation from the expansion point, with the constant
/// recalculated from the stored increment. No uncharged f64->f32 Jacobian exists.
fn add_gradient(
    stored: &mut f32,
    gradient: I,
    center: f64,
    radius: f64,
    dot_center: &mut I,
    affine_radius: &mut I,
    numeric_error: &mut I,
) -> Result<()> {
    if !gradient.finite() {
        return Err(NyError::UnsupportedOp(
            "nonfinite attention derivative".into(),
        ));
    }
    let old = *stored;
    let new = (old as f64 + gradient.midpoint()) as f32;
    if !new.is_finite() {
        return Err(NyError::UnsupportedOp(
            "attention coefficient overflow".into(),
        ));
    }
    let increment = I::point(new as f64).sub(I::point(old as f64));
    *stored = new;
    *dot_center = dot_center.add(increment.mul(I::point(center)));
    *affine_radius = affine_radius.add(I::point(increment.abs_max()).mul(I::point(radius)));
    *numeric_error =
        numeric_error.add(I::point(gradient.sub(increment).abs_max()).mul(I::point(radius)));
    Ok(())
}

/// Production adapter; the authoritative instance deadline is never cleared or
/// replaced by a fresh phase allowance. The checkpoint seam accepts earlier
/// caller deadlines when the common work-budget wrapper routes this operation.
pub(super) fn propagate(
    layer: &SelfAttentionLayer,
    node: &LinearBounds,
    q: &BoundedTensor,
    k: &BoundedTensor,
    v: &BoundedTensor,
) -> Result<AttnTernaryResult> {
    propagate_with_deadline(layer, node, q, k, v, ny_core::instance_budget::deadline())
}

pub(super) fn propagate_with_deadline(
    layer: &SelfAttentionLayer,
    node: &LinearBounds,
    q: &BoundedTensor,
    k: &BoundedTensor,
    v: &BoundedTensor,
    deadline: Option<Instant>,
) -> Result<AttnTernaryResult> {
    propagate_inner(
        layer,
        node,
        q,
        k,
        v,
        deadline,
        None,
        true,
        true,
        KeyStorage::Canonical,
    )
}

#[cfg(test)]
pub(super) fn propagate_with_budget(
    layer: &SelfAttentionLayer,
    node: &LinearBounds,
    q: &BoundedTensor,
    k: &BoundedTensor,
    v: &BoundedTensor,
    budget: &WorkBudget,
) -> Result<AttnTernaryResult> {
    propagate_with_budget_and_deadline(
        layer,
        node,
        q,
        k,
        v,
        budget,
        ny_core::instance_budget::deadline(),
    )
}

pub(super) fn propagate_with_budget_and_deadline(
    layer: &SelfAttentionLayer,
    node: &LinearBounds,
    q: &BoundedTensor,
    k: &BoundedTensor,
    v: &BoundedTensor,
    budget: &WorkBudget,
    deadline: Option<Instant>,
) -> Result<AttnTernaryResult> {
    propagate_inner(
        layer,
        node,
        q,
        k,
        v,
        deadline,
        Some(budget),
        true,
        true,
        KeyStorage::Canonical,
    )
}

pub(super) fn propagate_with_authored_kt_and_budget_and_deadline(
    layer: &SelfAttentionLayer,
    node: &LinearBounds,
    q: &BoundedTensor,
    kt: &BoundedTensor,
    v: &BoundedTensor,
    budget: &WorkBudget,
    deadline: Option<Instant>,
) -> Result<AttnTernaryResult> {
    propagate_inner(
        layer,
        node,
        q,
        kt,
        v,
        deadline,
        Some(budget),
        true,
        true,
        KeyStorage::AuthoredKt,
    )
}

#[cfg(test)]
pub(super) fn propagate_linear_mixed_for_test(
    layer: &SelfAttentionLayer,
    node: &LinearBounds,
    q: &BoundedTensor,
    k: &BoundedTensor,
    v: &BoundedTensor,
) -> Result<AttnTernaryResult> {
    propagate_inner(
        layer,
        node,
        q,
        k,
        v,
        ny_core::instance_budget::deadline(),
        None,
        false,
        true,
        KeyStorage::Canonical,
    )
}

/// The same checked arithmetic with zero-demand routing disabled. This is a
/// numerical control, not a second production authority or fallback.
#[cfg(test)]
pub(super) fn propagate_dense_queries_for_test(
    layer: &SelfAttentionLayer,
    node: &LinearBounds,
    q: &BoundedTensor,
    k: &BoundedTensor,
    v: &BoundedTensor,
) -> Result<AttnTernaryResult> {
    propagate_inner(
        layer,
        node,
        q,
        k,
        v,
        ny_core::instance_budget::deadline(),
        None,
        true,
        false,
        KeyStorage::Canonical,
    )
}

fn propagate_inner(
    layer: &SelfAttentionLayer,
    node: &LinearBounds,
    q: &BoundedTensor,
    k: &BoundedTensor,
    v: &BoundedTensor,
    deadline: Option<Instant>,
    budget: Option<&WorkBudget>,
    use_checked_tilt: bool,
    skip_zero_demand: bool,
    key_storage: KeyStorage,
) -> Result<AttnTernaryResult> {
    let deadline = match (deadline, ny_core::instance_budget::deadline()) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    };
    checkpoint(deadline, budget)?;
    // Adjacent-float widening requires binary64 nearest rounding and gradual
    // underflow. This is an arithmetic premise, not a resource admission.
    ny_core::require_f64_interval_proof_environment()?;
    let dims = match key_storage {
        KeyStorage::Canonical => parse_slice_dims(layer, q, k, v)?,
        KeyStorage::AuthoredKt => parse_slice_dims_with_key_storage(layer, q, k, v, key_storage)?,
    };
    let (sq, sk, dk, dv) = (dims.sq, dims.sk, dims.dk, dims.dv);
    // Match the shipped CausalSoftmax frontend's existing shape contract.
    // Windowed mismatch is refused earlier by the explicit mask-soundness gate.
    if layer.mask == AttentionMask::Causal && sq > sk {
        return Err(NyError::InvalidSpec(format!(
            "causal attention requires seq_q <= seq_k, got {sq} > {sk}"
        )));
    }
    let n_y = dims
        .n_slices
        .checked_mul(sq)
        .and_then(|n| n.checked_mul(dv))
        .ok_or_else(|| NyError::UnsupportedOp("attention shape overflow".into()))?;
    if node.num_inputs() != n_y {
        return Err(NyError::UnsupportedOp(
            "attention objective shape mismatch".into(),
        ));
    }
    let (n_q, n_k, n_v) = (q.len(), k.len(), v.len());
    let output_bytes = n_q
        .checked_add(n_k)
        .and_then(|n| n.checked_add(n_v))
        .and_then(|n| n.checked_mul(node.num_outputs()))
        .and_then(|n| n.checked_mul(8));
    if output_bytes.is_none_or(|bytes| bytes > MAX_OUTPUT_BYTES) {
        return Err(NyError::UnsupportedOp(
            "directional attention output exceeds memory admission".into(),
        ));
    }
    // Inspect all layouts and typed shapes before copies or center caches.
    let (ql, qu) = input_endpoints(q)?;
    let (kl, ku) = input_endpoints(k)?;
    let (vl, vu) = input_endpoints(v)?;
    let input = n_q
        .checked_add(n_k)
        .and_then(|n| n.checked_add(n_v))
        .ok_or_else(|| NyError::UnsupportedOp("attention input size overflow".into()))?;
    let preliminary = input
        .checked_mul(24)
        .and_then(|n| output_bytes.and_then(|o| n.checked_add(o)))
        .ok_or_else(|| NyError::UnsupportedOp("attention construction size overflow".into()))?;
    if preliminary > MAX_CONSTRUCTION_BYTES {
        return Err(NyError::UnsupportedOp(
            "directional attention construction admission".into(),
        ));
    }
    let demand = Demand::scan(layer, node, &dims, deadline, budget, skip_zero_demand)?;
    let request = predict_work(n_q, n_k, n_v, node.num_outputs(), dk, dv, &demand.summary)?;
    if request.scratch_bytes > MAX_CONSTRUCTION_BYTES
        || (budget.is_none() && request.work > MAX_DEFAULT_WORK)
    {
        return Err(NyError::UnsupportedOp(
            "directional attention construction admission".into(),
        ));
    }
    // The demand map and scan work already have a retained lease. Only the
    // remaining construction is acquired here, from the same common ledger.
    let remaining = WorkRequest {
        scratch_bytes: request.scratch_bytes - demand.summary.map_bytes - 16 * 1024,
        work: request.work - demand.summary.scan_work,
        ..request
    };
    let mut lease = budget
        .map(|b| b.lease(remaining).map_err(work_refusal))
        .transpose()?;
    let scale = layer.resolve_scale(q)? as f64;
    if !scale.is_finite() {
        return Err(NyError::UnsupportedOp("nonfinite attention scale".into()));
    }
    for (lower, upper) in [(ql, qu), (kl, ku), (vl, vu)] {
        validate_input_box(lower, upper, deadline, budget)?;
    }
    let centers = |len| {
        if demand.summary.active_queries == 0 {
            Vec::new()
        } else {
            vec![0.0_f64; len]
        }
    };
    let (mut qc, mut qr) = (centers(n_q), centers(n_q));
    let (mut kc, mut kr) = (centers(n_k), centers(n_k));
    let (mut vc, mut vr) = (centers(n_v), centers(n_v));
    let key_cache_len = dims
        .n_slices
        .checked_mul(sk)
        .ok_or_else(|| NyError::UnsupportedOp("attention key cache overflow".into()))?;
    let mut key_ready = if demand.summary.active_queries == 0 {
        Vec::new()
    } else {
        vec![0u8; key_cache_len]
    };
    checkpoint(deadline, budget)?;
    let outputs = node.num_outputs();
    let mut aq = [
        Array2::<f32>::zeros((outputs, n_q)),
        Array2::<f32>::zeros((outputs, n_q)),
    ];
    let mut ak = [
        Array2::<f32>::zeros((outputs, n_k)),
        Array2::<f32>::zeros((outputs, n_k)),
    ];
    let mut av = [
        Array2::<f32>::zeros((outputs, n_v)),
        Array2::<f32>::zeros((outputs, n_v)),
    ];
    let mut biases = [
        node.lower_b()
            .iter()
            .map(|&x| I::point(x as f64))
            .collect::<Vec<_>>(),
        node.upper_b()
            .iter()
            .map(|&x| I::point(x as f64))
            .collect::<Vec<_>>(),
    ];

    for slice in 0..dims.n_slices {
        let (qb, kb, vb, yb) = (
            slice * sq * dk,
            slice * sk * dk,
            slice * sk * dv,
            slice * sq * dv,
        );
        for query in 0..sq {
            checkpoint(deadline, budget)?;
            if !demand.active(slice, query, sq) {
                continue;
            }
            fill_centers(
                ql,
                qu,
                &mut qc,
                &mut qr,
                qb + query * dk..qb + (query + 1) * dk,
                deadline,
                budget,
            )?;
            let (key_start, key_end) = visible_range(layer, query, sk);
            for key in key_start..key_end {
                if key % 64 == 0 {
                    checkpoint(deadline, budget)?;
                }
                if key_ready[slice * sk + key] == 0 {
                    match key_storage {
                        KeyStorage::Canonical => fill_centers(
                            kl,
                            ku,
                            &mut kc,
                            &mut kr,
                            kb + key * dk..kb + (key + 1) * dk,
                            deadline,
                            budget,
                        )?,
                        KeyStorage::AuthoredKt => fill_authored_key_centers(
                            kl,
                            ku,
                            &mut kc,
                            &mut kr,
                            slice,
                            key,
                            sk,
                            dk,
                            |_| checkpoint(deadline, budget),
                        )?,
                    }
                    fill_centers(
                        vl,
                        vu,
                        &mut vc,
                        &mut vr,
                        vb + key * dv..vb + (key + 1) * dv,
                        deadline,
                        budget,
                    )?;
                    key_ready[slice * sk + key] = 1;
                }
            }
            let keys: Vec<usize> = (key_start..key_end).collect();
            if keys.is_empty() {
                return Err(NyError::UnsupportedOp("empty attention mask".into()));
            }
            let mut score_center = Vec::with_capacity(keys.len());
            let mut score_radius = Vec::with_capacity(keys.len());
            let mut original_radius = Vec::with_capacity(keys.len());
            let mut qk_residual = Vec::with_capacity(keys.len());
            for &key in &keys {
                if key % 64 == 0 {
                    checkpoint(deadline, budget)?;
                }
                let mut center = I::point(0.0);
                let mut radius = I::point(0.0);
                let mut residual = I::point(0.0);
                let mut relative_radius = I::point(0.0);
                for d in 0..dk {
                    let (qi, ki) = (qb + query * dk + d, kb + key * dk + d);
                    let reference = kb + keys[0] * dk + d;
                    // Subtract the SAME score carrier before ranging. Uniform
                    // center-key shifts cancel algebraically, including Q's
                    // uncertainty. This does not treat uncertain keys as fixed.
                    let difference = I::point(kc[ki]).sub(I::point(kc[reference]));
                    center = center.add(I::point(qc[qi]).mul(difference));
                    let cross = I::point(qr[qi]).mul(I::point(kr[ki]));
                    radius = radius
                        .add(I::point(qc[qi].abs()).mul(I::point(kr[ki])))
                        .add(I::point(kc[ki].abs()).mul(I::point(qr[qi])))
                        .add(cross);
                    residual = residual.add(cross);
                    if key != keys[0] {
                        let key_radius = I::point(kr[ki]).add(I::point(kr[reference]));
                        relative_radius = relative_radius
                            .add(I::point(qc[qi].abs()).mul(key_radius))
                            .add(I::point(difference.abs_max()).mul(I::point(qr[qi])))
                            .add(I::point(qr[qi]).mul(key_radius));
                    }
                }
                score_center.push(center.mul(I::point(scale)));
                score_radius.push(relative_radius.mul(I::point(scale.abs())).hi);
                original_radius.push(radius.mul(I::point(scale.abs())).hi);
                qk_residual.push(residual.mul(I::point(scale.abs())).hi);
            }
            if score_center.iter().any(|x| !x.finite())
                || score_radius.iter().any(|x| !x.is_finite())
            {
                return Err(NyError::UnsupportedOp(
                    "attention score arithmetic overflow".into(),
                ));
            }
            let shift = score_center
                .iter()
                .map(|x| x.hi)
                .fold(f64::NEG_INFINITY, f64::max);
            let mut exponentials = Vec::with_capacity(keys.len());
            let mut denominator = I::point(0.0);
            for s in &score_center {
                let shifted = s.sub(I::point(shift));
                let e = I::new(
                    exp_nonpositive(shifted.lo).lo,
                    exp_nonpositive(shifted.hi.min(0.0)).hi,
                );
                denominator = denominator.add(e);
                exponentials.push(e);
            }
            let probabilities: Vec<I> = if keys.len() == 1 {
                vec![I::point(1.0)]
            } else {
                exponentials
                    .iter()
                    .map(|e| e.div_positive(denominator))
                    .collect()
            };
            if probabilities.iter().any(|p| !p.finite()) {
                return Err(NyError::UnsupportedOp(
                    "unresolved attention center probability".into(),
                ));
            }
            let score_lower: Vec<f64> = score_center
                .iter()
                .zip(&score_radius)
                .map(|(s, &r)| s.sub(I::point(r)).lo)
                .collect();
            let score_upper: Vec<f64> = score_center
                .iter()
                .zip(&score_radius)
                .map(|(s, &r)| s.add(I::point(r)).hi)
                .collect();
            let support = WeightedScoreBox::new(&score_lower, &score_upper);
            let score_osc =
                deviation_oscillation(&score_radius).min(deviation_oscillation(&original_radius));
            checkpoint(deadline, budget)?;
            // For c=c0+dc and s=s0+ds, the exact mixed residual is
            // dc·(p(s0+ds)-p(s0)). It is bounded by osc(dc)*TV.
            // ds includes the QK bilinear increment; the linear derivative's
            // omitted QK term is charged separately below. The score-only
            // Taylor remainder and the affine carrier remain unchanged.
            let tv = if use_checked_tilt {
                softmax_tilt_tv_upper(score_osc)
            } else {
                f64::INFINITY
            };
            checkpoint(deadline, budget)?;

            for output in 0..outputs {
                if output % 16 == 0 {
                    checkpoint(deadline, budget)?;
                }
                for side in 0..2 {
                    let incoming = if side == 0 {
                        node.lower_a()
                    } else {
                        node.upper_a()
                    };
                    let incoming_err = if side == 0 {
                        node.lower_a_err()
                    } else {
                        node.upper_a_err()
                    };
                    let direction: Vec<f64> = (0..dv)
                        .map(|j| incoming[[output, yb + query * dv + j]] as f64)
                        .collect();
                    if direction.iter().any(|x| !x.is_finite()) {
                        return Err(NyError::UnsupportedOp(
                            "nonfinite attention objective".into(),
                        ));
                    }
                    let mut previous_error = I::point(0.0);
                    if let Some(err) = incoming_err {
                        for j in 0..dv {
                            let magnitude = keys
                                .iter()
                                .map(|&key| {
                                    vl[vb + key * dv + j].abs().max(vu[vb + key * dv + j].abs())
                                        as f64
                                })
                                .fold(0.0, f64::max);
                            previous_error = previous_error.add(
                                I::point(err[[output, yb + query * dv + j]] as f64)
                                    .mul(I::point(magnitude)),
                            );
                        }
                    }
                    if !previous_error.finite() {
                        return Err(NyError::UnsupportedOp(
                            "nonfinite incoming attention coefficient error".into(),
                        ));
                    }
                    biases[side][output] = if side == 0 {
                        biases[side][output].sub(previous_error)
                    } else {
                        biases[side][output].add(previous_error)
                    };
                    if direction.iter().all(|&c| c == 0.0) {
                        continue;
                    }
                    let mut values = Vec::with_capacity(keys.len());
                    let mut value_radius = Vec::with_capacity(keys.len());
                    let mut value_lower = Vec::with_capacity(keys.len());
                    let mut value_upper = Vec::with_capacity(keys.len());
                    for &key in &keys {
                        let mut center = I::point(0.0);
                        let mut radius = I::point(0.0);
                        let mut range = I::point(0.0);
                        for (j, &c) in direction.iter().enumerate() {
                            let vi = vb + key * dv + j;
                            center = center.add(I::point(c).mul(I::point(vc[vi])));
                            radius = radius.add(I::point(c.abs()).mul(I::point(vr[vi])));
                            range =
                                range.add(I::point(c).mul(I::new(vl[vi] as f64, vu[vi] as f64)));
                        }
                        values.push(center);
                        value_radius.push(radius.hi);
                        value_lower.push(range.lo);
                        value_upper.push(range.hi);
                    }
                    let mut value_center = I::point(0.0);
                    for (p, c) in probabilities.iter().zip(&values) {
                        value_center = value_center.add(p.mul(*c));
                    }
                    let score_gradient: Vec<I> = if keys.len() == 1 {
                        vec![I::point(0.0)]
                    } else {
                        probabilities
                            .iter()
                            .zip(&values)
                            .map(|(p, c)| p.mul(c.sub(value_center)))
                            .collect()
                    };
                    let score_term = I::point(oscillation(&values))
                        .mul(I::point(score_osc))
                        .mul(I::point(score_osc))
                        .mul(I::point(0.125));
                    let mixed_allowance = I::point(score_osc).mul(I::point(0.25)).hi.min(tv);
                    let mixed_term = I::point(deviation_oscillation(&value_radius))
                        .mul(I::point(mixed_allowance));
                    let mut remainder = score_term.add(mixed_term);
                    for (g, &r) in score_gradient.iter().zip(&qk_residual) {
                        remainder = remainder.add(I::point(g.abs_max()).mul(I::point(r)));
                    }
                    let (mut dot_center, mut affine_radius, mut numeric_error) =
                        (I::point(0.0), I::point(0.0), I::point(0.0));
                    for d in 0..dk {
                        let mut grad = I::point(0.0);
                        for (local, &key) in keys.iter().enumerate() {
                            let difference = I::point(kc[kb + key * dk + d])
                                .sub(I::point(kc[kb + keys[0] * dk + d]));
                            grad = grad.add(score_gradient[local].mul(difference));
                        }
                        let qi = qb + query * dk + d;
                        add_gradient(
                            &mut aq[side][[output, qi]],
                            grad.mul(I::point(scale)),
                            qc[qi],
                            qr[qi],
                            &mut dot_center,
                            &mut affine_radius,
                            &mut numeric_error,
                        )?;
                    }
                    for (local, &key) in keys.iter().enumerate() {
                        if local % 64 == 0 {
                            checkpoint(deadline, budget)?;
                        }
                        for d in 0..dk {
                            let ki = kb + key * dk + d;
                            let destination = key_storage.column(slice, key, d, sk, dk);
                            let grad = score_gradient[local]
                                .mul(I::point(scale))
                                .mul(I::point(qc[qb + query * dk + d]));
                            add_gradient(
                                &mut ak[side][[output, destination]],
                                grad,
                                kc[ki],
                                kr[ki],
                                &mut dot_center,
                                &mut affine_radius,
                                &mut numeric_error,
                            )?;
                        }
                        for (j, &c) in direction.iter().enumerate() {
                            let vi = vb + key * dv + j;
                            add_gradient(
                                &mut av[side][[output, vi]],
                                probabilities[local].mul(I::point(c)),
                                vc[vi],
                                vr[vi],
                                &mut dot_center,
                                &mut affine_radius,
                                &mut numeric_error,
                            )?;
                        }
                    }
                    let mut margin = remainder.add(numeric_error).hi;
                    if let Some(support) = &support {
                        let gap = if side == 0 {
                            value_center
                                .add(affine_radius)
                                .sub(I::point(support.lower(&value_lower)))
                        } else {
                            I::point(support.upper(&value_upper))
                                .sub(value_center.sub(affine_radius))
                        };
                        margin = margin.min(gap.hi.max(0.0));
                    }
                    if !margin.is_finite() || !value_center.finite() {
                        return Err(NyError::UnsupportedOp(
                            "attention Taylor enclosure overflow".into(),
                        ));
                    }
                    let constant = value_center.sub(dot_center);
                    biases[side][output] = biases[side][output].add(if side == 0 {
                        I::point(constant.lo).sub(I::point(up(margin)))
                    } else {
                        I::point(constant.hi).add(I::point(up(margin)))
                    });
                }
            }
        }
    }
    checkpoint(deadline, budget)?;
    let lower_bias = Array1::from_iter(biases[0].iter().enumerate().map(|(row, b)| {
        if b.lo == node.lower_b()[row] as f64 && b.hi == b.lo {
            node.lower_b()[row]
        } else {
            next_down_f32(b.lo as f32)
        }
    }));
    let upper_bias = Array1::from_iter(biases[1].iter().enumerate().map(|(row, b)| {
        if b.hi == node.upper_b()[row] as f64 && b.lo == b.hi {
            node.upper_b()[row]
        } else {
            next_up_f32(b.hi as f32)
        }
    }));
    let zeros = || Array1::<f32>::zeros(outputs);
    let [ql, qu] = aq;
    let [kl, ku] = ak;
    let [vl, vu] = av;
    let result = (
        vec![
            Some(LinearBounds::new_or_conservative(ql, zeros(), qu, zeros())?),
            Some(LinearBounds::new_or_conservative(kl, zeros(), ku, zeros())?),
            Some(LinearBounds::new_or_conservative(vl, zeros(), vu, zeros())?),
        ],
        lower_bias,
        upper_bias,
    );
    checkpoint(deadline, budget)?;
    if let Some(lease) = &mut lease {
        lease.complete().map_err(work_refusal)?;
    }
    Ok(result)
}

fn work_refusal(reason: WorkRefusal) -> NyError {
    if reason == WorkRefusal::Deadline {
        NyError::DeadlineExceeded("directional attention work budget".into())
    } else {
        NyError::UnsupportedOp(format!("directional attention work admission: {reason:?}"))
    }
}

#[allow(clippy::too_many_arguments)]
fn predict_work(
    nq: usize,
    nk: usize,
    nv: usize,
    outputs: usize,
    dk: usize,
    dv: usize,
    demand: &DemandSummary,
) -> Result<WorkRequest> {
    let overflow =
        || NyError::UnsupportedOp("directional attention work prediction overflow".into());
    let input = nq
        .checked_add(nk)
        .and_then(|n| n.checked_add(nv))
        .ok_or_else(overflow)?;
    let matrices = input
        .checked_mul(outputs)
        .and_then(|n| n.checked_mul(8))
        .ok_or_else(overflow)?;
    let scratch = input
        .checked_mul(24)
        .and_then(|n| n.checked_add(matrices))
        .and_then(|n| {
            demand
                .max_visible_keys
                .checked_mul(256)
                .and_then(|s| n.checked_add(s))
        })
        .and_then(|n| dv.checked_mul(8).and_then(|s| n.checked_add(s)))
        .and_then(|n| outputs.checked_mul(64).and_then(|s| n.checked_add(s)))
        .and_then(|n| n.checked_add(demand.map_bytes))
        .and_then(|n| n.checked_add(32 * 1024))
        .ok_or_else(overflow)?;
    let width = dk.checked_add(dv).ok_or_else(overflow)? as u64;
    let per_key = (outputs as u64)
        .checked_mul(width)
        .and_then(|n| n.checked_mul(512))
        .and_then(|n| n.checked_add(4096))
        .ok_or_else(overflow)?;
    let query_work = (outputs as u64)
        .checked_mul(2)
        .and_then(|n| n.checked_mul(MIXED_WORK_PER_DIRECTION))
        .and_then(|n| n.checked_add(TILT_CHECK_WORK_PER_QUERY))
        .ok_or_else(overflow)?;
    let linear_work = (input as u64)
        .checked_mul(
            (outputs as u64)
                .checked_mul(16)
                .and_then(|n| n.checked_add(64))
                .ok_or_else(overflow)?,
        )
        .ok_or_else(overflow)?;
    let work = (demand.visible_pairs as u64)
        .checked_mul(per_key)
        .and_then(|n| {
            (demand.active_queries as u64)
                .checked_mul(query_work)
                .and_then(|q| n.checked_add(q))
        })
        .and_then(|n| n.checked_add(demand.scan_work))
        .and_then(|n| n.checked_add(linear_work))
        .ok_or_else(overflow)?;
    Ok(WorkRequest {
        scratch_bytes: scratch,
        persistent_bytes: 0,
        work,
        proposals: 1,
    })
}

#[cfg(test)]
mod layout_tests;

#[cfg(test)]
mod work_tests {
    use super::*;
    #[test]
    fn direction_first_memory_prediction_removes_dense_jacobian_workspace() {
        let keys = 1024;
        let width = 64;
        let demand = DemandSummary {
            active_queries: 1,
            visible_pairs: keys,
            max_visible_keys: keys,
            scan_work: 128,
            map_bytes: 1,
        };
        let request =
            predict_work(width, keys * width, keys * width, 1, width, width, &demand).unwrap();
        let old_jacobian_bytes = 2 * width * (keys * width) * 4;
        assert!(request.scratch_bytes < old_jacobian_bytes / 4);
        assert!(predict_work(usize::MAX, 1, 1, 1, 1, 1, &demand).is_err());
        assert!(predict_work(1, 1, 1, usize::MAX, 1, 1, &demand).is_err());
    }
}
