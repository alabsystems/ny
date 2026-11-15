// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Fixed-space original-affine geometry ADVICE for exactly nine/ten cells.
//!
//! All f64 operations below are ordinary checked arithmetic, NOT outward
//! certificate arithmetic. In particular Q8*s*W is not assumed exact. No bound,
//! certificate, scalar merge or verdict leaves this module: only a normalized
//! group choice. The owner must separately authenticate raw source/map custody,
//! preflight this scan PLUS the full execution quote, and subsequently obtain
//! both fresh private certificates and the one checked complete fold.
use super::{
    checkpoint, empty_choice, normalize_group, put_group, AdmissionObservation, BundleChoice,
    NormalizedGroup,
};
use crate::certify::work::WorkLease;
use crate::layers::{normalization::BatchNormLayer, Layer};
use crate::network::{GraphNetwork, NETWORK_INPUT};
use ny_tensor::BoundedTensor;
use std::collections::HashMap;
use std::time::Instant;

const SOURCE_WORK: u64 = 1 << 20;
const ENUMERATION_WORK: u64 = 1 << 20;
const WORK_PER_TRIPLE_COLUMN: u64 = 1024;
const GRAM_WORK_PER_COLUMN: u64 = 8192;
const SHORTLIST: usize = 12;
const MAX_INPUT: usize = 65536;
const MAX_WIDTH: usize = 8192;
const SCRATCH: usize = 8192;

type Item = (usize, f64, f64);
#[derive(Clone, Copy)]
struct BnView<'a> {
    scale: &'a [f32],
    bias: &'a [f32],
    scale_error: &'a [f32],
    bias_error: &'a [f32],
}
/// Borrowed advice inputs; construction never authenticates a proof source.
pub(in super::super) struct GeometrySource<'a> {
    weights: &'a [f32],
    bias: Option<&'a [f32]>,
    lower: &'a [f32],
    upper: &'a [f32],
    bn: Option<BnView<'a>>,
    input_dim: usize,
    width: usize,
}
#[derive(Default)]
struct TripleWorkspace {
    linear: [[f64; 2]; 3],
    signed: [[f64; 2]; 4],
    scales: [f64; 3],
    shifts: [f64; 3],
    scale_errors: [f64; 3],
    bias_errors: [f64; 3],
    weighted: [f64; 3],
    column: [f64; 3],
}
struct GramWorkspace {
    // Upper triangle only: G[i,j] lives at j*(j+1)/2+i for i<=j.
    gram: [f64; 55],
    means: [f64; 10],
    scales: [f64; 10],
    column: [f64; 10],
}
impl Default for GramWorkspace {
    fn default() -> Self {
        Self {
            gram: [0.; 55],
            means: [0.; 10],
            scales: [0.; 10],
            column: [0.; 10],
        }
    }
}
#[derive(Clone, Copy, Default)]
struct ShortCandidate {
    mask: u16,
    residual: f64,
}
#[derive(Clone, Copy, Debug)]
struct AdviceScore {
    #[cfg_attr(not(test), expect(dead_code, reason = "component regression controls"))]
    unary: f64,
    #[cfg_attr(not(test), expect(dead_code, reason = "component regression controls"))]
    signed_max: f64,
    #[cfg_attr(not(test), expect(dead_code, reason = "component regression controls"))]
    radius: f64,
    score: f64,
}
fn safe(v: f64) -> Option<f64> {
    ny_core::is_crown_coeff_safe_f64(v).then_some(v)
}
fn scalar(v: f32) -> Option<f64> {
    ny_core::is_crown_coeff_safe(v).then_some(f64::from(v))
}
fn add(a: f64, b: f64) -> Option<f64> {
    safe(a + b)
}
fn mul(a: f64, b: f64) -> Option<f64> {
    safe(a * b)
}
fn name_ok(name: &str) -> bool {
    !name.is_empty() && name.len() <= 1024
}
fn vector_shape(bound: &BoundedTensor, n: usize) -> bool {
    (matches!(bound.shape(), [d] if *d == n) || matches!(bound.shape(), [1, d] if *d == n))
        && bound.len() == n
}
fn scratch_bytes() -> Option<usize> {
    // Parent ranking stays live. partition owns two Item8 arrays while
    // normalize_group owns a third sorted copy. triple_score has one workspace
    // (it returns only AdviceScore), so no fictitious workspace return is used.
    // Name the small shortlist/residual arrays explicitly within the original
    // 4KiB reserve; its remainder covers scalars, iterators, call metadata and
    // bounded shape/name/BN validation temporaries. Parent provisional quotes
    // have their own owner64KiB guard and are not hidden in this8KiB allowance.
    let small_arrays = size_of::<[(usize, usize); 10]>()
        .checked_add(size_of::<[usize; 3]>())?
        .checked_add(2 * size_of::<[f64; 3]>())?;
    let bookkeeping = 4096usize.checked_sub(small_arrays)?;
    size_of::<[Item; 16]>()
        .checked_add(4 * size_of::<BundleChoice>())?
        .checked_add(3 * size_of::<[Item; 8]>())?
        .checked_add(4 * size_of::<NormalizedGroup>())?
        .checked_add(2 * size_of::<GeometrySource<'_>>())?
        .checked_add(size_of::<TripleWorkspace>())?
        .checked_add(2 * size_of::<AdviceScore>())?
        .checked_add(size_of::<GramWorkspace>())?
        .checked_add(2 * size_of::<[ShortCandidate; SHORTLIST]>())?
        .checked_add(small_arrays)?
        .checked_add(bookkeeping)
}
fn validate_bn<'a>(
    bn: &'a BatchNormLayer,
    shape: &[usize],
    width: usize,
    lease: &WorkLease,
    end: Instant,
    obs: &mut AdmissionObservation,
) -> Option<BnView<'a>> {
    if bn.num_channels != width
        || [
            bn.scale.shape(),
            bn.bias.shape(),
            bn.scale_err.shape(),
            bn.bias_err.shape(),
        ]
        .iter()
        .any(|s| *s != [width])
    {
        return None;
    }
    let view = BnView {
        scale: bn.scale.as_slice()?,
        bias: bn.bias.as_slice()?,
        scale_error: bn.scale_err.as_slice()?,
        bias_error: bn.bias_err.as_slice()?,
    };
    for (values, radius) in [
        (view.scale, false),
        (view.bias, false),
        (view.scale_error, true),
        (view.bias_error, true),
    ] {
        for chunk in values.chunks(1024) {
            checkpoint(lease, end, obs)?;
            if chunk.iter().any(|&x| {
                !ny_core::is_crown_coeff_safe(x)
                    || (radius && x.to_bits() & 0x8000_0000 != 0 && x.to_bits() != 0x8000_0000)
            }) {
                return None;
            }
        }
    }
    // This helper has hidden channel scans; the four lengths were bounded first
    // and the fixed source debit includes those scans as well as the above ones.
    if bn.gpu_extraction_layout(shape, Some(width)).ok()? != (width, 1) {
        return None;
    }
    Some(view)
}

#[allow(clippy::too_many_arguments)]
pub(in super::super) fn geometry_quote<'a>(
    graph: &'a GraphNetwork,
    input: &'a BoundedTensor,
    map: &'a HashMap<String, BoundedTensor>,
    target: &str,
    lease: &mut WorkLease,
    end: Instant,
    obs: &mut AdmissionObservation,
) -> Option<GeometrySource<'a>> {
    obs.stage = "geometry-source";
    obs.reason = "geometry-source-work";
    lease.reserve_more_work(SOURCE_WORK).ok()?;
    checkpoint(lease, end, obs)?;
    obs.reason = "geometry-source";
    if scratch_bytes()? > SCRATCH || !name_ok(target) {
        return None;
    }
    let relu = graph.node(target)?;
    if !matches!(relu.layer(), Layer::ReLU(_)) {
        return None;
    }
    let pre_name = relu.require_unary_input().ok()?;
    if !name_ok(pre_name) {
        return None;
    }
    let pre = graph.node(pre_name)?;
    let (linear, parent, bn, bn_input) = match pre.layer() {
        Layer::Linear(l) => (l, pre.require_unary_input().ok()?, None, None),
        Layer::BatchNorm(b) => {
            let linear_name = pre.require_unary_input().ok()?;
            if !name_ok(linear_name) {
                return None;
            }
            let node = graph.node(linear_name)?;
            let Layer::Linear(l) = node.layer() else {
                return None;
            };
            (
                l,
                node.require_unary_input().ok()?,
                Some(b),
                Some(map.get(linear_name)?),
            )
        }
        _ => return None,
    };
    if !name_ok(parent) {
        return None;
    }
    let x = if parent == NETWORK_INPUT {
        input
    } else {
        map.get(parent)?
    };
    let (e, d) = (linear.in_features(), linear.out_features());
    if !(1..=MAX_INPUT).contains(&e)
        || !(9..=MAX_WIDTH).contains(&d)
        || !vector_shape(x, e)
        || !vector_shape(map.get(pre_name)?, d)
        || linear.weight.shape() != [d, e]
        || linear.bias.as_ref().is_some_and(|b| b.shape() != [d])
    {
        return None;
    }
    let bn = if let Some(b) = bn {
        let bx = bn_input?;
        if !vector_shape(bx, d) {
            return None;
        }
        Some(validate_bn(b, bx.shape(), d, lease, end, obs)?)
    } else {
        None
    };
    let weights = linear.weight.as_slice()?;
    let bias = match linear.bias.as_ref() {
        Some(b) => Some(b.as_slice()?),
        None => None,
    };
    let lower = x.lower().as_slice()?;
    let upper = x.upper().as_slice()?;
    if weights.len() != d.checked_mul(e)?
        || lower.len() != e
        || upper.len() != e
        || bias.is_some_and(|b| b.len() != d)
    {
        return None;
    }
    checkpoint(lease, end, obs)?;
    Some(GeometrySource {
        weights,
        bias,
        lower,
        upper,
        bn,
        input_dim: e,
        width: d,
    })
}

impl GeometrySource<'_> {
    pub(in super::super) fn work(&self, count: usize) -> Option<u64> {
        if !matches!(count, 9 | 10) {
            return None;
        }
        GRAM_WORK_PER_COLUMN
            .checked_add(WORK_PER_TRIPLE_COLUMN.checked_mul(SHORTLIST as u64)?)?
            .checked_mul(u64::try_from(self.input_dim).ok()?.checked_add(1)?)?
            .checked_add(ENUMERATION_WORK)
    }
    fn gram(
        &self,
        ranked: &[Item; 16],
        count: usize,
        out: &mut GramWorkspace,
        lease: &WorkLease,
        end: Instant,
        obs: &mut AdmissionObservation,
    ) -> Option<()> {
        for i in 0..count {
            let n = ranked[i].0;
            let (scale, shift) = self.bn.map_or((1., 0.), |v| (v.scale[n], v.bias[n]));
            out.scales[i] = scalar(scale)?;
            out.means[i] = add(
                mul(out.scales[i], scalar(self.bias.map_or(0., |b| b[n]))?)?,
                scalar(shift)?,
            )?;
        }
        for column in 0..self.input_dim {
            if column % 32 == 0 {
                checkpoint(lease, end, obs)?;
            }
            let lo = scalar(self.lower[column])?;
            let hi = scalar(self.upper[column])?;
            if lo > hi {
                return None;
            }
            let center = mul(add(lo, hi)?, 0.5)?;
            let radius = mul(add(hi, -lo)?, 0.5)?;
            for i in 0..count {
                let coefficient = mul(
                    out.scales[i],
                    scalar(self.weights[ranked[i].0 * self.input_dim + column])?,
                )?;
                out.means[i] = add(out.means[i], mul(coefficient, center)?)?;
                out.column[i] = mul(coefficient, radius)?;
            }
            for j in 0..count {
                for i in 0..=j {
                    let k = j * (j + 1) / 2 + i;
                    out.gram[k] = add(out.gram[k], mul(out.column[i], out.column[j])?)?;
                }
            }
        }
        // Include the nominal affine mean as one extra coordinate. This is
        // approximate Gram advice; BN uncertainty is charged by direct scoring,
        // and neither phase authenticates a certificate or emptiness result.
        for j in 0..count {
            for i in 0..=j {
                let k = j * (j + 1) / 2 + i;
                out.gram[k] = add(out.gram[k], mul(out.means[i], out.means[j])?)?;
            }
        }
        checkpoint(lease, end, obs)
    }
    fn triple_score(
        &self,
        group: &NormalizedGroup,
        lease: &WorkLease,
        end: Instant,
        obs: &mut AdmissionObservation,
    ) -> Option<AdviceScore> {
        if group.active != 3 {
            return None;
        }
        let mut work = TripleWorkspace::default();
        for i in 0..3 {
            let n = group.neurons[i];
            if n >= self.width {
                return None;
            }
            let b = scalar(self.bias.map_or(0., |v| v[n]))?;
            work.linear[i] = [b, b];
            work.weighted[i] = scalar(group.weights[i])?;
            let (s, t, es, et) = self.bn.map_or((1., 0., 0., 0.), |v| {
                (v.scale[n], v.bias[n], v.scale_error[n], v.bias_error[n])
            });
            work.scales[i] = scalar(s)?;
            work.shifts[i] = scalar(t)?;
            work.scale_errors[i] = scalar(es)?;
            work.bias_errors[i] = scalar(et)?;
            work.column[i] = mul(
                work.weighted[i],
                add(mul(work.scales[i], b)?, work.shifts[i])?,
            )?;
        }
        for mask in 0..4 {
            let mut center = work.column[0];
            for i in 1..3 {
                let term = if mask & (1 << (i - 1)) == 0 {
                    work.column[i]
                } else {
                    -work.column[i]
                };
                center = add(center, term)?;
            }
            work.signed[mask] = [center, center];
        }
        for j in 0..self.input_dim {
            if j % 32 == 0 {
                checkpoint(lease, end, obs)?;
            }
            let lo = scalar(self.lower[j])?;
            let hi = scalar(self.upper[j])?;
            if lo > hi {
                return None;
            }
            for i in 0..3 {
                let w = scalar(self.weights[group.neurons[i] * self.input_dim + j])?;
                let a = mul(w, lo)?;
                let b = mul(w, hi)?;
                work.linear[i][0] = add(work.linear[i][0], a.min(b))?;
                work.linear[i][1] = add(work.linear[i][1], a.max(b))?;
                work.column[i] = mul(mul(work.weighted[i], work.scales[i])?, w)?;
            }
            for mask in 0..4 {
                let mut coefficient = work.column[0];
                for i in 1..3 {
                    let term = if mask & (1 << (i - 1)) == 0 {
                        work.column[i]
                    } else {
                        -work.column[i]
                    };
                    coefficient = add(coefficient, term)?;
                }
                let a = mul(coefficient, lo)?;
                let b = mul(coefficient, hi)?;
                work.signed[mask][0] = add(work.signed[mask][0], a.min(b))?;
                work.signed[mask][1] = add(work.signed[mask][1], a.max(b))?;
            }
        }
        checkpoint(lease, end, obs)?;
        let mut unary = 0.;
        let mut radius = 0.;
        for i in 0..3 {
            let lo = work.linear[i][0];
            let hi = work.linear[i][1];
            let error = add(
                mul(work.scale_errors[i], lo.abs().max(hi.abs()))?,
                work.bias_errors[i],
            )?;
            if error < 0. {
                return None;
            }
            radius = add(radius, mul(work.weighted[i], error)?)?;
            let a = add(mul(work.scales[i], lo)?, work.shifts[i])?;
            let b = add(mul(work.scales[i], hi)?, work.shifts[i])?;
            let lower = add(a.min(b), -error)?;
            let upper = add(a.max(b), error)?;
            unary = add(unary, mul(work.weighted[i], lower.abs().max(upper.abs()))?)?;
        }
        let mut signed_max = 0f64;
        for [lo, hi] in work.signed {
            signed_max = signed_max.max(-lo).max(hi);
        }
        signed_max = add(signed_max, radius)?;
        let score = mul(
            scalar(group.multiplier)?,
            add(unary, -unary.min(signed_max))?,
        )?;
        if unary < 0. || signed_max < 0. || score < 0. {
            return None;
        }
        Some(AdviceScore {
            unary,
            signed_max,
            radius,
            score,
        })
    }

    /// Reserves the exact scan debit. Owner must check scan+fullfold combined
    /// admission BEFORE calling. No other-family retry is permitted after entry.
    pub(in super::super) fn evaluate(
        &self,
        ranked: &[Item; 16],
        count: usize,
        preferred: &BundleChoice,
        lease: &mut WorkLease,
        end: Instant,
        obs: &mut AdmissionObservation,
    ) -> Option<BundleChoice> {
        obs.stage = "geometry-advice";
        obs.reason = "geometry-work";
        lease.reserve_more_work(self.work(count)?).ok()?;
        checkpoint(lease, end, obs)?;
        obs.reason = "geometry-values";
        if scratch_bytes()? > SCRATCH || count > self.width {
            return None;
        }
        for i in 0..count {
            let (id, score, magnitude) = ranked[i];
            if id >= self.width
                || safe(score).is_none()
                || score <= 0.
                || safe(magnitude).is_none()
                || magnitude <= 0.
                || ranked[..i].iter().any(|v| v.0 == id)
            {
                return None;
            }
        }
        let mut gram = GramWorkspace::default();
        self.gram(ranked, count, &mut gram, lease, end, obs)?;
        let (shortlist, used) = shortlist(&gram, ranked, count, preferred, lease, end, obs)?;
        let mut best = *preferred;
        let mut best_score = 0f64;
        for candidate in &shortlist[..used] {
            checkpoint(lease, end, obs)?;
            let current = partition(ranked, count, candidate.mask)?;
            let group1 = NormalizedGroup {
                active: 3,
                neurons: current.neurons[1],
                weights: current.weights[1],
                multiplier: current.multipliers[1],
            };
            let advice = self.triple_score(&group1, lease, end, obs)?;
            if advice.score > best_score
                || (advice.score > 0.
                    && advice.score == best_score
                    && current.neurons < best.neurons)
            {
                best = current;
                best_score = advice.score;
            }
        }
        checkpoint(lease, end, obs)?;
        obs.reason = "none";
        Some(best)
    }
}

fn shortlist(
    gram: &GramWorkspace,
    ranked: &[Item; 16],
    count: usize,
    preferred: &BundleChoice,
    lease: &WorkLease,
    end: Instant,
    obs: &mut AdmissionObservation,
) -> Option<([ShortCandidate; SHORTLIST], usize)> {
    // Existing preferred normalization is retained exactly as a candidate.
    if preferred.active != [count - 3, 3] {
        return None;
    }
    let mut preferred_mask = 0u16;
    for id in &preferred.neurons[1][..3] {
        let position = ranked[..count].iter().position(|v| v.0 == *id)?;
        if preferred_mask & (1 << position) != 0 {
            return None;
        }
        preferred_mask |= 1 << position;
    }
    if partition(ranked, count, preferred_mask)? != *preferred {
        return None;
    }
    let mut shortlist = [ShortCandidate::default(); SHORTLIST];
    let mut used = 0usize;
    let mut preferred_entry = None;
    let mut ordered = [(0usize, 0usize); 10];
    for i in 0..count {
        ordered[i] = (ranked[i].0, i);
    }
    ordered[..count].sort_unstable_by_key(|v| v.0);
    for a in 0..count - 2 {
        for b in a + 1..count - 1 {
            for c in b + 1..count {
                checkpoint(lease, end, obs)?;
                let mask = (1u16 << a) | (1u16 << b) | (1u16 << c);
                let Some(current) = partition(ranked, count, mask) else {
                    continue;
                };
                let residual = gram_residual(gram, ranked, count, &current)?;
                let candidate = ShortCandidate { mask, residual };
                if mask == preferred_mask {
                    preferred_entry = Some(candidate);
                }
                let position = (0..used)
                    .find(|&i| {
                        residual < shortlist[i].residual
                            || (residual == shortlist[i].residual
                                && complement_less(mask, shortlist[i].mask, &ordered[..count]))
                    })
                    .unwrap_or(used);
                if position < SHORTLIST {
                    let next = (used + 1).min(SHORTLIST);
                    for i in (position + 1..next).rev() {
                        shortlist[i] = shortlist[i - 1];
                    }
                    shortlist[position] = candidate;
                    used = next;
                }
            }
        }
    }
    let preferred_entry = preferred_entry?;
    if !shortlist[..used].iter().any(|v| v.mask == preferred_mask) {
        if used < SHORTLIST {
            shortlist[used] = preferred_entry;
            used += 1;
        } else {
            shortlist[SHORTLIST - 1] = preferred_entry;
        }
    }
    Some((shortlist, used))
}

fn partition(ranked: &[Item; 16], count: usize, mask: u16) -> Option<BundleChoice> {
    if !matches!(count, 9 | 10) || mask.count_ones() != 3 || mask >> count != 0 {
        return None;
    }
    let mut triple = [(0, 0., 0.); 8];
    let mut complement = [(0, 0., 0.); 8];
    let mut t = 0;
    let mut rest = 0;
    for (i, &item) in ranked[..count].iter().enumerate() {
        if mask & (1 << i) != 0 {
            triple[t] = item;
            t += 1;
        } else {
            complement[rest] = item;
            rest += 1;
        }
    }
    let group0 = normalize_group(&complement[..rest]).ok()?;
    let group1 = normalize_group(&triple[..t]).ok()?;
    let mut choice = empty_choice();
    put_group(&mut choice, 0, group0);
    put_group(&mut choice, 1, group1);
    Some(choice)
}
fn complement_less(a: u16, b: u16, ordered: &[(usize, usize)]) -> bool {
    ordered
        .iter()
        .filter(|v| a & (1 << v.1) == 0)
        .map(|v| v.0)
        .cmp(ordered.iter().filter(|v| b & (1 << v.1) == 0).map(|v| v.0))
        == std::cmp::Ordering::Less
}
fn gram_residual(
    gram: &GramWorkspace,
    ranked: &[Item; 16],
    count: usize,
    choice: &BundleChoice,
) -> Option<f64> {
    let mut positions = [0usize; 3];
    let mut weights = [0f64; 3];
    for i in 0..3 {
        positions[i] = ranked[..count]
            .iter()
            .position(|v| v.0 == choice.neurons[1][i])?;
        weights[i] = scalar(choice.weights[1][i])?;
    }
    let entry = |i: usize, j: usize| {
        let (a, b) = (
            positions[i].min(positions[j]),
            positions[i].max(positions[j]),
        );
        gram.gram[b * (b + 1) / 2 + a]
    };
    let mut diagonal = 0.;
    for i in 0..3 {
        diagonal = add(diagonal, mul(mul(weights[i], weights[i])?, entry(i, i))?)?;
    }
    if diagonal < 0. {
        return None;
    }
    let mut best = f64::INFINITY;
    for mask in 0..4 {
        let signs = [
            1.,
            if mask & 1 == 0 { 1. } else { -1. },
            if mask & 2 == 0 { 1. } else { -1. },
        ];
        let mut residual = diagonal;
        for j in 0..3 {
            for i in 0..j {
                let term = mul(
                    mul(mul(2. * signs[i] * signs[j], weights[i])?, weights[j])?,
                    entry(i, j),
                )?;
                residual = add(residual, term)?;
            }
        }
        // Negative cancellation residuals are rounded advice, NOT proofs that
        // the rows are dependent. Direct source scoring follows the shortlist.
        best = best.min(residual.max(0.));
    }
    if diagonal == 0. {
        return (best == 0.).then_some(0.);
    }
    safe(best / diagonal)
}

#[cfg(test)]
#[path = "geometry_tests.rs"]
mod tests;
