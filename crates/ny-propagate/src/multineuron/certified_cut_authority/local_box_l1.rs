// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Bounded arithmetic only for a local original-affine weighted-L1 relation.
//!
//! This module has NO carrier, verdict, scalar merge, generation or public API.
//! Its graph/map borrows bind object identity, not the provenance of a root
//! invocation. A later same-call owner MUST authenticate PreserveRaw, this map's
//! domain and the selected Linear's exact authored-f32/zero-radius provenance,
//! hold the quoted lease, and seal all indices/weights before using the result.
//! Supplying `CertifiedWeightError::EXACT` is a checked declaration, not proof
//! of that provenance. Nonzero selected-Linear uncertainty is unsupported.

use crate::layers::{normalization::BatchNormLayer, Layer, LinearLayer};
use crate::network::{GraphNetwork, NETWORK_INPUT};
use ny_core::dd::{next_down_f64, next_up_f64};
use ny_core::{f32_to_f64_exact, f64_to_f32_up, CertifiedWeightError};
use ny_tensor::BoundedTensor;
use std::collections::HashMap;
use std::time::Instant;

const MAX_INPUT: usize = 65536;
const MAX_OUTPUT: usize = 8192;
const MAX_ACTIVE: usize = 8;
const MAX_SIGNS: usize = 128;
const SCRATCH_BYTES: usize = 64 << 10;
const WORK_PER_COLUMN: u64 = 65536;
const FIXED_WORK: u64 = 1 << 20;
// Additional fixed work, not reused from source resolution/BN validation.
const UNARY_REFINE_WORK: u64 = 1 << 20;
// Fixed capture/copy and the 17 additional numeric completed-event fields.
const UNARY_OBSERVATION_WORK: u64 = 1 << 16;
const UNARY_REFINE_BYTES: usize = 4096;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Refusal {
    Deadline,
    Cancelled,
    Environment,
    Budget,
    Shape,
    Source,
    Values,
    Arithmetic,
}
type LocalResult<T> = Result<T, Refusal>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct LocalBoxL1Quote {
    pub work: u64,
    pub scratch_bytes: usize,
}

/// Allocation-free shape-only cost. Reserve this BEFORE graph lookups or scans.
/// The evaluator rechecks exact E/D and this allowance; it cannot reserve an
/// external WorkBudget itself. Setup/owner/full-fold costs are additional.
pub(super) fn quote(input_dim: usize, output_dim: usize) -> Option<LocalBoxL1Quote> {
    if !(1..=MAX_INPUT).contains(&input_dim) || !(1..=MAX_OUTPUT).contains(&output_dim) {
        return None;
    }
    Some(LocalBoxL1Quote {
        work: WORK_PER_COLUMN
            .checked_mul(u64::try_from(input_dim).ok()?.checked_add(1)?)?
            .checked_add(FIXED_WORK)?
            .checked_add(UNARY_REFINE_WORK)?
            .checked_add(UNARY_OBSERVATION_WORK)?,
        scratch_bytes: SCRATCH_BYTES,
    })
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Interval {
    lo: f64,
    hi: f64,
}
const ZERO: Interval = Interval { lo: 0., hi: 0. };
impl Interval {
    fn checked(lo: f64, hi: f64) -> LocalResult<Self> {
        if ny_core::is_crown_coeff_safe_f64(lo) && ny_core::is_crown_coeff_safe_f64(hi) && lo <= hi
        {
            Ok(Self { lo, hi })
        } else {
            Err(Refusal::Arithmetic)
        }
    }
    fn point(v: f32) -> LocalResult<Self> {
        if !ny_core::is_crown_coeff_safe(v) {
            return Err(Refusal::Values);
        }
        let v = f32_to_f64_exact(v);
        Ok(Self { lo: v, hi: v })
    }
    fn is_zero(self) -> bool {
        self.lo == 0. && self.hi == 0.
    }
    fn neg(self) -> Self {
        Self {
            lo: -self.hi,
            hi: -self.lo,
        }
    }
    fn add(self, rhs: Self) -> LocalResult<Self> {
        // Operand identities only: a rounded-to-zero result is NOT exact zero.
        if self.is_zero() {
            return Ok(rhs);
        }
        if rhs.is_zero() {
            return Ok(self);
        }
        Self::checked(
            next_down_f64(self.lo + rhs.lo),
            next_up_f64(self.hi + rhs.hi),
        )
    }
    fn mul(self, rhs: Self) -> LocalResult<Self> {
        if self.is_zero() || rhs.is_zero() {
            return Ok(ZERO);
        }
        let p = [
            self.lo * rhs.lo,
            self.lo * rhs.hi,
            self.hi * rhs.lo,
            self.hi * rhs.hi,
        ];
        if p.iter().any(|x| !x.is_finite()) {
            return Err(Refusal::Arithmetic);
        }
        let lo = p.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = p.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Self::checked(next_down_f64(lo), next_up_f64(hi))
    }
}

fn up_add(a: f64, b: f64) -> LocalResult<f64> {
    if a < 0. || b < 0. {
        return Err(Refusal::Arithmetic);
    }
    let v = if a == 0. {
        b
    } else if b == 0. {
        a
    } else {
        next_up_f64(a + b)
    };
    if ny_core::is_crown_coeff_safe_f64(v) {
        Ok(v)
    } else {
        Err(Refusal::Arithmetic)
    }
}
fn up_mul(a: f64, b: f64) -> LocalResult<f64> {
    if a < 0. || b < 0. {
        return Err(Refusal::Arithmetic);
    }
    let v = if a == 0. || b == 0. {
        0.
    } else {
        next_up_f64(a * b)
    };
    if ny_core::is_crown_coeff_safe_f64(v) {
        Ok(v)
    } else {
        Err(Refusal::Arithmetic)
    }
}

struct Guard<'p> {
    deadline: Instant,
    poll: &'p mut dyn FnMut() -> bool,
    failed: Option<Refusal>,
}
impl Guard<'_> {
    fn check(&mut self) -> LocalResult<()> {
        if let Some(e) = self.failed {
            return Err(e);
        }
        let e = if Instant::now() >= self.deadline {
            Some(Refusal::Deadline)
        } else if !(self.poll)() {
            Some(Refusal::Cancelled)
        } else {
            None
        };
        self.failed = e;
        e.map_or(Ok(()), Err)
    }
    fn values(&mut self, v: &[f32], radius: bool) -> LocalResult<()> {
        for chunk in v.chunks(1024) {
            self.check()?;
            if chunk.iter().any(|x| {
                !ny_core::is_crown_coeff_safe(*x)
                    || (radius && x.to_bits() & 0x8000_0000 != 0 && x.to_bits() != 0x8000_0000)
            }) {
                return Err(Refusal::Values);
            }
        }
        Ok(())
    }
}

struct Source<'a> {
    graph: &'a GraphNetwork,
    root_input: &'a BoundedTensor,
    map: &'a HashMap<String, BoundedTensor>,
    target: &'a str,
    linear: &'a LinearLayer,
    linear_parent: &'a str,
    input: &'a BoundedTensor,
    // True original preactivation: post-BN for Linear->BN->ReLU.
    pre_box: &'a BoundedTensor,
    bn: Option<&'a BatchNormLayer>,
}
fn name_ok(s: &str) -> LocalResult<()> {
    if s.is_empty() || s.len() > 1024 {
        Err(Refusal::Shape)
    } else {
        Ok(())
    }
}
fn vector_shape(b: &BoundedTensor, n: usize) -> bool {
    b.len() == n && (matches!(b.shape(), [d] if *d == n) || matches!(b.shape(), [1, d] if *d == n))
}

fn resolve<'a>(
    graph: &'a GraphNetwork,
    input: &'a BoundedTensor,
    map: &'a HashMap<String, BoundedTensor>,
    target: &'a str,
    guard: &mut Guard<'_>,
) -> LocalResult<Source<'a>> {
    guard.check()?;
    name_ok(target)?;
    let relu = graph.node(target).ok_or(Refusal::Source)?;
    if !matches!(relu.layer(), Layer::ReLU(_)) {
        return Err(Refusal::Source);
    }
    let pre_name = relu.require_unary_input().map_err(|_| Refusal::Shape)?;
    name_ok(pre_name)?;
    let pre = graph.node(pre_name).ok_or(Refusal::Source)?;
    let (linear, parent, bn, bn_input) = match pre.layer() {
        Layer::Linear(l) => (
            l,
            pre.require_unary_input().map_err(|_| Refusal::Shape)?,
            None,
            None,
        ),
        Layer::BatchNorm(b) => {
            let linear_name = pre.require_unary_input().map_err(|_| Refusal::Shape)?;
            name_ok(linear_name)?;
            let n = graph.node(linear_name).ok_or(Refusal::Source)?;
            let Layer::Linear(l) = n.layer() else {
                return Err(Refusal::Source);
            };
            (
                l,
                n.require_unary_input().map_err(|_| Refusal::Shape)?,
                Some(b),
                Some(map.get(linear_name).ok_or(Refusal::Source)?),
            )
        }
        _ => return Err(Refusal::Source),
    };
    name_ok(parent)?;
    let x = if parent == NETWORK_INPUT {
        input
    } else {
        map.get(parent).ok_or(Refusal::Source)?
    };
    let (e, d) = (linear.in_features(), linear.out_features());
    if quote(e, d).is_none()
        || !vector_shape(x, e)
        || linear.weight.shape() != [d, e]
        || linear.bias.as_ref().is_some_and(|b| b.shape() != [d])
    {
        return Err(Refusal::Shape);
    }
    let pre_box = map.get(pre_name).ok_or(Refusal::Source)?;
    if !vector_shape(pre_box, d) {
        return Err(Refusal::Shape);
    }
    if let Some(b) = bn {
        // Bound every hidden scan before calling the existing source-layout helper.
        if b.num_channels != d
            || [
                b.scale.shape(),
                b.bias.shape(),
                b.scale_err.shape(),
                b.bias_err.shape(),
            ]
            .iter()
            .any(|s| *s != [d])
        {
            return Err(Refusal::Shape);
        }
        let bx = bn_input.ok_or(Refusal::Source)?;
        if !vector_shape(bx, d) {
            return Err(Refusal::Shape);
        }
        guard.values(b.scale.as_slice().ok_or(Refusal::Shape)?, false)?;
        guard.values(b.bias.as_slice().ok_or(Refusal::Shape)?, false)?;
        guard.values(b.scale_err.as_slice().ok_or(Refusal::Shape)?, true)?;
        guard.values(b.bias_err.as_slice().ok_or(Refusal::Shape)?, true)?;
        if b.gpu_extraction_layout(bx.shape(), Some(d))
            .map_err(|_| Refusal::Source)?
            != (d, 1)
        {
            return Err(Refusal::Source);
        }
    }
    guard.check()?;
    Ok(Source {
        graph,
        root_input: input,
        map,
        target,
        linear,
        linear_parent: parent,
        input: x,
        pre_box,
        bn,
    })
}

/// Arithmetic enclosure for the borrowed source; NOT a root proof certificate.
/// Private construction and immutable source borrows prevent a caller-supplied
/// scalar from masquerading as this computation. Later root custody is separate.
pub(super) struct LocalBoxL1Arithmetic<'a> {
    source: Source<'a>,
    neurons: [usize; MAX_ACTIVE],
    weights: [f32; MAX_ACTIVE],
    active: usize,
    signs: [Interval; MAX_SIGNS],
    sign_count: usize,
    linear_abs: [f64; MAX_ACTIVE],
    bn_radius: f64,
    bound: f32,
    original_local_bound: f64,
    pre_lower: [f32; MAX_ACTIVE],
    pre_upper: [f32; MAX_ACTIVE],
    quote: LocalBoxL1Quote,
}
impl LocalBoxL1Arithmetic<'_> {
    pub(super) fn matches_source(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        map: &HashMap<String, BoundedTensor>,
        target: &str,
    ) -> bool {
        std::ptr::eq(self.source.graph, graph)
            && std::ptr::eq(self.source.root_input, input)
            && std::ptr::eq(self.source.map, map)
            && self.source.target == target
    }
    /// Refined weighted-L1 upper bound, including the same-source unary map.
    /// It need not equal the maximum of the ORIGINAL sign_bounds() endpoints.
    pub(super) fn bound(&self) -> f32 {
        self.bound
    }
    /// Original post-BN sign maximum, copied before refinement. Keep f64:
    /// converting diagnostic-only data to f32 must not create a new refusal.
    pub(super) fn original_local_bound(&self) -> f64 {
        self.original_local_bound
    }
    pub(super) fn pre_lower(&self) -> &[f32; MAX_ACTIVE] {
        &self.pre_lower
    }
    pub(super) fn pre_upper(&self) -> &[f32; MAX_ACTIVE] {
        &self.pre_upper
    }
    pub(super) fn sign_count(&self) -> usize {
        self.sign_count
    }
    pub(super) fn bn_radius(&self) -> f64 {
        self.bn_radius
    }
    pub(super) fn quote(&self) -> LocalBoxL1Quote {
        self.quote
    }
    pub(super) fn neurons(&self) -> &[usize] {
        &self.neurons[..self.active]
    }
    pub(super) fn weights(&self) -> &[f32] {
        &self.weights[..self.active]
    }
    /// Original affine-box supports widened for BN uncertainty, before unary
    /// refinement. These remain available independently of the refined bound.
    pub(super) fn sign_bounds(&self, row: usize) -> Option<(f64, f64)> {
        (row < self.sign_count).then(|| (self.signs[row].lo, self.signs[row].hi))
    }
}

/// Sign row bit k-1 means a negative coefficient for selected coordinate k;
/// coordinate0 is always positive. Fill backwards to keep the fixed table in place.
fn expand(
    terms: &[Interval; MAX_ACTIVE],
    m: usize,
    out: &mut [Interval; MAX_SIGNS],
) -> LocalResult<usize> {
    out[0] = terms[0];
    let mut count = 1;
    for term in terms.iter().take(m).skip(1) {
        for r in (0..count).rev() {
            let old = out[r];
            out[r + count] = old.add(term.neg())?;
            out[r] = old.add(*term)?;
        }
        count *= 2;
    }
    Ok(count)
}

// All arrays coexist with the caller's original source/column/sign buffers.
// The extra 512 bytes cover fixed locals/references in these bounded helpers;
// this whole addition is counted on top of the existing 16KiB safety allowance.
struct UnaryWorkspace {
    upper: [f64; 2 * MAX_SIGNS],
    marginal: [[f64; 2]; MAX_ACTIVE],
    cost: [[f64; 2]; MAX_ACTIVE],
    pre_lower: [f32; MAX_ACTIVE],
    pre_upper: [f32; MAX_ACTIVE],
}
#[derive(Clone, Copy, Debug, PartialEq)]
struct UnaryRefinement {
    bound: f64,
    pre_lower: [f32; MAX_ACTIVE],
    pre_upper: [f32; MAX_ACTIVE],
}
fn unary_scratch_bytes() -> usize {
    // Conservatively include both return-place and caller-held result copies.
    size_of::<UnaryWorkspace>() + 2 * size_of::<UnaryRefinement>() + 512
}

/// Both operands are ORIGINAL f32 values; their exact product fits IEEE f64.
/// Never use this shortcut on accumulated f64 or a BN effective coefficient.
fn unary_product(weight: f32, endpoint: f32) -> LocalResult<f64> {
    if !ny_core::is_crown_coeff_safe(weight) || !ny_core::is_crown_coeff_safe(endpoint) {
        return Err(Refusal::Values);
    }
    let product = f32_to_f64_exact(weight) * f32_to_f64_exact(endpoint);
    if ny_core::is_crown_coeff_safe_f64(product) {
        Ok(product)
    } else {
        Err(Refusal::Arithmetic)
    }
}

/// Signed and monotone: no zero shortcut, including for a rounded-zero result.
/// Monotonicity lets each minimum retain all ordered subset candidates.
fn signed_up_add(a: f64, b: f64) -> LocalResult<f64> {
    if !ny_core::is_crown_coeff_safe_f64(a) || !ny_core::is_crown_coeff_safe_f64(b) {
        return Err(Refusal::Arithmetic);
    }
    let value = next_up_f64(a + b);
    if ny_core::is_crown_coeff_safe_f64(value) {
        Ok(value)
    } else {
        Err(Refusal::Arithmetic)
    }
}

/// Refine already BN-widened supports with the exact target's certified unary
/// box. For each coordinate, save BOTH old partners before either update:
/// H_i(s)=min(H_(i-1)(s), up(H_(i-1)(flip_i(s))+2*w_i*bound_i(s))).
/// Induction covers each subset of flipped coordinates once. Costs may be
/// negative for stable endpoints; the unary multipliers 2*w_i remain positive.
fn refine_unary_bound(
    signs: &[Interval; MAX_SIGNS],
    count: usize,
    neurons: &[usize],
    weights: &[f32],
    pre_box: &BoundedTensor,
    guard: &mut Guard<'_>,
) -> LocalResult<UnaryRefinement> {
    guard.check()?;
    let m = neurons.len();
    if !(1..=MAX_ACTIVE).contains(&m) || weights.len() != m || count != 1usize << (m - 1) {
        return Err(Refusal::Shape);
    }
    let lo = pre_box.lower().as_slice().ok_or(Refusal::Shape)?;
    let hi = pre_box.upper().as_slice().ok_or(Refusal::Shape)?;
    let mut table = UnaryWorkspace {
        upper: [0.; 2 * MAX_SIGNS],
        marginal: [[0.; 2]; MAX_ACTIVE],
        cost: [[0.; 2]; MAX_ACTIVE],
        pre_lower: [0.; MAX_ACTIVE],
        pre_upper: [0.; MAX_ACTIVE],
    };
    for i in 0..m {
        let n = neurons[i];
        let (&l, &u) = lo.get(n).zip(hi.get(n)).ok_or(Refusal::Shape)?;
        if !ny_core::is_crown_coeff_safe(l) || !ny_core::is_crown_coeff_safe(u) {
            return Err(Refusal::Values);
        }
        if l > u {
            return Err(Refusal::Values);
        }
        // Capture exactly the bits just validated; inactive slots stay +0.
        table.pre_lower[i] = l;
        table.pre_upper[i] = u;
        // Validated Q8 w is positive and <=1, so 2*w is exact original-f32.
        table.marginal[i] = [
            unary_product(weights[i], u)?,
            unary_product(weights[i], -l)?,
        ];
        table.cost[i] = [
            unary_product(2. * weights[i], u)?,
            unary_product(2. * weights[i], -l)?,
        ];
    }
    let total = 2 * count;
    let mask = total - 1;
    for (r, original) in signs.iter().take(count).enumerate() {
        Interval::checked(original.lo, original.hi)?;
        // Full-index bit0 negates coordinate0, unlike the representative rows.
        table.upper[2 * r] = original.hi;
        table.upper[(2 * r) ^ mask] = -original.lo;
    }
    for s in 0..total {
        guard.check()?;
        let mut unary = 0.;
        for i in 0..m {
            unary = signed_up_add(unary, table.marginal[i][(s >> i) & 1])?;
        }
        table.upper[s] = table.upper[s].min(unary);
    }
    for i in 0..m {
        guard.check()?;
        let bit = 1usize << i;
        let mut pairs = 0;
        for positive in 0..total {
            if positive & bit != 0 {
                continue;
            }
            if pairs % 32 == 0 {
                guard.check()?;
            }
            pairs += 1;
            let negative = positive | bit;
            let old_positive = table.upper[positive];
            let old_negative = table.upper[negative];
            let new_positive = signed_up_add(old_negative, table.cost[i][0])?;
            let new_negative = signed_up_add(old_positive, table.cost[i][1])?;
            table.upper[positive] = old_positive.min(new_positive);
            table.upper[negative] = old_negative.min(new_negative);
        }
    }
    let mut bound = 0f64;
    for s in 0..total {
        if -table.upper[s ^ mask] > table.upper[s] {
            // Never turn an apparent contradiction into an empty-domain proof.
            return Err(Refusal::Arithmetic);
        }
        bound = bound.max(table.upper[s]);
    }
    guard.check()?;
    Ok(UnaryRefinement {
        bound,
        pre_lower: table.pre_lower,
        pre_upper: table.pre_upper,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn evaluate<'a>(
    graph: &'a GraphNetwork,
    input: &'a BoundedTensor,
    map: &'a HashMap<String, BoundedTensor>,
    target: &'a str,
    neurons: &[usize],
    weights: &[f32],
    linear_error: CertifiedWeightError,
    paid: LocalBoxL1Quote,
    deadline: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> LocalResult<LocalBoxL1Arithmetic<'a>> {
    let mut guard = Guard {
        deadline,
        poll,
        failed: None,
    };
    guard.check()?;
    ny_core::require_f64_interval_proof_environment().map_err(|_| Refusal::Environment)?;
    if !ny_core::dd_selfcheck::dd_selfcheck_ok() {
        return Err(Refusal::Environment);
    }
    // Bit checks cannot launder nonzero subnormal radii under DAZ.
    if linear_error.weight_rel_err.to_bits() & 0x7fff_ffff != 0
        || linear_error.bias_abs_err.to_bits() & 0x7fff_ffff != 0
    {
        return Err(Refusal::Source);
    }
    if !(1..=MAX_ACTIVE).contains(&neurons.len())
        || weights.len() != neurons.len()
        || neurons.windows(2).any(|p| p[0] >= p[1])
    {
        return Err(Refusal::Shape);
    }
    for w in weights {
        let bits = w.to_bits();
        if bits < (1f32 / 256.).to_bits()
            || bits > 1f32.to_bits()
            || (f32_to_f64_exact(*w) * 256.).fract() != 0.
        {
            return Err(Refusal::Values);
        }
    }
    // Before graph access, enough fixed credit and storage for bounded resolution.
    if paid.work < FIXED_WORK
        || paid.scratch_bytes < SCRATCH_BYTES
        || unary_scratch_bytes() > UNARY_REFINE_BYTES
        || size_of::<LocalBoxL1Arithmetic<'_>>()
            + size_of::<[Interval; MAX_SIGNS]>()
            + size_of::<[Interval; MAX_ACTIVE]>() * 3
            + size_of::<Source<'_>>()
            + size_of::<Guard<'_>>()
            + unary_scratch_bytes()
            + 16384
            > SCRATCH_BYTES
    {
        return Err(Refusal::Budget);
    }
    let source = resolve(graph, input, map, target, &mut guard)?;
    let (e, d) = (source.linear.in_features(), source.linear.out_features());
    let required = quote(e, d).ok_or(Refusal::Shape)?;
    if paid.work < required.work || neurons[neurons.len() - 1] >= d {
        return Err(Refusal::Budget);
    }
    let w = source.linear.weight.as_slice().ok_or(Refusal::Shape)?;
    let bias = source
        .linear
        .bias
        .as_ref()
        .map(|x| x.as_slice().ok_or(Refusal::Shape))
        .transpose()?;
    let lo = source.input.lower().as_slice().ok_or(Refusal::Shape)?;
    let hi = source.input.upper().as_slice().ok_or(Refusal::Shape)?;
    let m = neurons.len();
    let mut selected = [0; MAX_ACTIVE];
    let mut weighted = [0.; MAX_ACTIVE];
    selected[..m].copy_from_slice(neurons);
    weighted[..m].copy_from_slice(weights);
    let mut terms = [ZERO; MAX_ACTIVE];
    let mut linear_bounds = [ZERO; MAX_ACTIVE];
    let mut scales = [1.; MAX_ACTIVE];
    let mut scale_errors = [0.; MAX_ACTIVE];
    let mut bias_errors = [0.; MAX_ACTIVE];
    for i in 0..m {
        let b = bias.map_or(0., |b| b[neurons[i]]);
        linear_bounds[i] = Interval::point(b)?;
        let mut center = linear_bounds[i];
        if let Some(bn) = source.bn {
            let n = neurons[i];
            scales[i] = bn.scale.as_slice().ok_or(Refusal::Shape)?[n];
            scale_errors[i] = bn.scale_err.as_slice().ok_or(Refusal::Shape)?[n];
            bias_errors[i] = bn.bias_err.as_slice().ok_or(Refusal::Shape)?[n];
            center = Interval::point(scales[i])?
                .mul(center)?
                .add(Interval::point(
                    bn.bias.as_slice().ok_or(Refusal::Shape)?[n],
                )?)?;
        }
        terms[i] = Interval::point(weights[i])?.mul(center)?;
    }
    let mut signs = [ZERO; MAX_SIGNS];
    let count = expand(&terms, m, &mut signs)?;
    let mut column = [ZERO; MAX_SIGNS];
    for j in 0..e {
        guard.check()?;
        let x = Interval::checked(Interval::point(lo[j])?.lo, Interval::point(hi[j])?.hi)?;
        for i in 0..m {
            let value = w[neurons[i] * e + j];
            let wc = Interval::point(value)?;
            linear_bounds[i] = linear_bounds[i].add(wc.mul(x)?)?;
            // The third f32 factor is NOT assumed exact; every multiply is enclosed.
            terms[i] = Interval::point(weights[i])?
                .mul(Interval::point(scales[i])?)?
                .mul(wc)?;
        }
        expand(&terms, m, &mut column)?;
        for r in 0..count {
            signs[r] = signs[r].add(column[r].mul(x)?)?;
        }
    }
    guard.check()?;
    let mut abs = [0.; MAX_ACTIVE];
    let mut radius = 0.;
    for i in 0..m {
        abs[i] = linear_bounds[i].lo.abs().max(linear_bounds[i].hi.abs());
        let err = up_add(
            up_mul(f32_to_f64_exact(scale_errors[i]), abs[i])?,
            f32_to_f64_exact(bias_errors[i]),
        )?;
        radius = up_add(radius, up_mul(f32_to_f64_exact(weights[i]), err)?)?;
    }
    let mut bound = 0f64;
    for item in signs.iter_mut().take(count) {
        if radius != 0. {
            *item = Interval::checked(
                next_down_f64(item.lo - radius),
                next_up_f64(item.hi + radius),
            )?;
        }
        bound = bound.max(item.hi).max(-item.lo);
    }
    // The source's selected unary box is about true post-BN preactivations.
    // Refinement must follow the original supports' common-radius widening.
    let original_local_bound = bound;
    let refined = refine_unary_bound(&signs, count, neurons, weights, source.pre_box, &mut guard)?;
    bound = bound.min(refined.bound);
    let bound = f64_to_f32_up(bound);
    if !ny_core::is_crown_coeff_safe(bound) || bound < 0. {
        return Err(Refusal::Arithmetic);
    }
    guard.check()?;
    Ok(LocalBoxL1Arithmetic {
        source,
        neurons: selected,
        weights: weighted,
        active: m,
        signs,
        sign_count: count,
        linear_abs: abs,
        bn_radius: radius,
        bound,
        original_local_bound,
        pre_lower: refined.pre_lower,
        pre_upper: refined.pre_upper,
        quote: required,
    })
}

#[cfg(test)]
#[path = "local_box_l1_tests.rs"]
mod tests;
