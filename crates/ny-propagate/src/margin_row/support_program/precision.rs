// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Paid binary32 reference computation over authenticated original operators.
//!
//! The nominal endpoint dots actually multiply and accumulate binary32. The
//! published endpoints are binary64; this is not binary32 reference storage or
//! a tableau producer. Parameters and input endpoints retain their complete
//! original-minus-executed cast gaps, including TwoSum subtraction residuals.
//! Original fold uncertainty is independent of this numerical error.
//!
//! The reduction uses the standard gamma_n model, with a separate absolute
//! loss for every possible flushed product/addition. See Higham, The Accuracy
//! of Floating Point Summation (1993), https://doi.org/10.1137/0914050, and
//! Goldberg (1991), https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html.
//! Explicitly normalizing operands/results makes the additive flush bound
//! independent of binary32 gradual underflow. Certificate arithmetic requires
//! qualified binary64 round-to-nearest with gradual underflow throughout.
//!
//! The sealed owner is the only publication seam. The prefix adapter consumes
//! it after checking its original root/operator/tensor binding; arbitrary
//! supplied endpoints cannot acquire its numerical provenance.

use super::{CertifiedTensorBox, OriginalRoot};
use crate::certify::work::{
    LeaseClass, PersistentReservation, WorkBudget, WorkRefusal, WorkRequest,
};
use crate::margin_row::net::{ConvOp, TwinOp};
use ny_core::dd::{next_down_f64, next_up_f64, two_sum};
use ny_core::{f32_to_f64_exact, NyError, Result};
use std::mem::size_of;
use std::time::Instant;

const POLL: usize = 256;
const OWNER_SLACK: usize = 512;
const UNIT32: f64 = 1.0 / 16_777_216.0;
const MIN_NORMAL32: f64 = f64::from_bits((1023_u64 - 126) << 52);

/// Completed observations, never a performance or whole-process receipt.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::margin_row) struct ReferenceF32Receipt {
    /// Genuine completed original convs with at least one nonzero dot tap.
    pub conv_ops: usize,
    pub affine_ops: usize,
    /// Count of actual binary32 multiply/add pairs across both endpoint dots.
    pub multiply_add_pairs: usize,
    pub converted_cells: usize,
    pub subnormal_casts_normalized: usize,
    /// Constant/bias-only outputs remain useful without inventing dot work.
    pub constant_outputs: usize,
}

pub(in crate::margin_row) struct ComputedReferenceF32 {
    root: OriginalRoot,
    op: usize,
    input: usize,
    tensor: usize,
    lo: Vec<f64>,
    hi: Vec<f64>,
    deadline: Option<Instant>,
    receipt: ReferenceF32Receipt,
    resident: PersistentReservation,
}

impl ComputedReferenceF32 {
    pub(in crate::margin_row) fn root(&self) -> &OriginalRoot {
        &self.root
    }
    pub(in crate::margin_row) fn op(&self) -> usize {
        self.op
    }
    pub(in crate::margin_row) fn tensor(&self) -> usize {
        self.tensor
    }
    pub(in crate::margin_row) fn lo(&self) -> &[f64] {
        &self.lo
    }
    pub(in crate::margin_row) fn hi(&self) -> &[f64] {
        &self.hi
    }
    pub(in crate::margin_row) fn receipt(&self) -> ReferenceF32Receipt {
        self.receipt
    }
    pub(in crate::margin_row) fn resident_bytes(&self) -> usize {
        self.resident.bytes()
    }
    pub(in crate::margin_row) fn deadline(&self) -> Result<Option<Instant>> {
        self.root.deadline(self.deadline)
    }

    /// Consuming authenticated transfer for the private prefix producer. The
    /// reservation already pays the transferred vectors and certificate owner.
    /// No second allocation/admission or public numeric-to-proof constructor.
    #[allow(clippy::type_complexity)]
    pub(in crate::margin_row) fn into_parts(
        self,
        root: &OriginalRoot,
        op: usize,
    ) -> Result<(
        Vec<f64>,
        Vec<f64>,
        Option<Instant>,
        PersistentReservation,
        ReferenceF32Receipt,
    )> {
        let deadline = self.deadline()?;
        ny_core::require_f64_interval_proof_environment()?;
        if !self.root().same_root(root)
            || self.op() != op
            || self.tensor() != op.checked_add(1).ok_or_else(overflow)?
            || !self.resident.shares_ledger(root.budget())
            || root.model().net().tsize.get(self.tensor()) != Some(&self.lo().len())
            || self.lo().len() != self.hi().len()
            || original_input(root, op)? != self.input
        {
            return Err(bad("stale original binary32 reference transfer"));
        }
        let actual = endpoint_bytes(self.lo.capacity(), self.hi.capacity())?;
        if actual > self.resident_bytes() {
            return Err(memory(
                actual,
                self.resident_bytes(),
                "binary32 reference transferred capacity",
            ));
        }
        checkpoint(root.budget(), deadline)?;
        let receipt = self.receipt();
        Ok((self.lo, self.hi, deadline, self.resident, receipt))
    }
}

#[derive(Clone, Copy)]
struct Cast {
    // Preserve the ACTUAL conversion bits, including a subnormal result.
    converted: f32,
    // Only normal/zero operands enter nominal binary32 arithmetic.
    operand: f32,
    gap: f64,
}

#[derive(Clone, Copy)]
struct Plan {
    output: usize,
    weights: usize,
    biases: usize,
    terms: usize,
    request: WorkRequest,
}

impl Plan {
    fn new(root: &OriginalRoot, op: usize, first: &CertifiedTensorBox) -> Result<Self> {
        let net = root.model().net();
        if op >= net.i_gemm1 || !root.same_root(first.root()) {
            return Err(bad(
                "binary32 reference requires the original trunk/input owner",
            ));
        }
        let output = *net
            .tsize
            .get(op + 1)
            .ok_or_else(|| bad("missing original output"))?;
        let input = original_input(root, op)?;
        if first.tensor() != input
            || net.tsize.get(input) != Some(&first.lo().len())
            || first.lo().len() != first.hi().len()
            || output == 0
        {
            return Err(bad(
                "binary32 reference input uses another original edge/shape",
            ));
        }
        let (weights, biases, terms, max_terms) = match &net.ops[op] {
            TwinOp::Conv(c) => {
                let k = c
                    .kernel
                    .1
                    .checked_mul(c.kernel.2)
                    .and_then(|n| n.checked_mul(c.kernel.3))
                    .ok_or_else(overflow)?;
                let p = c.oshape.1.checked_mul(c.oshape.2).ok_or_else(overflow)?;
                if k != c.k_fwd
                    || p == 0
                    || output != c.kernel.0.checked_mul(p).ok_or_else(overflow)?
                    || c.wmat.len() != c.kernel.0.checked_mul(k).ok_or_else(overflow)?
                    || c.bias.len() != c.kernel.0
                    || c.bias_err.len() != c.kernel.0
                    || c.gather.len() != k.checked_mul(p).ok_or_else(overflow)?
                {
                    return Err(bad("original binary32 convolution geometry disagrees"));
                }
                (
                    c.wmat.len(),
                    c.bias.len(),
                    output.checked_mul(k).ok_or_else(overflow)?,
                    k,
                )
            }
            TwinOp::ChannelAffine {
                scale,
                shift,
                shift_err,
                ..
            } => {
                if first.lo().len() != output
                    || scale.len() != output
                    || shift.len() != output
                    || shift_err.len() != output
                {
                    return Err(bad("original binary32 affine geometry disagrees"));
                }
                (output, output, output, 1)
            }
            _ => return Err(bad("operator has no binary32 reference kernel")),
        };
        gamma(max_terms)?;
        let cells = first
            .lo()
            .len()
            .checked_mul(2)
            .and_then(|n| n.checked_add(weights))
            .and_then(|n| n.checked_add(biases))
            .ok_or_else(overflow)?;
        let scratch = cells
            .checked_mul(size_of::<Cast>())
            .and_then(|n| n.checked_add(OWNER_SLACK + 4 * size_of::<Vec<Cast>>()))
            .ok_or_else(overflow)?;
        let persistent = endpoint_bytes(output, output)?;
        let work = terms
            .checked_mul(256)
            .and_then(|n| cells.checked_mul(128).and_then(|c| n.checked_add(c)))
            .and_then(|n| output.checked_mul(128).and_then(|c| n.checked_add(c)))
            .and_then(|n| n.checked_add(4096))
            .and_then(|n| u64::try_from(n).ok())
            .ok_or_else(overflow)?;
        Ok(Self {
            output,
            weights,
            biases,
            terms,
            request: WorkRequest {
                scratch_bytes: scratch,
                persistent_bytes: persistent,
                work,
                ..WorkRequest::default()
            },
        })
    }
}

/// Compute one original Conv/ChannelAffine reference with actual binary32 dots.
/// F64 certificate owners may call this component without changing their frozen
/// precision policy. The consuming prefix adapter retains its original binding.
pub(in crate::margin_row) fn derive_reference_f32(
    root: &OriginalRoot,
    op: usize,
    first: &CertifiedTensorBox,
    requested: Option<Instant>,
) -> Result<ComputedReferenceF32> {
    derive_inner(root, op, first, requested, None)
}

fn derive_inner(
    root: &OriginalRoot,
    op: usize,
    first: &CertifiedTensorBox,
    requested: Option<Instant>,
    mut observer: Option<&mut dyn FnMut(usize)>,
) -> Result<ComputedReferenceF32> {
    let deadline = root.deadline(earliest(first.deadline()?, requested))?;
    qualify()?;
    let plan = Plan::new(root, op, first)?;
    let budget = root.budget();
    let source = root
        .model()
        .resident_bytes()
        .checked_add(root.resident_bytes())
        .and_then(|n| n.checked_add(first.resident_bytes()))
        .ok_or_else(overflow)?;
    let live = budget.receipt().map_err(refusal)?;
    let current = live
        .live_scratch_bytes
        .checked_add(live.live_persistent_bytes)
        .ok_or_else(overflow)?;
    let peak = source
        .max(current)
        .checked_add(plan.request.scratch_bytes)
        .and_then(|n| n.checked_add(plan.request.persistent_bytes))
        .ok_or_else(overflow)?;
    let cap = crate::margin_row::root_box::ROOT_SOURCE_BYTES_CAP;
    if peak > cap {
        return Err(memory(peak, cap, "original binary32 reference total live"));
    }
    checkpoint(budget, deadline)?;
    let mut lease = budget
        .lease_for(LeaseClass::Fallback, plan.request)
        .map_err(refusal)?;
    let deadline = earliest(deadline, lease.deadline());
    checkpoint(budget, deadline)?;
    let mut receipt = ReferenceF32Receipt::default();
    let lows = casts(first.lo(), budget, deadline, &mut receipt)?;
    let highs = casts(first.hi(), budget, deadline, &mut receipt)?;
    let (weights, biases) = match &root.model().net().ops[op] {
        TwinOp::Conv(c) => (
            casts(&c.wmat, budget, deadline, &mut receipt)?,
            casts(&c.bias, budget, deadline, &mut receipt)?,
        ),
        TwinOp::ChannelAffine { scale, shift, .. } => (
            casts(scale, budget, deadline, &mut receipt)?,
            casts(shift, budget, deadline, &mut receipt)?,
        ),
        _ => return Err(bad("missing original binary32 kernel")),
    };
    if weights.len() != plan.weights || biases.len() != plan.biases {
        return Err(bad("binary32 parameter capacity changed"));
    }
    let mut lo = exact_vec::<f64>(plan.output, budget, deadline)?;
    let mut hi = exact_vec::<f64>(plan.output, budget, deadline)?;
    let actual_scratch = [
        lows.capacity(),
        highs.capacity(),
        weights.capacity(),
        biases.capacity(),
    ]
    .into_iter()
    .try_fold(0usize, |s, n| {
        s.checked_add(n.checked_mul(size_of::<Cast>()).ok_or_else(overflow)?)
            .ok_or_else(overflow)
    })?
    .checked_add(OWNER_SLACK + 4 * size_of::<Vec<Cast>>())
    .ok_or_else(overflow)?;
    let actual_output = endpoint_bytes(lo.capacity(), hi.capacity())?;
    if actual_scratch > plan.request.scratch_bytes || actual_output > plan.request.persistent_bytes
    {
        return Err(bad(
            "binary32 allocator capacities exceed admitted complete overlap",
        ));
    }
    lease.mark_materialized().map_err(refusal)?;
    let mut visited = 0usize;
    for output in 0..plan.output {
        poll(budget, deadline, output)?;
        let (lower, upper, terms) = match &root.model().net().ops[op] {
            TwinOp::Conv(c) => conv_row(
                c,
                output,
                first,
                &lows,
                &highs,
                &weights,
                &biases,
                budget,
                deadline,
                &mut visited,
            )?,
            TwinOp::ChannelAffine {
                scale,
                scale_rel_err,
                shift_err,
                ..
            } => {
                let w = scale[output];
                let lower_x = if w >= 0.0 {
                    lows[output]
                } else {
                    highs[output]
                };
                let upper_x = if w >= 0.0 {
                    highs[output]
                } else {
                    lows[output]
                };
                let mag = first.lo()[output].abs().max(first.hi()[output].abs());
                let mut lower = Dot::new(biases[output], shift_err[output])?;
                let mut upper = Dot::new(biases[output], shift_err[output])?;
                if w != 0.0 {
                    lower.step(w, weights[output], lower_x, *scale_rel_err, mag)?;
                    upper.step(w, weights[output], upper_x, *scale_rel_err, mag)?;
                }
                (
                    lower.finish(true)?,
                    upper.finish(false)?,
                    usize::from(w != 0.0),
                )
            }
            _ => return Err(bad("original binary32 kernel changed")),
        };
        if !(lower.is_finite() && upper.is_finite() && lower <= upper) {
            return Err(bad("nonfinite/crossed binary32 reference result"));
        }
        lo.push(lower);
        hi.push(upper);
        receipt.multiply_add_pairs = receipt
            .multiply_add_pairs
            .checked_add(terms.checked_mul(2).ok_or_else(overflow)?)
            .ok_or_else(overflow)?;
        receipt.constant_outputs = receipt
            .constant_outputs
            .checked_add(usize::from(terms == 0))
            .ok_or_else(overflow)?;
        if let Some(f) = observer.as_mut() {
            f(output);
        }
        checkpoint(budget, deadline)?;
    }
    if receipt.multiply_add_pairs > plan.terms.checked_mul(2).ok_or_else(overflow)?
        || receipt.converted_cells == 0
    {
        return Err(bad(
            "binary32 compute receipt has no actual conversion or exceeds original work",
        ));
    }
    if receipt.multiply_add_pairs != 0 {
        match &root.model().net().ops[op] {
            TwinOp::Conv(_) => receipt.conv_ops = 1,
            TwinOp::ChannelAffine { .. } => receipt.affine_ops = 1,
            _ => unreachable!(),
        }
    }
    drop((lows, highs, weights, biases));
    checkpoint(budget, deadline)?;
    qualify()?;
    let resident = lease.commit_persistent().map_err(refusal)?;
    Ok(ComputedReferenceF32 {
        root: root.clone(),
        op,
        input: first.tensor(),
        tensor: op + 1,
        lo,
        hi,
        deadline,
        receipt,
        resident,
    })
}

#[allow(clippy::too_many_arguments)]
fn conv_row(
    c: &ConvOp,
    output: usize,
    first: &CertifiedTensorBox,
    lows: &[Cast],
    highs: &[Cast],
    weights: &[Cast],
    biases: &[Cast],
    budget: &WorkBudget,
    deadline: Option<Instant>,
    visited: &mut usize,
) -> Result<(f64, f64, usize)> {
    let p = c.oshape.1.checked_mul(c.oshape.2).ok_or_else(overflow)?;
    let ch = output / p;
    let sp = output % p;
    let mut lower = Dot::new(biases[ch], c.bias_err[ch])?;
    let mut upper = Dot::new(biases[ch], c.bias_err[ch])?;
    let mut terms = 0usize;
    for tap in 0..c.k_fwd {
        poll(budget, deadline, *visited)?;
        *visited = visited.checked_add(1).ok_or_else(overflow)?;
        let source = c.gather[tap * p + sp];
        if source == usize::MAX {
            continue;
        }
        let weight = ch * c.k_fwd + tap;
        let w = c.wmat[weight];
        if w == 0.0 {
            continue;
        }
        let l = *lows
            .get(source)
            .ok_or_else(|| bad("original tap outside certified input"))?;
        let u = *highs
            .get(source)
            .ok_or_else(|| bad("original tap outside certified input"))?;
        let mag = first.lo()[source].abs().max(first.hi()[source].abs());
        lower.step(
            w,
            weights[weight],
            if w >= 0.0 { l } else { u },
            c.weight_rel_err,
            mag,
        )?;
        upper.step(
            w,
            weights[weight],
            if w >= 0.0 { u } else { l },
            c.weight_rel_err,
            mag,
        )?;
        terms = terms.checked_add(1).ok_or_else(overflow)?;
    }
    Ok((lower.finish(true)?, upper.finish(false)?, terms))
}

struct Dot {
    nominal: f32,
    cast_fold: f64,
    magnitude: f64,
    terms: usize,
}
impl Dot {
    fn new(bias: Cast, error: f64) -> Result<Self> {
        if !error.is_finite() || error < 0.0 {
            return Err(bad("invalid original bias uncertainty"));
        }
        Ok(Self {
            nominal: bias.operand,
            cast_fold: add_up(bias.gap, error),
            magnitude: f32_to_f64_exact(bias.operand).abs(),
            terms: 0,
        })
    }
    fn step(
        &mut self,
        original_w: f64,
        w: Cast,
        x: Cast,
        rho: f64,
        magnitude_x: f64,
    ) -> Result<()> {
        if !(rho.is_finite() && rho >= 0.0 && magnitude_x.is_finite() && magnitude_x >= 0.0) {
            return Err(bad("invalid original fold/input uncertainty"));
        }
        let ww = f32_to_f64_exact(w.operand);
        let xx = f32_to_f64_exact(x.operand);
        // Exact identity: w*x-ws*xs = w*(x-xs)+(w-ws)*xs.
        let cast = add_up(mul_up(original_w.abs(), x.gap), mul_up(xx.abs(), w.gap));
        let fold = mul_up(mul_up(rho, original_w.abs()), magnitude_x);
        self.cast_fold = add_up(self.cast_fold, add_up(cast, fold));
        // A product of two finite f32 values is EXACT in qualified f64:
        // <=48 significant bits and exponent in [-298,256].
        self.magnitude = add_up(self.magnitude, (ww * xx).abs());
        let product = normalize(std::hint::black_box(w.operand) * std::hint::black_box(x.operand))?;
        self.nominal =
            normalize(std::hint::black_box(self.nominal) + std::hint::black_box(product))?;
        self.terms = self.terms.checked_add(1).ok_or_else(overflow)?;
        if !(self.cast_fold.is_finite() && self.magnitude.is_finite()) {
            return Err(bad("unrepresentable binary32 certificate error"));
        }
        Ok(())
    }
    fn finish(self, lower: bool) -> Result<f64> {
        let (g, scale, operations) = gamma(self.terms)?;
        // Per-operation absolute flush losses may be amplified by subsequent
        // roundings; 1/(1-nu) <= 1+gamma conservatively covers that chain.
        let flush = if self.terms == 0 {
            0.0
        } else {
            mul_up(mul_up(operations as f64, MIN_NORMAL32), scale)
        };
        let error = add_up(self.cast_fold, add_up(mul_up(g, self.magnitude), flush));
        let center = f32_to_f64_exact(self.nominal);
        let result = if error == 0.0 {
            center
        } else if lower {
            next_down_f64(center - error)
        } else {
            next_up_f64(center + error)
        };
        if !error.is_finite() || !result.is_finite() {
            return Err(bad("unrepresentable outward binary32 reference endpoint"));
        }
        Ok(result)
    }
}

fn gamma(terms: usize) -> Result<(f64, f64, usize)> {
    let operations = terms
        .checked_mul(2)
        .and_then(|n| n.checked_add(2))
        .ok_or_else(overflow)?;
    // nu is exact, as is 1-nu; retain a finite strict gamma denominator.
    if operations >= 1 << 24 {
        return Err(bad("binary32 reduction exceeds gamma firewall"));
    }
    if terms == 0 {
        return Ok((0.0, 1.0, operations));
    }
    let nu = operations as f64 * UNIT32;
    let g = next_up_f64(nu / (1.0 - nu));
    Ok((g, next_up_f64(1.0 + g), operations))
}
fn cast(value: f64) -> Result<Cast> {
    if !value.is_finite() {
        return Err(bad("nonfinite original binary32 cast"));
    }
    let converted = std::hint::black_box(value) as f32;
    let operand = normalize(converted)?;
    let stored = f32_to_f64_exact(operand);
    let gap = full_gap(value, stored);
    if !gap.is_finite() {
        return Err(bad("binary32 cast gap overflow"));
    }
    Ok(Cast {
        converted,
        operand,
        gap,
    })
}
fn full_gap(original: f64, stored: f64) -> f64 {
    let (difference, residual) = two_sum(original, -stored);
    add_up(difference.abs(), residual.abs())
}
fn normalize(value: f32) -> Result<f32> {
    let bits = value.to_bits();
    let magnitude = bits & 0x7fff_ffff;
    if magnitude >= 0x7f80_0000 {
        return Err(bad("binary32 nominal overflow/nonfinite"));
    }
    Ok(if magnitude < 0x0080_0000 {
        f32::from_bits(bits & 0x8000_0000)
    } else {
        value
    })
}
fn casts(
    values: &[f64],
    budget: &WorkBudget,
    deadline: Option<Instant>,
    receipt: &mut ReferenceF32Receipt,
) -> Result<Vec<Cast>> {
    let mut out = exact_vec(values.len(), budget, deadline)?;
    for (i, &v) in values.iter().enumerate() {
        poll(budget, deadline, i)?;
        let value = cast(v)?;
        receipt.converted_cells = receipt
            .converted_cells
            .checked_add(1)
            .ok_or_else(overflow)?;
        if value.converted.to_bits() & 0x7fff_ffff != 0
            && value.converted.to_bits() & 0x7fff_ffff < 0x0080_0000
        {
            receipt.subnormal_casts_normalized = receipt
                .subnormal_casts_normalized
                .checked_add(1)
                .ok_or_else(overflow)?;
        }
        out.push(value);
    }
    checkpoint(budget, deadline)?;
    Ok(out)
}
fn original_input(root: &OriginalRoot, op: usize) -> Result<usize> {
    match root.model().net().ops.get(op) {
        Some(TwinOp::Conv(c)) => Ok(c.input),
        Some(TwinOp::ChannelAffine { input, .. }) => Ok(*input),
        _ => Err(bad("not an original binary32 reference operator")),
    }
}
fn endpoint_bytes(lo: usize, hi: usize) -> Result<usize> {
    lo.checked_add(hi)
        .and_then(|n| n.checked_mul(size_of::<f64>()))
        .and_then(|n| {
            n.checked_add(size_of::<ComputedReferenceF32>().max(size_of::<CertifiedTensorBox>()))
        })
        .and_then(|n| n.checked_add(OWNER_SLACK))
        .ok_or_else(overflow)
}
fn exact_vec<T>(n: usize, budget: &WorkBudget, deadline: Option<Instant>) -> Result<Vec<T>> {
    checkpoint(budget, deadline)?;
    n.checked_mul(size_of::<T>()).ok_or_else(overflow)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(n)
        .map_err(|_| bad("binary32 reference allocation refused"))?;
    checkpoint(budget, deadline)?;
    if values.capacity() != n {
        return Err(bad("binary32 allocator exceeded its prepaid capacity"));
    }
    Ok(values)
}
pub(super) fn qualify() -> Result<()> {
    ny_core::require_f64_interval_proof_environment()?;
    let one = std::hint::black_box(1.0_f32);
    let above = std::hint::black_box(f32::from_bits((127_u32 - 24) << 23));
    let below = std::hint::black_box(f32::from_bits((127_u32 - 25) << 23));
    let multiplicand = std::hint::black_box(f32::from_bits(0x3f80_0001));
    if std::hint::black_box(one + above).to_bits() != 1.0_f32.to_bits()
        || std::hint::black_box(one - below).to_bits() != 1.0_f32.to_bits()
        || std::hint::black_box(multiplicand * multiplicand).to_bits() != 0x3f80_0002
    {
        return Err(NyError::SoundnessRefusal(
            "original binary32 reference requires ordinary nearest binary32 arithmetic".into(),
        ));
    }
    if !ny_core::dd_selfcheck::dd_selfcheck_ok() {
        return Err(bad("binary32 full-gap TwoSum qualification refused"));
    }
    Ok(())
}
fn mul_up(a: f64, b: f64) -> f64 {
    if a == 0.0 || b == 0.0 {
        0.0
    } else {
        next_up_f64(a * b)
    }
}
fn add_up(a: f64, b: f64) -> f64 {
    if a == 0.0 {
        b
    } else if b == 0.0 {
        a
    } else {
        next_up_f64(a + b)
    }
}
fn earliest(a: Option<Instant>, b: Option<Instant>) -> Option<Instant> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    }
}
fn checkpoint(budget: &WorkBudget, deadline: Option<Instant>) -> Result<()> {
    budget
        .checkpoint_for(LeaseClass::Fallback)
        .map_err(refusal)?;
    if earliest(deadline, ny_core::instance_budget::deadline()).is_some_and(|d| Instant::now() >= d)
    {
        return Err(NyError::DeadlineExceeded(
            "original binary32 reference".into(),
        ));
    }
    Ok(())
}
fn poll(budget: &WorkBudget, deadline: Option<Instant>, visit: usize) -> Result<()> {
    if visit.is_multiple_of(POLL) {
        checkpoint(budget, deadline)?;
    }
    Ok(())
}
fn bad(message: &str) -> NyError {
    NyError::UnsupportedConfiguration(format!("original binary32 reference: {message}"))
}
fn overflow() -> NyError {
    bad("checked capacity/work overflow")
}
fn memory(required_bytes: usize, budget_bytes: usize, site: &'static str) -> NyError {
    NyError::CpuMemoryExceeded {
        required_bytes,
        budget_bytes,
        site,
    }
}
fn refusal(reason: WorkRefusal) -> NyError {
    match reason {
        WorkRefusal::Deadline => {
            NyError::DeadlineExceeded("original binary32 reference work".into())
        }
        reason => bad(&format!("same original work refused: {reason}")),
    }
}

#[cfg(test)]
#[path = "precision_tests.rs"]
mod tests;
