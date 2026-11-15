// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Leased, cooperative compilation of the actual original twin-wall spec.
//!
//! This is model ownership only. It grants no root-source product capability,
//! changes no existing compiler admission, and never clears source conflicts.

use std::mem::size_of;
use std::sync::Arc;
use std::time::Instant;

use ny_core::{NyError, Result};
use sha2::{Digest, Sha256};

use crate::certify::work::{LeaseClass, PersistentReservation, WorkBudget, WorkLease, WorkRequest};

use super::super::net::{ConvOp, TwinNet, TwinOp};
use super::super::spec::{TwinOpSpec, TwinSpec};
use super::ir::{checkpoint, earliest, invalid, refusal, OriginalParameterContract};
use super::liveness::{actual_compiled_bytes, compile_prediction, spec_capacity_bytes};

const POLL: usize = 4096;
const OWNER_BYTES: usize = 1024;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::margin_row) struct OriginalModelIdentity {
    source: [u8; 32],
    compiled: [u8; 32],
    contract: OriginalParameterContract,
}

impl OriginalModelIdentity {
    pub(in crate::margin_row) fn source_digest(&self) -> &[u8; 32] {
        &self.source
    }
    pub(in crate::margin_row) fn compiled_digest(&self) -> &[u8; 32] {
        &self.compiled
    }
    pub(in crate::margin_row) fn contract(&self) -> OriginalParameterContract {
        self.contract
    }
}

struct ModelInner {
    net: TwinNet,
    identity: OriginalModelIdentity,
    budget: WorkBudget,
    deadline: Option<Instant>,
    actual_bytes: usize,
    _resident: PersistentReservation,
}

/// Clone shares the exact immutable model and its last-owner byte claim.
/// There is no constructor from an existing mutable net, imported hash or
/// declared residency, and no mutable accessor or mutable dereference.
#[derive(Clone)]
pub(in crate::margin_row) struct OriginalModel(Arc<ModelInner>);

impl OriginalModel {
    pub(in crate::margin_row) fn compile(
        spec: TwinSpec,
        budget: &WorkBudget,
        requested_deadline: Option<Instant>,
    ) -> Result<Self> {
        if budget.memory_authority().is_none() {
            return Err(invalid(
                "compilation requires inherited shared memory authority",
            ));
        }
        let deadline = earliest(requested_deadline, ny_core::instance_budget::deadline());
        checkpoint(budget, deadline)?;
        // Price header inspection before scanning an arbitrarily long op list.
        let header_work = spec
            .ops
            .len()
            .checked_mul(128)
            .and_then(|n| n.checked_add(1024))
            .and_then(|n| u64::try_from(n).ok())
            .ok_or_else(|| invalid("header work overflow"))?;
        let header = budget
            .lease_for(
                LeaseClass::Fallback,
                WorkRequest {
                    work: header_work,
                    ..WorkRequest::default()
                },
            )
            .map_err(refusal)?;
        let deadline = earliest(deadline, header.deadline());
        let source_bytes = spec_capacity_bytes(&spec, budget, deadline)?;
        let metadata = spec
            .ops
            .len()
            .checked_add(1)
            .and_then(|n| n.checked_mul(32))
            .and_then(|n| n.checked_add(OWNER_BYTES))
            .ok_or_else(|| invalid("compiler metadata overflow"))?;
        let geometry_scratch = source_bytes
            .checked_add(metadata)
            .ok_or_else(|| invalid("source overlap overflow"))?;
        let local_cap = super::super::root_box::ROOT_SOURCE_BYTES_CAP;
        if geometry_scratch > local_cap {
            return Err(NyError::CpuMemoryExceeded {
                required_bytes: geometry_scratch,
                budget_bytes: local_cap,
                site: "original support source and geometry",
            });
        }
        let mut geometry = budget
            .lease_for(
                LeaseClass::Fallback,
                WorkRequest {
                    scratch_bytes: geometry_scratch,
                    work: header_work
                        .checked_mul(4)
                        .ok_or_else(|| invalid("geometry work overflow"))?,
                    ..WorkRequest::default()
                },
            )
            .map_err(refusal)?;
        checkpoint(budget, deadline)?;
        let (sizes, head) = validate_geometry(&spec, budget, deadline)?;
        let request = compile_prediction(&spec, &sizes, budget, deadline)?;
        let mut lease = lease_compilation(budget, geometry_scratch, request, deadline)?;
        checkpoint(budget, deadline)?;
        let source = source_identity(&spec, budget, deadline)?;
        let net = compile_owned(spec, sizes, head, budget, deadline)?;
        let actual_bytes = actual_compiled_bytes(&net, budget, deadline)?
            .checked_add(OWNER_BYTES)
            .ok_or_else(|| invalid("owner capacity overflow"))?;
        if actual_bytes > request.persistent_bytes {
            return Err(NyError::CpuMemoryExceeded {
                required_bytes: actual_bytes,
                budget_bytes: request.persistent_bytes,
                site: "original support model allocator capacity",
            });
        }
        let compiled = compiled_identity(&net, budget, deadline)?;
        // Source-only metadata is gone; the moved tensor-size vector is now
        // covered by the complete model lease, before geometry scratch drops.
        geometry.complete().map_err(refusal)?;
        drop(geometry);
        drop(header);
        checkpoint(budget, deadline)?;
        lease.mark_materialized().map_err(refusal)?;
        let resident = lease.commit_persistent().map_err(refusal)?;
        let model = Self(Arc::new(ModelInner {
            net,
            identity: OriginalModelIdentity {
                source,
                compiled,
                contract: OriginalParameterContract::Binary64WithFoldUncertainty,
            },
            budget: budget.clone(),
            deadline,
            actual_bytes,
            _resident: resident,
        }));
        checkpoint(budget, deadline)?;
        Ok(model)
    }

    pub(in crate::margin_row) fn net(&self) -> &TwinNet {
        &self.0.net
    }
    pub(in crate::margin_row) fn budget(&self) -> &WorkBudget {
        &self.0.budget
    }
    pub(in crate::margin_row) fn identity(&self) -> &OriginalModelIdentity {
        &self.0.identity
    }
    pub(in crate::margin_row) fn same_model(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
    pub(in crate::margin_row) fn resident_bytes(&self) -> usize {
        self.0._resident.bytes()
    }
    pub(in crate::margin_row) fn actual_bytes(&self) -> usize {
        self.0.actual_bytes
    }

    /// Capacity-frontier controls may conservatively overpay an authentic
    /// sole-owned model. Numerical data, identity and deadlines are untouched;
    /// equal integer limits or a foreign reservation cannot mint this claim.
    #[cfg(test)]
    pub(in crate::margin_row) fn reprice_for_test(
        mut self,
        resident: PersistentReservation,
    ) -> Result<Self> {
        if !resident.shares_ledger(self.budget()) || resident.bytes() < self.resident_bytes() {
            return Err(invalid(
                "test repricing requires a paid original model reservation",
            ));
        }
        let inner = Arc::get_mut(&mut self.0)
            .ok_or_else(|| invalid("test repricing requires sole original ownership"))?;
        inner._resident = resident;
        Ok(self)
    }
    pub(in crate::margin_row) fn deadline(
        &self,
        requested: Option<Instant>,
    ) -> Result<Option<Instant>> {
        let deadline = earliest(
            earliest(self.0.deadline, requested),
            ny_core::instance_budget::deadline(),
        );
        checkpoint(self.budget(), deadline)?;
        Ok(deadline)
    }
}

/// The source/geometry lease stays live through compilation. Its actual paid
/// capacity and ALL new compile scratch/persistent bytes share the unchanged
/// local ceiling; the original global ledger is still independently required.
fn lease_compilation(
    budget: &WorkBudget,
    geometry_scratch: usize,
    request: WorkRequest,
    deadline: Option<Instant>,
) -> Result<WorkLease> {
    checkpoint(budget, deadline)?;
    let required = geometry_scratch
        .checked_add(request.scratch_bytes)
        .and_then(|n| n.checked_add(request.persistent_bytes))
        .ok_or_else(|| invalid("compilation local peak overflow"))?;
    let cap = super::super::root_box::ROOT_SOURCE_BYTES_CAP;
    if required > cap {
        return Err(NyError::CpuMemoryExceeded {
            required_bytes: required,
            budget_bytes: cap,
            site: "original support compilation total live",
        });
    }
    let lease = budget
        .lease_for(LeaseClass::Fallback, request)
        .map_err(refusal)?;
    checkpoint(budget, earliest(deadline, lease.deadline()))?;
    Ok(lease)
}

fn product(values: &[usize]) -> Result<usize> {
    values.iter().try_fold(1usize, |n, &d| {
        if d == 0 {
            return Err(invalid("zero original tensor/kernel extent"));
        }
        n.checked_mul(d)
            .ok_or_else(|| invalid("original geometry product overflow"))
    })
}

fn validate_geometry(
    spec: &TwinSpec,
    budget: &WorkBudget,
    deadline: Option<Instant>,
) -> Result<(Vec<usize>, usize)> {
    if spec.n_in == 0 || spec.ops.is_empty() {
        return Err(invalid("empty original model"));
    }
    let mut sizes = reserved(spec.ops.len() + 1, budget, deadline)?;
    sizes.push(spec.n_in);
    let mut first_head = None;
    let mut gemms = 0usize;
    for (k, op) in spec.ops.iter().enumerate() {
        checkpoint(budget, deadline)?;
        let input = |id: usize| {
            sizes
                .get(id)
                .copied()
                .filter(|_| id <= k)
                .ok_or_else(|| invalid("future original tensor edge"))
        };
        let n = match op {
            TwinOpSpec::Conv {
                input: id,
                weight,
                bias,
                bias_err,
                kernel,
                stride,
                pads,
                ishape,
                oshape,
                ..
            }
            | TwinOpSpec::ConvTranspose {
                input: id,
                weight,
                bias,
                bias_err,
                kernel,
                stride,
                pads,
                ishape,
                oshape,
                ..
            } => {
                let (co, ci, kh, kw) = *kernel;
                let (ic, ih, iw) = *ishape;
                let (oc, oh, ow) = *oshape;
                let n = product(&[oc, oh, ow])?;
                if input(*id)? != product(&[ic, ih, iw])?
                    || ci != ic
                    || co != oc
                    || weight.len() != product(&[co, ci, kh, kw])?
                    || bias.len() != co
                    || bias_err.len() != co
                    || stride.0 == 0
                    || stride.1 == 0
                {
                    return Err(invalid("original convolution geometry mismatch"));
                }
                let (eh, ew) = match op {
                    TwinOpSpec::Conv { .. } => {
                        let axis = |i: usize,
                                    before: usize,
                                    after: usize,
                                    kernel: usize,
                                    stride: usize| {
                            i.checked_add(before)
                                .and_then(|n| n.checked_add(after))
                                .and_then(|n| n.checked_sub(kernel))
                                .and_then(|n| (n / stride).checked_add(1))
                        };
                        (
                            axis(ih, pads.0, pads.2, kh, stride.0),
                            axis(iw, pads.1, pads.3, kw, stride.1),
                        )
                    }
                    TwinOpSpec::ConvTranspose { out_pad, .. } => {
                        let axis = |i: usize,
                                    before: usize,
                                    after: usize,
                                    kernel: usize,
                                    stride: usize,
                                    pad: usize| {
                            i.checked_sub(1)
                                .and_then(|n| n.checked_mul(stride))
                                .and_then(|n| n.checked_add(kernel))
                                .and_then(|n| n.checked_add(pad))
                                .and_then(|n| {
                                    before.checked_add(after).and_then(|p| n.checked_sub(p))
                                })
                        };
                        (
                            axis(ih, pads.0, pads.2, kh, stride.0, out_pad.0),
                            axis(iw, pads.1, pads.3, kw, stride.1, out_pad.1),
                        )
                    }
                    _ => unreachable!(),
                };
                if eh != Some(oh) || ew != Some(ow) {
                    return Err(invalid("original convolution output geometry mismatch"));
                }
                // Signed gather arithmetic is bounded before any table loop.
                for value in [ih, iw, oh, ow, kh, kw, pads.0, pads.1] {
                    isize::try_from(value)
                        .map_err(|_| invalid("original signed geometry overflow"))?;
                }
                for (out, stride, kernel, pad) in [
                    (oh, stride.0, kh, pads.0),
                    (ow, stride.1, kw, pads.1),
                    (ih, stride.0, kh, pads.0),
                    (iw, stride.1, kw, pads.1),
                ] {
                    out.checked_mul(stride)
                        .and_then(|n| n.checked_add(kernel))
                        .and_then(|n| n.checked_add(pad))
                        .and_then(|n| isize::try_from(n).ok())
                        .ok_or_else(|| invalid("original anchor geometry overflow"))?;
                }
                n
            }
            TwinOpSpec::ChannelAffine {
                input: id,
                scale,
                shift,
                shift_err,
                shape,
                ..
            } => {
                let n = product(&[shape.0, shape.1, shape.2])?;
                if input(*id)? != n
                    || [scale.len(), shift.len(), shift_err.len()]
                        .iter()
                        .any(|&n| n != shape.0)
                {
                    return Err(invalid("original affine geometry mismatch"));
                }
                n
            }
            TwinOpSpec::Relu { input: id } => input(*id)?,
            TwinOpSpec::Flatten { input: id } => input(*id)?,
            TwinOpSpec::Add { lhs, rhs } => {
                let n = input(*lhs)?;
                if input(*rhs)? != n {
                    return Err(invalid("original add shape mismatch"));
                }
                n
            }
            TwinOpSpec::Gemm {
                input: id,
                weight,
                bias,
                bias_err,
                shape,
                ..
            } => {
                if input(*id)? != shape.1
                    || weight.len() != product(&[shape.0, shape.1])?
                    || bias.len() != shape.0
                    || !(bias_err.is_empty() || bias_err.len() == shape.0)
                {
                    return Err(invalid("original Gemm geometry mismatch"));
                }
                first_head.get_or_insert(k);
                gemms += 1;
                shape.0
            }
        };
        sizes.push(n);
    }
    let head = first_head.ok_or_else(|| invalid("missing original head"))?;
    // A genuine affine MLP starts at the first head Gemm and has an empty
    // trunk prefix. Its original input already supplies the head's exact flat
    // coordinates; no artificial convolution, gate or flatten is required.
    if gemms != 2
        || head.checked_add(3) != Some(spec.ops.len())
        || !matches!(spec.ops.get(head + 1), Some(TwinOpSpec::Relu { input }) if *input == head + 1)
        || !matches!(spec.ops.get(head + 2), Some(TwinOpSpec::Gemm { input, .. }) if *input == head + 2)
    {
        return Err(invalid("original head wiring mismatch"));
    }
    checkpoint(budget, deadline)?;
    Ok((sizes, head))
}

fn compile_owned(
    spec: TwinSpec,
    sizes: Vec<usize>,
    head: usize,
    budget: &WorkBudget,
    deadline: Option<Instant>,
) -> Result<TwinNet> {
    let total_ops = spec.ops.len();
    let mut ops = reserved(spec.ops.len(), budget, deadline)?;
    let mut trunk_relus = reserved(head, budget, deadline)?;
    let n_in = spec.n_in;
    for (k, op) in spec.ops.into_iter().enumerate() {
        checkpoint(budget, deadline)?;
        let compiled = match op {
            TwinOpSpec::Conv {
                input,
                weight,
                bias,
                bias_err,
                weight_rel_err,
                kernel,
                stride,
                pads,
                ishape,
                oshape,
            } => TwinOp::Conv(Box::new(conv(
                input,
                weight,
                bias,
                bias_err,
                weight_rel_err,
                kernel,
                stride,
                pads,
                ishape,
                oshape,
                false,
                budget,
                deadline,
            )?)),
            TwinOpSpec::ConvTranspose {
                input,
                weight,
                bias,
                bias_err,
                weight_rel_err,
                kernel,
                stride,
                pads,
                ishape,
                oshape,
                ..
            } => TwinOp::Conv(Box::new(conv(
                input,
                weight,
                bias,
                bias_err,
                weight_rel_err,
                kernel,
                stride,
                pads,
                ishape,
                oshape,
                true,
                budget,
                deadline,
            )?)),
            TwinOpSpec::ChannelAffine {
                input,
                scale,
                shift,
                shift_err,
                scale_rel_err,
                shape,
            } => {
                numeric(&scale, false, budget, deadline)?;
                numeric(&shift, false, budget, deadline)?;
                numeric(&shift_err, true, budget, deadline)?;
                nonnegative(scale_rel_err)?;
                let hw = shape.1 * shape.2;
                let expand = |values: Vec<f64>| -> Result<Vec<f64>> {
                    let mut out = reserved(shape.0 * hw, budget, deadline)?;
                    for value in values {
                        for _ in 0..hw {
                            poll(budget, deadline, out.len())?;
                            out.push(value);
                        }
                    }
                    Ok(out)
                };
                TwinOp::ChannelAffine {
                    input,
                    scale: expand(scale)?,
                    shift: expand(shift)?,
                    shift_err: expand(shift_err)?,
                    scale_rel_err,
                }
            }
            TwinOpSpec::Relu { input } => {
                let layer = if k < head {
                    trunk_relus.push(k);
                    trunk_relus.len() - 1
                } else {
                    usize::MAX
                };
                TwinOp::Relu { input, layer }
            }
            TwinOpSpec::Flatten { input } => TwinOp::Flatten { input },
            TwinOpSpec::Add { lhs, rhs } => TwinOp::Add { lhs, rhs },
            TwinOpSpec::Gemm {
                input,
                weight,
                bias,
                bias_err,
                weight_rel_err,
                shape,
            } => {
                numeric(&weight, false, budget, deadline)?;
                numeric(&bias, false, budget, deadline)?;
                numeric(&bias_err, true, budget, deadline)?;
                nonnegative(weight_rel_err)?;
                if k == total_ops - 1 {
                    if weight_rel_err != 0.0 {
                        return Err(invalid("final original Gemm carries relative fold error"));
                    }
                    // Parameter scans belong under the full compilation
                    // lease, not the earlier geometry-only admission.
                    for (i, &error) in bias_err.iter().enumerate() {
                        poll(budget, deadline, i)?;
                        if error != 0.0 {
                            return Err(invalid("final original Gemm carries bias fold error"));
                        }
                    }
                }
                let bias_err = if bias_err.is_empty() {
                    filled(shape.0, 0.0, budget, deadline)?
                } else {
                    bias_err
                };
                TwinOp::Gemm {
                    input,
                    weight,
                    bias,
                    bias_err,
                    weight_rel_err,
                    shape,
                }
            }
        };
        ops.push(compiled);
    }
    checkpoint(budget, deadline)?;
    Ok(TwinNet {
        n_in,
        n_y: sizes[head + 1],
        n_out: *sizes.last().unwrap(),
        ops,
        tsize: sizes,
        i_gemm1: head,
        trunk_relus,
    })
}

/// Conservative NEW allocation overlap for a zero-padded virtual convolution.
/// The authenticated caller separately retains/prices the original source,
/// every other virtual operator, plan metadata and coefficient scratch.
pub(in crate::margin_row) fn virtual_conv_allocation_bytes(
    original: &ConvOp,
    ishape: (usize, usize, usize),
    oshape: (usize, usize, usize),
) -> Result<usize> {
    virtual_conv_geometry(original, ishape, oshape)?;
    let (_, ci, kh, kw) = original.kernel;
    let input_plane = product(&[ishape.1, ishape.2])?;
    let taps = product(&[kh, kw])?;
    let gather = product(&[ci, taps, oshape.1, oshape.2])?;
    let records = input_plane
        .checked_mul(taps)
        .ok_or_else(|| invalid("virtual tap allocation overflow"))?;
    let mut bytes = size_of::<ConvOp>()
        .checked_add(1024)
        .ok_or_else(|| invalid("virtual convolution metadata overflow"))?;
    for (count, width) in [
        (original.wmat.len(), 2 * size_of::<f64>()),
        (original.bias.len(), size_of::<f64>()),
        (original.bias_err.len(), size_of::<f64>()),
        (gather, size_of::<usize>()),
        (input_plane, size_of::<Vec<(usize, usize, usize)>>()),
        (records, size_of::<(usize, usize, usize)>()),
    ] {
        bytes = count
            .checked_mul(width)
            .and_then(|n| bytes.checked_add(n))
            .ok_or_else(|| invalid("virtual convolution allocation overflow"))?;
    }
    Ok(bytes)
}

fn virtual_conv_geometry(
    original: &ConvOp,
    ishape: (usize, usize, usize),
    oshape: (usize, usize, usize),
) -> Result<()> {
    let (co, ci, kh, kw) = original.kernel;
    if original.transposed
        || ishape.0 != ci
        || oshape.0 != co
        || original.stride.0 == 0
        || original.stride.1 == 0
        || [
            co,
            ci,
            kh,
            kw,
            ishape.1,
            ishape.2,
            oshape.1,
            oshape.2,
            original.stride.0,
            original.stride.1,
        ]
        .iter()
        .any(|&n| n == 0 || isize::try_from(n).is_err())
        || original.wmat.len() != product(&[co, ci, kh, kw])?
        || original.bias.len() != co
        || !(original.bias_err.is_empty() || original.bias_err.len() == co)
    {
        return Err(invalid("invalid original/virtual convolution geometry"));
    }
    let high = |out: usize, step: usize, kernel: usize| {
        out.checked_sub(1)
            .and_then(|n| n.checked_mul(step))
            .and_then(|n| n.checked_add(kernel))
    };
    if high(oshape.1, original.stride.0, kh).is_none_or(|n| n > ishape.1)
        || high(oshape.2, original.stride.1, kw).is_none_or(|n| n > ishape.2)
    {
        return Err(invalid(
            "virtual convolution would require padding or overflow",
        ));
    }
    // These complete products are used by the original polled builder below.
    product(&[ci, ishape.1, ishape.2])?;
    product(&[ishape.1, ishape.2])?;
    product(&[ci, kh, kw, oshape.1, oshape.2])?;
    product(&[kh, kw, co])?;
    Ok(())
}

/// Arithmetic-only virtual tables, never a model/root/prefix proof authority.
/// The prepared-patch caller must preadmit the estimate and original-source
/// overlap on its SAME ledger before this helper copies any scalar buffer.
pub(in crate::margin_row) fn virtual_conv_from_original(
    original: &ConvOp,
    ishape: (usize, usize, usize),
    oshape: (usize, usize, usize),
    budget: &WorkBudget,
    deadline: Option<Instant>,
) -> Result<ConvOp> {
    checkpoint(budget, deadline)?;
    if budget.memory_authority().is_none() {
        return Err(invalid(
            "virtual convolution requires inherited shared memory authority",
        ));
    }
    let bytes = virtual_conv_allocation_bytes(original, ishape, oshape)?;
    let cap = super::super::root_box::ROOT_SOURCE_BYTES_CAP;
    if bytes > cap {
        return Err(NyError::CpuMemoryExceeded {
            required_bytes: bytes,
            budget_bytes: cap,
            site: "original virtual convolution allocations",
        });
    }
    let copy = |values: &[f64]| -> Result<Vec<f64>> {
        let mut out = reserved(values.len(), budget, deadline)?;
        for (i, &value) in values.iter().enumerate() {
            poll(budget, deadline, i)?;
            out.push(value);
        }
        checkpoint(budget, deadline)?;
        Ok(out)
    };
    let mut built = conv(
        original.input,
        copy(&original.wmat)?,
        copy(&original.bias)?,
        copy(&original.bias_err)?,
        original.weight_rel_err,
        original.kernel,
        original.stride,
        (0, 0, 0, 0),
        ishape,
        oshape,
        false,
        budget,
        deadline,
    )?;
    for (i, &gather) in built.gather.iter().enumerate() {
        poll(budget, deadline, i)?;
        if gather == usize::MAX {
            return Err(invalid("virtual gather unexpectedly requires padding"));
        }
    }
    // The patch caller's masks bind actual original nonzero terms. Its error
    // envelope stays the original full-operator envelope, as in legacy plans.
    built.k_bwd = original.k_bwd;
    checkpoint(budget, deadline)?;
    Ok(built)
}

#[allow(clippy::too_many_arguments)]
fn conv(
    input: usize,
    weight: Vec<f64>,
    bias: Vec<f64>,
    bias_err: Vec<f64>,
    rel: f64,
    kernel: (usize, usize, usize, usize),
    stride: (usize, usize),
    pads: (usize, usize, usize, usize),
    ishape: (usize, usize, usize),
    oshape: (usize, usize, usize),
    transposed: bool,
    budget: &WorkBudget,
    deadline: Option<Instant>,
) -> Result<ConvOp> {
    numeric(&weight, false, budget, deadline)?;
    numeric(&bias, false, budget, deadline)?;
    numeric(&bias_err, true, budget, deadline)?;
    nonnegative(rel)?;
    let (co, ci, kh, kw) = kernel;
    let (_, ih, iw) = ishape;
    let (_, oh, ow) = oshape;
    let k = ci * kh * kw;
    let p = oh * ow;
    let mut gather = filled(k * p, usize::MAX, budget, deadline)?;
    let mut visits = 0usize;
    for oy in 0..oh {
        for ox in 0..ow {
            for c in 0..ci {
                for ky in 0..kh {
                    for kx in 0..kw {
                        let at = (c * kh * kw + ky * kw + kx) * p + oy * ow + ox;
                        // Storage indices jump between K-major rows; poll a monotone visit
                        // count so a large spatial stride cannot postpone cancellation.
                        poll(budget, deadline, visits)?;
                        visits += 1;
                        let position = if transposed {
                            let ny = oy + pads.0;
                            let nx = ox + pads.1;
                            if ny >= ky
                                && nx >= kx
                                && (ny - ky).is_multiple_of(stride.0)
                                && (nx - kx).is_multiple_of(stride.1)
                            {
                                Some(((ny - ky) / stride.0, (nx - kx) / stride.1))
                            } else {
                                None
                            }
                        } else {
                            let y = (oy * stride.0 + ky) as isize - pads.0 as isize;
                            let x = (ox * stride.1 + kx) as isize - pads.1 as isize;
                            (y >= 0 && x >= 0).then_some((y as usize, x as usize))
                        };
                        if let Some((y, x)) = position.filter(|&(y, x)| y < ih && x < iw) {
                            gather[at] = c * ih * iw + y * iw + x;
                        }
                    }
                }
            }
        }
    }
    let tap_position = |iy: usize, ix: usize, ky: usize, kx: usize| {
        if transposed {
            let ny = iy * stride.0 + ky;
            let nx = ix * stride.1 + kx;
            if ny >= pads.0 && nx >= pads.1 {
                Some((ny - pads.0, nx - pads.1))
            } else {
                None
            }
        } else {
            let ny = iy + pads.0;
            let nx = ix + pads.1;
            if ny >= ky
                && nx >= kx
                && (ny - ky).is_multiple_of(stride.0)
                && (nx - kx).is_multiple_of(stride.1)
            {
                Some(((ny - ky) / stride.0, (nx - kx) / stride.1))
            } else {
                None
            }
        }
        .filter(|&(y, x)| y < oh && x < ow)
    };
    let mut back_taps = reserved(ih * iw, budget, deadline)?;
    let mut max_taps = 0;
    for iy in 0..ih {
        for ix in 0..iw {
            checkpoint(budget, deadline)?;
            let mut count = 0;
            for ky in 0..kh {
                for kx in 0..kw {
                    poll(budget, deadline, ky * kw + kx)?;
                    count += usize::from(tap_position(iy, ix, ky, kx).is_some());
                }
            }
            let mut taps = reserved(count, budget, deadline)?;
            for ky in 0..kh {
                for kx in 0..kw {
                    poll(budget, deadline, ky * kw + kx)?;
                    if let Some((y, x)) = tap_position(iy, ix, ky, kx) {
                        taps.push((ky, kx, y * ow + x));
                    }
                }
            }
            max_taps = max_taps.max(taps.len());
            back_taps.push(taps);
        }
    }
    let mut wt = filled(weight.len(), 0.0, budget, deadline)?;
    for o in 0..co {
        for c in 0..ci {
            for ky in 0..kh {
                for kx in 0..kw {
                    let src = ((o * ci + c) * kh + ky) * kw + kx;
                    poll(budget, deadline, src)?;
                    wt[((c * kh + ky) * kw + kx) * co + o] = weight[src];
                }
            }
        }
    }
    checkpoint(budget, deadline)?;
    Ok(ConvOp {
        input,
        kernel,
        ishape,
        oshape,
        stride,
        pads,
        transposed,
        wmat: weight,
        wt,
        bias,
        bias_err,
        weight_rel_err: rel,
        gather,
        back_taps,
        k_fwd: k,
        k_bwd: max_taps * co,
    })
}

pub(super) fn reserved<T>(
    n: usize,
    budget: &WorkBudget,
    deadline: Option<Instant>,
) -> Result<Vec<T>> {
    checkpoint(budget, deadline)?;
    n.checked_mul(size_of::<T>())
        .ok_or_else(|| invalid("original allocation overflow"))?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(n)
        .map_err(|_| invalid("original allocation refused"))?;
    if values.capacity() != n {
        return Err(invalid("original exact allocation capacity mismatch"));
    }
    checkpoint(budget, deadline)?;
    Ok(values)
}
fn filled<T: Copy>(
    n: usize,
    value: T,
    budget: &WorkBudget,
    deadline: Option<Instant>,
) -> Result<Vec<T>> {
    let mut out = reserved(n, budget, deadline)?;
    for i in 0..n {
        poll(budget, deadline, i)?;
        out.push(value);
    }
    Ok(out)
}
fn numeric(
    values: &[f64],
    error: bool,
    budget: &WorkBudget,
    deadline: Option<Instant>,
) -> Result<()> {
    for (i, &value) in values.iter().enumerate() {
        poll(budget, deadline, i)?;
        if !value.is_finite() || (error && value < 0.0) {
            return Err(invalid("nonfinite original parameter/fold budget"));
        }
    }
    checkpoint(budget, deadline)
}
fn nonnegative(value: f64) -> Result<()> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(invalid("invalid original relative fold budget"))
    }
}
fn poll(budget: &WorkBudget, deadline: Option<Instant>, i: usize) -> Result<()> {
    if i.is_multiple_of(POLL) {
        checkpoint(budget, deadline)?;
    }
    Ok(())
}

struct Fingerprint<'a> {
    hash: Sha256,
    budget: &'a WorkBudget,
    deadline: Option<Instant>,
}
impl Fingerprint<'_> {
    fn word(&mut self, n: usize) {
        self.hash.update((n as u64).to_le_bytes());
    }
    fn words(&mut self, words: &[usize]) {
        self.word(words.len());
        for &n in words {
            self.word(n);
        }
    }
    fn floats(&mut self, values: &[f64]) -> Result<()> {
        self.word(values.len());
        for chunk in values.chunks(1024) {
            checkpoint(self.budget, self.deadline)?;
            for &x in chunk {
                self.hash.update(x.to_bits().to_le_bytes());
            }
        }
        Ok(())
    }
    fn op(&mut self, op: &TwinOp) -> Result<()> {
        match op {
            TwinOp::Conv(c) => {
                self.word(0);
                self.words(&[
                    c.input,
                    c.kernel.0,
                    c.kernel.1,
                    c.kernel.2,
                    c.kernel.3,
                    c.stride.0,
                    c.stride.1,
                    c.pads.0,
                    c.pads.1,
                    c.pads.2,
                    c.pads.3,
                    c.ishape.0,
                    c.ishape.1,
                    c.ishape.2,
                    c.oshape.0,
                    c.oshape.1,
                    c.oshape.2,
                    usize::from(c.transposed),
                    c.k_fwd,
                    c.k_bwd,
                ]);
                for v in [&c.wmat, &c.wt, &c.bias, &c.bias_err] {
                    self.floats(v)?;
                }
                self.floats(&[c.weight_rel_err])?;
                self.word(c.gather.len());
                for chunk in c.gather.chunks(1024) {
                    checkpoint(self.budget, self.deadline)?;
                    for &n in chunk {
                        self.word(n);
                    }
                }
                self.word(c.back_taps.len());
                for taps in &c.back_taps {
                    checkpoint(self.budget, self.deadline)?;
                    self.word(taps.len());
                    for chunk in taps.chunks(1024) {
                        checkpoint(self.budget, self.deadline)?;
                        for &(a, b, c) in chunk {
                            self.words(&[a, b, c]);
                        }
                    }
                }
            }
            TwinOp::ChannelAffine {
                input,
                scale,
                shift,
                shift_err,
                scale_rel_err,
            } => {
                self.words(&[1, *input]);
                for v in [scale, shift, shift_err] {
                    self.floats(v)?;
                }
                self.floats(&[*scale_rel_err])?;
            }
            TwinOp::Relu { input, layer } => self.words(&[2, *input, *layer]),
            TwinOp::Add { lhs, rhs } => self.words(&[3, *lhs, *rhs]),
            TwinOp::Flatten { input } => self.words(&[4, *input]),
            TwinOp::Gemm {
                input,
                weight,
                bias,
                bias_err,
                weight_rel_err,
                shape,
            } => {
                self.words(&[5, *input, shape.0, shape.1]);
                for v in [weight, bias, bias_err] {
                    self.floats(v)?;
                }
                self.floats(&[*weight_rel_err])?;
            }
        }
        Ok(())
    }
}

fn source_identity(
    spec: &TwinSpec,
    budget: &WorkBudget,
    deadline: Option<Instant>,
) -> Result<[u8; 32]> {
    let mut h = Fingerprint {
        hash: Sha256::new(),
        budget,
        deadline,
    };
    h.hash.update(b"ny/original-support-source/v1");
    h.words(&[spec.n_in, spec.ops.len()]);
    for op in &spec.ops {
        checkpoint(budget, deadline)?;
        match op {
            TwinOpSpec::Conv {
                input,
                weight,
                bias,
                bias_err,
                weight_rel_err,
                kernel,
                stride,
                pads,
                ishape,
                oshape,
            }
            | TwinOpSpec::ConvTranspose {
                input,
                weight,
                bias,
                bias_err,
                weight_rel_err,
                kernel,
                stride,
                pads,
                ishape,
                oshape,
                ..
            } => {
                h.word(usize::from(matches!(op, TwinOpSpec::ConvTranspose { .. })));
                h.words(&[
                    *input, kernel.0, kernel.1, kernel.2, kernel.3, stride.0, stride.1, pads.0,
                    pads.1, pads.2, pads.3, ishape.0, ishape.1, ishape.2, oshape.0, oshape.1,
                    oshape.2,
                ]);
                if let TwinOpSpec::ConvTranspose { out_pad, .. } = op {
                    h.words(&[out_pad.0, out_pad.1]);
                }
                for v in [weight, bias, bias_err] {
                    h.floats(v)?;
                }
                h.floats(&[*weight_rel_err])?;
            }
            TwinOpSpec::ChannelAffine {
                input,
                scale,
                shift,
                shift_err,
                scale_rel_err,
                shape,
            } => {
                h.words(&[2, *input, shape.0, shape.1, shape.2]);
                for v in [scale, shift, shift_err] {
                    h.floats(v)?;
                }
                h.floats(&[*scale_rel_err])?;
            }
            TwinOpSpec::Relu { input } => h.words(&[3, *input]),
            TwinOpSpec::Add { lhs, rhs } => h.words(&[4, *lhs, *rhs]),
            TwinOpSpec::Flatten { input } => h.words(&[5, *input]),
            TwinOpSpec::Gemm {
                input,
                weight,
                bias,
                bias_err,
                weight_rel_err,
                shape,
            } => {
                h.words(&[6, *input, shape.0, shape.1]);
                for v in [weight, bias, bias_err] {
                    h.floats(v)?;
                }
                h.floats(&[*weight_rel_err])?;
            }
        }
    }
    checkpoint(budget, deadline)?;
    Ok(h.hash.finalize().into())
}
fn compiled_identity(
    net: &TwinNet,
    budget: &WorkBudget,
    deadline: Option<Instant>,
) -> Result<[u8; 32]> {
    let mut h = Fingerprint {
        hash: Sha256::new(),
        budget,
        deadline,
    };
    h.hash.update(b"ny/original-support-compiled/v1");
    h.words(&[net.n_in, net.n_y, net.n_out, net.i_gemm1, net.ops.len()]);
    for values in [&net.tsize, &net.trunk_relus] {
        h.word(values.len());
        for chunk in values.chunks(1024) {
            checkpoint(budget, deadline)?;
            for &n in chunk {
                h.word(n);
            }
        }
    }
    for op in &net.ops {
        checkpoint(budget, deadline)?;
        h.op(op)?;
    }
    checkpoint(budget, deadline)?;
    Ok(h.hash.finalize().into())
}

#[cfg(test)]
#[path = "compiler_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "compiler_affine_tests.rs"]
mod affine_tests;
