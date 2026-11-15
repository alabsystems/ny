// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Producer-owned original tensor intervals and immutable completed prefixes.
//!
//! The declared original box is a conditional verification premise. Every
//! non-input box is derived here from actual original operators. Finite supplied
//! arrays or a mutable RootGates cannot create a completed prefix. Numerical
//! boxes own only their endpoints and root; they never retain parent buffers.
//! Reference endpoints remain F64; an explicit binary32 nominal computation
//! can transfer its genuine paid enclosure here. Root/tableau F32 requests
//! still refuse. Row storage and split-region authority remain separate.

use std::sync::Arc;
use std::time::Instant;

use ny_core::{NyError, Result};

use super::compiler::OriginalModel;
use crate::certify::work::{
    LeaseClass, PersistentReservation, WorkBudget, WorkLease, WorkRefusal, WorkRequest,
};
use crate::margin_row::input_rows::{InputRowPair, InputRowView};
use crate::margin_row::net::TwinOp;
use crate::margin_row::root::LayerGates;
use crate::margin_row::rounding::{next_down, next_up, RoundMode};
use crate::margin_row::RootPrecisionRuntimePolicy;

const POLL: usize = 256;
// Arc counters/alignment and the persistent claim's private Arc metadata.
// Payload capacities are reconciled separately; this is not an RSS estimate.
const OWNER_METADATA: usize = 256;

#[derive(Clone)]
pub(in crate::margin_row) struct OriginalRoot(Arc<RootInner>);

struct RootInner {
    model: OriginalModel,
    lo: Vec<f64>,
    hi: Vec<f64>,
    mid: Vec<f64>,
    rad: Vec<f64>,
    xabs: Vec<f64>,
    policy: RootPrecisionRuntimePolicy,
    deadline: Option<Instant>,
    _resident: PersistentReservation,
}

/// Only the explicit original input box is admitted in this first owner.
/// Child/baked regions need actual parent/cover premises, never an array tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::margin_row) enum OriginalRegion {
    DeclaredBox,
}

impl OriginalRoot {
    pub(in crate::margin_row) fn new(
        model: OriginalModel,
        lo: &[f64],
        hi: &[f64],
        policy: RootPrecisionRuntimePolicy,
        requested_deadline: Option<Instant>,
    ) -> Result<Self> {
        let deadline = model.deadline(requested_deadline)?;
        checkpoint(model.budget(), deadline)?;
        ny_core::require_f64_interval_proof_environment()?;
        if policy.root_f32 || policy.tableau_f32 {
            return Err(bad("F32 root/tableau construction is not implemented"));
        }
        let n = model.net().n_in;
        if n == 0 || lo.len() != n || hi.len() != n {
            return Err(bad("original input dimensions disagree"));
        }
        let bytes = payload(n, 5, 8)?
            .checked_add(owner_bytes::<RootInner>()?)
            .ok_or_else(overflow)?;
        // Caller endpoint spans overlap all five owned output planes. An
        // exactly identical span counts once; partial overlap is conservative.
        let borrowed = payload(n, if std::ptr::eq(lo, hi) { 1 } else { 2 }, 8)?;
        let source = model
            .resident_bytes()
            .checked_add(borrowed)
            .ok_or_else(overflow)?;
        check_local_peak(source, bytes, "prepared original root visible overlap")?;
        let mut lease = admit(model.budget(), bytes, 0, work(n, 16)?, deadline)?;
        let mut lows = exact_vec(n)?;
        let mut highs = exact_vec(n)?;
        let mut mid = exact_vec(n)?;
        let mut rad = exact_vec(n)?;
        let mut xabs = exact_vec(n)?;
        checkpoint(model.budget(), deadline)?;
        lease.mark_materialized().map_err(refusal)?;
        for i in 0..n {
            poll(model.budget(), deadline, i)?;
            if !(lo[i].is_finite() && hi[i].is_finite() && lo[i] <= hi[i]) {
                return Err(bad("invalid declared original box"));
            }
            // Preserve the ordinary root's midpoint formula. Overflow refuses.
            #[allow(clippy::manual_midpoint)]
            let m = (lo[i] + hi[i]) / 2.0;
            let r = next_up((hi[i] - m).max(m - lo[i]));
            let mag = add_up(m.abs(), r);
            if !(m.is_finite() && r.is_finite() && r >= 0.0 && mag.is_finite()) {
                return Err(bad("original midpoint/radius cannot be represented"));
            }
            lows.push(lo[i]);
            highs.push(hi[i]);
            mid.push(m);
            rad.push(r);
            xabs.push(mag);
        }
        checkpoint(model.budget(), deadline)?;
        let resident = lease.commit_persistent().map_err(refusal)?;
        Ok(Self(Arc::new(RootInner {
            model,
            lo: lows,
            hi: highs,
            mid,
            rad,
            xabs,
            policy,
            deadline,
            _resident: resident,
        })))
    }

    pub(in crate::margin_row) fn model(&self) -> &OriginalModel {
        &self.0.model
    }
    pub(in crate::margin_row) fn budget(&self) -> &WorkBudget {
        self.model().budget()
    }
    pub(in crate::margin_row) fn lo(&self) -> &[f64] {
        &self.0.lo
    }
    pub(in crate::margin_row) fn hi(&self) -> &[f64] {
        &self.0.hi
    }
    pub(in crate::margin_row) fn mid(&self) -> &[f64] {
        &self.0.mid
    }
    pub(in crate::margin_row) fn rad(&self) -> &[f64] {
        &self.0.rad
    }
    pub(in crate::margin_row) fn xabs(&self) -> &[f64] {
        &self.0.xabs
    }
    pub(in crate::margin_row) fn policy(&self) -> RootPrecisionRuntimePolicy {
        self.0.policy
    }
    pub(in crate::margin_row) fn mode(&self) -> RoundMode {
        RoundMode::Outward
    }
    pub(in crate::margin_row) fn region(&self) -> OriginalRegion {
        OriginalRegion::DeclaredBox
    }
    pub(in crate::margin_row) fn same_root(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
    pub(in crate::margin_row) fn resident_bytes(&self) -> usize {
        self.0._resident.bytes()
    }
    pub(in crate::margin_row) fn deadline(
        &self,
        requested: Option<Instant>,
    ) -> Result<Option<Instant>> {
        let deadline = earliest(self.0.deadline, self.model().deadline(requested)?);
        checkpoint(self.budget(), deadline)?;
        Ok(deadline)
    }

    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "prepared collector integration")
    )]
    pub(in crate::margin_row) fn input_box(&self) -> Result<CertifiedTensorBox> {
        self.input_box_with_deadline(None)
    }

    pub(in crate::margin_row) fn input_box_with_deadline(
        &self,
        requested: Option<Instant>,
    ) -> Result<CertifiedTensorBox> {
        let deadline = self.deadline(requested)?;
        self.make_box_with_deadline(
            0,
            self.lo().len(),
            work(self.lo().len(), 2)?,
            deadline,
            self.visible_resident_bytes()?,
            |lo, hi, deadline| {
                for i in 0..self.lo().len() {
                    poll(self.budget(), deadline, i)?;
                    lo.push(self.lo()[i]);
                    hi.push(self.hi()[i]);
                }
                Ok(())
            },
        )
    }

    /// Consume a completed binary32 computation without copying endpoints or
    /// duplicating its existing persistent admission. Only the sealed original
    /// producer can supply this owner; arbitrary numeric arrays cannot enter.
    pub(in crate::margin_row) fn certify_reference_f32(
        &self,
        op: usize,
        computed: super::precision::ComputedReferenceF32,
    ) -> Result<(CertifiedTensorBox, super::precision::ReferenceF32Receipt)> {
        let (lo, hi, deadline, resident, receipt) = computed.into_parts(self, op)?;
        let tensor = op.checked_add(1).ok_or_else(overflow)?;
        checkpoint(self.budget(), deadline)?;
        Ok((
            CertifiedTensorBox {
                root: self.clone(),
                tensor,
                lo,
                hi,
                deadline,
                _resident: resident,
            },
            receipt,
        ))
    }

    pub(in crate::margin_row) fn derive_reference_f32_with_deadline(
        &self,
        op: usize,
        first: &CertifiedTensorBox,
        requested: Option<Instant>,
    ) -> Result<(CertifiedTensorBox, super::precision::ReferenceF32Receipt)> {
        let computed = super::precision::derive_reference_f32(self, op, first, requested)?;
        self.certify_reference_f32(op, computed)
    }

    /// Derive one actual original operator's certified reference result.
    /// ReLU commitment additionally seals the gate used by later backward work.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "prepared collector integration")
    )]
    pub(in crate::margin_row) fn derive(
        &self,
        op: usize,
        first: &CertifiedTensorBox,
        second: Option<&CertifiedTensorBox>,
    ) -> Result<CertifiedTensorBox> {
        self.derive_with_deadline(op, first, second, None)
    }

    pub(in crate::margin_row) fn derive_with_deadline(
        &self,
        op: usize,
        first: &CertifiedTensorBox,
        second: Option<&CertifiedTensorBox>,
        requested: Option<Instant>,
    ) -> Result<CertifiedTensorBox> {
        let deadline = self.deadline(earliest(
            requested,
            earliest(first.deadline, second.and_then(|value| value.deadline)),
        ))?;
        ny_core::require_f64_interval_proof_environment()?;
        let net = self.model().net();
        if op >= net.i_gemm1 {
            return Err(bad("reference program stops before the original head"));
        }
        let original = net
            .ops
            .get(op)
            .ok_or_else(|| bad("missing original operator"))?;
        let n = *net
            .tsize
            .get(op + 1)
            .ok_or_else(|| bad("missing original output shape"))?;
        self.check_box(first)?;
        let (input, rhs, terms) = match original {
            TwinOp::Conv(c) => (c.input, None, n.checked_mul(c.k_fwd).ok_or_else(overflow)?),
            TwinOp::ChannelAffine { input, .. } => (*input, None, n),
            TwinOp::Relu { input, .. } | TwinOp::Flatten { input } => (*input, None, n),
            TwinOp::Add { lhs, rhs } => (*lhs, Some(*rhs), n),
            TwinOp::Gemm { .. } => return Err(bad("reference program stops before head Gemm")),
        };
        if first.tensor != input || second.is_some() != rhs.is_some() {
            return Err(bad("reference inputs use another original edge"));
        }
        if let (Some(expected), Some(other)) = (rhs, second) {
            self.check_box(other)?;
            if other.tensor != expected {
                return Err(bad("residual rhs uses another original edge"));
            }
        }
        let charged_work = work(terms.checked_add(n).ok_or_else(overflow)?, 32)?;
        let mut source = self
            .visible_resident_bytes()?
            .checked_add(first.resident_bytes())
            .ok_or_else(overflow)?;
        if let Some(other) = second {
            if !std::ptr::eq(first, other) {
                source = source
                    .checked_add(other.resident_bytes())
                    .ok_or_else(overflow)?;
            }
        }
        self.make_box_with_deadline(
            op + 1,
            n,
            charged_work,
            deadline,
            source,
            |lo, hi, deadline| {
                match original {
                    TwinOp::Conv(c) => {
                        let p = c.oshape.1.checked_mul(c.oshape.2).ok_or_else(overflow)?;
                        let k = c.k_fwd;
                        if p == 0 || n != c.kernel.0.checked_mul(p).ok_or_else(overflow)? {
                            return Err(bad("original convolution output mapping is invalid"));
                        }
                        let mut visits = 0usize;
                        for j in 0..n {
                            poll(self.budget(), deadline, j)?;
                            let ch = j / p;
                            let sp = j % p;
                            let bias =
                                *c.bias.get(ch).ok_or_else(|| bad("missing original bias"))?;
                            let be = *c
                                .bias_err
                                .get(ch)
                                .ok_or_else(|| bad("missing bias error"))?;
                            let (mut low, mut high) = uncertain_bias(bias, be)?;
                            for tap in 0..k {
                                poll(self.budget(), deadline, visits)?;
                                visits = visits.checked_add(1).ok_or_else(overflow)?;
                                let source = *c
                                    .gather
                                    .get(
                                        tap.checked_mul(p)
                                            .and_then(|i| i.checked_add(sp))
                                            .ok_or_else(overflow)?,
                                    )
                                    .ok_or_else(|| bad("missing original convolution tap"))?;
                                if source == usize::MAX {
                                    continue;
                                }
                                let w = *c
                                    .wmat
                                    .get(
                                        ch.checked_mul(k)
                                            .and_then(|i| i.checked_add(tap))
                                            .ok_or_else(overflow)?,
                                    )
                                    .ok_or_else(|| bad("missing original convolution weight"))?;
                                let (pl, pu) = uncertain_product(
                                    w,
                                    c.weight_rel_err,
                                    *first
                                        .lo
                                        .get(source)
                                        .ok_or_else(|| bad("tap outside original input"))?,
                                    *first
                                        .hi
                                        .get(source)
                                        .ok_or_else(|| bad("tap outside original input"))?,
                                )?;
                                low = add_down(low, pl);
                                high = add_up(high, pu);
                            }
                            push_checked(lo, hi, low, high)?;
                        }
                    }
                    TwinOp::ChannelAffine {
                        scale,
                        shift,
                        scale_rel_err,
                        shift_err,
                        ..
                    } => {
                        if first.lo.len() != n
                            || [scale.len(), shift.len(), shift_err.len()]
                                .iter()
                                .any(|&len| len != n)
                        {
                            return Err(bad("original affine widths disagree"));
                        }
                        for i in 0..n {
                            poll(self.budget(), deadline, i)?;
                            let (pl, pu) = uncertain_product(
                                scale[i],
                                *scale_rel_err,
                                first.lo[i],
                                first.hi[i],
                            )?;
                            let (bl, bu) = uncertain_bias(shift[i], shift_err[i])?;
                            push_checked(lo, hi, add_down(pl, bl), add_up(pu, bu))?;
                        }
                    }
                    TwinOp::Add { .. } => {
                        let other = second.expect("validated residual input");
                        if first.lo.len() != n || other.lo.len() != n {
                            return Err(bad("residual input widths disagree"));
                        }
                        for i in 0..n {
                            poll(self.budget(), deadline, i)?;
                            push_checked(
                                lo,
                                hi,
                                add_down(first.lo[i], other.lo[i]),
                                add_up(first.hi[i], other.hi[i]),
                            )?;
                        }
                    }
                    TwinOp::Flatten { .. } | TwinOp::Relu { .. } => {
                        if first.lo.len() != n {
                            return Err(bad("original identity/ReLU width disagrees"));
                        }
                        let relu = matches!(original, TwinOp::Relu { .. });
                        for i in 0..n {
                            poll(self.budget(), deadline, i)?;
                            lo.push(if relu {
                                first.lo[i].max(0.0)
                            } else {
                                first.lo[i]
                            });
                            hi.push(if relu {
                                first.hi[i].max(0.0)
                            } else {
                                first.hi[i]
                            });
                        }
                    }
                    TwinOp::Gemm { .. } => unreachable!("refused before allocation"),
                }
                checkpoint(self.budget(), deadline)
            },
        )
        .and_then(|out| {
            checkpoint(self.budget(), deadline)?;
            Ok(out)
        })
    }

    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "prepared collector integration")
    )]
    pub(in crate::margin_row) fn empty_prefix(&self) -> Result<PreparedPrefix> {
        self.empty_prefix_with_deadline(None)
    }

    pub(in crate::margin_row) fn empty_prefix_with_deadline(
        &self,
        requested: Option<Instant>,
    ) -> Result<PreparedPrefix> {
        let deadline = self.deadline(requested)?;
        check_local_peak(
            self.visible_resident_bytes()?,
            owner_bytes::<PrefixNode>()?,
            "prepared empty prefix visible overlap",
        )?;
        let lease = admit(self.budget(), owner_bytes::<PrefixNode>()?, 0, 1, deadline)?;
        let resident = lease.commit_persistent().map_err(refusal)?;
        Ok(PreparedPrefix {
            inner: Some(Arc::new(PrefixNode {
                root: self.clone(),
                previous: None,
                gate: None,
                layers: 0,
                deadline,
                _resident: resident,
            })),
        })
    }

    fn check_box(&self, value: &CertifiedTensorBox) -> Result<()> {
        if !self.same_root(&value.root)
            || self.model().net().tsize.get(value.tensor) != Some(&value.lo.len())
            || value.lo.len() != value.hi.len()
        {
            return Err(bad("box has another original root/tensor scope"));
        }
        Ok(())
    }

    fn visible_resident_bytes(&self) -> Result<usize> {
        self.model()
            .resident_bytes()
            .checked_add(self.resident_bytes())
            .ok_or_else(overflow)
    }

    #[expect(dead_code, reason = "prepared collector integration")]
    fn make_box<F>(
        &self,
        tensor: usize,
        n: usize,
        charged_work: u64,
        fill: F,
    ) -> Result<CertifiedTensorBox>
    where
        F: FnOnce(&mut Vec<f64>, &mut Vec<f64>, Option<Instant>) -> Result<()>,
    {
        self.make_box_with_deadline(
            tensor,
            n,
            charged_work,
            None,
            self.visible_resident_bytes()?,
            fill,
        )
    }

    fn make_box_with_deadline<F>(
        &self,
        tensor: usize,
        n: usize,
        charged_work: u64,
        requested: Option<Instant>,
        visible_source_bytes: usize,
        fill: F,
    ) -> Result<CertifiedTensorBox>
    where
        F: FnOnce(&mut Vec<f64>, &mut Vec<f64>, Option<Instant>) -> Result<()>,
    {
        let deadline = self.deadline(requested)?;
        let bytes = payload(n, 2, 8)?
            .checked_add(owner_bytes::<CertifiedTensorBox>()?)
            .ok_or_else(overflow)?;
        check_local_peak(
            visible_source_bytes,
            bytes,
            if tensor == 0 {
                "prepared original input box visible overlap"
            } else {
                "prepared original reference box visible overlap"
            },
        )?;
        let mut lease = admit(self.budget(), bytes, 0, charged_work, deadline)?;
        let mut lo = exact_vec(n)?;
        let mut hi = exact_vec(n)?;
        checkpoint(self.budget(), deadline)?;
        lease.mark_materialized().map_err(refusal)?;
        fill(&mut lo, &mut hi, deadline)?;
        if lo.len() != n || hi.len() != n {
            return Err(bad("incomplete original box production"));
        }
        checkpoint(self.budget(), deadline)?;
        let resident = lease.commit_persistent().map_err(refusal)?;
        Ok(CertifiedTensorBox {
            root: self.clone(),
            tensor,
            lo,
            hi,
            deadline,
            _resident: resident,
        })
    }
}

/// No Clone: duplicating numerical endpoints requires an explicit paid step.
pub(in crate::margin_row) struct CertifiedTensorBox {
    root: OriginalRoot,
    tensor: usize,
    lo: Vec<f64>,
    hi: Vec<f64>,
    deadline: Option<Instant>,
    _resident: PersistentReservation,
}

impl CertifiedTensorBox {
    pub(in crate::margin_row) fn root(&self) -> &OriginalRoot {
        &self.root
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
    pub(in crate::margin_row) fn resident_bytes(&self) -> usize {
        self._resident.bytes()
    }
    pub(in crate::margin_row) fn deadline(&self) -> Result<Option<Instant>> {
        self.root.deadline(self.deadline)
    }
}

pub(in crate::margin_row) struct PreparedPrefix {
    inner: Option<Arc<PrefixNode>>,
}
struct PrefixNode {
    root: OriginalRoot,
    previous: Option<Arc<PrefixNode>>,
    gate: Option<LayerGates>,
    layers: usize,
    deadline: Option<Instant>,
    _resident: PersistentReservation,
}

impl Clone for PreparedPrefix {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

// Iterative last-owner release also covers very deep original graphs.
impl Drop for PreparedPrefix {
    fn drop(&mut self) {
        let mut cursor = self.inner.take();
        while let Some(node) = cursor {
            match Arc::try_unwrap(node) {
                Ok(mut owned) => {
                    cursor = owned.previous.take();
                }
                Err(shared) => {
                    drop(shared);
                    break;
                }
            }
        }
    }
}

impl PreparedPrefix {
    fn node(&self) -> &PrefixNode {
        self.inner.as_deref().expect("live prefix owner")
    }
    pub(in crate::margin_row) fn root(&self) -> &OriginalRoot {
        &self.node().root
    }
    pub(in crate::margin_row) fn layer_count(&self) -> usize {
        self.node().layers
    }
    pub(in crate::margin_row) fn deadline(
        &self,
        requested: Option<Instant>,
    ) -> Result<Option<Instant>> {
        self.root()
            .deadline(earliest(self.node().deadline, requested))
    }
    /// Shared immutable owner reservations, each counted once. This is a
    /// visible capacity census; it does not replace the common memory ledger.
    pub(in crate::margin_row) fn resident_bytes(
        &self,
        requested: Option<Instant>,
    ) -> Result<usize> {
        let deadline = self.deadline(requested)?;
        let root = self.root();
        let count = self.layer_count().checked_add(1).ok_or_else(overflow)?;
        let mut lease = admit(root.budget(), 0, 0, work(count, 2)?, deadline)?;
        let mut bytes = root
            .model()
            .resident_bytes()
            .checked_add(root.resident_bytes())
            .ok_or_else(overflow)?;
        let mut cursor = Some(self.node());
        let mut visits = 0usize;
        while let Some(node) = cursor {
            poll(root.budget(), deadline, visits)?;
            visits = visits.checked_add(1).ok_or_else(overflow)?;
            bytes = bytes
                .checked_add(node._resident.bytes())
                .ok_or_else(overflow)?;
            cursor = node.previous.as_deref();
        }
        if visits != count {
            return Err(bad("incomplete prefix owner census"));
        }
        checkpoint(root.budget(), deadline)?;
        lease.complete().map_err(refusal)?;
        Ok(bytes)
    }
    pub(in crate::margin_row) fn same_prefix(&self, other: &Self) -> bool {
        Arc::ptr_eq(
            self.inner.as_ref().expect("live"),
            other.inner.as_ref().expect("live"),
        )
    }
    pub(in crate::margin_row) fn gate(&self, layer: usize) -> Option<&LayerGates> {
        let mut current = Some(self.node());
        while let Some(node) = current {
            if node.layers == layer.checked_add(1)? {
                return node.gate.as_ref();
            }
            current = node.previous.as_deref();
        }
        None
    }

    pub(in crate::margin_row) fn for_relu(&self, op: usize) -> Result<(usize, usize)> {
        self.deadline(None)?;
        let net = self.root().model().net();
        if net.trunk_relus.get(self.layer_count()) != Some(&op) {
            return Err(bad("prefix is not completed through this original ReLU"));
        }
        match net.ops.get(op) {
            Some(TwinOp::Relu { input, layer }) if *layer == self.layer_count() => Ok((
                *input,
                *net.tsize
                    .get(*input)
                    .ok_or_else(|| bad("missing preactivation shape"))?,
            )),
            _ => Err(bad("requested prefix is not an original trunk ReLU")),
        }
    }

    /// Open a layer refinement transaction by MOVING a producer-certified
    /// original preactivation and its existing endpoint allocation/claim.
    /// The serial builder owns take/restore of any DAG reference slot.
    pub(in crate::margin_row) fn begin_refinement(
        &self,
        mut pre: CertifiedTensorBox,
        op: usize,
        requested: Option<Instant>,
    ) -> Result<RefinementAccumulator> {
        let deadline = self.deadline(earliest(pre.deadline, requested))?;
        let root = self.root();
        ny_core::require_f64_interval_proof_environment()?;
        root.check_box(&pre)?;
        let (input, n) = self.for_relu(op)?;
        if pre.tensor != input || pre.lo.len() != n || pre.hi.len() != n {
            return Err(bad(
                "refinement transaction uses another original preactivation",
            ));
        }
        let source = self
            .resident_bytes(deadline)?
            .checked_add(pre.resident_bytes())
            .ok_or_else(overflow)?;
        let bytes = owner_bytes::<RefinementAccumulator>()?;
        refinement_peak(
            root.budget(),
            source,
            bytes,
            "prepared refinement transaction owner",
        )?;
        let mut lease = admit(root.budget(), bytes, 0, work(1, 256)?, deadline)?;
        let deadline = earliest(deadline, lease.deadline());
        checkpoint(root.budget(), deadline)?;
        lease.mark_materialized().map_err(refusal)?;
        let resident = lease.commit_persistent().map_err(refusal)?;
        // A borrowed preactivation must carry the same inherited deadline as
        // the transaction, including before the first successful row tile.
        pre.deadline = deadline;
        Ok(RefinementAccumulator {
            prefix: self.clone(),
            pre: Some(pre),
            op,
            deadline,
            _resident: resident,
        })
    }

    /// Intersect an original producer box with authenticated input-relative
    /// rows from this exact immutable prefix. Partial tiles change only their
    /// ordered original IDs. A borrowed diagnostic root or supplied scalars
    /// cannot enter this transaction or construct its certified result.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "prepared collector integration")
    )]
    pub(in crate::margin_row) fn refine_relu(
        &self,
        pre: &CertifiedTensorBox,
        op: usize,
        rows: &InputRowPair<'_>,
        requested: Option<Instant>,
    ) -> Result<CertifiedTensorBox> {
        let root = self.root();
        let deadline = self.deadline(earliest(
            pre.deadline,
            earliest(requested, Some(rows.deadline())),
        ))?;
        ny_core::require_f64_interval_proof_environment()?;
        root.check_box(pre)?;
        let (input, n) = self.for_relu(op)?;
        if pre.tensor != input {
            return Err(bad("refinement uses another original preactivation"));
        }
        rows.validate_prepared_prefix(self, op, rows.targets())?;
        if rows.n_in() != root.lo().len()
            || rows.columns() != rows.targets().len()
            || rows.columns() == 0
        {
            return Err(bad("refinement row/source dimensions disagree"));
        }
        // Owned rows have an immutable finite producer deadline. It remains
        // attached to the result, subsequent operators and completed gates.
        let at = deadline.ok_or_else(|| bad("missing authenticated row deadline"))?;
        // The shared authority may have a larger allowance than the unchanged
        // root-source ceiling. Include the live preactivation separately: its
        // endpoints are never retained through the row proof's prefix owner.
        let output_bytes = payload(n, 2, 8)?
            .checked_add(owner_bytes::<CertifiedTensorBox>()?)
            .ok_or_else(overflow)?;
        let source_bytes = rows
            .resident_floor(at)?
            .checked_add(pre.resident_bytes())
            .ok_or_else(overflow)?;
        check_local_peak(
            source_bytes,
            output_bytes,
            "prepared prefix refinement visible overlap",
        )?;
        let row_visits = rows
            .columns()
            .checked_mul(rows.n_in())
            .ok_or_else(overflow)?;
        let charged_work = work(row_visits, 64)?
            .checked_add(work(n, 4)?)
            .and_then(|v| v.checked_add(work(rows.columns(), 8).ok()?))
            .ok_or_else(overflow)?;
        // All immutable row/source reservations remain live while these two
        // full endpoint planes are admitted against that SAME original ledger.
        root.make_box_with_deadline(
            pre.tensor,
            n,
            charged_work,
            Some(at),
            source_bytes,
            |lo, hi, deadline| {
                for i in 0..n {
                    poll(root.budget(), deadline, i)?;
                    lo.push(pre.lo[i]);
                    hi.push(pre.hi[i]);
                }
                for (column, &target) in rows.targets().iter().enumerate() {
                    poll(root.budget(), deadline, column)?;
                    if target >= n {
                        return Err(bad("refinement target is outside the original tensor"));
                    }
                    let lower = rows
                        .lower(column)
                        .ok_or_else(|| bad("missing explicit lower row"))?;
                    let upper = rows
                        .upper(column)
                        .ok_or_else(|| bad("missing explicit upper row"))?;
                    let lower_bound = input_row_bound(root, &lower, true, at)?;
                    let upper_bound = input_row_bound(root, &upper, false, at)?;
                    let l = lo[target].max(lower_bound);
                    let u = hi[target].min(upper_bound);
                    if !(l.is_finite() && u.is_finite() && l <= u) {
                        return Err(bad("nonfinite/crossed authenticated row intersection"));
                    }
                    lo[target] = l;
                    hi[target] = u;
                }
                checkpoint(root.budget(), deadline)
            },
        )
    }

    /// Commit only a producer-certified original preactivation. No gate-array
    /// completion entry exists. Untrusted finite alpha proposals are clamped
    /// only for genuinely unstable neurons; other gates remain exact.
    pub(in crate::margin_row) fn commit_relu(
        &self,
        pre: &CertifiedTensorBox,
        op: usize,
        alpha_seed: Option<&[f64]>,
    ) -> Result<(Self, CertifiedTensorBox)> {
        let root = self.root();
        let deadline = self.deadline(pre.deadline)?;
        ny_core::require_f64_interval_proof_environment()?;
        root.check_box(pre)?;
        let (input, n) = self.for_relu(op)?;
        if pre.tensor != input || alpha_seed.is_some_and(|seed| seed.len() != n) {
            return Err(bad(
                "gate preactivation/alpha proposal has another original shape",
            ));
        }
        let bytes = payload(n, 6, 8)?
            .checked_add(payload(n, 1, size_of::<usize>())?)
            .and_then(|v| v.checked_add(owner_bytes::<PrefixNode>().ok()?))
            .ok_or_else(overflow)?;
        let output_bytes = payload(n, 2, 8)?
            .checked_add(owner_bytes::<CertifiedTensorBox>()?)
            .ok_or_else(overflow)?;
        let source_bytes = self
            .resident_bytes(deadline)?
            .checked_add(pre.resident_bytes())
            .ok_or_else(overflow)?;
        check_local_peak(
            source_bytes,
            bytes.checked_add(output_bytes).ok_or_else(overflow)?,
            "prepared completed gate visible overlap",
        )?;
        let mut lease = admit(root.budget(), bytes, 0, work(n, 32)?, deadline)?;
        let mut l = exact_vec(n)?;
        let mut u = exact_vec(n)?;
        let mut alpha = exact_vec(n)?;
        let mut s = exact_vec(n)?;
        let mut c = exact_vec(n)?;
        let mut ms = exact_vec(n)?;
        let mut unst = exact_vec(n)?;
        checkpoint(root.budget(), deadline)?;
        lease.mark_materialized().map_err(refusal)?;
        for j in 0..n {
            poll(root.budget(), deadline, j)?;
            let (low, high) = (pre.lo[j], pre.hi[j]);
            let (mut a, slope, intercept) = if high <= 0.0 {
                (0.0, 0.0, 0.0)
            } else if low >= 0.0 {
                (1.0, 1.0, 0.0)
            } else {
                let slope = high / (high - low);
                if !(slope.is_finite() && (0.0..=1.0).contains(&slope)) {
                    return Err(bad("unrepresentable original gate slope"));
                }
                let intercept = mul_up(slope, -low).max(mul_up(sub_up(1.0, slope), high));
                unst.push(j);
                (if high >= -low { 1.0 } else { 0.0 }, slope, intercept)
            };
            if low < 0.0 && high > 0.0 {
                if let Some(seed) = alpha_seed {
                    if seed[j].is_finite() {
                        a = seed[j].clamp(0.0, 1.0);
                    }
                }
            }
            if !intercept.is_finite() {
                return Err(bad("unrepresentable original gate intercept"));
            }
            l.push(low);
            u.push(high);
            alpha.push(a);
            s.push(slope);
            c.push(intercept);
            ms.push(a.max(slope));
        }
        // The output allocation is separately admitted while all gate arrays
        // and the input preactivation are still live. Failure publishes neither.
        let output = root.derive_with_deadline(op, pre, None, deadline)?;
        checkpoint(root.budget(), deadline)?;
        let resident = lease.commit_persistent().map_err(refusal)?;
        let next = Self {
            inner: Some(Arc::new(PrefixNode {
                root: root.clone(),
                previous: self.inner.clone(),
                gate: Some(LayerGates {
                    op,
                    n,
                    l,
                    u,
                    alpha,
                    s,
                    c,
                    ms,
                    unst,
                    clip_rows: None,
                }),
                layers: self.layer_count().checked_add(1).ok_or_else(overflow)?,
                deadline,
                _resident: resident,
            })),
        };
        Ok((next, output))
    }

    /// Paid active metadata indexed by original layer number. The gate arrays
    /// are borrowed from linked immutable owners and never copied per frame.
    pub(in crate::margin_row) fn frame(
        &self,
        requested: Option<Instant>,
    ) -> Result<PreparedPrefixFrame<'_>> {
        let root = self.root();
        let deadline = self.deadline(requested)?;
        let n = self.layer_count();
        let scratch = payload(n, 1, size_of::<&LayerGates>())?
            .checked_add(size_of::<PreparedPrefixFrame<'_>>())
            .ok_or_else(overflow)?;
        check_local_peak(
            self.resident_bytes(deadline)?,
            scratch,
            "prepared prefix frame visible overlap",
        )?;
        let mut lease = admit(root.budget(), 0, scratch, work(n, 2)?, deadline)?;
        let mut layers = exact_vec(n)?;
        checkpoint(root.budget(), deadline)?;
        lease.mark_materialized().map_err(refusal)?;
        let mut current = Some(self.node());
        while let Some(node) = current {
            poll(root.budget(), deadline, layers.len())?;
            if let Some(gate) = &node.gate {
                layers.push(gate);
            }
            current = node.previous.as_deref();
        }
        if layers.len() != n {
            return Err(bad("incomplete prepared prefix metadata"));
        }
        for i in 0..n / 2 {
            poll(root.budget(), deadline, i)?;
            layers.swap(i, n - 1 - i);
        }
        checkpoint(root.budget(), deadline)?;
        lease.complete().map_err(refusal)?;
        Ok(PreparedPrefixFrame {
            prefix: self,
            layers,
            deadline,
            _lease: lease,
        })
    }
}

pub(in crate::margin_row) struct PreparedPrefixFrame<'a> {
    prefix: &'a PreparedPrefix,
    layers: Vec<&'a LayerGates>,
    deadline: Option<Instant>,
    _lease: WorkLease,
}

impl PreparedPrefixFrame<'_> {
    pub(in crate::margin_row) fn prefix(&self) -> &PreparedPrefix {
        self.prefix
    }
    pub(in crate::margin_row) fn root(&self) -> &OriginalRoot {
        self.prefix().root()
    }
    pub(in crate::margin_row) fn gate(&self, layer: usize) -> Option<&LayerGates> {
        self.layers.get(layer).copied()
    }
    pub(in crate::margin_row) fn layers(&self) -> &[&LayerGates] {
        &self.layers
    }
    pub(in crate::margin_row) fn mode(&self) -> RoundMode {
        RoundMode::Outward
    }
    pub(in crate::margin_row) fn deadline(&self) -> Option<Instant> {
        self.deadline
    }
    pub(in crate::margin_row) fn checkpoint(&self) -> Result<()> {
        checkpoint(self.root().budget(), self.deadline())
    }
}

fn earliest(a: Option<Instant>, b: Option<Instant>) -> Option<Instant> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (Some(a), None) => Some(a),
        (None, b) => b,
    }
}
fn checkpoint(budget: &WorkBudget, deadline: Option<Instant>) -> Result<()> {
    budget
        .checkpoint_for(LeaseClass::Fallback)
        .map_err(refusal)?;
    if deadline.is_some_and(|at| Instant::now() >= at) {
        return Err(NyError::DeadlineExceeded("prepared original prefix".into()));
    }
    Ok(())
}
fn poll(budget: &WorkBudget, deadline: Option<Instant>, i: usize) -> Result<()> {
    if i.is_multiple_of(POLL) {
        checkpoint(budget, deadline)?;
    }
    Ok(())
}
fn admit(
    budget: &WorkBudget,
    persistent: usize,
    scratch: usize,
    work: u64,
    deadline: Option<Instant>,
) -> Result<WorkLease> {
    checkpoint(budget, deadline)?;
    let lease = budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                persistent_bytes: persistent,
                scratch_bytes: scratch,
                work,
                proposals: 0,
            },
        )
        .map_err(refusal)?;
    checkpoint(budget, earliest(deadline, lease.deadline()))?;
    Ok(lease)
}
fn owner_bytes<T>() -> Result<usize> {
    size_of::<T>()
        .checked_add(OWNER_METADATA)
        .ok_or_else(overflow)
}
fn check_local_peak(source: usize, output: usize, site: &'static str) -> Result<()> {
    let required_bytes = source.checked_add(output).ok_or_else(overflow)?;
    let budget_bytes = crate::margin_row::root_box::ROOT_SOURCE_BYTES_CAP;
    if required_bytes > budget_bytes {
        return Err(NyError::CpuMemoryExceeded {
            required_bytes,
            budget_bytes,
            site,
        });
    }
    Ok(())
}
fn payload(n: usize, planes: usize, width: usize) -> Result<usize> {
    n.checked_mul(planes)
        .and_then(|v| v.checked_mul(width))
        .ok_or_else(overflow)
}
fn work(n: usize, per: usize) -> Result<u64> {
    u64::try_from(n.checked_mul(per).ok_or_else(overflow)?).map_err(|_| overflow())
}
fn exact_vec<T>(n: usize) -> Result<Vec<T>> {
    let mut out = Vec::new();
    out.try_reserve_exact(n)
        .map_err(|_| bad("original prefix allocation refused"))?;
    if out.capacity() != n {
        return Err(bad(
            "original prefix allocator capacity exceeds its paid payload",
        ));
    }
    Ok(out)
}
fn push_checked(lo: &mut Vec<f64>, hi: &mut Vec<f64>, l: f64, u: f64) -> Result<()> {
    if !(l.is_finite() && u.is_finite() && l <= u) {
        return Err(bad("nonfinite/crossed original reference result"));
    }
    lo.push(l);
    hi.push(u);
    Ok(())
}

/// The producer error is a pointwise algebraic ball; the nominal dot still
/// needs its own directed evaluation. Exact f32 bit widening is supplied by
/// the sealed view, so its full cast gap remains in E/eb during this step.
fn input_row_bound(
    root: &OriginalRoot,
    row: &InputRowView<'_>,
    lower: bool,
    deadline: Instant,
) -> Result<f64> {
    checkpoint(root.budget(), Some(deadline))?;
    let mut nominal = row.bias();
    if !nominal.is_finite() {
        return Err(bad("nonfinite nominal row bias"));
    }
    for (visit, (id, a, e)) in row.entries().enumerate() {
        poll(root.budget(), Some(deadline), visit)?;
        let l = *root
            .lo()
            .get(id)
            .ok_or_else(|| bad("row maps outside the original input"))?;
        let u = *root
            .hi()
            .get(id)
            .ok_or_else(|| bad("row maps outside the original input"))?;
        if !(a.is_finite() && e.is_finite() && e >= 0.0) {
            return Err(bad("invalid sealed coefficient/error entry"));
        }
        let endpoint = if (a >= 0.0) == lower { l } else { u };
        let term = if lower {
            mul_down(a, endpoint)
        } else {
            mul_up_signed(a, endpoint)
        };
        nominal = if lower {
            add_down(nominal, term)
        } else {
            add_up(nominal, term)
        };
        if !nominal.is_finite() {
            return Err(bad("unrepresentable nominal row evaluation"));
        }
    }
    let eta = row.pointwise_error(deadline)?;
    if !eta.is_finite() || eta < 0.0 {
        return Err(bad("invalid full pointwise row error"));
    }
    let bound = if lower {
        sub_down(nominal, eta)
    } else {
        add_up(nominal, eta)
    };
    checkpoint(root.budget(), Some(deadline))?;
    if !bound.is_finite() {
        return Err(bad("unrepresentable charged row bound"));
    }
    Ok(bound)
}
fn uncertain_bias(value: f64, error: f64) -> Result<(f64, f64)> {
    if !(value.is_finite() && error.is_finite() && error >= 0.0) {
        return Err(bad("invalid original bias uncertainty"));
    }
    if error == 0.0 {
        return Ok((value, value));
    }
    Ok((next_down(value - error), next_up(value + error)))
}
fn uncertain_product(w: f64, rho: f64, l: f64, u: f64) -> Result<(f64, f64)> {
    if !(w.is_finite() && rho.is_finite() && rho >= 0.0 && l.is_finite() && u.is_finite() && l <= u)
    {
        return Err(bad("invalid original weight/input uncertainty"));
    }
    if w == 0.0 {
        return Ok((0.0, 0.0));
    }
    let (low, high) = if w > 0.0 {
        (mul_down(w, l), mul_up_signed(w, u))
    } else {
        (mul_down(w, u), mul_up_signed(w, l))
    };
    let gap = mul_up(mul_up(rho, w.abs()), l.abs().max(u.abs()));
    Ok((sub_down(low, gap), add_up(high, gap)))
}
fn mul_down(a: f64, b: f64) -> f64 {
    if a == 0.0 || b == 0.0 {
        a * b
    } else {
        next_down(a * b)
    }
}
fn mul_up_signed(a: f64, b: f64) -> f64 {
    if a == 0.0 || b == 0.0 {
        a * b
    } else {
        next_up(a * b)
    }
}
fn mul_up(a: f64, b: f64) -> f64 {
    mul_up_signed(a, b)
}
fn add_down(a: f64, b: f64) -> f64 {
    if a == 0.0 {
        b
    } else if b == 0.0 {
        a
    } else {
        next_down(a + b)
    }
}
fn add_up(a: f64, b: f64) -> f64 {
    if a == 0.0 {
        b
    } else if b == 0.0 {
        a
    } else {
        next_up(a + b)
    }
}
fn sub_down(a: f64, b: f64) -> f64 {
    add_down(a, -b)
}
fn sub_up(a: f64, b: f64) -> f64 {
    add_up(a, -b)
}
fn overflow() -> NyError {
    bad("original prefix capacity/work overflow")
}
fn bad(message: &str) -> NyError {
    NyError::UnsupportedConfiguration(format!("prepared prefix: {message}"))
}
fn refusal(value: WorkRefusal) -> NyError {
    match value {
        WorkRefusal::Deadline => NyError::DeadlineExceeded("prepared original prefix work".into()),
        other => bad(&format!("shared work refused: {other}")),
    }
}

#[cfg(test)]
#[path = "prefix_tests.rs"]
mod tests;

/// Private numerical transaction, never a completion token for clipping or
/// retention coverage. Its endpoint vectors are moved once and reused for all
/// tiles. Dropping it releases only this transaction's existing owners.
pub(in crate::margin_row) struct RefinementAccumulator {
    prefix: PreparedPrefix,
    pre: Option<CertifiedTensorBox>,
    op: usize,
    deadline: Option<Instant>,
    _resident: PersistentReservation,
}
#[derive(Clone, Copy)]
struct RefinementProposal {
    target: usize,
    lower: f64,
    upper: f64,
    old_lower: f64,
    old_upper: f64,
}
impl RefinementAccumulator {
    pub(in crate::margin_row) fn preactivation(&self) -> &CertifiedTensorBox {
        self.pre.as_ref().expect("live refinement transaction")
    }
    pub(in crate::margin_row) fn prefix(&self) -> &PreparedPrefix {
        &self.prefix
    }
    pub(in crate::margin_row) fn relu_op(&self) -> usize {
        self.op
    }
    pub(in crate::margin_row) fn resident_bytes(&self) -> Result<usize> {
        self.preactivation()
            .resident_bytes()
            .checked_add(self._resident.bytes())
            .ok_or_else(overflow)
    }
    pub(in crate::margin_row) fn deadline(
        &self,
        requested: Option<Instant>,
    ) -> Result<Option<Instant>> {
        self.prefix().deadline(earliest(self.deadline, requested))
    }
    /// A complete selected tile is checked before writes. On cancellation
    /// during the bounded write phase, restore every changed endpoint bit.
    /// Successful unchanged intersections still inherit the row deadline.
    pub(in crate::margin_row) fn apply_rows(
        &mut self,
        rows: &InputRowPair<'_>,
        requested: Option<Instant>,
    ) -> Result<()> {
        self.apply_rows_inner(rows, requested, None)
    }
    fn apply_rows_inner(
        &mut self,
        rows: &InputRowPair<'_>,
        requested: Option<Instant>,
        mut after_write: Option<&mut dyn FnMut(usize)>,
    ) -> Result<()> {
        let deadline = self.deadline(earliest(requested, Some(rows.deadline())))?;
        let at = deadline.ok_or_else(|| bad("missing finite refinement row deadline"))?;
        let root = self.prefix.root();
        let budget = root.budget();
        ny_core::require_f64_interval_proof_environment()?;
        root.check_box(self.preactivation())?;
        let (input, n) = self.prefix.for_relu(self.relu_op())?;
        if self.preactivation().tensor != input
            || self.preactivation().lo.len() != n
            || self.preactivation().hi.len() != n
            || rows.n_in() != root.lo().len()
            || rows.columns() != rows.targets().len()
            || rows.columns() == 0
        {
            return Err(bad("refinement transaction row/source dimensions disagree"));
        }
        rows.validate_prepared_prefix(&self.prefix, self.op, rows.targets())?;
        let columns = rows.columns();
        let scratch = columns
            .checked_mul(size_of::<RefinementProposal>())
            .and_then(|n| n.checked_add(OWNER_METADATA))
            .ok_or_else(overflow)?;
        let source = rows
            .resident_floor(at)?
            .checked_add(self.resident_bytes()?)
            .ok_or_else(overflow)?;
        refinement_peak(
            budget,
            source,
            scratch,
            "prepared refinement tile visible overlap",
        )?;
        let visits = columns.checked_mul(rows.n_in()).ok_or_else(overflow)?;
        let charged = work(visits, 128)?
            .checked_add(work(columns, 32)?)
            .ok_or_else(overflow)?;
        let mut lease = admit(budget, 0, scratch, charged, Some(at))?;
        let deadline = earliest(Some(at), lease.deadline());
        let at = deadline.ok_or_else(|| bad("missing refinement lease deadline"))?;
        let mut proposals = exact_vec(columns)?;
        checkpoint(budget, Some(at))?;
        lease.mark_materialized().map_err(refusal)?;
        for (column, &target) in rows.targets().iter().enumerate() {
            poll(budget, Some(at), column)?;
            if target >= n {
                return Err(bad("refinement target is outside original tensor"));
            }
            let lower = rows
                .lower(column)
                .ok_or_else(|| bad("missing explicit lower refinement row"))?;
            let upper = rows
                .upper(column)
                .ok_or_else(|| bad("missing explicit upper refinement row"))?;
            let lb = input_row_bound(root, &lower, true, at)?;
            let ub = input_row_bound(root, &upper, false, at)?;
            if !(lb.is_finite() && ub.is_finite() && lb <= ub) {
                return Err(bad("crossed authenticated refinement row pair"));
            }
            let pre = self.preactivation();
            let l = pre.lo[target].max(lb);
            let u = pre.hi[target].min(ub);
            if !(l.is_finite() && u.is_finite() && l <= u) {
                return Err(bad(
                    "nonfinite/crossed authenticated refinement intersection",
                ));
            }
            proposals.push(RefinementProposal {
                target,
                lower: l,
                upper: u,
                old_lower: pre.lo[target],
                old_upper: pre.hi[target],
            });
        }
        checkpoint(budget, Some(at))?;
        // The only writes occur here, after the entire authenticated tile has
        // valid intersections. Undo storage was paid with the proposal slab.
        let pre = self.pre.as_mut().expect("live refinement transaction");
        let mut written = 0;
        let update = (|| -> Result<()> {
            for (i, p) in proposals.iter().enumerate() {
                poll(budget, Some(at), i)?;
                pre.lo[p.target] = p.lower;
                pre.hi[p.target] = p.upper;
                written += 1;
                if let Some(observer) = after_write.as_mut() {
                    observer(i);
                }
            }
            checkpoint(budget, Some(at))?;
            lease.complete().map_err(refusal)
        })();
        if let Err(error) = update {
            // Restoration is bounded by this already-paid tile. It does no
            // floating-point arithmetic, allocation or new numerical work.
            for p in &proposals[..written] {
                pre.lo[p.target] = p.old_lower;
                pre.hi[p.target] = p.old_upper;
            }
            return Err(error);
        }
        pre.deadline = deadline;
        self.deadline = deadline;
        Ok(())
    }
    /// Publish only the tightened producer box, moving its original vectors
    /// and endpoint reservation back out. Coverage belongs to the root manifest.
    pub(in crate::margin_row) fn finish(mut self) -> Result<CertifiedTensorBox> {
        let deadline = self.deadline(None)?;
        checkpoint(self.prefix.root().budget(), deadline)?;
        ny_core::require_f64_interval_proof_environment()?;
        let mut pre = self.pre.take().expect("live refinement transaction");
        pre.deadline = deadline;
        Ok(pre)
    }
}
fn refinement_peak(
    budget: &WorkBudget,
    visible: usize,
    output: usize,
    site: &'static str,
) -> Result<()> {
    let receipt = budget.receipt().map_err(refusal)?;
    let current = receipt
        .live_scratch_bytes
        .checked_add(receipt.live_persistent_bytes)
        .ok_or_else(overflow)?;
    check_local_peak(visible.max(current), output, site)
}
#[cfg(test)]
#[path = "prefix_refinement_tests.rs"]
mod refinement_tests;
