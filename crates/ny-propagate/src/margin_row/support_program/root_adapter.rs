// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Authentic immutable projection into the ordinary numerical root interface.
//!
//! This owner prices the original six gate vectors and domain geometry once.
//! Compiled weights and independent row error planes remain on their actual
//! producer owners. It is an initial unsplit frame, not a descendant/rebake
//! seal or a claim that all ordinary consumers already inherit the budget.

use std::mem::size_of;
use std::sync::Arc;
use std::time::Instant;

use ny_core::{NyError, Result};

use super::construction::PreparedConstructionStats;
use super::ir::{checkpoint, invalid, refusal, RootProductDemand};
use super::publication::PreparedRootSource;
use super::retained::{
    build_prepared_clip_rows, build_prepared_retained, PreparedClipRows, PreparedRetainedRows,
};
use crate::certify::work::{LeaseClass, PersistentReservation, WorkLease, WorkRequest};
use crate::margin_row::backward_interm::BackwardIntermExecutionEvidence;
use crate::margin_row::errlane::ErrLane;
use crate::margin_row::net::{TwinNet, TwinOp};
use crate::margin_row::root::{LayerGates, RetainCfg, RootGates, RootPrecisionExecutionEvidence};
use crate::margin_row::rounding::RoundMode;
use crate::margin_row::{BackwardIntermRuntimePolicy, RootPrecisionRuntimePolicy};

const META: usize = 4096;
const POLL: usize = 256;

/// Completed projection and requested product receipt. Configuration is not
/// execution: the source stats name actual reference/row work, and the legacy
/// tableau/BI counters never claim phases this producer did not run.
#[derive(Clone, Copy, Debug)]
pub(in crate::margin_row) struct PreparedRootAdapterReceipt {
    pub(in crate::margin_row) precision: RootPrecisionRuntimePolicy,
    pub(in crate::margin_row) backward_interm: BackwardIntermRuntimePolicy,
    pub(in crate::margin_row) clipping_requested: bool,
    pub(in crate::margin_row) retained_requested: bool,
    pub(in crate::margin_row) retained_per_layer: usize,
    pub(in crate::margin_row) retained_budget_bytes: usize,
    pub(in crate::margin_row) retained_payload_bytes: usize,
    pub(in crate::margin_row) layer_count: usize,
    pub(in crate::margin_row) final_unstable_rows: usize,
    pub(in crate::margin_row) reference_conv_ops: usize,
    pub(in crate::margin_row) reference: super::PreparedReferenceReceipt,
    pub(in crate::margin_row) source: PreparedConstructionStats,
}

struct AdapterInner {
    source: PreparedRootSource,
    gates: RootGates,
    clip: Option<PreparedClipRows>,
    retained: Option<PreparedRetainedRows>,
    receipt: PreparedRootAdapterReceipt,
    deadline: Instant,
    resident: PersistentReservation,
}

/// Only the genuine completed construction can mint this immutable frame.
/// Clone shares every allocation and its last-owner claim. No mutable frame,
/// raw consuming RootGates or copied numerical facts can escape this owner.
#[derive(Clone)]
pub(in crate::margin_row) struct PreparedRootAdapter(Arc<AdapterInner>);

impl PreparedRootAdapter {
    pub(in crate::margin_row) fn source(&self) -> &PreparedRootSource {
        &self.0.source
    }
    pub(in crate::margin_row) fn gates(&self) -> &RootGates {
        &self.0.gates
    }
    pub(in crate::margin_row) fn net(&self) -> &TwinNet {
        self.source().construction().prefix().root().model().net()
    }
    pub(in crate::margin_row) fn clip_rows(&self) -> Option<&PreparedClipRows> {
        self.0.clip.as_ref()
    }
    pub(in crate::margin_row) fn retained_rows(&self) -> Option<&PreparedRetainedRows> {
        self.0.retained.as_ref()
    }
    pub(in crate::margin_row) fn receipt(&self) -> PreparedRootAdapterReceipt {
        self.0.receipt
    }
    /// Projection allocation only. All source, row-product and sharing-header
    /// claims remain independently live on the same original ledger.
    pub(in crate::margin_row) fn resident_bytes(&self) -> usize {
        self.0.resident.bytes()
    }
    pub(in crate::margin_row) fn same_owner(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
    pub(in crate::margin_row) fn deadline(&self, requested: Option<Instant>) -> Result<Instant> {
        self.source().deadline(Some(
            requested.map_or(self.0.deadline, |d| d.min(self.0.deadline)),
        ))
    }
    pub(in crate::margin_row) fn checkpoint(&self, requested: Option<Instant>) -> Result<()> {
        let at = self.deadline(requested)?;
        self.source().checkpoint(Some(at))?;
        if !self.0.resident.shares_ledger(self.source().budget())
            || self.gates().mode != RoundMode::Outward
            || self.gates().root_precision_policy
                != self.source().construction().prefix().root().policy()
            || self.gates().root_precision_policy != self.receipt().precision
            || self.receipt().reference != self.source().construction().reference_receipt()
            || self.gates().layers.len() != self.receipt().layer_count
            || self.net().trunk_relus.len() != self.receipt().layer_count
        {
            return Err(invalid(
                "prepared root projection original identity/policy changed",
            ));
        }
        if self.receipt().clipping_requested != self.clip_rows().is_some()
            || self.receipt().retained_requested != self.retained_rows().is_some()
        {
            return Err(invalid(
                "prepared root projection lost requested product owner",
            ));
        }
        if let Some(rows) = self.clip_rows() {
            rows.validate_source(self.source(), Some(at))?;
            if rows.len() != self.receipt().final_unstable_rows {
                return Err(invalid(
                    "prepared root projection incomplete final clipping coverage",
                ));
            }
        }
        if let Some(rows) = self.retained_rows() {
            rows.validate_source(self.source(), Some(at))?;
            let cfg = rows.cfg();
            if rows.layers().len() != self.receipt().layer_count
                || cfg.per_layer != self.receipt().retained_per_layer
                || cfg.budget_bytes != self.receipt().retained_budget_bytes
                || rows.payload_bytes() != self.receipt().retained_payload_bytes
                || rows.payload_bytes() > cfg.budget_bytes
            {
                return Err(invalid(
                    "prepared root projection incomplete requested retention",
                ));
            }
        }
        self.source().checkpoint(Some(at))
    }
    pub(in crate::margin_row) fn validate_source(
        &self,
        source: &PreparedRootSource,
        requested: Option<Instant>,
    ) -> Result<()> {
        if !self.source().same_source(source) {
            return Err(invalid("prepared root projection foreign source"));
        }
        self.checkpoint(requested)
    }
}

/// The narrow projection does not silently fulfill enabled historical BI
/// doses, root/tableau binary32, an error-envelope policy, head-group source
/// authentication or descendant generations. Their ordinary adapters must
/// deliver the requested products or select the admitted legacy fallback.
#[allow(clippy::too_many_arguments)]
pub(in crate::margin_row) fn project_prepared_root(
    source: PreparedRootSource,
    expected_precision: RootPrecisionRuntimePolicy,
    bi_policy: BackwardIntermRuntimePolicy,
    clip_requested: bool,
    retain: Option<RetainCfg>,
    requested: Option<Instant>,
) -> Result<PreparedRootAdapter> {
    project_inner(
        source,
        RootProductDemand {
            precision: expected_precision,
            clipping: clip_requested,
            retention: retain,
        },
        bi_policy,
        requested,
        None,
    )
}

// A control observes real completed copies and can cancel their original
// authority. It cannot inject vectors, gates, certificates or a source seal.
#[allow(clippy::too_many_arguments)]
fn project_inner(
    source: PreparedRootSource,
    demand: RootProductDemand,
    bi_policy: BackwardIntermRuntimePolicy,
    requested: Option<Instant>,
    mut after_layer: Option<&mut dyn FnMut(usize)>,
) -> Result<PreparedRootAdapter> {
    let RootProductDemand {
        precision: expected_precision,
        clipping: clip_requested,
        retention: retain,
    } = demand;
    let mut at = source.deadline(requested)?;
    source.checkpoint(Some(at))?;
    let actual = source.construction().prefix().root().policy();
    if expected_precision != actual
        || expected_precision.root_f32
        || expected_precision.tableau_f32
        || expected_precision.err_lane != ErrLane::PerCol
        || bi_policy.enabled
    {
        return Err(invalid(
            "prepared root projection cannot deliver requested precision/BI dose",
        ));
    }
    source.construction().validate_reference(Some(at))?;
    // Optional product constructors have their own complete source-overlap
    // admissions. Neither a numerical copy nor an equal cap can mint them.
    let clip = if clip_requested {
        let rows = build_prepared_clip_rows(&source, Some(at))?;
        at = at.min(rows.deadline(None)?);
        Some(rows)
    } else {
        None
    };
    let retained = if let Some(cfg) = retain {
        let rows = build_prepared_retained(&source, cfg, Some(at))?;
        at = at.min(rows.deadline(None)?);
        Some(rows)
    } else {
        None
    };

    let count = source.construction().prefix().layer_count();
    // Pay geometry bookkeeping before even the length census. The frame
    // admits its actual borrowed gate-pointer registry independently.
    let mut header = admit(
        &source,
        0,
        0,
        work(add(count, source.construction().stats().derived_ops)?, 64)?,
        at,
    )?;
    at = min_lease(at, &header);
    let frame = source.construction().prefix().frame(Some(at))?;
    let root = frame.root();
    let net = root.model().net();
    if count != net.trunk_relus.len() || count != source.construction().layers().len() {
        return Err(invalid(
            "prepared root projection incomplete actual original layers",
        ));
    }
    let n = net.n_in;
    let mut cells = mul(n, 5)?;
    let mut unstable = 0;
    for (li, gate) in frame.layers().iter().enumerate() {
        poll(&source, at, li)?;
        if gate.op != net.trunk_relus[li]
            || net.tsize.get(add(gate.op, 1)?) != Some(&gate.n)
            || [
                gate.l.len(),
                gate.u.len(),
                gate.alpha.len(),
                gate.s.len(),
                gate.c.len(),
                gate.ms.len(),
            ]
            .iter()
            .any(|&length| length != gate.n)
        {
            return Err(invalid(
                "prepared root projection original gate geometry changed",
            ));
        }
        cells = add(cells, mul(gate.n, 6)?)?;
        unstable = add(unstable, gate.unst.len())?;
    }
    let mut conv_ops = 0;
    for (i, op) in net.ops.iter().take(net.i_gemm1).enumerate() {
        poll(&source, at, i)?;
        if matches!(op, TwinOp::Conv(_)) {
            conv_ops = add(conv_ops, 1)?;
        }
    }
    if source.construction().stats().derived_ops != net.i_gemm1
        || source.construction().reference_receipt().completed_conv_ops != conv_ops
    {
        return Err(invalid(
            "prepared root projection incomplete original reference execution",
        ));
    }
    header.complete().map_err(refusal)?;
    drop(header);
    let persistent = add(
        shared_owner_bytes()?,
        add(
            bytes::<LayerGates>(count)?,
            add(bytes::<f64>(cells)?, bytes::<usize>(unstable)?)?,
        )?,
    )?;
    // This admission is atomic with every already-owned source, provider,
    // active frame and concurrent original-budget participant.
    let mut lease = admit(&source, persistent, 0, work(add(cells, unstable)?, 32)?, at)?;
    at = min_lease(at, &lease);
    let mut layers = exact_vec(count)?;
    for (li, gate) in frame.layers().iter().enumerate() {
        poll(&source, at, li)?;
        layers.push(LayerGates {
            op: gate.op,
            n: gate.n,
            l: copy_vec(&gate.l, &source, at)?,
            u: copy_vec(&gate.u, &source, at)?,
            alpha: copy_vec(&gate.alpha, &source, at)?,
            s: copy_vec(&gate.s, &source, at)?,
            c: copy_vec(&gate.c, &source, at)?,
            ms: copy_vec(&gate.ms, &source, at)?,
            unst: copy_vec(&gate.unst, &source, at)?,
            // The explicit authenticated provider carries Al/Au and E/eb.
            // Producing M/D here would lose the independent affine constants.
            clip_rows: None,
        });
        if let Some(observe) = after_layer.as_mut() {
            observe(li);
        }
        source.checkpoint(Some(at))?;
    }
    let gates = RootGates {
        mode: RoundMode::Outward,
        lo: copy_vec(root.lo(), &source, at)?,
        hi: copy_vec(root.hi(), &source, at)?,
        mid: copy_vec(root.mid(), &source, at)?,
        rad: copy_vec(root.rad(), &source, at)?,
        xabs: copy_vec(root.xabs(), &source, at)?,
        layers,
        backward_interm_execution: BackwardIntermExecutionEvidence::from_policy(bi_policy),
        root_precision_policy: expected_precision,
        // Reference work is retained separately; no tableau kernels ran.
        root_precision_execution: RootPrecisionExecutionEvidence::completed(
            expected_precision,
            RoundMode::Outward,
            false,
            false,
            0,
            0,
        ),
        spec_alpha_policy: None,
        head_alpha: None,
        root_eval_memo: crate::margin_row::bab::RootEvalMemo::default(),
        head_group_source: None,
    };
    let receipt = PreparedRootAdapterReceipt {
        precision: expected_precision,
        backward_interm: bi_policy,
        clipping_requested: clip_requested,
        retained_requested: retain.is_some(),
        retained_per_layer: retain.map_or(0, |cfg| cfg.per_layer),
        retained_budget_bytes: retain.map_or(0, |cfg| cfg.budget_bytes),
        retained_payload_bytes: retained
            .as_ref()
            .map_or(0, PreparedRetainedRows::payload_bytes),
        layer_count: count,
        final_unstable_rows: unstable,
        reference_conv_ops: conv_ops,
        reference: source.construction().reference_receipt(),
        source: *source.construction().stats(),
    };
    drop(frame);
    source.checkpoint(Some(at))?;
    lease.mark_materialized().map_err(refusal)?;
    let resident = lease.commit_persistent().map_err(refusal)?;
    let result = PreparedRootAdapter(Arc::new(AdapterInner {
        source,
        gates,
        clip,
        retained,
        receipt,
        deadline: at,
        resident,
    }));
    result.checkpoint(Some(at))?;
    Ok(result)
}

fn copy_vec<T: Copy>(values: &[T], source: &PreparedRootSource, at: Instant) -> Result<Vec<T>> {
    source.checkpoint(Some(at))?;
    let mut copied = exact_vec(values.len())?;
    for (i, &value) in values.iter().enumerate() {
        poll(source, at, i)?;
        copied.push(value);
    }
    source.checkpoint(Some(at))?;
    Ok(copied)
}
fn exact_vec<T>(n: usize) -> Result<Vec<T>> {
    let mut copied = Vec::new();
    copied
        .try_reserve_exact(n)
        .map_err(|_| invalid("prepared root projection allocation refused"))?;
    if copied.capacity() != n {
        return Err(invalid(
            "prepared root projection actual capacity exceeds paid prediction",
        ));
    }
    Ok(copied)
}
fn admit(
    source: &PreparedRootSource,
    persistent: usize,
    scratch: usize,
    work: u64,
    at: Instant,
) -> Result<WorkLease> {
    source.checkpoint(Some(at))?;
    let lease = source
        .budget()
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                persistent_bytes: persistent,
                scratch_bytes: scratch,
                work,
                ..WorkRequest::default()
            },
        )
        .map_err(refusal)?;
    checkpoint(source.budget(), Some(min_lease(at, &lease)))?;
    Ok(lease)
}
fn poll(source: &PreparedRootSource, at: Instant, i: usize) -> Result<()> {
    if i.is_multiple_of(POLL) {
        checkpoint(source.budget(), Some(at))?;
    }
    Ok(())
}
fn min_lease(at: Instant, lease: &WorkLease) -> Instant {
    lease.deadline().map_or(at, |d| d.min(at))
}
fn shared_owner_bytes() -> Result<usize> {
    add(
        add(size_of::<AdapterInner>(), mul(2, size_of::<usize>())?)?,
        META,
    )
}
fn bytes<T>(n: usize) -> Result<usize> {
    mul(n, size_of::<T>())
}
fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b).ok_or_else(overflow)
}
fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b).ok_or_else(overflow)
}
fn work(n: usize, cost: usize) -> Result<u64> {
    u64::try_from(add(mul(n, cost)?, META)?).map_err(|_| overflow())
}
fn overflow() -> NyError {
    invalid("prepared root projection capacity/work overflow")
}

#[cfg(test)]
#[path = "root_adapter_tests.rs"]
mod tests;
