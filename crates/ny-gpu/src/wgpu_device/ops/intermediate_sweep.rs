// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Default-dark, authoritative WGPU implementation of the typed
//! `GpuIntermediateSweep` contract.
//!
//! The first admitted topology is deliberately narrow: a dense, single-edge
//! Unary/Identity chain.  Every requested row lives in one transaction-wide
//! carrier.  Rows injected below the output start as exact zero rows and are
//! reset, in place, to an exact identity row after the preceding unary fold.
//! The reset also clears every coefficient-error, bias-error, and taint-word
//! lane for that row.  Thus work performed while a row is dormant cannot enter
//! its eventual target bound, while all targets still share one resident walk.
//!
//! `peak_device_bytes` is the checked ceiling for every buffer retained by this
//! [`WgpuDevice`] (pool and all owned plan/weight caches), plus the complete
//! sweep-owned live working set. Admission snapshots that retained set and
//! reserves the ceiling while holding one exact-device serialization guard;
//! an owned cache without checked byte accounting is a capability refusal, not
//! an omitted charge. This is deliberately not a claim about physical/global
//! VRAM: the legacy public raw [`WgpuDevice::device`] accessor permits callers
//! to create buffers that this wrapper neither owns nor serializes. Closing or
//! wrapping that capability leak is follow-up work. Any resulting WGPU/OOM
//! error still invalidates the whole transaction and publishes no result.

use std::cell::{Cell, RefCell};
use std::mem::size_of;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ny_core::{
    GpuBackwardOp, GpuCrownLayer, GpuIntermediateSweepReceipt, GpuIntermediateSweepRequest,
    GpuIntermediateSweepResult, GpuIntermediateTargetResult, NyError, Result,
};

use super::super::WgpuDevice;
use super::crown_backward_sound_resident::ResidentFoldPlan;

const VALIDATION_POLL_STRIDE: usize = 4096;
/// Backend-local admission ceiling for the first resident chain scheduler.
/// Besides bounding host validation, this makes the per-layer mapped-uniform
/// allowance below a finite pre-dispatch quantity.
pub(super) const MAX_RESIDENT_CHAIN_OPS: usize = 256;
pub(super) const MAX_RESIDENT_CHAIN_LAYERS: usize = 64;
/// Conservative numerical-buffer allowance for every layer's mapped uniforms
/// and queued parameter uploads. The audited resident fold creates only small
/// fixed-layout records; 2 MiB/layer deliberately dominates their total while
/// keeping the ceiling derived from the admitted layer count rather than a
/// fixed request-independent constant.
pub(super) const UNIFORM_UPLOAD_BYTES_PER_LAYER: usize = 2 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SweepReset {
    pub(super) carrier_row: usize,
    pub(super) coordinate: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct SweepBoundary {
    pub(super) dim: usize,
    pub(super) resets: Vec<SweepReset>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct SweepWorkReceipt {
    pub(super) dispatches: usize,
    pub(super) host_to_device_bytes: usize,
    pub(super) device_to_host_bytes: usize,
    pub(super) readbacks: usize,
    pub(super) submits: usize,
    pub(super) synchronizations: usize,
}

struct ActiveSweep {
    boundaries: Vec<Option<SweepBoundary>>,
    next_boundary: usize,
    work: SweepWorkReceipt,
    /// Submissions recorded since the last SUCCESSFUL queue-idle
    /// synchronization (a bounded `poll(Wait)` that completed). Zero after at
    /// least one submit is a proof that nothing this sweep enqueued can still
    /// be in flight on the device.
    submits_since_last_sync: usize,
}

/// How an accepted sweep exited after its first submit, recorded by the FIRST
/// cause observed (later, less precise observers on the same unwind never
/// overwrite it). The reservation release maps each state to
/// restore / bounded-drain / poison.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PostSubmitAbort {
    /// No post-submit abort: the scope finished, or nothing was submitted.
    None,
    /// The sweep aborted, but a successful queue-idle synchronization followed
    /// its last submit: nothing can still be in flight, the ledger's
    /// accounting is truthful, and the reservation releases normally.
    ProvenDrained,
    /// The call-local deadline expired while a submission may still have been
    /// in flight (`note_post_submit_abort`). A bounded blocking drain of the
    /// already-submitted work restores the ledger; only a drain failure — a
    /// genuinely wedged device — falls back to the poison path.
    DeadlineInFlight,
    /// Any other abort with a possibly in-flight submission (validation
    /// fault, internal error, unwind). Fail closed: the device state is
    /// suspect and the ledger is permanently poisoned, exactly as before.
    UnknownInFlight,
}

thread_local! {
    static ACTIVE_SWEEP: RefCell<Option<ActiveSweep>> = const { RefCell::new(None) };
    /// Set when an accepted sweep exits after at least one submit without
    /// proving its final bounded readback/finish boundary. The reservation
    /// release classifies the abort (see [`PostSubmitAbort`]) instead of
    /// unconditionally claiming potentially in-flight allocations are gone.
    static SWEEP_POST_SUBMIT_ABORTED: Cell<PostSubmitAbort> =
        const { Cell::new(PostSubmitAbort::None) };
}

fn record_post_submit_abort(state: PostSubmitAbort) {
    SWEEP_POST_SUBMIT_ABORTED.with(|aborted| {
        if aborted.get() == PostSubmitAbort::None {
            aborted.set(state);
        }
    });
}

/// Called by the shared readback helpers when the call-local deadline gives up
/// on an in-flight submission. First-cause-wins: the later, generic
/// [`SweepScope`] drop on the same unwind cannot downgrade this to
/// [`PostSubmitAbort::UnknownInFlight`].
pub(in crate::wgpu_device) fn note_post_submit_abort() {
    let in_flight = ACTIVE_SWEEP.with(|slot| {
        slot.borrow()
            .as_ref()
            .map(|active| active.submits_since_last_sync > 0)
    });
    match in_flight {
        None => {}
        Some(true) => record_post_submit_abort(PostSubmitAbort::DeadlineInFlight),
        Some(false) => record_post_submit_abort(PostSubmitAbort::ProvenDrained),
    }
}

/// RAII ownership of the one active sweep on this calling thread.
pub(super) struct SweepScope {
    armed: bool,
}

impl SweepScope {
    fn arm(boundaries: Vec<SweepBoundary>) -> Result<Self> {
        ACTIVE_SWEEP.with(|slot| {
            let mut slot = slot.borrow_mut();
            if slot.is_some() {
                return Err(NyError::InternalError(
                    "nested WGPU intermediate sweep scope".into(),
                ));
            }
            *slot = Some(ActiveSweep {
                boundaries: boundaries.into_iter().map(Some).collect(),
                // Boundary zero was encoded into the host seed.
                next_boundary: 1,
                work: SweepWorkReceipt::default(),
                submits_since_last_sync: 0,
            });
            Ok(Self { armed: true })
        })
    }

    /// Arm work accounting for the DAG route. Identity injections are encoded
    /// by the DAG kernels rather than consumed from resident layer boundaries,
    /// so this scope intentionally owns no scripted boundary slots.
    pub(super) fn arm_dag() -> Result<Self> {
        ACTIVE_SWEEP.with(|slot| {
            let mut slot = slot.borrow_mut();
            if slot.is_some() {
                return Err(NyError::InternalError(
                    "nested WGPU intermediate sweep scope".into(),
                ));
            }
            *slot = Some(ActiveSweep {
                boundaries: Vec::new(),
                next_boundary: 0,
                work: SweepWorkReceipt::default(),
                submits_since_last_sync: 0,
            });
            Ok(Self { armed: true })
        })
    }

    pub(super) fn finish(mut self) -> Result<SweepWorkReceipt> {
        let active = ACTIVE_SWEEP
            .with(|slot| slot.borrow_mut().take())
            .ok_or_else(|| {
                NyError::InternalError("WGPU intermediate sweep scope disappeared".into())
            })?;
        self.armed = false;
        if active.next_boundary != active.boundaries.len() {
            return Err(NyError::InternalError(format!(
                "WGPU intermediate sweep consumed {} of {} resident boundaries",
                active.next_boundary,
                active.boundaries.len()
            )));
        }
        Ok(active.work)
    }
}

impl Drop for SweepScope {
    fn drop(&mut self) {
        if self.armed {
            ACTIVE_SWEEP.with(|slot| {
                if let Some(active) = slot.borrow_mut().take() {
                    if active.work.submits > 0 {
                        record_post_submit_abort(if active.submits_since_last_sync == 0 {
                            PostSubmitAbort::ProvenDrained
                        } else {
                            PostSubmitAbort::UnknownInFlight
                        });
                    }
                }
            });
        }
    }
}

pub(super) fn take_boundary(boundary: usize, dim: usize) -> Result<Option<SweepBoundary>> {
    ACTIVE_SWEEP.with(|slot| {
        let mut slot = slot.borrow_mut();
        let Some(active) = slot.as_mut() else {
            return Ok(None);
        };
        // The DAG route owns injection/reset dispatches at slot boundaries and
        // arms this scope only for work accounting. Its empty boundary script
        // is therefore an explicit no-op for resident layer callbacks.
        if active.boundaries.is_empty() {
            return Ok(None);
        }
        if boundary != active.next_boundary {
            return Err(NyError::InternalError(format!(
                "WGPU intermediate sweep reached boundary {boundary}, expected {}",
                active.next_boundary
            )));
        }
        let value = active
            .boundaries
            .get_mut(boundary)
            .and_then(Option::take)
            .ok_or_else(|| {
                NyError::InternalError(format!(
                    "WGPU intermediate sweep boundary {boundary} was already consumed"
                ))
            })?;
        if value.dim != dim {
            return Err(NyError::InternalError(format!(
                "WGPU intermediate sweep boundary {boundary} dimension {} != resident {dim}",
                value.dim
            )));
        }
        active.next_boundary += 1;
        Ok(Some(value))
    })
}

fn note(f: impl FnOnce(&mut SweepWorkReceipt)) {
    ACTIVE_SWEEP.with(|slot| {
        if let Some(active) = slot.borrow_mut().as_mut() {
            f(&mut active.work);
        }
    });
}

pub(super) fn note_dispatches(count: usize) {
    note(|work| work.dispatches = work.dispatches.saturating_add(count));
}

pub(super) fn note_host_to_device(bytes: usize) {
    note(|work| {
        work.host_to_device_bytes = work.host_to_device_bytes.saturating_add(bytes);
    });
}

pub(in crate::wgpu_device) fn note_device_to_host(
    bytes: usize,
    readbacks: usize,
    synchronizations: usize,
) {
    ACTIVE_SWEEP.with(|slot| {
        if let Some(active) = slot.borrow_mut().as_mut() {
            let work = &mut active.work;
            work.device_to_host_bytes = work.device_to_host_bytes.saturating_add(bytes);
            work.readbacks = work.readbacks.saturating_add(readbacks);
            work.synchronizations = work.synchronizations.saturating_add(synchronizations);
            if synchronizations > 0 {
                // Every noted synchronization is a SUCCESSFUL bounded
                // poll(Wait), which proves the queue was idle: nothing
                // submitted earlier can still be in flight.
                active.submits_since_last_sync = 0;
            }
        }
    });
}

pub(in crate::wgpu_device) fn note_submits(count: usize) {
    ACTIVE_SWEEP.with(|slot| {
        if let Some(active) = slot.borrow_mut().as_mut() {
            active.work.submits = active.work.submits.saturating_add(count);
            active.submits_since_last_sync = active.submits_since_last_sync.saturating_add(count);
        }
    });
}

struct PreparedSweep {
    layers: Vec<GpuCrownLayer>,
    lower_seed: Vec<f32>,
    upper_seed: Vec<f32>,
    boundaries: Vec<SweepBoundary>,
}

/// EXACT text of the ledger-poison ERROR line. The design's §7 kill criterion
/// (`SWEEP_POST_SUBMIT_KILL_LINE`, pinned against this source file by
/// ny-propagate's `sweep_kill_line_matches_the_sweep_source`) greps M2/M3 run
/// logs for its "exited after submission without a final drain" substring.
/// Never reword it; the last-resort poison path must stay grep-able.
pub(super) const POST_SUBMIT_POISON_LINE: &str =
    "WGPU intermediate sweep exited after submission without a final drain; memory ledger left permanently fail-closed";

/// Bounded blocking drain authority for a deadline-shaped post-submit abort.
/// Implemented by [`WgpuDevice`]; unit tests substitute scripted outcomes.
pub(super) trait SweepAbortDrain {
    /// Block until every already-submitted command buffer retires, or fail.
    fn drain_after_post_submit_abort(&self) -> Result<()>;
}

/// Upper bound on the post-abort drain. The wait covers only work this sweep
/// already submitted — the aborting thread still holds the exclusive GPU
/// transaction, so nothing new can be enqueued behind it — i.e. at most one
/// transaction's chunk of GPU time (measured ~1s/wave on cifar100). Ten
/// seconds is far above any healthy chunk while still small against an
/// instance budget; exceeding it means the device is genuinely wedged and the
/// poison path is correct.
const POST_SUBMIT_ABORT_DRAIN_TIMEOUT: Duration = Duration::from_secs(10);

impl SweepAbortDrain for WgpuDevice {
    fn drain_after_post_submit_abort(&self) -> Result<()> {
        // Deliberately NOT `poll_readback`: that helper honours the (already
        // expired) call-local deadline — the exact mechanism that abandoned
        // the readback in the first place. This wait is bounded by the
        // constant alone.
        if let Err(error) = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(POST_SUBMIT_ABORT_DRAIN_TIMEOUT),
        }) {
            return Err(NyError::InternalError(format!(
                "post-submit abort drain did not reach queue idle: {error}"
            )));
        }
        Ok(())
    }
}

/// Call-local reservation against the retained WGPU device. The outer checked
/// transaction holds the GPU serialization mutex first; this RAII entry is the
/// second lock in the invariant shared with `clear_crown_working_set`.
pub(super) struct SweepMemoryReservation<'a> {
    ledger: &'a std::sync::Mutex<usize>,
    /// Drain authority for a deadline-shaped post-submit abort. `None` (bare
    /// ledgers in unit tests) cannot prove queue idle, so such an abort keeps
    /// the fail-closed poison path.
    drain: Option<&'a dyn SweepAbortDrain>,
    bytes: usize,
    armed: bool,
}

impl SweepMemoryReservation<'_> {
    pub(super) fn release(&mut self) -> Result<()> {
        if !self.armed {
            return Ok(());
        }
        let mut reserved = self.ledger.lock().map_err(|err| {
            NyError::InternalError(format!(
                "intermediate sweep reservation release lock poisoned: {err}"
            ))
        })?;
        match SWEEP_POST_SUBMIT_ABORTED.with(|flag| flag.replace(PostSubmitAbort::None)) {
            PostSubmitAbort::None => {}
            PostSubmitAbort::ProvenDrained => {
                // A successful queue-idle synchronization followed the sweep's
                // last submit, so the abort left nothing in flight; release
                // normally below so later sweeps stay available.
                tracing::warn!(
                    "WGPU intermediate sweep aborted after submission with a proven \
                     queue-idle synchronization; memory ledger restored"
                );
            }
            PostSubmitAbort::DeadlineInFlight => {
                let drained = match self.drain {
                    Some(device) => device.drain_after_post_submit_abort(),
                    None => Err(NyError::InternalError(
                        "no drain authority is attached to this sweep reservation".into(),
                    )),
                };
                match drained {
                    Ok(()) => {
                        // Blocking on the already-submitted work (bounded by
                        // one chunk's GPU time) reached queue idle: in-flight
                        // allocations are retired and the ledger's accounting
                        // is truthful again; release normally below.
                        tracing::warn!(
                            "WGPU intermediate sweep deadline abort drained to queue idle; \
                             memory ledger restored"
                        );
                    }
                    Err(drain_error) => {
                        *reserved = usize::MAX;
                        self.armed = false;
                        tracing::error!(
                            %drain_error,
                            "WGPU intermediate sweep post-submit drain failed; treating the device as wedged"
                        );
                        tracing::error!("{}", POST_SUBMIT_POISON_LINE);
                        return Ok(());
                    }
                }
            }
            PostSubmitAbort::UnknownInFlight => {
                *reserved = usize::MAX;
                self.armed = false;
                tracing::error!("{}", POST_SUBMIT_POISON_LINE);
                return Ok(());
            }
        }
        if *reserved != self.bytes {
            let actual = *reserved;
            *reserved = usize::MAX;
            self.armed = false;
            return Err(NyError::InternalError(format!(
                "intermediate sweep reservation ledger mismatch: expected {}, found {actual}; \
                 device left fail-closed",
                self.bytes
            )));
        }
        *reserved = 0;
        self.armed = false;
        Ok(())
    }
}

impl Drop for SweepMemoryReservation<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.release() {
            // Drop cannot propagate, but this path is observable and the
            // ledger remains unavailable. Successful publication calls
            // `release` explicitly so a mismatch becomes the transaction Err.
            tracing::error!(%error, "intermediate sweep reservation release failed");
        }
    }
}

pub(super) fn deadline_check(deadline: Instant, context: &str) -> Result<()> {
    if Instant::now() >= deadline {
        Err(NyError::DeadlineExceeded(format!(
            "WGPU intermediate sweep deadline exceeded {context}"
        )))
    } else {
        Ok(())
    }
}

pub(super) fn capacity_decline(error: NyError) -> Result<Option<GpuIntermediateSweepResult>> {
    match error {
        NyError::UnsupportedOp(_) | NyError::GpuBatchCapacityExceeded { .. } => Ok(None),
        other => Err(other),
    }
}

fn supported_unary(layer: &GpuCrownLayer) -> bool {
    matches!(
        layer,
        GpuCrownLayer::Linear { .. }
            | GpuCrownLayer::Activation { .. }
            | GpuCrownLayer::Conv2d { .. }
    )
}

/// Produce the single-carrier schedule without touching the device.
fn prepare_chain(request: &GpuIntermediateSweepRequest<'_>) -> Result<Option<PreparedSweep>> {
    let plan = request.plan;
    let input_slot = plan.input_slot.index();
    // This slice is intentionally a dense single-edge chain, not a converging
    // DAG disguised as a unary stream.
    if plan.ops_backward.len() != input_slot || plan.ops_backward.len() > MAX_RESIDENT_CHAIN_OPS {
        return Ok(None);
    }
    if plan
        .ops_backward
        .iter()
        .any(|op| matches!(op, GpuBackwardOp::Add { .. } | GpuBackwardOp::Sub { .. }))
    {
        return Ok(None);
    }

    let mut layers = Vec::new();
    let mut unary_before_slot = vec![0usize; plan.slot_dims.len()];
    for (slot, op) in plan.ops_backward.iter().enumerate() {
        if slot.is_multiple_of(VALIDATION_POLL_STRIDE) {
            deadline_check(request.deadline, "while preparing the chain")?;
        }
        unary_before_slot[slot] = layers.len();
        match op {
            GpuBackwardOp::Unary {
                output,
                input,
                layer,
            } if output.index() == slot && input.index() == slot + 1 => {
                if !supported_unary(layer) {
                    return Ok(None);
                }
                layers.push((**layer).clone());
            }
            GpuBackwardOp::Identity { output, input }
                if output.index() == slot && input.index() == slot + 1 => {}
            GpuBackwardOp::Add { .. } | GpuBackwardOp::Sub { .. } => return Ok(None),
            _ => return Ok(None),
        }
    }
    unary_before_slot[input_slot] = layers.len();
    // The resident engine requires at least one arithmetic fold. Pure identity
    // plans remain a clean pre-dispatch decline in this first slice.
    if layers.is_empty() {
        return Ok(None);
    }
    if layers.len() > MAX_RESIDENT_CHAIN_LAYERS {
        return Ok(None);
    }

    let mut boundaries = vec![SweepBoundary::default(); layers.len() + 1];
    for (index, injection) in plan.injections.iter().enumerate() {
        if index.is_multiple_of(VALIDATION_POLL_STRIDE) {
            deadline_check(request.deadline, "while scheduling injections")?;
        }
        let slot = injection.slot.index();
        let boundary = unary_before_slot[slot];
        let dim = plan.slot_dims[slot];
        let scheduled = &mut boundaries[boundary];
        if scheduled.dim == 0 {
            scheduled.dim = dim;
        } else if scheduled.dim != dim {
            return Ok(None);
        }
        for (row, &coordinate) in injection.selected_rows.iter().enumerate() {
            scheduled.resets.push(SweepReset {
                carrier_row: injection.row_offset + row,
                coordinate: coordinate as usize,
            });
        }
    }
    // Empty boundaries still carry the resident frontier dimension so the
    // scheduler can prove that no layer was silently skipped.
    let mut current_dim = plan.slot_dims[0];
    if boundaries[0].dim == 0 {
        boundaries[0].dim = current_dim;
    }
    let mut layer_index = 0usize;
    for op in plan.ops_backward.iter() {
        if let GpuBackwardOp::Unary { layer, .. } = op {
            current_dim = match layer.as_ref() {
                GpuCrownLayer::Linear { in_features, .. } => *in_features,
                GpuCrownLayer::Activation { num_neurons, .. } => *num_neurons,
                GpuCrownLayer::Conv2d {
                    in_channels,
                    in_h,
                    in_w,
                    ..
                } => in_channels
                    .checked_mul(*in_h)
                    .and_then(|n| n.checked_mul(*in_w))
                    .ok_or_else(|| {
                        NyError::InvalidSpec(
                            "WGPU intermediate sweep conv frontier dimension overflow".into(),
                        )
                    })?,
                _ => unreachable!("supported_unary filtered the layer"),
            };
            layer_index += 1;
            if boundaries[layer_index].dim == 0 {
                boundaries[layer_index].dim = current_dim;
            }
        }
    }

    let output_dim = plan.slot_dims[0];
    let seed_elems = plan
        .total_rows
        .checked_mul(output_dim)
        .ok_or_else(|| NyError::InvalidSpec("WGPU intermediate sweep seed size overflow".into()))?;
    let mut lower_seed = vec![0.0f32; seed_elems];
    let mut upper_seed = vec![0.0f32; seed_elems];
    for reset in &boundaries[0].resets {
        let index = reset
            .carrier_row
            .checked_mul(output_dim)
            .and_then(|base| base.checked_add(reset.coordinate))
            .ok_or_else(|| {
                NyError::InvalidSpec("WGPU intermediate sweep seed index overflow".into())
            })?;
        let (Some(lower), Some(upper)) = (lower_seed.get_mut(index), upper_seed.get_mut(index))
        else {
            return Err(NyError::InvalidSpec(
                "WGPU intermediate sweep seed index is out of range".into(),
            ));
        };
        *lower = 1.0;
        *upper = 1.0;
    }

    Ok(Some(PreparedSweep {
        layers,
        lower_seed,
        upper_seed,
        boundaries,
    }))
}

fn checked_add(a: usize, b: usize, label: &str) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| NyError::InvalidSpec(format!("{label} byte count overflow")))
}

fn checked_mul(a: usize, b: usize, label: &str) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| NyError::InvalidSpec(format!("{label} byte count overflow")))
}

fn layer_weight_len(layer: &GpuCrownLayer) -> usize {
    match layer {
        GpuCrownLayer::Linear { weight, .. } => weight.len(),
        GpuCrownLayer::Conv2d { weight_col, .. } => weight_col.len(),
        _ => 0,
    }
}

pub(super) fn retained_device_bytes(device: &WgpuDevice) -> Result<usize> {
    [
        device.buffer_pool_retained_bytes()?,
        device.crown_plan_cache_bytes()?,
        device.conv_transpose_plan_cache_bytes()?,
        device.resident_weight_cache_bytes()?,
        device.point_vjp_resident_cache_bytes()?,
    ]
    .into_iter()
    .try_fold(0usize, |total, bytes| {
        total.checked_add(bytes).ok_or_else(|| {
            NyError::InternalError("WgpuDevice retained-cache byte count overflow".into())
        })
    })
}

/// Conservative, audited logical-buffer ceiling for all [`WgpuDevice`]-owned
/// retained buffers plus the complete request-owned resident/concretize
/// working sets. H2D payload traffic is reported separately; allocator/driver
/// physical residency and raw-accessor allocations are outside this receipt's
/// explicitly documented authority boundary.
fn sweep_peak_ceiling(
    request: &GpuIntermediateSweepRequest<'_>,
    prepared: &PreparedSweep,
    fold: ResidentFoldPlan,
    retained_device_bytes: usize,
) -> Result<usize> {
    let rows = request.plan.total_rows;
    let f32b = size_of::<f32>();
    let total_weights = prepared
        .layers
        .iter()
        .map(layer_weight_len)
        .try_fold(0usize, |total, weight| total.checked_add(weight))
        .ok_or_else(|| NyError::InvalidSpec("resident weight footprint overflow".into()))?;
    let mut max_reset_payload = 0usize;
    let mut total_reset_upload_bytes = 0usize;
    for boundary in &prepared.boundaries {
        let per_row = boundary
            .dim
            .checked_mul(8)
            .and_then(|value| value.checked_add(5))
            .ok_or_else(|| NyError::InvalidSpec("sweep reset payload overflow".into()))?;
        let payload = boundary
            .resets
            .len()
            .checked_mul(per_row)
            .ok_or_else(|| NyError::InvalidSpec("sweep reset payload overflow".into()))?;
        max_reset_payload = max_reset_payload.max(payload);
        total_reset_upload_bytes = checked_add(
            total_reset_upload_bytes,
            checked_mul(payload, f32b, "sweep reset upload bytes")?,
            "aggregate sweep reset uploads",
        )?;
    }
    let queued_fold_upload_bytes = usize::try_from(
        super::crown_backward_sound_resident::resident_fold_staging_capacity(
            &prepared.layers,
            fold.n_domains,
        )?,
    )
    .map_err(|_| NyError::InvalidSpec("resident fold staging exceeds usize".into()))?;

    // Value/error frontiers, scratch, conv scratch, activation state and final
    // staging. The taint route mirrors eleven coefficient-sized streams.
    let resident_elems = checked_add(
        checked_mul(fold.a_elems, 32, "resident coefficient workspaces")?,
        checked_add(
            checked_mul(fold.max_gemm_out, 6, "resident GEMM workspaces")?,
            checked_add(
                checked_mul(fold.slope_dim, 7, "resident activation workspaces")?,
                checked_add(
                    checked_mul(rows, fold.final_dim.saturating_mul(4), "resident staging")?,
                    checked_add(
                        checked_mul(rows, 16, "resident row workspaces")?,
                        checked_add(
                            checked_mul(fold.max_dim, 4, "resident vector workspaces")?,
                            checked_add(
                                checked_mul(total_weights, 3, "resident weights")?,
                                max_reset_payload,
                                "resident reset staging",
                            )?,
                            "resident weights and vectors",
                        )?,
                        "resident row and vector workspaces",
                    )?,
                    "resident staging workspaces",
                )?,
                "resident activation workspaces",
            )?,
            "resident GEMM workspaces",
        )?,
        "resident workspaces",
    )?;
    let resident_buffer_bytes = checked_add(
        checked_mul(resident_elems, f32b, "resident")?,
        checked_mul(
            prepared.layers.len(),
            UNIFORM_UPLOAD_BYTES_PER_LAYER,
            "resident per-layer uniform uploads",
        )?,
        "resident buffers and uniform uploads",
    )?;

    let coeff = checked_mul(rows, fold.final_dim, "concretize coefficients")?;
    let concretize_elems = checked_add(
        checked_mul(coeff, 4, "concretize coefficient streams")?,
        checked_add(
            checked_mul(fold.final_dim, 2, "concretize box")?,
            checked_mul(rows, 6, "concretize row streams")?,
            "concretize streams",
        )?,
        "concretize buffers",
    )?;
    let concretize_buffer_bytes = checked_add(
        checked_mul(concretize_elems, f32b, "concretize")?,
        4096,
        "concretize uniforms",
    )?;
    // `Queue::write_buffer` may retain an internal upload allocation until the
    // consuming submit completes. Charge a second complete explicit-buffer
    // footprint for both phases. This intentionally overcounts output-only
    // buffers and scratch that are never uploaded, but prevents a queued H2D
    // payload from living outside the caller's numerical-buffer ceiling.
    let resident_buffers_and_initial_uploads = checked_mul(
        resident_buffer_bytes,
        2,
        "resident buffers plus initial queued upload staging",
    )?;
    let resident_work_bytes = checked_add(
        resident_buffers_and_initial_uploads,
        checked_add(
            queued_fold_upload_bytes,
            total_reset_upload_bytes,
            "queued fold and reset uploads",
        )?,
        "resident buffers plus every queued upload",
    )?;
    let concretize_work_bytes = checked_mul(
        concretize_buffer_bytes,
        2,
        "concretize buffers plus queued upload staging",
    )?;
    // Request weights become retained cache entries during the walk (Raw+Abs;
    // use 3x to conservatively cover a transposed form created by adjacent
    // resident machinery). Existing retained weights stay live throughout.
    let new_weight_bytes = checked_mul(total_weights, 3 * f32b, "new retained weights")?;
    let persistent_after_walk = checked_add(
        retained_device_bytes,
        new_weight_bytes,
        "retained device buffers after sweep walk",
    )?;
    let resident_peak = checked_add(
        retained_device_bytes,
        resident_work_bytes,
        "resident peak with retained device buffers",
    )?;
    let concretize_peak = checked_add(
        persistent_after_walk,
        concretize_work_bytes,
        "concretize peak with retained cache",
    )?;
    Ok(resident_peak.max(concretize_peak).max(f32b))
}

fn memory_cap_admits(peak: usize, request_cap: usize, backend_cap: usize) -> bool {
    peak <= request_cap.min(backend_cap)
}

fn reserve_sweep_ledger<'a>(
    ledger: &'a std::sync::Mutex<usize>,
    drain: Option<&'a dyn SweepAbortDrain>,
    request_cap: usize,
    backend_cap: usize,
    peak: usize,
) -> Result<Option<SweepMemoryReservation<'a>>> {
    if !memory_cap_admits(peak, request_cap, backend_cap) {
        return Ok(None);
    }
    let mut reserved = ledger.lock().map_err(|err| {
        NyError::InternalError(format!(
            "intermediate sweep reservation lock poisoned: {err}"
        ))
    })?;
    if *reserved != 0 {
        return Ok(None);
    }
    *reserved = peak;
    SWEEP_POST_SUBMIT_ABORTED.with(|flag| flag.set(PostSubmitAbort::None));
    Ok(Some(SweepMemoryReservation {
        ledger,
        drain,
        bytes: peak,
        armed: true,
    }))
}

impl WgpuDevice {
    pub(super) fn reserve_intermediate_sweep_memory(
        &self,
        request_cap: usize,
        peak: usize,
    ) -> Result<Option<SweepMemoryReservation<'_>>> {
        let backend_cap = super::crown_memory_estimate::gpu_memory_budget_bytes();
        reserve_sweep_ledger(
            &self.intermediate_sweep_reserved_bytes,
            Some(self as &dyn SweepAbortDrain),
            request_cap,
            backend_cap,
            peak,
        )
    }

    pub(super) fn provides_intermediate_sweep(&self) -> bool {
        // Default-dark admission lives at the production caller's typed policy
        // seam. This backend predicate contributes no second ambient switch;
        // v1 is authorized only by the complete IEEE five-rung receipt. The
        // separately modeled charged-flush lane must earn its own sweep proof.
        self.sound_gpu_authority_cached()
    }

    pub(super) fn require_intermediate_sweep_authority(&self) -> Result<()> {
        if self.provides_intermediate_sweep() {
            Ok(())
        } else {
            Err(NyError::UnsupportedOp(
                "WGPU intermediate sweep retained-device verdict authority was lost; \
                 discarding the whole result"
                    .into(),
            ))
        }
    }

    pub(super) fn run_intermediate_sweep(
        &self,
        request: &GpuIntermediateSweepRequest<'_>,
    ) -> Result<Option<GpuIntermediateSweepResult>> {
        // The public trait entry performs this too. Keep the check here so no
        // future inherent caller can bypass the contract ordering.
        request.validate()?;
        if !self.provides_intermediate_sweep() {
            return Ok(None);
        }
        if request
            .plan
            .ops_backward
            .iter()
            .any(|op| matches!(op, GpuBackwardOp::Add { .. } | GpuBackwardOp::Sub { .. }))
        {
            return self.run_intermediate_sweep_dag(request);
        }
        // This first authority slice is pinned to the audited per-layer,
        // word-taint path. Optional experimental execution modes remain clean
        // pre-dispatch declines rather than widening the review surface.
        if super::crown_backward_sound_resident::fold_coalesce_enabled()
            || !self.taint_words_armed()
        {
            return Ok(None);
        }
        let Some(prepared) = prepare_chain(request)? else {
            return Ok(None);
        };
        // Qualification normally materializes these. Keep this pre-acceptance
        // call as a structural fence: no accepted transaction may synchronously
        // compile a verdict-critical resident or concretization pipeline.
        let _ = self.resident_backward_pipelines();
        if self.sound_concretize_pipeline_cached().is_none() {
            return Err(NyError::UnsupportedOp(
                "WGPU intermediate sweep sound-concretize pipeline was not materialized during qualification"
                    .into(),
            ));
        }
        if !self.denorm_preserve_contract_intact() {
            return Err(NyError::UnsupportedOp(
                "WGPU intermediate sweep pipeline loading contract was lost".into(),
            ));
        }
        deadline_check(request.deadline, "before device preflight")?;
        let limits = self.device.limits();
        let fold = match super::crown_backward_sound_resident::resident_fold_plan(
            &prepared.layers,
            request.plan.total_rows,
            request.plan.total_rows,
            request.plan.slot_dims[0],
            limits.max_compute_workgroups_per_dimension,
            limits.max_buffer_size,
            limits.max_storage_buffer_binding_size,
        ) {
            Ok(fold) => fold,
            Err(error) => return capacity_decline(error),
        };
        if let Err(error) = self.intermediate_sweep_concretize_preflight(
            request.plan.total_rows,
            fold.final_dim,
            request.input_lower,
            request.input_upper,
        ) {
            return capacity_decline(error);
        }
        // Acceptance starts only after acquiring the exact device's outer GPU
        // transaction. Retained caches cannot grow or clear between this
        // snapshot, cap reservation, either phase, and final validation.
        let mut gpu_transaction =
            self.begin_gpu_checked_transaction("WGPU intermediate sweep", request.deadline)?;
        let retained_device_bytes = retained_device_bytes(self)?;
        let peak_device_bytes =
            sweep_peak_ceiling(request, &prepared, fold, retained_device_bytes)?;
        let Some(mut memory_reservation) =
            self.reserve_intermediate_sweep_memory(request.max_device_bytes, peak_device_bytes)?
        else {
            return Ok(None);
        };
        deadline_check(request.deadline, "before accepting the request")?;

        let _deadline_scope =
            crate::wgpu_device::CallLocalCrownDeadlineScope::arm(request.deadline);
        let scope = SweepScope::arm(prepared.boundaries)?;
        // Exact default-path seed traffic: lower/upper values, two complete
        // error workspaces (including zero tails), four taint-word workspaces,
        // four bias lanes, the ones vector, and the beta scratch initialization.
        let seed_f32_elems = prepared
            .lower_seed
            .len()
            .checked_mul(2)
            .and_then(|value| value.checked_add(fold.a_elems.checked_mul(6)?))
            .and_then(|value| value.checked_add(request.plan.total_rows.checked_mul(4)?))
            .and_then(|value| value.checked_add(fold.max_dim))
            .and_then(|value| value.checked_add(fold.slope_dim))
            .ok_or_else(|| NyError::InvalidSpec("sweep seed transfer overflow".into()))?;
        note_host_to_device(
            seed_f32_elems
                .checked_mul(size_of::<f32>())
                .ok_or_else(|| NyError::InvalidSpec("sweep seed byte overflow".into()))?,
        );
        let zeros_a = vec![0.0f32; prepared.lower_seed.len()];
        let zeros_b = vec![0.0f32; request.plan.total_rows];
        let coeff = self.crown_backward_sound_resident_coeff_seeded_err(
            &prepared.layers,
            &prepared.lower_seed,
            &prepared.upper_seed,
            &zeros_a,
            &zeros_a,
            &zeros_b,
            &zeros_b,
            &zeros_b,
            &zeros_b,
            request.plan.total_rows,
            request.plan.slot_dims[0],
            &[],
            &[],
        )?;
        let (lower, upper) = self.concretize_resident_coeff(
            &coeff,
            request.plan.total_rows,
            request.input_lower,
            request.input_upper,
        )?;
        // R2: the concretize readback above ended in a successful bounded
        // poll(Wait), so every submission this sweep made has provably
        // drained. Retire the scope BEFORE consulting the deadline again:
        // expiry past this point discards a finished result and must never be
        // recorded as an undrained in-flight abort that poisons the ledger.
        let work = scope.finish()?;
        deadline_check(request.deadline, "after concretization")?;
        if lower.len() != request.plan.total_rows || upper.len() != request.plan.total_rows {
            return Err(NyError::InternalError(
                "WGPU intermediate sweep returned a partial carrier".into(),
            ));
        }
        if lower
            .iter()
            .zip(&upper)
            .any(|(&lo, &hi)| !lo.is_finite() || !hi.is_finite() || lo > hi)
        {
            return Err(NyError::InternalError(
                "WGPU intermediate sweep returned a non-publishable interval".into(),
            ));
        }

        let mut targets = Vec::with_capacity(request.plan.injections.len());
        for (index, injection) in request.plan.injections.iter().enumerate() {
            if index.is_multiple_of(VALIDATION_POLL_STRIDE) {
                deadline_check(request.deadline, "while associating results")?;
            }
            let end = injection
                .row_offset
                .checked_add(injection.selected_rows.len())
                .ok_or_else(|| NyError::InternalError("sweep result slice overflow".into()))?;
            let lower_bounds = lower
                .get(injection.row_offset..end)
                .ok_or_else(|| NyError::InternalError("sweep lower result slice missing".into()))?
                .to_vec();
            let upper_bounds = upper
                .get(injection.row_offset..end)
                .ok_or_else(|| NyError::InternalError("sweep upper result slice missing".into()))?
                .to_vec();
            targets.push(GpuIntermediateTargetResult {
                target_id: injection.target_id,
                row_offset: injection.row_offset,
                selected_rows: Arc::clone(&injection.selected_rows),
                lower_bounds,
                upper_bounds,
            });
        }
        self.require_intermediate_sweep_authority()?;
        let receipt = GpuIntermediateSweepReceipt {
            graph_identity_sha256: request.plan.graph_identity_sha256,
            input_identity_sha256: request.input_identity_sha256,
            bounds_identity_sha256: request.plan.bounds_identity_sha256,
            target_set_identity_sha256: request.plan.target_set_identity_sha256,
            requested_targets: request.plan.injections.len(),
            completed_targets: request.plan.injections.len(),
            requested_rows: request.plan.total_rows,
            completed_rows: request.plan.total_rows,
            peak_device_bytes,
            dispatches: work.dispatches,
            host_to_device_bytes: work.host_to_device_bytes,
            device_to_host_bytes: work.device_to_host_bytes,
            readbacks: work.readbacks,
            submits: work.submits,
            synchronizations: work.synchronizations,
            waves: 1,
        };
        // Validate atomically before returning. The opaque payload is consumed,
        // read only through the validated wrapper, then re-wrapped for the
        // caller's mandatory independent validation.
        let validated =
            GpuIntermediateSweepResult::new_unvalidated(targets, receipt).validate(request)?;
        self.require_intermediate_sweep_authority()?;
        memory_reservation.release()?;
        gpu_transaction.finish("WGPU intermediate sweep")?;
        // Authority can be revoked lazily by test hooks or qualification state;
        // recheck after the transaction's final error/deadline fence and
        // immediately before exposing the validated payload.
        self.require_intermediate_sweep_authority()?;
        let (targets, receipt) = validated.into_parts();
        Ok(Some(GpuIntermediateSweepResult::new_unvalidated(
            targets, receipt,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ny_core::{
        CertifiedWeightError, GpuBackwardSlot, GpuIntermediateInjection, GpuIntermediateSweepPlan,
    };
    use std::time::Duration;

    fn linear(out_features: usize, in_features: usize, weight: &[f32]) -> GpuCrownLayer {
        GpuCrownLayer::Linear {
            weight: Arc::from(weight),
            bias: None,
            out_features,
            in_features,
            cert_err: CertifiedWeightError::default(),
        }
    }

    fn request_for<'a>(
        plan: &'a GpuIntermediateSweepPlan,
        lower: &'a [f32],
        upper: &'a [f32],
    ) -> GpuIntermediateSweepRequest<'a> {
        GpuIntermediateSweepRequest {
            plan,
            input_identity_sha256: [4; 32],
            input_lower: lower,
            input_upper: upper,
            deadline: Instant::now() + Duration::from_secs(10),
            max_device_bytes: 1 << 30,
        }
    }

    fn unary_identity_chain_plan(
        unary_layers: usize,
        identity_ops: usize,
    ) -> GpuIntermediateSweepPlan {
        assert!(unary_layers > 0);
        let op_count = unary_layers + identity_ops;
        let mut ops = Vec::with_capacity(op_count);
        for slot in 0..op_count {
            let output = GpuBackwardSlot(slot as u32);
            let input = GpuBackwardSlot((slot + 1) as u32);
            if slot < unary_layers {
                ops.push(GpuBackwardOp::Unary {
                    output,
                    input,
                    layer: Box::new(linear(1, 1, &[1.0])),
                });
            } else {
                ops.push(GpuBackwardOp::Identity { output, input });
            }
        }
        GpuIntermediateSweepPlan {
            graph_identity_sha256: [1; 32],
            bounds_identity_sha256: [2; 32],
            target_set_identity_sha256: [3; 32],
            ops_backward: Arc::from(ops),
            slot_dims: Arc::from(vec![1; op_count + 1]),
            input_slot: GpuBackwardSlot(op_count as u32),
            injections: Arc::from([GpuIntermediateInjection {
                target_id: 10,
                slot: GpuBackwardSlot(0),
                target_shape: Arc::from([1]),
                selected_rows: Arc::from([0]),
                row_offset: 0,
            }]),
            total_rows: 1,
        }
    }

    #[test]
    fn resident_chain_caps_are_exact_and_decline_before_dispatch() {
        let lower = [-1.0];
        let upper = [1.0];

        let admitted = unary_identity_chain_plan(
            MAX_RESIDENT_CHAIN_LAYERS,
            MAX_RESIDENT_CHAIN_OPS - MAX_RESIDENT_CHAIN_LAYERS,
        );
        let admitted_request = request_for(&admitted, &lower, &upper);
        admitted_request.validate().unwrap();
        assert!(prepare_chain(&admitted_request).unwrap().is_some());

        let too_many_layers = unary_identity_chain_plan(MAX_RESIDENT_CHAIN_LAYERS + 1, 0);
        let layer_request = request_for(&too_many_layers, &lower, &upper);
        layer_request.validate().unwrap();
        assert!(prepare_chain(&layer_request).unwrap().is_none());

        let too_many_ops = unary_identity_chain_plan(1, MAX_RESIDENT_CHAIN_OPS);
        let op_request = request_for(&too_many_ops, &lower, &upper);
        op_request.validate().unwrap();
        assert!(prepare_chain(&op_request).unwrap().is_none());
    }

    #[test]
    fn peak_ceiling_charges_complete_queued_upload_staging() {
        let lower = [-1.0];
        let upper = [1.0];
        let one_layer = unary_identity_chain_plan(1, 0);
        let request = request_for(&one_layer, &lower, &upper);
        let prepared = prepare_chain(&request).unwrap().unwrap();
        let fold = ResidentFoldPlan {
            num_specs_u32: 1,
            num_specs_per_dom_u32: 1,
            n_domains: 1,
            seed_elems: 1,
            final_dim: 1,
            max_dim: 1,
            max_gemm_out: 1,
            a_elems: 1,
            slope_dim: 1,
            max_wg: 1,
        };
        let peak = sweep_peak_ceiling(&request, &prepared, fold, 0).unwrap();
        assert!(
            peak >= 2 * UNIFORM_UPLOAD_BYTES_PER_LAYER,
            "every explicit resident buffer/uniform allowance must be paired with queued upload staging"
        );
    }

    #[test]
    fn queued_activation_upload_ceiling_sums_every_layer() {
        let activation = GpuCrownLayer::Activation {
            lower_slope: vec![1.0; 1024],
            upper_slope: vec![1.0; 1024],
            lower_intercept: vec![0.0; 1024],
            upper_intercept: vec![0.0; 1024],
            num_neurons: 1024,
        };
        let one = super::super::crown_backward_sound_resident::resident_fold_staging_capacity(
            std::slice::from_ref(&activation),
            1,
        )
        .unwrap();
        let sixty_four =
            super::super::crown_backward_sound_resident::resident_fold_staging_capacity(
                &vec![activation; MAX_RESIDENT_CHAIN_LAYERS],
                1,
            )
            .unwrap();
        assert!(sixty_four > one * 32);
        assert!(sixty_four >= 64 * 6 * 1024 * size_of::<f32>() as u64);
    }

    #[test]
    fn queued_certified_bias_operand_is_charged_even_without_model_bias() {
        let exact = linear(1024, 1, &[]);
        let charged = GpuCrownLayer::Linear {
            weight: Arc::from([]),
            bias: None,
            out_features: 1024,
            in_features: 1,
            cert_err: CertifiedWeightError {
                weight_rel_err: 0.0,
                bias_abs_err: 1.0e-6,
            },
        };
        let exact_bytes =
            super::super::crown_backward_sound_resident::resident_fold_staging_capacity(
                &[exact],
                1,
            )
            .unwrap();
        let charged_bytes =
            super::super::crown_backward_sound_resident::resident_fold_staging_capacity(
                &[charged],
                1,
            )
            .unwrap();
        assert_eq!(charged_bytes - exact_bytes, 1024 * size_of::<f32>() as u64);
    }

    #[test]
    fn planner_maps_multiple_depths_into_one_carrier() {
        let plan = GpuIntermediateSweepPlan {
            graph_identity_sha256: [1; 32],
            bounds_identity_sha256: [2; 32],
            target_set_identity_sha256: [3; 32],
            ops_backward: Arc::from([
                GpuBackwardOp::Identity {
                    output: GpuBackwardSlot(0),
                    input: GpuBackwardSlot(1),
                },
                GpuBackwardOp::Unary {
                    output: GpuBackwardSlot(1),
                    input: GpuBackwardSlot(2),
                    layer: Box::new(linear(2, 3, &[1.0, 0.0, 1.0, 0.0, 1.0, 1.0])),
                },
                GpuBackwardOp::Identity {
                    output: GpuBackwardSlot(2),
                    input: GpuBackwardSlot(3),
                },
                GpuBackwardOp::Unary {
                    output: GpuBackwardSlot(3),
                    input: GpuBackwardSlot(4),
                    layer: Box::new(linear(3, 2, &[1.0, 0.0, 0.0, 1.0, 1.0, -1.0])),
                },
            ]),
            slot_dims: Arc::from([2, 2, 3, 3, 2]),
            input_slot: GpuBackwardSlot(4),
            injections: Arc::from([
                GpuIntermediateInjection {
                    target_id: 10,
                    slot: GpuBackwardSlot(0),
                    target_shape: Arc::from([2]),
                    selected_rows: Arc::from([1]),
                    row_offset: 0,
                },
                GpuIntermediateInjection {
                    target_id: 20,
                    slot: GpuBackwardSlot(2),
                    target_shape: Arc::from([3]),
                    selected_rows: Arc::from([0, 2]),
                    row_offset: 1,
                },
            ]),
            total_rows: 3,
        };
        let lower = [-1.0, -2.0];
        let upper = [1.0, 2.0];
        let request = request_for(&plan, &lower, &upper);
        request.validate().unwrap();
        let prepared = prepare_chain(&request).unwrap().unwrap();
        assert_eq!(prepared.layers.len(), 2);
        assert_eq!(prepared.lower_seed, vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(prepared.upper_seed, prepared.lower_seed);
        assert_eq!(prepared.boundaries[0].resets.len(), 1);
        assert_eq!(prepared.boundaries[1].dim, 3);
        assert_eq!(
            prepared.boundaries[1].resets,
            vec![
                SweepReset {
                    carrier_row: 1,
                    coordinate: 0
                },
                SweepReset {
                    carrier_row: 2,
                    coordinate: 2
                }
            ]
        );
    }

    #[test]
    fn add_and_sub_decline_before_a_schedule_exists() {
        for subtract in [false, true] {
            let binary = if subtract {
                GpuBackwardOp::Sub {
                    output: GpuBackwardSlot(0),
                    lhs: GpuBackwardSlot(1),
                    rhs: GpuBackwardSlot(2),
                }
            } else {
                GpuBackwardOp::Add {
                    output: GpuBackwardSlot(0),
                    lhs: GpuBackwardSlot(1),
                    rhs: GpuBackwardSlot(2),
                }
            };
            let plan = GpuIntermediateSweepPlan {
                graph_identity_sha256: [1; 32],
                bounds_identity_sha256: [2; 32],
                target_set_identity_sha256: [3; 32],
                ops_backward: Arc::from([
                    binary,
                    GpuBackwardOp::Identity {
                        output: GpuBackwardSlot(1),
                        input: GpuBackwardSlot(3),
                    },
                    GpuBackwardOp::Identity {
                        output: GpuBackwardSlot(2),
                        input: GpuBackwardSlot(3),
                    },
                ]),
                slot_dims: Arc::from([2, 2, 2, 2]),
                input_slot: GpuBackwardSlot(3),
                injections: Arc::from([GpuIntermediateInjection {
                    target_id: 10,
                    slot: GpuBackwardSlot(0),
                    target_shape: Arc::from([2]),
                    selected_rows: Arc::from([0]),
                    row_offset: 0,
                }]),
                total_rows: 1,
            };
            // Core accepts this small live fan-out, while this first WGPU slice
            // must decline it rather than dropping either contribution.
            let lower = [-1.0, -1.0];
            let upper = [1.0, 1.0];
            let request = request_for(&plan, &lower, &upper);
            request.validate().unwrap();
            assert!(prepare_chain(&request).unwrap().is_none());
        }
    }

    #[test]
    fn scripted_reset_discards_dormant_history_at_the_exact_depth() {
        // Scalar carrier model: row 0 starts at output, row 1 is injected only
        // after the first fold. Dormant row 1 deliberately accumulates garbage;
        // reset must make its final coefficient equal the suffix-only result.
        let weights = [2.0f64, -3.0];
        let mut carrier = [1.0f64, 0.0];
        let mut error = [0.0f64, 7.0];
        carrier[0] *= weights[0];
        carrier[1] = 1234.0;
        error[1] = 99.0;
        carrier[1] = 1.0;
        error[1] = 0.0;
        for row in 0..2 {
            carrier[row] *= weights[1];
        }
        assert_eq!(carrier, [-6.0, -3.0]);
        assert_eq!(error[1], 0.0);
    }

    #[test]
    fn memory_cap_uses_the_stricter_call_and_backend_authority() {
        assert!(memory_cap_admits(100, 100, 200));
        assert!(memory_cap_admits(100, 200, 100));
        assert!(!memory_cap_admits(101, 100, 200));
        assert!(!memory_cap_admits(101, 200, 100));
    }

    #[test]
    fn reservation_ledger_balances_on_decline_error_and_unwind() {
        let ledger = std::sync::Mutex::new(0usize);
        assert!(reserve_sweep_ledger(&ledger, None, 99, 200, 100)
            .unwrap()
            .is_none());
        assert_eq!(*ledger.lock().unwrap(), 0);

        let aborted: Result<()> = (|| {
            let _reservation = reserve_sweep_ledger(&ledger, None, 200, 200, 100)?
                .ok_or_else(|| NyError::InternalError("unexpected reservation decline".into()))?;
            assert!(reserve_sweep_ledger(&ledger, None, 200, 200, 1)?.is_none());
            Err(NyError::DeadlineExceeded("scripted abort".into()))
        })();
        assert!(matches!(aborted, Err(NyError::DeadlineExceeded(_))));
        assert_eq!(*ledger.lock().unwrap(), 0);

        let unwound = std::panic::catch_unwind(|| {
            let _reservation = reserve_sweep_ledger(&ledger, None, 200, 200, 100)
                .unwrap()
                .unwrap();
            panic!("scripted unwind");
        });
        assert!(unwound.is_err());
        assert_eq!(*ledger.lock().unwrap(), 0);

        let mut completed = reserve_sweep_ledger(&ledger, None, 200, 200, 100)
            .unwrap()
            .unwrap();
        completed.release().unwrap();
        assert_eq!(*ledger.lock().unwrap(), 0);

        let corrupt = std::sync::Mutex::new(0usize);
        let mut corrupted = reserve_sweep_ledger(&corrupt, None, 200, 200, 100)
            .unwrap()
            .unwrap();
        *corrupt.lock().unwrap() = 99;
        assert!(corrupted.release().is_err());
        assert_eq!(*corrupt.lock().unwrap(), usize::MAX);

        let post_submit_aborted = std::sync::Mutex::new(0usize);
        let reservation = reserve_sweep_ledger(&post_submit_aborted, None, 200, 200, 100)
            .unwrap()
            .unwrap();
        let scope = SweepScope::arm(vec![SweepBoundary::default()]).unwrap();
        note_submits(1);
        drop(scope);
        drop(reservation);
        assert_eq!(*post_submit_aborted.lock().unwrap(), usize::MAX);
        assert!(
            reserve_sweep_ledger(&post_submit_aborted, None, 200, 200, 1)
                .unwrap()
                .is_none()
        );
    }

    /// Scripted [`SweepAbortDrain`] so the post-submit ledger paths are
    /// exercised without a device, mirroring the module's flag-based tests.
    struct ScriptedDrain {
        calls: Cell<usize>,
        wedged: bool,
    }

    impl ScriptedDrain {
        fn new(wedged: bool) -> Self {
            Self {
                calls: Cell::new(0),
                wedged,
            }
        }
    }

    impl SweepAbortDrain for ScriptedDrain {
        fn drain_after_post_submit_abort(&self) -> Result<()> {
            self.calls.set(self.calls.get() + 1);
            if self.wedged {
                Err(NyError::DeadlineExceeded(
                    "scripted wedged-device drain timeout".into(),
                ))
            } else {
                Ok(())
            }
        }
    }

    /// Simulate an accepted sweep aborting after one recorded submission.
    /// `deadline_shaped` marks the abort as a call-local deadline expiry (the
    /// shared readback helpers' `note_post_submit_abort`); `synced` records a
    /// successful queue-idle synchronization after the submit first.
    fn abort_after_submit(deadline_shaped: bool, synced: bool) {
        let scope = SweepScope::arm(vec![SweepBoundary::default()]).unwrap();
        note_submits(1);
        if synced {
            note_device_to_host(0, 0, 1);
        }
        if deadline_shaped {
            note_post_submit_abort();
        }
        drop(scope);
    }

    #[test]
    fn deadline_abort_with_successful_drain_restores_the_ledger() {
        let ledger = std::sync::Mutex::new(0usize);
        let drain = ScriptedDrain::new(false);
        let reservation = reserve_sweep_ledger(&ledger, Some(&drain), 200, 200, 100)
            .unwrap()
            .unwrap();
        abort_after_submit(true, false);
        drop(reservation);
        // First-cause-wins: the scope drop on the same unwind must not have
        // downgraded the deadline abort, so the drain was consulted once.
        assert_eq!(drain.calls.get(), 1);
        assert_eq!(*ledger.lock().unwrap(), 0);
        assert!(reserve_sweep_ledger(&ledger, Some(&drain), 200, 200, 100)
            .unwrap()
            .is_some());
    }

    #[test]
    fn deadline_abort_with_wedged_drain_keeps_the_ledger_fail_closed() {
        let ledger = std::sync::Mutex::new(0usize);
        let drain = ScriptedDrain::new(true);
        let reservation = reserve_sweep_ledger(&ledger, Some(&drain), 200, 200, 100)
            .unwrap()
            .unwrap();
        abort_after_submit(true, false);
        drop(reservation);
        assert_eq!(drain.calls.get(), 1);
        assert_eq!(*ledger.lock().unwrap(), usize::MAX);
        assert!(reserve_sweep_ledger(&ledger, Some(&drain), 200, 200, 1)
            .unwrap()
            .is_none());
    }

    #[test]
    fn deadline_abort_without_drain_authority_keeps_the_ledger_fail_closed() {
        let ledger = std::sync::Mutex::new(0usize);
        let reservation = reserve_sweep_ledger(&ledger, None, 200, 200, 100)
            .unwrap()
            .unwrap();
        abort_after_submit(true, false);
        drop(reservation);
        assert_eq!(*ledger.lock().unwrap(), usize::MAX);
    }

    #[test]
    fn proven_drained_abort_restores_the_ledger_without_polling() {
        let ledger = std::sync::Mutex::new(0usize);
        // Wedged on purpose: a proven-drained abort must never consult it.
        let drain = ScriptedDrain::new(true);
        let reservation = reserve_sweep_ledger(&ledger, Some(&drain), 200, 200, 100)
            .unwrap()
            .unwrap();
        abort_after_submit(false, true);
        drop(reservation);
        assert_eq!(drain.calls.get(), 0);
        assert_eq!(*ledger.lock().unwrap(), 0);
        assert!(reserve_sweep_ledger(&ledger, Some(&drain), 200, 200, 100)
            .unwrap()
            .is_some());
    }

    #[test]
    fn unknown_in_flight_abort_still_poisons_even_with_drain_authority() {
        // Pins the non-deadline post-submit fault behavior relied on by the
        // DAG route's live validation-fault test: fail closed, no drain.
        let ledger = std::sync::Mutex::new(0usize);
        let drain = ScriptedDrain::new(false);
        let reservation = reserve_sweep_ledger(&ledger, Some(&drain), 200, 200, 100)
            .unwrap()
            .unwrap();
        abort_after_submit(false, false);
        drop(reservation);
        assert_eq!(drain.calls.get(), 0);
        assert_eq!(*ledger.lock().unwrap(), usize::MAX);
    }

    #[test]
    fn post_submit_poison_line_is_pinned_for_the_kill_grep() {
        // The design's §7 kill criterion and ny-propagate's
        // `sweep_kill_line_matches_the_sweep_source` grep for this EXACT text;
        // rewording it would silently blind both.
        assert_eq!(
            POST_SUBMIT_POISON_LINE,
            "WGPU intermediate sweep exited after submission without a final drain; memory ledger left permanently fail-closed"
        );
    }

    #[test]
    fn scripted_scope_requires_every_boundary_and_records_exact_work() {
        let scope = SweepScope::arm(vec![
            SweepBoundary {
                dim: 2,
                resets: Vec::new(),
            },
            SweepBoundary {
                dim: 3,
                resets: vec![SweepReset {
                    carrier_row: 0,
                    coordinate: 2,
                }],
            },
        ])
        .unwrap();
        note_dispatches(7);
        note_host_to_device(11);
        note_device_to_host(13, 2, 1);
        note_submits(3);
        assert_eq!(take_boundary(1, 3).unwrap().unwrap().resets.len(), 1);
        let work = scope.finish().unwrap();
        assert_eq!(
            work,
            SweepWorkReceipt {
                dispatches: 7,
                host_to_device_bytes: 11,
                device_to_host_bytes: 13,
                readbacks: 2,
                submits: 3,
                synchronizations: 1,
            }
        );
    }
}

#[cfg(all(test, feature = "gpu-tests"))]
mod gpu_tests {
    use super::*;
    use crate::wgpu_device::test_support::{gpu_test_serial_guard, require_verdict_device};
    use ny_core::{
        CertifiedWeightError, GpuBackwardSlot, GpuCrownBackward, GpuCrownSeed,
        GpuIntermediateInjection, GpuIntermediateSweepPlan, GpuResnetSegment,
    };
    use ny_test_utils::env::ScopedEnvVar;
    use std::sync::mpsc;
    use std::time::Duration;

    fn linear(weight: &[f32]) -> GpuCrownLayer {
        GpuCrownLayer::Linear {
            weight: Arc::from(weight),
            bias: None,
            out_features: 2,
            in_features: 2,
            cert_err: CertifiedWeightError::default(),
        }
    }

    fn scalar_linear(weight: f32, bias: f32) -> GpuCrownLayer {
        GpuCrownLayer::Linear {
            weight: Arc::from([weight]),
            bias: Some(Arc::from([bias])),
            out_features: 1,
            in_features: 1,
            cert_err: CertifiedWeightError::default(),
        }
    }

    #[test]
    fn live_biased_diamond_and_sub_execute_on_the_real_wgpu_backend() {
        let _serial = gpu_test_serial_guard();
        let _coalesce = ScopedEnvVar::unset("NY_FOLD_COALESCE");
        let _eft = ScopedEnvVar::unset("NY_EFT_ERR");
        let _words = ScopedEnvVar::unset("NY_GPU_TAINT_WORDS");
        let device = require_verdict_device();
        assert!(device.provides_sound_intermediate_sweep());

        // y = ((2x + 3) + (-x + 5)) + 7 = x + 15. The bias 7
        // reaches the binary fork and must be retained on exactly one branch.
        let diamond = GpuIntermediateSweepPlan {
            graph_identity_sha256: [31; 32],
            bounds_identity_sha256: [32; 32],
            target_set_identity_sha256: [33; 32],
            ops_backward: Arc::from([
                GpuBackwardOp::Unary {
                    output: GpuBackwardSlot(0),
                    input: GpuBackwardSlot(1),
                    layer: Box::new(scalar_linear(1.0, 7.0)),
                },
                GpuBackwardOp::Add {
                    output: GpuBackwardSlot(1),
                    lhs: GpuBackwardSlot(2),
                    rhs: GpuBackwardSlot(3),
                },
                GpuBackwardOp::Unary {
                    output: GpuBackwardSlot(2),
                    input: GpuBackwardSlot(4),
                    layer: Box::new(scalar_linear(2.0, 3.0)),
                },
                GpuBackwardOp::Unary {
                    output: GpuBackwardSlot(3),
                    input: GpuBackwardSlot(4),
                    layer: Box::new(scalar_linear(-1.0, 5.0)),
                },
            ]),
            slot_dims: Arc::from([1, 1, 1, 1, 1]),
            input_slot: GpuBackwardSlot(4),
            injections: Arc::from([
                GpuIntermediateInjection {
                    target_id: 301,
                    slot: GpuBackwardSlot(0),
                    target_shape: Arc::from([1]),
                    selected_rows: Arc::from([0]),
                    row_offset: 0,
                },
                GpuIntermediateInjection {
                    target_id: 302,
                    slot: GpuBackwardSlot(2),
                    target_shape: Arc::from([1]),
                    selected_rows: Arc::from([0]),
                    row_offset: 1,
                },
            ]),
            total_rows: 2,
        };
        let input_lower = [-1.0];
        let input_upper = [2.0];
        let diamond_request = GpuIntermediateSweepRequest {
            plan: &diamond,
            input_identity_sha256: [34; 32],
            input_lower: &input_lower,
            input_upper: &input_upper,
            deadline: Instant::now() + Duration::from_secs(30),
            max_device_bytes: 256 << 20,
        };
        let diamond_result = device
            .crown_backward_gpu_sound_intermediate_sweep(&diamond_request)
            .expect("live biased-diamond sweep")
            .expect("qualified Add DAG must be accepted")
            .validate(&diamond_request)
            .expect("biased-diamond atomic result validation");
        let targets = diamond_result.targets();
        assert_eq!(targets.len(), 2);
        assert_eq!((targets[0].target_id, targets[0].row_offset), (301, 0));
        assert_eq!((targets[1].target_id, targets[1].row_offset), (302, 1));
        assert!(targets[0].lower_bounds[0] <= 14.0);
        assert!(targets[0].upper_bounds[0] >= 17.0);
        assert!(targets[0].lower_bounds[0] > 13.99);
        assert!(targets[0].upper_bounds[0] < 17.01);
        assert!(targets[1].lower_bounds[0] <= 1.0);
        assert!(targets[1].upper_bounds[0] >= 7.0);
        assert!(targets[1].lower_bounds[0] > 0.99);
        assert!(targets[1].upper_bounds[0] < 7.01);

        // The same input reached through both arms of x - x must cancel. This
        // covers the centre-only negation and merge into an occupied slot.
        let subtract = GpuIntermediateSweepPlan {
            graph_identity_sha256: [41; 32],
            bounds_identity_sha256: [42; 32],
            target_set_identity_sha256: [43; 32],
            ops_backward: Arc::from([
                GpuBackwardOp::Sub {
                    output: GpuBackwardSlot(0),
                    lhs: GpuBackwardSlot(1),
                    rhs: GpuBackwardSlot(2),
                },
                GpuBackwardOp::Identity {
                    output: GpuBackwardSlot(1),
                    input: GpuBackwardSlot(3),
                },
                GpuBackwardOp::Identity {
                    output: GpuBackwardSlot(2),
                    input: GpuBackwardSlot(3),
                },
            ]),
            slot_dims: Arc::from([1, 1, 1, 1]),
            input_slot: GpuBackwardSlot(3),
            injections: Arc::from([GpuIntermediateInjection {
                target_id: 401,
                slot: GpuBackwardSlot(0),
                target_shape: Arc::from([1]),
                selected_rows: Arc::from([0]),
                row_offset: 0,
            }]),
            total_rows: 1,
        };
        let subtract_request = GpuIntermediateSweepRequest {
            plan: &subtract,
            input_identity_sha256: [44; 32],
            input_lower: &input_lower,
            input_upper: &input_upper,
            deadline: Instant::now() + Duration::from_secs(30),
            max_device_bytes: 256 << 20,
        };
        let subtract_result = device
            .crown_backward_gpu_sound_intermediate_sweep(&subtract_request)
            .expect("live Sub sweep")
            .expect("qualified Sub DAG must be accepted")
            .validate(&subtract_request)
            .expect("Sub atomic result validation");
        let target = &subtract_result.targets()[0];
        assert!(target.lower_bounds[0] <= 0.0);
        assert!(target.upper_bounds[0] >= 0.0);
        assert!(target.lower_bounds[0] > -1.0e-5);
        assert!(target.upper_bounds[0] < 1.0e-5);
    }

    #[test]
    fn live_multi_depth_chain_is_atomic_and_encloses_exact_targets() {
        let _serial = gpu_test_serial_guard();
        let _coalesce = ScopedEnvVar::unset("NY_FOLD_COALESCE");
        let _eft = ScopedEnvVar::unset("NY_EFT_ERR");
        let _words = ScopedEnvVar::unset("NY_GPU_TAINT_WORDS");
        let device = require_verdict_device();
        let plan = GpuIntermediateSweepPlan {
            graph_identity_sha256: [11; 32],
            bounds_identity_sha256: [12; 32],
            target_set_identity_sha256: [13; 32],
            ops_backward: Arc::from([
                GpuBackwardOp::Unary {
                    output: GpuBackwardSlot(0),
                    input: GpuBackwardSlot(1),
                    layer: Box::new(linear(&[2.0, 0.0, 0.0, 1.0])),
                },
                GpuBackwardOp::Unary {
                    output: GpuBackwardSlot(1),
                    input: GpuBackwardSlot(2),
                    layer: Box::new(linear(&[1.0, 1.0, -1.0, 2.0])),
                },
            ]),
            slot_dims: Arc::from([2, 2, 2]),
            input_slot: GpuBackwardSlot(2),
            injections: Arc::from([
                GpuIntermediateInjection {
                    target_id: 100,
                    slot: GpuBackwardSlot(0),
                    target_shape: Arc::from([2]),
                    selected_rows: Arc::from([0]),
                    row_offset: 0,
                },
                GpuIntermediateInjection {
                    target_id: 200,
                    slot: GpuBackwardSlot(1),
                    target_shape: Arc::from([2]),
                    selected_rows: Arc::from([1]),
                    row_offset: 1,
                },
            ]),
            total_rows: 2,
        };
        let input_lower = [-1.0, -2.0];
        let input_upper = [1.0, 3.0];
        let request = GpuIntermediateSweepRequest {
            plan: &plan,
            input_identity_sha256: [14; 32],
            input_lower: &input_lower,
            input_upper: &input_upper,
            deadline: Instant::now() + Duration::from_secs(30),
            max_device_bytes: 256 << 20,
        };
        assert!(device.provides_sound_intermediate_sweep());
        let opaque = device
            .crown_backward_gpu_sound_intermediate_sweep(&request)
            .expect("live WGPU sweep")
            .expect("qualified chain must be accepted");
        let validated = opaque.validate(&request).expect("atomic result validation");
        let targets = validated.targets();
        assert_eq!(targets.len(), 2);
        for (target, (exact_lower, exact_upper)) in
            targets.iter().zip([(-6.0f32, 8.0f32), (-5.0, 7.0)])
        {
            assert_eq!(target.lower_bounds.len(), 1);
            assert_eq!(target.upper_bounds.len(), 1);
            assert!(target.lower_bounds[0] <= exact_lower);
            assert!(target.upper_bounds[0] >= exact_upper);
        }
        let receipt = validated.receipt();
        assert_eq!(receipt.completed_rows, 2);
        assert_eq!(receipt.completed_targets, 2);
        assert!(receipt.peak_device_bytes <= request.max_device_bytes);
        assert!(receipt.dispatches > 0 && receipt.submits > 0 && receipt.readbacks > 0);
        assert_eq!(receipt.waves, 1);
    }

    #[test]
    fn live_working_set_clear_waits_for_reserved_transaction_and_balances() {
        let _serial = gpu_test_serial_guard();
        let device = require_verdict_device();
        device.clear_crown_working_set().expect("initial clear");

        let (armed_tx, armed_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let worker_device = Arc::clone(&device);
        let worker = std::thread::spawn(move || -> Result<()> {
            let mut transaction = worker_device.begin_gpu_checked_transaction(
                "scripted reservation race",
                Instant::now() + Duration::from_secs(10),
            )?;
            let reservation = worker_device
                .reserve_intermediate_sweep_memory(4096, 4096)?
                .ok_or_else(|| NyError::InternalError("scripted reservation declined".into()))?;
            armed_tx
                .send(())
                .map_err(|error| NyError::InternalError(error.to_string()))?;
            release_rx
                .recv()
                .map_err(|error| NyError::InternalError(error.to_string()))?;
            drop(reservation);
            transaction.finish("scripted reservation race")?;
            drop(transaction);
            Ok(())
        });
        armed_rx.recv().expect("worker reservation armed");

        let (clear_started_tx, clear_started_rx) = mpsc::channel();
        let (clear_done_tx, clear_done_rx) = mpsc::channel();
        let clear_device = Arc::clone(&device);
        let clearer = std::thread::spawn(move || {
            clear_started_tx.send(()).unwrap();
            clear_done_tx
                .send(clear_device.clear_crown_working_set())
                .unwrap();
        });
        clear_started_rx.recv().expect("clear thread started");
        let premature_clear = clear_done_rx.recv_timeout(Duration::from_millis(100));
        release_tx.send(()).expect("release worker");
        worker
            .join()
            .expect("worker thread")
            .expect("worker result");
        let (was_blocked, clear_result) = match premature_clear {
            Err(mpsc::RecvTimeoutError::Timeout) => (
                true,
                clear_done_rx
                    .recv_timeout(Duration::from_secs(10))
                    .expect("clear completion"),
            ),
            other => (false, other.expect("clear channel remained connected")),
        };
        clearer.join().expect("clear thread");
        assert!(was_blocked);
        clear_result.expect("clear succeeds after transaction");
        assert_eq!(*device.intermediate_sweep_reserved_bytes.lock().unwrap(), 0);
    }

    #[test]
    fn live_transaction_scope_catches_validation_before_a_raw_outer_scope() {
        let _serial = gpu_test_serial_guard();
        let device = require_verdict_device();
        let raw = device.device();
        let outer = raw.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut transaction = device
            .begin_gpu_checked_transaction(
                "nested raw-scope discriminator",
                Instant::now() + Duration::from_secs(10),
            )
            .expect("begin checked transaction");
        let invalid_size = raw.limits().max_buffer_size.saturating_add(1);
        let _invalid = raw.create_buffer(&wgpu::BufferDescriptor {
            label: Some("intentional-invalid-buffer-for-sweep-scope-test"),
            size: invalid_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let error = transaction
            .finish("nested raw-scope discriminator")
            .expect_err("the transaction-owned inner scope must observe validation failure");
        assert!(error.to_string().contains("validation"));
        drop(transaction);
        assert!(
            pollster::block_on(outer.pop()).is_none(),
            "the caller's outer scope must not intercept a checked transaction error"
        );
    }

    #[test]
    fn live_nine_row_empty_outer_beta_uses_the_real_wgpu_backend() {
        let _serial = gpu_test_serial_guard();
        let _coalesce = ScopedEnvVar::unset("NY_FOLD_COALESCE");
        let _words = ScopedEnvVar::unset("NY_GPU_TAINT_WORDS");
        let device = require_verdict_device();
        let activation = GpuCrownLayer::Activation {
            lower_slope: vec![1.0, 1.0],
            upper_slope: vec![1.0, 1.0],
            lower_intercept: vec![0.0, 0.0],
            upper_intercept: vec![0.0, 0.0],
            num_neurons: 2,
        };
        let segments = [GpuResnetSegment::Chain(vec![
            activation,
            linear(&[1.0, 0.0, 0.0, 1.0]),
        ])];
        let mut spec = Vec::with_capacity(18);
        for row in 0..9 {
            spec.extend_from_slice(if row % 2 == 0 {
                &[1.0, 0.0]
            } else {
                &[0.0, 1.0]
            });
        }
        let seed = GpuCrownSeed {
            lower_a: Arc::from(spec.clone()),
            upper_a: Arc::from(spec),
            lower_b: Arc::from(vec![0.0; 9]),
            upper_b: Arc::from(vec![0.0; 9]),
            num_specs: 9,
            current_dim: 2,
        };
        let result = device
            .crown_backward_gpu_resnet_sound_beta(
                &segments,
                &seed,
                &[-1.0, -2.0],
                &[1.0, 3.0],
                &[],
                &[],
                &[],
            )
            .expect("real WGPU beta entry must accept an empty outer beta table");
        assert_eq!(result.lower_bounds.len(), 9);
        assert_eq!(result.upper_bounds.len(), 9);
        assert!(result
            .lower_bounds
            .iter()
            .zip(&result.upper_bounds)
            .all(|(&lower, &upper)| lower.is_finite() && upper.is_finite() && lower <= upper));
    }
}
