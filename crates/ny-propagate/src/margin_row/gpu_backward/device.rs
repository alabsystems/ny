// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The lane's device handle: injected factory, once-per-process construction,
//! and the non-blocking exclusivity token (design §6 "admit", R3/R4).
//!
//! ny-propagate depends on ny-gpu only as a dev-dependency, so the concrete
//! executor (`ny_gpu::MarginRowLaneDevice`, a DEDICATED `wgpu` device + queue)
//! is injected through [`ny_core::MarginRowLaneGpu`] — registered once at
//! startup by ny-cli via [`set_margin_row_lane_gpu_factory`], exactly the
//! `sound_gpu_gate::set_sound_gpu_crown_factory` pattern. The factory closure
//! is cheap; the device is constructed lazily on the FIRST armed admission,
//! so a dark lane never touches the GPU.
//!
//! R3: this handle is lane-local. It never routes through
//! `gpu_crown_backward_route_with_deadline`, never installs a deadline on any
//! backend, and never participates in the sweep's serialization mutex or
//! memory ledger — the executor's own contract (`ny_core::margin_row_lane_gpu`
//! module docs) plus this module's refusal-only failure handling keep the
//! shared-fate surface at zero.
//!
//! R4: [`try_acquire_device`] is a compare-exchange on an `AtomicBool` —
//! refusal cost is O(ns), never a mutex wait. A busy device is
//! [`Refusal::Busy`] and the caller runs the CPU pass.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use ny_core::MarginRowLaneGpu;

/// The injected factory. First install wins; installing after the handle was
/// already resolved has no effect (the resolution is once-per-process).
type LaneGpuFactory = Box<dyn Fn() -> Option<Arc<dyn MarginRowLaneGpu>> + Send + Sync>;

static FACTORY: OnceLock<LaneGpuFactory> = OnceLock::new();

/// Register the process's margin-row lane GPU factory (called once by ny-cli
/// at startup; tests may install their own). The closure runs at most once,
/// on the first ARMED admission — registration itself costs nothing.
pub fn set_margin_row_lane_gpu_factory<F>(factory: F)
where
    F: Fn() -> Option<Arc<dyn MarginRowLaneGpu>> + Send + Sync + 'static,
{
    let _ = FACTORY.set(Box::new(factory));
}

/// Resolve (once per process) and return the lane's device executor.
///
/// `Err(reason)` is PERMANENT for this process — the caller latches the
/// lane's channel dead with it (no factory, or the factory could not build a
/// device, cannot heal without a restart; fail-closed beats flapping).
pub(super) fn lane_gpu() -> Result<&'static Arc<dyn MarginRowLaneGpu>, &'static str> {
    static HANDLE: OnceLock<Option<Arc<dyn MarginRowLaneGpu>>> = OnceLock::new();
    let resolved = HANDLE.get_or_init(|| match FACTORY.get() {
        None => None,
        Some(factory) => factory(),
    });
    match resolved {
        Some(gpu) => Ok(gpu),
        None => Err(if FACTORY.get().is_none() {
            "no margin-row lane GPU factory installed"
        } else {
            "margin-row lane GPU device construction failed"
        }),
    }
}

/// Lane-local exclusivity token (R4): non-blocking, RAII-released.
///
/// The device executor serializes its own submissions internally; this token
/// exists so an ADMISSION never waits — a pass that finds the lane busy
/// refuses (`Refusal::Busy`) in O(ns) and the caller runs the CPU pass. The
/// once-per-process M1 self-check deliberately does not take it: it runs
/// inside the authority `OnceLock` initializer, and no transaction can exist
/// before authority does, so it cannot contend with anything.
pub(super) struct DeviceToken;

static DEVICE_BUSY: AtomicBool = AtomicBool::new(false);

/// Try to acquire the lane device without blocking.
pub(super) fn try_acquire_device() -> Option<DeviceToken> {
    DEVICE_BUSY
        .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
        .is_ok()
        .then_some(DeviceToken)
}

impl Drop for DeviceToken {
    fn drop(&mut self) {
        DEVICE_BUSY.store(false, Ordering::Release);
    }
}
