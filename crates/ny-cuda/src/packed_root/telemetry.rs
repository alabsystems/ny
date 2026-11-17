// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Optional host-wall timing, including failed execute calls. Dispatch time
//! includes launch and mandatory drain; it is not device kernel time. Residual
//! time includes error cleanup. Setup includes cleanup of partial allocations.
//! Mutex wait and the caller's context restoration are outside execute time.
//! Emitting the diagnostic line is also outside execute time; it remains part
//! of the outer call wall time. Compare whole calls when measuring throughput.

use super::*;
use std::cell::Cell;
use std::time::Duration;

pub(super) struct Trace<'a> {
    request: &'a Request<'a>,
    cap: usize,
    start: Option<Instant>,
    phases: [Cell<Duration>; 7],
    dispatches: Cell<usize>,
    drained_owners: Cell<usize>,
    max_dispatch: Cell<Duration>,
    status: &'static str,
}

impl<'a> Trace<'a> {
    pub(super) fn new(request: &'a Request<'a>, limits: Limits, audit: Audit<'_>) -> Self {
        let enabled = audit.expected.is_none()
            && std::env::var_os("NY_PHASE_TELEMETRY").is_some_and(|v| v == "1");
        Self {
            request,
            cap: limits.max_dispatch_steps.min(MAX_STEPS),
            start: enabled.then(Instant::now),
            phases: std::array::from_fn(|_| Cell::new(Duration::ZERO)),
            dispatches: Cell::new(0),
            drained_owners: Cell::new(0),
            max_dispatch: Cell::new(Duration::ZERO),
            status: "unwind",
        }
    }

    pub(super) fn phase(&self, index: usize) -> Phase<'_> {
        Phase {
            elapsed: &self.phases[index],
            start: self.start.map(|_| Instant::now()),
            maximum: None,
        }
    }

    pub(super) fn measure<T>(&self, index: usize, run: impl FnOnce() -> T) -> T {
        let _phase = self.phase(index);
        run()
    }

    pub(super) fn dispatch(&self, count: usize, run: impl FnOnce() -> Result<()>) -> Result<()> {
        let Some(_) = self.start else { return run() };
        let _phase = Phase {
            elapsed: &self.phases[3],
            start: Some(Instant::now()),
            maximum: Some(&self.max_dispatch),
        };
        self.dispatches.set(self.dispatches.get() + 1);
        let result = run();
        if result.is_ok() {
            self.drained_owners.set(self.drained_owners.get() + count);
        }
        result
    }

    pub(super) fn finish(&mut self, result: &Result<Option<Output>>) {
        self.status = match result {
            Ok(Some(_)) => "complete",
            Ok(None) => "refused",
            Err(NyError::DeadlineExceeded(_)) => "deadline",
            Err(_) => "error",
        };
    }
}

impl Drop for Trace<'_> {
    fn drop(&mut self) {
        let Some(start) = self.start else { return };
        let elapsed = start.elapsed();
        let phases = self.phases.each_ref().map(|p| p.get());
        let other = elapsed.saturating_sub(phases.iter().sum());
        let [admission, maps, setup, dispatch, readback, checks, cleanup] =
            phases.map(|d| d.as_micros());
        let r = self.request;
        // One record per attempted execute, never one per dispatch. Startup
        // qualification is excluded. Existing aggregate V1 counters are intact.
        eprintln!("NY_PACKED_ROOT_CALL_V1 status={} spatial={} channels={} stride={} taps={} cap={} dispatches={} drained_owners={} max_dispatch_us={} admission_us={} maps_us={} setup_us={} dispatch_us={} readback_us={} checks_us={} cleanup_us={} other_us={} execute_us={}",
            self.status, r.output_spatial, r.output_channels, r.output_stride, r.taps,
            self.cap, self.dispatches.get(), self.drained_owners.get(), self.max_dispatch.get().as_micros(),
            admission, maps, setup, dispatch, readback, checks, cleanup, other.as_micros(), elapsed.as_micros());
    }
}

pub(super) struct Phase<'a> {
    elapsed: &'a Cell<Duration>,
    start: Option<Instant>,
    maximum: Option<&'a Cell<Duration>>,
}
impl Drop for Phase<'_> {
    fn drop(&mut self) {
        if let Some(start) = self.start {
            let elapsed = start.elapsed();
            self.elapsed.set(self.elapsed.get() + elapsed);
            if let Some(maximum) = self.maximum {
                maximum.set(maximum.get().max(elapsed));
            }
        }
    }
}
