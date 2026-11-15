// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Panic containment for every router entry (ROUTER_PROGRAM §6.3, JSR F4).
//!
//! Release builds UNWIND and CHECK OVERFLOW (`Cargo.toml` `[profile.release]`
//! `panic = "unwind"`, `overflow-checks = true`), so a router bug would panic
//! inside the verdict process: on a lane thread (a lost proof) or on the main
//! thread before publication (a lost verdict). Every router entry point —
//! plan, checkpoint, facts extraction, bus drain, telemetry — therefore runs
//! inside [`fail_open`], which catches the unwind and returns
//! `Err(RouterFault)`. The FIRST fault disables the router for the rest of the
//! process, stops the bus and (#router-r2) DISARMS the deadline authority, so
//! every lane reads `None` from its cell and the row runs exactly today's
//! route; (#router-r3) it also disarms the stop authority, so every phase
//! keeps today's stopping rule; (#router-r4) and the memory lease and FAR
//! hand-on authorities, so every memory gate keeps today's live decision and
//! no lane is handed on; (#router-r5) and the route authority, so every
//! capability is armed by its own key alone and every tier keeps its fair
//! share; (#router-r6) and the trial authority, so no lane starts a trial.

use std::any::Any;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

static FAULTS: AtomicU64 = AtomicU64::new(0);
static DISABLED: AtomicBool = AtomicBool::new(false);

/// A contained router failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouterFault {
    /// The entry point that failed (`plan`, `checkpoint`, `facts`, ...).
    pub site: &'static str,
    /// The panic message, or why the call was skipped.
    pub message: String,
    /// True when the call never ran because an earlier fault disabled the
    /// router; false for the fault itself.
    pub skipped: bool,
}

impl std::fmt::Display for RouterFault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.skipped {
            write!(
                f,
                "router disabled; {} skipped: {}",
                self.site, self.message
            )
        } else {
            write!(f, "router fault at {}: {}", self.site, self.message)
        }
    }
}

impl std::error::Error for RouterFault {}

/// Run one router entry fail-open: a panic inside `f` is caught, counted,
/// disables the router (and the bus) for the rest of the process, and comes
/// back as `Err`. Once disabled, `f` is not run at all.
pub fn fail_open<T>(site: &'static str, f: impl FnOnce() -> T) -> Result<T, RouterFault> {
    if DISABLED.load(Ordering::Acquire) {
        return Err(RouterFault {
            site,
            message: "an earlier fault disabled the router".to_owned(),
            skipped: true,
        });
    }
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(value) => Ok(value),
        Err(payload) => {
            FAULTS.fetch_add(1, Ordering::AcqRel);
            DISABLED.store(true, Ordering::Release);
            super::bus::disable();
            // #router-r2: fail open to the static plan. Every lane the
            // authority governed reads `None` from here on and keeps the
            // deadline its own rule derives today.
            crate::deadline_authority::global().disarm();
            // #router-r3: and every in-lane stop rule reads `None` again, so
            // each phase keeps its own cap and today's stopping rule.
            crate::stop_authority::global().disarm();
            // #router-r4: every gate keeps today's live memory decision and
            // no lane is handed on from here.
            crate::memory_lease::global().disarm();
            crate::far_authority::global().disarm();
            // #router-r5: and no capability is routed and no tier funded by
            // a contract from here: every arming site reads its own key.
            crate::route_authority::global().disarm();
            // #router-r6: and no lane builds a trial machine from here: every
            // trial site reads `None` and keeps today's value.
            crate::trial_authority::global().disarm();
            Err(RouterFault {
                site,
                message: panic_message(payload.as_ref()),
                skipped: false,
            })
        }
    }
}

fn panic_message(payload: &(dyn Any + Send)) -> String {
    if let Some(text) = payload.downcast_ref::<&str>() {
        (*text).to_owned()
    } else if let Some(text) = payload.downcast_ref::<String>() {
        text.clone()
    } else {
        "non-string panic payload".to_owned()
    }
}

/// Whether an earlier fault disabled the router.
#[must_use]
pub fn disabled() -> bool {
    DISABLED.load(Ordering::Acquire)
}

/// Faults contained so far.
#[must_use]
pub fn faults() -> u64 {
    FAULTS.load(Ordering::Acquire)
}

/// Clear the fault state (process reuse and tests).
pub fn reset() {
    FAULTS.store(0, Ordering::Release);
    DISABLED.store(false, Ordering::Release);
}
