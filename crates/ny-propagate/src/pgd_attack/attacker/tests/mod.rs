// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Attacker-local regression tests, split from monolithic `attacker.rs` inline tests.

mod common;
mod corner_seed;
pub(super) mod deadline;
mod dense_sweep;
mod nan_abort;
mod osi_init;
mod projection;
mod resident_ibp;
mod restart_when_stuck;
mod smooth_sign;
mod surrogate_sign;

use super::PgdAttacker;
use crate::pgd_attack::PgdConfig;
use std::time::{Duration, Instant};

#[test]
fn deadline_observer_latches_only_an_explicit_expiry_poll() {
    let future = Instant::now() + Duration::from_secs(1);
    let future_attacker = PgdAttacker::new(PgdConfig {
        deadline: Some(future),
        ..Default::default()
    });
    assert!(!future_attacker.deadline_exhausted());
    assert!(!future_attacker.past_deadline());
    assert!(!future_attacker.deadline_exhausted());

    let expired = Instant::now()
        .checked_sub(Duration::from_millis(1))
        .expect("one millisecond before now is representable");
    let expired_attacker = PgdAttacker::new(PgdConfig {
        deadline: Some(expired),
        ..Default::default()
    });
    assert!(!expired_attacker.deadline_exhausted());
    assert!(expired_attacker.past_deadline());
    assert!(expired_attacker.deadline_exhausted());
}
