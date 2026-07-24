// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Attacker-local regression tests, split from monolithic `attacker.rs` inline tests.

mod common;
mod dense_sweep;
mod nan_abort;
mod osi_init;
mod projection;
mod resident_ibp;
mod restart_when_stuck;
mod smooth_sign;
mod surrogate_sign;
