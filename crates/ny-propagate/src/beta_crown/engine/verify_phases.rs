// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Extracted sequential β-CROWN verification phases.

mod batch;
mod branching_dispatch;
mod initial;
mod state;

pub(in crate::beta_crown::engine) use batch::{pop_domain_batch, record_cut_gate_batch};
pub(in crate::beta_crown::engine) use initial::InitialPhaseOutcome;
pub(in crate::beta_crown::engine) use state::BabLoopState;
