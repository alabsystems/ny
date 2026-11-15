// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Downgrade-only guard library (ENGINE_PROGRAM §1.7 (a′), milestone M2d).
//!
//! Every guard here can only turn a verdict into `unknown` — it produces a
//! [`crate::certify::GuardReport`] trip, which `authorize_unsat` / `ratify`
//! consume as a refusal. Nothing here constructs a verdict, tightens a bound,
//! or changes which lane runs. Removing a guard therefore cannot make a
//! correct verdict wrong; it can only let an already-wrong one through. That
//! asymmetry is why they run in BOTH proof profiles: the profile chooses only
//! how many points they sample ([`ny_core::proof_profile::AuditRates`]).
//!
//! * [`containment`] — G2: certified boxes and rows must contain the f64
//!   reference semantics at sampled points, generic over a lane's IR through
//!   [`containment::CertifiedFrame`] (JS F9; the check that would have caught
//!   the Concat transformer defect behind vit's retracted unsats);
//! * [`spec_agreement`] — G3: the engine's objective rows must agree in sign
//!   with the sat gate's OWN spec evaluator wherever the margin is clear of
//!   the float band (JS F7; a spec misread passes every forward self-check);
//! * [`margin_hist`] — G5: the minimum certified margin and a decade
//!   histogram on the lane's summary line (JS F8);
//! * G4's rate is [`ny_core::proof_profile::AuditRates::crosscheck_every`],
//!   read by the optimised certified kernel that owns the cross-check (M1's
//!   frozen-dual block evaluator).

pub mod containment;
pub mod margin_hist;
pub mod spec_agreement;

#[cfg(test)]
mod tests;
