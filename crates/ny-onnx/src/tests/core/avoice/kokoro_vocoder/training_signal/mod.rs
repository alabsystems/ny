// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Kokoro vocoder training_signal integration tests (Packet D).
//!
//! Split from the monolithic `training_signal.rs` into lane-specific leaves:
//! - `output_width.rs` — prefix output-width smoke/ranking (Packet D)
//! - `property.rs` — deep-prefix property-lane canary

mod output_width;
mod property;
