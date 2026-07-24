// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Kokoro vocoder training_signal integration tests (#3520 Packet D, #3755).
//!
//! Split from the monolithic `training_signal.rs` into lane-specific leaves:
//! - `output_width.rs` — prefix output-width smoke/ranking (#3520 Packet D)
//! - `property.rs` — deep-prefix property-lane canary (#3755)
//!
//! Part of #3834 avoice training_signal structure dedup.

mod output_width;
mod property;
