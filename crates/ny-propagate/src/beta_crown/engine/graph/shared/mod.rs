// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared graph-BaB service layer.
//!
//! Provides reusable bootstrap, setup, and lifecycle state services used by
//! all three graph-BaB modes: single-objective ReLU split, multi-objective,
//! and GPU BaB.

pub(crate) mod init;
pub(crate) mod setup;
pub(crate) mod stabilize;
pub(crate) mod state;
