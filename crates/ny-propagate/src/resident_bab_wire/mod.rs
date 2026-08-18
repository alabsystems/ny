// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Backend-neutral retained-BaB topology wire formats.
//!
//! V1 data-transfer-object fields are public, so callers may construct values
//! for transport and compatibility tests. Such construction confers no
//! validation, provider, phase-open, or raw-execution authority. The accounted
//! independent decoder is the public validation surface; producer validation
//! and encoding remain crate-private until the default-dark bridge seals the
//! exact topology and wire together.

pub mod v1;
