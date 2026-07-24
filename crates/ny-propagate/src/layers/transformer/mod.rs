// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Legacy transformer-only IBP helper regressions.
//!
//! Production code now routes through `GELULayer` and `LayerNormLayer`; these
//! helpers remain only to preserve the historical unit tests that exercise the
//! underlying interval logic directly.

mod activation;
mod layer_norm;
