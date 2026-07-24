// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! CROWN and alpha-CROWN algorithm tests.

use crate::*;

mod alpha;
mod alpha_engine;
mod alpha_intermediates_fallback;
mod basic;
mod batched_backward;
mod dense_backward;
mod engine_surface;
mod gpu_fast_path;
mod gpu_fast_path_conv1d;
mod gpu_fast_path_nan_guard;
mod gpu_fast_path_soundness_gate;
mod gpu_partial_oracle;
pub(crate) mod helpers;
mod ibp;
mod overflow_clamping;
mod patches_backward;
mod patches_backward_alpha;
mod patches_conv2d;
mod patches_rows_3813;
mod patches_step_dispatch;
mod reference;
mod regressions;
mod step_dispatch;
mod truncated_backward_3813;
mod unsupported_fallback;
