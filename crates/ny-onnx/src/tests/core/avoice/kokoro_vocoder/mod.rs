// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::{Network, WeightStore};
use ndarray::{ArrayD, IxDyn};
use ny_core::LayerType;
use ny_propagate::GraphNetwork;
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// Kokoro vocoder structural verification (#3500 / #3591)
//
// `kokoro_vocoder.onnx` is a HiFi-GAN-style vocoder with three inputs:
// features [B, C, T], style [128], har [22, T].
//
// The real-weight slice freezes style/har as concrete auxiliary tensors and
// proves that the delivered export parses, graph construction fuses the
// decomposed normalization path back into `InstanceNorm1d`, and the shape
// contract holds.
//
// Full-graph IBP is NOT viable as a unit test:
//   - T=6 IBP timed out at 180005ms (180s budget, 2026-03-11 iter 1387)
//   - T=6 IBP timed out at >580s (600s budget, 2026-03-11 iter 1389)
//   - historical full-graph runtime-floor IBP probe at T=6 still timed out at
//     180002ms on 2026-03-12 before the failing always-on lane was removed
// Instead, a *prefix subgraph* containing only the first upsampling stage
// enables CPU-viable IBP and keeps the real-weight CROWN blocker localized to
// the current 1D-convolution runtime path.
//
// Reference: designs/2026-03-11-issue-3500-kokoro-vocoder-boundary-surface.md
// Reference: designs/2026-03-11-issue-3500-shallow-vocoder-subpath.md
// ---------------------------------------------------------------------------

mod boundary;
mod crossfade;
mod crossfade_support;
mod crown_crossfade;
mod crown_ibp_tightening;
mod graph_support;
mod model;
mod prefix;
mod round_trip;
mod seam;
mod structural;
mod training_signal;
pub(crate) mod verifier_smoke;

pub(super) use self::structural::kokoro_vocoder_concrete_waveform_from_ort;
