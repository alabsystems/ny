// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use ndarray::{ArrayD, IxDyn};
use ny_propagate::types::BoundsProvenance;
use ny_propagate::{GraphNode, Layer};
use std::time::{Duration, Instant};

mod compositional;
mod cosine_head;
mod crown_ibp_diag;
mod crown_ibp_tightening;
mod graph_smoke;
mod shared;
mod training_signal;
pub(crate) mod verifier_smoke;

pub(crate) use self::shared::{avoice_speaker_encoder_graph, SPEAKER_ENCODER_SEQUENCE_LEN};
