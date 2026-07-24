// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Verifier-local Kokoro forward-STFT graphs.
//!
//! Source contract:
//! - external model-translation reference for the Kokoro forward-STFT graph

use std::f64::consts::PI;
use std::sync::OnceLock;

use ndarray::{ArrayD, IxDyn};
use ny_core::{NyError, Result};

use crate::layers::binary_ops::Atan2Layer;
use crate::layers::{
    AddConstantLayer, AddLayer, ConcatLayer, Conv1dLayer, Layer, PadLayer, PadMode,
    PowConstantLayer, SqrtLayer,
};
use crate::network::{GraphNetwork, GraphNode};

pub const KOKORO_STFT_N_FFT: usize = 20;
pub const KOKORO_STFT_HOP: usize = 5;
pub const KOKORO_STFT_PAD: usize = KOKORO_STFT_N_FFT / 2;
pub const KOKORO_STFT_FREQ_BINS: usize = KOKORO_STFT_N_FFT / 2 + 1;
pub const KOKORO_STFT_MAG_EPS: f32 = 1e-9;

/// Compute the frame count for an unbatched Kokoro waveform shaped `[1, audio_len]`.
pub(crate) fn kokoro_forward_stft_frame_count(audio_len: usize) -> Result<usize> {
    if audio_len <= KOKORO_STFT_PAD {
        return Err(NyError::InvalidSpec(format!(
            "Kokoro forward-STFT reflect pad requires audio_len > {}, got {}",
            KOKORO_STFT_PAD, audio_len
        )));
    }

    let padded_len = audio_len
        .checked_add(2 * KOKORO_STFT_PAD)
        .ok_or_else(|| NyError::InvalidSpec("Kokoro forward-STFT padded length overflow".into()))?;
    if padded_len < KOKORO_STFT_N_FFT {
        return Err(NyError::InvalidSpec(format!(
            "Kokoro forward-STFT padded length {} smaller than n_fft {}",
            padded_len, KOKORO_STFT_N_FFT
        )));
    }

    Ok((padded_len - KOKORO_STFT_N_FFT) / KOKORO_STFT_HOP + 1)
}

/// Build the magnitude half of Kokoro forward-STFT.
///
/// Input shape: `[1, audio_len]`.
/// Output shape: `[11, n_frames]`.
pub fn build_kokoro_forward_stft_magnitude_graph(audio_len: usize) -> Result<GraphNetwork> {
    let mut graph = build_kokoro_forward_stft_base_graph(audio_len)?;
    add_kokoro_forward_stft_magnitude_nodes(&mut graph)?;
    graph.set_output("stft_magnitude");
    Ok(graph)
}

/// Build the phase half of Kokoro forward-STFT.
///
/// Input shape: `[1, audio_len]`.
/// Output shape: `[11, n_frames]`.
pub fn build_kokoro_forward_stft_phase_graph(audio_len: usize) -> Result<GraphNetwork> {
    let mut graph = build_kokoro_forward_stft_base_graph(audio_len)?;
    add_kokoro_forward_stft_phase_node(&mut graph)?;
    graph.set_output("stft_phase");
    Ok(graph)
}

/// Build the full Kokoro forward-STFT surface `[magnitude, phase]`.
///
/// Input shape: `[1, audio_len]`.
/// Output shape: `[22, n_frames]`.
pub fn build_kokoro_forward_stft_full_graph(audio_len: usize) -> Result<GraphNetwork> {
    let mut graph = build_kokoro_forward_stft_base_graph(audio_len)?;
    add_kokoro_forward_stft_magnitude_nodes(&mut graph)?;
    add_kokoro_forward_stft_phase_node(&mut graph)?;
    graph.try_add_node(GraphNode::binary(
        "stft_full",
        Layer::Concat(ConcatLayer::new(0)),
        "stft_magnitude",
        "stft_phase",
    ))?;
    graph.set_output("stft_full");
    Ok(graph)
}

fn build_kokoro_forward_stft_base_graph(audio_len: usize) -> Result<GraphNetwork> {
    let padded_len = audio_len
        .checked_add(2 * KOKORO_STFT_PAD)
        .ok_or_else(|| NyError::InvalidSpec("Kokoro forward-STFT padded length overflow".into()))?;
    kokoro_forward_stft_frame_count(audio_len)?;

    let mut graph = GraphNetwork::new();
    graph.try_add_node(GraphNode::from_input(
        "reflect_pad",
        Layer::Pad(PadLayer::new(
            vec![(0, 0), (KOKORO_STFT_PAD, KOKORO_STFT_PAD)],
            PadMode::Reflect,
        )),
    ))?;
    graph.try_add_node(GraphNode::new(
        "stft_real",
        Layer::Conv1d(build_kokoro_stft_conv_layer(
            real_dft_bank(),
            "Kokoro forward-STFT real DFT bank",
            padded_len,
        )?),
        vec!["reflect_pad".to_string()],
    ))?;
    graph.try_add_node(GraphNode::new(
        "stft_imag",
        Layer::Conv1d(build_kokoro_stft_conv_layer(
            imag_dft_bank(),
            "Kokoro forward-STFT imag DFT bank",
            padded_len,
        )?),
        vec!["reflect_pad".to_string()],
    ))?;
    Ok(graph)
}

fn add_kokoro_forward_stft_magnitude_nodes(graph: &mut GraphNetwork) -> Result<()> {
    graph.try_add_node(GraphNode::new(
        "stft_real_sq",
        Layer::PowConstant(PowConstantLayer::square()),
        vec!["stft_real".to_string()],
    ))?;
    graph.try_add_node(GraphNode::new(
        "stft_imag_sq",
        Layer::PowConstant(PowConstantLayer::square()),
        vec!["stft_imag".to_string()],
    ))?;
    graph.try_add_node(GraphNode::binary(
        "stft_power",
        Layer::Add(AddLayer),
        "stft_real_sq",
        "stft_imag_sq",
    ))?;
    graph.try_add_node(GraphNode::new(
        "stft_power_eps",
        Layer::AddConstant(AddConstantLayer::new(ArrayD::from_elem(
            IxDyn(&[]),
            KOKORO_STFT_MAG_EPS,
        ))),
        vec!["stft_power".to_string()],
    ))?;
    graph.try_add_node(GraphNode::new(
        "stft_magnitude",
        Layer::Sqrt(SqrtLayer::new()),
        vec!["stft_power_eps".to_string()],
    ))?;
    Ok(())
}

fn add_kokoro_forward_stft_phase_node(graph: &mut GraphNetwork) -> Result<()> {
    graph.try_add_node(GraphNode::binary(
        "stft_phase",
        Layer::Atan2(Atan2Layer),
        "stft_imag",
        "stft_real",
    ))?;
    Ok(())
}

fn build_kokoro_stft_conv_layer(
    weights: Vec<f32>,
    label: &str,
    padded_len: usize,
) -> Result<Conv1dLayer> {
    Conv1dLayer::with_input_length(
        weights_to_kernel(
            &[KOKORO_STFT_FREQ_BINS, 1, KOKORO_STFT_N_FFT],
            weights,
            label,
        )?,
        None,
        KOKORO_STFT_HOP,
        0,
        padded_len,
    )
}

fn weights_to_kernel(shape: &[usize], weights: Vec<f32>, label: &str) -> Result<ArrayD<f32>> {
    ArrayD::from_shape_vec(IxDyn(shape), weights)
        .map_err(|err| NyError::InvalidSpec(format!("{label} shape error: {err}")))
}

fn real_dft_bank() -> Vec<f32> {
    static REAL: OnceLock<Vec<f32>> = OnceLock::new();
    REAL.get_or_init(|| build_dft_bank(false)).clone()
}

fn imag_dft_bank() -> Vec<f32> {
    static IMAG: OnceLock<Vec<f32>> = OnceLock::new();
    IMAG.get_or_init(|| build_dft_bank(true)).clone()
}

fn build_dft_bank(imaginary: bool) -> Vec<f32> {
    let window = hann_window_f64(KOKORO_STFT_N_FFT);
    let mut weights = Vec::with_capacity(
        KOKORO_STFT_FREQ_BINS
            .checked_mul(KOKORO_STFT_N_FFT)
            .expect("Kokoro forward-STFT DFT bank size should fit usize"),
    );

    for k in 0..KOKORO_STFT_FREQ_BINS {
        for (n, &window_value) in window.iter().enumerate() {
            let angle = 2.0 * PI * k as f64 * n as f64 / KOKORO_STFT_N_FFT as f64;
            let value = if imaginary { -angle.sin() } else { angle.cos() };
            weights.push((window_value as f64 * value) as f32);
        }
    }

    weights
}

fn hann_window_f64(size: usize) -> Vec<f32> {
    (0..size)
        .map(|idx| {
            let phase = 2.0 * PI * idx as f64 / size as f64;
            (0.5 * (1.0 - phase.cos())) as f32
        })
        .collect()
}
