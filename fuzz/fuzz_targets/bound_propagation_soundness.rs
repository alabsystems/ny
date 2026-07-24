// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

#![no_main]

//! Fuzz the bound-propagation engine (IBP + CROWN) for panics and SOUNDNESS
//! violations.
//!
//! The existing fuzz targets only cover the parsers (`onnx_loader`,
//! `vnnlib_parser`) and the linear-bounds concretizer. This target exercises
//! the actual verification engine: it builds a small random ReLU MLP and a
//! random input box from the fuzz input, then runs both IBP and CROWN forward
//! bound propagation.
//!
//! ## Soundness invariant
//!
//! For *any* concrete point `x` inside the input box, the network output
//! `f(x)` must lie within the propagated output bounds `[lower, upper]`. A
//! verifier that produced bounds NOT containing a reachable output would
//! silently emit a wrong "verified" verdict — the worst possible bug.
//!
//! We obtain `f(x)` by running IBP on a *degenerate* (point) box `[x, x]`: for a
//! ReLU MLP, interval propagation on a zero-width box collapses to exact forward
//! evaluation. We then assert:
//!
//!   1. `lower <= upper` for both IBP and CROWN output bounds (well-formed).
//!   2. CROWN output bounds contain the forward output at the box center and
//!      corners (CROWN must be sound).
//!   3. IBP output bounds contain the same forward outputs (IBP must be sound).
//!
//! A small absolute+relative tolerance absorbs benign f32 rounding between the
//! point-box IBP "forward eval" and the box propagation.
//!
//! NOTE: This target is written against the `ny-fuzz` crate structure and
//! mirrors the existing targets exactly. It requires a nightly toolchain plus
//! `cargo-fuzz` to actually run (`cargo +nightly fuzz run
//! bound_propagation_soundness`). If that toolchain is unavailable in the
//! current environment the target still compiles as part of the fuzz crate.

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use ndarray::{Array1, Array2};
use ny_propagate::prelude::{Layer, LinearLayer, Network, ReLULayer};
use ny_tensor::BoundedTensor;

/// Cap layer sizes so fuzzing stays fast and matrices stay small.
const MAX_DIM: usize = 6;
/// Cap depth so we don't build pathologically deep nets.
const MAX_LAYERS: usize = 4;

#[derive(Debug, Arbitrary)]
struct FuzzNet {
    /// Layer widths (input dim, then each Linear output dim). Clamped to MAX_DIM.
    widths: Vec<u8>,
    /// Flat pool of weights, consumed as needed (wrapped/padded).
    weights: Vec<f32>,
    /// Flat pool of biases.
    biases: Vec<f32>,
    /// Whether a ReLU follows each Linear layer.
    relu_flags: Vec<bool>,
    /// Input box: center coordinates and (non-negative) radii.
    centers: Vec<f32>,
    radii: Vec<f32>,
}

/// Sanitize an f32: NaN/inf would make soundness comparisons meaningless and
/// the engine legitimately rejects them. Clamp to a tame finite range.
fn sane(x: f32) -> f32 {
    if x.is_finite() {
        x.clamp(-8.0, 8.0)
    } else {
        0.0
    }
}

/// Pull `n` sanitized values from `pool`, cycling if it is too short.
fn take(pool: &[f32], start: usize, n: usize) -> Vec<f32> {
    if pool.is_empty() {
        return vec![0.0; n];
    }
    (0..n)
        .map(|i| sane(pool[(start + i) % pool.len()]))
        .collect()
}

fuzz_target!(|net: FuzzNet| {
    // Build a clamped, non-empty width sequence: [in, h1, h2, ..., out].
    let mut widths: Vec<usize> = net
        .widths
        .iter()
        .map(|&w| (w as usize % MAX_DIM) + 1)
        .take(MAX_LAYERS + 1)
        .collect();
    if widths.len() < 2 {
        // Need at least one Linear layer (an input dim and an output dim).
        widths = vec![2, 2];
    }

    let in_dim = widths[0];

    // Assemble the network: Linear (+ optional ReLU) per consecutive width pair.
    let mut network = Network::new();
    let mut w_off = 0usize;
    let mut b_off = 0usize;
    for layer_idx in 0..(widths.len() - 1) {
        let rows = widths[layer_idx + 1]; // output dim
        let cols = widths[layer_idx]; // input dim
        let w_vals = take(&net.weights, w_off, rows * cols);
        let b_vals = take(&net.biases, b_off, rows);
        w_off = w_off.wrapping_add(rows * cols);
        b_off = b_off.wrapping_add(rows);

        let weight = match Array2::from_shape_vec((rows, cols), w_vals) {
            Ok(m) => m,
            Err(_) => return, // shape derived from lengths; should not happen
        };
        let bias = Array1::from_vec(b_vals);

        let linear = match LinearLayer::new(weight, Some(bias)) {
            Ok(l) => l,
            Err(_) => return,
        };
        network.add_layer(Layer::Linear(linear));

        let relu = net.relu_flags.get(layer_idx).copied().unwrap_or(true);
        if relu {
            network.add_layer(Layer::ReLU(ReLULayer));
        }
    }

    // Build the input box from sanitized centers and non-negative radii.
    let centers = take(&net.centers, 0, in_dim);
    let radii_raw = take(&net.radii, 0, in_dim);
    let mut lower = Vec::with_capacity(in_dim);
    let mut upper = Vec::with_capacity(in_dim);
    for i in 0..in_dim {
        let c = centers[i];
        let r = radii_raw[i].abs().min(4.0);
        lower.push(c - r);
        upper.push(c + r);
    }

    let input_box = match BoundedTensor::new(
        Array1::from_vec(lower.clone()).into_dyn(),
        Array1::from_vec(upper.clone()).into_dyn(),
    ) {
        Ok(b) => b,
        Err(_) => return,
    };

    // Run both engines. Errors (e.g. shape mismatches) are acceptable; panics
    // and unsoundness are not.
    let ibp = match network.propagate_ibp(&input_box) {
        Ok(b) => b,
        Err(_) => return,
    };
    let crown = match network.propagate_crown(&input_box) {
        Ok(b) => b,
        Err(_) => return,
    };

    let ibp_lower = ibp.lower();
    let ibp_upper = ibp.upper();
    let crown_lower = crown.lower();
    let crown_upper = crown.upper();

    // Invariant 1: bounds are well-formed (lower <= upper everywhere).
    let tol_wf = 1e-3_f32;
    for (lo, up) in ibp_lower.iter().zip(ibp_upper.iter()) {
        if lo.is_finite() && up.is_finite() {
            assert!(*lo <= *up + tol_wf, "IBP lower {lo} > upper {up}");
        }
    }
    for (lo, up) in crown_lower.iter().zip(crown_upper.iter()) {
        if lo.is_finite() && up.is_finite() {
            assert!(*lo <= *up + tol_wf, "CROWN lower {lo} > upper {up}");
        }
    }

    // Sample points inside the box: center and all-low / all-high corners.
    let mut samples: Vec<Vec<f32>> = Vec::new();
    samples.push((0..in_dim).map(|i| 0.5 * (lower[i] + upper[i])).collect());
    samples.push(lower.clone());
    samples.push(upper.clone());

    let contains = |bounds_lo: &ndarray::ArrayD<f32>,
                    bounds_hi: &ndarray::ArrayD<f32>,
                    fx: &ndarray::ArrayD<f32>,
                    who: &str| {
        for ((lo, hi), v) in bounds_lo.iter().zip(bounds_hi.iter()).zip(fx.iter()) {
            if !lo.is_finite() || !hi.is_finite() || !v.is_finite() {
                continue;
            }
            let scale = lo.abs().max(hi.abs()).max(v.abs()).max(1.0);
            let tol = scale * 1e-3 + 1e-3;
            assert!(
                *v >= *lo - tol && *v <= *hi + tol,
                "{who} bounds UNSOUND: f(x)={v} not in [{lo}, {hi}] (tol={tol})"
            );
        }
    };

    for point in &samples {
        // Degenerate (zero-width) box at `point`: IBP collapses to exact forward
        // evaluation for a ReLU MLP.
        let point_box = match BoundedTensor::new(
            Array1::from_vec(point.clone()).into_dyn(),
            Array1::from_vec(point.clone()).into_dyn(),
        ) {
            Ok(b) => b,
            Err(_) => continue,
        };
        let fx = match network.propagate_ibp(&point_box) {
            Ok(b) => b,
            Err(_) => continue,
        };
        // The point-box output should itself be tight (lower ~= upper).
        let fx_lower = fx.lower();
        let fx_upper = fx.upper();

        // CROWN must contain the forward output.
        contains(&crown_lower, &crown_upper, &fx_lower, "CROWN");
        contains(&crown_lower, &crown_upper, &fx_upper, "CROWN");
        // IBP must contain it too.
        contains(&ibp_lower, &ibp_upper, &fx_lower, "IBP");
        contains(&ibp_lower, &ibp_upper, &fx_upper, "IBP");
    }
});
