// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{ArrayD, IxDyn};
use ny_tensor::BoundedTensor;

use super::super::*;
use super::core::assert_bounds_no_nan_ordered;

/// Build non-uniform BoundedTensor from sinusoidal patterns.
fn make_sinusoidal_bounds(
    rows: usize,
    cols: usize,
    center_amp: f32,
    center_freq: f32,
    width_base: f32,
    width_var: f32,
    width_freq: f32,
) -> BoundedTensor {
    let n = rows * cols;
    let lower: Vec<f32> = (0..n)
        .map(|i| {
            let c = center_amp * ((i as f32 * center_freq).sin());
            let w = width_base + width_var * ((i as f32 * width_freq).cos().abs());
            c - w
        })
        .collect();
    let upper: Vec<f32> = (0..n)
        .map(|i| {
            let c = center_amp * ((i as f32 * center_freq).sin());
            let w = width_base + width_var * ((i as f32 * width_freq).cos().abs());
            c + w
        })
        .collect();
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[rows, cols]), lower).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[rows, cols]), upper).unwrap(),
    )
    .unwrap()
}

/// Build mixed-sign downstream with alternating positive/negative pattern.
fn make_alternating_downstream(
    out_dim: usize,
    z_size: usize,
) -> (ArrayD<f32>, ArrayD<f32>, ArrayD<f32>, ArrayD<f32>) {
    let mut ds_la = ArrayD::zeros(IxDyn(&[out_dim, z_size]));
    let mut ds_ua = ArrayD::zeros(IxDyn(&[out_dim, z_size]));
    for o in 0..out_dim {
        for z_idx in 0..z_size {
            let sign = if (o + z_idx) % 3 == 0 { 1.0 } else { -1.0 };
            let mag = 0.05 + 0.03 * ((z_idx as f32 * 0.7).sin().abs());
            ds_la[[o, z_idx]] = sign * mag;
            ds_ua[[o, z_idx]] = sign * mag * 1.2;
        }
    }
    let ds_lb = ArrayD::from_elem(IxDyn(&[out_dim]), -0.01_f32);
    let ds_ub = ArrayD::from_elem(IxDyn(&[out_dim]), 0.01_f32);
    (ds_la, ds_lb, ds_ua, ds_ub)
}

/// Compose IBP bounds with downstream via interval arithmetic (baseline).
fn compose_ibp_downstream(
    z_ibp: &BoundedTensor,
    ds_la: &ArrayD<f32>,
    ds_lb: &ArrayD<f32>,
    ds_ua: &ArrayD<f32>,
    ds_ub: &ArrayD<f32>,
    out_dim: usize,
    z_size: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut lo_vec = vec![0.0_f64; out_dim];
    let mut hi_vec = vec![0.0_f64; out_dim];
    let z_l: Vec<f32> = z_ibp.lower().iter().copied().collect();
    let z_u: Vec<f32> = z_ibp.upper().iter().copied().collect();
    for o in 0..out_dim {
        let (mut lo, mut hi) = (ds_lb[o] as f64, ds_ub[o] as f64);
        for z_idx in 0..z_size {
            let (al, au) = (ds_la[[o, z_idx]] as f64, ds_ua[[o, z_idx]] as f64);
            let (zl, zu) = (z_l[z_idx] as f64, z_u[z_idx] as f64);
            lo += al.max(0.0) * zl + al.min(0.0) * zu;
            hi += au.max(0.0) * zu + au.min(0.0) * zl;
        }
        lo_vec[o] = lo;
        hi_vec[o] = hi;
    }
    (lo_vec, hi_vec)
}

/// Flatten a BoundedTensor to 1D for concretization.
fn flatten_bounds(bt: &BoundedTensor, total: usize) -> BoundedTensor {
    BoundedTensor::new(
        bt.lower()
            .clone()
            .into_shape_with_order(IxDyn(&[total]))
            .unwrap(),
        bt.upper()
            .clone()
            .into_shape_with_order(IxDyn(&[total]))
            .unwrap(),
    )
    .unwrap()
}

/// Test BilinearCrown broadcast beyond the old seq=64 limit (#286).
///
/// Key acceptance test: identity_for_attention blocked seq > 64 due to
/// O(seq^4) dense identity. BilinearRelaxation avoids this entirely.
#[ntest::timeout(120000)]
#[test]
fn test_bilinear_broadcast_seq96_beyond_limit() {
    let (batch, heads, seq, d_k) = (1, 2, 96, 32);
    let (m, n, k) = (seq, seq, d_k);
    let z_size = m * n;
    let out_dim = 2;

    let q_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[batch, heads, m, k]), -0.5_f32),
        ArrayD::from_elem(IxDyn(&[batch, heads, m, k]), 0.5_f32),
    )
    .unwrap();
    let k_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[batch, heads, n, k]), -0.3_f32),
        ArrayD::from_elem(IxDyn(&[batch, heads, n, k]), 0.3_f32),
    )
    .unwrap();
    let downstream = crate::BatchedLinearBounds::new(
        ArrayD::from_elem(IxDyn(&[batch, heads, out_dim, z_size]), 0.01_f32),
        ArrayD::from_elem(IxDyn(&[batch, heads, out_dim]), -0.05_f32),
        ArrayD::from_elem(IxDyn(&[batch, heads, out_dim, z_size]), 0.02_f32),
        ArrayD::from_elem(IxDyn(&[batch, heads, out_dim]), 0.05_f32),
        vec![batch, heads, z_size],
        vec![batch, heads, out_dim],
    )
    .unwrap();

    let layer = BilinearCrownLayer::new(true, Some(1.0 / (d_k as f32).sqrt()));
    let (bq, bk) = layer
        .propagate_linear_batched_binary(&downstream, &q_bounds, &k_bounds)
        .expect("BilinearCrown at seq=96 should succeed (#286)");

    assert_eq!(bq.lower_a().shape(), &[batch, heads, out_dim, m * k]);
    assert_eq!(bk.lower_a().shape(), &[batch, heads, out_dim, n * k]);
    assert_bounds_no_nan_ordered("Q", &bq);
    assert_bounds_no_nan_ordered("K", &bk);
}

/// R1 stall checkpoint: broadcast McCormick CROWN vs IBP (#3320).
///
/// Reference: designs/2026-03-04-286-attention-bilinear-alternative.md Phase 3
#[ntest::timeout(60000)]
#[test]
fn test_broadcast_mccormick_vs_ibp_stall_gate_3320() {
    let (m, n, k) = (8, 8, 4);
    let scale = 1.0 / (k as f32).sqrt();
    let z_size = m * n;
    let out_dim = 4;

    let q_bounds = make_sinusoidal_bounds(m, k, 0.3, 1.7, 0.1, 0.05, 0.9);
    let k_bounds = make_sinusoidal_bounds(n, k, 0.2, 2.3, 0.08, 0.04, 1.1);

    let layer = BilinearCrownLayer::new(true, Some(scale));
    let z_ibp = layer.propagate_ibp_binary(&q_bounds, &k_bounds).unwrap();

    let (ds_la, ds_lb, ds_ua, ds_ub) = make_alternating_downstream(out_dim, z_size);
    let downstream = crate::BatchedLinearBounds::new(
        ds_la.clone(),
        ds_lb.clone(),
        ds_ua.clone(),
        ds_ub.clone(),
        vec![z_size],
        vec![out_dim],
    )
    .unwrap();

    let (ibp_lo, ibp_hi) =
        compose_ibp_downstream(&z_ibp, &ds_la, &ds_lb, &ds_ua, &ds_ub, out_dim, z_size);

    let (bq, bk) = layer
        .propagate_linear_batched_binary(&downstream, &q_bounds, &k_bounds)
        .expect("broadcast CROWN should succeed");
    assert_bounds_no_nan_ordered("Q_stall_gate", &bq);
    assert_bounds_no_nan_ordered("K_stall_gate", &bk);

    let concrete_q = bq.concretize(&flatten_bounds(&q_bounds, m * k)).unwrap();
    let concrete_k = bk.concretize(&flatten_bounds(&k_bounds, n * k)).unwrap();

    let (mut total_crown, mut total_ibp) = (0.0_f64, 0.0_f64);
    for o in 0..out_dim {
        let cw = (concrete_q.upper()[[o]] + concrete_k.upper()[[o]]) as f64
            - (concrete_q.lower()[[o]] + concrete_k.lower()[[o]]) as f64;
        total_crown += cw;
        total_ibp += ibp_hi[o] - ibp_lo[o];
    }

    eprintln!(
        "#3320 stall gate: IBP={total_ibp:.6}, CROWN={total_crown:.6}, \
         ratio={:.4}",
        total_crown / total_ibp.max(1e-12)
    );
    assert!(
        total_crown <= total_ibp + 1e-3,
        "#3320: CROWN ({total_crown:.6}) > IBP ({total_ibp:.6})"
    );
}

/// Scalability: broadcast McCormick at seq=128 (#3320 AC4).
///
/// Reference: designs/2026-03-04-286-attention-bilinear-alternative.md
#[ntest::timeout(120000)]
#[test]
fn test_bilinear_broadcast_seq128_scalability_3320() {
    let (batch, heads, seq, d_k) = (1, 2, 128, 32);
    let (m, n, k) = (seq, seq, d_k);
    let z_size = m * n;
    let out_dim = 2;

    let q_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[batch, heads, m, k]), -0.3_f32),
        ArrayD::from_elem(IxDyn(&[batch, heads, m, k]), 0.3_f32),
    )
    .unwrap();
    let k_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[batch, heads, n, k]), -0.2_f32),
        ArrayD::from_elem(IxDyn(&[batch, heads, n, k]), 0.2_f32),
    )
    .unwrap();
    let downstream = crate::BatchedLinearBounds::new(
        ArrayD::from_elem(IxDyn(&[batch, heads, out_dim, z_size]), 0.01_f32),
        ArrayD::from_elem(IxDyn(&[batch, heads, out_dim]), -0.05_f32),
        ArrayD::from_elem(IxDyn(&[batch, heads, out_dim, z_size]), 0.02_f32),
        ArrayD::from_elem(IxDyn(&[batch, heads, out_dim]), 0.05_f32),
        vec![batch, heads, z_size],
        vec![batch, heads, out_dim],
    )
    .unwrap();

    let layer = BilinearCrownLayer::new(true, Some(1.0 / (d_k as f32).sqrt()));
    let (bq, bk) = layer
        .propagate_linear_batched_binary(&downstream, &q_bounds, &k_bounds)
        .expect("BilinearCrown at seq=128 should succeed (#3320)");

    assert_eq!(bq.lower_a().shape(), &[batch, heads, out_dim, m * k]);
    assert_eq!(bk.lower_a().shape(), &[batch, heads, out_dim, n * k]);
    assert_bounds_no_nan_ordered("Q_seq128", &bq);
    assert_bounds_no_nan_ordered("K_seq128", &bk);

    // Verify O(seq*d_k) per-output storage, NOT O(seq^4).
    let total_q = bq.lower_a().len() + bq.upper_a().len();
    let max_elements = batch * heads * out_dim * m * k * 2;
    assert!(total_q <= max_elements, "O(seq^4) detected: {total_q}");
    eprintln!("#3320 AC4: seq={seq} OK. Q_elements={total_q}");
}
