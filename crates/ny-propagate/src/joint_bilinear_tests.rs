// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn typed_preset_bridge_is_default_silent_and_env_wins_both_directions() {
    assert_eq!(decode_bool_preset(encode_bool_preset(None)), None);
    assert_eq!(
        decode_bool_preset(encode_bool_preset(Some(false))),
        Some(false)
    );
    assert_eq!(
        decode_bool_preset(encode_bool_preset(Some(true))),
        Some(true)
    );

    let decl = &ny_levers::decls::dark_probes::JOINT_BILINEAR_QUAD;
    let resolve = |preset: Option<bool>, raw: Option<&str>| {
        let raw = raw.map(str::to_owned);
        ny_levers::read_over_config_with(
            decl,
            move |_| raw,
            preset.map(ny_levers::LeverValue::Bool),
        )
        .expect("bool preset must be admissible")
    };

    let absent = resolve(None, None);
    assert!(!absent.value.as_bool());
    assert_eq!(absent.source, ny_levers::Source::Default);

    let off = resolve(Some(false), None);
    assert!(!off.value.as_bool());
    assert_eq!(off.source, ny_levers::Source::Config);

    let on = resolve(Some(true), None);
    assert!(on.value.as_bool());
    assert_eq!(on.source, ny_levers::Source::Config);

    let env_off = resolve(Some(true), Some("0"));
    assert!(!env_off.value.as_bool());
    assert_eq!(env_off.source, ny_levers::Source::LegacyEnv);

    let env_on = resolve(Some(false), Some("1"));
    assert!(env_on.value.as_bool());
    assert_eq!(env_on.source, ny_levers::Source::LegacyEnv);
}

/// Enumerate a fine grid over the box and check the enclosure really encloses.
fn grid_max_min(
    a_coeff: &[f64],
    a_off: &[f64],
    b_coeff: &[f64],
    b_off: &[f64],
    rows: usize,
    cols: usize,
    lower: &[f64],
    upper: &[f64],
    steps: usize,
) -> (f64, f64) {
    let mut best_low = f64::INFINITY;
    let mut best_high = f64::NEG_INFINITY;
    let total = steps.pow(cols as u32);
    for index in 0..total {
        let mut point = vec![0.0f64; cols];
        let mut rest = index;
        for (j, value) in point.iter_mut().enumerate().take(cols) {
            let step = rest % steps;
            rest /= steps;
            let t = step as f64 / (steps - 1) as f64;
            *value = (1.0 - t) * lower[j] + t * upper[j];
        }
        let mut total_value = 0.0f64;
        for i in 0..rows {
            let mut av = a_off[i];
            let mut bv = b_off[i];
            for j in 0..cols {
                av += a_coeff[i * cols + j] * point[j];
                bv += b_coeff[i * cols + j] * point[j];
            }
            total_value += av * bv;
        }
        best_low = best_low.min(total_value);
        best_high = best_high.max(total_value);
    }
    (best_low, best_high)
}

fn exact_affine(coeff: Vec<f64>, offset: Vec<f64>, rows: usize, cols: usize) -> MidpointAffine {
    MidpointAffine {
        coeff,
        offset,
        radius: vec![0.0; rows],
        magnitude: vec![0.0; rows],
        rows,
        cols,
    }
}

fn with_magnitudes(mut affine: MidpointAffine, mid: &[f64], half: &[f64]) -> MidpointAffine {
    for i in 0..affine.rows {
        let mut value = affine.offset[i];
        let mut dev = 0.0f64;
        for j in 0..affine.cols {
            value += affine.coeff[i * affine.cols + j] * mid[j];
            dev += affine.coeff[i * affine.cols + j].abs() * half[j];
        }
        affine.magnitude[i] = value.abs() + dev;
    }
    affine
}

#[test]
fn joint_sum_bound_encloses_the_true_range_on_exact_affine_factors() {
    // Deliberately correlated: B is a scaled/shifted copy of A, so the
    // factor-wise product of the two ranges is far outside the truth.
    let rows = 3usize;
    let cols = 3usize;
    let a_coeff = vec![
        1.0, -0.5, 0.25, //
        0.3, 0.8, -0.2, //
        -0.7, 0.1, 0.9,
    ];
    let a_off = vec![0.1, -0.2, 0.05];
    let b_coeff = vec![
        -1.0, 0.5, -0.25, //
        0.6, 1.6, -0.4, //
        0.7, -0.1, -0.9,
    ];
    let b_off = vec![-0.1, 0.2, 0.15];
    let lower = vec![-1.0, -0.5, -2.0];
    let upper = vec![1.5, 0.75, 0.5];
    let mid: Vec<f64> = lower
        .iter()
        .zip(upper.iter())
        .map(|(l, u)| 0.5 * (l + u))
        .collect();
    let half: Vec<f64> = lower
        .iter()
        .zip(upper.iter())
        .map(|(l, u)| 0.5 * (u - l))
        .collect();
    let a = with_magnitudes(
        exact_affine(a_coeff.clone(), a_off.clone(), rows, cols),
        &mid,
        &half,
    );
    let b = with_magnitudes(
        exact_affine(b_coeff.clone(), b_off.clone(), rows, cols),
        &mid,
        &half,
    );
    let (low, high) = joint_sum_bound(&a, &b, &mid, &half).expect("joint bound");
    let (true_low, true_high) = grid_max_min(
        &a_coeff, &a_off, &b_coeff, &b_off, rows, cols, &lower, &upper, 25,
    );
    assert!(
        low <= true_low + 1e-9,
        "joint lower {low} is above a realised value {true_low}"
    );
    assert!(
        high >= true_high - 1e-9,
        "joint upper {high} is below a realised value {true_high}"
    );
}

#[test]
fn joint_sum_bound_beats_the_factor_wise_product_on_a_shared_variable() {
    // sum_i A_i B_i with B = -A: the true maximum is <= 0 everywhere, while
    // the factor-wise interval product is strictly positive.
    let rows = 2usize;
    let cols = 2usize;
    let a_coeff = vec![1.0, 0.0, 0.0, 1.0];
    let a_off = vec![0.0, 0.0];
    let b_coeff = vec![-1.0, 0.0, 0.0, -1.0];
    let b_off = vec![0.0, 0.0];
    let mid = vec![0.0, 0.0];
    let half = vec![1.0, 1.0];
    let a = with_magnitudes(exact_affine(a_coeff, a_off, rows, cols), &mid, &half);
    let b = with_magnitudes(exact_affine(b_coeff, b_off, rows, cols), &mid, &half);
    let (low, high) = joint_sum_bound(&a, &b, &mid, &half).expect("joint bound");
    assert!(low <= -2.0 + 1e-9, "lower {low} misses the true minimum -2");
    assert!(
        high <= 1e-6,
        "joint upper {high} should be ~0, not positive"
    );
    // Factor-wise would be max over the four corner products of [-1,1]*[-1,1]
    // summed = +2. The joint bound must be dramatically better than that.
    assert!(high < 2.0);
}

#[test]
fn quadratic_deviation_max_is_sound_for_an_indefinite_form() {
    let n = 2usize;
    let h = vec![1.0, 2.0, 2.0, -1.0];
    let half = vec![1.0, 1.0];
    let bound = quadratic_deviation_max(&h, n, &half);
    let mut worst = f64::NEG_INFINITY;
    for i in 0..=100 {
        for j in 0..=100 {
            let x = -1.0 + 2.0 * (i as f64) / 100.0;
            let y = -1.0 + 2.0 * (j as f64) / 100.0;
            let value = h[0] * x * x + 2.0 * h[1] * x * y + h[3] * y * y;
            worst = worst.max(value);
        }
    }
    assert!(
        bound >= worst - 1e-9,
        "curvature bound {bound} is below a realised value {worst}"
    );
}

#[test]
fn jacobi_residual_is_charged_so_a_truncated_sweep_stays_sound() {
    // A matrix whose Jacobi sweep is asked to converge is still bounded
    // soundly, because the residual term is computed from the actual output.
    let n = 4usize;
    let mut h = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            h[i * n + j] = ((i * 7 + j * 13) % 11) as f64 - 5.0;
        }
    }
    // symmetrize
    for i in 0..n {
        for j in 0..i {
            let s = 0.5 * (h[i * n + j] + h[j * n + i]);
            h[i * n + j] = s;
            h[j * n + i] = s;
        }
    }
    let half = vec![0.5, 1.0, 0.25, 2.0];
    let bound = quadratic_deviation_max(&h, n, &half);
    let mut worst = f64::NEG_INFINITY;
    let steps = 9usize;
    for index in 0..steps.pow(n as u32) {
        let mut rest = index;
        let mut point = vec![0.0f64; n];
        for (j, value) in point.iter_mut().enumerate() {
            let step = rest % steps;
            rest /= steps;
            let t = step as f64 / (steps - 1) as f64;
            *value = -half[j] + 2.0 * half[j] * t;
        }
        let mut value = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                value += h[i * n + j] * point[i] * point[j];
            }
        }
        worst = worst.max(value);
    }
    assert!(
        bound >= worst - 1e-9,
        "bound {bound} below realised {worst}"
    );
}

#[test]
fn next_up_and_down_step_outward() {
    assert!(next_up_f32(1.0) > 1.0);
    assert!(next_down_f32(1.0) < 1.0);
    assert!(next_up_f32(-1.0) > -1.0);
    assert!(next_down_f32(-1.0) < -1.0);
    assert!(next_up_f32(0.0) > 0.0);
    assert!(next_down_f32(0.0) < 0.0);
}

#[test]
fn lever_is_declared_dark() {
    // Read the DECLARATION, not the process environment: spelling the raw
    // variable name in a `.rs` file is exactly what the ny-levers ratchet
    // counts, and a test is where a copy-paste template would live.
    assert!(matches!(
        ny_levers::decls::dark_probes::JOINT_BILINEAR_QUAD.default,
        ny_levers::DefaultSpec::Bool(false)
    ));
}
