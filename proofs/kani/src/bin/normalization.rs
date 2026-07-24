// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Kani coverage for the centered-normalization forward-mode remainder helper.

#![cfg_attr(kani, allow(dead_code))]
#![cfg_attr(not(kani), allow(dead_code, unused_imports, unused_variables))]

#[cfg(kani)]
use kani::{any, assume};

#[cfg(not(kani))]
mod kani_stub {
    pub fn any<T: Default>() -> T {
        Default::default()
    }

    pub fn assume(_cond: bool) {}
}

#[cfg(not(kani))]
use kani_stub::{any, assume};

fn main() {}

fn centered_norm_projector_entry(same_index: bool, norm_size: usize) -> f64 {
    let inv_n = 1.0 / norm_size as f64;
    if same_index {
        1.0 - inv_n
    } else {
        -inv_n
    }
}

fn sqrt_norm_size(norm_size: usize) -> f64 {
    match norm_size {
        1 => 1.0,
        2 => 1.414_213_562_373_095_1,
        3 => 1.732_050_807_568_877_2,
        4 => 2.0,
        5 => 2.236_067_977_499_79,
        6 => 2.449_489_742_783_178,
        7 => 2.645_751_311_064_590_7,
        8 => 2.828_427_124_746_190_3,
        9 => 3.0,
        _ => f64::INFINITY,
    }
}

fn centered_norm_hessian_bracket_entry(
    z_i: f64,
    z_j: f64,
    z_k: f64,
    same_ij: bool,
    same_ik: bool,
    same_jk: bool,
    norm_size: usize,
) -> f64 {
    let inv_n = 1.0 / norm_size as f64;
    let projector_ij = centered_norm_projector_entry(same_ij, norm_size);
    let projector_ik = centered_norm_projector_entry(same_ik, norm_size);
    let projector_jk = centered_norm_projector_entry(same_jk, norm_size);
    -projector_ij * z_k - projector_ik * z_j - z_i * projector_jk + 3.0 * z_i * z_j * z_k * inv_n
}

fn centered_norm_hessian_entry_abs(
    abs_gamma: f64,
    sigma: f64,
    z_i: f64,
    z_j: f64,
    z_k: f64,
    same_ij: bool,
    same_ik: bool,
    same_jk: bool,
    norm_size: usize,
) -> f64 {
    if !abs_gamma.is_finite() || abs_gamma < 0.0 || !sigma.is_finite() || sigma <= 0.0 {
        return f64::INFINITY;
    }
    let bracket =
        centered_norm_hessian_bracket_entry(z_i, z_j, z_k, same_ij, same_ik, same_jk, norm_size);
    abs_gamma * bracket.abs() / (norm_size as f64 * sigma * sigma)
}

fn centered_norm_radius_sq_sum(radius: &[f64]) -> f64 {
    radius.iter().map(|value| value * value).sum()
}

fn centered_norm_second_order_remainder(
    abs_gamma: f64,
    radius_sq_sum: f64,
    norm_size: usize,
    sigma_min: f64,
) -> f64 {
    if !abs_gamma.is_finite()
        || abs_gamma < 0.0
        || !radius_sq_sum.is_finite()
        || radius_sq_sum < 0.0
        || !sigma_min.is_finite()
        || sigma_min <= 0.0
    {
        return f64::INFINITY;
    }
    let sqrt_n = sqrt_norm_size(norm_size);
    if !sqrt_n.is_finite() {
        return f64::INFINITY;
    }
    3.5 * abs_gamma * radius_sq_sum / (sqrt_n * sigma_min * sigma_min)
}

fn centered_norm_exact_hessian_contraction(
    abs_gamma: f64,
    sigma: f64,
    z: &[f64],
    radius: &[f64],
    output_index: usize,
) -> f64 {
    let norm_size = z.len();
    if norm_size == 0 || radius.len() != norm_size || output_index >= norm_size {
        return f64::INFINITY;
    }
    let mut total = 0.0_f64;
    for (j, radius_j) in radius.iter().enumerate() {
        for (k, radius_k) in radius.iter().enumerate() {
            let h_abs = centered_norm_hessian_entry_abs(
                abs_gamma,
                sigma,
                z[output_index],
                z[j],
                z[k],
                output_index == j,
                output_index == k,
                j == k,
                norm_size,
            );
            total += h_abs * radius_j.abs() * radius_k.abs();
        }
    }
    0.5 * total
}

fn any_bounded_i8(min: i8, max: i8) -> i8 {
    let value: i8 = any();
    assume(value >= min && value <= max);
    value
}

fn any_selector(limit: u8) -> u8 {
    let selector: u8 = any();
    assume(selector < limit);
    selector
}

fn any_half_step(min_num: i8, max_num: i8) -> f64 {
    any_bounded_i8(min_num, max_num) as f64 / 2.0
}

fn any_nonnegative_half_step(max_num: i8) -> f64 {
    any_bounded_i8(0, max_num) as f64 / 2.0
}

fn bracket_via_jacobian_expansion(
    z_i: f64,
    z_j: f64,
    z_k: f64,
    same_ij: bool,
    same_ik: bool,
    same_jk: bool,
    norm_size: usize,
) -> f64 {
    let inv_n = 1.0 / norm_size as f64;
    let projector_ij = centered_norm_projector_entry(same_ij, norm_size);
    let projector_ik = centered_norm_projector_entry(same_ik, norm_size);
    let projector_jk = centered_norm_projector_entry(same_jk, norm_size);
    let b_ij = projector_ij - z_i * z_j * inv_n;
    let b_ik = projector_ik - z_i * z_k * inv_n;
    let b_jk = projector_jk - z_j * z_k * inv_n;
    -(z_k * b_ij + z_j * b_ik + z_i * b_jk)
}

#[cfg(kani)]
#[kani::proof]
fn centered_norm_second_order_remainder_monotone_in_sigma_min() {
    let abs_gamma = any_nonnegative_half_step(8);
    let radius_sq_sum = any_nonnegative_half_step(8);
    let sigma_small = any_half_step(1, 8);
    let sigma_large = any_half_step(1, 8);
    assume(sigma_small <= sigma_large);

    let small = centered_norm_second_order_remainder(abs_gamma, radius_sq_sum, 4, sigma_small);
    let large = centered_norm_second_order_remainder(abs_gamma, radius_sq_sum, 4, sigma_large);

    assert!(small.is_finite());
    assert!(large.is_finite());
    assert!(small >= large);
}

#[cfg(kani)]
#[kani::proof]
fn centered_norm_second_order_remainder_fails_closed_on_invalid_scale() {
    let abs_gamma = any_nonnegative_half_step(8);
    let radius_sq_sum = any_nonnegative_half_step(8);
    let result = match any_selector(4) {
        0 => centered_norm_second_order_remainder(abs_gamma, radius_sq_sum, 4, 0.0),
        1 => centered_norm_second_order_remainder(abs_gamma, radius_sq_sum, 4, f64::INFINITY),
        2 => centered_norm_second_order_remainder(abs_gamma, radius_sq_sum, 4, -0.5),
        3 => centered_norm_second_order_remainder(abs_gamma, radius_sq_sum, 0, 1.0),
        _ => unreachable!(),
    };

    assert!(result.is_infinite());
    assert!(result.is_sign_positive());
}

#[cfg(kani)]
#[kani::proof]
fn centered_norm_hessian_bracket_matches_jacobian_derivative_expansion() {
    let z_i = any_half_step(-4, 4);
    let z_j = any_half_step(-4, 4);
    let z_k = any_half_step(-4, 4);
    let same_ij: bool = any();
    let same_ik: bool = any();
    let same_jk: bool = any();

    let exact = centered_norm_hessian_bracket_entry(z_i, z_j, z_k, same_ij, same_ik, same_jk, 4);
    let expanded = bracket_via_jacobian_expansion(z_i, z_j, z_k, same_ij, same_ik, same_jk, 4);

    assert!((exact - expanded).abs() <= 1e-12);
}

fn bounded_z_witness_3() -> [f64; 3] {
    let z0 = any_bounded_i8(-3, 3);
    let z1 = any_bounded_i8(-3, 3);
    let z2 = any_bounded_i8(-3, 3);
    let norm_sq = i16::from(z0) * i16::from(z0)
        + i16::from(z1) * i16::from(z1)
        + i16::from(z2) * i16::from(z2);
    assume(norm_sq <= 12);
    [z0 as f64 / 2.0, z1 as f64 / 2.0, z2 as f64 / 2.0]
}

fn bounded_radius_witness_3() -> [f64; 3] {
    [
        any_nonnegative_half_step(4),
        any_nonnegative_half_step(4),
        any_nonnegative_half_step(4),
    ]
}

fn bounded_z_witness_4() -> [f64; 4] {
    let z0 = any_bounded_i8(-3, 3);
    let z1 = any_bounded_i8(-3, 3);
    let z2 = any_bounded_i8(-3, 3);
    let z3 = any_bounded_i8(-3, 3);
    let norm_sq = i16::from(z0) * i16::from(z0)
        + i16::from(z1) * i16::from(z1)
        + i16::from(z2) * i16::from(z2)
        + i16::from(z3) * i16::from(z3);
    assume(norm_sq <= 16);
    [
        z0 as f64 / 2.0,
        z1 as f64 / 2.0,
        z2 as f64 / 2.0,
        z3 as f64 / 2.0,
    ]
}

fn bounded_radius_witness_4() -> [f64; 4] {
    [
        any_nonnegative_half_step(4),
        any_nonnegative_half_step(4),
        any_nonnegative_half_step(4),
        any_nonnegative_half_step(4),
    ]
}

#[cfg(kani)]
#[kani::proof]
fn centered_norm_hessian_contraction_bound_n3() {
    let z = bounded_z_witness_3();
    let radius = bounded_radius_witness_3();
    let radius_sq_sum = centered_norm_radius_sq_sum(&radius);
    let remainder = centered_norm_second_order_remainder(1.0, radius_sq_sum, 3, 1.0);

    for output_index in 0..3 {
        let contraction =
            centered_norm_exact_hessian_contraction(1.0, 1.0, &z, &radius, output_index);
        assert!(contraction <= remainder + 1e-12);
    }
}

#[cfg(kani)]
#[kani::proof]
fn centered_norm_hessian_contraction_bound_n4() {
    let z = bounded_z_witness_4();
    let radius = bounded_radius_witness_4();
    let radius_sq_sum = centered_norm_radius_sq_sum(&radius);
    let remainder = centered_norm_second_order_remainder(1.0, radius_sq_sum, 4, 1.0);

    for output_index in 0..4 {
        let contraction =
            centered_norm_exact_hessian_contraction(1.0, 1.0, &z, &radius, output_index);
        assert!(contraction <= remainder + 1e-12);
    }
}
