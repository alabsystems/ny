// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Exact rational reference evaluation of the ground-truth residuals.
//!
//! These functions mirror the [`crate::builders`] residual definitions but
//! evaluate them in exact arbitrary-precision rational arithmetic
//! ([`BigRational`]), then round once (correctly) to f64 on return. They are
//! the gold oracle for the golden tests and the exact-reference backend of
//! `ny gt eval` (via [`crate::sidecar::GroundTruthSpec::reference_eval`]): a
//! ground-truth graph evaluated at a zero-width point must produce a sound
//! enclosure containing the reference value.
//!
//! All arguments must be finite; each function panics otherwise (these are
//! test/authoring oracles, not the validated builder surface — the builders
//! reject non-finite parameters with typed errors).

use num_rational::BigRational;
use num_traits::ToPrimitive;

/// Exact rational of a finite f64.
pub(crate) fn rat(name: &str, v: f64) -> BigRational {
    BigRational::from_float(v).unwrap_or_else(|| panic!("{name} must be finite, got {v}"))
}

pub(crate) fn rat3(name: &str, v: [f64; 3]) -> [BigRational; 3] {
    [rat(name, v[0]), rat(name, v[1]), rat(name, v[2])]
}

fn sub3(a: &[BigRational; 3], b: &[BigRational; 3]) -> [BigRational; 3] {
    [&a[0] - &b[0], &a[1] - &b[1], &a[2] - &b[2]]
}

fn dot3(a: &[BigRational; 3], b: &[BigRational; 3]) -> BigRational {
    &a[0] * &b[0] + &a[1] * &b[1] + &a[2] * &b[2]
}

fn norm_sq3(a: &[BigRational; 3]) -> BigRational {
    dot3(a, a)
}

/// Component of `u` orthogonal to `axis` (assumes `‖axis‖ = 1`):
/// `u − (u·axis) axis`.
fn perp3(u: &[BigRational; 3], axis: &[BigRational; 3]) -> [BigRational; 3] {
    let t = dot3(u, axis);
    [
        &u[0] - &(&t * &axis[0]),
        &u[1] - &(&t * &axis[1]),
        &u[2] - &(&t * &axis[2]),
    ]
}

fn to_f64(q: &BigRational) -> f64 {
    q.to_f64()
        .expect("rational fits in f64 range for these residuals")
}

// --- exact rational residuals (shared by the f64 oracles and the sidecar) --

/// Exact `n·x + d` over rationals.
pub(crate) fn signed_plane_distance_rat(
    normal: &[BigRational; 3],
    offset: &BigRational,
    x: &[BigRational; 3],
) -> BigRational {
    dot3(normal, x) + offset
}

/// Exact `‖x−c‖² − r²` over rationals.
pub(crate) fn sphere_residual_rat(
    center: &[BigRational; 3],
    radius: &BigRational,
    x: &[BigRational; 3],
) -> BigRational {
    let u = sub3(x, center);
    norm_sq3(&u) - radius * radius
}

/// Exact `‖(x−p) − ((x−p)·a)a‖² − r²` over rationals (unit `a`).
pub(crate) fn cylinder_residual_rat(
    axis: &[BigRational; 3],
    point: &[BigRational; 3],
    radius: &BigRational,
    x: &[BigRational; 3],
) -> BigRational {
    let u = sub3(x, point);
    norm_sq3(&perp3(&u, axis)) - radius * radius
}

/// Exact `cos²α·‖x−q‖² − ((x−q)·a)²` over rationals (unit `a`).
pub(crate) fn cone_residual_rat(
    axis: &[BigRational; 3],
    apex: &[BigRational; 3],
    cos_half_angle_sq: &BigRational,
    x: &[BigRational; 3],
) -> BigRational {
    let u = sub3(x, apex);
    let t = dot3(&u, axis);
    cos_half_angle_sq * norm_sq3(&u) - &t * &t
}

/// Exact `(‖x−p‖² + R² − r²)² − 4R²·‖(x−p) − ((x−p)·a)a‖²` over rationals
/// (unit `a`).
pub(crate) fn torus_residual_rat(
    axis: &[BigRational; 3],
    center: &[BigRational; 3],
    major_radius: &BigRational,
    minor_radius: &BigRational,
    x: &[BigRational; 3],
) -> BigRational {
    let u = sub3(x, center);
    let s = norm_sq3(&u) + major_radius * major_radius - minor_radius * minor_radius;
    let four = BigRational::from_float(4.0).expect("4 is finite");
    let perp = perp3(&u, axis);
    &s * &s - four * major_radius * major_radius * norm_sq3(&perp)
}

// --- f64 oracle wrappers (round the exact value once) ----------------------

/// Reference `n·x + d` (see [`crate::builders::signed_plane_distance`]).
pub fn signed_plane_distance(normal: [f64; 3], offset: f64, x: [f64; 3]) -> f64 {
    to_f64(&signed_plane_distance_rat(
        &rat3("normal", normal),
        &rat("offset", offset),
        &rat3("x", x),
    ))
}

/// Reference `‖x−c‖² − r²` (see [`crate::builders::sphere_residual`]).
pub fn sphere_residual(center: [f64; 3], radius: f64, x: [f64; 3]) -> f64 {
    to_f64(&sphere_residual_rat(
        &rat3("center", center),
        &rat("radius", radius),
        &rat3("x", x),
    ))
}

/// Reference `‖(x−p) − ((x−p)·a)a‖² − r²` for unit `a`
/// (see [`crate::builders::cylinder_residual`]).
pub fn cylinder_residual(axis: [f64; 3], point: [f64; 3], radius: f64, x: [f64; 3]) -> f64 {
    to_f64(&cylinder_residual_rat(
        &rat3("axis", axis),
        &rat3("point", point),
        &rat("radius", radius),
        &rat3("x", x),
    ))
}

/// Reference `cos²α·‖x−q‖² − ((x−q)·a)²` for unit `a`
/// (see [`crate::builders::cone_residual`]).
pub fn cone_residual(axis: [f64; 3], apex: [f64; 3], cos_half_angle_sq: f64, x: [f64; 3]) -> f64 {
    to_f64(&cone_residual_rat(
        &rat3("axis", axis),
        &rat3("apex", apex),
        &rat("cos_half_angle_sq", cos_half_angle_sq),
        &rat3("x", x),
    ))
}

/// Reference `(‖x−p‖² + R² − r²)² − 4R²·‖(x−p) − ((x−p)·a)a‖²` for unit `a`
/// (see [`crate::builders::torus_residual`]).
pub fn torus_residual(
    axis: [f64; 3],
    center: [f64; 3],
    major_radius: f64,
    minor_radius: f64,
    x: [f64; 3],
) -> f64 {
    to_f64(&torus_residual_rat(
        &rat3("axis", axis),
        &rat3("center", center),
        &rat("major_radius", major_radius),
        &rat("minor_radius", minor_radius),
        &rat3("x", x),
    ))
}
