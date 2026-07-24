// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

/// Compute interpolated McCormick relaxation coefficients for bilinear z = x * y.
///
/// For z = x * y where x ∈ [x_l, x_u] and y ∈ [y_l, y_u], the standard McCormick
/// relaxation gives two valid lower bounds (L1, L2) and two upper bounds (U1, U2):
/// - L1: z ≥ y_l*x + x_l*y - x_l*y_l  (tight at (x_l, y_l))
/// - L2: z ≥ y_u*x + x_u*y - x_u*y_u  (tight at (x_u, y_u))
///
/// This function interpolates between these planes using r ∈ [0, 1]:
/// - r_l = 0: Uses L2 plane (tight at upper corner)
/// - r_l = 1: Uses L1 plane (tight at lower corner)
/// - 0 < r_l < 1: Convex combination interpolating from L2 towards L1
///
/// Similarly for upper bounds with r_u (U2 at r=0, U1 at r=1).
///
/// Note: This matches auto_LiRPA convention where torch.ones() initialization
/// starts optimization from L1/U1 planes (r=1).
///
/// # Returns
/// (alpha_l, beta_l, ny_l, alpha_u, beta_u, ny_u) where:
/// - Lower: z ≥ alpha_l*x + beta_l*y + ny_l
/// - Upper: z ≤ alpha_u*x + beta_u*y + ny_u
///
/// # Reference
/// auto_LiRPA/operators/bivariate.py:MulHelper.interpolated_relaxation
#[inline]
pub(super) fn interpolated_mccormick(
    x_l: f32,
    x_u: f32,
    y_l: f32,
    y_u: f32,
    r_l: f32,
    r_u: f32,
) -> (f32, f32, f32, f32, f32, f32) {
    // Lower bound interpolation (L2 at r=0, L1 at r=1)
    // L1: z ≥ y_l*x + x_l*y - x_l*y_l  (coeffs: y_l, x_l, -x_l*y_l)
    // L2: z ≥ y_u*x + x_u*y - x_u*y_u  (coeffs: y_u, x_u, -x_u*y_u)
    let alpha_l = (y_l - y_u) * r_l + y_u; // coef for x
    let beta_l = (x_l - x_u) * r_l + x_u; // coef for y
    let ny_l = (y_u * x_u - y_l * x_l) * r_l - y_u * x_u; // bias

    // Upper bound interpolation (U2 at r=0, U1 at r=1)
    // U1: z ≤ y_u*x + x_l*y - x_l*y_u  (coeffs: y_u, x_l, -x_l*y_u)
    // U2: z ≤ y_l*x + x_u*y - x_u*y_l  (coeffs: y_l, x_u, -x_u*y_l)
    let alpha_u = (y_u - y_l) * r_u + y_l; // coef for x
    let beta_u = (x_l - x_u) * r_u + x_u; // coef for y
    let ny_u = (y_l * x_u - y_u * x_l) * r_u - y_l * x_u; // bias

    (alpha_l, beta_l, ny_l, alpha_u, beta_u, ny_u)
}
