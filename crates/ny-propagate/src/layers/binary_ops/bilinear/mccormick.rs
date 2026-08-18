// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

/// Certify a McCormick plane against the exact bilinear product, in f64.
///
/// `plane(x,y) - x*y` is BILINEAR, so over the box its extremum is attained at a
/// CORNER. Evaluating the four corners therefore gives the EXACT worst-case
/// violation, not a bound on it. Push the constant by that amount and round it
/// OUTWARD, and the stored f32 plane is a genuine under-estimator (`lower`) /
/// over-estimator (`upper`) everywhere in the box.
///
/// `alpha`/`beta` are taken ALREADY ROUNDED to f32, so this also discharges their
/// storage error. `alpha*cx`, `beta*cy` and `cx*cy` are exact in f64 (f32xf32
/// needs <= 48 < 53 bits); only the two additions round, which the final outward
/// `next_down`/`next_up` covers.
#[inline]
pub(crate) fn certify_plane_constant(
    alpha: f32,
    beta: f32,
    ny: f64,
    x_l: f32,
    x_u: f32,
    y_l: f32,
    y_u: f32,
    lower: bool,
) -> f32 {
    let mut worst = 0.0f64;
    for (cx, cy) in [(x_l, y_l), (x_l, y_u), (x_u, y_l), (x_u, y_u)] {
        let plane = f64::from(alpha) * f64::from(cx) + f64::from(beta) * f64::from(cy) + ny;
        let prod = f64::from(cx) * f64::from(cy);
        let viol = if lower { plane - prod } else { prod - plane };
        if viol > worst {
            worst = viol;
        }
    }
    if lower {
        ny_tensor::next_down_f32((ny - worst) as f32)
    } else {
        ny_tensor::next_up_f32((ny + worst) as f32)
    }
}

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
///
/// Interpolated McCormick planes for `z = x*y`, CERTIFIED.
///
/// ROOT-CAUSE FIX (#mulbinary-mccormick-f32,
/// `docs/MULBINARY_MCCORMICK_F32_CANCELLATION_2026-07-28.md`): the coefficients
/// used to be built in f32, where `(x_l - x_u)*r + x_u` is catastrophically
/// cancelling — once `|x_l| < ulp(x_u)` it returns `x_u - x_u = 0` and `beta_l`
/// COLLAPSES from `x_l` to `0`, silently discarding the whole coefficient and
/// lifting the "lower" plane above the true product. Measured pre-fix on
/// `x=[-1, 2^24]`, `y=[1, 100]`, `r_l=1`: claimed lower bound `-1` at corner
/// `(-1, 100)` where the true product is `-100`. The same cancellation hits `ny`
/// through `(y_u*x_u - y_l*x_l) - y_u*x_u`.
///
/// This is NOT covered by the callers' `gamma_n_f32(n)*S` charge: that term bounds
/// the f32 ACCUMULATION depth of the `+=` loop, not coefficient CONSTRUCTION, and
/// it scales with `|alpha|` — so a coefficient collapsed to `0` is charged `0`.
///
/// In f64 every input term is EXACT (`f32 - f32` and `f32 * f32` both fit in 53
/// bits), so the interpolation carries `x_l` and `y_l*x_l` at full precision; the
/// residual f32 STORAGE error is then discharged exactly by
/// [`certify_plane_constant`].
///
/// This is the single canonical implementation — `MulBinaryLayer` delegates here
/// so the two cannot drift apart.
pub(crate) fn interpolated_mccormick(
    x_l: f32,
    x_u: f32,
    y_l: f32,
    y_u: f32,
    r_l: f32,
    r_u: f32,
) -> (f32, f32, f32, f32, f32, f32) {
    let (xl, xu, yl, yu) = (
        f64::from(x_l),
        f64::from(x_u),
        f64::from(y_l),
        f64::from(y_u),
    );
    let (rl, ru) = (f64::from(r_l), f64::from(r_u));

    // Lower bound interpolation (L2 at r=0, L1 at r=1)
    // L1: z ≥ y_l*x + x_l*y - x_l*y_l   L2: z ≥ y_u*x + x_u*y - x_u*y_u
    let alpha_l = ((yl - yu) * rl + yu) as f32; // coef for x
    let beta_l = ((xl - xu) * rl + xu) as f32; // coef for y
    let ny_l_f64 = (yu * xu - yl * xl) * rl - yu * xu; // bias

    // Upper bound interpolation (U2 at r=0, U1 at r=1)
    // U1: z ≤ y_u*x + x_l*y - x_l*y_u   U2: z ≤ y_l*x + x_u*y - x_u*y_l
    let alpha_u = ((yu - yl) * ru + yl) as f32; // coef for x
    let beta_u = ((xl - xu) * ru + xu) as f32; // coef for y
    let ny_u_f64 = (yl * xu - yu * xl) * ru - yl * xu; // bias

    let ny_l = certify_plane_constant(alpha_l, beta_l, ny_l_f64, x_l, x_u, y_l, y_u, true);
    let ny_u = certify_plane_constant(alpha_u, beta_u, ny_u_f64, x_l, x_u, y_l, y_u, false);

    (alpha_l, beta_l, ny_l, alpha_u, beta_u, ny_u)
}
