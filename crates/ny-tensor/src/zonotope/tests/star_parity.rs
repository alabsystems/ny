// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! IBP-parity soundness gate for the star-set affine transformers (S1-2).
//!
//! For a star built from a **box** input with an **empty** predicate (pure zonotope,
//! `α ∈ [-1,1]^m`), a single affine transformer is exact: the star's `interval_bounds()`
//! must equal — and be sound (`≤`/`≥`) against — the bounds the existing `ZonotopeTensor`
//! affine path (or an independent f64 IBP reference) computes on the same input+weights.
//!
//! Why parity is expected to be exact: for a box input with per-element error symbols,
//! `x_i = c_i + r_i·α_i` (independent `α_i ∈ [-1,1]`), so after `y = Wx+b` the zonotope
//! bound `|W c+b|_j ± Σ_i |W_ji| r_i` coincides with the interval-arithmetic (IBP) bound.
//! `Star::gemm`/`add`/`flatten` delegate to the very `ZonotopeTensor` ops under test, so
//! they are bit-identical (tolerance 1e-6). `Star::conv2d` uses its own im2col matmul, so
//! it is checked against an independent f64 IBP conv (tolerance 1e-4 to absorb f32 BLAS
//! accumulation vs the f64 exact reference; the soundness slack catches any real bug,
//! which would shift a bound by O(radius) ≫ 1e-4).

use super::super::*;
use crate::BoundedTensor;
use ndarray::{Array1, Array2, Array4, ArrayD, IxDyn};
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Helpers.
// ---------------------------------------------------------------------------

/// Build a box-input star: per-element error symbols preserving the tensor shape,
/// empty predicate (`α ∈ box`).
fn box_star(center: &ArrayD<f32>, eps: f32) -> Star {
    // `from_input_elementwise` flattens to [n]; reshape restores the value shape while
    // preserving the per-element generator structure (symbol i ↦ element i).
    let flat = ZonotopeTensor::from_input_elementwise(center, eps);
    let z = flat
        .reshape(center.shape())
        .expect("reshape to original shape is size-preserving");
    Star::from_zonotope(z)
}

/// Assert two [`BoundedTensor`]s agree per-coordinate within `tol` AND that `got`
/// soundly contains `reference` (got.lo ≤ ref.lo + tol, got.hi ≥ ref.hi − tol).
fn assert_parity_and_sound(got: &BoundedTensor, reference: &BoundedTensor, tol: f32, ctx: &str) {
    prop_assert_shapes(got.shape(), reference.shape(), ctx);
    let (glo, ghi) = got.lower_upper();
    let (rlo, rhi) = reference.lower_upper();
    for (idx, ((gl, gh), (rl, rh))) in glo
        .iter()
        .zip(ghi.iter())
        .zip(rlo.iter().zip(rhi.iter()))
        .enumerate()
    {
        // Closeness (parity).
        assert!(
            (gl - rl).abs() <= tol,
            "{ctx}: lower[{idx}] parity: got {gl}, ref {rl} (|Δ|={})",
            (gl - rl).abs()
        );
        assert!(
            (gh - rh).abs() <= tol,
            "{ctx}: upper[{idx}] parity: got {gh}, ref {rh} (|Δ|={})",
            (gh - rh).abs()
        );
        // Soundness: the star bound must not be tighter than the reference.
        assert!(
            *gl <= rl + tol,
            "{ctx}: lower[{idx}] UNSOUND: star lo {gl} > ref lo {rl}"
        );
        assert!(
            *gh >= rh - tol,
            "{ctx}: upper[{idx}] UNSOUND: star hi {gh} < ref hi {rh}"
        );
    }
}

fn prop_assert_shapes(a: &[usize], b: &[usize], ctx: &str) {
    assert_eq!(a, b, "{ctx}: shape mismatch");
}

/// Independent f64 IBP interval matmul reference: `y = W x + b` over the box `[lo, hi]`.
fn ibp_gemm_ref(
    lo: &ArrayD<f32>,
    hi: &ArrayD<f32>,
    weight: &Array2<f32>,
    bias: &Array1<f32>,
) -> BoundedTensor {
    let out = weight.nrows();
    let inn = weight.ncols();
    let mut ylo = ArrayD::<f32>::zeros(IxDyn(&[out]));
    let mut yhi = ArrayD::<f32>::zeros(IxDyn(&[out]));
    for o in 0..out {
        let mut l = bias[o] as f64;
        let mut u = bias[o] as f64;
        for i in 0..inn {
            let w = weight[[o, i]] as f64;
            let (xl, xh) = (lo[[i]] as f64, hi[[i]] as f64);
            if w >= 0.0 {
                l += w * xl;
                u += w * xh;
            } else {
                l += w * xh;
                u += w * xl;
            }
        }
        ylo[[o]] = l as f32;
        yhi[[o]] = u as f32;
    }
    BoundedTensor::new_allow_infinite(ylo, yhi).unwrap()
}

/// Independent f64 IBP interval conv reference (cross-correlation) over the box `[lo, hi]`.
fn ibp_conv_ref(
    lo: &ArrayD<f32>,
    hi: &ArrayD<f32>,
    weight: &Array4<f32>,
    bias: &Array1<f32>,
    stride: (usize, usize),
    padding: (usize, usize),
) -> BoundedTensor {
    let (c, h, w) = (lo.shape()[0], lo.shape()[1], lo.shape()[2]);
    let ws = weight.shape();
    let (oc, kh, kw) = (ws[0], ws[2], ws[3]);
    let (sh, sw) = stride;
    let (ph, pw) = padding;
    let out_h = (h + 2 * ph - kh) / sh + 1;
    let out_w = (w + 2 * pw - kw) / sw + 1;
    let mut out_lo = ArrayD::<f32>::zeros(IxDyn(&[oc, out_h, out_w]));
    let mut out_hi = ArrayD::<f32>::zeros(IxDyn(&[oc, out_h, out_w]));
    for co in 0..oc {
        for oh in 0..out_h {
            for ow in 0..out_w {
                let mut l = bias[co] as f64;
                let mut u = bias[co] as f64;
                for ci in 0..c {
                    for ki in 0..kh {
                        for kj in 0..kw {
                            let ih = (oh * sh + ki) as isize - ph as isize;
                            let iw = (ow * sw + kj) as isize - pw as isize;
                            let (xl, xh) =
                                if ih >= 0 && (ih as usize) < h && iw >= 0 && (iw as usize) < w {
                                    (
                                        lo[[ci, ih as usize, iw as usize]] as f64,
                                        hi[[ci, ih as usize, iw as usize]] as f64,
                                    )
                                } else {
                                    (0.0, 0.0)
                                };
                            let wt = weight[[co, ci, ki, kj]] as f64;
                            if wt >= 0.0 {
                                l += wt * xl;
                                u += wt * xh;
                            } else {
                                l += wt * xh;
                                u += wt * xl;
                            }
                        }
                    }
                }
                out_lo[[co, oh, ow]] = l as f32;
                out_hi[[co, oh, ow]] = u as f32;
            }
        }
    }
    BoundedTensor::new_allow_infinite(out_lo, out_hi).unwrap()
}

fn arb_vals(n: usize, mag: f32) -> impl Strategy<Value = Vec<f32>> {
    proptest::collection::vec(-mag..=mag, n)
}

/// Assert `outer` soundly *contains* `inner` per-coordinate:
/// `outer.lo ≤ inner.lo + tol` and `outer.hi ≥ inner.hi − tol`.
///
/// Used where the star is a legitimately *tighter* over-approximation than the
/// reference (e.g. multi-layer composition, where the star keeps correlations the
/// re-boxing IBP loses): the star must stay inside the looser reference.
fn assert_contains(outer: &BoundedTensor, inner: &BoundedTensor, tol: f32, ctx: &str) {
    let (olo, ohi) = outer.lower_upper();
    let (ilo, ihi) = inner.lower_upper();
    for (idx, ((ol, oh), (il, ih))) in olo
        .iter()
        .zip(ohi.iter())
        .zip(ilo.iter().zip(ihi.iter()))
        .enumerate()
    {
        assert!(
            *ol <= il + tol,
            "{ctx}: lower[{idx}]: outer {ol} does not contain inner {il}"
        );
        assert!(
            *oh >= ih - tol,
            "{ctx}: upper[{idx}]: outer {oh} does not contain inner {ih}"
        );
    }
}

/// Concrete (point) conv2d cross-correlation, f64 accumulation.
fn concrete_conv(
    x: &ArrayD<f32>,
    weight: &Array4<f32>,
    bias: &Array1<f32>,
    stride: (usize, usize),
    padding: (usize, usize),
) -> ArrayD<f32> {
    let (c, h, w) = (x.shape()[0], x.shape()[1], x.shape()[2]);
    let ws = weight.shape();
    let (oc, kh, kw) = (ws[0], ws[2], ws[3]);
    let (sh, sw) = stride;
    let (ph, pw) = padding;
    let out_h = (h + 2 * ph - kh) / sh + 1;
    let out_w = (w + 2 * pw - kw) / sw + 1;
    let mut out = ArrayD::<f32>::zeros(IxDyn(&[oc, out_h, out_w]));
    for co in 0..oc {
        for oh in 0..out_h {
            for ow in 0..out_w {
                let mut acc = bias[co] as f64;
                for ci in 0..c {
                    for ki in 0..kh {
                        for kj in 0..kw {
                            let ih = (oh * sh + ki) as isize - ph as isize;
                            let iw = (ow * sw + kj) as isize - pw as isize;
                            let xv = if ih >= 0 && (ih as usize) < h && iw >= 0 && (iw as usize) < w
                            {
                                x[[ci, ih as usize, iw as usize]] as f64
                            } else {
                                0.0
                            };
                            acc += (weight[[co, ci, ki, kj]] as f64) * xv;
                        }
                    }
                }
                out[[co, oh, ow]] = acc as f32;
            }
        }
    }
    out
}

/// Concrete (point) `y = W x + b`, f64 accumulation.
fn concrete_gemm(x: &ArrayD<f32>, weight: &Array2<f32>, bias: &Array1<f32>) -> Vec<f32> {
    let out = weight.nrows();
    let inn = weight.ncols();
    (0..out)
        .map(|o| {
            let mut acc = bias[o] as f64;
            for i in 0..inn {
                acc += (weight[[o, i]] as f64) * (x[[i]] as f64);
            }
            acc as f32
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Unit tests (structure / invariants).
// ---------------------------------------------------------------------------

#[test]
fn empty_predicate_is_pure_zonotope() {
    let center = ndarray::arr1(&[1.0_f32, -2.0, 3.0]).into_dyn();
    let z = ZonotopeTensor::from_input_elementwise(&center, 0.1);
    let star = Star::from_zonotope(z.clone());

    assert!(star.is_zonotope(), "empty predicate ⇒ pure zonotope");
    assert_eq!(star.num_constraints(), 0);
    assert_eq!(star.alpha_dim(), 3);
    let (a, b) = star.constraints();
    assert_eq!(a.nrows(), 0);
    assert_eq!(b.len(), 0);

    // interval_bounds must equal the underlying zonotope bound exactly.
    let sb = star.interval_bounds().unwrap();
    let zb = z.to_bounded_tensor().unwrap();
    for i in 0..3 {
        assert!((sb.lower()[[i]] - zb.lower()[[i]]).abs() < 1e-7);
        assert!((sb.upper()[[i]] - zb.upper()[[i]]).abs() < 1e-7);
    }
}

#[test]
fn new_rejects_inconsistent_predicate() {
    let center = ndarray::arr1(&[0.0_f32, 0.0]).into_dyn();
    let z = ZonotopeTensor::from_input_elementwise(&center, 0.1); // m = 2

    // rows != rhs len
    let bad_rows = Star::new(z.clone(), Array2::zeros((1, 2)), Array1::zeros(2));
    assert!(bad_rows.is_err(), "rows(1) != rhs len(2) must error");

    // cols != alpha dim
    let bad_cols = Star::new(z.clone(), Array2::zeros((1, 5)), Array1::zeros(1));
    assert!(bad_cols.is_err(), "cols(5) != alpha dim(2) must error");

    // A zero-row matrix still carries its alpha-space width. Accepting a
    // mismatched width here would violate the constructor's documented
    // invariant and make later constraint stacking ambiguous.
    let bad_empty_cols = Star::new(z.clone(), Array2::zeros((0, 5)), Array1::zeros(0));
    assert!(
        bad_empty_cols.is_err(),
        "empty predicate width must still equal alpha dim"
    );

    // consistent predicate accepted
    let ok = Star::new(z, Array2::zeros((1, 2)), Array1::zeros(1));
    assert!(
        ok.is_ok(),
        "consistent (1x2 / 1) predicate must be accepted"
    );
    assert_eq!(ok.unwrap().num_constraints(), 1);
}

#[test]
fn conv2d_rejects_invalid_geometry_without_panicking() {
    let center = ArrayD::<f32>::zeros(IxDyn(&[1, 2, 2]));
    let star = box_star(&center, 0.05);
    let weight = Array4::<f32>::zeros((1, 1, 1, 1));

    assert!(
        star.conv2d(&weight, None, (0, 1), (0, 0)).is_err(),
        "zero stride must be a typed error"
    );

    let oversized = Array4::<f32>::zeros((1, 1, 3, 3));
    assert!(
        star.conv2d(&oversized, None, (1, 1), (0, 0)).is_err(),
        "kernel larger than padded input must be a typed error"
    );
}

#[test]
#[allow(deprecated)]
fn deprecated_bounds_lp_fails_explicitly() {
    // bounds_lp cannot be implemented in ny-tensor (would require ny-tensor → ny-mip,
    // reversing the ny-mip → ny-tensor edge). It must return an explanatory error.
    let center = ndarray::arr1(&[0.0_f32]).into_dyn();
    let star = box_star(&center, 0.1);
    assert!(star.bounds_lp().is_err(), "bounds_lp must be a stub error");
}

#[test]
fn conv2d_output_shape_is_correct() {
    // (C=2,H=5,W=5), 3x3 kernel, stride 1, pad 1 → (out_C=3, 5, 5)
    let center = ArrayD::<f32>::zeros(IxDyn(&[2, 5, 5]));
    let star = box_star(&center, 0.05);
    let weight = Array4::<f32>::from_elem((3, 2, 3, 3), 0.1);
    let bias = Array1::<f32>::zeros(3);
    let out = star.conv2d(&weight, Some(&bias), (1, 1), (1, 1)).unwrap();
    assert_eq!(out.shape(), &[3, 5, 5]);
    assert_eq!(
        out.alpha_dim(),
        2 * 5 * 5,
        "affine map preserves α dimension"
    );
    assert!(out.is_zonotope(), "conv preserves empty predicate");
}

// ---------------------------------------------------------------------------
// Parity proptests (the soundness gate).
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(128) })]

    /// Gemm: `Star::gemm` must match the `ZonotopeTensor::linear` path and an f64 IBP
    /// reference (exact, since gemm delegates to `linear`).
    #[test]
    fn gemm_parity_matches_zonotope_and_ibp(
        center in arb_vals(4, 1.0),
        wflat in arb_vals(3 * 4, 1.0),
        bias in arb_vals(3, 1.0),
    ) {
        let center = ArrayD::from_shape_vec(IxDyn(&[4]), center).unwrap();
        let weight = Array2::from_shape_vec((3, 4), wflat).unwrap();
        let bias = Array1::from_vec(bias);
        let eps = 0.1_f32;

        let z = ZonotopeTensor::from_input_elementwise(&center, eps);
        let star = Star::from_zonotope(z.clone());

        let star_out = star.gemm(&weight, Some(&bias)).unwrap();
        prop_assert!(star_out.is_zonotope(), "gemm preserves empty predicate");
        prop_assert_eq!(star_out.alpha_dim(), 4);
        let star_b = star_out.interval_bounds().unwrap();

        // Existing ZonotopeTensor affine path (star delegates to this ⇒ bit-identical).
        let zono_b = z.linear(&weight, Some(&bias)).unwrap().to_bounded_tensor().unwrap();
        assert_parity_and_sound(&star_b, &zono_b, 1e-6, "gemm vs zono");

        // Independent f64 IBP reference on the box.
        let ibp = ibp_gemm_ref(z.to_bounded_tensor().unwrap().lower(),
                               z.to_bounded_tensor().unwrap().upper(),
                               &weight, &bias);
        assert_parity_and_sound(&star_b, &ibp, 1e-4, "gemm vs ibp");
    }

    /// Residual Add: two branches derived from one input star (shared α).
    /// `f(x) = W1 x + b1`, `g(x) = W2 x + b2`, `y = f(x) + g(x)`.
    #[test]
    fn residual_add_parity_matches_zonotope(
        center in arb_vals(4, 1.0),
        w1 in arb_vals(4 * 4, 1.0),
        w2 in arb_vals(4 * 4, 1.0),
        b1 in arb_vals(4, 1.0),
        b2 in arb_vals(4, 1.0),
    ) {
        let center = ArrayD::from_shape_vec(IxDyn(&[4]), center).unwrap();
        let weight1 = Array2::from_shape_vec((4, 4), w1).unwrap();
        let weight2 = Array2::from_shape_vec((4, 4), w2).unwrap();
        let bias1 = Array1::from_vec(b1);
        let bias2 = Array1::from_vec(b2);
        let eps = 0.1_f32;

        let z = ZonotopeTensor::from_input_elementwise(&center, eps);
        let star = Star::from_zonotope(z.clone());

        let branch1 = star.gemm(&weight1, Some(&bias1)).unwrap();
        let branch2 = star.gemm(&weight2, Some(&bias2)).unwrap();
        let sum = branch1.add(&branch2).unwrap();
        prop_assert!(sum.is_zonotope(), "add of empty-predicate stars ⇒ empty predicate");
        let star_b = sum.interval_bounds().unwrap();

        // Reference via the ZonotopeTensor path (shared α).
        let z1 = z.linear(&weight1, Some(&bias1)).unwrap();
        let z2 = z.linear(&weight2, Some(&bias2)).unwrap();
        let zono_b = z1.add(&z2).unwrap().to_bounded_tensor().unwrap();
        assert_parity_and_sound(&star_b, &zono_b, 1e-6, "residual add vs zono");

        // Reference via the combined affine (W1+W2) x + (b1+b2) IBP.
        let wsum = &weight1 + &weight2;
        let bsum = &bias1 + &bias2;
        let zb = z.to_bounded_tensor().unwrap();
        let ibp = ibp_gemm_ref(zb.lower(), zb.upper(), &wsum, &bsum);
        assert_parity_and_sound(&star_b, &ibp, 1e-4, "residual add vs combined ibp");
    }

    /// Flatten/reshape: identity on the affine data ⇒ bounds identical (reordered).
    #[test]
    fn flatten_parity_preserves_bounds(
        center in arb_vals(2 * 3, 1.0),
    ) {
        let center = ArrayD::from_shape_vec(IxDyn(&[2, 3]), center).unwrap();
        let star = box_star(&center, 0.1);

        let flat = star.flatten().unwrap();
        prop_assert_eq!(flat.shape(), &[6]);
        prop_assert!(flat.is_zonotope());
        prop_assert_eq!(flat.alpha_dim(), star.alpha_dim(), "reshape preserves α dim");

        let b2d = star.interval_bounds().unwrap();
        let b1d = flat.interval_bounds().unwrap();
        // Row-major flatten ⇒ element i of the flat bound equals element (i/3, i%3) of the 2D bound.
        for i in 0..6 {
            let (r, c) = (i / 3, i % 3);
            prop_assert!((b1d.lower()[[i]] - b2d.lower()[[r, c]]).abs() < 1e-7);
            prop_assert!((b1d.upper()[[i]] - b2d.upper()[[r, c]]).abs() < 1e-7);
        }
    }

    /// Conv2d, stride 1 / pad 1, multi-channel — vs independent f64 IBP conv.
    #[test]
    fn conv2d_parity_stride1_pad1(
        center in arb_vals(2 * 5 * 5, 0.5),
        wflat in arb_vals(3 * 2 * 3 * 3, 1.0),
        bias in arb_vals(3, 0.5),
    ) {
        let center = ArrayD::from_shape_vec(IxDyn(&[2, 5, 5]), center).unwrap();
        let weight = Array4::from_shape_vec((3, 2, 3, 3), wflat).unwrap();
        let bias = Array1::from_vec(bias);
        let eps = 0.05_f32;

        let star = box_star(&center, eps);
        let out = star.conv2d(&weight, Some(&bias), (1, 1), (1, 1)).unwrap();
        prop_assert_eq!(out.shape(), &[3, 5, 5]);
        prop_assert!(out.is_zonotope());
        let star_b = out.interval_bounds().unwrap();

        let zb = star.interval_bounds().unwrap();
        let ibp = ibp_conv_ref(zb.lower(), zb.upper(), &weight, &bias, (1, 1), (1, 1));
        assert_parity_and_sound(&star_b, &ibp, 1e-4, "conv stride1 pad1 vs ibp");
    }

    /// Conv2d, stride 2 / pad 0, single input channel — vs independent f64 IBP conv.
    #[test]
    fn conv2d_parity_stride2_pad0(
        // shapes: input (C=1,H=6,W=6)=36; weight (out_C=2,in_C=1,kH=2,kW=2)=8.
        center in arb_vals(36, 0.5),
        wflat in arb_vals(8, 1.0),
        bias in arb_vals(2, 0.5),
    ) {
        let center = ArrayD::from_shape_vec(IxDyn(&[1, 6, 6]), center).unwrap();
        let weight = Array4::from_shape_vec((2, 1, 2, 2), wflat).unwrap();
        let bias = Array1::from_vec(bias);
        let eps = 0.05_f32;

        let star = box_star(&center, eps);
        let out = star.conv2d(&weight, Some(&bias), (2, 2), (0, 0)).unwrap();
        prop_assert_eq!(out.shape(), &[2, 3, 3]);
        let star_b = out.interval_bounds().unwrap();

        let zb = star.interval_bounds().unwrap();
        let ibp = ibp_conv_ref(zb.lower(), zb.upper(), &weight, &bias, (2, 2), (0, 0));
        assert_parity_and_sound(&star_b, &ibp, 1e-4, "conv stride2 pad0 vs ibp");
    }

    /// Composed mini-network: Conv2d → Flatten → Gemm.
    ///
    /// Across composed layers the star keeps the shared error symbols, so it is a
    /// *tighter* over-approximation than a re-boxing end-to-end IBP: `T ⊆ star ⊆ IBP`
    /// where `T` is the true reachable set. We therefore assert (a) `star ⊆ IBP`
    /// (the star is sound and never looser than IBP), and (b) concrete Monte-Carlo
    /// containment: every sampled point `f(x)` for `x = center + eps·α`, `α ∈ [-1,1]^m`,
    /// lies inside the star bounds — the gold-standard soundness check.
    #[test]
    fn mini_network_conv_flatten_gemm_sound(
        // shapes: input (C=1,H=4,W=4)=16; conv weight (out_C=2,in_C=1,kH=3,kW=3)=18;
        // after conv (stride1,pad0) → (2,2,2) flatten → 8; gemm weight (5,8)=40.
        center in arb_vals(16, 0.5),
        wc in arb_vals(18, 1.0),
        bc in arb_vals(2, 0.5),
        wg in arb_vals(40, 1.0),
        bg in arb_vals(5, 0.5),
        alpha in proptest::collection::vec(-1.0_f32..=1.0, 16),
    ) {
        let center = ArrayD::from_shape_vec(IxDyn(&[1, 4, 4]), center).unwrap();
        let wconv = Array4::from_shape_vec((2, 1, 3, 3), wc).unwrap();
        let bconv = Array1::from_vec(bc);
        // Conv (stride1,pad0) on (1,4,4) → (2,2,2); flatten → [8]; gemm 5×8.
        let wgemm = Array2::from_shape_vec((5, 8), wg).unwrap();
        let bgemm = Array1::from_vec(bg);
        let eps = 0.05_f32;

        let star = box_star(&center, eps);
        let conv = star.conv2d(&wconv, Some(&bconv), (1, 1), (0, 0)).unwrap();
        prop_assert_eq!(conv.shape(), &[2, 2, 2]);
        let flat = conv.flatten().unwrap();
        prop_assert_eq!(flat.shape(), &[8]);
        let out = flat.gemm(&wgemm, Some(&bgemm)).unwrap();
        prop_assert_eq!(out.shape(), &[5]);
        prop_assert!(out.is_zonotope());
        let star_b = out.interval_bounds().unwrap();

        // (a) star ⊆ IBP: end-to-end re-boxing IBP is looser; the star must stay inside it.
        let zb = star.interval_bounds().unwrap();
        let conv_ibp = ibp_conv_ref(zb.lower(), zb.upper(), &wconv, &bconv, (1, 1), (0, 0));
        let flat_lo = conv_ibp.lower().clone().into_shape_with_order(IxDyn(&[8])).unwrap();
        let flat_hi = conv_ibp.upper().clone().into_shape_with_order(IxDyn(&[8])).unwrap();
        let out_ibp = ibp_gemm_ref(&flat_lo, &flat_hi, &wgemm, &bgemm);
        assert_contains(&out_ibp, &star_b, 1e-4, "mini-network: star ⊆ IBP");

        // (b) concrete Monte-Carlo containment: random α plus both extreme corners.
        let all_pos = vec![1.0_f32; 16];
        let all_neg = vec![-1.0_f32; 16];
        for sample in [&alpha, &all_pos, &all_neg] {
            let flat_center: Vec<f32> = center.iter().cloned().collect();
            let x: Vec<f32> = flat_center
                .iter()
                .zip(sample.iter())
                .map(|(c, a)| c + eps * a)
                .collect();
            let x = ArrayD::from_shape_vec(IxDyn(&[1, 4, 4]), x).unwrap();
            let cy = concrete_conv(&x, &wconv, &bconv, (1, 1), (0, 0));
            let cy_flat = cy.into_shape_with_order(IxDyn(&[8])).unwrap();
            let y = concrete_gemm(&cy_flat, &wgemm, &bgemm);
            for (i, yi) in y.iter().enumerate() {
                prop_assert!(
                    star_b.lower()[[i]] - 1e-3 <= *yi && *yi <= star_b.upper()[[i]] + 1e-3,
                    "mini-network concrete point escapes star: out[{}]={} not in [{}, {}]",
                    i, yi, star_b.lower()[[i]], star_b.upper()[[i]]
                );
            }
        }
    }
}
