// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Differential drift tests: bind the scalar mirrors in this crate to the
//! production relaxations in `ny-propagate` that they copy.
//!
//! The Kani proof harnesses in `proofs/kani/` verify THIS crate's functions,
//! but no production crate depends on ny-relaxation, so without these tests a
//! change to a production relaxation would silently invalidate what the
//! proofs appear to certify. Every mirrored pair is evaluated over a
//! deterministic adversarial grid and held to one of two policies:
//!
//! 1. BIT-EXACT — the implementations are copies; any bit difference in any
//!    output field fails: `abs`, `pow2`, `exp`, `gelu_eval`, `silu_eval`,
//!    `gelu_tanh_inflection_point`, the softmax/logsoftmax/logsumexp IBP
//!    helpers, and all `safe_*`/`interval_mul` helpers. (For `silu` the
//!    near-point path `|u-l| < 1e-8` is also bit-exact.)
//!
//! 2. DIRECTIONAL — the pair intentionally differs: the mirror keeps extra
//!    caller-side f32 multiplication-rounding corrections (`mul_err`, Kani
//!    lineage: proof `sqrt_alpha_upper_sound_mid_at_lower`) that production
//!    lacks. Slopes must be bit-equal; the mirror's intercepts must be
//!    OUTSIDE (looser than) production's by no more than the correction
//!    magnitude: `log`, `sqrt`. NOTE (residual gap): because the mirror band
//!    strictly contains the production band, the Kani proofs do NOT transfer
//!    to production — certifying the wider mirror band says nothing about the
//!    tighter production band at the ~1-ulp scale of the missing `mul_err`
//!    term. Closing the gap requires porting the mirror's corrections into
//!    ny-propagate. As a stopgap, every directional test ALSO empirically
//!    checks both bands against an f64 reference on a dense x-grid.
//!
//! Pairs that cannot be held to either policy are covered by
//! empirical-soundness + partial-relation tests, each with a NOTE naming the
//! residual gap: `silu` (branch-verification fallbacks may diverge), the
//! `gelu` sound relaxations (production adds a `clamp_lower_to_floor`
//! tightening the mirror lacks), and `relu` (the mirror is a standalone
//! reference; the production scalar `relu_linear_relaxation` is pub(crate)
//! and is exercised through the public `ReLULayer` CROWN backward path).

use ndarray::{ArrayD, IxDyn};
use ny_relaxation as mirror;
use ny_relaxation::LinearRelaxation as MirrorRelax;
use ny_tensor::BoundedTensor;

use ny_propagate::bounds as prod_bounds;
use ny_propagate::layers as prod_layers;
use ny_propagate::layers::LinearRelaxation as ProdRelax;
use ny_propagate::layers::ReLULayer;
use ny_propagate::LinearBounds;

type Band = (f32, f32, f32, f32);

/// Deterministic adversarial grid: signed zeros, subnormals, the narrow-
/// interval thresholds (1e-8, 1e-12), branch-point neighborhoods of every
/// mirrored function (silu inflection/critical points, gelu minimizer, exp
/// midpoint-cap legacy boundary 0.99/1.98), ULP neighbors of anchors, plus a
/// dense sweep of the transcendental-relevant window and extremes.
fn grid() -> Vec<f32> {
    let mut v: Vec<f32> = vec![
        0.0,
        -0.0,
        f32::from_bits(1), // smallest positive subnormal
        -f32::from_bits(1),
        f32::from_bits(0x0040_0000), // mid subnormal
        -f32::from_bits(0x0040_0000),
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        1e-38,
        -1e-38,
        1e-30,
        -1e-30,
        1e-12,
        -1e-12,
        2e-12,
        5e-9,
        -5e-9,
        1e-8,
        -1e-8,
        2e-8,
        -2e-8,
        1e-6,
        -1e-6,
        1e-3,
        -1e-3,
        0.05,
        -0.05,
        0.1,
        -0.1,
        0.5,
        -0.5,
        std::f32::consts::LN_2,
        -std::f32::consts::LN_2,
        // gelu minimizer neighborhood
        -0.7517916,
        -0.7517915,
        0.99,
        -0.99,
        1.0,
        -1.0,
        1.0 + f32::EPSILON,
        -1.0 - f32::EPSILON,
        1.5,
        -1.5,
        1.98,
        -1.98,
        2.0,
        -2.0,
        // silu critical point / inflection point neighborhoods
        -1.27846,
        -1.2784645,
        2.399,
        -2.399,
        2.3994,
        -2.3994,
        2.4,
        -2.4,
        std::f32::consts::E,
        -std::f32::consts::E,
        3.0,
        -3.0,
        4.0,
        -4.0,
        5.0,
        -5.0,
        8.0,
        -8.0,
        10.0,
        -10.0,
        19.25,
        -19.25,
        20.0,
        -20.0,
        50.0,
        -50.0,
        88.0,
        -88.0,
        100.0,
        -100.0,
        709.0,
        -709.0,
        1e4,
        -1e4,
        1e8,
        -1e8,
        1e20,
        -1e20,
        3e38,
        -3e38,
        f32::MAX,
        -f32::MAX,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ];
    for a in [0.0f32, 1.0, -1.0, 2.0, -2.0, 0.05, 1e-8, -1e-8] {
        v.push(mirror::next_up_f32(a));
        v.push(mirror::next_down_f32(a));
    }
    let n = 64;
    for i in 0..=n {
        v.push(-6.0 + 12.0 * (i as f32) / (n as f32));
    }
    v
}

/// All ordered pairs (l, u) with l <= u from the grid.
fn intervals() -> Vec<(f32, f32)> {
    let g = grid();
    let mut out = Vec::new();
    for &l in &g {
        for &u in &g {
            if l <= u {
                out.push((l, u));
            }
        }
    }
    out
}

/// Bit equality that treats every NaN payload as equal.
fn feq(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

fn band_eq(m: Band, p: Band) -> bool {
    feq(m.0, p.0) && feq(m.1, p.1) && feq(m.2, p.2) && feq(m.3, p.3)
}

fn mirror_band(r: MirrorRelax) -> Band {
    (
        r.lower_slope,
        r.lower_intercept,
        r.upper_slope,
        r.upper_intercept,
    )
}

fn prod_band(r: ProdRelax) -> Band {
    (
        r.lower_slope,
        r.lower_intercept,
        r.upper_slope,
        r.upper_intercept,
    )
}

/// Sample points of [l, u] as the caller would produce them in f32.
fn x_samples(l: f32, u: f32) -> Vec<f32> {
    let mut xs = vec![l, u];
    for i in 1..32 {
        let x = l + (u - l) * (i as f32) / 32.0;
        if x.is_finite() && x >= l && x <= u {
            xs.push(x);
        }
    }
    xs
}

/// Assert that `band`, evaluated in f32 exactly as production callers evaluate
/// it (`slope * x + intercept`), contains the f64 reference `f` on [l, u].
/// `rel_slack` absorbs only the reference's own evaluation error.
fn assert_band_sound(
    name: &str,
    which: &str,
    l: f32,
    u: f32,
    band: Band,
    f: &dyn Fn(f64) -> f64,
    rel_slack: f64,
    abs_slack: f64,
) {
    for x in x_samples(l, u) {
        let fx = f(x as f64);
        if !fx.is_finite() {
            continue;
        }
        let slack = rel_slack * (1.0 + fx.abs()) + abs_slack;
        let lo = band.0 * x + band.1;
        if lo.is_finite() {
            assert!(
                (lo as f64) <= fx + slack,
                "{name} {which} LOWER line above f: l={l:e} u={u:e} x={x:e} \
                 line={lo:e} f={fx:e} band={band:?}"
            );
        }
        let hi = band.2 * x + band.3;
        if hi.is_finite() {
            assert!(
                (hi as f64) >= fx - slack,
                "{name} {which} UPPER line below f: l={l:e} u={u:e} x={x:e} \
                 line={hi:e} f={fx:e} band={band:?}"
            );
        }
    }
}

/// Bound on how far the mirror's intercept may sit outside production's:
/// the mirror's extra `mul_err` correction (|slope|*max|x|*eps, rounded up)
/// plus a few ulps from the extra directed-rounding steps.
fn intercept_gap_bound(slope: f32, max_abs_x: f32, intercept: f32) -> f64 {
    let eps = f32::EPSILON as f64;
    let mul = (slope.abs() as f64) * (max_abs_x as f64) * eps;
    4.0 * mul + 8.0 * eps * (intercept.abs() as f64) + f32::MIN_POSITIVE as f64
}

// =========================================================================
// Tier 1: bit-exact pairs
// =========================================================================

#[test]
fn drift_abs_bit_exact() {
    for (l, u) in intervals() {
        let m = mirror_band(mirror::abs_linear_relaxation(l, u));
        let p = prod_band(prod_layers::abs_linear_relaxation(l, u));
        assert!(
            band_eq(m, p),
            "abs drift at l={l:e} u={u:e}: mirror={m:?} prod={p:?}"
        );
    }
}

#[test]
fn drift_pow2_bit_exact() {
    for (l, u) in intervals() {
        let m = mirror_band(mirror::pow2_linear_relaxation(l, u));
        let p = prod_band(prod_layers::pow2_linear_relaxation(l, u));
        assert!(
            band_eq(m, p),
            "pow2 drift at l={l:e} u={u:e}: mirror={m:?} prod={p:?}"
        );
    }
}

#[test]
fn drift_exp_bit_exact() {
    for (l, u) in intervals() {
        let m = mirror_band(mirror::exp_linear_relaxation(l, u));
        let p = prod_band(prod_layers::exp_linear_relaxation(l, u));
        assert!(
            band_eq(m, p),
            "exp drift at l={l:e} u={u:e}: mirror={m:?} prod={p:?}"
        );
    }
    // NaN guard parity.
    let m = mirror_band(mirror::exp_linear_relaxation(f32::NAN, 1.0));
    let p = prod_band(prod_layers::exp_linear_relaxation(f32::NAN, 1.0));
    assert!(
        band_eq(m, p),
        "exp NaN-guard drift: mirror={m:?} prod={p:?}"
    );
}

#[test]
fn drift_scalar_evals_bit_exact() {
    for x in grid() {
        let m = mirror::gelu_eval(x, mirror::GeluApproximation::Erf);
        let p = prod_layers::gelu_eval(x, prod_layers::GeluApproximation::Erf);
        assert!(feq(m, p), "gelu_eval(erf) drift at x={x:e}: {m:e} vs {p:e}");

        let m = mirror::gelu_eval(x, mirror::GeluApproximation::Tanh);
        let p = prod_layers::gelu_eval(x, prod_layers::GeluApproximation::Tanh);
        assert!(
            feq(m, p),
            "gelu_eval(tanh) drift at x={x:e}: {m:e} vs {p:e}"
        );

        let m = mirror::silu_eval(x);
        let p = prod_layers::silu_eval(x);
        assert!(feq(m, p), "silu_eval drift at x={x:e}: {m:e} vs {p:e}");
    }
    assert!(
        feq(
            mirror::gelu_tanh_inflection_point(),
            prod_layers::gelu_tanh_inflection_point()
        ),
        "gelu_tanh_inflection_point drift"
    );
}

#[test]
fn drift_softmax_helpers_bit_exact() {
    let g = grid();

    // exp_interval_bounds: Ok values bit-equal, Err-ness identical (covers
    // inverted intervals too — no l <= u filter here).
    for &a in &g {
        for &b in &g {
            match (
                mirror::exp_interval_bounds(a, b),
                prod_layers::exp_interval_bounds(a, b),
            ) {
                (Ok((ml, mu)), Ok((pl, pu))) => {
                    assert!(
                        feq(ml, pl) && feq(mu, pu),
                        "exp_interval_bounds drift at ({a:e},{b:e}): \
                         ({ml:e},{mu:e}) vs ({pl:e},{pu:e})"
                    );
                }
                (Err(_), Err(_)) => {}
                (m, p) => {
                    panic!("exp_interval_bounds Ok/Err drift at ({a:e},{b:e}): {m:?} vs {p:?}")
                }
            }

            let (ml, mu) = mirror::logsoftmax_ibp_bounds(a, b, b, a);
            let (pl, pu) = prod_layers::logsoftmax_ibp_bounds(a, b, b, a);
            assert!(
                feq(ml, pl) && feq(mu, pu),
                "logsoftmax_ibp_bounds drift at ({a:e},{b:e})"
            );
        }
    }

    // softmax_ibp_element_bounds over non-negative quadruples.
    let pos: Vec<f32> = g.iter().copied().filter(|x| *x >= 0.0).collect();
    for el in pos.iter().step_by(3) {
        for eu in pos.iter().step_by(3) {
            for sl in pos.iter().step_by(5) {
                for su in pos.iter().step_by(5) {
                    let (ml, mu) = mirror::softmax_ibp_element_bounds(*el, *eu, *sl, *su);
                    let (pl, pu) = prod_layers::softmax_ibp_element_bounds(*el, *eu, *sl, *su);
                    assert!(
                        feq(ml, pl) && feq(mu, pu),
                        "softmax_ibp_element_bounds drift at ({el:e},{eu:e},{sl:e},{su:e}): \
                         ({ml:e},{mu:e}) vs ({pl:e},{pu:e})"
                    );
                }
            }
        }
    }

    // logsumexp over deterministic slices, including NaN and saturated inputs.
    let slices: Vec<Vec<f32>> = vec![
        vec![],
        vec![0.0],
        vec![-0.0, 0.0],
        vec![1.0, 2.0, 3.0],
        vec![-1e30, 1e30],
        vec![f32::NEG_INFINITY, 0.0],
        vec![f32::INFINITY, 0.0],
        vec![f32::NAN, 1.0],
        vec![88.0, 88.0, 88.0],
        vec![-100.0, 0.0, 100.0],
        g,
    ];
    for s in &slices {
        let m = mirror::logsumexp_slice(s);
        let p = prod_layers::logsumexp_slice(s);
        assert!(
            feq(m, p),
            "logsumexp_slice drift on len={}: {m:e} vs {p:e}",
            s.len()
        );
    }
}

#[test]
fn drift_safe_math_bit_exact() {
    let mut specials = grid();
    specials.extend([f32::INFINITY, f32::NEG_INFINITY, f32::NAN]);

    for &a in &specials {
        for &b in &specials {
            let m = mirror::safe_mul_for_bounds(a, b);
            let p = prod_bounds::safe_mul_for_bounds(a, b);
            assert!(feq(m, p), "safe_mul_for_bounds drift at ({a:e},{b:e})");

            let m = mirror::safe_mul_pair_for_bounds(a, b);
            let p = prod_bounds::safe_mul_pair_for_bounds(a, b);
            assert!(feq(m, p), "safe_mul_pair_for_bounds drift at ({a:e},{b:e})");

            for is_lower in [false, true] {
                let m = mirror::safe_add_for_bounds_with_polarity(a, b, is_lower);
                let p = prod_bounds::safe_add_for_bounds_with_polarity(a, b, is_lower);
                assert!(
                    feq(m, p),
                    "safe_add_for_bounds_with_polarity drift at ({a:e},{b:e},{is_lower})"
                );
            }

            let m = mirror::safe_add_lower_for_bounds(a, b);
            let p = prod_bounds::safe_add_lower_for_bounds(a, b);
            assert!(
                feq(m, p),
                "safe_add_lower_for_bounds drift at ({a:e},{b:e})"
            );

            let m = mirror::safe_add_upper_for_bounds(a, b);
            let p = prod_bounds::safe_add_upper_for_bounds(a, b);
            assert!(
                feq(m, p),
                "safe_add_upper_for_bounds drift at ({a:e},{b:e})"
            );
        }
    }

    // interval_mul_for_bounds over a coarser 4-way cross.
    let sub: Vec<f32> = specials.iter().copied().step_by(7).collect();
    for &al in &sub {
        for &au in &sub {
            for &bl in &sub {
                for &bu in &sub {
                    let (ml, mu) = mirror::interval_mul_for_bounds(al, au, bl, bu);
                    let (pl, pu) = prod_bounds::interval_mul_for_bounds(al, au, bl, bu);
                    assert!(
                        feq(ml, pl) && feq(mu, pu),
                        "interval_mul_for_bounds drift at [{al:e},{au:e}]x[{bl:e},{bu:e}]"
                    );
                }
            }
        }
    }
}

// =========================================================================
// Tier 2: directional pairs (mirror = production + extra mul_err corrections)
// =========================================================================

#[test]
fn drift_log_directional() {
    for (l, u) in intervals() {
        let m = mirror_band(mirror::log_linear_relaxation(l, u));
        let p = prod_band(prod_layers::log_linear_relaxation(l, u));

        // Slopes are computed identically (chord / parallel-to-chord tangent
        // with identical branch conditions) and must match bit-for-bit.
        assert!(
            feq(m.0, p.0) && feq(m.2, p.2),
            "log slope drift at l={l:e} u={u:e}: mirror={m:?} prod={p:?}"
        );

        // Intercepts: mirror must be outside production by at most the
        // mirror's extra mul_err correction. NOTE (residual gap): production
        // log lacks the f32 multiplication-rounding term; the Kani log proofs
        // certify only the wider mirror band.
        let lc = if l > 1e-10 { l } else { 1e-10 };
        let uc = if u > 1e-10 { u } else { 1e-10 };
        // The infinite-upper branch derives its corrections from the clamped
        // lower bound only (and inf would poison the gap bound via 0 * inf).
        let max_abs_x = if u.is_finite() {
            lc.abs().max(uc.abs())
        } else {
            lc.abs()
        };
        if m.1.is_finite() && p.1.is_finite() {
            let gap = intercept_gap_bound(m.0, max_abs_x, p.1);
            assert!(
                m.1 <= p.1 && (p.1 as f64 - m.1 as f64) <= gap,
                "log lower intercept drift at l={l:e} u={u:e}: \
                 mirror={:e} prod={:e} gap_bound={gap:e}",
                m.1,
                p.1
            );
        }
        if m.3.is_finite() && p.3.is_finite() {
            let gap = intercept_gap_bound(m.2, max_abs_x, p.3);
            assert!(
                m.3 >= p.3 && (m.3 as f64 - p.3 as f64) <= gap,
                "log upper intercept drift at l={l:e} u={u:e}: \
                 mirror={:e} prod={:e} gap_bound={gap:e}",
                m.3,
                p.3
            );
        }

        // Empirical soundness of BOTH bands (production is not proof-covered).
        // log's domain contract is x > 0 (callers validate via IBP); check on
        // the clamped domain the implementations actually bound.
        //
        // #drift-log-narrow-tiny (RESOLVED): the shared narrow-interval path
        // used an ABSOLUTE width guard (|u-l| < 1e-8 after the 1e-10 clamp)
        // and returned the tangent at l for BOTH lines; a tangent lies ABOVE
        // concave ln, so the certified LOWER line overshot ln(u) by
        // r - ln(1+r), r = (u-l)/l — +45.09 at [1e-12, 5e-9] (post-clamp
        // [1e-10, 5e-9]), 1.1e-5 at [1e-6, 1e-6+5e-9]. Both crates now guard
        // RELATIVE to l ((u-l) < l * 1e-7) and return the endpoint-constant
        // band [ln(l) rounded down, ln(u) rounded up]. The constant band has
        // no slope and hence no mul_err divergence, so the narrow path is
        // held BIT-EXACT and the soundness sweep below runs with no
        // degenerate exemption. The historical adversarial intervals are
        // asserted as positive checks after the loop.
        let narrow = ((uc as f64) - (lc as f64)) < (lc as f64) * 1e-7;
        if u.is_finite() && narrow {
            assert!(
                band_eq(m, p),
                "log narrow-path drift at l={l:e} u={u:e}: mirror={m:?} prod={p:?}"
            );
        }
        if u.is_finite() {
            let f = |x: f64| x.max(1e-10).ln();
            assert_band_sound("log", "mirror", lc, uc, m, &f, 1e-12, 0.0);
            // KNOWN UNSOUND (#drift-log-missing-mul-err): production's f32-
            // evaluated lower line can exceed ln(x) by exactly the omitted
            // mul_err correction — observed at [1e-10, 1e8]: line(1e8) =
            // 18.420681 vs ln(1e8) = 18.42068074 (violation 2.6e-8). The
            // mirror (with mul_err) passes at zero extra slack; production is
            // granted precisely the missing-correction budget so any LARGER
            // violation still fails.
            let mul_gap = (m.0.abs().max(m.2.abs()) as f64)
                * (max_abs_x as f64)
                * (f32::EPSILON as f64)
                * 2.0;
            assert_band_sound("log", "prod", lc, uc, p, &f, 1e-12, mul_gap);
        }
    }

    // Positive soundness checks at the #drift-log-narrow-tiny adversarial
    // intervals (formerly exempted as KNOWN UNSOUND): the old absolute narrow
    // guard certified a LOWER line +45.09 above ln(u) on the first of these.
    // Only the post-clamp point interval still takes the (bit-exact) narrow
    // path; the others now fall through to the general chord/tangent path,
    // where production keeps its #drift-log-missing-mul-err budget.
    let f = |x: f64| x.max(1e-10).ln();
    let adversarial: [(f32, f32); 4] = [
        (1e-12, 5e-9),       // clamps to [1e-10, 5e-9]; was unsound by 45.09
        (1e-10, 5e-9),       // same interval, no clamp involved
        (1e-6, 1e-6 + 5e-9), // was unsound by 1.1e-5
        (1e-12, 2e-12),      // clamps to the point interval [1e-10, 1e-10]
    ];
    for (l, u) in adversarial {
        let lc = if l > 1e-10 { l } else { 1e-10 };
        let uc = if u > 1e-10 { u } else { 1e-10 };
        let m = mirror_band(mirror::log_linear_relaxation(l, u));
        let p = prod_band(prod_layers::log_linear_relaxation(l, u));
        if ((uc as f64) - (lc as f64)) < (lc as f64) * 1e-7 {
            assert!(
                band_eq(m, p),
                "log adversarial narrow drift at l={l:e} u={u:e}: mirror={m:?} prod={p:?}"
            );
        }
        assert_band_sound("log", "mirror", lc, uc, m, &f, 1e-12, 0.0);
        let mul_gap = (m.0.abs().max(m.2.abs()) as f64) * (uc as f64) * (f32::EPSILON as f64) * 2.0;
        assert_band_sound("log", "prod", lc, uc, p, &f, 1e-12, mul_gap);
    }
}

#[test]
fn drift_sqrt_directional() {
    for (l, u) in intervals() {
        let m = mirror_band(mirror::sqrt_linear_relaxation(l, u));
        let p = prod_band(prod_layers::sqrt_linear_relaxation(l, u));

        assert!(
            feq(m.0, p.0) && feq(m.2, p.2),
            "sqrt slope drift at l={l:e} u={u:e}: mirror={m:?} prod={p:?}"
        );

        // NOTE (residual gap): production sqrt lacks the mul-rounding
        // corrections the Kani proof `sqrt_alpha_upper_sound_mid_at_lower`
        // forced into the mirror; production's alpha path
        // (`sqrt_linear_relaxation_with_alpha`, pub(crate)) is additionally
        // unreachable from here and only its default tangent point is bound.
        let max_abs_x = l.max(0.0).abs().max(u.max(0.0).abs());
        if m.1.is_finite() && p.1.is_finite() {
            let gap = intercept_gap_bound(m.0, max_abs_x, p.1);
            assert!(
                m.1 <= p.1 && (p.1 as f64 - m.1 as f64) <= gap,
                "sqrt lower intercept drift at l={l:e} u={u:e}: \
                 mirror={:e} prod={:e} gap_bound={gap:e}",
                m.1,
                p.1
            );
        }
        if m.3.is_finite() && p.3.is_finite() {
            let gap = intercept_gap_bound(m.2, max_abs_x, p.3);
            assert!(
                m.3 >= p.3 && (m.3 as f64 - p.3 as f64) <= gap,
                "sqrt upper intercept drift at l={l:e} u={u:e}: \
                 mirror={:e} prod={:e} gap_bound={gap:e}",
                m.3,
                p.3
            );
        }

        if l.is_finite() && u.is_finite() {
            let f = |x: f64| x.max(0.0).sqrt();
            assert_band_sound("sqrt", "mirror", l, u, m, &f, 1e-13, 0.0);
            // KNOWN UNSOUND (#drift-sqrt-missing-mul-err): as for log,
            // production's f32-evaluated chord can cross sqrt(x) by the
            // omitted mul_err — observed at [0, 2e-8]: lower line(2e-8) =
            // 1.4142136e-4 vs sqrt = 1.41421356e-4 (violation 5.2e-12).
            // Production gets exactly the missing-correction budget.
            let mul_gap = (m.0.abs().max(m.2.abs()) as f64)
                * (max_abs_x as f64)
                * (f32::EPSILON as f64)
                * 2.0;
            assert_band_sound("sqrt", "prod", l, u, p, &f, 1e-13, mul_gap);
        }
    }
}

// =========================================================================
// Tier 3: structurally diverging pairs (partial relations + soundness)
// =========================================================================

#[test]
fn drift_silu() {
    let silu = |x: f64| x / (1.0 + (-x).exp());
    for (l, u) in intervals() {
        let m = mirror_band(mirror::silu_sound_linear_relaxation(l, u));
        let p = prod_band(prod_layers::silu_sound_linear_relaxation(l, u));

        if l.is_finite() && u.is_finite() && (u - l).abs() < 1e-8 {
            // Near-point path is a literal copy: bit-exact.
            assert!(
                band_eq(m, p),
                "silu near-point drift at l={l:e} u={u:e}: mirror={m:?} prod={p:?}"
            );
        } else {
            // NOTE (residual gap): outside the near-point path the pair can
            // legitimately diverge in branch, not just intercept: both sides
            // verify their tangent/chord candidates against silu and fall
            // back to constant bounds on failure, and the mirror's wider
            // (mul_err-corrected) intercepts can pass a verification the
            // tighter production intercepts fail (or vice versa). When
            // neither side fell back, slopes must agree bit-for-bit and the
            // mirror's intercepts must enclose production's.
            let same_shape = feq(m.0, p.0) && feq(m.2, p.2);
            let a_fallback = m.0 == 0.0 || m.2 == 0.0 || p.0 == 0.0 || p.2 == 0.0;
            assert!(
                same_shape || a_fallback,
                "silu slope drift without fallback at l={l:e} u={u:e}: \
                 mirror={m:?} prod={p:?}"
            );
            if same_shape && !a_fallback {
                assert!(
                    m.1 <= p.1 && m.3 >= p.3,
                    "silu intercept direction drift at l={l:e} u={u:e}: \
                     mirror={m:?} prod={p:?}"
                );
            }
        }

        // NOTE: 1e-7 (~2 f32 ulps) rather than the 1e-11 used elsewhere —
        // silu's constant bands (near-point path, min_val/max_val fallbacks)
        // are round-to-nearest `silu_eval` values without directed rounding,
        // so both crates are sound only to half an output ulp (e.g. at
        // l=u=0.05 the upper band sits 8.2e-10 BELOW true silu). Shared
        // mirror/production behavior; the Kani proof
        // `silu_sound_point_interval` likewise asserts only a tolerance.
        if l.is_finite() && u.is_finite() {
            assert_band_sound("silu", "mirror", l, u, m, &silu, 1e-7, 0.0);
            assert_band_sound("silu", "prod", l, u, p, &silu, 1e-7, 0.0);
        }
    }
}

#[test]
fn drift_gelu_sound() {
    let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
    let gelu_erf = move |x: f64| 0.5 * x * (1.0 + libm::erf(x * inv_sqrt2));
    let c: f64 = (2.0 / std::f64::consts::PI).sqrt();
    let gelu_tanh = move |x: f64| 0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh());

    for (l, u) in intervals() {
        for erf in [true, false] {
            let (m, p): (Band, Band) = if erf {
                (
                    mirror::gelu_sound_linear_relaxation(l, u),
                    prod_layers::gelu_sound_linear_relaxation(l, u),
                )
            } else {
                (
                    mirror::gelu_tanh_sound_linear_relaxation(l, u),
                    prod_layers::gelu_tanh_sound_linear_relaxation(l, u),
                )
            };
            let name = if erf { "gelu_erf" } else { "gelu_tanh" };

            // Upper: chord/tangent slope selection is computed identically;
            // the intercepts differ by the mirror's extra chord/mid mul_err
            // AND by `gelu_posthoc_adjust`, which widens each side by its own
            // observed sample violations, so the difference is bounded but
            // NOT directional. NOTE (residual gap): production's intercepts
            // are generally tighter than what the Kani gelu proofs certify.
            assert!(
                feq(m.2, p.2),
                "{name} upper slope drift at l={l:e} u={u:e}: mirror={m:?} prod={p:?}"
            );
            let max_abs_x = if l.is_finite() && u.is_finite() {
                l.abs().max(u.abs())
            } else {
                0.0
            };
            if m.3.is_finite() && p.3.is_finite() && m.2.is_finite() {
                let gap = 2.0 * intercept_gap_bound(m.2, max_abs_x, p.3);
                assert!(
                    (m.3 as f64 - p.3 as f64).abs() <= gap,
                    "{name} upper intercept drift at l={l:e} u={u:e}: \
                     mirror={:e} prod={:e} gap_bound={gap:e}",
                    m.3,
                    p.3
                );
            }

            // Lower: production additionally applies clamp_lower_to_floor
            // (raises the lower line to the constant interval minimum when
            // the line concretizes below it) — a tightening the mirror lacks.
            // NOTE (residual gap): when the clamp fires (prod slope 0.0) the
            // production lower bound is NOT covered by the Kani gelu proofs.
            if feq(m.0, p.0) {
                if m.1.is_finite() && p.1.is_finite() && m.0.is_finite() {
                    let gap = 2.0 * intercept_gap_bound(m.0, max_abs_x, p.1);
                    assert!(
                        (m.1 as f64 - p.1 as f64).abs() <= gap,
                        "{name} lower intercept drift at l={l:e} u={u:e}: \
                         mirror={:e} prod={:e} gap_bound={gap:e}",
                        m.1,
                        p.1
                    );
                }
            } else {
                assert!(
                    p.0 == 0.0,
                    "{name} lower drift is not the production clamp at l={l:e} u={u:e}: \
                     mirror={m:?} prod={p:?}"
                );
            }

            if l.is_finite() && u.is_finite() {
                let f: &dyn Fn(f64) -> f64 = if erf { &gelu_erf } else { &gelu_tanh };
                assert_band_sound(name, "mirror", l, u, m, f, 1e-11, 0.0);
                // KNOWN UNSOUND (#drift-gelu-missing-mul-err): same class as
                // log/sqrt — production's f32-evaluated lines can cross gelu
                // by the omitted mul_err (observed at l=u=-1e-3: lower line
                // -4.9960107e-4 vs gelu -4.9960108e-4, violation 1.1e-11).
                // Production gets exactly the missing-correction budget.
                let mul_gap = (m.0.abs().max(m.2.abs()) as f64)
                    * (max_abs_x as f64)
                    * (f32::EPSILON as f64)
                    * 2.0;
                let mul_gap = if mul_gap.is_finite() { mul_gap } else { 0.0 };
                assert_band_sound(name, "prod", l, u, p, f, 1e-11, mul_gap);
            }
        }
    }
}

/// Production's scalar ReLU relaxation (`relu_linear_relaxation` /
/// `relu_crossing_upper_chord`) is pub(crate); exercise it through the public
/// `ReLULayer` CROWN backward pass with a 1-neuron identity seed, which
/// composes exactly one relaxation. NOTE (residual gap): the mirror
/// `relu_crown_relaxation` is a standalone reference (denormal/extreme-ratio
/// branches, iterative intercept repair) and is NOT bit-comparable to the
/// production chord (single next_up directed rounding); both are instead held
/// to exact empirical soundness against ReLU, which is exactly representable.
#[test]
fn drift_relu() {
    let layer = ReLULayer::new();
    let relu = |x: f64| x.max(0.0);
    for (l, u) in intervals() {
        if !l.is_finite() || !u.is_finite() {
            continue; // mirror asserts finiteness
        }
        let m4 = mirror::relu_crown_relaxation(l, u);
        assert_band_sound("relu", "mirror", l, u, m4, &relu, 0.0, 0.0);

        let seed = LinearBounds::identity(1);
        let pre = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[1]), l),
            ArrayD::from_elem(IxDyn(&[1]), u),
        )
        .expect("valid pre-activation bounds");
        let out = layer
            .propagate_linear_with_bounds(&seed, &pre)
            .expect("ReLU CROWN backward");
        let p4: Band = (
            out.lower_a()[[0, 0]],
            out.lower_b()[0],
            out.upper_a()[[0, 0]],
            out.upper_b()[0],
        );

        // The stored production coefficients carry composition directed
        // rounding whose per-coefficient error lives in the LinearBounds err
        // matrices, consumed by `concretize_sound` — so production soundness
        // is asserted on the concretized output interval (the form verdict
        // paths consume), which must enclose [ReLU(l), ReLU(u)] exactly.
        let conc = out.concretize_sound(&pre);
        let (relu_l, relu_u) = (l.max(0.0), u.max(0.0));
        assert!(
            conc.lower()[[0]] <= relu_l && conc.upper()[[0]] >= relu_u,
            "relu prod concretized bound does not enclose ReLU range at \
             l={l:e} u={u:e}: [{:e}, {:e}] vs [{relu_l:e}, {relu_u:e}] (band={p4:?})",
            conc.lower()[[0]],
            conc.upper()[[0]],
        );

        // The lower-envelope heuristic (identity when u > -l, else zero) is
        // shared; production coefficients may carry composition directed
        // rounding, so compare the selected envelope, not exact bits.
        let m_alpha = m4.0;
        let p_alpha = p4.0;
        assert!(
            (m_alpha - p_alpha).abs() <= 2.0 * f32::EPSILON,
            "relu lower-envelope selection drift at l={l:e} u={u:e}: \
             mirror={m4:?} prod={p4:?}"
        );
    }
}
