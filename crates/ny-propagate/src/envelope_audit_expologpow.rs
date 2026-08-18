//! TEMPORARY audit harness (exp / log / pow / reciprocal envelope validity).
//! Not part of the shipped crate — delete after the audit.
#![allow(clippy::all)]

use crate::layers::activations::LinearRelaxation;
use crate::layers::activations::{exp_linear_relaxation, log_linear_relaxation};
use crate::layers::arithmetic::pow_relaxation::{
    pow2_linear_relaxation, pow_neg1_linear_relaxation,
    pow_positive_integer_nonnegative_linear_relaxation,
};
use crate::layers::misc::reciprocal::{
    reciprocal_linear_relaxation, reciprocal_linear_relaxation_with_alpha,
};

// ---------------------------------------------------------------- exact tools

/// Exact sum of two f64 (Knuth TwoSum).
#[inline]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bp = s - a;
    let ap = s - bp;
    let e = (a - ap) + (b - bp);
    (s, e)
}

/// line(x) - f(x), computed so that catastrophic cancellation does not hide
/// small violations. `slope`/`intercept` are f32 (so slope*x is EXACT in f64
/// when x is f32; when x is a general f64 the product is rounded, and we
/// account for that with an fma-based residual). `fx` is the f64 oracle.
#[inline]
fn line_minus_f(slope: f32, intercept: f32, x: f64, fx: f64) -> f64 {
    let s = f64::from(slope);
    let p = s * x;
    // exact product residual
    let p_err = f64::mul_add(s, x, -p);
    let (hi, lo) = two_sum(p, f64::from(intercept));
    // (hi + lo + p_err) - fx, ordered so the dominant cancellation is exact.
    ((hi - fx) + lo) + p_err
}

/// ULP of f64 value v (as an f32-scale reference we also report f32 ulps).
fn ulp_f32_of(v: f64) -> f64 {
    let a = v.abs() as f32;
    if a == 0.0 {
        return f64::from(f32::from_bits(1));
    }
    if !a.is_finite() {
        return f64::INFINITY;
    }
    let up = ny_tensor::next_up_f32(a);
    f64::from(up) - f64::from(a)
}

#[derive(Clone, Debug)]
struct Worst {
    l: f32,
    u: f32,
    param: f64,
    x: f64,
    viol: f64,
    fx: f64,
    rel: LinearRelaxation,
    which: &'static str,
}

#[derive(Default)]
struct Acc {
    worst: Option<Worst>,
    count: u64,
    checked: u64,
}

impl Acc {
    fn offer(&mut self, w: Worst) {
        if w.viol > 0.0 {
            self.count += 1;
            let better = match &self.worst {
                None => true,
                Some(c) => w.viol / (w.fx.abs().max(1e-300)) > c.viol / (c.fx.abs().max(1e-300)),
            };
            if better {
                self.worst = Some(w);
            }
        }
    }
    fn report(&self, name: &str) {
        println!(
            "=== {name}: checked {} points, {} violating points",
            self.checked, self.count
        );
        if let Some(w) = &self.worst {
            let ulps = w.viol / ulp_f32_of(w.fx);
            println!(
                "    WORST [{}]: l={:e} ({:#x}) u={:e} ({:#x}) param={:e}\n      x={:.17e}  f(x)={:.17e}\n      slopes/intercepts: ls={:e} li={:e} us={:e} ui={:e}\n      violation={:.6e}  ({:.3} f32-ULPs of f(x))",
                w.which,
                w.l,
                w.l.to_bits(),
                w.u,
                w.u.to_bits(),
                w.param,
                w.x,
                w.fx,
                w.rel.lower_slope,
                w.rel.lower_intercept,
                w.rel.upper_slope,
                w.rel.upper_intercept,
                w.viol,
                ulps
            );
        }
    }
}

// ---------------------------------------------------------------- RNG

struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        // splitmix64
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Random positive normal f32 with exponent field in [emin, emax].
    fn pos_f32(&mut self, emin: i32, emax: i32) -> f32 {
        let e = (emin + (self.next_u64() % ((emax - emin + 1) as u64)) as i32) as u32;
        let m = (self.next_u64() as u32) & 0x7F_FFFF;
        f32::from_bits((e << 23) | m)
    }
}

// ---------------------------------------------------------------- generic check

/// Check the envelope obligation on [l,u] for oracle `f` with derivative-inverse
/// `crit` (given a slope, returns the x where f'(x) == slope, or None).
fn check_interval<F, C>(
    acc_lo: &mut Acc,
    acc_hi: &mut Acc,
    l: f32,
    u: f32,
    param: f64,
    rel: LinearRelaxation,
    f: F,
    crit: C,
    tag_lo: &'static str,
    tag_hi: &'static str,
) where
    F: Fn(f64) -> f64,
    C: Fn(f64) -> Option<f64>,
{
    let l64 = f64::from(l);
    let u64_ = f64::from(u);
    if !(l64 <= u64_) || !l64.is_finite() {
        return;
    }
    let hi_end = if u64_.is_finite() {
        u64_
    } else {
        l64 * 1e6 + 1e6
    };

    let mut xs: Vec<f64> = Vec::with_capacity(600);
    xs.push(l64);
    xs.push(hi_end);
    // f32-grid + real grid
    let n = 200;
    for i in 0..=n {
        let t = i as f64 / n as f64;
        xs.push(l64 + (hi_end - l64) * t);
        // geometric grid too (matters when l,u span many decades)
        if l64 > 0.0 && hi_end > 0.0 {
            xs.push(l64 * (hi_end / l64).powf(t));
        }
        if hi_end < 0.0 {
            xs.push(-((-l64) * ((-hi_end) / (-l64)).powf(t)));
        }
    }
    // neighbours of endpoints
    for v in [l64, hi_end] {
        let vf = v as f32;
        xs.push(f64::from(ny_tensor::next_up_f32(vf)));
        xs.push(f64::from(ny_tensor::next_down_f32(vf)));
        xs.push(v * (1.0 + 1e-15));
        xs.push(v * (1.0 - 1e-15));
    }
    // analytic interior critical points for BOTH lines
    for slope in [f64::from(rel.lower_slope), f64::from(rel.upper_slope)] {
        if let Some(xc) = crit(slope) {
            if xc.is_finite() {
                for d in [0.0, 1e-16, -1e-16, 1e-12, -1e-12, 1e-8, -1e-8] {
                    xs.push(xc * (1.0 + d));
                }
                xs.push(f64::from(xc as f32));
                xs.push(f64::from(ny_tensor::next_up_f32(xc as f32)));
                xs.push(f64::from(ny_tensor::next_down_f32(xc as f32)));
            }
        }
    }

    for &x in &xs {
        if !(x >= l64 && x <= hi_end) {
            continue;
        }
        let fx = f(x);
        if !fx.is_finite() {
            continue;
        }
        acc_lo.checked += 1;
        acc_hi.checked += 1;
        // lower line must be <= f(x)
        let d_lo = line_minus_f(rel.lower_slope, rel.lower_intercept, x, fx);
        if d_lo > 0.0 {
            acc_lo.offer(Worst {
                l,
                u,
                param,
                x,
                viol: d_lo,
                fx,
                rel: rel.clone(),
                which: tag_lo,
            });
        }
        // upper line must be >= f(x)
        let d_hi = -line_minus_f(rel.upper_slope, rel.upper_intercept, x, fx);
        if d_hi > 0.0 {
            acc_hi.offer(Worst {
                l,
                u,
                param,
                x,
                viol: d_hi,
                fx,
                rel: rel.clone(),
                which: tag_hi,
            });
        }
    }
}

// ---------------------------------------------------------------- EXP

#[test]
fn audit_exp_envelope() {
    let mut lo = Acc::default();
    let mut hi = Acc::default();
    let f = |x: f64| x.exp();
    let crit = |s: f64| if s > 0.0 { Some(s.ln()) } else { None };

    let check = |l: f32, u: f32, lo: &mut Acc, hi: &mut Acc| {
        let rel = exp_linear_relaxation(l, u);
        check_interval(lo, hi, l, u, 0.0, rel, f, crit, "exp-lower", "exp-upper");
    };

    // degenerate / boundary
    let specials: &[f32] = &[
        -87.0, -50.0, -20.0, -1.0, -1e-10, -0.0, 0.0, 1e-10, 1e-30, 1.0, 10.0, 19.25, 40.0, 80.0,
        87.9, 88.0,
    ];
    for &a in specials {
        for &b in specials {
            if a <= b {
                check(a, b, &mut lo, &mut hi);
            }
        }
        check(a, a, &mut lo, &mut hi); // point interval
        check(a, ny_tensor::next_up_f32(a), &mut lo, &mut hi);
        check(a, a + 1e-9, &mut lo, &mut hi);
    }
    // subnormal widths around every magnitude
    let mut r = Rng(0xC0FFEE_1234);
    for _ in 0..40_000 {
        let a = (r.f64() * 175.0 - 87.0) as f32;
        let w = 10f64.powf(r.f64() * 12.0 - 10.0) as f32;
        let b = a + w;
        if b <= 88.0 && b.is_finite() {
            check(a, b, &mut lo, &mut hi);
        }
    }
    // wide intervals
    for _ in 0..10_000 {
        let a = (r.f64() * 175.0 - 87.0) as f32;
        let b = (r.f64() * 175.0 - 87.0) as f32;
        if a <= b {
            check(a, b, &mut lo, &mut hi);
        }
    }
    lo.report("exp lower");
    hi.report("exp upper");
    assert!(lo.count == 0 && hi.count == 0, "exp envelope violations");
}

// ---------------------------------------------------------------- LOG

#[test]
fn audit_log_envelope() {
    let mut lo = Acc::default();
    let mut hi = Acc::default();
    let f = |x: f64| x.ln();
    let crit = |s: f64| if s > 0.0 { Some(1.0 / s) } else { None };

    let check = |l: f32, u: f32, lo: &mut Acc, hi: &mut Acc| {
        let rel = log_linear_relaxation(l, u);
        check_interval(lo, hi, l, u, 0.0, rel, f, crit, "log-lower", "log-upper");
    };

    let specials: &[f32] = &[
        1e-38, 1.2e-38, 1e-30, 1e-10, 1e-6, 1e-3, 0.1, 0.5, 1.0, 2.7182817, 2.718282, 3.0, 10.0,
        1e3, 1e10, 1e20, 1e30, 3.4e38,
    ];
    for &a in specials {
        for &b in specials {
            if a <= b {
                check(a, b, &mut lo, &mut hi);
            }
        }
        check(a, a, &mut lo, &mut hi);
        check(a, ny_tensor::next_up_f32(a), &mut lo, &mut hi);
        check(a, a * (1.0 + 1e-7), &mut lo, &mut hi);
        check(a, a * (1.0 + 1e-6), &mut lo, &mut hi);
        check(a, a * 1.000001, &mut lo, &mut hi);
    }

    let mut r = Rng(0xBADC0DE_99);
    // all normal exponents
    for _ in 0..60_000 {
        let a = r.pos_f32(1, 254);
        let ratio = 1.0 + 10f64.powf(r.f64() * 12.0 - 9.0);
        let b = (f64::from(a) * ratio) as f32;
        if b.is_finite() && b >= a {
            check(a, b, &mut lo, &mut hi);
        }
    }
    // independent endpoints (very wide)
    for _ in 0..20_000 {
        let a = r.pos_f32(1, 254);
        let b = r.pos_f32(1, 254);
        if a <= b {
            check(a, b, &mut lo, &mut hi);
        }
    }
    // tangent-point-near-e sweep: pick u, solve for l so that d = 1/chord ~ e
    for _ in 0..20_000 {
        let u = r.pos_f32(100, 200);
        let k = 1.0 + r.f64() * 60.0;
        let l = (f64::from(u) / k.exp()) as f32;
        if l > 0.0 && l.is_finite() {
            check(l, u, &mut lo, &mut hi);
        }
    }
    lo.report("log lower");
    hi.report("log upper");
    assert!(lo.count == 0 && hi.count == 0, "log envelope violations");
}

// ---------------------------------------------------------------- POW2

#[test]
fn audit_pow2_envelope() {
    let mut lo = Acc::default();
    let mut hi = Acc::default();
    let f = |x: f64| x * x;
    let crit = |s: f64| Some(s / 2.0);
    let check = |l: f32, u: f32, lo: &mut Acc, hi: &mut Acc| {
        let rel = pow2_linear_relaxation(l, u);
        check_interval(lo, hi, l, u, 0.0, rel, f, crit, "pow2-lower", "pow2-upper");
    };
    let specials: &[f32] = &[
        -1e19, -1e10, -1e3, -3.0, -1.0, -1e-5, -1e-20, -1e-38, -0.0, 0.0, 1e-38, 1e-20, 1e-5, 1.0,
        3.0, 1e3, 1e10, 1e19,
    ];
    for &a in specials {
        for &b in specials {
            if a <= b {
                check(a, b, &mut lo, &mut hi);
            }
        }
        check(a, a, &mut lo, &mut hi);
        check(a, ny_tensor::next_up_f32(a), &mut lo, &mut hi);
    }
    let mut r = Rng(0x1122_3344);
    for _ in 0..60_000 {
        let sa = if r.next_u64() & 1 == 0 { -1.0 } else { 1.0 };
        let sb = if r.next_u64() & 1 == 0 { -1.0 } else { 1.0 };
        let a = sa * f64::from(r.pos_f32(1, 250));
        let b = sb * f64::from(r.pos_f32(1, 250));
        let (a, b) = if a <= b { (a, b) } else { (b, a) };
        let (a, b) = (a as f32, b as f32);
        if a.is_finite() && b.is_finite() && a <= b {
            check(a, b, &mut lo, &mut hi);
        }
    }
    // narrow intervals at all scales
    for _ in 0..40_000 {
        let a = f64::from(r.pos_f32(1, 250)) * if r.next_u64() & 1 == 0 { -1.0 } else { 1.0 };
        let w = a.abs() * 10f64.powf(r.f64() * 10.0 - 10.0);
        let (l, u) = ((a) as f32, (a + w) as f32);
        if l.is_finite() && u.is_finite() && l <= u {
            check(l, u, &mut lo, &mut hi);
        }
    }
    lo.report("pow2 lower");
    hi.report("pow2 upper");
    assert!(lo.count == 0 && hi.count == 0, "pow2 envelope violations");
}

// ---------------------------------------------------------------- POW -1

#[test]
fn audit_pow_neg1_envelope() {
    let mut lo = Acc::default();
    let mut hi = Acc::default();
    let f = |x: f64| 1.0 / x;
    let crit = |s: f64| {
        if s < 0.0 {
            Some((-1.0 / s).sqrt())
        } else {
            None
        }
    };
    let check = |l: f32, u: f32, lo: &mut Acc, hi: &mut Acc| {
        let rel = pow_neg1_linear_relaxation(l, u);
        check_interval(
            lo,
            hi,
            l,
            u,
            0.0,
            rel,
            f,
            crit,
            "pow_neg1-lower",
            "pow_neg1-upper",
        );
    };
    let specials: &[f32] = &[
        1e-38, 1e-30, 1e-10, 1e-3, 0.5, 1.0, 2.0, 3.0, 10.0, 1e3, 1e10, 1e20, 1e30, 3.4e38,
    ];
    for &a in specials {
        for &b in specials {
            if a <= b {
                check(a, b, &mut lo, &mut hi);
            }
        }
        check(a, a, &mut lo, &mut hi);
        check(a, ny_tensor::next_up_f32(a), &mut lo, &mut hi);
    }
    let mut r = Rng(0x5566_7788);
    for _ in 0..60_000 {
        let a = r.pos_f32(1, 254);
        let ratio = 1.0 + 10f64.powf(r.f64() * 12.0 - 9.0);
        let b = (f64::from(a) * ratio) as f32;
        if b.is_finite() && b >= a {
            check(a, b, &mut lo, &mut hi);
        }
    }
    for _ in 0..20_000 {
        let a = r.pos_f32(1, 254);
        let b = r.pos_f32(1, 254);
        if a <= b {
            check(a, b, &mut lo, &mut hi);
        }
    }
    lo.report("pow_neg1 lower");
    hi.report("pow_neg1 upper");
    assert!(
        lo.count == 0 && hi.count == 0,
        "pow_neg1 envelope violations"
    );
}

// ---------------------------------------------------------------- POW p>=2

#[test]
fn audit_pow_positive_integer_envelope() {
    let mut lo = Acc::default();
    let mut hi = Acc::default();
    let mut r = Rng(0x99AA_BBCC);
    for p in 2..=8i32 {
        let f = move |x: f64| x.powi(p);
        let crit = move |s: f64| {
            if s > 0.0 {
                Some((s / f64::from(p)).powf(1.0 / f64::from(p - 1)))
            } else {
                None
            }
        };
        let check = |l: f32, u: f32, lo: &mut Acc, hi: &mut Acc| {
            let rel = pow_positive_integer_nonnegative_linear_relaxation(p, l, u);
            check_interval(
                lo,
                hi,
                l,
                u,
                f64::from(p),
                rel,
                &f,
                &crit,
                "powp-lower",
                "powp-upper",
            );
        };
        let specials: &[f32] = &[
            0.0, 1e-38, 1e-20, 1e-6, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0, 1e4, 1e8,
        ];
        for &a in specials {
            for &b in specials {
                if a <= b && f64::from(b).powi(p) < 1e300 {
                    check(a, b, &mut lo, &mut hi);
                }
            }
            check(a, a, &mut lo, &mut hi);
            check(a, ny_tensor::next_up_f32(a), &mut lo, &mut hi);
        }
        for _ in 0..20_000 {
            let a = r.pos_f32(1, 200);
            let ratio = 1.0 + 10f64.powf(r.f64() * 10.0 - 9.0);
            let b = (f64::from(a) * ratio) as f32;
            if b.is_finite() && b >= a && f64::from(b).powi(p) < 1e300 {
                check(a, b, &mut lo, &mut hi);
            }
        }
    }
    lo.report("pow p>=2 lower");
    hi.report("pow p>=2 upper");
    assert!(lo.count == 0 && hi.count == 0, "pow p envelope violations");
}

// ---------------------------------------------------------------- RECIPROCAL

#[test]
fn audit_reciprocal_envelope() {
    let mut lo = Acc::default();
    let mut hi = Acc::default();
    let f = |x: f64| 1.0 / x;
    let crit = |s: f64| {
        if s < 0.0 {
            Some((-1.0 / s).sqrt())
        } else {
            None
        }
    };
    let mut r = Rng(0xDEAD_BEEF_01);

    let check = |l: f32, u: f32, mid: Option<f32>, lo: &mut Acc, hi: &mut Acc| {
        let rel = match mid {
            None => reciprocal_linear_relaxation(l, u),
            Some(m) => reciprocal_linear_relaxation_with_alpha(l, u, m),
        };
        // both-negative intervals: geometric grid handled in check_interval
        check_interval(
            lo,
            hi,
            l,
            u,
            mid.map(f64::from).unwrap_or(f64::NAN),
            rel,
            f,
            crit,
            "recip-lower",
            "recip-upper",
        );
    };

    let specials: &[f32] = &[
        1e-38, 1e-30, 1e-10, 1e-3, 0.5, 1.0, 2.0, 10.0, 1e3, 1e10, 1e20, 1e30, 3.4e38,
    ];
    for &a in specials {
        for &b in specials {
            if a <= b {
                check(a, b, None, &mut lo, &mut hi);
                check(-b, -a, None, &mut lo, &mut hi);
                for t in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
                    let m = a + (b - a) * t;
                    check(a, b, Some(m), &mut lo, &mut hi);
                    check(-b, -a, Some(-m), &mut lo, &mut hi);
                }
            }
        }
        check(a, a, None, &mut lo, &mut hi);
        check(-a, -a, None, &mut lo, &mut hi);
        check(a, ny_tensor::next_up_f32(a), None, &mut lo, &mut hi);
    }

    for _ in 0..40_000 {
        let a = r.pos_f32(1, 254);
        let ratio = 1.0 + 10f64.powf(r.f64() * 12.0 - 9.0);
        let b = (f64::from(a) * ratio) as f32;
        if !(b.is_finite() && b >= a) {
            continue;
        }
        check(a, b, None, &mut lo, &mut hi);
        check(-b, -a, None, &mut lo, &mut hi);
        let t = r.f64() as f32;
        let m = a + (b - a) * t;
        check(a, b, Some(m), &mut lo, &mut hi);
        check(-b, -a, Some(-m), &mut lo, &mut hi);
        // out-of-range alpha (clamped inside)
        check(a, b, Some(a * 0.1), &mut lo, &mut hi);
        check(a, b, Some(b * 10.0), &mut lo, &mut hi);
    }
    for _ in 0..20_000 {
        let a = r.pos_f32(1, 254);
        let b = r.pos_f32(1, 254);
        if a <= b {
            check(a, b, None, &mut lo, &mut hi);
            check(-b, -a, None, &mut lo, &mut hi);
        }
    }
    lo.report("reciprocal lower");
    hi.report("reciprocal upper");
    assert!(
        lo.count == 0 && hi.count == 0,
        "reciprocal envelope violations"
    );
}
