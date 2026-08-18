// TEMPORARY AUDIT HARNESS — not for commit.
//! Envelope-validity checker used by the silu/mish envelope audit tests.
//!
//! Rules enforced here (see audit methodology):
//!  - the obligation is NEVER evaluated in f32; everything is f64
//!  - the reference f is evaluated independently, in f64, with stable formulas
//!  - interior extrema are found by dense scan + golden-section refinement
//!    + Newton on v'(x) = 0, not by endpoint checking

#![allow(dead_code)]

use super::LinearRelaxation;

/// Independent f64 SiLU: x * sigmoid(x).
pub(crate) fn silu_ref(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == f64::INFINITY {
        return x;
    }
    if x == f64::NEG_INFINITY {
        return 0.0;
    }
    if x >= 0.0 {
        x / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        x * e / (1.0 + e)
    }
}

pub(crate) fn silu_deriv_ref(x: f64) -> f64 {
    let s = if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    };
    s * (1.0 + x * (1.0 - s))
}

/// Independent f64 Mish: x * tanh(softplus(x)).
pub(crate) fn mish_ref(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == f64::INFINITY {
        return x;
    }
    if x == f64::NEG_INFINITY {
        return 0.0;
    }
    // softplus, stable
    let sp = if x > 33.0 {
        x
    } else if x < -37.0 {
        x.exp()
    } else if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    };
    x * sp.tanh()
}

pub(crate) fn mish_deriv_ref(x: f64) -> f64 {
    let sp = if x > 33.0 {
        x
    } else if x < -37.0 {
        x.exp()
    } else if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    };
    let t = sp.tanh();
    let sig = if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    };
    t + x * (1.0 - t * t) * sig
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Viol {
    /// x where the obligation fails worst (violation amount is maximal)
    pub x: f64,
    /// how much the line is on the wrong side (positive = violation)
    pub amount: f64,
    /// f(x) at that point
    pub fx: f64,
    /// line(x) at that point
    pub line: f64,
    /// true = lower line above f; false = upper line below f
    pub is_lower: bool,
}

/// f32 ulp at magnitude |v| (as f64).
pub(crate) fn ulp_f32_at(v: f64) -> f64 {
    let a = v.abs() as f32;
    if a == 0.0 {
        return f32::from_bits(1) as f64;
    }
    if !a.is_finite() {
        return f64::INFINITY;
    }
    let up = f32::from_bits(a.to_bits() + 1);
    (up as f64) - (a as f64)
}

/// Maximize v over [l, u]: dense scan, golden-section refinement of the best
/// brackets, plus Newton on v'(x) = 0 from several starts.
fn maximize<V, D>(l: f64, u: f64, v: &V, dv: &D, n: usize) -> (f64, f64)
where
    V: Fn(f64) -> f64,
    D: Fn(f64) -> f64,
{
    let mut best_x = l;
    let mut best_v = v(l);
    let vu = v(u);
    if vu > best_v {
        best_v = vu;
        best_x = u;
    }
    if l == u {
        return (best_x, best_v);
    }

    // Dense scan; remember the top few brackets.
    let mut idx_vals: Vec<(usize, f64)> = Vec::with_capacity(n + 1);
    for i in 0..=n {
        let t = i as f64 / n as f64;
        let x = l + t * (u - l);
        let x = if i == n { u } else { x };
        let val = v(x);
        idx_vals.push((i, val));
        if val > best_v {
            best_v = val;
            best_x = x;
        }
    }
    let mut order = idx_vals.clone();
    order.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let pt = |i: usize| -> f64 {
        if i >= n {
            u
        } else {
            l + (i as f64 / n as f64) * (u - l)
        }
    };

    // Golden-section maximize inside the top brackets.
    let gr = 0.618_033_988_749_895_f64;
    for &(i, _) in order.iter().take(6) {
        let a0 = pt(i.saturating_sub(1));
        let b0 = pt((i + 1).min(n));
        let (mut a, mut b) = (a0, b0);
        if !(a < b) {
            continue;
        }
        let mut c = b - gr * (b - a);
        let mut d = a + gr * (b - a);
        let mut fc = v(c);
        let mut fd = v(d);
        for _ in 0..200 {
            if fc > fd {
                b = d;
                d = c;
                fd = fc;
                c = b - gr * (b - a);
                fc = v(c);
            } else {
                a = c;
                c = d;
                fc = fd;
                d = a + gr * (b - a);
                fd = v(d);
            }
            if (b - a).abs() <= f64::EPSILON * (a.abs() + b.abs() + 1.0) {
                break;
            }
        }
        for &x in &[a, b, c, d, 0.5 * (a + b)] {
            let x = x.clamp(l, u);
            let val = v(x);
            if val > best_v {
                best_v = val;
                best_x = x;
            }
        }
    }

    // Newton on v'(x) = 0 using a numerical second derivative of v.
    let starts = [l, u, 0.5 * (l + u), l + 0.25 * (u - l), l + 0.75 * (u - l), best_x];
    for &s in &starts {
        let mut x = s;
        for _ in 0..80 {
            let h = (x.abs() * 1e-6).max(1e-9);
            let g = dv(x);
            let gp = (dv(x + h) - dv(x - h)) / (2.0 * h);
            if !gp.is_finite() || gp.abs() < 1e-300 {
                break;
            }
            let step = g / gp;
            if !step.is_finite() {
                break;
            }
            let nx = (x - step).clamp(l, u);
            if nx == x {
                break;
            }
            x = nx;
        }
        let val = v(x);
        if val > best_v {
            best_v = val;
            best_x = x;
        }
    }
    (best_x, best_v)
}

/// Check the envelope obligation for `r` on [l, u] against reference `f`
/// (derivative `fp`). Returns the worst violation if any.
pub(crate) fn check_envelope<F, P>(
    l: f32,
    u: f32,
    r: &LinearRelaxation,
    f: &F,
    fp: &P,
    n: usize,
    extra_pts: &[f64],
) -> Option<Viol>
where
    F: Fn(f64) -> f64,
    P: Fn(f64) -> f64,
{
    let l64 = l as f64;
    let u64 = u as f64;
    if !l64.is_finite() || !u64.is_finite() || l64 > u64 {
        return None;
    }
    let ls = r.lower_slope as f64;
    let li = r.lower_intercept as f64;
    let us = r.upper_slope as f64;
    let ui = r.upper_intercept as f64;

    let mut worst: Option<Viol> = None;
    let mut record = |x: f64, amount: f64, fx: f64, line: f64, is_lower: bool| {
        if amount > 0.0 && amount.is_finite() {
            let better = match &worst {
                None => true,
                Some(w) => amount > w.amount,
            };
            if better {
                worst = Some(Viol {
                    x,
                    amount,
                    fx,
                    line,
                    is_lower,
                });
            }
        }
    };

    // ---- lower obligation: ls*x + li <= f(x) ; violation v(x) = line - f
    if li.is_finite() && ls.is_finite() {
        let v = |x: f64| (ls * x + li) - f(x);
        let dv = |x: f64| ls - fp(x);
        let (bx, bv) = maximize(l64, u64, &v, &dv, n);
        record(bx, bv, f(bx), ls * bx + li, true);
        for &x in extra_pts {
            if x >= l64 && x <= u64 {
                record(x, v(x), f(x), ls * x + li, true);
            }
        }
    }
    // ---- upper obligation: f(x) <= us*x + ui ; violation v(x) = f - line
    if ui.is_finite() && us.is_finite() {
        let v = |x: f64| f(x) - (us * x + ui);
        let dv = |x: f64| fp(x) - us;
        let (bx, bv) = maximize(l64, u64, &v, &dv, n);
        record(bx, bv, f(bx), us * bx + ui, false);
        for &x in extra_pts {
            if x >= l64 && x <= u64 {
                record(x, v(x), f(x), us * x + ui, false);
            }
        }
    }
    // NaN coefficients are themselves an envelope failure (no bound at all).
    if r.lower_slope.is_nan()
        || r.lower_intercept.is_nan()
        || r.upper_slope.is_nan()
        || r.upper_intercept.is_nan()
    {
        return Some(Viol {
            x: l64,
            amount: f64::NAN,
            fx: f(l64),
            line: f64::NAN,
            is_lower: true,
        });
    }
    worst
}

/// Tiny deterministic PRNG (xorshift64*).
pub(crate) struct Rng(pub u64);
impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}
