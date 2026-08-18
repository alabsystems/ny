// TEMPORARY envelope-validity audit harness — SIGMOID-FAMILY batch (not for merge).
//! Obligation checked, for all x in [l, u]:
//!   lower_slope*x + lower_intercept <= f(x) <= upper_slope*x + upper_intercept
//! Everything is evaluated in f64 (an f32*f32 product is exact in f64); f(x) uses an
//! f64 (exactly-rational for these four activations) oracle, never an f32 eval.

#![allow(clippy::all)]

use super::clip::audit_clip_relax;
use super::hard_sigmoid::audit_hard_sigmoid_relax;
use super::hard_swish::audit_hardswish_relax;
use super::softsign::audit_softsign_relax;
use super::LinearRelaxation;
use ny_tensor::next_up_f32;

// ── plumbing ────────────────────────────────────────────────────────────

fn next_down_f32_l(x: f32) -> f32 {
    -next_up_f32(-x)
}

fn next_up_f64(x: f64) -> f64 {
    if x.is_nan() || x == f64::INFINITY {
        return x;
    }
    if x == 0.0 {
        return f64::from_bits(1);
    }
    if x > 0.0 {
        f64::from_bits(x.to_bits() + 1)
    } else {
        f64::from_bits(x.to_bits() - 1)
    }
}
fn next_down_f64(x: f64) -> f64 {
    -next_up_f64(-x)
}

fn ulp_of_f32(v: f64) -> f64 {
    let a = (v as f32).abs();
    if a == 0.0 || !a.is_finite() {
        return f32::from_bits(1) as f64;
    }
    (next_up_f32(a) - a) as f64
}

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }
    fn u32(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 29) as u32
    }
    fn f32_any(&mut self) -> f32 {
        loop {
            let v = f32::from_bits(self.u32());
            if v.is_finite() {
                return v;
            }
        }
    }
    fn f32_moderate(&mut self) -> f32 {
        let e = (self.u32() % 61) as i32 - 30;
        let m = 1.0 + (self.u32() % (1 << 23)) as f32 / (1u32 << 23) as f32;
        let s = if self.u32() & 1 == 0 { 1.0 } else { -1.0 };
        s * m * 2.0f32.powi(e)
    }
    fn unit(&mut self) -> f64 {
        (self.u32() % 1_000_003) as f64 / 1_000_003.0
    }
}

#[derive(Clone)]
struct Viol {
    ulps: f64,
    gap: f64,
    l: f32,
    u: f32,
    params: String,
    x: f64,
    fx: f64,
    line: f64,
    coeffs: [f32; 4],
    side: &'static str,
}

struct Collector {
    tag: &'static str,
    n_checked: usize,
    n_viol: usize,
    worst: Vec<Viol>,
}

impl Collector {
    fn new(tag: &'static str) -> Self {
        Collector {
            tag,
            n_checked: 0,
            n_viol: 0,
            worst: Vec::new(),
        }
    }
    fn push(&mut self, v: Viol) {
        self.n_viol += 1;
        self.worst.push(v);
        if self.worst.len() > 600 {
            self.worst.sort_by(|a, b| {
                b.gap
                    .partial_cmp(&a.gap)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            self.worst.truncate(60);
        }
    }
    fn report(&mut self) {
        self.worst.sort_by(|a, b| {
            b.gap
                .partial_cmp(&a.gap)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        println!(
            "\n===== {}: {} intervals checked, {} point-violations =====",
            self.tag, self.n_checked, self.n_viol
        );
        let mut seen: Vec<(u32, u32, String, &str)> = Vec::new();
        let mut shown = 0;
        for v in &self.worst {
            let key = (v.l.to_bits(), v.u.to_bits(), v.params.clone(), v.side);
            if seen.iter().any(|k| *k == key) {
                continue;
            }
            seen.push(key);
            println!(
                "  [{}] {} l={:e} ({:#010x}) u={:e} ({:#010x}) {}\n      coeffs ls={:e}({:#010x}) li={:e}({:#010x}) us={:e}({:#010x}) ui={:e}({:#010x})\n      x={:.17e} f(x)={:.17e} line={:.17e} gap={:.6e} ({:.4} ULP of f(x))",
                self.tag, v.side, v.l, v.l.to_bits(), v.u, v.u.to_bits(), v.params,
                v.coeffs[0], v.coeffs[0].to_bits(),
                v.coeffs[1], v.coeffs[1].to_bits(),
                v.coeffs[2], v.coeffs[2].to_bits(),
                v.coeffs[3], v.coeffs[3].to_bits(),
                v.x, v.fx, v.line, v.gap, v.ulps
            );
            shown += 1;
            if shown >= 14 {
                break;
            }
        }
        if self.n_viol == 0 {
            println!("  CLEAN");
        }
    }
}

/// Core obligation check. `f` is an f64 oracle; `crit` supplies the analytically
/// derived interior critical points of `line(x) - f(x)` (kinks + stationary points).
fn probe(
    c: &mut Collector,
    l: f32,
    u: f32,
    params: &str,
    r: LinearRelaxation,
    f: &dyn Fn(f64) -> f64,
    crit: &[f64],
) {
    c.n_checked += 1;
    let lo = l as f64;
    let hi = u as f64;
    if !(lo <= hi) {
        return;
    }

    let mut xs: Vec<f64> = Vec::with_capacity(512);
    xs.push(lo);
    xs.push(hi);
    for k in 0..=128 {
        xs.push(lo + (k as f64 / 128.0) * (hi - lo));
    }
    for &k in crit {
        if !k.is_finite() {
            continue;
        }
        let k = if k < lo {
            lo
        } else if k > hi {
            hi
        } else {
            k
        };
        xs.push(k);
        xs.push(next_up_f64(k));
        xs.push(next_down_f64(k));
        // f32-representable neighbours (production x values are f32)
        let a = k as f32;
        for cand in [
            a as f64,
            next_up_f32(a) as f64,
            next_down_f32_l(a) as f64,
            next_up_f32(next_up_f32(a)) as f64,
            next_down_f32_l(next_down_f32_l(a)) as f64,
        ] {
            xs.push(cand);
        }
        for rel in [1e-15f64, 1e-12, 1e-9, 1e-6, 1e-3] {
            let d = k.abs().max(1e-30) * rel;
            xs.push(k + d);
            xs.push(k - d);
        }
    }
    for &e in [lo, hi].iter() {
        xs.push(next_up_f64(e));
        xs.push(next_down_f64(e));
    }

    let ls = r.lower_slope as f64;
    let li = r.lower_intercept as f64;
    let us = r.upper_slope as f64;
    let ui = r.upper_intercept as f64;
    let coeffs = [
        r.lower_slope,
        r.lower_intercept,
        r.upper_slope,
        r.upper_intercept,
    ];
    if ls.is_nan() || li.is_nan() || us.is_nan() || ui.is_nan() {
        c.push(Viol {
            ulps: f64::INFINITY,
            gap: f64::INFINITY,
            l,
            u,
            params: params.to_string(),
            x: f64::NAN,
            fx: f64::NAN,
            line: f64::NAN,
            coeffs,
            side: "NaN-coeff",
        });
        return;
    }

    for &x in &xs {
        if !x.is_finite() || x < lo || x > hi {
            continue;
        }
        let fx = f(x);
        if !fx.is_finite() {
            continue;
        }
        // Guard against f64 round-off in the oracle itself: only accept a
        // violation whose gap exceeds 1e-13 * scale (f64 has ~1e-16 relative).
        let mag = |a: f64, b: f64, d: f64| a.abs().max(b.abs()).max(d.abs());

        if li != f64::NEG_INFINITY {
            let line = ls * x + li;
            let gap = line - fx;
            let scale = mag(ls * x, li, fx).max(f64::MIN_POSITIVE);
            if gap > 1e-13 * scale && gap > 0.0 {
                c.push(Viol {
                    ulps: gap / ulp_of_f32(fx),
                    gap,
                    l,
                    u,
                    params: params.to_string(),
                    x,
                    fx,
                    line,
                    coeffs,
                    side: "LOWER>f",
                });
            }
        }
        if ui != f64::INFINITY {
            let line = us * x + ui;
            let gap = fx - line;
            let scale = mag(us * x, ui, fx).max(f64::MIN_POSITIVE);
            if gap > 1e-13 * scale && gap > 0.0 {
                c.push(Viol {
                    ulps: gap / ulp_of_f32(fx),
                    gap,
                    l,
                    u,
                    params: params.to_string(),
                    x,
                    fx,
                    line,
                    coeffs,
                    side: "UPPER<f",
                });
            }
        }
    }
}

// ── interval sampling ───────────────────────────────────────────────────

fn specials() -> Vec<f32> {
    vec![
        0.0,
        -0.0,
        f32::from_bits(1),
        -f32::from_bits(1),
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        1e-30,
        -1e-30,
        1e-9,
        -1e-9,
        1e-8,
        -1e-8,
        1e-7,
        -1e-7,
        0.1,
        -0.1,
        0.5,
        -0.5,
        1.0,
        -1.0,
        2.0,
        -2.0,
        3.0,
        -3.0,
        6.0,
        -6.0,
        1e3,
        -1e3,
        1e10,
        -1e10,
        1e30,
        -1e30,
        f32::MAX,
        f32::MIN,
    ]
}

/// All-exponent coarse pool across the whole normal f32 range.
fn exponent_pool() -> Vec<f32> {
    let mut v = Vec::new();
    let mut e = -126i32;
    while e <= 127 {
        for m in [0u32, 3, 6] {
            let b = ((1.0f64 + m as f64 / 8.0) * (2.0f64).powi(e)) as f32;
            if b.is_finite() && b != 0.0 {
                v.push(b);
                v.push(-b);
            }
        }
        e += 5;
    }
    v
}

fn around(k: f32) -> Vec<f32> {
    let mut v = vec![k];
    let mut a = k;
    let mut b = k;
    for _ in 0..4 {
        a = next_up_f32(a);
        b = next_down_f32_l(b);
        v.push(a);
        v.push(b);
    }
    let mut d = 1e-38f32;
    // Geometric ladder of f32 offsets around the kink: the x5.7 stride is
    // chosen so the offsets land off the binade grid, so the float bound is
    // the point of the loop. Do not reshape it into an integer count.
    #[allow(clippy::while_float)]
    while d < 1e31 {
        v.push(k + d);
        v.push(k - d);
        d *= 5.7;
    }
    v.retain(|x| x.is_finite());
    v
}

fn sample_intervals(rng: &mut Rng, n: usize, kinks: &[f32]) -> Vec<(f32, f32)> {
    let mut v: Vec<(f32, f32)> = Vec::with_capacity(n + 4000);
    let sp = specials();
    for &a in &sp {
        v.push((a, a));
        for &b in &sp {
            if a <= b {
                v.push((a, b));
            }
        }
    }
    for &k in kinks {
        if !k.is_finite() {
            continue;
        }
        let pts = around(k);
        for i in 0..pts.len() {
            for j in 0..pts.len() {
                if pts[i] <= pts[j] {
                    v.push((pts[i], pts[j]));
                }
            }
        }
    }
    let ep = exponent_pool();
    for i in 0..ep.len() {
        for j in 0..ep.len() {
            if ep[i] <= ep[j] {
                v.push((ep[i], ep[j]));
            }
        }
    }
    for _ in 0..n {
        match rng.u32() % 7 {
            0 => {
                let a = rng.f32_any();
                let b = rng.f32_any();
                v.push((a.min(b), a.max(b)));
            }
            1 => {
                let a = rng.f32_moderate();
                let b = rng.f32_moderate();
                v.push((a.min(b), a.max(b)));
            }
            2 => {
                let a = -rng.f32_moderate().abs();
                let b = rng.f32_moderate().abs();
                v.push((a, b));
            }
            3 => {
                let a = rng.f32_moderate();
                let w = a.abs() * (rng.unit() as f32) * 1e-3;
                v.push((a, a + w));
            }
            4 => {
                let k = if kinks.is_empty() {
                    0.0
                } else {
                    kinks[(rng.u32() as usize) % kinks.len()]
                };
                let d = (k.abs().max(1.0)) * (rng.unit() as f32) * (rng.unit() as f32);
                v.push((k - d, k + d));
            }
            5 => {
                // one endpoint exactly on a kink
                let k = if kinks.is_empty() {
                    0.0
                } else {
                    kinks[(rng.u32() as usize) % kinks.len()]
                };
                let d = (k.abs().max(1.0)) * (rng.unit() as f32) * (rng.unit() as f32);
                if rng.u32() & 1 == 0 {
                    v.push((k, k + d));
                } else {
                    v.push((k - d, k));
                }
            }
            _ => {
                let a = rng.f32_any();
                let w = a.abs() * (rng.unit() as f32);
                let b = a + w;
                v.push((a.min(b), a.max(b)));
            }
        }
    }
    v.retain(|(a, b)| a.is_finite() && b.is_finite() && a <= b);
    v
}

// ── hard sigmoid ────────────────────────────────────────────────────────

fn hs_f64(x: f64, alpha: f64, beta: f64) -> f64 {
    let t = alpha * x + beta;
    if t < 0.0 {
        0.0
    } else if t > 1.0 {
        1.0
    } else {
        t
    }
}

#[test]
fn audit_hard_sigmoid_envelope() {
    let mut rng = Rng::new(0xAA01);
    let mut c = Collector::new("hard_sigmoid_linear_relaxation");
    let mut params: Vec<(f32, f32)> = vec![
        (0.2, 0.5),       // ONNX default
        (1.0 / 6.0, 0.5), // PyTorch hardsigmoid
        (0.16666667, 0.5),
        (0.1, 0.5),
        (0.5, 0.5),
        (1.0, 0.0),
        (1.0, 0.5),
        (2.0, 1.0),
        (0.2, 0.0),
        (0.2, 1.0),
        (0.3, 0.7),
        (3.0, 0.25),
        (1e-6, 0.5),
        (1e6, 0.5),
        (0.2, -0.5),
        (0.25, 0.5),
        (0.125, 0.5),
        (0.7, 0.3),
    ];
    for _ in 0..40 {
        params.push((rng.f32_moderate().abs(), rng.f32_moderate()));
    }
    for &(alpha, beta) in &params {
        if !(alpha.is_finite() && beta.is_finite()) || alpha <= 0.0 {
            continue;
        }
        let xl_f32 = -beta / alpha;
        let xh_f32 = (1.0 - beta) / alpha;
        let xl_true = -(beta as f64) / (alpha as f64);
        let xh_true = (1.0 - beta as f64) / (alpha as f64);
        let mut kinks = vec![xl_f32, xh_f32];
        kinks.retain(|x| x.is_finite());
        let ivs = sample_intervals(&mut rng, 3000, &kinks);
        let ps = format!(
            "alpha={:e}({:#010x}) beta={:e}({:#010x})",
            alpha,
            alpha.to_bits(),
            beta,
            beta.to_bits()
        );
        let (a64, b64) = (alpha as f64, beta as f64);
        for (l, u) in ivs {
            let r = audit_hard_sigmoid_relax(l, u, alpha, beta);
            probe(
                &mut c,
                l,
                u,
                &ps,
                r,
                &move |x| hs_f64(x, a64, b64),
                &[xl_true, xh_true, xl_f32 as f64, xh_f32 as f64],
            );
        }
    }
    c.report();
}

// ── clip ────────────────────────────────────────────────────────────────

fn clip_f64(x: f64, mn: f64, mx: f64) -> f64 {
    if x < mn {
        mn
    } else if x > mx {
        mx
    } else {
        x
    }
}

#[test]
fn audit_clip_envelope() {
    let mut rng = Rng::new(0xAA02);
    let mut c = Collector::new("clip_linear_relaxation");
    let mut params: Vec<(f32, f32)> = vec![
        (0.0, 6.0),
        (-1.0, 1.0),
        (0.0, 1.0),
        (-6.0, 6.0),
        (0.0, 0.0),
        (-0.5, 0.5),
        (1e-30, 1e30),
        (-1e30, -1e-30),
        (0.1, 0.2),
        (-3.0, 3.0),
        (0.0, f32::MAX),
        (f32::MIN, 0.0),
        (-1e-8, 1e-8),
        (1.0, 1.0000001),
        (-1e20, 1e20),
    ];
    for _ in 0..40 {
        let a = rng.f32_moderate();
        let b = rng.f32_moderate();
        params.push((a.min(b), a.max(b)));
    }
    for _ in 0..20 {
        let a = rng.f32_any();
        let b = rng.f32_any();
        params.push((a.min(b), a.max(b)));
    }
    for &(mn, mx) in &params {
        if !(mn.is_finite() && mx.is_finite()) || mn > mx {
            continue;
        }
        let ivs = sample_intervals(&mut rng, 3000, &[mn, mx]);
        let ps = format!(
            "min={:e}({:#010x}) max={:e}({:#010x})",
            mn,
            mn.to_bits(),
            mx,
            mx.to_bits()
        );
        let (mn64, mx64) = (mn as f64, mx as f64);
        for (l, u) in ivs {
            let r = audit_clip_relax(l, u, mn, mx);
            probe(
                &mut c,
                l,
                u,
                &ps,
                r,
                &move |x| clip_f64(x, mn64, mx64),
                &[mn64, mx64],
            );
        }
    }
    c.report();
}

// ── hardswish ───────────────────────────────────────────────────────────

fn hsw_f64(x: f64) -> f64 {
    let t = (x + 3.0) / 6.0;
    let t = if t < 0.0 {
        0.0
    } else if t > 1.0 {
        1.0
    } else {
        t
    };
    x * t
}

#[test]
fn audit_hardswish_envelope() {
    let mut rng = Rng::new(0xAA03);
    let mut c = Collector::new("hardswish_linear_relaxation");
    let ivs = sample_intervals(&mut rng, 400_000, &[-3.0, 3.0, -1.5, 0.0]);
    for (l, u) in ivs {
        let r = audit_hardswish_relax(l, u);
        // interior stationary points of line(x)-f(x) on the quadratic region
        let mut crit = vec![-3.0f64, 3.0, -1.5, 0.0];
        for s in [r.lower_slope as f64, r.upper_slope as f64] {
            crit.push(3.0 * s - 1.5);
        }
        probe(&mut c, l, u, "", r, &|x| hsw_f64(x), &crit);
    }
    c.report();
}

// ── softsign ────────────────────────────────────────────────────────────

fn ss_f64(x: f64) -> f64 {
    x / (1.0 + x.abs())
}

#[test]
fn audit_softsign_envelope() {
    let mut rng = Rng::new(0xAA04);
    let mut c = Collector::new("softsign_linear_relaxation");
    let ivs = sample_intervals(&mut rng, 300_000, &[0.0, 1.0, -1.0]);
    for (l, u) in ivs {
        let r = audit_softsign_relax(l, u);
        // stationary points of line(x)-f(x): f'(x) = 1/(1+|x|)^2 = slope
        let mut crit = vec![0.0f64];
        for s in [r.lower_slope as f64, r.upper_slope as f64] {
            if s > 0.0 && s <= 1.0 {
                let a = 1.0 / s.sqrt() - 1.0;
                crit.push(a);
                crit.push(-a);
            }
        }
        // geometric ladder inside [l,u]
        let (lo, hi) = (l as f64, u as f64);
        let mut t = 1e-30f64;
        // Geometric ladder of magnitudes across [1e-30, 1e31); the f64 bound
        // and the x4 stride define the probe set. Do not reshape it.
        #[allow(clippy::while_float)]
        while t < 1e31 {
            if t >= lo && t <= hi {
                crit.push(t);
            }
            if -t >= lo && -t <= hi {
                crit.push(-t);
            }
            t *= 4.0;
        }
        probe(&mut c, l, u, "", r, &|x| ss_f64(x), &crit);
    }
    c.report();
}
