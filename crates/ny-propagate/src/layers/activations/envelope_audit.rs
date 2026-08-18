// TEMPORARY envelope-validity audit harness (not for merge).
//! Checks the ENVELOPE obligation
//!   lower_slope*x + lower_intercept <= f(x) <= upper_slope*x + upper_intercept
//! for all x in [l, u], evaluated in f64 (products of f32 are exact in f64).

#![allow(clippy::all)]

use super::leaky_relu::leaky_relu_linear_relaxation;
use super::prelu::{prelu_crossing_relaxation, prelu_linear_relaxation};
use super::shrink::ShrinkLayer;
use super::thresholded_relu::{thresholded_relu_crossing, thresholded_relu_linear_relaxation};
use super::LinearRelaxation;
use ny_tensor::next_up_f32;

// ── plumbing ────────────────────────────────────────────────────────────

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
        // exponent in [-30, 30]
        let e = (self.u32() % 61) as i32 - 30;
        let m = 1.0 + (self.u32() % (1 << 23)) as f32 / (1u32 << 23) as f32;
        let s = if self.u32() & 1 == 0 { 1.0 } else { -1.0 };
        s * m * 2.0f32.powi(e)
    }
    fn unit(&mut self) -> f64 {
        (self.u32() % 1_000_003) as f64 / 1_000_003.0
    }
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
    let a = (v.abs() as f32).abs();
    if a == 0.0 || !a.is_finite() {
        return f32::from_bits(1) as f64;
    }
    (next_up_f32(a) - a) as f64
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
        if self.worst.len() > 400 {
            self.worst
                .sort_by(|a, b| b.ulps.partial_cmp(&a.ulps).unwrap());
            self.worst.truncate(40);
        }
    }
    fn report(&mut self) {
        self.worst
            .sort_by(|a, b| b.ulps.partial_cmp(&a.ulps).unwrap());
        println!(
            "\n===== {}: {} intervals checked, {} point-violations =====",
            self.tag, self.n_checked, self.n_viol
        );
        // Deduplicate by (l,u,params,side)
        let mut seen: Vec<(f32, f32, String, &str)> = Vec::new();
        let mut shown = 0;
        for v in &self.worst {
            let key = (v.l, v.u, v.params.clone(), v.side);
            if seen.iter().any(|k| *k == key) {
                continue;
            }
            seen.push(key);
            println!(
                "  [{}] {} l={:e} ({:#010x}) u={:e} ({:#010x}) {}\n      coeffs ls={:e} li={:e} us={:e} ui={:e}\n      x={:.17e} f(x)={:.17e} line={:.17e} gap={:.6e} ({:.3} ULP of f(x))",
                self.tag, v.side, v.l, v.l.to_bits(), v.u, v.u.to_bits(), v.params,
                v.coeffs[0], v.coeffs[1], v.coeffs[2], v.coeffs[3],
                v.x, v.fx, v.line, v.gap, v.ulps
            );
            shown += 1;
            if shown >= 12 {
                break;
            }
        }
        if self.n_viol == 0 {
            println!("  CLEAN");
        }
    }
}

/// Core obligation check. `f` must be an f64-exact evaluation of the activation.
fn probe(
    c: &mut Collector,
    l: f32,
    u: f32,
    params: &str,
    r: LinearRelaxation,
    f: &dyn Fn(f64) -> f64,
    kinks: &[f64],
) {
    c.n_checked += 1;
    let lo_t = l as f64;
    let hi_t = u as f64;
    if !(lo_t <= hi_t) {
        return;
    }
    let lo = if l.is_finite() { lo_t } else { -1e30 };
    let hi = if u.is_finite() { hi_t } else { 1e30 };

    let mut xs: Vec<f64> = Vec::with_capacity(512);
    xs.push(lo);
    xs.push(hi);
    for k in 0..=200 {
        let t = k as f64 / 200.0;
        xs.push(lo + t * (hi - lo));
    }
    // geometric probes into any unbounded direction
    if !l.is_finite() {
        for e in -20..=30 {
            xs.push(-(10f64.powi(e)));
        }
    }
    if !u.is_finite() {
        for e in -20..=30 {
            xs.push(10f64.powi(e));
        }
    }
    for &k in kinks {
        if !k.is_finite() {
            continue;
        }
        xs.push(k);
        xs.push(next_up_f64(k));
        xs.push(next_down_f64(k));
        for m in [1u32, 2, 4, 8, 64, 1024] {
            let mut a = k;
            let mut b = k;
            for _ in 0..m {
                a = next_up_f64(a);
                b = next_down_f64(b);
            }
            xs.push(a);
            xs.push(b);
        }
        xs.push(k + 1e-6 * k.abs().max(1e-30));
        xs.push(k - 1e-6 * k.abs().max(1e-30));
    }
    // endpoint neighbourhoods
    for &e in [lo_t, hi_t].iter() {
        if e.is_finite() {
            xs.push(next_up_f64(e));
            xs.push(next_down_f64(e));
        }
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
            gap: f64::NAN,
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
        if !x.is_finite() || x < lo_t || x > hi_t {
            continue;
        }
        let fx = f(x);
        if !fx.is_finite() {
            continue;
        }
        // lower
        if li != f64::NEG_INFINITY {
            let line = ls * x + li;
            if line.is_finite() || line == f64::INFINITY {
                let gap = line - fx;
                let scale = (ls * x)
                    .abs()
                    .max(li.abs())
                    .max(fx.abs())
                    .max(f64::MIN_POSITIVE);
                if gap > 1e-11 * scale {
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
        }
        // upper
        if ui != f64::INFINITY {
            let line = us * x + ui;
            let gap = fx - line;
            let scale = (us * x)
                .abs()
                .max(ui.abs())
                .max(fx.abs())
                .max(f64::MIN_POSITIVE);
            if gap > 1e-11 * scale {
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

// ── interval sampling strategies ────────────────────────────────────────

fn sample_intervals(rng: &mut Rng, n: usize, kinks: &[f32]) -> Vec<(f32, f32)> {
    let mut v = Vec::with_capacity(n + 200);
    // degenerate / structured cases
    let specials: &[f32] = &[
        0.0,
        -0.0,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::from_bits(1),
        -f32::from_bits(1),
        1.0,
        -1.0,
        1e-8,
        -1e-8,
        1e-30,
        -1e-30,
        1e30,
        -1e30,
        f32::MAX,
        f32::MIN,
        3.5,
        -3.5,
        0.1,
        -0.1,
    ];
    for &a in specials {
        v.push((a, a)); // l == u
        for &b in specials {
            if a <= b {
                v.push((a, b));
            }
        }
    }
    for &k in kinks {
        let kf = k as f64;
        for scale in [1e-9f64, 1e-7, 1e-4, 1e-2, 1.0, 10.0] {
            let d = (kf.abs().max(1.0)) * scale;
            v.push(((kf - d) as f32, (kf + d) as f32));
            v.push((k, (kf + d) as f32));
            v.push(((kf - d) as f32, k));
        }
        v.push((next_down_f32_local(k), next_up_f32(k)));
        v.push((k, next_up_f32(k)));
        v.push((next_down_f32_local(k), k));
        v.push((k, k));
    }
    for _ in 0..n {
        match rng.u32() % 6 {
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
                // crossing zero
                let a = -rng.f32_moderate().abs();
                let b = rng.f32_moderate().abs();
                v.push((a, b));
            }
            3 => {
                // narrow relative width
                let a = rng.f32_moderate();
                let w = a.abs() * (rng.unit() as f32) * 1e-3;
                v.push((a, a + w));
            }
            4 => {
                // near a kink
                let k = if kinks.is_empty() {
                    0.0
                } else {
                    kinks[(rng.u32() as usize) % kinks.len()]
                };
                let d = (k.abs().max(1.0)) * (rng.unit() as f32) * (rng.unit() as f32);
                v.push((k - d, k + d));
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

fn next_down_f32_local(x: f32) -> f32 {
    -next_up_f32(-x)
}

// ── f64-exact activation references ─────────────────────────────────────

fn leaky_f64(x: f64, alpha: f64) -> f64 {
    if x >= 0.0 {
        x
    } else {
        alpha * x
    }
}
fn trelu_f64(x: f64, alpha: f64) -> f64 {
    if x > alpha {
        x
    } else {
        0.0
    }
}
fn shrink_f64(x: f64, bias: f64, lambd: f64) -> f64 {
    if x > lambd {
        x - bias
    } else if x < -lambd {
        x + bias
    } else {
        0.0
    }
}

// ── audits ──────────────────────────────────────────────────────────────

fn alphas(rng: &mut Rng, n: usize) -> Vec<f32> {
    let mut a = vec![
        0.01f32,
        0.1,
        0.2,
        0.25,
        0.5,
        0.9,
        1.0,
        1.5,
        2.0,
        3.0,
        10.0,
        100.0,
        0.0,
        -0.0,
        -0.01,
        -0.5,
        -1.0,
        -2.0,
        1e-8,
        1e8,
        -1e8,
        f32::MIN_POSITIVE,
        0.3,
        0.7,
    ];
    for _ in 0..n {
        if rng.u32() % 2 == 0 {
            a.push(rng.f32_moderate());
        } else {
            a.push(rng.f32_any());
        }
    }
    a
}

#[test]
fn audit_leaky_relu_envelope() {
    let mut rng = Rng::new(0xC0FFEE01);
    let mut c = Collector::new("leaky_relu_linear_relaxation");
    let als = alphas(&mut rng, 60);
    for &alpha in &als {
        let ivs = sample_intervals(&mut rng, 2500, &[0.0]);
        for (l, u) in ivs {
            let r = leaky_relu_linear_relaxation(l, u, alpha);
            let a64 = alpha as f64;
            probe(
                &mut c,
                l,
                u,
                &format!("alpha={:e}({:#010x})", alpha, alpha.to_bits()),
                r,
                &move |x| leaky_f64(x, a64),
                &[0.0],
            );
        }
        // infinite-domain arms (LeakyReLU uses a NaN-only domain guard, so these
        // are production-reachable)
        for fe in [0.5f32, 1.0, 3.0, 1e-3, 1e10, -0.5, -1.0, -3.0, -1e10, -1e-3] {
            for (l, u) in [
                (f32::NEG_INFINITY, fe),
                (fe, f32::INFINITY),
                (f32::NEG_INFINITY, f32::INFINITY),
            ] {
                let r = leaky_relu_linear_relaxation(l, u, alpha);
                let a64 = alpha as f64;
                probe(
                    &mut c,
                    l,
                    u,
                    &format!("INF-domain alpha={:e}", alpha),
                    r,
                    &move |x| leaky_f64(x, a64),
                    &[0.0],
                );
            }
        }
    }
    c.report();
}

#[test]
fn audit_prelu_envelope() {
    let mut rng = Rng::new(0xC0FFEE02);
    let mut c = Collector::new("prelu_linear_relaxation");
    let mut cx = Collector::new("prelu_crossing_relaxation");
    let als = alphas(&mut rng, 60);
    for &alpha in &als {
        let ivs = sample_intervals(&mut rng, 2500, &[0.0]);
        for (l, u) in ivs {
            let a64 = alpha as f64;
            let r = prelu_linear_relaxation(l, u, alpha);
            probe(
                &mut c,
                l,
                u,
                &format!("alpha={:e}({:#010x})", alpha, alpha.to_bits()),
                r,
                &move |x| leaky_f64(x, a64),
                &[0.0],
            );
            if l < 0.0 && u > 0.0 && l.is_finite() && u.is_finite() {
                let rc = prelu_crossing_relaxation(l, u, alpha);
                probe(
                    &mut cx,
                    l,
                    u,
                    &format!("alpha={:e}({:#010x})", alpha, alpha.to_bits()),
                    rc,
                    &move |x| leaky_f64(x, a64),
                    &[0.0],
                );
            }
        }
    }
    c.report();
    cx.report();
}

#[test]
fn audit_thresholded_relu_envelope() {
    let mut rng = Rng::new(0xC0FFEE03);
    let mut c = Collector::new("thresholded_relu_linear_relaxation");
    let mut cx = Collector::new("thresholded_relu_crossing");
    let als = alphas(&mut rng, 60);
    for &alpha in &als {
        if !alpha.is_finite() {
            continue;
        }
        let ivs = sample_intervals(&mut rng, 2500, &[0.0, alpha]);
        for (l, u) in ivs {
            let a64 = alpha as f64;
            let r = thresholded_relu_linear_relaxation(l, u, alpha);
            probe(
                &mut c,
                l,
                u,
                &format!("alpha={:e}({:#010x})", alpha, alpha.to_bits()),
                r,
                &move |x| trelu_f64(x, a64),
                &[0.0, alpha as f64],
            );
            if l.is_finite() && u.is_finite() && l <= alpha && alpha < u && (u - l).abs() >= 1e-8 {
                let rc = thresholded_relu_crossing(l, u, alpha);
                probe(
                    &mut cx,
                    l,
                    u,
                    &format!("alpha={:e}({:#010x})", alpha, alpha.to_bits()),
                    rc,
                    &move |x| trelu_f64(x, a64),
                    &[0.0, alpha as f64],
                );
            }
        }
        // u = +inf arm
        for fe in [-1e10f32, -3.0, -1.0, -1e-3, 0.0, 1e-3, 1.0, 3.0, 1e10] {
            let (l, u) = (fe, f32::INFINITY);
            if !(l <= alpha) {
                continue;
            }
            let a64 = alpha as f64;
            let r = thresholded_relu_linear_relaxation(l, u, alpha);
            probe(
                &mut c,
                l,
                u,
                &format!("INF-u alpha={:e}", alpha),
                r,
                &move |x| trelu_f64(x, a64),
                &[0.0, alpha as f64],
            );
        }
    }
    c.report();
    cx.report();
}

#[test]
fn audit_shrink_envelope() {
    let mut rng = Rng::new(0xC0FFEE04);
    let mut c = Collector::new("shrink_relaxation");
    let mut params: Vec<(f32, f32)> = vec![
        (0.0, 0.5),
        (0.0, 0.0),
        (0.5, 0.5),
        (1.0, 0.5),
        (0.5, 1.0),
        (0.0, 1.0),
        (2.0, 0.5),
        (0.25, 0.75),
        (1e-8, 1e-8),
        (1e8, 1e8),
        (-1.0, 0.5),
        (-0.5, 2.0),
        (3.0, 1.0),
        (0.1, 0.3),
    ];
    for _ in 0..40 {
        let b = rng.f32_moderate();
        let lam = rng.f32_moderate().abs();
        params.push((b, lam));
    }
    for _ in 0..20 {
        let b = rng.f32_any();
        let lam = rng.f32_any().abs();
        params.push((b, lam));
    }
    for &(bias, lambd) in &params {
        if !bias.is_finite() || !lambd.is_finite() || lambd < 0.0 {
            continue;
        }
        let layer = match ShrinkLayer::try_new(bias, lambd) {
            Ok(x) => x,
            Err(_) => continue,
        };
        let ivs = sample_intervals(&mut rng, 4000, &[0.0, lambd, -lambd]);
        for (l, u) in ivs {
            let r = layer.relaxation(l, u);
            let b64 = bias as f64;
            let lam64 = lambd as f64;
            probe(
                &mut c,
                l,
                u,
                &format!(
                    "bias={:e}({:#010x}) lambd={:e}({:#010x})",
                    bias,
                    bias.to_bits(),
                    lambd,
                    lambd.to_bits()
                ),
                r,
                &move |x| shrink_f64(x, b64, lam64),
                &[0.0, lambd as f64, -(lambd as f64)],
            );
        }
    }
    c.report();
}
