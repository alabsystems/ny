//! Soundness audits for Snake, comparison, and piecewise-constant envelopes.
//!
//! The ordinary tests use a bounded, deterministic corpus so every test run
//! checks these obligations.  The exhaustive profile preserves the much larger
//! randomized and breakpoint sweeps without hiding them behind `#[ignore]`.
//! Run it explicitly with:
//!
//! ```text
//! cargo run --release -p ny-propagate --example audit_envelopes_snakecompare
//! ```
//
// Obligation under test, for a `LinearRelaxation` produced from [l, u]:
//   FOR ALL real x in [l, u]:
//     lower_slope*x + lower_intercept <= f(x) <= upper_slope*x + upper_intercept
//
// Everything is evaluated in f64 (or better). `f64::from(coef) * x` where x is
// itself an f32 is EXACT; where x is an interior f64 sample the product rounds,
// so reported witnesses are re-verified in mpmath out-of-band.
#![allow(clippy::excessive_precision)]

use crate::layers::activations::snake::snake_linear_relaxation;
use crate::layers::activations::LinearRelaxation;
use crate::layers::misc::compare::{compare_crown_relaxation, CompareOp};
use crate::layers::misc::{CeilLayer, FloorLayer, RoundLayer, TruncLayer};

const TWO_PI: f64 = std::f64::consts::PI * 2.0;
const MAX_RECORDED_VIOLATIONS: usize = 1_024;

const EXHAUSTIVE_AUDIT_ENV: &str = "NY_ENVELOPE_AUDIT_SNAKECOMPARE";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AuditProfile {
    Bounded,
    Exhaustive,
}

impl AuditProfile {
    fn from_environment() -> Self {
        match std::env::var(EXHAUSTIVE_AUDIT_ENV) {
            Err(std::env::VarError::NotPresent) => Self::Bounded,
            Ok(value) if value == "exhaustive" => Self::Exhaustive,
            Ok(value) => {
                panic!("{EXHAUSTIVE_AUDIT_ENV} must be `exhaustive` when set, got {value:?}")
            }
            Err(error) => panic!("cannot read {EXHAUSTIVE_AUDIT_ENV}: {error}"),
        }
    }

    const fn is_exhaustive(self) -> bool {
        matches!(self, Self::Exhaustive)
    }
}

#[derive(Clone, Debug)]
struct Violation {
    l: f32,
    u: f32,
    param: f64,
    x: f64,
    side: &'static str,
    line: f64,
    fx: f64,
    amount: f64,
    r: LinearRelaxation,
}

fn ulp_of(v: f64) -> f64 {
    let a = v.abs() as f32;
    if a == 0.0 {
        return f64::from(f32::MIN_POSITIVE) * 2f64.powi(-23);
    }
    let n = ny_tensor::next_up_f32(a);
    f64::from(n) - f64::from(a)
}

/// Evaluate the two lines at x in f64 and record any obligation failure.
fn check_point(
    r: &LinearRelaxation,
    l: f32,
    u: f32,
    param: f64,
    x: f64,
    fx: f64,
    slack: f64,
    out: &mut Vec<Violation>,
) {
    let lo = f64::from(r.lower_slope) * x + f64::from(r.lower_intercept);
    let hi = f64::from(r.upper_slope) * x + f64::from(r.upper_intercept);
    if lo.is_finite() && lo - fx > slack && out.len() < MAX_RECORDED_VIOLATIONS {
        out.push(Violation {
            l,
            u,
            param,
            x,
            side: "lower",
            line: lo,
            fx,
            amount: lo - fx,
            r: *r,
        });
    }
    if hi.is_finite() && fx - hi > slack && out.len() < MAX_RECORDED_VIOLATIONS {
        out.push(Violation {
            l,
            u,
            param,
            x,
            side: "upper",
            line: hi,
            fx,
            amount: fx - hi,
            r: *r,
        });
    }
}

// ---------------------------------------------------------------------------
// Snake
// ---------------------------------------------------------------------------

/// Oracle: snake(x) = x + sin^2(a x)/a, computed in f64.
/// For x an exact f32 and a an exact f32, `a*x` is EXACT in f64 (<=48 sig bits).
fn snake_true(x: f64, a: f64) -> f64 {
    let t = a * x;
    let s = t.sin();
    x + s * s / a
}

/// Candidate x values for the snake obligation on [l, u] with slope `slope`.
///
/// h(x) = line(x) - f(x) has h'(x) = slope - 1 - sin(2 a x); its interior
/// critical points are x_k = (theta + 2 pi k)/(2a) for theta in
/// {asin(m), pi - asin(m)}, m = slope - 1. On a fixed theta branch,
/// sin^2(a x_k) is CONSTANT, so h(x_k) is affine in x_k and the extremes are at
/// the smallest / largest admissible k. Those, plus the endpoints, are the
/// candidate set. (Independent of the code under test, which enumerates ALL k.)
fn snake_candidates(
    l: f64,
    u: f64,
    a: f64,
    slopes: &[f64],
    grid_samples: usize,
    edge_samples: usize,
) -> Vec<f64> {
    let mut xs = vec![l, u, 0.5 * (l + u)];
    let push_branch = |base: f64, xs: &mut Vec<f64>| {
        let kl = ((2.0 * a * l - base) / TWO_PI).ceil();
        let ku = ((2.0 * a * u - base) / TWO_PI).floor();
        if !kl.is_finite() || !ku.is_finite() {
            return;
        }
        for d in 0..3i64 {
            for k in [kl + d as f64, ku - d as f64] {
                let x = (base + TWO_PI * k) / (2.0 * a);
                if x >= l && x <= u {
                    xs.push(x);
                }
            }
        }
    };
    for &slope in slopes {
        let m = slope - 1.0;
        if m.abs() <= 1.0 {
            let b1 = m.asin();
            push_branch(b1, &mut xs);
            push_branch(std::f64::consts::PI - b1, &mut xs);
        }
    }
    // sin(a x) = 0 (oscillation minimum) and sin^2(a x) = 1 (maximum).
    for base in [0.0f64, std::f64::consts::FRAC_PI_2] {
        let kl = ((a * l - base) / std::f64::consts::PI).ceil();
        let ku = ((a * u - base) / std::f64::consts::PI).floor();
        if kl.is_finite() && ku.is_finite() {
            for d in 0..3i64 {
                for k in [kl + d as f64, ku - d as f64] {
                    let x = (base + std::f64::consts::PI * k) / a;
                    if x >= l && x <= u {
                        xs.push(x);
                    }
                }
            }
        }
    }
    // Dense grid over the whole interval and over a few periods at each end.
    for i in 0..=grid_samples {
        xs.push(l + (u - l) * (i as f64) / (grid_samples as f64));
    }
    let win = (4.0 * std::f64::consts::PI / a).min(u - l);
    if win.is_finite() && win > 0.0 {
        for i in 0..=edge_samples {
            let t = (i as f64) / (edge_samples as f64);
            xs.push(l + win * t);
            xs.push(u - win * t);
        }
    }
    xs.retain(|x| x.is_finite() && *x >= l && *x <= u);
    xs
}

fn audit_snake_one(
    l: f32,
    u: f32,
    a: f32,
    grid_samples: usize,
    edge_samples: usize,
    out: &mut Vec<Violation>,
) {
    let r = snake_linear_relaxation(l, u, a);
    if !r.lower_slope.is_finite()
        || !r.upper_slope.is_finite()
        || (r.lower_intercept == f32::NEG_INFINITY && r.upper_intercept == f32::INFINITY)
    {
        return; // degenerate / fallback relaxation, nothing to enclose
    }
    let (l64, u64, a64) = (f64::from(l), f64::from(u), f64::from(a));
    if !l64.is_finite() || !u64.is_finite() {
        return;
    }
    let slopes = [f64::from(r.lower_slope), f64::from(r.upper_slope)];
    let xs = snake_candidates(l64, u64, a64, &slopes, grid_samples, edge_samples);
    // Slack: f64 evaluation error of the oracle plus the line evaluation.
    let scale = l64.abs().max(u64.abs()).max(1.0 / a64).max(1.0);
    let slack = 64.0 * f64::EPSILON * scale;
    let mut best: Option<Violation> = None;
    let mut local = Vec::new();
    for x in xs {
        let fx = snake_true(x, a64);
        local.clear();
        check_point(&r, l, u, a64, x, fx, slack, &mut local);
        // `local` is cleared and refilled on every iteration to reuse its
        // allocation; `into_iter()` would move it out on the first pass.
        #[allow(clippy::iter_with_drain)]
        for v in local.drain(..) {
            if best.as_ref().map(|b| v.amount > b.amount).unwrap_or(true) {
                best = Some(v);
            }
        }
    }
    if let Some(mut b) = best {
        // Local refinement around the best witness.
        let mut span = (std::f64::consts::PI / a64).min(u64 - l64);
        if !span.is_finite() || span <= 0.0 {
            span = (u64 - l64).max(0.0);
        }
        for _ in 0..6 {
            let mut improved = b.clone();
            for i in -64i32..=64 {
                let x = (b.x + span * (i as f64) / 64.0).clamp(l64, u64);
                let fx = snake_true(x, a64);
                local.clear();
                check_point(&r, l, u, a64, x, fx, slack, &mut local);
                // Same reuse as above: `local` outlives this loop.
                #[allow(clippy::iter_with_drain)]
                for v in local.drain(..) {
                    if v.amount > improved.amount {
                        improved = v;
                    }
                }
            }
            b = improved;
            span /= 8.0;
        }
        if out.len() < MAX_RECORDED_VIOLATIONS {
            out.push(b);
        }
    }
}

fn f32_grid() -> Vec<f32> {
    let mut v = vec![
        0.0f32,
        -0.0,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        1e-45,
        -1e-45,
        1e-30,
        1e-20,
        1e-10,
        1e-8,
        1e-6,
        1e-3,
        0.084,
        0.5,
        1.0,
        1.5,
        std::f32::consts::PI,
        3.7,
        10.0,
        100.0,
        1e4,
        1e6,
        1e12,
        1e20,
        1e30,
        f32::MAX,
    ];
    let neg: Vec<f32> = v.iter().map(|x| -x).collect();
    v.extend(neg);
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v.dedup();
    v
}

/// A compact corpus spanning signed zero, subnormal/normal boundaries,
/// discontinuities, ordinary magnitudes, and the finite f32 extremes.
fn bounded_f32_grid() -> Vec<f32> {
    vec![
        -f32::MAX,
        -1e12,
        -std::f32::consts::PI,
        -1.0,
        -0.5,
        -1e-6,
        -f32::MIN_POSITIVE,
        -f32::from_bits(1),
        -0.0,
        0.0,
        f32::from_bits(1),
        f32::MIN_POSITIVE,
        1e-6,
        0.5,
        1.0,
        std::f32::consts::PI,
        1e12,
        f32::MAX,
    ]
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn f32_any(&mut self) -> f32 {
        loop {
            let b = (self.next() & 0xFFFF_FFFF) as u32;
            let f = f32::from_bits(b);
            if f.is_finite() {
                return f;
            }
        }
    }
    fn f32_scaled(&mut self, max_exp: i32) -> f32 {
        let m = (self.next() % (1 << 23)) as u32;
        let e = (self.next() % 60) as i32 - 30;
        let e = e.clamp(-max_exp, max_exp);
        let sign = if self.next() & 1 == 0 { 1.0 } else { -1.0 };
        sign * f32::from_bits((127u32 << 23) | m) * 2f32.powi(e)
    }
}

fn report(name: &str, mut vs: Vec<Violation>) -> usize {
    vs.sort_by(|a, b| b.amount.partial_cmp(&a.amount).unwrap());
    let n = vs.len();
    println!(
        "=== {name}: {n} recorded violation(s) (recording capped at {MAX_RECORDED_VIOLATIONS}) ==="
    );
    for v in vs.iter().take(25) {
        let ulps = v.amount / ulp_of(v.fx);
        println!(
            "  l={:e} (0x{:08x}) u={:e} (0x{:08x}) param={:e}\n     side={} x={:.17e} line={:.17e} f(x)={:.17e} abs={:.6e} ulps={:.3e}\n     coeffs: ls={:e} li={:e} us={:e} ui={:e}",
            v.l,
            v.l.to_bits(),
            v.u,
            v.u.to_bits(),
            v.param,
            v.side,
            v.x,
            v.line,
            v.fx,
            v.amount,
            ulps,
            v.r.lower_slope,
            v.r.lower_intercept,
            v.r.upper_slope,
            v.r.upper_intercept
        );
        println!(
            "WITNESS {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {:016x} {}",
            v.l.to_bits(),
            v.u.to_bits(),
            (v.param as f32).to_bits(),
            v.r.lower_slope.to_bits(),
            v.r.lower_intercept.to_bits(),
            v.r.upper_slope.to_bits(),
            v.r.upper_intercept.to_bits(),
            v.x.to_bits(),
            v.side
        );
    }
    n
}

#[test]
fn audit_snake() {
    let profile = AuditProfile::from_environment();
    let exhaustive_alphas = [
        1e-9,
        5e-9,
        9.9e-9,
        1e-8,
        1.0000001e-8,
        2e-8,
        1e-7,
        1e-5,
        1e-3,
        0.01,
        0.1,
        0.5,
        1.0,
        2.0,
        3.7,
        10.0,
        100.0,
        1000.0,
        1e5,
        1e8,
        1e12,
        1e20,
        1e30,
        3.4e38,
    ];
    let bounded_alphas = [1e-9, 9.9e-9, 1e-8, 1.0000001e-8, 0.1, 1.0, 10.0, 1e8];
    let alphas: &[f32] = if profile.is_exhaustive() {
        &exhaustive_alphas
    } else {
        &bounded_alphas
    };
    let grid = if profile.is_exhaustive() {
        f32_grid()
    } else {
        bounded_f32_grid()
    };
    let (grid_samples, edge_samples, scaled_random_cases, any_random_cases) =
        if profile.is_exhaustive() {
            (512, 256, 30_000, 20_000)
        } else {
            (96, 48, 128, 128)
        };
    let mut out = Vec::new();
    let mut cases = 0usize;

    // (A) structured grid: endpoints from the exponent grid, plus widths.
    for &a in alphas {
        for &l in &grid {
            if !l.is_finite() {
                continue;
            }
            let mut us = vec![
                l,
                ny_tensor::next_up_f32(l),
                ny_tensor::next_up_f32(ny_tensor::next_up_f32(l)),
                l + 1e-9,
                l + 1e-8,
                l + 1e-7,
                l + 1e-3,
                l + 1.0,
                l + 100.0,
                l * 2.0,
                -l,
                l.abs() * 1e6,
            ];
            us.retain(|x| x.is_finite() && *x >= l);
            for u in us {
                cases += 1;
                audit_snake_one(l, u, a, grid_samples, edge_samples, &mut out);
            }
        }
    }

    // (B) random sweep across the whole f32 range.
    let mut rng = Rng(0x1234_5678_9abc_def0);
    for _ in 0..scaled_random_cases {
        let a = alphas[(rng.next() as usize) % alphas.len()];
        let (mut l, mut u) = (rng.f32_scaled(30), rng.f32_scaled(30));
        if l > u {
            std::mem::swap(&mut l, &mut u);
        }
        cases += 1;
        audit_snake_one(l, u, a, grid_samples, edge_samples, &mut out);
    }
    for _ in 0..any_random_cases {
        let a = rng.f32_any().abs().max(1e-12);
        let (mut l, mut u) = (rng.f32_any(), rng.f32_any());
        if l > u {
            std::mem::swap(&mut l, &mut u);
        }
        if !(l.is_finite() && u.is_finite() && a.is_finite()) {
            continue;
        }
        cases += 1;
        audit_snake_one(l, u, a, grid_samples, edge_samples, &mut out);
    }

    let minimum_cases = alphas.len() * grid.len() + scaled_random_cases + any_random_cases;
    assert!(
        cases >= minimum_cases,
        "snake audit skipped a required corpus segment"
    );
    println!("snake {profile:?} cases probed: {cases}");
    let n = report("snake_linear_relaxation", out);
    assert_eq!(n, 0, "snake envelope violations found");
}

#[test]
fn audit_snake_infinite() {
    let profile = AuditProfile::from_environment();
    let samples = if profile.is_exhaustive() { 20_000 } else { 512 };
    // snake_infinite_relaxation is reached only through snake_linear_relaxation
    // with a non-finite endpoint.
    let mut out = Vec::new();
    let mut cases = 0usize;
    for &a in &[1e-7f32, 0.1, 1.0, 10.0, 1e6] {
        for &b in &[-1e6f32, -1.0, 0.0, 1.0, 1e6] {
            for (l, u) in [(f32::NEG_INFINITY, b), (b, f32::INFINITY)] {
                cases += 1;
                let r = snake_linear_relaxation(l, u, a);
                let a64 = f64::from(a);
                let lo = if l.is_finite() { f64::from(l) } else { -1e9 };
                let hi = if u.is_finite() { f64::from(u) } else { 1e9 };
                for i in 0..=samples {
                    let x = lo + (hi - lo) * (i as f64) / (samples as f64);
                    let fx = snake_true(x, a64);
                    check_point(&r, l, u, a64, x, fx, 1e-9, &mut out);
                }
            }
        }
    }
    assert_eq!(cases, 50, "infinite-arm audit corpus changed unexpectedly");
    let n = report("snake_infinite_relaxation", out);
    assert_eq!(n, 0, "snake infinite-arm envelope violations found");
}

// ---------------------------------------------------------------------------
// Compare
// ---------------------------------------------------------------------------

fn compare_true(x: f64, t: f64, op: CompareOp) -> f64 {
    let b = match op {
        CompareOp::Gt => x > t,
        CompareOp::Ge => x >= t,
        CompareOp::Lt => x < t,
        CompareOp::Le => x <= t,
        CompareOp::Eq => x == t,
        CompareOp::Ne => x != t,
    };
    if b {
        1.0
    } else {
        0.0
    }
}

#[test]
fn audit_compare() {
    let profile = AuditProfile::from_environment();
    let ops = [
        CompareOp::Gt,
        CompareOp::Ge,
        CompareOp::Lt,
        CompareOp::Le,
        CompareOp::Eq,
        CompareOp::Ne,
    ];
    let grid = if profile.is_exhaustive() {
        f32_grid()
    } else {
        bounded_f32_grid()
    };
    let samples = if profile.is_exhaustive() { 1_000 } else { 64 };
    let mut out = Vec::new();
    let mut cases = 0usize;
    for &op in &ops {
        for &t in &grid {
            for &l in &grid {
                let us = [
                    l,
                    ny_tensor::next_up_f32(l),
                    l + 1.0,
                    l.abs() * 2.0,
                    t,
                    ny_tensor::next_up_f32(t),
                    ny_tensor::next_down_f32(t),
                    1e30,
                ];
                for u in us {
                    if !(u.is_finite() && u >= l) {
                        continue;
                    }
                    cases += 1;
                    let r = compare_crown_relaxation(l, u, t, op);
                    let (l64, u64, t64) = (f64::from(l), f64::from(u), f64::from(t));
                    let mut xs = vec![
                        l64,
                        u64,
                        0.5 * (l64 + u64),
                        t64,
                        t64.next_up(),
                        t64.next_down(),
                        t64 + 1e-300,
                        t64 - 1e-300,
                    ];
                    for i in 0..=samples {
                        xs.push(l64 + (u64 - l64) * (i as f64) / (samples as f64));
                    }
                    for x in xs {
                        if !(x.is_finite() && x >= l64 && x <= u64) {
                            continue;
                        }
                        let fx = compare_true(x, t64, op);
                        check_point(&r, l, u, t64, x, fx, 0.0, &mut out);
                    }
                }
            }
        }
    }
    assert!(
        cases >= ops.len() * grid.len() * grid.len(),
        "comparison audit skipped a required corpus segment"
    );
    println!("compare {profile:?} cases probed: {cases}");
    let n = report("compare_crown_relaxation", out);
    assert_eq!(n, 0, "compare envelope violations found");
}

// ---------------------------------------------------------------------------
// Floor / Ceil / Round / Trunc
// ---------------------------------------------------------------------------

fn pw_candidates(l: f64, u: f64, grid_samples: usize, breakpoint_limit: f64) -> Vec<f64> {
    let mut xs = vec![l, u, 0.5 * (l + u)];
    let k0 = l.ceil();
    let k1 = u.floor();
    // `k += 1.0` is only a STEP below 2^53; above it the f64 ulp exceeds 1 and
    // the increment is a silent no-op. `l = u = f32::MAX` reaches here with
    // `k1 - k0 == 0`, passing the span guard, and then spins forever pushing
    // into `xs` — the audit ran 120 s to 12.9 GB RSS and was OOM-killed, which
    // reads as a hung or leaking test rather than the non-terminating loop it
    // is. Nothing is lost by skipping: every f64 at that magnitude is already
    // an integer, so floor/ceil/trunc/round have no interior breakpoint there
    // (for f32-sourced values that is true from 2^24 up).
    const MAX_STEPPABLE_INTEGER: f64 = 9_007_199_254_740_992.0; // 2^53
    if k0.is_finite()
        && k1.is_finite()
        && k1 >= k0
        && (k1 - k0) < breakpoint_limit
        && k0.abs().max(k1.abs()) < MAX_STEPPABLE_INTEGER
    {
        let mut k = k0;
        // Walks the integer breakpoints in [ceil(l), floor(u)]. k0/k1 are
        // integral f64, the span is guarded by the profile's finite breakpoint
        // limit, and the magnitude guard above keeps `k += 1.0` an exact step,
        // so the float comparison terminates; do not reshape it.
        #[allow(clippy::while_float)]
        while k <= k1 {
            for d in [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0] {
                let x = k + d * (1.0 - f64::EPSILON);
                if x >= l && x <= u {
                    xs.push(x);
                }
            }
            xs.push(k.next_down());
            xs.push(k.next_up());
            xs.push(k - 0.5);
            xs.push(k + 0.5);
            k += 1.0;
        }
    }
    // half-integers explicitly (Round's discontinuity set)
    for h in [
        l.ceil() - 0.5,
        u.floor() + 0.5,
        (0.5 * (l + u)).round() + 0.5,
    ] {
        xs.push(h);
        xs.push(h.next_down());
        xs.push(h.next_up());
    }
    for i in 0..=grid_samples {
        xs.push(l + (u - l) * (i as f64) / (grid_samples as f64));
    }
    xs.retain(|x| x.is_finite() && *x >= l && *x <= u);
    xs
}

#[test]
fn pw_candidates_stop_at_f64_integer_fixed_point() {
    const FIRST_INTEGER_WITHOUT_UNIT_SUCCESSOR: f64 = 9_007_199_254_740_992.0;

    let k = FIRST_INTEGER_WITHOUT_UNIT_SUCCESSOR;
    assert_eq!(
        k + 1.0,
        k,
        "test value must be an f64 unit-step fixed point"
    );

    let xs = pw_candidates(k, k, 1, 128.0);
    assert!(
        !xs.is_empty(),
        "singleton interval must retain its endpoint"
    );
    assert!(
        xs.into_iter().all(|x| x == k),
        "singleton interval produced an out-of-range candidate"
    );
}

/// ONNX Round: round-half-to-EVEN (spec: "rounding to nearest even").
fn round_half_even(x: f64) -> f64 {
    let r = x.round(); // half away from zero
    if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        r - x.signum()
    } else {
        r
    }
}

fn audit_pw<F, G>(name: &str, relax: F, f: G, profile: AuditProfile) -> (usize, usize)
where
    F: Fn(f32, f32) -> LinearRelaxation,
    G: Fn(f64) -> f64,
{
    let mut out = Vec::new();
    let mut rng = Rng(0xdead_beef_cafe_1234);
    let mut cases = 0usize;
    let mut endpoints = if profile.is_exhaustive() {
        f32_grid()
    } else {
        bounded_f32_grid()
    };
    let integer_radius = if profile.is_exhaustive() { 20 } else { 3 };
    let (grid_samples, breakpoint_limit, random_cases) = if profile.is_exhaustive() {
        (1_024, 4_096.0, 20_000)
    } else {
        (64, 128.0, 256)
    };
    for k in -integer_radius..=integer_radius {
        for frac in [0.0f32, 0.25, 0.5, 0.75, -0.5, -0.25] {
            endpoints.push(k as f32 + frac);
            endpoints.push(ny_tensor::next_up_f32(k as f32 + frac));
            endpoints.push(ny_tensor::next_down_f32(k as f32 + frac));
        }
    }
    endpoints.retain(|x| x.is_finite());
    endpoints.sort_by(|a, b| a.partial_cmp(b).unwrap());
    endpoints.dedup();

    for &l in &endpoints {
        for &u in &endpoints {
            if u < l {
                continue;
            }
            cases += 1;
            let r = relax(l, u);
            for x in pw_candidates(f64::from(l), f64::from(u), grid_samples, breakpoint_limit) {
                check_point(&r, l, u, 0.0, x, f(x), 0.0, &mut out);
            }
        }
    }
    for _ in 0..random_cases {
        let (mut l, mut u) = (rng.f32_scaled(12), rng.f32_scaled(12));
        if l > u {
            std::mem::swap(&mut l, &mut u);
        }
        cases += 1;
        let r = relax(l, u);
        for x in pw_candidates(f64::from(l), f64::from(u), grid_samples, breakpoint_limit) {
            check_point(&r, l, u, 0.0, x, f(x), 0.0, &mut out);
        }
    }
    println!("{name} {profile:?} cases probed: {cases}");
    (cases, report(name, out))
}

#[test]
fn audit_piecewise_constant() {
    let profile = AuditProfile::from_environment();
    let (floor_cases, nf) = audit_pw(
        "Floor",
        FloorLayer::crown_relaxation_for_audit,
        |x| x.floor(),
        profile,
    );
    let (ceil_cases, nc) = audit_pw(
        "Ceil",
        CeilLayer::crown_relaxation_for_audit,
        |x| x.ceil(),
        profile,
    );
    let (trunc_cases, nt) = audit_pw(
        "Trunc",
        TruncLayer::crown_relaxation_for_audit,
        |x| x.trunc(),
        profile,
    );
    let (round_away_cases, nr_away) = audit_pw(
        "Round(half-away-from-zero, as documented)",
        RoundLayer::crown_relaxation_for_audit,
        |x| x.round(),
        profile,
    );
    let (round_even_cases, nr_even) = audit_pw(
        "Round(half-to-even, ONNX Round spec)",
        RoundLayer::crown_relaxation_for_audit,
        round_half_even,
        profile,
    );
    assert!(
        [
            floor_cases,
            ceil_cases,
            trunc_cases,
            round_away_cases,
            round_even_cases,
        ]
        .into_iter()
        .all(|cases| cases > 1_000),
        "piecewise-constant audit corpus unexpectedly empty"
    );
    println!("floor={nf} ceil={nc} trunc={nt} round_away={nr_away} round_even={nr_even}");
    assert_eq!(
        (nf, nc, nt, nr_away, nr_even),
        (0, 0, 0, 0, 0),
        "piecewise-constant envelope violations found"
    );
}
