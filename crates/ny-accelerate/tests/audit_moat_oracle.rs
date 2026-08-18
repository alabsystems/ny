// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! ADVERSARIAL AUDIT (independent of the crate's own tests).
//!
//! Question: with the seam ARMED vs UNARMED, is the PUBLISHED enclosure
//! `Ĉ ± γ_k·S` equal-or-wider, and does it still contain the exact real product?
//!
//! * UNARMED reference = faer blocked f64 matmul, which is literally what
//!   `crown_single::aw_f64_with_abssum_unbounded` runs when no engine is
//!   installed (`faer_parallelism::mat_mul_f64` → `matmul(Accum::Replace)`).
//! * ARMED = `AccelerateGemmEngine::gemm_f64` (cblas_dgemm).
//! * Truth = `BigRational`. Infinite precision, no reference error.
//!
//! Reported per case: enclosure of the exact value by BOTH intervals, the
//! radius ratio armed/unarmed, and whether the armed interval CONTAINS the
//! unarmed interval (the strict "equal or wider" reading).

#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use faer::linalg::matmul::matmul;
use faer::{Accum, MatMut, MatRef, Par};
use num_rational::BigRational;
use num_traits::{Signed, Zero};

use ny_accelerate::AccelerateGemmEngine;
use ny_core::GemmEngine;

fn engine() -> AccelerateGemmEngine {
    AccelerateGemmEngine::new_with_gates(true, false).expect("engine constructs")
}

fn exact(x: f64) -> BigRational {
    BigRational::from_float(x).expect("finite")
}

/// The gamma NY actually publishes (outward-rounded f64), as an exact rational.
fn gamma_published(k: usize) -> BigRational {
    exact(ny_core::dd::gamma_n_f64(k))
}

/// The UNARMED kernel, byte-for-byte the call `mat_mul_f64` makes.
fn faer_gemm(m: usize, k: usize, n: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
    let a = MatRef::from_row_major_slice(a, m, k);
    let b = MatRef::from_row_major_slice(b, k, n);
    let mut out = vec![0.0f64; m * n];
    {
        let dst = MatMut::from_row_major_slice_mut(&mut out, m, n);
        matmul(dst, Accum::Replace, a, b, 1.0, Par::Seq);
    }
    out
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }
    /// Uniform in [-1, 1).
    fn unit(&mut self) -> f64 {
        let s = self.next();
        ((s >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
    }
    fn f32_unit(&mut self) -> f64 {
        f64::from(self.unit() as f32)
    }
    /// Sign * (1+mant) * 2^e for e uniform in [-span, span).
    fn wide(&mut self, span: i32) -> f64 {
        let s = self.next();
        let e = ((s >> 40) % (2 * span as u64)) as i32 - span;
        let mant = ((s >> 12) & ((1u64 << 52) - 1)) as f64 / (1u64 << 52) as f64;
        let sign = if (s >> 3) & 1 == 1 { -1.0 } else { 1.0 };
        sign * (1.0 + mant) * f64::powi(2.0, e)
    }
}

#[derive(Clone, Copy, Debug)]
enum Dist {
    /// The REAL seam domain: exactly-widened f32 CROWN coefficients / weights.
    CrownF32Widened,
    /// Same domain, but every adjacent pair of A cancels exactly — drives
    /// |dot| → 0 while S stays O(k), the worst case for a relative-error claim.
    CrownCancelling,
    /// Same domain, all operands positive: no cancellation, the error
    /// accumulates monotonically instead of averaging out.
    CrownAllPositive,
    /// Genuine f64 with 2^±200 dynamic range (the conv / forward_linear surface).
    WideDynamicRange,
    /// f32 subnormal-adjacent entries (2^-149) mixed with 2^+100 entries:
    /// the largest magnitude spread the f32-widened domain can produce.
    F32ExtremeSpread,
    /// One row/col of exact small integers surrounded by 2^±300 noise:
    /// detects block mixing / Strassen contamination in the published entry.
    ExactIslandInNoise,
}

fn generate(dist: Dist, m: usize, k: usize, n: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut r = Lcg(seed | 1);
    let (mut a, mut b) = (vec![0.0f64; m * k], vec![0.0f64; k * n]);
    match dist {
        Dist::CrownF32Widened => {
            for v in a.iter_mut() {
                *v = r.f32_unit() * 0.5;
            }
            for v in b.iter_mut() {
                *v = r.f32_unit() * 0.1;
            }
        }
        Dist::CrownCancelling => {
            for v in a.iter_mut() {
                *v = r.f32_unit();
            }
            for v in b.iter_mut() {
                *v = r.f32_unit();
            }
            for i in 0..m {
                let mut l = 0;
                while l + 1 < k {
                    a[i * k + l + 1] = -a[i * k + l];
                    l += 2;
                }
            }
        }
        Dist::CrownAllPositive => {
            for v in a.iter_mut() {
                *v = r.f32_unit().abs() + f64::from(1e-3f32);
            }
            for v in b.iter_mut() {
                *v = r.f32_unit().abs() + f64::from(1e-3f32);
            }
        }
        Dist::WideDynamicRange => {
            for v in a.iter_mut() {
                *v = r.wide(200);
            }
            for v in b.iter_mut() {
                *v = r.wide(200);
            }
        }
        Dist::F32ExtremeSpread => {
            for v in a.iter_mut() {
                let s = r.next();
                let e = if s & 1 == 0 { -149 } else { 100 };
                *v = f64::from((r.f32_unit() as f32) * f32::powi(2.0, e));
            }
            for v in b.iter_mut() {
                let s = r.next();
                let e = if s & 1 == 0 { -149 } else { 100 };
                *v = f64::from((r.f32_unit() as f32) * f32::powi(2.0, e));
            }
        }
        Dist::ExactIslandInNoise => {
            for v in a.iter_mut() {
                *v = r.wide(300);
            }
            for v in b.iter_mut() {
                *v = r.wide(300);
            }
            for l in 0..k {
                a[(m - 1) * k + l] = ((r.next() >> 40) % 2047) as f64 - 1023.0;
                b[l * n + (n - 1)] = ((r.next() >> 40) % 2047) as f64 - 1023.0;
            }
        }
    }
    (a, b)
}

struct CaseVerdict {
    armed_violations: usize,
    unarmed_violations: usize,
    worst_armed_ratio: f64,
    worst_unarmed_ratio: f64,
    /// Entries where the ARMED radius is strictly SMALLER than the unarmed one.
    armed_radius_narrower: usize,
    worst_radius_ratio_min: f64,
    worst_radius_ratio_max: f64,
    /// Entries where the ARMED interval fails to contain the UNARMED interval.
    armed_not_superset: usize,
    /// Worst shortfall of that containment, in ulps of the armed radius.
    worst_containment_shortfall_rel: f64,
    /// Center movement |c_armed - c_unarmed| / armed radius, worst entry.
    worst_center_move_rel: f64,
}

fn ratio_f64(r: &BigRational) -> f64 {
    let (n, d) = (r.numer(), r.denom());
    let shift = n.bits().max(d.bits()).saturating_sub(900) as u32;
    let nn = n >> shift;
    let dd = d >> shift;
    let nf: f64 = nn.to_string().parse().unwrap_or(f64::NAN);
    let df: f64 = dd.to_string().parse().unwrap_or(f64::NAN);
    nf / df
}

fn run_case(eng: &AccelerateGemmEngine, dist: Dist, m: usize, k: usize, n: usize, seed: u64) {
    let (a, b) = generate(dist, m, k, n, seed);
    let (aa, ab): (Vec<f64>, Vec<f64>) = (
        a.iter().map(|x| x.abs()).collect(),
        b.iter().map(|x| x.abs()).collect(),
    );

    // ARMED: both the coefficient and the abs-sum through Accelerate.
    let c_armed = match eng.gemm_f64(m, k, n, &a, &b) {
        Ok(v) => v,
        Err(e) => {
            println!("  {dist:?} {m}x{k}x{n}: engine DECLINED ({e}) — unarmed path retained");
            return;
        }
    };
    let s_armed = eng.gemm_f64(m, k, n, &aa, &ab).expect("abs-sum accepted");
    // UNARMED: the faer kernel `mat_mul_f64` runs today.
    let c_unarmed = faer_gemm(m, k, n, &a, &b);
    let s_unarmed = faer_gemm(m, k, n, &aa, &ab);

    let gamma = gamma_published(k);
    let mut v = CaseVerdict {
        armed_violations: 0,
        unarmed_violations: 0,
        worst_armed_ratio: 0.0,
        worst_unarmed_ratio: 0.0,
        armed_radius_narrower: 0,
        worst_radius_ratio_min: f64::INFINITY,
        worst_radius_ratio_max: 0.0,
        armed_not_superset: 0,
        worst_containment_shortfall_rel: 0.0,
        worst_center_move_rel: 0.0,
    };

    for i in 0..m {
        for j in 0..n {
            let mut dot = BigRational::zero();
            for l in 0..k {
                dot += exact(a[i * k + l]) * exact(b[l * n + j]);
            }
            let ca = exact(c_armed[i * n + j]);
            let cu = exact(c_unarmed[i * n + j]);
            let ra = &gamma * exact(s_armed[i * n + j]);
            let ru = &gamma * exact(s_unarmed[i * n + j]);

            // (1) Enclosure of the EXACT product by the published interval.
            let ea = (&ca - &dot).abs();
            let eu = (&cu - &dot).abs();
            if ea > ra {
                v.armed_violations += 1;
            }
            if eu > ru {
                v.unarmed_violations += 1;
            }
            if !ra.is_zero() {
                v.worst_armed_ratio = v.worst_armed_ratio.max(ratio_f64(&(&ea / &ra)));
            } else if !ea.is_zero() {
                v.armed_violations += 1;
            }
            if !ru.is_zero() {
                v.worst_unarmed_ratio = v.worst_unarmed_ratio.max(ratio_f64(&(&eu / &ru)));
            }

            // (2) Radius comparison.
            if ra < ru {
                v.armed_radius_narrower += 1;
            }
            if !ru.is_zero() {
                let q = ratio_f64(&(&ra / &ru));
                v.worst_radius_ratio_min = v.worst_radius_ratio_min.min(q);
                v.worst_radius_ratio_max = v.worst_radius_ratio_max.max(q);
            }

            // (3) Strict "equal or wider": [ca-ra, ca+ra] ⊇ [cu-ru, cu+ru]?
            let lo_gap = (&ca - &ra) - (&cu - &ru); // must be <= 0
            let hi_gap = (&cu + &ru) - (&ca + &ra); // must be <= 0
            let worst_gap = if lo_gap > hi_gap { lo_gap } else { hi_gap };
            if worst_gap > BigRational::zero() {
                v.armed_not_superset += 1;
                if !ra.is_zero() {
                    v.worst_containment_shortfall_rel = v
                        .worst_containment_shortfall_rel
                        .max(ratio_f64(&(&worst_gap / &ra)));
                }
            }
            if !ra.is_zero() {
                v.worst_center_move_rel = v
                    .worst_center_move_rel
                    .max(ratio_f64(&((&ca - &cu).abs() / &ra)));
            }
        }
    }

    println!(
        "  {dist:?} {m}x{k}x{n}: enclose_exact armed_viol={} unarmed_viol={} | \
         worst|err|/(γS): armed={:.6} unarmed={:.6} | radius armed/unarmed ∈ [{:.9},{:.9}] \
         narrower={} | armed⊉unarmed={} (worst gap {:.3e}·r) | center move <= {:.3}·r \
         | radius rel-dev [{:.3e},{:.3e}]",
        v.armed_violations,
        v.unarmed_violations,
        v.worst_armed_ratio,
        v.worst_unarmed_ratio,
        v.worst_radius_ratio_min,
        v.worst_radius_ratio_max,
        v.armed_radius_narrower,
        v.armed_not_superset,
        v.worst_containment_shortfall_rel,
        v.worst_center_move_rel,
        v.worst_radius_ratio_min - 1.0,
        v.worst_radius_ratio_max - 1.0,
    );

    assert_eq!(
        v.armed_violations, 0,
        "STOP THE LINE: ARMED published interval does NOT contain the exact product \
         for {dist:?} at {m}x{k}x{n}"
    );
    assert_eq!(
        v.unarmed_violations, 0,
        "the UNARMED reference itself violated its envelope for {dist:?} at {m}x{k}x{n} \
         — the oracle or the reference is wrong"
    );
}

#[test]
fn published_enclosure_is_sound_armed_and_unarmed_across_k_and_distributions() {
    let eng = engine();
    println!("{}", eng.install_summary());
    let (m, n) = (8usize, 8usize);
    for &dist in &[
        Dist::CrownF32Widened,
        Dist::CrownCancelling,
        Dist::CrownAllPositive,
        Dist::WideDynamicRange,
        Dist::F32ExtremeSpread,
        Dist::ExactIslandInNoise,
    ] {
        println!("dist {dist:?}");
        for &k in &[512usize, 1024, 2048, 4096, 8192] {
            run_case(&eng, dist, m, k, n, 0x51ED_0000 ^ (k as u64));
        }
    }
}

/// Non-square, CROWN-shaped, and the m=1 / n=1 degenerate strips that a blocked
/// kernel handles with a different code path.
#[test]
fn published_enclosure_is_sound_at_crown_shapes() {
    let eng = engine();
    for &(m, k, n) in &[
        (1usize, 4096usize, 64usize),
        (64, 4096, 1),
        (7, 4099, 5),   // all prime-ish, no clean blocking
        (64, 512, 64),  // 2^21 MACs
        (33, 1023, 17), // odd everywhere
    ] {
        run_case(&eng, Dist::CrownF32Widened, m, k, n, 0xC0FFEE);
        run_case(&eng, Dist::CrownCancelling, m, k, n, 0xBEEF);
    }
}

/// G2: the underflow domain must be REFUSED, not computed.
#[test]
fn g2_refuses_the_subnormal_product_domain() {
    let eng = engine();
    let (m, k, n) = (8usize, 512usize, 8usize);
    let a = vec![f64::powi(2.0, -600); m * k];
    let b = vec![f64::powi(2.0, -600); k * n];
    let r = eng.gemm_f64(m, k, n, &a, &b);
    assert!(
        r.is_err(),
        "G2 admitted a product domain that can go subnormal"
    );
    println!("G2 refusal: {}", r.unwrap_err());
}
