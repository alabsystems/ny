// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! M4 level-search tests. Every oracle is INDEPENDENT of the kernel: a scalar
//! f64 transcription of the frozen-dual node value (the formula of
//! `FrozenDualProblem::eval_node`, sequential sums, no lanes, no tiles) and of
//! the per-node search step (`zono_bab::eval_search_node`, with this crate's
//! far-first rule).
//!
//! * `fd_f64_values_match_the_scalar_reference_at_the_proposed_points` — the
//!   f64 instantiation's value and `S` at every point it scored equal the
//!   reference AT THE MULTIPLIERS IT PROPOSES (the contract the lane relies
//!   on: it re-scores exactly those multipliers);
//! * `fd_f32_tracks_the_reference_at_its_own_points` — the shipped f32 search
//!   is within `2^-12 S` of the reference at its own proposed points;
//! * `fd_step_matches_the_f64_search_rules_on_exact_problems` — on dyadic
//!   problems (every sum exact in both scalars) the f64 instantiation's
//!   clamped, far and children's multipliers equal the transcription bit for
//!   bit, and the f32 search's VALUES are exact too;
//! * `fd_search_is_thread_count_and_chunk_independent` (R6);
//! * `fd_refuses_malformed_foreign_oversized_and_late_requests`;
//! * `fd_nonfinite_and_negative_multipliers_are_sanitized`;
//! * `fd_proposals_are_nonnegative_finite_dual_points`;
//! * `fd_far_first_skips_only_clearly_positive_nodes`;
//! * `fd_fast_ln_tracks_libm`;
//! * `fd_fast_exp_tracks_libm` (the platform-independent step `exp` the device
//!   kernels mirror);
//! * `fd_work_bytes_cover_the_workspace`;
//! * the integrator's `fd_cpu_rate_probe` (behind the `benchmarks` feature).

use std::time::{Duration, Instant};

use ny_core::search::{
    DeviceLease, FdHandle, FdSearchProblem, LevelChunk, LevelOut, Refusal, SearchBackend,
    FD_CLAMPED, FD_FAR, FD_FLOOR, FD_LOG_BUCKETS, FD_POINTS, FD_SEARCH_TOL, FD_WARM,
};

use crate::cpu::{fd_level, upload_fd};
use crate::fd::{
    fast_exp_f32, fast_ln_f32, tau_rel, work_bytes, workspace_capacity_bytes, FdProgram, FD_LANES,
    FD_TILE,
};
use crate::CpuF32Backend;

const BIG: DeviceLease = DeviceLease { bytes: 1 << 34 };

/// splitmix64 (the crate has no RNG dependency).
struct Mix(u64);

impl Mix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, k: u64) -> u64 {
        self.next() % k.max(1)
    }

    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }

    /// `k / den` with `k` uniform in `lo..=hi`.
    fn dyadic(&mut self, lo: i64, hi: i64, den: f64) -> f32 {
        let k = lo + self.below((hi - lo + 1) as u64) as i64;
        (k as f64 / den) as f32
    }
}

/// A random search copy (`exact`: dyadic entries whose every sum the search
/// forms is exact in f32 and f64).
fn problem(n: usize, depth: usize, seed: u64, exact: bool) -> FdSearchProblem {
    let mut rng = Mix(seed ^ 0x5EED_F00D);
    let depth = depth.min(n);
    let mut cols: Vec<u32> = (0..n as u32).collect();
    for i in 0..depth {
        let j = i + rng.below((n - i) as u64) as usize;
        cols.swap(i, j);
    }
    let a: Vec<f32> = (0..depth * n)
        .map(|_| {
            if rng.below(5) == 0 {
                0.0
            } else if exact {
                rng.dyadic(-16, 16, 16.0)
            } else {
                rng.range(-1.0, 1.0) as f32
            }
        })
        .collect();
    let row_abs: Vec<f32> = (0..depth)
        .map(|i| a[i * n..(i + 1) * n].iter().map(|v| v.abs()).sum::<f32>())
        .collect();
    let mut pair = |lo: f64, hi: f64, den: f64| -> [f32; 2] {
        if exact {
            [
                rng.dyadic((lo * den) as i64, (hi * den) as i64, den),
                rng.dyadic((lo * den) as i64, (hi * den) as i64, den),
            ]
        } else {
            [rng.range(lo, hi) as f32, rng.range(lo, hi) as f32]
        }
    };
    let rho: Vec<[f32; 2]> = (0..depth).map(|_| pair(-2.0, 2.0, 8.0)).collect();
    let c0s: Vec<[f32; 2]> = (0..depth).map(|_| pair(-0.5, 0.0, 32.0)).collect();
    let b0: Vec<[f32; 2]> = (0..depth).map(|_| pair(0.0, 3.0, 16.0)).collect();
    let b1: Vec<[f32; 2]> = (0..depth).map(|_| pair(0.0, 3.0, 16.0)).collect();
    let d: Vec<f32> = (0..n)
        .map(|_| {
            if exact {
                rng.dyadic(-16, 16, 16.0)
            } else {
                rng.range(-1.0, 1.0) as f32
            }
        })
        .collect();
    let abs_d: f32 = d.iter().map(|v| v.abs()).sum();
    // A margin constant that puts the raw values of random nodes near zero,
    // so the search sees positive and negative nodes alike.
    let c0 = if exact {
        (0.5 * abs_d).floor()
    } else {
        (0.5 * f64::from(abs_d)) as f32
    };
    FdSearchProblem {
        n,
        depth,
        a,
        enc: cols[..depth].to_vec(),
        rho,
        c0s,
        b0,
        b1,
        row_abs,
        d,
        c0,
        abs_d,
    }
}

/// A random level of `count` nodes of `depth` (`exact`: dyadic multipliers).
fn level(count: usize, depth: usize, seed: u64, exact: bool) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let mut rng = Mix(seed ^ 0x1E7E_1000);
    let mut sides = Vec::with_capacity(count * depth);
    let mut l0 = Vec::with_capacity(count * depth);
    let mut l1 = Vec::with_capacity(count * depth);
    for _ in 0..count * depth {
        sides.push(u8::from(rng.below(2) == 1));
        let mut m = || {
            if rng.below(4) == 0 {
                0.0
            } else if exact {
                rng.dyadic(0, 16, 8.0)
            } else {
                rng.range(0.0, 2.0) as f32
            }
        };
        l0.push(m());
        l1.push(m());
    }
    (sides, l0, l1)
}

fn chunk<'a>(depth: usize, sides: &'a [u8], l0: &'a [f32], l1: &'a [f32]) -> LevelChunk<'a> {
    LevelChunk {
        depth,
        count: sides.len() / depth,
        sides,
        l0,
        l1,
        deadline: None,
    }
}

/// The node value, `S`, `rc` and supergradient at `(m0, m1)` (clamped at 0),
/// in plain sequential f64 — `FrozenDualProblem::eval_node`'s formula on the
/// search copy.
struct RefNode {
    value: f64,
    big_s: f64,
    rc: Vec<f64>,
    s0: Vec<f64>,
    s1: Vec<f64>,
}

fn reference(p: &FdSearchProblem, sides: &[u8], m0: &[f64], m1: &[f64]) -> Option<RefNode> {
    let (n, d) = (p.n, sides.len());
    let mut rc: Vec<f64> = p.d.iter().map(|&v| f64::from(v)).collect();
    for &e in &p.enc[..d] {
        rc[e as usize] = 0.0;
    }
    let mut c0path = f64::from(p.c0);
    let mut c0abs = c0path.abs();
    let (mut lag, mut lagabs, mut tsum) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..d {
        let a = m0[i].max(0.0);
        let b = m1[i].max(0.0);
        if !a.is_finite() || !b.is_finite() {
            return None;
        }
        let s = usize::from(sides[i]);
        let sign = if s == 0 { 1.0 } else { -1.0 };
        let rho = f64::from(p.rho[i][s]);
        let mult = rho + sign * (a - b);
        for (j, r) in rc.iter_mut().enumerate() {
            *r += mult * f64::from(p.a[i * n + j]);
        }
        let cs = f64::from(p.c0s[i][s]);
        c0path += cs;
        c0abs += cs.abs();
        let (b0, b1) = (f64::from(p.b0[i][s]), f64::from(p.b1[i][s]));
        lag += a * b0 + b * b1;
        lagabs += a * b0.abs() + b * b1.abs();
        tsum += (rho.abs() + a + b) * f64::from(p.row_abs[i]);
    }
    let sumabs: f64 = rc.iter().map(|v| v.abs()).sum();
    let value = c0path - sumabs - lag;
    let big_s = c0abs + sumabs + f64::from(p.abs_d) + tsum + lagabs;
    let mut s0 = vec![0.0; d];
    let mut s1 = vec![0.0; d];
    for i in 0..d {
        let pv: f64 = (0..n)
            .map(|j| {
                let av = f64::from(p.a[i * n + j]);
                if rc[j] < 0.0 {
                    av
                } else {
                    -av
                }
            })
            .sum();
        let s = usize::from(sides[i]);
        let sign = if s == 0 { 1.0 } else { -1.0 };
        s0[i] = sign * pv - f64::from(p.b0[i][s]);
        s1[i] = -sign * pv - f64::from(p.b1[i][s]);
    }
    value.is_finite().then_some(RefNode {
        value,
        big_s,
        rc,
        s0,
        s1,
    })
}

/// The search step of one node, transcribed from `zono_bab::eval_search_node`
/// (log-bucket line search, clamped far probe, children's step) with THIS
/// crate's far-first rule. Returns `(clamped, far, next)` multipliers and the
/// point values (NaN = skipped).
#[allow(clippy::type_complexity)]
fn reference_step(
    p: &FdSearchProblem,
    sides: &[u8],
    l0: &[f64],
    l1: &[f64],
) -> Option<(
    (Vec<f64>, Vec<f64>),
    (Vec<f64>, Vec<f64>),
    (Vec<f64>, Vec<f64>),
    [f64; 4],
)> {
    let d = sides.len();
    let warm = reference(p, sides, l0, l1)?;
    let tol = FD_SEARCH_TOL;
    let (mut c0, mut c1) = (vec![0.0; d], vec![0.0; d]);
    let (mut sp0, mut sp1) = (vec![0.0; d], vec![0.0; d]);
    let (mut clamped_differs, mut any_mult, mut slope0) = (false, false, 0.0f64);
    let mut da = vec![0.0f64; p.n];
    for i in 0..d {
        c0[i] = if warm.s0[i] < 0.0 { 0.0 } else { l0[i] };
        c1[i] = if warm.s1[i] < 0.0 { 0.0 } else { l1[i] };
        clamped_differs |= c0[i] != l0[i] || c1[i] != l1[i];
        any_mult |= l0[i] != 0.0 || l1[i] != 0.0;
        sp0[i] = if l0[i] <= tol && warm.s0[i] < 0.0 {
            0.0
        } else {
            warm.s0[i]
        };
        sp1[i] = if l1[i] <= tol && warm.s1[i] < 0.0 {
            0.0
        } else {
            warm.s1[i]
        };
        slope0 += sp0[i] * warm.s0[i] + sp1[i] * warm.s1[i];
        let sign = if sides[i] == 0 { 1.0 } else { -1.0 };
        let m = sign * (sp0[i] - sp1[i]);
        if m != 0.0 {
            for (j, v) in da.iter_mut().enumerate() {
                *v += m * f64::from(p.a[i * p.n + j]);
            }
        }
    }
    // log_bucket_step
    let positive = |r: f64, v: f64| (v > 0.0 && r < 0.0) || (v < 0.0 && r > 0.0);
    let (mut emin, mut emax) = (f64::INFINITY, f64::NEG_INFINITY);
    for (&r, &v) in warm.rc.iter().zip(&da) {
        if positive(r, v) {
            let e = -r / v;
            emin = emin.min(e);
            emax = emax.max(e);
        }
    }
    let emin = emin.max(1e-12);
    let emax = emax.max(1e-12);
    let le = emin.ln();
    let lw = (emax.ln() - le).max(1e-30) / FD_LOG_BUCKETS as f64;
    let mut buckets = vec![0.0f64; FD_LOG_BUCKETS];
    for (&r, &v) in warm.rc.iter().zip(&da) {
        if positive(r, v) {
            let e = -r / v;
            let f = (((e.max(1e-12)).ln() - le) / lw).floor();
            let top = (FD_LOG_BUCKETS - 1) as f64;
            let idx = if f.is_nan() || f <= 0.0 {
                0
            } else if f >= top {
                FD_LOG_BUCKETS - 1
            } else {
                f as usize
            };
            buckets[idx] += 2.0 * v.abs();
        }
    }
    let mut cum = 0.0f64;
    let mut first = None;
    for (k, &b) in buckets.iter().enumerate() {
        cum += b;
        if cum >= slope0 {
            first = Some(k);
            break;
        }
    }
    let eta = first.map_or(emax, |k| (le + k as f64 * lw).exp());
    let eta = if eta.is_finite() { eta } else { 0.0 };
    let mut ebnd = f64::INFINITY;
    for i in 0..d {
        if sp0[i] < -tol {
            ebnd = ebnd.min(-l0[i] / sp0[i]);
        }
        if sp1[i] < -tol {
            ebnd = ebnd.min(-l1[i] / sp1[i]);
        }
    }
    let far = if ebnd.is_finite() { ebnd } else { emax * 1e6 };
    let mut eta_c = if first.is_some() { eta } else { far };
    if !(slope0 > tol && eta_c.is_finite()) {
        eta_c = 0.0;
    }
    let eta_c = eta_c.max(0.0);
    let f0: Vec<f64> = (0..d).map(|i| (l0[i] + eta_c * sp0[i]).max(0.0)).collect();
    let f1: Vec<f64> = (0..d).map(|i| (l1[i] + eta_c * sp1[i]).max(0.0)).collect();
    let far_v = reference(p, sides, &f0, &f1).map_or(f64::NAN, |r| r.value);
    let far_s = reference(p, sides, &f0, &f1).map_or(f64::NAN, |r| r.big_s);
    let rel = tau_rel(d);
    let strong = |v: f64, s: f64| v.is_finite() && s.is_finite() && v > rel * s;
    let skip = strong(warm.value, warm.big_s) || strong(far_v, far_s);
    let clamped_v = if !clamped_differs {
        warm.value
    } else if skip {
        f64::NAN
    } else {
        reference(p, sides, &c0, &c1).map_or(f64::NAN, |r| r.value)
    };
    let zeros = vec![0.0; d];
    let floor_v = if !any_mult {
        warm.value
    } else if skip {
        f64::NAN
    } else {
        reference(p, sides, &zeros, &zeros).map_or(f64::NAN, |r| r.value)
    };
    let mut pre_best = warm.value;
    for v in [clamped_v, floor_v] {
        if v.is_finite() {
            pre_best = pre_best.max(v);
        }
    }
    let pending = pre_best <= tol;
    let eta_s = if pending && slope0 > tol {
        eta.max(0.0)
    } else {
        0.0
    };
    let n0: Vec<f64> = (0..d).map(|i| (l0[i] + eta_s * sp0[i]).max(0.0)).collect();
    let n1: Vec<f64> = (0..d).map(|i| (l1[i] + eta_s * sp1[i]).max(0.0)).collect();
    Some((
        (c0, c1),
        (f0, f1),
        (n0, n1),
        [warm.value, clamped_v, floor_v, far_v],
    ))
}

fn widen(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| f64::from(x)).collect()
}

/// Run the `R` instantiation over a whole level in one call.
fn run<R: crate::fd::FdReal>(
    p: &FdSearchProblem,
    depth: usize,
    sides: &[u8],
    l0: &[f32],
    l1: &[f32],
    threads: usize,
) -> LevelOut {
    let h = upload_fd::<R>("test", p, &BIG, threads).expect("upload");
    let mut out = LevelOut::default();
    let _stats = fd_level::<R>(&h, &chunk(depth, sides, l0, l1), &mut out).expect("fd_level");
    out
}

/// The multipliers point `pt` of node `k` was scored at.
fn point_mults(
    view: &ny_core::search::LevelView<'_>,
    k: usize,
    pt: usize,
    l0: &[f32],
    l1: &[f32],
) -> (Vec<f64>, Vec<f64>) {
    let d = view.depth();
    match pt {
        FD_WARM => (
            widen(&l0[k * d..(k + 1) * d]),
            widen(&l1[k * d..(k + 1) * d]),
        ),
        FD_CLAMPED => {
            let (a, b) = view.clamped(k);
            (widen(a), widen(b))
        }
        FD_FLOOR => (vec![0.0; d], vec![0.0; d]),
        _ => {
            let (a, b) = view.far(k);
            (widen(a), widen(b))
        }
    }
}

#[test]
fn fd_f64_values_match_the_scalar_reference_at_the_proposed_points() {
    let mut checked = [0usize; FD_POINTS];
    for (n, depth, seed) in [
        (1usize, 1usize, 1u64),
        (37, 5, 2),
        (300, 12, 3),
        (700, 30, 4),
    ] {
        for d in [1, depth / 2 + 1, depth] {
            let p = problem(n, depth, seed, false);
            let count = 3 * FD_LANES + 5;
            let (sides, l0, l1) = level(count, d, seed ^ d as u64, false);
            let out = run::<f64>(&p, d, &sides, &l0, &l1, 3);
            let view = out.hint().into_ingest();
            assert!(view.well_formed());
            for k in 0..count {
                let s = &sides[k * d..(k + 1) * d];
                for pt in 0..FD_POINTS {
                    let v = f64::from(view.value(k, pt));
                    if !v.is_finite() {
                        continue;
                    }
                    let (m0, m1) = point_mults(&view, k, pt, &l0, &l1);
                    let r = reference(&p, s, &m0, &m1).expect("the reference scores it");
                    let tol = 1e-6 * r.big_s + 1e-30;
                    assert!(
                        (v - r.value).abs() <= tol,
                        "n={n} d={d} node {k} point {pt}: search {v:e} vs reference {:e} \
                         (tol {tol:e}): the value does not belong to the proposed multipliers",
                        r.value
                    );
                    let s_out = f64::from(view.scale(k, pt));
                    assert!(
                        (s_out - r.big_s).abs() <= 1e-6 * r.big_s,
                        "n={n} d={d} node {k} point {pt}: S {s_out:e} vs {:e}",
                        r.big_s
                    );
                    checked[pt] += 1;
                }
            }
        }
    }
    assert!(
        checked.iter().all(|&c| c > 0),
        "VOID: some point was never scored: {checked:?}"
    );
}

#[test]
fn fd_f32_tracks_the_reference_at_its_own_points() {
    let mut checked = 0usize;
    let mut worst = 0.0f64;
    for (n, depth, seed) in [(64usize, 6usize, 11u64), (513, 17, 12), (2049, 28, 13)] {
        let p = problem(n, depth, seed, false);
        let count = 2 * FD_LANES + 3;
        let (sides, l0, l1) = level(count, depth, seed, false);
        let out = run::<f32>(&p, depth, &sides, &l0, &l1, 2);
        let view = out.hint().into_ingest();
        for k in 0..count {
            let s = &sides[k * depth..(k + 1) * depth];
            for pt in 0..FD_POINTS {
                let v = f64::from(view.value(k, pt));
                if !v.is_finite() {
                    continue;
                }
                let (m0, m1) = point_mults(&view, k, pt, &l0, &l1);
                let r = reference(&p, s, &m0, &m1).expect("reference");
                let rel = (v - r.value).abs() / r.big_s;
                worst = worst.max(rel);
                assert!(
                    rel <= 2f64.powi(-12),
                    "n={n} node {k} point {pt}: f32 {v:e} vs f64 {:e} (S {:e}, rel {rel:e})",
                    r.value,
                    r.big_s
                );
                checked += 1;
            }
        }
    }
    eprintln!("[fd-f32-track] checked={checked} worst_rel={worst:e}");
    assert!(checked > 100, "VOID: only {checked} points checked");
}

#[test]
fn fd_step_matches_the_f64_search_rules_on_exact_problems() {
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    let (mut nodes, mut stepped, mut moved_far) = (0usize, 0usize, 0usize);
    for (n, depth, seed) in [(40usize, 4usize, 21u64), (300, 9, 22), (257, 12, 23)] {
        let p = problem(n, depth, seed, true);
        let count = 2 * FD_LANES + 7;
        let (sides, l0, l1) = level(count, depth, seed, true);
        let o64 = run::<f64>(&p, depth, &sides, &l0, &l1, 2);
        let o32 = run::<f32>(&p, depth, &sides, &l0, &l1, 2);
        let (v64, v32) = (o64.hint().into_ingest(), o32.hint().into_ingest());
        for k in 0..count {
            let r = k * depth..(k + 1) * depth;
            let s = &sides[r.clone()];
            let (a0, a1) = (widen(&l0[r.clone()]), widen(&l1[r]));
            let Some(((c0, c1), (f0, f1), (n0, n1), vals)) = reference_step(&p, s, &a0, &a1) else {
                continue;
            };
            let got = |(a, b): (&[f32], &[f32])| (widen(a), widen(b));
            let (gc0, gc1) = got(v64.clamped(k));
            let (gf0, gf1) = got(v64.far(k));
            let (gn0, gn1) = got(v64.next(k));
            // The f64 instantiation stores its proposals as f32: compare the
            // transcription rounded the same way.
            let r32 = |v: &[f64]| v.iter().map(|&x| f64::from(x as f32)).collect::<Vec<_>>();
            assert_eq!(
                (bits(&gc0), bits(&gc1)),
                (bits(&r32(&c0)), bits(&r32(&c1))),
                "node {k}: clamped"
            );
            assert_eq!(
                (bits(&gf0), bits(&gf1)),
                (bits(&r32(&f0)), bits(&r32(&f1))),
                "node {k}: far"
            );
            assert_eq!(
                (bits(&gn0), bits(&gn1)),
                (bits(&r32(&n0)), bits(&r32(&n1))),
                "node {k}: children's warm start"
            );
            for pt in 0..FD_POINTS {
                let want = vals[pt] as f32;
                for (tag, view) in [("f64", &v64), ("f32", &v32)] {
                    let have = view.value(k, pt);
                    // The f32 search's far point is its OWN (its line search
                    // uses the polynomial ln), so only compare it in f64.
                    if tag == "f32" && pt == FD_FAR {
                        continue;
                    }
                    if tag == "f32" && pt != FD_WARM && have.is_nan() != want.is_nan() {
                        // Far-first can skip differently once the far points differ.
                        continue;
                    }
                    if pt == FD_FAR && have.is_finite() && want.is_finite() {
                        // The far multipliers are not dyadic, so the order of
                        // the `rc` sums moves the last f64 bits of its value.
                        assert!(
                            (have - want).abs() <= 1e-6 * want.abs().max(1.0),
                            "{tag} node {k} far point: {have:e} vs {want:e}"
                        );
                        continue;
                    }
                    assert!(
                        have.to_bits() == want.to_bits() || (have.is_nan() && want.is_nan()),
                        "{tag} node {k} point {pt}: {have:e} vs the exact {want:e}"
                    );
                }
            }
            nodes += 1;
            stepped += usize::from(n0 != a0 || n1 != a1);
            moved_far += usize::from(f0 != a0 || f1 != a1);
        }
    }
    eprintln!("[fd-step] nodes={nodes} stepped={stepped} moved_far={moved_far}");
    assert!(
        nodes > 50 && stepped > 0 && moved_far > 0,
        "VOID: nodes={nodes} stepped={stepped} moved_far={moved_far}"
    );
}

#[test]
fn fd_search_is_thread_count_and_chunk_independent() {
    let depth = 20;
    let count = 7 * FD_LANES + 9;
    // (a) Generic random data: every row is live in every block.
    let generic = (
        problem(777, depth, 31, false),
        level(count, depth, 31, false),
    );
    // (b) Rows whose Off-side `rho` and every node's multipliers are 0: such a
    // row is all-zero in a block holding only Off nodes there and live
    // otherwise, so WHICH nodes share a block (the lane's chunking) decides
    // whether the kernel visits the row and how it pairs the rows around it.
    // A node's bits must not move with that (the per-lane row-by-row sums of
    // `axpy2`); a pair-folded `x + (m0 a0 + m1 a1)` fails here.
    let zero_rows = {
        let mut p = problem(777, depth, 32, false);
        let (mut sides, mut l0, mut l1) = level(count, depth, 32, false);
        for i in (0..depth).step_by(3) {
            p.rho[i][0] = 0.0;
            for k in 0..count {
                l0[k * depth + i] = 0.0;
                l1[k * depth + i] = 0.0;
            }
            // Node 0, which the first cut below searches ALONE, takes the Off
            // side: the row is all-zero in its solo block.
            sides[i] = 0;
        }
        assert!(
            (0..depth)
                .step_by(3)
                .any(|i| (1..FD_LANES).any(|k| sides[k * depth + i] == 1)),
            "VOID: no zeroed row is live in node 0's whole-level block, so the solo and \
             shared blocks visit the same rows"
        );
        (p, (sides, l0, l1))
    };
    let digest = |o: &LevelOut| {
        let v = o.hint().into_ingest();
        let mut out = Vec::new();
        for k in 0..v.count() {
            for pt in 0..FD_POINTS {
                out.push(v.value(k, pt).to_bits());
                out.push(v.scale(k, pt).to_bits());
            }
            for (a, b) in [v.clamped(k), v.far(k), v.next(k)] {
                out.extend(a.iter().chain(b).map(|x| x.to_bits()));
            }
        }
        out
    };
    for (tag, (p, (sides, l0, l1))) in [("generic", generic), ("zero-rows", zero_rows)] {
        let base = digest(&run::<f32>(&p, depth, &sides, &l0, &l1, 1));
        for threads in [2usize, 3, 8] {
            assert_eq!(
                base,
                digest(&run::<f32>(&p, depth, &sides, &l0, &l1, threads)),
                "{tag}: {threads} threads changed the search"
            );
        }
        // The lane may cut a level anywhere: split at odd places (node 0
        // alone, then blocks straddling the whole-level run's blocks).
        let h = upload_fd::<f32>("test", &p, &BIG, 4).expect("upload");
        let mut joined = Vec::new();
        let mut lo = 0usize;
        for cut in [1usize, 17, 40, count] {
            let mut out = LevelOut::default();
            let r = lo * depth..cut * depth;
            let _stats = fd_level::<f32>(
                &h,
                &chunk(depth, &sides[r.clone()], &l0[r.clone()], &l1[r]),
                &mut out,
            )
            .expect("fd_level");
            joined.extend(digest(&out));
            lo = cut;
        }
        assert_eq!(base, joined, "{tag}: chunking the level changed the search");
    }
}

#[test]
fn fd_refuses_malformed_foreign_oversized_and_late_requests() {
    let p = problem(50, 6, 41, false);
    let backend = CpuF32Backend::with_threads(2);
    let h = backend.upload_fd(&p, &BIG).expect("upload");
    assert_eq!(h.backend(), "cpu_f32");
    let (sides, l0, l1) = level(5, 3, 41, false);
    let ok = chunk(3, &sides, &l0, &l1);
    let mut out = LevelOut::default();
    let stats = backend
        .fd_level(&h, &ok, &mut out)
        .expect("a valid chunk searches")
        .into_hint()
        .into_ingest();
    assert_eq!(stats.nodes, 5);
    assert!(matches!(
        backend.fd_level(&h, &LevelChunk { count: 4, ..ok }, &mut out),
        Err(Refusal::Invalid(_))
    ));
    let (s7, a7, b7) = level(2, 7, 1, false);
    assert!(
        matches!(
            backend.fd_level(&h, &chunk(7, &s7, &a7, &b7), &mut out),
            Err(Refusal::Invalid(_))
        ),
        "a depth beyond the upload must refuse"
    );
    assert!(matches!(
        backend.fd_level(&FdHandle::new("other", 3u8), &ok, &mut out),
        Err(Refusal::Invalid(_))
    ));
    let h64 = upload_fd::<f64>("test", &p, &BIG, 1).expect("upload");
    assert!(
        matches!(
            backend.fd_level(&h64, &ok, &mut out),
            Err(Refusal::Invalid(_))
        ),
        "an f64 upload is not the f32 backend's"
    );
    assert!(matches!(
        backend.upload_fd(&p, &DeviceLease { bytes: 1024 }),
        Err(Refusal::Memory { .. })
    ));
    let mut bad = p;
    bad.c0 = f32::NAN;
    assert!(matches!(
        backend.upload_fd(&bad, &BIG),
        Err(Refusal::Invalid(_))
    ));
    let late = LevelChunk {
        deadline: Some(
            Instant::now()
                .checked_sub(Duration::from_millis(1))
                .unwrap_or_else(Instant::now),
        ),
        ..ok
    };
    assert!(matches!(
        backend.fd_level(&h, &late, &mut out),
        Err(Refusal::Deadline)
    ));
}

#[test]
fn fd_nonfinite_and_negative_multipliers_are_sanitized() {
    let p = problem(90, 5, 51, false);
    let depth = 5;
    let (sides, clean0, clean1) = level(FD_LANES, depth, 51, false);
    let mut dirty0 = clean0.clone();
    let mut dirty1 = clean1.clone();
    let mut zeroed0 = clean0;
    let mut zeroed1 = clean1;
    for (k, bad) in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -3.0, -0.0]
        .into_iter()
        .enumerate()
    {
        dirty0[k * depth + k % depth] = bad;
        zeroed0[k * depth + k % depth] = 0.0;
        dirty1[(k + 5) * depth] = bad;
        zeroed1[(k + 5) * depth] = 0.0;
    }
    let a = run::<f32>(&p, depth, &sides, &dirty0, &dirty1, 2);
    let b = run::<f32>(&p, depth, &sides, &zeroed0, &zeroed1, 2);
    let (va, vb) = (a.hint().into_ingest(), b.hint().into_ingest());
    for k in 0..FD_LANES {
        assert_eq!(
            va.value(k, FD_WARM).to_bits(),
            vb.value(k, FD_WARM).to_bits(),
            "node {k}: a non-finite or negative multiplier must act as 0"
        );
        assert!(va.value(k, FD_WARM).is_finite());
        for (x, y) in [va.next(k), va.far(k), va.clamped(k)] {
            assert!(x.iter().chain(y).all(|v| v.is_finite() && *v >= 0.0));
        }
    }
}

#[test]
fn fd_proposals_are_nonnegative_finite_dual_points() {
    let mut proposals = 0usize;
    for seed in 0..6u64 {
        let p = problem(200 + 37 * seed as usize, 10, 60 + seed, false);
        let (sides, l0, l1) = level(40, 10, 60 + seed, false);
        let out = run::<f32>(&p, 10, &sides, &l0, &l1, 3);
        let v = out.hint().into_ingest();
        for k in 0..v.count() {
            for (x, y) in [v.clamped(k), v.far(k), v.next(k)] {
                for m in x.iter().chain(y) {
                    assert!(m.is_finite() && *m >= 0.0, "seed {seed} node {k}: {m}");
                    proposals += 1;
                }
            }
        }
    }
    assert!(proposals > 1000, "VOID");
}

#[test]
fn fd_far_first_skips_only_clearly_positive_nodes() {
    // Lift the margin constant so every node is clearly positive at the warm
    // point: the clamped and floor points must be skipped (NaN) wherever
    // they differ from warm; lower it so none is, and they must be scored.
    // |c0| = 1e5 dominates every node's `sum |rc| + lag` here (at most
    // `sum |d| + 6 * sum_i row_abs_i + 72 < 3e3` on this fixture).
    let depth = 6;
    let (sides, l0, l1) = level(2 * FD_LANES, depth, 71, false);
    let base = problem(150, depth, 71, false);
    let mut high = base.clone();
    high.c0 = 1e5;
    let mut low = base;
    low.c0 = -1e5;
    let (mut skipped, mut scored) = (0usize, 0usize);
    for (p, positive) in [(high, true), (low, false)] {
        let out = run::<f32>(&p, depth, &sides, &l0, &l1, 2);
        let v = out.hint().into_ingest();
        for k in 0..v.count() {
            assert!(v.value(k, FD_WARM).is_finite());
            let any = l0[k * depth..(k + 1) * depth]
                .iter()
                .chain(&l1[k * depth..(k + 1) * depth])
                .any(|&m| m > 0.0);
            if !any {
                continue;
            }
            if positive {
                assert!(v.value(k, FD_WARM) > 0.0);
                assert!(
                    v.value(k, FD_FLOOR).is_nan(),
                    "node {k}: the floor of a clearly positive node was scored"
                );
                skipped += 1;
            } else {
                assert!(
                    v.value(k, FD_FLOOR).is_finite(),
                    "node {k}: the floor of a negative node was skipped"
                );
                scored += 1;
            }
        }
    }
    assert!(
        skipped > 0 && scored > 0,
        "VOID: skipped={skipped} scored={scored}"
    );
}

#[test]
fn fd_fast_ln_tracks_libm() {
    let mut x = 1e-12f32;
    let mut checked = 0usize;
    // x = 1e-12 * 3.7^k up to 1e30: 74 steps (3.7^74 ~ 1.1e42), so the
    // explicit 200-step bound never cuts the sweep short.
    for _ in 0..200 {
        if x >= 1e30 {
            break;
        }
        for f in [1.0f32, 1.1, 1.414, 1.4143, 1.9, 1.999_999] {
            let v = x * f;
            let want = f64::from(v).ln();
            let got = f64::from(fast_ln_f32(v));
            assert!(
                (got - want).abs() <= 1e-5 + 4e-7 * want.abs(),
                "ln({v:e}): {got} vs {want}"
            );
            checked += 1;
        }
        x *= 3.7;
    }
    assert_eq!(fast_ln_f32(f32::INFINITY), f32::INFINITY);
    assert!(checked > 100);
}

/// The line search's `exp` (the children's step size) is ours, not libm's, so
/// the f32 search is the same bits on every host and the device kernels can
/// mirror it: it must still track libm to search-grade accuracy over the whole
/// finite range, map the edges as documented, and be exact where it must be.
#[test]
fn fd_fast_exp_tracks_libm() {
    let mut checked = 0usize;
    let mut worst = 0.0f64;
    for k in 0..12_840u32 {
        let x = -87.3f32 + k as f32 * 0.013_7;
        let got = f64::from(fast_exp_f32(x));
        let want = f64::from(x).exp();
        let rel = (got - want).abs() / want;
        worst = worst.max(rel);
        assert!(rel <= 1e-6, "exp({x:e}): {got:e} vs {want:e} (rel {rel:e})");
        checked += 1;
    }
    for x in [
        -1e-7f32, -1e-30, 0.0, -0.0, 1e-30, 1e-7, 0.5, -0.5, 1.0, 20.0,
    ] {
        let got = f64::from(fast_exp_f32(x));
        let want = f64::from(x).exp();
        assert!((got - want).abs() <= 1e-6 * want, "exp({x:e})");
    }
    assert_eq!(fast_exp_f32(0.0), 1.0, "exp(0) must be exactly 1");
    assert!(fast_exp_f32(f32::NAN).is_nan());
    assert_eq!(fast_exp_f32(f32::INFINITY), f32::INFINITY);
    assert_eq!(fast_exp_f32(f32::NEG_INFINITY), 0.0);
    assert_eq!(fast_exp_f32(89.0), f32::INFINITY);
    assert_eq!(fast_exp_f32(-88.0), 0.0);
    let top = fast_exp_f32(f32::from_bits(0x42B1_7217));
    assert!(
        top.is_finite() && top > 3.4e38,
        "the largest admitted x must stay finite: {top:e}"
    );
    let bottom = fast_exp_f32(f32::from_bits(0xC2AE_AC4F));
    assert!(bottom > 0.0 && bottom < 1.2e-38, "{bottom:e}");
    // Monotone on a fine grid around 1 (the line search compares steps).
    let mut prev = 0.0f32;
    for k in 0..4_096u16 {
        let y = -2.0f32 + f32::from(k) / 1024.0;
        let v = fast_exp_f32(y);
        assert!(v >= prev, "exp is not monotone at {y:e}");
        prev = v;
    }
    eprintln!("[fd-fast-exp] checked={checked} worst_rel={worst:e}");
    assert!(checked > 10_000, "VOID: {checked}");
}

#[test]
fn fd_work_bytes_cover_the_workspace() {
    for (n, depth) in [
        (1usize, 1usize),
        (255, 3),
        (257, 64),
        (11418, 28),
        (3448, 512),
    ] {
        let have = workspace_capacity_bytes::<f32>(n, depth);
        let bound = work_bytes::<f32>(n, depth);
        assert!(
            have <= bound,
            "n={n} depth={depth}: the workspace reserves {have} B > the admitted {bound} B"
        );
        let have64 = workspace_capacity_bytes::<f64>(n, depth);
        assert!(have64 <= work_bytes::<f64>(n, depth));
    }
    // The lowering's own resident size is what it claims.
    let p = problem(300, 9, 81, false);
    let prog = FdProgram::<f32>::lower(&p).expect("lower");
    assert!(prog.bytes() >= (300 * 9 * 4) as u64);
    const { assert!(FD_TILE >= FD_LANES) };
}

/// The integrator's search-only rate probe (release, under `ny_box.sh
/// measure`): tiny-like and cifar-like shapes, 32,768 nodes, 1 and all
/// threads. Prints nodes/s; asserts nothing about speed. An evidence job, so
/// it lives behind the off-by-default `benchmarks` feature rather than
/// `#[ignore]`: `targo --unverified test --release --locked --offline -p ny-bbe
/// --features benchmarks --lib fd_cpu_rate_probe -- --nocapture`.
#[cfg(feature = "benchmarks")]
#[test]
fn fd_cpu_rate_probe() {
    let all = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    for (n, depth, tag) in [(11418usize, 28usize, "tiny-like"), (3448, 45, "cifar-like")] {
        let p = problem(n, depth, 0xFD4, false);
        let count = 32_768;
        let (sides, l0, l1) = level(count, depth, 0xFD4, false);
        for threads in [1usize, all] {
            let nodes = if threads == 1 { 4_096 } else { count };
            let h = upload_fd::<f32>("probe", &p, &BIG, threads).expect("upload");
            let mut out = LevelOut::default();
            let t0 = Instant::now();
            let stats = fd_level::<f32>(
                &h,
                &chunk(
                    depth,
                    &sides[..nodes * depth],
                    &l0[..nodes * depth],
                    &l1[..nodes * depth],
                ),
                &mut out,
            )
            .expect("fd_level")
            .into_hint()
            .into_ingest();
            let secs = t0.elapsed().as_secs_f64();
            eprintln!(
                "[fd-cpu-rate] shape={tag} n={n} depth={depth} threads={threads} nodes={nodes} \
                 secs={secs:.3} nps={:.0} evals_per_node={:.2}",
                nodes as f64 / secs.max(1e-9),
                stats.evals as f64 / nodes as f64
            );
        }
    }
}
