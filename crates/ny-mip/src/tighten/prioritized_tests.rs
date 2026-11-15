// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #relusplitter-lp-refine (wave-6 C7): soundness tests for
//! [`LpTightener::tighten_layer_rigorous_prioritized`].
//!
//! The load-bearing property is CONTAINMENT: every bound the pass returns must
//! enclose the TRUE (real-arithmetic) pre-activation range over the input box,
//! whatever the earlier layers' (sound) boxes were. It is checked against dense
//! float sampling (tolerance `1e-9 * (1 + |v|)` for the sampler's own f64
//! rounding) and, at every box vertex plus each neuron's extreme samples,
//! against an EXACT-RATIONAL forward with no tolerance at all. The remaining
//! tests pin the schedule contract: never widen, a spent deadline or a
//! per-layer cap leaves unreached neurons bit-identical, and only the
//! in-process rigorous backend is accepted.

use super::*;
use crate::config::MipBackend;
use num_rational::BigRational;
use num_traits::Zero;
use ny_tensor::{next_down_f32, next_up_f32};
use std::time::{Duration, Instant};

/// Deterministic xorshift64 (tests only).
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn unit(&mut self) -> f64 {
        #[allow(clippy::cast_precision_loss)]
        let v = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        v
    }

    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }
}

/// A random fully-connected ReLU MLP whose parameters are exactly
/// representable in f32 (as a loaded ONNX network's are).
struct Mlp {
    /// `[input, hidden..., output]`.
    dims: Vec<usize>,
    /// Row-major `[out x in]` per affine layer.
    w: Vec<Vec<f64>>,
    b: Vec<Vec<f64>>,
}

fn f32_grid(v: f64) -> f64 {
    #[allow(clippy::cast_possible_truncation)]
    let f = v as f32;
    f64::from(f)
}

impl Mlp {
    fn random(rng: &mut Rng, dims: &[usize]) -> Self {
        let mut w = Vec::new();
        let mut b = Vec::new();
        for pair in dims.windows(2) {
            let (fan_in, fan_out) = (pair[0], pair[1]);
            w.push(
                (0..fan_in * fan_out)
                    .map(|_| f32_grid(rng.uniform(-1.0, 1.0)))
                    .collect(),
            );
            b.push(
                (0..fan_out)
                    .map(|_| f32_grid(rng.uniform(-0.5, 0.5)))
                    .collect(),
            );
        }
        Self {
            dims: dims.to_vec(),
            w,
            b,
        }
    }

    /// Pre-activations of every affine layer (hidden layers and output), f64.
    fn preacts(&self, x: &[f64]) -> Vec<Vec<f64>> {
        let mut out = Vec::new();
        let mut a = x.to_vec();
        for (k, pair) in self.dims.windows(2).enumerate() {
            let (fan_in, fan_out) = (pair[0], pair[1]);
            let z: Vec<f64> = (0..fan_out)
                .map(|i| {
                    (0..fan_in)
                        .map(|j| self.w[k][i * fan_in + j] * a[j])
                        .sum::<f64>()
                        + self.b[k][i]
                })
                .collect();
            a = z.iter().map(|v| v.max(0.0)).collect();
            out.push(z);
        }
        out
    }

    /// Pre-activations of every affine layer in exact rational arithmetic.
    fn preacts_exact(&self, x: &[f64]) -> Vec<Vec<BigRational>> {
        let q = |v: f64| BigRational::from_float(v).expect("finite test value");
        let mut out = Vec::new();
        let mut a: Vec<BigRational> = x.iter().map(|&v| q(v)).collect();
        for (k, pair) in self.dims.windows(2).enumerate() {
            let (fan_in, fan_out) = (pair[0], pair[1]);
            let z: Vec<BigRational> = (0..fan_out)
                .map(|i| {
                    let mut acc = q(self.b[k][i]);
                    for (j, aj) in a.iter().enumerate().take(fan_in) {
                        acc += q(self.w[k][i * fan_in + j]) * aj;
                    }
                    acc
                })
                .collect();
            a = z
                .iter()
                .map(|v| {
                    if *v > BigRational::zero() {
                        v.clone()
                    } else {
                        BigRational::zero()
                    }
                })
                .collect();
            out.push(z);
        }
        out
    }
}

/// Sound interval bounds for every affine layer's pre-activation: f64 interval
/// arithmetic widened by a generous relative slack, then cast OUTWARD to f32.
fn ibp(mlp: &Mlp, input: &[Bound]) -> Vec<Vec<Bound>> {
    let mut out = Vec::new();
    let mut lo: Vec<f64> = input.iter().map(|b| f64::from(b.lower())).collect();
    let mut hi: Vec<f64> = input.iter().map(|b| f64::from(b.upper())).collect();
    for (k, pair) in mlp.dims.windows(2).enumerate() {
        let (fan_in, fan_out) = (pair[0], pair[1]);
        let mut layer = Vec::with_capacity(fan_out);
        let mut next_lo = Vec::with_capacity(fan_out);
        let mut next_hi = Vec::with_capacity(fan_out);
        for i in 0..fan_out {
            let mut l = mlp.b[k][i];
            let mut u = mlp.b[k][i];
            for j in 0..fan_in {
                let wij = mlp.w[k][i * fan_in + j];
                if wij >= 0.0 {
                    l += wij * lo[j];
                    u += wij * hi[j];
                } else {
                    l += wij * hi[j];
                    u += wij * lo[j];
                }
            }
            let slack = 1e-9 * (1.0 + l.abs().max(u.abs()));
            #[allow(clippy::cast_possible_truncation)]
            let (lf, uf) = (
                next_down_f32((l - slack) as f32),
                next_up_f32((u + slack) as f32),
            );
            layer.push(Bound::new(lf, uf));
            next_lo.push(f64::from(lf).max(0.0));
            next_hi.push(f64::from(uf).max(0.0));
        }
        out.push(layer);
        lo = next_lo;
        hi = next_hi;
    }
    out
}

fn config(deadline: Instant) -> MipConfig {
    MipConfig {
        timeout_secs: 10.0,
        ay_hard_deadline: Some(deadline),
        ..MipConfig::default()
    }
}

fn tightener(mlp: &Mlp, input: &[Bound], interm: &[Vec<Bound>], cfg: MipConfig) -> LpTightener {
    LpTightener::new(
        mlp.w.clone(),
        mlp.b.clone(),
        mlp.dims.clone(),
        input.to_vec(),
        interm.to_vec(),
        cfg,
    )
}

fn box_input(rng: &mut Rng, n: usize) -> Vec<Bound> {
    (0..n)
        .map(|_| {
            #[allow(clippy::cast_possible_truncation)]
            let c = rng.uniform(-1.0, 1.0) as f32;
            #[allow(clippy::cast_possible_truncation)]
            let r = rng.uniform(0.2, 0.8) as f32;
            Bound::new(c - r, c + r)
        })
        .collect()
}

/// Every corner of the input box (n <= 10).
fn vertices(input: &[Bound]) -> Vec<Vec<f64>> {
    let n = input.len();
    (0..(1usize << n))
        .map(|mask| {
            (0..n)
                .map(|j| {
                    if mask & (1 << j) == 0 {
                        f64::from(input[j].lower())
                    } else {
                        f64::from(input[j].upper())
                    }
                })
                .collect()
        })
        .collect()
}

#[test]
fn triangle_gap_priority_ranks_loosest_first_and_skips_stable() {
    let bounds = vec![
        Bound::new(0.0, 3.0),   // stable active: excluded
        Bound::new(-1.0, 1.0),  // gap 0.5
        Bound::new(-4.0, 4.0),  // gap 2.0
        Bound::new(-3.0, -0.5), // stable inactive: excluded
        Bound::new(-2.0, 2.0),  // gap 1.0
        Bound::new(-1.0, 1.0),  // gap 0.5 (tie with index 1: ascending index)
    ];
    assert_eq!(triangle_gap_priority(&bounds), vec![2, 4, 1, 5]);
    assert_eq!(count_unstable(&bounds), 4);
    assert_eq!(
        triangle_gap_priority(&[Bound::new(0.0, 1.0)]),
        [] as [usize; 0]
    );
}

/// The load-bearing test: progressive rigorous tightening of every hidden
/// layer AND the output layer on random MLPs, starting from loose (IBP)
/// boxes, is contained by the true forward — float-sampled and exact-rational
/// — and never wider than what it was given. It must also actually tighten
/// something, or the containment check is vacuous.
#[test]
#[ntest::timeout(120_000)]
fn prioritized_pass_is_contained_by_sampled_and_exact_forward() {
    let mut total_tightened = 0usize;
    let mut total_rigorous = 0usize;
    for seed in 1..=4u64 {
        let mut rng = Rng::new(0x5eed_0000 + seed);
        let mlp = Mlp::random(&mut rng, &[4, 8, 8, 6, 3]);
        let input = box_input(&mut rng, 4);
        let loose = ibp(&mlp, &input);
        let mut refined = loose.clone();
        let deadline = Instant::now() + Duration::from_secs(60);
        for layer in 1..mlp.dims.len() - 1 {
            let t = tightener(&mlp, &input, &refined, config(deadline));
            let (bounds, report) = t
                .tighten_layer_rigorous_prioritized(layer, &refined[layer])
                .expect("rigorous prioritized tightening succeeds on a small MLP");
            assert_eq!(bounds.len(), refined[layer].len());
            for (new, old) in bounds.iter().zip(&refined[layer]) {
                assert!(
                    new.lower() >= old.lower() && new.upper() <= old.upper(),
                    "seed {seed} layer {layer}: widened {old:?} -> {new:?}"
                );
            }
            assert_eq!(report.unstable_before, count_unstable(&refined[layer]));
            assert_eq!(report.unstable_after, count_unstable(&bounds));
            assert!(report.rigorous <= report.lp_solves);
            assert!(!report.deadline_hit, "a 60 s slice is ample for this net");
            total_tightened += report.tightened;
            total_rigorous += report.rigorous;
            refined[layer] = bounds;
        }

        // The retained independent IBP baseline must remain an enclosure
        // after every copied layer has been tightened.
        for (derived, initial) in refined.iter().zip(&loose) {
            assert_eq!(derived.len(), initial.len());
            for (new, old) in derived.iter().zip(initial) {
                assert!(new.lower() >= old.lower() && new.upper() <= old.upper());
            }
        }

        // Float sampling (interior + vertices).
        let mut worst_low: Vec<Vec<(f64, Vec<f64>)>> = refined
            .iter()
            .map(|l| vec![(f64::INFINITY, Vec::new()); l.len()])
            .collect();
        let mut worst_high: Vec<Vec<(f64, Vec<f64>)>> = refined
            .iter()
            .map(|l| vec![(f64::NEG_INFINITY, Vec::new()); l.len()])
            .collect();
        let mut points = vertices(&input);
        for _ in 0..20_000 {
            points.push(
                input
                    .iter()
                    .map(|b| rng.uniform(f64::from(b.lower()), f64::from(b.upper())))
                    .collect(),
            );
        }
        for x in &points {
            for (k, z) in mlp.preacts(x).iter().enumerate() {
                for (i, &v) in z.iter().enumerate() {
                    let tol = 1e-9 * (1.0 + v.abs());
                    assert!(
                        v >= f64::from(refined[k][i].lower()) - tol
                            && v <= f64::from(refined[k][i].upper()) + tol,
                        "seed {seed} layer {k} neuron {i}: sample {v} escapes {:?}",
                        refined[k][i]
                    );
                    if v < worst_low[k][i].0 {
                        worst_low[k][i] = (v, x.clone());
                    }
                    if v > worst_high[k][i].0 {
                        worst_high[k][i] = (v, x.clone());
                    }
                }
            }
        }

        // Exact rational forward: every vertex and every neuron's extreme
        // samples, NO tolerance.
        let mut exact_points = vertices(&input);
        for layer in worst_low.iter().chain(worst_high.iter()) {
            for (_, x) in layer {
                if !x.is_empty() {
                    exact_points.push(x.clone());
                }
            }
        }
        for x in &exact_points {
            for (k, z) in mlp.preacts_exact(x).iter().enumerate() {
                for (i, v) in z.iter().enumerate() {
                    let lo = BigRational::from_float(f64::from(refined[k][i].lower()))
                        .expect("finite bound");
                    let hi = BigRational::from_float(f64::from(refined[k][i].upper()))
                        .expect("finite bound");
                    assert!(
                        &lo <= v && v <= &hi,
                        "seed {seed} layer {k} neuron {i}: exact value escapes {:?}",
                        refined[k][i]
                    );
                }
            }
        }
    }
    assert!(
        total_rigorous > 0 && total_tightened > 0,
        "the pass never produced a rigorous tightening, so containment is vacuous \
         (rigorous={total_rigorous}, tightened={total_tightened})"
    );
}

/// On a one-input net the true range of the output pre-activation is attained
/// at a box end or a hidden breakpoint, so it can be computed EXACTLY. The
/// rigorous LP box must contain that exact range (rational comparison).
#[test]
#[ntest::timeout(30_000)]
fn prioritized_pass_contains_the_exact_one_dimensional_range() {
    // x in [-2, 2]; h = relu(x), relu(-x), relu(x - 1); y = h0 + h1 - 3 h2 - 0.5.
    let mlp = Mlp {
        dims: vec![1, 3, 1],
        w: vec![vec![1.0, -1.0, 1.0], vec![1.0, 1.0, -3.0]],
        b: vec![vec![0.0, 0.0, -1.0], vec![-0.5]],
    };
    let input = vec![Bound::new(-2.0, 2.0)];
    let interm = ibp(&mlp, &input);
    let t = tightener(
        &mlp,
        &input,
        &interm,
        config(Instant::now() + Duration::from_secs(20)),
    );
    // Force the output layer to be "unstable" so the pass runs on it.
    let wide = vec![Bound::new(-100.0, 100.0)];
    let (bounds, report) = t
        .tighten_layer_rigorous_prioritized(1, &wide)
        .expect("rigorous tighten of the output pre-activation");
    assert_eq!(report.attempted, 1);
    assert!(report.rigorous >= 1, "the LP must answer rigorously here");
    // Candidates: box ends and the breakpoints 0 and 1.
    let mut exact_min: Option<BigRational> = None;
    let mut exact_max: Option<BigRational> = None;
    for x in [-2.0, 0.0, 1.0, 2.0] {
        let y = mlp.preacts_exact(&[x])[1][0].clone();
        exact_min = Some(match exact_min {
            Some(m) if m <= y => m,
            _ => y.clone(),
        });
        exact_max = Some(match exact_max {
            Some(m) if m >= y => m,
            _ => y,
        });
    }
    let lo = BigRational::from_float(f64::from(bounds[0].lower())).expect("finite");
    let hi = BigRational::from_float(f64::from(bounds[0].upper())).expect("finite");
    assert!(lo <= exact_min.expect("candidates evaluated"));
    assert!(hi >= exact_max.expect("candidates evaluated"));
    assert!(
        bounds[0].lower() > -100.0 && bounds[0].upper() < 100.0,
        "the LP must actually tighten the deliberately wide box: {:?}",
        bounds[0]
    );
}

/// A spent deadline is a no-op: every bound bit-identical, reported as cut.
#[test]
fn expired_deadline_keeps_every_bound_bit_identical() {
    let mut rng = Rng::new(7);
    let mlp = Mlp::random(&mut rng, &[3, 6, 5, 2]);
    let input = box_input(&mut rng, 3);
    let interm = ibp(&mlp, &input);
    let past = Instant::now()
        .checked_sub(Duration::from_millis(5))
        .expect("representable");
    let t = tightener(&mlp, &input, &interm, config(past));
    let (bounds, report) = t
        .tighten_layer_rigorous_prioritized(1, &interm[1])
        .expect("an expired pass declines cleanly");
    assert_eq!(bounds, interm[1]);
    assert_eq!(report.lp_solves, 0);
    assert_eq!(report.tightened, 0);
    if report.unstable_before > 0 {
        assert!(report.deadline_hit);
    }
    assert_eq!(report.unstable_after, report.unstable_before);
}

/// `max_tighten_per_layer = 1` attempts exactly the loosest neuron; every
/// other neuron keeps its incoming bound bit for bit.
#[test]
#[ntest::timeout(30_000)]
fn per_layer_cap_attempts_only_the_top_priority_neuron() {
    for seed in 11..=20u64 {
        let mut rng = Rng::new(seed);
        let mlp = Mlp::random(&mut rng, &[4, 8, 8, 2]);
        let input = box_input(&mut rng, 4);
        let interm = ibp(&mlp, &input);
        let order = triangle_gap_priority(&interm[1]);
        if order.len() < 2 {
            continue;
        }
        let cfg = MipConfig {
            max_tighten_per_layer: 1,
            ..config(Instant::now() + Duration::from_secs(20))
        };
        let t = tightener(&mlp, &input, &interm, cfg);
        let (bounds, report) = t
            .tighten_layer_rigorous_prioritized(1, &interm[1])
            .expect("capped pass");
        assert_eq!(report.attempted, 1);
        for (i, (new, old)) in bounds.iter().zip(&interm[1]).enumerate() {
            if i != order[0] {
                assert_eq!(new, old, "seed {seed}: unreached neuron {i} changed");
            }
        }
        return;
    }
    panic!("no seed produced two unstable neurons; the cap test is VOID");
}

/// Only the in-process rigorous surface is accepted.
#[test]
fn subprocess_backend_is_refused() {
    let mut rng = Rng::new(3);
    let mlp = Mlp::random(&mut rng, &[2, 4, 2]);
    let input = box_input(&mut rng, 2);
    let interm = ibp(&mlp, &input);
    let cfg = MipConfig {
        backend: MipBackend::AyProc,
        ..config(Instant::now() + Duration::from_secs(5))
    };
    let t = tightener(&mlp, &input, &interm, cfg);
    assert!(t.tighten_layer_rigorous_prioritized(1, &interm[1]).is_err());
}

/// Shape errors are refused rather than silently mis-indexed.
#[test]
fn shape_mismatches_are_refused() {
    let mut rng = Rng::new(5);
    let mlp = Mlp::random(&mut rng, &[2, 4, 2]);
    let input = box_input(&mut rng, 2);
    let interm = ibp(&mlp, &input);
    let t = tightener(
        &mlp,
        &input,
        &interm,
        config(Instant::now() + Duration::from_secs(5)),
    );
    assert!(t.tighten_layer_rigorous_prioritized(1, &interm[0]).is_err());
    assert!(t.tighten_layer_rigorous_prioritized(7, &interm[1]).is_err());
}

/// Review of `5a3a434a1`: a tightener whose prefix is inconsistent must be
/// REFUSED, never encoded. The unboxed builder passes an earlier ReLU through
/// as the IDENTITY when its box is missing (`layer_idx < intermediate_bounds
/// .len()`), and a mis-sized input box makes `weights[k][i * fan_in + j]` read
/// the wrong weights — both are LPs whose feasible set need not contain the
/// true graph. Non-finite boxes would become free columns / NaN triangle rows.
#[test]
fn inconsistent_prefix_is_refused_not_encoded() {
    let mut rng = Rng::new(21);
    let mlp = Mlp::random(&mut rng, &[3, 5, 4, 2]);
    let input = box_input(&mut rng, 3);
    let interm = ibp(&mlp, &input);
    let far = || config(Instant::now() + Duration::from_secs(20));

    // Sanity: the consistent tightener is accepted on the deepest layer.
    let ok = tightener(&mlp, &input, &interm, far());
    assert!(ok.tighten_layer_rigorous_prioritized(2, &interm[2]).is_ok());

    // Layer 1's box missing: layer 2's LP would treat ReLU 1 as the identity.
    let short = tightener(&mlp, &input, &interm[..1], far());
    assert!(short
        .tighten_layer_rigorous_prioritized(2, &interm[2])
        .is_err());

    // Input box narrower than the network input: wrong weight indexing.
    let narrow = tightener(&mlp, &input[..2], &interm, far());
    assert!(narrow
        .tighten_layer_rigorous_prioritized(1, &interm[1])
        .is_err());

    // An earlier layer's box of the wrong width.
    let mut misfit = interm.clone();
    misfit[0].pop();
    let misfit = tightener(&mlp, &input, &misfit, far());
    assert!(misfit
        .tighten_layer_rigorous_prioritized(1, &interm[1])
        .is_err());

    // A non-finite earlier box.
    let mut infinite = interm.clone();
    infinite[0][0] = Bound::new_allow_infinite(f32::NEG_INFINITY, 1.0);
    let infinite = tightener(&mlp, &input, &infinite, far());
    assert!(infinite
        .tighten_layer_rigorous_prioritized(1, &interm[1])
        .is_err());

    // A non-finite target box.
    let mut target = interm[1].clone();
    target[0] = Bound::new_allow_infinite(-1.0, f32::INFINITY);
    assert!(ok.tighten_layer_rigorous_prioritized(1, &target).is_err());
}
