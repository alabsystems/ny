// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `#zono-exact-bn` pins: the certified zonotope core charges BatchNorm with
//! the binary64 enclosure of the EXACT-REAL coefficients that
//! `BatchNormLayer::new` computes (radius ~`1e-15` relative) instead of the
//! binary32 affine's radii (~`6e-8`), and nothing else in ny changes.
//!
//! * `zono_exact_bn_coefficients_enclose_the_exact_real_affine_in_rationals` —
//!   random binary32 statistics over six decades: every channel's binary64
//!   centre +/- radius contains a rational bracket (`2^-100` wide) of the
//!   exact-real `gamma / sqrt(var + eps)` and `beta - mean * scale`; the radii
//!   are within `64 u` of the coefficient; and the binary32 enclosure contains
//!   the same bracket (both certify the same real).
//! * `zono_exact_bn_channel_affine_step_contains_the_exact_real_affine_in_rationals`
//!   — the BatchNorm TRANSFORMER with those coefficients, checked in exact
//!   rationals at every vertex of (symbols, remainder) and both ends of the
//!   exact-real scale bracket (the residual is multilinear, so the vertices
//!   are the worst case: this is a proof, not a sample), plus a mutation pin:
//!   a scale mis-centred by `1e-12` relative — invisible under the old
//!   binary32 charge — must be caught.
//! * `zono_exact_bn_encloses_the_exact_real_affine_at_extreme_binary32_statistics`
//!   — the same coefficient and step checks on a deterministic grid at the
//!   edges of binary32 (subnormal / near-`f32::MAX` statistics, `eps = 0`,
//!   `var = 0`, a binary32 scale underflowed to zero, `|mean| = 1e20`),
//!   straddling `BatchNormLayer::new`'s refusal boundary.
//! * `zono_exact_bn_program_charges_exact_coefficients_and_contains_samples` —
//!   random BN conv stacks: the lane's own observable says every BatchNorm is
//!   `exact_f64`, the stripped twin says `binary32`, the two programs' concrete
//!   semantics agree (channel layout), both contain samples at every depth, and
//!   the exact program's largest certified remainder is orders smaller.
//! * `zono_exact_bn_enclosure_is_ignored_once_the_binary32_fields_drift` — any
//!   one-ulp drift of a public binary32 field, a `from_scale_bias` layer and an
//!   enclosure transplanted between layers all fall back to `binary32`, also at
//!   program level.
//! * `zono_exact_bn_leaves_every_binary32_consumer_bit_identical` — graph IBP,
//!   the per-node IBP map, public CROWN and the layer's own IBP are bit-identical
//!   with and without the enclosure; and a source census proves no code outside
//!   the zonotope core (and the layer that owns it) reads the enclosure.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ndarray::{ArrayD, IxDyn};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::Signed;
use ny_tensor::BoundedTensor;

use super::affine::{apply_channel_affine, BatchNormCoefficients, ChannelAffine};
use super::ids::SymbolId;
use super::kernels::RowPar;
use super::program::{StepOp, ZonoForwardOptions, ZonoProgram};
use super::rounding::UNIT;
use super::state::ZonoState;
use super::tests::{containment, random_child, sample_points, small_cgan_graph, Rng};
use crate::layers::{BatchNormLayer, BoundPropagation, Layer};
use crate::GraphNetwork;

// ---------------------------------------------------------------------------
// Exact-rational reference
// ---------------------------------------------------------------------------

fn rat(v: f64) -> BigRational {
    BigRational::from_float(v).expect("finite binary64 value")
}

fn two() -> BigRational {
    BigRational::from_integer(BigInt::from(2))
}

/// Rational `[a, b]` with `0 < a <= sqrt(w) <= b` and `b - a` at most `2^-100`
/// of the seed bracket: bisection with EXACT squaring, independent of every
/// binary64 rounding the enclosure under test performs.
fn sqrt_bracket(w: &BigRational, approx: f64) -> (BigRational, BigRational) {
    let mut lo = rat(approx * (1.0 - 1e-9));
    let mut hi = rat(approx * (1.0 + 1e-9));
    assert!(
        lo.is_positive() && &lo * &lo <= *w && *w <= &hi * &hi,
        "the seed bracket must contain sqrt(w) (approx {approx})"
    );
    for _ in 0..100 {
        let mid = (&lo + &hi) / two();
        if &mid * &mid <= *w {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo, hi)
}

/// Does `centre +/- radius` contain `[lo, hi]` (exact rationals)?
fn encloses(centre: f64, radius: f64, lo: &BigRational, hi: &BigRational) -> bool {
    let (c, r) = (rat(centre), rat(radius));
    &c - &r <= *lo && *hi <= &c + &r
}

/// Authored binary32 BatchNorm statistics.
struct BnStats {
    gamma: Vec<f32>,
    beta: Vec<f32>,
    mean: Vec<f32>,
    var: Vec<f32>,
    eps: f32,
}

type Bracket = (BigRational, BigRational);

fn channel_vector(values: &[f32]) -> ArrayD<f32> {
    ArrayD::from_shape_vec(IxDyn(&[values.len()]), values.to_vec()).expect("bn vector")
}

impl BnStats {
    fn layer(&self) -> BatchNormLayer {
        BatchNormLayer::new(
            &channel_vector(&self.gamma),
            &channel_vector(&self.beta),
            &channel_vector(&self.mean),
            &channel_vector(&self.var),
            self.eps,
        )
        .expect("BatchNormLayer::new")
    }

    /// Rational brackets of channel `c`'s exact-real `(scale, bias)`.
    fn brackets(&self, c: usize) -> (Bracket, Bracket) {
        let (v, e) = (f64::from(self.var[c]), f64::from(self.eps));
        let w = rat(v) + rat(e);
        let (root_lo, root_hi) = sqrt_bracket(&w, (v + e).sqrt());
        let g = rat(f64::from(self.gamma[c]));
        let (s_a, s_b) = (&g / &root_hi, &g / &root_lo);
        let (s_lo, s_hi) = if s_a <= s_b { (s_a, s_b) } else { (s_b, s_a) };
        let (b, m) = (rat(f64::from(self.beta[c])), rat(f64::from(self.mean[c])));
        // `beta - mean * s` is affine in `s`: its range over the scale bracket
        // is spanned by the bracket's ends.
        let t_a = &b - &m * &s_lo;
        let t_b = &b - &m * &s_hi;
        let (t_lo, t_hi) = if t_a <= t_b { (t_a, t_b) } else { (t_b, t_a) };
        ((s_lo, s_hi), (t_lo, t_hi))
    }
}

/// `10^U(lo, hi)`.
fn log_uniform(rng: &mut Rng, lo: f64, hi: f64) -> f64 {
    10f64.powf(lo + (hi - lo) * rng.unit())
}

/// `+/- 10^U(lo, hi)`, either sign with probability one half.
fn signed(rng: &mut Rng, lo: f64, hi: f64) -> f64 {
    let sign = if rng.unit() < 0.5 { -1.0 } else { 1.0 };
    sign * log_uniform(rng, lo, hi)
}

/// Statistics over six decades of scale, both signs, an exact-zero gamma now
/// and then, and four epsilons (including `0` and the ONNX default).
#[allow(clippy::cast_possible_truncation)]
fn random_stats(rng: &mut Rng, channels: usize, trial: usize) -> BnStats {
    let mut gamma = Vec::with_capacity(channels);
    let mut beta = Vec::with_capacity(channels);
    let mut mean = Vec::with_capacity(channels);
    let mut var = Vec::with_capacity(channels);
    for c in 0..channels {
        gamma.push(if (trial + c).is_multiple_of(23) {
            0.0
        } else {
            signed(rng, -3.0, 2.0) as f32
        });
        beta.push(signed(rng, -2.0, 1.0) as f32);
        mean.push(signed(rng, -2.0, 2.0) as f32);
        var.push(log_uniform(rng, -4.0, 2.0) as f32);
    }
    let eps = [1e-5f32, 1e-3, 1e-12, 0.0][trial % 4];
    BnStats {
        gamma,
        beta,
        mean,
        var,
        eps,
    }
}

// ---------------------------------------------------------------------------
// Coefficients
// ---------------------------------------------------------------------------

#[test]
fn zono_exact_bn_coefficients_enclose_the_exact_real_affine_in_rationals() {
    let mut rng = Rng::new(0xE8AC_7B01);
    let mut checked = 0usize;
    let mut zero_gamma = 0usize;
    let (mut sum_exact, mut sum_binary32) = (0.0f64, 0.0f64);
    for trial in 0..24 {
        let channels = 1 + trial % 7;
        let stats = random_stats(&mut rng, channels, trial);
        let layer = stats.layer();
        let exact = layer
            .exact_affine_channels()
            .expect("a BatchNormLayer::new layer must carry an intact exact enclosure");
        assert_eq!(exact.scale.len(), channels);
        let scale32: Vec<f64> = layer.scale.iter().map(|&v| f64::from(v)).collect();
        let bias32: Vec<f64> = layer.bias.iter().map(|&v| f64::from(v)).collect();
        let scale_err32: Vec<f64> = layer.scale_err.iter().map(|&v| f64::from(v)).collect();
        let bias_err32: Vec<f64> = layer.bias_err.iter().map(|&v| f64::from(v)).collect();
        for c in 0..channels {
            let ((s_lo, s_hi), (t_lo, t_hi)) = stats.brackets(c);
            let (s, t) = (exact.scale[c], exact.bias[c]);
            let (rs, rt) = (exact.scale_err[c], exact.bias_err[c]);
            assert!(
                rs.is_finite() && rs >= 0.0 && rt.is_finite() && rt >= 0.0,
                "trial {trial} channel {c}: radii {rs} / {rt}"
            );
            assert!(
                encloses(s, rs, &s_lo, &s_hi),
                "trial {trial} channel {c}: exact scale {s} +/- {rs} misses the exact-real scale"
            );
            assert!(
                encloses(t, rt, &t_lo, &t_hi),
                "trial {trial} channel {c}: exact bias {t} +/- {rt} misses the exact-real bias"
            );
            // Both representations certify the SAME real.
            assert!(
                encloses(scale32[c], scale_err32[c], &s_lo, &s_hi),
                "trial {trial} channel {c}: binary32 scale enclosure misses the exact-real scale"
            );
            assert!(
                encloses(bias32[c], bias_err32[c], &t_lo, &t_hi),
                "trial {trial} channel {c}: binary32 bias enclosure misses the exact-real bias"
            );
            // Tight: a few binary64 ulps (measured worst ~12 u / ~19 u).
            let bias_scale = f64::from(stats.beta[c]).abs() + (f64::from(stats.mean[c]) * s).abs();
            assert!(
                rs <= 64.0 * UNIT * s.abs() + 1e-300,
                "trial {trial} channel {c}: scale radius {rs} is not ~u relative to {s}"
            );
            assert!(
                rt <= 64.0 * UNIT * bias_scale + 1e-300,
                "trial {trial} channel {c}: bias radius {rt} is not ~u relative to {bias_scale}"
            );
            sum_exact += rs + rt;
            sum_binary32 += scale_err32[c] + bias_err32[c];
            if stats.gamma[c] == 0.0 {
                zero_gamma += 1;
            }
            checked += 1;
        }
    }
    assert!(checked >= 90, "only {checked} channels checked: VOID");
    assert!(
        zero_gamma > 0,
        "precondition: an exact-zero gamma was drawn"
    );
    assert!(
        sum_exact < 1e-4 * sum_binary32,
        "the exact radii ({sum_exact:e} summed) must be orders below the binary32 radii \
         ({sum_binary32:e}): otherwise this path buys nothing"
    );
}

// ---------------------------------------------------------------------------
// The BatchNorm transformer
// ---------------------------------------------------------------------------

/// Exact-rational violations of `apply_channel_affine(state, op)` against
/// the exact-real affine of `stats` (channel of element `j` is
/// `j / per_channel`), over every vertex of (generator symbols, remainder)
/// and both ends of each channel's exact-real scale bracket.
fn rational_violations(
    stats: &BnStats,
    per_channel: usize,
    state: &ZonoState,
    op: &ChannelAffine,
) -> (usize, usize) {
    let n = state.len();
    let k = state.num_generators();
    let out = apply_channel_affine(state, &state.abs_generator_sum(), op).expect("bn step");
    assert_eq!(out.num_generators(), k, "the BN step keeps the symbol set");
    let (mut violations, mut checked) = (0usize, 0usize);
    for j in 0..n {
        let c = j / per_channel;
        let ((s_lo, s_hi), _) = stats.brackets(c);
        let (beta, mean) = (rat(f64::from(stats.beta[c])), rat(f64::from(stats.mean[c])));
        let allowed = rat(out.remainder()[j]);
        for vertex in 0..(1usize << (k + 1)) {
            let e: Vec<BigRational> = (0..k)
                .map(|i| rat(if vertex >> i & 1 == 1 { 1.0 } else { -1.0 }))
                .collect();
            let rho = if vertex >> k & 1 == 1 {
                rat(state.remainder()[j])
            } else {
                -rat(state.remainder()[j])
            };
            let mut x = rat(state.center()[j]) + rho;
            let mut predicted = rat(out.center()[j]);
            for (i, e_i) in e.iter().enumerate() {
                x += rat(state.generator_row(i)[j]) * e_i;
                predicted += rat(out.generator_row(i)[j]) * e_i;
            }
            for s in [&s_lo, &s_hi] {
                // y = s x + (beta - mean s), the exact-real BatchNorm.
                let y = s * (&x - &mean) + &beta;
                if (y - &predicted).abs() > allowed {
                    violations += 1;
                }
                checked += 1;
            }
        }
    }
    (violations, checked)
}

fn exact_op(layer: &BatchNormLayer, per_channel: usize) -> ChannelAffine {
    let exact = layer.exact_affine_channels().expect("exact enclosure");
    let n = exact.scale.len() * per_channel;
    let expand = |v: &[f64]| -> Vec<f64> { (0..n).map(|j| v[j / per_channel]).collect() };
    ChannelAffine {
        scale: expand(&exact.scale),
        shift: expand(&exact.bias),
        scale_err: expand(&exact.scale_err),
        shift_err: expand(&exact.bias_err),
    }
}

#[test]
fn zono_exact_bn_channel_affine_step_contains_the_exact_real_affine_in_rationals() {
    let mut rng = Rng::new(0xE8AC_7B02);
    let mut checked = 0usize;
    for trial in 0..32 {
        let channels = 1 + trial % 4;
        let per_channel = 1 + (trial / 4) % 3;
        let n = channels * per_channel;
        let stats = random_stats(&mut rng, channels, trial);
        let op = exact_op(&stats.layer(), per_channel);
        #[allow(clippy::cast_possible_truncation)]
        let magnitude = 10f64.powi((trial % 5) as i32 - 2);
        let k = 3usize;
        let center: Vec<f64> = (0..n).map(|_| rng.sym() * magnitude).collect();
        let gens: Vec<f64> = (0..k * n).map(|_| rng.sym() * magnitude).collect();
        let rem: Vec<f64> = (0..n).map(|_| rng.unit() * 1e-3 * magnitude).collect();
        let state =
            ZonoState::from_parts(n, center, gens, (0..k as u64).map(SymbolId).collect(), rem)
                .expect("state");
        let (violations, count) = rational_violations(&stats, per_channel, &state, &op);
        assert_eq!(
            violations, 0,
            "trial {trial}: the exact-coefficient BN step misses the exact-real affine"
        );
        checked += count;
    }
    // 32 trials, sum over trials of (channels x per_channel) = 150 elements,
    // each at 2^(3+1) vertices x 2 scale ends.
    assert_eq!(checked, 4_800, "exact checks run: {checked}");

    // MUTATION PIN: mis-centre every scale by 1e-12 relative (about 4500
    // binary64 ulps, four decades below the old binary32 charge, so the old
    // path could never have seen it). The rational check must go red.
    let stats = BnStats {
        gamma: vec![1.3, -0.7],
        beta: vec![0.2, -0.05],
        mean: vec![0.1, 0.3],
        var: vec![0.7, 1.9],
        eps: 1e-5,
    };
    let state = ZonoState::from_parts(
        2,
        vec![0.5, -0.4],
        vec![0.5, 0.6, -0.25, 0.3, 0.75, -0.5],
        vec![SymbolId(0), SymbolId(1), SymbolId(2)],
        vec![1e-3, 2e-3],
    )
    .expect("state");
    let clean = exact_op(&stats.layer(), 1);
    assert_eq!(
        rational_violations(&stats, 1, &state, &clean).0,
        0,
        "positive control: the clean exact step is sound"
    );
    let mut mutated = clean;
    for s in &mut mutated.scale {
        *s *= 1.0 + 1e-12;
    }
    assert!(
        rational_violations(&stats, 1, &state, &mutated).0 > 0,
        "a 1e-12 mis-centred scale must violate the exact-real containment"
    );
}

/// The coefficient enclosure AND the BatchNorm step at the EDGES of binary32,
/// which the moderate-range tests above never reach (their variances stay in
/// `[1e-4, 1e2]`): subnormal and near-`f32::MAX` statistics, `eps = 0`,
/// `var = 0`, a binary32 scale that underflows to (signed) zero while the
/// exact-real one does not, binary32 scales above `1e20`, and a mean of `-1e20`.
/// Every grid point `BatchNormLayer::new` accepts must carry an intact exact
/// enclosure that contains the exact-real `(scale, bias)` in rationals, stays
/// within `64 u`, keeps an underflowed scale non-zero, and whose BatchNorm step
/// contains the exact-real affine at every vertex; a `1e-12` scale mutation
/// must be caught somewhere on the grid.
///
/// Mirrored first, bit for bit, by `scratchpad/vbs/diag14/
/// zono_exactbn_adversarial.py` (main checkout, log beside it): 602 accepted /
/// 118 refused, every category below non-empty, 4816 step checks, 0
/// violations, 352 mutation hits.
#[test]
fn zono_exact_bn_encloses_the_exact_real_affine_at_extreme_binary32_statistics() {
    let min_sub = f32::from_bits(1);
    let gammas = [0.0f32, min_sub, -1.0e-38, 0.75, -3.0e3, 3.0e38];
    let variances = [0.0f32, min_sub, 0.5, 1.0e30, 3.0e38];
    let epsilons = [0.0f32, min_sub, 1.0e-12, 1.0e20];
    let means = [0.0f32, -1.0e-30, -1.0e20];
    let betas = [1.5f32, -1.0e20];
    let state = ZonoState::from_parts(
        1,
        vec![0.7],
        vec![0.3, -0.2],
        vec![SymbolId(0), SymbolId(1)],
        vec![1e-3],
    )
    .expect("state");
    let (mut accepted, mut refused) = (0usize, 0usize);
    let (mut step_checks, mut mutation_hits) = (0usize, 0usize);
    let (mut zero_gamma, mut scale_underflow, mut huge_scale) = (0usize, 0usize, 0usize);
    let (mut eps_zero, mut var_zero, mut subnormal) = (0usize, 0usize, 0usize);
    for &gamma in &gammas {
        for &var in &variances {
            for &eps in &epsilons {
                for &mean in &means {
                    for &beta in &betas {
                        let stats = BnStats {
                            gamma: vec![gamma],
                            beta: vec![beta],
                            mean: vec![mean],
                            var: vec![var],
                            eps,
                        };
                        let what = format!(
                            "gamma {gamma:e} var {var:e} eps {eps:e} mean {mean:e} beta {beta:e}"
                        );
                        let Ok(layer) = BatchNormLayer::new(
                            &channel_vector(&stats.gamma),
                            &channel_vector(&stats.beta),
                            &channel_vector(&stats.mean),
                            &channel_vector(&stats.var),
                            eps,
                        ) else {
                            refused += 1;
                            continue;
                        };
                        accepted += 1;
                        let exact = layer.exact_affine_channels().unwrap_or_else(|| {
                            panic!("{what}: a new() layer must carry an intact exact enclosure")
                        });
                        let ((s_lo, s_hi), (t_lo, t_hi)) = stats.brackets(0);
                        let (s, t) = (exact.scale[0], exact.bias[0]);
                        let (rs, rt) = (exact.scale_err[0], exact.bias_err[0]);
                        assert!(
                            encloses(s, rs, &s_lo, &s_hi),
                            "{what}: exact scale {s:e} +/- {rs:e} misses the exact-real scale"
                        );
                        assert!(
                            encloses(t, rt, &t_lo, &t_hi),
                            "{what}: exact bias {t:e} +/- {rt:e} misses the exact-real bias"
                        );
                        assert!(
                            rs <= 64.0 * UNIT * s.abs() + 1e-300,
                            "{what}: scale radius {rs:e} is not ~u relative to {s:e}"
                        );
                        let bias_scale = f64::from(beta).abs() + (f64::from(mean) * s).abs();
                        assert!(
                            rt <= 64.0 * UNIT * bias_scale + 1e-300,
                            "{what}: bias radius {rt:e} is not ~u relative to {bias_scale:e}"
                        );
                        let clean = exact_op(&layer, 1);
                        let (violations, count) = rational_violations(&stats, 1, &state, &clean);
                        assert_eq!(
                            violations, 0,
                            "{what}: the exact-coefficient BN step misses the exact-real affine"
                        );
                        step_checks += count;
                        let mut mutated = clean;
                        for value in &mut mutated.scale {
                            *value *= 1.0 + 1e-12;
                        }
                        mutation_hits += rational_violations(&stats, 1, &state, &mutated).0;

                        let scale32 = layer.scale[[0]];
                        if gamma == 0.0 {
                            zero_gamma += 1;
                        } else if scale32 == 0.0 {
                            assert!(
                                s != 0.0,
                                "{what}: the exact path lost a scale binary32 underflowed"
                            );
                            scale_underflow += 1;
                        }
                        if scale32.abs() > 1.0e20 {
                            huge_scale += 1;
                        }
                        if eps == 0.0 {
                            eps_zero += 1;
                        }
                        if var == 0.0 {
                            var_zero += 1;
                        }
                        if [gamma, var, eps].iter().any(|value| value.is_subnormal()) {
                            subnormal += 1;
                        }
                    }
                }
            }
        }
    }
    assert!(
        accepted >= 500 && refused > 0,
        "the grid must straddle new()'s refusal boundary: {accepted} accepted, {refused} refused"
    );
    assert_eq!(
        step_checks,
        accepted * 16,
        "every accepted point's step checked"
    );
    for (category, count) in [
        ("zero gamma", zero_gamma),
        ("binary32 scale underflow", scale_underflow),
        ("binary32 scale above 1e20", huge_scale),
        ("eps = 0", eps_zero),
        ("var = 0", var_zero),
        ("subnormal statistic", subnormal),
    ] {
        assert!(
            count > 0,
            "precondition: no accepted `{category}` point: VOID"
        );
    }
    assert!(
        mutation_hits > 0,
        "a 1e-12 mis-centred scale must be caught somewhere on the edge grid"
    );
}

// ---------------------------------------------------------------------------
// Programs
// ---------------------------------------------------------------------------

/// `graph` with every BatchNorm's exact enclosure removed (the representation
/// every layer had before #zono-exact-bn), and how many were removed.
fn strip_exact(mut graph: GraphNetwork) -> (GraphNetwork, usize) {
    let mut stripped = 0usize;
    let names: Vec<String> = graph.nodes.keys().cloned().collect();
    for name in names {
        let node = graph.nodes.get_mut(&name).expect("node");
        if let Layer::BatchNorm(bn) = &mut node.layer {
            if bn.exact_affine.take().is_some() {
                stripped += 1;
            }
        }
    }
    (graph, stripped)
}

fn cgan_root_box() -> BoundedTensor {
    let lo = [-0.6f32, -0.4, 0.1, -0.9, 0.3];
    let hi = [0.5f32, 0.35, 0.8, -0.2, 0.9];
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[5]), lo.to_vec()).expect("lo"),
        ArrayD::from_shape_vec(IxDyn(&[5]), hi.to_vec()).expect("hi"),
    )
    .expect("root box")
}

fn sources(program: &ZonoProgram) -> Vec<BatchNormCoefficients> {
    program
        .batch_norm_coefficients()
        .iter()
        .map(|(_, source)| *source)
        .collect()
}

#[test]
fn zono_exact_bn_program_charges_exact_coefficients_and_contains_samples() {
    let opts = ZonoForwardOptions {
        max_generators: 16_384,
        record_trace: false,
        row_par: RowPar::Seq,
    };
    let mut compared_steps = 0usize;
    for (seed, convt_pad) in [
        (0xB0_01u64, 0usize),
        (0xB0_02, 1),
        (0xB0_03, 0),
        (0xB0_04, 1),
    ] {
        let graph = small_cgan_graph(seed, convt_pad);
        let (stripped, removed) = strip_exact(small_cgan_graph(seed, convt_pad));
        assert_eq!(removed, 3, "precondition: three BatchNorms built by new()");
        let root = cgan_root_box();
        let exact = ZonoProgram::compile_with_ibp_shapes(&graph, &root).expect("compile exact");
        let binary32 =
            ZonoProgram::compile_with_ibp_shapes(&stripped, &root).expect("compile binary32");
        // The lane's own observable.
        assert_eq!(sources(&exact), vec![BatchNormCoefficients::ExactF64; 3]);
        assert_eq!(sources(&binary32), vec![BatchNormCoefficients::Binary32; 3]);
        assert_eq!(BatchNormCoefficients::ExactF64.token(), "exact_f64");

        // Step by step: same real, far tighter radius.
        for (e_step, b_step) in exact.steps().iter().zip(binary32.steps()) {
            let (StepOp::ChannelAffine(e), StepOp::ChannelAffine(b)) = (&e_step.op, &b_step.op)
            else {
                continue;
            };
            for j in 0..e.scale.len() {
                assert!(
                    (e.scale[j] - b.scale[j]).abs() <= (e.scale_err[j] + b.scale_err[j]) * 1.01,
                    "`{}`[{j}]: the two scale enclosures are disjoint",
                    e_step.node
                );
                assert!(
                    (e.shift[j] - b.shift[j]).abs() <= (e.shift_err[j] + b.shift_err[j]) * 1.01,
                    "`{}`[{j}]: the two shift enclosures are disjoint",
                    e_step.node
                );
                assert!(e.scale_err[j] <= 64.0 * UNIT * e.scale[j].abs() + 1e-300);
            }
            compared_steps += 1;
        }

        // Concrete semantics agree to the binary32 rounding of the
        // coefficients: a channel-layout mistake would be O(1).
        let (points, rows) = sample_points(&root, 64, seed);
        let mut exact_values: Vec<Vec<f64>> = Vec::new();
        exact
            .evaluate_points(&points, rows, |_, values| {
                exact_values.push(values.to_vec())
            })
            .expect("evaluate exact");
        let mut index = 0usize;
        binary32
            .evaluate_points(&points, rows, |step, values| {
                let mine = &exact_values[index];
                index += 1;
                let scale = values.iter().fold(1.0f64, |m, v| m.max(v.abs()));
                for (j, (a, b)) in mine.iter().zip(values).enumerate() {
                    assert!(
                        (a - b).abs() <= 1e-5 * scale,
                        "step {step:?}[{j}]: exact-BN program {a} vs binary32 program {b}"
                    );
                }
            })
            .expect("evaluate binary32");
        assert_eq!(index, exact_values.len());

        let mut rng = Rng::new(seed ^ 0x5A5A);
        for depth in [0usize, 3, 6, 9] {
            let child = random_child(&root, depth, &mut rng);
            let (report_e, outcome_e) =
                containment(&exact, &child, &opts, &[], 1_500, depth as u64 + seed);
            let (report_b, outcome_b) =
                containment(&binary32, &child, &opts, &[], 1_500, depth as u64 + seed);
            assert_eq!(
                report_e.box_violations + report_e.form_violations,
                0,
                "seed {seed:#x} depth {depth}: exact-BN program {report_e:?}"
            );
            assert_eq!(
                report_b.box_violations + report_b.form_violations,
                0,
                "seed {seed:#x} depth {depth}: binary32 program {report_b:?}"
            );
            assert!(
                outcome_e.stats.max_remainder * 100.0 < outcome_b.stats.max_remainder,
                "seed {seed:#x} depth {depth}: exact-BN remainder {:e} is not orders below \
                 the binary32 one {:e}",
                outcome_e.stats.max_remainder,
                outcome_b.stats.max_remainder
            );
        }
    }
    assert_eq!(compared_steps, 12, "every BatchNorm step compared");
}

// ---------------------------------------------------------------------------
// Fail-closed fallback
// ---------------------------------------------------------------------------

#[test]
fn zono_exact_bn_enclosure_is_ignored_once_the_binary32_fields_drift() {
    let mut rng = Rng::new(0xE8AC_7B05);
    let stats = random_stats(&mut rng, 5, 1);
    let layer = stats.layer();
    assert!(
        layer.exact_affine_channels().is_some(),
        "positive control: a fresh new() layer is exact"
    );
    let bump = |v: &mut f32| *v = f32::from_bits(v.to_bits() + 1);
    for field in 0..4 {
        let mut drifted = layer.clone();
        let target = match field {
            0 => &mut drifted.scale,
            1 => &mut drifted.bias,
            2 => &mut drifted.scale_err,
            _ => &mut drifted.bias_err,
        };
        bump(target.iter_mut().nth(2).expect("channel 2"));
        assert!(
            drifted.exact_affine_channels().is_none(),
            "field {field}: a one-ulp drift must disable the exact enclosure"
        );
    }
    // from_scale_bias carries none.
    let plain = BatchNormLayer::from_scale_bias(layer.scale.clone(), layer.bias.clone())
        .expect("from_scale_bias");
    assert!(plain.exact_affine.is_none() && plain.exact_affine_channels().is_none());
    // An enclosure transplanted onto a layer it was not built beside.
    let mut transplanted = random_stats(&mut rng, 5, 2).layer();
    transplanted.exact_affine.clone_from(&layer.exact_affine);
    assert!(
        transplanted.exact_affine_channels().is_none(),
        "a transplanted enclosure must be ignored"
    );
    let mut narrower = random_stats(&mut rng, 3, 3).layer();
    narrower.exact_affine.clone_from(&layer.exact_affine);
    assert!(
        narrower.exact_affine_channels().is_none(),
        "an enclosure of another channel count must be ignored"
    );

    // Program level: drift ONE BatchNorm of the synthetic cGAN; only that
    // step falls back, and the program still contains samples.
    let mut graph = small_cgan_graph(0xB0_05, 0);
    let Layer::BatchNorm(bn2) = &mut graph.nodes.get_mut("bn2").expect("bn2 exists").layer else {
        panic!("precondition: bn2 is a BatchNorm");
    };
    bump(bn2.scale_err.iter_mut().next().expect("channel 0"));
    let root = cgan_root_box();
    let program = ZonoProgram::compile_with_ibp_shapes(&graph, &root).expect("compile");
    let observed: HashMap<String, BatchNormCoefficients> =
        program.batch_norm_coefficients().iter().cloned().collect();
    assert_eq!(observed.len(), 3);
    assert_eq!(observed["bn1"], BatchNormCoefficients::ExactF64);
    assert_eq!(observed["bn2"], BatchNormCoefficients::Binary32);
    assert_eq!(observed["bn3"], BatchNormCoefficients::ExactF64);
    let opts = ZonoForwardOptions {
        max_generators: 16_384,
        record_trace: false,
        row_par: RowPar::Seq,
    };
    let (report, _) = containment(&program, &root, &opts, &[], 1_000, 7);
    assert_eq!(
        report.box_violations + report.form_violations,
        0,
        "{report:?}"
    );
}

// ---------------------------------------------------------------------------
// Every other consumer is untouched
// ---------------------------------------------------------------------------

fn assert_bits_eq(a: &BoundedTensor, b: &BoundedTensor, what: &str) {
    assert_eq!(a.shape(), b.shape(), "{what}: shape");
    for (side, (x, y)) in [
        ("lower", (a.lower(), b.lower())),
        ("upper", (a.upper(), b.upper())),
    ] {
        for (j, (p, q)) in x.iter().zip(y.iter()).enumerate() {
            assert_eq!(
                p.to_bits(),
                q.to_bits(),
                "{what}: {side}[{j}] differs ({p} vs {q}) with vs without the exact enclosure"
            );
        }
    }
}

/// Every `.rs` file under `dir`, recursively (symlinks not followed).
fn rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries =
        std::fs::read_dir(dir).unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display()));
    for entry in entries {
        let entry = entry.expect("dir entry");
        let kind = entry.file_type().expect("file type");
        let path = entry.path();
        if kind.is_dir() {
            if path.file_name().is_some_and(|name| name == "target") {
                continue;
            }
            rust_sources(&path, out);
        } else if kind.is_file() && path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}

#[test]
fn zono_exact_bn_leaves_every_binary32_consumer_bit_identical() {
    let mut compared = 0usize;
    let mut public_crown = 0usize;
    for (seed, convt_pad) in [(0xB0_11u64, 0usize), (0xB0_12, 1)] {
        let graph = small_cgan_graph(seed, convt_pad);
        let (stripped, removed) = strip_exact(small_cgan_graph(seed, convt_pad));
        assert_eq!(
            removed, 3,
            "precondition: the enclosures exist to be stripped"
        );
        let root = cgan_root_box();
        let mut rng = Rng::new(seed);
        for depth in [0usize, 5] {
            let domain = random_child(&root, depth, &mut rng);
            let what = format!("seed {seed:#x} depth {depth}");
            // Whole-graph IBP.
            assert_bits_eq(
                &graph.propagate_ibp(&domain).expect("ibp"),
                &stripped.propagate_ibp(&domain).expect("ibp"),
                &format!("{what} graph IBP"),
            );
            // The per-node IBP map (every BatchNorm output included).
            let map_a = graph
                .collect_node_bounds_with_engine(&domain, None)
                .expect("node map");
            let map_b = stripped
                .collect_node_bounds_with_engine(&domain, None)
                .expect("node map");
            assert_eq!(map_a.len(), map_b.len());
            for (name, entry) in &map_a {
                assert_bits_eq(entry, &map_b[name], &format!("{what} node `{name}`"));
            }
            // The per-node CROWN-IBP collection (BatchNorm's CROWN backward).
            let crown_a = graph
                .collect_crown_ibp_bounds_dag_with_status_and_engine(&domain, None)
                .expect("crown-ibp collection");
            let crown_b = stripped
                .collect_crown_ibp_bounds_dag_with_status_and_engine(&domain, None)
                .expect("crown-ibp collection");
            assert_eq!(crown_a.bounds.len(), crown_b.bounds.len());
            assert_eq!(crown_a.provenance, crown_b.provenance);
            for (name, entry) in &crown_a.bounds {
                assert_bits_eq(
                    entry,
                    &crown_b.bounds[name],
                    &format!("{what} crown-ibp `{name}`"),
                );
            }
            // Public CROWN (alpha-CROWN, with its CROWN fallback): identical
            // bounds, or the identical refusal.
            match (
                graph.propagate_crown(&domain),
                stripped.propagate_crown(&domain),
            ) {
                (Ok(a), Ok(b)) => {
                    assert_bits_eq(&a, &b, &format!("{what} public CROWN"));
                    public_crown += 1;
                }
                (Err(a), Err(b)) => assert_eq!(a.to_string(), b.to_string(), "{what}"),
                (a, b) => panic!(
                    "{what}: public CROWN diverged: with {:?} / without {:?}",
                    a.map(|_| ()),
                    b.map(|_| ())
                ),
            }
            compared += 1;
        }
        // The layer's own IBP on a random box.
        for (name, node) in graph.nodes.iter() {
            let Layer::BatchNorm(bn) = &node.layer else {
                continue;
            };
            let Layer::BatchNorm(bare) = &stripped.nodes[name.as_str()].layer else {
                panic!("`{name}` is not a BatchNorm in the stripped twin");
            };
            assert!(bn.exact_affine.is_some() && bare.exact_affine.is_none());
            #[allow(clippy::cast_possible_truncation)]
            let lo: Vec<f32> = (0..bn.num_channels)
                .map(|_| (rng.sym() - 0.5) as f32)
                .collect();
            let hi: Vec<f32> = lo.iter().map(|&v| v + 0.75).collect();
            let input = BoundedTensor::new(
                ArrayD::from_shape_vec(IxDyn(&[bn.num_channels]), lo).expect("lo"),
                ArrayD::from_shape_vec(IxDyn(&[bn.num_channels]), hi).expect("hi"),
            )
            .expect("bn box");
            assert_bits_eq(
                &bn.propagate_ibp(&input).expect("bn ibp"),
                &bare.propagate_ibp(&input).expect("bn ibp"),
                &format!("layer `{name}` IBP"),
            );
            compared += 1;
        }
    }
    assert_eq!(compared, 4 + 6, "every consumer comparison ran");
    // Informational: how many public-CROWN comparisons produced bounds (a
    // matching refusal on this synthetic graph is still a bit-identity fact).
    eprintln!("[zono-exact-bn] public_crown_bounds_compared={public_crown}/4");

    // SOURCE CENSUS: only the layer that owns the enclosure and the zonotope
    // core read it. A new reader elsewhere must be added here deliberately.
    let crates_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let mut files = Vec::new();
    rust_sources(&crates_dir, &mut files);
    assert!(
        files.len() > 1_000,
        "the census saw only {} files: VOID",
        files.len()
    );
    let allowed = [
        "ny-propagate/src/layers/normalization/batch_norm/types.rs",
        "ny-propagate/src/layers/normalization/batch_norm/math.rs",
        "ny-propagate/src/zono_core/program.rs",
        "ny-propagate/src/zono_core/dag.rs",
        "ny-propagate/src/zono_core/exact_bn_tests.rs",
        "ny-propagate/src/zono_core/vit_root_tests.rs",
    ];
    let readers = [
        ".exact_affine",
        "exact_affine_channels",
        "expanded_exact_affine_parameters",
    ];
    let mut hits: HashMap<&str, usize> = HashMap::new();
    for path in &files {
        let text = String::from_utf8_lossy(&std::fs::read(path).expect("read source")).into_owned();
        if !readers.iter().any(|token| text.contains(token)) {
            continue;
        }
        let normalized = path.to_string_lossy().replace('\\', "/");
        let owner = allowed
            .iter()
            .find(|suffix| normalized.ends_with(*suffix))
            .unwrap_or_else(|| {
                panic!(
                    "{} reads the exact BatchNorm enclosure; only the zonotope core may \
                     (bit-identity of every other lane depends on it)",
                    path.display()
                )
            });
        *hits.entry(*owner).or_insert(0) += 1;
    }
    for required in [
        "ny-propagate/src/zono_core/program.rs",
        "ny-propagate/src/zono_core/dag.rs",
    ] {
        assert!(
            hits.contains_key(required),
            "positive control: the census did not see the reader in {required}: VOID"
        );
    }
}
