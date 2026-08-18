// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! STRICT, ZERO-TOLERANCE soundness tests for the linear CROWN-backward `A·W`
//! coefficient product (#vnncomp-aw-soundness).
//!
//! These tests deliberately do NOT use `FP_TOLERANCE` (the 1e-5 slack in the
//! other soundness proptests MASKS this exact bug — a ~10-ULP cancellation error
//! in the f32 `A·W` GEMM). Instead they assert the f32 CROWN concretized lower
//! bound against an EXACT-RATIONAL CROWN oracle (`ny-cert`'s `DeepReluProblem`)
//! and against the EXACT network value at a witness corner, with NO epsilon.
//!
//! The `A·W` rounding is REPRODUCED by the corner-flip pair
//! (`isolated_aw_corner_flip_zero_tolerance`, `batched_aw_corner_flip_zero_tolerance`)
//! and by the dyadic-oracle proptests: they drive a net whose exact composed input
//! coefficient is 0 with NO relaxation cushion, so the ~1.5e-8 f32 `A·W` error is
//! the entire bound. On the UNPATCHED f32 `A·W` path those FAIL (`L ≈ +1.49e-8 > 0`,
//! a false VERIFIED); the sound f64 `A·W` + certified-error fix restores `L <= 0`.
//!
//! MEMORY's "front3" net is carried here only as an END-TO-END verdict check: its
//! dyadic reconstruction rounds nowhere in `A·W` (see `front3_net`), so it pins the
//! verdict on a ReLU net with a real violation, not the coefficient error.

use crate::layers::common::BoundPropagation;
use crate::layers::Layer;
use crate::network::{GraphNetwork, Network};
use crate::{LinearBounds, LinearLayer, ReLULayer};
use ndarray::{Array1, Array2};
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

use ny_cert::crown_deep::DeepReluProblem;
use ny_cert::rational::Rat;

// =============================================================================
// f32  <->  exact Rat helpers
// =============================================================================

/// Exact conversion of a finite `f32` to a rational `Rat`.
///
/// Every finite f32 is `(-1)^s · m · 2^e` with integer mantissa `m` and integer
/// exponent `e`; we build that rational exactly (no rounding). Used to compare
/// the f32 concretized bound against the exact-rational oracle with zero slack.
fn f32_to_rat(x: f32) -> Rat {
    Rat::from_f32_exact(x).unwrap_or_else(|| panic!("f32_to_rat: non-finite {x}"))
}

/// Dyadic rational `k / 2^m`, exactly representable as both `Rat` and `f32`
/// (for small `k`, `m`), so the ONLY error in the test net is the `A·W` sum.
fn dyadic(k: i128, m: u32) -> Rat {
    Rat::new(k, 1i128 << m).unwrap()
}

fn rat_to_f32(r: Rat) -> f32 {
    use num_traits::ToPrimitive;
    r.to_big().to_f64().unwrap() as f32
}

// =============================================================================
// Build matching ny-propagate Network <-> ny-cert DeepReluProblem
// =============================================================================

/// A small dyadic Linear+ReLU net described by exact rationals, convertible to
/// BOTH an ny-propagate `Network` (f32) and an ny-cert `DeepReluProblem` (Rat).
#[derive(Debug, Clone)]
struct DyadicNet {
    /// hidden weight matrices W^(L): weights[L][out][in]
    weights: Vec<Vec<Vec<Rat>>>,
    /// hidden biases b^(L): biases[L][out]
    biases: Vec<Vec<Rat>>,
    /// scalar read-out weight (length = last hidden width)
    out_weight: Vec<Rat>,
    out_bias: Rat,
    input_lower: Vec<Rat>,
    input_upper: Vec<Rat>,
}

impl DyadicNet {
    fn to_deep_problem(&self) -> DeepReluProblem {
        DeepReluProblem {
            weights: self.weights.clone(),
            biases: self.biases.clone(),
            out_weight: self.out_weight.clone(),
            out_bias: self.out_bias,
            input_lower: self.input_lower.clone(),
            input_upper: self.input_upper.clone(),
            alpha: None, // adaptive default — matches ny-propagate's relu heuristic
            interm_round: false,
        }
    }

    fn to_network(&self) -> Network {
        let mut net = Network::new();
        let k = self.weights.len();
        for l in 0..k {
            let rows = self.weights[l].len();
            let cols = self.weights[l][0].len();
            let w = Array2::from_shape_fn((rows, cols), |(i, j)| rat_to_f32(self.weights[l][i][j]));
            let b = Array1::from_shape_fn(rows, |i| rat_to_f32(self.biases[l][i]));
            net.add_layer(Layer::Linear(LinearLayer::new(w, Some(b)).unwrap()));
            net.add_layer(Layer::ReLU(ReLULayer));
        }
        // read-out linear (scalar output)
        let cols = self.out_weight.len();
        let w = Array2::from_shape_fn((1, cols), |(_, j)| rat_to_f32(self.out_weight[j]));
        let b = Array1::from_elem(1, rat_to_f32(self.out_bias));
        net.add_layer(Layer::Linear(LinearLayer::new(w, Some(b)).unwrap()));
        net
    }

    fn input_box(&self) -> BoundedTensor {
        let n = self.input_lower.len();
        let lo = Array1::from_shape_fn(n, |i| rat_to_f32(self.input_lower[i])).into_dyn();
        let hi = Array1::from_shape_fn(n, |i| rat_to_f32(self.input_upper[i])).into_dyn();
        BoundedTensor::new(lo, hi).unwrap()
    }
}

/// Run the DEFAULT f32 CROWN verdict path
/// (`Network::propagate_crown_with_engine_and_deadline -> concretize_sound`) and
/// return the concretized scalar lower bound `L_f32`.
fn f32_crown_lower(net: &Network, input: &BoundedTensor) -> f32 {
    let out = net
        .propagate_crown_with_engine_and_deadline(input, None, None)
        .expect("CROWN propagation");
    let flat = out.flatten();
    flat.lower()[[0]]
}

/// Run the BATCHED f32 CROWN verdict path — the path used by β-CROWN/BaB for
/// hard instances — and return the concretized scalar lower bound `L_batched`.
///
/// This converts the same sequential `Network` to a `GraphNetwork` (linear
/// chain) and calls `GraphNetwork::propagate_crown_batched`, which dispatches
/// the N-D batched linear backward (`compute_batched_linear_coefficients_cpu`,
/// the `aw_f64_with_abssum` + certified-error path) and concretizes via
/// `BatchedLinearBounds::concretize_sound` (which applies `apply_coeff_err_penalty`).
/// If the batched path silently dropped the certified `A·W` error, this lower
/// bound would be ~1 ULP OPTIMISTIC and could exceed the true minimum.
fn batched_crown_lower(net: &Network, input: &BoundedTensor) -> f32 {
    let graph = GraphNetwork::from_sequential(net).expect("sequential -> graph conversion");
    let out = graph
        .propagate_crown_batched(input)
        .expect("batched CROWN propagation");
    let flat = out.flatten();
    flat.lower()[[0]]
}

// =============================================================================
// FRONT3 end-to-end verdict check, and the `A·W` corner-flip reproduce-then-catch
// =============================================================================

/// A 2 -> 2 -> 1 ReLU net in the shape of MEMORY's front3 false-VERIFIED case:
/// the hidden layer maps x = (x1, x2) to z1 = x1 + x2 - 1/2 and z2 = x1 - x2 - 1/2
/// (both breakpoints cross zero inside `[0,1]^2`), read out through near-cancelling
/// weights ±3/2 plus a -2^-21 bias that drags the true minimum just below zero.
/// The minimum -1/2097152 is attained at the corners (0,0) and (1,0); (1,1) is the
/// MAXIMUM, 9/4 - 2^-21.
///
/// front3's original read-out slopes 5/6 and 7/10 are not dyadic — they are what
/// rounds DOWN in f32 — but no dyadic substitute reproduces that, so this net's
/// weights (±1, ±1/2, ±3/2) are all exact in f32 and its 2-wide `A·W` contraction
/// rounds nowhere: the concretized bound is bit-identical with and without the
/// certified `A·W` error. What this net pins is the END-TO-END verdict — a ReLU net
/// with a real violation must never be reported `y >= 0` — over a CROWN relaxation
/// gap far wider than any coefficient error. The `A·W` error itself is reproduced
/// by the ReLU-free corner-flip tests below, where nothing cushions it.
fn front3_net() -> DyadicNet {
    // Hidden W1 (2x2), b1 (2): z1 = x1 + x2 - 1/2, z2 = x1 - x2 - 1/2.
    let w1 = vec![
        vec![dyadic(1, 0), dyadic(1, 0)],  // z1 = x1 + x2 + b
        vec![dyadic(1, 0), dyadic(-1, 0)], // z2 = x1 - x2 + b
    ];
    let b1 = vec![dyadic(-1, 1), dyadic(-1, 1)]; // -1/2 each (crossing inside [0,1]^2)
    let out_weight = vec![dyadic(3, 1), dyadic(-3, 1)]; // 3/2, -3/2 (cancel)
    let out_bias = dyadic(-1, 21); // -1/2097152 nudge into the negative regime
    DyadicNet {
        weights: vec![w1],
        biases: vec![b1],
        out_weight,
        out_bias,
        input_lower: vec![dyadic(0, 0), dyadic(0, 0)],
        input_upper: vec![dyadic(1, 0), dyadic(1, 0)],
    }
}

/// ZERO-TOLERANCE end-to-end verdict: the scalar CROWN path must not report
/// `y >= 0` on a net whose exact minimum is negative.
#[test]
fn front3_no_false_verified_zero_tolerance() {
    let net = front3_net();
    let problem = net.to_deep_problem();
    let nn = net.to_network();
    let input = net.input_box();

    // f32 CROWN verdict-path lower bound.
    let l_f32 = f32_crown_lower(&nn, &input);
    let l_f32_rat = f32_to_rat(l_f32);

    // EXACT network value at every box corner: the f32 lower bound must be <=
    // the true value at EVERY corner (else it is an unsound over-claim).
    let n = problem.input_lower.len();
    let mut true_min = problem
        .eval(&[problem.input_lower[0], problem.input_lower[1]])
        .unwrap();
    for mask in 0u32..(1u32 << n) {
        let x: Vec<Rat> = (0..n)
            .map(|d| {
                if mask & (1 << d) != 0 {
                    problem.input_upper[d]
                } else {
                    problem.input_lower[d]
                }
            })
            .collect();
        let y = problem.eval(&x).unwrap();
        if y < true_min {
            true_min = y;
        }
    }

    // ZERO-TOLERANCE soundness: L_f32 must never exceed the true minimum.
    assert!(
        l_f32_rat <= true_min,
        "UNSOUND front3: f32 CROWN lower bound {l_f32} ({}) > true network min ({}) — a sound \
         lower bound must sit at or below the exact minimum over the box.",
        l_f32_rat.to_clean_string().unwrap_or_default(),
        true_min.to_clean_string().unwrap_or_default(),
    );

    // And specifically: the true min is strictly negative (a real violation of
    // `y >= 0`), so a sound verifier must NOT report `L_f32 >= 0`.
    assert!(
        true_min.is_negative(),
        "test net must have a true violation (min < 0)"
    );
    assert!(
        l_f32 <= 0.0,
        "f32 CROWN must NOT falsely verify y >= 0 (got L_f32 = {l_f32})"
    );
}

/// ISOLATED `A·W` corner-flip regression at the `LinearBounds` level.
///
/// Builds the input-layer CROWN backward DIRECTLY (a 1×H read-out spec composed
/// through the input Linear `W1` of shape H×2), with ZERO biases so there is no
/// constant term to mask the coefficient error. The exact input coefficient for
/// x0 is `Σ_k OW[k]·W0[k]/2^26 = 0`, but the round-to-nearest f32 `A·W` GEMM
/// evaluates it to `+1.490116e-8`. Over the box `x0 ∈ [1, 2]` the lower-bound
/// concretization of an exact-0 coefficient is 0, but a stored `+1.49e-8`
/// coefficient yields `+1.49e-8 · 1 > 0` — a false over-claim of `y >= +1.49e-8`.
///
/// LEGACY f32 (`NY_AW_LEGACY_F32`): FAILS (`L_f32 ≈ +1.49e-8 > 0`).
/// PATCHED: the certified-error penalty makes `L_f32 <= 0`.
/// Read-out coefficient `OW[k] = raw/8192` (the `A` row of the corner-flip net).
pub(super) const CORNER_FLIP_OW: [f32; 64] = {
    let raw: [i32; 64] = [
        2590, 1707, 1872, 3050, -2002, -181, -3279, 1088, 1509, -404, 1882, -1707, -3938, 2386,
        -1085, -2294, 2359, 2679, 3521, 1147, 2933, 3948, -2548, -3592, 293, 620, -1406, 3454,
        -3725, 2486, 72, -1401, 3232, -859, -603, -3621, 3304, 3931, 3510, 3788, -1229, -127,
        -1706, -3933, 1721, 2437, -1977, -2743, 2181, -3692, -210, -1679, 3652, -3849, 2592, 43,
        -3949, -924, -194, -2831, 2198, 755, -1757, 1,
    ];
    let mut out = [0.0f32; 64];
    let mut i = 0;
    while i < 64 {
        out[i] = raw[i] as f32 / 8192.0;
        i += 1;
    }
    out
};
/// Input-layer column `W0[k] = raw/8192` chosen so `Σ_k OW[k]·W0[k] = 0` exactly
/// but the round-to-nearest f32 `A·W` GEMM rounds the x0 coefficient to `+1.49e-8`.
pub(super) const CORNER_FLIP_W0: [f32; 64] = {
    let raw: [i32; 64] = [
        -1042, 1835, -3961, -2919, -484, -1480, -3065, 1881, 1574, -1990, 1855, -3689, 678, 2919,
        3628, -1913, 3002, 3459, -1637, -915, -1130, 1403, -827, -1414, -2184, 1699, -581, -483,
        2878, 3348, -3328, 3608, -1020, 3857, -2246, -3520, 265, 139, -2957, 2986, 2737, -2609,
        -1287, 3497, -3258, -690, 2668, 2013, -1789, -1143, 2111, 1352, 524, -1489, -139, 1073,
        3885, 898, 899, -3414, -1597, 3456, 288, 3906,
    ];
    let mut out = [0.0f32; 64];
    let mut i = 0;
    while i < 64 {
        out[i] = raw[i] as f32 / 8192.0;
        i += 1;
    }
    out
};

#[test]
fn isolated_aw_corner_flip_zero_tolerance() {
    const OW: [f32; 64] = CORNER_FLIP_OW;
    const W0: [f32; 64] = CORNER_FLIP_W0;
    let h = OW.len();

    // Input layer W1: H x 2, column 0 = W0, column 1 = 0. No bias.
    let w1 = Array2::from_shape_fn((h, 2), |(k, j)| if j == 0 { W0[k] } else { 0.0 });
    let input_linear = LinearLayer::new(w1, None).unwrap();

    // Spec/read-out coefficient over hidden units: a single row = OW (1 x H), no bias.
    let spec_a = Array2::from_shape_fn((1, h), |(_, k)| OW[k]);
    let spec = LinearBounds::from_coefficients(spec_a.clone(), spec_a).unwrap();

    // Compose the read-out spec back through the input Linear (the H-wide A·W).
    let composed = input_linear.propagate_linear(&spec).unwrap();

    // Concretize over the box x0 ∈ [1, 2], x1 ∈ [1, 2] (strictly positive).
    let box_in = BoundedTensor::new(
        Array1::from_vec(vec![1.0f32, 1.0]).into_dyn(),
        Array1::from_vec(vec![2.0f32, 2.0]).into_dyn(),
    )
    .unwrap();
    let out = composed.concretize_sound(&box_in);
    let l_f32 = out.flatten().lower()[[0]];

    // The exact coefficient for x0 is 0 and there is no bias, so the exact lower
    // bound over the box is exactly 0. A sound `L_f32` must satisfy `L_f32 <= 0`.
    assert!(
        l_f32 <= 0.0,
        "UNSOUND A·W corner flip (isolated): concretized lower bound {l_f32} > 0, but the exact \
         input-coefficient for x0 is 0 (Σ OW·W0 = 0) so the true minimum over x0∈[1,2] is 0. \
         The f32 A·W rounded the coefficient to a positive value and concretize selected the \
         wrong corner — a false over-claim."
    );
}

/// BATCHED-PATH corner-flip regression — the decisive bite-test for the β-CROWN
/// /BaB verdict path.
///
/// Builds a REAL 2-Linear net (NO ReLU, so NO relaxation cushion) driven through
/// `GraphNetwork::propagate_crown_batched` (the batched verdict path):
///   - Layer 0: Linear H×2, column 0 = W0, column 1 = 0, NO bias.
///   - Layer 1: Linear 1×H = OW, NO bias.
///
/// The net is `y = OW·(W1·x) = (Σ_k OW[k]·W0[k])·x0 = 0·x0 = 0` EXACTLY for all x
/// (the composed x0 coefficient `Σ_k OW[k]·W0[k]` is 0 by construction; column 1
/// is all-zero). With NO ReLU and NO bias there is ZERO relaxation/cushion — the
/// batched lower bound equals exactly the concretization of the composed input
/// coefficient, just like the `isolated_aw_corner_flip` test but routed end-to-end
/// through the batched graph (`compute_batched_linear_coefficients_cpu`).
///
/// The round-to-nearest f32 batched `A·W` GEMM rounds the x0 coefficient to
/// `+1.49e-8`; concretizing over x0 ∈ [1, 2] (positive) the lower bound picks
/// x0 = 1, contributing `+1.49e-8 > 0` — a strict over-claim above the exact 0.
///
/// LEGACY f32 (`NY_AW_LEGACY_F32`): the batched bound is `> 0` (FAILS — proving
/// the test bites). PATCHED: the certified-error penalty restores `L_batched <= 0`.
#[test]
fn batched_aw_corner_flip_zero_tolerance() {
    const OW: [f32; 64] = CORNER_FLIP_OW;
    const W0: [f32; 64] = CORNER_FLIP_W0;
    let h = OW.len();

    let w1 = Array2::from_shape_fn((h, 2), |(k, j)| if j == 0 { W0[k] } else { 0.0 });
    let out_w = Array2::from_shape_fn((1, h), |(_, k)| OW[k]);

    // Confirm the composed x0 coefficient is EXACTLY 0 in rationals.
    let mut coeff_x0 = Rat::ZERO; // Σ OW·W0
    for k in 0..h {
        let ow = f32_to_rat(OW[k]);
        let w0 = f32_to_rat(W0[k]);
        coeff_x0 = coeff_x0.add(ow.mul(w0).unwrap()).unwrap();
    }
    assert!(
        coeff_x0.is_zero(),
        "test net x0 coefficient must be exactly 0, got {}",
        coeff_x0.to_clean_string().unwrap_or_default()
    );

    let mut net = Network::new();
    net.add_layer(Layer::Linear(LinearLayer::new(w1, None).unwrap()));
    net.add_layer(Layer::Linear(LinearLayer::new(out_w, None).unwrap()));

    // x0, x1 ∈ [1, 2] (strictly positive, so a +ε x0-coefficient raises lower).
    let box_in = BoundedTensor::new(
        Array1::from_vec(vec![1.0f32, 1.0]).into_dyn(),
        Array1::from_vec(vec![2.0f32, 2.0]).into_dyn(),
    )
    .unwrap();

    // EXACT output: y(x) = (Σ OW·W0)·x0 + 0·x1 = 0 for ALL x in the box.
    let exact_min = Rat::ZERO;

    let l_batched = batched_crown_lower(&net, &box_in);
    assert!(
        l_batched.is_finite(),
        "batched lower bound degraded to non-finite (unexpected for this small affine net)"
    );
    let l_batched_rat = f32_to_rat(l_batched);

    assert!(
        l_batched_rat <= exact_min,
        "UNSOUND BATCHED A·W corner flip: batched verdict-path lower bound {l_batched} ({}) > exact \
         network value 0 (the output is EXACTLY 0 over the box — the composed x0 coefficient \
         Σ OW·W0 = 0). The batched f32 A·W rounded the x0 coefficient positive and the missing \
         certified error let concretize over-claim — a false VERIFIED on the β-CROWN/BaB path.",
        l_batched_rat.to_clean_string().unwrap_or_default(),
    );
}

// =============================================================================
// General dyadic-net proptest vs the EXACT-RATIONAL CROWN oracle
// =============================================================================

/// Strategy: a 2 -> K -> 1 (single hidden layer) dyadic ReLU net with weights
/// `k/2^m` (small k, fixed m) that are EXACT in both f32 and Rat, an input box
/// straddling zero (so some hidden pre-activations are unstable), and a read-out
/// with mixed large +/- weights to force cancellation in `A·W`.
fn dyadic_net_strategy() -> impl Strategy<Value = DyadicNet> {
    // A single WIDE hidden layer (2 -> H -> 1). The input-layer linear backward
    // composes the ReLU-scaled read-out coefficient (length H) against W1 (H x 2),
    // an H-wide `A·W` contraction — exactly the sum whose f32 accumulation error
    // this fix covers. H = 64 makes the contraction wide enough to accumulate
    // multi-ULP cancellation error, while the single hidden layer keeps the
    // exact-rational ground-truth evaluation cheap. Read-out weights span a wide
    // dyadic range to drive cancellation in the composed coefficient; the input
    // box [-2, 2] amplifies any residual coefficient error into the bound.
    let hidden = 64usize;
    // Use a FINE dyadic grid (denominator 2^M) with a WIDE coefficient range so
    // the f32 products and their 64-wide sum exceed f32's 24-bit exact-integer
    // range and the accumulation genuinely ROUNDS. With M = 13 and |k| up to a
    // few thousand, each weight is `k/2^13` (still exact in a single f32, since
    // |k| < 2^24), but `Σ_k c·w` over the H-wide contraction rounds in f32 — the
    // exact bug this fix targets. The exact-rational ground truth stays cheap.
    const M: u32 = 13;
    let coeff = -6000i128..=6000i128; // weight ~ ±0.73
    let wide = -8000i128..=8000i128; // read-out ~ ±0.98, drives cancellation
    (
        proptest::collection::vec(proptest::collection::vec(coeff.clone(), 2), hidden), // W1: H x 2
        proptest::collection::vec(wide, hidden), // out_weight
        coeff,                                   // out_bias
    )
        .prop_map(move |(w1k, owk, obk)| {
            let w1: Vec<Vec<Rat>> = w1k
                .iter()
                .map(|row| row.iter().map(|&k| dyadic(k, M)).collect())
                .collect();
            // Force every hidden neuron STABLE-ACTIVE over the box [-2, 2]^2 by
            // choosing a bias `b1[k] >= |W1[k][0]|*2 + |W1[k][1]|*2`, so
            // `W1[k]·x + b1[k] >= 0` for all x in the box. With all ReLUs in the
            // identity regime the network is AFFINE and CROWN is EXACT — the ONLY
            // discrepancy between the f32 CROWN bound and the exact value is the
            // `A·W` accumulation error this fix targets, so the strict bound test
            // becomes a clean, sensitive detector of that error (no relaxation
            // gap to hide behind).
            let b1: Vec<Rat> = w1
                .iter()
                .map(|row| {
                    // bias >= |W1[k][0]|*2 + |W1[k][1]|*2 keeps the neuron active
                    // over [-2,2]^2: compute |w0|*2 + |w1|*2 exactly.
                    let w0 = if row[0].is_negative() {
                        row[0].neg()
                    } else {
                        row[0]
                    };
                    let w1c = if row[1].is_negative() {
                        row[1].neg()
                    } else {
                        row[1]
                    };
                    let two = Rat::from_int(2);
                    let s = w0.mul(two).unwrap().add(w1c.mul(two).unwrap()).unwrap();
                    // add a small positive slack so the breakpoint never sits exactly on the box.
                    s.add(dyadic(1, M)).unwrap()
                })
                .collect();
            let out_weight = owk.iter().map(|&k| dyadic(k, M)).collect();
            let out_bias = dyadic(obk, M);
            DyadicNet {
                weights: vec![w1],
                biases: vec![b1],
                out_weight,
                out_bias,
                input_lower: vec![dyadic(-8, 2), dyadic(-8, 2)], // [-2, 2]
                input_upper: vec![dyadic(8, 2), dyadic(8, 2)],
            }
        })
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 8000, ..ProptestConfig::with_cases(400) })]
    // 400 cases over a 64-wide all-active net: each case runs the full f32 CROWN
    // verdict path plus an exact-rational ground-truth minimum. The body is
    // engineered as a sensitive detector of the `A·W` accumulation error (fine
    // dyadic weights force the 64-wide f32 sum to round); fewer cases keep the
    // exact-rational ground truth within the time budget.

    /// ZERO-TOLERANCE: the f32 CROWN concretized lower bound must NEVER be
    /// tighter than the EXACT-RATIONAL CROWN lower bound from the same
    /// relaxation, AND must be `<=` the true network value at every box corner.
    #[ntest::timeout(180000)]
    #[test]
    fn dyadic_aw_lower_bound_never_exceeds_exact_oracle(net in dyadic_net_strategy()) {
        let problem = net.to_deep_problem();
        let nn = net.to_network();
        let input = net.input_box();

        let l_f32 = f32_crown_lower(&nn, &input);
        // Skip degenerate ±inf rows (conservative, trivially sound).
        prop_assume!(l_f32.is_finite());
        let l_f32_rat = f32_to_rat(l_f32);

        // ZERO-TOLERANCE ground truth: a sound lower bound `L_f32` must be `<=`
        // the EXACT network value f(x) for EVERY x in the box. We evaluate f
        // exactly in rationals over the box corners AND a dense dyadic grid
        // (which straddles the unstable-ReLU breakpoints), and assert `L_f32`
        // does not exceed any of them. This is unambiguous soundness — no second
        // sound implementation's relaxation choice is involved — and it is
        // exactly what catches an `A·W` over-claim (f32 reporting a higher lower
        // bound than the true minimum).
        let xmin = exact_true_min_over_box(&problem);
        prop_assert!(
            l_f32_rat <= xmin,
            "UNSOUND A·W: f32 CROWN lower bound {} ({}) > exact true network min over the box ({})",
            l_f32,
            l_f32_rat.to_clean_string().unwrap_or_default(),
            xmin.to_clean_string().unwrap_or_default(),
        );
    }
}

/// EXACT global minimum of the (relaxation-free) single-hidden-layer ReLU
/// network over the 2-D input box, in rationals.
///
/// The network is piecewise-linear; its minimum over the box is attained at a
/// VERTEX of the arrangement formed by the box edges and the hidden-neuron
/// breakpoint lines `W1[k]·x + b1[k] = 0`. We enumerate exactly:
/// - the 4 box corners,
/// - each breakpoint line's intersections with the 4 box edges,
/// - each pair of breakpoint lines' intersection (if inside the box),
///
/// and evaluate `f` exactly at every candidate, taking the minimum. This is the
/// EXACT infimum (no grid slack), so `L_f32 <= this` is a tight, NECESSARY
/// soundness condition that fires on any `A·W` over-claim — even a few-ULP one.
///
/// REQUIRES: a single hidden layer and 2 inputs (the proptest net shape).
fn exact_true_min_over_box(problem: &DeepReluProblem) -> Rat {
    assert_eq!(problem.input_lower.len(), 2);
    assert_eq!(problem.weights.len(), 1, "single hidden layer expected");
    let lo = &problem.input_lower;
    let hi = &problem.input_upper;
    let w1 = &problem.weights[0];
    let b1 = &problem.biases[0];
    let h = w1.len();

    let mut cands: Vec<[Rat; 2]> = Vec::new();
    // box corners
    for &x0 in &[lo[0], hi[0]] {
        for &x1 in &[lo[1], hi[1]] {
            cands.push([x0, x1]);
        }
    }
    let in_box = |p: &[Rat; 2]| p[0] >= lo[0] && p[0] <= hi[0] && p[1] >= lo[1] && p[1] <= hi[1];

    // Solve a·x0 + b·x1 + c = 0 for x1 given x0 (or x0 given x1).
    let line_at_fixed = |a: Rat, b: Rat, c: Rat, fixed_is_x0: bool, fixed: Rat| -> Option<Rat> {
        // a*x0 + b*x1 + c = 0
        if fixed_is_x0 {
            // solve for x1: x1 = -(a*x0 + c)/b
            if b.is_zero() {
                return None;
            }
            let num = a.mul(fixed).ok()?.add(c).ok()?.neg();
            num.mul(b.inv().ok()?).ok()
        } else {
            if a.is_zero() {
                return None;
            }
            let num = b.mul(fixed).ok()?.add(c).ok()?.neg();
            num.mul(a.inv().ok()?).ok()
        }
    };

    let push_if_in_box = |cands: &mut Vec<[Rat; 2]>, p: [Rat; 2]| {
        if in_box(&p) {
            cands.push(p);
        }
    };
    // Breakpoint-line intersections with the box edges, kept only if on the edge.
    // (For the all-active proptest net there are none in the box, so the corner
    // candidates determine the affine minimum; this branch makes the helper also
    // correct for unstable-ReLU nets.)
    for k in 0..h {
        let (a, b, c) = (w1[k][0], w1[k][1], b1[k]);
        // Cheap reject: if the line does not cross the box at all (all corners
        // strictly the same sign of `a·x0+b·x1+c`), it contributes no edge point.
        let s = |x0: Rat, x1: Rat| {
            a.mul(x0)
                .unwrap()
                .add(b.mul(x1).unwrap())
                .unwrap()
                .add(c)
                .unwrap()
        };
        let corners = [
            s(lo[0], lo[1]),
            s(lo[0], hi[1]),
            s(hi[0], lo[1]),
            s(hi[0], hi[1]),
        ];
        let all_pos = corners.iter().all(|v| v.is_positive() || v.is_zero());
        let all_neg = corners.iter().all(|v| v.is_negative() || v.is_zero());
        if all_pos || all_neg {
            continue; // line does not separate the box
        }
        if let Some(x1) = line_at_fixed(a, b, c, true, lo[0]) {
            push_if_in_box(&mut cands, [lo[0], x1]);
        }
        if let Some(x1) = line_at_fixed(a, b, c, true, hi[0]) {
            push_if_in_box(&mut cands, [hi[0], x1]);
        }
        if let Some(x0) = line_at_fixed(a, b, c, false, lo[1]) {
            push_if_in_box(&mut cands, [x0, lo[1]]);
        }
        if let Some(x0) = line_at_fixed(a, b, c, false, hi[1]) {
            push_if_in_box(&mut cands, [x0, hi[1]]);
        }
    }

    let mut best: Option<Rat> = None;
    for p in &cands {
        if let Ok(y) = problem.eval(&[p[0], p[1]]) {
            best = Some(match best {
                Some(b) if b <= y => b,
                _ => y,
            });
        }
    }
    best.expect("at least the box corners are candidates")
}

// =============================================================================
// BATCHED CROWN (β-CROWN/BaB verdict path) A·W soundness vs the exact oracle
// =============================================================================
//
// ADVERSARIAL ANGLE (#vnncomp-aw-soundness, batched). The strict scalar tests
// above prove the *single-domain* (`propagate_crown_with_engine`) path is
// sound. But the β-CROWN / branch-and-bound verdict path for HARD instances
// runs the *N-D batched* CROWN backward (`GraphNetwork::propagate_crown_batched`
// → `compute_batched_linear_coefficients_cpu` → `BatchedLinearBounds::
// concretize_sound`), a DIFFERENT code path that historically:
//   (1) used the round-to-nearest f32 faer `mat_mul` for `A·W` (vs the scalar
//       `aw_f64_with_abssum` exact-f32×f32-into-f64 accumulation) — a ~1-ULP
//       divergence (the `crown_batched_equiv` failures: batched 3.4901295 vs
//       scalar 3.4901297 at tolerance 0), and
//   (2) carried NO certified `γ_n·S` coefficient error, so `concretize_sound`
//       only applied a blanket 1-ULP widen.
// Either makes the batched lower bound up to ~1 ULP OPTIMISTIC = UNSOUND: it
// can report a tighter (higher) lower bound than the true minimum, a false
// VERIFIED on the BaB path where one wrong verdict costs -150.
//
// These tests assert the BATCHED concretized lower bound against the SAME
// exact-rational ground truth the scalar tests use, with ZERO tolerance. If the
// batched path is 1-ULP optimistic anywhere, the strict `<=` fails and the test
// catches it — exactly as the scalar test catches the single-domain bug.

/// Deterministic batched front3 end-to-end verdict: the BATCHED path must NOT
/// falsely verify `y >= 0` on the front3 net (true min -1/2097152 < 0, at the
/// corners (0,0) and (1,0)).
#[test]
fn front3_batched_no_false_verified_zero_tolerance() {
    let net = front3_net();
    let problem = net.to_deep_problem();
    let nn = net.to_network();
    let input = net.input_box();

    // BATCHED (β-CROWN/BaB) verdict-path lower bound.
    let l_batched = batched_crown_lower(&nn, &input);
    let l_batched_rat = f32_to_rat(l_batched);

    // EXACT network minimum over every box corner (front3 min is at a corner).
    let n = problem.input_lower.len();
    let mut true_min = problem
        .eval(&[problem.input_lower[0], problem.input_lower[1]])
        .unwrap();
    for mask in 0u32..(1u32 << n) {
        let x: Vec<Rat> = (0..n)
            .map(|d| {
                if mask & (1 << d) != 0 {
                    problem.input_upper[d]
                } else {
                    problem.input_lower[d]
                }
            })
            .collect();
        let y = problem.eval(&x).unwrap();
        if y < true_min {
            true_min = y;
        }
    }

    assert!(
        l_batched_rat <= true_min,
        "UNSOUND BATCHED front3: batched CROWN lower bound {l_batched} ({}) > true network min \
         ({}) — the batched (β-CROWN/BaB) verdict path over-claims on a net with a real violation.",
        l_batched_rat.to_clean_string().unwrap_or_default(),
        true_min.to_clean_string().unwrap_or_default(),
    );
    assert!(
        true_min.is_negative(),
        "test net must have a true violation (min < 0)"
    );
    assert!(
        l_batched <= 0.0,
        "BATCHED f32 CROWN must NOT falsely verify y >= 0 (got L_batched = {l_batched})"
    );
}

/// Cross-check: the SCALAR and BATCHED verdict paths must reach the SAME verdict
/// on front3 — neither may falsely verify `y >= 0`. A path-dependent verdict on a
/// net with a real violation is a soundness divergence regardless of which path
/// carries it.
#[test]
fn front3_scalar_and_batched_agree_no_false_verified() {
    let net = front3_net();
    let nn = net.to_network();
    let input = net.input_box();
    let l_scalar = f32_crown_lower(&nn, &input);
    let l_batched = batched_crown_lower(&nn, &input);
    // Both are SOUND lower bounds of the same quantity, so both must clear the
    // true minimum. Their relative ordering is NOT asserted: the two paths
    // accumulate their certified error differently and neither is uniformly
    // tighter (see `dyadic_aw_batched_and_scalar_both_below_exact_oracle`).
    assert!(
        l_batched <= 0.0 && l_scalar <= 0.0,
        "front3: scalar L={l_scalar}, batched L={l_batched}; both must be <= 0 (true min < 0)"
    );
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 8000, ..ProptestConfig::with_cases(400) })]

    /// ZERO-TOLERANCE BATCHED soundness: the BATCHED (β-CROWN/BaB) CROWN
    /// concretized lower bound must NEVER exceed the EXACT-RATIONAL true network
    /// minimum over the box — the same unambiguous ground truth used by the
    /// scalar `dyadic_aw_lower_bound_never_exceeds_exact_oracle` test. This
    /// proves the batched verdict path is sound, not 1-ULP optimistic.
    #[ntest::timeout(180000)]
    #[test]
    fn dyadic_aw_batched_lower_bound_never_exceeds_exact_oracle(net in dyadic_net_strategy()) {
        let problem = net.to_deep_problem();
        let nn = net.to_network();
        let input = net.input_box();

        let l_batched = batched_crown_lower(&nn, &input);
        // Skip degenerate ±inf rows (conservative, trivially sound).
        prop_assume!(l_batched.is_finite());
        let l_batched_rat = f32_to_rat(l_batched);

        let xmin = exact_true_min_over_box(&problem);
        prop_assert!(
            l_batched_rat <= xmin,
            "UNSOUND BATCHED A·W: batched CROWN lower bound {} ({}) > exact true network min over \
             the box ({}). The β-CROWN/BaB verdict path is 1-ULP optimistic — it must f64-accumulate \
             A·W (aw_f64_with_abssum) and carry the certified γ_n·S error through concretize.",
            l_batched,
            l_batched_rat.to_clean_string().unwrap_or_default(),
            xmin.to_clean_string().unwrap_or_default(),
        );
    }

    /// Cross-path soundness: BOTH the scalar AND the batched concretized lower
    /// bound must be `<=` the EXACT-RATIONAL true network minimum over the box.
    /// Running both on the SAME net in one case doubles coverage cheaply and
    /// guards against the batched path being soundness-divergent from scalar
    /// (e.g. carrying the error on one path but not the other).
    ///
    /// NOTE (deliberate non-assertion): the two SOUND lower bounds can legitimately
    /// differ by several f32 ULP — the scalar path composes the incoming
    /// coefficient error `Σ_k err_in·|W|` plus a bias-error term through the
    /// chain, while the batched path accumulates a fresh per-layer error, so
    /// neither bound is uniformly tighter. The ONLY soundness invariant is that
    /// each is `<=` the exact oracle, which is what we assert; asserting a fixed
    /// relative ordering between two sound bounds would be a false invariant.
    #[ntest::timeout(180000)]
    #[test]
    fn dyadic_aw_batched_and_scalar_both_below_exact_oracle(net in dyadic_net_strategy()) {
        let problem = net.to_deep_problem();
        let nn = net.to_network();
        let input = net.input_box();
        let l_scalar = f32_crown_lower(&nn, &input);
        let l_batched = batched_crown_lower(&nn, &input);
        let xmin = exact_true_min_over_box(&problem);

        if l_scalar.is_finite() {
            prop_assert!(
                f32_to_rat(l_scalar) <= xmin,
                "UNSOUND SCALAR A·W: scalar lower bound {} > exact true min {}",
                l_scalar,
                xmin.to_clean_string().unwrap_or_default(),
            );
        }
        if l_batched.is_finite() {
            prop_assert!(
                f32_to_rat(l_batched) <= xmin,
                "UNSOUND BATCHED A·W: batched lower bound {} > exact true min {}",
                l_batched,
                xmin.to_clean_string().unwrap_or_default(),
            );
        }
    }
}

// ===========================================================================
// f32 abs-sum seam (#f32-abssum-seam, docs/F32_ABSSUM_SEAM.md): `aw_via_engine`
// computes the certified-error base S = Σ|a|·|w| via a fast f32 Sgemm and
// inflates it to a guaranteed over-bound S_hat >= true_S. These ZERO-tolerance
// exact-rational oracles are the decisive soundness gate for that inflation:
// if S_hat ever under-bounds the exact real true_S, ε = γ_n·S_hat under-bounds
// the coefficient error → false VERIFIED. The engine's gemm_f32 accumulates in
// true f32, matching the round-to-nearest model the inflation is proven against.
// ===========================================================================

/// Exact f64 → `Rat` (mirror of `f32_to_rat` with the 52-bit mantissa / 1023
/// exponent bias). Needed to compare the f64 `S_hat` against the exact real
/// `true_S` with ZERO slack.
fn f64_to_rat(x: f64) -> Rat {
    Rat::from_f64_exact(x).unwrap_or_else(|| panic!("f64_to_rat: non-finite {x}"))
}

/// Exact real `true_S[i,j] = Σ_k |a_ik|·|w_kj|` in `Rat` (each `|f32|` is exact).
fn exact_true_s(a: &faer::Mat<f32>, w: &faer::Mat<f32>, i: usize, j: usize, k: usize) -> Rat {
    let mut s = Rat::ZERO;
    for kk in 0..k {
        let ax = f32_to_rat(a[(i, kk)].abs());
        let wy = f32_to_rat(w[(kk, j)].abs());
        s = s.add(ax.mul(wy).expect("Rat mul")).expect("Rat add");
    }
    s
}

/// Wide-magnitude mixed-sign f32 stream for the seam fixtures.
fn seam_f32_stream(seed: u64) -> impl FnMut() -> f32 {
    let mut rng = seed | 1;
    move || {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let e = ((rng >> 40) % 40) as i32 - 20; // exponent in [-20, 19]
        let mant = ((rng >> 12) & 0x7f_ffff) as f32 / (1u32 << 23) as f32; // [0,1)
        let sign = if (rng >> 3) & 1 == 1 { -1.0 } else { 1.0 };
        sign * (1.0 + mant) * 2f32.powi(e)
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(400))]

    /// DECISIVE (design §4.1): the seam's inflated `S_hat` must over-bound the
    /// EXACT real `true_S` for every entry, and the coefficient `aw` must be
    /// unaffected — driven through `NaiveCpuGemmEngine` (true-f32 `gemm_f32`,
    /// exact-widened f64 `gemm_f64`), zero tolerance.
    #[test]
    fn seam_f32_abssum_over_bounds_exact_true_s(
        m in 1usize..4,
        k in 1usize..48,
        p in 1usize..4,
        seed in any::<u64>(),
    ) {
        use crate::layers::linear::crown_single::aw_via_engine;
        use ny_core::NaiveCpuGemmEngine;

        let mut next = seam_f32_stream(seed);
        let a = faer::Mat::<f32>::from_fn(m, k, |_, _| next());
        let w = faer::Mat::<f32>::from_fn(k, p, |_, _| next());

        let (aw, s) = aw_via_engine(&NaiveCpuGemmEngine, &a, &w, m, k, p)
            .expect("seam returns Some for a CPU engine");

        for i in 0..m {
            for j in 0..p {
                let true_s = exact_true_s(&a, &w, i, j, k);
                prop_assert!(
                    s[[i, j]].is_finite(),
                    "S_hat[{i},{j}] must be finite for finite inputs (got {})",
                    s[[i, j]]
                );
                prop_assert!(
                    f64_to_rat(s[[i, j]]) >= true_s,
                    "UNSOUND f32 abs-sum: S_hat[{i},{j}]={} < exact true_S={} (k={k})",
                    s[[i, j]],
                    true_s.to_clean_string().unwrap_or_default(),
                );
                // Coefficient A·W is the f64 product, unchanged by the S seam.
                let mut true_aw_lo = 0.0f64;
                for kk in 0..k {
                    true_aw_lo += f64::from(a[(i, kk)]) * f64::from(w[(kk, j)]);
                }
                prop_assert!(
                    (aw[[i, j]] - true_aw_lo).abs() <= 1e-9 * (1.0 + true_aw_lo.abs()),
                    "coefficient aw[{i},{j}]={} drifted from f64 product {}",
                    aw[[i, j]],
                    true_aw_lo
                );
            }
        }
    }
}

/// FTZ underflow guard (design §4.3, §2 step 5): a GPU `gemm_f32` that flushes
/// subnormals to zero can return `fl32_result = 0` for a genuinely-positive
/// `true_S`; the additive `G = 2k·2^-126` guard must still lift `S_hat` above it.
/// `NaiveCpuGemmEngine` uses gradual underflow, so this needs a flush-to-zero
/// mock.
#[test]
fn seam_ftz_underflow_guard_covers_flushed_products() {
    use crate::layers::linear::crown_single::aw_via_engine;
    use ny_core::{NaiveCpuGemmEngine, Result as NyResult};

    struct MockFtzGemmEngine;
    impl ny_core::GemmEngine for MockFtzGemmEngine {
        fn gemm_f32(
            &self,
            m: usize,
            k: usize,
            n: usize,
            a: &[f32],
            b: &[f32],
        ) -> NyResult<Vec<f32>> {
            // f32 accumulation, flushing every subnormal product AND partial sum
            // to zero (models GPU Sgemm FTZ).
            let mut c = vec![0.0f32; m * n];
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0.0f32;
                    for kk in 0..k {
                        let mut prod = a[i * k + kk] * b[kk * n + j];
                        if prod != 0.0 && prod.abs() < f32::MIN_POSITIVE {
                            prod = 0.0;
                        }
                        acc += prod;
                        if acc != 0.0 && acc.abs() < f32::MIN_POSITIVE {
                            acc = 0.0;
                        }
                    }
                    c[i * n + j] = acc;
                }
            }
            Ok(c)
        }
        fn gemm_f64(
            &self,
            m: usize,
            k: usize,
            n: usize,
            a: &[f64],
            b: &[f64],
        ) -> NyResult<Vec<f64>> {
            NaiveCpuGemmEngine.gemm_f64(m, k, n, a, b)
        }
    }

    // |a| = |w| = 2^-64 → true product 2^-128 (subnormal) flushes to 0 in the
    // mock, so fl32_result = 0 while true_S = k·2^-128 > 0. S_hat must cover it.
    let k = 5usize;
    let val = 2f32.powi(-64);
    let a = faer::Mat::<f32>::from_fn(1, k, |_, _| val);
    let w = faer::Mat::<f32>::from_fn(k, 1, |_, _| val);

    let (_, s) = aw_via_engine(&MockFtzGemmEngine, &a, &w, 1, k, 1).expect("seam");
    let true_s = exact_true_s(&a, &w, 0, 0, k); // = k·2^-128, exact
    assert!(
        s[[0, 0]] > 0.0 && f64_to_rat(s[[0, 0]]) >= true_s,
        "FTZ guard failed: S_hat={} < true_S={} (flushed products under-counted)",
        s[[0, 0]],
        true_s.to_clean_string().unwrap_or_default()
    );
}
