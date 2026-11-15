// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Soundness pins for `#vit-zono-root` (the certified forward-zonotope root
//! map for vit_2023) on synthetic inputs and a miniature vit.
//!
//! * exact-rational band checks for the Exp / Reciprocal / Clip bands;
//! * exact-rational pointwise FORM containment of the live-live MatMul
//!   (both transposes, with and without scale, shared symbols in different
//!   row positions) and of the reduction;
//! * the softmax pairwise box: sampled containment, and strictly tighter than
//!   the independent-box bound when the logits share a symbol;
//! * a miniature vit (Conv2d patch embedding, CLS Concat, positional add,
//!   BatchNorm, Q/K/V, `Q K^T`, Softmax, `attn V`, residuals, ReLU MLP,
//!   ReduceMean, head), plain AND rewritten by the fixed-index softmax
//!   decomposition: every published box inside ny's IBP box and containing
//!   every sampled value;
//! * the runtime self-check refusing a transformer that encloses the wrong
//!   function and a compiler that models the wrong one;
//! * the reference-collector hook: dark = no computation, armed = shrink-only
//!   on the same route, the seeded CROWN-IBP route contains samples;
//! * refusals (unsupported op, memory budget), the exact-box cache, the
//!   shrink-only merge, the reduction map against ny's own layer, dark default;
//! * #zono-exact-bn on the DAG path: the BatchNorm charges the binary64
//!   enclosure of its exact-real coefficients, the stripped twin the binary32
//!   affine, and the exact map still self-checks and contains samples.
//!
//! Every check COUNTS what it checked; a test whose precondition (a MatMul
//! remainder symbol, a band, a tightened element, a mutation actually armed)
//! does not hold fails instead of passing vacuously.

use std::collections::HashMap;

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{Signed, Zero};
use ny_tensor::BoundedTensor;

use super::affine::{BatchNormCoefficients, ChannelAffine};
use super::concretize::concretize;
use super::dag::{DagOp, DagOpSet, DagProgram};
use super::elementwise::Band;
use super::ids::{SymbolAllocator, SymbolId};
use super::state::ZonoState;
use super::tests::Rng;
use super::vit_ops::{
    bilinear_matmul, box_state, clip_band, exp_band, fault, group_sum, reciprocal_band,
    reduce_group_map, softmax_box, softmax_eval, softmax_lanes, MatMulDims,
};
use super::vit_root::{
    cached_for, compute_vit_zono_root_map, merge_shrink_only, vit_zono_root_map,
    vit_zono_root_plan, VitZonoRootPlan,
};
use super::ZonoRefusal;
use crate::beta_crown::config::BetaCrownConfig;
use crate::layers::{
    AddConstantLayer, AddLayer, BatchNormLayer, BilinearCrownLayer, BoundPropagation, ConcatLayer,
    Conv2dLayer, Layer, LinearLayer, ReLULayer, ReduceMeanLayer, ReduceSumLayer, ReshapeLayer,
    SoftmaxLayer, TanhLayer, TransposeLayer,
};
use crate::{AlphaCrownConfig, GraphNetwork, GraphNode};

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

#[allow(clippy::cast_possible_truncation)]
fn rand_f32(rng: &mut Rng, n: usize, scale: f64) -> Vec<f32> {
    (0..n).map(|_| (rng.sym() * scale) as f32).collect()
}

fn linear(rng: &mut Rng, rows: usize, cols: usize, scale: f64) -> Layer {
    let w = Array2::from_shape_vec((rows, cols), rand_f32(rng, rows * cols, scale)).expect("w");
    let b = Array1::from_vec(rand_f32(rng, rows, 0.2));
    Layer::Linear(LinearLayer::new(w, Some(b)).expect("linear"))
}

fn node(name: &str, layer: Layer, inputs: &[&str]) -> GraphNode {
    GraphNode::new(
        name,
        layer,
        inputs.iter().map(|input| (*input).to_string()).collect(),
    )
}

#[allow(clippy::cast_possible_truncation)]
fn bn_layer(rng: &mut Rng, channels: usize) -> BatchNormLayer {
    let gamma: Vec<f32> = (0..channels).map(|_| (0.5 + rng.unit()) as f32).collect();
    let beta = rand_f32(rng, channels, 0.3);
    let mean = rand_f32(rng, channels, 0.3);
    let var: Vec<f32> = (0..channels).map(|_| (0.3 + rng.unit()) as f32).collect();
    let arr = |v: Vec<f32>| ArrayD::from_shape_vec(IxDyn(&[channels]), v).expect("bn vector");
    BatchNormLayer::new(&arr(gamma), &arr(beta), &arr(mean), &arr(var), 1e-5).expect("bn layer")
}

/// A miniature vit: `[1, 4, 4]` image -> Conv2d 2x2/2 patch embedding (6
/// channels) -> `[4, 6]` tokens -> CLS Concat -> `[5, 6]` + positional
/// embedding -> BatchNorm over the embedding axis -> Q/K/V (2 heads of 3) ->
/// `Q K^T * 0.5` -> Softmax -> `attn V` -> projection + residual -> ReLU MLP +
/// residual -> ReduceMean over tokens -> 3-way head. Exactly the vit_2023 op
/// set as ny lowers it (var x var MatMul = `BilinearCrown`).
pub(crate) fn mini_vit(seed: u64) -> GraphNetwork {
    let mut rng = Rng::new(seed);
    let mut g = GraphNetwork::new();
    let kernel =
        ArrayD::from_shape_vec(IxDyn(&[6, 1, 2, 2]), rand_f32(&mut rng, 24, 0.9)).expect("kernel");
    let conv = Conv2dLayer::new(
        kernel,
        Some(Array1::from_vec(rand_f32(&mut rng, 6, 0.1))),
        (2, 2),
        (0, 0),
    )
    .expect("patch conv");
    g.add_node(GraphNode::from_input("patch", Layer::Conv2d(conv)));
    g.add_node(node(
        "tokens_cm",
        Layer::Reshape(ReshapeLayer::new(vec![6, 4])),
        &["patch"],
    ));
    g.add_node(node(
        "tokens",
        Layer::Transpose(TransposeLayer::new(vec![1, 0])),
        &["tokens_cm"],
    ));
    let cls = BoundedTensor::concrete(
        ArrayD::from_shape_vec(IxDyn(&[1, 6]), rand_f32(&mut rng, 6, 0.5)).expect("cls"),
    )
    .expect("cls tensor");
    g.add_node(node(
        "with_cls",
        Layer::Concat(ConcatLayer::with_constants(
            0,
            vec![vec![1, 6], vec![4, 6]],
            vec![Some(cls), None],
        )),
        &["tokens"],
    ));
    g.add_node(node(
        "x0",
        Layer::AddConstant(AddConstantLayer::new(
            ArrayD::from_shape_vec(IxDyn(&[5, 6]), rand_f32(&mut rng, 30, 0.2)).expect("pos"),
        )),
        &["with_cls"],
    ));
    g.add_node(node(
        "x0_t",
        Layer::Transpose(TransposeLayer::new(vec![1, 0])),
        &["x0"],
    ));
    g.add_node(node(
        "bn",
        Layer::BatchNorm(bn_layer(&mut rng, 6)),
        &["x0_t"],
    ));
    g.add_node(node(
        "xn",
        Layer::Transpose(TransposeLayer::new(vec![1, 0])),
        &["bn"],
    ));
    for name in ["q", "k", "v"] {
        let heads = format!("{name}_h");
        let per_head = format!("{name}_t");
        g.add_node(node(name, linear(&mut rng, 6, 6, 0.7), &["xn"]));
        g.add_node(node(
            &heads,
            Layer::Reshape(ReshapeLayer::new(vec![5, 2, 3])),
            &[name],
        ));
        g.add_node(node(
            &per_head,
            Layer::Transpose(TransposeLayer::new(vec![1, 0, 2])),
            &[heads.as_str()],
        ));
    }
    g.add_node(GraphNode::binary(
        "scores",
        Layer::BilinearCrown(BilinearCrownLayer::new(true, Some(0.5))),
        "q_t",
        "k_t",
    ));
    g.add_node(node(
        "attn",
        Layer::Softmax(SoftmaxLayer::new(-1)),
        &["scores"],
    ));
    g.add_node(GraphNode::binary(
        "ctx",
        Layer::BilinearCrown(BilinearCrownLayer::new(false, None)),
        "attn",
        "v_t",
    ));
    g.add_node(node(
        "ctx_t",
        Layer::Transpose(TransposeLayer::new(vec![1, 0, 2])),
        &["ctx"],
    ));
    g.add_node(node(
        "ctx_f",
        Layer::Reshape(ReshapeLayer::new(vec![5, 6])),
        &["ctx_t"],
    ));
    g.add_node(node("proj", linear(&mut rng, 6, 6, 0.6), &["ctx_f"]));
    g.add_node(node("res1", Layer::Add(AddLayer), &["x0", "proj"]));
    g.add_node(node("fc1", linear(&mut rng, 8, 6, 0.7), &["res1"]));
    g.add_node(node("act", Layer::ReLU(ReLULayer), &["fc1"]));
    g.add_node(node("fc2", linear(&mut rng, 6, 8, 0.6), &["act"]));
    g.add_node(node("res2", Layer::Add(AddLayer), &["res1", "fc2"]));
    g.add_node(node(
        "pool",
        Layer::ReduceMean(ReduceMeanLayer::new(vec![0], false)),
        &["res2"],
    ));
    g.add_node(node("head", linear(&mut rng, 3, 6, 0.8), &["pool"]));
    g.set_output("head");
    g
}

/// A `[1, 4, 4]` box of half-width `radius` around a random centre.
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn mini_vit_input(seed: u64, radius: f64) -> BoundedTensor {
    let mut rng = Rng::new(seed);
    let centre: Vec<f64> = (0..16).map(|_| rng.sym()).collect();
    let lo: Vec<f32> = centre.iter().map(|c| (c - radius) as f32).collect();
    let hi: Vec<f32> = centre.iter().map(|c| (c + radius) as f32).collect();
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 4, 4]), lo).expect("lo"),
        ArrayD::from_shape_vec(IxDyn(&[1, 4, 4]), hi).expect("hi"),
    )
    .expect("box")
}

fn test_plan() -> VitZonoRootPlan {
    VitZonoRootPlan {
        max_bytes: 1 << 32,
        max_secs: 120.0,
        source: ny_levers::Source::Config,
    }
}

/// The centre, then alternately uniform points and vertices of the box.
fn sample_points(input: &BoundedTensor, count: usize, rng: &mut Rng) -> Vec<Vec<f64>> {
    let lo: Vec<f64> = input.lower().iter().map(|&v| f64::from(v)).collect();
    let hi: Vec<f64> = input.upper().iter().map(|&v| f64::from(v)).collect();
    let mut points = vec![lo
        .iter()
        .zip(&hi)
        .map(|(&l, &h)| 0.5 * l + 0.5 * h)
        .collect::<Vec<f64>>()];
    for k in 0..count {
        points.push(
            lo.iter()
                .zip(&hi)
                .map(|(&l, &h)| {
                    if k % 2 == 0 {
                        (l + (h - l) * rng.unit()).clamp(l, h)
                    } else if rng.unit() < 0.5 {
                        l
                    } else {
                        h
                    }
                })
                .collect(),
        );
    }
    points
}

/// Sampled containment of every node of `bounds` the program computes, with
/// the program's plain f64 semantics as the truth (the runtime self-check pins
/// those semantics to ny's own pointwise forward). Returns the element count.
fn sampled_containment(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    bounds: &HashMap<String, BoundedTensor>,
    points: usize,
    seed: u64,
) -> usize {
    let (program, _) = DagProgram::compile_with_ibp_opset(graph, input, None, DagOpSet::Vit)
        .expect("the vit program compiles");
    let mut rng = Rng::new(seed);
    let pts = sample_points(input, points, &mut rng);
    let mut checked = 0usize;
    for point in &pts {
        program
            .evaluate_points(point, 1, |_, name, values| {
                let Some(entry) = bounds.get(name) else {
                    return;
                };
                if entry.len() != values.len() {
                    return;
                }
                for (j, ((&v, &l), &u)) in values
                    .iter()
                    .zip(entry.lower().iter())
                    .zip(entry.upper().iter())
                    .enumerate()
                {
                    let tol = 1e-9 * (1.0 + v.abs());
                    assert!(
                        v >= f64::from(l) - tol && v <= f64::from(u) + tol,
                        "`{name}`[{j}] = {v:e} outside the published [{l:e}, {u:e}]"
                    );
                    checked += 1;
                }
            })
            .expect("evaluate");
    }
    checked
}

// ---------------------------------------------------------------------------
// Exact-rational band checks
// ---------------------------------------------------------------------------

fn rat(v: f64) -> BigRational {
    BigRational::from_float(v).expect("finite binary64 value")
}

fn int(v: u64) -> BigRational {
    BigRational::from_integer(BigInt::from(v))
}

/// Rational enclosure of `e^y` for `|y| <= 16` (Taylor to 90 terms, remainder
/// `e^|y| |y|^N / N! <= 3^ceil|y| |y|^N / N!`). Independent of the certified
/// exponential the band itself uses.
fn exp_bracket(y: &BigRational) -> (BigRational, BigRational) {
    let mut term = int(1);
    let mut sum = BigRational::zero();
    for n in 0..90u64 {
        sum += &term;
        term = term * y / int(n + 1);
    }
    let mut bound = int(1);
    let mut k = BigRational::zero();
    while k < y.abs() {
        bound *= int(3);
        k += int(1);
    }
    let r = term.abs() * bound;
    (&sum - &r, &sum + &r)
}

/// Check `lambda x + mu_lo <= f(x) <= lambda x + mu_hi` in exact rationals at
/// the endpoints, the midpoint, one ulp inside, the given extra points and
/// random interior points. Returns the number of points checked.
fn check_band_rational<F>(
    band: &Band,
    lo: &[f64],
    hi: &[f64],
    rng: &mut Rng,
    extra: &[f64],
    bracket: F,
) -> usize
where
    F: Fn(&BigRational) -> (BigRational, BigRational),
{
    let mut checked = 0usize;
    for j in 0..band.len() {
        let (l, u) = (lo[j], hi[j]);
        let mut xs = vec![l, u, 0.5 * l + 0.5 * u];
        if u > l {
            xs.push(l.next_up().min(u));
            xs.push(u.next_down().max(l));
            for _ in 0..16 {
                xs.push((l + (u - l) * rng.unit()).clamp(l, u));
            }
            xs.extend(extra.iter().copied().filter(|x| *x >= l && *x <= u));
        }
        let lam = rat(band.lambda[j]);
        let (mlo, mhi) = (rat(band.mu_lo[j]), rat(band.mu_hi[j]));
        for x in xs {
            let xr = rat(x);
            let (f_lo, f_hi) = bracket(&xr);
            let lower = &lam * &xr + &mlo;
            let upper = &lam * &xr + &mhi;
            assert!(
                lower <= f_lo,
                "band lower (mu_lo {:e}) exceeds f at x={x:e} on [{l:e}, {u:e}]",
                band.mu_lo[j]
            );
            assert!(
                f_hi <= upper,
                "band upper (mu_hi {:e}) below f at x={x:e} on [{l:e}, {u:e}]",
                band.mu_hi[j]
            );
            checked += 1;
        }
    }
    checked
}

/// The Exp band holds in exact rationals against an INDEPENDENT rational
/// exponential, on wide, narrow, point, negative and positive intervals, at
/// the chord's tangency point too.
#[test]
fn vit_zono_exp_band_contains_exp_in_exact_rationals() {
    let intervals: [(f64, f64); 9] = [
        (-3.0, 1.5),
        (-0.2, 0.3),
        (0.0, 0.0),
        (-1e-9, 1e-9),
        (-8.0, -7.5),
        (2.0, 2.000_000_1),
        (-15.0, 5.0),
        (1.0, 1.0),
        (-0.5, 12.0),
    ];
    let lo: Vec<f64> = intervals.iter().map(|p| p.0).collect();
    let hi: Vec<f64> = intervals.iter().map(|p| p.1).collect();
    let band = exp_band(&lo, &hi).expect("exp band");
    let tangency: Vec<f64> = band
        .lambda
        .iter()
        .filter(|lam| **lam > 0.0)
        .map(|lam| lam.ln())
        .collect();
    let mut rng = Rng::new(0xE0B0_0001);
    let checked = check_band_rational(&band, &lo, &hi, &mut rng, &tangency, exp_bracket);
    assert!(checked > 100, "checked only {checked} points");
    assert!(
        band.lambda.iter().filter(|lam| **lam > 0.0).count() >= 6,
        "precondition: the chord (not the constant fallback) must be exercised: {:?}",
        band.lambda
    );
    assert!(
        exp_band(&[0.0], &[800.0]).is_err(),
        "an overflowing exponential must refuse"
    );
}

/// The Reciprocal band holds in exact rationals, including the degenerate
/// slope-0 case (wave-6's `[l, +inf)` relaxation) reached by an overflowing
/// `l u`, and refuses an interval that touches the pole.
#[test]
fn vit_zono_reciprocal_band_contains_one_over_x_in_exact_rationals() {
    let intervals: [(f64, f64); 8] = [
        (1.0, 2.0),
        (1.0, 17.0),
        (0.5, 0.500_000_1),
        (1.0, 1.0),
        (3.7, 1e3),
        (1e-3, 5.0),
        (1e200, 1e250),
        (1.0 - 1e-12, 1.0 + 1e-12),
    ];
    let lo: Vec<f64> = intervals.iter().map(|p| p.0).collect();
    let hi: Vec<f64> = intervals.iter().map(|p| p.1).collect();
    let band = reciprocal_band(&lo, &hi).expect("reciprocal band");
    let tangency: Vec<f64> = band
        .lambda
        .iter()
        .filter(|lam| **lam < 0.0)
        .map(|lam| (-1.0 / lam).sqrt())
        .collect();
    let mut rng = Rng::new(0xE0B0_0002);
    let checked = check_band_rational(&band, &lo, &hi, &mut rng, &tangency, |x| {
        let f = int(1) / x;
        (f.clone(), f)
    });
    assert!(checked > 80, "checked only {checked} points");
    assert_eq!(
        band.lambda[6], 0.0,
        "precondition: the overflowing l*u must take the slope-0 [l, +inf) branch"
    );
    assert!(
        band.lambda[0] < 0.0,
        "the ordinary interval takes the chord"
    );
    for (l, u) in [(0.0, 1.0), (-1.0, 2.0), (f64::NAN, 1.0)] {
        assert!(
            reciprocal_band(&[l], &[u]).is_err(),
            "[{l}, {u}] touches the pole and must refuse"
        );
    }
}

/// The Clip band is exact in rationals on every placement of the interval
/// against the clip window, including a one-sided (infinite) window.
#[test]
fn vit_zono_clip_band_contains_clip_in_exact_rationals() {
    let intervals: [(f64, f64); 7] = [
        (-0.5, 0.5),
        (0.2, 0.8),
        (-2.0, -1.0),
        (1.5, 2.0),
        (-1.0, 2.0),
        (0.0, 1.0),
        (1.0, 1.0),
    ];
    let lo: Vec<f64> = intervals.iter().map(|p| p.0).collect();
    let hi: Vec<f64> = intervals.iter().map(|p| p.1).collect();
    let mut rng = Rng::new(0xE0B0_0003);
    let mut checked = 0usize;
    for (cmin, cmax) in [(0.0, 1.0), (f64::NEG_INFINITY, 0.5), (-0.25, f64::INFINITY)] {
        let band = clip_band(&lo, &hi, cmin, cmax).expect("clip band");
        checked += check_band_rational(&band, &lo, &hi, &mut rng, &[cmin, cmax], |x| {
            let mut v = x.clone();
            if cmin.is_finite() && v < rat(cmin) {
                v = rat(cmin);
            }
            if cmax.is_finite() && v > rat(cmax) {
                v = rat(cmax);
            }
            (v.clone(), v)
        });
    }
    assert!(checked > 200, "checked only {checked} points");
}

// ---------------------------------------------------------------------------
// Exact-rational form containment of the MatMul and the reduction
// ---------------------------------------------------------------------------

fn state_with(rng: &mut Rng, n: usize, ids: &[u64], scale: f64) -> ZonoState {
    let center: Vec<f64> = (0..n).map(|_| 2.0 * rng.sym()).collect();
    let gens: Vec<f64> = (0..ids.len() * n)
        .map(|_| {
            if rng.unit() < 0.25 {
                0.0
            } else {
                scale * rng.sym()
            }
        })
        .collect();
    let rem: Vec<f64> = (0..n).map(|_| 0.01 * rng.unit()).collect();
    ZonoState::from_parts(
        n,
        center,
        gens,
        ids.iter().map(|&id| SymbolId(id)).collect(),
        rem,
    )
    .expect("state")
}

/// Exact value of `state` at the symbol assignment `e` and remainder point
/// `rho` (every entry an exact rational).
fn exact_values(
    state: &ZonoState,
    e: &HashMap<SymbolId, BigRational>,
    rho: &[BigRational],
) -> Vec<BigRational> {
    (0..state.len())
        .map(|j| {
            let mut v = rat(state.center()[j]) + &rho[j];
            for (row, id) in state.ids().iter().enumerate() {
                v += rat(state.generator_row(row)[j]) * &e[id];
            }
            v
        })
        .collect()
}

/// Check the exact `truth` against `out`'s pointwise form: the symbols in `e`
/// are known, every other row (a fresh symbol) and the remainder bound the
/// residual. Returns the number of elements checked.
fn check_form_exact(
    out: &ZonoState,
    e: &HashMap<SymbolId, BigRational>,
    truth: &[BigRational],
) -> usize {
    let mut checked = 0usize;
    for (j, t) in truth.iter().enumerate() {
        let mut predicted = rat(out.center()[j]);
        let mut radius = rat(out.remainder()[j]);
        for (row, id) in out.ids().iter().enumerate() {
            let g = rat(out.generator_row(row)[j]);
            match e.get(id) {
                Some(value) => predicted += g * value,
                None => radius += g.abs(),
            }
        }
        assert!(
            (t - &predicted).abs() <= radius,
            "element {j}: exact value outside the output form"
        );
        checked += 1;
    }
    checked
}

fn draw_symbols(rng: &mut Rng, ids: &[SymbolId]) -> HashMap<SymbolId, BigRational> {
    ids.iter()
        .map(|&id| {
            let v = match rng.next_u64() % 5 {
                0 => -1.0,
                1 => 1.0,
                _ => rng.sym(),
            };
            (id, rat(v))
        })
        .collect()
}

fn draw_rho(rng: &mut Rng, state: &ZonoState) -> Vec<BigRational> {
    state
        .remainder()
        .iter()
        .map(|&r| rat(r * rng.sym()))
        .collect()
}

/// The live-live MatMul encloses the EXACT product `s A B` (or `s A B^T`) in
/// its pointwise form, with symbols shared by A and B sitting at different
/// row positions, symbols only one side knows, remainders on both sides, and
/// every combination of transpose and scale. Symbols align BY ID: permuting
/// B's rows changes nothing.
#[test]
fn vit_zono_bilinear_matmul_encloses_the_exact_product_by_symbol_id() {
    let mut rng = Rng::new(0xB111_0001);
    let mut checked = 0usize;
    let mut fresh_total = 0usize;
    for transpose_b in [false, true] {
        for scale in [None, Some(0.5f64)] {
            let dims = MatMulDims::new(2, 3, 4, 3, transpose_b, scale).expect("dims");
            let a = state_with(&mut rng, dims.a_len(), &[0, 1, 2, 3, 4], 0.3);
            let b = state_with(&mut rng, dims.b_len(), &[3, 1, 7, 8], 0.3);
            let mut alloc = SymbolAllocator::new();
            for _ in 0..16 {
                alloc.fresh();
            }
            let (out, fresh) = bilinear_matmul(&a, &b, &dims, &mut alloc, None).expect("matmul");
            fresh_total += fresh;
            let mut seen = out.ids().to_vec();
            seen.sort_unstable();
            seen.dedup();
            assert_eq!(seen.len(), out.ids().len(), "an ID appears twice");
            let input_ids: Vec<SymbolId> = [0u64, 1, 2, 3, 4, 7, 8].map(SymbolId).to_vec();
            for _ in 0..40 {
                let e = draw_symbols(&mut rng, &input_ids);
                let av = exact_values(&a, &e, &draw_rho(&mut rng, &a));
                let bv = exact_values(&b, &e, &draw_rho(&mut rng, &b));
                let s = rat(scale.unwrap_or(1.0));
                let mut truth = Vec::with_capacity(dims.out_len());
                for bt in 0..dims.batch {
                    for i in 0..dims.m {
                        for j in 0..dims.n {
                            let mut acc = BigRational::zero();
                            for l in 0..dims.k {
                                acc += &av[(bt * dims.m + i) * dims.k + l]
                                    * &bv[dims.b_index(bt, l, j)];
                            }
                            truth.push(&s * acc);
                        }
                    }
                }
                checked += check_form_exact(&out, &e, &truth);
            }

            // ID alignment: the same B with its rows permuted yields the same
            // generator for every symbol.
            let (n, center, gens, ids, rem) = b.clone().into_parts();
            let perm = [2usize, 0, 3, 1];
            let mut pgens = Vec::with_capacity(gens.len());
            let mut pids = Vec::with_capacity(ids.len());
            for &row in &perm {
                pgens.extend_from_slice(&gens[row * n..(row + 1) * n]);
                pids.push(ids[row]);
            }
            let permuted = ZonoState::from_parts(n, center, pgens, pids, rem).expect("perm");
            let mut alloc2 = SymbolAllocator::new();
            for _ in 0..16 {
                alloc2.fresh();
            }
            let (out2, _) =
                bilinear_matmul(&a, &permuted, &dims, &mut alloc2, None).expect("matmul perm");
            for id in [0u64, 1, 2, 3, 4, 7, 8].map(SymbolId) {
                let r1 = out.ids().iter().position(|x| *x == id);
                let r2 = out2.ids().iter().position(|x| *x == id);
                match (r1, r2) {
                    (Some(r1), Some(r2)) => {
                        for (x, y) in out.generator_row(r1).iter().zip(out2.generator_row(r2)) {
                            assert!(
                                (x - y).abs() <= 1e-12 * (1.0 + x.abs()),
                                "symbol {id:?}: {x} vs {y} after permuting B's rows"
                            );
                        }
                    }
                    (None, None) => {}
                    other => panic!("symbol {id:?} present in one output only: {other:?}"),
                }
            }
        }
    }
    assert!(checked > 1000, "checked only {checked} elements");
    assert!(
        fresh_total > 0,
        "precondition: second-order remainder symbols"
    );
}

/// `ReduceSum` encloses the exact sums in its pointwise form, over several
/// axis sets; the map it uses agrees with ny's OWN ReduceSum/ReduceMean layer
/// on integer inputs (exact in f32 and f64).
#[test]
fn vit_zono_group_sum_encloses_exact_sums_and_matches_ny_reduction() {
    let mut rng = Rng::new(0x5051_0001);
    let shape = [2usize, 3, 4];
    let n = 24usize;
    let mut checked = 0usize;
    for (axes, keepdims) in [
        (vec![-1i64], true),
        (vec![0, 2], false),
        (vec![], true),
        (vec![1], false),
    ] {
        let sum_layer = ReduceSumLayer::new(axes.clone(), keepdims);
        let mean_layer = ReduceMeanLayer::new(axes.clone(), keepdims);
        #[allow(clippy::cast_precision_loss)]
        let codes: Vec<f32> = (0..n).map(|i| ((i * 7) % 13) as f32 - 6.0).collect();
        let probe = BoundedTensor::concrete(
            ArrayD::from_shape_vec(IxDyn(&shape), codes.clone()).expect("probe"),
        )
        .expect("probe tensor");
        let ny_sum = sum_layer.propagate_ibp(&probe).expect("ny ReduceSum");
        let ny_mean = mean_layer.propagate_ibp(&probe).expect("ny ReduceMean");
        let (out_of, n_out, count) =
            reduce_group_map(&shape, &axes, keepdims, ny_sum.shape()).expect("map");
        assert_eq!(n_out, ny_sum.len());
        let mut sums = vec![0.0f64; n_out];
        for (i, &j) in out_of.iter().enumerate() {
            sums[j] += f64::from(codes[i]);
        }
        #[allow(clippy::cast_precision_loss)]
        let divisor = count as f64;
        for j in 0..n_out {
            let (l, u) = (
                ny_sum.lower().as_slice().expect("std")[j],
                ny_sum.upper().as_slice().expect("std")[j],
            );
            assert!(
                f64::from(l) <= sums[j] && sums[j] <= f64::from(u),
                "sum {j}"
            );
            let mean = sums[j] / divisor;
            let (l, u) = (
                ny_mean.lower().as_slice().expect("std")[j],
                ny_mean.upper().as_slice().expect("std")[j],
            );
            assert!(f64::from(l) <= mean && mean <= f64::from(u), "mean {j}");
        }

        let x = state_with(&mut rng, n, &[0, 1, 2, 5], 0.4);
        let abs_g = x.abs_generator_sum();
        let out = group_sum(&x, &abs_g, &out_of, n_out, count).expect("group sum");
        let ids: Vec<SymbolId> = [0u64, 1, 2, 5].map(SymbolId).to_vec();
        for _ in 0..40 {
            let e = draw_symbols(&mut rng, &ids);
            let xv = exact_values(&x, &e, &draw_rho(&mut rng, &x));
            let mut truth = vec![BigRational::zero(); n_out];
            for (i, &j) in out_of.iter().enumerate() {
                truth[j] += &xv[i];
            }
            checked += check_form_exact(&out, &e, &truth);
        }
    }
    assert!(checked > 500, "checked only {checked} elements");
    assert!(
        reduce_group_map(&shape, &[0, -3], true, &[1, 3, 4]).is_err(),
        "a duplicate axis must refuse"
    );
    assert!(
        reduce_group_map(&shape, &[-1], true, &[2, 3]).is_err(),
        "a shape that disagrees with ny's IBP shape must refuse"
    );
}

/// The direct-Softmax box contains every sampled softmax, and — because the
/// pairwise differences cancel a shared symbol exactly — it is strictly
/// narrower than the independent-box bound on logits that share one.
#[test]
fn vit_zono_softmax_box_contains_samples_and_beats_independent_boxes() {
    let mut rng = Rng::new(0x50F7_0001);
    let n = 10usize;
    // Two lanes of five logits: a large shared symbol 0 on every logit, small
    // private symbols 1..=10, a small remainder.
    let mut gens = vec![0.0f64; 11 * n];
    for j in 0..n {
        gens[j] = 1.5;
        gens[(j + 1) * n + j] = 0.05 * (1.0 + rng.unit());
    }
    let center: Vec<f64> = (0..n).map(|_| rng.sym()).collect();
    let rem = vec![1e-3; n];
    let ids: Vec<SymbolId> = (0..11u64).map(SymbolId).collect();
    let x = ZonoState::from_parts(n, center, gens, ids, rem).expect("logits");
    let abs_g = x.abs_generator_sum();
    let (lo, hi) = concretize(&x, &abs_g);
    let lanes = softmax_lanes(&[2, 5], 1).expect("lanes");
    let (p_lo, p_hi) = softmax_box(&x, &abs_g, &lo, &hi, &lanes, None).expect("softmax box");

    // Independent boxes: the same intervals with every correlation dropped.
    let mid: Vec<f64> = lo.iter().zip(&hi).map(|(l, h)| 0.5 * l + 0.5 * h).collect();
    let rad: Vec<f64> = lo
        .iter()
        .zip(&hi)
        .zip(&mid)
        .map(|((l, h), m)| (h - m).max(m - l).next_up())
        .collect();
    let boxed = ZonoState::from_parts(n, mid, Vec::new(), Vec::new(), rad).expect("box");
    let boxed_abs = boxed.abs_generator_sum();
    let (blo, bhi) = concretize(&boxed, &boxed_abs);
    let (q_lo, q_hi) =
        softmax_box(&boxed, &boxed_abs, &blo, &bhi, &lanes, None).expect("independent box");
    let width = |l: &[f64], h: &[f64]| -> f64 { l.iter().zip(h).map(|(a, b)| b - a).sum() };
    assert!(
        width(&p_lo, &p_hi) < 0.5 * width(&q_lo, &q_hi),
        "the pairwise box ({}) must be much narrower than the independent one ({})",
        width(&p_lo, &p_hi),
        width(&q_lo, &q_hi)
    );

    let mut checked = 0usize;
    for _ in 0..500 {
        let e: Vec<f64> = (0..11)
            .map(|_| match rng.next_u64() % 4 {
                0 => -1.0,
                1 => 1.0,
                _ => rng.sym(),
            })
            .collect();
        let values: Vec<f64> = (0..n)
            .map(|j| {
                let mut v = x.center()[j] + x.remainder()[j] * rng.sym();
                for (row, ev) in e.iter().enumerate() {
                    v += x.generator_row(row)[j] * ev;
                }
                v
            })
            .collect();
        let p = softmax_eval(&lanes, &values);
        for j in 0..n {
            let tol = 1e-12;
            assert!(
                p[j] >= p_lo[j] - tol && p[j] <= p_hi[j] + tol,
                "softmax[{j}] = {} outside [{}, {}]",
                p[j],
                p_lo[j],
                p_hi[j]
            );
            checked += 1;
        }
    }
    assert!(checked >= 5000);

    // The box state encloses its box with one fresh symbol per element.
    let mut alloc = SymbolAllocator::new();
    let (bs, fresh) = box_state(&p_lo, &p_hi, &mut alloc).expect("box state");
    assert_eq!(fresh, p_lo.iter().zip(&p_hi).filter(|(l, h)| l < h).count());
    let bs_abs = bs.abs_generator_sum();
    let (cl, ch) = concretize(&bs, &bs_abs);
    for j in 0..n {
        assert!(cl[j] <= p_lo[j] && ch[j] >= p_hi[j], "box state [{j}]");
    }
}

// ---------------------------------------------------------------------------
// The miniature vit, end to end
// ---------------------------------------------------------------------------

/// The root map on the miniature vit, plain and rewritten by the fixed-index
/// softmax decomposition (Exp / ReduceSum / Reciprocal / MulBinary / Clip /
/// Sub / MulConstant / Linear(1->n)): admitted, self-checked, every box inside
/// ny's IBP box, tighter than IBP somewhere, and containing every sampled
/// value at every node.
#[test]
fn vit_zono_root_mini_vit_contains_samples_plain_and_fixed_index() {
    for rewrite in [false, true] {
        let mut graph = mini_vit(0x517E_0001);
        let input = mini_vit_input(0x517E_0002, 0.03);
        if rewrite {
            let report = graph
                .decompose_softmax_complex_fixed_index(&input)
                .expect("the fixed-index rewrite runs");
            assert_eq!(
                report.decomposed,
                vec!["attn".to_string()],
                "precondition: the softmax is decomposed (skipped {:?})",
                report.skipped
            );
        }
        let map = compute_vit_zono_root_map(&graph, &input, &test_plan(), None)
            .unwrap_or_else(|d| panic!("rewrite={rewrite}: declined: {} ({})", d.reason, d.census));
        assert!(
            map.stats.fresh_bilinear > 0,
            "precondition: MatMul remainders"
        );
        if rewrite {
            assert!(
                map.stats.fresh_band > 0,
                "precondition: Exp/Reciprocal bands"
            );
            assert!(map.census.contains("exp:"), "{}", map.census);
            assert!(map.census.contains("recip:"), "{}", map.census);
            assert!(map.census.contains("reduce:"), "{}", map.census);
        } else {
            assert!(map.stats.fresh_softmax > 0, "precondition: the softmax box");
            assert!(map.census.contains("softmax:1"), "{}", map.census);
        }
        assert!(map.census.contains("conv:1"), "{}", map.census);
        assert!(map.census.contains("bn:1"), "{}", map.census);
        assert!(map.census.contains("matmul:2"), "{}", map.census);
        assert!(map.self_check_elems > 0, "the self-check compared nothing");
        assert!(
            map.tightened_elems > 0,
            "rewrite={rewrite}: the zonotope never tightened IBP"
        );

        let ibp = graph
            .collect_node_bounds_with_engine(&input, None)
            .expect("ibp");
        for (name, entry) in &map.bounds {
            let reference = ibp.get(name).expect("published node has an IBP entry");
            assert_eq!(entry.shape(), reference.shape(), "`{name}` shape");
            for (((&l, &u), &il), &iu) in entry
                .lower()
                .iter()
                .zip(entry.upper().iter())
                .zip(reference.lower().iter())
                .zip(reference.upper().iter())
            {
                assert!(
                    l >= il && u <= iu,
                    "`{name}` widened IBP: [{l}, {u}] vs [{il}, {iu}]"
                );
            }
        }
        for (name, entry) in &map.baseline {
            assert!(
                ibp.contains_key(name),
                "baseline has a node IBP lacks: {name}"
            );
            assert_eq!(entry.shape(), ibp[name].shape());
        }
        assert_eq!(map.baseline.len(), ibp.len(), "the baseline is complete");

        let checked = sampled_containment(&graph, &input, &map.bounds, 64, 0x517E_0003);
        assert!(checked > 5000, "rewrite={rewrite}: checked only {checked}");
    }
}

/// The runtime self-check refuses (a) a transformer that encloses a
/// DIFFERENT function than the program's point semantics (the MatMul drops its
/// scale) and (b) a compiler that models a different function than ny's layer
/// (the program drops the scale): nothing is published either way. The clean
/// control is admitted.
#[test]
fn vit_zono_root_self_check_refuses_wrong_enclosures_and_mis_modelled_layers() {
    let graph = mini_vit(0x517E_0011);
    let input = mini_vit_input(0x517E_0012, 0.02);
    compute_vit_zono_root_map(&graph, &input, &test_plan(), None)
        .unwrap_or_else(|d| panic!("clean control declined: {}", d.reason));
    for injected in [
        fault::VitFault::MatMulIgnoreScale,
        fault::VitFault::CompileDropScale,
    ] {
        fault::set(injected);
        let outcome = compute_vit_zono_root_map(&graph, &input, &test_plan(), None);
        fault::set(fault::VitFault::None);
        match outcome {
            Err(decline) => assert!(
                matches!(decline.reason, ZonoRefusal::Internal(_)),
                "{injected:?}: refused for the wrong reason: {}",
                decline.reason
            ),
            Ok(_) => panic!("{injected:?}: a wrong enclosure was published"),
        }
    }
}

// ---------------------------------------------------------------------------
// The reference-collector hook
// ---------------------------------------------------------------------------

/// Dark by default: config default off, serde fills every field, no plan, a
/// fresh graph carries none, and an unarmed reference collection computes
/// NOTHING. Armed on the same (non-CROWN-IBP) route, the collection computes
/// the map once and is elementwise inside the dark map, strictly tighter
/// somewhere.
#[test]
fn vit_zono_root_is_dark_by_default_and_armed_is_shrink_only() {
    let default = BetaCrownConfig::default();
    assert!(!default.vit_zono_root);
    assert_eq!(default.vit_zono_root_max_mb, 8192);
    assert!((default.vit_zono_root_max_secs - 30.0).abs() < f32::EPSILON);
    assert!(BetaCrownConfig::vit_zono_root_max_mb_is_valid(
        default.vit_zono_root_max_mb
    ));
    assert!(!BetaCrownConfig::vit_zono_root_max_mb_is_valid(0));
    assert!(!BetaCrownConfig::vit_zono_root_max_secs_is_valid(0.0));
    assert!(!BetaCrownConfig::vit_zono_root_max_secs_is_valid(f32::NAN));
    assert!(vit_zono_root_plan(&default).is_none());
    let absent: BetaCrownConfig = serde_json::from_str("{}").expect("empty config");
    assert!(!absent.vit_zono_root);
    assert_eq!(absent.vit_zono_root_max_mb, 8192);
    assert!(GraphNetwork::new().vit_zono_root_plan().is_none());
    let armed_config = BetaCrownConfig {
        vit_zono_root: true,
        ..BetaCrownConfig::default()
    };
    let plan = vit_zono_root_plan(&armed_config).expect("armed plan");
    assert_eq!(plan.source, ny_levers::Source::Config);
    assert_ne!(plan.source, ny_levers::Source::LegacyEnv);
    assert_eq!(plan.max_bytes, 8192u64 << 20);

    let graph = mini_vit(0x517E_0021);
    let input = mini_vit_input(0x517E_0022, 0.03);
    let order = graph.exec_order().expect("order").to_vec();
    let config = AlphaCrownConfig::default();
    assert!(
        config.fix_interm_bounds,
        "the default route is not CROWN-IBP"
    );
    let (dark, _) = graph
        .collect_alpha_reference_bounds_with_engine_and_source(&input, &config, None, &order)
        .expect("dark collection");
    assert!(
        !cached_for(&graph, &input),
        "an unarmed collection must not compute the zonotope"
    );

    let mut armed_graph = graph;
    armed_graph.set_vit_zono_root_plan(Some(test_plan()));
    let (armed, _) = armed_graph
        .collect_alpha_reference_bounds_with_engine_and_source(&input, &config, None, &order)
        .expect("armed collection");
    assert!(
        cached_for(&armed_graph, &input),
        "precondition: the armed collection computed the map"
    );
    let mut tighter = 0usize;
    for (name, d) in &dark {
        let a = armed.get(name).expect("armed map keeps every node");
        assert_eq!(a.shape(), d.shape(), "`{name}` shape");
        for (((&al, &au), &dl), &du) in a
            .lower()
            .iter()
            .zip(a.upper().iter())
            .zip(d.lower().iter())
            .zip(d.upper().iter())
        {
            assert!(
                al >= dl && au <= du,
                "`{name}` widened: [{al}, {au}] vs [{dl}, {du}]"
            );
            if al > dl || au < du {
                tighter += 1;
            }
        }
    }
    assert_eq!(armed.len(), dark.len(), "the merge never adds nodes");
    assert!(tighter > 0, "the armed map never tightened the dark one");
}

/// The `fix_interm_bounds: false` route runs the CROWN-IBP collector on the
/// zonotope-seeded baseline; its (merged) map contains every sampled value.
#[test]
fn vit_zono_root_seeded_crown_ibp_reference_contains_samples() {
    let mut graph = mini_vit(0x517E_0031);
    let input = mini_vit_input(0x517E_0032, 0.03);
    graph.set_vit_zono_root_plan(Some(test_plan()));
    let order = graph.exec_order().expect("order").to_vec();
    let config = AlphaCrownConfig {
        fix_interm_bounds: false,
        ..AlphaCrownConfig::default()
    };
    let (map, _) = graph
        .collect_alpha_reference_bounds_with_engine_and_source(&input, &config, None, &order)
        .expect("seeded collection");
    assert!(
        cached_for(&graph, &input),
        "precondition: the map was computed"
    );
    let checked = sampled_containment(&graph, &input, &map, 48, 0x517E_0033);
    assert!(checked > 3000, "checked only {checked}");
}

// ---------------------------------------------------------------------------
// Refusals, cache, merge
// ---------------------------------------------------------------------------

/// An op outside the vit set refuses at compile time; the Base op set (every
/// `#fwd-root-screen` caller) still refuses the vit layers; a byte budget
/// below the first live state refuses instead of running.
#[test]
fn vit_zono_root_refuses_unsupported_ops_and_memory() {
    let mut tanh = mini_vit(0x517E_0041);
    tanh.add_node(node("squash", Layer::Tanh(TanhLayer::new()), &["head"]));
    tanh.set_output("squash");
    let input = mini_vit_input(0x517E_0042, 0.03);
    match compute_vit_zono_root_map(&tanh, &input, &test_plan(), None) {
        Err(decline) => {
            assert!(
                matches!(decline.reason, ZonoRefusal::UnsupportedLayer { .. }),
                "{}",
                decline.reason
            );
            assert_eq!(decline.census, "-");
        }
        Ok(_) => panic!("Tanh is outside the vit op set"),
    }
    let graph = mini_vit(0x517E_0043);
    assert!(
        matches!(
            DagProgram::compile_with_ibp(&graph, &input, None),
            Err(ZonoRefusal::UnsupportedLayer { .. })
        ),
        "the Base op set must keep refusing the vit layers"
    );
    DagProgram::compile_with_ibp_opset(&graph, &input, None, DagOpSet::Vit)
        .expect("the vit op set compiles the miniature vit");
    let tiny = VitZonoRootPlan {
        max_bytes: 1024,
        ..test_plan()
    };
    match compute_vit_zono_root_map(&graph, &input, &tiny, None) {
        Err(decline) => assert!(
            matches!(decline.reason, ZonoRefusal::Memory { .. }),
            "{}",
            decline.reason
        ),
        Ok(_) => panic!("a 1 KiB budget cannot admit the pass"),
    }
}

/// The cache serves an EXACT box only: the same box hits, a sub-box misses
/// and computes its own map.
#[test]
fn vit_zono_root_cache_serves_the_exact_box_only() {
    let graph = mini_vit(0x517E_0051);
    let input = mini_vit_input(0x517E_0052, 0.03);
    let plan = test_plan();
    let (first, hit) = vit_zono_root_map(&graph, &input, &plan, None);
    assert!(first.is_some() && !hit, "first call computes");
    let (second, hit) = vit_zono_root_map(&graph, &input, &plan, None);
    assert!(second.is_some() && hit, "same box hits");
    let lower = input.lower().clone();
    let mut upper = input.upper().clone();
    if let Some(v) = upper.iter_mut().next() {
        *v = f32::midpoint(*v, lower.iter().next().copied().unwrap_or(*v));
    }
    let child = BoundedTensor::new(lower, upper).expect("child box");
    let (third, hit) = vit_zono_root_map(&graph, &child, &plan, None);
    assert!(
        third.is_some() && !hit,
        "a different box must not be served"
    );
}

/// A deadline refusal depends on the caller's budget, not on the (graph, box):
/// it publishes nothing, is NOT cached, and a later call with time computes.
#[test]
fn vit_zono_root_deadline_refusal_is_not_cached() {
    let graph = mini_vit(0x517E_0061);
    let input = mini_vit_input(0x517E_0062, 0.03);
    let expired = std::time::Instant::now()
        .checked_sub(std::time::Duration::from_secs(1))
        .unwrap_or_else(std::time::Instant::now);
    let (refused, hit) = vit_zono_root_map(&graph, &input, &test_plan(), Some(expired));
    assert!(refused.is_none() && !hit, "an expired deadline must refuse");
    assert!(
        !cached_for(&graph, &input),
        "a budget-dependent refusal must not be cached"
    );
    let (computed, hit) = vit_zono_root_map(&graph, &input, &test_plan(), None);
    assert!(
        computed.is_some() && !hit,
        "a later call with time computes the map"
    );
    assert!(cached_for(&graph, &input));
}

/// A deadline refusal under the plan's OWN wall cap (no caller deadline, or a
/// later one) is a property of the (graph, box, plan), not of any caller's
/// remaining budget: it IS cached, so later collections on the same box do
/// not each burn another `max_secs` of the row's budget on the same answer.
/// The caller-bound case above stays uncached.
#[test]
fn vit_zono_root_cap_bound_deadline_refusal_is_cached() {
    let graph = mini_vit(0x517E_0071);
    let input = mini_vit_input(0x517E_0072, 0.03);
    let capped = VitZonoRootPlan {
        max_secs: 1e-9,
        ..test_plan()
    };
    match compute_vit_zono_root_map(&graph, &input, &capped, None) {
        Err(decline) => {
            assert_eq!(
                decline.reason,
                ZonoRefusal::Deadline,
                "precondition: a 1 ns cap must refuse on its deadline"
            );
            assert!(
                !decline.caller_budget_bound,
                "no caller deadline: the refusal is the plan's own cap"
            );
        }
        Ok(_) => panic!("a 1 ns cap cannot admit the pass"),
    }
    let later = std::time::Instant::now() + std::time::Duration::from_secs(3_600);
    match compute_vit_zono_root_map(&graph, &input, &capped, Some(later)) {
        Err(decline) => assert!(
            decline.reason == ZonoRefusal::Deadline && !decline.caller_budget_bound,
            "a caller deadline LATER than the cap does not make the refusal caller-bound: {}",
            decline.reason
        ),
        Ok(_) => panic!("a 1 ns cap cannot admit the pass"),
    }
    let (refused, hit) = vit_zono_root_map(&graph, &input, &capped, None);
    assert!(refused.is_none() && !hit, "the first lane call computes");
    assert!(
        cached_for(&graph, &input),
        "a cap-bound deadline refusal is cached"
    );
    let (again, hit) = vit_zono_root_map(&graph, &input, &capped, None);
    assert!(
        again.is_none() && hit,
        "the cached cap-bound refusal is served, not recomputed"
    );
}

/// The merge is shrink-only: a strictly inside candidate tightens, a disjoint
/// element keeps the CURRENT value, a shape mismatch keeps the whole entry,
/// and a node the map lacks is never added.
#[test]
fn vit_zono_root_merge_is_shrink_only() {
    let t = |lo: &[f32], hi: &[f32]| {
        BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[lo.len()]), lo.to_vec()).expect("lo"),
            ArrayD::from_shape_vec(IxDyn(&[hi.len()]), hi.to_vec()).expect("hi"),
        )
        .expect("tensor")
    };
    let mut map: HashMap<String, BoundedTensor> = HashMap::new();
    map.insert("a".to_string(), t(&[0.0, 0.0, 0.0], &[1.0, 1.0, 1.0]));
    map.insert("b".to_string(), t(&[0.0, 0.0], &[1.0, 1.0]));
    let mut zono: HashMap<String, BoundedTensor> = HashMap::new();
    zono.insert("a".to_string(), t(&[0.25, -1.0, 2.0], &[0.5, 2.0, 3.0]));
    zono.insert("b".to_string(), t(&[0.5], &[0.6]));
    zono.insert("c".to_string(), t(&[0.0], &[1.0]));
    let stats = merge_shrink_only(&mut map, &zono);
    assert_eq!(stats.nodes_merged, 1);
    assert_eq!(stats.elems_tightened, 1);
    assert_eq!(stats.disjoint_kept, 1);
    assert_eq!(stats.shape_skipped, 1);
    assert!(!map.contains_key("c"), "the merge never adds a node");
    let a = &map["a"];
    let lower: Vec<f32> = a.lower().iter().copied().collect();
    let upper: Vec<f32> = a.upper().iter().copied().collect();
    assert_eq!(lower, vec![0.25, 0.0, 0.0]);
    assert_eq!(upper, vec![0.5, 1.0, 1.0]);
    let b = &map["b"];
    assert_eq!(
        b.lower().iter().copied().collect::<Vec<f32>>(),
        vec![0.0, 0.0]
    );
}

/// #zono-exact-bn on the DAG path: the vit program's BatchNorm charges the
/// binary64 enclosure of the exact-real coefficients (the program's own
/// observable says so), with radii ~u relative that intersect the binary32
/// enclosure of the same real; the stripped twin takes the binary32 path
/// exactly as before; and the root map computed with the exact path passes
/// the runtime self-check and contains every sampled value.
#[test]
fn vit_zono_root_batch_norm_charges_the_exact_f64_enclosure() {
    let graph = mini_vit(0x517E_00B1);
    let input = mini_vit_input(0x517E_00B2, 0.03);
    let (exact, _) = DagProgram::compile_with_ibp_opset(&graph, &input, None, DagOpSet::Vit)
        .expect("the vit program compiles");
    assert_eq!(
        exact.batch_norm_coefficients().to_vec(),
        vec![("bn".to_string(), BatchNormCoefficients::ExactF64)]
    );
    let mut stripped = mini_vit(0x517E_00B1);
    let Layer::BatchNorm(bn) = &mut stripped.nodes.get_mut("bn").expect("bn node").layer else {
        panic!("precondition: `bn` is a BatchNorm");
    };
    assert!(
        bn.exact_affine.take().is_some(),
        "precondition: BatchNormLayer::new attached an exact enclosure"
    );
    let (binary32, _) = DagProgram::compile_with_ibp_opset(&stripped, &input, None, DagOpSet::Vit)
        .expect("the stripped vit program compiles");
    assert_eq!(
        binary32.batch_norm_coefficients().to_vec(),
        vec![("bn".to_string(), BatchNormCoefficients::Binary32)]
    );
    let bn_op = |program: &DagProgram| -> ChannelAffine {
        program
            .steps()
            .iter()
            .find_map(|step| match &step.op {
                DagOp::ChannelAffine(op) if step.node == "bn" => Some(op.clone()),
                _ => None,
            })
            .expect("the bn step")
    };
    let (e, b) = (bn_op(&exact), bn_op(&binary32));
    assert_eq!(
        e.scale.len(),
        30,
        "BatchNorm over the [6, 5] token-major view"
    );
    let (mut sum_exact, mut sum_binary32) = (0.0f64, 0.0f64);
    for j in 0..e.scale.len() {
        assert!(
            (e.scale[j] - b.scale[j]).abs() <= (e.scale_err[j] + b.scale_err[j]) * 1.01,
            "[{j}]: disjoint scale enclosures of the same real"
        );
        assert!(
            (e.shift[j] - b.shift[j]).abs() <= (e.shift_err[j] + b.shift_err[j]) * 1.01,
            "[{j}]: disjoint shift enclosures of the same real"
        );
        assert!(
            e.scale_err[j] <= 32.0 * f64::EPSILON * e.scale[j].abs() + 1e-300,
            "[{j}]: exact scale radius {} is not ~u relative",
            e.scale_err[j]
        );
        sum_exact += e.scale_err[j] + e.shift_err[j];
        sum_binary32 += b.scale_err[j] + b.shift_err[j];
    }
    assert!(
        sum_exact < 1e-4 * sum_binary32,
        "exact radii {sum_exact:e} are not orders below binary32 {sum_binary32:e}"
    );
    let map = compute_vit_zono_root_map(&graph, &input, &test_plan(), None)
        .unwrap_or_else(|d| panic!("the exact-BatchNorm map declined: {}", d.reason));
    assert!(map.census.contains("bn:1"), "{}", map.census);
    assert!(map.self_check_elems > 0, "the self-check compared nothing");
    let checked = sampled_containment(&graph, &input, &map.bounds, 32, 0x517E_00B3);
    assert!(checked > 1_000, "checked only {checked}");
}
