// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #bi-patch soundness, dominance, refusal and dark-default tests.
//!
//! A child of `gpu_conv_tests` only to reuse its exact dyadic-rational forward
//! oracle (`exact_tensors`), sample points (`box_points`) and the resnet-shaped
//! fixture; nothing here runs on a device.
//!
//! * `bi_patch_coefficients_are_the_dense_coefficients` — the patch walk's
//!   input-row coefficients and bias EQUAL the dense pass's (up to the sign of
//!   zero), its error lane and bias error are DOMINATED entrywise — Lower and
//!   Upper, root gates and an `#interm-alpha` override, Outward and Parity.
//! * `bi_patch_never_looser_than_dense` — `lb_patch >= lb_dense`,
//!   `ub_patch <= ub_dense` exactly, within rounding, at every radius.
//! * `bi_patch_bounds_contain_exact_preactivations` — exact-rational
//!   containment at sampled box points, point boxes included.
//! * `patch_plan_frames_cover_dense_support` — every nonzero dense coefficient
//!   lies on a real row of the column's frame; a shrunk frame is refused.
//! * `patch_crossover_handoff_is_exact` — every crossover ratio: same
//!   coefficients, `dense <= converted <= all-patch` bounds.
//! * `patch_fused_pair_matches_two_passes` — bit-identical.
//! * `patch_refuses_what_it_cannot_express` — transposed conv, unequal-stride
//!   union, beta / per-column beta / infeasible overrides, bad columns,
//!   non-relu and head seeds, `no_gain`; an envelope root still gets PerCol.
//! * `patch_virtual_convs_have_no_border_taps_and_dense_envelopes`.
//! * `patch_cost_model_matches_counted_macs`.
//! * `bi_patch_phase_enters_and_stays_contained`,
//!   `bi_patch_phase_is_dark_by_default` and
//!   `bi_patch_refused_plans_fall_back_bit_identically` — the phase.
//! * `joint_refresh_patch_matches_dense_refresh`.

use ndarray::Array2;
use num_rational::BigRational;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use crate::margin_row::backward_interm::window_tests::chain_spec;
use crate::margin_row::backward_interm::{
    refresh_root_intermediates, refresh_root_intermediates_with_patch, BiState,
};
use crate::margin_row::engine::{
    BackwardEngine, BetaSplit, DomainGates, GateVecs, LaneDir, PassOut, PcBetaCol, Seed,
};
use crate::margin_row::errlane::ErrLane;
use crate::margin_row::net::{TwinNet, TwinOp};
use crate::margin_row::patch::{PatchCfg, PatchPlan, PatchWalk, DEFAULT_RATIO};
use crate::margin_row::root::{LayerGates, RootGates};
use crate::margin_row::rounding::{RoundMode, UNIT};
use crate::margin_row::spec::{TwinOpSpec, TwinSpec};
use crate::margin_row::{BackwardIntermRuntimePolicy, BiPatchRuntimePolicy};

use super::super::tiny_spec;
use super::{box_points, conv_op, exact_tensors, q, resnetish_spec, uniform};

// ---------------------------------------------------------------------------
//  Fixtures and helpers
// ---------------------------------------------------------------------------

/// A seeded random conv chain: `c0 x hw x hw` input, an optional input
/// ChannelAffine, 2-3 convs with kernels in {1, 3, 4, 5}, strides in {1, 2}
/// and independent (asymmetric, possibly zero) pads, each followed by a relu,
/// and the `Gemm -> Relu -> Gemm` head. `folded` gives every conv / affine a
/// nonzero certified fold charge.
fn random_chain(rng: &mut StdRng, affine: bool, folded: bool) -> TwinSpec {
    let c0 = rng.random_range(1..=3usize);
    let hw = rng.random_range(5..=9usize);
    let (mut c, mut h, mut w) = (c0, hw, hw);
    let mut ops: Vec<TwinOpSpec> = Vec::new();
    let mut cur = 0usize;
    if affine {
        let scale: Vec<f64> = (0..c).map(|_| rng.random_range(0.5..1.5)).collect();
        let shift = uniform(rng, c, 0.1);
        ops.push(TwinOpSpec::ChannelAffine {
            input: 0,
            scale,
            shift,
            scale_rel_err: if folded { 1e-16 } else { 0.0 },
            shift_err: vec![if folded { 1e-18 } else { 0.0 }; c],
            shape: (c, h, w),
        });
        cur = ops.len();
    }
    let n_conv = rng.random_range(2..=3usize);
    for _ in 0..n_conv {
        let mut k = [1usize, 3, 4, 5][rng.random_range(0..4usize)];
        let s = rng.random_range(1..=2usize);
        let pads = (
            rng.random_range(0..=2usize),
            rng.random_range(0..=2usize),
            rng.random_range(0..=2usize),
            rng.random_range(0..=2usize),
        );
        if h + pads.0 + pads.2 < k || w + pads.1 + pads.3 < k {
            k = 1;
        }
        let oh = (h + pads.0 + pads.2 - k) / s + 1;
        let ow = (w + pads.1 + pads.3 - k) / s + 1;
        let co = rng.random_range(1..=4usize);
        let wscale = 1.5 / ((c * k * k) as f64).sqrt();
        ops.push(TwinOpSpec::Conv {
            input: cur,
            weight: uniform(rng, co * c * k * k, wscale),
            bias: uniform(rng, co, 0.2),
            bias_err: vec![if folded { 1e-18 } else { 0.0 }; co],
            weight_rel_err: if folded { 1e-15 } else { 0.0 },
            kernel: (co, c, k, k),
            stride: (s, s),
            pads,
            ishape: (c, h, w),
            oshape: (co, oh, ow),
        });
        cur = ops.len();
        ops.push(TwinOpSpec::Relu { input: cur });
        cur = ops.len();
        (c, h, w) = (co, oh, ow);
    }
    let n_flat = c * h * w;
    ops.push(TwinOpSpec::Flatten { input: cur });
    cur = ops.len();
    ops.push(TwinOpSpec::Gemm {
        input: cur,
        weight: uniform(rng, 3 * n_flat, 1.5 / (n_flat as f64).sqrt()),
        bias: uniform(rng, 3, 0.1),
        bias_err: Vec::new(),
        weight_rel_err: 0.0,
        shape: (3, n_flat),
    });
    cur = ops.len();
    ops.push(TwinOpSpec::Relu { input: cur });
    cur = ops.len();
    ops.push(TwinOpSpec::Gemm {
        input: cur,
        weight: uniform(rng, 2 * 3, 1.0),
        bias: uniform(rng, 2, 0.1),
        bias_err: Vec::new(),
        weight_rel_err: 0.0,
        shape: (2, 3),
    });
    TwinSpec {
        n_in: c0 * hw * hw,
        ops,
    }
}

/// `(name, spec, box half-width)`: the croppable chain (input ChannelAffine),
/// the residual tiny net (stride-2 stem, identity skip), the resnet-shaped
/// net (residual block, stride-2 downsample, 1x1 conv) and seeded random
/// chains with and without affine / fold charges.
fn fixtures() -> Vec<(String, TwinSpec, f64)> {
    let mut rng = StdRng::seed_from_u64(0xB1_9A7C_0001);
    let mut out = vec![
        ("chain_2x8x8".to_string(), chain_spec(&mut rng, 0.6), 0.35),
        ("tiny_2x6x6".to_string(), tiny_spec(&mut rng, 0.6), 0.3),
        (
            "resnetish_3x10x10".to_string(),
            resnetish_spec(&mut rng, (3, 10, 10), 6, 8),
            0.2,
        ),
    ];
    for (i, (affine, folded)) in [(false, false), (true, false), (false, true), (true, true)]
        .into_iter()
        .enumerate()
    {
        out.push((
            format!("random_{i}_affine{affine}_folded{folded}"),
            random_chain(&mut rng, affine, folded),
            0.3,
        ));
    }
    out
}

fn boxed(center: &[f64], rad: f64) -> (Vec<f64>, Vec<f64>) {
    (
        center.iter().map(|c| c - rad).collect(),
        center.iter().map(|c| c + rad).collect(),
    )
}

/// Tensor id of trunk relu op `relu_op`'s pre-activation.
fn pre_tensor(net: &TwinNet, relu_op: usize) -> usize {
    let TwinOp::Relu { input, .. } = &net.ops[relu_op] else {
        panic!("op {relu_op} is not a relu");
    };
    *input
}

/// `(C, H, W)` of the seed tensor of trunk layer `li`, from the plan.
fn seed_shape(plan: &PatchPlan) -> (usize, usize, usize) {
    let f = plan.geo.frames[plan.geo.seed_tensor].expect("the seed tensor has a frame");
    (f.c, f.h, f.w)
}

/// Column sets of one layer: the top-left corner, a top-edge column, an
/// interior column, the bottom-right corner, and the whole layer.
fn column_sets((c, h, w): (usize, usize, usize)) -> Vec<Vec<usize>> {
    let n = c * h * w;
    let at = |ch: usize, y: usize, x: usize| ch * h * w + y * w + x;
    vec![
        vec![0],
        vec![at(0, 0, w / 2)],
        vec![at(c - 1, h / 2, w / 2)],
        vec![n - 1],
        (0..n).collect(),
    ]
}

/// One dense identity-seeded prefix pass on the unchanged engine.
fn dense_pass(
    net: &TwinNet,
    root: &RootGates,
    relu_op: usize,
    cols: &[usize],
    dir: LaneDir,
    dom: Option<&DomainGates>,
) -> PassOut {
    let n = net.tsize[pre_tensor(net, relu_op)];
    let mut s = Array2::<f64>::zeros((n, cols.len()));
    for (c, &j) in cols.iter().enumerate() {
        s[[j, c]] = 1.0;
    }
    BackwardEngine::new(net, root)
        .run_prefix_dom(&Seed { s, e: None }, relu_op, dir, dom)
        .expect("the dense prefix pass is certified")
}

/// One single-lane patch walk.
fn patch_walk(
    plan: &PatchPlan,
    net: &TwinNet,
    root: &RootGates,
    cols: &[usize],
    lower: bool,
    dom: Option<&DomainGates>,
) -> PatchWalk {
    plan.walk(net, root, cols, &vec![lower; cols.len()], dom, None)
        .expect("the plan expresses this request")
        .expect("the patch walk is certified")
}

/// The chord-parallel `#interm-alpha` candidate (policy 0): every unstable
/// prefix neuron's lower slope set to its upper chord's, `ms` re-derived.
fn chord_parallel_dom(root: &RootGates, prefix_len: usize) -> DomainGates {
    let mut dg = DomainGates::default();
    for li in 0..prefix_len {
        let layer = &root.layers[li];
        let mut alpha = layer.alpha.clone();
        for &j in &layer.unst {
            alpha[j] = layer.s[j].clamp(0.0, 1.0);
        }
        let ms: Vec<f64> = alpha.iter().zip(&layer.s).map(|(a, s)| a.max(*s)).collect();
        dg.layers.insert(
            li,
            GateVecs {
                alpha,
                s: layer.s.clone(),
                c: layer.c.clone(),
                ms,
            },
        );
    }
    dg
}

/// EQUAL coefficients and bias (the sign of zero aside), DOMINATED error lane.
///
/// And CARRIED, not dropped: an upper check alone is
/// satisfied by a walk that zeroes its error lane, and the concretize slack's
/// `gamma(n_in + 16) (|A||x| + |b|)` term then still covers these small nets'
/// real rounding, so neither containment nor never-looser would notice. Two
/// lower checks close that:
/// * coefficient error: the dense lane exceeds the patch lane only by the
///   out-of-support charges (`slack16(0)` = one subnormal per relu, carried
///   through `|W|` sums), which stay subnormal-scale (< 1e-300 here) and are
///   absorbed by any normal-scale row, so `E_patch >= E_dense - 1e-300`;
/// * bias error: every step that moves `b` charges `2u|b|` right after it and
///   `eb` never shrinks, so `eb >= 2u|b|` on the final bias.
fn assert_patch_matches_dense(what: &str, walk: &PatchWalk, dense: &PassOut, n_in: usize) {
    let (a, e) = PatchPlan::scatter_input(walk, n_in);
    assert_eq!(a.dim(), dense.a.dim(), "{what}: coefficient shape");
    for ((i, c), &v) in a.indexed_iter() {
        let d = dense.a[[i, c]];
        assert!(
            v == d,
            "{what}: coefficient ({i},{c}) patch {v:e} != dense {d:e}"
        );
    }
    for (c, (&bp, &bd)) in walk.b.iter().zip(&dense.b).enumerate() {
        assert!(
            bp == bd,
            "{what}: bias column {c} patch {bp:e} != dense {bd:e}"
        );
    }
    match (e.as_ref(), dense.e.as_ref()) {
        (Some(ep), Some(ed)) => {
            for ((i, c), &v) in ep.indexed_iter() {
                let d = ed[[i, c]];
                assert!(
                    v <= d,
                    "{what}: error ({i},{c}) patch {v:e} exceeds dense {d:e}"
                );
                assert!(
                    v >= d - 1e-300,
                    "{what}: error ({i},{c}) patch {v:e} DROPPED below dense {d:e}"
                );
            }
        }
        (None, None) => {}
        _ => panic!("{what}: the patch and dense error lanes disagree on presence"),
    }
    let outward = e.is_some();
    for (c, (&p, &d)) in walk.eb.iter().zip(&dense.eb).enumerate() {
        assert!(
            p <= d,
            "{what}: bias error column {c} patch {p:e} exceeds dense {d:e}"
        );
        if outward {
            let bp = walk.b[c];
            assert!(
                p >= 2.0 * UNIT * bp.abs(),
                "{what}: bias error column {c} patch {p:e} DROPPED below 2u|b| (b = {bp:e})"
            );
        }
    }
}

/// `[lb[k], ub[k]]` contains the EXACT pre-activation at every sample point.
fn assert_contains(
    what: &str,
    lb: &[f64],
    ub: &[f64],
    cols: &[usize],
    exact: &[Vec<Vec<BigRational>>],
    tensor: usize,
    checks: &mut usize,
) {
    assert_eq!(lb.len(), cols.len(), "{what}: lower bound count");
    assert_eq!(ub.len(), cols.len(), "{what}: upper bound count");
    for (p, tensors) in exact.iter().enumerate() {
        let z = &tensors[tensor];
        for (k, &j) in cols.iter().enumerate() {
            let (lo_q, hi_q) = (q(lb[k]), q(ub[k]));
            assert!(
                lo_q <= z[j] && z[j] <= hi_q,
                "CONTAINMENT FALSIFIED: {what}: neuron {j} at point {p}: patch bounds \
                 [{:e}, {:e}] do not enclose the exact pre-activation",
                lb[k],
                ub[k]
            );
            *checks += 1;
        }
    }
}

/// A fixed, deterministic phase policy with patch mode as given.
fn phase_policy(patch: BiPatchRuntimePolicy, interm_alpha: usize) -> BackwardIntermRuntimePolicy {
    BackwardIntermRuntimePolicy {
        enabled: true,
        secs: 1_000.0,
        chunk: 16,
        chunk_secs: 0.0,
        topk: 4_096,
        interm_alpha,
        window: false,
        gpu: false,
        patch,
    }
}

fn armed(ratio: f64) -> BiPatchRuntimePolicy {
    BiPatchRuntimePolicy {
        enabled: true,
        ratio,
        mb: 256,
    }
}

fn prefix_gates(root: &RootGates, li: usize) -> Vec<LayerGates> {
    root.layers[..li]
        .iter()
        .map(|g| LayerGates {
            op: g.op,
            n: g.n,
            l: g.l.clone(),
            u: g.u.clone(),
            alpha: g.alpha.clone(),
            s: g.s.clone(),
            c: g.c.clone(),
            ms: g.ms.clone(),
            unst: g.unst.clone(),
            clip_rows: None,
        })
        .collect()
}

/// Run the phase on trunk layer `li` of `root` and return its published box.
fn run_layer(st: &mut BiState, net: &TwinNet, root: &RootGates, li: usize) -> (Vec<f64>, Vec<f64>) {
    let mut prefix = prefix_gates(root, li);
    let (mut l, mut u) = (root.layers[li].l.clone(), root.layers[li].u.clone());
    st.tighten_layer(
        net,
        RoundMode::Outward,
        &root.mid,
        &root.rad,
        &root.xabs,
        &root.lo,
        &root.hi,
        &mut prefix,
        root.layers[li].op,
        &mut l,
        &mut u,
    );
    assert_eq!(prefix.len(), li, "the prefix gates must be handed back");
    (l, u)
}

/// Trunk layers past the affine-only layer 0 that have an unstable neuron.
fn working_layers(root: &RootGates) -> Vec<usize> {
    (1..root.layers.len())
        .filter(|&li| {
            let layer = &root.layers[li];
            layer
                .l
                .iter()
                .zip(&layer.u)
                .any(|(&a, &b)| a < 0.0 && b > 0.0)
        })
        .collect()
}

fn assert_bits(what: &str, a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    for (k, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{what}: entry {k}: {x:e} vs {y:e}"
        );
    }
}

// ---------------------------------------------------------------------------
//  Tests
// ---------------------------------------------------------------------------

/// THE LOAD-BEARING THEOREM. On every fixture, trunk layer >= 1, column set
/// (corner, edge, interior, whole layer), lane, gate source (root gates and a
/// chord-parallel `#interm-alpha` override) and mode (Outward and Parity):
/// the patch walk's input-row coefficients and bias EQUAL the dense pass's,
/// and (Outward) its error lane and bias error are DOMINATED entrywise. Both
/// the all-patch walk and the default-ratio walk (which converts on some
/// fixtures) are checked. VOID guard: a strictly smaller walk must occur.
#[test]
fn bi_patch_coefficients_are_the_dense_coefficients() {
    let mut rng = StdRng::seed_from_u64(0xB1_9A7C_0002);
    let mut strict = 0usize;
    let mut converted = 0usize;
    let mut compared = 0usize;
    for (name, spec, rad) in fixtures() {
        let net = TwinNet::compile(&spec).expect("compile");
        let center: Vec<f64> = (0..spec.n_in)
            .map(|_| rng.random_range(-rad..rad))
            .collect();
        let (lo, hi) = boxed(&center, rad);
        for mode in [RoundMode::Outward, RoundMode::Parity] {
            let root = RootGates::build(&net, &lo, &hi, mode, None).expect("root");
            for li in 1..root.layers.len() {
                let relu_op = root.layers[li].op;
                let dom = chord_parallel_dom(&root, li);
                for ratio in [f64::INFINITY, DEFAULT_RATIO] {
                    let plan = match PatchPlan::build(&net, relu_op, ratio) {
                        Ok(plan) => plan,
                        Err(why) => {
                            assert_eq!(
                                why, "no_gain",
                                "{name} layer={li}: only the no-gain refusal is expected here"
                            );
                            continue;
                        }
                    };
                    strict += usize::from(plan.geo.rows_col_patch < plan.geo.rows_col_dense);
                    converted += usize::from(plan.geo.convert_at.is_some());
                    for cols in column_sets(seed_shape(&plan)) {
                        for (lower, dir) in [(true, LaneDir::Lower), (false, LaneDir::Upper)] {
                            for d in [None, Some(&dom)] {
                                let what = format!(
                                    "{name} {mode:?} layer={li} ratio={ratio} cols={} \
                                     lower={lower} dom={}",
                                    cols.len(),
                                    d.is_some()
                                );
                                let walk = patch_walk(&plan, &net, &root, &cols, lower, d);
                                let dense = dense_pass(&net, &root, relu_op, &cols, dir, d);
                                assert_patch_matches_dense(&what, &walk, &dense, net.n_in);
                                compared += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(strict > 0, "no plan walked fewer rows than dense (VOID)");
    assert!(
        converted > 0,
        "no plan exercised the dense-tail crossover (VOID)"
    );
    assert!(compared > 100, "only {compared} comparisons ran");
}

/// NEVER LOOSER, EQUAL UP TO ROUNDING: at radii {0 (a point box), 1e-9, the
/// fixture's, ~1 (most neurons unstable)}, every patch pair's bounds sit
/// inside the dense pair's exactly and within `1e-9 * (1 + |bound|)`; a
/// fused pair and a dense pair over the whole layer agree on every column.
#[test]
fn bi_patch_never_looser_than_dense() {
    let mut rng = StdRng::seed_from_u64(0xB1_9A7C_0003);
    let mut pairs = 0usize;
    for (name, spec, rad0) in fixtures() {
        let net = TwinNet::compile(&spec).expect("compile");
        let center: Vec<f64> = (0..spec.n_in)
            .map(|_| rng.random_range(-rad0..rad0))
            .collect();
        for rad in [0.0, 1e-9, rad0, 1.0] {
            let (lo, hi) = boxed(&center, rad);
            let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
            for li in 1..root.layers.len() {
                let relu_op = root.layers[li].op;
                let Ok(plan) = PatchPlan::build(&net, relu_op, DEFAULT_RATIO) else {
                    continue;
                };
                for cols in column_sets(seed_shape(&plan)) {
                    let what = format!("{name} rad={rad:e} layer={li} cols={}", cols.len());
                    let pair = plan
                        .prefix_bounds(&net, &root, &cols, None, None)
                        .expect("expressible")
                        .expect("certified");
                    let eng = BackwardEngine::new(&net, &root);
                    let dl = eng.concretize_lower(&dense_pass(
                        &net,
                        &root,
                        relu_op,
                        &cols,
                        LaneDir::Lower,
                        None,
                    ));
                    let du = eng.concretize_upper(&dense_pass(
                        &net,
                        &root,
                        relu_op,
                        &cols,
                        LaneDir::Upper,
                        None,
                    ));
                    for k in 0..cols.len() {
                        assert!(
                            pair.lb[k] >= dl[k] && pair.ub[k] <= du[k],
                            "{what} col {}: patch [{:e}, {:e}] is LOOSER than dense [{:e}, {:e}]",
                            cols[k],
                            pair.lb[k],
                            pair.ub[k],
                            dl[k],
                            du[k]
                        );
                        let tol = 1e-9 * (1.0 + dl[k].abs().max(du[k].abs()));
                        assert!(
                            pair.lb[k] - dl[k] <= tol && du[k] - pair.ub[k] <= tol,
                            "{what} col {}: patch drifted beyond rounding from dense",
                            cols[k]
                        );
                        assert!(pair.lb[k] <= pair.ub[k], "{what}: crossed patch bounds");
                    }
                    pairs += 1;
                }
            }
        }
    }
    assert!(pairs > 50, "only {pairs} patch pairs ran");
}

/// EXACT-RATIONAL CONTAINMENT. Every patch bound of every neuron of every
/// trunk layer >= 1 encloses the EXACT pre-activation (dyadic-rational forward
/// of the stored parameters) at the box's center, both corners, random
/// corners and interior points — at a point box too, where only the certified
/// rounding separates the bound from the truth, so a missing charge would
/// falsify here first. The whole layer runs in 32-column fused chunks.
#[test]
fn bi_patch_bounds_contain_exact_preactivations() {
    let mut rng = StdRng::seed_from_u64(0xB1_9A7C_0004);
    let mut checks = 0usize;
    for (name, spec, rad0) in fixtures() {
        let net = TwinNet::compile(&spec).expect("compile");
        let center: Vec<f64> = (0..spec.n_in)
            .map(|_| rng.random_range(-rad0..rad0))
            .collect();
        let radii: Vec<f64> = if name.starts_with("resnetish") {
            vec![0.0, rad0]
        } else {
            vec![0.0, 1e-9, rad0, 1.0]
        };
        for &rad in &radii {
            let (lo, hi) = boxed(&center, rad);
            let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
            let extra = if rad > 0.0 { 2 } else { 0 };
            let exact: Vec<Vec<Vec<BigRational>>> = box_points(&mut rng, &lo, &hi, extra)
                .iter()
                .map(|x| exact_tensors(&spec, x))
                .collect();
            for li in 1..root.layers.len() {
                let (relu_op, n) = (root.layers[li].op, root.layers[li].n);
                let tensor = pre_tensor(&net, relu_op);
                for ratio in [DEFAULT_RATIO, f64::INFINITY] {
                    let Ok(plan) = PatchPlan::build(&net, relu_op, ratio) else {
                        continue;
                    };
                    let all: Vec<usize> = (0..n).collect();
                    for part in all.chunks(32) {
                        let what = format!("{name} rad={rad:e} layer={li} ratio={ratio}");
                        let pair = plan
                            .prefix_bounds(&net, &root, part, None, None)
                            .expect("expressible")
                            .expect("certified");
                        assert_contains(
                            &what,
                            &pair.lb,
                            &pair.ub,
                            part,
                            &exact,
                            tensor,
                            &mut checks,
                        );
                    }
                }
            }
        }
    }
    assert!(checks > 10_000, "only {checks} exact checks ran");
}

/// SUPPORT COVERAGE. For every single column, every NONZERO coefficient of
/// the dense pass at the input lies on a real row of the column's input frame
/// (the support lemma the whole restriction argument rests on), on residual
/// unions and stride-2 frames alike. And the walk FAILS CLOSED when a frame is
/// deliberately shrunk below a column's field: `None`, never a truncation.
#[test]
fn patch_plan_frames_cover_dense_support() {
    let mut rng = StdRng::seed_from_u64(0xB1_9A7C_0005);
    let mut covered = 0usize;
    let mut refused = 0usize;
    for (name, spec, rad) in fixtures() {
        let net = TwinNet::compile(&spec).expect("compile");
        let center: Vec<f64> = (0..spec.n_in)
            .map(|_| rng.random_range(-rad..rad))
            .collect();
        let (lo, hi) = boxed(&center, rad);
        let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
        for li in 1..root.layers.len() {
            let relu_op = root.layers[li].op;
            let Ok(plan) = PatchPlan::build(&net, relu_op, f64::INFINITY) else {
                continue;
            };
            let (c, h, w) = seed_shape(&plan);
            let n = c * h * w;
            for j in (0..n).step_by((n / 16).max(1)) {
                let dense = dense_pass(&net, &root, relu_op, &[j], LaneDir::Lower, None);
                let real: std::collections::BTreeSet<usize> = plan
                    .real_rows(0, j)
                    .expect("the walk reaches the input")
                    .into_iter()
                    .collect();
                for i in 0..net.n_in {
                    if dense.a[[i, 0]] != 0.0 {
                        assert!(
                            real.contains(&i),
                            "{name} layer={li} col={j}: dense coefficient at input {i} lies \
                             outside the column's frame"
                        );
                        covered += 1;
                    }
                }
            }
            // Shrink the input frame to ONE spatial position: a column whose
            // real field spans several positions now has real rows outside
            // the frame, and the walk must refuse rather than drop them.
            let mid = (c - 1) * h * w + (h / 2) * w + w / 2;
            let field = plan
                .real_rows(0, mid)
                .expect("the walk reaches the input")
                .len();
            let f0 = plan.geo.frames[0].expect("the input frame");
            if field > f0.c {
                let mut shrunk = PatchPlan::build(&net, relu_op, f64::INFINITY).expect("plan");
                if let Some(f) = shrunk.geo.frames[0].as_mut() {
                    f.ph = 1;
                    f.pw = 1;
                }
                assert!(
                    shrunk
                        .walk(&net, &root, &[mid], &[true], None, None)
                        .is_none(),
                    "{name} layer={li}: a shrunk frame must be refused, not truncated"
                );
                refused += 1;
            }
        }
    }
    assert!(covered > 1_000, "only {covered} dense coefficients checked");
    assert!(refused > 0, "the shrunk-frame refusal never ran (VOID)");
}

/// CROSSOVER. Forcing the dense-tail conversion onto every reachable op
/// (ratios from 0.05 up to an all-patch walk): the coefficients are the dense
/// ones at every ratio, and the bounds are ordered
/// `dense <= converted <= all-patch` (lower; reversed for upper) — the dense
/// tail can only carry MORE error (the dense pass's subnormal charges outside
/// the field), never less than the pure patch walk.
#[test]
fn patch_crossover_handoff_is_exact() {
    let mut rng = StdRng::seed_from_u64(0xB1_9A7C_0006);
    let mut conversions = std::collections::BTreeSet::new();
    for (name, spec, rad) in fixtures() {
        let net = TwinNet::compile(&spec).expect("compile");
        let center: Vec<f64> = (0..spec.n_in)
            .map(|_| rng.random_range(-rad..rad))
            .collect();
        let (lo, hi) = boxed(&center, rad);
        let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
        for li in 1..root.layers.len() {
            let relu_op = root.layers[li].op;
            let Ok(pure) = PatchPlan::build(&net, relu_op, f64::INFINITY) else {
                continue;
            };
            assert_eq!(
                pure.geo.convert_at, None,
                "an infinite ratio never converts"
            );
            let cols: Vec<usize> = column_sets(seed_shape(&pure))
                .into_iter()
                .flatten()
                .collect();
            let mut cols = cols;
            cols.sort_unstable();
            cols.dedup();
            let base = pure
                .prefix_bounds(&net, &root, &cols, None, None)
                .expect("expressible")
                .expect("certified");
            let eng = BackwardEngine::new(&net, &root);
            let dl = eng.concretize_lower(&dense_pass(
                &net,
                &root,
                relu_op,
                &cols,
                LaneDir::Lower,
                None,
            ));
            let du = eng.concretize_upper(&dense_pass(
                &net,
                &root,
                relu_op,
                &cols,
                LaneDir::Upper,
                None,
            ));
            for ratio in [0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0] {
                let Ok(plan) = PatchPlan::build(&net, relu_op, ratio) else {
                    continue;
                };
                let what = format!("{name} layer={li} ratio={ratio}");
                if let Some(t) = plan.geo.convert_at {
                    conversions.insert((name.clone(), li, t));
                }
                let pair = plan
                    .prefix_bounds(&net, &root, &cols, None, None)
                    .expect("expressible")
                    .expect("certified");
                for k in 0..cols.len() {
                    assert!(
                        dl[k] <= pair.lb[k] && pair.lb[k] <= base.lb[k],
                        "{what} col {}: lower {:e} not in [dense {:e}, all-patch {:e}]",
                        cols[k],
                        pair.lb[k],
                        dl[k],
                        base.lb[k]
                    );
                    assert!(
                        base.ub[k] <= pair.ub[k] && pair.ub[k] <= du[k],
                        "{what} col {}: upper {:e} not in [all-patch {:e}, dense {:e}]",
                        cols[k],
                        pair.ub[k],
                        base.ub[k],
                        du[k]
                    );
                }
                for (lower, dir) in [(true, LaneDir::Lower), (false, LaneDir::Upper)] {
                    let walk = patch_walk(&plan, &net, &root, &cols, lower, None);
                    let dense = dense_pass(&net, &root, relu_op, &cols, dir, None);
                    assert_patch_matches_dense(
                        &format!("{what} lower={lower}"),
                        &walk,
                        &dense,
                        net.n_in,
                    );
                }
            }
        }
    }
    assert!(
        conversions.len() >= 3,
        "the crossover landed on too few distinct tensors (VOID): {conversions:?}"
    );
}

/// The fused Lower+Upper walk is bit-identical to two single-lane walks.
#[test]
fn patch_fused_pair_matches_two_passes() {
    let mut rng = StdRng::seed_from_u64(0xB1_9A7C_0007);
    let mut pairs = 0usize;
    for (name, spec, rad) in fixtures() {
        let net = TwinNet::compile(&spec).expect("compile");
        let center: Vec<f64> = (0..spec.n_in)
            .map(|_| rng.random_range(-rad..rad))
            .collect();
        let (lo, hi) = boxed(&center, rad);
        let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
        for li in 1..root.layers.len() {
            let Ok(plan) = PatchPlan::build(&net, root.layers[li].op, DEFAULT_RATIO) else {
                continue;
            };
            let dom = chord_parallel_dom(&root, li);
            for cols in column_sets(seed_shape(&plan)) {
                for d in [None, Some(&dom)] {
                    let what = format!("{name} layer={li} cols={} dom={}", cols.len(), d.is_some());
                    let fused = plan
                        .prefix_bounds(&net, &root, &cols, d, None)
                        .expect("expressible")
                        .expect("certified");
                    assert_eq!(fused.fused_cols, 2 * cols.len());
                    let lower = patch_walk(&plan, &net, &root, &cols, true, d);
                    let upper = patch_walk(&plan, &net, &root, &cols, false, d);
                    let lb = PatchPlan::concretize(&root, &lower, &vec![true; cols.len()]);
                    let ub = PatchPlan::concretize(&root, &upper, &vec![false; cols.len()]);
                    assert_bits(&format!("{what} lower"), &fused.lb, &lb);
                    assert_bits(&format!("{what} upper"), &fused.ub, &ub);
                    pairs += 1;
                }
            }
        }
    }
    assert!(pairs > 20, "only {pairs} fused pairs compared");
}

/// FAIL-CLOSED. The planner refuses a transposed conv, a union of unequal
/// anchor strides, a non-relu seed, a head seed and a no-gain crossover; the
/// walk refuses a shared-beta, per-column-beta or `infeasible` override, an
/// out-of-range column and an empty request; a root whose error lane is the
/// envelope still gets the (tightest) per-column lane, never looser than the
/// dense envelope pair.
#[test]
fn patch_refuses_what_it_cannot_express() {
    let mut rng = StdRng::seed_from_u64(0xB1_9A7C_0008);
    let gemm = |input: usize, rng: &mut StdRng, shape: (usize, usize)| TwinOpSpec::Gemm {
        input,
        weight: uniform(rng, shape.0 * shape.1, 0.3),
        bias: uniform(rng, shape.0, 0.1),
        bias_err: Vec::new(),
        weight_rel_err: 0.0,
        shape,
    };
    // Transposed conv in the prefix: 1x4x4 -(convT k2 s2)-> 2x8x8.
    let transposed = TwinSpec {
        n_in: 16,
        ops: vec![
            TwinOpSpec::ConvTranspose {
                input: 0,
                weight: uniform(&mut rng, 2 * 4, 0.5),
                bias: uniform(&mut rng, 2, 0.1),
                bias_err: vec![0.0; 2],
                weight_rel_err: 0.0,
                kernel: (2, 1, 2, 2),
                stride: (2, 2),
                pads: (0, 0, 0, 0),
                ishape: (1, 4, 4),
                oshape: (2, 8, 8),
                out_pad: (0, 0),
            }, // t1
            TwinOpSpec::Relu { input: 1 }, // t2
            conv_op(
                2,
                uniform(&mut rng, 2 * 2 * 9, 0.4),
                uniform(&mut rng, 2, 0.1),
                (2, 2, 3, 3),
                (1, 1),
                1,
                (2, 8, 8),
                (2, 8, 8),
            ), // t3
            TwinOpSpec::Relu { input: 3 }, // t4
            TwinOpSpec::Flatten { input: 4 }, // t5
            gemm(5, &mut rng, (3, 128)),   // t6
            TwinOpSpec::Relu { input: 6 }, // t7
            gemm(7, &mut rng, (2, 3)),     // t8
        ],
    };
    let net = TwinNet::compile(&transposed).expect("compile");
    assert_eq!(
        PatchPlan::build(&net, 3, DEFAULT_RATIO).err(),
        Some("transposed")
    );
    assert_eq!(
        PatchPlan::build(&net, 2, DEFAULT_RATIO).err(),
        Some("not_relu")
    );
    assert_eq!(
        PatchPlan::build(&net, 6, DEFAULT_RATIO).err(),
        Some("not_trunk")
    );

    // Unequal anchor strides at one tensor: x feeds a (k2, s2, pad 2) conv and
    // a (k3, s1, pad 0) conv whose 6x6 outputs are added.
    let mismatch = TwinSpec {
        n_in: 64,
        ops: vec![
            conv_op(
                0,
                uniform(&mut rng, 2 * 4, 0.5),
                uniform(&mut rng, 2, 0.1),
                (2, 1, 2, 2),
                (2, 2),
                2,
                (1, 8, 8),
                (2, 6, 6),
            ), // t1
            conv_op(
                0,
                uniform(&mut rng, 2 * 9, 0.4),
                uniform(&mut rng, 2, 0.1),
                (2, 1, 3, 3),
                (1, 1),
                0,
                (1, 8, 8),
                (2, 6, 6),
            ), // t2
            TwinOpSpec::Add { lhs: 1, rhs: 2 }, // t3
            TwinOpSpec::Relu { input: 3 },      // t4
            TwinOpSpec::Flatten { input: 4 },   // t5
            gemm(5, &mut rng, (3, 72)),         // t6
            TwinOpSpec::Relu { input: 6 },      // t7
            gemm(7, &mut rng, (2, 3)),          // t8
        ],
    };
    let net = TwinNet::compile(&mismatch).expect("compile");
    assert_eq!(
        PatchPlan::build(&net, 3, DEFAULT_RATIO).err(),
        Some("stride_union")
    );

    // No gain: the chain's 4x4 seed tensors convert at once at rho = 0.05.
    let spec = chain_spec(&mut rng, 0.6);
    let net = TwinNet::compile(&spec).expect("compile");
    let (lo, hi) = (vec![-0.3; 128], vec![0.4; 128]);
    let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
    let relu1 = root.layers[1].op;
    assert_eq!(PatchPlan::build(&net, relu1, 0.05).err(), Some("no_gain"));

    let plan = PatchPlan::build(&net, relu1, DEFAULT_RATIO).expect("the chain plans");
    let n = root.layers[1].n;
    assert!(plan.prefix_bounds(&net, &root, &[], None, None).is_none());
    assert!(plan.prefix_bounds(&net, &root, &[n], None, None).is_none());
    let mut beta = DomainGates::default();
    beta.beta.insert(
        0,
        vec![BetaSplit {
            neuron: root.layers[0].unst.first().copied().unwrap_or(0),
            sign: 1,
            beta: 0.5,
        }],
    );
    assert!(
        plan.prefix_bounds(&net, &root, &[0], Some(&beta), None)
            .is_none(),
        "a shared-beta override breaks locality and must fall back"
    );
    let mut beta_pc = DomainGates::default();
    beta_pc.beta_pc.insert(
        0,
        vec![PcBetaCol {
            col: 0,
            terms: vec![BetaSplit {
                neuron: 0,
                sign: -1,
                beta: 0.25,
            }],
        }],
    );
    assert!(
        plan.prefix_bounds(&net, &root, &[0], Some(&beta_pc), None)
            .is_none(),
        "a per-column-beta override must fall back"
    );
    let infeasible = DomainGates {
        infeasible: true,
        ..DomainGates::default()
    };
    assert!(
        plan.prefix_bounds(&net, &root, &[0], Some(&infeasible), None)
            .is_none(),
        "an infeasible override must fall back"
    );

    // Envelope root lane: the patch publishes PerCol, never looser than the
    // dense envelope pair.
    let cols: Vec<usize> = (0..n).collect();
    let pair = plan
        .prefix_bounds(&net, &root, &cols, None, None)
        .expect("expressible")
        .expect("certified");
    let env = BackwardEngine::new(&net, &root)
        .with_err_lane(ErrLane::Envelope)
        .with_err_min_cols(1);
    let mut s = Array2::<f64>::zeros((n, n));
    for (c, &j) in cols.iter().enumerate() {
        s[[j, c]] = 1.0;
    }
    let seed = Seed { s, e: None };
    let el = env.concretize_lower(&env.run_prefix(&seed, relu1, LaneDir::Lower).expect("env"));
    let eu = env.concretize_upper(&env.run_prefix(&seed, relu1, LaneDir::Upper).expect("env"));
    for k in 0..n {
        assert!(
            pair.lb[k] >= el[k] && pair.ub[k] <= eu[k],
            "neuron {k}: the patch pair must never be looser than the envelope pair"
        );
    }
}

/// Every virtual conv reads only inside its natural input frame (no padding
/// taps: the planner's tripwire), keeps the full op's weights and fold
/// charges, and instantiates its envelopes at the FULL op's `k_bwd`.
#[test]
fn patch_virtual_convs_have_no_border_taps_and_dense_envelopes() {
    let mut virtuals = 0usize;
    for (name, spec, _) in fixtures() {
        let net = TwinNet::compile(&spec).expect("compile");
        for &relu_op in net.trunk_relus.iter().skip(1) {
            let Ok(plan) = PatchPlan::build(&net, relu_op, f64::INFINITY) else {
                continue;
            };
            for k in 0..relu_op {
                let Some(vop) = plan.virtual_conv(k) else {
                    continue;
                };
                let TwinOp::Conv(cv) = &net.ops[k] else {
                    panic!("{name}: op {k} carries a virtual conv but is not a conv");
                };
                assert!(
                    !vop.gather.contains(&usize::MAX),
                    "{name} op {k}: border tap"
                );
                assert_eq!(vop.k_bwd, cv.k_bwd, "{name} op {k}: envelope length");
                assert_eq!(vop.wt, cv.wt, "{name} op {k}: weights");
                assert_eq!(vop.bias, cv.bias, "{name} op {k}: bias");
                assert_eq!(vop.bias_err, cv.bias_err, "{name} op {k}: bias error");
                assert_eq!(
                    vop.weight_rel_err.to_bits(),
                    cv.weight_rel_err.to_bits(),
                    "{name} op {k}: fold charge"
                );
                assert_eq!(vop.kernel, cv.kernel);
                assert_eq!(vop.stride, cv.stride);
                virtuals += 1;
            }
        }
    }
    assert!(virtuals > 5, "only {virtuals} virtual convs checked");
}

/// THE COST MODEL IS THE KERNEL'S. For every all-patch plan, the planner's
/// per-column MACs equal the MACs the virtual kernels perform
/// (`sum |back_taps| * co * ci`), and never exceed the nominal dense count.
#[test]
fn patch_cost_model_matches_counted_macs() {
    let mut plans = 0usize;
    for (name, spec, _) in fixtures() {
        let net = TwinNet::compile(&spec).expect("compile");
        for &relu_op in net.trunk_relus.iter().skip(1) {
            let Ok(plan) = PatchPlan::build(&net, relu_op, f64::INFINITY) else {
                continue;
            };
            let mut counted = 0u64;
            for k in 0..relu_op {
                if let Some(vop) = plan.virtual_conv(k) {
                    let (co, ci, _, _) = vop.kernel;
                    let taps: usize = vop.back_taps.iter().map(Vec::len).sum();
                    counted += (taps * co * ci) as u64;
                }
            }
            assert_eq!(
                plan.geo.macs_col_patch, counted,
                "{name} relu op {relu_op}: the planner must price what the kernel does"
            );
            plans += 1;
        }
    }
    assert!(plans > 5, "only {plans} plans priced");
}

/// DARK BY DEFAULT: the default policy, a disabled phase with patch mode set,
/// and every rebake config build no patch state; patch mode's own knobs are
/// inert while it is disarmed (the phase is bit-identical under any ratio/mb).
#[test]
fn bi_patch_phase_is_dark_by_default() {
    let dark = BiState::new(
        BackwardIntermRuntimePolicy {
            enabled: true,
            ..BackwardIntermRuntimePolicy::default()
        },
        None,
    );
    assert!(!dark.patch_armed(), "#bi-patch must ship dark");
    assert_eq!(dark.patch_counts(), (0, 0));
    let disabled = BiState::new(
        BackwardIntermRuntimePolicy {
            enabled: false,
            patch: armed(0.5),
            ..BackwardIntermRuntimePolicy::default()
        },
        None,
    );
    assert!(
        !disabled.patch_armed(),
        "a disabled phase must never arm patch mode"
    );

    // The `#joint-refresh` suite's own fixture (`rebake.rs` joint_refresh_tests:
    // seed 0x7F_0001, scale 0.6), whose trunk layers carry unstable neurons.
    let spec = chain_spec(&mut StdRng::seed_from_u64(0x7F_0001), 0.6);
    let net = TwinNet::compile(&spec).expect("compile");
    let (lo, hi) = (vec![-0.4; 128], vec![0.35; 128]);
    let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
    let layers = working_layers(&root);
    assert!(!layers.is_empty(), "no working layer (VOID fixture)");
    for &li in &layers {
        let mut base = BiState::new(phase_policy(BiPatchRuntimePolicy::default(), 1), None);
        let mut knobs = BiState::new(
            phase_policy(
                BiPatchRuntimePolicy {
                    enabled: false,
                    ratio: 0.05,
                    mb: 1,
                },
                1,
            ),
            None,
        );
        let (bl, bu) = run_layer(&mut base, &net, &root, li);
        let (kl, ku) = run_layer(&mut knobs, &net, &root, li);
        assert_bits(&format!("layer={li} lower"), &bl, &kl);
        assert_bits(&format!("layer={li} upper"), &bu, &ku);
        assert_eq!(knobs.patch_counts(), (0, 0));
    }
}

/// FAIL-CLOSED, BIT-IDENTICAL: armed at rho = 0.05, every chain layer's plan
/// is refused (`no_gain`: the 4x4 seed tensors convert at once), counted once
/// per layer, and the armed phase publishes the DARK phase's boxes bit for bit
/// (same column order, same chunks, same dense passes).
#[test]
fn bi_patch_refused_plans_fall_back_bit_identically() {
    // The `#joint-refresh` suite's own fixture (see the dark-default test).
    let spec = chain_spec(&mut StdRng::seed_from_u64(0x7F_0001), 0.6);
    let net = TwinNet::compile(&spec).expect("compile");
    let (lo, hi) = (vec![-0.4; 128], vec![0.35; 128]);
    let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
    let layers = working_layers(&root);
    assert!(!layers.is_empty(), "no working layer (VOID fixture)");
    for &li in &layers {
        let mut dark = BiState::new(phase_policy(BiPatchRuntimePolicy::default(), 2), None);
        let mut refused = BiState::new(phase_policy(armed(0.05), 2), None);
        assert!(refused.patch_armed());
        let (dl, du) = run_layer(&mut dark, &net, &root, li);
        let (rl, ru) = run_layer(&mut refused, &net, &root, li);
        assert_bits(&format!("layer={li} lower"), &dl, &rl);
        assert_bits(&format!("layer={li} upper"), &du, &ru);
        assert_eq!(
            refused.patch_counts(),
            (0, 1),
            "layer={li}: the plan must be refused once, at the layer, and never retried"
        );
    }
}

/// ENGAGEMENT + CONTAINMENT. The armed phase (`tighten_layer`, one
/// `#interm-alpha` candidate per chunk) runs patch pairs on every working
/// layer of the chain, the residual tiny net and the resnet-shaped net (no
/// fallback), publishes shrink-only boxes that contain the EXACT
/// pre-activations and agree with the dark phase to rounding. `pairs=0` would
/// be VOID, and is asserted against.
#[test]
fn bi_patch_phase_enters_and_stays_contained() {
    let mut rng = StdRng::seed_from_u64(0xB1_9A7C_000B);
    let mut checks = 0usize;
    let mut worked = 0usize;
    for (name, spec, rad) in fixtures().into_iter().take(3) {
        let net = TwinNet::compile(&spec).expect("compile");
        let center: Vec<f64> = (0..spec.n_in)
            .map(|_| rng.random_range(-rad..rad))
            .collect();
        let (lo, hi) = boxed(&center, rad);
        let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
        let exact: Vec<Vec<Vec<BigRational>>> = box_points(&mut rng, &lo, &hi, 2)
            .iter()
            .map(|x| exact_tensors(&spec, x))
            .collect();
        let layers = working_layers(&root);
        worked += layers.len();
        for &li in &layers {
            let what = format!("{name} layer={li}");
            let mut dark = BiState::new(phase_policy(BiPatchRuntimePolicy::default(), 1), None);
            let mut st = BiState::new(phase_policy(armed(DEFAULT_RATIO), 1), None);
            assert!(
                st.patch_armed(),
                "{what}: the armed policy must reach the phase"
            );
            let (dl, du) = run_layer(&mut dark, &net, &root, li);
            let (l, u) = run_layer(&mut st, &net, &root, li);
            let (pairs, fallbacks) = st.patch_counts();
            assert!(
                pairs > 0,
                "{what}: the armed phase never ran a patch pair (VOID)"
            );
            assert_eq!(fallbacks, 0, "{what}: a plannable layer fell back");
            let (l0, u0) = (&root.layers[li].l, &root.layers[li].u);
            for j in 0..l.len() {
                assert!(
                    l[j] >= l0[j] && u[j] <= u0[j],
                    "{what}: the armed phase widened neuron {j}"
                );
                let tol = 1e-6 * (1.0 + dl[j].abs().max(du[j].abs()));
                assert!(
                    (l[j] - dl[j]).abs() <= tol && (u[j] - du[j]).abs() <= tol,
                    "{what} neuron {j}: armed [{}, {}] vs dark [{}, {}] beyond rounding",
                    l[j],
                    u[j],
                    dl[j],
                    du[j]
                );
            }
            let all: Vec<usize> = (0..l.len()).collect();
            let tensor = pre_tensor(&net, root.layers[li].op);
            assert_contains(&what, &l, &u, &all, &exact, tensor, &mut checks);
        }
    }
    assert!(worked > 0, "no fixture had a working layer (VOID)");
    assert!(checks > 1_000, "only {checks} exact checks ran");
}

/// `#joint-refresh` with patch mode: the refresh runs its pairs as patch
/// walks, tightens, stays shrink-only against the input root, contains the
/// EXACT pre-activations and agrees with the dense refresh to rounding.
#[test]
fn joint_refresh_patch_matches_dense_refresh() {
    // The `#joint-refresh` suite's own fixture (`rebake.rs` joint_refresh_tests:
    // seed 0x7F_0001, scale 0.6), where the refresh is pinned to tighten.
    let spec = chain_spec(&mut StdRng::seed_from_u64(0x7F_0001), 0.6);
    let mut rng = StdRng::seed_from_u64(0xB1_9A7C_000C);
    let net = TwinNet::compile(&spec).expect("compile");
    let (lo, hi) = (vec![-0.4; 128], vec![0.35; 128]);
    let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
    let exact: Vec<Vec<Vec<BigRational>>> = box_points(&mut rng, &lo, &hi, 3)
        .iter()
        .map(|x| exact_tensors(&spec, x))
        .collect();
    let (dense, ds) =
        refresh_root_intermediates(&net, &root, 4_096, 16, false, None, None).expect("dense");
    let (patched, ps) = refresh_root_intermediates_with_patch(
        &net,
        &root,
        4_096,
        16,
        false,
        Some(PatchCfg::new(DEFAULT_RATIO, 256)),
        None,
        None,
    )
    .expect("patched refresh");
    assert!(
        ps.patch_pairs > 0,
        "the refresh never ran a patch pair (VOID)"
    );
    assert_eq!(ps.patch_fallbacks, 0, "{ps:?}");
    assert!(
        ps.tightened > 0,
        "the patched refresh tightened nothing (VOID)"
    );
    assert_eq!(ds.patch_pairs, 0, "the dark refresh must run no patch pair");
    let mut checks = 0usize;
    for (li, ((lp, ld), l0)) in patched
        .layers
        .iter()
        .zip(&dense.layers)
        .zip(&root.layers)
        .enumerate()
    {
        let what = format!("joint refresh layer={li}");
        for j in 0..lp.n {
            assert!(
                lp.l[j] >= l0.l[j] && lp.u[j] <= l0.u[j],
                "{what}: the patched refresh widened neuron {j}"
            );
            let tol = 1e-6 * (1.0 + ld.l[j].abs().max(ld.u[j].abs()));
            assert!(
                (lp.l[j] - ld.l[j]).abs() <= tol && (lp.u[j] - ld.u[j]).abs() <= tol,
                "{what} neuron {j}: patched [{}, {}] vs dense [{}, {}] beyond rounding",
                lp.l[j],
                lp.u[j],
                ld.l[j],
                ld.u[j]
            );
        }
        let all: Vec<usize> = (0..lp.n).collect();
        let tensor = pre_tensor(&net, lp.op);
        assert_contains(&what, &lp.l, &lp.u, &all, &exact, tensor, &mut checks);
    }
    assert!(checks > 500, "only {checks} exact checks ran");
}

/// #root-box-source (ENGINE_PROGRAM M5b): the certified patch-backward ROOT box
/// source (`margin_row/root_box.rs`) is pinned on these fixtures and oracles.
mod root_box_tests;
