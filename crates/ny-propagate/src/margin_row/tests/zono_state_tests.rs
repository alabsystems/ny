// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `#frozen-dual-state` (V1 stage A) soundness tests — the Rust port of the P1
//! (cora FC) and P1x (cifar100 ResNet) replicas' containment checks and exact
//! rational spot checks, on the synthetic twin nets of the parent module
//! (conv stride/pad, residual adds, ConvTranspose, ChannelAffine, BatchNorm
//! fold errors, the Gemm head). The real-ONNX pin lives in ny-cli
//! (`frozen_dual_state_contains_onnx_samples`).
//!
//! Soundness tests first:
//!
//! * `frozen_dual_state_contains_exact_rational_samples` — box corners and
//!   interior points, the network evaluated in EXACT rationals, at the stored
//!   parameters AND at the extremes of the certified fold-error ball; every
//!   claim of the state (boxes, symbols in `[-1, 1]`, every row, the margin)
//!   checked exactly;
//! * `frozen_dual_state_contains_dd_samples` — the same claims on many more
//!   points through the public double-double checker, which is itself
//!   cross-checked against the exact evaluation;
//! * `frozen_dual_node_bounds_never_exceed_the_true_margin` and
//!   `frozen_dual_split_children_cover_parent` — for sampled points, EVERY
//!   node on the point's split path (at random non-negative multipliers,
//!   including hostile ones) certifies at most the point's margin;
//! * `frozen_dual_node_bound_exact_rational_spot_check` — random and
//!   search-pruned nodes re-evaluated in exact rationals: certified <= exact;
//! * `frozen_dual_never_prunes_without_certificate` — the mutation pin: with
//!   the gamma charge disabled the SAME evaluator certifies values above the
//!   exact node value, so the charge is load-bearing;
//! * `frozen_dual_state_root_not_tighter_than_certified_root` — the state root
//!   equals the lane's own direct (spec-direction) CROWN bound on the same
//!   boxes up to rounding, and never exceeds it beyond rounding;
//! * refusals, determinism, batch = single, the static order, dark default and
//!   the stage-A probe's verdict inertness.

use super::*;

use ndarray::Array2;
use num_rational::BigRational;
use num_traits::{Signed, ToPrimitive, Zero};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use crate::margin_row::bounds::head_gates_with;
use crate::margin_row::zono_state::order::lmin_for_test;
use crate::margin_row::zono_state::{
    check_point, dd_forward, split_scores, static_order, FrameInput, FrozenDualBuildStats,
    FrozenDualFrame, FrozenDualOptions, FrozenDualProblem, FrozenDualState, NodeScratch, ReluSite,
    Side, Subgradient,
};
use crate::zono_core::ZonoRefusal;

type Q = BigRational;

fn q(v: f64) -> Q {
    BigRational::from_float(v).expect("finite float")
}

fn qi(v: i64) -> Q {
    BigRational::from_integer(v.into())
}

/// One synthetic instance: spec, compiled net, certified root, root y-box.
struct Fx {
    name: String,
    spec: TwinSpec,
    net: TwinNet,
    root: RootGates,
    ybox: YBox,
    lo: Vec<f64>,
    hi: Vec<f64>,
    t: usize,
    adv: Vec<usize>,
}

/// Box `c_i +- half` with seeded centres in `[-0.2, 0.2]` (a non-symmetric box
/// exercises the centre terms); `degenerate` pins input 0 to a point, which
/// exercises the no-symbol branch.
fn fx(name: &str, spec: TwinSpec, half: f64, seed: u64, degenerate: bool) -> Fx {
    let net = TwinNet::compile(&spec).expect("compile");
    let mut rng = StdRng::seed_from_u64(seed ^ 0xF00D);
    let mut lo = Vec::with_capacity(spec.n_in);
    let mut hi = Vec::with_capacity(spec.n_in);
    for i in 0..spec.n_in {
        let c: f64 = rng.random_range(-0.2..0.2);
        if degenerate && i == 0 {
            lo.push(c);
            hi.push(c);
        } else {
            lo.push(c - half);
            hi.push(c + half);
        }
    }
    let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root gates");
    let (t, _, adv) = tiny_targets(&net, &lo, &hi);
    let eng = BackwardEngine::new(&net, &root);
    let (al, au) = eng.y_rows(None).expect("y rows");
    let ybox = YBox::from_rows(&eng, &al, &au);
    Fx {
        name: name.to_string(),
        spec,
        net,
        root,
        ybox,
        lo,
        hi,
        t,
        adv,
    }
}

fn fixtures() -> Vec<Fx> {
    let mut v = Vec::new();
    for seed in [3u64, 11] {
        let mut rng = StdRng::seed_from_u64(seed);
        v.push(fx(
            &format!("tiny{seed}"),
            tiny_spec(&mut rng, 0.4),
            0.4,
            seed,
            seed == 11,
        ));
    }
    let mut rng = StdRng::seed_from_u64(29);
    v.push(fx(
        "cgan_like",
        cgan_like_spec(&mut rng, 0.7),
        0.5,
        29,
        false,
    ));
    v.push(fx(
        "conv_bn",
        conv_bn_spec(3, 2, 2, 8, 5.0, 1e-13, 1e-6, 16, 6, 0xC0B1),
        1.0,
        7,
        false,
    ));
    v.push(fx(
        "residual",
        residual_add_spec(2, 2, 2, 8, 3, 0.25, 16, 6, 0x5E5),
        1.0,
        13,
        true,
    ));
    v
}

/// The first fixture alone (single-fixture tests need not build all five).
fn fixture0() -> Fx {
    let mut rng = StdRng::seed_from_u64(3);
    fx("tiny3", tiny_spec(&mut rng, 0.4), 0.4, 3, false)
}

fn frame_of(f: &Fx) -> FrozenDualFrame {
    FrozenDualFrame::from_root(&f.net, &f.root, &f.ybox, f.t).expect("frame")
}

fn state_of(f: &Fx, frame: &FrozenDualFrame, class: usize) -> FrozenDualState {
    FrozenDualState::build(&f.net, frame, class, &FrozenDualOptions::default()).expect("state")
}

/// A point of the declared box: every fourth sample is a corner.
fn sample(rng: &mut StdRng, lo: &[f64], hi: &[f64], k: usize) -> Vec<f64> {
    lo.iter()
        .zip(hi)
        .map(|(&l, &h)| {
            let u: f64 = rng.random_range(0.0..1.0);
            if k.is_multiple_of(4) {
                if u < 0.5 {
                    l
                } else {
                    h
                }
            } else {
                (l + (h - l) * u).clamp(l, h)
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Exact-rational network semantics (the spec, not the compiled tables)
// ---------------------------------------------------------------------------

/// A member of the certified fold-error ball: `Stored` is the stored network;
/// `Up` moves every folded parameter to the `+` edge; `Mixed` picks a seeded
/// sign per parameter.
#[derive(Clone, Copy, Debug)]
enum Pert {
    Stored,
    Up,
    Mixed(u64),
}

fn pert_sign(p: Pert, op: usize, idx: usize, which: u64) -> i64 {
    match p {
        Pert::Stored => 0,
        Pert::Up => 1,
        Pert::Mixed(seed) => {
            let mut h = seed ^ (op as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            h ^= (idx as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F) ^ which;
            h ^= h >> 29;
            h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            if (h >> 17) & 1 == 0 {
                1
            } else {
                -1
            }
        }
    }
}

/// `w (1 + s rel)` exactly.
fn qw(w: f64, rel: f64, s: i64) -> Q {
    let base = q(w);
    if s == 0 || rel == 0.0 {
        return base;
    }
    base.clone() + base * q(rel) * qi(s)
}

/// `b + s err` exactly.
fn qb(b: f64, err: f64, s: i64) -> Q {
    if s == 0 || err == 0.0 {
        return q(b);
    }
    q(b) + q(err) * qi(s)
}

fn exact_forward(spec: &TwinSpec, x: &[f64], p: Pert) -> Vec<Vec<Q>> {
    let mut t: Vec<Vec<Q>> = Vec::with_capacity(spec.ops.len() + 1);
    t.push(x.iter().map(|&v| q(v)).collect());
    for (k, op) in spec.ops.iter().enumerate() {
        let out: Vec<Q> = match op {
            TwinOpSpec::Conv {
                input,
                weight,
                bias,
                bias_err,
                weight_rel_err,
                kernel,
                stride,
                pads,
                ishape,
                oshape,
            } => {
                let src = &t[*input];
                let (co, ci, kh, kw) = *kernel;
                let (_, ih, iw) = *ishape;
                let (_, oh, ow) = *oshape;
                let (sh, sw) = *stride;
                let (pt, pl, _, _) = *pads;
                let mut out = Vec::with_capacity(co * oh * ow);
                for oc in 0..co {
                    let b = qb(bias[oc], bias_err[oc], pert_sign(p, k, oc, 1));
                    for oy in 0..oh {
                        for ox in 0..ow {
                            let mut acc = b.clone();
                            for c in 0..ci {
                                for ky in 0..kh {
                                    for kx in 0..kw {
                                        let iy = (oy * sh + ky) as isize - pt as isize;
                                        let ix = (ox * sw + kx) as isize - pl as isize;
                                        if iy < 0 || ix < 0 {
                                            continue;
                                        }
                                        let (iy, ix) = (iy as usize, ix as usize);
                                        if iy >= ih || ix >= iw {
                                            continue;
                                        }
                                        let wi = ((oc * ci + c) * kh + ky) * kw + kx;
                                        let w =
                                            qw(weight[wi], *weight_rel_err, pert_sign(p, k, wi, 2));
                                        acc += w * &src[c * ih * iw + iy * iw + ix];
                                    }
                                }
                            }
                            out.push(acc);
                        }
                    }
                }
                out
            }
            TwinOpSpec::ConvTranspose {
                input,
                weight,
                bias,
                bias_err,
                weight_rel_err,
                kernel,
                stride,
                pads,
                ishape,
                oshape,
                ..
            } => {
                let src = &t[*input];
                let (co, ci, kh, kw) = *kernel;
                let (_, ih, iw) = *ishape;
                let (_, oh, ow) = *oshape;
                let (sh, sw) = *stride;
                let (pt, pl, _, _) = *pads;
                let mut out: Vec<Q> = Vec::with_capacity(co * oh * ow);
                for oc in 0..co {
                    let b = qb(bias[oc], bias_err[oc], pert_sign(p, k, oc, 1));
                    for _ in 0..oh * ow {
                        out.push(b.clone());
                    }
                }
                for c in 0..ci {
                    for iy in 0..ih {
                        for ix in 0..iw {
                            let xv = &src[c * ih * iw + iy * iw + ix];
                            for oc in 0..co {
                                for ky in 0..kh {
                                    for kx in 0..kw {
                                        let ny = iy * sh + ky;
                                        let nx = ix * sw + kx;
                                        if ny < pt || nx < pl {
                                            continue;
                                        }
                                        let (oy, ox) = (ny - pt, nx - pl);
                                        if oy < oh && ox < ow {
                                            let wi = ((oc * ci + c) * kh + ky) * kw + kx;
                                            let w = qw(
                                                weight[wi],
                                                *weight_rel_err,
                                                pert_sign(p, k, wi, 2),
                                            );
                                            out[(oc * oh + oy) * ow + ox] += w * xv;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                out
            }
            TwinOpSpec::ChannelAffine {
                input,
                scale,
                shift,
                scale_rel_err,
                shift_err,
                shape,
            } => {
                let (c, h, w) = *shape;
                let hw = h * w;
                let src = &t[*input];
                (0..c * hw)
                    .map(|j| {
                        let ch = j / hw;
                        let s = qw(scale[ch], *scale_rel_err, pert_sign(p, k, ch, 3));
                        let b = qb(shift[ch], shift_err[ch], pert_sign(p, k, ch, 4));
                        s * &src[j] + b
                    })
                    .collect()
            }
            TwinOpSpec::Relu { input } => t[*input]
                .iter()
                .map(|v| {
                    if v.is_positive() {
                        v.clone()
                    } else {
                        Q::zero()
                    }
                })
                .collect(),
            TwinOpSpec::Add { lhs, rhs } => {
                t[*lhs].iter().zip(&t[*rhs]).map(|(a, b)| a + b).collect()
            }
            TwinOpSpec::Flatten { input } => t[*input].clone(),
            TwinOpSpec::Gemm {
                input,
                weight,
                bias,
                bias_err,
                weight_rel_err,
                shape,
            } => {
                let (no, ni) = *shape;
                let src = &t[*input];
                (0..no)
                    .map(|o| {
                        let be = bias_err.get(o).copied().unwrap_or(0.0);
                        let mut acc = qb(bias[o], be, pert_sign(p, k, o, 5));
                        for i in 0..ni {
                            let wi = o * ni + i;
                            if weight[wi] != 0.0 {
                                let w = qw(weight[wi], *weight_rel_err, pert_sign(p, k, wi, 6));
                                acc += w * &src[i];
                            }
                        }
                        acc
                    })
                    .collect()
            }
        };
        t.push(out);
    }
    t
}

/// Every claim of `state` at `x`, EXACTLY, for the network member `p`.
/// Returns the list of violated claims (empty = contained).
fn exact_claims(
    f: &Fx,
    frame: &FrozenDualFrame,
    st: &FrozenDualState,
    x: &[f64],
    p: Pert,
) -> Vec<String> {
    let vals = exact_forward(&f.spec, x, p);
    let mut bad = Vec::new();
    let one = qi(1);
    // Boxes (the frame's premise).
    for layer in &frame.layers {
        for (j, z) in vals[layer.input].iter().enumerate() {
            if *z < q(layer.l[j]) || *z > q(layer.u[j]) {
                bad.push(format!("box {:?} neuron {j}", layer.site));
            }
        }
    }
    // Symbols.
    let mut e: Vec<Q> = vec![Q::zero(); st.n_gens];
    for (gi, &dim) in frame.in_cols.iter().enumerate() {
        let v = (q(x[dim]) - q(frame.mid[dim])) / q(frame.rad[dim]);
        if v.abs() > one {
            bad.push(format!("input symbol {gi}"));
        }
        e[gi] = v;
    }
    for dim in 0..frame.n_in {
        if frame.rad[dim] == 0.0 && q(x[dim]) != q(frame.mid[dim]) {
            bad.push(format!("degenerate input {dim} off its centre"));
        }
    }
    let mut z_rows = Vec::with_capacity(st.num_unstable());
    for k in 0..st.num_unstable() {
        let layer = &frame.layers[st.layer[k]];
        let z = vals[layer.input][st.neuron[k]].clone();
        let relu = if z.is_positive() {
            z.clone()
        } else {
            Q::zero()
        };
        let mu = q(st.mu[k]);
        let ek = (relu - q(st.lam[k]) * &z - &mu) / &mu;
        if ek.abs() > one {
            bad.push(format!("relaxation symbol row {k}"));
        }
        e[st.enc[k]] = ek;
        z_rows.push(z);
    }
    // Rows.
    for (k, z) in z_rows.iter().enumerate() {
        let mut pred = q(st.c_in[k]);
        for (a, ev) in st.row(k).iter().zip(&e) {
            if *a != 0.0 {
                pred += q(*a) * ev;
            }
        }
        if (z - pred).abs() > q(st.r[k]) {
            bad.push(format!("row {k} (r = {:.3e})", st.r[k]));
        }
    }
    // Margin.
    let logits = &vals[f.spec.ops.len()];
    let m = &logits[st.t] - &logits[st.class];
    let mut pred = q(st.c_obj);
    for (a, ev) in st.d.iter().zip(&e) {
        if *a != 0.0 {
            pred += q(*a) * ev;
        }
    }
    if (m - pred).abs() > q(st.r_obj) {
        bad.push(format!("margin (r_obj = {:.3e})", st.r_obj));
    }
    bad
}

// ---------------------------------------------------------------------------
// Soundness: containment
// ---------------------------------------------------------------------------

#[test]
fn frozen_dual_state_contains_exact_rational_samples() {
    let mut checked = 0usize;
    let mut rows_seen = 0usize;
    let mut failures = Vec::new();
    for f in fixtures() {
        let frame = frame_of(&f);
        for &j in f.adv.iter().take(2) {
            let st = state_of(&f, &frame, j);
            rows_seen += st.num_unstable();
            let mut rng = StdRng::seed_from_u64(0xE8AC ^ j as u64);
            for s in 0..8 {
                let x = sample(&mut rng, &f.lo, &f.hi, s);
                for p in [Pert::Stored, Pert::Up, Pert::Mixed(s as u64 + 1)] {
                    let bad = exact_claims(&f, &frame, &st, &x, p);
                    checked += 1;
                    if !bad.is_empty() {
                        failures.push(format!("{} class {j} sample {s} {p:?}: {bad:?}", f.name));
                    }
                }
            }
        }
    }
    assert!(
        checked > 0 && rows_seen > 0,
        "no unstable row was checked: VOID"
    );
    assert!(
        failures.is_empty(),
        "FROZEN-DUAL CONTAINMENT FAILED (a false-unsat generator): {failures:#?}"
    );
}

#[test]
fn frozen_dual_state_contains_dd_samples() {
    let mut checked = 0usize;
    let mut worst = [f64::NEG_INFINITY; 4];
    let mut failures = Vec::new();
    for f in fixtures() {
        let frame = frame_of(&f);
        for &j in f.adv.iter().take(3) {
            let st = state_of(&f, &frame, j);
            let mut rng = StdRng::seed_from_u64(0xDD ^ (j as u64) << 8);
            for s in 0..300 {
                let x = sample(&mut rng, &f.lo, &f.hi, s);
                let pc = check_point(&f.net, &frame, &st, &x).expect("check");
                checked += 1;
                for (w, v) in worst.iter_mut().zip([
                    pc.box_excess,
                    pc.symbol_excess,
                    pc.row_excess,
                    pc.obj_excess,
                ]) {
                    *w = w.max(v);
                }
                if !pc.holds() {
                    failures.push(format!("{} class {j} sample {s}: {pc:?}", f.name));
                }
                // Cross-check the double-double checker against exact rationals
                // on a few points: the margins must agree to double-double.
                if s < 2 {
                    let vals = exact_forward(&f.spec, &x, Pert::Stored);
                    let logits = &vals[f.spec.ops.len()];
                    let m = &logits[st.t] - &logits[st.class];
                    let diff = (m - q(pc.margin)).abs().to_f64().unwrap_or(f64::INFINITY);
                    assert!(
                        diff <= 1e-15 * (1.0 + pc.margin.abs()),
                        "{}: double-double margin {} disagrees with exact by {diff:e}",
                        f.name,
                        pc.margin
                    );
                }
            }
        }
    }
    eprintln!(
        "[frozen-dual-dd] checked={checked} worst box={:.3e} symbol={:.3e} row={:.3e} obj={:.3e}",
        worst[0], worst[1], worst[2], worst[3]
    );
    assert!(checked > 0, "nothing checked: VOID");
    assert!(
        failures.is_empty(),
        "FROZEN-DUAL DD CONTAINMENT FAILED: {failures:#?}"
    );
}

// ---------------------------------------------------------------------------
// Soundness: node bounds
// ---------------------------------------------------------------------------

/// Random non-negative multipliers, including zeros, large values and a NaN
/// (which the evaluator must clamp to 0).
fn hostile_multipliers(rng: &mut StdRng, d: usize) -> (Vec<f64>, Vec<f64>) {
    let draw = |rng: &mut StdRng| -> f64 {
        match rng.random_range(0..6u32) {
            0 => 0.0,
            1 => rng.random_range(0.0..1.0),
            2 => rng.random_range(0.0..50.0),
            3 => -rng.random_range(0.0..5.0),
            4 => f64::NAN,
            _ => rng.random_range(0.0..0.01),
        }
    };
    let l0 = (0..d).map(|_| draw(rng)).collect();
    let l1 = (0..d).map(|_| draw(rng)).collect();
    (l0, l1)
}

/// Sides of a point along the problem's order (the child containing it);
/// `None` at a depth where `z` is (numerically) zero, so both children do.
fn point_sides(prob: &FrozenDualProblem, z: &[f64]) -> Vec<Option<Side>> {
    prob.order()
        .iter()
        .map(|&k| {
            let v = z[k];
            if v.abs() <= 1e-25 {
                None
            } else if v < 0.0 {
                Some(Side::Off)
            } else {
                Some(Side::On)
            }
        })
        .collect()
}

#[test]
fn frozen_dual_node_bounds_never_exceed_the_true_margin() {
    let mut nodes = 0usize;
    let mut failures = Vec::new();
    for f in fixtures() {
        let frame = frame_of(&f);
        for &j in f.adv.iter().take(2) {
            let st = state_of(&f, &frame, j);
            let order = static_order(&st);
            let prob = FrozenDualProblem::new(&st, &order).expect("problem");
            let mut scratch = NodeScratch::default();
            let mut rng = StdRng::seed_from_u64(0xB0 ^ j as u64);
            for s in 0..200 {
                let x = sample(&mut rng, &f.lo, &f.hi, s);
                let pc = check_point(&f.net, &frame, &st, &x).expect("check");
                let sides = point_sides(&prob, &pc.z);
                let max_d = prob.max_depth().min(14);
                for depth in 0..=max_d {
                    // A zero-crossing point lies in BOTH children: test Off.
                    let path: Vec<Side> = sides[..depth]
                        .iter()
                        .map(|s| s.unwrap_or(Side::Off))
                        .collect();
                    let (l0, l1) = hostile_multipliers(&mut rng, depth);
                    let nb = prob.eval_node(&path, &l0, &l1, &mut scratch, None);
                    nodes += 1;
                    if nb.certified > pc.margin + 1e-12 * (1.0 + pc.margin.abs()) {
                        failures.push(format!(
                            "{} class {j} sample {s} depth {depth}: certified {:.6e} > margin {:.6e}",
                            f.name, nb.certified, pc.margin
                        ));
                    }
                }
            }
        }
    }
    assert!(nodes > 1000, "only {nodes} nodes evaluated: VOID");
    assert!(
        failures.is_empty(),
        "FROZEN-DUAL NODE BOUND EXCEEDS A TRUE MARGIN (false unsat): {failures:#?}"
    );
}

#[test]
fn frozen_dual_split_children_cover_parent() {
    // Every point of a parent node lies in a child (its z sign picks it, both
    // at z = 0) whose certified bound is at most the point's margin: pruning
    // both children can never drop a point of the parent.
    let mut covered = 0usize;
    for f in fixtures() {
        let frame = frame_of(&f);
        let Some(&j) = f.adv.first() else { continue };
        let st = state_of(&f, &frame, j);
        let order = static_order(&st);
        let prob = FrozenDualProblem::new(&st, &order).expect("problem");
        let mut scratch = NodeScratch::default();
        let mut rng = StdRng::seed_from_u64(0xC0 ^ j as u64);
        for s in 0..150 {
            let x = sample(&mut rng, &f.lo, &f.hi, s);
            let pc = check_point(&f.net, &frame, &st, &x).expect("check");
            let sides = point_sides(&prob, &pc.z);
            for depth in 0..prob.max_depth().min(10) {
                let parent: Vec<Side> = sides[..depth]
                    .iter()
                    .map(|s| s.unwrap_or(Side::On))
                    .collect();
                let children: Vec<Side> = match sides[depth] {
                    Some(side) => vec![side],
                    None => vec![Side::Off, Side::On],
                };
                for c in children {
                    let mut path = parent.clone();
                    path.push(c);
                    let zeros = vec![0.0; path.len()];
                    let nb = prob.eval_node(&path, &zeros, &zeros, &mut scratch, None);
                    assert!(
                        nb.certified <= pc.margin + 1e-12 * (1.0 + pc.margin.abs()),
                        "{}: child {path:?} certifies {:.6e} above a covered point's margin {:.6e}",
                        f.name,
                        nb.certified,
                        pc.margin
                    );
                    covered += 1;
                }
            }
        }
    }
    assert!(covered > 100, "only {covered} child checks: VOID");
}

/// The exact value of the Lagrangian bound at the evaluator's CLAMPED
/// multipliers, from the problem's stored floats.
fn exact_node_value(prob: &FrozenDualProblem, sides: &[Side], l0: &[f64], l1: &[f64]) -> Q {
    let mut rc: Vec<Q> = prob.d().iter().map(|&v| q(v)).collect();
    for i in 0..sides.len() {
        rc[prob.split_row(i).enc] = Q::zero();
    }
    let mut c = q(prob.c0());
    let mut lag = Q::zero();
    for (i, side) in sides.iter().enumerate() {
        let row = prob.split_row(i);
        let si = usize::from(*side == Side::On);
        let m0 = q(l0[i].max(0.0));
        let m1 = q(l1[i].max(0.0));
        let sign = if *side == Side::Off { qi(1) } else { qi(-1) };
        let mult = q(row.rho[si]) + sign * (&m0 - &m1);
        if !mult.is_zero() {
            for (acc, &a) in rc.iter_mut().zip(row.a) {
                if a != 0.0 {
                    *acc += &mult * q(a);
                }
            }
        }
        c += q(row.c0s[si]);
        lag += m0 * q(row.b0[si]) + m1 * q(row.b1[si]);
    }
    let sumabs = rc.iter().fold(Q::zero(), |acc, v| acc + v.abs());
    c - sumabs - lag
}

/// A tiny certified BFS (static order, zero multipliers plus one projected
/// supergradient step): the nodes it PRUNES, P1's `exact_spot` population.
fn pruned_nodes(prob: &FrozenDualProblem, cap: usize) -> Vec<(Vec<Side>, Vec<f64>, Vec<f64>)> {
    let mut out = Vec::new();
    let mut frontier: Vec<Vec<Side>> = vec![Vec::new()];
    let mut scratch = NodeScratch::default();
    for depth in 1..=prob.max_depth().min(8) {
        let mut next = Vec::new();
        for parent in &frontier {
            for side in [Side::Off, Side::On] {
                let mut path = parent.clone();
                path.push(side);
                let zeros = vec![0.0; depth];
                let mut s0 = vec![0.0; depth];
                let mut s1 = vec![0.0; depth];
                let nb0 = prob.eval_node(
                    &path,
                    &zeros,
                    &zeros,
                    &mut scratch,
                    Some(Subgradient {
                        s0: &mut s0,
                        s1: &mut s1,
                    }),
                );
                let l0: Vec<f64> = s0.iter().map(|v| (0.05 * v).max(0.0)).collect();
                let l1: Vec<f64> = s1.iter().map(|v| (0.05 * v).max(0.0)).collect();
                let nb1 = prob.eval_node(&path, &l0, &l1, &mut scratch, None);
                if nb1.prunes() {
                    out.push((path, l0, l1));
                } else if nb0.prunes() {
                    out.push((path, zeros.clone(), zeros));
                } else if next.len() < cap {
                    next.push(path);
                }
            }
        }
        frontier = next;
        if frontier.is_empty() {
            break;
        }
    }
    out
}

#[test]
fn frozen_dual_node_bound_exact_rational_spot_check() {
    let mut checked = 0usize;
    let mut pruned_checked = 0usize;
    let mut raw_over = 0usize;
    let mut min_margin = f64::INFINITY;
    for f in fixtures() {
        let frame = frame_of(&f);
        for &j in f.adv.iter().take(2) {
            let st = state_of(&f, &frame, j);
            let order = static_order(&st);
            let prob = FrozenDualProblem::new(&st, &order).expect("problem");
            let mut scratch = NodeScratch::default();
            let mut rng = StdRng::seed_from_u64(0x5A ^ j as u64);
            let mut nodes: Vec<(Vec<Side>, Vec<f64>, Vec<f64>)> = pruned_nodes(&prob, 64);
            pruned_checked += nodes.len();
            for _ in 0..40 {
                let d = rng.random_range(0..=prob.max_depth().min(16));
                let sides: Vec<Side> = (0..d)
                    .map(|_| {
                        if rng.random_range(0..2u32) == 0 {
                            Side::Off
                        } else {
                            Side::On
                        }
                    })
                    .collect();
                let (l0, l1) = hostile_multipliers(&mut rng, d);
                nodes.push((sides, l0, l1));
            }
            for (sides, l0, l1) in &nodes {
                let nb = prob.eval_node(sides, l0, l1, &mut scratch, None);
                if !nb.certified.is_finite() {
                    continue;
                }
                let exact = exact_node_value(&prob, sides, l0, l1);
                let margin = (&exact - q(nb.certified)).to_f64().unwrap_or(f64::NAN);
                min_margin = min_margin.min(margin);
                assert!(
                    q(nb.certified) <= exact,
                    "{} class {j}: certified {:e} exceeds the EXACT node value by {:e}",
                    f.name,
                    nb.certified,
                    -margin
                );
                if q(nb.value) > exact {
                    raw_over += 1;
                }
                checked += 1;
            }
        }
    }
    eprintln!(
        "[frozen-dual-exact] checked={checked} pruned={pruned_checked} \
         min(exact - certified)={min_margin:.3e} raw_f64_over_exact={raw_over}"
    );
    assert!(checked > 100, "only {checked} nodes checked: VOID");
}

/// A synthetic cancellation-heavy state: large coefficients whose node values
/// nearly cancel, so the binary64 evaluation error is many ulps of the result.
fn cancellation_state(seed: u64) -> FrozenDualState {
    let mut rng = StdRng::seed_from_u64(seed);
    let s_rows = 12usize;
    let n_in = 40usize;
    let n = n_in + s_rows;
    let mut a = vec![0.0f64; s_rows * n];
    for k in 0..s_rows {
        for c in 0..(n_in + k) {
            a[k * n + c] = rng.random_range(-1.0e6..1.0e6);
        }
    }
    let d: Vec<f64> = (0..n).map(|_| rng.random_range(-1.0e7..1.0e7)).collect();
    let abs_d: f64 = d.iter().map(|v| v.abs()).sum();
    FrozenDualState {
        t: 0,
        class: 1,
        n_gens: n,
        n_in_gens: n_in,
        c_obj: abs_d * (1.0 + 1e-9),
        d,
        r_obj: 1e-9,
        a,
        c_in: (0..s_rows)
            .map(|_| rng.random_range(-1.0e5..1.0e5))
            .collect(),
        r: vec![1e-9; s_rows],
        lam: (0..s_rows).map(|_| rng.random_range(0.1..0.9)).collect(),
        mu: (0..s_rows).map(|_| rng.random_range(0.5..2.0)).collect(),
        lo: vec![-1.0e6; s_rows],
        hi: vec![1.0e6; s_rows],
        enc: (0..s_rows).map(|k| n_in + k).collect(),
        layer: vec![0; s_rows],
        site: vec![ReluSite::Trunk(0); s_rows],
        neuron: (0..s_rows).collect(),
        stats: FrozenDualBuildStats::default(),
    }
}

#[test]
fn frozen_dual_never_prunes_without_certificate() {
    // MUTATION PIN: the same evaluator with the gamma charge disabled must
    // certify values ABOVE the exact node value on this cancellation-heavy
    // state (otherwise the pin proves nothing), while the charged evaluator
    // never does.
    let mut uncharged_over = 0usize;
    let mut checked = 0usize;
    for seed in 0..4u64 {
        let st = cancellation_state(0xCA0 + seed);
        let order: Vec<usize> = (0..st.num_unstable()).collect();
        let prob = FrozenDualProblem::new(&st, &order).expect("problem");
        let mut scratch = NodeScratch::default();
        let mut rng = StdRng::seed_from_u64(seed);
        for _ in 0..200 {
            let d = rng.random_range(0..=prob.max_depth());
            let sides: Vec<Side> = (0..d)
                .map(|_| {
                    if rng.random_range(0..2u32) == 0 {
                        Side::Off
                    } else {
                        Side::On
                    }
                })
                .collect();
            let l0: Vec<f64> = (0..d).map(|_| rng.random_range(0.0..3.0)).collect();
            let l1: Vec<f64> = (0..d).map(|_| rng.random_range(0.0..3.0)).collect();
            let exact = exact_node_value(&prob, &sides, &l0, &l1);
            let charged = prob.eval_node(&sides, &l0, &l1, &mut scratch, None);
            let bare = prob.eval_node_inner(&sides, &l0, &l1, &mut scratch, None, false);
            assert!(
                q(charged.certified) <= exact,
                "charged certificate exceeds the exact node value"
            );
            if bare.certified.is_finite() && q(bare.certified) > exact {
                uncharged_over += 1;
            }
            checked += 1;
        }
    }
    assert!(checked > 0, "VOID");
    assert!(
        uncharged_over > 0,
        "with the gamma charge disabled no node exceeded its exact value: the mutation pin \
         is VOID (the cancellation fixture no longer stresses the rounding)"
    );
}

// ---------------------------------------------------------------------------
// Root parity against the lane's own direct CROWN bound
// ---------------------------------------------------------------------------

#[test]
fn frozen_dual_state_root_not_tighter_than_certified_root() {
    let mut compared = 0usize;
    for f in fixtures() {
        let frame = frame_of(&f);
        let eng = BackwardEngine::new(&f.net, &f.root);
        let gates = head_gates_with(&f.ybox, f.root.mode, f.root.head_alpha.as_deref());
        for &j in f.adv.iter().take(3) {
            let st = state_of(&f, &frame, j);
            let mb = MarginBatch::new(&f.net, f.t, &[j]).expect("margin batch");
            let ms = margin_seed(&mb, &gates, &f.ybox, f.root.mode);
            let pass = eng
                .run(&ms.seed, None, LaneDir::Lower, None, false)
                .expect("direct pass");
            let direct = per_class_direct(&eng, &pass, &ms, 0..1)[0];
            let root = st.root_bound();
            let tol = 1e-9 * (1.0 + direct.abs());
            assert!(
                root <= direct + tol,
                "{} class {j}: the state root {root:.12e} is TIGHTER than the lane's direct \
                 CROWN bound {direct:.12e} on the same boxes — the relaxation is not the one \
                 the boxes certify",
                f.name
            );
            assert!(
                root >= direct - 1e-6 * (1.0 + direct.abs()),
                "{} class {j}: the state root {root:.12e} lost more than rounding against the \
                 direct CROWN bound {direct:.12e} (direction-adaptive lambda not applied?)",
                f.name
            );
            let prob_root = FrozenDualProblem::new(&st, &[])
                .expect("problem")
                .root_bound();
            assert!(prob_root <= root + tol, "problem root above the state root");
            compared += 1;
        }
    }
    assert!(compared > 0, "VOID");
}

// ---------------------------------------------------------------------------
// Structure, refusals, determinism
// ---------------------------------------------------------------------------

#[test]
fn frozen_dual_state_structure_is_consistent() {
    let mut rows = 0usize;
    for f in fixtures() {
        let frame = frame_of(&f);
        assert_eq!(frame.n_gens, frame.n_in_gens + frame.n_unstable);
        let st = state_of(&f, &frame, f.adv[0]);
        assert_eq!(st.n_gens, frame.n_gens);
        assert_eq!(st.num_unstable(), frame.n_unstable);
        assert_eq!(st.d.len(), st.n_gens);
        for k in 0..st.num_unstable() {
            assert_eq!(st.enc[k], st.n_in_gens + k, "{}: row/symbol layout", f.name);
            let row = st.row(k);
            // A row depends only on the input and UPSTREAM symbols.
            let layer = &frame.layers[st.layer[k]];
            for (c, &v) in row.iter().enumerate().skip(layer.col0) {
                assert_eq!(
                    v, 0.0,
                    "{}: row {k} reads symbol {c} at/after its layer",
                    f.name
                );
            }
            assert!((0.0..=1.0).contains(&st.lam[k]));
            assert!(st.mu[k] > 0.0 && st.r[k] >= 0.0);
            assert!(st.lo[k] < 0.0 && st.hi[k] > 0.0);
        }
        assert!(st.r_obj >= 0.0);
        rows += st.num_unstable();
    }
    assert!(rows > 0, "no fixture has an unstable neuron: VOID");
}

#[test]
fn frozen_dual_state_is_deterministic() {
    let f = &fixture0();
    let frame = frame_of(f);
    let a = state_of(f, &frame, f.adv[0]);
    let b = state_of(f, &frame, f.adv[0]);
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&a.a), bits(&b.a));
    assert_eq!(bits(&a.d), bits(&b.d));
    assert_eq!(bits(&a.r), bits(&b.r));
    assert_eq!(bits(&a.c_in), bits(&b.c_in));
    assert_eq!(a.c_obj.to_bits(), b.c_obj.to_bits());
    assert_eq!(a.r_obj.to_bits(), b.r_obj.to_bits());
}

#[test]
fn frozen_dual_refuses_nonfinite_and_budget() {
    let f = &fixture0();
    let base = FrameInput::from_root(&f.root, &f.ybox);
    // NaN box.
    let mut input = base.clone();
    input.trunk[0].l[0] = f64::NAN;
    assert!(matches!(
        FrozenDualFrame::new(&f.net, input, f.t),
        Err(ZonoRefusal::NonFinite(_))
    ));
    // Crossed box.
    let mut input = base.clone();
    input.trunk[0].l[0] = input.trunk[0].u[0] + 1.0;
    assert!(FrozenDualFrame::new(&f.net, input, f.t).is_err());
    // A missing trunk layer.
    let mut input = base.clone();
    input.trunk.pop();
    assert!(matches!(
        FrozenDualFrame::new(&f.net, input, f.t),
        Err(ZonoRefusal::Shape(_))
    ));
    // A slope outside [0, 1] on an unstable neuron.
    let unstable = base.trunk.iter().enumerate().find_map(|(m, li)| {
        (0..li.l.len())
            .find(|&j| li.l[j] < 0.0 && li.u[j] > 0.0)
            .map(|j| (m, j))
    });
    if let Some((m, j)) = unstable {
        let mut input = base.clone();
        input.trunk[m].alpha[j] = 1.5;
        assert!(FrozenDualFrame::new(&f.net, input, f.t).is_err());
    }
    // A non-finite input box.
    let mut input = base;
    input.hi[1] = f64::INFINITY;
    assert!(FrozenDualFrame::new(&f.net, input, f.t).is_err());

    let frame = frame_of(f);
    // Memory admission.
    let tiny = FrozenDualOptions {
        max_state_bytes: 1,
        ..FrozenDualOptions::default()
    };
    assert!(matches!(
        FrozenDualState::build(&f.net, &frame, f.adv[0], &tiny),
        Err(ZonoRefusal::Memory { .. })
    ));
    // Deadline.
    let late = FrozenDualOptions {
        deadline: Some(std::time::Instant::now()),
        ..FrozenDualOptions::default()
    };
    assert!(matches!(
        FrozenDualState::build(&f.net, &frame, f.adv[0], &late),
        Err(ZonoRefusal::Deadline)
    ));
    // The true class is not a margin.
    assert!(FrozenDualState::build(&f.net, &frame, f.t, &FrozenDualOptions::default()).is_err());

    // Problem: repeated or out-of-range rows.
    let st = state_of(f, &frame, f.adv[0]);
    if st.num_unstable() > 0 {
        assert!(FrozenDualProblem::new(&st, &[0, 0]).is_err());
    }
    assert!(FrozenDualProblem::new(&st, &[st.num_unstable()]).is_err());

    // Node evaluation fails closed.
    let order = static_order(&st);
    let prob = FrozenDualProblem::new(&st, &order).expect("problem");
    let mut scratch = NodeScratch::default();
    if prob.max_depth() > 0 {
        let inf = prob.eval_node(&[Side::Off], &[f64::INFINITY], &[0.0], &mut scratch, None);
        assert_eq!(inf.certified, f64::NEG_INFINITY);
        assert!(!inf.prunes());
        let mismatch = prob.eval_node(&[Side::Off], &[], &[], &mut scratch, None);
        assert_eq!(mismatch.certified, f64::NEG_INFINITY);
        // NaN and negative multipliers clamp to 0 (a valid dual point).
        let nan = prob.eval_node(&[Side::On], &[f64::NAN], &[-3.0], &mut scratch, None);
        let zero = prob.eval_node(&[Side::On], &[0.0], &[0.0], &mut scratch, None);
        assert_eq!(nan.certified.to_bits(), zero.certified.to_bits());
    }
    let too_deep = vec![Side::Off; prob.max_depth() + 1];
    let zeros = vec![0.0; too_deep.len()];
    let deep = prob.eval_node(&too_deep, &zeros, &zeros, &mut scratch, None);
    assert_eq!(deep.certified, f64::NEG_INFINITY);
}

#[test]
fn frozen_dual_frame_refuses_a_root_that_is_not_a_whole_box_enclosure() {
    // The frame's claim is about the WHOLE declared box, so `from_root` must
    // refuse every root whose boxes are not whole-box certified enclosures.
    let f = &fixture0();
    assert!(
        FrozenDualFrame::from_root(&f.net, &f.root, &f.ybox, f.t).is_ok(),
        "the fixture's own outward, unbaked root must build a frame (else every refusal \
         below is VOID)"
    );
    // A Parity root: round-to-nearest boxes, not enclosures.
    let parity =
        RootGates::build(&f.net, &f.lo, &f.hi, RoundMode::Parity, None).expect("parity root");
    assert!(
        matches!(
            FrozenDualFrame::from_root(&f.net, &parity, &f.ybox, f.t),
            Err(ZonoRefusal::Internal(ref why)) if why.contains("certified-outward")
        ),
        "a Parity root must be refused"
    );
    // A gate layer paired with the wrong relu op.
    let mut shifted =
        RootGates::build(&f.net, &f.lo, &f.hi, RoundMode::Outward, None).expect("root");
    shifted.layers[0].op += 1;
    assert!(
        matches!(
            FrozenDualFrame::from_root(&f.net, &shifted, &f.ybox, f.t),
            Err(ZonoRefusal::Internal(ref why)) if why.contains("is relu op")
        ),
        "a root whose gate layers do not line up with the trunk relus must be refused"
    );
    // A split-baked (epoch child) root: the baked neuron keeps its straddling
    // box but leaves `unst`, and every box downstream of it is subdomain-only.
    let (li, idx) = f
        .root
        .layers
        .iter()
        .enumerate()
        .find_map(|(li, g)| g.unst.first().map(|&j| (li, j)))
        .expect("VOID: the fixture has no unstable trunk neuron to bake");
    for dir in [1i8, -1] {
        let (baked, _) = RootGates::build_retaining(
            &f.net,
            &f.lo,
            &f.hi,
            RoundMode::Outward,
            None,
            None,
            &[(li, idx, dir)],
        )
        .expect("baked root");
        let g = &baked.layers[li];
        assert!(
            g.l[idx] < 0.0 && g.u[idx] > 0.0 && !g.unst.contains(&idx),
            "VOID: the bake signature this test pins is absent (dir {dir})"
        );
        assert!(
            matches!(
                FrozenDualFrame::from_root(&f.net, &baked, &f.ybox, f.t),
                Err(ZonoRefusal::Internal(ref why)) if why.contains("split-baked")
            ),
            "a split-baked root (dir {dir}) must be refused: its downstream boxes hold only on \
             the split subdomain"
        );
    }
}

#[test]
fn frozen_dual_eval_batch_matches_single() {
    let f = &fixture0();
    let frame = frame_of(f);
    let st = state_of(f, &frame, f.adv[0]);
    let order = static_order(&st);
    let prob = FrozenDualProblem::new(&st, &order).expect("problem");
    let depth = prob.max_depth().min(6);
    assert!(depth > 0, "the fixture has no unstable neuron: VOID");
    let count = 17usize;
    let mut rng = StdRng::seed_from_u64(77);
    let mut sides = Vec::new();
    let mut l0 = Vec::new();
    let mut l1 = Vec::new();
    for _ in 0..count {
        for _ in 0..depth {
            sides.push(if rng.random_range(0..2u32) == 0 {
                Side::Off
            } else {
                Side::On
            });
            l0.push(rng.random_range(0.0..2.0));
            l1.push(rng.random_range(0.0..2.0));
        }
    }
    let batch = prob.eval_batch(count, depth, &sides, &l0, &l1);
    assert_eq!(batch.len(), count);
    let mut scratch = NodeScratch::default();
    for b in 0..count {
        let r = b * depth..(b + 1) * depth;
        let one = prob.eval_node(
            &sides[r.clone()],
            &l0[r.clone()],
            &l1[r],
            &mut scratch,
            None,
        );
        assert_eq!(one.certified.to_bits(), batch[b].certified.to_bits());
        assert_eq!(one.value.to_bits(), batch[b].value.to_bits());
    }
    // A malformed batch refuses every node.
    let bad = prob.eval_batch(count, depth, &sides[1..], &l0, &l1);
    assert!(bad.iter().all(|nb| nb.certified == f64::NEG_INFINITY));
}

#[test]
fn frozen_dual_static_order_is_a_permutation_and_lmin_is_the_exact_dual() {
    for f in fixtures() {
        let frame = frame_of(&f);
        let st = state_of(&f, &frame, f.adv[0]);
        let order = static_order(&st);
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..st.num_unstable()).collect::<Vec<_>>());
        let scores = split_scores(&st);
        for w in order.windows(2) {
            assert!(scores[w[0]] >= scores[w[1]] || scores[w[0]].is_nan());
        }
    }
    // lmin against a brute-force maximum over its breakpoints (the dual is
    // concave piecewise linear, so its max over lambda >= 0 sits at 0 or a
    // breakpoint, or is unbounded).
    let mut rng = StdRng::seed_from_u64(0x1A);
    for _ in 0..200 {
        let n = rng.random_range(1..8usize);
        let a: Vec<f64> = (0..n).map(|_| rng.random_range(-2.0..2.0)).collect();
        let d: Vec<f64> = (0..n).map(|_| rng.random_range(-2.0..2.0)).collect();
        let c0 = rng.random_range(-3.0..3.0);
        let beta = rng.random_range(-3.0..3.0);
        let g = |lam: f64| {
            c0 - d
                .iter()
                .zip(&a)
                .map(|(dv, av)| (dv + lam * av).abs())
                .sum::<f64>()
                - lam * beta
        };
        let mut cands = vec![0.0f64];
        for (dv, av) in d.iter().zip(&a) {
            if *av != 0.0 && -dv / av > 0.0 {
                cands.push(-dv / av);
            }
        }
        let best = cands
            .iter()
            .map(|&l| g(l))
            .fold(f64::NEG_INFINITY, f64::max);
        let far = cands.iter().copied().fold(0.0, f64::max) + 10.0;
        let unbounded = g(far + 1.0) > g(far) + 1e-12;
        let got = lmin_for_test(&a, &d, c0, beta);
        if unbounded {
            assert!(got.is_infinite(), "lmin {got} but the dual is unbounded");
        } else {
            assert!(
                (got - best).abs() <= 1e-9 * (1.0 + best.abs()),
                "lmin {got} vs {best}"
            );
        }
    }
}

#[test]
fn frozen_dual_bab_is_dark_by_default() {
    use ny_levers::decls::margin_row::MARGIN_ROW_FROZEN_DUAL_BAB as DECL;
    assert_eq!(DECL.default, ny_levers::DefaultSpec::Bool(false));
    assert_eq!(DECL.moat, ny_levers::MoatRisk::High);
    // A governed read (not a raw environment access, which the source
    // frontier would count as new reader debt).
    if ny_levers::read(&DECL).source == ny_levers::Source::Default {
        assert!(
            !crate::preset_gates::margin_row_frozen_dual_bab_enabled(),
            "with no environment and no preset cell the gate must resolve dark"
        );
    }
}

#[test]
fn frozen_dual_stage_a_probe_is_verdict_inert() {
    // The probe only reads the frozen root and prints: the tree's result on the
    // same root is bit-identical before and after it runs.
    let f = &fixture0();
    let run = || {
        let cfg = BabConfig {
            max_expansions: 40,
            frontier: 1,
            frontier_contextual: 1,
            ..BabConfig::default()
        };
        MarginRowBab::run(&f.net, &f.root, f.t, &f.adv, cfg)
    };
    let summary = |o: &MarginRowOutcome| match o {
        MarginRowOutcome::Unsat(s, _) => (true, s.root_bound.to_bits(), s.expansions, s.closed),
        MarginRowOutcome::Unknown { stats, .. } => stats.as_ref().map_or((false, 0, 0, 0), |s| {
            (false, s.root_bound.to_bits(), s.expansions, s.closed)
        }),
    };
    let before = summary(&run());
    let report = crate::margin_row::zono_state::stage_a_probe(&f.net, &f.root, f.t, &f.adv, None);
    let after = summary(&run());
    assert_eq!(before, after, "the stage-A probe changed the tree's result");
    assert_eq!(report.built + report.refused, report.classes);
}

#[test]
fn frozen_dual_frame_input_matches_the_lane_root() {
    // The frame's head relaxation is exactly the lane's head gates on the
    // root y-box, and its trunk relaxation the committed root gates.
    let f = &fixture0();
    let input = FrameInput::from_root(&f.root, &f.ybox);
    let gates = head_gates_with(&f.ybox, RoundMode::Outward, f.root.head_alpha.as_deref());
    assert_eq!(input.head.alpha, gates.alpha);
    assert_eq!(input.head.s, gates.s);
    assert_eq!(input.head.l, f.ybox.ly);
    assert_eq!(input.head.u, f.ybox.uy);
    for (li, g) in input.trunk.iter().zip(&f.root.layers) {
        assert_eq!(li.l, g.l);
        assert_eq!(li.u, g.u);
        assert_eq!(li.alpha, g.alpha);
        assert_eq!(li.s, g.s);
    }
    // dd_forward agrees with the lane's own forward on the head pre-activation.
    let x: Vec<f64> =
        f.lo.iter()
            .zip(&f.hi)
            .map(|(l, h)| f64::midpoint(*l, *h))
            .collect();
    let vals = dd_forward(&f.net, &x).expect("dd forward");
    let xs = Array2::from_shape_vec((x.len(), 1), x).expect("column");
    let (y, _) = f
        .net
        .forward_points(&xs, &std::collections::BTreeMap::new())
        .expect("forward");
    let yd = &vals[f.net.i_gemm1 + 1];
    for (k, v) in yd.iter().enumerate() {
        assert!((v.to_f64() - y[[k, 0]]).abs() <= 1e-12 * (1.0 + y[[k, 0]].abs()));
    }
}

/// #frozen-dual-bab (V1 stage B): the certified search over this module's
/// states — exact-rational re-verification of the search's own prunes, the
/// sat-class and planted-violation negative controls, the raw-vs-certified
/// mutation pin, the level ledger and the tier. A child of this module so it
/// reuses the fixtures and the exact node evaluator above.
mod zono_bab_tests;

/// `#fd-block-eval` (ENGINE_PROGRAM M1): the block evaluator is bit-identical
/// to `eval_node` and the block search walks the reference's tree. A missing
/// or empty module here compiles silently, so the builder's gate asserts its
/// tests BY NAME.
mod zono_block_tests;

mod progressive_state_tests;
/// Exact streaming construction: retained rows and certified objectives are
/// bit-identical to the full state's static-order prefix.
mod ranked_builder_tests;

/// `#fd-shared-state` (ENGINE_PROGRAM M3): the shared class-independent trunk
/// (its bit identity against the frozen pre-M3 builder and the per-class
/// state, containment of shared-slope states, the trunk/tail contract and its
/// mutation pins, garbage-slope fault injection) and the tier's `shared`
/// mode. A missing or empty module here compiles silently, so the gate
/// asserts its tests BY NAME.
mod zono_trunk_tests;

/// `#wide-fd-search` (ENGINE_PROGRAM M4): the frozen-dual tier's level search
/// on an injected untrusted backend, certified in f64 — default bit identity,
/// fault injection (garbage never closes), exact-rational re-verification,
/// mutation pins, fallbacks and thread-count independence. Asserted BY NAME
/// (a missing module compiles silently).
mod zono_wide_tests;

/// B2.3 (ENGINE_PROGRAM §1.7 B1): the tier's write-only prune trace — its
/// completeness and absolute literals (re-evaluated bit for bit by the
/// lane's own evaluator), the search's bit identity with capture on and off,
/// and the lane frame streaming the prunes instead of the claim. Asserted BY
/// NAME (a missing module compiles silently).
mod zono_proof_tests;

#[test]
fn frozen_dual_frame_preparation_obeys_original_deadline() {
    let f = fixture0();
    let deadline = std::time::Instant::now()
        .checked_sub(std::time::Duration::from_millis(1))
        .unwrap();
    assert!(matches!(
        FrozenDualFrame::from_root_until(&f.net, &f.root, &f.ybox, f.t, Some(deadline)),
        Err(ZonoRefusal::Deadline)
    ));
    let frame = FrozenDualFrame::from_root_until(
        &f.net,
        &f.root,
        &f.ybox,
        f.t,
        Some(std::time::Instant::now() + std::time::Duration::from_secs(30)),
    )
    .expect("live original frame");
    assert_eq!(frame.lo, f.lo);
    assert_eq!(frame.hi, f.hi);
    for class in &f.adv {
        let state = state_of(&f, &frame, *class);
        let point = check_point(&f.net, &frame, &state, &f.lo).expect("containment");
        assert!(point.row_excess <= 1e-20 && point.obj_excess <= 1e-20);
    }
}
