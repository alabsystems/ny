// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #root-box-source (ENGINE_PROGRAM M5b): containment, dominance, mutation,
//! fault-injection, determinism and dark-default tests of the certified
//! patch-backward root box source (`margin_row/root_box.rs`).
//!
//! A child of `patch_tests` to reuse its fixtures (`fixtures`, `random_chain`),
//! its dense/patch oracles (`dense_pass`, `patch_walk`,
//! `assert_patch_matches_dense`, `prefix_gates`) and, through it, the exact
//! dyadic-rational forward of `gpu_conv_tests` (`exact_tensors`, `box_points`)
//! and the #root-rounding fixtures of `margin_row::tests`
//! (`cancel_stable_spec`, `rounding_leak_head`, `tiny_targets`).
//!
//! * `root_box_patch_backward_contains_exact_preactivations` — every published
//!   trunk box contains the exact pre-activation at sampled points: every
//!   fixture (chain with input affine, residual tiny, resnet-shaped, random
//!   chains with and without fold charges), symmetric, ASYMMETRIC (both
//!   skews), all-negative and point boxes; the published gates are the
//!   certified construction on the published box (exact-rational chord check).
//! * `root_box_patch_backward_encloses_the_exact_affine_hull_on_cancelling_stable_nets`
//!   — the 96 #root-rounding nets (active relus, cancelling weights, the
//!   asymmetric box `[-1, 0.01]^12`): exact affine hull containment, and the
//!   PARITY-SCRATCH mutation pin must escape on every family.
//! * `root_box_patch_backward_encloses_the_minimal_rounding_leak_on_the_dense_fallback`
//!   — the reviewer's 1-D counterexample, which the patch planner refuses
//!   (`no_gain`), so the DENSE fallback certifies it.
//! * `root_box_e_patch_le_e_dense_on_the_built_prefix` — on the prefix gates
//!   the source itself built, the patch walk's coefficients equal the dense
//!   pass's and its error lane is dominated (`E_patch <= E_dense`), its bounds
//!   are never looser, and the published box is never looser than the dense
//!   pair.
//! * `root_box_tableau_default_is_bit_identical_and_never_enters_the_source`
//!   — the default policy is the tableau build bit for bit and never enters
//!   the module; an armed Parity build declines to the Parity tableau bit for
//!   bit.
//! * `root_box_garbage_alpha_seeds_never_break_containment_or_create_a_close`
//!   — fault injection through the one untrusted input (`#rebake` seeds):
//!   NaN, infinities, huge, out-of-range and wrong-length seeds.
//! * `root_box_patch_backward_is_thread_count_independent` — 1, 3 and 8
//!   rayon threads, each at the working-set chunk width and at TEST-ONLY
//!   chunk caps of 1 and 3 columns, publish the same bits.
//! * `root_box_mutation_pins_are_caught_by_exact_containment` — an inward
//!   shifted backward bound and an unclamped ReLU reference are each caught.
//! * `root_box_crossing_declines_to_the_tableau_bit_for_bit`.
//! * `root_box_patch_backward_baked_splits_contain_the_split_subdomain`.
//! * `root_box_patch_backward_refuses_what_it_cannot_compose_bit_for_bit` —
//!   an armed build that requests Tier-0 retention, the `root_f32` /
//!   `tableau_f32` kernels or `#clip-rows` retention is REFUSED before the
//!   source runs and is the tableau default bit for bit (root and retained
//!   rows); unconflicted, the source builds and publishes neither.
//!
//! Every armed build here holds the SHARED environment lock (or the write lock
//! it takes itself): the root build reads the `#clip-rows` gate per call, so a
//! concurrent test that sets `NY_MARGIN_ROW_CLIP_ROWS` must not be able to
//! turn an armed build into a refused one mid-test.

use num_rational::BigRational as Q;
use num_traits::{ToPrimitive, Zero};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use crate::margin_row::engine::{BackwardEngine, LaneDir};
use crate::margin_row::gpu_conv::GpuConvRoute;
use crate::margin_row::net::{TwinNet, TwinOp};
use crate::margin_row::patch::{PatchPlan, DEFAULT_RATIO};
use crate::margin_row::root::{RetainCfg, RetainedRows, RootGates};
use crate::margin_row::root_box::{
    with_root_box_chunk_cap_for_test, with_root_box_mutation_for_test, with_root_box_tap_for_test,
    LayerTap, Mutation, PatchBackwardConflicts,
};
use crate::margin_row::rounding::RoundMode;
use crate::margin_row::spec::{TwinOpSpec, TwinSpec};
use crate::margin_row::{BackwardIntermRuntimePolicy, RootBoxSource, RootPrecisionRuntimePolicy};

use super::super::super::{cancel_stable_spec, rounding_leak_head, tiny_targets};
use super::super::{box_points, exact_tensors, q};
use super::{
    assert_bits, assert_patch_matches_dense, dense_pass, fixtures, patch_walk, prefix_gates,
    random_chain,
};

// ---------------------------------------------------------------------------
//  Helpers
// ---------------------------------------------------------------------------

/// The armed precision policy: everything default except the box source.
fn patch_policy() -> RootPrecisionRuntimePolicy {
    RootPrecisionRuntimePolicy {
        box_source: RootBoxSource::PatchBackward,
        ..RootPrecisionRuntimePolicy::default()
    }
}

/// One armed build through the scored entry, with explicit policies (no env,
/// no preset cell), plus its per-layer tap.
fn build_armed(
    net: &TwinNet,
    lo: &[f64],
    hi: &[f64],
    mode: RoundMode,
    splits: &[(usize, usize, i8)],
) -> (RootGates, Vec<LayerTap>) {
    let _env = ny_test_utils::env::lock_env_shared();
    let (out, taps) = with_root_box_tap_for_test(|| {
        RootGates::build_retaining_with_policy(
            net,
            lo,
            hi,
            mode,
            None,
            None,
            splits,
            patch_policy(),
            BackwardIntermRuntimePolicy::default(),
            None,
        )
    });
    (out.expect("armed root build").0, taps)
}

fn build_patch(net: &TwinNet, lo: &[f64], hi: &[f64]) -> (RootGates, Vec<LayerTap>) {
    build_armed(net, lo, hi, RoundMode::Outward, &[])
}

/// Candidates and engagement over a build's taps: `(cands, patch, dense)`.
fn engagement(taps: &[LayerTap]) -> (usize, usize, usize) {
    taps.iter().fold((0, 0, 0), |(c, p, d), t| {
        (
            c + t.cand.len(),
            p + t.stats.patch_cols,
            d + t.stats.dense_cols,
        )
    })
}

/// Relu op `op`'s pre-activation tensor id.
fn pre_of(net: &TwinNet, op: usize) -> usize {
    match &net.ops[op] {
        TwinOp::Relu { input, .. } => *input,
        _ => panic!("op {op} is not a relu"),
    }
}

/// Every trunk box of `root` contains the exact pre-activation at every point
/// in `exact` (whole-network exact tensors per point). Returns the checks run.
fn check_boxes(
    what: &str,
    net: &TwinNet,
    root: &RootGates,
    exact: &[Vec<Vec<Q>>],
    failures: &mut Vec<String>,
) -> usize {
    let mut checks = 0usize;
    for (li, layer) in root.layers.iter().enumerate() {
        let tensor = pre_of(net, layer.op);
        for (p, tensors) in exact.iter().enumerate() {
            let z = &tensors[tensor];
            for j in 0..layer.n {
                let (lq, uq) = (q(layer.l[j]), q(layer.u[j]));
                if !(lq <= z[j] && z[j] <= uq) {
                    failures.push(format!(
                        "{what}: layer {li} neuron {j} point {p}: box [{:e}, {:e}] misses the \
                         exact pre-activation {:e}",
                        layer.l[j],
                        layer.u[j],
                        z[j].to_f64().unwrap_or(f64::NAN)
                    ));
                }
                checks += 1;
            }
        }
    }
    checks
}

/// The published gates are the certified construction on the published box:
/// `alpha` in `[0, 1]`, `ms = max(alpha, s)`, stable neurons carry their exact
/// fixed lines, `unst` is exactly the unstable set, and every unstable chord
/// satisfies `c >= 0`, `s l + c >= 0`, `s u + c >= u` in EXACT arithmetic
/// (linear over the convex relu: the kinks decide `s y + c >= relu(y)`).
fn check_gates(what: &str, root: &RootGates, failures: &mut Vec<String>) {
    for (li, g) in root.layers.iter().enumerate() {
        let unst: Vec<usize> = (0..g.n).filter(|&j| g.l[j] < 0.0 && g.u[j] > 0.0).collect();
        if unst != g.unst {
            failures.push(format!(
                "{what}: layer {li}: unst is not the published unstable set"
            ));
        }
        for j in 0..g.n {
            let (a, s, c, ms) = (g.alpha[j], g.s[j], g.c[j], g.ms[j]);
            if !(a.is_finite() && (0.0..=1.0).contains(&a)) || ms.to_bits() != a.max(s).to_bits() {
                failures.push(format!(
                    "{what}: layer {li} neuron {j}: alpha {a:e} / ms {ms:e} vs s {s:e}"
                ));
            }
            if g.l[j] >= 0.0 {
                if (a, s, c) != (1.0, 1.0, 0.0) {
                    failures.push(format!(
                        "{what}: layer {li} neuron {j}: active line {a},{s},{c}"
                    ));
                }
            } else if g.u[j] <= 0.0 {
                if (a, s, c) != (0.0, 0.0, 0.0) {
                    failures.push(format!(
                        "{what}: layer {li} neuron {j}: inactive line {a},{s},{c}"
                    ));
                }
            } else {
                let (lq, uq, sq, cq) = (q(g.l[j]), q(g.u[j]), q(s), q(c));
                if cq < Q::zero() || &sq * &lq + &cq < Q::zero() || &sq * &uq + &cq < uq {
                    failures.push(format!(
                        "{what}: layer {li} neuron {j}: chord s={s:e} c={c:e} does not dominate \
                         relu on [{:e}, {:e}]",
                        g.l[j], g.u[j]
                    ));
                }
            }
        }
    }
}

/// Bit-for-bit equality of two roots' box, gates and unstable sets.
fn assert_roots_bitwise(what: &str, a: &RootGates, b: &RootGates) {
    assert_eq!(a.mode, b.mode, "{what}: mode");
    assert_bits(&format!("{what}: mid"), &a.mid, &b.mid);
    assert_bits(&format!("{what}: rad"), &a.rad, &b.rad);
    assert_eq!(a.layers.len(), b.layers.len(), "{what}: layer count");
    for (li, (x, y)) in a.layers.iter().zip(&b.layers).enumerate() {
        assert_eq!((x.op, x.n), (y.op, y.n), "{what}: layer {li} shape");
        assert_bits(&format!("{what}: layer {li} l"), &x.l, &y.l);
        assert_bits(&format!("{what}: layer {li} u"), &x.u, &y.u);
        assert_bits(&format!("{what}: layer {li} alpha"), &x.alpha, &y.alpha);
        assert_bits(&format!("{what}: layer {li} s"), &x.s, &y.s);
        assert_bits(&format!("{what}: layer {li} c"), &x.c, &y.c);
        assert_bits(&format!("{what}: layer {li} ms"), &x.ms, &y.ms);
        assert_eq!(x.unst, y.unst, "{what}: layer {li} unst");
        assert_eq!(
            x.clip_rows.is_some(),
            y.clip_rows.is_some(),
            "{what}: layer {li} clip rows"
        );
    }
}

/// `(lo, hi)` around `center` with lower reach `a` and upper reach `b`,
/// shifted by `shift`.
fn skewed(center: &[f64], shift: f64, a: f64, b: f64) -> (Vec<f64>, Vec<f64>) {
    (
        center.iter().map(|c| c + shift - a).collect(),
        center.iter().map(|c| c + shift + b).collect(),
    )
}

/// EXACT hull of the first op's outputs when it is a conv on the input: the
/// op is affine, so the vertex sum `b + sum min/max(w lo, w hi)` is exact.
fn exact_first_conv_hull(spec: &TwinSpec, lo: &[f64], hi: &[f64]) -> Option<Vec<(Q, Q)>> {
    let TwinOpSpec::Conv {
        input: 0,
        weight,
        bias,
        kernel: (co, ci, kh, kw),
        stride,
        pads,
        ishape: (_, ih, iw),
        oshape: (_, oh, ow),
        ..
    } = &spec.ops[0]
    else {
        return None;
    };
    let mut out = Vec::with_capacity(co * oh * ow);
    for oc in 0..*co {
        for oy in 0..*oh {
            for ox in 0..*ow {
                let (mut mn, mut mx) = (q(bias[oc]), q(bias[oc]));
                for c in 0..*ci {
                    for ky in 0..*kh {
                        for kx in 0..*kw {
                            let iy = (oy * stride.0 + ky) as isize - pads.0 as isize;
                            let ix = (ox * stride.1 + kx) as isize - pads.1 as isize;
                            if iy < 0 || ix < 0 || iy as usize >= *ih || ix as usize >= *iw {
                                continue;
                            }
                            let i = c * ih * iw + iy as usize * iw + ix as usize;
                            let wq = q(weight[((oc * ci + c) * kh + ky) * kw + kx]);
                            let (ta, tb) = (&wq * &q(lo[i]), &wq * &q(hi[i]));
                            if ta <= tb {
                                mn += ta;
                                mx += tb;
                            } else {
                                mn += tb;
                                mx += ta;
                            }
                        }
                    }
                }
                out.push((mn, mx));
            }
        }
    }
    Some(out)
}

// ---------------------------------------------------------------------------
//  Tests
// ---------------------------------------------------------------------------

/// THE LOAD-BEARING CONTAINMENT. Every fixture, every box shape (symmetric,
/// asymmetric both ways, shifted all-negative, point): every published trunk
/// box contains the exact pre-activation of the stored network at every
/// sampled point (centre, corners, random corners, interior), and the
/// published gates are the certified construction on the published box. VOID
/// guards: some candidate must have run as a PATCH walk, and every fixture
/// must have produced a candidate somewhere.
#[test]
fn root_box_patch_backward_contains_exact_preactivations() {
    let mut rng = StdRng::seed_from_u64(0x800C_B0C5_0001);
    let mut checks = 0usize;
    let mut failures: Vec<String> = Vec::new();
    let (mut cands, mut patch_cols, mut dense_cols) = (0usize, 0usize, 0usize);
    for (name, spec, rad0) in fixtures() {
        let net = TwinNet::compile(&spec).expect("compile");
        let center: Vec<f64> = (0..spec.n_in)
            .map(|_| rng.random_range(-0.5..0.5))
            .collect();
        let mut fixture_cands = 0usize;
        for (shift, a, b) in [
            (0.0, rad0, rad0),
            (0.0, rad0, 0.15 * rad0),
            (0.0, 0.15 * rad0, rad0),
            (-1.25, 0.5 * rad0, 0.1 * rad0),
            (0.0, 0.0, 0.0),
        ] {
            let (lo, hi) = skewed(&center, shift, a, b);
            let what = format!("{name} box shift {shift} -{a:e}/+{b:e}");
            let (root, taps) = build_patch(&net, &lo, &hi);
            assert_eq!(root.layers.len(), net.trunk_relus.len(), "{what}: layers");
            assert_eq!(taps.len(), root.layers.len(), "{what}: one tap per layer");
            // SHRINK-ONLY: every published box lies inside its certified
            // reference, and the candidates are exactly its unstable set.
            for (tap, g) in taps.iter().zip(&root.layers) {
                assert_eq!(tap.op, g.op, "{what}: tap order");
                for j in 0..g.n {
                    assert!(
                        g.l[j] >= tap.ref_l[j] && g.u[j] <= tap.ref_u[j],
                        "{what}: layer {} neuron {j}: published [{:e}, {:e}] leaves the \
                         reference [{:e}, {:e}]",
                        tap.layer,
                        g.l[j],
                        g.u[j],
                        tap.ref_l[j],
                        tap.ref_u[j]
                    );
                }
                let unstable: Vec<usize> = (0..g.n)
                    .filter(|&j| tap.ref_l[j] < 0.0 && tap.ref_u[j] > 0.0)
                    .collect();
                assert_eq!(tap.cand, unstable, "{what}: candidates");
            }
            let (c, p, d) = engagement(&taps);
            fixture_cands += c;
            cands += c;
            patch_cols += p;
            dense_cols += d;
            let extra = if a + b > 0.0 { 5 } else { 0 };
            let exact: Vec<Vec<Vec<Q>>> = box_points(&mut rng, &lo, &hi, extra)
                .iter()
                .map(|x| exact_tensors(&spec, x))
                .collect();
            checks += check_boxes(&what, &net, &root, &exact, &mut failures);
            check_gates(&what, &root, &mut failures);
        }
        assert!(
            fixture_cands > 0,
            "VOID: fixture {name} never produced a reference-unstable candidate, so no \
             backward pass ran on it"
        );
    }
    eprintln!(
        "[root-box-test] containment: {checks} exact checks, {} failures, candidates {cands}, \
         patch cols {patch_cols}, dense cols {dense_cols}",
        failures.len()
    );
    assert!(
        failures.is_empty(),
        "PATCH-BACKWARD ROOT BOX MISSES THE EXACT PRE-ACTIVATION (false-unsat generator): \
         {failures:#?}"
    );
    assert!(checks > 5_000, "only {checks} exact checks ran");
    assert!(
        patch_cols > 0,
        "VOID: no candidate was certified by a PATCH walk (every layer fell back dense)"
    );
}

/// The #root-rounding sweep on the patch-backward source: 96 nets of
/// `cancel_stable_spec` (3 families x eps in {1e-1 .. 1e-4} x b1 scale x 4
/// seeds) — CANCELLING weights, ALWAYS-ACTIVE trunk-0 relus (proved exactly),
/// the ASYMMETRIC box `[-1, 0.01]^12`, the class that hid the historical root
/// leak. With trunk 0 active the trunk-1 pre-activation is affine, so its
/// exact hull is a vertex sum in BigRational: the published trunk-1 box must
/// contain it on every neuron of every net.
///
/// Premise (VOID otherwise): every trunk-1 neuron is a CANDIDATE (the interval
/// reference straddles 0) and is certified by a PATCH walk, so the backward
/// pass is what binds.
///
/// Mutation pin: the SAME passes on a Parity scratch (no error lane, raw
/// concretize) must miss the hull on some net of EVERY family (exact replica
/// of the walk order, `scratchpad/m5b/parity_teeth.py`: 32/32 plain, 32/32
/// affine, 27/32 residual nets escape, worst -1.2e-15), so the containment
/// check sees the uncharged-rounding class on every repaired site (conv +
/// gate; + channel affine; + residual Add).
#[test]
fn root_box_patch_backward_encloses_the_exact_affine_hull_on_cancelling_stable_nets() {
    let (cin, p, cw) = (3usize, 4usize, 8usize);
    let (lo_v, hi_v) = (-1.0f64, 0.01f64);
    let (qlo, qhi) = (q(lo_v), q(hi_v));
    let lo = vec![lo_v; cin * p];
    let hi = vec![hi_v; cin * p];
    let mut nets = [0usize; 3];
    let mut mutated_escapes = [0usize; 3];
    let mut failures: Vec<String> = Vec::new();
    let mut min_head = f64::INFINITY;
    let mut checked = 0usize;
    for family in 0u8..3 {
        for eps in [1e-1f64, 1e-2, 1e-3, 1e-4] {
            for bscale in [0.05f64, 0.5] {
                for seed in 0u64..4 {
                    let tag = format!("family {family} eps {eps:e} b1~{bscale} seed {seed}");
                    let (spec, cs) = cancel_stable_spec(seed, eps, bscale, family);
                    let net = TwinNet::compile(&spec).expect("compile");
                    for c in 0..cw {
                        let mut mn = q(cs.b1[c]);
                        for i in 0..cin {
                            let wv = q(cs.w1[c * cin + i]);
                            let (ta, tb) = (&wv * &qlo, &wv * &qhi);
                            mn += if ta <= tb { ta } else { tb };
                        }
                        assert!(
                            mn > Q::zero(),
                            "premise: {tag}: trunk-0 channel {c} not active"
                        );
                    }
                    let mut hull: Vec<(Q, Q)> = Vec::with_capacity(cw * p);
                    for n in 0..cw {
                        let mut arow: Vec<Q> = (0..cw).map(|c| q(cs.w2[n * cw + c])).collect();
                        if cs.residual {
                            arow[n] += q(1.0);
                        }
                        let mut cst = Q::zero();
                        let mut coef: Vec<Q> = vec![Q::zero(); cin];
                        for (c, ac) in arow.iter().enumerate() {
                            cst += ac * q(cs.b1[c]);
                            for (i, co) in coef.iter_mut().enumerate() {
                                *co += ac * q(cs.w1[c * cin + i]);
                            }
                        }
                        if let Some(sc) = &cs.scale {
                            let s = q(sc[n]);
                            for co in coef.iter_mut() {
                                *co = &*co * &s;
                            }
                            cst = &cst * &s;
                        }
                        let (mut mn, mut mx) = (cst.clone(), cst);
                        for co in &coef {
                            let (ta, tb) = (co * &qlo, co * &qhi);
                            if ta <= tb {
                                mn += ta;
                                mx += tb;
                            } else {
                                mn += tb;
                                mx += ta;
                            }
                        }
                        for _ in 0..p {
                            hull.push((mn.clone(), mx.clone()));
                        }
                    }
                    let (fixed, taps) = build_patch(&net, &lo, &hi);
                    let mutated = with_root_box_mutation_for_test(Mutation::ParityScratch, || {
                        build_patch(&net, &lo, &hi).0
                    });
                    let fi = usize::from(family);
                    nets[fi] += 1;
                    for (arm, root) in [("fixed", &fixed), ("mutated", &mutated)] {
                        assert_eq!(root.layers.len(), 2, "{arm}: {tag}: two trunk relus");
                        assert!(
                            root.layers[0].l.iter().all(|&v| v > 0.0),
                            "{arm}: {tag}: trunk 0 is no longer stable-active"
                        );
                        assert_eq!(root.layers[1].n, hull.len(), "{arm}: {tag}: width");
                    }
                    assert_eq!(taps.len(), 2, "{tag}: taps");
                    assert_eq!(
                        (taps[1].cand.len(), taps[1].stats.patch_cols),
                        (hull.len(), hull.len()),
                        "VOID premise: {tag}: every trunk-1 neuron must be a candidate certified \
                         by a PATCH walk"
                    );
                    let mut missed = false;
                    for (j, (mn, mx)) in hull.iter().enumerate() {
                        let (ml, mu) = (q(mutated.layers[1].l[j]), q(mutated.layers[1].u[j]));
                        if !(ml <= *mn && mu >= *mx) {
                            missed = true;
                        }
                        let (fl, fu) = (fixed.layers[1].l[j], fixed.layers[1].u[j]);
                        let (flq, fuq) = (q(fl), q(fu));
                        checked += 1;
                        let hd = (mn - &flq).min(&fuq - mx);
                        min_head = min_head.min(hd.to_f64().unwrap_or(f64::NEG_INFINITY));
                        if !(flq <= *mn && fuq >= *mx) {
                            failures.push(format!(
                                "{tag} neuron {j}: box [{fl:e}, {fu:e}] vs exact [{:e}, {:e}]",
                                mn.to_f64().unwrap_or(f64::NAN),
                                mx.to_f64().unwrap_or(f64::NAN),
                            ));
                        }
                    }
                    if missed {
                        mutated_escapes[fi] += 1;
                    }
                }
            }
        }
    }
    eprintln!(
        "[root-box-test] cancelling stable-active sweep: nets {nets:?}, parity-scratch nets \
         escaping {mutated_escapes:?}, neurons checked {checked}, failures {}, min headroom \
         {min_head:+.3e}",
        failures.len()
    );
    assert_eq!(nets, [32, 32, 32], "sweep did not run every net");
    assert!(
        mutated_escapes.iter().all(|&e| e > 0),
        "VOID: the parity-scratch mutation contains the exact hull on every net of some family \
         ({mutated_escapes:?} escaping of {nets:?}), so that family no longer detects an \
         uncharged backward rounding"
    );
    assert!(
        failures.is_empty(),
        "PATCH-BACKWARD TRUNK-1 BOX MISSES THE EXACT STORED-NETWORK HULL (false-unsat \
         generator): {failures:#?}"
    );
}

/// The reviewer's hand-checkable #root-rounding counterexample (`x in [-1,
/// 0.01]`, `z2 = relu(-x + 0.02) - relu((-1 + 1e-6) x + 0.02) = -1e-6 x` with
/// both trunk-0 relus active) on the patch-backward source. Its 1x1 spatial
/// layers give the patch planner nothing to gain (`no_gain`), so the trunk-1
/// candidate is certified by the DENSE fallback — pinned — and its box must
/// contain the exact affine hull.
#[test]
fn root_box_patch_backward_encloses_the_minimal_rounding_leak_on_the_dense_fallback() {
    let w1 = [-1.0f64, -1.0 + 1e-6];
    let b1 = [0.02f64, 0.02];
    let w2 = [1.0f64, -1.0];
    let mut ops = vec![
        TwinOpSpec::Conv {
            input: 0,
            weight: w1.to_vec(),
            bias: b1.to_vec(),
            bias_err: vec![0.0; 2],
            weight_rel_err: 0.0,
            kernel: (2, 1, 1, 1),
            stride: (1, 1),
            pads: (0, 0, 0, 0),
            ishape: (1, 1, 1),
            oshape: (2, 1, 1),
        },
        TwinOpSpec::Relu { input: 1 },
        TwinOpSpec::Conv {
            input: 2,
            weight: w2.to_vec(),
            bias: vec![0.0],
            bias_err: vec![0.0],
            weight_rel_err: 0.0,
            kernel: (1, 2, 1, 1),
            stride: (1, 1),
            pads: (0, 0, 0, 0),
            ishape: (2, 1, 1),
            oshape: (1, 1, 1),
        },
        TwinOpSpec::Relu { input: 3 },
    ];
    rounding_leak_head(&mut ops, 4, 1, 0x0001_EA4F);
    let spec = TwinSpec { n_in: 1, ops };
    let net = TwinNet::compile(&spec).expect("compile");
    let (lo, hi) = ([-1.0f64], [0.01f64]);
    let coef = q(w2[0]) * q(w1[0]) + q(w2[1]) * q(w1[1]);
    let cst = q(w2[0]) * q(b1[0]) + q(w2[1]) * q(b1[1]);
    let (va, vb) = (&coef * q(lo[0]) + &cst, &coef * q(hi[0]) + &cst);
    let (mn, mx) = if va <= vb { (va, vb) } else { (vb, va) };
    let (root, taps) = build_patch(&net, &lo, &hi);
    assert_eq!(root.layers.len(), 2, "two trunk relus");
    assert!(
        root.layers[0].l.iter().all(|&v| v > 0.0),
        "premise: both trunk-0 relus stable-active: {:?}",
        root.layers[0].l
    );
    assert_eq!(taps.len(), 2, "taps");
    assert_eq!(
        taps[1].cand,
        vec![0],
        "VOID premise: the trunk-1 neuron is a candidate"
    );
    assert_eq!(
        (taps[1].stats.plan_refused, taps[1].stats.dense_cols),
        (Some("no_gain"), 1),
        "VOID premise: the trunk-1 candidate is certified by the DENSE fallback"
    );
    let (l, u) = (root.layers[1].l[0], root.layers[1].u[0]);
    eprintln!(
        "[root-box-test] minimal leak (dense fallback): exact [{:e}, {:e}] box [{l:e}, {u:e}] \
         headroom lo {:+.3e} up {:+.3e}",
        mn.to_f64().unwrap_or(f64::NAN),
        mx.to_f64().unwrap_or(f64::NAN),
        (&mn - q(l)).to_f64().unwrap_or(f64::NAN),
        (q(u) - &mx).to_f64().unwrap_or(f64::NAN),
    );
    assert!(
        q(l) <= mn && q(u) >= mx,
        "PATCH-BACKWARD TRUNK-1 BOX MISSES THE EXACT HULL (false-unsat generator): box [{l:e}, \
         {u:e}] vs exact [{:e}, {:e}]",
        mn.to_f64().unwrap_or(f64::NAN),
        mx.to_f64().unwrap_or(f64::NAN),
    );
}

/// `E_patch <= E_dense` ON THE PREFIX THE SOURCE BUILT. For every fixture and
/// every layer with candidates, the patch walk over the source's own frozen
/// prefix gates (layers `0..li` of the published root) has EXACTLY the dense
/// pass's coefficients and bias and a DOMINATED error lane and bias error
/// (`assert_patch_matches_dense`), in both lanes; its concretized bounds are
/// never looser than the dense pair's; and the published box of every
/// candidate is never looser than the dense pair (it is the reference
/// intersected with the patch-or-dense certified bound).
#[test]
fn root_box_e_patch_le_e_dense_on_the_built_prefix() {
    let mut rng = StdRng::seed_from_u64(0x800C_B0C5_0004);
    let mut compared = 0usize;
    let off = GpuConvRoute::Off;
    for (name, spec, rad0) in fixtures() {
        let net = TwinNet::compile(&spec).expect("compile");
        let center: Vec<f64> = (0..spec.n_in)
            .map(|_| rng.random_range(-0.5..0.5))
            .collect();
        let (lo, hi) = skewed(&center, 0.0, rad0, 0.4 * rad0);
        let (root, taps) = build_patch(&net, &lo, &hi);
        // Layers the source certified entirely by patch walks (a layer with a
        // refused plan or an unexpressed chunk has no whole-layer patch walk).
        for tap in taps.iter().filter(|t| {
            !t.cand.is_empty() && t.stats.plan_refused.is_none() && t.stats.dense_cols == 0
        }) {
            let li = tap.layer;
            let what = format!("{name} layer {li}");
            assert_eq!(
                tap.stats.unavailable, 0,
                "{what}: premise: every candidate was certified"
            );
            // The source's own frozen prefix (layers 0..li of the published
            // root), in the certified mode, per-column error lane.
            let scratch = crate::margin_row::root_certify_once::untrusted_scratch(&root);
            let tmp = RootGates {
                mode: RoundMode::Outward,
                layers: prefix_gates(&root, li),
                ..scratch
            };
            let Ok(plan) = PatchPlan::build(&net, tap.op, DEFAULT_RATIO) else {
                continue;
            };
            let cols = &tap.cand;
            let lower_walk = patch_walk(&plan, &net, &tmp, cols, true, None);
            let upper_walk = patch_walk(&plan, &net, &tmp, cols, false, None);
            let lower_dense = dense_pass(&net, &tmp, tap.op, cols, LaneDir::Lower, None);
            let upper_dense = dense_pass(&net, &tmp, tap.op, cols, LaneDir::Upper, None);
            assert_patch_matches_dense(
                &format!("{what} lower"),
                &lower_walk,
                &lower_dense,
                net.n_in,
            );
            assert_patch_matches_dense(
                &format!("{what} upper"),
                &upper_walk,
                &upper_dense,
                net.n_in,
            );
            let lowers = vec![true; cols.len()];
            let uppers = vec![false; cols.len()];
            let lb_p = PatchPlan::concretize(&tmp, &lower_walk, &lowers);
            let ub_p = PatchPlan::concretize(&tmp, &upper_walk, &uppers);
            let eng = BackwardEngine::new(&net, &tmp).with_gpu_conv_route(off.clone());
            let lb_d = eng.concretize_lower(&lower_dense);
            let ub_d = eng.concretize_upper(&upper_dense);
            let published = &root.layers[li];
            for (k, &j) in cols.iter().enumerate() {
                assert!(
                    lb_p[k] >= lb_d[k] && ub_p[k] <= ub_d[k],
                    "{what} neuron {j}: patch [{:e}, {:e}] looser than dense [{:e}, {:e}]",
                    lb_p[k],
                    ub_p[k],
                    lb_d[k],
                    ub_d[k]
                );
                assert!(
                    published.l[j] >= lb_d[k] && published.u[j] <= ub_d[k],
                    "{what} neuron {j}: published [{:e}, {:e}] looser than the dense pair \
                     [{:e}, {:e}]",
                    published.l[j],
                    published.u[j],
                    lb_d[k],
                    ub_d[k]
                );
                compared += 1;
            }
        }
    }
    eprintln!("[root-box-test] E_patch <= E_dense on the built prefix: {compared} candidates");
    assert!(
        compared > 50,
        "VOID: only {compared} candidates were compared"
    );
}

/// DARK BY DEFAULT, AND BIT-IDENTICAL. The default precision policy (box
/// source `tableau`) never enters the patch-backward module (no tap record)
/// and publishes EXACTLY the tableau oracle entry's bits, symmetric and
/// asymmetric boxes alike; the explicit `Tableau` policy likewise. An ARMED
/// Parity build declines (Parity is not a certificate) and publishes the
/// Parity tableau's bits.
#[test]
fn root_box_tableau_default_is_bit_identical_and_never_enters_the_source() {
    let mut rng = StdRng::seed_from_u64(0x800C_B0C5_0005);
    let mut builds = 0usize;
    for (name, spec, rad0) in fixtures() {
        let net = TwinNet::compile(&spec).expect("compile");
        let center: Vec<f64> = (0..spec.n_in)
            .map(|_| rng.random_range(-0.5..0.5))
            .collect();
        for (a, b) in [(rad0, rad0), (rad0, 0.2 * rad0)] {
            let (lo, hi) = skewed(&center, 0.0, a, b);
            let what = format!("{name} -{a:e}/+{b:e}");
            let oracle = RootGates::build_prec(&net, &lo, &hi, RoundMode::Outward, None, false)
                .expect("tableau oracle");
            for policy in [
                RootPrecisionRuntimePolicy::default(),
                RootPrecisionRuntimePolicy {
                    box_source: RootBoxSource::Tableau,
                    ..RootPrecisionRuntimePolicy::default()
                },
            ] {
                let (built, taps) = with_root_box_tap_for_test(|| {
                    RootGates::build_retaining_with_policy(
                        &net,
                        &lo,
                        &hi,
                        RoundMode::Outward,
                        None,
                        None,
                        &[],
                        policy,
                        BackwardIntermRuntimePolicy::default(),
                        None,
                    )
                });
                let built = built.expect("default root").0;
                assert!(
                    taps.is_empty(),
                    "{what}: the default policy entered the patch-backward source"
                );
                assert_roots_bitwise(&format!("{what} default"), &built, &oracle);
                builds += 1;
            }
            let parity_oracle =
                RootGates::build_prec(&net, &lo, &hi, RoundMode::Parity, None, false)
                    .expect("parity tableau");
            let (parity_armed, taps) = build_armed(&net, &lo, &hi, RoundMode::Parity, &[]);
            assert!(
                taps.is_empty(),
                "{what}: an armed Parity build must decline"
            );
            assert_roots_bitwise(&format!("{what} parity"), &parity_armed, &parity_oracle);
        }
    }
    assert!(builds >= 20, "only {builds} default builds ran");
}

/// FAULT INJECTION through the ONE untrusted input the source admits, the
/// `#rebake` alpha seed: NaN, both infinities, +-1e300, out-of-range, exact
/// endpoints and wrong-length vectors, on the armed rebake entry. Every
/// published box still contains the exact pre-activations, every gate is the
/// certified construction (alpha in [0, 1], `ms = max(alpha, s)`, stable
/// neurons exact, chords dominating), and the certified root bound `dj`
/// computed on the result never exceeds the EXACT margin at any sampled point
/// — garbage never creates a close.
#[test]
fn root_box_garbage_alpha_seeds_never_break_containment_or_create_a_close() {
    let mut rng = StdRng::seed_from_u64(0x800C_B0C5_0006);
    let garbage = [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        1e300,
        -1e300,
        -0.5,
        1.5,
        0.0,
        1.0,
        0.3,
        f64::MIN_POSITIVE,
        -0.0,
    ];
    let mut checks = 0usize;
    let mut margins = 0usize;
    let mut failures: Vec<String> = Vec::new();
    for (name, spec, rad0) in fixtures() {
        let net = TwinNet::compile(&spec).expect("compile");
        let center: Vec<f64> = (0..spec.n_in)
            .map(|_| rng.random_range(-0.5..0.5))
            .collect();
        let (lo, hi) = skewed(&center, 0.0, rad0, 0.5 * rad0);
        for variant in 0..3usize {
            let seed: Vec<Vec<f64>> = net
                .trunk_relus
                .iter()
                .enumerate()
                .map(|(li, &op)| {
                    let n = net.tsize[op + 1];
                    // Variant 2 hands every other layer a WRONG-LENGTH seed.
                    let len = if variant == 2 && li % 2 == 1 {
                        n + 1
                    } else {
                        n
                    };
                    (0..len)
                        .map(|j| {
                            if variant == 1 {
                                rng.random_range(-3.0..3.0)
                            } else {
                                garbage[(j + li) % garbage.len()]
                            }
                        })
                        .collect()
                })
                .collect();
            let what = format!("{name} seed variant {variant}");
            let env = ny_test_utils::env::lock_env_shared();
            let (built, taps) = with_root_box_tap_for_test(|| {
                RootGates::build_rebaked(
                    &net,
                    &lo,
                    &hi,
                    RoundMode::Outward,
                    None,
                    None,
                    &seed,
                    None,
                    patch_policy(),
                )
            });
            drop(env);
            let root = built.expect("rebaked armed root").0;
            assert_eq!(taps.len(), root.layers.len(), "{what}: the source built it");
            let pts = box_points(&mut rng, &lo, &hi, 4);
            let exact: Vec<Vec<Vec<Q>>> = pts.iter().map(|x| exact_tensors(&spec, x)).collect();
            checks += check_boxes(&what, &net, &root, &exact, &mut failures);
            check_gates(&what, &root, &mut failures);
            let (t, _, adv) = tiny_targets(&net, &lo, &hi);
            let eng = BackwardEngine::new(&net, &root);
            let re = crate::margin_row::bab::root_eval(&eng, &net, t, &adv).expect("root_eval");
            for (p, tensors) in exact.iter().enumerate() {
                let scores = tensors.last().expect("scores");
                for (k, &o) in adv.iter().enumerate() {
                    let exact_margin = &scores[t] - &scores[o];
                    if q(re.dj[k]) > exact_margin {
                        failures.push(format!(
                            "{what}: certified dj[{k}] = {:e} exceeds the EXACT margin at point \
                             {p}: a garbage seed created a close",
                            re.dj[k]
                        ));
                    }
                    margins += 1;
                }
            }
        }
    }
    eprintln!(
        "[root-box-test] garbage seeds: {checks} box checks, {margins} margin checks, {} failures",
        failures.len()
    );
    assert!(
        failures.is_empty(),
        "GARBAGE SEED BROKE A CERTIFICATE: {failures:#?}"
    );
    assert!(
        checks > 1_000 && margins > 100,
        "VOID: {checks} / {margins} checks"
    );
}

/// THREAD-COUNT AND CHUNK-COMPOSITION INDEPENDENCE. The chunk width and wave
/// concurrency depend on the rayon pool size (they share the working set),
/// but every column's arithmetic is column-local and every failed or
/// unexpressed chunk is retried column by column, so 1, 3 and 8 threads, each
/// with the working-set width and with chunks capped at 1 and 3 columns
/// (TEST-ONLY cap), publish the same bits. On these small fixtures the
/// working-set width alone always holds a whole layer in one chunk, so the cap
/// is what makes the composition vary (VOID guard: some capped layer must
/// have been split into several chunks).
#[test]
fn root_box_patch_backward_is_thread_count_independent() {
    let mut rng = StdRng::seed_from_u64(0x800C_B0C5_0007);
    let mut compared = 0usize;
    let mut split_layers = 0usize;
    for (name, spec, rad0) in fixtures() {
        let net = TwinNet::compile(&spec).expect("compile");
        let center: Vec<f64> = (0..spec.n_in)
            .map(|_| rng.random_range(-0.5..0.5))
            .collect();
        let (lo, hi) = skewed(&center, 0.0, rad0, 0.3 * rad0);
        let mut first: Option<RootGates> = None;
        for threads in [1usize, 3, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("rayon pool");
            for cap in [None, Some(1usize), Some(3)] {
                let (root, taps) = pool.install(|| match cap {
                    Some(c) => with_root_box_chunk_cap_for_test(c, || build_patch(&net, &lo, &hi)),
                    None => build_patch(&net, &lo, &hi),
                });
                if let Some(c) = cap {
                    split_layers += taps.iter().filter(|t| t.cand.len() > c).count();
                }
                match first.as_ref() {
                    None => first = Some(root),
                    Some(base) => {
                        assert_roots_bitwise(
                            &format!("{name} threads {threads} chunk cap {cap:?}"),
                            &root,
                            base,
                        );
                        compared += 1;
                    }
                }
            }
        }
    }
    assert!(compared >= 50, "only {compared} comparisons ran");
    assert!(
        split_layers > 0,
        "VOID: no capped layer had more candidates than its cap, so chunk composition never varied"
    );
}

/// The per-layer taps of one armed Outward build, whatever it published (the
/// tap records every layer the source finished before any decline).
fn taps_of(net: &TwinNet, lo: &[f64], hi: &[f64]) -> Vec<LayerTap> {
    let _env = ny_test_utils::env::lock_env_shared();
    let (_, taps) = with_root_box_tap_for_test(|| {
        RootGates::build_retaining_with_policy(
            net,
            lo,
            hi,
            RoundMode::Outward,
            None,
            None,
            &[],
            patch_policy(),
            BackwardIntermRuntimePolicy::default(),
            None,
        )
    });
    taps
}

/// MUTATION PINS: the exact containment checks detect the two defects the
/// source's own composition could plausibly carry.
///
/// * An inward-shifted backward bound (`ShiftBackward(1e-9)`: far above every
///   certified slack on these nets, far below every candidate's width, so
///   nothing crosses at trunk 0): the published TRUNK-0 box of some candidate
///   must miss its EXACT affine hull (read from the per-layer tap, which
///   records trunk 0 before any later layer could decline). The unmutated
///   source must contain every hull.
/// * An UNCLAMPED ReLU reference (the output box carried as the raw
///   pre-activation box): on a net whose trunk-0 relu is stable-inactive
///   (`z0 = x - 3` on `[-1, 1]`) and whose trunk-1 pre-activation is exactly
///   `0.5`, the mutated reference claims `[-3.5, -1.5]`, certifies the neuron
///   stable-inactive and publishes a box that misses 0.5 (nothing is a
///   candidate, so nothing can cross). The unmutated source must contain it.
#[test]
fn root_box_mutation_pins_are_caught_by_exact_containment() {
    // (a) The shifted backward bound, on random conv chains.
    let mut rng = StdRng::seed_from_u64(0x800C_B0C5_0008);
    let (mut cands0, mut escapes, mut fixed_checks) = (0usize, 0usize, 0usize);
    for folded in [false, true] {
        for _ in 0..6 {
            let spec = random_chain(&mut rng, false, folded);
            let net = TwinNet::compile(&spec).expect("compile");
            let center: Vec<f64> = (0..spec.n_in)
                .map(|_| rng.random_range(-0.3..0.3))
                .collect();
            let (lo, hi) = skewed(&center, 0.0, 0.3, 0.2);
            let hull = exact_first_conv_hull(&spec, &lo, &hi).expect("a conv on the input");
            let (fixed, taps) = build_patch(&net, &lo, &hi);
            assert_eq!(hull.len(), fixed.layers[0].n, "hull width");
            for (j, (mn, mx)) in hull.iter().enumerate() {
                let (l, u) = (q(fixed.layers[0].l[j]), q(fixed.layers[0].u[j]));
                assert!(
                    l <= *mn && u >= *mx,
                    "UNMUTATED TRUNK-0 BOX MISSES ITS EXACT HULL at neuron {j}: [{:e}, {:e}]",
                    fixed.layers[0].l[j],
                    fixed.layers[0].u[j]
                );
                fixed_checks += 1;
            }
            cands0 += taps.first().map_or(0, |t| t.cand.len());
            let mutated = with_root_box_mutation_for_test(Mutation::ShiftBackward(1e-9), || {
                taps_of(&net, &lo, &hi)
            });
            if let Some(t0) = mutated.first() {
                assert_eq!(t0.layer, 0, "the first tap is trunk 0");
                for &j in &t0.cand {
                    let (mn, mx) = &hull[j];
                    if q(t0.out_l[j]) > *mn || q(t0.out_u[j]) < *mx {
                        escapes += 1;
                    }
                }
            }
        }
    }
    eprintln!(
        "[root-box-test] shifted-bound pin: {fixed_checks} unmutated hull checks, {cands0} \
         trunk-0 candidates, {escapes} mutated escapes"
    );
    assert!(
        cands0 > 0,
        "VOID: no trunk-0 candidate, the pin cannot see the mutation"
    );
    assert!(
        escapes > 0,
        "VOID: an inward-shifted certified backward bound was NOT caught by the exact trunk-0 \
         hull check"
    );

    // (b) The unclamped ReLU reference, on a hand-built stable-inactive net.
    let conv = |input: usize, bias: f64| TwinOpSpec::Conv {
        input,
        weight: vec![1.0],
        bias: vec![bias],
        bias_err: vec![0.0],
        weight_rel_err: 0.0,
        kernel: (1, 1, 1, 1),
        stride: (1, 1),
        pads: (0, 0, 0, 0),
        ishape: (1, 1, 1),
        oshape: (1, 1, 1),
    };
    let mut ops = vec![
        conv(0, -3.0),
        TwinOpSpec::Relu { input: 1 },
        conv(2, 0.5),
        TwinOpSpec::Relu { input: 3 },
    ];
    rounding_leak_head(&mut ops, 4, 1, 0x800C_B0C5);
    let spec = TwinSpec { n_in: 1, ops };
    let net = TwinNet::compile(&spec).expect("compile");
    let (lo, hi) = ([-1.0f64], [1.0f64]);
    let truth = q(0.5);
    let (fixed, _) = build_patch(&net, &lo, &hi);
    let mutated = with_root_box_mutation_for_test(Mutation::UnclampedReluRef, || {
        build_patch(&net, &lo, &hi).0
    });
    for (arm, root) in [("fixed", &fixed), ("mutated", &mutated)] {
        assert_eq!(root.layers.len(), 2, "{arm}: two trunk relus");
        assert!(
            root.layers[0].u[0] < 0.0,
            "{arm}: premise: trunk 0 is stable-inactive, box [{:e}, {:e}]",
            root.layers[0].l[0],
            root.layers[0].u[0]
        );
    }
    let (fl, fu) = (fixed.layers[1].l[0], fixed.layers[1].u[0]);
    assert!(
        q(fl) <= truth && truth <= q(fu),
        "UNMUTATED trunk-1 box [{fl:e}, {fu:e}] misses the exact 0.5"
    );
    let (ml, mu) = (mutated.layers[1].l[0], mutated.layers[1].u[0]);
    assert!(
        !(q(ml) <= truth && truth <= q(mu)),
        "VOID: the unclamped-reference mutation still contains the exact 0.5 ([{ml:e}, {mu:e}]), \
         so the containment check cannot see it"
    );
}

/// FAIL CLOSED ON A CROSSING. With every certified backward bound shifted
/// inward past the reference (`ShiftBackward(1e6)`: every candidate's two
/// certified enclosures cross), the source must DECLINE and the root must be
/// the unchanged tableau build, bit for bit — a crossing is a defect signal,
/// never an emptiness certificate and never a published box.
#[test]
fn root_box_crossing_declines_to_the_tableau_bit_for_bit() {
    let mut rng = StdRng::seed_from_u64(0x800C_B0C5_0009);
    let mut declined = 0usize;
    for (name, spec, rad0) in fixtures() {
        let net = TwinNet::compile(&spec).expect("compile");
        let center: Vec<f64> = (0..spec.n_in)
            .map(|_| rng.random_range(-0.5..0.5))
            .collect();
        let (lo, hi) = skewed(&center, 0.0, rad0, rad0);
        if engagement(&taps_of(&net, &lo, &hi)).0 == 0 {
            continue;
        }
        let armed = with_root_box_mutation_for_test(Mutation::ShiftBackward(1e6), || {
            let (root, taps) = build_patch(&net, &lo, &hi);
            assert!(
                taps.len() < root.layers.len(),
                "{name}: the crossed build was not declined (every layer tapped)"
            );
            root
        });
        let oracle = RootGates::build_prec(&net, &lo, &hi, RoundMode::Outward, None, false)
            .expect("tableau oracle");
        assert_roots_bitwise(&format!("{name} crossed"), &armed, &oracle);
        declined += 1;
    }
    assert!(
        declined >= 3,
        "VOID: only {declined} fixtures had a candidate to cross"
    );
}

/// EPOCH TIER-2 REBUILDS carry the source with the split BAKED: the split
/// neuron gets its exact fixed line and leaves `unst`, and every published box
/// contains the exact pre-activations at every sampled point of the SPLIT
/// SUBDOMAIN (points on the split's side of the neuron), for both sides.
#[test]
fn root_box_patch_backward_baked_splits_contain_the_split_subdomain() {
    let mut rng = StdRng::seed_from_u64(0x800C_B0C5_000A);
    let (mut checks, mut kept_total, mut built) = (0usize, 0usize, 0usize);
    let mut failures: Vec<String> = Vec::new();
    for (name, spec, rad0) in fixtures() {
        let net = TwinNet::compile(&spec).expect("compile");
        let center: Vec<f64> = (0..spec.n_in)
            .map(|_| rng.random_range(-0.5..0.5))
            .collect();
        let (lo, hi) = skewed(&center, 0.0, rad0, rad0);
        let (root0, _) = build_patch(&net, &lo, &hi);
        let Some(li) =
            (0..root0.layers.len().saturating_sub(1)).find(|&li| !root0.layers[li].unst.is_empty())
        else {
            continue;
        };
        let j = root0.layers[li].unst[0];
        let tensor = pre_of(&net, root0.layers[li].op);
        let exact: Vec<Vec<Vec<Q>>> = box_points(&mut rng, &lo, &hi, 30)
            .iter()
            .map(|x| exact_tensors(&spec, x))
            .collect();
        for dir in [1i8, -1] {
            let what = format!("{name} split ({li}, {j}, {dir})");
            let (root, taps) = build_armed(&net, &lo, &hi, RoundMode::Outward, &[(li, j, dir)]);
            let g = &root.layers[li];
            let line = if dir > 0 {
                (1.0, 1.0, 0.0)
            } else {
                (0.0, 0.0, 0.0)
            };
            assert_eq!((g.alpha[j], g.s[j], g.c[j]), line, "{what}: baked line");
            assert!(!g.unst.contains(&j), "{what}: the baked neuron left unst");
            built += usize::from(taps.len() == root.layers.len());
            let kept: Vec<Vec<Vec<Q>>> = exact
                .iter()
                .filter(|t| {
                    let z = &t[tensor][j];
                    if dir > 0 {
                        *z >= Q::zero()
                    } else {
                        *z <= Q::zero()
                    }
                })
                .cloned()
                .collect();
            kept_total += kept.len();
            checks += check_boxes(&what, &net, &root, &kept, &mut failures);
        }
    }
    eprintln!(
        "[root-box-test] baked splits: {built} source-built split roots, {kept_total} subdomain \
         points, {checks} checks, {} failures",
        failures.len()
    );
    assert!(
        failures.is_empty(),
        "A SPLIT ROOT BOX MISSES THE EXACT PRE-ACTIVATION ON ITS SUBDOMAIN: {failures:#?}"
    );
    assert!(
        built >= 4,
        "VOID: only {built} split roots were built by the source"
    );
    assert!(
        kept_total > 20,
        "VOID: only {kept_total} subdomain points were checked"
    );
}

/// COMPOSITION REFUSAL (ENGINE_PROGRAM M5b integration): the source builds no
/// `#clip-rows` rows, no Tier-0 retained rows and runs no f32 tableau kernel,
/// so an armed build that REQUESTS any of them is REFUSED before the source
/// runs — it must never publish a root that silently lacks what was asked
/// for. Refused, the build is the tableau default with the same arguments:
/// its root (boxes, gates, unstable sets, clip rows) and its retained rows
/// equal the tableau's bit for bit and the source taps no layer. The
/// `#clip-rows` case arms the governed gate through the environment (the
/// route a preset cannot take, since validation refuses the yaml) and must
/// see clip rows on the tableau oracle, or the refusal protected nothing
/// (VOID). Unconflicted, the source still builds every layer and publishes no
/// clip rows and no retained rows, and the reason tokens are the fixed
/// schema.
#[test]
fn root_box_patch_backward_refuses_what_it_cannot_compose_bit_for_bit() {
    let mut rng = StdRng::seed_from_u64(0x800C_B0C5_000B);
    let (name, spec, rad0) = fixtures()
        .into_iter()
        .find(|(n, _, _)| n.starts_with("resnetish"))
        .expect("the resnet-shaped fixture");
    let net = TwinNet::compile(&spec).expect("compile");
    let center: Vec<f64> = (0..spec.n_in)
        .map(|_| rng.random_range(-0.5..0.5))
        .collect();
    let (lo, hi) = skewed(&center, 0.0, rad0, rad0);
    let build = |retain: Option<&RetainCfg>, precision: RootPrecisionRuntimePolicy| {
        with_root_box_tap_for_test(|| {
            RootGates::build_retaining_with_policy(
                &net,
                &lo,
                &hi,
                RoundMode::Outward,
                None,
                retain,
                &[],
                precision,
                BackwardIntermRuntimePolicy::default(),
                None,
            )
        })
    };
    let armed = |precision: RootPrecisionRuntimePolicy| RootPrecisionRuntimePolicy {
        box_source: RootBoxSource::PatchBackward,
        ..precision
    };
    let assert_retained_equal = |what: &str, a: &Option<RetainedRows>, b: &Option<RetainedRows>| {
        assert_eq!(a.is_some(), b.is_some(), "{what}: retention presence");
        let (Some(a), Some(b)) = (a, b) else {
            return;
        };
        assert_eq!(a.layers.len(), b.layers.len(), "{what}: retained layers");
        for (li, (x, y)) in a.layers.iter().zip(&b.layers).enumerate() {
            assert_eq!(x.eligible, y.eligible, "{what}: layer {li} eligible");
            assert_eq!(x.idx, y.idx, "{what}: layer {li} idx");
            assert_eq!(x.unst_pos, y.unst_pos, "{what}: layer {li} unst_pos");
            assert_eq!(x.naug, y.naug, "{what}: layer {li} naug");
            let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&x.a_l), bits(&y.a_l), "{what}: layer {li} a_l");
            assert_eq!(bits(&x.a_u), bits(&y.a_u), "{what}: layer {li} a_u");
        }
    };

    // Unconflicted: the source builds every layer and publishes neither
    // product (nor was either asked for).
    {
        let _env = ny_test_utils::env::lock_env_shared();
        let (built, taps) = build(None, patch_policy());
        let (root, retained) = built.expect("armed root");
        assert!(retained.is_none(), "{name}: nothing retained unasked");
        assert_eq!(
            taps.len(),
            root.layers.len(),
            "{name}: the unconflicted source must build every layer"
        );
        for (li, g) in root.layers.iter().enumerate() {
            assert!(g.clip_rows.is_none(), "{name}: layer {li}: clip rows");
        }
        assert_eq!(root.backward_interm_execution.chunks_completed, 0);
        assert_eq!(root.backward_interm_execution.columns_completed, 0);
        let ex = root.root_precision_execution;
        assert_eq!((ex.eligible_conv_ops, ex.eligible_store_ops), (0, 0));
        assert!(!ex.tableau_refused && !ex.policy_mismatch, "{name}: {ex:?}");
        assert_eq!(
            root.root_precision_policy.box_source,
            RootBoxSource::PatchBackward,
            "the source must ride the root's precision policy into descendant roots"
        );
    }

    // Tier-0 retention and the two f32 tableau kernels: refused, the tableau
    // default bit for bit.
    let retain_cfg = RetainCfg::default();
    let f32_compute = RootPrecisionRuntimePolicy {
        root_f32: true,
        ..RootPrecisionRuntimePolicy::default()
    };
    let f32_store = RootPrecisionRuntimePolicy {
        tableau_f32: true,
        ..f32_compute
    };
    for (reason, retain, precision) in [
        (
            "retain",
            Some(&retain_cfg),
            RootPrecisionRuntimePolicy::default(),
        ),
        ("root_f32", None, f32_compute),
        ("root_f32+tableau_f32", None, f32_store),
    ] {
        let what = format!("{name} refused {reason}");
        let expected = PatchBackwardConflicts {
            clip_rows: false,
            retain: retain.is_some(),
            root_f32: precision.root_f32,
            tableau_f32: precision.tableau_f32,
        };
        assert!(expected.any(), "{what}: the case must conflict");
        assert_eq!(expected.reason(), reason, "{what}: reason tokens");
        let _env = ny_test_utils::env::lock_env_shared();
        let (oracle, oracle_taps) = build(retain, precision);
        let (oracle, oracle_retained) = oracle.expect("tableau oracle");
        assert!(
            oracle_taps.is_empty(),
            "{what}: the oracle entered the source"
        );
        let (built, taps) = build(retain, armed(precision));
        let (root, retained) = built.expect("refused armed root");
        assert!(
            taps.is_empty(),
            "{what}: the refused build entered the source ({} layers tapped)",
            taps.len()
        );
        assert_roots_bitwise(&what, &root, &oracle);
        assert_retained_equal(&what, &retained, &oracle_retained);
        assert_eq!(
            root.root_precision_execution, oracle.root_precision_execution,
            "{what}: the tableau kernels must have run exactly as unarmed"
        );
        if retain.is_some() {
            let rows: usize = retained
                .as_ref()
                .map_or(0, |r| r.layers.iter().map(|l| l.idx.len()).sum());
            assert!(
                rows > 0,
                "VOID: {what}: the tableau retained no row, so the refusal protected nothing"
            );
        }
    }

    // `#clip-rows` retention through the governed gate: refused, bit for bit.
    // The helper holds the WRITE lock for the whole scope, so nothing inside
    // may take the shared lock (`build` does not). The name comes from the
    // declaration, never a literal, so this test adds no source-frontier row.
    let clip_rows_env = ny_levers::decls::sound_channel::MARGIN_ROW_CLIP_ROWS.name;
    ny_test_utils::env::with_serialized_env_vars(&[(clip_rows_env, "1")], || {
        let what = format!("{name} refused clip_rows");
        let (oracle, _) = build(None, RootPrecisionRuntimePolicy::default());
        let (oracle, _) = oracle.expect("tableau oracle with clip rows");
        assert!(
            oracle.layers.iter().any(|g| g.clip_rows.is_some()),
            "VOID: {what}: the tableau retained no clip rows, so the refusal protected nothing"
        );
        let (built, taps) = build(None, patch_policy());
        let (root, _) = built.expect("refused armed root");
        assert!(
            taps.is_empty(),
            "{what}: the refused build entered the source"
        );
        assert_roots_bitwise(&what, &root, &oracle);
        for (li, (x, y)) in root.layers.iter().zip(&oracle.layers).enumerate() {
            let (Some(x), Some(y)) = (&x.clip_rows, &y.clip_rows) else {
                panic!("{what}: layer {li}: clip rows missing on one side");
            };
            let bits = |a: &ndarray::Array2<f64>| a.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&x.m), bits(&y.m), "{what}: layer {li} clip m");
            assert_eq!(bits(&x.d), bits(&y.d), "{what}: layer {li} clip d");
        }
    });

    // The reason schema: declaration order, `+`-joined, `-` when clear.
    assert!(!PatchBackwardConflicts::default().any());
    assert_eq!(PatchBackwardConflicts::default().reason(), "-");
    let all = PatchBackwardConflicts {
        clip_rows: true,
        retain: true,
        root_f32: true,
        tableau_f32: true,
    };
    assert_eq!(all.reason(), "clip_rows+retain+root_f32+tableau_f32");
    assert_eq!(
        PatchBackwardConflicts::TOKENS.join("+"),
        all.reason(),
        "the tokens are the reason vocabulary"
    );
}
