// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Seam tests. Five families, in the order the risk runs:
//!
//! * (a) GATE-OFF BIT IDENTITY — the dark arm is the established CPU lane.
//! * (b) DEVICE ENCLOSURE — the seamed bound encloses the CPU bound AND 64
//!   sampled forward realizations (a real falsifier), on a fixture with
//!   mixed-sign weights, a dead-ReLU column and NONZERO fold errors. Run on a
//!   CHAIN fixture and, since the segment egress landed, on a RESIDUAL one.
//! * (c) FAIL-CLOSED PINS — one per refusal reason.
//! * (d) GUARD PINS — a too-small certified error trips the floor, and any
//!   refusal leaves the caller on the exact CPU path.
//! * (e) RESIDUAL MAPPING — the `Add` decomposes into the segment shape, the
//!   host-side `node_abs` retargeting invariant stays in backend fold order,
//!   and the residuals the device cannot represent still refuse.

use super::*;
use crate::margin_row::engine::{domain_gates, BackwardEngine, LaneDir};
use crate::margin_row::net::TwinNet;
use crate::margin_row::root::RootGates;
use crate::margin_row::rounding::RoundMode;
use crate::margin_row::spec::{TwinOpSpec, TwinSpec};

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

#[allow(clippy::cast_precision_loss)]
fn spread(i: usize, m: usize, scale: f64) -> f64 {
    (((i * 37) % m) as f64 / m as f64 - 0.5) * scale
}

/// Deterministic CHAIN fixture (no residual `Add`, so the unary coefficient
/// entry admits it) with MIXED-SIGN weights, a DEAD ReLU column (output
/// channel 3 of `conv1` carries a large negative bias, so its whole
/// channel-wide pre-activation box sits below zero and the root gates collapse
/// it to `alpha = s = c = 0`), and NONZERO `weight_rel_err` / `bias_err` on
/// both convs — the terms the device is now required to charge.
pub(crate) fn chain_spec() -> TwinSpec {
    let w1: Vec<f64> = (0..(4 * 2 * 3 * 3)).map(|i| spread(i, 71, 0.7)).collect();
    let w2: Vec<f64> = (0..(4 * 4 * 3 * 3)).map(|i| spread(i, 67, 0.5)).collect();
    let wh: Vec<f64> = (0..(6 * 64)).map(|i| spread(i, 59, 0.4)).collect();
    let wo: Vec<f64> = (0..(3 * 6)).map(|i| spread(i, 23, 1.2)).collect();
    TwinSpec {
        n_in: 2 * 4 * 4,
        ops: vec![
            TwinOpSpec::Conv {
                input: 0,
                weight: w1,
                // Channel 3 is driven hard negative -> DEAD ReLU column.
                bias: vec![0.05, -0.02, 0.01, -40.0],
                bias_err: vec![1e-13, 2e-13, 0.0, 5e-13],
                weight_rel_err: 1e-15,
                kernel: (4, 2, 3, 3),
                stride: (1, 1),
                pads: (1, 1, 1, 1),
                ishape: (2, 4, 4),
                oshape: (4, 4, 4),
            }, // t1
            TwinOpSpec::Relu { input: 1 }, // t2, trunk relu 0
            TwinOpSpec::Conv {
                input: 2,
                weight: w2,
                bias: vec![0.03, -0.04, 0.02, 0.0],
                bias_err: vec![3e-13, 0.0, 1e-13, 0.0],
                weight_rel_err: 1e-15,
                kernel: (4, 4, 3, 3),
                stride: (1, 1),
                pads: (1, 1, 1, 1),
                ishape: (4, 4, 4),
                oshape: (4, 4, 4),
            }, // t3
            TwinOpSpec::Relu { input: 3 }, // t4, trunk relu 1
            TwinOpSpec::Flatten { input: 4 }, // t5
            TwinOpSpec::Gemm {
                input: 5,
                weight: wh,
                bias: vec![0.1, -0.1, 0.05, -0.05, 0.0, 0.2],
                shape: (6, 64),
            }, // t6 = y
            TwinOpSpec::Relu { input: 6 }, // t7, head relu
            TwinOpSpec::Gemm {
                input: 7,
                weight: wo,
                bias: vec![0.0, 0.1, -0.1],
                shape: (3, 6),
            }, // t8
        ],
    }
}

/// The same net with an identity-skip residual `Add` — the shape the flat
/// coefficient egress could not express at all (it refused at the `Add`), and
/// that the SEGMENT egress now maps to `Chain / Residual / Chain`.
pub(crate) fn residual_spec() -> TwinSpec {
    let mut spec = chain_spec();
    // conv2 -> Add(conv2, relu0) -> relu1 -> ...
    spec.ops.insert(3, TwinOpSpec::Add { lhs: 3, rhs: 2 });
    for op in spec.ops.iter_mut().skip(4) {
        match op {
            TwinOpSpec::Relu { input } | TwinOpSpec::Flatten { input } => *input += 1,
            TwinOpSpec::Gemm { input, .. } => *input += 1,
            _ => {}
        }
    }
    spec
}

pub(crate) fn compile(spec: &TwinSpec) -> (TwinNet, RootGates, Vec<f64>, Vec<f64>) {
    let net = TwinNet::compile(spec).expect("fixture compiles");
    let lo = vec![-0.05; spec.n_in];
    let hi = vec![0.05; spec.n_in];
    let (root, _) = RootGates::build_retaining(&net, &lo, &hi, RoundMode::Outward, None, None, &[])
        .expect("root gates");
    (net, root, lo, hi)
}

/// Test shim for the CHAIN fixtures: the unary layer list + `rho*`.
///
/// Every fixture that uses it is a chain net, so a residual plan here is a test
/// bug (panic), while a genuine refusal must still come back as `Err` — that is
/// what the fail-closed pins assert.
fn build_layers(
    eng: &BackwardEngine<'_>,
    dom: Option<&DomainGates>,
) -> Result<(Vec<GpuCrownLayer>, f64), Refusal> {
    match build_plan(eng, dom) {
        Ok((Plan::Chain(layers), _, rho)) => Ok((layers, rho)),
        Ok((Plan::Segments(_), _, _)) => panic!("fixture must be a unary chain"),
        Err(e) => Err(e),
    }
}

/// The `GpuCrownLayer`s of a plan in the BACKEND'S FOLD ORDER: segments in
/// order, layers within a segment in order, `F` before `P`.
fn fold_order_layers(plan: &Plan) -> Vec<&GpuCrownLayer> {
    match plan {
        Plan::Chain(l) => l.iter().collect(),
        Plan::Segments(segs) => {
            let mut out = Vec::new();
            for s in segs {
                match s {
                    GpuResnetSegment::Chain(l) | GpuResnetSegment::Residual(l) => out.extend(l),
                    GpuResnetSegment::ResidualProj(f, p) => {
                        out.extend(f);
                        out.extend(p);
                    }
                }
            }
            out
        }
    }
}

fn assert_bit_identical(got: &[f64], want: &[f64], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length moved");
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "{what}: row {i} moved");
    }
}

// ---------------------------------------------------------------------------
// (a) Gate-off bit identity
// ---------------------------------------------------------------------------

#[test]
fn arming_is_exact_and_default_dark() {
    for rejected in [
        None,
        Some(""),
        Some("0"),
        Some("true"),
        Some(" 1 "),
        Some("2"),
    ] {
        assert!(!armed_from_raw(rejected), "raw {rejected:?} must stay dark");
    }
    assert!(armed_from_raw(Some("1")));
}

/// The seamed entry points must be the established CPU lane, bit-for-bit,
/// while the gate is off. This is the pin that makes the whole module free.
#[test]
fn gate_off_seamed_entries_are_bit_identical_to_cpu() {
    let (net, root, _, _) = compile(&chain_spec());
    let eng = BackwardEngine::new(&net, &root);
    let ctx = SeamCtx::default();

    let (al_ref, au_ref) = eng.y_rows(None).expect("cpu y-rows");
    let (al, au) = eng.y_rows_seamed(None, &ctx).expect("seamed y-rows");
    assert_bit_identical(
        &eng.concretize_lower(&al),
        &eng.concretize_lower(&al_ref),
        "y_rows lower",
    );
    assert_bit_identical(
        &eng.concretize_upper(&au),
        &eng.concretize_upper(&au_ref),
        "y_rows upper",
    );

    let seed = eng.identity_seed();
    let want = eng
        .run(&seed, None, LaneDir::Lower, None, false)
        .expect("cpu pass");
    let got = eng
        .run_seamed(&seed, None, LaneDir::Lower, &ctx)
        .expect("seamed pass");
    assert_bit_identical(
        &eng.concretize_lower(&got),
        &eng.concretize_lower(&want),
        "run_seamed lower",
    );

    // The production entry itself must refuse before it looks at anything.
    assert_eq!(
        run_pass(&eng, &seed, None, LaneDir::Lower, &ctx).err(),
        Some(Refusal::Disabled)
    );
}

// ---------------------------------------------------------------------------
// (c) Fail-closed pins, one per refusal reason
// ---------------------------------------------------------------------------

#[test]
fn parity_mode_is_never_seamed() {
    let spec = chain_spec();
    let net = TwinNet::compile(&spec).expect("compiles");
    let lo = vec![-0.05; spec.n_in];
    let hi = vec![0.05; spec.n_in];
    let (root, _) = RootGates::build_retaining(&net, &lo, &hi, RoundMode::Parity, None, None, &[])
        .expect("root gates");
    let eng = BackwardEngine::new(&net, &root);
    let seed = eng.identity_seed();
    assert_eq!(
        run_pass_armed(&eng, &seed, None, &SeamCtx::default(), Some(LaneDir::Lower)).err(),
        Some(Refusal::NotOutward)
    );
}

// ---------------------------------------------------------------------------
// (e) Residual mapping — the gap that made the armed seam read
//     `gpu_seam_ok=0 gpu_seam_refused=2` on every cifar100 row.
// ---------------------------------------------------------------------------

/// The identity-skip fixture must decompose into the segment shape the resnet
/// coefficient egress consumes — and must NOT be a chain plan any more.
#[test]
fn residual_add_maps_to_an_identity_skip_segment() {
    let (net, root, _, _) = compile(&residual_spec());
    let eng = BackwardEngine::new(&net, &root);
    let (plan, node_abs, rho_star) = build_plan(&eng, None).expect("residual fixture must map");
    let Plan::Segments(segs) = &plan else {
        panic!("a residual net must not produce a unary chain plan");
    };
    // chain(head, relu1) / identity-skip residual(conv2) / chain(relu0, conv1).
    assert_eq!(segs.len(), 3, "chain / residual / chain: {}", segs.len());
    assert!(matches!(segs[0], GpuResnetSegment::Chain(_)));
    let GpuResnetSegment::Residual(f_branch) = &segs[1] else {
        panic!("the identity skip must map to Residual, never ResidualProj");
    };
    assert_eq!(f_branch.len(), 1, "F is the single conv2");
    assert!(matches!(f_branch[0], GpuCrownLayer::Conv2d { .. }));
    assert!(matches!(segs[2], GpuResnetSegment::Chain(_)));
    // The head Gemm is still the FIRST backward layer.
    assert!(matches!(
        fold_order_layers(&plan)[0],
        GpuCrownLayer::Linear { .. }
    ));
    assert!(rho_star > 0.0, "rho* must survive the segment split");
    assert_eq!(node_abs.len(), 2, "one node_abs per trunk ReLU");
}

/// `node_abs` is a HOST-SIDE batch-retargeting invariant, not coefficient
/// egress input: the [`CertifiedCoeffs`] contract requires the backend to ignore
/// it at that trait seam. A drift could still pair `relus[k]` with the wrong
/// emitted `Activation` while retargeting a domain, so pin the alignment: entry
/// `k` must match the `k`-th Activation in backend fold order (segments in
/// order, F before P), width for width. The outward-magnitude assertions below
/// authenticate the root record used by that host comparison; they do not
/// grant permission to discharge coefficient error into bias.
#[test]
fn plan_node_abs_is_in_backend_fold_order() {
    for spec in [chain_spec(), residual_spec()] {
        let (net, root, _, _) = compile(&spec);
        let eng = BackwardEngine::new(&net, &root);
        let (plan, node_abs, _) = build_plan(&eng, None).expect("fixture maps");
        let acts: Vec<&GpuCrownLayer> = fold_order_layers(&plan)
            .into_iter()
            .filter(|l| matches!(l, GpuCrownLayer::Activation { .. }))
            .collect();
        assert_eq!(acts.len(), node_abs.len(), "one frontier per Activation");
        for (k, act) in acts.iter().enumerate() {
            let GpuCrownLayer::Activation { num_neurons, .. } = act else {
                unreachable!("filtered")
            };
            assert_eq!(node_abs[k].len(), *num_neurons, "fold index {k} width");
        }
        // And the frontier is a genuine OUTWARD magnitude bound on the lane's
        // own certified pre-activation box, never a tightened one. Fold order is
        // REVERSE execution order for both fixtures (the residual block's F
        // branch contributes no Activation), so `root.layers[k]` — execution
        // order — pairs with `node_abs[len - 1 - k]`.
        for (k, rec) in root.layers.iter().enumerate() {
            for j in 0..rec.n {
                let want = rec.l[j].abs().max(rec.u[j].abs());
                assert!(
                    f64::from(node_abs[node_abs.len() - 1 - k][j]) >= want,
                    "node_abs understates |z| at layer {k} neuron {j}"
                );
            }
        }
    }
}

/// #margin-row-gpu-batch: the `relus` list must track the EMITTED Activations,
/// one-for-one, in the same fold order as `node_abs`.
///
/// The batched lane re-gates the `k`-th Activation from `relus[k]`, so a drift
/// here would hand one ReLU's piece fixes to a different ReLU — silently, and
/// with no shape error to catch it. This asserts the pairing three ways: the
/// counts agree, each entry is a real root-gate layer whose width matches the
/// Activation it is paired with, and re-gating at the ROOT reproduces the
/// reference plan exactly (identity re-gate).
#[test]
fn plan_relu_layers_track_the_emitted_activations() {
    for spec in [chain_spec(), residual_spec()] {
        let (net, root, _, _) = compile(&spec);
        let eng = BackwardEngine::new(&net, &root);
        let (plan, node_abs, _, relus) = build_plan_full(&eng, None).expect("fixture maps");
        let acts: Vec<&GpuCrownLayer> = fold_order_layers(&plan)
            .into_iter()
            .filter(|l| matches!(l, GpuCrownLayer::Activation { .. }))
            .collect();
        assert_eq!(acts.len(), relus.len(), "one relu record per Activation");
        assert_eq!(node_abs.len(), relus.len(), "relus track node_abs");
        for (k, (act, li)) in acts.iter().zip(&relus).enumerate() {
            let GpuCrownLayer::Activation { num_neurons, .. } = act else {
                unreachable!("filtered")
            };
            let rec = root.layers.get(*li).expect("relu record in range");
            assert_eq!(rec.n, *num_neurons, "fold index {k} paired with layer {li}");
        }
        // Identity re-gate: with `dom = None` the re-gated plan must be the
        // reference plan itself. If `relus` were permuted this would produce a
        // different gate vector for at least one Activation on any fixture whose
        // ReLU layers differ — which both of these do (widths and boxes differ).
        let same =
            retarget_plan(&plan, &relus, &node_abs, &eng, None).expect("identity re-gate maps");
        let want = fold_order_layers(&plan);
        let got = fold_order_layers(&same);
        assert_eq!(want.len(), got.len());
        for (k, (w, g)) in want.iter().zip(&got).enumerate() {
            match (w, g) {
                (
                    GpuCrownLayer::Activation {
                        lower_slope: a1,
                        upper_slope: s1,
                        lower_intercept: li1,
                        upper_intercept: c1,
                        num_neurons: n1,
                    },
                    GpuCrownLayer::Activation {
                        lower_slope: a2,
                        upper_slope: s2,
                        lower_intercept: li2,
                        upper_intercept: c2,
                        num_neurons: n2,
                    },
                ) => assert_eq!(
                    (a1, s1, li1, c1, n1),
                    (a2, s2, li2, c2, n2),
                    "identity re-gate moved Activation {k}"
                ),
                (GpuCrownLayer::Activation { .. }, _) | (_, GpuCrownLayer::Activation { .. }) => {
                    panic!("identity re-gate changed the layer VARIANT at {k}")
                }
                _ => {}
            }
        }
    }
}

/// A residual the device's fork/merge cannot represent must still refuse:
/// `out = z + z` has no `GpuResnetSegment` form.
#[test]
fn unrepresentable_residuals_still_refuse() {
    // Degenerate `t4 = t3 + t3`: both branches are the SAME tensor, so the
    // common ancestor IS the block output and neither branch has any layers.
    // `Residual` would silently halve it, so the plan must refuse.
    let mut degenerate = chain_spec();
    degenerate.ops.insert(3, TwinOpSpec::Add { lhs: 3, rhs: 3 });
    for op in degenerate.ops.iter_mut().skip(4) {
        match op {
            TwinOpSpec::Relu { input } | TwinOpSpec::Flatten { input } => *input += 1,
            TwinOpSpec::Gemm { input, .. } => *input += 1,
            _ => {}
        }
    }
    let (net, root, _, _) = compile(&degenerate);
    let eng = BackwardEngine::new(&net, &root);
    assert_eq!(
        build_plan(&eng, None).err(),
        Some(Refusal::Unmappable("degenerate residual add"))
    );
}

/// A `ChannelAffine` INSIDE a residual branch is as unmappable as one on the
/// trunk — the segment egress does not widen the layer vocabulary.
#[test]
fn channel_affine_inside_a_residual_branch_refuses() {
    let mut spec = residual_spec();
    // Replace conv2 (op index 2, the F branch) with a diagonal affine.
    spec.ops[2] = TwinOpSpec::ChannelAffine {
        input: 2,
        scale: vec![1.1, 0.9, 1.0, 1.0],
        shift: vec![0.0; 4],
        scale_rel_err: 1e-15,
        shift_err: vec![0.0; 4],
        shape: (4, 4, 4),
    };
    let (net, root, _, _) = compile(&spec);
    let eng = BackwardEngine::new(&net, &root);
    assert_eq!(
        build_plan(&eng, None).err(),
        Some(Refusal::Unmappable("ChannelAffine in a residual branch"))
    );
}

#[test]
fn channel_affine_refuses_the_whole_plan() {
    let mut spec = chain_spec();
    spec.ops[2] = TwinOpSpec::ChannelAffine {
        input: 2,
        scale: vec![1.1, 0.9, 1.0, 1.0],
        shift: vec![0.0; 4],
        scale_rel_err: 1e-15,
        shift_err: vec![0.0; 4],
        shape: (4, 4, 4),
    };
    let (net, root, _, _) = compile(&spec);
    let eng = BackwardEngine::new(&net, &root);
    assert_eq!(
        build_layers(&eng, None).err(),
        Some(Refusal::Unmappable("ChannelAffine has no GpuCrownLayer"))
    );
}

#[test]
fn conv_transpose_and_asymmetric_padding_refuse() {
    let (net, _root, _, _) = compile(&chain_spec());
    let TwinOp::Conv(cv) = &net.ops[0] else {
        panic!("op 0 is the first conv");
    };
    let mut transposed = (**cv).clone();
    transposed.transposed = true;
    assert_eq!(
        conv(&transposed).err(),
        Some(Refusal::Unmappable("ConvTranspose has no GpuCrownLayer"))
    );
    let mut lopsided = (**cv).clone();
    lopsided.pads = (1, 1, 0, 1);
    assert_eq!(
        conv(&lopsided).err(),
        Some(Refusal::Unmappable("asymmetric conv padding"))
    );
}

/// A seed that carries a certified error (every real margin-row seed does)
/// cannot be handed to a device that treats seeds as exact unless the caller
/// supplies the y-box magnitudes to concretize the discrepancy into.
#[test]
fn seed_error_without_y_abs_refuses() {
    let mut seed = Seed {
        s: Array2::<f64>::zeros((3, 2)),
        e: Some(Array2::<f64>::from_elem((3, 2), 1e-14)),
    };
    seed.s[[0, 0]] = 0.5;
    assert_eq!(
        seed_penalty(&seed, None, 3, 2).err(),
        Some(Refusal::SeedNeedsYAbs)
    );
    let y_abs = vec![2.0, 3.0, 4.0];
    let pen = seed_penalty(&seed, Some(&y_abs), 3, 2).expect("penalty");
    assert_eq!(pen.len(), 2);
    // 1e-14 against |y| <= {2,3,4} sums to >= 9e-14 per row.
    assert!(pen.iter().all(|p| *p >= 9e-14), "{pen:?}");
}

/// An exact seed (identity) needs no y-box and produces a negligible penalty.
#[test]
fn exact_seed_needs_no_y_abs() {
    let mut s = Array2::<f64>::zeros((3, 3));
    for j in 0..3 {
        s[[j, j]] = 1.0;
    }
    let seed = Seed { s, e: None };
    let pen = seed_penalty(&seed, None, 3, 3).expect("exact seed maps");
    assert!(pen.iter().all(|p| *p < 1e-300), "{pen:?}");
}

/// A CPU-only process (no registered sound GPU CROWN factory) must refuse for
/// lack of authority, and `run_seamed` must then be the CPU pass bit-for-bit.
#[test]
fn armed_without_authority_refuses_and_run_seamed_is_the_cpu_pass() {
    let (net, root, _, _) = compile(&chain_spec());
    let eng = BackwardEngine::new(&net, &root);
    let seed = eng.identity_seed();
    let ctx = SeamCtx::default();
    match run_pass_armed(&eng, &seed, None, &ctx, Some(LaneDir::Lower)) {
        Err(refusal) => assert!(
            matches!(refusal, Refusal::NoAuthority | Refusal::NoCoeffEgress),
            "a CPU-only process must refuse for authority, got {refusal:?}"
        ),
        Ok(_) => panic!("a CPU-only process must not reach a device"),
    }
    // Whatever the refusal, the wrapper is the established CPU call.
    let want = eng
        .run(&seed, None, LaneDir::Lower, None, false)
        .expect("cpu pass");
    let got = eng
        .run_seamed(&seed, None, LaneDir::Lower, &ctx)
        .expect("fallback");
    assert_bit_identical(
        &eng.concretize_lower(&got),
        &eng.concretize_lower(&want),
        "fallback",
    );
}

// ---------------------------------------------------------------------------
// Mapping pins
// ---------------------------------------------------------------------------

/// The composed weight error must dominate BOTH the BN fold's relative ball
/// and the f64->f32 downcast using the denominator required by
/// `CertifiedWeightError` (`|w32|`, not `|w_exact|`). The bias term must
/// dominate the fold's absolute error plus the bias downcast.
#[test]
fn certified_weight_error_dominates_fold_and_downcast() {
    let u32_unit = f64::from(f32::EPSILON) / 2.0;
    let rho = 1e-12;
    let err = weight_error(rho, &[2.0, -3.0], &[1e-13, 4e-13]).expect("composes");
    let rel = f64::from(err.weight_rel_err);
    let exact_weight_relative = rho + u32_unit + rho * u32_unit;
    let required_supplied_relative = exact_weight_relative / (1.0 - exact_weight_relative);
    assert!(
        rel >= required_supplied_relative,
        "stored-weight-relative radius {rel:e} under-charges required \
         {required_supplied_relative:e}"
    );
    assert!(
        rel > exact_weight_relative,
        "the denominator conversion must be load-bearing for nonzero R"
    );
    let abs = f64::from(err.bias_abs_err);
    assert!(
        abs >= 4e-13 + u32_unit * 3.0,
        "bias abs {abs} under-charges"
    );
    // Zero fold error still charges the downcast.
    let plain = weight_error(0.0, &[0.0], &[0.0]).expect("composes");
    assert!(
        f64::from(plain.weight_rel_err) >= u32_unit / (1.0 - u32_unit),
        "even the downcast-only ball needs the supplied-weight denominator"
    );
}

/// Once the exact-weight-relative ball reaches one, `|w32|` has no positive
/// lower bound in terms of `|w_exact|`; no finite radius can satisfy the public
/// supplied-weight-relative contract. Refuse instead of publishing a number
/// with no proof.
#[test]
fn certified_weight_error_refuses_a_noninvertible_denominator() {
    assert!(matches!(
        weight_error(1.0, &[0.0], &[0.0]),
        Err(Refusal::Unmappable(
            "weight error ball reaches unity before denominator conversion"
        ))
    ));
}

/// The weight downcast is VERIFIED against the relative ball the device is
/// told to charge: a value whose binary32 image moves further than that (the
/// DAZ/FTZ flush-to-zero case) refuses the plan rather than shipping an
/// under-charged weight.
#[test]
fn weight_downcast_outside_the_charged_ball_refuses() {
    // Ordinary magnitudes round to nearest well inside 2^-24 relative.
    let ok = to_f32_params_rel(&[1.0, -0.3333333333333333, 1e-8, 0.0], U32_UNIT);
    assert!(ok.is_ok(), "{ok:?}");
    // A binary32-subnormal magnitude: exactly the range a flushing conversion
    // would destroy. Either it downcasts within the ball or the plan refuses;
    // it must never pass through silently mis-charged.
    let subnormal = 1e-44_f64;
    match to_f32_params_rel(&[subnormal], U32_UNIT) {
        Ok(v) => assert!(
            (ny_core::f32_to_f64_exact(v[0]) - subnormal).abs() <= U32_UNIT * subnormal,
            "an accepted downcast must sit inside the charged ball"
        ),
        Err(Refusal::Unmappable(_)) => {}
        Err(other) => panic!("unexpected refusal {other:?}"),
    }
}

/// The f32 downcast of a ReLU upper line must never dip below the true ReLU on
/// `[l, u]`. Convexity makes the endpoint test sufficient, so this checks the
/// endpoints AND a dense interior sweep.
#[test]
fn downcast_relu_upper_line_still_encloses_relu() {
    #[allow(clippy::cast_precision_loss)]
    for k in 1..200usize {
        let l = -0.01 - (k as f64) * 0.037;
        let u = 0.013 + (k as f64) * 0.011;
        // The exact DeepPoly chord.
        let s = u / (u - l);
        let c = -l * u / (u - l);
        let (a32, s32, c32) = outward_gate_f32(&[0.0], &[s], &[c], &[l], &[u]).expect("gate maps");
        assert_eq!(a32[0], 0.0);
        let (sd, cd) = (f64::from(s32[0]), f64::from(c32[0]));
        for t in 0..=64 {
            #[allow(clippy::cast_precision_loss)]
            let x = l + (u - l) * (t as f64) / 64.0;
            assert!(
                sd * x + cd >= x.max(0.0) - 1e-18,
                "k={k} x={x}: downcast upper line {} dips below relu {}",
                sd * x + cd,
                x.max(0.0)
            );
        }
    }
}

/// Stable and piece-fixed gates are `(1,1,0)` / `(0,0,0)`, which are f32-exact,
/// so the repair must leave them ALONE — a piece-fixed neuron must never be
/// widened against the root box it no longer lives in.
#[test]
fn exact_gates_are_never_repaired() {
    let (a, s, c) = outward_gate_f32(
        &[1.0, 0.0],
        &[1.0, 0.0],
        &[0.0, 0.0],
        &[-3.0, -3.0],
        &[2.0, 2.0],
    )
    .expect("gate maps");
    assert_eq!(a, vec![1.0, 0.0]);
    assert_eq!(s, vec![1.0, 0.0]);
    assert_eq!(c, vec![0.0, 0.0], "an exact gate was repaired");
}

/// Domain gate overrides (piece-fixed trunk splits) must reach the device plan,
/// or the seam would describe a DIFFERENT relaxation than the pass it replaces.
#[test]
fn domain_gate_overrides_reach_the_plan() {
    let (net, root, _, _) = compile(&chain_spec());
    let eng = BackwardEngine::new(&net, &root);
    let (li, pos) = root
        .layers
        .iter()
        .enumerate()
        .find_map(|(li, rec)| (!rec.unst.is_empty()).then_some((li, 0usize)))
        .expect("fixture has an unstable trunk neuron");
    let idx = root.layers[li].unst[pos];
    let dom = domain_gates(&root, &[(li, pos, 1)]);
    let base = gate_tuples(&build_layers(&eng, None).expect("root plan").0, idx);
    let over = gate_tuples(&build_layers(&eng, Some(&dom)).expect("domain plan").0, idx);
    assert_ne!(base, over, "the piece-fixed gate never reached the plan");
    assert!(
        over.contains(&(1.0, 1.0, 0.0)),
        "the piece-fixed pass-through gate is missing: {over:?}"
    );
    // And the fix is discoverable for the probe's membership test.
    let fixes = domain_fixes(&eng, Some(&dom));
    assert_eq!(fixes.len(), 1);
    assert_eq!(fixes[0].1, vec![(idx, true)]);
}

fn gate_tuples(layers: &[GpuCrownLayer], idx: usize) -> Vec<(f32, f32, f32)> {
    layers
        .iter()
        .filter_map(|l| match l {
            GpuCrownLayer::Activation {
                lower_slope,
                upper_slope,
                upper_intercept,
                num_neurons,
                ..
            } if idx < *num_neurons => {
                Some((lower_slope[idx], upper_slope[idx], upper_intercept[idx]))
            }
            _ => None,
        })
        .collect()
}

/// The dead ReLU column must survive into the plan as a genuine zero gate, and
/// the head Gemm must be the FIRST backward layer.
#[test]
fn plan_shape_and_dead_column() {
    let (net, root, _, _) = compile(&chain_spec());
    let eng = BackwardEngine::new(&net, &root);
    let (layers, rho_star) = build_layers(&eng, None).expect("chain plan");
    assert!(
        matches!(layers[0], GpuCrownLayer::Linear { .. }),
        "head first"
    );
    assert!(
        rho_star >= f64::from(f32::EPSILON) / 2.0,
        "rho* under-charges"
    );
    let acts: Vec<&GpuCrownLayer> = layers
        .iter()
        .filter(|l| matches!(l, GpuCrownLayer::Activation { .. }))
        .collect();
    assert_eq!(acts.len(), 2, "two trunk relus");
    let GpuCrownLayer::Activation {
        lower_slope,
        upper_slope,
        lower_intercept,
        upper_intercept,
        ..
    } = acts[1]
    else {
        unreachable!("filtered")
    };
    assert!(
        lower_intercept.iter().all(|v| *v == 0.0),
        "relu lower line is through the origin"
    );
    for i in 48..64 {
        assert_eq!(lower_slope[i], 0.0, "dead column slope {i}");
        assert_eq!(upper_slope[i], 0.0, "dead column slope {i}");
        assert_eq!(upper_intercept[i], 0.0, "dead column intercept {i}");
    }
    assert!(
        (0..48).any(|i| upper_slope[i] != 0.0),
        "fixture has no unstable neuron"
    );
}

#[test]
fn input_box_is_rounded_outward_into_f32() {
    let lo = vec![-0.1_f64, 0.0, 0.5];
    let hi = vec![0.1_f64, 0.3, 1.0];
    let (l32, h32) = build_box(&lo, &hi, 3).expect("box maps");
    for i in 0..3 {
        assert!(
            f64::from(l32[i]) <= lo[i],
            "lower box rounded inward at {i}"
        );
        assert!(
            f64::from(h32[i]) >= hi[i],
            "upper box rounded inward at {i}"
        );
    }
}

#[test]
fn seed_transpose_matches_the_lane_layout() {
    let (net, root, _, _) = compile(&chain_spec());
    let eng = BackwardEngine::new(&net, &root);
    let seed = eng.identity_seed();
    let gseed = build_seed(net.n_y, &seed, net.n_y).expect("identity seed maps");
    assert_eq!(gseed.num_specs, net.n_y);
    assert_eq!(gseed.current_dim, net.n_y);
    for r in 0..net.n_y {
        for j in 0..net.n_y {
            let want = if r == j { 1.0 } else { 0.0 };
            assert_eq!(gseed.lower_a[r * net.n_y + j], want);
            assert_eq!(gseed.upper_a[r * net.n_y + j], want);
        }
    }
    assert!(gseed.lower_b.iter().all(|v| *v == 0.0));
}

// ---------------------------------------------------------------------------
// (d) Guard pins
// ---------------------------------------------------------------------------

/// Build a synthetic device payload: `dim x rows` coefficients with a uniform
/// relative certified error.
fn payload(rows: usize, dim: usize, coeff: f32, rel: f32) -> CertifiedCoeffs {
    let a = vec![coeff; rows * dim];
    let err = vec![coeff.abs() * rel; rows * dim];
    CertifiedCoeffs {
        lower_a: a.clone(),
        upper_a: a,
        lower_a_err: err.clone(),
        upper_a_err: err,
        lower_b: vec![0.0; rows],
        upper_b: vec![0.0; rows],
        lower_b_err: vec![0.0; rows],
        upper_b_err: vec![0.0; rows],
        num_specs: rows,
        dim,
    }
}

/// THE GUARD PIN. A device that returns an error far below the weight-error
/// ball it was told to charge is rejected; the same payload with an honest
/// error passes. Fail-closed means the caller then runs the CPU path.
#[test]
fn error_floor_rejects_an_undercharged_payload() {
    let (rows, dim) = (4usize, 16usize);
    let xabs = [1.0f64; 16];
    let zero_pen = [0.0f64; 4];
    let rho = 1e-6f64;

    let honest = payload(rows, dim, 0.25, 2e-6);
    let out =
        convert_and_check(&honest, LaneDir::Lower, dim, rows, &zero_pen).expect("payload converts");
    assert!(
        error_floor_ok(&out, &xabs, rho),
        "an honestly-charged payload must pass the floor"
    );

    let starved = payload(rows, dim, 0.25, 1e-9);
    let out = convert_and_check(&starved, LaneDir::Lower, dim, rows, &zero_pen)
        .expect("payload converts");
    assert!(
        !error_floor_ok(&out, &xabs, rho),
        "an under-charged payload must TRIP the floor"
    );

    // A zeroed error lane — the classic "bounds returned as coefficients"
    // failure — trips too.
    let zeroed = payload(rows, dim, 0.25, 0.0);
    let out =
        convert_and_check(&zeroed, LaneDir::Lower, dim, rows, &zero_pen).expect("payload converts");
    assert!(
        !error_floor_ok(&out, &xabs, rho),
        "a zero error lane must TRIP"
    );
}

/// The floor must never trip on an all-zero coefficient row (nothing to
/// charge), or the seam would fail closed on trivially-decided rows forever.
#[test]
fn error_floor_admits_a_zero_row() {
    let zeroed = payload(2, 8, 0.0, 0.0);
    let out = convert_and_check(&zeroed, LaneDir::Lower, 8, 2, &[0.0, 0.0]).expect("converts");
    assert!(error_floor_ok(&out, &[1.0f64; 8], 1e-6));
}

/// Structural rejection: wrong dims, non-finite entries, and a NEGATIVE error
/// lane are all payload refusals, never repairs.
#[test]
fn malformed_payloads_are_refused() {
    let good = payload(2, 4, 0.5, 1e-5);
    assert_eq!(
        convert_and_check(&good, LaneDir::Lower, 5, 2, &[0.0, 0.0]).err(),
        Some(Refusal::Payload),
        "dim mismatch"
    );
    assert_eq!(
        convert_and_check(&good, LaneDir::Lower, 4, 3, &[0.0, 0.0, 0.0]).err(),
        Some(Refusal::Payload),
        "row mismatch"
    );
    let mut nan = payload(2, 4, 0.5, 1e-5);
    nan.lower_a[3] = f32::NAN;
    assert_eq!(
        convert_and_check(&nan, LaneDir::Lower, 4, 2, &[0.0, 0.0]).err(),
        Some(Refusal::Payload)
    );
    let mut negative = payload(2, 4, 0.5, 1e-5);
    negative.lower_a_err[1] = -1e-6;
    assert_eq!(
        convert_and_check(&negative, LaneDir::Lower, 4, 2, &[0.0, 0.0]).err(),
        Some(Refusal::Payload)
    );
}

/// The seed penalty must land in `eb`, not be silently dropped: two payloads
/// that differ only in the seed penalty must differ by exactly that in `eb`.
#[test]
fn seed_penalty_lands_in_the_bias_error_lane() {
    let cc = payload(2, 4, 0.5, 1e-5);
    let bare = convert_and_check(&cc, LaneDir::Lower, 4, 2, &[0.0, 0.0]).expect("converts");
    let with = convert_and_check(&cc, LaneDir::Lower, 4, 2, &[1e-9, 2e-9]).expect("converts");
    assert!(with.eb[0] >= bare.eb[0] + 1e-9 * 0.99);
    assert!(with.eb[1] >= bare.eb[1] + 2e-9 * 0.99);
}

/// `f32 -> f64` is exact, so conversion must not perturb a coefficient.
#[test]
fn coefficient_conversion_is_exact_and_transposed() {
    let mut cc = payload(2, 3, 0.0, 0.0);
    // row-major [num_specs x dim]
    cc.lower_a = vec![1.5, -2.25, 0.125, 8.0, -0.5, 3.0];
    cc.lower_a_err = vec![0.0; 6];
    let out = convert_and_check(&cc, LaneDir::Lower, 3, 2, &[0.0, 0.0]).expect("converts");
    // The lane stores (dim, rows).
    assert_eq!(out.a[[0, 0]], 1.5);
    assert_eq!(out.a[[1, 0]], -2.25);
    assert_eq!(out.a[[2, 0]], 0.125);
    assert_eq!(out.a[[0, 1]], 8.0);
    assert_eq!(out.a[[1, 1]], -0.5);
    assert_eq!(out.a[[2, 1]], 3.0);
}

// ---------------------------------------------------------------------------
// (b) Device enclosure oracle
// ---------------------------------------------------------------------------

/// REAL DEVICE. On the chain fixture the seamed bound must (i) enclose the CPU
/// lane's bound and (ii) stay below every sampled realization of the seeded
/// functional — the genuine soundness falsifier, not a widening tautology.
///
/// Runs both the root gates and a piece-fixed domain; only the root sweep is a
/// valid global falsifier, because a piece-fixed domain restricts the region.
#[cfg(feature = "gpu-tests")]
#[test]
fn gpu_seamed_pass_encloses_cpu_and_sampled_realizations_on_device() {
    use ndarray::Array2;

    let device = ny_gpu::WgpuDevice::new_for_verdict(ny_gpu::WgpuVerdictRequest::new())
        .expect("gpu-tests requires a WGPU device passing all five authority rungs");
    crate::sound_gpu_gate::set_sound_gpu_crown_required(true);
    let shared: Arc<dyn ny_core::GemmEngine> = Arc::new(device);
    crate::sound_gpu_gate::set_sound_gpu_crown_factory(move || Some(shared.clone()));
    assert!(
        crate::sound_gpu_gate::prewarm_sound_gpu_crown(),
        "the explicitly requested device must advertise sound GPU CROWN"
    );

    let (net, root, lo, hi) = compile(&chain_spec());
    let eng = BackwardEngine::new(&net, &root);
    let split = root
        .layers
        .iter()
        .enumerate()
        .find_map(|(li, rec)| (!rec.unst.is_empty()).then_some((li, 0usize, 1i8)));

    for dom in [None, split.map(|s| domain_gates(&root, &[s]))] {
        let dom_ref = dom.as_ref();
        let seed = eng.identity_seed();
        let cpu = eng.concretize_lower(
            &eng.run(&seed, dom_ref, LaneDir::Lower, None, false)
                .expect("cpu pass"),
        );
        let gpu_pass = run_pass_armed(
            &eng,
            &seed,
            dom_ref,
            &SeamCtx::default(),
            Some(LaneDir::Lower),
        )
        .expect("the armed seam must reach the prewarmed sound device")
        .0
        .expect("lower lane");
        let published = eng.concretize_lower(&gpu_pass);

        // (i) the seamed bound is a valid enclosure alongside the CPU one: it
        // must not exceed the CPU bound (f32 relaxation is never tighter here
        // by construction of the outward gate repair + charged fold error).
        for (i, (p, c)) in published.iter().zip(&cpu).enumerate() {
            assert!(p <= c, "row {i}: seamed {p} exceeds the CPU authority {c}");
        }

        // (ii) the real falsifier.
        if dom_ref.is_none() {
            let samples = 64usize;
            let mut x = Array2::<f64>::zeros((net.n_in, samples));
            for s in 0..samples {
                for i in 0..net.n_in {
                    #[allow(clippy::cast_precision_loss)]
                    let t = ((i * 7 + s * 13) % 5) as f64 / 4.0;
                    x[[i, s]] = lo[i] + t * (hi[i] - lo[i]);
                }
            }
            let (y, _) = net.forward_points(&x, &BTreeMap::new()).expect("forward");
            for j in 0..net.n_y {
                for s in 0..samples {
                    assert!(
                        published[j] <= y[[j, s]] + 1e-9,
                        "row {j} sample {s}: published lower bound {} exceeds a realized \
                         value {}",
                        published[j],
                        y[[j, s]]
                    );
                }
            }
        }
    }
    assert_eq!(
        crate::margin_row::prof::counter(Counter::GpuSeamGuardTrip),
        0,
        "a healthy device must never trip a soundness guard"
    );
}

/// REAL DEVICE, RESIDUAL NET — the fixture the seam used to refuse outright.
///
/// This is the test that would have caught the measured `gpu_seam_ok=0
/// gpu_seam_refused=2`: it asserts the ok counter actually MOVES on a net with a
/// residual `Add`, and then re-runs the same two falsifiers as the chain test:
/// the published bound must not exceed the CPU lane's bound, and it must not
/// exceed any of 64 sampled forward realizations of the seeded functional.
#[cfg(feature = "gpu-tests")]
#[test]
fn gpu_seamed_residual_pass_runs_and_encloses_cpu_and_realizations_on_device() {
    use ndarray::Array2;

    let device = ny_gpu::WgpuDevice::new_for_verdict(ny_gpu::WgpuVerdictRequest::new())
        .expect("gpu-tests requires a WGPU device passing all five authority rungs");
    crate::sound_gpu_gate::set_sound_gpu_crown_required(true);
    let shared: Arc<dyn ny_core::GemmEngine> = Arc::new(device);
    crate::sound_gpu_gate::set_sound_gpu_crown_factory(move || Some(shared.clone()));
    assert!(
        crate::sound_gpu_gate::prewarm_sound_gpu_crown(),
        "the explicitly requested device must advertise sound GPU CROWN"
    );

    let (net, root, lo, hi) = compile(&residual_spec());
    let eng = BackwardEngine::new(&net, &root);
    // The fixture must genuinely exercise the SEGMENT egress, not silently fall
    // back to the flat chain one.
    assert!(
        matches!(build_plan(&eng, None), Ok((Plan::Segments(_), _, _))),
        "the residual fixture must dispatch through the segment egress"
    );

    // The counters production reads out of NY_MARGIN_ROW_PROFILE.
    crate::margin_row::prof::force_active_for_test(true);
    let ok_before = crate::margin_row::prof::counter(Counter::GpuSeamOk);
    let trips_before = crate::margin_row::prof::counter(Counter::GpuSeamGuardTrip);

    let seed = eng.identity_seed();
    let cpu = eng.concretize_lower(
        &eng.run(&seed, None, LaneDir::Lower, None, false)
            .expect("cpu pass"),
    );
    let gpu_pass =
        run_pass_armed_recorded(&eng, &seed, None, &SeamCtx::default(), Some(LaneDir::Lower))
            .expect("the armed seam must now REACH the device on a residual net")
            .0
            .expect("lower lane");
    let published = eng.concretize_lower(&gpu_pass);

    // (0) THE MEASUREMENT THAT WAS ZERO.
    assert_eq!(
        crate::margin_row::prof::counter(Counter::GpuSeamOk),
        ok_before + 1,
        "gpu_seam_ok must increment on a residual net — this is the counter that \
         read 0 with `gpu_seam_refused=2` before the segment egress existed"
    );

    // (i) the seamed bound is a valid enclosure alongside the CPU one.
    for (i, (p, c)) in published.iter().zip(&cpu).enumerate() {
        assert!(p <= c, "row {i}: seamed {p} exceeds the CPU authority {c}");
    }

    // (ii) the real falsifier: 64 sampled realizations of the same functional.
    let samples = 64usize;
    let mut x = Array2::<f64>::zeros((net.n_in, samples));
    for s in 0..samples {
        for i in 0..net.n_in {
            #[allow(clippy::cast_precision_loss)]
            let t = ((i * 7 + s * 13) % 5) as f64 / 4.0;
            x[[i, s]] = lo[i] + t * (hi[i] - lo[i]);
        }
    }
    let (y, _) = net.forward_points(&x, &BTreeMap::new()).expect("forward");
    for j in 0..net.n_y {
        for s in 0..samples {
            assert!(
                published[j] <= y[[j, s]] + 1e-9,
                "row {j} sample {s}: published lower bound {} exceeds a realized value {}",
                published[j],
                y[[j, s]]
            );
        }
    }

    assert_eq!(
        crate::margin_row::prof::counter(Counter::GpuSeamGuardTrip),
        trips_before,
        "a healthy device must never trip a soundness guard on the residual path"
    );
    crate::margin_row::prof::force_active_for_test(false);
}
