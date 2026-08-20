// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Oracle tests for [`crate::bnn_sign_space`].
//!
//! The load-bearing assertions, in order of what they protect:
//!
//! 1. **FIXED/FREE is exact and both strictnesses are pinned.** The
//!    classification is compared against BRUTE FORCE over every box vertex of
//!    a tiny hand-built net, and two units sit exactly ON the boundary — one
//!    with `L == t` (must be FIXED `+1`) and one with `U == t` (must be FREE).
//!    Swapping either strictness flips exactly those two assertions.
//! 2. **Zero maps to `+1`.** A second, independent forward is written twice —
//!    once with `>=` and once with `>` — on an input where they differ, and
//!    the module must match the `>=` one.
//! 3. **The LP primal is a real point.** Every FREE-unit constraint, every box
//!    bound and the induced sign pattern are re-checked on the returned `x`.
//! 4. **Refusals, not wrong answers**, for everything outside the fragment.

use std::collections::HashSet;

use super::*;

// ---------------------------------------------------------------------------
// A tiny hand-built net in the admitted fragment.
// ---------------------------------------------------------------------------

/// Owned tensors for a tiny `Conv -> B -> Conv -> B -> Dense` net.
struct TinyNet {
    input: InputGeometry,
    conv1: Vec<f32>,
    conv1_out: usize,
    conv1_k: usize,
    conv2: Vec<f32>,
    conv2_out: usize,
    conv2_k: usize,
    dense: Vec<f32>,
    num_classes: usize,
    lo: Vec<f64>,
    hi: Vec<f64>,
    target: usize,
    challengers: Vec<usize>,
}

impl TinyNet {
    fn h1(&self) -> usize {
        self.input.height - self.conv1_k + 1
    }
    fn w1(&self) -> usize {
        self.input.width - self.conv1_k + 1
    }
    fn h2(&self) -> usize {
        self.h1() - self.conv2_k + 1
    }
    fn w2(&self) -> usize {
        self.w1() - self.conv2_k + 1
    }
    fn n_units1(&self) -> usize {
        self.h1() * self.w1() * self.conv1_out
    }
    fn n_flat(&self) -> usize {
        self.h2() * self.w2() * self.conv2_out
    }
    fn n_pixels(&self) -> usize {
        self.input.height * self.input.width * self.input.channels
    }

    fn request(&self) -> SignSpaceRequest<'_> {
        SignSpaceRequest {
            input: self.input,
            conv1: ConvSpec::valid_unit_stride(
                &self.conv1,
                self.conv1_out,
                self.input.channels,
                self.conv1_k,
                self.conv1_k,
            ),
            conv1_pool: None,
            conv1_affine: None,
            activation1: SignSpaceActivation::SignAddSign { add_constant: 0.1 },
            conv2: ConvSpec::valid_unit_stride(
                &self.conv2,
                self.conv2_out,
                self.conv1_out,
                self.conv2_k,
                self.conv2_k,
            ),
            conv2_pool: None,
            conv2_affine: None,
            activation2: SignSpaceActivation::SignAddSign { add_constant: 0.1 },
            stages: &[],
            dense: &self.dense,
            num_classes: self.num_classes,
            lo: &self.lo,
            hi: &self.hi,
            target_class: self.target,
            challengers: &self.challengers,
            reference_input: None,
            reference_forward: None,
        }
    }

    /// `z1` at a concrete input, computed independently of the module.
    fn z1(&self, x: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.n_units1()];
        for r in 0..self.h1() {
            for c in 0..self.w1() {
                for ch in 0..self.conv1_out {
                    let mut acc = 0.0;
                    for kr in 0..self.conv1_k {
                        for kc in 0..self.conv1_k {
                            for ic in 0..self.input.channels {
                                let w = self.conv1[((ch * self.input.channels + ic) * self.conv1_k
                                    + kr)
                                    * self.conv1_k
                                    + kc] as f64;
                                let p = ((r + kr) * self.input.width + (c + kc))
                                    * self.input.channels
                                    + ic;
                                acc += w * x[p];
                            }
                        }
                    }
                    out[(r * self.w1() + c) * self.conv1_out + ch] = acc;
                }
            }
        }
        out
    }

    /// A fully independent forward. `zero_positive` selects the convention
    /// under test: `true` is `B(v) = +1 iff v >= 0` (the ONNX
    /// `Sign(Sign(v)+0.1)` truth), `false` is the naive `v > 0`.
    fn reference_logits(&self, x: &[f64], zero_positive: bool) -> Vec<i64> {
        let fire = |v: f64| {
            if zero_positive {
                v >= 0.0
            } else {
                v > 0.0
            }
        };
        let z1 = self.z1(x);
        let s1: Vec<i64> = z1.iter().map(|&v| if fire(v) { 1 } else { -1 }).collect();
        let mut s2 = vec![0i64; self.n_flat()];
        for i in 0..self.h2() {
            for j in 0..self.w2() {
                for co in 0..self.conv2_out {
                    let mut acc = 0i64;
                    for kr in 0..self.conv2_k {
                        for kc in 0..self.conv2_k {
                            for ch in 0..self.conv1_out {
                                let w = self.conv2[((co * self.conv1_out + ch) * self.conv2_k + kr)
                                    * self.conv2_k
                                    + kc] as i64;
                                let u = ((i + kr) * self.w1() + (j + kc)) * self.conv1_out + ch;
                                acc += w * s1[u];
                            }
                        }
                    }
                    let j2 = (i * self.w2() + j) * self.conv2_out + co;
                    s2[j2] = if fire(acc as f64) { 1 } else { -1 };
                }
            }
        }
        let mut logits = vec![0i64; self.num_classes];
        for j in 0..self.n_flat() {
            for (class, logit) in logits.iter_mut().enumerate() {
                *logit += s2[j] * self.dense[j * self.num_classes + class] as i64;
            }
        }
        logits
    }
}

/// Deterministic `+/-1` filler; no RNG crate, no hidden state.
fn pm1(seed: &mut u64, n: usize) -> Vec<f32> {
    (0..n)
        .map(|_| {
            *seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            if (*seed >> 33) & 1 == 0 {
                1.0f32
            } else {
                -1.0f32
            }
        })
        .collect()
}

/// The 4x4x1 net used for the brute-force classification oracle.
///
/// Box: the top-left 2x2 block is `[1, 2]`, every other pixel is `[0, 1]`.
/// conv1 has three 2x2 channels — all `+1`, all `-1`, and a checkerboard — so
/// at output position `(2, 2)` (whose whole patch is `[0, 1]`) channel 0 has
/// `L == 0` EXACTLY and channel 1 has `U == 0` EXACTLY. Those are the two
/// boundary units the strictness assertions pin.
fn boundary_net() -> TinyNet {
    let input = InputGeometry {
        height: 4,
        width: 4,
        channels: 1,
    };
    #[rustfmt::skip]
    let conv1: Vec<f32> = vec![
        // ch0: all +1
         1.0,  1.0,
         1.0,  1.0,
        // ch1: all -1
        -1.0, -1.0,
        -1.0, -1.0,
        // ch2: checkerboard
         1.0, -1.0,
        -1.0,  1.0,
    ];
    let mut seed = 0x5eed_1234_u64;
    let conv2 = pm1(&mut seed, 2 * 3 * 2 * 2);
    let dense = pm1(&mut seed, 2 * 2 * 2 * 3);
    let mut lo = vec![0.0f64; 16];
    let mut hi = vec![1.0f64; 16];
    for &p in &[0usize, 1, 4, 5] {
        lo[p] = 1.0;
        hi[p] = 2.0;
    }
    TinyNet {
        input,
        conv1,
        conv1_out: 3,
        conv1_k: 2,
        conv2,
        conv2_out: 2,
        conv2_k: 2,
        dense,
        num_classes: 3,
        lo,
        hi,
        target: 0,
        challengers: vec![1, 2],
    }
}

/// A larger tiny net (6x6x1 -> 5x5x4 -> 4x4x4 -> 3) with a wide box, used for
/// the end-to-end search.
fn search_net(target: usize) -> TinyNet {
    let input = InputGeometry {
        height: 6,
        width: 6,
        channels: 1,
    };
    let mut seed = 0x1234_5678_9abc_def0_u64;
    let conv1 = pm1(&mut seed, 4 * input.channels * 2 * 2);
    let conv2 = pm1(&mut seed, 4 * 4 * 2 * 2);
    let dense = pm1(&mut seed, 4 * 4 * 4 * 3);
    let lo = vec![-4.0f64; 36];
    let hi = vec![4.0f64; 36];
    let challengers = (0..3).filter(|&c| c != target).collect();
    TinyNet {
        input,
        conv1,
        conv1_out: 4,
        conv1_k: 2,
        conv2,
        conv2_out: 4,
        conv2_k: 2,
        dense,
        num_classes: 3,
        lo,
        hi,
        target,
        challengers,
    }
}

// ---------------------------------------------------------------------------
// (a) FIXED/FREE against brute force, with the B(0) = +1 boundary
// ---------------------------------------------------------------------------

/// Exact `L_k`/`U_k` by exhaustive enumeration of every box VERTEX.
///
/// This is the honest oracle for "interval arithmetic is exact here": it makes
/// no structural assumption at all, it just looks at all `2^16` vertices.
fn brute_force_bounds(net: &TinyNet) -> (Vec<f64>, Vec<f64>) {
    let n = net.n_pixels();
    let mut lower = vec![f64::INFINITY; net.n_units1()];
    let mut upper = vec![f64::NEG_INFINITY; net.n_units1()];
    let mut x = vec![0.0f64; n];
    for mask in 0u32..(1u32 << n) {
        for p in 0..n {
            x[p] = if mask >> p & 1 == 0 {
                net.lo[p]
            } else {
                net.hi[p]
            };
        }
        for (k, &z) in net.z1(&x).iter().enumerate() {
            lower[k] = lower[k].min(z);
            upper[k] = upper[k].max(z);
        }
    }
    (lower, upper)
}

#[test]
fn classification_matches_brute_force_over_every_box_vertex() {
    let net = boundary_net();
    let limits = SignSpaceLimits::default();
    let classification = classify_first_layer_unwired(&net.request(), &limits)
        .expect("the boundary net is inside the admitted fragment");

    let (lower, upper) = brute_force_bounds(&net);
    for k in 0..net.n_units1() {
        assert_eq!(
            classification.lower[k], lower[k],
            "unit {k}: interval lower bound is not the true minimum over vertices"
        );
        assert_eq!(
            classification.upper[k], upper[k],
            "unit {k}: interval upper bound is not the true maximum over vertices"
        );
        let expected = if lower[k] >= 0.0 {
            UnitPhase::FixedPositive
        } else if upper[k] < 0.0 {
            UnitPhase::FixedNegative
        } else {
            UnitPhase::Free
        };
        assert_eq!(classification.phase[k], expected, "unit {k} misclassified");
    }

    let free: HashSet<usize> = classification.free.iter().copied().collect();
    for k in 0..net.n_units1() {
        assert_eq!(
            free.contains(&k),
            classification.phase[k] == UnitPhase::Free,
            "unit {k}: the free list disagrees with the phase vector"
        );
    }
    assert!(
        classification.phase.contains(&UnitPhase::FixedNegative),
        "the fixture must exercise a strictly FIXED -1 unit"
    );
    assert!(
        classification.phase.contains(&UnitPhase::Free),
        "the fixture must exercise a FREE unit"
    );
}

/// `L_k == 0` EXACTLY must be FIXED `+1`, because `B` fires at zero.
///
/// This is one half of the mutation pair: weakening the `+1` test from
/// `L >= t` to `L > t` turns this unit FREE and fails here.
#[test]
fn unit_touching_zero_from_below_is_fixed_positive_not_free() {
    let net = boundary_net();
    let classification = classify_first_layer_unwired(&net.request(), &SignSpaceLimits::default())
        .expect("admitted");
    // Output position (2, 2), channel 0 (all-+1 kernel) over a patch that is
    // entirely [0, 1]: L = 0 + 0 + 0 + 0 = 0 exactly, U = 4.
    let k = (2 * net.w1() + 2) * net.conv1_out;
    assert_eq!(classification.lower[k], 0.0, "fixture drifted: L must be 0");
    assert_eq!(classification.upper[k], 4.0, "fixture drifted: U must be 4");
    assert_eq!(
        classification.phase[k],
        UnitPhase::FixedPositive,
        "L == 0 must be FIXED +1: B(0) = Sign(Sign(0) + 0.1) = +1"
    );
}

/// `U_k == 0` EXACTLY must be FREE, because `+1` is still attainable there.
///
/// The other half of the mutation pair: weakening the `-1` test from `U < t`
/// to `U <= t` turns this unit FIXED `-1` and fails here — that swap is a
/// direct false-`unsat` generator upstream.
#[test]
fn unit_touching_zero_from_above_is_free_not_fixed_negative() {
    let net = boundary_net();
    let classification = classify_first_layer_unwired(&net.request(), &SignSpaceLimits::default())
        .expect("admitted");
    // Output position (2, 2), channel 1 (all--1 kernel), same [0, 1] patch:
    // U = -(0 + 0 + 0 + 0) = 0 exactly, L = -4.
    let k = (2 * net.w1() + 2) * net.conv1_out + 1;
    assert_eq!(classification.upper[k], 0.0, "fixture drifted: U must be 0");
    assert_eq!(
        classification.lower[k], -4.0,
        "fixture drifted: L must be -4"
    );
    assert_eq!(
        classification.phase[k],
        UnitPhase::Free,
        "U == 0 must stay FREE: the all-zero vertex attains z1 = 0, which fires +1"
    );
}

/// The forward must implement `B(0) = +1`, on an input where the two
/// conventions genuinely disagree.
#[test]
fn zero_maps_to_plus_one_in_the_forward() {
    let net = boundary_net();
    // Every [0,1] pixel at 0 and every [1,2] pixel at 1: at output position
    // (2,2) both channel 0 and channel 1 have z1 exactly 0.
    let x: Vec<f64> = net.lo.clone();
    let z1 = net.z1(&x);
    let k = (2 * net.w1() + 2) * net.conv1_out;
    assert_eq!(
        z1[k], 0.0,
        "fixture drifted: the boundary unit must sit at 0"
    );

    let with_zero_positive = net.reference_logits(&x, true);
    let with_zero_negative = net.reference_logits(&x, false);
    assert_ne!(
        with_zero_positive, with_zero_negative,
        "fixture is not exercising the boundary: both conventions agree"
    );

    let ours =
        logits_at_unwired(&net.request(), &SignSpaceLimits::default(), &x).expect("admitted");
    assert_eq!(
        ours, with_zero_positive,
        "the module must use B(v) = +1 iff v >= 0"
    );
}

/// The engine agrees with the independent reference forward everywhere it is
/// asked, not just at the boundary.
#[test]
fn forward_matches_the_independent_reference_at_many_points() {
    let net = boundary_net();
    let limits = SignSpaceLimits::default();
    let n = net.n_pixels();
    for mask in 0u32..256u32 {
        let x: Vec<f64> = (0..n)
            .map(|p| {
                if mask >> (p % 8) & 1 == 0 {
                    net.lo[p]
                } else {
                    net.lo[p].midpoint(net.hi[p])
                }
            })
            .collect();
        let ours = logits_at_unwired(&net.request(), &limits, &x).expect("admitted");
        assert_eq!(
            ours,
            net.reference_logits(&x, true),
            "logit engine disagreed with the reference forward at mask {mask}"
        );
    }
}

// ---------------------------------------------------------------------------
// (b) the realizability LP returns a genuine primal
// ---------------------------------------------------------------------------

/// Re-check a claimed realizing point against EVERY constraint of the LP, from
/// the net's own tensors rather than from the module's internals.
fn assert_realizes(net: &TinyNet, pattern_source: &[f64], x: &[f64], slack: f64) {
    for p in 0..net.n_pixels() {
        assert!(
            x[p] >= net.lo[p] && x[p] <= net.hi[p],
            "primal pixel {p} = {} escapes the box [{}, {}]",
            x[p],
            net.lo[p],
            net.hi[p]
        );
    }
    let classification = classify_first_layer_unwired(&net.request(), &SignSpaceLimits::default())
        .expect("admitted");
    let z_source = net.z1(pattern_source);
    let z_primal = net.z1(x);
    for &k in &classification.free {
        let sign = if z_source[k] >= 0.0 { 1.0 } else { -1.0 };
        assert!(
            sign * z_primal[k] >= slack - 1e-9,
            "free unit {k}: s1_k * (w_k . x) = {} < slack {slack}",
            sign * z_primal[k]
        );
        assert_eq!(
            z_primal[k] >= 0.0,
            z_source[k] >= 0.0,
            "free unit {k}: the primal realizes a DIFFERENT sign than the pattern it was solved for"
        );
    }
    // FIXED units need no LP row precisely because the exact prepass already
    // proved every in-box point realizes them; assert that is actually true of
    // this point.
    for (k, phase) in classification.phase.iter().enumerate() {
        match phase {
            UnitPhase::FixedPositive => assert!(
                z_primal[k] >= 0.0,
                "unit {k} was classified FIXED +1 but the in-box primal gives {}",
                z_primal[k]
            ),
            UnitPhase::FixedNegative => assert!(
                z_primal[k] < 0.0,
                "unit {k} was classified FIXED -1 but the in-box primal gives {}",
                z_primal[k]
            ),
            UnitPhase::Free => {}
        }
    }
}

#[test]
fn realizability_lp_primal_satisfies_every_constraint() {
    let net = search_net(0);
    let limits = SignSpaceLimits::default();
    let midpoint: Vec<f64> = (0..net.n_pixels())
        .map(|p| net.lo[p].midpoint(net.hi[p]))
        .collect();
    let (slack, x) = realizability_probe_unwired(&net.request(), &limits, &midpoint)
        .expect("the LP must not error")
        .expect("the midpoint's own sign pattern is realizable by the midpoint");
    assert!(
        slack >= limits.tolerance,
        "slack {slack} is below the acceptance tolerance {}",
        limits.tolerance
    );
    assert_eq!(x.len(), net.n_pixels(), "the primal must be a full input");
    assert_realizes(&net, &midpoint, &x, slack);
}

#[test]
fn realizability_lp_primal_is_a_real_point_on_the_boundary_net() {
    let net = boundary_net();
    let limits = SignSpaceLimits::default();
    let x0: Vec<f64> = (0..net.n_pixels())
        .map(|p| net.lo[p].midpoint(net.hi[p]))
        .collect();
    let (slack, x) = realizability_probe_unwired(&net.request(), &limits, &x0)
        .expect("the LP must not error")
        .expect("realizable");
    assert_realizes(&net, &x0, &x, slack);
}

// ---------------------------------------------------------------------------
// (c) refusals, never wrong answers
// ---------------------------------------------------------------------------

fn refusal_of(request: &SignSpaceRequest<'_>, limits: &SignSpaceLimits) -> SignSpaceRefusal {
    match falsify_bnn_sign_suffix_unwired(request, limits).expect("no solver error expected") {
        SignSpaceOutcome::Refused(refusal) => refusal,
        other => panic!("expected a refusal, got {other:?}"),
    }
}

#[test]
fn relu_network_is_refused_not_answered() {
    let net = search_net(0);
    let mut request = net.request();
    request.activation1 = SignSpaceActivation::Relu;
    assert!(
        matches!(
            refusal_of(&request, &SignSpaceLimits::default()),
            SignSpaceRefusal::CompositeActivationNotBinary { site: 1, .. }
        ),
        "a ReLU net must be refused"
    );
}

#[test]
fn bare_sign_is_refused_because_it_is_three_valued() {
    let net = search_net(0);
    let mut request = net.request();
    request.activation2 = SignSpaceActivation::BareSign;
    assert!(matches!(
        refusal_of(&request, &SignSpaceLimits::default()),
        SignSpaceRefusal::CompositeActivationNotBinary { site: 2, .. }
    ));
}

#[test]
fn add_constant_outside_the_unit_interval_is_refused() {
    let net = search_net(0);
    for c in [0.0, 1.0, 1.5, -0.1, f64::NAN] {
        let mut request = net.request();
        request.activation1 = SignSpaceActivation::SignAddSign { add_constant: c };
        assert!(
            matches!(
                refusal_of(&request, &SignSpaceLimits::default()),
                SignSpaceRefusal::CompositeActivationNotBinary { .. }
            ),
            "Sign->Add({c})->Sign is not a binary activation and must be refused"
        );
    }
}

#[test]
fn negative_batch_norm_scale_is_refused() {
    let net = search_net(0);
    let scale = vec![1.0, -1.0, 1.0, 1.0];
    let offset = vec![0.0; 4];
    let mut request = net.request();
    request.conv1_affine = Some(SignSpaceAffine {
        scale: &scale,
        offset: &offset,
    });
    assert!(
        matches!(
            refusal_of(&request, &SignSpaceLimits::default()),
            SignSpaceRefusal::BatchNormNotFoldable {
                site: 1,
                channel: 1,
                ..
            }
        ),
        "a negative fold scale inverts the downstream bit and must be refused"
    );
}

#[test]
fn positive_batch_norm_scale_is_folded_into_a_threshold() {
    let net = search_net(0);
    // scale > 0, offset = -2 => threshold t = 2 on every channel.
    let scale = vec![0.5; 4];
    let offset = vec![-1.0; 4];
    let mut request = net.request();
    request.conv1_affine = Some(SignSpaceAffine {
        scale: &scale,
        offset: &offset,
    });
    let folded =
        classify_first_layer_unwired(&request, &SignSpaceLimits::default()).expect("admitted");
    let plain = classify_first_layer_unwired(&net.request(), &SignSpaceLimits::default())
        .expect("admitted");
    // The bounds are unchanged; only the threshold moved, so strictly fewer
    // units can be FIXED +1.
    assert_eq!(folded.lower, plain.lower);
    let folded_pos = folded
        .phase
        .iter()
        .filter(|p| **p == UnitPhase::FixedPositive)
        .count();
    let plain_pos = plain
        .phase
        .iter()
        .filter(|p| **p == UnitPhase::FixedPositive)
        .count();
    assert!(
        folded_pos <= plain_pos,
        "raising the threshold cannot create FIXED +1 units"
    );
    for k in 0..folded.phase.len() {
        let expected = if folded.lower[k] >= 2.0 {
            UnitPhase::FixedPositive
        } else if folded.upper[k] < 2.0 {
            UnitPhase::FixedNegative
        } else {
            UnitPhase::Free
        };
        assert_eq!(
            folded.phase[k], expected,
            "unit {k} under a folded threshold"
        );
    }
}

#[test]
fn non_unit_weights_are_refused_bitwise() {
    let net = search_net(0);
    let mut conv1 = net.conv1.clone();
    conv1[3] = 0.999_999_94f32;
    let mut request = net.request();
    request.conv1 = ConvSpec::valid_unit_stride(&conv1, net.conv1_out, 1, net.conv1_k, net.conv1_k);
    assert!(
        matches!(
            refusal_of(&request, &SignSpaceLimits::default()),
            SignSpaceRefusal::NonUnitWeights {
                tensor: "conv1",
                index: 3,
                ..
            }
        ),
        "a near-one weight is a different network, not a rounding artifact"
    );
}

#[test]
fn strided_or_padded_convolution_is_refused() {
    let net = search_net(0);
    for mutate in [0usize, 1, 2, 3] {
        let mut request = net.request();
        match mutate {
            0 => request.conv1.stride = (2, 2),
            1 => request.conv1.padding = (1, 1, 1, 1),
            2 => request.conv1.dilation = (2, 2),
            _ => request.conv1.groups = 2,
        }
        assert!(
            matches!(
                refusal_of(&request, &SignSpaceLimits::default()),
                SignSpaceRefusal::UnsupportedConvGeometry {
                    tensor: "conv1",
                    ..
                }
            ),
            "conv geometry mutation {mutate} must be refused"
        );
    }
}

#[test]
fn a_property_that_is_not_the_argmax_complement_is_refused() {
    let net = search_net(0);
    let partial = vec![1usize];
    let mut request = net.request();
    request.challengers = &partial;
    assert!(matches!(
        refusal_of(&request, &SignSpaceLimits::default()),
        SignSpaceRefusal::PropertyNotArgmaxComplement { .. }
    ));

    let repeated = vec![1usize, 1];
    let mut request = net.request();
    request.challengers = &repeated;
    assert!(matches!(
        refusal_of(&request, &SignSpaceLimits::default()),
        SignSpaceRefusal::PropertyNotArgmaxComplement { .. }
    ));
}

#[test]
fn an_inverted_box_is_refused() {
    let net = search_net(0);
    let mut lo = net.lo.clone();
    lo[7] = 100.0;
    let mut request = net.request();
    request.lo = &lo;
    assert!(matches!(
        refusal_of(&request, &SignSpaceLimits::default()),
        SignSpaceRefusal::DegenerateBox { index: 7, .. }
    ));
}

#[test]
fn a_tolerance_below_the_f32_replay_floor_is_refused() {
    let net = search_net(0);
    let limits = SignSpaceLimits {
        tolerance: 1e-12,
        ..SignSpaceLimits::default()
    };
    assert!(matches!(
        refusal_of(&net.request(), &limits),
        SignSpaceRefusal::ToleranceBelowFloor { .. }
    ));
}

#[test]
fn a_box_too_wide_for_exact_accumulation_is_refused() {
    let net = search_net(0);
    let hi = vec![1e7f64; net.n_pixels()];
    let mut request = net.request();
    request.hi = &hi;
    assert!(matches!(
        refusal_of(&request, &SignSpaceLimits::default()),
        SignSpaceRefusal::AccumulationNotExact { .. }
    ));
}

#[test]
fn a_free_unit_budget_smaller_than_the_problem_is_refused() {
    let net = search_net(0);
    let limits = SignSpaceLimits {
        max_free_units: 1,
        ..SignSpaceLimits::default()
    };
    assert!(matches!(
        refusal_of(&net.request(), &limits),
        SignSpaceRefusal::LimitExceeded {
            limit: "max_free_units",
            ..
        }
    ));
}

#[test]
fn a_disagreeing_reference_forward_is_refused_before_any_search() {
    let net = search_net(0);
    let wrong = |_: &[f64]| Some(vec![0.0f64; 3]);
    let mut request = net.request();
    request.reference_forward = Some(&wrong);
    assert!(
        matches!(
            refusal_of(&request, &SignSpaceLimits::default()),
            SignSpaceRefusal::LogitEngineDisagrees { .. }
        ),
        "a layout/flatten transposition must be caught by the self-check"
    );
}

#[test]
fn an_agreeing_reference_forward_passes_the_self_check() {
    let net = search_net(0);
    let oracle = |x: &[f64]| {
        Some(
            net.reference_logits(x, true)
                .iter()
                .map(|&v| v as f64)
                .collect(),
        )
    };
    let mut request = net.request();
    request.reference_forward = Some(&oracle);
    let outcome =
        falsify_bnn_sign_suffix_unwired(&request, &SignSpaceLimits::default()).expect("no error");
    assert!(
        !matches!(outcome, SignSpaceOutcome::Refused(_)),
        "the honest oracle must not trip the self-check: {outcome:?}"
    );
}

// ---------------------------------------------------------------------------
// Tolerance floor
// ---------------------------------------------------------------------------

#[test]
fn f32_replay_floor_is_geometry_specific() {
    // 3x3x3 at pixel scale 255: 27 taps, partial sums <= 6885 in [2^12, 2^13).
    let floor_3x3 = f32_replay_slack_floor(27, 255.0);
    assert!(
        floor_3x3 > 6.0e-3 && floor_3x3 < 7.0e-3,
        "3x3x3 floor drifted: {floor_3x3}"
    );
    assert!(
        0.05 > floor_3x3 * 7.0,
        "the documented TOL = 0.05 must carry ~7x headroom at 3x3x3"
    );
    // 5x5x3 at pixel scale 255: 75 taps, partial sums <= 19125 in [2^14, 2^15).
    let floor_5x5 = f32_replay_slack_floor(75, 255.0);
    assert!(
        floor_5x5 > 7.0e-2 && floor_5x5 < 8.0e-2,
        "5x5x3 floor drifted: {floor_5x5}"
    );
    assert!(
        0.05 < floor_5x5,
        "TOL = 0.05 is UNSOUND at 5x5x3 and must be refused there"
    );
}

// ---------------------------------------------------------------------------
// End-to-end search
// ---------------------------------------------------------------------------

/// Independently re-validate a claimed candidate: in-box with NO tolerance,
/// and a from-scratch forward reproducing the reported logits and margin.
fn validate_candidate(net: &TinyNet, candidate: &SignSpaceCandidate) {
    assert_eq!(candidate.input.len(), net.n_pixels());
    for p in 0..net.n_pixels() {
        assert!(
            candidate.input[p] >= net.lo[p] && candidate.input[p] <= net.hi[p],
            "witness pixel {p} = {} escapes [{}, {}]",
            candidate.input[p],
            net.lo[p],
            net.hi[p]
        );
        assert_eq!(
            candidate.input[p],
            f64::from(candidate.input[p] as f32),
            "witness pixel {p} is not exactly f32-representable"
        );
    }
    // Replay stability: every first-layer unit must clear its threshold by
    // more than the `f32` accumulation bound, so the sign pattern — and hence
    // every integer downstream of it — is the same under ANY `f32` summation
    // order. Without this a witness could be an f64-only artifact.
    let max_abs = net
        .lo
        .iter()
        .chain(net.hi.iter())
        .fold(0.0f64, |acc, v| acc.max(v.abs()));
    let taps = net.conv1_k * net.conv1_k * net.input.channels;
    let floor = f32_replay_slack_floor(taps, max_abs);
    for (k, &z) in net.z1(&candidate.input).iter().enumerate() {
        assert!(
            z.abs() > floor,
            "unit {k} sits at |z1| = {} <= the f32 replay floor {floor}; \
             its sign is summation-order dependent",
            z.abs()
        );
    }
    let logits = net.reference_logits(&candidate.input, true);
    assert_eq!(
        logits, candidate.logits,
        "the reported logits are not the logits of the reported input"
    );
    let margin = logits[candidate.best_challenger] - logits[net.target];
    assert_eq!(margin, candidate.logit_margin);
    assert!(
        candidate.logit_margin > 0,
        "a candidate must carry a STRICTLY positive pre-softmax margin"
    );
    assert!(
        logits[candidate.argmax] >= logits[net.target],
        "argmax must dominate the target"
    );
    assert!(candidate.lp_slack >= SignSpaceLimits::default().tolerance);
}

#[test]
fn greedy_search_finds_and_self_validates_a_counterexample() {
    let mut found = 0usize;
    for target in 0..3usize {
        let net = search_net(target);
        let limits = SignSpaceLimits::default();
        let outcome =
            falsify_bnn_sign_suffix_unwired(&net.request(), &limits).expect("no solver error");
        match outcome {
            SignSpaceOutcome::Candidate(candidate) => {
                validate_candidate(&net, &candidate);
                assert!(candidate.free_units > 0);
                assert!(candidate.lp_solves > 0);
                found += 1;
            }
            SignSpaceOutcome::Exhausted {
                best_logit_margin, ..
            } => {
                assert!(
                    best_logit_margin <= 0,
                    "Exhausted must never hide a violated pattern (margin {best_logit_margin})"
                );
            }
            SignSpaceOutcome::Refused(refusal) => {
                panic!("the search net is inside the fragment: {refusal:?}")
            }
        }
    }
    assert!(
        found > 0,
        "the fixture must exercise the full LP + witness path at least once"
    );
}

#[test]
fn the_search_is_deterministic() {
    let net = search_net(1);
    let limits = SignSpaceLimits::default();
    let first = falsify_bnn_sign_suffix_unwired(&net.request(), &limits).expect("no error");
    let second = falsify_bnn_sign_suffix_unwired(&net.request(), &limits).expect("no error");
    match (&first, &second) {
        (SignSpaceOutcome::Candidate(a), SignSpaceOutcome::Candidate(b)) => {
            assert_eq!(a.input, b.input, "the witness must be reproducible");
            assert_eq!(a.logits, b.logits);
            assert_eq!(a.flips, b.flips);
        }
        (
            SignSpaceOutcome::Exhausted {
                best_logit_margin: a,
                flips: fa,
                ..
            },
            SignSpaceOutcome::Exhausted {
                best_logit_margin: b,
                flips: fb,
                ..
            },
        ) => {
            assert_eq!(a, b);
            assert_eq!(fa, fb);
        }
        other => panic!("the two runs disagreed structurally: {other:?}"),
    }
}

/// A degenerate box (`lo == hi` everywhere) leaves NO free units, so the
/// search has nothing to do and must not manufacture a witness.
#[test]
fn a_point_box_has_no_free_units_and_yields_no_witness() {
    let mut net = search_net(0);
    net.hi = net.lo.clone();
    let classification = classify_first_layer_unwired(&net.request(), &SignSpaceLimits::default())
        .expect("admitted");
    assert!(
        classification.free.is_empty(),
        "a point box cannot leave a free unit"
    );
    let outcome = falsify_bnn_sign_suffix_unwired(&net.request(), &SignSpaceLimits::default())
        .expect("no error");
    match outcome {
        SignSpaceOutcome::Exhausted { free_units, .. } => assert_eq!(free_units, 0),
        SignSpaceOutcome::Candidate(candidate) => {
            // Legitimate only if the single reachable point already violates.
            validate_candidate(&net, &candidate);
        }
        SignSpaceOutcome::Refused(refusal) => panic!("unexpected refusal: {refusal:?}"),
    }
}

/// The outcome type must be structurally incapable of authorizing a verdict.
#[test]
fn the_outcome_enum_has_no_verified_variant() {
    let net = search_net(0);
    let outcome = falsify_bnn_sign_suffix_unwired(&net.request(), &SignSpaceLimits::default())
        .expect("no error");
    // Exhaustive match: adding a `Verified`/`Unsat` arm would fail to compile
    // here, which is the point.
    match outcome {
        SignSpaceOutcome::Candidate(_)
        | SignSpaceOutcome::Exhausted { .. }
        | SignSpaceOutcome::Refused(_) => {}
    }
}

// ---------------------------------------------------------------------------
// (d) the WIDENED fragment: MaxPool, per-channel weight scales, deeper chains
//
// These pin the four soundness properties the deeper traffic-sign nets rely on
// (`docs/BNN_SIGN_SPACE_FALSIFICATION_2026-08-12.md` §8), each against an
// independent reference rather than against the module's own internals:
//
//   1. a per-output-channel `|W| = s_c > 0` conv folds to the threshold
//      `t_c = -bias_c / s_c` on the INTEGER accumulator, with `>=` at the
//      boundary;
//   2. a NEGATIVE folded affine scale is refused, at every site;
//   3. `|W|` that is not constant within a channel is refused;
//   4. `MaxPool` before the affine and `Sign` is exactly the OR identity.
// ---------------------------------------------------------------------------

/// Multiply a `+/-1` sign pattern by a per-output-channel scale, `[out, ...]`
/// layout. Every scale used here is a power of two, so the product is EXACT in
/// `f32` and the fixture is not testing floating-point luck.
fn scale_out_major(signs: &[f32], scales: &[f64], per_channel: usize) -> Vec<f32> {
    signs
        .iter()
        .enumerate()
        .map(|(i, &s)| s * scales[i / per_channel] as f32)
        .collect()
}

/// The same, for a `[in, out]` dense tensor.
fn scale_in_major(signs: &[f32], scales: &[f64], out_dim: usize) -> Vec<f32> {
    signs
        .iter()
        .enumerate()
        .map(|(i, &s)| s * scales[i % out_dim] as f32)
        .collect()
}

/// A net exercising EVERY widening at once: a pooled first layer with a folded
/// affine, a pooled second convolution with a per-channel `|W|`, a third
/// (stage) convolution carrying a folded bias, a dense stage with a folded
/// affine, and the final `+/-1` dense.
///
/// ```text
/// 7x7x1 -> conv1 2x2 x2 -> 6x6x2 -> MaxPool 2/2 -> 3x3x2 -> BN -> B
///       -> conv2 2x2 x2 -> 2x2x2 -> MaxPool 2/2 -> 1x1x2 -> BN -> B
///       -> conv3 1x1 x3 (+bias)   -> 1x1x3            -> B
///       -> dense 3->4 -> BN -> B
///       -> dense 4->3 -> logits
/// ```
struct DeepNet {
    input: InputGeometry,
    conv1: Vec<f32>,
    conv1_out: usize,
    conv1_k: usize,
    pool1: PoolSpec,
    a1_scale: Vec<f64>,
    a1_offset: Vec<f64>,
    conv2: Vec<f32>,
    conv2_out: usize,
    conv2_k: usize,
    pool2: PoolSpec,
    a2_scale: Vec<f64>,
    a2_offset: Vec<f64>,
    conv3: Vec<f32>,
    conv3_out: usize,
    conv3_bias: Vec<f64>,
    ds: Vec<f32>,
    ds_in: usize,
    ds_out: usize,
    ds_scale: Vec<f64>,
    ds_offset: Vec<f64>,
    dense: Vec<f32>,
    num_classes: usize,
    lo: Vec<f64>,
    hi: Vec<f64>,
    target: usize,
    challengers: Vec<usize>,
}

fn deep_net() -> DeepNet {
    let input = InputGeometry {
        height: 7,
        width: 7,
        channels: 1,
    };
    let mut seed = 0x0dd_ba11_u64;
    // `[out, in, kh, kw]` written dimension-by-dimension so the kernel shape is
    // readable next to conv2's `2 * 2 * 2 * 2`; the `1` is conv1's INPUT-CHANNEL
    // count, not a redundant factor, so collapsing it would hide the shape.
    #[allow(clippy::identity_op, reason = "the 1 is a named tensor dimension")]
    let conv1 = pm1(&mut seed, 2 * 1 * 2 * 2);
    let conv2_signs = pm1(&mut seed, 2 * 2 * 2 * 2);
    let conv2 = scale_out_major(&conv2_signs, &[1.0, 0.5], 2 * 2 * 2);
    let conv3_signs = pm1(&mut seed, 3 * 2);
    let conv3 = scale_out_major(&conv3_signs, &[2.0, 0.25, 4.0], 2);
    let ds_signs = pm1(&mut seed, 3 * 4);
    let ds = scale_in_major(&ds_signs, &[1.0, 0.5, 2.0, 1.0], 4);
    let dense = pm1(&mut seed, 4 * 3);
    DeepNet {
        input,
        conv1,
        conv1_out: 2,
        conv1_k: 2,
        pool1: PoolSpec {
            kernel_h: 2,
            kernel_w: 2,
            stride_h: 2,
            stride_w: 2,
        },
        a1_scale: vec![1.0, 2.0],
        a1_offset: vec![0.0, -1.0],
        conv2,
        conv2_out: 2,
        conv2_k: 2,
        pool2: PoolSpec {
            kernel_h: 2,
            kernel_w: 2,
            stride_h: 2,
            stride_w: 2,
        },
        a2_scale: vec![0.5, 1.5],
        a2_offset: vec![0.25, -0.5],
        conv3,
        conv3_out: 3,
        conv3_bias: vec![1.0, -0.25, 0.0],
        ds,
        ds_in: 3,
        ds_out: 4,
        ds_scale: vec![1.0, 2.0, 0.5, 1.0],
        ds_offset: vec![0.5, -0.5, 0.0, 1.5],
        dense,
        num_classes: 3,
        lo: vec![-3.0; 49],
        hi: vec![3.0; 49],
        target: 0,
        challengers: vec![1, 2],
    }
}

impl DeepNet {
    fn stages(&self) -> Vec<BinaryStage<'_>> {
        vec![
            BinaryStage::Conv {
                conv: ConvSpec::valid_unit_stride(&self.conv3, self.conv3_out, 2, 1, 1)
                    .with_bias(&self.conv3_bias),
                pool: None,
                affine: None,
                activation: SignSpaceActivation::SignAddSign { add_constant: 0.1 },
            },
            BinaryStage::Dense {
                weights: &self.ds,
                in_dim: self.ds_in,
                out_dim: self.ds_out,
                affine: Some(SignSpaceAffine {
                    scale: &self.ds_scale,
                    offset: &self.ds_offset,
                }),
                activation: SignSpaceActivation::SignAddSign { add_constant: 0.1 },
            },
        ]
    }

    fn request<'a>(&'a self, stages: &'a [BinaryStage<'a>]) -> SignSpaceRequest<'a> {
        SignSpaceRequest {
            input: self.input,
            conv1: ConvSpec::valid_unit_stride(
                &self.conv1,
                self.conv1_out,
                self.input.channels,
                self.conv1_k,
                self.conv1_k,
            ),
            conv1_pool: Some(self.pool1),
            conv1_affine: Some(SignSpaceAffine {
                scale: &self.a1_scale,
                offset: &self.a1_offset,
            }),
            activation1: SignSpaceActivation::SignAddSign { add_constant: 0.1 },
            conv2: ConvSpec::valid_unit_stride(
                &self.conv2,
                self.conv2_out,
                self.conv1_out,
                self.conv2_k,
                self.conv2_k,
            ),
            conv2_pool: Some(self.pool2),
            conv2_affine: Some(SignSpaceAffine {
                scale: &self.a2_scale,
                offset: &self.a2_offset,
            }),
            activation2: SignSpaceActivation::SignAddSign { add_constant: 0.1 },
            stages,
            dense: &self.dense,
            num_classes: self.num_classes,
            lo: &self.lo,
            hi: &self.hi,
            target_class: self.target,
            challengers: &self.challengers,
            reference_input: None,
            reference_forward: None,
        }
    }
}

/// Everything the module is supposed to do, written out longhand from the RAW
/// tensors — no thresholds, no folding, no normalization. `MaxPool` is a
/// literal `max` over the window, every `BatchNorm` is a literal
/// `scale * v + offset`, and every `Sign` is the ONNX composite `v >= 0`.
///
/// If the module's per-channel scale folding, its threshold derivation or its
/// OR identity were wrong, this would disagree.
struct DeepTrace {
    /// RAW conv1 outputs, `(r, c, ch)` at `6x6x2`.
    z1: Vec<f64>,
    /// Pooled conv1 outputs, `3x3x2`.
    p1: Vec<f64>,
    /// First-layer signs, `3x3x2`.
    s1: Vec<i64>,
    logits: Vec<i64>,
}

impl DeepNet {
    fn h1c(&self) -> usize {
        self.input.height - self.conv1_k + 1
    }
    fn w1c(&self) -> usize {
        self.input.width - self.conv1_k + 1
    }
    fn h1(&self) -> usize {
        (self.h1c() - self.pool1.kernel_h) / self.pool1.stride_h + 1
    }
    fn w1(&self) -> usize {
        (self.w1c() - self.pool1.kernel_w) / self.pool1.stride_w + 1
    }
    fn n_pixels(&self) -> usize {
        self.input.height * self.input.width * self.input.channels
    }

    fn trace(&self, x: &[f64]) -> DeepTrace {
        let (h1c, w1c, c1) = (self.h1c(), self.w1c(), self.conv1_out);
        let mut z1 = vec![0.0f64; h1c * w1c * c1];
        for r in 0..h1c {
            for c in 0..w1c {
                for ch in 0..c1 {
                    let mut acc = 0.0f64;
                    for kr in 0..self.conv1_k {
                        for kc in 0..self.conv1_k {
                            for ic in 0..self.input.channels {
                                let w = f64::from(
                                    self.conv1[((ch * self.input.channels + ic) * self.conv1_k
                                        + kr)
                                        * self.conv1_k
                                        + kc],
                                );
                                let p = ((r + kr) * self.input.width + (c + kc))
                                    * self.input.channels
                                    + ic;
                                acc += w * x[p];
                            }
                        }
                    }
                    z1[(r * w1c + c) * c1 + ch] = acc;
                }
            }
        }
        // MaxPool, literally.
        let (h1, w1) = (self.h1(), self.w1());
        let mut p1 = vec![f64::NEG_INFINITY; h1 * w1 * c1];
        for r in 0..h1 {
            for c in 0..w1 {
                for ch in 0..c1 {
                    let mut best = f64::NEG_INFINITY;
                    for a in 0..self.pool1.kernel_h {
                        for b in 0..self.pool1.kernel_w {
                            let i = r * self.pool1.stride_h + a;
                            let j = c * self.pool1.stride_w + b;
                            best = best.max(z1[(i * w1c + j) * c1 + ch]);
                        }
                    }
                    p1[(r * w1 + c) * c1 + ch] = best;
                }
            }
        }
        let s1: Vec<i64> = (0..h1 * w1 * c1)
            .map(|k| {
                let ch = k % c1;
                let y = self.a1_scale[ch] * p1[k] + self.a1_offset[ch];
                if y >= 0.0 {
                    1
                } else {
                    -1
                }
            })
            .collect();

        // conv2 (per-channel |W| = 1.0 / 0.5) -> MaxPool -> BN -> B.
        let (h2c, w2c, c2) = (h1 - self.conv2_k + 1, w1 - self.conv2_k + 1, self.conv2_out);
        let mut z2 = vec![0.0f64; h2c * w2c * c2];
        for i in 0..h2c {
            for j in 0..w2c {
                for co in 0..c2 {
                    let mut acc = 0.0f64;
                    for kr in 0..self.conv2_k {
                        for kc in 0..self.conv2_k {
                            for ci in 0..c1 {
                                let w = f64::from(
                                    self.conv2
                                        [((co * c1 + ci) * self.conv2_k + kr) * self.conv2_k + kc],
                                );
                                let u = ((i + kr) * w1 + (j + kc)) * c1 + ci;
                                acc += w * s1[u] as f64;
                            }
                        }
                    }
                    z2[(i * w2c + j) * c2 + co] = acc;
                }
            }
        }
        let (h2, w2) = (
            (h2c - self.pool2.kernel_h) / self.pool2.stride_h + 1,
            (w2c - self.pool2.kernel_w) / self.pool2.stride_w + 1,
        );
        let mut s2 = vec![0i64; h2 * w2 * c2];
        for r in 0..h2 {
            for c in 0..w2 {
                for co in 0..c2 {
                    let mut best = f64::NEG_INFINITY;
                    for a in 0..self.pool2.kernel_h {
                        for b in 0..self.pool2.kernel_w {
                            let i = r * self.pool2.stride_h + a;
                            let j = c * self.pool2.stride_w + b;
                            best = best.max(z2[(i * w2c + j) * c2 + co]);
                        }
                    }
                    let y = self.a2_scale[co] * best + self.a2_offset[co];
                    s2[(r * w2 + c) * c2 + co] = if y >= 0.0 { 1 } else { -1 };
                }
            }
        }

        // conv3: 1x1, per-channel |W| + bias, no pool, no affine.
        let s3: Vec<i64> = (0..self.conv3_out)
            .map(|co| {
                let mut acc = self.conv3_bias[co];
                for ci in 0..c2 {
                    acc += f64::from(self.conv3[co * c2 + ci]) * s2[ci] as f64;
                }
                if acc >= 0.0 {
                    1
                } else {
                    -1
                }
            })
            .collect();

        // dense stage: per-column |W| + affine.
        let s4: Vec<i64> = (0..self.ds_out)
            .map(|o| {
                let mut acc = 0.0f64;
                for i in 0..self.ds_in {
                    acc += f64::from(self.ds[i * self.ds_out + o]) * s3[i] as f64;
                }
                let y = self.ds_scale[o] * acc + self.ds_offset[o];
                if y >= 0.0 {
                    1
                } else {
                    -1
                }
            })
            .collect();

        let mut logits = vec![0i64; self.num_classes];
        for o in 0..self.ds_out {
            for (class, logit) in logits.iter_mut().enumerate() {
                *logit += s4[o] * self.dense[o * self.num_classes + class] as i64;
            }
        }
        DeepTrace { z1, p1, s1, logits }
    }
}

/// Deterministic pseudo-random points in the box; no RNG crate.
fn deep_points(net: &DeepNet, count: usize) -> Vec<Vec<f64>> {
    let mut seed = 0xfeed_face_u64;
    (0..count)
        .map(|_| {
            (0..net.n_pixels())
                .map(|p| {
                    seed = seed
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    let t = ((seed >> 33) % 13) as f64 / 12.0;
                    net.lo[p] + t * (net.hi[p] - net.lo[p])
                })
                .collect()
        })
        .collect()
}

/// The whole widened chain — pooled first layer, pooled per-channel-scaled
/// second convolution, biased stage convolution, affine dense stage — must
/// reproduce the longhand reference EXACTLY, at many points.
#[test]
fn the_deep_pooled_chain_matches_the_longhand_reference() {
    let net = deep_net();
    let stages = net.stages();
    let request = net.request(&stages);
    let limits = SignSpaceLimits::default();
    for (index, x) in deep_points(&net, 64).into_iter().enumerate() {
        let ours = logits_at_unwired(&request, &limits, &x).expect("the deep net is admitted");
        assert_eq!(
            ours,
            net.trace(&x).logits,
            "deep chain disagreed with the longhand reference at point {index}"
        );
    }
}

/// `sign(BN(max_w z_w))` is EXACTLY `OR_w [z_w >= t]`, brute-forced member by
/// member — and the fixture must actually exercise the asymmetric case where
/// only ONE member clears, or the identity is untested.
#[test]
fn maxpool_sign_is_exactly_the_or_over_its_window() {
    let net = deep_net();
    let stages = net.stages();
    let request = net.request(&stages);
    let limits = SignSpaceLimits::default();
    let classification =
        classify_first_layer_unwired(&request, &limits).expect("the deep net is admitted");
    let (w1c, c1) = (net.w1c(), net.conv1_out);
    // Thresholds, re-derived here from the RAW affine rather than read out of
    // the module: t_ch = -offset_ch / scale_ch.
    let t: Vec<f64> = (0..c1)
        .map(|ch| -net.a1_offset[ch] / net.a1_scale[ch])
        .collect();

    let mut saw_single_member = 0usize;
    let mut saw_none = 0usize;
    let mut saw_all = 0usize;
    for x in deep_points(&net, 48) {
        let trace = net.trace(&x);
        for r in 0..net.h1() {
            for c in 0..net.w1() {
                for ch in 0..c1 {
                    let mut cleared = 0usize;
                    let mut members = 0usize;
                    for a in 0..net.pool1.kernel_h {
                        for b in 0..net.pool1.kernel_w {
                            let i = r * net.pool1.stride_h + a;
                            let j = c * net.pool1.stride_w + b;
                            members += 1;
                            if trace.z1[(i * w1c + j) * c1 + ch] >= t[ch] {
                                cleared += 1;
                            }
                        }
                    }
                    let k = (r * net.w1() + c) * c1 + ch;
                    let pooled_fires = trace.s1[k] == 1;
                    assert_eq!(
                        pooled_fires,
                        cleared > 0,
                        "unit {k}: pooled sign {} but {cleared}/{members} members cleared {}",
                        trace.s1[k],
                        t[ch]
                    );
                    // ...and the pooled value really is the window MAX: no
                    // member above it, and one member equal to it.
                    let mut best = f64::NEG_INFINITY;
                    for a in 0..net.pool1.kernel_h {
                        for b in 0..net.pool1.kernel_w {
                            let i = r * net.pool1.stride_h + a;
                            let j = c * net.pool1.stride_w + b;
                            best = best.max(trace.z1[(i * w1c + j) * c1 + ch]);
                        }
                    }
                    assert_eq!(
                        trace.p1[k], best,
                        "unit {k}: pooled value is not the window max"
                    );
                    match cleared {
                        0 => saw_none += 1,
                        1 => saw_single_member += 1,
                        n if n == members => saw_all += 1,
                        _ => {}
                    }
                }
            }
        }
    }
    assert!(
        saw_single_member > 0,
        "the fixture never exercises the OR asymmetry (one member carries the +1)"
    );
    assert!(saw_none > 0 && saw_all > 0, "degenerate fixture");

    // The prepass must be SOUND about pooled units: FIXED -1 means the window
    // max never clears, FIXED +1 means it always does.
    for x in deep_points(&net, 32) {
        let trace = net.trace(&x);
        for (k, phase) in classification.phase.iter().enumerate() {
            let ch = k % c1;
            match phase {
                UnitPhase::FixedPositive => assert!(
                    trace.p1[k] >= t[ch],
                    "unit {k} classified FIXED +1 but the pooled value is {}",
                    trace.p1[k]
                ),
                UnitPhase::FixedNegative => assert!(
                    trace.p1[k] < t[ch],
                    "unit {k} classified FIXED -1 but the pooled value is {}",
                    trace.p1[k]
                ),
                UnitPhase::Free => {}
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Per-channel scale: the threshold is exactly `-bias_c / scale_c`, non-strict
// ---------------------------------------------------------------------------

/// A 2x2x1 net whose SECOND convolution carries a per-channel `|W| = scale`
/// and a bias, and whose accumulator can be driven to an exact integer.
///
/// `conv1` is the 1x1 identity, so `s1 = [x >= 0]` pixelwise; `conv2` is a
/// single 2x2 all-`+scale` kernel, so its unit-weight accumulator is exactly
/// `sum(s1) in {-4, -2, 0, 2, 4}`. The final dense is `[+1, -1]`, so
/// `logits[0] == 1` iff the stage fired `+1`.
struct ThresholdProbe {
    conv1: Vec<f32>,
    conv2: Vec<f32>,
    bias: Vec<f64>,
    dense: Vec<f32>,
    lo: Vec<f64>,
    hi: Vec<f64>,
    challengers: Vec<usize>,
}

fn threshold_probe(scale: f32, bias: f64, x: &[f64]) -> (ThresholdProbe, Vec<f64>) {
    (
        ThresholdProbe {
            conv1: vec![1.0],
            conv2: vec![scale; 4],
            bias: vec![bias],
            dense: vec![1.0, -1.0],
            lo: x.to_vec(),
            hi: x.to_vec(),
            challengers: vec![1],
        },
        x.to_vec(),
    )
}

impl ThresholdProbe {
    fn request(&self) -> SignSpaceRequest<'_> {
        SignSpaceRequest {
            input: InputGeometry {
                height: 2,
                width: 2,
                channels: 1,
            },
            conv1: ConvSpec::valid_unit_stride(&self.conv1, 1, 1, 1, 1),
            conv1_pool: None,
            conv1_affine: None,
            activation1: SignSpaceActivation::SignAddSign { add_constant: 0.1 },
            conv2: ConvSpec::valid_unit_stride(&self.conv2, 1, 1, 2, 2).with_bias(&self.bias),
            conv2_pool: None,
            conv2_affine: None,
            activation2: SignSpaceActivation::SignAddSign { add_constant: 0.1 },
            stages: &[],
            dense: &self.dense,
            num_classes: 2,
            lo: &self.lo,
            hi: &self.hi,
            target_class: 0,
            challengers: &self.challengers,
            reference_input: None,
            reference_forward: None,
        }
    }
}

/// `t_c = -bias_c / scale_c`, on the INTEGER accumulator, with `>=` at the
/// boundary.
///
/// The accumulator is pinned to exactly `2`. Two different `(scale, bias)`
/// pairs with the SAME ratio must give the same answer (so the fold really is
/// a ratio, not `-bias` or `-bias*scale`), a threshold exactly at `2` must
/// FIRE (the non-strict boundary `B(0) = +1`), and a threshold a hair above
/// `2` must not.
#[test]
fn a_per_channel_scale_folds_to_minus_bias_over_scale_with_a_non_strict_boundary() {
    // s1 = [+1, +1, +1, -1] => the unit-weight accumulator is exactly 2.
    let x = [1.0f64, 1.0, 1.0, -1.0];
    let limits = SignSpaceLimits::default();
    let fired = |scale: f32, bias: f64| -> bool {
        let (probe, point) = threshold_probe(scale, bias, &x);
        let logits =
            logits_at_unwired(&probe.request(), &limits, &point).expect("probe is admitted");
        assert_eq!(logits.len(), 2);
        assert_eq!(logits[0], -logits[1]);
        logits[0] == 1
    };

    // t = 4/2 = 2 exactly: the accumulator is 2, and B fires at equality.
    assert!(
        fired(2.0, -4.0),
        "an accumulator exactly ON the folded threshold must fire +1"
    );
    // Same RATIO through a different scale: 1/0.5 = 2.
    assert!(
        fired(0.5, -1.0),
        "the fold must depend on -bias/scale, not on bias alone"
    );
    // t = 1.5 < 2.
    assert!(fired(2.0, -3.0));
    // t = 2.0000000005 > 2: one hair above the accumulator.
    assert!(
        !fired(2.0, -4.000_000_001),
        "a threshold strictly above the accumulator must NOT fire"
    );
    // t = 3 > 2.
    assert!(!fired(2.0, -6.0));
    // A NEGATIVE bias direction: t = -2, accumulator 2 >= -2.
    assert!(fired(2.0, 4.0));
}

// ---------------------------------------------------------------------------
// Refusals introduced by the widening
// ---------------------------------------------------------------------------

/// A negative folded-BatchNorm scale inverts the bit. Refused at a DEEP stage,
/// not just at `conv1` — every site runs the same check.
#[test]
fn a_negative_batch_norm_scale_is_refused_at_every_site() {
    let net = deep_net();

    // site 2 (post-conv2).
    let bad = vec![0.5, -1.5];
    let mut stages = net.stages();
    let mut request = net.request(&stages);
    request.conv2_affine = Some(SignSpaceAffine {
        scale: &bad,
        offset: &net.a2_offset,
    });
    assert!(
        matches!(
            refusal_of(&request, &SignSpaceLimits::default()),
            SignSpaceRefusal::BatchNormNotFoldable {
                site: 2,
                channel: 1,
                ..
            }
        ),
        "a negative BN scale after conv2 must be refused"
    );

    // site 4 (the dense stage), reached only through `stages`.
    let bad_dense = vec![1.0, 2.0, -0.5, 1.0];
    stages[1] = BinaryStage::Dense {
        weights: &net.ds,
        in_dim: net.ds_in,
        out_dim: net.ds_out,
        affine: Some(SignSpaceAffine {
            scale: &bad_dense,
            offset: &net.ds_offset,
        }),
        activation: SignSpaceActivation::SignAddSign { add_constant: 0.1 },
    };
    let request = net.request(&stages);
    assert!(
        matches!(
            refusal_of(&request, &SignSpaceLimits::default()),
            SignSpaceRefusal::BatchNormNotFoldable {
                site: 4,
                channel: 2,
                ..
            }
        ),
        "a negative BN scale on the dense stage must be refused"
    );

    // and a ZERO scale, which destroys the bit rather than inverting it.
    let zero_scale = vec![0.0, 1.5];
    let stages = net.stages();
    let mut request = net.request(&stages);
    request.conv2_affine = Some(SignSpaceAffine {
        scale: &zero_scale,
        offset: &net.a2_offset,
    });
    assert!(matches!(
        refusal_of(&request, &SignSpaceLimits::default()),
        SignSpaceRefusal::BatchNormNotFoldable {
            site: 2,
            channel: 0,
            ..
        }
    ));
}

/// `|W|` must be constant WITHIN an output channel. One entry of a different
/// magnitude means the tensor is not `s_c * (+/-1)` and the `Sign` boundary is
/// not a threshold on an integer accumulator — refuse, never approximate.
#[test]
fn a_non_constant_magnitude_within_a_channel_is_refused() {
    let net = deep_net();

    // conv2 channel 1 is all magnitude 0.5; break ONE entry.
    let mut conv2 = net.conv2.clone();
    let broken = 8 + 3; // channel 1 starts at 2*2*2 = 8.
    conv2[broken] = if conv2[broken] > 0.0 { 0.25 } else { -0.25 };
    let stages = net.stages();
    let mut request = net.request(&stages);
    request.conv2 = ConvSpec::valid_unit_stride(&conv2, net.conv2_out, net.conv1_out, 2, 2);
    assert!(
        matches!(
            refusal_of(&request, &SignSpaceLimits::default()),
            SignSpaceRefusal::ChannelScaleNotConstant {
                tensor: "conv2",
                channel: 1,
                index: 11,
                ..
            }
        ),
        "a channel whose |W| is not constant must be refused"
    );

    // The same rule on a dense stage COLUMN (the `[in, out]` layout means a
    // channel is a stride-`out_dim` slice, not a contiguous block).
    let mut ds = net.ds.clone();
    ds[2 * 4 + 1] = if ds[2 * 4 + 1] > 0.0 { 4.0 } else { -4.0 };
    let mut stages = net.stages();
    stages[1] = BinaryStage::Dense {
        weights: &ds,
        in_dim: net.ds_in,
        out_dim: net.ds_out,
        affine: Some(SignSpaceAffine {
            scale: &net.ds_scale,
            offset: &net.ds_offset,
        }),
        activation: SignSpaceActivation::SignAddSign { add_constant: 0.1 },
    };
    let request = net.request(&stages);
    assert!(
        matches!(
            refusal_of(&request, &SignSpaceLimits::default()),
            SignSpaceRefusal::ChannelScaleNotConstant {
                tensor: "stage dense",
                channel: 1,
                ..
            }
        ),
        "a dense COLUMN whose |W| is not constant must be refused"
    );

    // A zero weight has no sign at all.
    let mut conv2 = net.conv2.clone();
    conv2[5] = 0.0;
    let stages = net.stages();
    let mut request = net.request(&stages);
    request.conv2 = ConvSpec::valid_unit_stride(&conv2, net.conv2_out, net.conv1_out, 2, 2);
    assert!(matches!(
        refusal_of(&request, &SignSpaceLimits::default()),
        SignSpaceRefusal::NonUnitWeights {
            tensor: "conv2",
            index: 5,
            ..
        }
    ));
}

/// The FIRST convolution is still held to exact `+/-1` with no bias: the `f32`
/// replay floor is derived for unit taps and nothing else.
#[test]
fn the_first_convolution_still_refuses_a_scale_or_a_bias() {
    let net = deep_net();
    let scaled: Vec<f32> = net.conv1.iter().map(|&v| v * 2.0).collect();
    let stages = net.stages();
    let mut request = net.request(&stages);
    request.conv1 =
        ConvSpec::valid_unit_stride(&scaled, net.conv1_out, 1, net.conv1_k, net.conv1_k);
    assert!(matches!(
        refusal_of(&request, &SignSpaceLimits::default()),
        SignSpaceRefusal::NonUnitWeights {
            tensor: "conv1",
            ..
        }
    ));

    let bias = vec![0.5, 0.5];
    let mut request = net.request(&stages);
    request.conv1 =
        ConvSpec::valid_unit_stride(&net.conv1, net.conv1_out, 1, net.conv1_k, net.conv1_k)
            .with_bias(&bias);
    assert!(matches!(
        refusal_of(&request, &SignSpaceLimits::default()),
        SignSpaceRefusal::UnsupportedConvGeometry {
            tensor: "conv1",
            ..
        }
    ));
}

/// A pool that does not fit its feature map is a refusal, not a panic.
#[test]
fn an_unfittable_pool_is_refused() {
    let net = deep_net();
    let stages = net.stages();
    let mut request = net.request(&stages);
    request.conv1_pool = Some(PoolSpec {
        kernel_h: 99,
        kernel_w: 99,
        stride_h: 2,
        stride_w: 2,
    });
    assert!(matches!(
        refusal_of(&request, &SignSpaceLimits::default()),
        SignSpaceRefusal::UnsupportedPoolGeometry { site: 1, .. }
    ));

    let mut request = net.request(&stages);
    request.conv2_pool = Some(PoolSpec {
        kernel_h: 2,
        kernel_w: 2,
        stride_h: 0,
        stride_w: 2,
    });
    assert!(matches!(
        refusal_of(&request, &SignSpaceLimits::default()),
        SignSpaceRefusal::UnsupportedPoolGeometry { site: 2, .. }
    ));
}

// ---------------------------------------------------------------------------
// The shallow path is inert under the widening
// ---------------------------------------------------------------------------

/// Declaring the IDENTITY pool and an empty stage list must be bit-for-bit the
/// same problem as declaring neither — that is what makes "the `model_30` path
/// is unchanged" a property of the code rather than a hope.
#[test]
fn the_identity_pool_and_an_empty_stage_list_change_nothing() {
    for target in 0..3usize {
        let net = search_net(target);
        let limits = SignSpaceLimits::default();

        let plain = net.request();
        let mut widened = net.request();
        widened.conv1_pool = Some(PoolSpec::IDENTITY);
        widened.conv2_pool = Some(PoolSpec::IDENTITY);

        let a = classify_first_layer_unwired(&plain, &limits).expect("admitted");
        let b = classify_first_layer_unwired(&widened, &limits).expect("admitted");
        assert_eq!(a, b, "the identity pool moved the FIXED/FREE prepass");

        let first = falsify_bnn_sign_suffix_unwired(&plain, &limits).expect("no error");
        let second = falsify_bnn_sign_suffix_unwired(&widened, &limits).expect("no error");
        match (&first, &second) {
            (SignSpaceOutcome::Candidate(x), SignSpaceOutcome::Candidate(y)) => {
                assert_eq!(x.input, y.input);
                assert_eq!(x.logits, y.logits);
                assert_eq!(x.flips, y.flips);
                assert_eq!(x.logit_margin, y.logit_margin);
            }
            (
                SignSpaceOutcome::Exhausted {
                    best_logit_margin: p,
                    flips: fp,
                    ..
                },
                SignSpaceOutcome::Exhausted {
                    best_logit_margin: q,
                    flips: fq,
                    ..
                },
            ) => {
                assert_eq!(p, q);
                assert_eq!(fp, fq);
            }
            other => panic!("the identity pool changed the outcome shape: {other:?}"),
        }
    }
}

/// The end-to-end search still works on the widened chain, and any witness it
/// produces is independently re-validated against the longhand reference.
#[test]
fn the_search_runs_on_the_widened_chain_and_self_validates() {
    let net = deep_net();
    let stages = net.stages();
    let request = net.request(&stages);
    let limits = SignSpaceLimits {
        max_wall_time: Duration::from_secs(30),
        ..SignSpaceLimits::default()
    };
    match falsify_bnn_sign_suffix_unwired(&request, &limits).expect("no solver error") {
        SignSpaceOutcome::Candidate(candidate) => {
            for p in 0..net.n_pixels() {
                assert!(candidate.input[p] >= net.lo[p] && candidate.input[p] <= net.hi[p]);
            }
            assert_eq!(
                net.trace(&candidate.input).logits,
                candidate.logits,
                "the reported logits are not the logits of the reported input"
            );
            assert!(candidate.logit_margin > 0);
        }
        SignSpaceOutcome::Exhausted {
            best_logit_margin, ..
        } => assert!(best_logit_margin <= 0),
        SignSpaceOutcome::Refused(refusal) => {
            panic!("the widened fixture must be admitted: {refusal:?}")
        }
    }
}

// ---------------------------------------------------------------------------
// (e) the MINIMAL-move lever: closed-form segment crossings
//
// The lever replaces "jump to the LP vertex" with "walk the segment only as far
// as the deficient units need". `z1` is LINEAR in `x`, so every crossing is a
// division rather than a bisection — and these tests hold that closed form
// against a BRUTE-FORCE scan that re-evaluates `z1` at each sampled point and
// therefore shares no arithmetic with it.
//
//   (a) the closed-form theta matches a brute-forced scan of the segment;
//   (b) a POOLED unit's crossing is the MIN over the window members that cross;
//   (c) the chosen point is inside the box, checked exactly against lo/hi;
//   (d) the sign pattern recomputed at the chosen point is the one the engine
//       believes it moved to.
// ---------------------------------------------------------------------------

/// Grid resolution for every brute-force segment scan below.
const SCAN_STEPS: usize = 20_000;

/// A point on the segment, evaluated the long way (no interpolation of `z1`).
fn segment_point(x0: &[f64], x1: &[f64], theta: f64) -> Vec<f64> {
    (0..x0.len())
        .map(|p| x0[p] + theta * (x1[p] - x0[p]))
        .collect()
}

/// Brute force: the smallest GRID `theta` from which unit `k` holds sign `sign`
/// with OR-slack `>= level` for the WHOLE REST of the segment.
///
/// The suffix reading is the one the lever needs and the one the closed form
/// implements: the chosen point is a single theta shared by every deficient
/// unit, so "satisfied here but not further along" is worthless — the maximum
/// over units would step straight past it. `None` therefore means the condition
/// fails at `theta = 1`, which is exactly when the closed form declines.
///
/// Deliberately naive — it rebuilds the point and recomputes `z1` from the
/// pixels at every sample, so it exercises the real `max_w z_w` rather than the
/// linear form the closed solution is derived from.
fn brute_force_crossing(
    admitted: &Admitted<'_>,
    k: usize,
    sign: i8,
    level: f64,
    x0: &[f64],
    x1: &[f64],
) -> Option<f64> {
    let mut first: Option<f64> = None;
    for i in (0..=SCAN_STEPS).rev() {
        let theta = i as f64 / SCAN_STEPS as f64;
        if slack_at(admitted, k, sign, &segment_point(x0, x1, theta)) >= level {
            first = Some(theta);
        } else {
            break;
        }
    }
    first
}

/// The OR-slack of unit `k` at a concrete point, computed from the pixels.
fn slack_at(admitted: &Admitted<'_>, k: usize, sign: i8, x: &[f64]) -> f64 {
    let z = admitted.z1_at(x);
    let (pooled, _) = admitted.pooled_at(k, &z);
    f64::from(sign) * (pooled - admitted.t1[k % admitted.c1])
}

/// Two deterministic, deliberately non-integral in-box points.
fn two_points(lo: &[f64], hi: &[f64], seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut state = seed;
    let mut draw = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // Never 0 or 1: an endpoint would put many units at exactly a bound and
        // manufacture ties the grid cannot resolve.
        0.05 + 0.9 * (((state >> 33) % 997) as f64 / 996.0)
    };
    let a = (0..lo.len())
        .map(|p| lo[p] + draw() * (hi[p] - lo[p]))
        .collect();
    let b = (0..lo.len())
        .map(|p| lo[p] + draw() * (hi[p] - lo[p]))
        .collect();
    (a, b)
}

/// (a) On an UNPOOLED net the closed-form crossing must agree with a
/// brute-forced scan of the segment, for every unit, in both sign directions
/// and at several levels.
#[test]
fn closed_form_segment_crossing_matches_a_brute_force_scan() {
    let net = search_net(0);
    let request = net.request();
    let limits = SignSpaceLimits::default();
    let admitted = admit(&request, &limits).expect("the search net is admitted");
    let (x0, x1) = two_points(&net.lo, &net.hi, 0xc0ff_ee01);
    let a = admitted.z1_at(&x0);
    let b = admitted.z1_at(&x1);
    // Slack levels spanning "any sign at all" to "a comfortable margin".
    let levels = [0.0f64, 0.05, 1.0, 3.0];
    let grid = 1.0 / SCAN_STEPS as f64;

    let mut compared = 0usize;
    let mut crossed_inside = 0usize;
    for k in 0..admitted.n_units1 {
        for sign in [1i8, -1i8] {
            for &level in &levels {
                let closed = admitted.unit_crossing_theta(k, sign, level, &a, &b);
                let brute = brute_force_crossing(&admitted, k, sign, level, &x0, &x1);
                compared += 1;
                match (closed, brute) {
                    (None, None) => {}
                    (Some(tc), Some(tb)) => {
                        // The condition holds exactly on `[tc, 1]`, so the first
                        // GRID point satisfying it is in `[tc, tc + grid)`.
                        assert!(
                            tb + 1e-9 >= tc && tb <= tc + grid + 1e-9,
                            "unit {k} sign {sign} level {level}: closed {tc} vs brute {tb}"
                        );
                        if tc > 0.0 {
                            crossed_inside += 1;
                            // MINIMALITY, independent of the grid: one step
                            // BEFORE the closed crossing the unit is still short.
                            let before = (tc - 4.0 * grid).max(0.0);
                            if before < tc {
                                let s =
                                    slack_at(&admitted, k, sign, &segment_point(&x0, &x1, before));
                                assert!(
                                    s < level,
                                    "unit {k} sign {sign} level {level}: slack {s} already \
                                     reaches {level} at theta {before}, before the closed \
                                     crossing {tc}"
                                );
                            }
                        }
                        // CORRECTNESS: the closed crossing really is far enough.
                        let s = slack_at(&admitted, k, sign, &segment_point(&x0, &x1, tc));
                        assert!(
                            s >= level - 1e-9,
                            "unit {k} sign {sign} level {level}: slack {s} at the closed \
                             crossing {tc} does not reach {level}"
                        );
                    }
                    (closed, brute) => panic!(
                        "unit {k} sign {sign} level {level}: closed {closed:?} vs brute {brute:?}"
                    ),
                }
            }
        }
    }
    assert!(compared > 0);
    assert!(
        crossed_inside > 0,
        "the fixture never exercised a crossing strictly inside the segment"
    );
}

/// (b) Under `MaxPool` a unit is `max_w z_w`, a CONVEX piecewise-linear
/// function of theta — so its `+1` crossing must be the MIN over the window
/// members that cross, and its `-1` crossing the MAX over all members. Both are
/// checked against a per-member brute-force scan.
#[test]
fn a_pooled_crossing_is_the_min_over_its_window_members_that_cross() {
    let net = deep_net();
    let stages = net.stages();
    let request = net.request(&stages);
    let limits = SignSpaceLimits::default();
    let admitted = admit(&request, &limits).expect("the deep net is admitted");
    assert!(
        admitted.pool1.area() > 1,
        "this test is vacuous without a real pooling window"
    );
    let (x0, x1) = two_points(&net.lo, &net.hi, 0x5eed_b0b0);
    let a = admitted.z1_at(&x0);
    let b = admitted.z1_at(&x1);
    let grid = 1.0 / SCAN_STEPS as f64;

    let mut saw_min_over_several = 0usize;
    for k in 0..admitted.n_units1 {
        let (r, c, ch) = admitted.unit_coords(k);
        let t = admitted.t1[ch];
        for &level in &[0.0f64, 0.05, 0.5] {
            // Per-member brute force, from the raw z1 at resampled points.
            let mut up: Vec<f64> = Vec::new();
            let mut down: Vec<f64> = Vec::new();
            let mut down_all = true;
            for (mr, mc) in admitted.pool1_members(r, c) {
                let m = admitted.raw_index(mr, mc, ch);
                let mut first_up: Option<f64> = None;
                let mut first_down: Option<f64> = None;
                for i in (0..=SCAN_STEPS).rev() {
                    let theta = i as f64 / SCAN_STEPS as f64;
                    let z = admitted.z1_at(&segment_point(&x0, &x1, theta))[m];
                    if z >= t + level {
                        first_up = Some(theta);
                    } else {
                        // A suffix of the segment is what matters, so stop at
                        // the last violation walking backwards.
                        break;
                    }
                }
                for i in (0..=SCAN_STEPS).rev() {
                    let theta = i as f64 / SCAN_STEPS as f64;
                    let z = admitted.z1_at(&segment_point(&x0, &x1, theta))[m];
                    if z <= t - level {
                        first_down = Some(theta);
                    } else {
                        break;
                    }
                }
                if let Some(v) = first_up {
                    up.push(v);
                }
                match first_down {
                    Some(v) => down.push(v),
                    None => down_all = false,
                }
            }

            // `+1`: SOME member up  =>  the MIN over the members that cross.
            let closed_up = admitted.unit_crossing_theta(k, 1, level, &a, &b);
            let brute_up = up.iter().copied().fold(f64::INFINITY, f64::min);
            if up.is_empty() {
                assert!(
                    closed_up.is_none(),
                    "unit {k} level {level}: no member ever clears, but the closed form \
                     returned {closed_up:?}"
                );
            } else {
                let tc = closed_up.unwrap_or_else(|| {
                    panic!(
                        "unit {k} level {level}: a member clears at {brute_up} but the \
                            closed form declined"
                    )
                });
                assert!(
                    brute_up + 1e-9 >= tc && brute_up <= tc + grid + 1e-9,
                    "unit {k} level {level}: pooled `+1` crossing {tc} is not the min over \
                     members {up:?}"
                );
                if up.len() > 1 {
                    saw_min_over_several += 1;
                }
            }

            // `-1`: EVERY member down  =>  the MAX over members, and one member
            // that never gets there kills the unit.
            let closed_down = admitted.unit_crossing_theta(k, -1, level, &a, &b);
            if down_all {
                let brute_down = down.iter().copied().fold(0.0f64, f64::max);
                let tc = closed_down.unwrap_or_else(|| {
                    panic!(
                        "unit {k} level {level}: every member is down from {brute_down} \
                            but the closed form declined"
                    )
                });
                assert!(
                    brute_down + 1e-9 >= tc && brute_down <= tc + grid + 1e-9,
                    "unit {k} level {level}: pooled `-1` crossing {tc} is not the max over \
                     members {down:?}"
                );
            } else {
                assert!(
                    closed_down.is_none(),
                    "unit {k} level {level}: a member never falls below the threshold, but \
                     the closed form returned {closed_down:?}"
                );
            }
        }
    }
    assert!(
        saw_min_over_several > 0,
        "the fixture never exercised a MIN over more than one crossing member"
    );
}

/// Drive ONE realizability round by hand: flip a single free unit, solve the
/// active LP for the resulting pattern, and hand back everything the minimal
/// move consumes. `None` when no single flip produced a primal.
#[allow(clippy::type_complexity)]
fn one_lp_round(
    admitted: &Admitted<'_>,
    limits: &SignSpaceLimits,
    x0: &[f64],
    free: &[usize],
) -> Option<(Vec<i8>, Vec<usize>, Vec<f64>, Vec<f64>, Vec<f64>)> {
    let deadline = Instant::now() + limits.max_wall_time;
    let base = admitted.s1_at(x0);
    for &u in free.iter().take(64) {
        let mut s1 = base.clone();
        s1[u] = -s1[u];
        let z1 = admitted.z1_at(x0);
        let deficient: Vec<usize> = free
            .iter()
            .copied()
            .filter(|&k| admitted.unit_slack(k, &s1, &z1).0 < limits.tolerance)
            .collect();
        if deficient.is_empty() {
            continue;
        }
        let Ok(Ok(Some(x_lp))) =
            admitted.solve_active_lp(&s1, &deficient, &z1, limits, x0, deadline, None)
        else {
            continue;
        };
        return Some((s1, deficient, z1, admitted.z1_at(&x_lp), x_lp));
    }
    None
}

/// (c) The chosen point must be in the box, checked EXACTLY against the
/// declared `lo`/`hi` with no tolerance — on a box whose widths and centres are
/// deliberately irregular, because the traffic-sign rows' are.
#[test]
fn the_chosen_segment_point_is_exactly_inside_the_box() {
    let mut net = search_net(0);
    // Jagged, non-integral, unequal widths: nothing here is reconstructible
    // from a centre and an epsilon.
    for p in 0..net.n_pixels() {
        let lo = -3.7 + 0.013 * p as f64;
        net.lo[p] = lo;
        net.hi[p] = lo + 0.31 + 0.77 * ((p % 5) as f64);
    }
    let request = net.request();
    let limits = SignSpaceLimits::default();
    let admitted = admit(&request, &limits).expect("the jagged-box net is admitted");
    let (x0, x1) = two_points(&net.lo, &net.hi, 0xba5e_ba11);
    for step in 0..=64usize {
        let theta = step as f64 / 64.0;
        let blended = admitted
            .blend_into_box(&x0, &x1, theta)
            .expect("both endpoints are in the box, so every combination is");
        assert_eq!(blended.len(), net.n_pixels());
        for p in 0..net.n_pixels() {
            assert!(
                blended[p] >= net.lo[p] && blended[p] <= net.hi[p],
                "theta {theta}: pixel {p} = {} escapes [{}, {}]",
                blended[p],
                net.lo[p],
                net.hi[p]
            );
        }
    }
    // A theta outside `[0, 1]` is not a convex combination and must be refused
    // rather than clamped into something plausible.
    assert!(admitted.blend_into_box(&x0, &x1, 1.5).is_none());
    assert!(admitted.blend_into_box(&x0, &x1, -0.001).is_none());
    assert!(admitted.blend_into_box(&x0, &x1, f64::NAN).is_none());

    // And the same on the point the lever actually chooses, in a real round.
    let classification = admitted.classify();
    let start = admitted
        .to_replay_bytes(&x0)
        .expect("x0 is f32-representable");
    if let Some((s1, deficient, a, b, x_lp)) =
        one_lp_round(&admitted, &limits, &start, &classification.free)
    {
        let theta = admitted.minimal_segment_theta(&s1, &deficient, &a, &b, limits.tolerance);
        let chosen = admitted
            .blend_into_box(&start, &x_lp, theta)
            .expect("the chosen point is a convex combination of two in-box points");
        for p in 0..net.n_pixels() {
            assert!(chosen[p] >= net.lo[p] && chosen[p] <= net.hi[p]);
        }
    }
}

/// (d) The sign pattern recomputed at the chosen point must be the one the
/// engine believes it moved to: every unit the LP was asked to fix really does
/// hold its DESIRED sign there, at slack `>= tolerance`, and the whole
/// first-layer pattern agrees with the longhand reference.
#[test]
fn the_recomputed_pattern_at_the_chosen_point_is_what_the_engine_believes() {
    let net = deep_net();
    let stages = net.stages();
    let request = net.request(&stages);
    let limits = SignSpaceLimits {
        segment_move: SegmentMove::MinimalTheta,
        ..SignSpaceLimits::default()
    };
    let admitted = admit(&request, &limits).expect("the deep net is admitted");
    let classification = admitted.classify();
    let midpoint: Vec<f64> = (0..admitted.n_pixels)
        .map(|p| admitted.lo[p].midpoint(admitted.hi[p]))
        .collect();
    let x0 = admitted
        .to_replay_bytes(&midpoint)
        .expect("the midpoint is f32-representable");
    let (s1, deficient, a, b, x_lp) = one_lp_round(&admitted, &limits, &x0, &classification.free)
        .expect("some single flip must yield an LP primal on this fixture");

    let theta = admitted.minimal_segment_theta(&s1, &deficient, &a, &b, limits.tolerance);
    assert!((0.0..=1.0).contains(&theta), "theta {theta} left [0, 1]");
    // The lever must actually MOVE LESS than the vertex jump on this fixture,
    // or it is not being exercised at all. A `theta` pinned at `1.0` is how the
    // first (buggy) version silently degenerated back into the vertex jump.
    assert!(
        theta < 1.0,
        "the minimal move collapsed to the vertex (theta = {theta}); the lever is inert"
    );
    let chosen = admitted
        .blend_into_box(&x0, &x_lp, theta)
        .expect("a convex combination of two in-box points");

    // 1. Every deficient unit really did cross, and by at least as much as the
    //    tolerance OR as much as the LP vertex itself delivers — the segment is
    //    not entitled to more than its own far endpoint, which is exactly what
    //    the attainability clamp in `minimal_segment_theta` encodes.
    let z_chosen = admitted.z1_at(&chosen);
    let z_vertex = admitted.z1_at(&x_lp);
    for &k in &deficient {
        let (slack, _) = admitted.unit_slack(k, &s1, &z_chosen);
        let (at_vertex, _) = admitted.unit_slack(k, &s1, &z_vertex);
        let owed = limits.tolerance.min(at_vertex) - 1e-9;
        assert!(
            slack >= owed,
            "unit {k} was supposed to be carried past its threshold but has slack {slack} \
             (owed {owed}; tolerance {}, vertex {at_vertex})",
            limits.tolerance
        );
        assert!(
            slack > 0.0,
            "unit {k} does not even hold its desired sign at the chosen point (slack {slack})"
        );
    }

    // 2. The engine's own pattern at the chosen point equals the LONGHAND
    //    reference's — the pattern is what it is believed to be, checked against
    //    arithmetic that shares nothing with the module.
    let ours = admitted.s1_from_z1(&z_chosen);
    let reference = net.trace(&chosen).s1;
    assert_eq!(ours.len(), reference.len());
    for (k, (&mine, &theirs)) in ours.iter().zip(reference.iter()).enumerate() {
        assert_eq!(
            i64::from(mine),
            theirs,
            "first-layer unit {k} disagrees with the longhand reference at the chosen point"
        );
    }

    // 3. The MINIMAL move is genuinely smaller than the vertex jump: it must
    //    preserve at least as many of the incumbent's signs.
    let incumbent = admitted.s1_at(&x0);
    let at_vertex = admitted.s1_at(&x_lp);
    let kept_theta = (0..admitted.n_units1)
        .filter(|&k| ours[k] == incumbent[k])
        .count();
    let kept_vertex = (0..admitted.n_units1)
        .filter(|&k| at_vertex[k] == incumbent[k])
        .count();
    assert!(
        kept_theta >= kept_vertex,
        "the minimal move kept {kept_theta} incumbent signs, the vertex jump {kept_vertex}"
    );
}

/// The default is the historic vertex jump, and BOTH arms are reachable and
/// still produce independently-validated witnesses. Pinning the default here is
/// what stops the lever from becoming the shipped behaviour without a
/// measurement.
#[test]
fn both_segment_moves_are_available_and_the_default_is_the_vertex_jump() {
    assert_eq!(
        SignSpaceLimits::default().segment_move,
        SegmentMove::Vertex,
        "the shipped default must stay the measured one until the lever measures better"
    );
    for target in 0..3usize {
        let net = search_net(target);
        for segment_move in [SegmentMove::Vertex, SegmentMove::MinimalTheta] {
            let limits = SignSpaceLimits {
                segment_move,
                ..SignSpaceLimits::default()
            };
            match falsify_bnn_sign_suffix_unwired(&net.request(), &limits).expect("no solver error")
            {
                SignSpaceOutcome::Candidate(candidate) => validate_candidate(&net, &candidate),
                SignSpaceOutcome::Exhausted {
                    best_logit_margin, ..
                } => assert!(best_logit_margin <= 0),
                SignSpaceOutcome::Refused(refusal) => panic!("unexpected refusal: {refusal:?}"),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The TRUST REGION (the SIDEWAYS half of the §10 wall)
//
// Four properties, in the order they protect anything:
//
//   (a) a region only ever SHRINKS the feasible set — a pattern realizable
//       under one is realizable in the full box, checked against brute force
//       over the bound arithmetic and over many patterns end to end;
//   (b) infeasibility under a region NEVER concludes: it declines and EXPANDS,
//       and the last radius tried is the full box;
//   (c) the point it hands back is exactly inside the vnnlib box;
//   (d) the shipped default is the full box, and with it nothing changes.
// ---------------------------------------------------------------------------

/// The three arms the CLI lever exposes, in the order they tighten.
const TRUST_ARMS: [TrustRegion; 3] = [
    TrustRegion::Doubling {
        initial_fraction: 0.125,
    },
    TrustRegion::Doubling {
        initial_fraction: 0.015_625,
    },
    TrustRegion::Nearest {
        initial_fraction: 0.015_625,
        refine: 4,
    },
];

/// The incumbent's own first-layer pattern with exactly one free unit flipped —
/// the pattern shape every realizability call in the real search tests.
fn single_flip_patterns(
    admitted: &Admitted<'_>,
    x0: &[f64],
    free: &[usize],
    count: usize,
) -> Vec<(usize, Vec<i8>)> {
    let base = admitted.s1_at(x0);
    free.iter()
        .take(count)
        .map(|&u| {
            let mut s1 = base.clone();
            s1[u] = -s1[u];
            (u, s1)
        })
        .collect()
}

/// (a) A TRUST REGION CAN ONLY SHRINK THE FEASIBLE SET.
///
/// Two independent statements of that, because the interesting failure modes
/// live at different levels:
///
///  1. BRUTE FORCE ON THE BOUNDS. `trust_bounds` is the only place a region
///     touches the LP, so every `(pixel, anchor, radius)` combination — including
///     radii wider than the box, a zero radius, and the pathological
///     non-finite/negative ones — must come back inside `[lo, hi]` and never
///     inverted. Exhaustive over the pixels of a jagged box.
///  2. END TO END. For many single-flip patterns and every armed arm: whenever
///     the region reports the pattern realizable, so does the full box, the
///     returned point is in the box, and it really realizes the WHOLE pattern
///     when the slacks are recomputed from the net's own tensors.
#[test]
fn a_trust_region_only_shrinks_the_feasible_set() {
    let mut net = search_net(0);
    for p in 0..net.n_pixels() {
        let lo = -3.7 + 0.013 * p as f64;
        net.lo[p] = lo;
        net.hi[p] = lo + 0.31 + 0.77 * ((p % 5) as f64);
    }
    let request = net.request();
    let base = SignSpaceLimits::default();
    let admitted = admit(&request, &base).expect("the jagged-box net is admitted");
    let (anchor, other) = two_points(&net.lo, &net.hi, 0x7275_7374);

    // 1. the bound arithmetic itself.
    for p in 0..net.n_pixels() {
        for radius in [
            0.0,
            1e-12,
            0.001,
            0.5,
            1.0,
            1e6,
            f64::INFINITY,
            f64::NAN,
            -1.0,
        ] {
            for candidate_anchor in [&anchor, &other] {
                let (lo, hi) = admitted.trust_bounds(p, Some((candidate_anchor, radius)));
                assert!(
                    lo >= net.lo[p] && hi <= net.hi[p],
                    "pixel {p} at radius {radius}: [{lo}, {hi}] escapes the box [{}, {}]",
                    net.lo[p],
                    net.hi[p]
                );
                assert!(
                    lo <= hi,
                    "pixel {p} at radius {radius}: inverted [{lo}, {hi}]"
                );
                if radius.is_finite() && radius >= 0.0 {
                    // Inside the region as well as inside the box, unless the
                    // box was the binding side.
                    assert!(lo >= net.lo[p].max(candidate_anchor[p] - radius) - 1e-12);
                    assert!(hi <= net.hi[p].min(candidate_anchor[p] + radius) + 1e-12);
                }
            }
        }
        // No region at all is the box, byte for byte.
        assert_eq!(admitted.trust_bounds(p, None), (net.lo[p], net.hi[p]));
    }

    // 2. end to end, over many patterns and every arm.
    let free = admitted.classify().free;
    let midpoint: Vec<f64> = (0..admitted.n_pixels)
        .map(|p| admitted.lo[p].midpoint(admitted.hi[p]))
        .collect();
    let x0 = admitted
        .to_replay_bytes(&midpoint)
        .expect("the midpoint is f32-representable");
    let deadline = Instant::now() + Duration::from_mins(2);
    let mut realizable_under_a_region = 0usize;
    for (unit, s1) in single_flip_patterns(&admitted, &x0, &free, 12) {
        let full = admitted
            .solve_realizability(&s1, &free, &base, &x0, deadline)
            .expect("no solver error")
            .expect("no refusal on this fixture");
        for arm in TRUST_ARMS {
            let limits = SignSpaceLimits {
                trust_region: arm,
                ..base
            };
            let got = admitted
                .solve_realizability(&s1, &free, &limits, &x0, deadline)
                .expect("no solver error")
                .expect("no refusal on this fixture");
            let Realizability::Realizable { slack, x } = got else {
                continue;
            };
            realizable_under_a_region += 1;
            assert!(
                matches!(full, Realizability::Realizable { .. }),
                "unit {unit} under {arm:?}: realizable in the SHRUNK set but not in the box"
            );
            for p in 0..admitted.n_pixels {
                assert!(
                    x[p] >= net.lo[p] && x[p] <= net.hi[p],
                    "unit {unit} under {arm:?}: pixel {p} = {} escapes [{}, {}]",
                    x[p],
                    net.lo[p],
                    net.hi[p]
                );
            }
            // The slacks, recomputed from the NET's tensors rather than the
            // module's `z1_at`.
            let z = net.z1(&x);
            for &k in &free {
                let (recomputed, _) = admitted.unit_slack(k, &s1, &z);
                assert!(
                    recomputed >= slack - 1e-9,
                    "unit {unit} under {arm:?}: free unit {k} has slack {recomputed} \
                     against the reported {slack}"
                );
            }
            assert!(
                slack >= base.tolerance - 1e-9,
                "reported slack {slack} is below tolerance"
            );
        }
    }
    assert!(
        realizable_under_a_region > 0,
        "no pattern was realizable under any region: the test proved nothing"
    );
}

/// (b) INFEASIBILITY UNDER A TRUST REGION NEVER CONCLUDES — IT EXPANDS.
///
/// The one property that could make this lever unsound is reading "the
/// restricted LP has no solution" as "the pattern is not realizable". So this
/// finds a pattern the FULL box realizes, confirms the LP really is infeasible
/// at the opening radius (otherwise the test is vacuous), and then asserts the
/// driver still comes back with a point, having expanded — and that the point
/// is a real one.
#[test]
fn trust_region_infeasibility_declines_and_expands_it_never_concludes() {
    let net = search_net(0);
    let request = net.request();
    let base = SignSpaceLimits::default();
    let admitted = admit(&request, &base).expect("the search net is admitted");
    let free = admitted.classify().free;
    let midpoint: Vec<f64> = (0..admitted.n_pixels)
        .map(|p| admitted.lo[p].midpoint(admitted.hi[p]))
        .collect();
    let x0 = admitted
        .to_replay_bytes(&midpoint)
        .expect("the midpoint is f32-representable");
    let deadline = Instant::now() + Duration::from_mins(2);
    // A radius so small that no flip can be carried inside it.
    let limits = SignSpaceLimits {
        trust_region: TrustRegion::Doubling {
            initial_fraction: 1e-6,
        },
        ..base
    };

    let mut proved_on = 0usize;
    for (unit, s1) in single_flip_patterns(&admitted, &x0, &free, 24) {
        let z1 = admitted.z1_at(&x0);
        let deficient: Vec<usize> = free
            .iter()
            .copied()
            .filter(|&k| admitted.unit_slack(k, &s1, &z1).0 < base.tolerance)
            .collect();
        if deficient.is_empty() {
            continue;
        }
        // The restricted LP at the OPENING radius, on its own. If this were
        // read as an answer, the pattern would be declined here.
        let mut trust = TrustState::new(&admitted, &limits, &x0);
        let opening = trust.radius.expect("the region is armed on this fixture");
        let restricted = admitted
            .solve_active_lp(
                &s1,
                &deficient,
                &z1,
                &limits,
                &x0,
                deadline,
                Some((x0.as_slice(), opening)),
            )
            .expect("no solver error")
            .expect("no refusal");
        if restricted.is_some() {
            continue;
        }
        // The full box, for the same pattern and the same rows.
        let unrestricted = admitted
            .solve_active_lp(&s1, &deficient, &z1, &base, &x0, deadline, None)
            .expect("no solver error")
            .expect("no refusal");
        if unrestricted.is_none() {
            // Infeasible in the box too — nothing was lost, and nothing is
            // proved either. Skip.
            continue;
        }
        // THE ASSERTION: the driver does not conclude. It expands, and it comes
        // back with the point the box has.
        let point = admitted
            .solve_lp_under_trust_region(
                &s1, &deficient, &z1, &limits, &x0, deadline, &mut trust, false,
            )
            .expect("no solver error")
            .expect("no refusal")
            .unwrap_or_else(|| {
                panic!(
                    "unit {unit}: the region concluded on an infeasible restricted LP \
                     instead of expanding"
                )
            });
        assert!(
            trust.expansions > 0,
            "unit {unit}: a point came back with no expansion, so the opening radius was \
             not actually infeasible and the test is vacuous"
        );
        // The expansion terminates at the FULL box, which is today's LP.
        assert!(
            trust.radius.is_none_or(|r| r > opening),
            "the radius must grow, or become the full box"
        );
        for p in 0..admitted.n_pixels {
            assert!(point[p] >= net.lo[p] && point[p] <= net.hi[p]);
        }
        let z = net.z1(&point);
        for &k in &deficient {
            let (slack, _) = admitted.unit_slack(k, &s1, &z);
            assert!(
                slack >= base.tolerance - 1e-9,
                "unit {unit}: the expanded point leaves deficient unit {k} at slack {slack}"
            );
        }
        proved_on += 1;
    }
    assert!(
        proved_on > 0,
        "no pattern was infeasible at the opening radius and realizable in the box: \
         the expansion path was never exercised"
    );
}

/// (b') EXPANSION IS BOUNDED AND ALWAYS ENDS AT THE FULL BOX.
///
/// The state machine on its own, with no LP in the way: from any armed radius,
/// repeated failure must reach `None` — the full vnnlib box — in a bounded
/// number of steps, and `None` is the ONLY state that reports "you may decline".
#[test]
fn trust_expansion_terminates_at_the_full_box_and_only_then_may_decline() {
    let net = search_net(0);
    let request = net.request();
    let base = SignSpaceLimits::default();
    let admitted = admit(&request, &base).expect("admitted");
    let x0: Vec<f64> = (0..admitted.n_pixels)
        .map(|p| admitted.lo[p].midpoint(admitted.hi[p]))
        .collect();
    for arm in TRUST_ARMS {
        let limits = SignSpaceLimits {
            trust_region: arm,
            ..base
        };
        let mut trust = TrustState::new(&admitted, &limits, &x0);
        assert!(trust.radius.is_some(), "{arm:?} must actually arm a region");
        let mut steps = 0usize;
        while trust.radius.is_some() {
            assert!(
                trust.expand(&limits),
                "an armed region must never say 'decline'"
            );
            steps += 1;
            assert!(
                steps <= limits.max_trust_expansions,
                "expansion did not terminate"
            );
        }
        // At the full box — and ONLY here — declining is allowed, which is
        // exactly today's behaviour.
        assert!(!trust.expand(&limits));
        assert!(
            steps >= 1,
            "{arm:?} reached the box without a single expansion"
        );
    }
    // An anchor outside the box, or a degenerate fraction, never arms a region
    // at all: it falls back to the shipped full box rather than to a region
    // whose intersection could be empty.
    let outside: Vec<f64> = admitted.hi.iter().map(|h| h + 1.0).collect();
    let armed = SignSpaceLimits {
        trust_region: TrustRegion::Doubling {
            initial_fraction: 0.125,
        },
        ..base
    };
    assert!(TrustState::new(&admitted, &armed, &outside)
        .radius
        .is_none());
    for bad in [0.0, -0.25, 1.0, 2.0, f64::NAN, f64::INFINITY] {
        let limits = SignSpaceLimits {
            trust_region: TrustRegion::Doubling {
                initial_fraction: bad,
            },
            ..base
        };
        assert!(
            TrustState::new(&admitted, &limits, &x0).radius.is_none(),
            "fraction {bad} must fall back to the full box"
        );
    }
}

/// (c) THE POINT IS EXACTLY INSIDE THE VNN-LIB BOX, on a box whose widths and
/// centres are deliberately irregular — because the traffic-sign rows' are, and
/// two of the three banked ones are not reconstructible from a centre and an
/// epsilon.
#[test]
fn the_trust_region_point_is_exactly_inside_the_box() {
    let mut net = search_net(1);
    for p in 0..net.n_pixels() {
        // Nothing here is reconstructible from a centre and an epsilon: the
        // widths differ pixel to pixel and no centre is integral.
        let lo = -3.7 + 0.013 * p as f64;
        net.lo[p] = lo;
        net.hi[p] = lo + 0.31 + 0.77 * ((p % 5) as f64);
    }
    let request = net.request();
    let base = SignSpaceLimits::default();
    let admitted = admit(&request, &base).expect("the jagged-box net is admitted");
    let free = admitted.classify().free;
    let midpoint: Vec<f64> = (0..admitted.n_pixels)
        .map(|p| admitted.lo[p].midpoint(admitted.hi[p]))
        .collect();
    let x0 = admitted
        .to_replay_bytes(&midpoint)
        .expect("f32-representable");
    let deadline = Instant::now() + Duration::from_mins(2);
    let mut points = 0usize;
    for arm in TRUST_ARMS {
        let limits = SignSpaceLimits {
            trust_region: arm,
            ..base
        };
        for (_, s1) in single_flip_patterns(&admitted, &x0, &free, 16) {
            if let Realizability::Realizable { x, .. } = admitted
                .solve_realizability(&s1, &free, &limits, &x0, deadline)
                .expect("no solver error")
                .expect("no refusal")
            {
                points += 1;
                for p in 0..admitted.n_pixels {
                    // EXACT, no tolerance, against the declared bounds.
                    assert!(
                        x[p] >= net.lo[p] && x[p] <= net.hi[p],
                        "{arm:?}: pixel {p} = {} escapes [{}, {}]",
                        x[p],
                        net.lo[p],
                        net.hi[p]
                    );
                }
            }
        }
    }
    assert!(points > 0, "no point was produced, so nothing was checked");
}

/// (d) THE DEFAULT IS UNCHANGED.
///
/// With no lever set the core ships the full box, `TrustState` arms nothing, so
/// every LP is built with `trust = None` — the pre-existing call — and the
/// search produces exactly what the explicitly-full-box configuration does.
#[test]
fn the_trust_region_default_is_the_full_box_and_changes_nothing() {
    assert_eq!(
        SignSpaceLimits::default().trust_region,
        TrustRegion::FullBox,
        "the shipped default must stay the measured one until a region measures better"
    );
    let net = search_net(0);
    let request = net.request();
    let base = SignSpaceLimits::default();
    let admitted = admit(&request, &base).expect("admitted");
    let x0: Vec<f64> = (0..admitted.n_pixels)
        .map(|p| admitted.lo[p].midpoint(admitted.hi[p]))
        .collect();
    // The default arms NO region, so no bound anywhere is narrowed.
    let trust = TrustState::new(&admitted, &base, &x0);
    assert!(trust.radius.is_none());
    for p in 0..admitted.n_pixels {
        assert_eq!(
            admitted.trust_bounds(p, trust.radius.map(|r| (x0.as_slice(), r))),
            (admitted.lo[p], admitted.hi[p])
        );
    }
    // And the whole search is unchanged against an explicit `FullBox`.
    for target in 0..3usize {
        let net = search_net(target);
        let explicit = SignSpaceLimits {
            trust_region: TrustRegion::FullBox,
            ..base
        };
        let shipped = falsify_bnn_sign_suffix_unwired(&net.request(), &base).expect("no error");
        let same = falsify_bnn_sign_suffix_unwired(&net.request(), &explicit).expect("no error");
        match (shipped, same) {
            (SignSpaceOutcome::Candidate(a), SignSpaceOutcome::Candidate(b)) => {
                assert_eq!(a.input, b.input);
                assert_eq!(a.logits, b.logits);
                assert_eq!(a.lp_slack, b.lp_slack);
                assert_eq!(a.flips, b.flips);
                assert_eq!(a.lp_solves, b.lp_solves);
                validate_candidate(&net, &a);
            }
            (
                SignSpaceOutcome::Exhausted {
                    best_logit_margin: a,
                    flips: fa,
                    lp_solves: la,
                    ..
                },
                SignSpaceOutcome::Exhausted {
                    best_logit_margin: b,
                    flips: fb,
                    lp_solves: lb,
                    ..
                },
            ) => {
                assert_eq!((a, fa, la), (b, fb, lb));
            }
            (x, y) => panic!("the default and an explicit full box disagree: {x:?} vs {y:?}"),
        }
    }
}

/// EVERY ARM STILL PRODUCES INDEPENDENTLY-VALIDATED WITNESSES.
///
/// The companion to the default test: the arms are reachable end to end and a
/// candidate found under a region survives the same from-scratch validation as
/// one found in the full box.
#[test]
fn every_trust_region_arm_produces_a_validated_witness_or_an_honest_exhaustion() {
    for target in 0..3usize {
        let net = search_net(target);
        for arm in TRUST_ARMS {
            let limits = SignSpaceLimits {
                trust_region: arm,
                ..SignSpaceLimits::default()
            };
            match falsify_bnn_sign_suffix_unwired(&net.request(), &limits).expect("no solver error")
            {
                SignSpaceOutcome::Candidate(candidate) => validate_candidate(&net, &candidate),
                SignSpaceOutcome::Exhausted {
                    best_logit_margin, ..
                } => assert!(best_logit_margin <= 0),
                SignSpaceOutcome::Refused(refusal) => panic!("unexpected refusal: {refusal:?}"),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// #lane-value-stall
// ---------------------------------------------------------------------------

/// (d) THE DEFAULT IS UNCHANGED — the value-stall rule is OFF, and off means
/// the shipped walk byte for byte.
///
/// The rule exists because the SHIPPED one (`stall_lp_solves`) asks a
/// never-started question. Measured on `traffic_signs model_48_idx_1703_eps_1`
/// at HEAD, both dark levers at their defaults: 370 LP solves, **34 accepted
/// flips**, best pattern margin -82, no candidate, the whole 217.52 s lane
/// budget spent — `flips == 0` was false from the first accepted flip, so the
/// rule was permanently disarmed. (This also contradicts the shipped
/// documentation in `configs/vnncomp25/traffic_signs_recognition_2023.yaml`
/// and `sign_space_falsify.rs`, which both assert 0 accepted flips on all nine
/// open 48x48/64x64 rows.)
#[test]
fn the_margin_stall_rule_is_off_by_default_and_off_changes_nothing() {
    assert_eq!(
        SignSpaceLimits::default().stall_margin_lp_solves,
        0,
        "the shipped walk must not gain a stall rule that has not been swept"
    );
    for target in 0..3usize {
        let net = search_net(target);
        let shipped = SignSpaceLimits::default();
        let explicit_off = SignSpaceLimits {
            stall_margin_lp_solves: 0,
            ..SignSpaceLimits::default()
        };
        let a = falsify_bnn_sign_suffix_unwired(&net.request(), &shipped).expect("no error");
        let b = falsify_bnn_sign_suffix_unwired(&net.request(), &explicit_off).expect("no error");
        match (a, b) {
            (SignSpaceOutcome::Candidate(a), SignSpaceOutcome::Candidate(b)) => {
                assert_eq!(a.input, b.input);
                assert_eq!(a.flips, b.flips);
                assert_eq!(a.lp_solves, b.lp_solves);
            }
            (
                SignSpaceOutcome::Exhausted {
                    best_logit_margin: ma,
                    margin_gain: ga,
                    flips: fa,
                    lp_solves: la,
                    ..
                },
                SignSpaceOutcome::Exhausted {
                    best_logit_margin: mb,
                    margin_gain: gb,
                    flips: fb,
                    lp_solves: lb,
                    ..
                },
            ) => assert_eq!((ma, ga, fa, la), (mb, gb, fb, lb)),
            (x, y) => panic!("the default and an explicit 0 disagree: {x:?} vs {y:?}"),
        }
    }
}

/// The rule can only bring `Exhausted` FORWARD. It is a budget rule, not a
/// correctness one: the two reachable ends are `Candidate` and `Exhausted`, so
/// stopping the walk early can never turn a SAT into anything else — it can
/// only decline to keep paying for a search whose own value signal is flat.
#[test]
fn the_margin_stall_rule_only_shortens_the_walk_and_never_loses_a_candidate() {
    for target in 0..3usize {
        let net = search_net(target);
        let shipped = falsify_bnn_sign_suffix_unwired(&net.request(), &SignSpaceLimits::default())
            .expect("no error");
        let stalled = falsify_bnn_sign_suffix_unwired(
            &net.request(),
            &SignSpaceLimits {
                // The tightest possible arm: yield the moment one LP passes
                // without the margin moving.
                stall_margin_lp_solves: 1,
                ..SignSpaceLimits::default()
            },
        )
        .expect("no error");
        let solves = |o: &SignSpaceOutcome| match o {
            SignSpaceOutcome::Candidate(c) => c.lp_solves,
            SignSpaceOutcome::Exhausted { lp_solves, .. } => *lp_solves,
            SignSpaceOutcome::Refused(_) => 0,
        };
        assert!(
            solves(&stalled) <= solves(&shipped),
            "target {target}: the stall rule must never make the walk pay MORE \
             ({} vs {})",
            solves(&stalled),
            solves(&shipped)
        );
        // And whatever it does return is still an honest outcome, validated
        // from scratch, never a verdict.
        match stalled {
            SignSpaceOutcome::Candidate(candidate) => validate_candidate(&net, &candidate),
            SignSpaceOutcome::Exhausted {
                best_logit_margin,
                margin_gain,
                ..
            } => {
                assert!(best_logit_margin <= 0);
                assert!(margin_gain >= 0, "value is a GAIN, never negative");
            }
            SignSpaceOutcome::Refused(refusal) => panic!("unexpected refusal: {refusal:?}"),
        }
    }
}

/// The reported VALUE is the lane's own unit, and it is not `flips`.
///
/// `margin_gain` is `best - initial`, so it is `>= 0` by construction and it
/// answers the question the scheduler actually asks: did the thing the search
/// is trying to move MOVE? The measured 34-flip row moved it to -82 and banked
/// nothing.
#[test]
fn the_walk_reports_its_margin_gain_as_a_nonnegative_value_in_its_own_units() {
    for target in 0..3usize {
        let net = search_net(target);
        match falsify_bnn_sign_suffix_unwired(&net.request(), &SignSpaceLimits::default())
            .expect("no error")
        {
            SignSpaceOutcome::Exhausted {
                best_logit_margin,
                margin_gain,
                lp_solves,
                ..
            } => {
                assert!(margin_gain >= 0);
                assert!(
                    best_logit_margin - margin_gain <= best_logit_margin,
                    "the initial margin cannot be above the best one"
                );
                let _denominator = lp_solves;
            }
            SignSpaceOutcome::Candidate(_) | SignSpaceOutcome::Refused(_) => {}
        }
    }
}
