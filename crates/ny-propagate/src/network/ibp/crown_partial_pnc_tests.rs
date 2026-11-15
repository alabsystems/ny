// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #crown-partial-pnc — patch-native terminal concretization in the sequential
//! CROWN-IBP partial walk (`crown_partial.rs`).
//!
//! Three obligations, each its own test, each failing LOUDLY as VOID when the
//! route under test never fired (a test that passes because the new path was
//! silently skipped would convert "untested" into "verified"):
//!
//! 1. SOUNDNESS — on a small Conv(stride 1, pad 1) -> ReLU -> Conv(stride 2,
//!    pad 1) net, the RAW patch-native walk result (no IBP intersection) and
//!    the full collector's published bounds enclose the f64 forward at 256
//!    sampled inputs (both extreme corners, 62 random vertices, 192 interior
//!    points), with at most `TOL = 1e-9` slack in the unsound direction.
//! 2. ENGAGEMENT + BIT-IDENTITY — on both conv targets (target 2's terminal
//!    carries a composed `coeff_err`) the patch-native arm engages and
//!    publishes bounds bit-for-bit equal to the historical densify route.
//! 3. SCOPE — the pure scope predicate refuses every out-of-scope shape
//!    (route disabled, sparse Patches seed, unstable-rows Dense seed, Dense
//!    carrier, `unstable_idx` layout, Anchored geometry), and a live
//!    Dense-carrier walk (Linear target) never engages and is bit-identical
//!    between arms.

use super::*;
use crate::bounds::patches::{PatchesData, UnstableIdx};
use crate::layers::{Conv2dLayer, FlattenLayer, LinearLayer, ReLULayer};
use crate::network::ibp::NetworkIbpExt;
use crate::types::BoundsProvenance;
use crate::Network;
use ndarray::{ArrayD, IxDyn};
use std::time::Duration;

const IN_C: usize = 2;
const IN_H: usize = 8;
const IN_W: usize = 8;
/// Conv0 output channels (8x8 output: stride 1, pad 1).
const C1: usize = 3;
/// Conv2 output channels (4x4 output: stride 2, pad 1).
const C2: usize = 4;
/// Input box half-width around each center; wide enough that far fewer than
/// 90% of either conv's outputs are ReLU-stable, so no sparse seed is chosen.
const RADIUS: f32 = 0.25;
const SAMPLES: usize = 256;
const VERTEX_SAMPLES: usize = 64;
/// Maximum slack tolerated in the unsound direction.
const TOL: f64 = 1e-9;
const SEED: u64 = 0x9e37_c0de_2026_0924;

/// SplitMix64: deterministic, dependency-free.
struct SplitMix(u64);

impl SplitMix {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)` on a 2^-24 grid (exact in f32 and f64).
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 40) as f64 / (1u64 << 24) as f64
    }

    fn uniform_f32(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit() as f32
    }
}

fn live_deadline() -> Option<Instant> {
    // A live (unexpired) deadline is the scored configuration. With the
    // crown-partial finite-expiry lever at its default (unarmed) it declines the
    // optional GPU route and the sparse-seed discovery, and the shipped-armed
    // patches finite-expiry authority keeps the native Patches conv route, so
    // the walk is exactly the CPU Patches loop under test.
    Some(Instant::now() + Duration::from_secs(600))
}

fn conv3x3(rng: &mut SplitMix, out_c: usize, in_c: usize, stride: usize) -> Conv2dLayer {
    let kernel: Vec<f32> = (0..out_c * in_c * 9)
        .map(|_| rng.uniform_f32(-0.5, 0.5))
        .collect();
    let bias: Vec<f32> = (0..out_c).map(|_| rng.uniform_f32(-0.1, 0.1)).collect();
    Conv2dLayer::with_input_shape(
        ArrayD::from_shape_vec(IxDyn(&[out_c, in_c, 3, 3]), kernel).expect("kernel shape"),
        Some(Array1::from(bias)),
        (stride, stride),
        (1, 1),
        IN_H,
        IN_W,
    )
    .expect("conv fixture")
}

fn input_box(rng: &mut SplitMix) -> BoundedTensor {
    let n = IN_C * IN_H * IN_W;
    let centers: Vec<f32> = (0..n).map(|_| rng.uniform_f32(-1.0, 1.0)).collect();
    let lower: Vec<f32> = centers.iter().map(|c| c - RADIUS).collect();
    let upper: Vec<f32> = centers.iter().map(|c| c + RADIUS).collect();
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[IN_C, IN_H, IN_W]), lower).expect("lower shape"),
        ArrayD::from_shape_vec(IxDyn(&[IN_C, IN_H, IN_W]), upper).expect("upper shape"),
    )
    .expect("input box")
}

struct ConvFixture {
    network: Network,
    input: BoundedTensor,
    conv0: Conv2dLayer,
    conv2: Conv2dLayer,
}

/// Conv(2->3, 3x3, s1, p1) -> ReLU -> Conv(3->4, 3x3, s2, p1) on a 2x8x8 box.
/// The conv2 walk composes through padded incoming patches, so its terminal
/// is a composed 6D carrier that carries `coeff_err`.
fn conv_relu_conv() -> ConvFixture {
    let mut rng = SplitMix(SEED);
    let conv0 = conv3x3(&mut rng, C1, IN_C, 1);
    let conv2 = conv3x3(&mut rng, C2, C1, 2);
    let mut network = Network::new();
    network.add_layer(Layer::Conv2d(conv0.clone()));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Conv2d(conv2.clone()));
    let input = input_box(&mut rng);
    ConvFixture {
        network,
        input,
        conv0,
        conv2,
    }
}

/// Exact-as-f64 reference conv (cross-correlation, zero padding).
fn conv_forward_f64(layer: &Conv2dLayer, x: &[f64], in_c: usize) -> Vec<f64> {
    let shape = layer.kernel.shape().to_vec();
    let (out_c, kh, kw) = (shape[0], shape[2], shape[3]);
    assert_eq!(shape[1], in_c, "reference conv: kernel in_c");
    let (sh, sw) = layer.stride;
    let (ph, pw) = layer.padding;
    let out_h = (IN_H + 2 * ph - kh) / sh + 1;
    let out_w = (IN_W + 2 * pw - kw) / sw + 1;
    let mut y = vec![0.0f64; out_c * out_h * out_w];
    for oc in 0..out_c {
        for oh in 0..out_h {
            for ow in 0..out_w {
                let mut acc = layer.bias.as_ref().map_or(0.0, |b| f64::from(b[oc]));
                for ic in 0..in_c {
                    for ki in 0..kh {
                        for kj in 0..kw {
                            let ih = (oh * sh + ki) as isize - ph as isize;
                            let iw = (ow * sw + kj) as isize - pw as isize;
                            if ih < 0 || iw < 0 || ih >= IN_H as isize || iw >= IN_W as isize {
                                continue;
                            }
                            let (ih, iw) = (ih as usize, iw as usize);
                            acc += f64::from(layer.kernel[[oc, ic, ki, kj]])
                                * x[ic * IN_H * IN_W + ih * IN_W + iw];
                        }
                    }
                }
                y[oc * out_h * out_w + oh * out_w + ow] = acc;
            }
        }
    }
    y
}

/// f64 forward of every layer output: `[conv0, relu, conv2]`.
fn forward_f64(fx: &ConvFixture, x: &[f64]) -> [Vec<f64>; 3] {
    let y0 = conv_forward_f64(&fx.conv0, x, IN_C);
    let a1: Vec<f64> = y0.iter().map(|&v| v.max(0.0)).collect();
    let y2 = conv_forward_f64(&fx.conv2, &a1, C1);
    [y0, a1, y2]
}

fn flat_f32(values: &ArrayD<f32>) -> Vec<f32> {
    values.iter().copied().collect()
}

/// Both extreme corners, random vertices, then interior points.
fn sample_inputs(rng: &mut SplitMix, lower: &[f32], upper: &[f32]) -> Vec<Vec<f64>> {
    let mut points: Vec<Vec<f64>> = Vec::with_capacity(SAMPLES);
    points.push(lower.iter().map(|&l| f64::from(l)).collect());
    points.push(upper.iter().map(|&u| f64::from(u)).collect());
    while points.len() < VERTEX_SAMPLES {
        let vertex = lower
            .iter()
            .zip(upper)
            .map(|(&l, &u)| {
                if rng.next_u64() & 1 == 0 {
                    f64::from(l)
                } else {
                    f64::from(u)
                }
            })
            .collect();
        points.push(vertex);
    }
    while points.len() < SAMPLES {
        let interior = lower
            .iter()
            .zip(upper)
            .map(|(&l, &u)| {
                let (l, u) = (f64::from(l), f64::from(u));
                (l + rng.unit() * (u - l)).clamp(l, u)
            })
            .collect();
        points.push(interior);
    }
    points
}

fn assert_encloses(bounds: &BoundedTensor, values: &[f64], ctx: &str) {
    let lower = flat_f32(bounds.lower());
    let upper = flat_f32(bounds.upper());
    assert_eq!(lower.len(), values.len(), "{ctx}: bound length");
    for (i, &v) in values.iter().enumerate() {
        let (l, u) = (f64::from(lower[i]), f64::from(upper[i]));
        assert!(
            v >= l - TOL,
            "{ctx}: element {i}: concrete {v} is BELOW the published lower {l} (unsound)"
        );
        assert!(
            v <= u + TOL,
            "{ctx}: element {i}: concrete {v} is ABOVE the published upper {u} (unsound)"
        );
    }
}

fn assert_bitwise_eq(a: &BoundedTensor, b: &BoundedTensor, ctx: &str) {
    assert_eq!(a.shape(), b.shape(), "{ctx}: shape");
    for (side, (xs, ys)) in [
        ("lower", (a.lower(), b.lower())),
        ("upper", (a.upper(), b.upper())),
    ] {
        for (i, (x, y)) in xs.iter().zip(ys.iter()).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "{ctx}: {side}[{i}] patch-native={x} dense={y}"
            );
        }
    }
}

/// Certified per-layer boxes for `layers[..count]`, the collector's own
/// forward (independent of the route under test).
fn certified_prior(layers: &[Layer], input: &BoundedTensor, count: usize) -> Vec<BoundedTensor> {
    let mut boxes = Vec::with_capacity(count);
    let mut current = input.clone();
    for layer in layers.iter().take(count) {
        current = propagate_ibp_certified(layer, &current)
            .expect("certified forward")
            .into_tensor();
        boxes.push(current.clone());
    }
    boxes
}

/// One partial walk for `layers[..=target]` through the routed body.
fn walk(
    network: &Network,
    input: &BoundedTensor,
    target: usize,
    prior: &[BoundedTensor],
    deadline: Option<Instant>,
    patch_native: bool,
    ctx: &str,
) -> BoundedTensor {
    let layers = network.layers();
    match propagate_crown_partial_routed(
        layers,
        input,
        &layers[..=target],
        prior,
        None,
        deadline,
        None,
        patch_native,
        None,
    ) {
        Ok(PartialCrownPropagationResult::Crown(bounds)) => *bounds,
        Ok(PartialCrownPropagationResult::ForwardFallback(fallback)) => panic!(
            "{ctx}: walk degraded to the forward bound ({:?}): {}",
            fallback.reason, fallback.details
        ),
        Err(error) => panic!("{ctx}: walk failed: {error}"),
    }
}

/// Guard the reference: the f64 forward must sit inside ny's own CERTIFIED
/// forward of a POINT box (width ~1e-6), so a reference with the wrong conv
/// semantics fails here instead of passing or failing the enclosure test for
/// the wrong reason.
fn assert_reference_matches_network(fx: &ConvFixture) {
    let center: Vec<f32> = flat_f32(fx.input.lower())
        .iter()
        .zip(flat_f32(fx.input.upper()))
        .map(|(&l, u)| l + (u - l) * 0.5)
        .collect();
    let point = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[IN_C, IN_H, IN_W]), center.clone()).expect("point"),
        ArrayD::from_shape_vec(IxDyn(&[IN_C, IN_H, IN_W]), center.clone()).expect("point"),
    )
    .expect("point box");
    let boxes = certified_prior(fx.network.layers(), &point, 3);
    let x: Vec<f64> = center.iter().map(|&c| f64::from(c)).collect();
    let reference = forward_f64(fx, &x);
    for (layer, (bounds, values)) in boxes.iter().zip(reference.iter()).enumerate() {
        assert_encloses(
            bounds,
            values,
            &format!("reference self-check layer {layer}"),
        );
        let widest = flat_f32(bounds.lower())
            .iter()
            .zip(flat_f32(bounds.upper()))
            .map(|(&l, u)| f64::from(u) - f64::from(l))
            .fold(0.0f64, f64::max);
        assert!(
            widest < 1e-3,
            "reference self-check layer {layer}: point-box forward width {widest} is too \
             wide to discriminate the reference conv semantics"
        );
    }
}

/// Obligation 1 — SOUNDNESS of the patch-native route, raw and published.
#[test]
fn patch_native_terminal_encloses_sampled_forward() {
    let fx = conv_relu_conv();
    assert_reference_matches_network(&fx);

    let _ = take_terminal_route_trace();
    let collected = NetworkIbpExt::collect_crown_ibp_bounds_with_engine_deadline_and_status_impl(
        &fx.network,
        &fx.input,
        None,
        live_deadline(),
    )
    .expect("sequential CROWN-IBP collection");
    let collector_trace = take_terminal_route_trace();
    let collector_engaged = collector_trace
        .iter()
        .filter(|(route, _)| *route == ROUTE_PATCH_NATIVE)
        .count();
    assert!(
        collector_engaged >= 2,
        "VOID: the collector's two conv walks did not both take the patch-native \
         terminal route (trace {collector_trace:?}); the enclosure checks below would \
         not exercise the change"
    );
    assert_eq!(collected.bounds.len(), 3, "one bound per layer");
    for target in [0usize, 2] {
        assert_eq!(
            collected.provenance[target],
            BoundsProvenance::Crown,
            "VOID: layer {target} was not published from the CROWN walk"
        );
    }

    // The RAW patch-native results, before the collector's IBP intersection.
    let raw0 = walk(
        &fx.network,
        &fx.input,
        0,
        &[],
        live_deadline(),
        true,
        "raw target 0",
    );
    let raw2 = walk(
        &fx.network,
        &fx.input,
        2,
        &collected.bounds[..2],
        live_deadline(),
        true,
        "raw target 2",
    );
    let raw_trace = take_terminal_route_trace();
    assert_eq!(
        raw_trace
            .iter()
            .filter(|(route, _)| *route == ROUTE_PATCH_NATIVE)
            .count(),
        2,
        "VOID: the raw walks did not both engage the patch-native route ({raw_trace:?})"
    );

    let mut rng = SplitMix(SEED ^ 0x5a5a_5a5a);
    let lower = flat_f32(fx.input.lower());
    let upper = flat_f32(fx.input.upper());
    let points = sample_inputs(&mut rng, &lower, &upper);
    assert!(points.len() >= 200, "at least 200 samples");
    for (s, x) in points.iter().enumerate() {
        let [y0, a1, y2] = forward_f64(&fx, x);
        assert_encloses(
            &raw0,
            &y0,
            &format!("sample {s}: raw patch-native target 0"),
        );
        assert_encloses(
            &raw2,
            &y2,
            &format!("sample {s}: raw patch-native target 2"),
        );
        assert_encloses(
            &collected.bounds[0],
            &y0,
            &format!("sample {s}: published layer 0"),
        );
        assert_encloses(
            &collected.bounds[1],
            &a1,
            &format!("sample {s}: published layer 1"),
        );
        assert_encloses(
            &collected.bounds[2],
            &y2,
            &format!("sample {s}: published layer 2"),
        );
    }
}

/// Obligation 2 — the route ENGAGES (VOID otherwise) and is bit-identical to
/// the historical densify route, including on a terminal carrying `coeff_err`.
#[test]
fn patch_native_terminal_engages_and_matches_dense_route_bit_for_bit() {
    let fx = conv_relu_conv();
    let layers = fx.network.layers();
    let prior = certified_prior(layers, &fx.input, 2);
    let _ = take_terminal_route_trace();
    let mut engaged_with_err = 0usize;
    // Both deadline regimes: the scored one (live finite deadline) and the
    // no-deadline one (which also runs the sparse-seed discovery; this fixture
    // is far below its 90%-stable threshold, so the full seed is kept).
    let arms = [(0usize, true), (2, true), (0, false), (2, false)];
    for (target, deadlined) in arms {
        let deadline = if deadlined { live_deadline() } else { None };
        let ctx = format!("target {target} deadline={}", u8::from(deadlined));
        let patch_native = walk(
            &fx.network,
            &fx.input,
            target,
            &prior[..target],
            deadline,
            true,
            &ctx,
        );
        let native_trace = take_terminal_route_trace();
        assert_eq!(
            native_trace.len(),
            1,
            "VOID: {ctx}: expected exactly one terminal decision ({native_trace:?})"
        );
        assert_eq!(
            native_trace[0].0, ROUTE_PATCH_NATIVE,
            "VOID: {ctx}: the patch-native terminal route did not engage ({native_trace:?})"
        );
        if native_trace[0].1 {
            engaged_with_err += 1;
        }

        let dense = walk(
            &fx.network,
            &fx.input,
            target,
            &prior[..target],
            deadline,
            false,
            &ctx,
        );
        let dense_trace = take_terminal_route_trace();
        assert_eq!(
            dense_trace,
            vec![(ROUTE_DENSE_DISABLED, false)],
            "{ctx}: the control arm must take the historical densify route"
        );
        assert_bitwise_eq(&patch_native, &dense, &ctx);
    }
    assert!(
        engaged_with_err >= 1,
        "VOID: no engaged terminal carried coeff_err, so the certified-error arm of \
         the patch-native kernel was not compared against the dense route"
    );
}

/// Obligation 3 — SCOPE: out-of-scope shapes never engage, and a live
/// Dense-carrier walk is bit-identical between arms.
#[test]
fn patch_native_terminal_is_inert_outside_its_scope() {
    // (a) The pure scope predicate.
    let full = CrownBounds::Patches(Box::new(PatchesLinearBounds::identity(
        (2, 3, 3),
        (2, 3, 3),
    )));
    assert!(patch_native_terminal_scope(true, false, false, &full).is_ok());
    assert_eq!(
        patch_native_terminal_scope(false, false, false, &full).err(),
        Some(ROUTE_DENSE_DISABLED)
    );
    assert_eq!(
        patch_native_terminal_scope(true, true, false, &full).err(),
        Some(ROUTE_SPARSE_SEED)
    );
    assert_eq!(
        patch_native_terminal_scope(true, false, true, &full).err(),
        Some(ROUTE_DENSE_UNSTABLE_ROWS)
    );
    let dense = CrownBounds::Dense(LinearBounds::identity(18));
    assert_eq!(
        patch_native_terminal_scope(true, false, false, &dense).err(),
        Some(ROUTE_DENSE_CARRIER)
    );
    let sparse_layout = CrownBounds::Patches(Box::new(PatchesLinearBounds::sparse_identity(
        (2, 3, 3),
        (2, 3, 3),
        UnstableIdx {
            channels: vec![0, 1],
            heights: vec![0, 2],
            widths: vec![1, 0],
        },
    )));
    assert_eq!(
        patch_native_terminal_scope(true, false, false, &sparse_layout).err(),
        Some(ROUTE_SPARSE_LAYOUT)
    );
    let anchored_side = PatchesData {
        coeff_err: None,
        patches: Some(
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2, 1, 1]), vec![1.0, 2.0, 3.0, 4.0])
                .expect("anchored patches"),
        ),
        geometry: PatchGeometry::anchored(vec![0], vec![0, 1]).expect("anchored geometry"),
        identity: false,
        output_shape: (1, 1, 2),
        input_shape: (2, 1, 2),
        unstable_idx: None,
    };
    let anchored = CrownBounds::Patches(Box::new(PatchesLinearBounds {
        row_count: 2,
        lower_a: anchored_side.clone(),
        lower_b: Array1::zeros(2),
        upper_a: anchored_side,
        upper_b: Array1::zeros(2),
    }));
    assert_eq!(
        patch_native_terminal_scope(true, false, false, &anchored).err(),
        Some(ROUTE_ANCHORED_GEOMETRY)
    );

    // (b) A live walk whose terminal is a Dense carrier: Conv -> ReLU ->
    // Flatten -> Linear, target = the Linear output (1D, Dense seed).
    let mut rng = SplitMix(SEED ^ 0xd15a_b1ed);
    let conv0 = conv3x3(&mut rng, C1, IN_C, 1);
    let head_in = C1 * IN_H * IN_W;
    let head_out = 16usize;
    let head = Array2::from_shape_vec(
        (head_out, head_in),
        (0..head_out * head_in)
            .map(|_| rng.uniform_f32(-0.2, 0.2))
            .collect(),
    )
    .expect("head shape");
    let head_bias = Array1::from(
        (0..head_out)
            .map(|_| rng.uniform_f32(-0.1, 0.1))
            .collect::<Vec<f32>>(),
    );
    let mut network = Network::new();
    network.add_layer(Layer::Conv2d(conv0));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Flatten(FlattenLayer::flatten_all()));
    network.add_layer(Layer::Linear(
        LinearLayer::new(head, Some(head_bias)).expect("linear head"),
    ));
    let input = input_box(&mut rng);
    let prior = certified_prior(network.layers(), &input, 3);

    let _ = take_terminal_route_trace();
    let with_route = walk(
        &network,
        &input,
        3,
        &prior,
        live_deadline(),
        true,
        "dense-carrier arm",
    );
    let route_trace = take_terminal_route_trace();
    assert_eq!(
        route_trace.len(),
        1,
        "VOID: expected exactly one terminal decision on the Dense-carrier walk \
         ({route_trace:?})"
    );
    assert!(
        matches!(
            route_trace[0].0,
            ROUTE_DENSE_CARRIER | ROUTE_DENSE_UNSTABLE_ROWS
        ),
        "a Dense-carrier terminal must be classified out of scope, got {route_trace:?}"
    );
    let without_route = walk(
        &network,
        &input,
        3,
        &prior,
        live_deadline(),
        false,
        "dense-carrier control",
    );
    let control_trace = take_terminal_route_trace();
    assert_eq!(control_trace, vec![(ROUTE_DENSE_DISABLED, false)]);
    assert_bitwise_eq(&with_route, &without_route, "dense-carrier walk");
}
