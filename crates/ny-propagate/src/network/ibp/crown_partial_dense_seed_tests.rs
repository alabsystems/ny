// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #dense-sparse-seed — the Dense unstable-rows seed of the sequential
//! CROWN-IBP partial walk at alpha-beta-CROWN's threshold (`crown_partial.rs`).
//!
//! Fixture: a 4 -> 8 -> 8 -> 3 Gemm/ReLU network whose stability pattern is
//! DESIGNED rather than hoped for, so every assertion below is discriminating:
//!
//! * target 0 (Linear 4->8, ReLU successor): 2 stable-active, 2 stable-inactive
//!   and 4 unstable neurons (stable fraction 0.5);
//! * target 2 (Linear 8->8, ReLU successor): 3 stable-active, 2 stable-inactive
//!   and 3 unstable (stable fraction 0.625);
//! * target 4 (Linear 8->3, the network OUTPUT, no successor): 1 stable and 2
//!   unstable (stable fraction 0.333), so an armed threshold WOULD seed it
//!   sparse if the ReLU-successor restriction were missing.
//!
//! Every stable fraction sits in [0.1, 0.9): the historical 0.9 seeds the full
//! identity everywhere, and the armed 0.1 must seed sparse exactly on the two
//! ReLU-successor targets. Each test fails LOUDLY as VOID when the path under
//! test did not run.

use super::*;
use crate::layers::{LinearLayer, ReLULayer};
use crate::Network;
use ndarray::{ArrayD, IxDyn};
use std::time::Duration;

const IN: usize = 4;
const H1: usize = 8;
const H2: usize = 8;
const OUT: usize = 3;
/// Input box half-width around the origin.
const RADIUS: f32 = 0.5;
/// The armed value `configs/vnncomp26/challenging_certified_training_2026.yaml`
/// ships: alpha-beta-CROWN's `minimum_sparsity: 0.9` in ny's convention.
const ARMED: f32 = 0.1;
const SAMPLES: usize = 256;
const VERTEX_SAMPLES: usize = 64;
/// Maximum slack tolerated in the unsound direction.
const TOL: f64 = 1e-9;
const SEED: u64 = 0xd5e5_5eed_2026_0924;

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

    fn sign(&mut self) -> f32 {
        if self.next_u64() & 1 == 0 {
            1.0
        } else {
            -1.0
        }
    }
}

fn live_deadline() -> Option<Instant> {
    // The scored configuration. With the crown-partial finite-expiry lever at
    // its default (unarmed) a live deadline declines the optional GPU partial
    // route, so the walk is exactly the CPU Dense loop under test.
    Some(Instant::now() + Duration::from_secs(600))
}

fn vector_box(lower: Vec<f32>, upper: Vec<f32>) -> BoundedTensor {
    let n = lower.len();
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[n]), lower).expect("lower shape"),
        ArrayD::from_shape_vec(IxDyn(&[n]), upper).expect("upper shape"),
    )
    .expect("box")
}

fn flat(values: &ArrayD<f32>) -> Vec<f32> {
    values.iter().copied().collect()
}

/// `rows x cols` weights of fixed magnitude and random sign, so every row's
/// absolute sum (hence its IBP radius) is known exactly.
fn signed_weights(rng: &mut SplitMix, rows: usize, cols: usize, magnitude: f32) -> Array2<f32> {
    Array2::from_shape_vec(
        (rows, cols),
        (0..rows * cols).map(|_| magnitude * rng.sign()).collect(),
    )
    .expect("weight shape")
}

fn certified(layer: &Layer, input: &BoundedTensor) -> BoundedTensor {
    propagate_ibp_certified(layer, input)
        .expect("certified forward")
        .into_tensor()
}

/// Bias that CENTERS each row's certified box on zero, then shifts it by the
/// designed offset: 0 keeps the row straddling zero (unstable), a large
/// positive/negative offset makes it sign-stable with a wide margin.
fn centering_bias(weights: &Array2<f32>, input: &BoundedTensor, offsets: &[f32]) -> Array1<f32> {
    let unbiased = Layer::Linear(LinearLayer::new(weights.clone(), None).expect("unbiased layer"));
    let z = certified(&unbiased, input);
    let (lo, hi) = (flat(z.lower()), flat(z.upper()));
    assert_eq!(lo.len(), offsets.len(), "one offset per row");
    Array1::from(
        lo.iter()
            .zip(&hi)
            .zip(offsets)
            .map(|((&l, &h), &offset)| offset - f32::midpoint(l, h))
            .collect::<Vec<f32>>(),
    )
}

struct MlpFixture {
    network: Network,
    input: BoundedTensor,
    w0: Array2<f32>,
    b0: Array1<f32>,
    w2: Array2<f32>,
    b2: Array1<f32>,
    w4: Array2<f32>,
    b4: Array1<f32>,
}

fn mlp() -> MlpFixture {
    let mut rng = SplitMix(SEED);
    let input = vector_box(vec![-RADIUS; IN], vec![RADIUS; IN]);

    // Layer 0: |w| = 0.5, so every row's IBP radius is 0.5 * 4 * 0.5 = 1.0.
    let w0 = signed_weights(&mut rng, H1, IN, 0.5);
    let small: Vec<f32> = (0..4)
        .map(|_| (rng.unit() as f32 - 0.5) * 0.5) // in [-0.25, 0.25)
        .collect();
    let b0 = Array1::from(vec![
        3.0, 3.0, -3.0, -3.0, small[0], small[1], small[2], small[3],
    ]);
    let layer0 = Layer::Linear(LinearLayer::new(w0.clone(), Some(b0.clone())).expect("layer 0"));
    let a1 = certified(&Layer::ReLU(ReLULayer), &certified(&layer0, &input));

    // Layer 2: centred on the certified post-ReLU box, 3 stable-active,
    // 2 stable-inactive, 3 unstable (each unstable half-width >= ~0.87).
    let w2 = signed_weights(&mut rng, H2, H1, 0.25);
    let b2 = centering_bias(&w2, &a1, &[10.0, 10.0, 10.0, -10.0, -10.0, 0.0, 0.0, 0.0]);
    let layer2 = Layer::Linear(LinearLayer::new(w2.clone(), Some(b2.clone())).expect("layer 2"));
    let a3 = certified(&Layer::ReLU(ReLULayer), &certified(&layer2, &a1));

    // Layer 4 (output): 1 stable, 2 unstable.
    let w4 = signed_weights(&mut rng, OUT, H2, 0.3);
    let b4 = centering_bias(&w4, &a3, &[30.0, 0.0, 0.0]);
    let layer4 = Layer::Linear(LinearLayer::new(w4.clone(), Some(b4.clone())).expect("layer 4"));

    let mut network = Network::new();
    network.add_layer(layer0);
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(layer2);
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(layer4);
    MlpFixture {
        network,
        input,
        w0,
        b0,
        w2,
        b2,
        w4,
        b4,
    }
}

fn affine_f64(weights: &Array2<f32>, bias: &Array1<f32>, x: &[f64]) -> Vec<f64> {
    (0..weights.nrows())
        .map(|row| {
            let mut acc = f64::from(bias[row]);
            for (col, &xv) in x.iter().enumerate() {
                acc += f64::from(weights[[row, col]]) * xv;
            }
            acc
        })
        .collect()
}

fn relu_f64(values: &[f64]) -> Vec<f64> {
    values.iter().map(|&v| v.max(0.0)).collect()
}

/// f64 forward of every layer output: `[y0, a1, y2, a3, y4]`.
fn forward_f64(fx: &MlpFixture, x: &[f64]) -> [Vec<f64>; 5] {
    let y0 = affine_f64(&fx.w0, &fx.b0, x);
    let a1 = relu_f64(&y0);
    let y2 = affine_f64(&fx.w2, &fx.b2, &a1);
    let a3 = relu_f64(&y2);
    let y4 = affine_f64(&fx.w4, &fx.b4, &a3);
    [y0, a1, y2, a3, y4]
}

/// Both extreme corners, random vertices, then interior points.
fn sample_inputs(rng: &mut SplitMix) -> Vec<Vec<f64>> {
    let (l, u) = (-f64::from(RADIUS), f64::from(RADIUS));
    let mut points: Vec<Vec<f64>> = Vec::with_capacity(SAMPLES);
    points.push(vec![l; IN]);
    points.push(vec![u; IN]);
    while points.len() < VERTEX_SAMPLES {
        points.push(
            (0..IN)
                .map(|_| if rng.next_u64() & 1 == 0 { l } else { u })
                .collect(),
        );
    }
    while points.len() < SAMPLES {
        points.push(
            (0..IN)
                .map(|_| (l + rng.unit() * (u - l)).clamp(l, u))
                .collect(),
        );
    }
    points
}

fn assert_encloses(bounds: &BoundedTensor, values: &[f64], ctx: &str) {
    let lower = flat(bounds.lower());
    let upper = flat(bounds.upper());
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
            assert_eq!(x.to_bits(), y.to_bits(), "{ctx}: {side}[{i}] {x} vs {y}");
        }
    }
}

/// Flat indices whose certified box straddles zero (`lower < 0 < upper`),
/// the exact instability rule `find_unstable_dense_indices` applies.
fn unstable_positions(bounds: &BoundedTensor) -> Vec<usize> {
    let lower = flat(bounds.lower());
    let upper = flat(bounds.upper());
    (0..lower.len())
        .filter(|&i| lower[i] < 0.0 && upper[i] > 0.0)
        .collect()
}

/// One partial walk for `layers[..=target]` with an injected threshold.
fn walk(
    network: &Network,
    input: &BoundedTensor,
    target: usize,
    prior: &[BoundedTensor],
    dense_min_stable: Option<f32>,
    ctx: &str,
) -> BoundedTensor {
    let layers = network.layers();
    match propagate_crown_partial_routed(
        layers,
        input,
        &layers[..=target],
        prior,
        None,
        live_deadline(),
        None,
        PATCH_NATIVE_TERMINAL_CONCRETIZE,
        dense_min_stable,
    ) {
        Ok(PartialCrownPropagationResult::Crown(bounds)) => *bounds,
        Ok(PartialCrownPropagationResult::ForwardFallback(fallback)) => panic!(
            "{ctx}: walk degraded to the forward bound ({:?}): {}",
            fallback.reason, fallback.details
        ),
        Err(error) => panic!("{ctx}: walk failed: {error}"),
    }
}

fn decision(
    walk: usize,
    rows: usize,
    total: usize,
    min_stable: f32,
    attributable: bool,
) -> DenseSeedDecision {
    DenseSeedDecision {
        walk,
        rows,
        total,
        min_stable,
        attributable,
    }
}

/// Guard the reference: the f64 forward of an interior POINT must sit inside
/// ny's own certified forward of that point box, so a reference with the wrong
/// weight layout fails here rather than passing or failing the enclosure test
/// for the wrong reason.
fn assert_reference_matches_network(fx: &MlpFixture) {
    let mut rng = SplitMix(SEED ^ 0x0123_4567);
    let point: Vec<f32> = (0..IN)
        .map(|_| (rng.unit() as f32 - 0.5) * 2.0 * RADIUS)
        .collect();
    let mut current = vector_box(point.clone(), point.clone());
    let x: Vec<f64> = point.iter().map(|&p| f64::from(p)).collect();
    let reference = forward_f64(fx, &x);
    for (layer_index, layer) in fx.network.layers().iter().enumerate() {
        current = certified(layer, &current);
        assert_encloses(
            &current,
            &reference[layer_index],
            &format!("reference self-check layer {layer_index}"),
        );
        let widest = flat(current.lower())
            .iter()
            .zip(flat(current.upper()))
            .map(|(&l, u)| f64::from(u) - f64::from(l))
            .fold(0.0f64, f64::max);
        assert!(
            widest < 1e-3,
            "reference self-check layer {layer_index}: point-box width {widest} is too \
             wide to discriminate the reference semantics"
        );
    }
}

/// SOUNDNESS — the armed threshold on the collector's own chain: every raw
/// walk and every published (IBP-intersect-CROWN) bound encloses the f64
/// forward at 256 sampled inputs, and the sparse seed provably ran on both
/// ReLU-successor targets.
#[test]
fn dense_sparse_seed_armed_encloses_sampled_forward() {
    let fx = mlp();
    assert_reference_matches_network(&fx);
    let layers = fx.network.layers();
    let _ = take_dense_seed_trace();

    // The collector's incumbent: the certified per-layer forward from the input.
    let mut ibp = Vec::with_capacity(layers.len());
    let mut current = fx.input.clone();
    for layer in layers {
        current = certified(layer, &current);
        ibp.push(current.clone());
    }

    // Mirror `collect_core`: walk each Linear target over the bounds published
    // so far, intersect with the incumbent, and pass ReLUs through the
    // certified forward of the published pre-activation box.
    let publish = |incumbent: &BoundedTensor, raw: &BoundedTensor, ctx: &str| {
        let (tightened, disjoint) = incumbent
            .intersection_per_element(raw)
            .unwrap_or_else(|| panic!("{ctx}: incumbent/CROWN intersection failed"));
        assert_eq!(
            disjoint, 0,
            "{ctx}: two enclosures of the same range cannot be disjoint"
        );
        tightened
    };
    let raw0 = walk(
        &fx.network,
        &fx.input,
        0,
        &[],
        Some(ARMED),
        "armed target 0",
    );
    let pub0 = publish(&ibp[0], &raw0, "layer 0");
    let pub1 = certified(&layers[1], &pub0);
    let raw2 = walk(
        &fx.network,
        &fx.input,
        2,
        &[pub0.clone(), pub1.clone()],
        Some(ARMED),
        "armed target 2",
    );
    let pub2 = publish(&ibp[2], &raw2, "layer 2");
    let pub3 = certified(&layers[3], &pub2);
    let raw4 = walk(
        &fx.network,
        &fx.input,
        4,
        &[pub0.clone(), pub1.clone(), pub2.clone(), pub3.clone()],
        Some(ARMED),
        "armed target 4",
    );
    let pub4 = publish(&ibp[4], &raw4, "layer 4");

    let trace = take_dense_seed_trace();
    assert_eq!(
        trace,
        vec![
            decision(0, 4, H1, ARMED, true),
            decision(2, 3, H2, ARMED, true),
            decision(4, OUT, OUT, DENSE_SEED_HISTORICAL_MIN_STABLE, false),
        ],
        "VOID unless the armed sparse seed ran on BOTH ReLU-successor targets \
         (4/8 and 3/8 rows) and the output target kept the historical full seed"
    );

    let mut rng = SplitMix(SEED ^ 0x5a5a_5a5a);
    let points = sample_inputs(&mut rng);
    assert!(points.len() >= 200, "at least 200 samples");
    let published = [&pub0, &pub1, &pub2, &pub3, &pub4];
    for (s, x) in points.iter().enumerate() {
        let values = forward_f64(&fx, x);
        assert_encloses(
            &raw0,
            &values[0],
            &format!("sample {s}: raw armed target 0"),
        );
        assert_encloses(
            &raw2,
            &values[2],
            &format!("sample {s}: raw armed target 2"),
        );
        assert_encloses(&raw4, &values[4], &format!("sample {s}: raw target 4"));
        for (layer_index, bounds) in published.iter().enumerate() {
            assert_encloses(
                bounds,
                &values[layer_index],
                &format!("sample {s}: published layer {layer_index}"),
            );
        }
    }
}

/// POSITIONAL CORRECTNESS + TIGHTNESS PARITY on target 2: an untracked
/// (IBP-stable) row publishes the certified IBP box bit for bit, and a tracked
/// row carries the same CROWN value the full seed computes for that neuron (a
/// row-to-neuron scatter mix-up would move these by O(1)).
#[test]
fn dense_sparse_seed_tracked_rows_match_full_seed_and_untracked_rows_publish_certified_ibp() {
    let fx = mlp();
    let layers = fx.network.layers();
    let prior0 = certified(&layers[0], &fx.input);
    let prior1 = certified(&layers[1], &prior0);
    let prior = [prior0, prior1];
    let incumbent = certified(&layers[2], &prior[1]);
    let tracked = unstable_positions(&incumbent);
    assert_eq!(
        tracked,
        vec![5, 6, 7],
        "fixture drift: target 2 was designed with exactly rows 5..8 unstable"
    );

    let _ = take_dense_seed_trace();
    let full = walk(&fx.network, &fx.input, 2, &prior, Some(0.9), "full seed");
    let sparse = walk(&fx.network, &fx.input, 2, &prior, Some(ARMED), "armed seed");
    assert_eq!(
        take_dense_seed_trace(),
        vec![
            decision(2, H2, H2, DENSE_SEED_HISTORICAL_MIN_STABLE, false),
            decision(2, tracked.len(), H2, ARMED, true),
        ],
        "VOID unless the control walked all rows and the armed walk only the tracked ones"
    );

    let (full_lo, full_hi) = (flat(full.lower()), flat(full.upper()));
    let (sparse_lo, sparse_hi) = (flat(sparse.lower()), flat(sparse.upper()));
    let (ibp_lo, ibp_hi) = (flat(incumbent.lower()), flat(incumbent.upper()));
    for row in 0..H2 {
        if tracked.contains(&row) {
            for (side, s, f) in [
                ("lower", sparse_lo[row], full_lo[row]),
                ("upper", sparse_hi[row], full_hi[row]),
            ] {
                let (s, f) = (f64::from(s), f64::from(f));
                assert!(
                    (s - f).abs() <= 1e-5 * (1.0 + f.abs()),
                    "tracked row {row} {side}: armed {s} vs full-seed {f} — the sparse \
                     seed must walk the same per-row backward as the full seed"
                );
            }
        } else {
            assert_eq!(
                (sparse_lo[row].to_bits(), sparse_hi[row].to_bits()),
                (ibp_lo[row].to_bits(), ibp_hi[row].to_bits()),
                "untracked row {row} must publish the certified IBP box verbatim"
            );
            assert!(
                ibp_lo[row] >= 0.0 || ibp_hi[row] <= 0.0,
                "untracked row {row} must be IBP sign-stable"
            );
        }
    }
}

/// SCOPE: the armed key never reaches a walk without a ReLU successor (the
/// network output here), and an unset key is byte-identical to the historical
/// 0.9 literal on the production resolver.
#[test]
fn dense_sparse_seed_is_inert_on_the_output_target_and_when_unset() {
    let fx = mlp();
    let layers = fx.network.layers();
    let mut prior = Vec::new();
    let mut current = fx.input.clone();
    for layer in &layers[..4] {
        current = certified(layer, &current);
        prior.push(current.clone());
    }
    let output_incumbent = certified(&layers[4], &prior[3]);
    assert!(
        find_unstable_dense_indices(
            &flat(output_incumbent.lower()),
            &flat(output_incumbent.upper()),
            ARMED,
        )
        .is_some(),
        "fixture drift: the output target must be one the armed threshold WOULD seed \
         sparse, or this scope check proves nothing"
    );

    let _ = take_dense_seed_trace();
    let armed_output = walk(
        &fx.network,
        &fx.input,
        4,
        &prior,
        Some(ARMED),
        "armed output",
    );
    let historical_output = walk(&fx.network, &fx.input, 4, &prior, Some(0.9), "0.9 output");
    assert_eq!(
        take_dense_seed_trace(),
        vec![
            decision(4, OUT, OUT, DENSE_SEED_HISTORICAL_MIN_STABLE, false),
            decision(4, OUT, OUT, DENSE_SEED_HISTORICAL_MIN_STABLE, false),
        ],
        "the output target has no ReLU successor: the key must not be consulted"
    );
    assert_bitwise_eq(&armed_output, &historical_output, "output target");

    // The production resolver (`None`) with the environment layer removed and
    // no preset cell written in this crate's harness: the declaration default,
    // i.e. the historical literal, bit for bit.
    let decl = &ny_levers::decls::crown_partial_seed::CROWN_PARTIAL_DENSE_MIN_STABLE;
    ny_test_utils::env::with_serialized_env_vars_removed(&[decl.name], || {
        let _ = take_dense_seed_trace();
        let unset = walk(&fx.network, &fx.input, 2, &prior[..2], None, "unset key");
        let literal = walk(
            &fx.network,
            &fx.input,
            2,
            &prior[..2],
            Some(0.9),
            "0.9 literal",
        );
        assert_eq!(
            take_dense_seed_trace(),
            vec![
                decision(2, H2, H2, DENSE_SEED_HISTORICAL_MIN_STABLE, false),
                decision(2, H2, H2, DENSE_SEED_HISTORICAL_MIN_STABLE, false),
            ],
            "an unset key must resolve to the historical 0.9 and seed every row"
        );
        assert_bitwise_eq(&unset, &literal, "unset key vs historical literal");
    });
}

/// The pure pieces: the threshold selector never consults the key off a ReLU
/// successor, the successor predicate requires a true prefix, and the
/// "attributable" flag agrees exactly with the seed rule it summarizes.
#[test]
fn dense_sparse_seed_threshold_successor_and_attribution_predicates() {
    assert_eq!(
        dense_seed_threshold(false, || panic!("the key must not be resolved")),
        DENSE_SEED_HISTORICAL_MIN_STABLE
    );
    assert_eq!(dense_seed_threshold(true, || ARMED), ARMED);

    let fx = mlp();
    let layers = fx.network.layers();
    assert!(walk_has_relu_successor(layers, &layers[..=0]));
    assert!(walk_has_relu_successor(layers, &layers[..=2]));
    assert!(
        !walk_has_relu_successor(layers, &layers[..=1]),
        "a ReLU target feeds a Linear, not a ReLU"
    );
    assert!(
        !walk_has_relu_successor(layers, layers),
        "the output layer has no successor"
    );
    assert!(
        !walk_has_relu_successor(layers, &layers[2..=2]),
        "a non-prefix slice must not borrow the successor of an unrelated position"
    );

    for total in 1usize..=40 {
        for unstable in 0..=total {
            let lower: Vec<f32> = (0..total)
                .map(|i| if i < unstable { -1.0 } else { 1.0 })
                .collect();
            let upper = vec![2.0f32; total];
            assert_eq!(
                historical_threshold_seeds_sparse(unstable, total),
                find_unstable_dense_indices(&lower, &upper, DENSE_SEED_HISTORICAL_MIN_STABLE)
                    .is_some(),
                "attribution disagrees with the 0.9 seed rule at {unstable}/{total}"
            );
        }
    }
}
