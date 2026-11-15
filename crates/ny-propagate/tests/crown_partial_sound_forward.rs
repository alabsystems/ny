// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #crown-partial-sound-forward — the sequential CROWN-IBP collector must publish
//! a CERTIFIED per-layer box on every row it does not tighten with CROWN.
//!
//! # The defect
//!
//! `network/ibp/crown_partial.rs` computed the last layer's box with the PLAIN
//! `Layer::propagate_ibp`. For the conv family that arm is an im2col GEMM in
//! round-to-nearest f32 with no directed rounding and no Higham term, and
//! `conv2d/bound.rs` (`propagate_ibp_sound_with_engine`) documents that it can
//! sit INWARD of the true supremum under cancellation. That plain box then
//! (a) chose the sparse Patches seed rows, and (b) was merged VERBATIM into every
//! row the seed did not track (`sparse_merge::merge_sparse_crown_with_ibp`), so
//! the RETURNED CROWN-IBP bound carried the plain value on those rows. The
//! collector then intersected it with its incumbent — which was itself the plain
//! `collect_ibp_bounds` — and published it into the sequential β-CROWN engine's
//! ReLU relaxations and `Verified` checks. A box that excludes the true value is
//! the false-VERIFIED direction: -150 on the board.
//!
//! # The fixture
//!
//! One Conv2d channel carries the documented cancellation kernel
//! `[2^24, 1, -2^24, 4]` over a 2x2 single-channel input whose fourth pixel is
//! an interval `[1, 2]` (non-point, so the sparse-seed route is the one under
//! test). The plain W⁺/W⁻-split interval forward sums the W⁺ part
//! `2^24 + 1 + 4·x₃` in f32: `2^24 + 1` is exactly half an ULP at 2^24 and ties
//! to even, and `2^24 + 5` / `2^24 + 9` tie to even as well, so EVERY summation
//! order lands on `2^24 + 4` (lower) and `2^24 + 8` (upper); adding the W⁻ term
//! `-2^24` and the bias 100 gives the plain box `[104, 108]` against the true
//! range `[105, 109]`. BOTH endpoints land exactly
//! 1 below the truth, so the LOWER one is sound (wider) while the UPPER one is
//! INWARD: the box excludes the true maximum 109, which is the false-VERIFIED
//! direction and is what this fixture bites on. The bias keeps the channel
//! ReLU-STABLE (all positive) under both the plain and the certified box, so it
//! is an UNTRACKED row of the sparse seed and its published value is whatever
//! the merge takes from the per-layer box — exactly the injection site.
//!
//! A second channel straddles zero so the seed is non-empty and >90 % of the
//! 32 channels are stable (the sparse-mode threshold), and thirty zero-kernel
//! channels pad the count. The successor is ReLU, so the collector runs the
//! partial walk for the Conv2d output (`layer_output_needs_partial_crown`).
//!
//! # What is asserted
//!
//! 1. NEGATIVE CONTROL: the plain `Conv2dLayer::propagate_ibp` box on the
//!    cancellation channel EXCLUDES the exactly-evaluated (BigRational) corner
//!    values. If this ever stops holding the fixture no longer bites and must
//!    be replaced, not deleted.
//! 2. THE FIX: the collector's published bound for layer 0 ENCLOSES both
//!    corner values on that channel, with `Crown` provenance (the partial walk
//!    ran and published; it was not a forward fallback). This fails on the
//!    pre-fix tree — the merge published `[104, 108]` — and on any partial
//!    revert: a plain `output_bounds` loses through the merge, and a plain
//!    incumbent wins the collector's per-element intersection on the row.
//! 3. The same, with a live deadline and the `bab.cgan_reclaim` set armed —
//!    the only production configuration in which the sparse seed route runs on
//!    a deadlined collection (`crown_partial_expiry_armed`).

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use num_rational::BigRational;
use ny_propagate::layers::{BoundPropagation, Conv2dLayer, FlattenLayer, LinearLayer, ReLULayer};
use ny_propagate::types::{BoundsProvenance, CrownIbpFallbackReason};
use ny_propagate::{Layer, Network};
use ny_tensor::BoundedTensor;
use std::time::{Duration, Instant};

const TWO24: f32 = 16_777_216.0; // 2^24
const OUT_CHANNELS: usize = 32;
/// The cancellation channel: the row whose plain box is inward of the truth.
const CANCEL_CH: usize = 0;
/// The straddling channel: keeps the sparse seed non-empty.
const UNSTABLE_CH: usize = 1;
/// Kernel positions (0,0) (0,1) (1,0) (1,1) — the shape from `conv2d/bound.rs`.
const CANCEL_KERNEL: [f32; 4] = [TWO24, 1.0, -TWO24, 4.0];
/// Keeps the cancellation channel ReLU-stable (all positive) on both arms.
const CANCEL_BIAS: f32 = 100.0;
/// Input pixels (row-major over the 2x2 image); pixel 3 is the interval.
const INPUT_LOWER: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
const INPUT_UPPER: [f32; 4] = [1.0, 1.0, 1.0, 2.0];

fn rat(v: f32) -> BigRational {
    BigRational::from_float(v).expect("finite fixture value")
}

/// Exact `Σ W_k·x_k + bias` for the cancellation channel at one input corner.
fn exact_cancel_channel(x: &[f32; 4]) -> BigRational {
    let mut acc = rat(CANCEL_BIAS);
    for (w, xv) in CANCEL_KERNEL.iter().zip(x.iter()) {
        acc += rat(*w) * rat(*xv);
    }
    acc
}

fn conv_layer() -> Conv2dLayer {
    let mut kernel = ArrayD::<f32>::zeros(IxDyn(&[OUT_CHANNELS, 1, 2, 2]));
    let mut bias = Array1::<f32>::from_elem(OUT_CHANNELS, 1.0);
    for (k, w) in CANCEL_KERNEL.iter().enumerate() {
        kernel[[CANCEL_CH, 0, k / 2, k % 2]] = *w;
    }
    bias[CANCEL_CH] = CANCEL_BIAS;
    // Straddling channel: reads pixel 3 (in [1, 2]) minus 1.5 -> [-0.5, 0.5].
    kernel[[UNSTABLE_CH, 0, 1, 1]] = 1.0;
    bias[UNSTABLE_CH] = -1.5;
    Conv2dLayer::with_input_shape(kernel, Some(bias), (1, 1), (0, 0), 2, 2).expect("conv2d fixture")
}

fn fixture() -> (Network, BoundedTensor) {
    let mut network = Network::new();
    network.add_layer(Layer::Conv2d(conv_layer()));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Flatten(FlattenLayer::flatten_all()));
    let head = Array2::<f32>::from_elem((1, OUT_CHANNELS), 1.0 / OUT_CHANNELS as f32);
    network.add_layer(Layer::Linear(
        LinearLayer::new(head, Some(Array1::zeros(1))).expect("linear head"),
    ));
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), INPUT_LOWER.to_vec()).expect("lower"),
        ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), INPUT_UPPER.to_vec()).expect("upper"),
    )
    .expect("input box");
    (network, input)
}

/// The two corners that attain the cancellation channel's true min and max
/// (its only interval input, pixel 3, carries weight +4 > 0).
fn true_range() -> (BigRational, BigRational) {
    (
        exact_cancel_channel(&INPUT_LOWER),
        exact_cancel_channel(&INPUT_UPPER),
    )
}

fn channel_endpoints(bounds: &BoundedTensor, channel: usize) -> (f32, f32) {
    assert_eq!(
        bounds.shape(),
        &[OUT_CHANNELS, 1, 1],
        "layer-0 box must keep the conv output shape"
    );
    (
        bounds.lower()[[channel, 0, 0]],
        bounds.upper()[[channel, 0, 0]],
    )
}

/// Assertion 1 — the negative control that makes the fixture bite.
fn assert_plain_forward_is_inward(input: &BoundedTensor) {
    let plain = conv_layer()
        .propagate_ibp(input)
        .expect("plain conv forward");
    let (plain_lo, plain_hi) = channel_endpoints(&plain, CANCEL_CH);
    let (true_lo, true_hi) = true_range();
    assert!(
        rat(plain_hi) < true_hi,
        "negative control: the plain Conv2d forward must EXCLUDE the true maximum \
         (plain upper {plain_hi} vs exact {true_hi}); the fixture no longer bites"
    );
    // The inwardness here is ONE-SIDED, and that is the whole defect: the
    // tie-to-even that discards the `+1` term costs BOTH endpoints exactly 1,
    // so the plain box is [104, 108] against the true [105, 109]. A lower
    // endpoint 1 BELOW the truth is sound (wider); an upper endpoint 1 below is
    // INWARD, and a box that excludes the true maximum is the false-VERIFIED
    // direction. Assert non-enclosure, which is what the repair has to fix --
    // NOT that both endpoints are inward, which this fixture never produced.
    assert!(
        rat(plain_lo) <= true_lo,
        "fixture invariant: the plain lower endpoint is 1 below the truth \
         (plain lower {plain_lo} vs exact {true_lo}); if this flips, the \
         cancellation no longer works the way the module doc describes"
    );
    // The channel is ReLU-stable on the plain box, so it is an UNTRACKED row of
    // the sparse seed — the merge publishes the per-layer box on it verbatim.
    assert!(
        plain_lo > 0.0,
        "cancellation channel must be ReLU-stable (plain)"
    );
    let (u_lo, u_hi) = channel_endpoints(&plain, UNSTABLE_CH);
    assert!(
        u_lo < 0.0 && u_hi > 0.0,
        "straddling channel must be unstable so the sparse seed is non-empty"
    );
}

/// Assertion 2/3 — the published layer-0 box encloses the exact corners.
fn assert_published_box_encloses(bounds: &BoundedTensor, label: &str) {
    let (lo, hi) = channel_endpoints(bounds, CANCEL_CH);
    let (true_lo, true_hi) = true_range();
    assert!(
        rat(lo) <= true_lo,
        "{label}: published lower {lo} must enclose the exact minimum {true_lo} — \
         a plain (round-to-nearest) conv box reached the returned CROWN-IBP bound"
    );
    assert!(
        rat(hi) >= true_hi,
        "{label}: published upper {hi} must enclose the exact maximum {true_hi} — \
         a plain (round-to-nearest) conv box reached the returned CROWN-IBP bound"
    );
    assert!(lo <= hi, "{label}: inverted box [{lo}, {hi}]");
}

#[test]
fn crown_ibp_sparse_merge_publishes_certified_conv2d_box_not_plain() {
    // Arm the reclaim set BEFORE any collection in this process: the
    // crown-partial expiry lever is a process-wide OnceLock latched on first
    // use, and the deadline test below needs it armed (idempotent here).
    ny_propagate::set_cgan_reclaim_preset(true);
    let (network, input) = fixture();
    assert_plain_forward_is_inward(&input);

    let result = network
        .collect_crown_ibp_bounds_with_status(&input)
        .expect("sequential CROWN-IBP collection");
    assert_eq!(result.bounds.len(), 4, "one published box per layer");
    // The partial walk must have PUBLISHED its merged box: `Crown`, or the
    // honest-provenance relabel `CrownVacuousResult` (a one-layer partial cannot
    // beat IBP materially, so the vacuity guard may fire — it still publishes
    // the intersected box). Any OTHER fallback reason means the walk did not
    // run and the incumbent was published instead, which is not the site under
    // test.
    assert!(
        matches!(
            result.provenance[0],
            BoundsProvenance::Crown
                | BoundsProvenance::ForwardFallback(CrownIbpFallbackReason::CrownVacuousResult)
        ),
        "layer 0 must be a published CROWN partial result, got {:?} (fallback events: {:?})",
        result.provenance[0],
        result.fallback_events
    );
    assert_published_box_encloses(&result.bounds[0], "no deadline, sparse seed route");
}

#[test]
fn crown_ibp_deadlined_armed_sparse_route_publishes_certified_conv2d_box() {
    ny_propagate::set_cgan_reclaim_preset(true);
    let (network, input) = fixture();
    assert_plain_forward_is_inward(&input);

    // A deadline that cannot expire: with the reclaim set armed the sparse seed
    // route declines on EXPIRY, not presence, so this is the deadlined shape of
    // the same merge (unarmed it would be the full identity seed, which does
    // not merge at all).
    let deadline = Instant::now() + Duration::from_secs(300);
    let bounds = network
        .collect_crown_ibp_bounds_with_engine_and_deadline(&input, None, Some(deadline))
        .expect("deadlined sequential CROWN-IBP collection");
    assert_eq!(bounds.len(), 4, "one published box per layer");
    assert_published_box_encloses(&bounds[0], "live deadline, reclaim set armed");
}
