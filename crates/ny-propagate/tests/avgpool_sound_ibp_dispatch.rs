// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Regression for the AveragePool 1-ULP-arm certification (#vnncomp-aw-soundness,
//! #avgpool-1ulp-arm) — the AveragePool analogue of `conv_family_sound_ibp_dispatch.rs`.
//!
//! The SOUND IBP drivers (`GraphNetwork::propagate_ibp_sound`,
//! `Network::propagate_ibp_sound`, `Network::collect_ibp_bounds_sound`) must route
//! AveragePool to its certified `γ⁶⁴_{k+1}·S/d` Higham forward
//! (`AveragePoolLayer::propagate_ibp_sound`), not the generic 1-ULP arm.
//!
//! Why 1 ULP is provably insufficient — the plain forward accumulates the window
//! sum in f64 and directed-rounds only the final f64→f32 store. Cancellation window
//! (spatial row-major order) `[2^30, 2^-30, -2^30]`, kernel 1×3:
//!   f64 accumulation: 2^30 + 2^-30 rounds to 2^30 exactly (2^-30 is below the
//!                     half-ULP 2^-23 at magnitude 2^30), then − 2^30 → 0.0.
//!   true:             2^30 + 2^-30 − 2^30 = 2^-30, so true average = 2^-30/3 ≈ 3.1e-10.
//! `next_up_f32(0.0)` is the smallest subnormal ≈ 1.4e-45; even a further generic
//! 1-ULP widening stays ≈ 2.8e-45 — the "sound" box EXCLUDES the true output (a
//! false bound on the verdict / intermediate-bound path). The certified term
//! `γ⁶⁴_4 · S/d ≈ 4.4e-16 · 7.2e8 ≈ 3.2e-7` covers it.

use ndarray::{ArrayD, IxDyn};
use num_rational::BigRational;
use ny_propagate::layers::{AveragePoolLayer, BoundPropagation};
use ny_propagate::{GraphNetwork, Layer, Network};
use ny_tensor::{next_up_f32, BoundedTensor};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

const TWO30: f32 = 1_073_741_824.0; // 2^30
const TINY: f32 = 1.0 / 1_073_741_824.0; // 2^-30 exactly (power-of-two divide is exact)

fn true_avg() -> f64 {
    // 2^30 + 2^-30 - 2^30 = 2^-30 exactly (real arithmetic); average over 3.
    f64::from(TINY) / 3.0
}

/// 1×3 cancellation window as a (C=1, H=1, W=3) point box, spatial order chosen so
/// the f64 accumulation loses the 2^-30 term.
fn cancellation_case() -> (AveragePoolLayer, BoundedTensor) {
    let layer = AveragePoolLayer::new((1, 3), (1, 1), (0, 0), true);
    let pt = ArrayD::from_shape_vec(IxDyn(&[1, 1, 3]), vec![TWO30, TINY, -TWO30]).unwrap();
    let input = BoundedTensor::new(pt.clone(), pt).unwrap();
    (layer, input)
}

fn single_layer_network(layer: AveragePoolLayer) -> Network {
    let mut net = Network::new();
    net.add_layer(Layer::AveragePool(layer));
    net
}

fn only_upper(bounds: &BoundedTensor) -> f32 {
    let mut it = bounds.upper().iter().copied();
    let v = it.next().unwrap();
    assert!(it.next().is_none(), "expected a single output element");
    v
}

fn assert_encloses_true_avg(bounds: &BoundedTensor, label: &str) {
    let lower = bounds.lower().iter().copied().next().unwrap() as f64;
    let upper = only_upper(bounds) as f64;
    let t = true_avg();
    assert!(
        lower <= t && t <= upper,
        "{label}: certified sound box [{lower:e}, {upper:e}] must enclose the true \
         average {t:e}; an upper near 1e-45 means AveragePool fell into the generic \
         1-ULP arm instead of the certified γ⁶⁴ forward"
    );
    // Sanity on the other side: the certified widening is tiny (γ⁶⁴_4·S/d ≈ 3.2e-7),
    // not a uselessly wide box.
    assert!(
        upper <= 1e-4 && lower >= -1e-4,
        "{label}: certified box [{lower:e}, {upper:e}] far wider than the derived \
         γ⁶⁴_(k+1)·S/d ≈ 3.2e-7 error"
    );
}

/// The plain forward really exhibits the false tightness the certification exists
/// for, and it provably escapes a 1-ULP (even 2-ULP) widening — the certification
/// is not vacuous.
#[test]
fn plain_ibp_escapes_one_ulp_widening_under_cancellation() {
    let (layer, input) = cancellation_case();
    let plain = layer.propagate_ibp(&input).unwrap();
    let upper = only_upper(&plain);
    let t = true_avg();
    assert!(
        (upper as f64) < t,
        "plain AveragePool IBP upper {upper:e} should EXCLUDE the true average {t:e} \
         (cancellation); if this holds the case no longer exercises the bug"
    );
    // One MORE generic 1-ULP widening (what the drivers' `_` arm adds on top)
    // still excludes the true value: 1-ULP-arm soundness for AveragePool is a
    // false assumption, not merely untight.
    let upper_widened = next_up_f32(upper);
    assert!(
        (upper_widened as f64) < t,
        "even the generic 1-ULP-widened upper {upper_widened:e} must exclude the \
         true average {t:e} for this regression to be meaningful"
    );
}

#[test]
fn layer_sound_ibp_encloses_under_cancellation() {
    let (layer, input) = cancellation_case();
    let sound = layer.propagate_ibp_sound(&input).unwrap();
    assert_encloses_true_avg(&sound, "AveragePoolLayer::propagate_ibp_sound");
}

/// Dispatch pinning: every SOUND driver must route AveragePool to the certified
/// forward (the generic arm's box fails `assert_encloses_true_avg`).
#[test]
fn graph_sound_ibp_dispatches_avgpool_to_certified_forward() {
    let (layer, input) = cancellation_case();
    let graph = GraphNetwork::from_sequential(&single_layer_network(layer)).unwrap();
    let out = graph.propagate_ibp_sound(&input).unwrap();
    assert_encloses_true_avg(&out, "GraphNetwork::propagate_ibp_sound AveragePool");
}

#[test]
fn sequential_sound_ibp_dispatches_avgpool_to_certified_forward() {
    let (layer, input) = cancellation_case();
    let net = single_layer_network(layer);
    let out = net.propagate_ibp_sound(&input).unwrap();
    assert_encloses_true_avg(&out, "Network::propagate_ibp_sound AveragePool");
}

#[test]
fn sequential_collect_sound_ibp_dispatches_avgpool_to_certified_forward() {
    let (layer, input) = cancellation_case();
    let net = single_layer_network(layer);
    let bounds = net.collect_ibp_bounds_sound(&input).unwrap();
    assert_eq!(bounds.len(), 1, "one layer => one collected bound");
    assert_encloses_true_avg(&bounds[0], "Network::collect_ibp_bounds_sound AveragePool");
}

// ---------------------------------------------------------------------------
// Randomized enclosure against an EXACT rational reference.
// ---------------------------------------------------------------------------

fn rat(v: f32) -> BigRational {
    BigRational::from_float(f64::from(v)).expect("finite test value")
}

/// Exact rational average pool of one endpoint array (3D (C, H, W) only), in the
/// layer's output iteration order. AveragePool has positive weights, so the true
/// interval extremes are exactly the pools of the endpoint arrays.
fn exact_avgpool(arr: &ArrayD<f32>, layer: &AveragePoolLayer) -> Vec<BigRational> {
    let shape = arr.shape();
    let (channels, in_h, in_w) = (shape[0], shape[1], shape[2]);
    let ((kh, kw), (sh, sw), (ph, pw)) = if layer.is_global() {
        ((in_h, in_w), (1, 1), (0, 0))
    } else {
        (layer.kernel_size, layer.stride, layer.padding)
    };
    let (out_h, out_w) = layer.output_size(in_h, in_w).unwrap();
    let mut out = Vec::with_capacity(channels * out_h * out_w);
    for c in 0..channels {
        for oh in 0..out_h {
            for ow in 0..out_w {
                let mut sum = BigRational::from_float(0.0).unwrap();
                let mut count = 0usize;
                for kh_off in 0..kh {
                    for kw_off in 0..kw {
                        let ih = (oh * sh + kh_off) as isize - ph as isize;
                        let iw = (ow * sw + kw_off) as isize - pw as isize;
                        if ih >= 0 && ih < in_h as isize && iw >= 0 && iw < in_w as isize {
                            sum += rat(arr[[c, ih as usize, iw as usize]]);
                            count += 1;
                        } else if layer.count_include_pad {
                            count += 1;
                        }
                    }
                }
                let divisor = if layer.count_include_pad {
                    kh * kw
                } else {
                    count.max(1)
                };
                out.push(sum / rat(divisor as f32));
            }
        }
    }
    out
}

/// Random mixed-sign, mixed-magnitude boxes: the certified sound bounds must
/// ENCLOSE the exact rational average of each endpoint array, at the layer level
/// and through both sound drivers.
#[test]
fn sound_ibp_encloses_exact_reference_on_random_mixed_sign_windows() {
    let mut rng = StdRng::seed_from_u64(0x0041_7667_506f_6f6c); // "AvgPool"
    let geometries = [
        // (kernel, stride, padding, count_include_pad)
        ((3, 3), (1, 1), (0, 0), true),
        ((3, 2), (2, 1), (1, 1), false),
        ((2, 2), (2, 2), (1, 1), true),
        ((0, 0), (1, 1), (0, 0), true), // global pooling sentinel
    ];
    for (trial, &(kernel, stride, padding, cip)) in geometries.iter().cycle().take(24).enumerate() {
        let layer = AveragePoolLayer::new(kernel, stride, padding, cip);
        let (c, h, w) = (2usize, 5usize, 5usize);
        let n = c * h * w;
        // Mixed signs and magnitudes 2^-20..2^20 to force genuine rounding and
        // cancellation in the window sums.
        let draw = |rng: &mut StdRng| -> f32 {
            let sign = if rng.random_range(0..2) == 0 {
                -1.0
            } else {
                1.0
            };
            let mantissa: f32 = rng.random_range(1.0..2.0);
            let exp: i32 = rng.random_range(-20..=20);
            sign * mantissa * 2f32.powi(exp)
        };
        let a: Vec<f32> = (0..n).map(|_| draw(&mut rng)).collect();
        let b: Vec<f32> = (0..n).map(|_| draw(&mut rng)).collect();
        let lower: Vec<f32> = a.iter().zip(&b).map(|(&x, &y)| x.min(y)).collect();
        let upper: Vec<f32> = a.iter().zip(&b).map(|(&x, &y)| x.max(y)).collect();
        let lower = ArrayD::from_shape_vec(IxDyn(&[c, h, w]), lower).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[c, h, w]), upper).unwrap();
        let input = BoundedTensor::new(lower.clone(), upper.clone()).unwrap();

        let exact_lo = exact_avgpool(&lower, &layer);
        let exact_up = exact_avgpool(&upper, &layer);

        let check = |bounds: &BoundedTensor, label: &str| {
            let lo: Vec<f32> = bounds.lower().iter().copied().collect();
            let up: Vec<f32> = bounds.upper().iter().copied().collect();
            assert_eq!(lo.len(), exact_lo.len(), "{label}: output arity");
            for (o, ((l, u), (el, eu))) in lo
                .iter()
                .zip(&up)
                .zip(exact_lo.iter().zip(&exact_up))
                .enumerate()
            {
                assert!(
                    rat(*l) <= *el && *eu <= rat(*u),
                    "{label} trial {trial} output {o}: certified sound box \
                     [{l:e}, {u:e}] must enclose the exact rational range \
                     [{el}, {eu}] (kernel {kernel:?} stride {stride:?} \
                     padding {padding:?} count_include_pad {cip})"
                );
            }
        };

        check(
            &layer.propagate_ibp_sound(&input).unwrap(),
            "AveragePoolLayer::propagate_ibp_sound",
        );
        let net = single_layer_network(layer.clone());
        check(
            &net.propagate_ibp_sound(&input).unwrap(),
            "Network::propagate_ibp_sound",
        );
        let graph = GraphNetwork::from_sequential(&net).unwrap();
        check(
            &graph.propagate_ibp_sound(&input).unwrap(),
            "GraphNetwork::propagate_ibp_sound",
        );
    }
}
