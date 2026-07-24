// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! STRICT, ZERO-TOLERANCE soundness tests for the activation CROWN-backward
//! composed coefficient certificate (#vnncomp-aw-activation).
//!
//! The bug: `crown_elementwise_backward_indexed` composed each incoming
//! coefficient `a` with the relaxation slope, storing the directed-rounded f32
//! product `next_down_f32(a·slope)` / `next_up_f32(a·slope)` — but it only
//! attached a certified per-coefficient error (`lower_a_err`/`upper_a_err`) when
//! the INCOMING bounds already carried error (`propagate_err == true`). When the
//! incoming bounds had NO error (the common case: an activation directly above
//! the input, or above a layer that carried no err), the composed coefficient
//! was stored with its directed-rounding gap DROPPED — the returned bounds
//! reported `lower_a_err == None`, so a FURTHER backward layer treated the f32
//! composed coefficient as EXACT and under-counted the true error = false-proof
//! risk on the deep-resnet verdict path.
//!
//! These tests assert the CERTIFICATE CONTRACT directly, exactly mirroring the
//! conv sibling (`crown_conv_aw_soundness.rs`): for every output coefficient
//! `(i, j)`,
//!
//!     certified_err[i, j]  >=  |stored_f32_coeff[i, j] − f64_recompute[i, j]|
//!
//! where `f64_recompute` is the SAME composition done in f64 (exact f32→f64
//! widening; f32·f32 fits exactly in f64). On the BUGGY code the returned
//! `lower_a_err` is `None` while the gap is strictly positive for many
//! coefficients → the contract fails. After the fix the certificate is present
//! and covers every gap.

use crate::layers::activations::LinearRelaxation;
use crate::layers::common::crown_elementwise_backward_indexed;
use crate::LinearBounds;
use ndarray::Array2;
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

/// The lower-direction slope `compose_lower` selects for a coefficient of the
/// given sign (lower_slope for `a>0`, upper_slope for `a<0`, 0 for `a==0`).
fn lower_dir_slope(a: f32, relax: &LinearRelaxation) -> f64 {
    if a > 0.0 {
        relax.lower_slope as f64
    } else if a < 0.0 {
        relax.upper_slope as f64
    } else {
        0.0
    }
}

/// The upper-direction slope `compose_upper` selects.
fn upper_dir_slope(a: f32, relax: &LinearRelaxation) -> f64 {
    if a > 0.0 {
        relax.upper_slope as f64
    } else if a < 0.0 {
        relax.lower_slope as f64
    } else {
        0.0
    }
}

/// Build a WIDE, cancelling incoming spec (num_obj × num_neurons) plus per-neuron
/// pre-activation bounds that straddle zero (so each ReLU relaxation has a
/// fractional, non-dyadic lower slope `u/(u-l)` whose f32 product with the
/// incoming coefficient genuinely ROUNDS — the regime the dropped gap bites).
fn wide_relu_setup(num_obj: usize, num_neurons: usize, seed: u64) -> (LinearBounds, BoundedTensor) {
    let mut s = seed | 1;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        // map to a fine, cancelling range that forces f32 product rounding.
        ((s >> 8) as i64 % 20001 - 10000) as f32 / 13107.0
    };

    // Incoming coefficients (lower and upper distinct so both directions are
    // exercised). No err attached → the buggy path stores composed coeffs with
    // NO certified error.
    let lower_a = Array2::<f32>::from_shape_fn((num_obj, num_neurons), |_| next());
    let upper_a = Array2::<f32>::from_shape_fn((num_obj, num_neurons), |_| next());
    let spec = LinearBounds::from_coefficients(lower_a, upper_a).unwrap();
    assert!(
        spec.lower_a_err().is_none() && spec.upper_a_err().is_none(),
        "the incoming spec must carry NO error (this is the propagate_err==false path)"
    );

    // Pre-activation bounds straddling zero with irrational-ish magnitudes so the
    // ReLU relaxation lower slope u/(u-l) is a non-dyadic f32 (forces rounding).
    let mut pl = Vec::with_capacity(num_neurons);
    let mut pu = Vec::with_capacity(num_neurons);
    for _ in 0..num_neurons {
        let a = next().abs() * 3.0 + 0.123_456_7;
        let b = next().abs() * 3.0 + 0.234_567_8;
        // straddle zero: lower negative, upper positive (unstable ReLU).
        pl.push(-a);
        pu.push(b);
    }
    let pre = BoundedTensor::new(
        ndarray::Array1::from_vec(pl).into_dyn(),
        ndarray::Array1::from_vec(pu).into_dyn(),
    )
    .unwrap();
    (spec, pre)
}

/// The adaptive ReLU relaxation used in the test: for an unstable neuron
/// `l < 0 < u`, lower slope `= u/(u-l)` (adaptive lower line through the origin
/// region), upper slope `= u/(u-l)`, upper intercept `= -l·u/(u-l)`. The exact
/// slope value is irrelevant to the contract — only that the COMPOSED coefficient
/// is the f32 product `a·slope`, whose distance to the f64 truth the certificate
/// must cover.
fn relu_relax(l: f32, u: f32) -> LinearRelaxation {
    if u <= 0.0 {
        return LinearRelaxation::new(0.0, 0.0, 0.0, 0.0);
    }
    if l >= 0.0 {
        return LinearRelaxation::new(1.0, 0.0, 1.0, 0.0);
    }
    let slope = u / (u - l);
    let upper_int = -l * u / (u - l);
    // lower slope chosen as `slope` too (a valid adaptive lower line), upper line
    // is the chord. Both slopes are fractional non-dyadic f32 → products round.
    LinearRelaxation::new(slope, 0.0, slope, upper_int)
}

/// REPRODUCE + VERIFY: with NO incoming error, the activation CROWN-backward must
/// STILL attach a certified coefficient error covering the directed-rounding gap
/// of every composed coefficient.
#[test]
fn activation_aw_cert_covers_f32_gap_no_incoming_err() {
    let num_obj = 6;
    let num_neurons = 96;
    let (spec, pre) = wide_relu_setup(num_obj, num_neurons, 0x9E3779B97F4A7C15);

    // Precompute the relaxations exactly as the layer does.
    let pre_flat = pre.flatten();
    let pl = pre_flat
        .lower()
        .clone()
        .into_dimensionality::<ndarray::Ix1>()
        .unwrap();
    let pu = pre_flat
        .upper()
        .clone()
        .into_dimensionality::<ndarray::Ix1>()
        .unwrap();
    let relaxations: Vec<LinearRelaxation> =
        (0..num_neurons).map(|i| relu_relax(pl[i], pu[i])).collect();

    let result =
        crown_elementwise_backward_indexed(&spec, &pre, |l, u, _i| relu_relax(l, u)).unwrap();

    let lower_err = result.lower_a_err().expect(
        "activation CROWN-backward MUST attach a certified lower coefficient error even when \
         the incoming bounds carry no error (directed-rounding gap must not be dropped)",
    );
    let upper_err = result.upper_a_err().expect(
        "activation CROWN-backward MUST attach a certified upper coefficient error even when \
         the incoming bounds carry no error",
    );

    let stored_lower = result.lower_a();
    let stored_upper = result.upper_a();

    let mut worst_uncovered = 0.0f64;
    for i in 0..num_obj {
        for j in 0..num_neurons {
            let relax = &relaxations[j];

            // Lower direction: truth = a_f64 · lower_dir_slope_f64.
            let la = spec.lower_a()[[i, j]];
            let truth_l = la as f64 * lower_dir_slope(la, relax);
            let gap_l = (stored_lower[[i, j]] as f64 - truth_l).abs();
            let cert_l = lower_err[[i, j]] as f64;
            if gap_l - cert_l > worst_uncovered {
                worst_uncovered = gap_l - cert_l;
            }
            assert!(
                cert_l >= gap_l,
                "UNSOUND activation LOWER certificate at ({i},{j}): certified {cert_l:.3e} < \
                 |stored_f32 − f64_truth| {gap_l:.3e} (stored={}, truth={truth_l:.9e})",
                stored_lower[[i, j]]
            );

            // Upper direction.
            let ua = spec.upper_a()[[i, j]];
            let truth_u = ua as f64 * upper_dir_slope(ua, relax);
            let gap_u = (stored_upper[[i, j]] as f64 - truth_u).abs();
            let cert_u = upper_err[[i, j]] as f64;
            assert!(
                cert_u >= gap_u,
                "UNSOUND activation UPPER certificate at ({i},{j}): certified {cert_u:.3e} < \
                 |stored_f32 − f64_truth| {gap_u:.3e} (stored={}, truth={truth_u:.9e})",
                stored_upper[[i, j]]
            );
        }
    }

    // Sanity: the test net must actually exhibit nonzero rounding gaps somewhere,
    // otherwise it cannot detect the dropped-gap bug.
    let mut max_gap = 0.0f64;
    for i in 0..num_obj {
        for j in 0..num_neurons {
            let relax = &relaxations[j];
            let la = spec.lower_a()[[i, j]];
            let truth_l = la as f64 * lower_dir_slope(la, relax);
            let gap_l = (stored_lower[[i, j]] as f64 - truth_l).abs();
            if gap_l > max_gap {
                max_gap = gap_l;
            }
        }
    }
    eprintln!(
        "[activation aw repro] max composed-coeff rounding gap = {max_gap:.3e} \
         (must be > 0 for the test to bite)"
    );
    assert!(
        max_gap > 0.0,
        "test net produced zero rounding gap everywhere — not wide/fine enough to exhibit the bug"
    );
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 4000, ..ProptestConfig::with_cases(60) })]

    /// SOUNDNESS PROPTEST: random wide incoming specs (NO incoming error) — the
    /// patched activation certificate must cover `|stored_f32 − f64_recompute|`
    /// for EVERY composed coefficient in BOTH directions. Zero tolerance.
    #[ntest::timeout(60000)]
    #[test]
    fn proptest_activation_aw_cert_covers_f32_gap(
        seed in any::<u64>(),
        num_neurons in 32usize..=128,
    ) {
        let num_obj = 4;
        let (spec, pre) = wide_relu_setup(num_obj, num_neurons, seed);

        let pre_flat = pre.flatten();
        let pl = pre_flat.lower().clone().into_dimensionality::<ndarray::Ix1>().unwrap();
        let pu = pre_flat.upper().clone().into_dimensionality::<ndarray::Ix1>().unwrap();
        let relaxations: Vec<LinearRelaxation> =
            (0..num_neurons).map(|i| relu_relax(pl[i], pu[i])).collect();

        let result =
            crown_elementwise_backward_indexed(&spec, &pre, |l, u, _i| relu_relax(l, u)).unwrap();

        let lower_err = result.lower_a_err().expect("certified lower error must be present");
        let upper_err = result.upper_a_err().expect("certified upper error must be present");
        let stored_lower = result.lower_a();
        let stored_upper = result.upper_a();

        for i in 0..num_obj {
            for j in 0..num_neurons {
                let relax = &relaxations[j];
                let la = spec.lower_a()[[i, j]];
                let truth_l = la as f64 * lower_dir_slope(la, relax);
                let gap_l = (stored_lower[[i, j]] as f64 - truth_l).abs();
                prop_assert!(
                    lower_err[[i, j]] as f64 >= gap_l,
                    "UNSOUND activation LOWER cert seed {} at ({},{}): {:.3e} < {:.3e}",
                    seed, i, j, lower_err[[i, j]], gap_l
                );
                let ua = spec.upper_a()[[i, j]];
                let truth_u = ua as f64 * upper_dir_slope(ua, relax);
                let gap_u = (stored_upper[[i, j]] as f64 - truth_u).abs();
                prop_assert!(
                    upper_err[[i, j]] as f64 >= gap_u,
                    "UNSOUND activation UPPER cert seed {} at ({},{}): {:.3e} < {:.3e}",
                    seed, i, j, upper_err[[i, j]], gap_u
                );
            }
        }
    }
}
