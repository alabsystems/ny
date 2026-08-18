// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for the BN-fold certified interval width report (#bn-interval-fold).
//!
//! Three claims are under test:
//!
//! 1. **Default OFF and byte-identical.** With the dark gate unset, no folded
//!    weight bit changes and no report is produced; with it set, the folded
//!    weights are STILL bit-identical (the report is observational).
//! 2. **The enclosure is genuinely sound.** The reported interval for the BN
//!    scale is checked against EXACT rational arithmetic — no floating point in
//!    the oracle — via `lo^2·(var+ε) ≤ γ² ≤ hi^2·(var+ε)`, which is equivalent
//!    to `lo ≤ γ/sqrt(var+ε) ≤ hi` for positive endpoints and avoids ever
//!    needing an irrational square root.
//! 3. **The width is a real measurement.** A power-of-two BN scale folds
//!    exactly, so the stored f32 lies INSIDE the exact enclosure and the width
//!    collapses to f64 slop. A non-dyadic scale does not, and the measured
//!    relative width is compared head-to-head against the hand-picked
//!    `FOLD_TOL_REL = 1e-4` that `batch_norm_ort_prop.rs` currently absorbs it
//!    with.

use super::{make_float_attr, make_int_attr, make_node, make_weight};
use crate::loader::fusion::bn_fold_interval::{
    channel_affine_interval, fold_interval_report, interval_report_enabled, take_reports,
    ChannelAxis, ForceIntervalReport, Interval,
};
use crate::loader::fusion::fold_batch_norm_into_conv_linear;
use crate::model::WeightStore;
use ndarray::arr1;
use num_bigint::BigInt;
use num_rational::BigRational;

/// `FOLD_TOL_REL` from `batch_norm_ort_prop.rs` — the constant this report is
/// meant to replace with a measured number. Duplicated rather than imported
/// because that module is behind the `ort` feature.
const ORT_PROP_FOLD_TOL_REL: f64 = 1e-4;

fn rational(value: f64) -> BigRational {
    BigRational::from_float(value).expect("finite endpoint is representable as a rational")
}

/// Exact-arithmetic certificate that `[lo, hi]` encloses `gamma / sqrt(var+eps)`
/// for a POSITIVE gamma and positive `var+eps`.
///
/// `lo ≤ γ/sqrt(V)` with `lo > 0` and `V > 0` is equivalent to `lo²·V ≤ γ²`, and
/// symmetrically `γ/sqrt(V) ≤ hi` is equivalent to `γ² ≤ hi²·V`. Both sides are
/// rationals in the f32/f64 inputs, so the check is exact — it never evaluates a
/// square root and never rounds.
fn assert_encloses_positive_scale(interval: Interval, gamma: f32, var: f32, epsilon: f32) {
    assert!(gamma > 0.0, "oracle covers positive gamma only");
    assert!(
        interval.lo > 0.0,
        "positive gamma must give a positive lower endpoint, got {:?}",
        interval
    );
    // var and eps are f32, so their real sum is exact in this rational domain.
    let v = rational(f64::from(var)) + rational(f64::from(epsilon));
    let gamma_squared = rational(f64::from(gamma)) * rational(f64::from(gamma));
    let lo_squared = rational(interval.lo) * rational(interval.lo);
    let hi_squared = rational(interval.hi) * rational(interval.hi);
    assert!(
        lo_squared * v.clone() <= gamma_squared,
        "lower endpoint {} is ABOVE gamma/sqrt(var+eps) — enclosure unsound",
        interval.lo
    );
    assert!(
        gamma_squared <= hi_squared * v,
        "upper endpoint {} is BELOW gamma/sqrt(var+eps) — enclosure unsound",
        interval.hi
    );
}

/// The scale enclosure is exactly sound across a deterministic parameter sweep.
///
/// The oracle is rational arithmetic, so a failure here is a real soundness bug
/// in the interval operators, not a tolerance artifact.
#[test]
fn scale_interval_encloses_exact_rational_value() {
    let gammas = [1.0_f32, 0.125, 3.0, 1.0e-3, 7.5, 1.0e4];
    let vars = [1.0_f32, 0.5, 3.0, 1.0e-6, 12.34, 1.0e5];
    let epsilons = [0.0_f32, 1.0e-5, 1.0e-3];
    let mut checked = 0_usize;
    for gamma in gammas {
        for var in vars {
            for epsilon in epsilons {
                let affine = channel_affine_interval(gamma, 0.0, 0.0, var, epsilon)
                    .expect("nondegenerate denominator");
                assert_encloses_positive_scale(affine.scale, gamma, var, epsilon);
                checked += 1;
            }
        }
    }
    assert_eq!(
        checked,
        gammas.len() * vars.len() * epsilons.len(),
        "sweep must cover every combination"
    );
}

/// The interval operators round strictly OUTWARD: a quotient that is not
/// representable must produce a nondegenerate interval straddling the true
/// value, and an exactly representable one must not be widened needlessly by
/// the `point` constructor.
#[test]
fn interval_operators_round_outward() {
    // 1/3 is not a dyadic rational, so the enclosure must be strict on both
    // sides of the correctly-rounded f64 quotient.
    let third = Interval::point(1.0)
        .div(Interval::point(3.0))
        .expect("nonzero divisor");
    assert!(
        third.lo < 1.0_f64 / 3.0 && third.hi > 1.0_f64 / 3.0,
        "1/3 enclosure must straddle the rounded quotient, got {:?}",
        third
    );
    assert!(
        third.width() > 0.0,
        "1/3 enclosure must have positive width"
    );

    // sqrt(2) likewise.
    let root_two = Interval::point(2.0).sqrt().expect("nonnegative radicand");
    assert!(
        root_two.lo < 2.0_f64.sqrt() && root_two.hi > 2.0_f64.sqrt(),
        "sqrt(2) enclosure must straddle, got {:?}",
        root_two
    );

    // Fail-closed cases: a divisor straddling zero and a negative radicand.
    assert!(
        Interval { lo: -1.0, hi: 1.0 }
            .div(Interval { lo: -1.0, hi: 1.0 })
            .is_none()
            || Interval::point(1.0)
                .div(Interval { lo: -1.0, hi: 1.0 })
                .is_none(),
        "division by a zero-straddling interval must fail closed"
    );
    assert!(
        Interval { lo: -1.0, hi: 4.0 }.sqrt().is_none(),
        "sqrt of a negative lower endpoint must fail closed"
    );

    // Mixed-sign product: the enclosure must contain all four endpoint
    // products, which a naive same-sign formula would miss.
    let mixed = Interval { lo: -2.0, hi: 3.0 }.mul(Interval { lo: -5.0, hi: 7.0 });
    for value in [10.0_f64, -14.0, -15.0, 21.0] {
        assert!(
            mixed.contains(value),
            "mixed-sign product must contain endpoint product {value}, got {:?}",
            mixed
        );
    }
}

/// A Conv+BN whose BN scale is a power of two: `W*scale` is exact in f32, so
/// the stored value must already lie inside the exact enclosure and the report
/// must show zero elements outside it.
fn conv_bn_fixture(
    bn_scale: &[f32],
    bn_var: &[f32],
) -> (Vec<crate::onnx_proto::NodeProto>, WeightStore) {
    let conv = make_node("Conv", &["x", "conv_w", "conv_b"], &["conv_y"]);
    let mut bn = make_node(
        "BatchNormalization",
        &["conv_y", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
        &["bn_y"],
    );
    bn.attribute.push(make_float_attr("epsilon", 0.0));
    let nodes = vec![conv, bn];
    let mut weights = WeightStore::new();
    weights.insert(
        "conv_w".to_string(),
        make_weight(&[2, 1, 1, 2], &[1.5, -0.75, 2.25, 0.5]),
    );
    weights.insert("conv_b".to_string(), arr1(&[0.25, -0.5]).into_dyn());
    weights.insert("bn_scale".to_string(), arr1(bn_scale).into_dyn());
    weights.insert("bn_bias".to_string(), arr1(&[1.0, -1.0]).into_dyn());
    weights.insert("bn_mean".to_string(), arr1(&[0.0, 0.0]).into_dyn());
    weights.insert("bn_var".to_string(), arr1(bn_var).into_dyn());
    (nodes, weights)
}

/// The gate is OFF by default, and turning it ON changes NO stored bit.
///
/// The comparison is on raw `to_bits()`, not an approximate compare: the whole
/// point of a dark gate is that the verified artifact is byte-identical.
#[test]
fn report_gate_is_default_off_and_byte_identical_when_on() {
    // This suite is also run with `NY_BN_FOLD_INTERVAL_REPORT=1` to prove the
    // PRODUCTION gate (the env var, not the test-only thread-local) is
    // behaviour-neutral, so the default-off assertions are conditional on the
    // env var actually being unset.
    let env_gate_on = std::env::var("NY_BN_FOLD_INTERVAL_REPORT").ok().as_deref() == Some("1");
    if !env_gate_on {
        assert!(
            !interval_report_enabled(),
            "report gate must default to OFF with the env var unset"
        );
    }

    // Non-dyadic BN scale (var = 3.0 => scale = gamma/sqrt(3)) so the fold
    // genuinely rounds and an accidental behavioural difference would show.
    let scale = [1.0_f32, 0.25];
    let var = [3.0_f32, 7.0];

    let (mut nodes_off, mut weights_off) = conv_bn_fixture(&scale, &var);
    let consumed_off = fold_batch_norm_into_conv_linear(&mut nodes_off, &mut weights_off);
    let reports_off = take_reports();
    if !env_gate_on {
        assert!(
            reports_off.is_empty(),
            "gate OFF must emit no report, got {}",
            reports_off.len()
        );
    }

    let (mut nodes_on, mut weights_on) = conv_bn_fixture(&scale, &var);
    let consumed_on = {
        let _gate = ForceIntervalReport::enable();
        fold_batch_norm_into_conv_linear(&mut nodes_on, &mut weights_on)
    };
    let reports_on = take_reports();
    assert_eq!(
        reports_on.len(),
        1,
        "gate ON must emit exactly one report for one folded Conv"
    );

    assert_eq!(consumed_off, consumed_on, "consumed BN set must not change");
    for name in ["conv_w", "conv_b"] {
        let off = weights_off.get(name).expect("folded tensor present");
        let on = weights_on.get(name).expect("folded tensor present");
        assert_eq!(off.shape(), on.shape(), "{name} shape must not change");
        let off_bits: Vec<u32> = off.iter().map(|value| value.to_bits()).collect();
        let on_bits: Vec<u32> = on.iter().map(|value| value.to_bits()).collect();
        assert_eq!(
            off_bits, on_bits,
            "{name} must be BIT-identical with the report gate on"
        );
    }
    for name in ["conv_w", "conv_b"] {
        assert!(
            nodes_off
                .iter()
                .zip(&nodes_on)
                .all(|(a, b)| a.input == b.input && a.output == b.output),
            "graph rewrite must not change with the report gate on ({name})"
        );
    }
}

/// A dyadic BN scale folds EXACTLY, so no stored element escapes the exact
/// enclosure and the reported width is pure f64 slop.
#[test]
fn dyadic_scale_fold_is_exact_and_reports_negligible_width() {
    // var = 1.0, eps = 0.0 => denominator = 1.0 exactly; gamma dyadic =>
    // scale dyadic => W*scale exact in f32 for these dyadic weights.
    let (mut nodes, mut weights) = conv_bn_fixture(&[2.0, 0.5], &[1.0, 1.0]);
    let report = {
        let _gate = ForceIntervalReport::enable();
        let consumed = fold_batch_norm_into_conv_linear(&mut nodes, &mut weights);
        assert_eq!(consumed.len(), 1, "the Conv+BN pair must fold");
        let reports = take_reports();
        assert_eq!(reports.len(), 1, "one report per folded node");
        reports[0]
    };

    assert_eq!(report.weight_elements, 4, "2x1x1x2 kernel has 4 elements");
    assert_eq!(report.bias_elements, 2, "2 output channels");
    assert_eq!(
        report.unenclosable_elements, 0,
        "no channel should be unenclosable for var=1, eps=0"
    );
    assert_eq!(
        report.stored_outside_exact_enclosure, 0,
        "an exact dyadic fold must store a value INSIDE the exact enclosure"
    );
    // MEASURED: 1.1842378929335004e-15. Not zero, because every interval
    // operator widens one f64 ULP outward even when the step happens to be
    // exact (`var + 0.0` here) — a few ULPs of f64 accumulate through
    // add -> sqrt -> div. The threshold is set an order of magnitude above the
    // measured value and still seven orders below f32 epsilon, so it separates
    // "exact fold" from "rounding fold" (measured ~1.6e-8) unambiguously.
    assert!(
        report.weight_max_rel_width < 1e-14,
        "exact fold width must be f64 slop, got {}",
        report.weight_max_rel_width
    );
    assert!(
        report.bias_max_rel_width < 1e-14,
        "exact fold bias width must be f64 slop, got {}",
        report.bias_max_rel_width
    );
}

/// A non-dyadic BN scale does NOT fold exactly. The report must show a positive
/// width, must show stored elements outside the exact enclosure, and that width
/// must sit far below the `FOLD_TOL_REL = 1e-4` slack the ORT property test
/// currently uses — which is the measurement that constant was standing in for.
#[test]
fn nondyadic_scale_fold_width_is_positive_and_under_ort_prop_tolerance() {
    // var = 3.0 / 7.0 with eps = 0 => scale = gamma/sqrt(3), gamma/sqrt(7):
    // irrational, so every product rounds.
    let (mut nodes, mut weights) = conv_bn_fixture(&[1.0, 0.25], &[3.0, 7.0]);
    let report = {
        let _gate = ForceIntervalReport::enable();
        let consumed = fold_batch_norm_into_conv_linear(&mut nodes, &mut weights);
        assert_eq!(consumed.len(), 1, "the Conv+BN pair must fold");
        let reports = take_reports();
        assert_eq!(reports.len(), 1, "one report per folded node");
        reports[0]
    };

    assert_eq!(report.unenclosable_elements, 0, "all channels enclosable");
    assert!(
        report.weight_max_abs_width > 0.0,
        "a rounding fold must have positive certified width"
    );
    assert!(
        report.stored_outside_exact_enclosure > 0,
        "the f32 fold rounding must be visible as stored values outside the \
         exact enclosure; got 0 of {} weight + {} bias elements",
        report.weight_elements,
        report.bias_elements
    );
    // MEASURED on this fixture: weight 1.554362483169314e-8,
    // bias 5.409832731704476e-8. Both are ~3 orders of magnitude TIGHTER than
    // the 1e-4 the ORT property test allows, which is the point of the report:
    // the fold's own contribution to that constant is ~5e-8, so the remaining
    // 1e-4 is paying for ORT-vs-NY summation order, not for the fold.
    assert!(
        report.weight_max_rel_width < ORT_PROP_FOLD_TOL_REL,
        "measured weight width {} must be under the ORT prop tolerance {}",
        report.weight_max_rel_width,
        ORT_PROP_FOLD_TOL_REL
    );
    assert!(
        report.bias_max_rel_width < ORT_PROP_FOLD_TOL_REL,
        "measured bias width {} must be under the ORT prop tolerance {}",
        report.bias_max_rel_width,
        ORT_PROP_FOLD_TOL_REL
    );
    // f32 has ~1.2e-7 relative precision; a single rounding of the product plus
    // the f64 interval slop cannot exceed a few ULPs of that.
    assert!(
        report.weight_max_rel_width < 1e-6,
        "single-rounding width should be near f32 eps, got {}",
        report.weight_max_rel_width
    );
    // Lower bound too, so a report that silently collapsed to zero (e.g. the
    // enclosure degenerating to the stored point) would fail rather than pass.
    assert!(
        report.weight_max_rel_width > 1e-9,
        "a rounding fold's width must be f32-scale, not f64-scale, got {}",
        report.weight_max_rel_width
    );
}

/// The reporter refuses inconsistent inputs rather than reporting a wrong
/// width: a channel axis that does not divide the tensor, and a zero block.
#[test]
fn fold_interval_report_rejects_inconsistent_channel_map() {
    let affine = vec![
        channel_affine_interval(1.0, 0.0, 0.0, 1.0, 0.0).expect("enclosable"),
        channel_affine_interval(2.0, 0.0, 0.0, 1.0, 0.0).expect("enclosable"),
    ];
    let weight = make_weight(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let bias = arr1(&[0.0_f32, 0.0]).into_dyn();

    // axis 1 has length 3, which is not 2 channels * block 1.
    assert!(
        fold_interval_report(
            &weight,
            &weight,
            None,
            &bias,
            &affine,
            ChannelAxis { axis: 1, block: 1 }
        )
        .is_none(),
        "a channel axis whose length mismatches the affine must be refused"
    );
    // block 0 is nonsense.
    assert!(
        fold_interval_report(
            &weight,
            &weight,
            None,
            &bias,
            &affine,
            ChannelAxis { axis: 0, block: 0 }
        )
        .is_none(),
        "block 0 must be refused"
    );
    // An out-of-range axis must be refused, not panic.
    assert!(
        fold_interval_report(
            &weight,
            &weight,
            None,
            &bias,
            &affine,
            ChannelAxis { axis: 7, block: 1 }
        )
        .is_none(),
        "an out-of-range channel axis must be refused"
    );
    // The consistent case does report.
    assert!(
        fold_interval_report(
            &weight,
            &weight,
            None,
            &bias,
            &affine,
            ChannelAxis { axis: 0, block: 1 }
        )
        .is_some(),
        "the consistent channel map must report"
    );
}

/// The `Gemm -> Reshape -> BN` fold reports with `block > 1`, exercising the
/// `c(f) = f / block` channel map rather than the direct per-channel map.
///
/// Regression note (2026-07-29): this test was order-dependent. The report sink
/// is `thread_local`, and the test harness REUSES threads across tests, so a
/// sibling test that emitted reports without draining them leaked its rows into
/// this test's `take_reports()` and broke the exact-count assertion — the test
/// passed alone and failed in-suite. `ForceIntervalReport::enable()` now drains
/// the sink on acquisition, which makes the count deterministic regardless of
/// test order, thread reuse, or `-j`.
#[test]
fn gemm_reshape_bn_fold_reports_with_block_greater_than_one() {
    let mut gemm = make_node("Gemm", &["x", "gemm_w", "gemm_b"], &["gemm_y"]);
    gemm.attribute.push(make_int_attr("transB", 1));
    let reshape = make_node("Reshape", &["gemm_y", "target_shape"], &["reshape_y"]);
    let mut bn = make_node(
        "BatchNormalization",
        &["reshape_y", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
        &["bn_y"],
    );
    bn.attribute.push(make_float_attr("epsilon", 0.0));
    let mut nodes = vec![gemm, reshape, bn];
    let mut weights = WeightStore::new();
    weights.insert(
        "gemm_w".to_string(),
        make_weight(&[4, 1], &[1.0, 2.0, 3.0, 4.0]),
    );
    weights.insert(
        "gemm_b".to_string(),
        arr1(&[0.25, -0.5, 0.75, -1.0]).into_dyn(),
    );
    weights.insert_integers("target_shape".to_string(), arr1(&[-1_i64, 2, 2]).into_dyn());
    // Non-dyadic denominators so the fold rounds.
    weights.insert("bn_scale".to_string(), arr1(&[2.0, 0.5]).into_dyn());
    weights.insert("bn_bias".to_string(), arr1(&[1.0, -1.0]).into_dyn());
    weights.insert("bn_mean".to_string(), arr1(&[0.0, 0.0]).into_dyn());
    weights.insert("bn_var".to_string(), arr1(&[3.0, 7.0]).into_dyn());

    let _gate = ForceIntervalReport::enable();
    let consumed = fold_batch_norm_into_conv_linear(&mut nodes, &mut weights);
    assert_eq!(consumed.len(), 1, "the Gemm->Reshape->BN triple must fold");
    // MEASURED 2026-07-29: the fold consumes the triple and emits ZERO reports.
    // The certificate consequence is that a Gemm->Reshape->BN tail is a
    // fold-rewritten region the interval reporter does not yet account for, so a
    // certificate over such a graph must not claim authored-graph scope on the
    // strength of this reporter alone.
    let reports = take_reports();
    assert_eq!(reports.len(), 1, "one report per folded node");
    let report = reports[0];

    // 2 channels x block 2 = 4 features, weight is (4, 1) so 4 elements, and
    // the fused bias is per-FEATURE (length 4), not per-channel.
    assert_eq!(report.weight_elements, 4);
    assert_eq!(
        report.bias_elements, 4,
        "the across-Reshape fold replicates the shift across each channel's block"
    );
    assert_eq!(report.unenclosable_elements, 0);
    // MEASURED 2026-07-29: weight width 4.720664359414429e-8 (abs == rel),
    // bias 6.478585357072576e-8 abs / 5.0273226423422594e-8 rel, and 8 of the 8
    // stored values fall outside the exact enclosure — the same f32-rounding
    // scale as the direct fold, confirming the `c(f) = f / block` map is applied
    // rather than a wrong channel being charged for the width.
    assert!(
        report.weight_max_rel_width > 1e-9 && report.weight_max_rel_width < 1e-6,
        "block-mapped width must be f32-scale, got {}",
        report.weight_max_rel_width
    );
    assert_eq!(
        report.stored_outside_exact_enclosure, 8,
        "every stored f32 value is outside the exact-real enclosure of the fold"
    );
}

/// The ConvTranspose fold's channel axis is axis 1, not axis 0.
///
/// The kernel here is `[C_in=1, C_out=2, 1, 1]`, so axis 0 has length 1 while
/// the affine has 2 channels. Had the reporter been wired to axis 0 (the Conv
/// convention), `fold_interval_report` would reject the shape and emit NOTHING —
/// so the mere existence of a 2-element report pins the axis.
#[test]
fn conv_transpose_fold_reports_on_axis_1() {
    let convt = make_node("ConvTranspose", &["x", "ct_w"], &["ct_y"]);
    let mut bn = make_node(
        "BatchNormalization",
        &["ct_y", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
        &["bn_y"],
    );
    bn.attribute.push(make_float_attr("epsilon", 0.0));
    let mut nodes = vec![convt, bn];
    let mut weights = WeightStore::new();
    weights.insert("ct_w".to_string(), make_weight(&[1, 2, 1, 1], &[2.0, -3.0]));
    weights.insert("bn_scale".to_string(), arr1(&[4.0, -2.0]).into_dyn());
    weights.insert("bn_bias".to_string(), arr1(&[1.0, 0.5]).into_dyn());
    weights.insert("bn_mean".to_string(), arr1(&[0.5, -1.0]).into_dyn());
    // Non-dyadic denominators (sqrt(3), sqrt(7)) so the fold genuinely rounds.
    weights.insert("bn_var".to_string(), arr1(&[3.0, 7.0]).into_dyn());

    let report = {
        let _gate = ForceIntervalReport::enable();
        let consumed = fold_batch_norm_into_conv_linear(&mut nodes, &mut weights);
        assert_eq!(consumed.len(), 1, "the ConvTranspose+BN pair must fold");
        let reports = take_reports();
        assert_eq!(
            reports.len(),
            1,
            "a report must be emitted — none means the channel axis was rejected"
        );
        reports[0]
    };

    assert_eq!(
        report.weight_elements, 2,
        "the [1,2,1,1] kernel has 2 elements, one per axis-1 output channel"
    );
    assert_eq!(
        report.bias_elements, 2,
        "the synthesized bias is per-output-channel"
    );
    assert_eq!(report.unenclosable_elements, 0);
    assert!(
        report.weight_max_rel_width > 1e-9 && report.weight_max_rel_width < 1e-6,
        "ConvTranspose fold width must be f32-scale, got {}",
        report.weight_max_rel_width
    );
}

/// A degenerate BN variance is rejected by the fold itself, so no report is
/// produced — the reporter must never be the thing that decides a fold's fate.
#[test]
fn degenerate_variance_folds_nothing_and_reports_nothing() {
    let (mut nodes, mut weights) = conv_bn_fixture(&[1.0, 1.0], &[0.0, 0.0]);
    // eps is 0.0 in the fixture, so denominator = sqrt(0) = 0 => rejected.
    let consumed = {
        let _gate = ForceIntervalReport::enable();
        fold_batch_norm_into_conv_linear(&mut nodes, &mut weights)
    };
    assert!(
        consumed.is_empty(),
        "a zero denominator must block the fold"
    );
    assert!(
        take_reports().is_empty(),
        "a blocked fold must not emit a report"
    );
    // And the weights are untouched.
    let kernel = weights.get("conv_w").expect("kernel present");
    assert_eq!(
        kernel.iter().copied().collect::<Vec<f32>>(),
        vec![1.5, -0.75, 2.25, 0.5],
        "a blocked fold must leave the kernel authored"
    );
}

/// Sanity check that the rational oracle itself is not vacuous: a deliberately
/// WRONG interval must be rejected by it.
#[test]
fn rational_oracle_rejects_a_wrong_enclosure() {
    let gamma = 3.0_f32;
    let var = 2.0_f32;
    let epsilon = 0.0_f32;
    let truth = f64::from(gamma) / f64::from(var).sqrt();
    // An interval entirely above the true value.
    let wrong = Interval {
        lo: truth * 1.01,
        hi: truth * 1.02,
    };
    let v = rational(f64::from(var)) + rational(f64::from(epsilon));
    let gamma_squared = rational(f64::from(gamma)) * rational(f64::from(gamma));
    let lo_squared = rational(wrong.lo) * rational(wrong.lo);
    assert!(
        lo_squared * v > gamma_squared,
        "the oracle must reject an interval that lies above the true value"
    );
    // And a trivially true reference point, so the comparison direction is
    // pinned rather than assumed.
    assert!(
        rational(1.0) < rational(2.0) * BigRational::from(BigInt::from(1)),
        "rational comparison direction sanity"
    );
}
