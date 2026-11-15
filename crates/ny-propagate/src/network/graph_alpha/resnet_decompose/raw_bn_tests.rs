// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::layers::{AddLayer, LinearLayer, ReLULayer};
use crate::network::core::GraphNode;
use ndarray::{arr1, arr2, ArrayD, IxDyn};
use num_rational::BigRational;
use std::time::Duration;

fn exact(v: f32) -> BigRational {
    BigRational::from_float(f64::from(v)).unwrap()
}
fn tensor(shape: &[usize], l: f32, u: f32) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(shape), l),
        ArrayD::from_elem(IxDyn(shape), u),
    )
    .unwrap()
}
fn limits() -> ExtractionLimits {
    ExtractionLimits {
        max_nodes: 16,
        max_caller_bytes: 1 << 20,
    }
}
fn raw_bn() -> BatchNormLayer {
    // sqrt(9)=3, so the authored exact coefficients are rational; no tolerance
    // oracle or rounded recomputation of the original BN is used below.
    BatchNormLayer::new(
        &arr1(&[1., -2., 1.5]).into_dyn(),
        &arr1(&[0.5, 1., -0.25]).into_dyn(),
        &arr1(&[0.25, -0.5, 2.]).into_dyn(),
        &arr1(&[9., 9., 9.]).into_dyn(),
        0.,
    )
    .unwrap()
}
fn get_charge(bn: &BatchNormLayer) -> Option<CertifiedWeightError> {
    charge(
        bn,
        &mut Guard::new(
            Instant::now() + Duration::from_secs(10),
            &mut || true,
            false,
        ),
    )
}

#[test]
fn raw_bn_original_rational_parameters_are_enclosed_per_entry_and_output() {
    let bn = raw_bn();
    let error = get_charge(&bn).unwrap();
    let scale = [
        exact(1.) / exact(3.),
        exact(-2.) / exact(3.),
        exact(1.5) / exact(3.),
    ];
    let bias = [
        exact(0.5) - exact(0.25) * &scale[0],
        exact(1.) - exact(-0.5) * &scale[1],
        exact(-0.25) - exact(2.) * &scale[2],
    ];
    let pre = tensor(&[3, 2, 2], -8., 16.);
    let layer = bn_layer(
        &bn,
        &pre,
        &mut Guard::new(
            Instant::now() + Duration::from_secs(10),
            &mut || true,
            false,
        ),
    )
    .unwrap();
    let GpuCrownLayer::Conv2d {
        weight_col,
        bias_expanded,
        in_w,
        out_w,
        cert_err,
        ..
    } = layer
    else {
        panic!()
    };
    assert_eq!((in_w, out_w), (4, 4));
    assert_eq!(cert_err, error);
    let bias_expanded = bias_expanded.unwrap();
    for c in 0..3 {
        let s = exact(bn.scale[[c]]);
        let b = exact(bn.bias[[c]]);
        let delta_s = &scale[c] - &s;
        let scale_radius =
            exact(error.weight_rel_err) * if s < exact(0.) { -s.clone() } else { s.clone() };
        assert!(delta_s >= -scale_radius.clone() && delta_s <= scale_radius);
        let delta_b = &bias[c] - &b;
        assert!(delta_b >= -exact(error.bias_abs_err) && delta_b <= exact(error.bias_abs_err));
        for j in 0..3 {
            assert_eq!(
                weight_col[c * 3 + j].to_bits(),
                if c == j {
                    bn.scale[[c]].to_bits()
                } else {
                    0f32.to_bits()
                }
            );
        }
        for p in 0..4 {
            assert_eq!(bias_expanded[c * 4 + p].to_bits(), bn.bias[[c]].to_bits());
        }
        for x in [-8., -0.25, 0., 16.] {
            let nominal = &s * exact(x) + &b;
            let truth = &scale[c] * exact(x) + &bias[c];
            let magnitude = if x < 0. { -exact(x) } else { exact(x) };
            let radius = exact(error.weight_rel_err)
                * (if s < exact(0.) { -s.clone() } else { s.clone() })
                * magnitude
                + exact(error.bias_abs_err);
            assert!(truth >= &nominal - &radius && truth <= &nominal + &radius);
        }
    }
}

#[test]
fn raw_bn_zero_subnormal_negative_and_unrepresentable_relative_cases() {
    let mut bn = BatchNormLayer::from_scale_bias(
        arr1(&[0., -0., -2.]).into_dyn(),
        arr1(&[1., 2., 3.]).into_dyn(),
    )
    .unwrap();
    assert!(get_charge(&bn).unwrap().is_exact());
    bn.bias_err[[0]] = 0.125;
    let e = get_charge(&bn).unwrap();
    assert_eq!(e.weight_rel_err.to_bits(), 0f32.to_bits());
    assert_eq!(e.bias_abs_err, 0.125);
    bn.scale_err[[0]] = f32::from_bits(1);
    assert!(get_charge(&bn).is_none()); // zero center/nonzero radius cannot be encoded
    let mut tiny = BatchNormLayer::from_scale_bias(
        arr1(&[f32::from_bits(1)]).into_dyn(),
        arr1(&[0.]).into_dyn(),
    )
    .unwrap();
    tiny.scale_err[[0]] = f32::from_bits(1);
    assert_eq!(get_charge(&tiny).unwrap().weight_rel_err, 1.);
    tiny.scale_err[[0]] = 1.;
    assert!(get_charge(&tiny).is_none()); // finite absolute radius, overflowing relative radius
    tiny.scale[[0]] = 1024.;
    tiny.scale_err[[0]] = f32::from_bits(1);
    assert_eq!(get_charge(&tiny).unwrap().weight_rel_err.to_bits(), 1); // rounded ratio underflows
    tiny.bias_err[[0]] = -0.25;
    assert!(get_charge(&tiny).is_none());
    tiny.bias_err[[0]] = f32::NAN;
    assert!(get_charge(&tiny).is_none());
    tiny.bias_err[[0]] = 0.;
    tiny.scale[[0]] = ny_core::CROWN_COEFF_MAX;
    assert!(get_charge(&tiny).is_none()); // finite transported sentinel
}

fn fixture() -> (GraphNetwork, BoundedTensor, HashMap<String, BoundedTensor>) {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "linear",
        Layer::Linear(
            LinearLayer::new(arr2(&[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]), None).unwrap(),
        ),
    ));
    graph.add_node(GraphNode::new(
        "bn",
        Layer::BatchNorm(raw_bn()),
        vec!["linear".into()],
    ));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["bn".into()],
    ));
    graph.add_node(GraphNode::new(
        "out",
        Layer::Linear(LinearLayer::new(arr2(&[[-1., 2., 0.5]]), Some(arr1(&[0.25]))).unwrap()),
        vec!["relu".into()],
    ));
    graph.set_output("out");
    let input = tensor(&[3], -1., 1.);
    let bounds = graph.collect_node_bounds(&input).unwrap();
    (graph, input, bounds)
}

#[test]
fn raw_bn_chain_reaches_original_input_without_feeding_relu_discharge() {
    let (g, input, b) = fixture();
    assert!(super::super::extract_gpu_segments_with_relu_names_ext(
        &g, &input, "out", &b, &b, None, true, true, false
    )
    .is_none()); // backward BN->Linear has no host ReLU
    let got = extract_charged_raw_bn_chain(
        &g,
        &input,
        "out",
        &b,
        &b,
        None,
        limits(),
        Instant::now() + Duration::from_secs(10),
        &mut || true,
    )
    .unwrap();
    let shape_quote = quote_charged_raw_bn_chain(
        &g,
        &input,
        "out",
        &b,
        &b,
        None,
        limits(),
        Instant::now() + Duration::from_secs(10),
        &mut || true,
    )
    .unwrap();
    assert_eq!(shape_quote, got.quote);
    assert_eq!(
        (got.input_dim, got.output_dim, got.quote.charged_bn_layers),
        (3, 1, 1)
    );
    assert_eq!(got.relu_names, ["relu"]);
    assert_eq!(got.frontier_abs, vec![vec![1.; 3]]);
    assert_eq!(got.node_abs.len(), 1);
    let GpuResnetSegment::Chain(layers) = &got.segments[0] else {
        panic!()
    };
    assert_eq!(layers.len(), 4);
    let GpuCrownLayer::Conv2d { cert_err, .. } = &layers[2] else {
        panic!()
    };
    assert!(!cert_err.is_exact());
    // Ordinary extraction is unchanged; it still cannot silently ignore this BN.
    assert!(super::super::extract_gpu_segments_with_relu_names_ext(
        &g, &input, "out", &b, &b, None, true, false, false
    )
    .is_none());
    let prefix = extract_charged_raw_bn_chain(
        &g,
        &input,
        "linear",
        &b,
        &b,
        None,
        limits(),
        Instant::now() + Duration::from_secs(10),
        &mut || true,
    )
    .unwrap();
    assert_eq!(
        (
            prefix.input_dim,
            prefix.output_dim,
            prefix.quote.charged_bn_layers
        ),
        (3, 3, 0)
    );
}

#[test]
fn raw_bn_chain_payload_edge_missing_bounds_and_nonunary_refuse() {
    let (mut g, input, mut b) = fixture();
    let end = Instant::now() + Duration::from_secs(10);
    let got =
        extract_charged_raw_bn_chain(&g, &input, "out", &b, &b, None, limits(), end, &mut || true)
            .unwrap();
    let exact = ExtractionLimits {
        max_caller_bytes: got.quote.caller_owned_peak_bytes,
        ..limits()
    };
    assert!(
        extract_charged_raw_bn_chain(&g, &input, "out", &b, &b, None, exact, end, &mut || true)
            .is_some()
    );
    assert!(extract_charged_raw_bn_chain(
        &g,
        &input,
        "out",
        &b,
        &b,
        None,
        ExtractionLimits {
            max_caller_bytes: exact.max_caller_bytes - 1,
            ..exact
        },
        end,
        &mut || true
    )
    .is_none());
    assert!(extract_charged_raw_bn_chain(
        &g,
        &input,
        "out",
        &b,
        &b,
        None,
        ExtractionLimits {
            max_nodes: 3,
            ..limits()
        },
        end,
        &mut || true
    )
    .is_none());
    b.remove("bn");
    assert!(extract_charged_raw_bn_chain(
        &g,
        &input,
        "out",
        &b,
        &b,
        None,
        limits(),
        end,
        &mut || true
    )
    .is_none());
    g.add_node(GraphNode::new(
        "sum",
        Layer::Add(AddLayer),
        vec!["linear".into(), "out".into()],
    ));
    assert!(extract_charged_raw_bn_chain(
        &g,
        &input,
        "sum",
        &b,
        &b,
        None,
        limits(),
        end,
        &mut || true
    )
    .is_none());
}

#[test]
fn raw_bn_chain_every_poll_including_final_is_fail_closed_without_publication() {
    let (g, input, b) = fixture();
    let end = Instant::now() + Duration::from_secs(30);
    let mut total = 0usize;
    assert!(extract_charged_raw_bn_chain(
        &g,
        &input,
        "out",
        &b,
        &b,
        None,
        limits(),
        end,
        &mut || {
            total += 1;
            true
        }
    )
    .is_some());
    assert!(total > 10);
    for fail_at in 1..=total {
        let mut calls = 0usize;
        let result = extract_charged_raw_bn_chain(
            &g,
            &input,
            "out",
            &b,
            &b,
            None,
            limits(),
            end,
            &mut || {
                calls += 1;
                calls != fail_at
            },
        );
        assert!(result.is_none(), "poll {fail_at}/{total}");
        assert_eq!(calls, fail_at, "no resurrection after cancellation");
    }
    assert!(extract_charged_raw_bn_chain(
        &g,
        &input,
        "out",
        &b,
        &b,
        None,
        limits(),
        Instant::now(),
        &mut || true
    )
    .is_none());
}

#[test]
fn raw_bn_relu_advice_is_checked_against_original_bounds_without_tolerance() {
    let pre = tensor(&[1], -1., 1.);
    let mut good = GpuCrownLayer::Activation {
        lower_slope: vec![0.5],
        upper_slope: vec![0.5],
        lower_intercept: vec![0.],
        upper_intercept: vec![0.5],
        num_neurons: 1,
    };
    let check = |layer: &GpuCrownLayer| {
        valid_relu(
            layer,
            &pre,
            &mut Guard::new(
                Instant::now() + Duration::from_secs(10),
                &mut || true,
                false,
            ),
        )
    };
    assert!(check(&good).is_some());
    if let GpuCrownLayer::Activation { lower_slope, .. } = &mut good {
        lower_slope[0] = 1.25;
    }
    assert!(check(&good).is_none());
    if let GpuCrownLayer::Activation {
        lower_slope,
        upper_intercept,
        ..
    } = &mut good
    {
        lower_slope[0] = 0.5;
        upper_intercept[0] = ny_tensor::next_down_f32(0.5);
    }
    assert!(check(&good).is_none()); // one-ulp endpoint failure is a real refusal
    if let GpuCrownLayer::Activation {
        upper_slope,
        upper_intercept,
        ..
    } = &mut good
    {
        upper_slope[0] = 1.;
        upper_intercept[0] = 0.;
    }
    assert!(check(&good).is_none()); // stale stable-active mask on a crossing interval
}

fn reset_materialization_report() {
    MATERIALIZATION_FAILURE.with(|slot| slot.set(None));
    MATERIALIZATION_FAILURE_COUNT.with(|count| count.set(0));
}

fn take_materialization_report() -> (Option<MaterializationFailure>, usize) {
    (
        MATERIALIZATION_FAILURE.with(|slot| slot.take()),
        MATERIALIZATION_FAILURE_COUNT.with(|count| count.replace(0)),
    )
}

#[test]
fn materialization_exact_relu_refusals_report_without_descriptor_publication() {
    reset_materialization_report();
    let pre = tensor(&[1], -1., 1.);
    let bad = GpuCrownLayer::Activation {
        lower_slope: vec![0.5],
        upper_slope: vec![0.5],
        lower_intercept: vec![0.],
        upper_intercept: vec![ny_tensor::next_down_f32(0.5)],
        num_neurons: 1,
    };
    // The actual endpoint inequality fails by one binary32 ULP; this is not
    // cancellation or a fabricated diagnostic outcome.
    assert!(exact(0.5) * exact(1.) + exact(ny_tensor::next_down_f32(0.5)) < exact(1.));
    let candidate = {
        let mut poll = || true;
        let mut guard = Guard::new(Instant::now() + Duration::from_secs(30), &mut poll, true);
        guard.node(0, "relu");
        valid_relu(&bad, &pre, &mut guard).map(|()| bad)
    };
    assert!(candidate.is_none());
    assert_eq!(
        take_materialization_report(),
        (
            Some(MaterializationFailure {
                stage: "relu-exact-upper",
                reason: "endpoint-not-enclosed",
                node_ordinal: 0,
                layer_kind: "relu",
            }),
            1
        )
    );

    // Genuine graph extraction with equal (ordinary) invalid alpha must also
    // report the exact lower check, not convert it to a supported dual pair.
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
    graph.set_output("relu");
    let bounds = graph.collect_node_bounds(&pre).unwrap();
    let mut alpha = GraphAlphaState::new();
    alpha.add_relu_node("relu", &pre, false).unwrap();
    let (lower, upper) = alpha.relu_alpha_pair_mut("relu").unwrap();
    lower.fill(1.25);
    upper.fill(1.25);
    assert!(extract_charged_raw_bn_chain(
        &graph,
        &pre,
        "relu",
        &bounds,
        &bounds,
        Some(&alpha),
        limits(),
        Instant::now() + Duration::from_secs(30),
        &mut || true,
    )
    .is_none());
    assert_eq!(
        take_materialization_report(),
        (
            Some(MaterializationFailure {
                stage: "relu-exact-lower",
                reason: "invalid-global-minorant",
                node_ordinal: 0,
                layer_kind: "relu",
            }),
            1
        )
    );
}

#[test]
fn materialization_poll_and_deadline_reports_are_distinct_and_sticky() {
    let (g, input, b) = fixture();
    let end = Instant::now() + Duration::from_secs(60);
    reset_materialization_report();
    let mut total = 0;
    assert!(extract_charged_raw_bn_chain(
        &g,
        &input,
        "out",
        &b,
        &b,
        None,
        limits(),
        end,
        &mut || {
            total += 1;
            true
        },
    )
    .is_some());
    assert_eq!(take_materialization_report(), (None, 0));
    for fail_at in 1..=total {
        let mut calls = 0;
        assert!(extract_charged_raw_bn_chain(
            &g,
            &input,
            "out",
            &b,
            &b,
            None,
            limits(),
            end,
            &mut || {
                calls += 1;
                calls != fail_at
            },
        )
        .is_none());
        assert_eq!(
            calls, fail_at,
            "no added poll or resurrection after refusal"
        );
        let (report, count) = take_materialization_report();
        let report = report.unwrap();
        assert_eq!(count, 1);
        assert_eq!(report.reason, "poll");
        assert!(report.node_ordinal <= MAX_NODES);
        if fail_at == total {
            assert_eq!(report.stage, "final-guard");
        }
    }
    let mut polls = 0;
    assert!(extract_charged_raw_bn_chain(
        &g,
        &input,
        "out",
        &b,
        &b,
        None,
        limits(),
        Instant::now(),
        &mut || {
            polls += 1;
            true
        },
    )
    .is_none());
    assert_eq!(polls, 0);
    assert_eq!(
        take_materialization_report(),
        (
            Some(MaterializationFailure {
                stage: "entry",
                reason: "deadline",
                node_ordinal: MAX_NODES,
                layer_kind: "none",
            }),
            1
        )
    );
}

#[test]
fn materialization_source_and_bn_refusals_keep_bounded_node_identity() {
    reset_materialization_report();
    let (mut g, input, b) = fixture();
    let Layer::BatchNorm(bn) = &mut g.nodes.get_mut("bn").unwrap().layer else {
        panic!()
    };
    bn.scale[[0]] = 0.;
    bn.scale_err[[0]] = f32::from_bits(1);
    // Relative-error authority genuinely cannot encode a nonzero absolute
    // radius around zero. The original bounds need not be consumed because
    // this source check refuses before any descriptor can be returned.
    assert!(extract_charged_raw_bn_chain(
        &g,
        &input,
        "out",
        &b,
        &b,
        None,
        limits(),
        Instant::now() + Duration::from_secs(30),
        &mut || true,
    )
    .is_none());
    assert_eq!(
        take_materialization_report(),
        (
            Some(MaterializationFailure {
                stage: "raw-bn-charge",
                reason: "zero-scale-with-error",
                node_ordinal: 2,
                layer_kind: "batch-norm",
            }),
            1
        )
    );

    let (mut g, input, b) = fixture();
    let Layer::Linear(out) = &mut g.nodes.get_mut("out").unwrap().layer else {
        panic!()
    };
    *out = LinearLayer::new(
        arr2(&[[ny_core::CROWN_COEFF_MAX, 2., 0.5]]),
        Some(arr1(&[0.25])),
    )
    .unwrap();
    assert!(extract_charged_raw_bn_chain(
        &g,
        &input,
        "out",
        &b,
        &b,
        None,
        limits(),
        Instant::now() + Duration::from_secs(30),
        &mut || true,
    )
    .is_none());
    assert_eq!(
        take_materialization_report(),
        (
            Some(MaterializationFailure {
                stage: "source-layout-values",
                reason: "unsafe-value",
                node_ordinal: 0,
                layer_kind: "linear",
            }),
            1
        )
    );
}

#[test]
fn materialization_success_and_quote_are_silent_with_actual_fixed_stack_bound() {
    reset_materialization_report();
    assert!(fixed_stack_bytes().unwrap() <= 8192);
    assert!(size_of::<Guard<'_>>() <= 256);
    let (g, input, b) = fixture();
    let end = Instant::now() + Duration::from_secs(30);
    let quote =
        quote_charged_raw_bn_chain(&g, &input, "out", &b, &b, None, limits(), end, &mut || true)
            .unwrap();
    let chain =
        extract_charged_raw_bn_chain(&g, &input, "out", &b, &b, None, limits(), end, &mut || true)
            .unwrap();
    assert_eq!(chain.quote, quote);
    assert_eq!(chain.relu_names, ["relu"]);
    assert_eq!(take_materialization_report(), (None, 0));
    // Even real quote failure/cancellation emits no materialization report.
    assert!(quote_charged_raw_bn_chain(
        &g,
        &input,
        "out",
        &b,
        &b,
        None,
        ExtractionLimits {
            max_nodes: 1,
            ..limits()
        },
        end,
        &mut || true,
    )
    .is_none());
    assert!(quote_charged_raw_bn_chain(
        &g,
        &input,
        "out",
        &b,
        &b,
        None,
        limits(),
        end,
        &mut || false,
    )
    .is_none());
    assert_eq!(take_materialization_report(), (None, 0));
}
