// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::component::{
    cosine_distance_from_encoder_bounds, scalar_component_ibp_bounds, CosineDistanceResult,
};
use super::*;
use ny_propagate::bounds::AlphaCrownConfig;

/// Run alpha-CROWN on a scalar component graph and extract output bounds.
/// Falls back to `(fallback_lower, fallback_upper)` on error.
fn alpha_crown_scalar_output(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    config: &AlphaCrownConfig,
    label: &str,
    fallback: (f32, f32),
) -> (f32, f32) {
    let result = graph.collect_alpha_crown_bounds_dag(input, config);
    match &result {
        Ok((bounds, _alpha_state)) => {
            let output = bounds
                .get(graph.output_name())
                .unwrap_or_else(|| panic!("{label}: alpha-CROWN missing output node bounds"));
            let flat = output.flatten();
            assert_eq!(flat.lower().len(), 1, "{label}: output should be scalar");
            let (lower, upper) = (flat.lower()[0], flat.upper()[0]);
            eprintln!("alpha-CROWN {label}: [{lower}, {upper}]");
            (lower, upper)
        }
        Err(e) => {
            eprintln!("alpha-CROWN {label}: failed ({e}), using fallback");
            fallback
        }
    }
}

/// Compare alpha-CROWN vs IBP baseline for cosine distance components.
fn log_alpha_vs_ibp(
    ibp_r: &CosineDistanceResult,
    dot: (f32, f32),
    nsq: (f32, f32),
    alpha_dist: f32,
    alpha_nonvac: bool,
) {
    let dot_pct = width_reduction_pct(ibp_r.dot_lower, ibp_r.dot_upper, dot.0, dot.1);
    let nsq_pct = width_reduction_pct(ibp_r.nsq_lower, ibp_r.nsq_upper, nsq.0, nsq.1);
    eprintln!("alpha vs IBP: dot reduction={dot_pct:.1}%, nsq reduction={nsq_pct:.1}%");
    eprintln!(
        "alpha vs IBP: dot=[{}, {}]→[{}, {}]; nsq=[{}, {}]→[{}, {}]",
        ibp_r.dot_lower,
        ibp_r.dot_upper,
        dot.0,
        dot.1,
        ibp_r.nsq_lower,
        ibp_r.nsq_upper,
        nsq.0,
        nsq.1,
    );
    eprintln!(
        "alpha vs IBP: distance {}→{alpha_dist} (nonvacuous: {}→{alpha_nonvac})",
        ibp_r.distance_upper, ibp_r.nonvacuous,
    );
}

fn width_reduction_pct(old_l: f32, old_u: f32, new_l: f32, new_u: f32) -> f32 {
    let old_w = scalar_width(old_l, old_u);
    let new_w = scalar_width(new_l, new_u);
    if old_w > 0.0 {
        (1.0 - new_w / old_w) * 100.0
    } else {
        0.0
    }
}

fn assert_no_looser_than_ibp(alpha: (f32, f32), ibp: (f32, f32), label: &str) {
    let scale = alpha
        .0
        .abs()
        .max(alpha.1.abs())
        .max(ibp.0.abs())
        .max(ibp.1.abs())
        .max(1.0);
    let tol = 1e-4 * scale;
    assert!(
        alpha.0 >= ibp.0 - tol,
        "{label} lower looser: a={}, i={}",
        alpha.0,
        ibp.0
    );
    assert!(
        alpha.1 <= ibp.1 + tol,
        "{label} upper looser: a={}, i={}",
        alpha.1,
        ibp.1
    );
}

fn alpha_crown_config_with_deadline(deadline_secs: u64) -> AlphaCrownConfig {
    AlphaCrownConfig {
        iterations: 10,
        adaptive_skip: false,
        sparse_ratio: 0.2,
        deadline: Some(Instant::now() + Duration::from_secs(deadline_secs)),
        ..AlphaCrownConfig::default()
    }
}

fn ibp_baseline_distance(
    dot_graph: &GraphNetwork,
    norm_sq_graph: &GraphNetwork,
    input: &BoundedTensor,
    base_graph: &GraphNetwork,
) -> CosineDistanceResult {
    let enc_out = base_graph.output_name().to_string();
    let ibp_bounds = base_graph
        .collect_node_bounds(input)
        .expect("encoder IBP should succeed");
    cosine_distance_from_encoder_bounds(
        dot_graph,
        norm_sq_graph,
        input,
        &ibp_bounds,
        &enc_out,
        "ibp-baseline",
    )
}

fn alpha_crown_distance_components(
    dot_graph: &GraphNetwork,
    norm_sq_graph: &GraphNetwork,
    input: &BoundedTensor,
    ibp_r: &CosineDistanceResult,
    t_start: Instant,
) -> ((f32, f32), (f32, f32)) {
    let dot = alpha_crown_scalar_output(
        dot_graph,
        input,
        &alpha_crown_config_with_deadline(240),
        "dot",
        (ibp_r.dot_lower, ibp_r.dot_upper),
    );
    let remaining = 540u64.saturating_sub(t_start.elapsed().as_secs()).min(180);
    let nsq = alpha_crown_scalar_output(
        norm_sq_graph,
        input,
        &alpha_crown_config_with_deadline(remaining),
        "norm_sq",
        (ibp_r.nsq_lower, ibp_r.nsq_upper),
    );
    (dot, nsq)
}

fn log_alpha_crown_acceptance(
    alpha_nonvac: bool,
    alpha_dist: f32,
    dot_lower: f32,
    t_start: Instant,
) {
    if alpha_nonvac && alpha_dist < shared::SPEAKER_DISTANCE_ACCEPTANCE_UPPER {
        eprintln!("*** #3499 ACCEPTANCE CRITERION MET: distance={alpha_dist} ***");
    } else if !alpha_nonvac {
        eprintln!(
            "alpha-CROWN: still vacuous (dot_lower={} not positive)",
            dot_lower
        );
    }
    eprintln!(
        "alpha-CROWN cosine: total {:.1}s",
        t_start.elapsed().as_secs_f64()
    );
}

/// Epsilon sweep: find the perturbation scale at which IBP produces non-vacuous
/// cosine distance bounds on the 190-node ECAPA-TDNN speaker encoder (#3499).
///
/// Tests epsilons from 1e-3 down to 1e-7 using pure IBP on the dot and norm_sq
/// component graphs. Nested input boxes must produce nested component bounds,
/// and the sweep must reach a non-vacuous distance below the acceptance cap.
#[cfg_attr(not(debug_assertions), ntest::timeout(600000))]
#[test]
#[cfg(feature = "external-avoice")]
fn test_speaker_cosine_ibp_epsilon_sweep_3499() {
    crate::test_fixtures::assert_test_model_available!("speaker_encoder.onnx");
    let t_start = Instant::now();
    let (dot_graph, norm_sq_graph, _) = cosine_head::build_speaker_cosine_component_graphs();
    let model = shared::avoice_speaker_encoder();
    let epsilons: &[f32] = &[1e-3, 1e-4, 1e-5, 1e-6, 1e-7];
    eprintln!(
        "epsilon sweep: graphs built in {:.1}s, testing {} epsilons",
        t_start.elapsed().as_secs_f64(),
        epsilons.len()
    );

    let mut previous = None;
    let mut accepted = None;
    for &eps in epsilons {
        let eps_start = Instant::now();
        let input =
            shared::bounded_speaker_encoder_cosine_input(model, SPEAKER_ENCODER_SEQUENCE_LEN, eps);
        let (dot_l, dot_u) = scalar_component_ibp_bounds(&dot_graph, &input, "dot");
        let (nsq_l, nsq_u) = scalar_component_ibp_bounds(&norm_sq_graph, &input, "norm_sq");
        let (dist, nonvac) = cosine_head::speaker_cosine_distance_upper(dot_l, nsq_u);
        assert!(
            dot_l.is_finite()
                && dot_u.is_finite()
                && nsq_l.is_finite()
                && nsq_u.is_finite()
                && dist.is_finite(),
            "eps={eps:.0e}: component and distance bounds must be finite"
        );
        assert!(dot_l <= dot_u, "eps={eps:.0e}: inverted dot bounds");
        assert!(nsq_l <= nsq_u, "eps={eps:.0e}: inverted norm bounds");
        if let Some((prev_dot_l, prev_dot_u, prev_nsq_l, prev_nsq_u)) = previous {
            let scale = [
                dot_l, dot_u, nsq_l, nsq_u, prev_dot_l, prev_dot_u, prev_nsq_l, prev_nsq_u,
            ]
            .into_iter()
            .fold(1.0_f32, |acc, value| acc.max(value.abs()));
            let tol = 5e-5 * scale;
            assert!(
                dot_l >= prev_dot_l - tol && dot_u <= prev_dot_u + tol,
                "eps={eps:.0e}: shrinking the input box loosened dot bounds from \
                 [{prev_dot_l}, {prev_dot_u}] to [{dot_l}, {dot_u}]"
            );
            assert!(
                nsq_l >= prev_nsq_l - tol && nsq_u <= prev_nsq_u + tol,
                "eps={eps:.0e}: shrinking the input box loosened norm-squared bounds from \
                 [{prev_nsq_l}, {prev_nsq_u}] to [{nsq_l}, {nsq_u}]"
            );
        }
        previous = Some((dot_l, dot_u, nsq_l, nsq_u));
        let tag = if nonvac { "NON-VACUOUS" } else { "vacuous" };
        eprintln!(
            "eps={eps:.0e}: dot=[{dot_l:.6e}, {dot_u:.6e}] nsq=[{nsq_l:.6e}, {nsq_u:.6e}] \
             dist={dist} {tag} ({:.1}s)",
            eps_start.elapsed().as_secs_f64()
        );
        if nonvac && dist < shared::SPEAKER_DISTANCE_ACCEPTANCE_UPPER {
            accepted.get_or_insert((eps, dist));
            eprintln!("*** #3499 ACCEPTANCE at eps={eps:.0e}: distance={dist} ***");
        }
    }
    let (accepted_eps, accepted_distance) = accepted.expect(
        "the epsilon sweep must reach a non-vacuous speaker distance below the acceptance cap",
    );
    eprintln!(
        "epsilon sweep: first acceptance eps={accepted_eps:.0e}, distance={accepted_distance}; \
         total {:.1}s",
        t_start.elapsed().as_secs_f64()
    );
}

/// α-CROWN exploration: optimize ReLU slopes on cosine component graphs (#3499).
/// Uses `collect_alpha_crown_bounds_dag` with `adaptive_skip: false`.
#[cfg_attr(not(debug_assertions), ntest::timeout(600000))]
#[test]
#[cfg(feature = "external-avoice")]
fn test_speaker_cosine_alpha_crown_explores_tighter_distance_3499() {
    crate::test_fixtures::assert_test_model_available!("speaker_encoder.onnx");
    let t_start = Instant::now();
    let (dot_graph, norm_sq_graph, _) = cosine_head::build_speaker_cosine_component_graphs();
    let model = shared::avoice_speaker_encoder();
    let input = shared::bounded_speaker_encoder_cosine_input(
        model,
        SPEAKER_ENCODER_SEQUENCE_LEN,
        shared::SPEAKER_ENCODER_EPSILON,
    );

    let ibp_r = ibp_baseline_distance(
        &dot_graph,
        &norm_sq_graph,
        &input,
        avoice_speaker_encoder_graph(),
    );
    eprintln!(
        "alpha-CROWN: IBP baseline dist={} in {:.1}s",
        ibp_r.distance_upper,
        t_start.elapsed().as_secs_f64()
    );

    let (dot, nsq) =
        alpha_crown_distance_components(&dot_graph, &norm_sq_graph, &input, &ibp_r, t_start);

    let (alpha_dist, alpha_nonvac) = cosine_head::speaker_cosine_distance_upper(dot.0, nsq.1);
    log_alpha_vs_ibp(&ibp_r, dot, nsq, alpha_dist, alpha_nonvac);
    assert_no_looser_than_ibp(dot, (ibp_r.dot_lower, ibp_r.dot_upper), "alpha-CROWN dot");
    log_alpha_crown_acceptance(alpha_nonvac, alpha_dist, dot.0, t_start);
}
