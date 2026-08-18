// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::component::{cosine_distance_from_encoder_bounds, CosineDistanceResult};
use super::*;

fn assert_crown_ibp_no_looser_than_ibp(ibp: &CosineDistanceResult, cibp: &CosineDistanceResult) {
    let ibp_dot_w = scalar_width(ibp.dot_lower, ibp.dot_upper);
    let cibp_dot_w = scalar_width(cibp.dot_lower, cibp.dot_upper);
    let ibp_nsq_w = scalar_width(ibp.nsq_lower, ibp.nsq_upper);
    let cibp_nsq_w = scalar_width(cibp.nsq_lower, cibp.nsq_upper);
    let dot_scale = ibp_dot_w.max(cibp_dot_w).max(1.0);
    let nsq_scale = ibp_nsq_w.max(cibp_nsq_w).max(1.0);
    assert!(
        cibp_dot_w <= ibp_dot_w + 1e-6 * dot_scale,
        "CROWN-IBP dot width should not exceed IBP: cibp={cibp_dot_w}, ibp={ibp_dot_w}"
    );
    assert!(
        cibp_nsq_w <= ibp_nsq_w + 1e-6 * nsq_scale,
        "CROWN-IBP norm_sq width should not exceed IBP: cibp={cibp_nsq_w}, ibp={ibp_nsq_w}"
    );
}

fn log_cosine_distance_comparison(ibp: &CosineDistanceResult, cibp: &CosineDistanceResult) {
    let ibp_dot_w = scalar_width(ibp.dot_lower, ibp.dot_upper);
    let cibp_dot_w = scalar_width(cibp.dot_lower, cibp.dot_upper);
    let ibp_nsq_w = scalar_width(ibp.nsq_lower, ibp.nsq_upper);
    let cibp_nsq_w = scalar_width(cibp.nsq_lower, cibp.nsq_upper);
    let dot_pct = if ibp_dot_w > 0.0 {
        (1.0 - cibp_dot_w / ibp_dot_w) * 100.0
    } else {
        0.0
    };
    let nsq_pct = if ibp_nsq_w > 0.0 {
        (1.0 - cibp_nsq_w / ibp_nsq_w) * 100.0
    } else {
        0.0
    };
    eprintln!("CROWN-IBP cosine: dot reduction={dot_pct:.1}%, norm_sq reduction={nsq_pct:.1}%");
    eprintln!(
        "CROWN-IBP cosine: vacuous->nonvacuous: {} -> {}",
        !ibp.nonvacuous, cibp.nonvacuous
    );
}

/// Compute adaptive width skip threshold from IBP bounds (#3499).
///
/// Returns the median `max_width()` across all nodes, scaled by a fraction.
/// Nodes with IBP width below this threshold are already tight enough that
/// CROWN backward cannot meaningfully improve them.
fn adaptive_width_threshold(ibp_bounds: &HashMap<String, BoundedTensor>, fraction: f32) -> f32 {
    let mut widths: Vec<f32> = ibp_bounds.values().map(|b| b.max_width()).collect();
    widths.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = if widths.is_empty() {
        0.0
    } else {
        widths[widths.len() / 2]
    };
    let threshold = median * fraction;
    let below = widths.iter().filter(|&&w| w < threshold).count();
    eprintln!(
        "CROWN-IBP width skip: median={:.6}, threshold={:.6} ({}x), {}/{} nodes below",
        median,
        threshold,
        fraction,
        below,
        widths.len(),
    );
    threshold
}

/// Run CROWN-IBP tightening on the encoder and return tightened bounds (#3499).
fn tighten_encoder_bounds(
    base_graph: &GraphNetwork,
    input: &BoundedTensor,
    ibp_bounds: HashMap<String, BoundedTensor>,
    budget_secs: u64,
    min_width_to_tighten: Option<f32>,
    engine: Option<&dyn ny_core::GemmEngine>,
) -> HashMap<String, BoundedTensor> {
    let deadline = Instant::now() + Duration::from_secs(budget_secs);
    let cibp = match (min_width_to_tighten, engine) {
        (Some(threshold), Some(engine)) => base_graph
            .collect_crown_ibp_bounds_dag_with_precomputed_ibp_and_engine_and_width_threshold(
                input,
                ibp_bounds,
                Some(deadline),
                Some(engine),
                threshold,
            )
            .expect("CROWN-IBP tightening with width skip and engine should succeed"),
        (Some(threshold), None) => base_graph
            .collect_crown_ibp_bounds_dag_with_precomputed_ibp_and_width_threshold(
                input,
                ibp_bounds,
                Some(deadline),
                threshold,
            )
            .expect("CROWN-IBP tightening with width skip should succeed"),
        (None, Some(engine)) => base_graph
            .collect_crown_ibp_bounds_dag_with_precomputed_ibp_and_engine(
                input,
                ibp_bounds,
                Some(deadline),
                Some(engine),
            )
            .expect("CROWN-IBP tightening with engine should succeed"),
        (None, None) => base_graph
            .collect_crown_ibp_bounds_dag_with_precomputed_ibp(input, ibp_bounds, Some(deadline))
            .expect("CROWN-IBP tightening should succeed"),
    };
    let tightened = cibp
        .provenance
        .values()
        .filter(|p| matches!(p, BoundsProvenance::Crown))
        .count();
    let skipped = cibp
        .provenance
        .values()
        .filter(|p| {
            matches!(
                p,
                BoundsProvenance::ForwardFallback(
                    ny_propagate::types::CrownIbpFallbackReason::WidthBelowThreshold
                )
            )
        })
        .count();
    // Log fallback reason distribution for diagnostic insight (#3499).
    let mut reason_counts: HashMap<String, usize> = HashMap::new();
    for prov in cibp.provenance.values() {
        let key = match prov {
            BoundsProvenance::Crown => "Crown".to_string(),
            BoundsProvenance::ForwardFallback(r) => format!("{r:?}"),
        };
        *reason_counts.entry(key).or_insert(0) += 1;
    }
    eprintln!(
        "CROWN-IBP cosine: tightened {}/{} nodes, {} skipped (width below threshold)",
        tightened,
        cibp.bounds.len(),
        skipped,
    );
    for (reason, count) in &reason_counts {
        eprintln!("  provenance: {reason} = {count}");
    }
    cibp.bounds
}

fn encoder_ibp_baseline(
    base_graph: &GraphNetwork,
    input: &BoundedTensor,
    dot_graph: &GraphNetwork,
    norm_sq_graph: &GraphNetwork,
    t_start: Instant,
) -> (String, HashMap<String, BoundedTensor>, CosineDistanceResult) {
    let enc_out = base_graph.output_name().to_string();
    let ibp_bounds = base_graph
        .collect_node_bounds(input)
        .expect("encoder IBP should succeed");
    eprintln!(
        "CROWN-IBP cosine: IBP ({} nodes) in {:.1}s",
        ibp_bounds.len(),
        t_start.elapsed().as_secs_f64()
    );

    let ibp_r = cosine_distance_from_encoder_bounds(
        dot_graph,
        norm_sq_graph,
        input,
        &ibp_bounds,
        &enc_out,
        "baseline",
    );
    eprintln!(
        "baseline: dist={} (nonvacuous={}) in {:.1}s",
        ibp_r.distance_upper,
        ibp_r.nonvacuous,
        t_start.elapsed().as_secs_f64()
    );
    (enc_out, ibp_bounds, ibp_r)
}

fn log_crown_ibp_acceptance(result: &CosineDistanceResult) {
    if result.nonvacuous {
        assert!(result.distance_upper.is_finite(), "should be finite");
        eprintln!(
            "NON-VACUOUS distance = {} (acceptance < {})",
            result.distance_upper,
            shared::SPEAKER_DISTANCE_ACCEPTANCE_UPPER
        );
    }
}

/// Per-node CROWN-IBP tightening on the base encoder produces tighter cosine
/// distance bounds than pure IBP intermediates (#3499).
///
/// Uses a deadline to bound wall-clock time: ~5s per CROWN backward pass on
/// the 187-node ECAPA-TDNN encoder, so the deadline controls how many nodes
/// get tightened within the 600s cargo wrapper timeout.
#[cfg_attr(not(debug_assertions), ntest::timeout(600000))]
#[test]
#[cfg(feature = "external-avoice")]
fn test_speaker_cosine_crown_ibp_tightened_intermediates_improve_distance_3499() {
    crate::test_fixtures::assert_test_model_available!("speaker_encoder.onnx");
    let t_start = Instant::now();
    let (dot_graph, norm_sq_graph, _) = cosine_head::build_speaker_cosine_component_graphs();
    let model = shared::avoice_speaker_encoder();
    let input = shared::bounded_speaker_encoder_cosine_input(
        model,
        SPEAKER_ENCODER_SEQUENCE_LEN,
        shared::SPEAKER_ENCODER_EPSILON,
    );
    let base_graph = avoice_speaker_encoder_graph();
    let (enc_out, ibp_bounds, ibp_r) =
        encoder_ibp_baseline(base_graph, &input, &dot_graph, &norm_sq_graph, t_start);

    // Adaptive width threshold: skip CROWN for nodes whose IBP max_width
    // is below 10% of the median.  This saves ~5-7s per skipped node,
    // allowing the budget to reach deeper layers (#3499).
    let skip_threshold = adaptive_width_threshold(&ibp_bounds, 0.1);

    let budget = 300u64.saturating_sub(t_start.elapsed().as_secs());
    let cibp_bounds = tighten_encoder_bounds(
        base_graph,
        &input,
        ibp_bounds,
        budget,
        Some(skip_threshold),
        None,
    );

    let cibp_r = cosine_distance_from_encoder_bounds(
        &dot_graph,
        &norm_sq_graph,
        &input,
        &cibp_bounds,
        &enc_out,
        "crown-ibp",
    );
    eprintln!(
        "crown-ibp: dist={} (nonvacuous={}) in {:.1}s",
        cibp_r.distance_upper,
        cibp_r.nonvacuous,
        t_start.elapsed().as_secs_f64()
    );

    assert_crown_ibp_no_looser_than_ibp(&ibp_r, &cibp_r);
    log_cosine_distance_comparison(&ibp_r, &cibp_r);
    log_crown_ibp_acceptance(&cibp_r);
}
