// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::centroid::{centroid_bounds_from_softmax, centroid_monotonicity_gaps};
use super::fixtures::{
    bounded_hidden_states_input, talker_attention_softmax_output_graph_real_rope,
    TALKER_ATTENTION_SEQ_LEN,
};
use super::*;
use ny_propagate::types::BoundsProvenance;

struct CrownBoundaryPoint {
    provenance: BoundsProvenance,
    violations: usize,
    max_gap: f32,
}

fn crown_boundary_point(
    graph: &GraphNetwork,
    softmax_name: &str,
    eps: f32,
    label: &str,
) -> CrownBoundaryPoint {
    let input = bounded_hidden_states_input(TALKER_ATTENTION_SEQ_LEN, eps);
    let crown = graph
        .propagate_crown_with_provenance(&input)
        .expect("real-RoPE backward CROWN should not error at boundary points");
    let (lower, upper, query_seq_len) =
        centroid_bounds_from_softmax(&crown.bounds, &format!("{label} eps={eps:.0e}"));
    let gaps = centroid_monotonicity_gaps(&lower, &upper, query_seq_len);
    let violations = gaps.iter().filter(|&&gap| gap > 1e-4).count();
    let max_gap = gaps.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    println!(
        "{label}: eps={eps:.0e} node={softmax_name} provenance={:?} violations={violations} max_gap={max_gap:.6}",
        crown.provenance
    );

    CrownBoundaryPoint {
        provenance: crown.provenance,
        violations,
        max_gap,
    }
}

/// Boundary points re-measured 2026-07-19 against the 2026-06-03
/// `talker_attention_layer0.onnx` re-export.
///
/// The original phase used `TALKER_ATTENTION_EPSILON` (1e-3) / 2e-3, measured
/// on a pre-June export. The current export carries Qwen3 q/k RmsNorm whose
/// variance floor divides ~eps-scale activations by an ~1e-3 rms, so at
/// eps=1e-3 the QK^T logit IBP spans thousands and no relaxation can certify.
/// Measured on this export (backward CROWN, seq16 real RoPE):
/// - eps=2e-6: provenance=Crown, violations=0, max_gap=-0.1886 (certifies)
/// - eps=3e-6: provenance=Crown, violations=40, max_gap=+0.2000 (fails; IBP
///   critical epsilon is 2.535e-6 and backward CROWN currently tracks the
///   same boundary — the softmax-LSE + Div-concretization looseness
///   dominates, so CROWN's gain over IBP is marginal here)
const CROWN_BOUNDARY_CERT_EPSILON: f32 = 2e-6;
const CROWN_BOUNDARY_BEYOND_EPSILON: f32 = 3e-6;

// Wall-clock watchdog re-calibrated 2026-07-19 to the 2026-06-03 export cost.
// This test runs TWO full backward-CROWN passes over the seq16 real-RoPE talker
// attention graph on the deterministic NaiveCpuGemmEngine; measured uncontended
// (--test-threads=1, release): pass1(2e-6)=88.2s, pass2(3e-6)=78.3s, total 166s.
// The former 120s watchdog was smaller than the honest single-thread cost, so it
// fired before the (passing) assertions could complete. 600s matches the suite
// ceiling used for the equally heavy real-RoPE `crown_ibp_tightening` sweep and
// absorbs parallel-run CPU contention. The verification assertions below are
// unchanged and all pass.
#[cfg_attr(not(debug_assertions), ntest::timeout(600000))]
#[test]
fn test_centroid_monotonicity_real_rope_crown_boundary_3497() {
    crate::test_fixtures::require_test_model_or_skip!("talker_attention_layer0.onnx");
    let (graph, softmax_name) = talker_attention_softmax_output_graph_real_rope();

    let baseline = crown_boundary_point(
        &graph,
        &softmax_name,
        CROWN_BOUNDARY_CERT_EPSILON,
        "real_rope_crown_boundary",
    );
    assert_eq!(
        baseline.provenance,
        BoundsProvenance::Crown,
        "real-RoPE baseline should stay on the backward-CROWN path at epsilon={}",
        CROWN_BOUNDARY_CERT_EPSILON
    );
    assert_eq!(
        baseline.violations, 0,
        "real-RoPE backward CROWN should prove monotonicity at epsilon={}: max_gap={}",
        CROWN_BOUNDARY_CERT_EPSILON, baseline.max_gap
    );
    assert!(
        baseline.max_gap <= 1e-4,
        "real-RoPE backward CROWN baseline max_gap={} should remain below tolerance",
        baseline.max_gap
    );

    let beyond_ibp_bracket = crown_boundary_point(
        &graph,
        &softmax_name,
        CROWN_BOUNDARY_BEYOND_EPSILON,
        "real_rope_crown_boundary",
    );
    match beyond_ibp_bracket.provenance {
        BoundsProvenance::Crown => assert!(
            beyond_ibp_bracket.violations > 0 || beyond_ibp_bracket.max_gap > 1e-4,
            "real-RoPE backward CROWN unexpectedly certifies epsilon={}; update the #3497 report",
            CROWN_BOUNDARY_BEYOND_EPSILON
        ),
        other => {
            println!(
                "real_rope_crown_boundary: eps={CROWN_BOUNDARY_BEYOND_EPSILON:.0e} leaves \
                 backward-CROWN mode with {other:?}"
            )
        }
    }
}
