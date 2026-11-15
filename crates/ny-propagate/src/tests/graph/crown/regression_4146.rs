// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for batched unary fallback behavior.

use crate::*;
use ndarray::{ArrayD, IxDyn};
use ny_core::NyError;

/// LayerNorm epsilon used by both graphs below.
const LAYERNORM_EPS: f32 = 1e-5;

/// Half-width of the radial envelope of a LayerNorm with gamma = 1 and
/// beta = 0 over `n` features. For a centred row `d = x - mean`, the other
/// entries sum to `-d_i`, so Cauchy-Schwarz gives `d_i^2 <= (n-1)/n * |d|^2 =
/// (n-1) var`, hence `|y_i| = |d_i| / sqrt(var + eps) < sqrt(n-1)` at every
/// input. Reported for context only: a bound inside it can still be sound.
fn radial_envelope(n: usize) -> f64 {
    ((n - 1) as f64).sqrt()
}

/// A graph whose output is a LayerNorm (gamma = 1, beta = 0) of rows gathered
/// from the flat input: output element `r * n + m` is
/// `layernorm(x[rows[r][0]], ..., x[rows[r][n-1]])[m]`. Both graphs below
/// have this form (a transpose permutes, a nearest resize repeats).
struct LayerNormRows {
    rows: Vec<Vec<usize>>,
}

impl LayerNormRows {
    fn width(&self) -> usize {
        self.rows[0].len()
    }

    /// Binary64 value of one row and its scale `sqrt(var + eps)`.
    fn row(&self, x: &[f64], r: usize) -> (Vec<f64>, f64) {
        let t: Vec<f64> = self.rows[r].iter().map(|&q| x[q]).collect();
        let n = t.len() as f64;
        let mean = t.iter().sum::<f64>() / n;
        let var = t.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
        let scale = (var + f64::from(LAYERNORM_EPS)).sqrt();
        (t.iter().map(|v| (v - mean) / scale).collect(), scale)
    }

    fn values(&self, x: &[f64]) -> Vec<f64> {
        (0..self.rows.len())
            .flat_map(|r| self.row(x, r).0)
            .collect()
    }

    /// Exact gradient of output `r * n + m` with respect to the flat input:
    /// `dy_m/dt_j = (delta_mj - 1/n - y_m y_j / n) / scale`, summed over the
    /// row positions `j` that read input `q`.
    fn gradient(&self, x: &[f64], r: usize, m: usize) -> Vec<f64> {
        let (y, scale) = self.row(x, r);
        let n = y.len() as f64;
        let mut g = vec![0.0; x.len()];
        for (j, &q) in self.rows[r].iter().enumerate() {
            let delta = if j == m { 1.0 } else { 0.0 };
            g[q] += (delta - 1.0 / n - y[m] * y[j] / n) / scale;
        }
        g
    }
}

/// Deterministic sample of an input box: every corner when there are at most
/// 2^12 of them, otherwise 4096 pseudo-random corners, plus 4096 pseudo-random
/// interior points (xorshift64, fixed seed).
fn box_samples(lower: &[f32], upper: &[f32]) -> Vec<Vec<f32>> {
    let n = lower.len();
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut samples = Vec::new();
    if n <= 12 {
        for mask in 0..(1_u32 << n) {
            samples.push(
                (0..n)
                    .map(|i| {
                        if (mask >> i) & 1 == 0 {
                            lower[i]
                        } else {
                            upper[i]
                        }
                    })
                    .collect(),
            );
        }
    } else {
        for _ in 0..4096 {
            let bits = next();
            samples.push(
                (0..n)
                    .map(|i| {
                        if (bits >> (i % 64)) & 1 == 0 {
                            lower[i]
                        } else {
                            upper[i]
                        }
                    })
                    .collect(),
            );
        }
    }
    for _ in 0..4096 {
        samples.push(
            (0..n)
                .map(|i| {
                    let t = (next() >> 40) as f32 / (1_u32 << 24) as f32;
                    (lower[i] + t * (upper[i] - lower[i])).clamp(lower[i], upper[i])
                })
                .collect(),
        );
    }
    samples
}

/// Projected normalized-gradient ascent of `sign * y_k` from `start` inside
/// the box, halving the step on every non-improving move. Every iterate is a
/// point of the box, so the returned value is a value the network reaches.
fn refine_extreme(
    net: &LayerNormRows,
    lower: &[f64],
    upper: &[f64],
    start: &[f64],
    k: usize,
    sign: f64,
) -> f64 {
    let (r, m) = (k / net.width(), k % net.width());
    let mut x = start.to_vec();
    let mut best = sign * net.values(&x)[k];
    let mut step = (0..x.len())
        .map(|i| upper[i] - lower[i])
        .fold(0.0_f64, f64::max)
        / 4.0;
    for _ in 0..200 {
        let g = net.gradient(&x, r, m);
        let norm = g.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        if norm == 0.0 || step < 1e-12 {
            break;
        }
        let candidate: Vec<f64> = (0..x.len())
            .map(|i| (x[i] + step * sign * g[i] / norm).clamp(lower[i], upper[i]))
            .collect();
        let value = sign * net.values(&candidate)[k];
        if value > best {
            best = value;
            x = candidate;
        } else {
            step *= 0.5;
        }
    }
    sign * best
}

/// Batched CROWN must contain every value the network reaches on the box.
///
/// The test exhibits reachable values and requires CROWN to contain each:
/// the binary64 network value at every sampled point, and the minimum and
/// maximum of every output element refined by projected gradient ascent from
/// the best sample. These are necessary conditions for soundness; a bound
/// tighter than IBP or the radial envelope passes as long as it keeps every
/// reachable value. For the resize graph each output is `±d/sqrt(d^2 + eps)`
/// with `d = (a - b)/2`, monotone in each input, so its extremes are corners
/// and the check there is exact (all 16 corners are sampled).
///
/// This replaces "CROWN contains IBP", a sufficient condition the radial
/// normalization rows of 2026-10-01 (cc4a539a1) falsify while staying sound:
/// both graphs now publish `-sqrt(n-1)` rounded outward, -1.7320511 (3.3e-7
/// below -sqrt(3)) and -2.6457517 (4.0e-7 below -sqrt(7)), against IBP boxes
/// wider by 1.1e-4 and 3.8e-4. Element 0 reaches -0.99998 on the resize graph
/// and -2.62799 on the transpose graph (refined from a sampled -2.47216).
fn assert_crown_contains_reachable_4146(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    crown_bounds: &BoundedTensor,
    ibp_bounds: &BoundedTensor,
    net: &LayerNormRows,
    context: &str,
) {
    assert_eq!(
        crown_bounds.shape(),
        ibp_bounds.shape(),
        "{context}: batched CROWN shape must match IBP"
    );
    let crown_lower: Vec<f32> = crown_bounds.lower().iter().copied().collect();
    let crown_upper: Vec<f32> = crown_bounds.upper().iter().copied().collect();
    let ibp_lower: Vec<f32> = ibp_bounds.lower().iter().copied().collect();
    let ibp_upper: Vec<f32> = ibp_bounds.upper().iter().copied().collect();
    let lower: Vec<f32> = input.lower().iter().copied().collect();
    let upper: Vec<f32> = input.upper().iter().copied().collect();
    let (lower64, upper64): (Vec<f64>, Vec<f64>) = (
        lower.iter().map(|&v| f64::from(v)).collect(),
        upper.iter().map(|&v| f64::from(v)).collect(),
    );
    let outputs = crown_lower.len();

    let samples = box_samples(&lower, &upper);
    let mut lowest = vec![(f64::INFINITY, 0); outputs];
    let mut highest = vec![(f64::NEG_INFINITY, 0); outputs];
    for (sample_index, sample) in samples.iter().enumerate() {
        let x: Vec<f64> = sample.iter().map(|&v| f64::from(v)).collect();
        let values = net.values(&x);
        assert_eq!(values.len(), outputs, "{context}: reference length");
        if sample_index < 16 {
            // The reference must be this graph's function: cross-check it
            // against the graph's own concrete point forward.
            let point = BoundedTensor::concrete(
                ArrayD::from_shape_vec(IxDyn(input.shape()), sample.clone()).unwrap(),
            )
            .unwrap();
            let forward = graph.propagate_concrete_point(&point, None, None).unwrap();
            for (idx, (&got, &want)) in forward.lower().iter().zip(values.iter()).enumerate() {
                assert!(
                    (f64::from(got) - want).abs() <= 1e-4,
                    "{context}: reference[{idx}] {want} disagrees with the graph forward {got}"
                );
            }
        }
        for (idx, &value) in values.iter().enumerate() {
            if value < lowest[idx].0 {
                lowest[idx] = (value, sample_index);
            }
            if value > highest[idx].0 {
                highest[idx] = (value, sample_index);
            }
            assert!(
                f64::from(crown_lower[idx]) <= value && value <= f64::from(crown_upper[idx]),
                "{context}: CROWN [{}, {}] at [{idx}] misses the network value {value} at {sample:?}",
                crown_lower[idx],
                crown_upper[idx]
            );
            assert!(
                f64::from(ibp_lower[idx]) <= value && value <= f64::from(ibp_upper[idx]),
                "{context}: IBP [{}, {}] at [{idx}] misses the network value {value}",
                ibp_lower[idx],
                ibp_upper[idx]
            );
        }
    }

    let start =
        |index: usize| -> Vec<f64> { samples[index].iter().map(|&v| f64::from(v)).collect() };
    let mut refined = (f64::INFINITY, f64::NEG_INFINITY);
    for idx in 0..outputs {
        let min = refine_extreme(net, &lower64, &upper64, &start(lowest[idx].1), idx, -1.0);
        let max = refine_extreme(net, &lower64, &upper64, &start(highest[idx].1), idx, 1.0);
        assert!(
            f64::from(crown_lower[idx]) <= min,
            "{context}: CROWN lower[{idx}] {} exceeds the reachable value {min}",
            crown_lower[idx]
        );
        assert!(
            f64::from(crown_upper[idx]) >= max,
            "{context}: CROWN upper[{idx}] {} is below the reachable value {max}",
            crown_upper[idx]
        );
        if idx == 0 {
            refined = (min, max);
        }
    }
    eprintln!(
        "{context}: element 0 CROWN [{}, {}] IBP [{}, {}] sampled [{}, {}] refined [{}, {}] \
         radial ±{} ({} samples)",
        crown_lower[0],
        crown_upper[0],
        ibp_lower[0],
        ibp_upper[0],
        lowest[0].0,
        highest[0].0,
        refined.0,
        refined.1,
        radial_envelope(net.width()),
        samples.len()
    );
}

/// `resize(2, 2) -> layernorm(4)` on a `[1, 2, 2]` input: nearest resize
/// makes row `i` equal `[a, a, b, b]` with `a = x[i/2][0]`, `b = x[i/2][1]`.
fn resize_layernorm_rows_4146() -> LayerNormRows {
    LayerNormRows {
        rows: (0..4)
            .map(|i| {
                let (a, b) = ((i / 2) * 2, (i / 2) * 2 + 1);
                vec![a, a, b, b]
            })
            .collect(),
    }
}

/// `batched transpose -> layernorm(8)` on a `[1, 8, 3]` input: output row
/// `r` is input column `r`.
fn transpose_layernorm_rows_4146() -> LayerNormRows {
    LayerNormRows {
        rows: (0..3)
            .map(|r| (0..8).map(|k| k * 3 + r).collect())
            .collect(),
    }
}

fn resize_shape_mismatch_input_4146() -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![-0.5_f32, -0.25, 0.0, 0.25]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![0.75_f32, 0.5, 0.4, 0.9]).unwrap(),
    )
    .unwrap()
}

fn assert_resize_direct_shape_mismatch_4146(resize: &ResizeLayer, input: &BoundedTensor) {
    let resize_bounds = BatchedLinearBounds::identity(&[1, 4, 4]).unwrap();
    let err = resize
        .propagate_linear_batched(&resize_bounds, input)
        .expect_err("resize should still emit ShapeMismatch for this reduced-column setup");
    match err {
        NyError::ShapeMismatch { expected, got } => {
            assert_eq!(
                expected,
                vec![16],
                "resize output should flatten to 16 columns"
            );
            assert_eq!(
                got,
                vec![4],
                "layernorm backward should keep only the last axis"
            );
        }
        other => {
            panic!("expected ShapeMismatch from direct resize batched CROWN setup, got {other}")
        }
    }
}

fn build_resize_layernorm_graph_4146(resize: ResizeLayer) -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("resize", Layer::Resize(resize)));
    graph.add_node(GraphNode::new(
        "layernorm",
        Layer::LayerNorm(LayerNormLayer::new_default(4, LAYERNORM_EPS).unwrap()),
        vec!["resize".into()],
    ));
    graph.set_output("layernorm");
    graph
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_batched_crown_transpose_layernorm_succeeds_4146() {
    tests::with_crown_dense_budget_mb("2048", || {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "transpose",
            Layer::Transpose(TransposeLayer::batched_transpose()),
        ));
        graph.add_node(GraphNode::new(
            "layernorm",
            Layer::LayerNorm(LayerNormLayer::new_default(8, LAYERNORM_EPS).unwrap()),
            vec!["transpose".into()],
        ));
        graph.set_output("layernorm");

        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(
                IxDyn(&[1, 8, 3]),
                (0..24).map(|idx| -0.4_f32 + idx as f32 * 0.05).collect(),
            )
            .unwrap(),
            ArrayD::from_shape_vec(
                IxDyn(&[1, 8, 3]),
                (0..24).map(|idx| 0.35_f32 + idx as f32 * 0.05).collect(),
            )
            .unwrap(),
        )
        .unwrap();

        let ibp_bounds = graph.propagate_ibp(&input).unwrap();
        // With multi-dimensional LayerNorm routing, this graph-level
        // batched CROWN call must succeed end-to-end instead of surfacing the
        // old Transpose/LayerNorm routing failure. The stronger Kokoro-shaped
        // Conv1d -> Transpose -> LayerNorm DAG-CROWN provenance contract is
        // covered separately in fallback_reason.rs; this regression only
        // requires batched graph-level success.
        let crown_bounds = graph
            .propagate_crown_batched(&input)
            .expect("#4146 transpose -> layernorm batched CROWN should now succeed");
        assert_crown_contains_reachable_4146(
            &graph,
            &input,
            &crown_bounds,
            &ibp_bounds,
            &transpose_layernorm_rows_4146(),
            "#4146 transpose -> layernorm graph success",
        );
        // Guard against degenerate NaN/Inf bounds that technically "succeed"
        assert!(
            crown_bounds
                .lower()
                .iter()
                .chain(crown_bounds.upper().iter())
                .all(|v| v.is_finite()),
            "#4146 transpose -> layernorm graph-success bounds must be finite"
        );
    });
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_batched_crown_resize_shape_mismatch_falls_back_4146() {
    tests::with_crown_dense_budget_mb("2048", || {
        let resize = ResizeLayer::new(2, 2);
        let input = resize_shape_mismatch_input_4146();
        // Verify Resize still emits ShapeMismatch at the layer level
        // (Fix 1 only changed Transpose; Resize retains ShapeMismatch).
        assert_resize_direct_shape_mismatch_4146(&resize, &input);

        let graph = build_resize_layernorm_graph_4146(resize);

        let ibp_bounds = graph.propagate_ibp(&input).unwrap();
        // With multi-dimensional LayerNorm routing, the graph-level
        // dispatch must catch Resize's ShapeMismatch and return sound fallback
        // bounds end-to-end. Failure here is a real regression.
        let crown_bounds = graph
            .propagate_crown_batched(&input)
            .expect("#4146 resize -> layernorm batched CROWN fallback should now succeed");
        assert_crown_contains_reachable_4146(
            &graph,
            &input,
            &crown_bounds,
            &ibp_bounds,
            &resize_layernorm_rows_4146(),
            "#4146 resize -> layernorm ShapeMismatch fallback",
        );
        // Guard against degenerate NaN/Inf bounds that technically "succeed"
        assert!(
            crown_bounds
                .lower()
                .iter()
                .chain(crown_bounds.upper().iter())
                .all(|v| v.is_finite()),
            "#4146 resize -> layernorm CROWN fallback bounds must be finite"
        );
    });
}
