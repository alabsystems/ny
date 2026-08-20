// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! The star path against a REAL VNN-COMP row: ACAS Xu property 2 on the two networks NY
//! currently misses (`3_3` and `4_2`), where five other tools answer `unsat` in 5-110s and
//! NY returns `unknown` at ~111s.
//!
//! These are candidate-only measurements — see `StarUnsafeCandidateVerdict`. They say what
//! the search can REACH, not what NY may claim.

use std::time::{Duration, Instant};

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_tensor::zonotope::Star;

use super::star_verify::{
    star_search_unsafe_conjunction, StarBudget, StarLayer, StarUnsafeCandidateVerdict,
    StarUnsafeConjunction, StarUnsafeRow,
};

/// Load an exported ACAS Xu network (mean-subtraction already folded into layer 1).
fn load_net(tag: &str) -> Vec<StarLayer> {
    let path = format!("{}/corpus/acasxu/{tag}.json", env!("CARGO_MANIFEST_DIR"));
    let raw = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    // Minimal parse: the fixture is a flat {"weights": [[[..]]], "biases": [[..]]}.
    let v: serde_json::Value = serde_json::from_str(&raw).expect("fixture json");
    let ws = v["weights"].as_array().expect("weights");
    let bs = v["biases"].as_array().expect("biases");
    let mut layers = Vec::new();
    for (li, (w, b)) in ws.iter().zip(bs).enumerate() {
        let rows: Vec<Vec<f32>> = w
            .as_array()
            .expect("w rows")
            .iter()
            .map(|r| {
                r.as_array()
                    .expect("row")
                    .iter()
                    .map(|x| x.as_f64().expect("f64") as f32)
                    .collect()
            })
            .collect();
        let (nr, nc) = (rows.len(), rows[0].len());
        let flat: Vec<f32> = rows.into_iter().flatten().collect();
        let weight = Array2::from_shape_vec((nr, nc), flat).expect("weight shape");
        let bias: Array1<f32> = Array1::from(
            b.as_array()
                .expect("b")
                .iter()
                .map(|x| x.as_f64().expect("f64") as f32)
                .collect::<Vec<_>>(),
        );
        layers.push(StarLayer::Gemm {
            weight,
            bias: Some(bias),
        });
        if li + 1 < ws.len() {
            layers.push(StarLayer::Relu);
        }
    }
    layers
}

/// prop_2's input box, in the network's normalised coordinates.
const LO: [f32; 5] = [0.6, -0.5, -0.5, 0.45, -0.5];
// The upper bound is quoted at the VNN-COMP row's precision; truncating it would move the
// box, so keep every digit even though `f32` cannot hold them all.
#[allow(clippy::excessive_precision)]
const HI: [f32; 5] = [0.679_857_77, 0.5, 0.5, 0.5, -0.45];

fn input_star() -> Star {
    let center: Vec<f32> = (0..5).map(|i| LO[i].midpoint(HI[i])).collect();
    let radius: Vec<f32> = (0..5).map(|i| 0.5 * (HI[i] - LO[i])).collect();
    // One symbol per input, each scaled to its own half-width.
    let mut coeffs = vec![0.0f32; 6 * 5];
    coeffs[..5].copy_from_slice(&center);
    for i in 0..5 {
        coeffs[(i + 1) * 5 + i] = radius[i];
    }
    let arr = ArrayD::from_shape_vec(IxDyn(&[6, 5]), coeffs).expect("shape");
    Star::from_zonotope(ny_tensor::zonotope::ZonotopeTensor::new(arr).expect("zono"))
}

/// prop_2 is UNSAFE when COC (`Y_0`) is maximal: `Y_i - Y_0 <= 0` for i = 1..4, all at once.
fn unsafe_region() -> StarUnsafeConjunction {
    let rows = (1..5)
        .map(|i| {
            let mut c = vec![0.0f64; 5];
            c[i] = 1.0;
            c[0] = -1.0;
            StarUnsafeRow {
                coefficients: c,
                threshold: 0.0,
                strict: false,
            }
        })
        .collect();
    StarUnsafeConjunction { rows }
}

fn run(
    tag: &str,
    input_split: bool,
    secs: u64,
) -> (
    StarUnsafeCandidateVerdict,
    super::star_verify::StarStats,
    Duration,
) {
    let layers = load_net(tag);
    let mut b = StarBudget::new(5_000_000, 4096, Instant::now() + Duration::from_secs(secs));
    b.dual_iters = 0;
    b.prefer_input_split = input_split;
    b.exact_below_unstable = std::env::var("ACASXU_EXACT_BELOW")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let t = Instant::now();
    let (v, s) = star_search_unsafe_conjunction(&layers, &input_star(), &unsafe_region(), &b)
        .expect("search");
    (v, s, t.elapsed())
}

/// Bounded real-ACAS search smoke. `NY_FULL_MEASUREMENTS=1` restores both hard
/// fixtures and the historical 60-second default measurement budget.
#[test]
fn acasxu_prop2_input_vs_neuron_branching() {
    let full = std::env::var("NY_FULL_MEASUREMENTS").as_deref() == Ok("1");
    let only_input = std::env::var("ACASXU_INPUT_ONLY").as_deref() == Ok("1");
    let tags: &[&str] = if full { &["3_3", "4_2"] } else { &["3_3"] };
    for &tag in tags {
        for (label, input_split) in [("neuron", false), ("INPUT", true)] {
            if only_input && !input_split {
                continue;
            }
            let secs: u64 = std::env::var("ACASXU_SECS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(if full { 60 } else { 1 });
            let (v, s, dt) = run(tag, input_split, secs);
            assert!(s.popped > 0, "{tag} {label} must enter the search");
            println!(
                "{tag} {label:6}: {v:?} in {dt:?} | popped {} bisections {} DISCHARGED {} star-tail {} lp-empty {} lp {}",
                s.popped, s.input_bisections, s.discharged_by_overapprox, s.star_tail_hits, s.pruned_infeasible, s.exact_lp_calls
            );
            if input_split {
                for (d, (pop, dis)) in s.depth_histogram.iter().enumerate() {
                    if *pop >= 20 {
                        println!(
                            "    depth {d:3}: popped {pop:6} discharged {dis:6} rate {:.1}%",
                            100.0 * (*dis as f64) / (*pop as f64)
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn acasxu_fixture_loads_with_the_expected_shape() {
    let layers = load_net("3_3");
    // 7 Gemm + 6 Relu, 5 inputs -> 5 outputs.
    assert_eq!(layers.len(), 13, "6 hidden blocks plus the output layer");
    match &layers[0] {
        StarLayer::Gemm { weight, .. } => {
            assert_eq!(weight.shape(), &[50, 5], "first layer maps 5 inputs to 50");
        }
        other => panic!("expected Gemm first, got {other:?}"),
    }
    let star = input_star();
    assert_eq!(star.alpha_dim(), 5, "one symbol per input dimension");
    let b = star.interval_bounds().expect("bounds");
    for i in 0..5 {
        assert!((b.lower()[[i]] - LO[i]).abs() < 1e-5, "lo {i}");
        assert!((b.upper()[[i]] - HI[i]).abs() < 1e-5, "hi {i}");
    }
}
