// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #cgan-batched-stack: equivalence + soundness tests for the domain-stacked
//! dense-spec batched rebound (`input_split.stacked_rebound`).
//!
//! The stacked kernel concatenates the active domains' spec rows into one
//! Conv2d / ConvTranspose2d / BatchNorm backward call and (with
//! `ibp_enhancement`) intersects fresh per-subdomain IBP into each domain's
//! intermediate-bound cache. These tests pin:
//! 1. equivalence with the sequential per-domain kernel when the caches are
//!    identical (shared root reference bounds, no refresh) — the stacking is
//!    pure row bookkeeping and must reproduce the same bounds up to GEMM
//!    blocking roundoff;
//! 2. soundness of the stacked + refreshed bounds against sampled concrete
//!    network evaluations on a conv/BN/convT generator-shaped graph (the exact
//!    layer mix that hits the stacked whitelist on cgan);
//! 3. fail-closed behavior: bit-identical results on graphs with no
//!    whitelisted layer.

use ndarray::ArrayD;

use super::multi_objective_parity::build_multi_objective_child_parity_graph;
use super::*;
use crate::layers::{BatchNormLayer, Conv2dLayer, ConvTranspose2dLayer, FlattenLayer};

/// cgan-shaped micro-graph: Conv2d -> BatchNorm -> ReLU -> ConvTranspose2d ->
/// Flatten -> Linear. Input [2, 4, 4] (32 dims), output 2 dims.
fn build_conv_bn_convt_graph() -> GraphNetwork {
    // Deterministic pseudo-random weights (LCG) with mixed signs so ReLU is
    // genuinely unstable on the test boxes.
    let mut state = 0x9e3779b9_u64;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Map to [-0.6, 0.6].
        (((state >> 33) as f64 / u32::MAX as f64) as f32 - 0.5) * 1.2
    };

    let conv_kernel = ArrayD::from_shape_fn(ndarray::IxDyn(&[3, 2, 3, 3]), |_| next());
    let conv = Conv2dLayer::new(
        conv_kernel,
        Some(arr1(&[0.05_f32, -0.02, 0.01])),
        (1, 1),
        (0, 0),
    )
    .expect("valid conv2d");

    let bn = BatchNormLayer::new(
        &arr1(&[1.0_f32, 0.8, 1.2]).into_dyn(),
        &arr1(&[0.05_f32, -0.1, 0.0]).into_dyn(),
        &arr1(&[0.1_f32, 0.0, -0.1]).into_dyn(),
        &arr1(&[1.0_f32, 0.5, 2.0]).into_dyn(),
        1e-5,
    )
    .expect("valid batchnorm");

    let convt_kernel = ArrayD::from_shape_fn(ndarray::IxDyn(&[3, 2, 2, 2]), |_| next());
    let convt =
        ConvTranspose2dLayer::new(convt_kernel, Some(arr1(&[0.01_f32, -0.03])), (1, 1), (0, 0))
            .expect("valid convtranspose2d");

    // ConvT output: [2, 3, 3] -> 18 dims -> Linear to 2.
    let fc_w = Array2::from_shape_fn((2, 18), |_| next());
    let fc = LinearLayer::new(fc_w, Some(arr1(&[0.02_f32, -0.01]))).expect("valid fc");

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("conv1", Layer::Conv2d(conv)));
    graph.add_node(GraphNode::new(
        "bn1",
        Layer::BatchNorm(bn),
        vec!["conv1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["bn1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "convt1",
        Layer::ConvTranspose2d(convt),
        vec!["relu1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "flat1",
        Layer::Flatten(FlattenLayer::new(0)),
        vec!["convt1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "fc",
        Layer::Linear(fc),
        vec!["flat1".to_string()],
    ));
    graph.set_output("fc");
    graph
}

fn root_box() -> BoundedTensor {
    let lower = ArrayD::from_shape_fn(ndarray::IxDyn(&[2, 4, 4]), |idx| {
        -0.4 - 0.01 * (idx[2] as f32)
    });
    let upper = ArrayD::from_shape_fn(ndarray::IxDyn(&[2, 4, 4]), |idx| {
        0.4 + 0.01 * (idx[1] as f32)
    });
    BoundedTensor::new(lower, upper).expect("valid root box")
}

/// Four sub-boxes of the root box (as input-split children of depth 2).
fn sub_domains() -> Vec<BoundedTensor> {
    let root = root_box();
    (0..4usize)
        .map(|k| {
            let f_lo = 0.15 * k as f32; // 0.0, 0.15, 0.30, 0.45
            let f_hi = f_lo + 0.4; // covers 40% of the root width
            let lower = ndarray::Zip::from(root.lower())
                .and(root.upper())
                .map_collect(|&l, &u| l + f_lo * (u - l));
            let upper = ndarray::Zip::from(root.lower())
                .and(root.upper())
                .map_collect(|&l, &u| l + f_hi * (u - l));
            BoundedTensor::new(lower, upper).expect("valid sub-domain")
        })
        .collect()
}

fn spec() -> Array2<f32> {
    arr2(&[[1.0_f32, -0.5], [-0.7, 1.0]])
}

#[allow(clippy::too_many_arguments)]
fn run_batched(
    graph: &GraphNetwork,
    domains: &[BoundedTensor],
    base: Option<&HashMap<String, BoundedTensor>>,
    ibp_enhancement: bool,
    stacked_rebound: bool,
) -> BatchedSpecBounds {
    let refs: Vec<&BoundedTensor> = domains.iter().collect();
    compute_crown_or_ibp_bounds_batched_specs(
        graph,
        &refs,
        &spec(),
        None,
        base,
        None,
        None,
        None,
        None,
        ibp_enhancement,
        stacked_rebound,
    )
    .expect("batched dense-spec kernel should succeed")
}

/// 1. Stacked kernel == sequential per-domain kernel when the per-domain
///    caches are identical (shared root reference bounds, refresh off). The
///    stacked call is pure row bookkeeping through the same dispatch; only GEMM
///    blocking may differ, so compare with a tight tolerance.
#[test]
fn stacked_rebound_matches_sequential_with_shared_caches() {
    let graph = build_conv_bn_convt_graph();
    let base = graph
        .collect_node_bounds(&root_box())
        .expect("root IBP collection");
    let domains = sub_domains();

    let sequential = run_batched(&graph, &domains, Some(&base), false, false);
    let stacked = run_batched(&graph, &domains, Some(&base), false, true);

    assert_eq!(sequential.bounds.len(), domains.len());
    assert_eq!(stacked.bounds.len(), domains.len());
    for (d, (seq, stk)) in sequential
        .bounds
        .iter()
        .zip(stacked.bounds.iter())
        .enumerate()
    {
        let seq = seq.flatten();
        let stk = stk.flatten();
        for (i, (a, b)) in seq.lower().iter().zip(stk.lower().iter()).enumerate() {
            assert!(
                (a - b).abs() <= 1e-4 + 1e-4 * a.abs().max(b.abs()),
                "domain {d} lower[{i}] diverged: sequential={a}, stacked={b}"
            );
        }
        for (i, (a, b)) in seq.upper().iter().zip(stk.upper().iter()).enumerate() {
            assert!(
                (a - b).abs() <= 1e-4 + 1e-4 * a.abs().max(b.abs()),
                "domain {d} upper[{i}] diverged: sequential={a}, stacked={b}"
            );
        }
    }
}

/// Interval evaluation of `spec_row · [yl, yu]`.
fn spec_row_interval(row: ndarray::ArrayView1<'_, f32>, yl: &[f32], yu: &[f32]) -> (f32, f32) {
    let mut lo = 0.0_f32;
    let mut hi = 0.0_f32;
    for (j, &c) in row.iter().enumerate() {
        if c >= 0.0 {
            lo += c * yl[j];
            hi += c * yu[j];
        } else {
            lo += c * yu[j];
            hi += c * yl[j];
        }
    }
    (lo, hi)
}

/// Check that every domain's reported spec bounds contain the concrete network
/// outputs at sampled interior points (point-box IBP gives an enclosure of the
/// true evaluation; the spec interval over it must lie inside the claimed
/// bounds up to a small evaluation-roundoff epsilon).
fn assert_sound_on_samples(
    graph: &GraphNetwork,
    domains: &[BoundedTensor],
    result: &BatchedSpecBounds,
) {
    let spec = spec();
    let mut state = 0x51ed270b_u64;
    let mut next01 = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as f64 / u32::MAX as f64) as f32
    };

    for (d, dom) in domains.iter().enumerate() {
        let claimed = result.bounds[d].flatten();
        let cl = claimed.lower().as_slice().expect("flat lower").to_vec();
        let cu = claimed.upper().as_slice().expect("flat upper").to_vec();
        for sample in 0..12 {
            let point = ndarray::Zip::from(dom.lower())
                .and(dom.upper())
                .map_collect(|&l, &u| {
                    let t = if sample == 0 {
                        0.0
                    } else if sample == 1 {
                        1.0
                    } else {
                        next01()
                    };
                    l + t * (u - l)
                });
            let point_box = BoundedTensor::new(point.clone(), point).expect("valid point box");
            let node_bounds = graph
                .collect_node_bounds(&point_box)
                .expect("point-box IBP");
            let y = node_bounds.get("fc").expect("output node bounds").flatten();
            let yl: Vec<f32> = y.lower().iter().copied().collect();
            let yu: Vec<f32> = y.upper().iter().copied().collect();
            for r in 0..spec.nrows() {
                let (lo, hi) = spec_row_interval(spec.row(r), &yl, &yu);
                assert!(
                    cl[r] <= lo + 1e-3,
                    "UNSOUND: domain {d} sample {sample} spec row {r}: claimed lower {} > concrete {}",
                    cl[r],
                    lo
                );
                assert!(
                    cu[r] >= hi - 1e-3,
                    "UNSOUND: domain {d} sample {sample} spec row {r}: claimed upper {} < concrete {}",
                    cu[r],
                    hi
                );
            }
        }
    }
}

/// 2. Stacked + per-domain IBP refresh stays sound against concrete network
///    evaluations, and does not blow up the bounds relative to the historical
///    kernel (the refresh intersects — it can only tighten intermediate caches;
///    the hulled discharge is bounded by the shared root cache the historical
///    kernel used).
#[test]
fn stacked_rebound_with_ibp_refresh_is_sound_and_not_looser() {
    let graph = build_conv_bn_convt_graph();
    let base = graph
        .collect_node_bounds(&root_box())
        .expect("root IBP collection");
    let domains = sub_domains();

    let baseline = run_batched(&graph, &domains, Some(&base), false, false);
    let refreshed = run_batched(&graph, &domains, Some(&base), true, true);

    assert_sound_on_samples(&graph, &domains, &baseline);
    assert_sound_on_samples(&graph, &domains, &refreshed);

    // Aggregate non-regression: the refresh re-anchors relaxations on
    // intersected (tighter-or-equal) caches, so total width must not regress
    // meaningfully. Generous slack guards against alpha-heuristic flips
    // without masking a real hull/stacking bug (which inflates widths by the
    // root-vs-subdomain gap, orders of magnitude larger).
    let width = |b: &BatchedSpecBounds| -> f64 {
        b.bounds
            .iter()
            .map(|bt| {
                let f = bt.flatten();
                f.upper()
                    .iter()
                    .zip(f.lower().iter())
                    .map(|(u, l)| (u - l) as f64)
                    .sum::<f64>()
            })
            .sum()
    };
    let w_base = width(&baseline);
    let w_refr = width(&refreshed);
    assert!(
        w_refr <= w_base * 1.10 + 1e-6,
        "stacked+refresh total width regressed: baseline={w_base}, refreshed={w_refr}"
    );

    // The refresh must actually produce per-domain caches (domain-varying
    // relaxations): with 4 different sub-boxes the refreshed per-domain bounds
    // must not all be pinned to identical values on every spec row.
    let first = refreshed.bounds[0].flatten();
    let all_identical = refreshed.bounds.iter().skip(1).all(|b| {
        let f = b.flatten();
        f.lower()
            .iter()
            .zip(first.lower().iter())
            .all(|(a, b)| a == b)
            && f.upper()
                .iter()
                .zip(first.upper().iter())
                .all(|(a, b)| a == b)
    });
    assert!(
        !all_identical,
        "refreshed per-domain bounds are bitwise identical across 4 distinct sub-boxes"
    );
}

/// 3. Fail-closed: on a graph with no whitelisted layer (pure Linear/ReLU),
///    enabling `stacked_rebound` must be byte-identical to the historical kernel
///    (the whitelist never fires; the refresh is off without `ibp_enhancement`).
#[test]
fn stacked_rebound_is_identity_without_whitelisted_layers() {
    let graph = build_multi_objective_child_parity_graph();
    let root = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .expect("valid root");
    let base = graph.collect_node_bounds(&root).expect("root IBP");
    let domains = [
        BoundedTensor::new(
            arr1(&[-0.5_f32, -0.25]).into_dyn(),
            arr1(&[0.25_f32, 0.5]).into_dyn(),
        )
        .expect("d0"),
        BoundedTensor::new(
            arr1(&[0.0_f32, -1.0]).into_dyn(),
            arr1(&[1.0_f32, 0.0]).into_dyn(),
        )
        .expect("d1"),
    ];

    let spec_matrix = arr2(&[[1.0_f32, -0.35], [-0.6, 1.0]]);
    let run = |stacked: bool| -> BatchedSpecBounds {
        let refs: Vec<&BoundedTensor> = domains.iter().collect();
        compute_crown_or_ibp_bounds_batched_specs(
            &graph,
            &refs,
            &spec_matrix,
            None,
            Some(&base),
            None,
            None,
            None,
            None,
            false,
            stacked,
        )
        .expect("batched kernel should succeed")
    };

    let off = run(false);
    let on = run(true);
    for (d, (a, b)) in off.bounds.iter().zip(on.bounds.iter()).enumerate() {
        let a = a.flatten();
        let b = b.flatten();
        assert_eq!(
            a.lower().as_slice(),
            b.lower().as_slice(),
            "domain {d}: lower bounds must be bit-identical when no whitelisted layer exists"
        );
        assert_eq!(
            a.upper().as_slice(),
            b.upper().as_slice(),
            "domain {d}: upper bounds must be bit-identical when no whitelisted layer exists"
        );
    }
}
