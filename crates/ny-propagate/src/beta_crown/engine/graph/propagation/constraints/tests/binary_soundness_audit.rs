// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Adversarial soundness audit for binary/n-ary child propagation in the
//! constrained (child-domain beta-CROWN) backward pass (#268e242 follow-up).
//!
//! Commit 268e242 made `process_constrained_backward_node` route binary/n-ary
//! nodes (residual Add, Concat, MinBinary, MaxBinary, MulBinary) through the
//! n-ary dispatch (`apply_constrained_backward_dispatch_result`, which routes
//! coefficients to ALL inputs) instead of erroring. This is soundness-critical:
//! if any input's contribution is dropped or mis-routed, the verifier could emit
//! an UNSOUND "verified" verdict.
//!
//! Strategy: for each graph pattern, run the constrained backward
//! (`propagate_crown_with_graph_constraints` / `_with_spec_matrix`), then densely
//! sample the input box (>= 1000 deterministic points), evaluate the TRUE
//! function by hand (using the exact weights this module constructs, NOT the
//! verifier's own forward — avoiding circularity), and assert
//!     lower - TOL <= f(x) <= upper + TOL
//! at every sample, for every output coordinate / spec row.
//!
//! KEY adversarial check: branch B's weights are chosen so that perturbing
//! input_b genuinely moves the output. A silent identity-on-input[0] bug would
//! drop that contribution, and dense sampling near the box corners where B
//! dominates would expose the unsoundness.

use ndarray::{arr1, arr2, Array1, Array2, ArrayD, IxDyn};

use crate::beta_crown::{GraphCrownContext, GraphSplitHistory};
use crate::{
    AddLayer, BetaCrownConfig, BetaCrownVerifier, BoundedTensor, ConcatLayer, GraphNetwork,
    GraphNode, Layer, LinearLayer, MaxBinaryLayer, MinBinaryLayer, ReLULayer,
};

/// Containment tolerance for float comparisons. Bound arithmetic in the
/// verifier uses directed rounding so it should never under-cover, but we allow
/// a small slack for accumulated f32 error in the hand-computed reference.
const AUDIT_TOL: f32 = 1e-4;

/// Number of densely-sampled points per graph (mission requires >= 1000).
const N_SAMPLES: usize = 2000;

// ---------------------------------------------------------------------------
// Deterministic sampler (NO time / entropy). Splitmix64 -> [0,1) f64 -> f32.
// ---------------------------------------------------------------------------

struct DetRng {
    state: u64,
}

impl DetRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        // splitmix64
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    fn unit(&mut self) -> f32 {
        // Use top 53 bits for an f64 mantissa, then narrow.
        let bits = self.next_u64() >> 11;
        ((bits as f64) / ((1u64 << 53) as f64)) as f32
    }
}

/// Draw a sample inside the box `[lower, upper]`. To stress the corners (where
/// a dropped branch contribution is most visible) we bias the first samples to
/// the box vertices, then fill the interior with pseudo-random points.
fn sample_point(rng: &mut DetRng, lower: &[f32], upper: &[f32], idx: usize) -> Vec<f32> {
    let d = lower.len();
    // First 2^d samples (capped) enumerate the box corners exactly.
    let corner_count = 1usize << d.min(10);
    if idx < corner_count {
        return (0..d)
            .map(|j| {
                if (idx >> j) & 1 == 1 {
                    upper[j]
                } else {
                    lower[j]
                }
            })
            .collect();
    }
    (0..d)
        .map(|j| {
            let t = rng.unit();
            lower[j] + t * (upper[j] - lower[j])
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Hand-rolled reference evaluation primitives (the "truth").
// ---------------------------------------------------------------------------

fn affine(w: &Array2<f32>, b: &Array1<f32>, x: &[f32]) -> Vec<f32> {
    // y = W x + b, W is [out, in].
    let (out, inn) = (w.nrows(), w.ncols());
    assert_eq!(inn, x.len(), "affine input width mismatch");
    let mut y = vec![0.0f32; out];
    for i in 0..out {
        let mut acc = b[i];
        for j in 0..inn {
            acc += w[[i, j]] * x[j];
        }
        y[i] = acc;
    }
    y
}

fn relu_vec(v: &[f32]) -> Vec<f32> {
    v.iter().map(|&z| z.max(0.0)).collect()
}

fn add_vec(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

fn max_vec(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| x.max(*y)).collect()
}

fn min_vec(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| x.min(*y)).collect()
}

// ---------------------------------------------------------------------------
// Shared assertion: dense sampling soundness check.
// ---------------------------------------------------------------------------

/// Run the constrained backward and assert per-coordinate sound containment
/// over `N_SAMPLES` deterministic points. `truth` maps an input point to the
/// vector of true output coordinates (matching the bound vector length).
fn assert_sound_over_box<F>(
    label: &str,
    graph: &GraphNetwork,
    lower_in: &[f32],
    upper_in: &[f32],
    truth: F,
) where
    F: Fn(&[f32]) -> Vec<f32>,
{
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[lower_in.len()]), lower_in.to_vec()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[upper_in.len()]), upper_in.to_vec()).unwrap(),
    )
    .expect("valid input bounds");

    let history = GraphSplitHistory::new();
    let context = GraphCrownContext::for_history(&history);

    let (output, _cache) = verifier
        .propagate_crown_with_graph_constraints(graph, &input, &context, None, None)
        .unwrap_or_else(|e| panic!("[{label}] constrained backward failed: {e:?}"));

    let lo: Vec<f32> = output.lower().iter().copied().collect();
    let hi: Vec<f32> = output.upper().iter().copied().collect();
    assert!(
        lo.iter().chain(hi.iter()).all(|v| v.is_finite()),
        "[{label}] bounds must be finite: lo={lo:?} hi={hi:?}"
    );

    let mut rng = DetRng::new(0xA11D_5EED_u64 ^ (label.len() as u64));
    // Track the observed true min/max per coord so we can confirm the bound has
    // to STRETCH to cover the sampled extremes (a dropped branch-B contribution
    // would push these observed extremes outside the bound -> UNSOUND assert).
    let n = lo.len();
    let mut obs_min = vec![f32::INFINITY; n];
    let mut obs_max = vec![f32::NEG_INFINITY; n];
    for idx in 0..N_SAMPLES {
        let x = sample_point(&mut rng, lower_in, upper_in, idx);
        let t = truth(&x);
        assert_eq!(
            t.len(),
            n,
            "[{label}] truth width {} != bound width {}",
            t.len(),
            n
        );
        for k in 0..n {
            obs_min[k] = obs_min[k].min(t[k]);
            obs_max[k] = obs_max[k].max(t[k]);
            assert!(
                lo[k] <= t[k] + AUDIT_TOL,
                "[{label}] UNSOUND lower at coord {k}: lower={} > true={} at x={x:?} \
                 (full lo={lo:?} hi={hi:?})",
                lo[k],
                t[k]
            );
            assert!(
                hi[k] >= t[k] - AUDIT_TOL,
                "[{label}] UNSOUND upper at coord {k}: upper={} < true={} at x={x:?} \
                 (full lo={lo:?} hi={hi:?})",
                hi[k],
                t[k]
            );
        }
    }
    // Branch-B responsiveness / non-degeneracy: every output coord must have a
    // non-trivial observed true range, and the bound must enclose it. If the
    // backward silently ran identity-on-input[0], the truth (which DOES include
    // B) would exceed the bound and the asserts above would already have fired.
    for k in 0..n {
        assert!(
            obs_max[k] - obs_min[k] > 1e-3,
            "[{label}] coord {k} degenerate observed range [{},{}] — test would not \
             distinguish a dropped contribution",
            obs_min[k],
            obs_max[k]
        );
        println!(
            "[{label}] coord {k}: bound=[{:.4},{:.4}] encloses observed true=[{:.4},{:.4}]",
            lo[k], hi[k], obs_min[k], obs_max[k]
        );
    }
    println!("[{label}] SOUND over {N_SAMPLES} samples ({n} output coords).");
}

// ===========================================================================
// Pattern 1: Residual Add  y = relu(W1 x) + (W2 x)   (canonical ResNet block)
// ===========================================================================

#[test]
fn audit_pattern1_residual_add_skip_not_dropped() {
    // 2 inputs -> 2 outputs each branch. The SKIP branch (W2) must NOT be lost.
    let w1 = arr2(&[[1.0, -0.5], [0.3, 0.8]]);
    let b1 = arr1(&[0.1, -0.2]);
    // W2 chosen so input_b's coordinate-1 has a LARGE, distinct coefficient:
    // if the skip is dropped, the upper bound near x=[+,+] is badly too low.
    let w2 = arr2(&[[0.2, 2.0], [-1.5, 0.4]]);
    let b2 = arr1(&[0.0, 0.5]);

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "main_lin",
        Layer::Linear(LinearLayer::new(w1.clone(), Some(b1.clone())).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "main_relu",
        Layer::ReLU(ReLULayer),
        vec!["main_lin".to_string()],
    ));
    graph.add_node(GraphNode::from_input(
        "skip_lin",
        Layer::Linear(LinearLayer::new(w2.clone(), Some(b2.clone())).unwrap()),
    ));
    graph.add_node(GraphNode::binary(
        "residual",
        Layer::Add(AddLayer),
        "main_relu",
        "skip_lin",
    ));
    graph.set_output("residual");

    let lo = [-1.0f32, -1.0];
    let hi = [1.0f32, 1.0];
    assert_sound_over_box("p1_residual_add", &graph, &lo, &hi, |x| {
        let main = relu_vec(&affine(&w1, &b1, x));
        let skip = affine(&w2, &b2, x);
        add_vec(&main, &skip)
    });
}

// ===========================================================================
// Pattern 2: Add of two ReLU branches  y = relu(A x) + relu(B x)
// ===========================================================================

#[test]
fn audit_pattern2_add_two_relu_branches() {
    let wa = arr2(&[[1.2, -0.7], [0.5, 0.9]]);
    let ba = arr1(&[0.0, -0.3]);
    let wb = arr2(&[[-0.6, 1.1], [0.8, -1.3]]);
    let bb = arr1(&[0.2, 0.1]);

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "lin_a",
        Layer::Linear(LinearLayer::new(wa.clone(), Some(ba.clone())).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "relu_a",
        Layer::ReLU(ReLULayer),
        vec!["lin_a".to_string()],
    ));
    graph.add_node(GraphNode::from_input(
        "lin_b",
        Layer::Linear(LinearLayer::new(wb.clone(), Some(bb.clone())).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "relu_b",
        Layer::ReLU(ReLULayer),
        vec!["lin_b".to_string()],
    ));
    graph.add_node(GraphNode::binary(
        "sum",
        Layer::Add(AddLayer),
        "relu_a",
        "relu_b",
    ));
    graph.set_output("sum");

    let lo = [-1.5f32, -1.0];
    let hi = [1.0f32, 1.5];
    assert_sound_over_box("p2_add_two_relu", &graph, &lo, &hi, |x| {
        let a = relu_vec(&affine(&wa, &ba, x));
        let b = relu_vec(&affine(&wb, &bb, x));
        add_vec(&a, &b)
    });
}

// ===========================================================================
// Pattern 3: Concat of two branches then Linear
// ===========================================================================

#[test]
fn audit_pattern3_concat_then_linear() {
    // Each branch: 2 inputs -> 2 outputs; concat axis 0 -> 4; linear 4 -> 3.
    let wa = arr2(&[[1.0, 0.4], [-0.3, 0.7]]);
    let ba = arr1(&[0.1, -0.1]);
    let wb = arr2(&[[0.6, -1.2], [0.9, 0.2]]);
    let bb = arr1(&[-0.2, 0.3]);
    // The combine weights give branch B (cols 2,3) strong, distinct influence.
    let wc = arr2(&[
        [0.5, -0.3, 1.4, 0.2],
        [-0.8, 0.6, -1.1, 0.9],
        [0.2, 0.2, 0.7, -1.5],
    ]);
    let bc = arr1(&[0.0, 0.05, -0.05]);

    let concat = ConcatLayer::with_input_shapes(0, vec![vec![2], vec![2]]);

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "lin_a",
        Layer::Linear(LinearLayer::new(wa.clone(), Some(ba.clone())).unwrap()),
    ));
    graph.add_node(GraphNode::from_input(
        "lin_b",
        Layer::Linear(LinearLayer::new(wb.clone(), Some(bb.clone())).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "cat",
        Layer::Concat(concat),
        vec!["lin_a".to_string(), "lin_b".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "out",
        Layer::Linear(LinearLayer::new(wc.clone(), Some(bc.clone())).unwrap()),
        vec!["cat".to_string()],
    ));
    graph.set_output("out");

    let lo = [-1.0f32, -1.0];
    let hi = [1.0f32, 1.0];
    assert_sound_over_box("p3_concat_linear", &graph, &lo, &hi, |x| {
        let a = affine(&wa, &ba, x);
        let b = affine(&wb, &bb, x);
        let mut cat = a;
        cat.extend_from_slice(&b);
        affine(&wc, &bc, &cat)
    });
}

// ===========================================================================
// Pattern 4: MaxBinary / MinBinary of two affine branches (#1934 generalized)
// ===========================================================================

#[test]
fn audit_pattern4_max_binary_multidim() {
    let wa = arr2(&[[1.0, 0.5], [-0.4, 0.9]]);
    let ba = arr1(&[0.0, 0.2]);
    let wb = arr2(&[[0.3, 1.6], [1.1, -0.7]]);
    let bb = arr1(&[0.5, -0.3]);

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "lin_a",
        Layer::Linear(LinearLayer::new(wa.clone(), Some(ba.clone())).unwrap()),
    ));
    graph.add_node(GraphNode::from_input(
        "lin_b",
        Layer::Linear(LinearLayer::new(wb.clone(), Some(bb.clone())).unwrap()),
    ));
    graph.add_node(GraphNode::binary(
        "maxb",
        Layer::MaxBinary(MaxBinaryLayer),
        "lin_a",
        "lin_b",
    ));
    graph.set_output("maxb");

    let lo = [-1.0f32, -1.0];
    let hi = [1.0f32, 1.0];
    assert_sound_over_box("p4_max_binary", &graph, &lo, &hi, |x| {
        let a = affine(&wa, &ba, x);
        let b = affine(&wb, &bb, x);
        max_vec(&a, &b)
    });
}

#[test]
fn audit_pattern4_min_binary_multidim() {
    let wa = arr2(&[[0.9, -0.6], [0.7, 0.5]]);
    let ba = arr1(&[0.1, -0.2]);
    let wb = arr2(&[[-1.3, 0.4], [0.2, 1.5]]);
    let bb = arr1(&[-0.4, 0.6]);

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "lin_a",
        Layer::Linear(LinearLayer::new(wa.clone(), Some(ba.clone())).unwrap()),
    ));
    graph.add_node(GraphNode::from_input(
        "lin_b",
        Layer::Linear(LinearLayer::new(wb.clone(), Some(bb.clone())).unwrap()),
    ));
    graph.add_node(GraphNode::binary(
        "minb",
        Layer::MinBinary(MinBinaryLayer),
        "lin_a",
        "lin_b",
    ));
    graph.set_output("minb");

    let lo = [-1.0f32, -1.0];
    let hi = [1.0f32, 1.0];
    assert_sound_over_box("p4_min_binary", &graph, &lo, &hi, |x| {
        let a = affine(&wa, &ba, x);
        let b = affine(&wb, &bb, x);
        min_vec(&a, &b)
    });
}

// ===========================================================================
// Pattern 5: 2-level residual DAG (two stacked residual Adds)
// y = relu(W3 * (relu(W1 x) + W2 x)) + (relu(W1 x) + W2 x)
// ===========================================================================

#[test]
fn audit_pattern5_two_level_residual_dag() {
    let w1 = arr2(&[[1.0, -0.3], [0.4, 0.9]]);
    let b1 = arr1(&[0.1, 0.0]);
    let w2 = arr2(&[[0.2, 1.1], [-0.8, 0.5]]); // skip 1
    let b2 = arr1(&[0.0, 0.2]);
    let w3 = arr2(&[[0.6, -0.4], [0.3, 1.2]]);
    let b3 = arr1(&[-0.1, 0.05]);

    let mut graph = GraphNetwork::new();
    // Block 1: r1 = relu(W1 x) + W2 x
    graph.add_node(GraphNode::from_input(
        "lin1",
        Layer::Linear(LinearLayer::new(w1.clone(), Some(b1.clone())).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["lin1".to_string()],
    ));
    graph.add_node(GraphNode::from_input(
        "skip1",
        Layer::Linear(LinearLayer::new(w2.clone(), Some(b2.clone())).unwrap()),
    ));
    graph.add_node(GraphNode::binary(
        "res1",
        Layer::Add(AddLayer),
        "relu1",
        "skip1",
    ));
    // Block 2: r2 = relu(W3 * res1) + res1   (res1 is reused -> DAG fan-out)
    graph.add_node(GraphNode::new(
        "lin3",
        Layer::Linear(LinearLayer::new(w3.clone(), Some(b3.clone())).unwrap()),
        vec!["res1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu3",
        Layer::ReLU(ReLULayer),
        vec!["lin3".to_string()],
    ));
    graph.add_node(GraphNode::binary(
        "res2",
        Layer::Add(AddLayer),
        "relu3",
        "res1",
    ));
    graph.set_output("res2");

    let lo = [-1.0f32, -1.0];
    let hi = [1.0f32, 1.0];
    assert_sound_over_box("p5_two_level_residual", &graph, &lo, &hi, |x| {
        let r1 = add_vec(&relu_vec(&affine(&w1, &b1, x)), &affine(&w2, &b2, x));
        let inner = relu_vec(&affine(&w3, &b3, &r1));
        add_vec(&inner, &r1)
    });
}

// ===========================================================================
// Pattern 6: Multi-output spec matrix where num specs != raw output width.
// Raw output width = 4, spec matrix = 3 rows. Each spec row i bounds
// spec[i] . raw_output(x). Graph has a residual Add feeding the 4-wide output
// so binary routing is exercised under the spec-matrix seed path.
// ===========================================================================

#[test]
fn audit_pattern6_spec_matrix_offbyn() {
    // Branches: 2 in -> 4 out each, residual Add -> 4-wide raw output.
    let w1 = arr2(&[[1.0, -0.5], [0.3, 0.8], [-0.7, 0.2], [0.9, 0.4]]);
    let b1 = arr1(&[0.1, -0.2, 0.0, 0.3]);
    let w2 = arr2(&[[0.2, 1.3], [-1.1, 0.6], [0.5, -0.9], [0.7, 0.1]]);
    let b2 = arr1(&[0.0, 0.4, -0.1, 0.2]);

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "main_lin",
        Layer::Linear(LinearLayer::new(w1.clone(), Some(b1.clone())).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "main_relu",
        Layer::ReLU(ReLULayer),
        vec!["main_lin".to_string()],
    ));
    graph.add_node(GraphNode::from_input(
        "skip_lin",
        Layer::Linear(LinearLayer::new(w2.clone(), Some(b2.clone())).unwrap()),
    ));
    graph.add_node(GraphNode::binary(
        "residual",
        Layer::Add(AddLayer),
        "main_relu",
        "skip_lin",
    ));
    graph.set_output("residual");

    // 3-row spec matrix over the 4 raw outputs (num_specs=3 != raw=4).
    let spec = arr2(&[
        [1.0, -1.0, 0.0, 0.0], // raw0 - raw1
        [0.5, 0.5, 0.5, 0.5],  // mean-ish
        [0.0, 0.0, 1.0, -2.0], // raw2 - 2*raw3
    ]);

    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let lo = [-1.0f32, -1.0];
    let hi = [1.0f32, 1.0];
    let input =
        BoundedTensor::new(arr1(&lo).into_dyn(), arr1(&hi).into_dyn()).expect("valid input bounds");

    let history = GraphSplitHistory::new();
    let context = GraphCrownContext::for_history(&history);

    let (output, _cache, _la) = verifier
        .propagate_crown_with_graph_constraints_with_spec_matrix(
            &graph, &input, &context, None, &spec, None, false,
        )
        .expect("spec-matrix constrained backward should succeed");

    let bound_lo: Vec<f32> = output.lower().iter().copied().collect();
    let bound_hi: Vec<f32> = output.upper().iter().copied().collect();
    assert_eq!(
        bound_lo.len(),
        3,
        "spec-matrix output width should equal num specs (3), got {}; \
         off-by-N would yield raw width 4",
        bound_lo.len()
    );
    assert!(
        bound_lo
            .iter()
            .chain(bound_hi.iter())
            .all(|v| v.is_finite()),
        "spec bounds must be finite: lo={bound_lo:?} hi={bound_hi:?}"
    );

    let raw_out = |x: &[f32]| -> Vec<f32> {
        let main = relu_vec(&affine(&w1, &b1, x));
        let skip = affine(&w2, &b2, x);
        add_vec(&main, &skip)
    };

    let mut rng = DetRng::new(0x5EC0_0DAD_u64.wrapping_mul(2654435761));
    for idx in 0..N_SAMPLES {
        let x = sample_point(&mut rng, &lo, &hi, idx);
        let raw = raw_out(&x);
        for r in 0..3 {
            // true value of this spec = spec[r] . raw
            let mut tv = 0.0f32;
            for j in 0..4 {
                tv += spec[[r, j]] * raw[j];
            }
            assert!(
                bound_lo[r] <= tv + AUDIT_TOL,
                "[p6_spec] UNSOUND lower spec-row {r}: lower={} > true={tv} at x={x:?} \
                 (lo={bound_lo:?} hi={bound_hi:?})",
                bound_lo[r]
            );
            assert!(
                bound_hi[r] >= tv - AUDIT_TOL,
                "[p6_spec] UNSOUND upper spec-row {r}: upper={} < true={tv} at x={x:?} \
                 (lo={bound_lo:?} hi={bound_hi:?})",
                bound_hi[r]
            );
        }
    }
    println!("[p6_spec] sound over {N_SAMPLES} samples (3 spec rows, raw width 4)");
}
