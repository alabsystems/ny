// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! GPU per-domain β optimization tests (#w4-split-tightening).
//!
//! For 3 small random conv resnet DAGs (identity skip, projection skip, stacked
//! blocks) with real ReLU splits (unstable neurons, mixed active/inactive,
//! mixed zero/non-zero β):
//!
//! 1. **β-gradient PARITY**: the GPU A-value gather
//!    (`crown_backward_gpu_resnet_sound_beta_grad`, read from the pre-relaxation
//!    lower coefficient at each split ReLU) must match the CPU analytic
//!    optimizer's capture (`a_at_relu` from
//!    `propagate_crown_with_graph_beta_and_spec_matrix_storing_intermediates`)
//!    at every (spec row, split neuron) — the two sides of
//!    `∂lb_row/∂β_k = −sign_k · A_lower[row, k]`.
//! 2. **Never looser (monotonicity)**: the production per-domain β ascent
//!    (`gpu_beta_optimize_domain`) returns per-row bounds at least as tight as
//!    the single-shot inherited-β GPU lane — iterate 0 IS that pass, and the
//!    result is the element-wise tightest across sound iterates.
//! 3. **Enclosure under the split constraints**: for sampled inputs that
//!    satisfy the domain's split signs, `spec · f(x)` lies inside the optimized
//!    bounds (the β dual may exclude points violating the splits, so samples
//!    are filtered by the constraint).

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_propagate::beta_crown::gpu_beta_debug::{
    debug_cpu_beta_a_at_relu, debug_gpu_beta_gather, debug_gpu_beta_opt_vs_single, DebugSplit,
};
use ny_propagate::{
    layers::{AddLayer, Conv2dLayer, ReLULayer},
    GraphNetwork, GraphNode, Layer,
};
use ny_tensor::BoundedTensor;

use crate::wgpu_device::test_support::{gpu_test_serial_guard, require_device};

/// Deterministic LCG in [-1, 1).
struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn conv(
    name: &str,
    input: &str,
    in_c: usize,
    out_c: usize,
    k: usize,
    pad: usize,
    hw: usize,
    rng: &mut Lcg,
) -> GraphNode {
    let kernel = ArrayD::from_shape_fn(IxDyn(&[out_c, in_c, k, k]), |_| rng.next_f32() * 0.4);
    let bias = Array1::from_shape_fn(out_c, |_| rng.next_f32() * 0.1);
    let mut layer = Conv2dLayer::new(kernel, Some(bias), (1, 1), (pad, pad)).expect("conv layer");
    layer.input_shape = Some((hw, hw));
    if input.is_empty() {
        GraphNode::from_input(name, Layer::Conv2d(layer))
    } else {
        GraphNode::new(name, Layer::Conv2d(layer), vec![input.to_string()])
    }
}

fn relu(name: &str, input: &str) -> GraphNode {
    GraphNode::new(name, Layer::ReLU(ReLULayer), vec![input.to_string()])
}

fn add(name: &str, a: &str, b: &str) -> GraphNode {
    GraphNode::new(
        name,
        Layer::Add(AddLayer),
        vec![a.to_string(), b.to_string()],
    )
}

/// Graph A: identity-skip block. input → conv1 → relu1 → conv2 → add(conv2, conv1).
fn identity_skip_graph(hw: usize, rng: &mut Lcg) -> GraphNetwork {
    let mut g = GraphNetwork::new();
    g.add_node(conv("conv1", "", 2, 4, 3, 1, hw, rng));
    g.add_node(relu("relu1", "conv1"));
    g.add_node(conv("conv2", "relu1", 4, 4, 3, 1, hw, rng));
    g.add_node(add("add", "conv2", "conv1"));
    g.set_output("add");
    g
}

/// Graph B: projection-skip block. input → conv1 → relu1 → {conv2 (3×3), convp (1×1)} → add.
fn projection_skip_graph(hw: usize, rng: &mut Lcg) -> GraphNetwork {
    let mut g = GraphNetwork::new();
    g.add_node(conv("conv1", "", 2, 4, 3, 1, hw, rng));
    g.add_node(relu("relu1", "conv1"));
    g.add_node(conv("conv2", "relu1", 4, 6, 3, 1, hw, rng));
    g.add_node(conv("convp", "relu1", 4, 6, 1, 0, hw, rng));
    g.add_node(add("add", "conv2", "convp"));
    g.set_output("add");
    g
}

/// Graph C: two stacked identity blocks with ReLUs between.
fn stacked_blocks_graph(hw: usize, rng: &mut Lcg) -> GraphNetwork {
    let mut g = GraphNetwork::new();
    g.add_node(conv("conv0", "", 2, 4, 3, 1, hw, rng));
    g.add_node(relu("relu1", "conv0"));
    g.add_node(conv("conv1", "relu1", 4, 4, 3, 1, hw, rng));
    g.add_node(add("add1", "conv1", "conv0"));
    g.add_node(relu("relu2", "add1"));
    g.add_node(conv("conv2", "relu2", 4, 4, 3, 1, hw, rng));
    g.add_node(add("add2", "conv2", "add1"));
    g.set_output("add2");
    g
}

/// #seg-resident: append a 1×1 conv head so the FIRST backward segment is a
/// Chain (mirrors the real resnet's head; the device-resident stream requires
/// it — with the output ON the add, the skip merge would need the raw spec
/// seed as a device stream and resident mode correctly refuses).
fn with_head(mut g: GraphNetwork, prev: &str, ch: usize, hw: usize, rng: &mut Lcg) -> GraphNetwork {
    g.add_node(conv("head", prev, ch, 3, 1, 0, hw, rng));
    g.set_output("head");
    g
}

fn input_box(hw: usize, rng: &mut Lcg) -> BoundedTensor {
    let center = ArrayD::from_shape_fn(IxDyn(&[2, hw, hw]), |_| rng.next_f32() * 0.5);
    let radius = 0.08f32;
    BoundedTensor::new(center.mapv(|c| c - radius), center.mapv(|c| c + radius)).expect("input box")
}

/// Pick up to `n` unstable neurons (l < 0 < u) of `pre_node`'s IBP bounds and
/// return splits with alternating active/inactive signs and mixed β values
/// (0.0 exercises the "entry present, dual inert" case).
fn pick_splits(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    relu_node: &str,
    pre_node: &str,
    n: usize,
) -> Vec<DebugSplit> {
    let ibp_map = graph.collect_node_bounds(input).expect("IBP node bounds");
    let pre = ibp_map[pre_node].flatten();
    let mut splits = Vec::new();
    for i in 0..pre.len() {
        let (l, u) = (pre.lower()[[i]], pre.upper()[[i]]);
        if l < 0.0 && u > 0.0 {
            let k = splits.len();
            let beta = match k % 3 {
                0 => 0.05,
                1 => 0.0,
                _ => 0.12,
            };
            splits.push((relu_node.to_string(), i, k % 2 == 0, beta));
            if splits.len() == n {
                break;
            }
        }
    }
    assert!(
        !splits.is_empty(),
        "no unstable neurons found at {pre_node} — fixture too tight"
    );
    splits
}

/// Like [`pick_splits`] but each split's branch follows the side with more
/// interval mass (`is_active ⇔ u ≥ −l`), so a random box sample satisfies each
/// constraint with probability ≥ 1/2 — needed by the enclosure check, which
/// filters samples by the split signs.
fn pick_splits_mass(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    relu_node: &str,
    pre_node: &str,
    n: usize,
) -> Vec<DebugSplit> {
    let ibp_map = graph.collect_node_bounds(input).expect("IBP node bounds");
    let pre = ibp_map[pre_node].flatten();
    let mut splits = Vec::new();
    for i in 0..pre.len() {
        let (l, u) = (pre.lower()[[i]], pre.upper()[[i]]);
        if l < 0.0 && u > 0.0 {
            let k = splits.len();
            let beta = match k % 3 {
                0 => 0.05,
                1 => 0.0,
                _ => 0.12,
            };
            splits.push((relu_node.to_string(), i, u >= -l, beta));
            if splits.len() == n {
                break;
            }
        }
    }
    assert!(
        !splits.is_empty(),
        "no unstable neurons found at {pre_node} — fixture too tight"
    );
    splits
}

fn random_spec(rng: &mut Lcg, num_specs: usize, out_dim: usize) -> Array2<f32> {
    let mut spec = Array2::<f32>::zeros((num_specs, out_dim));
    for mut row in spec.rows_mut() {
        for v in row.iter_mut() {
            *v = rng.next_f32();
        }
    }
    spec
}

/// Interval product of one spec row with a bound box (flat).
fn spec_row_interval(row: &[f32], lower: &[f32], upper: &[f32]) -> (f32, f32) {
    let mut lo = 0.0f32;
    let mut hi = 0.0f32;
    for (j, &c) in row.iter().enumerate() {
        if c >= 0.0 {
            lo += c * lower[j];
            hi += c * upper[j];
        } else {
            lo += c * upper[j];
            hi += c * lower[j];
        }
    }
    (lo, hi)
}

/// CLAIM 1 — β-gradient parity: GPU-gathered A-values at every (spec row, split
/// neuron) match the CPU analytic optimizer's `a_at_relu` capture.
fn assert_beta_grad_parity(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    splits: &[DebugSplit],
    rng: &mut Lcg,
    label: &str,
) {
    let device = require_device();
    let ibp_map = graph.collect_node_bounds(input).expect("IBP node bounds");
    let out_dim: usize = ibp_map[graph.output_name()].len();
    let spec = random_spec(rng, 3, out_dim);

    let (cpu_a, _cpu_rows) =
        debug_cpu_beta_a_at_relu(graph, input, &spec, splits).expect("CPU a_at_relu");
    let gpu = debug_gpu_beta_gather(graph, input, &spec, splits, device.as_ref())
        .expect("GPU beta gather (segments must extract on these fixtures)");

    let num_specs = spec.nrows();
    let mut compared = 0usize;
    for (r, name) in gpu.relu_names.iter().enumerate() {
        let cols = &gpu.gather_idx[r];
        if cols.is_empty() {
            continue;
        }
        let gathered = &gpu.gathers[r];
        assert_eq!(
            gathered.len(),
            num_specs * cols.len(),
            "{label}: gather[{r}] ({name}) wrong shape"
        );
        let cpu_m = cpu_a
            .get(name)
            .unwrap_or_else(|| panic!("{label}: CPU a_at_relu missing node {name}"));
        for s in 0..num_specs {
            for (i, &col) in cols.iter().enumerate() {
                let g = gathered[s * cols.len() + i];
                let c = cpu_m[[s, col as usize]];
                let tol = 1e-4f32 + 2e-3 * c.abs().max(g.abs());
                assert!(
                    (g - c).abs() <= tol,
                    "{label}: A-value mismatch at relu {name} row {s} col {col}: \
                     GPU {g} vs CPU {c} (tol {tol})"
                );
                compared += 1;
            }
        }
    }
    assert!(
        compared >= splits.len(),
        "{label}: parity compared only {compared} entries for {} splits — \
         the gather channel did not fire",
        splits.len()
    );
}

/// CLAIM 2+3 — never-looser vs the single-shot lane + enclosure on
/// constraint-satisfying samples.
fn assert_beta_opt_monotone_and_sound(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    splits: &[DebugSplit],
    rng: &mut Lcg,
    label: &str,
) {
    let device = require_device();
    let ibp_map = graph.collect_node_bounds(input).expect("IBP node bounds");
    let out_dim: usize = ibp_map[graph.output_name()].len();
    let num_specs = 3usize;
    let spec = random_spec(rng, num_specs, out_dim);
    let thresholds = vec![0.0f32; num_specs];

    let ((single_lo, single_hi), (opt_lo, opt_hi)) =
        debug_gpu_beta_opt_vs_single(graph, input, &spec, splits, &thresholds, 6, device.as_ref())
            .expect("GPU beta opt-vs-single (fixture must take the GPU path)");

    // CLAIM 2: element-wise never looser than the single-shot lane (iterate 0
    // is the same dispatch sequence on the same buffers — deterministic — and
    // the merge takes the per-row tightest).
    for s in 0..num_specs {
        assert!(
            opt_lo[s] >= single_lo[s],
            "{label}: spec[{s}] optimized lower {} LOOSER than single-shot {}",
            opt_lo[s],
            single_lo[s]
        );
        assert!(
            opt_hi[s] <= single_hi[s],
            "{label}: spec[{s}] optimized upper {} LOOSER than single-shot {}",
            opt_hi[s],
            single_hi[s]
        );
        assert!(
            opt_lo[s].is_finite() && opt_hi[s].is_finite() && opt_lo[s] <= opt_hi[s],
            "{label}: spec[{s}] optimized bounds invalid [{}, {}]",
            opt_lo[s],
            opt_hi[s]
        );
    }

    // CLAIM 3: enclosure on samples SATISFYING the split constraints. The β
    // dual (and the constrained forward bounds) are only valid on the split
    // sub-domain, so samples violating a split sign are skipped.
    let in_lo: Vec<f32> = input.lower().iter().copied().collect();
    let in_hi: Vec<f32> = input.upper().iter().copied().collect();
    let in_shape: Vec<usize> = input.shape().to_vec();
    let mut kept = 0usize;
    for t in 0..200 {
        let point: Vec<f32> = (0..in_lo.len())
            .map(|i| {
                let f = f32::midpoint(rng.next_f32(), 1.0);
                in_lo[i] + f * (in_hi[i] - in_lo[i])
            })
            .collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&in_shape), point).expect("point shape");
        let point_box = BoundedTensor::new(arr.clone(), arr).expect("point box");
        let node_bounds = graph
            .collect_node_bounds(&point_box)
            .expect("point forward");

        // Filter: every split's PRE-activation sign must match its branch.
        let mut satisfies = true;
        for (relu_node, idx, is_active, _beta) in splits {
            let pre_name = &graph.node(relu_node).expect("relu node").inputs()[0];
            let pre = node_bounds[pre_name].flatten();
            let (l, u) = (pre.lower()[[*idx]], pre.upper()[[*idx]]);
            let ok = if *is_active { l >= -1e-6 } else { u <= 1e-6 };
            if !ok {
                satisfies = false;
                break;
            }
        }
        if !satisfies {
            continue;
        }
        kept += 1;
        let _ = t;

        let out_flat = node_bounds[graph.output_name()].flatten();
        let pl: Vec<f32> = out_flat.lower().iter().copied().collect();
        let pu: Vec<f32> = out_flat.upper().iter().copied().collect();
        for s in 0..num_specs {
            let row: Vec<f32> = spec.row(s).iter().copied().collect();
            let (slo, shi) = spec_row_interval(&row, &pl, &pu);
            assert!(
                shi >= opt_lo[s] - 1e-3,
                "{label}: UNSOUND optimized lower — constrained sample spec[{s}] \
                 value ≤ {shi} < optimized lower {}",
                opt_lo[s]
            );
            assert!(
                slo <= opt_hi[s] + 1e-3,
                "{label}: UNSOUND optimized upper — constrained sample spec[{s}] \
                 value ≥ {slo} > optimized upper {}",
                opt_hi[s]
            );
        }
    }
    assert!(
        kept >= 3,
        "{label}: only {kept} constraint-satisfying samples — fixture too constrained \
         for the enclosure check"
    );
}

#[test]
fn beta_grad_parity_identity_skip_w4() {
    let _g = gpu_test_serial_guard();
    let mut rng = Lcg(0xBE7A_0001);
    let hw = 5;
    let graph = identity_skip_graph(hw, &mut rng);
    let input = input_box(hw, &mut rng);
    let splits = pick_splits(&graph, &input, "relu1", "conv1", 3);
    assert_beta_grad_parity(&graph, &input, &splits, &mut rng, "identity-skip");
}

#[test]
fn beta_grad_parity_projection_skip_w4() {
    let _g = gpu_test_serial_guard();
    let mut rng = Lcg(0xBE7A_0002);
    let hw = 5;
    let graph = projection_skip_graph(hw, &mut rng);
    let input = input_box(hw, &mut rng);
    let splits = pick_splits(&graph, &input, "relu1", "conv1", 3);
    assert_beta_grad_parity(&graph, &input, &splits, &mut rng, "projection-skip");
}

#[test]
fn beta_grad_parity_stacked_blocks_w4() {
    let _g = gpu_test_serial_guard();
    let mut rng = Lcg(0xBE7A_0003);
    let hw = 5;
    let graph = stacked_blocks_graph(hw, &mut rng);
    let input = input_box(hw, &mut rng);
    // Splits on BOTH ReLUs: exercises fold-order alignment across segments.
    let mut splits = pick_splits(&graph, &input, "relu1", "conv0", 2);
    splits.extend(pick_splits(&graph, &input, "relu2", "add1", 2));
    assert_beta_grad_parity(&graph, &input, &splits, &mut rng, "stacked-blocks");
}

#[test]
fn beta_opt_never_looser_identity_skip_w4() {
    let _g = gpu_test_serial_guard();
    let mut rng = Lcg(0xBE7A_0011);
    let hw = 5;
    let graph = identity_skip_graph(hw, &mut rng);
    let input = input_box(hw, &mut rng);
    let splits = pick_splits_mass(&graph, &input, "relu1", "conv1", 3);
    assert_beta_opt_monotone_and_sound(&graph, &input, &splits, &mut rng, "identity-skip");
}

#[test]
fn beta_opt_never_looser_projection_skip_w4() {
    let _g = gpu_test_serial_guard();
    let mut rng = Lcg(0xBE7A_0012);
    let hw = 5;
    let graph = projection_skip_graph(hw, &mut rng);
    let input = input_box(hw, &mut rng);
    let splits = pick_splits_mass(&graph, &input, "relu1", "conv1", 3);
    assert_beta_opt_monotone_and_sound(&graph, &input, &splits, &mut rng, "projection-skip");
}

#[test]
fn beta_opt_never_looser_stacked_blocks_w4() {
    let _g = gpu_test_serial_guard();
    let mut rng = Lcg(0xBE7A_0013);
    let hw = 5;
    let graph = stacked_blocks_graph(hw, &mut rng);
    let input = input_box(hw, &mut rng);
    let mut splits = pick_splits_mass(&graph, &input, "relu1", "conv0", 2);
    splits.extend(pick_splits_mass(&graph, &input, "relu2", "add1", 2));
    assert_beta_opt_monotone_and_sound(&graph, &input, &splits, &mut rng, "stacked-blocks");
}

/// #seg-resident A/B: the device-resident segment stream (`NY_SEG_RESIDENT=1`)
/// vs the legacy per-segment download/merge/re-upload, on the SAME β-folded
/// sound resnet backward. Value lanes are bit-identical by construction (the
/// f32 RN add of two f32s IS the correctly-rounded f64 sum the CPU merge
/// computes); the on-device merge error lane carries an outward slack ≥ the
/// CPU merge's — so the resident bound may only WIDEN, and by at most the tiny
/// slack. Asserts: finite, enclosure (resident ⊇ legacy), and closeness.
fn assert_seg_resident_ab(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    splits: &[DebugSplit],
    rng: &mut Lcg,
    label: &str,
) {
    let device = require_device();
    let ibp_map = graph.collect_node_bounds(input).expect("IBP node bounds");
    let out_dim: usize = ibp_map[graph.output_name()].len();
    let num_specs = 3usize;
    let spec = random_spec(rng, num_specs, out_dim);
    let thresholds = vec![0.0f32; num_specs];
    let run = || {
        debug_gpu_beta_opt_vs_single(graph, input, &spec, splits, &thresholds, 2, device.as_ref())
            .expect("GPU beta single lane (fixture must take the GPU path)")
    };
    let ((leg_lo, leg_hi), (leg_opt_lo, leg_opt_hi)) =
        ny_test_utils::env::with_env_edits(|e| {
            e.remove("NY_SEG_RESIDENT");
            run()
        });
    let ((res_lo, res_hi), (res_opt_lo, res_opt_hi)) =
        ny_test_utils::env::with_env_edits(|e| {
            e.set("NY_SEG_RESIDENT", "1");
            run()
        });
    let check = |leg_lo: &[f32], leg_hi: &[f32], res_lo: &[f32], res_hi: &[f32], lane: &str| {
        assert_eq!(res_lo.len(), leg_lo.len(), "{label}/{lane}: length mismatch");
        for i in 0..leg_lo.len() {
            assert!(
                res_lo[i].is_finite() && res_hi[i].is_finite(),
                "{label}/{lane}: non-finite resident bound at {i}"
            );
            // Enclosure: the resident error lane is ≥ the CPU merge's, so the
            // concretized interval may only widen.
            assert!(
                res_lo[i] <= leg_lo[i] && res_hi[i] >= leg_hi[i],
                "{label}/{lane}: resident bound TIGHTER than legacy at {i}: \
                 [{}, {}] vs [{}, {}] — soundness ordering violated",
                res_lo[i],
                res_hi[i],
                leg_lo[i],
                leg_hi[i]
            );
            // Closeness: the widening is bounded by the tiny merge slack.
            let tol = 1e-4f32 + 1e-4 * leg_lo[i].abs().max(leg_hi[i].abs());
            assert!(
                (res_lo[i] - leg_lo[i]).abs() <= tol && (res_hi[i] - leg_hi[i]).abs() <= tol,
                "{label}/{lane}: resident bound drifted at {i}: [{}, {}] vs [{}, {}] (tol {tol})",
                res_lo[i],
                res_hi[i],
                leg_lo[i],
                leg_hi[i]
            );
        }
    };
    check(&leg_lo, &leg_hi, &res_lo, &res_hi, "single");
    check(&leg_opt_lo, &leg_opt_hi, &res_opt_lo, &res_opt_hi, "opt");
}

#[test]
fn seg_resident_ab_identity_skip_w4() {
    let _g = gpu_test_serial_guard();
    let mut rng = Lcg(0x5E97_0001);
    let hw = 5;
    let graph = with_head(identity_skip_graph(hw, &mut rng), "add", 4, hw, &mut rng);
    let input = input_box(hw, &mut rng);
    let splits = pick_splits(&graph, &input, "relu1", "conv1", 3);
    assert_seg_resident_ab(&graph, &input, &splits, &mut rng, "identity-skip+head");
}

#[test]
fn seg_resident_ab_projection_skip_w4() {
    let _g = gpu_test_serial_guard();
    let mut rng = Lcg(0x5E97_0002);
    let hw = 5;
    let graph = with_head(projection_skip_graph(hw, &mut rng), "add", 6, hw, &mut rng);
    let input = input_box(hw, &mut rng);
    let splits = pick_splits(&graph, &input, "relu1", "conv1", 3);
    assert_seg_resident_ab(&graph, &input, &splits, &mut rng, "projection-skip+head");
}

#[test]
fn seg_resident_ab_stacked_blocks_w4() {
    let _g = gpu_test_serial_guard();
    let mut rng = Lcg(0x5E97_0003);
    let hw = 5;
    let graph = with_head(stacked_blocks_graph(hw, &mut rng), "add2", 4, hw, &mut rng);
    let input = input_box(hw, &mut rng);
    let mut splits = pick_splits(&graph, &input, "relu1", "conv0", 2);
    splits.extend(pick_splits(&graph, &input, "relu2", "add1", 2));
    assert_seg_resident_ab(&graph, &input, &splits, &mut rng, "stacked-blocks+head");
}
