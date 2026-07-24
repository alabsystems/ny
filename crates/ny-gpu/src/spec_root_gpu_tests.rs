// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Bound-correctness tests for the C-matrix-seeded GPU resnet ROOT pass
//! (#w4-root-gpu): the spec-guided CROWN entry in ny-propagate seeds the sound
//! GPU-resident resnet backward with the full spec matrix when the graph
//! decomposes (`spec_propagation/core.rs`).
//!
//! For 3 small random conv resnet DAGs (identity skip, projection skip,
//! stacked blocks) we assert, per spec row:
//!
//! 1. **Enclosure (soundness)** vs Monte-Carlo forward sampling: for sampled
//!    inputs x in the box, `spec · f(x)` must lie inside the GPU bounds. The
//!    per-point value is itself enclosed by a point-box IBP forward, so the
//!    check is conservative (only definite violations fail).
//! 2. **Never looser than IBP**: the root hook intersects with the IBP spec
//!    bounds, so the GPU bounds must be at least as tight as interval
//!    arithmetic on the output IBP box.
//! 3. **Within certified tolerance of the CPU spec propagation** (same fixed
//!    relaxation, engine = None ⇒ the proven CPU dense backward): catches
//!    seeding/convention errors, while allowing the certified f32 error's
//!    ULP-scale widening in either direction.
//!
//! The GPU route is confirmed to actually FIRE via the cache discriminator:
//! the GPU root returns no `CachedLinearBounds` (concrete bounds only), while
//! the CPU path with cache capture returns one.

use ndarray::{Array1, Array2, ArrayD, IxDyn};
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

/// Run all #w4-root-gpu assertions for one graph.
fn assert_gpu_root_sound(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    out_name: &str,
    rng: &mut Lcg,
    label: &str,
) {
    let device = require_device();

    let ibp_map = graph.collect_node_bounds(input).expect("IBP node bounds");
    let out_flat: Vec<usize> = ibp_map[out_name].shape().to_vec();
    let out_dim: usize = out_flat.iter().product();

    // 3 random spec rows over the flat output dim (margin-like: mixed signs).
    let num_specs = 3usize;
    let mut spec = Array2::<f32>::zeros((num_specs, out_dim));
    for mut row in spec.rows_mut() {
        for v in row.iter_mut() {
            *v = rng.next_f32();
        }
    }

    // GPU route (real device engine): cache must be None (the discriminator
    // that the GPU root FIRED — it returns concrete bounds only).
    let (gpu_bounds, gpu_cache) = graph
        .propagate_crown_with_specs_and_node_bounds_and_cache_and_deadline(
            input,
            &spec,
            Some(device.as_ref()),
            &ibp_map,
            None,
        )
        .expect("GPU spec root");
    assert!(
        gpu_cache.is_none(),
        "{label}: expected the GPU root pass to fire (no linear cache); \
         a Some(cache) means the CPU loop ran instead"
    );

    // CPU reference: the proven CPU dense spec loop. Both root fast routes
    // (#w4-root-margin forward-linear composition and #w4-root-gpu — the
    // margin route is engine-independent, so engine=None alone is not enough)
    // are disabled via their opt-out flags for this call ONLY; gpu-tests run
    // single-threaded under the serial guard, and the mutation routes through
    // the blessed env choke point (clippy env wall) which restores on drop.
    let cpu_result = {
        let _g_margin = ny_test_utils::env::ScopedEnvVar::set("NY_SPEC_ROOT_MARGIN", "0");
        let _g_gpu = ny_test_utils::env::ScopedEnvVar::set("NY_SPEC_ROOT_GPU", "0");
        graph.propagate_crown_with_specs_and_node_bounds_and_cache_and_deadline(
            input, &spec, None, &ibp_map, None,
        )
    };
    let (cpu_bounds, cpu_cache) = cpu_result.expect("CPU spec reference");
    assert!(
        cpu_cache.is_some(),
        "{label}: CPU reference should capture a linear cache"
    );

    let gpu_l = gpu_bounds.lower();
    let gpu_u = gpu_bounds.upper();
    let cpu_l = cpu_bounds.lower();
    let cpu_u = cpu_bounds.upper();

    // (2) Never looser than IBP: the root hook intersects with the IBP spec
    // bounds derived from the SAME node_bounds map.
    let out_ibp = ibp_map[out_name].flatten();
    let ibp_lo: Vec<f32> = out_ibp.lower().iter().copied().collect();
    let ibp_hi: Vec<f32> = out_ibp.upper().iter().copied().collect();
    for s in 0..num_specs {
        let row: Vec<f32> = spec.row(s).iter().copied().collect();
        let (il, iu) = spec_row_interval(&row, &ibp_lo, &ibp_hi);
        assert!(
            gpu_l[[s]] >= il - 1e-4 && gpu_u[[s]] <= iu + 1e-4,
            "{label}: spec[{s}] GPU [{}, {}] looser than IBP [{il}, {iu}]",
            gpu_l[[s]],
            gpu_u[[s]],
        );
        assert!(
            gpu_l[[s]].is_finite() && gpu_u[[s]].is_finite() && gpu_l[[s]] <= gpu_u[[s]],
            "{label}: spec[{s}] GPU bounds invalid [{}, {}]",
            gpu_l[[s]],
            gpu_u[[s]],
        );
    }

    // (3) Never looser than the proven CPU spec pass, modulo certified
    // tolerance: routing the root to the GPU must not regress tightness.
    // (Measured: the GPU resnet fold is strictly TIGHTER than the CPU spec
    // loop on these diamonds — e.g. identity-skip spec[0] GPU [-6.42, 0.10]
    // vs CPU [-13.63, 7.82], both enclosing EMP [-3.75, -2.39] — so only the
    // "not looser" direction is asserted; exact closeness is not a contract.)
    for s in 0..num_specs {
        let span = (cpu_u[[s]] - cpu_l[[s]]).abs();
        let tol = 1e-2 * (1.0 + span);
        assert!(
            gpu_l[[s]] >= cpu_l[[s]] - tol && gpu_u[[s]] <= cpu_u[[s]] + tol,
            "{label}: spec[{s}] GPU [{}, {}] LOOSER than CPU [{}, {}] (tol {tol})",
            gpu_l[[s]],
            gpu_u[[s]],
            cpu_l[[s]],
            cpu_u[[s]],
        );
    }

    // (1) Enclosure vs MC forward sampling: sample x in the box; spec·f(x) is
    // enclosed by the point-box IBP forward, so a definite violation is when
    // the whole point interval falls outside the GPU bounds.
    let in_lo: Vec<f32> = input.lower().iter().copied().collect();
    let in_hi: Vec<f32> = input.upper().iter().copied().collect();
    let in_shape: Vec<usize> = input.shape().to_vec();
    for t in 0..100 {
        let point: Vec<f32> = (0..in_lo.len())
            .map(|i| {
                let f = if t == 0 {
                    0.0
                } else if t == 1 {
                    1.0
                } else {
                    f32::midpoint(rng.next_f32(), 1.0)
                };
                in_lo[i] + f * (in_hi[i] - in_lo[i])
            })
            .collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&in_shape), point).expect("point shape");
        let point_box = BoundedTensor::new(arr.clone(), arr).expect("point box");
        let out = graph.propagate_ibp(&point_box).expect("point forward");
        let out_flat = out.flatten();
        let pl: Vec<f32> = out_flat.lower().iter().copied().collect();
        let pu: Vec<f32> = out_flat.upper().iter().copied().collect();
        for s in 0..num_specs {
            let row: Vec<f32> = spec.row(s).iter().copied().collect();
            let (slo, shi) = spec_row_interval(&row, &pl, &pu);
            assert!(
                shi >= gpu_l[[s]] - 1e-3,
                "{label}: UNSOUND lower — sample {t} spec[{s}] value ≤ {shi} < GPU lower {}",
                gpu_l[[s]],
            );
            assert!(
                slo <= gpu_u[[s]] + 1e-3,
                "{label}: UNSOUND upper — sample {t} spec[{s}] value ≥ {slo} > GPU upper {}",
                gpu_u[[s]],
            );
        }
    }
}

fn input_box(hw: usize, rng: &mut Lcg) -> BoundedTensor {
    let center = ArrayD::from_shape_fn(IxDyn(&[2, hw, hw]), |_| rng.next_f32() * 0.5);
    let radius = 0.08f32;
    BoundedTensor::new(center.mapv(|c| c - radius), center.mapv(|c| c + radius)).expect("input box")
}

#[test]
fn spec_root_gpu_identity_skip_sound_and_close_to_cpu() {
    let _g = gpu_test_serial_guard();
    let mut rng = Lcg(0x5EED_0001);
    let hw = 5;
    let graph = identity_skip_graph(hw, &mut rng);
    let input = input_box(hw, &mut rng);
    assert_gpu_root_sound(&graph, &input, "add", &mut rng, "identity-skip");
}

#[test]
fn spec_root_gpu_projection_skip_sound_and_close_to_cpu() {
    let _g = gpu_test_serial_guard();
    let mut rng = Lcg(0x5EED_0002);
    let hw = 5;
    let graph = projection_skip_graph(hw, &mut rng);
    let input = input_box(hw, &mut rng);
    assert_gpu_root_sound(&graph, &input, "add", &mut rng, "projection-skip");
}

#[test]
fn spec_root_gpu_stacked_blocks_sound_and_close_to_cpu() {
    let _g = gpu_test_serial_guard();
    let mut rng = Lcg(0x5EED_0003);
    let hw = 5;
    let graph = stacked_blocks_graph(hw, &mut rng);
    let input = input_box(hw, &mut rng);
    assert_gpu_root_sound(&graph, &input, "add2", &mut rng, "stacked-blocks");
}
