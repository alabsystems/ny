// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #fl-f32-cpu-seam (CONVWALL_PANEL_VERDICT_2026-08-01, Lane A step 2):
//! with the faer CPU engine installed in the process-global `fast_f32_gemm`
//! registry — exactly what non-CUDA `ny` startup does via
//! `install_cpu_gemm_engine_if_absent()` (ny-cli/src/main.rs) — the
//! `NY_FORWARD_LINEAR_F32` value-GEMM seam in `forward_linear/image.rs` must
//! genuinely take the f32 path (registry telemetry increases), and with the
//! flag unset (the default) the registry must stay unconsulted and the
//! forward-linear bounds bit-identical to a run with no registry at all.
//!
//! This lives in its own integration-test binary so the process starts with a
//! virgin `OnceLock` registry and full ownership of the env flag. Everything
//! runs inside ONE `#[test]` because the install → materialize sequence is
//! process-global and order matters.
//!
//! The tight per-GEMM `γ_{K+4}^f32·S` oracle lives next to the seam
//! (`tests_image.rs::test_forward_f32_value_gemm_error_within_gamma_f32_bound`)
//! and at the engine (`faer_parallelism` unit tests); here the seam-level
//! comparison follows the same idiom as
//! `test_forward_f32_seam_contains_exact_conv_range_and_covers_f64`:
//! the f32-path bounds must never be TIGHTER than the f64-path bounds and may
//! only be wider by the (small) charged penalty.

use ndarray::{Array1, ArrayD, IxDyn};
use ny_propagate::faer_parallelism::install_cpu_gemm_engine_if_absent;
use ny_propagate::fast_f32_gemm;
use ny_propagate::layers::Conv2dLayer;
use ny_propagate::{GraphNetwork, GraphNode, Layer};
use ny_tensor::BoundedTensor;

/// Minimal LCG (mirrors the in-crate test fixture generators).
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next_f32(&mut self, scale: f32) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let unit = ((self.0 >> 11) as f64 / (1u64 << 53) as f64) as f32; // [0,1)
        (unit * 2.0 - 1.0) * scale
    }
}

fn random_kernel(rng: &mut Lcg, out_c: usize, in_c: usize, kh: usize, kw: usize) -> ArrayD<f32> {
    ArrayD::from_shape_fn(IxDyn(&[out_c, in_c, kh, kw]), |_| rng.next_f32(0.8))
}

/// Conv(3→8, 3x3, pad 1) → ReLU → Conv(8→6, 3x3, pad 1) on a 3x6x6 box.
/// Two chained convs so the second composes against a DENSE upstream affine
/// map (contraction 8·3·3 = 72): the f32 value GEMMs genuinely accumulate.
fn fixture() -> (GraphNetwork, BoundedTensor) {
    let mut rng = Lcg::new(0x05EA_1F32);
    let (in_c, in_h, in_w) = (3usize, 6usize, 6usize);

    let conv1 = Conv2dLayer::with_input_shape(
        random_kernel(&mut rng, 8, in_c, 3, 3),
        Some(Array1::from_iter((0..8).map(|_| rng.next_f32(0.3)))),
        (1, 1),
        (1, 1),
        in_h,
        in_w,
    )
    .expect("conv1");
    let conv2 = Conv2dLayer::with_input_shape(
        random_kernel(&mut rng, 6, 8, 3, 3),
        Some(Array1::from_iter((0..6).map(|_| rng.next_f32(0.3)))),
        (1, 1),
        (1, 1),
        in_h,
        in_w,
    )
    .expect("conv2");

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("conv1", Layer::Conv2d(conv1)));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ny_propagate::layers::ReLULayer::new()),
        vec!["conv1".into()],
    ));
    graph.add_node(GraphNode::new(
        "conv2",
        Layer::Conv2d(conv2),
        vec!["relu1".into()],
    ));
    graph.set_output("conv2");

    let lower = ArrayD::from_shape_fn(IxDyn(&[in_c, in_h, in_w]), |_| rng.next_f32(0.6) - 0.1);
    let upper = lower.mapv(|v| v + 0.15);
    let input = BoundedTensor::new(lower, upper).expect("input box");
    (graph, input)
}

fn collect(
    graph: &GraphNetwork,
    input: &BoundedTensor,
) -> std::collections::HashMap<String, BoundedTensor> {
    graph
        .collect_forward_linear_bounds_dag_with_engine(input, None)
        .expect("forward-linear collection")
}

fn assert_bitwise_equal(
    left: &std::collections::HashMap<String, BoundedTensor>,
    right: &std::collections::HashMap<String, BoundedTensor>,
    context: &str,
) {
    assert_eq!(left.len(), right.len(), "{context}: node set changed");
    for (name, lb) in left {
        let rb = right
            .get(name)
            .unwrap_or_else(|| panic!("{context}: node {name} missing"));
        for (side, lv, rv) in [
            ("lower", lb.lower(), rb.lower()),
            ("upper", lb.upper(), rb.upper()),
        ] {
            assert_eq!(lv.len(), rv.len(), "{context}: {name} {side} shape");
            for (a, b) in lv.iter().zip(rv.iter()) {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{context}: {name} {side} bound not bit-identical ({a} vs {b})"
                );
            }
        }
    }
}

#[test]
fn fl_seam_takes_registry_f32_path_and_default_stays_bit_identical() {
    // The test owns this process: no engine may be installed yet, and the seam
    // flag must start unset.
    assert!(
        !fast_f32_gemm::is_installed(),
        "fresh integration-test process must start with an empty registry"
    );
    assert_eq!(std::env::var_os("NY_FORWARD_LINEAR_F32"), None);

    let (graph, input) = fixture();

    // (1) Baseline: default flag, NO registry engine — the historical path.
    let baseline = collect(&graph, &input);

    // (2) Non-CUDA startup registration: installs the faer CPU floor.
    install_cpu_gemm_engine_if_absent();
    assert!(fast_f32_gemm::is_installed(), "startup floor must install");
    let snap = fast_f32_gemm::telemetry_snapshot();
    assert_eq!(snap.calls, 0, "installation alone must not issue calls");
    assert_eq!(
        snap.backend, None,
        "no backend may be reported before the engine materializes"
    );

    // (3) Parity: flag unset (default) with the registry installed. FL must
    // never consult the engine, and every published bound stays bit-identical.
    let default_with_registry = collect(&graph, &input);
    assert_bitwise_equal(
        &baseline,
        &default_with_registry,
        "default flag with registry installed",
    );
    assert_eq!(
        fast_f32_gemm::telemetry_snapshot().calls,
        0,
        "flag-unset forward-linear run must not consult the f32 registry"
    );

    // (4) Seam lit: NY_FORWARD_LINEAR_F32=1 routes the value GEMMs through the
    // registry engine (telemetry counts real calls; provenance is truthful).
    // The flag goes through the blessed serialized choke point (workspace env
    // wall); it was asserted unset above, so the guard restores it to unset as
    // the block ends — the same sequence as the raw set/collect/remove trio.
    let f32_path = {
        let _env_lock = ny_test_utils::env::lock_env();
        let _flag = ny_test_utils::env::ScopedEnvVar::set("NY_FORWARD_LINEAR_F32", "1");
        collect(&graph, &input)
    };
    let snap = fast_f32_gemm::telemetry_snapshot();
    assert!(
        snap.calls > 0,
        "the FL f32 seam must issue registry engine calls when the flag is set"
    );
    assert_eq!(snap.backend, Some("faer-cpu"));

    // (5) Soundness shape of the swap (the seam-test idiom): the f32-path
    // bounds are finite, NEVER tighter than the f64-path bounds, wider only by
    // the charged γ_{K+4}^f32·S + FTZ penalty — small relative to the bound
    // scale on this fixture (γ_{76}^f32 ≈ 4.5e-6) — and strictly wider
    // somewhere (proof the f32 path engaged rather than falling through).
    let mut widened = false;
    for (name, f64_bounds) in &default_with_registry {
        let f32_bounds = &f32_path[name];
        let fl: Vec<f32> = f32_bounds.lower().iter().copied().collect();
        let fu: Vec<f32> = f32_bounds.upper().iter().copied().collect();
        let dl: Vec<f32> = f64_bounds.lower().iter().copied().collect();
        let du: Vec<f32> = f64_bounds.upper().iter().copied().collect();
        for i in 0..dl.len() {
            let (fl_i, fu_i) = (fl[i] as f64, fu[i] as f64);
            let (dl_i, du_i) = (dl[i] as f64, du[i] as f64);
            assert!(
                fl_i.is_finite() && fu_i.is_finite(),
                "{name}[{i}]: f32-path bound must stay finite"
            );
            // Never tighter (containment up to one f32 ULP of slack).
            let slack = 1e-6 * (1.0 + dl_i.abs().max(du_i.abs()));
            assert!(
                fl_i <= dl_i + slack && fu_i >= du_i - slack,
                "{name}[{i}]: f32 seam TIGHTER than f64: [{fl_i}, {fu_i}] vs [{dl_i}, {du_i}]"
            );
            // Wider only by the charged penalty: band the drift at 5e-3 of the
            // bound scale — three orders looser than the γ_{K+4}^f32·S charge
            // on this fixture, three orders tighter than any real regression.
            let band = 5e-3 * (1.0 + dl_i.abs().max(du_i.abs()) + (du_i - dl_i).abs());
            assert!(
                (fl_i - dl_i).abs() <= band && (fu_i - du_i).abs() <= band,
                "{name}[{i}]: f32 seam drifted beyond the charged penalty: \
                 [{fl_i}, {fu_i}] vs [{dl_i}, {du_i}]"
            );
            if fu_i - fl_i > du_i - dl_i {
                widened = true;
            }
        }
    }
    assert!(
        widened,
        "f32 seam produced bounds identical to f64 everywhere — the f32 path \
         did not engage"
    );
}
