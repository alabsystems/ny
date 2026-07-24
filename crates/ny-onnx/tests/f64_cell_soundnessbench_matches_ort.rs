// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0
//
// #sb-rebank lever 2 acceptance: the exact-f64 witness gate must be LIVE on the
// soundnessbench net.
//
// The gate (`f64_forward_rejects_witness`, ny-cli vnncomp.rs) escalates an
// ORT-confirmed `sat` witness to the SOUND f64 interval forward
// (`GraphNetwork::propagate_ibp_f64_cell`) and bails via
// `supports_ibp_f64_cell` when any layer is off the whitelist. The
// soundnessbench model is exactly [Gemm, Relu, Reshape, Conv x6 (+Relu),
// Flatten, Gemm] — Flatten was the ONLY unsupported op, so the gate was inert
// on the whole benchmark. This test pins the gate OPEN and validates the
// Flatten arm end-to-end against ONNX Runtime at 100 random in-box points:
//
//   * `supports_ibp_f64_cell()` is true (the lever);
//   * the point-input f64 enclosure is well-formed and TIGHT (Higham-widened
//     width ~1e-12 at this depth — decisively below the benchmark's planted
//     margins ~1e-5);
//   * ORT's f32 forward sits inside the enclosure widened by an f32-noise
//     allowance (ORT rounds each op to nearest f32; the enclosure bounds the
//     REAL-arithmetic value, so they agree to f32 accumulation error ~1e-5,
//     while a WRONG Flatten — any permutation/misindex — diverges at O(1)).

#![cfg(feature = "ort")]

use ndarray::{ArrayD, IxDyn};
use ny_onnx::diff::{read_input_shape_maybe_gzip, OrtForward};
use ny_onnx::{load_onnx_with_config, GraphNetworkOptions, OnnxLoadConfig};
use ny_propagate::Interval64;

const MODEL: &str = "../../benchmarks/vnncomp2025/benchmarks/soundnessbench/onnx/model.onnx";
const VNNLIB: &str = "../../benchmarks/vnncomp2025/benchmarks/soundnessbench/vnnlib/model_0.vnnlib";

/// Deterministic xorshift stream (no dev-dep), matching the PGD samplers.
struct Rng(u64);
impl Rng {
    fn next_unit(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[test]
fn f64_cell_gate_is_live_and_encloses_ort_on_soundnessbench() {
    if !std::path::Path::new(MODEL).exists() {
        eprintln!("soundnessbench model not present; skipping");
        return;
    }

    let spec = ny_onnx::vnnlib::load_vnnlib(VNNLIB).expect("vnnlib parses");
    let dim = spec.input_bounds.len();
    assert_eq!(dim, 128, "soundnessbench declares 128 inputs");

    // Mirror the gate's own load path: protobuf input shape + graph network.
    let (_bytes, input_shape) =
        read_input_shape_maybe_gzip(std::path::Path::new(MODEL), dim).expect("input shape");
    let model = load_onnx_with_config(MODEL, &OnnxLoadConfig::default()).expect("model loads");
    let graph = model
        .to_graph_network_with_options(GraphNetworkOptions::default())
        .expect("graph network");

    // THE LEVER: with the Flatten arm, the whole-net escalation gate is open.
    assert!(
        graph.supports_ibp_f64_cell(),
        "supports_ibp_f64_cell must accept the soundnessbench net \
         (Gemm/Relu/Reshape/Conv/Flatten) — the exact-f64 witness gate was \
         previously inert on this benchmark"
    );

    let mut forward = OrtForward::from_path(MODEL, dim).expect("ORT forward");
    let mut rng = Rng(0x5b_2025);
    let mut max_width = 0.0f64;
    let mut max_noise = 0.0f64;
    for case in 0..100 {
        let x: Vec<f32> = spec
            .input_bounds
            .iter()
            .map(|&(lo, hi)| {
                let v = lo + rng.next_unit() * (hi - lo);
                (v as f32).clamp(lo as f32, hi as f32)
            })
            .collect();

        let ort = forward.run(&x).expect("ORT run");
        let point64 = ArrayD::from_shape_vec(IxDyn(&input_shape), x.clone())
            .expect("input tensor")
            .mapv(f64::from);
        let out = graph
            .propagate_ibp_f64_cell(&Interval64::point(point64))
            .expect("f64 cell forward");

        assert_eq!(out.lower.len(), ort.len(), "case {case}: output arity");
        for (j, (&o, (&lo, &hi))) in ort
            .iter()
            .zip(out.lower.iter().zip(out.upper.iter()))
            .enumerate()
        {
            assert!(
                lo.is_finite() && hi.is_finite() && lo <= hi,
                "case {case} Y_{j}: malformed enclosure [{lo}, {hi}]"
            );
            let width = hi - lo;
            max_width = max_width.max(width);
            // Point input => the Higham enclosure is tight. A generous 1e-9
            // ceiling still catches any accidental interval blow-up.
            assert!(
                width <= 1e-9,
                "case {case} Y_{j}: enclosure width {width:e} not point-tight"
            );
            // ORT (f32 round-to-nearest per op) vs the REAL-value enclosure:
            // agree to f32 accumulation noise. A wrong Flatten permutes conv
            // activations into the Gemm and diverges at O(0.1..10).
            let o = o as f64;
            let tol = 1e-4 + 1e-4 * o.abs();
            // Signed distance of the ORT value outside the enclosure (0 inside).
            let noise = (lo - o).max(o - hi).max(0.0);
            max_noise = max_noise.max(noise);
            assert!(
                o >= lo - tol && o <= hi + tol,
                "case {case} Y_{j}: ORT {o} outside f64 enclosure [{lo}, {hi}] \
                 by more than the f32-noise allowance {tol:e}"
            );
        }
    }
    eprintln!(
        "100 points enclosed: max enclosure width {max_width:.3e}, \
         max ORT-vs-enclosure distance {max_noise:.3e}"
    );
}
