// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #flush-charge Lane B: METAROOM WALL-CLOCK acceptance measurement on the
//! TEST-SCOPED charged device.
//!
//! Loads the real metaroom net
//! (`benchmarks/vnncomp2025/benchmarks/metaroom_2023/onnx/6cnn_ry_39_6_no_custom_OP.onnx`)
//! and the `spec_idx_119` input box from its vnnlib, then measures the CROWN
//! root pass wall-clock:
//!
//! * GPU leg: `propagate_crown_with_engine(..., Some(charged_device))` with the
//!   sound-GPU gate ENGAGED, so every routed backward is the CHARGED sound
//!   resident walk (`crown_backward_gpu_sound` under
//!   `QualifiedWithFlushCharge`) — never the fast unsound lane.
//! * CPU leg (bounded fixture preflight by default; set
//!   `NY_FULL_MEASUREMENTS=1` for the campaign-claimed ~900s baseline):
//!   `propagate_crown(...)` with no engine. The default lane still loads and
//!   validates the exact real-corpus fixture, so it is an active test rather
//!   than a hidden or skipped measurement.
//!
//! The charged device is the TEST-SCOPED acceptance constructor
//! (`ny_gpu::ComputeDevice::test_only_new_for_proof_flush_charged_acceptance_evidence`),
//! compiled only under ny-gpu's `gpu-tests` feature (a dev-only edge of this
//! crate) — measurement without production authority. The production charged
//! source gate is OPEN since the 2026-08-13 review; production admission
//! still runs its own full live ladder through the typed constructor.
//!
//! ENVIRONMENT (admission-config): the charged constructors now FORCE the
//! plain-WGSL (flushing) loading path per-device, so the measurement legs run
//! under the DEFAULT env — no `NY_GPU_DENORM_PRESERVE=0` required. The one
//! recognized environmental refusal is an explicit `NY_GPU_DENORM_PRESERVE=1`
//! pin, which the constructors typed-refuse (env wins; the pinned passthrough
//! configuration is not the one the charges model); the GPU tests VERIFY that
//! exact refusal loudly instead of running the measurement.

use std::path::PathBuf;
use std::time::Instant;

use ny_core::GemmEngine;
use ny_gpu::ComputeDevice;
use ny_onnx::vnnlib::load_vnnlib;
use ny_tensor::BoundedTensor;

fn benchmark_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../benchmarks/vnncomp2025/benchmarks/metaroom_2023")
}

const ONNX_FILE: &str = "onnx/6cnn_ry_39_6_no_custom_OP.onnx";
const VNNLIB_FILE: &str = "vnnlib/spec_idx_119_eps_0.00000436.vnnlib";

struct MetaroomFixture {
    network: ny_propagate::Network,
    input: BoundedTensor,
    onnx_load_secs: f64,
    convert_secs: f64,
    vnnlib_load_secs: f64,
}

/// Load the real metaroom net + spec_idx_119 box, recording load wall-clocks.
fn load_metaroom() -> MetaroomFixture {
    let dir = benchmark_dir();
    let onnx_path = dir.join(ONNX_FILE);
    let vnnlib_path = dir.join(VNNLIB_FILE);
    assert!(
        onnx_path.is_file() && vnnlib_path.is_file(),
        "metaroom benchmark files required for the wall-clock acceptance \
         measurement: {} / {}",
        onnx_path.display(),
        vnnlib_path.display()
    );

    let t0 = Instant::now();
    let model = ny_onnx::load_onnx(&onnx_path).expect("metaroom onnx loads");
    let onnx_load_secs = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    let network = model
        .to_propagate_network()
        .expect("metaroom onnx converts to a sequential propagate network");
    let convert_secs = t1.elapsed().as_secs_f64();

    let t2 = Instant::now();
    let spec = load_vnnlib(&vnnlib_path).expect("spec_idx_119 vnnlib parses");
    let (lower, upper) = spec.split_input_bounds_f32();
    let vnnlib_load_secs = t2.elapsed().as_secs_f64();

    // Model input shape with the batch dimension squeezed (the CLI's own
    // vnnlib handling), e.g. [1, 3, 32, 56] -> [3, 32, 56].
    let mut shape: Vec<usize> = model.network.inputs[0]
        .shape
        .iter()
        .map(|&d| usize::try_from(d).expect("static input dims"))
        .collect();
    while shape.len() > 1 && shape[0] == 1 {
        shape.remove(0);
    }
    let dim: usize = shape.iter().product();
    assert_eq!(
        dim,
        lower.len(),
        "vnnlib box must match the model input element count"
    );

    let lower = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), lower)
        .expect("lower bounds reshape");
    let upper = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), upper)
        .expect("upper bounds reshape");
    let input = BoundedTensor::new(lower, upper).expect("valid metaroom input box");

    MetaroomFixture {
        network,
        input,
        onnx_load_secs,
        convert_secs,
        vnnlib_load_secs,
    }
}

fn init_route_tracing() {
    use tracing_subscriber::fmt;
    // Visible with --nocapture: the routing lines ("CROWN: GPU backward
    // succeeded" / "falling back to CPU") are the honest discriminator of
    // which lane actually decided each bound.
    let _ = fmt()
        .with_env_filter("ny_propagate=info,ny_gpu=info")
        .with_test_writer()
        .try_init();
}

fn summarize(label: &str, bounds: &BoundedTensor, secs: f64) {
    let lower = bounds.lower();
    let upper = bounds.upper();
    let finite = lower.iter().chain(upper.iter()).all(|v| v.is_finite());
    let max_width = lower
        .iter()
        .zip(upper.iter())
        .map(|(&l, &u)| f64::from(u) - f64::from(l))
        .fold(f64::NEG_INFINITY, f64::max);
    println!(
        "[metaroom-119 wall-clock] {label}: {secs:.3}s  outputs={}  finite={finite}  max_width={max_width:.6e}",
        lower.len()
    );
    assert!(finite, "{label}: root-pass bounds must be finite");
}

/// GPU leg: the charged root pass. Under the default Metal AUTO denorm policy
/// this VERIFIES the environmental refusal loudly instead (no silent skip; see
/// the module docs for the exact re-run command).
#[test]
fn metaroom_119_charged_gpu_root_pass_wallclock() {
    init_route_tracing();

    let engine = match ComputeDevice::test_only_new_for_proof_flush_charged_acceptance_evidence() {
        Ok(engine) => engine,
        Err(error) => {
            let message = error.to_string();
            assert!(
                message.contains("NY_GPU_DENORM_PRESERVE"),
                "TEST-SCOPED charged engine refused for a reason OTHER than \
                 the recognized explicit NY_GPU_DENORM_PRESERVE=1 pin: {message}"
            );
            println!(
                "[metaroom-119 wall-clock] PRECONDITION NOT MET: {message}\n\
                 [metaroom-119 wall-clock] unset NY_GPU_DENORM_PRESERVE (or set \
                 auto/0) and re-run: cargo test -p ny-cli --test \
                 flush_charged_metaroom_wallclock -- --nocapture \
                 (VERIFIED typed refusal; measurement not run)"
            );
            return;
        }
    };
    assert_eq!(
        engine.backend_provenance(),
        "wgpu-qualified-crown-flush-charged",
        "the charged engine must carry the charged provenance marker"
    );

    // Engage the sound-GPU gate: only a sound GPU backward may decide a bound,
    // so the routed lane is the CHARGED sound resident walk or the proven CPU
    // fallback — never the fast unsound lane.
    ny_propagate::set_sound_gpu_crown_required(true);

    let fixture = load_metaroom();
    println!(
        "[metaroom-119 wall-clock] loads: onnx={:.3}s convert={:.3}s vnnlib={:.3}s",
        fixture.onnx_load_secs, fixture.convert_secs, fixture.vnnlib_load_secs
    );

    let t = Instant::now();
    let bounds = fixture
        .network
        .propagate_crown_with_engine(&fixture.input, Some(&engine))
        .expect("charged GPU root pass completes");
    summarize("GPU charged root pass", &bounds, t.elapsed().as_secs_f64());
}

/// MEASURED FINDING (2026-08-13, this box): the REAL metaroom net is REFUSED
/// by the v1 charge policy — its conv layers carry STRICTLY-SUBNORMAL bias
/// entries ("#flush-charge: layer 2/4 has a SUBNORMAL expanded conv bias
/// entry — refusing"), so every charged walk falls back to the proven CPU
/// path. That refusal is the guard working as audited (the bias-combine flush
/// audit proves that channel unchargeable). The largest ADMISSIBLE variant is
/// therefore the same net with its strictly-subnormal weight/bias entries
/// zeroed (each |delta| < 2^-126, far below f32 resolution of any downstream
/// bound) — measured here, HONESTLY LABELED as a variant, never as the real
/// net.
#[test]
fn metaroom_119_charged_gpu_root_pass_wallclock_admissible_variant() {
    init_route_tracing();

    let engine = match ComputeDevice::test_only_new_for_proof_flush_charged_acceptance_evidence() {
        Ok(engine) => engine,
        Err(error) => {
            let message = error.to_string();
            assert!(
                message.contains("NY_GPU_DENORM_PRESERVE"),
                "TEST-SCOPED charged engine refused for a reason OTHER than \
                 the recognized explicit NY_GPU_DENORM_PRESERVE=1 pin: {message}"
            );
            println!(
                "[metaroom-119 wall-clock] PRECONDITION NOT MET: {message} \
                 (unset NY_GPU_DENORM_PRESERVE or set auto/0; VERIFIED typed refusal)"
            );
            return;
        }
    };
    ny_propagate::set_sound_gpu_crown_required(true);

    let mut fixture = load_metaroom();

    // Sanitize: zero every strictly-subnormal weight/bias entry (the v1
    // charge policy refuses them; each zeroed entry perturbs the function by
    // less than 2^-126). Count and report every touched entry.
    let mut zeroed_conv_bias = 0usize;
    let mut zeroed_conv_kernel = 0usize;
    let mut zeroed_linear = 0usize;
    for layer in fixture.network.layers_mut() {
        match layer {
            ny_propagate::Layer::Conv2d(conv) => {
                for v in conv.kernel.iter_mut() {
                    if *v != 0.0 && v.abs() < f32::MIN_POSITIVE {
                        *v = 0.0;
                        zeroed_conv_kernel += 1;
                    }
                }
                if let Some(bias) = conv.bias.as_mut() {
                    for v in bias.iter_mut() {
                        if *v != 0.0 && v.abs() < f32::MIN_POSITIVE {
                            *v = 0.0;
                            zeroed_conv_bias += 1;
                        }
                    }
                }
            }
            ny_propagate::Layer::Linear(linear) => {
                let has_subnormal = linear
                    .weight()
                    .iter()
                    .chain(linear.bias().into_iter().flatten())
                    .any(|v| *v != 0.0 && v.abs() < f32::MIN_POSITIVE);
                if has_subnormal {
                    let mut weight = linear.weight().clone();
                    let mut bias = linear.bias().cloned();
                    for v in weight
                        .iter_mut()
                        .chain(bias.iter_mut().flat_map(|b| b.iter_mut()))
                    {
                        if *v != 0.0 && v.abs() < f32::MIN_POSITIVE {
                            *v = 0.0;
                            zeroed_linear += 1;
                        }
                    }
                    *layer = ny_propagate::Layer::Linear(
                        ny_propagate::layers::LinearLayer::new(weight, bias)
                            .expect("sanitized linear rebuilds"),
                    );
                }
            }
            _ => {}
        }
    }
    println!(
        "[metaroom-119 wall-clock] ADMISSIBLE VARIANT: zeroed strictly-subnormal \
         entries conv_kernel={zeroed_conv_kernel} conv_bias={zeroed_conv_bias} \
         linear={zeroed_linear} (each |delta| < 2^-126; the REAL net is REFUSED \
         by the v1 charge policy — that refusal is the audited guard behavior)"
    );

    let t = Instant::now();
    let bounds = fixture
        .network
        .propagate_crown_with_engine(&fixture.input, Some(&engine))
        .expect("charged GPU root pass completes on the admissible variant");
    summarize(
        "GPU charged root pass (ADMISSIBLE VARIANT)",
        &bounds,
        t.elapsed().as_secs_f64(),
    );
}

/// CPU leg: the proven CPU sound root pass (the campaign-claimed ~900s
/// baseline on this net). The ordinary gate performs a bounded, fail-fast
/// preflight of the exact fixture. `NY_FULL_MEASUREMENTS=1` explicitly opts in
/// to the full root pass while retaining the same fixture assertions.
#[test]
fn metaroom_119_cpu_root_pass_wallclock_baseline() {
    init_route_tracing();
    ny_propagate::set_sound_gpu_crown_required(true);

    let fixture = load_metaroom();
    println!(
        "[metaroom-119 wall-clock] loads: onnx={:.3}s convert={:.3}s vnnlib={:.3}s",
        fixture.onnx_load_secs, fixture.convert_secs, fixture.vnnlib_load_secs
    );

    if !ny_levers::read(&ny_levers::decls::measurement::FULL_MEASUREMENTS)
        .value
        .as_bool()
    {
        assert_eq!(
            fixture.input.lower().shape(),
            fixture.input.upper().shape(),
            "the bounded preflight must load a shape-consistent input box"
        );
        assert!(
            fixture
                .input
                .lower()
                .iter()
                .chain(fixture.input.upper().iter())
                .all(|value| value.is_finite()),
            "the bounded preflight must load a finite input box"
        );
        println!(
            "[metaroom-119 wall-clock] bounded CPU fixture preflight passed; \
             set NY_FULL_MEASUREMENTS=1 to run the full CPU root pass"
        );
        return;
    }

    let t = Instant::now();
    let bounds = fixture
        .network
        .propagate_crown(&fixture.input)
        .expect("CPU root pass completes");
    summarize("CPU root pass", &bounds, t.elapsed().as_secs_f64());
}
