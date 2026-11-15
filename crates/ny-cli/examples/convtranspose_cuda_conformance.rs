// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Explicit real-device conformance; never an ordinary CPU-CI test or benchmark.
//! Missing CUDA, construction failure, missed dispatch, or an oracle failure
//! makes this executable fail. Run in a fresh process: the sound engine is a
//! process-global once cell. No environment or production source is modified.

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use num_rational::BigRational;
use ny_core::{GemmEngine, NyError, Result};
use ny_cuda::CudaGemmEngine;
use ny_propagate::layers::ConvTranspose2dLayer;
use ny_propagate::LinearBounds;
use std::ffi::OsStr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const CAP: usize = 1 << 24;
const NORMAL: usize = 0;
const REFUSE_SECOND: usize = 1;
const DEADLINE_SECOND: usize = 2;
const EXPIRE_AFTER_SECOND: usize = 3;

#[derive(Clone, Copy, Debug)]
struct Call {
    m: usize,
    k: usize,
    n: usize,
    cap: usize,
}

struct ObservedCuda {
    device: CudaGemmEngine,
    mode: AtomicUsize,
    attempted: AtomicUsize,
    completed: AtomicUsize,
    ordinary: AtomicUsize,
    shapes: Mutex<Vec<Call>>,
}

impl ObservedCuda {
    fn reset(&self, mode: usize) {
        self.mode.store(mode, Ordering::SeqCst);
        self.attempted.store(0, Ordering::SeqCst);
        self.completed.store(0, Ordering::SeqCst);
        self.ordinary.store(0, Ordering::SeqCst);
        self.shapes.lock().expect("shape recorder").clear();
    }

    fn counts(&self, attempts: usize, completed: usize) {
        assert_eq!(self.attempted.load(Ordering::SeqCst), attempts);
        assert_eq!(self.completed.load(Ordering::SeqCst), completed);
        assert_eq!(self.ordinary.load(Ordering::SeqCst), 0);
    }

    fn check_shapes(&self, f: &Fixture) {
        let recorded = self.shapes.lock().expect("shape recorder");
        assert_eq!(recorded.len(), 2, "both lower/upper bounded calls required");
        for call in recorded.iter() {
            assert_eq!((call.m, call.k, call.n), f.gemm_shape());
            assert_eq!(call.cap, CAP, "the caller must not widen its cap");
            assert!(call.m * call.k * call.n > CAP);
        }
    }
}

impl GemmEngine for ObservedCuda {
    fn backend_provenance(&self) -> &'static str {
        "cuda-conformance-bounded-only-delegate"
    }

    fn gemm_f32(&self, _: usize, _: usize, _: usize, _: &[f32], _: &[f32]) -> Result<Vec<f32>> {
        self.ordinary.fetch_add(1, Ordering::SeqCst);
        panic!("public finite ConvTranspose caller entered ordinary f32 GEMM")
    }

    fn gemm_f64(&self, _: usize, _: usize, _: usize, _: &[f64], _: &[f64]) -> Result<Vec<f64>> {
        self.ordinary.fetch_add(1, Ordering::SeqCst);
        panic!("public finite ConvTranspose caller entered ordinary f64 GEMM")
    }

    fn gemm_f64_with_deadline(
        &self,
        m: usize,
        k: usize,
        n: usize,
        a: &[f64],
        b: &[f64],
        deadline: Instant,
        cap: usize,
    ) -> Result<Vec<f64>> {
        let call = self.attempted.fetch_add(1, Ordering::SeqCst) + 1;
        self.shapes
            .lock()
            .expect("shape recorder")
            .push(Call { m, k, n, cap });
        assert_eq!(cap, CAP);
        assert_eq!(a.len(), m * k);
        assert_eq!(b.len(), k * n);
        let mode = self.mode.load(Ordering::SeqCst);
        if call == 2 && mode == REFUSE_SECOND {
            return Err(NyError::UnsupportedOp(
                "conformance: refuse second dispatch".into(),
            ));
        }
        if call == 2 && mode == DEADLINE_SECOND {
            return Err(NyError::DeadlineExceeded(
                "conformance: second dispatch deadline".into(),
            ));
        }
        // The real constructed backend executes every non-injected dispatch.
        let answer = self
            .device
            .gemm_f64_with_deadline(m, k, n, a, b, deadline, cap)?;
        self.completed.fetch_add(1, Ordering::SeqCst);
        if call == 2 && mode == EXPIRE_AFTER_SECOND {
            // Deliberately violate only the wrapper's return-time obligation,
            // after real successful arithmetic, to test the caller's barrier.
            std::thread::sleep(
                deadline.saturating_duration_since(Instant::now()) + Duration::from_millis(1),
            );
            assert!(Instant::now() >= deadline);
        }
        Ok(answer)
    }
}

struct Fixture {
    name: &'static str,
    layer: ConvTranspose2dLayer,
    incoming: LinearBounds,
    in_c: usize,
    out_c: usize,
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
    cancellation: bool,
}

impl Fixture {
    fn gemm_shape(&self) -> (usize, usize, usize) {
        let shape = self.layer.kernel.shape();
        (
            self.in_h * self.in_w,
            self.out_c * shape[2] * shape[3],
            self.in_c,
        )
    }

    fn propagate(&self, deadline: Instant) -> Result<LinearBounds> {
        // Passing None is intentional: the new pair must use the installed
        // sound-f64 engine, not a caller-supplied generic engine capability.
        self.layer
            .propagate_linear_with_engine_and_deadline(&self.incoming, None, Some(deadline))
            .map(|result| result.into_owned())
    }

    fn exact(&self, upper: bool, channel: usize, y: usize, x: usize) -> BigRational {
        if self.cancellation {
            // Three exactly represented channels: 2^53 + 1 - 2^53 = 1;
            // upper side: 2^24 + 1, which cannot be stored in f32.
            // Every kernel channel in this column has the same dyadic weight.
            let numerator = if upper { (1i64 << 24) + 1 } else { 1 };
            return BigRational::new((numerator * (channel as i64 % 5 + 1)).into(), 8.into());
        }
        // Independent direct convolution in integer units of 1/128; no
        // im2col, GEMM reduction, or implementation index helper is reused.
        let a = if upper {
            self.incoming.upper_a()
        } else {
            self.incoming.lower_a()
        };
        let k = &self.layer.kernel;
        let mut numerator = 0i64;
        for c in 0..self.out_c {
            for ky in 0..k.shape()[2] {
                for kx in 0..k.shape()[3] {
                    let iy = y as isize * self.layer.stride.0 as isize
                        + ky as isize * self.layer.dilation.0 as isize
                        - self.layer.padding.0 as isize;
                    let ix = x as isize * self.layer.stride.1 as isize
                        + kx as isize * self.layer.dilation.1 as isize
                        - self.layer.padding.1 as isize;
                    if iy >= 0
                        && ix >= 0
                        && (iy as usize) < self.out_h
                        && (ix as usize) < self.out_w
                    {
                        let av = a[[0, (c * self.out_h + iy as usize) * self.out_w + ix as usize]];
                        numerator += (av * 8.0) as i64 * (k[[channel, c, ky, kx]] * 16.0) as i64;
                    }
                }
            }
        }
        BigRational::new(numerator.into(), 128.into())
    }

    fn check(&self, result: &LinearBounds) {
        let outputs = self.in_c * self.in_h * self.in_w;
        assert_eq!(result.lower_a().dim(), (1, outputs));
        assert_eq!(result.upper_a().dim(), (1, outputs));
        assert!(result.lower_b().iter().all(|x| *x == 0.0));
        assert!(result.upper_b().iter().all(|x| *x == 0.0));
        for (upper, a, error) in [
            (
                false,
                result.lower_a(),
                result.lower_a_err().expect("real lower error channel"),
            ),
            (
                true,
                result.upper_a(),
                result.upper_a_err().expect("real upper error channel"),
            ),
        ] {
            assert_eq!(a.dim(), error.dim());
            for c in 0..self.in_c {
                for y in 0..self.in_h {
                    for x in 0..self.in_w {
                        let column = (c * self.in_h + y) * self.in_w + x;
                        let value = a[[0, column]];
                        let err = error[[0, column]];
                        assert!(value.is_finite() && err.is_finite() && err >= 0.0);
                        let center =
                            BigRational::from_float(f64::from(value)).expect("finite coefficient");
                        let radius = BigRational::from_float(f64::from(err)).expect("finite error");
                        let exact = self.exact(upper, c, y, x);
                        assert!(&center - &radius <= exact && exact <= &center + &radius,
                            "{} upper={upper} c={c} y={y} x={x}: stored/error excludes exact coefficient",
                            self.name);
                        if self.cancellation && upper {
                            assert_ne!(
                                center, exact,
                                "cast-loss case must actually lose information"
                            );
                            assert!(err > 0.0);
                        }
                    }
                }
            }
        }
    }
}

fn cancellation_fixture() -> Fixture {
    let (in_c, out_c, in_h, in_w) = (256, 256, 16, 17);
    let spatial = in_h * in_w;
    let kernel =
        ArrayD::from_shape_fn(IxDyn(&[in_c, out_c, 1, 1]), |i| (i[0] % 5 + 1) as f32 / 8.0);
    let lower = Array2::from_shape_fn((1, out_c * spatial), |(_, p)| match p / spatial {
        0 => 2.0f32.powi(53),
        1 => 1.0,
        2 => -2.0f32.powi(53),
        _ => 0.0,
    });
    let upper = Array2::from_shape_fn((1, out_c * spatial), |(_, p)| match p / spatial {
        0 => 2.0f32.powi(24),
        1 => 1.0,
        _ => 0.0,
    });
    Fixture {
        name: "cancellation-and-cast-1x1",
        in_c,
        out_c,
        in_h,
        in_w,
        out_h: in_h,
        out_w: in_w,
        cancellation: true,
        layer: ConvTranspose2dLayer::with_input_shape(kernel, None, (1, 1), (0, 0), in_h, in_w)
            .expect("valid layer"),
        incoming: LinearBounds::new(lower, Array1::zeros(1), upper, Array1::zeros(1))
            .expect("valid incoming affine maps"),
    }
}

fn spatial_fixture() -> Fixture {
    let (in_c, out_c, in_h, in_w, out_h, out_w) = (64, 64, 30, 31, 60, 33);
    let kernel = ArrayD::from_shape_fn(IxDyn(&[in_c, out_c, 3, 3]), |i| {
        ((i[0] * 11 + i[1] * 7 + i[2] * 5 + i[3] * 3) % 17) as f32 / 16.0 - 0.5
    });
    let mut layer = ConvTranspose2dLayer::new_full(kernel, None, (2, 1), (1, 1), (1, 2), (1, 0))
        .expect("valid spatial layer");
    layer.set_input_shape(in_h, in_w);
    let lower = Array2::from_shape_fn((1, out_c * out_h * out_w), |(_, p)| {
        let x = p % out_w;
        let y = p / out_w % out_h;
        let c = p / (out_h * out_w);
        ((c * 7 + y * 5 + x * 3) % 23) as f32 / 8.0 - 1.375
    });
    let upper = lower.mapv(|x| -x + 0.25);
    Fixture {
        name: "rectangular-stride-padding-dilation",
        layer,
        in_c,
        out_c,
        in_h,
        in_w,
        out_h,
        out_w,
        cancellation: false,
        incoming: LinearBounds::new(lower, Array1::zeros(1), upper, Array1::zeros(1))
            .expect("valid incoming affine maps"),
    }
}

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let disabled = args == ["--expect-disabled"];
    assert!(
        args.is_empty() || disabled,
        "only optional --expect-disabled is accepted"
    );
    let policy_inputs = ny_levers::RawLeverInputs::capture(ny_levers::all());
    let gate = policy_inputs.get(&ny_levers::decls::kernel_inputs::CONVTRANSPOSE_SOUND_F64_GPU);
    if disabled {
        assert_eq!(
            gate,
            Some(OsStr::new("0")),
            "disabled control requires literal kill switch"
        );
    } else {
        assert!(
            gate.is_none() || gate == Some(OsStr::new("1")),
            "device control requires enabled finite policy"
        );
    }
    assert_ne!(
        policy_inputs.get(&ny_levers::decls::kernel_inputs::CONV_SKIP_DEAD_F32),
        Some(OsStr::new("0")),
        "caller f64 pair must be enabled"
    );
    assert!(
        !ny_propagate::sound_f64_gemm::is_installed(),
        "fresh executable owns factory installation"
    );
    let device = CudaGemmEngine::new()
        .expect("REQUIRED real CUDA device and constructed IEEE-f64 authority");
    eprintln!(
        "[convtranspose-cuda-conformance] device={:?} transport={} required=true",
        device.device_name(),
        device.deadline_f64_transport_name()
    );
    let engine = Arc::new(ObservedCuda {
        device,
        mode: AtomicUsize::new(NORMAL),
        attempted: AtomicUsize::new(0),
        completed: AtomicUsize::new(0),
        ordinary: AtomicUsize::new(0),
        shapes: Mutex::new(Vec::new()),
    });
    ny_propagate::sound_f64_gemm::set_sound_f64_gemm_engine(engine.clone());
    let first = cancellation_fixture();
    let second = spatial_fixture();
    for f in [&first, &second] {
        engine.reset(NORMAL);
        let result = f
            .propagate(Instant::now() + Duration::from_secs(60))
            .expect("complete public caller");
        f.check(&result);
        if disabled {
            engine.counts(0, 0);
        } else {
            engine.counts(2, 2);
            engine.check_shapes(f);
        }
        eprintln!("[convtranspose-cuda-conformance] case={} oracle=all-coefficients-exact-rational passed=true attempted={} completed={}", f.name, engine.attempted.load(Ordering::SeqCst), engine.completed.load(Ordering::SeqCst));
    }
    if !disabled {
        engine.reset(REFUSE_SECOND);
        let fallback = first
            .propagate(Instant::now() + Duration::from_secs(60))
            .expect("non-deadline refusal must retain CPU fallback");
        engine.counts(2, 1);
        first.check(&fallback);
        engine.reset(DEADLINE_SECOND);
        let error = first
            .propagate(Instant::now() + Duration::from_secs(60))
            .expect_err("typed deadline must not fall through to CPU");
        assert!(error.is_deadline_exceeded());
        engine.counts(2, 1);
        engine.reset(EXPIRE_AFTER_SECOND);
        let error = first
            .propagate(Instant::now() + Duration::from_secs(3))
            .expect_err("successful late pair must not publish");
        assert!(error.is_deadline_exceeded());
        engine.counts(2, 2);
        engine.reset(NORMAL);
        let error = first
            .propagate(
                Instant::now()
                    .checked_sub(Duration::from_millis(1))
                    .expect("test clock supports an expired deadline"),
            )
            .expect_err("already-expired caller must not dispatch");
        assert!(error.is_deadline_exceeded());
        engine.counts(0, 0);
        eprintln!("[convtranspose-cuda-conformance] fallback=passed typed-deadline=passed post-dispatch-expiry=passed expired-entry=passed");
    }
    eprintln!("[convtranspose-cuda-conformance] accepted=true disabled_control={disabled} benchmark=false");
}
