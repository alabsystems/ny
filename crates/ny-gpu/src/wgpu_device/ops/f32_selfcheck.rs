// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Per-adapter IEEE-754 f32-model diagnostic and future capability prerequisite
//! (#gpu-f32-selfcheck).
//!
//! # Why this exists (soundness)
//!
//! Candidate sound GPU-resident CROWN/IBP kernels rely on a conformant
//! IEEE-754 f32 backend:
//! Higham `γ_k` reduction radii, NORMAL-range (FTZ-safe) underflow floors, and
//! integer-bitcast directed rounding (`round_up_pos(x) = bitcast<f32>(bitcast<u32>(x)
//! + 1u)`). The ONE residual assumption is that the LIVE GPU adapter actually
//! executes WGSL f32 at true unit roundoff `u = 2^-24` with:
//!   * NO covert reduced precision (bf16/tf32 truncated mantissa),
//!   * NO fast-math reassociation that breaks the round-up-positive directed
//!     rounding, and
//!   * IEEE bit-exact `bitcast<u32>` / `bitcast<f32>`.
//!
//! This probe checks that prerequisite per adapter rather than assuming it (see
//! `docs/F32_ABSSUM_SEAM.md` §5.1 on the cuBLAS-TF32 concern). This module runs a
//! tiny one-time compute shader that measures those exact primitives and compares
//! the on-device bit patterns byte-for-byte against a CPU reference. On ANY mismatch,
//! readback error, or dispatch error it returns `false` (conservative and fail-closed).
//! A passing result is necessary but not sufficient for verdict authority.
//!
//! # Where this probe's result is CONSUMED (updated #gpu-typed-authority)
//!
//! This is rung 1 of the qualification ladder in `ops/sound_authority.rs`.
//! The U1/U3/U4/U5/U6 ledger and B0 review are discharged and the raw-device
//! source gate is open. The typed explicit verdict constructor plus all five
//! passing live probes may therefore expose the exact `WgpuDevice` sound-CROWN
//! seam; this rung alone never grants authority. Ordinary device construction
//! remains unarmed.
//!
//! `provides_sound_gpu_ibp` and `provides_sound_gpu_dag_ibp` DO remain
//! unconditionally `false`. The retained diagnostic ladder pertains to the
//! CROWN backward seam, which is the only one with a compensated-error channel.
//!
//! # The alleged contradiction with `crown_partial_gpu.rs` — RESOLVED, there is none
//!
//! `docs/MACOS_ACCELERATION_RESEARCH_2026-08-01.md` flagged this header's former
//! claim ("WGPU remains unconditionally quarantined; all `provides_sound_gpu_*`
//! return `false`") as contradicted by a "live sound-resident dispatch" at
//! `ny-propagate .../network/ibp/crown_partial_gpu.rs:150-160`. Traced
//! 2026-08-04: **both statements were true and there was no contradiction.**
//! That dispatch is `if use_sound { crown_backward_gpu_sound(..) } else {
//! crown_backward_gpu(..) }`, and `use_sound` is not a local decision — it is the
//! second element of the tuple returned by `gpu_crown_backward_route_with_deadline`,
//! which grants `true` only after `.filter(|g| g.provides_sound_gpu_crown())`.
//! At that checkpoint the sound-resident dispatch was reachable **on CUDA** and
//! statically unreachable on wgpu, so there was no quarantine leak. Since the
//! 2026-08-11 B0 review, a `WgpuDevice` can pass the same filter only after its
//! typed explicit constructor records a complete passing live ladder. The
//! `ComputeDevice` proof constructor delegates only that qualified CROWN
//! accessor. No unfiltered sound dispatch exists.
//!
//! # "WGSL has no f64" is NOT an authority blocker
//!
//! Struck across the tree 2026-08-04. The literal fact is true, but the sound
//! lane's certified error term has never needed f64: it is Higham `γ_k·S` in
//! pure f32 with host-side outward-rounded uniforms (`sound_consts.rs:13` —
//! "no f64 ever enters a WGSL body"), and the EFT channel adds an f64-grade
//! compensated residual, whose two primitives measured bit-exact on an Apple
//! M5 Max / Metal adapter on 2026-08-04 (fma TwoProduct 509/509 lanes,
//! fma-barrier TwoSum 307/307, 0.000e0 ULP). Bit-exact primitives are NOT
//! verdict authority. The later U1/U3/U4/U5/U6 discharges and B0 decision are
//! recorded in `ops/sound_authority.rs` and
//! `docs/CURRENT_STATE_2026-08-10.md`.
//!
//! # The probe (shader [`IEEE_F32_SELFCHECK_SHADER`], one single-thread dispatch)
//! Every operand is read from a storage buffer at runtime so the shader compiler
//! CANNOT constant-fold the arithmetic (which would silently defeat the
//! reduced-precision probes).
//!
//! 1. REDUCED-PRECISION:
//!    - `1.0 + 2^-23` must be the next f32 above 1.0 = `0x3F800001`. A bf16
//!      (8-bit) or tf32 (10-bit) mantissa rounds it back to `1.0 = 0x3F800000`.
//!    - product `(1 + 2^-12)^2` — under tf32, `1 + 2^-12` already rounds to `1.0`,
//!      so the product collapses to `1.0`, differing from the true-f32 `0x3F801000`.
//!    - a length-16 sequential accumulation `1.0 + Σ 2^-23`, exact in f32
//!      (`0x3F800010`) but lossy under reduced precision (stays `1.0`).
//!
//!    Each is compared BIT-EXACT to the CPU f32 result.
//! 2. DIRECTED-ROUNDING: for several positive `x` (and the `x ≤ 0` edge), the device
//!    `round_up_pos(x)` must equal the CPU `bitcast<u32>(x) + 1` — confirming the
//!    `bitcast` round-trips and the `+1u` integer add are bit-exact (the load-bearing
//!    outward-rounding primitive). Compared BIT-EXACT.
//! 3. FMA / reassociation sanity: `a*b + c` with catastrophic cancellation, checked
//!    to be WITHIN a generous within-model tolerance of the true (f64) value — NOT
//!    bit-exact, because FMA contraction (a single rounding) is ALLOWED and covered by
//!    `γ_k`. Only a GROSS deviation (fast-math run amok / reduced precision) fails it.

use ny_core::Result;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

use super::super::WgpuDevice;
use super::ibp_forward::create_buffer;

/// Uniform params for [`IEEE_F32_SELFCHECK_SHADER`]. 16 bytes, std140-clean. Layout
/// MUST match the WGSL `struct Params` exactly.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct F32SelfCheckParams {
    /// Iterations of the exact-in-f32 accumulation probe (runtime-bounded so the
    /// loop cannot be unrolled + constant-folded).
    acc_iters: u32,
    /// Count of directed-rounding inputs.
    n_round: u32,
    /// Index in `inp` where the directed-rounding inputs begin.
    round_base: u32,
    /// Index in `outp` where the directed-rounding results begin.
    out_round_base: u32,
}

/// Input operand bit patterns (host-provided; read at runtime by the shader). The
/// CPU reference below recomputes every expectation from these SAME bits, so there
/// is one source of truth for the operands.
const INP: [u32; 11] = [
    0x3F80_0000, // [0] 1.0
    0x3400_0000, // [1] 2^-23  (ULP of 1.0)
    0x3F80_0800, // [2] 1 + 2^-12
    0xBF80_1000, // [3] -(1 + 2^-11)
    // directed-rounding inputs [ROUND_BASE .. ROUND_BASE + N_ROUND):
    0x3F80_0000, // 1.0
    0x3400_0000, // 2^-23 (tiny positive)
    0x0C00_0000, // 2^-103 (small NORMAL — survives FTZ; not a subnormal)
    0x7148_1CC8, // ~9.9e29 (large finite)
    0x0000_0001, // smallest positive subnormal → normal FTZ-safe floor
    0x0000_0000, // +0.0  → round_up_pos returns 0
    0xBF80_0000, // -1.0  → negative input returns 0
];
const ACC_ITERS: u32 = 16;
const N_ROUND: u32 = 7;
const ROUND_BASE: u32 = 4;
const OUT_ROUND_BASE: u32 = 3;
/// 3 scalar probes (add/prod/acc) + `N_ROUND` directed-rounding + 1 FMA.
const OUT_LEN: usize = OUT_ROUND_BASE as usize + N_ROUND as usize + 1; // = 11

/// Generous within-model tolerance for the FMA/`a*b+c` probe. The legitimate spread
/// between an FMA (single rounding) and a two-step (double rounding) evaluation of
/// the probe is ≤ `2^-24 ≈ 6e-8`; a covert reduced-precision run misses the
/// cancellation by ≈ `2^-11 ≈ 4.9e-4`. `1e-4` sits far above the former and below the
/// latter, so a conformant adapter always passes and a grossly-broken one always fails.
const FMA_TOL: f64 = 1.0e-4;

/// WGSL source of the one-time IEEE-754 f32-model probe. Its OWN dedicated shader —
/// it does NOT touch the shared enclosure shaders (`shaders.rs`). `round_up_pos` is a
/// LOCAL copy that is byte-identical to `IBP_SOUND_PRELUDE::round_up_pos`, so the
/// probe verifies the exact primitive the enclosure relies on without importing it.
const IEEE_F32_SELFCHECK_SHADER: &str = r#"
struct Params { acc_iters: u32, n_round: u32, round_base: u32, out_round_base: u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read>       inp:  array<u32>;
@group(0) @binding(2) var<storage, read_write>  outp: array<u32>;

// Byte-identical to shaders.rs IBP_SOUND_PRELUDE::round_up_pos (the load-bearing
// outward-rounding primitive): a bit-classified successor with a normal FTZ floor.
fn round_up_pos(x: f32) -> f32 {
    let bits = bitcast<u32>(x);
    let magnitude = bits & 0x7fffffffu;
    if (magnitude >= 0x7f800000u) { return x; }
    if ((bits & 0x80000000u) != 0u || magnitude == 0u) { return 0.0; }
    if (magnitude < 0x00800000u) { return 1.1754944e-38; }
    return bitcast<f32>(bits + 1u);
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) { return; }

    // Operands via storage-buffer reads → the compiler cannot constant-fold them.
    let one  = bitcast<f32>(inp[0]);
    let ulp1 = bitcast<f32>(inp[1]);   // 2^-23
    let c12  = bitcast<f32>(inp[2]);   // 1 + 2^-12
    let cneg = bitcast<f32>(inp[3]);   // -(1 + 2^-11)

    // Probe 1a: reduced-precision add. 1.0 + 2^-23 must be the next f32 above 1.0.
    outp[0] = bitcast<u32>(one + ulp1);

    // Probe 1b: reduced-precision product. (1 + 2^-12)^2.
    outp[1] = bitcast<u32>(c12 * c12);

    // Probe 1c: length-N accumulation, exact in f32 but lossy under reduced precision.
    var acc: f32 = one;
    for (var i: u32 = 0u; i < params.acc_iters; i = i + 1u) {
        acc = acc + ulp1;
    }
    outp[2] = bitcast<u32>(acc);

    // Probe 2: directed-rounding primitive (bitcast round-trip + integer +1u).
    for (var j: u32 = 0u; j < params.n_round; j = j + 1u) {
        let x = bitcast<f32>(inp[params.round_base + j]);
        outp[params.out_round_base + j] = bitcast<u32>(round_up_pos(x));
    }

    // Probe 3: FMA / a*b + c sanity (within-model bound; checked on the host).
    outp[params.out_round_base + params.n_round] = bitcast<u32>(c12 * c12 + cneg);
}
"#;

/// Test/operator hook: force [`WgpuDevice::verify_ieee_f32_model`] to report FAILURE.
/// Set via [`set_force_f32_selfcheck_fail`] (tests) or the `NY_FORCE_GPU_F32_SELFCHECK_FAIL`
/// env var (operators). Read on every `verify_ieee_f32_model` call (NOT cached in the
/// device) so a test can flip it without the cached real result masking it.
static TEST_FORCE_FAIL: AtomicBool = AtomicBool::new(false);

/// Read the `NY_FORCE_GPU_F32_SELFCHECK_FAIL` env var ONCE. Operators set this before
/// process start to defensively pin the box onto the CPU-sound path; caching it keeps
/// `provides_sound_gpu_*` off the hot-path env lookup.
fn env_forces_fail() -> bool {
    static ENV: OnceLock<bool> = OnceLock::new();
    *ENV.get_or_init(|| std::env::var_os("NY_FORCE_GPU_F32_SELFCHECK_FAIL").is_some())
}

/// Whether the self-check is currently forced to fail (test hook OR operator env).
fn selfcheck_forced_to_fail() -> bool {
    TEST_FORCE_FAIL.load(Ordering::Relaxed) || env_forces_fail()
}

/// Never-opening visibility for higher-level authority predicates. The hook
/// can only revoke a previously measured prerequisite.
pub(crate) fn probe_forced_to_fail() -> bool {
    selfcheck_forced_to_fail()
}

/// Test hook: force / release a self-check failure to exercise the fail-safe route
/// without a non-conformant adapter. Process-global; tests hold `gpu_test_serial_guard`
/// while it is set so it cannot leak into a concurrent test. Gated to the gpu-tests
/// build (its only caller), so a plain `cargo test` sees no unused item.
#[cfg(all(test, feature = "gpu-tests"))]
pub(crate) fn set_force_f32_selfcheck_fail(force: bool) {
    TEST_FORCE_FAIL.store(force, Ordering::Relaxed);
}

/// CPU reference for the WGSL `round_up_pos` (mirrors its integer classification
/// exactly, including the normal floor for a positive nonzero subnormal).
fn round_up_pos_ref(bits: u32) -> u32 {
    let magnitude = bits & 0x7fff_ffff;
    if magnitude >= 0x7f80_0000 {
        return bits;
    }
    if bits & 0x8000_0000 != 0 || magnitude == 0 {
        return 0;
    }
    if magnitude < 0x0080_0000 {
        return 0x0080_0000;
    }
    bits + 1
}

impl WgpuDevice {
    /// One-time per-adapter f32 conformance diagnostic.
    ///
    /// `true` proves only the tested arithmetic primitives. It does not
    /// authorize verdict use: the independent WGPU production-authority
    /// quarantine remains engaged regardless. The result is cached once per
    /// device.
    pub(crate) fn verify_ieee_f32_model(&self) -> bool {
        // Fail-safe override (tests + operators): report failure without consulting —
        // or polluting — the cached real result, so the fallback path is exercised.
        if selfcheck_forced_to_fail() {
            return false;
        }
        *self
            .f32_selfcheck
            .get_or_init(|| self.run_ieee_f32_selfcheck())
    }

    /// Never-initializing cached read of [`Self::verify_ieee_f32_model`], for
    /// call sites already INSIDE a GPU-checked section (first-initializing the
    /// probe there would self-deadlock on the enclosing lock — same hazard the
    /// EFT gate documents). Uninitialized ⇒ `false` (fail-closed).
    pub(crate) fn f32_model_cached(&self) -> bool {
        if selfcheck_forced_to_fail() {
            return false;
        }
        self.f32_selfcheck.get().copied().unwrap_or(false)
    }

    /// Run (and log) the one-time probe. CONSERVATIVE: any mismatch OR GPU error → `false`.
    fn run_ieee_f32_selfcheck(&self) -> bool {
        match self.run_gpu_checked("verify_ieee_f32_model", || self.ieee_f32_selfcheck_inner()) {
            Ok(true) => true,
            Ok(false) => {
                tracing::warn!(
                    target: "ny_gpu::wgpu",
                    adapter = %self.adapter_info.name,
                    backend = ?self.adapter_info.backend,
                    "IEEE-754 f32-model diagnostic FAILED (a probe mismatched the CPU reference); \
                     refusing raw WGPU CROWN qualification"
                );
                false
            }
            Err(e) => {
                tracing::warn!(
                    target: "ny_gpu::wgpu",
                    adapter = %self.adapter_info.name,
                    backend = ?self.adapter_info.backend,
                    error = %e,
                    "IEEE-754 f32-model diagnostic could not run (GPU dispatch/readback error); \
                     refusing raw WGPU CROWN qualification"
                );
                false
            }
        }
    }

    /// Dispatch the probe shader and compare every returned bit pattern to the CPU
    /// reference. Returns `Ok(true)` only when ALL probes match their exact / within-model
    /// expectation; `Ok(false)` on any mismatch; `Err` on a GPU dispatch/readback fault.
    fn ieee_f32_selfcheck_inner(&self) -> Result<bool> {
        // ---- CPU reference (true IEEE-754 f32/f64), recomputed from INP ----
        let one = f32::from_bits(INP[0]);
        let ulp1 = f32::from_bits(INP[1]);
        let c12 = f32::from_bits(INP[2]);
        let cneg = f32::from_bits(INP[3]);

        let exp_add = (one + ulp1).to_bits(); // 0x3F800001
        let exp_prod = (c12 * c12).to_bits(); // 0x3F801000
        let mut acc = one;
        for _ in 0..ACC_ITERS {
            acc += ulp1;
        }
        let exp_acc = acc.to_bits(); // 0x3F800010
        let exp_round: Vec<u32> = (0..N_ROUND as usize)
            .map(|j| round_up_pos_ref(INP[ROUND_BASE as usize + j]))
            .collect();
        // FMA/`a*b+c` real (f64) reference for the within-model bound check.
        let fma_real = f64::from(c12) * f64::from(c12) + f64::from(cneg);

        // ---- Build buffers + pipeline (its OWN shader; shared enclosure shaders untouched) ----
        let params = F32SelfCheckParams {
            acc_iters: ACC_ITERS,
            n_round: N_ROUND,
            round_base: ROUND_BASE,
            out_round_base: OUT_ROUND_BASE,
        };
        let params_buf = create_buffer(
            &self.device,
            "f32_selfcheck_params",
            size_of::<F32SelfCheckParams>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        self.queue
            .write_buffer(&params_buf, 0, bytemuck::cast_slice(&[params]));

        let inp_buf = create_buffer(
            &self.device,
            "f32_selfcheck_inp",
            (INP.len() * size_of::<u32>()) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        self.queue
            .write_buffer(&inp_buf, 0, bytemuck::cast_slice(&INP));

        let out_bytes = (OUT_LEN * size_of::<u32>()) as u64;
        let out_buf = create_buffer(
            &self.device,
            "f32_selfcheck_out",
            out_bytes,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );

        // binding 1 = inp (read storage), binding 2 = outp (read_write storage).
        let (pipeline, layout) =
            self.create_simple_pipeline(IEEE_F32_SELFCHECK_SHADER, "f32_selfcheck", &[false, true]);

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("f32_selfcheck_bg"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: inp_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("f32_selfcheck_encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("f32_selfcheck_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        let staging = create_buffer(
            &self.device,
            "f32_selfcheck_staging",
            out_bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        encoder.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, out_bytes);
        self.queue.submit(std::iter::once(encoder.finish()));

        let out = WgpuDevice::read_u32_buffer(&self.device, &staging, OUT_LEN)?;
        if out.len() != OUT_LEN {
            return Ok(false);
        }

        // ---- Compare (fail-closed on ANY mismatch) ----
        // Probe 1: reduced precision (bit-exact vs the CPU f32 result).
        if out[0] != exp_add || out[1] != exp_prod || out[2] != exp_acc {
            return Ok(false);
        }
        // Probe 2: directed rounding (bit-exact vs the +1-ULP CPU reference).
        for (j, &expected) in exp_round.iter().enumerate() {
            if out[OUT_ROUND_BASE as usize + j] != expected {
                return Ok(false);
            }
        }
        // Probe 3: FMA / a*b + c within a generous within-model bound of the true value.
        let d = f32::from_bits(out[OUT_ROUND_BASE as usize + N_ROUND as usize]);
        if !d.is_finite() || (f64::from(d) - fma_real).abs() > FMA_TOL {
            return Ok(false);
        }

        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The CPU references match the hand-derived exact bit patterns documented in the
    /// module (a regression tripwire if `INP` or the arithmetic is ever perturbed).
    #[test]
    fn cpu_reference_bit_patterns_are_exact() {
        let one = f32::from_bits(INP[0]);
        let ulp1 = f32::from_bits(INP[1]);
        let c12 = f32::from_bits(INP[2]);
        assert_eq!(
            (one + ulp1).to_bits(),
            0x3F80_0001,
            "1.0 + 2^-23 is next-up(1.0)"
        );
        assert_eq!(
            (c12 * c12).to_bits(),
            0x3F80_1000,
            "(1+2^-12)^2 rounds to 1+2^-11"
        );
        let mut acc = one;
        for _ in 0..ACC_ITERS {
            acc += ulp1;
        }
        assert_eq!(acc.to_bits(), 0x3F80_0010, "1.0 + 16*2^-23");
        // Directed-rounding references.
        assert_eq!(round_up_pos_ref(0x3F80_0000), 0x3F80_0001); // 1.0 -> next up
        assert_eq!(round_up_pos_ref(0x0000_0001), 0x0080_0000); // subnormal -> normal floor
        assert_eq!(round_up_pos_ref(0x0000_0000), 0x0000_0000); // +0.0 -> 0
        assert_eq!(round_up_pos_ref(0xBF80_0000), 0x0000_0000); // -1.0 -> 0
    }

    /// `OUT_LEN` matches the shader's write layout (3 scalars + N_ROUND + 1 FMA).
    #[test]
    fn out_len_matches_layout() {
        assert_eq!(OUT_LEN, 11);
        assert_eq!(OUT_ROUND_BASE as usize + N_ROUND as usize, OUT_LEN - 1);
        assert_eq!(ROUND_BASE as usize + N_ROUND as usize, INP.len());
    }
}

/// GPU-backed self-check tests (require a real adapter).
#[cfg(all(test, feature = "gpu-tests"))]
mod gpu_tests {
    use super::*;
    use crate::wgpu_device::test_support::{gpu_test_serial_guard, require_device};
    use ny_core::{GemmEngine, GpuCrownBackward, GpuDagIbpForwardExt, GpuIbpForward};

    /// Clears the force-fail override on drop so a panicking assertion cannot leak the
    /// override into another test.
    struct ClearForceOnDrop;
    impl Drop for ClearForceOnDrop {
        fn drop(&mut self) {
            set_force_f32_selfcheck_fail(false);
        }
    }

    /// The hardware f32 probe may pass, but without an explicit request and the
    /// complete ladder the default raw route remains unavailable.
    #[test]
    fn passing_selfcheck_alone_does_not_bypass_full_qualification() {
        let _serial = gpu_test_serial_guard();
        let device = require_device();

        assert!(
            device.verify_ieee_f32_model(),
            "the Metal adapter must PASS the IEEE-754 f32-model self-check"
        );
        assert!(!device.provides_sound_gpu_crown());
        assert!(!device.provides_sound_gpu_ibp());
        assert!(!device.provides_sound_gpu_dag_ibp());

        // The gate's exact filter still exposes the sound GPU CROWN backward.
        // (Kept as the route's literal `.filter(...)` chain, bound to a local so the
        // predicate mirrors `gpu_crown_backward_route` byte-for-byte.)
        let engine: &dyn GemmEngine = &*device;
        let routed = engine
            .as_gpu_crown_backward()
            .filter(|g| g.provides_sound_gpu_crown());
        assert!(
            routed.is_none(),
            "one passing rung must not bypass the full WGPU qualification"
        );
    }

    /// FAIL-SAFE: with the self-check forced to fail, all three `provides_sound_gpu_*`
    /// predicates report `false`, and the gate's own filter
    /// (`as_gpu_crown_backward().filter(provides_sound_gpu_crown)`) yields `None` — so a
    /// bad adapter CANNOT decide a verdict (routing takes the CPU-sound fallback). After
    /// releasing the override the real (passing) result is restored.
    #[test]
    fn forced_failure_disables_sound_gpu_path() {
        let _serial = gpu_test_serial_guard();
        let device = require_device();
        let _clear = ClearForceOnDrop;

        set_force_f32_selfcheck_fail(true);
        assert!(!device.verify_ieee_f32_model());
        assert!(!device.provides_sound_gpu_crown());
        assert!(!device.provides_sound_gpu_ibp());
        assert!(!device.provides_sound_gpu_dag_ibp());

        // This is byte-for-byte the predicate `gpu_crown_backward_route` applies: a
        // failed self-check → None → CPU sound fallback (no GPU verdict).
        let engine: &dyn GemmEngine = &*device;
        let routed = engine
            .as_gpu_crown_backward()
            .filter(|g| g.provides_sound_gpu_crown());
        assert!(
            routed.is_none(),
            "forced self-check failure MUST mask the GPU CROWN backward from the verdict route"
        );

        set_force_f32_selfcheck_fail(false);
        assert!(
            device.verify_ieee_f32_model(),
            "releasing the override restores the real (passing) Metal result"
        );
        assert!(!device.provides_sound_gpu_crown());
    }
}
