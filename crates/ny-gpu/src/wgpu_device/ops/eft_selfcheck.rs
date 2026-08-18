// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Per-adapter self-check that AUTHORIZES the EFT compensated certified-error
//! channel (`docs/EFT_COMPENSATED_CERTIFIED_ERROR_DESIGN.md`).
//!
//! # Why a SEPARATE check from `f32_selfcheck`
//!
//! `verify_ieee_f32_model` authorizes the whole sound-GPU verdict path and must
//! stay untouched. The EFT channel is an OPTIONAL TIGHTENING on top of it: its
//! failure must only refuse the tightening (falling back to the a-priori
//! Higham charge byte-identically), never the sound path itself. So the gate
//! composes:
//!
//! ```text
//! verify_eft_primitives() = verify_ieee_f32_model()      // base f32 model
//!                        && verify_gradual_underflow()   // #u2b rung-3 policy
//!                        && <EFT probes>                 // this module
//! ```
//!
//! # `#u2b` — why the rung-3 subnormal policy is composed here
//!
//! The normal-range probes in THIS module establish that `fma`-TwoProduct and
//! the fma-barrier TwoSum reproduce the CPU reference bit-for-bit on those
//! lanes. They do NOT establish the
//! other half of the compensated channel's precondition, because **every probe
//! operand and every probe result in `PROD_PAIRS`/`SUM_PAIRS` is normal-range**
//! (the smallest magnitude anywhere in the table is `2^-100`). On an adapter
//! with nonconforming subnormal handling, these probes therefore PASS while the
//! un-floored channel's residual identity
//!
//! ```text
//! Σ aᵢ·wᵢ − accₙ = Σ epᵢ + Σ esᵢ
//! ```
//!
//! can be FALSE when a subnormal `ep`/`es` is silently replaced by `0`. The
//! current rung admits an exact-zero, subnormal-expected FMA result/residual
//! whose loss is covered by the always-on `rung3_flush_safe_additive` base,
//! scaled outward by the same downstream `r_slack`. A product-residual FMA may
//! also publish a larger finite nonnegative charge; that is conservative because
//! production accumulates its magnitude. Core add/multiply flushes and every
//! residual under-charge still refuse. Because the min-combine publishes
//! the SMALLER arm, this composed policy and floor are mandatory; the normal-
//! range primitive probe alone cannot authorize the channel.
//!
//! Before this change the two rungs were merely SIBLINGS on the authority
//! ladder, so `verify_eft_primitives()` could — and on Apple M5 Max/Metal
//! actually did — return `true` on an adapter measured to flush (rung2 true,
//! rung3 false). The verdict ladder as a whole still refused, but the two
//! PRODUCTION consumers of the compensated channel
//! (`crown_backward_sound_resident.rs`, `crown_concretize_sound.rs`) gate on
//! `eft_primitives_cached()` **alone**, under `NY_EFT_ERR=1`, with no reference
//! to the underflow rung. On this box that combination measurably authorized
//! and FIRED the compensated tightening (72/72 specs tightened) on hardware
//! whose flush behaviour breaks the identity the tightening rests on.
//!
//! The fix is ENTAILMENT, not adjacency: the EFT authorization now *contains*
//! rung-3 subnormal authorization, so no caller can obtain one without the
//! other, and the ladder can never again disagree in the unsafe direction. The
//! same composition is applied to [`WgpuDevice::eft_primitives_cached`] — the
//! read the production sites actually use — because a precondition enforced
//! only on the live path is not enforced at all.
//!
//! # What the probes verify (measured 2026-07-23 on the GB10, probe-pinned in
//! `ops/double_single_probe.rs`)
//!
//! 1. **fma TwoProduct (normal-range primitive lanes)**:
//!    `e = fma(a, b, −fl(a·b))` is the exact product residual when the measured
//!    FMA honors its operands. It is bit-compared against the CPU
//!    `ny_core::eft::two_prod_f32` (itself pinned by exact-rational oracles).
//!    The composed rung-3 probe separately covers subnormal multiplicands in
//!    both FMA slots and requires the published magnitude to enclose the exact
//!    residual.
//! 2. **fma-barrier TwoSum**: the Knuth residual with every subtraction routed
//!    through the `fma` intrinsic — the ONLY TwoSum form that survives the
//!    shader compiler (the plain-adds form is folded to 0; the Dekker split is
//!    broken by FMA contraction). Bit-compared against the CPU
//!    `ny_core::eft::two_sum_f32`.
//!
//! Every operand is read from a storage buffer at runtime so the compiler
//! cannot constant-fold the probes. ANY mismatch, dispatch fault, or readback
//! error ⇒ `false` (fail-closed): the EFT channel is refused and the Higham
//! channel ships unchanged. A driver/naga update that starts canonicalizing
//! through `fma` intrinsics is caught here (and by the re-pinned tripwires in
//! `double_single_probe.rs`).

use ny_core::Result;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

use super::super::WgpuDevice;
use super::ibp_forward::create_buffer;

/// Probe operand pairs `(a, b)` as f32 bit patterns. Diverse magnitudes +
/// cancellation; the CPU expectations are recomputed from these SAME bits via
/// `ny_core::eft`, so there is one source of truth for the operands.
const PROD_PAIRS: [(u32, u32); 4] = [
    (0x3F80_0800, 0x3F80_0800), // (1+2^-12)^2 → residual exactly 2^-24
    (0x4040_0000, 0x3EAA_AAAB), // 3.0 · fl(1/3)
    (0x5015_02F9, 0x2EDB_E6FF), // ~1e10 · ~1e-10
    (0xC0E0_0000, 0x3E12_4926), // -7.0 · ~1/7
];
const SUM_PAIRS: [(u32, u32); 4] = [
    (0x3F80_0000, 0x0D80_0000), // 1.0 + 2^-100 → residual is the tiny term
    (0x4B80_0000, 0x3F80_0000), // 2^24 + 1.0 → RN-even keeps 2^24, residual 1.0
    (0x3F80_0800, 0xBF80_1000), // (1+2^-12) + −(1+2^-11): cancellation
    (0x7148_1CC8, 0xF148_1CC8), // large x + (−x) → exact 0, residual 0
];
/// Output layout: `n_prod` (p, e) pairs then `n_sum` (s, t) pairs, u32 bits.
const OUT_LEN: usize = (PROD_PAIRS.len() + SUM_PAIRS.len()) * 2;

/// 16-byte std140-clean uniform params. Layout MUST match WGSL `struct Params`.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct EftSelfCheckParams {
    n_prod: u32,
    n_sum: u32,
    _pad0: u32,
    _pad1: u32,
}

/// WGSL probe. Its OWN shader; the shared enclosure shaders are untouched. The
/// `two_sum_fma_barrier` body must stay byte-identical to the form the shipped
/// EFT kernels use (and that `double_single_probe.rs` pins bit-exact).
const EFT_SELFCHECK_SHADER: &str = r#"
struct Params { n_prod: u32, n_sum: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read>       inp:  array<u32>; // (a,b) bit-pattern pairs: prods then sums
@group(0) @binding(2) var<storage, read_write>  outp: array<u32>;

// FMA-based TwoProduct: err = fma(a, b, -p), exact iff fma is truly fused.
fn two_prod_fma(a: f32, b: f32) -> vec2<f32> {
    let p = a * b;
    let err = fma(a, b, -p);
    return vec2<f32>(p, err);
}

// fma-barrier TwoSum: every subtraction of the Knuth sequence routed through
// the fma intrinsic so the compiler cannot fold the algebraically-zero
// compensation term (the plain-adds form IS folded — measured).
fn two_sum_fma_barrier(a: f32, b: f32) -> vec2<f32> {
    let s = a + b;
    let bb = fma(-1.0, a, s);   // s - a
    let sb = fma(-1.0, bb, s);  // s - bb
    let da = fma(-1.0, sb, a);  // a - (s - bb)
    let db = fma(-1.0, bb, b);  // b - bb
    return vec2<f32>(s, da + db);
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u) { return; }
    for (var i: u32 = 0u; i < params.n_prod; i = i + 1u) {
        let a = bitcast<f32>(inp[2u * i]);
        let b = bitcast<f32>(inp[2u * i + 1u]);
        let r = two_prod_fma(a, b);
        outp[2u * i] = bitcast<u32>(r.x);
        outp[2u * i + 1u] = bitcast<u32>(r.y);
    }
    let sum_in_base = 2u * params.n_prod;
    for (var j: u32 = 0u; j < params.n_sum; j = j + 1u) {
        let a = bitcast<f32>(inp[sum_in_base + 2u * j]);
        let b = bitcast<f32>(inp[sum_in_base + 2u * j + 1u]);
        let r = two_sum_fma_barrier(a, b);
        outp[sum_in_base + 2u * j] = bitcast<u32>(r.x);
        outp[sum_in_base + 2u * j + 1u] = bitcast<u32>(r.y);
    }
}
"#;

/// Test/operator hook: force [`WgpuDevice::verify_eft_primitives`] to report
/// FAILURE (mirrors the `f32_selfcheck` hook). Read on every call — NOT cached —
/// so a test can flip it without the cached real result masking it.
static TEST_FORCE_FAIL: AtomicBool = AtomicBool::new(false);

fn env_forces_fail() -> bool {
    static ENV: OnceLock<bool> = OnceLock::new();
    *ENV.get_or_init(|| std::env::var_os("NY_FORCE_GPU_EFT_SELFCHECK_FAIL").is_some())
}

fn selfcheck_forced_to_fail() -> bool {
    TEST_FORCE_FAIL.load(Ordering::Relaxed) || env_forces_fail()
}

/// Never-opening visibility for higher-level authority predicates. The hook
/// can only revoke a previously measured prerequisite.
pub(crate) fn probe_forced_to_fail() -> bool {
    selfcheck_forced_to_fail()
}

/// Complete a probe authorization only if the attested shader-loading contract
/// is still intact after the potentially lazy probe-module creation.
const fn post_probe_authorization(probe_passed: bool, loading_contract_intact: bool) -> bool {
    probe_passed && loading_contract_intact
}

/// Test hook: force / release an EFT self-check failure to exercise the
/// Higham-fallback route without a non-conformant adapter.
#[cfg(all(test, feature = "gpu-tests"))]
pub(crate) fn set_force_eft_selfcheck_fail(force: bool) {
    TEST_FORCE_FAIL.store(force, Ordering::Relaxed);
}

impl WgpuDevice {
    /// One-time per-adapter authorization for the EFT compensated
    /// certified-error channel: `true` ⇒ the adapter passes the full IEEE f32
    /// model check, passes the rung-3 subnormal policy (core add/multiply lanes
    /// bit-exact; primary products use mul, product-residual overcharges are
    /// admitted, and only exact-zero subnormal FMA results use the charged
    /// floor), AND executes both EFT primitives (fma TwoProduct, fma-barrier
    /// TwoSum) bit-exactly on the normal-range probe table. Together
    /// those checks make EFT-measured residuals plus the floor valid certified
    /// error bounds on this device.
    /// `false` ⇒ the EFT channel is REFUSED and callers keep the a-priori
    /// Higham charge byte-identically (fail-closed; the sound path itself is
    /// unaffected). Cached per device.
    ///
    /// This is an AUTHORIZATION, not a measurement. For the raw probe verdict
    /// in isolation (diagnostics only — it authorizes nothing) see
    /// [`Self::eft_primitives_raw_probe`].
    pub(crate) fn verify_eft_primitives(&self) -> bool {
        if selfcheck_forced_to_fail() {
            return false;
        }
        // PRECONDITION 1 — the base IEEE f32 model (RN adds at u=2^-24, no
        // covert reduced precision). Compose, never bypass.
        if !self.verify_ieee_f32_model() {
            return false;
        }
        // PRECONDITION 2 (#u2b) — the rung-3 subnormal policy. Checked BEFORE
        // the probes below because those probes are structurally incapable of
        // detecting a violation of it: their operands and results are all
        // normal-range, so they pass despite subnormal-lane defects. Rung 3
        // requires bit-exact core add/mul behavior and admits only the
        // exact-zero residual case paid by the production floor. Primary
        // products deliberately use mul rather than the measured FMA-DAZ form.
        // Ordering it first also
        // means a refusing adapter never dispatches the EFT probe at all — a
        // refusal costs one probe, not two.
        if !self.verify_gradual_underflow() {
            return false;
        }
        let probe_passed = self.eft_probe_passed();
        // Creating the probe module can itself discover that requested
        // DenormPreserve passthrough is unavailable and poison the process.
        // Recheck after the lazy creation; a pre-probe check alone attests a
        // different execution path from the module we just loaded.
        post_probe_authorization(probe_passed, self.denorm_preserve_contract_intact())
    }

    /// The RAW probe verdict — "did fma-TwoProduct and the fma-barrier TwoSum
    /// reproduce the CPU reference bit-for-bit" — with NO preconditions
    /// attached. This AUTHORIZES NOTHING; on an adapter that violates rung 3 it
    /// can be `true` while the compensated channel is unsound. Kept separate so
    /// a refusal stays diagnosable (which rung actually failed) instead of
    /// collapsing to a single bit.
    fn eft_probe_passed(&self) -> bool {
        *self.eft_selfcheck.get_or_init(|| self.run_eft_selfcheck())
    }

    /// Diagnostics-only accessor for [`Self::eft_probe_passed`]. Deliberately
    /// test-only: no production call site may branch on the raw probe, because
    /// doing so is precisely the `#u2b` hole this module now closes.
    #[cfg(all(test, feature = "gpu-tests"))]
    pub(crate) fn eft_primitives_raw_probe(&self) -> bool {
        self.eft_probe_passed()
    }

    /// Never-initializing cached read of the EFT gate, for call sites INSIDE a
    /// GPU-checked section (running the probe there would self-deadlock on the
    /// enclosing lock — see the eager init in `device.rs`). Uninitialized ⇒
    /// `false` (fail-closed: the Higham channel ships unchanged).
    ///
    /// Composes the SAME preconditions as [`Self::verify_eft_primitives`], via
    /// the never-initializing reads of each. This is the gate the two
    /// production consumers of the compensated channel actually call, so a
    /// precondition that were enforced only on the live path would not be
    /// enforced at all. Both caches are primed at device construction (see the
    /// eager init in `device.rs`), so the added conjuncts cost no reachable
    /// authorization; an unprimed cache reads `false`, which is the correct
    /// fail-closed answer.
    pub(crate) fn eft_primitives_cached(&self) -> bool {
        if selfcheck_forced_to_fail() {
            return false;
        }
        if !self.f32_model_cached() || !self.gradual_underflow_cached() {
            return false;
        }
        post_probe_authorization(
            self.eft_selfcheck.get().copied().unwrap_or(false),
            self.denorm_preserve_contract_intact(),
        )
    }

    /// Run (and log) the one-time probe. CONSERVATIVE: any mismatch OR GPU
    /// error → `false`.
    fn run_eft_selfcheck(&self) -> bool {
        match self.run_gpu_checked("verify_eft_primitives", || self.eft_selfcheck_inner()) {
            Ok(true) => true,
            Ok(false) => {
                tracing::warn!(
                    target: "ny_gpu::wgpu",
                    adapter = %self.adapter_info.name,
                    backend = ?self.adapter_info.backend,
                    "EFT-primitive self-check FAILED (fma TwoProduct or fma-barrier TwoSum \
                     mismatched the CPU reference): REFUSING the EFT compensated error \
                     channel on this adapter; the a-priori Higham charge ships unchanged \
                     (fail-closed, never a wrong verdict)"
                );
                false
            }
            Err(e) => {
                tracing::warn!(
                    target: "ny_gpu::wgpu",
                    adapter = %self.adapter_info.name,
                    backend = ?self.adapter_info.backend,
                    error = %e,
                    "EFT-primitive self-check could not run: REFUSING the EFT channel \
                     (fail-closed to the Higham charge)"
                );
                false
            }
        }
    }

    /// Dispatch the probe shader and bit-compare every (value, residual) pair
    /// against the CPU `ny_core::eft` reference.
    fn eft_selfcheck_inner(&self) -> Result<bool> {
        // The CPU reference must itself be sound on this host (rational-oracle
        // pinned; checks fused mul_add, RN, no FTZ). If the HOST is broken we
        // cannot trust the comparison — refuse.
        if ny_core::eft::eft_self_check().is_err() {
            return Ok(false);
        }

        // ---- CPU expectations, recomputed from the same operand bits ----
        let mut expected: Vec<u32> = Vec::with_capacity(OUT_LEN);
        for &(a, b) in &PROD_PAIRS {
            let (p, e) = ny_core::eft::two_prod_f32(f32::from_bits(a), f32::from_bits(b));
            expected.push(p.to_bits());
            expected.push(e.to_bits());
        }
        for &(a, b) in &SUM_PAIRS {
            let (s, t) = ny_core::eft::two_sum_f32(f32::from_bits(a), f32::from_bits(b));
            expected.push(s.to_bits());
            expected.push(t.to_bits());
        }

        // ---- Buffers + pipeline (its OWN shader) ----
        let params = EftSelfCheckParams {
            n_prod: PROD_PAIRS.len() as u32,
            n_sum: SUM_PAIRS.len() as u32,
            _pad0: 0,
            _pad1: 0,
        };
        let params_buf = create_buffer(
            &self.device,
            "eft_selfcheck_params",
            size_of::<EftSelfCheckParams>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        self.queue
            .write_buffer(&params_buf, 0, bytemuck::cast_slice(&[params]));

        let inp: Vec<u32> = PROD_PAIRS
            .iter()
            .chain(SUM_PAIRS.iter())
            .flat_map(|&(a, b)| [a, b])
            .collect();
        let inp_buf = create_buffer(
            &self.device,
            "eft_selfcheck_inp",
            (inp.len() * size_of::<u32>()) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        self.queue
            .write_buffer(&inp_buf, 0, bytemuck::cast_slice(&inp));

        let out_bytes = (OUT_LEN * size_of::<u32>()) as u64;
        let out_buf = create_buffer(
            &self.device,
            "eft_selfcheck_out",
            out_bytes,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );

        let (pipeline, layout) =
            self.create_simple_pipeline(EFT_SELFCHECK_SHADER, "eft_selfcheck", &[false, true]);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("eft_selfcheck_bg"),
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
                label: Some("eft_selfcheck_encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("eft_selfcheck_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        let staging = create_buffer(
            &self.device,
            "eft_selfcheck_staging",
            out_bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        encoder.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, out_bytes);
        self.queue.submit(std::iter::once(encoder.finish()));

        let out = WgpuDevice::read_u32_buffer(&self.device, &staging, OUT_LEN)?;
        if out.len() != OUT_LEN {
            return Ok(false);
        }
        // Bit-exact on EVERY value and EVERY residual — fail-closed on any drift.
        Ok(out.iter().zip(expected.iter()).all(|(got, exp)| got == exp))
    }
}

#[cfg(test)]
mod cpu_tests {
    use super::*;

    #[test]
    fn a_late_shader_loading_fallback_invalidates_a_passing_probe() {
        assert!(post_probe_authorization(true, true));
        assert!(
            !post_probe_authorization(true, false),
            "a passthrough fallback discovered while creating the probe must close authorization"
        );
        assert!(!post_probe_authorization(false, true));
    }

    /// The probe operands must be adversarial enough to DISCRIMINATE: every sum
    /// pair's true residual must be nonzero-or-boundary such that a folded
    /// (always-0) TwoSum fails the bit-compare on at least one lane, and the
    /// canonical product pair must have a nonzero residual.
    #[test]
    fn probe_operands_discriminate() {
        // Canonical product residual is exactly 2^-24 (nonzero).
        let (_, e) = ny_core::eft::two_prod_f32(
            f32::from_bits(PROD_PAIRS[0].0),
            f32::from_bits(PROD_PAIRS[0].1),
        );
        assert_eq!(e.to_bits(), 0x3380_0000, "2^-24");

        // At least one sum lane must have a NONZERO residual (else a broken,
        // always-zero TwoSum would pass the whole probe).
        let nonzero = SUM_PAIRS.iter().any(|&(a, b)| {
            let (_, t) = ny_core::eft::two_sum_f32(f32::from_bits(a), f32::from_bits(b));
            t != 0.0
        });
        assert!(nonzero, "sum probes must include a nonzero residual lane");
    }

    /// `#u2b` — the STRUCTURAL justification for composing
    /// `verify_gradual_underflow()` into `verify_eft_primitives()`.
    ///
    /// This table is BLIND to flush-to-zero: every operand and every emitted
    /// value/residual is normal-range or exactly zero, so a DAZ/FTZ adapter
    /// reproduces all 16 lanes bit-for-bit and passes. That is a property of
    /// the probe, not an accident of the hardware, and it is why the underflow
    /// rung must ENTAIL rather than merely accompany this one.
    ///
    /// If a future edit adds a subnormal lane here this test fails — and the
    /// correct response is still to keep the composition (a probe that
    /// sometimes catches flushing is not a proof that it always does); just
    /// re-pin the claim to the lanes that remain normal-range.
    #[test]
    fn probe_table_is_blind_to_flush_so_the_underflow_rung_must_entail_it() {
        let subnormal = |x: f32| x != 0.0 && x.abs() < f32::MIN_POSITIVE;
        let mut lanes = 0usize;

        for &(a, b) in &PROD_PAIRS {
            let (fa, fb) = (f32::from_bits(a), f32::from_bits(b));
            let (p, e) = ny_core::eft::two_prod_f32(fa, fb);
            for (what, v) in [
                ("operand a", fa),
                ("operand b", fb),
                ("value", p),
                ("residual", e),
            ] {
                assert!(
                    !subnormal(v),
                    "PROD_PAIRS ({a:#010x},{b:#010x}) {what} = {v:e} is SUBNORMAL: this \
                     probe would then partially detect flushing, invalidating the pinned \
                     claim below"
                );
            }
            lanes += 2;
        }
        for &(a, b) in &SUM_PAIRS {
            let (fa, fb) = (f32::from_bits(a), f32::from_bits(b));
            let (s, t) = ny_core::eft::two_sum_f32(fa, fb);
            for (what, v) in [
                ("operand a", fa),
                ("operand b", fb),
                ("value", s),
                ("residual", t),
            ] {
                assert!(
                    !subnormal(v),
                    "SUM_PAIRS ({a:#010x},{b:#010x}) {what} = {v:e} is SUBNORMAL: see above"
                );
            }
            lanes += 2;
        }
        assert_eq!(lanes, OUT_LEN, "every emitted lane must be covered");

        // The smallest magnitude anywhere in the table, pinned: 2^-100, which is
        // 26 binades above the smallest NORMAL (2^-126). Nothing here can round
        // into the subnormal band, so nothing here can observe FTZ.
        let smallest = PROD_PAIRS
            .iter()
            .chain(SUM_PAIRS.iter())
            .flat_map(|&(a, b)| [f32::from_bits(a), f32::from_bits(b)])
            .map(f32::abs)
            .filter(|&x| x != 0.0)
            .fold(f32::INFINITY, f32::min);
        assert_eq!(
            smallest.to_bits(),
            0x0D80_0000,
            "smallest nonzero probe operand must still be 2^-100 (26 binades above \
             the subnormal band)"
        );
    }
}

#[cfg(all(test, feature = "gpu-tests"))]
mod gpu_tests {
    use super::*;
    use crate::wgpu_device::test_support::{gpu_test_serial_guard, require_device};

    /// MEASUREMENT + ENTAILMENT (was: `adapter_passes_and_forced_fail_refuses`).
    ///
    /// The outcome assertion was REMOVED, not weakened. Whether the raw probes
    /// pass is a hardware fact; whether the CHANNEL is authorized additionally
    /// depends on the rung-3 subnormal policy, which is also a hardware fact.
    /// Asserting either outcome would make the suite adapter-specific instead
    /// of exercising the correct authorization/refusal. What IS asserted is the
    /// part that must hold on ANY adapter: the authorization implies every
    /// precondition, and both fail-closed hooks refuse.
    #[test]
    fn eft_authorization_entails_its_preconditions_and_fails_closed() {
        let _serial = gpu_test_serial_guard();
        let device = require_device();

        let raw = device.eft_primitives_raw_probe();
        let f32_model = device.verify_ieee_f32_model();
        let underflow = device.verify_gradual_underflow();
        let authorized = device.verify_eft_primitives();
        println!(
            "[eft gate] adapter={} backend={:?}\n  \
             raw EFT probe (fma TwoProduct + fma-barrier TwoSum) = {raw}\n  \
             precondition verify_ieee_f32_model                  = {f32_model}\n  \
             precondition verify_gradual_underflow (#u2b)        = {underflow}\n  \
             => verify_eft_primitives()                          = {authorized}",
            device.adapter_info.name, device.adapter_info.backend,
        );

        // THE #u2b INVARIANT: authorization implies EVERY precondition. In
        // particular it can never outrun the underflow rung.
        if authorized {
            assert!(f32_model, "EFT authorized while the f32 model rung FAILED");
            assert!(
                underflow,
                "EFT authorized while the rung-3 subnormal policy FAILED — this \
                 is exactly the #u2b hole: the compensated channel would be \
                 authorized without the core-lane checks and the precisely \
                 floor-covered residual exception"
            );
            assert!(raw, "EFT authorized while the raw probe FAILED");
        }
        // ...and the converse direction, stated so a vacuous implication cannot
        // pass: with all three preconditions holding, authorization must FOLLOW
        // (no unaccounted extra refusal).
        assert_eq!(
            authorized,
            f32_model && underflow && raw,
            "verify_eft_primitives() must be exactly the conjunction of its rungs"
        );

        // Fail-closed hook on THIS rung.
        set_force_eft_selfcheck_fail(true);
        assert!(
            !device.verify_eft_primitives(),
            "forced EFT-probe failure must refuse the EFT channel"
        );
        assert!(
            !device.eft_primitives_cached(),
            "forced EFT-probe failure must also refuse the CACHED read (the gate \
             the production sites actually call)"
        );
        set_force_eft_selfcheck_fail(false);
        assert_eq!(
            device.verify_eft_primitives(),
            authorized,
            "cache must survive the hook"
        );

        // #u2b, the load-bearing new behaviour: forcing the SUBNORMAL rung
        // closed must close the EFT channel too — on BOTH the live and the
        // cached read. Before this change neither responded to it at all.
        super::super::subnormal_selfcheck::set_force_subnormal_selfcheck_fail(true);
        assert!(
            !device.verify_eft_primitives(),
            "a forced gradual-underflow failure must refuse the EFT channel"
        );
        assert!(
            !device.eft_primitives_cached(),
            "a forced gradual-underflow failure must refuse the CACHED EFT read"
        );
        super::super::subnormal_selfcheck::set_force_subnormal_selfcheck_fail(false);
        assert_eq!(
            device.verify_eft_primitives(),
            authorized,
            "releasing the subnormal hook must restore the measured verdict"
        );
    }
}
