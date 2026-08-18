// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! One-time known-answer IEEE bit-exactness probe that authorizes engine
//! construction (#cuda-ieee-selfcheck).
//!
//! # Why this exists (soundness)
//!
//! Every sound seam built on this engine — the f64 `A·W` GEMM offload, the f32
//! abs-sum `S` seam, the f64-resident CROWN backward — certifies its rounding
//! error with Higham `γ_n` terms that assume plain IEEE round-to-nearest
//! arithmetic at unit roundoff `2^-24` (f32) / `2^-53` (f64); reduction-ORDER
//! independence is the only liberty the certificates grant. cuBLAS can silently
//! substitute reduced precision (TF32 tensor cores, BF16x9 SGEMM emulation,
//! Ozaki fixed-point DGEMM emulation on Blackwell). The env blocklist
//! (`CUBLAS_EMULATION_ENV_VARS`) and the `CUBLAS_DEFAULT_MATH` pin + read-back
//! only assert what cuBLAS was ASKED to do; `docs/F32_ABSSUM_SEAM.md` §5
//! mandates verifying what it DID: "run a small GEMM with a known exact f32
//! result and assert bit-accuracy; refuse the engine (fall back to f64 `S`) if
//! it fails. If TF32 ever leaks in, every `F` above is unsound."
//!
//! # The probes (port of `ny-gpu`'s wgpu `f32_selfcheck` operand patterns)
//!
//! Each probe is a GEMM whose products AND every partial sum (any subset of the
//! `k` products) are exactly representable in the target type, so a conformant
//! IEEE engine returns ONE bit pattern regardless of blocking, split-K reduction
//! order, FMA contraction, or compensated summation — bit-exactness can be
//! demanded without assuming anything cuBLAS does not guarantee. The operands
//! carry full-width mantissas that reduced-precision paths cannot represent:
//!
//! * accumulation (`32×128×32`): each row of `A` is `[1 + 2^-23, 2^-23 × 127]`,
//!   `B` all-ones ⇒ every `C` entry is exactly `1 + 128·2^-23`. TF32 (10-bit
//!   mantissa) or BF16 (7-bit) rounds `1 + 2^-23` to `1.0` at operand load,
//!   dropping the lead ULP: `C = 1 + 127·2^-23`, bit-different.
//! * product (`1×1×1`): `(1 + 2^-11)·(1 + 2^-12) = 1 + 2^-11 + 2^-12 + 2^-23`,
//!   exact in f32 (fraction bits 11, 12, 23); under TF32 both operands round to
//!   `1.0`, and any emulation dropping low×low cross terms loses the `2^-23`.
//! * the f64 (Dgemm) twins at `2^-52` / `2^-26` scale catch fixed-point DGEMM
//!   emulation with any effective mantissa below 52 bits.
//!
//! The probes run on the CONSTRUCTED engine's real `gemm_f32`/`gemm_f64`
//! dispatch (ATS zero-copy, unified buffer, or explicitly selected cached
//! device-copy transport, whichever this engine will use).
//! `gemm_f32_fast` is deliberately NOT probed: reduced precision is its
//! documented contract (attack-only traffic; it can never decide a verdict).
//! On ANY bit deviation (a NaN never bit-matches) construction fails closed:
//! [`CudaGemmEngine::with_ordinal`] refuses the engine and callers fall back to
//! the proven-sound CPU f64 path.

use ny_core::{GemmEngine, NyError, Result};

use crate::CudaGemmEngine;

/// Accumulation-probe GEMM shape. Multiples of the 16-wide tensor-core tiles
/// with a `k` deep enough for split-K blocking, so a TF32-defaulting cuBLAS
/// would route THIS shape to the same kernel class as the real seam shapes
/// (k≈64–512); a `1×k×1` probe could dodge them via a GEMV-style special case.
const PROBE_M: usize = 32;
const PROBE_K: usize = 128;
const PROBE_N: usize = 32;

/// `1 + 2^-11` and `1 + 2^-12` (f32 product-probe operands) as bit patterns,
/// wgpu-`INP` style, so the tripwire test can pin the hand-derived expectations.
const PROD_A_F32: u32 = 0x3F80_1000;
const PROD_B_F32: u32 = 0x3F80_0800;
/// `1 + 2^-26` (f64): its square `1 + 2^-25 + 2^-52` is exact in f64 and its
/// lowest bit vanishes under any sub-52-bit emulated mantissa.
const PROD_A_F64: u64 = 0x3FF0_0000_0400_0000;

/// Exact expected value of the f32 accumulation probe, computed on the host
/// with the SAME f32 arithmetic. Every step is exact (`1 + m·2^-23` for
/// `m ≤ 128` spans at most the 24 significand bits f32 has), so this equals
/// the true real value and is a valid expectation for ANY device reduction
/// order. `f32::EPSILON` is exactly `2^-23`, the ULP of 1.0.
fn acc_expected_f32() -> f32 {
    let mut acc = 1.0f32 + f32::EPSILON;
    for _ in 1..PROBE_K {
        acc += f32::EPSILON;
    }
    acc
}

/// f64 twin of [`acc_expected_f32`] (`1 + m·2^-52`, `m ≤ 128`, ≤ 53 bits).
fn acc_expected_f64() -> f64 {
    let mut acc = 1.0f64 + f64::EPSILON;
    for _ in 1..PROBE_K {
        acc += f64::EPSILON;
    }
    acc
}

/// Bit-compare a probe result against its exact expectation; ANY deviation
/// (including NaN and a wrong length) fails closed with the first offending
/// element named. Pure, so the fail-closed comparison is unit-testable without
/// a device. `check_bits_f64` is its byte-for-byte f64 twin.
fn check_bits_f32(probe: &str, got: &[f32], want: &[f32]) -> Result<()> {
    if got.len() != want.len() {
        return Err(NyError::InternalError(format!(
            "cuda: IEEE {probe} probe returned {} elements, want {}; engine refused",
            got.len(),
            want.len()
        )));
    }
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        if g.to_bits() != w.to_bits() {
            return Err(NyError::InternalError(format!(
                "cuda: IEEE {probe} probe NOT bit-exact at [{i}]: got {g:e} ({:#010x}), \
                 want {w:e} ({:#010x}) — reduced-precision (TF32/BF16x9/fixed-point-emulated) \
                 GEMM suspected; engine refused",
                g.to_bits(),
                w.to_bits()
            )));
        }
    }
    Ok(())
}

/// See [`check_bits_f32`].
fn check_bits_f64(probe: &str, got: &[f64], want: &[f64]) -> Result<()> {
    if got.len() != want.len() {
        return Err(NyError::InternalError(format!(
            "cuda: IEEE {probe} probe returned {} elements, want {}; engine refused",
            got.len(),
            want.len()
        )));
    }
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        if g.to_bits() != w.to_bits() {
            return Err(NyError::InternalError(format!(
                "cuda: IEEE {probe} probe NOT bit-exact at [{i}]: got {g:e} ({:#018x}), \
                 want {w:e} ({:#018x}) — reduced-precision (TF32/BF16x9/fixed-point-emulated) \
                 GEMM suspected; engine refused",
                g.to_bits(),
                w.to_bits()
            )));
        }
    }
    Ok(())
}

impl CudaGemmEngine {
    /// Run the four known-answer probes (module doc) on this engine's real
    /// GEMM dispatch. `Err` ⇒ the constructor must refuse the engine.
    pub(crate) fn assert_ieee_bit_exact(&self) -> Result<()> {
        // f32 Sgemm accumulation: rows [1+2^-23, 2^-23 ×(k-1)] · all-ones.
        let mut a32 = vec![f32::EPSILON; PROBE_M * PROBE_K];
        for row in a32.as_chunks_mut::<PROBE_K>().0 {
            row[0] = 1.0 + f32::EPSILON;
        }
        let b32 = vec![1.0f32; PROBE_K * PROBE_N];
        let got = self.gemm_f32(PROBE_M, PROBE_K, PROBE_N, &a32, &b32)?;
        let want = vec![acc_expected_f32(); PROBE_M * PROBE_N];
        check_bits_f32("f32 Sgemm accumulation", &got, &want)?;

        // f32 Sgemm product: (1+2^-11)·(1+2^-12), a single exact product.
        let pa = f32::from_bits(PROD_A_F32);
        let pb = f32::from_bits(PROD_B_F32);
        let got = self.gemm_f32(1, 1, 1, &[pa], &[pb])?;
        check_bits_f32("f32 Sgemm product", &got, &[pa * pb])?;

        // f64 Dgemm accumulation: rows [1+2^-52, 2^-52 ×(k-1)] · all-ones.
        let mut a64 = vec![f64::EPSILON; PROBE_M * PROBE_K];
        for row in a64.as_chunks_mut::<PROBE_K>().0 {
            row[0] = 1.0 + f64::EPSILON;
        }
        let b64 = vec![1.0f64; PROBE_K * PROBE_N];
        let got = self.gemm_f64(PROBE_M, PROBE_K, PROBE_N, &a64, &b64)?;
        let want = vec![acc_expected_f64(); PROBE_M * PROBE_N];
        check_bits_f64("f64 Dgemm accumulation", &got, &want)?;

        // f64 Dgemm product: (1+2^-26)², a single exact product.
        let qa = f64::from_bits(PROD_A_F64);
        let got = self.gemm_f64(1, 1, 1, &[qa], &[qa])?;
        check_bits_f64("f64 Dgemm product", &got, &[qa * qa])?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The host-side expectations match the hand-derived exact bit patterns (a
    /// regression tripwire if a probe operand or shape is ever perturbed into
    /// inexact-partial-sum territory, which would break the any-reduction-order
    /// argument the bit-exactness demand rests on).
    #[test]
    fn probe_expectations_match_hand_derived_bits() {
        assert_eq!(acc_expected_f32().to_bits(), 0x3F80_0080, "1 + 128·2^-23");
        assert_eq!(
            acc_expected_f64().to_bits(),
            0x3FF0_0000_0000_0080,
            "1 + 128·2^-52"
        );
        let pa = f32::from_bits(PROD_A_F32);
        let pb = f32::from_bits(PROD_B_F32);
        assert_eq!(
            (pa * pb).to_bits(),
            0x3F80_1801,
            "1 + 2^-11 + 2^-12 + 2^-23"
        );
        let qa = f64::from_bits(PROD_A_F64);
        assert_eq!(
            (qa * qa).to_bits(),
            0x3FF0_0000_0800_0001,
            "1 + 2^-25 + 2^-52"
        );
    }

    /// The fail-closed comparison itself: exact vectors pass; a 1-ULP
    /// deviation, a NaN, or a length mismatch each refuse.
    #[test]
    fn check_bits_fails_closed_on_any_deviation() {
        let want = [1.0f32, 1.5, -2.0];
        assert!(check_bits_f32("t", &want, &want).is_ok());
        let ulp = [1.0f32, f32::from_bits(1.5f32.to_bits() + 1), -2.0];
        assert!(check_bits_f32("t", &ulp, &want).is_err());
        let nan = [1.0f32, f32::NAN, -2.0];
        assert!(check_bits_f32("t", &nan, &want).is_err());
        assert!(check_bits_f32("t", &want[..2], &want).is_err());

        let want64 = [1.0f64, -0.5];
        assert!(check_bits_f64("t", &want64, &want64).is_ok());
        let ulp64 = [1.0f64, f64::from_bits((-0.5f64).to_bits() + 1)];
        assert!(check_bits_f64("t", &ulp64, &want64).is_err());
        assert!(check_bits_f64("t", &[f64::NAN, -0.5], &want64).is_err());
        assert!(check_bits_f64("t", &want64[..1], &want64).is_err());
    }

    /// ON-DEVICE: this box's cuBLAS `CUBLAS_DEFAULT_MATH` Sgemm/Dgemm must be
    /// bit-exact IEEE — the physical GB10 verification `docs/F32_ABSSUM_SEAM.md`
    /// §5 asks for. Construction itself runs the probes, so a reduced-precision
    /// device cannot construct an engine at all. On hardware-free CI the shared
    /// capability seam proves why no device dispatch is admitted; on a host
    /// advertising CUDA, construction or probe failure fails this test.
    #[test]
    fn cuda_ieee_known_answer_probes_are_exact_when_hardware_is_capable() {
        crate::with_capable_cuda(|engine| {
            engine
                .assert_ieee_bit_exact()
                .expect("IEEE known-answer probes must be bit-exact on this device");
            eprintln!(
                "CUDA device {:?}: Sgemm+Dgemm known-answer probes BIT-EXACT \
                 (IEEE f32/f64 confirmed; no TF32/BF16x9/fixed-point emulation)",
                engine.device_name()
            );
        });
    }
}
