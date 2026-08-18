// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `#u4` — the TAINT-TWIN CHAIN runner: the full modeled resident chain, taint
//! words riding beside every value buffer.
//!
//! `ops/sentinel_taint_selfcheck.rs` MEASURES the defect on the shipped value
//! chain: lanes 2 and 5 launder the in-band magnitude sentinel by downscaling
//! (`1e10 * 1e-20 = 1e-10` under every guard while the stored `1e10` stands for
//! a true coefficient up to `~3.4e38`; one activation slope of `1e-25` turns
//! the `1e30` degrade marker into an ordinary `2.0e5` charge).
//! `ops/taint_channel_probe.rs` proves the fix PER OP: an out-of-band `u32`
//! taint word, OR'd and never multiplied, with clean exact-zero annihilation —
//! 10/10
//! green on the GB10 for the GEMM and activation twins.
//!
//! This module composes those per-op proofs into the WHOLE modeled chain:
//!
//! ```text
//!   V, tV   = GEMM_F32_TAINT(A, W, tA, tW)              // signed value channel
//!   S, tS   = GEMM_F32_TAINT(|A|, |W|, tA, tW)          // s_prod
//!   P, tP   = GEMM_F32_TAINT(E, |W|, tE, tW)            // prop
//!   E', tE' = CROWN_AW_ERROR_COMBINE_TAINT(S, P, tS, tP)
//!   A'', tA'', E'', tE'' = CROWN_ACTIVATION_RESIDENT_TAINT(V, E', tV, tE')
//! ```
//!
//! so a chain-level test can pin what no per-op probe can: that the taint set
//! at the head of the chain is still VISIBLE at the exact point where the
//! retained magnitude-only controls look (`|A''| >= FALLBACK_BOUND || E'' >=
//! FALLBACK_BOUND || nonfinite`), even though both in-band magnitudes have been
//! laundered small on the way. The armed production C1 preflight additionally
//! consults that word.
//!
//! # Propagation rule (canon — matches both twins)
//!
//! ```text
//! taint_out = OR over inputs of
//!             (taint_in AND (partner_value != 0 OR partner_taint != 0))
//!          OR (this op itself saturated/degraded)
//! ```
//!
//! Clean exact-zero partners annihilate (`R * 0 == 0` for every finite real the
//! sentinel stands for). A tainted stored zero cannot authenticate that fact.
//! Saturating to `±inf` instead is REFUTED — `inf * 0 = NaN` would collapse
//! tightness on every dead ReLU; see the
//! [`sh::GEMM_F32_TAINT_SHADER`] doc block.
//!
//! # Diagnostic chain runner; production plumbing is separate
//!
//! This is the SELFCHECK chain runner: one spec row, one output neuron,
//! contraction `k`, every mid-chain taint word staged out for
//! characterization. Production plumbing is now landed separately: the
//! resident walk transports words on device, ResNet composition ORs row words,
//! and the armed concretize C1 preflight fails closed on absent or tainted rows.
//! This diagnostic remains useful because it exposes every intermediate word.
//! Two deliberate simplifications, both taint-neutral:
//!
//! * `|A|` / `|W|` are computed on the HOST rather than via `ABS_COPY_SHADER`.
//!   `|x| == 0` iff `x == 0`, so abs changes neither a value's zero-ness nor
//!   its taint word. The production walk uses the same incoming word beside
//!   each absolute-value view.
//! * The combine's uniform scalars (`gamma_k`, `slack`, `additive`,
//!   `row_abs_a`, `w_l1_max`) are host-derived exactly as
//!   `sentinel_taint_selfcheck` derives them, so the value channel here is the
//!   same arithmetic the shipped chain runs.

use super::super::shaders as sh;
use super::super::sound_consts::{combine_slack_f32, gamma_k_f32};
use super::super::WgpuDevice;
use super::ibp_forward::create_buffer;
use ny_core::{NyError, Result};

/// Incoming taint words for the three head-of-chain operands, one flag per
/// element (`a`/`e` are the `1 x k` spec row and error row, `w` the `k x 1`
/// weight column).
#[derive(Debug, Clone, Copy)]
pub(crate) struct ChainTaints<'a> {
    pub(crate) a: &'a [bool],
    pub(crate) w: &'a [bool],
    pub(crate) e: &'a [bool],
}

/// End-of-chain measurement plus every mid-chain taint word, so a failing
/// composition is CHARACTERIZED (which hop dropped the word) rather than
/// merely observed.
///
/// `value_bits` / `err_bits` are RAW BITS (`A''`, `E''`), never float-loaded on
/// the way back — a float load could canonicalize a NaN and the saturation
/// cases depend on the exact stored pattern (same rule as
/// `taint_channel_probe`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ChainOutcome {
    /// `A''` — end-of-chain coefficient, raw bits.
    pub(crate) value_bits: u32,
    /// `E''` — end-of-chain certified error, raw bits.
    pub(crate) err_bits: u32,
    /// `tA''` — end-of-chain value taint.
    pub(crate) taint_a: bool,
    /// `tE''` — end-of-chain error taint.
    pub(crate) taint_e: bool,
    /// Mid-chain: `tV` out of the signed value GEMM.
    pub(crate) taint_v: bool,
    /// Mid-chain: `tS` out of the `|A|@|W|` GEMM.
    pub(crate) taint_s_prod: bool,
    /// Mid-chain: `tP` out of the `E@|W|` GEMM.
    pub(crate) taint_prop: bool,
    /// Mid-chain: `tE'` out of the AW error combine.
    pub(crate) taint_e_combined: bool,
    /// Mid-chain: `E'` raw bits (`1e30` bits ⇒ the combine's degrade fired).
    pub(crate) combined_err_bits: u32,
}

/// `Params { n, slack, gamma_k, additive, k, out_cols, w_l1_max, _pad }` —
/// [`sh::CROWN_AW_ERROR_COMBINE_TAINT_SHADER`], byte-identical to the base
/// combine's uniform (the twin copies the base arithmetic byte-for-byte).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CombineParams {
    n: u32,
    slack: f32,
    gamma_k: f32,
    additive: f32,
    k: u32,
    out_cols: u32,
    w_l1_max: f32,
    _pad: u32,
}

impl WgpuDevice {
    /// Dispatch the FULL modeled `#u4` chain with taint words and read back the
    /// end state plus every mid-chain taint word.
    ///
    /// One spec row (`a_row`, `e_row`, both `1 x k`) against one weight column
    /// (`w_col`, `k x 1`), then combine and a single-neuron resident
    /// activation with `ls = slopes.0`, `us = slopes.1`, `beta_signed = beta`,
    /// `eft_mode` selecting the activation's legacy/EFT error branch (the
    /// taint rule is branch-independent; both are exercised by the tests).
    ///
    /// Diagnostic-grade only — see the module doc. Fail-closed: any shape
    /// mismatch or GPU/readback error is `Err`, never a defaulted outcome.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn taint_chain_probe(
        &self,
        k: usize,
        a_row: &[f32],
        w_col: &[f32],
        e_row: &[f32],
        slopes: (f32, f32),
        beta: f32,
        eft_mode: bool,
        taints: ChainTaints<'_>,
    ) -> Result<ChainOutcome> {
        if k == 0
            || a_row.len() != k
            || w_col.len() != k
            || e_row.len() != k
            || taints.a.len() != k
            || taints.w.len() != k
            || taints.e.len() != k
        {
            return Err(NyError::InternalError(format!(
                "taint_chain_probe: inconsistent shapes (k={k}, a={}, w={}, e={}, ta={}, tw={}, te={})",
                a_row.len(),
                w_col.len(),
                e_row.len(),
                taints.a.len(),
                taints.w.len(),
                taints.e.len(),
            )));
        }
        let k_u32 = super::gpu_checked_u32(k, "taint_chain k")?;

        // ---- host-side derived operands (taint-neutral; see module doc) ----
        let abs_a: Vec<f32> = a_row.iter().map(|x| x.abs()).collect();
        let abs_w: Vec<f32> = w_col.iter().map(|x| x.abs()).collect();
        // Uniform scalars exactly as the resident driver / selfcheck derive
        // them: per-spec-row ‖a‖₁ and the scalar over-bound on max_j‖w_j‖₁
        // (one column here, so its own L1).
        let row_abs_a: f32 = abs_a.iter().sum();
        let w_l1_max: f32 = abs_w.iter().sum();
        let gamma = gamma_k_f32(k)?;
        let slack = combine_slack_f32(k)?;
        let additive = ny_core::ftz_safe_underflow_floor(k_u32);

        let ta_words: Vec<u32> = taints.a.iter().map(|&t| u32::from(t)).collect();
        let tw_words: Vec<u32> = taints.w.iter().map(|&t| u32::from(t)).collect();
        let te_words: Vec<u32> = taints.e.iter().map(|&t| u32::from(t)).collect();

        // ---- device buffers ------------------------------------------------
        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let storage_src = storage | wgpu::BufferUsages::COPY_SRC;
        let uniform = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
        let dev = &self.device;
        let f32_buf = |label: &'static str, data: &[f32], usage: wgpu::BufferUsages| {
            let b = create_buffer(dev, label, (data.len() * 4) as u64, usage);
            self.queue.write_buffer(&b, 0, bytemuck::cast_slice(data));
            b
        };
        let u32_buf = |label: &'static str, data: &[u32], usage: wgpu::BufferUsages| {
            let b = create_buffer(dev, label, (data.len() * 4) as u64, usage);
            self.queue.write_buffer(&b, 0, bytemuck::cast_slice(data));
            b
        };

        let a_buf = f32_buf("tc_a", a_row, storage);
        let w_buf = f32_buf("tc_w", w_col, storage);
        let e_buf = f32_buf("tc_e", e_row, storage);
        let abs_a_buf = f32_buf("tc_abs_a", &abs_a, storage);
        let abs_w_buf = f32_buf("tc_abs_w", &abs_w, storage);
        let ls_buf = f32_buf("tc_ls", &[slopes.0], storage);
        let us_buf = f32_buf("tc_us", &[slopes.1], storage);
        let beta_buf = f32_buf("tc_beta", &[beta], storage);
        let row_abs_buf = f32_buf("tc_row_abs_a", &[row_abs_a], storage);

        let v_buf = f32_buf("tc_v", &[0.0], storage);
        let s_buf = f32_buf("tc_s_prod", &[0.0], storage);
        let p_buf = f32_buf("tc_prop", &[0.0], storage);
        let ecomb_buf = f32_buf("tc_e_combined", &[0.0], storage_src);
        let aout_buf = f32_buf("tc_a_out", &[0.0], storage_src);
        let eout_buf = f32_buf("tc_e_out", &[0.0], storage_src);

        let ta_buf = u32_buf("tc_taint_a", &ta_words, storage);
        let tw_buf = u32_buf("tc_taint_w", &tw_words, storage);
        let te_buf = u32_buf("tc_taint_e", &te_words, storage);
        let tv_buf = u32_buf("tc_taint_v", &[0u32], storage_src);
        let ts_buf = u32_buf("tc_taint_s", &[0u32], storage_src);
        let tp_buf = u32_buf("tc_taint_p", &[0u32], storage_src);
        let tec_buf = u32_buf("tc_taint_ec", &[0u32], storage_src);
        let taout_buf = u32_buf("tc_taint_a_out", &[0u32], storage_src);
        let teout_buf = u32_buf("tc_taint_e_out", &[0u32], storage_src);

        // ---- uniforms ------------------------------------------------------
        let gemm_p = create_buffer(dev, "tc_gemm_p", 16, uniform);
        // Params { m, k, n, _padding }
        self.queue
            .write_buffer(&gemm_p, 0, bytemuck::cast_slice(&[1u32, k_u32, 1u32, 0u32]));

        let combine_p = create_buffer(dev, "tc_combine_p", 32, uniform);
        self.queue.write_buffer(
            &combine_p,
            0,
            bytemuck::cast_slice(&[CombineParams {
                n: 1,
                slack,
                gamma_k: gamma,
                additive,
                k: k_u32,
                out_cols: 1,
                w_l1_max,
                _pad: 0,
            }]),
        );

        let act_p = create_buffer(dev, "tc_act_p", 32, uniform);
        // Params { num_specs, num_neurons, is_upper, additive(f32),
        //          num_specs_per_dom, eft_mode, _p1, _p2 }
        let mut act_words = [0u32; 8];
        act_words[0] = 1; // num_specs
        act_words[1] = 1; // num_neurons
        act_words[2] = 0; // is_upper = lower
        act_words[3] = additive.to_bits();
        act_words[4] = 1; // num_specs_per_dom
        act_words[5] = u32::from(eft_mode);
        self.queue
            .write_buffer(&act_p, 0, bytemuck::cast_slice(&act_words));

        // ---- pipelines: the three taint twins ------------------------------
        let gemm = self.create_simple_pipeline(
            sh::GEMM_F32_TAINT_SHADER,
            "tc_gemm",
            // a, b, out, taint_a, taint_b, taint_out
            &[false, false, true, false, false, true],
        );
        let combine = self.create_simple_pipeline(
            sh::CROWN_AW_ERROR_COMBINE_TAINT_SHADER,
            "tc_combine",
            // Base combine's EXACT list first (s_prod, prop, err_out,
            // row_abs_a), taints appended after:
            // taint_sprod_in, taint_prop_in, taint_e_out.
            &[false, false, true, false, false, false, true],
        );
        let act = self.create_simple_pipeline(
            sh::CROWN_ACTIVATION_RESIDENT_TAINT_SHADER,
            "tc_act",
            // a_in, err_in, ls, us, a_out, err_out, beta,
            // taint_a_in, taint_e_in, taint_a_out, taint_e_out
            &[
                false, false, false, false, true, true, false, false, false, true, true,
            ],
        );

        // ---- encode the chain (pass boundaries are the barriers) -----------
        let mut encoder = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("taint_chain_encoder"),
        });
        // V, tV = GEMM_TAINT(A, W, tA, tW) — m=1, n=1 ⇒ one 16x16 workgroup.
        self.pass_simple_2d(
            &mut encoder,
            &gemm,
            &gemm_p,
            &[&a_buf, &w_buf, &v_buf, &ta_buf, &tw_buf, &tv_buf],
            1,
            1,
        );
        // S, tS = GEMM_TAINT(|A|, |W|, tA, tW) — abs preserves zero-ness, so
        // the ORIGINAL taint words are the right inputs.
        self.pass_simple_2d(
            &mut encoder,
            &gemm,
            &gemm_p,
            &[&abs_a_buf, &abs_w_buf, &s_buf, &ta_buf, &tw_buf, &ts_buf],
            1,
            1,
        );
        // P, tP = GEMM_TAINT(E, |W|, tE, tW)
        self.pass_simple_2d(
            &mut encoder,
            &gemm,
            &gemm_p,
            &[&e_buf, &abs_w_buf, &p_buf, &te_buf, &tw_buf, &tp_buf],
            1,
            1,
        );
        // E', tE' = COMBINE_TAINT(S, P, tS, tP)
        self.pass_simple(
            &mut encoder,
            &combine,
            &combine_p,
            &[
                &s_buf,
                &p_buf,
                &ecomb_buf,
                &row_abs_buf,
                &ts_buf,
                &tp_buf,
                &tec_buf,
            ],
            1,
        );
        // A'', tA'', E'', tE'' = ACTIVATION_TAINT(V, E', tV, tE')
        self.pass_simple(
            &mut encoder,
            &act,
            &act_p,
            &[
                &v_buf, &ecomb_buf, &ls_buf, &us_buf, &aout_buf, &eout_buf, &beta_buf, &tv_buf,
                &tec_buf, &taout_buf, &teout_buf,
            ],
            1,
        );

        // ---- staging + readback -------------------------------------------
        // Storage buffers are not MAP_READ; every readback goes through a
        // 4-byte staging copy and is read as RAW BITS (never a float load).
        let stage = |label: &'static str| {
            create_buffer(
                dev,
                label,
                4,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            )
        };
        let st_val = stage("tc_st_val");
        let st_err = stage("tc_st_err");
        let st_ecomb = stage("tc_st_ecomb");
        let st_ta = stage("tc_st_ta");
        let st_te = stage("tc_st_te");
        let st_tv = stage("tc_st_tv");
        let st_ts = stage("tc_st_ts");
        let st_tp = stage("tc_st_tp");
        let st_tec = stage("tc_st_tec");
        for (src, dst) in [
            (&aout_buf, &st_val),
            (&eout_buf, &st_err),
            (&ecomb_buf, &st_ecomb),
            (&taout_buf, &st_ta),
            (&teout_buf, &st_te),
            (&tv_buf, &st_tv),
            (&ts_buf, &st_ts),
            (&tp_buf, &st_tp),
            (&tec_buf, &st_tec),
        ] {
            encoder.copy_buffer_to_buffer(src, 0, dst, 0, 4);
        }
        self.queue.submit(std::iter::once(encoder.finish()));

        let word = |staged: &wgpu::Buffer| -> Result<u32> {
            Ok(WgpuDevice::read_u32_buffer(dev, staged, 1)?[0])
        };
        Ok(ChainOutcome {
            value_bits: word(&st_val)?,
            err_bits: word(&st_err)?,
            taint_a: word(&st_ta)? != 0,
            taint_e: word(&st_te)? != 0,
            taint_v: word(&st_tv)? != 0,
            taint_s_prod: word(&st_ts)? != 0,
            taint_prop: word(&st_tp)? != 0,
            taint_e_combined: word(&st_tec)? != 0,
            combined_err_bits: word(&st_ecomb)?,
        })
    }
}

mod gpu_tests {
    use super::*;

    /// `FALLBACK_BOUND` — the in-band sentinel whose magnitude the downstream
    /// guards test, and which the chain launders in lanes 2/5.
    const SENTINEL: f32 = ny_core::FALLBACK_BOUND;

    /// The combine's deliberate in-band degrade charge — a magnitude that can
    /// be laundered, which is why the armed out-of-band word accompanies it.
    const ERR_TAINT: f32 = 1e30;

    /// Contraction length, matching `sentinel_taint_selfcheck`'s lanes.
    const K: usize = 4;

    const NO_TAINT: [bool; K] = [false; K];
    const HEAD_TAINT: [bool; K] = [true, false, false, false];

    fn device() -> WgpuDevice {
        WgpuDevice::new().expect("wgpu adapter for the taint-chain probe")
    }

    /// LANE-2 MODEL, END TO END: the sentinel in `A` times a `1e-20` weight.
    /// The VALUE must launder (measured `1e-10` on the shipped chain — that is
    /// the defect) while the out-of-band taint reaches BOTH end words through
    /// all five hops.
    #[test]
    fn lane2_sentinel_downscale_keeps_taint_end_to_end() {
        let dev = device();
        let a = [SENTINEL, 0.0, 0.0, 0.0];
        let w = [1e-20, 0.0, 0.0, 0.0];
        let e = [0.0; K];
        for eft in [false, true] {
            let out = dev
                .taint_chain_probe(
                    K,
                    &a,
                    &w,
                    &e,
                    (1.0, 1.0),
                    0.0,
                    eft,
                    ChainTaints {
                        a: &HEAD_TAINT,
                        w: &NO_TAINT,
                        e: &NO_TAINT,
                    },
                )
                .expect("chain dispatch");
            let value = f32::from_bits(out.value_bits);
            let err = f32::from_bits(out.err_bits);
            assert!(
                value.abs() < SENTINEL && err.is_finite() && err < SENTINEL,
                "eft={eft}: BOTH in-band magnitudes must launder — that is the \
                 defect being worked around (A''={value:e}, E''={err:e})"
            );
            assert!(
                out.taint_v && out.taint_s_prod,
                "eft={eft}: the value-side GEMMs must carry tA through the \
                 1e-20 downscale (out={out:?})"
            );
            assert!(
                out.taint_e_combined,
                "eft={eft}: the combine must OR tS into tE' (out={out:?})"
            );
            assert!(
                out.taint_a,
                "eft={eft}: tA'' must survive end-to-end (out={out:?})"
            );
            assert!(
                out.taint_e,
                "eft={eft}: a tainted coefficient's own rounding charge makes \
                 the value taint contaminate the error word (out={out:?})"
            );
        }
    }

    /// LANE-5 MODEL, END TO END: the `1e30` degrade marker in `E` against a
    /// `1e-25` activation slope. The `err@|W|` GEMM clamps `1e30` to the
    /// sentinel, the combine's `prop >= FALLBACK_BOUND` arm re-degrades to
    /// `1e30`, and the activation slope launders it to `~2.0e5` (measured) —
    /// but the taint word must ride through every hop.
    #[test]
    fn lane5_degrade_marker_survives_combine_and_tiny_slope() {
        let dev = device();
        let a = [1.0, 0.0, 0.0, 0.0];
        let w = [1.0, 0.0, 0.0, 0.0];
        let e = [ERR_TAINT, 0.0, 0.0, 0.0];
        for eft in [false, true] {
            let out = dev
                .taint_chain_probe(
                    K,
                    &a,
                    &w,
                    &e,
                    (1e-25, 1e-25),
                    0.0,
                    eft,
                    ChainTaints {
                        a: &NO_TAINT,
                        w: &NO_TAINT,
                        e: &HEAD_TAINT,
                    },
                )
                .expect("chain dispatch");
            assert!(
                out.taint_prop,
                "eft={eft}: tE must reach tP through the nonzero |W| column \
                 (and the clamped 1e30 seeds it besides) (out={out:?})"
            );
            assert_eq!(
                f32::from_bits(out.combined_err_bits),
                ERR_TAINT,
                "eft={eft}: the combine's `prop >= FALLBACK_BOUND` degrade arm \
                 must fire (out={out:?})"
            );
            assert!(
                out.taint_e_combined,
                "eft={eft}: the combine must carry the taint out of band too \
                 (out={out:?})"
            );
            let err = f32::from_bits(out.err_bits);
            assert!(
                err.is_finite() && err < SENTINEL,
                "eft={eft}: the in-band E'' must launder under the 1e-25 slope \
                 (measured 2.0000019e5 on the shipped chain), got {err:e}"
            );
            assert!(
                out.taint_e,
                "eft={eft}: tE'' must survive the tiny nonzero slope (out={out:?})"
            );
            assert!(
                !out.taint_a && !out.taint_v,
                "eft={eft}: no value taint may appear from nowhere (out={out:?})"
            );
        }
    }

    /// CLEAN CONTROL (the channel must not be trivially satisfiable by
    /// tainting everything) + DEAD-RELU ANNIHILATION at chain level (the case
    /// that refutes saturating to ±inf: `R * 0 == 0` exactly, so a slope of
    /// exact zero legitimately clears BOTH channels' taints).
    #[test]
    fn clean_stays_clean_and_dead_relu_annihilates() {
        let dev = device();

        // Clean: ordinary operands, no incoming taint, nothing saturates.
        let a = [2.0, 0.0, 0.0, 0.0];
        let w = [3.0, 0.0, 0.0, 0.0];
        let e = [1e-7, 0.0, 0.0, 0.0];
        for eft in [false, true] {
            let out = dev
                .taint_chain_probe(
                    K,
                    &a,
                    &w,
                    &e,
                    (0.7, 0.7),
                    0.0,
                    eft,
                    ChainTaints {
                        a: &NO_TAINT,
                        w: &NO_TAINT,
                        e: &NO_TAINT,
                    },
                )
                .expect("chain dispatch");
            // V = 2*3 = 6 exactly; A'' = fl(6 * 0.7), one RN multiply on both
            // host and device.
            assert_eq!(
                f32::from_bits(out.value_bits),
                6.0f32 * 0.7f32,
                "clean value must come through untouched (out={out:?})"
            );
            let err = f32::from_bits(out.err_bits);
            assert!(
                err.is_finite() && (0.0..SENTINEL).contains(&err),
                "eft={eft}: clean error stays an ordinary charge, got {err:e}"
            );
            assert!(
                !out.taint_a
                    && !out.taint_e
                    && !out.taint_v
                    && !out.taint_s_prod
                    && !out.taint_prop
                    && !out.taint_e_combined,
                "eft={eft}: no hop may invent taint on clean inputs (out={out:?})"
            );
        }

        // Dead ReLU: taint in BOTH channels, slope EXACTLY 0. The activation
        // annihilates the coefficient exactly, so both taints legitimately go
        // with it — under ±inf saturation this same lane would be inf*0 = NaN.
        for eft in [false, true] {
            let out = dev
                .taint_chain_probe(
                    K,
                    &[SENTINEL, 0.0, 0.0, 0.0],
                    &[1.0, 0.0, 0.0, 0.0],
                    &[ERR_TAINT, 0.0, 0.0, 0.0],
                    (0.0, 0.0),
                    0.0,
                    eft,
                    ChainTaints {
                        a: &HEAD_TAINT,
                        w: &NO_TAINT,
                        e: &HEAD_TAINT,
                    },
                )
                .expect("chain dispatch");
            assert!(
                out.taint_v && out.taint_e_combined,
                "eft={eft}: the taints must be ARMED going into the activation \
                 (else this proves nothing) (out={out:?})"
            );
            assert_eq!(
                f32::from_bits(out.value_bits),
                0.0,
                "eft={eft}: the dead ReLU must annihilate EXACTLY (out={out:?})"
            );
            assert!(
                !out.taint_a && !out.taint_e,
                "eft={eft}: an exact-zero slope must clear both taints — \
                 otherwise every dead ReLU poisons its row (out={out:?})"
            );
        }
    }
}
