// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `#u4` — device proof of the C2 consult (`TAINT_GUARD_AUDIT.md` §4): the EFT
//! min-combine refuses the chain's ONLY error-LOWERING op on a set taint word.
//!
//! `ops/sentinel_taint_selfcheck.rs` measures the laundering defect, and the
//! audit's G2 row grades the base shader's guards MAGNITUDE-only: a laundered
//! `s_prod` (lane 2: `1e10 * 1e-20 = 1e-10`) sails under both
//! `>= FALLBACK_BOUND` arms and lets `min(err_out, e_eft)` LOWER a
//! deliberately-degraded charge. That erasure happens mid-chain, per element,
//! never read back — so it re-opens the hole even after the C1 preflight
//! consult (C1 catches what survives; C2 prevents the one op that can
//! un-survive it). The twin [`sh::CROWN_EFT_MIN_COMBINE_TAINT_SHADER`] closes
//! it with two read-only `u32` word bindings and a refusal no downscale can
//! launder.
//!
//! # The properties under test
//!
//! * CLEAN — clean words, tightening applicable: the twin is BIT-IDENTICAL to
//!   a base-shader control run (the consult is purely additive; without this
//!   the twin could drift from the arithmetic it claims to copy).
//! * LAUNDER-S / LAUNDER-P — a set word refuses the tightening even when
//!   every magnitude is innocent (the lane-2 shape), where the base control
//!   run demonstrably tightens — proving the WORD, not a magnitude, refused.
//! * NONFINITE — a non-finite input refuses before anything, words clean, on
//!   both variants (the base refusal order is preserved).
//!
//! This probe module compiles only under the gpu-tests build (see `ops/mod.rs`).
//! The twin itself is selected in the ordinary resident walk when the resolved
//! word gate (AUTO/default when twins are available, forced by
//! `NY_GPU_TAINT_WORDS=1`, opted out by `=0`) and optional EFT min-tightening
//! are both active. The armed `PRODUCTION_GUARDS_CONSULT_TAINT_WORD` controls
//! C1/selfcheck classification, not C2 pipeline selection.

use super::super::shaders as sh;
use super::super::WgpuDevice;
use super::ibp_forward::create_buffer;
use ny_core::Result;

/// One element's worth of inputs to the EFT min-combine (both variants).
///
/// `err_in` is the Higham charge already sitting in `err_out` before the min;
/// a refusal must hand it back bit-untouched. The base shader has no taint
/// bindings, so `taint_s`/`taint_p` are simply not bound on a control run.
#[derive(Debug, Clone, Copy)]
pub(crate) struct EftMinCombineProbeInputs {
    pub(crate) v_twin: f32,
    pub(crate) r_in: f32,
    pub(crate) value: f32,
    pub(crate) prop: f32,
    pub(crate) err_in: f32,
    pub(crate) row_abs_a: f32,
    pub(crate) s_prod: f32,
    pub(crate) taint_s: bool,
    pub(crate) taint_p: bool,
}

impl WgpuDevice {
    /// Run one element through the EFT min-combine and report the stored
    /// `err_out` as RAW BITS (a float load could canonicalize a NaN).
    ///
    /// `use_taint_twin = false` dispatches the base
    /// [`sh::CROWN_EFT_MIN_COMBINE_SHADER`] as the control; `true` dispatches
    /// [`sh::CROWN_EFT_MIN_COMBINE_TAINT_SHADER`] with the two word buffers
    /// appended. Params are fixed at the identity slacks (`r_slack = slack =
    /// 1.0`, `additive = 0`, `k = 1`, `out_cols = 1`, `w_l1_max = 0`): valid
    /// probe values that keep both arithmetic paths byte-comparable. Mirrors
    /// [`WgpuDevice::taint_channel_probe_1x1`].
    pub(crate) fn eft_min_combine_probe_single(
        &self,
        inputs: &EftMinCombineProbeInputs,
        use_taint_twin: bool,
    ) -> Result<u32> {
        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let storage_src = storage | wgpu::BufferUsages::COPY_SRC;
        let dev = &self.device;

        let f32_buf = |label: &'static str, data: &[f32], usage: wgpu::BufferUsages| {
            let buffer = create_buffer(dev, label, (data.len() * 4) as u64, usage);
            self.queue
                .write_buffer(&buffer, 0, bytemuck::cast_slice(data));
            buffer
        };
        let u32_buf = |label: &'static str, data: &[u32], usage: wgpu::BufferUsages| {
            let buffer = create_buffer(dev, label, (data.len() * 4) as u64, usage);
            self.queue
                .write_buffer(&buffer, 0, bytemuck::cast_slice(data));
            buffer
        };

        let v_buf = f32_buf("eftmin_taint_v_twin", &[inputs.v_twin], storage);
        let r_buf = f32_buf("eftmin_taint_r_in", &[inputs.r_in], storage);
        let val_buf = f32_buf("eftmin_taint_value", &[inputs.value], storage);
        let prop_buf = f32_buf("eftmin_taint_prop", &[inputs.prop], storage);
        let err_buf = f32_buf("eftmin_taint_err_out", &[inputs.err_in], storage_src);
        let row_buf = f32_buf("eftmin_taint_row_abs_a", &[inputs.row_abs_a], storage);
        let sp_buf = f32_buf("eftmin_taint_s_prod", &[inputs.s_prod], storage);
        let ts_buf = u32_buf("eftmin_taint_word_s", &[u32::from(inputs.taint_s)], storage);
        let tp_buf = u32_buf("eftmin_taint_word_p", &[u32::from(inputs.taint_p)], storage);

        let params = create_buffer(
            dev,
            "eftmin_taint_params",
            32,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        // Params { n, r_slack, slack, additive, k, out_cols, w_l1_max, _pad }
        let mut words = [0u32; 8];
        words[0] = 1; // n
        words[1] = 1.0f32.to_bits(); // r_slack
        words[2] = 1.0f32.to_bits(); // slack
        words[3] = 0.0f32.to_bits(); // additive
        words[4] = 1; // k
        words[5] = 1; // out_cols (row_abs_a[0] read, as in production)
        words[6] = 0.0f32.to_bits(); // w_l1_max
        self.queue
            .write_buffer(&params, 0, bytemuck::cast_slice(&words));

        let (src, label, rw): (&str, &'static str, &'static [bool]) = if use_taint_twin {
            (
                sh::CROWN_EFT_MIN_COMBINE_TAINT_SHADER,
                "eftmin_taint_twin",
                // v_twin, r_in, value, prop, err_out, row_abs_a, s_prod, taint_s, taint_p
                &[false, false, false, false, true, false, false, false, false],
            )
        } else {
            (
                sh::CROWN_EFT_MIN_COMBINE_SHADER,
                "eftmin_taint_base",
                // v_twin, r_in, value, prop, err_out, row_abs_a, s_prod
                &[false, false, false, false, true, false, false],
            )
        };
        let pipeline = self.create_simple_pipeline(src, label, rw);

        let mut encoder = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("eftmin_taint_encoder"),
        });
        let mut buffers: Vec<&wgpu::Buffer> = vec![
            &v_buf, &r_buf, &val_buf, &prop_buf, &err_buf, &row_buf, &sp_buf,
        ];
        if use_taint_twin {
            buffers.push(&ts_buf);
            buffers.push(&tp_buf);
        }
        self.pass_simple_2d(&mut encoder, &pipeline, &params, &buffers, 1, 1);
        // Staging copy: storage buffers are not MAP_READ.
        let st_err = create_buffer(
            dev,
            "eftmin_taint_st_err",
            4,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        encoder.copy_buffer_to_buffer(&err_buf, 0, &st_err, 0, 4);
        self.queue.submit(std::iter::once(encoder.finish()));

        let bits = WgpuDevice::read_u32_buffer(dev, &st_err, 1)?;
        Ok(bits[0])
    }
}

mod gpu_tests {
    use super::*;

    fn device() -> WgpuDevice {
        WgpuDevice::new().expect("wgpu adapter for the EFT min-combine taint probe")
    }

    /// A configuration on which the EFT bound (`~2e-4`) is strictly tighter
    /// than the Higham charge (`1.0`) and every magnitude is ordinary: the min
    /// MUST fire unless something refuses it.
    fn tightening_inputs() -> EftMinCombineProbeInputs {
        EftMinCombineProbeInputs {
            v_twin: 5.0,
            r_in: 1e-4,
            value: 5.0,
            prop: 1e-4,
            err_in: 1.0,
            row_abs_a: 1.0,
            s_prod: 2.0,
            taint_s: false,
            taint_p: false,
        }
    }

    /// (a) CLEAN: with both words clear the twin's tightening is BIT-IDENTICAL
    /// to a base-shader control run — the consult is purely additive, and the
    /// arithmetic really is the base's byte-for-byte.
    #[test]
    fn clean_words_tighten_bit_identically_to_the_base() {
        let dev = device();
        let inputs = tightening_inputs();
        let twin = dev
            .eft_min_combine_probe_single(&inputs, true)
            .expect("twin dispatch");
        let base = dev
            .eft_min_combine_probe_single(&inputs, false)
            .expect("base control dispatch");
        assert_eq!(
            twin, base,
            "clean words must leave the twin bit-identical to the base"
        );
        let err = f32::from_bits(twin);
        assert!(
            err < inputs.err_in,
            "the tightening must actually fire on this configuration, got {err:e}"
        );
    }

    /// (b) LAUNDER-S: the lane-2 shape — `s_prod` holds an innocent `1e-10`
    /// (the `1e10` sentinel after ONE `1e-20` weight) so the base control run
    /// tightens straight past its magnitude arms, while the WORD refuses.
    #[test]
    fn laundered_s_prod_word_refuses_the_tightening() {
        let dev = device();
        let mut inputs = tightening_inputs();
        inputs.s_prod = 1e-10;
        inputs.taint_s = true;

        let base = dev
            .eft_min_combine_probe_single(&inputs, false)
            .expect("base control dispatch");
        assert!(
            f32::from_bits(base) < inputs.err_in,
            "control: the magnitude arms MUST launder here — that is the \
             measured defect this consult closes — got {:e}",
            f32::from_bits(base)
        );

        let twin = dev
            .eft_min_combine_probe_single(&inputs, true)
            .expect("twin dispatch");
        assert_eq!(
            twin,
            inputs.err_in.to_bits(),
            "taint_s must refuse the min: err_out has to come back bit-untouched"
        );
    }

    /// (c) LAUNDER-P: same shape on the propagated term — `prop` downscaled to
    /// an innocent `1e-10`, `taint_p` set. Base tightens, the word refuses.
    #[test]
    fn laundered_prop_word_refuses_the_tightening() {
        let dev = device();
        let mut inputs = tightening_inputs();
        inputs.prop = 1e-10;
        inputs.taint_p = true;

        let base = dev
            .eft_min_combine_probe_single(&inputs, false)
            .expect("base control dispatch");
        assert!(
            f32::from_bits(base) < inputs.err_in,
            "control: an innocent-magnitude prop must launder past the base, got {:e}",
            f32::from_bits(base)
        );

        let twin = dev
            .eft_min_combine_probe_single(&inputs, true)
            .expect("twin dispatch");
        assert_eq!(
            twin,
            inputs.err_in.to_bits(),
            "taint_p must refuse the min: err_out has to come back bit-untouched"
        );
    }

    /// (d) NONFINITE: with words CLEAN and the tightening otherwise
    /// applicable, a NaN in any read operand still refuses — on BOTH variants,
    /// bit-identically — pinning that the base's fail-closed arm survived the
    /// twin edit and still runs before anything can tighten.
    #[test]
    fn nonfinite_input_still_refuses_before_anything() {
        let dev = device();
        for field in 0..4u32 {
            let mut inputs = tightening_inputs();
            match field {
                0 => inputs.v_twin = f32::NAN,
                1 => inputs.r_in = f32::NAN,
                2 => inputs.value = f32::NAN,
                _ => inputs.prop = f32::NAN,
            }
            for use_twin in [false, true] {
                let got = dev
                    .eft_min_combine_probe_single(&inputs, use_twin)
                    .expect("dispatch");
                assert_eq!(
                    got,
                    inputs.err_in.to_bits(),
                    "twin={use_twin} field={field}: a non-finite input must \
                     refuse and hand err_out back bit-untouched"
                );
            }
        }
    }

    /// Source-level pin of the refusal ORDER inside the twin: non-finite
    /// first, then the word consult, then the (redundant) magnitude arms, then
    /// the min — behavior alone cannot distinguish the order of two refusals,
    /// so pin the text the way `contract_tests.rs` pins the base's arms. Also
    /// pins the tightening arithmetic byte-identical between base and twin.
    #[test]
    fn twin_source_pins_refusal_order_and_base_arithmetic() {
        let compact: String = sh::CROWN_EFT_MIN_COMBINE_TAINT_SHADER
            .chars()
            .filter(|ch| !ch.is_whitespace())
            .collect();
        let pos = |needle: &str| {
            compact
                .find(needle)
                .unwrap_or_else(|| panic!("twin lost `{needle}`"))
        };
        let nonfinite = pos(
            "if(is_nonfinite(v)||is_nonfinite(val)||is_nonfinite(r)||is_nonfinite(pr)){return;}",
        );
        let consult = pos("if(taint_s[i]!=0u||taint_p[i]!=0u){return;}");
        let arm_pr = pos("if(pr>=FALLBACK_BOUND){return;}");
        let arm_sp = pos("if(s_prod[i]>=FALLBACK_BOUND){return;}");
        let tighten = pos("err_out[i]=min(err_out[i],e_eft);");
        assert!(
            nonfinite < consult,
            "the non-finite refusal must stay FIRST (base order preserved)"
        );
        assert!(
            consult < arm_pr && arm_pr < arm_sp && arm_sp < tighten,
            "the word consult must precede the redundant magnitude arms, and \
             every refusal must precede the tightening min"
        );

        let base: String = sh::CROWN_EFT_MIN_COMBINE_SHADER
            .chars()
            .filter(|ch| !ch.is_whitespace())
            .collect();
        for needle in [
            "letd=abs(v-val);",
            "varflushacc=1.0+f32(p.k)+p.w_l1_max;",
            "letflush=p.additive+flushacc*p.slack*F32_MIN_NORMAL;",
            "lete_eft=round_up_pos((r+d)*p.r_slack+pr*p.slack+flush);",
            "if(is_nonfinite(e_eft)||e_eft<0.0){return;}",
        ] {
            assert!(
                base.contains(needle) && compact.contains(needle),
                "arithmetic drift between base and twin at `{needle}` — \
                 re-derive the twin from the base before trusting the probes"
            );
        }
    }
}
