// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `#u4` — device proof that the OUT-OF-BAND TAINT CHANNEL survives the two
//! compositions that destroy the in-band magnitude sentinel.
//!
//! `ops/sentinel_taint_selfcheck.rs` MEASURES the defect: the finite
//! `±FALLBACK_BOUND` overflow sentinel and the `1e30` degrade marker are
//! NUMBERS, both downstream guards are MAGNITUDE tests, and so one small weight
//! launders the taint. Its lanes 2 and 5 fail on every adapter, and because both
//! guards are magnitude comparisons against a finite constant the failure is in
//! NY's kernel design rather than the hardware — CUDA launders identically.
//!
//! This module is the other half: [`sh::GEMM_F32_TAINT_SHADER`] carries a `u32`
//! taint word beside the value, OR'd and never multiplied, and the tests here
//! run it ON DEVICE at the same four shapes the failing lanes use.
//!
//! # The property under test
//!
//! ```text
//! taint_out[row, col] = OR over k of
//!       (taint_a[row, k] AND (b[k, col] != 0 OR taint_b[k, col]))
//!    OR (taint_b[k, col] AND (a[row, k] != 0 OR taint_a[row, k]))
//!    OR  the output saturated
//! ```
//!
//! A clean stored zero authenticates exact annihilation; a tainted stored zero
//! does not. Each side of that rule is pinned below:
//!
//! * DOWNSCALE — tainted operand, tiny NONZERO partner. Must stay tainted. This
//!   is what the retained magnitude-only control fails: `1e10 * 1e-20 = 1e-10`
//!   sails under every magnitude guard while standing for a true coefficient
//!   up to `~3.4e38`; the armed production word channel closes that hole.
//! * ANNIHILATION — tainted operand, CLEAN exact-zero partner (a dead ReLU).
//!   Must clear, because `R * 0 == 0` for every finite real `R` the sentinel
//!   could stand for. Keeping the taint here is what saturating to `±inf` would
//!   do instead (`inf * 0 = NaN`), and dead ReLUs are the most common event in
//!   a deep network, so that would trade a laundering bug for a tightness
//!   collapse on the hot path. Two tainted stored zeros must remain tainted.
//! * SEEDING — an output that saturates at this op is itself tainted, so the
//!   channel starts without anyone having to prime it.
//! * CLEAN — untainted in, no saturation, must stay untainted. Without this the
//!   channel could be trivially satisfied by tainting everything.

use super::super::shaders as sh;
use super::super::WgpuDevice;
use super::ibp_forward::create_buffer;
use ny_core::Result;

/// `FALLBACK_BOUND` — the in-band sentinel whose magnitude the retained control
/// guards test, and which the production word channel no longer relies on alone.
const SENTINEL: f32 = 1e10;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TaintOutcome {
    pub(crate) value_bits: u32,
    pub(crate) tainted: bool,
}

impl WgpuDevice {
    /// Run one `1x1x1` GEMM through the taint channel and report both channels.
    ///
    /// `a`/`b` are the single value operands, `ta`/`tb` their incoming taint.
    pub(crate) fn taint_channel_probe_1x1(
        &self,
        a: f32,
        b: f32,
        ta: bool,
        tb: bool,
    ) -> Result<TaintOutcome> {
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

        let a_buf = f32_buf("taint_a_val", &[a], storage);
        let b_buf = f32_buf("taint_b_val", &[b], storage);
        let out_buf = f32_buf("taint_out_val", &[0.0], storage_src);
        let ta_buf = u32_buf("taint_a_word", &[u32::from(ta)], storage);
        let tb_buf = u32_buf("taint_b_word", &[u32::from(tb)], storage);
        let tout_buf = u32_buf("taint_out_word", &[0u32], storage_src);

        let params = create_buffer(
            dev,
            "taint_params",
            16,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        // Params { m, k, n, _padding }
        self.queue
            .write_buffer(&params, 0, bytemuck::cast_slice(&[1u32, 1u32, 1u32, 0u32]));

        let pipeline = self.create_simple_pipeline(
            sh::GEMM_F32_TAINT_SHADER,
            "taint_gemm",
            // a, b, out, taint_a, taint_b, taint_out
            &[false, false, true, false, false, true],
        );

        let mut encoder = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("taint_channel_encoder"),
        });
        self.pass_simple_2d(
            &mut encoder,
            &pipeline,
            &params,
            &[&a_buf, &b_buf, &out_buf, &ta_buf, &tb_buf, &tout_buf],
            1,
            1,
        );
        // Staging copies: storage buffers are not MAP_READ.
        let stage = |label: &'static str| {
            create_buffer(
                dev,
                label,
                4,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            )
        };
        let st_val = stage("taint_st_val");
        let st_taint = stage("taint_st_taint");
        encoder.copy_buffer_to_buffer(&out_buf, 0, &st_val, 0, 4);
        encoder.copy_buffer_to_buffer(&tout_buf, 0, &st_taint, 0, 4);
        self.queue.submit(std::iter::once(encoder.finish()));

        // Read as RAW BITS, never as a float: a float load could canonicalize a
        // NaN and the saturation cases depend on the exact stored pattern.
        let value = WgpuDevice::read_u32_buffer(dev, &st_val, 1)?;
        let taint = WgpuDevice::read_u32_buffer(dev, &st_taint, 1)?;
        Ok(TaintOutcome {
            value_bits: value[0],
            tainted: taint[0] != 0,
        })
    }
}

/// One-element outcome of the activation taint twin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ActivationTaintOutcome {
    pub(crate) taint_a: bool,
    pub(crate) taint_e: bool,
}

impl WgpuDevice {
    /// Run one single-neuron, single-spec element through
    /// [`sh::CROWN_ACTIVATION_RESIDENT_TAINT_SHADER`] and report both taint
    /// words. Mirrors [`WgpuDevice::taint_channel_probe_1x1`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn activation_taint_probe_single(
        &self,
        a: f32,
        e_in: f32,
        lower_slope: f32,
        upper_slope: f32,
        beta: f32,
        eft_mode: bool,
        is_upper: bool,
        taint_a: bool,
        taint_e: bool,
    ) -> Result<ActivationTaintOutcome> {
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

        let a_in = f32_buf("act_taint_a_in", &[a], storage);
        let e_buf = f32_buf("act_taint_e_in", &[e_in], storage);
        let ls = f32_buf("act_taint_ls", &[lower_slope], storage);
        let us = f32_buf("act_taint_us", &[upper_slope], storage);
        let a_out = f32_buf("act_taint_a_out", &[0.0], storage_src);
        let e_out = f32_buf("act_taint_e_out", &[0.0], storage_src);
        let beta_buf = f32_buf("act_taint_beta", &[beta], storage);
        let ta_in = u32_buf("act_taint_ta_in", &[u32::from(taint_a)], storage);
        let te_in = u32_buf("act_taint_te_in", &[u32::from(taint_e)], storage);
        let ta_out = u32_buf("act_taint_ta_out", &[0u32], storage_src);
        let te_out = u32_buf("act_taint_te_out", &[0u32], storage_src);

        let params = create_buffer(
            dev,
            "act_taint_params",
            32,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        // Params { num_specs, num_neurons, is_upper, additive(f32), num_specs_per_dom, eft_mode, _p1, _p2 }
        let mut words = [0u32; 8];
        words[0] = 1; // num_specs
        words[1] = 1; // num_neurons
        words[2] = u32::from(is_upper);
        words[3] = 0.0f32.to_bits(); // additive
        words[4] = 1; // num_specs_per_dom
        words[5] = u32::from(eft_mode);
        self.queue
            .write_buffer(&params, 0, bytemuck::cast_slice(&words));

        let pipeline = self.create_simple_pipeline(
            sh::CROWN_ACTIVATION_RESIDENT_TAINT_SHADER,
            "act_taint_probe",
            // a_in, err_in, ls, us, a_out, err_out, beta, ta_in, te_in, ta_out, te_out
            &[
                false, false, false, false, true, true, false, false, false, true, true,
            ],
        );

        let mut encoder = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("act_taint_encoder"),
        });
        self.pass_simple_2d(
            &mut encoder,
            &pipeline,
            &params,
            &[
                &a_in, &e_buf, &ls, &us, &a_out, &e_out, &beta_buf, &ta_in, &te_in, &ta_out,
                &te_out,
            ],
            1,
            1,
        );
        let stage = |label: &'static str| {
            create_buffer(
                dev,
                label,
                4,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            )
        };
        let st_ta = stage("act_taint_st_ta");
        let st_te = stage("act_taint_st_te");
        encoder.copy_buffer_to_buffer(&ta_out, 0, &st_ta, 0, 4);
        encoder.copy_buffer_to_buffer(&te_out, 0, &st_te, 0, 4);
        self.queue.submit(std::iter::once(encoder.finish()));

        let ta_word = WgpuDevice::read_u32_buffer(dev, &st_ta, 1)?;
        let te_word = WgpuDevice::read_u32_buffer(dev, &st_te, 1)?;
        Ok(ActivationTaintOutcome {
            taint_a: ta_word[0] != 0,
            taint_e: te_word[0] != 0,
        })
    }
}

mod gpu_tests {
    use super::*;

    fn device() -> WgpuDevice {
        WgpuDevice::new().expect("wgpu adapter for the taint-channel probe")
    }

    /// The defect, closed: a tainted coefficient times a TINY NONZERO weight
    /// keeps its taint, where the in-band magnitude sentinel loses it.
    ///
    /// This is `sentinel_taint_selfcheck`'s lane 2 in miniature: `1e10 * 1e-20`
    /// stores `1e-10`, which is below every downstream magnitude guard, while
    /// the stored `1e10` stood for a true coefficient up to `~3.4e38`.
    #[test]
    fn downscaling_does_not_launder_the_taint() {
        let dev = device();
        let out = dev
            .taint_channel_probe_1x1(SENTINEL, 1e-20, true, false)
            .expect("probe dispatch");
        let value = f32::from_bits(out.value_bits);
        assert!(
            value.abs() < SENTINEL,
            "the VALUE must launder — that is the defect being worked around, got {value:e}"
        );
        assert!(
            out.tainted,
            "the TAINT must survive a nonzero downscale: value={value:e}"
        );
    }

    /// The error-channel twin (lane 5): the `1e30` degrade marker times a
    /// `1e-25` activation slope. `1e30` is not a bound on anything — the combine
    /// writes it precisely because the true reduction is UNKNOWN and strictly
    /// larger — so scaling it is not a valid transport of that unknown.
    #[test]
    fn degrade_marker_survives_an_activation_slope() {
        let dev = device();
        let out = dev
            .taint_channel_probe_1x1(1e30, 1e-25, true, false)
            .expect("probe dispatch");
        assert!(
            out.tainted,
            "the 1e30 degrade marker must stay tainted through a nonzero slope"
        );
    }

    /// Clean exact annihilation still clears, and this is the case that rules out
    /// saturating to `±inf` instead: `R * 0 == 0` for every finite real `R`, so
    /// a dead ReLU legitimately drops the taint. Under `±inf` this would be
    /// `inf * 0 = NaN` and would degrade the row.
    #[test]
    fn exact_zero_annihilation_still_clears_the_taint() {
        let dev = device();
        let out = dev
            .taint_channel_probe_1x1(SENTINEL, 0.0, true, false)
            .expect("probe dispatch");
        let value = f32::from_bits(out.value_bits);
        assert_eq!(
            value, 0.0,
            "the product must be exactly zero, got {value:e}"
        );
        assert!(
            !out.tainted,
            "a CLEAN exact-zero partner must clear the taint — otherwise every dead \
             ReLU poisons its row"
        );
    }

    /// Symmetric: taint on the WEIGHT side propagates the same way.
    #[test]
    fn taint_propagates_from_either_operand() {
        let dev = device();
        let from_b = dev
            .taint_channel_probe_1x1(1e-20, SENTINEL, false, true)
            .expect("probe dispatch");
        assert!(from_b.tainted, "taint on b must reach the output");

        let annihilated = dev
            .taint_channel_probe_1x1(0.0, SENTINEL, false, true)
            .expect("probe dispatch");
        assert!(
            !annihilated.tainted,
            "an exact zero on a must clear b's taint symmetrically"
        );

        let untrusted_zeros = dev
            .taint_channel_probe_1x1(0.0, 0.0, true, true)
            .expect("probe dispatch");
        assert!(
            untrusted_zeros.tainted,
            "two tainted stored zeros do not authenticate an exact-zero product"
        );
    }

    /// Saturation SEEDS the channel, so nothing has to prime it, and a clean
    /// operand pair stays clean — without which the channel would be trivially
    /// satisfiable by tainting everything.
    #[test]
    fn saturation_seeds_and_clean_inputs_stay_clean() {
        let dev = device();
        let seeded = dev
            .taint_channel_probe_1x1(1e20, 1e20, false, false)
            .expect("probe dispatch");
        let value = f32::from_bits(seeded.value_bits);
        assert!(
            value.abs() >= SENTINEL,
            "the product must saturate, got {value:e}"
        );
        assert!(
            seeded.tainted,
            "an output that saturates at THIS op must seed its own taint"
        );

        let clean = dev
            .taint_channel_probe_1x1(2.0, 3.0, false, false)
            .expect("probe dispatch");
        assert_eq!(f32::from_bits(clean.value_bits), 6.0);
        assert!(!clean.tainted, "an ordinary product must stay untainted");
    }

    /// LANE-5 MODEL: a tainted error word survives an arbitrarily small NONZERO
    /// slope. This is the composition that launders the in-band 1e30 marker
    /// (1e30 * 1e-25 = 2.0e5, under every magnitude guard); the production word
    /// remains sticky.
    #[test]
    fn activation_downscale_keeps_error_taint() {
        let device = device();
        for eft in [false, true] {
            let out = device
                .activation_taint_probe_single(
                    1.0, 1e30, 1e-25, 1e-25, 0.0, eft, false, false, true,
                )
                .expect("probe");
            assert!(
                out.taint_e,
                "eft={eft}: error taint must survive a 1e-25 slope (out={out:?})"
            );
            assert!(
                !out.taint_a,
                "eft={eft}: value taint must not appear from nowhere"
            );
        }
    }

    /// LANE-4 MODEL: a dead ReLU (slope exactly 0) annihilates BOTH channels'
    /// taints — `R * 0 == 0` exactly for every finite real the sentinel stands
    /// for. Keeping taint here would be the ±inf tightness collapse.
    #[test]
    fn activation_dead_relu_clears_both_taints() {
        let device = device();
        for eft in [false, true] {
            let out = device
                .activation_taint_probe_single(1e10, 1e30, 0.0, 0.0, 0.0, eft, false, true, true)
                .expect("probe");
            assert!(
                !out.taint_a && !out.taint_e,
                "eft={eft}: exact-zero slope must clear both taints (out={out:?})"
            );
        }
    }

    /// A tainted VALUE coefficient contaminates the error word too: the error
    /// charge is computed FROM the (untrustworthy) coefficient, so its own
    /// rounding terms inherit the uncertainty.
    #[test]
    fn activation_value_taint_flows_into_error_word() {
        let device = device();
        let out = device
            .activation_taint_probe_single(1e10, 0.0, 1.0, 1.0, 0.0, false, false, true, false)
            .expect("probe");
        assert!(out.taint_a, "value taint transports through a unit slope");
        assert!(out.taint_e, "value taint must contaminate the error word");
    }

    /// CLEAN control: no incoming taint, ordinary operands — no taint appears.
    /// Without this the channel could be trivially satisfied by tainting all.
    #[test]
    fn activation_clean_stays_clean() {
        let device = device();
        for eft in [false, true] {
            let out = device
                .activation_taint_probe_single(0.5, 1e-7, 0.7, 0.7, 0.1, eft, false, false, false)
                .expect("probe");
            assert!(
                !out.taint_a && !out.taint_e,
                "eft={eft}: clean inputs must stay clean (out={out:?})"
            );
        }
    }

    /// A tainted stored sign cannot choose a zero slope and thereby erase the
    /// word when the opposite-sign slope is live. Exercise both directions and
    /// both the value and error channels in legacy and EFT modes.
    #[test]
    fn activation_asymmetric_slopes_do_not_launder_taint() {
        let device = device();
        for eft in [false, true] {
            for (lower_slope, upper_slope, is_upper) in [(0.0, 1.0, false), (1.0, 0.0, true)] {
                let value = device
                    .activation_taint_probe_single(
                        1.0,
                        0.0,
                        lower_slope,
                        upper_slope,
                        0.0,
                        eft,
                        is_upper,
                        true,
                        false,
                    )
                    .expect("value-taint probe");
                assert!(
                    value.taint_a && value.taint_e,
                    "eft={eft} is_upper={is_upper}: value taint was laundered: {value:?}"
                );

                let error = device
                    .activation_taint_probe_single(
                        1.0,
                        0.0,
                        lower_slope,
                        upper_slope,
                        0.0,
                        eft,
                        is_upper,
                        false,
                        true,
                    )
                    .expect("error-taint probe");
                assert!(
                    !error.taint_a && error.taint_e,
                    "eft={eft} is_upper={is_upper}: error taint was laundered: {error:?}"
                );
            }
        }
    }
}
