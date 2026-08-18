// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! SOUNDNESS PROOF ARTIFACT (`#daz-flush-cover-v2`): can subnormal FLUSHING be
//! soundly CHARGED by the shipped `flush` floor instead of refused?
//!
//! This box (Apple M5 Max / Metal) measurably flushes subnormals — DAZ **and**
//! FTZ, 7/15 probe lanes, including core add/multiply mismatches and an EFT
//! residual (`ops/subnormal_selfcheck.rs`). The core-lane mismatches therefore
//! still make rung 3 refuse sound-GPU authority outright; the later, narrowly
//! floor-covered exact-zero fma-residual exception does not make this adapter
//! pass. The open question was whether that refusal is *unnecessarily
//! conservative*: the
//! two AW-error combines already carry a flush-cover term
//!
//! ```text
//!   flush    = additive + flushacc · slack · F32_MIN_NORMAL
//!   flushacc = 1 + k + ‖a_i‖₁ + max_j‖w_j‖₁
//!   additive = ftz_safe_underflow_floor(k)   ( > 8·k·2^-126 )
//! ```
//!
//! so if that floor DOMINATES the flush-induced loss, FTZ/DAZ is a COST (a
//! slightly wider certified radius) rather than a BLOCKER.
//!
//! That is the formula modeled at this artifact's 2026-08-06 checkpoint. The
//! current production writers use the larger base
//! `rung3_flush_safe_additive(k) = ftz_safe_underflow_floor(9k + 4800)` and,
//! wherever the residual is subsequently multiplied by `r_slack`, round
//! `base·r_slack` outward through `rung3_flush_safe_additive_scaled`. Results
//! about the historical floor remain evidence for the base channels, but are
//! not a byte-for-byte model of the current uniform.
//!
//! The hardware oracle below models core and FMA denormal behavior separately.
//! In particular, [`mixed_policy_preserves_core_underflow_and_flushes_fma`]
//! pins the core-preserving/FMA-flushing policy seen by the current live probe,
//! and [`mixed_policy_product_residual_is_conservative_in_both_orders`] checks
//! the production `a * w` / `fma(a, w, -prod)` split against an exact residual.
//!
//! Measured on this box 2026-08-06; every ratio below is exact-rational,
//! produced by the tests in this file.
//!
//! # The answer, in three parts
//!
//! **(1) The twin's OWN RESIDUAL channels are dominated, ~34× over.** Every
//! place a flush can happen inside `GEMM_F32_EFT_TWIN_SHADER` is enumerated and
//! bounded; the worst adversary — one that drives the measured residual channel
//! to EXACTLY ZERO while the true error grows like `k·2^-126` — consumes 2.9% of
//! the shipped `flush` term. `additive > 8k·2^-126` against a demand of at most
//! `2k·2^-126`. Three structural lemmas do the work, all pinned as tests here:
//!
//! * [`residual_accumulator_can_never_flush`] — every term fed to `rsum` is a
//!   post-FTZ register value or the literal `F32_MIN_NORMAL`, i.e. lies in
//!   `{0} ∪ [2^-126, ∞)`. A running sum of such non-negatives is never a
//!   subnormal, so the residual accumulator NEVER flushes: that channel costs
//!   exactly zero and needs no charge at all.
//! * [`two_product_channel_deficit_is_below_one_min_normal`] — per tap, the
//!   uncharged part of `|ǎ·w̌ − prod|` is `< 2^-126` (measured max `0.999998`).
//! * [`two_sum_channel_deficit_is_below_one_min_normal`] — per tap, the
//!   uncharged part of `|(acc+prod) − s|` is `< 2^-126` (measured max
//!   `0.999989`). `additive > 8k·2^-126` covers both at 4× margin.
//!
//! **(2) The OPERAND-DAZ channel is covered EXACTLY, with no margin.** A
//! subnormal coefficient `a_il` is zeroed before the multiply, losing the whole
//! product; summed that is `< 2^-126·‖w_j‖₁` against a charge of
//! `‖w_j‖₁·slack·2^-126`. Measured consumption: 0.999999
//! ([`operand_daz_channel_is_covered_but_with_no_margin`]). Sound — and the
//! reason a SECOND channel of the same size is fatal.
//!
//! **(3) The SHARED propagated-error channel is NOT dominated.** The same `flush`
//! floor is the only cover for `prop = fl(err@|W|)`, and there the shipped
//! `flushacc` is short by a factor that the exact-rational oracle turns into
//! concrete non-enclosing radii — see
//! [`shipped_flush_cover_does_not_enclose_under_daz`]. Two independent
//! counterexamples, both on the `min(higham, eft)` production combinator:
//!
//! 1. SUBNORMAL WEIGHTS. DAZ zeroes the `|w_lj|` operand of the `prop` GEMM, so
//!    the propagated incoming radius `Σ_l err_il·|w_lj|` is reported as ZERO.
//!    Nothing in `flushacc` scales with `‖err_i‖₁`. Measured: the published
//!    radius is 5.86× too small at `k=16, err=100`, 58 608× too small at
//!    `err=10^6`.
//! 2. TWO `μ‖w‖₁` CHANNELS, ONE TERM. A subnormal coefficient `a_il` and a
//!    subnormal incoming radius `err_il` are two *independent* DAZ losses, each
//!    bounded by `μ·‖w_j‖₁`; `flushacc` carries `‖w_j‖₁` once. Measured:
//!    exactly 2.0× too small.
//!
//! Neither is reachable on this measured M5 path (the core subnormal lanes
//! refuse first), and on a non-flushing adapter both losses are identically
//! zero — see [`ieee_adapter_consumes_no_flush_budget`]. They are the standing
//! obligation that any future relaxation of that rung must discharge.
//!
//! # Verdict
//!
//! Flushing IS chargeable in principle — the compensated channel's own flush
//! surface costs a few percent of a floor the build already pays for. It is NOT
//! chargeable by the term the build currently ships, because that term was
//! derived for the coefficient GEMM alone and is reused verbatim as the only
//! cover for a propagated-error GEMM with two more operand-flush channels.
//! `#daz-flush-cover-v2` (`sound_consts::daz_flush_cover_w_l1` +
//! `refuse_subnormal_weight_under_daz_cover`, dark, default OFF) is the
//! corrected mechanism; [`derived_cover_plus_subnormal_weight_refusal_encloses`]
//! and [`randomised_daz_hunt_separates_the_two_covers`] are its evidence. The
//! rung-3 policy continues to refuse adapters whose core add/multiply lanes
//! flush until that mechanism is armed and the remaining sound-GPU shaders (the
//! IBP forward path, the bias/tree reductions, `crown_concretize_sound`) get the
//! same audit. Its only current relaxation is the separately charged exact-zero
//! subnormal fma-residual case; none of this file is authority to relax more.
//!
//! # Why this file is an oracle and not decoration
//!
//! Enclosure of an f32 radius cannot be decided in f32. Every finite f32 is an
//! integer multiple of `2^-149`, so a product of two f32 is an integer multiple
//! of `2^-298` and a whole dot product is carried LOSSLESSLY as a `BigInt` count
//! of `2^-298` units. Every verdict below is an exact integer comparison.
//!
//! The device model is a line-by-line transcription of the shipped WGSL, and
//! [`model_tracks_the_shipped_shader_text`] fails if the shader text it models
//! is edited out from under it.

#![allow(clippy::float_cmp, clippy::manual_is_multiple_of, clippy::identity_op)]

use num_bigint::BigInt;
use num_traits::Signed;

use super::sound_consts::{
    charged_act_bias_slack, charged_bias_slack, charged_concretize_slack, combine_slack_f32,
    eft_r_slack_f32, gamma_k_f32, rung3_flush_safe_additive, up_f32, CHARGED_ACT_BIAS_SLACK_FACTOR,
    CHARGED_BIAS_COMBINE_SLACK_FACTOR, CHARGED_CONCRETIZE_SLACK_FACTOR,
};

const F32_MIN_NORMAL: f32 = 1.1754944e-38; // 2^-126
const TWO_PROD_EXACT_FLOOR_F32: f32 = 3.9443045e-31; // 2^-101

// ---------------------------------------------------------------------------
// Exact oracle
// ---------------------------------------------------------------------------

/// Integer count of `2^-149` units — lossless, by construction.
fn f32_units(x: f32) -> BigInt {
    assert!(x.is_finite(), "oracle is defined on finite f32 only");
    let bits = x.to_bits();
    let neg = bits >> 31 == 1;
    let raw_exp = (bits >> 23) & 0xff;
    let frac = bits & 0x007f_ffff;
    let mut u = if raw_exp == 0 {
        BigInt::from(frac)
    } else {
        BigInt::from(0x0080_0000u32 + frac) << (raw_exp - 1)
    };
    if neg {
        u = -u;
    }
    u
}

/// The same f32 as an exact count of `2^-298` units (the product scale).
fn f32_prod_units(x: f32) -> BigInt {
    f32_units(x) << 149
}

fn mu_prod_units() -> BigInt {
    BigInt::from(1) << (298 - 126)
}

fn ratio(num: &BigInt, den: &BigInt) -> f64 {
    use num_traits::ToPrimitive;
    if den.sign() == num_bigint::Sign::NoSign {
        return f64::INFINITY;
    }
    num_rational::BigRational::new(num.clone(), den.clone())
        .to_f64()
        .unwrap_or(f64::NAN)
}

// ---------------------------------------------------------------------------
// Hardware model
// ---------------------------------------------------------------------------

/// Denormal behavior of the two instruction classes used by the production
/// twin. Core `+`, `-`, and `*` need not have the same FTZ/DAZ policy as FMA:
/// that mixed policy is observed on adapters where the core rung-3 lanes pass
/// while the direct-FMA lane flushes.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct Hw {
    core_ftz: bool,
    core_daz: bool,
    fma_ftz: bool,
    fma_daz: bool,
    /// Whether DAZ also applies to the `!=` comparison in the twin's exactness
    /// guard. Unspecified for MSL, so soundness is required under BOTH answers.
    daz_compare: bool,
}

const IEEE: Hw = Hw {
    core_ftz: false,
    core_daz: false,
    fma_ftz: false,
    fma_daz: false,
    daz_compare: false,
};
const METAL_CMP_DAZ: Hw = Hw {
    core_ftz: true,
    core_daz: true,
    fma_ftz: true,
    fma_daz: true,
    daz_compare: true,
};
const METAL_CMP_EXACT: Hw = Hw {
    core_ftz: true,
    core_daz: true,
    fma_ftz: true,
    fma_daz: true,
    daz_compare: false,
};
/// Core arithmetic preserves gradual underflow while FMA flushes both results
/// and operands. This deliberately models the strongest relevant mixed policy:
/// it covers both an exact residual flushed on FMA output and a subnormal
/// multiplicand flushed on FMA input.
const CORE_PRESERVE_FMA_FLUSH: Hw = Hw {
    core_ftz: false,
    core_daz: false,
    fma_ftz: true,
    fma_daz: true,
    daz_compare: false,
};

fn is_subnormal(x: f32) -> bool {
    x != 0.0 && x.is_finite() && x.abs() < F32_MIN_NORMAL
}

impl Hw {
    fn flush_result(ftz: bool, x: f32) -> f32 {
        if ftz && is_subnormal(x) {
            if x.is_sign_negative() {
                -0.0
            } else {
                0.0
            }
        } else {
            x
        }
    }
    fn flush_operand(daz: bool, x: f32) -> f32 {
        if daz && is_subnormal(x) {
            if x.is_sign_negative() {
                -0.0
            } else {
                0.0
            }
        } else {
            x
        }
    }
    fn core_r(self, x: f32) -> f32 {
        Self::flush_result(self.core_ftz, x)
    }
    fn core_o(self, x: f32) -> f32 {
        Self::flush_operand(self.core_daz, x)
    }
    fn fma_r(self, x: f32) -> f32 {
        Self::flush_result(self.fma_ftz, x)
    }
    fn fma_o(self, x: f32) -> f32 {
        Self::flush_operand(self.fma_daz, x)
    }
    fn add(self, a: f32, b: f32) -> f32 {
        self.core_r(self.core_o(a) + self.core_o(b))
    }
    fn sub(self, a: f32, b: f32) -> f32 {
        self.core_r(self.core_o(a) - self.core_o(b))
    }
    fn mul(self, a: f32, b: f32) -> f32 {
        self.core_r(self.core_o(a) * self.core_o(b))
    }
    fn fma(self, a: f32, b: f32, c: f32) -> f32 {
        self.fma_r(self.fma_o(a).mul_add(self.fma_o(b), self.fma_o(c)))
    }
    fn ne_zero(self, x: f32) -> bool {
        if self.daz_compare {
            Self::flush_operand(true, x) != 0.0
        } else {
            x != 0.0
        }
    }
}

// ---------------------------------------------------------------------------
// Transcribed device kernels
// ---------------------------------------------------------------------------

/// `GEMM_F32_EFT_TWIN_SHADER` inner loop, one output element. Tile-padded taps
/// are `(0,0)` on both sides and provably inert (`0·0` products, exact adds of
/// zero, no residual), so only the `k` real taps are executed here.
fn twin(hw: Hw, a: &[f32], w: &[f32]) -> (f32, f32) {
    let mut acc = 0.0f32;
    let mut rsum = 0.0f32;
    for i in 0..a.len() {
        let (ai, wi) = (a[i], w[i]);
        let prod = hw.mul(ai, wi);
        let ep = hw.fma(ai, wi, -prod);
        let mut eterm = ep.abs();
        if hw.ne_zero(ai) && hw.ne_zero(wi) && prod.abs() < TWO_PROD_EXACT_FLOOR_F32 {
            eterm = F32_MIN_NORMAL;
        }
        let s = hw.add(acc, prod);
        let bb = hw.fma(-1.0, acc, s);
        let sb = hw.fma(-1.0, bb, s);
        let da = hw.fma(-1.0, sb, acc);
        let db = hw.fma(-1.0, bb, prod);
        let es = hw.add(da, db);
        rsum = hw.add(hw.add(rsum, eterm), es.abs());
        acc = s;
    }
    (acc, rsum)
}

/// Any f32 GEMM accumulation (`GEMM_F32_SHADER` shape). Used for the shipped
/// `value = fl(A@W)`, for `s_prod = fl(|A|@|W|)`, for `prop = fl(err@|W|)`, and
/// for the `n=1` row-L1 reductions `row_abs_a = fl(|A|@ones)`.
fn gemm_dot(hw: Hw, x: &[f32], y: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..x.len() {
        acc = hw.fma(x[i], y[i], acc);
    }
    acc
}

fn round_up_pos(x: f32) -> f32 {
    let bits = x.to_bits();
    let magnitude = bits & 0x7fff_ffff;
    if magnitude >= 0x7f80_0000 {
        return x;
    }
    if (bits & 0x8000_0000) != 0 || magnitude == 0 {
        return 0.0;
    }
    if magnitude < 0x0080_0000 {
        return F32_MIN_NORMAL;
    }
    f32::from_bits(bits + 1)
}

/// Which `flushacc` formula to evaluate.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum FlushForm {
    /// Historical shipped-checkpoint form: `1 + k + ‖a_i‖₁ + max_j‖w_j‖₁`.
    Shipped,
    /// `#daz-flush-cover-v2`: `1 + k + ‖a_i‖₁ + 4·max_j‖w_j‖₁`, i.e. exactly
    /// what `sound_consts::daz_flush_cover_w_l1` uploads when the gate is on.
    /// Sound only together with the subnormal-weight refusal
    /// (`refuse_subnormal_weight_under_daz_cover`), which removes the
    /// `μ‖err_i‖₁` channel this term still does not carry.
    DerivedV2,
}

struct Row {
    a: Vec<f32>,
    w: Vec<f32>,
    err: Vec<f32>,
}

struct Published {
    /// `min(err_higham, err_eft)` — the production combinator.
    e_min: f32,
    e_eft: f32,
    e_higham: f32,
    flush: f32,
    /// pre-flush f32 core of the EFT arm.
    core: f32,
    r_dev: f32,
}

/// Evaluate both AW-error combines exactly as shipped at this oracle's
/// 2026-08-06 checkpoint. The current rung-3 additive is strictly larger.
fn publish(hw: Hw, form: FlushForm, row: &Row) -> Published {
    let k = row.a.len();
    let ones = vec![1.0f32; k];
    let abs_a: Vec<f32> = row.a.iter().map(|x| x.abs()).collect();
    let abs_w: Vec<f32> = row.w.iter().map(|x| x.abs()).collect();

    let (v, r) = twin(hw, &row.a, &row.w);
    let value = gemm_dot(hw, &row.a, &row.w);
    let s_prod = gemm_dot(hw, &abs_a, &abs_w);
    let prop = gemm_dot(hw, &row.err, &abs_w);
    let row_abs_a = gemm_dot(hw, &abs_a, &ones);

    // host uniforms
    let slack = combine_slack_f32(k).expect("finite Higham regime");
    let r_slack = eft_r_slack_f32(k).expect("finite Higham regime");
    let gamma = gamma_k_f32(k).expect("finite Higham regime");
    let additive = ny_core::ftz_safe_underflow_floor(u32::try_from(k).unwrap());
    let w_l1_exact = up_f32(row.w.iter().map(|x| f64::from(x.abs())).sum::<f64>());
    let w_l1_max = match form {
        FlushForm::Shipped => w_l1_exact,
        FlushForm::DerivedV2 => w_l1_exact * 4.0,
    };

    // flush term, as both shaders compute it
    let mut flushacc = hw.add(hw.add(1.0, k as f32), w_l1_max);
    flushacc = hw.add(flushacc, row_abs_a);
    let flush = hw.add(additive, hw.mul(hw.mul(flushacc, slack), F32_MIN_NORMAL));

    // CROWN_AW_ERROR_COMBINE_SHADER
    let core_h = hw.mul(hw.add(hw.mul(gamma, s_prod), prop), slack);
    let e_higham = round_up_pos(hw.add(core_h, flush));

    // CROWN_EFT_MIN_COMBINE_SHADER
    let d = hw.sub(v, value).abs();
    let core = hw.add(hw.mul(hw.add(r, d), r_slack), hw.mul(prop, slack));
    let e_eft = round_up_pos(hw.add(core, flush));

    Published {
        e_min: e_eft.min(e_higham),
        e_eft,
        e_higham,
        flush,
        core,
        r_dev: r,
    }
}

/// The EXACT certified claim for one output element: for every admissible
/// `Â` with `|Â − A| ≤ err` elementwise,
/// `|Â@W − value| ≤ |A@W − value| + Σ_l err_l·|w_l|`, all in `2^-298` units.
fn exact_required_radius(hw: Hw, row: &Row) -> BigInt {
    let value = gemm_dot(hw, &row.a, &row.w);
    let mut dot = BigInt::from(0);
    for i in 0..row.a.len() {
        dot += f32_units(row.a[i]) * f32_units(row.w[i]);
    }
    let mut prop = BigInt::from(0);
    for i in 0..row.a.len() {
        prop += f32_units(row.err[i]) * f32_units(row.w[i].abs());
    }
    (dot - f32_prod_units(value)).abs() + prop
}

/// `(encloses, budget_consumed)` where `budget_consumed` is the EXACT ratio of
/// what the flush term had to absorb to what it supplies. `> 1` ⇒ the term does
/// not dominate.
fn assess(hw: Hw, form: FlushForm, row: &Row) -> (bool, f64, Published) {
    let pub_ = publish(hw, form, row);
    let need = exact_required_radius(hw, row);
    let encloses = need <= f32_prod_units(pub_.e_min);
    let consumed = &need - f32_prod_units(pub_.core);
    let used = if consumed.sign() == num_bigint::Sign::Minus {
        0.0
    } else {
        ratio(&consumed, &f32_prod_units(pub_.flush))
    };
    (encloses, used, pub_)
}

// ---------------------------------------------------------------------------
// deterministic RNG + f32 constructors
// ---------------------------------------------------------------------------

struct Rng(u64);
impl Rng {
    fn u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 32) as u32
    }
}

/// f32 equal to `m · 2^(e−23)` for a 24-bit normalized `m`.
fn mant_exp(m: u64, e: i32) -> f32 {
    assert!((1 << 23..1 << 24).contains(&m));
    let v = (m as f64) * 2f64.powi(e - 23);
    let f = v as f32;
    assert_eq!(f64::from(f), v, "mant_exp({m},{e}) is not exact");
    f
}

/// Post-FTZ register domain: normal or zero, never subnormal.
fn register_f32(rng: &mut Rng, min_exp: i32, max_exp: i32) -> f32 {
    if rng.u32() % 16 == 0 {
        return 0.0;
    }
    let e = min_exp + (rng.u32() % ((max_exp - min_exp + 1) as u32)) as i32;
    let m = (1u64 << 23) | u64::from(rng.u32() & 0x7f_ffff);
    let v = mant_exp(m, e);
    if rng.u32() % 2 == 0 {
        v
    } else {
        -v
    }
}

fn largest_subnormal() -> f32 {
    f32::from_bits(0x007f_ffff)
}

// ---------------------------------------------------------------------------
// Mixed core-preserving / FMA-flushing policy
// ---------------------------------------------------------------------------

#[test]
fn mixed_policy_preserves_core_underflow_and_flushes_fma() {
    let hw = CORE_PRESERVE_FMA_FLUSH;
    let sub = largest_subnormal();
    let half_min_normal = f32::from_bits(0x0040_0000);

    // Core input and result handling both preserve gradual underflow.
    assert_eq!(hw.add(sub, 0.0).to_bits(), sub.to_bits());
    assert_eq!(hw.mul(sub, 1.0).to_bits(), sub.to_bits());
    assert_eq!(
        hw.mul(F32_MIN_NORMAL, 0.5).to_bits(),
        half_min_normal.to_bits()
    );

    // FMA DAZ loses a subnormal operand, and FMA FTZ loses a subnormal result.
    assert_eq!(hw.fma(sub, 1.0, 0.0), 0.0);
    assert_eq!(hw.fma(F32_MIN_NORMAL, 0.5, 0.0), 0.0);
}

#[test]
fn mixed_policy_product_residual_is_conservative_in_both_orders() {
    let hw = CORE_PRESERVE_FMA_FLUSH;
    let subnormal = f32::from_bits(0x0040_0001);
    let large = f32::from_bits(0x6eff_ffff);
    let expected_prod = f32::from_bits(0x2f80_0001);
    let expected_residual = f32::from_bits(0x237f_fffc);

    for (a, w) in [(subnormal, large), (large, subnormal)] {
        let prod = hw.mul(a, w);
        assert_eq!(prod.to_bits(), expected_prod.to_bits());

        let exact_residual = (f32_units(a) * f32_units(w) - f32_prod_units(prod)).abs();
        assert_eq!(
            exact_residual,
            f32_prod_units(expected_residual),
            "the exact discriminator changed"
        );

        // FMA DAZ turns one multiplicand into zero, so `ep` becomes `-prod`.
        // This is deliberately much larger than the true residual, but it is
        // conservative and therefore safe for the published error radius.
        let ep = hw.fma(a, w, -prod);
        assert_eq!(ep.to_bits(), (-prod).to_bits());
        assert!(
            f32_prod_units(ep.abs()) >= exact_residual,
            "mixed-policy residual undercharged for a={:#010x}, w={:#010x}",
            a.to_bits(),
            w.to_bits()
        );
        assert!(!two_product_deficit(hw, a, w).is_positive());
    }
}

// ---------------------------------------------------------------------------
// Lemma 1 — the residual accumulator never flushes
// ---------------------------------------------------------------------------

/// Every quantity added into the twin's `rsum` is either a post-FTZ register
/// value (`{0} ∪ [2^-126, ∞)`) or the literal `F32_MIN_NORMAL`. A running sum of
/// non-negatives drawn from that set is never subnormal, so the residual
/// accumulator has NO flush channel and needs no charge — this is what removes
/// the `2k·2^-126` term a naive count would demand.
#[test]
fn residual_accumulator_can_never_flush() {
    let hw = METAL_CMP_DAZ;
    let mut rng = Rng(0x00ab_cdef);
    let mut checked = 0usize;
    for _ in 0..200_000 {
        let terms = [
            register_f32(&mut rng, -126, 30).abs(),
            F32_MIN_NORMAL,
            register_f32(&mut rng, -126, 30).abs(),
            0.0,
        ];
        let mut r = 0.0f32;
        for t in terms {
            let raw = hw.core_o(r) + hw.core_o(t);
            checked += 1;
            assert!(
                !is_subnormal(raw),
                "rsum partial {raw:e} is subnormal (r={r:e}, t={t:e}); the \
                 no-flush lemma for the residual accumulator is broken"
            );
            r = hw.add(r, t);
        }
    }
    assert!(checked >= 800_000);
}

// ---------------------------------------------------------------------------
// Lemma 2/3 — per-tap deficits are strictly below one F32_MIN_NORMAL
// ---------------------------------------------------------------------------

/// Uncharged part of the TwoProduct residual for one tap, in `2^-298` units.
fn two_product_deficit(hw: Hw, a: f32, w: f32) -> BigInt {
    let prod = hw.mul(a, w);
    let ep = hw.fma(a, w, -prod);
    let mut eterm = ep.abs();
    if hw.ne_zero(a) && hw.ne_zero(w) && prod.abs() < TWO_PROD_EXACT_FLOOR_F32 {
        eterm = F32_MIN_NORMAL;
    }
    let rho = f32_units(hw.core_o(a)) * f32_units(hw.core_o(w)) - f32_prod_units(prod);
    rho.abs() - f32_prod_units(eterm)
}

/// Uncharged part of the fma-barrier TwoSum residual for one tap, `2^-149` units.
fn two_sum_deficit(hw: Hw, acc: f32, prod: f32) -> BigInt {
    let s = hw.add(acc, prod);
    let bb = hw.fma(-1.0, acc, s);
    let sb = hw.fma(-1.0, bb, s);
    let da = hw.fma(-1.0, sb, acc);
    let db = hw.fma(-1.0, bb, prod);
    let es = hw.add(da, db);
    let sigma = f32_units(hw.core_o(acc)) + f32_units(hw.core_o(prod)) - f32_units(s);
    sigma.abs() - f32_units(es).abs()
}

#[test]
fn two_product_channel_deficit_is_below_one_min_normal() {
    let mu = mu_prod_units();
    for hw in [METAL_CMP_DAZ, METAL_CMP_EXACT, CORE_PRESERVE_FMA_FLUSH] {
        let mut rng = Rng(0xfeed_1234);
        let mut worst = BigInt::from(-1);
        // Products just ABOVE the 2^-101 exactness guard are the dangerous band:
        // the guard does not fire, yet the exact residual can still be subnormal
        // and be silently zeroed.
        for ea in -80..=-40i32 {
            for t in 0..4_000 {
                let ma = (1u64 << 23) | u64::from(rng.u32() & 0x7f_ffff);
                let mw = (1u64 << 23) | u64::from(rng.u32() & 0x7f_ffff);
                let ew = -101 - ea;
                let a = mant_exp(ma, ea);
                let a = if t % 2 == 0 { a } else { -a };
                let d = two_product_deficit(hw, a, mant_exp(mw, ew));
                if d > worst {
                    worst = d;
                }
            }
        }
        // subnormal operands: the DAZ loss must land in the ‖a‖₁/‖w‖₁ terms, not
        // here, so the residual deficit must be exactly zero.
        for _ in 0..100_000 {
            let sub = f32::from_bits(1 + (rng.u32() % 0x0080_0000));
            let other = register_f32(&mut rng, -126, 60);
            for d in [
                two_product_deficit(hw, sub, other),
                two_product_deficit(hw, other, sub),
            ] {
                if d > worst {
                    worst = d;
                }
            }
        }
        for _ in 0..400_000 {
            let d = two_product_deficit(
                hw,
                register_f32(&mut rng, -126, 60),
                register_f32(&mut rng, -126, 60),
            );
            if d > worst {
                worst = d;
            }
        }
        assert!(
            worst < mu,
            "{hw:?}: TwoProduct deficit reached {} of 2^-126; the per-tap \
             charge model assumes < 1",
            ratio(&worst, &mu)
        );
    }
    // On a non-flushing adapter the EFT identity is exact: deficit is 0.
    let mut rng = Rng(7);
    for _ in 0..100_000 {
        let d = two_product_deficit(
            IEEE,
            register_f32(&mut rng, -126, 60),
            register_f32(&mut rng, -126, 60),
        );
        assert!(!d.is_positive(), "IEEE adapter must have zero deficit");
    }
}

#[test]
fn two_sum_channel_deficit_is_below_one_min_normal() {
    let mu149 = BigInt::from(1) << (149 - 126);
    for hw in [METAL_CMP_DAZ, METAL_CMP_EXACT, CORE_PRESERVE_FMA_FLUSH] {
        let mut rng = Rng(0x2026_0806);
        let mut worst = BigInt::from(-1);
        // Near-underflow band: this is where the ideal TwoSum residual is
        // subnormal and the fma-barrier chain can silently zero it.
        for ea in -126..=-95i32 {
            for ep in -126..=-95i32 {
                for t in 0..40 {
                    let ma = (1u64 << 23) | u64::from(rng.u32() & 0x7f_ffff);
                    let mp = (1u64 << 23) | u64::from(rng.u32() & 0x7f_ffff);
                    let mut acc = mant_exp(ma, ea);
                    let mut prod = mant_exp(mp, ep);
                    if t % 2 == 0 {
                        prod = -prod;
                    }
                    if t % 4 == 3 {
                        acc = -acc;
                    }
                    let d = two_sum_deficit(hw, acc, prod);
                    if d > worst {
                        worst = d;
                    }
                }
            }
        }
        // near-cancelling pairs across the whole normal range
        for _ in 0..400_000 {
            let acc = register_f32(&mut rng, -126, 40);
            let prod = f32::from_bits((-acc).to_bits() ^ (rng.u32() % 64));
            if !prod.is_finite() || is_subnormal(prod) {
                continue;
            }
            let d = two_sum_deficit(hw, acc, prod);
            if d > worst {
                worst = d;
            }
        }
        for _ in 0..400_000 {
            let d = two_sum_deficit(
                hw,
                register_f32(&mut rng, -126, 100),
                register_f32(&mut rng, -126, 100),
            );
            if d > worst {
                worst = d;
            }
        }
        assert!(
            worst < mu149,
            "{hw:?}: TwoSum deficit reached {} of 2^-126",
            ratio(&worst, &mu149)
        );
    }
}

// ---------------------------------------------------------------------------
// Adversarial corpora
// ---------------------------------------------------------------------------

/// Alternating taps whose f32 products cancel EXACTLY (so the accumulator
/// returns to zero every second tap and every TwoSum residual is exactly zero ⇒
/// the measured residual channel reports `R = 0`), while every EXACT product
/// carries a SUBNORMAL TwoProduct residual of the SAME SIGN just under `2^-126`.
/// On an FTZ adapter all of them are silently zeroed, so the true error grows
/// like `k·2^-126` against a residual channel that reports nothing.
fn ep_starvation(k: usize) -> Row {
    assert_eq!(k % 2, 0);
    let p: u64 = (1u64 << 23) << 24; // prod = 2^-100 exactly
    let window: u64 = 1 << 21; // |ep| < 2^21·2^-147 = 2^-126
    let (mut ta, mut tb) = (None, None);
    for ma in (1u64 << 23)..(1u64 << 24) {
        if ta.is_none() {
            let mw = p.div_ceil(ma);
            if (1 << 23..1 << 24).contains(&mw) && ma * mw > p && ma * mw - p < window {
                ta = Some((ma, mw));
            }
        }
        if tb.is_none() {
            let mw = p / ma;
            if (1 << 23..1 << 24).contains(&mw) && ma * mw < p && p - ma * mw < window {
                tb = Some((ma, mw));
            }
        }
        if ta.is_some() && tb.is_some() {
            break;
        }
    }
    let (maa, maw) = ta.expect("round-down tap");
    let (mba, mbw) = tb.expect("round-up tap");
    let (mut a, mut w) = (Vec::with_capacity(k), Vec::with_capacity(k));
    for i in 0..k {
        if i % 2 == 0 {
            a.push(mant_exp(maa, -50));
            w.push(mant_exp(maw, -51));
        } else {
            a.push(-mant_exp(mba, -50));
            w.push(mant_exp(mbw, -51));
        }
    }
    Row {
        err: vec![0.0; k],
        a,
        w,
    }
}

/// DAZ operand amplification: subnormal coefficients against large weights, so
/// the whole product `|a|·|w|` vanishes from BOTH the value and the residual
/// channel. Charged only by the `max_j‖w_j‖₁` half of the shipped floor.
fn daz_amplification(k: usize, w_mag: f32) -> Row {
    Row {
        a: vec![largest_subnormal(); k],
        w: vec![w_mag; k],
        err: vec![0.0; k],
    }
}

/// COUNTEREXAMPLE 1 — subnormal weights kill the propagated-error channel.
fn prop_subnormal_weight(k: usize, err_mag: f32) -> Row {
    Row {
        a: vec![0.0; k],
        w: vec![largest_subnormal(); k],
        err: vec![err_mag; k],
    }
}

/// COUNTEREXAMPLE 2 — two independent `μ‖w‖₁` channels, one `‖w‖₁` term.
fn double_daz(k: usize, w_mag: f32) -> Row {
    Row {
        a: vec![largest_subnormal(); k],
        w: vec![w_mag; k],
        err: vec![largest_subnormal(); k],
    }
}

/// The twin's OWN residual channels: FTZ inside `TwoProduct`/`TwoSum`/`rsum`.
/// These are the channels `additive = ftz_safe_underflow_floor(k) > 8k·2^-126`
/// is supposed to pay for.
fn residual_corpus() -> Vec<(String, Row)> {
    (0..4)
        .map(|i| {
            let k = [2usize, 16, 256, 4096][i];
            (format!("ep-starvation k={k}"), ep_starvation(k))
        })
        .collect()
}

/// The OPERAND-DAZ channel: a subnormal `a_il` zeroed before the multiply, so
/// the whole product `|a|·|w|` vanishes. Paid for by the `max_j‖w_j‖₁` half of
/// `flushacc` — and by nothing else, which is why it is EXACTLY tight.
fn operand_daz_corpus() -> Vec<(String, Row)> {
    let mut out = Vec::new();
    for k in [16usize, 1024] {
        for m in [1.0f32, 1.0e30] {
            out.push((
                format!("daz-amplification k={k} w={m:e}"),
                daz_amplification(k, m),
            ));
        }
    }
    out
}

fn twin_only_corpus() -> Vec<(String, Row)> {
    residual_corpus()
        .into_iter()
        .chain(operand_daz_corpus())
        .collect()
}

fn prop_corpus() -> Vec<(String, Row)> {
    let mut out = Vec::new();
    for k in [16usize, 1024] {
        for e in [100.0f32, 1.0e6] {
            out.push((
                format!("prop-subnormal-weight k={k} err={e:e}"),
                prop_subnormal_weight(k, e),
            ));
        }
        out.push((format!("double-daz k={k}"), double_daz(k, 1.0e30)));
    }
    out
}

// ---------------------------------------------------------------------------
// THE RESULT — half one: the twin's own channels ARE chargeable
// ---------------------------------------------------------------------------

/// POSITIVE HALF, part 1. The twin's OWN residual channels — including the
/// exact-zero subnormal residual case that the current rung-3 policy admits only
/// with a floor — are dominated even by the historical SHIPPED floor with an
/// order of magnitude to spare, even on an adversary that drives the measured
/// residual channel to EXACTLY ZERO while the true error grows like `k·2^-126`.
/// For these channels, flushing is a COST, not a blocker.
#[test]
fn twin_residual_flush_channels_are_dominated_by_the_shipped_cover() {
    let mut worst = 0.0f64;
    for hw in [METAL_CMP_DAZ, METAL_CMP_EXACT, CORE_PRESERVE_FMA_FLUSH] {
        for (name, row) in residual_corpus() {
            let (encloses, used, p) = assess(hw, FlushForm::Shipped, &row);
            assert!(
                encloses,
                "{hw:?} {name}: shipped radius {:e} does not enclose (R_dev={:e})",
                p.e_min, p.r_dev
            );
            worst = worst.max(used);
        }
    }
    // Measured 2026-08-06: 0.0294 — `additive > 8k·2^-126` against a demand of
    // at most `2k·2^-126` (one per-tap TwoProduct deficit + one TwoSum deficit,
    // each proven < 2^-126 above; the rsum channel is provably free).
    assert!(
        worst < 0.10,
        "residual-channel flush budget consumption rose to {worst:.6}; the two \
         per-tap deficit lemmas bound the demand at 2k·2^-126 against an \
         additive of >8k·2^-126, so anything near 1 means a channel is missing \
         from the derivation"
    );
}

/// POSITIVE HALF, part 2 — with a WARNING attached. The OPERAND-DAZ channel is
/// also covered by the shipped floor, but only just: the loss is
/// `Σ_l |a_il|·|w_lj| < 2^-126·‖w_j‖₁` and the charge is `‖w_j‖₁·slack·2^-126`,
/// so the ENTIRE margin is the `(1 − 2^-23)` by which the largest subnormal
/// falls short of `2^-126`, minus whatever the `flushacc` chain's own four f32
/// roundings eat. Measured consumption: 0.999999. It is sound, and it has no
/// engineering margin at all — which is exactly why a SECOND channel of the
/// same size (`prop`'s `err` operand, below) breaks it.
#[test]
fn operand_daz_channel_is_covered_but_with_no_margin() {
    let mut worst: f64 = 0.0;
    for hw in [METAL_CMP_DAZ, METAL_CMP_EXACT] {
        for (name, row) in operand_daz_corpus() {
            let (encloses, used, _) = assess(hw, FlushForm::Shipped, &row);
            assert!(encloses, "{hw:?} {name}: shipped cover fails to enclose");
            worst = worst.max(used);
        }
    }
    assert!(
        worst > 0.99,
        "operand-DAZ consumption fell to {worst:.6}; this test exists to record \
         that the channel is EXACTLY tight, so a drop means the term changed"
    );
    assert!(
        worst < 1.0,
        "operand-DAZ consumption reached {worst:.6} >= 1"
    );
    // The derived cover restores a real margin on the same rows.
    for hw in [METAL_CMP_DAZ, METAL_CMP_EXACT] {
        for (name, row) in operand_daz_corpus() {
            let (encloses, used, _) = assess(hw, FlushForm::DerivedV2, &row);
            assert!(encloses, "{hw:?} {name}: derived cover fails to enclose");
            assert!(
                used < 0.30,
                "{name}: derived cover consumption {used:.6} — expected ~0.25"
            );
        }
    }
}

/// The `R_dev = 0` starvation case is not hypothetical arithmetic: the twin
/// really does report a ZERO residual while the exact error is `≈ k·2^-126`.
/// If this ever stops holding the adversary above has gone stale.
#[test]
fn residual_starvation_really_starves_the_measured_channel() {
    let row = ep_starvation(256);
    let (v, r) = twin(METAL_CMP_DAZ, &row.a, &row.w);
    assert_eq!(r, 0.0, "residual channel should report exactly zero");
    let need = exact_required_radius(METAL_CMP_DAZ, &row);
    let in_mu = ratio(&need, &mu_prod_units());
    assert!(
        in_mu > 100.0,
        "starvation case should leave >100·2^-126 of true error uncharged by \
         the residual channel, got {in_mu}"
    );
    let (v_ieee, r_ieee) = twin(IEEE, &row.a, &row.w);
    assert_eq!(v, v_ieee, "twin VALUE must not depend on flushing here");
    assert!(
        r_ieee > 0.0,
        "on a non-flushing adapter the same residuals are captured"
    );
}

// ---------------------------------------------------------------------------
// THE RESULT — half two: the SHARED propagated-error channel is NOT chargeable
// ---------------------------------------------------------------------------

/// THE OBLIGATION. On a DAZ adapter the SHIPPED flush cover does not enclose:
/// two independent counterexamples, decided in exact rational arithmetic, break
/// BOTH arms and therefore the `min` the production combinator publishes.
/// This is why rung 3 may NOT be relaxed to admit DAZ/core-lane flushing on the
/// strength of the historical flush term alone.
#[test]
fn shipped_flush_cover_does_not_enclose_under_daz() {
    let mut broke = 0usize;
    let mut worst = 0.0f64;
    for hw in [METAL_CMP_DAZ, METAL_CMP_EXACT] {
        for (name, row) in prop_corpus() {
            let (encloses, used, p) = assess(hw, FlushForm::Shipped, &row);
            if !encloses {
                broke += 1;
                worst = worst.max(used);
                // both arms must be broken, else `min` would have rescued it
                let need = exact_required_radius(hw, &row);
                assert!(
                    need > f32_prod_units(p.e_higham),
                    "{name}: Higham arm unexpectedly encloses"
                );
                assert!(
                    need > f32_prod_units(p.e_eft),
                    "{name}: EFT arm unexpectedly encloses"
                );
            }
        }
    }
    assert!(
        broke >= 6,
        "the propagated-error counterexamples stopped reproducing ({broke} \
         non-enclosing rows). Either the flush term was fixed — in which case \
         update this test and the authority-ladder note — or the model drifted."
    );
    assert!(
        worst > 5.0,
        "worst shipped-cover shortfall fell to {worst:.3}x; expected ~5.9x or \
         more from the subnormal-weight counterexample"
    );
}

/// `#daz-flush-cover-v2`: the widened `w_l1_max` uploaded by
/// `sound_consts::daz_flush_cover_w_l1` (4×) TOGETHER WITH the subnormal-weight
/// refusal of `refuse_subnormal_weight_under_daz_cover` encloses the entire
/// corpus, with ~2× of measured margin.
#[test]
fn derived_cover_plus_subnormal_weight_refusal_encloses() {
    let mut worst = 0.0f64;
    let mut evaluated = 0usize;
    for hw in [METAL_CMP_DAZ, METAL_CMP_EXACT] {
        for (name, row) in twin_only_corpus().into_iter().chain(prop_corpus()) {
            // the host guard refuses exactly the rows whose uncovered channel is
            // the `μ‖err_i‖₁` one
            if row.w.iter().any(|x| is_subnormal(*x)) {
                continue;
            }
            evaluated += 1;
            let (encloses, used, p) = assess(hw, FlushForm::DerivedV2, &row);
            assert!(
                encloses,
                "{hw:?} {name}: derived cover fails to enclose (published {:e})",
                p.e_min
            );
            worst = worst.max(used);
        }
    }
    assert!(evaluated >= 16, "corpus shrank to {evaluated} rows");
    assert!(
        worst < 0.75,
        "derived cover budget consumption {worst:.6} left less than 25% margin"
    );
}

/// Randomised hunt over rows built specifically to trip DAZ: this is what turns
/// the two hand-built counterexamples into a population statement, and what
/// would catch a THIRD channel neither of them models.
#[test]
fn randomised_daz_hunt_separates_the_two_covers() {
    let hw = METAL_CMP_DAZ;
    let mut rng = Rng(0x1234_5678_9abc);
    let (mut shipped_fail, mut derived_fail, mut refused) = (0usize, 0usize, 0usize);
    let mut worst_derived = 0.0f64;
    let rows = 20_000;
    for _ in 0..rows {
        let k = 1 + (rng.u32() % 32) as usize;
        let we = -20 + (rng.u32() % 60) as i32;
        let mut row = Row {
            a: Vec::new(),
            w: Vec::new(),
            err: Vec::new(),
        };
        for _ in 0..k {
            row.a.push(if rng.u32() % 4 == 0 {
                register_f32(&mut rng, -126, 10)
            } else {
                f32::from_bits(1 + (rng.u32() % 0x0080_0000))
            });
            row.w.push(if rng.u32() % 8 == 0 {
                f32::from_bits(1 + (rng.u32() % 0x0080_0000))
            } else {
                let m = (1u64 << 23) | u64::from(rng.u32() & 0x7f_ffff);
                let v = mant_exp(m, we);
                if rng.u32() % 2 == 0 {
                    v
                } else {
                    -v
                }
            });
            row.err.push(match rng.u32() % 4 {
                0 => 0.0,
                1 => f32::from_bits(1 + (rng.u32() % 0x0080_0000)),
                2 => 1.0,
                _ => mant_exp(
                    (1u64 << 23) | u64::from(rng.u32() & 0x7f_ffff),
                    -10 + (rng.u32() % 30) as i32,
                ),
            });
        }
        if !assess(hw, FlushForm::Shipped, &row).0 {
            shipped_fail += 1;
        }
        if row.w.iter().any(|x| is_subnormal(*x)) {
            refused += 1;
            continue;
        }
        let (encloses, used, _) = assess(hw, FlushForm::DerivedV2, &row);
        if !encloses {
            derived_fail += 1;
        }
        worst_derived = worst_derived.max(used);
    }
    assert!(
        shipped_fail > 0,
        "the hunt found no shipped-cover failure in {rows} rows; it has lost \
         its discriminating power"
    );
    assert_eq!(
        derived_fail, 0,
        "derived cover + subnormal-weight refusal failed on {derived_fail} of \
         {rows} rows (refused {refused})"
    );
    assert!(
        worst_derived < 0.75,
        "derived cover consumption reached {worst_derived:.6}"
    );
}

/// CONTROL. On a non-flushing adapter every one of these channels is identically
/// zero: the flush term absorbs nothing and both covers enclose. This is what
/// makes `#daz-flush-cover-v2` a no-op for the modeled gradual-underflow policy
/// and confirms the failures above are caused by flushing and nothing else.
#[test]
fn ieee_adapter_consumes_no_flush_budget() {
    for form in [FlushForm::Shipped, FlushForm::DerivedV2] {
        for (name, row) in twin_only_corpus().into_iter().chain(prop_corpus()) {
            let (encloses, used, _) = assess(IEEE, form, &row);
            assert!(encloses, "{form:?} {name}: IEEE adapter must enclose");
            assert_eq!(used, 0.0, "{form:?} {name}: IEEE consumed {used} of flush");
        }
    }
}

// ---------------------------------------------------------------------------
// #flush-charge — charged-Metal mode extensions (lane M, 2026-08-12)
// ---------------------------------------------------------------------------
//
// The rung-3 probe model pin, the concretize legacy-branch transcription, and
// the bias-combine legacy-arm transcription. These derive (and pin) the
// charged-mode widening factors `CHARGED_CONCRETIZE_SLACK_FACTOR` and
// `CHARGED_BIAS_COMBINE_SLACK_FACTOR` consumed by
// `sound_authority::FlushChargePolicy`, plus the negative results that justify
// the policy's refusals (subnormal inputs / subnormal bias).

/// The rung-3 probe kernel (`ops/subnormal_selfcheck.rs`), transcribed under
/// this module's hardware model: `(a+b, a·b, TwoSum residual, |TwoProduct
/// residual|)` bit patterns.
fn probe_lanes(hw: Hw, a_bits: u32, b_bits: u32) -> [u32; 4] {
    let a = f32::from_bits(a_bits);
    let b = f32::from_bits(b_bits);
    let s = hw.add(a, b);
    let bb = hw.fma(-1.0, a, s);
    let sb = hw.fma(-1.0, bb, s);
    let da = hw.fma(-1.0, sb, a);
    let db = hw.fma(-1.0, bb, b);
    let t = hw.add(da, db);
    let prod = hw.mul(a, b);
    let ep = hw.fma(a, b, -prod).abs();
    [s.to_bits(), prod.to_bits(), t.to_bits(), ep.to_bits()]
}

/// ANTI-DRIFT PIN between the two flush models: the production PURE-FLUSH
/// classifier twin (`subnormal_selfcheck::pure_flush_expectations`) must agree
/// bit-exactly with this module's `METAL_CMP_DAZ` hardware on every probe
/// lane. The classifier's admission is therefore exactly "the hardware this
/// oracle derived the charges for" — and the measured Apple M5 Max table is
/// pinned equal to the classifier twin in
/// `subnormal_selfcheck::cpu_tests::measured_m5_table_is_pure_flush_and_matches_the_model_exactly`,
/// closing the triangle hardware == classifier twin == oracle model.
#[test]
fn pure_flush_classifier_twin_matches_the_oracle_hardware_model() {
    let twin = super::ops::subnormal_selfcheck::pure_flush_expectations();
    let mut model = Vec::with_capacity(twin.len());
    for &(a, b) in &super::ops::subnormal_selfcheck::PAIRS {
        model.extend_from_slice(&probe_lanes(METAL_CMP_DAZ, a, b));
    }
    assert_eq!(
        twin, model,
        "the production pure-flush twin and the oracle METAL_CMP_DAZ model \
         disagree; the charge derivation no longer covers the classifier's \
         admission"
    );
    // The IEEE control: with no flushing the probe model reproduces the exact
    // CPU expectations, so the classifier's Conformant arm is the same lane
    // table the rung-3 gate compares against.
    let ieee_expected = super::ops::subnormal_selfcheck::cpu_expectations();
    let mut ieee_model = Vec::with_capacity(ieee_expected.len());
    for &(a, b) in &super::ops::subnormal_selfcheck::PAIRS {
        ieee_model.extend_from_slice(&probe_lanes(IEEE, a, b));
    }
    assert_eq!(ieee_expected, ieee_model);
}

// ---------------------------------------------------------------------------
// #flush-charge §F — CROWN_CONCRETIZE_SOUND_SHADER, legacy branch (eft_mode=0)
// ---------------------------------------------------------------------------

/// Outward helpers of the concretize final assembly — bit manipulation only
/// (deliberately flush-immune in the shipped WGSL; transcribed verbatim).
fn next_down_f32_normal(x: f32) -> f32 {
    let bits = x.to_bits();
    let magnitude = bits & 0x7fff_ffff;
    let negative = bits & 0x8000_0000 != 0;
    if magnitude >= 0x7f80_0000 {
        return x;
    }
    if magnitude == 0 {
        return -F32_MIN_NORMAL;
    }
    if magnitude < 0x0080_0000 {
        return if negative { -F32_MIN_NORMAL } else { 0.0 };
    }
    let y_bits = if negative { bits + 1 } else { bits - 1 };
    if y_bits & 0x7fff_ffff < 0x0080_0000 {
        return if negative { -F32_MIN_NORMAL } else { 0.0 };
    }
    f32::from_bits(y_bits)
}

fn next_up_f32_normal(x: f32) -> f32 {
    let bits = x.to_bits();
    let magnitude = bits & 0x7fff_ffff;
    let negative = bits & 0x8000_0000 != 0;
    if magnitude >= 0x7f80_0000 {
        return x;
    }
    if magnitude == 0 {
        return F32_MIN_NORMAL;
    }
    if magnitude < 0x0080_0000 {
        return if negative { -0.0 } else { F32_MIN_NORMAL };
    }
    let y_bits = if negative { bits - 1 } else { bits + 1 };
    if y_bits & 0x7fff_ffff < 0x0080_0000 {
        return if negative { -0.0 } else { F32_MIN_NORMAL };
    }
    f32::from_bits(y_bits)
}

/// One concretize spec row: coefficients/errors per tap plus the input box.
struct ConcretizeRow {
    a_l: Vec<f32>,
    a_u: Vec<f32>,
    e_l: Vec<f32>,
    e_u: Vec<f32>,
    x_l: Vec<f32>,
    x_u: Vec<f32>,
}

/// `CROWN_CONCRETIZE_SOUND_SHADER`, legacy branch (`eft_mode = 0`), one spec
/// row, one 256-lane workgroup, transcribed line by line under `hw`.
/// `slack_factor` multiplies the host `slack` uniform exactly as
/// `charged_concretize_slack` does (factor 1 ⇒ the shipped uniform).
fn concretize_publish(hw: Hw, row: &ConcretizeRow, slack_factor: f32) -> (f32, f32) {
    let n = row.a_l.len();
    let gamma_n = gamma_k_f32(n).expect("finite Higham regime");
    let base_slack = combine_slack_f32(n).expect("finite Higham regime");
    let slack = if slack_factor == 1.0 {
        base_slack
    } else {
        charged_concretize_slack(base_slack, slack_factor).expect("finite charged slack")
    };
    let additive = rung3_flush_safe_additive(u32::try_from(n).unwrap()).unwrap();

    let mut sh_lb = [0.0f32; 256];
    let mut sh_ub = [0.0f32; 256];
    let mut sh_pl = [0.0f32; 256];
    let mut sh_pu = [0.0f32; 256];
    let mut sh_fa = [0.0f32; 256];
    for t in 0..256usize {
        let mut local_lb = 0.0f32;
        let mut local_ub = 0.0f32;
        let mut pen_l = 0.0f32;
        let mut pen_u = 0.0f32;
        let mut flushacc = 1.0f32;
        let mut j = t;
        while j < n {
            let a_l = row.a_l[j];
            let a_u = row.a_u[j];
            let e_l = row.e_l[j];
            let e_u = row.e_u[j];
            let x_l = row.x_l[j];
            let x_u = row.x_u[j];
            let a_l_pos = a_l.max(0.0);
            let a_l_neg = a_l.min(0.0);
            let a_u_pos = a_u.max(0.0);
            let a_u_neg = a_u.min(0.0);
            let xmax = x_l.abs().max(x_u.abs());
            local_lb = hw.add(hw.add(local_lb, hw.mul(a_l_pos, x_l)), hw.mul(a_l_neg, x_u));
            local_ub = hw.add(hw.add(local_ub, hw.mul(a_u_pos, x_u)), hw.mul(a_u_neg, x_l));
            pen_l = hw.add(pen_l, hw.mul(hw.add(e_l, hw.mul(gamma_n, a_l.abs())), xmax));
            pen_u = hw.add(pen_u, hw.mul(hw.add(e_u, hw.mul(gamma_n, a_u.abs())), xmax));
            flushacc = hw.add(flushacc, a_l.abs().max(a_u.abs()).max(xmax).max(1.0));
            j += 256;
        }
        sh_lb[t] = local_lb;
        sh_ub[t] = local_ub;
        sh_pl[t] = pen_l;
        sh_pu[t] = pen_u;
        sh_fa[t] = flushacc;
    }
    let mut stride = 128usize;
    while stride > 0 {
        for t in 0..stride {
            sh_lb[t] = hw.add(sh_lb[t], sh_lb[t + stride]);
            sh_ub[t] = hw.add(sh_ub[t], sh_ub[t + stride]);
            sh_pl[t] = hw.add(sh_pl[t], sh_pl[t + stride]);
            sh_pu[t] = hw.add(sh_pu[t], sh_pu[t + stride]);
            sh_fa[t] = hw.add(sh_fa[t], sh_fa[t + stride]);
        }
        stride >>= 1;
    }
    // final assembly (thread 0), rs = 0 in legacy mode, bias = 0 in the corpus
    let flush_inner = round_up_pos(hw.mul(sh_fa[0], slack));
    let flush_scaled = round_up_pos(hw.mul(flush_inner, F32_MIN_NORMAL));
    let flush = round_up_pos(hw.add(additive, flush_scaled));
    let prop_l = round_up_pos(hw.mul(sh_pl[0], slack));
    let prop_u = round_up_pos(hw.mul(sh_pu[0], slack));
    let resid_l = 0.0f32; // round_up_pos(rl * 0)
    let resid_u = 0.0f32;
    let pen_l = round_up_pos(hw.add(round_up_pos(hw.add(prop_l, resid_l)), flush));
    let pen_u = round_up_pos(hw.add(round_up_pos(hw.add(prop_u, resid_u)), flush));
    let cl = next_down_f32_normal(hw.add(sh_lb[0], 0.0));
    let cu = next_up_f32_normal(hw.add(sh_ub[0], 0.0));
    let lb = next_down_f32_normal(hw.sub(cl, pen_l));
    let ub = next_up_f32_normal(hw.add(cu, pen_u));
    (lb, ub)
}

/// The exact concretize claim for one spec row (bias 0): for every realization
/// `â ∈ [a − e, a + e]` elementwise, `lb ≤ Σ_j min(â_j·x_l_j, â_j·x_u_j)` and
/// `ub ≥ Σ_j max(â_j·x_l_j, â_j·x_u_j)`. The per-tap inf/sup of the piecewise
/// linear map is attained at an interval endpoint, so both sides reduce to a
/// min/max over the four endpoint products — carried exactly in `2^-298`
/// units.
fn concretize_exact_bounds(row: &ConcretizeRow) -> (BigInt, BigInt) {
    let mut inf_l = BigInt::from(0);
    let mut sup_u = BigInt::from(0);
    for j in 0..row.a_l.len() {
        let xl = f32_units(row.x_l[j]);
        let xu = f32_units(row.x_u[j]);
        // lower side: coefficient interval [a_l - e_l, a_l + e_l]
        let lo = f32_units(row.a_l[j]) - f32_units(row.e_l[j]);
        let hi = f32_units(row.a_l[j]) + f32_units(row.e_l[j]);
        let products = [&lo * &xl, &lo * &xu, &hi * &xl, &hi * &xu];
        inf_l += products.iter().min().expect("nonempty").clone();
        // upper side: coefficient interval [a_u - e_u, a_u + e_u]
        let lo = f32_units(row.a_u[j]) - f32_units(row.e_u[j]);
        let hi = f32_units(row.a_u[j]) + f32_units(row.e_u[j]);
        let products = [&lo * &xl, &lo * &xu, &hi * &xl, &hi * &xu];
        sup_u += products.iter().max().expect("nonempty").clone();
    }
    (inf_l, sup_u)
}

/// `(lower_encloses, upper_encloses)` for one row/model/factor.
fn concretize_encloses(hw: Hw, row: &ConcretizeRow, slack_factor: f32) -> (bool, bool) {
    let (lb, ub) = concretize_publish(hw, row, slack_factor);
    let (inf_l, sup_u) = concretize_exact_bounds(row);
    (f32_prod_units(lb) <= inf_l, f32_prod_units(ub) >= sup_u)
}

/// The double-channel adversary: subnormal coefficients AND subnormal
/// accumulated errors against large-magnitude inputs. Per tap the DAZ losses
/// are `≈ μ·xmax` in the VALUE dot plus `≈ μ·xmax` in the PENALTY dot against
/// ONE `max(|a_l|,|a_u|,xmax,1) = xmax` charge.
fn concretize_double_channel(n: usize, x_mag: f32) -> ConcretizeRow {
    ConcretizeRow {
        a_l: vec![largest_subnormal(); n],
        a_u: vec![largest_subnormal(); n],
        e_l: vec![largest_subnormal(); n],
        e_u: vec![largest_subnormal(); n],
        x_l: vec![-x_mag; n],
        x_u: vec![x_mag; n],
    }
}

/// The single-channel control: zero coefficients, subnormal errors only — the
/// exactly-tight analogue of the operand-DAZ channel.
fn concretize_err_only(n: usize, x_mag: f32) -> ConcretizeRow {
    ConcretizeRow {
        a_l: vec![0.0; n],
        a_u: vec![0.0; n],
        e_l: vec![largest_subnormal(); n],
        e_u: vec![largest_subnormal(); n],
        x_l: vec![-x_mag; n],
        x_u: vec![x_mag; n],
    }
}

/// The REFUSED channel: subnormal INPUT-box endpoints under a large
/// accumulated error. `e·xmax` loses `e·μ` per tap and no flushacc term scales
/// with `e` — unchargeable by any constant factor, hence
/// `FlushChargePolicy::refuse_subnormal_inputs`.
fn concretize_subnormal_input(n: usize, err_mag: f32) -> ConcretizeRow {
    ConcretizeRow {
        a_l: vec![0.0; n],
        a_u: vec![0.0; n],
        e_l: vec![err_mag; n],
        e_u: vec![err_mag; n],
        x_l: vec![-largest_subnormal(); n],
        x_u: vec![largest_subnormal(); n],
    }
}

/// #flush-charge §F THE RESULT: the shipped (factor-1) concretize uniform does
/// NOT enclose on a pure-flush adapter — the multi-channel DAZ adversary
/// breaks it — while `CHARGED_CONCRETIZE_SLACK_FACTOR` restores enclosure with
/// at least 2× margin (the half-factor also encloses on the whole corpus).
/// The IEEE control consumes nothing at factor 1.
#[test]
fn charged_concretize_factor_covers_the_multi_channel_daz_demand() {
    let factor = CHARGED_CONCRETIZE_SLACK_FACTOR;
    assert_eq!(factor, 8.0, "the audited factor moved; re-derive this test");

    let mut shipped_failures = 0usize;
    for hw in [METAL_CMP_DAZ, METAL_CMP_EXACT] {
        for n in [1usize, 8, 64, 300] {
            for x_mag in [1.0f32, 1.0e30] {
                let row = concretize_double_channel(n, x_mag);
                let (l1, u1) = concretize_encloses(hw, &row, 1.0);
                if !(l1 && u1) {
                    shipped_failures += 1;
                }
                // The audited factor AND its half must both enclose: the half
                // is the measured >= 2x margin statement.
                for f in [factor, factor / 2.0] {
                    let (l, u) = concretize_encloses(hw, &row, f);
                    assert!(
                        l && u,
                        "{hw:?} double-channel n={n} x={x_mag:e} factor={f}: \
                         charged concretize does not enclose"
                    );
                }
                // single-channel control stays enclosed even at factor 1
                let (l, u) = concretize_encloses(hw, &concretize_err_only(n, x_mag), 1.0);
                assert!(
                    l && u,
                    "{hw:?} err-only n={n} x={x_mag:e} broke at factor 1"
                );
            }
        }
    }
    assert!(
        shipped_failures >= 4,
        "the double-channel adversary stopped breaking the shipped uniform \
         ({shipped_failures} failures) — either the shader was fixed (update \
         the charged factor derivation) or the model drifted"
    );

    // IEEE control: no flushing, factor 1 encloses everything above.
    for n in [1usize, 8, 64, 300] {
        for row in [
            concretize_double_channel(n, 1.0e30),
            concretize_err_only(n, 1.0e30),
            concretize_subnormal_input(n, 1.0e6),
        ] {
            let (l, u) = concretize_encloses(IEEE, &row, 1.0);
            assert!(l && u, "IEEE n={n}: factor-1 must enclose");
        }
    }
}

/// #flush-charge §F NEGATIVE RESULT (the refusal's justification): a subnormal
/// input-box endpoint under a large accumulated coefficient error is NOT
/// covered even by the charged factor — the loss scales with `e`, the charge
/// does not. This is why `FlushChargePolicy::refuse_subnormal_inputs` exists
/// and why the concretize host guard refuses such boxes under charged
/// authority.
#[test]
fn charged_concretize_cannot_cover_subnormal_inputs() {
    let row = concretize_subnormal_input(16, 1.0e6);
    let (l, u) = concretize_encloses(METAL_CMP_DAZ, &row, CHARGED_CONCRETIZE_SLACK_FACTOR);
    assert!(
        !(l && u),
        "the subnormal-input adversary stopped breaking the charged cover; \
         if the shader gained an err-scaled term, the refusal can be \
         reconsidered — until then it is mandatory"
    );
}

/// Randomized hunt over subnormal-heavy concretize rows (normal-or-zero
/// inputs, as the charged guard enforces): the charged factor never fails,
/// the shipped factor measurably does.
#[test]
fn randomised_concretize_hunt_separates_shipped_from_charged() {
    let hw = METAL_CMP_DAZ;
    let mut rng = Rng(0x2026_0812_0001);
    let (mut shipped_fail, mut charged_fail) = (0usize, 0usize);
    let rows = 4_000;
    for _ in 0..rows {
        let n = 1 + (rng.u32() % 48) as usize;
        let mut row = ConcretizeRow {
            a_l: Vec::new(),
            a_u: Vec::new(),
            e_l: Vec::new(),
            e_u: Vec::new(),
            x_l: Vec::new(),
            x_u: Vec::new(),
        };
        for _ in 0..n {
            let mut coeff = || {
                if rng.u32() % 3 == 0 {
                    f32::from_bits(1 + (rng.u32() % 0x0080_0000))
                } else {
                    register_f32(&mut rng, -126, 8)
                }
            };
            let al = coeff();
            let au = coeff();
            row.a_l.push(al);
            row.a_u.push(au);
            let mut err = || {
                if rng.u32() % 3 == 0 {
                    f32::from_bits(1 + (rng.u32() % 0x0080_0000))
                } else {
                    register_f32(&mut rng, -126, 4).abs()
                }
            };
            row.e_l.push(err());
            row.e_u.push(err());
            // normal-or-zero inputs (subnormal endpoints are refused by policy)
            let xa = register_f32(&mut rng, -20, 30);
            let xb = register_f32(&mut rng, -20, 30);
            row.x_l.push(xa.min(xb));
            row.x_u.push(xa.max(xb));
        }
        let (l1, u1) = concretize_encloses(hw, &row, 1.0);
        if !(l1 && u1) {
            shipped_fail += 1;
        }
        let (lc, uc) = concretize_encloses(hw, &row, CHARGED_CONCRETIZE_SLACK_FACTOR);
        if !(lc && uc) {
            charged_fail += 1;
        }
    }
    assert_eq!(
        charged_fail, 0,
        "charged concretize factor failed on {charged_fail}/{rows} random rows"
    );
    assert!(
        shipped_fail > 0,
        "the hunt found no shipped-factor failure in {rows} rows; it has lost \
         its discriminating power"
    );
}

// ---------------------------------------------------------------------------
// #flush-charge §D — CROWN_BIAS_ERR_ACCUMULATE_SHADER, legacy arm (eft=0)
// ---------------------------------------------------------------------------

struct BiasRow {
    a: Vec<f32>,
    err: Vec<f32>,
    bias: Vec<f32>,
}

/// The bias combine's legacy arm for one spec row (running bias/err both 0),
/// transcribed line by line under `hw`; returns `(bias_out, bias_err_out)`.
/// `slack_factor` multiplies the host `slack` uniform exactly as
/// `charged_bias_slack` does (factor 1 ⇒ the shipped uniform).
fn bias_combine_publish(hw: Hw, row: &BiasRow, slack_factor: f32) -> (f32, f32) {
    let k = row.a.len();
    let gamma_k = gamma_k_f32(k).expect("finite Higham regime");
    let base_slack = combine_slack_f32(k).expect("finite Higham regime");
    let slack = if slack_factor == 1.0 {
        base_slack
    } else {
        charged_bias_slack(base_slack, slack_factor).expect("finite charged slack")
    };
    let additive = rung3_flush_safe_additive(u32::try_from(k).unwrap()).unwrap();

    let mut sv = [0.0f32; 256];
    let mut sa = [0.0f32; 256];
    let mut se = [0.0f32; 256];
    let mut sf = [0.0f32; 256];
    for t in 0..256usize {
        let mut v = 0.0f32;
        let mut av = 0.0f32;
        let mut ev = 0.0f32;
        let mut fa = 1.0f32;
        let mut j = t;
        while j < k {
            let aj = row.a[j];
            let bj = row.bias[j];
            v = hw.add(v, hw.mul(aj, bj));
            av = hw.add(av, hw.mul(aj, bj).abs());
            ev = hw.add(ev, hw.mul(row.err[j], bj.abs()));
            fa = hw.add(fa, aj.abs().max(bj.abs()).max(1.0));
            j += 256;
        }
        sv[t] = v;
        sa[t] = av;
        se[t] = ev;
        sf[t] = fa;
    }
    let mut stride = 128usize;
    while stride > 0 {
        for t in 0..stride {
            sv[t] = hw.add(sv[t], sv[t + stride]);
            sa[t] = hw.add(sa[t], sa[t + stride]);
            se[t] = hw.add(se[t], se[t + stride]);
            sf[t] = hw.add(sf[t], sf[t + stride]);
        }
        stride >>= 1;
    }
    // thread 0, legacy arm; old = old_err = 0
    let sum = hw.add(0.0, sv[0]);
    let flush_inner = round_up_pos(hw.mul(sf[0], slack));
    let flush_scaled = round_up_pos(hw.mul(flush_inner, F32_MIN_NORMAL));
    let flush = round_up_pos(hw.add(additive, flush_scaled));
    let reduced_err = round_up_pos(hw.add(hw.mul(gamma_k, sa[0]), se[0]));
    let local_err = round_up_pos(hw.mul(reduced_err, slack));
    let bias_err = round_up_pos(hw.add(round_up_pos(hw.add(0.0, local_err)), flush));
    (sum, bias_err)
}

/// The exact bias-combine claim: `|Σ a_j·b_j − bias_out| + Σ err_j·|b_j| ≤
/// bias_err_out`, all in `2^-298` units.
fn bias_combine_encloses(hw: Hw, row: &BiasRow, slack_factor: f32) -> bool {
    let (out, err_out) = bias_combine_publish(hw, row, slack_factor);
    let mut dot = BigInt::from(0);
    let mut prop = BigInt::from(0);
    for j in 0..row.a.len() {
        dot += f32_units(row.a[j]) * f32_units(row.bias[j]);
        prop += f32_units(row.err[j]) * f32_units(row.bias[j].abs());
    }
    (dot - f32_prod_units(out)).abs() + prop <= f32_prod_units(err_out)
}

/// Double-DAZ bias adversary: subnormal `a_j` (value channel) AND subnormal
/// `err_j` (propagated channel), each losing `≈ μ·|b_j|` against ONE
/// `max(|a_j|,|b_j|,1)` charge.
fn bias_double_daz(k: usize, b_mag: f32) -> BiasRow {
    BiasRow {
        a: vec![largest_subnormal(); k],
        err: vec![largest_subnormal(); k],
        bias: vec![b_mag; k],
    }
}

/// The REFUSED bias channel: subnormal `b_j` under large `err_j` loses
/// `err·μ` per tap with no covering term — hence
/// `FlushChargePolicy::refuse_subnormal_bias`.
fn bias_subnormal_bias(k: usize, err_mag: f32) -> BiasRow {
    BiasRow {
        a: vec![0.0; k],
        err: vec![err_mag; k],
        bias: vec![largest_subnormal(); k],
    }
}

/// #flush-charge §D THE RESULT: the shipped bias-combine uniform does NOT
/// enclose the double-DAZ adversary on a pure-flush adapter;
/// `CHARGED_BIAS_COMBINE_SLACK_FACTOR` restores enclosure with margin (the
/// half-factor also encloses), the subnormal-bias channel stays unchargeable
/// (justifying the refusal), and the IEEE control needs no factor at all.
#[test]
fn charged_bias_combine_factor_covers_the_double_daz_demand() {
    let factor = CHARGED_BIAS_COMBINE_SLACK_FACTOR;
    assert_eq!(factor, 4.0, "the audited factor moved; re-derive this test");

    let mut shipped_failures = 0usize;
    for hw in [METAL_CMP_DAZ, METAL_CMP_EXACT] {
        for k in [1usize, 16, 256, 2000] {
            for b_mag in [1.0f32, 1.0e30] {
                let row = bias_double_daz(k, b_mag);
                if !bias_combine_encloses(hw, &row, 1.0) {
                    shipped_failures += 1;
                }
                for f in [factor, factor / 2.0] {
                    assert!(
                        bias_combine_encloses(hw, &row, f),
                        "{hw:?} double-daz k={k} b={b_mag:e} factor={f}: \
                         charged bias combine does not enclose"
                    );
                }
            }
        }
        // the refused channel is not covered even charged
        assert!(
            !bias_combine_encloses(hw, &bias_subnormal_bias(16, 1.0e6), factor),
            "{hw:?}: the subnormal-bias adversary stopped breaking the charged \
             cover; the refuse_subnormal_bias policy predicate guards exactly \
             this channel"
        );
    }
    assert!(
        shipped_failures >= 4,
        "the double-DAZ bias adversary stopped breaking the shipped uniform \
         ({shipped_failures} failures)"
    );

    for k in [1usize, 16, 256] {
        for row in [bias_double_daz(k, 1.0e30), bias_subnormal_bias(k, 1.0e6)] {
            assert!(
                bias_combine_encloses(IEEE, &row, 1.0),
                "IEEE k={k}: factor-1 must enclose"
            );
        }
    }
}

/// Randomized hunt over subnormal-heavy bias rows (normal-or-zero bias, as the
/// charged guard enforces): the charged factor never fails; the shipped one
/// measurably does.
#[test]
fn randomised_bias_hunt_separates_shipped_from_charged() {
    let hw = METAL_CMP_DAZ;
    let mut rng = Rng(0x2026_0812_0002);
    let (mut shipped_fail, mut charged_fail) = (0usize, 0usize);
    let rows = 6_000;
    for _ in 0..rows {
        let k = 1 + (rng.u32() % 64) as usize;
        let mut row = BiasRow {
            a: Vec::new(),
            err: Vec::new(),
            bias: Vec::new(),
        };
        for _ in 0..k {
            row.a.push(if rng.u32() % 3 == 0 {
                f32::from_bits(1 + (rng.u32() % 0x0080_0000))
            } else {
                register_f32(&mut rng, -126, 8)
            });
            row.err.push(if rng.u32() % 3 == 0 {
                f32::from_bits(1 + (rng.u32() % 0x0080_0000))
            } else {
                register_f32(&mut rng, -126, 4).abs()
            });
            // normal-or-zero bias (subnormal bias entries are refused by policy)
            row.bias.push(register_f32(&mut rng, -20, 30));
        }
        if !bias_combine_encloses(hw, &row, 1.0) {
            shipped_fail += 1;
        }
        if !bias_combine_encloses(hw, &row, CHARGED_BIAS_COMBINE_SLACK_FACTOR) {
            charged_fail += 1;
        }
    }
    assert_eq!(
        charged_fail, 0,
        "charged bias factor failed on {charged_fail}/{rows} random rows"
    );
    assert!(
        shipped_fail > 0,
        "the hunt found no shipped-factor failure in {rows} rows"
    );
}

// ---------------------------------------------------------------------------
// #flush-charge §E — CROWN_ACTIVATION_INTERCEPT_BIAS_SHADER, legacy arm
// (eft_mode = 0; EFT is FORBIDDEN under charged authority, so the legacy arm
// is the only charged-reachable branch)
// ---------------------------------------------------------------------------
//
// THE CHANNEL ENUMERATION (per tap, per side, li/ui NORMAL-or-zero as the
// charged guard enforces; μ = 2^-126, M = max(|a|,|li|,|ui|,1) — the ONE
// flushacc charge the shader carries per tap):
//
//  1. value-DAZ of the coefficient: subnormal `a` zeroed in `a·sel_int`, loss
//     ≤ μ·|sel| ≤ μ·max(|li|,|ui|) ≤ μ·M — ONE max unit. Mutually exclusive
//     with (2) on the same multiply (the DAZ'd product is an exact ±0).
//  2. FTZ of the `a·sel_int` product (normal operands, subnormal result):
//     < μ, additive-covered (also drops the same product from `av`, a
//     second-order γ_k·μ effect).
//  3. DAZ of the incoming radius: subnormal `err` zeroed in
//     `err·(|li|+|ui|)`, vaporizing the propagated cover. The DEMAND that
//     cover was carrying is only `err·Lip ≤ μ⁻·max(|li|,|ui|)` — ONE max
//     unit, not two: `|li|+|ui|` is 2× over-provisioned against the
//     Lipschitz constant of `v ↦ v·int(v)` (the ADDITIVE1 lemma below).
//     Mutually exclusive with (4); NOT exclusive with (1) — independent
//     buffers, so the joint per-tap flushacc demand is TWO max units.
//  4. FTZ of the `err·(|li|+|ui|)` product: < μ, additive-covered.
//  5. `|li|+|ui|` itself: sum of two normal-or-zero magnitudes is never
//     subnormal — no channel (guard-enforced input class).
//  6. add-DAZ/FTZ in the running sums and the tree (`v/av/ev/fa` lanes):
//     < μ each, inside the rung-3 additive's `9k+4800` flush-point budget.
//  7. γ-cover interplay: a flushed product leaves `av` (and so `γ_k·sa[0]`)
//     smaller, but Higham's bound applies to the values ACTUALLY summed, so
//     the deficit is the flush loss itself (charged above) plus a
//     second-order `γ_k·μ` term — no extra max unit.
//  8. sel-misselection under a DAZ compare (`-tiny >= 0` flushing to
//     `-0 >= 0` = true): only reachable when `a` is subnormal, where the
//     product is DAZ-zeroed regardless of the branch and the demand bound of
//     (1) already maximizes over BOTH intercepts — no extra channel.
//  9. flushacc/round_up_pos assembly: `fa ≥ 1` per lane is never subnormal;
//     `round_up_pos` is bit-manipulation, flush-immune. No channel.
// 10. subnormal INTERCEPT: `err·μ⁻` lost with NOTHING scaling with `err` —
//     UNCHARGEABLE by any constant factor; refused permanently
//     (`FlushChargePolicy::refuse_subnormal_slopes` covers intercepts;
//     negative pin `charged_act_bias_cannot_cover_subnormal_intercepts`).
//
// Worst joint per-tap flushacc demand: (1)+(3) = 2μ·M ⇒
// `CHARGED_ACT_BIAS_SLACK_FACTOR = 4 = 2 channels × 2 margin`, exactly the
// §D bias-combine shape (as the 2026-08-12 adversarial review predicted).

/// #flush-charge §E hardware model: the legacy arm uses only core `*`, `+`
/// and one `>=` compare (no FMA), so the policy is expressed per OPERATION
/// CLASS. This is strictly finer than [`Hw`]'s shared core policy — it also
/// expresses the SUBSET-flush adversaries (`ACT_MUL_DAZ_ONLY`,
/// `ACT_ADD_FLUSH_ONLY`) that a shared-core model cannot.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct ActHw {
    mul_daz: bool,
    mul_ftz: bool,
    add_daz: bool,
    add_ftz: bool,
    /// Whether DAZ also applies to the `a_v >= 0.0` select compare.
    /// Unspecified for MSL, so soundness is required under BOTH answers.
    cmp_daz: bool,
}

/// The pure-flush admission class (Apple M5 Max / Metal): everything flushes.
const ACT_FULL_FLUSH: ActHw = ActHw {
    mul_daz: true,
    mul_ftz: true,
    add_daz: true,
    add_ftz: true,
    cmp_daz: true,
};
/// Same flush policy, exact compares (the other MSL answer).
const ACT_FULL_FLUSH_CMP_EXACT: ActHw = ActHw {
    mul_daz: true,
    mul_ftz: true,
    add_daz: true,
    add_ftz: true,
    cmp_daz: false,
};
/// Subset-flush adversary: only multiplies lose subnormal OPERANDS.
const ACT_MUL_DAZ_ONLY: ActHw = ActHw {
    mul_daz: true,
    mul_ftz: false,
    add_daz: false,
    add_ftz: false,
    cmp_daz: false,
};
/// Subset-flush adversary: only adds flush (operands and results).
const ACT_ADD_FLUSH_ONLY: ActHw = ActHw {
    mul_daz: false,
    mul_ftz: false,
    add_daz: true,
    add_ftz: true,
    cmp_daz: false,
};
/// Non-flushing control.
const ACT_IEEE: ActHw = ActHw {
    mul_daz: false,
    mul_ftz: false,
    add_daz: false,
    add_ftz: false,
    cmp_daz: false,
};

impl ActHw {
    fn mul(self, a: f32, b: f32) -> f32 {
        Hw::flush_result(
            self.mul_ftz,
            Hw::flush_operand(self.mul_daz, a) * Hw::flush_operand(self.mul_daz, b),
        )
    }
    fn add(self, a: f32, b: f32) -> f32 {
        Hw::flush_result(
            self.add_ftz,
            Hw::flush_operand(self.add_daz, a) + Hw::flush_operand(self.add_daz, b),
        )
    }
    fn ge_zero(self, x: f32) -> bool {
        if self.cmp_daz {
            Hw::flush_operand(true, x) >= 0.0
        } else {
            x >= 0.0
        }
    }
}

/// One spec row of the intercept-bias combine: pre-transform coefficients and
/// their certified radii against the layer's per-neuron intercepts.
struct ActBiasRow {
    a: Vec<f32>,
    err: Vec<f32>,
    li: Vec<f32>,
    ui: Vec<f32>,
}

/// `CROWN_ACTIVATION_INTERCEPT_BIAS_SHADER`, legacy arm (`eft_mode = 0`), one
/// spec row, one 256-lane workgroup, single domain (`sbase = 0`), running
/// bias/err both 0 — transcribed line by line under `hw`. `slack_factor`
/// multiplies the host `slack` uniform exactly as `charged_act_bias_slack`
/// does (factor 1 ⇒ the shipped uniform, byte-identical).
fn act_bias_publish(hw: ActHw, is_upper: bool, row: &ActBiasRow, slack_factor: f32) -> (f32, f32) {
    let nn = row.a.len();
    let gamma_k = gamma_k_f32(nn).expect("finite Higham regime");
    let base_slack = combine_slack_f32(nn).expect("finite Higham regime");
    let slack = if slack_factor == 1.0 {
        base_slack
    } else {
        charged_act_bias_slack(base_slack, slack_factor).expect("finite charged slack")
    };
    let additive = rung3_flush_safe_additive(u32::try_from(nn).unwrap()).unwrap();

    let mut sv = [0.0f32; 256];
    let mut sa = [0.0f32; 256];
    let mut se = [0.0f32; 256];
    let mut sf = [0.0f32; 256];
    for t in 0..256usize {
        let mut v = 0.0f32;
        let mut av = 0.0f32;
        let mut ev = 0.0f32;
        let mut fa = 1.0f32;
        let mut j = t;
        while j < nn {
            let a_v = row.a[j];
            let li = row.li[j];
            let ui = row.ui[j];
            let ge = hw.ge_zero(a_v);
            let sel = if is_upper {
                if ge {
                    ui
                } else {
                    li
                }
            } else if ge {
                li
            } else {
                ui
            };
            v = hw.add(v, hw.mul(a_v, sel));
            av = hw.add(av, hw.mul(a_v, sel).abs());
            ev = hw.add(ev, hw.mul(row.err[j], hw.add(li.abs(), ui.abs())));
            fa = hw.add(fa, a_v.abs().max(li.abs().max(ui.abs())).max(1.0));
            j += 256;
        }
        sv[t] = v;
        sa[t] = av;
        se[t] = ev;
        sf[t] = fa;
    }
    let mut stride = 128usize;
    while stride > 0 {
        for t in 0..stride {
            sv[t] = hw.add(sv[t], sv[t + stride]);
            sa[t] = hw.add(sa[t], sa[t + stride]);
            se[t] = hw.add(se[t], se[t + stride]);
            sf[t] = hw.add(sf[t], sf[t + stride]);
        }
        stride >>= 1;
    }
    // thread 0, legacy arm; old = old_err = 0
    let sum = hw.add(0.0, sv[0]);
    let flush_scaled = round_up_pos(hw.mul(round_up_pos(hw.mul(sf[0], slack)), F32_MIN_NORMAL));
    let flush = round_up_pos(hw.add(additive, flush_scaled));
    let reduced_err = round_up_pos(hw.add(hw.mul(gamma_k, sa[0]), se[0]));
    let local_err = round_up_pos(hw.mul(reduced_err, slack));
    let bias_err = round_up_pos(hw.add(round_up_pos(hw.add(0.0, local_err)), flush));
    (sum, bias_err)
}

/// The exact intercept-bias claim for one spec row — the SAME realization
/// class as the live U5 device oracle
/// (`act_intercept_bias_eft_err_encloses_worst_realization`): for every joint
/// realization `â_j ∈ [a_j − e_j, a_j + e_j]`,
/// `|Σ_j â_j·sel_int(â_j) − bias_out| ≤ bias_err_out`, where `sel_int`
/// switches on the TRUE sign of the realized coefficient. Per tap the map
/// `v ↦ v·int(v)` is piecewise linear through 0, so its inf/sup over the
/// interval is attained on the 3-candidate set {lo·int(lo), 0 (if the
/// interval straddles 0), hi·int(hi)} — carried exactly in `2^-298` units.
fn act_bias_exact_range(is_upper: bool, row: &ActBiasRow) -> (BigInt, BigInt) {
    let zero = BigInt::from(0);
    let mut inf = BigInt::from(0);
    let mut sup = BigInt::from(0);
    for j in 0..row.a.len() {
        // v >= 0 selects li on the lower side and ui on the upper (the WGSL
        // select); v < 0 the other intercept.
        let int_pos = f32_units(if is_upper { row.ui[j] } else { row.li[j] });
        let int_neg = f32_units(if is_upper { row.li[j] } else { row.ui[j] });
        let a = f32_units(row.a[j]);
        let e = f32_units(row.err[j].abs());
        let lo = &a - &e;
        let hi = &a + &e;
        let mut cands: Vec<BigInt> = Vec::with_capacity(3);
        cands.push(if hi >= zero {
            &hi * &int_pos
        } else {
            &hi * &int_neg
        });
        cands.push(if lo >= zero {
            &lo * &int_pos
        } else {
            &lo * &int_neg
        });
        if lo <= zero && zero <= hi {
            cands.push(BigInt::from(0));
        }
        inf += cands.iter().min().expect("nonempty").clone();
        sup += cands.iter().max().expect("nonempty").clone();
    }
    (inf, sup)
}

/// Enclosure of the published `(bias_out, bias_err_out)` against the exact
/// realization range, decided in `2^-298` units.
fn act_bias_encloses(hw: ActHw, is_upper: bool, row: &ActBiasRow, slack_factor: f32) -> bool {
    let (out, err_out) = act_bias_publish(hw, is_upper, row, slack_factor);
    let (inf, sup) = act_bias_exact_range(is_upper, row);
    let o = f32_prod_units(out);
    let e = f32_prod_units(err_out);
    &o - &e <= inf && sup <= &o + &e
}

/// Double-DAZ act-bias adversary — the §E shape the 2026-08-12 review proved
/// fatal at factor 1: a subnormal runtime coefficient (value channel) AND a
/// subnormal runtime radius (propagated channel), each losing `≈ μ·|int|`
/// against ONE `max(|a|,|li|,|ui|,1)` charge. Neither buffer is
/// host-refusable; the intercept is NORMAL.
fn act_double_daz(nn: usize, int_mag: f32) -> ActBiasRow {
    ActBiasRow {
        a: vec![largest_subnormal(); nn],
        err: vec![largest_subnormal(); nn],
        li: vec![int_mag; nn],
        ui: vec![int_mag; nn],
    }
}

/// Single-channel control: subnormal coefficients only (`err = 0`). Exactly
/// one max unit per tap — enclosed even by the shipped factor-1 uniform.
fn act_value_daz_only(nn: usize, int_mag: f32) -> ActBiasRow {
    ActBiasRow {
        a: vec![largest_subnormal(); nn],
        err: vec![0.0; nn],
        li: vec![int_mag; nn],
        ui: vec![int_mag; nn],
    }
}

/// Single-channel control: subnormal radii only (`a = 0`). One max unit per
/// tap — enclosed even at factor 1.
fn act_err_daz_only(nn: usize, int_mag: f32) -> ActBiasRow {
    ActBiasRow {
        a: vec![0.0; nn],
        err: vec![largest_subnormal(); nn],
        li: vec![int_mag; nn],
        ui: vec![int_mag; nn],
    }
}

/// The REFUSED channel: subnormal INTERCEPTS under a large runtime radius.
/// The exact demand is `nn·err·μ⁻` while the DAZ'd `err·(|li|+|ui|)` cover
/// reads 0 and the flushacc charge saturates at `max(...) = 1` — the loss
/// scales with `err`, the charge does not. Unchargeable by ANY constant
/// factor.
fn act_subnormal_intercept(nn: usize, err_mag: f32) -> ActBiasRow {
    ActBiasRow {
        a: vec![0.0; nn],
        err: vec![err_mag; nn],
        li: vec![largest_subnormal(); nn],
        ui: vec![largest_subnormal(); nn],
    }
}

/// #flush-charge §E THE RESULT: the shipped (factor-1) intercept-bias uniform
/// does NOT enclose the double-DAZ adversary on a pure-flush adapter;
/// `CHARGED_ACT_BIAS_SLACK_FACTOR` restores enclosure with ≥ 2× margin (the
/// half-factor also encloses on the whole corpus), both single-channel
/// controls stay enclosed at factor 1, and the IEEE control needs no factor
/// at all. Both sides (`is_upper`) and both MSL compare answers are covered.
#[test]
fn charged_act_bias_factor_covers_the_double_daz_demand() {
    let factor = CHARGED_ACT_BIAS_SLACK_FACTOR;
    assert_eq!(factor, 4.0, "the audited factor moved; re-derive this test");

    let mut shipped_failures = 0usize;
    for hw in [ACT_FULL_FLUSH, ACT_FULL_FLUSH_CMP_EXACT] {
        for nn in [1usize, 16, 256, 2000] {
            for int_mag in [1.0f32, 1.0e30] {
                for is_upper in [false, true] {
                    let row = act_double_daz(nn, int_mag);
                    if !act_bias_encloses(hw, is_upper, &row, 1.0) {
                        shipped_failures += 1;
                    }
                    // The audited factor AND its half must both enclose: the
                    // half is the measured >= 2x margin statement.
                    for f in [factor, factor / 2.0] {
                        assert!(
                            act_bias_encloses(hw, is_upper, &row, f),
                            "{hw:?} double-daz nn={nn} int={int_mag:e} \
                             is_upper={is_upper} factor={f}: charged act-bias \
                             does not enclose"
                        );
                    }
                    // single-channel controls stay enclosed even at factor 1
                    for (name, control) in [
                        ("value-only", act_value_daz_only(nn, int_mag)),
                        ("err-only", act_err_daz_only(nn, int_mag)),
                    ] {
                        assert!(
                            act_bias_encloses(hw, is_upper, &control, 1.0),
                            "{hw:?} {name} nn={nn} int={int_mag:e} \
                             is_upper={is_upper} broke at factor 1"
                        );
                    }
                }
            }
        }
    }
    assert!(
        shipped_failures >= 8,
        "the double-DAZ act-bias adversary stopped breaking the shipped \
         uniform ({shipped_failures} failures) — either the shader was fixed \
         (update the charged factor derivation) or the model drifted"
    );

    // IEEE control: no flushing, factor 1 encloses everything above.
    for nn in [1usize, 16, 256] {
        for is_upper in [false, true] {
            for row in [
                act_double_daz(nn, 1.0e30),
                act_value_daz_only(nn, 1.0e30),
                act_err_daz_only(nn, 1.0e30),
                act_subnormal_intercept(nn, 1.0e6),
            ] {
                assert!(
                    act_bias_encloses(ACT_IEEE, is_upper, &row, 1.0),
                    "IEEE nn={nn} is_upper={is_upper}: factor-1 must enclose"
                );
            }
        }
    }
}

/// #flush-charge §E NEGATIVE RESULT (the permanent refusal's justification):
/// a subnormal INTERCEPT under a large runtime radius is NOT covered even by
/// the charged factor — the loss scales with `err`, the charge does not. This
/// is channel 10 of the enumeration and why the guard's subnormal
/// slope/intercept refusal (`FlushChargePolicy::refuse_subnormal_slopes`)
/// stays PERMANENT while nonzero NORMAL intercepts are re-admitted.
#[test]
fn charged_act_bias_cannot_cover_subnormal_intercepts() {
    for is_upper in [false, true] {
        let row = act_subnormal_intercept(16, 1.0e6);
        assert!(
            !act_bias_encloses(
                ACT_FULL_FLUSH,
                is_upper,
                &row,
                CHARGED_ACT_BIAS_SLACK_FACTOR
            ),
            "the subnormal-intercept adversary stopped breaking the charged \
             cover; if the shader gained an err-scaled flush term the refusal \
             can be reconsidered — until then it is mandatory"
        );
    }
}

/// #flush-charge §E ADDITIVE1 lemma — ONE max per tap beyond the exact
/// covers, and per-multiply channel exclusivity.
///
/// (a) EXCLUSIVITY: on one multiply, operand-DAZ and result-FTZ can never
/// stack — a DAZ-zeroed multiplicand yields an exact ±0 product on which FTZ
/// has nothing to act, and a normal×normal product needs no DAZ. So the
/// full-flush multiply equals the DAZ-only multiply whenever an operand is
/// subnormal, and equals the FTZ-only multiply whenever both are normal.
///
/// (b) ONE MAX PER TAP: for a single tap under the full-flush model, the
/// exact demand `D = max(sup − out, out − inf)` obeys
///
/// ```text
///   D ≤ ev_exact + u·|a·sel_true| + μ·M
/// ```
///
/// with `ev_exact = e·(|li|+|ui|)` exact, `u = 2^-24`, and
/// `M = max(|a|,|li|,|ui|,1)`: ONE μ·M unit, because `|li|+|ui|` already
/// over-provisions the propagated Lipschitz demand 2×. The published factor
/// must be 4 and not 2×that only because a DAZ'd `err` ALSO vaporizes the
/// device's `ev` delivery (≤ one more max unit routed to the flushacc) —
/// which is exactly what the double-DAZ corpus measures. Decided exactly in
/// `2^-322` units (the inequality is multiplied through by `2^24`).
#[test]
fn act_bias_per_tap_demand_is_one_max_beyond_exact_covers() {
    // (a) exclusivity, over subnormal boundary patterns and random pairs.
    let mut rng = Rng(0x2026_0813_0005);
    for _ in 0..100_000 {
        let sub = f32::from_bits(1 + (rng.u32() % 0x0080_0000));
        let normal = register_f32(&mut rng, -126, 60);
        for (x, y) in [(sub, normal), (normal, sub), (sub, sub)] {
            assert_eq!(
                ACT_FULL_FLUSH.mul(x, y).to_bits(),
                ACT_MUL_DAZ_ONLY.mul(x, y).to_bits(),
                "a subnormal operand must make FTZ a no-op (exact ±0 product)"
            );
            assert_eq!(
                ACT_FULL_FLUSH.mul(x, y).abs().to_bits(),
                0.0f32.to_bits(),
                "a DAZ'd multiply must produce an exact zero"
            );
        }
        let (m, n) = (
            register_f32(&mut rng, -126, 20),
            register_f32(&mut rng, -126, 20),
        );
        assert_eq!(
            ACT_FULL_FLUSH.mul(m, n).to_bits(),
            ActHw {
                mul_daz: false,
                ..ACT_FULL_FLUSH
            }
            .mul(m, n)
            .to_bits(),
            "normal operands must make DAZ a no-op"
        );
    }

    // (b) the one-max bound, on randomized single taps (normal-or-zero
    // intercepts, as the charged guard enforces; a/e span zero, subnormal and
    // normal).
    let mut rng = Rng(0x2026_0813_0006);
    let mu_322 = BigInt::from(1) << (322 - 126 - 149); // μ·(one f32 unit), 2^-322 scale
    let mut checked = 0usize;
    for _ in 0..50_000 {
        let a = match rng.u32() % 4 {
            0 => 0.0f32,
            1 => {
                let s = f32::from_bits(1 + (rng.u32() % 0x0080_0000));
                if rng.u32() % 2 == 0 {
                    s
                } else {
                    -s
                }
            }
            _ => register_f32(&mut rng, -126, 8),
        };
        let e = match rng.u32() % 4 {
            0 => 0.0f32,
            1 => f32::from_bits(1 + (rng.u32() % 0x0080_0000)),
            _ => register_f32(&mut rng, -126, 4).abs(),
        };
        let intercept = |rng: &mut Rng| {
            if rng.u32() % 4 == 0 {
                0.0f32
            } else {
                register_f32(rng, -30, 30)
            }
        };
        let li = intercept(&mut rng);
        let ui = intercept(&mut rng);
        for is_upper in [false, true] {
            for hw in [ACT_FULL_FLUSH, ACT_FULL_FLUSH_CMP_EXACT] {
                let row = ActBiasRow {
                    a: vec![a],
                    err: vec![e],
                    li: vec![li],
                    ui: vec![ui],
                };
                let (out, _) = act_bias_publish(hw, is_upper, &row, 1.0);
                let (inf, sup) = act_bias_exact_range(is_upper, &row);
                let o = f32_prod_units(out);
                let d = (&sup - &o).max(&o - &inf).max(BigInt::from(0));
                // sel on the TRUE sign of a (the exclusivity above makes the
                // DAZ'd-compare branch irrelevant exactly when a is subnormal).
                let sel_true = if is_upper == (a >= 0.0) { ui } else { li };
                let ev_exact = f32_units(e) * (f32_units(li.abs()) + f32_units(ui.abs()));
                let m_units = f32_units(a.abs().max(li.abs().max(ui.abs())).max(1.0));
                // D·2^24 ≤ 2^24·ev + 2^24·μ·M + |a|·|sel|, all in 2^-322 units.
                let lhs = &d << 24;
                let rhs = (&ev_exact << 24)
                    + &m_units * &mu_322
                    + (f32_units(a) * f32_units(sel_true)).abs();
                assert!(
                    lhs <= rhs,
                    "ADDITIVE1 violated at a={:#010x} e={:#010x} li={:#010x} \
                     ui={:#010x} is_upper={is_upper} hw={hw:?}: demand exceeds \
                     ev_exact + u·|a·sel| + μ·M",
                    a.to_bits(),
                    e.to_bits(),
                    li.to_bits(),
                    ui.to_bits()
                );
                checked += 1;
            }
        }
    }
    assert!(checked >= 200_000);
}

/// Randomized hunt over guard-admissible act-bias rows (normal-or-zero
/// intercepts, subnormal-heavy coefficients and radii), across the FULL-flush
/// model AND the two subset-flush adversaries: the charged factor never
/// fails; the shipped factor measurably does under full flush.
#[test]
fn randomised_act_bias_hunt_separates_shipped_from_charged() {
    let mut rng = Rng(0x2026_0813_0007);
    let (mut shipped_fail_full, mut charged_fail) = (0usize, 0usize);
    let rows_per_hw = 2_000usize;
    for hw in [ACT_FULL_FLUSH, ACT_MUL_DAZ_ONLY, ACT_ADD_FLUSH_ONLY] {
        for _ in 0..rows_per_hw {
            let nn = 1 + (rng.u32() % 48) as usize;
            let mut row = ActBiasRow {
                a: Vec::new(),
                err: Vec::new(),
                li: Vec::new(),
                ui: Vec::new(),
            };
            for _ in 0..nn {
                row.a.push(if rng.u32() % 3 == 0 {
                    let s = f32::from_bits(1 + (rng.u32() % 0x0080_0000));
                    if rng.u32() % 2 == 0 {
                        s
                    } else {
                        -s
                    }
                } else {
                    register_f32(&mut rng, -126, 8)
                });
                row.err.push(if rng.u32() % 3 == 0 {
                    f32::from_bits(1 + (rng.u32() % 0x0080_0000))
                } else {
                    register_f32(&mut rng, -126, 4).abs()
                });
                // normal-or-zero intercepts (subnormal intercepts are refused
                // by policy — the unchargeable channel 10)
                row.li.push(if rng.u32() % 4 == 0 {
                    0.0
                } else {
                    register_f32(&mut rng, -20, 30)
                });
                row.ui.push(if rng.u32() % 4 == 0 {
                    0.0
                } else {
                    register_f32(&mut rng, -20, 30)
                });
            }
            let is_upper = rng.u32() % 2 == 1;
            if hw == ACT_FULL_FLUSH && !act_bias_encloses(hw, is_upper, &row, 1.0) {
                shipped_fail_full += 1;
            }
            if !act_bias_encloses(hw, is_upper, &row, CHARGED_ACT_BIAS_SLACK_FACTOR) {
                charged_fail += 1;
            }
        }
    }
    assert_eq!(
        charged_fail, 0,
        "charged act-bias factor failed on {charged_fail} random rows \
         (full-flush + subset-flush policies)"
    );
    assert!(
        shipped_fail_full > 0,
        "the hunt found no shipped-factor failure in {rows_per_hw} full-flush \
         rows; it has lost its discriminating power"
    );
}

/// #flush-charge §E anti-drift needle: the transcription above must not drift
/// away from the shipped intercept-bias WGSL (legacy arm + assembly).
#[test]
fn charged_act_bias_model_tracks_the_shipped_shader_text() {
    let src = super::shaders::CROWN_ACTIVATION_INTERCEPT_BIAS_SHADER;
    for needle in [
        "if (p.is_upper == 0u) { sel = select(ui, li, a_v >= 0.0); }",
        "else { sel = select(li, ui, a_v >= 0.0); }",
        "v = v + a_v * sel;",
        "av = av + abs(a_v * sel);",
        "ev = ev + err[idx] * (abs(li) + abs(ui));",
        "fa = fa + max(max(abs(a_v), max(abs(li), abs(ui))), 1.0);",
        "let flush_scaled = round_up_pos(round_up_pos(sf[0] * p.slack) * F32_MIN_NORMAL);",
        "let flush = round_up_pos(p.additive + flush_scaled);",
        "let reduced_err = round_up_pos(p.gamma_k * sa[0] + se[0]);",
        "let local_err = round_up_pos(reduced_err * p.slack);",
        "bias_err_out[s] = round_up_pos(round_up_pos(old_err + local_err) + flush);",
    ] {
        assert!(
            src.contains(needle),
            "CROWN_ACTIVATION_INTERCEPT_BIAS_SHADER no longer contains \
             `{needle}`; the charged act-bias transcription is stale and \
             CHARGED_ACT_BIAS_SLACK_FACTOR's derivation does not apply"
        );
    }
}

/// The transcriptions above must not drift away from the shipped WGSL.
#[test]
fn charged_models_track_the_shipped_shader_text() {
    let concretize = super::shaders::CROWN_CONCRETIZE_SOUND_SHADER;
    for needle in [
        "local_lb = local_lb + a_l_pos * x_l + a_l_neg * x_u;",
        "local_ub = local_ub + a_u_pos * x_u + a_u_neg * x_l;",
        "pen_l = pen_l + (e_l + params.gamma_n * abs(a_l)) * xmax;",
        "pen_u = pen_u + (e_u + params.gamma_n * abs(a_u)) * xmax;",
        "flushacc = flushacc + max(max(max(abs(a_l), abs(a_u)), xmax), 1.0);",
        "let flush_scaled = round_up_pos(round_up_pos(sh_fa[0] * params.slack) * F32_MIN_NORMAL);",
        "let flush = round_up_pos(params.additive + flush_scaled);",
        "let prop_l = round_up_pos(sh_pl[0] * params.slack);",
        "let pen_l = round_up_pos(round_up_pos(prop_l + resid_l) + flush);",
        "var lb = next_down_f32_normal(cl - pen_l);",
    ] {
        assert!(
            concretize.contains(needle),
            "CROWN_CONCRETIZE_SOUND_SHADER no longer contains `{needle}`; the \
             charged concretize transcription is stale and its factor \
             derivation does not apply"
        );
    }
    let bias = super::shaders::CROWN_BIAS_ERR_ACCUMULATE_SHADER;
    for needle in [
        "v = v + aj * bj;",
        "av = av + abs(aj * bj);",
        "ev = ev + a_err[s * p.k + j] * abs(bj);",
        "fa = fa + max(max(abs(aj), abs(bj)), 1.0);",
        "let flush_scaled = round_up_pos(round_up_pos(sf[0] * p.slack) * F32_MIN_NORMAL);",
        "let flush = round_up_pos(p.additive + flush_scaled);",
        "let reduced_err = round_up_pos(p.gamma_k * sa[0] + se[0]);",
        "let local_err = round_up_pos(reduced_err * p.slack);",
        "bias_err_out[s] = round_up_pos(round_up_pos(old_err + local_err) + flush);",
    ] {
        assert!(
            bias.contains(needle),
            "CROWN_BIAS_ERR_ACCUMULATE_SHADER no longer contains `{needle}`; \
             the charged bias transcription is stale"
        );
    }
}

// ---------------------------------------------------------------------------
// The model must not drift away from the shipped shader text
// ---------------------------------------------------------------------------

/// If the shipped WGSL changes shape, this module's conclusions no longer apply
/// to it. Pin the exact lines the transcription depends on.
#[test]
fn model_tracks_the_shipped_shader_text() {
    let twin_src = super::shaders::GEMM_F32_EFT_TWIN_SHADER;
    for needle in [
        "let prod = a * w;",
        "let ep = fma(a, w, -prod);",
        "eterm = F32_MIN_NORMAL;",
        "let s = acc + prod;",
        "let bb = fma(-1.0, acc, s);",
        "let sb = fma(-1.0, bb, s);",
        "let da = fma(-1.0, sb, acc);",
        "let db = fma(-1.0, bb, prod);",
        "let es = da + db;",
        "rsum = rsum + eterm + abs(es);",
    ] {
        assert!(
            twin_src.contains(needle),
            "GEMM_F32_EFT_TWIN_SHADER no longer contains `{needle}`; the CPU \
             twin in this module is stale and its bounds do not apply"
        );
    }
    for src in [
        super::shaders::CROWN_EFT_MIN_COMBINE_SHADER,
        super::shaders::CROWN_AW_ERROR_COMBINE_SHADER,
    ] {
        assert!(
            src.contains("1.0 + f32(p.k) + p.w_l1_max"),
            "the §0 flushacc formula changed; re-derive the cover before \
             trusting this module"
        );
        assert!(
            src.contains("F32_MIN_NORMAL"),
            "flush floor quantum changed"
        );
    }
}
