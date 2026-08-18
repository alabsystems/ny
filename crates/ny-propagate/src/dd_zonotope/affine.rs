// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Certified affine transformers (Conv2d, Linear) for the double-double
//! zonotope (`#dd-zonotope`).
//!
//! # The certificate
//!
//! For an exact affine map `y = W x + b` (`W`, `b` are f32 network weights and
//! therefore EXACT f64 values — no conversion error) applied to a state whose
//! center carries error `ec` and whose generators carry error `eg`:
//!
//! ```text
//! center' = W center + b          computed in DOUBLE-DOUBLE
//! gens'_j = W gens_j              computed in plain f64
//! ec'     = |W| ec + gamma_{2K+2}(U_DD) * (|W| |center| + |b|) + Ωc
//! eg'     = |W| eg + gamma_{K+1}(U_F64) * (|W| radius) + Ωg
//! ```
//!
//! * `|W| ec` transports the incoming center error through the map — this is
//!   the ONLY error-amplifying step, and it is exactly the IBP transfer
//!   operator, i.e. the ~42x-per-conv-layer growth the probe measured.
//! * The `gamma_{2K+2}(U_DD)` term is the working-precision rounding NEWLY
//!   committed by this layer's double-double dot product. `2K` because each
//!   input element enters the accumulator as two exact products
//!   (`W*center.hi` and `W*center.lo`), `+2` for the bias add and one term of
//!   slack. See `ny_core::dd` for the derivation of `U_DD`.
//! * `gamma_{K+1}(U_F64)` is the same for the plain-f64 generator columns,
//!   applied to `|W| * sum_j |gens_j|` — the sum over columns of each column's
//!   own dot-product error bound.
//! * `Ωc = (48K+32)·2^-1074` and
//!   `Ωg = ((4·n_generators+16)K+32)·2^-1074` are absolute floors. They
//!   conservatively charge every elementary operation in the double-double
//!   EFTs, every generator multiply/add, the radius aggregation, both
//!   transported error-channel reductions, and final assembly. This is needed
//!   because a purely relative gamma bound cannot see a result that rounded
//!   away in the subnormal range.
//!
//! Every error-channel value is rounded OUTWARD (`err_up`) after each step.

use ny_core::dd::{dd_fma, gamma_n_dd, gamma_n_f64, next_up_f64, Dd};
use ny_core::{NyError, Result};
use rayon::prelude::*;

use super::state::{err_up, DdZono};

/// Build an outward absolute rounding floor for a checked elementary-op count.
pub(super) fn operation_underflow_floor(operations: usize) -> Result<f64> {
    if operations == 0 {
        return Ok(0.0);
    }
    #[allow(clippy::cast_precision_loss)]
    let floor = operations as f64 * f64::from_bits(1);
    if !floor.is_finite() || floor <= 0.0 {
        return Err(NyError::SoundnessRefusal(
            "#dd-zonotope could not construct its affine operation-underflow floor".to_string(),
        ));
    }
    Ok(next_up_f64(floor))
}

/// Conservative absolute-floor operation counts for one affine output.
///
/// `center`: two `dd_fma`s per term, each with about 16 elementary
/// operations, plus `abs_upper`, the `s_c`/`ec_t` reductions, and assembly.
/// `generator`: `2K` multiply/add operations for every generator column,
/// `K` radius additions per column, the `s_g`/`eg_t` reductions, and assembly.
pub(super) fn affine_underflow_operation_counts(
    k: usize,
    generators: usize,
) -> Result<(usize, usize)> {
    let center = k
        .checked_mul(48)
        .and_then(|count| count.checked_add(32))
        .ok_or_else(|| {
            NyError::SoundnessRefusal("#dd-zonotope center operation count overflow".to_string())
        })?;
    let generator_ops_per_term = generators
        .checked_mul(4)
        .and_then(|count| count.checked_add(16))
        .ok_or_else(|| {
            NyError::SoundnessRefusal("#dd-zonotope generator operation count overflow".to_string())
        })?;
    let generator = k
        .checked_mul(generator_ops_per_term)
        .and_then(|count| count.checked_add(32))
        .ok_or_else(|| {
            NyError::SoundnessRefusal("#dd-zonotope generator operation count overflow".to_string())
        })?;
    Ok((center, generator))
}

/// Fully-resolved Conv2d geometry (unbatched NCHW).
#[derive(Debug, Clone, Copy)]
pub(crate) struct ConvPlan {
    pub(crate) in_c: usize,
    pub(crate) in_h: usize,
    pub(crate) in_w: usize,
    pub(crate) out_c: usize,
    pub(crate) out_h: usize,
    pub(crate) out_w: usize,
    pub(crate) kh: usize,
    pub(crate) kw: usize,
    pub(crate) sh: usize,
    pub(crate) sw: usize,
    pub(crate) ph: usize,
    pub(crate) pw: usize,
}

#[derive(Debug, Clone, Copy)]
struct ConvSizes {
    in_hw: usize,
    out_hw: usize,
    kernel_hw: usize,
    dot_terms: usize,
    input_numel: usize,
    output_numel: usize,
    weight_numel: usize,
}

impl ConvPlan {
    /// Build the plan, refusing (`None`) anything outside the supported
    /// surface: dilation must be 1, groups must be 1, and the output must be
    /// non-empty. Refusing is always sound — the caller falls back.
    pub(crate) fn build(
        in_shape: (usize, usize, usize),
        out_c: usize,
        kh: usize,
        kw: usize,
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        groups: usize,
    ) -> Option<Self> {
        if dilation != (1, 1)
            || groups != 1
            || stride.0 == 0
            || stride.1 == 0
            || out_c == 0
            || kh == 0
            || kw == 0
        {
            return None;
        }
        let (in_c, in_h, in_w) = in_shape;
        if in_c == 0 || in_h == 0 || in_w == 0 {
            return None;
        }
        let num_h = padding
            .0
            .checked_mul(2)
            .and_then(|pad| in_h.checked_add(pad))?;
        let num_w = padding
            .1
            .checked_mul(2)
            .and_then(|pad| in_w.checked_add(pad))?;
        if num_h < kh || num_w < kw {
            return None;
        }
        let out_h = (num_h - kh) / stride.0 + 1;
        let out_w = (num_w - kw) / stride.1 + 1;
        if out_h == 0 || out_w == 0 {
            return None;
        }
        let plan = ConvPlan {
            in_c,
            in_h,
            in_w,
            out_c,
            out_h,
            out_w,
            kh,
            kw,
            sh: stride.0,
            sw: stride.1,
            ph: padding.0,
            pw: padding.1,
        };
        plan.checked_sizes()?;
        Some(plan)
    }

    /// Revalidate crate-visible geometry and every derived product before
    /// indexed arithmetic.
    fn checked_sizes(&self) -> Option<ConvSizes> {
        if self.in_c == 0
            || self.in_h == 0
            || self.in_w == 0
            || self.out_c == 0
            || self.kh == 0
            || self.kw == 0
            || self.sh == 0
            || self.sw == 0
        {
            return None;
        }
        let padded_h = self
            .ph
            .checked_mul(2)
            .and_then(|pad| self.in_h.checked_add(pad))?;
        let padded_w = self
            .pw
            .checked_mul(2)
            .and_then(|pad| self.in_w.checked_add(pad))?;
        if padded_h < self.kh || padded_w < self.kw {
            return None;
        }
        let expected_out_h = (padded_h - self.kh) / self.sh + 1;
        let expected_out_w = (padded_w - self.kw) / self.sw + 1;
        if (self.out_h, self.out_w) != (expected_out_h, expected_out_w) {
            return None;
        }

        let in_hw = self.in_h.checked_mul(self.in_w)?;
        let out_hw = self.out_h.checked_mul(self.out_w)?;
        let kernel_hw = self.kh.checked_mul(self.kw)?;
        let dot_terms = self.in_c.checked_mul(kernel_hw)?;
        let input_numel = self.in_c.checked_mul(in_hw)?;
        let output_numel = self.out_c.checked_mul(out_hw)?;
        let weight_numel = self.out_c.checked_mul(dot_terms)?;
        Some(ConvSizes {
            in_hw,
            out_hw,
            kernel_hw,
            dot_terms,
            input_numel,
            output_numel,
            weight_numel,
        })
    }
}

/// Plain-f64 direct convolution: `out = W x (+ bias)`.
///
/// `w` is C-order `(out_c, in_c, kh, kw)`. Parallel over output channels; the
/// innermost loop is a contiguous AXPY over the output row so the backend can
/// vectorize it.
pub(crate) fn conv_f64(
    plan: &ConvPlan,
    w: &[f64],
    bias: Option<&[f64]>,
    x: &[f64],
) -> Option<Vec<f64>> {
    let sizes = plan.checked_sizes()?;
    if w.len() != sizes.weight_numel
        || x.len() != sizes.input_numel
        || bias.is_some_and(|values| values.len() != plan.out_c)
    {
        return None;
    }
    let mut out = vec![0.0_f64; sizes.output_numel];
    out.par_chunks_mut(sizes.out_hw)
        .enumerate()
        .for_each(|(oc, out_c_slice)| {
            let b = bias.map_or(0.0, |b| b[oc]);
            out_c_slice.fill(b);
            for ic in 0..plan.in_c {
                let wbase = (oc * plan.in_c + ic) * sizes.kernel_hw;
                let xbase = ic * sizes.in_hw;
                for ky in 0..plan.kh {
                    for kx in 0..plan.kw {
                        let wv = w[wbase + ky * plan.kw + kx];
                        if wv == 0.0 {
                            continue;
                        }
                        for oy in 0..plan.out_h {
                            let iy = oy * plan.sh + ky;
                            if iy < plan.ph || iy - plan.ph >= plan.in_h {
                                continue;
                            }
                            let iy = iy - plan.ph;
                            let row_out = &mut out_c_slice[oy * plan.out_w..(oy + 1) * plan.out_w];
                            let row_in = &x[xbase + iy * plan.in_w..xbase + (iy + 1) * plan.in_w];
                            for (ox, o) in row_out.iter_mut().enumerate() {
                                let ix = ox * plan.sw + kx;
                                if ix < plan.pw {
                                    continue;
                                }
                                let ix = ix - plan.pw;
                                if ix >= plan.in_w {
                                    continue;
                                }
                                *o += wv * row_in[ix];
                            }
                        }
                    }
                }
            }
        });
    Some(out)
}

/// Double-double direct convolution over a double-double input center.
///
/// Each input element contributes TWO exact products (`w*hi`, `w*lo`), so the
/// accumulator sees a dot of length `2K`; the caller's `gamma_{2K+2}(U_DD)`
/// term is sized for exactly that.
pub(crate) fn conv_dd(
    plan: &ConvPlan,
    w: &[f64],
    bias: Option<&[f64]>,
    x: &[Dd],
) -> Option<Vec<Dd>> {
    let sizes = plan.checked_sizes()?;
    if w.len() != sizes.weight_numel
        || x.len() != sizes.input_numel
        || bias.is_some_and(|values| values.len() != plan.out_c)
    {
        return None;
    }
    let mut out = vec![Dd::ZERO; sizes.output_numel];
    out.par_chunks_mut(sizes.out_hw)
        .enumerate()
        .for_each(|(oc, out_c_slice)| {
            let b = bias.map_or(0.0, |b| b[oc]);
            for o in out_c_slice.iter_mut() {
                *o = Dd::from_f64(b);
            }
            for ic in 0..plan.in_c {
                let wbase = (oc * plan.in_c + ic) * sizes.kernel_hw;
                let xbase = ic * sizes.in_hw;
                for ky in 0..plan.kh {
                    for kx in 0..plan.kw {
                        let wv = w[wbase + ky * plan.kw + kx];
                        if wv == 0.0 {
                            continue;
                        }
                        for oy in 0..plan.out_h {
                            let iy = oy * plan.sh + ky;
                            if iy < plan.ph || iy - plan.ph >= plan.in_h {
                                continue;
                            }
                            let iy = iy - plan.ph;
                            let row_out = &mut out_c_slice[oy * plan.out_w..(oy + 1) * plan.out_w];
                            let row_in = &x[xbase + iy * plan.in_w..xbase + (iy + 1) * plan.in_w];
                            for (ox, o) in row_out.iter_mut().enumerate() {
                                let ix = ox * plan.sw + kx;
                                if ix < plan.pw {
                                    continue;
                                }
                                let ix = ix - plan.pw;
                                if ix >= plan.in_w {
                                    continue;
                                }
                                let v = row_in[ix];
                                *o = dd_fma(*o, wv, v.hi);
                                *o = dd_fma(*o, wv, v.lo);
                            }
                        }
                    }
                }
            }
        });
    Some(out)
}

/// Plain-f64 dense matvec `out = W x (+ bias)`; `w` is C-order `(out, in)`.
pub(crate) fn linear_f64(
    out_features: usize,
    in_features: usize,
    w: &[f64],
    bias: Option<&[f64]>,
    x: &[f64],
) -> Option<Vec<f64>> {
    let weight_numel = out_features.checked_mul(in_features)?;
    if out_features == 0
        || in_features == 0
        || w.len() != weight_numel
        || x.len() != in_features
        || bias.is_some_and(|values| values.len() != out_features)
    {
        return None;
    }
    let mut out = vec![0.0_f64; out_features];
    out.par_iter_mut().enumerate().for_each(|(o, dst)| {
        let row = &w[o * in_features..(o + 1) * in_features];
        let mut acc = bias.map_or(0.0, |b| b[o]);
        for (wv, xv) in row.iter().zip(x.iter()) {
            acc += wv * xv;
        }
        *dst = acc;
    });
    Some(out)
}

/// Double-double dense matvec over a double-double input center.
pub(crate) fn linear_dd(
    out_features: usize,
    in_features: usize,
    w: &[f64],
    bias: Option<&[f64]>,
    x: &[Dd],
) -> Option<Vec<Dd>> {
    let weight_numel = out_features.checked_mul(in_features)?;
    if out_features == 0
        || in_features == 0
        || w.len() != weight_numel
        || x.len() != in_features
        || bias.is_some_and(|values| values.len() != out_features)
    {
        return None;
    }
    let mut out = vec![Dd::ZERO; out_features];
    out.par_iter_mut().enumerate().for_each(|(o, dst)| {
        let row = &w[o * in_features..(o + 1) * in_features];
        let mut acc = Dd::from_f64(bias.map_or(0.0, |b| b[o]));
        for (wv, xv) in row.iter().zip(x.iter()) {
            acc = dd_fma(acc, *wv, xv.hi);
            acc = dd_fma(acc, *wv, xv.lo);
        }
        *dst = acc;
    });
    Some(out)
}

/// The abstract affine map an op presents to [`apply_affine`].
/// Variant sizes intentionally differ (Conv carries its plan inline); the enum
/// is short-lived and stack-allocated per layer, so boxing buys nothing.
#[allow(clippy::large_enum_variant)]
pub(crate) enum AffineOp<'a> {
    Conv {
        plan: ConvPlan,
        /// C-order `(out_c, in_c, kh, kw)`, exact f64 lift of the f32 kernel.
        w: &'a [f64],
        /// C-order `(out_c, in_c, kh, kw)` of `|w|`.
        wabs: &'a [f64],
        bias: Option<&'a [f64]>,
    },
    Linear {
        out_features: usize,
        in_features: usize,
        w: &'a [f64],
        wabs: &'a [f64],
        bias: Option<&'a [f64]>,
    },
}

impl AffineOp<'_> {
    fn out_shape(&self) -> Vec<usize> {
        match self {
            AffineOp::Conv { plan, .. } => vec![plan.out_c, plan.out_h, plan.out_w],
            AffineOp::Linear { out_features, .. } => vec![*out_features],
        }
    }

    /// Validate every shape and slice used by the low-level indexed kernels,
    /// returning checked `(input_numel, output_numel, dot_terms)`.
    fn preflight(&self, z: &DdZono) -> Option<(usize, usize, usize)> {
        if !z.has_valid_layout() {
            return None;
        }
        match self {
            AffineOp::Conv {
                plan,
                w,
                wabs,
                bias,
            } => {
                let sizes = plan.checked_sizes()?;
                if z.shape.as_slice() != [plan.in_c, plan.in_h, plan.in_w]
                    || z.numel() != sizes.input_numel
                    || w.len() != sizes.weight_numel
                    || wabs.len() != sizes.weight_numel
                    || bias.is_some_and(|values| values.len() != plan.out_c)
                {
                    return None;
                }
                Some((sizes.input_numel, sizes.output_numel, sizes.dot_terms))
            }
            AffineOp::Linear {
                out_features,
                in_features,
                w,
                wabs,
                bias,
            } => {
                let weight_numel = out_features.checked_mul(*in_features)?;
                if *out_features == 0
                    || *in_features == 0
                    || z.numel() != *in_features
                    || w.len() != weight_numel
                    || wabs.len() != weight_numel
                    || bias.is_some_and(|values| values.len() != *out_features)
                {
                    return None;
                }
                Some((*in_features, *out_features, *in_features))
            }
        }
    }

    fn bias(&self) -> Option<&[f64]> {
        match self {
            AffineOp::Conv { bias, .. } | AffineOp::Linear { bias, .. } => *bias,
        }
    }

    fn apply_f64(&self, x: &[f64], with_bias: bool) -> Option<Vec<f64>> {
        match self {
            AffineOp::Conv { plan, w, bias, .. } => {
                conv_f64(plan, w, if with_bias { *bias } else { None }, x)
            }
            AffineOp::Linear {
                out_features,
                in_features,
                w,
                bias,
                ..
            } => linear_f64(
                *out_features,
                *in_features,
                w,
                if with_bias { *bias } else { None },
                x,
            ),
        }
    }

    /// Apply `|W|` with no bias — the error-channel transfer operator.
    fn apply_abs(&self, x: &[f64]) -> Option<Vec<f64>> {
        match self {
            AffineOp::Conv { plan, wabs, .. } => conv_f64(plan, wabs, None, x),
            AffineOp::Linear {
                out_features,
                in_features,
                wabs,
                ..
            } => linear_f64(*out_features, *in_features, wabs, None, x),
        }
    }

    fn apply_dd(&self, x: &[Dd]) -> Option<Vec<Dd>> {
        match self {
            AffineOp::Conv { plan, w, bias, .. } => conv_dd(plan, w, *bias, x),
            AffineOp::Linear {
                out_features,
                in_features,
                w,
                bias,
                ..
            } => linear_dd(*out_features, *in_features, w, *bias, x),
        }
    }
}

/// Apply a certified affine map to a zonotope state.
///
/// `poll` is called between the generator columns so a long pass can honour a
/// wall-clock deadline; returning `Err` aborts without publishing anything.
pub(crate) fn apply_affine(
    z: &DdZono,
    op: &AffineOp<'_>,
    mut poll: impl FnMut() -> Result<()>,
) -> Result<DdZono> {
    let (_, n_out, k) = op.preflight(z).ok_or_else(|| {
        NyError::SoundnessRefusal("#dd-zonotope affine shape/slice preflight failed".to_string())
    })?;

    // --- error channel, computed BEFORE the state is consumed --------------
    let rad = z.radius();
    let abs_c: Vec<f64> = z.center.iter().map(|c| c.abs_upper()).collect();

    let kernel_refusal =
        || NyError::SoundnessRefusal("#dd-zonotope affine kernel validation failed".to_string());
    let s_c = op.apply_abs(&abs_c).ok_or_else(kernel_refusal)?;
    poll()?;
    let s_g = op.apply_abs(&rad).ok_or_else(kernel_refusal)?;
    poll()?;
    let ec_t = op.apply_abs(&z.ec).ok_or_else(kernel_refusal)?;
    poll()?;
    let eg_t = op.apply_abs(&z.eg).ok_or_else(kernel_refusal)?;
    poll()?;

    // gamma_{2K+2} at the double-double effective unit roundoff: each input
    // element enters the accumulator as two exact products, plus the bias add.
    let dd_gamma_terms = k
        .checked_mul(2)
        .and_then(|count| count.checked_add(2))
        .ok_or_else(|| {
            NyError::SoundnessRefusal(
                "#dd-zonotope center gamma operation count overflow".to_string(),
            )
        })?;
    let g_dd = gamma_n_dd(dd_gamma_terms);
    // gamma_{K+1} at plain f64 for the generator columns.
    let f64_gamma_terms = k.checked_add(1).ok_or_else(|| {
        NyError::SoundnessRefusal(
            "#dd-zonotope generator gamma operation count overflow".to_string(),
        )
    })?;
    let g_f64 = gamma_n_f64(f64_gamma_terms);
    let bias = op.bias();

    // Absolute floors cover every multiply/add/EFT/reduction, not just the K
    // mathematical products. In particular, generator errors aggregate over
    // every live column, and ec_t/eg_t each perform another plain-f64 dot.
    let (center_operations, generator_operations) =
        affine_underflow_operation_counts(k, z.n_gens())?;
    let uf_ec = operation_underflow_floor(center_operations)?;
    let uf_eg = operation_underflow_floor(generator_operations)?;

    let mut ec = vec![0.0_f64; n_out];
    let mut eg = vec![0.0_f64; n_out];
    for i in 0..n_out {
        let babs = bias.map_or(0.0, |b| {
            // Conv bias is per output channel; Linear bias is per output.
            match op {
                AffineOp::Conv { plan, .. } => b[i / (plan.out_h * plan.out_w)].abs(),
                AffineOp::Linear { .. } => b[i].abs(),
            }
        });
        ec[i] = err_up(ec_t[i] + g_dd * (s_c[i] + babs) + uf_ec);
        eg[i] = err_up(eg_t[i] + g_f64 * s_g[i] + uf_eg);
    }

    // --- values ------------------------------------------------------------
    let center = op.apply_dd(&z.center).ok_or_else(kernel_refusal)?;
    poll()?;
    let mut gens: Vec<Vec<f64>> = Vec::with_capacity(z.n_gens());
    for g in &z.gens {
        gens.push(op.apply_f64(g, false).ok_or_else(kernel_refusal)?);
        poll()?;
    }

    Ok(DdZono {
        shape: op.out_shape(),
        center,
        gens,
        ec,
        eg,
    })
}
