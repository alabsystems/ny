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
//! ec'     = |W| ec + gamma_{2K+2}(U_DD) * (|W| |center| + |b|)
//! eg'     = |W| eg + gamma_{K+1}(U_F64) * (|W| radius)
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
//!
//! Every error-channel value is rounded OUTWARD (`err_up`) after each step.

use ny_core::dd::{dd_fma, gamma_n_dd, gamma_n_f64, Dd};
use ny_core::{NyError, Result};
use rayon::prelude::*;

use super::state::{err_up, DdZono};

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

impl ConvPlan {
    /// Dot length per output element (`in_c * kh * kw`).
    #[inline]
    pub(crate) fn k(&self) -> usize {
        self.in_c * self.kh * self.kw
    }

    #[inline]
    pub(crate) fn out_numel(&self) -> usize {
        self.out_c * self.out_h * self.out_w
    }

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
        if dilation != (1, 1) || groups != 1 || stride.0 == 0 || stride.1 == 0 {
            return None;
        }
        let (in_c, in_h, in_w) = in_shape;
        let num_h = in_h + 2 * padding.0;
        let num_w = in_w + 2 * padding.1;
        if num_h < kh || num_w < kw {
            return None;
        }
        let out_h = (num_h - kh) / stride.0 + 1;
        let out_w = (num_w - kw) / stride.1 + 1;
        if out_h == 0 || out_w == 0 || out_c == 0 {
            return None;
        }
        Some(ConvPlan {
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
        })
    }
}

/// Plain-f64 direct convolution: `out = W x (+ bias)`.
///
/// `w` is C-order `(out_c, in_c, kh, kw)`. Parallel over output channels; the
/// innermost loop is a contiguous AXPY over the output row so the backend can
/// vectorize it.
pub(crate) fn conv_f64(plan: &ConvPlan, w: &[f64], bias: Option<&[f64]>, x: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0_f64; plan.out_numel()];
    let ohw = plan.out_h * plan.out_w;
    let ihw = plan.in_h * plan.in_w;
    let kk = plan.kh * plan.kw;
    out.par_chunks_mut(ohw)
        .enumerate()
        .for_each(|(oc, out_c_slice)| {
            let b = bias.map_or(0.0, |b| b[oc]);
            out_c_slice.fill(b);
            for ic in 0..plan.in_c {
                let wbase = (oc * plan.in_c + ic) * kk;
                let xbase = ic * ihw;
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
    out
}

/// Double-double direct convolution over a double-double input center.
///
/// Each input element contributes TWO exact products (`w*hi`, `w*lo`), so the
/// accumulator sees a dot of length `2K`; the caller's `gamma_{2K+2}(U_DD)`
/// term is sized for exactly that.
pub(crate) fn conv_dd(plan: &ConvPlan, w: &[f64], bias: Option<&[f64]>, x: &[Dd]) -> Vec<Dd> {
    let mut out = vec![Dd::ZERO; plan.out_numel()];
    let ohw = plan.out_h * plan.out_w;
    let ihw = plan.in_h * plan.in_w;
    let kk = plan.kh * plan.kw;
    out.par_chunks_mut(ohw)
        .enumerate()
        .for_each(|(oc, out_c_slice)| {
            let b = bias.map_or(0.0, |b| b[oc]);
            for o in out_c_slice.iter_mut() {
                *o = Dd::from_f64(b);
            }
            for ic in 0..plan.in_c {
                let wbase = (oc * plan.in_c + ic) * kk;
                let xbase = ic * ihw;
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
    out
}

/// Plain-f64 dense matvec `out = W x (+ bias)`; `w` is C-order `(out, in)`.
pub(crate) fn linear_f64(
    out_features: usize,
    in_features: usize,
    w: &[f64],
    bias: Option<&[f64]>,
    x: &[f64],
) -> Vec<f64> {
    let mut out = vec![0.0_f64; out_features];
    out.par_iter_mut().enumerate().for_each(|(o, dst)| {
        let row = &w[o * in_features..(o + 1) * in_features];
        let mut acc = bias.map_or(0.0, |b| b[o]);
        for (wv, xv) in row.iter().zip(x.iter()) {
            acc += wv * xv;
        }
        *dst = acc;
    });
    out
}

/// Double-double dense matvec over a double-double input center.
pub(crate) fn linear_dd(
    out_features: usize,
    in_features: usize,
    w: &[f64],
    bias: Option<&[f64]>,
    x: &[Dd],
) -> Vec<Dd> {
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
    out
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

    fn out_numel(&self) -> usize {
        match self {
            AffineOp::Conv { plan, .. } => plan.out_numel(),
            AffineOp::Linear { out_features, .. } => *out_features,
        }
    }

    /// Dot length `K` per output element.
    fn k(&self) -> usize {
        match self {
            AffineOp::Conv { plan, .. } => plan.k(),
            AffineOp::Linear { in_features, .. } => *in_features,
        }
    }

    fn in_numel(&self) -> usize {
        match self {
            AffineOp::Conv { plan, .. } => plan.in_c * plan.in_h * plan.in_w,
            AffineOp::Linear { in_features, .. } => *in_features,
        }
    }

    fn bias(&self) -> Option<&[f64]> {
        match self {
            AffineOp::Conv { bias, .. } | AffineOp::Linear { bias, .. } => *bias,
        }
    }

    fn apply_f64(&self, x: &[f64], with_bias: bool) -> Vec<f64> {
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
    fn apply_abs(&self, x: &[f64]) -> Vec<f64> {
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

    fn apply_dd(&self, x: &[Dd]) -> Vec<Dd> {
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
    if z.numel() != op.in_numel() {
        return Err(NyError::InvalidSpec(format!(
            "#dd-zonotope affine arity mismatch: state has {} elements, op expects {}",
            z.numel(),
            op.in_numel()
        )));
    }
    let n_out = op.out_numel();
    let k = op.k();

    // --- error channel, computed BEFORE the state is consumed --------------
    let rad = z.radius();
    let abs_c: Vec<f64> = z.center.iter().map(|c| c.abs_upper()).collect();

    let s_c = op.apply_abs(&abs_c);
    poll()?;
    let s_g = op.apply_abs(&rad);
    poll()?;
    let ec_t = op.apply_abs(&z.ec);
    poll()?;
    let eg_t = op.apply_abs(&z.eg);
    poll()?;

    // gamma_{2K+2} at the double-double effective unit roundoff: each input
    // element enters the accumulator as two exact products, plus the bias add.
    let g_dd = gamma_n_dd(2 * k + 2);
    // gamma_{K+1} at plain f64 for the generator columns.
    let g_f64 = gamma_n_f64(k + 1);
    let bias = op.bias();

    // ABSOLUTE underflow floor (soundness audit 2026-07-23). `two_prod` is
    // error-free ONLY when the product does not underflow (< f64::MIN_POSITIVE);
    // a subnormal/flushed product drops a residual < 2^-1074 that the RELATIVE
    // gamma envelope (gamma*S, and S itself underflows) cannot see. Charge an
    // absolute floor of one lost residual per accumulated op. UNREACHABLE on
    // vggnet16 (f32 weights >= 2^-149, O(1) activations -> products O(1)),
    // negligible when reached (~1e-320), and sound when it is not -- fail-closed.
    let eta = f64::from_bits(1); // smallest positive subnormal, 2^-1074
    #[allow(clippy::cast_precision_loss)]
    let uf_ec = (2 * k + 2) as f64 * eta; // dd center dot: 2K products + bias
    #[allow(clippy::cast_precision_loss)]
    let uf_eg = (k + 1) as f64 * eta; // plain-f64 generator dot: K products

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
    let center = op.apply_dd(&z.center);
    poll()?;
    let mut gens: Vec<Vec<f64>> = Vec::with_capacity(z.n_gens());
    for g in &z.gens {
        gens.push(op.apply_f64(g, false));
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
