// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Compiled twin-wall net: validated op list + conv row-apply kernels
//! (#twinwall).
//!
//! Row matrices are `(n_neurons, R)` f64 arrays: row `i` is the length-`R`
//! payload of neuron `i` (R = backward-pass row count, tableau columns, or a
//! point batch). Conv application in either direction is a weight-stationary
//! loop over an index table with contiguous length-`R` axpys — the same
//! *verified gather semantics* as the Python reference (`core.py::_gather_idx`
//! / `conv_sparse`), without materializing a CSR.

use ndarray::{Array2, Axis};
use ny_core::{NyError, Result};
use rayon::prelude::*;
use std::sync::OnceLock;

use super::spec::{TwinOpSpec, TwinSpec};

/// Compiled conv: forward gather table + transposed (backward) tap table.
#[derive(Clone)]
pub struct ConvOp {
    /// Consumed tensor id.
    pub input: usize,
    /// (cout, cin, kh, kw).
    pub kernel: (usize, usize, usize, usize),
    /// Input (C, H, W) / output (C, H, W).
    pub ishape: (usize, usize, usize),
    /// Output shape.
    pub oshape: (usize, usize, usize),
    /// `(stride_h, stride_w)` of the ORIGINAL spec op.
    ///
    /// GEOMETRY METADATA ONLY: every kernel in this file reads the compiled
    /// `gather`/`back_taps` tables, which already encode the stride. This field
    /// exists so the default-dark GPU seam (`super::gpu_seam`) can rebuild the
    /// equivalent `GpuCrownLayer::Conv2d` descriptor without re-deriving the
    /// geometry from the tap tables. No bound reads it.
    pub stride: (usize, usize),
    /// `(pad_top, pad_left, pad_bottom, pad_right)` of the ORIGINAL spec op.
    /// Geometry metadata only, exactly like [`ConvOp::stride`].
    pub pads: (usize, usize, usize, usize),
    /// True when this op was compiled from a `ConvTranspose` spec op (the
    /// transpose-aware gather tables make it indistinguishable downstream).
    /// Geometry metadata only; the GPU seam REFUSES transposed convs because
    /// `GpuCrownLayer` has no transposed-conv variant.
    pub transposed: bool,
    /// Kernel `[cout][cin*kh*kw]` row-major (weight-stationary forward).
    pub wmat: Vec<f64>,
    /// Kernel transposed `[cin][kh][kw][cout]` (contiguous over cout;
    /// weight-stationary backward).
    pub wt: Vec<f64>,
    /// Per-output-channel bias.
    pub bias: Vec<f64>,
    /// Certified absolute per-channel bias error (vs exact real fold).
    pub bias_err: Vec<f64>,
    /// Certified relative weight error (vs exact real fold).
    pub weight_rel_err: f64,
    /// Forward gather: for each spatial out position `p` (of `P = Ho*Wo`),
    /// `K = cin*kh*kw` flat input indices (`usize::MAX` = zero padding).
    pub gather: Vec<usize>,
    /// Backward taps: for each spatial in position `(ih, iw)`, the valid
    /// `(kh, kw, out_spatial)` triples.
    pub back_taps: Vec<Vec<(usize, usize, usize)>>,
    /// Max accumulation terms per forward output row (for `gamma_n`).
    pub k_fwd: usize,
    /// Max accumulation terms per backward output row.
    pub k_bwd: usize,
}

/// One compiled op.
pub enum TwinOp {
    /// Convolution (BN folded). Also the compiled form of `ConvTranspose`
    /// (transpose-aware gather tables, same kernels; #epoch-bab Phase D).
    Conv(Box<ConvOp>),
    /// Per-channel affine (standalone BN, inference form; #epoch-bab).
    ChannelAffine {
        /// Consumed tensor id.
        input: usize,
        /// Per-NEURON scale (channel value broadcast over H*W).
        scale: Vec<f64>,
        /// Per-NEURON shift.
        shift: Vec<f64>,
        /// Certified relative error on `scale`.
        scale_rel_err: f64,
        /// Certified absolute per-NEURON error on `shift`.
        shift_err: Vec<f64>,
    },
    /// ReLU consuming `input`; `layer` = trunk relu index (usize::MAX = head).
    Relu {
        /// Consumed tensor id.
        input: usize,
        /// Trunk relu layer index.
        layer: usize,
    },
    /// Elementwise add.
    Add {
        /// Left tensor id.
        lhs: usize,
        /// Right tensor id.
        rhs: usize,
    },
    /// Flatten (flat identity).
    Flatten {
        /// Consumed tensor id.
        input: usize,
    },
    /// Dense layer (only the two head Gemms survive validation).
    Gemm {
        /// Consumed tensor id.
        input: usize,
        /// Row-major (n_out, n_in).
        weight: Vec<f64>,
        /// Bias.
        bias: Vec<f64>,
        /// (n_out, n_in).
        shape: (usize, usize),
    },
}

/// Compiled twin-wall network.
pub struct TwinNet {
    /// Flat input size.
    pub n_in: usize,
    /// Ops in execution order (tensor `k+1` = output of op `k`).
    pub ops: Vec<TwinOp>,
    /// Flat size of every tensor id (0 = input).
    pub tsize: Vec<usize>,
    /// Index of the first (head) Gemm op.
    pub i_gemm1: usize,
    /// Trunk relu op indices in execution order.
    pub trunk_relus: Vec<usize>,
    /// Head width (rows of the first Gemm).
    pub n_y: usize,
    /// Output classes (rows of the final Gemm).
    pub n_out: usize,
}

impl TwinNet {
    /// Compile and validate a [`TwinSpec`]. Fails closed on any structural
    /// deviation from the twin-wall family.
    pub fn compile(spec: &TwinSpec) -> Result<Self> {
        let bad = |m: &str| NyError::InvalidSpec(format!("margin_row twin net: {m}"));
        if spec.n_in == 0 || spec.ops.is_empty() {
            return Err(bad("empty spec"));
        }
        let mut tsize = vec![spec.n_in];
        let mut ops = Vec::with_capacity(spec.ops.len());
        let mut gemm_idx = Vec::new();
        let mut trunk_relus = Vec::new();
        for (k, op) in spec.ops.iter().enumerate() {
            let out_id = k + 1;
            let check_id = |id: usize| -> Result<usize> {
                if id >= out_id {
                    return Err(bad(&format!("op {k} references future tensor {id}")));
                }
                Ok(tsize[id])
            };
            match op {
                TwinOpSpec::Conv {
                    input,
                    weight,
                    bias,
                    bias_err,
                    weight_rel_err,
                    kernel,
                    stride,
                    pads,
                    ishape,
                    oshape,
                } => {
                    let in_sz = check_id(*input)?;
                    let (co, ci, kh, kw) = *kernel;
                    let (ic, ih, iw) = *ishape;
                    let (oc, oh, ow) = *oshape;
                    if in_sz != ic * ih * iw
                        || ci != ic
                        || co != oc
                        || weight.len() != co * ci * kh * kw
                        || bias.len() != co
                        || bias_err.len() != co
                        || bias_err.iter().any(|e| !e.is_finite() || *e < 0.0)
                        || !weight_rel_err.is_finite()
                        || *weight_rel_err < 0.0
                        || stride.0 == 0
                        || stride.1 == 0
                    {
                        return Err(bad(&format!("conv {k} inconsistent geometry")));
                    }
                    // Output geometry must match the conv formula exactly.
                    let eh = (ih + pads.0 + pads.2)
                        .checked_sub(kh)
                        .map(|v| v / stride.0 + 1);
                    let ew = (iw + pads.1 + pads.3)
                        .checked_sub(kw)
                        .map(|v| v / stride.1 + 1);
                    if eh != Some(oh) || ew != Some(ow) {
                        return Err(bad(&format!("conv {k} output shape mismatch")));
                    }
                    if weight.iter().any(|w| !w.is_finite()) || bias.iter().any(|b| !b.is_finite())
                    {
                        return Err(bad(&format!("conv {k} non-finite parameters")));
                    }
                    ops.push(TwinOp::Conv(Box::new(compile_conv(
                        *input,
                        weight,
                        bias,
                        bias_err,
                        *weight_rel_err,
                        *kernel,
                        *stride,
                        *pads,
                        *ishape,
                        *oshape,
                    ))));
                    tsize.push(oc * oh * ow);
                }
                TwinOpSpec::ConvTranspose {
                    input,
                    weight,
                    bias,
                    bias_err,
                    weight_rel_err,
                    kernel,
                    stride,
                    pads,
                    ishape,
                    oshape,
                    out_pad,
                } => {
                    let in_sz = check_id(*input)?;
                    let (co, ci, kh, kw) = *kernel;
                    let (ic, ih, iw) = *ishape;
                    let (oc, oh, ow) = *oshape;
                    if in_sz != ic * ih * iw
                        || ci != ic
                        || co != oc
                        || weight.len() != co * ci * kh * kw
                        || bias.len() != co
                        || bias_err.len() != co
                        || bias_err.iter().any(|e| !e.is_finite() || *e < 0.0)
                        || !weight_rel_err.is_finite()
                        || *weight_rel_err < 0.0
                        || stride.0 == 0
                        || stride.1 == 0
                    {
                        return Err(bad(&format!("convtranspose {k} inconsistent geometry")));
                    }
                    // Output geometry: (ih-1)*s - pt - pb + kh + opad.
                    let eh = ((ih - 1) * stride.0 + kh + out_pad.0).checked_sub(pads.0 + pads.2);
                    let ew = ((iw - 1) * stride.1 + kw + out_pad.1).checked_sub(pads.1 + pads.3);
                    if eh != Some(oh) || ew != Some(ow) {
                        return Err(bad(&format!("convtranspose {k} output shape mismatch")));
                    }
                    if weight.iter().any(|w| !w.is_finite()) || bias.iter().any(|b| !b.is_finite())
                    {
                        return Err(bad(&format!("convtranspose {k} non-finite parameters")));
                    }
                    ops.push(TwinOp::Conv(Box::new(compile_conv_transpose(
                        *input,
                        weight,
                        bias,
                        bias_err,
                        *weight_rel_err,
                        *kernel,
                        *stride,
                        *pads,
                        *ishape,
                        *oshape,
                    ))));
                    tsize.push(oc * oh * ow);
                }
                TwinOpSpec::ChannelAffine {
                    input,
                    scale,
                    shift,
                    scale_rel_err,
                    shift_err,
                    shape,
                } => {
                    let in_sz = check_id(*input)?;
                    let (c, h, w) = *shape;
                    if in_sz != c * h * w
                        || scale.len() != c
                        || shift.len() != c
                        || shift_err.len() != c
                        || shift_err.iter().any(|e| !e.is_finite() || *e < 0.0)
                        || !scale_rel_err.is_finite()
                        || *scale_rel_err < 0.0
                        || scale.iter().any(|v| !v.is_finite())
                        || shift.iter().any(|v| !v.is_finite())
                    {
                        return Err(bad(&format!("channel_affine {k} inconsistent parameters")));
                    }
                    // Broadcast per-channel values to per-neuron vectors so
                    // every consumer is a flat diagonal op.
                    let hw = h * w;
                    let mut sc = Vec::with_capacity(in_sz);
                    let mut sh = Vec::with_capacity(in_sz);
                    let mut se = Vec::with_capacity(in_sz);
                    for ch in 0..c {
                        sc.extend(std::iter::repeat_n(scale[ch], hw));
                        sh.extend(std::iter::repeat_n(shift[ch], hw));
                        se.extend(std::iter::repeat_n(shift_err[ch], hw));
                    }
                    ops.push(TwinOp::ChannelAffine {
                        input: *input,
                        scale: sc,
                        shift: sh,
                        scale_rel_err: *scale_rel_err,
                        shift_err: se,
                    });
                    tsize.push(in_sz);
                }
                TwinOpSpec::Relu { input } => {
                    let sz = check_id(*input)?;
                    let layer = if gemm_idx.is_empty() {
                        trunk_relus.push(k);
                        trunk_relus.len() - 1
                    } else {
                        usize::MAX
                    };
                    ops.push(TwinOp::Relu {
                        input: *input,
                        layer,
                    });
                    tsize.push(sz);
                }
                TwinOpSpec::Add { lhs, rhs } => {
                    let a = check_id(*lhs)?;
                    let b = check_id(*rhs)?;
                    if a != b {
                        return Err(bad(&format!("add {k} size mismatch {a} vs {b}")));
                    }
                    ops.push(TwinOp::Add {
                        lhs: *lhs,
                        rhs: *rhs,
                    });
                    tsize.push(a);
                }
                TwinOpSpec::Flatten { input } => {
                    let sz = check_id(*input)?;
                    ops.push(TwinOp::Flatten { input: *input });
                    tsize.push(sz);
                }
                TwinOpSpec::Gemm {
                    input,
                    weight,
                    bias,
                    shape,
                } => {
                    let in_sz = check_id(*input)?;
                    let (no, ni) = *shape;
                    if ni != in_sz || weight.len() != no * ni || bias.len() != no {
                        return Err(bad(&format!("gemm {k} shape mismatch")));
                    }
                    if weight.iter().any(|w| !w.is_finite()) || bias.iter().any(|b| !b.is_finite())
                    {
                        return Err(bad(&format!("gemm {k} non-finite parameters")));
                    }
                    gemm_idx.push(k);
                    ops.push(TwinOp::Gemm {
                        input: *input,
                        weight: weight.clone(),
                        bias: bias.clone(),
                        shape: *shape,
                    });
                    tsize.push(no);
                }
            }
        }
        // Head structure: ... Flatten -> Gemm1 -> Relu -> Gemm2 (last op).
        if gemm_idx.len() != 2 {
            return Err(bad(&format!("expected 2 gemms, got {}", gemm_idx.len())));
        }
        let i_gemm1 = gemm_idx[0];
        let i_gemm2 = gemm_idx[1];
        if i_gemm2 != spec.ops.len() - 1
            || i_gemm1 + 2 != i_gemm2
            || !matches!(ops.get(i_gemm1 + 1), Some(TwinOp::Relu { .. }))
        {
            return Err(bad("head is not Gemm -> Relu -> Gemm at the tail"));
        }
        // The head relu must consume gemm1's output and gemm2 the head relu's.
        match (&ops[i_gemm1 + 1], &ops[i_gemm2]) {
            (TwinOp::Relu { input: r_in, .. }, TwinOp::Gemm { input: g2_in, .. })
                if *r_in == i_gemm1 + 1 && *g2_in == i_gemm1 + 2 => {}
            _ => return Err(bad("head wiring mismatch")),
        }
        // Trunk ops before gemm1 must be conv/relu/add/flatten only (implied by
        // gemm_idx), and each trunk relu must have unstable-capable width.
        if trunk_relus.is_empty() {
            return Err(bad("no trunk relus"));
        }
        let n_y = match &ops[i_gemm1] {
            TwinOp::Gemm { shape, .. } => shape.0,
            _ => return Err(bad("gemm1 missing")),
        };
        let n_out = match &ops[i_gemm2] {
            TwinOp::Gemm { shape, .. } => shape.0,
            _ => return Err(bad("gemm2 missing")),
        };
        Ok(Self {
            n_in: spec.n_in,
            ops,
            tsize,
            i_gemm1,
            trunk_relus,
            n_y,
            n_out,
        })
    }

    /// First head Gemm (weights row-major (n_y, n_trunk_out), bias).
    pub fn gemm1(&self) -> (&[f64], &[f64], (usize, usize)) {
        match &self.ops[self.i_gemm1] {
            TwinOp::Gemm {
                weight,
                bias,
                shape,
                ..
            } => (weight, bias, *shape),
            _ => unreachable!("validated at compile"),
        }
    }

    /// Final Gemm (weights row-major (n_out, n_y), bias).
    pub fn gemm2(&self) -> (&[f64], &[f64], (usize, usize)) {
        match &self.ops[self.i_gemm1 + 2] {
            TwinOp::Gemm {
                weight,
                bias,
                shape,
                ..
            } => (weight, bias, *shape),
            _ => unreachable!("validated at compile"),
        }
    }

    /// Exact f64 forward evaluation of a point batch up to the head
    /// pre-activation `y` (points are COLUMNS: `x` is `(n_in, B)`).
    /// `pre_sel`: per relu-op index, neuron indices whose PRE-activations to
    /// collect (for split-membership filters). Returns `(y (n_y, B), pre)`.
    pub fn forward_points(
        &self,
        x: &Array2<f64>,
        pre_sel: &std::collections::BTreeMap<usize, Vec<usize>>,
    ) -> Result<(Array2<f64>, std::collections::BTreeMap<usize, Array2<f64>>)> {
        let b = x.ncols();
        if x.nrows() != self.n_in {
            return Err(NyError::shape_mismatch(vec![self.n_in], vec![x.nrows()]));
        }
        let mut vals: Vec<Option<Array2<f64>>> = vec![None; self.ops.len() + 1];
        vals[0] = Some(x.clone());
        let mut pre = std::collections::BTreeMap::new();
        for (k, op) in self.ops.iter().enumerate() {
            let out = match op {
                TwinOp::Conv(c) => {
                    let src = vals[c.input].as_ref().expect("topo order");
                    let mut dst = Array2::zeros((self.tsize[k + 1], b));
                    conv_apply_forward(c, src, &mut dst, false);
                    // add bias per out channel
                    let p = c.oshape.1 * c.oshape.2;
                    for (j, mut row) in dst.outer_iter_mut().enumerate() {
                        let ch = j / p;
                        let bias = c.bias[ch];
                        for v in &mut row {
                            *v += bias;
                        }
                    }
                    dst
                }
                TwinOp::Relu { input, .. } => {
                    let src = vals[*input].as_ref().expect("topo order");
                    if let Some(sel) = pre_sel.get(&k) {
                        let mut collected = Array2::zeros((sel.len(), b));
                        for (r, &idx) in sel.iter().enumerate() {
                            collected.row_mut(r).assign(&src.row(idx));
                        }
                        pre.insert(k, collected);
                    }
                    src.mapv(|v| v.max(0.0))
                }
                TwinOp::Add { lhs, rhs } => {
                    let a = vals[*lhs].as_ref().expect("topo order");
                    let c = vals[*rhs].as_ref().expect("topo order");
                    a + c
                }
                TwinOp::Flatten { input } => vals[*input].as_ref().expect("topo order").clone(),
                TwinOp::ChannelAffine {
                    input,
                    scale,
                    shift,
                    ..
                } => {
                    let src = vals[*input].as_ref().expect("topo order");
                    let mut dst = src.clone();
                    for (j, mut row) in dst.outer_iter_mut().enumerate() {
                        for v in &mut row {
                            *v = scale[j] * *v + shift[j];
                        }
                    }
                    dst
                }
                TwinOp::Gemm {
                    input,
                    weight,
                    bias,
                    shape,
                } => {
                    let src = vals[*input].as_ref().expect("topo order");
                    let (no, ni) = *shape;
                    let mut dst = Array2::zeros((no, b));
                    for o in 0..no {
                        let wrow = &weight[o * ni..(o + 1) * ni];
                        let mut acc = vec![bias[o]; b];
                        for (i, &w) in wrow.iter().enumerate() {
                            if w != 0.0 {
                                let srow = src.row(i);
                                let srow = srow.as_slice().expect("standard layout");
                                for (a, s) in acc.iter_mut().zip(srow) {
                                    *a += w * s;
                                }
                            }
                        }
                        dst.row_mut(o).assign(&ndarray::ArrayView1::from(&acc[..]));
                    }
                    dst
                }
            };
            if k == self.i_gemm1 {
                return Ok((out, pre));
            }
            vals[k + 1] = Some(out);
        }
        unreachable!("gemm1 always reached")
    }
}

#[allow(clippy::too_many_arguments)]
fn compile_conv(
    input: usize,
    weight: &[f64],
    bias: &[f64],
    bias_err: &[f64],
    weight_rel_err: f64,
    kernel: (usize, usize, usize, usize),
    stride: (usize, usize),
    pads: (usize, usize, usize, usize),
    ishape: (usize, usize, usize),
    oshape: (usize, usize, usize),
) -> ConvOp {
    let (co, ci, kh, kw) = kernel;
    let (_, ih, iw) = ishape;
    let (_, oh, ow) = oshape;
    let k = ci * kh * kw;
    let p = oh * ow;
    // Forward gather table: verified semantics of core.py::_gather_idx —
    // for out spatial (oy, ox) and tap (c, ky, kx): in index or sentinel.
    let mut gather = vec![usize::MAX; k * p];
    for oy in 0..oh {
        for ox in 0..ow {
            let sp = oy * ow + ox;
            for c in 0..ci {
                for ky in 0..kh {
                    for kx in 0..kw {
                        let iy = (oy * stride.0 + ky) as isize - pads.0 as isize;
                        let ix = (ox * stride.1 + kx) as isize - pads.1 as isize;
                        if iy >= 0 && (iy as usize) < ih && ix >= 0 && (ix as usize) < iw {
                            let idx = c * ih * iw + (iy as usize) * iw + ix as usize;
                            gather[(c * kh * kw + ky * kw + kx) * p + sp] = idx;
                        }
                    }
                }
            }
        }
    }
    // Backward taps per in spatial position: (ky, kx, out_spatial).
    let mut back_taps = vec![Vec::new(); ih * iw];
    for iy in 0..ih {
        for ix in 0..iw {
            let taps = &mut back_taps[iy * iw + ix];
            for ky in 0..kh {
                for kx in 0..kw {
                    let ny = iy + pads.0;
                    let nx = ix + pads.1;
                    if ny < ky || nx < kx {
                        continue;
                    }
                    let (dy, dx) = (ny - ky, nx - kx);
                    if dy % stride.0 != 0 || dx % stride.1 != 0 {
                        continue;
                    }
                    let (oy, ox) = (dy / stride.0, dx / stride.1);
                    if oy < oh && ox < ow {
                        taps.push((ky, kx, oy * ow + ox));
                    }
                }
            }
        }
    }
    // Transposed kernel [ci][kh][kw][co].
    let mut wt = vec![0.0; k * co];
    for o in 0..co {
        for c in 0..ci {
            for ky in 0..kh {
                for kx in 0..kw {
                    wt[((c * kh + ky) * kw + kx) * co + o] =
                        weight[((o * ci + c) * kh + ky) * kw + kx];
                }
            }
        }
    }
    let k_bwd = back_taps.iter().map(Vec::len).max().unwrap_or(0) * co;
    ConvOp {
        input,
        kernel,
        ishape,
        oshape,
        stride,
        pads,
        transposed: false,
        wmat: weight.to_vec(),
        wt,
        bias: bias.to_vec(),
        bias_err: bias_err.to_vec(),
        weight_rel_err,
        gather,
        back_taps,
        k_fwd: k,
        k_bwd,
    }
}

/// Default-on GEMM (im2col + matrix multiply) forward-conv path for the wide
/// root tableau. Set `NY_ROOT_BLAS=0` to use the bit-identical scalar fallback.
/// Linux/AArch64 defaults to faer's Rayon-parallel f64 kernel after its clean
/// CIFAR row-4 A/B reduced root construction by 40.5% with identical proof
/// statistics and result bytes. Other targets retain `ndarray::dot` as the
/// default. `NY_ROOT_GEMM=ndarray` is the portable fallback; explicit `faer`
/// remains available on every target, while unknown/empty values fail safe to
/// ndarray.
/// Provably-OUTWARD, not bit-identical: the GEMM reorders the
/// `k`-term contraction, but the twin-wall D-lane's `γ_n(k_fwd+·)` envelope
/// (built in `root.rs`) already dominates the error of ANY summation order
/// (Higham) — the identical order-independence certificate that
/// `sound_f64_gemm` / `mat_mul_f64` rely on for verdict-grade CROWN backward.
///
/// This is scoped to the margin-row lane and, at the call site, to wide
/// tableaus (`r >= 512 && k >= 2`). It became the production default after the
/// exact-rational enclosure gate passed and the legacy banked TinyImageNet
/// exact-timeout sweep showed that the scalar path misses its root-tableau
/// deadline. Backend promotions remain target-specific and evidence-gated.
pub(super) fn blas_conv_enabled_from_env(value: Option<&str>) -> bool {
    !value.is_some_and(|v| {
        let v = v.trim();
        v == "0"
            || v.eq_ignore_ascii_case("false")
            || v.eq_ignore_ascii_case("off")
            || v.eq_ignore_ascii_case("no")
    })
}

/// Wide-root GEMM implementation. This selects only an arithmetic kernel; both
/// variants compute the same f64 contraction under the same Higham envelope.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum RootGemmBackend {
    /// `ndarray::dot` fallback (platform BLAS on macOS, matrixmultiply on the
    /// platform-neutral Linux build).
    Ndarray,
    /// faer blocked/SIMD f64 GEMM with NY's Rayon-aware parallel policy.
    Faer,
}

const DEFAULT_ROOT_GEMM_BACKEND: RootGemmBackend =
    if cfg!(all(target_os = "linux", target_arch = "aarch64")) {
        RootGemmBackend::Faer
    } else {
        RootGemmBackend::Ndarray
    };

/// Resolve the two root-kernel knobs without reading process-global state.
///
/// `NY_ROOT_BLAS=0|false|off|no` always wins and returns the scalar fallback.
/// Otherwise an unset `NY_ROOT_GEMM` selects the platform default (faer on
/// Linux/AArch64, ndarray elsewhere). Explicit `faer` and `ndarray` select that
/// backend on every target; empty and unknown values conservatively select
/// ndarray.
pub(super) fn root_gemm_backend_from_env(
    root_blas: Option<&str>,
    root_gemm: Option<&str>,
) -> Option<RootGemmBackend> {
    if !blas_conv_enabled_from_env(root_blas) {
        return None;
    }
    match root_gemm.map(str::trim) {
        None => Some(DEFAULT_ROOT_GEMM_BACKEND),
        Some(value) if value.eq_ignore_ascii_case("faer") => Some(RootGemmBackend::Faer),
        Some(value) if value.eq_ignore_ascii_case("ndarray") => Some(RootGemmBackend::Ndarray),
        Some(_) => Some(RootGemmBackend::Ndarray),
    }
}

fn root_gemm_backend() -> Option<RootGemmBackend> {
    root_gemm_backend_from_env(
        std::env::var("NY_ROOT_BLAS").ok().as_deref(),
        std::env::var("NY_ROOT_GEMM").ok().as_deref(),
    )
}

/// im2col tile budget in f64 elements (`NY_ROOT_BLAS_TILE` overrides). Caps the
/// materialized `Col[k, spb*r]` block so peak memory stays bounded on the wide
/// tinyimagenet tableau (`r = 9409`). Read-once (a memory knob, never a bound).
fn blas_tile_elems() -> usize {
    static T: OnceLock<usize> = OnceLock::new();
    *T.get_or_init(|| {
        std::env::var("NY_ROOT_BLAS_TILE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&t| t >= (1 << 20))
            .unwrap_or(64 << 20)
    })
}

/// Compile a ConvTranspose into a [`ConvOp`] with transpose-aware gather /
/// back-tap tables (#epoch-bab Phase D). Weights arrive in the SAME
/// `[cout][cin][kh][kw]` layout as `Conv`; only the index tables differ:
/// output `(oy, ox)` gathers input `(iy, ix) = ((oy + pt - ky)/s, ...)`
/// where the division is exact and in range, so all existing forward /
/// backward kernels and the `k_fwd`/`k_bwd` Higham error accounting apply
/// unchanged.
#[allow(clippy::too_many_arguments)]
fn compile_conv_transpose(
    input: usize,
    weight: &[f64],
    bias: &[f64],
    bias_err: &[f64],
    weight_rel_err: f64,
    kernel: (usize, usize, usize, usize),
    stride: (usize, usize),
    pads: (usize, usize, usize, usize),
    ishape: (usize, usize, usize),
    oshape: (usize, usize, usize),
) -> ConvOp {
    let (co, ci, kh, kw) = kernel;
    let (_, ih, iw) = ishape;
    let (_, oh, ow) = oshape;
    let k = ci * kh * kw;
    let p = oh * ow;
    // Forward gather: out (oy, ox), tap (c, ky, kx) -> in index when
    // oy + pt - ky = iy * s_h exactly (and in range).
    let mut gather = vec![usize::MAX; k * p];
    for oy in 0..oh {
        for ox in 0..ow {
            let sp = oy * ow + ox;
            for c in 0..ci {
                for ky in 0..kh {
                    for kx in 0..kw {
                        let ny = oy + pads.0;
                        let nx = ox + pads.1;
                        if ny < ky || nx < kx {
                            continue;
                        }
                        let (dy, dx) = (ny - ky, nx - kx);
                        if dy % stride.0 != 0 || dx % stride.1 != 0 {
                            continue;
                        }
                        let (iy, ix) = (dy / stride.0, dx / stride.1);
                        if iy < ih && ix < iw {
                            let idx = c * ih * iw + iy * iw + ix;
                            gather[(c * kh * kw + ky * kw + kx) * p + sp] = idx;
                        }
                    }
                }
            }
        }
    }
    // Backward taps: input (iy, ix) contributes to outputs
    // oy = iy * s - pt + ky (in range).
    let mut back_taps = vec![Vec::new(); ih * iw];
    for iy in 0..ih {
        for ix in 0..iw {
            let taps = &mut back_taps[iy * iw + ix];
            for ky in 0..kh {
                for kx in 0..kw {
                    let ny = iy * stride.0 + ky;
                    let nx = ix * stride.1 + kx;
                    if ny < pads.0 || nx < pads.1 {
                        continue;
                    }
                    let (oy, ox) = (ny - pads.0, nx - pads.1);
                    if oy < oh && ox < ow {
                        taps.push((ky, kx, oy * ow + ox));
                    }
                }
            }
        }
    }
    // Transposed kernel [ci][kh][kw][co] (same transform as compile_conv —
    // the weight layout is already [co][ci][kh][kw]).
    let mut wt = vec![0.0; k * co];
    for o in 0..co {
        for c in 0..ci {
            for ky in 0..kh {
                for kx in 0..kw {
                    wt[((c * kh + ky) * kw + kx) * co + o] =
                        weight[((o * ci + c) * kh + ky) * kw + kx];
                }
            }
        }
    }
    let k_bwd = back_taps.iter().map(Vec::len).max().unwrap_or(0) * co;
    ConvOp {
        input,
        kernel,
        ishape,
        oshape,
        stride,
        pads,
        transposed: true,
        wmat: weight.to_vec(),
        wt,
        bias: bias.to_vec(),
        bias_err: bias_err.to_vec(),
        weight_rel_err,
        gather,
        back_taps,
        k_fwd: k,
        k_bwd,
    }
}

/// Forward conv on row matrices: `dst[(oc, sp)] = sum_taps w * src[gather]`
/// (no bias). `abs_w` uses `|w|` (the D-lane of the forward tableau).
///
/// # Kernel selection
///
/// * **default** (only for the WIDE root tableau `r >= 512 && k >= 2`) —
///   [`conv_apply_forward_blas`]: im2col + ndarray GEMM. Set
///   `NY_ROOT_GEMM=faer` for the experimental faer kernel, or `NY_ROOT_BLAS=0`
///   for the scalar fallback. PROVABLY-OUTWARD (not bit-identical): GEMM
///   reorders the `k`-term contraction, but the twin-wall D-lane envelope in
///   `root.rs` dominates the reordered rounding, so the concretized box stays
///   an outward enclosure of the exact real value. Point evaluation (small
///   `r`) always stays on the scalar path.
///
/// Otherwise one of three interchangeable, **bit-identical** scalar kernels
/// selected by `NY_MARGIN_ROW_ROOT_PAR` (proven bit-identical by the grain
/// tests — they change only which thread owns each output row and the loop
/// nesting, never the per-output tap order: row `(oc, sp)` always folds taps
/// `t = 0..K` in ascending order, skipping zero weights / padded taps
/// identically, into a zero-initialised accumulator via the same non-fused
/// `acc += w*s` step):
///
/// * default (`unset`) — [`conv_forward_blocked`]: cache-blocked, parallel over
///   output spatial positions, each gathered input row REUSED across all output
///   channels (≈`co`× less memory traffic — the big-net win). This is the moat
///   oracle (bit-identical to the serial per-channel grain).
/// * `NY_MARGIN_ROW_ROOT_PAR=row` — [`conv_forward_rowgrain`]: one work unit per
///   output row.
/// * `NY_MARGIN_ROW_ROOT_PAR=0` — [`conv_forward_ocgrain`]: legacy per-channel
///   grain (also the bit-identity oracle).
pub fn conv_apply_forward(c: &ConvOp, src: &Array2<f64>, dst: &mut Array2<f64>, abs_w: bool) {
    let (co, _, _, _) = c.kernel;
    let p = c.oshape.1 * c.oshape.2;
    let k = c.k_fwd;
    let r = src.ncols();
    debug_assert_eq!(dst.nrows(), co * p);
    debug_assert_eq!(dst.ncols(), r);
    // Default-on GEMM im2col path, only for the WIDE root tableau (large r).
    // Point evaluation (`forward_points`, small r) stays on the scalar kernels
    // below. `NY_ROOT_BLAS=0` is the operational kill switch; faer is opt-in.
    if r >= 512 && k >= 2 {
        match root_gemm_backend() {
            Some(RootGemmBackend::Ndarray) => {
                conv_apply_forward_blas(c, src, dst, abs_w, co, p, k, r);
                return;
            }
            Some(RootGemmBackend::Faer) => {
                conv_apply_forward_faer(c, src, dst, abs_w, co, p, k, r);
                return;
            }
            None => {}
        }
    }
    match std::env::var("NY_MARGIN_ROW_ROOT_PAR").ok().as_deref() {
        Some("0") => conv_forward_ocgrain(c, src, dst, abs_w),
        Some("row") => conv_forward_rowgrain(c, src, dst, abs_w),
        _ => conv_forward_blocked(c, src, dst, abs_w),
    }
}

/// Precision-selectable forward conv (`NY_MARGIN_ROW_ROOT_F32` fast path).
///
/// `use_f32 = false` routes to [`conv_apply_forward`] (the verdict default:
/// f64, bit-identical to the serial oracle / BLAS-envelope path). `use_f32 =
/// true` runs the same weight-stationary gather in f32 (halved memory traffic
/// on the bandwidth-bound tableau convs) and writes the f32 accumulator back
/// into the f64 `dst` EXACTLY (widening f32->f64). The extra rounding this
/// introduces is NEVER absorbed into `dst` as a verdict coefficient; the
/// root-tableau `build` charges a certified additive concretize slack
/// (`gamma_n_f32(k_fwd+8) * input-magnitude`, accumulated) that dominates the
/// worst-case effect of this rounding on every concretized box endpoint.
pub fn conv_apply_forward_prec(
    c: &ConvOp,
    src: &Array2<f64>,
    dst: &mut Array2<f64>,
    abs_w: bool,
    use_f32: bool,
) {
    conv_apply_forward_prec_masked(c, src, dst, abs_w, use_f32, None);
}

/// [`conv_apply_forward_prec`] with the optional `#tableau-support-mask`
/// column-block filter (`root.rs`).
///
/// `mask` is `(src_row_masks, out_row_masks, n_blocks)`. Both mask slices carry
/// ONE `u64` per row of their tensor; bit `b` means "columns of block `b` MAY
/// be nonzero in this row". The caller owns the invariant that a clear bit
/// means the coefficients there are EXACTLY `0.0` — see the module comment in
/// `root.rs`. Under that invariant the skipped output blocks are exactly zero,
/// so the masked kernel is bit-identical to the dense one on every column it
/// does compute and correct (`dst` arrives zeroed) on every column it skips.
pub fn conv_apply_forward_prec_masked(
    c: &ConvOp,
    src: &Array2<f64>,
    dst: &mut Array2<f64>,
    abs_w: bool,
    use_f32: bool,
    mask: Option<(&[u64], &[u64], usize)>,
) {
    if use_f32 {
        conv_forward_blocked_f32(c, src, dst, abs_w, mask);
    } else {
        conv_apply_forward(c, src, dst, abs_w);
    }
}

/// f32 cache-blocked forward conv. Mirrors [`conv_forward_blocked`] (parallel
/// over output spatial positions, each gathered input row reused across all
/// output channels) but converts the gathered input rows and weights to f32 ONCE
/// (an L2-friendly f32 scratch of `src`) and reduces in f32, halving the streamed
/// bytes on the reused-read path. The output is widened f32->f64 exactly.
///
/// Soundness note: unlike the f64 kernels this need NOT be bit-identical to any
/// reference — the caller's certified slack uses the order-independent Higham
/// `gamma_n_f32` envelope, valid for whatever tap order / SIMD reduction the
/// autovectorizer picks. Padded taps (`gather == usize::MAX`) and zero weights
/// are skipped identically so the term count never exceeds `k_fwd`. An f32
/// accumulation that overflows to +/-inf writes a non-finite `dst`, which the
/// build's finite-box firewall rejects (fail-closed to Unknown).
pub(crate) fn conv_forward_blocked_f32(
    c: &ConvOp,
    src: &Array2<f64>,
    dst: &mut Array2<f64>,
    abs_w: bool,
    mask: Option<(&[u64], &[u64], usize)>,
) {
    let (co, _, _, _) = c.kernel;
    let p = c.oshape.1 * c.oshape.2;
    let k = c.k_fwd;
    let r = src.ncols();
    debug_assert_eq!(dst.nrows(), co * p);
    debug_assert_eq!(dst.ncols(), r);
    // #tableau-support-mask: the column blocking is driven by the mask when one
    // is supplied (so a block is either wholly computed or wholly skipped), and
    // by the cache-tuned default otherwise.
    let rmax = r.max(1);
    let (blk, nblk) = match mask {
        Some((_, _, n)) => (r.div_ceil(n), n),
        None => {
            let b = ((1usize << 15) / co.max(1)).clamp(64.min(rmax), rmax);
            (b, r.div_ceil(b))
        }
    };
    let src_flat = src.as_slice().expect("standard layout");
    // f32 scratch of the whole input tensor (read once from f64, then reused
    // across the tap reduction from f32 — the bandwidth win). f64->f32 conversion
    // is nearest-rounding (rel err <= UNIT_F32), covered by the caller's slack.
    // Parallel: these tableau tensors are multi-GB, so a serial cast would itself
    // be a memory-bound pass that eats the win.
    //
    // With a mask the cast is restricted to the live blocks of each source row;
    // the rest is provably exact `0.0` and the scratch is already zeroed, so the
    // converted values are identical either way.
    let mut src32: Vec<f32> = vec![0.0; src_flat.len()];
    match mask {
        Some((src_mask, _, _)) => {
            src32
                .par_chunks_mut(r)
                .zip(src_flat.par_chunks(r))
                .zip(src_mask.par_iter())
                .for_each(|((drow, srow), &m)| {
                    for b in 0..nblk {
                        if m & (1u64 << b) == 0 {
                            continue;
                        }
                        let lo = (b * blk).min(r);
                        let hi = (lo + blk).min(r);
                        for (d, &s) in drow[lo..hi].iter_mut().zip(&srow[lo..hi]) {
                            *d = s as f32;
                        }
                    }
                });
        }
        None => {
            src32
                .par_chunks_mut(1 << 16)
                .zip(src_flat.par_chunks(1 << 16))
                .for_each(|(dchunk, schunk)| {
                    for (d, &s) in dchunk.iter_mut().zip(schunk) {
                        *d = s as f32;
                    }
                });
        }
    }
    // f32 weights (small, cached in registers/L1). |w| for the abs (D) lane.
    let w32: Vec<f32> = if abs_w {
        c.wmat.iter().map(|&w| (w as f32).abs()).collect()
    } else {
        c.wmat.iter().map(|&w| w as f32).collect()
    };
    let out_mask = mask.map(|(_, out, _)| out);
    let mut dst3 = dst
        .view_mut()
        .into_shape_with_order((co, p, r))
        .expect("standard layout reshape");
    dst3.axis_iter_mut(Axis(1))
        .into_par_iter()
        .enumerate()
        .for_each_init(
            || vec![0.0f32; co * blk],
            |acc, (sp, mut out_sp)| {
                // All output channels at one spatial position gather the same
                // source rows, so row `sp` (channel 0) carries the whole
                // position's mask. `nblk == SUPPORT_BLOCKS <= 64` exactly when a
                // mask is present, so the shift below is always in range.
                for b in 0..nblk {
                    if out_mask.is_some_and(|om| om[sp] & (1u64 << b) == 0) {
                        continue;
                    }
                    let rc = (b * blk).min(r);
                    let rb = (r - rc).min(blk);
                    if rb == 0 {
                        continue;
                    }
                    for v in acc[..co * rb].iter_mut() {
                        *v = 0.0;
                    }
                    for t in 0..k {
                        let gi = c.gather[t * p + sp];
                        if gi == usize::MAX {
                            continue;
                        }
                        let srow = &src32[gi * r + rc..gi * r + rc + rb];
                        for oc in 0..co {
                            let w = w32[oc * k + t];
                            if w == 0.0 {
                                continue;
                            }
                            axpy_f32(&mut acc[oc * rb..oc * rb + rb], w, srow);
                        }
                    }
                    for oc in 0..co {
                        let mut arow = out_sp.row_mut(oc);
                        let dr = arow.as_slice_mut().expect("contiguous output row");
                        for (d, &a) in dr[rc..rc + rb].iter_mut().zip(&acc[oc * rb..oc * rb + rb]) {
                            *d = f64::from(a);
                        }
                    }
                }
            },
        );
}

/// `acc += w * src` over contiguous f32 slices (autovectorized, non-fused).
#[inline]
fn axpy_f32(acc: &mut [f32], w: f32, src: &[f32]) {
    debug_assert_eq!(acc.len(), src.len());
    for (a, s) in acc.iter_mut().zip(src) {
        *a += w * s;
    }
}

/// im2col + DGEMM forward conv. Computes the
/// SAME per-output-row sum `dst[oc*p+sp] = Σ_t w[oc,t]·src[gather[t,sp]]` as
/// [`conv_apply_forward`], but as a dense GEMM `W[co,k] · Col[k, spb·r]` per
/// tile of `spb` output positions. `ndarray::dot` routes this to Accelerate on
/// macOS and matrixmultiply in the platform-neutral Linux build.
///
/// SOUNDNESS (provably-outward): DGEMM sums the `k` products in a blocked order.
/// Padding taps (`gather == MAX`) and zero weights contribute EXACT `0` (adding
/// `0.0` never changes an f64 sum), so the only difference vs the scalar path is
/// the summation order of the nonzero terms — whose f64 error is bounded by
/// `γ_k·Σ|terms|` for ANY order (Higham). The twin-wall D-lane envelope in
/// `root.rs` (`g·(|M|+D)` on the center conv, `certify_up(·, γ_n(k_fwd+8))` on
/// the radius conv) uses `γ_n(k_fwd+2..8) ≥ γ_k`, so it dominates the reordered
/// error and the concretized box stays an outward enclosure of the exact real
/// value — the identical certificate `sound_f64_gemm`/`mat_mul_f64` use. Proven
/// per-op by `blas_conv_within_gamma_envelope_of_exact` (compensated ref) and
/// `blas_conv_exact_rational_outward_envelope` (BigRational oracle, k≤9408).
pub(crate) fn conv_apply_forward_blas(
    c: &ConvOp,
    src: &Array2<f64>,
    dst: &mut Array2<f64>,
    abs_w: bool,
    co: usize,
    p: usize,
    k: usize,
    r: usize,
) {
    conv_apply_forward_gemm(c, src, dst, abs_w, co, p, k, r, RootGemmBackend::Ndarray);
}

/// faer twin of [`conv_apply_forward_blas`], enabled only by
/// `NY_ROOT_GEMM=faer`. It consumes the same row-major im2col tile through a
/// zero-copy faer view and changes only the f64 summation order. The exact same
/// Higham certificate documented on the ndarray path therefore applies.
pub(crate) fn conv_apply_forward_faer(
    c: &ConvOp,
    src: &Array2<f64>,
    dst: &mut Array2<f64>,
    abs_w: bool,
    co: usize,
    p: usize,
    k: usize,
    r: usize,
) {
    conv_apply_forward_gemm(c, src, dst, abs_w, co, p, k, r, RootGemmBackend::Faer);
}

fn conv_apply_forward_gemm(
    c: &ConvOp,
    src: &Array2<f64>,
    dst: &mut Array2<f64>,
    abs_w: bool,
    co: usize,
    p: usize,
    k: usize,
    r: usize,
    backend: RootGemmBackend,
) {
    enum Weights {
        Ndarray(Array2<f64>),
        Faer(Vec<f64>),
    }

    let src_flat = src.as_slice().expect("standard layout");
    // Weight matrix W [co, k] row-major (|W| for the D lane).
    let wvec: Vec<f64> = if abs_w {
        c.wmat.iter().map(|w| w.abs()).collect()
    } else {
        c.wmat.clone()
    };
    // Move the single weight buffer into the selected representation. In
    // particular, keep the established ndarray path's allocation/copy count
    // exactly as before; the experimental backend must not tax the default.
    let weights = match backend {
        RootGemmBackend::Ndarray => {
            Weights::Ndarray(Array2::from_shape_vec((co, k), wvec).expect("wmat shape"))
        }
        RootGemmBackend::Faer => Weights::Faer(wvec),
    };
    // Tile the output positions so Col[k, spb*r] fits the memory budget.
    let spb = (blas_tile_elems() / (k * r)).clamp(1, p);
    let dst_flat = dst.as_slice_mut().expect("standard layout");
    let mut sp0 = 0usize;
    while sp0 < p {
        let cur = spb.min(p - sp0);
        // Build Col[k, cur*r] row-major: row t is [src[gather[t,sp0+ls]]]_{ls},
        // zero on padding taps. Parallel over the k rows.
        let mut col = Array2::<f64>::zeros((k, cur * r));
        {
            let cs = col.as_slice_mut().expect("row-major");
            cs.par_chunks_mut(cur * r)
                .enumerate()
                .for_each(|(t, crow)| {
                    for ls in 0..cur {
                        let gi = c.gather[t * p + (sp0 + ls)];
                        if gi != usize::MAX {
                            crow[ls * r..ls * r + r].copy_from_slice(&src_flat[gi * r..gi * r + r]);
                        }
                    }
                });
        }
        // out[co, cur*r] = W[co,k] @ Col[k, cur*r]. Both kernels consume the
        // same row-major values and differ only in their certified sum order.
        match &weights {
            Weights::Ndarray(w_arr) => {
                let out = w_arr.dot(&col);
                let os = out.as_slice().expect("row-major");
                // Scatter: out row oc -> dst rows [oc*p+sp0 .. +cur]
                // (contiguous within the oc-chunk).
                dst_flat
                    .par_chunks_mut(p * r)
                    .enumerate()
                    .for_each(|(oc, dchunk)| {
                        dchunk[sp0 * r..sp0 * r + cur * r]
                            .copy_from_slice(&os[oc * cur * r..(oc + 1) * cur * r]);
                    });
            }
            Weights::Faer(wvec) => {
                let out = crate::faer_parallelism::mat_mul_f64_row_major(
                    wvec,
                    co,
                    k,
                    col.as_slice().expect("row-major im2col"),
                    cur * r,
                );
                dst_flat
                    .par_chunks_mut(p * r)
                    .enumerate()
                    .for_each(|(oc, dchunk)| {
                        let target = &mut dchunk[sp0 * r..sp0 * r + cur * r];
                        for (j, value) in target.iter_mut().enumerate() {
                            *value = out[(oc, j)];
                        }
                    });
            }
        }
        sp0 += cur;
    }
}

/// Cache-blocked forward conv (default). Parallelises over output spatial
/// positions `sp` and, for each, reduces ALL output channels together so every
/// gathered input row is read ONCE and reused across the `co` channels — the
/// naive per-channel kernels re-read each input row `co` times, which makes the
/// big tableau convs memory-bandwidth bound. Parallelism is expressed with
/// ndarray's safe parallel axis iterator (no unsafe): the `(co*P, R)` output is
/// viewed as `(co, P, R)` and each `sp` slice is an independent `(co, R)` view.
/// Bit-identical to [`conv_forward_ocgrain`]: for a fixed `(oc, sp, col)` the
/// additions are the same taps in the same ascending-`t` order with the same
/// non-fused `acc += w*s` rounding (zero weights / padded taps skipped alike).
pub(crate) fn conv_forward_blocked(
    c: &ConvOp,
    src: &Array2<f64>,
    dst: &mut Array2<f64>,
    abs_w: bool,
) {
    let (co, _, _, _) = c.kernel;
    let p = c.oshape.1 * c.oshape.2;
    let k = c.k_fwd;
    let r = src.ncols();
    debug_assert_eq!(dst.nrows(), co * p);
    debug_assert_eq!(dst.ncols(), r);
    let src_flat = src.as_slice().expect("standard layout");
    // ~256 KiB accumulator tile (co * r_blk f64), L2-resident so the tap
    // reduction streams the reused input slice from L1. Floor at min(64, r) so
    // the clamp bounds stay ordered for narrow row widths.
    let rmax = r.max(1);
    let r_blk = ((1usize << 15) / co.max(1)).clamp(64.min(rmax), rmax);
    let mut dst3 = dst
        .view_mut()
        .into_shape_with_order((co, p, r))
        .expect("standard layout reshape");
    dst3.axis_iter_mut(Axis(1))
        .into_par_iter()
        .enumerate()
        .for_each_init(
            || vec![0.0f64; co * r_blk],
            |acc, (sp, mut out_sp)| {
                // out_sp: (co, R) view; out_sp[[oc, col]] == dst[(oc*P + sp, col)].
                let mut rc = 0;
                while rc < r {
                    let rb = (r - rc).min(r_blk);
                    for v in acc[..co * rb].iter_mut() {
                        *v = 0.0;
                    }
                    for t in 0..k {
                        let gi = c.gather[t * p + sp];
                        if gi == usize::MAX {
                            continue;
                        }
                        let srow = &src_flat[gi * r + rc..gi * r + rc + rb];
                        for oc in 0..co {
                            let w0 = c.wmat[oc * k + t];
                            let w = if abs_w { w0.abs() } else { w0 };
                            if w == 0.0 {
                                continue;
                            }
                            axpy(&mut acc[oc * rb..oc * rb + rb], w, srow);
                        }
                    }
                    for oc in 0..co {
                        let mut arow = out_sp.row_mut(oc);
                        let dr = arow.as_slice_mut().expect("contiguous output row");
                        dr[rc..rc + rb].copy_from_slice(&acc[oc * rb..oc * rb + rb]);
                    }
                    rc += rb;
                }
            },
        );
}

/// Row-parallel forward conv: one rayon work unit per output row `(oc, sp)`,
/// i.e. `co*P` independent units. Each row folds its taps in ascending `t`
/// order into a freshly zeroed accumulator — bit-identical to
/// [`conv_forward_ocgrain`].
pub(crate) fn conv_forward_rowgrain(
    c: &ConvOp,
    src: &Array2<f64>,
    dst: &mut Array2<f64>,
    abs_w: bool,
) {
    let (co, _, _, _) = c.kernel;
    let p = c.oshape.1 * c.oshape.2;
    let k = c.k_fwd;
    let r = src.ncols();
    debug_assert_eq!(dst.nrows(), co * p);
    debug_assert_eq!(dst.ncols(), r);
    let src_flat = src.as_slice().expect("standard layout");
    let dst_slice = dst.as_slice_mut().expect("standard layout");
    dst_slice
        .par_chunks_mut(r)
        .enumerate()
        .for_each(|(row, acc)| {
            // Rows are output-channel major: row = oc*P + sp. rayon hands each
            // thread a contiguous run of rows, so `wrow` reuse across `sp` within
            // one channel is preserved just as in the per-channel grain.
            let oc = row / p;
            let sp = row - oc * p;
            let wrow = &c.wmat[oc * k..(oc + 1) * k];
            acc.fill(0.0);
            for (t, &w0) in wrow.iter().enumerate() {
                let w = if abs_w { w0.abs() } else { w0 };
                if w == 0.0 {
                    continue;
                }
                let gi = c.gather[t * p + sp];
                if gi == usize::MAX {
                    continue;
                }
                let srow = &src_flat[gi * r..(gi + 1) * r];
                axpy(acc, w, srow);
            }
        });
}

/// Legacy per-output-channel grain: one rayon work unit per output channel
/// (`co` units). Retained as the `NY_MARGIN_ROW_ROOT_PAR=0` fallback and as the
/// bit-identity oracle for the row-grain kernel.
pub(crate) fn conv_forward_ocgrain(
    c: &ConvOp,
    src: &Array2<f64>,
    dst: &mut Array2<f64>,
    abs_w: bool,
) {
    let (co, _, _, _) = c.kernel;
    let p = c.oshape.1 * c.oshape.2;
    let k = c.k_fwd;
    let r = src.ncols();
    debug_assert_eq!(dst.nrows(), co * p);
    debug_assert_eq!(dst.ncols(), r);
    let src_flat = src.as_slice().expect("standard layout");
    let dst_slice = dst.as_slice_mut().expect("standard layout");
    dst_slice
        .par_chunks_mut(p * r)
        .enumerate()
        .for_each(|(oc, dchunk)| {
            let wrow = &c.wmat[oc * k..(oc + 1) * k];
            for sp in 0..p {
                let acc = &mut dchunk[sp * r..(sp + 1) * r];
                acc.fill(0.0);
                for (t, &w0) in wrow.iter().enumerate() {
                    let w = if abs_w { w0.abs() } else { w0 };
                    if w == 0.0 {
                        continue;
                    }
                    let gi = c.gather[t * p + sp];
                    if gi == usize::MAX {
                        continue;
                    }
                    let srow = &src_flat[gi * r..(gi + 1) * r];
                    axpy(acc, w, srow);
                }
            }
        });
}

/// Backward (transposed) conv on row matrices:
/// `dst[(ic, isp)] = sum_{taps, oc} w * src[(oc, osp)]`.
///
/// # Kernel selection (`NY_MARGIN_ROW_CONV_BWD_BLOCKED`)
///
/// Two interchangeable, **bit-identical** kernels (proven by the grain test
/// `conv_backward_grains_bit_identical_to_ic_grain` — they change only which
/// thread owns each output row and the loop nesting, never the per-output-cell
/// add order: row `(ic, isp)` always folds its `back_taps[isp]` triples in
/// table order and, within each tap, output channels in ascending `oc` order,
/// skipping zero weights identically, into a zero-initialised accumulator via
/// the same non-fused `acc += w*s` step):
///
/// * unset (default) or any truthy value — [`conv_backward_blocked`]: the
///   backward mirror of
///   [`conv_forward_blocked`] — parallel over input spatial positions, each
///   gathered src row read ONCE and reused across all `ci` input channels.
/// * `0`, `false`, `off`, or `no` — [`conv_backward_icgrain`]: the
///   bit-identical row-grain fallback. It exposes the same spatial
///   parallelism after the row-regraining change, but still re-reads each
///   gathered source row once per input channel.
///
/// The blocked kernel is the scored default because it removes repeated source
/// reads from every margin-row tree-phase backward pass. Historical CIFAR100
/// sweep data made this a strong throughput target, but is not treated here as
/// sealed score evidence; the fallback remains available for live A/B checks.
pub(super) fn blocked_backward_enabled_from_env(value: Option<&str>) -> bool {
    !value.is_some_and(|value| {
        let value = value.trim();
        value == "0"
            || value.eq_ignore_ascii_case("false")
            || value.eq_ignore_ascii_case("off")
            || value.eq_ignore_ascii_case("no")
    })
}

/// Read-once kernel selector for [`conv_apply_backward`].
///
/// This used to be a bare `std::env::var` on EVERY call. `conv_apply_backward`
/// is ~17% of profiled samples and runs twice per conv op per backward pass
/// (once for the coefficient lane, once for the abs-weight error lane) --
/// ~38 calls per pass, several passes per BaB expansion, concurrently from
/// every rayon worker in the parallel frontier. `std::env::var` takes a
/// process-global lock and allocates a `String` each time, so that was
/// lock traffic and allocator churn injected directly into the hottest kernel
/// by a flag that cannot change during a run.
///
/// Matches `blas_tile_elems` above -- same file, same idiom, same rationale.
fn blocked_backward_enabled() -> bool {
    static B: OnceLock<bool> = OnceLock::new();
    *B.get_or_init(|| {
        blocked_backward_enabled_from_env(
            std::env::var("NY_MARGIN_ROW_CONV_BWD_BLOCKED")
                .ok()
                .as_deref(),
        )
    })
}

pub fn conv_apply_backward(c: &ConvOp, src: &Array2<f64>, dst: &mut Array2<f64>, abs_w: bool) {
    if blocked_backward_enabled() {
        conv_backward_blocked(c, src, dst, abs_w);
    } else {
        conv_backward_icgrain(c, src, dst, abs_w);
    }
}

/// Row-grain fallback retained as the bit-identity oracle for
/// [`conv_backward_blocked`].
pub(crate) fn conv_backward_icgrain(
    c: &ConvOp,
    src: &Array2<f64>,
    dst: &mut Array2<f64>,
    abs_w: bool,
) {
    let (co, ci, kh, kw) = c.kernel;
    let ip = c.ishape.1 * c.ishape.2;
    let op_ = c.oshape.1 * c.oshape.2;
    let r = src.ncols();
    debug_assert_eq!(dst.nrows(), ci * ip);
    debug_assert_eq!(dst.ncols(), r);
    let src_flat = src.as_slice().expect("standard layout");
    let dst_slice = dst.as_slice_mut().expect("standard layout");
    // One independent Rayon job per `(input-channel, input-spatial)` output
    // row. The previous `ip * r` grain exposed only `ci` jobs (three at the
    // CIFAR input convolution), leaving most cores idle for a single wide
    // candidate tableau and encouraging many memory-heavy BaB domains to run
    // concurrently. Regraining changes only row ownership: every row still
    // folds `(ky, kx, osp)` taps and then `oc = 0..co` in exactly the same
    // order, through the same `axpy`, so signed coefficients and the
    // absolute-weight certified-error lane remain bit-identical.
    dst_slice
        .par_chunks_mut(r)
        .enumerate()
        .for_each(|(input_row, acc)| {
            let ic = input_row / ip;
            let isp = input_row % ip;
            acc.fill(0.0);
            for &(ky, kx, osp) in &c.back_taps[isp] {
                let wbase = ((ic * kh + ky) * kw + kx) * co;
                for oc in 0..co {
                    let w0 = c.wt[wbase + oc];
                    let w = if abs_w { w0.abs() } else { w0 };
                    if w == 0.0 {
                        continue;
                    }
                    let srow = &src_flat[(oc * op_ + osp) * r..(oc * op_ + osp) * r + r];
                    axpy(acc, w, srow);
                }
            }
        });
}

/// Cache-blocked backward conv (default; `NY_MARGIN_ROW_CONV_BWD_BLOCKED=0`
/// selects the fallback).
/// Parallelises over input spatial positions `isp` and, for each, reduces ALL
/// input channels together so every gathered src row is read ONCE and reused
/// across the `ci` channels — the row-grain fallback re-reads each src row
/// `ci` times. This is the
/// backward mirror of [`conv_forward_blocked`]; parallelism is expressed with
/// ndarray's safe parallel axis iterator (no unsafe): the `(ci*IP, R)` output
/// is viewed as `(ci, IP, R)` and each `isp` slice is an independent `(ci, R)`
/// view. Bit-identical to [`conv_backward_icgrain`]: for a fixed
/// `(ic, isp, col)` the additions are the same `back_taps[isp]` triples in the
/// same table order and, within a tap, the same ascending-`oc` order with the
/// same non-fused `acc += w*s` rounding (zero weights skipped alike);
/// interleaving adds across the `ci` accumulator rows never reorders any
/// single row's fold, and the column blocking partitions `R` without touching
/// any column's add sequence.
pub(crate) fn conv_backward_blocked(
    c: &ConvOp,
    src: &Array2<f64>,
    dst: &mut Array2<f64>,
    abs_w: bool,
) {
    let (co, ci, kh, kw) = c.kernel;
    let ip = c.ishape.1 * c.ishape.2;
    let op_ = c.oshape.1 * c.oshape.2;
    let r = src.ncols();
    debug_assert_eq!(dst.nrows(), ci * ip);
    debug_assert_eq!(dst.ncols(), r);
    let src_flat = src.as_slice().expect("standard layout");
    // ~256 KiB accumulator tile (ci * r_blk f64), L2-resident so the tap
    // reduction streams the reused src slice from L1 — the mirror of
    // conv_forward_blocked's co-tile. Floor at min(64, r) so the clamp bounds
    // stay ordered for narrow row widths.
    let rmax = r.max(1);
    let r_blk = ((1usize << 15) / ci.max(1)).clamp(64.min(rmax), rmax);
    let mut dst3 = dst
        .view_mut()
        .into_shape_with_order((ci, ip, r))
        .expect("standard layout reshape");
    dst3.axis_iter_mut(Axis(1))
        .into_par_iter()
        .enumerate()
        .for_each_init(
            || vec![0.0f64; ci * r_blk],
            |acc, (isp, mut out_isp)| {
                // out_isp: (ci, R) view; out_isp[[ic, col]] == dst[(ic*IP + isp, col)].
                let taps = &c.back_taps[isp];
                let mut rc = 0;
                while rc < r {
                    let rb = (r - rc).min(r_blk);
                    for v in acc[..ci * rb].iter_mut() {
                        *v = 0.0;
                    }
                    for &(ky, kx, osp) in taps {
                        // wt is [ci][kh][kw][co]: entry (ic, ky, kx, oc) sits at
                        // ic*(kh*kw*co) + tbase + oc — identical indexing to the
                        // legacy grain's wbase + oc.
                        let tbase = (ky * kw + kx) * co;
                        for oc in 0..co {
                            let sbase = (oc * op_ + osp) * r + rc;
                            let srow = &src_flat[sbase..sbase + rb];
                            for ic in 0..ci {
                                let w0 = c.wt[ic * (kh * kw * co) + tbase + oc];
                                let w = if abs_w { w0.abs() } else { w0 };
                                if w == 0.0 {
                                    continue;
                                }
                                axpy(&mut acc[ic * rb..ic * rb + rb], w, srow);
                            }
                        }
                    }
                    for ic in 0..ci {
                        let mut arow = out_isp.row_mut(ic);
                        let dr = arow.as_slice_mut().expect("contiguous output row");
                        dr[rc..rc + rb].copy_from_slice(&acc[ic * rb..ic * rb + rb]);
                    }
                    rc += rb;
                }
            },
        );
}

/// `acc += w * src` over contiguous slices (autovectorized).
#[inline]
pub fn axpy(acc: &mut [f64], w: f64, src: &[f64]) {
    debug_assert_eq!(acc.len(), src.len());
    for (a, s) in acc.iter_mut().zip(src) {
        *a += w * s;
    }
}
