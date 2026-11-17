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
use std::time::Instant;

use super::rounding::next_up;
use super::spec::{TwinOpSpec, TwinSpec, MAX_ROOT_TABLEAU_ENTRIES};

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
        /// Certified absolute per-row bias error vs the exact real fold
        /// (#cct-margin-row-admission). Always `n_out` entries after compile;
        /// all zero unless a BatchNormalization was folded in.
        bias_err: Vec<f64>,
        /// Certified relative weight error (0.0 unless folded). The compiler
        /// refuses a nonzero budget on the FINAL Gemm.
        weight_rel_err: f64,
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
        Self::compile_with_root_source(
            spec,
            super::RootBoxSource::Tableau,
            super::PatchBackwardConflicts::default(),
        )
    }

    /// Source-aware compilation never permits an oversized forward tableau.
    /// A patch-only net may compile if its actual source/consumer census fits;
    /// every full-tableau entry separately enforces the original width guard.
    pub fn compile_with_root_source(
        spec: &TwinSpec,
        source: super::RootBoxSource,
        conflicts: super::PatchBackwardConflicts,
    ) -> Result<Self> {
        let bad = |m: &str| NyError::InvalidSpec(format!("margin_row twin net: {m}"));
        if spec.n_in == 0 || spec.ops.is_empty() {
            return Err(bad("empty spec"));
        }
        super::root_box::source_admission(spec, source, conflicts)?;
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
                    bias_err,
                    weight_rel_err,
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
                    // #cct-margin-row-admission: a folded head carries a
                    // certified error budget (empty `bias_err` = exact). Same
                    // admission rule as the Conv arm's budgets.
                    if !(bias_err.is_empty() || bias_err.len() == no)
                        || bias_err.iter().any(|e| !e.is_finite() || *e < 0.0)
                        || !weight_rel_err.is_finite()
                        || *weight_rel_err < 0.0
                    {
                        return Err(bad(&format!("gemm {k} inconsistent fold error budget")));
                    }
                    gemm_idx.push(k);
                    ops.push(TwinOp::Gemm {
                        input: *input,
                        weight: weight.clone(),
                        bias: bias.clone(),
                        bias_err: if bias_err.is_empty() {
                            vec![0.0; no]
                        } else {
                            bias_err.clone()
                        },
                        weight_rel_err: *weight_rel_err,
                        shape: *shape,
                    });
                    tsize.push(no);
                }
            }
        }
        // #cct-margin-row-admission: refuse a net whose root forward tableau
        // would exceed the shared ceiling (see `MAX_ROOT_TABLEAU_ENTRIES`).
        // The same predicate gates the ny-cli admission probe.
        if source == super::RootBoxSource::Tableau && !spec.root_tableau_within_cap() {
            return Err(bad(&format!(
                "root tableau needs {:?} entries, above the {} cap",
                spec.root_tableau_entries(),
                MAX_ROOT_TABLEAU_ENTRIES
            )));
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
        // #cct-margin-row-admission: only the FIRST head Gemm may carry a fold
        // error. Every consumer of `gemm2()` (MarginBatch, the margin seed, the
        // via-y composition) reads the final weights as EXACT, so a folded
        // final Gemm is refused rather than silently under-charged.
        if spec.ops[i_gemm2].is_folded_gemm() {
            return Err(bad(
                "final gemm carries a fold error; only the head gemm may be folded",
            ));
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

    pub fn root_tableau_within_cap(&self) -> bool {
        self.n_in
            .checked_add(1)
            .and_then(|columns| self.tsize.iter().copied().max()?.checked_mul(columns))
            .is_some_and(|entries| entries <= MAX_ROOT_TABLEAU_ENTRIES)
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

    /// Certified fold error of the FIRST head Gemm (#cct-margin-row-admission):
    /// `(per-row absolute bias error, relative weight error)`, both measured
    /// against the exact real-arithmetic `Gemm -> BatchNormalization` map.
    /// All zero unless the extractor folded a BN into the head. Every bound
    /// that reads [`Self::gemm1`] must charge these (the CPU seed in
    /// `engine.rs`, the GPU head statics in `gpu_seam` / `gpu_backward`).
    pub fn gemm1_fold_err(&self) -> (&[f64], f64) {
        match &self.ops[self.i_gemm1] {
            TwinOp::Gemm {
                bias_err,
                weight_rel_err,
                ..
            } => (bias_err, *weight_rel_err),
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
                // Point evaluation of the STORED (possibly BN-folded) head:
                // within the fold's certified error of the exact net. Nothing
                // verdict-bearing reads these values (sampling, probes,
                // membership filters).
                TwinOp::Gemm {
                    input,
                    weight,
                    bias,
                    shape,
                    ..
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

/// Compile one convolution's gather / backward-tap tables.
///
/// `pub(crate)` for #bi-window (`backward_interm::window`), which compiles
/// RECEPTIVE-FIELD CROPS of an already-validated conv: the same kernel, stride
/// and weights over a sub-rectangle of the output, with the crop's own
/// (possibly asymmetric) padding. The tables are a pure function of the
/// geometry, so a crop's taps are exactly the full conv's taps restricted to
/// the rectangle.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compile_conv(
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
    let r = src.ncols();
    let (blk, nblk) = conv_f32_blocking(mask, r, co);
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
    let src32_rows = src32.len() / r.max(1);
    match mask {
        Some((src_mask, _, _)) => {
            src32
                .par_chunks_mut(r)
                .zip(src_flat.par_chunks(r))
                .zip(src_mask.par_iter())
                .with_min_len(par_min_len(r, src32_rows))
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
    conv_f32_core(
        c,
        &src32,
        r,
        dst,
        abs_w,
        mask.map(|(_, out, _)| out),
        blk,
        nblk,
    );
}

/// f32-STORED forward conv (`NY_MARGIN_ROW_TABLEAU_F32` storage lane): the
/// source tableau is ALREADY `f32`, so the per-layer cast pass of
/// [`conv_forward_blocked_f32`] is deleted (that pass is a full extra
/// read+write of a multi-GB tensor per lane per layer), and the f32
/// accumulator is stored into the f32 `dst` DIRECTLY.
///
/// EXACTNESS OF THE STORE: the value [`conv_forward_blocked_f32`] writes today
/// is `f64::from(acc)` — an exact widening of the same f32 accumulator — so
/// storing `acc` itself is the same number, bit for bit (class (a) of the
/// storage-soundness taxonomy in `root.rs`; zero new rounding). The shared
/// [`conv_f32_core`] guarantees the accumulation is the identical code path.
/// The caller's `gamma_n_f32(k_fwd + 8)` slack still budgets one input
/// f64->f32 conversion that no longer happens — a sound over-count.
pub(crate) fn conv_forward_f32_native(
    c: &ConvOp,
    src: &Array2<f32>,
    dst: &mut Array2<f32>,
    abs_w: bool,
    mask: Option<(&[u64], &[u64], usize)>,
) {
    let (co, _, _, _) = c.kernel;
    let r = src.ncols();
    let (blk, nblk) = conv_f32_blocking(mask, r, co);
    let src_flat = src.as_slice().expect("standard layout");
    conv_f32_core(
        c,
        src_flat,
        r,
        dst,
        abs_w,
        mask.map(|(_, out, _)| out),
        blk,
        nblk,
    );
}

/// Deadline-aware native f32 contraction. On error the caller must discard
/// dst: joined workers may have stored complete independent column tiles.
pub(crate) fn conv_forward_f32_native_checked(
    c: &ConvOp,
    src: &Array2<f32>,
    dst: &mut Array2<f32>,
    abs_w: bool,
    mask: Option<(&[u64], &[u64], usize)>,
    deadline: Option<Instant>,
) -> Result<()> {
    if deadline.is_none() {
        conv_forward_f32_native(c, src, dst, abs_w, mask);
        return Ok(());
    }
    check_contraction_deadline::<true, false>(deadline, None)?;
    let r = src.ncols();
    let (blk, nblk) = conv_f32_blocking(mask, r, c.kernel.0);
    conv_f32_core_checked::<f32, true, false>(
        c,
        src.as_slice().expect("standard layout"),
        r,
        dst,
        abs_w,
        mask.map(|(_, out, _)| out),
        blk,
        nblk,
        deadline,
        None,
    )
}

/// Shared column blocking for the f32 convolution paths.
///
/// #tableau-support-mask: the column blocking is driven by the mask when one
/// is supplied (so a block is either wholly computed or wholly skipped), and
/// by the cache-tuned default otherwise.
fn conv_f32_blocking(mask: Option<(&[u64], &[u64], usize)>, r: usize, co: usize) -> (usize, usize) {
    let rmax = r.max(1);
    match mask {
        Some((_, _, n)) => (r.div_ceil(n), n),
        None => {
            let b = ((1usize << 15) / co.max(1)).clamp(64.min(rmax), rmax);
            (b, r.div_ceil(b))
        }
    }
}

/// Element type an f32 conv accumulator may be stored into.
///
/// `f64` widens exactly (the shipped `root_f32` compute lane); `f32` stores the
/// accumulator bits directly (the `NY_MARGIN_ROW_TABLEAU_F32` storage lane).
/// Both stores publish the SAME number, so the two lanes share one kernel and
/// the refactor is bit-identity-testable
/// (`conv_f32_native_store_is_bitwise_the_widened_path`).
pub(crate) trait ConvF32Dst: Copy + Send + Sync {
    fn from_acc(a: f32) -> Self;
}
impl ConvF32Dst for f64 {
    #[inline]
    fn from_acc(a: f32) -> f64 {
        f64::from(a)
    }
}
impl ConvF32Dst for f32 {
    #[inline]
    fn from_acc(a: f32) -> f32 {
        a
    }
}

/// Shared spatial/tap loop. Both storage lanes use the same per-coefficient
/// arithmetic; the untimed specialization eliminates deadline polling.
#[allow(clippy::too_many_arguments)]
fn conv_f32_core<O: ConvF32Dst>(
    c: &ConvOp,
    src32: &[f32],
    r: usize,
    dst: &mut Array2<O>,
    abs_w: bool,
    out_mask: Option<&[u64]>,
    blk: usize,
    nblk: usize,
) {
    conv_f32_core_checked::<O, false, false>(
        c, src32, r, dst, abs_w, out_mask, blk, nblk, None, None,
    )
    .expect("untimed forward contraction cannot expire");
}

// A work bound between cooperative polls, not a wall-time guarantee.
const FORWARD_POLL_SCALARS: usize = 16 * 1024;

#[allow(clippy::too_many_arguments)]
fn conv_f32_core_checked<O: ConvF32Dst, const TIMED: bool, const COOPERATIVE: bool>(
    c: &ConvOp,
    src32: &[f32],
    r: usize,
    dst: &mut Array2<O>,
    abs_w: bool,
    out_mask: Option<&[u64]>,
    blk: usize,
    nblk: usize,
    deadline: Option<Instant>,
    checkpoint: Option<&(dyn Fn() -> Result<()> + Sync)>,
) -> Result<()> {
    let poll = || check_contraction_deadline::<TIMED, COOPERATIVE>(deadline, checkpoint);
    poll()?;
    let (co, _, _, _) = c.kernel;
    let p = c.oshape.1 * c.oshape.2;
    let k = c.k_fwd;
    debug_assert_eq!(dst.nrows(), co * p);
    debug_assert_eq!(dst.ncols(), r);
    debug_assert_eq!(src32.len() % r.max(1), 0);
    // f32 weights (small, cached in registers/L1). |w| for the abs (D) lane.
    let mut w32 = Vec::with_capacity(c.wmat.len());
    for (i, &w) in c.wmat.iter().enumerate() {
        if TIMED && i % FORWARD_POLL_SCALARS == 0 {
            poll()?;
        }
        w32.push(if abs_w { (w as f32).abs() } else { w as f32 });
    }
    poll()?;
    let tile = if TIMED {
        blk.min(FORWARD_POLL_SCALARS)
    } else {
        blk
    }
    .max(1);
    let nacc = co
        .checked_mul(tile)
        .ok_or_else(|| NyError::InvalidSpec("forward convolution scratch overflow".into()))?;
    let mut dst3 = dst
        .view_mut()
        .into_shape_with_order((co, p, r))
        .expect("standard layout reshape");
    dst3.axis_iter_mut(Axis(1))
        .into_par_iter()
        .with_min_len(par_min_len(co * blk, p))
        .enumerate()
        .try_for_each_init(Vec::<f32>::new, |acc, (sp, mut out_sp)| -> Result<()> {
            poll()?;
            if acc.len() < nacc {
                acc.try_reserve_exact(nacc - acc.len()).map_err(|_| {
                    NyError::InternalError("forward convolution scratch allocation failed".into())
                })?;
                while acc.len() < nacc {
                    poll()?;
                    let chunk = if TIMED { FORWARD_POLL_SCALARS } else { nacc };
                    acc.resize(acc.len() + (nacc - acc.len()).min(chunk), 0.0);
                }
            }
            // All output channels at one spatial position gather the same
            // source rows, so row `sp` (channel 0) carries the whole
            // position's mask. `nblk == SUPPORT_BLOCKS <= 64` exactly when a
            // mask is present, so the shift below is always in range.
            for b in 0..nblk {
                poll()?;
                if out_mask.is_some_and(|om| om[sp] & (1u64 << b) == 0) {
                    continue;
                }
                let start = (b * blk).min(r);
                let end = start + (r - start).min(blk);
                // Subdivide within the original support block. Changing
                // blk/nblk would reinterpret the caller's support bits.
                for rc in (start..end).step_by(tile) {
                    poll()?;
                    let rb = (end - rc).min(tile);
                    let work_tile = if TIMED {
                        FORWARD_POLL_SCALARS
                    } else {
                        (co * rb).max(1)
                    };
                    for part in acc[..co * rb].chunks_mut(work_tile) {
                        poll()?;
                        part.fill(0.0);
                    }
                    let channels = if TIMED {
                        (FORWARD_POLL_SCALARS / rb).max(1)
                    } else {
                        co.max(1)
                    };
                    for t in 0..k {
                        poll()?;
                        let gi = c.gather[t * p + sp];
                        if gi == usize::MAX {
                            continue;
                        }
                        let srow = &src32[gi * r + rc..gi * r + rc + rb];
                        for oc0 in (0..co).step_by(channels) {
                            poll()?;
                            for oc in oc0..(oc0 + channels).min(co) {
                                let w = w32[oc * k + t];
                                if w == 0.0 {
                                    continue;
                                }
                                axpy_f32(&mut acc[oc * rb..oc * rb + rb], w, srow);
                            }
                        }
                    }
                    for oc0 in (0..co).step_by(channels) {
                        poll()?;
                        for oc in oc0..(oc0 + channels).min(co) {
                            let mut arow = out_sp.row_mut(oc);
                            let dr = arow.as_slice_mut().expect("contiguous output row");
                            for (d, &a) in
                                dr[rc..rc + rb].iter_mut().zip(&acc[oc * rb..oc * rb + rb])
                            {
                                *d = O::from_acc(a);
                            }
                        }
                    }
                }
            }
            poll()
        })?;
    poll()
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
    conv_forward_blocked_min_len(c, src, dst, abs_w, None);
}

/// [`conv_forward_blocked`] with an explicit rayon `min_len` over output
/// spatial positions (`None` = the production policy, [`blocked_min_len`]).
///
/// The override exists for `narrow_partition_bit_identical`: the partition
/// only decides which task owns each `sp`, never the fold of any output
/// element (see the bit-identity argument on [`blocked_min_len`]).
pub(crate) fn conv_forward_blocked_min_len(
    c: &ConvOp,
    src: &Array2<f64>,
    dst: &mut Array2<f64>,
    abs_w: bool,
    min_len: Option<usize>,
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
    let min_len =
        min_len.unwrap_or_else(|| blocked_min_len(NarrowKernel::Forward, co * r_blk, p, r_blk));
    let mut dst3 = dst
        .view_mut()
        .into_shape_with_order((co, p, r))
        .expect("standard layout reshape");
    dst3.axis_iter_mut(Axis(1))
        .into_par_iter()
        .with_min_len(min_len.max(1))
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

/// Minimum spatial positions per rayon task, so a task is worth its own wakeup.
///
/// These conv kernels parallelise over spatial position: one work unit per
/// `sp`. That is right for a ResNet layer and badly wrong for a 4x4 probe net,
/// where a unit is a few microseconds and rayon's per-task sleep/wake costs
/// more than the arithmetic. Sampling `engine_and_bab_bit_identical_across_
/// backward_gate` -- 261s wall for 31.7s of CPU, i.e. 88% IDLE -- put the
/// process in `rayon_core::sleep::Sleep::wake_specific_thread` contending on a
/// pthread mutex, not in any kernel of ours.
///
/// Grouping positions until a task carries ~16Ki output elements keeps large
/// convs fully parallel (min_len collapses to 1) while a small one degenerates
/// to a single task and skips the scheduler entirely. Positions are computed
/// independently into disjoint output rows, so grouping them changes no
/// accumulation order and the result stays BIT-IDENTICAL -- which the conv-grain
/// oracles assert directly.
const PAR_MIN_TASK_ELEMS: usize = 1 << 14;

fn par_min_len(elems_per_unit: usize, units: usize) -> usize {
    par_min_len_with(PAR_MIN_TASK_ELEMS, elems_per_unit, units)
}

/// [`par_min_len`] with the per-task element target as a parameter
/// (`task_elems`; [`PAR_MIN_TASK_ELEMS`] is the historical value).
fn par_min_len_with(task_elems: usize, elems_per_unit: usize, units: usize) -> usize {
    let units = units.max(1);
    // WHOLE JOB IS TINY: one task, so rayon is entered but never splits. This is
    // the 4x4 probe-net case the oracles hammer.
    if units.saturating_mul(elems_per_unit) < task_elems {
        return units;
    }
    // OTHERWISE: coarsen toward ~16Ki-element tasks, but never so far that the
    // machine goes idle. Grouping purely by work took
    // `engine_and_bab_bit_identical_across_backward_gate` -- 101s of real CPU,
    // not a tiny job -- down to ~9 tasks on 14 cores and made it SLOWER
    // (16.31s -> 20.54s). The floor keeps at least one task per thread.
    let by_work = task_elems.div_ceil(elems_per_unit.max(1)).max(1);
    let keep_cores_busy = (units / rayon::current_num_threads().max(1)).max(1);
    by_work.min(keep_cores_busy)
}

// ---------------------------------------------------------------------------
//  #narrow-r — the narrow-width granularity floor (throughput plan B1(d))
// ---------------------------------------------------------------------------
//
// THE DEFECT. `par_min_len` sizes a task in OUTPUT ELEMENTS x columns: a job is
// one task when `tensor x r_blk < 16Ki`. At R = 1 the blocked kernels' `r_blk`
// is 1, so every conv whose tensor is under 16Ki elements runs its whole
// backward (and forward) as ONE serial task. Per the col-scaling doc (§2)
// cifar100 resnet_medium's WIDEST tensor is 14 400, i.e. every one of its
// convs — so the "cheap" narrow passes the tree runs after column retirement
// (R = nf = 1..3: main eval, beta trials, domain-alpha trials, both linearized
// walks) would be serial walls, which fits the standalone curve's 329 ms at
// R = 1 against 130 ms at R = 2. DERIVED from code + that doc (plan F2), not
// measured: `col-scaling-v2`'s 1-thread/20-thread arms (`absw`, `pass`) are
// the measurement.
//
// THE FLOOR. For `r_blk <= NARROW_R_MAX` the task target is
// `NARROW_TASK_ELEMS_PER_COL x r_blk` (2 Ki x r_blk) instead of 16 Ki, so an
// R <= 4 pass over a tensor of >= 2 Ki elements still splits across rayon
// tasks (and `keep_cores_busy` still caps the grouping at one task per
// thread). `r_blk >= 8` already crossed at 2 Ki elements and is unchanged.
//
// BIT-IDENTITY (the kernels' own ordering claim, re-read for this change).
// Both blocked kernels parallelise over spatial positions ONLY: each task owns
// whole `sp` (forward) / `isp` (backward) slices of the output, and for every
// owned position it zeroes its accumulator tile, folds that position's taps in
// the fixed table order with the same non-fused `acc += w*s` (`axpy`), skipping
// zero weights / padded taps identically, then copies the tile out. The column
// blocking (`r_blk`) is NOT touched here. So `min_len` only changes WHICH
// thread owns a position and how many positions share one reused tile, never
// the sequence of additions into any output element — the same argument that
// already lets `par_min_len` group positions. Pinned bit-for-bit by
// `narrow_partition_bit_identical` (forced single-task vs maximal split vs the
// policy, R = 1..4, signed and |W|, padded taps, zero and -0.0 weights,
// -0.0 inputs, against the row-grain oracles).

/// Widest per-task column block the narrow floor applies to.
const NARROW_R_MAX: usize = 4;
/// Narrow task target, per column of the block (2 Ki output elements).
const NARROW_TASK_ELEMS_PER_COL: usize = 1 << 11;

static NARROW_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static NARROW_ENGAGED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static NARROW_ANNOUNCED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Which blocked kernel asked (for the one-time engagement line).
#[derive(Clone, Copy, Debug)]
enum NarrowKernel {
    Forward,
    Backward,
}

/// The blocked conv kernels' `min_len` policy: [`par_min_len`], with the
/// narrow-width floor for `r_blk <= NARROW_R_MAX`. Never coarser than the
/// legacy rule (a smaller task target can only lower `min_len`).
fn blocked_min_len(
    kernel: NarrowKernel,
    elems_per_unit: usize,
    units: usize,
    r_blk: usize,
) -> usize {
    use std::sync::atomic::Ordering::Relaxed;
    let legacy = par_min_len(elems_per_unit, units);
    if r_blk == 0 || r_blk > NARROW_R_MAX {
        return legacy;
    }
    let calls = NARROW_CALLS.fetch_add(1, Relaxed) + 1;
    let narrow =
        par_min_len_with(NARROW_TASK_ELEMS_PER_COL * r_blk, elems_per_unit, units).min(legacy);
    if narrow < legacy {
        NARROW_ENGAGED.fetch_add(1, Relaxed);
        if !NARROW_ANNOUNCED.swap(true, Relaxed) {
            eprintln!(
                "[narrow-r] floor={} calls={calls} engaged=1 first kernel={kernel:?} \
                 r_blk={r_blk} units={units} elems_per_unit={elems_per_unit} \
                 min_len {legacy} -> {narrow} threads={}",
                NARROW_TASK_ELEMS_PER_COL,
                rayon::current_num_threads()
            );
        }
    }
    narrow
}

/// Reset the narrow-floor counters and its one-time line (lane start).
pub(crate) fn narrow_r_reset() {
    use std::sync::atomic::Ordering::Relaxed;
    NARROW_CALLS.store(0, Relaxed);
    NARROW_ENGAGED.store(0, Relaxed);
    NARROW_ANNOUNCED.store(false, Relaxed);
}

/// `(narrow-regime calls, calls where the floor split a job the legacy rule
/// would have run coarser)` since the last reset.
pub fn narrow_r_counters() -> (u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (NARROW_CALLS.load(Relaxed), NARROW_ENGAGED.load(Relaxed))
}

/// The `[narrow-r]` stop receipt: present (with `engaged=0` if it never
/// fired) in every lane stop of a binary that carries the floor.
pub fn narrow_r_stop_line() -> String {
    let (calls, engaged) = narrow_r_counters();
    format!(
        "[narrow-r] floor={NARROW_TASK_ELEMS_PER_COL} r_max={NARROW_R_MAX} calls={calls} \
         engaged={engaged}"
    )
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
        .with_min_len(par_min_len(r, co * p))
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
        .with_min_len(par_min_len(p * r, co))
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

/// Maximum contraction terms or initialized/copied cells between polls in
/// the optional CPU contraction. This bounds work, not scheduling latency:
/// an allocator or a descheduled Rayon worker cannot be interrupted here.
const DEADLINE_SCALARS: usize = 256;

/// Absolute budget for a nonzero nominal product in the gradual-underflow
/// range. Under qualified round-to-nearest and gradual underflow its absolute
/// rounding error is at most half this unit, including a product rounded to
/// zero. Ordinary products retain their existing relative-error envelope.
#[inline]
pub(super) fn nominal_product_dust(a: f64, b: f64) -> f64 {
    if a != 0.0 && b != 0.0 && (a * b).abs() < f64::MIN_POSITIVE {
        f64::from_bits(1)
    } else {
        0.0
    }
}

/// Preserve a NONNEGATIVE error product before a later contraction can
/// amplify it. The ordinary product is unchanged. A nonzero exact product
/// rounded into the subnormal/zero range is widened by one unit there, which
/// dominates its absolute nearest-rounding error; no underflowed budget may
/// silently become exact zero. The caller still prices ordinary product and
/// running-sum rounding with its existing outward error expression.
#[inline]
pub(super) fn underflow_product_up(a: f64, b: f64) -> f64 {
    // A nominal overflow can leave NaN in a later error expression before
    // the pass's final non-finite firewall runs. Never panic or turn that
    // invalid budget into a finite one: +Inf propagates a conservative
    // refusal through both the per-column and grouped error lanes.
    if !a.is_finite() || !b.is_finite() || a < 0.0 || b < 0.0 {
        return f64::INFINITY;
    }
    let p = a * b;
    if a != 0.0 && b != 0.0 && p < f64::MIN_POSITIVE {
        next_up(p)
    } else {
        p
    }
}

/// One optional error-contraction term. Local directed addition retains the
/// nominal product's absolute budget even if `p` is much larger than it. By
/// combining before the single accumulator addition, this does not increase
/// the accumulation depth priced by the caller's existing gamma envelope.
#[inline]
fn error_product_with_dust(weight: f64, error: f64, nominal: f64) -> f64 {
    let p = underflow_product_up(weight, error);
    let dust = nominal_product_dust(weight, nominal);
    if dust != 0.0 {
        next_up(p + dust)
    } else {
        p
    }
}

/// A group's nominal maximum cannot decide whether a smaller column's
/// product underflows: that predicate is not monotone in the magnitude.
/// Any nonzero source group therefore carries one absolute unit for each
/// nonzero weight. Local directed addition makes this term dominate the
/// repaired per-column term without adding another running-sum rounding.
#[inline]
fn envelope_error_product_with_dust(weight: f64, error: f64, lambda: f64) -> f64 {
    let p = underflow_product_up(weight, error);
    if lambda > 0.0 && weight != 0.0 {
        next_up(p + f64::from_bits(1))
    } else {
        p
    }
}

#[derive(Clone, Copy)]
enum ErrorNominalRole<'a> {
    None,
    Actual(&'a Array2<f64>),
    EnvelopeUpper(&'a Array2<f64>),
}

/// Shared refusal used by the optional, deadline-carrying native pass.
#[inline]
pub(super) fn check_deadline(deadline: Instant) -> Result<()> {
    if Instant::now() >= deadline {
        Err(NyError::DeadlineExceeded(
            "margin_row optional native pass".into(),
        ))
    } else {
        Ok(())
    }
}

/// Deadline-carrying CPU backward convolution for the optional native pass.
///
/// This uses the blocked kernel's exact gather/tap order, ascending output
/// channel order, zero-weight skips, and non-fused [`axpy`] arithmetic. Smaller
/// column/channel tiles change ownership and interleaving only, so a completed
/// contraction is bit-identical to BOTH unrestricted CPU selector routes.
/// It never dispatches an uncancellable GPU or GEMM contraction.
///
/// `dst` is caller-owned scratch. On `Err`, it can contain an incomplete
/// contraction and MUST be discarded; only `Ok(())` authorizes the caller to
/// use it. Expiry on entry leaves `dst` untouched. Rayon joins its tasks before
/// returning, and a final deadline check refuses a late, otherwise complete
/// contraction. Existing coefficient/fold error charges remain the caller's
/// obligation, exactly as for [`conv_apply_backward`].
pub(super) fn conv_apply_backward_with_deadline(
    c: &ConvOp,
    src: &Array2<f64>,
    dst: &mut Array2<f64>,
    abs_w: bool,
    deadline: Instant,
) -> Result<()> {
    conv_apply_backward_error_inner::<true, false>(
        c,
        src,
        dst,
        abs_w,
        ErrorNominalRole::None,
        None,
        Some(deadline),
        None,
    )
}

/// Optional error convolution paired with the SAME pre-contraction nominal
/// source as the signed contraction. In addition to the existing `|W| * E`
/// envelope it retains nonzero error products rounded to subnormal/zero and
/// charges absolute dust for every corresponding underflowed `W * L` product.
/// The caller applies its existing post-contraction gamma/certify charge.
/// Positive subnormal additions are exact on the integer-subnormal lattice;
/// normal running additions retain the existing relative-error accounting.
/// Like the nominal helper, incomplete `dst` must be discarded on any `Err`.
pub(super) fn error_conv_with_deadline(
    c: &ConvOp,
    combined_error: &Array2<f64>,
    nominal_src: &Array2<f64>,
    dst: &mut Array2<f64>,
    deadline: Instant,
) -> Result<()> {
    conv_apply_backward_error_inner::<true, false>(
        c,
        combined_error,
        dst,
        true,
        ErrorNominalRole::Actual(nominal_src),
        None,
        Some(deadline),
        None,
    )
}

/// Paired absolute-underflow error contraction for ordinary Outward PerCol
/// CPU passes. No synthetic future deadline or clock reads are used. It has
/// the same error arithmetic as [`error_conv_with_deadline`], retaining the
/// unrestricted blocked kernel's original column tile width.
///
/// #mr-errconv-screen: the result is BIT-IDENTICAL to the unscreened
/// per-element kernel. A per-source-row floor ([`error_row_floors`]) proves,
/// for each `(weight, source row)` block, that no error product and no nominal
/// product of that block can reach the gradual-underflow range; such a block
/// takes the vectorized [`axpy`] that computes exactly the terms the repaired
/// arithmetic would (`underflow_product_up(w, e) == w * e` and
/// `nominal_product_dust(w, l) == 0` for every element). Every other block
/// keeps the per-element repair: a block on a VALID row (every entry finite,
/// no negative error) runs the same terms branch-free -- the widened error
/// products alone ([`error_products_widen_only`]) when the row's nominal floor
/// proves the nominal dust absent, else the full terms
/// ([`error_products_branch_free`]) -- and only an invalid row or weight keeps
/// the scalar helper. No charge is dropped: the screen only skips work whose
/// outcome it has already decided.
pub(super) fn error_conv_without_deadline(
    c: &ConvOp,
    combined_error: &Array2<f64>,
    nominal_src: &Array2<f64>,
    dst: &mut Array2<f64>,
) -> Result<()> {
    #[cfg(test)]
    if !errconv_screens_enabled() {
        return error_conv_without_deadline_unscreened(c, combined_error, nominal_src, dst);
    }
    let floors = error_row_floors(combined_error, nominal_src);
    if floors.is_some() {
        ERRCONV_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        #[cfg(test)]
        ERRCONV_TEST_SCREENED_CALLS.with(|n| n.set(n.get() + 1));
    }
    conv_apply_backward_error_inner::<false, false>(
        c,
        combined_error,
        dst,
        true,
        ErrorNominalRole::Actual(nominal_src),
        floors.as_deref(),
        None,
        None,
    )
}

/// The unscreened per-element kernel: the bit-identity oracle for
/// [`error_conv_without_deadline`]'s #mr-errconv-screen.
#[cfg(test)]
pub(super) fn error_conv_without_deadline_unscreened(
    c: &ConvOp,
    combined_error: &Array2<f64>,
    nominal_src: &Array2<f64>,
    dst: &mut Array2<f64>,
) -> Result<()> {
    conv_apply_backward_error_inner::<false, false>(
        c,
        combined_error,
        dst,
        true,
        ErrorNominalRole::Actual(nominal_src),
        None,
        None,
        None,
    )
}

/// Test-only global switch routing EVERY [`error_conv_without_deadline`] call
/// through the unscreened oracle, so whole-engine passes can be compared bit
/// for bit with the screen on and off. Concurrent tests that observe it only
/// run the (bit-identical) oracle.
#[cfg(test)]
pub(super) static ERRCONV_SCREEN_TEST_OFF: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

// Test-only per-thread count of screened contractions: proves a whole-engine
// pass on the calling thread actually reached the screen.
#[cfg(test)]
thread_local! {
    pub(super) static ERRCONV_TEST_SCREENED_CALLS: std::cell::Cell<u64> =
        const { std::cell::Cell::new(0) };
}

// #mr-errconv-screen engagement receipt (lane-scoped; reset with the census).
// Counted once per contraction / per input-spatial task, never per element,
// so the counters add no contention to the inner loop.
static ERRCONV_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static ERRCONV_FAST_BLOCKS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static ERRCONV_WIDEN_BLOCKS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static ERRCONV_EXACT_BLOCKS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static ERRCONV_SLOW_BLOCKS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
// The same screen on the per-element repaired charges of the conv bias arm
// and of concretize (valid rows run branch-free / rows that kept the scalar
// repaired loop).
static ERRCONV_BIAS_ROWS_FAST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static ERRCONV_BIAS_ROWS_SLOW: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static ERRCONV_CONC_ROWS_FAST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static ERRCONV_CONC_ROWS_SLOW: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Whether the #mr-errconv-screen is active: always in production; tests can
/// route every screened site through its unscreened arithmetic at once.
#[inline]
pub(super) fn errconv_screens_enabled() -> bool {
    #[cfg(test)]
    {
        !ERRCONV_SCREEN_TEST_OFF.load(std::sync::atomic::Ordering::SeqCst)
    }
    #[cfg(not(test))]
    {
        true
    }
}

/// Add one call's `(fast, slow)` row counts of the conv bias-arm screen.
pub(super) fn errconv_note_bias_rows(fast: u64, slow: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    if fast != 0 {
        ERRCONV_BIAS_ROWS_FAST.fetch_add(fast, Relaxed);
    }
    if slow != 0 {
        ERRCONV_BIAS_ROWS_SLOW.fetch_add(slow, Relaxed);
    }
}

/// Add one call's `(fast, slow)` row counts of the concretize screen.
pub(super) fn errconv_note_conc_rows(fast: u64, slow: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    if fast != 0 {
        ERRCONV_CONC_ROWS_FAST.fetch_add(fast, Relaxed);
    }
    if slow != 0 {
        ERRCONV_CONC_ROWS_SLOW.fetch_add(slow, Relaxed);
    }
}

/// Reset the #mr-errconv-screen counters (lane start).
pub(crate) fn errconv_screen_reset() {
    use std::sync::atomic::Ordering::Relaxed;
    ERRCONV_CALLS.store(0, Relaxed);
    ERRCONV_FAST_BLOCKS.store(0, Relaxed);
    ERRCONV_WIDEN_BLOCKS.store(0, Relaxed);
    ERRCONV_EXACT_BLOCKS.store(0, Relaxed);
    ERRCONV_SLOW_BLOCKS.store(0, Relaxed);
    ERRCONV_BIAS_ROWS_FAST.store(0, Relaxed);
    ERRCONV_BIAS_ROWS_SLOW.store(0, Relaxed);
    ERRCONV_CONC_ROWS_FAST.store(0, Relaxed);
    ERRCONV_CONC_ROWS_SLOW.store(0, Relaxed);
}

/// `(screened contractions, (weight, row) blocks proved underflow-free and
/// run as `axpy`, valid-row blocks proved free of nominal dust and run as the
/// branch-free widened products, other valid-row blocks run through the
/// branch-free exact terms, blocks that kept the scalar per-element repair)`
/// since the last reset.
pub fn errconv_screen_counters() -> (u64, u64, u64, u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (
        ERRCONV_CALLS.load(Relaxed),
        ERRCONV_FAST_BLOCKS.load(Relaxed),
        ERRCONV_WIDEN_BLOCKS.load(Relaxed),
        ERRCONV_EXACT_BLOCKS.load(Relaxed),
        ERRCONV_SLOW_BLOCKS.load(Relaxed),
    )
}

/// The `[mr-errconv]` stop receipt: present (zeros included) on every lane
/// stop of a binary that carries the screen, so its absence identifies a
/// binary without it.
pub fn errconv_screen_stop_line() -> String {
    use std::sync::atomic::Ordering::Relaxed;
    let (calls, fast, widen, exact, slow) = errconv_screen_counters();
    format!(
        "[mr-errconv] screen=row-floor calls={calls} fast_blocks={fast} widen_blocks={widen} \
         exact_blocks={exact} slow_blocks={slow} bias_rows_fast={} bias_rows_slow={} \
         conc_rows_fast={} conc_rows_slow={}",
        ERRCONV_BIAS_ROWS_FAST.load(Relaxed),
        ERRCONV_BIAS_ROWS_SLOW.load(Relaxed),
        ERRCONV_CONC_ROWS_FAST.load(Relaxed),
        ERRCONV_CONC_ROWS_SLOW.load(Relaxed)
    )
}

/// Per-source-row floors for the #mr-errconv-screen, or `None` when the two
/// sources are not aligned standard-layout matrices (the kernel then refuses
/// with its own shape error, exactly as before).
///
/// Row `i` gets `[floor, lfloor]`: `floor <= e` for every NONZERO error entry
/// and `floor <= |l|` for every NONZERO nominal entry of that row, and
/// `lfloor <= |l|` for every NONZERO nominal entry alone (`+inf` when there is
/// none). A row holding any non-finite entry or a negative error entry gets
/// `[0.0, 0.0]`, which no weight can clear, so the scalar per-element
/// arithmetic handles it (including its `+inf` refusal budget) exactly as
/// before. A VALID row therefore has `floor > 0`.
fn error_row_floors(error: &Array2<f64>, nominal: &Array2<f64>) -> Option<Vec<[f64; 2]>> {
    let r = error.ncols();
    if r == 0 || nominal.dim() != error.dim() {
        return None;
    }
    let (es, ls) = (error.as_slice()?, nominal.as_slice()?);
    Some(
        es.par_chunks(r)
            .zip(ls.par_chunks(r))
            .with_min_len((PAR_MIN_TASK_ELEMS / r).max(1))
            .map(|(er, lr)| error_row_floor(er, lr))
            .collect(),
    )
}

#[inline]
fn error_row_floor(error: &[f64], nominal: &[f64]) -> [f64; 2] {
    let (mut efloor, mut lfloor) = (f64::INFINITY, f64::INFINITY);
    for (&e, &l) in error.iter().zip(nominal) {
        if !e.is_finite() || e < 0.0 || !l.is_finite() {
            return [0.0, 0.0];
        }
        if e != 0.0 && e < efloor {
            efloor = e;
        }
        let m = l.abs();
        if m != 0.0 && m < lfloor {
            lfloor = m;
        }
    }
    [efloor.min(lfloor), lfloor]
}

/// Whether every product of weight `w` with a NONZERO entry `>= floor` in
/// magnitude is proved to stay out of the gradual-underflow range.
///
/// Correctly rounded multiplication is monotone in each nonnegative operand,
/// so for every nonzero `e >= floor` and nonzero `|l| >= floor` the stored
/// products satisfy `fl(w*e) >= fl(w*floor)` and `|fl(w*l)| = fl(w*|l|) >=
/// fl(w*floor)`. Requiring `fl(w*floor) >= 2*MIN_POSITIVE` (one binade of
/// margin) keeps both at or above `MIN_POSITIVE` under ANY IEEE rounding mode
/// and under flush-to-zero / denormals-are-zero (a subnormal floor computes
/// as zero and fails). With the row's `floor` neither
/// [`underflow_product_up`]'s widening nor [`nominal_product_dust`] can fire
/// for any element (zero entries make neither fire either); with its `lfloor`
/// the nominal dust cannot. A non-finite or nonpositive weight never passes.
#[inline]
fn screened_block_is_normal(w: f64, floor: f64) -> bool {
    w.is_finite() && w * floor >= 2.0 * f64::MIN_POSITIVE
}

/// Whether one `(weight, source row)` block meets the branch-free kernels'
/// preconditions: a VALID row (floor `> 0`: every error entry finite and not
/// negative, every nominal entry finite; see [`error_row_floor`]) and a finite
/// positive weight.
#[inline]
fn screened_row_is_valid(w: f64, floor: f64) -> bool {
    floor > 0.0 && w.is_finite() && w > 0.0
}

/// [`underflow_product_up`] accumulated over one block, branch-free, when the
/// nominal dust is proved absent for the whole block.
///
/// PRECONDITIONS: [`screened_row_is_valid`] and
/// `screened_block_is_normal(w, lfloor)` (so [`nominal_product_dust`] is `0`
/// for every element, which is why the nominal row is not read). Then each
/// term equals [`error_product_with_dust`]'s: the refusal branch cannot fire,
/// and `p = w * e` is widened exactly when `e != 0 && p < MIN_POSITIVE`; there
/// `e > 0`, so `p` is `+0.0` or positive subnormal and its [`next_up`] is
/// `bits + 1`. Adding the 0/1 widening flag to the bits computes both cases
/// without a branch (an unwidened `p`, `-0.0` and `+inf` included, keeps its
/// bits). The adds are `wrapping_add` only so the release profile's overflow
/// checks add no per-element branch: no operand here is near `u64::MAX`.
// Non-short-circuit `&` is the point: the loop must stay branch-free.
#[allow(clippy::needless_bitwise_bool)]
#[inline]
fn error_products_widen_only(acc: &mut [f64], w: f64, error: &[f64]) {
    debug_assert!(w.is_finite() && w > 0.0);
    for (a, &e) in acc.iter_mut().zip(error) {
        let p = w * e;
        let widen = u64::from((e != 0.0) & (p < f64::MIN_POSITIVE));
        *a += f64::from_bits(p.to_bits().wrapping_add(widen));
    }
}

/// [`error_product_with_dust`] accumulated over one block, branch-free.
///
/// PRECONDITIONS (the caller's [`screened_row_is_valid`]): `w` finite and
/// `> 0`; every `e` finite and not negative (`-0.0` allowed); every `l`
/// finite. Under them each term equals the scalar helper's bit for bit:
/// - the widened product is [`error_products_widen_only`]'s;
/// - [`nominal_product_dust`] fires exactly when `l != 0 && |w*l| <
///   MIN_POSITIVE` (`w*l` is never NaN for finite operands);
/// - with dust the term is `next_up(up + eta)`; `up + eta` is positive (an
///   `up` of `-0.0` from `e = -0.0` gives `eta`) or `+inf` (an overflowed
///   product), so its `next_up` is `bits + 1` when finite and itself at
///   `+inf` -- exactly [`next_up`]'s positive-finite and infinite branches;
/// - the dust flag selects between the two by an all-ones / all-zeros mask.
///
/// The accumulation `*a += term` is unchanged, so the block's result equals
/// the scalar loop's.
// Non-short-circuit `&` is the point: the loop must stay branch-free.
#[allow(clippy::needless_bitwise_bool)]
#[inline]
fn error_products_branch_free(acc: &mut [f64], w: f64, error: &[f64], nominal: &[f64]) {
    const ETA: f64 = f64::from_bits(1);
    debug_assert!(w.is_finite() && w > 0.0);
    for ((a, &e), &l) in acc.iter_mut().zip(error).zip(nominal) {
        let p = w * e;
        let widen = u64::from((e != 0.0) & (p < f64::MIN_POSITIVE));
        let up = p.to_bits().wrapping_add(widen);
        let s = f64::from_bits(up) + ETA;
        let s_up = s.to_bits().wrapping_add(u64::from(s < f64::INFINITY));
        let dust = 0u64.wrapping_sub(u64::from((l != 0.0) & ((w * l).abs() < f64::MIN_POSITIVE)));
        *a += f64::from_bits((s_up & dust) | (up & !dust));
    }
}

/// One-column grouped error contraction dominating repaired per-column
/// errors. Each `lambda_array[[j, 0]]` must bound every nominal coefficient
/// magnitude represented by source row `j`; the caller constructs it from
/// its source-owned group maxima. Universal absolute dust is required even
/// when the product at that maximum is normal. Invalid group budgets refuse
/// before touching scratch; the caller retains its existing gamma charge.
pub(super) fn envelope_error_conv_without_deadline(
    c: &ConvOp,
    combined_env: &Array2<f64>,
    lambda_array: &Array2<f64>,
    dst: &mut Array2<f64>,
) -> Result<()> {
    if combined_env.ncols() != 1
        || lambda_array.dim() != combined_env.dim()
        || lambda_array.iter().any(|v| !v.is_finite() || *v < 0.0)
    {
        return Err(NyError::InvalidSpec(
            "margin_row envelope conv requires aligned finite nonnegative group maxima".into(),
        ));
    }
    conv_apply_backward_error_inner::<false, false>(
        c,
        combined_env,
        dst,
        true,
        ErrorNominalRole::EnvelopeUpper(lambda_array),
        None,
        None,
        None,
    )
}

/// Same finite nominal kernel with the original producer's checkpoint at every
/// existing bounded arithmetic check, including Rayon worker loops. The
/// callback carries resource/cancellation checks only and cannot change an
/// arithmetic operand. Incomplete scratch must still be discarded on refusal.
pub(super) fn conv_apply_backward_with_checkpoint(
    c: &ConvOp,
    src: &Array2<f64>,
    dst: &mut Array2<f64>,
    abs_w: bool,
    deadline: Instant,
    checkpoint: &(dyn Fn() -> Result<()> + Sync),
) -> Result<()> {
    conv_apply_backward_error_inner::<true, true>(
        c,
        src,
        dst,
        abs_w,
        ErrorNominalRole::None,
        None,
        Some(deadline),
        Some(checkpoint),
    )
}

/// Paired coefficient-error kernel under the same original producer authority.
pub(super) fn error_conv_with_checkpoint(
    c: &ConvOp,
    combined_error: &Array2<f64>,
    nominal_src: &Array2<f64>,
    dst: &mut Array2<f64>,
    deadline: Instant,
    checkpoint: &(dyn Fn() -> Result<()> + Sync),
) -> Result<()> {
    conv_apply_backward_error_inner::<true, true>(
        c,
        combined_error,
        dst,
        true,
        ErrorNominalRole::Actual(nominal_src),
        None,
        Some(deadline),
        Some(checkpoint),
    )
}

#[inline]
fn check_contraction_deadline<const TIMED: bool, const COOPERATIVE: bool>(
    deadline: Option<Instant>,
    checkpoint: Option<&(dyn Fn() -> Result<()> + Sync)>,
) -> Result<()> {
    if TIMED {
        check_deadline(deadline.expect("timed contraction has a deadline"))?;
    }
    if COOPERATIVE {
        checkpoint.expect("cooperative contraction has original authority")()?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn conv_apply_backward_error_inner<const TIMED: bool, const COOPERATIVE: bool>(
    c: &ConvOp,
    src: &Array2<f64>,
    dst: &mut Array2<f64>,
    abs_w: bool,
    nominal_role: ErrorNominalRole<'_>,
    // #mr-errconv-screen: per-source-row floors from [`error_row_floors`]
    // (`Actual` role only; `None` keeps every block on the per-element path).
    row_floor: Option<&[[f64; 2]]>,
    deadline: Option<Instant>,
    checkpoint: Option<&(dyn Fn() -> Result<()> + Sync)>,
) -> Result<()> {
    check_contraction_deadline::<TIMED, COOPERATIVE>(deadline, checkpoint)?;
    let (co, ci, kh, kw) = c.kernel;
    let ip = c.ishape.1 * c.ishape.2;
    let op_ = c.oshape.1 * c.oshape.2;
    let r = src.ncols();
    if ci == 0
        || co == 0
        || r == 0
        || src.nrows() != co * op_
        || dst.dim() != (ci * ip, r)
        || !dst.is_standard_layout()
    {
        return Err(NyError::InvalidSpec(
            "margin_row deadline conv shape mismatch".into(),
        ));
    }
    let src_flat = src.as_slice().ok_or_else(|| {
        NyError::InvalidSpec("margin_row deadline conv requires standard source layout".into())
    })?;
    let envelope_role = matches!(nominal_role, ErrorNominalRole::EnvelopeUpper(_));
    let nominal_flat = match nominal_role {
        ErrorNominalRole::Actual(nominal) | ErrorNominalRole::EnvelopeUpper(nominal) => {
            if nominal.dim() != src.dim() {
                return Err(NyError::InvalidSpec(
                    "margin_row deadline error conv nominal source shape mismatch".into(),
                ));
            }
            Some(nominal.as_slice().ok_or_else(|| {
                NyError::InvalidSpec(
                    "margin_row deadline error conv requires standard nominal source layout".into(),
                )
            })?)
        }
        ErrorNominalRole::None => None,
    };
    // The screen is defined only for the per-column (`Actual`) error role and
    // only against a floor table covering every source row; anything else
    // keeps the per-element arithmetic for every block.
    let row_floor = row_floor
        .filter(|floors| floors.len() == src.nrows())
        .filter(|_| matches!(nominal_role, ErrorNominalRole::Actual(_)));
    let r_blk = ((1usize << 15) / ci).clamp(64.min(r), r);
    let r_blk = if TIMED {
        r_blk.min(DEADLINE_SCALARS)
    } else {
        r_blk
    };
    let nacc = ci.checked_mul(r_blk).ok_or_else(|| {
        NyError::InvalidSpec("margin_row deadline conv scratch size overflow".into())
    })?;
    let min_len = blocked_min_len(NarrowKernel::Backward, nacc, ip, r_blk);
    let mut dst3 = dst
        .view_mut()
        .into_shape_with_order((ci, ip, r))
        .map_err(|_| {
            NyError::InvalidSpec("margin_row deadline conv requires standard output layout".into())
        })?;
    dst3.axis_iter_mut(Axis(1))
        .into_par_iter()
        .with_min_len(min_len.max(1))
        .enumerate()
        .try_for_each_init(Vec::<f64>::new, |acc, (isp, mut out_isp)| -> Result<()> {
            check_contraction_deadline::<TIMED, COOPERATIVE>(deadline, checkpoint)?;
            if acc.len() < nacc {
                acc.try_reserve_exact(nacc - acc.len()).map_err(|_| {
                    NyError::InternalError(
                        "margin_row deadline conv scratch allocation failed".into(),
                    )
                })?;
                // Prepared producers reserve the exact per-worker accumulator
                // capacity before contraction. Preserve legacy selectors;
                // their historic policies do not supply this owned claim.
                if COOPERATIVE && acc.capacity() > nacc {
                    return Err(NyError::CpuMemoryExceeded {
                        required_bytes: acc.capacity().saturating_mul(size_of::<f64>()),
                        budget_bytes: nacc.saturating_mul(size_of::<f64>()),
                        site: "original prefix convolution worker capacity",
                    });
                }
                if TIMED {
                    while acc.len() < nacc {
                        check_contraction_deadline::<TIMED, COOPERATIVE>(deadline, checkpoint)?;
                        acc.resize((acc.len() + DEADLINE_SCALARS).min(nacc), 0.0);
                    }
                } else {
                    acc.resize(nacc, 0.0);
                }
            }
            let taps = &c.back_taps[isp];
            let (mut fast_blocks, mut widen_blocks, mut exact_blocks, mut slow_blocks) =
                (0u64, 0u64, 0u64, 0u64);
            let mut rc = 0;
            while rc < r {
                check_contraction_deadline::<TIMED, COOPERATIVE>(deadline, checkpoint)?;
                let rb = (r - rc).min(r_blk);
                if TIMED {
                    for tile in acc[..ci * rb].chunks_mut(DEADLINE_SCALARS) {
                        check_contraction_deadline::<TIMED, COOPERATIVE>(deadline, checkpoint)?;
                        tile.fill(0.0);
                    }
                } else {
                    acc[..ci * rb].fill(0.0);
                }
                // At most DEADLINE_SCALARS multiply-add terms (or weight
                // visits when all weights are zero) between checks.
                let ic_tile = if TIMED {
                    (DEADLINE_SCALARS / rb).max(1)
                } else {
                    ci
                };
                for &(ky, kx, osp) in taps {
                    check_contraction_deadline::<TIMED, COOPERATIVE>(deadline, checkpoint)?;
                    let tbase = (ky * kw + kx) * co;
                    for oc in 0..co {
                        let sbase = (oc * op_ + osp) * r + rc;
                        let srow = &src_flat[sbase..sbase + rb];
                        // 0.0 (unscreened) fails every weight's screen.
                        let [floor, lfloor] =
                            row_floor.map_or([0.0, 0.0], |floors| floors[oc * op_ + osp]);
                        for ic0 in (0..ci).step_by(ic_tile) {
                            check_contraction_deadline::<TIMED, COOPERATIVE>(deadline, checkpoint)?;
                            for ic in ic0..(ic0 + ic_tile).min(ci) {
                                let w0 = c.wt[ic * (kh * kw * co) + tbase + oc];
                                let w = if abs_w { w0.abs() } else { w0 };
                                if w == 0.0 {
                                    continue;
                                }
                                let arow = &mut acc[ic * rb..ic * rb + rb];
                                if nominal_flat.is_some()
                                    && !envelope_role
                                    && screened_block_is_normal(w, floor)
                                {
                                    // #mr-errconv-screen: every term of this
                                    // block is exactly `w * e` with no dust,
                                    // so this IS the per-element result.
                                    fast_blocks += 1;
                                    axpy(arow, w, srow);
                                } else if nominal_flat.is_some()
                                    && !envelope_role
                                    && screened_row_is_valid(w, floor)
                                    && screened_block_is_normal(w, lfloor)
                                {
                                    // #mr-errconv-screen: no nominal dust can
                                    // fire (the eta error budgets certify_up
                                    // leaves on zero errors are the usual
                                    // cause of a declined block): the widened
                                    // error products alone, branch-free.
                                    widen_blocks += 1;
                                    error_products_widen_only(arow, w, srow);
                                } else if nominal_flat.is_some()
                                    && !envelope_role
                                    && screened_row_is_valid(w, floor)
                                {
                                    // #mr-errconv-screen: a valid row whose
                                    // nominal products may underflow too: the
                                    // same per-element terms, branch-free.
                                    #[allow(clippy::unnecessary_unwrap)]
                                    // the branch condition checked it
                                    let nominal_row =
                                        &nominal_flat.expect("checked above")[sbase..sbase + rb];
                                    exact_blocks += 1;
                                    error_products_branch_free(arow, w, srow, nominal_row);
                                } else if let Some(nominal) = nominal_flat {
                                    slow_blocks += u64::from(row_floor.is_some());
                                    let nominal_row = &nominal[sbase..sbase + rb];
                                    for ((a, &e), &l) in arow.iter_mut().zip(srow).zip(nominal_row)
                                    {
                                        *a += if envelope_role {
                                            envelope_error_product_with_dust(w, e, l)
                                        } else {
                                            error_product_with_dust(w, e, l)
                                        };
                                    }
                                } else {
                                    axpy(arow, w, srow);
                                }
                            }
                        }
                    }
                }
                for ic0 in (0..ci).step_by(ic_tile) {
                    check_contraction_deadline::<TIMED, COOPERATIVE>(deadline, checkpoint)?;
                    for ic in ic0..(ic0 + ic_tile).min(ci) {
                        let mut arow = out_isp.row_mut(ic);
                        let dr = arow.as_slice_mut().expect("contiguous output row");
                        dr[rc..rc + rb].copy_from_slice(&acc[ic * rb..ic * rb + rb]);
                    }
                }
                rc += rb;
            }
            if fast_blocks != 0 {
                ERRCONV_FAST_BLOCKS.fetch_add(fast_blocks, std::sync::atomic::Ordering::Relaxed);
            }
            if widen_blocks != 0 {
                ERRCONV_WIDEN_BLOCKS.fetch_add(widen_blocks, std::sync::atomic::Ordering::Relaxed);
            }
            if exact_blocks != 0 {
                ERRCONV_EXACT_BLOCKS.fetch_add(exact_blocks, std::sync::atomic::Ordering::Relaxed);
            }
            if slow_blocks != 0 {
                ERRCONV_SLOW_BLOCKS.fetch_add(slow_blocks, std::sync::atomic::Ordering::Relaxed);
            }
            check_contraction_deadline::<TIMED, COOPERATIVE>(deadline, checkpoint)
        })?;
    check_contraction_deadline::<TIMED, COOPERATIVE>(deadline, checkpoint)
}

/// [`axpy`] with bounded deadline checks for the optional pass. On `Err`, the
/// caller must discard its scratch: some complete column tiles may have been
/// changed. A live completion keeps every cell's original multiplication and
/// addition order, including signed-zero behavior.
pub(super) fn axpy_with_deadline(
    acc: &mut [f64],
    w: f64,
    src: &[f64],
    deadline: Instant,
) -> Result<()> {
    check_deadline(deadline)?;
    if acc.len() != src.len() {
        return Err(NyError::InvalidSpec(
            "margin_row deadline axpy shape mismatch".into(),
        ));
    }
    for (a, s) in acc
        .chunks_mut(DEADLINE_SCALARS)
        .zip(src.chunks(DEADLINE_SCALARS))
    {
        check_deadline(deadline)?;
        axpy(a, w, s);
    }
    check_deadline(deadline)
}

#[cfg(test)]
#[path = "net_deadline_tests.rs"]
mod deadline_tests;

#[cfg(test)]
#[path = "net_forward_deadline_tests.rs"]
mod forward_deadline_tests;

#[cfg(test)]
#[path = "errconv_screen_tests.rs"]
mod errconv_screen_tests;

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
        .with_min_len(par_min_len(r, ci * ip))
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
    conv_backward_blocked_min_len(c, src, dst, abs_w, None);
}

/// [`conv_backward_blocked`] with an explicit rayon `min_len` over input
/// spatial positions (`None` = the production policy, [`blocked_min_len`]);
/// see [`conv_forward_blocked_min_len`].
pub(crate) fn conv_backward_blocked_min_len(
    c: &ConvOp,
    src: &Array2<f64>,
    dst: &mut Array2<f64>,
    abs_w: bool,
    min_len: Option<usize>,
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
    let min_len =
        min_len.unwrap_or_else(|| blocked_min_len(NarrowKernel::Backward, ci * r_blk, ip, r_blk));
    let mut dst3 = dst
        .view_mut()
        .into_shape_with_order((ci, ip, r))
        .expect("standard layout reshape");
    dst3.axis_iter_mut(Axis(1))
        .into_par_iter()
        .with_min_len(min_len.max(1))
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

#[cfg(test)]
mod backward_gemm_probe {
    use super::{compile_conv, conv_backward_blocked};
    use ndarray::Array2;
    use std::time::Instant;

    /// `#bi-gemm` GATING MEASUREMENT — run this BEFORE writing a GEMM backward.
    ///
    /// # The claim under test
    ///
    /// `conv_backward_blocked` is a rank-1 `axpy` accumulation of what is
    /// algebraically one GEMM per kernel tap:
    ///   `acc[ci x r] += W_tap[ci x co] @ src_tap[co x r]`.
    /// Its arithmetic intensity is ~2 flops per 3 doubles touched (read acc,
    /// read src, write acc) = 0.67 flop/double, and `axpy` is a separate
    /// FMUL+FADD (`*a += w * s`, non-contracted by construction for bit
    /// identity), so it is L1-BANDWIDTH-bound rather than compute-bound.
    /// Meanwhile `conv_apply_forward` IN THIS FILE already has a default-on faer
    /// GEMM path (`r >= 512 && k >= 2`); the backward has none — no GEMM, no
    /// f32, no GPU dispatch. Backward-interm is 49.3% of the cifar100
    /// resnet_large root build.
    ///
    /// # Why this is the gate and not the build
    ///
    /// If a same-shape faer GEMM is not materially faster than the scalar
    /// kernel on THIS machine, the entire `#bi-gemm` direction is dead and
    /// nothing was spent on it. Run this first. It is a MEASUREMENT, not an
    /// assertion about a target: it prints and only fails if the kernel is
    /// outright broken.
    #[test]
    fn backward_kernel_vs_gemm_throughput() {
        // resnet-ish block, sized so the test runs in seconds rather than the
        // full layer (the RATIO is what transfers, not the absolute).
        let (co, ci, kh, kw) = (64usize, 64usize, 3usize, 3usize);
        let (h, w) = (8usize, 8usize);
        let r = 128usize;
        let weight: Vec<f64> = (0..co * ci * kh * kw)
            .map(|i| ((i % 17) as f64 - 8.0) / 32.0)
            .collect();
        let conv = compile_conv(
            0,
            &weight,
            &vec![0.0; co],
            &vec![0.0; co],
            0.0,
            (co, ci, kh, kw),
            (1, 1),
            (1, 1, 1, 1),
            (ci, h, w),
            (co, h, w),
        );
        let ip = h * w;
        let op_ = h * w;
        let src = Array2::from_shape_fn((co * op_, r), |(a, b)| ((a + 3 * b) % 13) as f64 / 8.0);
        let mut dst = Array2::<f64>::zeros((ci * ip, r));

        // The kernel under test.
        let t0 = Instant::now();
        conv_backward_blocked(&conv, &src, &mut dst, false);
        let scalar_s = t0.elapsed().as_secs_f64().max(1e-9);

        // Total MACs the backward performs: every (ci, ip, r) accumulator takes
        // one tap-weighted contribution per (tap, co).
        let macs = (ci * ip * r * kh * kw * co) as f64;
        let scalar_gflops = 2.0 * macs / scalar_s / 1e9;

        // The same arithmetic expressed as kh*kw GEMMs of (ci x co) @ (co x ip*r).
        let a: Vec<f64> = (0..ci * co)
            .map(|i| ((i % 11) as f64 - 5.0) / 16.0)
            .collect();
        let b: Vec<f64> = (0..co * ip * r)
            .map(|i| ((i % 7) as f64 - 3.0) / 8.0)
            .collect();
        // ⚠ `mat_mul_f64_row_major` returns an OWNED `Mat<f64>`, so every call
        // allocates a (ci x ip*r) output — 4 MB here, kh*kw times — while the
        // scalar kernel accumulates IN PLACE and allocates nothing. Timing the
        // two naively compares GEMM+malloc against a malloc-free kernel and
        // reports a ratio that is mostly allocator. Measure the allocation
        // separately and report BOTH figures; the with-alloc number is what an
        // in-place GEMM would have to beat, the without-alloc number is what the
        // arithmetic itself costs.
        let t1 = Instant::now();
        let mut sink = 0.0f64;
        for _ in 0..(kh * kw) {
            let out = crate::faer_parallelism::mat_mul_f64_row_major(&a, ci, co, &b, ip * r);
            sink += out[(0, 0)];
        }
        let gemm_s = t1.elapsed().as_secs_f64().max(1e-9);
        assert!(sink.is_finite());

        // Allocation-only control: same count, same shape, no arithmetic.
        let t2 = Instant::now();
        let mut asink = 0usize;
        for _ in 0..(kh * kw) {
            let z = vec![0.0f64; ci * ip * r];
            asink += z.len();
        }
        let alloc_s = t2.elapsed().as_secs_f64().max(1e-9);
        assert!(asink > 0);

        let gemm_gflops = 2.0 * macs / gemm_s / 1e9;
        let gemm_net_s = (gemm_s - alloc_s).max(1e-9);
        let gemm_net_gflops = 2.0 * macs / gemm_net_s / 1e9;

        // ⚠ THE SHAPE ABOVE IS SKINNY-K AND THAT IS NOT THE ONLY OPTION.
        // Per-tap, the backward is (ci x co) @ (co x ip*r), so k = co = 64-128.
        // Skinny-k GEMMs do not approach peak in ANY library, so a poor number
        // there indicts the FORMULATION, not GEMM. The formulation with a good
        // shape is the BATCHED-TAP one — k = co*kh*kw (576 here) — which is the
        // backward analogue of the im2col the FORWARD path already uses
        // (`conv_apply_forward_faer`). Measure that ceiling too, on the same
        // total flops, so the gate is decided by the best available formulation
        // rather than the first one tried.
        let kfat = co * kh * kw;
        let nfat = ip * r / (kh * kw).max(1);
        let a2: Vec<f64> = (0..ci * kfat)
            .map(|i| ((i % 11) as f64 - 5.0) / 16.0)
            .collect();
        let b2: Vec<f64> = (0..kfat * nfat)
            .map(|i| ((i % 7) as f64 - 3.0) / 8.0)
            .collect();
        let t3 = Instant::now();
        let out2 = crate::faer_parallelism::mat_mul_f64_row_major(&a2, ci, kfat, &b2, nfat);
        let fat_s = t3.elapsed().as_secs_f64().max(1e-9);
        assert!(out2[(0, 0)].is_finite());
        let fat_macs = (ci * kfat * nfat) as f64;
        let fat_gflops = 2.0 * fat_macs / fat_s / 1e9;
        eprintln!(
            "[bi-gemm-probe] BATCHED-TAP shape ({ci}x{kfat} @ {kfat}x{nfat}): \
{fat_s:.4}s = {fat_gflops:.1} GFLOP/s | ratio vs scalar {:.2}x \
<- THIS is the ceiling that decides the gate",
            fat_gflops / scalar_gflops.max(1e-9)
        );
        eprintln!(
            "[bi-gemm-probe] alloc-only {alloc_s:.4}s ({:.0}% of the GEMM timing) \
=> GEMM net {gemm_net_s:.4}s = {gemm_net_gflops:.1} GFLOP/s",
            100.0 * alloc_s / gemm_s
        );

        eprintln!(
            "[bi-gemm-probe] scalar {scalar_s:.4}s = {scalar_gflops:.1} GFLOP/s | \
GEMM+alloc {gemm_s:.4}s = {gemm_gflops:.1} GFLOP/s (ratio {:.2}x) | \
GEMM net-of-alloc = {gemm_net_gflops:.1} GFLOP/s (ratio {:.2}x) | \
GATE on the NET ratio >= 2.5; an in-place GEMM pays no allocation",
            gemm_gflops / scalar_gflops.max(1e-9),
            gemm_net_gflops / scalar_gflops.max(1e-9)
        );

        // ── IS FAER EVEN PARALLEL? ─────────────────────────────────────────
        // `current_par()` (faer_parallelism.rs:663) returns
        // `faer::get_global_parallelism()` outside a Rayon task. But
        // `set_global_parallelism` is called NOWHERE in this workspace, so that
        // global is whatever faer defaults to. If it is sequential then EVERY
        // faer matmul in ny runs single-threaded IN PRODUCTION — including the
        // default-on forward conv GEMM (`conv_apply_forward_faer`) — and this
        // probe's ratios are measuring one core against one core.
        eprintln!(
            "[bi-gemm-probe] faer global parallelism = {:?} | rayon threads available = {}",
            faer::get_global_parallelism(),
            rayon::current_num_threads()
        );
        // Same calibration matmul under EXPLICIT rayon parallelism. If this is
        // much faster than the default-parallelism run below, the missing
        // `set_global_parallelism` call is a live production defect, not a test
        // artifact.
        {
            let cn2 = 512usize;
            let pa: Vec<f64> = (0..cn2 * cn2)
                .map(|i| ((i % 13) as f64 - 6.0) / 32.0)
                .collect();
            let pb: Vec<f64> = (0..cn2 * cn2)
                .map(|i| ((i % 9) as f64 - 4.0) / 16.0)
                .collect();
            let a_ref = faer::MatRef::from_row_major_slice(&pa, cn2, cn2);
            let b_ref = faer::MatRef::from_row_major_slice(&pb, cn2, cn2);
            let mut out = faer::Mat::<f64>::zeros(cn2, cn2);
            let t = Instant::now();
            faer::linalg::matmul::matmul(
                &mut out,
                faer::Accum::Replace,
                a_ref,
                b_ref,
                1.0,
                faer::Par::rayon(0),
            );
            let ps = t.elapsed().as_secs_f64().max(1e-9);
            eprintln!(
                "[bi-gemm-probe] SAME 512^3 under Par::rayon(0): {ps:.4}s = {:.1} GFLOP/s",
                2.0 * (cn2 * cn2 * cn2) as f64 / ps / 1e9
            );
            assert!(out[(0, 0)].is_finite());
        }

        // ── CALIBRATION: what can this wrapper do AT ALL? ──────────────────
        // Without a peak reference the numbers above are uninterpretable. Both
        // the scalar kernel (3.4 GFLOP/s) and the GEMM (1.9-2.4) sit far below
        // any plausible f64 peak, which is the signature of a constraint COMMON
        // to both — threading, layout, or copies inside the row-major wrapper —
        // rather than of the algorithms. One large square matmul settles it:
        //   ~peak  => the conv shapes really are too small/skinny, gate failure REAL
        //   ~2 GF  => the wrapper/harness is the ceiling, every ratio above VOID
        let cn = 512usize;
        let ca: Vec<f64> = (0..cn * cn)
            .map(|i| ((i % 13) as f64 - 6.0) / 32.0)
            .collect();
        let cb: Vec<f64> = (0..cn * cn)
            .map(|i| ((i % 9) as f64 - 4.0) / 16.0)
            .collect();
        let t4 = Instant::now();
        let cout = crate::faer_parallelism::mat_mul_f64_row_major(&ca, cn, cn, &cb, cn);
        let cal_s = t4.elapsed().as_secs_f64().max(1e-9);
        assert!(cout[(0, 0)].is_finite());
        let cal_gflops = 2.0 * (cn * cn * cn) as f64 / cal_s / 1e9;
        eprintln!(
            "[bi-gemm-probe] CALIBRATION {cn}^3 f64 matmul: {cal_s:.4}s = {cal_gflops:.1} GFLOP/s \
| if this is also ~2 GFLOP/s the wrapper is the ceiling and EVERY ratio above is VOID, \
not negative"
        );

        // Only fails if a kernel is broken, not if the ratio disappoints — the
        // point is to LEARN the ratio, not to assert a hoped-for one.
        assert!(scalar_gflops > 0.0 && gemm_gflops > 0.0);
    }
}

/// #narrow-r (throughput plan B1(d)): the narrow-width granularity floor
/// changes WHICH task owns each spatial position, never any output element's
/// fold, so the blocked kernels stay bit-identical to their row-grain oracles
/// under every partition.
#[cfg(test)]
mod narrow_partition_tests {
    use super::{
        blocked_min_len, compile_conv, conv_backward_blocked, conv_backward_blocked_min_len,
        conv_backward_icgrain, conv_forward_blocked, conv_forward_blocked_min_len,
        conv_forward_ocgrain, par_min_len, ConvOp, NarrowKernel, NARROW_R_MAX,
    };
    use ndarray::Array2;

    /// Deterministic LCG in [-2, 2), with exact +0.0 / -0.0 sprinkled in.
    struct Lcg(u64);

    impl Lcg {
        fn next_f64(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit = (self.0 >> 11) as f64 / (1u64 << 53) as f64;
            4.0 * unit - 2.0
        }

        fn value(&mut self, i: usize) -> f64 {
            match i % 11 {
                3 => 0.0,
                7 => -0.0,
                _ => self.next_f64(),
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn conv(
        rng: &mut Lcg,
        kernel: (usize, usize, usize, usize),
        stride: (usize, usize),
        pads: (usize, usize, usize, usize),
        ishape: (usize, usize, usize),
        oshape: (usize, usize, usize),
    ) -> ConvOp {
        let (co, ci, kh, kw) = kernel;
        let weight: Vec<f64> = (0..co * ci * kh * kw).map(|i| rng.value(i)).collect();
        compile_conv(
            0,
            &weight,
            &vec![0.0; co],
            &vec![0.0; co],
            0.0,
            kernel,
            stride,
            pads,
            ishape,
            oshape,
        )
    }

    fn bits(m: &Array2<f64>) -> Vec<u64> {
        m.iter().map(|v| v.to_bits()).collect()
    }

    /// `narrow_partition_bit_identical`: forced single-task vs maximal split vs
    /// the production policy vs the row-grain oracle, to the last bit, at
    /// R = 1..4, signed and |W| weights, padded taps, zero and -0.0 weights,
    /// -0.0 inputs, on a multi-thread pool so the split really splits. Also
    /// pins that the floor ENGAGES where it is meant to (a >= 2 Ki tensor at
    /// R = 1 that the legacy rule ran as one task) and never coarsens.
    #[test]
    fn narrow_partition_bit_identical() {
        let mut rng = Lcg(0x5EED_0B1D);
        let cases = [
            // 4x4 3x3 s1 p1 on 24x24: both tensors 2304 >= 2 Ki, < 16 Ki —
            // exactly the band the legacy rule serialised at R = 1.
            conv(
                &mut rng,
                (4, 4, 3, 3),
                (1, 1),
                (1, 1, 1, 1),
                (4, 24, 24),
                (4, 24, 24),
            ),
            // Stride 2 downsample (backward tensor 3200, forward 800).
            conv(
                &mut rng,
                (8, 8, 3, 3),
                (2, 2),
                (1, 1, 1, 1),
                (8, 20, 20),
                (8, 10, 10),
            ),
            // 1x1 kernel, no padding, forward tensor 16 Ki.
            conv(
                &mut rng,
                (16, 4, 1, 1),
                (1, 1),
                (0, 0, 0, 0),
                (4, 32, 32),
                (16, 32, 32),
            ),
            // Asymmetric 2x3 kernel, stride (2, 1), pads (0, 1, 1, 0).
            conv(
                &mut rng,
                (5, 4, 2, 3),
                (2, 1),
                (0, 1, 1, 0),
                (4, 33, 41),
                (5, 17, 40),
            ),
        ];
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("4-thread pool");
        pool.install(|| {
            for c in &cases {
                let n_in_t = c.ishape.0 * c.ishape.1 * c.ishape.2;
                let n_out_t = c.oshape.0 * c.oshape.1 * c.oshape.2;
                let p = c.oshape.1 * c.oshape.2;
                let ip = c.ishape.1 * c.ishape.2;
                for r in 1..=NARROW_R_MAX {
                    for abs_w in [false, true] {
                        // ---- forward ----
                        let src = Array2::from_shape_fn((n_in_t, r), |(i, j)| rng.value(i * 7 + j));
                        let mut oracle = Array2::<f64>::zeros((n_out_t, r));
                        conv_forward_ocgrain(c, &src, &mut oracle, abs_w);
                        let want = bits(&oracle);
                        for (label, min_len) in
                            [("single", Some(p)), ("split", Some(1)), ("policy", None)]
                        {
                            let mut got = Array2::<f64>::zeros((n_out_t, r));
                            conv_forward_blocked_min_len(c, &src, &mut got, abs_w, min_len);
                            assert_eq!(
                                bits(&got),
                                want,
                                "forward {label} kernel={:?} r={r} abs_w={abs_w}",
                                c.kernel
                            );
                        }
                        let mut dflt = Array2::<f64>::zeros((n_out_t, r));
                        conv_forward_blocked(c, &src, &mut dflt, abs_w);
                        assert_eq!(bits(&dflt), want, "forward default r={r}");
                        // ---- backward ----
                        let src =
                            Array2::from_shape_fn((n_out_t, r), |(i, j)| rng.value(i * 5 + j));
                        let mut oracle = Array2::<f64>::zeros((n_in_t, r));
                        conv_backward_icgrain(c, &src, &mut oracle, abs_w);
                        let want = bits(&oracle);
                        for (label, min_len) in
                            [("single", Some(ip)), ("split", Some(1)), ("policy", None)]
                        {
                            let mut got = Array2::<f64>::zeros((n_in_t, r));
                            conv_backward_blocked_min_len(c, &src, &mut got, abs_w, min_len);
                            assert_eq!(
                                bits(&got),
                                want,
                                "backward {label} kernel={:?} r={r} abs_w={abs_w}",
                                c.kernel
                            );
                        }
                        let mut dflt = Array2::<f64>::zeros((n_in_t, r));
                        conv_backward_blocked(c, &src, &mut dflt, abs_w);
                        assert_eq!(bits(&dflt), want, "backward default r={r}");
                    }
                }
            }
            // ENGAGEMENT: case 0 at R = 1 (r_blk = 1) — the legacy rule ran the
            // whole 2304-element job as ONE task; the floor splits it.
            let c = &cases[0];
            let (ci, ip) = (c.kernel.1, c.ishape.1 * c.ishape.2);
            let legacy = par_min_len(ci, ip);
            let floor = blocked_min_len(NarrowKernel::Backward, ci, ip, 1);
            assert_eq!(legacy, ip, "the legacy rule serialises this job");
            assert!(
                floor < legacy,
                "the floor must split it: {floor} vs {legacy}"
            );
            // NEVER COARSER, and inert above the narrow band.
            for epu in [1usize, 3, 16, 64, 256, 4096] {
                for units in [1usize, 7, 64, 900, 4096, 50_000] {
                    for r_blk in 1..=12 {
                        let legacy = par_min_len(epu * r_blk, units);
                        let got = blocked_min_len(NarrowKernel::Forward, epu * r_blk, units, r_blk);
                        assert!(got <= legacy, "coarsened: {got} > {legacy}");
                        assert!(got >= 1);
                        if r_blk > NARROW_R_MAX {
                            assert_eq!(got, legacy, "the floor must be inert at r_blk={r_blk}");
                        }
                    }
                }
            }
        });
    }
}
