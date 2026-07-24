// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Certified forward-linear compositions for image-class conv DAGs
//! (#vnncomp-image-forward-linear).
//!
//! Fills the forward-substitution (DeepPoly-style) op surface for
//! Conv2d / ConvTranspose2d / BatchNorm / binary Add / ReLU / Linear / shape
//! ops so 17-conv ResNets and cGAN-style generator chains get O(L)
//! finite intermediate bounds instead of exploding plain-IBP intervals (which
//! drive the CROWN backward NaN firewall and a vacuous -inf root bound).
//!
//! # Soundness contract (#vnncomp-aw-soundness)
//!
//! Every composition here is on the production verdict path, so every
//! floating-point rounding is certified:
//!
//! * Exact affine maps (Conv2d, Linear) are composed with the upstream
//!   [`LinearBounds`] via the **center–radius identity** in **f64**:
//!   `C⁺U_l + C⁻U_u = C·U_c − |C|·U_r` and `C⁺U_u + C⁻U_l = C·U_c + |C|·U_r`
//!   with `U_c = (U_l+U_u)/2`, `U_r = (U_u−U_l)/2` (exact in f64 for f32
//!   inputs; the identity is algebraic and needs no sign assumption on `U_r`).
//!   The f64 GEMM accumulation error is bounded by the Higham factor
//!   `γ_{K+4}·S` with `S = |C|(|U_c|+|U_r|)` (order-independent, so any IEEE
//!   f64 GEMM backend is admissible — see `sound_f64_gemm`).
//! * The final f64→f32 round-to-nearest coefficient cast gap is **measured
//!   per entry** (`|stored_f32 − value_f64|`, exact because f32→f64 widening
//!   is exact).
//! * Both error sources are discharged immediately through the existing
//!   certified coefficient-error channel semantics: the per-row penalty
//!   `Σ_j err_ij·max(|x_l_j|,|x_u_j|)` — exactly what
//!   `LinearBounds::fold_coeff_err_into_bias` / `concretize_sound` would
//!   apply — is folded OUTWARD into the bias (lower decreases, upper
//!   increases) with directed rounding. Folding eagerly at each op is
//!   algebraically identical to carrying the error matrices and discharging
//!   at concretization (the downstream `C⁺/C⁻` bias split multiplies a
//!   symmetric ±p widening by exactly `|C|`, the same transform
//!   `coeff_err_carrier` propagation applies), but needs no O(N·n) error
//!   matrices.
//! * All bias arithmetic runs in f64 and commits with directed f32 rounding
//!   (`next_down_f32` / `next_up_f32`); the 1-ULP directed step dominates the
//!   ~1e-12-relative residual f64 rounding by >4 orders of magnitude.
//! * Non-finite coefficients degrade the affected row to `A=0, b=±inf`
//!   (sound, maximally loose) via `detect_and_fix_nonfinite_rows`; NaN biases
//!   are mapped to ∓inf. The `LinearBounds::new_or_conservative` NaN firewall
//!   stays as the backstop.

use std::time::Instant;

use ndarray::{s, Array1, Array2, ArrayD, ArrayView2, IxDyn};
use ny_core::{checked_shape_product, GemmEngine, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};
use rayon::prelude::*;

use crate::bounds::{safe_mul_for_bounds_f64, LinearBounds};
use crate::layers::activations::relu::relu_linear_relaxation;
use crate::layers::convolution::crown_helpers::detect_and_fix_nonfinite_rows;
use crate::layers::convolution::{Conv2dLayer, ConvTranspose2dLayer};
use crate::layers::linear::crown_single_gamma_n_f32 as gamma_n_f32;
use crate::layers::linear::crown_single_gamma_n_f64 as gamma_n_f64;
use crate::layers::BatchNormLayer;

/// Blanket multiplicative inflation applied to every accumulated f64 penalty
/// before it is folded into the bias. Covers all second-order f64 roundings in
/// the penalty accumulation itself (relative error ≤ γ_n^f64 ≈ 5e-13 for the
/// widest sums here) by >5 orders of magnitude while staying negligible
/// relative to the penalty.
const PENALTY_INFLATE: f64 = 1.0 + 1e-7;

/// Opt-in (`NY_FORWARD_LINEAR_F32=1`, default OFF) sound f32 fast path for the
/// forward-linear composition's big *value* GEMMs (`A·W`, `|A|·|W|`). The dense
/// f64 conv-forward composition is the measured #1 cost of BaB-bound conv-ResNet
/// instances (cifar100_resnet_medium), and f64 cannot use the wgpu GPU (no f64)
/// so it stalls on the weak GB10 cuBLAS Dgemm (0.41 TF/s). Routing the value
/// GEMMs through the fast RN-f32 path (cuBLAS Sgemm, ~40× on the GB10) is SOUND
/// because the larger f32 accumulation error is bounded by the Higham factor
/// `γ_{K+4}^f32·S` (vs the f64 `γ_{K+4}^f64·S`) plus an FTZ underflow guard —
/// the SAME `S`-scaled certified-error channel the f64 path already discharges
/// into the bias (see [`compose_conv2d_forward`]). PRECISION-NEGATIVE (looser
/// intermediates → risk of regressing categories ny currently verifies), so it
/// stays default-OFF pending broad verdict-parity validation. The S-BASE GEMMs
/// (`v_abs` → `s_coeff`/`s_bias`) stay f64 so `S` is never under-estimated.
#[inline]
fn forward_linear_f32_gemm_enabled() -> bool {
    // Uncached env read (matches `forward_linear_reference_enabled`); the
    // forward composition runs O(alpha-iters·layers) times per instance with
    // huge GEMMs, so a per-call env probe is negligible.
    matches!(
        std::env::var("NY_FORWARD_LINEAR_F32").ok().as_deref(),
        Some("1")
    )
}

/// FTZ-safe underflow addend for the f32 value GEMMs, discharged into the bias.
/// Under flush-to-zero a length-`k` f32 dot product can lose ≤ `2k` roundings of
/// `< 2^-126` each (design mirror of the f32-abs-sum seam, `crown_single.rs`);
/// there are two value GEMMs (`A·W` center and `|A|·|W|` radius), so `4k·2^-126`
/// bounds the per-coefficient underflow error, and `·Σ_j mag_j` discharges it
/// across the input columns exactly as the `γ·S` penalty is discharged.
#[inline]
fn forward_f32_ftz_bias(contraction: usize, mag_sum: f64) -> f64 {
    4.0 * (contraction as f64) * 2f64.powi(-126) * mag_sum
}

/// Row-major RN-f32 GEMM `C = A @ B` for the forward-linear value seam. `a`/`b`
/// hold f32-representable-or-wider f64 values (conv im2col of the upstream
/// center/radius, and the f32 kernel widened to f64); both are rounded to f32,
/// multiplied by a plain IEEE round-to-nearest f32 GEMM (the coefficient error
/// of the resulting f32 accumulation is charged to the caller's `γ_n^f32·S`
/// penalty). Tries the process-global fast f32 accelerator (cuBLAS `Sgemm` on
/// `--features cuda`) first, then the per-call engine's `gemm_f32` (so tests can
/// inject an engine without the process-global `OnceLock`). Returns `None` (→
/// caller falls back to the certified f64 path) on any unavailable/failed
/// engine or dimension mismatch.
fn forward_value_gemm_f32(
    m: usize,
    k: usize,
    n: usize,
    a: &[f64],
    b: &[f64],
    engine: Option<&dyn GemmEngine>,
) -> Option<Vec<f64>> {
    let a32: Vec<f32> = a.iter().map(|&x| x as f32).collect();
    let b32: Vec<f32> = b.iter().map(|&x| x as f32).collect();
    let r32 = crate::fast_f32_gemm::with_engine(|e| e.gemm_f32(m, k, n, &a32, &b32).ok())
        .flatten()
        .or_else(|| engine.and_then(|e| e.gemm_f32(m, k, n, &a32, &b32).ok()))?;
    if r32.len() != m * n {
        return None;
    }
    Some(r32.into_iter().map(f64::from).collect())
}

/// Row-major f64 GEMM `C = A @ B` with a certified-soundness contract: the
/// backend must compute plain IEEE round-to-nearest **f64** dot products (any
/// summation order — the Higham `γ_n·S` bound used by callers is
/// order-independent). Tries the process-global sound f64 accelerator
/// (e.g. cuBLAS Dgemm), then the per-call engine's `gemm_f64`, then faer CPU.
fn certified_f64_gemm(
    m: usize,
    k: usize,
    n: usize,
    a: &[f64],
    b: &[f64],
    engine: Option<&dyn GemmEngine>,
    allow_f32: bool,
) -> Vec<f64> {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    // Sound f32 fast path (opt-in, VALUE GEMMs only — never the S base). The
    // caller resolves the seam gate and charges the larger `γ_{K+4}^f32·S` + FTZ
    // error to the bias, so any RN-f32 summation order is admissible; falls
    // through to the certified f64 path when the seam is off for this GEMM or no
    // f32 engine is available.
    if allow_f32 {
        if let Some(res) = forward_value_gemm_f32(m, k, n, a, b, engine) {
            return res;
        }
    }
    if let Some(Some(res)) =
        crate::sound_f64_gemm::with_engine(|eng| eng.gemm_f64(m, k, n, a, b).ok())
    {
        if res.len() == m * n {
            return res;
        }
    }
    if let Some(eng) = engine {
        if let Ok(res) = eng.gemm_f64(m, k, n, a, b) {
            if res.len() == m * n {
                return res;
            }
        }
    }
    let am = faer::Mat::<f64>::from_fn(m, k, |i, j| a[i * k + j]);
    let bm = faer::Mat::<f64>::from_fn(k, n, |i, j| b[i * n + j]);
    let mut dst = faer::Mat::<f64>::zeros(m, n);
    faer::linalg::matmul::matmul(
        &mut dst,
        faer::Accum::Replace,
        &am,
        &bm,
        1.0,
        crate::faer_parallelism::current_par(),
    );
    let mut out = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            out[i * n + j] = dst[(i, j)];
        }
    }
    out
}

/// Commit an f64 bias value to f32 with directed rounding after folding the
/// certified penalty outward. NaN degrades to the conservative infinity.
fn commit_lower_bias(value: f64, penalty: f64) -> f32 {
    let v = value - penalty * PENALTY_INFLATE;
    if v.is_nan() {
        f32::NEG_INFINITY
    } else {
        next_down_f32(v as f32)
    }
}

fn commit_upper_bias(value: f64, penalty: f64) -> f32 {
    let v = value + penalty * PENALTY_INFLATE;
    if v.is_nan() {
        f32::INFINITY
    } else {
        next_up_f32(v as f32)
    }
}

/// Cast one composed f64 coefficient pair to f32 (round-to-nearest) and
/// accumulate the measured cast gap, weighted by the input-box magnitude, into
/// the per-row penalties. The f32→f64 widening of the stored value is exact,
/// so `gap = |stored − value|` is the true cast error.
#[inline]
fn cast_coeff_with_gap(value: f64, mag: f64, penalty: &mut f64) -> f32 {
    let stored = value as f32;
    if stored.is_finite() {
        *penalty += (stored as f64 - value).abs() * mag;
    }
    stored
}

/// Contiguous row-major views of the upstream coefficient matrices. Fails
/// closed (caller falls back to IBP) when a matrix is not standard-layout —
/// `LinearBounds` are constructed row-major everywhere, so this is defensive.
fn upstream_slices<'a>(
    upstream: &'a LinearBounds,
    node_name: &str,
) -> Result<(&'a [f32], &'a [f32])> {
    match (upstream.lower_a().as_slice(), upstream.upper_a().as_slice()) {
        (Some(l), Some(u)) => Ok((l, u)),
        _ => Err(NyError::UnsupportedConfiguration(format!(
            "forward-linear image bounds: node '{node_name}' upstream coefficients are not \
             standard-layout"
        ))),
    }
}

/// Geometry of a 2D convolution derived from the layer + the predecessor shape.
/// Shared with the forward-map alpha optimizer (`alpha_opt`, #w4-root-alpha-opt).
pub(super) struct ConvGeometry {
    pub(super) in_c: usize,
    pub(super) in_h: usize,
    pub(super) in_w: usize,
    pub(super) out_c: usize,
    pub(super) out_h: usize,
    pub(super) out_w: usize,
    pub(super) kh: usize,
    pub(super) kw: usize,
    pub(super) stride: (usize, usize),
    pub(super) padding: (usize, usize),
    pub(super) dilation: (usize, usize),
    /// Contraction width per output: in_c * kh * kw.
    pub(super) contraction: usize,
}

impl ConvGeometry {
    pub(super) fn conv_in_size(&self) -> usize {
        self.in_c * self.in_h * self.in_w
    }
    pub(super) fn conv_out_size(&self) -> usize {
        self.out_c * self.out_h * self.out_w
    }
    pub(super) fn spatial(&self) -> usize {
        self.out_h * self.out_w
    }
}

pub(super) fn resolve_conv_geometry(
    node_name: &str,
    layer: &Conv2dLayer,
    pred_shape: &[usize],
    upstream_outputs: usize,
    output_dim: usize,
) -> Result<ConvGeometry> {
    if layer.groups != 1 {
        return Err(NyError::UnsupportedConfiguration(format!(
            "forward-linear image bounds: node '{node_name}' has groups={} (only groups=1 supported)",
            layer.groups
        )));
    }
    let kshape = layer.kernel.shape();
    if kshape.len() != 4 {
        return Err(NyError::UnsupportedConfiguration(format!(
            "forward-linear image bounds: node '{node_name}' kernel must be 4-D, got {kshape:?}"
        )));
    }
    let (out_c, in_c, kh, kw) = (kshape[0], kshape[1], kshape[2], kshape[3]);

    // Predecessor shape: strip leading batch-1 dims down to (C, H, W).
    let mut dims: Vec<usize> = pred_shape.to_vec();
    while dims.len() > 3 && dims[0] == 1 {
        dims.remove(0);
    }
    if dims.len() != 3 || dims[0] != in_c {
        return Err(NyError::UnsupportedConfiguration(format!(
            "forward-linear image bounds: node '{node_name}' expects (C={in_c}, H, W) input, got {pred_shape:?}"
        )));
    }
    let (in_h, in_w) = (dims[1], dims[2]);
    if in_c * in_h * in_w != upstream_outputs {
        return Err(NyError::ShapeMismatch {
            expected: vec![in_c * in_h * in_w],
            got: vec![upstream_outputs],
        });
    }

    let (sh, sw) = layer.stride;
    let (ph, pw) = layer.padding;
    let (dh, dw) = layer.dilation;
    if sh == 0 || sw == 0 || dh == 0 || dw == 0 {
        return Err(NyError::UnsupportedConfiguration(format!(
            "forward-linear image bounds: node '{node_name}' has zero stride/dilation"
        )));
    }
    let eff_kh = dh * (kh - 1) + 1;
    let eff_kw = dw * (kw - 1) + 1;
    let padded_h = in_h + 2 * ph;
    let padded_w = in_w + 2 * pw;
    if padded_h < eff_kh || padded_w < eff_kw {
        return Err(NyError::UnsupportedConfiguration(format!(
            "forward-linear image bounds: node '{node_name}' effective kernel exceeds padded input"
        )));
    }
    let out_h = (padded_h - eff_kh) / sh + 1;
    let out_w = (padded_w - eff_kw) / sw + 1;
    if out_c * out_h * out_w != output_dim {
        return Err(NyError::ShapeMismatch {
            expected: vec![out_c * out_h * out_w],
            got: vec![output_dim],
        });
    }
    Ok(ConvGeometry {
        in_c,
        in_h,
        in_w,
        out_c,
        out_h,
        out_w,
        kh,
        kw,
        stride: (sh, sw),
        padding: (ph, pw),
        dilation: (dh, dw),
        contraction: in_c * kh * kw,
    })
}

/// Apply the forward convolution to each ROW of `rows` (shape
/// `(n_obj, conv_in_size)`, f64) via im2col + certified f64 GEMM. `kernel_col`
/// is the reshaped kernel `(contraction, out_c)` (from [`kernel_col_f64`]).
/// Output is `(n_obj, conv_out_size)` in `(oc, oh, ow)` C-order per row —
/// the same contraction as `conv2d_forward_batched_gemm`, in f64.
fn conv_apply_rows_f64(
    rows: ArrayView2<'_, f64>,
    kernel_col: &[f64],
    geo: &ConvGeometry,
    engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
    allow_f32: bool,
) -> Result<Array2<f64>> {
    let n_obj = rows.nrows();
    let spatial = geo.spatial();
    let total_rows = n_obj * spatial;
    let k = geo.contraction;
    let (sh, sw) = geo.stride;
    let (ph, pw) = geo.padding;
    let (dh, dw) = geo.dilation;
    let kernel_spatial = geo.kh * geo.kw;
    let input_spatial = geo.in_h * geo.in_w;

    if deadline.is_some_and(|d| Instant::now() >= d) {
        return Err(NyError::DeadlineExceeded(
            "forward-linear image bounds: deadline exceeded before conv im2col".into(),
        ));
    }

    // im2col gather: (n_obj*out_h*out_w, contraction), row-major; one
    // contiguous block per objective, so the gather parallelizes cleanly.
    let mut im2col = vec![0.0f64; total_rows * k];
    im2col
        .par_chunks_mut(spatial * k)
        .enumerate()
        .for_each(|(obj, block)| {
            let row_view = rows.row(obj);
            // Rows of a row-major (possibly row-sliced) matrix are contiguous.
            let row = row_view.to_slice().expect("contiguous objective row");
            for oh in 0..geo.out_h {
                for ow in 0..geo.out_w {
                    let base = (oh * geo.out_w + ow) * k;
                    for col in 0..k {
                        let ic = col / kernel_spatial;
                        let rem = col % kernel_spatial;
                        let ki = rem / geo.kw;
                        let kj = rem % geo.kw;
                        let ih = (oh * sh + ki * dh) as isize - ph as isize;
                        let iw = (ow * sw + kj * dw) as isize - pw as isize;
                        if ih >= 0 && ih < geo.in_h as isize && iw >= 0 && iw < geo.in_w as isize {
                            block[base + col] =
                                row[ic * input_spatial + ih as usize * geo.in_w + iw as usize];
                        }
                    }
                }
            }
        });

    if deadline.is_some_and(|d| Instant::now() >= d) {
        return Err(NyError::DeadlineExceeded(
            "forward-linear image bounds: deadline exceeded before conv GEMM".into(),
        ));
    }

    let gemm = certified_f64_gemm(
        total_rows, k, geo.out_c, &im2col, kernel_col, engine, allow_f32,
    );

    // Scatter to (n_obj, out_c*out_h*out_w) with (oc, oh, ow) C-order per row.
    let mut out = Array2::<f64>::zeros((n_obj, geo.conv_out_size()));
    let conv_out = geo.conv_out_size();
    out.as_slice_mut()
        .expect("freshly allocated row-major")
        .par_chunks_mut(conv_out)
        .enumerate()
        .for_each(|(obj, row_out)| {
            for oh in 0..geo.out_h {
                for ow in 0..geo.out_w {
                    let gemm_row = obj * spatial + oh * geo.out_w + ow;
                    for oc in 0..geo.out_c {
                        row_out[oc * spatial + oh * geo.out_w + ow] =
                            gemm[gemm_row * geo.out_c + oc];
                    }
                }
            }
        });
    Ok(out)
}

/// Reshape the conv kernel `(out_c, in_c, kh, kw)` to a column matrix
/// `(contraction, out_c)` in f64, optionally taking absolute values.
fn kernel_col_f64(layer: &Conv2dLayer, geo: &ConvGeometry, absolute: bool) -> Vec<f64> {
    let kernel_spatial = geo.kh * geo.kw;
    let mut w = vec![0.0f64; geo.contraction * geo.out_c];
    for oc in 0..geo.out_c {
        for col in 0..geo.contraction {
            let ic = col / kernel_spatial;
            let rem = col % kernel_spatial;
            let ki = rem / geo.kw;
            let kj = rem % geo.kw;
            let v = layer.kernel[[oc, ic, ki, kj]] as f64;
            w[col * geo.out_c + oc] = if absolute { v.abs() } else { v };
        }
    }
    w
}

/// Compose the upstream forward-linear bounds through a Conv2d node with
/// certified rounding (see module docs). O(input_dim) forward conv passes via
/// im2col + f64 GEMM, chunked over the network-input columns.
#[allow(clippy::too_many_arguments)]
pub(super) fn compose_conv2d_forward(
    node_name: &str,
    layer: &Conv2dLayer,
    upstream: &LinearBounds,
    pred_shape: &[usize],
    output_dim: usize,
    input_mag: &[f64],
    engine: Option<&dyn GemmEngine>,
    deadline: Option<Instant>,
    // Seam gate for the value GEMMs; `None` reads `NY_FORWARD_LINEAR_F32` (the
    // production default), tests pass `Some(_)` to force the path race-free.
    use_f32_override: Option<bool>,
) -> Result<LinearBounds> {
    let use_f32 = use_f32_override.unwrap_or_else(forward_linear_f32_gemm_enabled);
    let geo = resolve_conv_geometry(
        node_name,
        layer,
        pred_shape,
        upstream.num_outputs(),
        output_dim,
    )?;
    let n = upstream.num_inputs();
    if input_mag.len() != n {
        return Err(NyError::ShapeMismatch {
            expected: vec![n],
            got: vec![input_mag.len()],
        });
    }
    let conv_in = geo.conv_in_size();
    let conv_out = geo.conv_out_size();

    let w_col = kernel_col_f64(layer, &geo, false);
    let wabs_col = kernel_col_f64(layer, &geo, true);

    let mut new_lower_a = Array2::<f32>::zeros((conv_out, n));
    let mut new_upper_a = Array2::<f32>::zeros((conv_out, n));
    let mut penalty_l = vec![0.0f64; conv_out];
    let mut penalty_u = vec![0.0f64; conv_out];

    // Raw row-major slices: the composition loops below are the hot path and
    // ndarray's checked `[[i, j]]` indexing dominates them in dev profiles.
    // Fail closed (caller falls back to IBP) on a non-standard layout.
    let (ul, uu) = upstream_slices(upstream, node_name)?;

    // Mag-weighted column absolute sums for the γ·S penalty, computed in one
    // parallel pass over features (contiguous upstream rows):
    // w_s[k] = Σ_j (|U_c[k,j]| + |U_r[k,j]|) · mag_j.
    let w_s: Vec<f64> = (0..conv_in)
        .into_par_iter()
        .map(|k_feat| {
            let row_l = &ul[k_feat * n..k_feat * n + n];
            let row_u = &uu[k_feat * n..k_feat * n + n];
            let mut acc = 0.0f64;
            for j in 0..n {
                let l = row_l[j] as f64;
                let u = row_u[j] as f64;
                // Bit-identical to `(l + u) * 0.5`: finite f32-cast operands stay on
                // f64::midpoint's non-overflow `(a + b) * 0.5` path.
                let c = f64::midpoint(l, u);
                let r = (u - l) * 0.5;
                acc += (c.abs() + r.abs()) * input_mag[j];
            }
            acc
        })
        .collect();

    // Chunk the network-input columns so the transient f64 im2col stays
    // bounded (~256 MB): rows_per_chunk * spatial * contraction * 8 bytes.
    let spatial = geo.spatial();
    let budget_rows = (256usize << 20) / 8 / geo.contraction.max(1);
    let chunk_cols = (budget_rows / spatial.max(1)).clamp(16, 1024).min(n.max(1));

    let mut rows_c = Array2::<f64>::zeros((chunk_cols, conv_in));
    let mut rows_r = Array2::<f64>::zeros((chunk_cols, conv_in));

    let mut j0 = 0usize;
    while j0 < n {
        if deadline.is_some_and(|d| Instant::now() >= d) {
            return Err(NyError::DeadlineExceeded(format!(
                "forward-linear image bounds: deadline exceeded inside conv node '{node_name}'"
            )));
        }
        let cb = chunk_cols.min(n - j0);
        // Build center/radius rows for this chunk (exact in f64: f32 widening
        // is exact and (l+u), (u−l), /2 are exact f64 ops on f32 inputs).
        // Parallel over chunk rows (each objective row jj is contiguous).
        {
            let rows_c_flat = rows_c.as_slice_mut().expect("row-major rows_c");
            let rows_r_flat = rows_r.as_slice_mut().expect("row-major rows_r");
            rows_c_flat[..cb * conv_in]
                .par_chunks_mut(conv_in)
                .zip(rows_r_flat[..cb * conv_in].par_chunks_mut(conv_in))
                .enumerate()
                .for_each(|(jj, (crow, rrow))| {
                    let j = j0 + jj;
                    for k_feat in 0..conv_in {
                        let l = ul[k_feat * n + j] as f64;
                        let u = uu[k_feat * n + j] as f64;
                        // Bit-identical (f32-cast operands, f64::midpoint fast path).
                        crow[k_feat] = f64::midpoint(l, u);
                        rrow[k_feat] = (u - l) * 0.5;
                    }
                });
        }
        // Value GEMMs (center/radius coefficients): routed to the sound f32 fast
        // path iff the seam is on — their error is charged to the `γ^f32·S`
        // penalty below.
        let g_center = conv_apply_rows_f64(
            rows_c.slice(s![..cb, ..]),
            &w_col,
            &geo,
            engine,
            deadline,
            use_f32,
        )?;
        let g_radius = conv_apply_rows_f64(
            rows_r.slice(s![..cb, ..]),
            &wabs_col,
            &geo,
            engine,
            deadline,
            use_f32,
        )?;

        // Cast + measured-gap accumulation, parallel over output rows p
        // (each thread owns row p of both coefficient matrices and its
        // penalty slots).
        {
            let gc = g_center.as_slice().expect("row-major g_center");
            let gr = g_radius.as_slice().expect("row-major g_radius");
            let la_flat = new_lower_a.as_slice_mut().expect("row-major lower_a");
            let ua_flat = new_upper_a.as_slice_mut().expect("row-major upper_a");
            la_flat
                .par_chunks_mut(n)
                .zip(ua_flat.par_chunks_mut(n))
                .zip(penalty_l.par_iter_mut().zip(penalty_u.par_iter_mut()))
                .enumerate()
                .for_each(|(p, ((lrow, urow), (pl, pu)))| {
                    for jj in 0..cb {
                        let j = j0 + jj;
                        let mag = input_mag[j];
                        let c = gc[jj * conv_out + p];
                        let r = gr[jj * conv_out + p];
                        lrow[j] = cast_coeff_with_gap(c - r, mag, pl);
                        urow[j] = cast_coeff_with_gap(c + r, mag, pu);
                    }
                });
        }
        j0 += cb;
    }

    // γ·S penalty (coefficient accumulation error) + bias terms via
    // single-vector conv passes.
    let up_lb = upstream.lower_b();
    let up_ub = upstream.upper_b();
    let mut small = Array2::<f64>::zeros((4, conv_in));
    for k_feat in 0..conv_in {
        let l = up_lb[k_feat] as f64;
        let u = up_ub[k_feat] as f64;
        // Bit-identical (f32-cast operands, f64::midpoint fast path).
        let c = f64::midpoint(l, u);
        let r = (u - l) * 0.5;
        small[[0, k_feat]] = c; // u_c  (through W)
        small[[1, k_feat]] = r; // u_r  (through |W|)
        small[[2, k_feat]] = c.abs() + r.abs(); // bias S base (through |W|)
        small[[3, k_feat]] = w_s[k_feat]; // coefficient S base (through |W|)
    }
    // S-BASE GEMMs stay f64: `v_abs` produces `s_bias`/`s_coeff` (the certified
    // error base), which must never be under-estimated, and the bias values
    // (`v_center`/`v_abs` row 0) are tiny so f64 costs nothing here.
    let v_center = conv_apply_rows_f64(
        small.slice(s![0..1, ..]),
        &w_col,
        &geo,
        engine,
        deadline,
        false,
    )?;
    let v_abs = conv_apply_rows_f64(
        small.slice(s![1..4, ..]),
        &wabs_col,
        &geo,
        engine,
        deadline,
        false,
    )?;

    // The value GEMMs (`g_center`/`g_radius`) ran in f32 iff the seam is on, so
    // the coefficient accumulation error is `γ^f32` (much larger) plus an FTZ
    // underflow addend discharged across the input columns. `γ^f32 ≥ γ^f64`
    // conservatively bounds the (f64) bias-GEMM error too, so a single factor is
    // sound for both `s_coeff` (f32 value GEMM) and `s_bias` (f64 bias GEMM).
    let gamma = if use_f32 {
        gamma_n_f32(geo.contraction + 4)
    } else {
        gamma_n_f64(geo.contraction + 4)
    };
    let ftz = if use_f32 {
        let mag_sum: f64 = input_mag.iter().sum();
        forward_f32_ftz_bias(geo.contraction, mag_sum)
    } else {
        0.0
    };
    let mut new_lower_b = Array1::<f32>::zeros(conv_out);
    let mut new_upper_b = Array1::<f32>::zeros(conv_out);
    for p in 0..conv_out {
        let oc = p / spatial;
        let conv_bias = layer.bias.as_ref().map_or(0.0f64, |b| b[oc] as f64);
        let vc = v_center[[0, p]];
        let vr = v_abs[[0, p]];
        let s_bias = v_abs[[1, p]];
        let s_coeff = v_abs[[2, p]];
        let gamma_pen = gamma * (s_coeff + s_bias) + ftz;
        new_lower_b[p] = commit_lower_bias(vc - vr + conv_bias, penalty_l[p] + gamma_pen);
        new_upper_b[p] = commit_upper_bias(vc + vr + conv_bias, penalty_u[p] + gamma_pen);
    }

    detect_and_fix_nonfinite_rows(
        &mut new_lower_a,
        &mut new_upper_a,
        &mut new_lower_b,
        &mut new_upper_b,
        n,
        "forward-linear Conv2d",
    );
    LinearBounds::new_or_conservative(new_lower_a, new_lower_b, new_upper_a, new_upper_b)
}

/// Pack both forward-relaxation coefficient matrices as a leading batch of
/// concrete feature maps: `[lower columns..., upper columns...]`.
///
/// A lower-relaxation coefficient is not necessarily <= the corresponding
/// upper-relaxation coefficient, so these MUST be separate concrete packets;
/// treating `(lower_a, upper_a)` as an interval tensor would be unsound.
fn pack_affine_coefficient_sides(
    node_name: &str,
    upstream: &LinearBounds,
    pred_shape: &[usize],
) -> Result<BoundedTensor> {
    let pred_dim = checked_shape_product(pred_shape).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "forward-linear image bounds: node '{node_name}' predecessor shape {pred_shape:?} \
             overflows usize"
        ))
    })?;
    if pred_dim != upstream.num_outputs() {
        return Err(NyError::ShapeMismatch {
            expected: vec![upstream.num_outputs()],
            got: vec![pred_dim],
        });
    }
    let n = upstream.num_inputs();
    let side_batch = 2usize.checked_mul(n).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "forward-linear image bounds: node '{node_name}' coefficient batch overflows usize"
        ))
    })?;
    let total = side_batch.checked_mul(pred_dim).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "forward-linear image bounds: node '{node_name}' coefficient packet overflows usize"
        ))
    })?;
    let mut values = Vec::with_capacity(total);
    for matrix in [upstream.lower_a(), upstream.upper_a()] {
        for input_col in 0..n {
            for feature in 0..pred_dim {
                values.push(matrix[[feature, input_col]]);
            }
        }
    }
    let mut shape = Vec::with_capacity(pred_shape.len() + 1);
    shape.push(side_batch);
    shape.extend_from_slice(pred_shape);
    let values = ArrayD::from_shape_vec(IxDyn(&shape), values).map_err(|e| {
        NyError::InternalError(format!(
            "forward-linear image bounds: node '{node_name}' cannot pack coefficients: {e}"
        ))
    })?;
    BoundedTensor::concrete(values)
}

/// Pack the lower/upper affine biases as a two-element leading batch.
fn pack_affine_bias_sides(
    node_name: &str,
    upstream: &LinearBounds,
    pred_shape: &[usize],
) -> Result<BoundedTensor> {
    let pred_dim = checked_shape_product(pred_shape).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "forward-linear image bounds: node '{node_name}' predecessor shape {pred_shape:?} \
             overflows usize"
        ))
    })?;
    if pred_dim != upstream.num_outputs() {
        return Err(NyError::ShapeMismatch {
            expected: vec![upstream.num_outputs()],
            got: vec![pred_dim],
        });
    }
    let total = 2usize.checked_mul(pred_dim).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "forward-linear image bounds: node '{node_name}' bias packet overflows usize"
        ))
    })?;
    let mut values = Vec::with_capacity(total);
    values.extend(upstream.lower_b().iter().copied());
    values.extend(upstream.upper_b().iter().copied());
    let mut shape = Vec::with_capacity(pred_shape.len() + 1);
    shape.push(2);
    shape.extend_from_slice(pred_shape);
    let values = ArrayD::from_shape_vec(IxDyn(&shape), values).map_err(|e| {
        NyError::InternalError(format!(
            "forward-linear image bounds: node '{node_name}' cannot pack biases: {e}"
        ))
    })?;
    BoundedTensor::concrete(values)
}

#[inline]
fn outward_add_lower(a: f32, b: f32) -> f32 {
    next_down_f32(((a as f64) + (b as f64)) as f32)
}

#[inline]
fn outward_add_upper(a: f32, b: f32) -> f32 {
    next_up_f32(((a as f64) + (b as f64)) as f32)
}

/// Choose a finite stored coefficient inside a certified enclosure and return
/// a certified absolute error bound for the gap to every real value in it.
/// The caller discharges that error through `max(|x_l|, |x_u|)` into the bias.
#[inline]
fn coefficient_from_enclosure(lower: f32, upper: f32) -> (f32, f64) {
    if !lower.is_finite() || !upper.is_finite() || lower > upper {
        return (0.0, f64::INFINITY);
    }
    let midpoint = f64::midpoint(lower as f64, upper as f64);
    let stored = midpoint as f32;
    if !stored.is_finite() {
        return (0.0, f64::INFINITY);
    }
    let err = ((stored as f64) - (lower as f64))
        .abs()
        .max(((upper as f64) - (stored as f64)).abs());
    (stored, next_up_f32(err as f32) as f64)
}

/// Compose through ConvTranspose2d using its existing certified forward
/// interval kernel rather than introducing a second convolution arithmetic
/// implementation.
///
/// For `l(x) <= h <= u(x)`, the affine split is
/// `W+ l + W- u <= W h <= W+ u + W- l`.  Lower/upper coefficient matrices are
/// packed as separate concrete batches and propagated through the sound
/// ConvTranspose kernel.  The resulting enclosure of every exact composed
/// coefficient is stored at its midpoint; its radius is discharged outward
/// into the bias over the original input box.  This preserves correlation with
/// the (small, five-dimensional for cGAN) network input without trusting an
/// f32 scatter accumulation as exact.
#[allow(clippy::too_many_arguments)]
pub(super) fn compose_conv_transpose2d_forward(
    node_name: &str,
    layer: &ConvTranspose2dLayer,
    upstream: &LinearBounds,
    pred_shape: &[usize],
    output_dim: usize,
    input_mag: &[f64],
    engine: Option<&dyn GemmEngine>,
) -> Result<LinearBounds> {
    // The certified ConvTranspose forward kernel accepts unbatched CHW or
    // batched NCHW.  Coefficient columns are carried on the leading batch axis.
    if pred_shape.len() != 3 {
        return Err(NyError::UnsupportedConfiguration(format!(
            "forward-linear ConvTranspose2d '{node_name}' expects squeezed [C,H,W], got {pred_shape:?}"
        )));
    }
    let n = upstream.num_inputs();
    if input_mag.len() != n {
        return Err(NyError::ShapeMismatch {
            expected: vec![n],
            got: vec![input_mag.len()],
        });
    }
    if upstream
        .lower_b()
        .iter()
        .chain(upstream.upper_b().iter())
        .any(|v| !v.is_finite())
    {
        return Ok(LinearBounds::conservative(output_dim, n));
    }

    let coeff_input = pack_affine_coefficient_sides(node_name, upstream, pred_shape)?;
    let bias_input = pack_affine_bias_sides(node_name, upstream, pred_shape)?;

    let mut positive = layer.clone();
    positive.kernel.mapv_inplace(|w| w.max(0.0));
    positive.bias = None;
    let mut negative = layer.clone();
    negative.kernel.mapv_inplace(|w| w.min(0.0));
    negative.bias = None;

    // Existing Higham-widened interval kernels certify the scatter sums.
    let coeff_expected = 2usize
        .checked_mul(n)
        .and_then(|v| v.checked_mul(output_dim))
        .ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "forward-linear ConvTranspose2d '{node_name}' output packet overflows usize"
            ))
        })?;
    let upper_side_offset = n.checked_mul(output_dim).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "forward-linear ConvTranspose2d '{node_name}' coefficient offset overflows usize"
        ))
    })?;
    let coeff_pos = positive.propagate_ibp_sound_with_engine(&coeff_input, engine)?;
    let coeff_neg = negative.propagate_ibp_sound_with_engine(&coeff_input, engine)?;
    if coeff_pos.len() != coeff_expected || coeff_neg.len() != coeff_expected {
        return Err(NyError::ShapeMismatch {
            expected: vec![coeff_expected],
            got: vec![coeff_pos.len().max(coeff_neg.len())],
        });
    }
    let cpl = coeff_pos.lower().as_slice().ok_or_else(|| {
        NyError::UnsupportedConfiguration(format!(
            "forward-linear ConvTranspose2d '{node_name}' produced non-contiguous coefficients"
        ))
    })?;
    let cpu = coeff_pos.upper().as_slice().ok_or_else(|| {
        NyError::UnsupportedConfiguration(format!(
            "forward-linear ConvTranspose2d '{node_name}' produced non-contiguous coefficients"
        ))
    })?;
    let cnl = coeff_neg.lower().as_slice().ok_or_else(|| {
        NyError::UnsupportedConfiguration(format!(
            "forward-linear ConvTranspose2d '{node_name}' produced non-contiguous coefficients"
        ))
    })?;
    let cnu = coeff_neg.upper().as_slice().ok_or_else(|| {
        NyError::UnsupportedConfiguration(format!(
            "forward-linear ConvTranspose2d '{node_name}' produced non-contiguous coefficients"
        ))
    })?;

    let mut new_lower_a = Array2::<f32>::zeros((output_dim, n));
    let mut new_upper_a = Array2::<f32>::zeros((output_dim, n));
    let mut penalty_l = vec![0.0f64; output_dim];
    let mut penalty_u = vec![0.0f64; output_dim];
    for input_col in 0..n {
        let lower_base = input_col.checked_mul(output_dim).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "forward-linear ConvTranspose2d '{node_name}' lower offset overflows usize"
            ))
        })?;
        let upper_base = upper_side_offset.checked_add(lower_base).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "forward-linear ConvTranspose2d '{node_name}' upper offset overflows usize"
            ))
        })?;
        for p in 0..output_dim {
            // Exact lower coefficient: W+*lower_A + W-*upper_A.
            let lower_lo = outward_add_lower(cpl[lower_base + p], cnl[upper_base + p]);
            let lower_hi = outward_add_upper(cpu[lower_base + p], cnu[upper_base + p]);
            let (stored_l, err_l) = coefficient_from_enclosure(lower_lo, lower_hi);
            new_lower_a[[p, input_col]] = stored_l;
            penalty_l[p] += safe_mul_for_bounds_f64(err_l, input_mag[input_col]);

            // Exact upper coefficient: W+*upper_A + W-*lower_A.
            let upper_lo = outward_add_lower(cpl[upper_base + p], cnl[lower_base + p]);
            let upper_hi = outward_add_upper(cpu[upper_base + p], cnu[lower_base + p]);
            let (stored_u, err_u) = coefficient_from_enclosure(upper_lo, upper_hi);
            new_upper_a[[p, input_col]] = stored_u;
            penalty_u[p] += safe_mul_for_bounds_f64(err_u, input_mag[input_col]);
        }
    }

    // Bias packets use the same affine sign split.  Include the layer bias in
    // the positive packet exactly once; the negative packet has no bias.
    let mut positive_with_bias = positive;
    positive_with_bias.bias = layer.bias.clone();
    let bias_expected = 2usize.checked_mul(output_dim).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "forward-linear ConvTranspose2d '{node_name}' bias output overflows usize"
        ))
    })?;
    let bias_pos = positive_with_bias.propagate_ibp_sound_with_engine(&bias_input, engine)?;
    let bias_neg = negative.propagate_ibp_sound_with_engine(&bias_input, engine)?;
    if bias_pos.len() != bias_expected || bias_neg.len() != bias_expected {
        return Err(NyError::ShapeMismatch {
            expected: vec![bias_expected],
            got: vec![bias_pos.len().max(bias_neg.len())],
        });
    }
    let bpl = bias_pos.lower().as_slice().ok_or_else(|| {
        NyError::UnsupportedConfiguration(format!(
            "forward-linear ConvTranspose2d '{node_name}' produced non-contiguous biases"
        ))
    })?;
    let bpu = bias_pos.upper().as_slice().ok_or_else(|| {
        NyError::UnsupportedConfiguration(format!(
            "forward-linear ConvTranspose2d '{node_name}' produced non-contiguous biases"
        ))
    })?;
    let bnl = bias_neg.lower().as_slice().ok_or_else(|| {
        NyError::UnsupportedConfiguration(format!(
            "forward-linear ConvTranspose2d '{node_name}' produced non-contiguous biases"
        ))
    })?;
    let bnu = bias_neg.upper().as_slice().ok_or_else(|| {
        NyError::UnsupportedConfiguration(format!(
            "forward-linear ConvTranspose2d '{node_name}' produced non-contiguous biases"
        ))
    })?;
    let mut new_lower_b = Array1::<f32>::zeros(output_dim);
    let mut new_upper_b = Array1::<f32>::zeros(output_dim);
    for p in 0..output_dim {
        let lower_bias = outward_add_lower(bpl[p], bnl[output_dim + p]);
        let upper_bias = outward_add_upper(bpu[output_dim + p], bnu[p]);
        new_lower_b[p] = commit_lower_bias(lower_bias as f64, penalty_l[p]);
        new_upper_b[p] = commit_upper_bias(upper_bias as f64, penalty_u[p]);
    }

    detect_and_fix_nonfinite_rows(
        &mut new_lower_a,
        &mut new_upper_a,
        &mut new_lower_b,
        &mut new_upper_b,
        n,
        &format!("forward-linear ConvTranspose2d '{node_name}'"),
    );
    LinearBounds::new_or_conservative(new_lower_a, new_lower_b, new_upper_a, new_upper_b)
}

/// Compose a shape-aware inference BatchNorm through the forward affine map.
/// The nominal per-channel scale is an exact diagonal affine composition; its
/// f64->f32 coefficient cast gap is discharged through `input_mag`.  The
/// existing BatchNorm precompute-error bounds are expanded with the SAME
/// channel-layout decoder used by IBP/CROWN, then folded as
/// `scale_err*max(|pre_l|,|pre_u|) + bias_err` into each output bias.
pub(super) fn compose_batch_norm_forward(
    node_name: &str,
    layer: &BatchNormLayer,
    upstream: &LinearBounds,
    pre_activation: &BoundedTensor,
    output_dim: usize,
    input_mag: &[f64],
) -> Result<LinearBounds> {
    let n = upstream.num_inputs();
    if upstream.num_outputs() != output_dim || pre_activation.len() != output_dim {
        return Err(NyError::ShapeMismatch {
            expected: vec![upstream.num_outputs()],
            got: vec![output_dim.max(pre_activation.len())],
        });
    }
    if input_mag.len() != n {
        return Err(NyError::ShapeMismatch {
            expected: vec![n],
            got: vec![input_mag.len()],
        });
    }
    let (scale, bias, scale_err, bias_err) =
        layer.expanded_affine_parameters(pre_activation.shape(), output_dim)?;
    if scale_err
        .iter()
        .chain(bias_err.iter())
        .any(|&err| err.is_nan() || err < 0.0)
    {
        return Err(NyError::NumericalInstability(format!(
            "forward-linear BatchNorm '{node_name}' has NaN or negative certified error metadata"
        )));
    }
    let pre_l: Vec<f32> = pre_activation.lower().iter().copied().collect();
    let pre_u: Vec<f32> = pre_activation.upper().iter().copied().collect();

    let mut new_lower_a = Array2::<f32>::zeros((output_dim, n));
    let mut new_upper_a = Array2::<f32>::zeros((output_dim, n));
    let mut new_lower_b = Array1::<f32>::zeros(output_dim);
    let mut new_upper_b = Array1::<f32>::zeros(output_dim);
    for p in 0..output_dim {
        let s = scale[p] as f64;
        let b = bias[p] as f64;
        let (src_lower_a, src_lower_b, src_upper_a, src_upper_b) = if s >= 0.0 {
            (
                upstream.lower_a(),
                upstream.lower_b()[p],
                upstream.upper_a(),
                upstream.upper_b()[p],
            )
        } else {
            (
                upstream.upper_a(),
                upstream.upper_b()[p],
                upstream.lower_a(),
                upstream.lower_b()[p],
            )
        };

        let mut penalty_l = 0.0f64;
        let mut penalty_u = 0.0f64;
        for j in 0..n {
            let lower_exact = safe_mul_for_bounds_f64(s, src_lower_a[[p, j]] as f64);
            let upper_exact = safe_mul_for_bounds_f64(s, src_upper_a[[p, j]] as f64);
            new_lower_a[[p, j]] = cast_coeff_with_gap(lower_exact, input_mag[j], &mut penalty_l);
            new_upper_a[[p, j]] = cast_coeff_with_gap(upper_exact, input_mag[j], &mut penalty_u);
        }

        // The true (unrounded-precompute) BN affine differs from the stored
        // scale/bias by this constant over the certified pre-activation box.
        let xmag = (pre_l[p] as f64).abs().max((pre_u[p] as f64).abs());
        let parameter_margin =
            safe_mul_for_bounds_f64(xmag, scale_err[p] as f64) + bias_err[p] as f64;
        penalty_l += parameter_margin;
        penalty_u += parameter_margin;

        let lower_base = safe_mul_for_bounds_f64(s, src_lower_b as f64) + b;
        let upper_base = safe_mul_for_bounds_f64(s, src_upper_b as f64) + b;
        new_lower_b[p] = commit_lower_bias(lower_base, penalty_l);
        new_upper_b[p] = commit_upper_bias(upper_base, penalty_u);
    }

    detect_and_fix_nonfinite_rows(
        &mut new_lower_a,
        &mut new_upper_a,
        &mut new_lower_b,
        &mut new_upper_b,
        n,
        &format!("forward-linear BatchNorm '{node_name}'"),
    );
    LinearBounds::new_or_conservative(new_lower_a, new_lower_b, new_upper_a, new_upper_b)
}

/// Compose the upstream forward-linear bounds through a dense affine layer
/// `y = W h + b` (Linear/Gemm) with certified rounding (see module docs).
#[allow(clippy::too_many_arguments)]
pub(super) fn compose_dense_affine_forward(
    node_name: &str,
    weight: &Array2<f32>,
    bias: Option<&Array1<f32>>,
    upstream: &LinearBounds,
    input_mag: &[f64],
    engine: Option<&dyn GemmEngine>,
    // Seam gate for the value GEMMs; `None` reads `NY_FORWARD_LINEAR_F32`.
    use_f32_override: Option<bool>,
) -> Result<LinearBounds> {
    let use_f32 = use_f32_override.unwrap_or_else(forward_linear_f32_gemm_enabled);
    let m = weight.nrows();
    let k = weight.ncols();
    let n = upstream.num_inputs();
    if upstream.num_outputs() != k {
        return Err(NyError::ShapeMismatch {
            expected: vec![k],
            got: vec![upstream.num_outputs()],
        });
    }
    if input_mag.len() != n {
        return Err(NyError::ShapeMismatch {
            expected: vec![n],
            got: vec![input_mag.len()],
        });
    }

    let up_l = upstream.lower_a();
    let up_u = upstream.upper_a();
    let up_lb = upstream.lower_b();
    let up_ub = upstream.upper_b();

    // Center/radius of the upstream coefficients (exact in f64) + the
    // mag-weighted column S base for the γ penalty.
    let mut uc = vec![0.0f64; k * n];
    let mut ur = vec![0.0f64; k * n];
    let mut w_s = vec![0.0f64; k];
    for kk in 0..k {
        for j in 0..n {
            let l = up_l[[kk, j]] as f64;
            let u = up_u[[kk, j]] as f64;
            // Bit-identical (f32-cast operands, f64::midpoint fast path).
            let c = f64::midpoint(l, u);
            let r = (u - l) * 0.5;
            uc[kk * n + j] = c;
            ur[kk * n + j] = r;
            w_s[kk] += (c.abs() + r.abs()) * input_mag[j];
        }
    }
    let w64: Vec<f64> = weight.iter().map(|&v| v as f64).collect();
    let wabs64: Vec<f64> = weight.iter().map(|&v| (v as f64).abs()).collect();

    // Value GEMMs (center/radius): routed to the sound f32 fast path iff on.
    let g_center = certified_f64_gemm(m, k, n, &w64, &uc, engine, use_f32);
    let g_radius = certified_f64_gemm(m, k, n, &wabs64, &ur, engine, use_f32);

    let mut new_lower_a = Array2::<f32>::zeros((m, n));
    let mut new_upper_a = Array2::<f32>::zeros((m, n));
    let mut penalty_l = vec![0.0f64; m];
    let mut penalty_u = vec![0.0f64; m];
    for i in 0..m {
        for j in 0..n {
            let c = g_center[i * n + j];
            let r = g_radius[i * n + j];
            new_lower_a[[i, j]] = cast_coeff_with_gap(c - r, input_mag[j], &mut penalty_l[i]);
            new_upper_a[[i, j]] = cast_coeff_with_gap(c + r, input_mag[j], &mut penalty_u[i]);
        }
    }

    // Value GEMMs (`g_center`/`g_radius`) ran in f32 iff the seam is on; the
    // bias terms below are exact f64 CPU sums, so only the coefficient error
    // grows to `γ^f32`. `γ^f32 ≥ γ^f64` conservatively covers the bias `s_bias`.
    let gamma = if use_f32 {
        gamma_n_f32(k + 4)
    } else {
        gamma_n_f64(k + 4)
    };
    let ftz = if use_f32 {
        let mag_sum: f64 = input_mag.iter().sum();
        forward_f32_ftz_bias(k, mag_sum)
    } else {
        0.0
    };
    let mut new_lower_b = Array1::<f32>::zeros(m);
    let mut new_upper_b = Array1::<f32>::zeros(m);
    for i in 0..m {
        let mut vc = 0.0f64; // W  @ u_c
        let mut vr = 0.0f64; // |W| @ u_r
        let mut s_bias = 0.0f64; // |W| @ (|u_c|+|u_r|)
        let mut s_coeff = 0.0f64; // |W| @ w_s
        for kk in 0..k {
            let w = weight[[i, kk]] as f64;
            let wa = w.abs();
            let l = up_lb[kk] as f64;
            let u = up_ub[kk] as f64;
            // Bit-identical (f32-cast operands, f64::midpoint fast path).
            let c = f64::midpoint(l, u);
            let r = (u - l) * 0.5;
            vc += w * c;
            vr += wa * r;
            s_bias += wa * (c.abs() + r.abs());
            s_coeff += wa * w_s[kk];
        }
        let b = bias.map_or(0.0f64, |b| b[i] as f64);
        let gamma_pen = gamma * (s_coeff + s_bias) + ftz;
        new_lower_b[i] = commit_lower_bias(vc - vr + b, penalty_l[i] + gamma_pen);
        new_upper_b[i] = commit_upper_bias(vc + vr + b, penalty_u[i] + gamma_pen);
    }

    detect_and_fix_nonfinite_rows(
        &mut new_lower_a,
        &mut new_upper_a,
        &mut new_lower_b,
        &mut new_upper_b,
        n,
        &format!("forward-linear Linear '{node_name}'"),
    );
    LinearBounds::new_or_conservative(new_lower_a, new_lower_b, new_upper_a, new_upper_b)
}

/// Compose the upstream forward-linear bounds through a ReLU node using the
/// per-neuron diagonal relaxation (no dense identity materialization — the
/// generic identity-trick path is O(N²) memory and infeasible at image scale).
///
/// Uses the production `relu_linear_relaxation` (matched to the reference;
/// sound chord upper with bumped intercept). Slope·coefficient products are
/// exact in f64 (f32×f32 fits in 53 bits); only the measured f32 cast gap is
/// discharged. Rows with non-finite relaxation intercepts (NaN/±inf
/// pre-activation) degrade to `A=0, b=±inf`.
///
/// # Optimized lower slopes (#w4-root-alpha)
///
/// `alpha_lower`, when present, supplies per-neuron lower slopes (e.g. the
/// alpha-CROWN warmup's optimized values). For a CROSSING neuron
/// (finite `l < 0 < u`) the lower relaxation `y >= α·x + 0` is sound for ANY
/// `α ∈ [0, 1]` with intercept exactly 0, independent of `l`/`u`:
/// on `x ∈ [l, 0]`, `ReLU(x) = 0 >= α·x` (α >= 0, x <= 0); on `x ∈ [0, u]`,
/// `ReLU(x) = x >= α·x` (α <= 1, x >= 0). The adaptive rule is the
/// `α ∈ {0, 1}` special case, so feeding the adaptive value reproduces the
/// legacy composition bit-for-bit. Stable neurons keep their exact
/// identity/zero relaxation (α ignored); NaN α falls back to the adaptive
/// rule; values outside [0, 1] are clamped. The UPPER relaxation is always
/// the sound chord — never touched by α.
pub(super) fn compose_relu_diag_forward(
    node_name: &str,
    upstream: &LinearBounds,
    pre_activation: &BoundedTensor,
    input_mag: &[f64],
    alpha_lower: Option<&[f32]>,
) -> Result<LinearBounds> {
    let m = upstream.num_outputs();
    let n = upstream.num_inputs();
    let pre_flat = pre_activation.flatten();
    if pre_flat.len() != m {
        return Err(NyError::ShapeMismatch {
            expected: vec![m],
            got: vec![pre_flat.len()],
        });
    }
    if input_mag.len() != n {
        return Err(NyError::ShapeMismatch {
            expected: vec![n],
            got: vec![input_mag.len()],
        });
    }
    let pre_l: Vec<f32> = pre_flat.lower().iter().copied().collect();
    let pre_u: Vec<f32> = pre_flat.upper().iter().copied().collect();

    let up_lb = upstream.lower_b();
    let up_ub = upstream.upper_b();
    let (ul, uu) = upstream_slices(upstream, node_name)?;

    let mut new_lower_a = Array2::<f32>::zeros((m, n));
    let mut new_upper_a = Array2::<f32>::zeros((m, n));
    let mut new_lower_b = Array1::<f32>::zeros(m);
    let mut new_upper_b = Array1::<f32>::zeros(m);

    {
        let la_flat = new_lower_a.as_slice_mut().expect("row-major lower_a");
        let ua_flat = new_upper_a.as_slice_mut().expect("row-major upper_a");
        let lb_flat = new_lower_b.as_slice_mut().expect("contiguous lower_b");
        let ub_flat = new_upper_b.as_slice_mut().expect("contiguous upper_b");
        la_flat
            .par_chunks_mut(n)
            .zip(ua_flat.par_chunks_mut(n))
            .zip(lb_flat.par_iter_mut().zip(ub_flat.par_iter_mut()))
            .enumerate()
            .for_each(|(i, ((lrow, urow), (lb, ub)))| {
                let relax = relu_linear_relaxation(pre_l[i], pre_u[i]);

                // Lower side: y_i >= d·h_i + c with d = lower_slope >= 0 for
                // ReLU; written sign-generally (d < 0 selects the upstream
                // UPPER side) so the helper stays sound for future monotone
                // activations.
                //
                // #w4-root-alpha: a caller-supplied α replaces the adaptive
                // lower slope ONLY for finite crossing neurons, where
                // `y >= α·x` is sound with intercept 0 for any α ∈ [0, 1]
                // (see fn docs). All other cases keep the proven adaptive
                // relaxation (including its NaN/±inf fallbacks).
                let optimized = alpha_lower.and_then(|alpha| {
                    let a = alpha[i];
                    (pre_l[i] < 0.0
                        && pre_u[i] > 0.0
                        && pre_l[i].is_finite()
                        && pre_u[i].is_finite()
                        && a.is_finite())
                    .then(|| f64::from(a.clamp(0.0, 1.0)))
                });
                let (d, c) = match optimized {
                    Some(a) => (a, 0.0f64),
                    None => (relax.lower_slope as f64, relax.lower_intercept as f64),
                };
                if !d.is_finite() || !c.is_finite() {
                    *lb = f32::NEG_INFINITY;
                } else if d == 0.0 {
                    *lb = commit_lower_bias(c, 0.0);
                } else {
                    let (src_a, src_b) = if d >= 0.0 {
                        (&ul[i * n..i * n + n], up_lb[i] as f64)
                    } else {
                        (&uu[i * n..i * n + n], up_ub[i] as f64)
                    };
                    let mut pen = 0.0f64;
                    for j in 0..n {
                        // f32×f32 in f64 is exact; only the cast gap is discharged.
                        lrow[j] =
                            cast_coeff_with_gap(d * (src_a[j] as f64), input_mag[j], &mut pen);
                    }
                    *lb = commit_lower_bias(d * src_b + c, pen);
                }

                // Upper side: y_i <= d·h_i + c.
                let d = relax.upper_slope as f64;
                let c = relax.upper_intercept as f64;
                if !d.is_finite() || !c.is_finite() {
                    for v in urow.iter_mut() {
                        *v = 0.0;
                    }
                    *ub = f32::INFINITY;
                } else if d == 0.0 {
                    *ub = commit_upper_bias(c, 0.0);
                } else {
                    let (src_a, src_b) = if d >= 0.0 {
                        (&uu[i * n..i * n + n], up_ub[i] as f64)
                    } else {
                        (&ul[i * n..i * n + n], up_lb[i] as f64)
                    };
                    let mut pen = 0.0f64;
                    for j in 0..n {
                        urow[j] =
                            cast_coeff_with_gap(d * (src_a[j] as f64), input_mag[j], &mut pen);
                    }
                    *ub = commit_upper_bias(d * src_b + c, pen);
                }
            });
    }

    detect_and_fix_nonfinite_rows(
        &mut new_lower_a,
        &mut new_upper_a,
        &mut new_lower_b,
        &mut new_upper_b,
        n,
        &format!("forward-linear ReLU '{node_name}'"),
    );
    LinearBounds::new_or_conservative(new_lower_a, new_lower_b, new_upper_a, new_upper_b)
}

/// Compose a binary residual Add: `y = h_a + h_b`. The Jacobian is the
/// identity toward both parents (both weights +1 ≥ 0), so lower composes with
/// lowers and upper with uppers. Coefficient sums are exact in f64; the
/// measured f32 cast gap is discharged into the bias.
pub(super) fn compose_add_forward(
    node_name: &str,
    a: &LinearBounds,
    b: &LinearBounds,
    input_mag: &[f64],
) -> Result<LinearBounds> {
    let m = a.num_outputs();
    let n = a.num_inputs();
    if b.num_outputs() != m || b.num_inputs() != n {
        return Err(NyError::ShapeMismatch {
            expected: vec![m, n],
            got: vec![b.num_outputs(), b.num_inputs()],
        });
    }
    if input_mag.len() != n {
        return Err(NyError::ShapeMismatch {
            expected: vec![n],
            got: vec![input_mag.len()],
        });
    }
    let (alb, aub) = (a.lower_b(), a.upper_b());
    let (blb, bub) = (b.lower_b(), b.upper_b());
    let (als, aus) = upstream_slices(a, node_name)?;
    let (bls, bus) = upstream_slices(b, node_name)?;

    let mut new_lower_a = Array2::<f32>::zeros((m, n));
    let mut new_upper_a = Array2::<f32>::zeros((m, n));
    let mut new_lower_b = Array1::<f32>::zeros(m);
    let mut new_upper_b = Array1::<f32>::zeros(m);
    {
        let la_flat = new_lower_a.as_slice_mut().expect("row-major lower_a");
        let ua_flat = new_upper_a.as_slice_mut().expect("row-major upper_a");
        let lb_flat = new_lower_b.as_slice_mut().expect("contiguous lower_b");
        let ub_flat = new_upper_b.as_slice_mut().expect("contiguous upper_b");
        la_flat
            .par_chunks_mut(n)
            .zip(ua_flat.par_chunks_mut(n))
            .zip(lb_flat.par_iter_mut().zip(ub_flat.par_iter_mut()))
            .enumerate()
            .for_each(|(i, ((lrow, urow), (lb, ub)))| {
                let arow_l = &als[i * n..i * n + n];
                let arow_u = &aus[i * n..i * n + n];
                let brow_l = &bls[i * n..i * n + n];
                let brow_u = &bus[i * n..i * n + n];
                let mut pen_l = 0.0f64;
                let mut pen_u = 0.0f64;
                for j in 0..n {
                    // f32+f32 in f64 is exact; only the f32 cast gap is discharged.
                    let lo = arow_l[j] as f64 + brow_l[j] as f64;
                    let hi = arow_u[j] as f64 + brow_u[j] as f64;
                    lrow[j] = cast_coeff_with_gap(lo, input_mag[j], &mut pen_l);
                    urow[j] = cast_coeff_with_gap(hi, input_mag[j], &mut pen_u);
                }
                *lb = commit_lower_bias(alb[i] as f64 + blb[i] as f64, pen_l);
                *ub = commit_upper_bias(aub[i] as f64 + bub[i] as f64, pen_u);
            });
    }

    detect_and_fix_nonfinite_rows(
        &mut new_lower_a,
        &mut new_upper_a,
        &mut new_lower_b,
        &mut new_upper_b,
        n,
        &format!("forward-linear Add '{node_name}'"),
    );
    LinearBounds::new_or_conservative(new_lower_a, new_lower_b, new_upper_a, new_upper_b)
}
