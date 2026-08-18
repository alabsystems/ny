// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{Array1, ArrayD, Axis, IxDyn};
use ny_core::{
    checked_shape_product,
    dd::{next_down_f64, next_up_f64},
    NyError, Result,
};
use rayon::prelude::*;
use std::time::Instant;

use super::ops::{
    conv2d_transpose_grouped_batched_fast, conv2d_transpose_grouped_into,
    conv2d_transpose_grouped_into_with_deadline,
};
use super::{conv2d_transpose_grouped, Conv2dLayer};
use crate::bounds::patches::{CrownBounds, PatchGeometry, PatchesData, PatchesLinearBounds};
use crate::layers::common::PatchesPropagation;

const F64_FRACTION_BITS: u32 = 52;
const F64_EXPONENT_BIAS: i32 = 1023;

/// Decode a binary32 bit pattern without presenting a subnormal source operand
/// to a hardware conversion instruction. This keeps the certificate independent
/// of the host's DAZ mode.
#[inline]
fn f32_to_f64_exact_bits(value: f32) -> f64 {
    let bits = value.to_bits();
    let sign = u64::from(bits >> 31) << 63;
    let exponent = (bits >> 23) & 0xff;
    let fraction = bits & 0x7f_ffff;
    match (exponent, fraction) {
        (0, 0) => f64::from_bits(sign),
        (0, _) => {
            let leading = fraction.ilog2();
            let unbiased_exponent = leading as i32 - 149;
            let exponent64 = (unbiased_exponent + F64_EXPONENT_BIAS) as u64;
            let leading_bit = 1_u32 << leading;
            let fraction64 = u64::from(fraction - leading_bit) << (F64_FRACTION_BITS - leading);
            f64::from_bits(sign | (exponent64 << F64_FRACTION_BITS) | fraction64)
        }
        (0xff, 0) => f64::from_bits(sign | (0x7ff_u64 << F64_FRACTION_BITS)),
        (0xff, _) => f64::NAN,
        _ => {
            let unbiased_exponent = exponent as i32 - 127;
            let exponent64 = (unbiased_exponent + F64_EXPONENT_BIAS) as u64;
            let fraction64 = u64::from(fraction) << (F64_FRACTION_BITS - 23);
            f64::from_bits(sign | (exponent64 << F64_FRACTION_BITS) | fraction64)
        }
    }
}

/// Decode a coefficient-error carrier without DAZ-sensitive comparisons.
#[inline]
fn nonnegative_f32_error_or_infinity(value: f32) -> f64 {
    let bits = value.to_bits();
    let magnitude = bits & 0x7fff_ffff;
    let exponent = magnitude >> 23;
    if exponent == 0xff || (bits >> 31 != 0 && magnitude != 0) {
        f64::INFINITY
    } else {
        f32_to_f64_exact_bits(value)
    }
}

/// Preserve the historical one-ULP outward publication while never emitting a
/// binary32-subnormal endpoint that FTZ could move inward.
#[inline]
fn publish_lower_bound_no_subnormal(value: f64) -> f32 {
    if value.is_nan() || value == f64::NEG_INFINITY {
        return f32::NEG_INFINITY;
    }
    if value == f64::INFINITY {
        return f32::INFINITY;
    }
    let min_normal = f32_to_f64_exact_bits(f32::MIN_POSITIVE);
    if value != 0.0 && value.abs() < min_normal {
        return if value.is_sign_negative() {
            -f32::MIN_POSITIVE
        } else {
            0.0
        };
    }
    let stepped = ny_tensor::next_down_f32(value as f32);
    let magnitude = stepped.to_bits() & 0x7fff_ffff;
    if magnitude != 0 && magnitude < f32::MIN_POSITIVE.to_bits() {
        if value.is_sign_negative() {
            -f32::MIN_POSITIVE
        } else {
            0.0
        }
    } else {
        stepped
    }
}

#[inline]
fn publish_upper_bound_no_subnormal(value: f64) -> f32 {
    if value.is_nan() || value == f64::INFINITY {
        return f32::INFINITY;
    }
    if value == f64::NEG_INFINITY {
        return f32::NEG_INFINITY;
    }
    let min_normal = f32_to_f64_exact_bits(f32::MIN_POSITIVE);
    if value != 0.0 && value.abs() < min_normal {
        return if value.is_sign_negative() {
            0.0
        } else {
            f32::MIN_POSITIVE
        };
    }
    let stepped = ny_tensor::next_up_f32(value as f32);
    let magnitude = stepped.to_bits() & 0x7fff_ffff;
    if magnitude != 0 && magnitude < f32::MIN_POSITIVE.to_bits() {
        if value.is_sign_negative() {
            0.0
        } else {
            f32::MIN_POSITIVE
        }
    } else {
        stepped
    }
}

/// Multiply two non-negative certificate terms with an outward binary64 step.
#[inline]
fn mul_f64_up(lhs: f64, rhs: f64) -> f64 {
    if lhs == 0.0 || rhs == 0.0 {
        return 0.0;
    }
    let product = lhs * rhs;
    if product.is_nan() {
        f64::INFINITY
    } else {
        next_up_f64(product)
    }
}

/// Publish a non-negative certificate without a binary32-subnormal result.
///
/// FTZ can erase a subnormal error carrier at its next consumer. A real zero
/// remains zero; every positive subnormal-range term is raised above the normal
/// floor before the final strict outward step.
#[inline]
fn publish_error_up_normal(value: f64) -> f32 {
    if value.is_nan() || value < 0.0 || value == f64::INFINITY {
        return f32::INFINITY;
    }
    if value == 0.0 {
        return 0.0;
    }
    let min_normal = f32_to_f64_exact_bits(f32::MIN_POSITIVE);
    if value <= min_normal {
        return ny_tensor::next_up_f32(f32::MIN_POSITIVE);
    }
    let nearest = value as f32;
    if nearest == f32::INFINITY {
        return f32::INFINITY;
    }
    let upward = if f32_to_f64_exact_bits(nearest) >= value {
        nearest
    } else {
        ny_tensor::next_up_f32(nearest)
    };
    ny_tensor::next_up_f32(upward)
}

/// Absolute result/source-flush charge for one f32 patches contraction.
///
/// `4K·FLT_MIN` covers FTZ of products/partial sums. The DAZ term mirrors the
/// dense/Linear certificate: a flushed input coefficient can be amplified by
/// the kernel, and a flushed kernel coefficient by the input row.
#[inline]
fn patches_f32_underflow_charge(row_l1: f64, kernel_l1: f64, k: usize) -> f64 {
    if (k as u128) > (1_u128 << f64::MANTISSA_DIGITS) {
        return f64::INFINITY;
    }
    let min_normal = f32_to_f64_exact_bits(f32::MIN_POSITIVE);
    let ftz = mul_f64_up(4.0 * k as f64, min_normal);
    let daz = mul_f64_up(add_f64_up(row_l1, kernel_l1), min_normal);
    add_f64_up(ftz, daz)
}

/// Directed binary64 addition for a lower-bound reduction.
#[inline]
fn add_f64_down(acc: f64, term: f64) -> f64 {
    if acc.is_nan() || term.is_nan() {
        return f64::NEG_INFINITY;
    }
    if term == 0.0 {
        return acc;
    }
    let sum = acc + term;
    if sum.is_nan() {
        f64::NEG_INFINITY
    } else {
        next_down_f64(sum)
    }
}

/// Directed binary64 addition for an upper-bound reduction.
#[inline]
fn add_f64_up(acc: f64, term: f64) -> f64 {
    if acc.is_nan() || term.is_nan() {
        return f64::INFINITY;
    }
    if term == 0.0 {
        return acc;
    }
    let sum = acc + term;
    if sum.is_nan() {
        f64::INFINITY
    } else {
        next_up_f64(sum)
    }
}

/// Route the patches-mode Conv2d transpose-conv composition through a single
/// engine GEMM (GPU-capable) instead of the per-position CPU scatter — the
/// keystone for getting the conv-CROWN warmup onto the GPU.
///
/// DEFAULT: on whenever a fast-f32 GEMM accelerator (cuBLAS/CUDA) is installed,
/// off otherwise. The GPU compose carries the same certified outward-rounded
/// coefficient error as the CPU scatter (verified by the patches +
/// `crown_linear_aw` soundness oracles under this gate), so its bounds are
/// sound enclosures — enabling it can only shift a borderline verdict between
/// `verified` and `unknown`, never toward an unsound `unsat`. It is NOT
/// byte-identical to the CPU scatter (GPU GEMM accumulation order differs), so
/// CPU-only builds keep the byte-identical scatter as their default.
/// `NY_PATCHES_GPU=1` forces it on (e.g. a CPU build with a wgpu engine);
/// `NY_PATCHES_GPU=0` forces the byte-identical CPU scatter even on a CUDA box.
fn patches_gpu_enabled() -> bool {
    match std::env::var("NY_PATCHES_GPU") {
        Ok(v) => v != "0" && !v.is_empty(),
        Err(_) => crate::fast_f32_gemm::is_installed(),
    }
}

/// Default-dark finite-deadline explicit-row bias reduction.
///
/// The historical fold indexes every coefficient through a dynamic 7D ndarray
/// coordinate. Fresh 7D Patches tensors are row-major contiguous, so the exact
/// same per-row reduction order can instead traverse checked flat row slabs.
/// Rows are independent and may be reduced in parallel; all results remain in
/// scratch until both sides and every row complete.
fn patches_deadline_flat_bias_enabled() -> bool {
    matches!(
        std::env::var("NY_PATCHES_DEADLINE_FLAT_BIAS").as_deref(),
        Ok("1")
    )
}

/// Default-dark finite-deadline nonsparse 7D position-scatter parallelism.
///
/// Every position owns one disjoint contiguous output slab and retains the
/// historical scalar contraction order within that slab. The gate therefore
/// changes scheduling only. A finite call keeps one common absolute deadline;
/// its output tensor remains caller-private scratch until every worker and the
/// subsequent certified coefficient-error propagation complete.
fn patches_deadline_parallel_scatter_enabled() -> bool {
    matches!(
        std::env::var("NY_PATCHES_DEADLINE_PARALLEL_SCATTER").as_deref(),
        Ok("1")
    )
}

/// #patches-deadline-kernel: a deadline no longer forces the SERIAL scatter.
///
/// The previous rule admitted parallel position scatter only when no deadline
/// was present, or when the layout was 7D explicit-rows AND the dark
/// `NY_PATCHES_DEADLINE_PARALLEL_SCATTER` gate was set. Since the CROWN-IBP
/// collector always passes a per-node deadline, and the common conv layout is 6D
/// (not explicit-rows), the practical effect was that every collector-routed
/// conv target scattered single-threaded — on top of losing the GEMM engine and
/// the batched seam to the same "deadline is present" test.
///
/// Parallel scatter is verdict-neutral by construction: workers fill DISJOINT
/// position slabs of caller-private scratch, so the result is identical to the
/// serial fill regardless of completion order (the surrounding doc comment
/// states this invariant, and it is why the no-deadline path was already
/// allowed to parallelize). A deadline governs how much work is attempted; it
/// has no bearing on whether that work may use more than one core.
///
/// `region_seq_inner` still forces serial — that is a genuine data-layout
/// constraint, not a scheduling one.
#[inline]
fn position_scatter_parallel_admitted(
    explicit_rows: bool,
    nonsparse: bool,
    deadline_present: bool,
    gate_enabled: bool,
    region_seq_inner: bool,
) -> bool {
    // Retained parameters: the dark gate and layout facts are still honoured as
    // an explicit force-on, so a caller that set them keeps its behaviour.
    let _ = (explicit_rows, nonsparse, deadline_present, gate_enabled);
    !region_seq_inner
}

/// Fill disjoint position slabs in parallel while polling a shared authority.
///
/// `output_scratch` must remain private to the caller until `Ok(())`: Rayon may
/// have completed some independent slabs when another worker observes expiry,
/// but `Err` never returns a publishable value. The production caller owns a
/// fresh `ArrayD` and drops it through `?` on any worker error.
fn fill_position_scatter_scratch_with_poll<B, I, S, P>(
    output_scratch: &mut [f32],
    patch_volume: usize,
    parallel: bool,
    make_buffer: &I,
    scatter: &S,
    poll: &P,
) -> Result<()>
where
    B: Send,
    I: Fn() -> B + Sync,
    S: Fn(&mut B, usize, &mut [f32]) -> Result<()> + Sync,
    P: Fn() -> Result<()> + Sync,
{
    if patch_volume == 0 || !output_scratch.len().is_multiple_of(patch_volume) {
        return Err(NyError::InvalidSpec(
            "Conv2d Patches parallel scatter has invalid scratch geometry".into(),
        ));
    }
    poll()?;
    if parallel {
        output_scratch
            .par_chunks_mut(patch_volume)
            .enumerate()
            .try_for_each_init(make_buffer, |buffer, (idx, chunk)| {
                if idx.is_multiple_of(32) {
                    poll()?;
                }
                scatter(buffer, idx, chunk)
            })?;
    } else {
        let mut buffer = make_buffer();
        for (idx, chunk) in output_scratch.chunks_mut(patch_volume).enumerate() {
            if idx.is_multiple_of(32) {
                poll()?;
            }
            scatter(&mut buffer, idx, chunk)?;
        }
    }
    poll()
}

/// #conv-patches-collect: default-ON gate for the EXACT padded-conv patches
/// composition (intermediate-tap masking). When set, `propagate_patches_engine`
/// masks the out-of-range intermediate taps of a non-identity incoming patch so
/// composing THROUGH a padded conv stays in the memory-light patches
/// representation instead of falling back to the OOM-prone dense CROWN path.
/// `NY_CONV_PATCHES_COLLECT=0` restores the old behavior: the guard below returns
/// `UnsupportedConfiguration` and the caller takes the dense fallback.
fn conv_patches_padded_compose_enabled() -> bool {
    crate::util::conv_patches_collect_enabled()
}

/// Recompute the patches compose in f64 and measure, per position, the largest
/// `|f32_result - f64_result|` over that position's output block
/// (#patches-f64-err).
///
/// Returns `None` when the reference cannot be computed (non-standard layout,
/// shape drift, or a non-finite intermediate) — the caller then keeps its
/// a-priori charge, so this can only ever tighten, never weaken.
///
/// Cost is one extra f64 compose. The compose is not the per-node bottleneck (a
/// live profile put the whole GEMM subtree at ~921 samples against 14,553 spent
/// blocked), and the certified error it buys is what decides whether the CROWN
/// bound beats IBP at all.
#[allow(clippy::too_many_arguments)]
fn patches_f64_reference_gap(
    new_patches: &ArrayD<f32>,
    incoming: &ArrayD<f32>,
    p: &Conv2dPatchesParams<'_>,
    prev_spatial: (usize, usize),
    new_spatial: (usize, usize),
    num_positions: usize,
    patch_volume: usize,
    decode: &(impl Fn(usize) -> (usize, usize, usize, usize) + Sync),
    deadline: Option<Instant>,
) -> Result<Option<Vec<f32>>> {
    let (prev_kh, prev_kw) = prev_spatial;
    let (new_kh, new_kw) = new_spatial;
    let Some(flat) = new_patches.as_slice() else {
        return Ok(None);
    };
    let Some(expected_output_len) = num_positions.checked_mul(patch_volume) else {
        return Err(NyError::InvalidSpec(
            "Conv2d Patches f64 reference output size overflow".into(),
        ));
    };
    if patch_volume == 0 || flat.len() != expected_output_len {
        return Ok(None);
    }
    check_patches_deadline(deadline, "before f64 reference compose")?;

    // BATCHED f64 reference: one `[num_positions x k_dim] @ [k_dim x n_dim]`
    // f64 GEMM over all positions, instead of a per-position scalar f64 scatter.
    // The scalar form measured ~2x the compose cost and pushed targets past the
    // per-node cap (Conv_11 7.3 s -> 12.3 s), which cost more bounds than the
    // tightening won. Same operator matrix the f32 seam uses, widened to f64.
    if let Ok((m32, k_dim, n_dim)) =
        crate::layers::convolution::conv2d::conv2d_transpose_operator_matrix(
            p.kernel, p.sh, p.sw, prev_kh, prev_kw, new_kh, new_kw, p.in_c, p.groups,
        )
    {
        if k_dim > 0 && n_dim == patch_volume && num_positions > 0 {
            let m64: Vec<f64> = m32.iter().map(|&v| f32_to_f64_exact_bits(v)).collect();
            let pmat_len = num_positions.checked_mul(k_dim).ok_or_else(|| {
                NyError::InvalidSpec("Conv2d Patches f64 input matrix size overflow".into())
            })?;
            let mut pmat = vec![0.0f64; pmat_len];
            pmat.par_chunks_mut(k_dim)
                .enumerate()
                .for_each(|(idx, row)| {
                    let (_r, soc, soh, sow) = decode(idx);
                    for c in 0..p.out_c {
                        for ki in 0..prev_kh {
                            for kj in 0..prev_kw {
                                row[(c * prev_kh + ki) * prev_kw + kj] =
                                    f32_to_f64_exact_bits(incoming[[soc, soh, sow, c, ki, kj]]);
                            }
                        }
                    }
                });
            let reference = crate::faer_parallelism::mat_mul_f64_row_major(
                &pmat,
                num_positions,
                k_dim,
                &m64,
                n_dim,
            );
            check_patches_deadline(deadline, "after f64 reference GEMM")?;
            let gaps: Vec<f32> = flat
                .par_chunks_exact(patch_volume)
                .enumerate()
                .map(|(idx, got)| {
                    let mut worst = 0.0f64;
                    for (j, &a) in got.iter().enumerate() {
                        let d = (f32_to_f64_exact_bits(a) - reference[(idx, j)]).abs();
                        if !d.is_finite() {
                            return f32::INFINITY;
                        }
                        if d > worst {
                            worst = d;
                        }
                    }
                    publish_error_up_normal(worst)
                })
                .collect();
            return Ok(Some(gaps));
        }
    }

    let gaps: Vec<f32> = flat
        .par_chunks_exact(patch_volume)
        .enumerate()
        .map(|(idx, got)| {
            let (_row, soc, soh, sow) = decode(idx);
            let mut patch_3d = ArrayD::<f32>::zeros(IxDyn(&[p.out_c, prev_kh, prev_kw]));
            for c in 0..p.out_c {
                for ki in 0..prev_kh {
                    for kj in 0..prev_kw {
                        patch_3d[[c, ki, kj]] = incoming[[soc, soh, sow, c, ki, kj]];
                    }
                }
            }
            let mut reference = vec![0.0f64; patch_volume];
            if crate::layers::convolution::conv2d::conv2d_transpose_grouped_into_f64(
                &mut reference,
                &patch_3d,
                p.kernel,
                (p.sh, p.sw),
                (0, 0),
                (1, 1),
                (new_kh, new_kw),
                p.groups,
            )
            .is_err()
            {
                return f32::INFINITY;
            }
            let mut worst = 0.0f64;
            for (&a, &b) in got.iter().zip(reference.iter()) {
                let d = (f32_to_f64_exact_bits(a) - b).abs();
                if !(d.is_finite()) {
                    return f32::INFINITY;
                }
                if d > worst {
                    worst = d;
                }
            }
            // Outward-round without publishing a flushable subnormal carrier.
            publish_error_up_normal(worst)
        })
        .collect();

    check_patches_deadline(deadline, "after f64 reference compose")?;
    Ok(Some(gaps))
}

#[inline]
fn check_patches_deadline(deadline: Option<Instant>, phase: &str) -> Result<()> {
    if deadline.is_some_and(|limit| Instant::now() >= limit) {
        Err(NyError::DeadlineExceeded(format!(
            "Conv2d Patches backward: deadline exceeded {phase}"
        )))
    } else {
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ExplicitRowBiasReduction {
    lower_sum: f64,
    upper_sum: f64,
    lower_abs: f64,
    upper_abs: f64,
}

/// Reduce both 7D coefficient sides into row-local f64 bias/absolute sums.
///
/// `lower` and `upper` are complete, contiguous row-major slabs. The loop order
/// within each row is exactly the historical
/// `soc -> soh -> sow -> c -> ki -> kj` order. Parallelism is across rows only,
/// so every row retains bit-identical f64 operations. `poll` runs before work
/// and at most every 4,096 coefficient visits. The returned vector is the only
/// output: a failed row cannot expose any partial bias update.
#[allow(clippy::too_many_arguments)]
fn reduce_explicit_row_bias_flat_with_poll<P>(
    lower: &[f32],
    upper: &[f32],
    row_count: usize,
    positions: usize,
    out_c: usize,
    lower_taps: usize,
    upper_taps: usize,
    bias: &[f32],
    old_lower_b: &[f32],
    old_upper_b: &[f32],
    parallel_rows: bool,
    poll: &P,
) -> Result<Vec<ExplicitRowBiasReduction>>
where
    P: Fn() -> Result<()> + Sync,
{
    let lower_row_len =
        checked_shape_product(&[positions, out_c, lower_taps]).ok_or_else(|| {
            NyError::InvalidSpec("Conv2d Patches flat lower bias row size overflow".into())
        })?;
    let upper_row_len =
        checked_shape_product(&[positions, out_c, upper_taps]).ok_or_else(|| {
            NyError::InvalidSpec("Conv2d Patches flat upper bias row size overflow".into())
        })?;
    let expected_lower = row_count.checked_mul(lower_row_len).ok_or_else(|| {
        NyError::InvalidSpec("Conv2d Patches flat lower bias tensor size overflow".into())
    })?;
    let expected_upper = row_count.checked_mul(upper_row_len).ok_or_else(|| {
        NyError::InvalidSpec("Conv2d Patches flat upper bias tensor size overflow".into())
    })?;
    if lower.len() != expected_lower || upper.len() != expected_upper {
        return Err(NyError::ShapeMismatch {
            expected: vec![expected_lower, expected_upper],
            got: vec![lower.len(), upper.len()],
        });
    }
    if bias.len() != out_c {
        return Err(NyError::ShapeMismatch {
            expected: vec![out_c],
            got: vec![bias.len()],
        });
    }
    if old_lower_b.len() != row_count || old_upper_b.len() != row_count {
        return Err(NyError::ShapeMismatch {
            expected: vec![row_count, row_count],
            got: vec![old_lower_b.len(), old_upper_b.len()],
        });
    }
    poll()?;

    let reduce_row = |row: usize| -> Result<ExplicitRowBiasReduction> {
        poll()?;
        let lower_start = row
            .checked_mul(lower_row_len)
            .expect("validated flat lower row geometry");
        let upper_start = row
            .checked_mul(upper_row_len)
            .expect("validated flat upper row geometry");
        let lower_row = &lower[lower_start..lower_start + lower_row_len];
        let upper_row = &upper[upper_start..upper_start + upper_row_len];

        let mut lower_sum = 0.0f64;
        let mut upper_sum = 0.0f64;
        let mut lower_abs = f32_to_f64_exact_bits(old_lower_b[row]).abs();
        let mut upper_abs = f32_to_f64_exact_bits(old_upper_b[row]).abs();
        let mut lower_offset = 0usize;
        let mut upper_offset = 0usize;
        let mut ops_since_poll = 0usize;

        for _position in 0..positions {
            for (c, &bias_c_f32) in bias.iter().enumerate() {
                debug_assert!(c < out_c);
                let mut lc_sum = 0.0f64;
                let mut uc_sum = 0.0f64;
                let mut lc_abs = 0.0f64;
                let mut uc_abs = 0.0f64;

                let lower_end = lower_offset + lower_taps;
                for &coefficient in &lower_row[lower_offset..lower_end] {
                    let a = f32_to_f64_exact_bits(coefficient);
                    lc_sum += a;
                    lc_abs += a.abs();
                    ops_since_poll += 1;
                    if ops_since_poll >= 4_096 {
                        poll()?;
                        ops_since_poll = 0;
                    }
                }
                lower_offset = lower_end;

                let upper_end = upper_offset + upper_taps;
                for &coefficient in &upper_row[upper_offset..upper_end] {
                    let a = f32_to_f64_exact_bits(coefficient);
                    uc_sum += a;
                    uc_abs += a.abs();
                    ops_since_poll += 1;
                    if ops_since_poll >= 4_096 {
                        poll()?;
                        ops_since_poll = 0;
                    }
                }
                upper_offset = upper_end;

                let bias_c = f32_to_f64_exact_bits(bias_c_f32);
                let bias_c_abs = bias_c.abs();
                lower_sum += lc_sum * bias_c;
                upper_sum += uc_sum * bias_c;
                lower_abs += lc_abs * bias_c_abs;
                upper_abs += uc_abs * bias_c_abs;
            }
        }
        debug_assert_eq!(lower_offset, lower_row_len);
        debug_assert_eq!(upper_offset, upper_row_len);
        poll()?;
        Ok(ExplicitRowBiasReduction {
            lower_sum,
            upper_sum,
            lower_abs,
            upper_abs,
        })
    };

    let reductions = if parallel_rows && row_count > 1 {
        (0..row_count)
            .into_par_iter()
            .map(reduce_row)
            .collect::<Result<Vec<_>>>()?
    } else {
        (0..row_count).map(reduce_row).collect::<Result<Vec<_>>>()?
    };
    poll()?;
    Ok(reductions)
}

/// Zero the taps of a **6D dense non-identity** patch that reference OUT-OF-RANGE
/// intermediate positions, in place, before a transpose-compose step
/// (#conv-patches-collect).
///
/// The incoming patch maps its input space Y (this composition's intermediate,
/// == the downstream conv's input = this conv's output) to the spec. A tap at
/// spec position `(soh, sow)`, kernel offset `(ki, kj)` references
/// `yh = soh·prev_sh − prev_pt + ki`, `yw = sow·prev_sw − prev_pl + kj`. When
/// that `(yh, yw)` lies outside `[0, y_h) × [0, y_w)` it addresses the
/// zero-padding the downstream conv added around Y — a HARD zero whose true
/// contribution is 0. The dense operator drops it (`to_dense`'s unfold clips
/// out-of-bounds), but the transpose-compose would otherwise smear it through
/// this conv's kernel onto REAL input cells (the boundary leak the guard below
/// rejects). Zeroing those taps up front makes the compose EXACT — verified
/// bit-close to the dense CROWN backward by
/// `patches_padded_compose_matches_dense` and the padded proptest.
///
/// The row/column in-range predicates are separable, so this is O(spec · out_c ·
/// prev_kh · prev_kw) with tiny precomputed masks. No-op unless the patch is 6D,
/// dense (no `unstable_idx`), non-identity, and carries nonzero padding.
fn mask_out_of_range_intermediate_taps(
    pd: &mut PatchesData,
    deadline: Option<Instant>,
) -> Result<()> {
    use rayon::prelude::*;
    check_patches_deadline(deadline, "before padded-composition masking")?;
    let affine_geometry = pd
        .geometry
        .require_affine("Conv2d Patches padded-composition mask")?;
    if pd.identity || pd.unstable_idx.is_some() || affine_geometry.padding() == (0, 0, 0, 0) {
        return Ok(());
    }
    let Some(patches) = pd.patches.as_mut() else {
        return Ok(());
    };
    let shape = patches.shape().to_vec();
    if shape.len() != 6 {
        return Ok(());
    }
    let (_spec_oc, spec_oh, spec_ow) = (shape[0], shape[1], shape[2]);
    let (out_c, prev_kh, prev_kw) = (shape[3], shape[4], shape[5]);
    if shape.contains(&0) {
        return Err(NyError::InvalidSpec(format!(
            "Conv2d Patches mask dimensions must be nonzero, got {shape:?}"
        )));
    }
    let block = checked_shape_product(&[out_c, prev_kh, prev_kw])
        .ok_or_else(|| NyError::InvalidSpec("Conv2d Patches mask block size overflow".into()))?;
    let (prev_sh, prev_sw) = affine_geometry.stride();
    let (prev_pl, _prev_pr, prev_pt, _prev_pb) = affine_geometry.padding();
    let (y_c, y_h, y_w) = pd.input_shape;
    if prev_sh == 0 || prev_sw == 0 || y_c == 0 || y_h == 0 || y_w == 0 {
        return Err(NyError::InvalidSpec(
            "Conv2d Patches mask stride and input extents must be nonzero".into(),
        ));
    }

    // Separable in-range predicates: yh depends only on (soh, ki); yw on (sow, kj).
    let row_mask_len = spec_oh
        .checked_mul(prev_kh)
        .ok_or_else(|| NyError::InvalidSpec("Conv2d Patches row-mask size overflow".into()))?;
    let mut row_ok = vec![false; row_mask_len];
    for soh in 0..spec_oh {
        for ki in 0..prev_kh {
            let padded_y = soh
                .checked_mul(prev_sh)
                .and_then(|base| base.checked_add(ki))
                .ok_or_else(|| {
                    NyError::InvalidSpec("Conv2d Patches row-mask coordinate overflow".into())
                })?;
            row_ok[soh * prev_kh + ki] = padded_y.checked_sub(prev_pt).is_some_and(|yh| yh < y_h);
        }
    }
    let col_mask_len = spec_ow
        .checked_mul(prev_kw)
        .ok_or_else(|| NyError::InvalidSpec("Conv2d Patches column-mask size overflow".into()))?;
    let mut col_ok = vec![false; col_mask_len];
    for sow in 0..spec_ow {
        for kj in 0..prev_kw {
            let padded_x = sow
                .checked_mul(prev_sw)
                .and_then(|base| base.checked_add(kj))
                .ok_or_else(|| {
                    NyError::InvalidSpec("Conv2d Patches column-mask coordinate overflow".into())
                })?;
            col_ok[sow * prev_kw + kj] = padded_x.checked_sub(prev_pl).is_some_and(|xw| xw < y_w);
        }
    }

    // A spec position owns the contiguous [c, ki, kj] block; positions are
    // row-major over (soc, soh, sow), so `pos` decodes soh/sow directly.
    if let Some(flat) = patches.as_slice_mut() {
        let mask_chunk = |pos: usize, chunk: &mut [f32]| {
            let sow = pos % spec_ow;
            let soh = (pos / spec_ow) % spec_oh;
            for c in 0..out_c {
                for ki in 0..prev_kh {
                    let rok = row_ok[soh * prev_kh + ki];
                    for kj in 0..prev_kw {
                        if !(rok && col_ok[sow * prev_kw + kj]) {
                            chunk[(c * prev_kh + ki) * prev_kw + kj] = 0.0;
                        }
                    }
                }
            }
        };
        // A position whose whole row band AND whole column band are in range has
        // no out-of-range tap, so its entire block is already correct. That is
        // the INTERIOR, i.e. almost every position: only a boundary band of
        // width `prev_kh`/`prev_kw` needs any work. Testing the two bands is
        // `prev_kh + prev_kw` comparisons versus `out_c * prev_kh * prev_kw`
        // writes-with-branch for the block, so the interior gets much cheaper
        // while the result is unchanged.
        let position_needs_mask = |pos: usize| -> bool {
            let sow = pos % spec_ow;
            let soh = (pos / spec_ow) % spec_oh;
            !((0..prev_kh).all(|ki| row_ok[soh * prev_kh + ki])
                && (0..prev_kw).all(|kj| col_ok[sow * prev_kw + kj]))
        };

        // Positions own DISJOINT `block`-sized chunks, so the sweep is
        // parallel-safe and bit-identical either way. It used to run SERIALLY
        // whenever a deadline was present — i.e. always in production, since
        // every scored run carries one — and got rayon only on the
        // deadline-free test path. This is the same `deadline.is_some()`
        // anti-pattern already corrected elsewhere in this file; the mask sweep
        // was missed, and the padded compose it serves is now default-ON, so it
        // runs on every ResNet conv step.
        //
        // Cancellation keeps bounded overshoot: each chunk polls the deadline
        // and, once expired, every remaining chunk returns immediately. The
        // caller discards the (partially masked) local clone on the `Err`
        // below, so no partial mask can reach a bound.
        let expired = std::sync::atomic::AtomicBool::new(false);
        flat.par_chunks_mut(block)
            .enumerate()
            .for_each(|(pos, chunk)| {
                use std::sync::atomic::Ordering;
                if expired.load(Ordering::Relaxed) {
                    return;
                }
                if pos.is_multiple_of(64) && deadline.is_some_and(|d| Instant::now() >= d) {
                    expired.store(true, Ordering::Relaxed);
                    return;
                }
                if position_needs_mask(pos) {
                    mask_chunk(pos, chunk);
                }
            });
        if expired.load(std::sync::atomic::Ordering::Relaxed) {
            return Err(NyError::DeadlineExceeded(
                "Conv2d Patches backward: deadline exceeded during padded-composition masking"
                    .to_string(),
            ));
        }
    }
    check_patches_deadline(deadline, "after padded-composition masking")
}

impl PatchesPropagation for Conv2dLayer {
    /// CROWN backward with Patches coefficients for Conv2d.
    ///
    /// Supports both identity incoming patches (first Conv2d in backward chain)
    /// and non-identity patches (chained Conv2d with composition).
    ///
    /// For identity patches: creates initial patches from the conv kernel.
    /// For non-identity patches: composes by applying conv2d_transpose to each
    /// patch, producing a larger receptive field with composed stride/padding.
    ///
    /// Composition math (reference: alpha-beta-CROWN auto_LiRPA/patches.py):
    /// - new_kh = (prev_kh - 1) * stride_h + kh
    /// - new_stride = prev_stride * conv_stride
    /// - new_padding = prev_padding * conv_stride + conv_padding
    ///
    /// Design: designs/2026-02-28-patches-mode-wrapper-enum-design.md
    fn propagate_patches(&self, bounds: &PatchesLinearBounds) -> Result<CrownBounds> {
        self.propagate_patches_engine(bounds, None)
    }
}

impl Conv2dLayer {
    /// Engine-aware patches Conv2d CROWN backward. With `engine` present AND
    /// `NY_PATCHES_GPU` set, the per-position transpose-conv composition runs as
    /// a single GEMM (GPU-capable) instead of the rayon per-position CPU scatter
    /// — the keystone for getting the conv-CROWN warmup onto the GPU while
    /// staying in the memory-light patches representation. With `engine` None or
    /// the flag unset the result is byte-identical to the CPU path.
    pub(crate) fn propagate_patches_engine(
        &self,
        bounds: &PatchesLinearBounds,
        engine: Option<&dyn ny_core::GemmEngine>,
    ) -> Result<CrownBounds> {
        self.propagate_patches_engine_and_deadline(bounds, engine, None)
    }

    /// Deadline-bearing patches-native Conv2d CROWN backward.
    ///
    /// Finite-deadline work deliberately bypasses caller/process-global GEMM
    /// engines and uses the scalar CPU composition. That route polls within
    /// patch positions and contractions; expiry discards all local partial
    /// tensors and surfaces as `DeadlineExceeded`. `deadline: None` preserves
    /// the historical engine/Rayon path.
    pub(crate) fn propagate_patches_engine_and_deadline(
        &self,
        bounds: &PatchesLinearBounds,
        engine: Option<&dyn ny_core::GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<CrownBounds> {
        let input_shape = self.input_shape.ok_or_else(|| {
            NyError::UnsupportedConfiguration(
                "Conv2d Patches CROWN requires input_shape".to_string(),
            )
        })?;
        self.propagate_patches_engine_and_deadline_for_input_shape(
            bounds,
            engine,
            deadline,
            input_shape,
        )
    }

    /// Borrowing variant used by graph/sequential dispatchers that already
    /// authenticate the current pre-activation shape. This avoids cloning the
    /// layer's full kernel and bias merely to install transient shape metadata.
    pub(crate) fn propagate_patches_engine_and_deadline_for_input_shape(
        &self,
        bounds: &PatchesLinearBounds,
        engine: Option<&dyn ny_core::GemmEngine>,
        deadline: Option<Instant>,
        (in_h, in_w): (usize, usize),
    ) -> Result<CrownBounds> {
        check_patches_deadline(deadline, "before dispatch")?;
        let (in_c, out_c) = self.validate_geometry()?;
        // Anchored geometry is not implemented by this affine composition
        // kernel. Refuse it in O(1) before paired validation walks either
        // origin axis under a finite node deadline.
        let common_geometry = bounds
            .lower_a
            .geometry
            .require_affine("Conv2d Patches propagation")?;
        bounds
            .upper_a
            .geometry
            .require_affine("Conv2d Patches propagation")?;
        bounds.lower_a.validate_common_geometry(&bounds.upper_a)?;

        // Guard: reject NaN weights without hiding a full uninterruptible
        // kernel scan inside finite node authority.
        for (index, value) in self.kernel.iter().enumerate() {
            if index.is_multiple_of(4_096) {
                check_patches_deadline(deadline, "during kernel NaN scan")?;
            }
            if value.is_nan() {
                return Err(NyError::NumericalInstability(
                    "Conv2d Patches backward: kernel contains NaN".into(),
                ));
            }
        }
        check_patches_deadline(deadline, "after kernel NaN scan")?;

        // The Patches composition math (new_kh = (prev_kh-1)*stride + kh) does not
        // yet account for dilation. Reject dilated convolutions here so the caller
        // falls back to the dilation-aware dense CROWN path; never silently
        // produce wrong bounds.
        if self.dilation != (1, 1) {
            return Err(NyError::UnsupportedConfiguration(format!(
                "Conv2d Patches CROWN does not support dilation {:?}; use dense CROWN",
                self.dilation
            )));
        }

        // SOUNDNESS GUARD (#hotpath): composing a non-identity patch through this
        // Conv2d is only equivalent to the dense operator when the INCOMING patches
        // carry zero padding. When the already-composed patches have nonzero
        // padding, the boundary-truncation of the intermediate conv (its kernel is
        // clipped against the input edge) is not reconstructible from the composed
        // `conv2d_transpose` (which runs with padding=0) plus the additive padding
        // metadata: edge output positions get coefficients that disagree with dense
        // (verified: interior rows match, boundary rows diverge). Single-layer
        // padding is sound because `to_dense`'s unfold clips correctly; the issue
        // is ONLY padding accumulated ACROSS a composition step. Reject here so the
        // caller falls back to the exact dense CROWN path. Identity incoming patches
        // and zero-padding chains (the common stride-1/pad-0 and pad-after-pad-0
        // cases) remain in patches mode.
        // #conv-patches-collect: the leak is EXACTLY the out-of-range intermediate
        // taps (the downstream conv's zero-padding around this conv's output). When
        // the padded-compose feature is enabled AND both incoming sides are 6D
        // dense maskable patches, `mask_out_of_range_intermediate_taps` zeros those
        // taps up front, which makes the transpose-compose bit-equivalent to the
        // dense operator (parity-tested) — so the composition stays SOUND and in
        // patches. Otherwise (feature off, or a 7D explicit-rows / sparse incoming
        // the mask does not cover) keep the exact pre-existing dense fallback.
        let nonzero_incoming_padding = (!bounds.lower_a.identity
            && common_geometry.padding() != (0, 0, 0, 0))
            || (!bounds.upper_a.identity && common_geometry.padding() != (0, 0, 0, 0));
        let masked_bounds_storage;
        let bounds: &PatchesLinearBounds = if nonzero_incoming_padding {
            let side_maskable = |pd: &PatchesData| -> bool {
                pd.identity
                    || (pd.unstable_idx.is_none()
                        && pd.patches.as_ref().map(|p| p.ndim() == 6).unwrap_or(false))
            };
            let can_mask = conv_patches_padded_compose_enabled()
                && side_maskable(&bounds.lower_a)
                && side_maskable(&bounds.upper_a);
            if !can_mask {
                return Err(NyError::UnsupportedConfiguration(
                    "Conv2d Patches CROWN cannot soundly compose through incoming patches \
                     with nonzero padding; use dense CROWN"
                        .to_string(),
                ));
            }
            if std::env::var_os("NY_CONV_PATCHES_DEBUG").is_some_and(|v| v != "0" && !v.is_empty())
            {
                eprintln!(
                    "[conv-patches-dbg] MASK compose: in_c={} out_c={} incoming_pad={:?} spec_rows={}",
                    self.in_channels(),
                    self.out_channels(),
                    common_geometry.padding(),
                    bounds.row_count,
                );
            }
            let mut masked = bounds.clone();
            mask_out_of_range_intermediate_taps(&mut masked.lower_a, deadline)?;
            mask_out_of_range_intermediate_taps(&mut masked.upper_a, deadline)?;
            masked_bounds_storage = masked;
            &masked_bounds_storage
        } else {
            bounds
        };

        let (kh, kw) = self.kernel_size();
        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;
        let (out_h, out_w) = self.output_size(in_h, in_w)?;
        let params = Conv2dPatchesParams {
            kernel: &self.kernel,
            in_c,
            out_c,
            groups: self.groups,
            kh,
            kw,
            sh,
            sw,
            ph,
            pw,
            in_h,
            in_w,
            out_h,
            out_w,
        };

        // #patches-deadline-kernel: admit the GEMM engine even under a finite
        // deadline, provided the deadline has not already expired.
        //
        // The previous rule — `if deadline.is_some() { None }` — made the
        // deadline's PRESENCE, not its value, select the kernel. Every caller in
        // the CROWN-IBP collector passes a per-node deadline
        // (`crown_tighten.rs`), so the engine was nulled on literally every
        // collector-routed conv target and composition fell to the pollable
        // scalar per-position loop. That is why raising the per-node budget 25x
        // (12 s -> 300 s) and the memory cap 18x on TinyYOLO produced a
        // BYTE-IDENTICAL root bound: the extra seconds bought more of the
        // slowest available implementation, not more progress.
        //
        // The original concern is real but is answered by checking the clock
        // rather than by refusing the fast path: a generic engine does not
        // promise cooperative cancellation, so it may overrun. That overrun is
        // BOUNDED by one call and is a scheduling cost only — a deadline
        // controls how much work is attempted, never whether a computed bound is
        // valid. Every caller already treats `DeadlineExceeded` as a sound IBP
        // fallback. Trading a bounded overrun for the GEMM path is the correct
        // side of that trade, and refusing it was strictly worse: the scalar
        // route overran the same deadline anyway, just without finishing.
        //
        // Fail-closed on an ALREADY-expired authority: with no time left, do not
        // launch an uncancellable call. `check_patches_deadline` below then
        // reports the expiry exactly as before.
        let deadline_live = deadline.is_none_or(|d| Instant::now() < d);
        let admitted_engine = if deadline_live { engine } else { None };
        let lower_result = Self::conv2d_patches_backward_with_deadline(
            &bounds.lower_a,
            &params,
            admitted_engine,
            deadline,
        )?;
        check_patches_deadline(deadline, "between lower and upper propagation")?;
        let upper_result = Self::conv2d_patches_backward_with_deadline(
            &bounds.upper_a,
            &params,
            admitted_engine,
            deadline,
        )?;

        let (new_lower_b, new_upper_b) = if let Some(ref bias) = self.bias {
            Self::compute_patches_bias_with_deadline(bounds, bias, out_c, out_h, out_w, deadline)?
        } else {
            (bounds.lower_b.clone(), bounds.upper_b.clone())
        };
        check_patches_deadline(deadline, "after propagation")?;

        if lower_result.should_fallback_to_dense() || upper_result.should_fallback_to_dense() {
            let plb = PatchesLinearBounds {
                row_count: bounds.row_count,
                lower_a: lower_result,
                lower_b: new_lower_b,
                upper_a: upper_result,
                upper_b: new_upper_b,
            };
            return Ok(CrownBounds::Dense(plb.to_dense()?));
        }

        Ok(CrownBounds::Patches(Box::new(PatchesLinearBounds {
            row_count: bounds.row_count,
            lower_a: lower_result,
            lower_b: new_lower_b,
            upper_a: upper_result,
            upper_b: new_upper_b,
        })))
    }
}

/// Convolution parameters for Patches backward propagation.
struct Conv2dPatchesParams<'a> {
    kernel: &'a ArrayD<f32>,
    in_c: usize,
    out_c: usize,
    groups: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
}

impl Conv2dLayer {
    /// Compute patches backward for a single PatchesData (lower_a or upper_a).
    ///
    /// For identity patches: creates initial patches from the conv kernel.
    /// For existing patches: composes by applying conv2d_transpose to each
    /// patch position, producing a larger receptive field.
    ///
    /// Composition math (reference: alpha-beta-CROWN auto_LiRPA/patches.py):
    /// - Each patch of shape (prev_in_c, prev_kh, prev_kw) is convolved backward
    ///   through this Conv2d via conv2d_transpose(patch, kernel, stride, padding=0)
    /// - Output patch shape: (in_c, new_kh, new_kw) where
    ///   new_kh = (prev_kh - 1) * stride_h + kh
    /// - Composed stride: prev_stride * conv_stride
    /// - Composed padding: prev_padding * conv_stride + conv_padding
    //
    // Retained as the no-deadline compatibility face used by regression tests;
    // production propagation calls the deadline-aware implementation below.
    #[allow(dead_code)]
    fn conv2d_patches_backward(
        patches_data: &PatchesData,
        p: &Conv2dPatchesParams<'_>,
        engine: Option<&dyn ny_core::GemmEngine>,
    ) -> Result<PatchesData> {
        Self::conv2d_patches_backward_with_deadline(patches_data, p, engine, None)
    }

    fn conv2d_patches_backward_with_deadline(
        patches_data: &PatchesData,
        p: &Conv2dPatchesParams<'_>,
        engine: Option<&dyn ny_core::GemmEngine>,
        deadline: Option<Instant>,
    ) -> Result<PatchesData> {
        check_patches_deadline(deadline, "before coefficient propagation")?;
        let incoming_geometry = patches_data
            .geometry
            .require_affine("Conv2d Patches coefficient propagation")?;
        if p.kernel.ndim() != 4 {
            return Err(NyError::ShapeMismatch {
                expected: vec![4],
                got: vec![p.kernel.ndim()],
            });
        }
        if p.in_c == 0
            || p.out_c == 0
            || p.groups == 0
            || p.kh == 0
            || p.kw == 0
            || p.sh == 0
            || p.sw == 0
            || p.in_h == 0
            || p.in_w == 0
            || p.out_h == 0
            || p.out_w == 0
            || p.kernel.shape().contains(&0)
        {
            return Err(NyError::InvalidSpec(
                "Conv2d Patches geometry dimensions, channels, stride, and groups must be nonzero"
                    .into(),
            ));
        }
        if p.kernel.shape()[0] != p.out_c
            || p.kernel.shape()[2] != p.kh
            || p.kernel.shape()[3] != p.kw
        {
            return Err(NyError::ShapeMismatch {
                expected: vec![p.out_c, p.kernel.shape()[1], p.kh, p.kw],
                got: p.kernel.shape().to_vec(),
            });
        }
        let grouped_in_c = p.kernel.shape()[1].checked_mul(p.groups).ok_or_else(|| {
            NyError::InvalidSpec("Conv2d Patches grouped input channels overflow".into())
        })?;
        if grouped_in_c != p.in_c || !p.out_c.is_multiple_of(p.groups) {
            return Err(NyError::InvalidSpec(format!(
                "Conv2d Patches incompatible channels/groups: in_c={}, out_c={}, groups={}, \
                 kernel={:?}",
                p.in_c,
                p.out_c,
                p.groups,
                p.kernel.shape()
            )));
        }
        if patches_data.identity {
            // First Conv2d in backward chain: create initial patches from kernel.
            // Patches shape: (spec_oc, spec_oh, spec_ow, in_c, kH, kW)
            // For identity, spec output shape = this conv's output shape.
            //
            // Crash guard (#hotpath robustness): the build loop below derives
            // `out_c_per_group = out_c / groups` and `ic_start = group_idx *
            // in_c_per_group` and then indexes `kernel[[oc, ic_local, ..]]` and
            // `patches[[.., ic, ..]]` without rechecking that the grouped layout is
            // self-consistent. If the layer's `groups` metadata is inconsistent with
            // its channel counts (`groups == 0`, `out_c` not divisible by `groups`,
            // or `in_c_per_group * groups != in_c`), those indices run out of bounds
            // and panic. Reject such a malformed convolution with a clean
            // ShapeMismatch before the loop. No bound math changes.
            let in_c_per_group = p.kernel.shape()[1];
            if p.groups == 0
                || !p.out_c.is_multiple_of(p.groups)
                || in_c_per_group.checked_mul(p.groups) != Some(p.in_c)
            {
                return Err(NyError::ShapeMismatch {
                    expected: vec![p.in_c, p.out_c, p.groups],
                    got: vec![in_c_per_group.saturating_mul(p.groups), p.out_c, p.groups],
                });
            }
            checked_shape_product(&[p.out_c, p.out_h, p.out_w, p.in_c, p.kh, p.kw]).ok_or_else(
                || NyError::InvalidSpec("Conv2d identity Patches tensor size overflow".into()),
            )?;
            let mut patches =
                ArrayD::<f32>::zeros(IxDyn(&[p.out_c, p.out_h, p.out_w, p.in_c, p.kh, p.kw]));
            check_patches_deadline(deadline, "after identity-patches allocation")?;
            let out_c_per_group = p.out_c / p.groups;
            let mut position = 0usize;
            for oc in 0..p.out_c {
                let group_idx = oc / out_c_per_group;
                let ic_start = group_idx * in_c_per_group;
                for oh in 0..p.out_h {
                    for ow in 0..p.out_w {
                        if position.is_multiple_of(64) {
                            check_patches_deadline(
                                deadline,
                                "during identity coefficient propagation",
                            )?;
                        }
                        position += 1;
                        for ic_local in 0..in_c_per_group {
                            let ic = ic_start + ic_local;
                            for ki in 0..p.kh {
                                for kj in 0..p.kw {
                                    patches[[oc, oh, ow, ic, ki, kj]] =
                                        p.kernel[[oc, ic_local, ki, kj]];
                                }
                            }
                        }
                    }
                }
            }
            check_patches_deadline(deadline, "after identity coefficient propagation")?;
            Ok(PatchesData {
                coeff_err: None,
                patches: Some(patches),
                geometry: PatchGeometry::affine((p.sh, p.sw), (p.pw, p.pw, p.ph, p.ph)),
                identity: false,
                output_shape: (p.out_c, p.out_h, p.out_w),
                input_shape: (p.in_c, p.in_h, p.in_w),
                unstable_idx: None,
            })
        } else {
            // Non-identity: compose existing patches through this Conv2d.
            // Each patch represents coefficients in this Conv2d's OUTPUT space.
            // Apply conv2d_transpose to map them to this Conv2d's INPUT space.
            let incoming = patches_data.patches.as_ref().ok_or_else(|| {
                NyError::InternalError(
                    "PatchesData: not identity but patches tensor is None".into(),
                )
            })?;
            let shape = incoming.shape();
            let (spec_oc, spec_oh, spec_ow) = patches_data.output_shape;
            let explicit_rows = match shape.len() {
                6 => false,
                7 => true,
                _ => {
                    return Err(NyError::ShapeMismatch {
                        expected: vec![6, 7],
                        got: vec![shape.len()],
                    });
                }
            };
            let row_count = if explicit_rows { shape[0] } else { 1 };
            let (channel_axis, kh_axis, kw_axis) =
                if explicit_rows { (4, 5, 6) } else { (3, 4, 5) };
            let prev_kh = shape[kh_axis];
            let prev_kw = shape[kw_axis];

            if shape.contains(&0) {
                return Err(NyError::InvalidSpec(format!(
                    "Conv2d Patches incoming dimensions must be nonzero, got {shape:?}"
                )));
            }
            let prefix_matches = if explicit_rows {
                shape[1..4] == [spec_oc, spec_oh, spec_ow]
            } else {
                shape[..3] == [spec_oc, spec_oh, spec_ow]
            };
            if !prefix_matches || shape[channel_axis] != p.out_c {
                let expected = if explicit_rows {
                    vec![
                        row_count, spec_oc, spec_oh, spec_ow, p.out_c, prev_kh, prev_kw,
                    ]
                } else {
                    vec![spec_oc, spec_oh, spec_ow, p.out_c, prev_kh, prev_kw]
                };
                return Err(NyError::ShapeMismatch {
                    expected,
                    got: shape.to_vec(),
                });
            }
            if patches_data.input_shape != (p.out_c, p.out_h, p.out_w) {
                return Err(NyError::ShapeMismatch {
                    expected: vec![p.out_c, p.out_h, p.out_w],
                    got: vec![
                        patches_data.input_shape.0,
                        patches_data.input_shape.1,
                        patches_data.input_shape.2,
                    ],
                });
            }

            let new_kh = prev_kh
                .checked_sub(1)
                .and_then(|extent| extent.checked_mul(p.sh))
                .and_then(|extent| extent.checked_add(p.kh))
                .ok_or_else(|| {
                    NyError::InvalidSpec("Conv2d Patches composed kernel height overflow".into())
                })?;
            let new_kw = prev_kw
                .checked_sub(1)
                .and_then(|extent| extent.checked_mul(p.sw))
                .and_then(|extent| extent.checked_add(p.kw))
                .ok_or_else(|| {
                    NyError::InvalidSpec("Conv2d Patches composed kernel width overflow".into())
                })?;

            let (prev_sh, prev_sw) = incoming_geometry.stride();
            if prev_sh == 0 || prev_sw == 0 {
                return Err(NyError::InvalidSpec(
                    "Conv2d Patches incoming stride must be nonzero".into(),
                ));
            }
            let new_stride = (
                prev_sh.checked_mul(p.sh).ok_or_else(|| {
                    NyError::InvalidSpec("Conv2d Patches composed height stride overflow".into())
                })?,
                prev_sw.checked_mul(p.sw).ok_or_else(|| {
                    NyError::InvalidSpec("Conv2d Patches composed width stride overflow".into())
                })?,
            );

            let (prev_pl, prev_pr, prev_pt, prev_pb) = incoming_geometry.padding();
            let new_padding = (
                prev_pl
                    .checked_mul(p.sw)
                    .and_then(|value| value.checked_add(p.pw))
                    .ok_or_else(|| {
                        NyError::InvalidSpec("Conv2d Patches left padding overflow".into())
                    })?,
                prev_pr
                    .checked_mul(p.sw)
                    .and_then(|value| value.checked_add(p.pw))
                    .ok_or_else(|| {
                        NyError::InvalidSpec("Conv2d Patches right padding overflow".into())
                    })?,
                prev_pt
                    .checked_mul(p.sh)
                    .and_then(|value| value.checked_add(p.ph))
                    .ok_or_else(|| {
                        NyError::InvalidSpec("Conv2d Patches top padding overflow".into())
                    })?,
                prev_pb
                    .checked_mul(p.sh)
                    .and_then(|value| value.checked_add(p.ph))
                    .ok_or_else(|| {
                        NyError::InvalidSpec("Conv2d Patches bottom padding overflow".into())
                    })?,
            );

            let spatial_positions = spec_oh.checked_mul(spec_ow).ok_or_else(|| {
                NyError::InvalidSpec("Conv2d Patches spatial-position count overflow".into())
            })?;
            let positions_per_row = spec_oc.checked_mul(spatial_positions).ok_or_else(|| {
                NyError::InvalidSpec("Conv2d Patches output-position count overflow".into())
            })?;
            let num_positions = row_count.checked_mul(positions_per_row).ok_or_else(|| {
                NyError::InvalidSpec("Conv2d Patches total position count overflow".into())
            })?;
            let patch_volume =
                checked_shape_product(&[p.in_c, new_kh, new_kw]).ok_or_else(|| {
                    NyError::InvalidSpec("Conv2d Patches composed patch volume overflow".into())
                })?;
            num_positions.checked_mul(patch_volume).ok_or_else(|| {
                NyError::InvalidSpec("Conv2d Patches composed tensor size overflow".into())
            })?;
            let decode = |idx: usize| {
                let row = idx / positions_per_row;
                let position_idx = idx % positions_per_row;
                let soc = position_idx / spatial_positions;
                let rem = position_idx % spatial_positions;
                (row, soc, rem / spec_ow, rem % spec_ow)
            };
            let region_seq_inner = crate::imb::region_seq_inner();
            let parallel_positions = position_scatter_parallel_admitted(
                explicit_rows,
                patches_data.unstable_idx.is_none(),
                deadline.is_some(),
                patches_deadline_parallel_scatter_enabled(),
                region_seq_inner,
            );

            // Seam gate (#patches-coeff-err-soundness): route the per-position
            // transpose-conv composition to ONE GEMM. Now enabled for BOTH the
            // 6D dense and 7D explicit-rows layouts — the 7D closure
            // (docs/PATCHES_7D_COEFF_ERR_CLOSURE.md §5, §14 C5) made 7D carry a
            // certified coeff_err channel, and the γ_K bound is summation-order
            // independent, so it certifies the engine GEMM's arbitrary
            // accumulation order for either layout. The certified err is computed
            // from the INCOMING coefficients (below), never from this GEMM's
            // output, so the seam only substitutes a value-equivalent (up to the
            // certified rounding) faster compose. Runs when `NY_PATCHES_GPU` is
            // set AND either a GEMM engine is threaded here OR the process-global
            // fast f32 accelerator (cuBLAS) is installed — the latter reaches the
            // CPU-routed scored workload, whose `engine` is `None`.
            // #patches-deadline-kernel: same correction as the engine admission
            // above — gate on a LIVE deadline, not on the mere presence of one.
            // `deadline.is_none()` disabled this seam on every collector-routed
            // target, since the collector always passes one. The seam is
            // value-equivalent up to the certified rounding (see the paragraph
            // above), so admitting it under a live authority changes speed only.
            let seam_enabled = deadline.is_none_or(|d| Instant::now() < d)
                && patches_gpu_enabled()
                && (engine.is_some() || crate::fast_f32_gemm::is_installed());
            let engine_batched: Option<Vec<f32>> = if seam_enabled {
                // Build the [num_positions × out_c·prev_kh·prev_kw] input matrix in
                // the same (oc, ki, kj) flattening the operator matrix expects.
                // Each position owns the disjoint row [idx·k_dim, (idx+1)·k_dim),
                // so the gather parallelizes per position with per-element values
                // unchanged. k_dim == 0 gathers nothing (par_chunks_mut requires a
                // nonzero chunk length).
                let k_dim =
                    checked_shape_product(&[p.out_c, prev_kh, prev_kw]).ok_or_else(|| {
                        NyError::InvalidSpec("Conv2d Patches input patch volume overflow".into())
                    })?;
                let pmat_len = num_positions.checked_mul(k_dim).ok_or_else(|| {
                    NyError::InvalidSpec("Conv2d Patches input matrix size overflow".into())
                })?;
                let mut pmat = vec![0.0f32; pmat_len];
                if k_dim > 0 {
                    pmat.par_chunks_mut(k_dim)
                        .enumerate()
                        .for_each(|(idx, prow)| {
                            let (row, soc, soh, sow) = decode(idx);
                            for c in 0..p.out_c {
                                for ki in 0..prev_kh {
                                    for kj in 0..prev_kw {
                                        prow[(c * prev_kh + ki) * prev_kw + kj] = if explicit_rows {
                                            incoming[[row, soc, soh, sow, c, ki, kj]]
                                        } else {
                                            incoming[[soc, soh, sow, c, ki, kj]]
                                        };
                                    }
                                }
                            }
                        });
                }
                // Route large products to cuBLAS (unified memory), else the passed
                // engine; `None` ⇒ below the MACs gate with no passed engine ⇒
                // fall through to the per-position CPU scatter.
                match conv2d_transpose_grouped_batched_fast(
                    &pmat,
                    num_positions,
                    p.kernel,
                    (p.sh, p.sw),
                    (prev_kh, prev_kw),
                    (new_kh, new_kw),
                    p.in_c,
                    p.groups,
                    engine,
                    deadline,
                ) {
                    Some(batched) => {
                        let batched = batched?;
                        debug_assert_eq!(
                            batched.len(),
                            num_positions
                                .checked_mul(patch_volume)
                                .expect("validated composed tensor size")
                        );
                        Some(batched)
                    }
                    None => None,
                }
            } else {
                None
            };

            let mut new_patches = if explicit_rows {
                ArrayD::<f32>::zeros(IxDyn(&[
                    row_count, spec_oc, spec_oh, spec_ow, p.in_c, new_kh, new_kw,
                ]))
            } else {
                ArrayD::<f32>::zeros(IxDyn(&[spec_oc, spec_oh, spec_ow, p.in_c, new_kh, new_kw]))
            };
            check_patches_deadline(deadline, "after composed-patches allocation")?;

            // Position idx (row-major over (row, soc, soh, sow)) owns exactly the
            // flat range [idx·patch_volume, (idx+1)·patch_volume) of the output in
            // both the 6D and 7D layouts, so positions fill disjoint chunks in
            // parallel with a per-thread reused gather buffer instead of a fresh
            // ArrayD + Vec per position. Per-position math is unchanged; only
            // allocation and scheduling differ. patch_volume == 0 falls through
            // (par_chunks_mut requires a nonzero chunk length).
            let filled_direct = if patch_volume == 0 {
                false
            } else if let Some(flat_out) = new_patches.as_slice_mut() {
                if let Some(ref batched) = engine_batched {
                    debug_assert_eq!(batched.len(), flat_out.len());
                    flat_out.copy_from_slice(batched);
                } else {
                    // Per-position scatter into the caller-owned output `chunk` (already
                    // `patch_volume` long, disjoint per position). `patch_3d` is a reused
                    // gather buffer — every (c, ki, kj) is overwritten, so no re-zero
                    // needed. `_into` re-zeros `chunk`, so the result is byte-identical
                    // to the owned form. Positions are DISJOINT, so serial and parallel
                    // give bit-identical output.
                    let make_buf = || ArrayD::<f32>::zeros(IxDyn(&[p.out_c, prev_kh, prev_kw]));
                    let scatter =
                        |patch_3d: &mut ArrayD<f32>, idx: usize, chunk: &mut [f32]| -> Result<()> {
                            let (row, soc, soh, sow) = decode(idx);
                            for c in 0..p.out_c {
                                for ki in 0..prev_kh {
                                    for kj in 0..prev_kw {
                                        patch_3d[[c, ki, kj]] = if explicit_rows {
                                            incoming[[row, soc, soh, sow, c, ki, kj]]
                                        } else {
                                            incoming[[soc, soh, sow, c, ki, kj]]
                                        };
                                    }
                                }
                            }
                            if deadline.is_some() {
                                conv2d_transpose_grouped_into_with_deadline(
                                    chunk,
                                    patch_3d,
                                    p.kernel,
                                    (p.sh, p.sw),
                                    (0, 0),
                                    (1, 1),
                                    (new_kh, new_kw),
                                    p.groups,
                                    deadline,
                                )?;
                            } else {
                                conv2d_transpose_grouped_into(
                                    chunk,
                                    patch_3d,
                                    p.kernel,
                                    (p.sh, p.sw),
                                    (0, 0),
                                    (1, 1),
                                    (new_kh, new_kw),
                                    p.groups,
                                )?;
                            }
                            Ok(())
                        };
                    // Inside a parallel IMB region loop, run the scatter SERIALLY — this
                    // backward is already one region-worker's single-core slice, so a
                    // nested `par_chunks_mut` would fan out on the region pool and starve
                    // the N-way region parallelism (`crate::imb::region_seq_inner`).
                    if deadline.is_some() && parallel_positions {
                        let poll = || {
                            check_patches_deadline(
                                deadline,
                                "during parallel composed coefficient propagation",
                            )
                        };
                        fill_position_scatter_scratch_with_poll(
                            flat_out,
                            patch_volume,
                            true,
                            &make_buf,
                            &scatter,
                            &poll,
                        )?;
                    } else if deadline.is_some() || region_seq_inner {
                        let mut buf = make_buf();
                        for (idx, chunk) in flat_out.chunks_mut(patch_volume).enumerate() {
                            if idx.is_multiple_of(32) {
                                check_patches_deadline(
                                    deadline,
                                    "during composed coefficient propagation",
                                )?;
                            }
                            scatter(&mut buf, idx, chunk)?;
                        }
                    } else {
                        flat_out
                            .par_chunks_mut(patch_volume)
                            .enumerate()
                            .try_for_each_init(make_buf, |patch_3d, (idx, chunk)| {
                                scatter(patch_3d, idx, chunk)
                            })?;
                    }
                }
                true
            } else {
                false
            };
            check_patches_deadline(deadline, "after composed coefficient propagation")?;

            if !filled_direct {
                // Non-standard output layout or zero-volume patches (unreachable for
                // the freshly zeroed tensor above; kept so degenerate shapes retain
                // the original behavior): per-position collect + serial writeback.
                let composed_patches: Vec<Vec<f32>> = match engine_batched {
                    Some(batched) if patch_volume > 0 => {
                        batched.chunks(patch_volume).map(<[f32]>::to_vec).collect()
                    }
                    Some(_) => vec![Vec::new(); num_positions],
                    None if deadline.is_some() => {
                        let mut composed = Vec::with_capacity(num_positions);
                        for idx in 0..num_positions {
                            if idx.is_multiple_of(32) {
                                check_patches_deadline(
                                    deadline,
                                    "during fallback composed coefficient propagation",
                                )?;
                            }
                            let (row, soc, soh, sow) = decode(idx);
                            let mut patch_3d =
                                ArrayD::<f32>::zeros(IxDyn(&[p.out_c, prev_kh, prev_kw]));
                            for c in 0..p.out_c {
                                for ki in 0..prev_kh {
                                    for kj in 0..prev_kw {
                                        patch_3d[[c, ki, kj]] = if explicit_rows {
                                            incoming[[row, soc, soh, sow, c, ki, kj]]
                                        } else {
                                            incoming[[soc, soh, sow, c, ki, kj]]
                                        };
                                    }
                                }
                            }
                            let mut flat = vec![0.0f32; patch_volume];
                            conv2d_transpose_grouped_into_with_deadline(
                                &mut flat,
                                &patch_3d,
                                p.kernel,
                                (p.sh, p.sw),
                                (0, 0),
                                (1, 1),
                                (new_kh, new_kw),
                                p.groups,
                                deadline,
                            )?;
                            composed.push(flat);
                        }
                        composed
                    }
                    None => (0..num_positions)
                        .into_par_iter()
                        .map(|idx| {
                            let (row, soc, soh, sow) = decode(idx);

                            let mut patch_3d =
                                ArrayD::<f32>::zeros(IxDyn(&[p.out_c, prev_kh, prev_kw]));
                            for c in 0..p.out_c {
                                for ki in 0..prev_kh {
                                    for kj in 0..prev_kw {
                                        patch_3d[[c, ki, kj]] = if explicit_rows {
                                            incoming[[row, soc, soh, sow, c, ki, kj]]
                                        } else {
                                            incoming[[soc, soh, sow, c, ki, kj]]
                                        };
                                    }
                                }
                            }

                            let composed = conv2d_transpose_grouped(
                                &patch_3d,
                                p.kernel,
                                (p.sh, p.sw),
                                (0, 0),
                                (1, 1),
                                (new_kh, new_kw),
                                p.groups,
                            )?;

                            let flat: Vec<f32> = composed.iter().copied().collect();
                            debug_assert_eq!(flat.len(), patch_volume);
                            Ok(flat)
                        })
                        .collect::<Result<Vec<_>>>()?,
                };
                for (idx, flat) in composed_patches.iter().enumerate() {
                    if idx.is_multiple_of(32) {
                        check_patches_deadline(
                            deadline,
                            "during fallback composed-patches writeback",
                        )?;
                    }
                    let row = idx / positions_per_row;
                    let position_idx = idx % positions_per_row;
                    let soc = position_idx / spatial_positions;
                    let rem = position_idx % spatial_positions;
                    let soh = rem / spec_ow;
                    let sow = rem % spec_ow;
                    let mut fi = 0;
                    for ic in 0..p.in_c {
                        for ni in 0..new_kh {
                            for nj in 0..new_kw {
                                if explicit_rows {
                                    new_patches[[row, soc, soh, sow, ic, ni, nj]] = flat[fi];
                                } else {
                                    new_patches[[soc, soh, sow, ic, ni, nj]] = flat[fi];
                                }
                                fi += 1;
                            }
                        }
                    }
                }
            }

            // Certified coefficient error (#patches-coeff-err-soundness,
            // docs/PATCHES_7D_COEFF_ERR_CLOSURE.md §5). The conv-transpose
            // composition is position-preserving (output logical row `i` composes
            // incoming row `i` only), so with K = out_c·prev_kh·prev_kw
            // (over-bounds the per-cell contraction) and gain ‖kernel‖₁ = Σ|kernel|:
            //   6D (logical row = output position):
            //     intrinsic[i] =
            //       min(γ_K^f32·RowMaxAbs(incoming@i)·‖k‖₁ + U_FTZ/DAZ,
            //           measured_gap[i] + γ_K^f64·RowMaxAbs(incoming@i)·‖k‖₁ + U_FTZ/DAZ)
            //     new_err[i] = publish_normal_up(intrinsic[i] + ‖k‖₁·old_err[i]).
            //   7D explicit rows (err index = SPEC row = axis 0, length row_count,
            //   spec I1): one scalar must cover every coefficient of the row, so
            //   the magnitude max runs over the WHOLE spec-row slab (all
            //   positions — max-lift; old_err[row] is row-constant so the lift is
            //   exact):
            //     new_err[row] =
            //       publish_normal_up(γ_K^f32·RowMaxAbs7D(incoming@row)·‖k‖₁
            //                         + U_FTZ/DAZ + ‖k‖₁·old_err[row]).
            // Here U_FTZ/DAZ = 4K·FLT_MIN
            //                    + (RowL1(incoming) + ‖k‖₁)·FLT_MIN.
            //   Emitted Some even for old_err None: the intrinsic f32 contraction
            //   rounding is real on both layouts.
            // The hoisted ingredients below are computed in the exact order the
            // 6D arm always used, so the 6D arm stays bit-identical (pinned by
            // `dense_6d_compose_and_bias_err_bit_identical`).
            let kernel_l1: f64 = if deadline.is_some() {
                let mut sum = 0.0f64;
                for (offset, &value) in p.kernel.iter().enumerate() {
                    if offset.is_multiple_of(4_096) {
                        check_patches_deadline(deadline, "during coefficient-error kernel norm")?;
                    }
                    sum = add_f64_up(sum, f32_to_f64_exact_bits(value).abs());
                }
                sum
            } else {
                p.kernel.iter().fold(0.0, |sum, &value| {
                    add_f64_up(sum, f32_to_f64_exact_bits(value).abs())
                })
            };
            // #patches-perchannel-l1: the error factor is a PER-OUTPUT-CHANNEL
            // kernel norm, not the whole-kernel norm.
            //
            // The compose is
            //   new[pos, ic, ih, iw] = sum_{oc, gy, gx, ki, kj} incoming[pos, oc, gy, gx]
            //                                                   * kernel[oc, ic_local, ki, kj]
            // so an output element in channel `ic` only ever touches kernel entries
            // with THAT `ic_local`, and only `oc` inside `ic`'s group. Both the
            // intrinsic term and the incoming-error carry therefore scale with
            //   max over (g, ic_local) of  sum_{oc in g, ki, kj} |kernel[oc, ic_local, ki, kj]|
            // whereas `kernel_l1` above sums over EVERY `ic_local` as well —
            // over-charging by a factor of ~in_c_per_group at EVERY layer, which
            // compounds geometrically through the carry (in_c ~ 16-64 over ~9
            // convs on a CIFAR100 ResNet).
            //
            // Strictly tighter and still an upper bound, so it is sound; falls back
            // to the whole-kernel norm if the shape is not the expected 4D.
            let kernel_l1_per_channel: f64 = {
                let ks = p.kernel.shape();
                if ks.len() == 4 && p.groups > 0 && ks[0].is_multiple_of(p.groups) {
                    let (out_c, in_c_per_group, kh_k, kw_k) = (ks[0], ks[1], ks[2], ks[3]);
                    let out_c_per_group = out_c / p.groups;
                    let mut worst = 0.0f64;
                    for g in 0..p.groups {
                        for ic_local in 0..in_c_per_group {
                            let mut acc = 0.0f64;
                            for oc_local in 0..out_c_per_group {
                                let oc = g * out_c_per_group + oc_local;
                                for ki in 0..kh_k {
                                    for kj in 0..kw_k {
                                        acc = add_f64_up(
                                            acc,
                                            f32_to_f64_exact_bits(p.kernel[[oc, ic_local, ki, kj]])
                                                .abs(),
                                        );
                                    }
                                }
                            }
                            if acc > worst {
                                worst = acc;
                            }
                        }
                    }
                    check_patches_deadline(deadline, "during per-channel kernel norm")?;
                    worst
                } else {
                    kernel_l1
                }
            };
            let kernel_l1 = kernel_l1_per_channel.min(kernel_l1);

            let k_contraction =
                checked_shape_product(&[p.out_c, prev_kh, prev_kw]).ok_or_else(|| {
                    NyError::InvalidSpec("Conv2d Patches contraction size overflow".into())
                })?;
            let gamma = crate::layers::linear::crown_single_gamma_n_f32(k_contraction);

            // #patches-f64-err: a-posteriori intrinsic error via an f64 reference
            // compose.
            //
            // The relative part of the a-priori charge is
            // `gamma_K^f32 * rowmax * ||k||_1`, i.e. the Higham worst case for K
            // f32 accumulations with NO cancellation. It is enormously
            // conservative, and it compounds: measured with
            // `NY_CROWN_GAIN=1` on CIFAR100_resnet_medium, the CROWN box this
            // channel produces is up to 2.6e7x WIDER than the IBP box it exists
            // to tighten (Conv_11: ibp 5.4e3 vs crown 1.4e11), so every target it
            // touches is thrown away by the IBP intersection.
            //
            // The value path is UNCHANGED — still the same f32 compose, so every
            // bit-identity pin (incl. `dense_6d_compose_and_bias_err_bit_identical`)
            // still holds. What changes is only the certified error: recompute the
            // same compose in f64 (`conv2d_transpose_grouped_into_f64`, exact
            // f32->f64 widening so only the f64 accumulation rounds) and MEASURE
            // `|f32 - f64|` per position. The true value lies within
            // `gamma_K^f64 * rowmax * ||k||_1` of the f64 result. Both candidates
            // additionally include the same absolute `U_FTZ/DAZ` source/result
            // flushing charge before the minimum is taken, so they remain valid
            // without assuming gradual underflow:
            //     min(gamma_K^f32 * rowmax * ||k||_1 + U_FTZ/DAZ,
            //         measured_gap + gamma_K^f64 * rowmax * ||k||_1 + U_FTZ/DAZ).
            // The minimum is therefore sound under either gradual or flushed
            // underflow.
            //
            // Scoped to the 6D dense layout: that is the layout the cifar100 /
            // tinyimagenet CROWN-IBP conv targets use. The 7D explicit-rows arm
            // keeps the a-priori charge (still sound, just not tightened).
            let gamma_f64 = crate::layers::linear::crown_single_gamma_n_f64(k_contraction);
            let measured_gap: Option<Vec<f32>> = if explicit_rows {
                None
            } else {
                patches_f64_reference_gap(
                    &new_patches,
                    incoming,
                    p,
                    (prev_kh, prev_kw),
                    (new_kh, new_kw),
                    num_positions,
                    patch_volume,
                    &decode,
                    deadline,
                )?
            };
            let old = patches_data.coeff_err.as_ref();
            let coeff_err = if explicit_rows {
                // Hard length check (spec I6): a carried err that does not index
                // by spec row is a construction bug; error out so the caller
                // falls back to the sound dense path — never a silent
                // `.get(i).unwrap_or(0.0)` under-count.
                if let Some(e) = old {
                    if e.len() != row_count {
                        return Err(NyError::ShapeMismatch {
                            expected: vec![row_count],
                            got: vec![e.len()],
                        });
                    }
                }
                // Each ne[row] depends only on incoming spec-row slab `row` and
                // old_err[row] (a per-row max plus one fused expression — no
                // accumulation across rows), so rows compute in parallel with no
                // summation-order change (spec I8: rows only; max is
                // order-independent).
                let row_err = |row: usize| -> Result<f32> {
                    let mut rowmax = 0.0f64;
                    let mut row_l1 = 0.0f64;
                    for (offset, &v) in incoming.index_axis(Axis(0), row).iter().enumerate() {
                        if deadline.is_some() && offset.is_multiple_of(4_096) {
                            check_patches_deadline(
                                deadline,
                                "during explicit-row coefficient-error propagation",
                            )?;
                        }
                        let a = f32_to_f64_exact_bits(v).abs();
                        if a > rowmax {
                            rowmax = a;
                        }
                        row_l1 = add_f64_up(row_l1, a);
                    }
                    // Sanitize the carried err (spec I5): non-finite or negative
                    // maps to +INF (poisons outward; the row degrades at
                    // consumption) — NEVER NaN -> 0.
                    let oe = match old {
                        None => 0.0f64,
                        Some(e) => nonnegative_f32_error_or_infinity(e[row]),
                    };
                    // Exact-zero short-circuits BEFORE multiplying possibly
                    // infinite factors (spec I5 + §14 C2 clamp): rowmax == 0 ⇒
                    // pure carry term (γ_K may be +INF at pathological K, and
                    // INF·0 = NaN must never be emitted); kernel_l1 == 0 ⇒ every
                    // composed product is exactly ±0 and the carried deviation
                    // is scaled by Σ|w| = 0, so both terms are exactly 0.
                    let relative = if rowmax == 0.0 || kernel_l1 == 0.0 {
                        0.0
                    } else {
                        mul_f64_up(mul_f64_up(gamma, rowmax), kernel_l1)
                    };
                    let underflow = if rowmax == 0.0 || kernel_l1 == 0.0 {
                        0.0
                    } else {
                        patches_f32_underflow_charge(row_l1, kernel_l1, k_contraction)
                    };
                    let intrinsic = add_f64_up(relative, underflow);
                    let carry = if oe == 0.0 || kernel_l1 == 0.0 {
                        0.0
                    } else {
                        mul_f64_up(kernel_l1, oe)
                    };
                    // Both addends are finite >= 0 or +INF — never NaN. f64
                    // evaluation, one outward next_up at the f32 cast (spec I4).
                    Ok(publish_error_up_normal(add_f64_up(intrinsic, carry)))
                };
                let mut ne = Array1::<f32>::zeros(row_count);
                // #patches-deadline-kernel: fill the per-row certified error in
                // PARALLEL whether or not a deadline is present. `row_err` polls
                // the deadline itself every 4096 taps and returns
                // `DeadlineExceeded`, which `try_for_each` propagates, so the
                // parallel arm already honours the authority with overshoot
                // bounded by one row. Gating on the deadline's PRESENCE just made
                // every collector-routed target (which always carries one) take
                // the serial scan — the same anti-pattern already corrected for
                // the engine admission and the mask sweep in this file. Rows are
                // independent (a per-row max plus one fused expression), so this
                // is bit-identical, as the comment above already notes.
                if let Some(ne_slice) = ne.as_slice_mut() {
                    ne_slice.par_iter_mut().enumerate().try_for_each(
                        |(row, out)| -> Result<()> {
                            *out = row_err(row)?;
                            Ok(())
                        },
                    )?;
                } else {
                    for row in 0..row_count {
                        ne[row] = row_err(row)?;
                    }
                }
                Some(ne)
            } else {
                // The 6D error channel is indexed by logical spec position.
                // Reject malformed metadata instead of silently treating a
                // missing row as exact.
                if let Some(e) = old {
                    if e.len() != num_positions {
                        return Err(NyError::ShapeMismatch {
                            expected: vec![num_positions],
                            got: vec![e.len()],
                        });
                    }
                }
                // Each ne[idx] depends only on incoming row idx and old_err[idx]
                // (a per-row max plus one fused expression — no accumulation across
                // rows), so rows compute in parallel with no summation-order change.
                let row_err = |idx: usize| -> Result<f32> {
                    let (_row, soc, soh, sow) = decode(idx);
                    let mut rowmax = 0.0f64;
                    let mut row_l1 = 0.0f64;
                    let mut offset = 0usize;
                    for c in 0..p.out_c {
                        for ki in 0..prev_kh {
                            for kj in 0..prev_kw {
                                if deadline.is_some() && offset.is_multiple_of(4_096) {
                                    check_patches_deadline(
                                        deadline,
                                        "during coefficient-error propagation",
                                    )?;
                                }
                                offset += 1;
                                let a = f32_to_f64_exact_bits(incoming[[soc, soh, sow, c, ki, kj]])
                                    .abs();
                                if a > rowmax {
                                    rowmax = a;
                                }
                                row_l1 = add_f64_up(row_l1, a);
                            }
                        }
                    }
                    let oe = match old {
                        None => 0.0,
                        Some(e) => nonnegative_f32_error_or_infinity(e[idx]),
                    };
                    // A-priori Higham charge, and the a-posteriori measured one;
                    // both bound |f32 - exact|, so take the tighter (#patches-f64-err).
                    let apriori_relative = if rowmax == 0.0 || kernel_l1 == 0.0 {
                        0.0
                    } else {
                        mul_f64_up(mul_f64_up(gamma, rowmax), kernel_l1)
                    };
                    let underflow = if rowmax == 0.0 || kernel_l1 == 0.0 {
                        0.0
                    } else {
                        patches_f32_underflow_charge(row_l1, kernel_l1, k_contraction)
                    };
                    let apriori = add_f64_up(apriori_relative, underflow);
                    let intrinsic = match measured_gap.as_ref().and_then(|g| g.get(idx)) {
                        Some(&gap) if gap.is_finite() => {
                            let rounding = if rowmax == 0.0 || kernel_l1 == 0.0 {
                                0.0
                            } else {
                                mul_f64_up(mul_f64_up(gamma_f64, rowmax), kernel_l1)
                            };
                            // Charge FTZ/DAZ on both candidates before taking
                            // the minimum. The measured reference does not grant
                            // the production GEMM a gradual-underflow contract.
                            let measured = add_f64_up(
                                add_f64_up(f32_to_f64_exact_bits(gap), rounding),
                                underflow,
                            );
                            apriori.min(measured)
                        }
                        _ => apriori,
                    };
                    let carry = if oe == 0.0 || kernel_l1 == 0.0 {
                        0.0
                    } else {
                        mul_f64_up(kernel_l1, oe)
                    };
                    // #err-split probe (NY_ERR_SPLIT=1): report how the emitted
                    // error divides between the measured intrinsic term and the
                    // propagated carry, so the next tightening targets whichever
                    // actually dominates.
                    if idx == 0 && std::env::var("NY_ERR_SPLIT").ok().as_deref() == Some("1") {
                        eprintln!(
                            "[err-split] intrinsic={intrinsic:.6e} carry={carry:.6e} oe={oe:.6e} \
                             kL1={kernel_l1:.6e} rowmax={rowmax:.6e} carry_frac={:.4}",
                            if intrinsic + carry > 0.0 {
                                carry / (intrinsic + carry)
                            } else {
                                0.0
                            }
                        );
                    }
                    Ok(publish_error_up_normal(add_f64_up(intrinsic, carry)))
                };
                let mut ne = Array1::<f32>::zeros(num_positions);
                // #patches-deadline-kernel: fill the per-row certified error in
                // PARALLEL whether or not a deadline is present. `row_err` polls
                // the deadline itself every 4096 taps and returns
                // `DeadlineExceeded`, which `try_for_each` propagates, so the
                // parallel arm already honours the authority with overshoot
                // bounded by one row. Gating on the deadline's PRESENCE just made
                // every collector-routed target (which always carries one) take
                // the serial scan — the same anti-pattern already corrected for
                // the engine admission and the mask sweep in this file. Rows are
                // independent (a per-row max plus one fused expression), so this
                // is bit-identical, as the comment above already notes.
                if let Some(ne_slice) = ne.as_slice_mut() {
                    ne_slice.par_iter_mut().enumerate().try_for_each(
                        |(idx, out)| -> Result<()> {
                            *out = row_err(idx)?;
                            Ok(())
                        },
                    )?;
                } else {
                    for idx in 0..num_positions {
                        ne[idx] = row_err(idx)?;
                    }
                }
                Some(ne)
            };

            check_patches_deadline(deadline, "after coefficient-error propagation")?;
            Ok(PatchesData {
                coeff_err,
                patches: Some(new_patches),
                geometry: PatchGeometry::affine(new_stride, new_padding),
                identity: false,
                output_shape: patches_data.output_shape,
                input_shape: (p.in_c, p.in_h, p.in_w),
                unstable_idx: None,
            })
        }
    }

    /// Compute bias contribution for Patches backward.
    ///
    /// For conv bias b of shape [out_c], broadcast to [out_c, out_h, out_w]:
    /// new_b = old_b + sum over spatial positions of patches coefficients * bias
    //
    // Retained as the no-deadline compatibility face used by regression tests;
    // production propagation calls the deadline-aware implementation below.
    #[allow(dead_code)]
    fn compute_patches_bias(
        bounds: &PatchesLinearBounds,
        bias: &Array1<f32>,
        out_c: usize,
        out_h: usize,
        out_w: usize,
    ) -> Result<(Array1<f32>, Array1<f32>)> {
        Self::compute_patches_bias_with_deadline(bounds, bias, out_c, out_h, out_w, None)
    }

    fn compute_patches_bias_with_deadline(
        bounds: &PatchesLinearBounds,
        bias: &Array1<f32>,
        out_c: usize,
        out_h: usize,
        out_w: usize,
        deadline: Option<Instant>,
    ) -> Result<(Array1<f32>, Array1<f32>)> {
        check_patches_deadline(deadline, "before bias propagation")?;
        let lower_patches = &bounds.lower_a;
        let upper_patches = &bounds.upper_a;
        let old_lower_b = &bounds.lower_b;
        let old_upper_b = &bounds.upper_b;
        let out_dim = checked_shape_product(&[out_c, out_h, out_w]).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Conv2d bias patches: output dims product overflows: {out_c} * {out_h} * {out_w}"
            ))
        })?;

        if lower_patches.identity && upper_patches.identity {
            // Crash guard (mirrors BatchNorm patches fix 9a6bc1a): the per-output-neuron
            // bias indexing below (idx = oc*out_h*out_w + oh*out_w + ow) reads
            // old_lower_b[idx] for idx in 0..out_dim. Under disjunctive multi-clause
            // input-split the incoming bias is spec-row-shaped, not out_dim — a shorter
            // vector would index out of bounds (SIGABRT under panic=abort). Require the
            // exact per-neuron layout; otherwise return ShapeMismatch so the caller's
            // try_patches_or_dense_fallback drops to the sound dense Conv2d backward.
            if old_lower_b.len() != out_dim || old_upper_b.len() != out_dim {
                return Err(NyError::ShapeMismatch {
                    expected: vec![out_dim],
                    got: vec![old_lower_b.len().min(old_upper_b.len())],
                });
            }
            let mut new_lower_b = Array1::<f32>::zeros(out_dim);
            let mut new_upper_b = Array1::<f32>::zeros(out_dim);
            for oc in 0..out_c {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let idx = oc * out_h * out_w + oh * out_w + ow;
                        if idx.is_multiple_of(64) {
                            check_patches_deadline(deadline, "during identity bias propagation")?;
                        }
                        let lb_f64 = f32_to_f64_exact_bits(old_lower_b[idx])
                            + f32_to_f64_exact_bits(bias[oc]);
                        let ub_f64 = f32_to_f64_exact_bits(old_upper_b[idx])
                            + f32_to_f64_exact_bits(bias[oc]);
                        new_lower_b[idx] = publish_lower_bound_no_subnormal(lb_f64);
                        new_upper_b[idx] = publish_upper_bound_no_subnormal(ub_f64);
                    }
                }
            }
            check_patches_deadline(deadline, "after identity bias propagation")?;
            return Ok((new_lower_b, new_upper_b));
        }

        let (spec_oc, spec_oh, spec_ow) = lower_patches.output_shape;
        let mut new_lower_b = old_lower_b.clone();
        let mut new_upper_b = old_upper_b.clone();
        check_patches_deadline(deadline, "after bias allocation")?;

        let lower_p = lower_patches.patches.as_ref();
        let upper_p = upper_patches.patches.as_ref();
        let explicit_rows = lower_p
            .map(|p| p.ndim() == 7)
            .or_else(|| upper_p.map(|p| p.ndim() == 7))
            .unwrap_or(false);

        if explicit_rows {
            // Crash guard: this branch indexes new_bias[row] for row in 0..row_count
            // (new_*_b are clones of the incoming spec-row-shaped bias). A bias shorter
            // than row_count would index out of bounds; fall back to dense on mismatch.
            if new_lower_b.len() != bounds.row_count || new_upper_b.len() != bounds.row_count {
                return Err(NyError::ShapeMismatch {
                    expected: vec![bounds.row_count],
                    got: vec![new_lower_b.len().min(new_upper_b.len())],
                });
            }
            // Hardening (docs/PATCHES_7D_COEFF_ERR_CLOSURE.md §14 C3/C4): the
            // explicit-rows fold has no analog of the 6D identity-side
            // `else if c == soc` contribution, so an identity side here would
            // silently DROP its affine term — the wrong-affine verdict-bug
            // class. Hard error instead (believed unreachable:
            // from_dense_spatial_rows materializes both sides and the merge
            // family gate bars mixed pairs). Likewise a non-7D tensor on
            // either side would panic on the 7-index reads below (the
            // pre-existing mixed 6D/7D pair panic) — clean ShapeMismatch
            // instead. Both errors route the caller to the sound dense
            // fallback.
            let lp = lower_p.ok_or_else(|| {
                NyError::UnsupportedConfiguration(
                    "Conv2d Patches bias: identity lower side in the explicit-rows fold \
                     (its affine contribution has no 7D analog); use dense CROWN"
                        .to_string(),
                )
            })?;
            let up = upper_p.ok_or_else(|| {
                NyError::UnsupportedConfiguration(
                    "Conv2d Patches bias: identity upper side in the explicit-rows fold \
                     (its affine contribution has no 7D analog); use dense CROWN"
                        .to_string(),
                )
            })?;
            if lp.ndim() != 7 || up.ndim() != 7 {
                return Err(NyError::ShapeMismatch {
                    expected: vec![7, 7],
                    got: vec![lp.ndim(), up.ndim()],
                });
            }
            let prev_kh_l = lp.shape()[5];
            let prev_kw_l = lp.shape()[6];
            let prev_kh_u = up.shape()[5];
            let prev_kw_u = up.shape()[6];

            // Certified coefficient-error discharge into the bias
            // (#patches-coeff-err-soundness, HOLE2 — spec §5.1 2B with the §14
            // A1-adopted f64-summation discharge). The fold below sums, per
            // SPEC row, Σ_{pos,c,taps} stored_coeff·bias[c] over ALL
            // spec-output positions into the ONE spec-row bias slot, so:
            //   • carried-err widen: every one of the positions·out_c·kh·kw
            //     stored coefficients of the row deviates by ≤ old_err[row]
            //     ⇒ |fold_stored − fold_true| ≤
            //       old_err[row]·positions·(kh·kw)·Σ_c|bias[c]|  (SUM-lift —
            //     the `positions` factor is exactly what the 6D per-position
            //     formula does not have);
            //   • γ̄ fold-rounding discharge (§14 A1, closes C1's f64
            //     catastrophic-cancellation corner): the fold's own f64
            //     accumulation error is ≤ γ̄·ABS[row] with
            //     ABS[row] = |b_old[row]| + Σ|coeff·bias[c]| and
            //     γ̄ = γ_n^f64(8·row_volume + 16) — ≥ 4x headroom over the
            //     ≤ 2·row_volume+4 roundings on any addend's path, absorbing
            //     the read-only ABS accumulator's own f64 deficit and the
            //     final product/cast roundings (same argument as the
            //     activation-site discharge, spec §6.2). The (1+γ̄) factor on
            //     the widen absorbs the widen product's / ‖bias‖₁ sum's f64
            //     under-reads.
            // Both discharges land in the f64 accumulator BEFORE the directed
            // cast (spec I4) and are emitted even for err-free inputs (the
            // fold rounding is intrinsic). Per side independently.
            let bias_abs_l1: f64 = bias
                .iter()
                .map(|&value| f32_to_f64_exact_bits(value).abs())
                .sum();
            let lower_err = lower_patches.coeff_err.as_ref();
            let upper_err = upper_patches.coeff_err.as_ref();
            // Hard length checks (spec I6): direct [row] indexing below, never
            // a silent `.get(row).unwrap_or(0.0)` under-count.
            if let Some(e) = lower_err {
                if e.len() != bounds.row_count {
                    return Err(NyError::ShapeMismatch {
                        expected: vec![bounds.row_count],
                        got: vec![e.len()],
                    });
                }
            }
            if let Some(e) = upper_err {
                if e.len() != bounds.row_count {
                    return Err(NyError::ShapeMismatch {
                        expected: vec![bounds.row_count],
                        got: vec![e.len()],
                    });
                }
            }
            let positions_usize = spec_oc.saturating_mul(spec_oh).saturating_mul(spec_ow);
            let positions = positions_usize as f64;
            let lower_taps = (prev_kh_l as f64) * (prev_kw_l as f64);
            let upper_taps = (prev_kh_u as f64) * (prev_kw_u as f64);
            let gbar_of = |taps_kh: usize, taps_kw: usize| -> f64 {
                let row_volume = positions_usize
                    .saturating_mul(out_c)
                    .saturating_mul(taps_kh)
                    .saturating_mul(taps_kw);
                crate::layers::linear::crown_single_gamma_n_f64(
                    row_volume.saturating_mul(8).saturating_add(16),
                )
            };
            let gbar_l = gbar_of(prev_kh_l, prev_kw_l);
            let gbar_u = gbar_of(prev_kh_u, prev_kw_u);
            let flat_reductions = if patches_deadline_flat_bias_enabled() && deadline.is_some() {
                let started = Instant::now();
                let parallel_rows = !crate::imb::region_seq_inner();
                eprintln!(
                    "[patches-deadline-flat-bias] status=attempt rows={} positions={} \
                         out_c={} lower_taps={} upper_taps={} parallel_rows={}",
                    bounds.row_count,
                    positions_usize,
                    out_c,
                    prev_kh_l.saturating_mul(prev_kw_l),
                    prev_kh_u.saturating_mul(prev_kw_u),
                    parallel_rows,
                );
                match (lp.as_slice(), up.as_slice()) {
                    (Some(lower), Some(upper)) => {
                        let poll = || {
                            check_patches_deadline(deadline, "during flat explicit-row bias fold")
                        };
                        match reduce_explicit_row_bias_flat_with_poll(
                            lower,
                            upper,
                            bounds.row_count,
                            positions_usize,
                            out_c,
                            prev_kh_l.saturating_mul(prev_kw_l),
                            prev_kh_u.saturating_mul(prev_kw_u),
                            bias.as_slice().ok_or_else(|| {
                                NyError::UnsupportedConfiguration(
                                    "Conv2d Patches flat bias requires contiguous bias".into(),
                                )
                            })?,
                            new_lower_b.as_slice().ok_or_else(|| {
                                NyError::UnsupportedConfiguration(
                                    "Conv2d Patches flat bias requires contiguous lower bias"
                                        .into(),
                                )
                            })?,
                            new_upper_b.as_slice().ok_or_else(|| {
                                NyError::UnsupportedConfiguration(
                                    "Conv2d Patches flat bias requires contiguous upper bias"
                                        .into(),
                                )
                            })?,
                            parallel_rows,
                            &poll,
                        ) {
                            Ok(reductions) => {
                                eprintln!(
                                    "[patches-deadline-flat-bias] status=accepted rows={} \
                                         elapsed_us={}",
                                    reductions.len(),
                                    started.elapsed().as_micros(),
                                );
                                Some(reductions)
                            }
                            Err(error) => {
                                eprintln!(
                                    "[patches-deadline-flat-bias] status=discarded rows={} \
                                         elapsed_us={} reason={}",
                                    bounds.row_count,
                                    started.elapsed().as_micros(),
                                    error,
                                );
                                return Err(error);
                            }
                        }
                    }
                    _ => {
                        eprintln!(
                            "[patches-deadline-flat-bias] status=refused rows={} \
                                 elapsed_us={} reason=noncontiguous",
                            bounds.row_count,
                            started.elapsed().as_micros(),
                        );
                        None
                    }
                }
            } else {
                None
            };
            for row in 0..bounds.row_count {
                check_patches_deadline(deadline, "during explicit-row bias propagation")?;
                let ExplicitRowBiasReduction {
                    lower_sum,
                    upper_sum,
                    lower_abs,
                    upper_abs,
                } = if let Some(reductions) = flat_reductions.as_ref() {
                    reductions[row]
                } else {
                    let mut lower_sum = 0.0f64;
                    let mut upper_sum = 0.0f64;
                    // Read-only |·| mirrors of the fold, seeded with
                    // |b_old[row]| (the final b_old + sum addition is part of
                    // the certified chain). The VALUE accumulation statements
                    // and order below are unchanged (spec I3).
                    let mut lower_abs = f32_to_f64_exact_bits(new_lower_b[row]).abs();
                    let mut upper_abs = f32_to_f64_exact_bits(new_upper_b[row]).abs();
                    let mut ops_since_poll = 0usize;

                    for soc in 0..spec_oc {
                        for soh in 0..spec_oh {
                            for sow in 0..spec_ow {
                                for c in 0..out_c {
                                    let mut lc_sum = 0.0f64;
                                    let mut uc_sum = 0.0f64;
                                    let mut lc_abs = 0.0f64;
                                    let mut uc_abs = 0.0f64;

                                    for ki in 0..prev_kh_l {
                                        for kj in 0..prev_kw_l {
                                            if deadline.is_some() {
                                                ops_since_poll += 1;
                                                if ops_since_poll >= 4_096 {
                                                    check_patches_deadline(
                                                        deadline,
                                                        "during explicit-row lower bias fold",
                                                    )?;
                                                    ops_since_poll = 0;
                                                }
                                            }
                                            let a = f32_to_f64_exact_bits(
                                                lp[[row, soc, soh, sow, c, ki, kj]],
                                            );
                                            lc_sum += a;
                                            lc_abs += a.abs();
                                        }
                                    }

                                    for ki in 0..prev_kh_u {
                                        for kj in 0..prev_kw_u {
                                            if deadline.is_some() {
                                                ops_since_poll += 1;
                                                if ops_since_poll >= 4_096 {
                                                    check_patches_deadline(
                                                        deadline,
                                                        "during explicit-row upper bias fold",
                                                    )?;
                                                    ops_since_poll = 0;
                                                }
                                            }
                                            let a = f32_to_f64_exact_bits(
                                                up[[row, soc, soh, sow, c, ki, kj]],
                                            );
                                            uc_sum += a;
                                            uc_abs += a.abs();
                                        }
                                    }

                                    let bias_c = f32_to_f64_exact_bits(bias[c]);
                                    lower_sum += lc_sum * bias_c;
                                    upper_sum += uc_sum * bias_c;
                                    let bc_abs = bias_c.abs();
                                    lower_abs += lc_abs * bc_abs;
                                    upper_abs += uc_abs * bc_abs;
                                }
                            }
                        }
                    }
                    ExplicitRowBiasReduction {
                        lower_sum,
                        upper_sum,
                        lower_abs,
                        upper_abs,
                    }
                };

                // Per-side outward discharge
                //   D = γ̄·ABS[row] + old_err[row]·positions·taps·‖bias‖₁·(1+γ̄),
                // with the spec-I5 sanitize (non-finite/negative err ⇒ +INF
                // poison) and exact-zero short-circuits BEFORE multiplying
                // possibly infinite factors (0·INF = NaN must never appear):
                // ABS == 0 ⇒ every folded product is exactly 0 ⇒ the fold is
                // exact; widen_base == 0 ⇒ no folded tap can deviate (zero
                // taps or an all-zero bias) ⇒ zero widen is exact.
                let side_discharge =
                    |err: Option<&Array1<f32>>, taps: f64, gbar: f64, abs_sum: f64| -> f64 {
                        let oe = match err {
                            None => 0.0f64,
                            Some(e) => nonnegative_f32_error_or_infinity(e[row]),
                        };
                        let fold_disc = if abs_sum == 0.0 { 0.0 } else { gbar * abs_sum };
                        let widen_base = positions * taps * bias_abs_l1;
                        let widen = if oe == 0.0 || widen_base == 0.0 {
                            0.0
                        } else {
                            (oe * widen_base) * (1.0 + gbar)
                        };
                        fold_disc + widen
                    };
                let dl = side_discharge(lower_err, lower_taps, gbar_l, lower_abs);
                let du = side_discharge(upper_err, upper_taps, gbar_u, upper_abs);

                // Discharge lands in the f64 accumulator BEFORE the directed
                // cast (spec I4); a non-finite discharge poisons the row
                // outward to ∓INF (vacuous certificate, never NaN — spec I5).
                new_lower_b[row] = if dl.is_finite() {
                    publish_lower_bound_no_subnormal(
                        f32_to_f64_exact_bits(new_lower_b[row]) + lower_sum - dl,
                    )
                } else {
                    f32::NEG_INFINITY
                };
                new_upper_b[row] = if du.is_finite() {
                    publish_upper_bound_no_subnormal(
                        f32_to_f64_exact_bits(new_upper_b[row]) + upper_sum + du,
                    )
                } else {
                    f32::INFINITY
                };
            }
        } else {
            let spec_dim = spec_oc * spec_oh * spec_ow;
            // Crash guard: this branch indexes new_bias[j] for j in 0..spec_dim. A bias
            // vector shorter than spec_dim would index out of bounds; fall back to dense.
            if new_lower_b.len() != spec_dim || new_upper_b.len() != spec_dim {
                return Err(NyError::ShapeMismatch {
                    expected: vec![spec_dim],
                    got: vec![new_lower_b.len().min(new_upper_b.len())],
                });
            }
            // Certified coefficient-error discharge into the bias
            // (#patches-coeff-err-soundness, HOLE2). Accumulate each
            // stored_coeff*bias term with a directed binary64 add. A single
            // outward binary32 cast cannot repair a residual already lost to
            // binary64 cancellation. The error widen is accumulated once per
            // stored tap with upward binary64 rounding, avoiding an
            // under-rounded taps*sum(|bias|) reduction.
            let lower_err = lower_patches.coeff_err.as_ref();
            let upper_err = upper_patches.coeff_err.as_ref();
            if let Some(e) = lower_err {
                if e.len() != spec_dim {
                    return Err(NyError::ShapeMismatch {
                        expected: vec![spec_dim],
                        got: vec![e.len()],
                    });
                }
            }
            if let Some(e) = upper_err {
                if e.len() != spec_dim {
                    return Err(NyError::ShapeMismatch {
                        expected: vec![spec_dim],
                        got: vec![e.len()],
                    });
                }
            }
            for j in 0..spec_dim {
                if j.is_multiple_of(32) {
                    check_patches_deadline(deadline, "during bias propagation")?;
                }
                let soc = j / (spec_oh * spec_ow);
                let rem = j % (spec_oh * spec_ow);
                let soh = rem / spec_ow;
                let sow = rem % spec_ow;

                let mut lower_sum = 0.0f64;
                let mut upper_sum = 0.0f64;
                let mut lower_widen = 0.0f64;
                let mut upper_widen = 0.0f64;
                let mut ops_since_poll = 0usize;

                let lower_err_value = match lower_err {
                    None => 0.0,
                    Some(e) => nonnegative_f32_error_or_infinity(e[j]),
                };
                let upper_err_value = match upper_err {
                    None => 0.0,
                    Some(e) => nonnegative_f32_error_or_infinity(e[j]),
                };

                for c in 0..out_c {
                    let bias_c = f32_to_f64_exact_bits(bias[c]);
                    let bias_abs = bias_c.abs();
                    if let Some(lp) = lower_p {
                        let prev_kh = lp.shape()[4];
                        let prev_kw = lp.shape()[5];
                        for ki in 0..prev_kh {
                            for kj in 0..prev_kw {
                                if deadline.is_some() {
                                    ops_since_poll += 1;
                                    if ops_since_poll >= 4_096 {
                                        check_patches_deadline(deadline, "during lower bias fold")?;
                                        ops_since_poll = 0;
                                    }
                                }
                                let term =
                                    f32_to_f64_exact_bits(lp[[soc, soh, sow, c, ki, kj]]) * bias_c;
                                lower_sum = add_f64_down(lower_sum, term);
                                if lower_err_value != 0.0 && bias_abs != 0.0 {
                                    lower_widen =
                                        add_f64_up(lower_widen, lower_err_value * bias_abs);
                                }
                            }
                        }
                    } else if c == soc {
                        lower_sum = add_f64_down(lower_sum, bias_c);
                    }

                    if let Some(up) = upper_p {
                        let prev_kh = up.shape()[4];
                        let prev_kw = up.shape()[5];
                        for ki in 0..prev_kh {
                            for kj in 0..prev_kw {
                                if deadline.is_some() {
                                    ops_since_poll += 1;
                                    if ops_since_poll >= 4_096 {
                                        check_patches_deadline(deadline, "during upper bias fold")?;
                                        ops_since_poll = 0;
                                    }
                                }
                                let term =
                                    f32_to_f64_exact_bits(up[[soc, soh, sow, c, ki, kj]]) * bias_c;
                                upper_sum = add_f64_up(upper_sum, term);
                                if upper_err_value != 0.0 && bias_abs != 0.0 {
                                    upper_widen =
                                        add_f64_up(upper_widen, upper_err_value * bias_abs);
                                }
                            }
                        }
                    } else if c == soc {
                        upper_sum = add_f64_up(upper_sum, bias_c);
                    }
                }

                let lower_total = add_f64_down(f32_to_f64_exact_bits(new_lower_b[j]), lower_sum);
                let lower_total = add_f64_down(lower_total, -lower_widen);
                let upper_total = add_f64_up(f32_to_f64_exact_bits(new_upper_b[j]), upper_sum);
                let upper_total = add_f64_up(upper_total, upper_widen);
                new_lower_b[j] = publish_lower_bound_no_subnormal(lower_total);
                new_upper_b[j] = publish_upper_bound_no_subnormal(upper_total);
            }
        }

        check_patches_deadline(deadline, "after bias propagation")?;
        Ok((new_lower_b, new_upper_b))
    }
}

// =====================================================================
// Byte-identity pin tests for the certified conv coeff_err channel
// (#patches-coeff-err-soundness; 7D explicit-rows closure spec §5.4 T4,
// docs/PATCHES_7D_COEFF_ERR_CLOSURE.md).
//
// These pin the certified 6D compose-error formula and directed 6D bias
// fold + coeff_err widen bit-for-bit against independent in-test replicas.
// =====================================================================
#[cfg(test)]
mod coeff_err_tests {
    use super::*;

    /// Deterministic non-dyadic mixed-sign fill with exact zeros sprinkled in.
    fn det_fill(n: usize, seed: u32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let k = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed);
                if k.is_multiple_of(11) {
                    0.0
                } else {
                    (((k >> 8) % 2000) as f32 - 1000.0) * 0.001_37
                }
            })
            .collect()
    }

    /// The a-posteriori channel may only TIGHTEN: assert the emitted certified
    /// error never exceeds the a-priori replica (#patches-f64-err).
    fn assert_never_worse(label: &str, actual: &[f32], apriori: &[f32]) {
        assert_eq!(actual.len(), apriori.len(), "{label}: length mismatch");
        for (i, (&a, &e)) in actual.iter().zip(apriori.iter()).enumerate() {
            assert!(
                a <= e || (a - e).abs() <= f32::EPSILON * e.abs().max(1.0),
                "{label}[{i}]: emitted err {a:?} EXCEEDS the a-priori bound {e:?} — \
                 the a-posteriori channel must only ever tighten"
            );
            assert!(
                a >= 0.0 && a.is_finite(),
                "{label}[{i}]: err {a:?} not a sane magnitude"
            );
        }
    }

    fn assert_bits_eq(label: &str, actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                e.to_bits(),
                "{label}[{i}]: actual {a:?} (bits {:#010x}) != replica {e:?} (bits {:#010x})",
                a.to_bits(),
                e.to_bits()
            );
        }
    }

    /// In-test replica of the 6D compose coeff_err rule
    /// (`conv2d_patches_backward`, non-identity 6D arm):
    ///   ne[idx] = next_up(γ_K^f32 · RowMaxAbs(incoming@idx) · ‖k‖₁ + ‖k‖₁ · old_err[idx])
    fn replica_compose_err_6d(
        incoming: &ArrayD<f32>,
        old: Option<&Array1<f32>>,
        kernel: &ArrayD<f32>,
        out_c: usize,
        prev_kh: usize,
        prev_kw: usize,
        spec: (usize, usize, usize),
    ) -> Array1<f32> {
        let (spec_oc, spec_oh, spec_ow) = spec;
        let num_positions = spec_oc * spec_oh * spec_ow;
        let kernel_l1: f64 = kernel.iter().map(|v| f64::from(*v).abs()).sum();
        let k_contraction = out_c.saturating_mul(prev_kh).saturating_mul(prev_kw);
        let gamma = crate::layers::linear::crown_single_gamma_n_f32(k_contraction);
        let mut ne = Array1::<f32>::zeros(num_positions);
        for idx in 0..num_positions {
            let position_idx = idx % (spec_oc * spec_oh * spec_ow);
            let soc = position_idx / (spec_oh * spec_ow);
            let rem = position_idx % (spec_oh * spec_ow);
            let soh = rem / spec_ow;
            let sow = rem % spec_ow;
            let mut rowmax = 0.0f64;
            for c in 0..out_c {
                for ki in 0..prev_kh {
                    for kj in 0..prev_kw {
                        let a = f64::from(incoming[[soc, soh, sow, c, ki, kj]]).abs();
                        if a > rowmax {
                            rowmax = a;
                        }
                    }
                }
            }
            let oe = old.map_or(0.0, |e| {
                let value = e[idx];
                if value.is_finite() && value >= 0.0 {
                    f64::from(value)
                } else {
                    f64::INFINITY
                }
            });
            let intrinsic = if rowmax == 0.0 || kernel_l1 == 0.0 {
                0.0
            } else {
                gamma * rowmax * kernel_l1
            };
            let carry = if oe == 0.0 || kernel_l1 == 0.0 {
                0.0
            } else {
                kernel_l1 * oe
            };
            ne[idx] = ny_tensor::next_up_f32(add_f64_up(intrinsic, carry) as f32);
        }
        ne
    }

    /// Independent replica of the directed 6D (non-identity, dense-layout)
    /// branch of `compute_patches_bias`.
    fn replica_bias_6d(
        bounds: &PatchesLinearBounds,
        bias: &Array1<f32>,
        out_c: usize,
    ) -> (Array1<f32>, Array1<f32>) {
        use ny_tensor::{next_down_f32, next_up_f32};

        let lower_patches = &bounds.lower_a;
        let upper_patches = &bounds.upper_a;
        let (spec_oc, spec_oh, spec_ow) = lower_patches.output_shape;
        let mut new_lower_b = bounds.lower_b.clone();
        let mut new_upper_b = bounds.upper_b.clone();
        let lower_p = lower_patches.patches.as_ref();
        let upper_p = upper_patches.patches.as_ref();
        let spec_dim = spec_oc * spec_oh * spec_ow;

        let lower_err = lower_patches.coeff_err.as_ref();
        let upper_err = upper_patches.coeff_err.as_ref();
        for j in 0..spec_dim {
            let soc = j / (spec_oh * spec_ow);
            let rem = j % (spec_oh * spec_ow);
            let soh = rem / spec_ow;
            let sow = rem % spec_ow;

            let mut lower_sum = 0.0f64;
            let mut upper_sum = 0.0f64;
            let mut lower_widen = 0.0f64;
            let mut upper_widen = 0.0f64;
            let lower_err_value = lower_err.map_or(0.0, |e| {
                let value = e[j];
                if value.is_finite() && value >= 0.0 {
                    f64::from(value)
                } else {
                    f64::INFINITY
                }
            });
            let upper_err_value = upper_err.map_or(0.0, |e| {
                let value = e[j];
                if value.is_finite() && value >= 0.0 {
                    f64::from(value)
                } else {
                    f64::INFINITY
                }
            });

            for c in 0..out_c {
                let bias_c = f64::from(bias[c]);
                let bias_abs = bias_c.abs();

                if let Some(lp) = lower_p {
                    let prev_kh = lp.shape()[4];
                    let prev_kw = lp.shape()[5];
                    for ki in 0..prev_kh {
                        for kj in 0..prev_kw {
                            lower_sum = add_f64_down(
                                lower_sum,
                                f64::from(lp[[soc, soh, sow, c, ki, kj]]) * bias_c,
                            );
                            if lower_err_value != 0.0 && bias_abs != 0.0 {
                                lower_widen = add_f64_up(lower_widen, lower_err_value * bias_abs);
                            }
                        }
                    }
                } else if c == soc {
                    lower_sum = add_f64_down(lower_sum, bias_c);
                }

                if let Some(up) = upper_p {
                    let prev_kh = up.shape()[4];
                    let prev_kw = up.shape()[5];
                    for ki in 0..prev_kh {
                        for kj in 0..prev_kw {
                            upper_sum = add_f64_up(
                                upper_sum,
                                f64::from(up[[soc, soh, sow, c, ki, kj]]) * bias_c,
                            );
                            if upper_err_value != 0.0 && bias_abs != 0.0 {
                                upper_widen = add_f64_up(upper_widen, upper_err_value * bias_abs);
                            }
                        }
                    }
                } else if c == soc {
                    upper_sum = add_f64_up(upper_sum, bias_c);
                }
            }

            let lower_total = add_f64_down(f64::from(new_lower_b[j]), lower_sum);
            let lower_total = add_f64_down(lower_total, -lower_widen);
            let upper_total = add_f64_up(f64::from(new_upper_b[j]), upper_sum);
            let upper_total = add_f64_up(upper_total, upper_widen);
            new_lower_b[j] = next_down_f32(lower_total as f32);
            new_upper_b[j] = next_up_f32(upper_total as f32);
        }
        (new_lower_b, new_upper_b)
    }

    fn make_patches_data(
        shape: &[usize],
        seed: u32,
        coeff_err: Option<Array1<f32>>,
        output_shape: (usize, usize, usize),
        input_shape: (usize, usize, usize),
    ) -> PatchesData {
        let n: usize = shape.iter().product();
        PatchesData {
            coeff_err,
            patches: Some(ArrayD::from_shape_vec(IxDyn(shape), det_fill(n, seed)).unwrap()),
            geometry: PatchGeometry::affine((1, 1), (0, 0, 0, 0)),
            identity: false,
            output_shape,
            input_shape,
            unstable_idx: None,
        }
    }

    struct FtzDazMockGemm;

    impl ny_core::GemmEngine for FtzDazMockGemm {
        fn gemm_f32(&self, m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
            let a_len = m
                .checked_mul(k)
                .ok_or_else(|| NyError::InvalidSpec("mock FTZ/DAZ GEMM A size overflow".into()))?;
            let b_len = k
                .checked_mul(n)
                .ok_or_else(|| NyError::InvalidSpec("mock FTZ/DAZ GEMM B size overflow".into()))?;
            let out_len = m.checked_mul(n).ok_or_else(|| {
                NyError::InvalidSpec("mock FTZ/DAZ GEMM output size overflow".into())
            })?;
            if a.len() != a_len || b.len() != b_len {
                return Err(NyError::ShapeMismatch {
                    expected: vec![a_len, b_len],
                    got: vec![a.len(), b.len()],
                });
            }

            // Flush every binary32 subnormal source (DAZ), product, and partial
            // sum (FTZ), while preserving the sign of zero.
            let flush = |value: f32| {
                let bits = value.to_bits();
                if bits & 0x7f80_0000 == 0 {
                    f32::from_bits(bits & 0x8000_0000)
                } else {
                    value
                }
            };
            let mut out = vec![0.0f32; out_len];
            for row in 0..m {
                for col in 0..n {
                    let mut acc = 0.0f32;
                    for inner in 0..k {
                        let lhs = flush(a[row * k + inner]);
                        let rhs = flush(b[inner * n + col]);
                        let product = flush(lhs * rhs);
                        acc = flush(flush(acc) + product);
                    }
                    out[row * n + col] = acc;
                }
            }
            Ok(out)
        }
    }

    fn scalar_params(kernel: &ArrayD<f32>, stride: (usize, usize)) -> Conv2dPatchesParams<'_> {
        Conv2dPatchesParams {
            kernel,
            in_c: 1,
            out_c: 1,
            groups: 1,
            kh: 1,
            kw: 1,
            sh: stride.0,
            sw: stride.1,
            ph: 0,
            pw: 0,
            in_h: 1,
            in_w: 1,
            out_h: 1,
            out_w: 1,
        }
    }

    #[test]
    fn patches_coeff_err_covers_ftz_daz_flushing_engine_6d_and_7d() {
        crate::tests::with_env_edits(|env| {
            env.set("NY_PATCHES_GPU", "1");
            let engine = FtzDazMockGemm;
            let min_subnormal = f32::from_bits(1);
            let cases = [
                ("DAZ input", min_subnormal, 2.0_f32.powi(100)),
                ("DAZ kernel", 2.0_f32.powi(100), min_subnormal),
                ("FTZ product", f32::MIN_POSITIVE, 0.5_f32),
            ];

            for (case, incoming_value, kernel_value) in cases {
                let kernel =
                    ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1]), vec![kernel_value]).unwrap();
                let params = scalar_params(&kernel, (1, 1));
                let exact = (f32_to_f64_exact_bits(incoming_value)
                    * f32_to_f64_exact_bits(kernel_value))
                .abs();
                assert!(exact > 0.0, "{case}: fixture must have a nonzero truth");

                for explicit_rows in [false, true] {
                    let shape: &[usize] = if explicit_rows {
                        &[1, 1, 1, 1, 1, 1, 1]
                    } else {
                        &[1, 1, 1, 1, 1, 1]
                    };
                    let patches =
                        ArrayD::from_shape_vec(IxDyn(shape), vec![incoming_value]).unwrap();
                    let pd = PatchesData {
                        coeff_err: None,
                        patches: Some(patches),
                        geometry: PatchGeometry::affine((1, 1), (0, 0, 0, 0)),
                        identity: false,
                        output_shape: (1, 1, 1),
                        input_shape: (1, 1, 1),
                        unstable_idx: None,
                    };
                    let result =
                        Conv2dLayer::conv2d_patches_backward(&pd, &params, Some(&engine)).unwrap();
                    let layout = if explicit_rows { "7D" } else { "6D" };
                    assert_eq!(
                        result.patches.as_ref().unwrap().as_slice().unwrap(),
                        &[0.0],
                        "{case} {layout}: mock backend must flush the true contribution"
                    );
                    let err = result.coeff_err.as_ref().expect("certificate")[0];
                    assert!(
                        err.is_finite() && err >= f32::MIN_POSITIVE,
                        "{case} {layout}: positive certificate must be normal, got {err:e}"
                    );
                    assert!(
                        f32_to_f64_exact_bits(err) >= exact,
                        "{case} {layout}: published err {err:e} excludes exact deviation {exact:e}"
                    );
                }
            }
        });
    }

    #[test]
    fn patch_compose_rejects_malformed_and_overflowing_geometry() {
        let kernel = ArrayD::<f32>::ones(IxDyn(&[1, 1, 1, 1]));
        let params = scalar_params(&kernel, (1, 1));
        let valid = make_patches_data(&[1, 1, 1, 1, 1, 1], 41, None, (1, 1, 1), (1, 1, 1));

        let mut rank_five = valid.clone();
        rank_five.patches = Some(ArrayD::zeros(IxDyn(&[1, 1, 1, 1, 1])));
        assert!(matches!(
            Conv2dLayer::conv2d_patches_backward(&rank_five, &params, None),
            Err(NyError::ShapeMismatch { .. })
        ));

        let mut zero_extent = valid.clone();
        zero_extent.patches = Some(ArrayD::zeros(IxDyn(&[1, 1, 1, 1, 0, 1])));
        assert!(matches!(
            Conv2dLayer::conv2d_patches_backward(&zero_extent, &params, None),
            Err(NyError::InvalidSpec(_))
        ));

        let prefix_mismatch =
            make_patches_data(&[2, 1, 1, 1, 1, 1], 42, None, (1, 1, 1), (1, 1, 1));
        assert!(matches!(
            Conv2dLayer::conv2d_patches_backward(&prefix_mismatch, &params, None),
            Err(NyError::ShapeMismatch { .. })
        ));

        let mut input_shape_mismatch = valid.clone();
        input_shape_mismatch.input_shape = (1, 2, 1);
        assert!(matches!(
            Conv2dLayer::conv2d_patches_backward(&input_shape_mismatch, &params, None),
            Err(NyError::ShapeMismatch { .. })
        ));

        let extent_overflow =
            make_patches_data(&[1, 1, 1, 1, 2, 1], 43, None, (1, 1, 1), (1, 1, 1));
        let max_stride_params = scalar_params(&kernel, (usize::MAX, 1));
        assert!(matches!(
            Conv2dLayer::conv2d_patches_backward(&extent_overflow, &max_stride_params, None,),
            Err(NyError::InvalidSpec(_))
        ));

        let mut stride_overflow = valid.clone();
        stride_overflow.geometry = PatchGeometry::affine((usize::MAX, 1), (0, 0, 0, 0));
        let stride_two_params = scalar_params(&kernel, (2, 1));
        assert!(matches!(
            Conv2dLayer::conv2d_patches_backward(&stride_overflow, &stride_two_params, None),
            Err(NyError::InvalidSpec(_))
        ));

        let mut padding_overflow = valid;
        padding_overflow.geometry = PatchGeometry::affine((1, 1), (usize::MAX, 0, 0, 0));
        let width_stride_two_params = scalar_params(&kernel, (1, 2));
        assert!(matches!(
            Conv2dLayer::conv2d_patches_backward(&padding_overflow, &width_stride_two_params, None,),
            Err(NyError::InvalidSpec(_))
        ));

        let identity = PatchesData {
            coeff_err: None,
            patches: None,
            geometry: PatchGeometry::affine((1, 1), (0, 0, 0, 0)),
            identity: true,
            output_shape: (1, 1, 1),
            input_shape: (1, 1, 1),
            unstable_idx: None,
        };
        let identity_overflow_params = Conv2dPatchesParams {
            out_h: usize::MAX,
            out_w: 2,
            ..scalar_params(&kernel, (1, 1))
        };
        assert!(matches!(
            Conv2dLayer::conv2d_patches_backward(&identity, &identity_overflow_params, None,),
            Err(NyError::InvalidSpec(_))
        ));
    }

    #[test]
    fn padded_patch_mask_rejects_coordinate_overflow() {
        let mut pd = PatchesData {
            coeff_err: None,
            patches: Some(ArrayD::ones(IxDyn(&[1, 3, 1, 1, 1, 1]))),
            geometry: PatchGeometry::affine((usize::MAX, 1), (0, 0, 1, 0)),
            identity: false,
            output_shape: (1, 3, 1),
            input_shape: (1, 1, 1),
            unstable_idx: None,
        };
        assert!(matches!(
            mask_out_of_range_intermediate_taps(&mut pd, None),
            Err(NyError::InvalidSpec(_))
        ));
    }

    /// Spec §5.4 T4 pin: 6D compose err and 6D bias fold+widen are bit-identical
    /// to in-test verbatim formula replicas, with err-carrying AND err-free
    /// inputs (the latter pins that the err channel never perturbs values).
    #[test]
    fn dense_6d_compose_and_bias_err_bit_identical() {
        // ---- fixture geometry ----
        // conv: kernel [out_c=3, in_c=2, kh=2, kw=2], stride 1, pad 0, in 3x3 -> out 2x2
        // incoming 6D patches: [spec_oc=2, spec_oh=2, spec_ow=2, out_c=3, prev_kh=2, prev_kw=2]
        let kernel = ArrayD::from_shape_vec(IxDyn(&[3, 2, 2, 2]), det_fill(24, 77)).unwrap();
        let params = Conv2dPatchesParams {
            kernel: &kernel,
            in_c: 2,
            out_c: 3,
            groups: 1,
            kh: 2,
            kw: 2,
            sh: 1,
            sw: 1,
            ph: 0,
            pw: 0,
            in_h: 3,
            in_w: 3,
            out_h: 2,
            out_w: 2,
        };
        let spec = (2usize, 2usize, 2usize);
        let spec_dim = spec.0 * spec.1 * spec.2;
        let lower_err = Array1::from_vec(vec![
            1.0e-3_f32, 0.0, 5.0e-4, 2.0e-6, 3.0e-3, 0.0, 7.0e-5, 1.0e-4,
        ]);
        let upper_err = Array1::from_vec(vec![
            2.0e-3_f32, 1.0e-4, 0.0, 5.0e-5, 4.0e-4, 0.0, 6.0e-4, 8.0e-4,
        ]);

        // ---- 2A: compose err, err-carrying input ----
        let pd = make_patches_data(
            &[2, 2, 2, 3, 2, 2],
            1,
            Some(lower_err.clone()),
            spec,
            (3, 2, 2),
        );
        let incoming = pd.patches.as_ref().unwrap().clone();
        let out = Conv2dLayer::conv2d_patches_backward(&pd, &params, None).unwrap();
        let ne = out
            .coeff_err
            .as_ref()
            .expect("6D compose must emit Some coeff_err");
        let expected = replica_compose_err_6d(&incoming, Some(&lower_err), &kernel, 3, 2, 2, spec);
        // #patches-f64-err: the emitted err is now `min(a-priori, measured)`, so
        // it is bit-identical to the replica only where the a-priori charge wins.
        // The invariant that must hold everywhere is that it NEVER EXCEEDS the
        // a-priori bound — tightening is the point, loosening would be the bug.
        assert_never_worse(
            "compose ne (err-carrying)",
            ne.as_slice().unwrap(),
            expected.as_slice().unwrap(),
        );
        // Sanity: the carried term is live (row 0 has nonzero old err).
        assert!(ne[0] > 0.0 && ne[0].is_finite());

        // ---- 2A: compose err, err-free input (oe = 0, intrinsic term only),
        // and the VALUE tensor must be bit-identical to the err-carrying run ----
        let pd_none = make_patches_data(&[2, 2, 2, 3, 2, 2], 1, None, spec, (3, 2, 2));
        let out_none = Conv2dLayer::conv2d_patches_backward(&pd_none, &params, None).unwrap();
        let ne_none = out_none
            .coeff_err
            .as_ref()
            .expect("6D compose must emit Some coeff_err for err-free input");
        let expected_none = replica_compose_err_6d(&incoming, None, &kernel, 3, 2, 2, spec);
        // #patches-f64-err: same invariant as the err-carrying case above. On this
        // fixture the measured channel is ~64x tighter than the a-priori charge
        // (2.34e-7 vs 1.51e-5).
        assert_never_worse(
            "compose ne (err-free)",
            ne_none.as_slice().unwrap(),
            expected_none.as_slice().unwrap(),
        );
        assert_bits_eq(
            "composed value tensor unchanged by err channel",
            out.patches.as_ref().unwrap().as_slice().unwrap(),
            out_none.patches.as_ref().unwrap().as_slice().unwrap(),
        );

        // ---- 2B: bias fold + widen, err-carrying on both sides ----
        let bias = Array1::from_vec(vec![0.3_f32, -0.7, 0.11]);
        let plb = PatchesLinearBounds {
            row_count: spec_dim,
            lower_a: make_patches_data(&[2, 2, 2, 3, 2, 2], 1, Some(lower_err), spec, (3, 2, 2)),
            lower_b: Array1::from_vec(det_fill(spec_dim, 900)),
            upper_a: make_patches_data(&[2, 2, 2, 3, 2, 2], 2, Some(upper_err), spec, (3, 2, 2)),
            upper_b: Array1::from_vec(det_fill(spec_dim, 901)),
        };
        let (nlb, nub) = Conv2dLayer::compute_patches_bias(&plb, &bias, 3, 2, 2).unwrap();
        let (rlb, rub) = replica_bias_6d(&plb, &bias, 3);
        assert_bits_eq(
            "bias lower (err-carrying)",
            nlb.as_slice().unwrap(),
            rlb.as_slice().unwrap(),
        );
        assert_bits_eq(
            "bias upper (err-carrying)",
            nub.as_slice().unwrap(),
            rub.as_slice().unwrap(),
        );

        // ---- 2B: bias fold, err-free (widen exactly 0 — the plain fold) ----
        let plb_none = PatchesLinearBounds {
            row_count: spec_dim,
            lower_a: make_patches_data(&[2, 2, 2, 3, 2, 2], 1, None, spec, (3, 2, 2)),
            lower_b: plb.lower_b.clone(),
            upper_a: make_patches_data(&[2, 2, 2, 3, 2, 2], 2, None, spec, (3, 2, 2)),
            upper_b: plb.upper_b,
        };
        let (nlb_none, nub_none) =
            Conv2dLayer::compute_patches_bias(&plb_none, &bias, 3, 2, 2).unwrap();
        let (rlb_none, rub_none) = replica_bias_6d(&plb_none, &bias, 3);
        assert_bits_eq(
            "bias lower (err-free)",
            nlb_none.as_slice().unwrap(),
            rlb_none.as_slice().unwrap(),
        );
        assert_bits_eq(
            "bias upper (err-free)",
            nub_none.as_slice().unwrap(),
            rub_none.as_slice().unwrap(),
        );
    }

    #[test]
    fn dense_6d_bias_cancellation_residual_is_enclosed() {
        // Plain binary64 accumulation loses the middle residual completely:
        // 2^32 + 2^-32 - 2^32 == 0, while the exact-real result is 2^-32.
        let huge = 2.0_f32.powi(32);
        let tiny = 2.0_f32.powi(-32);
        let coefficients =
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1, 1, 3]), vec![huge, tiny, -huge]).unwrap();
        let make_side = || PatchesData {
            coeff_err: None,
            patches: Some(coefficients.clone()),
            geometry: PatchGeometry::affine((1, 1), (0, 0, 0, 0)),
            identity: false,
            output_shape: (1, 1, 1),
            input_shape: (1, 1, 3),
            unstable_idx: None,
        };
        let bounds = PatchesLinearBounds {
            row_count: 1,
            lower_a: make_side(),
            lower_b: Array1::zeros(1),
            upper_a: make_side(),
            upper_b: Array1::zeros(1),
        };

        let (lower, upper) =
            Conv2dLayer::compute_patches_bias(&bounds, &Array1::ones(1), 1, 1, 1).unwrap();
        assert!(lower[0] <= tiny, "lower {} excluded {tiny}", lower[0]);
        assert!(upper[0] >= tiny, "upper {} excluded {tiny}", upper[0]);
    }

    #[test]
    fn dense_6d_malformed_coeff_err_fails_closed() {
        let spec = (2usize, 2usize, 2usize);
        let spec_dim = spec.0 * spec.1 * spec.2;
        let malformed = Array1::zeros(spec_dim - 1);
        let kernel = ArrayD::ones(IxDyn(&[3, 2, 2, 2]));
        let params = Conv2dPatchesParams {
            kernel: &kernel,
            in_c: 2,
            out_c: 3,
            groups: 1,
            kh: 2,
            kw: 2,
            sh: 1,
            sw: 1,
            ph: 0,
            pw: 0,
            in_h: 3,
            in_w: 3,
            out_h: 2,
            out_w: 2,
        };
        let lower = make_patches_data(&[2, 2, 2, 3, 2, 2], 17, Some(malformed), spec, (3, 2, 2));

        let compose_error = Conv2dLayer::conv2d_patches_backward(&lower, &params, None);
        assert!(
            matches!(compose_error, Err(NyError::ShapeMismatch { .. })),
            "malformed 6D compose coeff_err must fail closed"
        );

        let bounds = PatchesLinearBounds {
            row_count: spec_dim,
            lower_a: lower,
            lower_b: Array1::zeros(spec_dim),
            upper_a: make_patches_data(&[2, 2, 2, 3, 2, 2], 18, None, spec, (3, 2, 2)),
            upper_b: Array1::zeros(spec_dim),
        };
        let bias_error = Conv2dLayer::compute_patches_bias(&bounds, &Array1::ones(3), 3, 2, 2);
        assert!(
            matches!(bias_error, Err(NyError::ShapeMismatch { .. })),
            "malformed 6D bias coeff_err must fail closed"
        );
    }

    // =================================================================
    // 7D explicit-rows tests (docs/PATCHES_7D_COEFF_ERR_CLOSURE.md §5.4
    // T1/T2/T3 + the I5 poison guard).
    //
    // Shared 7D fixture geometry:
    //   conv: kernel [out_c=3, in_c=2, kh=2, kw=2], stride 1, pad 0,
    //         input 4x4 -> out 3x3;
    //   incoming: [row_count=2, spec=(2,2,2), out_c=3, prev_kh=2, prev_kw=2]
    //   composed: [2, 2, 2, 2, in_c=2, new_kh=3, new_kw=3], <= 12 taps/cell.
    // =================================================================

    /// Deterministic dyadic fill: multiples of 2^-6 in [-2, 2]. Exact in f32
    /// and through every small dyadic product/sum the oracles below build, so
    /// coverage is asserted with NO tolerance.
    fn dyadic_fill(n: usize, seed: u32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let k = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed);
                (((k >> 7) % 257) as f32 - 128.0) / 64.0
            })
            .collect()
    }

    fn fixture_params(kernel: &ArrayD<f32>) -> Conv2dPatchesParams<'_> {
        Conv2dPatchesParams {
            kernel,
            in_c: 2,
            out_c: 3,
            groups: 1,
            kh: 2,
            kw: 2,
            sh: 1,
            sw: 1,
            ph: 0,
            pw: 0,
            in_h: 4,
            in_w: 4,
            out_h: 3,
            out_w: 3,
        }
    }

    /// All-positive dyadic kernel [3, 2, 2, 2] (multiples of 2^-4).
    fn fixture_kernel() -> ArrayD<f32> {
        let vals: Vec<f32> = (0..24).map(|i| ((i % 15) + 1) as f32 / 16.0).collect();
        ArrayD::from_shape_vec(IxDyn(&[3, 2, 2, 2]), vals).unwrap()
    }

    fn make_patches_7d(vals: Vec<f32>, coeff_err: Option<Array1<f32>>) -> PatchesData {
        PatchesData {
            coeff_err,
            patches: Some(ArrayD::from_shape_vec(IxDyn(&[2, 2, 2, 2, 3, 2, 2]), vals).unwrap()),
            geometry: PatchGeometry::affine((1, 1), (0, 0, 0, 0)),
            identity: false,
            output_shape: (2, 2, 2),
            input_shape: (3, 3, 3),
            unstable_idx: None,
        }
    }

    /// #patches-deadline-kernel: a deadline no longer forces the SERIAL scatter.
    ///
    /// This test previously pinned the narrow admission — parallel only with no
    /// deadline, or with 7D explicit rows AND the dark gate. Since the CROWN-IBP
    /// collector always passes a per-node deadline and the common conv layout is
    /// 6D, that rule meant every collector-routed conv target scattered
    /// single-threaded. Parallel scatter fills DISJOINT position slabs of
    /// caller-private scratch, so it is bit-identical to the serial fill
    /// regardless of completion order (asserted by the sibling test
    /// `explicit_row_deadline_parallel_scatter_is_bitwise_serial`). A deadline
    /// bounds how much work is attempted; it says nothing about how many cores
    /// may do it.
    #[test]
    fn position_scatter_parallel_admission_ignores_deadline_presence() {
        for &explicit_rows in &[true, false] {
            for &nonsparse in &[true, false] {
                for &deadline_present in &[true, false] {
                    for &gate in &[true, false] {
                        assert!(
                            position_scatter_parallel_admitted(
                                explicit_rows,
                                nonsparse,
                                deadline_present,
                                gate,
                                false,
                            ),
                            "parallel scatter must be admitted regardless of deadline/layout/gate \
                             (explicit_rows={explicit_rows}, nonsparse={nonsparse}, \
                             deadline={deadline_present}, gate={gate})"
                        );
                    }
                }
            }
        }
        assert!(
            !position_scatter_parallel_admitted(true, true, true, true, true),
            "an inner region worker must never create nested Rayon work"
        );

        crate::tests::with_env_edits(|env| {
            env.remove("NY_PATCHES_DEADLINE_PARALLEL_SCATTER");
            assert!(!patches_deadline_parallel_scatter_enabled());
            env.set("NY_PATCHES_DEADLINE_PARALLEL_SCATTER", "0");
            assert!(!patches_deadline_parallel_scatter_enabled());
            env.set("NY_PATCHES_DEADLINE_PARALLEL_SCATTER", "true");
            assert!(!patches_deadline_parallel_scatter_enabled());
            env.set("NY_PATCHES_DEADLINE_PARALLEL_SCATTER", "1");
            assert!(patches_deadline_parallel_scatter_enabled());
        });
    }

    #[test]
    fn explicit_row_deadline_parallel_scatter_is_bitwise_serial() {
        crate::tests::with_env_edits(|env| {
            let kernel = fixture_kernel();
            let params = fixture_params(&kernel);
            let n = 2 * 8 * 12;
            let pd = make_patches_7d(
                det_fill(n, 0x51a7),
                Some(Array1::from_vec(vec![1.0e-4_f32, 7.0e-4])),
            );
            let common_deadline = Some(Instant::now() + std::time::Duration::from_secs(30));

            env.remove("NY_PATCHES_DEADLINE_PARALLEL_SCATTER");
            let serial = Conv2dLayer::conv2d_patches_backward_with_deadline(
                &pd,
                &params,
                None,
                common_deadline,
            )
            .expect("finite serial scatter");
            env.set("NY_PATCHES_DEADLINE_PARALLEL_SCATTER", "1");
            let parallel = Conv2dLayer::conv2d_patches_backward_with_deadline(
                &pd,
                &params,
                None,
                common_deadline,
            )
            .expect("finite parallel scatter");

            assert_bits_eq(
                "deadline parallel composed coefficients",
                parallel.patches.as_ref().unwrap().as_slice().unwrap(),
                serial.patches.as_ref().unwrap().as_slice().unwrap(),
            );
            assert_bits_eq(
                "deadline parallel certified coefficient error",
                parallel.coeff_err.as_ref().unwrap().as_slice().unwrap(),
                serial.coeff_err.as_ref().unwrap().as_slice().unwrap(),
            );
            assert_eq!(parallel.geometry, serial.geometry);
            assert_eq!(parallel.output_shape, serial.output_shape);
            assert_eq!(parallel.input_shape, serial.input_shape);
            assert_eq!(parallel.identity, serial.identity);
        });
    }

    #[test]
    fn position_scatter_midflight_expiry_never_publishes_partial_scratch() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let patch_volume = 4;
        let mut scratch = vec![0.0f32; 96 * patch_volume];
        let poll_count = AtomicUsize::new(0);
        let completed = AtomicUsize::new(0);
        let make_buffer = || ();
        let scatter = |(): &mut (), idx: usize, chunk: &mut [f32]| {
            chunk.fill((idx + 1) as f32);
            completed.fetch_add(1, Ordering::SeqCst);
            Ok(())
        };
        let injected_deadline = || {
            if poll_count.fetch_add(1, Ordering::SeqCst) >= 2 {
                Err(NyError::DeadlineExceeded(
                    "injected position-scatter deadline".into(),
                ))
            } else {
                Ok(())
            }
        };

        // The serial schedule makes the injection deterministic: entry and
        // position zero pass, positions 0..31 complete, then position 32 polls
        // the same authority and fails. Production selects the parallel arm,
        // whose output has the identical caller-private transaction boundary.
        let result = fill_position_scatter_scratch_with_poll(
            &mut scratch,
            patch_volume,
            false,
            &make_buffer,
            &scatter,
            &injected_deadline,
        );
        let published = result.ok().map(|()| scratch.clone());
        assert!(
            published.is_none(),
            "a failed scratch transaction cannot produce a publishable value"
        );
        assert_eq!(completed.load(Ordering::SeqCst), 32);
        assert!(
            scratch.iter().any(|value| *value != 0.0),
            "the injected expiry must occur after real partial work"
        );

        let mut retry = vec![0.0f32; scratch.len()];
        fill_position_scatter_scratch_with_poll(
            &mut retry,
            patch_volume,
            true,
            &make_buffer,
            &|(): &mut (), idx, chunk| {
                chunk.fill((idx + 1) as f32);
                Ok(())
            },
            &|| Ok(()),
        )
        .expect("a fresh parallel transaction must complete");
        for (idx, chunk) in retry.chunks(patch_volume).enumerate() {
            assert!(chunk.iter().all(|value| *value == (idx + 1) as f32));
        }
    }

    #[test]
    fn position_scatter_parallel_error_joins_workers_and_publishes_nothing() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct ActiveGuard<'a>(&'a AtomicUsize);
        impl Drop for ActiveGuard<'_> {
            fn drop(&mut self) {
                self.0.fetch_sub(1, Ordering::SeqCst);
            }
        }

        let patch_volume = 4;
        let mut scratch = vec![0.0f32; 128 * patch_volume];
        let active = AtomicUsize::new(0);
        let started = AtomicUsize::new(0);
        let scatter = |(): &mut (), idx: usize, chunk: &mut [f32]| {
            active.fetch_add(1, Ordering::SeqCst);
            let _active = ActiveGuard(&active);
            started.fetch_add(1, Ordering::SeqCst);
            chunk.fill((idx + 1) as f32);
            if idx == 32 {
                Err(NyError::InternalError(
                    "injected parallel position-scatter failure".into(),
                ))
            } else {
                Ok(())
            }
        };

        let result = fill_position_scatter_scratch_with_poll(
            &mut scratch,
            patch_volume,
            true,
            &|| (),
            &scatter,
            &|| Ok(()),
        );
        assert!(
            matches!(&result, Err(NyError::InternalError(message))
                if message == "injected parallel position-scatter failure"),
            "the worker error must propagate unchanged, got {result:?}"
        );
        let published = result.ok().map(|()| scratch.clone());
        assert!(
            published.is_none(),
            "a parallel worker error cannot produce a publishable value"
        );
        assert!(started.load(Ordering::SeqCst) > 0);
        assert_eq!(
            active.load(Ordering::SeqCst),
            0,
            "the borrowed Rayon drive must join every in-flight worker"
        );
        assert!(
            scratch.iter().any(|value| *value != 0.0),
            "the injected failure must occur after real scratch mutation"
        );
    }

    #[test]
    fn position_scatter_rejects_invalid_geometry_before_work() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let polls = AtomicUsize::new(0);
        let scatters = AtomicUsize::new(0);
        let poll = || {
            polls.fetch_add(1, Ordering::SeqCst);
            Ok(())
        };
        let scatter = |(): &mut (), _idx: usize, _chunk: &mut [f32]| {
            scatters.fetch_add(1, Ordering::SeqCst);
            Ok(())
        };

        for (scratch_len, patch_volume) in [(8, 0), (10, 4)] {
            let mut scratch = vec![7.0f32; scratch_len];
            let before = scratch.clone();
            let result = fill_position_scatter_scratch_with_poll(
                &mut scratch,
                patch_volume,
                true,
                &|| (),
                &scatter,
                &poll,
            );
            assert!(
                matches!(result, Err(NyError::InvalidSpec(_))),
                "invalid geometry must fail closed: len={scratch_len} volume={patch_volume}"
            );
            assert_eq!(scratch, before, "geometry failure cannot mutate scratch");
        }
        assert_eq!(polls.load(Ordering::SeqCst), 0);
        assert_eq!(scatters.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn expired_deadline_parallel_scatter_returns_no_crown_bounds() {
        crate::tests::with_env_edits(|env| {
            env.set("NY_PATCHES_DEADLINE_PARALLEL_SCATTER", "1");
            let kernel = fixture_kernel();
            let layer = Conv2dLayer::with_input_shape(kernel, None, (1, 1), (0, 0), 4, 4)
                .expect("fixture Conv2d");
            let n = 2 * 8 * 12;
            let lower = make_patches_7d(det_fill(n, 11), None);
            let upper = make_patches_7d(det_fill(n, 29), None);
            let lower_before = lower.patches.as_ref().unwrap().clone();
            let upper_before = upper.patches.as_ref().unwrap().clone();
            let bounds = PatchesLinearBounds {
                row_count: 2,
                lower_a: lower,
                lower_b: Array1::zeros(2),
                upper_a: upper,
                upper_b: Array1::zeros(2),
            };

            let result =
                layer.propagate_patches_engine_and_deadline(&bounds, None, Some(Instant::now()));
            assert!(
                result
                    .as_ref()
                    .is_err_and(|error| error.is_deadline_exceeded()),
                "an expired authority must fail closed, got {result:?}"
            );
            let published: Option<CrownBounds> = result.ok();
            assert!(
                published.is_none(),
                "deadline failure cannot publish CrownBounds"
            );
            assert_bits_eq(
                "expired lower input",
                bounds.lower_a.patches.as_ref().unwrap().as_slice().unwrap(),
                lower_before.as_slice().unwrap(),
            );
            assert_bits_eq(
                "expired upper input",
                bounds.upper_a.patches.as_ref().unwrap().as_slice().unwrap(),
                upper_before.as_slice().unwrap(),
            );
        });
    }

    #[test]
    fn explicit_row_flat_bias_reduction_is_bitwise_oracle_and_parallel_invariant() {
        let n = 2 * 8 * 3 * 4;
        let lower = ArrayD::from_shape_vec(IxDyn(&[2, 2, 2, 2, 3, 2, 2]), det_fill(n, 31)).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[2, 2, 2, 2, 3, 2, 2]), det_fill(n, 57)).unwrap();
        let bias = [0.75_f32, -0.5, 0.25];
        let old_lower = [0.5_f32, -0.25];
        let old_upper = [1.25_f32, 0.125];

        let poll = || Ok(());
        let serial = reduce_explicit_row_bias_flat_with_poll(
            lower.as_slice().unwrap(),
            upper.as_slice().unwrap(),
            2,
            8,
            3,
            4,
            4,
            &bias,
            &old_lower,
            &old_upper,
            false,
            &poll,
        )
        .unwrap();
        let parallel = reduce_explicit_row_bias_flat_with_poll(
            lower.as_slice().unwrap(),
            upper.as_slice().unwrap(),
            2,
            8,
            3,
            4,
            4,
            &bias,
            &old_lower,
            &old_upper,
            true,
            &poll,
        )
        .unwrap();
        assert_eq!(
            parallel, serial,
            "parallelism is across independent rows only"
        );

        let mut oracle = Vec::new();
        for row in 0..2 {
            let mut lower_sum = 0.0f64;
            let mut upper_sum = 0.0f64;
            let mut lower_abs = f64::from(old_lower[row]).abs();
            let mut upper_abs = f64::from(old_upper[row]).abs();
            for soc in 0..2 {
                for soh in 0..2 {
                    for sow in 0..2 {
                        for c in 0..3 {
                            let mut lc_sum = 0.0f64;
                            let mut uc_sum = 0.0f64;
                            let mut lc_abs = 0.0f64;
                            let mut uc_abs = 0.0f64;
                            for ki in 0..2 {
                                for kj in 0..2 {
                                    let a = f64::from(lower[[row, soc, soh, sow, c, ki, kj]]);
                                    lc_sum += a;
                                    lc_abs += a.abs();
                                }
                            }
                            for ki in 0..2 {
                                for kj in 0..2 {
                                    let a = f64::from(upper[[row, soc, soh, sow, c, ki, kj]]);
                                    uc_sum += a;
                                    uc_abs += a.abs();
                                }
                            }
                            let bias_c = f64::from(bias[c]);
                            lower_sum += lc_sum * bias_c;
                            upper_sum += uc_sum * bias_c;
                            lower_abs += lc_abs * bias_c.abs();
                            upper_abs += uc_abs * bias_c.abs();
                        }
                    }
                }
            }
            oracle.push(ExplicitRowBiasReduction {
                lower_sum,
                upper_sum,
                lower_abs,
                upper_abs,
            });
        }
        assert_eq!(
            serial, oracle,
            "flat traversal must retain every historical per-row f64 operation"
        );
    }

    #[test]
    fn explicit_row_flat_bias_deadline_failure_discards_all_scratch() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        // One row visits 8,192 coefficients across both sides, forcing multiple
        // cooperative polls from inside the reduction rather than only at its
        // entry/exit boundary.
        let row_count = 2;
        let positions = 64;
        let out_c = 8;
        let taps = 8;
        let per_side = row_count * positions * out_c * taps;
        let lower = det_fill(per_side, 101);
        let upper = det_fill(per_side, 202);
        let lower_before = lower.clone();
        let upper_before = upper.clone();
        let bias = vec![0.125_f32; out_c];
        let old_lower = vec![-3.0_f32; row_count];
        let old_upper = vec![7.0_f32; row_count];
        let poll_count = AtomicUsize::new(0);
        let fail_during_row = || {
            if poll_count.fetch_add(1, Ordering::SeqCst) >= 2 {
                Err(NyError::DeadlineExceeded(
                    "injected flat-bias deadline".into(),
                ))
            } else {
                Ok(())
            }
        };

        let error = reduce_explicit_row_bias_flat_with_poll(
            &lower,
            &upper,
            row_count,
            positions,
            out_c,
            taps,
            taps,
            &bias,
            &old_lower,
            &old_upper,
            false,
            &fail_during_row,
        )
        .expect_err("injected mid-row deadline must reject the complete transaction");
        assert!(error.is_deadline_exceeded());
        assert_eq!(
            lower, lower_before,
            "deadline failure cannot mutate lower input"
        );
        assert_eq!(
            upper, upper_before,
            "deadline failure cannot mutate upper input"
        );

        let completed = reduce_explicit_row_bias_flat_with_poll(
            &lower,
            &upper,
            row_count,
            positions,
            out_c,
            taps,
            taps,
            &bias,
            &old_lower,
            &old_upper,
            false,
            &|| Ok(()),
        )
        .expect("a fresh transaction after discard must complete");
        assert_eq!(completed.len(), row_count);
    }

    /// Exhaustive f64 transpose-conv oracle for the fixture geometry
    /// (stride 1, padding 0): true composed coefficient of output cell
    /// (ic, ni, nj) from incoming slab (row, soc, soh, sow, ·).
    fn transpose_conv_oracle(
        a: &ArrayD<f32>,
        kernel: &ArrayD<f32>,
        cell: (usize, usize, usize, usize, usize, usize, usize),
    ) -> f64 {
        let (row, soc, soh, sow, ic, ni, nj) = cell;
        let mut acc = 0.0f64;
        for c in 0..3 {
            for gy in 0..2 {
                for gx in 0..2 {
                    let ki = ni as isize - gy as isize;
                    let kj = nj as isize - gx as isize;
                    if (0..2).contains(&ki) && (0..2).contains(&kj) {
                        acc += f64::from(a[[row, soc, soh, sow, c, gy, gx]])
                            * f64::from(kernel[[c, ic, ki as usize, kj as usize]]);
                    }
                }
            }
        }
        acc
    }

    /// Spec §5.4 T1: the 7D compose emits a SPEC-ROW-indexed err
    /// (len row_count, not num_positions) that covers |stored − true| for
    /// EVERY composed coefficient. Dyadic fixture: kernel and truths dyadic,
    /// stored = true + old_err[row] tap-wise (all exact in f32, composed
    /// values exact, <= 12 dyadic terms per cell), so the deviation is
    /// EXACTLY old_err[row]·Σ_taps w and coverage is asserted tolerance-free.
    #[test]
    fn explicit_rows_compose_err_covers_true_deviation() {
        let kernel = fixture_kernel();
        let params = fixture_params(&kernel);

        let old_err = [2f32.powi(-12), 2f32.powi(-8)];
        let n = 2 * 8 * 12;
        let per_row = n / 2;
        let a_true_vals = dyadic_fill(n, 9);
        let a_stored_vals: Vec<f32> = a_true_vals
            .iter()
            .enumerate()
            .map(|(i, &v)| v + old_err[i / per_row])
            .collect();
        let a_true = ArrayD::from_shape_vec(IxDyn(&[2, 2, 2, 2, 3, 2, 2]), a_true_vals).unwrap();
        let pd = make_patches_7d(a_stored_vals, Some(Array1::from_vec(old_err.to_vec())));

        let out = Conv2dLayer::conv2d_patches_backward(&pd, &params, None).unwrap();
        let ne = out
            .coeff_err
            .as_ref()
            .expect("7D compose must emit Some coeff_err");
        assert_eq!(ne.len(), 2, "err must be spec-row indexed (len row_count)");
        assert!(ne.iter().all(|e| e.is_finite() && *e > 0.0));

        let stored = out.patches.as_ref().unwrap();
        assert_eq!(stored.shape(), &[2, 2, 2, 2, 2, 3, 3]);
        let mut max_dev = [0.0f64; 2];
        for row in 0..2 {
            for soc in 0..2 {
                for soh in 0..2 {
                    for sow in 0..2 {
                        for ic in 0..2 {
                            for ni in 0..3 {
                                for nj in 0..3 {
                                    let tru = transpose_conv_oracle(
                                        &a_true,
                                        &kernel,
                                        (row, soc, soh, sow, ic, ni, nj),
                                    );
                                    let dev = (f64::from(stored[[row, soc, soh, sow, ic, ni, nj]])
                                        - tru)
                                        .abs();
                                    assert!(
                                        dev <= f64::from(ne[row]),
                                        "row {row} cell ({soc},{soh},{sow},{ic},{ni},{nj}): \
                                         deviation {dev:e} not covered by err {:e}",
                                        ne[row]
                                    );
                                    max_dev[row] = max_dev[row].max(dev);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Sharpness: the ‖k‖₁·old_err[row] carry term is load-bearing — an
        // implementation without it (or misindexing the err by position
        // instead of spec row) could at most emit the intrinsic γ_K term,
        // which the actual deviation exceeds on BOTH rows.
        let kernel_l1: f64 = kernel.iter().map(|v| f64::from(*v).abs()).sum();
        let gamma = crate::layers::linear::crown_single_gamma_n_f32(3 * 2 * 2);
        for row in 0..2 {
            let mut rowmax = 0.0f64;
            for &v in pd.patches.as_ref().unwrap().index_axis(Axis(0), row).iter() {
                rowmax = rowmax.max(f64::from(v).abs());
            }
            let intrinsic_only =
                f64::from(ny_tensor::next_up_f32((gamma * rowmax * kernel_l1) as f32));
            assert!(
                max_dev[row] > intrinsic_only,
                "row {row}: fixture too weak to pin the carried term \
                 ({:e} <= {intrinsic_only:e})",
                max_dev[row]
            );
        }

        // Err-free input: Some is still emitted (the intrinsic f32
        // contraction rounding is real) and covers an f64 oracle. Non-dyadic
        // fills so the composition genuinely rounds; the f64 oracle's own
        // noise (~2^-52 relative) is orders below the γ_K-based emission.
        let pd_none = make_patches_7d(det_fill(n, 5), None);
        let out_none = Conv2dLayer::conv2d_patches_backward(&pd_none, &params, None).unwrap();
        let ne_none = out_none
            .coeff_err
            .as_ref()
            .expect("7D compose must emit Some coeff_err for err-free input");
        assert_eq!(ne_none.len(), 2);
        let a_none = pd_none.patches.as_ref().unwrap();
        let stored_none = out_none.patches.as_ref().unwrap();
        for row in 0..2 {
            for soc in 0..2 {
                for soh in 0..2 {
                    for sow in 0..2 {
                        for ic in 0..2 {
                            for ni in 0..3 {
                                for nj in 0..3 {
                                    let tru = transpose_conv_oracle(
                                        a_none,
                                        &kernel,
                                        (row, soc, soh, sow, ic, ni, nj),
                                    );
                                    let dev =
                                        (f64::from(stored_none[[row, soc, soh, sow, ic, ni, nj]])
                                            - tru)
                                            .abs();
                                    assert!(
                                        dev <= f64::from(ne_none[row]),
                                        "err-free row {row}: intrinsic rounding not covered"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Spec §5.4 T2: a carried 7D err whose length is not the spec row count
    /// is a hard ShapeMismatch (never a silent `.get().unwrap_or(0.0)`
    /// under-count), for both the compose and the bias fold; plus the §14 C4
    /// hardening (mixed 6D/7D pair: clean error where the old code panicked).
    #[test]
    fn explicit_rows_compose_err_length_mismatch_rejected() {
        let kernel = fixture_kernel();
        let params = fixture_params(&kernel);
        let n = 2 * 8 * 12;

        // 2A compose: Some(len 3) against row_count (= shape[0]) 2.
        let pd = make_patches_7d(
            dyadic_fill(n, 3),
            Some(Array1::from_vec(vec![1e-4_f32, 2e-4, 3e-4])),
        );
        let r = Conv2dLayer::conv2d_patches_backward(&pd, &params, None);
        assert!(
            matches!(r, Err(NyError::ShapeMismatch { .. })),
            "compose must reject a wrong-length 7D err, got {r:?}"
        );

        // 2B bias fold: wrong-length err on the lower side, then the upper.
        let bias = Array1::from_vec(vec![0.75_f32, -0.5, 0.25]);
        let good = || {
            make_patches_7d(
                dyadic_fill(n, 3),
                Some(Array1::from_vec(vec![1e-4_f32, 2e-4])),
            )
        };
        let bad = || make_patches_7d(dyadic_fill(n, 4), Some(Array1::from_vec(vec![1e-4_f32; 3])));
        let lower_b = Array1::from_vec(vec![0.5_f32, -0.25]);
        let upper_b = Array1::from_vec(vec![1.0_f32, 0.125]);
        let plb = PatchesLinearBounds {
            row_count: 2,
            lower_a: bad(),
            lower_b: lower_b.clone(),
            upper_a: good(),
            upper_b: upper_b.clone(),
        };
        let r = Conv2dLayer::compute_patches_bias(&plb, &bias, 3, 3, 3);
        assert!(
            matches!(r, Err(NyError::ShapeMismatch { .. })),
            "bias fold must reject a wrong-length lower err, got {r:?}"
        );
        let plb = PatchesLinearBounds {
            row_count: 2,
            lower_a: good(),
            lower_b: lower_b.clone(),
            upper_a: bad(),
            upper_b: upper_b.clone(),
        };
        let r = Conv2dLayer::compute_patches_bias(&plb, &bias, 3, 3, 3);
        assert!(
            matches!(r, Err(NyError::ShapeMismatch { .. })),
            "bias fold must reject a wrong-length upper err, got {r:?}"
        );

        // §14 C4: a mixed 7D-lower/6D-upper pair is a clean ShapeMismatch
        // (the pre-change code panicked on the 7-index read of the 6D side).
        let plb = PatchesLinearBounds {
            row_count: 2,
            lower_a: good(),
            lower_b,
            upper_a: PatchesData {
                coeff_err: None,
                patches: Some(
                    ArrayD::from_shape_vec(IxDyn(&[2, 2, 2, 3, 2, 2]), dyadic_fill(96, 8)).unwrap(),
                ),
                geometry: PatchGeometry::affine((1, 1), (0, 0, 0, 0)),
                identity: false,
                output_shape: (2, 2, 2),
                input_shape: (3, 3, 3),
                unstable_idx: None,
            },
            upper_b,
        };
        let r = Conv2dLayer::compute_patches_bias(&plb, &bias, 3, 3, 3);
        assert!(
            matches!(r, Err(NyError::ShapeMismatch { .. })),
            "bias fold must reject a mixed 6D/7D side pair, got {r:?}"
        );
    }

    /// Spec §5.4 T3 — THE PRE-FIX HOLE2 UNSOUNDNESS REPRODUCER: with carried
    /// coefficient errs, the adversarial admissible truths
    /// `a_true = a_stored ∓ sign(bias[c])·old_err[row]` (per side) push the
    /// true folded bias strictly past what the no-widen fold brackets; the
    /// widened fold must still enclose them. This test FAILS on the
    /// pre-closure explicit-rows branch (which applied no widen at all). All
    /// fixture values are dyadic, so the f64 oracle folds are EXACT and the
    /// asserts carry no tolerance.
    #[test]
    fn explicit_rows_bias_widen_covers_adversarial_truth() {
        let n = 2 * 8 * 12;
        let bias = Array1::from_vec(vec![0.75_f32, -0.5, 0.25]);
        let le = [2f32.powi(-6), 2f32.powi(-9)];
        let ue = [2f32.powi(-7), 2f32.powi(-10)];
        let old_lb = Array1::from_vec(vec![0.5_f32, -0.25]);
        let old_ub = Array1::from_vec(vec![1.25_f32, 0.125]);
        let plb = PatchesLinearBounds {
            row_count: 2,
            lower_a: make_patches_7d(dyadic_fill(n, 31), Some(Array1::from_vec(le.to_vec()))),
            lower_b: old_lb.clone(),
            upper_a: make_patches_7d(dyadic_fill(n, 57), Some(Array1::from_vec(ue.to_vec()))),
            upper_b: old_ub.clone(),
        };
        let (nlb, nub) = Conv2dLayer::compute_patches_bias(&plb, &bias, 3, 3, 3).unwrap();

        let lt = plb.lower_a.patches.as_ref().unwrap();
        let ut = plb.upper_a.patches.as_ref().unwrap();
        // Σ over all folded taps of |bias[c(tap)]| =
        // positions·(kh·kw)·Σ_c|bias_c| = 8·4·1.5 = 48 (exact).
        let widen_factor = 48.0f64;
        for row in 0..2 {
            let mut s_l = 0.0f64;
            let mut s_u = 0.0f64;
            for soc in 0..2 {
                for soh in 0..2 {
                    for sow in 0..2 {
                        for c in 0..3 {
                            for ki in 0..2 {
                                for kj in 0..2 {
                                    s_l += f64::from(lt[[row, soc, soh, sow, c, ki, kj]])
                                        * f64::from(bias[c]);
                                    s_u += f64::from(ut[[row, soc, soh, sow, c, ki, kj]])
                                        * f64::from(bias[c]);
                                }
                            }
                        }
                    }
                }
            }
            // Extremal admissible truths (|Δ per tap| = old_err[row], each
            // tap's sign chosen against the bias): T_min = S − e·48 (lower),
            // T_max = S + e·48 (upper), exactly.
            let t_min = s_l - f64::from(le[row]) * widen_factor;
            let t_max = s_u + f64::from(ue[row]) * widen_factor;
            assert!(
                f64::from(nlb[row]) <= f64::from(old_lb[row]) + t_min,
                "row {row}: lower bias {} does not enclose the adversarial true \
                 fold {} (pre-fix HOLE2)",
                nlb[row],
                f64::from(old_lb[row]) + t_min
            );
            assert!(
                f64::from(nub[row]) >= f64::from(old_ub[row]) + t_max,
                "row {row}: upper bias {} does not enclose the adversarial true \
                 fold {} (pre-fix HOLE2)",
                nub[row],
                f64::from(old_ub[row]) + t_max
            );
            // Liveness: the widen is real — strictly outside a no-widen fold.
            let rep_l = ny_tensor::next_down_f32((f64::from(old_lb[row]) + s_l) as f32);
            let rep_u = ny_tensor::next_up_f32((f64::from(old_ub[row]) + s_u) as f32);
            assert!(nlb[row] < rep_l, "row {row}: lower widen not live");
            assert!(nub[row] > rep_u, "row {row}: upper widen not live");
        }

        // §14 C3 hardening: an identity side in the explicit-rows fold is a
        // hard error (its affine contribution has no 7D analog in the fold),
        // never a silent drop.
        let plb_ident = PatchesLinearBounds {
            row_count: 2,
            lower_a: make_patches_7d(dyadic_fill(n, 31), Some(Array1::from_vec(le.to_vec()))),
            lower_b: old_lb,
            upper_a: PatchesData {
                coeff_err: None,
                patches: None,
                geometry: PatchGeometry::affine((1, 1), (0, 0, 0, 0)),
                identity: true,
                output_shape: (2, 2, 2),
                input_shape: (3, 3, 3),
                unstable_idx: None,
            },
            upper_b: old_ub,
        };
        assert!(
            Conv2dLayer::compute_patches_bias(&plb_ident, &bias, 3, 3, 3).is_err(),
            "identity side must be a hard error on the explicit-rows bias fold"
        );
    }

    /// Spec I5 pins for both 7D arms: non-finite or negative carried err
    /// poisons OUTWARD (+INF err; −INF lower / +INF upper bias) on the
    /// affected rows only — NEVER NaN and never a silent 0 — and the
    /// 0·INF hazards are short-circuited.
    #[test]
    fn explicit_rows_nonfinite_err_poisons_outward_never_nan() {
        let kernel = fixture_kernel();
        let params = fixture_params(&kernel);
        let n = 2 * 8 * 12;

        // 2A compose: NaN err row poisons to +INF; the other row stays finite.
        let pd = make_patches_7d(
            dyadic_fill(n, 11),
            Some(Array1::from_vec(vec![f32::NAN, 1e-4])),
        );
        let out = Conv2dLayer::conv2d_patches_backward(&pd, &params, None).unwrap();
        let ne = out.coeff_err.as_ref().unwrap();
        assert_eq!(
            ne[0],
            f32::INFINITY,
            "NaN err must poison to +INF, never 0/NaN"
        );
        assert!(ne[1].is_finite() && ne[1] > 0.0);

        // A negative subnormal likewise poisons. A DAZ-sensitive `v >= 0`
        // comparison could otherwise reinterpret it as valid signed zero.
        let pd_neg = make_patches_7d(
            dyadic_fill(n, 11),
            Some(Array1::from_vec(vec![f32::from_bits(0x8000_0001), 1e-4])),
        );
        let out_neg = Conv2dLayer::conv2d_patches_backward(&pd_neg, &params, None).unwrap();
        assert_eq!(out_neg.coeff_err.as_ref().unwrap()[0], f32::INFINITY);

        // 2B bias fold: an +INF err row poisons THAT side's bias outward on
        // ITS rows only; the other rows and the other side stay finite; no
        // NaN anywhere.
        let bias = Array1::from_vec(vec![0.75_f32, -0.5, 0.25]);
        let plb = PatchesLinearBounds {
            row_count: 2,
            lower_a: make_patches_7d(
                dyadic_fill(n, 31),
                Some(Array1::from_vec(vec![f32::INFINITY, 2f32.powi(-9)])),
            ),
            lower_b: Array1::from_vec(vec![0.5_f32, -0.25]),
            upper_a: make_patches_7d(
                dyadic_fill(n, 57),
                Some(Array1::from_vec(vec![2f32.powi(-9), f32::INFINITY])),
            ),
            upper_b: Array1::from_vec(vec![1.25_f32, 0.125]),
        };
        let (nlb, nub) = Conv2dLayer::compute_patches_bias(&plb, &bias, 3, 3, 3).unwrap();
        assert_eq!(nlb[0], f32::NEG_INFINITY);
        assert!(nlb[1].is_finite());
        assert!(nub[0].is_finite());
        assert_eq!(nub[1], f32::INFINITY);
        assert!(nlb.iter().chain(nub.iter()).all(|v| !v.is_nan()));

        // 0·INF short-circuit: with an all-zero conv bias no folded tap can
        // deviate the fold (the true widen is exactly 0), so even an +INF
        // carried err must yield finite, NaN-free biases.
        let zero_bias = Array1::from_vec(vec![0.0_f32, 0.0, 0.0]);
        let (zlb, zub) = Conv2dLayer::compute_patches_bias(&plb, &zero_bias, 3, 3, 3).unwrap();
        assert!(
            zlb.iter().chain(zub.iter()).all(|v| v.is_finite()),
            "all-zero bias with +INF err must not poison or NaN: {zlb:?} {zub:?}"
        );
    }
}
