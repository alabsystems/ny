// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Batched CROWN backward for element-wise activation functions.
//!
//! Operates on `ArrayD` A-matrices reshaped to 3D (batch × output × input).

use ndarray::{Array2, ArrayD, IxDyn};
use ny_core::dd::next_up_f64;
use ny_core::{checked_shape_product, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

use super::compose;
use crate::layers::activations::LinearRelaxation;
use crate::layers::linear::bias::{add_f64_down, add_f64_up, f32_to_f64_exact};
use crate::BatchedLinearBounds;

/// Outward `|a| + |b|` for the sign-flip two-line cover (#vnncomp-aw-constant).
///
/// PRIVATE COPY of `crown_dense::cover_sum_up`, kept in step with it deliberately:
/// the dense and batched arms are the SAME composition reached through two entry
/// points, and `src/tests/proptest_soundness/crown_batched_equiv.rs` asserts their
/// published biases match at ZERO tolerance. A cover computed differently here
/// would charge the same fold differently depending only on which arm the caller
/// took — the exact asymmetry this class of defect is made of. If one copy
/// changes, change both.
///
/// The bare form `(a.abs() + b.abs()) as f64` performs the addition in BINARY32
/// under round-to-nearest, so it can land BELOW the true sum. That cover
/// multiplies the incoming coefficient error into both published halves —
/// `new_*_a_err` and, through `int_cover`, the constant widening `*_b_err` — so a
/// cover rounded the wrong way under-counts BOTH, in the false-proof direction.
/// Converting exactly first also keeps a binary32-subnormal operand alive on a
/// DAZ host, where a hardware `as f64` may read it as zero, and removes a
/// spurious binary32 overflow to `+inf` for two near-`f32::MAX` intercepts.
///
/// `sum == 0.0` returns an exact `0.0` rather than `next_up_f64(0.0) = 5e-324`:
/// a relaxation with two zero intercepts contributes exactly nothing, and
/// stepping it off zero would smear a subnormal charge across every cell.
///
/// Non-finite operands propagate: `+inf` stays `+inf`, NaN stays NaN and is
/// sanitized to `+inf` by `set_coeff_err` — both vacuous, sound.
#[inline]
fn cover_sum_up(a: f32, b: f32) -> f64 {
    let sum = f32_to_f64_exact(a).abs() + f32_to_f64_exact(b).abs();
    if sum == 0.0 {
        0.0
    } else {
        next_up_f64(sum)
    }
}

/// Publish one accumulated binary64 LOWER constant as binary32, OUTWARD and
/// saturating (#activation-batched-const-fold).
///
/// `next_down_f32(x as f32)` — the historical publish step — is sound only while
/// `x as f32` stays finite. For a FINITE `x` above `f32::MAX` the nearest-cast
/// saturates to `+inf`, and `next_down_f32` returns `±inf` unchanged (see its
/// documented special case), so the row would publish `+inf` as a LOWER
/// constant: the most optimistic value representable, on a row whose true
/// constant is merely very large. `f32::MAX` is the largest finite binary32 at
/// or below such an `x` — a `nearest` of `+inf` means `x > f32::MAX` — so it is
/// the correct saturation endpoint. A `nearest` of `-inf` needs no repair:
/// `-inf` is already a sound lower bound.
///
/// A NON-finite accumulator fails CLOSED to `-inf`. `+inf` there does NOT mean
/// "the true constant is enormous": binary64 addition saturates irreversibly,
/// so `+1e308, +1e308, -2e308` accumulates to `+inf` while the exact sum is
/// `0`. Once the accumulator has saturated, no finite lower bound is
/// justifiable. NaN takes the same exit — and note this is a TIGHTENING of the
/// previous behaviour in aggregate: a NaN constant used to reach
/// `new_or_conservative` and degrade the WHOLE bounds object to conservative,
/// where now only the poisoned row goes to `-inf`.
///
/// For every finite `x` whose binary32 image is finite this is EXACTLY the
/// historical `next_down_f32(x as f32)`, bit for bit.
#[inline]
fn publish_constant_down(value: f64) -> f32 {
    if !value.is_finite() {
        return f32::NEG_INFINITY;
    }
    let nearest = value as f32;
    if nearest == f32::INFINITY {
        return f32::MAX;
    }
    next_down_f32(nearest)
}

/// Upper-direction counterpart of [`publish_constant_down`].
///
/// A finite `x` below `-f32::MAX` casts to `-inf`, which `next_up_f32` leaves
/// alone, publishing `-inf` as an UPPER constant — again the most optimistic
/// value representable. `f32::MIN` (`= -f32::MAX`) is the smallest finite
/// binary32 at or above such an `x`. A non-finite accumulator fails closed to
/// `+inf`.
#[inline]
fn publish_constant_up(value: f64) -> f32 {
    if !value.is_finite() {
        return f32::INFINITY;
    }
    let nearest = value as f32;
    if nearest == f32::NEG_INFINITY {
        return f32::MIN;
    }
    next_up_f32(nearest)
}

/// Generic batched CROWN backward for element-wise activation functions.
///
/// Delegates to [`crown_elementwise_backward_batched_indexed`] with a no-op neuron index.
pub fn crown_elementwise_backward_batched<F>(
    bounds: &BatchedLinearBounds,
    pre_activation: &BoundedTensor,
    relaxation_fn: F,
) -> Result<BatchedLinearBounds>
where
    F: Fn(f32, f32) -> LinearRelaxation,
{
    crown_elementwise_backward_batched_indexed(bounds, pre_activation, |l, u, _i| {
        relaxation_fn(l, u)
    })
}

/// Indexed batched CROWN backward for element-wise activations.
///
/// Like [`crown_elementwise_backward_batched`], but the relaxation function receives
/// the neuron index `i`. Supports per-neuron parameters (e.g., PReLU per-channel slopes).
///
/// Bias accumulation uses f64 to prevent catastrophic cancellation, and
/// takes a DIRECTED f64 step after every addition so the fold's own rounding is
/// charged rather than assumed away by the final cast
/// (#activation-batched-const-fold; see the comment on the composition loop).
pub fn crown_elementwise_backward_batched_indexed<F>(
    bounds: &BatchedLinearBounds,
    pre_activation: &BoundedTensor,
    relaxation_fn: F,
) -> Result<BatchedLinearBounds>
where
    F: Fn(f32, f32, usize) -> LinearRelaxation,
{
    let pre_shape = pre_activation.shape();
    let a_shape = bounds.lower_a().shape();

    if a_shape.len() < 2 {
        return Err(NyError::InvalidSpec(
            "BatchedLinearBounds must have at least 2 dimensions".to_string(),
        ));
    }

    let out_dim = a_shape[a_shape.len() - 2];
    let in_dim = a_shape[a_shape.len() - 1];
    let batch_dims = &a_shape[..a_shape.len() - 2];
    let total_batch: usize = checked_shape_product(batch_dims)
        .ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "compose_batched_bounds_3d: batch dimensions {batch_dims:?} overflow usize",
            ))
        })?
        .max(1);

    let pre_in_dim = *pre_shape.last().unwrap_or(&0);
    if pre_in_dim != in_dim {
        return Err(NyError::ShapeMismatch {
            expected: vec![in_dim],
            got: vec![pre_in_dim],
        });
    }

    // Reshape to working dimensions
    let pre_lower_flat = pre_activation
        .lower()
        .view()
        .into_shape_with_order((total_batch, in_dim))
        .map_err(|_| NyError::InvalidSpec("Cannot reshape pre_lower".to_string()))?;
    let pre_upper_flat = pre_activation
        .upper()
        .view()
        .into_shape_with_order((total_batch, in_dim))
        .map_err(|_| NyError::InvalidSpec("Cannot reshape pre_upper".to_string()))?;

    let lower_a_3d = bounds
        .lower_a()
        .view()
        .into_shape_with_order((total_batch, out_dim, in_dim))
        .map_err(|_| NyError::InvalidSpec("Cannot reshape lower_a".to_string()))?;
    let upper_a_3d = bounds
        .upper_a()
        .view()
        .into_shape_with_order((total_batch, out_dim, in_dim))
        .map_err(|_| NyError::InvalidSpec("Cannot reshape upper_a".to_string()))?;
    let lower_b_2d = bounds
        .lower_b()
        .view()
        .into_shape_with_order((total_batch, out_dim))
        .map_err(|_| NyError::InvalidSpec("Cannot reshape lower_b".to_string()))?;
    let upper_b_2d = bounds
        .upper_b()
        .view()
        .into_shape_with_order((total_batch, out_dim))
        .map_err(|_| NyError::InvalidSpec("Cannot reshape upper_b".to_string()))?;

    // Output arrays — bias in f64 for numerical stability
    let total_rows = total_batch * out_dim;
    let mut new_lower_a = Array2::<f32>::zeros((total_rows, in_dim));
    let mut new_upper_a = Array2::<f32>::zeros((total_rows, in_dim));
    let mut new_lower_b = Array2::<f64>::zeros((total_batch, out_dim));
    let mut new_upper_b = Array2::<f64>::zeros((total_batch, out_dim));

    // Incoming certified coefficient error (#vnncomp-aw-soundness). The batched
    // activation backward MUST propagate it through the relaxation exactly like
    // the scalar `crown_elementwise_backward` (crown_dense.rs). Every composed
    // coefficient carries the fresh f32 product gap, even when no incoming error
    // exists. Incoming error uses the chosen envelope while its sign is stable;
    // otherwise both envelopes cover a possible sign flip. Intercept uncertainty
    // is folded OUTWARD into the bias. Dropping any of these terms makes the
    // batched (β-CROWN/BaB) verdict optimistic relative to the real composition.
    let in_lower_err = bounds.lower_a_err.as_ref();
    let in_upper_err = bounds.upper_a_err.as_ref();
    let in_lower_err_3d = in_lower_err
        .map(|e| {
            e.view()
                .into_shape_with_order((total_batch, out_dim, in_dim))
        })
        .transpose()
        .map_err(|_| NyError::InvalidSpec("Cannot reshape incoming lower_a_err".to_string()))?;
    let in_upper_err_3d = in_upper_err
        .map(|e| {
            e.view()
                .into_shape_with_order((total_batch, out_dim, in_dim))
        })
        .transpose()
        .map_err(|_| NyError::InvalidSpec("Cannot reshape incoming upper_a_err".to_string()))?;
    let mut new_lower_a_err = Array2::<f32>::zeros((total_rows, in_dim));
    let mut new_upper_a_err = Array2::<f32>::zeros((total_rows, in_dim));
    // Per-(batch,out) certified intercept error: subtracted from lower, added to upper.
    let mut lower_b_err = Array2::<f64>::zeros((total_batch, out_dim));
    let mut upper_b_err = Array2::<f64>::zeros((total_batch, out_dim));

    // Track which output rows have non-finite coefficients after coeff × slope.
    let mut lower_nonfinite_rows = vec![false; total_rows];
    let mut upper_nonfinite_rows = vec![false; total_rows];

    // Seed each cell's constant fold with the incoming binary32 bias. The decode
    // is EXACT (binary32 is a subset of binary64), so the seed carries no error
    // and needs no charge — the accumulation below is where the error is born.
    // `f32_to_f64_exact` rather than `as f64`, matching the dense twin: a plain
    // hardware conversion may observe a binary32-SUBNORMAL seed as signed zero on
    // a DAZ host, which for a lower bias reads `-1.4e-45` as `0.0` and moves the
    // bound INWARD.
    for b in 0..total_batch {
        for j in 0..out_dim {
            new_lower_b[[b, j]] = f32_to_f64_exact(lower_b_2d[[b, j]]);
            new_upper_b[[b, j]] = f32_to_f64_exact(upper_b_2d[[b, j]]);
        }
    }

    // CERTIFIED CONSTANT FOLD (#activation-batched-const-fold).
    //
    // THE DEFECT THIS CLOSES. The intercept fold used to be a plain
    // round-to-nearest `+=` over `in_dim` summands, published through
    // `next_down_f32(x as f32)` / `next_up_f32(x as f32)`. That cast
    // covers the binary64 -> binary32 NARROWING and nothing else, and
    // its allowance is one ULP OF THE RESULT. The accumulation error
    // is not result-scaled: Higham (*Accuracy and Stability*, Thm 3.1)
    // bounds a recursive sum by `gamma_d * SUM|summand|`, which does
    // NOT collapse when the result does. Worked shape, exactly the one
    // the test at the bottom of this file builds: `+2^60`, 4096 x `-1`,
    // `-2^60` accumulate to EXACTLY `0.0` in binary64 (`ulp(2^60) = 256`
    // swallows every `-1`) where the truth is `-4096`. One ULP of `0.0`
    // is `1.4e-45`; the real error is `4096`. Both directions cancel
    // this way, so both were publishing a constant on the WRONG side of
    // the true relaxation — a false-`unsat` mechanism in the graph
    // alpha/beta BaB lane, which runs this arm per node.
    //
    // THE IDIOM. This is the codebase's PER-STEP DIRECTED BINARY64
    // accumulation (`layers/linear/bias.rs`: `add_f64_down` /
    // `add_f64_up`), not the magnitude-sum + Higham-gamma charge used by
    // `widen_mccormick_constant` and its private twins. It is the right
    // idiom here for two reasons the other sites do not enjoy:
    //   1. EVERY SUMMAND IS EXACT. `ComposeResult::intercept_contrib` is
    //      `coeff as f64 * intercept as f64` — a product of two exact
    //      binary32 values, needing at most 48 significand bits against
    //      binary64's 53, and whose exponent cannot leave the binary64
    //      normal range (`2^-298 .. 2^254`). So the ONLY error in the
    //      whole fold is the addition, and charging each addition its
    //      own directed step is not merely sound but essentially tight:
    //      it costs `ulp64(partial)` per term, ~2^-52 relative, where a
    //      `gamma_d * SUM|summand|` charge pays the same order but must
    //      also carry a `1/(1 - gamma)` inflation for the magnitude
    //      sum's own accumulation deficit.
    //   2. PER-CELL STATE COMES FOR FREE, WHICH IS THE TRAP HERE. Each
    //      `[[b, j]]` cell is an INDEPENDENT fold, and the loop order is
    //      `b -> i -> j` with `j` INNERMOST: a cell is revisited once per
    //      `i`, interleaved with every other `j`. A magnitude-sum
    //      certificate would need auxiliary `[[b, j]]`-shaped
    //      magnitude/count arrays, and the natural — wrong —
    //      implementation hoists them into scalars, which either
    //      under-charges a dense cell or charges one cell's magnitude
    //      against another's. Keeping the accumulator IN the array makes
    //      the per-cell property structural rather than a thing to
    //      remember. (`matmul/crown_batched.rs` solves the same problem
    //      the other way, with per-cell locals, because its `d` loop is
    //      OUTERMOST and a cell's fold runs to completion inside it.)
    //
    // DIRECTION. `add_f64_down` returns `next_down_f64(acc + term)` and
    // `add_f64_up` returns `next_up_f64(acc + term)`, so by induction the
    // lower accumulator is always <= and the upper always >= the exact
    // real partial sum. `term == 0.0` short-circuits, which keeps a
    // relaxation with zero intercepts (a stable ReLU, an identity)
    // BIT-IDENTICAL to the historical output. Non-finite inputs fail
    // CLOSED (`-inf` / `+inf`).
    for b in 0..total_batch {
        for i in 0..in_dim {
            let l = pre_lower_flat[[b, i]];
            let u = pre_upper_flat[[b, i]];
            let relax = relaxation_fn(l, u, i);
            // Sign-flip two-line covers, summed OUTWARD (see `cover_sum_up`).
            // `int_sum` reaches the CONSTANT through `int_cover` and the
            // `*_b_err` widening with no further outward step of its own, so a
            // cover that rounded inward would under-count the widening. The dense
            // twin computes these with the identical helper.
            let slope_sum = cover_sum_up(relax.lower_slope, relax.upper_slope);
            let int_sum = cover_sum_up(relax.lower_intercept, relax.upper_intercept);
            // Direction-selected slopes actually used by compose_lower/compose_upper
            // (lower_slope for a>0, upper_slope for a<0, in the lower direction).
            let lr_slope = |a: f32| {
                if a > 0.0 {
                    relax.lower_slope as f64
                } else if a < 0.0 {
                    relax.upper_slope as f64
                } else {
                    0.0
                }
            };
            let ur_slope = |a: f32| {
                if a > 0.0 {
                    relax.upper_slope as f64
                } else if a < 0.0 {
                    relax.lower_slope as f64
                } else {
                    0.0
                }
            };

            for j in 0..out_dim {
                let la = lower_a_3d[[b, j, i]];
                let ua = upper_a_3d[[b, j, i]];
                let row_idx = b * out_dim + j;

                let lr = compose::compose_lower(la, &relax);
                new_lower_a[[row_idx, i]] = lr.new_coeff;
                new_lower_b[[b, j]] = add_f64_down(new_lower_b[[b, j]], lr.intercept_contrib);
                lower_nonfinite_rows[row_idx] |= lr.nonfinite;

                let ur = compose::compose_upper(ua, &relax);
                new_upper_a[[row_idx, i]] = ur.new_coeff;
                new_upper_b[[b, j]] = add_f64_up(new_upper_b[[b, j]], ur.intercept_contrib);
                upper_nonfinite_rows[row_idx] |= ur.nonfinite;

                // Always carry the fresh coefficient-product rounding gap, even
                // when the incoming bound had no error carrier. Otherwise a
                // first batched activation silently treats its stored f32
                // coefficient as the exact real product. Directed coefficient
                // rounding alone is not a box-independent enclosure: moving a
                // lower coefficient downward raises the affine value at negative
                // inputs (and conversely for an upper coefficient).
                {
                    let gap = if la != 0.0 {
                        (la as f64 * lr_slope(la) - lr.new_coeff as f64).abs()
                    } else {
                        0.0
                    };
                    let ea = in_lower_err_3d
                        .as_ref()
                        .map_or(0.0, |e| e[[b, j, i]] as f64);
                    if gap != 0.0 || ea != 0.0 {
                        let (slope_cover, int_cover) = if (la as f64).abs() > ea {
                            let slope = lr_slope(la).abs();
                            let intercept = if la > 0.0 {
                                relax.lower_intercept.abs() as f64
                            } else {
                                relax.upper_intercept.abs() as f64
                            };
                            (slope, intercept)
                        } else {
                            (slope_sum, int_sum)
                        };
                        new_lower_a_err[[row_idx, i]] =
                            next_up_f32((ea * slope_cover + gap) as f32);
                        if ea != 0.0 {
                            // The intercept-error fold is a NONNEGATIVE magnitude
                            // that is later subtracted from the lower constant, so
                            // it must round UP: under-stating it under-states the
                            // widening. One `add_f64_up` covers BOTH roundings here,
                            // not just the addition: `ea >= 0` (`set_coeff_err`
                            // sanitizes every entry to finite-nonnegative, and
                            // `validate_no_nan` rejects a negative carrier) and
                            // `int_cover >= 0` (an `abs`, or a sum of two), so the
                            // running `acc >= 0` and `acc + term >= term`. The
                            // product's round-to-nearest can fall short by at most
                            // `0.5*ulp(term)`, while `next_up_f64` adds a full
                            // `ulp(acc + term) >= ulp(term)`.
                            lower_b_err[[b, j]] = add_f64_up(lower_b_err[[b, j]], ea * int_cover);
                        }
                    }
                }
                {
                    let gap = if ua != 0.0 {
                        (ua as f64 * ur_slope(ua) - ur.new_coeff as f64).abs()
                    } else {
                        0.0
                    };
                    let ea = in_upper_err_3d
                        .as_ref()
                        .map_or(0.0, |e| e[[b, j, i]] as f64);
                    if gap != 0.0 || ea != 0.0 {
                        let (slope_cover, int_cover) = if (ua as f64).abs() > ea {
                            let slope = ur_slope(ua).abs();
                            let intercept = if ua > 0.0 {
                                relax.upper_intercept.abs() as f64
                            } else {
                                relax.lower_intercept.abs() as f64
                            };
                            (slope, intercept)
                        } else {
                            (slope_sum, int_sum)
                        };
                        new_upper_a_err[[row_idx, i]] =
                            next_up_f32((ea * slope_cover + gap) as f32);
                        if ea != 0.0 {
                            // Same nonnegative-magnitude fold as the lower arm; it
                            // is ADDED to the upper constant, so it rounds UP for
                            // the same reason.
                            upper_b_err[[b, j]] = add_f64_up(upper_b_err[[b, j]], ea * int_cover);
                        }
                    }
                }
            }
        }
    }

    // Fold the certified intercept error OUTWARD into the bias accumulators BEFORE
    // the directed cast (lower decreases, upper increases). The widening SUBTRACT
    // and ADD are themselves binary64 operations and round like any other, so they
    // take the same directed step as the accumulation above — otherwise the last
    // arithmetic op in the whole certificate would be the one nothing charges.
    // `-0.0 == 0.0` in IEEE 754, so a cell with no incoming error short-circuits
    // inside `add_f64_down`/`add_f64_up` and is left bit-identical.
    for b in 0..total_batch {
        for j in 0..out_dim {
            new_lower_b[[b, j]] = add_f64_down(new_lower_b[[b, j]], -lower_b_err[[b, j]]);
            new_upper_b[[b, j]] = add_f64_up(new_upper_b[[b, j]], upper_b_err[[b, j]]);
        }
    }

    // Non-finite row fallback for batched activation CROWN backward.
    let lower_affected = lower_nonfinite_rows.iter().filter(|&&r| r).count();
    let upper_affected = upper_nonfinite_rows.iter().filter(|&&r| r).count();
    compose::log_nonfinite_fallback(
        "Batched activation",
        lower_affected,
        upper_affected,
        total_rows,
    );

    // Convert bias to f32 with directed rounding, then apply row fallback. The
    // binary64 totals reaching this point have ALREADY absorbed their own
    // accumulation and widening error via the per-step directed adds above; this
    // cast only covers the narrowing, which is NOT sufficient on its own
    // (#activation-batched-const-fold). `publish_constant_down`/`_up` are
    // `next_down_f32(x as f32)` / `next_up_f32(x as f32)` bit for bit on every
    // finite in-range value, and additionally refuse to publish the `±inf` that a
    // saturating nearest-cast would otherwise hand back on the OPTIMISTIC side.
    let (new_lower_b_raw, _) = new_lower_b.into_raw_vec_and_offset();
    let mut new_lower_b_f32: Vec<f32> = new_lower_b_raw
        .into_iter()
        .map(publish_constant_down)
        .collect();
    let (new_upper_b_raw, _) = new_upper_b.into_raw_vec_and_offset();
    let mut new_upper_b_f32: Vec<f32> = new_upper_b_raw
        .into_iter()
        .map(publish_constant_up)
        .collect();

    for row_idx in 0..total_rows {
        let b = row_idx / out_dim;
        let j = row_idx % out_dim;
        let bias_idx = b * out_dim + j;
        if lower_nonfinite_rows[row_idx] {
            for i in 0..in_dim {
                new_lower_a[[row_idx, i]] = 0.0;
                new_lower_a_err[[row_idx, i]] = 0.0;
            }
            new_lower_b_f32[bias_idx] = f32::NEG_INFINITY;
        }
        if upper_nonfinite_rows[row_idx] {
            for i in 0..in_dim {
                new_upper_a[[row_idx, i]] = 0.0;
                new_upper_a_err[[row_idx, i]] = 0.0;
            }
            new_upper_b_f32[bias_idx] = f32::INFINITY;
        }
    }

    // Reshape back to original batch structure
    let (new_lower_a_vec, _) = new_lower_a.into_raw_vec_and_offset();
    let (new_upper_a_vec, _) = new_upper_a.into_raw_vec_and_offset();

    let out_a_shape: Vec<usize> = batch_dims
        .iter()
        .cloned()
        .chain([out_dim, in_dim])
        .collect();
    let out_b_shape: Vec<usize> = batch_dims.iter().cloned().chain([out_dim]).collect();

    // CROWN backward NaN firewall: shared path for ALL activation CROWN backward
    // (ReLU, Sigmoid, Exp, Log, SiLU, etc.). Falls back to conservative bounds instead of
    // aborting verification when any relaxation function produces NaN.
    let mut result = BatchedLinearBounds::new_or_conservative(
        ArrayD::from_shape_vec(IxDyn(&out_a_shape), new_lower_a_vec)
            .map_err(|_| NyError::InvalidSpec("Cannot reshape new_lower_a".to_string()))?,
        ArrayD::from_shape_vec(IxDyn(&out_b_shape), new_lower_b_f32)
            .map_err(|_| NyError::InvalidSpec("Cannot reshape new_lower_b".to_string()))?,
        ArrayD::from_shape_vec(IxDyn(&out_a_shape), new_upper_a_vec)
            .map_err(|_| NyError::InvalidSpec("Cannot reshape new_upper_a".to_string()))?,
        ArrayD::from_shape_vec(IxDyn(&out_b_shape), new_upper_b_f32)
            .map_err(|_| NyError::InvalidSpec("Cannot reshape new_upper_b".to_string()))?,
        bounds.input_shape.clone(),
        bounds.output_shape.clone(),
    )?;
    // Attach the propagated certified error (#vnncomp-aw-soundness). `set_coeff_err`
    // no-ops on shape mismatch (i.e. if the NaN firewall degraded to conservative),
    // which is sound: the conservative bounds already dominate any penalty.
    let le = ArrayD::from_shape_vec(
        IxDyn(&out_a_shape),
        new_lower_a_err.into_raw_vec_and_offset().0,
    )
    .map_err(|_| NyError::InvalidSpec("Cannot reshape new_lower_a_err".to_string()))?;
    let ue = ArrayD::from_shape_vec(
        IxDyn(&out_a_shape),
        new_upper_a_err.into_raw_vec_and_offset().0,
    )
    .map_err(|_| NyError::InvalidSpec("Cannot reshape new_upper_a_err".to_string()))?;
    result.set_coeff_err(le, ue);
    Ok(result)
}

#[cfg(test)]
mod batched_activation_constant_certificate_tests {
    use ndarray::{ArrayD, IxDyn};
    use ny_tensor::BoundedTensor;

    use super::crown_elementwise_backward_batched_indexed;
    use crate::layers::activations::LinearRelaxation;
    use crate::BatchedLinearBounds;

    /// The accumulated intercept CONSTANT of the batched element-wise activation
    /// backward must be published with its own certified error, in BOTH
    /// directions, in ONE call (#activation-batched-const-fold).
    ///
    /// COUNTEREXAMPLE. The activation is the EXACT affine function `f(x) = x + 1`
    /// on `x in [-1, 1]`, relaxed by the single line `y = x + 1` in both
    /// directions (`lower_slope = upper_slope = 1`,
    /// `lower_intercept = upper_intercept = 1`). Because both envelope lines
    /// coincide, `compose_lower`/`compose_upper` pick the same intercept whatever
    /// the coefficient's sign, so every summand of the constant fold is exactly
    /// the incoming coefficient:
    ///
    /// ```text
    /// intercept_contrib_i = a_i (as f64) * 1.0 = a_i,  exactly.
    /// ```
    ///
    /// One batch position, ONE output row, `K = 4098` inputs. The relaxation is
    /// exact and the incoming coefficients carry no error, so the ENTIRE
    /// soundness question reduces to the constant: the published lower constant
    /// must not exceed the exact fold, and the published upper constant must not
    /// fall below it.
    ///
    /// The taps are SIGN-SYMMETRIC across the two directions, which is what gives
    /// one call teeth in both. `lower_a` folds `+2^60`, 4096 x `-1`, `-2^60`;
    /// `upper_a` folds the exact negation. Every `2^60 -+ 1` rounds straight back
    /// to `2^60` — the binary64 ULP there is `2^8 = 256`, so a `1` is swallowed
    /// whole — hence BOTH plain-`+=` accumulators return EXACTLY `0.0` while the
    /// exact constants are `-4096` and `+4096`.
    ///
    /// Publishing those zeros through `next_down_f32(0.0)` / `next_up_f32(0.0)`
    /// yields `-1.4e-45` and `+1.4e-45`: the lower relaxation then sits `4096`
    /// ABOVE the true line at every feasible point and the upper sits `4096`
    /// BELOW it, which is a false-`unsat` in the graph alpha/beta BaB lane. A ULP
    /// of `0.0` is `1.4e-45`; the real error is `4096`. So the lower assertion and
    /// the upper assertion BOTH FAIL on the pre-fix code, and no single-direction
    /// repair can pass.
    ///
    /// The finiteness/looseness pins at the end stop the opposite failure: a
    /// certificate that "passes" by poisoning the row to `-inf`/`+inf`, or by
    /// charging wildly more than the fold's own rounding. The per-step directed
    /// certificate lands at exactly `-+524416.0625` here: all 4097 charged steps
    /// sit just BELOW `2^60`, in the binade whose binary64 ULP is `2^7 = 128`
    /// (`next_down` from `2^60` crosses into it immediately), so the charge is
    /// `4097 * 128 = 524416` plus the final publishing ULP. The `1e8` pin
    /// therefore leaves more than two orders of magnitude of headroom — enough
    /// for an equivalent magnitude-sum charge (`gamma_4099 * 2^61 ~ 1.05e6`) —
    /// while still catching a poisoned row.
    #[ntest::timeout(30000)]
    #[test]
    fn batched_activation_constant_carries_its_fold_error_in_both_directions() {
        const K: usize = 4098;
        // 2^60. Exactly representable in binary32 (exponent 60 << 127) and in
        // binary64, where its ULP is 2^(60-52) = 256.
        const BIG: f32 = 1_152_921_504_606_846_976.0;

        assert_eq!(BIG as f64, 2.0_f64.powi(60), "BIG must be exactly 2^60");

        // Index 0 = +2^60, indices 1..=4096 = -1, index 4097 = -2^60. The
        // production loop accumulates in ascending `i`, so this IS the fold order.
        let mut lower_a_vals = vec![-1.0_f32; K];
        lower_a_vals[0] = BIG;
        lower_a_vals[K - 1] = -BIG;
        let upper_a_vals: Vec<f32> = lower_a_vals.iter().map(|v| -v).collect();

        // Preconditions of the counterexample, pinned so a future change cannot
        // quietly turn this into a test of nothing.
        let exact_lower: i128 = lower_a_vals.iter().map(|&v| v as f64 as i128).sum();
        let exact_upper: i128 = upper_a_vals.iter().map(|&v| v as f64 as i128).sum();
        assert_eq!(exact_lower, -4096, "exact lower constant");
        assert_eq!(exact_upper, 4096, "exact upper constant");

        let mut folded_lower = 0.0_f64;
        let mut folded_upper = 0.0_f64;
        for (&lo, &up) in lower_a_vals.iter().zip(upper_a_vals.iter()) {
            folded_lower += lo as f64;
            folded_upper += up as f64;
        }
        assert_eq!(
            folded_lower, 0.0,
            "the plain f64 lower fold must cancel to 0 for this to exercise the defect"
        );
        assert_eq!(
            folded_upper, 0.0,
            "the plain f64 upper fold must cancel to 0 for this to exercise the defect"
        );

        let bounds = BatchedLinearBounds::new(
            ArrayD::from_shape_vec(IxDyn(&[1, K]), lower_a_vals).expect("lower_a shape"),
            ArrayD::from_elem(IxDyn(&[1]), 0.0_f32),
            ArrayD::from_shape_vec(IxDyn(&[1, K]), upper_a_vals).expect("upper_a shape"),
            ArrayD::from_elem(IxDyn(&[1]), 0.0_f32),
            vec![K],
            vec![1],
        )
        .expect("incoming bounds");

        let pre = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[1, K]), -1.0_f32),
            ArrayD::from_elem(IxDyn(&[1, K]), 1.0_f32),
        )
        .expect("pre-activation box");

        // f(x) = x + 1, relaxed exactly: both envelope lines are y = x + 1.
        let result = crown_elementwise_backward_batched_indexed(&bounds, &pre, |_l, _u, _i| {
            LinearRelaxation::new(1.0, 1.0, 1.0, 1.0)
        })
        .expect("batched activation CROWN backward");

        let published_lower = result.lower_b()[IxDyn(&[0])] as f64;
        let published_upper = result.upper_b()[IxDyn(&[0])] as f64;

        // FAILS PRE-FIX: the bare fold publishes ~0, which is 4096 too HIGH.
        assert!(
            published_lower <= -4096.0,
            "published LOWER constant {published_lower} must be <= -4096 — otherwise the \
             lower relaxation sits ABOVE the true line at every feasible point"
        );
        // FAILS PRE-FIX: the bare fold publishes ~0, which is 4096 too LOW.
        assert!(
            published_upper >= 4096.0,
            "published UPPER constant {published_upper} must be >= 4096 — otherwise the \
             upper relaxation sits BELOW the true line at every feasible point"
        );

        // The certificate must be proportional to the fold's own rounding, not a
        // blanket poisoning of the row.
        assert!(
            published_lower.is_finite() && published_lower >= -1.0e8,
            "LOWER constant {published_lower} must stay finite and proportionate"
        );
        assert!(
            published_upper.is_finite() && published_upper <= 1.0e8,
            "UPPER constant {published_upper} must stay finite and proportionate"
        );
    }
}
