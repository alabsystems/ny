// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Dense CROWN backward for element-wise activation functions.
//!
//! Operates on `Array2` A-matrices (output_neurons × input_neurons).

use ndarray::Array2;
use ny_core::dd::next_up_f64;
use ny_core::{NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

use super::compose;
use crate::layers::activations::LinearRelaxation;
use crate::layers::linear::bias::{add_f64_down, add_f64_up, f32_to_f64_exact};
use crate::LinearBounds;

/// Helper function for CROWN backward propagation through element-wise activation functions.
///
/// Delegates to [`crown_elementwise_backward_indexed`] with a no-op neuron index.
pub(crate) fn crown_elementwise_backward<F>(
    bounds: &LinearBounds,
    pre_activation: &BoundedTensor,
    relaxation_fn: F,
) -> Result<LinearBounds>
where
    F: Fn(f32, f32) -> LinearRelaxation,
{
    crown_elementwise_backward_indexed(bounds, pre_activation, |l, u, _i| relaxation_fn(l, u))
}

/// Indexed variant of CROWN backward for element-wise activations.
///
/// Like [`crown_elementwise_backward`], but the relaxation function receives the
/// neuron index `i` (column of the weight matrix). This supports layers where
/// the relaxation depends on a per-neuron parameter (e.g., PReLU's per-channel slopes).
///
/// # Arguments
/// * `bounds` - Incoming linear bounds from layers above
/// * `pre_activation` - Pre-activation bounds for this layer's inputs
/// * `relaxation_fn` - `(l, u, neuron_idx) -> LinearRelaxation`
pub fn crown_elementwise_backward_indexed<F>(
    bounds: &LinearBounds,
    pre_activation: &BoundedTensor,
    relaxation_fn: F,
) -> Result<LinearBounds>
where
    F: Fn(f32, f32, usize) -> LinearRelaxation,
{
    let pre_flat = pre_activation.flatten();
    let pre_lower = pre_flat
        .lower()
        .clone()
        .into_dimensionality::<ndarray::Ix1>()
        .map_err(|_| NyError::ShapeMismatch {
            expected: vec![pre_flat.len()],
            got: pre_flat.lower().shape().to_vec(),
        })?;
    let pre_upper = pre_flat
        .upper()
        .clone()
        .into_dimensionality::<ndarray::Ix1>()
        .map_err(|_| NyError::ShapeMismatch {
            expected: vec![pre_flat.len()],
            got: pre_flat.upper().shape().to_vec(),
        })?;

    let num_neurons = pre_lower.len();
    if bounds.num_inputs() != num_neurons {
        return Err(NyError::ShapeMismatch {
            expected: vec![num_neurons],
            got: vec![bounds.num_inputs()],
        });
    }

    let num_outputs = bounds.num_outputs();

    // Precompute per-neuron relaxation parameters
    let pre_lower_slice = pre_lower
        .as_slice()
        .ok_or_else(|| NyError::InternalError("Non-contiguous pre_lower array".into()))?;
    let pre_upper_slice = pre_upper
        .as_slice()
        .ok_or_else(|| NyError::InternalError("Non-contiguous pre_upper array".into()))?;
    let relaxations =
        compose::precompute_relaxations(pre_lower_slice, pre_upper_slice, &relaxation_fn);

    // Backward propagation: compose the linear relaxation.
    //
    // CERTIFIED CONSTANT (#vnncomp-aw-constant). Widening the bias accumulator
    // to binary64 does NOT certify the sum, and the comment that used
    // to stand here said only that it did. Higham Thm 3.1 bounds the binary64
    // error of a length-`d` sum by `gamma^f64_d * SUM|summand|` — scaled by the
    // MAGNITUDE of the terms — while the outward binary32 cast at publication
    // allows only `~2^-24 * |result|`. Under cancellation the result collapses
    // toward zero and that allowance collapses with it; the true error does
    // not. Worked shape, at `num_neurons = 4098`: `+2^60`, 4096 x `-1`, `-2^60`
    // sum to EXACTLY 0.0 in binary64 (ulp(2^60) = 256 swallows every `-1`)
    // where the truth is `-4096`, and the ULP of 0.0 is 1.4e-45. Published as a
    // lower bias that claims a bound 4096 ABOVE the truth — a false proof, on
    // the path every non-ReLU activation reaches a verdict through.
    //
    // The fix is idiom (b) of this crate's three constant-certificate idioms
    // (`layers/linear/bias.rs`, also used by `pooling/max{,_patches}.rs`,
    // `normalization/batch_norm/*` and this module's patches sibling): every
    // bias accumulation below takes a DIRECTED binary64 step after each
    // addition — `add_f64_down` on the lower accumulators, `add_f64_up` on the
    // upper and on the non-negative error accumulators — so each accumulator is
    // a certified bound on the exact real partial sum at every step, not merely
    // at the end.
    //
    // Why (b) rather than (a)'s magnitude-sum + gamma charge: each
    // `intercept_contrib` is an f32 x f32 product formed in f64, hence EXACT
    // (24 + 24 <= 53 significand bits), so the accumulation is the ONLY error
    // source and there is nothing else for a magnitude sum to cover. (b) is
    // strictly stronger — it bounds the error actually realized rather than its
    // worst case — and costs no second accumulator and no second pass.
    //
    // Seeds are decoded with `f32_to_f64_exact` rather than `as f64`: a plain
    // hardware conversion may observe a binary32-subnormal seed as signed zero
    // on a DAZ host, which for a lower bias reads `-1.4e-45` as `0.0` and moves
    // the bound INWARD.
    let mut new_lower_a = Array2::<f32>::zeros((num_outputs, num_neurons));
    let mut new_lower_b_f64 = bounds.lower_b().mapv(f32_to_f64_exact);
    let mut new_upper_a = Array2::<f32>::zeros((num_outputs, num_neurons));
    let mut new_upper_b_f64 = bounds.upper_b().mapv(f32_to_f64_exact);

    // Certified coefficient-error propagation through the relaxation
    // (#vnncomp-aw-soundness). For input coefficient `a` with certified error
    // `err_a`, the composed coefficient `a·slope(sign a)` has certified error
    // `err_a·(|lower_slope| + |upper_slope|) + directed-rounding-gap` (the
    // slope-sum term covers a possible sign-flip of `a` selecting the other
    // envelope), and the intercept contribution picks up
    // `err_a·(|lower_intercept| + |upper_intercept|)` folded into the bias
    // widening. Validated at 0 violations / 6M trials.
    //
    // SOUNDNESS (#vnncomp-aw-activation): the composed coefficient is the
    // directed-rounded f32 of the f32 product `a·slope`, which is NOT exactly
    // the true real coefficient `a_f64·slope_f64`. The `next_down_f32`/
    // `next_up_f32` in `compose_*` only makes the stored coefficient sound for
    // concretizing AT THIS layer's input box; once this activation's output
    // bounds are composed by a FURTHER backward layer, that layer reads the
    // stored f32 coefficient and (absent a certified error) treats it as exact,
    // dropping the directed-rounding gap and UNDER-counting the true error =
    // false-proof risk. So we ALWAYS carry the per-coefficient
    // directed-rounding gap `|a_f64·slope_f64 − stored_f32|` in the error
    // matrices, regardless of whether the incoming bounds carry error, and add
    // the propagated incoming-error term only when present.
    let in_lower_err = bounds.lower_a_err();
    let in_upper_err = bounds.upper_a_err();
    let mut new_lower_a_err = Array2::<f32>::zeros((num_outputs, num_neurons));
    let mut new_upper_a_err = Array2::<f32>::zeros((num_outputs, num_neurons));
    // Per-row certified intercept (bias) error, subtracted from lower / added
    // to upper. These are non-negative charge accumulators, so BOTH are summed
    // with `add_f64_up` (#vnncomp-aw-constant): a round-to-nearest `+=` here
    // UNDER-counts the charge — the same Higham/ULP mismatch as the bias sums,
    // in the direction that shrinks a widening — and an under-counted widening
    // is a false proof exactly like an over-tight bias.
    let mut lower_b_err = vec![0.0f64; num_outputs];
    let mut upper_b_err = vec![0.0f64; num_outputs];

    // Track which output rows have non-finite coefficients after coeff × slope.
    let mut lower_nonfinite_rows = vec![false; num_outputs];
    let mut upper_nonfinite_rows = vec![false; num_outputs];

    // Parallel row lane (#conv-fill-hoist leg: `[walk-share] act_bwd` was
    // 6.9 s of the biasfield collection). Output rows are fully independent —
    // every accumulator (bias f64 sums, intercept-error sums, non-finite
    // flags) is per-row — so rows fan out across threads with the inner
    // i-order, and therefore every f64 accumulation order, unchanged: values
    // are byte-identical to the serial loop. Contiguity is guarded; any
    // strided input keeps the serial path below.
    let can_par = bounds.lower_a().as_slice().is_some()
        && bounds.upper_a().as_slice().is_some()
        && in_lower_err.is_none_or(|e| e.as_slice().is_some())
        && in_upper_err.is_none_or(|e| e.as_slice().is_some());
    if can_par {
        use rayon::prelude::*;
        struct RowState {
            lb: f64,
            ub: f64,
            lbe: f64,
            ube: f64,
            lnf: bool,
            unf: bool,
        }
        let la_flat = bounds.lower_a().as_slice().expect("checked contiguous");
        let ua_flat = bounds.upper_a().as_slice().expect("checked contiguous");
        let le_flat = in_lower_err.map(|e| e.as_slice().expect("checked contiguous"));
        let ue_flat = in_upper_err.map(|e| e.as_slice().expect("checked contiguous"));
        let mut row_states: Vec<RowState> = (0..num_outputs)
            .map(|j| RowState {
                lb: new_lower_b_f64[j],
                ub: new_upper_b_f64[j],
                lbe: 0.0,
                ube: 0.0,
                lnf: false,
                unf: false,
            })
            .collect();
        let nla_flat = new_lower_a.as_slice_mut().expect("fresh row-major");
        let nua_flat = new_upper_a.as_slice_mut().expect("fresh row-major");
        let nle_flat = new_lower_a_err.as_slice_mut().expect("fresh row-major");
        let nue_flat = new_upper_a_err.as_slice_mut().expect("fresh row-major");
        nla_flat
            .par_chunks_mut(num_neurons)
            .zip(nua_flat.par_chunks_mut(num_neurons))
            .zip(nle_flat.par_chunks_mut(num_neurons))
            .zip(nue_flat.par_chunks_mut(num_neurons))
            .zip(row_states.par_iter_mut())
            .enumerate()
            .for_each(|(j, ((((nla, nua), nle), nue), st))| {
                let la_row = &la_flat[j * num_neurons..(j + 1) * num_neurons];
                let ua_row = &ua_flat[j * num_neurons..(j + 1) * num_neurons];
                let le_row = le_flat.map(|e| &e[j * num_neurons..(j + 1) * num_neurons]);
                let ue_row = ue_flat.map(|e| &e[j * num_neurons..(j + 1) * num_neurons]);
                for i in 0..num_neurons {
                    let la = la_row[i];
                    let ua = ua_row[i];
                    let relax = &relaxations[i];

                    let lr = compose::compose_lower(la, relax);
                    nla[i] = lr.new_coeff;
                    // Certified accumulation, byte-for-byte the serial arm's
                    // (#vnncomp-aw-constant). A rayon arm that charges
                    // differently from its serial twin is the same asymmetry
                    // one level down.
                    st.lb = add_f64_down(st.lb, lr.intercept_contrib);
                    st.lnf |= lr.nonfinite;

                    let ur = compose::compose_upper(ua, relax);
                    nua[i] = ur.new_coeff;
                    st.ub = add_f64_up(st.ub, ur.intercept_contrib);
                    st.unf |= ur.nonfinite;

                    // Outward two-line covers (see `cover_sum_up`); the serial
                    // arm computes the identical values.
                    let slope_sum = cover_sum_up(relax.lower_slope, relax.upper_slope);
                    let int_sum = cover_sum_up(relax.lower_intercept, relax.upper_intercept);

                    // Lower direction (same certification as the serial loop).
                    {
                        let gap = if la != 0.0 {
                            (la as f64 * lr_slope(la, relax) - lr.new_coeff as f64).abs()
                        } else {
                            0.0
                        };
                        let ea = le_row.map_or(0.0, |r| r[i] as f64);
                        if gap != 0.0 || ea != 0.0 {
                            let (slope_cover, int_cover) = if (la as f64).abs() > ea {
                                let slope = lr_slope(la, relax).abs();
                                let intercept = if la > 0.0 {
                                    relax.lower_intercept.abs() as f64
                                } else {
                                    relax.upper_intercept.abs() as f64
                                };
                                (slope, intercept)
                            } else {
                                (slope_sum, int_sum)
                            };
                            nle[i] = next_up_f32((ea * slope_cover + gap) as f32);
                            if ea != 0.0 {
                                st.lbe = add_f64_up(st.lbe, ea * int_cover);
                            }
                        }
                    }

                    // Upper direction.
                    {
                        let gap = if ua != 0.0 {
                            (ua as f64 * ur_slope(ua, relax) - ur.new_coeff as f64).abs()
                        } else {
                            0.0
                        };
                        let ea = ue_row.map_or(0.0, |r| r[i] as f64);
                        if gap != 0.0 || ea != 0.0 {
                            let (slope_cover, int_cover) = if (ua as f64).abs() > ea {
                                let slope = ur_slope(ua, relax).abs();
                                let intercept = if ua > 0.0 {
                                    relax.upper_intercept.abs() as f64
                                } else {
                                    relax.lower_intercept.abs() as f64
                                };
                                (slope, intercept)
                            } else {
                                (slope_sum, int_sum)
                            };
                            nue[i] = next_up_f32((ea * slope_cover + gap) as f32);
                            if ea != 0.0 {
                                st.ube = add_f64_up(st.ube, ea * int_cover);
                            }
                        }
                    }
                }
                // Same end-of-row fold as the serial loop, and equally
                // directed: the discharge is itself a binary64 addition, and it
                // is the one most likely to cancel — `-st.lbe` is subtracted
                // from a bias of arbitrary sign and magnitude.
                st.lb = add_f64_down(st.lb, -st.lbe);
                st.ub = add_f64_up(st.ub, st.ube);
            });
        for (j, st) in row_states.iter().enumerate() {
            new_lower_b_f64[j] = st.lb;
            new_upper_b_f64[j] = st.ub;
            lower_nonfinite_rows[j] = st.lnf;
            upper_nonfinite_rows[j] = st.unf;
        }
    } else {
        for j in 0..num_outputs {
            for i in 0..num_neurons {
                let la = bounds.lower_a()[[j, i]];
                let ua = bounds.upper_a()[[j, i]];

                let lr = compose::compose_lower(la, &relaxations[i]);
                new_lower_a[[j, i]] = lr.new_coeff;
                // Certified accumulation (#vnncomp-aw-constant); see the
                // rationale above the accumulator declarations.
                new_lower_b_f64[j] = add_f64_down(new_lower_b_f64[j], lr.intercept_contrib);
                lower_nonfinite_rows[j] |= lr.nonfinite;

                let ur = compose::compose_upper(ua, &relaxations[i]);
                new_upper_a[[j, i]] = ur.new_coeff;
                new_upper_b_f64[j] = add_f64_up(new_upper_b_f64[j], ur.intercept_contrib);
                upper_nonfinite_rows[j] |= ur.nonfinite;

                // ALWAYS certify the per-coefficient error (#vnncomp-aw-activation).
                // Two independent contributions:
                //   1. directed-rounding gap = |a_f64·slope_used_f64 − stored_f32|,
                //      the exact distance between the f32 composed coefficient and
                //      the real product. Present even with NO incoming error.
                //   2. incoming-error term, with a SIGN-STABILITY refinement
                //      (#cgan-conv-err-compose, the fix named in
                //      docs/CGAN_BOUND_QUALITY_ROOT_CAUSE.md): the true incoming
                //      coefficient lies in `[a−e, a+e]`.
                //      - When `|a| > e`, the true coefficient has the SAME SIGN as
                //        `a`, so the envelope line selected by `compose_*` (chosen by
                //        sign) is the valid sound substitute for the true coefficient
                //        too: `err_out = e·|slope_chosen| + gap` and the bias picks up
                //        only `e·|intercept_chosen|` (the same line's intercept scaled
                //        by the coefficient uncertainty). No doubling: for ReLU the
                //        chosen slope is ≤ 1, so carried error can only SHRINK here —
                //        this kills the 2^L growth across L stable-identity ReLU
                //        layers that previously made the carried error reach
                //        intermediate-box scale on deep conv stacks.
                //      - When `|a| ≤ e`, the sign may flip and select the other
                //        envelope: keep the unconditional two-line cover
                //        `e·(|lower_slope|+|upper_slope|)` + `e·(|i_l|+|i_u|)` into
                //        the bias. This branch now also covers `a == 0` with `e > 0`
                //        (previously the incoming error was silently DROPPED for
                //        exact-zero coefficients — an unsound hole in the
                //        false-proof direction).
                let relax = &relaxations[i];
                // Outward two-line covers (see `cover_sum_up`).
                let slope_sum = cover_sum_up(relax.lower_slope, relax.upper_slope);
                let int_sum = cover_sum_up(relax.lower_intercept, relax.upper_intercept);

                // Lower direction.
                {
                    let gap = if la != 0.0 {
                        (la as f64 * lr_slope(la, relax) - lr.new_coeff as f64).abs()
                    } else {
                        0.0
                    };
                    let ea = in_lower_err.map_or(0.0, |e| e[[j, i]] as f64);
                    if gap != 0.0 || ea != 0.0 {
                        let (slope_cover, int_cover) = if (la as f64).abs() > ea {
                            // Sign-stable: the chosen line covers the true coefficient.
                            let slope = lr_slope(la, relax).abs();
                            let intercept = if la > 0.0 {
                                relax.lower_intercept.abs() as f64
                            } else {
                                relax.upper_intercept.abs() as f64
                            };
                            (slope, intercept)
                        } else {
                            (slope_sum, int_sum)
                        };
                        new_lower_a_err[[j, i]] = next_up_f32((ea * slope_cover + gap) as f32);
                        if ea != 0.0 {
                            lower_b_err[j] = add_f64_up(lower_b_err[j], ea * int_cover);
                        }
                    }
                }

                // Upper direction.
                {
                    let gap = if ua != 0.0 {
                        (ua as f64 * ur_slope(ua, relax) - ur.new_coeff as f64).abs()
                    } else {
                        0.0
                    };
                    let ea = in_upper_err.map_or(0.0, |e| e[[j, i]] as f64);
                    if gap != 0.0 || ea != 0.0 {
                        let (slope_cover, int_cover) = if (ua as f64).abs() > ea {
                            // Sign-stable: the chosen line covers the true coefficient.
                            let slope = ur_slope(ua, relax).abs();
                            let intercept = if ua > 0.0 {
                                relax.upper_intercept.abs() as f64
                            } else {
                                relax.lower_intercept.abs() as f64
                            };
                            (slope, intercept)
                        } else {
                            (slope_sum, int_sum)
                        };
                        new_upper_a_err[[j, i]] = next_up_f32((ea * slope_cover + gap) as f32);
                        if ea != 0.0 {
                            upper_b_err[j] = add_f64_up(upper_b_err[j], ea * int_cover);
                        }
                    }
                }
            }
            // Fold the certified intercept error into the bias accumulators
            // BEFORE the directed cast: lower decreases, upper increases. The
            // fold is a binary64 addition like any other, so it takes the same
            // directed step (#vnncomp-aw-constant).
            new_lower_b_f64[j] = add_f64_down(new_lower_b_f64[j], -lower_b_err[j]);
            new_upper_b_f64[j] = add_f64_up(new_upper_b_f64[j], upper_b_err[j]);
        }
    }

    // Non-finite row fallback — zero A-row, set bias to ±Inf (sound but maximally loose).
    let lower_affected = lower_nonfinite_rows.iter().filter(|&&r| r).count();
    let upper_affected = upper_nonfinite_rows.iter().filter(|&&r| r).count();
    compose::log_nonfinite_fallback("Activation", lower_affected, upper_affected, num_outputs);
    // Outward-round the f64 bias into f32 — preserving EXACT ZERO. An
    // unconditional `next_down_f32(0.0)` manufactures a -1e-45 bias on rows
    // whose true bias is exactly 0.0: a needless 1-ulp loosening, and a
    // violation of the stacked-seed invariant that zero rows stay exactly
    // zero through every admitted step (#cgan-stacked-backward injection
    // guard refuses the whole pass on a nonzero find).
    //
    // CORRECTED RATIONALE (#vnncomp-aw-constant). This comment used to end
    // "`x == 0.0` casts exactly, so no widening is required there." That claim
    // is true of the CAST and was FALSE of the ACCUMULATION that produced `x`,
    // and `x == 0.0` is precisely the CANCELLING case: the case where the
    // binary32 allowance `~2^-24 * |x|` collapses to 1.4e-45 while the true
    // binary64 accumulation error does not collapse at all. Under the previous
    // round-to-nearest `+=`, a row whose exact bias was `-4096` could reach
    // exactly 0.0 (see the accumulation comment above) and this branch then
    // published a lower bias 4096 ABOVE the truth.
    //
    // The branch is sound now only because every accumulator above takes a
    // directed binary64 step per addition. `add_f64_down`/`add_f64_up`
    // short-circuit `term == 0.0` exactly, and otherwise step strictly outward,
    // so a LOWER accumulator can hold exactly `+0.0` in only two ways: every
    // term it saw was exactly zero (the structural zero row), or it stepped
    // down from `+5e-324`, whose true sum is therefore >= 0.0. Both make `0.0`
    // a sound lower bound. Symmetrically for the upper accumulator and `-0.0`.
    // A row whose NONZERO terms merely cancel no longer lands on zero at all:
    // it ends strictly outside and is published one binary32 ulp further out.
    //
    // The invariant the injection guard checks is therefore untouched.
    // `inject_stacked_identity_block` refuses unless a seed row's coefficients,
    // bias AND coefficient-error are all exactly zero; a row with all-zero
    // incoming coefficients composes to `ComposeResult::ZERO`, contributes
    // exactly 0.0 to every accumulator (short-circuited, never stepped), skips
    // the error block entirely (`gap == 0.0 && ea == 0.0`), and still publishes
    // an exact `+0.0` bias with a 0.0 error.
    let mut new_lower_b = new_lower_b_f64.mapv(|x| {
        if x == 0.0 {
            0.0
        } else {
            next_down_f32(x as f32)
        }
    });
    let mut new_upper_b =
        new_upper_b_f64.mapv(|x| if x == 0.0 { 0.0 } else { next_up_f32(x as f32) });
    for j in 0..num_outputs {
        if lower_nonfinite_rows[j] {
            for i in 0..num_neurons {
                new_lower_a[[j, i]] = 0.0;
                new_lower_a_err[[j, i]] = 0.0;
            }
            new_lower_b[j] = f32::NEG_INFINITY;
        }
        if upper_nonfinite_rows[j] {
            for i in 0..num_neurons {
                new_upper_a[[j, i]] = 0.0;
                new_upper_a_err[[j, i]] = 0.0;
            }
            new_upper_b[j] = f32::INFINITY;
        }
    }

    // CROWN backward NaN firewall: falls back to conservative bounds instead
    // of aborting the verification chain when relaxation produces NaN/Inf coefficients.
    // Non-finite rows already zeroed above, but missed corruption is caught here.
    //
    // We ALWAYS attach the certified coefficient error (#vnncomp-aw-activation):
    // the composed f32 coefficient differs from the true real product by the
    // directed-rounding gap, which a downstream backward layer must not drop.
    LinearBounds::new_or_conservative_with_err(
        new_lower_a,
        new_lower_b,
        new_upper_a,
        new_upper_b,
        new_lower_a_err,
        new_upper_a_err,
    )
}

/// Outward `|a| + |b|` for the sign-flip two-line cover (#vnncomp-aw-constant).
///
/// The bare form `(a.abs() + b.abs()) as f64` performs the addition in
/// BINARY32 under round-to-nearest, so it can land BELOW the true sum. That
/// cover multiplies the incoming coefficient error into both published halves —
/// `new_*_a_err` and, through `int_cover`, the bias widening `*_b_err` — so a
/// cover rounded the wrong way under-counts BOTH, in the false-proof direction.
/// Converting exactly first also keeps a binary32-subnormal operand alive on a
/// DAZ host, where a hardware `as f64` may read it as zero, and removes a
/// spurious binary32 overflow to `+inf` for two near-`f32::MAX` intercepts.
///
/// Non-finite operands propagate: `+inf` stays `+inf`, NaN stays NaN and is
/// sanitized to `+inf` by `new_or_conservative_with_err` — both vacuous, sound.
#[inline]
fn cover_sum_up(a: f32, b: f32) -> f64 {
    let sum = f32_to_f64_exact(a).abs() + f32_to_f64_exact(b).abs();
    if sum == 0.0 {
        0.0
    } else {
        next_up_f64(sum)
    }
}

/// The lower-direction slope actually used by [`compose::compose_lower`] for a
/// coefficient of the given sign (lower_slope for `a>0`, upper_slope for `a<0`).
#[inline]
fn lr_slope(a: f32, relax: &LinearRelaxation) -> f64 {
    if a > 0.0 {
        relax.lower_slope as f64
    } else if a < 0.0 {
        relax.upper_slope as f64
    } else {
        0.0
    }
}

/// The upper-direction slope actually used by [`compose::compose_upper`].
#[inline]
fn ur_slope(a: f32, relax: &LinearRelaxation) -> f64 {
    if a > 0.0 {
        relax.upper_slope as f64
    } else if a < 0.0 {
        relax.lower_slope as f64
    } else {
        0.0
    }
}

/// Cancelling-constant certificate for the dense element-wise CROWN backward
/// (#vnncomp-aw-constant).
///
/// These tests FAIL on the pre-fix code, where both bias accumulators were a
/// bare round-to-nearest binary64 `+=` and the publication branch preserved the
/// resulting exact zero. They are named individually so a filter that stops
/// matching them fails loudly rather than silently passing zero tests.
#[cfg(test)]
mod certified_constant_tests {
    use super::*;
    use ndarray::{Array1, Array2};

    /// `2^60`. `ulp(2^60) = 256`, so a round-to-nearest binary64 accumulator
    /// sitting at `2^60` swallows every `±1` added to it, exactly.
    const BIG: f32 = 1_152_921_504_606_846_976.0;
    /// How many unit terms vanish into `BIG`.
    const UNITS: usize = 4096;
    /// One `+BIG`, `UNITS` unit terms, one `-BIG`.
    const N: usize = UNITS + 2;
    /// Row 0 carries the cancelling shape; row 1 is a structural zero row.
    const ROWS: usize = 2;
    /// Exact real sum of row 0's lower-line intercepts: `2^60 - 4096 - 2^60`.
    const TRUE_LOWER_SUM: f32 = -4096.0;
    /// Exact real sum of row 0's upper-line intercepts (the mirror image).
    const TRUE_UPPER_SUM: f32 = 4096.0;

    /// Zero slopes, cancelling intercepts. The lower line's intercepts sum
    /// EXACTLY to `-4096` and the upper line's to `+4096`, so one call violates
    /// BOTH directions: the lower bias is published above its truth and the
    /// upper bias below its truth.
    fn cancelling_relaxation(_l: f32, _u: f32, i: usize) -> LinearRelaxation {
        let lower_intercept = if i == 0 {
            BIG
        } else if i == N - 1 {
            -BIG
        } else {
            -1.0
        };
        LinearRelaxation::new(0.0, lower_intercept, 0.0, -lower_intercept)
    }

    /// Row 0 all-ones (so `intercept_contrib == intercept`), row 1 all-zero.
    ///
    /// `contiguous == false` builds the SAME values in a transposed layout, so
    /// `as_slice()` is `None` and the `can_par` guard takes the serial arm.
    /// That is this test's own observable for which arm ran.
    fn coeff_matrix(contiguous: bool) -> Array2<f32> {
        if contiguous {
            let mut data = vec![0.0f32; ROWS * N];
            data[..N].fill(1.0);
            Array2::from_shape_vec((ROWS, N), data).expect("ROWS x N data length")
        } else {
            let mut data = Vec::with_capacity(ROWS * N);
            for _ in 0..N {
                data.push(1.0f32);
                data.push(0.0f32);
            }
            Array2::from_shape_vec((N, ROWS), data)
                .expect("N x ROWS data length")
                .reversed_axes()
        }
    }

    /// Run one cancelling-shape backward and return the published biases.
    fn run(contiguous: bool) -> (Array1<f32>, Array1<f32>) {
        let lower_a = coeff_matrix(contiguous);
        let upper_a = coeff_matrix(contiguous);
        assert_eq!(
            lower_a.as_slice().is_some(),
            contiguous,
            "test setup: contiguity decides which arm `can_par` selects, so it \
             must be what this case intended"
        );
        let bounds = LinearBounds::new(lower_a, Array1::zeros(ROWS), upper_a, Array1::zeros(ROWS))
            .expect("finite coefficients and biases");
        let pre = BoundedTensor::new(
            Array1::from_elem(N, -1.0f32).into_dyn(),
            Array1::from_elem(N, 1.0f32).into_dyn(),
        )
        .expect("matching 1-D pre-activation shapes");
        let out = crown_elementwise_backward_indexed(&bounds, &pre, cancelling_relaxation)
            .expect("finite relaxation must not degrade to conservative");
        (out.lower_b().clone(), out.upper_b().clone())
    }

    /// Both directions, one call. Widening to binary64 is not a certificate:
    /// the exact bias is `∓4096` and a bare `+=` lands on exactly `0.0`.
    fn assert_certified(lower_b: &Array1<f32>, upper_b: &Array1<f32>, arm: &str) {
        assert!(
            lower_b[0] <= TRUE_LOWER_SUM,
            "{arm}: lower bias {} is ABOVE its exact value {TRUE_LOWER_SUM} — a \
             false proof by {} (cancellation ate the binary64 accumulation error \
             and the binary32 ULP of the result cannot pay for it)",
            lower_b[0],
            lower_b[0] - TRUE_LOWER_SUM
        );
        assert!(
            upper_b[0] >= TRUE_UPPER_SUM,
            "{arm}: upper bias {} is BELOW its exact value {TRUE_UPPER_SUM} — a \
             false proof by {}",
            upper_b[0],
            TRUE_UPPER_SUM - upper_b[0]
        );
        assert!(
            lower_b[0].is_finite() && upper_b[0].is_finite(),
            "{arm}: the certificate must loosen the bias, not vacate the row: \
             lower={} upper={}",
            lower_b[0],
            upper_b[0]
        );
        // The structural zero row must still publish an exact +0.0 in both
        // directions — the invariant `inject_stacked_identity_block` refuses on.
        assert_eq!(
            lower_b[1].to_bits(),
            0,
            "{arm}: an all-zero-coefficient row must keep an exact +0.0 lower \
             bias, got {}",
            lower_b[1]
        );
        assert_eq!(
            upper_b[1].to_bits(),
            0,
            "{arm}: an all-zero-coefficient row must keep an exact +0.0 upper \
             bias, got {}",
            upper_b[1]
        );
    }

    #[test]
    fn cancelling_constants_are_certified_in_both_directions_parallel_arm() {
        let (lower_b, upper_b) = run(true);
        assert_certified(&lower_b, &upper_b, "parallel arm");
    }

    #[test]
    fn cancelling_constants_are_certified_in_both_directions_serial_arm() {
        let (lower_b, upper_b) = run(false);
        assert_certified(&lower_b, &upper_b, "serial arm");
    }

    /// A rayon arm that charges differently from its serial twin is the same
    /// asymmetry one level down. Both arms keep the inner `i` order, so the
    /// directed accumulation must be byte-identical, not merely both-sound.
    #[test]
    fn serial_and_parallel_arms_publish_bit_identical_certified_constants() {
        let (par_lower, par_upper) = run(true);
        let (ser_lower, ser_upper) = run(false);
        for j in 0..ROWS {
            assert_eq!(
                par_lower[j].to_bits(),
                ser_lower[j].to_bits(),
                "row {j}: parallel lower bias {} != serial lower bias {}",
                par_lower[j],
                ser_lower[j]
            );
            assert_eq!(
                par_upper[j].to_bits(),
                ser_upper[j].to_bits(),
                "row {j}: parallel upper bias {} != serial upper bias {}",
                par_upper[j],
                ser_upper[j]
            );
        }
    }
}
