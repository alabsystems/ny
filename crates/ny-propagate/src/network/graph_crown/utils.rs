// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::bounds::patches::CrownBounds;
#[cfg(test)]
use crate::bounds::patches_batched::BatchedCrownBounds;
use crate::bounds::{safe_mul_for_bounds_f64, BatchedLinearBounds, LinearBounds};

use ndarray::{Array, Array1, ArrayBase, ArrayView1, Data, Dimension, Zip};
use ny_core::{NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

use super::super::core::{GraphNetwork, NETWORK_INPUT};
use crate::network::CrownMergeAccumulator;

impl GraphNetwork {
    /// Fallback for spec-guided CROWN: compute spec bounds from IBP output bounds.
    // Generic over the map value (`BoundedTensor` or `Arc<BoundedTensor>` via
    // `Borrow`): the batched BaB caches are `Arc`-shared (#cone-delta
    // increment 2) while the graph-CROWN spec lanes pass owned maps.
    pub(crate) fn propagate_crown_with_specs_fallback_ibp<V: std::borrow::Borrow<BoundedTensor>>(
        &self,
        _input: &BoundedTensor,
        spec_matrix: &ndarray::Array2<f32>,
        node_bounds: &std::collections::HashMap<String, V>,
        output_node_name: &str,
    ) -> Result<BoundedTensor> {
        let output_bounds = node_bounds
            .get(output_node_name)
            .map(std::borrow::Borrow::borrow)
            .ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "Output node {} not found for IBP fallback",
                    output_node_name
                ))
            })?;

        let sanitized = Self::sanitize_bounds_for_fallback(output_bounds);
        let flat = sanitized.flatten();
        let lower_values: Vec<f32> = flat.lower().iter().copied().collect();
        let upper_values: Vec<f32> = flat.upper().iter().copied().collect();
        let num_specs = spec_matrix.nrows();
        let mut lower = Array1::<f32>::zeros(num_specs);
        let mut upper = Array1::<f32>::zeros(num_specs);

        for (i, spec_row) in spec_matrix.rows().into_iter().enumerate() {
            let (l, u) = Self::spec_row_interval_bounds(spec_row, &lower_values, &upper_values);
            lower[i] = l;
            upper[i] = u;
        }

        // Inf bounds are sound (conservative); NaN has been replaced above.
        BoundedTensor::new_allow_infinite(lower.into_dyn(), upper.into_dyn())
    }

    /// Reduce one spec row against an output box with interval arithmetic, rounding
    /// the returned endpoints OUTWARD.
    ///
    /// Sign-split rule: a coefficient `c >= 0` pairs with the box's lower endpoint for
    /// the lower bound and its upper endpoint for the upper bound; `c < 0` swaps them.
    /// Accumulated in f64 and closed with a directed f32 cast, mirroring
    /// [`LinearBounds::concretize_sound`].
    ///
    /// SOUNDNESS (#concretize-soundness-hardening): reducing a row with plain f32
    /// multiply-accumulate rounds to nearest at every product and every add, so the
    /// returned `lower` can land ABOVE the row's true minimum over the box (and `upper`
    /// below its true maximum) by up to `gamma_n * sum |c_j * y_j|` — a bias that
    /// cancellation makes unbounded relative to the result. The verdict consumers of
    /// this reduction (`domain_is_verified_for_mode`, `disjunctive_domain_verified`)
    /// compare against their threshold with no error budget and then permanently prune
    /// the domain, so an inward endpoint is undetectable and terminal. `f32 x f32`
    /// promoted to f64 is exact (48 < 53 significand bits), leaving only the f64
    /// accumulation for the one-ULP directed cast to cover.
    ///
    /// `safe_mul_for_bounds_f64` keeps the interval convention that a zero coefficient
    /// contributes exactly 0 even against a saturated (infinite) endpoint, so one Inf
    /// output cannot poison the whole row (#3044). A NaN accumulator (0*Inf with a NaN
    /// coefficient, or Inf-Inf) collapses that side to its saturating bound: NaN lower
    /// → -Inf, NaN upper → +Inf, both sound over-approximations.
    ///
    /// REQUIRES: `lower_values.len() == upper_values.len() >= spec_row.len()`.
    /// ENSURES: `lower <= c^T y <= upper` for every `y` in the box.
    pub(crate) fn spec_row_interval_bounds(
        spec_row: ArrayView1<'_, f32>,
        lower_values: &[f32],
        upper_values: &[f32],
    ) -> (f32, f32) {
        let mut l = 0.0f64;
        let mut u = 0.0f64;
        for (j, &c) in spec_row.iter().enumerate() {
            let c = c as f64;
            let input_l = lower_values[j] as f64;
            let input_u = upper_values[j] as f64;
            if c >= 0.0 {
                l += safe_mul_for_bounds_f64(c, input_l);
                u += safe_mul_for_bounds_f64(c, input_u);
            } else {
                l += safe_mul_for_bounds_f64(c, input_u);
                u += safe_mul_for_bounds_f64(c, input_l);
            }
        }

        let lower = if l.is_nan() {
            f32::NEG_INFINITY
        } else {
            next_down_f32(l as f32)
        };
        let upper = if u.is_nan() {
            f32::INFINITY
        } else {
            next_up_f32(u as f32)
        };
        (lower, upper)
    }

    /// Accumulate CrownBounds for a given input node (Phase 1b, #2613).
    ///
    /// On first insertion, preserves the CrownBounds variant (Patches stays Patches
    /// so the next layer backward can operate on sparse structure). On subsequent
    /// accumulations (merge points), delegates to `merge_crown` which preserves
    /// compatible Patches carriers in-place and falls back to Dense+f64 for
    /// incompatible or mixed carriers (#4382).
    pub(crate) fn accumulate_crown_bounds_to_input(
        &self,
        input_name: &str,
        new_bounds: CrownBounds,
        node_crown_bounds: &mut CrownMergeAccumulator,
        _output_dim: usize,
        _input_dim: usize,
        input_accumulated: &mut bool,
    ) -> Result<()> {
        let is_network_input = input_name == NETWORK_INPUT;
        let is_first = if is_network_input {
            !*input_accumulated
        } else {
            !node_crown_bounds.contains_key(input_name)
        };

        if is_first {
            // First insertion: preserve Patches if applicable.
            if is_network_input {
                node_crown_bounds.insert(NETWORK_INPUT.to_string(), new_bounds);
                *input_accumulated = true;
            } else {
                node_crown_bounds.insert(input_name.to_string(), new_bounds);
            }
        } else {
            let key = if is_network_input {
                NETWORK_INPUT
            } else {
                input_name
            };
            node_crown_bounds.merge_crown(key, new_bounds)?;
        }
        Ok(())
    }

    /// Concretize the current DAG/spec-CROWN frontier and collapse it onto the
    /// network input accumulator.
    ///
    /// This is the shared truncation primitive for #3813: the backward pass can
    /// stop at any frontier as long as each remaining node contribution is
    /// concretized with sound forward bounds before the final input concretization.
    /// Mirrors alpha-beta-CROWN's use of fixed intermediate bounds during BaB.
    pub(crate) fn concretize_crown_frontier_to_network_input(
        &self,
        node_crown_bounds: &mut CrownMergeAccumulator,
        node_bounds: &std::collections::HashMap<String, BoundedTensor>,
        output_dim: usize,
        input_dim: usize,
        input_accumulated: &mut bool,
    ) -> Result<LinearBounds> {
        let mut network_input_bounds = None;

        for (node_name, bounds) in node_crown_bounds.drain() {
            if node_name == NETWORK_INPUT {
                network_input_bounds = Some(bounds);
                continue;
            }

            let node_ibp = node_bounds.get(&node_name).ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "Truncated CROWN frontier missing forward bounds for node '{}'",
                    node_name
                ))
            })?;
            let concretized = bounds.into_dense()?.concretize_sound(node_ibp).flatten();
            Self::accumulate_bias_to_network_input_crown(
                &Array1::from_iter(concretized.lower().iter().copied()),
                &Array1::from_iter(concretized.upper().iter().copied()),
                node_crown_bounds,
                output_dim,
                input_dim,
                input_accumulated,
            );
        }

        if let Some(bounds) = network_input_bounds {
            self.accumulate_crown_bounds_to_input(
                NETWORK_INPUT,
                bounds,
                node_crown_bounds,
                output_dim,
                input_dim,
                input_accumulated,
            )?;
        }

        node_crown_bounds
            .take(NETWORK_INPUT)?
            .ok_or_else(|| NyError::InvalidSpec("No path to network input found".to_string()))?
            .into_dense()
    }

    /// Element-wise addition of two arrays with NaN→conservative-infinity
    /// fallback (#2951). Generic over array dimension: works for `Array1`,
    /// `Array2`, `ArrayD`, etc. Returns a conservative (all ±Inf) array if
    /// shapes mismatch instead of panicking (#2907).
    ///
    /// # Inf coefficient invariant (#3032)
    ///
    /// When accumulating CROWN backward contributions across DAG merge points,
    /// NaN results (from inf + (-inf) cancellation) are replaced with ±Inf.
    /// This means `LinearBounds` coefficients may contain ±Inf after
    /// accumulation, which is intentional: Inf coefficients produce
    /// maximally-loose-but-sound bounds at concretization time. See the
    /// "Two-Phase Invariant" documentation on [`LinearBounds`].
    pub(crate) fn safe_add<D: Dimension>(
        existing: &Array<f32, D>,
        new: &Array<f32, D>,
        is_lower: bool,
    ) -> Array<f32, D> {
        if existing.shape() != new.shape() {
            if existing.len() == new.len() {
                if let Ok(reshaped) = new.view().into_shape_with_order(existing.raw_dim()) {
                    tracing::debug!(
                        existing_shape = ?existing.shape(),
                        new_shape = ?new.shape(),
                        "safe_add reshaping same-count contribution before accumulation"
                    );
                    return Self::safe_add_same_shape(existing, &reshaped, is_lower);
                }
            }
            tracing::warn!(
                existing_shape = ?existing.shape(),
                new_shape = ?new.shape(),
                "safe_add shape mismatch; returning conservative bounds"
            );
            let conservative = if is_lower {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            };
            return Array::from_elem(existing.raw_dim(), conservative);
        }
        Self::safe_add_same_shape(existing, new, is_lower)
    }

    /// Single element of [`safe_add`](Self::safe_add): NaN-safe f32 addition
    /// with the conservative-infinity degrade. Extracted so the flat SoA
    /// batched-backward merge (#lsnc-batched-bwd,
    /// `propagation/batched/batched_bwd.rs`) applies the IDENTICAL per-element
    /// arithmetic as the scalar reference (bit-parity by construction).
    #[inline]
    pub(crate) fn safe_add_elem(e: f32, n: f32, is_lower: bool) -> f32 {
        // NaN check: sum.is_nan() catches all cases:
        //   - NaN + x = NaN, x + NaN = NaN (input NaN)
        //   - inf + (-inf) = NaN (cancellation)
        let sum = e + n;
        if sum.is_nan() {
            if is_lower {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            }
        } else {
            sum
        }
    }

    /// Inner implementation of `safe_add` for shape-matched arrays.
    ///
    /// Replaces every NaN element in the sum with the conservative bound
    /// (−Inf for lower, +Inf for upper). This is the source of ±Inf
    /// coefficients in accumulated `LinearBounds` (see #3032).
    fn safe_add_same_shape<D, S>(
        existing: &Array<f32, D>,
        new: &ArrayBase<S, D>,
        is_lower: bool,
    ) -> Array<f32, D>
    where
        D: Dimension,
        S: Data<Elem = f32>,
    {
        // Use map_collect to construct the output directly, avoiding a
        // clone() + overwrite cycle that doubles allocation (#2220 F13).
        Zip::from(existing)
            .and(new)
            .map_collect(|&e, &n| Self::safe_add_elem(e, n, is_lower))
    }

    pub(crate) fn sanitize_bounds_for_fallback(bounds: &BoundedTensor) -> BoundedTensor {
        let mut lower = bounds.lower().clone();
        let mut upper = bounds.upper().clone();

        Zip::from(&mut lower).and(&mut upper).for_each(|l, u| {
            if !l.is_finite() {
                *l = f32::NEG_INFINITY;
            }
            if !u.is_finite() {
                *u = f32::INFINITY;
            }
            if *l > *u {
                *l = f32::NEG_INFINITY;
                *u = f32::INFINITY;
            }
        });

        // NaN replaced with ±Inf above; Inf is sound conservative fallback.
        match BoundedTensor::new_allow_infinite(lower, upper) {
            Ok(sanitized) => sanitized,
            Err(err) => {
                // #2812 Finding 49: Return conservative infinity bounds, not the
                // original unsanitized bounds which may contain NaN or inverted pairs.
                tracing::warn!(
                    "sanitize_bounds_for_fallback: shape mismatch {err}; \
                     returning conservative infinity bounds"
                );
                BoundedTensor::new_conservative(bounds.lower().shape())
            }
        }
    }

    /// Concretize accumulated linear bounds using IBP bounds as a partial CROWN fallback.
    ///
    /// When a layer's CROWN backward propagation fails or produces non-finite coefficients,
    /// we fall back to concretizing the accumulated linear bounds at this node using the
    /// pre-computed IBP bounds.
    ///
    /// Checks both IBP bounds AND accumulated linear bounds for Inf/NaN. If either is
    /// non-finite, sanitizes to IBP-only bounds. concretize_sound() guarantees no
    /// NaN/inversion (#2287).
    ///
    /// Part of design: crown-batched-partial-fallback-dedup (Slice 1).
    pub(crate) fn partial_crown_fallback(
        node_lb: &BatchedLinearBounds,
        ibp_bounds: &BoundedTensor,
        output_shape: &[usize],
    ) -> Result<BoundedTensor> {
        let ibp_has_bad = ibp_bounds
            .lower()
            .iter()
            .chain(ibp_bounds.upper().iter())
            .any(|v: &f32| !v.is_finite());

        let lb_has_bad = Self::has_non_finite_coefficients(node_lb);

        let result = if ibp_has_bad || lb_has_bad {
            Self::sanitize_bounds_for_fallback(ibp_bounds)
        } else {
            node_lb.concretize_sound(ibp_bounds)?
        };

        if result.shape() != output_shape {
            result.reshape(output_shape)
        } else {
            Ok(result)
        }
    }

    /// Check if any coefficient in the batched linear bounds is Inf or NaN.
    ///
    /// Part of design: crown-batched-partial-fallback-dedup (Slice 5).
    pub(crate) fn has_non_finite_coefficients(lb: &BatchedLinearBounds) -> bool {
        lb.lower_a
            .iter()
            .chain(lb.upper_a.iter())
            .chain(lb.lower_b.iter())
            .chain(lb.upper_b.iter())
            .any(|v: &f32| !v.is_finite())
    }

    /// Accumulate `BatchedCrownBounds` for a given input node (Phase 4, #2613).
    ///
    /// Mirrors [`accumulate_crown_bounds_to_input`] for the batched path.
    /// On first insertion, preserves the variant (Patches stays Patches).
    /// On merge (multiple paths converge), converts both to Dense and keeps the
    /// merged dense carrier in f64 until the node is consumed.
    #[cfg(test)]
    pub(crate) fn accumulate_batched_crown_bounds_to_input(
        &self,
        input_name: &str,
        new_bounds: BatchedCrownBounds,
        node_bounds: &mut std::collections::HashMap<String, BatchedCrownBounds>,
        input_accumulated: &mut bool,
    ) -> Result<()> {
        let is_network_input = input_name == NETWORK_INPUT;
        let is_first = if is_network_input {
            !*input_accumulated
        } else {
            !node_bounds.contains_key(input_name)
        };

        if is_first {
            // First insertion: preserve variant (Patches stays Patches).
            if is_network_input {
                node_bounds.insert(NETWORK_INPUT.to_string(), new_bounds);
                *input_accumulated = true;
            } else {
                node_bounds.insert(input_name.to_string(), new_bounds);
            }
        } else {
            // Merge point: convert both to Dense, then safe_add (#3550: checked).
            let new_blb = new_bounds
                .into_batched_dense_checked("accumulate_batched_crown_bounds_to_input:new")?;
            let key = if is_network_input {
                NETWORK_INPUT
            } else {
                input_name
            };
            if let Some(existing_bcb) = node_bounds.get_mut(key) {
                existing_bcb.merge_dense_checked(
                    new_blb,
                    "accumulate_batched_crown_bounds_to_input:existing",
                )?;
            } else {
                tracing::error!(
                    "accumulate_batched_crown_bounds_to_input: merge expected but {} \
                     missing — bounds dropped",
                    key
                );
                debug_assert!(false, "BatchedCrownBounds entry missing during merge");
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "utils_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "utils_fallback_tests.rs"]
mod fallback_tests;

#[cfg(test)]
#[path = "utils_batched_merge_tests.rs"]
mod batched_merge_tests;
