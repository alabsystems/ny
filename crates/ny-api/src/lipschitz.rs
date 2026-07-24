// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! **Sound** deterministic global Lipschitz certification (NY ext 2).
//!
//! [`certify_upper_bound`] computes a certified upper bound on the global
//! Lipschitz constant (ℓ₂ → ℓ₂) of a sequential network in **exact rational
//! arithmetic** ([`ny_cert::Rat`]). It is deliberately distinct from
//! [`crate::probabilistic::estimate_lipschitz_from_network`], which multiplies
//! `f32` spectral norms and silently treats unrecognized layers as
//! 1-Lipschitz ("optimistic", flagged only via
//! `LipschitzEstimate::is_sound`). This module instead **fails closed**: any
//! layer outside the certified fragment is an error, and every number in the
//! result is an exact rational computed without rounding (the one square root
//! is taken with a certified outward integer square root).
//!
//! # Bound formula and soundness argument
//!
//! For a sequential composition `f = L_m ∘ … ∘ L_1`, the Lipschitz constant
//! satisfies `Lip(f) ≤ ∏ᵢ Lip(L_i)` (composition of Lipschitz maps). Per
//! layer:
//!
//! - **Linear (`y = Wx + b`)**: `Lip = ‖W‖₂` (largest singular value), which
//!   is soundly upper-bounded by
//!   `‖W‖₂ ≤ min( √(‖W‖₁ · ‖W‖∞), ‖W‖_F )`, computed exactly:
//!   `‖W‖₂² = ρ(WᵀW) ≤ ‖WᵀW‖∞ ≤ ‖Wᵀ‖∞·‖W‖∞ = ‖W‖₁·‖W‖∞` (Hölder /
//!   interpolation bound), and `‖W‖₂ ≤ ‖W‖_F` because the squared Frobenius
//!   norm is the sum of *all* squared singular values. `‖W‖₁` (max column
//!   abs-sum), `‖W‖∞` (max row abs-sum), and `‖W‖_F²` are all exact rationals
//!   over the exact dyadic values of the `f32` weights.
//! - **Conv1d / Conv2d**: the layer is a linear operator `A` (the unrolled
//!   convolution matrix, never materialized). Each unrolled row for output
//!   channel `o` contains each kernel entry `w[o, c, k]` **at most once**
//!   (zero padding only drops terms), so `‖A‖∞ ≤ maxₒ Σ_{c,k} |w[o,c,k]|`;
//!   symmetrically, for a fixed input position each `(o, k)` pair contributes
//!   at most one matrix entry per column (the output position is determined
//!   by the stride/dilation relation), so
//!   `‖A‖₁ ≤ max_c Σ_{o ∈ group(c), k} |w[o,c,k]|`. Both bounds hold for any
//!   stride ≥ 1, padding, dilation, and grouping. Then
//!   `‖A‖₂² ≤ ‖A‖₁·‖A‖∞` as above. (The Frobenius alternative is *not* used
//!   for conv layers: kernel entries repeat across output positions, so the
//!   kernel's Frobenius norm does not bound `‖A‖_F`.)
//! - **ReLU**: exactly 1-Lipschitz componentwise
//!   (`|max(0,a) − max(0,b)| ≤ |a − b|`), hence 1-Lipschitz in ℓ₂.
//! - **Reshape / Flatten / Transpose**: permutations of coordinates — exact
//!   ℓ₂ isometries (Lipschitz constant 1).
//!
//! The per-layer *squared* bounds `Sᵢ` are multiplied exactly:
//! `Lip(f)² ≤ ∏ᵢ Sᵢ = Q`, and the returned [`SoundLipschitz::bound`] is
//! `r = sqrt_upper(Q)` with `r² ≥ Q` certified by construction
//! ([`Rat::sqrt_upper`]), so `Lip(f) ≤ √Q ≤ r`. Taking the single square root
//! *after* the product keeps `r` at least as tight as multiplying per-layer
//! roots.
//!
//! # Scope and caveats
//!
//! - Certified fragment: sequential `Linear`/`Conv1d`/`Conv2d`/`ReLU`/
//!   `Reshape`/`Flatten`/`Transpose`. Anything else returns an error naming
//!   the offending layer (fail closed) — use the probabilistic module's
//!   estimate if an optimistic value is acceptable.
//! - The bound is on the network's **exact real-arithmetic function**. A
//!   floating-point *evaluation* of the network additionally incurs rounding
//!   error; this certificate does not cover that error.
//! - Non-finite (NaN/∞) weights are rejected.

use ndarray::{Array2, ArrayD, Axis};
use ny_cert::Rat;
use ny_core::{NyError, Result};
use ny_propagate::layers::Layer;
use ny_propagate::Network as SequentialNetwork;

/// Grid precision (in bits) for the one outward square root taken on the
/// exact product of squared per-layer bounds. The overestimate is at most
/// `1/(d·2⁶⁴)` where `d` is the denominator of the exact squared product.
const SQRT_PRECISION_BITS: u32 = 64;

/// Which sound upper bound was selected for a layer's ℓ₂ operator norm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormBoundKind {
    /// `‖W‖₂² ≤ ‖W‖₁ · ‖W‖∞` won (or was the only candidate, for conv).
    OneInfProduct,
    /// `‖W‖₂ ≤ ‖W‖_F` won.
    Frobenius,
    /// Exactly 1-Lipschitz (ReLU) or an exact ℓ₂ isometry (shape ops).
    UnitLipschitz,
}

/// Certified per-layer contribution to the global bound.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerLipschitzBound {
    /// Position of the layer in the sequential network.
    pub index: usize,
    /// Layer type name (as reported by `Layer::layer_type`).
    pub layer_type: &'static str,
    /// Which of the sound norm bounds was selected.
    pub norm_kind: NormBoundKind,
    /// Exact rational upper bound on the layer's **squared** ℓ₂ operator
    /// norm (this is the value that enters the global product — no rounding).
    pub squared_bound: Rat,
    /// Certified upper bound on the layer's ℓ₂ operator norm
    /// (`sqrt_upper` of `squared_bound`; informational — the global bound is
    /// computed from the exact squared product, not from these roots).
    pub bound: Rat,
}

/// A **sound** certified global Lipschitz upper bound (ℓ₂ → ℓ₂).
///
/// Unlike `LipschitzEstimate` from the probabilistic module, this carries no
/// `is_sound` flag: values of this type are sound by construction, and
/// networks outside the certified fragment are rejected with an error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SoundLipschitz {
    /// Certified global upper bound: `Lip(f) ≤ bound`, exact rational.
    pub bound: Rat,
    /// Exact rational upper bound on the **squared** Lipschitz constant
    /// (the product of per-layer squared bounds; `bound² ≥ squared_bound`).
    pub squared_bound: Rat,
    /// Per-layer certified bounds, in network order.
    pub per_layer: Vec<LayerLipschitzBound>,
}

impl SoundLipschitz {
    /// Nearest-`f64` approximation of [`Self::bound`] for display only —
    /// the certified value is the exact rational `bound`.
    #[must_use]
    pub fn bound_approx(&self) -> f64 {
        self.bound.to_f64_approx()
    }
}

/// Certify a sound global Lipschitz upper bound for a sequential network.
///
/// See the module docs for the exact bound formula, its soundness argument,
/// and the supported layer fragment.
///
/// # Errors
///
/// - the network is empty;
/// - any layer is outside the certified fragment (fail closed — the error
///   names the layer);
/// - any weight is non-finite;
/// - conv layer metadata is inconsistent (e.g. `groups` does not divide the
///   output channels).
pub fn certify_upper_bound(network: &SequentialNetwork) -> Result<SoundLipschitz> {
    if network.num_layers() == 0 {
        return Err(NyError::InvalidSpec(
            "cannot certify a Lipschitz bound for an empty network".to_string(),
        ));
    }

    let mut per_layer = Vec::with_capacity(network.num_layers());
    let mut squared_global = Rat::ONE;
    for (index, layer) in network.layers().iter().enumerate() {
        let (squared_bound, norm_kind) = match layer {
            Layer::Linear(linear) => linear_squared_bound(&linear.weight, index)?,
            Layer::Conv1d(conv) => conv_squared_bound(&conv.kernel, conv.groups, index)?,
            Layer::Conv2d(conv) => conv_squared_bound(&conv.kernel, conv.groups, index)?,
            Layer::ReLU(_) | Layer::Reshape(_) | Layer::Flatten(_) | Layer::Transpose(_) => {
                (Rat::ONE, NormBoundKind::UnitLipschitz)
            }
            other => {
                return Err(NyError::InvalidSpec(format!(
                    "sound Lipschitz certification covers only sequential \
                     Linear/Conv1d/Conv2d/ReLU/Reshape/Flatten/Transpose networks; \
                     layer {index} is '{}'. For an optimistic (possibly unsound) value use \
                     ny_api::probabilistic::estimate_lipschitz_from_network",
                    other.layer_type()
                )));
            }
        };
        squared_global = squared_global.mul(squared_bound).map_err(rat_err)?;
        let bound = sqrt_upper_checked(squared_bound)?;
        per_layer.push(LayerLipschitzBound {
            index,
            layer_type: layer.layer_type(),
            norm_kind,
            squared_bound,
            bound,
        });
    }

    let bound = sqrt_upper_checked(squared_global)?;
    Ok(SoundLipschitz {
        bound,
        squared_bound: squared_global,
        per_layer,
    })
}

/// Map the (infallible-by-construction) exact-arithmetic error channel.
fn rat_err(e: ny_cert::RatError) -> NyError {
    NyError::InternalError(format!("exact rational arithmetic failed: {e}"))
}

/// Exact |w| as a rational; rejects NaN/∞ weights (fail closed).
fn rat_abs(w: f32, index: usize) -> Result<Rat> {
    Rat::from_f32_exact(w).map(Rat::abs).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "layer {index} has a non-finite weight ({w}); cannot certify a Lipschitz bound"
        ))
    })
}

/// Certified outward square root of a (non-negative) squared bound.
fn sqrt_upper_checked(squared: Rat) -> Result<Rat> {
    squared.sqrt_upper(SQRT_PRECISION_BITS).ok_or_else(|| {
        NyError::InternalError("squared Lipschitz bound was negative (impossible)".to_string())
    })
}

/// `min(‖W‖₁·‖W‖∞, ‖W‖_F²)` for a dense matrix, in exact rational arithmetic.
fn linear_squared_bound(weight: &Array2<f32>, index: usize) -> Result<(Rat, NormBoundKind)> {
    let mut row_sums = vec![Rat::ZERO; weight.nrows()];
    let mut col_sums = vec![Rat::ZERO; weight.ncols()];
    let mut frob_sq = Rat::ZERO;
    for ((i, j), &w) in weight.indexed_iter() {
        let a = rat_abs(w, index)?;
        row_sums[i] = row_sums[i].add(a).map_err(rat_err)?;
        col_sums[j] = col_sums[j].add(a).map_err(rat_err)?;
        frob_sq = frob_sq.add(a.mul(a).map_err(rat_err)?).map_err(rat_err)?;
    }
    let norm_inf = row_sums.into_iter().max().unwrap_or(Rat::ZERO);
    let norm_one = col_sums.into_iter().max().unwrap_or(Rat::ZERO);
    let one_inf = norm_one.mul(norm_inf).map_err(rat_err)?;
    if one_inf <= frob_sq {
        Ok((one_inf, NormBoundKind::OneInfProduct))
    } else {
        Ok((frob_sq, NormBoundKind::Frobenius))
    }
}

/// `‖A‖₁·‖A‖∞` bound for the unrolled convolution operator, computed from the
/// kernel alone (sound for any stride/padding/dilation/grouping; see module
/// docs). Kernel layout: `(out_channels, in_channels/groups, spatial…)`.
fn conv_squared_bound(
    kernel: &ArrayD<f32>,
    groups: usize,
    index: usize,
) -> Result<(Rat, NormBoundKind)> {
    let shape = kernel.shape();
    if shape.len() < 3 {
        return Err(NyError::InvalidSpec(format!(
            "layer {index}: conv kernel must have shape (out, in/groups, spatial…), got {shape:?}"
        )));
    }
    let out_channels = shape[0];
    let in_per_group = shape[1];
    if groups == 0 || !out_channels.is_multiple_of(groups) {
        return Err(NyError::InvalidSpec(format!(
            "layer {index}: conv groups={groups} must be positive and divide \
             out_channels={out_channels}"
        )));
    }
    let out_per_group = out_channels / groups;

    // ‖A‖∞ ≤ max over output channels of the kernel abs-sum feeding it.
    let mut row_max = Rat::ZERO;
    // ‖A‖₁ ≤ max over (group, in-channel-within-group) of the abs-sum of all
    // kernel entries reading that channel (only that group's outputs do).
    let mut col_max = Rat::ZERO;

    for o in 0..out_channels {
        let mut row_sum = Rat::ZERO;
        for &w in kernel.index_axis(Axis(0), o).iter() {
            row_sum = row_sum.add(rat_abs(w, index)?).map_err(rat_err)?;
        }
        row_max = row_max.max(row_sum);
    }
    for g in 0..groups {
        for c in 0..in_per_group {
            let mut col_sum = Rat::ZERO;
            for o in (g * out_per_group)..((g + 1) * out_per_group) {
                let per_out = kernel.index_axis(Axis(0), o);
                for &w in per_out.index_axis(Axis(0), c).iter() {
                    col_sum = col_sum.add(rat_abs(w, index)?).map_err(rat_err)?;
                }
            }
            col_max = col_max.max(col_sum);
        }
    }

    Ok((
        col_max.mul(row_max).map_err(rat_err)?,
        NormBoundKind::OneInfProduct,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;
    use ny_propagate::layers::LinearLayer;

    fn linear(rows: &[[f32; 2]; 2]) -> Layer {
        Layer::Linear(LinearLayer::new(arr2(rows), None).expect("valid linear"))
    }

    #[test]
    fn empty_network_is_rejected() {
        let network = SequentialNetwork::new();
        assert!(certify_upper_bound(&network).is_err());
    }

    #[test]
    fn frobenius_wins_for_rank_one_row() {
        // W = [3 4]: ‖W‖₂ = 5 exactly; ‖W‖₁·‖W‖∞ = 4·7 = 28 > 25 = ‖W‖_F².
        let mut network = SequentialNetwork::new();
        network.add_layer(Layer::Linear(
            LinearLayer::new(arr2(&[[3.0_f32, 4.0]]), None).expect("valid linear"),
        ));
        let sound = certify_upper_bound(&network).expect("certifiable");
        assert_eq!(sound.bound, Rat::from_int(5));
        assert_eq!(sound.per_layer[0].norm_kind, NormBoundKind::Frobenius);
    }

    #[test]
    fn one_inf_wins_for_diagonal() {
        // W = diag(2, 3): ‖W‖₂ = 3; ‖W‖₁·‖W‖∞ = 9 < 13 = ‖W‖_F².
        let mut network = SequentialNetwork::new();
        network.add_layer(linear(&[[2.0, 0.0], [0.0, 3.0]]));
        let sound = certify_upper_bound(&network).expect("certifiable");
        assert_eq!(sound.bound, Rat::from_int(3));
        assert_eq!(sound.per_layer[0].norm_kind, NormBoundKind::OneInfProduct);
    }

    #[test]
    fn unsupported_layer_fails_closed() {
        use ny_propagate::layers::SigmoidLayer;
        let mut network = SequentialNetwork::new();
        network.add_layer(linear(&[[1.0, 0.0], [0.0, 1.0]]));
        network.add_layer(Layer::Sigmoid(SigmoidLayer));
        let err = certify_upper_bound(&network).expect_err("must fail closed");
        assert!(
            err.to_string().contains("Sigmoid"),
            "error names the layer: {err}"
        );
    }

    #[test]
    fn non_finite_weight_is_rejected() {
        let mut network = SequentialNetwork::new();
        network.add_layer(linear(&[[f32::NAN, 0.0], [0.0, 1.0]]));
        assert!(certify_upper_bound(&network).is_err());
    }
}
