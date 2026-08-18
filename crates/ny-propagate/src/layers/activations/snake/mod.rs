// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Snake activation layer: y = x + (1/a) * sin²(a*x)
//!
//! Introduced in "Neural Networks Fail to Learn Periodic Functions and How to Fix It"
//! (Ziyin et al., 2020). Used in neural audio synthesis (e.g., BigVGAN).
//!
//! Properties:
//! - Monotonically non-decreasing: f'(x) = 1 + sin(2ax) >= 0
//! - Output range: f(x) ∈ [x, x + 1/a] for all x
//! - Frequency parameter `a` controls oscillation frequency
//!
//! Reference: avoice Snake kernel uses this as `x + (1/alpha) * sin²(alpha * x)`.

use ndarray::Array1;
use ny_core::{NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

use super::LinearRelaxation;
use crate::layers::common::{impl_elementwise_activation, BoundPropagation};

/// Snake activation: y = x + (1/a) * sin²(a*x)
///
/// Monotonically non-decreasing periodic activation for audio neural networks.
/// Frequency parameter `a` controls the periodicity of the oscillatory component.
#[derive(Debug, Clone)]
pub struct SnakeLayer {
    /// Per-channel alpha values. `len() == 1` means scalar broadcast.
    pub(crate) alpha: Array1<f32>,
}

impl SnakeLayer {
    /// Create a new Snake layer with the given scalar alpha parameter.
    ///
    /// Requires `a > 0`. The Ziyin et al. 2020 paper defines Snake only for
    /// positive alpha parameters. Negative `a` causes unsound CROWN
    /// relaxation bounds because `enumerate_periodic_points` produces an
    /// empty range when `a < 0` (#3095).
    pub fn new(a: f32) -> Result<Self> {
        if a <= 0.0 || !a.is_finite() {
            return Err(NyError::InvalidSpec(format!(
                "Snake frequency parameter `a` must be positive and finite, got {a}"
            )));
        }
        Ok(Self {
            alpha: Array1::from_elem(1, a),
        })
    }

    /// Create a Snake layer with per-channel alpha values.
    pub fn per_channel(alpha: Array1<f32>) -> Result<Self> {
        if alpha.is_empty() {
            return Err(NyError::InvalidSpec(
                "Snake alpha must be non-empty".to_string(),
            ));
        }
        for (i, &a) in alpha.iter().enumerate() {
            if a <= 0.0 || !a.is_finite() {
                return Err(NyError::InvalidSpec(format!(
                    "Snake alpha[{i}] must be positive and finite, got {a}"
                )));
            }
        }
        Ok(Self { alpha })
    }

    /// Per-channel alpha values (read-only accessor for cross-crate inspection).
    pub fn alpha(&self) -> &Array1<f32> {
        &self.alpha
    }

    /// Create a Snake layer with default alpha = 1.0.
    pub fn default_frequency() -> Self {
        Self {
            alpha: Array1::from_elem(1, 1.0),
        }
    }

    /// The alpha value for a given element index (handles scalar broadcast).
    ///
    /// WARNING: Uses `idx % alpha.len()` which is only correct when each channel
    /// has exactly one element (1D inputs). For inputs with spatial dims like
    /// `[C, T]`, use [`alpha_for_flat`] with the correct stride instead.
    #[cfg(test)]
    #[inline]
    fn alpha_at(&self, idx: usize) -> f32 {
        if self.alpha.len() == 1 {
            self.alpha[0]
        } else {
            self.alpha[idx % self.alpha.len()]
        }
    }

    /// Compute per-channel spatial stride via shared helper. Part of #4169.
    fn per_channel_stride(&self, total_elements: usize) -> Result<usize> {
        crate::layers::common::per_channel::per_channel_spatial_stride(
            total_elements,
            self.alpha.len(),
            "Snake",
        )
    }

    /// Alpha for a flat index given the per-channel stride.
    #[inline]
    fn alpha_for_flat(&self, flat_idx: usize, stride: usize) -> f32 {
        if self.alpha.len() == 1 {
            self.alpha[0]
        } else {
            self.alpha[crate::layers::common::per_channel::channel_index_for_flat(flat_idx, stride)]
        }
    }
}

/// Evaluate snake(x) = x + (1/a) * sin²(a*x) in f64 for precision.
pub(crate) fn snake_eval_f64(x: f64, a: f64) -> f64 {
    // Only the exact zero-frequency limit is identity. Any nonzero binary32
    // alpha can still accumulate an O(1/|a|) residual at a sufficiently large
    // input, so a magnitude cutoff is not a sound semantic shortcut.
    if a == 0.0 {
        return x;
    }
    let sin_ax = (a * x).sin();
    x + sin_ax * sin_ax / a
}

/// Evaluate snake(x) in f32 (test-only; production IBP uses f64 with directed rounding).
#[cfg(test)]
pub(crate) fn snake_eval_f32(x: f32, a: f32) -> f32 {
    snake_eval_f64(x as f64, a as f64) as f32
}

impl BoundPropagation for SnakeLayer {
    /// IBP for Snake: exact bounds via monotonicity.
    ///
    /// f'(x) = 1 + sin(2ax) >= 0, so f([l, u]) = [f(l), f(u)].
    /// This is the key insight solving #3051: the composite Snake function is monotone
    /// and produces tight bounds directly, unlike the Sin/Pow/Div composition.
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let flat_lower = input
            .lower()
            .view()
            .into_shape_with_order(input.lower().len())
            .map_err(|e| {
                NyError::InvalidSpec(format!("Snake: failed to flatten lower bounds: {e}"))
            })?;
        let flat_upper = input
            .upper()
            .view()
            .into_shape_with_order(input.upper().len())
            .map_err(|e| {
                NyError::InvalidSpec(format!("Snake: failed to flatten upper bounds: {e}"))
            })?;

        let mut lower = Array1::zeros(flat_lower.len());
        let mut upper = Array1::zeros(flat_upper.len());

        // Directed rounding: snake_eval_f64 computes in f64 but the as-f32 cast
        // does nearest rounding. Apply next_down/next_up for soundness. (#3245)
        //
        // Per-channel alpha: compute stride so flat index maps to correct channel.
        // For [C, T] input with C-channel alpha, stride = T, channel = i / T.
        let stride = self.per_channel_stride(flat_lower.len())?;
        for i in 0..flat_lower.len() {
            let alpha = self.alpha_for_flat(i, stride);
            lower[i] = next_down_f32(snake_eval_f64(flat_lower[i] as f64, alpha as f64) as f32);
            upper[i] = next_up_f32(snake_eval_f64(flat_upper[i] as f64, alpha as f64) as f32);
        }

        let lower = lower
            .into_shape_with_order(input.shape())
            .map_err(|e| {
                NyError::InvalidSpec(format!(
                    "Snake: failed to reshape lower bounds to {:?}: {e}",
                    input.shape()
                ))
            })?
            .to_owned();
        let upper = upper
            .into_shape_with_order(input.shape())
            .map_err(|e| {
                NyError::InvalidSpec(format!(
                    "Snake: failed to reshape upper bounds to {:?}: {e}",
                    input.shape()
                ))
            })?
            .to_owned();
        BoundedTensor::new(lower, upper)
    }

    impl_elementwise_activation!(
        @trait_methods
        SnakeLayer,
        NyError::InvalidSpec(
            "Snake CROWN propagation requires pre-activation bounds. \
             Use propagate_linear_with_bounds() instead."
                .to_string()
        )
    );
}

/// Compute max/min deviation of snake(x) from a chord line on [l, u].
///
/// The chord passes through (l, snake(l)) and (u, snake(u)). This function
/// enumerates critical points of h(x) = snake(x) - chord(x) to find extrema.
/// Returns (min_deviation, max_deviation) where deviations are relative to chord.
///
/// When the number of critical points exceeds [`MAX_PERIODIC_POINTS`] (large
/// `a * interval_width`), falls back to conservative bounds: the sin²(ax)/a
/// component has range [0, 1/a], so chord deviation is bounded by ±1/a.
/// This is sound but may be looser than the exact enumeration.
fn chord_deviation_bounds(
    l64: f64,
    u64: f64,
    a64: f64,
    chord_slope: f64,
    chord_intercept: f64,
) -> (f64, f64) {
    let mut max_h: f64 = 0.0;
    let mut min_h: f64 = 0.0;
    let two_pi = 2.0 * std::f64::consts::PI;

    // Conservative fallback: sin²(ax)/a ∈ [0, 1/a], so maximum deviation
    // of the oscillatory part from any chord is at most 1/a.
    let conservative = (-1.0 / a64, 1.0 / a64);

    // Check critical points where h'(x) = 0, i.e., sin(2ax) = chord_slope - 1
    let target = chord_slope - 1.0;
    if target.abs() <= 1.0 {
        let base1 = target.asin();
        let base2 = std::f64::consts::PI - base1;
        for base in [base1, base2] {
            if !enumerate_periodic_points(l64, u64, a64, base, two_pi, |x| {
                let h = snake_eval_f64(x, a64) - chord_slope * x - chord_intercept;
                max_h = max_h.max(h);
                min_h = min_h.min(h);
            }) {
                return conservative;
            }
        }
    }

    // Also check extrema of f' (where sin(2ax) = ±1)
    for base in [-std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2] {
        if !enumerate_periodic_points(l64, u64, a64, base, two_pi, |x| {
            let h = snake_eval_f64(x, a64) - chord_slope * x - chord_intercept;
            max_h = max_h.max(h);
            min_h = min_h.min(h);
        }) {
            return conservative;
        }
    }

    (min_h, max_h)
}

/// Maximum number of periodic points to enumerate before falling back
/// to conservative worst-case deviation bounds. This avoids O(a*width)
/// iteration while maintaining soundness.
const MAX_PERIODIC_POINTS: i64 = 10_000;

/// Enumerate x values where 2*a*x = base + period*k and x ∈ [l, u].
///
/// Calls `callback(x)` for each such point. If the range contains more than
/// [`MAX_PERIODIC_POINTS`] points, returns `false` to signal the caller should
/// use conservative worst-case bounds instead.
fn enumerate_periodic_points(
    l: f64,
    u: f64,
    a: f64,
    base: f64,
    period: f64,
    mut callback: impl FnMut(f64),
) -> bool {
    let k_start_f64 = ((2.0 * a * l - base) / period).ceil();
    let k_end_f64 = ((2.0 * a * u - base) / period).floor();
    // SAFETY(#3100): If the division produces non-finite values (Inf/NaN from
    // very large a*l or a*u), or values outside i64 range, fall back to
    // conservative bounds. Saturating `as i64` on Inf/huge values would produce
    // garbage iteration bounds, potentially skipping critical points (unsound).
    if !k_start_f64.is_finite()
        || !k_end_f64.is_finite()
        || k_start_f64 < i64::MIN as f64
        || k_end_f64 > i64::MAX as f64
    {
        return false;
    }
    let k_start = k_start_f64 as i64;
    let k_end = k_end_f64 as i64;
    if k_end - k_start > MAX_PERIODIC_POINTS {
        return false;
    }
    for k in k_start..=k_end {
        let x = (base + period * k as f64) / (2.0 * a);
        if x >= l && x <= u {
            callback(x);
        }
    }
    true
}

/// Analytical linear relaxation for Snake on interval [l, u].
///
/// Uses chord from endpoints plus analytical deviation bounds. Narrow intervals
/// take the exact monotone enclosure; small `|a|` and unresolvable phase take
/// the universal band.
///
/// THE THRESHOLDS HERE ARE RELATIVE TO `a`, NOT ABSOLUTE. Every earlier
/// violation this function had came from an absolute cutoff applied to a
/// function whose only intrinsic scale is `1/a`: `|a| < 1e-8 ⇒ identity` (the
/// deviation is `1/|a|` there, not zero) and `u - l < 1e-8 ⇒ tangent` (at
/// `a = 1e8` that width is a full radian of oscillation, not a point). Both are
/// now decided against `a`.
pub(crate) fn snake_linear_relaxation(l: f32, u: f32, a: f32) -> LinearRelaxation {
    if l.is_nan() || u.is_nan() || a.is_nan() {
        return LinearRelaxation::nan_fallback();
    }
    let l64 = f64::from(l);
    let u64 = f64::from(u);
    let a64 = f64::from(a);

    // The band relaxation is sound for EVERY `a` and every interval; it is only
    // loose. So it is the answer in both regimes where the oscillation term
    // cannot be pinned down: `|a|` too small for the identity limit to have
    // arrived, and `|a·x|` too large for `sin` to have a meaningful argument.
    if a.abs() < 1e-8 {
        return snake_band_relaxation(l, u, a);
    }
    // Ordered BEFORE the phase check on purpose: an unbounded side makes
    // `|a·x|` unresolvable by construction, and the constant-bound shape that
    // #3083 pinned is what callers of a half-infinite domain expect.
    // `snake_infinite_relaxation` makes its own phase check for its one finite
    // endpoint.
    if l.is_infinite() || u.is_infinite() {
        return snake_infinite_relaxation(l, u, a);
    }
    if phase_is_beyond_f64_resolution(a64, l64, u64) {
        return snake_band_relaxation(l, u, a);
    }

    // A point interval, and any interval too narrow to take a chord across
    // without amplifying the endpoint cancellation, gets the EXACT enclosure
    // instead of a line: `f' = 1 + sin(2ax) ≥ 0` everywhere, so `f` is
    // non-decreasing and `f([l, u]) = [f(l), f(u))]`. A tangent line used to be
    // returned here, which is not an enclosure of a non-convex function over a
    // non-degenerate interval, and at `|l| ~ 1e12` its `intercept = y - slope·l`
    // also cancelled away most of its own significance on the way back to f32.
    // `1e-8 / |a|` is the width in x of a fixed small phase, so this stays a
    // point test as `a` grows instead of quietly becoming a full period.
    if (u64 - l64).abs() < 1e-8 / a64.abs() {
        return snake_monotone_enclosure(l64, u64, a64);
    }

    let fl = snake_eval_f64(l64, a64);
    let fu = snake_eval_f64(u64, a64);
    let chord_slope = (fu - fl) / (u64 - l64);
    let chord_intercept = fl - chord_slope * l64;

    let (min_h, max_h) = chord_deviation_bounds(l64, u64, a64, chord_slope, chord_intercept);

    let max_abs_x = l.abs().max(u.abs()) as f64;
    let cs_f32 = chord_slope as f32;
    let cs_err = next_up_f32(((chord_slope - cs_f32 as f64).abs() * max_abs_x) as f32);
    LinearRelaxation::new(
        cs_f32,
        next_down_f32(((chord_intercept + min_h) as f32) - cs_err),
        cs_f32,
        next_up_f32(((chord_intercept + max_h) as f32) + cs_err),
    )
}

/// Whether `sin(a·x)` has any meaning at this scale.
///
/// `a` and `x` are exact f32 values, so `a64 * x64` is one correctly-rounded
/// product — but its ABSOLUTE error is `|a·x|·2^-53`, and that is an error in
/// RADIANS. Past roughly `2^43` the angle is uncertain by more than a
/// milliradian; past `2^52` its ulp exceeds a full radian and `sin` of it
/// carries no information about the true phase at all. Every branch that
/// evaluates `sin` must refuse those inputs rather than relax around a number
/// the hardware invented — the audit's `a = 2.96e35` witness had `|a·x| ≈ 2e27`.
fn phase_is_beyond_f64_resolution(a64: f64, l64: f64, u64: f64) -> bool {
    const MAX_RESOLVABLE_PHASE: f64 = 8.796_093_022_208e12; // 2^43
    let max_abs_x = l64.abs().max(u64.abs());
    // `!(<=)` rather than `>`: a NaN product must count as unresolvable, and
    // `>` would answer `false` for it.
    !matches!(
        (a64.abs() * max_abs_x).partial_cmp(&MAX_RESOLVABLE_PHASE),
        Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
    )
}

/// The EXACT enclosure of Snake over `[l, u]`, as two constants.
///
/// `f'(x) = 1 + sin(2ax) ≥ 0` for every real `a` and `x`, so Snake is
/// non-decreasing everywhere and `f([l, u]) = [f(l), f(u)]` — there is nothing
/// to relax. Constants carry no correlation with `x` downstream, which is why
/// this is reserved for intervals narrow enough that the correlation is worth
/// nothing anyway; in exchange it has no cancellation to widen for, unlike a
/// line whose intercept must undo `slope · l`.
fn snake_monotone_enclosure(l64: f64, u64: f64, a64: f64) -> LinearRelaxation {
    let fl = snake_eval_f64(l64, a64);
    let fu = snake_eval_f64(u64, a64);
    // One f32 ulp outward covers the f64 evaluation error by ~9 orders of
    // magnitude; `next_down(nearest(v)) < v < next_up(nearest(v))` holds because
    // the nearest f32 is within half an ulp.
    LinearRelaxation::new(0.0, next_down_f32(fl as f32), 0.0, next_up_f32(fu as f32))
}

/// Slope-1 relaxation carrying the full deviation band — sound for any `a`.
///
/// The small-`|a|` branch previously returned [`LinearRelaxation::identity`]
/// outright, on the reading that Snake degenerates to `f(x) = x` as `a → 0`. It
/// does not.
/// The module header states the true range — `f(x) ∈ [x, x + 1/a]` — and that
/// band WIDENS as `a` shrinks. At `a = 1e-9` the identity line can sit `1e9`
/// below the function, and the field-wide envelope audit found exactly that:
/// 117 violations, the worst `6.8e8` low at `l = u = -1e12`. An upper envelope
/// beneath the function is the false-proof direction.
///
/// The deviation is `d(x) = sin²(ax)/a`, and `sin²(t) ≤ min(1, t²)` gives
///
/// ```text
///     |d(x)| ≤ D = min(1/|a|, |a|·M²),   M = max(|l|, |u|)
/// ```
///
/// Both arms of that minimum matter. `1/|a|` is what bounds the oscillation once
/// `|a·x|` is large enough to complete a period — the regime the shortcut got
/// wrong. `|a|·M²` is what keeps ordinary inputs sharp: at `M = 1`, `a = 1e-9`
/// the band is `1e-9`, so this stays the identity relaxation to within an ULP
/// wherever the old code was actually right.
///
/// `d` is single-signed — it takes the sign of `a`, since `sin² ≥ 0` — so the
/// band is one-sided and only the intercept on that side moves.
fn snake_band_relaxation(l: f32, u: f32, a: f32) -> LinearRelaxation {
    let a64 = f64::from(a);
    // `M` may be infinite (this branch is reached before the infinite-bound
    // check). Then `|a|·M²` is infinite and the minimum is `1/|a|`, which is
    // the correct finite band — a strictly better answer than the constant
    // bounds `snake_infinite_relaxation` would have given.
    let m = f64::from(l.abs().max(u.abs()));
    let deviation = if a64 == 0.0 {
        // The exact limit: `sin²(0·x)/a` is `0/0` in arithmetic but `0` in the
        // limit, because `sin²(ax) ≤ a²x²` vanishes faster than `a`.
        0.0
    } else {
        (1.0 / a64.abs()).min(a64.abs() * m * m)
    };
    // Round the band OUTWARD. `next_up_f32` of a rounded-to-nearest f32 is a
    // sound cover of the f64 value; the slope is exactly 1, so it contributes
    // no error of its own to widen for.
    let band = next_up_f32(deviation as f32);
    if a64 < 0.0 {
        // `d ≤ 0`: the function sits at or BELOW the identity line.
        LinearRelaxation::new(1.0, next_down_f32(-band), 1.0, 0.0)
    } else {
        LinearRelaxation::new(1.0, 0.0, 1.0, band)
    }
}

/// Relaxation for infinite input bounds. Sound constant bounds.
///
/// The finite endpoint is evaluated through `sin`, so it gets the same phase
/// check every other `sin`-evaluating branch gets — an unbounded side does not
/// excuse relaxing around a phase the hardware invented. When the check refuses,
/// the universal band is still available and is in fact TIGHTER here than the
/// constants below, since it replaces an infinite side with `x ± 1/|a|`.
fn snake_infinite_relaxation(l: f32, _u: f32, a: f32) -> LinearRelaxation {
    if l.is_infinite() && _u.is_infinite() {
        return LinearRelaxation::new(0.0, f32::NEG_INFINITY, 0.0, f32::INFINITY);
    }
    let finite_endpoint = if l.is_infinite() { _u } else { l };
    if phase_is_beyond_f64_resolution(
        f64::from(a),
        f64::from(finite_endpoint),
        f64::from(finite_endpoint),
    ) {
        return snake_band_relaxation(l, _u, a);
    }
    if l.is_infinite() {
        let fu = snake_eval_f64(_u as f64, a as f64);
        return LinearRelaxation::new(0.0, f32::NEG_INFINITY, 0.0, next_up_f32(fu as f32));
    }
    // u = +inf, l finite: f is monotone so f(x) >= f(l) for all x >= l.
    // Lower bound = constant f(l), upper bound = +inf.
    let fl = snake_eval_f64(l as f64, a as f64);
    LinearRelaxation::new(0.0, next_down_f32(fl as f32), 0.0, f32::INFINITY)
}

impl SnakeLayer {
    /// CROWN backward propagation with pre-activation bounds.
    ///
    /// For per-channel alpha, computes channel stride from pre-activation shape
    /// to correctly map flat neuron index → channel index. This avoids the
    /// modulo bug where `flat_idx % C` gives the wrong channel for inputs with
    /// spatial dimensions (e.g., `[C, T]` audio tensors). (#4117)
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &crate::LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<crate::LinearBounds> {
        crate::layers::common::non_finite_domain_guard("Snake", pre_activation)?;
        let stride = self.per_channel_stride(pre_activation.len())?;
        let relax_fn = |l: f32, u: f32, i: usize| {
            snake_linear_relaxation(l, u, self.alpha_for_flat(i, stride))
        };
        crate::layers::common::crown_elementwise_backward_indexed(bounds, pre_activation, relax_fn)
    }

    /// Batched CROWN backward propagation with pre-activation bounds.
    ///
    /// Same stride-based channel mapping as [`propagate_linear_with_bounds`].
    pub fn propagate_linear_batched_with_bounds(
        &self,
        bounds: &crate::BatchedLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<crate::BatchedLinearBounds> {
        crate::layers::common::non_finite_domain_guard("Snake", pre_activation)?;
        let stride = self.per_channel_stride(pre_activation.len())?;
        let relax_fn = |l: f32, u: f32, i: usize| {
            snake_linear_relaxation(l, u, self.alpha_for_flat(i, stride))
        };
        crate::layers::common::crown_elementwise_backward_batched_indexed(
            bounds,
            pre_activation,
            relax_fn,
        )
    }

    /// Patches CROWN backward propagation with pre-activation bounds.
    ///
    /// Scalar Snake keeps the existing Patches fast path. Per-channel Snake
    /// falls back to Dense until the common indexed-patches helper exists.
    pub(crate) fn propagate_patches_with_bounds(
        &self,
        bounds: &crate::bounds::patches::PatchesLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<crate::bounds::patches::CrownBounds> {
        crate::layers::common::non_finite_domain_guard("Snake", pre_activation)?;
        if self.alpha.len() == 1 {
            let alpha = self.alpha[0];
            return crate::layers::common::crown_elementwise_backward_patches(
                bounds,
                pre_activation,
                |l, u| snake_linear_relaxation(l, u, alpha),
            );
        }
        Err(NyError::NumericalInstability(
            "Snake Patches backward does not yet support per-channel alpha; falling back to Dense"
                .to_string(),
        ))
    }
}

#[cfg(test)]
mod tests;
