// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! DERIVED cross-lane tolerances for the `#deadline-parity` f32/f64 Linear split.
//!
//! # Why this module exists
//!
//! `9f06e35b2` moved the DOMAIN-BATCHED CROWN backward's `Layer::Linear` arm
//! (`batched/backward_core.rs::dispatch_node_backward`) from
//! `propagate_linear_with_engine_and_deadline` onto `propagate_linear_pollable_cpu`,
//! so that the published bound no longer depends on whether a deadline exists.
//! That is correct and stays. Its consequence is that the batched lane and the
//! per-domain/direct lane no longer run the same arithmetic when the per-domain
//! lane is handed a `GemmEngine`:
//!
//! | | stored coefficient | certified error |
//! | --- | --- | --- |
//! | per-domain WITH engine (`propagate_linear_via_gemm`) | engine's round-to-nearest **f32** `A·W` | <code>&#124;c_f32 − a64&#124; + γ_n^{f32}·S</code> |
//! | batched (`propagate_linear_pollable_cpu`) | `fl32(a64)` of the **f64** contraction | <code>&#124;a64 − fl32(a64)&#124; + γ_n^{f64}·S</code> |
//!
//! with `a64[i,j] = Σ_k (A_ik as f64)·(W_kj as f64)` (the f32×f32 products are
//! exact in f64, so only the sum rounds), `S[i,j] = Σ_k |A_ik|·|W_kj|`, and
//! `γ_n^{u} = n·u/(1 − n·u)` (Higham Thm 3.1) at `u = 2^-24` / `2^-53`.
//! `γ_n^{f32}/γ_n^{f64} ≈ 2^29`, so the two certificates differ by far more than
//! the ~1 ULP by which the two POINT coefficients differ.
//!
//! A test that compares one lane against the other therefore owes a tolerance
//! derived from that certificate gap. Everything below computes that gap from
//! the fixture's own weights and node boxes and from the SAME `γ_n` functions the
//! production code uses — never from an observed diff.
//!
//! # The algebra
//!
//! Fix one Linear layer with contraction width `n` (= `W.nrows()`, the layer's
//! OUTPUT width; the CROWN backward contracts the incoming coefficient row over
//! it). Both lanes publish a certified interval that contains the exact real
//! `A·W`. Re-centre each on `a64`:
//!
//! * engine lane: `|c_f32 − a64| ≤ (γ_n^{f32} + γ_n^{f64})·S`, and the interval is
//!   `c_f32 ± (|c_f32 − a64| + γ_n^{f32}·S)`, so it sits inside
//!   `a64 ± (2·|c_f32 − a64| + γ_n^{f32}·S) ⊆ a64 ± (3·γ_n^{f32} + 2·γ_n^{f64})·S`.
//! * batched lane: `|a64 − fl32(a64)| ≤ 2^-24·|a64| ≤ 2^-24·(1 + γ_n^{f64})·S`, so
//!   by the same step it sits inside `a64 ± (2^-23·(1+γ_n^{f64}) + γ_n^{f64})·S`.
//!
//! Both are inside `a64 ± R_n·S` with
//!
//! ```text
//! R_n := 3·γ_n^{f32} + 2·γ_n^{f64} + 2^-23
//! ```
//!
//! (the `2^-23` term dominates the batched lane's cast; `3·γ_n^{f32}` dominates the
//! engine lane's), so the two lanes' certified coefficient intervals differ by at
//! most `2·R_n·S` — this is [`COEFF_INTERVAL_SPREAD`] × `R_n·S`.
//!
//! For the POINT coefficients alone (what `LinearBounds::lower_a()` stores, with
//! no error plane) the gap is smaller:
//!
//! ```text
//! r_n := γ_n^{f32} + γ_n^{f64} + 2^-24·(1 + γ_n^{f64})
//! ```
//!
//! (`|c_f32 − a64| + |a64 − fl32(a64)|`).
//!
//! # Propagation to the input space
//!
//! A perturbation `δ` introduced on the coefficient row over some node `v` is
//! carried down by the remaining backward. Two facts bound its effect:
//!
//! 1. **Coefficients.** Every remaining step multiplies by that layer's `W` and,
//!    at a ReLU, by a relaxation slope in `[0, 1]`. So `|δ|` reaching the input
//!    space is at most `|δ|` times the elementwise-ABSOLUTE chain of the
//!    remaining weights. ReLU contributes a factor of at most 1 and is skipped.
//! 2. **Value.** The relaxation used at `v` is built from, and valid over, `v`'s
//!    own box `[l_v, u_v]`, so every point the relaxed region can present at `v`
//!    satisfies `|v_j| ≤ MAG(v)_j := max(|l_vj|, |u_vj|)`. Hence the perturbation
//!    term `Σ_j δ_j·v_j` — whichever way the remaining backward evaluates it —
//!    lies in `± Σ_j |δ_j|·MAG(v)_j`.
//!
//! Fact 2 gives a bound on the AFFINE FUNCTION difference that holds pointwise
//! over the domain, which is what a concretized-bound comparison needs. Fact 1
//! gives the input-space COEFFICIENT difference, which is what an `lower_a()`
//! comparison needs.
//!
//! Note the bias `b` is NOT bounded separately: CROWN's `(A, b)` split is not
//! canonical across lanes and only `A·x + b` is owed. When `0` is inside the
//! domain box, evaluating the pointwise bound at `x = 0` yields
//! `|b₁ − b₂| ≤ affine`, which is how callers should use it — see
//! [`LaneGap::bias_tolerance_requires_origin_in_box`].

use ndarray::Array2;
use ny_tensor::BoundedTensor;

use crate::layers::linear::{crown_single_gamma_n_f32, crown_single_gamma_n_f64};
use crate::LinearBounds;

/// The two lanes' certified coefficient intervals both sit inside `a64 ± R_n·S`,
/// so their endpoints can differ by at most `2·R_n·S`. Named so the `2` in the
/// derivation is not a bare literal.
const COEFF_INTERVAL_SPREAD: f64 = 2.0;

/// `R_n = 3·γ_n^{f32} + 2·γ_n^{f64} + 2^-23` — the common radius, about `a64`,
/// that contains BOTH lanes' certified coefficient intervals. See the module
/// docs for the three-line derivation of each term.
fn certified_interval_radius(n: usize) -> f64 {
    3.0 * crown_single_gamma_n_f32(n) + 2.0 * crown_single_gamma_n_f64(n) + 2f64.powi(-23)
}

/// `r_n = γ_n^{f32} + γ_n^{f64} + 2^-24·(1 + γ_n^{f64})` — the gap between the two
/// lanes' STORED point coefficients (`|c_f32 − a64| + |a64 − fl32(a64)|`), with no
/// error plane involved.
fn point_coefficient_gap(n: usize) -> f64 {
    let g64 = crown_single_gamma_n_f64(n);
    crown_single_gamma_n_f32(n) + g64 + 2f64.powi(-24) * (1.0 + g64)
}

/// One Linear backward step, output-first (the step nearest the spec matrix
/// comes first). ReLU steps are omitted: their relaxation slopes are in `[0, 1]`
/// and therefore can only shrink a perturbation.
pub(super) struct LaneStep {
    /// `|W|` for this Linear layer, shape `(out_features, in_features)`. The
    /// CROWN backward contracts the incoming row over `out_features`, so the
    /// contraction width is `w_abs.nrows()`.
    pub w_abs: Array2<f64>,
    /// `MAG(v)_j = max(|l_j|, |u_j|)` for this layer's INPUT node `v`, taken from
    /// the box the run itself used for that node (root/alpha node bounds when the
    /// caller supplied them, the domain box at the network input).
    pub input_mag: Vec<f64>,
}

impl LaneStep {
    /// Build a step from an f32 weight matrix and an f32 magnitude vector.
    pub(super) fn new(weight: &Array2<f32>, input_mag: &[f32]) -> Self {
        Self {
            w_abs: weight.mapv(|v| f64::from(v.abs())),
            input_mag: input_mag.iter().map(|v| f64::from(v.abs())).collect(),
        }
    }
}

/// The derived tolerances for one fixture + one domain.
pub(super) struct LaneGap {
    /// Uniform bound on `|(A₁·x + b₁) − (A₂·x + b₂)|` over the domain box, for
    /// every objective row. Use for concretized-bound comparisons.
    pub affine: f32,
    /// Bound on `|A₁ − A₂|` entrywise at the network-input coefficient space.
    pub coeff: f32,
}

impl LaneGap {
    /// The bias tolerance is [`Self::affine`] **only when `0` lies inside the
    /// domain box** — it is the pointwise affine bound evaluated at `x = 0`.
    /// Callers must assert that precondition on their own fixture; this method
    /// exists to make the requirement impossible to use by accident.
    pub(super) fn bias_tolerance_requires_origin_in_box(
        &self,
        lower: &[f32],
        upper: &[f32],
    ) -> f32 {
        assert!(
            lower.iter().all(|&v| v <= 0.0) && upper.iter().all(|&v| v >= 0.0),
            "the derived bias tolerance is the affine gap evaluated at x = 0, which \
             requires 0 to be inside the domain box; got lower={lower:?} upper={upper:?}"
        );
        self.affine
    }
}

/// Accumulate the cross-lane gap down a Linear/ReLU chain.
///
/// `spec_abs` is `|C|` for the spec matrix (rows = objectives, cols = network
/// output width); `steps` are the Linear layers output-first.
pub(super) fn cross_lane_gap(spec_abs: &Array2<f64>, steps: &[LaneStep]) -> LaneGap {
    // `t` = elementwise-absolute coefficient chain from the spec down to the
    // CURRENT coefficient space (rows = objectives).
    let mut t = spec_abs.clone();
    // `coeff_acc` = accumulated point-coefficient perturbation, carried into the
    // same space as `t` so later steps multiply it by their own |W| (fact 1).
    let mut coeff_acc = Array2::<f64>::zeros(t.raw_dim());
    let mut affine = 0.0_f64;

    for step in steps {
        let n = step.w_abs.nrows();
        assert_eq!(
            t.ncols(),
            n,
            "lane-gap chain is mis-wired: incoming coefficient width {} does not \
             match this layer's output width {n}",
            t.ncols()
        );
        assert_eq!(
            step.w_abs.ncols(),
            step.input_mag.len(),
            "lane-gap step: |W| has {} input columns but {} input magnitudes",
            step.w_abs.ncols(),
            step.input_mag.len()
        );

        // S[r, j] = Σ_k t[r, k]·|W[k, j]| — the S factor of the certificate, with
        // |A| over-bounded by the absolute chain `t`.
        let s = t.dot(&step.w_abs);

        // Fact 2: this layer's certificate gap, evaluated over its INPUT node box.
        let radius = COEFF_INTERVAL_SPREAD * certified_interval_radius(n);
        let worst_row = (0..s.nrows())
            .map(|r| {
                (0..s.ncols())
                    .map(|j| s[[r, j]] * step.input_mag[j])
                    .sum::<f64>()
            })
            .fold(0.0_f64, f64::max);
        affine += radius * worst_row;

        // Fact 1: carry the already-accumulated coefficient perturbation down one
        // more layer, then add this layer's own point-coefficient gap.
        let point_gap = point_coefficient_gap(n);
        coeff_acc = coeff_acc.dot(&step.w_abs) + s.mapv(|v| v * point_gap);
        t = s;
    }

    let coeff = coeff_acc.iter().fold(0.0_f64, |acc, &v| acc.max(v));
    LaneGap {
        // Round the derived f64 bound UP into f32 so the comparison, which runs in
        // f32, never rejects a difference the derivation actually permits.
        affine: next_up_f32(affine as f32),
        coeff: next_up_f32(coeff as f32),
    }
}

/// Smallest f32 strictly greater than `v` (for finite non-negative `v`).
fn next_up_f32(v: f32) -> f32 {
    f32::from_bits(v.to_bits() + 1)
}

/// Compare two lanes' [`LinearBounds`] within the DERIVED gap, collecting every
/// divergence rather than aborting at the first.
///
/// Shapes must match EXACTLY — a shape change is a structural defect, not float
/// noise. The coefficient planes are compared at [`LaneGap::coeff`]. The biases
/// are compared at the affine tolerance, because CROWN's `(A, b)` split is not
/// canonical across lanes: only `A·x + b` is owed, and evaluating the derived
/// pointwise affine bound at `x = 0` (which the domain box must contain) gives
/// the bias bound.
///
/// The certified-error planes are deliberately NOT compared: they are exactly
/// where the two lanes are supposed to differ (`γ_n^{f32}·S` vs `γ_n^{f64}·S`),
/// and the property that survives that difference is the concretized bound,
/// which callers assert separately.
pub(super) fn assert_linear_bounds_within_gap(
    label: &str,
    actual: &LinearBounds,
    baseline: &LinearBounds,
    gap: &LaneGap,
    domain: &BoundedTensor,
) {
    assert_eq!(
        actual.lower_a().dim(),
        baseline.lower_a().dim(),
        "{label} lower_a shape diverged from baseline"
    );
    assert_eq!(
        actual.upper_a().dim(),
        baseline.upper_a().dim(),
        "{label} upper_a shape diverged from baseline"
    );
    assert_eq!(
        actual.lower_b().len(),
        baseline.lower_b().len(),
        "{label} lower_b shape diverged from baseline"
    );
    assert_eq!(
        actual.upper_b().len(),
        baseline.upper_b().len(),
        "{label} upper_b shape diverged from baseline"
    );

    let lower: Vec<f32> = domain.lower().iter().copied().collect();
    let upper: Vec<f32> = domain.upper().iter().copied().collect();
    let bias_tol = gap.bias_tolerance_requires_origin_in_box(&lower, &upper);

    let fields: [(&str, Vec<f32>, Vec<f32>, f32); 4] = [
        (
            "lower_a",
            actual.lower_a().iter().copied().collect(),
            baseline.lower_a().iter().copied().collect(),
            gap.coeff,
        ),
        (
            "upper_a",
            actual.upper_a().iter().copied().collect(),
            baseline.upper_a().iter().copied().collect(),
            gap.coeff,
        ),
        (
            "lower_b",
            actual.lower_b().iter().copied().collect(),
            baseline.lower_b().iter().copied().collect(),
            bias_tol,
        ),
        (
            "upper_b",
            actual.upper_b().iter().copied().collect(),
            baseline.upper_b().iter().copied().collect(),
            bias_tol,
        ),
    ];
    let mut failures = Vec::new();
    for (field, actual_values, baseline_values, tol) in fields {
        for (idx, (a, b)) in actual_values.iter().zip(baseline_values.iter()).enumerate() {
            let diff = (a - b).abs();
            if diff > tol {
                failures.push(format!(
                    "{label} {field}[{idx}]: batched {a} vs baseline {b}, diff {diff} exceeds \
                     derived tolerance {tol:e}"
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// `max(|l|, |u|)` per coordinate of a node's box — the `MAG(v)` of the module
/// derivation.
pub(super) fn box_magnitudes(bounds: &BoundedTensor) -> Vec<f32> {
    bounds
        .lower()
        .iter()
        .zip(bounds.upper().iter())
        .map(|(&l, &u)| l.abs().max(u.abs()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One Linear step of contraction width `n`, fed by an all-ones spec row of
    /// the matching width. `t.ncols()` must equal `w_abs.nrows()` — the chain
    /// assertion inside [`cross_lane_gap`] enforces exactly that.
    fn one_step(n: usize, input_mag: f64) -> LaneGap {
        cross_lane_gap(
            &Array2::from_elem((1, n), 1.0),
            &[LaneStep {
                w_abs: Array2::from_elem((n, 4), 1.0),
                input_mag: vec![input_mag; 4],
            }],
        )
    }

    /// The derivation must scale with the contraction width `n` and with the S
    /// factor. A constant would pass the fixtures and fail this.
    #[test]
    fn lane_gap_grows_with_contraction_width_and_scale() {
        let narrow = one_step(1, 1.0);
        let wide = one_step(64, 1.0);
        assert!(
            wide.affine > narrow.affine * 8.0,
            "γ_n is linear in n: a 64-wide contraction must dominate a 1-wide one \
             (narrow={}, wide={})",
            narrow.affine,
            wide.affine
        );

        let big_box = one_step(1, 1000.0);
        assert!(
            big_box.affine > narrow.affine * 100.0,
            "the affine gap is S-scaled and box-scaled (narrow={}, big_box={})",
            narrow.affine,
            big_box.affine
        );
    }

    /// The point-coefficient gap must be strictly smaller than the certified
    /// interval gap — the whole reason the two tolerances are separate.
    #[test]
    fn point_coefficient_gap_is_smaller_than_the_certified_interval_gap() {
        for n in [1_usize, 2, 4, 64, 4096] {
            assert!(
                point_coefficient_gap(n) < certified_interval_radius(n),
                "n={n}: point gap {} should be below the certified radius {}",
                point_coefficient_gap(n),
                certified_interval_radius(n)
            );
        }
    }

    #[test]
    fn bias_tolerance_rejects_a_box_that_excludes_the_origin() {
        let gap = LaneGap {
            affine: 1e-6,
            coeff: 1e-7,
        };
        let caught = std::panic::catch_unwind(|| {
            gap.bias_tolerance_requires_origin_in_box(&[0.25], &[0.75])
        });
        assert!(
            caught.is_err(),
            "a box that excludes 0 must not yield a bias tolerance"
        );
        assert_eq!(
            gap.bias_tolerance_requires_origin_in_box(&[-0.25], &[0.75]),
            1e-6
        );
    }
}
