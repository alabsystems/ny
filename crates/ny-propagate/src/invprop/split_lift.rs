// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Split-lifting: express a BaB neuron split as an INPUT-BASIS dual row so split
//! information reaches layers BELOW the split.
//!
//! See `docs/INVPROP_ASSUME_VIOLATION_DESIGN.md` §5. This is the novel channel
//! orthogonal to beta-CROWN: beta injects at the split neuron's own column and
//! contributes exactly zero to the intermediate bounds of layers `k < j`
//! (their backward starts at `k` and never visits `j`) — precisely NY's measured
//! `<= 0.002%` wall. Split-lifting re-expresses the layer-`j` split as an
//! input-basis affine row `g(x) >= 0` that is PRESENT at `z_k`'s input
//! concretization, re-tightening `[l_k, u_k]` under the split assumption.
//!
//! **RESEARCH / high-variance / ENV-GATED OFF** (`NY_INVPROP_SPLIT_LIFT`). Ships
//! disabled behind the flag until validated on real deep-resnets, mirroring the
//! kFSB "measure-before-enable" discipline. It NEVER runs on the default path.
//!
//! # Soundness (per child, exact arithmetic)
//!
//! On the BaB child `S`, the lifted row is chosen so `g_true(x) >= 0` for all
//! `x in S`:
//! - **Active** child (`z_i >= 0`): `g = ` the CROWN UPPER affine bound of `z_i`
//!   (`g >= z_i >= 0`).
//! - **Inactive** child (`z_i <= 0`): `g = -(`CROWN LOWER affine bound of `z_i)`
//!   (`g = -(a_l x + c_l) >= -z_i >= 0`).
//!
//! For any lower bound `f(x) = A_k x + b_k` of a lower-layer neuron (or the
//! objective) and any `gamma_split >= 0`:
//! `f(x) >= f(x) - gamma_split * g(x) >= min_box[f - gamma_split * g]` for `x in S`
//! (because `g_true >= 0` on `S` and `S subset box`). So the concretized minimum
//! of `f - gamma_split*g` is a valid lower bound of `min_S f`. A wrong/loose
//! `gamma_split` only subtracts a certified-nonneg quantity => it can weaken,
//! never inflate.
//!
//! # The false-HOLD firewall (`gamma_split * g_err`)
//!
//! The lifted coefficient `a_g` is itself a CROWN-derived f32 carrying its OWN
//! certified error `g_err` (`|stored - true| <= g_err`). Folding only the
//! single-mutation ULP gap (as the beta-split helper does) would leave the total
//! error short by `gamma_split * g_err`; near the child boundary the
//! effectively-subtracted quantity could dip below the true `gamma_split*g_true`,
//! inflating the bound. This fold therefore carries `gamma_split * g_err` (plus
//! the mutation rounding) OUTWARD into the target's certified error matrix, so
//! `concretize_sound` discharges it and the stored row stays `<=` the true
//! `f - gamma_split*g_true`.

// Gated Stage-4 research API: the sound fold, per-child sign selection, and the
// `gamma_split * g_err` firewall are implemented and unit-tested below. Wiring the
// split-neuron input-basis affine-form CAPTURE into the BaB backward is the flagged
// validation phase (design §5.7), so several items here are only exercised by tests
// and the (flag-off) future capture path in a normal build.
#![allow(dead_code)]

use crate::bounds::LinearBounds;
use ndarray::{Array1, Array2};
use ny_tensor::{next_down_f32, next_up_f32};

/// Which BaB child a split constraint belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChildSense {
    /// `z_i >= 0` (ReLU forced active on this child).
    Active,
    /// `z_i <= 0` (ReLU forced inactive on this child).
    Inactive,
}

/// Whether split-lifting is enabled (`NY_INVPROP_SPLIT_LIFT` truthy). Default OFF.
#[must_use]
pub fn split_lift_enabled() -> bool {
    std::env::var("NY_INVPROP_SPLIT_LIFT")
        .ok()
        .map(|v| {
            let v = v.trim();
            v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("on")
        })
        .unwrap_or(false)
}

/// A lifted, provably-nonneg-on-the-child affine row `g(x) = a_g . x + c_g`, with
/// certified per-coefficient outward error `err` (`|stored - true| <= err`),
/// derived from a split neuron's CROWN affine bound in the input basis.
#[derive(Debug, Clone)]
pub struct LiftedRow {
    /// Input-basis coefficients of `g`.
    pub a_g: Array1<f32>,
    /// Constant term of `g`.
    pub c_g: f32,
    /// Per-coefficient certified outward error of `a_g` (`>= 0`).
    pub err: Array1<f32>,
}

impl LiftedRow {
    /// Build the child-appropriate `g >= 0` row from neuron `z_i`'s CROWN affine
    /// LOWER (`a_l, c_l, l_err`) and UPPER (`a_u, c_u, u_err`) bounds in the input
    /// basis. The per-child sign selection is the load-bearing soundness choice
    /// (using the wrong side yields no `g >= 0` guarantee).
    #[must_use]
    pub fn from_neuron_bound(
        sense: ChildSense,
        a_l: &Array1<f32>,
        c_l: f32,
        l_err: &Array1<f32>,
        a_u: &Array1<f32>,
        c_u: f32,
        u_err: &Array1<f32>,
    ) -> Self {
        match sense {
            // Active: g = upper bound of z_i (g >= z_i >= 0 on the child).
            ChildSense::Active => LiftedRow {
                a_g: a_u.clone(),
                c_g: c_u,
                err: u_err.clone(),
            },
            // Inactive: g = -(lower bound of z_i) (g = -(a_l x + c_l) >= -z_i >= 0).
            ChildSense::Inactive => LiftedRow {
                a_g: a_l.mapv(|v| -v),
                c_g: -c_l,
                // Error magnitude is unaffected by negation.
                err: l_err.clone(),
            },
        }
    }

    /// Evaluate the stored (f32) affine form at a concrete input point.
    #[must_use]
    pub fn eval(&self, x: &[f32]) -> f32 {
        let mut acc = self.c_g as f64;
        for (k, &xk) in x.iter().enumerate() {
            acc += self.a_g[k] as f64 * xk as f64;
        }
        acc as f32
    }
}

/// Fold `-gamma_split * g` into the LOWER bound of output row `out_row` of
/// `target`, carrying `gamma_split * g.err` OUTWARD into `lower_a_err` (the
/// false-HOLD firewall).
///
/// SOUND for any `gamma_split >= 0` and any `g` with `g_true >= 0` on the child
/// (guaranteed by [`LiftedRow::from_neuron_bound`]). With `gamma_split == 0` this
/// is the identity map.
pub fn fold_split_lift_lower(
    target: &mut LinearBounds,
    out_row: usize,
    g: &LiftedRow,
    gamma_split: f32,
) {
    if gamma_split == 0.0 || !gamma_split.is_finite() {
        return;
    }
    let (n_out, n_in) = (target.lower_a.nrows(), target.lower_a.ncols());
    if out_row >= n_out || g.a_g.len() != n_in || g.err.len() != n_in {
        return; // shape guard: fold nothing rather than risk an unsound partial write
    }

    // Materialize BOTH err matrices before storing a round-to-nearest A-delta: a
    // `None` err marks coefficients EXACT and silently skips the concretize
    // outward penalty (the same trap as the seed augment).
    if target.lower_a_err.is_none() {
        target.lower_a_err = Some(Array2::<f32>::zeros((n_out, n_in)));
    }
    if target.upper_a_err.is_none() {
        target.upper_a_err = Some(Array2::<f32>::zeros((n_out, n_in)));
    }

    let g64 = gamma_split as f64;
    for k in 0..n_in {
        let ag = g.a_g[k] as f64;
        if ag == 0.0 && g.err[k] == 0.0 {
            continue;
        }
        let old = target.lower_a[[out_row, k]] as f64;
        let exact = old - g64 * ag; // subtract gamma_split * g
        let stored = exact as f32;
        let mutation_gap = (exact - stored as f64).abs();
        target.lower_a[[out_row, k]] = stored;
        // Certified outward error: existing + mutation rounding + gamma_split*g_err.
        let add = next_up_f32((mutation_gap + g64 * g.err[k] as f64) as f32);
        if let Some(le) = target.lower_a_err.as_mut() {
            le[[out_row, k]] = next_up_f32(le[[out_row, k]] + add);
        }
    }
    // Bias: lower -= gamma_split * c_g, directed DOWN (outward for a lower bound).
    let exact_b = target.lower_b[out_row] as f64 - g64 * g.c_g as f64;
    target.lower_b[out_row] = next_down_f32(exact_b as f32);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};
    use ny_tensor::BoundedTensor;

    fn box_2d(l: [f32; 2], u: [f32; 2]) -> BoundedTensor {
        BoundedTensor::new(arr1(&l).into_dyn(), arr1(&u).into_dyn()).unwrap()
    }

    /// SOUNDNESS oracle: after folding a correct child row `g >= 0`, the
    /// concretized lower bound is `<=` `f(x)` at every sampled input in the child
    /// region `{x : g(x) >= 0}`. (A wrong-side fold would produce a lower bound
    /// exceeding `f` somewhere in that region — see the next test.)
    #[test]
    fn split_lift_lower_is_sound_over_child() {
        // f(x) = x0 - x1 over the box [-1,1]^2 (min_box f = -2).
        let a = arr2(&[[1.0f32, -1.0]]);
        let b = arr1(&[0.0f32]);
        let mut target = LinearBounds::symmetric(a, b).unwrap();
        let input = box_2d([-1.0, -1.0], [1.0, 1.0]);

        // Active child of a neuron whose UPPER affine bound is g(x) = x0 + x1 + 0.5
        // (>= 0 on the child region). Exact (err = 0) for the base soundness check.
        let a_u = arr1(&[1.0f32, 1.0]);
        let a_l = arr1(&[0.0f32, 0.0]);
        let zero = arr1(&[0.0f32, 0.0]);
        let g =
            LiftedRow::from_neuron_bound(ChildSense::Active, &a_l, 0.0, &zero, &a_u, 0.5, &zero);

        fold_split_lift_lower(&mut target, 0, &g, 0.75);
        let folded = target.concretize_sound(&input);
        let lower = folded.lower()[[0]];

        // Sample the box; on the child {g >= 0}, the folded lower must not exceed f.
        for i in 0..=20 {
            for j in 0..=20 {
                let x0 = -1.0 + i as f32 / 10.0;
                let x1 = -1.0 + j as f32 / 10.0;
                if g.eval(&[x0, x1]) >= 0.0 {
                    let fx = x0 - x1;
                    assert!(
                        lower <= fx + 1e-5,
                        "split-lift lower {lower} exceeds f({x0},{x1})={fx} on the child",
                    );
                }
            }
        }
    }

    /// The per-child sign selection is load-bearing: using the WRONG side (the
    /// lower bound for an Active child) yields a `g` that is negative somewhere on
    /// the child, so a fold with it would NOT be `g_true >= 0`-backed. This test
    /// documents that `from_neuron_bound(Active)` picks the UPPER bound (g >= 0),
    /// and that the alternative (lower bound) does go negative on the region.
    #[test]
    fn split_lift_per_child_sign_selection() {
        let a_l = arr1(&[1.0f32, 0.0]); // lower bound z_i >= x0 (negative for x0<0)
        let a_u = arr1(&[1.0f32, 0.0]);
        let zero = arr1(&[0.0f32, 0.0]);
        // Active picks the UPPER: g = a_u x + 0.5.
        let g_active =
            LiftedRow::from_neuron_bound(ChildSense::Active, &a_l, -0.5, &zero, &a_u, 0.5, &zero);
        assert_eq!(g_active.c_g, 0.5);
        assert_eq!(g_active.a_g, a_u);
        // The wrong side (lower, c=-0.5) evaluates negative at x0=0 => not g>=0.
        assert!((a_l[0] * 0.0 + (-0.5)) < 0.0);
        // Inactive negates the lower bound: g = -(a_l x + c_l).
        let g_inactive =
            LiftedRow::from_neuron_bound(ChildSense::Inactive, &a_l, -0.5, &zero, &a_u, 0.5, &zero);
        assert_eq!(g_inactive.a_g, arr1(&[-1.0f32, 0.0]));
        assert_eq!(g_inactive.c_g, 0.5);
    }

    /// `gamma_split == 0` is the identity map (no coefficient/bias/err change).
    #[test]
    fn split_lift_gamma_zero_is_identity() {
        let a = arr2(&[[1.0f32, -1.0]]);
        let b = arr1(&[0.0f32]);
        let mut target = LinearBounds::symmetric(a.clone(), b.clone()).unwrap();
        let g = LiftedRow {
            a_g: arr1(&[1.0f32, 1.0]),
            c_g: 0.5,
            err: arr1(&[0.1f32, 0.1]),
        };
        fold_split_lift_lower(&mut target, 0, &g, 0.0);
        assert_eq!(target.lower_a, a);
        assert_eq!(target.lower_b, b);
        assert!(target.lower_a_err.is_none());
    }

    /// The `gamma_split * g_err` firewall: with a nonzero coefficient error on `g`,
    /// the fold must attach outward error so the concretized lower stays sound even
    /// when the stored `a_g` under-represents the true coefficient by up to `err`.
    #[test]
    fn split_lift_carries_g_err_outward() {
        let a = arr2(&[[1.0f32, 0.0]]);
        let b = arr1(&[0.0f32]);
        let mut target = LinearBounds::symmetric(a, b).unwrap();
        let input = box_2d([-1.0, -1.0], [1.0, 1.0]);

        // g = x0 + 0.5 with a large certified error on the x0 coefficient.
        let g = LiftedRow {
            a_g: arr1(&[1.0f32, 0.0]),
            c_g: 0.5,
            err: arr1(&[0.5f32, 0.0]),
        };
        fold_split_lift_lower(&mut target, 0, &g, 1.0);
        // Error matrix must be materialized and carry the gamma*g_err contribution.
        let le = target
            .lower_a_err
            .as_ref()
            .expect("err must be materialized");
        assert!(
            le[[0, 0]] >= 0.5 - 1e-6,
            "lower_a_err must carry gamma*g_err (>= 0.5), got {}",
            le[[0, 0]]
        );
        // Soundness over the child {g >= 0}: folded lower <= f(x) = x0.
        let folded = target.concretize_sound(&input);
        let lower = folded.lower()[[0]];
        for i in 0..=20 {
            let x0 = -1.0 + i as f32 / 10.0;
            if g.eval(&[x0, 0.0]) >= 0.0 {
                assert!(
                    lower <= x0 + 1e-5,
                    "split-lift lower {lower} exceeds f={x0} with g_err present",
                );
            }
        }
    }
}
