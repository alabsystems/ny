// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Certified MaxPool2d transformer for the double-double zonotope
//! (`#dd-zonotope`).
//!
//! # The relaxation
//!
//! Identical rule to the one already proven in ny's CROWN path
//! (`layers/pooling/max.rs:258` `propagate_linear_with_bounds`, comments at
//! 454-481) — only the OUTPUT form is new (a zonotope symbol instead of a
//! linear row + bias interval).
//!
//! For a window `{x_i}` with certified boxes `[l_i, u_i]`, pick
//! `i* = argmax_i l_i`. Then
//!
//! ```text
//! max_i x_i >= x_{i*}                                        (exact)
//! max_i x_i <= x_{i*} + sum_{i != i*} max(u_i - l_{i*}, 0)    (slack)
//! ```
//!
//! The upper rule holds because `max_i x_i - x_{i*} = max_i (x_i - x_{i*})`,
//! each term is at most `max(u_i - l_{i*}, 0)`, the `i = i*` term is `0`, and
//! a sum of nonnegative terms dominates their max. It is ZERO when the window
//! is strictly dominated — the overwhelmingly common case here.
//!
//! So `y = x_{i*} + t` with `t in [0, slack]`, which is the zonotope element
//! `center += slack/2` plus a `+- slack/2` symbol. `slack` is rounded OUTWARD
//! once; the halving is exact.
//!
//! # What is deliberately refused
//!
//! Padding is rejected (`None`): ny's MaxPool2d carries a
//! `use_negative_inf_padding` mode whose interaction with the argmax rule is
//! not exercised by any instance this lane targets, and a silently wrong
//! window enumeration is exactly risk R2. VGG16's pools are `2x2 / stride 2 /
//! pad 0`.

use ny_core::dd::{dd_add_f64, U_DD};

use super::relu::{up_nonneg, RelaxOutcome, FOLD_RATIO};
use super::state::{err_up, DdZono};

/// MaxPool2d geometry (unbatched NCHW, no padding).
#[derive(Debug, Clone, Copy)]
pub(crate) struct PoolPlan {
    pub(crate) c: usize,
    pub(crate) in_h: usize,
    pub(crate) in_w: usize,
    pub(crate) out_h: usize,
    pub(crate) out_w: usize,
    pub(crate) kh: usize,
    pub(crate) kw: usize,
    pub(crate) sh: usize,
    pub(crate) sw: usize,
}

impl PoolPlan {
    /// Build the plan, refusing anything with padding or a degenerate window.
    pub(crate) fn build(
        in_shape: (usize, usize, usize),
        kernel: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Option<Self> {
        if padding != (0, 0) || stride.0 == 0 || stride.1 == 0 || kernel.0 == 0 || kernel.1 == 0 {
            return None;
        }
        let (c, in_h, in_w) = in_shape;
        if c == 0 {
            return None;
        }
        if in_h < kernel.0 || in_w < kernel.1 {
            return None;
        }
        let out_h = (in_h - kernel.0) / stride.0 + 1;
        let out_w = (in_w - kernel.1) / stride.1 + 1;
        let plan = PoolPlan {
            c,
            in_h,
            in_w,
            out_h,
            out_w,
            kh: kernel.0,
            kw: kernel.1,
            sh: stride.0,
            sw: stride.1,
        };
        plan.checked_sizes()?;
        Some(plan)
    }

    fn checked_sizes(&self) -> Option<(usize, usize, usize)> {
        if self.c == 0
            || self.kh == 0
            || self.kw == 0
            || self.sh == 0
            || self.sw == 0
            || self.in_h < self.kh
            || self.in_w < self.kw
        {
            return None;
        }
        let expected_out_h = (self.in_h - self.kh) / self.sh + 1;
        let expected_out_w = (self.in_w - self.kw) / self.sw + 1;
        if (self.out_h, self.out_w) != (expected_out_h, expected_out_w) {
            return None;
        }
        let in_hw = self.in_h.checked_mul(self.in_w)?;
        let out_hw = self.out_h.checked_mul(self.out_w)?;
        self.c.checked_mul(in_hw)?;
        let n_out = self.c.checked_mul(out_hw)?;
        Some((in_hw, out_hw, n_out))
    }

    #[cfg(test)]
    pub(crate) fn out_numel(&self) -> usize {
        // Every plan returned by `build` has passed `checked_sizes`; runtime
        // consumers revalidate before indexing because the fields are crate-visible.
        self.c * self.out_h * self.out_w
    }
}

/// Apply the certified MaxPool2d transformer, producing a NEW state.
///
/// Returns `None` when a certified bound is non-finite.
pub(crate) fn apply_maxpool(z: &DdZono, plan: &PoolPlan) -> Option<(DdZono, RelaxOutcome)> {
    let (ihw, ohw, n_out) = plan.checked_sizes()?;
    if !z.has_valid_layout() || z.shape.as_slice() != [plan.c, plan.in_h, plan.in_w] {
        return None;
    }
    let rad = z.radius();
    let (lo, up) = z.concretize_with_radius(&rad);
    if lo.iter().chain(up.iter()).any(|v| !v.is_finite()) {
        return None;
    }

    let err = z.error_half_width(&rad, &lo, &up);
    let mut sel = vec![0usize; n_out];
    let mut half = vec![0.0_f64; n_out];
    // Exact bound on the part of each window's slack that the ERROR CHANNEL
    // alone could have manufactured; the fold rule below is sized against it.
    let mut err_win = vec![0.0_f64; n_out];
    let mut relaxed = 0usize;

    for ch in 0..plan.c {
        for oy in 0..plan.out_h {
            for ox in 0..plan.out_w {
                let o = ch * ohw + oy * plan.out_w + ox;
                // Enumerate the window.
                let mut star = usize::MAX;
                let mut l_star = f64::NEG_INFINITY;
                for ky in 0..plan.kh {
                    let iy = oy * plan.sh + ky;
                    for kx in 0..plan.kw {
                        let ix = ox * plan.sw + kx;
                        let idx = ch * ihw + iy * plan.in_w + ix;
                        if lo[idx] > l_star || star == usize::MAX {
                            l_star = lo[idx];
                            star = idx;
                        }
                    }
                }
                debug_assert_ne!(star, usize::MAX);
                let mut slack = 0.0_f64;
                // `u_i - l_star = (c_i - c_star) + (rad_i + rad_star)
                //                 + (e_i + e_star)`. The last group is the only
                // part the ERROR channel contributes, so a slack that is purely
                // rounding-manufactured obeys
                // `slack <= sum_{i != star} (e_i + e_star)`. That exact sum is
                // the fold budget below — sizing it off the selected element
                // alone under-counts by the window size and makes every window
                // of equal activations (after ReLU, most of them are exactly
                // zero) buy a generator column.
                let e_star = err[star];
                let mut err_budget = 0.0_f64;
                for ky in 0..plan.kh {
                    let iy = oy * plan.sh + ky;
                    for kx in 0..plan.kw {
                        let ix = ox * plan.sw + kx;
                        let idx = ch * ihw + iy * plan.in_w + ix;
                        if idx == star {
                            continue;
                        }
                        err_budget += err[idx] + e_star;
                        let d = up[idx] - l_star;
                        if d > 0.0 {
                            slack += d;
                        }
                    }
                }
                err_win[o] = err_budget;
                let slack = up_nonneg(slack);
                sel[o] = star;
                // Exact halving.
                half[o] = slack * 0.5;
                if slack > 0.0 {
                    relaxed += 1;
                }
            }
        }
    }

    let mut center = Vec::with_capacity(n_out);
    let mut ec = vec![0.0_f64; n_out];
    let mut eg = vec![0.0_f64; n_out];
    let mut spent: Vec<(usize, f64)> = Vec::new();
    let mut folded = 0usize;

    for o in 0..n_out {
        let s = sel[o];
        let abs_c = z.center[s].abs_upper();
        let newc = dd_add_f64(z.center[s], half[o]);
        center.push(newc);
        let mut ec_o = err_up(z.ec[s] + U_DD * (abs_c + 2.0 * half[o]));
        let eg_o = z.eg[s];
        if half[o] > 0.0 {
            // Budget = 2x the exact error-only bound. The factor of two is not
            // slack for its own sake: a window of EQUAL activations (after
            // ReLU, most windows are all-zero) produces `slack` exactly equal
            // to `err_win`, and `slack` is then rounded outward by one ulp, so
            // an exact-equality test misses every one of them by a single ulp.
            // MEASURED on vgg16-7: 286446 of 802816 windows at the first pool.
            // Still self-limiting — the fold adds at most `err_win` to `ec`,
            // a bounded factor per pool layer.
            let window_budget = err_win[o];
            if half[o] <= FOLD_RATIO * (ec_o + eg_o) || half[o] <= window_budget {
                ec_o = err_up(ec_o + half[o]);
                folded += 1;
            } else {
                spent.push((o, half[o]));
            }
        }
        ec[o] = ec_o;
        eg[o] = eg_o;
    }

    let gens: Vec<Vec<f64>> = z
        .gens
        .iter()
        .map(|g| sel.iter().map(|&s| g[s]).collect())
        .collect();

    Some((
        DdZono {
            shape: vec![plan.c, plan.out_h, plan.out_w],
            center,
            gens,
            ec,
            eg,
        },
        RelaxOutcome {
            spent,
            relaxed,
            folded,
        },
    ))
}

#[cfg(test)]
mod plan_validation_tests {
    use ny_core::dd::Dd;

    use super::{apply_maxpool, PoolPlan};
    use crate::dd_zonotope::state::DdZono;

    fn state() -> DdZono {
        DdZono {
            shape: vec![1, 2, 2],
            center: vec![Dd::ZERO; 4],
            gens: vec![vec![0.0; 4]],
            ec: vec![0.0; 4],
            eg: vec![0.0; 4],
        }
    }

    #[test]
    fn pool_plan_rejects_zero_channels_and_overflowing_products() {
        assert!(PoolPlan::build((0, 2, 2), (1, 1), (1, 1), (0, 0)).is_none());
        assert!(PoolPlan::build((usize::MAX, 2, 2), (1, 1), (1, 1), (0, 0)).is_none());
    }

    #[test]
    fn maxpool_refuses_malformed_plan_or_state_before_indexing() {
        let mut malformed_plan = PoolPlan::build((1, 2, 2), (1, 1), (1, 1), (0, 0)).unwrap();
        malformed_plan.out_h = 3;
        assert!(apply_maxpool(&state(), &malformed_plan).is_none());

        let valid_plan = PoolPlan::build((1, 2, 2), (1, 1), (1, 1), (0, 0)).unwrap();
        let mut malformed_state = state();
        malformed_state.ec.pop();
        assert!(apply_maxpool(&malformed_state, &valid_plan).is_none());
    }
}
