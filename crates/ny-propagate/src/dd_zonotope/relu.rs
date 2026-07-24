// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Certified DeepZ ReLU transformer for the double-double zonotope
//! (`#dd-zonotope`).
//!
//! # The relaxation, and why the certificate does not depend on `lambda`
//!
//! For a pre-activation `x` certified to lie in `[lo, up]` and ANY slope
//! `lambda in [0, 1]`, the deviation `d(x) = relu(x) - lambda*x` satisfies
//!
//! ```text
//! d(x) in [0, M],   M = max(-lambda*lo, (1 - lambda)*up)
//! ```
//!
//! (on `[lo, 0]`, `d = -lambda*x` rises to `-lambda*lo`; on `[0, up]`,
//! `d = (1-lambda)*x` rises to `(1-lambda)*up`). Hence
//!
//! ```text
//! relu(x) = lambda*x + M/2 + s,   |s| <= M/2.
//! ```
//!
//! This module computes `M` from BOTH branch expressions and takes the max,
//! rounded outward. That makes the certificate independent of how `lambda` was
//! rounded: any `lambda in [0,1]` is sound, so the f64 division
//! `up / (up - lo)` needs no error term of its own. (Using the textbook
//! `mu = -lambda*lo/2` and *assuming* `lambda` is exactly `up/(up-lo)` would
//! silently under-widen when the division rounds up.)
//!
//! `lambda <= 1`, so ReLU CONTRACTS both error channels — it is never an
//! amplifier.
//!
//! # The `mu` spend policy (a tractability guard, not an optimisation)
//!
//! Each crossing neuron's `+-M/2` term can be spent as a new zonotope
//! generator (keeping it symbolic so it can cancel downstream) or folded into
//! the interval channel `ec` (sound: an interval over-approximates a `+-mu`
//! symbol). The reference probe MEASURED that at plain f64 the inflated error
//! channel manufactures 31121 spurious crossings; a hard generator cap without
//! a fold rule would then simply refuse.
//!
//! The rule here is RELATIVE, not an absolute `tau`: fold iff
//! `mu <= FOLD_RATIO * (ec_i + eg_i)`. Two properties follow:
//!
//! * A rounding-manufactured crossing has `|lo| <~ ec + eg` by construction,
//!   so `mu <~ (ec + eg)/2` and it is folded — no spurious column.
//! * A GENUINE relaxation is orders of magnitude larger than the error channel
//!   (measured: `1e-6` relaxation vs `1e-12` rounding), so it is never folded.
//! * The fold can inflate `ec_i` by at most `FOLD_RATIO * (ec_i + eg_i)`, i.e.
//!   a bounded factor per layer — unlike an absolute `tau`, which could inject
//!   a fixed `tau` at layer 1 and see it amplified by the measured `~2^66`.

use ny_core::dd::{dd_add_f64, dd_mul_f64, next_up_f64, U_DD, U_F64};

use super::state::{err_up, DdZono};

/// Fold a relaxation term into the interval channel when it is no larger than
/// this multiple of the error channel already carried at that element.
pub(crate) const FOLD_RATIO: f64 = 1.0;

/// Round a relaxation magnitude OUTWARD, keeping an exact zero exact.
///
/// `next_up_f64(0.0)` is the smallest positive subnormal, which would make
/// every structurally-exact op (a stable ReLU, a strictly dominated max-pool
/// window) report a non-zero relaxation and buy a generator column for it.
#[inline]
pub(crate) fn up_nonneg(x: f64) -> f64 {
    if x <= 0.0 {
        0.0
    } else {
        next_up_f64(x)
    }
}

/// Outcome of one ReLU/MaxPool relaxation step.
pub(crate) struct RelaxOutcome {
    /// Elements that bought a new generator column, with their coefficients.
    pub(crate) spent: Vec<(usize, f64)>,
    /// How many elements were crossing (ReLU) / had a non-zero slack (MaxPool).
    pub(crate) relaxed: usize,
    /// How many relaxation terms were folded into the interval channel.
    pub(crate) folded: usize,
}

/// Apply the certified DeepZ ReLU in place.
///
/// Returns `None` when any certified bound is non-finite (the certificate is
/// then meaningless and the caller must refuse, never publish).
pub(crate) fn apply_relu(z: &mut DdZono) -> Option<RelaxOutcome> {
    let n = z.numel();
    let rad = z.radius();
    let (lo, up) = z.concretize_with_radius(&rad);
    // Fold budget: the NON-RADIUS half-width actually used by the
    // concretization, which includes the `2u|c|` double-double collapse term.
    // Sizing the budget off `ec + eg` alone under-counts it by orders of
    // magnitude at VGG16 activation scales — see `DdZono::error_half_width`.
    let err = z.error_half_width(&rad, &lo, &up);

    let mut spent: Vec<(usize, f64)> = Vec::new();
    let mut relaxed = 0usize;
    let mut folded = 0usize;
    let mut lams = vec![0.0_f64; n];

    for i in 0..n {
        if !lo[i].is_finite() || !up[i].is_finite() || lo[i] > up[i] {
            return None;
        }
        let (lam, mu) = if lo[i] >= 0.0 {
            (1.0_f64, 0.0_f64)
        } else if up[i] <= 0.0 {
            (0.0_f64, 0.0_f64)
        } else {
            relaxed += 1;
            let den = up[i] - lo[i];
            let lam = (up[i] / den).clamp(0.0, 1.0);
            // M = max(-lam*lo, (1-lam)*up), rounded OUTWARD. Valid for ANY
            // lam in [0,1], so lam's own rounding needs no error term.
            let m1 = up_nonneg(-lam * lo[i]);
            let m2 = up_nonneg((1.0 - lam) * up[i]);
            let m = up_nonneg(m1.max(m2));
            (lam, m * 0.5)
        };
        lams[i] = lam;

        let abs_c = z.center[i].abs_upper();
        let newc = dd_add_f64(dd_mul_f64(z.center[i], lam), mu);
        z.center[i] = newc;
        // lam * ec transports the incoming center error (contraction, lam<=1);
        // U_DD * (|c| + 2mu) pays for the two double-double ops just performed.
        let mut ec_i = err_up(lam * z.ec[i] + U_DD * (abs_c + 2.0 * mu));
        // lam * eg transports the generator error; U_F64 * lam * rad pays for
        // the f64 rounding of the `gens[j][i] *= lam` rescale below.
        let eg_i = err_up(lam * z.eg[i] + U_F64 * lam * rad[i]);

        if mu > 0.0 {
            if mu <= FOLD_RATIO * err[i] {
                ec_i = err_up(ec_i + mu);
                folded += 1;
            } else {
                spent.push((i, mu));
            }
        }
        z.ec[i] = ec_i;
        z.eg[i] = eg_i;
    }

    for g in z.gens.iter_mut() {
        for (v, lam) in g.iter_mut().zip(lams.iter()) {
            *v *= *lam;
        }
    }

    Some(RelaxOutcome {
        spent,
        relaxed,
        folded,
    })
}

/// Drop generator columns that became identically zero (ReLU with `lam == 0`
/// zeroes many). Purely a memory/time saving — the represented set is
/// unchanged.
pub(crate) fn prune_zero_generators(z: &mut DdZono) {
    z.gens.retain(|g| g.iter().any(|v| *v != 0.0));
}
