// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #row-weights — the row-player of the max-min α objective (DARK: its only
//! production caller is independently gated; §5 of
//! `docs/THEORY_EXACT_GAP_ATTRIBUTION_AND_MAXMIN_ALPHA_2026-08-03.md`).
//!
//! The property ny verifies on cifar100 is a conjunction of 99 margin rows, so
//! the α objective is a **max-min**, not a sum:
//!
//! ```text
//!   V = max_alpha min_r slack_r(alpha) = max_alpha min_{p in simplex} SUM_r p_r slack_r(alpha)
//! ```
//!
//! a zero-sum game between an α-player and a row-player on the simplex. This
//! module is the row-player, run as multiplicative weights (exponentiated
//! gradient):
//!
//! ```text
//!   loss_{t,r} = (slack_{t,r} - min(slack_t)) / range(slack_t)
//!   p_{t+1,r}  proportional to  p_{t,r} * exp(-eta * loss_{t,r})
//! ```
//!
//! Rows doing badly (low slack) gain weight; rows already clearing their
//! threshold by a wide margin lose it.
//!
//! ## Why MW rather than the alternatives
//!
//! The inner problem is **linear in `p`**, so the row-player's bounded-loss
//! regret guarantee holds *unconditionally, whatever the α-player does* —
//! including when `slack_r(alpha)` is non-concave in α, which it is — on the
//! horizon-certified production constructor. That constructor refuses updates
//! past the stated horizon, and every update transactionally refuses rather
//! than projecting a zero or non-finite probability. The guarantee is in the per-round range-
//! normalised loss units above; it must not be misreported as a scale-free
//! bound on cumulative raw margins.
//!
//! There is a second, independent reason, and it is a fact about ny's adjoint
//! rather than about optimisation theory. The algebraically correct route for
//! a weighted objective is to scale separate **seed rows**
//! (`joint_alpha_grad.rs:101-174`); the ξ seed there is sign-only, so in exact
//! arithmetic a **strictly positive** per-row scale leaves every relaxation-
//! branch selection invariant and scales that row's branch-fixed adjoint
//! contribution. The f32 implementation remains subject to normal rounding.
//!
//! MW's weights are strictly positive by construction, which lets its fixed-
//! shape route carry every row without zero-scale ambiguity. Other sparse
//! objectives can still be exact if zero-weight rows are omitted rather than
//! combined before the fold; positivity is an invariant of this route, not a
//! claim that MW is the only representable row rule.
//!
//! **That makes strict positivity a correctness requirement of this module, not
//! a nicety.** `exp` can underflow to exactly 0.0 for sufficiently negative
//! exponents. The update is staged and returns a typed refusal without mutating
//! the player if that happens; it never projects the distribution to a nearby
//! floored algorithm.

#![allow(dead_code)]

use ny_core::{NyError, Result};

/// Multiplicative-weights row-player over a conjunctive specification.
#[derive(Debug, Clone)]
pub(crate) struct RowWeights {
    p: Vec<f64>,
    eta: f64,
    rounds: usize,
    /// `Some(T)` makes the derived learning rate's advertised horizon an
    /// enforced runtime contract. `None` is the explicit-rate research route.
    certified_horizon: Option<usize>,
}

impl RowWeights {
    /// Uniform start with an explicit learning rate.
    pub(crate) fn new(rows: usize, eta: f64) -> Result<Self> {
        if rows == 0 {
            return Err(NyError::InvalidSpec(
                "#row-weights: refusing an empty row set".to_string(),
            ));
        }
        if !eta.is_finite() || eta <= 0.0 {
            return Err(NyError::InvalidSpec(format!(
                "#row-weights: eta must be finite and positive, got {eta}"
            )));
        }
        Ok(Self {
            p: vec![1.0 / rows as f64; rows],
            eta,
            rounds: 0,
            certified_horizon: None,
        })
    }

    /// Uniform start with `eta` set from the MW regret bound rather than tuned.
    ///
    /// The standard choice `eta = sqrt(8 ln R / T)` minimises the regret bound
    /// `sqrt(T ln R / 2)` over a known horizon `T` for losses in `[0, 1]`.
    /// [`Self::update`] normalises each finite slack vector to exactly that
    /// interval before applying Hedge; using raw, arbitrarily scaled margins
    /// here would invalidate the bound and collapse the distribution.
    ///
    /// This is a derived constant, not a knob. Updates after `T` are refused,
    /// and any finite-precision underflow before then is a transactional typed
    /// error rather than a projection that would change the Hedge algorithm.
    pub(crate) fn with_horizon(rows: usize, horizon: usize) -> Result<Self> {
        if rows == 0 {
            return Err(NyError::InvalidSpec(
                "#row-weights: refusing an empty row set".to_string(),
            ));
        }
        if horizon == 0 {
            return Err(NyError::InvalidSpec(
                "#row-weights: refusing a zero update horizon".to_string(),
            ));
        }
        if rows == 1 {
            // ln(1) = 0 would give eta = 0. With a single row the row-player
            // has nothing to decide, so any positive eta is equivalent.
            let mut weights = Self::new(1, 1.0)?;
            weights.certified_horizon = Some(horizon);
            return Ok(weights);
        }
        let t = horizon as f64;
        let eta = (8.0 * (rows as f64).ln() / t).sqrt();
        let mut weights = Self::new(rows, eta)?;
        weights.certified_horizon = Some(horizon);
        Ok(weights)
    }

    pub(crate) fn weights(&self) -> &[f64] {
        &self.p
    }

    pub(crate) fn rows(&self) -> usize {
        self.p.len()
    }

    pub(crate) fn eta(&self) -> f64 {
        self.eta
    }

    pub(crate) fn rounds(&self) -> usize {
        self.rounds
    }

    /// Smallest weight currently held. Strictly positive by the module's
    /// invariant; exposed so a caller wiring the seed-scaling route can assert
    /// it before relying on the exactness argument.
    pub(crate) fn min_weight(&self) -> f64 {
        self.p.iter().copied().fold(f64::INFINITY, f64::min)
    }

    /// One MW round against the observed per-row slacks.
    ///
    /// `slacks[r]` is `lb_r - threshold_r` (negative = violated). Rows with the
    /// lowest slack gain weight.
    pub(crate) fn update(&mut self, slacks: &[f32]) -> Result<()> {
        if self
            .certified_horizon
            .is_some_and(|horizon| self.rounds >= horizon)
        {
            return Err(NyError::InvalidSpec(format!(
                "#row-weights: refusing update {} beyond certified horizon {}",
                self.rounds + 1,
                self.certified_horizon.unwrap_or_default(),
            )));
        }
        if slacks.len() != self.p.len() {
            return Err(NyError::ShapeMismatch {
                expected: vec![self.p.len()],
                got: vec![slacks.len()],
            });
        }
        // Map the observed payoff range to [0, 1]. Hedge's advertised regret
        // bound assumes bounded losses; raw verification margins have no fixed
        // scale and can differ by many orders of magnitude between models.
        // This normalisation is invariant to every positive affine rescaling,
        // keeps the worst row at loss 0 and the most comfortable at loss 1,
        // and prevents a large but otherwise equivalent C scale from turning
        // the first update into a point mass.
        let mut min_slack = f64::INFINITY;
        let mut max_slack = f64::NEG_INFINITY;
        for &s in slacks {
            let s = f64::from(s);
            if !s.is_finite() {
                return Err(NyError::InvalidSpec(format!(
                    "#row-weights: non-finite slack {s} — refusing to update weights \
                     from a diverged fold"
                )));
            }
            min_slack = min_slack.min(s);
            max_slack = max_slack.max(s);
        }
        let range = max_slack - min_slack;
        if !range.is_finite() || range < 0.0 {
            return Err(NyError::InvalidSpec(format!(
                "#row-weights: invalid finite slack range {range}"
            )));
        }

        // Stage the complete distribution so an unexpected floor violation on
        // the certified route cannot partially mutate the row player.
        let mut next = Vec::new();
        next.try_reserve_exact(self.p.len()).map_err(|_| {
            NyError::InvalidSpec("#row-weights: could not reserve update weights".to_string())
        })?;
        let mut total = 0.0f64;
        for (&weight, &s) in self.p.iter().zip(slacks) {
            let loss = if range == 0.0 {
                0.0
            } else {
                ((f64::from(s) - min_slack) / range).clamp(0.0, 1.0)
            };
            let updated = weight * (-self.eta * loss).exp();
            if !updated.is_finite() || updated <= 0.0 {
                return Err(NyError::InvalidSpec(
                    "#row-weights: Hedge update produced a zero or non-finite weight".to_string(),
                ));
            }
            total += updated;
            next.push(updated);
        }
        if !total.is_finite() || total <= 0.0 {
            return Err(NyError::InvalidSpec(format!(
                "#row-weights: weight mass collapsed to {total}"
            )));
        }
        for w in &mut next {
            *w /= total;
            if !w.is_finite() || *w <= 0.0 {
                return Err(NyError::InvalidSpec(
                    "#row-weights: Hedge normalisation produced a zero or non-finite weight"
                        .to_string(),
                ));
            }
        }
        // One final normalization absorbs the ordinary summation rounding;
        // unlike the former floor, this does not change the Hedge update.
        let total2: f64 = next.iter().sum();
        if !total2.is_finite() || total2 <= 0.0 {
            return Err(NyError::InvalidSpec(format!(
                "#row-weights: normalized weight mass collapsed to {total2}"
            )));
        }
        for w in &mut next {
            *w /= total2;
        }
        self.p = next;
        debug_assert!(
            self.p.iter().all(|w| *w > 0.0),
            "#row-weights: strict positivity is what makes the seed-scaling \
             route exact and must never be violated"
        );
        self.rounds += 1;
        Ok(())
    }

    /// `SUM_r p_r * slack_r` — the scalar the α-player is handed this round.
    pub(crate) fn weighted_slack(&self, slacks: &[f32]) -> Result<f64> {
        if slacks.len() != self.p.len() {
            return Err(NyError::ShapeMismatch {
                expected: vec![self.p.len()],
                got: vec![slacks.len()],
            });
        }
        Ok(self
            .p
            .iter()
            .zip(slacks)
            .map(|(w, &s)| w * f64::from(s))
            .sum())
    }

    /// Build the positively-scaled seed for the exact adjoint route: row `r` of
    /// `objectives` scaled by `p_r`. Flattened `rows x output_dim`, the layout
    /// the joint-α fold consumes.
    ///
    /// This is the whole point of the module — see the exactness argument in
    /// the module docs. Refuses if any weight is non-positive, because the
    /// argument does not hold then.
    pub(crate) fn scaled_seed(&self, objectives: &[Vec<f32>]) -> Result<Vec<f32>> {
        if objectives.len() != self.p.len() {
            return Err(NyError::ShapeMismatch {
                expected: vec![self.p.len()],
                got: vec![objectives.len()],
            });
        }
        let Some(od) = objectives.first().map(Vec::len) else {
            return Err(NyError::InvalidSpec(
                "#row-weights: empty objective set".to_string(),
            ));
        };
        if od == 0 {
            return Err(NyError::InvalidSpec(
                "#row-weights: empty objective row".to_string(),
            ));
        }
        let seed_len =
            self.p.len().checked_mul(od).ok_or_else(|| {
                NyError::InvalidSpec("#row-weights: seed size overflow".to_string())
            })?;
        let mut seed = Vec::new();
        seed.try_reserve_exact(seed_len).map_err(|_| {
            NyError::InvalidSpec(format!(
                "#row-weights: could not reserve {seed_len} seed coefficients"
            ))
        })?;
        for (w, obj) in self.p.iter().zip(objectives) {
            let wf = *w as f32;
            if !w.is_finite() || *w <= 0.0 || !wf.is_finite() || wf <= 0.0 {
                return Err(NyError::InvalidSpec(format!(
                    "#row-weights: weight {w} is not strictly positive — the \
                     seed-scaling route is only exact for positive scales"
                )));
            }
            if obj.len() != od {
                return Err(NyError::ShapeMismatch {
                    expected: vec![od],
                    got: vec![obj.len()],
                });
            }
            for &coefficient in obj {
                let scaled = coefficient * wf;
                if !coefficient.is_finite()
                    || !scaled.is_finite()
                    || (coefficient != 0.0
                        && (scaled == 0.0
                            || scaled.is_sign_negative() != coefficient.is_sign_negative()))
                {
                    return Err(NyError::InvalidSpec(
                        "#row-weights: weighted seed did not preserve a finite nonzero coefficient's sign"
                            .to_string(),
                    ));
                }
                seed.push(scaled);
            }
        }
        debug_assert_eq!(seed.len(), seed_len);
        Ok(seed)
    }
}

#[cfg(test)]
mod tests;
