// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! One nominal joint line inside the already paid coordinate envelope.
use super::*;

#[derive(Clone, Copy, Debug)]
pub(super) struct Schedule {
    pub enabled: bool,
    pub visits: usize,
    pub sweeps: usize,
    pub credit: usize,
    pub line_work: usize,
    pub actual_bound: usize,
    d: usize,
    k: usize,
    p: usize,
    masks: usize,
}
impl Schedule {
    pub(super) fn legacy(active: usize) -> Self {
        Self {
            enabled: false,
            visits: active,
            sweeps: 2,
            credit: 0,
            line_work: 0,
            actual_bound: 0,
            d: 0,
            k: 0,
            p: 0,
            masks: 0,
        }
    }
    /// Arithmetic only; its bounded preflight is paid before search admission.
    pub(super) fn new(d: usize, k: usize, p: usize, active: usize) -> Option<Self> {
        if !(1..=128).contains(&d) || !(1..=6).contains(&k) || !(1..=16).contains(&p) || active > p
        {
            return None;
        }
        let w = d.checked_add(1)?;
        let masks = 1usize.checked_shl(k as u32)?;
        let unit = search::coordinate_work(d)?;
        let per_visit = masks.checked_mul(unit)?;
        let credit = per_visit.checked_mul(active)?.checked_mul(2)?;
        let mut result = Self::legacy(active);
        result.credit = credit;
        result.actual_bound = credit;
        if p < 2 || active == 0 {
            return Some(result);
        }
        let setup = 16usize.checked_mul(p)?.checked_mul(w)?.checked_add(32)?;
        let per_mask = unit
            .checked_add(
                8usize
                    .checked_mul(w)?
                    .checked_mul(p.checked_mul(2)?.checked_add(k)?.checked_add(4)?)?,
            )?
            .checked_add(32usize.checked_mul(p.checked_add(1)?)?)?;
        let line_work = setup.checked_add(masks.checked_mul(per_mask)?)?;
        if line_work > credit {
            return Some(result);
        }
        let visits = active.min(credit.checked_sub(line_work)? / per_visit);
        Some(Self {
            enabled: true,
            visits,
            sweeps: 1,
            credit,
            line_work,
            actual_bound: line_work.checked_add(visits.checked_mul(per_visit)?)?,
            d,
            k,
            p,
            masks,
        })
    }
}

pub(super) struct Run {
    schedule: Schedule,
    source: usize,
    end: Instant,
    next_mask: usize,
    cancelled: bool,
}
pub(super) struct Outcome {
    pub completed: bool,
    pub improved: bool,
    pub proposals: usize,
}
impl Run {
    pub(super) fn new(schedule: Schedule, p: &HullMarginProblem, end: Instant) -> Self {
        Self {
            schedule,
            source: std::ptr::from_ref(p) as usize,
            end,
            next_mask: 0,
            cancelled: schedule.line_work > schedule.actual_bound
                || schedule.actual_bound > schedule.credit,
        }
    }
    /// The owning Ticket already claimed the one bounded search role. This
    /// local cursor additionally prevents repeating/skipping a mask's line.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn apply(
        &mut self,
        p: &HullMarginProblem,
        c: &Component,
        g: &[f64],
        mask: usize,
        alpha: &mut [f64],
        q: &mut Vec<f64>,
        constant: &mut f64,
        end: Instant,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<Outcome> {
        let s = self.schedule;
        let end = end.min(self.end);
        if !live(end) {
            self.cancelled = true;
            return None;
        }
        if !s.enabled
            || self.cancelled
            || self.source != std::ptr::from_ref(p) as usize
            || p.input_dim != s.d
            || c.negative.len() != s.k
            || c.positive.len() != s.p
            || c.positive_units.len() != s.p
            || alpha.len() != s.p
            || p.lower.len() != s.d
            || p.upper.len() != s.d
            || c.positive.iter().any(|j| *j >= p.hidden_dim)
            || c.negative.iter().any(|j| *j >= p.hidden_dim)
            || q.len() != s.d
            || g.len() != s.d
            || mask != self.next_mask
            || mask >= s.masks
        {
            return None;
        }
        self.next_mask += 1;
        let mut proposals = 0;
        let result = {
            let cancelled = &mut self.cancelled;
            let mut sticky = || {
                if *cancelled {
                    return false;
                }
                let ok = live(end) && poll();
                if !ok {
                    *cancelled = true;
                }
                ok
            };
            trial(
                p,
                c,
                g,
                mask,
                alpha,
                q,
                *constant,
                &mut proposals,
                end,
                &mut sticky,
            )
        };
        if self.cancelled || !live(end) {
            return None;
        }
        match result {
            None => Some(Outcome {
                completed: false,
                improved: false,
                proposals,
            }),
            Some(TrialChange::Unchanged) => Some(Outcome {
                completed: true,
                improved: false,
                proposals,
            }),
            Some(TrialChange::Improved((values, new_q, new_constant))) => {
                alpha.copy_from_slice(&values);
                *q = new_q;
                *constant = new_constant;
                Some(Outcome {
                    completed: true,
                    improved: true,
                    proposals,
                })
            }
        }
    }
}
fn score(p: &HullMarginProblem, q: &[f64], constant: f64) -> Option<f64> {
    if q.len() != p.input_dim {
        return None;
    }
    q.iter()
        .enumerate()
        .try_fold(finite(constant)?, |v, (i, x)| {
            finite(v + (x * p.lower[i]).min(x * p.upper[i]))
        })
}
type Trial = (Vec<f64>, Vec<f64>, f64);
enum TrialChange {
    Unchanged,
    Improved(Trial),
}
/// Both named states are completed lines. None is numeric refusal or
/// cancellation; the caller's sticky guard distinguishes the latter.
#[allow(clippy::too_many_arguments)]
fn trial(
    p: &HullMarginProblem,
    c: &Component,
    g: &[f64],
    mask: usize,
    alpha: &[f64],
    q: &[f64],
    constant: f64,
    proposals: &mut usize,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<TrialChange> {
    if !poll()
        || alpha
            .iter()
            .any(|v| !v.is_finite() || !(0. ..=1.).contains(v))
    {
        return None;
    }
    let mut direction = vec![0.; p.input_dim];
    let mut bias = 0.;
    for (slot, &head) in c.positive.iter().enumerate() {
        if !poll() {
            return None;
        }
        let weight = finite(c.nominal_coefficient(p, slot)? * (1. - alpha[slot]))?;
        bias = finite(bias + weight * p.b1[head])?;
        for i in 0..p.input_dim {
            direction[i] = finite(direction[i] + weight * p.w_row(head)[i])?;
        }
    }
    // d==0 is NOT a refusal: h>0 can be the entire coordinated improvement.
    let theta = if search::alpha::work(p.input_dim)? < search::legacy_work(p.input_dim)? {
        search::alpha::choose(
            p, q, &direction, constant, bias, 0., q, constant, proposals, end, poll,
        )
    } else {
        search::legacy_choose(p, q, &direction, constant, bias, 0., q, constant, proposals)
    }?;
    if !poll() {
        return None;
    }
    if theta == 0. {
        return Some(TrialChange::Unchanged);
    }
    let values = alpha
        .iter()
        .map(|old| {
            let value = if theta == 1. {
                1.
            } else {
                finite(old + theta * (1. - old))?
            };
            Some(finite(value)?.clamp(0., 1.))
        })
        .collect::<Option<Vec<_>>>()?;
    // Rebuild the ACTUAL stored alpha rather than trusting aggregate arithmetic.
    // Helper Events/test vectors have dropped before these buffers are created.
    let (new_q, new_constant) = search::mask_cache(p, c, g, mask, &values)?;
    let improved = score(p, &new_q, new_constant)? > score(p, q, constant)?;
    if !poll() {
        return None;
    }
    if improved {
        Some(TrialChange::Improved((values, new_q, new_constant)))
    } else {
        Some(TrialChange::Unchanged)
    }
}

#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_root_coupling_line_tests.rs"]
mod tests;
