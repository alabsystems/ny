// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Paid collection and move-only selection of four distinct input slopes.
use super::*;
use certificate::I;

#[derive(Debug)]
pub(super) struct State {
    /// Each pair owns one complete-table buffer and its own stored input slope.
    pub(super) alternatives: [(Vec<f64>, Vec<f64>); 3],
    pub(super) primary_g: Vec<f64>,
    pub(super) target: Vec<f64>,
}

#[derive(Debug)]
pub(super) struct Selection {
    pub(super) values: Vec<f64>,
    /// None means the original Ray table and original component slope.
    pub(super) g: Option<Vec<f64>>,
    pub(super) candidate: Option<usize>,
}

// Keep the reviewed inline accounting explicit on the supported host lane.
#[cfg(target_pointer_width = "64")]
const _: () = {
    assert!(size_of::<State>() == 192);
    assert!(size_of::<Selection>() == 64);
};

fn masks(c: &Component) -> Option<usize> {
    (1..=8)
        .contains(&c.negative.len())
        .then(|| 1usize << c.negative.len())
}

fn vector(values: &[f64], expected: usize, poll: &mut dyn FnMut() -> bool) -> Option<()> {
    if values.len() != expected {
        return None;
    }
    for value in values {
        if !poll() || !value.is_finite() {
            return None;
        }
    }
    Some(())
}

/// A choice among formal table supports is advice. Final publication still
/// authenticates the actual selector columns and repairs its serialized row.
fn score(
    c: &Component,
    values: &[f64],
    g: &[f64],
    target: &[f64],
    activity: &[f64],
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<I> {
    let k = c.negative.len();
    if g.len() != target.len() || activity.len() != k {
        return None;
    }
    let weights = support_weights(values, &c.negative, Some(activity))?;
    let mut coefficients = [0.; 8];
    for i in 0..k {
        if !poll() {
            return None;
        }
        coefficients[i] = finite(-2. * weights[i])?;
    }
    let lower = repair(values, &coefficients[..k], end, poll)?;
    let mapped = [true; 8];
    let (coefficients, lower) = support::choose(
        values,
        coefficients,
        lower,
        &mapped[..k],
        Some(activity),
        end,
        poll,
    )?;
    let mut value = support::score(lower, &coefficients[..k], Some(activity))?;
    for (&g, &x) in g.iter().zip(target) {
        if !poll() {
            return None;
        }
        // Compare whole rows, including their actual stored input slope.
        // Values from tables with different g are never maxed together.
        value = value.add(I::point(g)?.scale(2.)?.scale(x)?)?;
    }
    Some(value)
}

impl State {
    /// All work, target storage, four slopes and three additional tables must
    /// already be funded. Optional refusal keeps the caller's original Joint
    /// buffer; cancellation is the outer None. No partial State is returned.
    pub(super) fn begin(
        p: &HullMarginProblem,
        c: &Component,
        target: Vec<f64>,
        scratch: &mut Vec<f64>,
        end: Instant,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<CellAttempt<Self>> {
        let mut cancelled = false;
        let mut sticky = || {
            if cancelled {
                return false;
            }
            let ok = live(end) && poll();
            cancelled |= !ok;
            ok
        };
        let result = (|| {
            if !sticky() {
                return None;
            }
            let m = masks(c)?;
            let d = p.input_dim;
            if !(1..=32).contains(&d)
                || target.len() != d
                || target.capacity() != d
                || !scratch.is_empty()
                || scratch.capacity() != m
            {
                return None;
            }
            let [primary_g, first, second, third] =
                input_phase::propose(p, c, &target, scratch, end, &mut sticky)?;
            if !scratch.is_empty()
                || scratch.capacity() != m
                || [&primary_g, &first, &second, &third]
                    .into_iter()
                    .any(|g| g.len() != d || g.capacity() != d)
            {
                return None;
            }
            if !sticky() {
                return None;
            }
            let alternatives = [first, second, third].map(|g| (Vec::with_capacity(m), g));
            if alternatives
                .iter()
                .any(|(values, _)| values.capacity() != m)
            {
                return None;
            }
            Some(Self {
                alternatives,
                primary_g,
                target,
            })
        })();
        sticky().then_some(match result {
            Some(value) => CellAttempt::Ready(value),
            None => CellAttempt::Refused,
        })
    }

    /// The caller first handles any legacy Joint/Ray mismatch. All four
    /// numeric values must succeed before any additional table grows. The
    /// primary value is returned for the caller's existing Joint allocation.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prices(
        &mut self,
        p: &HullMarginProblem,
        c: &Component,
        mask: usize,
        q: &[I],
        constant: I,
        advice: Option<(usize, f64)>,
        end: Instant,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<CellAttempt<f64>> {
        let mut cancelled = false;
        let mut sticky = || {
            if cancelled {
                return false;
            }
            let ok = live(end) && poll();
            cancelled |= !ok;
            ok
        };
        let result = (|| {
            if !sticky() {
                return None;
            }
            let m = masks(c)?;
            let d = p.input_dim;
            if !(1..=32).contains(&d)
                || mask >= m
                || self.primary_g.len() != d
                || self.primary_g.capacity() != d
                || self.target.len() != d
                || self.target.capacity() != d
                || self.alternatives.iter().any(|(values, g)| {
                    values.len() != mask
                        || values.capacity() != m
                        || g.len() != d
                        || g.capacity() != d
                })
            {
                return None;
            }
            let primary = input_phase::price(
                p,
                c,
                mask,
                q,
                constant,
                &self.primary_g,
                advice,
                end,
                &mut sticky,
            )?;
            let mut checked = [0.; 3];
            for (value, (_, g)) in checked.iter_mut().zip(&self.alternatives) {
                *value = input_phase::price(p, c, mask, q, constant, g, advice, end, &mut sticky)?;
            }
            if !sticky() {
                return None;
            }
            for ((values, _), value) in self.alternatives.iter_mut().zip(checked) {
                // len==mask<M and capacity==M were checked before evaluation.
                values.push(value);
            }
            Some(primary)
        })();
        sticky().then_some(match result {
            Some(value) => CellAttempt::Ready(value),
            None => CellAttempt::Refused,
        })
    }

    /// Evaluate all five complete supports under separately prepaid work.
    /// Optional shape/numeric refusal restores the valid Ray table into the
    /// existing primary allocation. Cancellation returns None. A malformed
    /// baseline or insufficient primary capacity also returns None because
    /// there is then no sound allocation-free fallback.
    pub(super) fn finish(
        self,
        c: &Component,
        mut primary: Vec<f64>,
        ray: &[f64],
        activity: &[f64],
        end: Instant,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<Selection> {
        let mut cancelled = false;
        let mut sticky = || {
            if cancelled {
                return false;
            }
            let ok = live(end) && poll();
            cancelled |= !ok;
            ok
        };
        if !sticky() {
            return None;
        }
        let m = masks(c)?;
        if primary.capacity() != m {
            return None;
        }
        // This is the sole immutable baseline used by every optional fallback.
        vector(ray, m, &mut sticky)?;
        let choice = (|| {
            let d = c.g.len();
            if !(1..=32).contains(&d)
                || self.target.capacity() != d
                || self.primary_g.capacity() != d
                || self
                    .alternatives
                    .iter()
                    .any(|(values, g)| values.capacity() != m || g.capacity() != d)
            {
                return None;
            }
            vector(&c.g, d, &mut sticky)?;
            vector(&self.target, d, &mut sticky)?;
            vector(activity, c.negative.len(), &mut sticky)?;
            if activity.iter().any(|value| !(0. ..=1.).contains(value)) {
                return None;
            }
            vector(&primary, m, &mut sticky)?;
            vector(&self.primary_g, d, &mut sticky)?;
            for (values, g) in &self.alternatives {
                vector(values, m, &mut sticky)?;
                vector(g, d, &mut sticky)?;
            }
            let mut best = score(c, ray, &c.g, &self.target, activity, end, &mut sticky)?;
            let mut choice = None;
            let proposed = score(
                c,
                &primary,
                &self.primary_g,
                &self.target,
                activity,
                end,
                &mut sticky,
            )?;
            if proposed.lo > best.hi {
                best = proposed;
                choice = Some(0);
            }
            for (index, (values, g)) in self.alternatives.iter().enumerate() {
                let proposed = score(c, values, g, &self.target, activity, end, &mut sticky)?;
                if proposed.lo > best.hi {
                    best = proposed;
                    choice = Some(index + 1);
                }
            }
            Some(choice)
        })();
        if !sticky() {
            return None;
        }
        let Self {
            mut alternatives,
            primary_g,
            target,
        } = self;
        drop(target);
        let selection = match choice.flatten() {
            None => {
                drop(alternatives);
                drop(primary_g);
                primary.clear();
                // capacity==ray.len()==M: copying cannot allocate or grow.
                primary.extend_from_slice(ray);
                Selection {
                    values: primary,
                    g: None,
                    candidate: None,
                }
            }
            Some(0) => {
                drop(alternatives);
                Selection {
                    values: primary,
                    g: Some(primary_g),
                    candidate: Some(0),
                }
            }
            Some(index) => {
                let (values, g) = std::mem::take(alternatives.get_mut(index - 1)?);
                drop(alternatives);
                drop(primary);
                drop(primary_g);
                Selection {
                    values,
                    g: Some(g),
                    candidate: Some(index),
                }
            }
        };
        sticky().then_some(selection)
    }
}
