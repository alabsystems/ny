// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Complete outward Gray cover; nominal theta selection has no authority.
use super::*;

#[derive(Default)]
pub(in super::super) struct Attempt {
    pub(in super::super) completed_masks: usize,
    pub(in super::super) source_replays: usize,
}

struct Gray {
    increments: Vec<Affine>,
    form: Affine,
    mask: usize,
}

impl Gray {
    fn new(
        source: &Source,
        negative: &Direction,
        eta: &[(usize, f64)],
        budget: &mut Budget,
    ) -> Option<Self> {
        let d = source.spec.inputs;
        if !valid_terms(source, &negative.heads, MAX_SUPPORT)
            || !valid_eta(source, eta)
            || source.first_error.len() != source.spec.heads
            || source.spec.first_bias.len() != source.spec.heads
            || source.lo.len() != d
            || source.hi.len() != d
            || source.spec.first.len() != d.checked_mul(source.spec.heads)?
        {
            return None;
        }
        budget.charge(d.checked_add(eta.len())?)?;
        let mut form = Affine::zero(d);
        for &(j, value) in eta {
            form.a[j] = I::p(value);
        }
        let mut increments = Vec::with_capacity(negative.heads.len());
        for &(head, tau) in &negative.heads {
            budget.charge(d.checked_mul(2)?.checked_add(4)?)?;
            let error = source.first_error[head];
            if !good(error) || error < 0.0 {
                return None;
            }
            let seed = I::p(-tau);
            let mut increment = Affine::zero(d);
            increment.b =
                seed.mul(I::p(f64::from(source.spec.first_bias[head])).add(I::p(error))?)?;
            for j in 0..d {
                if j & 31 == 0 {
                    budget.charge(0)?;
                }
                increment.a[j] = seed.mul(I::p(f64::from(
                    source.spec.first[j * source.spec.heads + head],
                )))?;
            }
            increments.push(increment);
        }
        Some(Self {
            increments,
            form,
            mask: 0,
        })
    }

    fn advance(&mut self, step: usize, budget: &mut Budget) -> Option<()> {
        let mask = step ^ (step >> 1);
        let changed = self.mask ^ mask;
        if changed.count_ones() != 1 {
            return None;
        }
        let bit = changed.trailing_zeros() as usize;
        let increment = self.increments.get(bit)?;
        budget.charge(self.form.a.len().checked_mul(2)?.checked_add(4)?)?;
        let add = mask & changed != 0;
        self.form.b = if add {
            self.form.b.add(increment.b)?
        } else {
            self.form.b.sub(increment.b)?
        };
        for (j, (a, &delta)) in self.form.a.iter_mut().zip(&increment.a).enumerate() {
            if j & 31 == 0 {
                budget.charge(0)?;
            }
            // Removing an interval increment retains accumulated width.
            *a = if add { a.add(delta)? } else { a.sub(delta)? };
        }
        self.mask = mask;
        Some(())
    }
}

/// Five bounded nominal candidates replace four *interval* replays here.
/// Any finite selected theta in[0,1] is valid; missing an optimum only weakens.
fn choose_theta(
    source: &Source,
    form: &Affine,
    positive: &Affine,
    ratio: f64,
    budget: &mut Budget,
) -> Option<f64> {
    let d = source.spec.inputs;
    budget.charge(d.checked_mul(12)?.checked_add(32)?)?;
    let mut coefficients = Vec::with_capacity(d);
    let (mut product, mut norm, mut strongest, mut kink) = (0.0, 0.0, 0.0, ratio);
    for j in 0..d {
        if j & 31 == 0 {
            budget.charge(0)?;
        }
        let (a, p) = (point(form.a[j])?, point(positive.a[j])?);
        let radius = 0.5 * source.hi[j] - 0.5 * source.lo[j];
        product += (a * radius) * (p * radius);
        norm += (p * radius) * (p * radius);
        let importance = (p * radius).abs();
        let trial = -a / p;
        if importance > strongest && good(trial) {
            strongest = importance;
            kink = trial.clamp(0.0, 1.0);
        }
        coefficients.push((a, p));
    }
    let fitted = if good(product) && good(norm) && norm > 0.0 {
        (-product / norm).clamp(0.0, 1.0)
    } else {
        ratio
    };
    let (mut best, mut selected) = (f64::NEG_INFINITY, ratio);
    for theta in [0.0, 1.0, ratio, fitted, kink] {
        if !good(theta) || !(0.0..=1.0).contains(&theta) {
            continue;
        }
        let mut value = point(form.b)? + theta * point(positive.b)?;
        for (j, &(a, p)) in coefficients.iter().enumerate() {
            let a = a + theta * p;
            value += a * if a < 0.0 { source.hi[j] } else { source.lo[j] };
        }
        if good(value) && value > best {
            best = value;
            selected = theta;
        }
    }
    good(best).then_some(selected)
}

/// Conservative credit bound for this exact path, including refused masks.
pub(in super::super) fn work_bound(
    source: &Source,
    negative: &Direction,
    positive: &PositiveTerms,
    eta: usize,
) -> Option<usize> {
    let d = source.spec.inputs;
    let n = negative.heads.len();
    let masks = 1usize.checked_shl(n as u32)?;
    let terms = positive.0.len().checked_add(n)?;
    source
        .spec
        .heads
        .checked_add(d.checked_mul(4)?)?
        .checked_add(terms.checked_mul(d.checked_mul(2)?.checked_add(12)?)?)?
        .checked_add(eta.checked_mul(4)?)?
        .checked_add(32)?
        .checked_add(masks.checked_mul(d.checked_mul(18)?.checked_add(40)?.checked_add(n)?)?)
}

#[cfg(test)]
pub(in super::super) fn certify<'a>(
    source: &'a Source,
    negative: Direction,
    positive: PositiveTerms,
    eta: Vec<(usize, f64)>,
    budget: &mut Budget,
) -> Option<CertifiedDirection<'a>> {
    let authority = PublicationAuthority {
        deadline: budget.deadline,
    };
    certify_protected(
        source,
        negative,
        positive,
        eta,
        budget,
        authority,
        &mut Attempt::default(),
    )
}

pub(in super::super) fn certify_protected<'a>(
    source: &'a Source,
    negative: Direction,
    positive: PositiveTerms,
    eta: Vec<(usize, f64)>,
    budget: &mut Budget,
    authority: PublicationAuthority,
    attempt: &mut Attempt,
) -> Option<CertifiedDirection<'a>> {
    let d = source.spec.inputs;
    budget.charge(source.spec.heads.checked_add(d)?)?;
    if !valid_geometry(source, &negative, &positive, &eta)
        || source.lo.len() != d
        || source.hi.len() != d
        || source.spec.first.len() != d.checked_mul(source.spec.heads)?
        || source.first_error.len() != source.spec.heads
        || source.spec.first_bias.len() != source.spec.heads
        || source.y.len() != source.spec.heads
    {
        return None;
    }
    let count = 1usize.checked_shl(negative.heads.len() as u32)?;
    if budget.masks.checked_add(count)? > MAX_MASKS {
        return None;
    }
    let positive_form = aggregate(source, &positive, budget)?;
    let mut gray = Gray::new(source, &negative, &eta, budget)?;
    budget.charge(negative.heads.len())?;
    let total: f64 = negative.heads.iter().map(|&(_, a)| a).sum();
    if !good(total) || total <= 0.0 {
        return None;
    }
    let mut lower = f64::INFINITY;
    for step in 0..count {
        if step != 0 {
            gray.advance(step, budget)?;
        }
        budget.charge(negative.heads.len().checked_add(1)?)?;
        let selected: f64 = negative
            .heads
            .iter()
            .enumerate()
            .filter(|&(bit, _)| gray.mask & (1 << bit) != 0)
            .map(|(_, &(_, a))| a)
            .sum();
        let ratio = (selected / total).clamp(0.0, 1.0);
        let theta = choose_theta(source, &gray.form, &positive_form, ratio, budget)?;
        attempt.source_replays += 1;
        lower = lower.min(replay_theta(
            source,
            &gray.form,
            &positive_form,
            theta,
            budget,
        )?);
        attempt.completed_masks += 1;
        budget.complete_mask()?;
    }
    // The proof stop remains budget.deadline. The certificate is BORN with
    // the preexisting root publication authority, not extended after expiry.
    finish_at(
        source,
        negative,
        positive,
        eta,
        lower,
        budget,
        authority.deadline,
    )
}

#[cfg(test)]
pub(in super::super) fn test_form(
    source: &Source,
    negative: &Direction,
    eta: &[(usize, f64)],
    step: usize,
    budget: &mut Budget,
) -> Option<Affine> {
    if step >= 1usize.checked_shl(negative.heads.len() as u32)? {
        return None;
    }
    let mut gray = Gray::new(source, negative, eta, budget)?;
    for at in 1..=step {
        gray.advance(at, budget)?;
    }
    Some(gray.form)
}

#[cfg(test)]
pub(in super::super) fn test_theta(
    source: &Source,
    negative: &Direction,
    positive: &PositiveTerms,
    eta: &[(usize, f64)],
    step: usize,
    theta: f64,
    budget: &mut Budget,
) -> Option<f64> {
    if !good(theta) || !(0.0..=1.0).contains(&theta) {
        return None;
    }
    if !valid_geometry(source, negative, positive, eta) {
        return None;
    }
    let form = test_form(source, negative, eta, step, budget)?;
    let p = aggregate(source, positive, budget)?;
    replay_theta(source, &form, &p, theta, budget)
}
