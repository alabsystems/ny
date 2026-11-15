// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Directed cell prices for four bounded changes to the input coefficient.
//!
//! `propose` is advice
//! only. `price` must receive the original affine intervals from the same
//! authenticated Source callback and component. Its caller owns all funding,
//! table completeness, ownership, deadline, old-row retention and seal checks.
use super::*;
use certificate::I;

fn shape(p: &HullMarginProblem, c: &Component) -> Option<usize> {
    let d = p.input_dim;
    let k = c.negative.len();
    if !(1..=32).contains(&d)
        || !(1..=8).contains(&k)
        || c.positive.len() > 16
        || c.positive_units.len() != c.positive.len()
        || c.g.len() != d
        || p.lower.len() != d
        || p.upper.len() != d
        || p.c.len() != p.hidden_dim
        || p.b1.len() != p.hidden_dim
        || p.w1.len() != p.hidden_dim.checked_mul(d)?
    {
        return None;
    }
    let m = 1usize << k;
    (c.alpha.len() == m.checked_mul(c.positive.len())?).then_some(m)
}

fn nominal_add(
    p: &HullMarginProblem,
    head: usize,
    coefficient: f64,
    q: &mut [I],
    constant: &mut f64,
    poll: &mut dyn FnMut() -> bool,
) -> Option<()> {
    if head >= p.hidden_dim || q.len() != p.input_dim || !poll() {
        return None;
    }
    *constant = finite(*constant + finite(coefficient * p.b1[head])?)?;
    for (i, value) in q.iter_mut().enumerate() {
        if !poll() {
            return None;
        }
        // Point intervals here only adapt nominal advice to Ray's API.
        // This arithmetic has no certificate authority.
        *value = I::point(finite(value.lo + finite(coefficient * p.w_row(head)[i])?)?)?;
    }
    Some(())
}

fn nominal_affine(
    p: &HullMarginProblem,
    c: &Component,
    mask: usize,
    q: &mut [I],
    poll: &mut dyn FnMut() -> bool,
) -> Option<f64> {
    if q.len() != p.input_dim {
        return None;
    }
    for (value, &g) in q.iter_mut().zip(&c.g) {
        if !poll() {
            return None;
        }
        *value = I::point(-g)?;
    }
    let mut constant = 0.;
    let alpha = c
        .alpha
        .get(mask * c.positive.len()..(mask + 1) * c.positive.len())?;
    for (slot, (&head, &alpha)) in c.positive.iter().zip(alpha).enumerate() {
        if !alpha.is_finite() || !(0. ..=1.).contains(&alpha) {
            return None;
        }
        let coefficient = finite(c.nominal_coefficient(p, slot)? * alpha)?;
        nominal_add(p, head, coefficient, q, &mut constant, poll)?;
    }
    for (bit, &head) in c.negative.iter().enumerate() {
        let coefficient = *p.c.get(head)?;
        if !coefficient.is_finite() || coefficient >= 0. {
            return None;
        }
        if mask & (1 << bit) != 0 {
            nominal_add(p, head, coefficient, q, &mut constant, poll)?;
        }
    }
    Some(constant)
}

fn nominal_price(
    p: &HullMarginProblem,
    q: &[I],
    mut constant: f64,
    poll: &mut dyn FnMut() -> bool,
) -> Option<f64> {
    for (i, value) in q.iter().enumerate() {
        if !poll() {
            return None;
        }
        let endpoint = if value.lo >= 0. {
            p.lower[i]
        } else {
            p.upper[i]
        };
        constant = finite(constant + finite(value.lo * endpoint)?)?;
    }
    Some(constant)
}

fn legal_advice(
    p: &HullMarginProblem,
    c: &Component,
    mask: usize,
    (head, signed_beta): (usize, f64),
) -> Option<()> {
    let bit = c.negative.iter().position(|&j| j == head)?;
    let coefficient = *p.c.get(head)?;
    if head >= p.hidden_dim
        || !coefficient.is_finite()
        || coefficient >= 0.
        || !signed_beta.is_finite()
    {
        return None;
    }
    let legal = if mask & (1 << bit) == 0 {
        signed_beta >= 0.
    } else {
        signed_beta <= 0.
    };
    legal.then_some(())
}

fn seed_residual(
    p: &HullMarginProblem,
    c: &Component,
    mask: usize,
    q: &mut [I],
    poll: &mut dyn FnMut() -> bool,
) -> Option<()> {
    let mut constant = nominal_affine(p, c, mask, q, poll)?;
    if let Some(advice @ (head, signed_beta)) = ray::proposal(p, c, mask, q, poll) {
        legal_advice(p, c, mask, advice)?;
        nominal_add(p, head, signed_beta, q, &mut constant, poll)?;
    }
    // A cancelled Ray proposal must not become a successful base-only seed.
    poll().then_some(())
}

/// Return four finite stored g vectors in a fixed order:
/// top axis full/half, all common-sign faces full/half.
///
/// `scratch` is the already paid, empty Joint raw-table allocation. No reserve
/// or reallocation occurs; its exact M capacity is required, and it is empty
/// again on every exit after the entry guards. The Source and every allocation
/// remain the caller's responsibility. This routine never reads a solver.
pub(super) fn propose(
    p: &HullMarginProblem,
    c: &Component,
    target: &[f64],
    scratch: &mut Vec<f64>,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<[Vec<f64>; 4]> {
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
    let masks = shape(p, c)?;
    let d = p.input_dim;
    if target.len() != d || !scratch.is_empty() || scratch.capacity() != masks {
        return None;
    }
    let result = (|| {
        for i in 0..d {
            if !sticky()
                || !target[i].is_finite()
                || !p.lower[i].is_finite()
                || !p.upper[i].is_finite()
                || p.lower[i] > p.upper[i]
                || target[i] < p.lower[i]
                || target[i] > p.upper[i]
            {
                return None;
            }
        }
        let mut q = [I::zero(); 32];
        for mask in 0..masks {
            if !sticky() {
                return None;
            }
            let constant = nominal_affine(p, c, mask, &mut q[..d], &mut sticky)?;
            scratch.push(nominal_price(p, &q[..d], constant, &mut sticky)?);
        }
        let mut selected = None;
        for mask in 0..masks / 2 {
            if !sticky() {
                return None;
            }
            let score = finite(scratch[mask] + scratch[masks - 1 - mask])?;
            // Strict comparison leaves ties at the first original mask.
            if selected.is_none_or(|(_, old)| score < old) {
                selected = Some((mask, score));
            }
        }
        let first = selected?.0;
        seed_residual(p, c, first, &mut q[..d], &mut sticky)?;
        let mut common = [0.; 32];
        for (out, value) in common[..d].iter_mut().zip(&q[..d]) {
            if !sticky() {
                return None;
            }
            *out = value.lo;
        }
        seed_residual(p, c, masks - 1 - first, &mut q[..d], &mut sticky)?;
        let mut strict_common = false;
        for i in 0..d {
            if !sticky() {
                return None;
            }
            strict_common |= (common[i] > 0. && q[i].lo > 0.) || (common[i] < 0. && q[i].lo < 0.);
        }
        let mut top = None;
        for i in 0..d {
            if !sticky() {
                return None;
            }
            let (left, right) = (common[i], q[i].lo);
            let step = if left > 0. && right > 0. {
                left.min(right)
            } else if left < 0. && right < 0. {
                left.max(right)
            } else if !strict_common && (left == 0.) != (right == 0.) {
                // A zero affine coefficient is a flat box face, not evidence
                // that this axis is exhausted. Only when EVERY strict-common
                // direction is absent, also propose the other seed's finite
                // breakpoint. This avoids changing the actual nonzero seed
                // policy and does not assume a unique subgradient at zero.
                if left == 0. {
                    right
                } else {
                    left
                }
            } else {
                0.
            };
            common[i] = step;
            if step != 0. {
                let endpoint = if step > 0. { p.lower[i] } else { p.upper[i] };
                let score = finite(finite(2. * step)? * finite(target[i] - endpoint)?)?;
                if top.is_none_or(|(_, old)| score > old) {
                    top = Some((i, score));
                }
            }
        }
        // If both residuals vanish or every nonzero pair has opposite signs,
        // this bounded heuristic still has no direction. The zero-face
        // fallback above is not a claim of general dual or input optimality.
        let axis = top?.0;
        let mut out: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::with_capacity(d));
        if out.iter().any(|g| g.capacity() != d) {
            return None;
        }
        let mut changed = false;
        for i in 0..d {
            if !sticky() {
                return None;
            }
            let step = common[i];
            let half = finite(0.5 * step)?;
            for (candidate, increment) in out.iter_mut().zip([
                if i == axis { step } else { 0. },
                if i == axis { half } else { 0. },
                step,
                half,
            ]) {
                let g = if increment == 0. {
                    c.g[i]
                } else {
                    finite(c.g[i] + increment)?
                };
                changed |= g != c.g[i];
                candidate.push(g);
            }
        }
        changed.then_some(out)
    })();
    scratch.clear();
    if sticky() {
        result
    } else {
        None
    }
}

fn valid_interval(value: I) -> bool {
    value.lo.is_finite() && value.hi.is_finite() && value.lo <= value.hi
}

fn box_price(
    p: &HullMarginProblem,
    q: &[I],
    constant: I,
    poll: &mut dyn FnMut() -> bool,
) -> Option<f64> {
    let mut lower = constant.lo;
    for (i, value) in q.iter().enumerate() {
        if !poll() {
            return None;
        }
        lower = finite(add_down(lower, value.minimum(p.lower[i], p.upper[i])?))?;
    }
    Some(lower)
}

/// Check only values with the SAME stored new_g. An old Cell/Ray/Joint table
/// at c.g is never an argument to the max below. The caller must not combine
/// entries from these different input coefficients into one phase table.
#[allow(clippy::too_many_arguments)]
pub(super) fn price(
    p: &HullMarginProblem,
    c: &Component,
    mask: usize,
    original_q: &[I],
    constant: I,
    new_g: &[f64],
    old_ray_advice: Option<(usize, f64)>,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<f64> {
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
    let masks = shape(p, c)?;
    let d = p.input_dim;
    if mask >= masks || original_q.len() != d || new_g.len() != d || !valid_interval(constant) {
        return None;
    }
    let mut q = [I::zero(); 32];
    for i in 0..d {
        if !sticky()
            || !valid_interval(original_q[i])
            || !p.lower[i].is_finite()
            || !p.upper[i].is_finite()
            || p.lower[i] > p.upper[i]
        {
            return None;
        }
        let old = I::point(c.g[i])?;
        let new = I::point(new_g[i])?;
        // new_g is the exact stored row coefficient. Its finite construction
        // may have rounded; enclosure of old-new, not the nominal increment,
        // is therefore the authority. Exact equality avoids needless width.
        q[i] = if c.g[i] == new_g[i] {
            original_q[i]
        } else {
            original_q[i].add(old.sub(new)?)?
        };
    }
    let mut best = box_price(p, &q[..d], constant, &mut sticky)?;
    if let Some(advice @ (head, signed_beta)) = old_ray_advice {
        if !sticky() {
            return None;
        }
        legal_advice(p, c, mask, advice)?;
        let mut corrected_constant = constant;
        numeric::add_head(
            p,
            head,
            I::point(signed_beta)?,
            &mut q[..d],
            &mut corrected_constant,
        )?;
        if !sticky() {
            return None;
        }
        best = best.max(box_price(p, &q[..d], corrected_constant, &mut sticky)?);
    }
    sticky().then_some(best)
}
