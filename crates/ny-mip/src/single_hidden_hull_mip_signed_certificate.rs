// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Full original signed mass, all masks, every box coordinate; no LP authority.
use super::*;
#[derive(Clone, Copy, Debug)]
pub(super) struct I {
    pub lo: f64,
    pub hi: f64,
}
impl I {
    pub(super) fn point(x: f64) -> Option<Self> {
        finite(x).map(|x| Self { lo: x, hi: x })
    }
    pub(super) fn zero() -> Self {
        Self { lo: 0., hi: 0. }
    }
    pub(super) fn add(self, b: Self) -> Option<Self> {
        if b.lo == 0. && b.hi == 0. {
            return Some(self);
        }
        if self.lo == 0. && self.hi == 0. {
            return Some(b);
        }
        Some(Self {
            lo: finite(add_down(self.lo, b.lo))?,
            hi: finite(add_up(self.hi, b.hi))?,
        })
    }
    pub(super) fn sub(self, b: Self) -> Option<Self> {
        self.add(Self {
            lo: -b.hi,
            hi: -b.lo,
        })
    }
    pub(super) fn scale(self, v: f64) -> Option<Self> {
        finite(v)?;
        if v == 0. || (self.lo == 0. && self.hi == 0.) {
            return Some(Self::zero());
        }
        if v == 1. {
            return Some(self);
        }
        if v == -1. {
            return Some(Self {
                lo: -self.hi,
                hi: -self.lo,
            });
        }
        Some(Self {
            lo: finite(mul_down(self.lo, v).min(mul_down(self.hi, v)))?,
            hi: finite(mul_up(self.lo, v).max(mul_up(self.hi, v)))?,
        })
    }
    pub(super) fn minimum(self, l: f64, u: f64) -> Option<f64> {
        finite(
            mul_down(self.lo, l)
                .min(mul_down(self.lo, u))
                .min(mul_down(self.hi, l))
                .min(mul_down(self.hi, u)),
        )
    }
}
#[derive(Debug)]
pub(super) struct Checked {
    pub g: Vec<f64>,
    pub constant: f64,
    pub root_lower: f64,
    pub tables: Option<Vec<Vec<f64>>>,
    pub component_minima: Option<Vec<f64>>,
    pub conditional: Option<phase::Conditional>,
    pub cell: Option<phase::cube::cell::Table>,
}
pub(super) fn mask_lower(
    cuts: &HullRootMipCuts<'_>,
    c: &Component,
    mask: usize,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<f64> {
    mask_lower_collect(cuts, c, mask, end, poll, 0, None, None)
}
#[allow(clippy::too_many_arguments)]
fn mask_lower_collect(
    cuts: &HullRootMipCuts<'_>,
    c: &Component,
    mask: usize,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
    index: usize,
    cell: Option<&mut phase::cube::cell::Collector>,
    corners: Option<&mut corners::Frame>,
) -> Option<f64> {
    let p = cuts.problem;
    let mut coeff: Vec<_> = c.g.iter().map(|v| I::point(-v)).collect::<Option<_>>()?;
    let mut constant = I::zero();
    let alpha = c
        .alpha
        .get(mask * c.positive.len()..(mask + 1) * c.positive.len())?;
    for (slot, (&j, &value)) in c.positive.iter().zip(alpha).enumerate() {
        if !live(end) || !poll() {
            return None;
        }
        // The point half product can underflow. Enclose the exact mass first.
        let k = c.coefficient(p, slot)?.scale(value)?;
        constant = constant.add(k.scale(p.b1[j])?)?;
        for i in 0..p.input_dim {
            coeff[i] = coeff[i].add(k.scale(p.w_row(j)[i])?)?;
        }
    }
    for (bit, &j) in c.negative.iter().enumerate() {
        if mask & (1 << bit) == 0 {
            continue;
        }
        if !live(end) || !poll() {
            return None;
        }
        let k = I::point(p.c[j])?;
        constant = constant.add(k.scale(p.b1[j])?)?;
        for i in 0..p.input_dim {
            coeff[i] = coeff[i].add(k.scale(p.w_row(j)[i])?)?;
        }
    }
    let mut lower = constant.lo;
    for i in 0..p.input_dim {
        lower = finite(add_down(lower, coeff[i].minimum(p.lower[i], p.upper[i])?))?;
    }
    let mut corners = corners;
    if let Some(corners) = corners.as_deref_mut() {
        corners.push(p, c, index, mask, &coeff, constant, lower, end, poll)?;
    }
    if let Some(cell) = cell {
        let (captured, advice) =
            cell.apply_with_advice(cuts, c, index, mask, &coeff, constant, lower, end, poll)?;
        if captured {
            if let Some(corners) = corners {
                corners.capture_ray(index, mask, advice);
            }
        }
    }
    (live(end) && poll()).then_some(lower)
}
/// Caller precharges a whole pass. Completion counters follow the last poll,
/// never merely the charged number of masks.
pub(super) fn check(
    cuts: &HullRootMipCuts<'_>,
    plan: &Plan,
    b: &Budget,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<Checked> {
    check_inner(cuts, plan, b, end, poll, false, false, None, None)
}
/// Only a separately paid first-publication collector may request O(C) minima.
pub(super) fn check_components(
    cuts: &HullRootMipCuts<'_>,
    plan: &Plan,
    b: &Budget,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
) -> Option<Checked> {
    check_inner(cuts, plan, b, end, poll, false, true, None, None)
}

/// The opaque collector is minted only by the already-paid mass final permit.
pub(super) fn check_portfolio(
    cuts: &HullRootMipCuts<'_>,
    plan: &Plan,
    b: &Budget,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
    collector: &mut portfolio::conditional::Collector,
) -> Option<Checked> {
    check_inner(cuts, plan, b, end, poll, false, true, Some(collector), None)
}

/// The optional collector is owned by this same authenticated replay.
#[allow(clippy::too_many_arguments)]
pub(super) fn check_replay(
    cuts: &HullRootMipCuts<'_>,
    plan: &Plan,
    b: &Budget,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
    collect: bool,
    components: bool,
    corners: Option<&mut corners::Frame>,
) -> Option<Checked> {
    check_inner(cuts, plan, b, end, poll, collect, components, None, corners)
}

#[allow(clippy::too_many_arguments)]
fn check_inner(
    cuts: &HullRootMipCuts<'_>,
    plan: &Plan,
    b: &Budget,
    end: Instant,
    poll: &mut dyn FnMut() -> bool,
    collect: bool,
    components: bool,
    mut losing: Option<&mut portfolio::conditional::Collector>,
    mut corners: Option<&mut corners::Frame>,
) -> Option<Checked> {
    require_hull_environment().ok()?;
    shape(cuts)?;
    plan.valid(cuts)?;
    let p = cuts.problem;
    let mut g = vec![0.; p.input_dim];
    let mut exact = vec![I::zero(); p.input_dim];
    let mut constant = 0.;
    let mut tables = collect.then(|| Vec::with_capacity(plan.components.len()));
    let mut component_minima = components.then(|| Vec::with_capacity(plan.components.len()));
    let phase_enabled = components && phase::collecting_plan(b, plan);
    let mut conditional = phase_enabled.then(|| Vec::with_capacity(plan.components.len()));
    let mut cell = if collect && components {
        phase::cube::cell::begin(cuts, plan, b, end, poll)?.into_option()
    } else {
        None
    };
    let losing_table = !collect && losing.as_ref().is_some_and(|c| c.wants_table());
    for (index, c) in plan.components.iter().enumerate() {
        let mut pair =
            (phase_enabled || losing.is_some()).then(|| vec![[f64::INFINITY; 2]; c.negative.len()]);
        let mut coverage = [[0usize; 2]; 8];
        let mut values =
            (collect || losing_table).then(|| Vec::with_capacity(1usize << c.negative.len()));
        let mut lower = f64::INFINITY;
        for mask in 0..1usize << c.negative.len() {
            let candidate = mask_lower_collect(
                cuts,
                c,
                mask,
                end,
                poll,
                index,
                cell.as_mut(),
                corners.as_deref_mut(),
            )?;
            lower = lower.min(candidate);
            if let Some(pair) = &mut pair {
                for (bit, minima) in pair.iter_mut().enumerate() {
                    let side = (mask >> bit) & 1;
                    minima[side] = minima[side].min(candidate);
                    coverage[bit][side] = coverage[bit][side].checked_add(1)?;
                }
            }
            if let Some(values) = &mut values {
                values.push(candidate);
            }
            b.state.lock().ok()?.checked_masks += 1;
        }
        if let Some(all) = &mut conditional {
            let count = 1usize.checked_shl(c.negative.len() as u32)? / 2;
            if coverage[..c.negative.len()]
                .iter()
                .any(|v| *v != [count, count])
            {
                conditional = None; // Optional incomplete metadata is no authority.
            } else {
                all.push(pair.take()?);
            }
        }
        if let Some(collector) = losing.as_mut() {
            collector.push(
                c,
                pair,
                &coverage,
                losing_table.then(|| values.take()).flatten(),
            );
        }
        if let Some(tables) = &mut tables {
            tables.push(values?);
        }
        if let Some(minima) = &mut component_minima {
            minima.push(finite(lower)?);
        }
        constant = finite(add_down(constant, finite(lower)?))?;
        for i in 0..p.input_dim {
            g[i] = finite(g[i] + c.g[i])?;
            exact[i] = exact[i].add(I::point(c.g[i])?)?;
        }
    }
    for &j in &plan.leftover {
        if !live(end) || !poll() {
            return None;
        }
        let k = I::point(p.c[j])?.scale(sanitize_alpha(cuts.multipliers.alpha[j]))?;
        constant = finite(add_down(constant, k.scale(p.b1[j])?.lo))?;
        for i in 0..p.input_dim {
            let v = k.scale(p.w_row(j)[i])?;
            exact[i] = exact[i].add(v)?;
            g[i] =
                finite(g[i] + p.c[j] * sanitize_alpha(cuts.multipliers.alpha[j]) * p.w_row(j)[i])?;
        }
    }
    // The exported point sum is not the exact sum. Price its ENTIRE residual.
    for i in 0..p.input_dim {
        if !live(end) || !poll() {
            return None;
        }
        let residue = exact[i].sub(I::point(g[i])?)?;
        constant = finite(add_down(constant, residue.minimum(p.lower[i], p.upper[i])?))?;
    }
    let mut root_lower = finite(add_down(p.d, constant))?;
    for i in 0..p.input_dim {
        root_lower = finite(add_down(
            root_lower,
            I::point(g[i])?.minimum(p.lower[i], p.upper[i])?,
        ))?;
    }
    let conditional = match conditional {
        Some(values) => phase::Conditional::finish(cuts, values),
        None => None,
    };
    let mut cell_cancelled = false;
    let cell = cell.and_then(|collector| {
        collector.finish(plan, b, end, &mut || {
            if cell_cancelled {
                return false;
            }
            let ok = poll();
            cell_cancelled |= !ok;
            ok
        })
    });
    if cell_cancelled {
        return None;
    }
    require_hull_environment().ok()?;
    (live(end) && poll()).then_some(Checked {
        g,
        constant,
        root_lower,
        tables,
        component_minima,
        conditional,
        cell,
    })
}
