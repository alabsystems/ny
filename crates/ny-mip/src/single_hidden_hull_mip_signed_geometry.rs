// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Original head-index mass partitions; old directions are proposals only.
use super::*;
#[derive(Debug, Clone)]
pub(super) struct Component {
    pub negative: Vec<usize>,
    pub positive: Vec<usize>,
    pub positive_units: Vec<u8>,
    pub g: Vec<f64>,
    pub alpha: Vec<f64>,
}
impl Component {
    /// One authoritative mass factor; never a rounded point half coefficient.
    pub(super) fn coefficient(&self, p: &HullMarginProblem, slot: usize) -> Option<certificate::I> {
        let j = *self.positive.get(slot)?;
        let c = *p.c.get(j)?;
        if c.partial_cmp(&0.) != Some(std::cmp::Ordering::Greater) {
            return None;
        }
        certificate::I::point(c)?.scale(self.weight(slot)?)
    }
    /// Advice only. The complete interval checker encloses underflow/rounding.
    pub(super) fn nominal_coefficient(&self, p: &HullMarginProblem, slot: usize) -> Option<f64> {
        let c = *p.c.get(*self.positive.get(slot)?)?;
        if c.partial_cmp(&0.) != Some(std::cmp::Ordering::Greater) {
            return None;
        }
        finite(c * self.weight(slot)?)
    }
    pub(super) fn weight(&self, slot: usize) -> Option<f64> {
        if self.positive_units.len() != self.positive.len() {
            return None;
        }
        match self.positive_units.get(slot)? {
            1 => Some(0.5),
            2 => Some(1.),
            _ => None,
        }
    }
    /// Resource cardinality only; weight/shape validation still owns admissibility.
    pub(super) fn half_units(&self) -> usize {
        self.positive_units
            .iter()
            .map(|unit| usize::from(*unit == 1))
            .sum()
    }
}
#[derive(Debug, Clone)]
pub(super) struct Plan {
    pub components: Vec<Component>,
    pub leftover: Vec<usize>,
}
impl Plan {
    pub(super) fn masks(&self) -> Option<usize> {
        self.components.iter().try_fold(0usize, |v, c| {
            if c.negative.is_empty() || c.negative.len() > 8 {
                return None;
            }
            v.checked_add(1usize.checked_shl(c.negative.len() as u32)?)
        })
    }
    pub(super) fn cost(&self, p: &HullMarginProblem) -> Option<usize> {
        let mut n =
            64usize.checked_mul(p.input_dim.checked_mul(p.hidden_dim)? + p.hidden_dim + 1)?;
        for c in &self.components {
            if c.negative.is_empty() || c.negative.len() > 8 {
                return None;
            }
            if c.positive_units.len() != c.positive.len()
                || c.positive_units.iter().any(|u| !matches!(u, 1 | 2))
            {
                return None;
            }
            n = n.checked_add(
                1usize
                    .checked_shl(c.negative.len() as u32)?
                    .checked_mul(p.input_dim + 1)?
                    .checked_mul(c.positive.len() + c.negative.len() + 3)?
                    .checked_mul(4)?,
            )?;
            n = n.checked_add(
                (1usize << c.negative.len())
                    .checked_mul(c.half_units())?
                    .checked_mul(4)?,
            )?;
        }
        Some(n)
    }
    pub(super) fn valid(&self, cuts: &HullRootMipCuts<'_>) -> Option<()> {
        let p = cuts.problem;
        if !capacity(p) || p.c.len() != p.hidden_dim {
            return None;
        }
        let mut seen = vec![false; p.hidden_dim];
        let mut units = vec![0u8; p.hidden_dim];
        if self.components.is_empty()
            || self.components.len() > p.hidden_dim
            || self.masks()? > MAX_MASKS
        {
            return None;
        }
        for c in &self.components {
            if c.negative.is_empty()
                || c.negative.len() > 8
                || c.positive.len() > MAX_POSITIVE
                || c.positive_units.len() != c.positive.len()
                || c.g.len() != p.input_dim
                || c.g.iter().any(|v| !v.is_finite())
                || c.alpha.len() != (1usize << c.negative.len()).checked_mul(c.positive.len())?
                || c.alpha
                    .iter()
                    .any(|v| !v.is_finite() || !(0. ..=1.).contains(v))
            {
                return None;
            }
            for &j in &c.negative {
                if j >= p.hidden_dim
                    || seen[j]
                    || p.c[j].partial_cmp(&0.) != Some(std::cmp::Ordering::Less)
                {
                    return None;
                }
                seen[j] = true;
            }
            for (slot, (&j, &u)) in c.positive.iter().zip(&c.positive_units).enumerate() {
                if j >= p.hidden_dim
                    || p.c[j].partial_cmp(&0.) != Some(std::cmp::Ordering::Greater)
                    || !matches!(u, 1 | 2)
                    || c.positive[..slot].contains(&j)
                {
                    return None;
                }
                units[j] = units[j].checked_add(u)?;
                if units[j] > 2 {
                    return None;
                }
            }
        }
        for &j in &self.leftover {
            if j >= p.hidden_dim
                || units[j] != 0
                || p.c[j].partial_cmp(&0.) != Some(std::cmp::Ordering::Greater)
            {
                return None;
            }
            units[j] = 2;
        }
        p.c.iter()
            .enumerate()
            .all(|(j, c)| {
                if *c < 0. {
                    seen[j] && units[j] == 0
                } else if *c > 0. {
                    !seen[j] && units[j] == 2
                } else {
                    *c == 0. && !seen[j] && units[j] == 0
                }
            })
            .then_some(())
    }
}
fn add_component(
    cuts: &HullRootMipCuts<'_>,
    heads: &[usize],
    old_g: &[f64],
    used: &mut [bool],
    out: &mut Vec<Component>,
) -> Option<()> {
    let p = cuts.problem;
    if heads.is_empty()
        || heads.len() > 8
        || old_g.len() != p.input_dim
        || old_g.iter().any(|v| !v.is_finite())
    {
        return None;
    }
    for &j in heads {
        if j >= p.hidden_dim || used[j] || p.c[j] >= 0. {
            return None;
        }
        used[j] = true;
    }
    out.push(Component {
        negative: heads.to_vec(),
        positive: Vec::new(),
        positive_units: Vec::new(),
        g: old_g.iter().map(|g| -g).collect(),
        alpha: Vec::new(),
    });
    Some(())
}
pub(super) fn plan(cuts: &HullRootMipCuts<'_>, b: &Budget) -> Option<Plan> {
    let p = cuts.problem;
    let positives = p.c.iter().filter(|c| **c > 0.).count();
    let negatives = p.c.iter().filter(|c| **c < 0.).count();
    let work = positives
        .checked_mul(negatives)?
        .checked_mul(p.input_dim + 1)?
        .checked_add(p.input_dim.checked_mul(p.hidden_dim)? + p.hidden_dim + 1)?
        .checked_mul(8)?;
    b.charge(work)?;
    let mut used = vec![false; p.hidden_dim];
    let mut components = Vec::new();
    if let Some(old) = &cuts.compact {
        for r in &old.groups {
            add_component(cuts, r.heads(), r.g(), &mut used, &mut components)?;
        }
        for r in &old.pairs {
            add_component(cuts, &r.heads, &r.g, &mut used, &mut components)?;
        }
    }
    for j in 0..p.hidden_dim {
        if p.c[j] < 0. && !used[j] {
            let a = -p.c[j];
            let g: Vec<_> = cuts.multipliers.g[j * p.input_dim..(j + 1) * p.input_dim]
                .iter()
                .map(|v| a * v)
                .collect();
            add_component(cuts, &[j], &g, &mut used, &mut components)?;
        }
    }
    let mut leftover = Vec::new();
    for j in 0..p.hidden_dim {
        if !live(b.search_end) {
            return None;
        }
        if p.c[j] <= 0. {
            continue;
        }
        let mut choice = None;
        for (index, c) in components.iter().enumerate() {
            if c.positive.len() >= MAX_POSITIVE {
                continue;
            }
            let mut score = 0f64;
            for &k in &c.negative {
                score = score.max(affinity(p, j, k));
            }
            if score > 0. && choice.is_none_or(|(_, old)| score > old) {
                choice = Some((index, score));
            }
        }
        if let Some((index, _)) = choice {
            components[index].positive.push(j);
            components[index].positive_units.push(2);
        } else {
            leftover.push(j);
        }
    }
    for c in &mut components {
        let initial: Vec<_> = c
            .positive
            .iter()
            .map(|&j| sanitize_alpha(cuts.multipliers.alpha[j]))
            .collect();
        c.alpha = initial.repeat(1usize << c.negative.len());
        for (slot, (&j, &alpha)) in c.positive.iter().zip(&initial).enumerate() {
            let weight = c.nominal_coefficient(p, slot)?;
            for i in 0..p.input_dim {
                c.g[i] = finite(c.g[i] + weight * alpha * p.w_row(j)[i])?;
            }
        }
    }
    let plan = Plan {
        components,
        leftover,
    };
    plan.valid(cuts)?;
    Some(plan)
}
pub(super) fn affinity(p: &HullMarginProblem, j: usize, k: usize) -> f64 {
    let mut dot = p.b1[j] * p.b1[k];
    let mut a = p.b1[j] * p.b1[j];
    let mut b = p.b1[k] * p.b1[k];
    for i in 0..p.input_dim {
        let width = p.upper[i] - p.lower[i];
        let x = p.w_row(j)[i] * width;
        let y = p.w_row(k)[i] * width;
        dot += x * y;
        a += x * x;
        b += y * y;
    }
    let score = dot / (a.sqrt() * b.sqrt());
    if score.is_finite() {
        score.max(0.)
    } else {
        0.
    }
}

#[cfg(test)]
mod half_unit_tests {
    use super::*;

    #[test]
    fn half_unit_resource_count_matches_every_sixteen_slot_mass_partition() {
        let mut component = Component {
            negative: Vec::new(),
            positive: (0..16).collect(),
            positive_units: vec![2; 16],
            g: Vec::new(),
            alpha: Vec::new(),
        };
        for mask in 0..=u16::MAX {
            for (slot, unit) in component.positive_units.iter_mut().enumerate() {
                *unit = if mask & (1u16 << slot) == 0 { 2 } else { 1 };
            }
            assert_eq!(
                u32::try_from(component.half_units()).unwrap(),
                mask.count_ones()
            );
        }
        // Counting cannot turn malformed unit tags into valid mass: the
        // unchanged weight and Plan validators still authenticate their schema.
        component.positive_units = vec![0, 1, 2, 255, 1];
        assert_eq!(component.half_units(), 2);
        component.positive_units.clear();
        assert_eq!(component.half_units(), 0);
    }
}
