// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
use super::*;

/// Construction consumes only private completed certificates. Later publication
/// independently replays every selected mask using the SAME stored ledger.
pub(in crate::single_hidden_hull::mip_cuts) fn build_group_aggregate(
    cuts: &HullRootMipCuts<'_>,
    groups: Vec<group::GroupMajorant>,
    budget: &group::Budget,
    poll: &mut dyn FnMut() -> bool,
) -> Option<Aggregate> {
    build_groups(cuts, groups, budget, poll, false)
}
pub(super) fn build_groups(
    cuts: &HullRootMipCuts<'_>,
    mut groups: Vec<group::GroupMajorant>,
    budget: &group::Budget,
    poll: &mut dyn FnMut() -> bool,
    replay: bool,
) -> Option<Aggregate> {
    let p = cuts.problem;
    let end = (if replay {
        budget.authority
    } else {
        budget.search_end
    })
    .min(cuts.deadline);
    if groups.is_empty() || groups.len() > group::MAX_ATTEMPTS {
        return None;
    }
    budget.charge(
        16usize.checked_mul((p.hidden_dim + 1).checked_mul(p.input_dim + 1)?)?,
        0,
        replay,
    )?;
    let _scratch = budget.memory(64 * (p.input_dim + p.hidden_dim + 1))?;
    let memory = budget.memory(
        (p.hidden_dim / 2)
            .checked_mul(p.input_dim + 12)?
            .checked_mul(8)?
            .checked_add((p.input_dim + groups.len() + 1) * 32)?,
    )?;
    checked_planes(cuts, end, poll)?;
    groups.sort_by(|a, b| a.heads().cmp(b.heads()));
    let mut used = vec![false; p.hidden_dim];
    let mut sum = Sum::new(p.input_dim);
    let mut masks = 0usize;
    for row in &groups {
        if !row.owned_by(budget) {
            return None;
        }
        budget.charge(
            (row.heads().len() + 1)
                .checked_mul(p.input_dim + 1)?
                .checked_mul(3)?
                .checked_add(p.hidden_dim * row.heads().len())?,
            0,
            replay,
        )?;
        row.matches(cuts)?;
        masks = masks.checked_add(row.mask_count())?;
        if masks > group::MAX_SELECTED_MASKS {
            return None;
        }
        for &j in row.heads() {
            if used[j] {
                return None;
            }
            used[j] = true;
        }
        sum.add(row.g(), row.upper(), 1., end, poll)?;
    }
    // Retain every completed old pair whose heads this partition does not use.
    // This avoids trading unrelated pair gains away during incremental growth.
    let mut pairs = Vec::with_capacity(cuts.compact.as_ref().map_or(0, |r| r.pairs.len()));
    if let Some(old) = cuts.compact.as_ref() {
        for row in &old.pairs {
            if row.heads.iter().any(|j| used[*j]) {
                continue;
            }
            pair_replay(cuts, row, end, poll)?;
            for &j in &row.heads {
                used[j] = true;
            }
            sum.add(&row.g, row.upper, 1., end, poll)?;
            pairs.push(row.clone());
        }
    }
    for plane in &cuts.planes {
        if !used[plane.head] {
            sum.add(&plane.g, plane.upper, -p.c[plane.head], end, poll)?;
        }
    }
    let (g, upper) = sum.finish(p, end, poll)?;
    let improvement = uniform_gain(cuts, &g, upper, end, poll)?;
    if !live(end) || !poll() {
        return None;
    }
    require_hull_environment().ok()?;
    Some(Aggregate {
        pairs,
        groups,
        group_budget: Some(budget.clone()),
        _group_memory: Some(memory),
        g,
        upper,
        improvement,
    })
}
