// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! One prepaid whole-Plan proposal. Only the original Source checker is authority.
use super::*;
use std::sync::Mutex;
#[path = "single_hidden_hull_mip_signed_root_coupling_geometry.rs"]
mod geometry;
#[path = "single_hidden_hull_mip_signed_root_coupling_payload.rs"]
mod payload;
#[path = "single_hidden_hull_mip_signed_root_coupling_preserved.rs"]
mod preserved;
#[path = "single_hidden_hull_mip_signed_root_coupling_resources.rs"]
mod resources;
#[path = "single_hidden_hull_mip_signed_root_coupling_steering.rs"]
mod steering;
pub(super) const LIMIT: usize = 8_000_000;
pub(super) const PREFLIGHT: usize = 4096;

#[derive(Debug, Clone, PartialEq)]
pub struct SignedRootCouplingReport {
    pub requested: bool,
    pub admitted: bool,
    pub work_units: usize,
    pub used_work_units: usize,
    pub admitted_bytes: usize,
    pub producer_attempts: usize,
    pub checker_slots: usize,
    pub support_attempts: usize,
    pub completed_supports: usize,
    pub selected: bool,
    pub left: Option<usize>,
    pub right: Option<usize>,
    pub parent_count: usize,
    pub parent_indices: [usize; 6],
    pub selected_negatives: usize,
    pub tail_negatives: usize,
    pub positive_heads: usize,
    pub recipient_positive_heads: usize,
    pub funded_coordinates: usize,
    pub optimized_coordinates: usize,
    pub line_mask_attempts: usize,
    pub line_masks_completed: usize,
    pub line_masks_improved: usize,
    pub coordinate_visits: usize,
    pub quoted_check_work: usize,
    pub actual_check_work: usize,
    pub quoted_search_work: usize,
    pub metadata_work: usize,
    pub old_masks: usize,
    pub new_masks: usize,
    pub baseline_lower: Option<f64>,
    pub before_global_lower: Option<f64>,
    pub proposed_lower: Option<f64>,
    pub after_global_lower: Option<f64>,
    pub stop_reason: &'static str,
}
impl Default for SignedRootCouplingReport {
    fn default() -> Self {
        Self {
            requested: false,
            admitted: false,
            work_units: 0,
            used_work_units: 0,
            admitted_bytes: 0,
            producer_attempts: 0,
            checker_slots: 0,
            support_attempts: 0,
            completed_supports: 0,
            selected: false,
            left: None,
            right: None,
            parent_count: 0,
            parent_indices: [0; 6],
            selected_negatives: 0,
            tail_negatives: 0,
            positive_heads: 0,
            recipient_positive_heads: 0,
            funded_coordinates: 0,
            optimized_coordinates: 0,
            line_mask_attempts: 0,
            line_masks_completed: 0,
            line_masks_improved: 0,
            coordinate_visits: 0,
            quoted_check_work: 0,
            actual_check_work: 0,
            quoted_search_work: 0,
            metadata_work: 0,
            old_masks: 0,
            new_masks: 0,
            baseline_lower: None,
            before_global_lower: None,
            proposed_lower: None,
            after_global_lower: None,
            stop_reason: "not-attempted",
        }
    }
}
fn note(b: &Budget, f: impl FnOnce(&mut SignedRootCouplingReport)) {
    if let Ok(mut c) = b.state.lock() {
        f(&mut c.root_coupling);
    }
}
fn stop(b: &Budget, s: &'static str) {
    note(b, |r| r.stop_reason = s);
}
#[derive(Debug)]
struct Use {
    used: usize,
    producer: bool,
    checker: bool,
    selection: bool,
    build: bool,
    selected_search: bool,
    recipient_search: bool,
    choice: Option<[u64; 10]>,
    parents: [usize; 6],
    parent_count: usize,
    parent_seal: Option<[usize; 7]>,
    quote: usize,
    paid_choice: bool,
    transported_started: bool,
    // Sealed once by preserved selection, before the exact choice is paid.
    oblique_choice: Option<[u64; 10]>,
    oblique_fee: usize,
    oblique_started: bool,
}
#[derive(Clone, Copy)]
pub(super) enum SearchRole {
    Selected,
    Recipient,
}
#[derive(Debug)]
pub(super) struct Ticket {
    source: usize,
    context: usize,
    dimensions: (usize, usize, usize),
    ledger: Arc<Mutex<Counts>>,
    authority: Instant,
    search_end: Instant,
    payload: usize,
    unfunded_base: usize,
    base_bytes: usize,
    fixed_end: Instant,
    baseline_seal: Box<[u64]>,
    baseline_capacity: usize,
    preserved_mode: bool,
    transported_payload: bool,
    old_cost: usize,
    old_components: usize,
    pub(super) end: Instant,
    state: Mutex<Use>,
}
impl Ticket {
    pub(super) fn owned(&self, cuts: &HullRootMipCuts<'_>, b: &Budget) -> bool {
        self.source == std::ptr::from_ref(cuts.problem) as usize
            && self.source == b.source
            && self.context == std::ptr::from_ref(cuts) as usize
            && self.dimensions == b.dimensions
            && Arc::ptr_eq(&self.ledger, &b.state)
            && self.authority == b.authority
            && self.search_end == b.search_end
            && b.authority <= cuts.deadline
            && b.admitted_bytes == self.base_bytes
            && self.payload <= MAX_BYTES
            && self.unfunded_base.checked_add(self.payload) == Some(self.base_bytes)
            && self.end == self.fixed_end
            && self.end <= b.search_end
            && self.end <= b.authority
    }
    pub(super) fn matches_plan(&self, plan: &Plan, cost: usize) -> bool {
        if cost != self.old_cost
            || plan.components.len() != self.old_components
            || plan.leftover.len() > self.dimensions.1
        {
            return false;
        }
        for c in &plan.components {
            if !(1..=8).contains(&c.negative.len())
                || c.positive.len() > 16
                || c.positive_units.len() != c.positive.len()
                || c.g.len() != self.dimensions.0
                || c.alpha.len() != (1usize << c.negative.len()) * c.positive.len()
            {
                return false;
            }
        }
        resources::words(plan) == Some(self.baseline_seal.len())
            && payload::capacity_matches(
                plan,
                self.baseline_capacity,
                self.baseline_seal.len(),
                self.old_components,
                self.dimensions.0,
            )
            && payload::matches(plan, &self.baseline_seal)
    }
    pub(super) fn alive(
        &self,
        cuts: &HullRootMipCuts<'_>,
        b: &Budget,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<()> {
        (self.owned(cuts, b) && live(self.end) && poll()).then_some(())
    }
    fn begin(&self, cuts: &HullRootMipCuts<'_>, b: &Budget) -> Option<()> {
        if !self.owned(cuts, b) || !live(self.end) {
            return None;
        }
        let mut u = self.state.lock().ok()?;
        if u.producer {
            return None;
        }
        u.producer = true;
        note(b, |r| r.producer_attempts += 1);
        Some(())
    }
    fn claim_selection(
        &self,
        cuts: &HullRootMipCuts<'_>,
        b: &Budget,
        plan: &Plan,
        cost: usize,
    ) -> Option<()> {
        self.claim_selection_mode(cuts, b, plan, cost, false)
    }
    fn claim_selection_mode(
        &self,
        cuts: &HullRootMipCuts<'_>,
        b: &Budget,
        plan: &Plan,
        cost: usize,
        preserved_mode: bool,
    ) -> Option<()> {
        if self.preserved_mode != preserved_mode || !self.owned(cuts, b) || !live(self.end) {
            return None;
        }
        {
            let mut u = self.state.lock().ok()?;
            if !u.producer || u.selection {
                return None;
            }
            u.selection = true;
        }
        self.matches_plan(plan, cost).then_some(())
    }
    fn bind_union_choice(
        &self,
        cuts: &HullRootMipCuts<'_>,
        b: &Budget,
        choice: &resources::Choice,
        parents: &[usize],
    ) -> Option<()> {
        if self.preserved_mode
            || !self.owned(cuts, b)
            || !live(self.end)
            || choice.check.checked_add(choice.search)? != choice.total
            || !(2..=6).contains(&parents.len())
            || parents.windows(2).any(|w| w[0] >= w[1])
            || parents.last().copied()? >= self.old_components
            || choice.left != parents[0]
            || choice.right != parents[1]
            || !(1..=6).contains(&choice.selected)
            || !(1..=16).contains(&choice.positive)
            || choice.recipient > 16
            || choice.active > choice.positive.min(6)
            || choice.check == 0
        {
            return None;
        }
        let mut u = self.state.lock().ok()?;
        if !u.selection || u.choice.is_some() {
            return None;
        }
        u.parents[..parents.len()].copy_from_slice(parents);
        u.parent_count = parents.len();
        let mut seal = [0; 7];
        seal[0] = parents.len();
        seal[1..=parents.len()].copy_from_slice(parents);
        u.parent_seal = Some(seal);
        u.choice = Some(choice.bits());
        u.quote = choice.total;
        Some(())
    }
    fn parents(
        &self,
        cuts: &HullRootMipCuts<'_>,
        b: &Budget,
        choice: &resources::Choice,
    ) -> Option<([usize; 6], usize)> {
        if self.preserved_mode || !self.owned(cuts, b) || !live(self.end) {
            return None;
        }
        let u = self.state.lock().ok()?;
        if u.choice != Some(choice.bits())
            || !(2..=6).contains(&u.parent_count)
            || u.parents[..u.parent_count].windows(2).any(|w| w[0] >= w[1])
            || u.parents[u.parent_count - 1] >= self.old_components
            || choice.left != u.parents[0]
            || choice.right != u.parents[1]
        {
            return None;
        }
        let mut seal = [0; 7];
        seal[0] = u.parent_count;
        seal[1..=u.parent_count].copy_from_slice(&u.parents[..u.parent_count]);
        if u.parent_seal != Some(seal) {
            return None;
        }
        Some((u.parents, u.parent_count))
    }
    fn claim_build(
        &self,
        cuts: &HullRootMipCuts<'_>,
        b: &Budget,
        plan: &Plan,
        choice: &resources::Choice,
    ) -> Option<()> {
        if !self.owned(cuts, b) || !live(self.end) {
            return None;
        }
        self.parents(cuts, b, choice)?;
        {
            let mut u = self.state.lock().ok()?;
            if u.build || !u.paid_choice || u.choice != Some(choice.bits()) {
                return None;
            }
            u.build = true;
        }
        self.matches_plan(plan, self.old_cost).then_some(())
    }
    fn claim_search(
        &self,
        cuts: &HullRootMipCuts<'_>,
        b: &Budget,
        c: &Component,
        active: usize,
        role: SearchRole,
    ) -> Option<()> {
        if self.preserved_mode || !self.owned(cuts, b) || !live(self.end) {
            return None;
        }
        let mut u = self.state.lock().ok()?;
        if !u.build || !u.paid_choice {
            return None;
        }
        let bits = u.choice?;
        let (max_k, max_p, max_active) = match role {
            SearchRole::Selected => {
                if u.selected_search {
                    return None;
                }
                u.selected_search = true;
                (bits[2] as usize, bits[3] as usize, bits[5] as usize)
            }
            SearchRole::Recipient => {
                if u.recipient_search {
                    return None;
                }
                u.recipient_search = true;
                (1, bits[4] as usize, bits[4] as usize)
            }
        };
        drop(u);
        if !(1..=6).contains(&max_k) || max_p > 16 || max_active > 16 {
            return None;
        }
        if c.negative.len() != max_k
            || c.positive.is_empty()
            || c.positive.len() > max_p
            || c.positive_units.len() != c.positive.len()
            || c.g.len() != self.dimensions.0
            || c.alpha.len() != (1usize << max_k) * c.positive.len()
            || active > max_active
            || c.negative.iter().any(|j| {
                *j >= cuts.problem.hidden_dim
                    || cuts.problem.c[*j].partial_cmp(&0.) != Some(std::cmp::Ordering::Less)
            })
            || c.positive.iter().any(|j| {
                *j >= cuts.problem.hidden_dim
                    || cuts.problem.c[*j].partial_cmp(&0.) != Some(std::cmp::Ordering::Greater)
            })
            || c.positive_units.iter().any(|u| !matches!(u, 1 | 2))
        {
            return None;
        }
        Some(())
    }
    pub(super) fn remaining(&self, cuts: &HullRootMipCuts<'_>, b: &Budget) -> Option<usize> {
        if !self.owned(cuts, b) || !live(self.end) {
            return None;
        }
        LIMIT.checked_sub(self.state.lock().ok()?.used)
    }
    fn pay(&self, cuts: &HullRootMipCuts<'_>, b: &Budget, work: usize) -> Option<()> {
        if !self.owned(cuts, b) || !live(self.end) {
            return None;
        }
        let mut u = self.state.lock().ok()?;
        if !u.producer {
            return None;
        }
        let next = u.used.checked_add(work)?;
        if next > LIMIT {
            return None;
        }
        u.used = next;
        if u.choice.is_some() && u.quote == work {
            u.paid_choice = true;
        }
        note(b, |r| r.used_work_units = next);
        Some(())
    }
    fn claim_checker(&self, cuts: &HullRootMipCuts<'_>, b: &Budget) -> Option<()> {
        self.claim_checker_mode(cuts, b, false)
    }
    fn claim_checker_mode(
        &self,
        cuts: &HullRootMipCuts<'_>,
        b: &Budget,
        preserved_mode: bool,
    ) -> Option<()> {
        if self.preserved_mode != preserved_mode || !self.owned(cuts, b) || !live(self.end) {
            return None;
        }
        let mut u = self.state.lock().ok()?;
        if !u.producer || !u.build || !u.paid_choice || u.checker {
            return None;
        }
        u.checker = true;
        note(b, |r| r.support_attempts += 1);
        Some(())
    }
}
pub(super) fn admit(
    cuts: &HullRootMipCuts<'_>,
    plan: &Plan,
    old_cost: usize,
    b: &mut Budget,
) -> Option<(Arc<Ticket>, usize)> {
    if b.root_coupling.is_some() || b.prepaid.is_some() {
        return None;
    }
    {
        let mut c = b.state.lock().ok()?;
        if c.root_coupling.requested || c.replay_attempts != 0 {
            return None;
        }
        c.root_coupling.requested = true;
        c.root_coupling.stop_reason = "preflight";
    }
    b.charge(PREFLIGHT)?;
    note(b, |r| {
        r.work_units = PREFLIGHT;
        r.used_work_units = PREFLIGHT;
    });
    let preserved_mode = preserved::eligible(plan);
    let (meta, payload, _, transported_payload) = if preserved_mode {
        preserved::admission_with_transport(plan, b)?
    } else {
        let (meta, payload, words) = resources::admission(plan, b)?;
        (meta, payload, words, false)
    };
    let baseline_capacity = payload::heap(plan)?;
    let end = Instant::now()
        .checked_add(Duration::from_millis(50))?
        .min(b.search_end)
        .min(b.authority)
        .min(cuts.deadline);
    if meta.checked_add(PREFLIGHT)? > LIMIT || !live(end) {
        return None;
    }
    let base = b.admitted_bytes.checked_add(payload)?;
    // Ignore removed old components for a conservative metadata-only bound.
    // BOTH alternatives' publication costs are protected before the8M debit.
    let max_check = old_cost
        .checked_add(resources::component_cost(b.dimensions.0, 6, 16, 16)?)?
        .checked_add(15usize.checked_mul(resources::component_cost(b.dimensions.0, 1, 0, 0)?)?)?
        .checked_add(
            resources::component_cost(b.dimensions.0, 1, 16, 16)?
                .checked_sub(resources::component_cost(b.dimensions.0, 1, 0, 0)?)?,
        )?;

    let mut c = b.state.lock().ok()?;
    let reserved = c.reserved.max(max_check.checked_mul(2)?);
    if c.checks >= MAX_CHECKS
        || c.replay_attempts != 0
        || c.component_rows.attempted
        || c.work
            .checked_add(LIMIT - PREFLIGHT)?
            .checked_add(reserved)?
            .checked_add(c.component_reserved)?
            > MAX_WORK
        || base
            .checked_add(c.shared_bytes)?
            .checked_add(c.direction.cache_bytes)?
            .checked_add(c.child.cache_bytes)?
            .checked_add(c.component_rows.admitted_bytes)?
            > MAX_BYTES
    {
        return None;
    }
    c.work = c.work.checked_add(LIMIT - PREFLIGHT)?;
    c.checks += 1;
    c.reserved = reserved;
    c.root_coupling.admitted = true;
    c.root_coupling.work_units = LIMIT;
    c.root_coupling.admitted_bytes = payload;
    c.root_coupling.checker_slots = 1;
    c.root_coupling.metadata_work = meta;
    c.root_coupling.used_work_units = PREFLIGHT.checked_add(meta)?;
    c.root_coupling.stop_reason = "funded";
    drop(c);
    let unfunded_base = b.admitted_bytes;
    b.admitted_bytes = base;
    // The full metadata fee is irrevocably consumed BEFORE creating this seal.
    let baseline_seal = payload::seal(plan)?;
    let ticket = Arc::new(Ticket {
        source: b.source,
        context: std::ptr::from_ref(cuts) as usize,
        dimensions: b.dimensions,
        ledger: b.state.clone(),
        authority: b.authority,
        search_end: b.search_end,
        payload,
        unfunded_base,
        base_bytes: base,
        end,
        fixed_end: end,
        baseline_seal,
        baseline_capacity,
        preserved_mode,
        transported_payload,
        old_cost,
        old_components: plan.components.len(),
        state: Mutex::new(Use {
            used: PREFLIGHT.checked_add(meta)?,
            producer: false,
            checker: false,
            selection: false,
            build: false,
            selected_search: false,
            recipient_search: false,
            choice: None,
            parents: [0; 6],
            parent_count: 0,
            parent_seal: None,
            quote: 0,
            paid_choice: false,
            transported_started: false,
            oblique_choice: None,
            oblique_fee: 0,
            oblique_started: false,
        }),
    });
    b.root_coupling = Some(ticket.clone());
    Some((ticket, meta))
}
/// True means the exclusive debt/slot/payload was admitted, even on later
/// refusal. A baseline checked owner survives every unsuccessful proposal.
pub(super) fn run(
    cuts: &HullRootMipCuts<'_>,
    plan: &mut Plan,
    baseline: &mut certificate::Checked,
    cost: &mut usize,
    floor: f64,
    b: &mut Budget,
    poll: &mut dyn FnMut() -> bool,
) -> bool {
    let Some((ticket, _meta)) = admit(cuts, plan, *cost, b) else {
        stop(b, "admission-refused");
        return false;
    };
    note(b, |r| {
        r.baseline_lower = Some(baseline.root_lower);
        r.before_global_lower = Some(baseline.root_lower.max(floor));
        r.after_global_lower = r.before_global_lower;
        r.old_masks = plan.masks().unwrap_or(0);
    });
    if ticket.begin(cuts, b).is_none() {
        stop(b, "metadata-refused");
        return true;
    }
    if ticket.preserved_mode {
        return preserved::run(cuts, plan, baseline, cost, floor, b, &ticket, poll);
    }
    let Some(choice) = resources::choose(cuts, plan, b, &ticket, *cost, poll) else {
        stop(b, "no-funded-union");
        return true;
    };
    if b.reserve_at_least((*cost).max(choice.check)).is_none() {
        stop(b, "publication-holds");
        return true;
    }
    if ticket.pay(cuts, b, choice.total).is_none() {
        stop(b, "whole-quote-refused");
        return true;
    }
    let Some((parents, parent_count)) = ticket.parents(cuts, b, &choice) else {
        stop(b, "parent-seal-refused");
        return true;
    };
    note(b, |r| {
        r.left = Some(choice.left);
        r.right = Some(choice.right);
        r.parent_count = parent_count;
        r.parent_indices = parents;
        r.selected_negatives = choice.selected;
        r.tail_negatives = parents[..parent_count]
            .iter()
            .map(|i| plan.components[*i].negative.len())
            .sum::<usize>()
            - choice.selected;
        r.quoted_check_work = choice.check;
        r.quoted_search_work = choice.search;
        r.funded_coordinates = choice.active + choice.recipient;
    });
    let Some(mut built) = geometry::build(cuts, plan, &choice, b, &ticket, poll) else {
        stop(b, "geometry-refused");
        return true;
    };
    let Some(visited) = steering::optimize(
        cuts,
        &mut built.plan.components[built.selected],
        choice.active,
        SearchRole::Selected,
        b,
        &ticket,
        poll,
    ) else {
        stop(b, "selected-search-refused");
        return true;
    };
    note(b, |r| r.optimized_coordinates = visited);
    if let Some(index) = built.recipient {
        let active = built.plan.components[index].positive.len();
        if steering::optimize(
            cuts,
            &mut built.plan.components[index],
            active,
            SearchRole::Recipient,
            b,
            &ticket,
            poll,
        )
        .is_none()
        {
            stop(b, "recipient-search-refused");
            return true;
        }
        note(b, |r| r.optimized_coordinates += active);
    }
    let Some(new_cost) = built.plan.cost(cuts.problem) else {
        stop(b, "whole-cost-refused");
        return true;
    };
    if new_cost > choice.check || ticket.claim_checker(cuts, b).is_none() {
        stop(b, "checker-permit");
        return true;
    }
    let Some(checked) = certificate::check(cuts, &built.plan, b, ticket.end, poll) else {
        stop(b, "whole-source-refused");
        return true;
    };
    note(b, |r| {
        r.completed_supports += 1;
        r.actual_check_work = new_cost;
        r.new_masks = built.plan.masks().unwrap_or(0);
        r.positive_heads = built.plan.components[built.selected].positive.len();
        r.recipient_positive_heads = built
            .recipient
            .map_or(0, |i| built.plan.components[i].positive.len());
        r.proposed_lower = Some(checked.root_lower);
    });
    if ticket.alive(cuts, b, poll).is_none() || require_hull_environment().is_err() {
        stop(b, "completion-barrier");
        return true;
    }
    if checked.root_lower > baseline.root_lower.max(floor) {
        *plan = built.plan;
        *baseline = checked;
        *cost = new_cost;
        note(b, |r| {
            r.selected = true;
            r.after_global_lower = Some(baseline.root_lower);
            r.stop_reason = "root-improved";
        });
    } else {
        stop(b, "baseline-retained");
    }
    true
}

#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_root_coupling_default_tests.rs"]
mod default_tests;
#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_root_coupling_retention_tests.rs"]
mod retention_tests;
#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_root_coupling_test_fixture.rs"]
mod test_fixture;
#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_root_coupling_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_root_coupling_default_fixture.rs"]
mod default_fixture;

#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_root_coupling_growth_tests.rs"]
mod growth_tests;

#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_root_coupling_line_source_tests.rs"]
mod line_source_tests;
