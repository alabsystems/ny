// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Completed signed-mask constants reused only inside the original phase tree.
use super::certificate::I;
use super::*;
#[path = "single_hidden_hull_mip_signed_child_reservation.rs"]
mod reservation;
use crate::single_hidden_hull::bab::HullSplit;
pub(super) use reservation::prepay;

pub(in crate::single_hidden_hull) const MAX_CHILD_CALLS: usize = 256;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct SignedChildReport {
    pub collection_attempts: usize,
    pub cached_masks: usize,
    /// Retained payload admission, including a prepaid hold before collection.
    pub cache_bytes: usize,
    pub bind_attempts: usize,
    pub attempted_calls: usize,
    pub completed_calls: usize,
    pub work_units: usize,
    pub total_work_units: usize,
    pub source_replays: usize,
    pub positive_calls: usize,
    pub lp_skips: usize,
    /// Admitted optional axes; zero when every corner mode refused.
    pub corner_axes: usize,
    /// Consumed optional child-price credits, including numerical refusals.
    pub corner_price_attempts: usize,
    /// Stored floor from an attached extension; telemetry, not bind authority.
    pub corner_root_lower: Option<f64>,
    /// Admitted root-Ray axes; zero when its optional quote refused.
    pub corner_ray_axes: usize,
    /// Baseline floor: original corner floor, or ordinary certificate in adaptive mode.
    pub corner_base_root_lower: Option<f64>,
    /// Complete alternate floor, including non-improving values; telemetry only.
    pub corner_ray_root_lower: Option<f64>,
    /// Root-only adaptive mode was paid; this does not assert materialization.
    pub corner_cover_admitted: bool,
    /// Complete attached cover leaves; zero before attachment or on refusal.
    pub corner_cover_leaves: usize,
    /// Complete adaptive floor, including non-improving values; telemetry only.
    pub corner_cover_root_lower: Option<f64>,
    pub stop_reason: &'static str,
}

#[derive(Debug)]
pub(super) struct Collection {
    state: Arc<std::sync::Mutex<Counts>>,
    bytes: usize,
    masks: usize,
}

/// Collection consumes its one prepaid credit or pays from original spare work.
pub(super) fn admit(
    cuts: &HullRootMipCuts<'_>,
    plan: &Plan,
    b: &Budget,
    end: Instant,
) -> Option<Collection> {
    let mut counts = b.state.lock().ok()?;
    if counts.child.collection_attempts != 0 {
        return None;
    }
    counts.child.collection_attempts = 1;
    let prepaid = std::mem::take(&mut counts.prepaid_collection_work);
    counts.child.stop_reason = "collection-admission";
    let quote = reservation::quote(cuts, plan)?;
    let (masks, bytes, cost) = (quote.masks, quote.bytes, quote.collection_work);
    if b.source != std::ptr::from_ref(cuts.problem) as usize || b.authority > cuts.deadline {
        return None;
    }
    if prepaid != 0 {
        if cost > prepaid || bytes > counts.child.cache_bytes || !live(end.min(b.authority)) {
            counts.child.stop_reason = "prepaid-collection-refused";
            return None;
        }
        let bytes = counts.child.cache_bytes;
        drop(counts);
        return Some(Collection {
            state: b.state.clone(),
            bytes,
            masks,
        });
    }
    let total = b
        .admitted_bytes
        .checked_add(counts.shared_bytes)?
        .checked_add(counts.direction.cache_bytes)?
        .checked_add(counts.component_rows.admitted_bytes)?
        .checked_add(counts.child.cache_bytes.max(bytes))?;
    if total > MAX_BYTES || !live(end.min(b.authority)) {
        counts.child.stop_reason = "collection-admission";
        return None;
    }
    let next = counts.work.checked_add(cost)?;
    if next
        .checked_add(counts.reserved)?
        .checked_add(counts.component_reserved)?
        > MAX_WORK
    {
        counts.child.stop_reason = "collection-work";
        return None;
    }
    counts.work = next;
    counts.child.work_units = counts.child.work_units.checked_add(cost)?;
    counts.child.cache_bytes = bytes;
    drop(counts);
    Some(Collection {
        state: b.state.clone(),
        bytes,
        masks,
    })
}

#[derive(Debug)]
struct Table {
    heads: Vec<usize>,
    values: Vec<f64>,
    minimum: f64,
}

#[derive(Debug)]
pub(in crate::single_hidden_hull) struct Cache {
    source: usize,
    signature: Arc<RetainedWords>,
    seal: Vec<u64>,
    tables: Vec<Table>,
    head_slot: Vec<Option<(usize, usize)>>,
    g: Vec<f64>,
    constant: f64,
    root_lower: f64,
    budget: Budget,
    corners: Option<Box<corners::Extension>>,
}

impl Cache {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn finish(
        cuts: &HullRootMipCuts<'_>,
        plan: &Plan,
        checked: &certificate::Checked,
        tables: Vec<Vec<f64>>,
        signature: Arc<RetainedWords>,
        b: &Budget,
        permit: Collection,
        corner: Option<corners::Finished>,
        end: Instant,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<Arc<Self>> {
        if !Arc::ptr_eq(&permit.state, &b.state)
            || tables.len() != plan.components.len()
            || !live(end.min(b.authority))
            || !poll()
        {
            return None;
        }
        let mut head_slot = vec![None; cuts.problem.hidden_dim];
        let mut owned = Vec::with_capacity(tables.len());
        let mut total = 0usize;
        for (index, (component, values)) in plan.components.iter().zip(tables).enumerate() {
            if component.negative.is_empty() || component.negative.len() > 8 {
                return None;
            }
            let size = 1usize.checked_shl(component.negative.len() as u32)?;
            if values.len() != size || values.iter().any(|x| !x.is_finite()) {
                return None;
            }
            total = total.checked_add(size)?;
            for (bit, &head) in component.negative.iter().enumerate() {
                let slot = head_slot.get_mut(head)?;
                if slot.is_some() {
                    return None;
                }
                *slot = Some((index, bit));
            }
            let minimum = values.iter().copied().fold(f64::INFINITY, f64::min);
            owned.push(Table {
                heads: component.negative.clone(),
                values,
                minimum,
            });
        }
        if total != permit.masks || total > MAX_MASKS {
            return None;
        }
        let mut cache = Self {
            source: b.source,
            signature,
            seal: Vec::new(),
            tables: owned,
            head_slot,
            g: checked.g.clone(),
            constant: checked.constant,
            root_lower: checked.root_lower,
            budget: b.clone(),
            corners: None,
        };
        let mut seal = Vec::with_capacity(cache.bits().count());
        seal.extend(cache.bits());
        cache.seal = seal;
        require_hull_environment().ok()?;
        if !live(end.min(b.authority)) || !poll() {
            return None;
        }
        let mut counts = b.state.lock().ok()?;
        counts.child.cached_masks = total;
        counts.child.cache_bytes = permit.bytes;
        counts.child.stop_reason = "cached";
        drop(counts);
        let mut cache = Arc::new(cache);
        match corner {
            Some(corners::Finished::Data(data, alternate)) => {
                let owner = Arc::as_ptr(&cache) as usize;
                let extension = corners::Extension::finish_with_root_ray(
                    data, alternate, &checked.g, b, owner, end, poll,
                )?;
                Arc::get_mut(&mut cache)?.corners = extension;
            }
            Some(corners::Finished::AdaptiveRoot(root)) => {
                let owner = Arc::as_ptr(&cache) as usize;
                let extension = corners::Extension::finish_adaptive(
                    root,
                    checked.root_lower,
                    b,
                    owner,
                    end,
                    poll,
                )?;
                Arc::get_mut(&mut cache)?.corners = extension;
            }
            _ => {}
        }
        Some(cache)
    }

    fn bits(&self) -> impl Iterator<Item = u64> + '_ {
        [
            self.g.len() as u64,
            self.tables.len() as u64,
            self.constant.to_bits(),
            self.root_lower.to_bits(),
            self.head_slot.len() as u64,
        ]
        .into_iter()
        .chain(self.head_slot.iter().flat_map(|slot| match slot {
            Some((c, b)) => [1, *c as u64, *b as u64],
            None => [0, 0, 0],
        }))
        .chain(self.g.iter().map(|x| x.to_bits()))
        .chain(self.tables.iter().flat_map(|t| {
            [
                t.heads.len() as u64,
                t.values.len() as u64,
                t.minimum.to_bits(),
            ]
            .into_iter()
            .chain(t.heads.iter().map(|x| *x as u64))
            .chain(t.values.iter().map(|x| x.to_bits()))
        }))
    }

    pub(in crate::single_hidden_hull) fn stats(&self) -> SignedChildReport {
        let mut report = stats(&self.budget);
        report.corner_root_lower = self
            .corners
            .as_ref()
            .map(|extension| extension.root())
            .filter(|lower| lower.is_finite());
        report.corner_base_root_lower = self
            .corners
            .as_ref()
            .map(|extension| extension.base_root())
            .filter(|lower| lower.is_finite());
        report.corner_ray_root_lower = self
            .corners
            .as_ref()
            .and_then(|extension| extension.ray_root())
            .filter(|lower| lower.is_finite());
        report.corner_cover_leaves = self
            .corners
            .as_ref()
            .map_or(0, |extension| extension.cover_leaves());
        report.corner_cover_root_lower = self
            .corners
            .as_ref()
            .and_then(|extension| extension.cover_root())
            .filter(|lower| lower.is_finite());
        report
    }

    pub(in crate::single_hidden_hull) fn bind<'a>(
        &'a self,
        cuts: &'a HullRootMipCuts<'_>,
        end: Instant,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<BoundCache<'a>> {
        let end = end.min(cuts.deadline).min(self.budget.authority);
        // Claim/burn the first-bind credit even when an untrusted shape makes
        // its quote overflow. MAX is an unaffordable refusal, not actual work.
        let cost = cuts
            .problem
            .input_dim
            .checked_mul(cuts.problem.hidden_dim)
            .and_then(|v| v.checked_add(self.seal.len()))
            .and_then(|v| v.checked_add(cuts.problem.hidden_dim))
            .and_then(|v| v.checked_add(1))
            .and_then(|v| v.checked_mul(64))
            .unwrap_or(usize::MAX);
        self.budget.child_charge(cost, false, end)?;
        shape(cuts)?;
        if self.source != std::ptr::from_ref(cuts.problem) as usize
            || !signature(cuts).eq(self.signature.iter().copied())
            || !self.bits().eq(self.seal.iter().copied())
            || self.g.len() != cuts.problem.input_dim
            || self.head_slot.len() != cuts.problem.hidden_dim
            || !live(end)
            || !poll()
        {
            return None;
        }
        require_hull_environment().ok()?;
        let corner = self
            .corners
            .as_ref()
            .and_then(|v| v.bind(&self.budget, std::ptr::from_ref(self) as usize));
        require_hull_environment().ok()?;
        if !live(end) || !poll() {
            return None;
        }
        Some(BoundCache {
            corner,
            cache: self,
            problem: cuts.problem,
            end,
        })
    }
}

/// One validated immutable original problem borrow, shared by child workers.
pub(in crate::single_hidden_hull) struct BoundCache<'a> {
    corner: Option<&'a corners::Extension>,
    cache: &'a Cache,
    problem: &'a HullMarginProblem,
    end: Instant,
}

impl BoundCache<'_> {
    pub(in crate::single_hidden_hull) fn matches_problem(
        &self,
        problem: &HullMarginProblem,
    ) -> bool {
        std::ptr::eq(self.problem, problem)
    }
    pub(in crate::single_hidden_hull) fn lower(
        &self,
        splits: &[HullSplit],
        beta: &[f64],
        end: Instant,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<f64> {
        self.lower_inner(splits, Some(beta), end, poll)
    }

    pub(in crate::single_hidden_hull) fn lower_zero(
        &self,
        splits: &[HullSplit],
        end: Instant,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<f64> {
        self.lower_inner(splits, None, end, poll)
    }

    fn lower_inner(
        &self,
        splits: &[HullSplit],
        beta: Option<&[f64]>,
        end: Instant,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<f64> {
        let p = self.problem;
        let end = end.min(self.end).min(self.cache.budget.authority);
        if beta.is_some_and(|b| b.len() != splits.len()) || splits.len() > p.hidden_dim {
            return None;
        }
        let masks = self
            .cache
            .tables
            .iter()
            .try_fold(0usize, |n, t| n.checked_add(t.values.len()))?;
        let cost = p
            .input_dim
            .checked_mul(splits.len().checked_add(1)?)?
            .checked_add(masks)?
            .checked_add(p.hidden_dim)?
            .checked_add(self.cache.tables.len())?
            .checked_add(1)?
            .checked_mul(32)?;
        self.cache.budget.child_charge(cost, true, end)?;
        require_hull_environment().ok()?;
        let mut phases = vec![false; p.hidden_dim];
        let mut on = vec![0usize; self.cache.tables.len()];
        let mut off = vec![0usize; self.cache.tables.len()];
        for split in splits {
            let j = usize::try_from(split.neuron).ok()?;
            let present = phases.get_mut(j)?;
            if *present {
                return None;
            }
            *present = true;
            if let Some((component, bit)) = self.cache.head_slot[j] {
                if split.active {
                    on[component] |= 1usize << bit;
                } else {
                    off[component] |= 1usize << bit;
                }
            }
        }
        let mut gain = 0.;
        for (index, table) in self.cache.tables.iter().enumerate() {
            let mut minimum = f64::INFINITY;
            for (mask, &value) in table.values.iter().enumerate() {
                if !live(end) || !poll() {
                    return None;
                }
                if mask & on[index] == on[index] && mask & off[index] == 0 {
                    minimum = minimum.min(value);
                }
            }
            if !minimum.is_finite() || minimum < table.minimum {
                return None;
            }
            let delta = finite(add_down(minimum, -table.minimum))?.max(0.);
            if delta != 0. {
                gain = finite(add_down(gain, delta))?;
            }
        }
        let intercept = if gain == 0. {
            self.cache.constant
        } else {
            finite(add_down(self.cache.constant, gain))?
        };
        let mut constant = I::point(p.d)?.add(I::point(intercept)?)?;
        let mut coeff = Vec::with_capacity(p.input_dim);
        for &g in &self.cache.g {
            coeff.push(I::point(g)?);
        }
        let base_constant = constant;
        if let Some(beta) = beta {
            for (split, &raw) in splits.iter().zip(beta) {
                if !live(end) || !poll() || raw.is_infinite() {
                    return None;
                }
                let value = if raw > 0. { raw } else { 0. };
                let q = if split.active { -value } else { value };
                let head = usize::try_from(split.neuron).ok()?;
                constant = constant.add(I::point(q)?.scale(p.b1[head])?)?;
                for i in 0..p.input_dim {
                    coeff[i] = coeff[i].add(I::point(q)?.scale(p.w_row(head)[i])?)?;
                }
            }
        }
        let mut lower = constant.lo;
        for i in 0..p.input_dim {
            if !live(end) || !poll() {
                return None;
            }
            lower = finite(add_down(lower, coeff[i].minimum(p.lower[i], p.upper[i])?))?;
        }
        if let Some(corner) = self.corner {
            // Subtract interval enclosures of the identical base expression:
            // this encloses ONLY beta bias, not another copy of d/intercept/gain.
            let bias = if beta.is_none() {
                Some(I::zero())
            } else {
                constant.sub(base_constant)
            };
            let mut cancelled = false;
            let mut guard = || {
                let ok = !cancelled && live(end) && poll();
                cancelled |= !ok;
                ok
            };
            let optional = bias.and_then(|bias| {
                corner.lower(&self.cache.budget, &coeff, bias, &on, &off, &mut guard)
            });
            if cancelled || !live(end) {
                return None;
            }
            if let Some(value) = optional {
                lower = lower.max(value);
            }
        }
        require_hull_environment().ok()?;
        if !live(end) || !poll() {
            return None;
        }
        let mut counts = self.cache.budget.state.lock().ok()?;
        require_hull_environment().ok()?;
        if !live(end) {
            return None;
        }
        counts.child.completed_calls += 1;
        counts.child.positive_calls += usize::from(lower > 0.);
        Some(lower)
    }

    pub(in crate::single_hidden_hull) fn root_lower(&self) -> Option<f64> {
        (live(self.end.min(self.cache.budget.authority)) && require_hull_environment().is_ok())
            .then_some(self.corner.map_or(self.cache.root_lower, |v| {
                self.cache.root_lower.max(v.root())
            }))
    }

    pub(in crate::single_hidden_hull) fn skipped_lp(&self) {
        if let Ok(mut c) = self.cache.budget.state.lock() {
            c.child.lp_skips += 1;
        }
    }
}

pub(in crate::single_hidden_hull::mip_cuts) fn continue_native(
    cuts: &HullRootMipCuts<'_>,
    end: Instant,
    options: bab::HullBabOptions,
) -> Option<HullMipContinuation> {
    let cached = cuts.signed_child_cache.get();
    let bound = cached.and_then(|cache| cache.bind(cuts, end, &mut || live(end)));
    let report = if let Some(bound) = &bound {
        bab::from_completed_with_cache(
            cuts.problem,
            &cuts.multipliers,
            cuts.checked_lower,
            end,
            options,
            Some(bound),
        )
    } else {
        bab::from_completed(
            cuts.problem,
            &cuts.multipliers,
            cuts.checked_lower,
            end,
            options,
        )
    }?;
    let signed_child = cached
        .map(|cache| cache.stats())
        .or_else(|| cuts.signed.as_ref().map(|row| row.child_stats()));
    Some(HullMipContinuation {
        report,
        signed_child,
        deadline: end,
    })
}

pub(in crate::single_hidden_hull::mip_cuts) fn continue_native_with_rows(
    cuts: &HullRootMipCuts<'_>,
    end: Instant,
    options: bab::HullBabOptions,
) -> Option<HullMipContinuation> {
    let native = cuts
        .signed
        .as_ref()
        .and_then(|row| row.bind_native_rows(cuts, end, &mut || live(end)));
    let cached = cuts.signed_child_cache.get();
    let bound = cached.and_then(|cache| cache.bind(cuts, end, &mut || live(end)));
    let report = bab::from_completed_with_native(
        cuts.problem,
        &cuts.multipliers,
        cuts.checked_lower,
        end,
        options,
        bound.as_ref(),
        native.as_ref(),
    )?;
    let signed_child = cached
        .map(|cache| cache.stats())
        .or_else(|| cuts.signed.as_ref().map(|row| row.child_stats()));
    Some(HullMipContinuation {
        report,
        signed_child,
        deadline: end,
    })
}

pub(super) fn stats(b: &Budget) -> SignedChildReport {
    let counts = b.state.lock().expect("private signed ledger poisoned");
    let mut report = counts.child.clone();
    report.total_work_units = counts.work;
    report.source_replays = counts.source_replays;
    if report.stop_reason.is_empty() {
        report.stop_reason = "not-attempted";
    }
    report
}

#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_child_retention_tests.rs"]
mod retention_tests;
#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_child_tests.rs"]
mod tests;

// Test-only access to a uniquely owned completed cache; no production authority.
#[cfg(test)]
pub(super) fn corner_extension_for_test(cache: &mut Cache) -> &mut Option<Box<corners::Extension>> {
    &mut cache.corners
}
