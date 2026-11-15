// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Optional shared-input floors collected during the two original replays.
use super::certificate::I;
use super::*;
#[path = "single_hidden_hull_mip_signed_corner_adaptive.rs"]
pub(in crate::single_hidden_hull::mip_cuts::signed) mod adaptive;
#[path = "single_hidden_hull_mip_signed_corner_core.rs"]
mod core;
#[path = "single_hidden_hull_mip_signed_corner_quote.rs"]
mod quote;
#[path = "single_hidden_hull_mip_signed_corner_root_ray.rs"]
pub(in crate::single_hidden_hull::mip_cuts::signed) mod root_ray;

// Construction, ownership headers and the two-word independent seal suffix.
// Core's payload already includes the table+seal, bounded transient storage
// and eight child workers. No payload is cloned at a stage transition.
const WRAPPER_BYTES: usize = 2048;
// Seven candidate quotes scan at most128 component counts; this bounded
// setup is paid even when every mode refuses.
const SETUP_WORK: usize = 32768;
// Pair each optional price with its ordinary entry computation before search
// can spend the remaining credit. Larger calls retain the ordinary debit path.
const BASE_CALL_SPLITS: usize = 16;
// The root-only mode replaces the old sixteen optional corner-price credits.
// These four credits pay ordinary cache entry work even if optional proof fails.
const ADAPTIVE_BASE_CALLS: usize = 4;
// One quote uses fewer scans than the old seven-mode admission. This covers
// 1024 guard/arithmetic units plus seven 128-entry scans at eight units each.
const ADAPTIVE_SETUP_WORK: usize = 8192;
const ADAPTIVE_WRAPPER_BYTES: usize = 2048;
const ADAPTIVE_SEAL: u64 = 0x4144_4150_5449_5645;
pub(super) const FIXED_BYTES: usize = size_of::<State>()
    + align_of::<State>()
    + size_of::<Option<Box<Extension>>>()
    + 8 * size_of::<Option<&Extension>>()
    + 2 * size_of::<Frame>()
    + 2 * size_of::<Finished>()
    + 2 * size_of::<root_ray::Quote>()
    + 2 * size_of::<core::RootReplacement<'static>>()
    + 2 * size_of::<adaptive::Quote>()
    + 256 // bounded adaptive admission/refusal locals
    + 64 // retained and returned adaptive telemetry copies
    + 128 // optional root-only refusal/dispatch locals
    + 512 // bounded quote/capture locals and128 counts, including refusals
    // Retained Counts.child fields plus one returned report; no payload clones.
    + 2 * (3 * size_of::<usize>() + 3 * size_of::<Option<f64>>());

#[derive(Debug, Default)]
pub(super) struct State {
    attempted: bool,
    ray_attempted: bool,
    adaptive_attempted: bool,
    adaptive: Option<Box<AdaptiveGrant>>,
    grant: Option<Box<Grant>>,
}
#[derive(Debug)]
struct Grant {
    quote: quote::Quote,
    bits: Vec<u8>,
    stage: u8,
    axes: Option<Box<core::Axes>>,
    witness: Option<Arc<RetainedWords>>,
    bind: bool,
    prices: usize,
    base_call_cap: usize,
    base_calls: usize,
    root_ray: Option<Box<root_ray::Grant>>,
}

#[derive(Debug)]
struct AdaptiveGrant {
    quote: adaptive::Quote,
    bits: Vec<u8>,
    stage: u8,
    cover: Option<Box<adaptive::Cover>>,
    witness: Option<Arc<RetainedWords>>,
    bind: bool,
    base_call_cap: usize,
    base_calls: usize,
}

fn base_call_cap(d: usize, h: usize, masks: usize, components: usize) -> Option<usize> {
    d.checked_mul(BASE_CALL_SPLITS.checked_add(1)?)?
        .checked_add(masks)?
        .checked_add(h)?
        .checked_add(components)?
        .checked_add(1)?
        .checked_mul(32)
}

/// Called only while the original Budget ledger is locked. This prepaid work
/// remains usable by the ordinary cache if an optional sidecar later refuses.
/// Every eligible attempt burns one shared credit, including deadline/numeric
/// failures. An over-cap call never consumes it and follows the original path.
pub(super) fn take_base_call(state: &mut State, cost: usize) -> usize {
    if let Some(grant) = state.adaptive.as_mut() {
        if grant.base_calls == 0 || cost > grant.base_call_cap {
            return 0;
        }
        grant.base_calls -= 1;
        return grant.base_call_cap;
    }
    let Some(grant) = state.grant.as_mut() else {
        return 0;
    };
    if grant.base_calls == 0 || cost > grant.base_call_cap {
        return 0;
    }
    grant.base_calls -= 1;
    grant.base_call_cap
}

/// Irreversible extra work and payload, after base native and child holds.
/// Child cache_bytes owns this payload throughout collection; the immutable
/// Budget.admitted_bytes identity and ordinary prepaid credits stay unchanged.
pub(super) fn prepay(cuts: &HullRootMipCuts<'_>, plan: &Plan, b: &Budget) -> Option<()> {
    let funded = native_publication::funding(b).funded_calls;
    let mut c = b.state.lock().ok()?;
    if c.corner.attempted || c.corner.adaptive.is_some() {
        return None;
    }
    c.corner.attempted = true;
    if !b.portfolio_phase_enabled
        || b.source != std::ptr::from_ref(cuts.problem) as usize
        || b.authority > cuts.deadline
        || !live(b.search_end)
        || c.replay_attempts != 0
        || c.source_replays != 0
        || c.child.collection_attempts != 0
        || c.child.bind_attempts != 0
        || c.child.cache_bytes == 0
        || c.prepaid_collection_work == 0
        || c.prepaid_bind_work == 0
        || funded == 0
        || cuts.problem.input_dim == 0
        || cuts.problem.input_dim > core::MAX_D
        || plan.components.is_empty()
        || plan.components.len() > 128
    {
        return None;
    }
    let work = c.work.checked_add(SETUP_WORK)?;
    let child_work = c.child.work_units.checked_add(SETUP_WORK)?;
    if work
        .checked_add(c.reserved)?
        .checked_add(c.component_reserved)?
        > MAX_WORK
    {
        return None;
    }
    c.work = work;
    c.child.work_units = child_work;
    // Fixed bounded stack advice; no heap allocation before full admission.
    let mut bits = [0u8; 128];
    for (slot, component) in bits.iter_mut().zip(&plan.components) {
        *slot = u8::try_from(component.negative.len()).ok()?;
    }
    let bits = &bits[..plan.components.len()];
    let masks = bits.iter().try_fold(0usize, |m, &n| {
        if n == 0 || n > 8 {
            return None;
        }
        m.checked_add(1usize << n)
    })?;
    // Two replay callback dispatches/guards per mask, plus independent bind,
    // immutable attachment, sixteen child-price dispatches and bounded scans.
    let wrapper_work = 16_384usize
        .checked_add(32usize.checked_mul(masks)?)?
        .checked_add(128usize.checked_mul(bits.len())?)?;
    let base_call_cap = base_call_cap(
        cuts.problem.input_dim,
        cuts.problem.hidden_dim,
        masks,
        bits.len(),
    )?;
    let base_call_work = quote::CHILD_PRICES.checked_mul(base_call_cap)?;
    for k in (1..=cuts.problem.input_dim.min(core::MAX_K)).rev() {
        let q = quote::quote(cuts.problem.input_dim, bits, k)?;
        let work = q
            .work
            .checked_add(wrapper_work)?
            .checked_add(base_call_work)?;
        let bytes = q.bytes.checked_add(WRAPPER_BYTES)?;
        let next_work = c.work.checked_add(work)?;
        let next_child_work = c.child.work_units.checked_add(work)?;
        let next_bytes = c.child.cache_bytes.checked_add(bytes)?;
        let total_bytes = b
            .admitted_bytes
            .checked_add(c.shared_bytes)?
            .checked_add(c.direction.cache_bytes)?
            .checked_add(c.component_rows.admitted_bytes)?
            .checked_add(next_bytes)?;
        if next_work
            .checked_add(c.reserved)?
            .checked_add(c.component_reserved)?
            > MAX_WORK
            || total_bytes > MAX_BYTES
        {
            continue;
        }
        if !live(b.search_end) {
            return None;
        }
        c.work = next_work;
        c.child.work_units = next_child_work;
        c.child.cache_bytes = next_bytes;
        c.child.corner_axes = k;
        c.corner.grant = Some(Box::new(Grant {
            quote: q,
            bits: bits.to_vec(),
            stage: 0,
            axes: None,
            witness: None,
            bind: true,
            prices: quote::CHILD_PRICES,
            base_call_cap,
            base_calls: quote::CHILD_PRICES,
            root_ray: None,
        }));
        return Some(());
    }
    None
}

/// Default root-only endpoint cover. Refusal leaves the old admission path
/// available, but setup work and the attempt latch are never refunded.
/// This mode intentionally does not promise the old conditional corner floors.
pub(super) fn prepay_adaptive(cuts: &HullRootMipCuts<'_>, plan: &Plan, b: &Budget) -> Option<()> {
    let funded = native_publication::funding(b).funded_calls;
    let mut c = b.state.lock().ok()?;
    if std::mem::replace(&mut c.corner.adaptive_attempted, true) {
        return None;
    }
    if c.corner.attempted
        || c.corner.grant.is_some()
        || c.corner.adaptive.is_some()
        || !b.portfolio_phase_enabled
        || b.source != std::ptr::from_ref(cuts.problem) as usize
        || b.authority > cuts.deadline
        || !live(b.search_end)
        || c.replay_attempts != 0
        || c.source_replays != 0
        || c.child.collection_attempts != 0
        || c.child.bind_attempts != 0
        || c.child.cache_bytes == 0
        || c.prepaid_collection_work == 0
        || c.prepaid_bind_work == 0
        || funded == 0
        || cuts.problem.input_dim == 0
        || cuts.problem.input_dim > core::MAX_D
        || plan.components.is_empty()
        || plan.components.len() > 128
    {
        return None;
    }
    let work = c.work.checked_add(ADAPTIVE_SETUP_WORK)?;
    let child_work = c.child.work_units.checked_add(ADAPTIVE_SETUP_WORK)?;
    if work
        .checked_add(c.reserved)?
        .checked_add(c.component_reserved)?
        > MAX_WORK
    {
        return None;
    }
    c.work = work;
    c.child.work_units = child_work;
    let mut bits = [0u8; 128];
    for (slot, component) in bits.iter_mut().zip(&plan.components) {
        *slot = u8::try_from(component.negative.len()).ok()?;
    }
    let bits = &bits[..plan.components.len()];
    let k = cuts.problem.input_dim.min(core::MAX_K);
    let q = adaptive::quote(cuts.problem.input_dim, bits, k)?;
    let base_cap = base_call_cap(
        cuts.problem.input_dim,
        cuts.problem.hidden_dim,
        q.masks,
        bits.len(),
    )?;
    // Both complete replay hooks, independent immutable attachment/bind,
    // four ordinary-entry dispatches and bounded component scans.
    let wrapper_work = 16_384usize
        .checked_add(32usize.checked_mul(q.masks)?)?
        .checked_add(128usize.checked_mul(bits.len())?)?;
    let extra_work = q
        .work
        .checked_add(wrapper_work)?
        .checked_add(ADAPTIVE_BASE_CALLS.checked_mul(base_cap)?)?;
    let bytes = q.bytes.checked_add(ADAPTIVE_WRAPPER_BYTES)?;
    let next_work = c.work.checked_add(extra_work)?;
    let next_child_work = c.child.work_units.checked_add(extra_work)?;
    let next_bytes = c.child.cache_bytes.checked_add(bytes)?;
    let total_bytes = b
        .admitted_bytes
        .checked_add(c.shared_bytes)?
        .checked_add(c.direction.cache_bytes)?
        .checked_add(c.component_rows.admitted_bytes)?
        .checked_add(next_bytes)?;
    if next_work
        .checked_add(c.reserved)?
        .checked_add(c.component_reserved)?
        > MAX_WORK
        || total_bytes > MAX_BYTES
        || !live(b.search_end)
    {
        return None;
    }
    // Every checked ledger replacement is ready before the grant is committed.
    c.work = next_work;
    c.child.work_units = next_child_work;
    c.child.cache_bytes = next_bytes;
    c.child.corner_axes = k;
    c.child.corner_cover_admitted = true;
    c.corner.adaptive = Some(Box::new(AdaptiveGrant {
        quote: q,
        bits: bits.to_vec(),
        stage: 0,
        cover: None,
        witness: None,
        bind: true,
        base_call_cap: base_cap,
        base_calls: ADAPTIVE_BASE_CALLS,
    }));
    Some(())
}

#[derive(Debug)]
enum Collector {
    AdaptiveCapture(Box<adaptive::Capture>),
    AdaptiveProof(Box<adaptive::Collector>),
    Rank(Box<core::Ranker>),
    Corners(Box<core::Collector>),
}
#[derive(Debug)]
pub(super) struct Frame {
    collector: Option<Collector>,
    ray: Option<Box<RayCollector>>,
}
#[derive(Debug)]
enum RayCollector {
    Capture(root_ray::Capture),
    Corners(root_ray::Builder),
}
#[derive(Debug)]
pub(super) enum Finished {
    Cover(Box<adaptive::Cover>),
    AdaptiveRoot(Box<adaptive::CheckedRoot>),
    Axes(Box<core::Axes>, Option<Box<root_ray::Advice>>),
    Data(Box<core::Data>, Option<Box<root_ray::Table>>),
}

pub(super) fn begin(
    cuts: &HullRootMipCuts<'_>,
    plan: &Plan,
    b: &Budget,
    witness: &Arc<RetainedWords>,
    first: bool,
    second: bool,
) -> Option<Frame> {
    let mut c = b.state.lock().ok()?;
    if let Some(grant) = c.corner.adaptive.as_mut() {
        if grant.bits.len() != plan.components.len()
            || grant
                .bits
                .iter()
                .zip(&plan.components)
                .any(|(&n, p)| n as usize != p.negative.len())
            || !live(b.authority.min(cuts.deadline))
        {
            return None;
        }
        let collector = if first && grant.stage == 0 {
            grant.stage = 1;
            Collector::AdaptiveCapture(Box::new(adaptive::Capture::new(
                &cuts.problem.lower,
                &cuts.problem.upper,
                &grant.bits,
                grant.quote,
            )?))
        } else if second && grant.stage == 2 {
            grant.stage = 3;
            if !Arc::ptr_eq(grant.witness.as_ref()?, witness) {
                return None;
            }
            let cover = grant.cover.take()?;
            if !cover.matches_quote(grant.quote, &grant.bits)
                || !cover.matches_bounds(&cuts.problem.lower, &cuts.problem.upper)
            {
                return None;
            }
            Collector::AdaptiveProof(Box::new(adaptive::Collector::new(*cover)?))
        } else {
            return None;
        };
        return Some(Frame {
            collector: Some(collector),
            ray: None,
        });
    }
    let grant = c.corner.grant.as_mut()?;
    if grant.bits.len() != plan.components.len()
        || grant
            .bits
            .iter()
            .zip(&plan.components)
            .any(|(&n, p)| n as usize != p.negative.len())
        || !live(b.authority.min(cuts.deadline))
    {
        return None;
    }
    let ray;
    let collector = if first && grant.stage == 0 {
        ray = grant
            .root_ray
            .as_ref()
            .and_then(|g| root_ray::Capture::new(g.quote))
            .map(|v| Box::new(RayCollector::Capture(v)));
        grant.stage = 1; // Attempts consume authority even on numeric refusal.
        Collector::Rank(Box::new(core::Ranker::new(
            &cuts.problem.lower,
            &cuts.problem.upper,
            &grant.bits,
            grant.quote.k,
        )?))
    } else if second && grant.stage == 2 {
        grant.stage = 3;
        let axes = grant.axes.take()?;
        if !Arc::ptr_eq(grant.witness.as_ref()?, witness) {
            return None;
        }
        ray = grant
            .root_ray
            .as_mut()
            .and_then(|g| root_ray::Builder::new(g.advice.take()?, &axes, g.quote))
            .map(|v| Box::new(RayCollector::Corners(v)));
        Collector::Corners(Box::new(core::Collector::new(*axes)?))
    } else {
        return None;
    };
    Some(Frame {
        collector: Some(collector),
        ray,
    })
}

impl Frame {
    pub(super) fn capture_ray(
        &mut self,
        component: usize,
        mask: usize,
        advice: Option<(usize, f64)>,
    ) {
        if let Some(RayCollector::Capture(capture)) = self.ray.as_deref_mut() {
            capture.push(component, mask, advice);
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn push(
        &mut self,
        problem: &HullMarginProblem,
        plan_component: &Component,
        component: usize,
        mask: usize,
        q: &[I],
        constant: I,
        lower: f64,
        end: Instant,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<()> {
        let mut cancelled = false;
        let mut guard = || {
            let ok = !cancelled && live(end) && poll();
            cancelled |= !ok;
            ok
        };
        let result = match self.collector.as_mut() {
            Some(Collector::AdaptiveCapture(r)) => r.push(component, mask, q, lower, &mut guard),
            Some(Collector::AdaptiveProof(r)) => r.push(component, mask, q, lower, &mut guard),
            Some(Collector::Rank(r)) => r.push(component, mask, q, lower, &mut guard),
            Some(Collector::Corners(r)) => r.push(component, mask, q, lower, &mut guard),
            None => return Some(()),
        };
        if result.is_none() {
            self.collector = None;
            self.ray = None;
        } else if let Some(RayCollector::Corners(builder)) = self.ray.as_deref_mut() {
            if builder
                .push(
                    problem,
                    plan_component,
                    component,
                    mask,
                    q,
                    constant,
                    lower,
                    &mut guard,
                )
                .is_none()
            {
                self.ray = None;
            }
        }
        if cancelled || !live(end) || require_hull_environment().is_err() {
            return None;
        }
        Some(())
    }
    #[expect(
        clippy::option_option,
        reason = "Outer None refuses the operation; inner None completes without an optional contribution."
    )]
    pub(super) fn finish(
        self,
        checked: &certificate::Checked,
        source_d: f64,
        end: Instant,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<Option<Finished>> {
        let mut cancelled = false;
        let mut guard = || {
            let ok = !cancelled && live(end) && poll();
            cancelled |= !ok;
            ok
        };
        let result = match self.collector {
            Some(Collector::AdaptiveCapture(r)) => r
                .finish(checked.constant, source_d, &checked.g, &mut guard)
                .map(|v| Finished::Cover(Box::new(v))),
            Some(Collector::AdaptiveProof(r)) => r
                .finish(checked.constant, source_d, &checked.g, &mut guard)
                .map(|v| Finished::AdaptiveRoot(Box::new(v))),
            Some(Collector::Rank(r)) => {
                let advice = match self.ray.map(|v| *v) {
                    Some(RayCollector::Capture(capture)) => capture.finish(),
                    _ => None,
                };
                r.finish(checked.constant, source_d, &checked.g, &mut guard)
                    .map(|v| Finished::Axes(Box::new(v), advice))
            }
            Some(Collector::Corners(r)) => {
                let table = match self.ray.map(|v| *v) {
                    Some(RayCollector::Corners(builder)) => builder.finish(),
                    _ => None,
                };
                r.finish(checked.constant, source_d, &mut guard)
                    .map(|v| Finished::Data(Box::new(v), table))
            }
            None => None,
        };
        if cancelled || !live(end) || require_hull_environment().is_err() {
            return None;
        }
        Some(result)
    }
}

/// Called under the original replay's final counter lock, after its last gate.
pub(super) fn publish_axes(
    c: &mut Counts,
    axes: Box<core::Axes>,
    advice: Option<Box<root_ray::Advice>>,
    witness: &Arc<RetainedWords>,
) {
    if let Some(grant) = c.corner.grant.as_mut() {
        if grant.stage == 1 {
            grant.axes = Some(axes);
            if let Some(ray) = grant.root_ray.as_mut() {
                ray.advice = advice;
            }
            grant.witness = Some(witness.clone());
            grant.stage = 2;
        }
    }
}

/// Only called at the same final original replay barrier as publish_axes.
pub(super) fn publish_cover(
    c: &mut Counts,
    cover: Box<adaptive::Cover>,
    witness: &Arc<RetainedWords>,
) {
    if let Some(grant) = c.corner.adaptive.as_mut() {
        if grant.stage == 1 && cover.matches_quote(grant.quote, &grant.bits) {
            grant.cover = Some(cover);
            grant.witness = Some(witness.clone());
            grant.stage = 2;
        }
    }
}

#[derive(Debug)]
pub(super) struct Extension {
    data: Option<Box<core::Data>>,
    seal: Vec<u64>,
    owner: usize,
    root: f64,
    // Read-only telemetry; only the existing sealed final root is authority.
    base_root: f64,
    ray_root: Option<f64>,
    cover_root: Option<f64>,
    cover_leaves: usize,
    // Controls reuse of an already sealed GLOBAL root, never proof authority.
    inherit_root: bool,
}
impl Extension {
    #[expect(
        clippy::option_option,
        reason = "Outer None refuses the operation; inner None completes without an optional contribution."
    )]
    #[cfg(test)]
    pub(super) fn finish(
        data: Box<core::Data>,
        g: &[f64],
        b: &Budget,
        owner: usize,
        end: Instant,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<Option<Box<Self>>> {
        Self::finish_with_root_ray(data, None, g, b, owner, end, poll)
    }
    #[allow(clippy::too_many_arguments)]
    #[expect(
        clippy::option_option,
        reason = "Outer None refuses the operation; inner None completes without an optional contribution."
    )]
    pub(super) fn finish_with_root_ray(
        data: Box<core::Data>,
        alternate: Option<Box<root_ray::Table>>,
        g: &[f64],
        b: &Budget,
        owner: usize,
        end: Instant,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<Option<Box<Self>>> {
        let end = end.min(b.authority);
        let mut cancelled = false;
        let mut guard = || {
            let ok = !cancelled && live(end) && poll();
            cancelled |= !ok;
            ok
        };
        let result = (|| {
            let c = b.state.lock().ok()?;
            let grant = c.corner.grant.as_ref()?;
            if grant.stage != 3
                || data.negative_bits() != grant.bits.as_slice()
                || data.selected().len() != grant.quote.k
                || g.len() > core::MAX_D
            {
                return None;
            }
            let expected = grant.quote.seal_words;
            drop(c);
            let mut coeff = [I::zero(); core::MAX_D];
            for (q, &v) in coeff.iter_mut().zip(g) {
                *q = I::point(v)?;
            }
            let zeros = [0usize; 128];
            let count = data.negative_bits().len();
            let root = data.lower(
                &coeff[..g.len()],
                I::zero(),
                &zeros[..count],
                &zeros[..count],
                &mut guard,
            )?;
            // The alternate table is private, complete, and came through both
            // original replay barriers. Its original offset is never replaced.
            let candidate = alternate.as_ref().and_then(|table| {
                data.replacement_root(&coeff[..g.len()], table.view(), &mut guard)
            });
            let inherit_root = candidate.is_some_and(|value| value > root);
            let base_root = root;
            let root = candidate.map_or(root, |value| root.max(value));
            let mut seal = Vec::with_capacity(expected.checked_add(2)?);
            seal.extend(data.words().take(expected.checked_add(1)?));
            if seal.len() != expected {
                return None;
            }
            seal.extend([owner as u64, root.to_bits()]);
            Some(Box::new(Self {
                data: Some(data),
                seal,
                owner,
                root,
                base_root,
                ray_root: candidate,
                cover_root: None,
                cover_leaves: 0,
                inherit_root,
            }))
        })();
        if !guard() || require_hull_environment().is_err() {
            return None;
        }
        Some(result)
    }
    /// The numerical constructor is private to the complete second Source
    /// replay. An external f64 cannot mint this capability.
    #[expect(
        clippy::boxed_local,
        reason = "Consume the admitted checked-root allocation through its original owner."
    )]
    #[expect(
        clippy::option_option,
        reason = "Outer None refuses the operation; inner None completes without an optional contribution."
    )]
    pub(super) fn finish_adaptive(
        checked: Box<adaptive::CheckedRoot>,
        ordinary_root: f64,
        b: &Budget,
        owner: usize,
        end: Instant,
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<Option<Box<Self>>> {
        let end = end.min(b.authority);
        let mut cancelled = false;
        let mut guard = || {
            let ok = !cancelled && live(end) && poll();
            cancelled |= !ok;
            ok
        };
        let result = (|| {
            if !guard() || require_hull_environment().is_err() {
                return None;
            }
            let c = b.state.lock().ok()?;
            let grant = c.corner.adaptive.as_ref()?;
            if grant.stage != 3 || !checked.matches_quote(grant.quote) {
                return None;
            }
            let cover_root = finite(checked.lower())?;
            let base_root = finite(ordinary_root)?;
            let root = base_root.max(cover_root);
            let cover_leaves = checked.leaves();
            drop(c);
            let seal = vec![ADAPTIVE_SEAL, owner as u64, root.to_bits()];
            Some(Box::new(Self {
                data: None,
                seal,
                owner,
                root,
                base_root,
                ray_root: None,
                cover_root: Some(cover_root),
                cover_leaves,
                inherit_root: true,
            }))
        })();
        if !guard() || require_hull_environment().is_err() {
            return None;
        }
        Some(result)
    }

    pub(super) fn bind<'a>(&'a self, b: &Budget, owner: usize) -> Option<&'a Self> {
        let mut c = b.state.lock().ok()?;
        if self.data.is_none() {
            let grant = c.corner.adaptive.as_mut()?;
            if !std::mem::take(&mut grant.bind)
                || grant.stage != 3
                || owner != self.owner
                || !self.root.is_finite()
                || self.seal.as_slice() != [ADAPTIVE_SEAL, owner as u64, self.root.to_bits()]
            {
                return None;
            }
            return Some(self);
        }
        let data = self.data.as_ref()?;
        let grant = c.corner.grant.as_mut()?;
        if !std::mem::take(&mut grant.bind) {
            return None;
        }
        let expected = grant.quote.seal_words;
        if grant.stage != 3
            || owner != self.owner
            || !self.root.is_finite()
            || self.seal.len() != expected.checked_add(2)?
            || !data
                .words()
                .take(expected.checked_add(1)?)
                .chain([owner as u64, self.root.to_bits()])
                .eq(self.seal.iter().copied())
        {
            return None;
        }
        Some(self)
    }
    pub(super) fn root(&self) -> f64 {
        self.root
    }
    pub(super) fn base_root(&self) -> f64 {
        self.base_root
    }
    pub(super) fn ray_root(&self) -> Option<f64> {
        self.ray_root
    }
    pub(super) fn cover_root(&self) -> Option<f64> {
        self.cover_root
    }
    pub(super) fn cover_leaves(&self) -> usize {
        self.cover_leaves
    }
    pub(super) fn lower(
        &self,
        b: &Budget,
        coeff: &[I],
        beta_bias: I,
        on: &[usize],
        off: &[usize],
        poll: &mut dyn FnMut() -> bool,
    ) -> Option<f64> {
        if self.data.is_none() {
            // A sealed global floor remains valid on every child. This mode
            // performs no new corner traversal and consumes no corner-price credit.
            return (live(b.authority) && poll() && require_hull_environment().is_ok())
                .then_some(self.root);
        }
        {
            let mut c = b.state.lock().ok()?;
            let grant = c.corner.grant.as_mut()?;
            if grant.prices == 0 {
                return None;
            }
            grant.prices -= 1; // Shared, one-shot, no refund after cancellation.
            c.child.corner_price_attempts += 1;
        }
        self.data
            .as_ref()?
            .lower(coeff, beta_bias, on, off, poll)
            .map(|value| {
                if self.inherit_root {
                    value.max(self.root)
                } else {
                    value
                }
            })
    }
}

#[cfg(target_pointer_width = "64")]
const _: () = assert!(
    size_of::<Grant>()
        + size_of::<core::Ranker>()
        + size_of::<core::Collector>()
        + size_of::<core::Axes>()
        + size_of::<core::Data>()
        + size_of::<Extension>()
        + 2 * size_of::<[u64; 2]>()
        + 256 // bounded admission/routing locals, Vec headers and alignment
        <= WRAPPER_BYTES
);

// The kernel separately pays its concrete capture/proof/table headers and
// stack. This cap is only the additional wrapper and scalar attachment.
#[cfg(target_pointer_width = "64")]
const _: () = assert!(
    size_of::<AdaptiveGrant>()
        + size_of::<Extension>()
        + 3 * size_of::<u64>()
        + 512 // bounded attachment/bind locals and ownership wrappers
        <= ADAPTIVE_WRAPPER_BYTES
);
#[cfg(target_pointer_width = "64")]
const _: () =
    assert!(2 * (size_of::<bool>() + 7 + size_of::<usize>() + size_of::<Option<f64>>()) <= 64);

#[cfg(test)]
#[path = "single_hidden_hull_mip_signed_corner_production_tests.rs"]
mod production_tests;
