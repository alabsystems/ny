// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Paid, bounded advice. Neither these scores nor admission observations are
//! source authority; both local certificates are recomputed under one lease.
use super::*;

#[path = "certified_root_two_local_l1/geometry.rs"]
pub(super) mod geometry;

pub(super) const BUNDLE_OBSERVATION_WORK: u64 = 65_536;
const SELECTION_FIXED_WORK: u64 = (1 << 24) + 8192;
const SELECTION_WORK_PER_CELL: u64 = 1024;
// Additional advice only after a preferred Q8-zero refusal; never paid on a
// currently valid split. No additional graph, coefficient or box scans.
pub(super) const WINDOW_RESCUE_WORK: u64 = 1 << 22;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct BundleChoice {
    pub(super) active: [usize; 2],
    pub(super) neurons: [[usize; 8]; 2],
    pub(super) weights: [[f32; 8]; 2],
    pub(super) multipliers: [f32; 2],
}

/// Owner-only retained ranking. Geometry may repartition this same paid pool;
/// it cannot discover more neurons or change the source numerical authority.
#[derive(Clone, Copy)]
pub(super) struct BundleSelection {
    pub(super) choice: BundleChoice,
    pub(super) ranked: [(usize, f64, f64); 16],
    pub(super) count: usize,
}

/// A complete geometry scan may start only if its fresh re-quote and the whole
/// provisional execution envelope fit the SAME remaining owner ledger.
pub(super) fn geometry_work_fits(remaining: u64, advice: u64, execution: u64) -> bool {
    advice
        .checked_add(QUOTE_WORK)
        .and_then(|v| v.checked_add(execution))
        .is_some_and(|needed| needed <= remaining)
}

/// Fixed observations from already-paid work only. Counts are complete only
/// when scan_complete is true; a partial scan never proves insufficient cells.
#[derive(Clone, Copy, Debug)]
pub(super) struct AdmissionObservation {
    pub(super) capability: bool,
    pub(super) stage: &'static str,
    pub(super) reason: &'static str,
    pub(super) scan_started: bool,
    pub(super) scan_complete: bool,
    pub(super) crossing: usize,
    pub(super) negative_crossing: usize,
    pub(super) groups: [usize; 2],
    pub(super) failed_group: Option<usize>,
}
impl Default for AdmissionObservation {
    fn default() -> Self {
        Self {
            capability: false,
            stage: "not-attempted",
            reason: "not-attempted",
            scan_started: false,
            scan_complete: false,
            crossing: 0,
            negative_crossing: 0,
            groups: [0; 2],
            failed_group: None,
        }
    }
}

fn checkpoint(lease: &WorkLease, end: Instant, obs: &mut AdmissionObservation) -> Option<()> {
    if lease.checkpoint().is_err() {
        obs.reason = "cancelled-or-owner-deadline";
        return None;
    }
    if Instant::now() >= end {
        obs.reason = "deadline";
        return None;
    }
    Some(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn bundle_selection(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    bounds: &HashMap<String, BoundedTensor>,
    lease: &mut WorkLease,
    objective: &[f32],
    target: &str,
    end: Instant,
    obs: &mut AdmissionObservation,
) -> Option<BundleSelection> {
    obs.stage = "selection";
    obs.reason = "selection-work";
    lease.reserve_more_work(SELECTION_FIXED_WORK).ok()?;
    checkpoint(lease, end, obs)?;
    obs.reason = "layout-or-source";
    // Simultaneous ranking, both normalized choices/return places, per-group
    // sort temporaries and bounded observation state fit the existing8KiB.
    let bytes = size_of::<[(usize, f64, f64); 16]>()
        .checked_add(2 * size_of::<BundleChoice>())?
        .checked_add(2 * size_of::<[(usize, f64, f64); 8]>())?
        .checked_add(size_of::<AdmissionObservation>())?
        .checked_add(2048)?;
    if bytes > FAMILY_SCRATCH
        || target.len() > 1024
        || graph.output_name().len() > 1024
        || objective.is_empty()
        || objective.len() > 8192
    {
        return None;
    }
    let relu = graph.node(target)?;
    if !matches!(relu.layer(), Layer::ReLU(_)) {
        return None;
    }
    let parent = relu.require_unary_input().ok()?;
    if parent.len() > 1024 {
        return None;
    }
    let pre = if parent == crate::network::NETWORK_INPUT {
        input
    } else {
        bounds.get(parent)?
    };
    let width = pre.len();
    if width == 0 || width > 8192 || pre.shape().len() > 4 {
        return None;
    }
    let last = graph.node(graph.output_name())?;
    let Layer::Linear(linear) = last.layer() else {
        return None;
    };
    if linear.in_features() != width
        || linear.out_features() != objective.len()
        || width.checked_mul(objective.len())? > (1 << 20)
    {
        return None;
    }
    let mut current = last.require_unary_input().ok()?;
    for _ in 0..MAX_NODES {
        checkpoint(lease, end, obs)?;
        obs.reason = "layout-or-source";
        if current.len() > 1024 {
            return None;
        }
        if current == target {
            break;
        }
        let shape = graph.node(current)?;
        if !matches!(shape.layer(), Layer::Flatten(_) | Layer::Reshape(_)) {
            return None;
        }
        let parent = shape.require_unary_input().ok()?;
        for name in [current, parent] {
            if name.len() > 1024 {
                return None;
            }
            let b = bounds.get(name)?;
            if b.shape().len() > 4
                || b.len() != width
                || b.lower().as_slice().is_none()
                || b.upper().as_slice().is_none()
            {
                return None;
            }
        }
        current = parent;
    }
    if current != target {
        return None;
    }
    let lower = pre.lower().as_slice()?;
    let upper = pre.upper().as_slice()?;
    let weights = linear.weight.as_slice()?;
    if lower.len() != width
        || upper.len() != width
        || weights.len() != width.checked_mul(objective.len())?
    {
        return None;
    }
    let scan = SELECTION_WORK_PER_CELL
        .checked_mul(u64::try_from(width).ok()?)?
        .checked_mul(u64::try_from(objective.len()).ok()?.checked_add(1)?)?;
    obs.reason = "selection-work";
    lease.reserve_more_work(scan).ok()?;
    obs.scan_started = true;
    obs.stage = "coefficient-scan";
    let mut best = [(0usize, 0f64, 0f64); 16];
    let mut count = 0;
    for i in 0..width {
        if i % 128 == 0 {
            checkpoint(lease, end, obs)?;
        }
        obs.reason = "source-values";
        let (lo, hi) = (lower[i], upper[i]);
        if !ny_core::is_crown_coeff_safe(lo) || !ny_core::is_crown_coeff_safe(hi) || lo > hi {
            return None;
        }
        if lo >= 0. || hi <= 0. {
            continue;
        }
        obs.crossing += 1;
        let mut coefficient = 0f64;
        for (row, &c) in objective.iter().enumerate() {
            if row % 128 == 0 {
                checkpoint(lease, end, obs)?;
            }
            obs.reason = "source-values";
            let w = weights[row * width + i];
            if !ny_core::is_crown_coeff_safe(c) || !ny_core::is_crown_coeff_safe(w) {
                return None;
            }
            coefficient += f64::from(c) * f64::from(w);
        }
        if !coefficient.is_finite() {
            return None;
        }
        if coefficient >= 0. {
            continue;
        }
        obs.negative_crossing += 1;
        let gap = -f64::from(lo) * f64::from(hi) / (f64::from(hi) - f64::from(lo));
        let score = -coefficient * gap;
        if !score.is_finite() || score <= 0. {
            return None;
        }
        // Same tie rule and score operations as current top8; keeping16 entries
        // leaves its first8 unchanged. Split BEFORE either group's index sort.
        let position = (0..count).find(|&p| score > best[p].1).unwrap_or(count);
        if position < 16 {
            for p in ((position + 1)..=count.min(15)).rev() {
                best[p] = best[p - 1];
            }
            best[position] = (i, score, -coefficient);
            count = (count + 1).min(16);
        }
    }
    checkpoint(lease, end, obs)?;
    obs.scan_complete = true;
    obs.groups = [count.min(8), count.saturating_sub(8)];
    obs.reason = "insufficient-negative-crossings";
    let groups = preferred_groups(count)?;
    if count <= 8 {
        // Retain the established single-local route whenever one common Q8
        // scale represents every selected coefficient. A rounded-zero weight
        // is a domain-coverage refusal, not evidence that the source relation
        // is useless: two independently normalized groups can retain it.
        obs.stage = "single-scale-check";
        obs.reason = "q8-or-multiplier";
        match normalize_group(&best[..count]) {
            Ok(_) => {
                obs.reason = "single-local-representable";
                return None;
            }
            Err(NormalizationFailure::Zero) => {}
            Err(NormalizationFailure::Other) => return None,
        }
    }
    obs.groups = groups;
    obs.stage = "source-layout";
    obs.reason = "layout-or-source";
    let node = graph.node(parent)?;
    let selected_linear = match node.layer() {
        Layer::Linear(l) => l,
        Layer::BatchNorm(_) => {
            let name = node.require_unary_input().ok()?;
            if name.len() > 1024 {
                return None;
            }
            let Layer::Linear(l) = graph.node(name)?.layer() else {
                return None;
            };
            l
        }
        _ => return None,
    };
    if selected_linear.in_features() == 0
        || selected_linear.in_features() > 65536
        || selected_linear.out_features() > 8192
    {
        return None;
    }
    obs.stage = "normalization";
    obs.reason = "q8-or-multiplier";
    let mut result = empty_choice();
    let mut first_zero = None;
    // Check both preferred groups before rescue, so a Q8 zero cannot conceal a
    // separate invalid multiplier or arithmetic failure in the other group.
    for group in 0..2 {
        obs.failed_group = Some(group);
        let start = if group == 0 { 0 } else { groups[0] };
        let len = groups[group];
        match normalize_group(&best[start..start + len]) {
            Ok(normalized) => put_group(&mut result, group, normalized),
            Err(NormalizationFailure::Zero) => {
                first_zero.get_or_insert(group);
            }
            Err(NormalizationFailure::Other) => return None,
        }
    }
    if let Some(group) = first_zero {
        obs.failed_group = Some(group);
        result = rescue_windows(&best, count, lease, end, obs)?;
    }
    obs.groups = result.active;
    obs.failed_group = None;
    checkpoint(lease, end, obs)?;
    obs.stage = "selected";
    obs.reason = "none";
    Some(BundleSelection {
        choice: result,
        ranked: best,
        count,
    })
}

/// Preliminary ranking/normalization seam retained for existing controls.
/// Root ownership consumes bundle_selection and pays any geometry separately.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(super) fn bundle_target(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    bounds: &HashMap<String, BoundedTensor>,
    lease: &mut WorkLease,
    objective: &[f32],
    target: &str,
    end: Instant,
    obs: &mut AdmissionObservation,
) -> Option<BundleChoice> {
    bundle_selection(graph, input, bounds, lease, objective, target, end, obs)
        .map(|selection| selection.choice)
}

/// Six through eight are considered only after a common-scale Q8-zero
/// refusal. Nine and ten use6+3 and7+3; old11..16 splits stay8+(count-8).
fn preferred_groups(count: usize) -> Option<[usize; 2]> {
    if !(6..=16).contains(&count) {
        return None;
    }
    let first = 8.min(count - 3);
    Some([first, count - first])
}

#[derive(Clone, Copy)]
struct NormalizedGroup {
    active: usize,
    neurons: [usize; 8],
    weights: [f32; 8],
    multiplier: f32,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NormalizationFailure {
    Zero,
    Other,
}
fn empty_choice() -> BundleChoice {
    BundleChoice {
        active: [0; 2],
        neurons: [[0; 8]; 2],
        weights: [[0.; 8]; 2],
        multipliers: [0.; 2],
    }
}
fn put_group(choice: &mut BundleChoice, group: usize, value: NormalizedGroup) {
    choice.active[group] = value.active;
    choice.neurons[group] = value.neurons;
    choice.weights[group] = value.weights;
    choice.multipliers[group] = value.multiplier;
}
fn normalize_group(items: &[(usize, f64, f64)]) -> Result<NormalizedGroup, NormalizationFailure> {
    if !(1..=8).contains(&items.len()) {
        return Err(NormalizationFailure::Other);
    }
    let magnitude = items.iter().map(|v| v.2).fold(0f64, f64::max);
    let multiplier = (0.5 * magnitude) as f32;
    if !ny_core::is_crown_coeff_safe(multiplier) || multiplier <= 0. {
        return Err(NormalizationFailure::Other);
    }
    let mut sorted = [(0usize, 0f64, 0f64); 8];
    sorted[..items.len()].copy_from_slice(items);
    sorted[..items.len()].sort_unstable_by_key(|v| v.0);
    let mut result = NormalizedGroup {
        active: items.len(),
        neurons: [0; 8],
        weights: [0.; 8],
        multiplier,
    };
    let mut zero = false;
    for (position, item) in sorted[..items.len()].iter().enumerate() {
        let quantized = (256. * item.2 / magnitude).round();
        if !quantized.is_finite() || !(0. ..=256.).contains(&quantized) {
            return Err(NormalizationFailure::Other);
        }
        zero |= quantized == 0.;
        result.neurons[position] = item.0;
        result.weights[position] = (quantized as f32) / 256.;
    }
    if zero {
        Err(NormalizationFailure::Zero)
    } else {
        Ok(result)
    }
}

/// All simultaneous caller/helper places, including the saved ranking array,
/// magnitude copy, both normalized groups/returns and candidate/best choices.
/// The existing 2048-byte slack remains; no new lease or stack allowance.
pub(super) fn rescue_storage_bytes() -> Option<usize> {
    (2 * size_of::<[(usize, f64, f64); 16]>())
        .checked_add(2 * size_of::<[(usize, f64, f64); 8]>())?
        .checked_add(6 * size_of::<BundleChoice>())?
        .checked_add(4 * size_of::<NormalizedGroup>())?
        .checked_add(2 * size_of::<AdmissionObservation>())?
        // The small-pool helper coexists with the caller's bounded places.
        .checked_add(2 * size_of::<BundleChoice>())?
        .checked_add(4 * size_of::<NormalizedGroup>())?
        .checked_add(128)?
        .checked_add(2048)
}

/// Two ordered disjoint nonempty magnitude windows. There are C(n+2,4)
/// pairs, at most 210 for n<=8; considering both labels gives 420 candidates,
/// below the established 621-pair rescue ceiling. No extra source scan.
fn rescue_small_windows(
    sorted: &[(usize, f64, f64); 16],
    count: usize,
    lease: &WorkLease,
    end: Instant,
    obs: &mut AdmissionObservation,
) -> Option<BundleChoice> {
    let mut best = None;
    let mut best_score = 0.;
    for first in 0..count {
        for first_end in first + 1..=count {
            checkpoint(lease, end, obs)?;
            let Ok(a) = normalize_group(&sorted[first..first_end]) else {
                continue;
            };
            for second in first_end..count {
                for second_end in second + 1..=count {
                    checkpoint(lease, end, obs)?;
                    let Ok(b) = normalize_group(&sorted[second..second_end]) else {
                        continue;
                    };
                    let score = sorted[first..first_end]
                        .iter()
                        .chain(sorted[second..second_end].iter())
                        .map(|v| v.1)
                        .sum::<f64>();
                    if !score.is_finite() || score <= 0. {
                        obs.reason = "window-score";
                        return None;
                    }
                    for reverse in [false, true] {
                        let mut candidate = empty_choice();
                        put_group(&mut candidate, 0, if reverse { b } else { a });
                        put_group(&mut candidate, 1, if reverse { a } else { b });
                        if best.as_ref().is_none_or(|old: &BundleChoice| {
                            score > best_score
                                || (score == best_score
                                    && (candidate.active, candidate.neurons)
                                        < (old.active, old.neurons))
                        }) {
                            best = Some(candidate);
                            best_score = score;
                        }
                    }
                }
            }
        }
    }
    checkpoint(lease, end, obs)?;
    best
}

/// Fixed finite advice family, not geometry/proof authority. Group 0 retains
/// the preferred length 6, 7 or 8 for the established 9..16 pool. Smaller pools
/// use all nonempty disjoint magnitude windows, permitting pairs/singletons
/// and leaving incompatible rows in the complete original objective.
/// Windows are disjoint before canonical index sorting. Fresh private certificates still
/// bind the actual selected indices, Q8 weights and multipliers after entry.
pub(super) fn rescue_windows(
    ranked: &[(usize, f64, f64); 16],
    count: usize,
    lease: &mut WorkLease,
    end: Instant,
    obs: &mut AdmissionObservation,
) -> Option<BundleChoice> {
    obs.stage = "window-rescue";
    obs.reason = "window-work";
    lease.reserve_more_work(WINDOW_RESCUE_WORK).ok()?;
    checkpoint(lease, end, obs)?;
    obs.reason = "window-storage";
    let preferred_len = preferred_groups(count)?[0];
    if rescue_storage_bytes()? > FAMILY_SCRATCH {
        return None;
    }
    let mut sorted = *ranked;
    sorted[..count].sort_unstable_by(|a, b| a.2.total_cmp(&b.2).then(a.0.cmp(&b.0)));
    if count <= 8 {
        obs.reason = "window-no-compatible-groups";
        return rescue_small_windows(&sorted, count, lease, end, obs);
    }
    let mut best = None;
    let mut best_score = 0f64;
    obs.reason = "window-no-compatible-groups";
    // At most 9*69=621 range pairs; the smaller pools stay below this ceiling.
    // No window list, source-sized buffer, E scan or extra backend call.
    for first_len in preferred_len..=preferred_len {
        for first in 0..=count - first_len {
            checkpoint(lease, end, obs)?;
            let Ok(a) = normalize_group(&sorted[first..first + first_len]) else {
                continue;
            };
            for len in 3..=8.min(count) {
                for second in 0..=count - len {
                    checkpoint(lease, end, obs)?;
                    if first < second + len && second < first + first_len {
                        continue;
                    }
                    let Ok(b) = normalize_group(&sorted[second..second + len]) else {
                        continue;
                    };
                    let mut candidate = empty_choice();
                    put_group(&mut candidate, 0, a);
                    put_group(&mut candidate, 1, b);
                    let score = sorted[first..first + first_len]
                        .iter()
                        .chain(sorted[second..second + len].iter())
                        .map(|v| v.1)
                        .sum::<f64>();
                    if !score.is_finite() || score <= 0. {
                        obs.reason = "window-score";
                        return None;
                    }
                    let improves = best.as_ref().is_none_or(|old: &BundleChoice| {
                        score > best_score
                            || (score == best_score
                                && (candidate.active, candidate.neurons)
                                    < (old.active, old.neurons))
                    });
                    if improves {
                        best = Some(candidate);
                        best_score = score;
                    }
                }
            }
        }
    }
    checkpoint(lease, end, obs)?;
    best
}
