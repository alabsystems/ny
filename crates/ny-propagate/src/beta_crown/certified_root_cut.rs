// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! One original-query attempt for a same-source, lower-only root cut.
//!
//! The public owner carries resource custody, never mathematical authority.
//! Only the private raw-root evaluator can construct a completed result. No
//! shadow scalar, child domain, exported certificate, or folded-BN graph enters
//! this optional path. Cloning a verifier clones this owner, not its allowance.

use crate::beta_crown::config::VerificationArtifactAuthority;
use crate::bounds::GraphAlphaState;
use crate::certify::work::{WorkBudget, WorkLease, WorkRequest};
use crate::layers::Layer;
use crate::network::crown_memory::{
    cpu_crown_dense_budget_bytes, process_memory_envelope, ProcessMemoryEnvelope,
};
use crate::network::raw_root_cut::{
    evaluate_raw_local_l1_root_cut, quote_raw_local_l1_root_cut, CompletedRawLocalL1RootCut,
    RawLocalL1RootCutQuote,
};
use crate::network::raw_root_cut::{
    evaluate_raw_triplet_root_cut, quote_raw_triplet_root_cut, CompletedRawTripletRootCut,
    RawTripletRootCutQuote,
};
use crate::network::raw_root_cut::{
    evaluate_raw_two_local_l1_root_cut, quote_raw_two_local_l1_root_cut,
    CompletedRawTwoLocalL1RootCut, RawTwoLocalL1RootCutQuote,
};
use crate::network::raw_root_cut::{
    evaluate_raw_two_pair_root_cut, quote_raw_two_pair_root_cut, CompletedRawRootCut,
    CompletedRawTwoPairRootCut, RawRootCutQuote, RawTwoPairRootCutQuote,
};
use crate::network::{evaluate_raw_root_cut, quote_raw_root_cut, RawRootCutLimits};
use crate::{BetaCrownConfig, GraphNetwork};
use ny_core::GemmEngine;
use ny_tensor::BoundedTensor;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[path = "certified_root_spatial.rs"]
mod spatial;
pub(crate) use spatial::Advice as SpatialAdvice;

#[cfg(test)]
#[path = "certified_root_two_pair_tests.rs"]
mod two_pair_tests;

#[path = "certified_root_finite_family.rs"]
mod finite_family;
use finite_family::{family_target, FamilyChoice, FAMILY_SCRATCH};
#[cfg(test)]
#[path = "certified_root_finite_family_tests.rs"]
mod finite_family_tests;

#[path = "certified_root_local_l1.rs"]
mod local_l1;
use local_l1::local_target;
#[cfg(test)]
#[path = "certified_root_local_l1_tests.rs"]
mod local_l1_tests;

#[path = "certified_root_two_local_l1.rs"]
mod two_local_l1;
#[cfg(test)]
use two_local_l1::bundle_target;
use two_local_l1::{AdmissionObservation, BUNDLE_OBSERVATION_WORK};
#[cfg(test)]
#[path = "certified_root_two_local_l1_tests.rs"]
mod two_local_l1_tests;

const MAX_SCRATCH: usize = 2 << 30;
const MAX_WORK: u64 = 16_000_000_000;
const QUOTE_SCRATCH: usize = 64 << 10;
const QUOTE_WORK: u64 = 2_000_000;
const MAX_NODES: usize = 128;
const MAX_DIM: usize = 1 << 20;
const MAX_ROWS: usize = 4096;
const MAX_SLICE: Duration = Duration::from_secs(2);
const DOWNSTREAM_RESERVE: Duration = Duration::from_secs(1);
// Paid before any new graph/name/shape lookup. Bounds are 128 nodes, 1024-byte
// names and rank4. This includes refusal paths and a fixed top-four array.
const JOINT_SELECTION_FIXED_WORK: u64 = (1 << 24) + 8192;
const JOINT_SELECTION_WORK_PER_CELL: u64 = 256;

// Scheduling opportunity only. This bit can never authenticate a source,
// initialize a backend, or replace its live checked capability predicates.
static CHECKED_FACTORY_REGISTERED: AtomicBool = AtomicBool::new(false);

/// Startup may call this only when registering the reviewed checked CUDA
/// factory. It changes no engine slot and does not initialize that factory.
pub fn note_checked_root_cut_factory_registered() {
    CHECKED_FACTORY_REGISTERED.store(true, Ordering::Release);
}

fn checked_capabilities(engine: &dyn GemmEngine) -> bool {
    engine
        .as_gpu_crown_backward()
        .is_some_and(|gpu| gpu.provides_bounded_support4() && gpu.provides_bounded_lower_cut())
}

fn joint_capabilities(engine: &dyn GemmEngine) -> bool {
    engine.as_gpu_crown_backward().is_some_and(|gpu| {
        gpu.provides_bounded_support4() && gpu.provides_bounded_two_pair_lower_cut()
    })
}

fn triplet_capabilities(engine: &dyn GemmEngine) -> bool {
    engine
        .as_gpu_crown_backward()
        .is_some_and(|g| g.provides_bounded_support4() && g.provides_bounded_triplet_lower_cut())
}
fn local_l1_capabilities(engine: &dyn GemmEngine) -> bool {
    engine
        .as_gpu_crown_backward()
        .is_some_and(|g| g.provides_bounded_local_l1_lower_cut())
}
fn two_local_l1_capabilities(engine: &dyn GemmEngine) -> bool {
    engine
        .as_gpu_crown_backward()
        .is_some_and(|g| g.provides_bounded_two_local_l1_lower_cut())
}
fn raw_load_opportunity(local: Option<&dyn GemmEngine>, registered: bool, cpu_only: bool) -> bool {
    !cpu_only
        && (registered
            || local.is_some_and(|e| {
                checked_capabilities(e)
                    || joint_capabilities(e)
                    || triplet_capabilities(e)
                    || local_l1_capabilities(e)
                    || two_local_l1_capabilities(e)
            }))
}

/// Freeze this opportunity once before model loading. Cold registered CUDA
/// remains eligible, but CPU/WGPU-only processes without a complete checked method
/// retain their historical automatic load policy. Numerical entry later uses
/// only an already-published engine; this scheduling bit grants no authority.
pub fn checked_backend_possible_for_raw_load(local: Option<&dyn GemmEngine>) -> bool {
    raw_load_opportunity(
        local,
        CHECKED_FACTORY_REGISTERED.load(Ordering::Acquire),
        crate::sound_f64_gemm::cpu_only_f64_active(),
    )
}

#[cfg(test)]
fn select_checked_engine<'a>(
    local: Option<&'a dyn GemmEngine>,
    cpu_only: bool,
    ready: impl FnOnce() -> [Option<&'a dyn GemmEngine>; 3],
) -> Option<&'a dyn GemmEngine> {
    if cpu_only {
        return None;
    }
    if local.is_some_and(checked_capabilities) {
        return local;
    }
    ready()
        .into_iter()
        .flatten()
        .find(|engine| checked_capabilities(*engine))
}

#[derive(Clone, Copy)]
struct ReadyMethods<'a> {
    two_local_l1: Option<&'a dyn GemmEngine>,
    single: Option<&'a dyn GemmEngine>,
    joint: Option<&'a dyn GemmEngine>,
    triplet: Option<&'a dyn GemmEngine>,
    local_l1: Option<&'a dyn GemmEngine>,
}

fn select_ready_methods<'a>(
    local: Option<&'a dyn GemmEngine>,
    cpu_only: bool,
    ready: impl FnOnce() -> [Option<&'a dyn GemmEngine>; 3],
) -> Option<ReadyMethods<'a>> {
    if cpu_only {
        return None;
    }
    let mut methods = ReadyMethods {
        two_local_l1: local.filter(|e| two_local_l1_capabilities(*e)),
        single: local.filter(|e| checked_capabilities(*e)),
        joint: local.filter(|e| joint_capabilities(*e)),
        triplet: local.filter(|e| triplet_capabilities(*e)),
        local_l1: local.filter(|e| local_l1_capabilities(*e)),
    };
    // A local old-only backend must not conceal a new-capable ready owner.
    // Only a local owner with all enabled capabilities avoids ready-slot lookup.
    if methods.single.is_none()
        || methods.joint.is_none()
        || methods.triplet.is_none()
        || methods.local_l1.is_none()
        || methods.two_local_l1.is_none()
    {
        for engine in ready().into_iter().flatten() {
            if methods.two_local_l1.is_none() && two_local_l1_capabilities(engine) {
                methods.two_local_l1 = Some(engine);
            }
            if methods.local_l1.is_none() && local_l1_capabilities(engine) {
                methods.local_l1 = Some(engine);
            }
            if methods.triplet.is_none() && triplet_capabilities(engine) {
                methods.triplet = Some(engine);
            }
            if methods.joint.is_none() && joint_capabilities(engine) {
                methods.joint = Some(engine);
            }
            if methods.single.is_none() && checked_capabilities(engine) {
                methods.single = Some(engine);
            }
        }
    }
    (methods.single.is_some()
        || methods.joint.is_some()
        || methods.triplet.is_some()
        || methods.local_l1.is_some()
        || methods.two_local_l1.is_some())
    .then_some(methods)
}

/// Runtime-only participating owner. Construction does not authenticate a
/// model, promise a proof, reserve the entire process, or permit more time.
#[derive(Debug)]
pub struct QueryOwner {
    enabled: bool,
    attempted: AtomicBool,
    deadline: Instant,
    budget: WorkBudget,
}

impl QueryOwner {
    /// Scheduling only: an original raw root may reserve work within the same
    /// query clock. This neither claims the cut nor grants numerical authority.
    pub(crate) fn bootstrap_deadline(&self) -> Option<Instant> {
        (self.enabled && !self.attempted.load(Ordering::Acquire)).then_some(self.deadline)
    }

    /// Called once at the instance router, before verifier/config cloning.
    /// `None` is an unbounded/missing clock and refuses this optional method.
    pub fn new(deadline: Option<Instant>, enabled: bool) -> Option<Arc<Self>> {
        let deadline = deadline?;
        let budget =
            WorkBudget::new(Some(deadline), MAX_SCRATCH, 0, MAX_WORK, 1, Duration::ZERO).ok()?;
        Some(Arc::new(Self {
            enabled,
            attempted: AtomicBool::new(false),
            deadline,
            budget,
        }))
    }

    fn claim(&self, root_deadline: Instant) -> Option<Instant> {
        if !self.enabled
            || self
                .attempted
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
        {
            return None;
        }
        // A refused/expired attempt remains spent. No child or later graph
        // clone can retry with a fresh fraction of the same query clock.
        let now = Instant::now();
        let end = self.deadline.min(root_deadline);
        let remaining = end.checked_duration_since(now)?;
        let candidate_end = end.checked_sub(DOWNSTREAM_RESERVE)?;
        let available = candidate_end.checked_duration_since(now)?;
        let slice = (remaining / 8).min(MAX_SLICE).min(available);
        if slice.is_zero() {
            return None;
        }
        now.checked_add(slice).map(|t| t.min(candidate_end))
    }
}

fn memory_cap() -> Option<usize> {
    let headroom = match process_memory_envelope() {
        ProcessMemoryEnvelope::Bounded { headroom_bytes } => {
            usize::try_from(headroom_bytes / 4).ok()?
        }
        ProcessMemoryEnvelope::Unbounded => MAX_SCRATCH,
        ProcessMemoryEnvelope::Unavailable => return None,
    };
    let cap = MAX_SCRATCH
        .min(cpu_crown_dense_budget_bytes())
        .min(headroom);
    (cap > QUOTE_SCRATCH).then_some(cap - QUOTE_SCRATCH)
}

/// First unresolved row, then the deepest eligible ReLU and its first two
/// crossing coordinates, ranked by negative incoming objective coefficients
/// when a final Linear supplies them. This is bounded advice, not a model or
/// category allowlist. The evaluator authenticates every source and slope.
fn target<'a>(
    graph: &'a GraphNetwork,
    input: &'a BoundedTensor,
    bounds: &'a HashMap<String, BoundedTensor>,
    lease: &mut WorkLease,
    objective: &[f32],
    end: Instant,
) -> Option<(&'a str, [usize; 2], f32)> {
    let mut current = graph.output_name();
    if current.is_empty() || current.len() > 1024 {
        return None;
    }
    for _ in 0..MAX_NODES {
        lease.checkpoint().ok()?;
        if Instant::now() >= end {
            return None;
        }
        if current.len() > 1024 {
            return None;
        }
        let node = graph.node(current)?;
        let [parent] = node.inputs() else {
            return None;
        };
        if parent.len() > 1024 {
            return None;
        }
        if matches!(node.layer(), Layer::ReLU(_)) {
            let pre = if parent == crate::network::NETWORK_INPUT {
                input
            } else {
                bounds.get(parent)?
            };
            if pre.is_empty() || pre.len() > MAX_DIM || pre.shape().len() > 4 {
                return None;
            }
            lease
                .reserve_more_work(u64::try_from(pre.len()).ok()?.checked_mul(8)?)
                .ok()?;
            // Only direct final-Linear coefficients are proposed here. No
            // backwards relaxed coefficient is mistaken for proof authority.
            let last = graph.node(graph.output_name())?;
            let direct = match last.layer() {
                Layer::Linear(linear)
                    if last.inputs().len() == 1
                        && last.inputs()[0] == current
                        && linear.in_features() == pre.len()
                        && linear.out_features() == objective.len()
                        && objective.len().checked_mul(pre.len())? <= (1 << 20) =>
                {
                    let weights = linear.weight.as_slice()?;
                    if weights.len() != objective.len().checked_mul(pre.len())? {
                        return None;
                    }
                    lease
                        .reserve_more_work(u64::try_from(weights.len()).ok()?.checked_mul(4)?)
                        .ok()?;
                    Some(weights)
                }
                _ => None,
            };
            let mut pair = [0usize; 2];
            let mut found = 0;
            let mut best_pair = [0usize; 2];
            let mut best_a = [0.0f64; 2];
            let mut negative = 0usize;
            for (i, (&lo, &hi)) in pre.lower().iter().zip(pre.upper().iter()).enumerate() {
                if i % 4096 == 0 {
                    lease.checkpoint().ok()?;
                    if Instant::now() >= end {
                        return None;
                    }
                }
                if !ny_core::is_crown_coeff_safe(lo) || !ny_core::is_crown_coeff_safe(hi) || lo > hi
                {
                    return None;
                }
                if lo < 0.0 && hi > 0.0 {
                    if found < 2 {
                        pair[found] = i;
                        found += 1;
                    }
                    if let Some(weights) = direct {
                        let mut a = 0.0f64;
                        for (r, &c) in objective.iter().enumerate() {
                            let w = weights[r * pre.len() + i];
                            if !c.is_finite() || !w.is_finite() {
                                return None;
                            }
                            a += f64::from(c) * f64::from(w);
                        }
                        if !a.is_finite() {
                            return None;
                        }
                        if a < 0.0 {
                            if negative == 0 || a < best_a[0] {
                                if negative > 0 {
                                    best_a[1] = best_a[0];
                                    best_pair[1] = best_pair[0];
                                }
                                best_a[0] = a;
                                best_pair[0] = i;
                            } else if negative == 1 || a < best_a[1] {
                                best_a[1] = a;
                                best_pair[1] = i;
                            }
                            negative = (negative + 1).min(2);
                        }
                    }
                }
            }
            if found == 2 {
                if negative == 2 {
                    let scale = (0.5 * (-best_a[0]).min(-best_a[1])) as f32;
                    if !scale.is_finite() || scale <= 0.0 {
                        return None;
                    }
                    best_pair.sort_unstable();
                    return Some((current, best_pair, scale));
                }
                return Some((current, pair, 1.0));
            }
        }
        if parent == crate::network::NETWORK_INPUT {
            return None;
        }
        current = parent;
    }
    None
}

/// Optional advice for the same deepest ReLU selected by the old policy.
/// Four negative incoming coefficients are ranked by magnitude times the
/// triangle gap -l*u/(u-l), with index order breaking ties. All arithmetic is
/// advice; the same-source evaluator re-proves the actual stored facets.
fn ranked_target(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    bounds: &HashMap<String, BoundedTensor>,
    lease: &mut WorkLease,
    objective: &[f32],
    target: &str,
    end: Instant,
) -> Option<([(usize, f64, f64); 4], usize)> {
    lease.reserve_more_work(JOINT_SELECTION_FIXED_WORK).ok()?;
    lease.checkpoint().ok()?;
    if Instant::now() >= end
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
    if !(3..=MAX_DIM).contains(&width) || pre.shape().len() > 4 {
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
        lease.checkpoint().ok()?;
        if Instant::now() >= end || current.len() > 1024 {
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
    // Pay the full D*O scan even when only crossing cells use those weights.
    // The +1 column covers every bound/gap/rank check, including stable cells.
    let scan = JOINT_SELECTION_WORK_PER_CELL
        .checked_mul(u64::try_from(width).ok()?)?
        .checked_mul(u64::try_from(objective.len()).ok()?.checked_add(1)?)?;
    lease.reserve_more_work(scan).ok()?;
    let mut best = [(0usize, 0f64, 0f64); 4]; // index, score, negative magnitude
    let mut count = 0;
    for i in 0..width {
        if i % 128 == 0 {
            lease.checkpoint().ok()?;
            if Instant::now() >= end {
                return None;
            }
        }
        let (lo, hi) = (lower[i], upper[i]);
        if !ny_core::is_crown_coeff_safe(lo) || !ny_core::is_crown_coeff_safe(hi) || lo > hi {
            return None;
        }
        if lo >= 0. || hi <= 0. {
            continue;
        }
        let mut coefficient = 0f64;
        for (row, &c) in objective.iter().enumerate() {
            if row % 128 == 0 {
                lease.checkpoint().ok()?;
                if Instant::now() >= end {
                    return None;
                }
            }
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
        let gap = -f64::from(lo) * f64::from(hi) / (f64::from(hi) - f64::from(lo));
        let score = -coefficient * gap;
        if !score.is_finite() || score <= 0. {
            return None;
        }
        // Stable insertion order breaks equal scores by increasing coordinate.
        let position = (0..count).find(|&p| score > best[p].1).unwrap_or(count);
        if position < 4 {
            for p in ((position + 1)..=count.min(3)).rev() {
                best[p] = best[p - 1];
            }
            best[position] = (i, score, -coefficient);
            count = (count + 1).min(4);
        }
    }
    lease.checkpoint().ok()?;
    (count >= 3 && Instant::now() < end).then_some((best, count))
}
fn joint_target(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    bounds: &HashMap<String, BoundedTensor>,
    lease: &mut WorkLease,
    objective: &[f32],
    target: &str,
    end: Instant,
) -> Option<([[usize; 2]; 2], [f32; 2])> {
    let (best, count) = ranked_target(graph, input, bounds, lease, objective, target, end)?;
    if count != 4 {
        return None;
    }
    let mut pairs = [[best[0].0, best[1].0], [best[2].0, best[3].0]];
    let mut scales = [0f32; 2];
    for n in 0..2 {
        scales[n] = (0.5 * best[2 * n].2.min(best[2 * n + 1].2)) as f32;
        if !ny_core::is_crown_coeff_safe(scales[n]) || scales[n] <= 0. {
            return None;
        }
        pairs[n].sort_unstable();
    }
    // Freeze pair order before constructing any support seed.
    if pairs[1] < pairs[0] {
        pairs.swap(0, 1);
        scales.swap(0, 1);
    }
    lease.checkpoint().ok()?;
    (Instant::now() < end).then_some((pairs, scales))
}

#[expect(
    clippy::large_enum_variant,
    reason = "Keep inline ownership and accounted storage; boxing adds an allocation"
)]
enum RootProposal<'a, 'e> {
    TwoLocalL1(RawTwoLocalL1RootCutQuote<'a>, &'e dyn GemmEngine),
    LocalL1(RawLocalL1RootCutQuote<'a>, &'e dyn GemmEngine),
    Single(RawRootCutQuote<'a>, &'e dyn GemmEngine),
    Joint(RawTwoPairRootCutQuote<'a>, &'e dyn GemmEngine),
    Triplet(RawTripletRootCutQuote<'a>, &'e dyn GemmEngine),
}
impl RootProposal<'_, '_> {
    fn work(&self) -> u64 {
        match self {
            Self::TwoLocalL1(q, _) => q.work(),
            Self::LocalL1(q, _) => q.work(),
            Self::Single(q, _) => q.work(),
            Self::Joint(q, _) => q.work(),
            Self::Triplet(q, _) => q.work(),
        }
    }
    fn scratch(&self) -> usize {
        match self {
            Self::TwoLocalL1(q, _) => q.scratch_bytes(),
            Self::LocalL1(q, _) => q.scratch_bytes(),
            Self::Single(q, _) => q.scratch_bytes(),
            Self::Joint(q, _) => q.scratch_bytes(),
            Self::Triplet(q, _) => q.scratch_bytes(),
        }
    }
    fn candidates(&self) -> usize {
        match self {
            Self::TwoLocalL1(q, _) => q.candidate_count(),
            Self::LocalL1(q, _) => q.candidate_count(),
            Self::Single(q, _) => q.candidate_count(),
            Self::Joint(q, _) => q.candidate_count(),
            Self::Triplet(q, _) => q.candidate_count(),
        }
    }
    fn method(&self) -> &'static str {
        match self {
            Self::TwoLocalL1(..) => "two-local-l1",
            Self::LocalL1(..) => "local-l1",
            Self::Single(..) => "single-pair",
            Self::Joint(..) => "two-pair",
            Self::Triplet(..) => "triplet",
        }
    }
    fn evaluate(self, poll: &mut dyn FnMut() -> bool) -> Option<RootCompletion> {
        match self {
            Self::TwoLocalL1(q, e) => {
                evaluate_raw_two_local_l1_root_cut(q, e, poll).map(RootCompletion::TwoLocalL1)
            }
            Self::LocalL1(q, e) => {
                evaluate_raw_local_l1_root_cut(q, e, poll).map(RootCompletion::LocalL1)
            }
            Self::Single(q, e) => evaluate_raw_root_cut(q, e, poll).map(RootCompletion::Single),
            Self::Triplet(q, e) => {
                evaluate_raw_triplet_root_cut(q, e, poll).map(RootCompletion::Triplet)
            }
            Self::Joint(q, e) => {
                evaluate_raw_two_pair_root_cut(q, e, poll).map(RootCompletion::Joint)
            }
        }
    }
}
#[expect(
    clippy::large_enum_variant,
    reason = "Keep inline ownership and accounted storage; boxing adds an allocation"
)]
enum RootCompletion {
    TwoLocalL1(CompletedRawTwoLocalL1RootCut),
    LocalL1(CompletedRawLocalL1RootCut),
    Single(CompletedRawRootCut),
    Joint(CompletedRawTwoPairRootCut),
    Triplet(CompletedRawTripletRootCut),
}
impl RootCompletion {
    fn bounds(&self) -> (f32, f32) {
        match self {
            Self::TwoLocalL1(v) => v.bounds(),
            Self::LocalL1(v) => v.bounds(),
            Self::Single(v) => v.bounds(),
            Self::Joint(v) => v.bounds(),
            Self::Triplet(v) => v.bounds(),
        }
    }
    fn support_calls(&self) -> usize {
        match self {
            Self::TwoLocalL1(v) => v.support_calls(),
            Self::LocalL1(v) => v.support_calls(),
            Self::Single(v) => v.support_calls(),
            Self::Joint(v) => v.support_calls(),
            Self::Triplet(v) => v.support_calls(),
        }
    }
    fn cut_calls(&self) -> usize {
        match self {
            Self::TwoLocalL1(v) => v.cut_calls(),
            Self::LocalL1(v) => v.cut_calls(),
            Self::Single(v) => v.cut_calls(),
            Self::Joint(v) => v.cut_calls(),
            Self::Triplet(v) => v.cut_calls(),
        }
    }
}

/// One bounded diagnostic per numerically eligible original-root invocation.
/// These fields describe reached stages; they grant no source or proof authority.
/// No timer, engine accessor, collection scan or owned payload lives here.
struct RootCutObservation {
    bundle: AdmissionObservation,
    row: usize,
    stage: &'static str,
    method: &'static str,
    joint_fallback: &'static str,
    source_raw: bool,
    backend_ready: bool,
    slice_granted: bool,
    memory_cap: Option<usize>,
    setup_admitted: bool,
    target_selected: bool,
    quoted_work: Option<u64>,
    quoted_scratch: Option<usize>,
    quoted_candidates: Option<usize>,
    execution_admitted: bool,
    completed_support_calls: Option<usize>,
    completed_cut_calls: Option<usize>,
    completed_normal: Option<[f32; 4]>,
    completed_lambda_scale: Option<f32>,
    completed_joint_pairs: Option<[[usize; 2]; 2]>,
    completed_joint_normals: Option<[[f32; 4]; 2]>,
    completed_joint_scales: Option<[f32; 2]>,
}

impl Drop for RootCutObservation {
    fn drop(&mut self) {
        // Capability-absent and pre-lease exits keep exactly the historical
        // event. The eleven new fields are formatted only after the additional
        // fixed observation debit has been admitted; no unpaid log expansion.
        if self.bundle.capability && self.setup_admitted {
            tracing::info!(
                bundle_capability = self.bundle.capability,
                bundle_stage = self.bundle.stage,
                bundle_reason = self.bundle.reason,
                bundle_scan_started = self.bundle.scan_started,
                bundle_scan_complete = self.bundle.scan_complete,
                bundle_crossing_count = self.bundle.crossing,
                bundle_negative_crossing_count = self.bundle.negative_crossing,
                bundle_group_0_count = self.bundle.groups[0],
                bundle_group_1_count = self.bundle.groups[1],
                bundle_failed_group_present = self.bundle.failed_group.is_some(),
                bundle_failed_group = self.bundle.failed_group.unwrap_or(0),
                row = self.row,
                stage = self.stage,
                method = self.method,
                joint_fallback = self.joint_fallback,
                source_raw = self.source_raw,
                backend_ready = self.backend_ready,
                slice_granted = self.slice_granted,
                memory_cap_present = self.memory_cap.is_some(),
                memory_cap = self.memory_cap.unwrap_or(0),
                setup_admitted = self.setup_admitted,
                target_selected = self.target_selected,
                quoted_work_present = self.quoted_work.is_some(),
                quoted_work = self.quoted_work.unwrap_or(0),
                quoted_scratch_present = self.quoted_scratch.is_some(),
                quoted_scratch = self.quoted_scratch.unwrap_or(0),
                quoted_candidates_present = self.quoted_candidates.is_some(),
                quoted_candidates = self.quoted_candidates.unwrap_or(0),
                execution_admitted = self.execution_admitted,
                completed_support_calls_present = self.completed_support_calls.is_some(),
                completed_support_calls = self.completed_support_calls.unwrap_or(0),
                completed_cut_calls_present = self.completed_cut_calls.is_some(),
                completed_cut_calls = self.completed_cut_calls.unwrap_or(0),
                completed_normal_present = self.completed_normal.is_some(),
                normal_pre_0 = self.completed_normal.map_or(0., |n| n[0]),
                normal_pre_1 = self.completed_normal.map_or(0., |n| n[1]),
                normal_post_0 = self.completed_normal.map_or(0., |n| n[2]),
                normal_post_1 = self.completed_normal.map_or(0., |n| n[3]),
                completed_lambda_scale_present = self.completed_lambda_scale.is_some(),
                lambda_scale = self.completed_lambda_scale.unwrap_or(0.),
                completed_joint_present = self.completed_joint_pairs.is_some(),
                joint_pair_0_first = self.completed_joint_pairs.map_or(0, |p| p[0][0]),
                joint_pair_0_second = self.completed_joint_pairs.map_or(0, |p| p[0][1]),
                joint_pair_1_first = self.completed_joint_pairs.map_or(0, |p| p[1][0]),
                joint_pair_1_second = self.completed_joint_pairs.map_or(0, |p| p[1][1]),
                joint_normal_0_pre_0 = self.completed_joint_normals.map_or(0., |n| n[0][0]),
                joint_normal_0_pre_1 = self.completed_joint_normals.map_or(0., |n| n[0][1]),
                joint_normal_0_post_0 = self.completed_joint_normals.map_or(0., |n| n[0][2]),
                joint_normal_0_post_1 = self.completed_joint_normals.map_or(0., |n| n[0][3]),
                joint_normal_1_pre_0 = self.completed_joint_normals.map_or(0., |n| n[1][0]),
                joint_normal_1_pre_1 = self.completed_joint_normals.map_or(0., |n| n[1][1]),
                joint_normal_1_post_0 = self.completed_joint_normals.map_or(0., |n| n[1][2]),
                joint_normal_1_post_1 = self.completed_joint_normals.map_or(0., |n| n[1][3]),
                joint_scale_0 = self.completed_joint_scales.map_or(0., |s| s[0]),
                joint_scale_1 = self.completed_joint_scales.map_or(0., |s| s[1]),
                accepted = self.stage == "accepted",
                "Certified root cut outcome"
            );
        } else {
            tracing::info!(
                row = self.row,
                stage = self.stage,
                method = self.method,
                joint_fallback = self.joint_fallback,
                source_raw = self.source_raw,
                backend_ready = self.backend_ready,
                slice_granted = self.slice_granted,
                memory_cap_present = self.memory_cap.is_some(),
                memory_cap = self.memory_cap.unwrap_or(0),
                setup_admitted = self.setup_admitted,
                target_selected = self.target_selected,
                quoted_work_present = self.quoted_work.is_some(),
                quoted_work = self.quoted_work.unwrap_or(0),
                quoted_scratch_present = self.quoted_scratch.is_some(),
                quoted_scratch = self.quoted_scratch.unwrap_or(0),
                quoted_candidates_present = self.quoted_candidates.is_some(),
                quoted_candidates = self.quoted_candidates.unwrap_or(0),
                execution_admitted = self.execution_admitted,
                completed_support_calls_present = self.completed_support_calls.is_some(),
                completed_support_calls = self.completed_support_calls.unwrap_or(0),
                completed_cut_calls_present = self.completed_cut_calls.is_some(),
                completed_cut_calls = self.completed_cut_calls.unwrap_or(0),
                completed_normal_present = self.completed_normal.is_some(),
                normal_pre_0 = self.completed_normal.map_or(0., |n| n[0]),
                normal_pre_1 = self.completed_normal.map_or(0., |n| n[1]),
                normal_post_0 = self.completed_normal.map_or(0., |n| n[2]),
                normal_post_1 = self.completed_normal.map_or(0., |n| n[3]),
                completed_lambda_scale_present = self.completed_lambda_scale.is_some(),
                lambda_scale = self.completed_lambda_scale.unwrap_or(0.),
                completed_joint_present = self.completed_joint_pairs.is_some(),
                joint_pair_0_first = self.completed_joint_pairs.map_or(0, |p| p[0][0]),
                joint_pair_0_second = self.completed_joint_pairs.map_or(0, |p| p[0][1]),
                joint_pair_1_first = self.completed_joint_pairs.map_or(0, |p| p[1][0]),
                joint_pair_1_second = self.completed_joint_pairs.map_or(0, |p| p[1][1]),
                joint_normal_0_pre_0 = self.completed_joint_normals.map_or(0., |n| n[0][0]),
                joint_normal_0_pre_1 = self.completed_joint_normals.map_or(0., |n| n[0][1]),
                joint_normal_0_post_0 = self.completed_joint_normals.map_or(0., |n| n[0][2]),
                joint_normal_0_post_1 = self.completed_joint_normals.map_or(0., |n| n[0][3]),
                joint_normal_1_pre_0 = self.completed_joint_normals.map_or(0., |n| n[1][0]),
                joint_normal_1_pre_1 = self.completed_joint_normals.map_or(0., |n| n[1][1]),
                joint_normal_1_post_0 = self.completed_joint_normals.map_or(0., |n| n[1][2]),
                joint_normal_1_post_1 = self.completed_joint_normals.map_or(0., |n| n[1][3]),
                joint_scale_0 = self.completed_joint_scales.map_or(0., |s| s[0]),
                joint_scale_1 = self.completed_joint_scales.map_or(0., |s| s[1]),
                accepted = self.stage == "accepted",
                "Certified root cut outcome"
            );
        }
    }
}

/// Called ONLY by the genuine original-root coordinator, after its complete
/// ordinary result and before root completion/branching. No caller-supplied
/// candidate scalar is accepted; all caller state remains unchanged on refusal.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn try_tighten(
    config: &BetaCrownConfig,
    graph: &GraphNetwork,
    input: &BoundedTensor,
    objectives: &[Vec<f32>],
    thresholds: &[f32],
    bounds: &mut [(f32, f32)],
    node_bounds: &HashMap<String, BoundedTensor>,
    alpha: Option<&GraphAlphaState>,
    engine: Option<&dyn GemmEngine>,
    root_deadline: Option<Instant>,
) -> Option<usize> {
    try_tighten_with_spatial_advice(
        config,
        graph,
        input,
        objectives,
        thresholds,
        bounds,
        node_bounds,
        alpha,
        engine,
        root_deadline,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn try_tighten_with_spatial_advice(
    config: &BetaCrownConfig,
    graph: &GraphNetwork,
    input: &BoundedTensor,
    objectives: &[Vec<f32>],
    thresholds: &[f32],
    bounds: &mut [(f32, f32)],
    node_bounds: &HashMap<String, BoundedTensor>,
    alpha: Option<&GraphAlphaState>,
    engine: Option<&dyn GemmEngine>,
    root_deadline: Option<Instant>,
    advice: Option<SpatialAdvice<'_>>,
) -> Option<usize> {
    if config.verification_artifact_authority != VerificationArtifactAuthority::VerdictOnly
        || !config.certified_root_cut_raw_source
        || config.crown_backward_layers.is_some()
        || objectives.is_empty()
        || objectives.len() > MAX_ROWS
        || objectives.len() != thresholds.len()
        || objectives.len() != bounds.len()
    {
        return None;
    }
    let owner = config.certified_root_cut_owner.as_ref()?;
    if !owner.enabled || owner.attempted.load(Ordering::Acquire) {
        return None;
    }
    let row = bounds.iter().zip(thresholds).position(|(&(lo, hi), &t)| {
        lo.is_finite() && hi.is_finite() && t.is_finite() && lo <= hi && lo <= t
    })?;
    let mut observation = RootCutObservation {
        bundle: AdmissionObservation::default(),
        row,
        stage: "backend-unavailable",
        method: "none",
        joint_fallback: "not-attempted",
        source_raw: config.certified_root_cut_raw_source,
        backend_ready: false,
        slice_granted: false,
        memory_cap: None,
        setup_admitted: false,
        target_selected: false,
        quoted_work: None,
        quoted_scratch: None,
        quoted_candidates: None,
        execution_admitted: false,
        completed_support_calls: None,
        completed_cut_calls: None,
        completed_normal: None,
        completed_lambda_scale: None,
        completed_joint_pairs: None,
        completed_joint_normals: None,
        completed_joint_scales: None,
    };
    // The ordinary local engine may be CPU even when its completed proof
    // folds used global CUDA. Borrow only ready slots, never lazy factories.
    let spatial_engine = advice.and_then(|_| spatial::ready(engine));
    let methods =
        select_ready_methods(engine, crate::sound_f64_gemm::cpu_only_f64_active(), || {
            let ordinary = crate::sound_f64_gemm::preinitialized_root_cut_engine();
            let [resident, wide] = crate::sound_gpu_gate::preinitialized_root_cut_engines();
            [ordinary, resident, wide]
        })
        .or_else(|| {
            spatial_engine.map(|_| ReadyMethods {
                two_local_l1: None,
                single: None,
                joint: None,
                triplet: None,
                local_l1: None,
            })
        })?;
    observation.bundle.capability = methods.two_local_l1.is_some();
    if observation.bundle.capability {
        observation.bundle.stage = "owner";
        observation.bundle.reason = "owner-deadline";
    }
    observation.backend_ready = true;
    observation.stage = "owner-deadline";
    let end = owner.claim(root_deadline?)?;
    observation.slice_granted = true;
    observation.stage = "memory";
    let mut extra_setup = if methods.triplet.is_some()
        || methods.local_l1.is_some()
        || methods.two_local_l1.is_some()
    {
        FAMILY_SCRATCH
    } else {
        0
    };
    if spatial_engine.is_some() {
        extra_setup = extra_setup.checked_add(spatial::FIXED_BYTES)?;
    }
    if observation.bundle.capability {
        observation.bundle.stage = "setup";
        observation.bundle.reason = "memory";
        if size_of::<RootCutObservation>()
            + size_of::<ReadyMethods<'_>>()
            + 2 * size_of::<AdmissionObservation>()
            + size_of::<RootProposal<'_, '_>>()
            + size_of::<RootCompletion>()
            + 2 * size_of::<two_local_l1::BundleChoice>()
            + 2 * size_of::<two_local_l1::BundleSelection>()
            + 2 * size_of::<RawTwoLocalL1RootCutQuote<'_>>()
            + 2 * size_of::<two_local_l1::geometry::GeometrySource<'_>>()
            + 2048
            > QUOTE_SCRATCH
        {
            observation.bundle.reason = "metadata-layout";
            return None;
        }
    }
    let cap = memory_cap()?.checked_sub(extra_setup)?;
    observation.memory_cap = Some(cap);
    observation.stage = "setup-budget";
    if observation.bundle.capability {
        observation.bundle.reason = "setup-budget";
    }
    // The additional fixed diagnostic debit applies only to ready bundle
    // capability. It is reserved before target/advice and never refunded.
    let setup_work = QUOTE_WORK.checked_add(if observation.bundle.capability {
        BUNDLE_OBSERVATION_WORK
    } else {
        0
    })?;
    let mut setup = owner
        .budget
        .lease(WorkRequest {
            scratch_bytes: QUOTE_SCRATCH.checked_add(extra_setup)?,
            work: setup_work,
            proposals: 1,
            ..WorkRequest::default()
        })
        .ok()?;
    observation.setup_admitted = true;
    if let (Some(advice), Some(spatial_engine)) = (advice, spatial_engine) {
        let remaining = MAX_WORK.checked_sub(owner.budget.receipt().ok()?.work_reserved)?;
        if let Some(quote) = spatial::plan(
            graph,
            input,
            node_bounds,
            alpha,
            &objectives[row],
            bounds[row],
            row,
            objectives.len(),
            advice,
            spatial_engine,
            &mut setup,
            remaining,
            cap,
            end,
        ) {
            if let Ok(mut execution) = owner.budget.lease(WorkRequest {
                scratch_bytes: quote.bytes(),
                work: quote.work(),
                ..WorkRequest::default()
            }) {
                observation.method = "spatial-prefix-feedback";
                observation.target_selected = true;
                observation.quoted_work = Some(quote.work());
                observation.quoted_scratch = Some(quote.bytes());
                observation.quoted_candidates = Some(quote.calls());
                observation.execution_admitted = true;
                observation.stage = "execution";
                let old = bounds[row];
                let gpu = spatial_engine.as_gpu_crown_backward()?;
                let completed = spatial::evaluate(&quote, gpu, &mut || {
                    Instant::now() < end
                        && setup.checkpoint().is_ok()
                        && execution.checkpoint().is_ok()
                })?;
                if Instant::now() >= end
                    || setup.checkpoint().is_err()
                    || execution.complete().is_err()
                {
                    return None;
                }
                tracing::info!(
                    row,
                    selected_rows = completed.rows,
                    completed_prefix_calls = completed.prefix_calls,
                    completed_original_replays = 1usize,
                    before = old.0,
                    after = completed.lower,
                    upper_unchanged = true,
                    "Certified spatial intermediate feedback completed"
                );
                if completed.lower <= old.0 {
                    observation.stage = "no-gain";
                    return None;
                }
                bounds[row].0 = completed.lower;
                observation.stage = "accepted";
                return Some(row);
            }
        }
    }
    observation.stage = "target";
    if observation.bundle.capability {
        observation.bundle.stage = "target";
        observation.bundle.reason = "target-refused";
    }
    let (relu, pair, lambda_scale) =
        target(graph, input, node_bounds, &mut setup, &objectives[row], end)?;
    observation.target_selected = true;
    let old = bounds[row];
    let mut alive = true;
    let mut admitted = None;
    let mut joint_quote_attempted = false;
    // Advice is an explicit numerical scheduling barrier: after its E scan
    // starts, only its one freshly quoted bundle may execute.
    let mut geometry_started = false;
    // Two disjoint groups have independent normalization. Only scheduling
    // advice is inspected here; both certificates and the predeclared full
    // fold(s) remain private work after the common numerical barrier below.
    if let Some(engine) = methods.two_local_l1 {
        observation.stage = "two-local-selection";
        if let Some(selection) = two_local_l1::bundle_selection(
            graph,
            input,
            node_bounds,
            &mut setup,
            &objectives[row],
            relu,
            end,
            &mut observation.bundle,
        ) {
            let choice = selection.choice;
            let compare_single3 = observation.bundle.scan_complete
                && observation.bundle.negative_crossing == selection.count
                && (9..=10).contains(&selection.count)
                && local_l1_capabilities(engine);
            joint_quote_attempted = true;
            observation.bundle.stage = "quote";
            observation.bundle.reason = "quote-refused";
            let remaining = MAX_WORK.checked_sub(owner.budget.receipt().ok()?.work_reserved)?;
            let mut candidate = quote_raw_two_local_l1_root_cut(
                graph,
                input,
                node_bounds,
                alpha,
                &objectives[row],
                old,
                relu,
                choice.active,
                choice.neurons,
                choice.weights,
                choice.multipliers,
                RawRootCutLimits {
                    max_nodes: MAX_NODES,
                    max_scratch_bytes: cap,
                    max_work: remaining,
                },
                end,
                &mut || {
                    alive &= Instant::now() < end && setup.checkpoint().is_ok();
                    alive
                },
            );
            if compare_single3 {
                candidate = candidate.and_then(|q| {
                    q.with_single3_comparison(RawRootCutLimits {
                        max_nodes: MAX_NODES,
                        max_scratch_bytes: cap,
                        max_work: remaining,
                    })
                });
            }
            if (6..=8).contains(&selection.count) {
                // This route exists only when a common Q8 scale lost a
                // selected coefficient. The optional original-parent-box
                // candidate and its required bundle prefix share this quote,
                // execution lease and absolute deadline.
                if let Some(q) = candidate.as_mut() {
                    q.try_enable_head_box_floor(
                        thresholds[row],
                        RawRootCutLimits {
                            max_nodes: MAX_NODES,
                            max_scratch_bytes: cap,
                            max_work: remaining,
                        },
                    );
                    // Preserve the complete box/prefix plan first. Joint
                    // multiplier advice can improve that same checked plan
                    // only when its additional work and storage also fit.
                    q.try_enable_full_head_advice(RawRootCutLimits {
                        max_nodes: MAX_NODES,
                        max_scratch_bytes: cap,
                        max_work: remaining,
                    });
                }
            }
            if (9..=10).contains(&selection.count) {
                // The old same-source quote is shape/preflight authority only;
                // no certificate or fold has run. Drop it after an independent
                // fresh quote binds the actual geometry-selected IDs/weights.
                candidate = candidate.and_then(|mut q| {
                    let source = two_local_l1::geometry::geometry_quote(
                        graph,
                        input,
                        node_bounds,
                        relu,
                        &mut setup,
                        end,
                        &mut observation.bundle,
                    )?;
                    let advice = source.work(selection.count)?;
                    let remaining =
                        MAX_WORK.checked_sub(owner.budget.receipt().ok()?.work_reserved)?;
                    observation.bundle.stage = "geometry-preflight";
                    observation.bundle.reason = "combined-work";
                    // The optional full-head advice is part of the predeclared
                    // execution quote. Keep the exact original two-fold quote
                    // if its additional shape-only charge cannot fit alongside
                    // this geometry scan and fresh quote. No E work has started.
                    if compare_single3 {
                        if let Some(execution_work) = remaining
                            .checked_sub(advice)
                            .and_then(|v| v.checked_sub(QUOTE_WORK))
                        {
                            q.try_enable_full_head_advice(RawRootCutLimits {
                                max_nodes: MAX_NODES,
                                max_scratch_bytes: cap,
                                max_work: execution_work,
                            });
                            q.try_enable_head_box_floor(
                                thresholds[row],
                                RawRootCutLimits {
                                    max_nodes: MAX_NODES,
                                    max_scratch_bytes: cap,
                                    max_work: execution_work,
                                },
                            );
                        }
                    }
                    if !two_local_l1::geometry_work_fits(remaining, advice, q.work()) {
                        // No E scan or advice debit has occurred. Old routes
                        // retain this owner/deadline and the residual ledger.
                        return None;
                    }
                    if !alive || Instant::now() >= end || setup.checkpoint().is_err() {
                        observation.bundle.reason = "deadline-or-cancelled";
                        return None;
                    }
                    setup.reserve_more_work(QUOTE_WORK).ok()?;
                    geometry_started = true;
                    let selected = source.evaluate(
                        &selection.ranked,
                        selection.count,
                        &choice,
                        &mut setup,
                        end,
                        &mut observation.bundle,
                    )?;
                    observation.bundle.groups = selected.active;
                    observation.bundle.stage = "geometry-requote";
                    observation.bundle.reason = "quote-refused";
                    let remaining =
                        MAX_WORK.checked_sub(owner.budget.receipt().ok()?.work_reserved)?;
                    let fresh = quote_raw_two_local_l1_root_cut(
                        graph,
                        input,
                        node_bounds,
                        alpha,
                        &objectives[row],
                        old,
                        relu,
                        selected.active,
                        selected.neurons,
                        selected.weights,
                        selected.multipliers,
                        RawRootCutLimits {
                            max_nodes: MAX_NODES,
                            max_scratch_bytes: cap,
                            max_work: remaining,
                        },
                        end,
                        &mut || {
                            alive &= Instant::now() < end && setup.checkpoint().is_ok();
                            alive
                        },
                    )?;
                    let fresh = if compare_single3 {
                        fresh.with_single3_comparison(RawRootCutLimits {
                            max_nodes: MAX_NODES,
                            max_scratch_bytes: cap,
                            max_work: remaining,
                        })?
                    } else {
                        fresh
                    };
                    let mut fresh = fresh;
                    if q.uses_full_head_advice()
                        && !fresh.try_enable_full_head_advice(RawRootCutLimits {
                            max_nodes: MAX_NODES,
                            max_scratch_bytes: cap,
                            max_work: remaining,
                        })
                    {
                        observation.bundle.reason = "full-head-advice-requote-refused";
                        return None;
                    }
                    if q.uses_head_box_floor()
                        && !fresh.try_enable_head_box_floor(
                            thresholds[row],
                            RawRootCutLimits {
                                max_nodes: MAX_NODES,
                                max_scratch_bytes: cap,
                                max_work: remaining,
                            },
                        )
                    {
                        observation.bundle.reason = "head-box-requote-refused";
                        return None;
                    }
                    // Current quotes price the same target and both maximum
                    // kernel allowances independently of the repartition.
                    // Refuse if that source contract ever changes.
                    if fresh.work() != q.work() || fresh.scratch_bytes() != q.scratch_bytes() {
                        observation.bundle.reason = "execution-envelope-changed";
                        return None;
                    }
                    Some(fresh)
                });
            }
            if let Some(q) = candidate {
                observation.bundle.stage = "admission";
                observation.bundle.reason = "execution-budget";
                if alive && Instant::now() < end && setup.checkpoint().is_ok() {
                    if let Ok(execution) = owner.budget.lease(WorkRequest {
                        scratch_bytes: q.scratch_bytes(),
                        work: q.work(),
                        ..WorkRequest::default()
                    }) {
                        admitted = Some((RootProposal::TwoLocalL1(q, engine), execution));
                        observation.bundle.stage = "admitted";
                        observation.bundle.reason = "none";
                        observation.joint_fallback = "none";
                    }
                }
            }
        }
    }
    if geometry_started && admitted.is_none() {
        // No extra family call after source geometry was entered, including
        // arithmetic refusal, cancellation, late re-quote or lease refusal.
        return None;
    }
    if admitted.is_none() {
        // This extra guard belongs only to the new pre-entry fallback. The
        // capability-absent old path keeps its original checkpoint ordering.
        if methods.two_local_l1.is_some()
            && (!alive || Instant::now() >= end || setup.checkpoint().is_err())
        {
            return None;
        }
        // Prefer a larger local relation only for five to eight eligible channels.
        // Shape/selection refusal can fall back; after evaluate starts below there
        // is no path back to any other family, even if local arithmetic refuses.
        if let Some(engine) = methods.local_l1 {
            if joint_quote_attempted {
                setup.reserve_more_work(QUOTE_WORK).ok()?;
            }
            observation.stage = "local-selection";
            if let Some(choice) = local_target(
                graph,
                input,
                node_bounds,
                &mut setup,
                &objectives[row],
                relu,
                end,
            ) {
                joint_quote_attempted = true;
                let remaining = MAX_WORK.checked_sub(owner.budget.receipt().ok()?.work_reserved)?;
                let candidate = quote_raw_local_l1_root_cut(
                    graph,
                    input,
                    node_bounds,
                    alpha,
                    &objectives[row],
                    old,
                    relu,
                    choice.active,
                    choice.neurons,
                    choice.weights,
                    choice.multiplier,
                    RawRootCutLimits {
                        max_nodes: MAX_NODES,
                        max_scratch_bytes: cap,
                        max_work: remaining,
                    },
                    end,
                    &mut || {
                        alive &= Instant::now() < end && setup.checkpoint().is_ok();
                        alive
                    },
                );
                if let Some(q) = candidate {
                    if alive && Instant::now() < end && setup.checkpoint().is_ok() {
                        if let Ok(execution) = owner.budget.lease(WorkRequest {
                            scratch_bytes: q.scratch_bytes(),
                            work: q.work(),
                            ..WorkRequest::default()
                        }) {
                            admitted = Some((RootProposal::LocalL1(q, engine), execution));
                            observation.joint_fallback = "none";
                        }
                    }
                }
            }
        }
    }
    if admitted.is_none() && alive && Instant::now() < end && setup.checkpoint().is_ok() {
        if let Some(engine) = methods.triplet {
            if joint_quote_attempted {
                setup.reserve_more_work(QUOTE_WORK).ok()?;
            }

            observation.stage = "family-selection";
            if let Some(choice) = family_target(
                graph,
                input,
                node_bounds,
                &mut setup,
                &objectives[row],
                relu,
                end,
                methods.joint.is_some(),
            ) {
                joint_quote_attempted = true;
                let remaining = MAX_WORK.checked_sub(owner.budget.receipt().ok()?.work_reserved)?;
                let limits = RawRootCutLimits {
                    max_nodes: MAX_NODES,
                    max_scratch_bytes: cap,
                    max_work: remaining,
                };
                let mut poll = || {
                    alive &= Instant::now() < end && setup.checkpoint().is_ok();
                    alive
                };
                let candidate = match choice {
                    FamilyChoice::Triplet {
                        neurons,
                        weights,
                        multiplier,
                    } => quote_raw_triplet_root_cut(
                        graph,
                        input,
                        node_bounds,
                        alpha,
                        &objectives[row],
                        old,
                        relu,
                        neurons,
                        weights,
                        multiplier,
                        limits,
                        end,
                        &mut poll,
                    )
                    .map(|q| RootProposal::Triplet(q, engine)),
                    FamilyChoice::Joint { pairs, scales } => quote_raw_two_pair_root_cut(
                        graph,
                        input,
                        node_bounds,
                        alpha,
                        &objectives[row],
                        old,
                        relu,
                        pairs,
                        scales,
                        limits,
                        end,
                        &mut poll,
                    )
                    .and_then(|q| methods.joint.map(|e| RootProposal::Joint(q, e))),
                };
                if let Some(candidate) = candidate {
                    if alive && Instant::now() < end && setup.checkpoint().is_ok() {
                        if let Ok(execution) = owner.budget.lease(WorkRequest {
                            scratch_bytes: candidate.scratch(),
                            work: candidate.work(),
                            ..WorkRequest::default()
                        }) {
                            admitted = Some((candidate, execution));
                            observation.joint_fallback = "none";
                        }
                    }
                }
            }
        }
    }
    // Refusal before numeric execution may still select the old joint method.
    // Shape rescans and failed screening debits are never refunded.
    if admitted.is_none() && alive && Instant::now() < end && setup.checkpoint().is_ok() {
        if let Some(engine) = methods.joint {
            if joint_quote_attempted {
                setup.reserve_more_work(QUOTE_WORK).ok()?;
            }
            observation.stage = "joint-selection";
            observation.joint_fallback = "selection-refused";
            if let Some((pairs, scales)) = joint_target(
                graph,
                input,
                node_bounds,
                &mut setup,
                &objectives[row],
                relu,
                end,
            ) {
                observation.stage = "joint-quote";
                joint_quote_attempted = true;
                observation.joint_fallback = "quote-refused";
                let remaining = MAX_WORK.checked_sub(owner.budget.receipt().ok()?.work_reserved)?;
                let quote = quote_raw_two_pair_root_cut(
                    graph,
                    input,
                    node_bounds,
                    alpha,
                    &objectives[row],
                    old,
                    relu,
                    pairs,
                    scales,
                    RawRootCutLimits {
                        max_nodes: MAX_NODES,
                        max_scratch_bytes: cap,
                        max_work: remaining,
                    },
                    end,
                    &mut || {
                        alive &= Instant::now() < end && setup.checkpoint().is_ok();
                        alive
                    },
                );
                if let Some(quote) = quote {
                    observation.stage = "joint-execution-budget";
                    observation.joint_fallback = "admission-refused";
                    if alive && Instant::now() < end && setup.checkpoint().is_ok() {
                        if let Ok(execution) = owner.budget.lease(WorkRequest {
                            scratch_bytes: quote.scratch_bytes(),
                            work: quote.work(),
                            ..WorkRequest::default()
                        }) {
                            admitted = Some((RootProposal::Joint(quote, engine), execution));
                            observation.joint_fallback = "none";
                        }
                    }
                }
            }
        } else {
            observation.joint_fallback = "backend-unavailable";
        }
    }
    // Pre-execution refusal may use the old method with the SAME owner, end
    // and residual work. An entered joint numeric call never retries here.
    if admitted.is_none() {
        if !alive || Instant::now() >= end || setup.checkpoint().is_err() {
            return None;
        }
        let engine = methods.single?;
        if joint_quote_attempted {
            // A second whole shape quote is work too, even after refusal.
            setup.reserve_more_work(QUOTE_WORK).ok()?;
        }
        observation.stage = "quote";
        let remaining = MAX_WORK.checked_sub(owner.budget.receipt().ok()?.work_reserved)?;
        let quote = quote_raw_root_cut(
            graph,
            input,
            node_bounds,
            alpha,
            &objectives[row],
            old,
            relu,
            pair,
            lambda_scale,
            RawRootCutLimits {
                max_nodes: MAX_NODES,
                max_scratch_bytes: cap,
                max_work: remaining,
            },
            end,
            &mut || {
                alive &= Instant::now() < end && setup.checkpoint().is_ok();
                alive
            },
        )?;
        observation.stage = "quote-final-check";
        if !alive || Instant::now() >= end || setup.checkpoint().is_err() {
            return None;
        }
        observation.stage = "execution-budget";
        let execution = owner
            .budget
            .lease(WorkRequest {
                scratch_bytes: quote.scratch_bytes(),
                work: quote.work(),
                ..WorkRequest::default()
            })
            .ok()?;
        admitted = Some((RootProposal::Single(quote, engine), execution));
    }
    let (proposal, mut execution) = admitted?;
    observation.method = proposal.method();
    observation.quoted_work = Some(proposal.work());
    observation.quoted_scratch = Some(proposal.scratch());
    observation.quoted_candidates = Some(proposal.candidates());
    observation.execution_admitted = true;
    observation.stage = "execution";
    let mut run_alive = true;
    let mut poll = || {
        run_alive &=
            Instant::now() < end && setup.checkpoint().is_ok() && execution.checkpoint().is_ok();
        run_alive
    };
    let entered_bundle = matches!(&proposal, RootProposal::TwoLocalL1(..));
    if entered_bundle {
        observation.bundle.stage = "execution";
        observation.bundle.reason = "evaluator-refused";
    }
    let completed = proposal.evaluate(&mut poll)?;
    if entered_bundle {
        observation.bundle.stage = "completed";
        observation.bundle.reason = "none";
    }
    observation.completed_support_calls = Some(completed.support_calls());
    observation.completed_cut_calls = Some(completed.cut_calls());
    if let RootCompletion::TwoLocalL1(v) = &completed {
        if let Some(advice) = v.multiplier_advice() {
            tracing::info!(
                row,
                search = "bounded_joint_kelley",
                original_complement_multiplier = advice.original[0],
                original_triple_multiplier = advice.original[1],
                chosen_complement_multiplier = advice.chosen[0],
                chosen_triple_multiplier = advice.chosen[1],
                complement_branch_upper = advice.branch_upper[0],
                triple_branch_upper = advice.branch_upper[1],
                original_advisory_score = advice.original_score,
                chosen_advisory_score = advice.chosen_score,
                score_evaluations = advice.score_evaluations,
                full_scans = advice.full_scans,
                support_planes = advice.support_planes,
                lp_candidates = advice.lp_candidates,
                "Full-head multiplier advice completed"
            );
        }
    }
    match &completed {
        RootCompletion::TwoLocalL1(v) => tracing::info!(
            row,
            certificates = v.certificates(),
            candidate_count = v.candidate_count(),
            completed_prefix_calls = v.cut_calls(),
            head_box_candidate_present = v.head_box_candidate_lower().is_some(),
            head_box_candidate_lower = v.head_box_candidate_lower().unwrap_or(0.),
            single3_candidate_present = v.single3_candidate_lower().is_some(),
            single3_candidate_lower = v.single3_candidate_lower().unwrap_or(0.),
            bundle_candidate_present = v.bundle_candidate_lower().is_some(),
            bundle_candidate_lower = v.bundle_candidate_lower().unwrap_or(0.),
            group_0_active_len = v.active_lens()[0],
            group_0_multiplier = v.multipliers()[0],
            group_0_support_bound = v.support_bounds()[0],
            group_0_original_local_bound = v.original_local_bounds()[0],
            group_0_bn_radius = v.bn_radii()[0],
            group_0_sign_count = v.sign_counts()[0],
            group_0_neuron_0 = v.ordered_neurons()[0][0],
            group_0_weight_0 = v.weights()[0][0],
            group_0_pre_lower_0 = v.pre_lower()[0][0],
            group_0_pre_upper_0 = v.pre_upper()[0][0],
            group_0_neuron_1 = v.ordered_neurons()[0][1],
            group_0_weight_1 = v.weights()[0][1],
            group_0_pre_lower_1 = v.pre_lower()[0][1],
            group_0_pre_upper_1 = v.pre_upper()[0][1],
            group_0_neuron_2 = v.ordered_neurons()[0][2],
            group_0_weight_2 = v.weights()[0][2],
            group_0_pre_lower_2 = v.pre_lower()[0][2],
            group_0_pre_upper_2 = v.pre_upper()[0][2],
            group_0_neuron_3 = v.ordered_neurons()[0][3],
            group_0_weight_3 = v.weights()[0][3],
            group_0_pre_lower_3 = v.pre_lower()[0][3],
            group_0_pre_upper_3 = v.pre_upper()[0][3],
            group_0_neuron_4 = v.ordered_neurons()[0][4],
            group_0_weight_4 = v.weights()[0][4],
            group_0_pre_lower_4 = v.pre_lower()[0][4],
            group_0_pre_upper_4 = v.pre_upper()[0][4],
            group_0_neuron_5 = v.ordered_neurons()[0][5],
            group_0_weight_5 = v.weights()[0][5],
            group_0_pre_lower_5 = v.pre_lower()[0][5],
            group_0_pre_upper_5 = v.pre_upper()[0][5],
            group_0_neuron_6 = v.ordered_neurons()[0][6],
            group_0_weight_6 = v.weights()[0][6],
            group_0_pre_lower_6 = v.pre_lower()[0][6],
            group_0_pre_upper_6 = v.pre_upper()[0][6],
            group_0_neuron_7 = v.ordered_neurons()[0][7],
            group_0_weight_7 = v.weights()[0][7],
            group_0_pre_lower_7 = v.pre_lower()[0][7],
            group_0_pre_upper_7 = v.pre_upper()[0][7],
            group_1_active_len = v.active_lens()[1],
            group_1_multiplier = v.multipliers()[1],
            group_1_support_bound = v.support_bounds()[1],
            group_1_original_local_bound = v.original_local_bounds()[1],
            group_1_bn_radius = v.bn_radii()[1],
            group_1_sign_count = v.sign_counts()[1],
            group_1_neuron_0 = v.ordered_neurons()[1][0],
            group_1_weight_0 = v.weights()[1][0],
            group_1_pre_lower_0 = v.pre_lower()[1][0],
            group_1_pre_upper_0 = v.pre_upper()[1][0],
            group_1_neuron_1 = v.ordered_neurons()[1][1],
            group_1_weight_1 = v.weights()[1][1],
            group_1_pre_lower_1 = v.pre_lower()[1][1],
            group_1_pre_upper_1 = v.pre_upper()[1][1],
            group_1_neuron_2 = v.ordered_neurons()[1][2],
            group_1_weight_2 = v.weights()[1][2],
            group_1_pre_lower_2 = v.pre_lower()[1][2],
            group_1_pre_upper_2 = v.pre_upper()[1][2],
            group_1_neuron_3 = v.ordered_neurons()[1][3],
            group_1_weight_3 = v.weights()[1][3],
            group_1_pre_lower_3 = v.pre_lower()[1][3],
            group_1_pre_upper_3 = v.pre_upper()[1][3],
            group_1_neuron_4 = v.ordered_neurons()[1][4],
            group_1_weight_4 = v.weights()[1][4],
            group_1_pre_lower_4 = v.pre_lower()[1][4],
            group_1_pre_upper_4 = v.pre_upper()[1][4],
            group_1_neuron_5 = v.ordered_neurons()[1][5],
            group_1_weight_5 = v.weights()[1][5],
            group_1_pre_lower_5 = v.pre_lower()[1][5],
            group_1_pre_upper_5 = v.pre_upper()[1][5],
            group_1_neuron_6 = v.ordered_neurons()[1][6],
            group_1_weight_6 = v.weights()[1][6],
            group_1_pre_lower_6 = v.pre_lower()[1][6],
            group_1_pre_upper_6 = v.pre_upper()[1][6],
            group_1_neuron_7 = v.ordered_neurons()[1][7],
            group_1_weight_7 = v.weights()[1][7],
            group_1_pre_lower_7 = v.pre_lower()[1][7],
            group_1_pre_upper_7 = v.pre_upper()[1][7],
            before = old.0,
            after = v.bounds().0,
            raw_candidate_lower = v.raw_candidate_lower(),
            upper_unchanged = v.bounds().1.to_bits() == old.1.to_bits(),
            "Certified two-local L1 root cut completed"
        ),
        RootCompletion::LocalL1(v) => tracing::info!(
            row,
            active_len = v.active_len(),
            neuron_0 = v.ordered_neurons()[0],
            weight_0 = v.weights()[0],
            neuron_1 = v.ordered_neurons()[1],
            weight_1 = v.weights()[1],
            neuron_2 = v.ordered_neurons()[2],
            weight_2 = v.weights()[2],
            neuron_3 = v.ordered_neurons()[3],
            weight_3 = v.weights()[3],
            neuron_4 = v.ordered_neurons()[4],
            weight_4 = v.weights()[4],
            neuron_5 = v.ordered_neurons()[5],
            weight_5 = v.weights()[5],
            neuron_6 = v.ordered_neurons()[6],
            weight_6 = v.weights()[6],
            neuron_7 = v.ordered_neurons()[7],
            weight_7 = v.weights()[7],
            multiplier = v.multiplier(),
            support_bound = v.support_bound(),
            bn_radius = v.bn_radius(),
            sign_count = v.sign_count(),
            original_local_bound = v.original_local_bound(),
            pre_lower_0 = v.pre_lower()[0],
            pre_upper_0 = v.pre_upper()[0],
            pre_lower_1 = v.pre_lower()[1],
            pre_upper_1 = v.pre_upper()[1],
            pre_lower_2 = v.pre_lower()[2],
            pre_upper_2 = v.pre_upper()[2],
            pre_lower_3 = v.pre_lower()[3],
            pre_upper_3 = v.pre_upper()[3],
            pre_lower_4 = v.pre_lower()[4],
            pre_upper_4 = v.pre_upper()[4],
            pre_lower_5 = v.pre_lower()[5],
            pre_upper_5 = v.pre_upper()[5],
            pre_lower_6 = v.pre_lower()[6],
            pre_upper_6 = v.pre_upper()[6],
            pre_lower_7 = v.pre_lower()[7],
            pre_upper_7 = v.pre_upper()[7],
            before = old.0,
            after = v.bounds().0,
            upper_unchanged = v.bounds().1.to_bits() == old.1.to_bits(),
            "Certified local L1 root cut completed"
        ),
        RootCompletion::Triplet(v) => tracing::info!(
            row,
            neuron_0 = v.ordered_neurons()[0],
            neuron_1 = v.ordered_neurons()[1],
            neuron_2 = v.ordered_neurons()[2],
            weight_0 = v.weights()[0],
            weight_1 = v.weights()[1],
            weight_2 = v.weights()[2],
            multiplier = v.multiplier(),
            support_bound = v.support_bound(),
            before = old.0,
            after = v.bounds().0,
            upper_unchanged = v.bounds().1.to_bits() == old.1.to_bits(),
            "Certified triplet root cut completed"
        ),
        RootCompletion::Single(v) => {
            observation.completed_normal = Some(v.cut_normal());
            observation.completed_lambda_scale = Some(v.lambda_scale());
        }
        RootCompletion::Joint(v) => {
            observation.completed_joint_pairs = Some(v.ordered_pairs());
            observation.completed_joint_normals = Some(v.cut_normals());
            observation.completed_joint_scales = Some(v.lambda_scales());
        }
    }
    observation.stage = "execution-final-check";
    if !poll() {
        return None;
    }
    observation.stage = "completion-guard";
    let result = completed.bounds();
    // Independent final consumption guard; constructor/source authority lives
    // in the completed-result module, not in this scalar sanity check.
    if !result.0.is_finite()
        || result.0 < old.0
        || result.0 > old.1
        || result.1.to_bits() != old.1.to_bits()
        || bounds[row].0.to_bits() != old.0.to_bits()
        || bounds[row].1.to_bits() != old.1.to_bits()
    {
        return None;
    }
    observation.stage = "completion-budget";
    execution.complete().ok()?;
    setup.complete().ok()?;
    observation.stage = "completion-deadline";
    if Instant::now() >= end {
        return None;
    }
    observation.stage = "no-gain";
    if result.0 <= old.0 {
        return None;
    }
    bounds[row] = result;
    observation.stage = "accepted";
    match &completed {
        RootCompletion::TwoLocalL1(v) => tracing::info!(
            row,
            before = old.0,
            after = result.0,
            upper_unchanged = true,
            group_0_active_len = v.active_lens()[0],
            group_1_active_len = v.active_lens()[1],
            certificates = v.certificates(),
            "Certified same-source two-local L1 root cut accepted"
        ),
        RootCompletion::LocalL1(v) => tracing::info!(
            row,
            before = old.0,
            after = result.0,
            upper_unchanged = true,
            active_len = v.active_len(),
            "Certified same-source local L1 root cut accepted"
        ),
        RootCompletion::Triplet(v) => tracing::info!(
            row,
            before = old.0,
            after = result.0,
            upper_unchanged = true,
            neuron_0 = v.ordered_neurons()[0],
            neuron_1 = v.ordered_neurons()[1],
            neuron_2 = v.ordered_neurons()[2],
            "Certified same-source triplet root cut accepted"
        ),
        RootCompletion::Single(completed) => tracing::info!(
            row,
            before = old.0,
            after = result.0,
            upper_unchanged = true,
            normal_pre_0 = completed.cut_normal()[0],
            normal_pre_1 = completed.cut_normal()[1],
            normal_post_0 = completed.cut_normal()[2],
            normal_post_1 = completed.cut_normal()[3],
            lambda_scale = completed.lambda_scale(),
            "Certified same-source root cut accepted"
        ),
        RootCompletion::Joint(completed) => tracing::info!(
            row,
            before = old.0,
            after = result.0,
            upper_unchanged = true,
            pair_0_first = completed.ordered_pairs()[0][0],
            pair_0_second = completed.ordered_pairs()[0][1],
            pair_1_first = completed.ordered_pairs()[1][0],
            pair_1_second = completed.ordered_pairs()[1][1],
            lambda_0 = completed.lambda_scales()[0],
            lambda_1 = completed.lambda_scales()[1],
            "Certified same-source two-pair root cut accepted"
        ),
    }
    Some(row)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::{LinearLayer, ReLULayer};
    use crate::network::raw_root_cut::raw_root_cut_test_support::ExactToyBackend;
    use crate::GraphNode;
    use ndarray::{arr1, arr2};

    fn strict_fixture() -> (GraphNetwork, BoundedTensor, HashMap<String, BoundedTensor>) {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "pre",
            Layer::Linear(
                LinearLayer::new(
                    arr2(&[[0.5, 0.5], [0.5, -0.5], [1.0, 0.0]]),
                    Some(arr1(&[0.0, 0.0, 2.0])),
                )
                .unwrap(),
            ),
        ));
        graph.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer::new()),
            vec!["pre".into()],
        ));
        graph.add_node(GraphNode::new(
            "out",
            Layer::Linear(
                LinearLayer::new(arr2(&[[-2.0, -2.0, 1.0]]), Some(arr1(&[-0.75]))).unwrap(),
            ),
            vec!["relu".into()],
        ));
        graph.set_output("out");
        let input =
            BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn())
                .unwrap();
        let nodes = graph.collect_node_bounds(&input).unwrap();
        (graph, input, nodes)
    }

    struct RouteEngine {
        gpu: bool,
        support: bool,
        cut: bool,
    }
    impl GemmEngine for RouteEngine {
        fn gemm_f32(
            &self,
            _: usize,
            _: usize,
            _: usize,
            _: &[f32],
            _: &[f32],
        ) -> ny_core::Result<Vec<f32>> {
            panic!("route choice must not execute a contraction")
        }
        fn as_gpu_crown_backward(&self) -> Option<&dyn ny_core::GpuCrownBackward> {
            if self.gpu {
                Some(self)
            } else {
                None
            }
        }
    }
    impl ny_core::GpuCrownBackward for RouteEngine {
        fn crown_backward_gpu(
            &self,
            _: &[ny_core::GpuCrownLayer],
            _: &[f32],
            _: usize,
            _: &[f32],
            _: &[f32],
        ) -> ny_core::Result<ny_core::GpuCrownResult> {
            panic!("route choice must not execute ordinary CROWN")
        }
        fn provides_bounded_support4(&self) -> bool {
            self.support
        }
        fn provides_bounded_lower_cut(&self) -> bool {
            self.cut
        }
    }
    fn same_engine(a: &dyn GemmEngine, b: &dyn GemmEngine) -> bool {
        std::ptr::eq(
            std::ptr::from_ref::<dyn GemmEngine>(a).cast::<()>(),
            std::ptr::from_ref::<dyn GemmEngine>(b).cast::<()>(),
        )
    }
    #[test]
    fn checked_route_local_precedence_and_cpu_only_never_read_ready_slots() {
        let local = RouteEngine {
            gpu: true,
            support: true,
            cut: true,
        };
        let selected =
            select_checked_engine(Some(&local), false, || panic!("local capability wins")).unwrap();
        assert!(same_engine(selected, &local));
        assert!(select_checked_engine(Some(&local), true, || panic!(
            "host-only scope forbids lookup"
        ))
        .is_none());
    }
    #[test]
    fn checked_route_cpu_or_partial_local_uses_only_complete_ready_capabilities() {
        let cpu = RouteEngine {
            gpu: false,
            support: false,
            cut: false,
        };
        let support_only = RouteEngine {
            gpu: true,
            support: true,
            cut: false,
        };
        let cut_only = RouteEngine {
            gpu: true,
            support: false,
            cut: true,
        };
        let ready = RouteEngine {
            gpu: true,
            support: true,
            cut: true,
        };
        for local in [&cpu, &support_only, &cut_only] {
            let selected = select_checked_engine(Some(local), false, || {
                [Some(&support_only), Some(&cut_only), Some(&ready)]
            })
            .unwrap();
            assert!(same_engine(selected, &ready));
        }
        assert!(select_checked_engine(Some(&cpu), false, || [None, None, None]).is_none());
        assert!(select_checked_engine(None, false, || [
            Some(&support_only),
            Some(&cut_only),
            None
        ])
        .is_none());
    }
    #[test]
    fn cold_registration_only_schedules_raw_load_and_never_grants_checked_execution() {
        let cpu = RouteEngine {
            gpu: false,
            support: false,
            cut: false,
        };
        let wgpu = RouteEngine {
            gpu: true,
            support: false,
            cut: false,
        };
        let checked = RouteEngine {
            gpu: true,
            support: true,
            cut: true,
        };
        assert!(!raw_load_opportunity(Some(&cpu), false, false));
        assert!(!raw_load_opportunity(Some(&wgpu), false, false));
        assert!(raw_load_opportunity(Some(&cpu), true, false));
        assert!(raw_load_opportunity(None, true, false));
        assert!(raw_load_opportunity(Some(&checked), false, false));
        assert!(!raw_load_opportunity(Some(&checked), true, true));
        // The opportunity bit is absent from execution selection. A cold
        // registered factory with no published backend cannot be invoked here.
        assert!(select_checked_engine(Some(&cpu), false, || [None, None, None]).is_none());
    }
    #[test]
    fn genuine_root_host_only_scope_preserves_old_bounds_and_owner_allowance() {
        let (graph, input, nodes) = strict_fixture();
        let end = Instant::now() + Duration::from_secs(30);
        let owner = QueryOwner::new(Some(end), true).unwrap();
        let defaults = BetaCrownConfig::default();
        let config = BetaCrownConfig {
            certified_root_cut_owner: Some(Arc::clone(&owner)),
            certified_root_cut_raw_source: true,
            verification_artifact_authority: VerificationArtifactAuthority::VerdictOnly,
            ..defaults
        };
        let engine = ExactToyBackend::default();
        let mut bounds = [(-0.75, 1.25)];
        let _scope = crate::sound_f64_gemm::CpuOnlyF64Guard::new();
        assert!(try_tighten(
            &config,
            &graph,
            &input,
            &[vec![1.0]],
            &[0.0],
            &mut bounds,
            &nodes,
            None,
            Some(&engine),
            Some(end)
        )
        .is_none());
        assert_eq!(bounds, [(-0.75, 1.25)]);
        assert!(!owner.attempted.load(Ordering::Acquire));
        assert_eq!(engine.support_calls.load(Ordering::SeqCst), 0);
        assert_eq!(engine.cut_calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn query_attempt_is_shared_and_never_replenished_by_clone() {
        let end = Instant::now() + Duration::from_secs(30);
        let owner = QueryOwner::new(Some(end), true).unwrap();
        let other = Arc::clone(&owner);
        assert!(owner.claim(end).is_some());
        assert!(other.claim(end).is_none());
    }

    #[test]
    fn expired_attempt_remains_spent_and_disabled_owner_never_enters() {
        let end = Instant::now() + Duration::from_secs(30);
        let owner = QueryOwner::new(Some(end), true).unwrap();
        assert!(owner
            .claim(Instant::now().checked_sub(Duration::from_secs(1)).unwrap())
            .is_none());
        assert!(owner.claim(end).is_none());
        assert!(QueryOwner::new(Some(end), false)
            .unwrap()
            .claim(end)
            .is_none());
        assert!(QueryOwner::new(None, true).is_none());
        let too_short = Instant::now() + Duration::from_millis(500);
        assert!(QueryOwner::new(Some(too_short), true)
            .unwrap()
            .claim(too_short)
            .is_none());
        let reserved = QueryOwner::new(Some(end), true)
            .unwrap()
            .claim(end)
            .unwrap();
        assert!(reserved <= end.checked_sub(DOWNSTREAM_RESERVE).unwrap());
    }

    #[test]
    fn authentic_chain_selects_the_two_crossing_negative_outputs_and_unit_scale() {
        let (graph, input, nodes) = strict_fixture();
        let end = Instant::now() + Duration::from_secs(30);
        let owner = QueryOwner::new(Some(end), true).unwrap();
        let mut setup = owner
            .budget
            .lease(WorkRequest {
                scratch_bytes: QUOTE_SCRATCH,
                work: QUOTE_WORK,
                proposals: 1,
                ..WorkRequest::default()
            })
            .unwrap();
        let (name, pair, scale) = target(&graph, &input, &nodes, &mut setup, &[1.0], end).unwrap();
        assert_eq!(name, "relu");
        assert_eq!(pair, [0, 1]);
        assert_eq!(scale.to_bits(), 1.0f32.to_bits());
        // This is selection and original-function algebra, not backend proof.
        // F=1.25-max(|u|,|v|); its min is1/4, while independent triangles give
        // exactly-3/4. The fixed facet is max(|u|,|v|)<=1.
        for u in [-1.0f64, -0.5, 0.0, 0.5, 1.0] {
            for v in [-1.0f64, -0.5, 0.0, 0.5, 1.0] {
                let a = f64::midpoint(u, v);
                let b = (u - v) / 2.0;
                let actual = -0.75 - 2.0 * a.max(0.0) - 2.0 * b.max(0.0) + (u + 2.0);
                assert_eq!(actual, 1.25 - u.abs().max(v.abs()));
                assert!(actual >= 0.25);
                assert_eq!(-0.75 - (a + 1.0) - (b + 1.0) + (u + 2.0), -0.75);
            }
        }
    }

    #[test]
    fn folded_or_export_scope_refuses_without_consuming_an_original_query_attempt() {
        let (graph, input, nodes) = strict_fixture();
        let end = Instant::now() + Duration::from_secs(30);
        let owner = QueryOwner::new(Some(end), true).unwrap();
        let defaults = BetaCrownConfig::default();
        let mut config = BetaCrownConfig {
            certified_root_cut_owner: Some(Arc::clone(&owner)),
            verification_artifact_authority: VerificationArtifactAuthority::VerdictOnly,
            ..defaults
        };
        let mut bounds = [(-0.75, 1.25)];
        assert!(try_tighten(
            &config,
            &graph,
            &input,
            &[vec![1.0]],
            &[0.0],
            &mut bounds,
            &nodes,
            None,
            None,
            Some(end)
        )
        .is_none());
        assert_eq!(bounds, [(-0.75, 1.25)]);
        assert!(!owner.attempted.load(Ordering::Acquire));
        config.certified_root_cut_raw_source = true;
        config.verification_artifact_authority = VerificationArtifactAuthority::CertificateExport;
        assert!(try_tighten(
            &config,
            &graph,
            &input,
            &[vec![1.0]],
            &[0.0],
            &mut bounds,
            &nodes,
            None,
            None,
            Some(end)
        )
        .is_none());
        assert_eq!(bounds, [(-0.75, 1.25)]);
        assert!(!owner.attempted.load(Ordering::Acquire));
    }

    #[test]
    fn genuine_root_consumer_uses_checked_support_and_cut_then_preserves_upper() {
        let (graph, input, nodes) = strict_fixture();
        let end = Instant::now() + Duration::from_secs(30);
        let owner = QueryOwner::new(Some(end), true).unwrap();
        let defaults = BetaCrownConfig::default();
        let config = BetaCrownConfig {
            certified_root_cut_owner: Some(Arc::clone(&owner)),
            certified_root_cut_raw_source: true,
            verification_artifact_authority: VerificationArtifactAuthority::VerdictOnly,
            ..defaults
        };
        let engine = ExactToyBackend::default();
        let mut bounds = [(-0.75f32, 1.25f32)];
        let old_upper = bounds[0].1.to_bits();
        // This runs the real source selection, query-token and aggregate
        // lease, borrowed quote, exact four-row support, sealed two-cut
        // evaluation, and final consumer. The test backend interprets the
        // actual supplied descriptors/carrier; it cannot return a canned bound.
        assert_eq!(
            try_tighten(
                &config,
                &graph,
                &input,
                &[vec![1.0]],
                &[0.0],
                &mut bounds,
                &nodes,
                None,
                Some(&engine),
                Some(end)
            ),
            Some(0)
        );
        assert_eq!(bounds[0].0.to_bits(), 0.25f32.to_bits());
        assert_eq!(bounds[0].1.to_bits(), old_upper);
        assert_eq!(engine.support_calls.load(Ordering::SeqCst), 1);
        assert_eq!(engine.cut_calls.load(Ordering::SeqCst), 2);
        assert!(owner.attempted.load(Ordering::Acquire));
        // A later participating call with a still-unresolved row and the
        // same query owner cannot acquire a second proposal allowance.
        let mut next_bounds = [(-0.75f32, 1.25f32)];
        assert!(try_tighten(
            &config,
            &graph,
            &input,
            &[vec![1.0]],
            &[0.0],
            &mut next_bounds,
            &nodes,
            None,
            Some(&engine),
            Some(end)
        )
        .is_none());
        assert_eq!(next_bounds, [(-0.75, 1.25)]);
        assert_eq!(engine.support_calls.load(Ordering::SeqCst), 1);
        assert_eq!(engine.cut_calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn genuine_root_consumer_discards_a_successful_partial_cut_on_late_refusal() {
        let (graph, input, nodes) = strict_fixture();
        let end = Instant::now() + Duration::from_secs(30);
        let owner = QueryOwner::new(Some(end), true).unwrap();
        let defaults = BetaCrownConfig::default();
        let config = BetaCrownConfig {
            certified_root_cut_owner: Some(Arc::clone(&owner)),
            certified_root_cut_raw_source: true,
            verification_artifact_authority: VerificationArtifactAuthority::VerdictOnly,
            ..defaults
        };
        let engine = ExactToyBackend::default();
        engine.refuse_cut.store(2, Ordering::SeqCst);
        let mut bounds = [(-0.75f32, 1.25f32)];
        assert!(try_tighten(
            &config,
            &graph,
            &input,
            &[vec![1.0]],
            &[0.0],
            &mut bounds,
            &nodes,
            None,
            Some(&engine),
            Some(end)
        )
        .is_none());
        assert_eq!(bounds, [(-0.75, 1.25)]);
        assert_eq!(engine.support_calls.load(Ordering::SeqCst), 1);
        assert_eq!(engine.cut_calls.load(Ordering::SeqCst), 2);
        assert!(owner.attempted.load(Ordering::Acquire));
        assert!(owner.claim(end).is_none());
    }

    #[test]
    fn genuine_root_consumer_retains_fresh_stability_and_once_only_custody() {
        use crate::network::raw_root_cut::raw_root_cut_test_support::fresh_stable_fixture;
        for margin in [0.25f32, -0.25] {
            let (graph, input, nodes, old) = fresh_stable_fixture(0.5, -0.25, 0., margin);
            let end = Instant::now() + Duration::from_secs(30);
            let owner = QueryOwner::new(Some(end), true).unwrap();
            let defaults = BetaCrownConfig::default();
            let config = BetaCrownConfig {
                certified_root_cut_owner: Some(Arc::clone(&owner)),
                certified_root_cut_raw_source: true,
                verification_artifact_authority: VerificationArtifactAuthority::VerdictOnly,
                ..defaults
            };
            let engine = ExactToyBackend::default();
            let mut bounds = [old];
            // The genuine target selector sees two crossing original cells.
            // Exact support stabilizes them; the same owner must now retain
            // that information through certification, both folds and consume.
            assert_eq!(
                try_tighten(
                    &config,
                    &graph,
                    &input,
                    &[vec![1.]],
                    &[0.],
                    &mut bounds,
                    &nodes,
                    None,
                    Some(&engine),
                    Some(end),
                ),
                Some(0)
            );
            assert_eq!(bounds[0].0.to_bits(), margin.to_bits());
            assert_eq!(bounds[0].1.to_bits(), old.1.to_bits());
            assert_eq!(engine.support_calls.load(Ordering::SeqCst), 1);
            assert_eq!(engine.cut_calls.load(Ordering::SeqCst), 2);
            assert!(owner.attempted.load(Ordering::Acquire));
            let mut second = [old];
            assert!(try_tighten(
                &config,
                &graph,
                &input,
                &[vec![1.]],
                &[0.],
                &mut second,
                &nodes,
                None,
                Some(&engine),
                Some(end),
            )
            .is_none());
            assert_eq!(second, [old]);
            assert_eq!(engine.support_calls.load(Ordering::SeqCst), 1);
            assert_eq!(engine.cut_calls.load(Ordering::SeqCst), 2);
        }
    }
    #[test]
    fn head_box_floor_original_root_owner_admits_fresh_box_proof_and_counts_only_executed_prefixes()
    {
        use crate::network::raw_root_cut::raw_two_local_l1_root_cut_test_support::{
            baseline, full_head_fixture, ComparisonBackend,
        };
        for shift in [0., -23. / 16.] {
            let (graph, input, nodes) = full_head_fixture(shift);
            let end = Instant::now() + Duration::from_secs(30);
            let old = baseline(&graph, &input, &nodes, end);
            let owner = QueryOwner::new(Some(end), true).unwrap();
            let defaults = BetaCrownConfig::default();
            let config = BetaCrownConfig {
                certified_root_cut_owner: Some(Arc::clone(&owner)),
                certified_root_cut_raw_source: true,
                verification_artifact_authority: VerificationArtifactAuthority::VerdictOnly,
                ..defaults
            };
            let engine = ComparisonBackend::default();
            let mut bounds = [old];
            assert_eq!(
                try_tighten(
                    &config,
                    &graph,
                    &input,
                    &[vec![1.]],
                    &[0.],
                    &mut bounds,
                    &nodes,
                    None,
                    Some(&engine),
                    Some(end)
                ),
                Some(0)
            );
            assert!(bounds[0].0 > old.0);
            assert_eq!(bounds[0].1.to_bits(), old.1.to_bits());
            if shift == 0. {
                assert!(bounds[0].0 > 0.);
                assert_eq!(engine.entered.load(Ordering::SeqCst), 0);
            } else {
                assert!(bounds[0].0 < 0.);
            }
            assert!(owner.attempted.load(Ordering::Acquire));
            assert!(owner.claim(end).is_none());
        }
    }
}
