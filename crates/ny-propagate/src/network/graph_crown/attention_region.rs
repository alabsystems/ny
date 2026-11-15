// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Proof-only composition of an original decomposed attention contribution.
//!
//! The forward graph is never rewritten. A private, borrowed descriptor binds
//! the current graph, original edges, scale bits and current endpoint boxes.
//! Only the incoming PV contribution is replaced: independently pending score,
//! probability, transpose and residual contributions remain in the walk.
//!
//! This ledger prices incremental region work. Other graph allocations retain
//! their existing admission policy; it is not a whole-process memory receipt.

use std::cell::RefCell;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::mem::size_of;
use std::rc::Rc;
use std::time::{Duration, Instant};

use ndarray::Array1;
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;

use crate::bounds::{BatchedLinearBounds, LinearBounds};
#[cfg(test)]
use crate::certify::work::WorkReceipt;
use crate::certify::work::{WorkBudget, WorkLease, WorkRequest};
use crate::layers::{AttentionMask, Layer, SelfAttentionLayer};
use crate::network::core::graph::batched_accumulator::BatchedCrownAccumulator;
use crate::network::crown_memory::cpu_crown_dense_budget_bytes;
use crate::network::{CrownMergeAccumulator, GraphNetwork, GraphNode, NETWORK_INPUT};

mod consume;
mod global;
mod recognize;
#[cfg(test)]
mod tests;

const MAX_REGIONS: usize = 64;
const MAX_ATTEMPTS: usize = 256;
const MAX_GLOBAL_STEPS: usize = 64;
const MAX_BYTES: usize = 512 * 1024 * 1024;
const MAX_WORK: u64 = 10_000_000_000;
const HEADER_BYTES: usize = 16 * 1024;
const RECOGNITION_WORK: u64 = 128 * 1024;

/// Transferable resource context for a worker participating in the SAME
/// region query. Equal numeric limits alone never authorize a fresh ledger.
#[derive(Clone)]
pub(crate) struct AttentionRegionAuthority {
    budget: Option<WorkBudget>,
    cap: usize,
    deadline: Option<Instant>,
}

thread_local! {
    static REGION_AUTHORITY: RefCell<Option<AttentionRegionAuthority>> = const { RefCell::new(None) };
}

/// TLS bridges old entrypoint signatures only. It conveys resource counters,
/// never graph scope, proof eligibility or certificate authority. The guard is
/// deliberately !Send: workers must explicitly clone/install the authority.
pub(crate) struct AttentionRegionScope {
    previous: Option<AttentionRegionAuthority>,
    authority: AttentionRegionAuthority,
    _thread_bound: PhantomData<Rc<()>>,
}

impl AttentionRegionScope {
    pub(crate) fn enter(deadline: Option<Instant>) -> Self {
        let inherited = REGION_AUTHORITY.with(|slot| slot.borrow().clone());
        let mut authority = inherited.unwrap_or_else(|| {
            let cap = cpu_crown_dense_budget_bytes().min(MAX_BYTES);
            let deadline = earlier(deadline, ny_core::instance_budget::deadline());
            let budget = WorkBudget::new(
                deadline,
                cap,
                cap,
                MAX_WORK,
                MAX_REGIONS as u64,
                Duration::ZERO,
            )
            .ok();
            AttentionRegionAuthority {
                budget,
                cap,
                deadline,
            }
        });
        // Restrict the transferable context too: workers must not lose an
        // earlier explicit nested deadline, while sharing the ORIGINAL ledger.
        authority.deadline = earlier(
            authority.deadline,
            earlier(deadline, ny_core::instance_budget::deadline()),
        );
        Self::install(authority)
    }

    pub(crate) fn authority(&self) -> AttentionRegionAuthority {
        self.authority.clone()
    }

    /// Explicit worker seam; new region kernels themselves spawn no workers.
    /// A future parallel dispatcher must install this SAME context in each
    /// worker, rather than relying on TLS to inherit across thread creation.
    pub(crate) fn install(authority: AttentionRegionAuthority) -> Self {
        let previous = REGION_AUTHORITY.with(|slot| slot.replace(Some(authority.clone())));
        Self {
            previous,
            authority,
            _thread_bound: PhantomData,
        }
    }
}

impl Drop for AttentionRegionScope {
    fn drop(&mut self) {
        REGION_AUTHORITY.with(|slot| {
            slot.replace(self.previous.take());
        });
    }
}

/// Observed route counters from the default decomposed-attention backward walk.
///
/// These counters describe completed coefficient propagation, not a certificate,
/// independent replay, solve result, or whole-process memory measurement. Work
/// and bytes remain separately admitted predictions. A public batched receipt
/// covers its batched attempt; the ordinary DAG fallback retains provenance.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AttentionRegionReceipt {
    pub attempted: usize,
    pub recognized: usize,
    pub checked: usize,
    pub composed: usize,
    pub declined: usize,
    pub scalar: usize,
    pub batched: usize,
    pub rows: usize,
    pub returned_coefficient_elements: usize,
    /// Completed original authored-Kt regions; these are storage observations.
    pub authored_k_direct_regions: usize,
    /// Logical endpoint copy cells elided relative to the original transpose
    /// adapter, not measured resident bytes or elapsed time.
    pub elided_k_endpoint_copy_elements: usize,
    /// Logical nominal coefficient copy cells elided by direct original writes.
    pub elided_k_coefficient_copy_elements: usize,
}

/// One allowance shared by every region in this backward query. Optional
/// refusal always leaves the original dispatch available. Completed carrier
/// leases conservatively remain live until the entire query drops.
pub(crate) struct AttentionRegionQuery {
    budget: Option<WorkBudget>,
    cap: usize,
    held: Vec<WorkLease>,
    receipt: AttentionRegionReceipt,
    deadline: Option<Instant>,
    _scope: Option<AttentionRegionScope>,
    global_steps: usize,
    global_step_limit: usize,
    held_slots: usize,
}

/// Complete future control storage, derived before allocating any lease slot.
/// This changes bookkeeping capacity only; every numerical step still admits
/// its original bytes/work on the same query ledger.
#[derive(Clone, Copy)]
struct HeldPlan {
    global_steps: usize,
    slots: usize,
    bytes: usize,
    work: u64,
}

impl HeldPlan {
    fn new(global_steps: usize, graph_derived: bool) -> Option<Self> {
        let slots = MAX_REGIONS.checked_add(global_steps)?.checked_add(1)?;
        let bytes = slots
            .checked_mul(size_of::<WorkLease>())?
            .checked_add(Self::owner_bytes()?)?
            .max(HEADER_BYTES);
        // Conservatively prepay the future control pushes. The legacy
        // constructor retains its established zero-work header receipt.
        let work = if graph_derived {
            u64::try_from(slots).ok()?.checked_mul(2)?
        } else {
            0
        };
        Some(Self {
            global_steps,
            slots,
            bytes,
            work,
        })
    }

    fn owner_bytes() -> Option<usize> {
        size_of::<AttentionRegionQuery>()
            .checked_add(size_of::<HeldPlan>().checked_mul(2)?)?
            .checked_add(256)
    }

    fn actual_bytes(self, capacity: usize) -> Option<usize> {
        capacity
            .checked_mul(size_of::<WorkLease>())?
            .checked_add(Self::owner_bytes()?)
    }
}

pub(super) struct NamedCandidate<'a> {
    sources: [&'a str; 3],
    bounds: [LinearBounds; 3],
    lower_bias: Array1<f32>,
    upper_bias: Array1<f32>,
}

impl AttentionRegionQuery {
    /// Scalar/spec/alpha walks use only the bounded attention-region adapter;
    /// they do not call the global unary corridor and retain this constructor.
    pub(crate) fn new(deadline: Option<Instant>) -> Self {
        let scope = AttentionRegionScope::enter(deadline);
        let authority = scope.authority();
        Self::from_authority(
            authority,
            Some(scope),
            earlier(deadline, ny_core::instance_budget::deadline()),
        )
    }

    /// A normal batched reverse DAG walk visits each original node at most
    /// once. Its immutable node count bounds successful global unary steps.
    /// The count conveys resource geometry, never graph/proof authority.
    pub(crate) fn new_for_graph(graph: &GraphNetwork, deadline: Option<Instant>) -> Self {
        let scope = AttentionRegionScope::enter(deadline);
        let authority = scope.authority();
        Self::from_authority_with_graph_bound(
            authority,
            Some(scope),
            earlier(deadline, ny_core::instance_budget::deadline()),
            Some(graph.num_nodes()),
        )
    }

    #[cfg(test)]
    fn with_limits(deadline: Option<Instant>, cap: usize, work: u64) -> Self {
        let budget =
            WorkBudget::new(deadline, cap, cap, work, MAX_REGIONS as u64, Duration::ZERO).ok();
        Self::from_authority(
            AttentionRegionAuthority {
                budget,
                cap,
                deadline,
            },
            None,
            deadline,
        )
    }

    fn from_authority(
        authority: AttentionRegionAuthority,
        scope: Option<AttentionRegionScope>,
        deadline: Option<Instant>,
    ) -> Self {
        Self::from_authority_with_graph_bound(authority, scope, deadline, None)
    }

    fn from_authority_with_graph_bound(
        authority: AttentionRegionAuthority,
        scope: Option<AttentionRegionScope>,
        deadline: Option<Instant>,
        graph_nodes: Option<usize>,
    ) -> Self {
        let deadline = earlier(deadline, authority.deadline);
        let mut query = Self {
            budget: authority.budget,
            cap: authority.cap,
            held: Vec::new(),
            receipt: AttentionRegionReceipt::default(),
            deadline,
            _scope: scope,
            global_steps: 0,
            global_step_limit: MAX_GLOBAL_STEPS,
            held_slots: 0,
        };
        let Some(budget) = query.budget.clone() else {
            return query;
        };
        let graph_plan = graph_nodes.and_then(|n| HeldPlan::new(n.max(MAX_GLOBAL_STEPS), true));
        let legacy_plan = HeldPlan::new(MAX_GLOBAL_STEPS, false);
        // A failed graph candidate drops BOTH vector and lease inside
        // allocate_header before the paid legacy candidate starts. There is
        // no in-place growth, old/new allocation overlap or fresh allowance.
        for plan in [graph_plan, legacy_plan].into_iter().flatten() {
            match Self::allocate_header(&budget, query.cap, deadline, plan) {
                Ok(held) => {
                    query.held = held;
                    query.held_slots = plan.slots;
                    query.global_step_limit = plan.global_steps;
                    return query;
                }
                Err(error) => {
                    tracing::debug!(
                        global_step_limit = plan.global_steps,
                        header_bytes = plan.bytes,
                        %error,
                        "attention query control admission declined; same-ledger bounded fallback retained"
                    );
                }
            }
        }
        query.budget = None;
        query
    }

    fn allocate_header(
        budget: &WorkBudget,
        cap: usize,
        deadline: Option<Instant>,
        plan: HeldPlan,
    ) -> Result<Vec<WorkLease>> {
        poll(deadline, budget)?;
        let live = budget.receipt().map_err(refusal)?;
        let peak = live
            .live_scratch_bytes
            .checked_add(live.live_persistent_bytes)
            .and_then(|n| n.checked_add(plan.bytes))
            .unwrap_or(usize::MAX);
        if peak > cap.min(cpu_crown_dense_budget_bytes()) {
            return Err(NyError::UnsupportedOp(
                "attention query control memory admission".into(),
            ));
        }
        let mut header = budget
            .lease(WorkRequest {
                persistent_bytes: plan.bytes,
                work: plan.work,
                ..WorkRequest::default()
            })
            .map_err(refusal)?;
        poll(deadline, budget)?;
        let mut held = Vec::new();
        held.try_reserve_exact(plan.slots)
            .map_err(|_| NyError::UnsupportedOp("attention query control allocation".into()))?;
        poll(deadline, budget)?;
        if held.capacity() < plan.slots
            || plan
                .actual_bytes(held.capacity())
                .is_none_or(|n| n > plan.bytes)
        {
            return Err(NyError::UnsupportedOp(
                "attention query control allocator exceeded admission".into(),
            ));
        }
        header.mark_materialized().map_err(refusal)?;
        poll(deadline, budget)?;
        held.push(header);
        Ok(held)
    }

    fn has_held_slot(&self) -> bool {
        self.held.len() < self.held_slots && self.held.len() < self.held.capacity()
    }

    fn hold_completed(&mut self, lease: WorkLease) -> Result<()> {
        if !self.has_held_slot() {
            return Err(NyError::UnsupportedOp(
                "attention query prepaid control slots exhausted".into(),
            ));
        }
        self.held.push(lease);
        Ok(())
    }

    pub(crate) fn receipt(&self) -> AttentionRegionReceipt {
        self.receipt
    }

    #[cfg(test)]
    fn work_receipt(&self) -> WorkReceipt {
        self.budget
            .as_ref()
            .and_then(|b| b.receipt().ok())
            .unwrap_or_default()
    }

    fn begin_attempt(&mut self, node: &GraphNode) -> bool {
        if !matches!(node.layer, Layer::MatMul(_) | Layer::BilinearCrown(_))
            || !crate::network::backward_dispatch::attn_crown_ternary_enabled()
        {
            return false;
        }
        if self.budget.is_none()
            || self.receipt.attempted >= MAX_ATTEMPTS
            || self.receipt.checked >= MAX_REGIONS
        {
            self.receipt.declined = self.receipt.declined.saturating_add(1);
            return false;
        }
        self.receipt.attempted += 1;
        true
    }

    fn candidate<'a>(
        &mut self,
        graph: &'a GraphNetwork,
        node: &'a GraphNode,
        incoming: &LinearBounds,
        input: &'a BoundedTensor,
        boxes: &'a HashMap<String, BoundedTensor>,
        frontier_bytes: usize,
        deadline: Option<Instant>,
    ) -> Result<Option<NamedCandidate<'a>>> {
        let deadline = earlier(deadline, self.deadline);
        if !self.begin_attempt(node) {
            return Ok(None);
        }
        let Some(budget) = self.budget.clone() else {
            return Ok(None);
        };
        // All recognized names/ranks are bounded before their fixed number of
        // lookups; this reservation pays even for a rejected topology.
        let recognition = budget.lease(WorkRequest {
            work: RECOGNITION_WORK,
            ..WorkRequest::default()
        });
        let Ok(mut recognition) = recognition else {
            self.receipt.declined += 1;
            return Ok(None);
        };
        let result = (|| {
            poll(deadline, &budget)?;
            let Some(region) = recognize::recognize(graph, node, input, boxes)? else {
                return Ok(None);
            };
            self.receipt.recognized += 1;
            self.compose_region(
                region,
                incoming,
                input.len(),
                frontier_bytes,
                deadline,
                &budget,
            )
        })();
        // A failed optional candidate cannot change the old accumulator.
        // The surrounding original walk retains its own deadline/fallback.
        match result {
            Ok(result) => {
                if recognition.complete().is_err() {
                    self.receipt.declined += 1;
                    return Ok(None);
                }
                if result.is_none() {
                    self.receipt.declined += 1;
                }
                Ok(result)
            }
            Err(error) => {
                self.receipt.declined += 1;
                tracing::debug!(node = node.name(), %error, "attention region declined; original dispatch retained");
                Ok(None)
            }
        }
    }

    // Scalar and batched adapters share this already-recognized checked path.
    // Recognition, attempt admission and optional fallback belong to callers.
    fn compose_region<'a>(
        &mut self,
        region: recognize::Region<'a>,
        incoming: &LinearBounds,
        input_len: usize,
        frontier_bytes: usize,
        deadline: Option<Instant>,
        budget: &WorkBudget,
    ) -> Result<Option<NamedCandidate<'a>>> {
        if self.receipt.checked >= MAX_REGIONS || !self.has_held_slot() {
            return Ok(None);
        }
        if incoming.num_inputs() != region.output_len {
            return Ok(None);
        }
        incoming.validate_internal_shapes()?;
        let rows = incoming.num_outputs();
        let request = region.request(rows, input_len, incoming.memory_bytes())?;
        self.admit_peak(frontier_bytes, request, region.kernel_scratch(rows)?)?;
        let mut lease = budget.lease(request).map_err(refusal)?;
        poll(deadline, budget)?;
        // LinearBounds exposes mutable biases and the ordinary accumulation
        // profile permits signed infinities. Reject NaN on this original lease
        // before invoking the expensive optional kernel.
        consume::validate_biases(incoming.lower_b(), incoming.upper_b(), deadline, budget)?;
        let attn = SelfAttentionLayer::new(AttentionMask::Standard, Some(region.scale));
        let (mut outputs, lower_bias, upper_bias) = if region.authored_kt {
            attn.propagate_crown_ternary_with_authored_kt_and_budget_and_deadline(
                incoming, region.q, region.k, region.v, budget, deadline,
            )?
        } else {
            attn.propagate_crown_ternary_with_budget_and_deadline(
                incoming, region.q, region.k, region.v, budget, deadline,
            )?
        };
        consume::validate_biases(&lower_bias, &upper_bias, deadline, budget)?;
        if outputs.len() != 3 {
            return Err(NyError::InternalError(
                "attention region returned wrong arity".into(),
            ));
        }
        let v = outputs
            .pop()
            .flatten()
            .ok_or_else(|| NyError::InternalError("attention region missing V".into()))?;
        let k = outputs
            .pop()
            .flatten()
            .ok_or_else(|| NyError::InternalError("attention region missing K".into()))?;
        let q = outputs
            .pop()
            .flatten()
            .ok_or_else(|| NyError::InternalError("attention region missing Q".into()))?;
        for bounds in [&q, &k, &v] {
            consume::validate_linear(bounds, deadline, budget)?;
            GraphNetwork::verify_split_path_bias_zero(bounds, "attention region source")?;
        }
        poll(deadline, budget)?;
        lease.complete().map_err(refusal)?;
        // No fresh allowance after a declined region or fallback.
        self.hold_completed(lease)?;
        self.receipt.checked += 1;
        self.receipt.rows = self.receipt.rows.saturating_add(rows);
        if region.authored_kt {
            self.receipt.authored_k_direct_regions =
                self.receipt.authored_k_direct_regions.saturating_add(1);
            self.receipt.elided_k_endpoint_copy_elements = self
                .receipt
                .elided_k_endpoint_copy_elements
                .saturating_add(region.k.len().saturating_mul(2));
            self.receipt.elided_k_coefficient_copy_elements = self
                .receipt
                .elided_k_coefficient_copy_elements
                .saturating_add(rows.saturating_mul(region.k.len()).saturating_mul(2));
        }
        self.receipt.returned_coefficient_elements =
            self.receipt.returned_coefficient_elements.saturating_add(
                rows.saturating_mul(
                    region
                        .q
                        .len()
                        .saturating_add(region.k.len())
                        .saturating_add(region.v.len()),
                )
                .saturating_mul(2),
            );
        Ok(Some(NamedCandidate {
            sources: region.sources,
            bounds: [q, k, v],
            lower_bias,
            upper_bias,
        }))
    }

    fn admit_peak(&self, frontier: usize, request: WorkRequest, kernel: usize) -> Result<()> {
        let live = self
            .budget
            .as_ref()
            .and_then(|b| b.receipt().ok())
            .map_or(usize::MAX, |r| {
                r.live_scratch_bytes.saturating_add(r.live_persistent_bytes)
            });
        // Frontier includes independently pending paths. The conservative merge
        // overlap is part of request.scratch_bytes, not a cap enlargement.
        let peak = frontier
            .checked_add(live)
            .and_then(|n| n.checked_add(request.scratch_bytes))
            .and_then(|n| n.checked_add(request.persistent_bytes))
            .and_then(|n| n.checked_add(kernel))
            .unwrap_or(usize::MAX);
        if peak > self.cap.min(cpu_crown_dense_budget_bytes()) {
            return Err(NyError::UnsupportedOp(
                "attention region whole-frontier memory admission".into(),
            ));
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn try_scalar(
        &mut self,
        graph: &GraphNetwork,
        node: &GraphNode,
        incoming: &LinearBounds,
        input: &BoundedTensor,
        boxes: &HashMap<String, BoundedTensor>,
        accumulator: &mut CrownMergeAccumulator,
        rows: usize,
        input_dim: usize,
        input_accumulated: &mut bool,
        deadline: Option<Instant>,
    ) -> Result<bool> {
        let deadline = earlier(deadline, self.deadline);
        if rows != incoming.num_outputs() || input_dim != input.len() {
            return Ok(false);
        }
        let Some(candidate) = self.candidate(
            graph,
            node,
            incoming,
            input,
            boxes,
            accumulator.logical_frontier_payload_bytes(),
            deadline,
        )?
        else {
            return Ok(false);
        };
        if self
            .budget
            .as_ref()
            .is_none_or(|budget| poll(deadline, budget).is_err())
        {
            return Ok(false);
        }
        // All source bindings and complete carriers were checked before this
        // publication. A merge refusal aborts the complete walk; no partial
        // candidate is ever retried as if it were the original PV contribution.
        GraphNetwork::accumulate_bias_to_network_input_crown_with_deadline(
            &candidate.lower_bias,
            &candidate.upper_bias,
            accumulator,
            rows,
            input_dim,
            input_accumulated,
            deadline,
        )?;
        for (source, bounds) in candidate.sources.into_iter().zip(candidate.bounds) {
            graph.accumulate_dense_bounds_to_input_with_deadline(
                source,
                bounds,
                accumulator,
                rows,
                input_dim,
                input_accumulated,
                deadline,
            )?;
        }
        self.receipt.scalar += 1;
        self.receipt.composed += 1;
        Ok(true)
    }

    pub(crate) fn try_batched(
        &mut self,
        graph: &GraphNetwork,
        node: &GraphNode,
        incoming: &BatchedLinearBounds,
        input: &BoundedTensor,
        boxes: &HashMap<String, BoundedTensor>,
        accumulator: &mut BatchedCrownAccumulator,
        deadline: Option<Instant>,
    ) -> Result<bool> {
        consume::batched(
            self,
            graph,
            node,
            incoming,
            input,
            boxes,
            accumulator,
            deadline,
        )
    }

    /// Original full-column affine/reduction corridor. This shares the SAME
    /// query ledger with attention; it grants no new graph or proof authority.
    pub(crate) fn try_global_unary(
        &mut self,
        node: &GraphNode,
        incoming: &BatchedLinearBounds,
        input: &BoundedTensor,
        boxes: &HashMap<String, BoundedTensor>,
        accumulator: &mut BatchedCrownAccumulator,
    ) -> Result<bool> {
        global::propagate(self, node, incoming, input, boxes, accumulator)
    }
}

impl Drop for AttentionRegionQuery {
    fn drop(&mut self) {
        if self.receipt.attempted != 0 {
            let work = self
                .budget
                .as_ref()
                .and_then(|b| b.receipt().ok())
                .unwrap_or_default();
            tracing::debug!(
                attempted = self.receipt.attempted,
                recognized = self.receipt.recognized,
                composed = self.receipt.composed,
                declined = self.receipt.declined,
                scalar = self.receipt.scalar,
                batched = self.receipt.batched,
                rows = self.receipt.rows,
                returned_coefficient_elements = self.receipt.returned_coefficient_elements,
                authored_k_direct_regions = self.receipt.authored_k_direct_regions,
                elided_k_endpoint_copy_elements = self.receipt.elided_k_endpoint_copy_elements,
                elided_k_coefficient_copy_elements =
                    self.receipt.elided_k_coefficient_copy_elements,
                work_reserved = work.work_reserved,
                peak_leased_bytes = work.peak_leased_bytes,
                "attention region query receipt"
            );
        }
    }
}

fn earlier(a: Option<Instant>, b: Option<Instant>) -> Option<Instant> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    }
}

fn poll(deadline: Option<Instant>, budget: &WorkBudget) -> Result<()> {
    if earlier(deadline, ny_core::instance_budget::deadline()).is_some_and(|d| Instant::now() >= d)
    {
        return Err(NyError::DeadlineExceeded(
            "attention region deadline".into(),
        ));
    }
    budget.checkpoint().map_err(refusal)
}

fn refusal(reason: crate::certify::work::WorkRefusal) -> NyError {
    if reason == crate::certify::work::WorkRefusal::Deadline {
        NyError::DeadlineExceeded("attention region work authority".into())
    } else {
        NyError::UnsupportedOp(format!("attention region admission: {reason}"))
    }
}
