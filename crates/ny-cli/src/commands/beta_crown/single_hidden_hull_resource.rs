// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Exact original FC ingress using the caller's existing E1 ledger. No
//! folding, LP session, fresh budget or global mode change is hidden inside
//! this profile. The original five-node MatMul/bias topology borrows authored
//! F32 constants directly. Other folded/Conv models require a priced adapter.

use super::*;
#[cfg(test)]
use ny_mip::single_hidden_hull::resource::HullQueryRefusal;
use ny_mip::single_hidden_hull::resource::{
    checked_hull_triangle, import_hull_original, replay_hull_original, HullBiasData,
    HullCheckedCertificate, HullLeaseClass, HullOriginalData, HullProofOrigin, HullQueryAuthority,
    HullQueryContext, HullResourceLease, HullResourceRefusal, HullRetainedResource,
    HullWorkRequest,
};
use ny_propagate::certify::work::{
    LeaseClass, PersistentReservation, WorkBudget, WorkLease, WorkRefusal, WorkRequest,
};
use std::sync::Arc;

fn refusal(value: WorkRefusal) -> HullResourceRefusal {
    match value {
        WorkRefusal::Deadline => HullResourceRefusal::Deadline,
        WorkRefusal::Cancelled => HullResourceRefusal::Cancelled,
        WorkRefusal::ScratchLimit | WorkRefusal::PersistentLimit | WorkRefusal::MemoryLimit => {
            HullResourceRefusal::Memory
        }
        WorkRefusal::WorkLimit | WorkRefusal::ProposalLimit => HullResourceRefusal::Work,
        WorkRefusal::Overflow => HullResourceRefusal::Overflow,
        WorkRefusal::Unavailable => HullResourceRefusal::Unavailable,
    }
}
fn class(value: HullLeaseClass) -> LeaseClass {
    match value {
        HullLeaseClass::Optional => LeaseClass::Optional,
        HullLeaseClass::Replay => LeaseClass::Replay,
    }
}
struct BudgetAuthority {
    budget: WorkBudget,
    deadline: Instant,
}
struct Lease {
    lease: WorkLease,
    deadline: Instant,
}
struct Claim {
    _reservation: PersistentReservation,
}
impl HullRetainedResource for Claim {}
impl HullResourceLease for Lease {
    fn checkpoint(&self) -> Result<(), HullResourceRefusal> {
        self.lease.checkpoint().map_err(refusal)?;
        if Instant::now() >= self.deadline {
            return Err(HullResourceRefusal::Deadline);
        }
        Ok(())
    }
    fn retain(self: Box<Self>) -> Result<Arc<dyn HullRetainedResource>, HullResourceRefusal> {
        self.checkpoint()?;
        let Self { lease, .. } = *self;
        Ok(Arc::new(Claim {
            _reservation: lease.commit_persistent().map_err(refusal)?,
        }))
    }
}
impl HullQueryAuthority for BudgetAuthority {
    fn lease(
        &self,
        phase: HullLeaseClass,
        request: HullWorkRequest,
    ) -> Result<Box<dyn HullResourceLease>, HullResourceRefusal> {
        self.checkpoint(phase)?;
        let mut lease = self
            .budget
            .lease_for(
                class(phase),
                WorkRequest {
                    scratch_bytes: request.scratch_bytes,
                    persistent_bytes: request.persistent_bytes,
                    work: request.work,
                    proposals: request.proposals,
                },
            )
            .map_err(refusal)?;
        // The bridge immediately commits these admitted capacities to its
        // original/scratch owners. Materialized means committed capacity,
        // never physically touched pages or measured RSS.
        lease.mark_materialized().map_err(refusal)?;
        Ok(Box::new(Lease {
            lease,
            deadline: self.deadline,
        }))
    }
    fn checkpoint(&self, phase: HullLeaseClass) -> Result<(), HullResourceRefusal> {
        self.budget.checkpoint_for(class(phase)).map_err(refusal)?;
        if Instant::now() >= self.deadline {
            return Err(HullResourceRefusal::Deadline);
        }
        Ok(())
    }
}

/// One adapter per caller query; clones of this context share both its
/// identity and the original WorkBudget/cancellation/memory authority.
pub(in crate::commands::beta_crown) fn hull_context_for_budget(
    budget: &WorkBudget,
    deadline: Instant,
) -> HullQueryContext {
    // Pass the existing hull_root_deadline slice; neither this restriction nor
    // the adapter can extend the caller ledger's own class/instance deadline.
    HullQueryContext::new(BudgetAuthority {
        budget: budget.clone(),
        deadline,
    })
}

/// Authenticate one authored bias vector against the positionwise Linear's
/// one-vector shape. Scalar or non-singleton broadcasting changes this profile
/// and declines. No values are scanned or copied before the importer leases
/// the original parameter work and storage.
fn authored_bias(
    layer: &ny_propagate::layers::AddConstantLayer,
    width: usize,
    vector_rank: usize,
) -> Result<HullBiasData<'_>, &'static str> {
    let constant = layer.constant();
    let shape = constant.shape();
    if shape.is_empty()
        || shape.len() > vector_rank
        || shape.last().copied() != Some(width)
        || constant.len() != width
        || shape[..shape.len() - 1].iter().any(|&dim| dim != 1)
    {
        return Err("not_one_original_bias_vector");
    }
    Ok(HullBiasData::Binary32(
        constant.as_slice().ok_or("nonstandard_original_bias")?,
    ))
}

/// Original sequential FC layout only. Ordered Network layers are unary edges
/// in this profile; binary/dynamic operators and extra views never match it.
/// Shape/spec admission precedes imports. Metadata checks are bounded to eight
/// singleton axes; coefficient scans/copies run only under the importer lease.
fn data<'a>(
    network: &'a Network,
    input: &'a BoundedTensor,
    vnnlib: &VnnLibSpec,
    objective: &'a [(usize, f64)],
) -> Result<HullOriginalData<'a>, &'static str> {
    if vnnlib.is_disjunction {
        return Err("disjunctive_spec");
    }
    if vnnlib.dual_network.is_some() {
        return Err("dual_network_spec");
    }
    if vnnlib
        .per_clause_input_bounds
        .iter()
        .any(|clause| !clause.is_empty())
    {
        return Err("per_clause_input_box");
    }
    let vector_shape = input.shape();
    if vector_shape.is_empty()
        || vector_shape.len() > 8
        || vector_shape[..vector_shape.len() - 1]
            .iter()
            .any(|&dim| dim != 1)
    {
        return Err("not_one_original_input_vector");
    }
    let (first, last, authored) = match network.layers() {
        [Layer::Linear(first), Layer::ReLU(_), Layer::Linear(last)] => (first, last, None),
        [Layer::Linear(first), Layer::AddConstant(b1), Layer::ReLU(_), Layer::Linear(last), Layer::AddConstant(b2)] =>
        {
            if first.bias().is_some() || last.bias().is_some() {
                return Err("original_bias_already_present");
            }
            (first, last, Some((b1, b2)))
        }
        _ => return Err("unpriced_canonicalization"),
    };
    let (hidden_dim, input_dim) = first.weight().dim();
    let (output_dim, last_input) = last.weight().dim();
    if last_input != hidden_dim
        || input.lower().len() != input_dim
        || input.upper().len() != input_dim
        || vnnlib.num_inputs != input_dim
        || vnnlib.num_outputs != output_dim
    {
        return Err("dimension_mismatch");
    }
    if vector_shape.last().copied() != Some(input_dim) {
        return Err("not_one_original_input_vector");
    }
    let w1 = first
        .weight()
        .as_slice()
        .ok_or("nonstandard_original_layout")?;
    let w2 = last
        .weight()
        .as_slice()
        .ok_or("nonstandard_original_layout")?;
    let (b1, b2) = match authored {
        Some((b1, b2)) => (
            authored_bias(b1, hidden_dim, vector_shape.len())?,
            authored_bias(b2, output_dim, vector_shape.len())?,
        ),
        None => (
            match first.bias() {
                Some(b) => HullBiasData::Binary32(b.as_slice().ok_or("nonstandard_bias")?),
                None => HullBiasData::Zeros,
            },
            match last.bias() {
                Some(b) => HullBiasData::Binary32(b.as_slice().ok_or("nonstandard_bias")?),
                None => HullBiasData::Zeros,
            },
        ),
    };
    Ok(HullOriginalData {
        input_dim,
        hidden_dim,
        output_dim,
        w1,
        w2,
        b1,
        b2,
        objective,
        objective_constant: 0.0,
        lower: input.lower().as_slice().ok_or("nonstandard_input_box")?,
        upper: input.upper().as_slice().ok_or("nonstandard_input_box")?,
    })
}

/// The sealed replay receipt is retained alongside the result until the
/// consumer actually publishes/drops it; a mutable plain report never closes.
pub(in crate::commands::beta_crown) struct HullBudgetedRootResult {
    pub result: BetaCrownResult,
    pub proof: HullCheckedCertificate,
}

/// Explicit budgeted root profile. Off/BaB/default callers remain unchanged;
/// nonpositive triangle results decline rather than enter an unpriced LP.
pub(in crate::commands::beta_crown) fn try_single_hidden_hull_root_in(
    mode: SingleHiddenHullMode,
    network: &Network,
    input: &BoundedTensor,
    vnnlib: &VnnLibSpec,
    context: &HullQueryContext,
    start_time: Instant,
) -> Result<HullBudgetedRootResult, &'static str> {
    if mode != SingleHiddenHullMode::Root {
        return Err("unsupported_budgeted_mode");
    }
    context
        .checkpoint(HullLeaseClass::Optional)
        .map_err(|_| "resource_refused")?;
    let objective = single_relational_margin(vnnlib).ok_or("not_one_relational_row")?;
    let originals = data(network, input, vnnlib, &objective)?;
    let owner = import_hull_original(originals, context).map_err(|_| "original_import_refused")?;
    let checked = checked_hull_triangle(&owner, context).map_err(|_| "triangle_refused")?;
    checked
        .require_strict_positive()
        .map_err(|_| "unpriced_lp_or_bab")?;
    let proof = replay_hull_original(&checked.certificate, context)
        .map_err(|_| "original_replay_refused")?;
    if proof.origin() != HullProofOrigin::OriginalReplay
        || !proof.verified()
        || proof.scope() != owner.scope()
    {
        return Err("replay_not_strict_positive");
    }
    let result = BetaCrownResult {
        result: BabVerificationStatus::Verified,
        domains_explored: 1,
        domains_verified: 1,
        cuts_generated: 0,
        max_depth_reached: 0,
        time_elapsed: start_time.elapsed(),
        output_bounds: None,
    };
    context
        .checkpoint(HullLeaseClass::Replay)
        .map_err(|_| "publication_resource_refused")?;
    Ok(HullBudgetedRootResult { result, proof })
}

#[cfg(test)]
#[path = "single_hidden_hull_resource_tests.rs"]
mod tests;
