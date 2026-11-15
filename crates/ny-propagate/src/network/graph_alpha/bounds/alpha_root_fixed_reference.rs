// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Explicit root-only fixed-reference alpha transaction.
//!
//! The caller owns the local phase deadline, reference tightening allocation,
//! global reserve, and root eligibility. This entry creates the reference from
//! its own immutable graph/input borrows; it accepts no caller-supplied map.

use super::crown::CrownIbpCollectOptions;
use super::{
    AlphaCrownConfig, AlphaReferenceBoundsSource, GraphAlphaCollectionOutcome, GraphNetwork,
    NyError, PrecomputedAlphaReferenceBounds, Result,
};
use crate::bounds::GradientMethod;
use crate::network::graph_alpha::propagate_dag::{
    DagAlphaCollectionArtifact, RootFixedReferenceCollection,
};
use ny_core::GemmEngine;
use ny_tensor::BoundedTensor;
use std::time::Duration;

impl GraphNetwork {
    /// RAII thread-local test observation; dropping the closure restores its predecessor.
    #[cfg(test)]
    pub(crate) fn root_fixed_reference_collection_counter_for_test() -> impl Fn() -> usize {
        let counter = super::crown::RootFixedReferenceCollectionCounter::start();
        move || counter.entries()
    }

    /// Build one capped, certified CROWN-IBP reference for this root domain,
    /// optimize with fixed intermediates, and publish the optimizer-owned map.
    ///
    /// Both foundational IBP and alpha obey `config.deadline`; `reference_cap`
    /// additionally limits only CROWN tightening after IBP. The caller must
    /// budget those separately and leave time for fresh objective evaluation.
    /// No new publication-map clone, retry, fresh deadline, or tail recollection
    /// is introduced; existing input/output/state allocations remain in force.
    ///
    /// Sequential GraphNetwork chains intentionally enter the same DAG core as
    /// existing precomputed-reference callers. General sequential/SPSA APIs and
    /// the older phase-cap checkpoint policy retain their existing behavior.
    /// INVPROP-conditioned, non-fixed, unbounded, and zero-iteration requests
    /// are rejected before collection. A zero-completed optimizer is refused;
    /// no-optimizer initialization/skip returns an explicit ReferenceOnly.
    /// Returned map/alpha state carries no verdict authority.
    pub(crate) fn collect_root_fixed_reference_alpha_with_engine(
        &self,
        input: &BoundedTensor,
        config: &AlphaCrownConfig,
        engine: Option<&dyn GemmEngine>,
        reference_cap: Duration,
    ) -> Result<GraphAlphaCollectionOutcome> {
        validate_request(config, reference_cap)?;
        self.ensure_proof_eligible()?;
        let _l2_lever_off = crate::l2_lever_gate::L2LeverGuard::disabled();
        require_live_phase(config)?;
        let reference = self.collect_crown_ibp_bounds_dag_with_options(
            input,
            reference_options(config, engine, reference_cap),
        )?;
        // A cache hit or tightening fallback can return at the phase boundary.
        // Do not initialize a new optimizer after its local authority expired.
        require_live_phase(config)?;
        let collected = self.propagate_dag_alpha_crown_collect_root_fixed_reference(
            input,
            config,
            engine,
            PrecomputedAlphaReferenceBounds {
                bounds: reference.bounds,
                source: AlphaReferenceBoundsSource::CrownIbp,
            },
        );
        match collected? {
            RootFixedReferenceCollection::Optimized(artifact) => {
                publish_completed_collection(Ok(Some(artifact)), config.past_deadline())
            }
            RootFixedReferenceCollection::ReferenceOnly(artifact) => {
                if artifact.completed_iterations != 0
                    || artifact.optimizer_updates_completed != 0
                    || !matches!(
                        artifact.reference_bounds_source,
                        AlphaReferenceBoundsSource::CrownIbp
                    )
                {
                    return Err(NyError::InvalidSpec(
                        "root reference-only result carried optimizer credit or wrong provenance"
                            .into(),
                    ));
                }
                Ok(GraphAlphaCollectionOutcome::ReferenceOnly {
                    result: (artifact.reference_bounds, artifact.alpha_state),
                })
            }
        }
    }
}

fn validate_request(config: &AlphaCrownConfig, reference_cap: Duration) -> Result<()> {
    if !config.fix_interm_bounds
        || config.gradient_method == GradientMethod::Spsa
        || config.invprop.enabled
        || config.deadline.is_none()
        || config.iterations == 0
        || reference_cap.is_zero()
    {
        return Err(NyError::InvalidSpec(
            "root fixed-reference alpha requires fixed intermediates, non-SPSA, disabled INVPROP, a finite phase deadline, and positive iterations/reference cap".to_string(),
        ));
    }
    Ok(())
}

fn require_live_phase(config: &AlphaCrownConfig) -> Result<()> {
    if config.past_deadline() {
        return Err(NyError::DeadlineExceeded(
            "root fixed-reference alpha: phase expired before the first completed fold".to_string(),
        ));
    }
    Ok(())
}

fn reference_options<'a>(
    config: &AlphaCrownConfig,
    engine: Option<&'a dyn GemmEngine>,
    reference_cap: Duration,
) -> CrownIbpCollectOptions<'a> {
    CrownIbpCollectOptions {
        engine,
        deadline: config.deadline,
        deadline_includes_ibp: true,
        tightening_cap: Some(reference_cap),
        complete_cache_lookup_only: true,
        original_root_head_rows: Some(super::head_rows::OriginalRootHeadRows),
        ..Default::default()
    }
}

fn publish_completed_collection(
    collected: Result<Option<DagAlphaCollectionArtifact>>,
    phase_deadline_expired: bool,
) -> Result<GraphAlphaCollectionOutcome> {
    // Numeric/shape/provenance errors remain errors. Only the existing loop
    // policy may convert a *later deadline* into a completed artifact.
    let artifact = collected?;
    let Some(artifact) = artifact.filter(|artifact| artifact.completed_iterations > 0) else {
        return Err(if phase_deadline_expired {
            NyError::DeadlineExceeded(
                "root fixed-reference alpha: no completed fold before the phase deadline"
                    .to_string(),
            )
        } else {
            NyError::InvalidSpec(
                "root fixed-reference alpha: optimizer returned no completed fold".to_string(),
            )
        });
    };
    let DagAlphaCollectionArtifact {
        alpha_state,
        reference_bounds,
        reference_bounds_source,
        completed_iterations,
        optimizer_updates_completed,
        ..
    } = artifact;
    if !matches!(
        reference_bounds_source,
        AlphaReferenceBoundsSource::CrownIbp
    ) {
        return Err(NyError::InvalidSpec(
            "root fixed-reference alpha: unexpected reference provenance".to_string(),
        ));
    }
    let result = (reference_bounds, alpha_state);
    Ok(if phase_deadline_expired {
        GraphAlphaCollectionOutcome::PhaseCapCheckpoint {
            result,
            completed_iterations,
            optimizer_updates_completed,
        }
    } else {
        GraphAlphaCollectionOutcome::Complete(result)
    })
}

#[cfg(test)]
#[path = "alpha_root_fixed_reference_tests.rs"]
mod tests;
