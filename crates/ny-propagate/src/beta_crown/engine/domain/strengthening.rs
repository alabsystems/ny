// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Strengthened-cut verification for BiCCOS constraint strengthening.

use std::sync::Arc;

use ny_core::{GemmEngine, Result};
use ny_tensor::BoundedTensor;

use crate::beta_crown::bab_cuts::{CutPool, CuttingPlane, StrengthenedCut};
use crate::beta_crown::branching::SplitHistory;
use crate::beta_crown::domain::BabDomain;
use crate::beta_crown::state::{BetaState, DomainAlphaState};
use crate::Network;

use super::super::tensor_ext::BoundedTensorExt;
use super::super::BetaCrownVerifier;

impl BetaCrownVerifier {
    // Justification: BiCCOS cut strengthening requires the full verification context
    // (network, input, threshold, layer bounds, domain, engine) as independent parameters.
    // Grouping into a struct would obscure the function's interface since callers assemble
    // these from different sources.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn try_add_strengthened_cut(
        &self,
        cut_pool: &mut CutPool,
        network: &Network,
        input: &BoundedTensor,
        threshold: f32,
        base_layer_bounds: &[Arc<BoundedTensor>],
        domain: &BabDomain,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<bool> {
        if !self.config.enable_biccos_constraint_strengthening {
            return Ok(false);
        }
        if domain.history.constraints.is_empty() {
            return Ok(false);
        }
        // Skip constraint strengthening for input-split domains.
        if domain.input_bounds.is_some() {
            return Ok(false);
        }

        let strengthened = CuttingPlane::from_verified_domain_strengthened(
            &domain.history,
            &domain.beta_state,
            self.config.biccos_drop_ratio,
        )?;
        let Some(StrengthenedCut {
            cut,
            history,
            dropped_constraints,
        }) = strengthened
        else {
            return Ok(false);
        };

        if dropped_constraints > 0 {
            let verified = self.reverify_strengthened_history(
                network,
                input,
                threshold,
                base_layer_bounds,
                &history,
                engine,
            )?;
            if !verified {
                return Ok(false);
            }
        }

        Ok(cut_pool.add_cut(cut))
    }

    fn reverify_strengthened_history(
        &self,
        network: &Network,
        input: &BoundedTensor,
        threshold: f32,
        base_layer_bounds: &[Arc<BoundedTensor>],
        history: &SplitHistory,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<bool> {
        let Some(layer_bounds) = self.apply_history_constraints(base_layer_bounds, history)? else {
            return Ok(false);
        };

        let mut beta_state = BetaState::from_history(history)?;
        let mut alpha_state = if self.config.use_alpha_crown {
            DomainAlphaState::from_layer_bounds_and_constraints(network, &layer_bounds, history)
        } else {
            DomainAlphaState::empty()
        };

        let mut empty_cuts = CutPool::new(0);
        let (bounds, _) = self.optimize_joint_bounds(
            network,
            input,
            history,
            &layer_bounds,
            &mut beta_state,
            &mut alpha_state,
            &mut empty_cuts,
            engine,
        )?;

        let lower = bounds.lower_scalar();
        let upper = bounds.upper_scalar();
        Ok(self.config.domain_is_verified(lower, upper, threshold))
    }
}
