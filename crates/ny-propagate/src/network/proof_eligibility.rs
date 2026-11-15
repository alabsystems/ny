// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Whole-network authority check before supplied/cached bounds or early proof
//! returns can bypass a layer's interval/backward refusal.
//!
//! A point-only layer is deliberately refused even for an empty specification,
//! a zero coefficient, or an unreachable node. Diagnostic execution does not
//! qualify a graph for proof. This scan is allocation-free on ordinary graphs
//! and reads the current layers, so mutation and cache adoption cannot stale it.

use ny_core::{NyError, Result};

use super::{GraphNetwork, Network};

pub(crate) fn diagnostic_only() -> NyError {
    NyError::UnsupportedOp(
        "PointLinear is diagnostic point-only: this network is ineligible for interval, CROWN, cached-bound, or verifier proof authority".into(),
    )
}

impl Network {
    /// Refuse diagnostic-only layers before an interval or verifier request.
    ///
    /// This checks representation eligibility only. `Ok(())` does not certify
    /// ordinary layers, supplied bounds, arithmetic, or a property.
    pub fn ensure_proof_eligible(&self) -> Result<()> {
        if self
            .layers
            .iter()
            .any(|layer| layer.is_diagnostic_point_only())
        {
            return Err(diagnostic_only());
        }
        Ok(())
    }
}

impl GraphNetwork {
    /// Refuse diagnostic-only layers before any proof-producing shortcut.
    ///
    /// Every stored node is checked, including nodes omitted by a target walk.
    /// `Ok(())` is not a numeric certificate or permission to trust a bound map.
    pub fn ensure_proof_eligible(&self) -> Result<()> {
        if self
            .nodes
            .values()
            .any(|node| node.layer().is_diagnostic_point_only())
        {
            return Err(diagnostic_only());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
