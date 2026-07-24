// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Configuration helpers for verification methods.

use super::Verifier;
use crate::beta_crown::BetaCrownConfig;
use crate::AlphaCrownConfig;
use ny_core::VerificationSpec;
use std::time::Instant;

impl Verifier {
    /// Build α-CROWN config from propagation config with optional deadline (#2987).
    ///
    /// When `deadline` is `Some`, the α-CROWN optimization loop will bail early
    /// via `past_deadline()` if the verification timeout budget is exhausted.
    pub(super) fn alpha_crown_config(&self, deadline: Option<Instant>) -> AlphaCrownConfig {
        let mut config = AlphaCrownConfig::default();
        config.iterations = self.config.max_iterations;
        config.tolerance = self.config.tolerance;
        config.deadline = deadline;
        config
    }

    pub(super) fn beta_crown_config(&self, spec: &VerificationSpec) -> BetaCrownConfig {
        let mut config = BetaCrownConfig::default();
        config.alpha_config.iterations = self.config.max_iterations;
        config.alpha_config.tolerance = self.config.tolerance;
        config.beta_iterations = self.config.max_iterations;
        config.beta_tolerance = self.config.tolerance;
        config.root_beta_iterations = config.root_beta_iterations.min(self.config.max_iterations);
        config.timeout = spec
            .timeout_ms()
            .map(std::time::Duration::from_millis)
            .unwrap_or(config.timeout);
        config
    }
}
