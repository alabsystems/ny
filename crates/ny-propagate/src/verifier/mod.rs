// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Main verifier struct for neural network verification.

mod beta_crown_finalize;
mod config;
mod graph;
mod network;
mod spec;

#[cfg(test)]
mod tests;

use ny_core::GemmEngine;
use std::sync::Arc;

use crate::types::PropagationConfig;

// Re-export types needed by test submodules via `use super::super::*`.
#[cfg(test)]
use crate::network::Network;
#[cfg(test)]
use crate::types::PropagationMethod;

/// Main verifier struct.
///
/// Provides a unified interface for neural network verification using
/// different bound propagation methods (IBP, CROWN, α-CROWN, β-CROWN).
///
/// # Example
/// ```rust,no_run
/// use ny_propagate::{Verifier, PropagationConfig, PropagationMethod};
///
/// let config = PropagationConfig {
///     method: PropagationMethod::Crown,
///     ..Default::default()
/// };
/// let verifier = Verifier::new(config);
/// // let result = verifier.verify(&network, &spec).unwrap();
/// ```
pub struct Verifier {
    config: PropagationConfig,
    engine: Option<Arc<dyn GemmEngine>>,
}

impl Verifier {
    pub fn new(config: PropagationConfig) -> Self {
        Self {
            config,
            engine: None,
        }
    }

    pub fn new_with_engine(config: PropagationConfig, engine: Arc<dyn GemmEngine>) -> Self {
        Self {
            config,
            engine: Some(engine),
        }
    }

    #[deprecated(note = "renamed to new_with_engine for consistency")]
    pub fn with_engine(config: PropagationConfig, engine: Arc<dyn GemmEngine>) -> Self {
        Self::new_with_engine(config, engine)
    }

    pub(crate) fn engine(&self) -> Option<&dyn GemmEngine> {
        self.engine.as_deref()
    }

    /// Resolve engine precedence: per-call engine overrides stored engine.
    ///
    /// Mirrors `BetaCrownVerifier::resolve_engine` for API consistency (#4099).
    pub(crate) fn resolve_engine<'a>(
        &'a self,
        engine: Option<&'a dyn GemmEngine>,
    ) -> Option<&'a dyn GemmEngine> {
        engine.or_else(|| self.engine())
    }

    pub(crate) fn engine_arc(&self) -> Option<Arc<dyn GemmEngine>> {
        self.engine.clone()
    }
}
