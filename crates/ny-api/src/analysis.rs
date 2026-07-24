// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Curated dead-neuron analysis and elimination surface for external consumers.
//!
//! Classify each neuron as live / dead / constant over an input region
//! ([`analyze_neurons`], [`analyze_neurons_with_epsilon`]), then construct an
//! optimized network from the analysis with a verifiable certificate
//! ([`eliminate_dead_neurons`], [`eliminate_and_verify`]). Promoted to the
//! stable facade so downstream consumers no longer need to reach past it into
//! ny-propagate.

/// Dead-neuron analysis: classify each neuron as live / dead / constant.
pub use ny_propagate::{
    analyze_neurons, analyze_neurons_with_epsilon, AnalysisResult, NeuronAnalysis, NeuronStatus,
};

/// Dead-neuron elimination: construct an optimized network from analysis.
pub use ny_propagate::{
    eliminate_and_verify, eliminate_dead_neurons, EliminationAction, EliminationCertificate,
    EliminationEntry, EliminationVerification,
};
