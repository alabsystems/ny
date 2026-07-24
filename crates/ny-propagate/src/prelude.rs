// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Convenience prelude for ny-propagate.
//!
//! This module re-exports the most commonly used types for neural network
//! verification. Use `use ny_propagate::prelude::*;` for ergonomic imports.
//!
//! # Example
//!
//! ```text
//! use ny_propagate::prelude::*;
//! use ndarray::array;
//!
//! // Create a simple network (2D input -> ReLU -> 2D output)
//! let weights = array![[1.0, 0.0], [0.0, 1.0]];
//! let bias = array![0.0, 0.0];
//! let mut network = Network::new();
//! network.add_layer(Layer::Linear(LinearLayer::new(weights, Some(bias))?));
//! network.add_layer(Layer::ReLU(ReLULayer));
//!
//! // Create a verification spec: input in [-1, 1]^2, output > 0
//! let spec = VerificationSpec::new(
//!     vec![Bound::new(-1.0, 1.0), Bound::new(-1.0, 1.0)],
//!     vec![Bound::new(0.0, f32::INFINITY), Bound::new(0.0, f32::INFINITY)],
//! )?;
//!
//! // Run verification
//! let config = PropagationConfig::default();
//! let verifier = Verifier::new(config);
//! let result = verifier.verify(&network, &spec)?;
//! ```
//!
//! # What's Included
//!
//! - **Verifiers**: [`Verifier`], [`BetaCrownVerifier`]
//! - **Configuration**: [`BetaCrownConfig`], [`AlphaCrownConfig`], [`PropagationConfig`],
//!   [`PropagationMethod`], [`BranchingHeuristic`], [`GradientMethod`], [`Optimizer`],
//!   [`MultiSpecKeep`], [`KfsbReduceOp`]
//! - **Results**: [`BetaCrownResult`], [`BabVerificationStatus`], [`BlockProgress`],
//!   [`LayerProgress`], [`VerificationCheckpoint`]
//! - **Layers**: [`Layer`], [`BoundPropagation`], [`LinearLayer`], [`ReLULayer`], [`GELULayer`],
//!   [`LayerNormLayer`], [`SoftmaxLayer`], [`MatMulLayer`], [`Conv2dLayer`]
//! - **Networks**: [`Network`], [`GraphNetwork`], [`GraphNode`]
//! - **Core types**: [`Result`], [`NyError`], [`BoundedTensor`], [`VerificationResult`],
//!   [`VerificationSpec`], [`Bound`]
//! - **Parallel verification**: [`ParallelVerifier`], [`ParallelConfig`], [`verify_parallel`],
//!   [`verify_parallel_with_method`], [`ParallelVerificationResult`]
//! - **Progress utilities**: [`truncate_name`], [`compute_model_hash`], [`BlockBoundsInfo`],
//!   [`NodeBoundsInfo`], [`BlockWiseResult`], [`LayerByLayerResult`]

// =============================================================================
// Verifiers
// =============================================================================

pub use crate::beta_crown::BetaCrownVerifier;
pub use crate::verifier::Verifier;

// =============================================================================
// Configuration
// =============================================================================

pub use crate::beta_crown::{BetaCrownConfig, BranchingHeuristic, KfsbReduceOp};
pub use crate::bounds::{AlphaCrownConfig, GradientMethod, MultiSpecKeep, Optimizer};
pub use crate::types::{MulBinaryRelaxationMode, PropagationConfig, PropagationMethod};

// =============================================================================
// Results
// =============================================================================

pub use crate::beta_crown::{BabVerificationStatus, BetaCrownResult};
pub use crate::types::{
    compute_model_hash, BlockBoundsInfo, BlockProgress, BlockWiseResult, BoundsProvenance,
    CrownIbpBoundsResult, CrownIbpFallbackEvent, CrownIbpFallbackReason, GraphCrownIbpBoundsResult,
    LayerByLayerResult, LayerProgress, NodeBoundsInfo, VerificationCheckpoint,
};

// =============================================================================
// Layers
// =============================================================================

pub use crate::layers::{BoundPropagation, Layer};

// Common layer types used in external code
pub use crate::layers::{
    AttentionMask, Conv2dLayer, ConvTranspose1dLayer, ConvTranspose2dLayer, GELULayer,
    LayerNormCrownMode, LayerNormLayer, LayerNormMode, LinearLayer, MatMulIbpMode, MatMulLayer,
    ReLULayer, SelfAttentionLayer, SiLULayer, SoftmaxLayer,
};

// =============================================================================
// Networks
// =============================================================================

pub use crate::network::{GraphNetwork, GraphNode, Network};

// =============================================================================
// Core Types (from ny_core/ny_tensor)
// =============================================================================

pub use ny_core::{Bound, NyError, Result, VerificationResult, VerificationSpec};
pub use ny_tensor::BoundedTensor;

// =============================================================================
// Parallel Verification
// =============================================================================

pub use crate::parallel::{
    verify_parallel, verify_parallel_with_engine, verify_parallel_with_method,
    verify_parallel_with_method_and_engine, ParallelConfig, ParallelVerificationResult,
    ParallelVerifier,
};

// =============================================================================
// Composition (multi-model pipeline verification)
// =============================================================================

pub use crate::composition::certificate::{
    BoundCertificate, BoundCertificationResult, BoundProvenance,
};
pub use crate::composition::pipeline::{PipelineCertificate, PipelineVerifier};

// =============================================================================
// Progress Utilities
// =============================================================================

pub use crate::types::truncate_name;
