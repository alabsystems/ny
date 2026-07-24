// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Parallel position verification for sequence models.
//!
//! Provides parallel verification across sequence positions, enabling near-linear
//! speedup with cores for position-independent verification. Use when verifying
//! position-independent properties and sequence length >= number of CPU cores.

use crate::faer_parallelism::RayonTaskGuard;
use crate::network::GraphNetwork;
use crate::types::PropagationMethod;
use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use rayon::prelude::*;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use tracing::{debug, info, trace, warn};

/// Configuration for parallel verification.
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Propagation method to use (IBP, CROWN, etc.)
    pub method: PropagationMethod,

    /// Minimum number of positions before enabling parallelism.
    /// Below this threshold, serial execution is used to avoid overhead.
    pub min_positions_for_parallel: usize,

    /// Maximum number of threads to use.
    /// None means use rayon's default (typically number of cores).
    pub max_threads: Option<usize>,

    /// Whether to report progress during verification.
    pub report_progress: bool,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            method: PropagationMethod::Ibp,
            min_positions_for_parallel: 4,
            max_threads: None,
            report_progress: false,
        }
    }
}

/// Result of parallel position verification.
#[derive(Debug)]
pub struct ParallelVerificationResult {
    /// Output bounds for each position, stacked back together.
    pub output_bounds: BoundedTensor,

    /// Number of positions verified.
    pub num_positions: usize,

    /// Number of positions verified in parallel (vs serial).
    pub parallel_positions: usize,

    /// Total verification time in milliseconds.
    pub total_time_ms: u64,

    /// Average time per position in milliseconds.
    pub avg_position_time_ms: f64,
}

/// Parallel verifier for sequence models.
///
/// Distributes verification across sequence positions using rayon,
/// achieving near-linear speedup with available cores.
pub struct ParallelVerifier {
    config: ParallelConfig,
    engine: Option<Arc<dyn GemmEngine>>,
}

impl ParallelVerifier {
    /// Create a new parallel verifier with the given configuration.
    pub fn new(config: ParallelConfig) -> Self {
        Self {
            config,
            engine: None,
        }
    }

    /// Create a parallel verifier that reuses a stored GemmEngine.
    pub fn new_with_engine(config: ParallelConfig, engine: Arc<dyn GemmEngine>) -> Self {
        Self {
            config,
            engine: Some(engine),
        }
    }

    fn engine(&self) -> Option<&dyn GemmEngine> {
        self.engine.as_deref()
    }

    /// Verify each position along the specified axis in parallel.
    ///
    /// This is ideal for verifying position-independent properties on
    /// sequence models like transformers.
    ///
    /// # Arguments
    /// * `graph` - The network graph to verify
    /// * `input` - Input bounded tensor with shape [..., axis_size, ...]
    /// * `axis` - The axis to parallelize over (typically seq_len axis)
    ///
    /// # Returns
    /// Combined output bounds with same shape as serial verification.
    pub fn verify_positions_parallel(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        axis: usize,
    ) -> Result<ParallelVerificationResult> {
        let start_time = std::time::Instant::now();
        let shape = input.shape();

        if axis >= shape.len() {
            return Err(NyError::InvalidSpec(format!(
                "Axis {} out of bounds for tensor with {} dimensions",
                axis,
                shape.len()
            )));
        }

        let num_positions = shape[axis];
        info!(
            "Parallel verification: {} positions along axis {}, method {:?}",
            num_positions, axis, self.config.method
        );

        // Decide whether to use parallel or serial
        let use_parallel = num_positions >= self.config.min_positions_for_parallel;
        let parallel_positions = if use_parallel { num_positions } else { 0 };

        let output_positions = if use_parallel {
            self.verify_parallel_impl(graph, input, axis, num_positions)?
        } else {
            debug!(
                "Using serial verification ({} positions < threshold {})",
                num_positions, self.config.min_positions_for_parallel
            );
            self.verify_serial_impl(graph, input, axis, num_positions)?
        };

        // Stack outputs back together
        let output_bounds = BoundedTensor::stack(&output_positions, axis)?;

        let total_time_ms = start_time.elapsed().as_millis() as u64;
        let avg_position_time_ms = total_time_ms as f64 / num_positions as f64;

        info!(
            "Parallel verification complete: {}ms total, {:.2}ms/position",
            total_time_ms, avg_position_time_ms
        );

        Ok(ParallelVerificationResult {
            output_bounds,
            num_positions,
            parallel_positions,
            total_time_ms,
            avg_position_time_ms,
        })
    }

    /// Verify each sample in the batch dimension in parallel.
    ///
    /// Useful when batch size > 1 and each sample is independent.
    pub fn verify_batch_parallel(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        batch_axis: usize,
    ) -> Result<ParallelVerificationResult> {
        // Same implementation, just typically called with axis=0
        self.verify_positions_parallel(graph, input, batch_axis)
    }

    /// Internal parallel implementation using rayon.
    fn verify_parallel_impl(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        axis: usize,
        num_positions: usize,
    ) -> Result<Vec<BoundedTensor>> {
        self.verify_parallel_impl_with_engine(graph, input, axis, num_positions, self.engine())
    }

    /// Internal parallel implementation with GPU engine (#3598).
    fn verify_parallel_impl_with_engine(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        axis: usize,
        num_positions: usize,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<Vec<BoundedTensor>> {
        let progress = AtomicUsize::new(0);

        // Build the thread pool with optional thread limit
        let pool = if let Some(max_threads) = self.config.max_threads {
            Some(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(max_threads)
                    .build()
                    .map_err(|e| {
                        NyError::InvalidSpec(format!("Failed to create thread pool: {}", e))
                    })?,
            )
        } else {
            None
        };

        let report_progress = self.config.report_progress;
        let method = self.config.method;

        // Closure to verify a single position
        let verify_position = |pos: usize| -> Result<BoundedTensor> {
            let _rayon_task_guard = RayonTaskGuard::new();
            trace!("Verifying position {}/{}", pos + 1, num_positions);

            let pos_input = input.slice_axis(axis, pos)?;
            let pos_output = propagate_position(graph, &pos_input, method, pos, engine)?;

            if report_progress {
                let completed = progress.fetch_add(1, Ordering::Relaxed) + 1;
                if completed.is_multiple_of(10) || completed == num_positions {
                    debug!("Progress: {}/{} positions", completed, num_positions);
                }
            }

            Ok(pos_output)
        };

        // Execute in parallel
        if let Some(pool) = pool {
            pool.install(|| {
                (0..num_positions)
                    .into_par_iter()
                    .map(verify_position)
                    .collect()
            })
        } else {
            (0..num_positions)
                .into_par_iter()
                .map(verify_position)
                .collect()
        }
    }

    /// Internal serial implementation for small position counts.
    fn verify_serial_impl(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        axis: usize,
        num_positions: usize,
    ) -> Result<Vec<BoundedTensor>> {
        self.verify_serial_impl_with_engine(graph, input, axis, num_positions, self.engine())
    }

    /// Internal serial implementation with GPU engine (#3598).
    fn verify_serial_impl_with_engine(
        &self,
        graph: &GraphNetwork,
        input: &BoundedTensor,
        axis: usize,
        num_positions: usize,
        engine: Option<&dyn GemmEngine>,
    ) -> Result<Vec<BoundedTensor>> {
        let mut results = Vec::with_capacity(num_positions);

        for pos in 0..num_positions {
            trace!("Verifying position {}/{} (serial)", pos + 1, num_positions);

            let pos_input = input.slice_axis(axis, pos)?;
            let pos_output =
                propagate_position(graph, &pos_input, self.config.method, pos, engine)?;
            results.push(pos_output);
        }

        Ok(results)
    }
}

/// Dispatch a single position through the configured propagation method.
///
/// Shared by both parallel and serial verification paths.
fn propagate_position(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    method: PropagationMethod,
    pos: usize,
    engine: Option<&dyn GemmEngine>,
) -> Result<BoundedTensor> {
    match method {
        PropagationMethod::Ibp => graph.propagate_ibp(input),
        PropagationMethod::Crown => crown_with_fallback(graph, input, pos, "CROWN", engine),
        PropagationMethod::AlphaCrown => {
            crown_with_fallback(graph, input, pos, "alpha-CROWN", engine)
        }
        // SDP-CROWN's ReLU offsets and concretization are only valid over an ℓ2 input ball,
        // and a `BoundedTensor` carries per-element ℓ∞ bounds. The ball of the box's
        // half-width ε covers a strict subset of the box (its corners sit at ℓ2 distance
        // ε√n), and the ball that does contain the box has radius ε√n, over which
        // ‖a‖₂·ε√n >= ‖a‖₁·ε leaves the concretization no tighter than CROWN's. Neither
        // answers a box input, so refuse rather than certify a region we did not bound.
        PropagationMethod::SdpCrown => Err(NyError::UnsupportedOp(
            "SDP-CROWN requires an ℓ2 input ball, but the input bounds form an \
             ℓ∞ box; use CROWN or α-CROWN instead"
                .to_string(),
        )),
        PropagationMethod::BetaCrown => {
            crown_with_fallback(graph, input, pos, "beta-CROWN", engine)
        }
    }
}

/// Convenience function for parallel position verification with default config.
///
/// # Example
/// ```rust,no_run
/// # use ny_propagate::parallel::verify_parallel;
/// # use ny_propagate::{GraphNetwork, GraphNode, Layer};
/// # use ny_propagate::layers::LinearLayer;
/// # use ny_tensor::BoundedTensor;
/// # use ndarray::{Array1, Array2, ArrayD, IxDyn};
/// # fn example() -> ny_core::Result<()> {
/// # let w = Array2::<f32>::eye(4);
/// # let l = LinearLayer::new(w, Some(Array1::zeros(4)))?;
/// # let mut graph = GraphNetwork::new();
/// # graph.try_add_node(GraphNode::from_input("l", Layer::Linear(l)))?;
/// # graph.set_output("l");
/// # let input = BoundedTensor::from_epsilon(ArrayD::from_elem(IxDyn(&[2,4]), 0.5f32), 0.1)?;
/// let output = verify_parallel(&graph, &input, 0)?;
/// # Ok(())
/// # }
/// ```
pub fn verify_parallel(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    axis: usize,
) -> Result<BoundedTensor> {
    let config = ParallelConfig::default();
    let verifier = ParallelVerifier::new(config);
    Ok(verifier
        .verify_positions_parallel(graph, input, axis)?
        .output_bounds)
}

/// Engine-aware convenience function for parallel position verification (#3772).
pub fn verify_parallel_with_engine(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    axis: usize,
    engine: Arc<dyn GemmEngine>,
) -> Result<BoundedTensor> {
    let config = ParallelConfig::default();
    let verifier = ParallelVerifier::new_with_engine(config, engine);
    Ok(verifier
        .verify_positions_parallel(graph, input, axis)?
        .output_bounds)
}

/// Convenience function for parallel position verification with custom method.
pub fn verify_parallel_with_method(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    axis: usize,
    method: PropagationMethod,
) -> Result<BoundedTensor> {
    let config = ParallelConfig {
        method,
        ..Default::default()
    };
    let verifier = ParallelVerifier::new(config);
    Ok(verifier
        .verify_positions_parallel(graph, input, axis)?
        .output_bounds)
}

/// Engine-aware convenience function for parallel position verification with custom method (#3772).
pub fn verify_parallel_with_method_and_engine(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    axis: usize,
    method: PropagationMethod,
    engine: Arc<dyn GemmEngine>,
) -> Result<BoundedTensor> {
    let config = ParallelConfig {
        method,
        ..Default::default()
    };
    let verifier = ParallelVerifier::new_with_engine(config, engine);
    Ok(verifier
        .verify_positions_parallel(graph, input, axis)?
        .output_bounds)
}

/// CROWN→IBP fallback chain with logging at each degradation step (#2240).
///
/// Tries batched CROWN, then flat CROWN, then IBP. Each failure is logged
/// with the position index and error so silent bound degradation is diagnosable.
fn crown_with_fallback(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    pos: usize,
    context: &str,
    engine: Option<&dyn GemmEngine>,
) -> Result<BoundedTensor> {
    // #3598: Use engine-aware variants for GPU acceleration.
    graph
        .propagate_crown_batched_with_provenance_and_engine(input, engine)
        .map(|result| result.bounds)
        .or_else(|e| {
            warn!(position = pos, error = %e, "{context}: CROWN batched failed, falling back to CROWN");
            graph.propagate_crown_with_engine(input, engine)
        })
        .or_else(|e| {
            warn!(position = pos, error = %e, "{context}: CROWN failed, falling back to IBP");
            graph.propagate_ibp(input)
        })
}

#[cfg(test)]
mod tests;
