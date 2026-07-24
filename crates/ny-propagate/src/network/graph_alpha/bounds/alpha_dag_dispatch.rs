// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! DAG gradient dispatch delegation for the root alpha collection path.
//!
//! Part of #4036: when `config.gradient_method` is not SPSA, the root
//! collection loop delegates to the full DAG optimizer which properly
//! dispatches AnalyticChain, FD, and Analytic gradient methods.

use super::*;
use crate::bounds::GradientMethod;

impl GraphNetwork {
    /// Try to delegate alpha collection to the DAG optimizer for non-SPSA
    /// gradient methods.
    ///
    /// Returns `Ok(Some(result))` if delegation succeeded, `Ok(None)` if SPSA
    /// is configured (the caller should fall through to the SPSA loop).
    pub(in crate::network::graph_alpha) fn try_dag_gradient_dispatch(
        &self,
        input: &BoundedTensor,
        config: &AlphaCrownConfig,
        engine: Option<&dyn ny_core::GemmEngine>,
        exec_order: &[String],
    ) -> Result<Option<GraphAlphaCollectionResult>> {
        if config.gradient_method == GradientMethod::Spsa {
            return Ok(None);
        }

        let (output_bounds, alpha_state) =
            match self.propagate_dag_alpha_crown_collect_with_engine(input, config, engine)? {
                Some(pair) => pair,
                None => return Ok(None),
            };

        let reference_bounds =
            self.collect_alpha_reference_bounds_with_engine(input, config, engine, exec_order)?;
        if config.fix_interm_bounds {
            return Ok(Some((reference_bounds, alpha_state)));
        }

        // Collect bounds at all nodes using the optimized alpha state.
        let all_bounds = self.collect_crown_bounds_with_alpha(
            input,
            &reference_bounds,
            &alpha_state,
            engine,
            config.deadline,
        )?;

        // Intersect the DAG optimizer's output bounds with the collection bounds.
        let output_node = if self.output_node.is_empty() {
            exec_order.last().map(String::as_str).unwrap_or("")
        } else {
            &self.output_node
        };
        let mut merged = all_bounds;
        if let Some(existing) = merged.get(output_node) {
            if existing.shape() == output_bounds.shape() {
                if let Some((tightened, _)) = existing.intersection_per_element(&output_bounds) {
                    merged.insert(output_node.to_string(), tightened);
                }
            }
        }

        Ok(Some((merged, alpha_state)))
    }
}
