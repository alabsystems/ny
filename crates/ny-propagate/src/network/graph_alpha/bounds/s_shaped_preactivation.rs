// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Resolve the exact producer enclosure used to initialize S-shaped alphas.

use std::collections::HashMap;

use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;

use crate::network::core::{GraphNode, NETWORK_INPUT};

/// Return the output enclosure of the monotone activation's unique producer.
///
/// Reference-bound maps are keyed by the node that *produced* each stored
/// output. Looking up `node_name` would therefore return the Sigmoid/Tanh
/// output, a strict nonlinear image of its input, rather than the
/// pre-activation interval from which tangent-validity limits are derived.
/// Missing or non-unary producer identity is refused instead of guessing.
pub(super) fn resolve<'a>(
    node: &GraphNode,
    node_name: &str,
    network_input: &'a BoundedTensor,
    reference_bounds: &'a HashMap<String, BoundedTensor>,
) -> Result<&'a BoundedTensor> {
    let producer = node.require_unary_input()?;
    if producer == NETWORK_INPUT {
        return Ok(network_input);
    }
    reference_bounds.get(producer).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "pre-activation bounds for producer '{producer}' consumed by monotone node \
             '{node_name}' are missing"
        ))
    })
}
