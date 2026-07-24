// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Node-level and layer-by-layer bounds information.

use super::helpers::truncate_name;
use serde::{Deserialize, Serialize};

/// Information about bounds at a specific node in the graph.
///
/// Used for layer-by-layer verification to track bound growth through the network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeBoundsInfo {
    /// Node name in the graph.
    pub name: String,
    /// Layer type (e.g., "Linear", "MatMul", "GELU").
    pub layer_type: String,
    /// Input bound width (maximum width across all input elements).
    pub input_width: f32,
    /// Output bound width (maximum width across all output elements).
    pub output_width: f32,
    /// Sensitivity = output_width / input_width.
    pub sensitivity: f32,
    /// Output shape.
    pub output_shape: Vec<usize>,
    /// Minimum bound value across all outputs.
    pub min_bound: f32,
    /// Maximum bound value across all outputs.
    pub max_bound: f32,
    /// Whether bounds have saturated (near f32::MAX).
    pub saturated: bool,
    /// Whether any bounds are NaN.
    pub has_nan: bool,
    /// Whether any bounds are infinite.
    pub has_infinite: bool,
}

impl NodeBoundsInfo {
    /// Check if this node's bounds have degraded (saturated, NaN, or infinite).
    pub fn has_degraded(&self) -> bool {
        self.saturated || self.has_nan || self.has_infinite
    }

    /// Get the status string for this node.
    pub fn status(&self) -> &'static str {
        if self.has_nan {
            "NAN"
        } else if self.has_infinite {
            "INF"
        } else if self.saturated {
            "SATURATED"
        } else if self.sensitivity > 100.0 {
            "HIGH"
        } else if self.sensitivity > 10.0 {
            "MODERATE"
        } else if self.sensitivity < 1.0 {
            "STABLE"
        } else {
            "OK"
        }
    }
}

/// Result of layer-by-layer verification through a GraphNetwork.
#[derive(Debug, Clone)]
pub struct LayerByLayerResult {
    /// Per-node bounds information.
    pub nodes: Vec<NodeBoundsInfo>,
    /// Input epsilon.
    pub input_epsilon: f32,
    /// Final output bound width.
    pub final_width: f32,
    /// Index of first node where bounds degraded (if any).
    pub degraded_at_node: Option<usize>,
    /// Total nodes processed.
    pub total_nodes: usize,
}

impl LayerByLayerResult {
    /// Generate a summary table of the verification results.
    pub fn summary(&self) -> String {
        let mut lines = Vec::new();
        lines.push("Layer-by-Layer Verification".to_string());
        lines.push("===========================".to_string());
        lines.push(format!(
            "{:<40} | {:>10} | {:>10} | {:>8} | Status",
            "Node", "In Width", "Out Width", "Sens."
        ));
        lines.push(format!(
            "{:-<40}-+-{:-<10}-+-{:-<10}-+-{:-<8}-+--------",
            "", "", "", ""
        ));

        for node in &self.nodes {
            let marker = if node.has_degraded() { " <<<" } else { "" };
            lines.push(format!(
                "{:<40} | {:>10.3e} | {:>10.3e} | {:>8.2} | {}{}",
                truncate_name(&node.name, 40),
                node.input_width,
                node.output_width,
                node.sensitivity,
                node.status(),
                marker
            ));
        }

        lines.push(String::new());
        lines.push(format!(
            "Input epsilon: {:.2e} → Final width: {:.2e}",
            self.input_epsilon, self.final_width
        ));
        lines.push(format!("Total nodes: {}", self.total_nodes));

        if let Some(idx) = self.degraded_at_node {
            if let Some(node) = self.nodes.get(idx) {
                lines.push(format!(
                    "WARNING: Bounds degraded at node {} ({})",
                    node.name,
                    node.status()
                ));
            }
        }

        lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_node_info(
        name: &str,
        sensitivity: f32,
        saturated: bool,
        has_nan: bool,
        has_infinite: bool,
    ) -> NodeBoundsInfo {
        NodeBoundsInfo {
            name: name.to_string(),
            layer_type: "Linear".to_string(),
            input_width: 0.1,
            output_width: sensitivity * 0.1,
            sensitivity,
            output_shape: vec![10],
            min_bound: -1.0,
            max_bound: 1.0,
            saturated,
            has_nan,
            has_infinite,
        }
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_node_bounds_info_has_degraded_false() {
        let node = make_node_info("test", 5.0, false, false, false);
        assert!(!node.has_degraded());
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_node_bounds_info_has_degraded_saturated() {
        let node = make_node_info("test", 5.0, true, false, false);
        assert!(node.has_degraded());
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_node_bounds_info_has_degraded_nan() {
        let node = make_node_info("test", 5.0, false, true, false);
        assert!(node.has_degraded());
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_node_bounds_info_has_degraded_infinite() {
        let node = make_node_info("test", 5.0, false, false, true);
        assert!(node.has_degraded());
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_node_bounds_info_status_nan() {
        let node = make_node_info("test", 5.0, false, true, false);
        assert_eq!(node.status(), "NAN");
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_node_bounds_info_status_inf() {
        let node = make_node_info("test", 5.0, false, false, true);
        assert_eq!(node.status(), "INF");
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_node_bounds_info_status_saturated() {
        let node = make_node_info("test", 5.0, true, false, false);
        assert_eq!(node.status(), "SATURATED");
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_node_bounds_info_status_high() {
        let node = make_node_info("test", 150.0, false, false, false);
        assert_eq!(node.status(), "HIGH");
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_node_bounds_info_status_moderate() {
        let node = make_node_info("test", 50.0, false, false, false);
        assert_eq!(node.status(), "MODERATE");
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_node_bounds_info_status_stable() {
        let node = make_node_info("test", 0.5, false, false, false);
        assert_eq!(node.status(), "STABLE");
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_node_bounds_info_status_ok() {
        let node = make_node_info("test", 5.0, false, false, false);
        assert_eq!(node.status(), "OK");
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_layer_by_layer_result_summary_empty() {
        let result = LayerByLayerResult {
            nodes: vec![],
            input_epsilon: 0.01,
            final_width: 0.0,
            degraded_at_node: None,
            total_nodes: 0,
        };
        let summary = result.summary();
        assert!(summary.contains("Layer-by-Layer Verification"));
        assert!(summary.contains("Input epsilon:"));
        assert!(summary.contains("Total nodes: 0"));
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_layer_by_layer_result_summary_with_nodes() {
        let result = LayerByLayerResult {
            nodes: vec![make_node_info("layer0", 2.5, false, false, false)],
            input_epsilon: 0.01,
            final_width: 0.025,
            degraded_at_node: None,
            total_nodes: 1,
        };
        let summary = result.summary();
        assert!(summary.contains("layer0"));
        assert!(summary.contains("OK"));
    }

    #[ntest::timeout(5000)]
    #[test]
    fn test_layer_by_layer_result_summary_with_degraded() {
        let result = LayerByLayerResult {
            nodes: vec![make_node_info("layer0", 2.5, true, false, false)],
            input_epsilon: 0.01,
            final_width: 0.025,
            degraded_at_node: Some(0),
            total_nodes: 1,
        };
        let summary = result.summary();
        assert!(summary.contains("WARNING"));
        assert!(summary.contains("SATURATED"));
    }
}
