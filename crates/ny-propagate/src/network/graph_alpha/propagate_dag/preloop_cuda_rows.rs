// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! DAG-alpha pre-loop adapter for the generic atomic CUDA-row transaction.

use std::collections::HashMap;

use ndarray::Array2;
use ny_tensor::BoundedTensor;

use crate::network::core::GraphNetwork;
use crate::network::graph_alpha::atomic_cuda_rows::{
    root_alpha_cuda_rows_enabled, AtomicCudaRowsCommit, AtomicCudaRowsOutcome,
    AtomicCudaRowsRefusal, AtomicCudaRowsRequest,
};

use super::super::runtime_state::DagAlphaRuntimeState;
use super::DagAlphaLoopContext;

const SUPPORTED_IDENTITY_ROWS: [usize; 2] = [99, 100];

pub(super) enum PreloopCudaRowsOutcome {
    NotRequested,
    RefusedBeforeCommit {
        refusal: AtomicCudaRowsRefusal,
    },
    CudaIntersection(BoundedTensor),
    ReferenceRetained {
        bounds: BoundedTensor,
        refusal: AtomicCudaRowsRefusal,
    },
    DeadlineExceeded,
}

fn identity_spec(rows: usize) -> Array2<f32> {
    let mut spec = Array2::zeros((rows, rows));
    for row in 0..rows {
        spec[(row, row)] = 1.0;
    }
    spec
}

impl GraphNetwork {
    /// Attempt the exact-dark identity-output transaction before the alpha loop.
    pub(super) fn atomic_preloop_cuda_rows(
        &self,
        ctx: &DagAlphaLoopContext<'_>,
        node_bounds: &HashMap<String, BoundedTensor>,
        runtime: &DagAlphaRuntimeState,
    ) -> PreloopCudaRowsOutcome {
        if !root_alpha_cuda_rows_enabled() {
            return PreloopCudaRowsOutcome::NotRequested;
        }
        if !SUPPORTED_IDENTITY_ROWS.contains(&ctx.output_dim) {
            return PreloopCudaRowsOutcome::RefusedBeforeCommit {
                refusal: AtomicCudaRowsRefusal::UnsupportedRowCount {
                    rows: ctx.output_dim,
                },
            };
        }
        let output_node = if self.output_node.is_empty() {
            let Some(name) = ctx.exec_order.last() else {
                return PreloopCudaRowsOutcome::RefusedBeforeCommit {
                    refusal: AtomicCudaRowsRefusal::MissingOutputNode,
                };
            };
            name.as_str()
        } else {
            self.output_node.as_str()
        };
        let Some(reference) = node_bounds.get(output_node) else {
            return PreloopCudaRowsOutcome::RefusedBeforeCommit {
                refusal: AtomicCudaRowsRefusal::MissingReferenceOutput,
            };
        };
        let Some(deadline) = ctx.config.deadline else {
            return PreloopCudaRowsOutcome::RefusedBeforeCommit {
                refusal: AtomicCudaRowsRefusal::MissingDeadline,
            };
        };
        let spec = identity_spec(ctx.output_dim);
        match AtomicCudaRowsRequest::new(
            self,
            ctx.input,
            output_node,
            node_bounds,
            runtime.graph(),
            &spec,
            reference,
            deadline,
        )
        .run()
        {
            AtomicCudaRowsOutcome::RefusedBeforeCommit {
                refusal: AtomicCudaRowsRefusal::DeadlineExceeded,
            } => PreloopCudaRowsOutcome::DeadlineExceeded,
            AtomicCudaRowsOutcome::RefusedBeforeCommit { refusal } => {
                PreloopCudaRowsOutcome::RefusedBeforeCommit { refusal }
            }
            AtomicCudaRowsOutcome::Committed(AtomicCudaRowsCommit::CudaIntersection(bounds)) => {
                PreloopCudaRowsOutcome::CudaIntersection(*bounds)
            }
            AtomicCudaRowsOutcome::Committed(AtomicCudaRowsCommit::ReferenceRetained {
                bounds,
                refusal: AtomicCudaRowsRefusal::DeadlineExceeded,
            }) => {
                drop(bounds);
                PreloopCudaRowsOutcome::DeadlineExceeded
            }
            AtomicCudaRowsOutcome::Committed(AtomicCudaRowsCommit::ReferenceRetained {
                bounds,
                refusal,
            }) => PreloopCudaRowsOutcome::ReferenceRetained {
                bounds: *bounds,
                refusal,
            },
            AtomicCudaRowsOutcome::Committed(AtomicCudaRowsCommit::DeadlineExceeded) => {
                PreloopCudaRowsOutcome::DeadlineExceeded
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preloop_identity_rows_are_exact() {
        for rows in [99, 100] {
            let spec = identity_spec(rows);
            assert_eq!(spec.shape(), [rows, rows]);
            for row in 0..rows {
                assert_eq!(
                    spec.row(row).iter().filter(|&&value| value != 0.0).count(),
                    1
                );
                assert_eq!(spec[(row, row)], 1.0);
            }
        }
    }
}
