// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};

/// Heuristic for selecting which neuron to split.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum BranchingHeuristic {
    /// Split the neuron with largest bound width (u - l).
    #[default]
    LargestBoundWidth,
    /// Split the neuron that most affects the output bound (BaBSR-like).
    BoundImpact,
    /// Filtered Smart Branching (FSB): evaluate a small set of high-scoring BaBSR candidates
    /// by estimating both child bounds and choosing the best worst-child improvement.
    FilteredSmartBranching,
    /// kFSB: k-filtered smart branching with configurable reduce operation.
    /// Uses both BaBSR alpha score and intercept backup score, evaluates top-k candidates,
    /// and combines branch scores using the configured reduce_op.
    /// Matches alpha-beta-CROWN kfsb heuristic.
    Kfsb,
    /// kFSB with intercept-only scoring (no alpha/CROWN coefficient weighting).
    /// Uses pure relaxation gap (triangle intercept) for scoring.
    /// Matches alpha-beta-CROWN kfsb-intercept-only heuristic.
    KfsbInterceptOnly,
    /// Split neurons in order (layer by layer, neuron by neuron).
    Sequential,
    /// Input splitting: divide input space instead of ReLU activation space.
    /// More effective than ReLU splitting for small networks with tight input bounds
    /// (e.g., ACAS-Xu with 5 input dimensions). Each split halves one input dimension,
    /// creating tighter output bounds for each subdomain.
    InputSplit,
    /// GenBaB: branch-and-bound for general nonlinearities (GeLU, Sigmoid, Tanh, etc.).
    /// Uses BBPS heuristic with uniform or optimized branching points.
    /// See `designs/2026-01-28-genbab-branching.md` for details.
    GenBaB(crate::beta_crown::nonlinear_branching::NonlinearBranchingConfig),
}
