// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_propagate::{BranchingHeuristic, KfsbReduceOp};
use pyo3::prelude::*;

/// Branching heuristic for β-CROWN search.
#[pyclass(name = "BranchingHeuristic", from_py_object)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PyBranchingHeuristic {
    /// Split the neuron with largest bound width (u - l).
    LargestBoundWidth,
    /// Split the neuron that most affects the output bound (BaBSR-like).
    BoundImpact,
    /// Filtered Smart Branching: evaluate FSB candidates and choose best.
    FilteredSmartBranching,
    /// k-filtered smart branching with configurable reduce operation.
    Kfsb,
    /// kFSB with intercept-only scoring.
    KfsbInterceptOnly,
    /// Split neurons in sequential order.
    Sequential,
    /// Input space splitting (effective for small networks).
    InputSplit,
    /// GenBaB: general nonlinearity branching (GeLU, Sigmoid, Tanh).
    GenBaB,
}

#[pymethods]
impl PyBranchingHeuristic {
    pub(crate) fn __repr__(&self) -> String {
        match self {
            PyBranchingHeuristic::LargestBoundWidth => "BranchingHeuristic.LargestBoundWidth",
            PyBranchingHeuristic::BoundImpact => "BranchingHeuristic.BoundImpact",
            PyBranchingHeuristic::FilteredSmartBranching => {
                "BranchingHeuristic.FilteredSmartBranching"
            }
            PyBranchingHeuristic::Kfsb => "BranchingHeuristic.Kfsb",
            PyBranchingHeuristic::KfsbInterceptOnly => "BranchingHeuristic.KfsbInterceptOnly",
            PyBranchingHeuristic::Sequential => "BranchingHeuristic.Sequential",
            PyBranchingHeuristic::InputSplit => "BranchingHeuristic.InputSplit",
            PyBranchingHeuristic::GenBaB => "BranchingHeuristic.GenBaB",
        }
        .to_string()
    }
}

impl From<PyBranchingHeuristic> for BranchingHeuristic {
    fn from(h: PyBranchingHeuristic) -> Self {
        match h {
            PyBranchingHeuristic::LargestBoundWidth => BranchingHeuristic::LargestBoundWidth,
            PyBranchingHeuristic::BoundImpact => BranchingHeuristic::BoundImpact,
            PyBranchingHeuristic::FilteredSmartBranching => {
                BranchingHeuristic::FilteredSmartBranching
            }
            PyBranchingHeuristic::Kfsb => BranchingHeuristic::Kfsb,
            PyBranchingHeuristic::KfsbInterceptOnly => BranchingHeuristic::KfsbInterceptOnly,
            PyBranchingHeuristic::Sequential => BranchingHeuristic::Sequential,
            PyBranchingHeuristic::InputSplit => BranchingHeuristic::InputSplit,
            PyBranchingHeuristic::GenBaB => BranchingHeuristic::GenBaB(Default::default()),
        }
    }
}

/// Reduce operation for kFSB branching scores.
#[pyclass(name = "KfsbReduceOp", from_py_object)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PyKfsbReduceOp {
    /// Conservative: take worst-case of both branches (default).
    Min,
    /// Optimistic: take best-case of both branches.
    Max,
    /// Balanced: average of both branches.
    Mean,
}

#[pymethods]
impl PyKfsbReduceOp {
    pub(crate) fn __repr__(&self) -> String {
        match self {
            PyKfsbReduceOp::Min => "KfsbReduceOp.Min",
            PyKfsbReduceOp::Max => "KfsbReduceOp.Max",
            PyKfsbReduceOp::Mean => "KfsbReduceOp.Mean",
        }
        .to_string()
    }
}

impl From<PyKfsbReduceOp> for KfsbReduceOp {
    fn from(op: PyKfsbReduceOp) -> Self {
        match op {
            PyKfsbReduceOp::Min => KfsbReduceOp::Min,
            PyKfsbReduceOp::Max => KfsbReduceOp::Max,
            PyKfsbReduceOp::Mean => KfsbReduceOp::Mean,
        }
    }
}
