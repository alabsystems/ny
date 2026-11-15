// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Native proof-only transport of signed differences through paired ReLUs.
mod chain;
mod engine;
pub use engine::{
    BoundReport, BoxDomain, LpProofReport, NativeLpReport, OutputTemplateCover, PairedChain,
    PairedPhase, ParallelWorkers, PhaseChildCertificate, PhaseChildStatus, PhaseCoverReport,
    PhaseInfeasibilityReport, PhaseLpReport, PhaseProofNode, PhaseProofTree, PhaseTreeReport,
    PreparedRootRelaxation, RootRelaxationReport, SearchReport, SoftGuardCover, SoftGuardLeaf,
    Stage,
};
