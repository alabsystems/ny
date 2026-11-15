// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Original model ownership, certified reference boxes and immutable prefixes.
//!
//! Construction is independently admitted before any requested row product.
//! These owners do not certify that clipping, retention or precision demands
//! have been delivered, and do not themselves change the default root route.

mod cli_compilation;
mod cli_proof;
mod compiler;
mod construction;
mod domain;
mod head_preactivation_rows;
mod ir;
mod liveness;
mod model_dag_liveness;
mod ordinary_pass;
mod precision;
mod prefix;
mod publication;
mod ranked_row_advice;
mod reference_execution;
mod replay;
mod retained;
mod retained_input_rows;
mod root_adapter;
mod sparse_input_rows;
mod terminal;

pub use crate::margin_row::input_clipping::InputSplitSide as PreparedInputSplitSide;
pub use cli_compilation::PreparedRootCompilation;
pub use cli_proof::{
    PreparedRootAttemptReceipt, PreparedRootConsumerOptions, PreparedRootOpen, PreparedRootOutcome,
    PreparedRootProof, PreparedRootProofOptions, PreparedRootRetention, PreparedRootSession,
    PreparedRowStorageReceipt,
};
pub use head_preactivation_rows::{PreparedHeadPreactivationRow, PreparedHeadPreactivationRows};
pub use model_dag_liveness::PreparedModelDagLiveness;
pub use ranked_row_advice::{
    PreparedRankedInputRow, PreparedRankedRowAdvice, PreparedRankedRowAdviceLayer,
};
pub use reference_execution::{PreparedReferenceArithmetic, PreparedReferenceReceipt};
pub use replay::{PreparedReplayLeaf, PreparedReplayLiteral, PreparedReplayStats};
pub use retained::PreparedSplit as PreparedInputSplit;
pub use retained_input_rows::{PreparedInputHalfspaceBatch, PreparedInputRows};
pub use sparse_input_rows::{
    PreparedSparseInputHalfspace, PreparedSparseInputRow, PreparedSparseInputRows,
};

#[cfg(test)]
pub(super) use compiler::OriginalModel;
pub(super) use compiler::{virtual_conv_allocation_bytes, virtual_conv_from_original};
pub(super) use construction::PreparedRootConstruction;
#[cfg(test)]
pub(super) use construction::{build_prepared_root, PreparedConstructionOptions};
#[cfg(test)]
pub(super) use domain::PreparedSplitSite;
pub(super) use domain::{
    PreparedDomain, PreparedDomainFrame, PreparedDomainPath, PreparedPhase, PreparedPhaseSide,
};
#[cfg(test)]
pub(super) use ordinary_pass::build_prepared_ordinary_rows;
pub(super) use ordinary_pass::{
    concretize_prepared_pass, prepared_ordinary_walk_plan, PreparedOrdinaryRowOwner,
    PreparedOrdinaryRows, PreparedOrdinaryWalkPlan,
};
pub(super) use prefix::{CertifiedTensorBox, OriginalRoot, PreparedPrefix, PreparedPrefixFrame};
#[cfg(test)]
pub(super) use publication::share_prepared_root;
pub(super) use publication::PreparedRootSource;
#[cfg(test)]
pub(super) use root_adapter::project_prepared_root;
pub(super) use root_adapter::PreparedRootAdapter;
