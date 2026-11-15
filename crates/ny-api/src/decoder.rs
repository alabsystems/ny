// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Complete-vocabulary affine-head verification over an independently supplied
//! hidden-state enclosure. The guarantee covers this head and box; decoder
//! trajectory, tokenizer, normalization and runtime lowering require their own
//! certificates before composing an end-to-end decoder result.

pub use ny_propagate::certify::residency::{CancellationToken, MemoryAuthority, MemoryReceipt};
pub use ny_propagate::certify::work::{
    LeaseClass, PersistentReservation, WorkBudget, WorkLease, WorkLimits, WorkReceipt, WorkRefusal,
    WorkRequest, WorkReserve,
};
pub use ny_propagate::decoder::{
    prepare_model_vocab_tree, prepare_vocab_tree, replay_next_token, replay_next_token_json,
    verify_dense_next_token, verify_next_token, AffineHead, Binary32Reduction, BudgetedWireExport,
    CertifiedNextToken, DecoderRefusal, GroupBounds, GroupEvidence, HeadArithmetic, HiddenBox,
    ModelTreePreparationStats, ModelTreeQueryStats, ModelTreeReport, NextTokenOutcome,
    NextTokenProvenance, NextTokenReport, PreparedModelVocabTree, PreparedVocabTree,
    ResidentAffineHead, ResidentAffineQuery, SharedAffineHead, VocabEvidence, VocabOptions,
    VocabStats, WireExportStats,
};

pub use ny_propagate::decoder::normalized_head::shared::{
    replay_shared_normalized_next_token, verify_shared_normalized_next_token,
};
pub use ny_propagate::decoder::normalized_head::wire::{
    export_shared_normalized_json, normalized_publication_work_bound, normalized_wire_size_bound,
    replay_shared_normalized_json, NormalizedWireExport, NormalizedWireReplay, NormalizedWireStats,
};
/// Composed mathematical real final-normalization/head proofs require an
/// explicitly supplied pre-normalization box; no upstream or runtime proof.
pub use ny_propagate::decoder::normalized_head::{
    normalized_work_bound, replay_normalized_next_token, verify_normalized_next_token,
    CertifiedNormalizedNextToken, FinalNormalization, FinalNormalizationKind,
    NormalizedHeadEvidence, NormalizedHeadOptions, NormalizedHeadRefusal, NormalizedHeadReport,
    NormalizedHeadStats, NormalizedNextTokenOutcome, NormalizedRowEvidence,
    NormalizedSupportMethod,
};
