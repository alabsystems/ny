// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Private raw-source root-cut boundary. Mathematical authority is constructed
//! within one borrowed call; no raw facet/support/scalar can be imported.
pub(crate) use super::graph_alpha::resnet_decompose::raw_bn::{
    bounded_fold_bytes, bounded_fold_work, extract_charged_raw_bn_chain,
    quote_charged_raw_bn_chain, ExtractionLimits, ExtractionQuote,
};
pub(crate) use crate::multineuron::{
    evaluate_raw_local_l1_root_cut, evaluate_raw_root_cut, evaluate_raw_triplet_root_cut,
    evaluate_raw_two_local_l1_root_cut, evaluate_raw_two_pair_root_cut,
    quote_raw_local_l1_root_cut, quote_raw_root_cut, quote_raw_triplet_root_cut,
    quote_raw_two_local_l1_root_cut, quote_raw_two_pair_root_cut, CompletedRawLocalL1RootCut,
    CompletedRawRootCut, CompletedRawTripletRootCut, CompletedRawTwoLocalL1RootCut,
    CompletedRawTwoPairRootCut, RawLocalL1RootCutQuote, RawRootCutLimits, RawRootCutQuote,
    RawTripletRootCutQuote, RawTwoLocalL1RootCutQuote, RawTwoPairRootCutQuote,
};

#[cfg(test)]
pub(crate) use crate::multineuron::test_support as raw_root_cut_test_support;

#[cfg(test)]
pub(crate) use crate::multineuron::two_pair_test_support as raw_two_pair_root_cut_test_support;

#[cfg(test)]
pub(crate) use crate::multineuron::triplet_test_support as raw_triplet_root_cut_test_support;

#[cfg(test)]
pub(crate) use crate::multineuron::local_l1_test_support as raw_local_l1_root_cut_test_support;

#[cfg(test)]
pub(crate) use crate::multineuron::two_local_l1_test_support as raw_two_local_l1_root_cut_test_support;
