// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! NY-owned compatibility attributes for contract-annotated Rust source.
//!
//! This crate exists only so NY source using `#[trust::requires]`,
//! `#[trust::ensures]`, `#[trust::invariant]`, or `#[trust::cite]` parses on
//! stable Rust. Each macro ignores its arguments and emits the annotated item
//! unchanged. The attributes do not check a condition, add a runtime assertion,
//! or prove anything.
//!
//! An external verification toolchain may interpret the same attribute
//! spellings before macro expansion. That behavior belongs to the toolchain;
//! this crate contains no verifier code and is not a copy of one.

#![forbid(unsafe_code)]

use proc_macro::TokenStream;

/// Stable-Rust compatibility marker for a precondition.
///
/// The condition tokens are ignored and the annotated item is preserved.
#[proc_macro_attribute]
pub fn requires(_condition: TokenStream, annotated_item: TokenStream) -> TokenStream {
    annotated_item
}

/// Stable-Rust compatibility marker for a postcondition.
///
/// The condition tokens are ignored and the annotated item is preserved.
#[proc_macro_attribute]
pub fn ensures(_condition: TokenStream, annotated_item: TokenStream) -> TokenStream {
    annotated_item
}

/// Stable-Rust compatibility marker for an invariant.
///
/// The invariant tokens are ignored and the annotated item is preserved.
#[proc_macro_attribute]
pub fn invariant(_condition: TokenStream, annotated_item: TokenStream) -> TokenStream {
    annotated_item
}

/// Stable-Rust compatibility marker for a proof citation.
///
/// The citation tokens are ignored and the annotated item is preserved. NY's
/// separate citation-integrity checks, not this macro, validate cited proofs.
#[proc_macro_attribute]
pub fn cite(_citation: TokenStream, annotated_item: TokenStream) -> TokenStream {
    annotated_item
}
