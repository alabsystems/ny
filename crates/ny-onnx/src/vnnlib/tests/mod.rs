// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

pub(crate) use super::{
    contains_output_constraint, get_number, load_vnnlib, parse_expr, parse_expressions,
    parse_var_index, parse_vnnlib, parse_vnnlib_assignment_declarations, resolve_var_info,
    tokenize, Expr, OutputConstraint, TensorDeclarationKind, VnnLibSpec,
};
pub(crate) use ny_core::NyError;

// Avoid re-exporting vnnlib::normalize to prevent test module name collisions.
mod bounds;
mod check_unsafe;
#[cfg(feature = "external-vnncomp")]
mod dual_formula_real;
mod gz;
mod normalize_output;
mod output_constraints;
mod parse;
mod parse_memo;
mod shrink;
mod spec;
mod tokenize_tests;
mod v20;
mod version;

pub(crate) fn assert_invalid_spec_contains(err: NyError, needle: &str) {
    match err {
        NyError::InvalidSpec(msg) => {
            assert!(
                msg.contains(needle),
                "Expected InvalidSpec to contain '{}', got '{}'",
                needle,
                msg
            );
        }
        other => panic!("Expected InvalidSpec error, got {:?}", other),
    }
}
