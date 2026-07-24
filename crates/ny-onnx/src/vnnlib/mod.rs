// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! VNN-LIB property specification format parser.
//!
//! **Supported Version:** VNN-LIB 1.0 syntax + partial VNN-LIB 2.0 tensor declarations.
//!
//! VNN-LIB is the standard property specification language for VNN-COMP
//! (Verification of Neural Networks Competition). It uses SMT-LIB v2 syntax
//! to specify input constraints and output properties.
//!
//! # Format Specification
//!
//! - Variable declarations: `(declare-const X_0 Real)` for inputs, `(declare-const Y_0 Real)` for outputs
//! - Input bounds: `(assert (<= X_0 upper))` and `(assert (>= X_0 lower))`
//! - Output constraints: `(assert (<= Y_0 Y_1))` means property holds if Y_0 ≤ Y_1
//!
//! # Safety Properties
//!
//! VNN-LIB specifies **unsafe regions** - the property is verified (safe) if
//! the neural network output CANNOT satisfy the output constraints for any
//! input in the specified region.
//!
//! # Version Compatibility
//!
//! This parser supports VNN-LIB 1.0 syntax:
//! - `(declare-const X_N Real)` variable declarations
//! - Standard comparison operators `<= >= < > =`
//! - Boolean connectives `and`, `or`
//!
//! VNN-LIB 2.0 features (released 2025-12-15) are partially supported:
//! - `(declare-input X Type [shape])` tensor declarations
//! - `(declare-output Y Type [shape])` tensor declarations
//! - Tensor-indexed variable references like `X[0]`, `Y[1,2]`, or `(select X 0)`
//! - `(declare-network ...)` wrapper (input/output declarations are parsed; network metadata ignored)
//! - Linear arithmetic in constraints (add/sub, mul/div by constants)
//!
//! Still unsupported:
//! - Non-linear arithmetic expressions within constraints
//!
//! If a `(vnnlib-version 2.0)` declaration is detected, a warning is emitted for
//! unsupported features (e.g., non-linear arithmetic).
//!
//! # Usage
//!
//! ```rust,no_run
//! use ny_onnx::vnnlib::{load_vnnlib, VnnLibSpec};
//!
//! let spec = load_vnnlib("property.vnnlib").unwrap();
//! println!("Inputs: {}, Outputs: {}", spec.num_inputs, spec.num_outputs);
//! for (i, (lower, upper)) in spec.input_bounds.iter().enumerate() {
//!     println!("  X_{}: [{}, {}]", i, lower, upper);
//! }
//! ```

mod certified_input_box;
/// Full-coverage dual-network formula DNF extraction (relational gate-flip).
pub mod dual_formula;
mod normalize;
mod parser;
mod spec;
mod syntax;

#[cfg(test)]
mod tests;

pub use certified_input_box::{
    load_vnnlib_with_certified_input_box, parse_vnnlib_with_certified_input_box, CertifiedInputBox,
};
/// VNN-LIB parsers for file-based and in-memory property specifications.
pub use parser::{load_vnnlib, parse_vnnlib};
pub use parser::{load_vnnlib_assignment_declarations, parse_vnnlib_assignment_declarations};
/// Parsed VNN-LIB specification types and output-constraint representation.
pub use spec::{
    DeclaredNetwork, DualNetworkProperty, DualNetworkSpec, DualNetworkValidation,
    IsomorphicAtomRelation, IsomorphicOutputAtom, NetworkRelation, OutputConstraint,
    TensorDeclaration, TensorDeclarationKind, VnnLibSpec,
};

#[cfg(test)]
pub(crate) use parser::contains_output_constraint;
#[cfg(test)]
pub(crate) use syntax::{
    get_number, parse_expr, parse_expressions, parse_var_index, resolve_var_info, tokenize, Expr,
};
