// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Curated network-equivalence surface for external consumers.
//!
//! Equivalence verification proves that two networks `f` and `g` agree within a
//! tolerance, i.e. `||f(x) - g(x)|| < eps` for every `x` in an input region. It
//! does so by constructing a *difference network* `h(x) = f(x) - g(x)` (both
//! networks share the same input, with a final subtraction node) and verifying
//! that every output of `h` lies within `[-eps, eps]` via bound propagation.
//!
//! Configuration reuses the curated `ny_api::verify::PropagationConfig`
//! (IBP / CROWN / alpha-CROWN), so no equivalence-specific config type is
//! needed.

/// Verify that two networks produce equivalent outputs within `epsilon`.
///
/// Builds the difference network `h(x) = f(x) - g(x)` and checks that all of
/// `h`'s outputs fall within `[-epsilon, epsilon]` over the input region. The
/// returned [`EquivalenceResult`] reports the worst-case proven output
/// difference.
pub use ny_propagate::verify_equivalence;

/// Build the difference network `h(x) = f(x) - g(x)` from two graph networks.
///
/// Both networks share the same input; node names are prefixed to avoid
/// collisions, and a final subtraction node produces the element-wise output
/// difference. Useful when you want to inspect or verify the difference network
/// directly instead of going through [`verify_equivalence`].
pub use ny_propagate::build_difference_network;

/// Outcome of an equivalence check: equivalent, not-equivalent (bounds too
/// loose to prove agreement), or unknown (timeout / inconclusive), each
/// carrying the relevant worst-case output-difference bound.
pub use ny_propagate::EquivalenceResult;
