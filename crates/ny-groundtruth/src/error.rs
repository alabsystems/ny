// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Error type for ground-truth graph construction and verification.

use ny_core::NyError;

/// Errors raised while building or verifying ground-truth graphs.
///
/// The constant-handling variants implement the contract of
/// `docs/GEOMETRIC_GROUND_TRUTH_PLAN.md` §2.3: a constant that would have to
/// be *silently rounded* to enter the graph is rejected instead. (The plan's
/// alternative — emitting an interval-widened constant — is a follow-up; M1
/// is strictly reject-or-exact.)
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum GroundTruthError {
    /// A caller-supplied parameter is NaN or infinite.
    #[error("parameter `{name}` is not finite: {value}")]
    NonFiniteParameter {
        /// Parameter name as documented on the builder.
        name: String,
        /// Offending value.
        value: f64,
    },

    /// A caller-supplied f64 parameter does not round-trip exactly through
    /// f32 (the graph constant width). Silently rounding it would change the
    /// ground-truth function, so the builder rejects it (plan §2.3).
    #[error(
        "parameter `{name}` = {value} does not round-trip f64 -> f32 exactly; \
         ground-truth constants must enter the graph exactly (plan §2.3: \
         interval-widened constants are the follow-up, silent rounding is never allowed)"
    )]
    InexactParameter {
        /// Parameter name as documented on the builder.
        name: String,
        /// Offending value.
        value: f64,
    },

    /// A constant derived at build time (via exact rational arithmetic, e.g.
    /// `r^2` or a projection-matrix entry `a_i * a_j`) is not exactly
    /// representable in f32.
    #[error(
        "derived constant `{name}` is not exactly representable in f32; choose \
         parameters whose build-time products and sums stay within 24-bit \
         significands (plan §2.3)"
    )]
    InexactDerivedConstant {
        /// Description of the derived constant (e.g. `radius^2`).
        name: String,
    },

    /// An axis parameter is not exactly unit length. The projection
    /// `I - a a^T` used by the cylinder/cone/torus residuals is only a
    /// projection for `||a|| = 1`, so a non-unit axis would silently change
    /// the zero set away from the intended surface.
    #[error(
        "axis parameter `{name}` must be exactly unit length, got ||a||^2 = {norm_sq}; \
         in f32 the exactly-unit axes are the signed standard basis vectors \
         (sum-of-three-squares descent), so general fitted axes need the plan \
         §2.3 interval-widening follow-up"
    )]
    AxisNotUnit {
        /// Parameter name as documented on the builder.
        name: String,
        /// Exact `||a||^2`, rounded once to f64 for display.
        norm_sq: f64,
    },

    /// A parameter is outside its documented domain (e.g. `radius <= 0`).
    #[error("parameter `{name}` is degenerate: {reason}")]
    DegenerateParameter {
        /// Parameter name as documented on the builder.
        name: String,
        /// Why the value is rejected.
        reason: String,
    },

    /// A composition request is malformed (e.g. empty primitive list).
    #[error("invalid composition: {0}")]
    InvalidComposition(String),

    /// A `.gt.json` sidecar spec is malformed (wrong `format`, missing or
    /// ambiguous `builder`/`compose`, bad params shape, JSON syntax error).
    /// Parameter-*value* rejections surface as the builder errors above —
    /// the loader validates through the M1 exact-constant contract.
    #[error("invalid .gt.json ground-truth spec: {0}")]
    InvalidSidecar(String),

    /// A `.gt.json` sidecar file could not be read.
    #[error("failed to read ground-truth spec `{path}`: {reason}")]
    SidecarIo {
        /// Path as given by the caller.
        path: String,
        /// Underlying IO error description.
        reason: String,
    },

    /// An underlying ny error (graph construction, propagation, spec).
    #[error(transparent)]
    Ny(#[from] NyError),
}

/// Convenience result alias for this crate.
pub type Result<T> = std::result::Result<T, GroundTruthError>;
