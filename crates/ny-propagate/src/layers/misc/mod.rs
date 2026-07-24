// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Miscellaneous layers for bound propagation.
//!
//! Layers that don't fit neatly into other categories:
//! - Skip/placeholder layers (`SkipMergeLayer`, `OpaqueSkipLayer`)
//! - Piecewise constant layers (`FloorLayer`, `CeilLayer`, `RoundLayer`, `SignLayer`, `TruncLayer`)
//! - Element-wise nonlinear (`ReciprocalLayer`)
//! - Conditional (`WhereLayer`)
//! - Index-producing (`NonZeroLayer`)

pub(crate) mod compare;
mod nonzero;
mod piecewise_constant;
mod qdq_perturbation;
pub(crate) mod reciprocal;
mod skip_merge;
mod where_layer;

pub use compare::{CompareLayer, CompareOp};
pub use nonzero::NonZeroLayer;
pub use piecewise_constant::{CeilLayer, FloorLayer, RoundLayer, SignLayer, TruncLayer};
pub use qdq_perturbation::QdqPerturbationLayer;
pub use reciprocal::ReciprocalLayer;
pub use skip_merge::{OpaqueSkipLayer, SkipMergeLayer};
pub use where_layer::WhereLayer;
