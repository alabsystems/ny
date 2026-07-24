// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Transform layers for bound propagation.
//!
//! Each operator family lives in its own submodule:
//! - `reshape`: ReshapeLayer, FlattenLayer
//! - `pad`: PadLayer
//! - `resize`: ResizeLayer
//! - `transpose`: TransposeLayer
//! - `tile`: TileLayer
//! - `expand`: ExpandLikeLastAxisLayer
//! - `gather`: GatherLayer
//! - `accumulate`: ScatterAddLayer, IndexAddLayer
//! - `scatter_nd`: ScatterNdLayer
//! - `slice`: SliceLayer
//! - `squeeze`: SqueezeLayer, UnsqueezeLayer

mod accumulate;
mod expand;
mod gather;
mod pad;
mod reshape;
mod reshape_preserve_leading_axis;
mod resize;
mod scatter_nd;
mod slice;
mod squeeze;
mod tile;
mod transpose;

pub use accumulate::{IndexAddLayer, ScatterAddLayer};
pub use expand::ExpandLikeLastAxisLayer;
pub use gather::GatherLayer;
pub use pad::{PadLayer, PadMode};
pub use reshape::{FlattenLayer, ReshapeLayer};
pub use resize::ResizeLayer;
pub use scatter_nd::ScatterNdLayer;
pub use slice::SliceLayer;
pub use squeeze::{SqueezeLayer, UnsqueezeLayer};
pub use tile::TileLayer;
pub use transpose::{normalize_transpose_perm_for_rank, TransposeLayer};

#[cfg(test)]
mod reshape_preserve_leading_axis_tests;
#[cfg(test)]
mod reshape_tests;
#[cfg(test)]
mod tests;
