// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_core::Result;

use crate::bounds::LinearBounds;

use super::PatchesLinearBounds;

/// Wrapper enum: the backward engine operates on this instead of bare LinearBounds.
///
/// This is the key type that enables Patches mode without changing LinearBounds.
/// The backward engine loop dispatches on this enum. Layers that don't support
/// Patches natively receive Dense (via automatic conversion).
// LinearBounds is 224 bytes (hot path in backward loop); Patches is heap-allocated
// via Box. The size difference is acceptable — boxing Dense would add deref overhead
// on every backward step for non-CNN networks (the common case).
#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub(crate) enum CrownBounds {
    /// Standard dense A-matrix bounds. Wraps the existing LinearBounds unchanged.
    Dense(LinearBounds),
    /// Sparse conv patches bounds for CNN optimization.
    Patches(Box<PatchesLinearBounds>),
}

impl CrownBounds {
    /// Get as Dense, converting if necessary. Consumes self.
    #[track_caller]
    pub(crate) fn into_dense(self) -> Result<LinearBounds> {
        match self {
            CrownBounds::Dense(lb) => Ok(lb),
            CrownBounds::Patches(pb) => pb.to_dense(),
        }
    }

    /// Convert Patches to Dense in-place, then return mutable ref to LinearBounds.
    ///
    /// If already Dense, returns the inner LinearBounds directly.
    /// If Patches, materializes to Dense first.
    #[track_caller]
    pub(crate) fn ensure_dense(&mut self) -> Result<&mut LinearBounds> {
        if matches!(self, CrownBounds::Patches(_)) {
            let dense = match std::mem::replace(self, CrownBounds::Dense(LinearBounds::identity(0)))
            {
                CrownBounds::Patches(pb) => pb.to_dense()?,
                _ => unreachable!(),
            };
            *self = CrownBounds::Dense(dense);
        }
        match self {
            CrownBounds::Dense(lb) => Ok(lb),
            CrownBounds::Patches(_) => unreachable!(),
        }
    }

    /// Total heap memory used by the current bounds representation, in bytes.
    pub(crate) fn memory_bytes(&self) -> usize {
        match self {
            CrownBounds::Dense(lb) => lb.memory_bytes(),
            CrownBounds::Patches(pb) => pb.memory_bytes(),
        }
    }

    /// Whether this is currently in Patches mode.
    pub(crate) fn is_patches(&self) -> bool {
        matches!(self, CrownBounds::Patches(_))
    }
}
