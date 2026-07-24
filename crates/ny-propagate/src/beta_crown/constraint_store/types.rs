// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared types for the arena-based constraint store.
//!
//! Contains enums, headers, and view types used by both
//! [`ArenaConstraintStore`](super::ArenaConstraintStore) and
//! [`DomainConstraintStore`](super::DomainConstraintStore).

use std::mem::{align_of, size_of};

/// Sense of a linear constraint: `≤` or `≥`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum ConstraintSense {
    /// Less than or equal (coeffs · x ≤ bias).
    #[default]
    Le = 0,
    /// Greater than or equal (coeffs · x ≥ bias).
    Ge = 1,
}

/// Origin of a constraint for selective clipping passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum ConstraintOrigin {
    /// Constraint from ReLU/activation split.
    #[default]
    Split = 0,
    /// Constraint from output property (verification spec).
    Output = 1,
    /// Constraint derived from bound propagation.
    BoundProp = 2,
}

/// Compact header for a single constraint in the arena.
///
/// 16 bytes aligned for efficient GPU transfer.
/// Coefficients and indices share the same offset since they have 1:1 correspondence.
///
/// Implements `bytemuck::Pod` for zero-copy GPU buffer transfer.
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct ConstraintHeader {
    /// Index into both `coeffs` and `indices` arenas.
    pub data_start: u32,
    /// Number of coefficients/indices.
    pub data_len: u16,
    /// Constraint origin (Split=0, Output=1, BoundProp=2).
    pub origin: u8,
    /// Constraint sense (Le=0, Ge=1).
    pub sense: u8,
    /// Right-hand side bias.
    pub bias: f32,
    /// Padding to 16 bytes for alignment.
    _padding: u32,
}

// Compile-time ABI guard for CPU↔WGSL interop. If any of these change, the
// host and shader layouts diverge and GPU constraint evaluation becomes unsound.
const _: [(); 16] = [(); size_of::<ConstraintHeader>()];
const _: [(); 4] = [(); align_of::<ConstraintHeader>()];
const _: [(); 0] = [(); std::mem::offset_of!(ConstraintHeader, data_start)];
const _: [(); 4] = [(); std::mem::offset_of!(ConstraintHeader, data_len)];
const _: [(); 6] = [(); std::mem::offset_of!(ConstraintHeader, origin)];
const _: [(); 7] = [(); std::mem::offset_of!(ConstraintHeader, sense)];
const _: [(); 8] = [(); std::mem::offset_of!(ConstraintHeader, bias)];

impl ConstraintHeader {
    /// Create a new constraint header.
    pub fn new(
        data_start: u32,
        data_len: u16,
        bias: f32,
        sense: ConstraintSense,
        origin: ConstraintOrigin,
    ) -> Self {
        Self {
            data_start,
            data_len,
            origin: origin as u8,
            sense: sense as u8,
            bias,
            _padding: 0,
        }
    }

    /// Get the constraint sense.
    ///
    /// Returns `Err` if the byte is not a valid `ConstraintSense` discriminant,
    /// which indicates GPU buffer corruption (#2261).
    pub fn sense(&self) -> ny_core::Result<ConstraintSense> {
        match self.sense {
            0 => Ok(ConstraintSense::Le),
            1 => Ok(ConstraintSense::Ge),
            invalid => Err(ny_core::NyError::InternalError(format!(
                "ConstraintHeader: invalid sense byte {} (expected 0=Le or 1=Ge) — \
                 possible GPU buffer corruption (#2261)",
                invalid
            ))),
        }
    }

    /// Get the constraint origin.
    ///
    /// Returns `Err` if the byte is not a valid `ConstraintOrigin` discriminant,
    /// which indicates GPU buffer corruption (#2261).
    pub fn origin(&self) -> ny_core::Result<ConstraintOrigin> {
        match self.origin {
            0 => Ok(ConstraintOrigin::Split),
            1 => Ok(ConstraintOrigin::Output),
            2 => Ok(ConstraintOrigin::BoundProp),
            invalid => Err(ny_core::NyError::InternalError(format!(
                "ConstraintHeader: invalid origin byte {} (expected 0-2) — \
                 possible GPU buffer corruption (#2261)",
                invalid
            ))),
        }
    }

    /// Get the range of indices in the arena.
    pub fn data_range(&self) -> std::ops::Range<usize> {
        let start = self.data_start as usize;
        let end = start + self.data_len as usize;
        start..end
    }
}

/// A single linear constraint row (view into arena).
///
/// This is a lightweight view into the `ArenaConstraintStore` for iteration.
#[derive(Debug, Clone)]
pub struct LinearConstraintRef<'a> {
    /// Variable indices (sparse representation).
    pub indices: &'a [u32],
    /// Coefficients corresponding to indices.
    pub coeffs: &'a [f32],
    /// Right-hand side bias.
    pub bias: f32,
    /// Constraint sense.
    pub sense: ConstraintSense,
    /// Constraint origin.
    pub origin: ConstraintOrigin,
}
