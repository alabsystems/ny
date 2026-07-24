// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared inverted-bounds repair utilities for propagation/export call sites.

use ndarray::ArrayD;

/// Strategy for repairing inverted bounds (`lower > upper`) after propagation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InversionRepair {
    /// Swap lower/upper to recover the tightest sound interval.
    Swap,
    /// Widen repaired elements to `[-inf, +inf]`.
    WidenToInf,
    /// Widen repaired elements to `[-bound, +bound]`.
    WidenToFallback(f32),
}

/// Repair inverted bounds in flat storage and return the number of repaired elements.
///
/// `Swap` only repairs true finite inversions (`lower > upper`). If either
/// endpoint is NaN, the element is left unchanged so callers must sanitize NaN
/// before using the swap strategy. The widening strategies treat NaN as
/// unrecoverable corruption and widen conservatively.
#[must_use]
pub fn repair_inverted_bounds(
    lower: &mut [f32],
    upper: &mut [f32],
    strategy: InversionRepair,
) -> usize {
    assert_eq!(
        lower.len(),
        upper.len(),
        "repair_inverted_bounds: lower len {} != upper len {}",
        lower.len(),
        upper.len()
    );

    let mut repaired = 0usize;
    for (l, u) in lower.iter_mut().zip(upper.iter_mut()) {
        let should_repair = match strategy {
            InversionRepair::Swap => *l > *u,
            InversionRepair::WidenToInf | InversionRepair::WidenToFallback(_) => {
                *l > *u || l.is_nan() || u.is_nan()
            }
        };
        if should_repair {
            match strategy {
                InversionRepair::Swap => std::mem::swap(l, u),
                InversionRepair::WidenToInf => {
                    *l = f32::NEG_INFINITY;
                    *u = f32::INFINITY;
                }
                InversionRepair::WidenToFallback(bound) => {
                    *l = -bound;
                    *u = bound;
                }
            }
            repaired += 1;
        }
    }

    repaired
}

/// Repair inverted bounds in ndarray storage and return the number of repaired elements.
///
/// See [`repair_inverted_bounds`] for strategy semantics.
#[must_use]
pub fn repair_inverted_bounds_nd(
    lower: &mut ArrayD<f32>,
    upper: &mut ArrayD<f32>,
    strategy: InversionRepair,
) -> usize {
    assert_eq!(
        lower.shape(),
        upper.shape(),
        "repair_inverted_bounds_nd: lower shape {:?} != upper shape {:?}",
        lower.shape(),
        upper.shape()
    );

    let mut repaired = 0usize;
    ndarray::Zip::from(&mut *lower)
        .and(&mut *upper)
        .for_each(|l, u| {
            let should_repair = match strategy {
                InversionRepair::Swap => *l > *u,
                InversionRepair::WidenToInf | InversionRepair::WidenToFallback(_) => {
                    *l > *u || l.is_nan() || u.is_nan()
                }
            };
            if should_repair {
                match strategy {
                    InversionRepair::Swap => std::mem::swap(l, u),
                    InversionRepair::WidenToInf => {
                        *l = f32::NEG_INFINITY;
                        *u = f32::INFINITY;
                    }
                    InversionRepair::WidenToFallback(bound) => {
                        *l = -bound;
                        *u = bound;
                    }
                }
                repaired += 1;
            }
        });

    repaired
}
