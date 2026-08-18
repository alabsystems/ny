// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::fmt;
use std::mem::size_of;

use ny_core::NyError;

pub(super) const RESIDENT_BAB_COMPOSE_POLL_STRIDE: usize = 1024;
// Conservative bucket-side charge: hashbrown control bytes, alignment, and
// allocator metadata are covered in addition to the key/value payload.
const HASH_BUCKET_OVERHEAD_BYTES: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::beta_crown::engine::graph) struct ResidentBabAdapterHostCapV1 {
    /// Absolute adapter-host byte ceiling for the entire simultaneous live
    /// set, including the baseline and composer output/scratch.
    pub limit_bytes: usize,
    /// Caller-owned bytes already live when composition starts. This must
    /// include every borrowed topology, wire/static artifact, history, beta,
    /// alpha, endpoint, and source object that overlaps the operation. The
    /// returned peak/retained accounting already includes this baseline.
    pub resident_bytes_before: usize,
}

#[derive(Debug)]
pub(in crate::beta_crown::engine::graph) enum ResidentBabComposeErrorV1 {
    /// Ordinary pre-open capability miss. Contract corruption remains
    /// `Invalid`, and a deadline remains a timeout rather than fallback.
    Unsupported(&'static str),
    Deadline(NyError),
    Invalid(NyError),
    Capacity {
        required_bytes: usize,
        limit_bytes: usize,
    },
    AllocationRefused(&'static str),
}

impl ResidentBabComposeErrorV1 {
    /// Whether an untouched pre-open caller may preserve the legacy path.
    /// This says nothing about fallback after a provider/lease has been opened.
    pub(super) fn allows_preopen_legacy_fallback(&self) -> bool {
        matches!(
            self,
            Self::Unsupported(_) | Self::Capacity { .. } | Self::AllocationRefused(_)
        )
    }
}

impl fmt::Display for ResidentBabComposeErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported(reason) => {
                write!(formatter, "retained-BaB v1 is unsupported: {reason}")
            }
            Self::Deadline(error) => error.fmt(formatter),
            Self::Invalid(error) => error.fmt(formatter),
            Self::Capacity {
                required_bytes,
                limit_bytes,
            } => write!(
                formatter,
                "retained-BaB adapter host cap exceeded: required={required_bytes}, limit={limit_bytes}"
            ),
            Self::AllocationRefused(label) => {
                write!(formatter, "retained-BaB {label} allocation was refused")
            }
        }
    }
}

impl From<NyError> for ResidentBabComposeErrorV1 {
    fn from(error: NyError) -> Self {
        match error {
            error @ NyError::DeadlineExceeded(_) => Self::Deadline(error),
            NyError::CpuMemoryExceeded {
                required_bytes,
                budget_bytes,
                ..
            } => Self::Capacity {
                required_bytes,
                limit_bytes: budget_bytes,
            },
            error => Self::Invalid(error),
        }
    }
}

pub(super) fn invalid(message: impl Into<String>) -> ResidentBabComposeErrorV1 {
    ResidentBabComposeErrorV1::Invalid(NyError::InvalidSpec(message.into()))
}

pub(super) fn checked_add(
    total: &mut usize,
    value: usize,
) -> Result<(), ResidentBabComposeErrorV1> {
    *total = total
        .checked_add(value)
        .ok_or_else(|| invalid("retained-BaB adapter-host accounting overflows usize"))?;
    Ok(())
}

pub(super) fn checked_elements<T>(count: usize) -> Result<usize, ResidentBabComposeErrorV1> {
    count
        .checked_mul(size_of::<T>())
        .ok_or_else(|| invalid("retained-BaB adapter-host element bytes overflow usize"))
}

pub(super) fn checked_hash_entries<K, V>(count: usize) -> Result<usize, ResidentBabComposeErrorV1> {
    let bucket = size_of::<(K, V)>()
        .checked_add(HASH_BUCKET_OVERHEAD_BYTES)
        .ok_or_else(|| invalid("retained-BaB hash bucket charge overflows"))?;
    count
        .checked_mul(bucket)
        .ok_or_else(|| invalid("retained-BaB hash table charge overflows"))
}

pub(super) struct ResidentBabHostBudgetV1 {
    limit_bytes: usize,
    charged_peak_bytes: usize,
}

impl ResidentBabHostBudgetV1 {
    pub(super) fn begin(
        cap: ResidentBabAdapterHostCapV1,
        nominal_extra_bytes: usize,
    ) -> Result<Self, ResidentBabComposeErrorV1> {
        let charged_peak_bytes = cap
            .resident_bytes_before
            .checked_add(nominal_extra_bytes)
            .ok_or_else(|| invalid("retained-BaB adapter-host total overflows usize"))?;
        if cap.limit_bytes == 0 || charged_peak_bytes > cap.limit_bytes {
            return Err(ResidentBabComposeErrorV1::Capacity {
                required_bytes: charged_peak_bytes,
                limit_bytes: cap.limit_bytes,
            });
        }
        Ok(Self {
            limit_bytes: cap.limit_bytes,
            charged_peak_bytes,
        })
    }

    fn charge_excess(&mut self, bytes: usize) -> Result<(), ResidentBabComposeErrorV1> {
        let required_bytes = self
            .charged_peak_bytes
            .checked_add(bytes)
            .ok_or_else(|| invalid("retained-BaB observed-capacity charge overflows usize"))?;
        if required_bytes > self.limit_bytes {
            return Err(ResidentBabComposeErrorV1::Capacity {
                required_bytes,
                limit_bytes: self.limit_bytes,
            });
        }
        self.charged_peak_bytes = required_bytes;
        Ok(())
    }

    pub(super) fn charge_observed_excess(
        &mut self,
        bytes: usize,
    ) -> Result<(), ResidentBabComposeErrorV1> {
        self.charge_excess(bytes)
    }

    pub(super) fn reserve_vec<T>(
        &mut self,
        values: &mut Vec<T>,
        nominal_count: usize,
        label: &'static str,
    ) -> Result<(), ResidentBabComposeErrorV1> {
        values
            .try_reserve_exact(nominal_count)
            .map_err(|_| ResidentBabComposeErrorV1::AllocationRefused(label))?;
        let excess = values.capacity().saturating_sub(nominal_count);
        self.charge_excess(checked_elements::<T>(excess)?)
    }

    /// Reserve an output allocation whose nominal bytes were not part of the
    /// prospective scratch charge, then charge its full observed capacity.
    pub(super) fn reserve_vec_full<T>(
        &mut self,
        values: &mut Vec<T>,
        count: usize,
        label: &'static str,
    ) -> Result<(), ResidentBabComposeErrorV1> {
        self.charge_excess(checked_elements::<T>(count)?)?;
        values
            .try_reserve_exact(count)
            .map_err(|_| ResidentBabComposeErrorV1::AllocationRefused(label))?;
        let excess = values.capacity().saturating_sub(count);
        self.charge_excess(checked_elements::<T>(excess)?)
    }

    pub(super) fn reserve_string(
        &mut self,
        value: &mut String,
        nominal_bytes: usize,
        label: &'static str,
    ) -> Result<(), ResidentBabComposeErrorV1> {
        value
            .try_reserve_exact(nominal_bytes)
            .map_err(|_| ResidentBabComposeErrorV1::AllocationRefused(label))?;
        self.charge_excess(value.capacity().saturating_sub(nominal_bytes))
    }

    /// Reserve one retained string not booked in the prospective scratch
    /// charge, then charge the full observed byte capacity.
    pub(super) fn reserve_string_full(
        &mut self,
        value: &mut String,
        bytes: usize,
        label: &'static str,
    ) -> Result<(), ResidentBabComposeErrorV1> {
        self.charge_excess(bytes)?;
        value
            .try_reserve_exact(bytes)
            .map_err(|_| ResidentBabComposeErrorV1::AllocationRefused(label))?;
        self.charge_excess(value.capacity().saturating_sub(bytes))
    }

    pub(super) fn charge_hash_capacity<K, V>(
        &mut self,
        nominal_count: usize,
        observed_capacity: usize,
    ) -> Result<(), ResidentBabComposeErrorV1> {
        let excess = observed_capacity.saturating_sub(nominal_count);
        self.charge_excess(checked_hash_entries::<K, V>(excess)?)
    }

    pub(super) fn peak_bytes(&self) -> usize {
        self.charged_peak_bytes
    }
}

pub(super) fn poll_scaled(
    check: &mut dyn FnMut(&'static str) -> ny_core::Result<()>,
    label: &'static str,
    index: usize,
) -> Result<(), ResidentBabComposeErrorV1> {
    if index.is_multiple_of(RESIDENT_BAB_COMPOSE_POLL_STRIDE) {
        check(label)?;
    }
    Ok(())
}
