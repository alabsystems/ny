// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Memory-aware microbatch sizing shared by graph BaB routes.
//!
//! The controller is constructed only when `auto_enlarge_batch_size` is true
//! and `NY_ADAPTIVE_MICROBATCH_CONTROLLER=1` is set exactly. Callers retain
//! their historical queue-pick and execution path unless both opt-ins hold.
//! When enabled, a queue batch remains owned by the caller while
//! [`OrderedBatchCursor`] advances through independently-sized device
//! microbatches. A refused attempt does not advance the cursor, which is the
//! no-loss/no-reordering retry contract.

use std::collections::BTreeMap;
use std::ops::Range;
use std::time::Duration;

use ny_core::NyError;

use crate::batched_domain::{CachedLinearBounds, DomainList, DomainMetadata, PickedDomains};
use crate::beta_crown::domain::GraphBabDomain;

use crate::beta_crown::config::AUTO_ENLARGE_BATCH_CAP;

pub(crate) const ADAPTIVE_MICROBATCH_GATE_ENV: &str = "NY_ADAPTIVE_MICROBATCH_CONTROLLER";

const MIB: usize = 1024 * 1024;
const DEFAULT_DEVICE_BUDGET_MIB: usize = 8192;
const RESERVE_NUMERATOR: usize = 1;
const RESERVE_DENOMINATOR: usize = 5;
const LONG_PASS_ABSOLUTE: Duration = Duration::from_secs(2);
const LONG_PASS_STREAK_TO_SHRINK: u8 = 2;
const REFUSAL_GROWTH_COOLDOWN: u8 = 2;

/// Route using the shared adaptive controller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AdaptiveBatchRoute {
    GraphReluSplit,
    DomainListInputSplit,
}

impl AdaptiveBatchRoute {
    fn label(self) -> &'static str {
        match self {
            Self::GraphReluSplit => "graph_relu_split",
            Self::DomainListInputSplit => "domain_list_input_split",
        }
    }
}

/// Stable reason code for a retryable batch refusal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum MicrobatchRefusalReason {
    /// A device allocation or device-memory budget gate refused the batch.
    DeviceAllocation,
    /// A host allocation or dense-materialization budget gate refused the batch.
    HostAllocation,
    /// The backend refused an otherwise valid dispatch.
    DeviceDispatch,
}

impl MicrobatchRefusalReason {
    pub(crate) fn code(self) -> &'static str {
        match self {
            Self::DeviceAllocation => "device_allocation",
            Self::HostAllocation => "host_allocation",
            Self::DeviceDispatch => "device_dispatch",
        }
    }

    /// Classify only errors for which retrying the same ordered domains in a
    /// smaller microbatch is safe and meaningful.
    pub(crate) fn from_error(error: &NyError) -> Option<Self> {
        match error {
            NyError::GpuMemoryExceeded { .. } => Some(Self::DeviceAllocation),
            NyError::CpuMemoryExceeded { .. } => Some(Self::HostAllocation),
            // WGPU error scopes currently surface runtime OOM/internal refusal
            // as structured InternalError messages.  Keep the matching narrow
            // to those locally-owned prefixes; validation errors are bugs and
            // are deliberately not retried.
            NyError::InternalError(detail) if detail.starts_with("wgpu out-of-memory ") => {
                Some(Self::DeviceAllocation)
            }
            NyError::InternalError(detail)
                if detail.starts_with("wgpu internal error ")
                    || detail.starts_with("uncaptured wgpu error ") =>
            {
                Some(Self::DeviceDispatch)
            }
            NyError::LayerError { source, .. } => Self::from_error(source),
            _ => None,
        }
    }
}

/// Result of [`AdaptiveMicrobatchController::on_refusal`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RefusalAction {
    /// Retry the uncommitted range at the smaller size.
    Retry { previous: usize, next: usize },
    /// A one-domain attempt was refused, so no safe smaller retry exists.
    Exhausted,
}

/// Backend and host memory envelope used by the controller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MicrobatchMemoryBudget {
    pub(crate) backend_bytes: usize,
    pub(crate) host_bytes: usize,
    pub(crate) reserve_bytes: usize,
}

impl MicrobatchMemoryBudget {
    #[cfg(test)]
    pub(crate) fn fixed(total_bytes: usize, reserve_bytes: usize) -> Self {
        Self {
            backend_bytes: total_bytes,
            host_bytes: total_bytes,
            reserve_bytes: reserve_bytes.min(total_bytes.saturating_sub(1)),
        }
    }

    pub(crate) fn runtime(has_device_engine: bool) -> Self {
        let host_bytes = runtime_host_budget_bytes();
        // GemmEngine does not expose allocator-active or live-free bytes.
        // For device routes this is NY's existing WGPU/Metal configured/system
        // budget fallback; observed tensor bytes below provide the measured
        // component.  Do not interpret it as a CUDA/VRAM availability query.
        let backend_bytes = if has_device_engine {
            runtime_device_budget_bytes()
        } else {
            host_bytes
        };
        let limiting_bytes = backend_bytes.min(host_bytes).max(1);
        Self {
            backend_bytes,
            host_bytes,
            reserve_bytes: limiting_bytes
                .saturating_mul(RESERVE_NUMERATOR)
                .checked_div(RESERVE_DENOMINATOR)
                .unwrap_or(0),
        }
    }

    fn limiting_bytes(self) -> usize {
        self.backend_bytes.min(self.host_bytes).max(1)
    }

    fn usable_bytes(self) -> usize {
        self.limiting_bytes()
            .saturating_sub(self.reserve_bytes)
            .max(1)
    }
}

/// Deterministic per-run telemetry.  `BTreeMap` ordering makes logs and tests
/// stable across randomized `HashMap` seeds.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct AdaptiveMicrobatchTelemetry {
    pub(crate) batch_histogram: BTreeMap<usize, u64>,
    pub(crate) grow_count: u64,
    pub(crate) backoff_count: u64,
    pub(crate) shrink_count: u64,
    pub(crate) refusal_count: u64,
    pub(crate) refusal_reasons: BTreeMap<&'static str, u64>,
}

/// Adaptive device-microbatch controller.
pub(crate) struct AdaptiveMicrobatchController {
    route: AdaptiveBatchRoute,
    current: usize,
    bytes_per_domain: usize,
    budget: MicrobatchMemoryBudget,
    consecutive_long_passes: u8,
    growth_cooldown: u8,
    telemetry: AdaptiveMicrobatchTelemetry,
}

impl AdaptiveMicrobatchController {
    pub(crate) fn new(
        route: AdaptiveBatchRoute,
        configured_batch_size: usize,
        bytes_per_domain: usize,
        budget: MicrobatchMemoryBudget,
    ) -> Self {
        let bytes_per_domain = bytes_per_domain.max(1);
        let configured = configured_batch_size.clamp(1, AUTO_ENLARGE_BATCH_CAP);
        let memory_cap =
            (budget.usable_bytes() / bytes_per_domain).clamp(1, AUTO_ENLARGE_BATCH_CAP);
        Self {
            route,
            current: configured.min(memory_cap),
            bytes_per_domain,
            budget,
            consecutive_long_passes: 0,
            growth_cooldown: 0,
            telemetry: AdaptiveMicrobatchTelemetry::default(),
        }
    }

    pub(crate) fn current(&self) -> usize {
        self.current
    }

    /// Queue-pick size for the next outer iteration.  The returned value is a
    /// snapshot: refusal can reduce the device microbatch while the already
    /// picked queue batch and its order remain unchanged.
    pub(crate) fn queue_pick_size(&self, queue_len: usize) -> usize {
        self.current.min(queue_len).max(1)
    }

    #[cfg(test)]
    pub(crate) fn telemetry(&self) -> &AdaptiveMicrobatchTelemetry {
        &self.telemetry
    }

    fn memory_cap(&self) -> usize {
        (self.budget.usable_bytes() / self.bytes_per_domain).clamp(1, AUTO_ENLARGE_BATCH_CAP)
    }

    /// Record a successful, committed microbatch and update the next size.
    ///
    /// `remaining` is sampled after the pass.  Growth is suppressed when a
    /// larger uninterruptible pass would consume the deadline guard (three
    /// observed pass durations, with a one-second floor).
    pub(crate) fn on_success(
        &mut self,
        requested: usize,
        actual: usize,
        observed_bytes_per_domain: usize,
        elapsed: Duration,
        remaining: Option<Duration>,
    ) {
        if actual > 0 {
            *self.telemetry.batch_histogram.entry(actual).or_default() += 1;
        }

        self.bytes_per_domain = self.bytes_per_domain.max(observed_bytes_per_domain.max(1));
        let memory_cap = self.memory_cap();
        if self.current > memory_cap {
            self.current = memory_cap;
            self.telemetry.shrink_count += 1;
        }

        let deadline_guard = elapsed.saturating_mul(3).max(Duration::from_secs(1));
        if remaining.is_some_and(|left| left <= deadline_guard) && self.current > 1 {
            self.current = (self.current / 2).max(1);
            self.consecutive_long_passes = 0;
            self.growth_cooldown = REFUSAL_GROWTH_COOLDOWN;
            self.telemetry.shrink_count += 1;
            return;
        }

        let deadline_long =
            remaining.is_some_and(|left| !left.is_zero() && elapsed.saturating_mul(4) >= left);
        let is_long = elapsed >= LONG_PASS_ABSOLUTE || deadline_long;
        if is_long {
            self.consecutive_long_passes = self.consecutive_long_passes.saturating_add(1);
        } else {
            self.consecutive_long_passes = 0;
        }

        if self.consecutive_long_passes >= LONG_PASS_STREAK_TO_SHRINK && self.current > 1 {
            self.current = (self.current / 2).max(1);
            self.consecutive_long_passes = 0;
            self.growth_cooldown = REFUSAL_GROWTH_COOLDOWN;
            self.telemetry.shrink_count += 1;
            return;
        }

        if actual < requested {
            let next = actual.max(1).min(self.current);
            if next < self.current {
                self.current = next;
                self.telemetry.shrink_count += 1;
            }
            return;
        }

        if self.growth_cooldown > 0 {
            self.growth_cooldown -= 1;
            return;
        }

        let full = actual == requested && requested == self.current;
        if !full || self.current >= AUTO_ENLARGE_BATCH_CAP {
            return;
        }

        if remaining.is_some_and(|left| left <= deadline_guard) {
            return;
        }

        let candidate = self
            .current
            .saturating_mul(2)
            .min(AUTO_ENLARGE_BATCH_CAP)
            .min(memory_cap);
        if candidate > self.current
            && candidate.saturating_mul(self.bytes_per_domain) <= self.budget.usable_bytes()
        {
            self.current = candidate;
            self.telemetry.grow_count += 1;
        }
    }

    /// Record a refusal and reduce the next attempt without committing the
    /// caller's ordered range.
    pub(crate) fn on_refusal(&mut self, reason: MicrobatchRefusalReason) -> RefusalAction {
        self.telemetry.refusal_count += 1;
        *self
            .telemetry
            .refusal_reasons
            .entry(reason.code())
            .or_default() += 1;

        self.growth_cooldown = REFUSAL_GROWTH_COOLDOWN;
        if self.current <= 1 {
            return RefusalAction::Exhausted;
        }
        let previous = self.current;
        self.current = (self.current / 2).max(1);
        self.telemetry.backoff_count += 1;
        RefusalAction::Retry {
            previous,
            next: self.current,
        }
    }
}

/// The new controller is independently default-dark even for presets that
/// already enable the legacy `auto_enlarge_batch_size` policy. Only the exact
/// raw spelling `1` arms it; unset, `0`, non-Unicode, and malformed values all
/// preserve the prior route.
pub(crate) fn adaptive_microbatch_controller_enabled(auto_enlarge_batch_size: bool) -> bool {
    auto_enlarge_batch_size
        && parse_adaptive_microbatch_gate(
            std::env::var(ADAPTIVE_MICROBATCH_GATE_ENV).ok().as_deref(),
        )
}

fn parse_adaptive_microbatch_gate(raw: Option<&str>) -> bool {
    raw == Some("1")
}

impl Drop for AdaptiveMicrobatchController {
    fn drop(&mut self) {
        tracing::info!(
            target: "ny_propagate::adaptive_microbatch",
            route = self.route.label(),
            final_microbatch = self.current,
            bytes_per_domain = self.bytes_per_domain,
            backend_budget_bytes = self.budget.backend_bytes,
            host_budget_bytes = self.budget.host_bytes,
            reserve_bytes = self.budget.reserve_bytes,
            grow_count = self.telemetry.grow_count,
            backoff_count = self.telemetry.backoff_count,
            shrink_count = self.telemetry.shrink_count,
            refusal_count = self.telemetry.refusal_count,
            batch_histogram = ?self.telemetry.batch_histogram,
            refusal_reasons = ?self.telemetry.refusal_reasons,
            "adaptive microbatch telemetry"
        );
    }
}

/// Cursor over one queue batch.  The cursor advances only after a successful
/// commit, so calling `next_range()` again after refusal returns the same
/// ordered prefix at the controller's new, smaller size.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OrderedBatchCursor {
    len: usize,
    committed: usize,
}

impl OrderedBatchCursor {
    pub(crate) fn new(len: usize) -> Self {
        Self { len, committed: 0 }
    }

    pub(crate) fn is_done(&self) -> bool {
        self.committed == self.len
    }

    pub(crate) fn next_range(&self, microbatch_size: usize) -> Range<usize> {
        let end = self
            .committed
            .saturating_add(microbatch_size.max(1))
            .min(self.len);
        self.committed..end
    }

    pub(crate) fn commit(&mut self, range: Range<usize>) {
        debug_assert_eq!(range.start, self.committed);
        debug_assert!(range.end <= self.len);
        self.committed = range.end;
    }
}

/// Estimate the transfer/working metadata carried by one graph domain.
pub(crate) fn estimate_graph_domain_bytes(domain: &GraphBabDomain) -> usize {
    let tensor_bytes = domain.node_bounds.iter().fold(
        domain
            .input_bounds
            .len()
            .saturating_mul(2 * size_of::<f32>()),
        |acc, (name, bounds)| {
            acc.saturating_add(name.len())
                .saturating_add(bounds.len().saturating_mul(2 * size_of::<f32>()))
        },
    );
    let history_bytes = domain
        .history
        .constraints
        .iter()
        .map(|constraint| size_of_val(constraint).saturating_add(constraint.node_name.len()))
        .sum::<usize>()
        .saturating_add(
            domain
                .history
                .genbab_constraints
                .iter()
                .map(|constraint| {
                    size_of_val(constraint).saturating_add(constraint.node_name.len())
                })
                .sum::<usize>(),
        );
    let beta_bytes = domain
        .beta_state
        .entries
        .iter()
        .map(|entry| size_of_val(entry).saturating_add(entry.node_name.len()))
        .sum::<usize>();
    let alpha_bytes = domain
        .alpha_state
        .neurons
        .iter()
        .chain(domain.alpha_state.upper_neurons.iter())
        .map(|(name, neurons)| {
            name.len().saturating_add(neurons.len().saturating_mul(
                size_of::<usize>() + size_of::<crate::beta_crown::state::AlphaNeuronState>(),
            ))
        })
        .sum::<usize>();
    let cached_bytes = domain
        .cached_la
        .as_deref()
        .map(estimate_cached_linear_bounds_bytes)
        .unwrap_or(0);

    size_of::<GraphBabDomain>()
        .saturating_add(tensor_bytes)
        .saturating_add(history_bytes)
        .saturating_add(beta_bytes)
        .saturating_add(alpha_bytes)
        .saturating_add(cached_bytes)
        .saturating_add(
            domain
                .delta_pre_nodes
                .iter()
                .map(String::len)
                .sum::<usize>(),
        )
        .max(1)
}

/// Estimate the average row footprint in the DomainList before pick-out.
pub(crate) fn estimate_domain_list_bytes_per_domain(domain_list: &DomainList) -> usize {
    let tensor_elements = domain_list
        .config
        .layer_names
        .iter()
        .filter_map(|name| domain_list.config.layer_shapes.get(name))
        .map(|shape| shape.iter().copied().fold(1usize, usize::saturating_mul))
        .sum::<usize>()
        .saturating_add(
            domain_list
                .config
                .input_shape
                .iter()
                .copied()
                .fold(1usize, usize::saturating_mul),
        )
        .saturating_add(1);
    let tensor_bytes = tensor_elements.saturating_mul(2 * size_of::<f32>());
    let metadata_bytes = if domain_list.metadata.is_empty() {
        size_of::<DomainMetadata>()
    } else {
        domain_list
            .metadata
            .iter()
            .map(estimate_domain_metadata_bytes)
            .sum::<usize>()
            .div_ceil(domain_list.metadata.len())
    };
    tensor_bytes.saturating_add(metadata_bytes).max(1)
}

/// Estimate the actual tensor and metadata bytes in an extracted batch.
pub(crate) fn estimate_picked_bytes_per_domain(picked: &PickedDomains) -> usize {
    if picked.batch_size == 0 {
        return 1;
    }
    let tensor_elements = picked
        .layer_lowers
        .values()
        .chain(picked.layer_uppers.values())
        .map(ndarray::ArrayBase::len)
        .sum::<usize>()
        .saturating_add(picked.input_lowers.len())
        .saturating_add(picked.input_uppers.len())
        .saturating_add(picked.global_lbs.len())
        .saturating_add(picked.global_ubs.len());
    let metadata_bytes = picked
        .metadata
        .iter()
        .map(estimate_domain_metadata_bytes)
        .sum::<usize>();
    tensor_elements
        .saturating_mul(size_of::<f32>())
        .saturating_add(metadata_bytes)
        .div_ceil(picked.batch_size)
        .max(1)
}

fn estimate_domain_metadata_bytes(metadata: &DomainMetadata) -> usize {
    let constraints = metadata
        .constraints
        .iter()
        .map(|(name, _, _, _)| size_of_val(&(name, 0usize, false, None::<f32>)) + name.len())
        .sum::<usize>();
    let cached = metadata
        .cached_la
        .as_deref()
        .map(estimate_cached_linear_bounds_bytes)
        .unwrap_or(0);
    let override_bytes = metadata
        .node_bounds_override
        .as_deref()
        .map(|bounds| {
            bounds
                .iter()
                .map(|(name, tensor)| {
                    name.len()
                        .saturating_add(tensor.len().saturating_mul(2 * size_of::<f32>()))
                })
                .sum::<usize>()
        })
        .unwrap_or(0);
    let alpha_bytes = metadata
        .alpha_state_byte_census()
        .map(|census| census.estimated_total_bytes)
        .unwrap_or(0);
    size_of::<DomainMetadata>()
        .saturating_add(constraints)
        .saturating_add(cached)
        .saturating_add(override_bytes)
        .saturating_add(alpha_bytes)
}

fn estimate_cached_linear_bounds_bytes(bounds: &CachedLinearBounds) -> usize {
    let arrays = bounds
        .lower_a
        .iter()
        .chain(bounds.upper_a.iter())
        .map(|(name, array)| name.len().saturating_add(array.len() * size_of::<f32>()))
        .sum::<usize>()
        .saturating_add(
            bounds
                .lower_b
                .iter()
                .chain(bounds.upper_b.iter())
                .map(|(name, array)| name.len().saturating_add(array.len() * size_of::<f32>()))
                .sum::<usize>(),
        );
    size_of::<CachedLinearBounds>().saturating_add(arrays)
}

fn runtime_host_budget_bytes() -> usize {
    let configured = crate::network::crown_memory::cpu_crown_dense_budget_bytes();
    crate::network::crown_memory::process_memory_headroom_bytes()
        .and_then(|bytes| usize::try_from(bytes).ok())
        .map_or(configured, |headroom| configured.min(headroom))
        .max(1)
}

fn runtime_device_budget_bytes() -> usize {
    if let Ok(mebibytes) = std::env::var("NY_GPU_MEMORY_BUDGET_MB") {
        if let Ok(value) = mebibytes.parse::<usize>() {
            return value.saturating_mul(MIB).max(1);
        }
    }
    let system = system_memory_bytes();
    if system == 0 {
        DEFAULT_DEVICE_BUDGET_MIB * MIB
    } else {
        (system / 2).clamp(1, DEFAULT_DEVICE_BUDGET_MIB * MIB)
    }
}

fn system_memory_bytes() -> usize {
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()
            .and_then(|output| {
                String::from_utf8_lossy(&output.stdout)
                    .trim()
                    .parse::<usize>()
                    .ok()
            })
            .unwrap_or(0)
    }
    #[cfg(not(target_os = "macos"))]
    {
        std::fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|contents| {
                contents
                    .lines()
                    .find_map(|line| line.strip_prefix("MemTotal:"))
                    .and_then(|rest| rest.split_whitespace().next())
                    .and_then(|kib| kib.parse::<usize>().ok())
                    .and_then(|kib| kib.checked_mul(1024))
            })
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;
    use ny_tensor::{BoundedTensor, TreeTraversal};
    use proptest::prelude::*;
    use std::collections::HashMap;

    fn controller(initial: usize, per_domain: usize, total: usize) -> AdaptiveMicrobatchController {
        AdaptiveMicrobatchController::new(
            AdaptiveBatchRoute::GraphReluSplit,
            initial,
            per_domain,
            MicrobatchMemoryBudget::fixed(total, total / 5),
        )
    }

    #[test]
    fn grows_only_after_full_success_with_reserve() {
        let mut ctl = controller(4, 100, 2_000);
        ctl.on_success(4, 3, 100, Duration::from_millis(10), None);
        assert_eq!(ctl.current(), 3, "underfill shrinks instead of growing");

        let mut ctl = controller(4, 100, 2_000);
        ctl.on_success(4, 4, 100, Duration::from_millis(10), None);
        assert_eq!(ctl.current(), 8);
        assert_eq!(ctl.telemetry().grow_count, 1);

        let mut no_headroom = controller(4, 300, 2_000);
        no_headroom.on_success(4, 4, 300, Duration::from_millis(10), None);
        assert_eq!(
            no_headroom.current(),
            5,
            "growth may use the last safe size below the reserve, but not double past it"
        );
        assert!(no_headroom.current() * 300 <= 1_600);
        assert!((no_headroom.current() + 1) * 300 > 1_600);
    }

    #[test]
    fn refusal_retries_same_cursor_range_then_preserves_order() {
        let source: Vec<usize> = (0..8).collect();
        let mut ctl = controller(8, 1, 1_000);
        let mut cursor = OrderedBatchCursor::new(source.len());

        let refused = cursor.next_range(ctl.current());
        assert_eq!(&source[refused.clone()], &[0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(
            ctl.on_refusal(MicrobatchRefusalReason::DeviceAllocation),
            RefusalAction::Retry {
                previous: 8,
                next: 4
            }
        );
        let retry = cursor.next_range(ctl.current());
        assert_eq!(
            retry.start, refused.start,
            "refusal must not advance cursor"
        );
        assert_eq!(&source[retry.clone()], &[0, 1, 2, 3]);
        cursor.commit(retry);
        let second = cursor.next_range(ctl.current());
        assert_eq!(&source[second.clone()], &[4, 5, 6, 7]);
        cursor.commit(second);
        assert!(cursor.is_done());
        assert_eq!(ctl.telemetry().refusal_count, 1);
        assert_eq!(ctl.telemetry().backoff_count, 1);
        assert_eq!(ctl.telemetry().refusal_reasons["device_allocation"], 1);
    }

    #[test]
    fn observed_memory_pressure_shrinks_and_refusal_cooldown_delays_regrowth() {
        let mut pressure = controller(8, 100, 2_000);
        pressure.on_success(8, 8, 300, Duration::from_millis(10), None);
        assert_eq!(
            pressure.current(),
            5,
            "actual tensor bytes must lower the next memory-capped batch"
        );
        assert_eq!(pressure.telemetry().shrink_count, 1);

        let mut retry = controller(4, 10, 2_000);
        assert!(matches!(
            retry.on_refusal(MicrobatchRefusalReason::DeviceDispatch),
            RefusalAction::Retry {
                previous: 4,
                next: 2
            }
        ));
        retry.on_success(2, 2, 10, Duration::from_millis(10), None);
        retry.on_success(2, 2, 10, Duration::from_millis(10), None);
        assert_eq!(retry.current(), 2, "cooldown requires two stable passes");
        retry.on_success(2, 2, 10, Duration::from_millis(10), None);
        assert_eq!(retry.current(), 4);

        let mut host_fallback = controller(1, 10, 2_000);
        assert_eq!(
            host_fallback.on_refusal(MicrobatchRefusalReason::DeviceAllocation),
            RefusalAction::Exhausted
        );
        host_fallback.on_success(1, 1, 10, Duration::from_millis(10), None);
        assert_eq!(
            host_fallback.current(),
            1,
            "a successful host fallback must not immediately regrow the refused device batch"
        );
    }

    #[test]
    fn repeated_long_passes_shrink_and_near_deadline_blocks_growth() {
        let mut ctl = controller(8, 1, 1_000);
        ctl.on_success(
            8,
            8,
            1,
            Duration::from_secs(3),
            Some(Duration::from_secs(30)),
        );
        assert_eq!(ctl.current(), 16);
        ctl.on_success(
            16,
            16,
            1,
            Duration::from_secs(3),
            Some(Duration::from_secs(20)),
        );
        assert_eq!(ctl.current(), 8, "second long pass triggers shrink");

        let mut deadline = controller(8, 1, 1_000);
        deadline.on_success(
            8,
            8,
            1,
            Duration::from_millis(400),
            Some(Duration::from_millis(900)),
        );
        assert_eq!(
            deadline.current(),
            4,
            "near deadline must proactively cap the next uninterruptible pass"
        );
    }

    #[test]
    fn refusal_reason_classifier_is_narrow_and_reason_coded() {
        assert_eq!(
            MicrobatchRefusalReason::from_error(&NyError::GpuMemoryExceeded {
                required_bytes: 2,
                budget_bytes: 1,
            }),
            Some(MicrobatchRefusalReason::DeviceAllocation)
        );
        assert_eq!(
            MicrobatchRefusalReason::from_error(&NyError::CpuMemoryExceeded {
                required_bytes: 2,
                budget_bytes: 1,
                site: "test",
            }),
            Some(MicrobatchRefusalReason::HostAllocation)
        );
        assert_eq!(
            MicrobatchRefusalReason::from_error(&NyError::InvalidSpec("not retryable".into())),
            None
        );
    }

    #[test]
    fn independent_gate_is_default_dark_exact_and_requires_auto_enlarge() {
        for raw in [
            None,
            Some(""),
            Some("0"),
            Some("true"),
            Some("01"),
            Some(" 1"),
        ] {
            assert!(
                !parse_adaptive_microbatch_gate(raw),
                "unset/zero/malformed values must stay dark: {raw:?}"
            );
        }
        assert!(parse_adaptive_microbatch_gate(Some("1")));
        assert!(
            !adaptive_microbatch_controller_enabled(false),
            "the independent gate cannot bypass auto_enlarge_batch_size=false"
        );

        let legacy_auto_enlarge = true;
        let disabled =
            (legacy_auto_enlarge && parse_adaptive_microbatch_gate(Some("0"))).then(|| {
                AdaptiveMicrobatchController::new(
                    AdaptiveBatchRoute::DomainListInputSplit,
                    64,
                    1_024,
                    MicrobatchMemoryBudget::fixed(1_000_000, 100_000),
                )
            });
        assert!(
            disabled.is_none(),
            "gate-dark presets keep the legacy route"
        );

        let queue_len = 100;
        let configured = 64usize;
        let legacy_pick = configured.min(queue_len);
        assert_eq!(legacy_pick, 64);
    }

    #[test]
    fn byte_estimates_scale_with_actual_tensor_and_metadata_payloads() {
        let input = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();
        let small = GraphBabDomain::root(HashMap::new(), -1.0, 1.0, &input, false).unwrap();
        let mut node_bounds = HashMap::new();
        node_bounds.insert(
            "wide_node".to_string(),
            BoundedTensor::new(arr1(&[-1.0; 32]).into_dyn(), arr1(&[1.0; 32]).into_dyn()).unwrap(),
        );
        let wide = GraphBabDomain::root(node_bounds, -1.0, 1.0, &input, false).unwrap();
        assert!(
            estimate_graph_domain_bytes(&wide) >= estimate_graph_domain_bytes(&small) + 32 * 8,
            "graph estimate must include both f32 tensor endpoints"
        );

        let small_list = DomainList::new(crate::batched_domain::DomainListConfig {
            traversal: TreeTraversal::BreadthFirst,
            layer_names: vec!["pre".into()],
            layer_shapes: HashMap::from([("pre".into(), vec![2])]),
            input_shape: vec![4],
            initial_capacity: 1,
            max_queue_size: 0,
        })
        .unwrap();
        let wide_list = DomainList::new(crate::batched_domain::DomainListConfig {
            traversal: TreeTraversal::BreadthFirst,
            layer_names: vec!["pre".into()],
            layer_shapes: HashMap::from([("pre".into(), vec![34])]),
            input_shape: vec![4],
            initial_capacity: 1,
            max_queue_size: 0,
        })
        .unwrap();
        assert_eq!(
            estimate_domain_list_bytes_per_domain(&wide_list)
                - estimate_domain_list_bytes_per_domain(&small_list),
            32 * 2 * size_of::<f32>(),
            "DomainList estimate must be derived from configured row tensor shapes"
        );
    }

    #[test]
    fn both_production_routes_share_reason_coded_retry_semantics() {
        for route in [
            AdaptiveBatchRoute::GraphReluSplit,
            AdaptiveBatchRoute::DomainListInputSplit,
        ] {
            let mut ctl = AdaptiveMicrobatchController::new(
                route,
                8,
                10,
                MicrobatchMemoryBudget::fixed(10_000, 1_000),
            );
            let mut cursor = OrderedBatchCursor::new(8);
            let original = cursor.next_range(ctl.current());
            assert!(matches!(
                ctl.on_refusal(MicrobatchRefusalReason::DeviceDispatch),
                RefusalAction::Retry {
                    previous: 8,
                    next: 4
                }
            ));
            let retry = cursor.next_range(ctl.current());
            assert_eq!(retry.start, original.start);
            cursor.commit(retry);
            let tail = cursor.next_range(ctl.current());
            assert_eq!(tail, 4..8);
        }
    }

    proptest! {
        #[test]
        fn cursor_retry_property_has_no_loss_or_reordering(
            len in 1usize..256,
            initial in 1usize..128,
            refusals in 0usize..8,
        ) {
            let source: Vec<usize> = (0..len).collect();
            let mut ctl = controller(initial, 1, 1_000_000);
            let mut cursor = OrderedBatchCursor::new(len);
            let mut output = Vec::new();
            let mut remaining_refusals = refusals;

            while !cursor.is_done() {
                let range = cursor.next_range(ctl.current());
                if remaining_refusals > 0 && ctl.current() > 1 {
                    let before = range.start;
                    let action = ctl.on_refusal(MicrobatchRefusalReason::DeviceDispatch);
                    prop_assert_ne!(action, RefusalAction::Exhausted);
                    prop_assert_eq!(cursor.next_range(ctl.current()).start, before);
                    remaining_refusals -= 1;
                    continue;
                }
                output.extend_from_slice(&source[range.clone()]);
                cursor.commit(range);
            }
            prop_assert_eq!(output, source);
        }
    }
}
