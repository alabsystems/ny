// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Process-wide accounting for GPU buffer bytes (#gpu-pool-highwater).
//!
//! # Why this exists
//!
//! Before this module nothing in the tree summed GPU allocations.
//! `gpu_memory_budget_bytes()` compares a per-call *estimate* against
//! `min(hw.memsize / 2, 8 GiB)` to decide whether the GPU CROWN path is
//! ATTEMPTED; it is not a running total, it never observes what was actually
//! allocated, and it does not see [`BufferPool`](crate::wgpu_device) retention
//! at all.
//!
//! # What this module is NOT evidence for
//!
//! This ledger was written to prove that the 2026-07-30 host OOM was GPU buffer
//! retention. **It disproved that**, and the correction belongs here rather than
//! only in the doc, because the next person to read this file will otherwise
//! re-derive the same wrong theory from the same `vmmap` output.
//!
//! `vmmap` on a run holding 10.9 GiB attributes almost all of it to
//! `IOAccelerator` against a ~1 MB `MALLOC_SMALL` row. That reads like GPU
//! memory and is not. `ny` installs **mimalloc** as its global allocator
//! (`ny-cli/src/main.rs`), which bypasses the system malloc zones and reserves
//! large aligned arenas by `mmap`; `vmmap` labels those 128 MiB slabs
//! `IOAccelerator`. The ~1 MB malloc-zone total for a process doing GB-scale
//! f64 linear algebra is the tell that the label is wrong.
//!
//! Measured, one run at a time, on `yolo_2023/TinyYOLO_prop_000001_eps_1_255`:
//! `NY_GPU_MEMORY_BUDGET_MB` cut 16x moved peak RSS 5%; running the preset with
//! `device: cpu` moved it to 14 856 MB -- *higher* than the wgpu run's
//! 11 364 MB, with `IOAccelerator` unchanged at 8.7 GiB. And under
//! `NY_GPU_MEM_TRACE=1` this ledger fired **zero** times during such a run,
//! i.e. under 256 MiB flowed through both choke points it instruments.
//!
//! So the footprint is a genuine CPU-side working set. See
//! `docs/HOST_OOM_ROOT_CAUSE_2026-07-30.md` for the full table. What remains
//! true, and why this module still earns its place: nothing summed GPU bytes
//! before it, so there was no way to answer the question either way.
//!
//! # What it does and does not claim
//!
//! This ledger counts bytes this crate asks wgpu to allocate. It is a
//! *policy-side* figure, deliberately not a claim about driver-side residency:
//! Metal suballocates from 128 MiB slabs and may keep a freed buffer's slab
//! mapped, so RSS lags the ledger downward. Use it to compare configurations
//! and to find which label dominates, not as a substitute for `vmmap`.
//!
//! Nothing here supplies a bound or certificate; it records sizes. When armed,
//! label allocation and logging can still perturb timing and memory on a
//! deadline-sensitive path.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::sync::OnceLock;

/// Live bytes this crate believes are allocated on the device.
static LIVE_BYTES: AtomicU64 = AtomicU64::new(0);

/// High-water mark of [`LIVE_BYTES`] over the process lifetime.
static PEAK_BYTES: AtomicU64 = AtomicU64::new(0);

/// Per-label high-water, populated only under `NY_GPU_MEM_TRACE=1`.
///
/// Attribution is the expensive part (a `String` key per allocation), and the
/// whole point of this module is to not be the thing that costs memory, so it
/// stays off unless asked for.
static LABEL_PEAK: OnceLock<Mutex<HashMap<String, u64>>> = OnceLock::new();

/// Whether per-label attribution is enabled (`NY_GPU_MEM_TRACE=1`, exact
/// `"1"`).
///
/// [`record_alloc`] calls this per GPU buffer allocation — hot — so the RAW
/// env string is latched once through the ny-levers chokepoint's raw view and
/// the decision is derived per call (lever-debt batch B1 preparation). This
/// remains process-wide; Phase 2 must replace it with an injected per-run
/// `LeverSet`.
pub fn trace_enabled() -> bool {
    static RAW: OnceLock<Option<String>> = OnceLock::new();
    RAW.get_or_init(|| ny_levers::read_raw(&ny_levers::decls::telemetry::GPU_MEM_TRACE))
        .as_deref()
        == Some("1")
}

/// Record `bytes` newly allocated on the device under `label`.
pub fn record_alloc(label: &str, bytes: u64) {
    let live = LIVE_BYTES.fetch_add(bytes, Ordering::Relaxed) + bytes;
    let prev_peak = PEAK_BYTES.fetch_max(live, Ordering::Relaxed);
    if !trace_enabled() {
        return;
    }
    if let Ok(mut map) = LABEL_PEAK.get_or_init(|| Mutex::new(HashMap::new())).lock() {
        let entry = map.entry(label.to_string()).or_insert(0);
        *entry = (*entry).max(bytes);
    }
    // Report only on a materially higher water mark. Allocation is a hot path
    // and a line per buffer is how the crashed session produced 1.2 GiB of
    // trace; a line per 256 MiB of NEW peak is a readable growth curve.
    const REPORT_STEP: u64 = 256 * 1024 * 1024;
    if live / REPORT_STEP > prev_peak / REPORT_STEP {
        tracing::info!(
            "#gpu-mem-ledger peak crossed {:.1} MiB (this alloc: {label} {:.1} MiB)",
            live as f64 / (1024.0 * 1024.0),
            bytes as f64 / (1024.0 * 1024.0)
        );
    }
}

/// Record `bytes` released on the device.
///
/// Saturating: a double-free or an unpaired release must not wrap the counter
/// to `u64::MAX` and make every subsequent reading nonsense.
// trust-1.99 deprecates `fetch_update` (renamed `try_update`); the public
// 1.95 pin lacks `try_update` — keep the spelling both toolchains accept.
#[allow(deprecated)]
pub fn record_free(bytes: u64) {
    let _ = LIVE_BYTES.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |live| {
        Some(live.saturating_sub(bytes))
    });
}

/// Bytes this crate believes are live on the device right now.
pub fn live_bytes() -> u64 {
    LIVE_BYTES.load(Ordering::Relaxed)
}

/// High-water mark of [`live_bytes`] over the process lifetime.
pub fn peak_bytes() -> u64 {
    PEAK_BYTES.load(Ordering::Relaxed)
}

/// Render the ledger for a log line, including per-label attribution when
/// `NY_GPU_MEM_TRACE=1` armed it.
pub fn summary() -> String {
    let mib = |b: u64| b as f64 / (1024.0 * 1024.0);
    let mut out = format!(
        "#gpu-mem-ledger live={:.1} MiB peak={:.1} MiB",
        mib(live_bytes()),
        mib(peak_bytes())
    );
    if trace_enabled() {
        if let Some(map) = LABEL_PEAK.get() {
            if let Ok(map) = map.lock() {
                let mut rows: Vec<_> = map.iter().map(|(k, v)| (*v, k.clone())).collect();
                rows.sort_unstable_by_key(|row| std::cmp::Reverse(row.0));
                out.push_str(" | largest-single-allocation by label:");
                for (bytes, label) in rows.iter().take(12) {
                    out.push_str(&format!(" {label}={:.1}MiB", mib(*bytes)));
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The counters are process-wide, so this test owns the sequencing rather
    /// than asserting absolute values other tests could perturb.
    #[test]
    fn alloc_and_free_move_live_bytes_and_peak_is_monotonic() {
        let before_live = live_bytes();
        let before_peak = peak_bytes();

        record_alloc("test_slot", 4096);
        assert_eq!(live_bytes(), before_live + 4096);
        assert!(peak_bytes() >= before_peak);

        record_free(4096);
        assert_eq!(live_bytes(), before_live);
        assert!(
            peak_bytes() >= before_peak,
            "peak must never decrease when memory is released"
        );
    }

    /// An unpaired or double free must clamp at zero rather than wrap, or every
    /// later reading becomes garbage.
    #[test]
    fn free_saturates_instead_of_wrapping() {
        record_free(u64::MAX);
        assert_eq!(
            live_bytes(),
            0,
            "an over-large free must saturate to zero, not wrap"
        );
    }
}
