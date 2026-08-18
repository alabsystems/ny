// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #span-profile — hierarchical wall-time accounting across rayon workers.
//!
//! Built because `sample(1)` was not good enough to answer "where do the
//! margin-row BaB worker's 262.5 seconds go?". A sampling profiler on an
//! optimised build gives INCLUSIVE counts against inlined, mangled frames, and
//! on this workload ~95% of the samples land in rayon plumbing
//! (`bridge_producer_consumer`, `wait_until_cold`) rather than in anything
//! nameable. That tells you the pool is busy; it does not tell you which phase
//! of the algorithm owns the time.
//!
//! This gives **self time** (time in a region excluding its instrumented
//! children) and **total time**, per named region, summed over every thread.
//!
//! ## Design
//!
//! - **Off by default and cheap when off.** `NY_SPAN_PROFILE=1` (exact `"1"`)
//!   is read once into a `OnceLock<bool>`; when off, [`span`] returns an inert
//!   guard whose `Drop` does nothing.
//! - **No cross-thread contention on the hot path.** Each thread owns its own
//!   accumulator behind its own mutex and registers a clone of the `Arc` in a
//!   global registry the first time it opens a span. A span exit locks only its
//!   OWN mutex, which is uncontended, so the cost is one atomic CAS plus a hash
//!   lookup. The global registry lock is taken once per thread, ever, and again
//!   at [`report`].
//! - **Nesting is per-thread.** A guard opened on a rayon worker is dropped on
//!   that same worker, so the stack discipline holds. A child's total time is
//!   subtracted from its parent's self time.
//! - **Names are `&'static str`** so the map key is pointer-cheap and no
//!   allocation happens on the hot path.
//!
//! ## Contract
//!
//! Instrument COARSE regions — a backward collect, a BaB node evaluation, a
//! candidate scoring pass. Do not put a span inside a per-neuron inner loop:
//! the mutex and `Instant::now()` would then dominate what they measure. The
//! unit tests pin the self/total arithmetic, not the overhead.
//!
//! This is diagnostics only. It never influences a bound, a verdict, or a
//! scheduling decision, and with the gate unset the process is byte-identical.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

/// Exact `"1"`, matching every other dark gate in this tree.
const ENV: &str = "NY_SPAN_PROFILE";

/// Per-region accumulator.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Acc {
    /// Nanoseconds inside this region, EXCLUDING instrumented children.
    pub self_ns: u128,
    /// Nanoseconds inside this region, including children.
    pub total_ns: u128,
    /// Number of times the region was entered.
    pub calls: u64,
}

type Table = HashMap<&'static str, Acc>;

/// Registry of every thread's accumulator. Locked once per thread on first
/// span, and again at [`report`] — never on the hot path.
static REGISTRY: Mutex<Vec<Arc<Mutex<Table>>>> = Mutex::new(Vec::new());

static ENABLED: OnceLock<bool> = OnceLock::new();

/// Whether span profiling is armed. Read once per process.
#[must_use]
pub fn enabled() -> bool {
    *ENABLED.get_or_init(|| std::env::var(ENV).is_ok_and(|v| v == "1"))
}

struct Frame {
    name: &'static str,
    start: Instant,
    /// Total time of instrumented children opened under this frame.
    child_ns: u128,
}

struct Local {
    stack: Vec<Frame>,
    table: Arc<Mutex<Table>>,
}

thread_local! {
    static LOCAL: std::cell::RefCell<Option<Local>> = const { std::cell::RefCell::new(None) };
}

fn with_local<R>(f: impl FnOnce(&mut Local) -> R) -> Option<R> {
    LOCAL
        .try_with(|cell| {
            let mut slot = cell.borrow_mut();
            if slot.is_none() {
                let table: Arc<Mutex<Table>> = Arc::new(Mutex::new(HashMap::new()));
                // Register once per thread. A poisoned registry means another
                // thread panicked mid-registration; profiling is diagnostics, so
                // degrade to "this thread is not reported" rather than propagate.
                if let Ok(mut reg) = REGISTRY.lock() {
                    reg.push(Arc::clone(&table));
                }
                *slot = Some(Local {
                    stack: Vec::new(),
                    table,
                });
            }
            f(slot.as_mut().expect("just initialised"))
        })
        .ok()
}

/// RAII span guard. Inert when profiling is off.
#[derive(Debug)]
pub struct SpanGuard {
    armed: bool,
}

impl SpanGuard {
    /// An explicitly inert guard, for call sites that want to hold the type
    /// unconditionally.
    #[must_use]
    pub const fn inert() -> Self {
        Self { armed: false }
    }
}

impl Drop for SpanGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        with_local(|local| {
            let Some(frame) = local.stack.pop() else {
                return;
            };
            let total_ns = frame.start.elapsed().as_nanos();
            // Children cannot legitimately exceed the parent, but clamp rather
            // than underflow if a clock is non-monotonic under virtualisation.
            let self_ns = total_ns.saturating_sub(frame.child_ns);
            if let Some(parent) = local.stack.last_mut() {
                parent.child_ns = parent.child_ns.saturating_add(total_ns);
            }
            if let Ok(mut table) = local.table.lock() {
                let acc = table.entry(frame.name).or_default();
                acc.self_ns = acc.self_ns.saturating_add(self_ns);
                acc.total_ns = acc.total_ns.saturating_add(total_ns);
                acc.calls = acc.calls.saturating_add(1);
            }
        });
    }
}

/// Open a span. Drop the returned guard to close it.
///
/// ```rust
/// let _s = ny_core::span_profile::span("margin_row::collect");
/// ```
#[must_use]
pub fn span(name: &'static str) -> SpanGuard {
    if !enabled() {
        return SpanGuard::inert();
    }
    let opened = with_local(|local| {
        local.stack.push(Frame {
            name,
            start: Instant::now(),
            child_ns: 0,
        });
    })
    .is_some();
    SpanGuard { armed: opened }
}

/// Merge every thread's table into one.
#[must_use]
pub fn merged() -> Table {
    let mut out: Table = HashMap::new();
    let Ok(reg) = REGISTRY.lock() else {
        return out;
    };
    for table in reg.iter() {
        let Ok(t) = table.lock() else { continue };
        for (name, acc) in t.iter() {
            let e = out.entry(name).or_default();
            e.self_ns = e.self_ns.saturating_add(acc.self_ns);
            e.total_ns = e.total_ns.saturating_add(acc.total_ns);
            e.calls = e.calls.saturating_add(acc.calls);
        }
    }
    out
}

/// A sorted flat profile, heaviest self time first.
///
/// Self time is summed over threads, so on a K-core machine it can exceed wall
/// clock by up to K — that is the point: it attributes CPU, and comparing it
/// against wall clock is how you spot a region that is parallel-inefficient.
#[must_use]
pub fn report() -> String {
    let table = merged();
    if table.is_empty() {
        return "[span-profile] no spans recorded\n".to_string();
    }
    let mut rows: Vec<(&'static str, Acc)> = table.into_iter().collect();
    rows.sort_by(|a, b| b.1.self_ns.cmp(&a.1.self_ns).then_with(|| a.0.cmp(b.0)));
    let grand: u128 = rows.iter().map(|(_, a)| a.self_ns).sum();
    let mut out = String::from(
        "[span-profile] region                                   self_s   total_s    calls   self%\n",
    );
    for (name, acc) in &rows {
        let pct = if grand > 0 {
            (acc.self_ns as f64) * 100.0 / (grand as f64)
        } else {
            0.0
        };
        out.push_str(&format!(
            "[span-profile] {:<48} {:>8.3} {:>9.3} {:>8} {:>6.2}\n",
            name,
            acc.self_ns as f64 / 1e9,
            acc.total_ns as f64 / 1e9,
            acc.calls,
            pct
        ));
    }
    out.push_str(&format!(
        "[span-profile] TOTAL self {:.3}s across {} regions (summed over threads)\n",
        grand as f64 / 1e9,
        rows.len()
    ));
    out
}

/// Drop every recorded sample. Test/bench support.
pub fn reset() {
    if let Ok(reg) = REGISTRY.lock() {
        for table in reg.iter() {
            if let Ok(mut t) = table.lock() {
                t.clear();
            }
        }
    }
    let _ = with_local(|local| local.stack.clear());
}

#[cfg(test)]
mod tests {
    use super::*;

    /// These tests share a PROCESS-GLOBAL registry and several call `reset()`,
    /// so under the parallel test harness they would clobber each other. Every
    /// test takes this lock, making them serial and deterministic. (They passed
    /// without it, which was luck, not design.)
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    /// Take the serialising lock, tolerating poisoning from an unrelated
    /// failing test so one failure does not cascade into six.
    fn serial() -> std::sync::MutexGuard<'static, ()> {
        TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The gate is read once into a `OnceLock`, so tests cannot flip it. They
    /// exercise the accounting directly instead, which is the part that can be
    /// wrong; `enabled()` is a one-line env read.
    fn open(name: &'static str) -> SpanGuard {
        with_local(|local| {
            local.stack.push(Frame {
                name,
                start: Instant::now(),
                child_ns: 0,
            });
        });
        SpanGuard { armed: true }
    }

    fn busy(ms: u64) {
        let t = Instant::now();
        while t.elapsed().as_millis() < u128::from(ms) {
            std::hint::spin_loop();
        }
    }

    #[test]
    fn inert_guard_records_nothing() {
        let _serial = serial();
        reset();
        {
            let _g = SpanGuard::inert();
            busy(5);
        }
        assert!(merged().is_empty(), "an inert guard must not record");
    }

    #[test]
    fn self_time_excludes_instrumented_children() {
        let _serial = serial();
        reset();
        {
            let _outer = open("outer");
            busy(10);
            {
                let _inner = open("inner");
                busy(30);
            }
        }
        let t = merged();
        let outer = t["outer"];
        let inner = t["inner"];
        // The child's total is subtracted from the parent's self time. That is
        // the whole point of the module: `sample` could not do this.
        assert!(
            outer.total_ns > inner.total_ns,
            "parent total {} must exceed child total {}",
            outer.total_ns,
            inner.total_ns
        );
        assert!(
            outer.self_ns < inner.self_ns,
            "parent self {} should be well under child self {} (10ms vs 30ms)",
            outer.self_ns,
            inner.self_ns
        );
        assert_eq!(outer.calls, 1);
        assert_eq!(inner.calls, 1);
        reset();
    }

    #[test]
    fn repeated_entries_accumulate_and_count() {
        let _serial = serial();
        reset();
        for _ in 0..4 {
            let _g = open("loop_region");
            busy(2);
        }
        let acc = merged()["loop_region"];
        assert_eq!(acc.calls, 4);
        assert!(acc.self_ns >= 4 * 1_000_000, "should have accrued ~8ms");
        reset();
    }

    #[test]
    fn siblings_do_not_steal_each_others_time() {
        let _serial = serial();
        reset();
        {
            let _p = open("parent");
            {
                let _a = open("a");
                busy(8);
            }
            {
                let _b = open("b");
                busy(8);
            }
        }
        let t = merged();
        assert_eq!(t["a"].calls, 1);
        assert_eq!(t["b"].calls, 1);
        // Both children's totals come off the parent's self time.
        assert!(
            t["parent"].self_ns < t["a"].self_ns,
            "parent self {} should be under one child's self {}",
            t["parent"].self_ns,
            t["a"].self_ns
        );
        reset();
    }

    #[test]
    fn accumulates_across_threads() {
        let _serial = serial();
        reset();
        let handles: Vec<_> = (0..4)
            .map(|_| {
                std::thread::spawn(|| {
                    let _g = open("worker");
                    busy(10);
                })
            })
            .collect();
        for h in handles {
            h.join().expect("worker joined");
        }
        let acc = merged()["worker"];
        assert_eq!(acc.calls, 4, "every thread's table must be merged");
        // Self time SUMS over threads, so 4 threads x ~10ms exceeds the ~10ms
        // of wall clock they took. That is intended and is how the report
        // exposes parallel efficiency.
        assert!(
            acc.self_ns >= 30_000_000,
            "expected ~40ms of summed CPU, got {}ns",
            acc.self_ns
        );
        reset();
    }

    #[test]
    fn report_is_sorted_by_self_time_and_totals_up() {
        let _serial = serial();
        reset();
        {
            let _s = open("small");
            busy(3);
        }
        {
            let _b = open("big");
            busy(25);
        }
        let r = report();
        let big = r.find("big").expect("big listed");
        let small = r.find("small").expect("small listed");
        assert!(big < small, "heaviest region must sort first:\n{r}");
        assert!(r.contains("TOTAL self"));
        reset();
    }

    #[test]
    fn empty_report_says_so_rather_than_printing_a_bare_header() {
        let _serial = serial();
        reset();
        assert!(report().contains("no spans recorded"));
    }
}
