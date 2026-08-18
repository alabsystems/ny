// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Dark, print-only PHASE telemetry for the root pipeline (#phase-telemetry).
//!
//! `NY_PHASE_TELEMETRY=1` enables; the declared `false` default emits no output.
//! The banking ledger (docs/BANKING_SWEEP_2026-07-18.md, last entries)
//! established that single-row wall-time deltas are unpriceable across builds
//! (~±15% layout noise) — lever pricing needs PHASE boundaries. Each marker is
//! ONE stderr line:
//!
//! ```text
//! [phase] <name> t=<secs-since-first-marker:.1>s
//! ```
//!
//! All markers share ONE process-wide clock (an `Instant` captured lazily at
//! the first emitted marker), so per-phase durations are simple differences
//! between adjacent lines in a log. Print-only: no marker feeds any bound,
//! verdict, or schedule decision. Call sites check the gate FIRST, so the
//! declared-false path is one latched-string compare — no formatting or
//! allocation. Armed-vs-unarmed deadline/verdict parity is not claimed.
//! Existing lane markers (`[root-crown-interm-tighten]` END elapsed, the
//! `[converge]` per-batch prints, the margin-row arm/report lines) are
//! unchanged and complementary.

use std::sync::OnceLock;
use std::time::Instant;

/// Pure gate predicate: exactly `"1"` enables. Split from the cached env
/// reader so the semantics stay unit-testable without mutating the process
/// environment (env-var tests are racy under parallel test threads — same
/// idiom as `resnet_decompose::env_gate_default_on`).
fn gate_on(raw: Option<&str>) -> bool {
    raw == Some("1")
}

/// Uncached env read through the ny-levers chokepoint's raw view — the cache
/// initializer, and the deterministic seam the smoke test drives under the
/// crate's `with_serialized_env_vars` idiom.
fn raw_uncached() -> Option<String> {
    ny_levers::read_raw(&ny_levers::decls::telemetry::PHASE_TELEMETRY)
}

/// Latched RAW env string (lever-debt batch B1 preparation). Marker sites are
/// hot (per-depth in the batched BaB lane), so the STRING is latched once and
/// the decision is derived per call by [`gate_on`]. This remains process-wide;
/// Phase 2 must replace it with an injected per-run `LeverSet`.
fn env_raw() -> Option<&'static str> {
    static RAW: OnceLock<Option<String>> = OnceLock::new();
    RAW.get_or_init(raw_uncached).as_deref()
}

/// Process-wide gate over the latched raw string. Checked FIRST at every
/// marker site; when the env is unset this is a latched-string compare and
/// nothing else.
pub(crate) fn phase_telemetry_enabled() -> bool {
    gate_on(env_raw())
}

/// Shared marker clock: captured at the FIRST emitted marker so every line in
/// a process prints seconds since the same epoch.
fn epoch() -> Instant {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    *EPOCH.get_or_init(Instant::now)
}

/// Core emitter, pure of I/O: `None` when the gate is off (BEFORE any
/// formatting — gate-off allocates nothing), otherwise the formatted marker
/// line. Split out so the smoke test can force the gate on/off and assert the
/// off path produces nothing without capturing stderr.
fn marker_line_if(enabled: bool, name: &str) -> Option<String> {
    if !enabled {
        return None;
    }
    Some(format!(
        "[phase] {name} t={:.1}s",
        epoch().elapsed().as_secs_f64()
    ))
}

/// Emit one `[phase]` marker line to stderr if `NY_PHASE_TELEMETRY=1`.
///
/// Call sites with DYNAMIC content (iteration indices, counts) must guard
/// their own `format!` behind [`phase_telemetry_enabled`] so the gate-off path
/// stays allocation-free; static-`&str` sites may call this directly.
pub(crate) fn phase_marker(name: &str) {
    if let Some(line) = marker_line_if(phase_telemetry_enabled(), name) {
        eprintln!("{line}");
    }
}

/// Pure formatting core for a `[frontier]` frame (#boxlift), I/O-free: `None`
/// when the gate is off (BEFORE any formatting — gate-off allocates nothing),
/// otherwise the formatted frame line. `secs` is injected so the unit test can
/// assert the exact output for given inputs without touching the shared epoch.
///
/// ```text
/// [frontier] d=<depth> worst=<margin:.5> domains=<cumulative> t=<secs:.1>s
/// ```
fn frontier_frame_line_if(
    enabled: bool,
    depth: usize,
    worst_margin: f32,
    domains_cumulative: u64,
    secs: f64,
) -> Option<String> {
    if !enabled {
        return None;
    }
    Some(format!(
        "[frontier] d={depth} worst={worst_margin:.5} domains={domains_cumulative} t={secs:.1}s"
    ))
}

/// Emit one `[frontier]` frame line to stderr if `NY_PHASE_TELEMETRY=1`
/// (#boxlift, per-depth worst-child telemetry for the resnet batched BaB
/// lane). Same contract as [`phase_marker`]: the cached gate is checked FIRST
/// (gate-off is one boolean load — no formatting, no clock read), timestamps
/// come from the shared process epoch, and the frame is print-only — nothing
/// downstream ever reads it.
pub(crate) fn frontier_frame(depth: usize, worst_margin: f32, domains_cumulative: u64) {
    if !phase_telemetry_enabled() {
        return;
    }
    if let Some(line) = frontier_frame_line_if(
        true,
        depth,
        worst_margin,
        domains_cumulative,
        epoch().elapsed().as_secs_f64(),
    ) {
        eprintln!("{line}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Uncached decision, rebuilt from the raw chokepoint view on every call —
    /// the deterministic seam the env-gate test drives (the production path
    /// latches the string in [`env_raw`], which another test in this process
    /// may already have initialized).
    fn enabled_uncached() -> bool {
        gate_on(raw_uncached().as_deref())
    }

    /// Smoke test (#phase-telemetry): the gate-off path produces NOTHING and
    /// the gate-on path produces the bootstrap markers in the documented
    /// format. Capturing `eprintln!` in-process is awkward under libtest, so
    /// this drives the marker helper's pure core directly with the gate
    /// forced on/off (scope-sanctioned equivalent).
    #[test]
    fn phase_marker_gate_off_prints_nothing_gate_on_prints_bootstrap_markers() {
        // Gate OFF: nothing is produced (and nothing is formatted).
        assert_eq!(marker_line_if(false, "graph-bab-bootstrap start"), None);
        assert_eq!(marker_line_if(false, "graph-bab-bootstrap end"), None);

        // Gate ON: both bootstrap markers emit in the documented format
        // "[phase] <name> t=<secs:.1>s" against the shared process epoch.
        for name in ["graph-bab-bootstrap start", "graph-bab-bootstrap end"] {
            let line = marker_line_if(true, name).expect("gate-on must emit a marker line");
            let prefix = format!("[phase] {name} t=");
            assert!(
                line.starts_with(&prefix) && line.ends_with('s'),
                "malformed marker line: {line:?}"
            );
            let secs: f64 = line[prefix.len()..line.len() - 1]
                .parse()
                .expect("marker t= field must parse as seconds");
            assert!(
                secs >= 0.0 && secs.is_finite(),
                "marker seconds must be finite and non-negative, got {secs}"
            );
        }
    }

    /// #boxlift smoke test: the gate-off path of the `[frontier]` frame
    /// produces NOTHING (the `None` arm — no formatting is ever reached) and
    /// the gate-on path formats the documented frame exactly from given
    /// inputs. Drives the pure core directly with the gate forced on/off and
    /// an injected clock, same idiom as the `[phase]` marker smoke test above.
    #[test]
    fn frontier_frame_gate_off_prints_nothing_gate_on_formats_frame() {
        // Gate OFF: nothing is produced (and nothing is formatted).
        assert_eq!(frontier_frame_line_if(false, 5, -0.134, 512, 42.71), None);

        // Gate ON: exact documented format
        // "[frontier] d=<depth> worst=<margin:.5> domains=<cumulative> t=<secs:.1>s".
        let line = frontier_frame_line_if(true, 5, -0.134, 512, 42.71)
            .expect("gate-on must emit a frontier frame");
        assert_eq!(line, "[frontier] d=5 worst=-0.13400 domains=512 t=42.7s");

        // Positive-margin / zero-domain corners keep the same field widths.
        let line = frontier_frame_line_if(true, 0, 0.03, 1, 0.0)
            .expect("gate-on must emit a frontier frame");
        assert_eq!(line, "[frontier] d=0 worst=0.03000 domains=1 t=0.0s");
    }

    /// The env gate reads exactly `"1"` as ON — forced on/off via the crate's
    /// serialized env-var idiom against the UNCACHED reader (the `OnceLock`
    /// wrapper is deliberately not asserted here: another test in this
    /// process may already have initialized it through a production path).
    #[test]
    fn phase_telemetry_env_gate_semantics() {
        crate::tests::with_serialized_env_vars_removed(&["NY_PHASE_TELEMETRY"], || {
            assert!(!enabled_uncached(), "unset must be OFF (silent)");
        });
        crate::tests::with_serialized_env_vars(&[("NY_PHASE_TELEMETRY", "0")], || {
            assert!(!enabled_uncached(), "\"0\" must be OFF (silent)");
        });
        crate::tests::with_serialized_env_vars(&[("NY_PHASE_TELEMETRY", "true")], || {
            assert!(!enabled_uncached(), "non-\"1\" must be OFF (silent)");
        });
        crate::tests::with_serialized_env_vars(&[("NY_PHASE_TELEMETRY", "1")], || {
            assert!(enabled_uncached(), "\"1\" must be ON");
        });
        // Pure predicate: same table without touching process env.
        assert!(!gate_on(None));
        assert!(!gate_on(Some("0")));
        assert!(!gate_on(Some("")));
        assert!(gate_on(Some("1")));
    }
}
