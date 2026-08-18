// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Flight recorder v0 (#flight-record, invariant I7 of
//! `docs/SYSTEM_DESIGN_ONE_PIPELINE_2026-07-30.md`).
//!
//! Every scored run must leave a machine-readable record of WHICH methods
//! ran/skipped and on what backend — in the result ARTIFACT, not the log
//! stream: the scored entry points run under `RUST_LOG=error`
//! (`run_instance.sh`, `vnncomp_sweep.rs`), so an info-level line is invisible
//! exactly where the record matters. Post-hoc questions like "did the upfront
//! attack lane ever arm on this row?" have repeatedly been unanswerable from
//! the bank (design doc §1 instances 3/6/7/9/10); the sidecar written here is
//! the artifact that answers them.
//!
//! v0 is deliberately shallow: the command layer notes the seams it already
//! knows (lane consulted/skipped, route chosen, preset resolved), with no
//! plumbing into ny-propagate. The record is process-global because the scored
//! contract is one instance per process (`ny vnncomp` is spawned per row by
//! both the competition harness and the sweep).
//!
//! Best-effort by construction: a sidecar write failure must NEVER affect the
//! verdict or the exit code — the recorder logs on stderr (which survives
//! `RUST_LOG=error` and lands in the per-instance logs) and continues.

use serde::Serialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Bump when the sidecar shape changes; consumers (the sweep's metadata bank,
/// the expected-methods contracts) key on this.
///
/// v2: `ambient_env` (NY_* + OMP_NUM_THREADS snapshot, I10 flag-zero) and the
/// terminal `run_complete` event (lifecycle v0.5).
///
/// v3: adds a two-stage `levers` envelope. `begin` freezes the registered raw
/// environment inputs and records `not_materialized`; after the run context is
/// known and any preset snapshot is semantically validated, the command
/// replaces that state once with a layered registry receipt or an explicit
/// `invalid_config` reason. Older v2 sidecars remain a supported input to the
/// archival/qualification consumers.
const SCHEMA_VERSION: u32 = 3;

/// What happened to one method/lane at a seam the command layer controls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum FlightStatus {
    /// The lane/decision was entered or consulted.
    Ran,
    /// The lane was reachable but a gate declined it; `reason` says which.
    Skipped,
    /// Control flow never got there (an earlier stage terminated the run).
    #[allow(dead_code)] // Wire-contract sentinel exercised by recorder tests.
    NotReached,
    /// Terminal disposition (lifecycle v0.5): used only by the one
    /// `run_complete` event that closes a record, so consumers can find the
    /// final verdict by type instead of by grepping method names.
    Complete,
}

/// One recorded seam: method name, disposition, why, and when.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct FlightEvent {
    pub(crate) method: String,
    pub(crate) status: FlightStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) reason: Option<String>,
    /// Seconds since `begin` — the scored instance start, near enough for v0.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) at_secs: Option<f64>,
}

/// Whether this record reached the point where layered registry resolution was
/// possible.
///
/// The scored command begins recording before it validates input files or
/// loads a category preset. An early error must not invent a default-only
/// receipt, because a valid run might later have received a typed preset.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub(crate) enum LeverReceiptState {
    /// Preset/config resolution did not complete. This is affirmative evidence
    /// that no layered receipt is being claimed, not a missing field.
    NotMaterialized,
    /// Preset/config resolution failed, or the typed receipt projection
    /// disagreed with its declaration. The reason is machine-readable
    /// evidence that no layered receipt is being claimed.
    InvalidConfig { reason: String },
    /// The registered raw environment snapshot was layered over the exact
    /// validated preset/config values for this run.
    Resolved { receipt: serde_json::Value },
}

/// The per-run record serialized to `<results_file>.flight.json`.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct FlightRecord {
    pub(crate) schema_version: u32,
    /// `cuda` / `metal` / `gpu` / `cpu-only` from `compute_backend::detect()`.
    pub(crate) backend_kind: String,
    /// The one-line provenance summary sealed alongside the kind.
    pub(crate) backend_summary: String,
    /// Static host identity (#host-provenance): which machine class produced
    /// this timing.
    pub(crate) host: crate::compute_backend::HostProbe,
    /// 1/5/15-minute load averages sampled at `begin` — whether the box was
    /// quiet when the clock started.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) load_avg_at_begin: Option<[f64; 3]>,
    /// Sampled again when the sidecar is written, i.e. after the verdict —
    /// together with `load_avg_at_begin` this brackets the run. A timing
    /// whose brackets show heavy contention (load ≫ cores) is evidence of a
    /// measurement problem, recorded instead of remembered.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) load_avg_at_end: Option<[f64; 3]>,
    pub(crate) category: String,
    /// The scored competition budget, not any internal tier.
    pub(crate) budget_secs: u64,
    /// Every `NY_*` environment variable plus `OMP_NUM_THREADS`, snapshotted
    /// at `begin` (I10 #flag-zero, week-one enforcement): a dev-override that
    /// shaped this run must be visible in the artifact, because the bank
    /// linter's later refusal of non-allowlisted-flag rows can only judge
    /// what the record shows. BTreeMap so the serialization is sorted and
    /// deterministic. Always present — an empty map is affirmative evidence
    /// the run was flag-clean, not a missing measurement.
    pub(crate) ambient_env: BTreeMap<String, String>,
    /// Two-stage declared-input evidence. See [`LeverReceiptState`].
    pub(crate) levers: LeverReceiptState,
    pub(crate) events: Vec<FlightEvent>,
}

struct RecorderState {
    record: FlightRecord,
    /// Exact registered environment values captured at `begin`, including
    /// present non-UTF-8 values. Preset materialization must resolve against
    /// this frozen input rather than reopening the process environment.
    raw_levers: ny_levers::RawLeverInputs,
    /// Anchor for `at_secs`; monotonic so a wall-clock step cannot produce
    /// negative or wildly wrong offsets in the artifact.
    started: std::time::Instant,
    /// Terminal verdict supplied by `finish`. Held OUT of `record.events` so
    /// the `run_complete` event is appended exactly once per serialized
    /// sidecar, however many times `write_sidecar` runs.
    verdict: Option<String>,
}

/// Filter an arbitrary `(name, value)` stream down to the ambient flags the
/// record captures: every `NY_*` variable plus `OMP_NUM_THREADS`.
///
/// Takes an iterator instead of reading the process environment so unit tests
/// need no env mutation — the real environment is process-global and tests run
/// in parallel; injected pairs sidestep that entirely.
pub(crate) fn ambient_env_from(
    vars: impl IntoIterator<Item = (String, String)>,
) -> BTreeMap<String, String> {
    vars.into_iter()
        .filter(|(name, _)| name.starts_with("NY_") || name == "OMP_NUM_THREADS")
        .collect()
}

/// The production capture: the real process environment, lossily decoded so a
/// non-UTF-8 value stays visible in the artifact rather than silently vanishing.
fn ambient_env_snapshot() -> BTreeMap<String, String> {
    ambient_env_from(std::env::vars_os().map(|(name, value)| {
        (
            name.to_string_lossy().into_owned(),
            value.to_string_lossy().into_owned(),
        )
    }))
}

/// Event sink with interior mutability so call sites need no plumbing.
///
/// Tests construct their own instances; production code uses [`global`]. Lock
/// poisoning is deliberately shrugged off (`into_inner`): the recorder must
/// never convert someone else's panic into a second failure on the verdict
/// path.
pub(crate) struct FlightRecorder {
    state: Mutex<Option<RecorderState>>,
}

impl FlightRecorder {
    pub(crate) const fn new() -> Self {
        Self {
            state: Mutex::new(None),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Option<RecorderState>> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Arm the recorder for this run, replacing any previous record.
    ///
    /// Until `begin` runs, `note` and `write_sidecar` are no-ops — entry
    /// points that never opted into flight recording cannot leave partial or
    /// misattributed sidecars.
    pub(crate) fn begin(
        &self,
        backend_kind: &str,
        backend_summary: &str,
        category: &str,
        budget_secs: u64,
    ) {
        // Anchor timings before provenance capture: enumerating the host and
        // environment is part of the run whose timing the record describes.
        let started = std::time::Instant::now();
        let raw_levers = ny_levers::RawLeverInputs::capture(ny_levers::all());
        *self.lock() = Some(RecorderState {
            record: FlightRecord {
                schema_version: SCHEMA_VERSION,
                backend_kind: backend_kind.to_string(),
                backend_summary: backend_summary.to_string(),
                host: crate::compute_backend::host().clone(),
                load_avg_at_begin: crate::compute_backend::load_average(),
                load_avg_at_end: None,
                category: category.to_string(),
                budget_secs,
                ambient_env: ambient_env_snapshot(),
                levers: LeverReceiptState::NotMaterialized,
                events: Vec::new(),
            },
            raw_levers,
            started,
            verdict: None,
        });
    }

    /// Materialize the layered registry receipt after typed preset/config
    /// validation succeeds.
    ///
    /// `config_value` supplies the contextual lower layer. The raw environment
    /// captured by [`Self::begin`] has precedence, including a present invalid
    /// value that deliberately suppresses a preset. A no-preset run calls this
    /// with an all-`None` lookup and therefore still gets a resolved receipt.
    pub(crate) fn materialize_levers(
        &self,
        config_value: impl FnMut(&'static ny_levers::LeverDecl) -> Option<ny_levers::LeverValue>,
    ) {
        if let Some(state) = self.lock().as_mut() {
            if !matches!(state.record.levers, LeverReceiptState::NotMaterialized) {
                // Like `finish`, first authority wins. A duplicate call is a
                // recorder bug, but instrumentation must not rewrite evidence
                // or perturb the verifier's outcome.
                return;
            }
            state.record.levers = match ny_levers::LeverSet::resolve_layered(
                ny_levers::all(),
                &state.raw_levers,
                config_value,
            ) {
                Ok(levers) => LeverReceiptState::Resolved {
                    receipt: levers.receipt(),
                },
                Err(error) => LeverReceiptState::InvalidConfig {
                    reason: error.to_string(),
                },
            };
        }
    }

    /// Record that the frozen preset/config was invalid before a layered
    /// registry receipt could be materialized.
    ///
    /// This is evidence only. It must never replace an already-resolved state
    /// or become a second error path for the verifier.
    pub(crate) fn mark_levers_invalid_config(&self, reason: &str) {
        if let Some(state) = self.lock().as_mut() {
            if matches!(state.record.levers, LeverReceiptState::NotMaterialized) {
                state.record.levers = LeverReceiptState::InvalidConfig {
                    reason: reason.to_owned(),
                };
            }
        }
    }

    /// Append one event. A no-op before `begin` (see there for why).
    pub(crate) fn note(&self, method: &str, status: FlightStatus, reason: Option<String>) {
        if let Some(state) = self.lock().as_mut() {
            let at_secs = state.started.elapsed().as_secs_f64();
            state.record.events.push(FlightEvent {
                method: method.to_string(),
                status,
                reason,
                at_secs: Some(at_secs),
            });
        }
    }

    /// A copy of the current record, for tests and the final serialization.
    pub(crate) fn snapshot(&self) -> Option<FlightRecord> {
        self.lock().as_ref().map(|state| state.record.clone())
    }

    /// Record the terminal verdict (lifecycle v0.5). A no-op before `begin`.
    ///
    /// The first verdict wins: the command's exit paths are mutually
    /// exclusive, so a second call can only be a bug, and the record must
    /// keep the verdict that was actually published, not the latest caller's
    /// opinion of it.
    pub(crate) fn finish(&self, verdict: &str) {
        if let Some(state) = self.lock().as_mut() {
            if state.verdict.is_none() {
                state.verdict = Some(verdict.to_string());
            }
        }
    }

    /// Serialize the record to `<results_file>.flight.json`, best-effort.
    ///
    /// INFALLIBLE BY CONTRACT: whatever goes wrong (record never begun, JSON
    /// failure, unwritable path), the verdict and exit code are already
    /// decided elsewhere and must not change — log on stderr and return. The
    /// sidecar sits NEXT TO the results file because that is the one location
    /// every harness (competition wrapper, sweep child tempdir) already
    /// collects.
    pub(crate) fn write_sidecar(&self, results_file: &Path) {
        let Some(mut record) = self.snapshot() else {
            return;
        };
        let terminal = self.lock().as_ref().and_then(|state| {
            state
                .verdict
                .clone()
                .map(|verdict| (verdict, state.started.elapsed().as_secs_f64()))
        });
        // Terminal disposition (lifecycle v0.5): appended to the serialized
        // copy, never the live record, so the sidecar carries exactly one
        // `run_complete` however many exit paths write it. `at_secs` here is
        // the total elapsed since `begin` — the run's wall time.
        if let Some((verdict, elapsed)) = terminal {
            record.events.push(FlightEvent {
                method: "run_complete".to_string(),
                status: FlightStatus::Complete,
                reason: Some(verdict),
                at_secs: Some(elapsed),
            });
        }
        // Close the load bracket at write time — after the verdict, so the
        // sample covers the run it describes.
        record.load_avg_at_end = crate::compute_backend::load_average();
        let path = sidecar_path(results_file);
        let body = match serde_json::to_string_pretty(&record) {
            Ok(body) => body,
            Err(error) => {
                eprintln!(
                    "flight recorder: serialization failed ({error}); the verdict is unaffected"
                );
                return;
            }
        };
        if let Err(error) = std::fs::write(&path, body) {
            eprintln!(
                "flight recorder: could not write {} ({error}); the verdict is unaffected",
                path.display()
            );
        }
    }
}

/// The process-global recorder the scored `ny vnncomp` path writes through.
pub(crate) fn global() -> &'static FlightRecorder {
    static GLOBAL: FlightRecorder = FlightRecorder::new();
    &GLOBAL
}

/// `<results_file>.flight.json` — appended, not `with_extension`, so
/// `result.txt` maps to `result.txt.flight.json` and can never collide with
/// the results file itself.
pub(crate) fn sidecar_path(results_file: &Path) -> PathBuf {
    let mut os = results_file.as_os_str().to_os_string();
    os.push(".flight.json");
    PathBuf::from(os)
}

/// Convenience for call sites deep in the command layer.
pub(crate) fn note(method: &str, status: FlightStatus, reason: Option<String>) {
    global().note(method, status, reason);
}

/// Convenience for the command's exit paths: record the terminal verdict on
/// the global recorder before the sidecar write.
pub(crate) fn finish(verdict: &str) {
    global().finish(verdict);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recorder_captures_events_in_order_with_timing_and_header() {
        let recorder = FlightRecorder::new();
        recorder.begin("cpu-only", "cpu-only [test]", "acasxu_2023", 116);
        recorder.note("upfront_attack", FlightStatus::Ran, None);
        recorder.note(
            "margin_row_concurrent",
            FlightStatus::Skipped,
            Some("not armed".into()),
        );
        recorder.note("postbab_leftover", FlightStatus::NotReached, None);

        let record = recorder.snapshot().expect("record exists after begin");
        assert_eq!(record.schema_version, 3);
        assert_eq!(record.backend_kind, "cpu-only");
        assert_eq!(record.category, "acasxu_2023");
        assert_eq!(record.budget_secs, 116);
        let methods: Vec<&str> = record
            .events
            .iter()
            .map(|event| event.method.as_str())
            .collect();
        assert_eq!(
            methods,
            [
                "upfront_attack",
                "margin_row_concurrent",
                "postbab_leftover"
            ],
            "events must keep call order — the record IS the execution trace"
        );
        assert_eq!(record.events[1].status, FlightStatus::Skipped);
        assert_eq!(record.events[1].reason.as_deref(), Some("not armed"));
        for event in &record.events {
            let at = event.at_secs.expect("every noted event is timestamped");
            assert!(at >= 0.0, "monotonic anchor cannot go negative: {at}");
        }
    }

    #[test]
    fn sidecar_lands_next_to_the_results_file_as_machine_readable_json() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let results_file = tmp.path().join("result.txt");
        let recorder = FlightRecorder::new();
        recorder.begin("metal", "metal [test]", "cgan_2023", 300);
        recorder.note(
            "preset",
            FlightStatus::Ran,
            Some("configs/vnncomp26/cgan_2023.yaml".into()),
        );
        recorder.note("internal_verifier", FlightStatus::NotReached, None);
        recorder.write_sidecar(&results_file);

        let sidecar = tmp.path().join("result.txt.flight.json");
        let body = std::fs::read_to_string(&sidecar).expect("sidecar written next to results");
        let json: serde_json::Value = serde_json::from_str(&body).expect("sidecar is valid JSON");
        assert_eq!(json["schema_version"], 3);
        assert_eq!(json["levers"]["status"], "not_materialized");
        assert_eq!(json["backend_kind"], "metal");
        assert_eq!(json["budget_secs"], 300);
        assert!(
            json["ambient_env"].is_object(),
            "the flag snapshot is always present, even when empty (flag-clean is a measurement)"
        );
        assert_eq!(json["events"][0]["method"], "preset");
        assert_eq!(json["events"][0]["status"], "ran");
        // The wire spelling of NotReached is contract: consumers grep for it.
        assert_eq!(json["events"][1]["status"], "not_reached");
        // No `finish` was called, so no terminal event may be invented.
        let events = json["events"].as_array().expect("events array");
        assert!(
            events.iter().all(|event| event["method"] != "run_complete"),
            "run_complete requires an explicit verdict from the caller"
        );
    }

    #[test]
    fn a_failed_sidecar_write_is_non_fatal() {
        let tmp = tempfile::tempdir().expect("tempdir");
        // A regular file where a directory is needed makes every write under
        // it fail with ENOTDIR on every platform we build for.
        let blocker = tmp.path().join("blocker");
        std::fs::write(&blocker, b"not a directory").expect("write blocker");
        let results_file = blocker.join("result.txt");

        let recorder = FlightRecorder::new();
        recorder.begin("cpu-only", "cpu-only [test]", "safenlp_2024", 20);
        recorder.note("internal_verifier", FlightStatus::Ran, None);
        // Must return normally: the write failure may cost the artifact but
        // never the verdict or the exit code.
        recorder.write_sidecar(&results_file);
        assert!(
            recorder.snapshot().is_some(),
            "the in-memory record survives a failed write"
        );
    }

    #[test]
    fn before_begin_the_recorder_is_inert() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let results_file = tmp.path().join("result.txt");
        let recorder = FlightRecorder::new();
        recorder.note("upfront_attack", FlightStatus::Ran, None);
        recorder.write_sidecar(&results_file);
        assert!(
            recorder.snapshot().is_none(),
            "note before begin is a no-op"
        );
        assert!(
            !sidecar_path(&results_file).exists(),
            "no partial sidecar may appear for a run that never began recording"
        );
    }

    #[test]
    fn ambient_capture_keeps_ny_flags_and_omp_threads_only_sorted() {
        // Injected pairs, not the process environment: env is process-global
        // and tests run in parallel, so the capture is tested at the seam
        // that filters, with production supplying the real iterator.
        let captured = ambient_env_from([
            ("PATH".to_string(), "/usr/bin".to_string()),
            ("NY_TEST_FLAG".to_string(), "1".to_string()),
            ("NY_ATTACK_ARMING_BLOCK".to_string(), "1".to_string()),
            ("OMP_NUM_THREADS".to_string(), "8".to_string()),
            ("NY_A_EARLIER".to_string(), "yes".to_string()),
            // Prefix must anchor at the start: an embedded NY_ is not a flag.
            ("SOME_NY_LOOKALIKE".to_string(), "no".to_string()),
            ("HOME".to_string(), "/var/empty".to_string()),
        ]);
        let pairs: Vec<(&str, &str)> = captured
            .iter()
            .map(|(name, value)| (name.as_str(), value.as_str()))
            .collect();
        assert_eq!(
            pairs,
            [
                ("NY_ATTACK_ARMING_BLOCK", "1"),
                ("NY_A_EARLIER", "yes"),
                ("NY_TEST_FLAG", "1"),
                ("OMP_NUM_THREADS", "8"),
            ],
            "NY_* plus OMP_NUM_THREADS, sorted by name — deterministic artifact bytes"
        );
    }

    #[test]
    fn begin_records_that_layered_levers_are_not_yet_materialized() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let results_file = tmp.path().join("result.txt");
        let recorder = FlightRecorder::new();
        recorder.begin("cpu-only", "cpu-only [test]", "acasxu_2023", 116);
        recorder.write_sidecar(&results_file);

        let body = std::fs::read_to_string(sidecar_path(&results_file)).expect("sidecar written");
        let json: serde_json::Value = serde_json::from_str(&body).expect("sidecar is valid JSON");
        assert_eq!(json["levers"]["status"], "not_materialized");
        assert!(
            json["levers"].get("receipt").is_none(),
            "begin cannot claim a layered default before preset resolution"
        );
    }

    #[test]
    fn materialize_levers_projects_the_contextual_config_layer() {
        ny_test_utils::env::with_serialized_env_vars_removed(&["NY_ALPHA_ZERO_YIELD_FRAC"], || {
            let recorder = FlightRecorder::new();
            recorder.begin("cpu-only", "cpu-only [test]", "custom", 116);
            recorder.materialize_levers(|decl| {
                std::ptr::eq(
                    decl,
                    &raw const ny_levers::decls::root_alpha::ALPHA_ZERO_YIELD_FRAC,
                )
                .then_some(ny_levers::LeverValue::F64(0.25))
            });
            recorder.materialize_levers(|decl| {
                std::ptr::eq(
                    decl,
                    &raw const ny_levers::decls::root_alpha::ALPHA_ZERO_YIELD_FRAC,
                )
                .then_some(ny_levers::LeverValue::F64(0.5))
            });
            let record = recorder.snapshot().expect("record exists");
            let LeverReceiptState::Resolved { receipt } = record.levers else {
                panic!("validated config must resolve the receipt");
            };
            assert_eq!(receipt["schema"], "ny-levers/receipt/v2");
            let levers = receipt["levers"].as_array().expect("levers array");
            let zero_yield = levers
                .iter()
                .find(|l| l["name"] == "NY_ALPHA_ZERO_YIELD_FRAC")
                .expect("declared lever present");
            assert_eq!(zero_yield["value"], 0.25);
            assert_eq!(zero_yield["source"], "config");
        });
    }

    #[test]
    fn inadmissible_typed_projection_is_recorded_without_panicking() {
        let recorder = FlightRecorder::new();
        recorder.begin("cpu-only", "cpu-only [test]", "custom", 116);
        recorder.materialize_levers(|decl| {
            std::ptr::eq(
                decl,
                &raw const ny_levers::decls::root_alpha::ALPHA_ZERO_YIELD_FRAC,
            )
            .then_some(ny_levers::LeverValue::F64(0.95))
        });

        let record = recorder.snapshot().expect("record exists");
        let LeverReceiptState::InvalidConfig { reason } = record.levers else {
            panic!("inadmissible typed config must be explicit evidence");
        };
        assert!(reason.contains("NY_ALPHA_ZERO_YIELD_FRAC"));
    }

    #[test]
    fn invalid_preset_reason_is_explicit_and_first_authority_wins() {
        let recorder = FlightRecorder::new();
        recorder.begin("cpu-only", "cpu-only [test]", "custom", 116);
        recorder.mark_levers_invalid_config("preset field was out of range");
        recorder.materialize_levers(|_| None);

        let record = recorder.snapshot().expect("record exists");
        let LeverReceiptState::InvalidConfig { reason } = record.levers else {
            panic!("invalid preset must remain explicit evidence");
        };
        assert_eq!(reason, "preset field was out of range");
    }

    #[test]
    fn finish_appends_exactly_one_terminal_event_with_verdict_and_elapsed() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let results_file = tmp.path().join("result.txt");
        let recorder = FlightRecorder::new();
        recorder.begin("cpu-only", "cpu-only [test]", "acasxu_2023", 116);
        recorder.note("result_publish", FlightStatus::Ran, Some("unsat".into()));
        recorder.finish("unsat");
        // A buggy second finish must not rewrite the published verdict.
        recorder.finish("sat");
        // Two writes model two exit paths racing to serialize; each sidecar
        // must still end with exactly one run_complete.
        recorder.write_sidecar(&results_file);
        recorder.write_sidecar(&results_file);

        let body = std::fs::read_to_string(sidecar_path(&results_file)).expect("sidecar written");
        let json: serde_json::Value = serde_json::from_str(&body).expect("sidecar is valid JSON");
        let events = json["events"].as_array().expect("events array");
        let terminals: Vec<_> = events
            .iter()
            .filter(|event| event["method"] == "run_complete")
            .collect();
        assert_eq!(terminals.len(), 1, "exactly one terminal disposition");
        let terminal = events.last().expect("at least the terminal event");
        assert_eq!(
            terminal["method"], "run_complete",
            "the terminal event closes the record"
        );
        assert_eq!(terminal["status"], "complete");
        assert_eq!(terminal["reason"], "unsat", "first verdict wins");
        let elapsed = terminal["at_secs"].as_f64().expect("total elapsed");
        assert!(elapsed >= 0.0, "elapsed is measured from begin: {elapsed}");
    }

    #[test]
    fn finish_before_begin_is_inert() {
        let recorder = FlightRecorder::new();
        recorder.finish("unsat");
        assert!(
            recorder.snapshot().is_none(),
            "a verdict cannot arm a recorder that never began"
        );
    }

    #[test]
    fn sidecar_path_appends_rather_than_replacing_the_extension() {
        assert_eq!(
            sidecar_path(Path::new("/tmp/result.txt")),
            Path::new("/tmp/result.txt.flight.json"),
            "with_extension would map result.txt and result.csv to the same sidecar"
        );
    }
}
