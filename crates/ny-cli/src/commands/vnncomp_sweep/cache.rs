// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Content-addressed verdict cache for `ny benchmarks run`.
//!
//! WHY. A lever search measures the same (row, arm) pair many times: re-running a
//! sweep after adding one arm re-measures every pair that did not change, and a
//! moat gate re-measures the same 41 rows for every candidate. At ~100 s per row
//! that dominates everything else the platform does. The cache turns an arm x row
//! matrix into new-pairs-only.
//!
//! WHY IT IS SAFE TO CACHE A VERDICT AT ALL. A verdict is a pure function of the
//! binary, the configuration, the CATEGORY, the instance, the budget and the arm
//! — provided every one of those is in the key. [`CacheKey`] therefore includes
//! all of them, and the three that are easiest to forget are the ones this
//! repository has already been burned by: the ARM (a lever exported into the
//! parent shell used to be invisible to the artifact), the HOST/BACKEND (two
//! machines can both report `cuda` and still produce incomparable timings), and
//! the CATEGORY (it selects the preset the child loads, so two categories over
//! byte-identical model/property files are two different measurements).
//!
//! THE TWO ADMISSION RULES ARE NOT OPTIONAL. Both encode a class of fake result
//! this repository has actually produced:
//!
//! 1. **Never serve a `timeout` measured at a smaller budget.** A row that timed
//!    out at 30 s says nothing about the same row at 100 s — it may well solve.
//!    Serving it would manufacture a timeout that was never observed. (The
//!    reverse IS sound and is allowed: a timeout at a LARGER budget implies a
//!    timeout at a smaller one, which is why `capped_from` is in the key rather
//!    than being used to widen a hit.)
//!
//!    HOW MUCH OF THIS RULE IS LIVE AS WIRED, precisely. `budget_secs` is a
//!    [`CacheKey`] field and [`VerdictCache::get`] requires the stored key to
//!    equal the requested one before [`admission_refusal`] runs, so a differing
//!    budget is already a `NotPresent`/`Unusable` miss and the rule cannot fire
//!    on a well-formed entry written by [`VerdictCache::put`]: `put` stores a row
//!    whose `budget_secs` came from the same plan entry as the key's. What the
//!    rule still catches is the case the key comparison does NOT cover — the
//!    stored ROW is not compared field-by-field, only the stored KEY is, so an
//!    entry whose row body was edited or truncated to a smaller `budget_secs` is
//!    refused here rather than served. It is otherwise a standing guard on a
//!    FUTURE relaxation: the moment budget leaves the key (to let a 200 s
//!    timeout answer a 100 s request, which is sound), this branch is the only
//!    thing standing between a 30 s timeout and a manufactured 100 s verdict.
//!    That is why it stays.
//! 2. **Never serve a row measured under heavy contention.** `flight`'s
//!    `load_avg_at_begin` brackets the run; a row begun with load far above the
//!    core count is not a measurement of the verifier, it is a measurement of the
//!    machine's queue. Caching it would make one bad afternoon permanent. This
//!    one IS live on the wired path: load is not in the key, so it is checked
//!    only here.
//!
//! A miss is always safe. Every failure in this module — unreadable entry, bad
//! JSON, hash failure, missing directory — degrades to a miss and a fresh
//! measurement, never to an error and never to a stale hit.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// How the cache participates in a sweep.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum CacheMode {
    /// Ignore the cache entirely: measure everything, store nothing.
    #[default]
    Off,
    /// Serve hits, but do not record new measurements.
    Read,
    /// Serve hits and record every fresh measurement.
    ReadWrite,
}

impl CacheMode {
    pub(crate) fn parse(raw: &str) -> Result<Self> {
        match raw {
            "off" => Ok(Self::Off),
            "read" => Ok(Self::Read),
            "read-write" => Ok(Self::ReadWrite),
            other => anyhow::bail!("--cache expects off|read|read-write, got {other:?}"),
        }
    }

    pub(crate) fn reads(self) -> bool {
        matches!(self, Self::Read | Self::ReadWrite)
    }

    pub(crate) fn writes(self) -> bool {
        matches!(self, Self::ReadWrite)
    }
}

/// Everything a verdict depends on.
///
/// Serialized with sorted keys (via `BTreeMap` and declaration order) and hashed;
/// the hash is the entry filename. Adding a field is a cache-invalidating change
/// and MUST bump `v`.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct CacheKey {
    /// Key schema version. Bump on ANY change to the field set or semantics —
    /// an un-bumped change silently serves rows keyed under the old meaning.
    v: u32,
    exe_sha256: String,
    build_provenance: String,
    configs_sha256: String,
    /// The benchmark category, which is an INPUT to the child, not a label.
    ///
    /// `run_instance` invokes `ny vnncomp v1 <category> ...`; the category picks
    /// `<configs>/vnncomp*/{category}.yaml` and gates the category-fenced routes
    /// inside it. `configs_sha256` digests the whole configuration tree and so
    /// cannot say WHICH preset a row ran under. Omitting this was a live defect:
    /// two categories whose model and property files are byte-identical hashed
    /// to one entry, and a single cold `--cache read-write` sweep over such a
    /// corpus served the second category the first's verdict.
    category: String,
    onnx_sha256: String,
    vnnlib_sha256: String,
    budget_secs: u64,
    capped_from: Option<u64>,
    /// Sorted lever assignment. Sorting is load-bearing: `[(A,1),(B,0)]` and
    /// `[(B,0),(A,1)]` are the same arm and must hash identically.
    arm: Vec<(String, String)>,
    /// #arm-sealing: the AMBIENT lever set the sweep runs under, as sealed into
    /// `SweepManifest::ambient_env`.
    ///
    /// Omitting this was a live defect. `run_instance` only ADDS to the child
    /// environment, so a lever exported into the parent shell reaches every row
    /// while appearing nowhere in `arm` — two sweeps differing only by an
    /// exported `NY_*` var hashed identically, and the second would have been
    /// served the first's verdicts. That is precisely the "confident wrong
    /// measurement" this cache is otherwise built to avoid.
    ambient_env: BTreeMap<String, String>,
    compute_backend: String,
    host: String,
}

impl CacheKey {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        exe_sha256: String,
        build_provenance: String,
        configs_sha256: String,
        category: String,
        onnx: &Path,
        vnnlib: &Path,
        budget_secs: u64,
        capped_from: Option<u64>,
        arm: &[(String, String)],
        compute_backend: String,
        host: String,
        ambient_env: BTreeMap<String, String>,
        hashes: &mut FileHashMemo,
    ) -> Result<Self> {
        let mut arm = arm.to_vec();
        arm.sort();
        Ok(Self {
            // v2 added `ambient_env`. Every v1 digest described a run whose
            // parent-shell levers were unknown, so nothing keyed under v1 may
            // be served: the bump is the invalidation.
            //
            // v3 adds `category`. Every v2 digest described a run whose preset
            // was unknown, and a cross-category hit is a verdict for an instance
            // that was never measured — worse than no cache at all. Same rule:
            // the bump is the invalidation.
            v: 3,
            exe_sha256,
            build_provenance,
            configs_sha256,
            category,
            onnx_sha256: hashes.hash(onnx)?,
            vnnlib_sha256: hashes.hash(vnnlib)?,
            budget_secs,
            capped_from,
            arm,
            ambient_env,
            compute_backend,
            host,
        })
    }

    /// The entry's content address.
    pub(crate) fn digest(&self) -> Result<String> {
        let canonical = serde_json::to_vec(self).context("serialize cache key")?;
        let mut hasher = Sha256::new();
        hasher.update(&canonical);
        Ok(format!("{:x}", hasher.finalize()))
    }
}

/// Memoized file hashing.
///
/// A 100 MB ONNX is hashed once per sweep rather than once per row. Keyed on
/// `(path, len, mtime)` so an edited model is re-hashed rather than being served
/// under its old digest — the whole point of content addressing would be lost if
/// a stale mtime could pin an old hash.
#[derive(Debug, Default)]
pub(crate) struct FileHashMemo {
    seen: BTreeMap<(PathBuf, u64, i64), String>,
}

impl FileHashMemo {
    pub(crate) fn hash(&mut self, path: &Path) -> Result<String> {
        let meta = std::fs::metadata(path)
            .with_context(|| format!("stat {} for cache key", path.display()))?;
        let mtime = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map_or(0, |d| d.as_secs() as i64);
        let memo_key = (path.to_path_buf(), meta.len(), mtime);
        if let Some(found) = self.seen.get(&memo_key) {
            return Ok(found.clone());
        }
        let mut hasher = Sha256::new();
        super::hash_file_into(path, &mut hasher)?;
        let digest = format!("{:x}", hasher.finalize());
        self.seen.insert(memo_key, digest.clone());
        Ok(digest)
    }
}

/// A stored measurement plus the provenance needed to police the admission rules.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CacheEntry {
    /// The key, stored verbatim so an entry is self-describing under inspection
    /// and a digest collision cannot silently serve the wrong row.
    key: serde_json::Value,
    row: serde_json::Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    witness_text: Option<String>,
}

/// One measurement as the cache stores and returns it.
///
/// A `sat` row's product is its verdict AND its counterexample, and the banked
/// `witness` record is a path into the bank that measured it — a hit cannot
/// simply copy that record, because it names a file THIS bank does not contain.
/// The witness TEXT therefore rides along, so a served `sat` row re-banks the
/// real bytes here and stays as replayable as a measured one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CachedMeasurement {
    /// The whole `SweepRow`, including its embedded child flight record.
    pub(crate) row: serde_json::Value,
    pub(crate) witness_text: Option<String>,
}

/// Why a lookup did not produce a hit. Recorded so a sweep can never claim a
/// cache is working when it is silently refusing everything.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MissReason {
    NotPresent,
    /// Entry was unreadable or malformed; treated as absent.
    Unusable,
    /// Rule 1: a `timeout` measured under a smaller budget. Reachable only via a
    /// tampered row body or a future key that drops `budget_secs`; see the
    /// module docs for exactly how much of this rule is live as wired.
    TimeoutUnderSmallerBudget,
    /// Rule 2: measured while the machine was heavily loaded.
    Contended,
}

/// Verdict cache rooted at a directory.
#[derive(Debug)]
pub(crate) struct VerdictCache {
    root: PathBuf,
    mode: CacheMode,
    cores: f64,
}

impl VerdictCache {
    pub(crate) fn new(root: PathBuf, mode: CacheMode) -> Self {
        let cores = std::thread::available_parallelism().map_or(1.0, |n| n.get() as f64);
        Self { root, mode, cores }
    }

    fn entry_path(&self, digest: &str) -> PathBuf {
        // Two-level fan-out keeps directories small on a big matrix.
        self.root.join(&digest[0..2]).join(format!("{digest}.json"))
    }

    /// Look up a measured row. Any problem is a miss.
    pub(crate) fn get(&self, key: &CacheKey) -> std::result::Result<CachedMeasurement, MissReason> {
        if !self.mode.reads() {
            return Err(MissReason::NotPresent);
        }
        let Ok(digest) = key.digest() else {
            return Err(MissReason::Unusable);
        };
        let path = self.entry_path(&digest);
        let Ok(bytes) = std::fs::read(&path) else {
            return Err(MissReason::NotPresent);
        };
        let Ok(entry) = serde_json::from_slice::<CacheEntry>(&bytes) else {
            return Err(MissReason::Unusable);
        };
        // A digest is not a proof. Compare the stored key to the requested one so
        // a truncated/edited/colliding entry cannot serve a different row.
        let Ok(requested) = serde_json::to_value(key) else {
            return Err(MissReason::Unusable);
        };
        if entry.key != requested {
            return Err(MissReason::Unusable);
        }
        if let Some(reason) = admission_refusal(&entry.row, key.budget_secs, self.cores) {
            return Err(reason);
        }
        Ok(CachedMeasurement {
            row: entry.row,
            witness_text: entry.witness_text,
        })
    }

    /// Record a fresh measurement. Storage failures are non-fatal: a sweep must
    /// never fail because a cache could not be written.
    pub(crate) fn put(&self, key: &CacheKey, measurement: &CachedMeasurement) {
        if !self.mode.writes() {
            return;
        }
        let Ok(digest) = key.digest() else { return };
        // Never persist a row the admission rules would refuse to serve; storing
        // it would just burn disk and invite a future relaxation to leak it.
        if admission_refusal(&measurement.row, key.budget_secs, self.cores).is_some() {
            return;
        }
        let Ok(stored_key) = serde_json::to_value(key) else {
            return;
        };
        let entry = CacheEntry {
            key: stored_key,
            row: measurement.row.clone(),
            witness_text: measurement.witness_text.clone(),
        };
        let path = self.entry_path(&digest);
        if let Some(parent) = path.parent() {
            if std::fs::create_dir_all(parent).is_err() {
                return;
            }
        }
        // Write-then-rename so a concurrent reader never sees a half-written
        // entry: parallel workers share one cache root by design.
        let temp = path.with_extension(format!("tmp.{}", std::process::id()));
        if serde_json::to_vec(&entry)
            .ok()
            .and_then(|bytes| std::fs::write(&temp, bytes).ok())
            .is_some()
        {
            let _ = std::fs::rename(&temp, &path);
        }
    }
}

/// The two non-negotiable admission rules. `None` means the row may be served.
fn admission_refusal(
    row: &serde_json::Value,
    requested_budget: u64,
    cores: f64,
) -> Option<MissReason> {
    // RULE 1. A `timeout` is only evidence at the budget it was measured at, and
    // only downward. Serving a 30 s timeout as a 100 s result manufactures a
    // verdict that was never observed.
    let verdict = row.get("verdict").and_then(serde_json::Value::as_str);
    if verdict == Some("timeout") {
        let measured = row
            .get("budget_secs")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        if measured < requested_budget {
            return Some(MissReason::TimeoutUnderSmallerBudget);
        }
    }
    // RULE 2. A row begun under heavy load measured the machine's queue, not the
    // verifier. `load_avg_at_begin` is the 1/5/15-minute triple; the 1-minute
    // figure is the one that reflects what was actually running.
    let load = row
        .get("flight")
        .and_then(|flight| flight.get("load_avg_at_begin"))
        .and_then(serde_json::Value::as_array)
        .and_then(|values| values.first())
        .and_then(serde_json::Value::as_f64);
    if let Some(one_minute) = load {
        if one_minute > cores {
            return Some(MissReason::Contended);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn key_with(budget: u64, arm: &[(&str, &str)]) -> CacheKey {
        key_in(budget, arm, BTreeMap::new())
    }

    fn key_in(
        budget: u64,
        arm: &[(&str, &str)],
        ambient_env: BTreeMap<String, String>,
    ) -> CacheKey {
        key_for_category("demo_a", budget, arm, ambient_env)
    }

    fn key_for_category(
        category: &str,
        budget: u64,
        arm: &[(&str, &str)],
        ambient_env: BTreeMap<String, String>,
    ) -> CacheKey {
        let mut arm: Vec<(String, String)> = arm
            .iter()
            .map(|(a, b)| ((*a).to_string(), (*b).to_string()))
            .collect();
        arm.sort();
        CacheKey {
            v: 3,
            exe_sha256: "exe".into(),
            build_provenance: "sealed".into(),
            configs_sha256: "cfg".into(),
            category: category.to_string(),
            onnx_sha256: "onnx".into(),
            vnnlib_sha256: "vnnlib".into(),
            budget_secs: budget,
            capped_from: None,
            arm,
            ambient_env,
            compute_backend: "cuda".into(),
            host: "gb10".into(),
        }
    }

    fn row(verdict: &str, budget: u64, load: Option<f64>) -> serde_json::Value {
        let mut row = json!({"verdict": verdict, "budget_secs": budget});
        if let Some(load) = load {
            row["flight"] = json!({"load_avg_at_begin": [load, load, load]});
        }
        row
    }

    fn measured(row: serde_json::Value) -> CachedMeasurement {
        CachedMeasurement {
            row,
            witness_text: None,
        }
    }

    #[test]
    fn arm_order_does_not_change_the_key() {
        // [(A,1),(B,0)] and [(B,0),(A,1)] are the same arm; if they hashed
        // differently the cache would miss every time the caller reordered.
        let a = key_with(100, &[("NY_A", "1"), ("NY_B", "0")])
            .digest()
            .unwrap();
        let b = key_with(100, &[("NY_B", "0"), ("NY_A", "1")])
            .digest()
            .unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn a_different_arm_is_a_different_key() {
        let a = key_with(100, &[("NY_A", "1")]).digest().unwrap();
        let b = key_with(100, &[("NY_A", "0")]).digest().unwrap();
        assert_ne!(a, b, "arm must be part of the identity of a verdict");
    }

    /// The defect this field closes: `run_instance` only ADDS to the child
    /// environment, so an `NY_*` exported into the parent shell reaches every
    /// row without ever appearing in `arm`. Keyed without the ambient set, the
    /// two runs below are one entry, and the second is served the first's
    /// verdicts — a wrong measurement served with full confidence.
    #[test]
    fn an_exported_ambient_lever_is_a_different_key() {
        let clean = key_with(100, &[]).digest().unwrap();
        let exported = key_in(
            100,
            &[],
            BTreeMap::from([(
                "NY_ROOT_COMPREHENSIVE_GPU_INTERM_CROWN".to_string(),
                "1".to_string(),
            )]),
        )
        .digest()
        .unwrap();
        assert_ne!(
            clean, exported,
            "a lever exported into the parent shell changes what is measured"
        );
    }

    /// The defect this field closes. The category is an INPUT to the child (it
    /// selects `<configs>/vnncomp*/{category}.yaml` and gates category-fenced
    /// routes), and `configs_sha256` digests the whole tree, so it cannot say
    /// which preset applied. Two categories over byte-identical model and
    /// property files therefore hashed identically, and a single cold
    /// `--cache read-write` sweep across both printed `1 served, 1 measured`
    /// with ONE on-disk entry for two rows — the second category was handed a
    /// verdict measured under the first category's preset.
    #[test]
    fn two_categories_over_identical_bytes_do_not_share_an_entry() {
        let a = key_for_category("demo_a", 100, &[], BTreeMap::new());
        let b = key_for_category("demo_b", 100, &[], BTreeMap::new());
        // Everything a content address can see about the instance is identical.
        assert_eq!(a.onnx_sha256, b.onnx_sha256);
        assert_eq!(a.vnnlib_sha256, b.vnnlib_sha256);
        assert_eq!(a.configs_sha256, b.configs_sha256);
        assert_ne!(
            a.digest().unwrap(),
            b.digest().unwrap(),
            "the category selects the preset the child loads, so it is part of \
             the identity of a verdict"
        );

        // And end to end: an entry stored under one category must not be served
        // to the other.
        let dir = tempfile::tempdir().unwrap();
        let cache = VerdictCache::new(dir.path().to_path_buf(), CacheMode::ReadWrite);
        cache.put(&a, &measured(row("unsat", 100, Some(0.2))));
        assert!(
            cache.get(&a).is_ok(),
            "the category it was measured in hits"
        );
        assert_eq!(
            cache.get(&b),
            Err(MissReason::NotPresent),
            "a different category must MEASURE, never inherit"
        );
    }

    #[test]
    fn a_different_budget_is_a_different_key() {
        assert_ne!(
            key_with(100, &[]).digest().unwrap(),
            key_with(30, &[]).digest().unwrap()
        );
    }

    #[test]
    fn rule_one_refuses_a_timeout_measured_at_a_smaller_budget() {
        // The class of fake result this rule exists for: a 30s timeout says
        // nothing about the same row at 100s.
        assert_eq!(
            admission_refusal(&row("timeout", 30, None), 100, 64.0),
            Some(MissReason::TimeoutUnderSmallerBudget)
        );
    }

    #[test]
    fn rule_one_allows_a_timeout_measured_at_the_same_or_larger_budget() {
        assert_eq!(
            admission_refusal(&row("timeout", 100, None), 100, 64.0),
            None
        );
        // A timeout at 200s does imply a timeout at 100s.
        assert_eq!(
            admission_refusal(&row("timeout", 200, None), 100, 64.0),
            None
        );
    }

    #[test]
    fn rule_one_does_not_touch_decided_verdicts() {
        // unsat/sat are budget-monotone in the safe direction: solving at 30s
        // means solving at 100s, so a smaller measured budget is fine.
        for verdict in ["unsat", "sat"] {
            assert_eq!(admission_refusal(&row(verdict, 30, None), 100, 64.0), None);
        }
    }

    #[test]
    fn rule_two_refuses_a_contended_row() {
        assert_eq!(
            admission_refusal(&row("unsat", 100, Some(90.0)), 100, 20.0),
            Some(MissReason::Contended)
        );
    }

    #[test]
    fn rule_two_allows_a_quiet_row_and_a_row_without_a_flight_record() {
        assert_eq!(
            admission_refusal(&row("unsat", 100, Some(0.4)), 100, 20.0),
            None
        );
        assert_eq!(admission_refusal(&row("unsat", 100, None), 100, 20.0), None);
    }

    #[test]
    fn round_trip_hit_then_refusal_after_the_rules_bite() {
        let dir = tempfile::tempdir().unwrap();
        let cache = VerdictCache::new(dir.path().to_path_buf(), CacheMode::ReadWrite);
        let key = key_with(100, &[("NY_A", "1")]);
        cache.put(&key, &measured(row("unsat", 100, Some(0.2))));
        assert!(cache.get(&key).is_ok(), "a clean row must round-trip");

        // A timeout stored at 30s must not be served for a 100s request. Note
        // WHICH miss this is: `budget_secs` is in the key, so the 100s request
        // does not even address the 30s entry. Rule 1 is not what refuses it.
        let small = key_with(30, &[]);
        cache.put(&small, &measured(row("timeout", 30, Some(0.2))));
        let large = key_with(100, &[]);
        assert_eq!(cache.get(&large), Err(MissReason::NotPresent));
    }

    /// Pins the honest scope of rule 1 on the WIRED path (see the module docs).
    /// Key equality already separates budgets, and `put` writes a row whose
    /// `budget_secs` came from the same plan entry as the key's — so the only
    /// live way to reach the rule today is an entry whose ROW BODY disagrees
    /// with its own key, which the key comparison does not cover because only
    /// the stored KEY is compared.
    #[test]
    fn rule_one_is_reachable_through_a_tampered_row_body() {
        let dir = tempfile::tempdir().unwrap();
        let cache = VerdictCache::new(dir.path().to_path_buf(), CacheMode::ReadWrite);
        let key = key_with(100, &[]);
        cache.put(&key, &measured(row("timeout", 100, Some(0.2))));
        assert!(
            cache.get(&key).is_ok(),
            "an honest 100s timeout round-trips"
        );

        let path = cache.entry_path(&key.digest().unwrap());
        let mut entry: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        entry["row"]["budget_secs"] = json!(30);
        std::fs::write(&path, serde_json::to_vec(&entry).unwrap()).unwrap();

        assert_eq!(
            cache.get(&key),
            Err(MissReason::TimeoutUnderSmallerBudget),
            "a row claiming a smaller budget than its key must be refused, not served"
        );
    }

    /// A served `sat` row must be able to re-bank its counterexample: the stored
    /// `witness` RECORD points into the bank that measured it, so only the text
    /// makes a hit as replayable as a fresh measurement.
    #[test]
    fn a_sat_hit_carries_the_witness_text_back() {
        let dir = tempfile::tempdir().unwrap();
        let cache = VerdictCache::new(dir.path().to_path_buf(), CacheMode::ReadWrite);
        let key = key_with(100, &[]);
        let stored = CachedMeasurement {
            row: row("sat", 100, Some(0.2)),
            witness_text: Some("(X_0 0.5)\n(Y_0 1.0)\n".into()),
        };
        cache.put(&key, &stored);
        assert_eq!(cache.get(&key), Ok(stored));
    }

    #[test]
    fn a_contended_row_is_never_even_stored() {
        let dir = tempfile::tempdir().unwrap();
        let cache = VerdictCache::new(dir.path().to_path_buf(), CacheMode::ReadWrite);
        let key = key_with(100, &[]);
        cache.put(&key, &measured(row("unsat", 100, Some(500.0))));
        assert_eq!(cache.get(&key), Err(MissReason::NotPresent));
    }

    #[test]
    fn read_mode_never_writes_and_off_mode_never_reads() {
        let dir = tempfile::tempdir().unwrap();
        let key = key_with(100, &[]);

        let read_only = VerdictCache::new(dir.path().to_path_buf(), CacheMode::Read);
        read_only.put(&key, &measured(row("unsat", 100, Some(0.1))));
        assert_eq!(read_only.get(&key), Err(MissReason::NotPresent));

        let rw = VerdictCache::new(dir.path().to_path_buf(), CacheMode::ReadWrite);
        rw.put(&key, &measured(row("unsat", 100, Some(0.1))));
        assert!(rw.get(&key).is_ok());

        let off = VerdictCache::new(dir.path().to_path_buf(), CacheMode::Off);
        assert_eq!(off.get(&key), Err(MissReason::NotPresent));
    }

    #[test]
    fn a_tampered_entry_is_a_miss_not_a_wrong_hit() {
        // The stored key is compared field-by-field, so a corrupted or colliding
        // entry degrades to a fresh measurement instead of serving another row.
        let dir = tempfile::tempdir().unwrap();
        let cache = VerdictCache::new(dir.path().to_path_buf(), CacheMode::ReadWrite);
        let key = key_with(100, &[]);
        cache.put(&key, &measured(row("unsat", 100, Some(0.1))));

        let path = cache.entry_path(&key.digest().unwrap());
        let mut entry: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        entry["key"]["host"] = json!("a-different-machine");
        std::fs::write(&path, serde_json::to_vec(&entry).unwrap()).unwrap();

        assert_eq!(cache.get(&key), Err(MissReason::Unusable));
    }

    #[test]
    fn malformed_entries_degrade_to_a_miss() {
        let dir = tempfile::tempdir().unwrap();
        let cache = VerdictCache::new(dir.path().to_path_buf(), CacheMode::ReadWrite);
        let key = key_with(100, &[]);
        let path = cache.entry_path(&key.digest().unwrap());
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"not json at all").unwrap();
        assert_eq!(cache.get(&key), Err(MissReason::Unusable));
    }
}
