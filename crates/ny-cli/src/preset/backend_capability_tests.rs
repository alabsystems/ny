// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! STATIC capability guard: every shipped preset's declared `general.device`
//! must be a device this binary actually executes proofs on, or be covered by a
//! dated waiver in `configs/backend_capability_waivers.yaml`.
//!
//! WHY: 1ede1d30 quarantined the WGPU proof adapter. Sixteen shipped presets
//! declare `device: wgpu`, and from that commit on every one of them silently
//! ran the CPU verifier behind a lone `tracing::warn` that the scored path never
//! prints (RUST_LOG is ignored there). Two banked categories zeroed unnoticed
//! for months — including soundnessbench, the repo's MANDATED soundness gate.
//! The quarantine was right; the missing artefact was any record that its
//! capability cost had been looked at.
//!
//! These tests need no benchmark data, no GPU and no verification run: they are
//! pure config-vs-code coherence and finish in milliseconds, so a future
//! fail-closed gate cannot land silently.

use super::load_preset;
use crate::commands::backend::{apply_preset_device, honour_requested_backend};
use crate::BackendArg;
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// Device tokens `apply_preset_device` recognises. Anything else falls back to
/// CPU behind a `warn!`, which is the same silent-capability-loss shape as the
/// quarantine — so an unrecognised token is a hard failure here.
const RECOGNISED_DEVICES: [&str; 2] = ["cpu", "wgpu"];

#[derive(Debug, Deserialize)]
struct WaiverFile {
    overrides: Vec<Waiver>,
}

#[derive(Debug, Deserialize)]
struct Waiver {
    declared: String,
    effective: String,
    since: String,
    commit: String,
    reason: String,
    measured_cost: String,
    presets: Vec<String>,
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Every shipped preset YAML under `configs/vnncomp*/`, repo-relative, sorted.
///
/// Deliberately includes the `*_canary` / `*_gpu_bab` experiment presets: a
/// canary whose declared device is silently swapped measures the wrong thing,
/// which is how an A/B lane reaches a false conclusion.
fn shipped_presets() -> Vec<String> {
    let configs = repo_root().join("configs");
    let mut out = Vec::new();
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(&configs)
        .expect("configs/ must exist")
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_dir()
                && path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("vnncomp"))
        })
        .collect();
    dirs.sort();
    for dir in dirs {
        let mut yamls: Vec<PathBuf> = std::fs::read_dir(&dir)
            .expect("preset dir readable")
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| path.extension().and_then(|e| e.to_str()) == Some("yaml"))
            .collect();
        yamls.sort();
        for yaml in yamls {
            let rel = yaml
                .strip_prefix(repo_root())
                .expect("preset under repo root")
                .to_string_lossy()
                // Normalise the `crates/ny-cli/../..` join so the paths match
                // the waiver file's repo-relative spelling on every platform.
                .replace('\\', "/");
            out.push(rel);
        }
    }
    assert!(
        out.len() >= 40,
        "expected the shipped preset inventory (>=40 files), found {}",
        out.len()
    );
    out
}

/// (declared device, effective device) for every preset that pins one.
fn declared_devices() -> BTreeMap<String, (String, String)> {
    let root = repo_root();
    let mut map = BTreeMap::new();
    for rel in shipped_presets() {
        let preset = load_preset(&root.join(&rel))
            .unwrap_or_else(|err| panic!("shipped preset {rel} must parse: {err}"));
        let Some(device) = preset.general.device.as_deref() else {
            continue;
        };
        assert!(
            RECOGNISED_DEVICES.contains(&device),
            "{rel} declares general.device: '{device}', which apply_preset_device does not \
             recognise — it would silently fall back to CPU behind a warn!. Add the device to \
             the resolver or fix the preset; do not leave a preset asking for a device that \
             does not exist."
        );
        let requested = apply_preset_device(BackendArg::Cpu, false, Some(device));
        let honoured = honour_requested_backend(requested);
        map.insert(rel, (device.to_string(), honoured.effective.to_string()));
    }
    map
}

fn load_waivers() -> Vec<Waiver> {
    let path = repo_root().join("configs/backend_capability_waivers.yaml");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|err| {
        panic!(
            "the backend capability waiver registry must exist at {}: {err}",
            path.display()
        )
    });
    let parsed: WaiverFile = serde_yaml::from_str(&text)
        .unwrap_or_else(|err| panic!("{} must be valid YAML: {err}", path.display()));
    for waiver in &parsed.overrides {
        assert!(
            !waiver.reason.trim().is_empty(),
            "waiver {} -> {} must state a reason",
            waiver.declared,
            waiver.effective
        );
        // Provenance: without the commit and date, nobody can tell whether the
        // ledgers a waiver covers predate the substitution.
        assert!(
            !waiver.since.trim().is_empty() && !waiver.commit.trim().is_empty(),
            "waiver {} -> {} must record the commit and date that introduced it",
            waiver.declared,
            waiver.effective
        );
        assert!(
            RECOGNISED_DEVICES.contains(&waiver.declared.as_str())
                && RECOGNISED_DEVICES.contains(&waiver.effective.as_str()),
            "waiver {} -> {} names a device the resolver does not know",
            waiver.declared,
            waiver.effective
        );
        // An unmeasured capability cost is the exact defect this registry
        // exists to prevent, so refuse the placeholder outright.
        let cost = waiver.measured_cost.trim();
        assert!(
            !cost.is_empty() && !cost.eq_ignore_ascii_case("unmeasured"),
            "waiver {} -> {} must record the MEASURED capability cost; 'unmeasured' is not an \
             acceptable value",
            waiver.declared,
            waiver.effective
        );
    }
    parsed.overrides
}

/// THE GUARD. A preset asking for a backend this binary refuses to run is
/// either a loud failure here or a recorded waiver — never a silent WARN.
#[test]
fn every_preset_declared_device_is_honoured_or_waived() {
    let devices = declared_devices();
    let waivers = load_waivers();

    let mut waived: BTreeMap<&str, (&str, &str)> = BTreeMap::new();
    for waiver in &waivers {
        for preset in &waiver.presets {
            let previous = waived.insert(
                preset.as_str(),
                (waiver.declared.as_str(), waiver.effective.as_str()),
            );
            assert!(
                previous.is_none(),
                "{preset} appears in two backend capability waivers; one preset has one \
                 declared device, so exactly one waiver can apply"
            );
        }
    }

    let mut undeclared = Vec::new();
    let mut mismatched = Vec::new();
    for (preset, (declared, effective)) in &devices {
        if declared == effective {
            continue;
        }
        match waived.get(preset.as_str()) {
            None => undeclared.push(format!("{preset}: declares {declared}, runs {effective}")),
            Some((waived_declared, waived_effective)) => {
                if waived_declared != declared || waived_effective != effective {
                    mismatched.push(format!(
                        "{preset}: waiver says {waived_declared} -> {waived_effective}, runtime \
                         does {declared} -> {effective}"
                    ));
                }
            }
        }
    }

    let honoured: BTreeSet<&str> = devices
        .iter()
        .filter(|(_, (declared, effective))| declared == effective)
        .map(|(preset, _)| preset.as_str())
        .collect();
    let unknown: Vec<&str> = waived
        .keys()
        .copied()
        .filter(|preset| !devices.contains_key(*preset))
        .collect();
    let stale: Vec<&str> = waived
        .keys()
        .copied()
        .filter(|preset| honoured.contains(*preset))
        .collect();

    assert!(
        undeclared.is_empty(),
        "UNDECLARED BACKEND CAPABILITY COST — {} shipped preset(s) ask for a device this binary \
         does not execute proofs on, and no waiver records it. Either honour the device or add a \
         dated, MEASURED entry to configs/backend_capability_waivers.yaml:\n  {}",
        undeclared.len(),
        undeclared.join("\n  ")
    );
    assert!(
        mismatched.is_empty(),
        "BACKEND WAIVER OUT OF DATE — the recorded substitution no longer matches what the \
         binary does:\n  {}",
        mismatched.join("\n  ")
    );
    assert!(
        unknown.is_empty(),
        "BACKEND WAIVER REFERENCES A PRESET THAT NO LONGER DECLARES A DEVICE (renamed or \
         deleted?): {}",
        unknown.join(", ")
    );
    assert!(
        stale.is_empty(),
        "STALE BACKEND WAIVER — these presets' declared devices ARE honoured now, so the waiver \
         is a lie about current capability. Delete the entries (and re-sweep the ledgers the \
         waiver was covering): {}",
        stale.join(", ")
    );
}

/// Every recognized preset backend now has a public proof-construction route.
/// WGPU can still refuse at runtime, but that typed, host-specific outcome is
/// not a static preset capability gap and is emitted in each run receipt.
#[test]
fn every_recognised_preset_backend_reaches_runtime_qualification() {
    let devices = declared_devices();
    let gap: Vec<&str> = devices
        .iter()
        .filter(|(_, (declared, effective))| declared != effective)
        .map(|(preset, _)| preset.as_str())
        .collect();
    assert!(
        gap.is_empty(),
        "recognized preset backends must reach their public proof constructor; runtime WGPU \
         qualification failures belong in per-run receipts, not static waivers: {gap:?}"
    );
}

/// Static admission must not pre-empt the typed live WGPU qualification.
#[test]
fn wgpu_backend_is_admitted_to_runtime_qualification() {
    let honoured = honour_requested_backend(BackendArg::Wgpu);
    assert_eq!(honoured.effective, BackendArg::Wgpu);
    assert!(honoured.override_reason.is_none());
    let cpu = honour_requested_backend(BackendArg::Cpu);
    assert_eq!(cpu.effective, BackendArg::Cpu);
    assert!(
        cpu.override_reason.is_none(),
        "an honoured backend must not report an override"
    );
}
