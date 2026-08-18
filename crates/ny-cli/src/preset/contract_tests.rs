// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The test `docs/CONV_CROWN_WALL_DESIGN_2026-07-27.md` §S3 asks for: every
//! field in every shipped preset under `configs/` is honoured by the current
//! engine.
//!
//! That property does NOT hold at HEAD: eight fields across the shipped presets
//! are requested and dropped. Asserting it outright would leave a permanently
//! red test, which is a test nobody reads, so the remaining mismatches are
//! enumerated in [`ACCEPTED_DEBT`] with a reason and a citation and the test is
//! green against that LEDGER. Both directions are locked, which is what makes
//! the ledger worth more than a red test:
//!
//!   * a field that becomes unhonoured and is NOT on the list fails the test —
//!     that is the regression this whole module exists to prevent;
//!   * a field that gets FIXED also fails the test, because its ledger entry is
//!     then unexercised. The list can therefore only shrink, never rot.
//!
//! Coverage caveat, stated once and honestly: "honoured" here means "not
//! reported by `contract::validate_preset`", and that registry is
//! hand-maintained. A silently-ignored field nobody has noticed is absent from
//! both the registry and this ledger, so a green run means "no KNOWN mismatch
//! is unaccounted for", never "every key is wired up". The independent defences
//! are `load_preset`'s unknown-key rejection and the per-field propagation
//! assertions in `preset::tests` / `preset::vnncomp_preset_tests`.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use super::contract::{validate_preset, ContractSeverity, EngineContract};
use super::load_preset;

/// Preset fields that shipped presets request and this build does not honour.
///
/// Each entry is `(dotted field, why it is still here)`. Deleting an entry is
/// the last step of fixing the field; adding one requires the same argument the
/// registry entry carries — in particular, which DIRECTION the omission fails
/// in, because only "looser bounds / fewer proofs" is acceptable debt.
const ACCEPTED_DEBT: &[(&str, &str)] = &[
    (
        "general.loss_reduction_func",
        "alpha-beta-CROWN attack-loss reduction has no ny counterpart; attack-direction \
         only, so it cannot reach a verdict path.",
    ),
    (
        "attack.pgd_order",
        "'middle' only. ENABLEMENT plus the 'before' and deferred 'after' placements are \
         honoured end-to-end; the interleaved 'middle' schedule is not, so those presets \
         run PGD on the upfront schedule through their explicit ny_pgd_order_compat \
         contract (preset::apply::apply_attack_preset, 'middle' arm).",
    ),
    (
        "attack.attack_tolerance",
        "acceptance is the zero-tolerance trusted-oracle gate; ignoring a requested slack \
         is strictly stricter than honouring it.",
    ),
    (
        "attack.surrogate_sign_gradient",
        "read ONLY by the graph disjunctive PGD loop. traffic_signs_recognition_2023 arms \
         it for a binarized Sign network — the right technique — but that category's \
         budget goes to the wrapper's upfront DLR-APGD lane, which builds no PgdConfig, so \
         the loop is never entered (MEASURED: 0 entries at the full 480s budget). Losing an \
         attack technique costs `sat` rows and cannot admit a false one. Routing it into \
         the upfront lane was measured to convert 0 rows, so this is reported rather than \
         claimed fixed — docs/CANDIDATE_BRANCH_FINDINGS_2026-08-13.md §3.",
    ),
    (
        "attack.attack_mode",
        "the GAMA half only, and only off the graph-PGD route — same measured lane gap as \
         attack.surrogate_sign_gradient above (0 'gama' occurrences at 480s). The \
         diversified-restart half of diversed_GAMA_PGD IS honoured. Attack-direction only.",
    ),
    (
        "bab.clip.interm_domain",
        "SEQUENTIAL engine only — the graph engine honours it. Sequential \
         IntermediateLinearBounds are layer-output-relative and lack certified \
         input-relative provenance, so the tightening is SKIPPED and bounds stay looser \
         (ny_propagate::sequential_clip_interm_domain_supported).",
    ),
    (
        "bab.pruning_in_iteration",
        "declared on AlphaCrownConfig, read by nothing; running without in-iteration \
         pruning is more work and never a looser bound.",
    ),
    (
        "bab.branching.nonlinear_split.filter",
        "GenBaB candidate filtering is unimplemented; branching order and cost only.",
    ),
    (
        "bab.branching.nonlinear_split.filter_beta",
        "GenBaB candidate filtering is unimplemented; branching order and cost only.",
    ),
    (
        "bab.invprop.apply_output_constraints_to",
        "InvpropPreset is parsed for key compatibility but not applied; ny's INVPROP is \
         driven by the --invprop-* CLI flags. Omitting a tightener leaves bounds looser.",
    ),
];

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Every `configs/vnncomp*/**.yaml` preset, sorted for a stable report.
fn shipped_presets() -> Vec<PathBuf> {
    let configs_root = repo_root().join("configs");
    let mut presets = Vec::new();
    for entry in fs::read_dir(&configs_root).expect("list configs/") {
        let dir = entry.expect("read configs/ entry").path();
        let is_vnncomp_dir = dir
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with("vnncomp"));
        if !is_vnncomp_dir || !dir.is_dir() {
            continue;
        }
        for file in fs::read_dir(&dir).expect("list preset dir") {
            let path = file.expect("read preset entry").path();
            if path.extension().and_then(|extension| extension.to_str()) == Some("yaml") {
                presets.push(path);
            }
        }
    }
    presets.sort();
    presets
}

/// Display path relative to the repo root, so failures name `configs/...`.
///
/// Separators are normalized to `/` because this string is a KEY, not just
/// display text: the debt tables below are written with POSIX literals like
/// `configs/vnncomp25/relusplitter.yaml`, and `Path::display` emits the host
/// separator — so on Windows every lookup missed and the S3 debt test reported
/// that a still-present finding had been fixed.
fn short(path: &Path) -> String {
    path.strip_prefix(repo_root())
        .unwrap_or(path)
        .display()
        .to_string()
        .replace('\\', "/")
}

/// field -> presets that request it and do not get it.
fn observed_debt() -> BTreeMap<&'static str, Vec<String>> {
    let contract = EngineContract::current();
    let mut observed: BTreeMap<&'static str, Vec<String>> = BTreeMap::new();
    let presets = shipped_presets();
    assert!(
        presets.len() > 20,
        "expected every shipped vnncomp preset, found only {} — did configs/ move?",
        presets.len()
    );
    for path in presets {
        let preset = load_preset(&path)
            .unwrap_or_else(|error| panic!("shipped preset {} must load: {error:#}", short(&path)));
        for entry in validate_preset(&preset, &contract) {
            observed.entry(entry.field).or_default().push(short(&path));
        }
    }
    observed
}

/// THE §S3 TEST. Every field of every shipped preset is honoured, except the
/// enumerated debt.
#[test]
fn every_shipped_preset_field_is_honoured_or_accepted_debt() {
    let observed = observed_debt();
    let observed_fields: BTreeSet<&str> = observed.keys().copied().collect();
    let accepted_fields: BTreeSet<&str> = ACCEPTED_DEBT.iter().map(|(field, _)| *field).collect();

    let unaccounted: Vec<String> = observed_fields
        .difference(&accepted_fields)
        .map(|field| {
            let requesters = observed
                .get(*field)
                .map_or_else(String::new, |presets| presets.join(", "));
            format!("  {field} — requested by {requesters}")
        })
        .collect();
    assert!(
        unaccounted.is_empty(),
        "shipped presets request field(s) this build does not honour, with no ledger \
         entry:\n{}\nEither wire the field up, or add it to ACCEPTED_DEBT with the \
         DIRECTION its omission fails in (looser bounds are acceptable debt; anything \
         that could narrow a bound is not — fix that instead).",
        unaccounted.join("\n")
    );

    let stale: Vec<&str> = accepted_fields
        .difference(&observed_fields)
        .copied()
        .collect();
    assert!(
        stale.is_empty(),
        "ACCEPTED_DEBT entries no longer reproduce: {stale:?}. Either the field was \
         implemented (delete the entry — the ledger only shrinks) or the last preset \
         setting it was removed (delete it too).",
    );
}

/// The default strict mode refuses to start on a VERDICT-AFFECTING field whose
/// failure direction nobody has argued. No shipped preset may be in that state,
/// or the check would abort a benchmark instead of reporting it.
#[test]
fn no_shipped_preset_hits_an_unargued_verdict_affecting_mismatch() {
    let contract = EngineContract::current();
    let mut offenders = Vec::new();
    for path in shipped_presets() {
        let preset = load_preset(&path).expect("shipped preset loads");
        for entry in validate_preset(&preset, &contract) {
            if entry.severity == ContractSeverity::VerdictAffecting && entry.accepted.is_none() {
                offenders.push(format!("{}: {}", short(&path), entry.field));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "these shipped presets would ABORT at load under the default strict mode:\n{}",
        offenders.join("\n")
    );
}

/// The remaining mismatches from the design document, re-checked at HEAD.
///
/// `general.device` is no longer among them: WGPU now reaches the typed public
/// proof constructor and either retains a qualified device or emits a runtime
/// CPU-fallback receipt. `phase_budget.upfront_pgd_fraction`, recorded there as
/// having "no runtime consumer at all", is also absent: it reaches
/// `PhaseBudgetLedger::upfront_pgd_deadline`, which both the graph and the
/// sequential upfront-PGD lanes consult. It is deliberately absent from the
/// registry rather than checked, because inventing a check for a field the
/// engine does honour would be a false report in the other direction.
#[test]
fn design_s3_findings_still_reproduce_at_head() {
    let observed = observed_debt();

    for (field, preset) in [
        ("attack.pgd_order", "configs/vnncomp25/relusplitter.yaml"),
        (
            "bab.clip.interm_domain",
            "configs/vnncomp25/relusplitter.yaml",
        ),
        ("attack.pgd_order", "configs/vnncomp25/soundnessbench.yaml"),
    ] {
        let presets = observed.get(field).unwrap_or_else(|| {
            panic!("design S3 named {field} as unhonoured; it is no longer reported")
        });
        assert!(
            presets.iter().any(|name| name.as_str() == preset),
            "{field} should still be unhonoured for {preset}; reported for {presets:?}"
        );
    }
}

/// Every ledger entry carries a real explanation. A one-word "TODO" entry would
/// turn the allow-list back into the silence it replaced.
#[test]
fn accepted_debt_entries_carry_a_reason() {
    for (field, why) in ACCEPTED_DEBT {
        assert!(
            why.len() > 40,
            "ACCEPTED_DEBT entry for {field} needs a real reason, got {why:?}"
        );
    }
    let unique: BTreeSet<&str> = ACCEPTED_DEBT.iter().map(|(field, _)| *field).collect();
    assert_eq!(
        unique.len(),
        ACCEPTED_DEBT.len(),
        "duplicate ACCEPTED_DEBT field entries"
    );
}
