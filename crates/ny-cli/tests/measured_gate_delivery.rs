// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Guard: a `NY_*` lever that documents a MEASURED result AND says it is OFF by
//! default must declare how that result reaches the scored path.
//!
//! WHY THIS EXISTS. The workspace reads several hundred distinct `NY_*` variables,
//! but the scored entry point exports exactly ONE of them:
//! `vnncomp_scripts/run_instance.sh` sets `NY_UPFRONT_ATTACK=1` for `safenlp_2024`
//! and nothing else. Everything else runs at its COMPILED DEFAULT during scoring.
//! So an `NY_*` gate that is default-off is, in competition, dead code — no matter
//! what its A/B measured. Several investigations have ended with "the lever is
//! already implemented and measured, it just cannot fire", and each time the
//! mechanism was the same: the measurement was written into a doc comment and the
//! delivery step was never taken.
//!
//! WHAT IS CHECKED. Not all 600+ variables — a blanket check would be noise and
//! would be deleted. Only the narrow intersection that carries the hazard:
//!
//!   * the site is a `std::env::var{,_os}("NY_…")` read in NON-test workspace code;
//!   * the gate's OWN documentation (the comment block attached to the read, plus
//!     the doc comment of the enclosing `fn`/`const`/`static`) contains the
//!     all-caps token `MEASURED` — this repository's deliberate convention for
//!     "this is an observed number, not a guess"; and
//!   * that same documentation says the lever is off unless someone turns it on
//!     (`default-off` / `default off` / `opt-in`, any case).
//!
//! Today that is 3 sites out of 796 reads (`NY_JOINT_MARGIN_LP`,
//! `NY_REL_WHOLE_MIP_OBBT`, `NY_BRANCH_KFSB_CHILDSIM`) — run this test with
//! `--nocapture` to print the current population. The module-level `//!` header is
//! deliberately NOT part of the window: one shared header (`kfsb_multi.rs`) would
//! otherwise tag 15 unrelated gates in the same file, and `MEASURED` inside a path
//! like `docs/MEASURED_KFSB_GATES.md` is a filename, not a claim, so token
//! boundaries are required.
//!
//! WHAT SATISFIES IT. Either half of an honest disposition:
//!
//!   1. DELIVERY — some `NY_*` name in the gate's own documentation (its own, or a
//!      documented equivalent such as the `NY_BRANCH_KFSB_CHILDSIM` -> `NY_MO_KFSB`
//!      alias) appears in the typed preset schema (`crates/ny-cli/src/preset/*.rs`,
//!      non-test) or is exported by `vnncomp_scripts/run_instance.sh`. That is what
//!      a "typed preset key" means here: a per-benchmark YAML key can reach it.
//!   2. `DARK` — the all-caps marker already used across this workspace for a lever
//!      that is deliberately unreachable in competition (a research probe, a
//!      reproducibility harness, or a measurement that came back negative).
//!
//! LIMIT, STATED PLAINLY. `DARK` is an escape hatch a human writes, so this guard
//! forces a decision; it cannot make the decision. In particular it does NOT catch
//! `#root-cap-retry` (`NY_ROOT_CAP_RETRY`), which is already labelled
//! `// #root-cap-retry (DARK, NY_ROOT_CAP_RETRY=1).` at
//! `crates/ny-propagate/src/beta_crown/engine/graph/multi_objective/root.rs:1850`
//! while recovering a measured 186 s of forfeited budget (0 verdicts at 100 s;
//! `docs/WHY_BETA_CROWN_FORFEITS_205S_2026-08-05.md:335,727`). No text rule catches
//! a lever that is honestly labelled DARK. The smallest thing that would is a
//! REVIEWED LEDGER: one checked-in table of every `DARK` marker in `crates/`, with
//! a required "measured effect" and "why not delivered" column, and a guard
//! asserting the marker set and the ledger rows agree. That turns "is anything
//! measured-good sitting dark?" into a list a human reads once, instead of a
//! property prose can express.
//!
//! Do not relax the guard to silence a failure. The two honest fixes are: wire the
//! lever to a typed preset key, or write down that it is `DARK` and why.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// Floor on the number of `NY_*` read sites the scanner must still find. If the
/// read idiom changes and this scanner silently matches nothing, the guard would
/// pass vacuously forever; 796 sites exist today, so 400 is a wide margin that
/// still trips on a broken scanner.
const MIN_EXPECTED_READ_SITES: usize = 400;

/// Preset sources that constitute a typed delivery path.
const PRESET_DIR: &str = "crates/ny-cli/src/preset";
/// The scored entry point; the only script that exports an `NY_*` for scoring.
const RUN_INSTANCE: &str = "vnncomp_scripts/run_instance.sh";

fn workspace_root() -> PathBuf {
    // CARGO_MANIFEST_DIR is <root>/crates/ny-cli.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root is two levels above crates/ny-cli")
        .to_path_buf()
}

/// One `std::env::var{,_os}("NY_…")` read together with the documentation that
/// belongs to it.
#[derive(Debug)]
struct GateSite {
    file: String,
    line: usize,
    var: String,
    doc: String,
}

/// Rust files whose contents are test scaffolding rather than shipped behavior.
/// A doc comment in a test says nothing about what runs during scoring.
fn is_test_path(rel: &str) -> bool {
    let mut parts = rel.split('/').collect::<Vec<_>>();
    let base = parts.pop().unwrap_or_default();
    parts.contains(&"tests")
        || base == "tests.rs"
        || base.ends_with("_tests.rs")
        || base.starts_with("tests_")
}

/// Every `NY_[A-Z0-9_]+` token in `text`, on identifier boundaries.
fn ny_tokens(text: &str) -> BTreeSet<String> {
    let bytes = text.as_bytes();
    let mut out = BTreeSet::new();
    let mut i = 0usize;
    while i + 3 <= bytes.len() {
        // A UTF-8 continuation byte is >= 0x80, so a 3-byte ASCII match can only
        // start on a character boundary — the slices below are always valid.
        if &bytes[i..i + 3] == b"NY_"
            && (i == 0 || !(bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_'))
        {
            let mut j = i + 3;
            while j < bytes.len()
                && (bytes[j].is_ascii_uppercase() || bytes[j].is_ascii_digit() || bytes[j] == b'_')
            {
                j += 1;
            }
            if j > i + 3 {
                out.insert(text[i..j].to_string());
            }
            i = j;
        } else {
            i += 1;
        }
    }
    out
}

/// `needle` appears in `text` on identifier boundaries (so `DARK` does not match
/// `DARKNET`, and stripping is not needed for substrings of longer words).
fn contains_token(text: &str, needle: &str) -> bool {
    let bytes = text.as_bytes();
    let n = needle.as_bytes();
    if n.is_empty() || bytes.len() < n.len() {
        return false;
    }
    (0..=bytes.len() - n.len()).any(|i| {
        &bytes[i..i + n.len()] == n
            && (i == 0 || !(bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_'))
            && (i + n.len() == bytes.len()
                || !(bytes[i + n.len()].is_ascii_alphanumeric() || bytes[i + n.len()] == b'_'))
    })
}

/// Does this line start a `fn` / `const` / `static` item?
fn is_item_start(line: &str) -> bool {
    let mut s = line.trim_start();
    if let Some(rest) = s.strip_prefix("pub") {
        let rest = rest.trim_start();
        s = match rest.strip_prefix('(') {
            // `pub(crate)`, `pub(in crate::a::b)` — no nested parens in practice.
            Some(inner) => match inner.find(')') {
                Some(close) => inner[close + 1..].trim_start(),
                None => rest,
            },
            None => rest,
        };
    }
    for kw in ["async ", "unsafe "] {
        if let Some(rest) = s.strip_prefix(kw) {
            s = rest.trim_start();
        }
    }
    s.starts_with("fn ") || s.starts_with("const ") || s.starts_with("static ")
}

/// The documentation that belongs to the read on line `idx` (0-based): the
/// contiguous comment block immediately above it, plus the enclosing item's
/// signature and its own doc block. Deliberately excludes the file's `//!` header,
/// which is shared by every gate in the file and therefore attributes one gate's
/// measurement to all of its neighbours.
fn gate_doc(lines: &[&str], idx: usize) -> String {
    let mut out: Vec<&str> = Vec::new();

    let mut i = idx;
    let mut above: Vec<&str> = Vec::new();
    while i > 0 && lines[i - 1].trim_start().starts_with("//") {
        i -= 1;
        above.push(lines[i]);
    }
    above.reverse();
    out.extend(above);

    let mut j = idx;
    loop {
        if is_item_start(lines[j]) {
            out.push(lines[j]);
            let mut k = j;
            let mut doc: Vec<&str> = Vec::new();
            while k > 0 {
                let prev = lines[k - 1].trim_start();
                if prev.starts_with("//") || prev.starts_with("#[") {
                    k -= 1;
                    doc.push(lines[k]);
                } else {
                    break;
                }
            }
            doc.reverse();
            out.extend(doc);
            break;
        }
        if j == 0 {
            break;
        }
        j -= 1;
    }

    out.join("\n")
}

/// The `NY_*` names read by `std::env::var{,_os}` on this line.
fn env_reads_in_line(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut from = 0usize;
    while let Some(offset) = line[from..].find("\"NY_") {
        let quote = from + offset;
        let prefix = &line[..quote];
        if prefix.ends_with("env::var(") || prefix.ends_with("env::var_os(") {
            let rest = &line[quote + 1..];
            if let Some(end) = rest.find('"') {
                let name = &rest[..end];
                if name
                    .bytes()
                    .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'_')
                {
                    out.push(name.to_string());
                }
            }
        }
        from = quote + 4;
    }
    out
}

fn scan_source(rel: &str, text: &str) -> Vec<GateSite> {
    let lines: Vec<&str> = text.lines().collect();
    let mut out = Vec::new();
    for (idx, line) in lines.iter().enumerate() {
        for var in env_reads_in_line(line) {
            out.push(GateSite {
                file: rel.to_string(),
                line: idx + 1,
                var,
                doc: gate_doc(&lines, idx),
            });
        }
    }
    out
}

fn rust_sources(dir: &Path, root: &Path, out: &mut Vec<(String, PathBuf)>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if path.file_name().is_some_and(|n| n == "target") {
                continue;
            }
            rust_sources(&path, root, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            let rel = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");
            out.push((rel, path));
        }
    }
}

/// The gate's documentation claims an observed result. `NY_*` names are stripped
/// first so a variable literally called `NY_PATCHES_REENTRY_MEASURED` does not
/// count as a measurement claim.
fn claims_measured_result(doc: &str) -> bool {
    let mut stripped = doc.to_string();
    for token in ny_tokens(doc) {
        stripped = stripped.replace(&token, " ");
    }
    contains_token(&stripped, "MEASURED")
}

/// The gate's documentation says the lever does nothing unless someone sets it.
/// A kill switch (default ON, env turns it OFF) already delivers its result and is
/// not the hazard this guard is about.
fn declares_default_off(doc: &str) -> bool {
    let lower = doc.to_lowercase();
    ["default-off", "default off", "opt-in"]
        .iter()
        .any(|needle| lower.contains(needle))
}

fn marked_dark(doc: &str) -> bool {
    contains_token(doc, "DARK")
}

/// The gate is in scope for this guard at all: it documents an observed result AND
/// says it does nothing unless someone sets the variable.
fn is_candidate(site: &GateSite) -> bool {
    claims_measured_result(&site.doc) && declares_default_off(&site.doc)
}

/// The one predicate this guard is built on, kept pure so it can be negative-tested
/// on synthetic input below.
fn is_stranded(site: &GateSite, delivered: &BTreeSet<String>) -> bool {
    if !is_candidate(site) {
        return false;
    }
    if marked_dark(&site.doc) {
        return false;
    }
    let mut names = ny_tokens(&site.doc);
    names.insert(site.var.clone());
    !names.iter().any(|name| delivered.contains(name))
}

/// `NY_*` names that a per-benchmark preset or the scored entry point can reach.
fn delivery_tokens(root: &Path) -> BTreeSet<String> {
    let mut text = std::fs::read_to_string(root.join(RUN_INSTANCE)).unwrap_or_else(|e| {
        panic!("cannot read {RUN_INSTANCE}: {e} — the scored entry point must exist")
    });
    let mut files = Vec::new();
    rust_sources(&root.join(PRESET_DIR), root, &mut files);
    for (rel, path) in files {
        if is_test_path(&rel) {
            continue;
        }
        text.push('\n');
        text.push_str(&std::fs::read_to_string(&path).unwrap_or_default());
    }
    ny_tokens(&text)
}

fn scan_workspace(root: &Path) -> Vec<GateSite> {
    let mut files = Vec::new();
    rust_sources(&root.join("crates"), root, &mut files);
    let mut sites = Vec::new();
    for (rel, path) in files {
        if is_test_path(&rel) {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        if !text.contains("\"NY_") {
            continue;
        }
        sites.extend(scan_source(&rel, &text));
    }
    sites
}

#[test]
fn every_measured_default_off_gate_declares_a_delivery_path() {
    let root = workspace_root();
    let sites = scan_workspace(&root);
    assert!(
        sites.len() >= MIN_EXPECTED_READ_SITES,
        "the env-gate scanner found only {} `std::env::var(\"NY_…\")` read sites (expected \
         >= {MIN_EXPECTED_READ_SITES}). The read idiom probably changed and this guard is now \
         matching nothing — fix `env_reads_in_line`, do not lower the floor.",
        sites.len()
    );

    let delivered = delivery_tokens(&root);
    assert!(
        !delivered.is_empty(),
        "no NY_* name found in {PRESET_DIR} or {RUN_INSTANCE}; the delivery-path detector is \
         broken, so every gate would look stranded."
    );

    let violations: Vec<&GateSite> = sites
        .iter()
        .filter(|site| is_stranded(site, &delivered))
        .collect();

    assert!(
        violations.is_empty(),
        "{} MEASURED, default-off NY_* lever(s) have no way to reach the scored path:\n{}\n\n\
         `vnncomp_scripts/run_instance.sh` exports exactly one NY_* variable \
         (NY_UPFRONT_ATTACK, fenced to safenlp_2024), so a default-off gate is dead code \
         during scoring and the measured result above it can never be collected.\n\n\
         Pick one, in the gate's own doc comment:\n\
         \x20 (a) DELIVER it — add a typed key in {PRESET_DIR} (see \
         `MarginRowPreset::reserve_max_frac`, \"Typed form of `NY_MARGIN_ROW_RESERVE_MAX_FRAC`\") \
         and set it in the presets of the benchmarks the A/B covered; or name the already-\
         delivered variable this one aliases.\n\
         \x20 (b) Write `DARK` in that doc comment, plus the reason it stays unreachable \
         (research probe / reproducibility harness / the measurement came back negative). \
         That is a commitment, not a rubber stamp: it says nobody should expect these numbers \
         on the scoreboard.\n\n\
         Do NOT delete this guard or widen the escape vocabulary to make it green.",
        violations.len(),
        violations
            .iter()
            .map(|v| format!("  {} — {}:{}", v.var, v.file, v.line))
            .collect::<Vec<_>>()
            .join("\n")
    );
}

/// The check must stay NARROW to stay alive. A guard that flags dozens of sites is
/// noise and gets deleted, which is worse than no guard. If this trips, the textual
/// signal has gone diffuse — retighten the keying, do not raise the ceiling.
const MAX_EXPECTED_CANDIDATES: usize = 12;

#[test]
fn the_candidate_population_stays_narrow() {
    let root = workspace_root();
    let sites = scan_workspace(&root);
    let candidates: Vec<&GateSite> = sites.iter().filter(|s| is_candidate(s)).collect();
    assert!(
        candidates.len() <= MAX_EXPECTED_CANDIDATES,
        "the MEASURED + default-off keying now selects {} of {} NY_* read sites:\n{}\n\n\
         That is too diffuse to be actionable. Retighten the signal in \
         `claims_measured_result` / `declares_default_off` rather than raising \
         MAX_EXPECTED_CANDIDATES.",
        candidates.len(),
        sites.len(),
        candidates
            .iter()
            .map(|c| format!("  {} — {}:{}", c.var, c.file, c.line))
            .collect::<Vec<_>>()
            .join("\n")
    );
    // Visible with `--nocapture`; this is the whole population the guard governs.
    for c in &candidates {
        println!("candidate: {} — {}:{}", c.var, c.file, c.line);
    }
    println!(
        "{} candidate(s) out of {} NY_* read sites",
        candidates.len(),
        sites.len()
    );
}

// ---------------------------------------------------------------------------
// Negative tests. A guard that only ever passes proves nothing, so the predicate
// is exercised on synthetic sources that are deliberately in violation.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod scanner {
    use super::*;

    fn delivered() -> BTreeSet<String> {
        ["NY_MARGIN_ROW_RESERVE_MAX_FRAC".to_string()]
            .into_iter()
            .collect()
    }

    fn only_site(src: &str) -> GateSite {
        let mut sites = scan_source("crates/ny-fixture/src/lib.rs", src);
        assert_eq!(
            sites.len(),
            1,
            "fixture should contain exactly one read site"
        );
        sites.remove(0)
    }

    const VIOLATING: &str = r#"
/// #synthetic-lever gate.
///
/// MEASURED on cifar100_2024 (official 100 s budget, 16 rows): converts 3 rows
/// from unknown to unsat. Default OFF, so unset is byte-identical.
fn synthetic_lever_enabled() -> bool {
    std::env::var("NY_SYNTHETIC_LEVER").is_ok_and(|v| v == "1")
}
"#;

    #[test]
    fn a_measured_default_off_gate_with_no_delivery_path_is_flagged() {
        assert!(
            is_stranded(&only_site(VIOLATING), &delivered()),
            "the guard must flag a lever that documents a MEASURED win, says it is default-off, \
             and has neither a typed preset key nor a DARK marker"
        );
    }

    #[test]
    fn the_same_gate_passes_once_it_is_marked_dark() {
        let fixed = VIOLATING.replace(
            "Default OFF, so unset is byte-identical.",
            "DARK: research probe, no competition delivery planned. Default OFF.",
        );
        assert!(
            !is_stranded(&only_site(&fixed), &delivered()),
            "an explicit DARK marker is one of the two accepted dispositions"
        );
    }

    #[test]
    fn the_same_gate_passes_once_it_names_a_typed_preset_key() {
        let fixed = VIOLATING.replace(
            "Default OFF, so unset is byte-identical.",
            "Default OFF; delivered through the typed key behind \
             `NY_MARGIN_ROW_RESERVE_MAX_FRAC`.",
        );
        assert!(
            !is_stranded(&only_site(&fixed), &delivered()),
            "naming a variable that the preset schema exposes is a delivery path"
        );
    }

    #[test]
    fn a_measured_kill_switch_is_not_a_candidate() {
        // Default ON, env turns it OFF: the measured result already ships.
        let src = VIOLATING.replace(
            "Default OFF, so unset is byte-identical.",
            "Default ON; set the variable to 0 to restore the scalar reference.",
        );
        assert!(
            !is_stranded(&only_site(&src), &delivered()),
            "a default-ON kill switch already delivers its measurement"
        );
    }

    #[test]
    fn an_undocumented_default_off_gate_is_not_a_candidate() {
        // No MEASURED claim: nothing is asserted to be lost, so nothing to deliver.
        let src = VIOLATING.replace("MEASURED on", "Believed to help on");
        assert!(
            !is_stranded(&only_site(&src), &delivered()),
            "the guard is keyed on an explicit MEASURED claim, not on every default-off gate"
        );
    }

    #[test]
    fn a_variable_named_measured_does_not_fake_the_claim() {
        let src = r#"
/// Default-off diagnostic.
fn reentry() -> bool {
    std::env::var("NY_PATCHES_REENTRY_MEASURED").is_ok()
}
"#;
        assert!(
            !is_stranded(&only_site(src), &delivered()),
            "MEASURED inside a variable NAME is not a documented measurement"
        );
    }

    #[test]
    fn the_module_header_is_not_attributed_to_individual_gates() {
        let src = r#"//! MEASURED MOTIVATION for this whole module.

/// Unrelated default-off probe.
fn probe() -> bool {
    std::env::var("NY_UNRELATED_PROBE").is_ok()
}
"#;
        assert!(
            !is_stranded(&only_site(src), &delivered()),
            "a shared `//!` header must not tag every gate in the file"
        );
    }

    #[test]
    fn test_scaffolding_is_excluded() {
        assert!(is_test_path("crates/ny-cli/src/preset/tests.rs"));
        assert!(is_test_path(
            "crates/ny-cli/src/preset/vnncomp_preset_tests.rs"
        ));
        assert!(is_test_path("crates/ny-cli/tests/ay_pin_unity.rs"));
        assert!(!is_test_path("crates/ny-cli/src/preset/mod.rs"));
        assert!(!is_test_path("crates/ny-test-utils/src/lib.rs"));
    }
}
