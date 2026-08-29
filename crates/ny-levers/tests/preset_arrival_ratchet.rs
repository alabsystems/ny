// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! PRESET ARRIVAL RATCHET — does a value written in a shipped yaml actually
//! REACH the lane that is supposed to consume it?
//!
//! # Why this exists, and why the two existing ratchets are not enough
//!
//! `tests/ratchet.rs` freezes raw `NY_*` reads. `tests/reachability_ratchet.rs`
//! proves every verdict-affecting axis is DELIVERABLE by a typed preset key.
//! Both are necessary. Neither is sufficient, because **deliverable is not
//! delivered**. Five mechanisms were found disconnected in a single day, and
//! every one of them was already deliverable, already had a consumer, and
//! already passed both existing ratchets:
//!
//! | mechanism | what the config said | what actually happened |
//! |---|---|---|
//! | cgan input clip | `bab.clip.relaxed: true` in `configs/vnncomp25/cgan_2023.yaml` | CLEARED at runtime. `BatchNorm` makes `Graph::is_input_split_batch_stack_safe` false, `should_disable_disjunctive_relaxed_clip` fires, and `config_with_disjunctive_clip_routes_disabled` zeroes `enable_relaxed_clip` on a clone of the resolved config. |
//! | metaroom CZ | a "constrained zonotope" lane | `ConstrainedZonotopeTailLpLimits { max_constraints: 0, max_constraint_elements: 0, max_constraint_nonzeros: 0 }` — an UNCONDITIONAL struct literal. Every domain carrying a constraint row is refused at plan time, so the lane ran as plain DeepZ. |
//! | `alpha_iteration: 1` | "one warm step" | ZERO SPSA steps. The warm-start loop breaks on `iter == config.iterations - 1`, which fires at `iter == 0`, BEFORE the first gradient. |
//! | margin-row beta per-column | `cols=8` requested | `cols=1` delivered on every call: the live column set is `min(configured, nf)` and the classwise route builds `nf == 1` by construction. Bit-identical to shared beta — a provable identity. |
//! | `tableau_f32` | implemented and correct | armed in NO shipped yaml until somebody happened to look. |
//!
//! Every one of those produced nulls that read as evidence, and three fronts
//! were closed on such nulls in one week. That is the single largest source of
//! wasted effort in the campaign, so it gets a gate rather than a habit.
//!
//! # The three failure modes, and the check that catches each
//!
//! 1. **RUNTIME CLEARING.** A resolved config field is overwritten downstream
//!    of resolution, before the lane reads it. Caught by
//!    [`preset_override_frontier_is_exact`]: the DELIVERY SURFACE (every engine
//!    field `preset::apply` writes) is derived from source, every write to one
//!    of those fields anywhere else in first-party non-test code is an OVERRIDE
//!    SITE, and the exact set of override sites is checked in. A new one cannot
//!    land silently. The same frontier holds CONFIG REBUILDS — a
//!    `BetaCrownConfig::default()` outside the resolver drops all one hundred
//!    and sixty-four preset fields at once, which no per-field scan can see and
//!    which the input-split rebound adapter does today, carrying three fields
//!    forward by hand.
//! 2. **REQUESTED-VS-DELIVERED.** The mechanism runs with an effective value
//!    that is not the configured one. Static analysis cannot decide this in
//!    general, so [`DELIVERY_CAPPED`] names the keys whose delivered value is
//!    known to be clamped by a runtime quantity, and
//!    [`delivery_capped_keys_are_adjudicated_before_arming`] fails if a shipped
//!    yaml arms one without an adjudication row. The obligation is placed at the
//!    moment of ARMING, which is the moment the measurement starts being taken.
//! 3. **SEMANTICALLY DEAD VALUES.** A legal value that makes the mechanism a
//!    no-op. Two shapes, two checks:
//!    [`dead_value_armings_are_exactly_the_declared_debt`] for the yaml side
//!    (`alpha_iteration: 1`), and
//!    [`capability_zero_frontier_is_exact`] for the code side
//!    (`max_constraints: 0`).
//!
//! 4. **DORMANT VALUES.** A value that is legal, live, and simply not consumed
//!    on the route this preset takes — until somebody edits a NEIGHBOURING key
//!    months later and the unexercised number takes effect at full size.
//!    [`dormant_value_states_are_exactly_the_declared_frontier`] puts the
//!    dormant/live STATE in the row identity, so flipping the gate flips the
//!    row and fails the gate at review time. `bab.batch_size: 64` sat
//!    unconsumed behind `reorder_bab: false` — the input-split loop discards it
//!    for a hardcoded `1` — on a host where an over-admission is a global OOM
//!    rather than a refusal. Neither of the other yaml-side checks can see it:
//!    it is not a dead value (64 is a fine batch size) and nothing overwrites
//!    it.
//!
//! Before any of that, [`every_delivered_field_has_a_consumer`] asks the
//! question that comes first: does anything read the field at all? That one is
//! a hard zero rather than a ratchet, because the debt is already zero.
//!
//! Plus the two adjacent shapes that produced the same class of silent null:
//! [`unarmed_promotions_are_exactly_the_declared_set`] (built, deliverable, and
//! armed by nothing — `tableau_f32`) and [`no_source_file_is_empty`] (a `.rs`
//! truncated to zero bytes still compiles as a valid empty module and takes its
//! `#[test]` functions with it — found live in the shared tree).
//!
//! # Reading a failure
//!
//! Every check reports **ALL** of its violations in one run, and each check is
//! its own `#[test]` so `cargo test` runs the rest after one fails. That is
//! deliberate: `tests/source_frontier.rs` asserts three baselines sequentially
//! in a single test and therefore reveals exactly one debt per run, which cost
//! two people several cycles in a single day. Do not merge these into one
//! assertion.
//!
//! # What this is NOT
//!
//! It is a lexical policy over first-party source, not a Rust name resolver and
//! not a data-flow analysis. It cannot prove a value arrives; it makes every
//! structural opportunity for it not to arrive **declared and reviewed**. The
//! frontier files therefore hold today's measured debt: an entry is not an
//! endorsement, it is a receipt. Additions AND removals fail, so fixing a debt
//! is a deliberate edit in the same commit.
//!
//! **Regenerate a manifest ONLY from this file's own failure output.** THESE
//! SCANNERS ARE THE AUTHORITY; any out-of-band approximation of them is a
//! second implementation that will disagree in exactly the cases that matter.
//! This was demonstrated the first week: a Python cross-check that missed the
//! `#[cfg(all(test, ...))]` needle and swept `UNSAFE_AXES` alongside `axes()`
//! rewrote two correct manifests with nine phantom rows, and the next run's
//! UNDECLARED/STALE output is what caught it. That is the same lesson
//! `source_frontier.rs` already teaches — take every value from the ratchet's
//! own failure output, not from a summary.
//!
//! Two neighbouring surfaces are DELIBERATELY out of scope, and the reasoning
//! matters because both look like holes:
//!
//! * **`ny_levers::read_over_config`** (42 production call sites) overrides a
//!   resolved value at the READ site rather than in the struct, so no field
//!   scan can see it. It is excluded because its precedence is env-over-config
//!   and `vnncomp_scripts/run_instance.sh` exports **zero** `NY_*` variables:
//!   on a scored run the environment layer is empty and the config wins by
//!   construction. That is the premise the whole preset system rests on, and it
//!   is checked by `tests/ratchet.rs` and `tests/reachability_ratchet.rs`
//!   rather than here. If `run_instance.sh` ever exports an `NY_*` variable,
//!   this exclusion stops being sound and these sites become the first thing to
//!   add.
//! * **Process-global `force_*(Option<bool>)` seams** override a
//!   config-derived decision without touching a config struct and WITHOUT an
//!   env precedence argument, so they are a genuine gap in this gate rather
//!   than a reasoned exclusion. They are named here so the next reader inherits
//!   the finding instead of re-deriving it.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use ny_levers::space::{Class, Deliver};

// ---------------------------------------------------------------------------
// Checked-in frontiers
// ---------------------------------------------------------------------------

const OVERRIDE_MANIFEST: &str = include_str!("data/preset_override_frontier.tsv");
const CAPABILITY_ZERO_MANIFEST: &str = include_str!("data/preset_capability_zero_frontier.tsv");
const DEAD_VALUE_MANIFEST: &str = include_str!("data/preset_dead_value_frontier.tsv");
const DORMANT_MANIFEST: &str = include_str!("data/preset_dormant_frontier.tsv");
const UNARMED_PROMOTION_MANIFEST: &str = include_str!("data/preset_unarmed_promotions.tsv");

/// The file whose `config.<field> = ...` writes DEFINE the delivery surface.
/// Every yaml key lands here and nowhere else.
const APPLY_RS: &str = "crates/ny-cli/src/preset/apply.rs";

/// Config years that ship. `vnncomp26` SHADOWS `vnncomp25`: the resolver
/// searches newest-year-first, so a same-named yaml in the newer directory
/// wins and edits to the older one are placebos. Both are scanned, and the
/// year is part of every reported identity so a shadowed pair is visible.
const CONFIG_YEARS: &[&str] = &["vnncomp24", "vnncomp25", "vnncomp26"];

/// Not first-party competition source. `/target/` is build output, `/.lake/`
/// is a vendored Lean toolchain, `/proofs/lean/` is vendored proof source, and
/// `/examples/` and `/benches/` cannot run on a scored instance.
const SKIP_FRAGMENTS: &[&str] = &[
    "/target/",
    "/.lake/",
    "/proofs/lean/",
    "/examples/",
    "/benches/",
];

// ---------------------------------------------------------------------------
// Mode 2 registry: keys whose DELIVERED value is clamped by a runtime quantity
// ---------------------------------------------------------------------------

/// A preset key whose effective value is not its configured value.
struct DeliveryCapped {
    /// Dotted key exactly as it is written in a yaml.
    key: &'static str,
    /// The runtime quantity that clamps it, as a source citation.
    capped_by: &'static str,
    /// The evidence symbol that lets an operator SEE requested vs delivered on
    /// a real run. A key with no such symbol cannot be adjudicated at all.
    evidence: &'static str,
    /// What was measured when this was last looked at.
    measured: &'static str,
    /// Exact identifiers, in first-party non-test source, that carry the
    /// REQUESTED and the DELIVERED value respectively.
    ///
    /// Checked to exist on every run. Static analysis cannot decide whether the
    /// delivered value equals the configured one — that needs the lane, at
    /// runtime, on a real instance — so the durable half of the obligation is
    /// this: the pair of fields that lets an operator ANSWER the question must
    /// still be there. A renamed or deleted evidence field is how a
    /// requested-vs-delivered gap goes back to being invisible, and it is
    /// exactly what happened to the `[beta-pc] armed (… cols=8)` banner, which
    /// the runbook still quotes and which no longer exists in the tree.
    observable: [&'static str; 2],
}

/// Keys that are known to deliver something other than what they say.
///
/// Arming one of these without an adjudication row in
/// `data/preset_dead_value_frontier.tsv` is a violation, because that is
/// exactly the state in which a measurement through the key is uninterpretable:
/// a null cannot be told apart from the mechanism never having widened.
const DELIVERY_CAPPED: &[DeliveryCapped] = &[
    DeliveryCapped {
        key: "margin_row.beta_columns",
        capped_by: "margin_row::bab — the live column set is truncated to \
                    min(configured, candidate_count), and candidate_count is bounded by \
                    MarginBatch::nf(); the classwise route constructs MarginBatch::new(net, t, \
                    &[class]) with exactly one class, so nf == 1 by construction",
        evidence: "PolicyExecutionEvidence::beta_columns (frozen configured width) vs \
                   ::beta_columns_max_candidates (largest live set actually seen) and \
                   ::beta_columns_binding_evals, rendered by \
                   commands::margin_row_bab::render_margin_row_policy_execution",
        measured: "cols=1 on every call on the cifar100/tinyimagenet route, where the root \
                   discharges all classes but one; per-column beta is then bit-identical to \
                   shared beta",
        observable: ["beta_columns", "beta_columns_max_candidates"],
    },
    DeliveryCapped {
        key: "margin_row.beta_per_column",
        capped_by: "same nf == 1 collapse as margin_row.beta_columns — the per-column \
                    mechanism has one column to be per",
        evidence: "PolicyExecutionEvidence::beta_per_column_evals vs \
                   ::beta_columns_max_candidates",
        measured: "a provable identity with shared beta whenever nf == 1",
        observable: ["beta_per_column_evals", "beta_columns_max_candidates"],
    },
];

// ---------------------------------------------------------------------------
// Mode 4 registry: values that lie DORMANT until a neighbouring key wakes them
// ---------------------------------------------------------------------------

/// A key whose value is not consumed on the route the preset currently takes,
/// and that becomes live when a DIFFERENT key changes.
///
/// This is a fourth failure mode, and it is the nastiest of the four, because
/// the dangerous moment is not when the value is written — it is when somebody
/// edits a neighbouring key months later. Until then the value looks armed,
/// reviews as armed, and has never applied to anything, so nothing about it has
/// ever been validated. The day the neighbour flips, an unexercised number
/// takes effect at full size with no measurement behind it.
///
/// Not hypothetical: `bab.batch_size: 64` sat unconsumed until `reorder_bab`
/// flipped, and then cost an 11.7x memory regression (6.3 GB -> 73.4 GB) on a
/// unified-memory host where over-admission is a global OOM rather than a
/// refusal.
///
/// Neither of the other yaml-side checks can see this. It is not a dead VALUE
/// (64 is a perfectly good batch size) and it is not a runtime CLEAR (nothing
/// overwrites it). It is a live value on a route nobody takes.
struct DormantUntil {
    /// The keys that can carry the value, in the resolver's own application
    /// order — LOWEST precedence first. `preset::apply` applies
    /// `solver.batch_size` and then `bab.batch_size`, so when a yaml sets both
    /// (as `nn4sys.yaml` does, at 256 and 64) the engine sees the LAST one.
    /// Tracking the alias rather than the effective key would report a number
    /// the run never used.
    key_aliases: &'static [&'static str],
    /// The key that selects the route on which the dormancy holds, and the
    /// value that selects it. A yaml that does not take this route is not
    /// tracked at all — the dormancy is a property of the route, not of the
    /// key.
    route: (&'static str, &'static str),
    /// The key whose value decides whether the carried value is reached.
    gate: &'static str,
    /// The `gate` value at which the carried value becomes live.
    live_when: &'static str,
    /// The route argument: why the value is unreachable while the gate is
    /// elsewhere. Must cite the discarding site.
    why: &'static str,
    /// What was measured the last time the pair was live together. This is the
    /// number a reviewer needs, and it is the one nobody had.
    measured: &'static str,
}

/// Pairs whose co-arming state is tracked in
/// `data/preset_dormant_frontier.tsv`.
///
/// The frontier records, per shipped yaml, whether each pair is `dormant` or
/// `live`. A yaml that flips its gate flips that row, the exact-set diff fails,
/// and somebody has to look at the dormant value BEFORE it takes effect —
/// which is the whole point, and the one thing that would have caught the
/// batch-size regression at review time instead of at 73.4 GB.
const DORMANT_UNTIL: &[DormantUntil] = &[DormantUntil {
    key_aliases: &["solver.batch_size", "bab.batch_size"],
    route: ("bab.branching.method", "input"),
    gate: "bab.branching.input_split.reorder_bab",
    live_when: "true",
    why: "on the CPU input-split routes the configured batch size is not merely re-derived when \
          reorder_bab is off — it is DISCARDED for a hardcoded literal. All three sites have the \
          same shape: `let loop_batch_size = if self.config.reorder_bab { \
          input_split_loop_batch_size(self.config.batch_size, ..) } else { 1 };` \
          (input_split/single_objective.rs:474, input_split/multi_objective.rs:700, \
          input_split/disjunctive_multi_clause.rs:1437). `loop_batch_size` is the sole consumer \
          and sets how many domains are bounded simultaneously, so live memory scales linearly \
          with it. SCOPE, and this is the part a naive encoding gets wrong: the value is NOT \
          globally inert when reorder_bab is false — it is live on the ReLU-split \
          (relu_split/bab_loop.rs:287), multi-objective ReLU-split \
          (multi_objective/verify.rs:461), DomainList/gpu_bab (gpu_bab/mod.rs:351) and legacy \
          (engine/core.rs:550) routes regardless of the gate. That is why this entry is scoped \
          to bab.branching.method: input.",
    measured: "NO in-tree measurement corroborates the reported 11.7x memory regression (6.3 GB \
               -> 73.4 GB); searches for those figures return nothing in docs/, designs/, \
               .gatelogs/ or git history, so the SHAPE is encoded here and the numbers are not. \
               What IS in tree is cgan_2023.yaml:136-140 in its own words: with reorder_bab on, \
               `at batch 64 it breadth-expanded to 319 domains without a verified child in \
               120 s. Batch 8 was much healthier`. And docs/DEAD_MECHANISM_CENSUS_2026-08-25.md \
               :266-297 found the sibling `stacked_rebound: true` inert on the same yaml for the \
               same reason, describing the mechanism as `one dispatch per node instead of 64` — \
               the 64 being this very key — without naming batch_size as the carrier.",
}];

// ---------------------------------------------------------------------------
// Mode 3 registry (yaml side): values that make a mechanism a no-op
// ---------------------------------------------------------------------------

/// Is this dotted key an iteration COUNT whose loop performs
/// `iterations - 1` optimizer steps?
///
/// Written as a family rule rather than a literal key list on purpose: the
/// defect is a property of the LOOP SHAPE, so any key that feeds an
/// `AlphaCrownConfig::iterations`-style counter inherits it. The warm-start
/// loop breaks on `iter == config.iterations - 1` BEFORE computing the first
/// SPSA gradient, so a configured `1` buys exactly zero optimizer steps while
/// reading, in the yaml and in every summary, as "refinement is on".
fn is_iteration_count_key(key: &str) -> bool {
    let leaf = key.rsplit('.').next().unwrap_or(key);
    leaf == "iterations"
        || leaf == "alpha_iteration"
        || leaf == "beta_iterations"
        || leaf.ends_with("_iterations")
}

/// Why a given `(key, value)` pair is a no-op, or `None` if it is not one.
///
/// Fail-closed by construction: this answers "is this value dead", and every
/// `Some` must appear in the dead-value frontier or the check fails.
fn dead_value_reason(key: &str, value: &str) -> Option<&'static str> {
    if is_iteration_count_key(key) && value.trim() == "1" {
        return Some(
            "an iteration count of 1 runs ZERO optimizer steps on the warm-start path: \
             graph_alpha::bounds::warm_start breaks on `iter == config.iterations - 1`, which \
             fires at iter == 0, before the first SPSA gradient. The child is bounded with the \
             parent's frozen alphas while the config claims refinement is enabled.",
        );
    }
    None
}

// ---------------------------------------------------------------------------
// Lexical helpers
// ---------------------------------------------------------------------------

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .canonicalize()
        .expect("workspace root must resolve")
}

/// Replace comment and string-literal bytes with spaces, preserving newlines
/// and byte offsets so line numbers stay exact.
///
/// Without this the scan counts its own documentation: this very file names
/// `enable_relaxed_clip` and `max_constraints: 0` in prose.
fn blank_comments_and_strings(src: &str) -> String {
    let bytes = src.as_bytes();
    let mut out = vec![b' '; bytes.len()];
    let mut i = 0usize;
    while i < bytes.len() {
        let c = bytes[i];
        if c == b'\n' {
            out[i] = b'\n';
            i += 1;
        } else if c == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
        } else if c == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
            // Rust block comments nest.
            let mut depth = 1usize;
            i += 2;
            while i < bytes.len() && depth > 0 {
                if bytes[i] == b'\n' {
                    out[i] = b'\n';
                    i += 1;
                } else if bytes[i] == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
                    depth += 1;
                    i += 2;
                } else if bytes[i] == b'*' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
                    depth -= 1;
                    i += 2;
                } else {
                    i += 1;
                }
            }
        } else if c == b'r'
            && bytes
                .get(i + 1..)
                .is_some_and(|rest| raw_string_open(rest).is_some())
        {
            let hashes = raw_string_open(&bytes[i + 1..]).expect("checked above");
            i += 1 + hashes + 1; // r + #* + "
            loop {
                if i >= bytes.len() {
                    break;
                }
                if bytes[i] == b'\n' {
                    out[i] = b'\n';
                    i += 1;
                    continue;
                }
                if bytes[i] == b'"' && bytes[i + 1..].iter().take(hashes).all(|b| *b == b'#') {
                    i += 1 + hashes;
                    break;
                }
                i += 1;
            }
        } else if c == b'"' {
            i += 1;
            while i < bytes.len() {
                if bytes[i] == b'\\' {
                    i += 2;
                    continue;
                }
                if bytes[i] == b'\n' {
                    out[i] = b'\n';
                    i += 1;
                    continue;
                }
                if bytes[i] == b'"' {
                    i += 1;
                    break;
                }
                i += 1;
            }
        } else if c == b'\'' {
            // Char literal, or a lifetime. A lifetime is `'` + ident and no
            // closing quote; blanking it would be harmless, but consuming to
            // the next quote would eat real code.
            if bytes.get(i + 1) == Some(&b'\\') {
                i += 2;
                while i < bytes.len() && bytes[i] != b'\'' {
                    i += 1;
                }
                i += 1;
            } else if bytes.get(i + 2) == Some(&b'\'') {
                i += 3;
            } else {
                out[i] = bytes[i];
                i += 1;
            }
        } else {
            out[i] = bytes[i];
            i += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// `Some(n)` when `rest` opens a raw string with `n` hashes (`r"`, `r#"`, ...).
fn raw_string_open(rest: &[u8]) -> Option<usize> {
    let hashes = rest.iter().take_while(|b| **b == b'#').count();
    (rest.get(hashes) == Some(&b'"')).then_some(hashes)
}

/// Blank every `#[cfg(test)]`-annotated item, brace-matched.
///
/// An inline `#[cfg(test)] mod tests { ... }` is test code living in a
/// production file; counting its fixtures as override sites would bury the
/// real ones under hundreds of `enable_relaxed_clip: false` test literals.
fn blank_cfg_test_items(src: &str) -> String {
    const NEEDLES: &[&str] = &["#[cfg(test)]", "#[cfg(all(test", "#[cfg(any(test"];
    let bytes = src.as_bytes();
    let mut out = bytes.to_vec();
    let mut search = 0usize;
    while let Some((start, needle)) = NEEDLES
        .iter()
        .filter_map(|n| src[search..].find(n).map(|k| (search + k, *n)))
        .min_by_key(|(k, _)| *k)
    {
        let mut i = start + needle.len();
        while i < bytes.len() && bytes[i] != b'{' && bytes[i] != b';' {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }
        let end = if bytes[i] == b';' {
            i
        } else {
            let mut depth = 0usize;
            let mut j = i;
            while j < bytes.len() {
                if bytes[j] == b'{' {
                    depth += 1;
                } else if bytes[j] == b'}' {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                j += 1;
            }
            j.min(bytes.len() - 1)
        };
        for slot in out.iter_mut().take(end + 1).skip(start) {
            if *slot != b'\n' {
                *slot = b' ';
            }
        }
        search = end + 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Is this a test file rather than production source?
///
/// Deliberately broad on the basename side: `tests_graph_disjunctive.rs` is
/// test code and does not end in `_tests.rs`.
fn is_test_source(relative: &str) -> bool {
    let base = relative.rsplit('/').next().unwrap_or(relative);
    relative.contains("/tests/")
        || base == "tests.rs"
        || base == "test.rs"
        || base.starts_with("tests_")
        || base.starts_with("test_")
        || base.ends_with("_tests.rs")
        || base.ends_with("_test.rs")
}

fn collect_rs(dir: &Path, root: &Path, out: &mut Vec<(String, PathBuf)>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(kind) = entry.file_type() else {
            continue;
        };
        if kind.is_symlink() {
            continue;
        }
        let relative = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");
        if kind.is_dir() {
            let probe = format!("/{relative}/");
            if SKIP_FRAGMENTS.iter().any(|f| probe.contains(f)) {
                continue;
            }
            collect_rs(&path, root, out);
        } else if kind.is_file()
            && path
                .extension()
                .is_some_and(|extension| extension.eq_ignore_ascii_case("rs"))
        {
            let probe = format!("/{relative}");
            if SKIP_FRAGMENTS.iter().any(|f| probe.contains(f)) {
                continue;
            }
            out.push((relative, path));
        }
    }
}

const fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

// ---------------------------------------------------------------------------
// The delivery surface
// ---------------------------------------------------------------------------

/// Every engine field that `preset::apply` writes — i.e. every field a shipped
/// yaml key can land in.
///
/// Derived from source rather than listed, so a new preset key joins the
/// surface the moment its `config.<field> = ...` line is written and every
/// override of it becomes a declared site in the same commit.
fn delivery_surface(apply_rs: &Path) -> BTreeSet<String> {
    let Ok(raw) = std::fs::read_to_string(apply_rs) else {
        return BTreeSet::new();
    };
    let src = blank_cfg_test_items(&blank_comments_and_strings(&raw));
    let bytes = src.as_bytes();
    let mut out = BTreeSet::new();
    let mut i = 0usize;
    while let Some(k) = src[i..].find("config.") {
        let at = i + k;
        i = at + "config.".len();
        // `config` must be a whole token: reject `self.config.` / `my_config.`.
        if at > 0 && (is_ident_byte(bytes[at - 1]) || bytes[at - 1] == b'.') {
            continue;
        }
        let start = at + "config.".len();
        let mut j = start;
        while j < bytes.len() && is_ident_byte(bytes[j]) {
            j += 1;
        }
        if j == start {
            continue;
        }
        let mut k = j;
        while k < bytes.len() && (bytes[k] == b' ' || bytes[k] == b'\t') {
            k += 1;
        }
        // A single `=` not followed by `=` is an assignment.
        if bytes.get(k) == Some(&b'=') && bytes.get(k + 1) != Some(&b'=') {
            out.insert(src[start..j].to_string());
        }
    }
    out
}

// ---------------------------------------------------------------------------
// The scanned corpus
// ---------------------------------------------------------------------------

/// One first-party non-test source file, with comments, string literals and
/// `#[cfg(test)]` items blanked out.
struct Scanned {
    /// Repo-relative path, e.g. `crates/ny-propagate/src/margin_row/bab.rs`.
    path: String,
    /// Same byte length and line numbering as the original.
    blanked: String,
}

/// Walk and blank the first-party non-test corpus.
///
/// Pure in its argument so the demonstrations can point it at a fixture tree.
fn scan_sources(crates_root: &Path) -> Vec<Scanned> {
    let mut files = Vec::new();
    collect_rs(crates_root, crates_root, &mut files);
    files.sort();

    let mut out = Vec::with_capacity(files.len());
    for (relative, path) in &files {
        let full = format!("crates/{relative}");
        if is_test_source(&full) {
            continue;
        }
        let Ok(raw) = std::fs::read_to_string(path) else {
            continue;
        };
        out.push(Scanned {
            path: full,
            blanked: blank_cfg_test_items(&blank_comments_and_strings(&raw)),
        });
    }
    out
}

/// The real tree's corpus, blanked exactly once per test binary.
///
/// `crates/` is 66 MiB of source and every check below walks it. Blanking it
/// per test would make this gate slow enough that somebody eventually takes it
/// out of `scripts/gate.sh`, which is the only way a ratchet really fails.
fn workspace_sources() -> &'static [Scanned] {
    static CACHE: std::sync::OnceLock<Vec<Scanned>> = std::sync::OnceLock::new();
    CACHE.get_or_init(|| scan_sources(&workspace_root().join("crates")))
}

// ---------------------------------------------------------------------------
// Mode 1: override sites
// ---------------------------------------------------------------------------

/// One `(file, field)` pair that writes a delivery-surface field outside the
/// resolver, with how many times it does so.
type OverrideSites = BTreeMap<(String, String), usize>;

/// Find every write to a delivery-surface field in first-party non-test source.
///
/// An assignment whose right-hand side mentions the SAME field is an
/// accumulator (`stats.max_depth = stats.max_depth.max(d)`), not an override of
/// a configured value, and is excluded — that distinction removes the bulk of
/// the telemetry noise without weakening the check, because an accumulator
/// cannot deliver a different value than the yaml asked for.
fn override_sites(sources: &[Scanned], surface: &BTreeSet<String>) -> OverrideSites {
    let mut out: OverrideSites = BTreeMap::new();
    for file in sources {
        if file.path == APPLY_RS {
            continue;
        }
        for (field, _line) in field_assignments(&file.blanked) {
            if surface.contains(field) {
                *out.entry((file.path.clone(), field.to_string()))
                    .or_insert(0) += 1;
            }
        }
    }
    out
}

/// Every delivery-surface field that is READ at least once in first-party
/// non-test source outside the resolver.
///
/// A field the resolver writes and nobody reads is a preset key that parses,
/// validates, type-checks and does NOTHING — the exact class
/// `crates/ny-cli/src/preset/contract.rs` was written to report, except that
/// module's registry is hand-maintained and therefore, by construction, cannot
/// contain a dead field nobody has noticed yet. This is the derived half.
fn consumed_fields(sources: &[Scanned], surface: &BTreeSet<String>) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for file in sources {
        if file.path == APPLY_RS {
            continue;
        }
        for field in field_reads(&file.blanked) {
            if surface.contains(field) {
                out.insert(field.to_string());
            }
        }
    }
    out
}

/// Is the byte at `at` the `=` of a PLAIN assignment?
///
/// Rejects `==`, `!=`, `>=`, `<=`, `+=` and friends — and, critically, **`=>`**.
/// A match guard reads `Err(e) if config.some_flag => { .. }`, which puts a
/// config field immediately before an `=`; without this arm the scanner calls
/// that an assignment. It produced exactly three false "override sites" on the
/// first real run (`forward_linear_deadline_fallback_to_ibp` twice,
/// `root_crown_interm_dense_head` once) — and in [`field_reads`] the same bug
/// fails OPEN, hiding a genuine consumer and making
/// [`every_delivered_field_has_a_consumer`] able to report a live field as
/// dead.
fn is_assignment_operator(bytes: &[u8], at: usize) -> bool {
    if bytes.get(at) != Some(&b'=') {
        return false;
    }
    if matches!(bytes.get(at + 1), Some(b'=' | b'>')) {
        return false;
    }
    at == 0
        || !matches!(
            bytes[at - 1],
            b'+' | b'-' | b'*' | b'/' | b'%' | b'!' | b'<' | b'>'
        )
}

/// Field names appearing as `.<ident>` where the access is NOT a plain
/// assignment. `x.f == y`, `x.f.min(z)`, `if x.f {`, `x.f += 1` and the match
/// guard `if x.f =>` are all reads; only `x.f = y` is excluded.
fn field_reads(src: &str) -> Vec<&str> {
    let bytes = src.as_bytes();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] != b'.' {
            i += 1;
            continue;
        }
        let mut j = i + 1;
        while j < bytes.len() && (bytes[j] == b' ' || bytes[j] == b'\t') {
            j += 1;
        }
        let start = j;
        while j < bytes.len() && is_ident_byte(bytes[j]) {
            j += 1;
        }
        if j == start || bytes[start].is_ascii_digit() {
            i += 1;
            continue;
        }
        let mut k = j;
        while k < bytes.len() && (bytes[k] == b' ' || bytes[k] == b'\t' || bytes[k] == b'\n') {
            k += 1;
        }
        let is_plain_assignment = is_assignment_operator(bytes, k);
        if !is_plain_assignment {
            out.push(&src[start..j]);
        }
        i = j;
    }
    out
}

/// The resolved-config types that shipped yaml keys land in.
///
/// Rebuilding one of these from `::default()` outside the resolver DROPS EVERY
/// PRESET VALUE AT ONCE — a strictly worse version of clearing a single field,
/// and one that no per-field scan can see. `shared_specs.rs` does exactly this
/// for a nested rebound adapter and carries three fields forward by hand out of
/// a hundred and sixty-four.
const RESOLVED_CONFIG_TYPES: &[&str] =
    &["BetaCrownConfig", "AlphaCrownConfig", "PhaseBudgetConfig"];

/// `(file, "<Type>::default")` -> occurrences, over first-party non-test source
/// outside the resolver.
///
/// Shares the override frontier's identity scheme deliberately: a whole-config
/// rebuild and a single-field clear are the same failure with different blast
/// radii, and a reviewer should meet them in one file.
fn config_rebuild_sites(sources: &[Scanned]) -> OverrideSites {
    let mut out: OverrideSites = BTreeMap::new();
    for file in sources {
        if file.path == APPLY_RS {
            continue;
        }
        let src = &file.blanked;
        for ty in RESOLVED_CONFIG_TYPES {
            let needle = format!("{ty}::default()");
            let mut from = 0usize;
            while let Some(at) = src[from..].find(&needle) {
                let absolute = from + at;
                from = absolute + needle.len();
                // Require a type boundary: `MyBetaCrownConfig::default()` is a
                // different type.
                if absolute > 0 && is_ident_byte(src.as_bytes()[absolute - 1]) {
                    continue;
                }
                *out.entry((file.path.clone(), format!("{ty}::default")))
                    .or_insert(0) += 1;
            }
        }
    }
    out
}

/// `(field, line)` for every `<receiver>.<field> = <rhs>;` statement.
///
/// Requires a receiver (a bare `x = 1` is a local, not a config write) and
/// rejects `==`, `!=`, `>=`, `<=`, `+=` and friends.
fn field_assignments(src: &str) -> Vec<(&str, usize)> {
    let bytes = src.as_bytes();
    let mut out = Vec::new();
    let mut line = 1usize;
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'\n' {
            line += 1;
            i += 1;
            continue;
        }
        if bytes[i] != b'.' {
            i += 1;
            continue;
        }
        // A field access: `.` then ident. Require an ident/`)`/`]` immediately
        // before (possibly across whitespace) so `..base` and float literals
        // are not mistaken for one.
        let mut back = i;
        while back > 0 && (bytes[back - 1] == b' ' || bytes[back - 1] == b'\t') {
            back -= 1;
        }
        if back == 0 {
            i += 1;
            continue;
        }
        let prev = bytes[back - 1];
        if !(is_ident_byte(prev) || prev == b')' || prev == b']') {
            i += 1;
            continue;
        }
        let mut j = i + 1;
        while j < bytes.len() && (bytes[j] == b' ' || bytes[j] == b'\t') {
            j += 1;
        }
        let start = j;
        while j < bytes.len() && is_ident_byte(bytes[j]) {
            j += 1;
        }
        if j == start || bytes[start].is_ascii_digit() {
            i += 1;
            continue;
        }
        let field = &src[start..j];
        let mut k = j;
        while k < bytes.len() && (bytes[k] == b' ' || bytes[k] == b'\t' || bytes[k] == b'\n') {
            k += 1;
        }
        if !is_assignment_operator(bytes, k) {
            i = j;
            continue;
        }
        // Accumulator exclusion, deliberately NARROW: only a monotone
        // high-water mark over the field's own value
        // (`stats.max_depth = stats.max_depth.max(d)`) is excluded. A
        // `.min(...)` over the field's own value is NOT an accumulator — it
        // narrows what the yaml asked for, which is precisely an override, and
        // `clause_config.timeout = clause_config.timeout.min(remaining)` is a
        // real one. Excluding it would have been the fail-OPEN direction.
        let rhs_end = src[k..]
            .find(';')
            .map_or(src.len(), |offset| (k + offset).min(src.len()));
        let rhs = &src[k..rhs_end];
        if rhs.contains(field) && rhs.contains(".max(") {
            i = j;
            continue;
        }
        out.push((field, line));
        i = j;
    }
    out
}

// ---------------------------------------------------------------------------
// Mode 3 (code side): capability structs built with zero capacity
// ---------------------------------------------------------------------------

/// A field name that states a CAPACITY — the thing a "constrained" lane needs
/// to be non-zero in order to be constrained at all.
fn is_capacity_field(name: &str) -> bool {
    name.starts_with("max_")
        || name.ends_with("_max")
        || name.ends_with("_limit")
        || name.ends_with("_cap")
        || name.ends_with("_capacity")
        || name.ends_with("_budget")
        || name.ends_with("_quota")
}

/// `(file, sorted-field-set)` -> occurrences, for every non-test struct literal
/// that sets TWO OR MORE capacity fields to a literal zero.
///
/// Two is the threshold because one zero capacity is routinely an honest
/// "unlimited" or an initialised counter, while a literal that zeroes several
/// at once is a lane that has been configured out of existence — which is
/// exactly what `ConstrainedZonotopeTailLpLimits { max_constraints: 0,
/// max_constraint_elements: 0, max_constraint_nonzeros: 0 }` did to the
/// metaroom constrained-zonotope lane while its name went on saying
/// "constrained".
fn capability_zero_literals(sources: &[Scanned]) -> BTreeMap<(String, String), usize> {
    let mut out: BTreeMap<(String, String), usize> = BTreeMap::new();
    for file in sources {
        for zeros in zero_capacity_groups(&file.blanked) {
            let key = (file.path.clone(), zeros.join("+"));
            *out.entry(key).or_insert(0) += 1;
        }
    }
    out
}

/// Group `field: value,` initialisers by their innermost brace scope and
/// return, for each scope with two or more zeroed capacity fields, the sorted
/// field names.
fn zero_capacity_groups(src: &str) -> Vec<Vec<String>> {
    let mut scopes: Vec<Vec<String>> = vec![Vec::new()];
    let mut stack: Vec<usize> = vec![0];
    for raw_line in src.lines() {
        for byte in raw_line.bytes() {
            match byte {
                b'{' => {
                    scopes.push(Vec::new());
                    stack.push(scopes.len() - 1);
                }
                b'}' if stack.len() > 1 => {
                    stack.pop();
                }
                _ => {}
            }
        }
        let trimmed = raw_line.trim();
        let Some((name, value)) = trimmed.split_once(':') else {
            continue;
        };
        let name = name.trim();
        if name.is_empty()
            || !name.bytes().all(is_ident_byte)
            || name.starts_with(|c: char| c.is_ascii_digit())
        {
            continue;
        }
        let value = value.trim().trim_end_matches(',').trim();
        if value != "0" || !is_capacity_field(name) {
            continue;
        }
        // A `{` opened on this same line already pushed its scope, so the
        // initialiser belongs to the current innermost scope.
        let current = *stack.last().expect("scope stack is never empty");
        scopes[current].push(name.to_string());
    }
    scopes
        .into_iter()
        .filter(|fields| fields.len() >= 2)
        .map(|mut fields| {
            fields.sort();
            fields.dedup();
            fields
        })
        .filter(|fields| fields.len() >= 2)
        .collect()
}

// ---------------------------------------------------------------------------
// The shipped yaml surface
// ---------------------------------------------------------------------------

/// One key actually SET by a shipped yaml.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct ArmedKey {
    /// Repo-relative yaml path, e.g. `configs/vnncomp25/cgan_2023.yaml`.
    file: String,
    /// Dotted key path, e.g. `bab.branching.input_split.alpha_iteration`.
    key: String,
    /// Raw value token as written, e.g. `1` or `true`. NOT coerced: a yaml
    /// `true` and a yaml `1` are different armings and conflating them would
    /// make `clip.relaxed: true` indistinguishable from an iteration count.
    value: String,
    line: usize,
}

/// Every armed key across every shipped config year, plus any yaml line whose
/// shape this scanner cannot classify.
///
/// The second half is the fail-closed half: an unrecognised line must be a
/// VIOLATION, not a skip, or the scanner silently stops covering a file the
/// day somebody adds a shape to it.
fn armed_keys(configs_root: &Path) -> (Vec<ArmedKey>, Vec<String>) {
    let mut armed = Vec::new();
    let mut unclassified = Vec::new();
    for year in CONFIG_YEARS {
        let dir = configs_root.join(year);
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        let mut paths: Vec<PathBuf> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "yaml" || x == "yml"))
            .collect();
        paths.sort();
        for path in paths {
            let relative = format!(
                "configs/{year}/{}",
                path.file_name().unwrap_or_default().to_string_lossy()
            );
            let Ok(text) = std::fs::read_to_string(&path) else {
                unclassified.push(format!("{relative}: unreadable"));
                continue;
            };
            parse_yaml_keys(&relative, &text, &mut armed, &mut unclassified);
        }
    }
    armed.sort();
    (armed, unclassified)
}

/// Indentation-driven key-path extractor for the shipped preset dialect.
///
/// The presets are plain nested maps of scalars plus a handful of string
/// sequences; this handles exactly that and REFUSES anything else. Validated
/// against a real YAML parser over all 50 shipped files: identical key sets.
fn parse_yaml_keys(
    file: &str,
    text: &str,
    armed: &mut Vec<ArmedKey>,
    unclassified: &mut Vec<String>,
) {
    let mut stack: Vec<(usize, String)> = Vec::new();
    let mut sequence_owner: Option<String> = None;
    for (index, raw) in text.lines().enumerate() {
        let line = index + 1;
        if raw.trim().is_empty() || raw.trim_start().starts_with('#') || raw.trim() == "---" {
            continue;
        }
        if raw.contains('\t') {
            unclassified.push(format!("{file}:{line}: tab indentation is not supported"));
            continue;
        }
        let indent = raw.len() - raw.trim_start().len();
        let body = raw.trim_start();

        if let Some(item) = body.strip_prefix("- ") {
            // A sequence item belongs to the nearest shallower open key, which
            // is therefore a LEAF holding a sequence rather than a sub-map.
            if stack.is_empty() {
                unclassified.push(format!("{file}:{line}: sequence item with no owning key"));
                continue;
            }
            let path = dotted(&stack);
            if sequence_owner.as_deref() != Some(path.as_str()) {
                armed.push(ArmedKey {
                    file: file.to_string(),
                    key: path.clone(),
                    value: format!("[{}]", item.trim()),
                    line,
                });
                sequence_owner = Some(path);
            }
            continue;
        }

        let Some((name, rest)) = body.split_once(':') else {
            unclassified.push(format!(
                "{file}:{line}: not `key: value` and not a sequence item: {body:?}"
            ));
            continue;
        };
        // alpha-beta-CROWN compatibility spellings use `-` (`alpha-crown`).
        if name.is_empty()
            || !name
                .bytes()
                .all(|b| is_ident_byte(b) || b == b'-' || b == b'.')
        {
            unclassified.push(format!("{file}:{line}: unsupported key token {name:?}"));
            continue;
        }
        while stack.last().is_some_and(|(i, _)| *i >= indent) {
            stack.pop();
        }
        let value = strip_yaml_comment(rest);
        if value.is_empty() {
            stack.push((indent, name.to_string()));
            sequence_owner = None;
        } else {
            let mut path = dotted(&stack);
            if path.is_empty() {
                path = name.to_string();
            } else {
                path.push('.');
                path.push_str(name);
            }
            armed.push(ArmedKey {
                file: file.to_string(),
                key: path,
                value: value.to_string(),
                line,
            });
        }
    }
}

fn dotted(stack: &[(usize, String)]) -> String {
    stack
        .iter()
        .map(|(_, name)| name.as_str())
        .collect::<Vec<_>>()
        .join(".")
}

/// Drop a trailing `# comment`, honouring a quoted value.
fn strip_yaml_comment(rest: &str) -> &str {
    let trimmed = rest.trim();
    if trimmed.starts_with('"') || trimmed.starts_with('\'') {
        let quote = trimmed.as_bytes()[0];
        if let Some(close) = trimmed[1..].find(quote as char) {
            return trimmed[..close + 2].trim();
        }
        return trimmed;
    }
    match trimmed.find(" #") {
        Some(at) => trimmed[..at].trim(),
        None => trimmed,
    }
}

// ---------------------------------------------------------------------------
// Manifest parsing
// ---------------------------------------------------------------------------

/// A tab-separated frontier row: the leading columns form the identity, the
/// last two are `class` and `note`.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ManifestRow {
    identity: Vec<String>,
    count: usize,
    class: String,
    note: String,
}

fn parse_manifest(
    raw: &str,
    identity_columns: usize,
    label: &str,
) -> BTreeMap<Vec<String>, ManifestRow> {
    let mut out = BTreeMap::new();
    for (index, line) in raw.lines().enumerate() {
        let trimmed = line.trim_end();
        if trimmed.trim().is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = trimmed.split('\t').collect();
        assert!(
            fields.len() == identity_columns + 3,
            "{label} line {}: expected {} tab-separated columns, found {}: {trimmed:?}",
            index + 1,
            identity_columns + 3,
            fields.len()
        );
        let identity: Vec<String> = fields[..identity_columns]
            .iter()
            .map(|s| (*s).to_string())
            .collect();
        let count = fields[identity_columns]
            .parse::<usize>()
            .unwrap_or_else(|_| panic!("{label} line {}: count is not an integer", index + 1));
        let row = ManifestRow {
            identity: identity.clone(),
            count,
            class: fields[identity_columns + 1].to_string(),
            note: fields[identity_columns + 2].to_string(),
        };
        assert!(
            !row.note.trim().is_empty(),
            "{label} line {}: every row must carry a note; an unexplained row is the \
             silent state this gate exists to end",
            index + 1
        );
        assert!(
            out.insert(identity, row).is_none(),
            "{label} line {}: duplicate identity",
            index + 1
        );
    }
    out
}

/// Compare a measured frontier against its manifest and describe EVERY
/// difference. Additions and removals both fail: slack in either direction
/// lets a fixed debt come back, or a new one hide behind a retired one.
fn diff_frontier(
    measured: &BTreeMap<Vec<String>, usize>,
    manifest: &BTreeMap<Vec<String>, ManifestRow>,
    what: &str,
) -> Vec<String> {
    let mut violations = Vec::new();
    for (identity, count) in measured {
        match manifest.get(identity) {
            None => violations.push(format!(
                "UNDECLARED {what}: {}  (x{count})\n    Add a row to the frontier file with a \
                 class and a note that says WHY this cannot silently disconnect an armed \
                 preset key — or remove the site.",
                identity.join("  ")
            )),
            Some(row) if row.count != *count => violations.push(format!(
                "COUNT CHANGED for {what}: {}  declared {}, measured {count}",
                identity.join("  "),
                row.count
            )),
            Some(_) => {}
        }
    }
    for identity in manifest.keys() {
        if !measured.contains_key(identity) {
            violations.push(format!(
                "STALE {what} row: {}  no longer present in the tree. Delete the row in the \
                 same commit that removed the site, so the frontier keeps protecting the \
                 ground that was just won.",
                identity.join("  ")
            ));
        }
    }
    violations
}

/// Source files that legitimately contain no code, with their EXACT byte
/// count.
///
/// A code-free `.rs` is normally the truncation bug: an empty module compiles,
/// so a file reached only by `mod foo;` or `#[path = "..."]` vanishes in
/// silence and takes its `#[test]` functions with it. These three are the
/// deliberate cases, each verified byte-identical to `origin/main` and each
/// saying in its own text why it is present.
///
/// The byte count is what makes this a ratchet rather than a blanket
/// exemption: if one of these gains or loses a byte, the declaration fails and
/// somebody looks. A truncation of any OTHER file still fails immediately.
const EMPTY_BY_DESIGN: &[(&str, usize, &str)] = &[
    (
        "crates/ny-propagate/src/beta_crown/engine/graph/gpu_bab/grouped_disjunctive.rs",
        477,
        "declared by `gpu_bab/mod.rs:45` as `mod grouped_disjunctive;` and says so in its own \
         header: the file exists to preserve the module boundary for the #4398 follow-on \
         without changing current single-objective behaviour.",
    ),
    (
        "crates/ny-propagate/src/elimination_tests.rs",
        66,
        "included by `elimination.rs:457` via `#[path = \"elimination_tests.rs\"]`; the file's \
         one line says unit tests live here when that path is active. An EMPTY test module \
         behind a `#[path]` include is precisely the silent shape this check exists for, so it \
         is declared rather than tolerated.",
    ),
    (
        "crates/ny-propagate/src/equivalence_tests.rs",
        66,
        "included by `equivalence.rs:376` via `#[path = \"equivalence_tests.rs\"]`; same shape \
         and same reasoning as elimination_tests.rs.",
    ),
];

/// The registry's own class, as a stable manifest token.
const fn class_token(class: Class) -> &'static str {
    match class {
        Class::VerdictAffecting => "verdict-affecting",
        Class::SafeToSearch => "safe-to-search",
        Class::Unsafe => "unsafe",
    }
}

fn report(violations: &[String], headline: &str) {
    assert!(
        violations.is_empty(),
        "{headline}\n\n{} violation(s), ALL of them listed (this gate never reveals one debt \
         per run):\n\n{}\n",
        violations.len(),
        violations.join("\n\n")
    );
}

// ---------------------------------------------------------------------------
// Anti-vacuity
// ---------------------------------------------------------------------------

#[test]
fn the_scan_actually_scans_something() {
    // Every check below is a comparison against a manifest, so a scanner that
    // silently returns nothing turns the whole file into a no-op that reports
    // "STALE" once and then passes forever. Guard the inputs first.
    let root = workspace_root();
    let surface = delivery_surface(&root.join(APPLY_RS));
    assert!(
        surface.len() >= 100,
        "delivery surface is {} fields; {APPLY_RS} is known to write well over a hundred. \
         The scanner is broken, not the tree.",
        surface.len()
    );
    for field in [
        "enable_relaxed_clip",
        "input_split_alpha_iteration",
        "timeout",
    ] {
        assert!(
            surface.contains(field),
            "delivery surface is missing `{field}`, which {APPLY_RS} demonstrably writes"
        );
    }

    let mut files = Vec::new();
    let crates_root = root.join("crates");
    collect_rs(&crates_root, &crates_root, &mut files);
    assert!(
        files.len() > 1_000,
        "expected to walk the whole crates/ tree, saw only {} .rs files",
        files.len()
    );

    let (armed, unclassified) = armed_keys(&root.join("configs"));
    assert!(
        armed.len() > 500,
        "expected hundreds of armed preset settings across the shipped years, found {}",
        armed.len()
    );
    report(
        &unclassified,
        "SHIPPED YAML HAS A SHAPE THIS SCANNER CANNOT CLASSIFY.\n\nAn unclassifiable line is a \
         key the arrival checks below silently stop covering, so it fails rather than skips. \
         Either teach `parse_yaml_keys` the shape (and re-validate it against a real YAML \
         parser over every shipped file), or write the key in the supported dialect.",
    );
}

// ---------------------------------------------------------------------------
// Mode 1 — runtime clearing
// ---------------------------------------------------------------------------

#[test]
fn preset_override_frontier_is_exact() {
    let root = workspace_root();
    let surface = delivery_surface(&root.join(APPLY_RS));
    let mut sites = override_sites(workspace_sources(), &surface);
    for (identity, count) in config_rebuild_sites(workspace_sources()) {
        *sites.entry(identity).or_insert(0) += count;
    }
    let measured: BTreeMap<Vec<String>, usize> = sites
        .into_iter()
        .map(|((file, field), count)| (vec![file, field], count))
        .collect();
    let manifest = parse_manifest(OVERRIDE_MANIFEST, 2, "preset_override_frontier.tsv");

    let violations = diff_frontier(&measured, &manifest, "override site");
    report(
        &violations,
        "PRESET OVERRIDE FRONTIER CHANGED.\n\nA write to a field that `preset::apply` fills from \
         a shipped yaml, made anywhere outside the resolver, can silently replace what the yaml \
         asked for before the lane reads it. That is how `bab.clip.relaxed: true` shipped armed \
         on six cgan presets and ran cleared: `config_with_disjunctive_clip_routes_disabled` \
         zeroes `enable_relaxed_clip` on a clone of the resolved config whenever the graph is \
         not batch-stack-safe, which a BatchNorm guarantees.\n\nDeclaring a site is cheap and is \
         the point: it forces somebody to write down whether an armed key can die there.",
    );
}

#[test]
fn every_delivered_field_has_a_consumer() {
    // The FIRST question about an armed key, before any of the arrival
    // machinery: does anything read it at all? A field the resolver writes and
    // nobody reads is a preset key that parses, validates, type-checks, appears
    // in every summary — and does nothing.
    //
    // This is a hard zero, not a ratchet, because the debt IS zero: all 125
    // delivery-surface fields are read outside the resolver today. A new preset
    // key with no reader therefore cannot land, which is the state
    // `preset::contract` has to describe by hand for the fields that reached
    // this condition before anyone was checking.
    let root = workspace_root();
    let surface = delivery_surface(&root.join(APPLY_RS));
    let consumed = consumed_fields(workspace_sources(), &surface);
    let violations: Vec<String> = surface
        .iter()
        .filter(|field| !consumed.contains(*field))
        .map(|field| {
            format!(
                "`{field}` is written by {APPLY_RS} and READ NOWHERE in first-party non-test \
                 source.\n    Either wire it to the lane that is supposed to act on it, or \
                 remove the preset key — and if it is deliberate compatibility with \
                 alpha-beta-CROWN's key set, record it in \
                 crates/ny-cli/src/preset/contract.rs so `NY_PRESET_STRICT` reports it \
                 instead of the config silently lying."
            )
        })
        .collect();
    report(
        &violations,
        "A PRESET KEY LANDS IN A CONFIG FIELD THAT NOTHING READS.",
    );
}

#[test]
fn every_runtime_clearing_site_names_its_guard() {
    // A `runtime-clear` row is the dangerous class: it does not derive a value
    // from the preset, it REPLACES one. Its note must name the condition under
    // which it fires, or the row records that a hazard exists without telling
    // the next reader whether their benchmark hits it.
    let manifest = parse_manifest(OVERRIDE_MANIFEST, 2, "preset_override_frontier.tsv");
    let mut violations = Vec::new();
    for row in manifest.values() {
        if row.class != "runtime-clear" && row.class != "config-rebuild" {
            continue;
        }
        if !row.note.contains("when=") {
            violations.push(format!(
                "{} row {} has no `when=` clause in its note. Name the guard that fires it — \
                 `when=unconditional` is a legitimate and important answer — so a reader can \
                 tell whether their rows are affected without re-deriving the control flow: {}",
                row.class,
                row.identity.join("  "),
                row.note
            ));
        }
    }
    report(
        &violations,
        "A SITE THAT REPLACES AN ARMED VALUE DOES NOT NAME ITS GUARD.\n\nThis is the column \
         that turns the frontier from an inventory into an answer: `when=` is what tells the \
         next person whether the null they just measured was taken through a live mechanism \
         or a cleared one.",
    );
}

#[test]
fn override_frontier_classes_are_from_the_closed_set() {
    const CLASSES: &[&str] = &[
        // Replaces an armed value at runtime. The dangerous class.
        "runtime-clear",
        // Rebuilds a whole resolved config from `::default()`, dropping every
        // preset value at once.
        "config-rebuild",
        // The documented CLI-over-preset precedence layer.
        "cli-precedence",
        // A non-competition language binding.
        "binding",
        // Derives a scoped sub-solve config from the parent's resolved value.
        "derived-scope",
        // Writes a value computed FROM the preset value, not over it.
        "derived-value",
        // Telemetry or statistics, not a verifier config.
        "telemetry",
        // Measured, reviewed, and not yet adjudicated into one of the above.
        // May only ever shrink.
        "legacy-unreviewed",
    ];
    let manifest = parse_manifest(OVERRIDE_MANIFEST, 2, "preset_override_frontier.tsv");
    let mut violations = Vec::new();
    for row in manifest.values() {
        if !CLASSES.contains(&row.class.as_str()) {
            violations.push(format!(
                "row {} has class {:?}, which is not in the closed set {CLASSES:?}",
                row.identity.join("  "),
                row.class
            ));
        }
    }
    report(&violations, "OVERRIDE FRONTIER HAS AN UNKNOWN CLASS.");
}

/// `legacy-unreviewed` is ACCEPTED DEBT: a site the census measured and nobody
/// has yet adjudicated. Accepted debt only stays honest if it is capped, and
/// these are EXACT, not budgets.
///
/// A new site cannot land here at all — an undeclared identity fails the
/// frontier diff before it reaches this test — so the only thing these numbers
/// govern is whether adjudication progress is recorded. Lower one in the commit
/// that adjudicates a row. Never raise one.
///
/// 2026-08-26  Landing measurement.
const BASELINE_LEGACY_UNREVIEWED: &[(&str, usize)] = &[
    ("preset_override_frontier.tsv", 0),
    ("preset_capability_zero_frontier.tsv", 0),
];

#[test]
fn unreviewed_debt_is_exactly_the_declared_baseline() {
    // Both directions, and ALL of them in one run: a ratchet that only checks
    // the ceiling silently stops protecting ground that has already been won,
    // which is how the raw-read baseline drifted twice.
    let manifests: [(&str, &str, usize); 2] = [
        ("preset_override_frontier.tsv", OVERRIDE_MANIFEST, 2),
        (
            "preset_capability_zero_frontier.tsv",
            CAPABILITY_ZERO_MANIFEST,
            2,
        ),
    ];
    let mut violations = Vec::new();
    for (name, raw, identity_columns) in manifests {
        let Some(&(_, baseline)) = BASELINE_LEGACY_UNREVIEWED.iter().find(|(n, _)| *n == name)
        else {
            panic!("{name} has no declared baseline");
        };
        let n = parse_manifest(raw, identity_columns, name)
            .values()
            .filter(|row| row.class == "legacy-unreviewed")
            .count();
        if n > baseline {
            violations.push(format!(
                "{name}: unreviewed debt ROSE to {n}, baseline {baseline}. A site may not be \
                 reclassified INTO accepted debt; classify it."
            ));
        } else if n < baseline {
            violations.push(format!(
                "{name}: unreviewed debt has FALLEN to {n} but the baseline is still \
                 {baseline}. Lower it to {n} in this commit and record what was adjudicated."
            ));
        }
    }
    report(&violations, "UNREVIEWED-DEBT BASELINE IS STALE.");
}

// ---------------------------------------------------------------------------
// Mode 2 — requested vs delivered
// ---------------------------------------------------------------------------

#[test]
fn delivery_capped_keys_are_adjudicated_before_arming() {
    let root = workspace_root();
    let (armed, _) = armed_keys(&root.join("configs"));
    let manifest = parse_manifest(DEAD_VALUE_MANIFEST, 3, "preset_dead_value_frontier.tsv");

    let mut violations = Vec::new();
    for capped in DELIVERY_CAPPED {
        for key in armed.iter().filter(|k| k.key == capped.key) {
            let identity = vec![key.file.clone(), key.key.clone(), key.value.clone()];
            if !manifest.contains_key(&identity) {
                violations.push(format!(
                    "{}:{} arms `{}` = {} with NO adjudication row.\n    delivered value is \
                     capped by: {}\n    look at: {}\n    last measured: {}\n    A measurement \
                     taken through this key is uninterpretable until somebody records what the \
                     lane actually received; add a row to \
                     data/preset_dead_value_frontier.tsv naming the evidence field and the \
                     number it showed.",
                    key.file,
                    key.line,
                    capped.key,
                    key.value,
                    capped.capped_by,
                    capped.evidence,
                    capped.measured
                ));
            }
        }
    }
    report(
        &violations,
        "A KEY WHOSE DELIVERED VALUE IS NOT ITS CONFIGURED VALUE WAS ARMED WITHOUT \
         ADJUDICATION.\n\n`margin_row.beta_columns: 8` requested eight columns and delivered \
         one on every call, because the classwise margin-row route builds a single-class \
         MarginBatch and the live set is truncated to `min(configured, nf)`. Per-column beta \
         was then a provable IDENTITY with shared beta — and the A/B that measured it read as \
         a negative result about per-column beta rather than as a null from a mechanism that \
         never widened.",
    );
}

#[test]
fn delivery_capped_keys_keep_a_live_requested_vs_delivered_observable() {
    // A capped key with no evidence symbol cannot be adjudicated at all: there
    // would be nothing for the reviewer to look at, so the obligation above
    // would be unsatisfiable and the honest move is to fail here instead.
    //
    // And the symbols are checked to EXIST, not merely to be written down. A
    // deleted or renamed evidence field is how a requested-vs-delivered gap
    // goes back to being invisible: the `[beta-pc] armed (… cols=8)` banner is
    // still quoted by the margin-row beta runbook and no longer exists in the
    // tree, so the "requested" half of that pair is already unobservable on
    // stderr. This is the one thing a static gate CAN guarantee about a runtime
    // question.
    let sources = workspace_sources();
    let mut violations = Vec::new();
    for capped in DELIVERY_CAPPED {
        if capped.evidence.trim().is_empty() {
            violations.push(format!("{} has no evidence symbol", capped.key));
        }
        if capped.capped_by.trim().is_empty() {
            violations.push(format!("{} does not say what caps it", capped.key));
        }
        if capped.measured.trim().is_empty() {
            violations.push(format!("{} records no measurement", capped.key));
        }
        for symbol in capped.observable {
            let found = sources
                .iter()
                .any(|file| contains_identifier(&file.blanked, symbol));
            if !found {
                violations.push(format!(
                    "{}: the observable `{symbol}` is GONE from first-party non-test source. \
                     Without it nobody can tell what the lane actually received, so the key \
                     becomes unmeasurable and every result taken through it becomes \
                     uninterpretable. Restore it, or point this registry entry at whatever \
                     replaced it.\n    the pair is: {}",
                    capped.key, capped.evidence
                ));
            }
        }
    }
    report(
        &violations,
        "A DELIVERY-CAPPED KEY HAS LOST THE OBSERVABLE THAT MAKES IT MEASURABLE.",
    );
}

/// Whole-identifier search: `beta_columns` must not match
/// `beta_columns_max_candidates`.
fn contains_identifier(haystack: &str, needle: &str) -> bool {
    let bytes = haystack.as_bytes();
    let mut from = 0usize;
    while let Some(at) = haystack[from..].find(needle) {
        let start = from + at;
        let end = start + needle.len();
        from = start + 1;
        let before_ok = start == 0 || !is_ident_byte(bytes[start - 1]);
        let after_ok = bytes.get(end).is_none_or(|b| !is_ident_byte(*b));
        if before_ok && after_ok {
            return true;
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Mode 3 — semantically dead values
// ---------------------------------------------------------------------------

#[test]
fn dead_value_armings_are_exactly_the_declared_debt() {
    let root = workspace_root();
    let (armed, _) = armed_keys(&root.join("configs"));
    let manifest = parse_manifest(DEAD_VALUE_MANIFEST, 3, "preset_dead_value_frontier.tsv");

    let mut measured: BTreeMap<Vec<String>, usize> = BTreeMap::new();
    let mut reasons: BTreeMap<Vec<String>, &'static str> = BTreeMap::new();
    for key in &armed {
        if let Some(reason) = dead_value_reason(&key.key, &key.value) {
            let identity = vec![key.file.clone(), key.key.clone(), key.value.clone()];
            *measured.entry(identity.clone()).or_insert(0) += 1;
            reasons.insert(identity, reason);
        }
    }

    // Rows that exist only to adjudicate a DELIVERY_CAPPED arming are not
    // dead-value findings and must not be reported as stale here.
    let capped_keys: BTreeSet<&str> = DELIVERY_CAPPED.iter().map(|c| c.key).collect();
    let dead_only: BTreeMap<Vec<String>, ManifestRow> = manifest
        .into_iter()
        .filter(|(identity, _)| {
            identity
                .get(1)
                .is_none_or(|key| !capped_keys.contains(key.as_str()))
        })
        .collect();

    let mut violations = diff_frontier(&measured, &dead_only, "dead-value arming");
    for (identity, reason) in &reasons {
        if !dead_only.contains_key(identity) {
            let key = identity.get(1).map_or("?", String::as_str);
            let value = identity.get(2).map_or("?", String::as_str);
            violations.push(format!("    ^ {key} = {value}: {reason}"));
        }
    }
    report(
        &violations,
        "A SHIPPED PRESET ARMS A SEMANTICALLY DEAD VALUE THAT IS NOT DECLARED.\n\nThese are \
         legal values that disable the thing they appear to configure. Six shipped presets set \
         `alpha_iteration: 1`, which reads as `one warm refinement step` and performs ZERO: the \
         warm-start loop breaks on `iter == iterations - 1` before the first SPSA gradient, so \
         the whole tree runs on the root's frozen alphas. Declaring the arming does not fix it \
         — it stops the next person measuring through it and banking the null.",
    );
}

/// The co-arming state of every tracked dormant pair, per shipped yaml.
///
/// `live` when the yaml sets the gate to the waking value, `dormant` otherwise.
fn dormant_states(armed: &[ArmedKey]) -> BTreeMap<Vec<String>, usize> {
    let setting = |file: &str, key: &str| -> Option<&str> {
        armed
            .iter()
            .find(|k| k.file == file && k.key == key)
            .map(|k| k.value.as_str())
    };
    let files: BTreeSet<&str> = armed.iter().map(|k| k.file.as_str()).collect();

    let mut out = BTreeMap::new();
    for pair in DORMANT_UNTIL {
        for file in &files {
            // The dormancy is a property of the ROUTE. A yaml that does not
            // take it is not tracked, because the value is live there.
            if setting(file, pair.route.0) != Some(pair.route.1) {
                continue;
            }
            // Resolve the alias the way the resolver does: last write wins.
            let Some((key, value)) = pair
                .key_aliases
                .iter()
                .filter_map(|alias| setting(file, alias).map(|v| (*alias, v)))
                .next_back()
            else {
                continue; // this yaml does not carry the value at all
            };
            let state = if setting(file, pair.gate) == Some(pair.live_when) {
                "live"
            } else {
                "dormant"
            };
            out.insert(
                vec![
                    (*file).to_string(),
                    key.to_string(),
                    value.to_string(),
                    state.to_string(),
                ],
                1,
            );
        }
    }
    out
}

#[test]
fn dormant_value_states_are_exactly_the_declared_frontier() {
    let root = workspace_root();
    let (armed, _) = armed_keys(&root.join("configs"));
    let measured = dormant_states(&armed);
    let manifest = parse_manifest(DORMANT_MANIFEST, 4, "preset_dormant_frontier.tsv");
    let violations = diff_frontier(&measured, &manifest, "dormant-value state");
    report(
        &violations,
        "A DORMANT PRESET VALUE CHANGED STATE.\n\nThe row identity ends in `dormant` or `live`, \
         so flipping the GATE key flips the row and fails this gate. That is the entire \
         purpose: the dangerous moment for a dormant value is not when it is written, it is \
         when a neighbouring key changes months later and an unexercised number takes effect \
         at full size.\n\n`bab.batch_size: 64` sat unconsumed until `reorder_bab` flipped, and \
         then cost an 11.7x memory regression — 6.3 GB to 73.4 GB — on a unified-memory host \
         where over-admission is a global OOM, not a refusal. Nothing about that value had ever \
         been measured, because it had never applied to anything.\n\nIf a row moved to `live`: \
         MEASURE THE VALUE BEFORE UPDATING THE ROW. If it moved to `dormant`, the mechanism \
         just went dark and any result you were about to attribute to it is VOID.",
    );
}

#[test]
fn dormant_registry_entries_are_actionable() {
    let mut violations = Vec::new();
    for pair in DORMANT_UNTIL {
        let name = pair.key_aliases.last().copied().unwrap_or("<no key>");
        if pair.key_aliases.is_empty() {
            violations.push("a DORMANT_UNTIL entry has no key".to_string());
        }
        if pair.key_aliases.contains(&pair.gate) {
            violations.push(format!("{name}: a key cannot gate itself"));
        }
        if pair.route.0 == pair.gate {
            violations.push(format!(
                "{name}: the route selector and the gate are the same key, so the entry cannot \
                 distinguish `off this route` from `dormant on it`"
            ));
        }
        for (field, what) in [
            (pair.live_when, "live_when"),
            (pair.why, "why"),
            (pair.measured, "measured"),
        ] {
            if field.trim().is_empty() {
                violations.push(format!("{name}: {what} is empty"));
            }
        }
        // The `why` must cite the site that discards the value, or the entry is
        // an assertion rather than a finding.
        if !pair.why.contains(".rs:") {
            violations.push(format!(
                "{name}: `why` cites no source line. A dormancy claim with no citation is the \
                 same unverifiable assertion as a preset rationale."
            ));
        }
    }
    report(&violations, "A DORMANT_UNTIL ENTRY IS NOT ACTIONABLE.");
}

#[test]
fn capability_zero_frontier_is_exact() {
    let measured: BTreeMap<Vec<String>, usize> = capability_zero_literals(workspace_sources())
        .into_iter()
        .map(|((file, fields), count)| (vec![file, fields], count))
        .collect();
    let manifest = parse_manifest(
        CAPABILITY_ZERO_MANIFEST,
        2,
        "preset_capability_zero_frontier.tsv",
    );
    let violations = diff_frontier(&measured, &manifest, "zero-capacity literal");
    report(
        &violations,
        "A CAPABILITY STRUCT IS BUILT WITH ITS CAPACITIES AT ZERO.\n\nThe metaroom \
         constrained-zonotope lane ran as plain DeepZ for its entire life because \
         `ConstrainedZonotopeTailLpLimits` was constructed with `max_constraints: 0`, \
         `max_constraint_elements: 0` and `max_constraint_nonzeros: 0` in an unconditional \
         struct literal: every domain carrying a constraint row was refused at plan time, so \
         the lane could never price a constraint. Nothing about that is visible from the call \
         site, the type name, or any config.",
    );
}

#[test]
fn dead_value_and_capability_zero_classes_are_from_their_closed_sets() {
    const DORMANT_CLASSES: &[&str] = &["dormant", "dormant-woken"];
    const DEAD_VALUE_CLASSES: &[&str] = &["dead-value", "capped-adjudicated"];
    const CAPABILITY_ZERO_CLASSES: &[&str] = &[
        // The zeros make the lane a no-op. The note must say what it degrades
        // to, because the type name will keep claiming otherwise.
        "lane-disabled",
        // Counters and statistics initialised to zero.
        "telemetry-init",
        // Zero genuinely means "no limit" or "not applicable" here.
        "not-a-capacity",
        // Measured and not yet adjudicated. Capped.
        "legacy-unreviewed",
    ];
    let mut violations = Vec::new();
    for row in parse_manifest(DEAD_VALUE_MANIFEST, 3, "preset_dead_value_frontier.tsv").values() {
        if !DEAD_VALUE_CLASSES.contains(&row.class.as_str()) {
            violations.push(format!(
                "dead-value row {} has class {:?}, not in {DEAD_VALUE_CLASSES:?}",
                row.identity.join("  "),
                row.class
            ));
        }
    }
    for row in parse_manifest(
        CAPABILITY_ZERO_MANIFEST,
        2,
        "preset_capability_zero_frontier.tsv",
    )
    .values()
    {
        if !CAPABILITY_ZERO_CLASSES.contains(&row.class.as_str()) {
            violations.push(format!(
                "capability-zero row {} has class {:?}, not in {CAPABILITY_ZERO_CLASSES:?}",
                row.identity.join("  "),
                row.class
            ));
        }
    }
    for row in parse_manifest(DORMANT_MANIFEST, 4, "preset_dormant_frontier.tsv").values() {
        if !DORMANT_CLASSES.contains(&row.class.as_str()) {
            violations.push(format!(
                "dormant row {} has class {:?}, not in {DORMANT_CLASSES:?}",
                row.identity.join("  "),
                row.class
            ));
        }
        // The class must agree with the state in the identity, or the file can
        // say `dormant` about a row the tree calls live.
        let state = row.identity.get(3).map(String::as_str);
        let expected = match state {
            Some("live") => "dormant-woken",
            Some("dormant") => "dormant",
            _ => {
                violations.push(format!(
                    "dormant row {} has an unrecognised state column {state:?}",
                    row.identity.join("  ")
                ));
                continue;
            }
        };
        if row.class != expected {
            violations.push(format!(
                "dormant row {} is in state {state:?} but classed {:?}; expected {expected:?}. \
                 The class and the state are two statements about the same fact and must not \
                 disagree.",
                row.identity.join("  "),
                row.class
            ));
        }
    }
    report(&violations, "A FRONTIER ROW HAS AN UNKNOWN CLASS.");
}

// ---------------------------------------------------------------------------
// Promotion is not arming
// ---------------------------------------------------------------------------

#[test]
fn unarmed_promotions_are_exactly_the_declared_set() {
    let root = workspace_root();
    let (armed, _) = armed_keys(&root.join("configs"));
    let armed_paths: BTreeSet<&str> = armed.iter().map(|k| k.key.as_str()).collect();

    // Identity is (preset key, NY_* axis name) so the manifest names BOTH ends
    // of the delivery route: a reader who sees a key here can jump straight to
    // the declaration without re-deriving the mapping.
    let mut measured: BTreeMap<Vec<String>, usize> = BTreeMap::new();
    let mut declared_class: BTreeMap<Vec<String>, &'static str> = BTreeMap::new();
    for axis in ny_levers::space::axes() {
        let Deliver::PresetKey(key) = axis.deliver else {
            continue;
        };
        if !armed_paths.contains(key) {
            let identity = vec![key.to_string(), axis.name.to_string()];
            *measured.entry(identity.clone()).or_insert(0) += 1;
            declared_class.insert(identity, class_token(axis.class));
        }
    }
    let manifest = parse_manifest(
        UNARMED_PROMOTION_MANIFEST,
        2,
        "preset_unarmed_promotions.tsv",
    );
    let mut violations = diff_frontier(&measured, &manifest, "unarmed promotion");
    // The class column is checked against the LIVE registry, not just carried:
    // a reclassification that quietly moves a lever out of VerdictAffecting is
    // exactly the move the reachability ratchet's own header warns about.
    for (identity, expected) in &declared_class {
        if let Some(row) = manifest.get(identity) {
            if row.class != *expected {
                violations.push(format!(
                    "CLASS DRIFT for {}: manifest says {:?}, the registry now says {expected:?}. \
                     If the reclassification is correct, update the row and say why in the \
                     note; if it is not, the axis is wrong.",
                    identity.join("  "),
                    row.class
                ));
            }
        }
    }
    report(
        &violations,
        "THE SET OF PROMOTED-BUT-UNARMED PRESET KEYS CHANGED.\n\nPROMOTION IS NOT ARMING. A \
         typed preset key makes a mechanism DELIVERABLE; it does not put a value down the \
         wire. `margin_row.tableau_f32` was implemented, correct, declared, bridged and \
         reachable — and set by no shipped yaml at all, so it was worth exactly zero on a \
         scored run until somebody happened to look.\n\nThis set is checked in so that state \
         is READABLE instead of discoverable. Adding a promotion adds a line here (deliberate, \
         one line). Arming a key REMOVES a line here, in the commit that carries the \
         measurement justifying it.",
    );
}

// ---------------------------------------------------------------------------
// The sixth shape: a source file that compiles to nothing
// ---------------------------------------------------------------------------

#[test]
fn no_source_file_is_empty() {
    // A `.rs` truncated to zero bytes is a VALID EMPTY MODULE. A file reached
    // only by `mod foo;` therefore vanishes in silence — green build, green
    // suite, tests simply gone. Found live in the shared tree, where
    // `graph_pgd_exact_vjp_evidence.rs` was 0 bytes against 19,652 at origin
    // and carried two `#[test]` functions. `git status` shows these as an
    // ordinary `M`, so nothing else in the workflow catches it.
    let root = workspace_root();
    let crates_root = root.join("crates");
    let mut files = Vec::new();
    collect_rs(&crates_root, &crates_root, &mut files);
    files.sort();

    let mut violations = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    for (relative, path) in &files {
        let Ok(text) = std::fs::read_to_string(path) else {
            continue;
        };
        let full = format!("crates/{relative}");
        if !text.is_empty() && !blank_comments_and_strings(&text).trim().is_empty() {
            // Has code. If it is declared empty-by-design, that declaration is
            // now stale and must go.
            if let Some((declared, _, _)) = EMPTY_BY_DESIGN.iter().find(|(p, _, _)| *p == full) {
                violations.push(format!(
                    "{declared} is declared EMPTY_BY_DESIGN but now HAS code. Delete the \
                     declaration in the same commit."
                ));
            }
            continue;
        }
        match EMPTY_BY_DESIGN.iter().find(|(p, _, _)| *p == full) {
            Some((_, bytes, _)) if *bytes == text.len() => {
                seen.insert(full.clone());
            }
            Some((_, bytes, _)) => violations.push(format!(
                "{full} is declared empty-by-design at {bytes} bytes but is now {}. A size \
                 change on a code-free file is either the placeholder gaining content or the \
                 file being truncated further; both need a look.",
                text.len()
            )),
            None if text.is_empty() => violations.push(format!("{full}: ZERO BYTES")),
            None => violations.push(format!(
                "{full}: {} bytes, but NOTHING outside comments — it compiles as an empty \
                 module and takes any tests it used to hold with it",
                text.len()
            )),
        }
    }
    for (path, _, _) in EMPTY_BY_DESIGN {
        if !seen.contains(*path) {
            violations.push(format!(
                "{path} is declared EMPTY_BY_DESIGN but was not found. Delete the declaration \
                 in the same commit that deleted the file."
            ));
        }
    }
    report(
        &violations,
        "A FIRST-PARTY SOURCE FILE HAS NO CODE IN IT.\n\nCompare byte sizes against \
         origin/main, not line diffs. This is the `cargo test` matched-nothing trap arriving \
         through interrupted I/O rather than through a renamed test. If the file is a \
         deliberate placeholder, add it to EMPTY_BY_DESIGN with its exact byte count — the \
         count is what makes the declaration a ratchet rather than a blanket exemption.",
    );
}

// ---------------------------------------------------------------------------
// Demonstrations — the scanners are shown to FIRE, not asserted to
// ---------------------------------------------------------------------------
//
// A ratchet that passes on a tree known to contain five instances of the bug it
// targets is worthless. Each test below reconstructs one of the five and proves
// detection against it.

#[cfg(test)]
mod demonstrations {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
            .join("preset_arrival_ratchet")
            .join(name);
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    fn write(root: &Path, relative: &str, contents: &str) {
        let path = root.join(relative);
        std::fs::create_dir_all(path.parent().expect("parent")).expect("mkdir");
        std::fs::write(path, contents).expect("write");
    }

    /// FAILURE 1 — the cgan input clip, cleared at runtime.
    ///
    /// Rebuilt to the shape it actually has in tree: a resolver that fills
    /// `enable_relaxed_clip` from `bab.clip.relaxed`, and an engine function
    /// that clones the resolved config and zeroes the field behind a graph
    /// predicate.
    #[test]
    fn demo_detects_runtime_clearing_of_an_armed_clip() {
        let root = scratch("runtime_clearing");
        write(
            &root,
            "crates/ny-cli/src/preset/apply.rs",
            r#"
pub fn apply_clip_preset(config: &mut BetaCrownConfig, preset: &ClipPreset) {
    if let Some(relaxed) = preset.relaxed {
        config.enable_relaxed_clip = relaxed;
    }
    if let Some(fresh) = preset.input_split_fresh_domain_clip {
        config.input_split_fresh_domain_clip = fresh;
    }
}
"#,
        );
        write(
            &root,
            "crates/ny-propagate/src/engine/disjunctive_multi_clause.rs",
            r#"
/// Conservatively disable reused-bound relaxed clipping whenever the graph
/// cannot soundly batch-stack input-split domains.
fn config_with_disjunctive_clip_routes_disabled(config: &BetaCrownConfig) -> BetaCrownConfig {
    let mut config = config.clone();
    config.enable_relaxed_clip = false;
    config.input_split_fresh_domain_clip = false;
    config
}
"#,
        );

        let surface = delivery_surface(&root.join("crates/ny-cli/src/preset/apply.rs"));
        assert!(
            surface.contains("enable_relaxed_clip"),
            "the resolver's own write must define the delivery surface; got {surface:?}"
        );

        let sites = override_sites(&scan_sources(&root.join("crates")), &surface);
        let key = (
            "crates/ny-propagate/src/engine/disjunctive_multi_clause.rs".to_string(),
            "enable_relaxed_clip".to_string(),
        );
        assert_eq!(
            sites.get(&key),
            Some(&1),
            "the runtime clear of an armed clip was NOT detected; sites = {sites:?}"
        );
        let fresh_key = (
            "crates/ny-propagate/src/engine/disjunctive_multi_clause.rs".to_string(),
            "input_split_fresh_domain_clip".to_string(),
        );
        assert!(
            sites.contains_key(&fresh_key),
            "the second cleared field was not detected; sites = {sites:?}"
        );

        // And it fails LOUDLY against a manifest that does not declare it.
        let measured: BTreeMap<Vec<String>, usize> = sites
            .into_iter()
            .map(|((f, x), c)| (vec![f, x], c))
            .collect();
        let violations = diff_frontier(&measured, &BTreeMap::new(), "override site");
        assert!(
            violations.iter().any(|v| v.contains("enable_relaxed_clip")),
            "an undeclared runtime clear must be reported by name; got {violations:?}"
        );
    }

    /// FAILURE 0 — the question that comes before all the others: a preset key
    /// that lands in a field NOTHING reads.
    #[test]
    fn demo_detects_a_delivered_field_with_no_consumer() {
        let root = scratch("no_consumer");
        write(
            &root,
            "crates/ny-cli/src/preset/apply.rs",
            r#"
pub fn apply(config: &mut BetaCrownConfig, preset: &BabPreset) {
    if let Some(v) = preset.wired {
        config.wired_knob = v;
    }
    if let Some(v) = preset.orphan {
        config.orphan_knob = v;
    }
}
"#,
        );
        write(
            &root,
            "crates/ny-propagate/src/engine/lane.rs",
            r#"
fn run(config: &BetaCrownConfig) -> bool {
    // The orphan knob is only ever MENTIONED in a comment: config.orphan_knob
    config.wired_knob && config.other.orphan_knob_like_name
}
"#,
        );
        let surface = delivery_surface(&root.join("crates/ny-cli/src/preset/apply.rs"));
        assert_eq!(
            surface,
            ["orphan_knob".to_string(), "wired_knob".to_string()]
                .into_iter()
                .collect::<BTreeSet<_>>(),
            "the surface must be exactly the two fields the resolver writes"
        );
        let consumed = consumed_fields(&scan_sources(&root.join("crates")), &surface);
        assert!(
            consumed.contains("wired_knob"),
            "a field the lane reads must count as consumed; got {consumed:?}"
        );
        assert!(
            !consumed.contains("orphan_knob"),
            "a field mentioned only in a COMMENT, and a longer name that merely contains it, \
             must not count as a consumer; got {consumed:?}"
        );
    }

    /// FAILURE 1b — the blast-radius version of the same failure: a whole
    /// resolved config rebuilt from `::default()`, dropping every preset value
    /// and carrying three fields forward by hand.
    #[test]
    fn demo_detects_a_whole_config_rebuilt_from_default() {
        let root = scratch("config_rebuild");
        write(
            &root,
            "crates/ny-cli/src/preset/apply.rs",
            "pub fn apply(config: &mut C) { config.enable_relaxed_clip = true; }\n",
        );
        write(
            &root,
            "crates/ny-propagate/src/engine/shared_specs.rs",
            r#"
fn nested_rebound(config: &BetaCrownConfig, deadline: Option<Instant>) -> BetaCrownConfig {
    let mut verifier_config = BetaCrownConfig::default();
    verifier_config.alpha_config.deadline = deadline;
    verifier_config.input_split_stacked_rebound = config.input_split_stacked_rebound;
    verifier_config
}
"#,
        );
        // A `MyBetaCrownConfig::default()` is a different type and must not be
        // confused for one.
        write(
            &root,
            "crates/ny-propagate/src/engine/lookalike.rs",
            "fn f() -> X { MyBetaCrownConfig::default() }\n",
        );
        let measured = config_rebuild_sites(&scan_sources(&root.join("crates")));
        assert_eq!(
            measured.get(&(
                "crates/ny-propagate/src/engine/shared_specs.rs".to_string(),
                "BetaCrownConfig::default".to_string()
            )),
            Some(&1),
            "the whole-config rebuild was NOT detected; got {measured:?}"
        );
        assert_eq!(
            measured.len(),
            1,
            "a differently-named type must not be counted; got {measured:?}"
        );
    }

    /// FAILURE 2 — the metaroom constrained-zonotope lane, built with zero
    /// constraint capacity, so it ran as plain DeepZ.
    #[test]
    fn demo_detects_a_capability_struct_built_with_zero_capacity() {
        let root = scratch("capability_zero");
        write(
            &root,
            "crates/ny-cli/src/commands/cz_metaroom_qualification.rs",
            r#"
fn metaroom_tail_lp_config(target_output: usize) -> ConstrainedZonotopeTailLpConfig {
    ConstrainedZonotopeTailLpConfig {
        target_output,
        limits: ConstrainedZonotopeTailLpLimits {
            max_value_dim: 256,
            max_alpha_dim: 2_321,
            max_constraints: 0,
            max_constraint_elements: 0,
            max_generator_nonzeros: 592_384,
            max_constraint_nonzeros: 0,
            max_output_dim: 20,
        },
    }
}
"#,
        );
        let measured = capability_zero_literals(&scan_sources(&root.join("crates")));
        let key = (
            "crates/ny-cli/src/commands/cz_metaroom_qualification.rs".to_string(),
            "max_constraint_elements+max_constraint_nonzeros+max_constraints".to_string(),
        );
        assert_eq!(
            measured.get(&key),
            Some(&1),
            "the zeroed constrained-zonotope capacities were NOT detected; got {measured:?}"
        );
    }

    /// FAILURE 3 — `alpha_iteration: 1`, which performs zero optimizer steps.
    #[test]
    fn demo_detects_a_semantically_dead_iteration_count() {
        let root = scratch("dead_value");
        write(
            &root,
            "configs/vnncomp25/cgan_2023.yaml",
            "bab:\n  branching:\n    input_split:\n      alpha_iteration: 1  # one warm step\n",
        );
        let (armed, unclassified) = armed_keys(&root.join("configs"));
        assert!(
            unclassified.is_empty(),
            "fixture must parse: {unclassified:?}"
        );
        let key = armed
            .iter()
            .find(|k| k.key == "bab.branching.input_split.alpha_iteration")
            .expect("the armed key must be found by its full dotted path");
        assert_eq!(key.value, "1", "the trailing comment must be stripped");
        assert!(
            dead_value_reason(&key.key, &key.value).is_some(),
            "alpha_iteration: 1 must be recognised as a dead value"
        );
        // The honest values around it are NOT flagged.
        assert!(dead_value_reason(&key.key, "0").is_none());
        assert!(dead_value_reason(&key.key, "20").is_none());
        assert!(dead_value_reason("bab.clip.relaxed", "1").is_none());
    }

    /// FAILURE 4 — `margin_row.beta_columns: 8` requested, `cols=1` delivered.
    ///
    /// Static analysis cannot see the collapse, so the gate fires on the ACT
    /// OF ARMING a key already known to be capped, and demands the
    /// requested-vs-delivered adjudication before any measurement is taken.
    #[test]
    fn demo_detects_arming_a_key_whose_delivered_value_is_capped() {
        let root = scratch("delivery_capped");
        write(
            &root,
            "configs/vnncomp25/cifar100_2024.yaml",
            "margin_row:\n  beta_per_column: true\n  beta_columns: 8\n",
        );
        let (armed, unclassified) = armed_keys(&root.join("configs"));
        assert!(
            unclassified.is_empty(),
            "fixture must parse: {unclassified:?}"
        );

        let capped: BTreeSet<&str> = DELIVERY_CAPPED.iter().map(|c| c.key).collect();
        let hits: Vec<&ArmedKey> = armed
            .iter()
            .filter(|k| capped.contains(k.key.as_str()))
            .collect();
        assert_eq!(
            hits.len(),
            2,
            "both capped margin-row keys must be seen as armed; got {hits:?}"
        );
        // With an empty adjudication manifest, every arming is a violation.
        let manifest: BTreeMap<Vec<String>, ManifestRow> = BTreeMap::new();
        let unadjudicated: Vec<&&ArmedKey> = hits
            .iter()
            .filter(|k| {
                !manifest.contains_key(&vec![k.file.clone(), k.key.clone(), k.value.clone()])
            })
            .collect();
        assert_eq!(
            unadjudicated.len(),
            2,
            "arming a capped key with no adjudication row must be a violation"
        );
    }

    /// FAILURE 5 — `tableau_f32`: implemented, deliverable, and armed by
    /// nothing.
    ///
    /// Run against the REAL configs tree with the one line that arms it
    /// removed, which is the state the tree was in before somebody happened to
    /// look. The key must then appear in the unarmed-promotion set.
    #[test]
    fn demo_detects_a_promoted_key_that_no_shipped_yaml_arms() {
        let root = workspace_root();
        let (armed, _) = armed_keys(&root.join("configs"));
        let armed_paths: BTreeSet<&str> = armed.iter().map(|k| k.key.as_str()).collect();

        // Today it IS armed, by exactly one file. That is the fixed state.
        assert!(
            armed_paths.contains("margin_row.tableau_f32"),
            "margin_row.tableau_f32 is expected to be armed in the current tree; if it has \
             been disarmed, the unarmed-promotion manifest must gain a row for it"
        );

        // Reconstruct the state before it was armed, and confirm the census
        // names it.
        let scratch_root = scratch("unarmed_promotion");
        std::fs::create_dir_all(scratch_root.join("configs/vnncomp25")).expect("mkdir");
        let mut copied = 0usize;
        for year in CONFIG_YEARS {
            let dir = root.join("configs").join(year);
            let Ok(entries) = std::fs::read_dir(&dir) else {
                continue;
            };
            std::fs::create_dir_all(scratch_root.join("configs").join(year)).expect("mkdir");
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().is_none_or(|x| x != "yaml") {
                    continue;
                }
                let text = std::fs::read_to_string(&path).expect("read shipped yaml");
                let mut without = String::with_capacity(text.len());
                for line in text
                    .lines()
                    .filter(|line| !line.trim_start().starts_with("tableau_f32:"))
                {
                    without.push_str(line);
                    without.push('\n');
                }
                let name = path.file_name().expect("file name");
                std::fs::write(scratch_root.join("configs").join(year).join(name), without)
                    .expect("write");
                copied += 1;
            }
        }
        assert!(
            copied > 40,
            "expected the whole shipped config set, copied {copied}"
        );

        let (armed_before, unclassified) = armed_keys(&scratch_root.join("configs"));
        assert!(
            unclassified.is_empty(),
            "the shipped configs must all parse: {unclassified:?}"
        );
        let before: BTreeSet<&str> = armed_before.iter().map(|k| k.key.as_str()).collect();
        assert!(
            !before.contains("margin_row.tableau_f32"),
            "the fixture must reproduce the unarmed state"
        );

        let unarmed: Vec<&str> = ny_levers::space::axes()
            .iter()
            .filter_map(|axis| match axis.deliver {
                Deliver::PresetKey(key) if !before.contains(key) => Some(key),
                _ => None,
            })
            .collect();
        assert!(
            unarmed.contains(&"margin_row.tableau_f32"),
            "the unarmed-promotion census must name margin_row.tableau_f32 when no yaml arms \
             it; census had {} entries",
            unarmed.len()
        );
    }

    /// The sixth shape, which arrives through interrupted I/O rather than
    /// through a config: a source file truncated to zero bytes.
    #[test]
    fn demo_detects_a_zero_byte_source_file() {
        let root = scratch("empty_source");
        write(&root, "crates/ny-propagate/src/real.rs", "pub fn f() {}\n");
        write(&root, "crates/ny-propagate/src/truncated.rs", "");
        write(
            &root,
            "crates/ny-propagate/src/comments_only.rs",
            "// everything below was lost\n",
        );

        let crates_root = root.join("crates");
        let mut files = Vec::new();
        collect_rs(&crates_root, &crates_root, &mut files);
        let mut empty = Vec::new();
        for (relative, path) in &files {
            let text = std::fs::read_to_string(path).expect("read");
            if text.is_empty() || blank_comments_and_strings(&text).trim().is_empty() {
                empty.push(relative.clone());
            }
        }
        empty.sort();
        assert_eq!(
            empty,
            vec![
                "ny-propagate/src/comments_only.rs".to_string(),
                "ny-propagate/src/truncated.rs".to_string()
            ],
            "both the zero-byte file and the comments-only file must be caught, and the real \
             one must not be"
        );
    }

    // -- scanner correctness, so a green ratchet is not a broken scanner ----

    #[test]
    fn demo_scanner_ignores_comments_strings_and_test_modules() {
        let root = scratch("scanner_hygiene");
        write(
            &root,
            "crates/ny-cli/src/preset/apply.rs",
            "pub fn apply(config: &mut C) { config.enable_relaxed_clip = true; }\n",
        );
        write(
            &root,
            "crates/ny-propagate/src/decoy.rs",
            r#"
// config.enable_relaxed_clip = false;   <- a comment, not a site
/* config.enable_relaxed_clip = false; */
const BANNER: &str = "config.enable_relaxed_clip = false;";
fn real(cfg: &mut C) {
    cfg.enable_relaxed_clip = false;
}
#[cfg(test)]
mod tests {
    #[test]
    fn fixture() {
        let mut c = C::default();
        c.enable_relaxed_clip = false;
    }
}
"#,
        );
        let surface = delivery_surface(&root.join("crates/ny-cli/src/preset/apply.rs"));
        let sites = override_sites(&scan_sources(&root.join("crates")), &surface);
        assert_eq!(
            sites.get(&(
                "crates/ny-propagate/src/decoy.rs".to_string(),
                "enable_relaxed_clip".to_string()
            )),
            Some(&1),
            "exactly ONE real site: the comment, the block comment, the string literal and the \
             #[cfg(test)] fixture must all be invisible. Got {sites:?}"
        );
    }

    #[test]
    fn demo_scanner_ignores_accumulators_and_comparisons() {
        let root = scratch("scanner_accumulators");
        write(
            &root,
            "crates/ny-cli/src/preset/apply.rs",
            "pub fn apply(config: &mut C) { config.max_depth = d; config.timeout = t; }\n",
        );
        write(
            &root,
            "crates/ny-propagate/src/stats.rs",
            r#"
fn tick(stats: &mut S, depth: usize, cfg: &C) {
    stats.max_depth = stats.max_depth.max(depth);
    if cfg.max_depth == depth {}
    if cfg.max_depth >= depth {}
    stats.timeout = deadline;
}
"#,
        );
        let surface = delivery_surface(&root.join("crates/ny-cli/src/preset/apply.rs"));
        let sites = override_sites(&scan_sources(&root.join("crates")), &surface);
        assert_eq!(
            sites.get(&(
                "crates/ny-propagate/src/stats.rs".to_string(),
                "max_depth".to_string()
            )),
            None,
            "a self-referential max() accumulator is not an override; got {sites:?}"
        );
        assert_eq!(
            sites.get(&(
                "crates/ny-propagate/src/stats.rs".to_string(),
                "timeout".to_string()
            )),
            Some(&1),
            "a genuine write must still be seen; got {sites:?}"
        );
    }

    #[test]
    fn demo_yaml_scanner_fails_closed_on_a_shape_it_cannot_read() {
        let root = scratch("yaml_fail_closed");
        write(
            &root,
            "configs/vnncomp25/odd.yaml",
            "bab:\n  clip:\n    relaxed: true\n  ? complex-key\n  : value\n",
        );
        let (armed, unclassified) = armed_keys(&root.join("configs"));
        assert!(
            armed.iter().any(|k| k.key == "bab.clip.relaxed"),
            "the readable part must still be read"
        );
        assert!(
            !unclassified.is_empty(),
            "an unreadable line must be a VIOLATION, not a silent skip — otherwise the gate \
             stops covering the file the day somebody adds a shape to it"
        );
    }

    #[test]
    fn demo_yaml_scanner_matches_the_shipped_dialect() {
        // Sequences, quoted values, trailing comments, and the alpha-beta-CROWN
        // hyphenated compatibility spelling all ship today.
        let root = scratch("yaml_dialect");
        write(
            &root,
            "configs/vnncomp25/dialect.yaml",
            concat!(
                "# leading comment\n",
                "general:\n",
                "  root_path: ../../benchmarks  # trailing comment\n",
                "solver:\n",
                "  alpha-crown:\n",
                "    share_alphas: true\n",
                "bab:\n",
                "  apply_output_constraints_to:\n",
                "    - \"/input-48\"  # PyTorch 2.3.1\n",
                "    - \"/input-52\"\n",
                "  batch_size: 64\n",
            ),
        );
        let (armed, unclassified) = armed_keys(&root.join("configs"));
        assert!(
            unclassified.is_empty(),
            "dialect must parse: {unclassified:?}"
        );
        let keys: Vec<&str> = armed.iter().map(|k| k.key.as_str()).collect();
        assert_eq!(
            keys,
            vec![
                "bab.apply_output_constraints_to",
                "bab.batch_size",
                "general.root_path",
                "solver.alpha-crown.share_alphas",
            ],
            "got {armed:?}"
        );
        let root_path = armed
            .iter()
            .find(|k| k.key == "general.root_path")
            .expect("root_path");
        assert_eq!(root_path.value, "../../benchmarks");
    }

    /// FAILURE 6 — a DORMANT value woken by a neighbouring key.
    ///
    /// `bab.batch_size: 64` was never consumed on the route the preset took;
    /// when `reorder_bab` flipped, the unexercised value took effect. (The
    /// reported 11.7x memory blowup has NO in-tree corroboration — see the
    /// registry's `measured` field — so what is encoded here is the shape, not
    /// the numbers.) The gate fires on the STATE TRANSITION, which is the only
    /// moment a reviewer can still act.
    #[test]
    fn demo_detects_a_dormant_value_woken_by_a_neighbouring_key() {
        // Driven by the PRODUCTION registry and the production function, so
        // this fixture cannot pass against a registry that no longer describes
        // the real pair.
        let yaml = |batch: &str, gate: &str| {
            format!(
                "bab:\n  batch_size: {batch}\n  branching:\n    method: input\n    \
                 input_split:\n      reorder_bab: {gate}\n"
            )
        };
        let root = scratch("dormant_value");
        write(&root, "configs/vnncomp25/before.yaml", &yaml("64", "false"));
        write(&root, "configs/vnncomp25/after.yaml", &yaml("64", "true"));
        // Same keys, but on the ReLU-split route, where the value is live
        // regardless of the gate. Encoding the dormancy without this scope
        // would have been FALSE on cifar100, yolo and tinyimagenet.
        write(
            &root,
            "configs/vnncomp25/other_route.yaml",
            "bab:\n  batch_size: 64\n  branching:\n    method: kfsb\n    input_split:\n      \
             reorder_bab: false\n",
        );
        // Carries the gate but no batch size at all.
        write(
            &root,
            "configs/vnncomp25/unrelated.yaml",
            "bab:\n  branching:\n    method: input\n    input_split:\n      reorder_bab: true\n",
        );
        // Both aliases set, as nn4sys.yaml does: the LAST one applied wins.
        write(
            &root,
            "configs/vnncomp25/aliased.yaml",
            "solver:\n  batch_size: 256\nbab:\n  batch_size: 64\n  branching:\n    method: \
             input\n    input_split:\n      reorder_bab: true\n",
        );

        let (armed, unclassified) = armed_keys(&root.join("configs"));
        assert!(
            unclassified.is_empty(),
            "fixture must parse: {unclassified:?}"
        );
        let states = dormant_states(&armed);
        let state_of = |file: &str| -> Option<(String, String)> {
            states
                .keys()
                .find(|k| k[0] == file)
                .map(|k| (format!("{}={}", k[1], k[2]), k[3].clone()))
        };

        assert_eq!(
            state_of("configs/vnncomp25/before.yaml"),
            Some(("bab.batch_size=64".into(), "dormant".into())),
            "with the gate off the value is carried but discarded for a literal 1; states = \
             {states:?}"
        );
        assert_eq!(
            state_of("configs/vnncomp25/after.yaml"),
            Some(("bab.batch_size=64".into(), "live".into())),
            "flipping the gate wakes the value - this is the transition the gate must catch"
        );
        assert_eq!(
            state_of("configs/vnncomp25/other_route.yaml"),
            None,
            "the dormancy is scoped to bab.branching.method: input. On the ReLU-split route the \
             batch size is live regardless of reorder_bab, so tracking it there would be a \
             FALSE dormancy claim on cifar100, yolo and tinyimagenet"
        );
        assert_eq!(
            state_of("configs/vnncomp25/unrelated.yaml"),
            None,
            "a yaml that never carried the value is not tracked at all"
        );
        assert_eq!(
            state_of("configs/vnncomp25/aliased.yaml"),
            Some(("bab.batch_size=64".into(), "live".into())),
            "when both aliases are set the resolver applies solver.batch_size first and \
             bab.batch_size second, so the EFFECTIVE value is 64, not 256. Reporting the alias \
             would quote a number the run never used"
        );

        // The transition is what fails: a frontier declaring `dormant` no
        // longer matches a tree that says `live`.
        let mut declared: BTreeMap<Vec<String>, ManifestRow> = BTreeMap::new();
        let identity = vec![
            "configs/vnncomp25/after.yaml".to_string(),
            "bab.batch_size".to_string(),
            "64".to_string(),
            "dormant".to_string(),
        ];
        declared.insert(
            identity.clone(),
            ManifestRow {
                identity,
                count: 1,
                class: "dormant".into(),
                note: "was dormant when this row was written".into(),
            },
        );
        let mut measured: BTreeMap<Vec<String>, usize> = BTreeMap::new();
        measured.insert(
            vec![
                "configs/vnncomp25/after.yaml".to_string(),
                "bab.batch_size".to_string(),
                "64".to_string(),
                "live".to_string(),
            ],
            1,
        );
        let violations = diff_frontier(&measured, &declared, "dormant-value state");
        assert!(
            violations.iter().any(|v| v.contains("live"))
                && violations.iter().any(|v| v.contains("dormant")),
            "waking a dormant value must be reported as BOTH a new `live` state and a stale \
             `dormant` one, so the reviewer sees the transition rather than a bare count: \
             {violations:?}"
        );
    }

    /// REGRESSION: a match guard is a READ, not an assignment.
    ///
    /// `Err(e) if config.flag => { .. }` puts a config field immediately before
    /// an `=`. The first real run of this gate reported three such guards as
    /// override sites — and the same bug in `field_reads` fails OPEN, hiding a
    /// genuine consumer, which would let
    /// `every_delivered_field_has_a_consumer` call a live field dead. Both
    /// directions are pinned here.
    #[test]
    fn demo_scanner_reads_a_match_guard_as_a_read_not_a_write() {
        let root = scratch("match_guard");
        write(
            &root,
            "crates/ny-cli/src/preset/apply.rs",
            "pub fn apply(config: &mut C) { config.fallback_to_ibp = v; config.plain = v; }\n",
        );
        write(
            &root,
            "crates/ny-propagate/src/engine/guarded.rs",
            r#"
fn bound(config: &C, r: Result<T, E>) -> Result<T, E> {
    match r {
        Ok(v) => Ok(v),
        Err(error @ E::DeadlineExceeded(_))
            if config.fallback_to_ibp =>
        {
            ibp(config)
        }
        Err(e) => Err(e),
    }
}
fn set(config: &mut C) {
    config.plain = false;
}
"#,
        );
        let surface = delivery_surface(&root.join("crates/ny-cli/src/preset/apply.rs"));
        let sources = scan_sources(&root.join("crates"));

        let sites = override_sites(&sources, &surface);
        assert_eq!(
            sites.get(&(
                "crates/ny-propagate/src/engine/guarded.rs".to_string(),
                "fallback_to_ibp".to_string()
            )),
            None,
            "a match guard is NOT an override site; got {sites:?}"
        );
        assert_eq!(
            sites.get(&(
                "crates/ny-propagate/src/engine/guarded.rs".to_string(),
                "plain".to_string()
            )),
            Some(&1),
            "the real assignment beside it must still be seen; got {sites:?}"
        );

        let consumed = consumed_fields(&sources, &surface);
        assert!(
            consumed.contains("fallback_to_ibp"),
            "a match guard IS a consumer — missing it would let this gate call a live field \
             dead; got {consumed:?}"
        );
    }

    #[test]
    fn demo_assignment_operator_rejects_every_lookalike() {
        let probe = |text: &str| {
            let bytes = text.as_bytes();
            let at = text.find('=').expect("probe needs an `=`");
            is_assignment_operator(bytes, at)
        };
        assert!(probe("x.f = 1"));
        assert!(
            !probe("x.f => 1"),
            "fat arrow is a match guard, not a write"
        );
        assert!(!probe("x.f == 1"));
        assert!(!probe("x.f != 1"));
        assert!(!probe("x.f >= 1"));
        assert!(!probe("x.f <= 1"));
        assert!(!probe("x.f += 1"));
    }

    #[test]
    fn demo_identifier_search_does_not_match_a_longer_name() {
        // The observable check would be vacuously green if `beta_columns`
        // matched `beta_columns_max_candidates`.
        assert!(contains_identifier("let beta_columns = 8;", "beta_columns"));
        assert!(!contains_identifier(
            "let beta_columns_max_candidates = 1;",
            "beta_columns"
        ));
        assert!(contains_identifier(
            "self.beta_columns_max_candidates += 1;",
            "beta_columns_max_candidates"
        ));
        assert!(!contains_identifier("xbeta_columns", "beta_columns"));
    }

    #[test]
    fn demo_frontier_diff_reports_every_violation_in_one_run() {
        // The trap this file exists not to repeat: source_frontier.rs asserts
        // three baselines sequentially and panics on the first, revealing one
        // debt per run.
        let mut measured: BTreeMap<Vec<String>, usize> = BTreeMap::new();
        measured.insert(vec!["a.rs".into(), "f".into()], 1);
        measured.insert(vec!["b.rs".into(), "g".into()], 3);
        let mut manifest: BTreeMap<Vec<String>, ManifestRow> = BTreeMap::new();
        manifest.insert(
            vec!["b.rs".into(), "g".into()],
            ManifestRow {
                identity: vec!["b.rs".into(), "g".into()],
                count: 1,
                class: "runtime-clear".into(),
                note: "when=x".into(),
            },
        );
        manifest.insert(
            vec!["c.rs".into(), "h".into()],
            ManifestRow {
                identity: vec!["c.rs".into(), "h".into()],
                count: 1,
                class: "runtime-clear".into(),
                note: "when=y".into(),
            },
        );
        let violations = diff_frontier(&measured, &manifest, "site");
        assert_eq!(
            violations.len(),
            3,
            "one undeclared, one count change and one stale row must ALL be reported: \
             {violations:?}"
        );
        assert!(violations.iter().any(|v| v.starts_with("UNDECLARED")));
        assert!(violations.iter().any(|v| v.starts_with("COUNT CHANGED")));
        assert!(violations.iter().any(|v| v.starts_with("STALE")));
    }
}
