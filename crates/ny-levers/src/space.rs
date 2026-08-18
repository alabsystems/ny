// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The lever SEARCH SPACE: which knobs an automated search may move, what values
//! it may give them, and which combinations are provably inert.
//!
//! WHY THIS IS NOT `crate::all()`. The registry is a *declaration* of levers that
//! happen to be governed; it is not a statement about what is worth searching, and
//! it is not complete. Three facts make a naive sweep over `all()` waste almost
//! its entire budget:
//!
//! 1. **Most points are inert.** The four root-phase levers have FIVE distinct
//!    behaviours, not sixteen: an earlier phase that fires OWNS the slot and the
//!    later ones never run. Twelve declared levers do nothing at all unless a
//!    companion is also set, and five of those companions are not declared here.
//! 2. **Some levers must never be moved automatically.** [`Class::Unsafe`] marks
//!    the ones that change what is being proved, replace a certified error bound,
//!    or hard-fail engine construction.
//! 3. **A measurement lever is not a treatment.** `MoatRisk::Low` means "cannot
//!    touch a bound" — it does NOT mean free. Extra clocks, formatting and stderr
//!    perturb a deadline-sensitive run, so telemetry axes are instruments and must
//!    never appear in a timing arm.
//!
//! THE PARSER CONTRACT, which a search violates by accident. `resolve_raw` arms a
//! Bool on the EXACT byte string `"1"` and disarms on exact `"0"`; `"true"`,
//! `"01"`, `" 1"` and `""` are REJECTIONS that resolve to the declaration default
//! and are recorded in `Resolved::rejected_raw`. A harness that emits `"true"`
//! therefore measures the baseline and reports it as a treatment. [`Domain::Bool`]
//! only ever emits `"0"` / `"1"`, and callers are expected to assert
//! `rejected_raw` is empty on every completed run.
//!
//! Two legacy parsers are deliberately NOT modelled as Bool, because for them the
//! usual tokens mean the opposite of what they look like:
//! `NY_MIP_TRACE` is presence-armed — `"0"` and even `""` ARM it — and
//! `NY_CONV_PATCHES_DEBUG` arms on any non-empty value that is not exactly `"0"`,
//! so `" 0"` (with a space) ARMS it. Both are telemetry, hence excluded anyway,
//! but they are listed here so nobody re-adds them as ordinary Bools.

use std::collections::BTreeMap;

/// What values a search may give an axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Domain {
    /// Exactly `"0"` or `"1"`. No other spelling survives the parser.
    Bool,
    /// A fixed set of admissible tokens, emitted verbatim.
    Enum(&'static [&'static str]),
    /// Integer grid, emitted with `to_string()`.
    Grid(&'static [u64]),
}

/// How much damage a wrong value can do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Class {
    /// Never moved by an automated search. See each axis's `why` field.
    Unsafe,
    /// May change a published bound. Searchable, but every surviving candidate
    /// must clear the moat gate before promotion.
    VerdictAffecting,
    /// Cannot change a verdict's soundness. Still costs wall clock.
    SafeToSearch,
}

/// Whether a value this search picks can actually reach a scored run.
///
/// This is the single most expensive thing to get wrong.
/// `vnncomp_scripts/run_instance.sh` exports exactly ONE `NY_*` variable
/// (`NY_UPFRONT_ATTACK=1`, safenlp only); everything else runs at its COMPILED
/// DEFAULT during scoring. An axis that is [`Deliver::EnvOnly`] can be searched
/// and can be measured, but its result is worth zero until a typed preset key
/// exists — the failure mode `crates/ny-cli/tests/measured_gate_delivery.rs`
/// was written to catch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Deliver {
    /// Reaches a scored run through this preset key.
    PresetKey(&'static str),
    /// Environment only: measurable, but NOT shippable as-is.
    EnvOnly,
}

/// One searchable dimension.
#[derive(Debug, Clone, Copy)]
pub struct Axis {
    pub name: &'static str,
    pub domain: Domain,
    pub class: Class,
    pub deliver: Deliver,
    /// Why this axis is classified as it is. Load-bearing for review.
    pub why: &'static str,
}

/// A condition another axis must satisfy for this one to do anything.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Requirement {
    /// The named axis must be exactly `"1"`.
    Armed(&'static str),
    /// The named axis must be present and NOT `"0"`.
    NonZero(&'static str),
    /// The named axis must parse to a value strictly greater than the bound.
    GreaterThan(&'static str, u64),
    /// The named axis must be absent or `"0"` — it OWNS the slot when armed.
    NotArmed(&'static str),
}

/// `child` does nothing unless `requires` holds.
#[derive(Debug, Clone, Copy)]
pub struct Edge {
    pub child: &'static str,
    pub requires: Requirement,
    /// The `&&` / `?` / early-return that makes this true, for auditing.
    pub site: &'static str,
}

/// Why a sample was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Inert {
    /// The axis that would have had no effect.
    pub axis: String,
    /// The unmet requirement, rendered.
    pub because: String,
}

impl std::fmt::Display for Inert {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "`{}` is inert: {}", self.axis, self.because)
    }
}

/// UNSAFE axes, excluded permanently. Listed rather than omitted so that a future
/// reader can see they were considered and rejected on purpose.
const UNSAFE_AXES: &[Axis] = &[
    Axis {
        name: "NY_EFT_ERR",
        domain: Domain::Bool,
        class: Class::Unsafe,
        deliver: Deliver::EnvOnly,
        why: "replaces the a-priori gamma_n certified-error bound with an \
              EFT-MEASURED one, i.e. swaps a proof for an observation, and does it \
              through raw ungoverned reads outside the registry chokepoint",
    },
    Axis {
        name: "NY_CUDA_DISCRETE_MODE",
        domain: Domain::Bool,
        class: Class::Unsafe,
        deliver: Deliver::EnvOnly,
        why: "ny-cuda applies its own stricter parser and FAILS ENGINE \
              CONSTRUCTION on any token that is not exactly 0 or 1; a search that \
              emits anything else does not measure a treatment, it kills the run",
    },
    Axis {
        name: "NY_STRIP_TERMINAL_SOFTMAX",
        domain: Domain::Bool,
        class: Class::Unsafe,
        deliver: Deliver::EnvOnly,
        why: "rewrites the network before verification, i.e. changes the property \
              being proved; a 'win' here is not a win",
    },
];

/// Telemetry axes. Instruments, never treatment arms.
///
/// They cannot touch a bound, but they are NOT free: each adds clocks, formatting
/// and stderr to a deadline-sensitive run. `NY_ITER0_PARITY_TRACE` emits one line
/// per node per backward walk and must never be on in a timing arm.
const INSTRUMENT_ONLY: &[&str] = &[
    "NY_PHASE_TELEMETRY",
    "NY_BETA_GPU_PROBE",
    "NY_SEG_PROBE",
    "NY_MARGIN_ROW_PROFILE",
    "NY_INPUT_SPLIT_PROBE",
    "NY_ENVELOPE_RESCALE_PROBE",
    "NY_ENVELOPE_XSTAR_PROBE",
    "NY_ITER0_PARITY_TRACE",
    "NY_PATCHES_CARRIER_TRACE",
    "NY_GPU_MEM_TRACE",
    "NY_MIP_TRACE",
    "NY_CONV_PATCHES_DEBUG",
];

/// Test-only axes, excluded: they exist to drive fixtures, not the scored path.
const TEST_ONLY: &[&str] = &[
    "NY_FULL_MEASUREMENTS",
    "NY_BENCH_ROOT",
    "NY_BENCH_ROOT_2026",
    "NY_FORCE_GPU_BAB_BOUND_SELFCHECK_FAIL",
];

/// The searchable axes.
const AXES: &[Axis] = &[
    // --- root intermediate-tightening family (mutually exclusive, see EDGES) ---
    Axis {
        name: "NY_ROOT_COMPREHENSIVE_GPU_INTERM_CROWN",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("bab.root_comprehensive_gpu_interm"),
        why: "atomic all-target sound-GPU root sweep; shrink-only, but it changes \
              published intermediate bounds",
    },
    Axis {
        name: "NY_INTERM_ROW_CHUNKS",
        domain: Domain::Grid(&[1, 4, 16, 64]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("bab.root_comprehensive_gpu_interm_chunks"),
        why: "disjoint row windows accumulated by the comprehensive sweep; 1 is \
              byte-identical to the historical single sweep",
    },
    Axis {
        name: "NY_ROOT_PHASE_RESIDENT_CROWN",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::EnvOnly,
        why: "OWNS the root slot when armed, suppressing the comprehensive and \
              wide-demanded phases entirely",
    },
    Axis {
        name: "NY_ROOT_WIDE_DEMANDED_INTERM_CROWN",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::EnvOnly,
        why: "one-target wide demanded intermediate CROWN; reachable only when \
              both earlier root phases are unarmed",
    },
    Axis {
        name: "NY_ROOT_CPU_PARALLEL_INTERM_CROWN",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::EnvOnly,
        why: "replaces the dense-head tightener outright rather than adding to it",
    },
    Axis {
        name: "NY_ROOT_COMP_GPU_INTERM_ROWS",
        domain: Domain::Grid(&[16, 32, 64, 128]),
        class: Class::VerdictAffecting,
        deliver: Deliver::EnvOnly,
        why: "per-target row ceiling for the comprehensive sweep; the device class \
              policy may cap it below the requested value",
    },
    Axis {
        name: "NY_ROOT_COMP_GPU_INTERM_SECS",
        domain: Domain::Grid(&[10, 20, 40]),
        class: Class::VerdictAffecting,
        deliver: Deliver::EnvOnly,
        why: "local authority slice for the comprehensive sweep; capped in turn by \
              half the remaining global budget, so large values saturate",
    },
    // --- margin-row GPU seam ---
    Axis {
        name: "NY_MARGIN_ROW_GPU",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::EnvOnly,
        why: "routes the margin-row lane through the GPU seam; LATCHED in a \
              OnceLock, so it can only be varied across processes",
    },
    Axis {
        name: "NY_MARGIN_ROW_GPU_BATCH",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::EnvOnly,
        why: "batches the margin-row GPU seam; inert unless the seam itself is on",
    },
    // --- wide BaB lane ---
    Axis {
        name: "NY_BAB_RESNET_WIDE",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::EnvOnly,
        why: "domain-stacked wide resnet BaB lane",
    },
    Axis {
        name: "NY_BAB_RESNET_WIDE_SUBGROUP",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::EnvOnly,
        why: "subgroup path within the wide lane; inert unless the wide lane is on",
    },
    // --- alpha ---
    Axis {
        name: "NY_ALPHA_ENVELOPE_GRAD",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::EnvOnly,
        why: "envelope gradient for the alpha ascent; changes which alpha is \
              reached, hence the bound",
    },
    Axis {
        name: "NY_ALPHA_ZERO_YIELD_FRAC",
        // `LeverKind::F64Open { min: 0.0, max: 0.9 }` is an OPEN interval, so
        // tokens must parse as f64 STRICTLY inside it. This axis previously
        // offered the integer grid [0, 25, 50]; every one of those is rejected
        // by the parser and silently resolves to the declaration default, so a
        // search would have spent a full instance budget measuring the baseline
        // and labelled it a treatment — the exact failure this module exists to
        // prevent. Pinned by `every_domain_token_survives_the_real_parser`.
        domain: Domain::Enum(&["0.1", "0.25", "0.5"]),
        class: Class::VerdictAffecting,
        deliver: Deliver::PresetKey("bab.alpha_crown.alpha_zero_yield_frac"),
        why: "the ONLY axis in the registry carrying Provenance::Measured — and \
              its recorded delta is zero row conversions",
    },
    // --- misc engine ---
    Axis {
        name: "NY_TRUE_GRAD_GPU_REPLAY",
        domain: Domain::Bool,
        class: Class::VerdictAffecting,
        deliver: Deliver::EnvOnly,
        why: "GPU replay of the true-gradient lane",
    },
    Axis {
        name: "NY_INPUT_SPLIT_NESTED_DEADLINE",
        domain: Domain::Bool,
        class: Class::SafeToSearch,
        deliver: Deliver::EnvOnly,
        why: "nested deadline discipline for input split; scheduling only",
    },
    Axis {
        name: "NY_PATCHES_FINITE_EXPIRY",
        domain: Domain::Bool,
        class: Class::SafeToSearch,
        deliver: Deliver::EnvOnly,
        why: "finite expiry on the patches path; refusal timing, not bound math",
    },
    Axis {
        name: "NY_NO_WALK_RECORD_ADMISSION",
        domain: Domain::Bool,
        class: Class::SafeToSearch,
        deliver: Deliver::EnvOnly,
        why: "admission bookkeeping; LATCHED in a OnceLock",
    },
];

/// Prerequisite and exclusion edges.
///
/// Every one of these corresponds to a literal `&&`, `?` or early return in the
/// engine. A sample that violates one is not a cheap experiment — it is a 100 s
/// measurement of the baseline, mislabelled as a treatment.
const EDGES: &[Edge] = &[
    Edge {
        child: "NY_MARGIN_ROW_GPU_BATCH",
        requires: Requirement::Armed("NY_MARGIN_ROW_GPU"),
        site: "margin_row/gpu_seam/batch.rs — batch seam checks the seam gate first",
    },
    Edge {
        child: "NY_BAB_RESNET_WIDE_SUBGROUP",
        requires: Requirement::NonZero("NY_BAB_RESNET_WIDE"),
        site: "decls/wide_lane.rs — subgroup is a refinement of the wide lane",
    },
    Edge {
        child: "NY_INTERM_ROW_CHUNKS",
        requires: Requirement::Armed("NY_ROOT_COMPREHENSIVE_GPU_INTERM_CROWN"),
        site: "multi_objective/root.rs — chunks are consumed only by the \
               comprehensive sweep policy",
    },
    Edge {
        child: "NY_ROOT_COMP_GPU_INTERM_ROWS",
        requires: Requirement::Armed("NY_ROOT_COMPREHENSIVE_GPU_INTERM_CROWN"),
        site: "multi_objective/root.rs — row ceiling belongs to the comprehensive \
               policy",
    },
    Edge {
        child: "NY_ROOT_COMP_GPU_INTERM_SECS",
        requires: Requirement::Armed("NY_ROOT_COMPREHENSIVE_GPU_INTERM_CROWN"),
        site: "multi_objective/root.rs — slice belongs to the comprehensive policy",
    },
    // The root-phase ownership lattice: an earlier phase that fires OWNS the slot.
    Edge {
        child: "NY_ROOT_COMPREHENSIVE_GPU_INTERM_CROWN",
        requires: Requirement::NotArmed("NY_ROOT_PHASE_RESIDENT_CROWN"),
        site: "multi_objective/root.rs phase_resident_or_comprehensive — resident \
               wins and the comprehensive closure is never called",
    },
    Edge {
        child: "NY_ROOT_WIDE_DEMANDED_INTERM_CROWN",
        requires: Requirement::NotArmed("NY_ROOT_PHASE_RESIDENT_CROWN"),
        site: "multi_objective/root.rs — resident owns the slot",
    },
    Edge {
        child: "NY_ROOT_WIDE_DEMANDED_INTERM_CROWN",
        requires: Requirement::NotArmed("NY_ROOT_COMPREHENSIVE_GPU_INTERM_CROWN"),
        site: "multi_objective/root.rs comprehensive_gpu_or_legacy_wide — a \
               comprehensive Some(0) still owns the slot, so wide never runs",
    },
];

/// The searchable axes, excluding unsafe, telemetry and test-only names.
#[must_use]
pub fn axes() -> &'static [Axis] {
    AXES
}

/// Axes that are permanently excluded, with the reason recorded.
#[must_use]
pub fn unsafe_axes() -> &'static [Axis] {
    UNSAFE_AXES
}

/// Names that are instruments rather than treatments.
#[must_use]
pub fn instrument_only() -> &'static [&'static str] {
    INSTRUMENT_ONLY
}

/// Names excluded because they only drive tests.
#[must_use]
pub fn test_only() -> &'static [&'static str] {
    TEST_ONLY
}

/// The interaction lattice.
#[must_use]
pub fn edges() -> &'static [Edge] {
    EDGES
}

#[must_use]
fn lookup(sample: &BTreeMap<&str, String>, name: &str) -> Option<String> {
    sample.get(name).cloned()
}

fn armed(sample: &BTreeMap<&str, String>, name: &str) -> bool {
    lookup(sample, name).as_deref() == Some("1")
}

fn non_zero(sample: &BTreeMap<&str, String>, name: &str) -> bool {
    match lookup(sample, name) {
        Some(value) => value != "0",
        None => false,
    }
}

fn greater_than(sample: &BTreeMap<&str, String>, name: &str, bound: u64) -> bool {
    lookup(sample, name)
        .and_then(|value| value.parse::<u64>().ok())
        .is_some_and(|value| value > bound)
}

/// Turn a sample into the environment assignment to apply, or explain why the
/// sample is inert.
///
/// A sample only mentions the axes it wants to move; an absent axis means "leave
/// at the compiled default". `expand` refuses when the sample sets an axis whose
/// prerequisite is unmet, because running it would spend a full instance budget
/// re-measuring the baseline. That refusal is the whole point of this module.
///
/// # Errors
/// Returns [`Inert`] naming the first axis whose requirement is unmet, and the
/// requirement that failed.
pub fn expand(sample: &BTreeMap<&str, String>) -> Result<Vec<(String, String)>, Inert> {
    for (name, value) in sample {
        if UNSAFE_AXES.iter().any(|axis| axis.name == *name) {
            return Err(Inert {
                axis: (*name).to_string(),
                because: "axis is Class::Unsafe and must never be set by a search".to_string(),
            });
        }
        // An axis left at its default cannot be inert — it is not being moved.
        let is_default_bool = value == "0";
        for edge in EDGES {
            if edge.child != *name || is_default_bool {
                continue;
            }
            let (ok, rendered) = match edge.requires {
                Requirement::Armed(parent) => (
                    armed(sample, parent),
                    format!("requires `{parent}` = \"1\" ({})", edge.site),
                ),
                Requirement::NonZero(parent) => (
                    non_zero(sample, parent),
                    format!("requires `{parent}` present and not \"0\" ({})", edge.site),
                ),
                Requirement::GreaterThan(parent, bound) => (
                    greater_than(sample, parent, bound),
                    format!("requires `{parent}` > {bound} ({})", edge.site),
                ),
                Requirement::NotArmed(parent) => (
                    !armed(sample, parent),
                    format!("suppressed while `{parent}` = \"1\" ({})", edge.site),
                ),
            };
            if !ok {
                return Err(Inert {
                    axis: (*name).to_string(),
                    because: rendered,
                });
            }
        }
    }
    Ok(sample
        .iter()
        .map(|(name, value)| ((*name).to_string(), value.clone()))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(pairs: &[(&'static str, &str)]) -> BTreeMap<&'static str, String> {
        pairs
            .iter()
            .map(|(name, value)| (*name, (*value).to_string()))
            .collect()
    }

    #[test]
    fn every_axis_name_is_declared_in_the_registry() {
        // The space may EXCLUDE declared levers, but it must never invent one:
        // a typo'd axis is a search dimension that silently does nothing.
        let declared: Vec<&str> = crate::all().all().iter().map(|decl| decl.name).collect();
        for axis in AXES.iter().chain(UNSAFE_AXES) {
            assert!(
                declared.contains(&axis.name),
                "axis `{}` is not a declared lever",
                axis.name
            );
        }
    }

    #[test]
    fn no_axis_is_both_searchable_and_excluded() {
        for axis in AXES {
            assert!(
                !INSTRUMENT_ONLY.contains(&axis.name),
                "`{}` is searchable and instrument-only",
                axis.name
            );
            assert!(
                !TEST_ONLY.contains(&axis.name),
                "`{}` is searchable and test-only",
                axis.name
            );
            assert!(
                axis.class != Class::Unsafe,
                "`{}` is Unsafe but listed as searchable",
                axis.name
            );
        }
    }

    #[test]
    fn every_edge_refers_to_a_known_axis() {
        let known: Vec<&str> = AXES.iter().map(|axis| axis.name).collect();
        for edge in EDGES {
            assert!(known.contains(&edge.child), "edge child `{}`", edge.child);
            let parent = match edge.requires {
                Requirement::Armed(parent)
                | Requirement::NonZero(parent)
                | Requirement::NotArmed(parent) => parent,
                Requirement::GreaterThan(parent, _) => parent,
            };
            assert!(known.contains(&parent), "edge parent `{parent}`");
        }
    }

    #[test]
    fn unsafe_axes_are_refused() {
        let err = expand(&sample(&[("NY_STRIP_TERMINAL_SOFTMAX", "1")])).unwrap_err();
        assert_eq!(err.axis, "NY_STRIP_TERMINAL_SOFTMAX");
        assert!(err.because.contains("Unsafe"), "{}", err.because);
    }

    #[test]
    fn chunks_without_the_comprehensive_sweep_are_refused() {
        // This is the concrete waste the module exists to prevent: chunks are read
        // only by the comprehensive policy, so this sample would spend a full
        // instance budget measuring the baseline.
        let err = expand(&sample(&[("NY_INTERM_ROW_CHUNKS", "64")])).unwrap_err();
        assert_eq!(err.axis, "NY_INTERM_ROW_CHUNKS");
        assert!(
            err.because
                .contains("NY_ROOT_COMPREHENSIVE_GPU_INTERM_CROWN"),
            "{}",
            err.because
        );
    }

    #[test]
    fn chunks_with_the_comprehensive_sweep_are_accepted() {
        let expanded = expand(&sample(&[
            ("NY_ROOT_COMPREHENSIVE_GPU_INTERM_CROWN", "1"),
            ("NY_INTERM_ROW_CHUNKS", "64"),
        ]))
        .expect("armed sample must expand");
        assert_eq!(expanded.len(), 2);
    }

    #[test]
    fn phase_resident_suppresses_the_comprehensive_sweep() {
        let err = expand(&sample(&[
            ("NY_ROOT_PHASE_RESIDENT_CROWN", "1"),
            ("NY_ROOT_COMPREHENSIVE_GPU_INTERM_CROWN", "1"),
        ]))
        .unwrap_err();
        assert_eq!(err.axis, "NY_ROOT_COMPREHENSIVE_GPU_INTERM_CROWN");
        assert!(err.because.contains("suppressed"), "{}", err.because);
    }

    #[test]
    fn wide_demanded_needs_both_earlier_phases_unarmed() {
        for blocker in [
            "NY_ROOT_PHASE_RESIDENT_CROWN",
            "NY_ROOT_COMPREHENSIVE_GPU_INTERM_CROWN",
        ] {
            let err = expand(&sample(&[
                (blocker, "1"),
                ("NY_ROOT_WIDE_DEMANDED_INTERM_CROWN", "1"),
            ]))
            .unwrap_err();
            assert!(
                err.axis == "NY_ROOT_WIDE_DEMANDED_INTERM_CROWN" || err.axis == blocker,
                "unexpected refusal {err}"
            );
        }
        expand(&sample(&[("NY_ROOT_WIDE_DEMANDED_INTERM_CROWN", "1")]))
            .expect("reachable at (0,0,1)");
    }

    #[test]
    fn margin_row_batch_requires_the_seam() {
        let err = expand(&sample(&[("NY_MARGIN_ROW_GPU_BATCH", "1")])).unwrap_err();
        assert_eq!(err.axis, "NY_MARGIN_ROW_GPU_BATCH");
        expand(&sample(&[
            ("NY_MARGIN_ROW_GPU", "1"),
            ("NY_MARGIN_ROW_GPU_BATCH", "1"),
        ]))
        .expect("armed seam must accept the batch axis");
    }

    #[test]
    fn disarming_a_child_is_never_inert() {
        // Setting a child to "0" is not a treatment that needs its parent; it is
        // an explicit disarm and must always be emittable.
        expand(&sample(&[("NY_INTERM_ROW_CHUNKS", "0")])).expect("explicit disarm");
        expand(&sample(&[("NY_MARGIN_ROW_GPU_BATCH", "0")])).expect("explicit disarm");
    }

    /// EVERY token of EVERY axis must survive the REAL parser.
    ///
    /// This replaces a version that inspected only `Domain::Enum` and was
    /// therefore vacuous for every axis in this file (all were Bool/Grid) — a
    /// guard that passed while `NY_ALPHA_ZERO_YIELD_FRAC` shipped an integer
    /// grid against an `F64Open { 0.0, 0.9 }` declaration, i.e. three tokens the
    /// parser rejects outright.
    ///
    /// A rejected token does not fail loudly: it resolves to the declaration
    /// default and lands in `rejected_raw`, so the run measures the BASELINE and
    /// a search reports it as a treatment. Inspecting token shape by hand cannot
    /// catch that; only the parser can. So ask it, through the same `read_with`
    /// path production uses.
    #[test]
    fn every_domain_token_survives_the_real_parser() {
        for axis in AXES {
            let decl = crate::all()
                .get(axis.name)
                .unwrap_or_else(|| panic!("axis `{}` is not declared", axis.name));
            for token in domain_tokens(axis.domain) {
                let resolved = crate::read_with(decl, |_| Some(token.clone()));
                assert!(
                    resolved.rejected_raw.is_none(),
                    "axis `{}` offers token {token:?}, which the parser REJECTS. \
                     A rejected token silently resolves to the declaration \
                     default, so this value would measure the baseline and be \
                     reported as a treatment.",
                    axis.name
                );
                assert_eq!(
                    resolved.source,
                    crate::Source::LegacyEnv,
                    "axis `{}` token {token:?} did not take effect",
                    axis.name
                );
            }
        }
    }

    /// The guard above is only meaningful if it can fail. Prove that it does,
    /// using the exact tokens this file used to ship for that axis.
    #[test]
    fn the_parser_guard_is_not_vacuous() {
        let decl = crate::all()
            .get("NY_ALPHA_ZERO_YIELD_FRAC")
            .expect("declared");
        for bad in ["0", "25", "50"] {
            let resolved = crate::read_with(decl, |_| Some(bad.to_string()));
            assert!(
                resolved.rejected_raw.is_some(),
                "expected the parser to reject {bad:?} for an F64Open(0.0, 0.9) \
                 lever; if this passes, the guard above proves nothing"
            );
        }
    }

    fn domain_tokens(domain: Domain) -> Vec<String> {
        match domain {
            Domain::Bool => vec!["0".to_string(), "1".to_string()],
            Domain::Enum(values) => values.iter().map(|v| (*v).to_string()).collect(),
            Domain::Grid(values) => values.iter().map(u64::to_string).collect(),
        }
    }
}
