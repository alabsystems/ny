// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Per-node scheduling policy for graph/sequential CROWN-IBP collection.
//!
//! The per-node floor/cap controls are unusual migration candidates: the scored
//! preset route predates their environment aliases. `BabPreset` already exposes
//! `bab.crown_ibp_per_node_{floor,cap}_secs`, and several shipped cgan/yolo
//! presets use the cap, but the aliases were never declared and neither field
//! appeared in the compiled search space.  The result was a parameter built
//! expressly for measurement that the governed optimizer could not move.
//! This module also owns the coupled objective-chunk controls and adjacent
//! Patches/provenance guards that must remain excluded from automated search.

use crate::{
    declare_levers, Bucket, DefaultSpec, LeverDecl, LeverKind, MoatRisk, Provenance, ReaderSite,
    Scope,
};

const CROWN_IBP_BUDGET: Scope = Scope {
    package: "ny-propagate",
    subsystem: "crown-ibp-per-node-budget",
};

const PATCHES_DEADLINE_SCOPE: Scope = Scope {
    package: "ny-propagate",
    subsystem: "crown-patches-deadline-scheduling",
};

const CROWN_PROVENANCE_SCOPE: Scope = Scope {
    package: "ny-propagate",
    subsystem: "crown-bound-provenance",
};

const BETA_CROWN_SEQUENTIAL_SCOPE: Scope = Scope {
    package: "ny-propagate",
    subsystem: "beta-crown-sequential-runtime",
};

const CLI_PRESET: Scope = Scope {
    package: "ny-cli",
    subsystem: "preset",
};

const CLI_VNNCOMP: Scope = Scope {
    package: "ny-cli",
    subsystem: "vnncomp-entry",
};

const CLI_BETA_CROWN: Scope = Scope {
    package: "ny-cli",
    subsystem: "beta-crown-runtime-config",
};

const READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: CLI_PRESET,
        role: "load the typed bab.crown_ibp_per_node_{floor,cap}_secs fields into BetaCrownConfig",
        site: "crates/ny-cli/src/preset/apply.rs:apply_bab_preset",
    },
    ReaderSite {
        scope: CROWN_IBP_BUDGET,
        role: "layer the legacy operator override over the typed preset and freeze the resolved scheduling policy",
        site: "crates/ny-propagate/src/network/graph_alpha/bounds/budget_policy.rs:resolve_per_node_time_budget",
    },
    ReaderSite {
        scope: CROWN_IBP_BUDGET,
        role: "divide the remaining collector window subject to the resolved floor and cap",
        site: "crates/ny-propagate/src/network/graph_alpha/bounds/budget_policy.rs:compute_weighted_per_node_budget_secs",
    },
];

const OBJECTIVE_CHUNK_WAVE_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: CROWN_IBP_BUDGET,
        role: "derive graph-local wave scheduling from the exact frozen run authority, with a legacy compatibility fallback",
        site: "crates/ny-propagate/src/network/core/graph/mod.rs:crown_ibp_runtime_policy",
    },
    ReaderSite {
        scope: CLI_VNNCOMP,
        role: "invert the entry-time frozen aggregate-memory guard into the positive graph execution decision carried by the scored run",
        site: "crates/ny-cli/src/commands/vnncomp.rs:run_and_translate",
    },
    ReaderSite {
        scope: CLI_BETA_CROWN,
        role: "resolve the interactive compatibility environment once before verification when no scored frozen decision was supplied",
        site: "crates/ny-cli/src/commands/beta_crown/mod.rs:resolve_chunk_wave_parallel_enabled",
    },
];

const OBJECTIVE_CHUNK_WIDTH_READERS: &[ReaderSite] = &[ReaderSite {
    scope: CROWN_IBP_BUDGET,
    role: "derive the objective-row chunk width used by graph execution and cache identity from one frozen run authority",
    site: "crates/ny-propagate/src/network/core/graph/mod.rs:crown_ibp_runtime_policy",
}];

const OBJECTIVE_CHUNK_ABORT_READERS: &[ReaderSite] = &[ReaderSite {
    scope: CROWN_IBP_BUDGET,
    role: "derive projected early abort versus the legacy full-budget chunk loop from one frozen run authority",
    site: "crates/ny-propagate/src/network/core/graph/mod.rs:crown_ibp_runtime_policy",
}];

const OBJECTIVE_CHUNK_GROW_READERS: &[ReaderSite] = &[ReaderSite {
    scope: CROWN_IBP_BUDGET,
    role: "derive adaptive objective-chunk widening versus the legacy fixed-width loop from one frozen run authority",
    site: "crates/ny-propagate/src/network/core/graph/mod.rs:crown_ibp_runtime_policy",
}];

const PATCHES_DEADLINE_FLAT_BIAS_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: PATCHES_DEADLINE_SCOPE,
        role: "preserve live compatibility for non-graph Patches callers that have no frozen run authority",
        site: "crates/ny-propagate/src/layers/convolution/conv2d/bound_patches.rs:patches_deadline_flat_bias_enabled",
    },
    ReaderSite {
        scope: CROWN_IBP_BUDGET,
        role: "derive the graph execution and cache decision from one frozen run authority",
        site: "crates/ny-propagate/src/network/core/graph/mod.rs:crown_ibp_runtime_policy",
    },
    ReaderSite {
        scope: BETA_CROWN_SEQUENTIAL_SCOPE,
        role: "derive direct sequential Patches execution from the verifier's exact frozen run authority",
        site: "crates/ny-propagate/src/beta_crown/engine/bounds.rs:sequential_crown_ibp_runtime_policy",
    },
    ReaderSite {
        scope: PATCHES_DEADLINE_SCOPE,
        role: "resolve the compatibility policy once for a sequential Patches step reached without a frozen run authority",
        site: "crates/ny-propagate/src/network/core/sequential/crown/patches_step.rs:crown_backward_step_patches_with_deadline_authority",
    },
    ReaderSite {
        scope: PATCHES_DEADLINE_SCOPE,
        role: "resolve the compatibility policy for the spec-guided Patches dispatcher, which carries no frozen run authority",
        site: "crates/ny-propagate/src/network/core/sequential/crown/patches_step.rs:crown_backward_step_patches_spec_crown",
    },
];

const CROWN_HONEST_PROVENANCE_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: CROWN_PROVENANCE_SCOPE,
        role: "preserve live compatibility for callers that have no frozen graph or verifier authority",
        site: "crates/ny-propagate/src/network/crown_quality.rs:honest_provenance_enabled",
    },
    ReaderSite {
        scope: CROWN_IBP_BUDGET,
        role: "derive graph-native quality publication and cache identity from one frozen run authority",
        site: "crates/ny-propagate/src/network/core/graph/mod.rs:crown_ibp_runtime_policy",
    },
    ReaderSite {
        scope: BETA_CROWN_SEQUENTIAL_SCOPE,
        role: "derive direct sequential quality publication from the verifier's exact frozen run authority",
        site: "crates/ny-propagate/src/beta_crown/engine/bounds.rs:sequential_crown_ibp_runtime_policy",
    },
];

declare_levers! {
    registry CROWN_IBP_BUDGET_LEVERS;

    /// `NY_PER_NODE_FLOOR_SECS` — minimum useful CROWN-IBP target slice.
    pub PER_NODE_FLOOR_SECS = LeverDecl {
        name: "NY_PER_NODE_FLOOR_SECS",
        // The historical parser is untrimmed parse::<f64>(), finite, > 0.
        kind: LeverKind::F64Open { min: 0.0, max: f64::MAX },
        // Absence retains the contextual built-in 2 s floor.  It is not a
        // registry default because the collector, rather than the parser,
        // owns that fallback.
        default: DefaultSpec::Unset,
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Minimum per-node share, in seconds, below which a CROWN-IBP target declines and \
keeps its certified IBP fallback. The typed competition route is \
`bab.crown_ibp_per_node_floor_secs`; the legacy environment token wins when it \
is a finite untrimmed number strictly above zero. Absence is execution-equivalent \
to the collector's built-in 2-second floor. A larger floor can skip more targets \
and a smaller floor can spend time on work that does not finish, so it changes \
which certified bounds and verdicts are reached before the deadline even though \
every individual fallback remains sound.",
        provenance: Provenance::Unmeasured {
            why_ok: "absence preserves the established 2-second collector floor; explicit values only schedule independently sound CROWN work versus its certified IBP fallback",
        },
        owner: CROWN_IBP_BUDGET,
        readers: READERS,
    };

    /// `NY_PER_NODE_CAP_SECS` — fixed cap replacing the adaptive cap.
    pub PER_NODE_CAP_SECS = LeverDecl {
        name: "NY_PER_NODE_CAP_SECS",
        // Same exact legacy parser as the floor: no trimming, finite, > 0.
        kind: LeverKind::F64Open { min: 0.0, max: f64::MAX },
        // Absence is semantically distinct from every number: it selects 25%
        // of the live remaining budget, clamped to 12..=600 seconds.
        default: DefaultSpec::Unset,
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Explicit base cap, in seconds, for one CROWN-IBP target. The typed competition \
route is `bab.crown_ibp_per_node_cap_secs`; absence selects the adaptive policy \
(25% of the remaining collection budget, clamped to 12..=600 seconds), so it \
cannot be represented by a magic numeric default. Explicit caps are still \
dimension-scaled above the 28,800-row reference width. This operator override \
was introduced because the historical 12-second cap accumulated \
`PerNodeDeadlineExceeded` fallbacks, but it was omitted from the governed search \
schema. Cap choice changes which sound tightening work completes before the \
deadline and is therefore verdict-affecting despite never weakening the \
validity of a completed bound.",
        provenance: Provenance::Unmeasured {
            why_ok: "absence preserves the adaptive shipped policy and every explicit cap merely schedules sound CROWN completion versus a certified IBP fallback",
        },
        owner: CROWN_IBP_BUDGET,
        readers: READERS,
    };

    /// `NY_CROWN_OBJ_CHUNK` — explicit objective-row chunk width.
    pub CROWN_OBJ_CHUNK = LeverDecl {
        name: "NY_CROWN_OBJ_CHUNK",
        kind: LeverKind::UsizeTrimmed,
        default: DefaultSpec::U64(0),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Explicit objective-row chunk width for graph CROWN backward. Absence, zero, or \
malformed input resolve to zero, which leaves width selection to the ordinary \
single-pass/deadline-safety policy. A positive value requests that many rows \
before the finite ConvTranspose cap and adaptive driver are applied.

This is a coupled resource knob, not a free batching coordinate: it changes \
prefix replay count, deadline polling granularity, retained chunks, and the \
frontier size seen by memory admission. It remains available for controlled \
measurement but is excluded from automated search until it has typed delivery \
and a closed memory/time receipt.",
        provenance: Provenance::Unmeasured {
            why_ok: "zero preserves the established automatic/single-pass width policy; positive overrides remain explicit research inputs",
        },
        owner: CROWN_IBP_BUDGET,
        readers: OBJECTIVE_CHUNK_WIDTH_READERS,
    };

    /// `NY_NO_CHUNK_ABORT` — rollback of projected deadline abort.
    pub NO_CHUNK_ABORT = LeverDecl {
        name: "NY_NO_CHUNK_ABORT",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Rollback for measured-projection early abort. Absence, exact `0`, and malformed \
input keep the production abort policy enabled. Exact `1` disables it and \
restores the legacy loop that can spend the entire per-node share even when \
measured chunk throughput projects that the target cannot finish. This control \
is coupled to chunk width, adaptive growth, salvage, and the per-node deadline; \
it is retained for controlled comparison and excluded from automated search.",
        provenance: Provenance::Unmeasured {
            why_ok: "the false default preserves the production projected-abort policy; only an explicit exact-one research override restores the historical full-budget loop",
        },
        owner: CROWN_IBP_BUDGET,
        readers: OBJECTIVE_CHUNK_ABORT_READERS,
    };

    /// `NY_NO_CHUNK_GROW` — rollback of adaptive objective-chunk widening.
    pub NO_CHUNK_GROW = LeverDecl {
        name: "NY_NO_CHUNK_GROW",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Rollback for adaptive objective-chunk widening. Absence, exact `0`, and \
malformed input keep adaptive growth enabled. Exact `1` restores the legacy \
fixed-width loop. The rollback can multiply full-prefix replay and change how \
much certified work finishes before the deadline, so it is a coupled research \
control rather than an independently searchable parameter.",
        provenance: Provenance::Unmeasured {
            why_ok: "the false default preserves adaptive widening; only an explicit exact-one research override restores fixed-width replay",
        },
        owner: CROWN_IBP_BUDGET,
        readers: OBJECTIVE_CHUNK_GROW_READERS,
    };

    /// `NY_NO_CHUNK_WAVE_PAR` — aggregate-memory guard for objective chunks.
    pub NO_CHUNK_WAVE_PAR = LeverDecl {
        name: "NY_NO_CHUNK_WAVE_PAR",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(true),
        bucket: Bucket::DefaultOn,
        moat: MoatRisk::High,
        doc: "\
Aggregate-memory guard for graph CROWN objective chunks. Absence and exact `1` \
select the production sequential adaptive driver, so at most one complete \
backward frontier is live. Exact `0` disarms the guard and admits the legacy \
fixed-wave/anchor-parallel experiment. Every other token is rejected to the \
armed default and recorded as such.

WHY THIS IS A GUARD. The old wave cap divided a row ceiling by the initial \
chunk width, but every worker independently admitted against the full \
request-local Dense budget. That quotient did not reserve Patches maps and \
scratch, deep bound clones, retained output chunks, coefficient-error state, \
or GPU/GEMM/ConvTranspose workspaces. On the 28,800-row cGAN target, the \
reachable policy could therefore launch 32 independently full-budget \
frontiers while describing the row quotient as a memory cap. Production keeps \
one frontier live until a shared reservation and complete per-worker byte \
receipt exist. The `0` arm remains only for controlled comparison and must \
never be selected by automated search.",
        provenance: Provenance::Guard {
            protects: "the verifier process and host from aggregate objective-chunk memory overcommit by keeping one unreceipted backward frontier live",
        },
        owner: CROWN_IBP_BUDGET,
        readers: OBJECTIVE_CHUNK_WAVE_READERS,
    };

    /// `NY_PATCHES_DEADLINE_FLAT_BIAS` — checked flat-row bias reduction.
    pub PATCHES_DEADLINE_FLAT_BIAS = LeverDecl {
        name: "NY_PATCHES_DEADLINE_FLAT_BIAS",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Default-dark finite-deadline explicit-row Patches bias reduction. Exact `1` \
selects checked flat row slabs; absence, exact `0`, and malformed values keep \
the historical dynamic-index fold. Rows retain the same coefficient order and \
publish only after both sides complete, but scheduling can change which CROWN \
target finishes before its deadline. Runtime and truncated-cache identity both \
use the same central parser. The organizer-facing wrapper scrubs this research \
override; the lower developer wrapper preserves it for sealed A/B work.",
        provenance: Provenance::Unmeasured {
            why_ok: "false preserves the historical dynamic-index reduction; the flat-row arm remains an explicit research treatment",
        },
        owner: PATCHES_DEADLINE_SCOPE,
        readers: PATCHES_DEADLINE_FLAT_BIAS_READERS,
    };

    /// `NY_CROWN_HONEST_PROVENANCE` — quality-publication guard.
    pub CROWN_HONEST_PROVENANCE = LeverDecl {
        name: "NY_CROWN_HONEST_PROVENANCE",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(true),
        bucket: Bucket::DefaultOn,
        moat: MoatRisk::High,
        doc: "\
Guard preventing a completed CROWN target that lost nearly all finite IBP \
information from receiving Crown/tight provenance. Absence and exact `1` keep \
the guard armed; exact `0` restores unconditional historical publication; \
malformed input is rejected to the armed default. Finite-but-regressed output \
is reported as a strategy signal but intentionally remains Crown provenance, \
matching the original measured-quality policy. Runtime plus complete- and \
truncated-cache identity use the same central parser. The organizer-facing \
wrapper scrubs this rollback; the lower developer wrapper preserves it for \
sealed A/B work.",
        provenance: Provenance::Guard {
            protects: "bound provenance and downstream cut/publication decisions from treating an almost-entirely non-finite CROWN result as a genuine tightening",
        },
        owner: CROWN_PROVENANCE_SCOPE,
        readers: CROWN_HONEST_PROVENANCE_READERS,
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{read_over_config_with, LeverValue, Source};

    #[test]
    fn positive_seconds_parser_matches_the_legacy_contract() {
        for decl in [&PER_NODE_FLOOR_SECS, &PER_NODE_CAP_SECS] {
            let accepted = read_over_config_with(
                decl,
                |_| Some("12.5".to_owned()),
                Some(LeverValue::F64(3.0)),
            )
            .expect("typed value is admissible");
            assert_eq!(accepted.value, LeverValue::F64(12.5));
            assert_eq!(accepted.source, Source::LegacyEnv);

            for raw in ["0", "-1", "NaN", "inf", " 2", "2 ", ""] {
                let rejected = read_over_config_with(
                    decl,
                    |_| Some(raw.to_owned()),
                    Some(LeverValue::F64(3.0)),
                )
                .expect("invalid environment input resolves fail-closed");
                assert_eq!(rejected.value, LeverValue::Unset, "raw={raw:?}");
                assert_eq!(rejected.source, Source::LegacyEnvRejected, "raw={raw:?}");
            }
        }
    }

    #[test]
    fn typed_value_is_used_only_when_environment_is_absent() {
        for decl in [&PER_NODE_FLOOR_SECS, &PER_NODE_CAP_SECS] {
            let configured = read_over_config_with(decl, |_| None, Some(LeverValue::F64(19.0)))
                .expect("typed value is admissible");
            assert_eq!(configured.value, LeverValue::F64(19.0));
            assert_eq!(configured.source, Source::Config);

            let absent = read_over_config_with(decl, |_| None, None)
                .expect("absence retains the contextual runtime policy");
            assert_eq!(absent.value, LeverValue::Unset);
            assert_eq!(absent.source, Source::Default);
        }
    }

    #[test]
    fn objective_chunk_wave_guard_accepts_only_exact_boolean_overrides() {
        let default = read_over_config_with(&NO_CHUNK_WAVE_PAR, |_| None, None)
            .expect("the guard default is admissible");
        assert_eq!(default.value, LeverValue::Bool(true));
        assert_eq!(default.source, Source::Default);

        for (raw, expected) in [("0", false), ("1", true)] {
            let resolved =
                read_over_config_with(&NO_CHUNK_WAVE_PAR, |_| Some(raw.to_owned()), None)
                    .expect("an exact Boolean override is admissible");
            assert_eq!(resolved.value, LeverValue::Bool(expected), "raw={raw:?}");
            assert_eq!(resolved.source, Source::LegacyEnv, "raw={raw:?}");
        }

        for raw in ["", "true", "00", " 0", "0 "] {
            let rejected =
                read_over_config_with(&NO_CHUNK_WAVE_PAR, |_| Some(raw.to_owned()), None)
                    .expect("malformed input fails closed to the guard default");
            assert_eq!(rejected.value, LeverValue::Bool(true), "raw={raw:?}");
            assert_eq!(rejected.source, Source::LegacyEnvRejected, "raw={raw:?}");
            assert_eq!(rejected.rejected_raw.as_deref(), Some(raw));
        }
    }

    #[test]
    fn objective_chunk_width_and_rollbacks_preserve_their_legacy_parsers() {
        for (raw, expected) in [
            (None, 0),
            (Some(""), 0),
            (Some("0"), 0),
            (Some(" 32 "), 32),
            (Some("bad"), 0),
        ] {
            let resolved =
                read_over_config_with(&CROWN_OBJ_CHUNK, |_| raw.map(str::to_owned), None)
                    .expect("chunk-width resolution is total");
            assert_eq!(resolved.value.as_u64(), Some(expected), "raw={raw:?}");
        }

        for decl in [&NO_CHUNK_ABORT, &NO_CHUNK_GROW] {
            for (raw, expected) in [
                (None, false),
                (Some("0"), false),
                (Some("1"), true),
                (Some("true"), false),
            ] {
                let resolved = read_over_config_with(decl, |_| raw.map(str::to_owned), None)
                    .expect("rollback resolution is total");
                assert_eq!(
                    resolved.value,
                    LeverValue::Bool(expected),
                    "{} raw={raw:?}",
                    decl.name
                );
            }
        }
    }

    #[test]
    fn adjacent_crown_cache_policies_resolve_to_effective_booleans() {
        for decl in [&PATCHES_DEADLINE_FLAT_BIAS] {
            for (raw, expected) in [
                (None, false),
                (Some("0"), false),
                (Some("1"), true),
                (Some("true"), false),
            ] {
                let resolved = read_over_config_with(decl, |_| raw.map(str::to_owned), None)
                    .expect("deadline-kernel resolution is total");
                assert_eq!(
                    resolved.value,
                    LeverValue::Bool(expected),
                    "{} raw={raw:?}",
                    decl.name
                );
            }
        }

        for (raw, expected) in [
            (None, true),
            (Some("0"), false),
            (Some("1"), true),
            (Some("true"), true),
        ] {
            let resolved =
                read_over_config_with(&CROWN_HONEST_PROVENANCE, |_| raw.map(str::to_owned), None)
                    .expect("provenance-guard resolution is total");
            assert_eq!(resolved.value, LeverValue::Bool(expected), "raw={raw:?}");
        }
    }
}
