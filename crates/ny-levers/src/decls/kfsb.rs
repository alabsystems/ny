// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Governance for the multi-objective wave-batched kFSB selector.
//!
//! `NY_MO_KFSB` is the canonical master and has a typed delivery path through
//! `bab.branching.kfsb_multi`. Its live candidate-count and reduction controls
//! are delivered by `bab.branching.kfsb_multi_candidates` and
//! `bab.branching.kfsb_multi_reduceop`, with the existing generic branching
//! fields as compatibility fallbacks. `NY_BRANCH_KFSB_CHILDSIM` predates that
//! master and is deliberately declared separately: it is not a spelling alias.
//! When the canonical environment variable is absent, the legacy switch both
//! arms the selector and changes an otherwise-unspecified candidate count to
//! two.

use crate::{
    declare_levers, Bucket, DefaultSpec, LeverDecl, LeverKind, MoatRisk, Provenance, ReaderSite,
    Scope,
};

const KFSB_SCOPE: Scope = Scope {
    package: "ny-propagate",
    subsystem: "multi-objective-kfsb",
};

const KFSB_RESOLVER_SCOPE: Scope = Scope {
    package: "ny-levers",
    subsystem: "kfsb-gate-resolution",
};

const KFSB_OUTPUT_SCOPE: Scope = Scope {
    package: "ny-cli",
    subsystem: "beta-crown-effective-treatment",
};

const KFSB_PRESET_SCOPE: Scope = Scope {
    package: "ny-cli",
    subsystem: "beta-crown-preset-delivery",
};

const CANONICAL_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: KFSB_RESOLVER_SCOPE,
        role: "layer the canonical exact-0/1 environment master over typed config \
               and suppress the legacy compound switch on any canonical presence",
        site: "crates/ny-levers/src/decls/kfsb.rs:resolve_gate",
    },
    ReaderSite {
        scope: KFSB_SCOPE,
        role: "consume the shared resolution to admit the wave-batched selector \
               and its round-aware typed Select policy",
        site: "crates/ny-propagate/src/beta_crown/engine/graph/multi_objective/\
               batched/kfsb_multi.rs:kfsb_multi_gate_receipt and \
               BetaCrownVerifier::kfsb_multi_wave_policy_at_round",
    },
    ReaderSite {
        scope: KFSB_OUTPUT_SCOPE,
        role: "project the same effective canonical override into the treatment receipt",
        site: "crates/ny-cli/src/commands/beta_crown/output.rs:\
               RuntimeTreatmentGates::capture",
    },
];

const LEGACY_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: KFSB_RESOLVER_SCOPE,
        role: "read the compatibility fallback only when the canonical master is absent",
        site: "crates/ny-levers/src/decls/kfsb.rs:resolve_gate",
    },
    ReaderSite {
        scope: KFSB_SCOPE,
        role: "consume legacy admission and its implicit candidate count of two",
        site: "crates/ny-propagate/src/beta_crown/engine/graph/multi_objective/\
               batched/kfsb_multi.rs:kfsb_multi_gate_receipt, \
               kfsb_multi_candidate_count_with_gate, and \
               BetaCrownVerifier::kfsb_multi_wave_policy_at_round",
    },
    ReaderSite {
        scope: KFSB_OUTPUT_SCOPE,
        role: "project legacy admission and its implicit candidate count into the \
               effective-treatment receipt",
        site: "crates/ny-cli/src/commands/beta_crown/output.rs:\
               RuntimeTreatmentGates::capture",
    },
];

const CANDIDATE_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: KFSB_PRESET_SCOPE,
        role: "deliver bab.branching.kfsb_multi_candidates with the generic candidate count \
               as its compatibility fallback",
        site: "crates/ny-cli/src/preset/apply.rs:apply_branching_preset",
    },
    ReaderSite {
        scope: KFSB_PRESET_SCOPE,
        role: "project the dedicated and generic candidate fields through the typed \
               delivery receipt",
        site: "crates/ny-cli/src/commands/vnncomp.rs:preset_delivered_config_value",
    },
    ReaderSite {
        scope: KFSB_RESOLVER_SCOPE,
        role: "layer the exact integer environment override over the configured candidate count",
        site: "crates/ny-levers/src/decls/kfsb.rs:resolve_candidate_count",
    },
    ReaderSite {
        scope: KFSB_SCOPE,
        role: "bound the candidate simulation portfolio and decline a zero-width lane",
        site: "crates/ny-propagate/src/beta_crown/engine/graph/multi_objective/\
               batched/kfsb_multi.rs:kfsb_multi_candidate_count_with_gate and \
               BetaCrownVerifier::select_graph_branch_kfsb_multi_batched_with_policy",
    },
    ReaderSite {
        scope: KFSB_OUTPUT_SCOPE,
        role: "project the same effective candidate count into the treatment receipt",
        site: "crates/ny-cli/src/commands/beta_crown/output.rs:\
               RuntimeTreatmentGates::capture",
    },
];

const REDUCE_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: KFSB_PRESET_SCOPE,
        role: "deliver bab.branching.kfsb_multi_reduceop with the generic reduce policy \
               as its compatibility fallback",
        site: "crates/ny-cli/src/preset/apply.rs:apply_branching_preset",
    },
    ReaderSite {
        scope: KFSB_PRESET_SCOPE,
        role: "project the dedicated and generic reduce fields through the typed \
               delivery receipt",
        site: "crates/ny-cli/src/commands/vnncomp.rs:preset_delivered_config_value",
    },
    ReaderSite {
        scope: KFSB_RESOLVER_SCOPE,
        role: "layer the exact min/max environment override over the configured reduce policy",
        site: "crates/ny-levers/src/decls/kfsb.rs:resolve_reduce_policy",
    },
    ReaderSite {
        scope: KFSB_SCOPE,
        role: "capture one governed wave policy and consume that exact reducer for \
               each simulated candidate's two certified child scores",
        site: "crates/ny-propagate/src/beta_crown/engine/graph/multi_objective/\
               batched/kfsb_multi.rs:kfsb_multi_reduce_policy, \
               BetaCrownVerifier::kfsb_multi_wave_policy_at_round, and \
               BetaCrownVerifier::select_graph_branch_kfsb_multi_batched_with_policy",
    },
    ReaderSite {
        scope: KFSB_OUTPUT_SCOPE,
        role: "project the same effective reduction policy into the treatment receipt",
        site: "crates/ny-cli/src/commands/beta_crown/output.rs:\
               RuntimeTreatmentGates::capture",
    },
];

const CERT_REUSE_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: KFSB_PRESET_SCOPE,
        role: "deliver bab.branching.kfsb_cert_reuse into BetaCrownConfig and project it \
               through the generic typed flight receipt",
        site: "crates/ny-cli/src/preset/apply.rs:apply_branching_preset and \
               crates/ny-cli/src/commands/vnncomp.rs:preset_delivered_config_value",
    },
    ReaderSite {
        scope: KFSB_RESOLVER_SCOPE,
        role: "layer the exact Boolean environment override over interactive typed config",
        site: "crates/ny-levers/src/decls/kfsb.rs:resolve_cert_reuse",
    },
    ReaderSite {
        scope: KFSB_PRESET_SCOPE,
        role: "consume the scored run's frozen LeverSet value as an immutable instance override",
        site: "crates/ny-cli/src/commands/vnncomp.rs:run_and_translate and \
               crates/ny-cli/src/commands/beta_crown/mod.rs:apply_instance_overrides",
    },
    ReaderSite {
        scope: KFSB_SCOPE,
        role: "admit proof-carrying historical child certificates under the scored \
               frozen policy or the governed interactive compatibility policy",
        site: "crates/ny-propagate/src/beta_crown/engine/graph/multi_objective/\
               batched/kfsb_multi.rs:\
               select_graph_branch_kfsb_multi_batched_with_policy",
    },
    ReaderSite {
        scope: KFSB_OUTPUT_SCOPE,
        role: "project configured and effective certificate-reuse policy separately",
        site: "crates/ny-cli/src/commands/beta_crown/output.rs:\
               RuntimeTreatmentGates::capture",
    },
];

declare_levers! {
    registry KFSB_LEVERS;

    /// Canonical wave-batched multi-objective kFSB master.
    pub MO_KFSB = LeverDecl {
        name: "NY_MO_KFSB",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Canonical exact-`\"1\"` arm and exact-`\"0\"` kill switch for the wave-batched \
multi-objective kFSB selector. The lower configuration layer is the typed \
`BetaCrownConfig::use_kfsb_multi_branching` value delivered by \
`bab.branching.kfsb_multi`. An admissible environment value overrides that \
typed value in either direction. A present malformed or non-UTF-8 value is a \
recorded rejection and resolves to this declaration's dark default, suppressing \
both the typed arm and the legacy compatibility switch.

This selector evaluates prospective children and changes which split is \
committed. Every resulting child bound remains independently certified, but \
branch order changes which domains are reached before the deadline and can \
therefore move a verdict. It is High-risk and dark by declaration; selected \
presets opt in through their typed field.",
        provenance: Provenance::Unmeasured {
            why_ok: "the declaration default is dark; benchmark-specific typed presets \
                     opt in separately, and exact 0 remains an environment rollback",
        },
        owner: KFSB_SCOPE,
        readers: CANONICAL_READERS,
    };

    /// Candidate budget for the wave-batched multi-objective kFSB selector.
    pub MO_KFSB_K = LeverDecl {
        name: "NY_MO_KFSB_K",
        kind: LeverKind::U64,
        default: DefaultSpec::U64(8),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Candidate count used by the wave-batched multi-objective kFSB child simulator. \
The historical environment parser is an exact, untrimmed unsigned integer \
parse: zero is admissible and declines the lane; whitespace, signs that the \
unsigned parser refuses, overflow, and non-UTF-8 are rejections. An admissible \
environment value overrides `bab.branching.kfsb_multi_candidates` in either \
direction. Absence preserves that dedicated typed count, or inherits ordinary \
`bab.branching.candidates` when the dedicated field is unset for shipped-preset \
compatibility. A present rejected value suppresses both typed layers and resolves \
to the historical `BetaCrownConfig` default of eight.

When the absence-only legacy `NY_BRANCH_KFSB_CHILDSIM=1` path owns admission, \
an absent candidate override retains its compound implicit count of two. Any \
present candidate token, including a rejected one, owns the candidate layer \
instead. Candidate width changes simulation cost and the evaluated split set, \
so it can change which certified domains are reached before the deadline. It \
is searchable only together with the canonical `NY_MO_KFSB` master.",
        provenance: Provenance::Measured {
            commit: "a5b6ce356",
            date: "2026-07-16",
            artifact: "commit a5b6ce356 message and retained kFSB multi-objective A/B logs",
            delta: "the first isolated candidate-width A/B found k=7 halved depth \
                    throughput; later childsim measurement selected k=2 as the \
                    throughput/quality point, while the declared default eight \
                    remains the pre-existing contextual solver default rather \
                    than a claimed optimum",
        },
        owner: KFSB_SCOPE,
        readers: CANDIDATE_READERS,
    };

    /// Two-child score reduction for the wave-batched kFSB selector.
    pub MO_KFSB_REDUCE = LeverDecl {
        name: "NY_MO_KFSB_REDUCE",
        kind: LeverKind::LayeredEnum {
            env_members: &["min", "max"],
            config_members: &["min", "max", "mean"],
        },
        default: DefaultSpec::Text("min"),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Reduction used to turn a candidate split's two simulated child bounds into one \
ranking score. The historical environment parser accepts only exact lowercase \
`min` and `max`; it deliberately does not accept `mean`, whitespace, alternate \
case, or non-UTF-8. The dedicated typed \
`bab.branching.kfsb_multi_reduceop` surface remains broader and may select \
`min`, `max`, or `mean`; when it is unset, absence inherits the ordinary \
`bab.branching.reduceop` policy for shipped-preset compatibility. A present \
admissible environment value overrides either typed source. A present token \
outside the narrow environment domain is receipted as rejected, suppresses the \
typed layer, and fails to the conservative declaration default `min`.

All three operations rank already certified child scores and cannot manufacture \
a bound, but they can commit different splits and change the deadline-bounded \
verdict. Automated search may therefore vary only exact `min|max`, and only \
alongside the canonical `NY_MO_KFSB` master.",
        provenance: Provenance::Measured {
            commit: "a5b6ce356",
            date: "2026-07-16",
            artifact: "commit a5b6ce356 message and retained kFSB multi-objective A/B logs",
            delta: "the isolated Min/Max A/B found Max favored one-sided verified \
                    children rather than the worst-child objective; the result \
                    established that the policy is behaviorally live, not a \
                    category-wide winning default",
        },
        owner: KFSB_SCOPE,
        readers: REDUCE_READERS,
    };

    /// Proof-carrying reuse of certified wave-kFSB child results.
    pub MO_KFSB_CERT_REUSE = LeverDecl {
        name: "NY_MO_KFSB_CERT_REUSE",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Reuse a strictly-authorized scalar lower certificate produced by historical \
wave-kFSB child simulation. Exact `\"1\"` arms and exact `\"0\"` disarms; \
absence inherits the typed `bab.branching.kfsb_cert_reuse` value. Any other \
present token, including non-UTF-8, is a recorded rejection that suppresses the \
typed layer and restores this declaration's dark default.

An accepted receipt may close an objective row, a committed child, or an \
untouched parent and may skip later child evaluation. Parent-history identity, \
scope, finite lower values, and both receipt and publication deadlines are \
validated fail-closed, but the resulting pruning changes deadline-bounded \
verdict reachability. Scored VNN-COMP runs therefore consume the entry-time \
frozen LeverSet value rather than reopening the process environment.",
        provenance: Provenance::Unmeasured {
            why_ok: "default-dark outside the explicitly configured CIFAR100 preset; \
                     exact 0 remains an operator rollback, while scored optimization \
                     stays delivery-only until a certificate-acceptance T1 contract exists",
        },
        owner: KFSB_SCOPE,
        readers: CERT_REUSE_READERS,
    };

    /// Legacy compound child-simulation compatibility switch.
    pub BRANCH_KFSB_CHILDSIM = LeverDecl {
        name: "NY_BRANCH_KFSB_CHILDSIM",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Legacy exact-`\"1\"` compatibility switch, consulted only when `NY_MO_KFSB` \
is absent. It is deliberately NOT an alias for the canonical master: in \
addition to arming the same wave-batched child-simulation selector, it changes \
an otherwise-unspecified `NY_MO_KFSB_K` candidate budget to the measured legacy \
default of two. Exact `\"0\"`, malformed values, and absence do not override the \
typed canonical policy; a present canonical master of any spelling suppresses \
this legacy behavior completely.

Because one token changes both lane admission and candidate count, treating it \
as a second searchable spelling of `NY_MO_KFSB` would confound two mechanisms. \
It remains governed for receipt completeness and backwards compatibility, but \
the search space excludes it as an unsafe compound axis.",
        provenance: Provenance::Unmeasured {
            why_ok: "dark compatibility path retained for existing experiment commands; \
                     canonical typed delivery is the supported searchable treatment",
        },
        owner: KFSB_SCOPE,
        readers: LEGACY_READERS,
    };
}

/// Effective resolution of the canonical kFSB master family.
///
/// `env_override` is the environment-level answer a consumer that still has a
/// second typed policy (such as depth-two Select) must layer over that policy.
/// It is `Some(false)` for both canonical exact `0` and a rejected canonical
/// token: both suppress every lower configuration layer. `legacy_childsim` is
/// kept distinct because only that compatibility arm supplies implicit `k=2`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KfsbGateResolution {
    pub armed: bool,
    pub env_override: Option<bool>,
    pub legacy_childsim: bool,
}

fn env_present(resolved: &crate::Resolved) -> bool {
    matches!(
        resolved.source,
        crate::Source::LegacyEnv | crate::Source::LegacyEnvRejected
    )
}

/// Resolve an already-layered canonical value over its optional legacy
/// fallback. This is public so a future frozen [`crate::LeverSet`] consumer can
/// apply the identical cross-lever precedence without reopening the process
/// environment.
#[must_use]
pub fn resolve_gate_from_resolved(
    canonical: &crate::Resolved,
    legacy: Option<&crate::Resolved>,
) -> KfsbGateResolution {
    if env_present(canonical) {
        let armed = canonical.value.as_bool();
        return KfsbGateResolution {
            armed,
            env_override: Some(armed),
            legacy_childsim: false,
        };
    }

    let legacy_childsim = legacy.is_some_and(|resolved| resolved.value.as_bool());
    KfsbGateResolution {
        armed: canonical.value.as_bool() || legacy_childsim,
        env_override: legacy_childsim.then_some(true),
        legacy_childsim,
    }
}

/// Resolve the live canonical environment over a typed configuration arm,
/// consulting the legacy compound switch only when the canonical variable is
/// absent.
#[must_use]
pub fn resolve_gate(config_armed: bool) -> KfsbGateResolution {
    let canonical = crate::read_over_config(&MO_KFSB, Some(crate::LeverValue::Bool(config_armed)))
        .expect("a Bool kFSB config value must satisfy its Bool declaration");
    let legacy = (!env_present(&canonical)).then(|| crate::read(&BRANCH_KFSB_CHILDSIM));
    resolve_gate_from_resolved(&canonical, legacy.as_ref())
}

/// Apply candidate/environment layering plus the legacy compound switch.
///
/// An accepted or rejected environment token owns the candidate layer. On
/// absence, legacy childsim supplies two; otherwise the typed/configured value
/// (or declaration default eight) stands.
#[must_use]
pub fn resolve_candidate_count_from_resolved(
    candidate: &crate::Resolved,
    gate: KfsbGateResolution,
) -> usize {
    if !env_present(candidate) && gate.legacy_childsim {
        return 2;
    }
    candidate
        .value
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
        .unwrap_or(0)
}

/// Resolve the live candidate environment over the typed solver count.
#[must_use]
pub fn resolve_candidate_count(configured: usize, gate: KfsbGateResolution) -> usize {
    let candidate =
        crate::read_over_config(&MO_KFSB_K, Some(crate::LeverValue::U64(configured as u64)))
            .expect("a usize kFSB candidate count must satisfy its U64 declaration");
    resolve_candidate_count_from_resolved(&candidate, gate)
}

/// Governed representation of the kFSB two-child reduction policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KfsbReducePolicy {
    Min,
    Max,
    Mean,
}

impl KfsbReducePolicy {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Min => "min",
            Self::Max => "max",
            Self::Mean => "mean",
        }
    }
}

/// Convert a declaration resolution into the closed runtime reduction enum.
#[must_use]
pub fn resolve_reduce_policy_from_resolved(resolved: &crate::Resolved) -> KfsbReducePolicy {
    match resolved.value.as_str() {
        Some("min") => KfsbReducePolicy::Min,
        Some("max") => KfsbReducePolicy::Max,
        Some("mean") => KfsbReducePolicy::Mean,
        other => panic!(
            "{} resolved outside its declared policy domain: {other:?}",
            MO_KFSB_REDUCE.name
        ),
    }
}

/// Resolve the live narrow environment override over the broader typed policy.
#[must_use]
pub fn resolve_reduce_policy(configured: KfsbReducePolicy) -> KfsbReducePolicy {
    let resolved = crate::read_over_config(
        &MO_KFSB_REDUCE,
        Some(crate::LeverValue::Text(configured.as_str().to_owned())),
    )
    .expect("a closed kFSB reduction policy must satisfy its layered declaration");
    resolve_reduce_policy_from_resolved(&resolved)
}

/// Resolve proof-carrying kFSB certificate reuse over an interactive typed
/// configuration value. Scored VNN-COMP execution instead consumes the same
/// declaration from its entry-time frozen [`crate::LeverSet`].
#[must_use]
pub fn resolve_cert_reuse(configured: bool) -> bool {
    crate::read_over_config(
        &MO_KFSB_CERT_REUSE,
        Some(crate::LeverValue::Bool(configured)),
    )
    .expect("a Bool kFSB certificate-reuse config must satisfy its declaration")
    .value
    .as_bool()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LeverSet, Source};

    #[test]
    fn malformed_canonical_value_is_visible_as_a_receipt_rejection() {
        let set = LeverSet::freeze_with(crate::all(), |name| {
            (name == MO_KFSB.name).then(|| "true".to_owned())
        });
        let resolved = set.resolved(&MO_KFSB).expect("canonical declaration");
        assert!(!resolved.value.as_bool());
        assert_eq!(resolved.source, Source::LegacyEnvRejected);
        assert_eq!(resolved.rejected_raw.as_deref(), Some("true"));

        let receipt = set.receipt();
        assert_eq!(receipt["env_rejected"], 1);
        let entry = receipt["levers"]
            .as_array()
            .expect("receipt lever array")
            .iter()
            .find(|entry| entry["name"] == MO_KFSB.name)
            .expect("canonical receipt entry");
        assert_eq!(entry["source"], "legacy_env_rejected");
        assert_eq!(entry["rejected_raw"], "true");
        assert_eq!(entry["value"], false);
    }

    #[test]
    fn cross_lever_resolution_keeps_rejection_and_legacy_compound_semantics_distinct() {
        let canonical = |raw: Option<&str>, config_armed: bool| {
            let raw = raw.map(str::to_owned);
            crate::read_over_config_with(
                &MO_KFSB,
                move |_| raw,
                Some(crate::LeverValue::Bool(config_armed)),
            )
            .expect("Bool config")
        };
        let candidate = |raw: Option<&str>, configured: usize| {
            let raw = raw.map(str::to_owned);
            crate::read_over_config_with(
                &MO_KFSB_K,
                move |_| raw,
                Some(crate::LeverValue::U64(configured as u64)),
            )
            .expect("U64 candidate config")
        };
        let legacy_on = crate::read_with(&BRANCH_KFSB_CHILDSIM, |_| Some("1".to_owned()));

        let absent = canonical(None, false);
        let legacy = resolve_gate_from_resolved(&absent, Some(&legacy_on));
        assert_eq!(
            legacy,
            KfsbGateResolution {
                armed: true,
                env_override: Some(true),
                legacy_childsim: true,
            }
        );
        assert_eq!(
            resolve_candidate_count_from_resolved(&candidate(None, 7), legacy),
            2
        );
        assert_eq!(
            resolve_candidate_count_from_resolved(&candidate(Some("5"), 7), legacy),
            5
        );

        for raw in ["0", "true"] {
            let canonical = canonical(Some(raw), true);
            let resolved = resolve_gate_from_resolved(&canonical, Some(&legacy_on));
            assert_eq!(
                resolved,
                KfsbGateResolution {
                    armed: false,
                    env_override: Some(false),
                    legacy_childsim: false,
                },
                "{raw:?}"
            );
            assert_eq!(
                resolve_candidate_count_from_resolved(&candidate(None, 7), resolved),
                7
            );
        }
    }

    #[test]
    fn candidate_and_reduce_layers_preserve_exact_parsers_and_fail_to_defaults() {
        let candidate = |raw: Option<&str>, configured: u64| {
            let raw = raw.map(str::to_owned);
            crate::read_over_config_with(
                &MO_KFSB_K,
                move |_| raw,
                Some(crate::LeverValue::U64(configured)),
            )
            .expect("candidate config")
        };
        assert_eq!(candidate(None, 7).value.as_u64(), Some(7));
        assert_eq!(candidate(Some("0"), 7).value.as_u64(), Some(0));
        assert_eq!(candidate(Some("5"), 7).value.as_u64(), Some(5));
        for malformed in [" 5", "5 ", "-1", "five", ""] {
            let resolved = candidate(Some(malformed), 7);
            assert_eq!(resolved.value.as_u64(), Some(8), "{malformed:?}");
            assert_eq!(resolved.source, Source::LegacyEnvRejected, "{malformed:?}");
            assert_eq!(resolved.rejected_raw.as_deref(), Some(malformed));
        }

        let reduce = |raw: Option<&str>, configured: KfsbReducePolicy| {
            let raw = raw.map(str::to_owned);
            crate::read_over_config_with(
                &MO_KFSB_REDUCE,
                move |_| raw,
                Some(crate::LeverValue::Text(configured.as_str().to_owned())),
            )
            .expect("reduce config")
        };
        assert_eq!(
            resolve_reduce_policy_from_resolved(&reduce(None, KfsbReducePolicy::Mean)),
            KfsbReducePolicy::Mean,
            "typed mean remains valid even though the environment domain is narrower"
        );
        assert_eq!(
            resolve_reduce_policy_from_resolved(&reduce(Some("max"), KfsbReducePolicy::Min)),
            KfsbReducePolicy::Max
        );
        for malformed in ["mean", "MAX", " max", "max ", ""] {
            let resolved = reduce(Some(malformed), KfsbReducePolicy::Max);
            assert_eq!(
                resolve_reduce_policy_from_resolved(&resolved),
                KfsbReducePolicy::Min,
                "{malformed:?}"
            );
            assert_eq!(resolved.source, Source::LegacyEnvRejected, "{malformed:?}");
            assert_eq!(resolved.rejected_raw.as_deref(), Some(malformed));
        }
    }

    #[test]
    fn malformed_candidate_and_reduce_values_are_visible_in_the_receipt() {
        let set = LeverSet::freeze_with(crate::all(), |name| match name {
            name if name == MO_KFSB_K.name => Some("7 ".to_owned()),
            name if name == MO_KFSB_REDUCE.name => Some("mean".to_owned()),
            _ => None,
        });
        assert_eq!(set.source(&MO_KFSB_K), Source::LegacyEnvRejected);
        assert_eq!(set.get(&MO_KFSB_K).as_u64(), Some(8));
        assert_eq!(set.source(&MO_KFSB_REDUCE), Source::LegacyEnvRejected);
        assert_eq!(set.get(&MO_KFSB_REDUCE).as_str(), Some("min"));

        let receipt = set.receipt();
        assert_eq!(receipt["env_rejected"], 2);
        for (name, raw, value) in [
            (MO_KFSB_K.name, "7 ", serde_json::json!(8)),
            (MO_KFSB_REDUCE.name, "mean", serde_json::json!("min")),
        ] {
            let entry = receipt["levers"]
                .as_array()
                .expect("receipt lever array")
                .iter()
                .find(|entry| entry["name"] == name)
                .expect("governed kFSB parameter receipt entry");
            assert_eq!(entry["source"], "legacy_env_rejected");
            assert_eq!(entry["rejected_raw"], raw);
            assert_eq!(entry["value"], value);
        }
    }

    #[test]
    fn certificate_reuse_layering_is_exact_fail_closed_and_receipted() {
        let resolve = |raw: Option<&str>, configured: bool| {
            let raw = raw.map(str::to_owned);
            crate::read_over_config_with(
                &MO_KFSB_CERT_REUSE,
                move |_| raw,
                Some(crate::LeverValue::Bool(configured)),
            )
            .expect("Bool certificate-reuse config")
        };

        assert!(!resolve(None, false).value.as_bool());
        assert!(resolve(None, true).value.as_bool());
        assert!(resolve(Some("1"), false).value.as_bool());
        assert!(!resolve(Some("0"), true).value.as_bool());
        for malformed in ["", "01", "true", " 1", "1 "] {
            let resolved = resolve(Some(malformed), true);
            assert!(!resolved.value.as_bool(), "{malformed:?}");
            assert_eq!(resolved.source, Source::LegacyEnvRejected, "{malformed:?}");
            assert_eq!(resolved.rejected_raw.as_deref(), Some(malformed));
        }
    }

    #[cfg(unix)]
    #[test]
    fn certificate_reuse_non_unicode_override_suppresses_typed_arm() {
        use std::ffi::OsString;
        use std::os::unix::ffi::OsStringExt;

        let non_unicode = OsString::from_vec(vec![b'1', 0xff]);
        let raw = crate::RawLeverInputs::capture_with(crate::all(), |name| {
            (name == MO_KFSB_CERT_REUSE.name).then(|| non_unicode.clone())
        });
        let set = LeverSet::resolve_layered(crate::all(), &raw, |decl| {
            std::ptr::eq(decl, &raw const MO_KFSB_CERT_REUSE)
                .then_some(crate::LeverValue::Bool(true))
        })
        .expect("typed certificate-reuse arm is admissible");
        let resolved = set
            .resolved(&MO_KFSB_CERT_REUSE)
            .expect("certificate-reuse declaration is registered");
        assert!(!resolved.value.as_bool());
        assert_eq!(resolved.source, Source::LegacyEnvRejected);
        assert_eq!(resolved.env_utf8, Some(false));
        assert!(resolved.rejected_raw.is_some());
    }
}
