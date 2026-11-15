// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! One frozen decision governs structural raw loading and the query owner.

use crate::{
    declare_levers, Bucket, DefaultSpec, LeverDecl, LeverKind, MoatRisk, Provenance, ReaderSite,
    Scope,
};

const OWNER: Scope = Scope {
    package: "ny-cli",
    subsystem: "certified-original-root-cut",
};

declare_levers! {
    registry ROOT_CERTIFIED_CUT_LEVERS;

    /// Default-on structural raw-source admission; only literal `0` disables.
    pub ROOT_CERTIFIED_CUT = LeverDecl {
        name: "NY_ROOT_CERTIFIED_CUT",
        kind: LeverKind::BoolKillOverride,
        default: DefaultSpec::Bool(true),
        bucket: Bucket::DefaultOn,
        moat: MoatRisk::High,
        doc: "One frozen per-query boolean governs both structural PreserveRaw loading and \
              the once-only checked-root-cut owner. Only the exact environment token 0 \
              disables it; other tokens leave the default/lower typed policy authoritative. \
              Enabling the policy does not authenticate a graph or guarantee admission. \
              Original raw-source provenance, same-call checked support, exact facet \
              validation, complete downward replay, prepaid resources and the original \
              clock are still required. The new floor can only replace an ordinary \
              root lower bound after all checks, with its upper word unchanged. \
              High risk reflects changed proof authority, load policy and wall allocation; \
              this is not a measured speedup or a competition-points claim.",
        provenance: Provenance::OwnerDirected {
            directive: "The user requests useful verifier improvements default on so they \
                        receive attention and fixes; root explicitly authorizes this \
                        structurally gated checked-root route with literal0 rollback",
            limitations: "Default-on is an owner-directed policy, not measured superiority. \
                          Raw-source loading and complete same-call certification are \
                          mandatory. Toy separation and arithmetic controls do not prove \
                          original CCT engagement or end-to-end competition improvement",
        },
        owner: OWNER,
        readers: &[
            ReaderSite {
                scope: OWNER,
                role: "resolve the entry-time receipt-owning LeverSet value once and share it between raw-load selection and the query owner",
                site: "crates/ny-cli/src/commands/vnncomp.rs:run_and_translate",
            },
            ReaderSite {
                scope: OWNER,
                role: "consume the scored frozen value or perform one interactive freeze, before structural loading and owner construction",
                site: "crates/ny-cli/src/commands/beta_crown/mod.rs:resolve_certified_root_cut_policy",
            },
        ],
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LeverSet, Source};
    use std::cell::RefCell;

    fn frozen(raw: Option<&str>) -> LeverSet {
        LeverSet::freeze_with(crate::all(), |name| {
            (name == ROOT_CERTIFIED_CUT.name)
                .then(|| raw.map(str::to_owned))
                .flatten()
        })
    }

    #[test]
    fn default_and_literal_zero_preserve_the_exact_rollback_contract() {
        let default = frozen(None);
        assert!(default.get(&ROOT_CERTIFIED_CUT).as_bool());
        assert_eq!(
            default.resolved(&ROOT_CERTIFIED_CUT).unwrap().source,
            Source::Default
        );
        let disabled = frozen(Some("0"));
        assert!(!disabled.get(&ROOT_CERTIFIED_CUT).as_bool());
        assert_eq!(
            disabled.resolved(&ROOT_CERTIFIED_CUT).unwrap().source,
            Source::LegacyEnv
        );
        for token in ["", "1", "false", "off", "00", " 0", "0 ", "\n0"] {
            let set = frozen(Some(token));
            assert!(set.get(&ROOT_CERTIFIED_CUT).as_bool(), "token={token:?}");
            let resolved = set.resolved(&ROOT_CERTIFIED_CUT).unwrap();
            assert_eq!(resolved.source, Source::Default);
            assert_eq!(resolved.ignored_raw.as_deref(), Some(token));
        }
    }

    #[test]
    fn frozen_query_value_survives_later_lookup_mutation_in_both_directions() {
        let raw = RefCell::new(None::<String>);
        let before = LeverSet::freeze_with(crate::all(), |name| {
            if name == ROOT_CERTIFIED_CUT.name {
                raw.borrow().clone()
            } else {
                None
            }
        });
        raw.replace(Some("0".into()));
        assert!(before.get(&ROOT_CERTIFIED_CUT).as_bool());
        let disabled = LeverSet::freeze_with(crate::all(), |name| {
            if name == ROOT_CERTIFIED_CUT.name {
                raw.borrow().clone()
            } else {
                None
            }
        });
        raw.replace(Some("1".into()));
        assert!(!disabled.get(&ROOT_CERTIFIED_CUT).as_bool());
        assert!(before.get(&ROOT_CERTIFIED_CUT).as_bool());
        let later = LeverSet::freeze_with(crate::all(), |name| {
            if name == ROOT_CERTIFIED_CUT.name {
                raw.borrow().clone()
            } else {
                None
            }
        });
        assert!(later.get(&ROOT_CERTIFIED_CUT).as_bool());
    }

    #[test]
    fn receipt_discloses_owner_direction_and_the_actual_frozen_choice() {
        for (raw, enabled) in [(None, true), (Some("0"), false), (Some("false"), true)] {
            let set = frozen(raw);
            let receipt = set.receipt();
            let entry = receipt["levers"]
                .as_array()
                .unwrap()
                .iter()
                .find(|entry| entry["name"].as_str() == Some(ROOT_CERTIFIED_CUT.name))
                .unwrap();
            assert_eq!(entry["value"], enabled);
            assert_eq!(entry["provenance"], "owner_directed");
            assert_eq!(set.get(&ROOT_CERTIFIED_CUT).as_bool(), enabled);
        }
        assert_eq!(ROOT_CERTIFIED_CUT.moat, MoatRisk::High);
        assert_eq!(ROOT_CERTIFIED_CUT.bucket, Bucket::DefaultOn);
    }
}
