// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! [`LeverSet`] — the frozen, per-run configuration snapshot.

use crate::decl::LeverDecl;
use crate::env::{resolve_raw, LeverValue, RawLeverInputs, ResolveError, Resolved, Source};
use crate::registry::LeverRegistry;

/// A run's configuration, resolved once and then immutable.
///
/// This is the Phase-2 vehicle. Today's readers latch their `NY_*` value in a
/// process-wide `OnceLock`, which has two costs: a latch cannot differ between
/// two instances verified in one process (so per-instance automated selection
/// is structurally impossible), and tests can only influence it by mutating
/// the process environment behind a global mutex.
///
/// A `LeverSet` fixes both by construction. It is resolved from the
/// environment ONCE at run entry and then carried by value: it can differ per
/// run, and a test can build one directly with [`LeverSet::freeze_with`]
/// instead of touching the process.
///
/// Freezing is total — every declaration in the registry gets an entry, not
/// just the ones somebody read. During migration, [`LeverSet::receipt`] is an
/// exhaustive record of declared input resolution; it proves runtime arming
/// only after that same set is threaded to the reader.
#[derive(Debug, Clone)]
pub struct LeverSet {
    entries: Vec<Resolved>,
}

impl LeverSet {
    /// Resolve every declared lever against the process environment, once.
    ///
    /// After this returns, the process environment is irrelevant to this set:
    /// a later `set_var` cannot change a value it already resolved. That
    /// invisibility is the point, not a side effect — it is what makes an A/B
    /// measurement reproducible from its receipt.
    pub fn freeze(registry: &LeverRegistry) -> Self {
        let raw = RawLeverInputs::capture(registry);
        Self::resolve(registry, &raw)
    }

    /// Resolve every declared lever against an arbitrary lookup.
    ///
    /// Tests and (later) preset/CLI layering use this; `lookup` is called at
    /// most once per declaration and never again.
    pub fn freeze_with<F>(registry: &LeverRegistry, mut lookup: F) -> Self
    where
        F: FnMut(&str) -> Option<String>,
    {
        let raw = RawLeverInputs::capture_with(registry, |name| {
            lookup(name).map(std::ffi::OsString::from)
        });
        Self::resolve(registry, &raw)
    }

    /// Resolve a previously captured environment snapshot over declaration
    /// defaults, with no contextual config layer.
    pub fn resolve(registry: &LeverRegistry, raw: &RawLeverInputs) -> Self {
        Self::resolve_layered(registry, raw, |_| None)
            .expect("default-only lever resolution cannot reject typed config")
    }

    /// Resolve environment-over-config-over-default for every declaration.
    ///
    /// `raw` is authoritative: this function never consults the process
    /// environment. `config` supplies an optional typed preset/run-context
    /// value for each declaration and is called once per declaration. A
    /// present valid legacy environment value wins over config. A present
    /// invalid value suppresses config and resolves to the declaration
    /// default as [`Source::LegacyEnvRejected`].
    ///
    /// # Errors
    ///
    /// Returns [`ResolveError`] if `config` supplies a value incompatible with
    /// a declaration's kind or range. This is fallible so evidence capture can
    /// record configuration drift without panicking a scored verifier.
    ///
    /// # Panics
    ///
    /// Panics if `raw` was not captured over every declaration in `registry`.
    pub fn resolve_layered<F>(
        registry: &LeverRegistry,
        raw: &RawLeverInputs,
        mut config: F,
    ) -> Result<Self, ResolveError>
    where
        F: FnMut(&'static LeverDecl) -> Option<LeverValue>,
    {
        let entries = registry
            .all()
            .iter()
            .map(|decl| resolve_raw(decl, raw.get(decl), config(decl)))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { entries })
    }

    /// Number of resolved levers.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the set is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The resolution record for one declaration.
    ///
    /// `None` means the declaration was not in the registry this set was
    /// frozen from — a programming error at the call site, not a missing
    /// environment variable.
    pub fn resolved(&self, decl: &'static LeverDecl) -> Option<&Resolved> {
        self.entries.iter().find(|e| std::ptr::eq(e.decl, decl))
    }

    /// The frozen value for one declaration.
    ///
    /// # Panics
    ///
    /// Panics when the declaration was not part of the registry used to build
    /// this set. Silently returning its default would make an unregistered
    /// declaration look valid while omitting it from the receipt.
    pub fn get(&self, decl: &'static LeverDecl) -> LeverValue {
        self.resolved(decl)
            .unwrap_or_else(|| panic!("{} is not present in this LeverSet's registry", decl.name))
            .value
            .clone()
    }

    /// Boolean view of a gate.
    pub fn is_armed(&self, decl: &'static LeverDecl) -> bool {
        self.get(decl).as_bool()
    }

    /// Where the frozen value came from.
    ///
    /// # Panics
    ///
    /// Panics when the declaration was not part of this set's registry.
    pub fn source(&self, decl: &'static LeverDecl) -> Source {
        self.resolved(decl)
            .unwrap_or_else(|| panic!("{} is not present in this LeverSet's registry", decl.name))
            .source
    }

    /// Every resolved lever, in registry (name-sorted) order.
    pub fn entries(&self) -> &[Resolved] {
        &self.entries
    }

    /// The receipt: every registered lever's name, value and source.
    ///
    /// Phase 0c writes this into flight-v3 JSON after typed preset resolution.
    /// It proves frozen raw inputs and any projected typed config; runtime
    /// consumption becomes authoritative incrementally as readers migrate to
    /// this same set.
    /// `rejected_raw` appears only when the environment supplied a value the
    /// parser refused — a typo that silently did nothing is a first-class
    /// finding here, not a silence.
    pub fn receipt(&self) -> serde_json::Value {
        let levers: Vec<serde_json::Value> = self
            .entries
            .iter()
            .map(|e| {
                let mut obj = serde_json::Map::new();
                obj.insert(
                    "name".to_owned(),
                    serde_json::Value::String(e.decl.name.to_owned()),
                );
                obj.insert("value".to_owned(), e.value.to_json());
                obj.insert(
                    "source".to_owned(),
                    serde_json::Value::String(e.source.as_str().to_owned()),
                );
                obj.insert(
                    "bucket".to_owned(),
                    serde_json::Value::String(e.decl.bucket.as_str().to_owned()),
                );
                obj.insert(
                    "moat".to_owned(),
                    serde_json::Value::String(e.decl.moat.as_str().to_owned()),
                );
                obj.insert(
                    "provenance".to_owned(),
                    serde_json::Value::String(e.decl.provenance.tag().to_owned()),
                );
                if let Some(env_utf8) = e.env_utf8 {
                    obj.insert("env_utf8".to_owned(), serde_json::Value::Bool(env_utf8));
                }
                if let Some(raw) = &e.rejected_raw {
                    obj.insert(
                        "rejected_raw".to_owned(),
                        serde_json::Value::String(raw.clone()),
                    );
                }
                serde_json::Value::Object(obj)
            })
            .collect();

        let env_accepted = self
            .entries
            .iter()
            .filter(|entry| entry.source == Source::LegacyEnv)
            .count();
        let env_rejected = self
            .entries
            .iter()
            .filter(|entry| entry.source == Source::LegacyEnvRejected)
            .count();
        let env_present = env_accepted + env_rejected;

        serde_json::json!({
            "schema": "ny-levers/receipt/v2",
            "lever_count": levers.len(),
            "env_present": env_present,
            "env_accepted": env_accepted,
            "env_rejected": env_rejected,
            "levers": levers,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::LeverSet;
    use std::ffi::OsString;

    use crate::decl::{Bucket, DefaultSpec, LeverDecl, LeverKind, MoatRisk, Provenance, Scope};
    use crate::env::{LeverValue, RawLeverInputs, Source};
    use crate::registry::{collect, Registry};

    crate::declare_levers! {
        registry SET_SELFTEST_LEVERS;

        /// Boolean fixture.
        pub GATE = LeverDecl {
            name: "NY_LEVERS_SET_SELFTEST_GATE",
            kind: LeverKind::Bool,
            default: DefaultSpec::Bool(false),
            bucket: Bucket::Debug,
            moat: MoatRisk::None,
            doc: "set unit-test fixture",
            provenance: Provenance::Unmeasured { why_ok: "test" },
            owner: Scope { package: "ny-levers", subsystem: "selftest" },
            readers: &[],
        };

        /// Fraction fixture with no default.
        pub FRAC = LeverDecl {
            name: "NY_LEVERS_SET_SELFTEST_FRAC",
            kind: LeverKind::F64Open { min: 0.0, max: 0.9 },
            default: DefaultSpec::Unset,
            bucket: Bucket::Auto,
            moat: MoatRisk::Low,
            doc: "set unit-test fixture",
            provenance: Provenance::Unmeasured { why_ok: "test" },
            owner: Scope { package: "ny-levers", subsystem: "selftest" },
            readers: &[],
        };

        /// Presence-gated text fixture.
        pub PRESENCE = LeverDecl {
            name: "NY_LEVERS_SET_SELFTEST_PRESENCE",
            kind: LeverKind::Text,
            default: DefaultSpec::Unset,
            bucket: Bucket::Debug,
            moat: MoatRisk::None,
            doc: "set unit-test fixture",
            provenance: Provenance::Unmeasured { why_ok: "test" },
            owner: Scope { package: "ny-levers", subsystem: "selftest" },
            readers: &[],
        };
    }

    fn registry() -> crate::LeverRegistry {
        let regs: &[&'static Registry] = &[&SET_SELFTEST_LEVERS];
        collect(regs).expect("fixture registry merges")
    }

    #[test]
    fn freeze_is_total_over_the_registry() {
        let set = LeverSet::freeze_with(&registry(), |_| None);
        assert_eq!(set.len(), 3, "every declaration gets an entry, read or not");
        assert!(!set.is_empty());
        assert_eq!(set.source(&GATE), Source::Default);
    }

    #[test]
    fn a_later_lookup_change_is_invisible_to_a_frozen_set() {
        let before = LeverSet::freeze_with(&registry(), |_| None);
        assert!(!before.is_armed(&GATE), "unset must resolve to the default");

        assert!(
            !before.is_armed(&GATE),
            "a frozen set cannot perform a second lookup"
        );
        assert_eq!(before.source(&GATE), Source::Default);

        let after = LeverSet::freeze_with(&registry(), |name| {
            (name == GATE.name).then(|| "1".to_owned())
        });
        assert!(
            after.is_armed(&GATE),
            "a later snapshot sees the later lookup value"
        );
        assert_eq!(after.source(&GATE), Source::LegacyEnv);

        // Two snapshots, one process, different values — exactly what a
        // process-wide OnceLock latch cannot express.
        assert_ne!(before.is_armed(&GATE), after.is_armed(&GATE));
    }

    #[test]
    fn receipt_lists_every_lever_with_value_and_source() {
        let set = LeverSet::freeze_with(&registry(), |name| {
            if name == "NY_LEVERS_SET_SELFTEST_FRAC" {
                Some("0.25".to_owned())
            } else if name == "NY_LEVERS_SET_SELFTEST_PRESENCE" {
                None
            } else {
                Some("ture".to_owned())
            }
        });
        let receipt = set.receipt();

        assert_eq!(receipt["schema"], "ny-levers/receipt/v2");
        assert_eq!(receipt["lever_count"], 3);
        assert_eq!(receipt["env_present"], 2);
        assert_eq!(receipt["env_accepted"], 1);
        assert_eq!(receipt["env_rejected"], 1);

        let levers = receipt["levers"].as_array().expect("levers array");
        assert_eq!(levers.len(), 3);

        let frac = levers
            .iter()
            .find(|l| l["name"] == "NY_LEVERS_SET_SELFTEST_FRAC")
            .expect("frac present");
        assert_eq!(frac["value"], 0.25);
        assert_eq!(frac["source"], "legacy_env");
        assert_eq!(frac["bucket"], "auto");
        assert_eq!(frac["moat"], "low");
        assert!(frac.get("rejected_raw").is_none());

        let gate = levers
            .iter()
            .find(|l| l["name"] == "NY_LEVERS_SET_SELFTEST_GATE")
            .expect("gate present");
        assert_eq!(gate["value"], false);
        assert_eq!(gate["source"], "legacy_env_rejected");
        assert_eq!(
            gate["rejected_raw"], "ture",
            "a typo that silently did nothing must be visible in the receipt"
        );
        assert_eq!(gate["provenance"], "unmeasured");
    }

    #[test]
    fn layered_precedence_matrix_is_env_over_config_over_default() {
        struct Case {
            label: &'static str,
            raw: Option<&'static str>,
            config: Option<LeverValue>,
            want_value: LeverValue,
            want_source: Source,
        }

        let cases = [
            Case {
                label: "absent without config uses declaration default",
                raw: None,
                config: None,
                want_value: LeverValue::Unset,
                want_source: Source::Default,
            },
            Case {
                label: "absent exposes config",
                raw: None,
                config: Some(LeverValue::F64(0.25)),
                want_value: LeverValue::F64(0.25),
                want_source: Source::Config,
            },
            Case {
                label: "valid env replaces config",
                raw: Some("0.50"),
                config: Some(LeverValue::F64(0.25)),
                want_value: LeverValue::F64(0.50),
                want_source: Source::LegacyEnv,
            },
            Case {
                label: "invalid env kills config",
                raw: Some("not-a-fraction"),
                config: Some(LeverValue::F64(0.25)),
                want_value: LeverValue::Unset,
                want_source: Source::LegacyEnvRejected,
            },
        ];

        for case in cases {
            let raw = RawLeverInputs::capture_with(&registry(), |name| {
                (name == FRAC.name)
                    .then_some(case.raw)
                    .flatten()
                    .map(OsString::from)
            });
            let set = LeverSet::resolve_layered(&registry(), &raw, |decl| {
                (decl.name == FRAC.name)
                    .then(|| case.config.clone())
                    .flatten()
            })
            .expect("fixture config is admissible");
            assert_eq!(set.get(&FRAC), case.want_value, "{}", case.label);
            assert_eq!(set.source(&FRAC), case.want_source, "{}", case.label);
        }
    }

    #[test]
    fn invalid_bool_env_also_suppresses_a_lower_config() {
        let raw = RawLeverInputs::capture_with(&registry(), |name| {
            (name == GATE.name).then(|| OsString::from("true"))
        });
        let set = LeverSet::resolve_layered(&registry(), &raw, |decl| {
            (decl.name == GATE.name).then_some(LeverValue::Bool(true))
        })
        .expect("fixture config is admissible");

        assert_eq!(set.get(&GATE), LeverValue::Bool(false));
        assert_eq!(set.source(&GATE), Source::LegacyEnvRejected);
        assert_eq!(
            set.resolved(&GATE).expect("gate registered").rejected_raw,
            Some("true".to_owned())
        );
    }

    #[test]
    fn captured_inputs_are_authoritative_across_later_resolution() {
        let mut live = Some(OsString::from("0.25"));
        let mut lookups = Vec::new();
        let captured = RawLeverInputs::capture_with(&registry(), |name| {
            lookups.push(name.to_owned());
            (name == FRAC.name).then(|| live.clone()).flatten()
        });
        assert_eq!(
            lookups,
            vec![
                FRAC.name.to_owned(),
                GATE.name.to_owned(),
                PRESENCE.name.to_owned(),
            ],
            "capture calls the lookup exactly once per registered name, in registry order"
        );
        live = Some(OsString::from("0.50"));

        let set = LeverSet::resolve_layered(&registry(), &captured, |_| None)
            .expect("default-only resolution is infallible");
        assert_eq!(set.get(&FRAC), LeverValue::F64(0.25));
        assert_eq!(set.source(&FRAC), Source::LegacyEnv);
        assert_eq!(live, Some(OsString::from("0.50")));
    }

    #[test]
    fn incompatible_typed_config_returns_an_evidence_error() {
        let raw = RawLeverInputs::capture_with(&registry(), |_| None);
        let error = LeverSet::resolve_layered(&registry(), &raw, |decl| {
            (decl.name == FRAC.name).then_some(LeverValue::F64(0.90))
        })
        .expect_err("out-of-range typed config must be rejected");
        assert_eq!(error.lever(), FRAC.name);
        assert_eq!(
            error.to_string(),
            "typed config supplied an inadmissible value for NY_LEVERS_SET_SELFTEST_FRAC"
        );
    }

    #[cfg(unix)]
    #[test]
    fn non_utf8_numeric_is_rejected_but_non_utf8_text_presence_is_accepted() {
        use std::os::unix::ffi::OsStringExt;

        let non_utf8 = OsString::from_vec(vec![b'x', 0xff]);
        let raw = RawLeverInputs::capture_with(&registry(), |name| {
            matches!(
                name,
                "NY_LEVERS_SET_SELFTEST_FRAC" | "NY_LEVERS_SET_SELFTEST_PRESENCE"
            )
            .then(|| non_utf8.clone())
        });
        assert_eq!(raw.len(), 3);
        assert!(!raw.is_empty());
        assert!(raw
            .get(&PRESENCE)
            .expect("presence was captured")
            .to_str()
            .is_none());

        let set = LeverSet::resolve_layered(&registry(), &raw, |decl| {
            (decl.name == FRAC.name).then_some(LeverValue::F64(0.25))
        })
        .expect("fixture config is admissible");

        let frac = set.resolved(&FRAC).expect("frac registered");
        assert_eq!(frac.value, LeverValue::Unset, "invalid env kills config");
        assert_eq!(frac.source, Source::LegacyEnvRejected);
        assert_eq!(frac.env_utf8, Some(false));
        assert!(frac.rejected_raw.is_some());

        let presence = set.resolved(&PRESENCE).expect("presence registered");
        assert_eq!(presence.source, Source::LegacyEnv);
        assert_eq!(presence.env_utf8, Some(false));
        assert!(matches!(presence.value, LeverValue::Text(_)));

        let receipt = set.receipt();
        assert_eq!(receipt["env_present"], 2);
        assert_eq!(receipt["env_accepted"], 1);
        assert_eq!(receipt["env_rejected"], 1);
        assert_eq!(
            receipt["env_present"].as_u64(),
            receipt["env_accepted"]
                .as_u64()
                .zip(receipt["env_rejected"].as_u64())
                .map(|(accepted, rejected)| accepted + rejected)
        );
    }

    #[test]
    #[should_panic(expected = "is not present in this LeverSet's registry")]
    fn an_unregistered_declaration_cannot_fall_back_silently() {
        static OUTSIDER: LeverDecl = LeverDecl {
            name: "NY_LEVERS_SET_SELFTEST_OUTSIDER",
            kind: LeverKind::Bool,
            default: DefaultSpec::Bool(false),
            bucket: Bucket::Debug,
            moat: MoatRisk::None,
            doc: "not filed in the fixture registry",
            provenance: Provenance::Unmeasured { why_ok: "test" },
            owner: Scope {
                package: "ny-levers",
                subsystem: "selftest",
            },
            readers: &[],
        };

        let set = LeverSet::freeze_with(&registry(), |_| None);
        let _ = set.get(&OUTSIDER);
    }

    #[test]
    #[should_panic(expected = "is not present in this LeverSet's registry")]
    fn a_same_name_impostor_cannot_borrow_the_registered_value() {
        static IMPOSTOR: LeverDecl = LeverDecl {
            name: "NY_LEVERS_SET_SELFTEST_GATE",
            kind: LeverKind::Text,
            default: DefaultSpec::Text("different semantics"),
            bucket: Bucket::Debug,
            moat: MoatRisk::High,
            doc: "same spelling as GATE but deliberately incompatible metadata",
            provenance: Provenance::Unmeasured { why_ok: "test" },
            owner: Scope {
                package: "ny-levers",
                subsystem: "impostor",
            },
            readers: &[],
        };

        let set = LeverSet::freeze_with(&registry(), |_| Some("1".to_owned()));
        let _ = set.get(&IMPOSTOR);
    }
}
