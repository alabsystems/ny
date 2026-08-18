// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The ONE process-environment chokepoint.
//!
//! Every `NY_*` value in the workspace is eventually meant to arrive through
//! [`read`] (or, once frozen, through [`crate::LeverSet`]). Nothing else in
//! this crate touches `std::env`.
//!
//! Note for the ratchet test: the chokepoint reads `std::env::var_os(decl.name)`
//! — a *dynamic* name — so it is invisible to the ratchet's literal scan by
//! construction. That is intended: the ratchet counts ad-hoc reads, and the
//! chokepoint is the thing they are supposed to become.

use std::ffi::{OsStr, OsString};

use crate::decl::{DefaultSpec, LeverDecl, LeverKind};
use crate::registry::LeverRegistry;

/// A typed contextual value did not satisfy its declaration.
///
/// This is an integration/configuration error, not a reason to panic on a
/// scored path. Callers that project presets into a receipt can record the
/// failure while allowing the verifier's existing validation path to retain
/// ownership of the verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolveError {
    lever: &'static str,
}

impl ResolveError {
    /// The declaration whose contextual value was inadmissible.
    pub const fn lever(&self) -> &'static str {
        self.lever
    }
}

impl std::fmt::Display for ResolveError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "typed config supplied an inadmissible value for {}",
            self.lever
        )
    }
}

impl std::error::Error for ResolveError {}

/// Where a resolved value came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// The declaration's shipped default.
    Default,
    /// A typed preset or other run-context configuration supplied the value.
    Config,
    /// The legacy process environment supplied an admissible value.
    LegacyEnv,
    /// A present legacy process-environment value was rejected.
    ///
    /// The resolved value is the declaration default, not a lower contextual
    /// value. A malformed explicit override is a kill switch for an
    /// otherwise-selected preset, not permission to reveal that preset.
    LegacyEnvRejected,
}

impl Source {
    /// Stable lowercase name for receipts.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Config => "config",
            Self::LegacyEnv => "legacy_env",
            Self::LegacyEnvRejected => "legacy_env_rejected",
        }
    }
}

/// A parsed lever value.
#[derive(Debug, Clone, PartialEq)]
pub enum LeverValue {
    /// Boolean gate.
    Bool(bool),
    /// Unsigned integer.
    U64(u64),
    /// Floating-point scalar.
    F64(f64),
    /// Seconds.
    Secs(f64),
    /// Text (enum member, path, list).
    Text(String),
    /// No value: the lever has no default and the environment did not supply
    /// an admissible one. Semantically "off / absent".
    Unset,
}

impl LeverValue {
    /// Boolean view: `Bool(b)` is `b`, everything else is `false`.
    pub const fn as_bool(&self) -> bool {
        matches!(self, Self::Bool(true))
    }

    /// Numeric view for `F64Open` / `Secs` levers.
    pub const fn as_f64(&self) -> Option<f64> {
        match self {
            Self::F64(v) | Self::Secs(v) => Some(*v),
            _ => None,
        }
    }

    /// Integer view for `U64` levers.
    pub const fn as_u64(&self) -> Option<u64> {
        match self {
            Self::U64(v) => Some(*v),
            _ => None,
        }
    }

    /// Text view for `Enum` / `Text` levers.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Text(v) => Some(v.as_str()),
            _ => None,
        }
    }

    /// JSON projection for the receipt.
    pub fn to_json(&self) -> serde_json::Value {
        match self {
            Self::Bool(v) => serde_json::Value::Bool(*v),
            Self::U64(v) => serde_json::Value::from(*v),
            Self::F64(v) | Self::Secs(v) => serde_json::Number::from_f64(*v)
                .map_or(serde_json::Value::String(v.to_string()), |n| {
                    serde_json::Value::Number(n)
                }),
            Self::Text(v) => serde_json::Value::String(v.clone()),
            Self::Unset => serde_json::Value::Null,
        }
    }
}

/// A lever resolved against one environment.
#[derive(Debug, Clone)]
pub struct Resolved {
    /// The declaration this value belongs to.
    pub decl: &'static LeverDecl,
    /// The parsed value.
    pub value: LeverValue,
    /// Which configuration layer determined the effective value.
    pub source: Source,
    /// The raw string, when the environment supplied one that was REJECTED
    /// (malformed, out of range, or not the exact arming token). Recording it
    /// is the point: a typo'd `NY_FOO=ture` silently doing nothing is exactly
    /// the contamination class the receipt exists to expose.
    pub rejected_raw: Option<String>,
    /// Whether a present legacy environment value was valid UTF-8.
    ///
    /// `None` means that the environment variable was absent. `Some(false)`
    /// keeps a present non-Unicode value distinguishable from absence even
    /// though JSON can only project its lossy text representation.
    pub env_utf8: Option<bool>,
}

/// One frozen snapshot of all registered legacy environment inputs.
///
/// Entries are total over the registry, including explicit `None` entries for
/// absent names. Values remain [`OsString`]s until resolution, so a present
/// non-UTF-8 value cannot collapse into absence. Capturing and resolving are
/// deliberately separate: a run may capture its environment at entry, load a
/// typed preset later, and still resolve both layers against the authoritative
/// entry-time environment.
#[derive(Debug, Clone)]
pub struct RawLeverInputs {
    entries: Vec<RawLeverInput>,
}

#[derive(Debug, Clone)]
struct RawLeverInput {
    decl: &'static LeverDecl,
    value: Option<OsString>,
}

impl RawLeverInputs {
    /// Capture every registered name from the process environment exactly
    /// once, preserving non-UTF-8 values.
    pub fn capture(registry: &LeverRegistry) -> Self {
        Self::capture_with(registry, |name| std::env::var_os(name))
    }

    /// Capture every registered name through a deterministic lookup.
    ///
    /// `lookup` is called exactly once for every declaration, in the
    /// registry's stable name order. This is the test seam and also makes the
    /// temporal authority of the snapshot explicit.
    pub fn capture_with<F>(registry: &LeverRegistry, mut lookup: F) -> Self
    where
        F: FnMut(&str) -> Option<OsString>,
    {
        let entries = registry
            .all()
            .iter()
            .map(|decl| RawLeverInput {
                decl,
                value: lookup(decl.name),
            })
            .collect();
        Self { entries }
    }

    /// Number of declarations represented by this total snapshot.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether this snapshot was captured from an empty registry.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The exact captured value for one declaration.
    ///
    /// `None` means the registered variable was absent. A declaration not in
    /// the registry used for capture is a programming error and panics rather
    /// than masquerading as an absent environment variable.
    ///
    /// # Panics
    ///
    /// Panics when `decl` was not part of the registry used for capture.
    pub fn get(&self, decl: &'static LeverDecl) -> Option<&OsStr> {
        self.entry(decl).value.as_deref()
    }

    fn entry(&self, decl: &'static LeverDecl) -> &RawLeverInput {
        self.entries
            .iter()
            .find(|entry| std::ptr::eq(entry.decl, decl))
            .unwrap_or_else(|| {
                panic!(
                    "{} is not present in this RawLeverInputs snapshot",
                    decl.name
                )
            })
    }
}

/// Read one lever from the process environment.
///
/// This is the chokepoint. Prefer freezing a whole [`crate::LeverSet`] once at
/// run entry; this exists for the migration period and for the set's own
/// internals.
pub fn read(decl: &'static LeverDecl) -> Resolved {
    resolve_raw(decl, std::env::var_os(decl.name).as_deref(), None)
        .expect("default-only lever resolution cannot reject typed config")
}

/// Raw, unparsed view of one lever's environment string.
///
/// This exists for exactly two caller shapes, both of which derive the
/// DECISION themselves instead of taking the parsed value:
///
/// * a HOT-PATH reader (per-iteration, per-allocation) that cannot afford a
///   process-env lookup per call and therefore latches THIS string once,
///   deriving the decision from it on every call — the
///   `alpha_zero_yield_env_raw` idiom. This preserves the legacy parser during
///   migration preparation, but the raw latch is still process-wide. Phase 2
///   replaces both raw-string and decision latches with an injected per-run
///   [`crate::LeverSet`];
/// * a reader whose unit-tested pure predicate (`gate_on(raw)`) IS the spec of
///   its arming rule and must stay in the production path.
///
/// Everything else should use [`read`] or capture a [`RawLeverInputs`], which
/// parse, range-check and record rejections for the receipt. This compatibility
/// function remains while its existing hot-path consumers migrate to an
/// injected [`crate::LeverSet`]; new code must not use it as a second raw-env
/// architecture.
pub fn read_raw(decl: &'static LeverDecl) -> Option<String> {
    std::env::var(decl.name).ok()
}

/// Whether a [`LeverKind::Presence`] lever is SET, whatever its value.
///
/// The lookup is `var_os`, deliberately: a non-UTF-8 value is still present, and
/// `env::var` would have reported it absent — silently disarming a gate the
/// operator set. That difference is the whole reason this is a separate
/// chokepoint rather than `read_raw(..).is_some()`.
///
/// # Panics
///
/// Panics when `decl` is not a [`LeverKind::Presence`] declaration. Reading an
/// exact-`"1"` gate as presence would arm it on `NY_X=0`, which is the specific
/// confusion this kind exists to make impossible; a wrong reader here is a
/// programming error, not a runtime condition to absorb.
pub fn read_presence(decl: &'static LeverDecl) -> bool {
    assert!(
        matches!(decl.kind, LeverKind::Presence),
        "{} is not a presence lever; use `read` for exact-\"1\" gates",
        decl.name
    );
    std::env::var_os(decl.name).is_some()
}

/// Read one lever from an arbitrary lookup.
///
/// This compatibility seam accepts `String`, so it cannot represent a
/// non-UTF-8 value. New snapshot code should use
/// [`RawLeverInputs::capture_with`].
pub fn read_with<F>(decl: &'static LeverDecl, lookup: F) -> Resolved
where
    F: FnOnce(&str) -> Option<String>,
{
    let raw = lookup(decl.name).map(OsString::from);
    resolve_raw(decl, raw.as_deref(), None)
        .expect("default-only lever resolution cannot reject typed config")
}

/// Read ONE lever from the process environment over a typed config value.
///
/// [`LeverSet::resolve_layered`](crate::LeverSet::resolve_layered) is the
/// registry-wide form of this; it needs a config answer for every declaration
/// and a captured snapshot. This is the single-declaration form, for a reader
/// that has a typed preset value for ITS lever and nothing to say about the
/// other ~850. The layering is the same code path
/// ([`resolve_raw`]), so the two cannot drift:
///
/// * an admissible environment value wins, in BOTH directions — `"1"` arms and
///   `"0"` disarms whatever the preset asked for;
/// * a PRESENT but inadmissible value (`"true"`, `" 1"`, `""`) suppresses the
///   config layer and resolves to the DECLARATION DEFAULT, recorded as
///   [`Source::LegacyEnvRejected`]. A typo is a kill switch, never a silent
///   promotion of the preset;
/// * absence falls through to `config`, then to the declaration default.
///
/// # Errors
///
/// Returns [`ResolveError`] when `config` is incompatible with the
/// declaration's kind or range, so a configuration mistake is reported rather
/// than panicking a scored run.
pub fn read_over_config(
    decl: &'static LeverDecl,
    config: Option<LeverValue>,
) -> Result<Resolved, ResolveError> {
    resolve_raw(decl, std::env::var_os(decl.name).as_deref(), config)
}

/// [`read_over_config`] against an arbitrary lookup instead of the process
/// environment.
///
/// The test seam: a unit test can exercise the REAL layering rule on the REAL
/// declaration without mutating the process. Accepts `String`, so it cannot
/// represent a non-UTF-8 value; use [`RawLeverInputs::capture_with`] when that
/// distinction matters.
///
/// # Errors
///
/// Same as [`read_over_config`].
pub fn read_over_config_with<F>(
    decl: &'static LeverDecl,
    lookup: F,
    config: Option<LeverValue>,
) -> Result<Resolved, ResolveError>
where
    F: FnOnce(&str) -> Option<String>,
{
    let raw = lookup(decl.name).map(OsString::from);
    resolve_raw(decl, raw.as_deref(), config)
}

/// Resolve one captured raw value over an optional typed configuration value.
///
/// Layer precedence is legacy environment, then contextual config/preset,
/// then the declaration default. A PRESENT but inadmissible environment value
/// suppresses `config` and resolves to the declaration default with
/// [`Source::LegacyEnvRejected`].
pub(crate) fn resolve_raw(
    decl: &'static LeverDecl,
    raw: Option<&OsStr>,
    config: Option<LeverValue>,
) -> Result<Resolved, ResolveError> {
    let (fallback_value, fallback_source) = match config {
        Some(value) => {
            if !is_admissible_config(decl, &value) {
                return Err(ResolveError { lever: decl.name });
            }
            (value, Source::Config)
        }
        None => (default_value(decl), Source::Default),
    };

    let Some(raw) = raw else {
        return Ok(Resolved {
            decl,
            value: fallback_value,
            source: fallback_source,
            rejected_raw: None,
            env_utf8: None,
        });
    };

    if let Some(raw_utf8) = raw.to_str() {
        return Ok(match parse(decl.kind, raw_utf8) {
            Some(value) => Resolved {
                decl,
                value,
                source: Source::LegacyEnv,
                rejected_raw: None,
                env_utf8: Some(true),
            },
            None => rejected(decl, raw, true),
        });
    }

    // `Text + Unset` is the declared shape for a presence gate. Preserve the
    // exact bytes in RawLeverInputs and accept a lossy runtime projection so
    // ANY present value remains armed. Other kinds cannot parse non-Unicode
    // input and therefore take the explicit-rejection path.
    if matches!(decl.kind, LeverKind::Text) && matches!(decl.default, DefaultSpec::Unset) {
        Ok(Resolved {
            decl,
            value: LeverValue::Text(raw.to_string_lossy().into_owned()),
            source: Source::LegacyEnv,
            rejected_raw: None,
            env_utf8: Some(false),
        })
    } else {
        Ok(rejected(decl, raw, false))
    }
}

fn rejected(decl: &'static LeverDecl, raw: &OsStr, utf8: bool) -> Resolved {
    Resolved {
        decl,
        // An invalid explicit override suppresses a contextual config value.
        // Falling all the way back to the declaration default is the legacy
        // kill-switch contract used by the typed root-alpha preset seam.
        value: default_value(decl),
        source: Source::LegacyEnvRejected,
        rejected_raw: Some(raw.to_string_lossy().into_owned()),
        env_utf8: Some(utf8),
    }
}

fn is_admissible_config(decl: &'static LeverDecl, value: &LeverValue) -> bool {
    match (decl.kind, value) {
        (LeverKind::Bool, LeverValue::Bool(_))
        | (LeverKind::U64 | LeverKind::U64Trimmed, LeverValue::U64(_))
        | (LeverKind::Text, LeverValue::Text(_)) => true,
        (LeverKind::Presence, LeverValue::Bool(_)) => true,
        (LeverKind::UsizeTrimmed, LeverValue::U64(value)) => usize::try_from(*value).is_ok(),
        (LeverKind::F64Open { min, max }, LeverValue::F64(value)) => {
            value.is_finite() && *value > min && *value < max
        }
        (LeverKind::F64ClosedTrimmed { min, max }, LeverValue::F64(value)) => {
            value.is_finite() && *value >= min && *value <= max
        }
        (LeverKind::Secs, LeverValue::Secs(value)) => value.is_finite() && *value >= 0.0,
        (LeverKind::Enum(members), LeverValue::Text(value)) => {
            members.iter().any(|member| *member == value)
        }
        _ => false,
    }
}

/// The declaration's default as a value.
pub(crate) fn default_value(decl: &'static LeverDecl) -> LeverValue {
    match decl.default {
        DefaultSpec::Bool(v) => LeverValue::Bool(v),
        DefaultSpec::U64(v) => LeverValue::U64(v),
        DefaultSpec::F64(v) => LeverValue::F64(v),
        DefaultSpec::Secs(v) => LeverValue::Secs(v),
        DefaultSpec::Text(v) => LeverValue::Text(v.to_owned()),
        DefaultSpec::Unset => LeverValue::Unset,
    }
}

/// Parse a raw environment string per the lever's kind.
///
/// `None` means "the environment did not supply an admissible value"; the
/// caller falls back to the default. That fallback (rather than an error) is
/// the legacy contract: every in-tree reader of the form
/// `var(..).ok().and_then(parse).filter(..)` behaves exactly this way, and a
/// migration that started erroring would change behaviour on malformed input.
///
/// # Booleans: the exact `"1"` rule
///
/// A boolean lever is armed by the exact string `"1"` and by nothing else —
/// not `"true"`, not `"yes"`, not `"TRUE"`, not `"01"`, not `" 1"`. This is
/// the repo's dominant idiom (`var("NY_X").ok().as_deref() == Some("1")`) and
/// the reason to keep it is not taste, it is the HALF-ENABLE hazard: the
/// surface has ~850 ad-hoc reads whose parsers disagree, several of them
/// reading the SAME name from different crates. If the chokepoint accepted
/// `"true"` while an unmigrated site still compared against `"1"`, then
/// `NY_EFT_ERR=true` would arm one half of a deliberately-twinned channel and
/// leave the other half dark — a configuration nobody designed, on a path that
/// carries a verdict. One token, one meaning, everywhere; a rejected token is
/// recorded in the receipt rather than being generously reinterpreted.
///
/// The other live boolean idiom — "on unless exactly `0`" — is expressible
/// without a second parser: it is [`LeverKind::Bool`] with
/// `DefaultSpec::Bool(true)`. Absent takes the `true` default, `"0"` disarms,
/// `"1"` arms, and anything else is a recorded rejection that leaves the
/// default in place.
fn parse(kind: LeverKind, raw: &str) -> Option<LeverValue> {
    match kind {
        LeverKind::Bool => {
            if raw == "1" {
                Some(LeverValue::Bool(true))
            } else if raw == "0" {
                // An explicit "0" is a real, admissible disarm — distinct from
                // a rejected token, and it must not be reported as such.
                Some(LeverValue::Bool(false))
            } else {
                None
            }
        }
        // Presence cannot reject: the variable is set, so the gate is armed
        // regardless of the bytes. `resolve_raw` only calls this when a value is
        // present, so `true` is the only answer it can give.
        LeverKind::Presence => Some(LeverValue::Bool(true)),
        LeverKind::U64 => raw.parse::<u64>().ok().map(LeverValue::U64),
        LeverKind::U64Trimmed => raw.trim().parse::<u64>().ok().map(LeverValue::U64),
        LeverKind::UsizeTrimmed => raw
            .trim()
            .parse::<usize>()
            .ok()
            .map(|value| LeverValue::U64(value as u64)),
        LeverKind::F64Open { min, max } => raw
            .parse::<f64>()
            .ok()
            .filter(|v| v.is_finite() && *v > min && *v < max)
            .map(LeverValue::F64),
        LeverKind::F64ClosedTrimmed { min, max } => raw
            .trim()
            .parse::<f64>()
            .ok()
            .filter(|v| v.is_finite() && *v >= min && *v <= max)
            .map(LeverValue::F64),
        LeverKind::Secs => raw
            .parse::<f64>()
            .ok()
            .filter(|v| v.is_finite() && *v >= 0.0)
            .map(LeverValue::Secs),
        LeverKind::Enum(members) => members
            .iter()
            .find(|m| **m == raw)
            .map(|m| LeverValue::Text((*m).to_owned())),
        LeverKind::Text => Some(LeverValue::Text(raw.to_owned())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decl::{Bucket, MoatRisk, Provenance, Scope};

    static GATE: LeverDecl = LeverDecl {
        name: "NY_LEVERS_SELFTEST_GATE",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::None,
        doc: "unit-test fixture",
        provenance: Provenance::Unmeasured { why_ok: "test" },
        owner: Scope {
            package: "ny-levers",
            subsystem: "selftest",
        },
        readers: &[],
    };

    static FRAC: LeverDecl = LeverDecl {
        name: "NY_LEVERS_SELFTEST_FRAC",
        kind: LeverKind::F64Open { min: 0.0, max: 0.9 },
        default: DefaultSpec::Unset,
        bucket: Bucket::Debug,
        moat: MoatRisk::None,
        doc: "unit-test fixture",
        provenance: Provenance::Unmeasured { why_ok: "test" },
        owner: Scope {
            package: "ny-levers",
            subsystem: "selftest",
        },
        readers: &[],
    };

    fn with(raw: Option<&str>, decl: &'static LeverDecl) -> Resolved {
        let owned = raw.map(str::to_owned);
        read_with(decl, move |_| owned)
    }

    #[test]
    fn exact_one_arms_and_nothing_else_does() {
        assert!(with(Some("1"), &GATE).value.as_bool());
        for reject in ["true", "TRUE", "yes", "on", "01", " 1", "1 ", "", "2", "-1"] {
            let r = with(Some(reject), &GATE);
            assert!(
                !r.value.as_bool(),
                "{reject:?} must not arm an exact-\"1\" gate"
            );
            assert_eq!(r.source, Source::LegacyEnvRejected, "{reject:?}");
            assert_eq!(
                r.rejected_raw.as_deref(),
                Some(reject),
                "{reject:?} must be recorded as rejected"
            );
        }
    }

    #[test]
    fn explicit_zero_is_an_admissible_disarm_not_a_rejection() {
        let r = with(Some("0"), &GATE);
        assert!(!r.value.as_bool());
        assert_eq!(r.source, Source::LegacyEnv);
        assert_eq!(r.rejected_raw, None);
    }

    #[test]
    fn absent_takes_the_default() {
        let r = with(None, &GATE);
        assert_eq!(r.value, LeverValue::Bool(false));
        assert_eq!(r.source, Source::Default);
        assert_eq!(r.rejected_raw, None);
    }

    /// The single-declaration config seam layers exactly like
    /// [`LeverSet::resolve_layered`](crate::LeverSet::resolve_layered): env
    /// over config over default, with a REJECTED env token suppressing config
    /// rather than falling through to it.
    #[test]
    fn one_lever_layers_env_over_config_over_default() {
        let armed = || Some(LeverValue::Bool(true));
        let layered = |raw: Option<&str>, config: Option<LeverValue>| {
            let owned = raw.map(str::to_owned);
            read_over_config_with(&GATE, move |_| owned, config).expect("Bool config is admissible")
        };

        // Config alone arms a lever whose declaration default is off.
        let from_config = layered(None, armed());
        assert!(from_config.value.as_bool());
        assert_eq!(from_config.source, Source::Config);

        // An admissible env value wins in BOTH directions.
        assert!(layered(Some("1"), None).value.as_bool());
        let killed = layered(Some("0"), armed());
        assert!(
            !killed.value.as_bool(),
            "an explicit 0 must disarm a preset"
        );
        assert_eq!(killed.source, Source::LegacyEnv);

        // A near-miss token suppresses config and lands on the DEFAULT.
        for reject in ["true", "TRUE", "yes", "on", "01", " 1", "1 ", "", "2"] {
            let r = layered(Some(reject), armed());
            assert!(
                !r.value.as_bool(),
                "{reject:?} must not be reinterpreted as arming a config-armed lever"
            );
            assert_eq!(r.source, Source::LegacyEnvRejected, "{reject:?}");
        }

        // Config of the wrong KIND is an error, not a silent default.
        assert!(read_over_config_with(&GATE, |_| None, Some(LeverValue::U64(1))).is_err());
    }

    #[test]
    fn open_interval_rejects_its_endpoints_and_nonfinite() {
        assert_eq!(with(Some("0.25"), &FRAC).value, LeverValue::F64(0.25));
        for reject in ["0", "0.0", "0.9", "1.0", "-0.1", "nan", "inf", "abc"] {
            let r = with(Some(reject), &FRAC);
            assert_eq!(r.value, LeverValue::Unset, "{reject:?}");
            assert_eq!(r.source, Source::LegacyEnvRejected, "{reject:?}");
        }
    }
}
