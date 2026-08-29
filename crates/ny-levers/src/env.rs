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
    /// The value that would become effective after removing the current
    /// effective authority layer.
    ///
    /// For an environment-effective value this is contextual config (when
    /// supplied) or the declaration default. For a config-effective value it
    /// is the declaration default. A default-effective value is its own stable
    /// lower answer. This metadata is kept out of receipt JSON until a schema
    /// version explicitly publishes it.
    pub lower_value: LeverValue,
    /// The authority that supplied [`Self::lower_value`].
    pub lower_source: Source,
    /// The raw string, when the environment supplied one that was REJECTED
    /// (malformed, out of range, or not the exact arming token). Recording it
    /// is the point: a typo'd `NY_FOO=ture` silently doing nothing is exactly
    /// the contamination class the receipt exists to expose.
    pub rejected_raw: Option<String>,
    /// A present environment value intentionally ignored by an override-style
    /// parser while the contextual config/default remained authoritative.
    ///
    /// This is distinct from [`Self::rejected_raw`]: rejection suppresses the
    /// config layer, whereas ignored input explicitly falls through to it.
    pub ignored_raw: Option<String>,
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
    read_raw_os(decl).and_then(|raw| raw.into_string().ok())
}

/// Raw, unparsed view that preserves both presence and non-UTF-8 bytes.
///
/// This is the narrow companion to [`read_raw`] for a legacy predicate whose
/// historical UTF-8 input was `Option<&str>` but whose typed-config precedence
/// still needs to distinguish ABSENT from PRESENT/non-UTF-8. New consumers
/// should freeze a [`RawLeverInputs`] / [`crate::LeverSet`] at run entry.
pub fn read_raw_os(decl: &'static LeverDecl) -> Option<OsString> {
    std::env::var_os(decl.name)
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
/// * an admissible environment value owns the effective layer according to
///   its declaration: ordinary Boolean overrides accept both `"1"` and `"0"`,
///   while one-way rollback kinds such as [`LeverKind::BoolKillOverride`]
///   accept only their explicit kill token;
/// * for ordinary kinds, a PRESENT but inadmissible value (`"true"`, `" 1"`,
///   `""`) suppresses the config layer and resolves to the DECLARATION DEFAULT,
///   recorded as [`Source::LegacyEnvRejected`]. A typo is a kill switch, never
///   a silent promotion of the preset;
/// * override-style kinds ([`LeverKind::BoolOverride`],
///   [`LeverKind::BoolKillOverride`],
///   [`LeverKind::U64Override`], [`LeverKind::UsizeTrimmedOverride`],
///   [`LeverKind::UsizeOverride`] and [`LeverKind::F64OpenOverride`])
///   deliberately differ: an unrecognized token is recorded in
///   [`Resolved::ignored_raw`] and leaves config/default authoritative,
///   preserving selectors whose historical rule falls through when no valid
///   environment override was supplied;
/// * Unicode-owned kinds ([`LeverKind::F64OpenUnicodeOwned`] and
///   [`LeverKind::UsizeTrimmedUnicodeOwned`]) let invalid Unicode suppress
///   config but ignore non-Unicode bytes that their legacy `std::env::var`
///   readers could not observe;
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
/// ordinarily suppresses `config` and resolves to the declaration default with
/// [`Source::LegacyEnvRejected`]; override-style kinds instead record it as
/// ignored and retain the lower layer.
pub(crate) fn resolve_raw(
    decl: &'static LeverDecl,
    raw: Option<&OsStr>,
    config: Option<LeverValue>,
) -> Result<Resolved, ResolveError> {
    let declaration_default = default_value(decl);
    let (fallback_value, fallback_source) = match config {
        Some(value) => {
            if !is_admissible_config(decl, &value) {
                return Err(ResolveError { lever: decl.name });
            }
            (value, Source::Config)
        }
        None => (declaration_default.clone(), Source::Default),
    };
    // Last use of `declaration_default`, so move it. The `None` arm above still
    // clones because it borrows before this point.
    let fallback_lower_value = declaration_default;
    let fallback_lower_source = Source::Default;

    let Some(raw) = raw else {
        return Ok(Resolved {
            decl,
            value: fallback_value,
            source: fallback_source,
            lower_value: fallback_lower_value,
            lower_source: fallback_lower_source,
            rejected_raw: None,
            ignored_raw: None,
            env_utf8: None,
        });
    };

    if let Some(raw_utf8) = raw.to_str() {
        return Ok(match parse(decl.kind, raw_utf8) {
            Some(value) => Resolved {
                decl,
                value,
                source: Source::LegacyEnv,
                lower_value: fallback_value,
                lower_source: fallback_source,
                rejected_raw: None,
                ignored_raw: None,
                env_utf8: Some(true),
            },
            None if ignores_invalid_unicode(decl.kind) => Resolved {
                decl,
                value: fallback_value,
                source: fallback_source,
                lower_value: fallback_lower_value,
                lower_source: fallback_lower_source,
                rejected_raw: None,
                ignored_raw: Some(raw.to_string_lossy().into_owned()),
                env_utf8: Some(true),
            },
            None => rejected(decl, raw, true),
        });
    }

    // `Text + Unset` is the declared shape for a presence gate. Preserve the
    // exact bytes in RawLeverInputs and accept a lossy runtime projection so
    // ANY present value remains armed. Override-style parsers ignore other
    // non-Unicode inputs and preserve their lower layer; remaining kinds take
    // the explicit-rejection path.
    if matches!(decl.kind, LeverKind::Text) && matches!(decl.default, DefaultSpec::Unset) {
        Ok(Resolved {
            decl,
            value: LeverValue::Text(raw.to_string_lossy().into_owned()),
            source: Source::LegacyEnv,
            lower_value: fallback_value,
            lower_source: fallback_source,
            rejected_raw: None,
            ignored_raw: None,
            env_utf8: Some(false),
        })
    } else if ignores_non_unicode(decl.kind) {
        Ok(Resolved {
            decl,
            value: fallback_value,
            source: fallback_source,
            lower_value: fallback_lower_value,
            lower_source: fallback_lower_source,
            rejected_raw: None,
            ignored_raw: Some(raw.to_string_lossy().into_owned()),
            env_utf8: Some(false),
        })
    } else {
        Ok(rejected(decl, raw, false))
    }
}

fn rejected(decl: &'static LeverDecl, raw: &OsStr, utf8: bool) -> Resolved {
    let default = default_value(decl);
    Resolved {
        decl,
        // An invalid explicit override suppresses a contextual config value.
        // Falling all the way back to the declaration default is the legacy
        // kill-switch contract used by the typed root-alpha preset seam.
        value: default.clone(),
        source: Source::LegacyEnvRejected,
        lower_value: default,
        lower_source: Source::Default,
        rejected_raw: Some(raw.to_string_lossy().into_owned()),
        ignored_raw: None,
        env_utf8: Some(utf8),
    }
}

fn ignores_invalid_unicode(kind: LeverKind) -> bool {
    matches!(
        kind,
        LeverKind::BoolOverride
            | LeverKind::BoolKillOverride
            | LeverKind::U64Override
            | LeverKind::UsizeTrimmedOverride { .. }
            | LeverKind::UsizeOverride { .. }
            | LeverKind::F64OpenOverride { .. }
    )
}

fn ignores_non_unicode(kind: LeverKind) -> bool {
    ignores_invalid_unicode(kind)
        || matches!(
            kind,
            LeverKind::TruthyTrimmed
                | LeverKind::TrimmedExactOne
                | LeverKind::UsizeTrimmedUnicodeOwned
                | LeverKind::F64OpenUnicodeOwned { .. }
        )
}

fn is_admissible_config(decl: &'static LeverDecl, value: &LeverValue) -> bool {
    match (decl.kind, value) {
        (
            LeverKind::Bool
            | LeverKind::BoolOverride
            | LeverKind::BoolKillOverride
            | LeverKind::TruthyTrimmed
            | LeverKind::TrimmedExactOne,
            LeverValue::Bool(_),
        )
        | (LeverKind::U64 | LeverKind::U64Override | LeverKind::U64Trimmed, LeverValue::U64(_))
        | (LeverKind::Text, LeverValue::Text(_)) => true,
        (LeverKind::Presence, LeverValue::Bool(_)) => true,
        (LeverKind::UsizeTrimmed | LeverKind::UsizeTrimmedUnicodeOwned, LeverValue::U64(value)) => {
            usize::try_from(*value).is_ok()
        }
        (
            LeverKind::UsizeTrimmedOverride { .. } | LeverKind::UsizeOverride { .. },
            LeverValue::U64(value),
        ) => usize::try_from(*value).is_ok(),
        (LeverKind::Usize { min }, LeverValue::U64(value)) => {
            usize::try_from(*value).is_ok_and(|value| value >= min)
        }
        (LeverKind::F32Positive, LeverValue::F64(value)) => {
            let narrowed = *value as f32;
            narrowed.is_finite() && narrowed > 0.0 && f64::from(narrowed) == *value
        }
        (LeverKind::F32NonNegative, LeverValue::F64(value)) => {
            let narrowed = *value as f32;
            narrowed.is_finite() && narrowed >= 0.0 && f64::from(narrowed) == *value
        }
        (
            LeverKind::F64Open { min, max }
            | LeverKind::F64OpenUnicodeOwned { min, max }
            | LeverKind::F64OpenOverride { min, max },
            LeverValue::F64(value),
        ) => value.is_finite() && *value > min && *value < max,
        (LeverKind::F64ClosedTrimmed { min, max }, LeverValue::F64(value)) => {
            value.is_finite() && *value >= min && *value <= max
        }
        (LeverKind::Secs, LeverValue::Secs(value)) => value.is_finite() && *value >= 0.0,
        (LeverKind::Enum(members), LeverValue::Text(value)) => {
            members.iter().any(|member| *member == value)
        }
        (LeverKind::LayeredEnum { config_members, .. }, LeverValue::Text(value)) => {
            config_members.iter().any(|member| *member == value)
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
/// resolution policy either records a rejection to the declaration default or,
/// for an override-style kind, records ignored input and preserves its lower
/// contextual/default layer. Falling back rather than erroring is the legacy
/// contract: every in-tree reader of the form
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
        LeverKind::Bool | LeverKind::BoolOverride => {
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
        LeverKind::BoolKillOverride => (raw == "0").then_some(LeverValue::Bool(false)),
        LeverKind::TruthyTrimmed => {
            Some(LeverValue::Bool(matches!(raw.trim(), "1" | "true" | "on")))
        }
        LeverKind::TrimmedExactOne => Some(LeverValue::Bool(raw.trim() == "1")),
        // Presence cannot reject: the variable is set, so the gate is armed
        // regardless of the bytes. `resolve_raw` only calls this when a value is
        // present, so `true` is the only answer it can give.
        LeverKind::Presence => Some(LeverValue::Bool(true)),
        LeverKind::U64 | LeverKind::U64Override => raw.parse::<u64>().ok().map(LeverValue::U64),
        LeverKind::U64Trimmed => raw.trim().parse::<u64>().ok().map(LeverValue::U64),
        LeverKind::UsizeTrimmed | LeverKind::UsizeTrimmedUnicodeOwned => raw
            .trim()
            .parse::<usize>()
            .ok()
            .map(|value| LeverValue::U64(value as u64)),
        LeverKind::UsizeTrimmedOverride { min } => raw
            .trim()
            .parse::<usize>()
            .ok()
            .filter(|value| *value >= min)
            .map(|value| LeverValue::U64(value as u64)),
        LeverKind::UsizeOverride { min } => raw
            .parse::<usize>()
            .ok()
            .filter(|value| *value >= min)
            .map(|value| LeverValue::U64(value as u64)),
        LeverKind::Usize { min } => raw
            .parse::<usize>()
            .ok()
            .filter(|value| *value >= min)
            .map(|value| LeverValue::U64(value as u64)),
        LeverKind::F32Positive => raw
            .parse::<f32>()
            .ok()
            .filter(|value| value.is_finite() && *value > 0.0)
            .map(|value| LeverValue::F64(f64::from(value))),
        LeverKind::F32NonNegative => raw
            .parse::<f32>()
            .ok()
            .filter(|value| value.is_finite() && *value >= 0.0)
            .map(|value| LeverValue::F64(f64::from(value))),
        LeverKind::F64Open { min, max }
        | LeverKind::F64OpenUnicodeOwned { min, max }
        | LeverKind::F64OpenOverride { min, max } => raw
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
            .trim()
            .parse::<f64>()
            .ok()
            .filter(|v| v.is_finite() && *v >= 0.0)
            .map(LeverValue::Secs),
        LeverKind::Enum(members) => members
            .iter()
            .find(|m| **m == raw)
            .map(|m| LeverValue::Text((*m).to_owned())),
        LeverKind::LayeredEnum { env_members, .. } => env_members
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

    static OVERRIDE_GATE: LeverDecl = LeverDecl {
        name: "NY_LEVERS_SELFTEST_OVERRIDE_GATE",
        kind: LeverKind::BoolOverride,
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

    static TRIMMED_UNICODE_OWNED_USIZE: LeverDecl = LeverDecl {
        name: "LEVER_SELFTEST_TRIMMED_UNICODE_OWNED_USIZE",
        kind: LeverKind::UsizeTrimmedUnicodeOwned,
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

    static UNTRIMMED_USIZE_OVERRIDE: LeverDecl = LeverDecl {
        name: "LEVER_SELFTEST_UNTRIMMED_USIZE_OVERRIDE",
        kind: LeverKind::UsizeOverride { min: 0 },
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

    static UNTRIMMED_POSITIVE_USIZE_OVERRIDE: LeverDecl = LeverDecl {
        name: "LEVER_SELFTEST_UNTRIMMED_POSITIVE_USIZE_OVERRIDE",
        kind: LeverKind::UsizeOverride { min: 1 },
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
    fn lower_authority_is_the_next_effective_layer() {
        let configured =
            read_over_config_with(&GATE, |_| None, Some(LeverValue::Bool(true))).unwrap();
        assert_eq!(configured.source, Source::Config);
        assert_eq!(configured.lower_value, LeverValue::Bool(false));
        assert_eq!(configured.lower_source, Source::Default);

        let environment = read_over_config_with(
            &GATE,
            |_| Some("0".to_owned()),
            Some(LeverValue::Bool(true)),
        )
        .unwrap();
        assert_eq!(environment.source, Source::LegacyEnv);
        assert_eq!(environment.lower_value, LeverValue::Bool(true));
        assert_eq!(environment.lower_source, Source::Config);

        let defaulted = read_over_config_with(&GATE, |_| None, None).unwrap();
        assert_eq!(defaulted.source, Source::Default);
        assert_eq!(defaulted.lower_value, LeverValue::Bool(false));
        assert_eq!(defaulted.lower_source, Source::Default);
    }

    #[test]
    fn override_bool_preserves_exact_overrides_and_ignores_near_misses() {
        let layered = |raw: Option<&str>, config: Option<bool>| {
            let owned = raw.map(str::to_owned);
            read_over_config_with(&OVERRIDE_GATE, move |_| owned, config.map(LeverValue::Bool))
                .expect("Bool config is admissible")
        };

        let forced = layered(Some("1"), Some(false));
        assert_eq!(forced.value, LeverValue::Bool(true));
        assert_eq!(forced.source, Source::LegacyEnv);

        let killed = layered(Some("0"), Some(true));
        assert_eq!(killed.value, LeverValue::Bool(false));
        assert_eq!(killed.source, Source::LegacyEnv);

        for ignored in ["", "00", "true", "off", "1 "] {
            let configured = layered(Some(ignored), Some(true));
            assert_eq!(configured.value, LeverValue::Bool(true), "{ignored:?}");
            assert_eq!(configured.source, Source::Config, "{ignored:?}");
            assert_eq!(configured.rejected_raw, None, "{ignored:?}");
            assert_eq!(
                configured.ignored_raw.as_deref(),
                Some(ignored),
                "{ignored:?}"
            );

            let defaulted = layered(Some(ignored), None);
            assert_eq!(defaulted.value, LeverValue::Bool(false), "{ignored:?}");
            assert_eq!(defaulted.source, Source::Default, "{ignored:?}");
            assert_eq!(defaulted.ignored_raw.as_deref(), Some(ignored));
        }
    }

    #[test]
    fn schedule_u64_override_is_untrimmed_and_invalid_unicode_falls_through() {
        use crate::decls::margin_row::MARGIN_ROW_RESERVE_SECS;

        let layered = |raw: Option<&str>, config: Option<u64>| {
            let owned = raw.map(str::to_owned);
            read_over_config_with(
                &MARGIN_ROW_RESERVE_SECS,
                move |_| owned,
                config.map(LeverValue::U64),
            )
            .expect("u64 schedule config is admissible")
        };

        let configured = layered(None, Some(82));
        assert_eq!(configured.value, LeverValue::U64(82));
        assert_eq!(configured.source, Source::Config);

        for (raw, expected) in [("0", 0), ("82", 82), ("+7", 7)] {
            let resolved = layered(Some(raw), Some(91));
            assert_eq!(resolved.value, LeverValue::U64(expected), "{raw:?}");
            assert_eq!(resolved.source, Source::LegacyEnv, "{raw:?}");
            assert_eq!(resolved.lower_value, LeverValue::U64(91), "{raw:?}");
            assert_eq!(resolved.lower_source, Source::Config, "{raw:?}");
        }

        for raw in [" 82", "82 ", "", "-1", "1.5", "18446744073709551616"] {
            let resolved = layered(Some(raw), Some(91));
            assert_eq!(resolved.value, LeverValue::U64(91), "{raw:?}");
            assert_eq!(resolved.source, Source::Config, "{raw:?}");
            assert_eq!(resolved.rejected_raw, None, "{raw:?}");
            assert_eq!(resolved.ignored_raw.as_deref(), Some(raw), "{raw:?}");
        }

        let defaulted = layered(Some("invalid"), None);
        assert_eq!(defaulted.value, LeverValue::U64(45));
        assert_eq!(defaulted.source, Source::Default);
        assert_eq!(defaulted.ignored_raw.as_deref(), Some("invalid"));
    }

    #[test]
    fn schedule_open_fraction_kinds_preserve_their_distinct_layering() {
        use crate::decls::margin_row::{
            MARGIN_ROW_ADAPTIVE_RELEASE_FRAC, MARGIN_ROW_RESERVE_MAX_FRAC,
        };

        let layered = |decl: &'static LeverDecl, raw: Option<&str>, config: Option<f64>| {
            let owned = raw.map(str::to_owned);
            read_over_config_with(decl, move |_| owned, config.map(LeverValue::F64))
                .expect("open-fraction config is admissible")
        };

        for decl in [
            &MARGIN_ROW_RESERVE_MAX_FRAC,
            &MARGIN_ROW_ADAPTIVE_RELEASE_FRAC,
        ] {
            let valid = layered(decl, Some("0.25"), Some(0.75));
            assert_eq!(valid.value, LeverValue::F64(0.25), "{}", decl.name);
            assert_eq!(valid.source, Source::LegacyEnv, "{}", decl.name);
            assert_eq!(valid.lower_value, LeverValue::F64(0.75), "{}", decl.name);
            assert_eq!(valid.lower_source, Source::Config, "{}", decl.name);
        }

        // The ceiling treats every invalid Unicode token as an owned kill:
        // typed config is suppressed all the way to its Unset declaration
        // default.
        for raw in ["0", "1", " 0.5", "0.5 ", "NaN", "inf", "invalid"] {
            let killed = layered(&MARGIN_ROW_RESERVE_MAX_FRAC, Some(raw), Some(0.75));
            assert_eq!(killed.value, LeverValue::Unset, "{raw:?}");
            assert_eq!(killed.source, Source::LegacyEnvRejected, "{raw:?}");
            assert_eq!(killed.rejected_raw.as_deref(), Some(raw), "{raw:?}");
            assert_eq!(killed.lower_value, LeverValue::Unset, "{raw:?}");
            assert_eq!(killed.lower_source, Source::Default, "{raw:?}");
        }

        // The release threshold uses ordinary override fallthrough for the
        // same invalid grammar.
        for raw in ["0", "1", " 0.5", "0.5 ", "NaN", "inf", "invalid"] {
            let configured = layered(&MARGIN_ROW_ADAPTIVE_RELEASE_FRAC, Some(raw), Some(0.75));
            assert_eq!(configured.value, LeverValue::F64(0.75), "{raw:?}");
            assert_eq!(configured.source, Source::Config, "{raw:?}");
            assert_eq!(configured.rejected_raw, None, "{raw:?}");
            assert_eq!(configured.ignored_raw.as_deref(), Some(raw), "{raw:?}");
        }
        let defaulted = layered(&MARGIN_ROW_ADAPTIVE_RELEASE_FRAC, Some("bad"), None);
        assert_eq!(defaulted.value, LeverValue::F64(0.40));
        assert_eq!(defaulted.source, Source::Default);
    }

    #[test]
    fn trimmed_exact_one_accepts_every_unicode_token_as_a_bool_override() {
        use crate::decls::margin_row::MARGIN_ROW_CONCURRENT;

        let layered = |raw: Option<&str>, config: bool| {
            let owned = raw.map(str::to_owned);
            read_over_config_with(
                &MARGIN_ROW_CONCURRENT,
                move |_| owned,
                Some(LeverValue::Bool(config)),
            )
            .expect("bool schedule config is admissible")
        };

        for raw in ["1", " 1", "1 ", "\t1\n"] {
            let armed = layered(Some(raw), false);
            assert_eq!(armed.value, LeverValue::Bool(true), "{raw:?}");
            assert_eq!(armed.source, Source::LegacyEnv, "{raw:?}");
            assert_eq!(armed.rejected_raw, None, "{raw:?}");
        }
        for raw in ["0", "true", "on", "01", "garbage", "", " 0 "] {
            let killed = layered(Some(raw), true);
            assert_eq!(killed.value, LeverValue::Bool(false), "{raw:?}");
            assert_eq!(killed.source, Source::LegacyEnv, "{raw:?}");
            assert_eq!(killed.rejected_raw, None, "{raw:?}");
            assert_eq!(killed.ignored_raw, None, "{raw:?}");
        }
        let configured = layered(None, true);
        assert_eq!(configured.value, LeverValue::Bool(true));
        assert_eq!(configured.source, Source::Config);
    }

    #[cfg(unix)]
    #[test]
    fn schedule_kinds_preserve_all_five_non_unicode_layering_contracts() {
        use crate::decls::margin_row::{
            MARGIN_ROW_ADAPTIVE_RELEASE_FRAC, MARGIN_ROW_ADAPTIVE_RESERVE, MARGIN_ROW_CONCURRENT,
            MARGIN_ROW_RESERVE_MAX_FRAC, MARGIN_ROW_RESERVE_SECS,
        };
        use std::os::unix::ffi::OsStringExt;

        let raw = OsString::from_vec(vec![b'1', 0xff]);
        let cases = [
            (&MARGIN_ROW_RESERVE_SECS, LeverValue::U64(82)),
            (&MARGIN_ROW_RESERVE_MAX_FRAC, LeverValue::F64(0.75)),
            (&MARGIN_ROW_ADAPTIVE_RELEASE_FRAC, LeverValue::F64(0.75)),
            (&MARGIN_ROW_CONCURRENT, LeverValue::Bool(true)),
        ];
        for (decl, config) in cases {
            let resolved = resolve_raw(decl, Some(raw.as_os_str()), Some(config.clone())).unwrap();
            assert_eq!(resolved.value, config, "{}", decl.name);
            assert_eq!(resolved.source, Source::Config, "{}", decl.name);
            assert_eq!(resolved.env_utf8, Some(false), "{}", decl.name);
            assert_eq!(resolved.rejected_raw, None, "{}", decl.name);
            assert!(resolved.ignored_raw.is_some(), "{}", decl.name);
        }

        // Adaptive reserve is deliberately the exception: its var_os reader
        // sees non-Unicode presence and fails closed, suppressing typed true.
        let adaptive = resolve_raw(
            &MARGIN_ROW_ADAPTIVE_RESERVE,
            Some(raw.as_os_str()),
            Some(LeverValue::Bool(true)),
        )
        .unwrap();
        assert_eq!(adaptive.value, LeverValue::Bool(false));
        assert_eq!(adaptive.source, Source::LegacyEnvRejected);
        assert_eq!(adaptive.env_utf8, Some(false));
        assert!(adaptive.rejected_raw.is_some());
        assert_eq!(adaptive.ignored_raw, None);
    }

    #[test]
    fn trimmed_usize_override_ignores_invalid_input_and_preserves_lower_layer() {
        use crate::decls::margin_row::MARGIN_ROW_K_HEAD;

        let layered = |raw: Option<String>, config: Option<u64>| {
            read_over_config_with(
                &MARGIN_ROW_K_HEAD,
                move |_| raw,
                config.map(LeverValue::U64),
            )
            .expect("fixture config is admissible")
        };

        let valid = layered(Some("  +4\n".to_owned()), Some(16));
        assert_eq!(valid.value, LeverValue::U64(4));
        assert_eq!(valid.source, Source::LegacyEnv);
        assert_eq!(valid.lower_value, LeverValue::U64(16));
        assert_eq!(valid.lower_source, Source::Config);
        assert_eq!(valid.ignored_raw, None);

        let overflow = format!("{}0", usize::MAX);
        for ignored in ["0".to_owned(), "malformed".to_owned(), overflow] {
            let configured = layered(Some(ignored.clone()), Some(16));
            assert_eq!(configured.value, LeverValue::U64(16), "{ignored:?}");
            assert_eq!(configured.source, Source::Config, "{ignored:?}");
            assert_eq!(configured.lower_value, LeverValue::U64(8), "{ignored:?}");
            assert_eq!(configured.lower_source, Source::Default, "{ignored:?}");
            assert_eq!(configured.rejected_raw, None, "{ignored:?}");
            assert_eq!(configured.ignored_raw.as_deref(), Some(ignored.as_str()));
        }

        let defaulted = layered(Some("0".to_owned()), None);
        assert_eq!(defaulted.value, LeverValue::U64(8));
        assert_eq!(defaulted.source, Source::Default);
        assert_eq!(defaulted.lower_value, LeverValue::U64(8));
        assert_eq!(defaulted.lower_source, Source::Default);
        assert_eq!(defaulted.ignored_raw.as_deref(), Some("0"));

        let zero_config =
            read_over_config_with(&MARGIN_ROW_K_HEAD, |_| None, Some(LeverValue::U64(0)))
                .expect("the environment floor must not reject a typed zero sentinel");
        assert_eq!(zero_config.value, LeverValue::U64(0));
        assert_eq!(zero_config.source, Source::Config);
        assert_eq!(zero_config.lower_value, LeverValue::U64(8));
        assert_eq!(zero_config.lower_source, Source::Default);

        let ignored_zero_over_zero = layered(Some("0".to_owned()), Some(0));
        assert_eq!(ignored_zero_over_zero.value, LeverValue::U64(0));
        assert_eq!(ignored_zero_over_zero.source, Source::Config);
        assert_eq!(ignored_zero_over_zero.lower_value, LeverValue::U64(8));
        assert_eq!(ignored_zero_over_zero.lower_source, Source::Default);
        assert_eq!(ignored_zero_over_zero.ignored_raw.as_deref(), Some("0"));
    }

    #[test]
    fn trimmed_unicode_owned_usize_suppresses_config_only_for_unicode_presence() {
        let layered = |raw: Option<String>, config: Option<u64>| {
            read_over_config_with(
                &TRIMMED_UNICODE_OWNED_USIZE,
                move |_| raw,
                config.map(LeverValue::U64),
            )
            .expect("platform-sized fixture config is admissible")
        };

        for (raw, expected) in [
            ("0".to_owned(), 0),
            ("  +4\n".to_owned(), 4),
            (usize::MAX.to_string(), usize::MAX as u64),
        ] {
            let resolved = layered(Some(raw.clone()), Some(16));
            assert_eq!(resolved.value, LeverValue::U64(expected), "{raw:?}");
            assert_eq!(resolved.source, Source::LegacyEnv, "{raw:?}");
            assert_eq!(resolved.lower_value, LeverValue::U64(16), "{raw:?}");
            assert_eq!(resolved.lower_source, Source::Config, "{raw:?}");
        }

        let overflow = format!("{}0", usize::MAX);
        for rejected_raw in [
            String::new(),
            " \t\n".to_owned(),
            "malformed".to_owned(),
            "-1".to_owned(),
            "-0".to_owned(),
            "1.5".to_owned(),
            "\u{0664}".to_owned(),
            overflow,
        ] {
            let rejected = layered(Some(rejected_raw.clone()), Some(16));
            assert_eq!(rejected.value, LeverValue::Unset, "{rejected_raw:?}");
            assert_eq!(
                rejected.source,
                Source::LegacyEnvRejected,
                "{rejected_raw:?}"
            );
            assert_eq!(
                rejected.rejected_raw.as_deref(),
                Some(rejected_raw.as_str()),
                "{rejected_raw:?}"
            );
            assert_eq!(rejected.ignored_raw, None, "{rejected_raw:?}");
            assert_eq!(rejected.lower_value, LeverValue::Unset, "{rejected_raw:?}");
            assert_eq!(rejected.lower_source, Source::Default, "{rejected_raw:?}");
        }

        let configured_zero = layered(None, Some(0));
        assert_eq!(configured_zero.value, LeverValue::U64(0));
        assert_eq!(configured_zero.source, Source::Config);
    }

    #[test]
    fn untrimmed_usize_override_ignores_parse_failures_and_preserves_zero() {
        let layered = |raw: Option<String>, config: Option<u64>| {
            read_over_config_with(
                &UNTRIMMED_USIZE_OVERRIDE,
                move |_| raw,
                config.map(LeverValue::U64),
            )
            .expect("platform-sized fixture config is admissible")
        };

        for (raw, expected) in [
            ("0".to_owned(), 0),
            ("+4".to_owned(), 4),
            (usize::MAX.to_string(), usize::MAX as u64),
        ] {
            let resolved = layered(Some(raw.clone()), Some(16));
            assert_eq!(resolved.value, LeverValue::U64(expected), "{raw:?}");
            assert_eq!(resolved.source, Source::LegacyEnv, "{raw:?}");
            assert_eq!(resolved.lower_value, LeverValue::U64(16), "{raw:?}");
            assert_eq!(resolved.lower_source, Source::Config, "{raw:?}");
        }

        let overflow = format!("{}0", usize::MAX);
        for ignored_raw in [
            " 0".to_owned(),
            "0 ".to_owned(),
            "\t0".to_owned(),
            "malformed".to_owned(),
            "-1".to_owned(),
            "1.5".to_owned(),
            "\u{0664}".to_owned(),
            overflow,
        ] {
            let configured = layered(Some(ignored_raw.clone()), Some(16));
            assert_eq!(configured.value, LeverValue::U64(16), "{ignored_raw:?}");
            assert_eq!(configured.source, Source::Config, "{ignored_raw:?}");
            assert_eq!(configured.rejected_raw, None, "{ignored_raw:?}");
            assert_eq!(
                configured.ignored_raw.as_deref(),
                Some(ignored_raw.as_str()),
                "{ignored_raw:?}"
            );
        }

        let configured_zero = layered(None, Some(0));
        assert_eq!(configured_zero.value, LeverValue::U64(0));
        assert_eq!(configured_zero.source, Source::Config);

        let env_zero_over_typed_zero = resolve_raw(
            &UNTRIMMED_POSITIVE_USIZE_OVERRIDE,
            Some(OsStr::new("0")),
            Some(LeverValue::U64(0)),
        )
        .expect("the environment floor must not constrain typed config");
        assert_eq!(env_zero_over_typed_zero.value, LeverValue::U64(0));
        assert_eq!(env_zero_over_typed_zero.source, Source::Config);
        assert_eq!(env_zero_over_typed_zero.ignored_raw.as_deref(), Some("0"));

        let env_one_over_typed_zero = resolve_raw(
            &UNTRIMMED_POSITIVE_USIZE_OVERRIDE,
            Some(OsStr::new("1")),
            Some(LeverValue::U64(0)),
        )
        .expect("the positive environment override is admissible");
        assert_eq!(env_one_over_typed_zero.value, LeverValue::U64(1));
        assert_eq!(env_one_over_typed_zero.source, Source::LegacyEnv);
        assert_eq!(env_one_over_typed_zero.lower_value, LeverValue::U64(0));
        assert_eq!(env_one_over_typed_zero.lower_source, Source::Config);
    }

    #[cfg(unix)]
    #[test]
    fn scheduler_usize_kinds_ignore_non_unicode_bytes_and_preserve_config() {
        use std::os::unix::ffi::OsStringExt;

        let raw = OsString::from_vec(vec![b'4', 0xff]);
        for decl in [&TRIMMED_UNICODE_OWNED_USIZE, &UNTRIMMED_USIZE_OVERRIDE] {
            let resolved = resolve_raw(decl, Some(raw.as_os_str()), Some(LeverValue::U64(16)))
                .expect("platform-sized fixture config is admissible");
            assert_eq!(resolved.value, LeverValue::U64(16), "{}", decl.name);
            assert_eq!(resolved.source, Source::Config, "{}", decl.name);
            assert_eq!(resolved.lower_value, LeverValue::Unset, "{}", decl.name);
            assert_eq!(resolved.lower_source, Source::Default, "{}", decl.name);
            assert_eq!(resolved.env_utf8, Some(false), "{}", decl.name);
            assert_eq!(resolved.rejected_raw, None, "{}", decl.name);
            assert!(resolved.ignored_raw.is_some(), "{}", decl.name);
        }
    }

    #[cfg(unix)]
    #[test]
    fn trimmed_usize_override_ignores_non_unicode_and_preserves_lower_layer() {
        use crate::decls::margin_row::MARGIN_ROW_K_HEAD;
        use std::os::unix::ffi::OsStringExt;

        let raw = OsString::from_vec(vec![b'4', 0xff]);
        let resolved = resolve_raw(
            &MARGIN_ROW_K_HEAD,
            Some(raw.as_os_str()),
            Some(LeverValue::U64(16)),
        )
        .unwrap();
        assert_eq!(resolved.value, LeverValue::U64(16));
        assert_eq!(resolved.source, Source::Config);
        assert_eq!(resolved.lower_value, LeverValue::U64(8));
        assert_eq!(resolved.lower_source, Source::Default);
        assert_eq!(resolved.env_utf8, Some(false));
        assert!(resolved.rejected_raw.is_none());
        assert!(resolved.ignored_raw.is_some());
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

    #[test]
    fn trimmed_seconds_preserve_margin_row_alpha_legacy_grammar() {
        use crate::decls::sound_channel::MARGIN_ROW_ALPHA_SECS;

        for (raw, expected) in [("12.5", 12.5), (" 12.5 ", 12.5), ("\t1\n", 1.0)] {
            let resolved = with(Some(raw), &MARGIN_ROW_ALPHA_SECS);
            assert_eq!(resolved.value, LeverValue::Secs(expected), "{raw:?}");
            assert_eq!(resolved.source, Source::LegacyEnv, "{raw:?}");
            assert_eq!(resolved.rejected_raw, None, "{raw:?}");
        }

        for rejected in ["malformed", "NaN", "inf", "-1"] {
            let resolved = with(Some(rejected), &MARGIN_ROW_ALPHA_SECS);
            assert_eq!(resolved.value, LeverValue::Secs(20.0), "{rejected:?}");
            assert_eq!(resolved.source, Source::LegacyEnvRejected, "{rejected:?}");
            assert_eq!(
                resolved.rejected_raw.as_deref(),
                Some(rejected),
                "{rejected:?}"
            );
        }

        // Zero is syntactically valid seconds and remains visible in the
        // receipt. The consuming alpha policy preserves the historical `> 0`
        // effective-value filter and maps it to its 20-second fallback.
        let zero = with(Some(" 0 "), &MARGIN_ROW_ALPHA_SECS);
        assert_eq!(zero.value, LeverValue::Secs(0.0));
        assert_eq!(zero.source, Source::LegacyEnv);
    }

    #[test]
    fn exact_vjp_default_on_selectors_preserve_exact_zero_kills() {
        use crate::decls::exact_vjp_pgd::{EXACT_BATCHED, EXPLOIT, FAST_GRAD, VJP_BATCH};

        for decl in [&EXACT_BATCHED, &FAST_GRAD, &VJP_BATCH, &EXPLOIT] {
            assert_eq!(with(None, decl).value, LeverValue::Bool(true));
            assert_eq!(with(Some("0"), decl).value, LeverValue::Bool(false));
            assert_eq!(with(Some("1"), decl).value, LeverValue::Bool(true));
            for rejected in ["", "false", "00", " 0", "0 ", "2"] {
                let resolved = with(Some(rejected), decl);
                assert_eq!(resolved.value, LeverValue::Bool(true), "{rejected:?}");
                assert_eq!(resolved.source, Source::LegacyEnvRejected, "{rejected:?}");
            }
        }
    }

    #[cfg(unix)]
    #[test]
    fn exact_vjp_default_on_non_unicode_restores_on_default() {
        use crate::decls::exact_vjp_pgd::{EXACT_BATCHED, FAST_GRAD};
        use std::os::unix::ffi::OsStringExt;

        let raw = OsString::from_vec(vec![b'0', 0xff]);
        for decl in [&EXACT_BATCHED, &FAST_GRAD] {
            let resolved = resolve_raw(decl, Some(raw.as_os_str()), None).unwrap();
            assert_eq!(resolved.value, LeverValue::Bool(true));
            assert_eq!(resolved.source, Source::LegacyEnvRejected);
            assert_eq!(resolved.env_utf8, Some(false));
        }
    }

    #[cfg(unix)]
    #[test]
    fn truthy_trimmed_non_unicode_falls_through_to_typed_config() {
        use crate::decls::margin_row::MARGIN_ROW_ROOT_F32;
        use std::os::unix::ffi::OsStringExt;

        let raw = OsString::from_vec(vec![b'1', 0xff]);
        let configured = resolve_raw(
            &MARGIN_ROW_ROOT_F32,
            Some(raw.as_os_str()),
            Some(LeverValue::Bool(true)),
        )
        .unwrap();
        assert_eq!(configured.value, LeverValue::Bool(true));
        assert_eq!(configured.source, Source::Config);
        assert_eq!(configured.env_utf8, Some(false));
        assert!(configured.rejected_raw.is_none());
        assert!(configured.ignored_raw.is_some());

        let defaulted = resolve_raw(&MARGIN_ROW_ROOT_F32, Some(raw.as_os_str()), None).unwrap();
        assert_eq!(defaulted.value, LeverValue::Bool(false));
        assert_eq!(defaulted.source, Source::Default);
        assert_eq!(defaulted.env_utf8, Some(false));
        assert!(defaulted.rejected_raw.is_none());
        assert!(defaulted.ignored_raw.is_some());
    }

    #[test]
    fn exact_vjp_usize_is_untrimmed_platform_sized_and_range_checked() {
        use crate::decls::exact_vjp_pgd::{LEGACY_LANES, VJP_K};

        assert_eq!(with(Some("1"), &VJP_K).value, LeverValue::U64(1));
        assert_eq!(with(Some("+1"), &VJP_K).value, LeverValue::U64(1));
        assert_eq!(
            with(Some(&usize::MAX.to_string()), &VJP_K).value,
            LeverValue::U64(usize::MAX as u64)
        );
        for rejected in ["0", " 1", "1 ", "-1", "1.0"] {
            let resolved = with(Some(rejected), &VJP_K);
            assert_eq!(resolved.value, LeverValue::Unset, "{rejected:?}");
            assert_eq!(resolved.source, Source::LegacyEnvRejected, "{rejected:?}");
        }
        let overflow = format!("{}0", usize::MAX);
        assert_eq!(with(Some(&overflow), &VJP_K).value, LeverValue::Unset);
        assert_eq!(with(Some("0"), &LEGACY_LANES).value, LeverValue::U64(0));
    }

    #[test]
    fn exact_vjp_f32_parsers_preserve_sign_finiteness_and_rounding() {
        use crate::decls::exact_vjp_pgd::{HINGE_TAU, ODI_ETA};

        let canonical_eta = f64::from(0.0001_f32);
        assert_eq!(
            with(Some("0.0001"), &ODI_ETA).value,
            LeverValue::F64(canonical_eta)
        );
        assert_eq!(with(Some("+1"), &ODI_ETA).value, LeverValue::F64(1.0));
        for rejected in [
            "0", "-0", "-1", " 1", "1 ", "NaN", "inf", "-inf", "1e-50", "1e100",
        ] {
            let resolved = with(Some(rejected), &ODI_ETA);
            assert_eq!(resolved.value, LeverValue::F64(1.0), "{rejected:?}");
            assert_eq!(resolved.source, Source::LegacyEnvRejected, "{rejected:?}");
        }

        assert_eq!(with(Some("0"), &HINGE_TAU).value, LeverValue::F64(0.0));
        assert_eq!(with(Some("-0"), &HINGE_TAU).value, LeverValue::F64(-0.0));
        for rejected in ["-1", " 0", "0 ", "NaN", "inf", "-inf", "1e100"] {
            let resolved = with(Some(rejected), &HINGE_TAU);
            assert_eq!(resolved.value, LeverValue::F64(0.0), "{rejected:?}");
            assert_eq!(resolved.source, Source::LegacyEnvRejected, "{rejected:?}");
        }
    }

    #[test]
    fn exact_vjp_f32_config_must_round_trip_to_runtime_f32_exactly() {
        use crate::decls::exact_vjp_pgd::{HINGE_TAU, ODI_ETA};

        assert!(
            read_over_config_with(&ODI_ETA, |_| None, Some(LeverValue::F64(1.00000001))).is_err()
        );
        let canonical = f64::from(0.0001_f32);
        let resolved =
            read_over_config_with(&HINGE_TAU, |_| None, Some(LeverValue::F64(canonical))).unwrap();
        assert_eq!(resolved.value, LeverValue::F64(canonical));
        assert_eq!(resolved.source, Source::Config);
    }
}
