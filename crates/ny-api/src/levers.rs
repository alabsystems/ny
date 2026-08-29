// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! The `NY_*` LEVER SURFACE, for an embedded consumer.
//!
//! ny's runtime configuration lives in `ny-levers`: a central registry of
//! declarations (what each lever is, what it parses to, what its shipped
//! default is, who reads it) plus a SEARCH SPACE that classifies each lever as
//! [`SearchClass::Unsafe`], [`SearchClass::VerdictAffecting`] or
//! [`SearchClass::SafeToSearch`] and records whether a typed preset key can
//! deliver it to a scored run. Until this module existed, none of that was
//! reachable from this facade: the `ny` binary got tuned configurations and
//! every library consumer got declaration defaults, with no way to enumerate
//! the surface, no receipt, and no typed way to change anything.
//!
//! This module is deliberately narrow. It offers three things:
//!
//! 1. **Discovery** — [`catalog`], [`describe`], [`settable`]. Read straight
//!    out of `ny_levers::all()` and `ny_levers::space`, never a second copy of
//!    the list.
//! 2. **Delivery for the `SafeToSearch` subset only** — [`LeverPlan`] plus
//!    [`install`]. Naming an [`SearchClass::Unsafe`] or
//!    [`SearchClass::VerdictAffecting`] lever is a typed [`Refusal`] that says
//!    why.
//! 3. **Audit** — [`ambient_overrides`] and [`receipt_json`], which report what
//!    the PROCESS ENVIRONMENT is already contributing to this run.
//!
//! # This module never writes the process environment
//!
//! Not "prefers not to" — never. `ny_levers::read` calls `std::env::var_os` on
//! every invocation, and ny-propagate calls it from live propagation paths
//! while verification threads are running. A library that mutated the
//! environment would therefore be changing verifier behaviour underneath a
//! running verification, from another thread, with no synchronisation. The
//! delivery route used here is the same one `ny-cli` uses: typed
//! preset-override CELLS in `ny_levers::gate`, written through
//! `ny_propagate::preset_gates`, which layer UNDER the environment and over the
//! declaration default.
//!
//! # AMBIENT INHERITANCE — the hazard this module exists to expose
//!
//! Because every read goes to the live process environment, an embedded
//! consumer inherits the entire `NY_*` surface from whatever shell, CI runner
//! or launch agent started the host process, WITHOUT OPTING IN. That inheritance
//! is invisible: a value picked up this way is indistinguishable, from the
//! caller's side, from the shipped default.
//!
//! * Exposure begins exactly at the `propagate` feature. `ny-core`,
//!   `ny-tensor`, `ny-build` and this crate contain no `NY_*` reads of their
//!   own; `ny-propagate` is where the readers live, and it is what `propagate`
//!   pulls in. That is why this module is gated on the same feature.
//! * Most of the registry is live in that closure. Measured on 2026-08-27 over
//!   the 197 declarations: **136 declare a reader in an `ny-api` closure
//!   package**, of which **107 carry `MoatRisk::High`** — the declaration's own
//!   words for "can change a published bound or verdict on the authoritative
//!   route". So yes: verdict-affecting levers are inherited, in bulk, by a
//!   consumer that never asked for one. [`catalog`] recomputes this at runtime;
//!   `the_ambient_inheritance_note_matches_the_registry` keeps the shape
//!   pinned rather than trusting this paragraph.
//! * **Ten of the live ones are [`SearchClass::Unsafe`]** — the class
//!   `ny-levers` defines as "never moved by an automated search":
//!   `NY_CROWN_HONEST_PROVENANCE`, `NY_CROWN_OBJ_CHUNK`, `NY_NO_CHUNK_ABORT`,
//!   `NY_NO_CHUNK_GROW`, `NY_PATCHES_DEADLINE_FLAT_BIAS`,
//!   `NY_BRANCH_KFSB_CHILDSIM`, `NY_JOINT_BILINEAR_QUAD`,
//!   `NY_ROOT_JOINT_INTERM_ALPHA_LAYERS`, `NY_MARGIN_ROW_EPOCH_PARENT_TOPK`
//!   and `NY_EFT_ERR` (whose ny-propagate arm its own declaration marks as
//!   staged and unwired, so count that one as reachable-on-paper). The sharpest is
//!   `NY_CROWN_HONEST_PROVENANCE`: it ships ON, its exact-`0` arm restores
//!   unconditional Crown/tight provenance for an almost-entirely non-finite
//!   CROWN result, and its reader
//!   (`ny_propagate::network::crown_quality::honest_provenance_enabled`) sits
//!   squarely inside this facade's dependency closure. Nothing between a host
//!   process's environment and that switch asks the consumer's permission.
//!   Bounds stay sound — what moves is the PROVENANCE LABEL a consumer uses to
//!   decide whether to trust a result, which is a reporting-integrity failure
//!   rather than a false bound, but it is one the registry itself calls unsafe.
//!
//! And the registry is not the whole exposure. Measured over
//! `crates/ny-propagate/src` on 2026-08-27 —
//! `grep -rhoE 'env::var(_os)?\("NY_[A-Z0-9_]*"'` — ny-propagate reads **346
//! distinct `NY_*` names as raw literals**, of which only 5 are also registry
//! declarations. The remaining **341 are UNDECLARED**: no `LeverDecl`, no
//! class, and — this is the part that matters — no line in the receipt.
//! [`catalog`] and [`ambient_overrides`] can only report the 197 declared
//! levers, so treat their output as a lower bound on what a contaminated
//! environment is contributing, never as a complete inventory. `ny-levers`
//! tracks that gap itself as the raw-read ratchet
//! (`crates/ny-levers/src/ratchet_baseline.txt`, 710 tree-wide against a stated
//! target of 0).
//!
//! There is no supported way for a library to make ny IGNORE its process
//! environment: `ny_levers::LeverSet::freeze_with(|_| None)` builds a clean
//! defaults-only snapshot, but no runtime reader consumes a `LeverSet` yet
//! (that is `ny-levers`' own Phase 2), so such a snapshot would describe a run
//! that did not happen. The two defences that actually work today are (a) scrub
//! `NY_*` from the environment in the HOST process, before ny is first called,
//! and (b) record [`ambient_overrides`] into your evidence so a contaminated
//! run is identifiable after the fact.
//!
//! # Why there is no `PresetConfig` here
//!
//! ny ships 50 tuned YAML presets (`configs/vnncomp2{4,5,6}/*.yaml`) and a typed
//! `PresetConfig` schema that loads them. Neither can be offered through this
//! facade, and the blocker is
//! packaging rather than design: the schema, its loader, its validator and its
//! `apply_preset` all live in `crates/ny-cli`, which declares only
//! `[[bin]] name = "ny"` and has no `src/lib.rs`. Every item is `pub(crate)` in
//! a binary target, so no crate in the workspace — let alone an external
//! consumer — can name `PresetConfig`. Closing that gap means extracting a
//! library crate; it is not something this module can work around, and
//! pretending otherwise by re-typing the schema here would create exactly the
//! two-copies-that-drift failure the registry exists to prevent. [`LeverPlan`]
//! is the data-shaped substitute for the part that is safe to expose.
//!
//! # Deliverable is not the same as reached
//!
//! [`install`] confirms that a lever RESOLVES to the value you asked for. It
//! cannot promise that the code reading it executes. Of the twelve levers wired
//! here, eight belong to the MARGIN-ROW lane (`NY_MARGIN_ROW_*`, `NY_EPOCH_*`),
//! and `run_margin_row_lane` has no caller outside `ny-cli` and ny-propagate's
//! own tests — this facade exposes no margin-row entry point at all. Setting
//! them is honest and verified, and currently inert for an `ny-api` consumer.
//! The four that sit on paths an `ny-api` verification actually walks are
//! `NY_PATCHES_FINITE_EXPIRY` (the CROWN Patches step and the sound-GPU gate),
//! `NY_NO_WALK_RECORD_ADMISSION` (the α-CROWN budget policy),
//! `NY_CONV_IBP_BLOCKED` (the certified f64 Conv2d IBP forward) and
//! `NY_CGAN_STACKED_BUDGET_MB` (the stacked-backward memory admission).
//!
//! # Ordering — read this before calling [`install`]
//!
//! Several readers latch their answer in a process-wide `OnceLock` the first
//! time they run (`patches_finite_expiry_enabled` and
//! `margin_row_clip_rows_enabled` among the levers wired here). A cell written
//! after the latch has been taken is invisible, and `ny_levers::gate`
//! deliberately offers no reset outside tests. So [`install`] must be called
//! ONCE, during setup, before the first verification — and it refuses a second
//! call rather than silently writing cells nothing will read.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::OnceLock;

use ny_levers::space::{self, Class, Deliver};
use ny_levers::{LeverDecl, LeverValue, Source};

// ---------------------------------------------------------------------------
// Discovery
// ---------------------------------------------------------------------------

/// How an automated search may treat a lever — `ny_levers::space::Class`,
/// widened with the case that has no axis at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchClass {
    /// Never moved by an automated search: it changes what is being proved,
    /// replaces a certified error bound, or hard-fails engine construction.
    /// [`install`] refuses these.
    Unsafe,
    /// May change a published bound. Searchable under review, but not through
    /// this facade: [`install`] refuses these too.
    VerdictAffecting,
    /// Cannot change a verdict's SOUNDNESS. This is the only class [`install`]
    /// will deliver.
    ///
    /// Note that "safe to search" is not "free" and not "cannot change the
    /// answer": several of these carry `MoatRisk::High`, because a looser bound
    /// can turn a proof into an `Unknown`. They cost tightness and wall clock,
    /// never validity. [`LeverInfo::moat`] reports the declaration's own blast
    /// radius so the two cannot be confused.
    SafeToSearch,
    /// Declared in the registry but not a search dimension — an instrument, a
    /// test-only name, or a lever nobody has classified.
    NotAnAxis,
}

impl SearchClass {
    /// Stable lowercase name, matching `ny-levers`' own receipt vocabulary.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unsafe => "unsafe",
            Self::VerdictAffecting => "verdict_affecting",
            Self::SafeToSearch => "safe_to_search",
            Self::NotAnAxis => "not_an_axis",
        }
    }

    const fn from_axis(class: Class) -> Self {
        match class {
            Class::Unsafe => Self::Unsafe,
            Class::VerdictAffecting => Self::VerdictAffecting,
            Class::SafeToSearch => Self::SafeToSearch,
        }
    }
}

/// What an embedded consumer can actually do about a lever.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Delivery {
    /// [`install`] delivers this one, in-process, before the first
    /// verification. The lever is [`SearchClass::SafeToSearch`], a typed cell
    /// writer exists inside this facade's dependency closure, and there is a
    /// side-effect-free reader to confirm the write with.
    ///
    /// DELIVERABLE IS NOT THE SAME AS REACHED. The cell is written and the
    /// resolver confirms it, but whether the reading code runs at all depends
    /// on which lane your verification takes — see the module documentation's
    /// note on the margin-row lane.
    Settable {
        /// The typed preset key this lever is promoted under.
        preset_key: &'static str,
    },
    /// A typed preset key exists and a writer exists — and this facade refuses
    /// to expose it, because the lever is [`SearchClass::Unsafe`] or
    /// [`SearchClass::VerdictAffecting`]. Reaching past the facade to
    /// `ny_propagate::preset_gates` is possible and is not a supported use of
    /// this API.
    WithheldByClass {
        /// The typed preset key, so the refusal can name it.
        preset_key: &'static str,
    },
    /// The lever is safe to search and has a typed preset key, but this facade
    /// wires no writer for it. One of three reasons, all outside this module's
    /// control: the only writer is private to the `ny` binary (`ny-cli` has no
    /// `[lib]` target); or the value is delivered as an ordinary field on a
    /// config struct you already set directly, such as
    /// [`crate::bab::BetaCrownConfig`]; or the in-process writer has no
    /// side-effect-free reader, and [`install`] will not report a delivery it
    /// cannot verify.
    NotWiredHere {
        /// The typed preset key, so a caller can find the other route.
        preset_key: &'static str,
    },
    /// No typed preset key is recorded for this lever, so the process
    /// environment is the only layer that can change it — and this module never
    /// writes the process environment.
    EnvironmentOnly,
}

/// One lever, as an embedded consumer sees it.
#[derive(Debug, Clone, Copy)]
pub struct LeverInfo {
    /// The environment-variable name, e.g. `"NY_PATCHES_FINITE_EXPIRY"`.
    pub name: &'static str,
    /// Search classification, from `ny_levers::space`.
    pub search_class: SearchClass,
    /// What this facade offers for it.
    pub delivery: Delivery,
    /// The declaration's blast radius: `"none"`, `"low"` or `"high"`. NOT the
    /// same statement as [`Self::search_class`] — see [`SearchClass::SafeToSearch`].
    pub moat: &'static str,
    /// Who the declaration says may choose the value: `"default_on"`, `"auto"`,
    /// `"cli"` or `"debug"`.
    pub bucket: &'static str,
    /// The declaration's prose description of what the lever does when set.
    pub doc: &'static str,
    /// The search space's recorded rationale for the classification, when the
    /// lever is an axis.
    pub why_classified: Option<&'static str>,
    /// True when at least one DECLARED reader of this lever sits in a package
    /// inside this facade's own dependency closure — i.e. the lever is LIVE for
    /// a consumer of `ny-api`, and an ambient `NY_*` value for it will be
    /// picked up.
    ///
    /// Declared readers only, and in BOTH directions. `ny-propagate` contains
    /// undeclared raw `env::var` reads of `"NY_…"` literals that no registry entry
    /// covers, so this undercounts; and a declared reader may itself be staged
    /// or `#[allow(dead_code)]` (the `ny-propagate` arm of `NY_EFT_ERR` says so
    /// in its own `role` text), so for a specific lever it can also overcount.
    /// Read `LeverDecl::readers` when the exact site matters.
    pub live_in_process: bool,
}

/// Packages inside this facade's normal (non-dev) dependency closure.
///
/// Verified with `cargo tree -p ny-api --features propagate -e normal`:
/// ny-api, ny-build, ny-contracts, ny-core, ny-levers, ny-propagate,
/// ny-tensor. Notably ABSENT: ny-cli, ny-gpu, ny-onnx, ny-cuda — levers whose
/// only declared readers live there cannot fire for a consumer of this facade.
fn facade_packages() -> Vec<&'static str> {
    let mut packages = vec![
        "ny-api",
        "ny-build",
        "ny-contracts",
        "ny-core",
        "ny-levers",
        "ny-propagate",
        "ny-tensor",
    ];
    if cfg!(feature = "cert") {
        packages.push("ny-cert");
    }
    if cfg!(feature = "complete") {
        packages.push("ny-mip");
    }
    packages
}

fn axis_for(name: &str) -> Option<&'static space::Axis> {
    space::axes()
        .iter()
        .chain(space::unsafe_axes())
        .find(|axis| axis.name == name)
}

fn info_for(decl: &'static LeverDecl, packages: &[&'static str]) -> LeverInfo {
    let axis = axis_for(decl.name);
    let search_class = axis.map_or(SearchClass::NotAnAxis, |axis| {
        SearchClass::from_axis(axis.class)
    });
    let preset_key = axis.and_then(|axis| match axis.deliver {
        Deliver::PresetKey(key) => Some(key),
        Deliver::EnvOnly => None,
    });
    let delivery = match preset_key {
        None => Delivery::EnvironmentOnly,
        Some(preset_key) => {
            if search_class == SearchClass::SafeToSearch {
                if wired(decl.name).is_some() {
                    Delivery::Settable { preset_key }
                } else {
                    Delivery::NotWiredHere { preset_key }
                }
            } else {
                Delivery::WithheldByClass { preset_key }
            }
        }
    };
    LeverInfo {
        name: decl.name,
        search_class,
        delivery,
        moat: decl.moat.as_str(),
        bucket: decl.bucket.as_str(),
        doc: decl.doc,
        why_classified: axis.map(|axis| axis.why),
        live_in_process: decl
            .reader_packages()
            .iter()
            .any(|package| packages.contains(package)),
    }
}

/// Every declared lever, name-sorted, with its class and what this facade
/// offers for it.
///
/// The list is read from `ny_levers::all()` at call time. This module keeps no
/// copy of it: a lever added, removed or reclassified in `ny-levers` shows up
/// here on the next build with no edit.
#[must_use]
pub fn catalog() -> Vec<LeverInfo> {
    let packages = facade_packages();
    let mut out: Vec<LeverInfo> = ny_levers::all()
        .all()
        .iter()
        .map(|decl| info_for(decl, &packages))
        .collect();
    out.sort_unstable_by_key(|info| info.name);
    out
}

/// One lever by environment-variable name, or `None` when nothing declares it.
#[must_use]
pub fn describe(name: &str) -> Option<LeverInfo> {
    let decl = ny_levers::all().get(name)?;
    Some(info_for(decl, &facade_packages()))
}

/// The levers [`install`] will actually deliver, name-sorted.
///
/// A strict subset of the [`SearchClass::SafeToSearch`] axes; everything else
/// in the registry is reported by [`catalog`] with a [`Delivery`] explaining
/// why it is not here.
#[must_use]
pub fn settable() -> Vec<&'static str> {
    let mut names: Vec<&'static str> = WIRED.iter().map(|wired| wired.decl.name).collect();
    names.sort_unstable();
    names
}

// ---------------------------------------------------------------------------
// The wired subset
// ---------------------------------------------------------------------------

/// A typed value for a settable lever.
#[derive(Debug, Clone, PartialEq)]
pub enum Setting {
    /// A gate.
    Bool(bool),
    /// A count or capacity.
    Count(usize),
    /// A fraction, rate or step size.
    Real(f64),
    /// An enumerated or free-form token.
    Token(String),
}

impl Setting {
    const fn shape(&self) -> &'static str {
        match self {
            Self::Bool(_) => "Bool",
            Self::Count(_) => "Count",
            Self::Real(_) => "Real",
            Self::Token(_) => "Token",
        }
    }
}

impl fmt::Display for Setting {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bool(value) => write!(f, "{value}"),
            Self::Count(value) => write!(f, "{value}"),
            Self::Real(value) => write!(f, "{value}"),
            Self::Token(value) => write!(f, "{value:?}"),
        }
    }
}

/// The write half plus the confirm half of one wired lever.
///
/// Both halves are function pointers into `ny_propagate::preset_gates`, which
/// is the same typed-cell route `ny-cli` uses. The confirm half is required:
/// a lever with no side-effect-free reader is not wired, because [`install`]
/// would then have to report a delivery it could not verify.
#[derive(Clone, Copy)]
enum Port {
    Bool {
        write: fn(bool),
        read: fn() -> bool,
    },
    Count {
        write: fn(usize),
        read: fn() -> usize,
    },
    OptionalCount {
        write: fn(u64),
        read: fn() -> Option<u64>,
    },
    Real {
        write: fn(f64),
        read: fn() -> f64,
    },
    Token {
        write: fn(String),
        read: fn() -> Option<String>,
    },
}

impl Port {
    const fn shape(self) -> &'static str {
        match self {
            Self::Bool { .. } => "Bool",
            Self::Count { .. } | Self::OptionalCount { .. } => "Count",
            Self::Real { .. } => "Real",
            Self::Token { .. } => "Token",
        }
    }

    /// The typed value this port would hand `ny_levers` as the CONFIG layer.
    ///
    /// `None` means the setting's shape does not match the port's.
    fn as_config(self, setting: &Setting) -> Option<LeverValue> {
        match (self, setting) {
            (Self::Bool { .. }, Setting::Bool(value)) => Some(LeverValue::Bool(*value)),
            (Self::Count { .. } | Self::OptionalCount { .. }, Setting::Count(value)) => {
                u64::try_from(*value).ok().map(LeverValue::U64)
            }
            (Self::Real { .. }, Setting::Real(value)) => Some(LeverValue::F64(*value)),
            (Self::Token { .. }, Setting::Token(value)) => Some(LeverValue::Text(value.clone())),
            _ => None,
        }
    }

    fn write(self, setting: &Setting) {
        match (self, setting) {
            (Self::Bool { write, .. }, Setting::Bool(value)) => write(*value),
            (Self::Count { write, .. }, Setting::Count(value)) => write(*value),
            (Self::OptionalCount { write, .. }, Setting::Count(value)) => {
                // Pre-validated by `as_config`; a value that does not fit `u64`
                // never reaches here.
                if let Ok(value) = u64::try_from(*value) {
                    write(value);
                }
            }
            (Self::Real { write, .. }, Setting::Real(value)) => write(*value),
            (Self::Token { write, .. }, Setting::Token(value)) => write(value.clone()),
            _ => {}
        }
    }

    fn read(self) -> Option<Setting> {
        match self {
            Self::Bool { read, .. } => Some(Setting::Bool(read())),
            Self::Count { read, .. } => Some(Setting::Count(read())),
            Self::OptionalCount { read, .. } => {
                read().and_then(|value| usize::try_from(value).ok().map(Setting::Count))
            }
            Self::Real { read, .. } => Some(Setting::Real(read())),
            Self::Token { read, .. } => read().map(Setting::Token),
        }
    }
}

struct Wired {
    decl: &'static LeverDecl,
    port: Port,
}

macro_rules! wired {
    ($decl:path, Bool, $write:ident, $read:ident) => {
        Wired {
            decl: &$decl,
            port: Port::Bool {
                write: ny_propagate::preset_gates::$write,
                read: ny_propagate::preset_gates::$read,
            },
        }
    };
    ($decl:path, Count, $write:ident, $read:ident) => {
        Wired {
            decl: &$decl,
            port: Port::Count {
                write: ny_propagate::preset_gates::$write,
                read: ny_propagate::preset_gates::$read,
            },
        }
    };
    ($decl:path, OptionalCount, $write:ident, $read:ident) => {
        Wired {
            decl: &$decl,
            port: Port::OptionalCount {
                write: ny_propagate::preset_gates::$write,
                read: ny_propagate::preset_gates::$read,
            },
        }
    };
    ($decl:path, Real, $write:ident, $read:ident) => {
        Wired {
            decl: &$decl,
            port: Port::Real {
                write: ny_propagate::preset_gates::$write,
                read: ny_propagate::preset_gates::$read,
            },
        }
    };
}

/// The levers this facade delivers.
///
/// Every entry is pinned by `wired_levers_are_all_safe_to_search` to be
/// `Class::SafeToSearch` with a `Deliver::PresetKey` in `ny_levers::space`, so
/// a reclassification upstream breaks the build's tests rather than silently
/// widening what a consumer may set.
static WIRED: &[Wired] = &[
    // --- bab / propagation ------------------------------------------------
    wired!(
        ny_levers::decls::diagnostics::PATCHES_FINITE_EXPIRY,
        Bool,
        set_patches_finite_expiry_preset,
        patches_finite_expiry_enabled
    ),
    wired!(
        ny_levers::decls::collection::NO_WALK_RECORD_ADMISSION,
        Bool,
        set_no_walk_record_admission_preset,
        no_walk_record_admission_enabled
    ),
    wired!(
        ny_levers::decls::cgan_stacked::CGAN_STACKED_BUDGET_MB,
        OptionalCount,
        set_cgan_stacked_budget_mb_preset,
        cgan_stacked_budget_mb_override
    ),
    // --- margin row -------------------------------------------------------
    wired!(
        ny_levers::decls::sound_channel::MARGIN_ROW_CLIP_ROWS,
        Bool,
        set_margin_row_clip_rows_preset,
        margin_row_clip_rows_enabled
    ),
    wired!(
        ny_levers::decls::margin_row::MARGIN_ROW_LRU,
        Count,
        set_margin_row_lru_preset,
        margin_row_lru_cap
    ),
    wired!(
        ny_levers::decls::margin_row::EPOCH_TIER0,
        Count,
        set_epoch_tier0_preset,
        epoch_tier0
    ),
    wired!(
        ny_levers::decls::margin_row::EPOCH_UNIVERSE,
        Count,
        set_epoch_universe_preset,
        epoch_universe
    ),
    wired!(
        ny_levers::decls::margin_row::EPOCH_RETAIN,
        Count,
        set_epoch_retain_preset,
        epoch_retain
    ),
    wired!(
        ny_levers::decls::margin_row::MARGIN_ROW_BETA_ITERS,
        Count,
        set_margin_row_beta_iters_preset,
        margin_row_beta_iters
    ),
    wired!(
        ny_levers::decls::margin_row::MARGIN_ROW_BETA_ETA,
        Real,
        set_margin_row_beta_eta_preset,
        margin_row_beta_eta
    ),
    wired!(
        ny_levers::decls::margin_row::MARGIN_ROW_BETA_LAMBDA,
        Real,
        set_margin_row_beta_lambda_preset,
        margin_row_beta_lambda
    ),
    // `set_conv_ibp_blocked_preset` is generic over `Into<String>`, so it needs
    // a monomorphic shim before it can be a function pointer.
    Wired {
        decl: &ny_levers::decls::conv_ibp_blocked::CONV_IBP_BLOCKED,
        port: Port::Token {
            write: write_conv_ibp_blocked,
            read: ny_propagate::preset_gates::conv_ibp_blocked_token,
        },
    },
];

fn write_conv_ibp_blocked(token: String) {
    ny_propagate::preset_gates::set_conv_ibp_blocked_preset(token);
}

fn wired(name: &str) -> Option<&'static Wired> {
    WIRED.iter().find(|wired| wired.decl.name == name)
}

/// The value a settable lever resolves to RIGHT NOW: process environment first,
/// then any installed cell, then the declaration default.
///
/// `None` when `name` is not one of [`settable`]. This is the honest answer to
/// "did my setting take?", and it is also how an ambient environment override
/// becomes visible after the fact.
#[must_use]
pub fn effective(name: &str) -> Option<Setting> {
    wired(name).and_then(|wired| wired.port.read())
}

// ---------------------------------------------------------------------------
// Refusals
// ---------------------------------------------------------------------------

/// Why a lever request was refused. Every variant carries enough to explain
/// the refusal without a second lookup.
#[derive(Debug, Clone, PartialEq)]
pub enum RefusalReason {
    /// No declaration in `ny_levers::all()` carries this name. A typo'd
    /// `NY_*` name is refused here rather than silently doing nothing, which is
    /// exactly the contamination class the lever receipt exists to expose.
    NotDeclared,
    /// The lever is `Unsafe` or `VerdictAffecting`. This facade delivers only
    /// the `SafeToSearch` subset.
    WithheldByClass {
        /// The class that caused the refusal.
        class: SearchClass,
        /// The declaration's blast radius: `"none"`, `"low"`, `"high"`.
        moat: &'static str,
        /// The search space's own recorded rationale for the classification.
        why: &'static str,
    },
    /// Safe to search, but this facade wires no verifiable writer for it. See
    /// [`Delivery::NotWiredHere`].
    NotWiredHere {
        /// The typed preset key, so the caller can find the other route.
        preset_key: &'static str,
    },
    /// No typed preset key exists, and this module never writes the process
    /// environment. See [`Delivery::EnvironmentOnly`].
    EnvironmentOnly,
    /// The value's shape does not match the lever's.
    WrongShape {
        /// The shape the lever accepts: `"Bool"`, `"Count"`, `"Real"`, `"Token"`.
        expected: &'static str,
        /// The shape that was supplied.
        supplied: &'static str,
    },
    /// The declaration's own kind/range check rejects this value, so writing it
    /// would be silently discarded at resolution time.
    OutsideDeclaration,
    /// The PROCESS ENVIRONMENT already names this lever. An environment value
    /// wins over a typed cell, so installing the plan would either be
    /// overridden or — worse — appear to succeed while the run used the
    /// inherited value. Nothing is written; scrub the variable in the host
    /// process, or drop the lever from the plan.
    OverriddenByProcessEnvironment {
        /// The inherited value, rendered lossily when it is not UTF-8.
        inherited: String,
    },
    /// [`install`] has already run in this process. The readers latch, so a
    /// second, different installation could not take effect anyway; refusing is
    /// the only honest answer.
    AlreadyInstalled,
    /// The cell was written and the confirming read disagreed. This should be
    /// unreachable — the value was validated against the declaration and the
    /// environment was checked to be silent — and it is reported rather than
    /// ignored because a delivery that cannot be confirmed must not be
    /// described as one. Cells written earlier in the same call STAND.
    NotConfirmed {
        /// What was asked for.
        requested: Setting,
        /// What the lever actually resolves to now.
        effective: Option<Setting>,
    },
}

/// One refused lever request.
#[derive(Debug, Clone, PartialEq)]
pub struct Refusal {
    /// The lever the caller named.
    pub lever: String,
    /// Why it was refused.
    pub reason: RefusalReason,
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "lever `{}` refused: ", self.lever)?;
        match &self.reason {
            RefusalReason::NotDeclared => write!(
                f,
                "no such lever is declared in the ny registry (check the spelling against \
                 `ny_api::levers::catalog()`)"
            ),
            RefusalReason::WithheldByClass { class, moat, why } => write!(
                f,
                "it is classified {} (moat: {moat}) and this facade delivers only the \
                 safe_to_search subset — {why}",
                class.as_str()
            ),
            RefusalReason::NotWiredHere { preset_key } => write!(
                f,
                "it is safe to search and has the typed preset key `{preset_key}`, but this \
                 facade wires no verifiable in-process writer for it"
            ),
            RefusalReason::EnvironmentOnly => write!(
                f,
                "it has no typed preset key, and this facade never writes the process \
                 environment"
            ),
            RefusalReason::WrongShape { expected, supplied } => {
                write!(f, "it takes a {expected} value, not a {supplied}")
            }
            RefusalReason::OutsideDeclaration => write!(
                f,
                "the value is inadmissible for the lever's declared kind or range, so it \
                 would be discarded at resolution time and the shipped default would stand"
            ),
            RefusalReason::OverriddenByProcessEnvironment { inherited } => write!(
                f,
                "the process environment already sets it to {inherited:?}, and an environment \
                 value wins over a typed preset — nothing was written"
            ),
            RefusalReason::AlreadyInstalled => write!(
                f,
                "levers were already installed in this process, and ny's readers latch their \
                 answer once, so a second installation could not take effect"
            ),
            RefusalReason::NotConfirmed {
                requested,
                effective,
            } => match effective {
                Some(effective) => write!(
                    f,
                    "the cell was written with {requested} but the lever resolves to \
                     {effective}"
                ),
                None => write!(
                    f,
                    "the cell was written with {requested} but the lever resolves to nothing"
                ),
            },
        }
    }
}

impl std::error::Error for Refusal {}

fn refusal(lever: &str, reason: RefusalReason) -> Refusal {
    Refusal {
        lever: lever.to_owned(),
        reason,
    }
}

// ---------------------------------------------------------------------------
// The plan, and its one-shot installation
// ---------------------------------------------------------------------------

/// A set of `SafeToSearch` lever values, as DATA.
///
/// Building a plan validates each entry immediately — class, value shape, and
/// the declaration's own kind/range rule — so an inadmissible request is a
/// [`Refusal`] at the call site rather than a value silently discarded during a
/// run. Building a plan changes nothing; [`install`] is what delivers it.
#[derive(Debug, Clone, Default)]
pub struct LeverPlan {
    entries: BTreeMap<&'static str, Setting>,
}

impl LeverPlan {
    /// An empty plan.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one lever value, replacing any previous value for that lever.
    ///
    /// # Errors
    ///
    /// [`Refusal`] when the lever is undeclared, is not `SafeToSearch`, is not
    /// wired here, or when the value does not fit the declaration.
    pub fn set(&mut self, name: &str, value: Setting) -> Result<&mut Self, Refusal> {
        let Some(decl) = ny_levers::all().get(name) else {
            return Err(refusal(name, RefusalReason::NotDeclared));
        };
        let Some(wired) = wired(decl.name) else {
            // Not wired: say which of the three reasons applies, using the
            // search space rather than a second hand-written list.
            let axis = axis_for(decl.name);
            let class = axis.map_or(SearchClass::NotAnAxis, |axis| {
                SearchClass::from_axis(axis.class)
            });
            let preset_key = axis.and_then(|axis| match axis.deliver {
                Deliver::PresetKey(key) => Some(key),
                Deliver::EnvOnly => None,
            });
            let reason = match (class, preset_key) {
                (SearchClass::SafeToSearch, Some(preset_key)) => {
                    RefusalReason::NotWiredHere { preset_key }
                }
                (SearchClass::Unsafe | SearchClass::VerdictAffecting, _) => {
                    RefusalReason::WithheldByClass {
                        class,
                        moat: decl.moat.as_str(),
                        why: axis.map_or("no rationale recorded", |axis| axis.why),
                    }
                }
                _ => RefusalReason::EnvironmentOnly,
            };
            return Err(refusal(name, reason));
        };

        let Some(config) = wired.port.as_config(&value) else {
            return Err(refusal(
                name,
                RefusalReason::WrongShape {
                    expected: wired.port.shape(),
                    supplied: value.shape(),
                },
            ));
        };
        // The declaration is the authority on admissibility. Ask it with an
        // EMPTY environment lookup: this is a validity question about the
        // value, not a question about what this process inherited.
        if ny_levers::read_over_config_with(decl, |_| None, Some(config)).is_err() {
            return Err(refusal(name, RefusalReason::OutsideDeclaration));
        }

        self.entries.insert(wired.decl.name, value);
        Ok(self)
    }

    /// The levers this plan would deliver, name-sorted.
    #[must_use]
    pub fn levers(&self) -> Vec<&'static str> {
        self.entries.keys().copied().collect()
    }

    /// How many levers the plan carries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the plan carries nothing.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// What [`install`] delivered.
#[derive(Debug, Clone, PartialEq)]
pub struct Installed {
    entries: Vec<(&'static str, Setting)>,
}

impl Installed {
    /// Each delivered lever and the value it resolves to, name-sorted. Read
    /// back from the live resolver, not echoed from the request.
    #[must_use]
    pub fn effective(&self) -> &[(&'static str, Setting)] {
        &self.entries
    }

    /// Whether anything was delivered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

static INSTALLED: OnceLock<()> = OnceLock::new();

/// Deliver a plan to this process, once.
///
/// Call this during setup, BEFORE the first verification — several readers
/// latch their answer in a process-wide `OnceLock` and cannot see a later
/// write. An empty plan is a no-op and does not consume the one-shot.
///
/// The sequence is deliberately fail-closed:
///
/// 1. every lever in the plan is checked against the PROCESS ENVIRONMENT
///    first; if the environment names any of them, nothing is written and the
///    call is refused, because an inherited value would win and the caller
///    would be told it had configured something it had not;
/// 2. the one-shot is claimed, so a concurrent second call is refused rather
///    than interleaved;
/// 3. the typed cells are written;
/// 4. each lever is READ BACK through the live resolver, and the returned
///    [`Installed`] reports what the run will actually use.
///
/// # Errors
///
/// [`Refusal`] with [`RefusalReason::OverriddenByProcessEnvironment`],
/// [`RefusalReason::AlreadyInstalled`], or [`RefusalReason::NotConfirmed`].
pub fn install(plan: &LeverPlan) -> Result<Installed, Refusal> {
    if plan.is_empty() {
        return Ok(Installed {
            entries: Vec::new(),
        });
    }

    for name in plan.entries.keys() {
        let Some(wired) = wired(name) else {
            continue;
        };
        if let Some(inherited) = ny_levers::read_raw_os(wired.decl) {
            return Err(Refusal {
                lever: (*name).to_owned(),
                reason: RefusalReason::OverriddenByProcessEnvironment {
                    inherited: inherited.to_string_lossy().into_owned(),
                },
            });
        }
    }

    if INSTALLED.set(()).is_err() {
        return Err(Refusal {
            lever: plan
                .entries
                .keys()
                .next()
                .map_or_else(String::new, |name| (*name).to_owned()),
            reason: RefusalReason::AlreadyInstalled,
        });
    }

    let mut entries = Vec::with_capacity(plan.entries.len());
    for (name, requested) in &plan.entries {
        let Some(wired) = wired(name) else {
            continue;
        };
        wired.port.write(requested);
        let effective = wired.port.read();
        if effective.as_ref() != Some(requested) {
            return Err(Refusal {
                lever: (*name).to_owned(),
                reason: RefusalReason::NotConfirmed {
                    requested: requested.clone(),
                    effective,
                },
            });
        }
        entries.push((*name, requested.clone()));
    }
    Ok(Installed { entries })
}

/// Whether [`install`] has already delivered a non-empty plan in this process.
#[must_use]
pub fn installed() -> bool {
    INSTALLED.get().is_some()
}

// ---------------------------------------------------------------------------
// Audit: what this process is inheriting
// ---------------------------------------------------------------------------

/// One lever whose value in THIS process does not come from the shipped
/// default, because the process environment supplied something.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AmbientOverride {
    /// The environment-variable name.
    pub name: &'static str,
    /// The inherited raw value, rendered lossily when it is not UTF-8.
    pub raw: String,
    /// How `ny-levers` resolved it: `"legacy_env"` when the value was accepted,
    /// `"legacy_env_rejected"` when it was present but inadmissible (a typo
    /// silently resolving to the default), or `"default"`/`"config"` when an
    /// override-style parser deliberately ignored it.
    pub source: &'static str,
    /// True when the raw value was PRESENT but the parser refused it — a
    /// `NY_FOO=ture` that looks configured and is not.
    pub rejected: bool,
    /// The declaration's blast radius: `"none"`, `"low"`, `"high"`.
    pub moat: &'static str,
    /// Search classification.
    pub search_class: SearchClass,
    /// Whether a declared reader of this lever sits inside this facade's
    /// dependency closure. `false` means the inheritance is inert for an
    /// `ny-api` consumer.
    pub live_in_process: bool,
}

/// Every `NY_*` lever this process is inheriting from its ambient environment.
///
/// This is the answer to "what did I pick up without asking for it". It freezes
/// the whole registry against the live environment (one `getenv` per
/// declaration), so call it once, at session start, and record the result
/// alongside the verdict.
///
/// An entry with `rejected == true` is the sharpest signal: the variable was
/// SET, the parser refused it, and the run silently used the shipped default.
///
/// LOWER BOUND, not an inventory: only DECLARED levers can appear here. See the
/// module documentation for the 341 undeclared `NY_*` names ny-propagate reads
/// as raw literals, which no registry entry and therefore no receipt covers.
#[must_use]
pub fn ambient_overrides() -> Vec<AmbientOverride> {
    let packages = facade_packages();
    let set = ny_levers::LeverSet::freeze(ny_levers::all());
    let mut out: Vec<AmbientOverride> = set
        .entries()
        .iter()
        .filter(|resolved| {
            resolved.env_utf8.is_some()
                || resolved.rejected_raw.is_some()
                || resolved.ignored_raw.is_some()
        })
        .map(|resolved| {
            let decl = resolved.decl;
            let info = info_for(decl, &packages);
            let raw = resolved
                .rejected_raw
                .clone()
                .or_else(|| resolved.ignored_raw.clone())
                .unwrap_or_else(|| match resolved.env_utf8 {
                    Some(false) => "<non-utf8>".to_owned(),
                    _ => ny_levers::read_raw(decl).unwrap_or_default(),
                });
            AmbientOverride {
                name: decl.name,
                raw,
                source: resolved.source.as_str(),
                rejected: resolved.source == Source::LegacyEnvRejected
                    || resolved.rejected_raw.is_some(),
                moat: decl.moat.as_str(),
                search_class: info.search_class,
                live_in_process: info.live_in_process,
            }
        })
        .collect();
    out.sort_unstable_by_key(|entry| entry.name);
    out
}

/// The full `ny-levers` receipt for this process, as JSON text.
///
/// Every declared lever with its resolved value, the layer that supplied it,
/// and any raw input the parser rejected or ignored — the same
/// `ny-levers/receipt/v3` document `ny-cli` records into flight evidence.
/// Freeze it once at session start and attach it to whatever you publish; it is
/// the cheapest defence against every ambient-inheritance hazard in this
/// module's documentation.
#[must_use]
pub fn receipt_json() -> String {
    ny_levers::LeverSet::freeze(ny_levers::all())
        .receipt()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::{
        ambient_overrides, catalog, describe, effective, receipt_json, settable, Delivery,
        LeverPlan, Refusal, RefusalReason, SearchClass, Setting,
    };
    use ny_levers::space::{self, Class, Deliver};

    /// Discovery reports the REGISTRY, not a copy of it.
    #[test]
    fn discovery_returns_the_registrys_real_entries() {
        let catalog = catalog();
        assert_eq!(
            catalog.len(),
            ny_levers::all().len(),
            "the catalog must cover every declared lever, with no local list"
        );
        assert!(
            catalog.windows(2).all(|pair| pair[0].name < pair[1].name),
            "the catalog must be name-sorted and duplicate-free"
        );
        for info in &catalog {
            let decl = ny_levers::all()
                .get(info.name)
                .expect("every catalog name must be a declaration");
            assert_eq!(info.doc, decl.doc);
            assert_eq!(info.moat, decl.moat.as_str());
            assert_eq!(info.bucket, decl.bucket.as_str());
        }

        // A lever whose class is recorded in `space` must agree with `space`.
        for axis in space::axes().iter().chain(space::unsafe_axes()) {
            let Some(info) = describe(axis.name) else {
                continue;
            };
            let expected = match axis.class {
                Class::Unsafe => SearchClass::Unsafe,
                Class::VerdictAffecting => SearchClass::VerdictAffecting,
                Class::SafeToSearch => SearchClass::SafeToSearch,
            };
            assert_eq!(info.search_class, expected, "{}", axis.name);
            assert_eq!(info.why_classified, Some(axis.why), "{}", axis.name);
        }
    }

    /// The wired set may only ever grow inside `Class::SafeToSearch`. This is
    /// the ratchet: reclassifying one of these upstream breaks here rather
    /// than silently widening what a consumer may set.
    #[test]
    fn wired_levers_are_all_safe_to_search() {
        for name in settable() {
            let axis = space::axes()
                .iter()
                .find(|axis| axis.name == name)
                .unwrap_or_else(|| panic!("{name} must be a searchable axis"));
            assert_eq!(
                axis.class,
                Class::SafeToSearch,
                "{name} is wired for embedded delivery but is no longer SafeToSearch"
            );
            assert!(
                matches!(axis.deliver, Deliver::PresetKey(_)),
                "{name} is wired for embedded delivery but has no typed preset key"
            );
            assert!(
                matches!(
                    describe(name)
                        .expect("wired lever must be declared")
                        .delivery,
                    Delivery::Settable { .. }
                ),
                "{name} must report itself as Settable"
            );
        }
        assert!(
            !space::unsafe_axes()
                .iter()
                .any(|axis| settable().contains(&axis.name)),
            "no Class::Unsafe axis may ever be wired"
        );
    }

    /// An Unsafe axis and a VerdictAffecting axis are both refused BY NAME,
    /// and the refusal explains itself.
    #[test]
    fn unsafe_and_verdict_affecting_levers_are_refused_by_name() {
        let mut plan = LeverPlan::new();

        let refusal = plan
            .set("NY_CROWN_HONEST_PROVENANCE", Setting::Bool(false))
            .expect_err("an Unsafe lever must be refused");
        assert_eq!(refusal.lever, "NY_CROWN_HONEST_PROVENANCE");
        assert!(
            matches!(
                refusal.reason,
                RefusalReason::WithheldByClass {
                    class: SearchClass::Unsafe,
                    ..
                }
            ),
            "{refusal:?}"
        );
        assert!(refusal.to_string().contains("safe_to_search"));

        let verdict_affecting = space::axes()
            .iter()
            .find(|axis| axis.class == Class::VerdictAffecting)
            .expect("the search space must carry verdict-affecting axes");
        let refusal = plan
            .set(verdict_affecting.name, Setting::Bool(true))
            .expect_err("a VerdictAffecting lever must be refused");
        assert!(
            matches!(
                refusal.reason,
                RefusalReason::WithheldByClass {
                    class: SearchClass::VerdictAffecting,
                    ..
                }
            ),
            "{refusal:?}"
        );

        let refusal = plan
            .set("NY_NOT_A_REAL_LEVER", Setting::Bool(true))
            .expect_err("an undeclared name must be refused");
        assert_eq!(refusal.reason, RefusalReason::NotDeclared);

        assert!(plan.is_empty(), "a refused plan must stay empty");
    }

    /// Wrong shapes and out-of-range values are refused at plan time, not
    /// discarded silently at resolution time.
    #[test]
    fn a_plan_validates_shape_and_range_up_front() {
        let mut plan = LeverPlan::new();
        let refusal = plan
            .set("NY_PATCHES_FINITE_EXPIRY", Setting::Count(3))
            .expect_err("a Count for a Bool lever must be refused");
        assert!(
            matches!(
                refusal.reason,
                RefusalReason::WrongShape {
                    expected: "Bool",
                    supplied: "Count"
                }
            ),
            "{refusal:?}"
        );

        let refusal = plan
            .set("NY_MARGIN_ROW_BETA_LAMBDA", Setting::Real(f64::NAN))
            .expect_err("a non-finite fraction must be refused by the declaration");
        assert_eq!(refusal.reason, RefusalReason::OutsideDeclaration);
        assert!(plan.is_empty());
    }

    /// With nothing installed and nothing in the environment, every settable
    /// lever reads its DECLARATION DEFAULT. This is the property the whole
    /// promotion scheme rests on: an unset cell is byte-identical.
    #[test]
    fn defaults_are_unchanged_when_nothing_is_set() {
        assert!(
            !super::installed(),
            "the library test binary must not install levers — that belongs in \
             tests/levers_install_once.rs, which runs in its own process"
        );
        for name in settable() {
            let decl = ny_levers::all().get(name).expect("declared");
            if ny_levers::read_raw_os(decl).is_some() {
                // The ambient environment owns this lever in this process, so
                // the declaration default is not the expected answer — and that
                // inheritance is itself what `ambient_overrides` reports.
                continue;
            }
            // `Unset` declarations have no single rendering: the reader's own
            // fallback decides, and `install` is what confirms those.
            let expected = match ny_levers::read(decl).value {
                ny_levers::LeverValue::Bool(value) => Some(Setting::Bool(value)),
                ny_levers::LeverValue::U64(value) => {
                    usize::try_from(value).ok().map(Setting::Count)
                }
                ny_levers::LeverValue::F64(value) => Some(Setting::Real(value)),
                ny_levers::LeverValue::Text(value) => Some(Setting::Token(value)),
                ny_levers::LeverValue::Unset | ny_levers::LeverValue::Secs(_) => None,
            };
            if let Some(expected) = expected {
                assert_eq!(
                    effective(name),
                    Some(expected),
                    "{name} must read its declaration default"
                );
            }
        }
    }

    /// The audit surface answers, and answers about THIS process.
    #[test]
    fn the_ambient_audit_reports_the_live_environment() {
        let receipt = receipt_json();
        assert!(receipt.contains("ny-levers/receipt/v3"));
        assert!(receipt.contains("\"lever_count\":"));

        for entry in ambient_overrides() {
            assert!(
                ny_levers::all().get(entry.name).is_some(),
                "an ambient override must name a declared lever"
            );
        }
    }

    /// The note in the module documentation is measured, not asserted: this
    /// recomputes it from the registry and fails if the shape changes.
    #[test]
    fn the_ambient_inheritance_note_matches_the_registry() {
        let catalog = catalog();
        let live: Vec<_> = catalog.iter().filter(|info| info.live_in_process).collect();
        assert!(
            live.len() > catalog.len() / 3,
            "a large fraction of the registry is live for an ny-api consumer; got {} of {}",
            live.len(),
            catalog.len()
        );
        let live_unsafe: Vec<&str> = live
            .iter()
            .filter(|info| info.search_class == SearchClass::Unsafe)
            .map(|info| info.name)
            .collect();
        assert!(
            live_unsafe.contains(&"NY_CROWN_HONEST_PROVENANCE"),
            "the documented Unsafe ambient exposure must still be live: {live_unsafe:?}"
        );
        assert!(
            live.iter().any(|info| info.moat == "high"),
            "verdict-affecting levers are among the ambient-inherited set"
        );
    }

    /// A `Refusal` is a real error type and says something useful.
    #[test]
    fn a_refusal_renders_its_reason() {
        let refusal = Refusal {
            lever: "NY_X".to_owned(),
            reason: RefusalReason::EnvironmentOnly,
        };
        let rendered = refusal.to_string();
        assert!(rendered.contains("NY_X"), "{rendered}");
        assert!(
            rendered.contains("never writes the process environment"),
            "{rendered}"
        );
    }
}
