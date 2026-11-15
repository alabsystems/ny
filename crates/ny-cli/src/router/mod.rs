// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #router-r1 — the `ny-cli` half of the SHADOW router (ROUTER_PROGRAM §11).
//!
//! The policy (types, the pure plan and checkpoints, the progress bus,
//! `fail_open`) lives in `ny_core::router` and names no verdict. This module
//! is the glue the scored entry point calls:
//!
//! * [`RouterPolicy::from_levers`] — the ONE reader of `router.mode` /
//!   `optimizer.mode`, from the frozen, receipt-owning `LeverSet`.
//! * [`start`] — after the lever freeze: extracts static facts, asks every
//!   registered capability's own predicate, computes and logs the plan, and
//!   enables the progress bus. A no-op (no facts, no output) when both modes
//!   are `off`.
//! * [`finish_after_publication`] — called from `crate::flight::finish`, i.e.
//!   AFTER the verdict was published on every exit path that records one:
//!   drains the bus, evaluates c1/c2/c3 retrospectively, logs what the
//!   optimizer rules would have done, prints `[router] done` and writes the
//!   flight `router` record.
//! * [`replay`] — `ny vnncomp plan --replay`: re-runs the pure plan and
//!   checkpoints on a flight record and reports any divergence.
//!
//! `off` and `shadow` CHANGE NO VERDICT PATH. No value flows back into a
//! lane: the plan emits nothing, no cap or deadline moves, the pre-written
//! `unknown`, watchdogs and reserves are untouched, and every entry runs
//! inside `fail_open` (a panic is contained and disables the router for the
//! rest of the row).
//!
//! #router-r2: `on` ACTS (effective `acting`) through the seven rules of
//! acting set 1. [`start`] commits them purely
//! (`ny_core::router::acting::commit`) and [`acting::arm`] arms the ONE
//! deadline authority; the lanes then consult it at their own read sites
//! through the helpers in [`acting`], each of which returns the lane's own
//! value unchanged whenever the authority does not govern it. The session's
//! close ([`finish_after_publication`]) snapshots the authority into the
//! flight record and DISARMS it.
//!
//! #router-r3: `optimizer.mode: on` ACTS (effective `acting`) through the four
//! O-STOP rules of acting set 2. [`start`] commits them purely
//! (`ny_core::router::ostop::commit`, which reads R2's commitments for the
//! rules that ride on CR-1) and [`ostop::arm`] arms the stop authority the
//! phases read; [`finish_after_publication`] closes every open defer,
//! snapshots it into the flight record and DISARMS it.
//!
//! #router-r4: `router.mode: on` also ACTS through acting set 3. [`start`]
//! commits it purely (`ny_core::router::r4::commit`, which reads R2's
//! commitments for CR-F's cells), [`lease::arm`] arms the memory lease
//! authority the optional-heavy gates consult (CR-L), and [`far::arm`] arms
//! the FAR authority the source lanes consult at their own safe points
//! (CR-F); [`finish_after_publication`] snapshots both into the flight record
//! and DISARMS them. Safe points are enforced at every acting site that ends
//! or shortens a lane (`ny_core::router::safepoint`).
//!
//! #router-r5: `router.mode: on` also ACTS through acting set 4. [`start`]
//! commits it purely (`ny_core::router::route::commit`, reading the typed
//! keys the row pinned, [`StartContext::pinned_keys`]) and [`route::arm`]
//! arms the route authority: the capabilities CR-RS routed (only at
//! SR.seal — none today) and whether CR-G's gap-order contract funds
//! multi-objective classes. The routed capabilities are configured at their
//! own boundaries ([`route`]); [`finish_after_publication`] snapshots the
//! authority into the flight record and DISARMS it.
//!
//! #router-r6: `optimizer.mode: on` also ACTS through the four O-TRIAL rules.
//! [`start`] commits them purely (`ny_core::router::otrial::commit`, reading
//! the trial knobs the row pinned, [`StartContext::trial_pinned_keys`]) and
//! [`otrial::arm`] arms the trial authority; the lanes that own the knobs
//! build their own pure machines from it at their own safe points;
//! [`finish_after_publication`] snapshots the authority into the flight
//! record and DISARMS it.
//!
//! #admission-census: [`census`] is `ny vnncomp census` — for every row of a
//! corpus, CR-RS's own decision for each routable capability as if it were
//! sealed (the rows the seal pipeline must moat), from the facts
//! [`policy_input`] builds for the scored entry point. It arms nothing.
//!
//! #router-r7: `optimizer.priors: true` (default `false`) makes [`start`] read
//! the COMPILED offline cost-model prior (`ny_core::router::priors`): the
//! row's prior view joins the policy input (hashed, logged as `[router]
//! priors`, replayed), the bound capabilities get a priced first rung, and
//! the pure plan reads it for CR-C contracts (and, through the plan, R2's
//! CR-A keeps a slot whose priced rung does not fit on today's deadline).
//! A table that fails to load or validate prices nothing (`[router] priors
//! unavailable`), and a host class other than the table's prices nothing.
//!
//! #admission-precision: the start context also carries the property's
//! archetype and routing shape ([`StartContext::spec`]), and four routable
//! consumers' preconditions name the site inside their parent lane they sit
//! at (`ny_core::router::decl::AdmitSite`), evaluated with the lanes' own
//! route checks (`ny_core::router::admission`) that the lanes call inline.

pub(crate) mod acting;
#[cfg(test)]
mod admission_tests;
pub(crate) mod census;
#[cfg(test)]
mod census_tests;
mod facts;
pub(crate) mod far;
pub(crate) mod lease;
pub(crate) mod ostop;
pub(crate) mod otrial;
pub(crate) mod registry;
pub(crate) mod route;
pub(crate) mod telemetry;
#[cfg(test)]
mod tests;

use std::sync::{Mutex, PoisonError};
use std::time::{Duration, Instant};

use ny_core::router::acting::{self as acting_policy, Commitments, RuleSwitches};
use ny_core::router::bus::{self, ProgressSample, SampleRecord};
use ny_core::router::checkpoint::{self, Amendment, CheckpointId};
use ny_core::router::decl::ids;
use ny_core::router::facts::HostClass;
use ny_core::router::ostop::{self as ostop_policy, OptimizerSwitches};
use ny_core::router::otrial::{self as otrial_policy, R6Input, TrialSwitches};
use ny_core::router::plan::{self, BudgetView, CapInput, Plan, PolicyInput};
use ny_core::router::priors::PriorTable;
use ny_core::router::r4::{self as r4_policy, R4Switches};
use ny_core::router::route::{self as route_policy, R5Input, R5Switches};
use ny_core::router::{
    effective, effective_optimizer, guard, project, EffectiveMode, OptimizerMode, RouterMode,
    ACTING_OPTIMIZER_RULES, ACTING_RULES, PRIORS_VERSION, ROUTER_SCHEMA,
};

use telemetry::{CheckpointRecord, DoneRecord, OptimizerRecord, RouterRecord};

/// The frozen router policy of one run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RouterPolicy {
    /// `router.mode`.
    pub(crate) mode: RouterMode,
    /// `optimizer.mode`.
    pub(crate) optimizer: OptimizerMode,
    /// #router-r2: the seven `router.rules.*` ablation switches.
    pub(crate) rules: RuleSwitches,
    /// #router-r3: the four `optimizer.rules.*` ablation switches.
    pub(crate) optimizer_rules: OptimizerSwitches,
    /// #router-r4: the two acting-set-3 `router.rules.*` ablation switches.
    pub(crate) r4_rules: R4Switches,
    /// #router-r5: the two acting-set-4 `router.rules.*` ablation switches.
    pub(crate) r5_rules: R5Switches,
    /// #router-r6: the four O-TRIAL `optimizer.rules.trial_*` ablation
    /// switches.
    pub(crate) trial_rules: TrialSwitches,
    /// #router-r7: `optimizer.priors` — read the compiled offline prior.
    pub(crate) priors: bool,
}

impl RouterPolicy {
    /// Both dark.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) const OFF: Self = Self {
        mode: RouterMode::Off,
        optimizer: OptimizerMode::Off,
        rules: RuleSwitches::ALL_ON,
        optimizer_rules: OptimizerSwitches::ALL_ON,
        r4_rules: R4Switches::ALL_ON,
        r5_rules: R5Switches::ALL_ON,
        trial_rules: TrialSwitches::ALL_ON,
        priors: false,
    };

    /// Resolve both modes from the run's frozen, receipt-owning lever set —
    /// the same authority the flight receipt serialises. A missing or ill-typed
    /// entry is an error, never a silent fallback to a second authority.
    pub(crate) fn from_levers(levers: &ny_levers::LeverSet) -> Result<Self, String> {
        use ny_levers::decls::router::{OPTIMIZER_MODE, ROUTER_MODE};

        let token =
            |decl: &'static ny_levers::LeverDecl| -> Result<String, String> {
                let resolved = levers
                    .resolved(decl)
                    .ok_or_else(|| format!("{} is missing from the frozen LeverSet", decl.name))?;
                resolved.value.as_str().map(str::to_owned).ok_or_else(|| {
                    format!("{} resolved to non-text {:?}", decl.name, resolved.value)
                })
            };
        let mode_token = token(&ROUTER_MODE)?;
        let mode = RouterMode::from_token(&mode_token).ok_or_else(|| {
            format!(
                "{} is not one of {:?}: {mode_token:?}",
                ROUTER_MODE.name,
                RouterMode::TOKENS
            )
        })?;
        let optimizer_token = token(&OPTIMIZER_MODE)?;
        let optimizer = OptimizerMode::from_token(&optimizer_token).ok_or_else(|| {
            format!(
                "{} is not one of {:?}: {optimizer_token:?}",
                OPTIMIZER_MODE.name,
                OptimizerMode::TOKENS
            )
        })?;
        let rules = rule_switches(levers)?;
        let optimizer_rules = optimizer_switches(levers)?;
        let r4_rules = r4_switches(levers)?;
        let r5_rules = r5_switches(levers)?;
        let trial_rules = trial_switches(levers)?;
        let priors = priors_switch(levers)?;
        Ok(Self {
            mode,
            optimizer,
            rules,
            optimizer_rules,
            r4_rules,
            r5_rules,
            trial_rules,
            priors,
        })
    }

    /// Whether both modes are off (the router then does nothing at all).
    pub(crate) const fn is_off(self) -> bool {
        matches!(self.mode, RouterMode::Off) && matches!(self.optimizer, OptimizerMode::Off)
    }
}

/// #router-r2: the seven rule switches from the frozen LeverSet — the same
/// authority the receipt serialises. A missing or ill-typed entry is an
/// error, never a silent default.
fn rule_switches(levers: &ny_levers::LeverSet) -> Result<RuleSwitches, String> {
    use ny_levers::decls::router::{
        ROUTER_RULE_ADMISSION, ROUTER_RULE_CAP_AUTHORITY, ROUTER_RULE_CONTRACTS,
        ROUTER_RULE_DERIVED_CAPS, ROUTER_RULE_HAND_ON, ROUTER_RULE_PUBLISH_GRACE,
        ROUTER_RULE_TERMINAL,
    };
    let switch = |decl: &'static ny_levers::LeverDecl| -> Result<bool, String> {
        let resolved = levers
            .resolved(decl)
            .ok_or_else(|| format!("{} is missing from the frozen LeverSet", decl.name))?;
        match resolved.value {
            ny_levers::LeverValue::Bool(on) => Ok(on),
            ref other => Err(format!("{} resolved to non-boolean {other:?}", decl.name)),
        }
    };
    Ok(RuleSwitches {
        admission: switch(&ROUTER_RULE_ADMISSION)?,
        contracts: switch(&ROUTER_RULE_CONTRACTS)?,
        cap_authority: switch(&ROUTER_RULE_CAP_AUTHORITY)?,
        terminal: switch(&ROUTER_RULE_TERMINAL)?,
        hand_on: switch(&ROUTER_RULE_HAND_ON)?,
        publish_grace: switch(&ROUTER_RULE_PUBLISH_GRACE)?,
        derived_caps: switch(&ROUTER_RULE_DERIVED_CAPS)?,
    })
}

/// #router-r3: the four O-STOP rule switches from the frozen LeverSet — the
/// same authority the receipt serialises. A missing or ill-typed entry is an
/// error, never a silent default.
fn optimizer_switches(levers: &ny_levers::LeverSet) -> Result<OptimizerSwitches, String> {
    use ny_levers::decls::router::{
        OPTIMIZER_RULE_ATTACK_DEFER, OPTIMIZER_RULE_GOVERNOR, OPTIMIZER_RULE_IA_YIELD,
        OPTIMIZER_RULE_ROOT_ADMISSION,
    };
    let switch = |decl: &'static ny_levers::LeverDecl| -> Result<bool, String> {
        let resolved = levers
            .resolved(decl)
            .ok_or_else(|| format!("{} is missing from the frozen LeverSet", decl.name))?;
        match resolved.value {
            ny_levers::LeverValue::Bool(on) => Ok(on),
            ref other => Err(format!("{} resolved to non-boolean {other:?}", decl.name)),
        }
    };
    Ok(OptimizerSwitches {
        governor: switch(&OPTIMIZER_RULE_GOVERNOR)?,
        ia_yield: switch(&OPTIMIZER_RULE_IA_YIELD)?,
        attack_defer: switch(&OPTIMIZER_RULE_ATTACK_DEFER)?,
        root_admission: switch(&OPTIMIZER_RULE_ROOT_ADMISSION)?,
    })
}

/// #router-r4: the two acting-set-3 rule switches from the frozen LeverSet —
/// the same authority the receipt serialises. A missing or ill-typed entry is
/// an error, never a silent default.
fn r4_switches(levers: &ny_levers::LeverSet) -> Result<R4Switches, String> {
    use ny_levers::decls::router::{ROUTER_RULE_FAR_HAND_ON, ROUTER_RULE_LEASE};
    let switch = |decl: &'static ny_levers::LeverDecl| -> Result<bool, String> {
        let resolved = levers
            .resolved(decl)
            .ok_or_else(|| format!("{} is missing from the frozen LeverSet", decl.name))?;
        match resolved.value {
            ny_levers::LeverValue::Bool(on) => Ok(on),
            ref other => Err(format!("{} resolved to non-boolean {other:?}", decl.name)),
        }
    };
    Ok(R4Switches {
        lease: switch(&ROUTER_RULE_LEASE)?,
        far_hand_on: switch(&ROUTER_RULE_FAR_HAND_ON)?,
    })
}

/// #router-r5: the two acting-set-4 rule switches from the frozen LeverSet —
/// the same authority the receipt serialises. A missing or ill-typed entry is
/// an error, never a silent default.
fn r5_switches(levers: &ny_levers::LeverSet) -> Result<R5Switches, String> {
    use ny_levers::decls::router::{ROUTER_RULE_GAP_CONTRACT, ROUTER_RULE_ROUTE};
    let switch = |decl: &'static ny_levers::LeverDecl| -> Result<bool, String> {
        let resolved = levers
            .resolved(decl)
            .ok_or_else(|| format!("{} is missing from the frozen LeverSet", decl.name))?;
        match resolved.value {
            ny_levers::LeverValue::Bool(on) => Ok(on),
            ref other => Err(format!("{} resolved to non-boolean {other:?}", decl.name)),
        }
    };
    Ok(R5Switches {
        route: switch(&ROUTER_RULE_ROUTE)?,
        gap_contract: switch(&ROUTER_RULE_GAP_CONTRACT)?,
    })
}

/// #router-r6: the four O-TRIAL rule switches from the frozen LeverSet — the
/// same authority the receipt serialises. A missing or ill-typed entry is an
/// error, never a silent default.
fn trial_switches(levers: &ny_levers::LeverSet) -> Result<TrialSwitches, String> {
    use ny_levers::decls::router::{
        OPTIMIZER_RULE_TRIAL_ATTACK_HALVING, OPTIMIZER_RULE_TRIAL_LR_BACKOFF,
        OPTIMIZER_RULE_TRIAL_SPLIT_RESTART, OPTIMIZER_RULE_TRIAL_WIDTH,
    };
    let switch = |decl: &'static ny_levers::LeverDecl| -> Result<bool, String> {
        let resolved = levers
            .resolved(decl)
            .ok_or_else(|| format!("{} is missing from the frozen LeverSet", decl.name))?;
        match resolved.value {
            ny_levers::LeverValue::Bool(on) => Ok(on),
            ref other => Err(format!("{} resolved to non-boolean {other:?}", decl.name)),
        }
    };
    Ok(TrialSwitches {
        width: switch(&OPTIMIZER_RULE_TRIAL_WIDTH)?,
        lr_backoff: switch(&OPTIMIZER_RULE_TRIAL_LR_BACKOFF)?,
        split_restart: switch(&OPTIMIZER_RULE_TRIAL_SPLIT_RESTART)?,
        attack_halving: switch(&OPTIMIZER_RULE_TRIAL_ATTACK_HALVING)?,
    })
}

/// #router-r7: `optimizer.priors` from the frozen LeverSet — the same
/// authority the receipt serialises. A missing or ill-typed entry is an
/// error, never a silent default.
fn priors_switch(levers: &ny_levers::LeverSet) -> Result<bool, String> {
    use ny_levers::decls::router::OPTIMIZER_PRIORS;
    let resolved = levers.resolved(&OPTIMIZER_PRIORS).ok_or_else(|| {
        format!(
            "{} is missing from the frozen LeverSet",
            OPTIMIZER_PRIORS.name
        )
    })?;
    match resolved.value {
        ny_levers::LeverValue::Bool(on) => Ok(on),
        ref other => Err(format!(
            "{} resolved to non-boolean {other:?}",
            OPTIMIZER_PRIORS.name
        )),
    }
}

/// Whether `decl` is one of the router's own selection keys (its two modes,
/// the seven router rule switches and, since R3, the four optimizer rule
/// switches): they choose what the router does, not what it plans, so they
/// are not pins of the plan.
fn is_router_selector(decl: &'static ny_levers::LeverDecl) -> bool {
    use ny_levers::decls::router::{
        OPTIMIZER_MODE, OPTIMIZER_PRIORS, OPTIMIZER_RULE_ATTACK_DEFER, OPTIMIZER_RULE_GOVERNOR,
        OPTIMIZER_RULE_IA_YIELD, OPTIMIZER_RULE_ROOT_ADMISSION,
        OPTIMIZER_RULE_TRIAL_ATTACK_HALVING, OPTIMIZER_RULE_TRIAL_LR_BACKOFF,
        OPTIMIZER_RULE_TRIAL_SPLIT_RESTART, OPTIMIZER_RULE_TRIAL_WIDTH, ROUTER_MODE,
        ROUTER_RULE_ADMISSION, ROUTER_RULE_CAP_AUTHORITY, ROUTER_RULE_CONTRACTS,
        ROUTER_RULE_DERIVED_CAPS, ROUTER_RULE_FAR_HAND_ON, ROUTER_RULE_GAP_CONTRACT,
        ROUTER_RULE_HAND_ON, ROUTER_RULE_LEASE, ROUTER_RULE_PUBLISH_GRACE, ROUTER_RULE_ROUTE,
        ROUTER_RULE_TERMINAL,
    };
    [
        &ROUTER_MODE,
        &OPTIMIZER_MODE,
        // #router-r7: which prior the router READS is a selector, not a pin.
        &OPTIMIZER_PRIORS,
        &ROUTER_RULE_ADMISSION,
        &ROUTER_RULE_CONTRACTS,
        &ROUTER_RULE_CAP_AUTHORITY,
        &ROUTER_RULE_TERMINAL,
        &ROUTER_RULE_HAND_ON,
        &ROUTER_RULE_PUBLISH_GRACE,
        &ROUTER_RULE_DERIVED_CAPS,
        &OPTIMIZER_RULE_GOVERNOR,
        &OPTIMIZER_RULE_IA_YIELD,
        &OPTIMIZER_RULE_ATTACK_DEFER,
        &OPTIMIZER_RULE_ROOT_ADMISSION,
        &ROUTER_RULE_LEASE,
        &ROUTER_RULE_FAR_HAND_ON,
        &ROUTER_RULE_ROUTE,
        &ROUTER_RULE_GAP_CONTRACT,
        &OPTIMIZER_RULE_TRIAL_WIDTH,
        &OPTIMIZER_RULE_TRIAL_LR_BACKOFF,
        &OPTIMIZER_RULE_TRIAL_SPLIT_RESTART,
        &OPTIMIZER_RULE_TRIAL_ATTACK_HALVING,
    ]
    .into_iter()
    .any(|candidate| std::ptr::eq(candidate, decl))
}

/// Whether `levers` ask for the router at all: the entry point builds a
/// [`StartContext`] (and stats nothing) only when this is true, so an `off`
/// row pays one lever lookup and nothing else.
pub(crate) fn requested(levers: &ny_levers::LeverSet) -> bool {
    RouterPolicy::from_levers(levers).is_ok_and(|policy| !policy.is_off())
}

/// Where the policy came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Authority {
    /// The frozen, receipt-owning `LeverSet`.
    Frozen,
    /// The preset was invalid, so no typed layer exists; the modes come from
    /// the entry-time environment layer the flight recorder captured.
    EntryEnv,
}

impl Authority {
    const fn token(self) -> &'static str {
        match self {
            Self::Frozen => "frozen",
            Self::EntryEnv => "entry_env",
        }
    }
}

/// What the entry point hands the router at the lever freeze. Facts only:
/// never a category, a model path or a property path.
pub(crate) struct StartContext<'a> {
    /// The lever set the policy is read from (`None` = no authority: off).
    pub(crate) levers: Option<&'a ny_levers::LeverSet>,
    /// Which authority that set is.
    pub(crate) authority: Authority,
    /// The plan resolver's existing model scan (`None` = unavailable).
    pub(crate) model: Option<&'a crate::plan_resolver::ModelFacts>,
    /// The model's on-disk size (explains an unavailable scan).
    pub(crate) model_bytes: Option<u64>,
    /// Why no scan exists when the route never runs one (`None` = derive the
    /// reason from `model_bytes`).
    pub(crate) facts_reason: Option<&'static str>,
    /// Networks the property declares (the relational route's own count).
    pub(crate) declared_networks: Option<u64>,
    /// `RelationalRouteDecision::selected`, as the entry point computed it.
    pub(crate) relational_selected: Option<bool>,
    /// The scored budget.
    pub(crate) official_budget_secs: u64,
    /// The internal verifier tier.
    pub(crate) internal_secs: u64,
    /// The concurrent-lane join margin.
    pub(crate) join_margin: Duration,
    /// The command's start (the bus clock's anchor).
    pub(crate) anchor: Instant,
    /// A declared target host (measurement only; `None` on scored runs).
    pub(crate) target_host: Option<HostClass>,
    /// #router-r5: the typed key paths of every registered route that the
    /// row pinned (its own file, or a non-default lever source;
    /// [`route::pinned_keys`]). A route never overrides a pin.
    pub(crate) pinned_keys: Vec<String>,
    /// #router-r6: the trial knobs the row pinned (its own file, or a
    /// non-default lever source; [`otrial::pinned_keys`]). A pinned knob is
    /// frozen: its trial site is not armed.
    pub(crate) trial_pinned_keys: Vec<String>,
    /// #admission-precision: the property's archetype and routing shape, read
    /// by the entry point with the lanes' parser and the one extractor the
    /// lanes route on (`beta_crown::spec_shape::read_spec`); `Unknown` when
    /// the route reads no property at start.
    pub(crate) spec: crate::commands::beta_crown::spec_shape::SpecRead,
}

/// One row's live router session.
struct RouterSession {
    record: RouterRecord,
    effective: EffectiveMode,
    optimizer: EffectiveMode,
}

static SESSION: Mutex<Option<RouterSession>> = Mutex::new(None);

fn session_slot() -> std::sync::MutexGuard<'static, Option<RouterSession>> {
    SESSION.lock().unwrap_or_else(PoisonError::into_inner)
}

fn ms(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

/// Typed keys the run pinned (every lever resolved from the CONFIG layer),
/// sorted; the router's own selectors (its two modes and, since R2, its seven
/// rule switches) are excluded so enabling the router by yaml or by
/// environment plans identically.
fn pins(levers: Option<&ny_levers::LeverSet>) -> Vec<String> {
    let Some(levers) = levers else {
        return Vec::new();
    };
    let mut out: Vec<String> = levers
        .entries()
        .iter()
        .filter(|entry| entry.source == ny_levers::Source::Config)
        .filter(|entry| !is_router_selector(entry.decl))
        .map(|entry| entry.decl.name.to_owned())
        .collect();
    out.sort();
    out.dedup();
    out
}

/// The static policy input for a row (pure glue: facts, host, budget, every
/// registered capability with its own predicate's answer). No prior is read:
/// [`policy_input_with`] with `priors = false`.
pub(crate) fn policy_input(ctx: &StartContext<'_>) -> PolicyInput {
    policy_input_with(ctx, false)
}

/// #router-r7: the compiled prior, when it loads and validates. `Err` is a
/// table that cannot be read: the session prices nothing and says so.
fn compiled_prior() -> Result<PriorTable, String> {
    PriorTable::compiled().map_err(|error| error.to_string())
}

/// The static policy input, reading the compiled offline prior when `priors`
/// (#router-r7). Without it the input is R5's, byte for byte: no capability
/// is priced, `priors_version` is `uniform-v1` and `priors` is `None`.
pub(crate) fn policy_input_with(ctx: &StartContext<'_>, priors: bool) -> PolicyInput {
    let facts = facts::instance_facts(
        ctx.model,
        ctx.model_bytes,
        ctx.declared_networks,
        ctx.facts_reason,
        &ctx.spec,
    );
    let host = facts::host_envelope(ctx.target_host);
    let admission_ctx = registry::Context {
        facts: &facts,
        relational_selected: ctx.relational_selected,
        levers: ctx.levers,
    };
    // #router-r7: the prior's view of THIS row (static: compiled table x
    // static facts x the planning host). A table that cannot be read prices
    // nothing — today's caps, exactly as without the key.
    let view = if priors {
        compiled_prior().ok().map(|table| table.view(&facts, &host))
    } else {
        None
    };
    let capabilities: Vec<CapInput> = registry::REGISTRY
        .iter()
        .map(|decl| {
            let rung = view.as_ref().and_then(|v| v.rung_ms(decl.id.as_str()));
            CapInput::from_decl(decl, registry::admit(decl, &admission_ctx), rung)
        })
        .collect();
    let priors_version = view
        .as_ref()
        .map_or_else(|| PRIORS_VERSION.to_owned(), |v| v.version.clone());
    PolicyInput {
        schema: ROUTER_SCHEMA.to_owned(),
        facts,
        host,
        budget: BudgetView {
            official_ms: ctx.official_budget_secs.saturating_mul(1000),
            internal_ms: ctx.internal_secs.saturating_mul(1000),
            join_margin_ms: ms(ctx.join_margin),
        },
        capabilities,
        pins: pins(ctx.levers),
        priors_version,
        registry_version: registry::registry_version(),
        priors: view,
    }
}

fn record_for(
    policy: RouterPolicy,
    effective: EffectiveMode,
    optimizer: EffectiveMode,
    authority: Authority,
    started_ms: u64,
    input: PolicyInput,
) -> RouterRecord {
    let plan = plan::plan(&input);
    RouterRecord {
        schema: ROUTER_SCHEMA.to_owned(),
        mode: policy.mode.token().to_owned(),
        effective: effective.token().to_owned(),
        optimizer: policy.optimizer.token().to_owned(),
        optimizer_effective: optimizer.token().to_owned(),
        authority: authority.token().to_owned(),
        started_ms,
        input,
        plan,
        switches: None,
        commitments: None,
        acting: None,
        optimizer_switches: None,
        stop_commitments: None,
        stop: None,
        r4_switches: None,
        r4_commitments: None,
        lease: None,
        far: None,
        r5_switches: None,
        r5_input: None,
        r5_commitments: None,
        route: None,
        trial_switches: None,
        r6_input: None,
        trial_commitments: None,
        trial: None,
        checkpoints: Vec::new(),
        optimizer_rules: Vec::new(),
        done: None,
    }
}

/// Start the row's router session. Does NOTHING — no facts, no plan, no bus,
/// no output — when both modes are off or no authority resolved.
pub(crate) fn start(ctx: &StartContext<'_>) {
    let Some(levers) = ctx.levers else {
        return;
    };
    let policy = match RouterPolicy::from_levers(levers) {
        Ok(policy) => policy,
        Err(error) => {
            // A declaration/registry mismatch: a programming error. The row
            // runs exactly today's route.
            eprintln!("[router] policy unresolved ({error}); router off for this row");
            return;
        }
    };
    if policy.is_off() {
        return;
    }
    let effective_mode = effective(policy.mode, ACTING_RULES);
    let optimizer_mode = effective_optimizer(policy.optimizer, ACTING_OPTIMIZER_RULES);
    let started_ms = ms(ctx.anchor.elapsed());
    let built = guard::fail_open("start", || {
        let input = policy_input_with(ctx, policy.priors);
        record_for(
            policy,
            effective_mode,
            optimizer_mode,
            ctx.authority,
            started_ms,
            input,
        )
    });
    let record = match built {
        Ok(record) => record,
        Err(fault) => {
            eprintln!("{}", telemetry::fault_line(&fault));
            return;
        }
    };
    let printed = guard::fail_open("telemetry", || {
        let mut lines = vec![telemetry::banner_line(
            policy,
            effective_mode,
            optimizer_mode,
            record.authority.as_str(),
            &record.input,
        )];
        lines.extend(record.input.capabilities.iter().map(telemetry::cap_line));
        // #router-r7: what the offline prior said, when the key asked for it.
        if policy.priors {
            lines.push(match &record.input.priors {
                Some(view) => telemetry::priors_line(view),
                None => telemetry::priors_unavailable_line(
                    &compiled_prior()
                        .err()
                        .unwrap_or_else(|| "unread".to_owned()),
                ),
            });
        }
        lines.push(telemetry::plan_line(&record.plan));
        lines
    });
    match printed {
        Ok(lines) => {
            for line in lines {
                eprintln!("{line}");
            }
        }
        Err(fault) => {
            eprintln!("{}", telemetry::fault_line(&fault));
            return;
        }
    }
    // #router-r2: the ACTING half. Only `on` (effective `acting`) commits the
    // seven rules and arms the deadline authority; `off`/`shadow` never reach
    // this block, so they stay exactly R1.
    let mut record = record;
    if effective_mode == EffectiveMode::Acting {
        let committed = guard::fail_open("acting", || {
            let commitments =
                acting_policy::commit(&record.plan, &record.input, effective_mode, &policy.rules);
            let lines = acting::arm(&commitments, ctx.join_margin);
            (commitments, lines)
        });
        match committed {
            Ok((commitments, lines)) => {
                for line in lines {
                    eprintln!("{line}");
                }
                crate::flight::note(
                    "router_acting",
                    crate::flight::FlightStatus::Ran,
                    Some(format!(
                        "sha={} armed={} rules_acting={}",
                        commitments.sha8(),
                        u8::from(ny_core::deadline_authority::global().is_armed()),
                        commitments.rules.iter().filter(|r| r.acts).count()
                    )),
                );
                record.switches = Some(policy.rules);
                record.commitments = Some(commitments);
            }
            Err(fault) => eprintln!("{}", telemetry::fault_line(&fault)),
        }
        // #router-r4: acting set 3 — CR-L (the memory lease) and CR-F (the
        // FAR hand-on) — rides on the same acting session and reads R2's
        // commitments for CR-F's cells. A fault above leaves no R2
        // commitments, and then R4 does not arm either.
        if let Some(r2) = record.commitments.clone() {
            let committed = guard::fail_open("r4", || {
                let commitments = r4_policy::commit(
                    &record.plan,
                    &record.input,
                    effective_mode,
                    &policy.r4_rules,
                    &r2,
                );
                let mut lines = acting::act_lines_r4(&commitments);
                lines.extend(lease::arm(commitments.lease.as_ref()));
                lines.extend(far::arm(&record.plan, commitments.far.as_ref(), ctx.anchor));
                (commitments, lines)
            });
            match committed {
                Ok((commitments, lines)) => {
                    for line in lines {
                        eprintln!("{line}");
                    }
                    crate::flight::note(
                        "router_r4",
                        crate::flight::FlightStatus::Ran,
                        Some(format!(
                            "sha={} lease_armed={} far_armed={} rules_acting={}",
                            commitments.sha8(),
                            u8::from(ny_core::memory_lease::global().is_armed()),
                            u8::from(ny_core::far_authority::global().is_armed()),
                            commitments.rules.iter().filter(|r| r.acts).count()
                        )),
                    );
                    record.r4_switches = Some(policy.r4_rules);
                    record.r4_commitments = Some(commitments);
                }
                Err(fault) => eprintln!("{}", telemetry::fault_line(&fault)),
            }
        }
        // #router-r5: acting set 4 — CR-RS (capabilities routed only at
        // SR.seal) and CR-G (gap-order contracts) — rides on the same acting
        // session and reads no other rule's commitments. A fault anywhere
        // above disabled the router, and this block is then skipped too
        // (`fail_open`); a fault here leaves the authority unarmed.
        let r5_input = R5Input {
            pinned_keys: ctx.pinned_keys.clone(),
        };
        let committed = guard::fail_open("r5", || {
            let commitments =
                route_policy::commit(&record.input, &r5_input, effective_mode, &policy.r5_rules);
            let lines = route::arm(&commitments);
            (commitments, lines)
        });
        match committed {
            Ok((commitments, lines)) => {
                for line in lines {
                    eprintln!("{line}");
                }
                crate::flight::note(
                    "router_r5",
                    crate::flight::FlightStatus::Ran,
                    Some(format!(
                        "sha={} route_armed={} routed={} pin_debt={} rules_acting={}",
                        commitments.sha8(),
                        u8::from(ny_core::route_authority::global().is_armed()),
                        commitments.routed().count(),
                        commitments.pin_debt().count(),
                        commitments.rules.iter().filter(|r| r.acts).count()
                    )),
                );
                record.r5_switches = Some(policy.r5_rules);
                record.r5_input = Some(r5_input);
                record.r5_commitments = Some(commitments);
            }
            Err(fault) => eprintln!("{}", telemetry::fault_line(&fault)),
        }
    }
    // #router-r3: the OPTIMIZER's acting half. Only `optimizer.mode: on`
    // (effective `acting`) commits the four O-STOP rules and arms the stop
    // authority; `off`/`shadow` never reach this block. The rules that ride
    // on CR-1 read R2's commitments (`None` when the router did not act).
    if optimizer_mode == EffectiveMode::Acting {
        let committed = guard::fail_open("ostop", || {
            let commitments = ostop_policy::commit(
                &record.plan,
                &record.input,
                optimizer_mode,
                &policy.optimizer_rules,
                record.commitments.as_ref(),
            );
            let lines = ostop::arm(&commitments);
            (commitments, lines)
        });
        match committed {
            Ok((commitments, lines)) => {
                for line in lines {
                    eprintln!("{line}");
                }
                crate::flight::note(
                    "router_optimizer",
                    crate::flight::FlightStatus::Ran,
                    Some(format!(
                        "sha={} armed={} rules_acting={}",
                        commitments.sha8(),
                        u8::from(ny_core::stop_authority::global().is_armed()),
                        commitments.rules.iter().filter(|r| r.acts).count()
                    )),
                );
                record.optimizer_switches = Some(policy.optimizer_rules);
                record.stop_commitments = Some(commitments);
            }
            Err(fault) => eprintln!("{}", telemetry::fault_line(&fault)),
        }
        // #router-r6: the O-TRIAL rules ride on the same acting optimizer
        // session and read no other rule's commitments. A fault anywhere
        // above disabled the router, and this block is then skipped too
        // (`fail_open`); a fault here leaves the trial authority unarmed.
        let r6_input = R6Input {
            pinned_keys: ctx.trial_pinned_keys.clone(),
        };
        let committed = guard::fail_open("otrial", || {
            let commitments = otrial_policy::commit(
                &record.plan,
                &record.input,
                &r6_input,
                optimizer_mode,
                &policy.trial_rules,
            );
            let lines = otrial::arm(&commitments);
            (commitments, lines)
        });
        match committed {
            Ok((commitments, lines)) => {
                for line in lines {
                    eprintln!("{line}");
                }
                crate::flight::note(
                    "router_trial",
                    crate::flight::FlightStatus::Ran,
                    Some(format!(
                        "sha={} armed={} sites={} rules_acting={}",
                        commitments.sha8(),
                        u8::from(ny_core::trial_authority::global().is_armed()),
                        commitments.sites.iter().filter(|s| s.armed).count(),
                        commitments.rules.iter().filter(|r| r.acts).count()
                    )),
                );
                record.trial_switches = Some(policy.trial_rules);
                record.r6_input = Some(r6_input);
                record.trial_commitments = Some(commitments);
            }
            Err(fault) => eprintln!("{}", telemetry::fault_line(&fault)),
        }
    }
    crate::flight::note(
        "router_plan",
        crate::flight::FlightStatus::Ran,
        Some(format!(
            "sha={} mode={} effective={} optimizer={} authority={}",
            record.plan.sha8(),
            record.mode,
            record.effective,
            record.optimizer_effective,
            record.authority
        )),
    );
    bus::enable(ctx.anchor);
    *session_slot() = Some(RouterSession {
        record,
        effective: effective_mode,
        optimizer: optimizer_mode,
    });
}

/// The samples one checkpoint reads, cut to the record's remaining room.
fn checkpoint_record(
    plan: &Plan,
    samples: &[ProgressSample],
    id: CheckpointId,
    room: &mut usize,
) -> CheckpointRecord {
    let at_ms = plan.checkpoint_at_ms(id).unwrap_or(0);
    let mut selected = checkpoint::select_samples(plan, samples, id);
    let truncated = selected.len() > *room;
    selected.truncate(*room);
    *room -= selected.len();
    CheckpointRecord {
        id,
        at_ms,
        samples: selected.iter().map(ProgressSample::record).collect(),
        truncated,
        observations: checkpoint::observe(plan, samples, id),
        amendments: checkpoint::checkpoint(plan, samples, id),
    }
}

/// `rule.governor` in shadow: where the margin-row alpha ascent's own yield
/// rule would have stopped it (gain < 0.15 of its best, after 2 samples).
fn governor_record(samples: &[ProgressSample]) -> OptimizerRecord {
    let alpha: Vec<&ProgressSample> = samples
        .iter()
        .filter(|s| s.cap == ids::MARGIN_ROW_ALPHA)
        .collect();
    let gains: Vec<f64> = alpha.iter().map(|s| s.margin_best()).collect();
    let at = project::governor_yield_index(&gains, project::GOVERNOR_RATIO, 2)
        .and_then(|index| alpha.get(index))
        .map(|s| s.unit_index);
    OptimizerRecord {
        rule: "rule.governor".to_owned(),
        site: ids::MARGIN_ROW_ALPHA.as_str().to_owned(),
        would: if at.is_some() { "stop" } else { "none" }.to_owned(),
        at,
        samples: alpha.len() as u64,
    }
}

/// The retrospective half: checkpoints, optimizer rules, and the done
/// summary, from the drained samples. Pure apart from its inputs.
fn retrospect(
    record: &mut RouterRecord,
    optimizer: EffectiveMode,
    samples: &[ProgressSample],
    verdict: &str,
    stop_reason: Option<String>,
) {
    let mut room = telemetry::MAX_RECORD_SAMPLES;
    record.checkpoints = CheckpointId::ALL
        .into_iter()
        .map(|id| checkpoint_record(&record.plan, samples, id, &mut room))
        .collect();
    if optimizer.observes() {
        record.optimizer_rules = vec![governor_record(samples)];
    }
    let decisions: Vec<(&CheckpointId, &Vec<Amendment>, Vec<&'static str>)> = record
        .checkpoints
        .iter()
        .map(|c| {
            (
                &c.id,
                &c.amendments,
                c.observations.iter().map(|o| o.class.token()).collect(),
            )
        })
        .collect();
    record.done = Some(DoneRecord {
        faults: guard::faults(),
        bus: bus::stats(),
        drained: samples.len() as u64,
        stop_reason: stop_reason.unwrap_or_else(|| "none-recorded".to_owned()),
        verdict: verdict.to_owned(),
        published_before_ckpt: true,
        decisions_sha: telemetry::digest_of(&decisions),
    });
}

/// Called from `crate::flight::finish`, AFTER the verdict was published: the
/// retrospective checkpoints, the optimizer's would-lines, `[router] done`
/// and the flight record. A no-op when no session started.
pub(crate) fn finish_after_publication(verdict: &str) {
    // #router-r2: close the acting half FIRST, whatever happened to the
    // shadow session: snapshot and DISARM the deadline authority so nothing
    // after publication can read a cell. #router-r3: likewise the stop
    // authority (every open defer is closed with its outcome first).
    let acting_record = acting::finish();
    let stop_record = ostop::finish(verdict);
    // #router-r4: and the lease and FAR authorities.
    let lease_record = lease::finish();
    let far_record = far::finish();
    // #router-r5: and the route authority.
    let route_record = route::finish();
    // #router-r6: and the trial authority.
    let trial_record = otrial::finish();
    let Some(mut session) = session_slot().take() else {
        return;
    };
    let stop_reason = crate::flight::global().stop_reason();
    let drained = guard::fail_open("drain", bus::drain);
    bus::disable();
    let samples = match drained {
        Ok(samples) => samples,
        Err(fault) => {
            eprintln!("{}", telemetry::fault_line(&fault));
            Vec::new()
        }
    };
    let optimizer = session.optimizer;
    let _ = session.effective;
    let outcome = guard::fail_open("checkpoint", || {
        retrospect(
            &mut session.record,
            optimizer,
            &samples,
            verdict,
            stop_reason.clone(),
        );
        let mut lines: Vec<String> = session
            .record
            .checkpoints
            .iter()
            .map(telemetry::checkpoint_line)
            .collect();
        lines.extend(
            session
                .record
                .optimizer_rules
                .iter()
                .map(telemetry::optimizer_line),
        );
        lines
    });
    match outcome {
        Ok(lines) => {
            for line in lines {
                eprintln!("{line}");
            }
        }
        Err(fault) => eprintln!("{}", telemetry::fault_line(&fault)),
    }
    crate::flight::note(
        "router_checkpoint",
        crate::flight::FlightStatus::Ran,
        Some(format!(
            "retrospective after publication: checkpoints={} samples={}",
            session.record.checkpoints.len(),
            samples.len()
        )),
    );
    // The done line prints even after a fault (with the fault counted).
    let done = session.record.done.clone().unwrap_or_else(|| DoneRecord {
        faults: guard::faults(),
        bus: bus::stats(),
        drained: samples.len() as u64,
        stop_reason: stop_reason.unwrap_or_else(|| "none-recorded".to_owned()),
        verdict: verdict.to_owned(),
        published_before_ckpt: true,
        decisions_sha: "-".to_owned(),
    });
    if let Some(acting_record) = acting_record {
        eprintln!("{}", acting::done_line(&acting_record));
        session.record.acting = Some(acting_record);
    }
    if let Some(stop_record) = stop_record {
        eprintln!("{}", ostop::done_line(&stop_record));
        session.record.stop = Some(stop_record);
    }
    if let Some(lease_record) = lease_record {
        eprintln!("{}", lease::done_line(&lease_record));
        session.record.lease = Some(lease_record);
    }
    if let Some((far_record, lines)) = far_record {
        for line in lines {
            eprintln!("{line}");
        }
        session.record.far = Some(far_record);
    }
    if let Some((route_record, lines)) = route_record {
        for line in lines {
            eprintln!("{line}");
        }
        session.record.route = Some(route_record);
    }
    if let Some(trial_record) = trial_record {
        eprintln!("{}", otrial::done_line(&trial_record));
        session.record.trial = Some(trial_record);
    }
    eprintln!("{}", telemetry::done_line(&done));
    crate::flight::note(
        "router_done",
        crate::flight::FlightStatus::Ran,
        Some(format!(
            "faults={} bus_pushed={} bus_dropped={} decisions_sha={}",
            done.faults, done.bus.pushed, done.bus.dropped, done.decisions_sha
        )),
    );
    session.record.done = Some(done);
    match serde_json::to_value(&session.record) {
        Ok(value) => crate::flight::global().note_router(value),
        Err(error) => eprintln!("[router] record serialization failed ({error})"),
    }
}

/// The result of a replay.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReplayReport {
    /// The replayed plan's `sha8`.
    pub(crate) plan_sha8: String,
    /// Checkpoints re-run.
    pub(crate) checkpoints: usize,
    /// Checkpoints skipped because their samples were truncated.
    pub(crate) skipped: usize,
}

/// Re-run the pure plan and checkpoints on a flight `router` record. Any
/// divergence — a different plan, different amendments, an unknown capability
/// or registry — is an error.
pub(crate) fn replay(record: &serde_json::Value) -> Result<ReplayReport, String> {
    let record: RouterRecord = serde_json::from_value(record.clone())
        .map_err(|error| format!("not a {ROUTER_SCHEMA} flight record: {error}"))?;
    if record.schema != ROUTER_SCHEMA {
        return Err(format!(
            "record schema {} is not {ROUTER_SCHEMA}",
            record.schema
        ));
    }
    let here = registry::registry_version();
    if record.input.registry_version != here {
        return Err(format!(
            "the record was made by registry {} and this binary carries {here}",
            record.input.registry_version
        ));
    }
    // #router-r7: a logged prior view re-derives from the compiled table on
    // the logged facts and host — a different table is a divergence, not a
    // silent re-plan.
    if let Some(logged) = &record.input.priors {
        let table = compiled_prior()?;
        let rederived = table.view(&record.input.facts, &record.input.host);
        if &rederived != logged {
            return Err(format!(
                "prior view diverged: recorded {} replayed {}",
                logged.version, rederived.version
            ));
        }
        let unpriced_or_equal = record
            .input
            .capabilities
            .iter()
            .all(|cap| cap.first_rung_ms == rederived.rung_ms(&cap.id));
        if !unpriced_or_equal || record.input.priors_version != rederived.version {
            return Err(
                "the logged first rungs or priors version do not match the prior view".to_owned(),
            );
        }
    }
    let replanned = plan::plan(&record.input);
    if replanned != record.plan {
        return Err(format!(
            "plan diverged: recorded sha={} replayed sha={}",
            record.plan.sha8(),
            replanned.sha8()
        ));
    }
    // #router-r2: the acting commitments are pure too.
    if let Some(logged) = &record.commitments {
        let switches = record
            .switches
            .ok_or_else(|| "the record carries commitments but no rule switches".to_owned())?;
        let mode = match record.effective.as_str() {
            "acting" => EffectiveMode::Acting,
            "shadow" => EffectiveMode::Shadow,
            _ => EffectiveMode::Off,
        };
        let recommitted = acting_policy::commit(&replanned, &record.input, mode, &switches);
        if &recommitted != logged {
            return Err(format!(
                "acting commitments diverged: recorded sha={} replayed sha={}",
                logged.sha8(),
                recommitted.sha8()
            ));
        }
    }
    // #router-r3: the O-STOP commitments are pure too (they read the logged
    // R2 commitments, which were just re-derived bit for bit).
    if let Some(logged) = &record.stop_commitments {
        let switches = record.optimizer_switches.ok_or_else(|| {
            "the record carries O-STOP commitments but no optimizer switches".to_owned()
        })?;
        let mode = match record.optimizer_effective.as_str() {
            "acting" => EffectiveMode::Acting,
            "shadow" => EffectiveMode::Shadow,
            _ => EffectiveMode::Off,
        };
        let recommitted = ostop_policy::commit(
            &replanned,
            &record.input,
            mode,
            &switches,
            record.commitments.as_ref(),
        );
        if &recommitted != logged {
            return Err(format!(
                "O-STOP commitments diverged: recorded sha={} replayed sha={}",
                logged.sha8(),
                recommitted.sha8()
            ));
        }
    }
    // #router-r4: acting set 3's commitments are pure too (they read the
    // logged R2 commitments, re-derived above), and every LIVE CR-F decision
    // re-derives from the samples the authority recorded with it.
    if let Some(logged) = &record.r4_commitments {
        let switches = record
            .r4_switches
            .ok_or_else(|| "the record carries R4 commitments but no R4 switches".to_owned())?;
        let r2 = record
            .commitments
            .as_ref()
            .ok_or_else(|| "the record carries R4 commitments but no R2 commitments".to_owned())?;
        let mode = match record.effective.as_str() {
            "acting" => EffectiveMode::Acting,
            "shadow" => EffectiveMode::Shadow,
            _ => EffectiveMode::Off,
        };
        let recommitted = r4_policy::commit(&replanned, &record.input, mode, &switches, r2);
        if &recommitted != logged {
            return Err(format!(
                "R4 commitments diverged: recorded sha={} replayed sha={}",
                logged.sha8(),
                recommitted.sha8()
            ));
        }
    }
    // #router-r5: acting set 4's commitments are pure too (static input plus
    // the recorded pinned keys).
    if let Some(logged) = &record.r5_commitments {
        let switches = record
            .r5_switches
            .ok_or_else(|| "the record carries R5 commitments but no R5 switches".to_owned())?;
        let r5_input = record
            .r5_input
            .as_ref()
            .ok_or_else(|| "the record carries R5 commitments but no R5 input".to_owned())?;
        let mode = match record.effective.as_str() {
            "acting" => EffectiveMode::Acting,
            "shadow" => EffectiveMode::Shadow,
            _ => EffectiveMode::Off,
        };
        let recommitted = route_policy::commit(&record.input, r5_input, mode, &switches);
        if &recommitted != logged {
            return Err(format!(
                "R5 commitments diverged: recorded sha={} replayed sha={}",
                logged.sha8(),
                recommitted.sha8()
            ));
        }
    }
    // #router-r6: the O-TRIAL commitments are pure too (static input plus
    // the recorded pinned trial knobs).
    if let Some(logged) = &record.trial_commitments {
        let switches = record.trial_switches.ok_or_else(|| {
            "the record carries O-TRIAL commitments but no trial switches".to_owned()
        })?;
        let r6_input = record
            .r6_input
            .as_ref()
            .ok_or_else(|| "the record carries O-TRIAL commitments but no R6 input".to_owned())?;
        let mode = match record.optimizer_effective.as_str() {
            "acting" => EffectiveMode::Acting,
            "shadow" => EffectiveMode::Shadow,
            _ => EffectiveMode::Off,
        };
        let recommitted =
            otrial_policy::commit(&replanned, &record.input, r6_input, mode, &switches);
        if &recommitted != logged {
            return Err(format!(
                "O-TRIAL commitments diverged: recorded sha={} replayed sha={}",
                logged.sha8(),
                recommitted.sha8()
            ));
        }
    }
    if let Some(snapshot) = record.far.as_ref().and_then(|far| far.snapshot.as_ref()) {
        for event in &snapshot.events {
            let samples = event
                .samples
                .iter()
                .map(|s: &SampleRecord| {
                    s.to_sample(registry::find(&s.cap).map(|decl| decl.id))
                        .ok_or_else(|| format!("sample names unregistered capability {}", s.cap))
                })
                .collect::<Result<Vec<ProgressSample>, String>>()?;
            let replayed = checkpoint::far_decision(
                &replanned,
                &samples,
                &event.decision.cap,
                event.decision.at,
            );
            if replayed != event.decision {
                return Err(format!(
                    "live CR-F decision diverged at {} {}: recorded {:?}, replayed {:?}",
                    event.decision.cap,
                    event.decision.at.token(),
                    event.decision,
                    replayed
                ));
            }
        }
    }
    let mut checked = 0usize;
    let mut skipped = 0usize;
    for logged in &record.checkpoints {
        if logged.truncated {
            skipped += 1;
            continue;
        }
        let samples = logged
            .samples
            .iter()
            .map(|s: &SampleRecord| {
                s.to_sample(registry::find(&s.cap).map(|decl| decl.id))
                    .ok_or_else(|| format!("sample names unregistered capability {}", s.cap))
            })
            .collect::<Result<Vec<ProgressSample>, String>>()?;
        let amendments = checkpoint::checkpoint(&replanned, &samples, logged.id);
        if amendments != logged.amendments {
            return Err(format!(
                "checkpoint {} diverged: recorded {:?}, replayed {:?}",
                logged.id.token(),
                logged.amendments,
                amendments
            ));
        }
        checked += 1;
    }
    Ok(ReplayReport {
        plan_sha8: replanned.sha8().to_owned(),
        checkpoints: checked,
        skipped,
    })
}

/// The shadow plan for the `ny vnncomp plan --router` printer.
pub(crate) fn printer_plan(ctx: &StartContext<'_>) -> Result<(PolicyInput, Plan), String> {
    guard::fail_open("printer", || {
        let input = policy_input(ctx);
        let plan = plan::plan(&input);
        (input, plan)
    })
    .map_err(|fault| fault.to_string())
}

/// Printer lines for a plan (the same grammar as the scored session).
pub(crate) fn printer_lines(input: &PolicyInput, plan: &Plan) -> Vec<String> {
    let mut lines = vec![telemetry::banner_line(
        RouterPolicy {
            mode: RouterMode::Shadow,
            optimizer: OptimizerMode::Shadow,
            rules: RuleSwitches::ALL_ON,
            optimizer_rules: OptimizerSwitches::ALL_ON,
            r4_rules: R4Switches::ALL_ON,
            r5_rules: R5Switches::ALL_ON,
            trial_rules: TrialSwitches::ALL_ON,
            priors: input.priors.is_some(),
        },
        EffectiveMode::Shadow,
        EffectiveMode::Shadow,
        "printer",
        input,
    )];
    lines.extend(input.capabilities.iter().map(telemetry::cap_line));
    lines.push(telemetry::plan_line(plan));
    lines.extend(
        plan.would
            .iter()
            .map(|w| format!("[router] would rule={} {}", w.rule, w.what)),
    );
    // #router-r2: what `on` would commit on this row (the printer arms nothing).
    let commitments: Commitments =
        acting_policy::commit(plan, input, EffectiveMode::Acting, &RuleSwitches::ALL_ON);
    lines.extend(acting::act_lines(&commitments));
    // #router-r3: what `optimizer.mode: on` would commit (the printer arms
    // nothing).
    let stops = ostop_policy::commit(
        plan,
        input,
        EffectiveMode::Acting,
        &OptimizerSwitches::ALL_ON,
        Some(&commitments),
    );
    lines.extend(ostop::act_lines(&stops));
    // #router-r4: what acting set 3 would commit (the printer arms nothing).
    let r4 = r4_policy::commit(
        plan,
        input,
        EffectiveMode::Acting,
        &R4Switches::ALL_ON,
        &commitments,
    );
    lines.extend(acting::act_lines_r4(&r4));
    // #router-r5: what acting set 4 would commit (the printer arms nothing;
    // it has no category file, so nothing is pinned).
    let r5 = route_policy::commit(
        input,
        &R5Input::default(),
        EffectiveMode::Acting,
        &R5Switches::ALL_ON,
    );
    lines.extend(route::decision_lines(&r5));
    // #router-r6: what the O-TRIAL rules would commit (the printer arms
    // nothing; it has no category file, so nothing is pinned).
    let trials = otrial_policy::commit(
        plan,
        input,
        &R6Input::default(),
        EffectiveMode::Acting,
        &TrialSwitches::ALL_ON,
    );
    lines.extend(otrial::act_lines(&trials));
    lines
}
