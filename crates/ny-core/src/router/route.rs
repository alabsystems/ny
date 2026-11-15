// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #router-r5 — CAPABILITIES ROUTED, NOT ARMED (ROUTER_PROGRAM §3.5, §10 R5):
//! acting set 4. PURE: no clock, no environment, no global; the same
//! `(input, r5 input, mode, switches)` gives the same [`R5Commitments`] bit
//! for bit, which `ny vnncomp plan --replay` re-checks.
//!
//! | rule | key (`router.rules.*`) | what it decides here | consulted by |
//! |---|---|---|---|
//! | CR-RS | `route` | which capabilities the router CONFIGURES on this row, per capability, name-blind: only at SR.seal (S7 + S8 done, envelope recorded), only where its own precondition admits the row's facts and host, its sealed envelope covers the row's operator signature, its parent's admission holds (or is re-checked at the consumer's own site) — or, for a lane of its own, its OWN predicate admits the row — the row did not pin its keys, and its exclusive slot is free | each capability's own arming read site, through `ny_core::route_authority` |
//! | CR-G | `gap_contract` | inside a multi-objective lane, fund classes in gap order with a contract of at least one build plus two levels (replacing the frozen-dual tier's fair share `left / (total - i)`); a class whose rung does not fit declines instead of half-running | the frozen-dual tier's class loop |
//!
//! A rule acts only when `router.mode` is `on` (effective `acting`) and its
//! own key is on (the declaration default; the key is an ablation switch).
//! Neither rides on CR-1: a route sets a capability's own typed keys, and
//! CR-G re-divides a tier's wall that the tier already owns.
//!
//! # The seal gate, and why it is two checks
//!
//! Every routable capability is declared with its SR ladder; the router
//! routes it only when BOTH its derived level is `seal` AND its ledger is
//! `Sealed`. Independently, a capability whose route sets a `High` moat-risk
//! key is never auto-selected unless it is sealed ([`may_auto_select`]).
//! Today no registry entry is sealed, so with the real registry CR-RS routes
//! NOTHING on any row; that is the expected state, pinned by a test. Every
//! failing check is recorded ([`RouteDecision::reasons`]), not only the
//! first, so a weakened seal check still shows the moat refusal.
//!
//! # Pins
//!
//! A typed key the row's own configuration set is a pin: a route never
//! overrides it (`pin:<key>`). A pin on a SEALED capability is PIN DEBT
//! ([`RouteDecision::pin_debt`]): the category file still arms (or disarms)
//! what the router would decide per instance. The static census of such
//! lines is the yaml-debt ratchet (`ny-cli::router`), which may only fall.
//!
//! # Soundness
//!
//! A route sets the typed keys a category file could set, to the value the
//! capability's own S7/S8 evidence arm ran, on a row inside the envelope that
//! evidence covers. It never touches a bound, a tolerance, an error term, a
//! gate or the publisher, and every routed capability still runs its own
//! admission and its own certificate checks in its lane. CR-G only re-divides
//! a tier's wall among its classes; a declined class stays open in the tree.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::acting::RuleAct;
use super::decl::{
    Admission, KnobRisk, OpEnvelope, OpKind, OpSet, SpecSet, SrLevel, UNKNOWN_INLINE,
};
use super::plan::{
    AdmissionState, CapInput, LedgerState, PolicyInput, RouteSpec, CR_1, CR_A, CR_C, CR_F, CR_G,
    CR_H, CR_K, CR_L, CR_R, CR_RS, CR_T,
};
use super::{EffectiveMode, RuleId};

/// Record schema of [`R5Commitments`].
pub const R5_SCHEMA: &str = "ny-router-r5/v1";

/// The R5 acting rules, in evaluation order.
pub const R5_RULES: [RuleId; 2] = [CR_RS, CR_G];

/// Every router rule that ACTS in this binary under `router.mode: on`: acting
/// set 1 (R2), acting set 3 (R4), then acting set 4 (R5).
pub const ACTING_SET: [RuleId; 11] = [
    CR_1, CR_A, CR_C, CR_T, CR_H, CR_R, CR_K, CR_L, CR_F, CR_RS, CR_G,
];

/// The two `router.rules.*` ablation switches of acting set 4. Each defaults
/// ON (participates when the router acts).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct R5Switches {
    /// CR-RS.
    pub route: bool,
    /// CR-G.
    pub gap_contract: bool,
}

impl R5Switches {
    /// Both rules participate (the declarations' defaults).
    pub const ALL_ON: Self = Self {
        route: true,
        gap_contract: true,
    };

    /// The switch for `rule`.
    #[must_use]
    pub fn get(&self, rule: RuleId) -> bool {
        match rule.as_str() {
            "CR-RS" => self.route,
            "CR-G" => self.gap_contract,
            _ => false,
        }
    }
}

/// What CR-RS reads beyond the static policy input: the typed key paths the
/// row's own configuration set (sorted, deduplicated). Recorded with the
/// commitments so a replay re-derives them.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct R5Input {
    /// Typed key paths the row pinned (any value).
    pub pinned_keys: Vec<String>,
}

/// CR-RS's decision for one routable capability.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RouteDecision {
    /// The capability.
    pub cap: String,
    /// Its SR level.
    pub sr: SrLevel,
    /// Whether the router routes it on this row.
    pub routed: bool,
    /// Every failing check, in evaluation order (empty iff routable here).
    pub reasons: Vec<String>,
    /// A pin on a SEALED capability: the row's file decides what the router
    /// would decide per instance.
    pub pin_debt: bool,
    /// `(typed key path, value)` the route applies.
    pub settings: Vec<(String, String)>,
}

impl RouteDecision {
    /// The first failing check (`None` when routed).
    #[must_use]
    pub fn blocked_by(&self) -> Option<&str> {
        self.reasons.first().map(String::as_str)
    }
}

/// What acting set 4 commits to on a row (pure; replayable).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct R5Commitments {
    /// [`R5_SCHEMA`].
    pub schema: String,
    /// Whether the router acts at all on this row.
    pub acting: bool,
    /// Every R5 rule's decision, in [`R5_RULES`] order.
    pub rules: Vec<RuleAct>,
    /// CR-RS: one decision per routable capability, registry order (empty
    /// when the router does not act).
    pub routes: Vec<RouteDecision>,
    /// CR-G: whether the gap-order contract funds multi-objective classes.
    pub gap_contract: bool,
    /// Hex sha256 of everything above.
    pub sha: String,
}

impl R5Commitments {
    /// The first 8 hex digits of [`R5Commitments::sha`].
    #[must_use]
    pub fn sha8(&self) -> &str {
        self.sha.get(..8).unwrap_or(&self.sha)
    }

    /// Whether `rule` acts on this row.
    #[must_use]
    pub fn acts(&self, rule: RuleId) -> bool {
        self.rules
            .iter()
            .any(|act| act.rule == rule.as_str() && act.acts)
    }

    /// The routed decisions.
    pub fn routed(&self) -> impl Iterator<Item = &RouteDecision> {
        self.routes.iter().filter(|decision| decision.routed)
    }

    /// The decisions that carry pin debt.
    pub fn pin_debt(&self) -> impl Iterator<Item = &RouteDecision> {
        self.routes.iter().filter(|decision| decision.pin_debt)
    }
}

fn act(rule: RuleId, acts: bool, blocked_by: &str, what: String) -> RuleAct {
    RuleAct {
        rule: rule.as_str().to_owned(),
        acts,
        blocked_by: (!acts).then(|| blocked_by.to_owned()),
        what,
    }
}

/// Whether a capability counts as SEALED: its derived SR level is `seal`
/// and its ledger is `Sealed` (both; either alone is not a seal).
#[must_use]
pub fn sealed(spec: &RouteSpec, ledger: &LedgerState) -> bool {
    spec.sr == SrLevel::Seal && matches!(ledger, LedgerState::Sealed)
}

/// The moat guard, independent of the seal checks: a route that sets a
/// `High` moat-risk key is never selected automatically unless the
/// capability is sealed.
#[must_use]
pub const fn may_auto_select(risk: KnobRisk, sealed: bool) -> bool {
    match risk {
        KnobRisk::Low => true,
        KnobRisk::High => sealed,
    }
}

fn ledger_token(ledger: &LedgerState) -> &'static str {
    match ledger {
        LedgerState::Open(_) => "open",
        LedgerState::Shadow => "shadow",
        LedgerState::SealedAsShipped => "sealed_as_shipped",
        LedgerState::Sealed => "sealed",
    }
}

/// Whether the parent's own admission lets a consumer inside it be routed:
/// admitted, or inline (the parent re-checks its own admission at the site
/// the consumer lives in, so a route there is only ever reached on rows the
/// parent admitted itself).
fn parent_check(input: &PolicyInput, parent: &str) -> Option<String> {
    let Some(cap) = input.capabilities.iter().find(|cap| cap.id == parent) else {
        return Some(format!("parent:{parent}:unregistered"));
    };
    match &cap.admission {
        AdmissionState::Admit => None,
        AdmissionState::Unknown(why) if why == UNKNOWN_INLINE => None,
        other => Some(format!("parent:{parent}:{}", other.token())),
    }
}

/// #prepared-root-route: whether a routable capability that is a lane of its
/// own (no parent: its declaration names its OWN predicate) admits the row by
/// that predicate: admitted, or inline (the lane re-checks its own admission
/// at its site, exactly as a parent's inline admission is deferred). Every
/// consumer inside a parent lane is checked by [`parent_check`] instead.
fn own_check(cap: &CapInput) -> Option<String> {
    match &cap.admission {
        AdmissionState::Admit => None,
        AdmissionState::Unknown(why) if why == UNKNOWN_INLINE => None,
        other => Some(format!("own:{}", other.token())),
    }
}

/// Whether the sealed envelope covers the row: every operator kind of the
/// row's signature is inside it, and its specification archetypes cover the
/// row's archetype — or, when that archetype is unknown at router start,
/// every archetype the consumer's call site serves.
fn envelope_check(input: &PolicyInput, spec: &RouteSpec) -> Option<String> {
    if !input.facts.ops.is_subset_of(spec.envelope_ops) {
        return Some("envelope:ops".to_owned());
    }
    let covered = match input.facts.spec {
        Some(archetype) => spec.envelope_specs.contains(archetype),
        None => {
            let served = if spec.admit_when.specs == SpecSet::EMPTY {
                SpecSet::ALL
            } else {
                spec.admit_when.specs
            };
            served.is_subset_of(spec.envelope_specs)
        }
    };
    (!covered).then(|| "envelope:spec".to_owned())
}

/// Every failing check for one routable capability, in evaluation order.
fn checks(
    cap: &CapInput,
    spec: &RouteSpec,
    input: &PolicyInput,
    r5: &R5Input,
    taken: &[(String, String)],
) -> (Vec<String>, bool) {
    let mut reasons = Vec::new();
    let is_sealed = sealed(spec, &cap.ledger);
    if spec.sr != SrLevel::Seal {
        reasons.push(format!("sr:{}", spec.sr.token()));
    }
    if !matches!(cap.ledger, LedgerState::Sealed) {
        reasons.push(format!("ledger:{}", ledger_token(&cap.ledger)));
    }
    if !may_auto_select(spec.risk, is_sealed) {
        reasons.push("moat:high_unsealed".to_owned());
    }
    match spec.admit_when.evaluate(&input.facts, &input.host) {
        Admission::Admit => {}
        Admission::Decline(why) => reasons.push(format!("admit:decline:{why}")),
        Admission::Unknown(why) => reasons.push(format!("admit:unknown:{why}")),
    }
    let admission = match &spec.parent {
        Some(parent) => parent_check(input, parent),
        None => own_check(cap),
    };
    if let Some(why) = admission {
        reasons.push(why);
    }
    if let Some(why) = envelope_check(input, spec) {
        reasons.push(why);
    }
    let mut pinned = false;
    for (key, _) in &spec.settings {
        if r5.pinned_keys.iter().any(|pin| pin == key) {
            pinned = true;
            reasons.push(format!("pin:{key}"));
        }
    }
    if let Some(slot) = &spec.slot {
        if let Some((_, holder)) = taken.iter().find(|(taken_slot, _)| taken_slot == slot) {
            reasons.push(format!("slot:{slot}:{holder}"));
        }
    }
    (reasons, is_sealed && pinned)
}

/// CR-RS over every routable capability, registry order. `enabled` false
/// (the rule's key) records `key` first and routes nothing.
fn route_decisions(input: &PolicyInput, r5: &R5Input, enabled: bool) -> Vec<RouteDecision> {
    let mut taken: Vec<(String, String)> = Vec::new();
    let mut out = Vec::new();
    for cap in &input.capabilities {
        let Some(spec) = &cap.route else {
            continue;
        };
        let (mut reasons, pin_debt) = checks(cap, spec, input, r5, &taken);
        if !enabled {
            reasons.insert(0, "key".to_owned());
        }
        let routed = reasons.is_empty();
        if routed {
            if let Some(slot) = &spec.slot {
                taken.push((slot.clone(), cap.id.clone()));
            }
        }
        out.push(RouteDecision {
            cap: cap.id.clone(),
            sr: spec.sr,
            routed,
            reasons,
            pin_debt,
            settings: spec.settings.clone(),
        });
    }
    out
}

/// The row's R5 commitments. Pure and total.
#[must_use]
pub fn commit(
    input: &PolicyInput,
    r5: &R5Input,
    mode: EffectiveMode,
    switches: &R5Switches,
) -> R5Commitments {
    let acting = matches!(mode, EffectiveMode::Acting);
    if !acting {
        let reason = format!("mode:{}", mode.token());
        let rules = R5_RULES
            .iter()
            .map(|rule| act(*rule, false, &reason, "router not acting".to_owned()))
            .collect();
        return finish(R5Commitments {
            schema: R5_SCHEMA.to_owned(),
            acting,
            rules,
            routes: Vec::new(),
            gap_contract: false,
            sha: String::new(),
        });
    }

    // CR-RS: route only at SR.seal.
    let routes = route_decisions(input, r5, switches.route);
    let routable = routes.len();
    let at_seal = routes.iter().filter(|d| d.sr == SrLevel::Seal).count();
    let routed = routes.iter().filter(|d| d.routed).count();
    let debt = routes.iter().filter(|d| d.pin_debt).count();
    let mut rules = Vec::with_capacity(R5_RULES.len());
    rules.push(act(
        CR_RS,
        switches.route && routed > 0,
        if !switches.route {
            "key"
        } else if at_seal == 0 {
            "no_sealed_capability"
        } else {
            "none_admitted"
        },
        format!(
            "route a capability only at SR.seal where its own precondition admits the row, its \
             sealed envelope covers it, its parent admits it and the row did not pin its keys; \
             routable={routable} at_seal={at_seal} routed={routed} pin_debt={debt}"
        ),
    ));

    // CR-G: gap-order contracts inside multi-objective lanes.
    rules.push(act(
        CR_G,
        switches.gap_contract,
        "key",
        "fund multi-objective classes in gap order with a contract >= one build + 2 levels \
         (frozen-dual tier; replaces the fair share left/(total-i)); a class whose rung does not \
         fit declines and stays open in the tree"
            .to_owned(),
    ));

    finish(R5Commitments {
        schema: R5_SCHEMA.to_owned(),
        acting,
        rules,
        routes,
        gap_contract: switches.gap_contract,
        sha: String::new(),
    })
}

/// Stamp the digest (over everything but the digest itself).
fn finish(mut commitments: R5Commitments) -> R5Commitments {
    commitments.sha = String::new();
    let mut hash = Sha256::new();
    hash.update(format!("{commitments:?}").as_bytes());
    let digest: [u8; 32] = hash.finalize().into();
    commitments.sha = super::hash::hex(&digest);
    commitments
}

// ---- #admission-census -------------------------------------------------------
//
// The seal pipeline must moat EXACTLY the rows the router would route a
// capability on once it is sealed. Those are CR-RS's own decisions, so the
// census below asks `commit` itself, on the live policy input, with ONE
// capability pretended sealed: its SR level `seal`, its ledger `Sealed`, and
// its envelope the one the census names. Every other routable capability is
// taken out of the census input, so a decision is the capability's ADMISSION
// ALONE (its precondition, its parent's admission, the envelope) and never a
// slot another capability holds; the census input carries no pins (a pin can
// only refuse, and the census describes no row file). Nothing here decides
// anything the router does not: the reasons are `commit`'s own.

/// #admission-census: the schema of a census (the binary's row table and the
/// binding the seal pipeline checks it by). #admission-precision: v2 adds the
/// `spec` column (the row's archetype and routing shape as the router read
/// them at start) and the `spec_read` binding line.
pub const CENSUS_SCHEMA: &str = "ny-admission-census/v2";

/// Every operator kind except `other`, which no seal ever records.
const fn ops_but_other() -> OpSet {
    let mut set = OpSet::EMPTY;
    let mut i = 0;
    while i < OpKind::ALL.len() {
        if !matches!(OpKind::ALL[i], OpKind::Other) {
            set = set.with(OpKind::ALL[i]);
        }
        i += 1;
    }
    set
}

/// #admission-census: the WIDEST envelope a seal can record — every operator
/// kind but `other` (never sealed), every archetype. CR-RS's envelope check is
/// monotone in the envelope (a row inside a narrower envelope is inside a
/// wider one), so the rows a census admits under this envelope are a SUPERSET
/// of the rows the router routes under any envelope a seal can record: the
/// moat a census defines this way covers whatever envelope the seal derives.
pub const CENSUS_WIDEST_ENVELOPE: OpEnvelope = OpEnvelope {
    ops: ops_but_other(),
    specs: SpecSet::ALL,
};

/// #admission-census: the ids of the routable capabilities of `input`, in
/// registry order.
#[must_use]
pub fn census_capabilities(input: &PolicyInput) -> Vec<&str> {
    input
        .capabilities
        .iter()
        .filter(|cap| cap.route.is_some())
        .map(|cap| cap.id.as_str())
        .collect()
}

/// #admission-census: CR-RS's decision for the routable capability `cap` on
/// `input`, with `cap` PRETENDED sealed under `envelope` (`None` when `cap` is
/// not a routable capability of `input`). Pure and total: it is `commit` on a
/// copy of `input` in which `cap`'s SR level is `seal`, its ledger `Sealed`
/// and its envelope `envelope`, every OTHER capability's route is removed (the
/// admission alone: no slot is ever held), and nothing is pinned. Its
/// `reasons` are therefore exactly the admission checks `commit` makes — the
/// precondition (`admit:`), the parent's admission (`parent:`) or, for a lane
/// of its own, its own predicate's (`own:`), and the envelope (`envelope:`) —
/// and never a seal check.
#[must_use]
pub fn census_decision(
    input: &PolicyInput,
    cap: &str,
    envelope: OpEnvelope,
) -> Option<RouteDecision> {
    let mut pretend = input.clone();
    let mut found = false;
    for candidate in &mut pretend.capabilities {
        if candidate.id == cap {
            let spec = candidate.route.as_mut()?;
            spec.sr = SrLevel::Seal;
            spec.envelope_ops = envelope.ops;
            spec.envelope_specs = envelope.specs;
            candidate.ledger = LedgerState::Sealed;
            found = true;
        } else {
            candidate.route = None;
        }
    }
    if !found {
        return None;
    }
    commit(
        &pretend,
        &R5Input::default(),
        EffectiveMode::Acting,
        &R5Switches::ALL_ON,
    )
    .routes
    .into_iter()
    .find(|decision| decision.cap == cap)
}
