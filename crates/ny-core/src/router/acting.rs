// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #router-r2 — ACTING SET 1 (ROUTER_PROGRAM §10 R2): the pure decision of
//! what the router DOES on a row. PURE: no clock, no environment, no global;
//! the same `(plan, input, mode, switches)` gives the same [`Commitments`]
//! bit for bit, which `ny vnncomp plan --replay` re-checks.
//!
//! # The seven rules, each its own capability and its own typed key
//!
//! | rule | key (`router.rules.*`) | what it decides here | consulted by |
//! |---|---|---|---|
//! | CR-1 | `cap_authority` | arm the ONE deadline authority (`ny_core::deadline_authority`) with one monotone cell per governed lane, capped per slot | every governed lane's deadline read |
//! | CR-A | `admission` | which lanes the authority GOVERNS: a lane whose own predicate declined, whose ledger is not sealed, or whose first rung does not fit keeps today's deadline | `Authority::governs` |
//! | CR-C | `contracts` | the attack slice is committed ONCE, up front (the `#lane-budget-frame` identity plan), each lane claims `now + contract` at its own start | the attack-slice windows |
//! | CR-T | `terminal` | the internal verifier is the terminal: never contracted, never shortened, the recipient of every hand-on | the verifier's claim |
//! | CR-H | `hand_on` | a lane that ends undecided hands its unspent contract to the terminal; a root warmup phase that expires early hands its time to the search instead of abandoning it | lane releases; `bab.initial_bounds_warmup_fallback` |
//! | CR-R | `publish_grace` | a completed certificate may publish up to `g` past its lane's search deadline, `0 < g < 3 s`, `g <` join margin; every collector of that lane waits the same `g` | the margin-row lanes and their collectors |
//! | CR-K | `derived_caps` | per-clause retry passes and the depth cap derived from the deadline and the memory lease (`derive`), never below today | `bab.disjunctive_clause_retry_passes`, `bab.max_depth` |
//!
//! A rule acts only when `router.mode` is `on` (effective `acting`), its own
//! key is on (the declaration default; the key is an ablation switch), and
//! the rules it rides on act (every cell rule rides on CR-1). A blocked rule
//! says why (`blocked_by`). No rule touches precision, a certified error term,
//! a tolerance, a verdict or the publisher; a value a rule derives applies
//! only where the row's own configuration did not pin it (pins win, and the
//! applying site logs `blocked_by=pin`).
//!
//! # Name-blind
//!
//! The input is the plan and the policy input, which carry no name and no
//! path (type-level blindness, `router_policy_input_has_no_name_or_path_field`).

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::derive;
use super::plan::{AdmissionState, Plan, PolicyInput, CR_1, CR_A, CR_C, CR_H, CR_K, CR_R, CR_T};
use super::{EffectiveMode, RuleId};
use crate::deadline_authority::{LaneSlot, SLOTS};

/// Record schema of [`Commitments`].
pub const ACTING_SCHEMA: &str = "ny-router-acting/v1";

/// The R2 acting rules, in evaluation order.
pub const R2_RULES: [RuleId; 7] = [CR_1, CR_A, CR_C, CR_T, CR_H, CR_R, CR_K];

/// The capability ids each governed slot stands for (registry ids).
pub const SLOT_CAPS: [(LaneSlot, &[&str]); SLOTS] = [
    (LaneSlot::AttackSignSpace, &["attack.bnn_lp"]),
    (LaneSlot::AttackStePgd, &["attack.bnn_ste"]),
    (LaneSlot::AttackUpfront, &["attack.upfront_apgd"]),
    (
        LaneSlot::Verifier,
        &["relu_split.graph", "input_split", "lowdim", "mip.ay"],
    ),
    (LaneSlot::MarginRowConcurrent, &["margin-row"]),
    (LaneSlot::MarginRowInline, &["margin-row"]),
];

/// Today's depth cap default (`BetaCrownConfig::default().max_depth`); the
/// derived cap is never below it.
pub const TODAY_MAX_DEPTH: u64 = 100;

/// Per-rule switches: the seven `router.rules.*` typed keys. Each defaults
/// ON — it participates when the router acts — so the router's own mode is
/// the only arming decision, and a key set `false` ablates one rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RuleSwitches {
    /// CR-A.
    pub admission: bool,
    /// CR-C.
    pub contracts: bool,
    /// CR-1.
    pub cap_authority: bool,
    /// CR-T.
    pub terminal: bool,
    /// CR-H.
    pub hand_on: bool,
    /// CR-R.
    pub publish_grace: bool,
    /// CR-K.
    pub derived_caps: bool,
}

impl RuleSwitches {
    /// Every rule participates (the declarations' defaults).
    pub const ALL_ON: Self = Self {
        admission: true,
        contracts: true,
        cap_authority: true,
        terminal: true,
        hand_on: true,
        publish_grace: true,
        derived_caps: true,
    };

    /// The switch for `rule`.
    #[must_use]
    pub fn get(&self, rule: RuleId) -> bool {
        match rule.as_str() {
            "CR-A" => self.admission,
            "CR-C" => self.contracts,
            "CR-1" => self.cap_authority,
            "CR-T" => self.terminal,
            "CR-H" => self.hand_on,
            "CR-R" => self.publish_grace,
            "CR-K" => self.derived_caps,
            _ => false,
        }
    }
}

/// One rule's decision on a row.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RuleAct {
    /// The rule id.
    pub rule: String,
    /// Whether it acts.
    pub acts: bool,
    /// Why it does not (`mode:<token>`, `key`, `CR-1`, a slot left ungoverned, ...).
    pub blocked_by: Option<String>,
    /// What it does (or would do).
    pub what: String,
}

/// CR-A's decision for one slot.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SlotGovernance {
    /// The slot.
    pub slot: LaneSlot,
    /// Whether the authority governs its deadline.
    pub governed: bool,
    /// Why (`admit`, `deferred:<why>` — the lane re-checks its own predicate
    /// at its site — `declined:<why>`, `ledger:<status>`, `rung`, `cr-a-off`,
    /// `unregistered`).
    pub reason: String,
    /// The slot's cap as a back-off from the scored deadline, milliseconds.
    pub backoff_ms: u64,
}

/// CR-K / CR-H: the values the verifier's configuration receives where the
/// row did not pin them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct DerivedPlan {
    /// Derived per-clause retry passes (`None`: CR-K does not act).
    pub clause_passes: Option<u64>,
    /// Derived depth cap, applied only when [`derive::depth_lift_gate`]
    /// admits it at the applying site (`None`: CR-K does not act).
    pub max_depth: Option<u64>,
    /// The host's static memory lease, bytes (the depth gate's bound).
    pub lease_bytes: u64,
    /// CR-H: a root warmup that expires early hands its time to the search.
    pub warmup_hand_on: bool,
}

/// What the router commits to on a row (pure; replayable).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Commitments {
    /// [`ACTING_SCHEMA`].
    pub schema: String,
    /// Whether the router acts at all on this row.
    pub acting: bool,
    /// Every R2 rule's decision, in [`R2_RULES`] order.
    pub rules: Vec<RuleAct>,
    /// CR-1: arm the deadline authority.
    pub arm_authority: bool,
    /// CR-A: per-slot governance, in slot order (empty when not armed).
    pub governance: Vec<SlotGovernance>,
    /// CR-T: the terminal slot.
    pub terminal: Option<LaneSlot>,
    /// CR-C: commit the attack slice once, up front.
    pub attack_contracts: bool,
    /// CR-H: releases hand their unspent contract on.
    pub release_hand_on: bool,
    /// CR-R: the publish grace, milliseconds.
    pub grace_ms: Option<u64>,
    /// CR-K / CR-H verifier values.
    pub derived: DerivedPlan,
    /// Hex sha256 of everything above.
    pub sha: String,
}

impl Commitments {
    /// The first 8 hex digits of [`Commitments::sha`].
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

    /// CR-A's decision for `slot`.
    #[must_use]
    pub fn governed(&self, slot: LaneSlot) -> bool {
        self.governance.iter().any(|g| g.slot == slot && g.governed)
    }

    /// Per-slot governance flags, index order (the authority's arming input).
    #[must_use]
    pub fn governed_flags(&self) -> [bool; SLOTS] {
        let mut flags = [false; SLOTS];
        for (flag, slot) in flags.iter_mut().zip(LaneSlot::ALL) {
            *flag = self.governed(slot);
        }
        flags
    }

    /// Per-slot cap back-offs from the scored deadline, milliseconds.
    #[must_use]
    pub fn backoffs_ms(&self) -> [u64; SLOTS] {
        let mut out = [0u64; SLOTS];
        for (value, slot) in out.iter_mut().zip(LaneSlot::ALL) {
            *value = self
                .governance
                .iter()
                .find(|g| g.slot == slot)
                .map_or(0, |g| g.backoff_ms);
        }
        out
    }
}

fn act(rule: RuleId, acts: bool, blocked_by: Option<&str>, what: String) -> RuleAct {
    RuleAct {
        rule: rule.as_str().to_owned(),
        acts,
        blocked_by: if acts {
            None
        } else {
            blocked_by.map(str::to_owned)
        },
        what,
    }
}

/// The back-off of `slot`'s cap from the scored deadline: the verifier tier's
/// own reserve (`official - internal`: the publication floor the router never
/// plans below) for the attack slice and the terminal, and the margin-row
/// lanes' own wrap-up reserve for them. Never a new reserve: each is the one
/// today's route already keeps.
#[must_use]
pub fn slot_backoff_ms(slot: LaneSlot, input: &PolicyInput) -> u64 {
    match slot {
        LaneSlot::MarginRowConcurrent | LaneSlot::MarginRowInline => derive::MARGIN_ROW_TAIL_MS,
        _ => input
            .budget
            .official_ms
            .saturating_sub(input.budget.internal_ms),
    }
}

/// CR-A for one slot.
fn govern(caps: &[&str], plan: &Plan, input: &PolicyInput) -> (bool, String) {
    let mapped: Vec<_> = input
        .capabilities
        .iter()
        .filter(|cap| caps.contains(&cap.id.as_str()))
        .collect();
    if mapped.is_empty() {
        return (false, "unregistered".to_owned());
    }
    let unfit = |id: &str| {
        plan.declined
            .iter()
            .any(|(cap, why)| cap == id && why.starts_with("rung_does_not_fit"))
    };
    // The verifier runs whichever route its own dispatch picks, so it is
    // governed unless EVERY route declined; a single-capability slot follows
    // its capability.
    let live: Vec<_> = mapped
        .iter()
        .filter(|cap| !matches!(cap.admission, AdmissionState::Decline(_)))
        .collect();
    let Some(first_live) = live.first() else {
        let why = mapped
            .first()
            .map_or_else(String::new, |cap| cap.admission.token());
        return (
            false,
            format!("declined:{}", why.trim_start_matches("decline:")),
        );
    };
    if let Some(open) = live.iter().find(|cap| !cap.ledger.is_sealed()) {
        return (false, format!("ledger:{}", open.id));
    }
    if live.iter().any(|cap| unfit(&cap.id)) {
        return (false, "rung".to_owned());
    }
    match &first_live.admission {
        AdmissionState::Admit => (true, "admit".to_owned()),
        AdmissionState::Unknown(why) => (true, format!("deferred:{why}")),
        AdmissionState::Decline(why) => (false, format!("declined:{why}")),
    }
}

/// The row's commitments. Pure and total.
#[must_use]
pub fn commit(
    plan: &Plan,
    input: &PolicyInput,
    mode: EffectiveMode,
    switches: &RuleSwitches,
) -> Commitments {
    let acting = matches!(mode, EffectiveMode::Acting);
    let mut rules = Vec::with_capacity(R2_RULES.len());
    let off_reason = format!("mode:{}", mode.token());

    if !acting {
        for rule in R2_RULES {
            rules.push(act(
                rule,
                false,
                Some(off_reason.as_str()),
                "router not acting".to_owned(),
            ));
        }
        return finish(Commitments {
            schema: ACTING_SCHEMA.to_owned(),
            acting,
            rules,
            arm_authority: false,
            governance: Vec::new(),
            terminal: None,
            attack_contracts: false,
            release_hand_on: false,
            grace_ms: None,
            derived: DerivedPlan::default(),
            sha: String::new(),
        });
    }

    // CR-1: the one deadline authority.
    let cap_authority = switches.cap_authority;
    rules.push(act(
        CR_1,
        cap_authority,
        Some("key"),
        format!(
            "arm one monotone deadline cell per governed lane; hard={}ms",
            input.budget.official_ms
        ),
    ));

    // CR-A: which lanes the authority governs.
    let admission = cap_authority && switches.admission;
    let governance: Vec<SlotGovernance> = if cap_authority {
        SLOT_CAPS
            .iter()
            .map(|(slot, caps)| {
                let (governed, reason) = if admission {
                    govern(caps, plan, input)
                } else {
                    (true, "cr-a-off".to_owned())
                };
                SlotGovernance {
                    slot: *slot,
                    governed,
                    reason,
                    backoff_ms: slot_backoff_ms(*slot, input),
                }
            })
            .collect()
    } else {
        Vec::new()
    };
    let governed = |slot: LaneSlot| governance.iter().any(|g| g.slot == slot && g.governed);
    let governed_list = governance
        .iter()
        .filter(|g| g.governed)
        .map(|g| g.slot.token())
        .collect::<Vec<_>>()
        .join(",");
    rules.push(act(
        CR_A,
        admission,
        Some(if cap_authority { "key" } else { "CR-1" }),
        format!(
            "governed=[{}]",
            if governed_list.is_empty() {
                "-"
            } else {
                governed_list.as_str()
            }
        ),
    ));

    // CR-C: the attack slice committed once, up front.
    let attack_governed = LaneSlot::ALL
        .into_iter()
        .any(|slot| slot.in_attack_slice() && governed(slot));
    let attack_contracts = cap_authority && switches.contracts && attack_governed;
    rules.push(act(
        CR_C,
        attack_contracts,
        Some(if !cap_authority {
            "CR-1"
        } else if !switches.contracts {
            "key"
        } else {
            "attack_slice_ungoverned"
        }),
        "commit the attack slice once up front; each lane claims now+contract at its start"
            .to_owned(),
    ));

    // CR-T: the verifier is the terminal.
    let terminal_acts = cap_authority && switches.terminal && governed(LaneSlot::Verifier);
    rules.push(act(
        CR_T,
        terminal_acts,
        Some(if !cap_authority {
            "CR-1"
        } else if !switches.terminal {
            "key"
        } else {
            "verifier_ungoverned"
        }),
        "the verifier takes the remainder: never contracted, never shortened".to_owned(),
    ));

    // CR-H: hand-on (lane releases need a terminal; the phase hand-on does not).
    let hand_on = switches.hand_on;
    let release_hand_on = hand_on && terminal_acts;
    rules.push(act(
        CR_H,
        hand_on,
        Some("key"),
        format!(
            "releases hand on={}; a root warmup that expires early hands its time to the search",
            u8::from(release_hand_on)
        ),
    ));

    // CR-R: the publish grace.
    let grace = derive::publish_grace_ms(input.budget.join_margin_ms);
    let margin_row_governed =
        governed(LaneSlot::MarginRowConcurrent) && governed(LaneSlot::MarginRowInline);
    let publish_grace =
        cap_authority && switches.publish_grace && margin_row_governed && grace.is_some();
    rules.push(act(
        CR_R,
        publish_grace,
        Some(if !cap_authority {
            "CR-1"
        } else if !switches.publish_grace {
            "key"
        } else if !margin_row_governed {
            "margin_row_ungoverned"
        } else {
            "join_margin_too_small"
        }),
        format!(
            "grace={}ms inside join={}ms (write reserve {}ms)",
            grace.unwrap_or(0),
            input.budget.join_margin_ms,
            derive::WRITE_RESERVE_MS
        ),
    ));

    // CR-K: derived caps.
    let derived_caps = switches.derived_caps;
    let lease_bytes = input.host.envelope_bytes();
    let window = plan.window_ms;
    let derived = DerivedPlan {
        clause_passes: derived_caps.then(|| derive::clause_passes(window)),
        max_depth: derived_caps.then(|| derive::max_depth(window, TODAY_MAX_DEPTH)),
        lease_bytes,
        warmup_hand_on: hand_on,
    };
    rules.push(act(
        CR_K,
        derived_caps,
        Some("key"),
        format!(
            "clause_passes={} max_depth={} (lifted only under a queue byte budget <= lease/{}) \
             window={window}ms lease={lease_bytes}B",
            derived.clause_passes.unwrap_or(0),
            derived.max_depth.unwrap_or(0),
            derive::QUEUE_LEASE_DEN,
        ),
    ));

    finish(Commitments {
        schema: ACTING_SCHEMA.to_owned(),
        acting,
        rules,
        arm_authority: cap_authority,
        governance,
        terminal: terminal_acts.then_some(LaneSlot::Verifier),
        attack_contracts,
        release_hand_on,
        grace_ms: if publish_grace { grace } else { None },
        derived,
        sha: String::new(),
    })
}

/// Stamp the digest (over everything but the digest itself).
fn finish(mut commitments: Commitments) -> Commitments {
    commitments.sha = String::new();
    let mut hash = Sha256::new();
    hash.update(format!("{commitments:?}").as_bytes());
    let digest: [u8; 32] = hash.finalize().into();
    commitments.sha = super::hash::hex(&digest);
    commitments
}
