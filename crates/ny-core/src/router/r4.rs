// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #router-r4 — ACTING SET 3 (ROUTER_PROGRAM §10 R4): the memory LEASE (CR-L)
//! and the FAR hand-on (CR-F). PURE: no clock, no environment, no global; the
//! same `(plan, input, mode, switches, r2)` gives the same [`R4Commitments`]
//! bit for bit, which `ny vnncomp plan --replay` re-checks.
//!
//! | rule | key (`router.rules.*`) | what it decides here | consulted by |
//! |---|---|---|---|
//! | CR-L | `lease` | arm the ONE memory lease authority (`ny_core::memory_lease`) with the declared host's static envelope: the verifier holds `E/2`, every graph-heavy optional lane acquires its memory model's bytes before it starts, and is refused when the aggregate would pass `E`; the live envelope stays the floor at every gate | the optional-heavy memory gates in `ny vnncomp` |
//! | CR-F | `far_hand_on` | arm the FAR hand-on authority (`ny_core::far_authority`) with the routes `(source -> STRICTLY sealed recipient)` the plan admits; at a source lane's own safe point after c2 / c3, two trailing windows reading FAR (work units) hand its unspent cell to the recipient's cell. With no route (every R4 row: no registry entry is `Sealed`) it arms in OBSERVE mode — it classifies and logs, and hands nothing on | a source lane's safe point (lowdim's wave loop) |
//!
//! A rule acts only when `router.mode` is `on` (effective `acting`) and its
//! own key is on (the declaration default; the key is an ablation switch).
//! CR-F rides on CR-1 (it moves time only through the deadline cells) and
//! never acts on a short row (CR-S). A blocked rule says why (`blocked_by`).
//!
//! # Soundness
//!
//! Admission and time only. A lease decides WHETHER an optional lane runs; a
//! hand-on ends a lane at one of ITS declared safe points (an undecided lane
//! reports undecided) and moves its unspent seconds to another lane's cell
//! (`credited <= unspent`, never past the recipient's cap). Neither touches a
//! bound, a tolerance, an error term or the publisher.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::acting::{Commitments, RuleAct, SLOT_CAPS};
use super::checkpoint::far_recipient;
use super::lease::{self, LeasePlan};
use super::plan::{Plan, PolicyInput, CR_1, CR_A, CR_C, CR_F, CR_H, CR_K, CR_L, CR_R, CR_T};
use super::{EffectiveMode, RuleId};
use crate::deadline_authority::LaneSlot;

/// Record schema of [`R4Commitments`].
pub const R4_SCHEMA: &str = "ny-router-r4/v1";

/// The R4 acting rules, in evaluation order.
pub const R4_RULES: [RuleId; 2] = [CR_L, CR_F];

/// Every router rule that ACTS in this binary under `router.mode: on`: acting
/// set 1 (R2) followed by acting set 3 (R4).
pub const ACTING_SET: [RuleId; 9] = [CR_1, CR_A, CR_C, CR_T, CR_H, CR_R, CR_K, CR_L, CR_F];

/// The two `router.rules.*` ablation switches of acting set 3. Each defaults
/// ON (participates when the router acts).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct R4Switches {
    /// CR-L.
    pub lease: bool,
    /// CR-F.
    pub far_hand_on: bool,
}

impl R4Switches {
    /// Both rules participate (the declarations' defaults).
    pub const ALL_ON: Self = Self {
        lease: true,
        far_hand_on: true,
    };

    /// The switch for `rule`.
    #[must_use]
    pub fn get(&self, rule: RuleId) -> bool {
        match rule.as_str() {
            "CR-L" => self.lease,
            "CR-F" => self.far_hand_on,
            _ => false,
        }
    }
}

/// One CR-F route: a source lane and the strictly sealed recipient its
/// unspent contract would reach, with the deadline cells each one owns.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FarRoute {
    /// The source capability (declares safe points).
    pub from: String,
    /// The recipient capability (`Sealed`, admitted, a prover).
    pub to: String,
    /// The source's deadline slot.
    pub from_slot: LaneSlot,
    /// The recipient's deadline slot.
    pub to_slot: LaneSlot,
}

/// What CR-F arms.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FarCommit {
    /// The routes a hand-on may take (empty: observe only).
    pub routes: Vec<FarRoute>,
    /// True when no route exists: the authority classifies and logs, and
    /// hands nothing on.
    pub observe_only: bool,
}

/// What acting set 3 commits to on a row (pure; replayable).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct R4Commitments {
    /// [`R4_SCHEMA`].
    pub schema: String,
    /// Whether the router acts at all on this row.
    pub acting: bool,
    /// Every R4 rule's decision, in [`R4_RULES`] order.
    pub rules: Vec<RuleAct>,
    /// CR-L: the lease plan to arm (`None`: no lease, today's live gate).
    pub lease: Option<LeasePlan>,
    /// CR-F: what the FAR authority arms with (`None`: not armed).
    pub far: Option<FarCommit>,
    /// Hex sha256 of everything above.
    pub sha: String,
}

impl R4Commitments {
    /// The first 8 hex digits of [`R4Commitments::sha`].
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
}

fn act(rule: RuleId, acts: bool, blocked_by: &str, what: String) -> RuleAct {
    RuleAct {
        rule: rule.as_str().to_owned(),
        acts,
        blocked_by: (!acts).then(|| blocked_by.to_owned()),
        what,
    }
}

/// The ONE deadline slot `cap` owns, or `None` when it owns none or several
/// (a capability whose time lives in two cells cannot be credited without
/// choosing between them, so it is never a CR-F endpoint).
#[must_use]
pub fn unique_slot(cap: &str) -> Option<LaneSlot> {
    let mut slots = SLOT_CAPS
        .iter()
        .filter(|(_, caps)| caps.contains(&cap))
        .map(|(slot, _)| *slot);
    let first = slots.next()?;
    slots.next().is_none().then_some(first)
}

/// CR-F's routes on a row: every capability in the plan's view whose own
/// [`far_recipient`] exists, mapped to cells — both endpoints own exactly one
/// slot, the slots differ, and CR-A governs both (pure).
#[must_use]
pub fn far_routes(plan: &Plan, r2: &Commitments) -> Vec<FarRoute> {
    plan.caps
        .iter()
        .filter_map(|cap| {
            let to = far_recipient(plan, &cap.id).ok()?;
            let from_slot = unique_slot(&cap.id)?;
            let to_slot = unique_slot(&to.id)?;
            (from_slot != to_slot && r2.governed(from_slot) && r2.governed(to_slot)).then(|| {
                FarRoute {
                    from: cap.id.clone(),
                    to: to.id.clone(),
                    from_slot,
                    to_slot,
                }
            })
        })
        .collect()
}

/// The row's R4 commitments. Pure and total. `r2` is the router's acting
/// commitments (acting set 1) on the same row.
#[must_use]
pub fn commit(
    plan: &Plan,
    input: &PolicyInput,
    mode: EffectiveMode,
    switches: &R4Switches,
    r2: &Commitments,
) -> R4Commitments {
    let acting = matches!(mode, EffectiveMode::Acting);
    if !acting {
        let reason = format!("mode:{}", mode.token());
        let rules = R4_RULES
            .iter()
            .map(|rule| act(*rule, false, &reason, "router not acting".to_owned()))
            .collect();
        return finish(R4Commitments {
            schema: R4_SCHEMA.to_owned(),
            acting,
            rules,
            lease: None,
            far: None,
            sha: String::new(),
        });
    }
    let mut rules = Vec::with_capacity(R4_RULES.len());

    // CR-L: the memory lease against the declared host's static envelope.
    let plan_lease = lease::lease_plan(&input.host);
    let lease_acts = switches.lease && plan_lease.is_some();
    rules.push(act(
        CR_L,
        lease_acts,
        if switches.lease {
            "envelope_unknown"
        } else {
            "key"
        },
        plan_lease.map_or_else(
            || "no static envelope: every gate keeps today's live decision".to_owned(),
            |p| {
                format!(
                    "lease envelope={}B (declared host {} threads={}) terminal={}B heavy_lane={}B; \
                     live envelope stays the floor",
                    p.envelope_bytes,
                    p.host.token(),
                    p.threads,
                    p.terminal_bytes(),
                    lease::OPTIONAL_HEAVY_LEASE_BYTES
                )
            },
        ),
    ));

    // CR-F: the FAR hand-on, riding on CR-1's cells; never on a short row.
    let cr1 = r2.acts(CR_1);
    let eligible = switches.far_hand_on && cr1 && !plan.short_row;
    let routes = if eligible {
        far_routes(plan, r2)
    } else {
        Vec::new()
    };
    let far_acts = eligible && !routes.is_empty();
    rules.push(act(
        CR_F,
        far_acts,
        if !switches.far_hand_on {
            "key"
        } else if !cr1 {
            "CR-1"
        } else if plan.short_row {
            "short_row"
        } else {
            "no_sealed_recipient"
        },
        format!(
            "FAR (two trailing windows, work units) at c2/c3 hands the unspent cell to a STRICTLY \
             sealed recipient at the lane's own safe point; routes={}{}",
            routes.len(),
            if eligible && routes.is_empty() {
                " (observe only: classify and log)"
            } else {
                ""
            }
        ),
    ));

    finish(R4Commitments {
        schema: R4_SCHEMA.to_owned(),
        acting,
        rules,
        lease: if lease_acts { plan_lease } else { None },
        far: eligible.then_some(FarCommit {
            observe_only: routes.is_empty(),
            routes,
        }),
        sha: String::new(),
    })
}

/// Stamp the digest (over everything but the digest itself).
fn finish(mut commitments: R4Commitments) -> R4Commitments {
    commitments.sha = String::new();
    let mut hash = Sha256::new();
    hash.update(format!("{commitments:?}").as_bytes());
    let digest: [u8; 32] = hash.finalize().into();
    commitments.sha = super::hash::hex(&digest);
    commitments
}
