// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The PURE plan (ROUTER_PROGRAM §2.2 PLAN, §4.1, §4.2).
//!
//! `plan(&PolicyInput) -> Plan` has no clock, no environment, no filesystem
//! and no globals: the same input gives the same plan, bit for bit, which is
//! what `ny vnncomp plan --replay` checks. [`PolicyInput`] has no name, path
//! or profile field, so the plan is blind to them by type.
//!
//! In R1 the plan is DESCRIPTIVE (it predicts today's structural route and
//! caps, so a shadow run can be diffed against the `route=` receipts) plus
//! PRESCRIPTIVE `would` entries for the R2 rules (CR-C, CR-1, CR-H, CR-R,
//! CR-K). It EMITS NOTHING: [`Plan::emitted`] is always empty and no rule is
//! acting ([`super::ACTING_RULES`] is empty).
//!
//! All time is integer milliseconds (saturating), so plans serialize and
//! replay exactly.

use serde::{Deserialize, Serialize};

use super::checkpoint::CheckpointId;
use super::decl::{
    Admission, AdmissionRef, AdmitWhen, AmendmentSet, CapKind, CapabilityDecl, KnobRisk, LaneRole,
    LedgerStatus, OpSet, ProgressKind, RuleId, SafePoints, SpecSet, SrLevel, WorkUnit,
};
use super::facts::{HostClass, HostEnvelope, InstanceFacts};
use super::priors::{PriorView, RULE_PRIORS};
use super::ROUTER_SCHEMA;

/// CR-A: admission = own predicate ∧ ledger ∧ not pinned off ∧ first rung fits.
pub const CR_A: RuleId = RuleId("CR-A");
/// CR-P: today's structural order (screen → attack burst → provers).
pub const CR_P: RuleId = RuleId("CR-P");
/// CR-C: every non-terminal lane gets a contract ≥ its first rung.
pub const CR_C: RuleId = RuleId("CR-C");
/// CR-1: one cap authority (a monotone deadline cell per lane).
pub const CR_1: RuleId = RuleId("CR-1");
/// CR-T: the last admitted prover is terminal and takes the remainder.
pub const CR_T: RuleId = RuleId("CR-T");
/// CR-H: a lane that stops undecided hands its unspent contract on.
pub const CR_H: RuleId = RuleId("CR-H");
/// CR-R: publication reserve and publish grace.
pub const CR_R: RuleId = RuleId("CR-R");
/// CR-K: caps derived from the deadline and the memory lease.
pub const CR_K: RuleId = RuleId("CR-K");
/// CR-F: FAR hand-on at c2/c3 to a sealed recipient.
pub const CR_F: RuleId = RuleId("CR-F");
/// #router-r4 CR-L: the memory lease against the declared host's static
/// envelope (replaces the live headroom gate under `router.mode: on`).
pub const CR_L: RuleId = RuleId("CR-L");
/// CR-X: exposure (≤ 2 concurrent provers).
pub const CR_X: RuleId = RuleId("CR-X");
/// CR-S: short rows (B ≤ 30 s) take no active probe and no checkpoint action.
pub const CR_S: RuleId = RuleId("CR-S");
/// #router-r5 CR-RS: a capability is routed — configured on this row by the
/// router, not armed by a category file — only at SR.seal, where its own
/// precondition admits the row, its sealed envelope covers it, its parent's
/// admission holds (or is re-checked at the consumer's site) and the row did
/// not pin its keys.
pub const CR_RS: RuleId = RuleId("CR-RS");
/// #router-r5 CR-G: within a multi-objective lane, classes are funded in gap
/// order with a contract of at least one build plus two levels; a class whose
/// rung does not fit declines instead of half-running.
pub const CR_G: RuleId = RuleId("CR-G");

/// The publication reserve floor: `max(5 s, B/20)` (§2.3). The router never
/// plans below it.
pub const PUBLICATION_FLOOR_MS: u64 = 5_000;
/// Short-row ceiling (CR-S).
pub const SHORT_ROW_MS: u64 = 30_000;
/// Exposure ceiling (CR-X).
pub const MAX_CONCURRENT_PROVERS: usize = 2;
/// Publish-grace ceiling (CR-R: `0 < grace < 3 s` and below the join margin).
pub const PUBLISH_GRACE_CEILING_MS: u64 = 3_000;

/// The publication reserve for an official budget of `official_ms`.
#[must_use]
pub const fn publication_floor_ms(official_ms: u64) -> u64 {
    let twentieth = official_ms / 20;
    if twentieth > PUBLICATION_FLOOR_MS {
        twentieth
    } else {
        PUBLICATION_FLOOR_MS
    }
}

/// The budget as the policy sees it: STATIC values only (the live remaining
/// wall is an observation, logged separately and never hashed).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BudgetView {
    /// The scored competition budget.
    pub official_ms: u64,
    /// The internal verifier tier.
    pub internal_ms: u64,
    /// The concurrent-lane join margin.
    pub join_margin_ms: u64,
}

/// The kind of a capability, as plain owned data.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapClass {
    /// A lane.
    Lane(LaneRole),
    /// A tier inside `parent`.
    Tier {
        /// Parent lane.
        parent: String,
    },
    /// A phase feeding `lane`.
    Phase {
        /// The lane fed.
        lane: String,
    },
    /// A variant of a slot.
    Variant {
        /// The slot.
        slot: String,
    },
    /// A router / optimizer rule.
    Rule,
}

/// A ledger status, as plain owned data.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LedgerState {
    /// Not routable.
    Open(String),
    /// Shadow only.
    Shadow,
    /// Today's shipped route runs it on its envelope.
    SealedAsShipped,
    /// S7 + S8 done.
    Sealed,
}

impl LedgerState {
    /// Whether the router may act on the capability inside its envelope.
    #[must_use]
    pub const fn is_sealed(&self) -> bool {
        matches!(self, Self::SealedAsShipped | Self::Sealed)
    }
}

/// An admission answer, as plain owned data.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdmissionState {
    /// Admitted.
    Admit,
    /// Declined.
    Decline(String),
    /// Could not be evaluated (a refusal).
    Unknown(String),
}

impl AdmissionState {
    /// Whether the row is admitted.
    #[must_use]
    pub const fn is_admit(&self) -> bool {
        matches!(self, Self::Admit)
    }

    /// `admit`, `decline:<why>` or `unknown:<why>`.
    #[must_use]
    pub fn token(&self) -> String {
        match self {
            Self::Admit => "admit".to_owned(),
            Self::Decline(why) => format!("decline:{why}"),
            Self::Unknown(why) => format!("unknown:{why}"),
        }
    }
}

impl From<Admission> for AdmissionState {
    fn from(admission: Admission) -> Self {
        match admission {
            Admission::Admit => Self::Admit,
            Admission::Decline(why) => Self::Decline(why.to_owned()),
            Admission::Unknown(why) => Self::Unknown(why.to_owned()),
        }
    }
}

/// One capability as the policy sees it: the declaration's data plus the
/// row's admission answer.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CapInput {
    /// Identity.
    pub id: String,
    /// Kind.
    pub class: CapClass,
    /// Ledger status.
    pub ledger: LedgerState,
    /// Progress kind.
    pub progress: ProgressKind,
    /// Work unit.
    pub unit: WorkUnit,
    /// First-rung cost (`None` = unpriced: today's cap).
    pub first_rung_ms: Option<u64>,
    /// Safe points.
    pub safe_points: SafePoints,
    /// Amendments honoured.
    pub accepts: AmendmentSet,
    /// The row's admission answer from the lane's own predicate.
    pub admission: AdmissionState,
    /// #router-r5: the capability's routing data (`None`: never routed).
    #[serde(default)]
    pub route: Option<RouteSpec>,
}

/// #router-r5: a routable capability's routing data, as plain owned data
/// (ROUTER_PROGRAM §3.5, §10 R5). Static: it derives from the compiled
/// registry, so it is hashed with the rest of the input.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RouteSpec {
    /// The SR level, derived from the registry's S-ladder and the envelope.
    pub sr: SrLevel,
    /// The capability whose own admission the consumer sits inside.
    pub parent: Option<String>,
    /// The routing precondition (facts and host only).
    pub admit_when: AdmitWhen,
    /// Operator kinds the sealed envelope's evidence rows cover.
    pub envelope_ops: OpSet,
    /// Specification archetypes the sealed envelope's evidence rows cover.
    pub envelope_specs: SpecSet,
    /// The highest moat risk among the route's keys.
    pub risk: KnobRisk,
    /// `(typed key path, value)` a route applies, in declaration order.
    pub settings: Vec<(String, String)>,
    /// The exclusive slot the capability fills, if any.
    pub slot: Option<String>,
}

impl RouteSpec {
    /// Project a declaration's routing data (`None` when it is not routable).
    #[must_use]
    pub fn from_decl(decl: &CapabilityDecl) -> Option<Self> {
        let routable = decl.routable?;
        Some(Self {
            sr: decl.sr_level(),
            parent: match decl.admission {
                AdmissionRef::Parent(parent) => Some(parent.as_str().to_owned()),
                _ => None,
            },
            admit_when: routable.admit_when,
            envelope_ops: decl.envelope.ops,
            envelope_specs: decl.envelope.specs,
            risk: routable.route.risk,
            settings: routable
                .route
                .settings
                .iter()
                .map(|s| (s.key.to_owned(), s.value.to_owned()))
                .collect(),
            slot: routable.route.slot.map(str::to_owned),
        })
    }
}

impl CapInput {
    /// Project a declaration and its admission answer. `first_rung_ms` is the
    /// priced first rung (the priors resolve cost models; uniform v1 priors
    /// price nothing).
    #[must_use]
    pub fn from_decl(
        decl: &CapabilityDecl,
        admission: Admission,
        first_rung_ms: Option<u64>,
    ) -> Self {
        let class = match decl.kind {
            CapKind::Lane { role } => CapClass::Lane(role),
            CapKind::Tier { parent } => CapClass::Tier {
                parent: parent.as_str().to_owned(),
            },
            CapKind::Phase { lane, .. } => CapClass::Phase {
                lane: lane.as_str().to_owned(),
            },
            CapKind::Variant { slot, .. } => CapClass::Variant {
                slot: slot.to_owned(),
            },
            CapKind::Rule { .. } => CapClass::Rule,
        };
        let ledger = match decl.ledger {
            LedgerStatus::Open(why) => LedgerState::Open(why.to_owned()),
            LedgerStatus::Shadow => LedgerState::Shadow,
            LedgerStatus::SealedAsShipped => LedgerState::SealedAsShipped,
            LedgerStatus::Sealed => LedgerState::Sealed,
        };
        Self {
            id: decl.id.as_str().to_owned(),
            class,
            ledger,
            progress: decl.progress,
            unit: decl.unit,
            first_rung_ms,
            safe_points: decl.safe_points,
            accepts: decl.accepts,
            admission: admission.into(),
            route: RouteSpec::from_decl(decl),
        }
    }

    /// The lane role, when this capability is a lane.
    #[must_use]
    pub const fn role(&self) -> Option<LaneRole> {
        match self.class {
            CapClass::Lane(role) => Some(role),
            _ => None,
        }
    }
}

/// Everything the policy may read (§4.1). No name, no path, no profile.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PolicyInput {
    /// `ny-router/v1`.
    pub schema: String,
    /// Static row facts.
    pub facts: InstanceFacts,
    /// The host the plan is made against.
    pub host: HostEnvelope,
    /// The static budget view.
    pub budget: BudgetView,
    /// Every registered capability with its admission answer, registry order.
    pub capabilities: Vec<CapInput>,
    /// Explicitly pinned typed keys (sorted), for `blocked_by` accounting.
    pub pins: Vec<String>,
    /// The compiled priors table version (`uniform-v1` when no prior is read;
    /// `cost-v1+<sha8>` under #router-r7's `optimizer.priors: true`).
    pub priors_version: String,
    /// The compiled capability registry's digest.
    pub registry_version: String,
    /// #router-r7: what the compiled cost-model prior says about this row
    /// (`None` unless `optimizer.priors: true` and the table loaded). Absent
    /// from the serialized input and from the static hash when `None`, so an
    /// input without a prior is byte-identical to R5's.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub priors: Option<PriorView>,
}

/// One lane's contract.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Contract {
    /// The lane.
    pub cap: String,
    /// Committed milliseconds (`None` = unpriced: today's cap).
    pub ms: Option<u64>,
    /// Whether this lane is terminal (takes the remainder).
    pub terminal: bool,
}

/// The reserves the plan keeps out of every lane's reach.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Reserves {
    /// Publication reserve (≥ [`publication_floor_ms`]).
    pub r_pub_ms: u64,
    /// Concurrent-lane join margin.
    pub join_margin_ms: u64,
}

/// A planned checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CheckpointPlan {
    /// Which checkpoint.
    pub id: CheckpointId,
    /// When, in milliseconds since the row started.
    pub at_ms: u64,
    /// True for c1 (observation only) and on short rows.
    pub observe_only: bool,
}

/// The logged width plan (never a decision in v1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WidthPlan {
    /// Worker threads on the planned host.
    pub threads: u32,
    /// Host class.
    pub host: HostClass,
}

/// The logged memory plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MemPlan {
    /// The static envelope `E_static`.
    pub envelope_bytes: u64,
}

/// A router-emitted knob value. Always empty in R1.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EmittedValue {
    /// The typed key.
    pub key: String,
    /// The value token.
    pub value: String,
}

/// A prescriptive entry: what an R2+ rule WOULD do on this row.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Would {
    /// The rule.
    pub rule: String,
    /// What it would do.
    pub what: String,
}

/// A capability the plan keeps in view for checkpoints.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PlannedCap {
    /// Identity.
    pub id: String,
    /// Progress kind.
    pub progress: ProgressKind,
    /// Safe points.
    pub safe_points: SafePoints,
    /// Amendments honoured.
    pub accepts: AmendmentSet,
    /// Whether its ledger is sealed (`SealedAsShipped` or `Sealed`).
    pub sealed: bool,
    /// #router-r4: whether its ledger is STRICTLY `Sealed` (S7 + S8 done on
    /// its envelope). Only such a capability may RECEIVE a FAR hand-on
    /// (CR-F): `SealedAsShipped` is today's route on today's shapes, not
    /// evidence that it converts handed-on time (ROUTER_PROGRAM §1.1: "today
    /// none does").
    #[serde(default)]
    pub sealed_s7s8: bool,
    /// Whether it is a prover lane.
    pub prover: bool,
    /// Whether its own predicate admitted the row.
    pub admitted: bool,
}

/// The plan (§2.2).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Plan {
    /// `ny-router/v1`.
    pub schema: String,
    /// Capabilities whose own predicate admitted the row, registry order.
    pub admitted: Vec<String>,
    /// `(capability, reason)` for every refusal (decline, unknown, unfit rung).
    pub declined: Vec<(String, String)>,
    /// Lane order (CR-P).
    pub order: Vec<String>,
    /// The terminal lane (CR-T).
    pub terminal: Option<String>,
    /// Contracts (CR-C), in order.
    pub contracts: Vec<Contract>,
    /// Reserves (CR-R).
    pub reserves: Reserves,
    /// The prover window: `official − r_pub − join` (saturating).
    pub window_ms: u64,
    /// Checkpoints (at most 3).
    pub checkpoints: Vec<CheckpointPlan>,
    /// Width plan (logged).
    pub width_plan: WidthPlan,
    /// Memory plan (logged).
    pub mem_plan: MemPlan,
    /// Router-emitted values. ALWAYS EMPTY IN R1.
    pub emitted: Vec<EmittedValue>,
    /// Rules that fired (descriptively).
    pub rules_fired: Vec<String>,
    /// Prescriptive entries.
    pub would: Vec<Would>,
    /// CR-S: a short row takes no checkpoint action.
    pub short_row: bool,
    /// Capabilities in view for checkpoints (every lane, tier and phase with
    /// a progress signal), registry order.
    pub caps: Vec<PlannedCap>,
    /// Hex sha256 of the static input ([`super::hash::static_hash`]).
    pub static_hash: String,
}

impl Plan {
    /// When checkpoint `id` falls, if planned.
    #[must_use]
    pub fn checkpoint_at_ms(&self, id: CheckpointId) -> Option<u64> {
        self.checkpoints
            .iter()
            .find(|c| c.id == id)
            .map(|c| c.at_ms)
    }

    /// The first 8 hex digits of the static hash.
    #[must_use]
    pub fn sha8(&self) -> &str {
        self.static_hash.get(..8).unwrap_or(&self.static_hash)
    }
}

/// Structural rank of a lane in today's order (CR-P).
const fn lane_rank(role: LaneRole) -> u8 {
    match role {
        LaneRole::Screen => 0,
        LaneRole::Attack => 1,
        LaneRole::Prover { .. } => 2,
    }
}

/// #router-r7: the prior's planning entries — the per-phase predictions,
/// what it priced, and the cheapest-first order of the contracted lanes within
/// each structural rank (logged for the O-TRIAL step; CR-P keeps today's
/// order). Logged, never emitted.
fn prior_would(view: &PriorView, lanes: &[(u8, usize, &CapInput)], threads: u32) -> Vec<Would> {
    let rule = RULE_PRIORS.as_str().to_owned();
    let mut out = Vec::new();
    let priced = view
        .priced
        .iter()
        .map(|p| format!("{}:{}ms", p.cap, p.ms))
        .collect::<Vec<_>>()
        .join(",");
    let unpriced = view
        .unpriced
        .iter()
        .map(|u| format!("{}:{}", u.cap, u.reason))
        .collect::<Vec<_>>()
        .join(",");
    out.push(Would {
        rule: rule.clone(),
        what: format!(
            "priors={} host={} cell={} priced=[{}] unpriced=[{}]",
            view.version,
            view.host,
            view.cell.as_deref().unwrap_or("unknown"),
            if priced.is_empty() {
                "-"
            } else {
                priced.as_str()
            },
            if unpriced.is_empty() {
                "-"
            } else {
                unpriced.as_str()
            },
        ),
    });
    for p in &view.predictions {
        out.push(Would {
            rule: rule.clone(),
            what: format!(
                "predict {} median={}ms upper={}ms cell={} threads={threads}",
                p.phase, p.median_ms, p.upper_ms, p.cell
            ),
        });
    }
    let mut ranked: Vec<(u8, u64, usize, &str)> = lanes
        .iter()
        .filter_map(|(rank, index, cap)| {
            cap.first_rung_ms
                .map(|ms| (*rank, ms, *index, cap.id.as_str()))
        })
        .collect();
    ranked.sort_unstable();
    if !ranked.is_empty() {
        out.push(Would {
            rule,
            what: format!(
                "cost order within rank (cheapest first rung): {}",
                ranked
                    .iter()
                    .map(|(_, ms, _, id)| format!("{id}={ms}ms"))
                    .collect::<Vec<_>>()
                    .join("<")
            ),
        });
    }
    out
}

/// The plan for one row. Pure and total.
#[must_use]
pub fn plan(input: &PolicyInput) -> Plan {
    let official = input.budget.official_ms;
    let join = input.budget.join_margin_ms;
    let r_pub = publication_floor_ms(official);
    let window = official.saturating_sub(r_pub).saturating_sub(join);
    let short_row = official <= SHORT_ROW_MS;

    let mut admitted = Vec::new();
    let mut declined = Vec::new();
    for cap in &input.capabilities {
        if cap.admission.is_admit() {
            admitted.push(cap.id.clone());
        } else {
            declined.push((cap.id.clone(), cap.admission.token()));
        }
    }

    // CR-P: today's structural order, stable within a rank (registry order).
    let mut lanes: Vec<(u8, usize, &CapInput)> = input
        .capabilities
        .iter()
        .enumerate()
        .filter(|(_, cap)| cap.admission.is_admit())
        .filter_map(|(index, cap)| cap.role().map(|role| (lane_rank(role), index, cap)))
        .collect();
    lanes.sort_by_key(|(rank, index, _)| (*rank, *index));

    // CR-T: the last admitted prover that may be terminal.
    let terminal = lanes
        .iter()
        .rev()
        .find(|(_, _, cap)| matches!(cap.role(), Some(LaneRole::Prover { terminal_ok: true })))
        .map(|(_, _, cap)| cap.id.clone());

    // #router-r7: under the offline prior, a lane whose own predicate runs at
    // its site (admission `Unknown`: deferred) but whose first rung the prior
    // PRICED is contracted too, so a rung that cannot fit the window declines
    // it here and R2's CR-A leaves that slot on today's deadline. Never the
    // terminal (CR-T reads admitted lanes only), never counted by CR-X. With
    // no prior the contract set IS `lanes`, as in R5.
    let contract_lanes: Vec<(u8, usize, &CapInput)> = if input.priors.is_some() {
        let mut out: Vec<(u8, usize, &CapInput)> = input
            .capabilities
            .iter()
            .enumerate()
            .filter(|(_, cap)| {
                cap.admission.is_admit()
                    || (matches!(cap.admission, AdmissionState::Unknown(_))
                        && cap.first_rung_ms.is_some())
            })
            .filter_map(|(index, cap)| cap.role().map(|role| (lane_rank(role), index, cap)))
            .collect();
        out.sort_by_key(|(rank, index, _)| (*rank, *index));
        out
    } else {
        lanes.clone()
    };

    // CR-C: contracts ≥ first rung; a rung that does not fit declines.
    let mut left = window;
    let mut order = Vec::new();
    let mut contracts = Vec::new();
    for (_, _, cap) in &contract_lanes {
        if terminal.as_deref() == Some(cap.id.as_str()) {
            continue;
        }
        match cap.first_rung_ms {
            Some(rung) if rung <= left => {
                left -= rung;
                order.push(cap.id.clone());
                contracts.push(Contract {
                    cap: cap.id.clone(),
                    ms: Some(rung),
                    terminal: false,
                });
            }
            Some(rung) => declined.push((
                cap.id.clone(),
                format!("rung_does_not_fit:{rung}ms>{left}ms"),
            )),
            None => {
                order.push(cap.id.clone());
                contracts.push(Contract {
                    cap: cap.id.clone(),
                    ms: None,
                    terminal: false,
                });
            }
        }
    }
    if let Some(term) = &terminal {
        order.push(term.clone());
        contracts.push(Contract {
            cap: term.clone(),
            ms: Some(left),
            terminal: true,
        });
    }

    let checkpoints = vec![
        CheckpointPlan {
            id: CheckpointId::C1,
            at_ms: window / 4,
            observe_only: true,
        },
        CheckpointPlan {
            id: CheckpointId::C2,
            at_ms: window / 2,
            observe_only: short_row,
        },
        CheckpointPlan {
            id: CheckpointId::C3,
            at_ms: window / 4 * 3 + (window % 4) * 3 / 4,
            observe_only: short_row,
        },
    ];

    let caps: Vec<PlannedCap> = input
        .capabilities
        .iter()
        .filter(|cap| !matches!(cap.class, CapClass::Rule))
        .map(|cap| PlannedCap {
            id: cap.id.clone(),
            progress: cap.progress,
            safe_points: cap.safe_points,
            accepts: cap.accepts,
            sealed: cap.ledger.is_sealed(),
            sealed_s7s8: matches!(cap.ledger, LedgerState::Sealed),
            prover: matches!(cap.role(), Some(LaneRole::Prover { .. })),
            admitted: cap.admission.is_admit(),
        })
        .collect();

    let provers = lanes
        .iter()
        .filter(|(_, _, cap)| matches!(cap.role(), Some(LaneRole::Prover { .. })))
        .count();
    let sealed = input
        .capabilities
        .iter()
        .filter(|cap| cap.ledger.is_sealed())
        .count();

    let mut rules_fired = vec![
        CR_A.as_str().to_owned(),
        CR_P.as_str().to_owned(),
        CR_C.as_str().to_owned(),
        CR_R.as_str().to_owned(),
        CR_X.as_str().to_owned(),
    ];
    if terminal.is_some() {
        rules_fired.push(CR_T.as_str().to_owned());
    }
    if short_row {
        rules_fired.push(CR_S.as_str().to_owned());
    }

    let mut would = vec![
        Would {
            rule: CR_A.as_str().to_owned(),
            what: format!(
                "admitted={} of {}; ledger sealed={sealed} (R2 gates admission on the ledger)",
                admitted.len(),
                input.capabilities.len()
            ),
        },
        Would {
            rule: CR_1.as_str().to_owned(),
            what: format!("one monotone deadline cell per lane: lanes={}", order.len()),
        },
    ];
    for contract in &contracts {
        if contract.terminal {
            continue;
        }
        would.push(Would {
            rule: CR_C.as_str().to_owned(),
            what: match contract.ms {
                Some(ms) => format!("contract {}={ms}ms (first rung)", contract.cap),
                None => format!("contract {}=unpriced (today's cap)", contract.cap),
            },
        });
    }
    if order.len() > 1 {
        would.push(Would {
            rule: CR_H.as_str().to_owned(),
            what: format!("hand unspent contract on along {}", order.join(">")),
        });
    }
    would.push(Would {
        rule: CR_R.as_str().to_owned(),
        what: format!(
            "r_pub={r_pub}ms (floor max({PUBLICATION_FLOOR_MS}, B/20)); publish grace 0<g<{}ms inside join={join}ms",
            PUBLISH_GRACE_CEILING_MS.min(join)
        ),
    });
    would.push(Would {
        rule: CR_K.as_str().to_owned(),
        what: format!(
            "derive depth/domain/queue caps from window={window}ms and lease={}B",
            input.host.envelope_bytes()
        ),
    });
    would.push(Would {
        rule: CR_X.as_str().to_owned(),
        what: format!(
            "admitted provers={provers} (<= {MAX_CONCURRENT_PROVERS} concurrent: {})",
            if provers <= MAX_CONCURRENT_PROVERS {
                "ok"
            } else {
                "sequential"
            }
        ),
    });
    if short_row {
        would.push(Would {
            rule: CR_S.as_str().to_owned(),
            what: format!("B={official}ms <= {SHORT_ROW_MS}ms: no probes, no checkpoint action"),
        });
    }
    if let Some(view) = &input.priors {
        rules_fired.push(RULE_PRIORS.as_str().to_owned());
        would.extend(prior_would(view, &contract_lanes, input.host.threads));
    }

    Plan {
        schema: ROUTER_SCHEMA.to_owned(),
        admitted,
        declined,
        order,
        terminal,
        contracts,
        reserves: Reserves {
            r_pub_ms: r_pub,
            join_margin_ms: join,
        },
        window_ms: window,
        checkpoints,
        width_plan: WidthPlan {
            threads: input.host.threads,
            host: input.host.class,
        },
        mem_plan: MemPlan {
            envelope_bytes: input.host.envelope_bytes(),
        },
        emitted: Vec::new(),
        rules_fired,
        would,
        short_row,
        caps,
        static_hash: super::hash::hex(&super::hash::static_hash(input)),
    }
}
