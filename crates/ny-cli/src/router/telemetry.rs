// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The router's observables (ROUTER_PROGRAM §7): the `[router]` /
//! `[optimizer]` stderr lines and the bounded flight `router` record.
//!
//! Every line is rendered by a pure function so tests pin the exact grammar
//! the integrator greps for; printing happens in `super` inside `fail_open`.

use ny_core::router::bus::{BusStats, SampleRecord};
use ny_core::router::checkpoint::{Amendment, CapObservation, CheckpointId};
use ny_core::router::facts::{FactsStatus, HostEnvelope, InstanceFacts};
use ny_core::router::plan::{CapInput, LedgerState, Plan, PolicyInput};
use ny_core::router::EffectiveMode;
use serde::{Deserialize, Serialize};

use super::RouterPolicy;

/// Samples a flight record may carry across all checkpoints (§7: bounded).
pub(crate) const MAX_RECORD_SAMPLES: usize = 256;

/// One retrospective checkpoint as recorded (and replayed).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct CheckpointRecord {
    /// Which checkpoint.
    pub(crate) id: CheckpointId,
    /// When it falls, milliseconds since the row started.
    pub(crate) at_ms: u64,
    /// The samples the checkpoint read (its trailing selection).
    pub(crate) samples: Vec<SampleRecord>,
    /// Whether `samples` was cut to stay inside [`MAX_RECORD_SAMPLES`] (such
    /// a checkpoint is not replayable).
    pub(crate) truncated: bool,
    /// Per-capability classifications.
    pub(crate) observations: Vec<CapObservation>,
    /// What CR-F would have done.
    pub(crate) amendments: Vec<Amendment>,
}

/// One optimizer rule's retrospective verdict.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct OptimizerRecord {
    /// The rule.
    pub(crate) rule: String,
    /// The capability it watched.
    pub(crate) site: String,
    /// `stop` or `none`.
    pub(crate) would: String,
    /// The unit index it would have stopped at.
    pub(crate) at: Option<u64>,
    /// Samples read.
    pub(crate) samples: u64,
}

/// The `[router] done` summary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct DoneRecord {
    /// Faults contained over the whole row.
    pub(crate) faults: u64,
    /// Bus counters.
    pub(crate) bus: BusStats,
    /// Samples drained after publication.
    pub(crate) drained: u64,
    /// The engine's own terminal stop reason (`none-recorded` when absent).
    pub(crate) stop_reason: String,
    /// The published verdict token.
    pub(crate) verdict: String,
    /// Always true: checkpoints run strictly after publication.
    pub(crate) published_before_ckpt: bool,
    /// Digest of every checkpoint decision (`sha8`).
    pub(crate) decisions_sha: String,
}

/// The flight `router` record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct RouterRecord {
    /// `ny-router/v1`.
    pub(crate) schema: String,
    /// Requested router mode.
    pub(crate) mode: String,
    /// What it does in this binary.
    pub(crate) effective: String,
    /// Requested optimizer mode.
    pub(crate) optimizer: String,
    /// What it does in this binary.
    pub(crate) optimizer_effective: String,
    /// `frozen` (the receipt's LeverSet) or `entry_env` (preset invalid).
    pub(crate) authority: String,
    /// When the session started, milliseconds since the row started (an
    /// observation, never hashed).
    pub(crate) started_ms: u64,
    /// The static policy input.
    pub(crate) input: PolicyInput,
    /// The plan.
    pub(crate) plan: Plan,
    /// #router-r2: the rule switches the session resolved (`None` unless the
    /// router acted).
    #[serde(default)]
    pub(crate) switches: Option<ny_core::router::acting::RuleSwitches>,
    /// #router-r2: the pure acting commitments (`None` unless the router
    /// acted); `ny vnncomp plan --replay` re-derives them bit for bit.
    #[serde(default)]
    pub(crate) commitments: Option<ny_core::router::acting::Commitments>,
    /// #router-r2: what the acting rules did — lane events, derived values,
    /// the deadline authority's final snapshot (cells, contracts, hand-ons).
    #[serde(default)]
    pub(crate) acting: Option<super::acting::ActingRecord>,
    /// #router-r3: the four O-STOP rule switches the session resolved
    /// (`None` unless the optimizer acted).
    #[serde(default)]
    pub(crate) optimizer_switches: Option<ny_core::router::ostop::OptimizerSwitches>,
    /// #router-r3: the pure O-STOP commitments (`None` unless the optimizer
    /// acted); `ny vnncomp plan --replay` re-derives them bit for bit.
    #[serde(default)]
    pub(crate) stop_commitments: Option<ny_core::router::ostop::StopCommitments>,
    /// #router-r3: what the O-STOP rules did — yields, hand-ons, defers and
    /// root admissions (the stop authority's final snapshot).
    #[serde(default)]
    pub(crate) stop: Option<super::ostop::StopRecord>,
    /// #router-r4: the two acting-set-3 switches the session resolved
    /// (`None` unless the router acted).
    #[serde(default)]
    pub(crate) r4_switches: Option<ny_core::router::r4::R4Switches>,
    /// #router-r4: the pure acting-set-3 commitments (CR-L lease plan, CR-F
    /// routes); `ny vnncomp plan --replay` re-derives them bit for bit.
    #[serde(default)]
    pub(crate) r4_commitments: Option<ny_core::router::r4::R4Commitments>,
    /// #router-r4: the memory lease authority's final snapshot (leases held,
    /// granted, denied, released; peak against the envelope).
    #[serde(default)]
    pub(crate) lease: Option<super::lease::LeaseRecord>,
    /// #router-r4: the FAR authority's final snapshot (every live checkpoint
    /// decision with the samples it read, every refused suspension).
    #[serde(default)]
    pub(crate) far: Option<super::far::FarRecord>,
    /// #router-r5: the two acting-set-4 switches the session resolved
    /// (`None` unless the router acted).
    #[serde(default)]
    pub(crate) r5_switches: Option<ny_core::router::route::R5Switches>,
    /// #router-r5: what CR-RS read beyond the static input (the typed keys
    /// the row pinned), recorded so a replay re-derives the routes.
    #[serde(default)]
    pub(crate) r5_input: Option<ny_core::router::route::R5Input>,
    /// #router-r5: the pure acting-set-4 commitments (one route decision per
    /// routable capability, CR-G's flag); `ny vnncomp plan --replay`
    /// re-derives them bit for bit.
    #[serde(default)]
    pub(crate) r5_commitments: Option<ny_core::router::route::R5Commitments>,
    /// #router-r5: the route authority's final snapshot (routed capabilities
    /// and their consults, settings applied at configuration boundaries, the
    /// tiers CR-G funded).
    #[serde(default)]
    pub(crate) route: Option<super::route::RouteRecord>,
    /// #router-r6: the four O-TRIAL switches the session resolved (`None`
    /// unless the optimizer acted).
    #[serde(default)]
    pub(crate) trial_switches: Option<ny_core::router::otrial::TrialSwitches>,
    /// #router-r6: what the trial commit read beyond the static input (the
    /// trial knobs the row pinned), recorded so a replay re-derives it.
    #[serde(default)]
    pub(crate) r6_input: Option<ny_core::router::otrial::R6Input>,
    /// #router-r6: the pure O-TRIAL commitments (one decision per rule and
    /// per site); `ny vnncomp plan --replay` re-derives them bit for bit.
    #[serde(default)]
    pub(crate) trial_commitments: Option<ny_core::router::otrial::TrialCommitments>,
    /// #router-r6: the trial authority's final snapshot (every trial event).
    #[serde(default)]
    pub(crate) trial: Option<super::otrial::TrialRecord>,
    /// Retrospective checkpoints (empty until publication).
    pub(crate) checkpoints: Vec<CheckpointRecord>,
    /// Retrospective optimizer rules.
    pub(crate) optimizer_rules: Vec<OptimizerRecord>,
    /// The done summary (`None` until publication).
    pub(crate) done: Option<DoneRecord>,
}

/// `facts=<sha8>` for known facts, `facts=unknown:<why>` otherwise.
pub(crate) fn facts_token(facts: &InstanceFacts) -> String {
    match &facts.status {
        FactsStatus::Known => {
            let digest = sha256(&format!("{facts:?}"));
            ny_core::router::hash::sha8(&digest)
        }
        FactsStatus::Unknown(why) => format!("unknown:{why}"),
    }
}

fn sha256(text: &str) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut hash = Sha256::new();
    hash.update(text.as_bytes());
    hash.finalize().into()
}

/// `sha8` of any serializable value's canonical JSON.
pub(crate) fn digest_of<T: Serialize>(value: &T) -> String {
    let text = serde_json::to_string(value).unwrap_or_default();
    ny_core::router::hash::sha8(&sha256(&text))
}

/// `[router] v1 mode=... effective=...` — the session banner.
pub(crate) fn banner_line(
    policy: RouterPolicy,
    effective: EffectiveMode,
    optimizer: EffectiveMode,
    authority: &str,
    input: &PolicyInput,
) -> String {
    let host: &HostEnvelope = &input.host;
    format!(
        "[router] v1 mode={} effective={} optimizer={} optimizer_effective={} authority={authority} \
         facts={} host={} threads={} env_gib={} dev={} declared_host={} priors={} registry={} \
         spec={}",
        policy.mode.token(),
        effective.token(),
        policy.optimizer.token(),
        optimizer.token(),
        facts_token(&input.facts),
        host.class.token(),
        host.threads,
        host.envelope_bytes() >> 30,
        host.device.token(),
        u8::from(host.declared),
        input.priors_version,
        input.registry_version,
        // #admission-precision: the archetype the router read at start (the
        // new fact's own observable; `-` when it read none).
        input.facts.spec.map_or("-", |spec| spec.token()),
    )
}

/// #router-r7: `[router] priors v=<cost-v1+sha8> host=<class> cell=<key|unknown>
/// priced=<cap:ms,..|-> unpriced=<cap:why,..|-> predict=<phase:median/upper,..|->`.
pub(crate) fn priors_line(view: &ny_core::router::priors::PriorView) -> String {
    let join = |items: Vec<String>| {
        if items.is_empty() {
            "-".to_owned()
        } else {
            items.join(",")
        }
    };
    format!(
        "[router] priors v={} host={} cell={} priced={} unpriced={} predict={}",
        view.version,
        view.host,
        view.cell.as_deref().unwrap_or("unknown"),
        join(
            view.priced
                .iter()
                .map(|p| format!("{}:{}ms", p.cap, p.ms))
                .collect()
        ),
        join(
            view.unpriced
                .iter()
                .map(|u| format!("{}:{}", u.cap, u.reason))
                .collect()
        ),
        join(
            view.predictions
                .iter()
                .map(|p| format!("{}:{}/{}ms", p.phase, p.median_ms, p.upper_ms))
                .collect()
        ),
    )
}

/// #router-r7: `[router] priors unavailable why=<...>` — the key asked for the
/// prior and the compiled table could not be read: nothing is priced.
pub(crate) fn priors_unavailable_line(why: &str) -> String {
    format!("[router] priors unavailable why={why}")
}

fn ledger_token(ledger: &LedgerState) -> String {
    match ledger {
        LedgerState::Open(why) => format!("open({why})"),
        LedgerState::Shadow => "shadow".to_owned(),
        LedgerState::SealedAsShipped => "sealed_as_shipped".to_owned(),
        LedgerState::Sealed => "sealed".to_owned(),
    }
}

/// `[router] cap=<id> admit=<...> ledger=<...> rung=<...>`.
pub(crate) fn cap_line(cap: &CapInput) -> String {
    format!(
        "[router] cap={} admit={} ledger={} rung={}",
        cap.id,
        cap.admission.token(),
        ledger_token(&cap.ledger),
        cap.first_rung_ms
            .map_or_else(|| "unknown".to_owned(), |ms| format!("{ms}ms")),
    )
}

/// `[router] plan sha=...` — the whole plan in one line.
pub(crate) fn plan_line(plan: &Plan) -> String {
    let contracts = plan
        .contracts
        .iter()
        .map(|c| match c.ms {
            Some(ms) => format!("{}:{ms}ms{}", c.cap, if c.terminal { "*" } else { "" }),
            None => format!("{}:today", c.cap),
        })
        .collect::<Vec<_>>()
        .join(",");
    let checkpoints = plan
        .checkpoints
        .iter()
        .map(|c| format!("{}:{}ms", c.id.token(), c.at_ms))
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "[router] plan sha={} order={} terminal={} contracts={} r_pub={}ms join={}ms window={}ms \
         ckpt=[{checkpoints}] short_row={} rules={} admitted={} declined={} emitted={}",
        plan.sha8(),
        if plan.order.is_empty() {
            "-".to_owned()
        } else {
            plan.order.join(">")
        },
        plan.terminal.as_deref().unwrap_or("-"),
        if contracts.is_empty() {
            "-".to_owned()
        } else {
            contracts
        },
        plan.reserves.r_pub_ms,
        plan.reserves.join_margin_ms,
        plan.window_ms,
        u8::from(plan.short_row),
        plan.rules_fired.join(","),
        plan.admitted.len(),
        plan.declined.len(),
        plan.emitted.len(),
    )
}

fn fmt_rate(rate: Option<f64>) -> String {
    rate.map_or_else(|| "-".to_owned(), |r| format!("{r:.3}"))
}

fn amendment_token(amendment: &Amendment) -> String {
    match amendment {
        Amendment::HandOn { from, to, ms } => format!("hand_on({from}->{to},{ms}ms)"),
        Amendment::Defer { from, ms } => format!("defer({from},{ms}ms)"),
        Amendment::ArmPublishGrace { cap } => format!("publish_grace({cap})"),
        Amendment::DeriveCap { cap, what, value } => format!("derive({cap},{what}={value})"),
    }
}

/// `[router] ckpt=c2 t=... <id>:{...} would=...`.
pub(crate) fn checkpoint_line(record: &CheckpointRecord) -> String {
    let observed = record
        .observations
        .iter()
        .map(|o| {
            format!(
                "{}:{{kind={} w1={} w2={} open={} class={}}}",
                o.cap,
                o.kind.token(),
                fmt_rate(o.w1),
                fmt_rate(o.w2),
                o.open.map_or_else(|| "-".to_owned(), |n| n.to_string()),
                o.class.token(),
            )
        })
        .collect::<Vec<_>>();
    let would = if record.amendments.is_empty() {
        "none".to_owned()
    } else {
        record
            .amendments
            .iter()
            .map(amendment_token)
            .collect::<Vec<_>>()
            .join(",")
    };
    format!(
        "[router] ckpt={} t={:.3}s observed={} {} would={would}{}",
        record.id.token(),
        record.at_ms as f64 / 1000.0,
        observed.len(),
        observed.join(" "),
        if record.truncated { " truncated=1" } else { "" },
    )
}

/// `[optimizer] rule=<id> would=<stop|none> site=<cap> ...`.
pub(crate) fn optimizer_line(record: &OptimizerRecord) -> String {
    format!(
        "[optimizer] rule={} would={} site={} value={} samples={}",
        record.rule,
        record.would,
        record.site,
        record
            .at
            .map_or_else(|| "-".to_owned(), |at| format!("iter={at}")),
        record.samples,
    )
}

/// `[router] done faults=... bus_pushed=...` — on every exit path that records
/// a verdict, after publication.
pub(crate) fn done_line(done: &DoneRecord) -> String {
    format!(
        "[router] done faults={} bus_pushed={} bus_dropped={} bus_overwritten={} drained={} \
         verdict={} stop_reason={} published_before_ckpt={} decisions_sha={}",
        done.faults,
        done.bus.pushed,
        done.bus.dropped,
        done.bus.overwritten,
        done.drained,
        done.verdict,
        done.stop_reason.replace(char::is_whitespace, "_"),
        u8::from(done.published_before_ckpt),
        done.decisions_sha,
    )
}

/// `[router] fault site=... disabled=1` — a contained panic.
pub(crate) fn fault_line(fault: &ny_core::router::guard::RouterFault) -> String {
    format!(
        "[router] fault site={} disabled=1 skipped={} message={}",
        fault.site,
        u8::from(fault.skipped),
        fault.message.replace(char::is_whitespace, "_"),
    )
}
