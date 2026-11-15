// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The runtime router and optimizer modes (#router-r1, ROUTER_PROGRAM §11).
//!
//! Two typed keys, `router.mode` and `optimizer.mode` (`off | shadow | on`,
//! default `off`), delivered by preset through the generic axis projection
//! into the frozen scored-run `LeverSet` and read ONCE by
//! `ny-cli::router::RouterPolicy::from_levers`. No shipped yaml sets either
//! key, and no category yaml ever arms the router: rollout is a compiled
//! default move after a both-year moat (§2.5).
//!
//! R1 changes no verdict path. `shadow` extracts static facts, computes and
//! logs the plan, drains the write-only progress bus AFTER publication and
//! logs what the checkpoints and the optimizer rules would have done.
//!
//! #router-r2 (ROUTER_PROGRAM §10 R2): `on` now ACTS through acting set 1,
//! seven rules, each its own capability with its own typed ablation key
//! `router.rules.<rule>` declared below (default `true` = participates when
//! the router acts; `router.mode` stays default `off`, so no default arms
//! anything). The keys are deliberately withheld from search (`UNSAFE_AXES`):
//! a rule joins or leaves the shipped acting set by a compiled move after its
//! moat, never by a searched value.
//!
//! #router-r3 (ROUTER_PROGRAM §10 R3): `optimizer.mode: on` now ACTS through
//! acting set 2, the four O-STOP rules, each its own capability with its own
//! typed ablation key `optimizer.rules.<rule>` declared below (default `true`
//! = participates when the optimizer acts; `optimizer.mode` stays default
//! `off`). Same delivery and the same UNSAFE_AXES withholding as the R2 keys.
//!
//! #router-r4 (ROUTER_PROGRAM §10 R4): `router.mode: on` also ACTS through
//! acting set 3 — CR-L (the memory lease) and CR-F (the FAR hand-on) — each
//! its own capability with its own typed ablation key `router.rules.lease` /
//! `router.rules.far_hand_on` declared below. Same delivery, same default
//! (`true` = participates when the router acts) and the same UNSAFE_AXES
//! withholding as the R2 keys.
//!
//! #router-r5 (ROUTER_PROGRAM §10 R5): `router.mode: on` also ACTS through
//! acting set 4 — CR-RS (capabilities ROUTED per instance, only at SR.seal)
//! and CR-G (the gap-order contract inside multi-objective lanes) — each its
//! own capability with its own typed ablation key `router.rules.route` /
//! `router.rules.gap_contract` declared below. Same delivery, same default
//! and the same UNSAFE_AXES withholding as the R2 keys.
//!
//! #router-r6 (ROUTER_PROGRAM §10 R6): `optimizer.mode: on` also ACTS through
//! the four O-TRIAL rules — `rule.trial_width` (AIMD on a fast lane's width),
//! `rule.trial_lr_backoff` (the root ascent's step backoff),
//! `rule.trial_split_restart` (one restart with the other split heuristic)
//! and `rule.trial_attack_halving` (successive halving over the upfront
//! attack's step pool) — each its own capability with its own typed ablation
//! key `optimizer.rules.trial_*` declared below. Same delivery, same default
//! (`true` = participates when the optimizer acts) and the same UNSAFE_AXES
//! withholding as the R3 keys.
//!
//! #router-r7 (ROUTER_PROGRAM §5.5, §10 R7): `optimizer.priors` (default
//! `false`) lets the router READ the compiled offline cost-model prior
//! (`ny_core::router::priors`) for planning — first rungs for CR-C contracts,
//! per-phase predictions logged — and only on the host class the table was
//! measured on. Same delivery and the same UNSAFE_AXES withholding as the
//! rule keys; unlike them it defaults OFF, so even `router.mode: on` reads no
//! prior unless the key is set.

use crate::{
    declare_levers, Bucket, DefaultSpec, LeverDecl, LeverKind, MoatRisk, Provenance, ReaderSite,
    Scope,
};

const ROUTER_SCOPE: Scope = Scope {
    package: "ny-cli",
    subsystem: "router",
};

const ROUTER_MODE_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: ROUTER_SCOPE,
        role: "project router.mode into the frozen scored-run LeverSet",
        site: "crates/ny-cli/src/commands/vnncomp.rs:preset_delivered_config_value",
    },
    ReaderSite {
        scope: ROUTER_SCOPE,
        role: "project router.mode into the RELATIONAL route's frozen LeverSet (that route \
               freezes only what it executes, and it runs the router)",
        site: "crates/ny-cli/src/commands/vnncomp.rs:relational_router_config_value",
    },
    ReaderSite {
        scope: ROUTER_SCOPE,
        role: "resolve the router mode from the frozen LeverSet once per run and start the \
               shadow session (facts, plan, bus) only when it is not off",
        site: "crates/ny-cli/src/router/mod.rs:RouterPolicy::from_levers",
    },
    ReaderSite {
        scope: ROUTER_SCOPE,
        role: "#router-r2: `on` is effective `acting` (ACTING_RULES is acting set 1); the \
               session commits the seven rules and arms the one deadline authority",
        site: "crates/ny-cli/src/router/acting.rs:arm",
    },
];

const OPTIMIZER_MODE_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: ROUTER_SCOPE,
        role: "project optimizer.mode into the frozen scored-run LeverSet",
        site: "crates/ny-cli/src/commands/vnncomp.rs:preset_delivered_config_value",
    },
    ReaderSite {
        scope: ROUTER_SCOPE,
        role: "project optimizer.mode into the RELATIONAL route's frozen LeverSet",
        site: "crates/ny-cli/src/commands/vnncomp.rs:relational_router_config_value",
    },
    ReaderSite {
        scope: ROUTER_SCOPE,
        role: "resolve the optimizer mode from the frozen LeverSet once per run; shadow logs \
               what the O-STOP rules would do after publication",
        site: "crates/ny-cli/src/router/mod.rs:RouterPolicy::from_levers",
    },
    ReaderSite {
        scope: ROUTER_SCOPE,
        role: "#router-r3: `on` is effective `acting` (ACTING_OPTIMIZER_RULES is acting set \
               2); the session commits the four O-STOP rules and arms the stop authority",
        site: "crates/ny-cli/src/router/ostop.rs:arm",
    },
];

const OPTIMIZER_RULE_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: ROUTER_SCOPE,
        role: "project optimizer.rules.* into the frozen scored-run LeverSet",
        site: "crates/ny-cli/src/commands/vnncomp.rs:preset_delivered_config_value",
    },
    ReaderSite {
        scope: ROUTER_SCOPE,
        role: "project optimizer.rules.* into the RELATIONAL route's frozen LeverSet",
        site: "crates/ny-cli/src/commands/vnncomp.rs:relational_router_config_value",
    },
    ReaderSite {
        scope: ROUTER_SCOPE,
        role: "resolve the four O-STOP rule switches from the frozen LeverSet once per run \
               (ny_core::router::ostop::OptimizerSwitches)",
        site: "crates/ny-cli/src/router/mod.rs:RouterPolicy::from_levers",
    },
];

const ROUTER_RULE_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: ROUTER_SCOPE,
        role: "project router.rules.* into the frozen scored-run LeverSet",
        site: "crates/ny-cli/src/commands/vnncomp.rs:preset_delivered_config_value",
    },
    ReaderSite {
        scope: ROUTER_SCOPE,
        role: "project router.rules.* into the RELATIONAL route's frozen LeverSet",
        site: "crates/ny-cli/src/commands/vnncomp.rs:relational_router_config_value",
    },
    ReaderSite {
        scope: ROUTER_SCOPE,
        role: "resolve the seven rule switches from the frozen LeverSet once per run \
               (ny_core::router::acting::RuleSwitches)",
        site: "crates/ny-cli/src/router/mod.rs:RouterPolicy::from_levers",
    },
];

const R4_RULE_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: ROUTER_SCOPE,
        role: "project router.rules.lease / far_hand_on into the frozen scored-run LeverSet",
        site: "crates/ny-cli/src/commands/vnncomp.rs:preset_delivered_config_value",
    },
    ReaderSite {
        scope: ROUTER_SCOPE,
        role: "project router.rules.lease / far_hand_on into the RELATIONAL route's frozen \
               LeverSet",
        site: "crates/ny-cli/src/commands/vnncomp.rs:relational_router_config_value",
    },
    ReaderSite {
        scope: ROUTER_SCOPE,
        role: "#router-r4: resolve acting set 3's two rule switches from the frozen LeverSet \
               once per run (ny_core::router::r4::R4Switches)",
        site: "crates/ny-cli/src/router/mod.rs:RouterPolicy::from_levers",
    },
];

const R5_RULE_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: ROUTER_SCOPE,
        role: "project router.rules.route / gap_contract into the frozen scored-run LeverSet",
        site: "crates/ny-cli/src/commands/vnncomp.rs:preset_delivered_config_value",
    },
    ReaderSite {
        scope: ROUTER_SCOPE,
        role: "project router.rules.route / gap_contract into the RELATIONAL route's frozen \
               LeverSet",
        site: "crates/ny-cli/src/commands/vnncomp.rs:relational_router_config_value",
    },
    ReaderSite {
        scope: ROUTER_SCOPE,
        role: "#router-r5: resolve acting set 4's two rule switches from the frozen LeverSet \
               once per run (ny_core::router::route::R5Switches)",
        site: "crates/ny-cli/src/router/mod.rs:RouterPolicy::from_levers",
    },
];

const TRIAL_RULE_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: ROUTER_SCOPE,
        role: "project optimizer.rules.trial_* into the frozen scored-run LeverSet",
        site: "crates/ny-cli/src/commands/vnncomp.rs:preset_delivered_config_value",
    },
    ReaderSite {
        scope: ROUTER_SCOPE,
        role: "project optimizer.rules.trial_* into the RELATIONAL route's frozen LeverSet",
        site: "crates/ny-cli/src/commands/vnncomp.rs:relational_router_config_value",
    },
    ReaderSite {
        scope: ROUTER_SCOPE,
        role: "#router-r6: resolve the four O-TRIAL rule switches from the frozen LeverSet \
               once per run (ny_core::router::otrial::TrialSwitches)",
        site: "crates/ny-cli/src/router/mod.rs:RouterPolicy::from_levers",
    },
];

const PRIORS_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: ROUTER_SCOPE,
        role: "project optimizer.priors into the frozen scored-run LeverSet",
        site: "crates/ny-cli/src/commands/vnncomp.rs:preset_delivered_config_value",
    },
    ReaderSite {
        scope: ROUTER_SCOPE,
        role: "project optimizer.priors into the RELATIONAL route's frozen LeverSet",
        site: "crates/ny-cli/src/commands/vnncomp.rs:relational_router_config_value",
    },
    ReaderSite {
        scope: ROUTER_SCOPE,
        role: "#router-r7: resolve whether the session reads the compiled offline prior from \
               the frozen LeverSet once per run (RouterPolicy::priors)",
        site: "crates/ny-cli/src/router/mod.rs:RouterPolicy::from_levers",
    },
];

declare_levers! {
    registry ROUTER_LEVERS;

    /// `router.mode`: the runtime router (ROUTER_PROGRAM R1).
    pub ROUTER_MODE = LeverDecl {
        name: "NY_ROUTER_MODE",
        kind: LeverKind::Enum(&["off", "shadow", "on"]),
        default: DefaultSpec::Text("off"),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
#router-r1 (ROUTER_PROGRAM R1) / #router-r2 (R2). The verdict-free runtime \
router, delivered by the typed key `router.mode`. MoatRisk High since R2: `on` \
moves lane deadlines and rule-owned caps, which can change which proofs finish \
inside the budget (never what a proof proves).

* `off` (default): dark. No facts are extracted, no plan is computed, the \
progress bus stays disabled (every tap is one relaxed atomic load), and nothing \
is printed: byte-identical to a binary without the router.
* `shadow`: after the lever freeze the router extracts static row facts from \
scans the entry point already made, asks every declared capability's OWN \
admission predicate, and computes a pure, hashed plan (admissions, today's \
structural order, contracts, reserves, <= 3 checkpoints, and what the R2 rules \
would do). It emits NO knob value and moves no cap. Strictly AFTER the verdict \
is published it drains the write-only progress bus and evaluates the c1/c2/c3 \
checkpoints retrospectively. Every router entry runs inside `fail_open`: a \
panic is contained, counted, and disables the router for the rest of the row.
* `on`: runs the acting rules of this binary (`effective=acting`). R2 ships \
acting set 1 (`ny_core::router::ACTING_RULES`): CR-1 arms the ONE deadline \
authority (one monotone deadline cell per governed lane); CR-A decides which \
lanes it governs from their own admission answers and ledgers; CR-C commits the \
attack slice once, up front; CR-T makes the internal verifier the terminal that \
takes the remainder; CR-H hands unspent contracts on and hands an expired root \
warmup's time to the search; CR-R gives a completed margin-row certificate a \
publish grace below the join margin (every collector waits the same grace); \
CR-K derives per-clause retry passes and the depth cap from the deadline and \
the memory lease. Each rule is ablated by its own `router.rules.*` key; a value \
a rule derives never overrides a key the row's own configuration pinned; any \
router fault disarms the authority and the row runs today's route.

Observables: `[router] v1 mode= effective=`, `[router] cap=<id> admit=`, \
`[router] plan sha=`, `[router] ckpt=`, `[router] done faults= bus_pushed=` on \
every exit path that records a verdict, and the flight `router` record; with \
`on`, also `[router] act rule=<id> acts=`, `[router] authority armed`, \
`[router] lane slot=` and `[router] derived key=`. A run with the router not off \
and no `[router] done` line is VOID. Replay: \
`ny vnncomp plan --replay <result>.flight.json`.",
        provenance: Provenance::Unmeasured {
            why_ok: "default `off` is dark and byte-identical; `shadow` reads counters and \
                     writes stderr/flight; `on` is a measurement arm no category yaml may \
                     set (ROUTER_PROGRAM 2.5) until R2's both-year moat moves the \
                     compiled default",
        },
        owner: ROUTER_SCOPE,
        readers: ROUTER_MODE_READERS,
    };

    /// `optimizer.mode`: the runtime optimizer (ROUTER_PROGRAM §5).
    pub OPTIMIZER_MODE = LeverDecl {
        name: "NY_OPTIMIZER_MODE",
        kind: LeverKind::Enum(&["off", "shadow", "on"]),
        default: DefaultSpec::Text("off"),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
#router-r1 (ROUTER_PROGRAM §5) / #router-r3 (R3). The runtime optimizer's \
O-STOP / O-DERIVE / O-TRIAL rules, delivered by the typed key `optimizer.mode`. \
MoatRisk High since R3: `on` ends phases early (the alpha ascent, the interm-alpha \
candidates, a flat upfront attack, a root phase that does not fit), which can \
change which proofs or sats finish inside the budget (never what a proof proves).

* `off` (default): dark.
* `shadow`: after publication, logs what each optimizer rule WOULD have done \
from the drained progress samples (`[optimizer] rule=<id> would=`), e.g. the \
per-iteration governor yield of the margin-row alpha ascent. It changes no \
iterate, cap or clamp.
* `on`: runs the acting optimizer rules of this binary (`effective=acting`). R3 \
ships acting set 2 (`ny_core::router::ACTING_OPTIMIZER_RULES`), the O-STOP rules: \
`rule.governor` (the root alpha ascent yields when an iteration's gain falls below \
0.15 of its own best after 5 samples; the best iterate is kept and certified once), \
`rule.ia_yield` (the interm-alpha candidate loop yields at 0.15 after 2 layers; \
rides on the router's CR-1, which makes the ascent read a contract that excludes \
the freed interm time), `rule.attack_defer` (CR-ATK: a flat upfront attack defers \
its unspent slice to a late sat reserve held out of the verifier; rides on CR-1 \
and the upfront slot's governance) and `rule.root_admission` (a root tightening \
phase whose padded predicted cost does not fit its window is skipped, its inputs \
logged). Each rule is ablated by its own `optimizer.rules.*` key; any router fault \
disarms the stop authority and every phase keeps today's rule.

Parameters move search and effort only, never a certified computation; every \
rule joins the acting set in its own milestone with its own moat. Observables: \
`[optimizer] act rule=<id> acts=`, `[optimizer] authority armed`, \
`[optimizer] yield`, `[optimizer] hand-on`, `[optimizer] defer`, \
`[optimizer] root-admission`, `[optimizer] done` and the flight `router.stop` record.",
        provenance: Provenance::Unmeasured {
            why_ok: "default `off` is dark and byte-identical; `shadow` reads counters and \
                     writes stderr/flight; `on` is a measurement arm no category yaml may \
                     set (ROUTER_PROGRAM 2.5) until R3's both-year moat moves the compiled \
                     default",
        },
        owner: ROUTER_SCOPE,
        readers: OPTIMIZER_MODE_READERS,
    };

    /// `router.rules.admission`: CR-A (rule.admission), an acting
    /// router rule's ablation switch.
    pub ROUTER_RULE_ADMISSION = LeverDecl {
        name: "NY_ROUTER_RULE_ADMISSION",
        kind: LeverKind::BoolOverride,
        default: DefaultSpec::Bool(true),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
#router-r2 CR-A (rule.admission, ROUTER_PROGRAM §4.2), delivered by the typed key \
`router.rules.admission`. Participates ONLY when `router.mode` is `on` (declaration \
default `off`; no category yaml may set it), so the default `true` arms \
nothing. `false` ablates this one rule: it logs `[router] act rule=CR-A acts=0 \
blocked_by=key` and the lanes it would move keep today's value. decides which lanes \
the one deadline authority GOVERNS, from each lane's own admission answer (an inline \
admission defers to the lane's own site: governed), its ledger (only a sealed lane is \
governed) and its first rung (a rung that does not fit is not governed). An ungoverned \
lane keeps exactly the deadline its own rule derives today. Off: every lane is \
governed. High: it decides which lanes the other acting rules may move. \
Observable: `[router] act rule=CR-A acts=1` on every acting row.",
        provenance: Provenance::Unmeasured {
            why_ok: "participates only under router.mode: on, a measurement arm no category \
                     yaml may set until R2's both-year moat moves the compiled default; the \
                     default `true` arms nothing on its own",
        },
        owner: ROUTER_SCOPE,
        readers: ROUTER_RULE_READERS,
    };

    /// `router.rules.contracts`: CR-C (rule.contracts), an acting
    /// router rule's ablation switch.
    pub ROUTER_RULE_CONTRACTS = LeverDecl {
        name: "NY_ROUTER_RULE_CONTRACTS",
        kind: LeverKind::BoolOverride,
        default: DefaultSpec::Bool(true),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
#router-r2 CR-C (rule.contracts, ROUTER_PROGRAM §4.2), delivered by the typed key \
`router.rules.contracts`. Participates ONLY when `router.mode` is `on` (declaration \
default `off`; no category yaml may set it), so the default `true` arms \
nothing. `false` ablates this one rule: it logs `[router] act rule=CR-C acts=0 \
blocked_by=key` and the lanes it would move keep today's value. commits the attack \
slice ONCE, up front: each attack lane's contract is the window today's own sizing \
rules produce against one reference reading (the `#lane-budget-frame` identity plan, \
admitted by the router without its lever), recorded in the authority, and each lane \
claims `now + contract` from its cell at its own start. A lane that declines no longer \
grows the next lane's live-clock window (measured: 71.0 -> 234.7 s absorption on \
traffic); its seconds fall to the terminal. High: it moves time between attack lanes \
and the verifier. \
Observable: `[router] act rule=CR-C acts=1` on every acting row.",
        provenance: Provenance::Unmeasured {
            why_ok: "participates only under router.mode: on, a measurement arm no category \
                     yaml may set until R2's both-year moat moves the compiled default; the \
                     default `true` arms nothing on its own",
        },
        owner: ROUTER_SCOPE,
        readers: ROUTER_RULE_READERS,
    };

    /// `router.rules.cap_authority`: CR-1 (rule.cap_authority), an acting
    /// router rule's ablation switch.
    pub ROUTER_RULE_CAP_AUTHORITY = LeverDecl {
        name: "NY_ROUTER_RULE_CAP_AUTHORITY",
        kind: LeverKind::BoolOverride,
        default: DefaultSpec::Bool(true),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
#router-r2 CR-1 (rule.cap_authority, ROUTER_PROGRAM §4.2), delivered by the typed key \
`router.rules.cap_authority`. Participates ONLY when `router.mode` is `on` (declaration \
default `off`; no category yaml may set it), so the default `true` arms \
nothing. `false` ablates this one rule: it logs `[router] act rule=CR-1 acts=0 \
blocked_by=key` and the lanes it would move keep today's value. arms the ONE deadline \
authority (`ny_core::deadline_authority`): one monotone deadline cell per governed \
lane, each capped at the reserve today's route already keeps for that lane (the \
verifier tier's end for the attack slice and the verifier, scored - 3 s for the \
margin-row lanes). Every other cell rule (CR-A, CR-C, CR-T, CR-R) rides on it. High: \
it is the delivery vehicle of every deadline the router moves. \
Observable: `[router] act rule=CR-1 acts=1` on every acting row.",
        provenance: Provenance::Unmeasured {
            why_ok: "participates only under router.mode: on, a measurement arm no category \
                     yaml may set until R2's both-year moat moves the compiled default; the \
                     default `true` arms nothing on its own",
        },
        owner: ROUTER_SCOPE,
        readers: ROUTER_RULE_READERS,
    };

    /// `router.rules.terminal`: CR-T (rule.terminal), an acting
    /// router rule's ablation switch.
    pub ROUTER_RULE_TERMINAL = LeverDecl {
        name: "NY_ROUTER_RULE_TERMINAL",
        kind: LeverKind::BoolOverride,
        default: DefaultSpec::Bool(true),
        bucket: Bucket::Debug,
        moat: MoatRisk::Low,
        doc: "\
#router-r2 CR-T (rule.terminal, ROUTER_PROGRAM §4.2), delivered by the typed key \
`router.rules.terminal`. Participates ONLY when `router.mode` is `on` (declaration \
default `off`; no category yaml may set it), so the default `true` arms \
nothing. `false` ablates this one rule: it logs `[router] act rule=CR-T acts=0 \
blocked_by=key` and the lanes it would move keep today's value. designates the \
internal verifier as the TERMINAL: never contracted, never shortened (its claim \
records exactly the deadline its own rule derives), and the recipient of every \
hand-on; its claim checks that the credited seconds are inside its window \
(`realized=`). Low: the verifier's own timeout is identical with the rule on or off; \
off, releases return to the pool instead of being credited. \
Observable: `[router] act rule=CR-T acts=1` on every acting row.",
        provenance: Provenance::Unmeasured {
            why_ok: "participates only under router.mode: on, a measurement arm no category \
                     yaml may set until R2's both-year moat moves the compiled default; the \
                     default `true` arms nothing on its own",
        },
        owner: ROUTER_SCOPE,
        readers: ROUTER_RULE_READERS,
    };

    /// `router.rules.hand_on`: CR-H (rule.hand_on), an acting
    /// router rule's ablation switch.
    pub ROUTER_RULE_HAND_ON = LeverDecl {
        name: "NY_ROUTER_RULE_HAND_ON",
        kind: LeverKind::BoolOverride,
        default: DefaultSpec::Bool(true),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
#router-r2 CR-H (rule.hand_on, ROUTER_PROGRAM §4.2), delivered by the typed key \
`router.rules.hand_on`. Participates ONLY when `router.mode` is `on` (declaration \
default `off`; no category yaml may set it), so the default `true` arms \
nothing. `false` ablates this one rule: it logs `[router] act rule=CR-H acts=0 \
blocked_by=key` and the lanes it would move keep today's value. hands unspent time \
downstream instead of losing it: a contracted lane that ends undecided releases its \
unspent contract to the terminal (ledgered, `credited <= unspent`); and a root alpha \
warmup that expires early hands its time to the search \
(`bab.initial_bounds_warmup_fallback`, applied only where the row's own configuration \
did not pin it; sound by monotonicity: the bootstrap enclosure is wider). High: it \
changes which search runs in the freed time. \
Observable: `[router] act rule=CR-H acts=1` on every acting row.",
        provenance: Provenance::Unmeasured {
            why_ok: "participates only under router.mode: on, a measurement arm no category \
                     yaml may set until R2's both-year moat moves the compiled default; the \
                     default `true` arms nothing on its own",
        },
        owner: ROUTER_SCOPE,
        readers: ROUTER_RULE_READERS,
    };

    /// `router.rules.publish_grace`: CR-R (rule.publish_grace), an acting
    /// router rule's ablation switch.
    pub ROUTER_RULE_PUBLISH_GRACE = LeverDecl {
        name: "NY_ROUTER_RULE_PUBLISH_GRACE",
        kind: LeverKind::BoolOverride,
        default: DefaultSpec::Bool(true),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
#router-r2 CR-R (rule.publish_grace, ROUTER_PROGRAM §4.2), delivered by the typed key \
`router.rules.publish_grace`. Participates ONLY when `router.mode` is `on` (declaration \
default `off`; no category yaml may set it), so the default `true` arms \
nothing. `false` ablates this one rule: it logs `[router] act rule=CR-R acts=0 \
blocked_by=key` and the lanes it would move keep today's value. gives a COMPLETED \
margin-row certificate a publish grace past its lane's search deadline, `0 < g < 3 s` \
and below the join margin with a 0.5 s write reserve left (1.5 s at today's 2 s join), \
and makes every collector of that lane wait the same `g` (the inline lane's external \
cap, the concurrent lane's final join). Search never uses it. Measured on the \
precursor constant: discarded completed proofs 2 -> 0, idx_7886 3/3 vs 0/9 with the \
governor, 51-row moat held. High: a published certificate that was discarded before \
now publishes. \
Observable: `[router] act rule=CR-R acts=1` on every acting row.",
        provenance: Provenance::Unmeasured {
            why_ok: "participates only under router.mode: on, a measurement arm no category \
                     yaml may set until R2's both-year moat moves the compiled default; the \
                     default `true` arms nothing on its own",
        },
        owner: ROUTER_SCOPE,
        readers: ROUTER_RULE_READERS,
    };

    /// `router.rules.derived_caps`: CR-K (rule.derived_caps), an acting
    /// router rule's ablation switch.
    pub ROUTER_RULE_DERIVED_CAPS = LeverDecl {
        name: "NY_ROUTER_RULE_DERIVED_CAPS",
        kind: LeverKind::BoolOverride,
        default: DefaultSpec::Bool(true),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
#router-r2 CR-K (rule.derived_caps, ROUTER_PROGRAM §4.2), delivered by the typed key \
`router.rules.derived_caps`. Participates ONLY when `router.mode` is `on` (declaration \
default `off`; no category yaml may set it), so the default `true` arms \
nothing. `false` ablates this one rule: it logs `[router] act rule=CR-K acts=0 \
blocked_by=key` and the lanes it would move keep today's value. O-DERIVE: derives caps \
from the deadline and the memory lease instead of constants, applied only where the \
row's own configuration did not pin the key: `bab.disjunctive_clause_retry_passes` = \
one pass per second of window (the loop itself stops when a pass decides nothing and \
never eats the Graph-MIP reserve); `bab.max_depth` = 1000 per second of window, never \
below today's 100, lifted ONLY while a resident-queue byte budget is in force and fits \
half the host's static lease (depth was doing memory work otherwise). High: it lets \
searches run that constants cut. \
Observable: `[router] act rule=CR-K acts=1` on every acting row.",
        provenance: Provenance::Unmeasured {
            why_ok: "participates only under router.mode: on, a measurement arm no category \
                     yaml may set until R2's both-year moat moves the compiled default; the \
                     default `true` arms nothing on its own",
        },
        owner: ROUTER_SCOPE,
        readers: ROUTER_RULE_READERS,
    };

    /// `optimizer.rules.governor`: rule.governor, an acting optimizer rule's ablation
    /// switch.
    pub OPTIMIZER_RULE_GOVERNOR = LeverDecl {
        name: "NY_OPTIMIZER_RULE_GOVERNOR",
        kind: LeverKind::BoolOverride,
        default: DefaultSpec::Bool(true),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
#router-r3 rule.governor (ROUTER_PROGRAM §5.2, §10 R3), delivered by the typed key \
`optimizer.rules.governor`. Participates ONLY when `optimizer.mode` is `on` (declaration \
default `off`; no category yaml may set it), so the default `true` arms nothing. \
`false` ablates this one rule: it logs `[optimizer] act rule=rule.governor acts=0 \
blocked_by=key` and the phase it would stop keeps today's rule. O-STOP on the margin-row root alpha ascent: per-iteration gain g_i against the \
ascent's OWN best g*; the ascent yields when g_i < 0.15 g* once 5 samples were seen \
(the moat-cleared artifact: 51 banked unsat rows held 51/51, yields at iterations \
5-13), keeps its best iterate and certifies it ONCE through the unchanged published \
guard, and the freed time reaches the tree (`[optimizer] hand-on`). Measured with \
the 2 s publish grace: idx_7886 3/3 vs 0/9. Ratio FITTED on cifar100: detector(n=1) \
until a second family reproduces a converting stop. High: a shorter ascent leaves a \
looser (still valid) root, and the tree's closure ladder prices from the remaining \
wall, so a thin-margin proof can be lost (never a wrong one minted). \
Observable: `[optimizer] act rule=rule.governor acts=1` on every acting row.",
        provenance: Provenance::Unmeasured {
            why_ok: "participates only under optimizer.mode: on, a measurement arm no \
                     category yaml may set until R3's both-year moat moves the compiled \
                     default; the default `true` arms nothing on its own",
        },
        owner: ROUTER_SCOPE,
        readers: OPTIMIZER_RULE_READERS,
    };

    /// `optimizer.rules.ia_yield`: rule.ia_yield, an acting optimizer rule's ablation
    /// switch.
    pub OPTIMIZER_RULE_IA_YIELD = LeverDecl {
        name: "NY_OPTIMIZER_RULE_IA_YIELD",
        kind: LeverKind::BoolOverride,
        default: DefaultSpec::Bool(true),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
#router-r3 rule.ia_yield (ROUTER_PROGRAM §5.2, §10 R3), delivered by the typed key \
`optimizer.rules.ia_yield`. Participates ONLY when `optimizer.mode` is `on` (declaration \
default `off`; no category yaml may set it), so the default `true` arms nothing. \
`false` ablates this one rule: it logs `[optimizer] act rule=rule.ia_yield acts=0 \
blocked_by=key` and the phase it would stop keeps today's rule. O-STOP on the #interm-alpha candidate loop inside backward_interm: per-layer \
extra-tightenings per second against its own best; the candidates yield at 0.15 after \
2 layers (the measured profile 255.6 -> 20.4 -> 5.7 /s fires on the second layer) \
while the baseline shrink-only pass continues. Rides on the router's CR-1 \
(`router.mode: on`): the ascent then reads a CONTRACT that excludes any interm time \
the yield freed (`[optimizer] contract`), so the freed time reaches the tree instead \
of inflating the ascent's 0.4 x remaining share. High: fewer interm tightenings leave \
looser (still valid) boxes; measured 10.79 s mean returned, 0/4 conversions. \
Observable: `[optimizer] act rule=rule.ia_yield acts=1` on every acting row.",
        provenance: Provenance::Unmeasured {
            why_ok: "participates only under optimizer.mode: on, a measurement arm no \
                     category yaml may set until R3's both-year moat moves the compiled \
                     default; the default `true` arms nothing on its own",
        },
        owner: ROUTER_SCOPE,
        readers: OPTIMIZER_RULE_READERS,
    };

    /// `optimizer.rules.attack_defer`: rule.attack_defer (CR-ATK), an acting optimizer rule's ablation
    /// switch.
    pub OPTIMIZER_RULE_ATTACK_DEFER = LeverDecl {
        name: "NY_OPTIMIZER_RULE_ATTACK_DEFER",
        kind: LeverKind::BoolOverride,
        default: DefaultSpec::Bool(true),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
#router-r3 rule.attack_defer (CR-ATK) (ROUTER_PROGRAM §5.2, §10 R3), delivered by the typed key \
`optimizer.rules.attack_defer`. Participates ONLY when `optimizer.mode` is `on` (declaration \
default `off`; no category yaml may set it), so the default `true` arms nothing. \
`false` ablates this one rule: it logs `[optimizer] act rule=rule.attack_defer acts=0 \
blocked_by=key` and the phase it would stop keeps today's rule. DEFER, never cut: at a restart boundary (its safe point) the upfront DLR-APGD attack \
stops when its best margin rose by no more than the confirmation noise floor over the \
trailing half of its slice (slope <= eps/W, W >= 0.5 of the slice; the RATE, never the \
level), provided the unspent slice is >= 5 s and a late sat reserve is admissible; the \
unspent slice is released to the reserve (credited 0 to the verifier), held out of the \
verifier's window (which stays exactly today's), and spent by the post-verifier attack \
seeded by the tree. Floors = today's caps: the attack's total time never falls below \
today's cap when realized, and no extension beyond it (that waits for the x86 sat rig). \
Rides on CR-1 and CR-A governing the upfront slot; never on a short row (CR-S). High: \
a sat the plateaued attack would have found later in its slice is moved behind the \
verifier (tinyimagenet lost 8/15 -> 3/15 when an attack was CUT; this rule defers). \
Observable: `[optimizer] act rule=rule.attack_defer acts=1` on every acting row.",
        provenance: Provenance::Unmeasured {
            why_ok: "participates only under optimizer.mode: on, a measurement arm no \
                     category yaml may set until R3's both-year moat moves the compiled \
                     default; the default `true` arms nothing on its own",
        },
        owner: ROUTER_SCOPE,
        readers: OPTIMIZER_RULE_READERS,
    };

    /// `optimizer.rules.root_admission`: rule.root_admission, an acting optimizer rule's ablation
    /// switch.
    pub OPTIMIZER_RULE_ROOT_ADMISSION = LeverDecl {
        name: "NY_OPTIMIZER_RULE_ROOT_ADMISSION",
        kind: LeverKind::BoolOverride,
        default: DefaultSpec::Bool(true),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
#router-r3 rule.root_admission (ROUTER_PROGRAM §5.2, §10 R3), delivered by the typed key \
`optimizer.rules.root_admission`. Participates ONLY when `optimizer.mode` is `on` (declaration \
default `off`; no category yaml may set it), so the default `true` arms nothing. \
`false` ablates this one rule: it logs `[optimizer] act rule=rule.root_admission acts=0 \
blocked_by=key` and the phase it would stop keeps today's rule. Root-phase admission (CR-A at phase granularity): before a graph-engine root \
tightening phase that declares a predicted cost, `phase_window::admit` is asked \
whether the padded prediction fits the affordable share of the window it would run in \
(the verifier's R2 cell when governed, else the instance remaining); a phase that does \
not fit is SKIPPED rather than cut off, and every consult logs its inputs \
(`[optimizer] root-admission phase= predicted_ms= padded_ms= remaining_ms= source= \
affordable_ms= admit=`). Today's route runs it only under the legacy raw-env gate \
in root_phases.rs. High: a skipped phase leaves wider (still sound) \
intermediate boxes, which can cost a proof. \
Observable: `[optimizer] act rule=rule.root_admission acts=1` on every acting row.",
        provenance: Provenance::Unmeasured {
            why_ok: "participates only under optimizer.mode: on, a measurement arm no \
                     category yaml may set until R3's both-year moat moves the compiled \
                     default; the default `true` arms nothing on its own",
        },
        owner: ROUTER_SCOPE,
        readers: OPTIMIZER_RULE_READERS,
    };

    /// `router.rules.lease`: CR-L, acting set 3's memory-lease ablation switch.
    pub ROUTER_RULE_LEASE = LeverDecl {
        name: "NY_ROUTER_RULE_LEASE",
        kind: LeverKind::BoolOverride,
        default: DefaultSpec::Bool(true),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
#router-r4 CR-L (rule.memory_lease, ROUTER_PROGRAM §10 R4, F6), delivered by the typed key \
`router.rules.lease`. Participates ONLY when `router.mode` is `on` (declaration default \
`off`; no category yaml may set it), so the default `true` arms nothing. `false` ablates \
this one rule: it logs `[router] act rule=CR-L acts=0 blocked_by=key` and every \
optional-heavy memory gate keeps today's live decision. The MEMORY LEASE: one envelope per \
row from the DECLARED host's static envelope (`min(MemTotal, cgroup memory.max) - \
reserve`), never the live headroom. The internal verifier holds half of it while it runs; \
every graph-heavy optional lane (the concurrent and inline margin-row workers, the \
post-verifier attack tail and its graph loads and f64 oracle, the ORT VJP refine wave) \
acquires its memory model's 16 GiB before it starts and returns it when its working set is \
gone, and is REFUSED when the outstanding leases would pass the envelope. Admission \
control only, never a verdict input; today's live gate stays the floor (a lease can only \
refuse more): it replaces the `Unbounded => true` arm, i.e. every competition run, which \
has no cgroup. High: a refused optional lane can cost a proof or a sat it would have \
found (never mints one). Observable: `[router] act rule=CR-L acts=1`, `[router] lease \
armed`, `[router] lease holder= op= decision=`, `[router] lease done`.",
        provenance: Provenance::Unmeasured {
            why_ok: "participates only under router.mode: on, a measurement arm no category \
                     yaml may set until R4's both-year moat moves the compiled default; the \
                     default `true` arms nothing on its own",
        },
        owner: ROUTER_SCOPE,
        readers: R4_RULE_READERS,
    };

    /// `router.rules.far_hand_on`: CR-F, acting set 3's FAR hand-on ablation switch.
    pub ROUTER_RULE_FAR_HAND_ON = LeverDecl {
        name: "NY_ROUTER_RULE_FAR_HAND_ON",
        kind: LeverKind::BoolOverride,
        default: DefaultSpec::Bool(true),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
#router-r4 CR-F (rule.far_hand_on, ROUTER_PROGRAM §4.2, §4.3, §10 R4), delivered by the \
typed key `router.rules.far_hand_on`. Participates ONLY when `router.mode` is `on` \
(declaration default `off`; no category yaml may set it), so the default `true` arms \
nothing. `false` ablates this one rule: it logs `[router] act rule=CR-F acts=0 \
blocked_by=key` and no lane is ever handed on. At a source lane's own declared safe point \
after c2 (50%) and c3 (75%) of the prover window, two trailing windows reading FAR in \
WORK UNITS (FrontierBinary/Kary: closes per expansion <= T_FAR = 0.10 with the frontier \
growing; Bfs: growth >= 1 past depth 8; Volume: certified volume flat with the queue \
growing; never an attack, whose flat slope CR-ATK defers) hand the lane's unspent deadline \
cell to the recipient's cell, and the lane stops there undecided. The recipient must be \
STRICTLY sealed (S7 + S8 on its envelope), admitted by its own predicate, a prover that \
accepts a hand-on; no registry entry is sealed in R4, so CR-F arms in OBSERVE mode on \
every row (classifies and logs, hands nothing on; `acts=0 \
blocked_by=no_sealed_recipient`). Rides on CR-1; never on a short row (CR-S). T_FAR is \
FITTED on one family (pre-registered in R0). High: a hand-on ends a lane early, which \
can cost a proof (never mints one). Observable: `[router] act rule=CR-F`, `[router] far \
armed`, `[router] far cap= ckpt= class= act=`, `[router] far done`.",
        provenance: Provenance::Unmeasured {
            why_ok: "participates only under router.mode: on, a measurement arm no category \
                     yaml may set until R4's both-year moat moves the compiled default; with \
                     no sealed recipient it hands nothing on",
        },
        owner: ROUTER_SCOPE,
        readers: R4_RULE_READERS,
    };

    /// `router.rules.route`: CR-RS, acting set 4's routing ablation switch.
    pub ROUTER_RULE_ROUTE = LeverDecl {
        name: "NY_ROUTER_RULE_ROUTE",
        kind: LeverKind::BoolOverride,
        default: DefaultSpec::Bool(true),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
#router-r5 CR-RS (rule.route, ROUTER_PROGRAM §3.5, §10 R5), delivered by the typed key \
`router.rules.route`. Participates ONLY when `router.mode` is `on` (declaration default \
`off`; no category yaml may set it), so the default `true` arms nothing. `false` ablates \
this one rule: it logs `[router] act rule=CR-RS acts=0 blocked_by=key` and every \
capability keeps the arming its own typed key gives it. CAPABILITIES ROUTED, NOT ARMED: \
per instance and name-blind, the router configures a capability (sets the typed keys its \
S7/S8 evidence arm ran) only at SR.seal — S7 and S8 done and the envelope those evidence \
rows cover recorded — where its own routing precondition admits the row's static facts \
and the host (specification archetype its consumer serves, an operator it acts on, size \
ceilings its code declares, device), its sealed envelope covers the row's operator \
signature, its parent's admission holds or is re-checked at the consumer's site, the row \
did not pin its keys (a pin on a sealed capability is reported as pin debt) and its \
exclusive slot is free; a High moat-risk key is never selected for an unsealed \
capability. No registry entry is sealed today, so CR-RS routes NOTHING on any row. High: \
a routed capability changes which sound computation a lane runs, which can cost a proof \
(never mints one: every routed capability keeps its own admission and certificate \
checks). Observable: `[router] act rule=CR-RS`, `[router] route cap= sr= routed=`, \
`[router] route applied`, `[router] route done`.",
        provenance: Provenance::Unmeasured {
            why_ok: "participates only under router.mode: on, a measurement arm no category \
                     yaml may set until R5's both-year moat moves the compiled default; with \
                     no sealed capability it routes nothing",
        },
        owner: ROUTER_SCOPE,
        readers: R5_RULE_READERS,
    };

    /// `router.rules.gap_contract`: CR-G, acting set 4's gap-order contract switch.
    pub ROUTER_RULE_GAP_CONTRACT = LeverDecl {
        name: "NY_ROUTER_RULE_GAP_CONTRACT",
        kind: LeverKind::BoolOverride,
        default: DefaultSpec::Bool(true),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
#router-r5 CR-G (ROUTER_PROGRAM §4.2 CR-G, F8, F9, §10 R5), delivered by the typed key \
`router.rules.gap_contract`. Participates ONLY when `router.mode` is `on` (declaration \
default `off`; no category yaml may set it), so the default `true` arms nothing. `false` \
ablates this one rule: it logs `[router] act rule=CR-G acts=0 blocked_by=key` and every \
multi-objective tier keeps today's fair share. Inside the frozen-dual tier (wherever it \
runs), classes keep their gap order (easiest first) but are funded with a CONTRACT of at \
least one state build plus two BFS levels, priced from what the tier measured on the \
classes it already ran (the last build and per-level time, or twice the longest window a \
build was cut at), instead of the fair share `left / (classes left)`; the first class and \
every class whose rung the fair share already covers keep the fair share; a rung the tier \
can no longer hold DECLINES the class (it never half-runs) and the classes behind it stay \
open in the tree, which takes back the tier's unspent wall. High: re-dividing a tier's \
wall can cost a class it would have closed (never mints one: a class closes only through \
its own certified search). Observable: `[router] act rule=CR-G`, `[frozen-dual] contract \
class= window=`, `[router] route done ... tiers=`.",
        provenance: Provenance::Unmeasured {
            why_ok: "participates only under router.mode: on, a measurement arm no category \
                     yaml may set until R5's both-year moat moves the compiled default; it \
                     re-divides a tier's own wall among its classes only",
        },
        owner: ROUTER_SCOPE,
        readers: R5_RULE_READERS,
    };
    /// `optimizer.rules.trial_width`: rule.trial_width, an acting optimizer
    /// trial rule's ablation switch.
    pub OPTIMIZER_RULE_TRIAL_WIDTH = LeverDecl {
        name: "NY_OPTIMIZER_RULE_TRIAL_WIDTH",
        kind: LeverKind::BoolOverride,
        default: DefaultSpec::Bool(true),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
#router-r6 rule.trial_width (ROUTER_PROGRAM §5.1, §5.2, §5.3, §10 R6), delivered by the \
typed key `optimizer.rules.trial_width`. Participates ONLY when `optimizer.mode` is `on` \
(declaration default `off`; no category yaml may set it), so the default `true` arms \
nothing. `false` ablates this one rule: it logs `[optimizer] act rule=rule.trial_width \
acts=0 blocked_by=key` and every width site keeps today's value. O-TRIAL on FAST lanes: \
AIMD on certified closes per second over windows of >= 16 of the lane's own work units, \
inside the lane's own static memory budget: x2 on a >= 5% rise with lease use < 70%, /2 on \
a >= 10% fall; a raise that falls twice goes back to the last good width and FREEZES; a \
lease use > 85% reverts a raise at once (or halves, a guard); kept on >= baseline x 1.05, \
else reverted and that direction made tabu; <= 2 trials per knob per row, >= 8 windows \
left, never in the last 25% of the lane. Sites: the margin-row wave tree's width \
(margin_row.tree_wave_batch, inside its declared span 64..1024), the REORDERED input-split \
loop's live batch (bab.batch_size is dormant without reorder_bab, so the site is blocked \
there rather than trialling a value no loop reads) and the lowdim lane's wave width (a \
code constant, 2048 -> 512..8192). A site whose knob the row pinned is frozen. High: the \
width decides which domains are bounded together inside the budget, which can cost a \
proof (never mints one: every close is the lane's own certified close). Observable: \
`[optimizer] act rule=rule.trial_width acts=`, `[router] trial-site`, `[router] trial \
knob= dir= decision=kept|reverted|tabu|...`, `[router] trial done`.",
        provenance: Provenance::Unmeasured {
            why_ok: "participates only under optimizer.mode: on, a measurement arm no \
                     category yaml may set until R6's both-year moat moves the compiled \
                     default; the default `true` arms nothing on its own",
        },
        owner: ROUTER_SCOPE,
        readers: TRIAL_RULE_READERS,
    };

    /// `optimizer.rules.trial_lr_backoff`: rule.trial_lr_backoff, an acting
    /// optimizer trial rule's ablation switch.
    pub OPTIMIZER_RULE_TRIAL_LR_BACKOFF = LeverDecl {
        name: "NY_OPTIMIZER_RULE_TRIAL_LR_BACKOFF",
        kind: LeverKind::BoolOverride,
        default: DefaultSpec::Bool(true),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
#router-r6 rule.trial_lr_backoff (ROUTER_PROGRAM §5.2 learning rates, §10 R6), delivered by \
the typed key `optimizer.rules.trial_lr_backoff`. Participates ONLY when `optimizer.mode` \
is `on` (declaration default `off`; no category yaml may set it), so the default `true` \
arms nothing. `false` ablates this one rule: it logs `[optimizer] act rule=\
rule.trial_lr_backoff acts=0 blocked_by=key` and the ascent keeps today's step. BACKOFF \
ONLY, never a raise: on the margin-row root alpha ascent (Adam, `margin_row.alpha_lr` > 0) \
three REJECTED iterates in a row halve the step multiplier (floor lr0 / 8); the next three \
iterates' gain per second is compared with the three that triggered it, and a fall reverts \
the step and makes the backoff tabu for the row; <= 2 backoffs per row, never in the last \
25% of the phase. Every iterate is a valid relaxation (alpha clamped to [0, 1]); the best \
one stays the committed one and is checked once by the ascent's unchanged guard. A row \
that pinned margin_row.alpha_lr is frozen. High: a smaller step changes which iterate the \
ascent reaches inside its budget, which can cost a thin-margin proof (never mints one). \
Observable: `[optimizer] act rule=rule.trial_lr_backoff acts=`, `[router] trial \
knob=margin_row.alpha_lr dir=down decision=`.",
        provenance: Provenance::Unmeasured {
            why_ok: "participates only under optimizer.mode: on, a measurement arm no \
                     category yaml may set until R6's both-year moat moves the compiled \
                     default; the default `true` arms nothing on its own",
        },
        owner: ROUTER_SCOPE,
        readers: TRIAL_RULE_READERS,
    };

    /// `optimizer.rules.trial_split_restart`: rule.trial_split_restart, an
    /// acting optimizer trial rule's ablation switch.
    pub OPTIMIZER_RULE_TRIAL_SPLIT_RESTART = LeverDecl {
        name: "NY_OPTIMIZER_RULE_TRIAL_SPLIT_RESTART",
        kind: LeverKind::BoolOverride,
        default: DefaultSpec::Bool(true),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
#router-r6 rule.trial_split_restart (ROUTER_PROGRAM §5.2 split heuristic, §6.1, §10 R6), \
delivered by the typed key `optimizer.rules.trial_split_restart`. Participates ONLY when \
`optimizer.mode` is `on` (declaration default `off`; no category yaml may set it), so the \
default `true` arms nothing. `false` ablates this one rule: it logs `[optimizer] act \
rule=rule.trial_split_restart acts=0 blocked_by=key` and the lane keeps its heuristic. \
RESTART FORM, at most once per row: on the lowdim lane, when its two trailing windows of \
>= 16 waves read FAR in work units (certified volume flat while the queue grows) AND more \
than 90% of the children bounded in them did not lift their parent's bound, the lane \
DISCARDS its tree and restarts from the root box with the other split heuristic (widest <-> \
sbwr); no fact of the discarded tree is carried (a split-kind switch is a restart, §6.1), so \
a proof needs the new tree to close every leaf. >= 8 windows left, never in the last 25% \
of the lane. A row that pinned bab.branching.input_split.lowdim_split is frozen. High: a \
restart throws away work, which can cost a proof (never mints one: any bisection covers \
its box). Observable: `[optimizer] act rule=rule.trial_split_restart acts=`, `[router] \
trial knob=bab.branching.input_split.lowdim_split dir=switch decision=started`.",
        provenance: Provenance::Unmeasured {
            why_ok: "participates only under optimizer.mode: on, a measurement arm no \
                     category yaml may set until R6's both-year moat moves the compiled \
                     default; the default `true` arms nothing on its own",
        },
        owner: ROUTER_SCOPE,
        readers: TRIAL_RULE_READERS,
    };

    /// `optimizer.rules.trial_attack_halving`: rule.trial_attack_halving, an
    /// acting optimizer trial rule's ablation switch.
    pub OPTIMIZER_RULE_TRIAL_ATTACK_HALVING = LeverDecl {
        name: "NY_OPTIMIZER_RULE_TRIAL_ATTACK_HALVING",
        kind: LeverKind::BoolOverride,
        default: DefaultSpec::Bool(true),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
#router-r6 rule.trial_attack_halving (ROUTER_PROGRAM §5.2 attack restarts / length, §10 R6), \
delivered by the typed key `optimizer.rules.trial_attack_halving`. Participates ONLY when \
`optimizer.mode` is `on` (declaration default `off`; no category yaml may set it), so the \
default `true` arms nothing. `false` ablates this one rule: it logs `[optimizer] act \
rule=rule.trial_attack_halving acts=0 blocked_by=key` and every restart starts at today's \
step. SUCCESSIVE HALVING across a fixed initial-step pool {0.25 (today's), 0.5, 0.125, \
0.0625} of the upfront DLR-APGD lane (APGD restarts only; sign / global-step geometries \
keep their own rule): restart 0 keeps today's step; each round gives every surviving step \
the same number of restarts (alternating restart parity, so each sees both losses), and \
the better half by best-margin improvement per second survives; no elimination past 75% of \
the slice. The lane's slice, restart cap and acceptance gate are unchanged (floors = \
today's caps): every candidate still passes the trusted-oracle gate. High: a different \
step changes which counterexample a restart reaches inside the slice, which can cost a sat \
(never mints one). Observable: `[optimizer] act rule=rule.trial_attack_halving acts=`, \
`[router] trial knob=attack.upfront_apgd.step0 dir=halve decision=kept`.",
        provenance: Provenance::Unmeasured {
            why_ok: "participates only under optimizer.mode: on, a measurement arm no \
                     category yaml may set until R6's both-year moat moves the compiled \
                     default; the default `true` arms nothing on its own",
        },
        owner: ROUTER_SCOPE,
        readers: TRIAL_RULE_READERS,
    };

    /// `optimizer.priors`: #router-r7, read the compiled offline cost-model prior.
    pub OPTIMIZER_PRIORS = LeverDecl {
        name: "NY_OPTIMIZER_PRIORS",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
#router-r7 (ROUTER_PROGRAM §5.5, §10 R7), delivered by the typed key `optimizer.priors`. \
Default `false`: the router reads NO prior and every policy input carries `uniform-v1`, \
byte-identical to R5. `true` (only where the router runs at all: `router.mode` or \
`optimizer.mode` not `off`, which no category yaml may set) makes the session read the \
COMPILED cost-model table `ny_core::router::priors` (time per phase from static facts, keyed \
only by two operator bits with one shared size exponent; anti-fingerprint gated; Layer-B \
success curves uniform) on the host class it was measured on — any other class prices \
nothing. The row's prior view is hashed and logged, the bound capabilities get a priced \
first rung, the plan contracts a priced deferred lane at that rung, and under `router.mode: \
on` R2's CR-A leaves a slot whose priced rung does not fit the window on TODAY's deadline. \
High: a planning number can change which slot the deadline authority governs, which can cost \
a proof (never mints one: no bound, tolerance, gate or publisher is read or written). \
Observable: `[router] priors v=`, the plan's `rule.priors` entries, \
`priors=cost-v1+<sha8>` in the `[router] v1` banner.",
        provenance: Provenance::Unmeasured {
            why_ok: "default off; reads a compiled table for planning only, under a router \
                     mode no category yaml may set; a priced rung can only keep a lane on \
                     today's deadline or log a number",
        },
        owner: ROUTER_SCOPE,
        readers: PRIORS_READERS,
    };
}
