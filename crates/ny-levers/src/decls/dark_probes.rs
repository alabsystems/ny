// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Dark diagnostic probes and A/B shape switches added after the 823 baseline.
//!
//! WHY THIS MODULE EXISTS. Eight raw `NY_*` process-environment reads landed
//! after `dbda7dbdc` set the ratchet at 823, taking it to 831. The ratchet
//! forbids absorbing a positive delta by raising the baseline, and says why: a
//! fresh direct-literal read is unenumerable, its parser disagrees with the
//! other ~850, and it will not appear in the flight receipt. These four levers
//! are declared here so all three are false. The exact-star measurement
//! controls live together in the sibling `star` registry.
//!
//! (Note for anyone editing this file: the ratchet counts literal SUBSTRINGS,
//! so spelling the direct-read call form in a comment here would itself score
//! as a raw read. That is not a flaw in the gate — a doc comment is exactly
//! where a copy-paste template would live. Describe the form; do not spell it.)
//!
//! Every parser below is preserved EXACTLY as the reader had it. That is the
//! point of the declaration: the lever's public parser is part of its contract,
//! and a migration that quietly widened `"1"` to `"true"`, or started trimming
//! whitespace where the reader did not, would be a behaviour change smuggled in
//! as bookkeeping.
//!
//! [`FALSIFY_PORTFOLIO_LANE`] is the second non-diagnostic entry: armed, it
//! admits the ported `ny-falsify` strategy portfolio into the attack slice,
//! where it can produce a scored `sat`. Unlike the sign-space lane it has NO
//! typed preset key and its declaration default is off, so a competition
//! harness cannot reach it at all. Its wall cap,
//! [`FALSIFY_PORTFOLIO_SECONDS`], is unreachable behind it.
//!
//! All four of the ORIGINAL entries are dark by default and diagnostic, EXCEPT
//! [`INPUT_SPLIT_NESTED_DEADLINE`], which ships armed and whose `"0"` opt-out
//! removes a rebound's interruptibility — recorded as `MoatRisk::High` for that
//! reason, not because either arm is unsound.
//!
//! [`BNN_SIGN_SPACE_LANE`] joined later and is the one entry here that is not
//! diagnostic: armed, it admits a falsification lane that can produce a scored
//! `sat`. Its DECLARATION default is still off, but it is no longer dark on
//! the scored path — the `traffic_signs_recognition_2023` preset arms it
//! through the typed `attack.bnn_sign_space` config layer, and this variable
//! is now the OVERRIDE rather than the only way in. Every candidate it
//! produces still has to survive the unchanged trusted-oracle gate before
//! publication.

use crate::{
    declare_levers, Bucket, DefaultSpec, LeverDecl, LeverKind, MoatRisk, Provenance, ReaderSite,
    Scope,
};

const GRAPH_ALPHA: Scope = Scope {
    package: "ny-propagate",
    subsystem: "graph-alpha-envelope",
};

const INPUT_SPLIT: Scope = Scope {
    package: "ny-propagate",
    subsystem: "input-split-rebound",
};

const BNN_SIGN_SPACE: Scope = Scope {
    package: "ny-cli",
    subsystem: "bnn-sign-space-falsification",
};

const FALSIFY_PORTFOLIO: Scope = Scope {
    package: "ny-cli",
    subsystem: "falsification-portfolio",
};
const ENVELOPE_XSTAR_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: GRAPH_ALPHA,
        role: "emit the x* envelope diagnostic from the DAG alpha gradient path",
        site: "crates/ny-propagate/src/network/graph_alpha/propagate_dag/gradients/mod.rs:2339",
    },
    ReaderSite {
        scope: GRAPH_ALPHA,
        role: "emit the same diagnostic from the backward alpha gradient path",
        site: "crates/ny-propagate/src/network/graph_alpha/backward/gradients.rs:415",
    },
];

const ENVELOPE_RESCALE_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: GRAPH_ALPHA,
        role: "gate the rescale diagnostic on the backward alpha gradient path",
        site: "crates/ny-propagate/src/network/graph_alpha/backward/gradients.rs:210",
    },
    ReaderSite {
        scope: GRAPH_ALPHA,
        role: "gate the first-iterate rescale diagnostic on the DAG path",
        site: "crates/ny-propagate/src/network/graph_alpha/propagate_dag/gradients/mod.rs:2387",
    },
];

const INPUT_SPLIT_PROBE_READERS: &[ReaderSite] = &[ReaderSite {
    scope: INPUT_SPLIT,
    role: "print the per-rebound domain/deadline/stacking line",
    site: "crates/ny-propagate/src/beta_crown/engine/graph/input_split/shared_specs.rs:309",
}];

const INPUT_SPLIT_DEADLINE_READERS: &[ReaderSite] = &[ReaderSite {
    scope: INPUT_SPLIT,
    role: "drop the nested alpha deadline to restore the pre-6f49a660 shape for an A/B",
    site: "crates/ny-propagate/src/beta_crown/engine/graph/input_split/shared_specs.rs:307",
}];

const BNN_SIGN_SPACE_MOVE_READERS: &[ReaderSite] = &[ReaderSite {
    scope: BNN_SIGN_SPACE,
    role: "pick which in-box point each lazy row-generation round of the realizability \
           search adopts from its LP primal: the LP VERTEX (shipped) or the MINIMAL \
           point on the segment to it that carries the deficient units past their \
           thresholds",
    site: "crates/ny-cli/src/commands/beta_crown/sign_space_falsify.rs \
           (SignSpaceProblem::segment_move)",
}];

const BNN_SIGN_SPACE_TRUST_READERS: &[ReaderSite] = &[ReaderSite {
    scope: BNN_SIGN_SPACE,
    role: "restrict WHERE the realizability LP may put the pixel vector: the full vnnlib \
           box (shipped) or an L-infinity trust region around the incumbent that is \
           doubled, never concluded on, whenever the restricted LP fails",
    site: "crates/ny-cli/src/commands/beta_crown/sign_space_falsify.rs \
           (SignSpaceProblem::trust_region)",
}];

const BNN_SIGN_SPACE_TRACE_READERS: &[ReaderSite] = &[ReaderSite {
    scope: Scope {
        package: "ny-mip",
        subsystem: "bnn-sign-space",
    },
    role: "print one stderr line per lazy row-generation round of the realizability \
           search: round index, active-set size, rows added, deficient-unit count, \
           worst true OR-slack and (on the minimal-move arm) the chosen theta",
    site: "crates/ny-mip/src/bnn_sign_space.rs (solve_realizability)",
}];

const ATTACK_OBJECTIVE: Scope = Scope {
    package: "ny-cli",
    subsystem: "upfront-attack-objective",
};

const PRE_SOFTMAX_ATTACK_OBJECTIVE_READERS: &[ReaderSite] = &[ReaderSite {
    scope: ATTACK_OBJECTIVE,
    role: "admit scoring the incumbent gradient-guided falsification lane against \
           PRE-Softmax logits instead of the trusted forward's post-Softmax \
           probabilities; disarmed, the reader returns before the admission guard \
           runs, before any pre-Softmax tensor is read, and the lane's objective, \
           direction row and DLR denominator are the historical post-Softmax ones",
    site: "crates/ny-cli/src/commands/vnncomp.rs (pre_softmax_attack_objective_armed)",
}];

const BNN_STE_PGD_READERS: &[ReaderSite] = &[ReaderSite {
    scope: BNN_SIGN_SPACE,
    role: "the admission of the STE-PGD falsification lane over the SAME structurally \
           admitted binarized fragment; disarmed, the lane returns its `Disarmed` outcome \
           before it reads the model, the property, or constructs any request",
    site: "crates/ny-cli/src/commands/beta_crown/sign_space_falsify.rs \
           (ste_pgd_falsify_armed)",
}];

const BNN_SIGN_SPACE_READERS: &[ReaderSite] = &[ReaderSite {
    scope: BNN_SIGN_SPACE,
    role: "the admission of the LP-guided sign-space falsification lane, over the typed \
           `attack.bnn_sign_space` preset layer; disarmed, the lane returns its \
           `Disarmed` outcome before it reads the model, the property, or constructs \
           any `SignSpaceRequest`",
    site: "crates/ny-cli/src/commands/beta_crown/sign_space_falsify.rs \
           (sign_space_falsify_armed)",
}];
const FALSIFY_PORTFOLIO_READERS: &[ReaderSite] = &[ReaderSite {
    scope: FALSIFY_PORTFOLIO,
    role: "the admission of the ported `ny-falsify` strategy portfolio (S1 `special`, \
           S9 `square`) inside the attack slice; disarmed, the lane returns before it \
           parses the property, builds a search box, or constructs an ORT session",
    site: "crates/ny-cli/src/commands/vnncomp/falsify_portfolio.rs \
           (portfolio_falsify_armed)",
}];

const LANE_SCHEDULER: Scope = Scope {
    package: "ny-cli",
    subsystem: "marginal-value-lane-scheduler",
};

const LANE_VALUE_SCHEDULER_READERS: &[ReaderSite] = &[ReaderSite {
    scope: LANE_SCHEDULER,
    role: "route the per-instance attack slice through ONE marginal-value ledger \
           instead of a chain of private fixed fractions: every lane re-derives its \
           cap from the LIVE remaining budget, reports its actual cost and its \
           progress in its own work units, and a stalled lane's unspent seconds \
           return to the pool for the next lane's CAP. Disarmed, the reader returns \
           before the ledger is built and each lane computes exactly the window it \
           computes today",
    site: "crates/ny-cli/src/commands/lane_schedule.rs (lane_value_scheduler_armed)",
}];

const LANE_ALLOCATOR: Scope = Scope {
    package: "ny-cli",
    subsystem: "lane-budget-mckp-allocator",
};

const LANE_BUDGET_ALLOCATOR_READERS: &[ReaderSite] = &[ReaderSite {
    scope: LANE_ALLOCATOR,
    role: "choose the per-instance attack-slice caps JOINTLY and UP FRONT by solving \
           the Layer-A multiple-choice knapsack in `ny_mip::lane_allocation`, instead \
           of letting each lane take a private fraction of whatever it was handed: the \
           LP sign-space, STE-PGD and upfront-APGD lanes are handed the cap the \
           allocator committed to, a lane granted zero seconds is SKIPPED, and every \
           second not granted stays with the branch-and-bound residual claimant. \
           Disarmed, the reader returns before the objective probe runs, before any \
           allocation request is built and before the solver is entered, and each lane \
           derives exactly the window it derives today, from the same private helper, \
           in the same order",
    site: "crates/ny-cli/src/commands/lane_allocation.rs (lane_budget_allocator_armed)",
}];

const FALSIFY_PORTFOLIO_SECONDS_READERS: &[ReaderSite] = &[ReaderSite {
    scope: FALSIFY_PORTFOLIO,
    role: "the wall-clock cap of the portfolio phase, read only after the lane is \
           already armed, so it is unreachable on a default run",
    site: "crates/ny-cli/src/commands/vnncomp/falsify_portfolio.rs \
           (portfolio_wall_cap)",
}];

declare_levers! {
    registry DARK_PROBE_LEVERS;

    /// `NY_BNN_SIGN_SPACE` — override for the LP-guided sign-space
    /// falsification lane, which the traffic-signs preset now arms by default.
    pub BNN_SIGN_SPACE_LANE = LeverDecl {
        name: "NY_BNN_SIGN_SPACE",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Admits the LP-guided sign-space falsification lane on a binarized (`Sign`) conv \
suffix, ahead of the ordinary upfront attack. Exact \"1\" arms it and exact \
\"0\" disarms it; every other byte string is a RECORDED REJECTION that resolves \
to this declaration's `false` default. On the disarmed arm the reader returns \
before any model load, property parse or `SignSpaceRequest` construction — so \
that path is byte-identical to the unwired tree.

THIS IS AN OVERRIDE, NOT THE ONLY WAY IN. The typed config layer \
(`attack.bnn_sign_space` in a `configs/vnncomp*/` preset, resolved through \
`read_over_config`) is what arms the lane on a scored run, because a \
competition harness exports no `NY_*`. `traffic_signs_recognition_2023` sets \
it: measured 30/45 armed vs 27/45 unarmed at the official 480s budget. This \
variable still WINS wherever it is present, in both directions, which is what \
keeps the unarmed baseline reproducible on the shipped configuration — and a \
near-miss token suppresses the preset rather than riding it, so a typo cannot \
silently leave a scored run in a state nobody chose.

WHAT THE LANE CAN AND CANNOT DO. `ny_mip::falsify_bnn_sign_suffix_unwired` has \
no verified/unsat outcome by construction: it can only ever exhibit a claimed \
counterexample, so it is structurally incapable of causing a false `unsat`. \
Its `Refused` and `Exhausted` outcomes fall through to the unchanged solver \
path and never become a verdict of any kind. A `Candidate` is a CLAIM, not a \
verdict: the caller re-forwards it through the ORIGINAL model and the UNCHANGED \
`gate_sat_with_trusted_oracle` (real ONNX Runtime forward on the original graph \
plus the true-f64 recheck) before anything can be published.

MoatRisk::High nonetheless, for two honest reasons that have nothing to do with \
false `unsat`. First, the lane is a NEW `sat` SOURCE on the scored path, and a \
new source of published verdicts deserves the higher classification even when \
every one of them passes the pre-existing gate. Second, it is not free: armed, \
it spends a bounded slice of the scored instance budget before the ordinary \
attack and the BaB verifier, so a row that would have been answered can time \
out instead. That is the same cost profile as `NY_STAR_DARK_SECONDS`, and it \
is why the config layer arms it per CATEGORY, where that spend is measured, \
rather than globally.",
        provenance: Provenance::Measured {
            commit: "0728daea1",
            date: "2026-08-14",
            artifact: "reports/measured-2026/traffic_signs_recognition_2023{.csv,_NOTES.md}",
            delta: "traffic_signs_recognition_2023 at the official 480s budget: 30/45 \
                    armed vs 27/45 unarmed. Exactly the three model_30 eps=1 rows \
                    flip (error 485s -> sat 132.4/58.3/97.9s); no other row changes \
                    verdict. On the non-admitted 48x48 net the armed sweep ran 442s \
                    vs 445s unarmed, so the structural refusal costs no measurable \
                    budget.",
        },
        owner: BNN_SIGN_SPACE,
        readers: BNN_SIGN_SPACE_READERS,
    };

    /// `NY_BNN_STE_PGD` — the straight-through-estimator PGD falsification
    /// lane over the same structurally admitted binarized fragment.
    pub BNN_STE_PGD_LANE = LeverDecl {
        name: "NY_BNN_STE_PGD",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Admits the STE-PGD falsification lane on a binarized (`Sign`) conv suffix, \
inside the same attack slice as the LP-guided sign-space lane and over the \
same STRUCTURALLY admitted fragment (`ny_mip::bnn_sign_space::admit`; no \
filename, category or benchmark-name test exists anywhere on the path). Exact \
\"1\" arms it and exact \"0\" disarms it; every other byte string is a \
RECORDED REJECTION that resolves to this declaration's `false` default. On the \
disarmed arm the reader returns before any model load, property parse or \
request construction — so that path is byte-identical to the unwired tree.

WHY IT EXISTS. `Sign` is piecewise constant, so its true derivative is zero \
almost everywhere — but the STRAIGHT-THROUGH ESTIMATOR, the standard technique \
for binarized networks, runs the REAL `Sign` forward and substitutes a \
surrogate derivative in the backward pass only. The forward stays exact (every \
accumulator and logit is the integer the network computes) while the backward \
produces a usable ascent direction. The step schedule is the part that matters \
as much as the gradient: integer ROUNDING at every iterate, a step decaying \
from about the box half-width down to one grey level, momentum on the \
normalized gradient, and iterated-local-search restarts around the incumbent. \
That combination reaches points 483-1483 first-layer flips from the box \
centre, where the LP lane's realizability search accepts 7-16 flips per lane \
and stalls (`docs/BNN_SIGN_SPACE_FALSIFICATION_2026-08-12.md` sections 5 and \
10).

IT IS AN ENVIRONMENT INSTRUMENT AND NOT A PRESET KEY, deliberately, and for \
the same reason `NY_BNN_SIGN_SPACE_MINIMAL_MOVE` and \
`NY_BNN_SIGN_SPACE_TRUST_REGION` are not: promoting it to the typed config \
layer would let it reach a SCORED run before it has a measurement to stand on. \
The shipped default changes — and the `attack.*` key gets added — only once an \
armed-versus-unarmed sweep of the whole 45-row family measures a win with no \
regression.

WHAT THE LANE CAN AND CANNOT DO. It returns `ny_mip::SignSpaceOutcome`, which \
has no verified/unsat variant by construction, so it is structurally incapable \
of causing a false `unsat`. Its `Refused` and `Exhausted` outcomes fall \
through to the unchanged solver path and never become a verdict of any kind. A \
`Candidate` is a CLAIM: it is finalized only after an exact from-scratch \
forward at the returned point, ZERO-TOLERANCE box membership against the \
vnnlib `lo`/`hi`, and an `f32`-replay-stability margin on every sign decision; \
then the caller re-forwards it through the ORIGINAL model and the UNCHANGED \
`gate_sat_with_trusted_oracle` before anything can be published.

MoatRisk::High for the same two honest reasons as `NY_BNN_SIGN_SPACE`: armed, \
it is a NEW `sat` SOURCE on the path, and it is not free — it spends a bounded \
slice of the instance budget before the ordinary attack and the BaB verifier, \
so a row that would have been answered can time out instead.",
        provenance: Provenance::Unmeasured {
            why_ok: "Dark by default and env-only, so no scored run reaches it: with the \
                     variable absent the lane returns `Disarmed` before loading the model \
                     or parsing the property, which is byte-identical to the unwired tree \
                     (pinned by `defaults_are_unchanged_with_no_lever_set`). Armed it can \
                     only add a `sat` that the pre-existing trusted-oracle gate confirms \
                     against the ORIGINAL graph; it has no verified/unsat outcome, so it \
                     cannot cause a false `unsat` on any setting.",
        },
        owner: BNN_SIGN_SPACE,
        readers: BNN_STE_PGD_READERS,
    };

    /// `NY_BNN_SIGN_SPACE_MINIMAL_MOVE` — A/B switch for which point the
    /// sign-space realizability search adopts from each LP primal.
    pub BNN_SIGN_SPACE_MINIMAL_MOVE = LeverDecl {
        name: "NY_BNN_SIGN_SPACE_MINIMAL_MOVE",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::Low,
        doc: "\
Selects the SHAPE of one step of the sign-space realizability search. Dark (the \
default) the search adopts the realizability LP's primal VERTEX, which is what \
every banked traffic-signs measurement was taken on. Exact \"1\" adopts instead \
the MINIMAL point on the segment from the incumbent to that vertex: `z1` is \
linear in `x`, so `z1(x0 + t*(x_LP - x0))` is a closed form in `t` from two \
already-computed arrays, and the search walks only as far as the units the LP \
was asked to fix actually need. Every other byte string is a recorded rejection \
resolving to the `false` default.

WHY IT EXISTS. On `model_48`/`model_64` the vertex jump is destructive: the \
flipped unit sits 20-75 from its threshold, but the vertex breaks 40-60 OTHER \
free units, and lazy row generation then chases them (active set 4 -> 46 -> 88 \
-> ... -> 297) until the exact-rational LP hits its 1s per-solve cap. See \
`docs/BNN_SIGN_SPACE_FALSIFICATION_2026-08-12.md` section 10. The lever exists \
so the two shapes can be A/B'd on the SAME row before either becomes the \
default.

NEITHER ARM CAN AFFECT SOUNDNESS. Both only choose which in-box point the next \
round evaluates. Acceptance is still decided by evaluating every free unit's \
true OR-slack at a concrete point; the chosen point is a convex combination of \
two in-box points and its membership is re-checked against the vnnlib lo/hi \
exactly; and a candidate is still rebuilt from scratch and replayed through the \
unchanged `gate_sat_with_trusted_oracle` before publication. MoatRisk::Low \
because the arms can differ in WHICH rows are captured and in how long a lane \
consultation runs, not in whether a captured row is real.

IT DID NOT WIN, WHICH IS WHY IT IS STILL DARK. Measured verdict-neutral on all \
six rows it was built for (see the provenance below). The armed arm also ships \
its witnesses with 2-12x less f32 replay headroom, because it lands the point \
at exactly `TOL + f32_replay_slack_floor` rather than wherever the vertex sat. \
Spending headroom for no rows is a bad trade, so the default stays on the \
vertex and this remains an instrument.",
        provenance: Provenance::Measured {
            commit: "96d4cfa96",
            date: "2026-08-16",
            artifact: "docs/BNN_SIGN_SPACE_FALSIFICATION_2026-08-12.md section 10",
            delta: "Six traffic_signs rows at the official 480s budget, one at a time, \
                    NY_BNN count 0: NO verdict, margin or flip count moved on any of \
                    them. The three model_30 eps=1 rows stay sat with identical flip \
                    and LP counts (78/311, 99/370, 83/342) and identical logit margins \
                    (+2/+8/+4), differing only in witness slack (0.667/0.188/0.091 -> \
                    0.0566 on all three). The three deep-net rows keep 0 accepted \
                    flips and the same pattern-space margins (-110/-384/-110) on \
                    slightly fewer LP solves (130->118, 224->218, 199->196). The lever \
                    is live and the wall is elsewhere: the LP caps its slack column at \
                    1, so the vertex does not overshoot along the flip direction and \
                    the minimal move is ~98% of it.",
        },
        owner: BNN_SIGN_SPACE,
        readers: BNN_SIGN_SPACE_MOVE_READERS,
    };

    /// `NY_BNN_SIGN_SPACE_TRUST_REGION` — A/B switch for WHERE the sign-space
    /// realizability LP may put the pixel vector.
    pub BNN_SIGN_SPACE_TRUST_REGION = LeverDecl {
        name: "NY_BNN_SIGN_SPACE_TRUST_REGION",
        kind: LeverKind::Enum(&["box", "tight", "linf"]),
        default: DefaultSpec::Unset,
        bucket: Bucket::Debug,
        moat: MoatRisk::Low,
        doc: "\
Selects the BOUNDS the sign-space realizability LP gets on its pixel columns. \
Unset (the default) they are the vnnlib box, which is what every banked \
traffic-signs measurement was taken on. The three admissible arms replace them \
with an L-infinity trust region around the incumbent, expressed as a fraction \
of the box's widest half-width: \"box\" opens at 1/8, \"tight\" at 1/64, and \
\"linf\" opens at 1/64 and then spends 4 BISECTION steps between the last \
radius that failed and the first that worked. Any other byte string is a \
recorded rejection resolving to the unset default.

WHY IT EXISTS. The LP maximizes ONE column — a slack capped at 1 — and the \
~6900 pixel columns appear in no objective at all, so the LP places them \
anywhere in the box. Measured consequence on model_48/model_64: adopting the \
primal breaks 40-60 OTHER free units, lazy row generation chases them, and the \
active set grows 4 -> 46 -> 88 -> ... -> 297 without the worst slack \
converging. The MINIMAL_MOVE lever already measured out the forward reading of \
that (the slack cap means the vertex does not overshoot; the minimal move is a \
median 97.6% of it), which leaves SIDEWAYS travel, and this is the knob for it. \
The \"linf\" arm is the proximity objective the null result pointed at — \
min ||x - x0||_inf subject to the same rows — computed by bisecting the radius \
rather than by adding a column and 2*n_pixels rows to an exact-rational LP.

NEITHER ARM CAN AFFECT SOUNDNESS. Every bound is an INTERSECTION with the \
vnnlib box, so the restricted feasible set is a subset of the full one: a point \
it returns is in the box and is accepted only by evaluating every free unit's \
true OR-slack on it. And because shrinking the box can only make the LP more \
constrained, a restricted LP that fails proves NOTHING — so failure never \
declines the pattern, it doubles the radius and re-solves, and the last radius \
tried is always the full box, whose answer is exactly the shipped one. A \
candidate is still replayed through the unchanged trusted-oracle gate before \
publication.

IT IS DARK BECAUSE IT MOVED NO ROW, not because it did nothing. It fixes the \
convergence the section-10 diagnosis named -- the active set stops growing and \
the pattern becomes realizable -- and the deeper nets start accepting flips for \
the first time. What it then exposes is a THIRD wall: greedy single-flip \
margin progress. Section 9's rule is unchanged, so the default stays on the \
full box until an arm moves a verdict.",
        provenance: Provenance::Measured {
            commit: "42f18b13f",
            date: "2026-08-16",
            artifact: "docs/BNN_SIGN_SPACE_FALSIFICATION_2026-08-12.md section 10",
            delta: "IT MOVES THE DIAGNOSTIC AND NOT THE SCORE. On \
                    model_48_idx_1703_eps_3 at the 120s budget the first realizability \
                    call's active set goes 4 -> 46 -> 88 -> ... -> 245 without \
                    converging on the shipped arm, and 4 -> 17 -> 31 -> 42 -> 45 -> 48 \
                    -> 49 -> 49 (box/tight) or 4 -> 11 -> 23 -> 31 -> 36 -> 37 -> 38 -> \
                    38 (linf) WITH the worst true OR-slack crossing zero -- the pattern \
                    becomes realizable, which on this net it never was. Accepted flips \
                    go 0 -> 5 and the best pattern-space margin -110 -> -90. On all NINE \
                    eps>=3 rows abc falsifies, at the official 480s budget one row at a \
                    time on the same binary, the shipped arm accepts 0 flips and linf \
                    accepts 7-16, improving the margin by 20-120 (e.g. -264 -> -184 with \
                    16 flips on model_64_idx_6371_eps_3). NO VERDICT MOVED: all nine are \
                    timeout on both arms, because a violation needs margin > 0 and these \
                    start at -104 to -346. Dark until it moves a row.",
        },
        owner: BNN_SIGN_SPACE,
        readers: BNN_SIGN_SPACE_TRUST_READERS,
    };

    /// `NY_BNN_SIGN_SPACE_TRACE` — presence-gated per-round tracing of the
    /// sign-space realizability search's lazy row generation.
    pub BNN_SIGN_SPACE_TRACE = LeverDecl {
        name: "NY_BNN_SIGN_SPACE_TRACE",
        kind: LeverKind::Text,
        default: DefaultSpec::Unset,
        bucket: Bucket::Debug,
        moat: MoatRisk::Low,
        doc: "\
Enables one `NY_BNN_SIGN_SPACE_TRACE round=..` stderr line per lazy \
row-generation round inside ONE realizability test: the round index, the \
ACTIVE-SET size, how many rows that round added, how many free units are short \
of the tolerance, the worst true OR-slack, and — on the minimal-move arm — the \
theta the segment step chose. PRESENCE gate, like `NY_MIP_TRACE`: any present \
value arms it, including \"0\" and the empty string; absent emits nothing.

WHY IT EXISTS. The wall on the deeper traffic-signs nets is the CONVERGENCE of \
this loop, not admission: the active set was observed growing 4 -> 46 -> 88 -> \
... -> 297 while the worst slack refused to converge. That trajectory was \
measured with throwaway instrumentation, which meant the next question about it \
needed the instrumentation written again. This is that instrumentation, \
declared.

DIAGNOSTIC ONLY, AND NOT FREE. Every number printed is one the round already \
computed, and nothing about acceptance, the LP, or the witness path reads this \
lever — but the formatting and stderr traffic run inside a WALL-CLOCK-BUDGETED \
search, so an armed run can complete fewer rounds than an unarmed one. Arm it \
to understand a row, not to measure one.",
        provenance: Provenance::Unmeasured {
            why_ok: "dark by default and print-only; the armed arm perturbs the \
                     timing of a budgeted search, so it is deliberately not \
                     used for any banked verdict measurement",
        },
        owner: Scope {
            package: "ny-mip",
            subsystem: "bnn-sign-space",
        },
        readers: BNN_SIGN_SPACE_TRACE_READERS,
    };

    /// `NY_ATTACK_PRE_SOFTMAX_OBJECTIVE` — score the incumbent attack lane's
    /// search on PRE-Softmax logits when the proven strip guard admits the
    /// network and the property.
    pub ATTACK_PRE_SOFTMAX_OBJECTIVE = LeverDecl {
        name: "NY_ATTACK_PRE_SOFTMAX_OBJECTIVE",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Changes WHICH SCALAR the incumbent `gradient_guided_falsify_with_traffic_objective` \
lane hill-climbs. Exact \"1\" arms it and exact \"0\" disarms it; every other byte \
string is a RECORDED REJECTION resolving to this declaration's `false` default. \
Disarmed, the reader returns before the admission guard runs and before any \
pre-Softmax tensor is read, so that path is byte-identical to the unwired tree.

THE DEFECT IT ADDRESSES IS GENERAL, ITS MEASURED REACH IS NOT. Any network \
ending in Softmax with confident logits saturates the trusted forward's f32 \
output. Measured on `traffic_signs_recognition_2023`: 42 of 43 outputs are \
EXACTLY `0.0f` and the true class is `1.0f` at every sampled point of every \
row, so `property_margin` returns the SAME f64 bit pattern everywhere and the \
search hill-climbs a provably constant objective — `best_x` freezes at the seed \
and the clause pick degenerates to the last tied clause. Re-measured \
2026-08-18 through ONNX Runtime at 64 random points of the eps=1 box on all \
three shipped nets: the post-Softmax vector carries exactly TWO distinct \
values (`1.0f` and `0.0f`) at every point of all three, while the pre-Softmax \
logits at those same points carry 27-43 distinct values and span [-70, 164] \
(model_48), [-226, 532] (model_64) and [-1336, 2570] (model_30). BE PLAIN \
ABOUT THE REACH: a scan of every distinct ONNX named by every 2026 category \
(472 of the 480 model references resolve in the corpus; 8 ship no payload) \
found a \
TERMINAL Softmax in exactly ONE family, this one — 45 rows of the board, of \
which 36 are already won. `vit_2023` and `smart_turn_multimodal_2026` contain \
Softmax nodes but only inside attention and end in `Gemm`/`Sigmoid`, so the \
guard refuses them on the model side; every other family already emits raw \
logits, which is exactly why the defect never showed up elsewhere. The \
capability is general and cheap; its measured 2026 value is one family.

IT REUSES THE PROVEN PREDICATE AND WEAKENS NOTHING. Admission is one call to \
`ny_onnx::admit_pre_softmax_attack_scoring`, whose body is the SAME \
`strip_terminal_softmax_guard` that gates `NY_STRIP_TERMINAL_SOFTMAX`: terminal \
Softmax with one input/output, an authenticated integer axis, concrete equal \
in/out shapes, ONE normalization group, no other consumer, shape exactly \
`[1, num_outputs]`, and a spec that is exactly an argmax-complement disjunction \
of bare non-strict output-vs-output atoms sharing one true label. It therefore \
refuses a comparison against a CONSTANT (softmax preserves ORDER, not VALUE: \
`z=[0.6,10]` makes `Y_0 >= 0.5` true on logits and false on probabilities), \
outputs from DIFFERENT softmaxes, a non-terminal Softmax, the wrong axis, any \
linear combination, and a dual-network spec. Refusal is CHEAP — metadata checks \
only, no forward pass — and is LOGGED with its reason.

WHY THIS IS SAFER THAN THE BOUND-PATH STRIP, AND WHY IT NEEDS NO LATTICE \
CERTIFICATE. The strip changes what the solver PROVES, so it must clear the f32 \
tie window on exactly authenticated model bytes. This lever changes only which \
points the search VISITS. Acceptance is untouched: every candidate is still \
checked by `property_violated_f64` on the trusted ONNX Runtime forward of the \
ORIGINAL, UNMODIFIED graph, and every witness still passes the UNCHANGED \
`gate_sat_with_trusted_oracle`. A mis-scored search finds nothing; it is \
structurally incapable of emitting a verdict, so it cannot emit a wrong one.

MoatRisk::High is about TIME, not soundness. Armed, the lane computes one \
extra non-certified ny forward per step to read the logits, and — more \
importantly — a lane that was plateaued and is now actually searching spends \
more of its slice. Rows banked at up to 454.8s against a 480s budget have \
about 25s of headroom, so an armed A/B can convert a banked `sat` into a \
`timeout`. That is why the shipped default is off and changes only on a \
measurement.",
        provenance: Provenance::Unmeasured {
            why_ok: "Dark by default and env-only, so no scored run reaches it: with the \
                     variable absent the reader returns before the admission guard, no \
                     pre-Softmax tensor is read, and the objective, direction row and DLR \
                     denominator are the historical post-Softmax ones (pinned by \
                     `defaults_are_unchanged_with_no_lever_set`). Armed it changes only \
                     which points the search visits; acceptance stays the unchanged ORT \
                     forward on the ORIGINAL graph plus the unchanged trusted-oracle \
                     gate, so it can neither publish an unconfirmed witness nor produce \
                     any `unsat`.",
        },
        owner: ATTACK_OBJECTIVE,
        readers: PRE_SOFTMAX_ATTACK_OBJECTIVE_READERS,
    };

    /// `NY_ENVELOPE_XSTAR_PROBE` — dark x* envelope diagnostics.
    pub ENVELOPE_XSTAR_PROBE = LeverDecl {
        name: "NY_ENVELOPE_XSTAR_PROBE",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::Low,
        doc: "\
Emits the x* envelope diagnostic from both alpha-gradient paths. Exact \"1\" \
arms it; absence and every other byte string leave it dark, matching the \
readers' `== Some(\"1\")` test verbatim. The output feeds no bound, but its \
formatting and stderr traffic can perturb a deadline-sensitive run, which is \
why this is MoatRisk::Low rather than None.",
        provenance: Provenance::Unmeasured {
            why_ok: "dark by default and diagnostic only; armed-vs-unarmed \
                     deadline and verdict parity has not been measured",
        },
        owner: GRAPH_ALPHA,
        readers: ENVELOPE_XSTAR_READERS,
    };

    /// `NY_ENVELOPE_RESCALE_PROBE` — dark envelope-rescale diagnostics.
    pub ENVELOPE_RESCALE_PROBE = LeverDecl {
        name: "NY_ENVELOPE_RESCALE_PROBE",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::Low,
        doc: "\
Gates the envelope-rescale diagnostic on both alpha-gradient paths; the DAG \
site additionally restricts it to the first iterate (`k == 0`). Exact \"1\" \
arms it. Same dark-by-default, formatting-only cost profile as \
`NY_ENVELOPE_XSTAR_PROBE`.",
        provenance: Provenance::Unmeasured {
            why_ok: "dark by default and diagnostic only; armed-vs-unarmed \
                     deadline and verdict parity has not been measured",
        },
        owner: GRAPH_ALPHA,
        readers: ENVELOPE_RESCALE_READERS,
    };

    /// `NY_INPUT_SPLIT_PROBE` — dark input-split rebound diagnostics.
    pub INPUT_SPLIT_PROBE = LeverDecl {
        name: "NY_INPUT_SPLIT_PROBE",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::Low,
        doc: "\
Prints the `[input-split-rebound]` line carrying the domain count, whether the \
nested deadline is finite, whether it was dropped, and whether the rebound \
stacked. Exact \"1\" arms it. Diagnostic only; it reports the state that \
`NY_INPUT_SPLIT_NESTED_DEADLINE` controls rather than changing it.",
        provenance: Provenance::Unmeasured {
            why_ok: "dark by default and diagnostic only; armed-vs-unarmed \
                     deadline and verdict parity has not been measured",
        },
        owner: INPUT_SPLIT,
        readers: INPUT_SPLIT_PROBE_READERS,
    };

    /// `NY_INPUT_SPLIT_NESTED_DEADLINE` — A/B the nested alpha deadline away.
    pub INPUT_SPLIT_NESTED_DEADLINE = LeverDecl {
        name: "NY_INPUT_SPLIT_NESTED_DEADLINE",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(true),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Ships ARMED. Exact \"0\" drops the nested alpha deadline on the input-split \
rebound, restoring the pre-`6f49a660` shape for an A/B; absence and every other \
value keep the deadline. Note the polarity is inverted relative to the other \
levers here — the reader tests `== Some(\"0\")`, so this is an OPT-OUT, and the \
declaration's `Bool` default of `true` records the shipped arm.

MoatRisk::High because dropping the deadline removes this rebound's \
INTERRUPTIBILITY: the alpha pass can then run past the point the caller \
intended to reclaim, which changes what the remaining budget can prove. Both \
arms are sound — this is a scheduling shape, not a bound — but the opt-out must \
never become the shipped default, and the reader's own comment says so.",
        provenance: Provenance::Unmeasured {
            why_ok: "the shipped arm is the deadline-carrying one; the opt-out \
                     exists only to reproduce the pre-6f49a660 shape in an A/B \
                     and has no measured row-level comparison",
        },
        owner: INPUT_SPLIT,
        readers: INPUT_SPLIT_DEADLINE_READERS,
    };

    /// `NY_FALSIFY_PORTFOLIO` — admits the ported `ny-falsify` strategy
    /// portfolio into the attack slice.
    pub FALSIFY_PORTFOLIO_LANE = LeverDecl {
        name: "NY_FALSIFY_PORTFOLIO",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Admits the `ny-falsify` strategy portfolio (S1 `special`, S9 `square`) inside \
the same attack slice as the two binarized-net lanes, after them and before the \
ordinary upfront attack. Exact \"1\" arms it and exact \"0\" disarms it; every \
other byte string is a RECORDED REJECTION that resolves to this declaration's \
`false` default. On the disarmed arm the reader returns BEFORE the property is \
parsed, before a search box is built and before any ONNX Runtime session is \
constructed — so that path is byte-identical to the unwired tree.

WHY IT EXISTS. `scripts/audit_unsat_by_falsification.py` has run for months as \
a one-sided unsat AUDITOR and has never influenced a verdict. Its self-test \
(`reports/falsification_audit/selftest_calibration.json`, 60 s, 100 known-SAT \
rows) refuted 75 of them, and the winner attribution names SIX strategies with \
no dominance among them: `special` won 34 rows at 2% of the plan budget, \
`square` won the only `soundnessbench` and the only `traffic_signs` row anything \
took. Those two are the ones ny ships no equivalent of — ny's corner lane is \
capped at five variable dimensions and emits vertices only, while `special` won \
a 200-free-input row and four of its eight patterns are interior; and `square` \
is block sign-flip hill climbing against a FLAT objective, which is exactly the \
`#deadlane` set where every estimated gradient is identically zero.

IT IS AN ENVIRONMENT INSTRUMENT AND NOT A PRESET KEY, for the same reason \
`NY_BNN_STE_PGD` was not one before its sweep: a competition harness exports no \
`NY_*`, so an env-only lever cannot reach a scored run. Promoting it to the \
typed `attack.*` config layer requires an armed-versus-unarmed sweep of a whole \
family that measures a win with no regression. No such sweep exists. The \
measurement that gated the port (E1: 42 open-row measurements on `cora_2024`, \
`traffic_signs` and `soundnessbench`, at official and 10-12x budgets) returned \
ZERO counterexamples.

WHAT THE LANE CAN AND CANNOT DO. `ny-falsify` has no dependency on any \
workspace crate, so `VnncompResult` is not nameable inside it; its return type \
`Proposal` is `Candidate` / `Exhausted` / `Declined` and has no verdict-bearing \
variant by construction. A `Candidate` carries INPUTS ONLY — there is no output \
vector on the type — so no search arithmetic can reach a published witness. The \
candidate is rendered input-only, its `Y_j` values are supplied by a real ONNX \
Runtime forward on the ORIGINAL graph, and it becomes a `sat` only by passing \
the EXISTING, UNCHANGED `gate_sat_with_trusted_oracle`. A candidate the gate \
drops falls through to the unchanged verification path.

MoatRisk::High for the two honest reasons `NY_BNN_SIGN_SPACE` carries: armed, \
this is a NEW `sat` SOURCE on the scored path, and it is not free — it spends a \
bounded slice of the instance budget ahead of the ordinary attack and the BaB \
verifier, so a row that would have been answered can time out instead.",
        provenance: Provenance::Unmeasured {
            why_ok: "Dark by default and env-only, so no scored run reaches it: with the \
                     variable absent the reader returns before the property parse, the \
                     search-box build and the ORT session, which is byte-identical to the \
                     unwired tree. Armed it can only add a `sat` that the pre-existing \
                     trusted-oracle gate confirms against the ORIGINAL graph; the crate \
                     behind it has no verdict-bearing return variant, so it cannot cause a \
                     false `unsat` on any setting. The armed arm WAS measured and gained \
                     NOTHING: 0 rows on 27 open `cora_2024` rows and 21 open \
                     `challenging_certified_training_2026` rows, at the official slice and \
                     at ~9x it (31.0M trusted ORT forwards), with verdict-identical \
                     dark-vs-armed results on a 30-row decided-`cora_2024` sample, 4 \
                     `traffic_signs` rows and 3 `monotonic_acasxu_2026` rows -- \
                     `reports/falsification_audit/wired_portfolio_2026-08-19/`. That zero \
                     is why the default did not move.",
        },
        owner: FALSIFY_PORTFOLIO,
        readers: FALSIFY_PORTFOLIO_READERS,
    };

    /// `NY_FALSIFY_PORTFOLIO_SECONDS` — wall cap for the portfolio phase.
    pub FALSIFY_PORTFOLIO_SECONDS = LeverDecl {
        name: "NY_FALSIFY_PORTFOLIO_SECONDS",
        kind: LeverKind::U64Trimmed,
        default: DefaultSpec::U64(0),
        bucket: Bucket::Debug,
        moat: MoatRisk::Low,
        doc: "\
Hard wall-clock cap, in whole seconds, on the `ny-falsify` portfolio phase. \
Zero (the default) selects the derived rule instead: 8% of the remaining \
instance budget less a 3 s publication margin, capped at 60 s — the fraction \
the upfront attack already uses, and the total budget the calibration self-test \
was run at. A malformed value falls back to the default, matching the legacy \
`trim().parse::<u64>()` readers this kind exists for.

IT IS UNREACHABLE UNLESS `NY_FALSIFY_PORTFOLIO` IS ALREADY ARMED — the reader \
runs after the arming gate, so on a default run this declaration has no reader \
at all. It exists so an over-budget A/B (does the portfolio convert an open row \
given 10x the seconds?) can be run without a rebuild; every arm is still \
clamped by the instance deadline through `bounded_work_deadline`, so it cannot \
push past the scored deadline whatever it is set to.

MoatRisk::Low: it chooses how long a falsification search runs, never whether a \
candidate is accepted. Acceptance is the unchanged `gate_sat_with_trusted_oracle` \
on every arm.",
        provenance: Provenance::Unmeasured {
            why_ok: "unreachable behind a dark lever; the default of 0 selects the same \
                     derived budget the lane would have had with no declaration at all",
        },
        owner: FALSIFY_PORTFOLIO,
        readers: FALSIFY_PORTFOLIO_SECONDS_READERS,
    };


    /// `NY_LANE_BUDGET_ALLOCATOR` — chooses the attack-slice caps jointly and
    /// up front by solving the Layer-A multiple-choice knapsack.
    pub LANE_BUDGET_ALLOCATOR = LeverDecl {
        name: "NY_LANE_BUDGET_ALLOCATOR",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Admits LAYER A of the per-instance lane budget allocator over the attack slice: \
a multiple-choice knapsack over per-lane cap ladders, solved exactly on the ay \
backend under a 10 ms wall, that commits every lane's cap BEFORE the first lane \
starts. Exact \"1\" arms it and exact \"0\" disarms it; every other byte string is \
a RECORDED REJECTION resolving to this declaration's `false` default. Disarmed, \
the reader returns before the objective probe is taken, before an \
`AllocationRequest` is built and before the solver is entered, and each lane \
derives exactly the window it derives today, from the same private helper, in \
the same order -- the disarmed arm is the unwired tree.

HOW IT COMPOSES WITH `NY_LANE_VALUE_SCHEDULER`. It does not replace it. The \
scheduler is a strict pipeline walk that handles yields IN FLIGHT: a stalled \
lane's unspent seconds return to the pool for the next lane's cap. This lever \
chooses the caps JOINTLY and UP FRONT, which a greedy marginal-value walk \
provably cannot do when the value of a cap is step-like in the cap -- every \
block before the step has marginal value zero, so a greedy rule never climbs \
it. When both are armed the scheduler owns the two BNN lanes and this allocator \
stands down, so there is exactly one authority over a lane's cap at any time.

THE DEFECT IT ADDRESSES, MEASURED. Per-lane ledger, `traffic_signs` \
`model_48_idx_1703_eps_1`, 480 s budget, wall 452.81 s, timeout: the LP \
sign-space lane held 217.52 s for 370 LP solves, 34 flips and NO candidate; \
STE-PGD held 117.51 s; the upfront DLR-APGD lane held 71.00 s for 405 gradient \
steps and NO candidate; branch-and-bound then got 46.78 s of a 50 s grant and \
explored ZERO domains. The control row `model_30_idx_1703_eps_1`, which ny \
WINS, spent 131.04 s of the same 217.5 s LP cap and produced the candidate \
before any later lane was reached -- so on the rows the LP lane wins the cap is \
not binding, and on the rows it stalls the cap is pure waste.

WHAT IT CHANGES WHEN ARMED. (i) One pool, the SUM OF THE CAPS TODAY'S FIXED \
FRACTIONS WOULD HAND THE SAME LANES, so the branch-and-bound residual claimant \
can never be handed less than it gets today and no row can exceed its official \
budget. (ii) A structural zero -- a FLAT objective tier, one distinct float32 \
value over the in-box probe -- pins a gradient-guided lane that steers on that \
same objective to ZERO seconds, and the lane is SKIPPED rather than run under a \
small cap. (iii) The freed seconds land on a cap ladder rung the receiving lane \
can plan against, never as a dribble of leftover seconds; the measured target \
is STE-PGD's declared 240 s `max_wall_time` rung, the cap that won three rows \
that 217.5 s did not. (iv) Anything but a proven optimum inside 10 ms FAILS \
OPEN to exactly today's plan.

MoatRisk::High for the honest reason `NY_LANE_VALUE_SCHEDULER` carries: armed, \
this changes HOW MUCH BUDGET each falsification lane gets on a scored row, and \
it can take a lane to zero, so a row that would have been answered can be \
answered differently. It never changes WHETHER a candidate is accepted -- that \
is the unchanged `gate_sat_with_trusted_oracle` on every arm -- and it cannot \
produce an `unsat`, because no allocated lane has a verdict-bearing return type \
and `ny_mip::lane_allocation`'s public surface cannot name one.",
        provenance: Provenance::Unmeasured {
            why_ok: "DARK by default, so no scored run reaches it: a competition harness \
                     exports no `NY_*` and this lever has no typed preset key. The \
                     allocator itself is verdict-neutral by construction (it selects \
                     which lane runs under what cap and can never change what a lane may \
                     publish), so the worst outcome on the armed arm is a MISSED row and \
                     never a WRONG one. The A/B that would make it Measured is a \
                     leave-one-family-out sweep at scored budgets across the 2026 board, \
                     which the development host cannot run: it throttles under sustained \
                     load and cifar100 root bootstrap alone costs 71 s here. Until that \
                     sweep exists the default must stay off.",
        },
        owner: LANE_ALLOCATOR,
        readers: LANE_BUDGET_ALLOCATOR_READERS,
    };

    /// `NY_LANE_VALUE_SCHEDULER` — routes the attack slice through the
    /// marginal-value lane ledger instead of a chain of private fractions.
    pub LANE_VALUE_SCHEDULER = LeverDecl {
        name: "NY_LANE_VALUE_SCHEDULER",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Admits the cross-lane marginal-value scheduler over the per-instance attack \
slice. Exact \"1\" arms it and exact \"0\" disarms it; every other byte string is \
a RECORDED REJECTION resolving to this declaration's `false` default. Disarmed, \
the reader returns before any ledger is constructed and each lane derives \
exactly the window it derives today, from the same private helper, in the same \
order — the disarmed arm is the unwired tree.

THE DEFECT IT ADDRESSES, MEASURED. The per-instance budget is carved into fixed \
slices handed to lanes that run blind. On `traffic_signs model_48_idx_1703_eps_1` \
the LP sign-space lane held 217.52 s, accepted 34 flips, moved the pattern \
margin to -82 and produced NOTHING; the STE-PGD lane behind it then computed a \
117.51 s cap from what was left. On `model_64_idx_1703_eps_1` the same LP lane \
yielded at 53.56 s and STE's cap computed to its 240.10 s ceiling — a 163.96 s \
yield became a 122.59 s CAP increase, and the cap is a SCHEDULE, not a quantity \
(the STE stage boundary moved 88.1 s -> 180.1 s with it). That reallocation \
happened only because the trailing lane sizes itself by subtraction from the \
LIVE remaining; every other lane in the file sizes itself by a private fraction \
of what it was HANDED, so a yield upstream of them evaporates.

WHAT IT CHANGES WHEN ARMED. (i) One pool, `remaining - the 45 s publication \
margin`, that the scheduled lanes may never collectively exceed. (ii) Every \
lane's cap is re-derived from the LIVE remaining at ITS OWN admission, through \
that lane's own unchanged plan function, so an upstream yield propagates as a \
CAP the downstream lane can plan against. (iii) Each lane reports its actual \
cost and its progress in its own work units (LP solves, probes) — a lane that \
declines is charged ZERO, a lane that overruns is charged what it took. (iv) It \
arms the LP walk's value-based stall rule, `stall_margin_lp_solves`, which asks \
whether the pattern margin has MOVED rather than the shipped rule's question of \
whether any flip was ever accepted; the measured 34-flip/-82-margin row disarms \
the shipped rule permanently.

MoatRisk::High for the honest reason `NY_BNN_SIGN_SPACE` carries: armed, this \
changes HOW MUCH BUDGET each falsification lane gets on a scored row, so a row \
that would have been answered can be answered differently. It never changes \
WHETHER a candidate is accepted — that is the unchanged \
`gate_sat_with_trusted_oracle` on every arm — and it cannot produce an `unsat`, \
because no scheduled lane has a verdict-bearing return type.",
        provenance: Provenance::Measured {
            commit: "661f4ef63129f6025e95b31e163fa68b5501230a",
            date: "2026-08-19",
            artifact: "reports/measured-2026/traffic_signs_recognition_2023_NOTES.md \
                       (section: the lane-value scheduler A/B)",
            delta: "ZERO ROWS. Nine open `traffic_signs` rows, armed vs dark, sequential \
                    at the official 480 s budget on the development host: 9/9 `timeout` \
                    on BOTH arms. The reallocation itself is real and is the thing that \
                    was proven -- on the FIVE rows where the LP walk accepts flips (so \
                    the shipped accepted-flip stall rule never fires) it burned 215.9- \
                    217.5 s dark and 8.0-86.0 s armed, freeing 957.0 s, of which the \
                    STE-PGD lane absorbed 614.7 s as a CAP increase from 116.0-117.5 s \
                    to 240.0 s, roughly doubling its work (e.g. 1782 -> 3515 LP solves \
                    on `model_48_idx_1703_eps_1`). It bought nothing: STE exhausted at \
                    240 s on every one, and on `model_64_idx_178_eps_1` the DOUBLED cap \
                    scored WORSE (margin gain +164 at 117.0 s vs +140 at 240.1 s), which \
                    is budget-parameterization cutting the other way. The remaining four \
                    rows are structurally inert: the LP lane already refuses on \
                    `max_free_units` in 0.03-0.05 s or stalls at 32 LP solves, so both \
                    arms already give STE its 240 s `LANE_WALL_CAP` ceiling -- and that \
                    ceiling, not the yield, is what now binds. Regression: a full \
                    45-row armed sweep held all 15 `model_30` rows and 5 of the 6 \
                    STE-gained rows; the sixth (`model_48_idx_178_eps_5`) timed out \
                    1.5 h into sustained load and reproduced `sat` in 4/4 isolated \
                    controls (dark 207.2/188.1 s, armed 188.4/188.2 s), where both arms \
                    are byte-identical (same refusal, same 760 flips / 429 LP solves). \
                    Max wall 456.8 s against the 480 s budget, 0 `error` rows.",
        },
        owner: LANE_SCHEDULER,
        readers: LANE_VALUE_SCHEDULER_READERS,
    };

}
