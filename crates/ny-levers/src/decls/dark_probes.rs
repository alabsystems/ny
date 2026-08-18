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
//! All four are dark by default and diagnostic, EXCEPT
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

const BNN_SIGN_SPACE_READERS: &[ReaderSite] = &[ReaderSite {
    scope: BNN_SIGN_SPACE,
    role: "the admission of the LP-guided sign-space falsification lane, over the typed \
           `attack.bnn_sign_space` preset layer; disarmed, the lane returns its \
           `Disarmed` outcome before it reads the model, the property, or constructs \
           any `SignSpaceRequest`",
    site: "crates/ny-cli/src/commands/beta_crown/sign_space_falsify.rs \
           (sign_space_falsify_armed)",
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

}
