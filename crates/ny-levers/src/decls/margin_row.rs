// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The margin-row (#twinwall) lane's three BUILT-BUT-UNMEASURED programs:
//! `#margin-row-beta`, `#backward-interm`, and the certified GPU backward
//! (`#margin-row-gpu-eft`).
//!
//! These are grouped because they share one property that decides every
//! classification below: each one is DARK by default, and each one, when armed,
//! changes what the lane's ONE certified concretize is handed — which makes it
//! verdict-affecting on the authoritative route rather than a diagnostic. The
//! margin-row lane publishes `Unsat`; anything that moves a bound it publishes
//! is [`MoatRisk::High`] here even where a structural argument says the bound
//! stays sound.
//!
//! * [`MARGIN_ROW_BETA`] (+ [`MARGIN_ROW_BETA_ETA`], [`MARGIN_ROW_BETA_ITERS`])
//!   attach one `beta_j >= 0` Lagrangian term per split so a child domain's
//!   bound can actually improve on its parent's. Weak duality makes each term
//!   valid for ANY `beta >= 0`, and the unchanged certified pass is still the
//!   scorer that accepts or rejects a proposal — but the accepted proposal
//!   changes the PUBLISHED bound, and each trial costs one more certified pass
//!   against the same deadline.
//! * [`MARGIN_ROW_BACKWARD_INTERM`] (+ [`MARGIN_ROW_BI_SECS`],
//!   [`MARGIN_ROW_BI_CHUNK`], [`MARGIN_ROW_BI_TOPK`]) recompute each trunk
//!   ReLU's input box with the lane's own certified backward engine and
//!   shrink-only intersect it with the forward tableau box. Intersecting two
//!   valid enclosures is sound by construction, and skipping any layer, chunk
//!   or neuron is always sound — but `backward_interm.rs` documents ONE
//!   ORDERING TRAP (`LayerGates::clip_rows` slack must be calibrated against
//!   the FORWARD-only bounds, or every Clip-and-Verify halfspace built from the
//!   line cuts into the true subdomain and becomes a false-`unsat` generator).
//!   A lever that arms a phase with a named false-`unsat` failure mode is High.
//! * [`MARGIN_ROW_GPU_EFT`] gates a lane that is designed to be
//!   GPU-AUTHORITATIVE, not a shadow. Its device transaction is not delivered
//!   yet, so today an armed run stops at the authority gate with
//!   `Refusal::Unimplemented`; the declaration is classified for what the gate
//!   admits, exactly as `NY_EFT_ERR`'s staged CPU arm is.
//!
//! NONE of them carries a measurement. `docs/BIG3_CORE_CAMPAIGN_2026-08-19.md`
//! and `docs/MARGIN_ROW_BETA_AND_BACKWARD_INTERM_2026-08-19.md` both record
//! these as "BUILT, unmeasured", which is why every declaration here is
//! [`Bucket::Debug`] + [`Provenance::Unmeasured`] and none may become
//! `DefaultOn` until a retained scored-row A/B exists.

use crate::{
    declare_levers, Bucket, DefaultSpec, LeverDecl, LeverKind, MoatRisk, Provenance, ReaderSite,
    Scope,
};

const BETA: Scope = Scope {
    package: "ny-propagate",
    subsystem: "margin-row-beta",
};

const BACKWARD_INTERM: Scope = Scope {
    package: "ny-propagate",
    subsystem: "margin-row-backward-interm",
};

const GPU_EFT: Scope = Scope {
    package: "ny-propagate",
    subsystem: "margin-row-gpu-eft",
};

/// All three `#backward-interm` tuning knobs are resolved by the same function,
/// per BUILD rather than through a process-wide latch, so a sealed A/B or a
/// unit test can flip arms without restarting the process.
const BI_TUNING_READER: ReaderSite = ReaderSite {
    scope: BACKWARD_INTERM,
    role: "size the backward-intermediate phase's budget, pass width and per-layer selection",
    site: "crates/ny-propagate/src/margin_row/backward_interm.rs:from_env",
};

const BETA_LAMBDA_READERS: &[ReaderSite] = &[ReaderSite {
    scope: BETA,
    role: "size the Polyak step as a fraction of the direct-path gap",
    site: "crates/ny-propagate/src/margin_row/beta.rs:lambda",
}];

const BETA_POLYAK_READERS: &[ReaderSite] = &[ReaderSite {
    scope: BETA,
    role: "opt OUT of Polyak step sizing, falling back to the fixed eta step",
    site: "crates/ny-propagate/src/margin_row/beta.rs:polyak_enabled",
}];

const BETA_HEADS_READERS: &[ReaderSite] = &[ReaderSite {
    scope: BETA,
    role: "opt OUT of head-split beta terms",
    site: "crates/ny-propagate/src/margin_row/beta.rs:heads_enabled",
}];

declare_levers! {
    registry MARGIN_ROW_LEVERS;

    /// `NY_MARGIN_ROW_BETA_LAMBDA` — Polyak relaxation factor.
    pub MARGIN_ROW_BETA_LAMBDA = LeverDecl {
        name: "NY_MARGIN_ROW_BETA_LAMBDA",
        kind: LeverKind::F64ClosedTrimmed { min: 0.0, max: f64::MAX },
        default: DefaultSpec::F64(1.0),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
The fraction of the direct-path gap a Polyak step is sized to close. Parser is \
`trim().parse::<f64>()`, finite, with the `> 0.0` half of the legacy filter \
left AT THE READER so an explicit `0` still resolves and stays distinguishable \
from absence in the receipt; both land on the 1.0 default, so this is a \
receipt-fidelity choice and not a behaviour change.

Inert unless `NY_MARGIN_ROW_BETA` is armed, which ships dark. MoatRisk::High \
because when the beta lane IS armed this scales every ascent step and so \
changes which certified bound the authoritative margin-row lane publishes.",
        provenance: Provenance::Unmeasured {
            why_ok: "the beta program is recorded BUILT-but-unmeasured; this tunes a \
                     lane that is itself dark, so no shipped path reads it",
        },
        owner: BETA,
        readers: BETA_LAMBDA_READERS,
    };

    /// `NY_MARGIN_ROW_BETA_POLYAK` — opt OUT of Polyak step sizing.
    pub MARGIN_ROW_BETA_POLYAK = LeverDecl {
        name: "NY_MARGIN_ROW_BETA_POLYAK",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(true),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
OPT-OUT, and the polarity is the point: the reader tests `!= Some(\"0\")`, so \
absence and every token other than exact `0` leave Polyak sizing ENGAGED. The \
declaration records that shipped arm as `DefaultSpec::Bool(true)`.

`DefaultSpec::Bool(true)` here does NOT mean this lever arms anything on a \
scored run. It is a sub-switch of `NY_MARGIN_ROW_BETA`, which ships dark, so \
the whole beta lane — and therefore this choice — is inert by default. That is \
why the group test admits an armed default for it and for \
`NY_MARGIN_ROW_BETA_HEADS` while forbidding one anywhere else in this module.",
        provenance: Provenance::Unmeasured {
            why_ok: "sub-switch of a dark master gate; unreachable on any shipped path",
        },
        owner: BETA,
        readers: BETA_POLYAK_READERS,
    };

    /// `NY_MARGIN_ROW_BETA_HEADS` — opt OUT of head-split beta terms.
    pub MARGIN_ROW_BETA_HEADS = LeverDecl {
        name: "NY_MARGIN_ROW_BETA_HEADS",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(true),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
OPT-OUT with the same shape and the same reasoning as \
`NY_MARGIN_ROW_BETA_POLYAK`: exact `0` disables head-split beta terms, \
everything else leaves them engaged, and the lane is inert while \
`NY_MARGIN_ROW_BETA` ships dark.",
        provenance: Provenance::Unmeasured {
            why_ok: "sub-switch of a dark master gate; unreachable on any shipped path",
        },
        owner: BETA,
        readers: BETA_HEADS_READERS,
    };


    /// `NY_MARGIN_ROW_BETA` — the lane's β-CROWN split Lagrangians.
    pub MARGIN_ROW_BETA = LeverDecl {
        name: "NY_MARGIN_ROW_BETA",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Arms `#margin-row-beta`: one `beta_j >= 0` Lagrangian term per branch-and-bound \
split in the margin-row lane's backward pass, plus the head-split terms carried \
on the margin seed. Exact `\"1\"` arms and exact `\"0\"` disarms; every other byte \
string (`\"true\"`, `\" 1\"`, `\"01\"`, `\"\"`, non-Unicode) is a RECORDED REJECTION \
that resolves to this declaration's `false` default. An armed run announces \
itself once on stderr (`[beta] armed`) even if no domain ever carries a split, \
so an INERT arming is detectable in one log line — that telemetry is why the \
decision is latched in a `OnceLock`.

WHY IT EXISTS: without it the lane's splits only piece-fix a neuron's gate, so \
the split constraint never enters the bound as a dual term and children barely \
improve on parents (measured frontier explosion: idx_8600 goes 18 -> 415 open \
domains at depth 30 while idx_6659 drains and proves).

WHY High. The soundness argument is strong — weak duality makes \
`min_region f >= min_region [f - sum_j beta_j s_j z_j]` hold for ANY \
`beta >= 0`, the engine realizes the terms as coefficient shifts BEFORE the one \
unchanged certified concretize, and a proposal is accepted only because that \
same certified pass reported a better bound. But High is not a soundness \
score: armed, this changes which certified bound the lane PUBLISHES for every \
domain that carries a split, and it spends extra certified passes against the \
instance deadline, so it can convert a row in either direction. Disarmed the \
`DomainGates::beta` map stays empty, the seed is the untouched base margin seed, \
and the engine's application site is never entered — bit-identical passes.",
        provenance: Provenance::Unmeasured {
            why_ok: "dark by default and bit-identical disarmed; the campaign ledger \
                     records this build as BUILT, unmeasured, and no armed-vs-unarmed \
                     scored-row A/B is retained, so it cannot leave Bucket::Debug",
        },
        owner: BETA,
        readers: &[ReaderSite {
            scope: BETA,
            role: "arm the lane's β-CROWN split Lagrangians (latched once, with engagement telemetry)",
            site: "crates/ny-propagate/src/margin_row/beta.rs:enabled",
        }],
    };

    /// `NY_MARGIN_ROW_BETA_ETA` — step size for the LEGACY sign-step arm.
    pub MARGIN_ROW_BETA_ETA = LeverDecl {
        name: "NY_MARGIN_ROW_BETA_ETA",
        // Trimmed, finite, unbounded above: the legacy reader is
        // `trim().parse::<f64>()` with NO upper filter, so `max` is `f64::MAX`
        // rather than an invented ceiling. The lower end is CLOSED at zero on
        // purpose — see the doc: the `> 0.0` half of the filter stays at the
        // reader so an explicit `0` is still distinguishable from absence.
        kind: LeverKind::F64ClosedTrimmed { min: 0.0, max: f64::MAX },
        default: DefaultSpec::F64(0.5),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Relative β ascent step size, in units of the split neuron's incoming \
`|coefficient|`, default 0.5. Surrounding whitespace is trimmed before \
`parse::<f64>()`; the reader then keeps the legacy `is_finite() && > 0.0` \
filter, so an explicit `0`, a negative value, `nan`, `inf` and any malformed \
token all leave 0.5. Latched once per process.

IT ONLY APPLIES TO ONE ARM. The shipped step rule is the gap-targeted Polyak \
step `t = lambda*(0 - b_direct)/sum g^2`, which IGNORES eta entirely; this knob \
is read only when `NY_MARGIN_ROW_BETA_POLYAK=0` selects the legacy sign-only \
step `±eta*|v_k|`. It is therefore inert twice over on a shipped run: β must be \
armed AND the Polyak rule must be disarmed.

High rather than Low because when it does apply it changes which β proposals \
the certified scorer accepts, and therefore which certified bound the \
authoritative margin-row lane publishes — a different value here is a different \
proof attempt, not a different amount of logging.",
        provenance: Provenance::Unmeasured {
            why_ok: "doubly inert on the shipped configuration (β is dark, and the \
                     Polyak rule that ignores eta is the default step rule when it is not); \
                     no retained sweep qualifies any other step size",
        },
        owner: BETA,
        readers: &[ReaderSite {
            scope: BETA,
            role: "step size for the legacy sign-only β step (the NY_MARGIN_ROW_BETA_POLYAK=0 A/B arm)",
            site: "crates/ny-propagate/src/margin_row/beta.rs:eta",
        }],
    };

    /// `NY_MARGIN_ROW_BETA_ITERS` — β ascent trials per domain evaluation.
    pub MARGIN_ROW_BETA_ITERS = LeverDecl {
        name: "NY_MARGIN_ROW_BETA_ITERS",
        kind: LeverKind::UsizeTrimmed,
        default: DefaultSpec::U64(1),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
β ascent trials per expanded domain, default 1. Whitespace is trimmed before \
`parse::<usize>()`; absent, malformed, negative and overflowing input all leave \
1, and the reader then CLAMPS the resolved value to `1..=8`, so an in-range \
number outside that window is pulled to the nearest bound rather than rejected. \
Latched once per process.

EACH TRIAL IS ONE MORE CERTIFIED PASS on every expanded domain that carries a \
split. That is the whole cost model of `#margin-row-beta`, which is why this \
is High and not Low: raising it both changes which β is accepted (hence the \
published bound) and multiplies the lane's per-expansion cost against a fixed \
instance deadline, so it can lose rows the shipped setting would have proved. \
Inert unless NY_MARGIN_ROW_BETA is armed.",
        provenance: Provenance::Unmeasured {
            why_ok: "inert while its parent lever is dark, which is the shipped state; \
                     the runbook proposes sweeping 3 as an experiment, which is a plan, \
                     not a measurement",
        },
        owner: BETA,
        readers: &[ReaderSite {
            scope: BETA,
            role: "ascent trials per domain evaluation for the lane's β-CROWN",
            site: "crates/ny-propagate/src/margin_row/beta.rs:iters",
        }],
    };

    /// `NY_MARGIN_ROW_BACKWARD_INTERM` — backward-computed root intermediates,
    /// layered OVER the typed preset.
    pub MARGIN_ROW_BACKWARD_INTERM = LeverDecl {
        name: "NY_MARGIN_ROW_BACKWARD_INTERM",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Arms `#backward-interm`: each trunk ReLU's INPUT box is recomputed with the \
lane's own certified backward engine (identity rows seeded at that layer's \
pre-activation, run through the already-frozen prefix gates, concretized over \
the root box — alpha-CROWN intermediate bounds in the lane's arithmetic) and \
INTERSECTED shrink-only with the forward (M, D) tableau box before \
`gates_from_box` derives `(alpha, s, c, ms)`. It runs DURING the forward build, \
layer by layer, so each layer's pass consumes gates that earlier layers have \
already tightened and the effect COMPOUNDS down the trunk. Root build only \
(`splits.is_empty()`), never inside Tier-2 epoch rebuilds, and never in the \
bit-parity `RoundMode` — the parity oracle against `core.py` must not move.

THIS IS AN OVERRIDE, NOT THE ONLY WAY IN. The typed preset key \
`margin_row.backward_interm` is what can arm this on a SCORED run, because \
`vnncomp_scripts/run_instance.sh` exports exactly one `NY_*` variable and an \
env-only lever cannot fire in competition however well it measures. Layering is \
`read_over_config`'s, which is precisely the three-way rule the raw reader \
already implemented: exact `\"1\"` arms and exact `\"0\"` disarms, IN BOTH \
DIRECTIONS over the preset; a PRESENT near-miss token (`\"true\"`, `\" 1\"`, `\"\"`) \
is a recorded rejection that suppresses the preset and lands on this \
declaration's `false` default, so a typo is a kill switch rather than a silent \
promotion; absence falls through to the preset, then to `false`.

WHY High, given the intersection is shrink-only. Two valid enclosures intersect \
to a valid enclosure, a crossed intersection is treated as a defect signal \
rather than an infeasibility certificate (this is the ROOT box; acting on \
\"empty\" would be a false-UNSAT), and skipping any layer, chunk or neuron \
simply leaves the forward box standing. The risk is the documented ORDERING \
TRAP: `LayerGates::clip_rows` slack calibration must recover the LINE's own \
certified slack, and against a TIGHTENED published `l` that difference \
under-recovers (clamping to 0), which would make every Clip-and-Verify \
halfspace built from that line cut into the true subdomain — a false-`unsat` \
generator. `root.rs` calibrates against the forward-only bounds captured before \
this phase runs, and that ordering is the only thing standing between an armed \
run and an unsound cut. It also publishes different intermediate bounds and \
spends a budget slice, either of which can move a verdict.",
        provenance: Provenance::Unmeasured {
            why_ok: "dark by default in BOTH layers (the declaration default is false and \
                     the preset flag ships off); disarmed the root build is byte-identical \
                     to its history, and the campaign ledger records this build as BUILT, \
                     unmeasured",
        },
        owner: BACKWARD_INTERM,
        readers: &[ReaderSite {
            scope: BACKWARD_INTERM,
            role: "arm the backward-intermediate phase over the typed `margin_row.backward_interm` preset",
            site: "crates/ny-propagate/src/margin_row/backward_interm.rs:from_env",
        }],
    };

    /// `NY_MARGIN_ROW_BI_SECS` — wall-clock budget for that phase.
    pub MARGIN_ROW_BI_SECS = LeverDecl {
        name: "NY_MARGIN_ROW_BI_SECS",
        // `trim().parse::<f64>()` filtered on `is_finite() && >= 0.0`, with no
        // upper filter — a CLOSED lower end (zero is a real setting: budget
        // exhausted before the first chunk) and `f64::MAX` above.
        kind: LeverKind::F64ClosedTrimmed { min: 0.0, max: f64::MAX },
        default: DefaultSpec::F64(20.0),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Seconds the backward-intermediate phase may spend, default 20. Whitespace is \
trimmed before `parse::<f64>()`; non-finite, negative and malformed values all \
leave 20. ZERO IS ADMISSIBLE and means what it says — the budget is exhausted \
before the first chunk, so the phase does nothing — which is why the lower \
bound is closed rather than a `> 0` filter that would silently restore 20.

The value is a REQUEST, not an authority: it is further capped to 40% of the \
remaining deadline at construction (the same starvation guard `alpha_opt` \
uses), and the clock is checked before every chunk, so the tree search cannot \
be starved however large a number is supplied.

High because more budget means more chunks, more chunks means more trunk \
neurons get a TIGHTER published intermediate box, and the tightened box is what \
`gates_from_box` re-certifies the gates on. Stopping early is always sound, so \
the risk is not truncation — it is that this knob decides how much of the \
armed phase's bound-moving behaviour actually happens, and it takes wall clock \
from branch-and-bound to do it. Inert unless #backward-interm is armed.",
        provenance: Provenance::Unmeasured {
            why_ok: "inert while its parent phase is dark, which is the shipped state; \
                     the runbook lists a 40 s single-row sweep as the NEXT experiment, \
                     so by construction nothing is measured yet",
        },
        owner: BACKWARD_INTERM,
        readers: &[BI_TUNING_READER],
    };

    /// `NY_MARGIN_ROW_BI_CHUNK` — columns per prefix pass (memory grain).
    pub MARGIN_ROW_BI_CHUNK = LeverDecl {
        name: "NY_MARGIN_ROW_BI_CHUNK",
        kind: LeverKind::UsizeTrimmed,
        default: DefaultSpec::U64(256),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Columns per backward prefix pass, default 256 — the memory grain of the \
backward-intermediate phase, whose working set is `O(max_tensor * chunk)`. \
Whitespace is trimmed before `parse::<usize>()`; absent, malformed, negative \
and overflowing input all leave 256, and the reader then CLAMPS the resolved \
value to `1..=4096`, so `0` and 99999 are pulled to the nearest bound rather \
than rejected.

Chunking is a BATCHING decision, not a mathematical one: each chunk runs one \
Lower and one Upper prefix pass over its own disjoint columns and commits a \
shrink-only intersect, so the union of what a large chunk and many small chunks \
tighten is the same set of neurons. What it really trades is memory against \
per-chunk deadline granularity — a larger chunk checks the clock less often and \
so is likelier to be cut off mid-layer, leaving different neurons tightened. \
High for that reason: a different value produces a different set of published \
intermediate bounds on the authoritative lane. Inert unless #backward-interm \
is armed.",
        provenance: Provenance::Unmeasured {
            why_ok: "inert while its parent phase is dark, which is the shipped state; \
                     no retained A/B qualifies any other grain",
        },
        owner: BACKWARD_INTERM,
        readers: &[BI_TUNING_READER],
    };

    /// `NY_MARGIN_ROW_BI_TOPK` — unstable neurons re-derived per layer.
    pub MARGIN_ROW_BI_TOPK = LeverDecl {
        name: "NY_MARGIN_ROW_BI_TOPK",
        kind: LeverKind::UsizeTrimmed,
        default: DefaultSpec::U64(1024),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
How many unstable neurons the backward-intermediate phase re-derives per layer, \
widest first, default 1024. Whitespace is trimmed before `parse::<usize>()`; \
absent, malformed, negative and overflowing input all leave 1024, and the \
reader then applies `.max(1)`, so an explicit `0` becomes 1 rather than \
disabling the phase — disarming is `NY_MARGIN_ROW_BACKWARD_INTERM=0`, not a \
zero here.

This is the SELECTION knob: it decides WHICH pre-activations get a tightened, \
published box, and because the phase runs layer by layer and later layers \
consume earlier layers' tightened gates, the choice compounds down the trunk. \
High for exactly that reason — it changes the published intermediate bounds on \
the authoritative margin-row route (and spends the phase's budget doing it), \
even though every individual selection is sound and skipping any neuron leaves \
its forward box standing. Inert unless #backward-interm is armed.",
        provenance: Provenance::Unmeasured {
            why_ok: "inert while its parent phase is dark, which is the shipped state; \
                     no retained A/B qualifies any other width",
        },
        owner: BACKWARD_INTERM,
        readers: &[BI_TUNING_READER],
    };

    /// `NY_MARGIN_ROW_GPU_EFT` — the certified GPU backward for the lane.
    pub MARGIN_ROW_GPU_EFT = LeverDecl {
        name: "NY_MARGIN_ROW_GPU_EFT",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Arms the margin-row lane's certified GPU backward (`#margin-row-gpu-eft`): an \
admitted certified-outward pass would be computed by the EFT / double-single \
device kernels instead of the CPU backward walk, and concretized by the lane's \
own unchanged f64 concretize. Exact `\"1\"` arms; every other value — including \
`\"0\"`, `\"true\"` and non-Unicode bytes — leaves it dark, and the reason the \
declaration is a `Bool` rather than a presence gate is that `NY_..._EFT=0` must \
mean OFF here, as `Refusal::Disabled` states.

READ AS A LATCHED RAW STRING, deliberately. `env_raw` latches the STRING once \
in a `OnceLock` and `armed_from_raw` derives the DECISION per call — the same \
discipline as `gpu_seam::env_raw` — because `armed_from_raw` is a pure, \
unit-tested predicate that IS the spec of the arming rule and must stay in the \
production path rather than being restated by the chokepoint.

STAGED, NOT DEAD. The on-device M1 self-check and the device transaction are \
not delivered yet, so an armed run today gets past the pure admission \
predicates and stops at the AUTHORITY GATE with `Refusal::Unimplemented`; \
`run_transaction` requires a `VerdictAuthority` token, so a prematurely-wired \
call site cannot reach verdict math — a property of the types, not of comment \
discipline.

High is classified for what the gate ADMITS, exactly as `NY_EFT_ERR`'s staged \
CPU arm is: this lane is designed to be AUTHORITATIVE rather than a shadow \
(removing the CPU backward walk is the cost it exists to remove), so once the \
transaction lands, arming it decides whether a published margin-row bound came \
from the device. Its guards — the certified-error floor, the realization probe, \
the NaN/Inf firewall, the lane-local channel-death latch — all fail closed to \
the exact CPU pass, and none of them lowers what a wrong value here could cost.",
        provenance: Provenance::Unmeasured {
            why_ok: "dark by default and structurally unreachable past the authority gate \
                     in this delivery, so the OFF arm is the only arm that computes \
                     anything; it cannot leave Bucket::Debug without a measured A/B on \
                     the delivered transaction",
        },
        owner: GPU_EFT,
        readers: &[ReaderSite {
            scope: GPU_EFT,
            role: "latch the raw gate string for the pure `armed_from_raw` arming predicate",
            site: "crates/ny-propagate/src/margin_row/gpu_backward/mod.rs:env_raw",
        }],
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{read_over_config_with, read_with, LeverValue, Source};

    fn resolve(decl: &'static LeverDecl, raw: Option<&str>) -> (LeverValue, Source) {
        let owned = raw.map(str::to_owned);
        let resolved = read_with(decl, move |_| owned);
        (resolved.value, resolved.source)
    }

    /// Both exact-`"1"` gates keep the repo's arming rule byte for byte, and
    /// `"0"` stays an ADMISSIBLE disarm rather than a rejection — which is what
    /// `NY_MARGIN_ROW_GPU_EFT`'s `Refusal::Disabled` and the
    /// `#backward-interm` env-wins-both-directions rule both depend on.
    #[test]
    fn exact_one_gates_preserve_the_arming_rule() {
        for decl in [
            &MARGIN_ROW_BETA,
            &MARGIN_ROW_GPU_EFT,
            &MARGIN_ROW_BACKWARD_INTERM,
        ] {
            assert_eq!(
                resolve(decl, Some("1")),
                (LeverValue::Bool(true), Source::LegacyEnv),
                "{}",
                decl.name
            );
            assert_eq!(
                resolve(decl, Some("0")),
                (LeverValue::Bool(false), Source::LegacyEnv),
                "{}",
                decl.name
            );
            assert_eq!(
                resolve(decl, None),
                (LeverValue::Bool(false), Source::Default),
                "{}",
                decl.name
            );
            for reject in ["true", "TRUE", "yes", " 1", "01", "", "2"] {
                let (value, source) = resolve(decl, Some(reject));
                assert_eq!(value, LeverValue::Bool(false), "{} {reject:?}", decl.name);
                assert_eq!(
                    source,
                    Source::LegacyEnvRejected,
                    "{} {reject:?}",
                    decl.name
                );
            }
        }
    }

    /// The three-way `Some("1") / Some(_) / None` reader is exactly
    /// `read_over_config`'s layering, with the typed
    /// `margin_row.backward_interm` preset as the config layer.
    #[test]
    fn backward_interm_layers_env_over_preset_over_dark() {
        let layered = |raw: Option<&str>, preset: bool| {
            let owned = raw.map(str::to_owned);
            read_over_config_with(
                &MARGIN_ROW_BACKWARD_INTERM,
                move |_| owned,
                Some(LeverValue::Bool(preset)),
            )
            .expect("Bool config is admissible for a Bool declaration")
        };

        // Absent: the preset decides, in both directions.
        let from_preset = layered(None, true);
        assert!(from_preset.value.as_bool());
        assert_eq!(from_preset.source, Source::Config);
        assert!(!layered(None, false).value.as_bool());

        // Present and admissible: env wins over the preset, both ways.
        assert!(layered(Some("1"), false).value.as_bool());
        let killed = layered(Some("0"), true);
        assert!(
            !killed.value.as_bool(),
            "an explicit 0 must disarm the preset"
        );
        assert_eq!(killed.source, Source::LegacyEnv);

        // Present and inadmissible: `Some(_) => return None` in the legacy
        // reader. A near-miss token suppresses the preset instead of riding it.
        for reject in ["true", " 1", "01", "", "2"] {
            let r = layered(Some(reject), true);
            assert!(!r.value.as_bool(), "{reject:?} must not arm the phase");
            assert_eq!(r.source, Source::LegacyEnvRejected, "{reject:?}");
        }
    }

    /// The trimming integer parsers, and the reader-side clamps that are
    /// deliberately NOT folded into the declarations.
    #[test]
    fn trimmed_integer_knobs_keep_their_defaults_and_leave_clamping_to_readers() {
        for (decl, default) in [
            (&MARGIN_ROW_BETA_ITERS, 1),
            (&MARGIN_ROW_BI_CHUNK, 256),
            (&MARGIN_ROW_BI_TOPK, 1024),
        ] {
            assert_eq!(
                resolve(decl, None),
                (LeverValue::U64(default), Source::Default),
                "{}",
                decl.name
            );
            assert_eq!(
                resolve(decl, Some(" 12 ")),
                (LeverValue::U64(12), Source::LegacyEnv),
                "{}",
                decl.name
            );
            assert_eq!(
                resolve(decl, Some("bad")).1,
                Source::LegacyEnvRejected,
                "{}",
                decl.name
            );
            // An explicit 0 RESOLVES; `clamp(1, ..)` / `.max(1)` at the reader
            // is what turns it into 1, and "explicitly zero" must stay
            // distinguishable from "absent" at the chokepoint.
            assert_eq!(
                resolve(decl, Some("0")),
                (LeverValue::U64(0), Source::LegacyEnv),
                "{}",
                decl.name
            );
        }
    }

    /// `trim().parse::<f64>()` with `is_finite()` and a CLOSED lower bound.
    #[test]
    fn trimmed_float_knobs_preserve_their_filters() {
        assert_eq!(
            resolve(&MARGIN_ROW_BI_SECS, None),
            (LeverValue::F64(20.0), Source::Default)
        );
        assert_eq!(
            resolve(&MARGIN_ROW_BI_SECS, Some(" 40 ")),
            (LeverValue::F64(40.0), Source::LegacyEnv)
        );
        // Zero is a real budget, not a rejection: the phase simply runs no chunk.
        assert_eq!(
            resolve(&MARGIN_ROW_BI_SECS, Some("0")),
            (LeverValue::F64(0.0), Source::LegacyEnv)
        );
        for reject in ["-1", "nan", "inf", "lots"] {
            let (value, source) = resolve(&MARGIN_ROW_BI_SECS, Some(reject));
            assert_eq!(value, LeverValue::F64(20.0), "{reject:?}");
            assert_eq!(source, Source::LegacyEnvRejected, "{reject:?}");
        }

        // eta keeps its `> 0.0` half AT THE READER, so `0` resolves here.
        assert_eq!(
            resolve(&MARGIN_ROW_BETA_ETA, None),
            (LeverValue::F64(0.5), Source::Default)
        );
        assert_eq!(
            resolve(&MARGIN_ROW_BETA_ETA, Some(" 2.0 ")),
            (LeverValue::F64(2.0), Source::LegacyEnv)
        );
        assert_eq!(
            resolve(&MARGIN_ROW_BETA_ETA, Some("0")),
            (LeverValue::F64(0.0), Source::LegacyEnv)
        );
        assert_eq!(
            resolve(&MARGIN_ROW_BETA_ETA, Some("-0.5")).0,
            LeverValue::F64(0.5)
        );
    }

    /// Nothing in this group may ship armed: all three programs are recorded as
    /// BUILT, unmeasured, and every one of them can move a published bound.
    #[test]
    fn the_whole_group_is_dark_high_risk_and_unmeasured() {
        for decl in MARGIN_ROW_LEVERS.decls() {
            assert_eq!(decl.bucket, Bucket::Debug, "{}", decl.name);
            assert_eq!(decl.moat, MoatRisk::High, "{}", decl.name);
            assert!(
                matches!(decl.provenance, Provenance::Unmeasured { .. }),
                "{}",
                decl.name
            );
            // The two OPT-OUTS are the documented exception: they ship
            // `Bool(true)` because their reader tests `!= Some("0")`, and they
            // are sub-switches of `NY_MARGIN_ROW_BETA`, which ships dark — so
            // an armed default here still arms nothing on a scored run.
            let is_beta_opt_out = matches!(
                decl.name,
                "NY_MARGIN_ROW_BETA_POLYAK" | "NY_MARGIN_ROW_BETA_HEADS"
            );
            assert!(
                is_beta_opt_out || !matches!(decl.default, DefaultSpec::Bool(true)),
                "{}: no gate in this group ships armed",
                decl.name
            );
        }
    }
}
