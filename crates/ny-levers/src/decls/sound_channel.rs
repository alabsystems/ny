// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The S2 / EFT certified-error channel.

use crate::{
    declare_levers, Bucket, DefaultSpec, LeverDecl, LeverKind, MoatRisk, Provenance, ReaderSite,
    Scope,
};

const PROPAGATE: Scope = Scope {
    package: "ny-propagate",
    subsystem: "eft-err",
};

const GPU: Scope = Scope {
    package: "ny-gpu",
    subsystem: "eft-err",
};

/// The four governed production reads of `NY_EFT_ERR`, spanning two packages.
///
/// This slice is the whole point of the declaration: the census's first pass
/// classified the `ny-propagate` read as an accidental cross-crate collision
/// and proposed deleting it. Reading the comment at the read site says the
/// opposite. Written down here, the sharing is a design; discovered by a
/// grep, it looks like a bug.
const EFT_ERR_READERS: &[ReaderSite] = &[
    ReaderSite {
        scope: PROPAGATE,
        role: "CPU S2 arm: gate `eft_err_channel_enabled`, reached via \
               `aw_f32_sound_bound`, which is itself unwired BY DESIGN (staged \
               work, `#[allow(dead_code)]`)",
        site: "crates/ny-propagate/src/layers/linear/crown_single.rs:1179",
    },
    ReaderSite {
        scope: GPU,
        role: "GPU twin: gate `eft_err_env_enabled`, read once per FOLD (not \
               per layer) so differential A/B tests can flip it",
        site: "crates/ny-gpu/src/wgpu_device/ops/crown_backward_sound_resident.rs:115",
    },
    ReaderSite {
        scope: GPU,
        role: "GPU concretization twin: select the EFT-certified concretization \
               channel only when the primitive self-check cache is armed",
        site: "crates/ny-gpu/src/wgpu_device/ops/crown_concretize_sound.rs:493",
    },
    ReaderSite {
        scope: GPU,
        role: "eager capability probe at device construction: runs \
               `verify_eft_primitives` outside the GPU lock so every in-fold \
               read hits the cache and fails closed to the Higham channel",
        site: "crates/ny-gpu/src/wgpu_device/device.rs:209",
    },
];

declare_levers! {
    registry SOUND_CHANNEL_LEVERS;

    /// `NY_EFT_ERR` — the S2 / EFT-compensated certified-error channel.
    pub EFT_ERR = LeverDecl {
        name: "NY_EFT_ERR",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Arms the EFT-compensated certified-error channel (S2): the a-posteriori \
enclosure of A·W widened by the EFT-MEASURED rounding error of the actual f32 \
fold, instead of the a-priori gamma_n^f32 * S worst case. DARK by default; \
with the flag absent the EFT passes are never dispatched and the result is \
byte-identical.

DELIBERATE CROSS-CRATE TWIN GATE — NOT a collision. \
`crates/ny-propagate/src/layers/linear/crown_single.rs:1172-1176` states the \
naming outright: the CPU gate is \"named/valued identically to the GPU twin's \
gate ... so one switch arms the whole S2 channel and scoped-env tests can flip \
it\". ONE lever, four production reads, two packages: \
ny-propagate (`layers/linear/crown_single.rs:1179`, the CPU arm, which is \
`#[allow(dead_code)]` and unwired BY DESIGN as staged work) and \
ny-gpu (`wgpu_device/ops/crown_backward_sound_resident.rs:115`, the in-fold \
gate; `wgpu_device/ops/crown_concretize_sound.rs:493`, the concretization \
twin; plus `wgpu_device/device.rs:209`, the eager `verify_eft_primitives` \
capability probe). The census's DEAD verdict on the CPU side must NOT be \
executed as a deletion: it would destroy staged work.

Both gates are deliberately UNCACHED (no OnceLock) so scoped-env differential \
tests can flip them mid-process; each is read once per GEMM / per fold, never \
per element.

MoatRisk::High because the flag bundles two claims beyond residual \
measurement, both now discharged by dedicated oracles (sound_authority.rs \
ledger, 2026-08-10): U5, the a-priori Lipschitz propagation swap, validated in \
`u5_activation_lipschitz::{act_eft_err_encloses_worst_realization, \
act_intercept_bias_eft_err_encloses_worst_realization}`; and U6, that \
concretize is not value-neutral in EFT mode (by design — both modes enclose \
and refusal is bit-identical), validated in \
`crown_concretize_sound::tests::\
u6_concretize_eft_vs_legacy_enclose_and_fail_closed_identity`.

Exact-\"1\" arming matters more here than anywhere else on the surface: a \
generously-parsed token that armed one twin and not the other would produce a \
half-enabled verdict-carrying channel that no one designed.",
        provenance: Provenance::Unmeasured {
            why_ok: "Dark by default and staged: with the gate absent the EFT \
                     passes are never dispatched, so the OFF arm is \
                     byte-identical and the CPU arm is not even reached. It is \
                     Bucket::Debug precisely because it has no discriminating \
                     measurement on the current sound path; it cannot become \
                     DefaultOn without one.",
        },
        owner: PROPAGATE,
        readers: EFT_ERR_READERS,
    };

    /// `NY_MARGIN_ROW_GPU` — the margin-row lane's AUTHORITATIVE GPU seam.
    pub MARGIN_ROW_GPU = LeverDecl {
        name: "NY_MARGIN_ROW_GPU",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Arms the margin-row twin-wall lane's GPU seam: an admitted certified-outward \
backward pass is computed by the sound GPU-resident CROWN walk \
(`crown_backward_gpu_seeded_sound_coeffs` for a flat chain, or the ResNet \
coefficient entry for a residual plan) INSTEAD OF the CPU walk, and its returned \
coefficients + certified error are concretized by the lane's own unchanged f64 \
`concretize_*`.

THIS IS AUTHORITATIVE, NOT A SHADOW. That is deliberate — the CPU backward \
walk is the cost the seam exists to remove — and it is why the risk is High. \
The published bound is sound because (1) every relaxation handed to the device \
encloses the real op, with ReLU upper lines REPAIRED after the f32 downcast \
using the convexity endpoint test; (2) the BN-fold weight/bias error and the \
f64->f32 downcast are composed into the layer's `CertifiedWeightError`, which \
the kernels charge; (3) the seed's own certified error and downcast residual \
are concretized against the caller's y-box into the bias-error lane instead of \
being dropped; (4) coefficient egress ignores the signature-parity input box \
and abs-max tables, preserving coefficient radii rather than discharging them \
into bias; and (5) the lane, not the device, concretizes over its f64 root box.

Argument is not evidence, so every admitted pass is additionally checked by a \
certified-error FLOOR (the penalty must dominate `rho*T/(1+rho)` for the \
weight-error ball the device was told to charge) and by a REALIZATION PROBE (a \
single exact f64 forward pass at the box midpoint; a published lower bound \
above a realized value of the same functional is a proof of unsoundness). A \
trip fails closed to the exact CPU pass and increments \
`margin_row::prof::Counter::GpuSeamGuardTrip`, which must read zero.

Every refusal — no prewarmed sound backend, no coefficient egress, a deadline \
the backend will not honor, an unmappable op (`ChannelAffine`, \
`ConvTranspose`, asymmetric padding, an unsupported residual shape such as a \
nested/degenerate `Add`, per-row gate exceptions), a malformed payload, either \
guard — runs the untouched CPU pass. \
Exact-\"1\" arming; anything else leaves the lane byte-for-byte as it is today.",
        provenance: Provenance::Unmeasured {
            why_ok: "dark by default: with the gate absent the seam refuses \
                     before it builds a plan or observes a device, and every \
                     seamed entry point is then the established CPU call \
                     bit-for-bit. The ON arm is verdict-carrying and must not \
                     be promoted past Bucket::Debug without a sealed \
                     device-backed enclosure sweep.",
        },
        owner: Scope {
            package: "ny-propagate",
            subsystem: "margin-row",
        },
        readers: &[ReaderSite {
            scope: Scope {
                package: "ny-propagate",
                subsystem: "margin-row",
            },
            role: "gate the whole seam at `gpu_seam::enabled`; the raw string \
                   is latched once in a process-wide `OnceLock` because the \
                   predicate is consulted per backward pass",
            site: "crates/ny-propagate/src/margin_row/gpu_seam.rs",
        }],
    };

    /// `NY_MARGIN_ROW_GPU_BATCH` — DOMAIN BATCHING for the margin-row GPU seam.
    pub MARGIN_ROW_GPU_BATCH = LeverDecl {
        name: "NY_MARGIN_ROW_GPU_BATCH",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Arms DOMAIN BATCHING on top of `NY_MARGIN_ROW_GPU` (which must ALSO be armed — \
this gate never enables the seam on its own). Instead of one certified GPU \
backward per popped BaB domain, the lane stacks every LRU-missing domain of a \
frontier batch into ONE wide certified call \
(`crown_backward_gpu_resnet_sound_batched_coeffs`) and splits the returned \
per-domain coefficient frontiers back out.

WHY IT EXISTS: the margin-row lane is the lane that actually proves the \
deciding cifar100/tinyimagenet rows, and it processes domains ONE AT A TIME. A \
per-pass seam buys a few x on a single domain's backward; the measured gap on \
that pool is ~32x (docs/VNNCOMP_CRITICAL_PATH_2026-08-12.md), which only a \
domain-batched wide pass can reach.

WHAT VARIES PER DOMAIN — and therefore what the batch must keep separate: ONLY \
the per-ReLU relaxation triple (alpha, s, c) written by the domain's piece \
fixes. The weights, the head Gemm, the certified BN-fold error charges, the \
identity seed, the host-only per-ReLU `node_abs` retargeting invariant and the \
signature-parity ROOT input box are all domain-INDEPENDENT in this lane (a \
piece fix changes gates, never `l`/`u`, never the box). The coefficient egress \
receives empty abs-max tables and ignores the box fields. The shared skeleton is \
therefore literally the SAME `Arc` allocations for every domain, and the \
device's homogeneity gate passes by pointer identity rather than by an \
O(weights) value compare.

MOAT: a SLOT/PERMUTATION error — publishing for domain A the bound computed \
for domain B's gates — is the killer defect of this lane, so the mapping is \
domain-major, contiguous, length-checked before any slicing, pinned on the \
device side (`batched_split_is_domain_major_and_contiguous`) and on the lane \
side (`batched_slot_permutation_is_detectable`). Every domain keeps its OWN \
certified-error floor guard and its OWN realization probe, evaluated against \
its OWN gates. Any refusal — gate off, chain plan, heterogeneous skeleton, \
unbatchable layer kind, device dispatch limit, deadline, authority, shape, \
either guard — falls back to the per-pass seam and then to the exact CPU walk. \
Exact-\"1\" arming; anything else leaves the lane byte-for-byte as it is today.",
        provenance: Provenance::Unmeasured {
            why_ok: "dark by default AND subordinate: with either this gate or \
                     `NY_MARGIN_ROW_GPU` absent the batch prefill returns \
                     before it builds a plan or observes a device, so the \
                     frontier's y-packs are rebuilt by the established \
                     one-at-a-time path bit-for-bit. The ON arm is \
                     verdict-carrying and must not be promoted past \
                     Bucket::Debug without a sealed device-backed enclosure \
                     sweep of its own.",
        },
        owner: Scope {
            package: "ny-propagate",
            subsystem: "margin-row",
        },
        readers: &[ReaderSite {
            scope: Scope {
                package: "ny-propagate",
                subsystem: "margin-row",
            },
            role: "gate the batched prefill at `gpu_seam::batch::enabled`; the \
                   raw string is latched once in a process-wide `OnceLock` \
                   because the predicate is consulted per frontier batch",
            site: "crates/ny-propagate/src/margin_row/gpu_seam/batch.rs",
        }],
    };

    /// `NY_MARGIN_ROW_CLIP_ROWS` — retain the clip halfspace rows on the
    /// margin-row root gates.
    pub MARGIN_ROW_CLIP_ROWS = LeverDecl {
        name: "NY_MARGIN_ROW_CLIP_ROWS",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::Low,
        doc: "\
Retains the clip halfspace rows on `LayerGates::clip_rows` instead of dropping \
them. Exact `1` arms it; absence and every other value leave `clip_rows` at \
`None`. RECLASSIFIED 2026-08-22 from MoatRisk::None: the declaration used to say \
\"nothing reads these rows yet\", and that is no longer true. Two consumers now \
exist — `clip::halfspaces_for_splits` and `clip::tighten_intermediate` — and both \
BAIL OUT (`clip_rows.as_ref() else continue/return`) when retention is off, which \
makes this the third gate whose absence silently voids any Clip-and-Verify \
measurement: with it unset the whole mechanism refuses while still reading as \
armed. Retention itself still changes no bound; what changed is that the rows are \
now load-bearing for a path that does. \
\
The #closure-adaptive policy deliberately does NOT arm this: retention happens \
ONCE at root build, so it would have to be bought up front on every row, and a \
24-run matrix measured its consumers BIT-IDENTICAL to baseline (mean_rel_shrink \
8.4e-16 to 1.6e-4) -- which does not pay for it. \
\
SOUNDNESS, and it holds identically at BOTH consumers: every halfspace is a \
necessary condition of a split the domain has already fixed, so the constrained \
set over-approximates the true subdomain; `halfspaces_for_splits` pays the \
certified `sl_lo`/`sl_up` slack the root already paid, and \
`tighten_intermediate` pays the same slack and commits `max`/`min` against the \
stored interval, so it can only SHRINK. Both are therefore proof-losing, not \
proof-manufacturing — CONDITIONAL on the retained rows being in the right frame, \
which is not an argument but a checked precondition: `clip::frame_ok` \
re-concretizes every retained row over the root box once per process and \
disables the whole path on any disagreement.",
        provenance: Provenance::Unmeasured {
            why_ok: "default-off, and the armed arm's two consumers are themselves default-off \
                     (NY_MARGIN_ROW_CLIP and NY_MARGIN_ROW_CLIP_INTERM), \
                     so the shipped state retains nothing and consumes nothing; the armed arm \
                     is bound-affecting and needs a moat run before any preset arms it",
        },
        owner: PROPAGATE,
        readers: &[
            ReaderSite {
                scope: PROPAGATE,
                role: "WRITER: retain the input-relative affine rows at root build; latched \
                       once per process",
                site: "crates/ny-propagate/src/margin_row/root.rs:clip_rows_enabled",
            },
            ReaderSite {
                scope: PROPAGATE,
                role: "CONSUMER: build a split's halfspace from the retained row; refuses the \
                       split entirely when the layer retained nothing",
                site: "crates/ny-propagate/src/margin_row/clip.rs:halfspaces_for_splits",
            },
            ReaderSite {
                scope: PROPAGATE,
                role: "CONSUMER: re-concretize the top-k unstable neurons over \
                       `box INTERSECT halfspaces` and shrink their intervals; returns empty \
                       when the layer retained nothing, which is what makes an unretained \
                       clip measurement vacuous rather than negative",
                site: "crates/ny-propagate/src/margin_row/clip.rs:tighten_intermediate",
            },
        ],
    };

    /// `NY_MARGIN_ROW_ALPHA_OPT` — the margin-row lane's own alpha optimization.
    pub MARGIN_ROW_ALPHA_OPT = LeverDecl {
        name: "NY_MARGIN_ROW_ALPHA_OPT",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Arms the margin-row lane's own alpha ascent. Exact `1` arms it; absence and every \
other value leave it dark. It changes which relaxation the lane publishes a bound \
from, so it is verdict-affecting on the authoritative margin-row route.",
        provenance: Provenance::Measured {
            commit: "98acbe08cde8417fc8c1b1547c8d1c3595f1c0aa",
            date: "2026-08-25",
            artifact: "docs/ALPHA_OPT_MARGIN_ROW_ROOT_2026-08-18.md and the \
                       retained cifar100 preset rationale",
            delta: "on the current sound path, cifar100 idx_9502 changed from \
                    timeout at root -0.197208 with alpha disabled to UNSAT 3/3 \
                    at root -0.174755 with the shipped optimizer; the current \
                    moat was UNSAT 9/9",
        },
        owner: PROPAGATE,
        readers: &[
            ReaderSite {
                scope: PROPAGATE,
                role: "freeze the complete root-alpha/rebake policy from the scored LeverSet",
                site: "crates/ny-propagate/src/margin_row/mod.rs:RootAlphaRuntimePolicy::from_levers",
            },
            ReaderSite {
                scope: PROPAGATE,
                role: "resolve the legacy layered process only for compatibility callers",
                site: "crates/ny-propagate/src/margin_row/mod.rs:RootAlphaRuntimePolicy::from_layered_process",
            },
        ],
    };

    /// `NY_MARGIN_ROW_ALPHA_ITERS` — iteration count for that ascent.
    pub MARGIN_ROW_ALPHA_ITERS = LeverDecl {
        name: "NY_MARGIN_ROW_ALPHA_ITERS",
        kind: LeverKind::UsizeTrimmed,
        default: DefaultSpec::U64(20),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Iterations for the margin-row alpha ascent, default 20 — alpha-beta-CROWN's own \
root-alpha iteration count. Whitespace is trimmed; absent or malformed leaves 20, \
and the reader CLAMPS the resolved value to 1..=64, so values outside that range \
are pulled to the nearest bound rather than rejected. The typed preset key is \
`margin_row.alpha_iters` and the environment wins over it in both directions. \
Inert unless NY_MARGIN_ROW_ALPHA_OPT is armed.

WHY 20 AND NOT THE ORIGINAL 8. Measured 2026-08-24 on the six arm-C rows of \
`docs/ABC_ROOT_GAP_ON_WINNABLE_ROWS_2026-08-24.md`: the ascent accepted 8 of 8 \
iterations on EVERY row, `eta` never halved once, and the per-iteration gain \
decayed by only ~4% per step. It did not converge and it did not overshoot — it \
ran out of iterations while still climbing. The wall cap (NY_MARGIN_ROW_ALPHA_SECS, \
further capped to 40% of the remaining deadline) is what actually binds at the \
scored budget, so this number is an upper limit rather than a promise; \
`[alpha-opt] done iters=` reports how many were really spent.",
        provenance: Provenance::Measured {
            commit: "98acbe08cde8417fc8c1b1547c8d1c3595f1c0aa",
            date: "2026-08-25",
            artifact: "docs/ALPHA_OPT_MARGIN_ROW_ROOT_2026-08-18.md",
            delta: "the shipped 20-step cifar100 ascent saturated by iterations \
                    11-12 while tinyimagenet reached only 9/20 at its wall cap; \
                    this separates iteration saturation from budget starvation",
        },
        owner: PROPAGATE,
        readers: &[
            ReaderSite {
                scope: PROPAGATE,
                role: "freeze the standalone alpha iteration cap",
                site: "crates/ny-propagate/src/margin_row/mod.rs:RootAlphaRuntimePolicy::from_levers",
            },
            ReaderSite {
                scope: PROPAGATE,
                role: "resolve the compatibility process policy",
                site: "crates/ny-propagate/src/margin_row/mod.rs:RootAlphaRuntimePolicy::from_layered_process",
            },
        ],
    };

    /// `NY_MARGIN_ROW_ALPHA_SECS` — wall-clock slice for that ascent.
    pub MARGIN_ROW_ALPHA_SECS = LeverDecl {
        name: "NY_MARGIN_ROW_ALPHA_SECS",
        kind: LeverKind::Secs,
        default: DefaultSpec::Secs(20.0),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Seconds the margin-row alpha ascent may spend. Absent, malformed, non-finite and \
non-positive values all leave 20.0. Inert unless NY_MARGIN_ROW_ALPHA_OPT is armed; \
when armed it trades budget away from the rest of the lane. Surrounding whitespace \
is accepted, matching the original `raw.trim().parse::<f64>()` reader.",
        provenance: Provenance::Measured {
            commit: "98acbe08cde8417fc8c1b1547c8d1c3595f1c0aa",
            date: "2026-08-25",
            artifact: "docs/ALPHA_OPT_MARGIN_ROW_ROOT_2026-08-18.md",
            delta: "the 20-second request left cifar100 converged but truncated \
                    tinyimagenet at 9/20 iterations; tinyimagenet stayed dark \
                    because that pre-tree wall cost displaced needed tree work",
        },
        owner: PROPAGATE,
        readers: &[
            ReaderSite {
                scope: PROPAGATE,
                role: "freeze the standalone alpha wall request",
                site: "crates/ny-propagate/src/margin_row/mod.rs:RootAlphaRuntimePolicy::from_levers",
            },
            ReaderSite {
                scope: PROPAGATE,
                role: "resolve the compatibility process policy",
                site: "crates/ny-propagate/src/margin_row/mod.rs:RootAlphaRuntimePolicy::from_layered_process",
            },
        ],
    };

    /// `NY_MARGIN_ROW_ALPHA_LR` — Adam step for that ascent (#alpha-opt-adam).
    pub MARGIN_ROW_ALPHA_LR = LeverDecl {
        name: "NY_MARGIN_ROW_ALPHA_LR",
        // Trimmed, finite, CLOSED at zero: zero is MEANINGFUL here — it selects
        // the legacy normalized-supergradient step — so the reader keeps a
        // `>= 0.0` filter rather than the `> 0.0` one the ETA levers use.
        kind: LeverKind::F64ClosedTrimmed { min: 0.0, max: f64::MAX },
        default: DefaultSpec::F64(0.1),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Adam step size for the margin-row ROOT alpha ascent, default 0.1. This is NY's \
retained rate, not alpha-beta-CROWN's cifar100 `lr_alpha` (0.25). The raw \
supergradient is first divided by its \
own max, so this is denominated in the same units as the legacy `eta`: slope \
travel of the DOMINANT neuron through the legal `[0, 1]` range in one iteration. \
The difference is that Adam is scale-free PER COORDINATE, so every neuron with \
gradient support moves ~lr, where the legacy rule moved a neuron at a tenth of \
the peak gradient a tenth as far. Decays by 0.98 per iteration (abc's \
`lr_decay`). Exactly `0` restores the legacy global-max-normalized step, which \
is the control arm. Trimmed; absent, malformed, negative, `nan` and `inf` all \
leave 0.1; the reader clamps to `[0, 1]`. The typed preset key is \
`margin_row.alpha_lr` and the environment wins over it in both directions. It is \
consumed by either the standalone root-alpha optimizer or the joint re-bake \
optimizer; it is inert only when both parent mechanisms are dark.

WHY IT EXISTS. Measured 2026-08-24, the legacy step moved only 76..278 of \
thousands of unstable neurons and recovered +0.0061 of root bound against abc's \
+0.1218 — 20x too weak — while accepting 8 of 8 iterations with `eta` never \
halving, i.e. failing by taking steps far too small rather than by pointing the \
wrong way. High for its parents' reasons: armed it changes the certified bound \
the authoritative margin-row lane publishes at the root, and the root is what \
every BaB domain inherits.",
        provenance: Provenance::Measured {
            commit: "98acbe08cde8417fc8c1b1547c8d1c3595f1c0aa",
            date: "2026-08-25",
            artifact: "docs/ALPHA_OPT_MARGIN_ROW_ROOT_2026-08-18.md",
            delta: "the coupled shipped vector (Adam LR=0.1 and K=8 steering) \
                    produced median +0.0328 published root gain versus +0.0061 \
                    for the legacy vector, and converted cifar100 idx_9502; \
                    this retained comparison does not isolate LR from K",
        },
        owner: PROPAGATE,
        readers: &[
            ReaderSite {
                scope: PROPAGATE,
                role: "freeze the shared standalone/rebake step size",
                site: "crates/ny-propagate/src/margin_row/mod.rs:RootAlphaRuntimePolicy::from_levers",
            },
            ReaderSite {
                scope: PROPAGATE,
                role: "resolve the compatibility process policy",
                site: "crates/ny-propagate/src/margin_row/mod.rs:RootAlphaRuntimePolicy::from_layered_process",
            },
        ],
    };

    /// `NY_MARGIN_ROW_ALPHA_COLS` — steering width for that ascent.
    pub MARGIN_ROW_ALPHA_COLS = LeverDecl {
        name: "NY_MARGIN_ROW_ALPHA_COLS",
        kind: LeverKind::UsizeTrimmed,
        default: DefaultSpec::U64(8),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
How many of the WORST class columns the margin-row root alpha ascent averages \
its steering gradient over, default 8. Trimmed; absent or malformed leaves 8, \
and the reader clamps to 1..=64. `1` reproduces the legacy single-worst-column \
steering exactly. The typed preset key is `margin_row.alpha_cols` and the \
environment wins over it in both directions. It is consumed by either the \
standalone root-alpha optimizer or the joint re-bake optimizer; it is inert only \
when both parent mechanisms are dark.

WHY IT EXISTS. The frozen-frame objective is a MIN over the class columns that \
survive the head frame (measured: 3 to 37 of them on the arm-C rows), so its \
exact supergradient is the gradient at the argmin column alone — and a neuron \
whose incoming coefficient is negative at THAT column has `vp = 0`, no gradient, \
and can never move however much it would buy the others. Averaging the K worst \
columns is the standard CVaR surrogate for a max-min. Tightness-only: the true \
min is still what scores every iterate, so a bad K wastes passes and cannot \
manufacture a proof. Each extra column costs one more single-column forward walk \
(milliseconds) per iteration, not another certified backward pass.",
        provenance: Provenance::Measured {
            commit: "98acbe08cde8417fc8c1b1547c8d1c3595f1c0aa",
            date: "2026-08-25",
            artifact: "docs/ALPHA_OPT_MARGIN_ROW_ROOT_2026-08-18.md",
            delta: "the coupled shipped vector used K=8 worst-column steering, \
                    produced median +0.0328 published root gain, converted \
                    cifar100 idx_9502, and preserved the 9/9 current moat; the \
                    retained comparison does not isolate K from the Adam rate",
        },
        owner: PROPAGATE,
        readers: &[
            ReaderSite {
                scope: PROPAGATE,
                role: "freeze the shared standalone/rebake steering width",
                site: "crates/ny-propagate/src/margin_row/mod.rs:RootAlphaRuntimePolicy::from_levers",
            },
            ReaderSite {
                scope: PROPAGATE,
                role: "resolve the compatibility process policy",
                site: "crates/ny-propagate/src/margin_row/mod.rs:RootAlphaRuntimePolicy::from_layered_process",
            },
        ],
    };

    /// `NY_MARGIN_ROW_ALPHA_GUARD` — published-frame revert guard.
    pub MARGIN_ROW_ALPHA_GUARD = LeverDecl {
        name: "NY_MARGIN_ROW_ALPHA_GUARD",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(true),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Score the margin-row root alpha ascent's commit through the UNCHANGED \
`root_eval` and revert to the heuristic gates unless the PUBLISHED bound \
strictly improves. Default ON. Exact `\"1\"` arms and exact `\"0\"` disarms in both \
directions over the typed preset key `margin_row.alpha_guard`.

WHY IT EXISTS, AND WHY IT DEFAULTS ON. The ascent is monotone on a FROZEN frame \
— y-pack, head gates, m1 and m2v are built once under the heuristic gates and \
held fixed while trunk alpha moves — but the lane publishes a different number, \
which `root_eval` recomputes under whatever gates the root ends up carrying. \
Soundness of the VALUES is not at issue (`alpha*y <= relu(y)` holds pointwise \
for any `alpha in [0, 1]`, and `ms = max(alpha, s)` is re-derived on commit), so \
this guard is not what makes the bound valid. It is what makes the bound \
MONOTONE: with it, what the lane publishes is `max(binary, optimized)` chosen \
between two separately certified lower bounds on the same margin — the \
intersection of the two certificates — so arming the ascent cannot lose a banked \
proof to a frozen-frame gain that does not transfer. Current-path measurements \
retained in `docs/ALPHA_OPT_MARGIN_ROW_ROOT_2026-08-18.md` show median transfer \
of 103.5% across nine rows, `idx_1298` at 69%, and zero guard reversions. That \
observation does not make the check redundant: an unobserved negative transfer \
would discard a banked bound. Costs two extra `root_eval` evaluations. Inert \
unless NY_MARGIN_ROW_ALPHA_OPT is armed.",
        provenance: Provenance::Guard {
            protects: "the already-certified published root bound from a frozen-frame alpha \
                       improvement that fails to transfer through root_eval",
        },
        owner: PROPAGATE,
        readers: &[
            ReaderSite {
                scope: PROPAGATE,
                role: "freeze the standalone published-frame guard policy",
                site: "crates/ny-propagate/src/margin_row/mod.rs:RootAlphaRuntimePolicy::from_levers",
            },
            ReaderSite {
                scope: PROPAGATE,
                role: "resolve the compatibility process policy",
                site: "crates/ny-propagate/src/margin_row/mod.rs:RootAlphaRuntimePolicy::from_layered_process",
            },
        ],
    };

    /// `NY_MARGIN_ROW_SPEC_ALPHA` — the per-START-NODE (spec) alpha axis.
    pub MARGIN_ROW_SPEC_ALPHA = LeverDecl {
        name: "NY_MARGIN_ROW_SPEC_ALPHA",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
#spec-alpha. After the margin-row root DIRECT pass has produced `dj[k]` per \
adversarial class, run additional certified backward passes whose COLUMNS are \
`(hardest class k, candidate alpha e)` pairs and fold the result into `dj[k]` by \
ELEMENT-WISE MAX. Default OFF; the typed preset key is \
`margin_row.spec_alpha` and the environment wins over it in both directions.

WHY IT EXISTS. alpha-beta-CROWN optimises `alpha[relu][start_node]` of shape \
`[2, spec, batch, *relu_shape]` — a distinct lower-relaxation slope per OUTPUT \
NEURON of every start node — and gains a median +0.163 at the root on the rows \
ny fails. ny holds ONE shared `(layer, neuron)` alpha for every class, and its \
stand-in for the axis (`#interm-alpha`) is a fixed 5-candidate ladder measured \
at +0.0025. This key adds exactly the axis abc KEEPS when it discards the rest: \
the FINAL start node's spec axis, i.e. the adversarial class. (abc's \
`share_alphas` collapses the intermediate start nodes' spec axis to 1 but \
exempts the final node; `alpha_drop_unused` discards the intermediates entirely \
before BaB.)

WHY IT CANNOT COST A PROOF. Nothing is mutated — not `alpha`, `ms`, `s`, `c`, \
`l`, `u`, `clip_rows`, the y-pack, the head gates or the tree's domain gates. \
The candidates ride the engine's existing per-`(row, neuron)` `Exceptions` path, \
which re-derives `ms = max(alpha, s)` per entry in shipped code, so this adds \
ZERO lines of certified arithmetic. The published `dj[k]` is an element-wise max \
of independently certified lower bounds, so `dj_armed[k] >= dj_dark[k]` for \
every k and every candidate; a garbage candidate costs wall and nothing else. \
Its cost is the wall the extra passes spend, which is why the width, candidate \
and round counts are separate keys.",
        provenance: Provenance::Unmeasured {
            why_ok: "the per-START-NODE alpha axis has never been built on this lane, so no \
                     retained A/B qualifies a value; the declaration default is dark and the \
                     dark path is one Option discriminant, byte-identical to the shipped root",
        },
        owner: PROPAGATE,
        readers: &[
            ReaderSite {
                scope: PROPAGATE,
                role: "freeze the scored per-start-node alpha policy",
                site: "crates/ny-propagate/src/margin_row/mod.rs:RootAlphaRuntimePolicy::from_levers",
            },
            ReaderSite {
                scope: PROPAGATE,
                role: "resolve the compatibility process policy",
                site: "crates/ny-propagate/src/margin_row/mod.rs:RootAlphaRuntimePolicy::from_layered_process",
            },
        ],
    };

    /// `NY_MARGIN_ROW_SPEC_ALPHA_WIDTH` — hardest failing classes given candidates.
    pub MARGIN_ROW_SPEC_ALPHA_WIDTH = LeverDecl {
        name: "NY_MARGIN_ROW_SPEC_ALPHA_WIDTH",
        kind: LeverKind::UsizeTrimmed,
        default: DefaultSpec::U64(3),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
#spec-alpha `W`: how many of the HARDEST still-failing adversarial classes \
receive their own alpha candidates, default 3. Trimmed; absent or malformed \
leaves 3, and the reader clamps to 1..=64. Selection is hardest-`dj`-first with \
class id ascending as the data-independent tie-break, so it is deterministic. \
Typed preset key `margin_row.spec_alpha_width`; inert when \
`NY_MARGIN_ROW_SPEC_ALPHA` is dark.

The refine pass costs `W * E` COLUMNS, not `W * E` passes — candidates are pass \
WIDTH, because `MarginBatch::new` builds every row independently from its class \
id with no dedup, so a replicated class list yields bit-identical duplicate \
columns. That is the property the measured `#interm-alpha` 5->9 dose regression \
(more candidate PASSES, strictly worse frontier on 5/5 near-closure rows) says \
a successor must have.",
        provenance: Provenance::Unmeasured {
            why_ok: "the per-START-NODE alpha axis has never been built on this lane, so no \
                     retained A/B qualifies a value; the declaration default is dark and the \
                     dark path is one Option discriminant, byte-identical to the shipped root",
        },
        owner: PROPAGATE,
        readers: &[
            ReaderSite {
                scope: PROPAGATE,
                role: "freeze the scored per-start-node alpha policy",
                site: "crates/ny-propagate/src/margin_row/mod.rs:RootAlphaRuntimePolicy::from_levers",
            },
            ReaderSite {
                scope: PROPAGATE,
                role: "resolve the compatibility process policy",
                site: "crates/ny-propagate/src/margin_row/mod.rs:RootAlphaRuntimePolicy::from_layered_process",
            },
        ],
    };

    /// `NY_MARGIN_ROW_SPEC_ALPHA_CANDS` — candidates per selected class.
    pub MARGIN_ROW_SPEC_ALPHA_CANDS = LeverDecl {
        name: "NY_MARGIN_ROW_SPEC_ALPHA_CANDS",
        kind: LeverKind::UsizeTrimmed,
        default: DefaultSpec::U64(3),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
#spec-alpha `E`: candidate alphas per selected class, PER FAMILY, default 3 and \
clamped to 1..=8 by the reader. There are two families and their caps DIFFER, \
which is stated here rather than left to be discovered:

* the GRADIENT family — `clamp(alpha_incumbent + eta*sign(g_k), 0, 1)`, the \
  per-start-node axis itself — takes `min(E, 8)`, the cardinality of \
  `SPEC_ALPHA_ETA` (`1, 0.5, 0.25, 0.75, 0.125, 0.375, 0.0625, 0.03125`, \
  most-distinct-first). The table was three entries, so `E > 3` silently \
  delivered three; it is now eight and `E` is honoured end to end.
* the LADDER family takes `min(E, 5)`, the cardinality of the five closed forms \
  `start_node_gates` already enumerates — the rejected area-heuristic vertex, \
  chord-parallel `s`, `0.5`, `midpoint(s, 1)`, `s/2` — REORDERED \
  most-distinct-first so a truncated `E` keeps the informative ones. Those are \
  GLOBAL UNIFORM alphas, identical for every class, so the ladder's per-class \
  content is SELECTION ONLY and it is the already-measured `#interm-alpha` \
  family.

So round 1 issues up to `W * (min(E,8) + min(E,5))` columns and `E = 8` is the \
first value at which both families are fully expressed. Typed preset key \
`margin_row.spec_alpha_cands`; inert when the parent is dark.",
        provenance: Provenance::Unmeasured {
            why_ok: "the per-START-NODE alpha axis has never been built on this lane, so no \
                     retained A/B qualifies a value; the declaration default is dark and the \
                     dark path is one Option discriminant, byte-identical to the shipped root",
        },
        owner: PROPAGATE,
        readers: &[
            ReaderSite {
                scope: PROPAGATE,
                role: "freeze the scored per-start-node alpha policy",
                site: "crates/ny-propagate/src/margin_row/mod.rs:RootAlphaRuntimePolicy::from_levers",
            },
            ReaderSite {
                scope: PROPAGATE,
                role: "resolve the compatibility process policy",
                site: "crates/ny-propagate/src/margin_row/mod.rs:RootAlphaRuntimePolicy::from_layered_process",
            },
        ],
    };

    /// `NY_MARGIN_ROW_SPEC_ALPHA_ROUNDS` — gradient rounds after the ladder.
    pub MARGIN_ROW_SPEC_ALPHA_ROUNDS = LeverDecl {
        name: "NY_MARGIN_ROW_SPEC_ALPHA_ROUNDS",
        kind: LeverKind::UsizeTrimmed,
        default: DefaultSpec::U64(1),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
#spec-alpha rounds, default 1, clamped to 1..=3. `1` runs the closed-form \
ladder only. `>= 2` takes each class's BEST candidate from the previous round \
and steps it by THAT CLASS'S OWN supergradient — which is what makes the axis \
OPTIMISED rather than enumerated. Typed preset key \
`margin_row.spec_alpha_rounds`; inert when the parent is dark.

The gradient is `alpha_opt::supergradient` reused UNCHANGED: it already returns \
the per-`(neuron, column)` quantity `vp * U` for one specific column, and \
`mean_supergradient` then destroys that resolution by averaging over the K worst \
columns. This key is the consumer that never averages. Each round costs one more \
certified pass, so it is the key that prices the mechanism against the wall.",
        provenance: Provenance::Unmeasured {
            why_ok: "the per-START-NODE alpha axis has never been built on this lane, so no \
                     retained A/B qualifies a value; the declaration default is dark and the \
                     dark path is one Option discriminant, byte-identical to the shipped root",
        },
        owner: PROPAGATE,
        readers: &[
            ReaderSite {
                scope: PROPAGATE,
                role: "freeze the scored per-start-node alpha policy",
                site: "crates/ny-propagate/src/margin_row/mod.rs:RootAlphaRuntimePolicy::from_levers",
            },
            ReaderSite {
                scope: PROPAGATE,
                role: "resolve the compatibility process policy",
                site: "crates/ny-propagate/src/margin_row/mod.rs:RootAlphaRuntimePolicy::from_layered_process",
            },
        ],
    };

    /// `NY_MARGIN_ROW_SPEC_ALPHA_ENTRIES` — hard cap on committed exceptions.
    pub MARGIN_ROW_SPEC_ALPHA_ENTRIES = LeverDecl {
        name: "NY_MARGIN_ROW_SPEC_ALPHA_ENTRIES",
        kind: LeverKind::UsizeTrimmed,
        default: DefaultSpec::U64(65_536),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
#spec-alpha memory governor: hard cap on exception entries committed per refine \
pass, default 65536, reader-capped at 4194304 and additionally by the module's \
own absolute ceiling. `0` REFUSES and prints `[spec-alpha] refuse: \
entries_cap=0` rather than silently running unbounded. Over the cap, WHOLE \
candidates are dropped from the end of the ladder — never partial ones, because \
a half-applied ladder value is a different and undocumented relaxation. Typed \
preset key `margin_row.spec_alpha_entries`; inert when the parent is dark.

WHY IT IS A KEY AND NOT A CONSTANT. The margin-row lane consults NO memory \
governor — `crown_memory`'s adaptive dense budget has zero callers under \
`margin_row/` — and competition invokes ny with no cgroup, so \
`process_memory_headroom_bytes()` returns `None` and nothing clamps anything. \
This cap and the module constant are the only bounds that exist on this \
allocation.",
        provenance: Provenance::Unmeasured {
            why_ok: "the per-START-NODE alpha axis has never been built on this lane, so no \
                     retained A/B qualifies a value; the declaration default is dark and the \
                     dark path is one Option discriminant, byte-identical to the shipped root",
        },
        owner: PROPAGATE,
        readers: &[
            ReaderSite {
                scope: PROPAGATE,
                role: "freeze the scored per-start-node alpha policy",
                site: "crates/ny-propagate/src/margin_row/mod.rs:RootAlphaRuntimePolicy::from_levers",
            },
            ReaderSite {
                scope: PROPAGATE,
                role: "resolve the compatibility process policy",
                site: "crates/ny-propagate/src/margin_row/mod.rs:RootAlphaRuntimePolicy::from_layered_process",
            },
        ],
    };

    /// `NY_MARGIN_ROW_K_ADAPT` — override the preset's adaptive-k choice.
    pub MARGIN_ROW_K_ADAPT = LeverDecl {
        name: "NY_MARGIN_ROW_K_ADAPT",
        kind: LeverKind::Bool,
        // Unset, NOT Bool(false): absence must fall through to the PRESET's
        // choice (`k_adaptive_preset()`), which is neither on nor off a priori.
        // A Bool default here would silently override the preset with `false`.
        default: DefaultSpec::Unset,
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Forces the margin-row adaptive-k width policy on (`1`) or off (`0`), overriding \
the preset. Absence defers to the preset; a malformed present value is rejected \
and suppresses the preset to the declaration's unset/off fallback. This is why \
the default is Unset and the reader uses the centralized layered resolver.",
        provenance: Provenance::Unmeasured {
            why_ok: "an explicit A/B override for a preset-chosen policy; the unset path is \
                     the shipped behaviour and is byte-identical",
        },
        owner: PROPAGATE,
        readers: &[
            ReaderSite {
                scope: PROPAGATE,
                role: "freeze the scored effective and contextual adaptive-width policy",
                site: "crates/ny-propagate/src/margin_row/mod.rs:MarginRowWidthRuntimePolicy::from_levers",
            },
            ReaderSite {
                scope: PROPAGATE,
                role: "resolve the non-scored compatibility adaptive-width policy",
                site: "crates/ny-propagate/src/margin_row/mod.rs:MarginRowWidthRuntimePolicy::from_layered_process",
            },
        ],
    };

    /// `NY_MARGIN_ROW_CLIP` — the margin-row Clip-Verify half.
    pub MARGIN_ROW_CLIP = LeverDecl {
        name: "NY_MARGIN_ROW_CLIP",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Arms the margin-row lane's Clip-Verify half. Exact `1` arms it; absence and every \
other value leave it dark. Verdict-affecting on the authoritative margin-row route: \
it changes which bound the lane publishes. Latched once per process.",
        provenance: Provenance::Unmeasured {
            why_ok: "dark exact-one Debug opt-in landed with 043d7ff75; no scored-row A/B retained",
        },
        owner: PROPAGATE,
        readers: &[ReaderSite {
            scope: PROPAGATE,
            role: "arm the Clip-Verify half of the margin-row lane",
            site: "crates/ny-propagate/src/margin_row/bab.rs",
        }],
    };

    /// `NY_MARGIN_ROW_CLIP_HEAD` — per-domain HEAD-split halfspaces in that half.
    pub MARGIN_ROW_CLIP_HEAD = LeverDecl {
        name: "NY_MARGIN_ROW_CLIP_HEAD",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Arms HEAD-split halfspaces in the margin-row Clip-Verify half. Exact `1` arms it; \
absence and every other value leave it dark. Read PER CALL (never latched): the \
1620640eb incident froze three clip gates at their dark value before the vnncomp \
bridge ran, and they then read as armed while doing nothing. \
\
WHAT IT DOES. `halfspaces_for_splits` builds halfspaces only from \
`LayerGates::clip_rows`, and `RootGates::layers` holds TRUNK relus only, so a \
HEAD sign fix produces no constraint: it moves exactly one interval \
(`YBox::clamp`) and nothing else in the domain tightens. On the traced \
alpha-beta-CROWN rows the head layer is the DOMINANT clip term (10 of 12 splits \
land there; `clip_interm_domain` then re-concretizes 13-17 OTHER head neurons per \
domain). This key builds the missing halfspaces from the y-pack's own certified \
head lines and re-concretizes the other head columns before `head_gates`. \
\
WHY MoatRisk::High, given every step is shrink-only. The halfspace is only a \
NECESSARY condition of its split if the certified slack the lane already paid is \
subtracted from the line: a halfspace without it cuts into the TRUE subdomain, \
which is a false-`unsat` generator and is the class fixed five times in the trunk \
port. The slack is RECOVERED from the pack's untightened forward concretization \
(never from `ly0/uy0` after `clip_tighten_pack`, never from the inherited entry \
box), a rounding pad is ADDED so an over-recovery can only loosen, and a NEGATIVE \
recovered slack refuses the whole pack rather than clamping. Deltas come from the \
untouched `ybox_deltas` dual (non-negative by construction, `None` on refusal) and \
commit with `max`/`min` against the forward base; a crossed interval is counted \
and refused rather than read as an emptiness certificate. \
\
It needs NO `clip_rows` retention and does not consult `clip::frame_ok` — the head \
lines come from the pack the eval already holds — so it can arm on a category that \
retains nothing. Engagement observable: `[clip-head] armed`, then \
`[clip-head] call= ... hs= moved= closes=`; `hs=0` everywhere is VOID, not \
negative.",
        provenance: Provenance::Unmeasured {
            why_ok: "dark exact-one Debug opt-in; the OFF arm computes no slack and takes no \
                     branch, so it is byte-identical to the pre-change lane",
        },
        owner: PROPAGATE,
        readers: &[ReaderSite {
            scope: PROPAGATE,
            role: "arm head-split halfspaces in the Clip-Verify half",
            site: "crates/ny-propagate/src/margin_row/bab.rs",
        }],
    };

    /// `NY_MARGIN_ROW_CLIP_INTERM` — intermediate re-concretization in that half.
    pub MARGIN_ROW_CLIP_INTERM = LeverDecl {
        name: "NY_MARGIN_ROW_CLIP_INTERM",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Re-concretizes intermediate bounds inside the margin-row Clip-Verify half. Exact \
`1` arms it; absence and every other value leave it dark. Latched once per process.",
        provenance: Provenance::Unmeasured {
            why_ok: "dark exact-one Debug opt-in; inert unless the Clip-Verify half is armed",
        },
        owner: PROPAGATE,
        readers: &[ReaderSite {
            scope: PROPAGATE,
            role: "arm intermediate re-concretization in the Clip-Verify half",
            site: "crates/ny-propagate/src/margin_row/engine.rs",
        }],
    };

    /// `NY_MARGIN_ROW_CLIP_TOPK` — unstable neurons per layer to re-concretize.
    pub MARGIN_ROW_CLIP_TOPK = LeverDecl {
        name: "NY_MARGIN_ROW_CLIP_TOPK",
        kind: LeverKind::U64,
        default: DefaultSpec::U64(20),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
How many unstable neurons per layer the Clip-Verify half re-concretizes. Absent or \
malformed leaves 20, which the reader's own comment records as matching \
alpha-beta-CROWN's `bab.clip.interm_topk` default. The parser does NOT trim, so a \
padded value is rejected to the default — preserved verbatim from the reader. \
Latched once per process; inert unless NY_MARGIN_ROW_CLIP_INTERM is armed.",
        provenance: Provenance::Unmeasured {
            why_ok: "inert while its parent lever is dark, which is the shipped state",
        },
        owner: PROPAGATE,
        readers: &[ReaderSite {
            scope: PROPAGATE,
            role: "per-layer re-concretization width for the Clip-Verify half",
            site: "crates/ny-propagate/src/margin_row/engine.rs:clip_interm_topk",
        }],
    };

    /// `NY_RESNET_ERR_CONCRETIZE` — per-segment error concretization on the
    /// SOUND GPU-resident ResNet CROWN walk.
    pub RESNET_ERR_CONCRETIZE = LeverDecl {
        name: "NY_RESNET_ERR_CONCRETIZE",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "Arms per-segment error concretization on the sound GPU-resident ResNet CROWN backward walk: after each segment the accumulated coefficient error is concretized against the frontier magnitudes instead of being carried forward as coefficient radii. Exact `1` arms it; absence and every other value leave it dark. MoatRisk::High, NOT the None a diagnostic probe would get. This sits on the CERTIFIED lane and changes WHERE the error budget is discharged, so it changes the published bound on any run that takes the sound resident ResNet path -- it is a bound-affecting knob that merely looks like a plumbing switch. It is also consulted by the main-bound AUTO-FALLBACK, so arming it changes which of two bounds the walk publishes rather than only how one is computed. DELIBERATELY UNCACHED (no OnceLock), for the same reason `NY_EFT_ERR`'s twin gates are: `crown_backward_sound_resident.rs` flips this mid-process with `ScopedEnvVar::set` in its own differential tests, and a latch would make those tests measure the pre-set value while reporting the post-set one. The chokepoint `read` performs a live `var_os` lookup per call, and all six readers are once per WALK, never per element. Its `_FINE` sibling (`NY_RESNET_ERR_CONCRETIZE_FINE`, per-ReLU rather than per-segment) is a separate raw reader and is NOT declared here; declaring it is the next increment of the same debt.",
        provenance: Provenance::Unmeasured {
            why_ok: "dark exact-one Debug opt-in on a lane that is itself opt-in; the OFF arm                      is the shipped path and is byte-identical, and this declaration is a                      RATCHET migration of six pre-existing byte-identical raw readers -- it                      changes no behaviour, so there is nothing new to measure",
        },
        owner: GPU,
        readers: &[ReaderSite {
            scope: GPU,
            role: "SIX readers in one file, all with the byte-identical exact-\"1\" parser,                    routed through one uncached chokepoint helper: the coefficient-egress                    gate, and five call sites that thread the same decision into the seeded                    resident ResNet walk",
            site: "crates/ny-gpu/src/wgpu_device/ops/crown_backward_sound_resident.rs:\
                   resnet_err_concretize_armed",
        }],
    };

    /// `NY_RESNET_ERR_CONCRETIZE_FINE` — the PER-RELU refinement of the gate
    /// above.
    pub RESNET_ERR_CONCRETIZE_FINE = LeverDecl {
        name: "NY_RESNET_ERR_CONCRETIZE_FINE",
        kind: LeverKind::Bool,
        default: DefaultSpec::Bool(false),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "Refines `NY_RESNET_ERR_CONCRETIZE` from per-SEGMENT to per-RELU: each branch's accumulated error is concretized at every ReLU against the node magnitudes. Exact `1` arms it; absence and every other value leave it dark. Additionally admissible only when the walk was handed non-empty `node_abs`, so an armed value can still resolve to no change — which is why it is declared separately from its parent rather than folded in: a merged declaration would put a decision on the receipt that the walk did not make. MoatRisk::High for the same reason as its parent: it is on the CERTIFIED lane and moves where the error budget is discharged, hence the published bound. DELIBERATELY UNCACHED, same `ScopedEnvVar` mid-process test contract.",
        provenance: Provenance::Unmeasured {
            why_ok: "dark exact-one Debug opt-in, inert unless its parent gate is armed AND                      the walk carries node magnitudes; this declaration is a RATCHET                      migration of one pre-existing byte-identical raw reader and changes no                      behaviour",
        },
        owner: GPU,
        readers: &[ReaderSite {
            scope: GPU,
            role: "per-ReLU error concretization gate on the sound resident ResNet walk",
            site: "crates/ny-gpu/src/wgpu_device/ops/crown_backward_sound_resident.rs:\
                   resnet_err_concretize_fine_armed",
        }],
    };

    /// `NY_MARGIN_ROW_CLOSURE_ADAPTIVE` — three-state override of the
    /// closure-driven runtime escalation policy.
    pub MARGIN_ROW_CLOSURE_ADAPTIVE = LeverDecl {
        name: "NY_MARGIN_ROW_CLOSURE_ADAPTIVE",
        kind: LeverKind::Bool,
        // Unset, NOT Bool(false): absence must fall through to the PRESET's
        // choice (`margin_row.closure_adaptive`), exactly like
        // NY_MARGIN_ROW_K_ADAPT. A Bool default here would silently override
        // the preset with `false`.
        default: DefaultSpec::Unset,
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Forces the margin-row closure-adaptive escalation policy on (`1`) or off (`0`), \
overriding the preset. Absence defers to the preset; a malformed present value \
is rejected and suppresses the preset to the declaration's unset/off fallback. The \
policy measures the lane's OWN child-closure fraction over a trailing window of \
expansions and, when it falls below the threshold, arms the existing tightening \
mechanisms in cost order (beta split dual, then Clip-and-Verify intermediate \
gates, then the Clip-and-Verify y-box). Every tier is sound independently of the \
policy; the certified pass remains the only scorer. Ships DARK. Typed form is \
the preset key `margin_row.closure_adaptive`.",
        provenance: Provenance::Unmeasured {
            why_ok: "three-state override of a preset-chosen policy that itself ships dark; \
                     the unset path is byte-identical to the shipped behaviour",
        },
        owner: PROPAGATE,
        readers: &[ReaderSite {
            scope: PROPAGATE,
            role: "three-state override of the preset closure-adaptive choice",
            site: "crates/ny-propagate/src/margin_row/closure.rs:armed",
        }],
    };

    /// `NY_MARGIN_ROW_CLOSURE_THRESH` — the closure fraction that triggers it.
    pub MARGIN_ROW_CLOSURE_THRESH = LeverDecl {
        name: "NY_MARGIN_ROW_CLOSURE_THRESH",
        kind: LeverKind::F64Open { min: 0.0, max: 1.0 },
        default: DefaultSpec::F64(0.40),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Child-closure fraction below which the closure-adaptive policy escalates. \
Default 0.40, taken from alpha-beta-CROWN's own measured bands on the 16 \
cifar100_2024 rows this lane fails: abc's LOSSES top out at 0.34 and its WINS \
start at 0.50, and 0.40 sits strictly inside that empty band. Absent, malformed, \
non-finite and out-of-range values all leave 0.40. Inert unless the policy is \
armed.",
        provenance: Provenance::Unmeasured {
            why_ok: "inert while its parent policy is dark, which is the shipped state",
        },
        owner: PROPAGATE,
        readers: &[ReaderSite {
            scope: PROPAGATE,
            role: "escalation threshold for the closure-adaptive policy",
            site: "crates/ny-propagate/src/margin_row/closure.rs:threshold",
        }],
    };

    /// `NY_MARGIN_ROW_CLOSURE_WARMUP` — expansions before the first decision.
    pub MARGIN_ROW_CLOSURE_WARMUP = LeverDecl {
        name: "NY_MARGIN_ROW_CLOSURE_WARMUP",
        kind: LeverKind::U64,
        default: DefaultSpec::U64(24),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Expansions the closure-adaptive policy observes before its FIRST decision. \
Default 24 = 1.5x the smallest abc proof on the calibration rows (16 nodes), so a \
row on a winning trajectory has already discharged a batch by then. Absent or \
malformed leaves 24; the reader clamps to 1..=100000. Inert unless the policy is \
armed.",
        provenance: Provenance::Unmeasured {
            why_ok: "inert while its parent policy is dark, which is the shipped state",
        },
        owner: PROPAGATE,
        readers: &[ReaderSite {
            scope: PROPAGATE,
            role: "warm-up window for the closure-adaptive policy",
            site: "crates/ny-propagate/src/margin_row/closure.rs:warmup",
        }],
    };

    /// `NY_MARGIN_ROW_CLOSURE_WINDOW` — expansions per re-evaluation.
    pub MARGIN_ROW_CLOSURE_WINDOW = LeverDecl {
        name: "NY_MARGIN_ROW_CLOSURE_WINDOW",
        kind: LeverKind::U64,
        default: DefaultSpec::U64(16),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Expansions per observation window after the warm-up. Default 16 (about 32 \
children, abc's batch scale). Absent or malformed leaves 16; the reader clamps to \
1..=100000. Inert unless the policy is armed.",
        provenance: Provenance::Unmeasured {
            why_ok: "inert while its parent policy is dark, which is the shipped state",
        },
        owner: PROPAGATE,
        readers: &[ReaderSite {
            scope: PROPAGATE,
            role: "observation window for the closure-adaptive policy",
            site: "crates/ny-propagate/src/margin_row/closure.rs:window",
        }],
    };

    /// `NY_MARGIN_ROW_CLOSURE_MAX_TIER` — cap for single-mechanism arms.
    pub MARGIN_ROW_CLOSURE_MAX_TIER = LeverDecl {
        name: "NY_MARGIN_ROW_CLOSURE_MAX_TIER",
        kind: LeverKind::U64,
        default: DefaultSpec::U64(3),
        bucket: Bucket::Debug,
        moat: MoatRisk::High,
        doc: "\
Highest tier the closure-adaptive ladder may reach: 0 observe-only, 1 beta, \
2 + Clip-and-Verify intermediate gates, 3 + the Clip-and-Verify y-box. Exists so \
a measurement can isolate one mechanism. Absent or malformed leaves 3; the reader \
clamps to the top of the ladder. Inert unless the policy is armed.",
        provenance: Provenance::Unmeasured {
            why_ok: "inert while its parent policy is dark, which is the shipped state",
        },
        owner: PROPAGATE,
        readers: &[ReaderSite {
            scope: PROPAGATE,
            role: "ladder cap for the closure-adaptive policy",
            site: "crates/ny-propagate/src/margin_row/closure.rs:max_tier",
        }],
    };
}
