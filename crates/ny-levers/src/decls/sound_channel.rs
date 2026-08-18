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
}
