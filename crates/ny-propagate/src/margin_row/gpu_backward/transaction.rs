// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The L-lane device transaction (design §6): one certified margin-row
//! backward pass assembled as a single drain-safe multi-step program —
//! head gemm, per-ReLU gate transform + bias reduce, per-conv bias reduce +
//! ds gather, residual merges — ONE submit, UNCONDITIONAL drain, then the
//! host-side directed-rounding egress and the ported guards.
//!
//! # Reuse, not fork (design §5 "plan building")
//!
//! The walk plan and every coefficient layout come from `gpu_seam`'s builder
//! ([`gpu_seam::build_plan_full`]): the checked f32 weight downcast
//! (`to_f32_params_rel`), the `outward_gate_f32` intercept repair, the
//! residual `Add` decomposition with its common-ancestor proof, the
//! `CertifiedWeightError` model-ball composition, and [`gpu_seam::seed_penalty`]
//! port UNCHANGED. `gpu_seam` itself stays dark — only its builder functions
//! are called; its dispatch path (process-global backend, backend deadline
//! lease) is exactly what this module exists to not repeat.
//!
//! # The certified-error contract (§4.2, per accumulation site)
//!
//! Value lanes are double-single (ds32) end to end, INCLUDING the weights and
//! biases: every f64 parameter rides as an exact-split ds pair (normal-band
//! residual charged through [`DS_SPLIT_REL`]; the f32-subnormal band's
//! absolute residue rides the exact split-floor channel — see
//! [`DS_SPLIT_COVER`]), so the old u32 downcast ball — MEASURED
//! as the dominant published-radius term, ~5e6x over the CPU pass on the
//! fixtures — does not exist on this lane. Every remaining ds charge uses the
//! DOCUMENTED per-op bounds (`gamma_ds_dot`: 10u^2/tap vs the old blanket
//! 32u^2/tap) and RIDES THE MASS it belongs to:
//!
//! * head gemm — `g_head = up(gamma_ds_dot(n_y+2) (+) ds_weight_rel(rel))`
//!   gathered on `sum |W||S|` in-kernel, where `rel` is the head's certified
//!   BN-fold error (`TwinNet::gemm1_fold_err`, #cct-margin-row-admission) and
//!   is 0 for every unfolded head (the historical exact-algebra charge); the
//!   head bias statics carry the fold's `bias_err` through the same
//!   `ds_bias_abs` the conv biases use;
//! * gate transform — E contracts by `m = max(|slopes|)` plus one `DS_UNIT`
//!   op on the produced magnitude, in-kernel; intercept partials carry
//!   `E*|c| + DS_UNIT*|partial|` and the reduce adds `gamma_ds_dot` on the
//!   partial mass;
//! * conv backward — `g_src = up(gamma_ds_dot(k_bwd+2) (+)
//!   ds_weight_rel(cv.weight_rel_err))` entering the `|W| * (E + g(|L|+E))`
//!   gather (the CPU `combined_err` shape with the units swapped; the fold
//!   ball is the SAME model error the CPU arm charges);
//! * bias reduces — the CPU relu/conv bias arms' `E*|c| + (|L|+E)*berr`
//!   terms plus `gamma_ds_dot` on the partial mass (charged ONCE — the
//!   pre-tightening build ALSO passed the same envelope as a caller-side
//!   `g_src` on the same mass, a pure double charge, removed) plus the
//!   `2*DS_UNIT*|b_running|` carry;
//! * residual merge — `E_a + E_b + 2*DS_UNIT*|L_sum|` (the CPU `merge_into`
//!   carry with the unit swapped);
//! * split floor — the EXACT per-element split
//!   residuals (`(v - hi) - lo`, exact in f64, [`ds_split_residual`]) in
//!   EXCESS of the guaranteed in-kernel relative covers are accumulated
//!   host-side per split site and carried through host-tracked operator
//!   norms on the plane's coefficient mass into two ABSOLUTE egress floors
//!   ([`BuiltProgram::split_e`]/[`BuiltProgram::split_eb`], added per
//!   element exactly like the flush floors) — identically zero for every
//!   normal-band parameter, and the ONLY charge that covers the
//!   f32-subnormal split band the relative balls provably miss.
//!
//! The E lanes themselves are plain f32 ON DEVICE (WGSL has no directed
//! rounding); ALL certification direction happens HERE at egress — the
//! "certify-final-pass" split: every E/eb readback is widened in f64 by the
//! composed `(1 + gamma_f32(k))` factor of its whole on-device accumulation
//! (nonneg sums, so Higham bounds the f32 under-accumulation — this widens
//! the E-lane's OWN arithmetic, which the device never charges, so it is NOT
//! a double charge; audited 2026-08-20, it is a multiplicative `1 + ~1e-3`
//! per conv stage on an already-small quantity), rounded UP at every host op,
//! and floored by the unconditional `ops * 2^-126` flush charge (see
//! [`FLUSH_FLOOR`] for the under-DenormPreserve derivation of why the floor
//! is irreducible in kind and ~1e-33 in magnitude). Coefficient and bias
//! VALUES convert ds -> f64 exactly (both components exact in f64, the sum
//! needs <= ~49 bits).
//!
//! # Guards (ported per design R5)
//!
//! [`gpu_seam::error_floor_ok`] against the plan's `rho*` (the model ball
//! keeps the floor meaningful — §7) and [`gpu_seam::probe_guard`] (a realized
//! value of the functional must stay bracketed) run on every pass, after the
//! NaN/Inf firewall. Any trip refuses; nothing repairs.

use std::sync::OnceLock;
use std::time::Instant;

use ndarray::Array2;
use ny_core::{
    GpuCrownLayer, GpuResnetSegment, LaneArenaBuffer, LaneProgramSpec, LaneProgramTiming,
    LaneResidentId, LaneStepBinding, LaneStepSpec, MarginRowLaneGpu,
};

use super::super::engine::{BackwardEngine, DomainGates, Exceptions, LaneDir, PassOut, Seed};
use super::super::gpu_seam::{self, Plan};
use super::super::net::TwinOp;
use super::super::rounding::{gamma_n_f32, next_down, next_up};
use super::authority::VerdictAuthority;
use super::cache::{self, PlanCache, PlanStatic, ResidentPlane};
use super::ds::gamma_ds_dot;
use super::{
    PassShape, Refusal, COLLECT_ABS_WGSL, CONV_BACKWARD_WGSL, DS_PRIMITIVES_WGSL,
    GATE_TRANSFORM_WGSL, HEAD_GEMM_WGSL, MERGE_ADD_WGSL, REDUCE_WGSL,
};

/// Smallest positive NORMAL f32 (2^-126), as f64: the absolute flush floor —
/// one flushed/unrepresentable rounding costs at most this much, absolutely.
///
/// RE-DERIVATION UNDER PRESERVED DENORMALS (the resolved DenormPreserve
/// channel is an ADMISSION requirement, so the floor may assume it): the
/// barrier TwoSum residual is exactly representable for ALL inputs when
/// subnormals exist (Hauser's theorem — the error of an
/// f32 addition is itself an f32), so ADDS owe nothing here. What remains,
/// and keeps the floor IRREDUCIBLE IN KIND rather than removable:
///
/// 1. the TwoProduct exact-residual band (`ny_core::eft`'s 2^-101 guard):
///    a product's residual can need exponents below 2^-149 REGARDLESS of any
///    flush behavior — a representability gap, not FTZ; bounded by 2^-126
///    absolute per product (the banked 2026-08-02 band analysis, reused);
/// 2. the adapter caveat that an fma whose RESULT is subnormal may still
///    flush even with the word channel ON (design §4.1) — 2^-126 absolute
///    per fused rounding, value and E lanes alike.
///
/// Both are absolute and margin-invisible (the whole charge totals ~1e-33 on
/// a resnet-scale pass — see the charge-profile telemetry) but are CHARGED,
/// never assumed away.
const FLUSH_FLOOR: f64 = 1.175_494_350_822_287_5e-38;

/// Per-accumulation-op multiplier for the flush-floor op count: one ds_mul +
/// ds_add tap is ~27 roundings (7 + 20) plus ~3 E-lane ops; 32 per tap
/// dominates with headroom. Loosening-only.
const FLUSH_OPS_PER_TAP: f64 = 32.0;

/// Representation ball of an f64 parameter carried as an exact-split ds32
/// pair, RELATIVE TO THE SUPPLIED `|hi| + |lo|` mass: `2^-47`, exact in f64.
///
/// Derivation (NORMAL BAND — `lo = fl32(w - hi)` rounding relatively, i.e.
/// `|w - hi| >= 2^-126`): `hi = fl32(w)` has `|w - hi| <= 2^-24 |w|`;
/// `w - f64::from(hi)` is EXACT in f64 because the exact difference is a
/// multiple of `ulp_f64(w)` with magnitude `<= 2^-24 |w|`, so its ~29
/// significant low-order bits fit a 53-bit significand and the subtraction
/// commits no rounding (REPRESENTABILITY, not the "Sterbenz" the earlier
/// wording cited — Sterbenz's factor-of-two precondition also happens to
/// hold, see [`ds_split_residual`], but the bit-count argument is the load-
/// bearing one). Then `lo = fl32(w - hi)` leaves
/// `|w - (hi + lo)| <= 2^-24 |w - hi| <= 2^-48 |w|` ([`ds_split_f64`]).
/// Relative to the supplied mass
/// `|hi| + |lo| >= |hi + lo| >= (1 - 2^-48)|w|` that is
/// `<= 2^-48/(1 - 2^-48)`; `2^-47` covers it with ~2x headroom. This replaces
/// the old `U32_UNIT` (2^-24) downcast ball everywhere in this lane — the
/// single largest term of the pre-tightening published radius.
///
/// SUBNORMAL EXCEPTION: when
/// `hi` or `lo` lands f32-SUBNORMAL (`|w| < ~2^-102` puts `lo` there,
/// `|w| < 2^-126` puts `hi` there), that component's rounding is ABSOLUTE
/// (`<= 2^-150` half-ulp of the subnormal grid), not relative, and the
/// `2^-48 |w|` claim above is MEASURED-FALSE — worst counterexample 2.1e6x
/// under (`(1 + 2^-26 + 2^-52) * 2^-130`; pinned in
/// `tests::split_residual_excess_covers_the_subnormal_band`). The band is
/// NOT waved off on realistic-exposure grounds (charged-not-assumed-away):
/// every split site ALSO charges the EXACT per-element residual excess over
/// its guaranteed in-kernel cover ([`ds_split_excess`] with
/// [`DS_SPLIT_COVER`] / [`DS_GATE_COVER`]) into the absolute split-floor
/// channel ([`BuiltProgram::split_e`] / [`BuiltProgram::split_eb`]), which
/// is exact-by-construction in EVERY band and identically zero for
/// normal-band parameters.
pub(super) const DS_SPLIT_REL: f64 = 7.105_427_357_601_002e-15;

/// Guaranteed IN-KERNEL cover base for one weight/bias split residual,
/// relative to the pair mass `|hi| + |lo|`: `2^-48`, exact — HALF of
/// [`DS_SPLIT_REL`]. Every weight/bias split rides a per-tap kernel charge
/// of at least `DS_SPLIT_REL * fl32(|hi|+|lo|) * fl32(|coeff|)`
/// ([`ds_weight_rel`] composes UPWARD from `DS_SPLIT_REL`; [`ds_bias_abs`]
/// adds `DS_SPLIT_REL * max|b| >= DS_SPLIT_REL * |b_j|` into `berr`), and
/// the in-kernel f32 mass products shave `< 2^-22` relative — so HALF the
/// constant floors what the kernel certainly charges, with ~2x slack
/// absorbing those roundings. [`ds_split_excess`] charges everything above
/// THIS base exactly, so cover + excess dominates the true residual in
/// every band. In the corner where the kernel's cover term itself flushes
/// (`|W||L| < 2^-126`) the whole needed cover is `< 2^-126` absolute per
/// element — dominated by the per-element [`FLUSH_FLOOR`] egress charge.
pub(super) const DS_SPLIT_COVER: f64 = 3.552_713_678_800_501e-15; // 2^-48, exact

/// Guaranteed IN-KERNEL cover base for one GATE split residual, relative to
/// the gate pair mass: `2^-45`, exact — HALF of `U_DS` (`2^-44`). The gates
/// are split HOST-side ([`ds_split_f64`]; the kernel never splits), and the
/// gate transform charges `2 * DS_UNIT * |o|` per element with
/// `|o| >= (1 - ~2^-22) * mass * |v|` away from the o-flush corner, so one
/// of the two `DS_UNIT` ops certainly covers `2^-45 * mass * |v|`; the
/// subnormal-band excess above that base is charged exactly through
/// [`ds_split_excess`] into the split-floor channel, and the o-flush corner
/// (`|slope * v| < 2^-126`) needs `< 2^-45 * 2^-126` absolute — dominated
/// by the per-element [`FLUSH_FLOOR`] egress charge.
pub(super) const DS_GATE_COVER: f64 = 2.842_170_943_040_401e-14; // 2^-45, exact

/// wgpu's per-axis dispatch limit; a wider pass refuses (fail-closed) rather
/// than splitting this session. (65535 workgroups x 256 threads covers every
/// cifar100/tinyimagenet plane at competition row counts.)
///
/// STATED CAP (fix A5, exact): a plane of `width` elements over `R` pass
/// rows dispatches `ceil(width*R / 256)` 256-lane workgroups, so admission
/// requires `width * R <= 65535 * 256 = 16_776_960` threads — NOT
/// `65536 * 256`: at frontier width 65536, `R <= 255` fits, `R = 256` is
/// one workgroup over and refuses (fail-closed -> the CPU pass).
const MAX_WORKGROUPS_X: u64 = 65_535;

/// The STATED B4 dispatch cap, exact (fix A5): a batched-children scoring
/// pass over `R = 2*k_head + 2*k_trunk` candidate blocks dispatches
/// `ceil(width*R / 256)` 256-lane workgroups on its widest plane, so the pass
/// is admissible only while
///
/// ```text
///     width * R <= 65535 * 256 = 16_776_960
/// ```
///
/// NOT `65536 * 256`: at frontier width 65536, `R <= 255` fits and `R = 256`
/// is one workgroup over. Enforced structurally by [`Builder::workgroups_1d`]
/// on EVERY plane (so the widest one decides) and re-checked up front by
/// [`dispatch_thread_cap`] so a refusal reads as a cap refusal rather than as
/// an anonymous mid-build `Unmappable`. Fail-closed either way: over the cap
/// the pass refuses and the caller runs the CPU pass.
pub(super) const MAX_DISPATCH_THREADS: u64 = MAX_WORKGROUPS_X * 256;

/// `width * rows` against [`MAX_DISPATCH_THREADS`] (fix A5), as a pure
/// predicate so the cap is unit-testable without a device or a plan.
pub(super) fn dispatch_thread_cap(width: usize, rows: usize) -> Result<(), Refusal> {
    let threads = (width as u64).saturating_mul(rows as u64);
    if threads == 0 || threads > MAX_DISPATCH_THREADS {
        return Err(Refusal::Unmappable("pass exceeds the 65535x256 thread cap"));
    }
    Ok(())
}

/// Cap on B4 exception RECORDS per relu layer. Every thread of the gate
/// transform scans the layer's record list, so the list length multiplies the
/// kernel's inner cost; the batched-children pass compresses its per-row
/// exceptions into ONE record per (neuron, contiguous row block), which is
/// `2 * k_trunk` records at the shipped preset (k=16 => 32). A pass that
/// cannot be expressed inside this cap refuses (fail-closed -> CPU) rather
/// than silently paying an O(records) inner loop.
pub(super) const MAX_EXC_RECORDS: usize = 256;

/// PER-PASS ARENA BYTE BUDGET (design §5 risk-3 mitigation; FIX D2, impl-v9).
///
/// The lane's device is dedicated but its MEMORY is not: the GB10 is 121 GiB
/// SHARED CPU+GPU, and the internal verifier's intermediate sweep runs ~2 GB x
/// 34 waves through the same pool while the lane shadows. Before v9 the only
/// bounds were a THREAD cap ([`dispatch_thread_cap`]) and the executor's pool
/// RETENTION valve (`margin_row_lane.rs::MAX_POOLED_BYTES`) — neither of which
/// limits how much a single program allocates. At the shipped B4 shape
/// (`R = 2*k_head + 2*k_trunk = 64` at the shipped k=16 preset) one relu
/// allocates ~24 B x n x R (coeff 8 + err 4 + bias_partial 8 + partial_err 4),
/// i.e. ~22 MB at n = 14400; x19 relus plus the conv planes puts a single
/// cifar100 program at ~0.4-0.5 GB live.
///
/// 1 GiB is therefore ~2x the measured worst shipped shape and <1% of the
/// box, while refusing the runaway shapes (a wider frontier, or R >= 256)
/// that would contend with the sweep. FAIL-CLOSED: over budget the pass
/// refuses and the caller runs the exact CPU pass (the `run_seamed`
/// contract), so this can only cost throughput, never a bound.
///
/// IF IT EVER REFUSES A SHIPPED SHAPE it is LOUD, by construction: the refusal
/// carries its own reason string, the per-site ledger counts it, and
/// `[gpu-backward] refused site=... : pass arena exceeds the per-pass byte
/// budget` prints once per site (`mod.rs::note_refusal`). Raise the constant
/// deliberately after reading that line — never silence it, and never widen it
/// past what the sweep leaves free.
pub(super) const MAX_PASS_ARENA_BYTES: u64 = 1 << 30;

/// Buffer bytes as the EXECUTOR will allocate them: wgpu requires
/// `COPY_BUFFER_ALIGNMENT` (4) and the lane rounds up defensively
/// (`margin_row_lane.rs::create_buffer`), so the budget counts the same
/// padded size the device does.
pub(super) fn padded_buffer_len(len: u64) -> u64 {
    len.max(4).next_multiple_of(4)
}

/// `arena_bytes` against [`MAX_PASS_ARENA_BYTES`] (FIX D2), as a pure
/// predicate so the budget is unit-testable without a device or a plan.
pub(super) fn arena_byte_cap(bytes: u64) -> Result<(), Refusal> {
    if bytes > MAX_PASS_ARENA_BYTES {
        return Err(Refusal::Unmappable(
            "pass arena exceeds the per-pass byte budget",
        ));
    }
    Ok(())
}

pub(super) fn shader_head() -> &'static str {
    static S: OnceLock<String> = OnceLock::new();
    S.get_or_init(|| format!("{DS_PRIMITIVES_WGSL}\n{HEAD_GEMM_WGSL}"))
}
pub(super) fn shader_gate() -> &'static str {
    static S: OnceLock<String> = OnceLock::new();
    S.get_or_init(|| format!("{DS_PRIMITIVES_WGSL}\n{GATE_TRANSFORM_WGSL}"))
}
pub(super) fn shader_conv() -> &'static str {
    static S: OnceLock<String> = OnceLock::new();
    S.get_or_init(|| format!("{DS_PRIMITIVES_WGSL}\n{CONV_BACKWARD_WGSL}"))
}
pub(super) fn shader_reduce() -> &'static str {
    static S: OnceLock<String> = OnceLock::new();
    S.get_or_init(|| format!("{DS_PRIMITIVES_WGSL}\n{REDUCE_WGSL}"))
}
pub(super) fn shader_merge() -> &'static str {
    static S: OnceLock<String> = OnceLock::new();
    S.get_or_init(|| format!("{DS_PRIMITIVES_WGSL}\n{MERGE_ADD_WGSL}"))
}
/// B5: the heuristic-only per-unstable |coefficient| collector.
pub(super) fn shader_collect() -> &'static str {
    static S: OnceLock<String> = OnceLock::new();
    S.get_or_init(|| format!("{DS_PRIMITIVES_WGSL}\n{COLLECT_ABS_WGSL}"))
}

/// Map a plan-builder / guard refusal from the reused `gpu_seam` machinery
/// into this lane's refusal vocabulary (fail-closed on anything device-y the
/// builder should never return).
pub(super) fn map_seam_refusal(refusal: gpu_seam::Refusal) -> Refusal {
    match refusal {
        gpu_seam::Refusal::NotOutward => Refusal::NotOutward,
        gpu_seam::Refusal::Rows => Refusal::Rows,
        gpu_seam::Refusal::Unmappable(what) => Refusal::Unmappable(what),
        gpu_seam::Refusal::SeedNeedsYAbs => Refusal::Unmappable("seed error needs y_abs"),
        gpu_seam::Refusal::NonFinite => Refusal::NonFinite,
        gpu_seam::Refusal::Payload => Refusal::Payload,
        gpu_seam::Refusal::ErrorFloor => Refusal::ErrorFloor,
        gpu_seam::Refusal::Probe => Refusal::Probe,
        _ => Refusal::Unmappable("unexpected plan-builder refusal"),
    }
}

// ---------------------------------------------------------------------------
// Program assembly
// ---------------------------------------------------------------------------

pub(super) enum Init<'p> {
    /// Per-pass bytes owned by the program (seed plane, gate quads).
    Data(Vec<u8>),
    /// Pass-invariant bytes BORROWED from the [`PlanStatic`] (B1: the
    /// residency-refused fallback — same bytes, still uploaded per pass).
    DataRef(&'p [u8]),
    /// Zero-initialized intermediate (pooled device-side by the executor).
    Zeroed(u64),
    /// Pass-invariant plane already resident on the device (B1: no per-pass
    /// upload at all).
    Resident(LaneResidentId),
}

pub(super) struct Step {
    pub(super) label: &'static str,
    pub(super) entry: &'static str,
    pub(super) shader: &'static str,
    pub(super) uniforms: Vec<u8>,
    pub(super) bindings: Vec<LaneStepBinding>,
    pub(super) workgroups: [u32; 3],
}

/// The current backward frontier: coefficient ds plane, error f32 plane,
/// logical width (neurons) — plus the HOST-tracked split-floor bookkeeping:
/// `amax` bounds the plane's largest coefficient
/// magnitude and `sabs` the per-element ABSOLUTE split-residual error the
/// device E plane does NOT cover (both directed UP at every host op; both
/// exactly 0.0 whenever every parameter split so far was normal-band).
#[derive(Clone, Copy)]
struct Frontier {
    coeff: usize,
    err: usize,
    width: usize,
    /// Upper bound on `max |hi + lo|` over the plane (stored device values).
    amax: f64,
    /// Per-element absolute split-floor slack carried by this plane.
    sabs: f64,
}

/// One B5 collect readback: which relu layer it belongs to and how many
/// unstable positions it carries (`RootGates::layers[layer].unst.len()`).
/// HEURISTIC-ONLY (see `kernels/collect_abs.wgsl`): nothing verdict-bearing
/// reads the published values.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct CollectOut {
    pub(super) layer: usize,
    pub(super) positions: usize,
}

pub(super) struct BuiltProgram<'p> {
    pub(super) buffers: Vec<Init<'p>>,
    pub(super) steps: Vec<Step>,
    /// `[final coeff (ds), final err (f32), bias_acc (ds), ebias_acc (f32)]`
    /// followed by ONE ds buffer per [`Self::colls`] entry, in order (B5).
    pub(super) readbacks: Vec<usize>,
    /// B5 collect descriptors, index-aligned with `readbacks[4..]`.
    pub(super) colls: Vec<CollectOut>,
    /// Was the `Collect{unst_abs:true}` shape requested? The CPU arm
    /// publishes `Some(map)` whenever it was — EMPTY if no layer had
    /// unstable neurons (`engine.rs:493`) — so egress must too, or a
    /// downstream `unwrap_or_default()` would silently differ in kind.
    pub(super) collect_unst_abs: bool,
    /// Egress widening factor for the coefficient E plane (composed
    /// `(1 + gamma_f32(k))` over every on-device E stage, rounded up).
    pub(super) e_widen: f64,
    /// Same for the bias-error accumulator.
    pub(super) eb_widen: f64,
    /// Absolute flush floors (§4.1), per E element / per eb entry.
    pub(super) flush_e: f64,
    pub(super) flush_eb: f64,
    /// Absolute split-floor channel: the exact
    /// per-element parameter-split residual excess (subnormal band) carried
    /// to the FINAL coefficient plane / the bias accumulators through the
    /// host-tracked operator norms. Added per E element / per eb entry at
    /// egress exactly like the flush floors. 0.0 on normal-band parameters.
    pub(super) split_e: f64,
    pub(super) split_eb: f64,
    /// THIS lane's model-ball floor for `error_floor_ok`: the largest
    /// fold(+split) relative weight ball any layer was charged — NOT the plan
    /// builder's `rho_star`, which includes the u32 downcast this lane's ds
    /// pairs replaced (flooring against a ball that is not charged would
    /// refuse every pass).
    pub(super) rho_star_ds: f64,
    /// The FINAL frontier's host-tracked coefficient-magnitude bound
    /// (`Frontier::amax`). Never an egress term — published for the FIX A1
    /// counterexample pin, which asserts that an exception whose true
    /// coefficients exceed the BASE gate `amax` is nevertheless covered.
    pub(super) amax: f64,
    /// Once-per-process charge-profile telemetry lines (R9: the per-charge
    /// breakdown must be readable off any shadow run log).
    pub(super) profile: Vec<String>,
    /// FIX D2: total per-pass arena bytes this program will allocate on the
    /// device (padded exactly as the executor pads), EXCLUDING resident
    /// planes (registry-owned, counted by `resident_bytes`). Checked against
    /// [`MAX_PASS_ARENA_BYTES`] at build time; published for the D2 telemetry
    /// and the pin.
    pub(super) arena_bytes: u64,
    pub(super) n_in: usize,
    pub(super) rows: usize,
}

struct Builder<'a, 'b> {
    rows: usize,
    dir: LaneDir,
    /// Gate + PARAMETER provenance: the engine's root/domain gates addressed
    /// through the plan's fold-order relu list (as `gpu_seam::retarget_plan`
    /// does), and the engine's f64 conv weights addressed through the same
    /// traversal's `conv_ops` list (`gpu_seam::build_plan_ext`).
    eng: &'a BackwardEngine<'b>,
    dom: Option<&'a DomainGates>,
    /// B4: per-(row, neuron) gate overrides, VERDICT-BEARING (the scored
    /// child bound is banked as the child's inherited bound, bab.rs:1612).
    exc: Option<&'a Exceptions>,
    /// B5: emit the heuristic-only per-unstable |coefficient| collector.
    collect_unst_abs: bool,
    /// B5 collect descriptors + their arena buffers, in emission order.
    colls: Vec<(CollectOut, usize)>,
    /// B1: the pass-invariant half (weight/bias ds planes + their charges,
    /// computed ONCE per RootGates instance by the same helper functions
    /// this builder used to call per pass — bit-identical by construction).
    statics: &'a PlanStatic,
    relus: &'a [usize],
    next_relu: usize,
    convs: &'a [usize],
    next_conv: usize,
    buffers: Vec<Init<'a>>,
    steps: Vec<Step>,
    bias_acc: usize,
    ebias_acc: usize,
    dummy_scale: usize,
    e_widen: f64,
    eb_widen: f64,
    flush_ops_e: f64,
    flush_ops_eb: f64,
    /// Absolute split-floor accumulated into the bias channel (verify-fix):
    /// each reduce site adds its plane's uncovered split slack + the scale
    /// pairs' own residual excess, directed UP.
    split_eb: f64,
    rho_star_ds: f64,
    profile: Vec<String>,
    /// FIX D2: running per-pass arena byte total (see [`arena_byte_cap`]).
    arena_bytes: u64,
    head_seen: bool,
}

/// `up(a * b)` — one directed product for the split-floor bookkeeping.
/// EXACT-ZERO-PRESERVING: a product with a zero OPERAND is exactly 0.0 and
/// owes no up-step (a blind `next_up(0.0)` would seed the channel with one
/// minimal subnormal per site and the all-normal-band profile would read
/// ~1e-320 instead of the contractual 0.0). The operand check, not
/// `a * b == 0.0`, so a genuine underflow-to-zero still rounds UP.
pub(super) fn mul_up(a: f64, b: f64) -> f64 {
    if a == 0.0 || b == 0.0 {
        return 0.0;
    }
    next_up(a * b)
}

/// `up(acc + x)` with the same exact-zero discipline: adding 0.0 is exact,
/// so the accumulator takes an up-step only for a genuinely nonzero addend.
pub(super) fn add_up(acc: f64, x: f64) -> f64 {
    if x == 0.0 {
        return acc;
    }
    next_up(acc + x)
}

/// Operator-norm inflation for the split-floor propagation: the in-kernel ds
/// fold rounds relative to the ACTUAL (deviation-carrying) masses, so the
/// host-tracked `amax`/`sabs` transforms are inflated by the same per-tap
/// envelope the kernel charges — `up(1 + gamma_ds_dot(taps + 2))`.
/// Loosening-only (the channel is 0.0 unless a subnormal split occurred).
pub(super) fn norm_inflate(norm: f64, taps: usize) -> f64 {
    next_up(norm * next_up(1.0 + gamma_ds_dot(taps + 2)))
}

fn pack_words(words: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(words.len() * 4);
    for w in words {
        out.extend_from_slice(&w.to_le_bytes());
    }
    out
}

fn u32_of(x: usize, what: &'static str) -> Result<u32, Refusal> {
    u32::try_from(x).map_err(|_| Refusal::Unmappable(what))
}

/// `f64 -> f32` toward +inf, refusing non-finite (a saturated gamma becomes a
/// huge-but-finite charge, which the NaN/Inf firewall then rejects via the
/// error floor of usefulness rather than by wrapping).
pub(super) fn f32_up(x: f64) -> Result<f32, Refusal> {
    if !x.is_finite() {
        return Err(Refusal::Unmappable("non-finite error charge"));
    }
    let out = ny_core::f64_to_f32_up(x);
    if !out.is_finite() {
        return Err(Refusal::Unmappable("error charge overflows f32"));
    }
    Ok(out)
}

/// Compose `gamma (+) rho` upward: `up(gamma + rho + gamma*rho)` — the CPU
/// conv arm's charge shape (engine.rs), each add stepped separately.
fn compose_charge(gamma: f64, rho: f64) -> f64 {
    next_up(next_up(gamma + rho) + next_up(gamma * rho))
}

/// The lane's weight ball: the engine's FOLD error (the model term the CPU
/// arm charges too — `cv.weight_rel_err`, exact-weight-relative) composed
/// with the ds-split representation residual, converted to the SUPPLIED-mass
/// denominator with the same `R/(1-R)` shape (and the same directed steps) as
/// `gpu_seam::weight_error` — but with [`DS_SPLIT_REL`] in place of the u32
/// downcast unit, because this lane's weights are ds pairs, not f32 scalars.
/// NORMAL band only: the f32-subnormal split band is charged exactly through
/// the split-floor channel ([`DS_SPLIT_COVER`], [`ds_split_excess`]).
pub(super) fn ds_weight_rel(fold_rel: f64) -> Result<f64, Refusal> {
    if !fold_rel.is_finite() || fold_rel < 0.0 {
        return Err(Refusal::Unmappable("non-finite fold weight_rel_err"));
    }
    let cross = next_up(fold_rel * DS_SPLIT_REL);
    let exact_rel = next_up(next_up(fold_rel + DS_SPLIT_REL) + cross);
    if exact_rel >= 1.0 {
        return Err(Refusal::Unmappable("ds weight ball reaches unity"));
    }
    let denominator = next_down(1.0 - exact_rel);
    if denominator <= 0.0 {
        return Err(Refusal::Unmappable("ds weight ball denominator"));
    }
    Ok(next_up(exact_rel / denominator))
}

/// The lane's absolute bias ball: the engine's per-channel FOLD `bias_err`
/// maximum plus the ds-split residual on the bias magnitude (per entry, in
/// the NORMAL band: `|b - pair| <= 2^-48 |b| <= DS_SPLIT_REL * max|b|`) —
/// the `gpu_seam::weight_error` bias line with the split unit in place of
/// u32. The f32-subnormal band's absolute residue is NOT covered by this
/// relative line (the refuted charge): each reduce site charges the exact
/// PER-ENTRY excess ([`ds_split_excess`]) into the split-floor channel.
pub(super) fn ds_bias_abs(bias: &[f64], bias_err: &[f64]) -> Result<f64, Refusal> {
    let mut abs_bias = 0.0f64;
    for &b in bias {
        if !b.is_finite() {
            return Err(Refusal::Unmappable("non-finite bias"));
        }
        abs_bias = abs_bias.max(b.abs());
    }
    let mut err_bias = 0.0f64;
    for &e in bias_err {
        if !(e.is_finite() && e >= 0.0) {
            return Err(Refusal::Unmappable("non-finite bias_err"));
        }
        err_bias = err_bias.max(e);
    }
    Ok(next_up(err_bias + DS_SPLIT_REL * abs_bias))
}

impl<'a, 'b> Builder<'a, 'b> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        rows: usize,
        dir: LaneDir,
        eng: &'a BackwardEngine<'b>,
        dom: Option<&'a DomainGates>,
        shape: PassShape<'a>,
        statics: &'a PlanStatic,
        relus: &'a [usize],
        convs: &'a [usize],
    ) -> Self {
        let rows_u64 = rows as u64;
        let buffers = vec![
            // Running per-row bias accumulators (ds value + f32 error).
            Init::Zeroed(rows_u64 * 8),
            Init::Zeroed(rows_u64 * 4),
            // One-PAIR dummy scale for use_scale = 0 reduces (the kernel applies
            // the identity pair instead of reading it).
            Init::Data(vec![0u8; 8]),
        ];
        // FIX D2: seed the arena budget with the three buffers pushed above
        // (`add_buffer` is the only other way in, and it accounts there).
        let arena_bytes = padded_buffer_len(rows_u64 * 8)
            .saturating_add(padded_buffer_len(rows_u64 * 4))
            .saturating_add(padded_buffer_len(8));
        Self {
            rows,
            dir,
            eng,
            dom,
            exc: shape.exc,
            collect_unst_abs: shape.collect_unst_abs,
            colls: Vec::new(),
            statics,
            relus,
            next_relu: 0,
            convs,
            next_conv: 0,
            buffers,
            steps: Vec::new(),
            bias_acc: 0,
            ebias_acc: 1,
            dummy_scale: 2,
            e_widen: 1.0,
            eb_widen: 1.0,
            flush_ops_e: 0.0,
            flush_ops_eb: 0.0,
            split_eb: 0.0,
            rho_star_ds: 0.0,
            profile: Vec::new(),
            arena_bytes,
            head_seen: false,
        }
    }

    /// Register one arena buffer, charging its DEVICE size against the
    /// per-pass byte budget (FIX D2). Fail-closed: the first buffer that
    /// crosses [`MAX_PASS_ARENA_BYTES`] refuses the whole pass — before any
    /// device allocation happens — and the caller runs the exact CPU pass.
    /// Resident planes cost 0 here: they are registry-owned, uploaded once,
    /// and accounted by the executor's `resident_bytes`.
    fn add_buffer(&mut self, init: Init<'a>) -> Result<usize, Refusal> {
        let bytes = match &init {
            Init::Data(b) => padded_buffer_len(b.len() as u64),
            Init::DataRef(b) => padded_buffer_len(b.len() as u64),
            Init::Zeroed(len) => padded_buffer_len(*len),
            Init::Resident(_) => 0,
        };
        self.arena_bytes = self.arena_bytes.saturating_add(bytes);
        arena_byte_cap(self.arena_bytes)?;
        self.buffers.push(init);
        Ok(self.buffers.len() - 1)
    }

    /// Bind a pass-invariant plane: resident handle when the one-time upload
    /// was accepted, else the borrowed bytes as a per-pass upload (B1
    /// fail-closed fallback — same bytes either way).
    fn add_plane(&mut self, plane: &'a ResidentPlane) -> Result<usize, Refusal> {
        match plane.resident {
            Some(id) => self.add_buffer(Init::Resident(id)),
            None => self.add_buffer(Init::DataRef(&plane.bytes)),
        }
    }

    /// Compose one E-stage's f32 rounding envelope into a widening factor
    /// (upward at every host op, so the factor dominates the true product).
    fn widen(factor: &mut f64, ops: usize) {
        *factor = next_up(*factor * next_up(1.0 + gamma_n_f32(ops)));
    }

    fn workgroups_1d(total: usize) -> Result<[u32; 3], Refusal> {
        let groups = (total as u64).div_ceil(256);
        if groups == 0 || groups > MAX_WORKGROUPS_X {
            return Err(Refusal::Unmappable("dispatch exceeds one workgroup axis"));
        }
        #[allow(clippy::cast_possible_truncation)]
        Ok([groups as u32, 1, 1])
    }

    /// One 256-lane WORKGROUP PER ROW (the B2 tree reduce): `rows` must fit
    /// the workgroup axis on its own (competition rows are <= ~100).
    fn workgroups_rows(rows: usize) -> Result<[u32; 3], Refusal> {
        if rows == 0 || rows as u64 > MAX_WORKGROUPS_X {
            return Err(Refusal::Unmappable("reduce rows exceed the workgroup axis"));
        }
        #[allow(clippy::cast_possible_truncation)]
        Ok([rows as u32, 1, 1])
    }

    /// One bias_reduce dispatch folding `plane`/`err` (width n) into the
    /// running accumulators. The fold's own ds envelope is charged EXACTLY
    /// once, in-kernel, as `g_reduce * sum|q|` (the pre-tightening build also
    /// passed the same `gamma_ds(n+2)` as a caller-side `g_src` on the same
    /// mass — a double charge, removed). `split_contrib` is the
    /// caller-computed per-row split-floor slack this fold moves into the
    /// bias channel (plane slack x scale mass + scale split excess x plane
    /// amax; the bias accumulators are terminal, so no later amplification).
    ///
    /// B2 (perf session): dispatches the TREE entry `bias_reduce_wg` — one
    /// 256-lane workgroup per row, contiguous per-lane blocks folded
    /// serially then a fixed binary tree across lanes.
    ///
    /// CHARGE ARGUMENT (`g_reduce = gamma_ds_dot(n+2)` UNCHANGED — valid for
    /// ANY summation order): the Higham-style envelope counts, per rounding
    /// ds op, one `DS_TAP_EPS`-relative error against the absolute mass
    /// `sum|q|`; every multiply rounds relative to its term (`<= mass`) and
    /// every add relative to a PARTIAL SUM of a SUBSET of the terms
    /// (`<= mass` in any order — subsets of nonneg-mass-bounded terms).
    /// The total count of rounding ops in a tree over n terms is n
    /// multiplies + at most n-1 non-trivial adds (an `eft`-exact add of an
    /// exact-zero padding lane commits NO rounding — `two_sum(x, 0)` is
    /// exact), i.e. exactly the SERIAL count, and each element's own
    /// accumulation path is `<= chunk + ceil(log2 256) + 1 <= n + 2` ops.
    /// The serial charge `gamma_ds_dot(n+2)` therefore strictly dominates
    /// the tree fold too (kept conservative rather than re-derived
    /// tighter). The E-side f32 sums (`esum`/`msum`) also keep their
    /// serial-count host widening (`6n+16` ops: tree total = 2n fma +
    /// n + 2(n-1) adds <= 6n+16 for all n) — so the HOST-side charges are
    /// BIT-IDENTICAL to the serial build (the B1 pin), and only the device
    /// readback's low bits move with the fold order (the B2 pin: host twin
    /// mirrors the tree op-for-op; radius-regression pins stay green).
    fn push_reduce(
        &mut self,
        plane: usize,
        err: usize,
        scale: Option<usize>,
        n: usize,
        berr: f32,
        split_contrib: f64,
    ) -> Result<(), Refusal> {
        self.split_eb = add_up(self.split_eb, split_contrib);
        let g_reduce = f32_up(gamma_ds_dot(n + 2))?;
        self.profile.push(format!(
            "reduce[n={n} scaled={}] g_reduce={g_reduce:.3e} berr={berr:.3e}",
            scale.is_some()
        ));
        let mut uniforms = pack_words(&[
            u32_of(n, "reduce width")?,
            u32_of(self.rows, "reduce rows")?,
            u32::from(scale.is_some()),
            0,
        ]);
        for v in [berr, g_reduce, 0.0f32, 0.0f32] {
            uniforms.extend_from_slice(&v.to_le_bytes());
        }
        let bindings = vec![
            LaneStepBinding {
                buffer: plane,
                read_only: true,
            },
            LaneStepBinding {
                buffer: err,
                read_only: true,
            },
            LaneStepBinding {
                buffer: scale.unwrap_or(self.dummy_scale),
                read_only: true,
            },
            LaneStepBinding {
                buffer: self.bias_acc,
                read_only: false,
            },
            LaneStepBinding {
                buffer: self.ebias_acc,
                read_only: false,
            },
        ];
        self.steps.push(Step {
            label: "mr_gpu_backward_bias_reduce_wg",
            entry: "bias_reduce_wg",
            shader: shader_reduce(),
            uniforms,
            bindings,
            workgroups: Self::workgroups_rows(self.rows)?,
        });
        // Serial-count envelopes deliberately KEPT for the tree (see the
        // charge argument above): host-side charges stay bit-identical.
        Self::widen(&mut self.eb_widen, 6 * n + 16);
        self.flush_ops_eb += FLUSH_OPS_PER_TAP * (n as f64 + 1.0);
        Ok(())
    }

    /// B5: one `collect_abs` dispatch over this relu's unstable positions,
    /// reading the PRE-transform frontier. HEURISTIC-ONLY — it charges
    /// NOTHING into `e_widen`/`eb_widen`/the flush or split floors, and it
    /// must not, because nothing verdict-bearing reads `PassOut::coll`
    /// (`engine.rs:600-611` consumers: `trunk_shortlist` bab.rs:3781-3806 and
    /// the tier-0 pool bab.rs:3703 — both rankers). The kernel header
    /// restates this; the ds accumulation exists so shortlist RANKINGS flip
    /// only on true ties, not to make the number certifiable.
    fn push_collect(
        &mut self,
        frontier: Frontier,
        layer: usize,
        unst: &[usize],
    ) -> Result<(), Refusal> {
        let positions = unst.len();
        let mut idx = Vec::with_capacity(positions);
        for &j in unst {
            if j >= frontier.width {
                return Err(Refusal::Unmappable("collect index outside the plane"));
            }
            idx.push(u32_of(j, "collect index")?);
        }
        let idx_buf = self.add_buffer(Init::Data(pack_words(&idx)))?;
        let out = self.add_buffer(Init::Zeroed(positions as u64 * 8))?;
        let uniforms = pack_words(&[
            u32_of(positions, "collect positions")?,
            u32_of(self.rows, "collect rows")?,
            0,
            0,
        ]);
        self.steps.push(Step {
            label: "mr_gpu_backward_collect_abs",
            entry: "collect_abs",
            shader: shader_collect(),
            uniforms,
            bindings: vec![
                LaneStepBinding {
                    buffer: idx_buf,
                    read_only: true,
                },
                LaneStepBinding {
                    buffer: frontier.coeff,
                    read_only: true,
                },
                LaneStepBinding {
                    buffer: out,
                    read_only: false,
                },
            ],
            workgroups: Self::workgroups_1d(positions)?,
        });
        self.colls.push((CollectOut { layer, positions }, out));
        Ok(())
    }

    fn apply_linear(
        &mut self,
        frontier: Frontier,
        bias_present: bool,
        out_features: usize,
        in_features: usize,
    ) -> Result<Frontier, Refusal> {
        // The lane's only Linear is the head gemm, and it is the FIRST layer
        // of the backward plan (gpu_seam::build_plan_ext constructs it so).
        // VALUES come from the engine's own f64 head weights (`net.gemm1()` —
        // the same source the plan builder downcast from), carried as
        // exact-split ds pairs; the plan layer contributes only its shape,
        // cross-checked here fail-closed.
        if self.head_seen || frontier.width != out_features {
            return Err(Refusal::Unmappable("Linear outside the head position"));
        }
        let (w1, b1, (n_y, n_h)) = self.eng.net.gemm1();
        if n_y != out_features || n_h != in_features || w1.len() != n_y * n_h || b1.len() != n_y {
            return Err(Refusal::Unmappable("head gemm f64/plan shape drift"));
        }
        self.head_seen = true;

        // B1: the head statics (ds weight ball, ds-pair planes + charges,
        // split-floor column norm) come from the PlanStatic — computed once
        // by the SAME helper calls this arm used to make per pass, so every
        // downstream composition below is bit-identical to the uncached
        // build (test-pinned). Byte-length cross-checks stay fail-closed.
        let hs = &self.statics.head;
        if hs.weight.bytes.len() != n_y * n_h * 8 {
            return Err(Refusal::Unmappable("cached head plane length drift"));
        }
        let rel = hs.rel;
        self.rho_star_ds = self.rho_star_ds.max(rel);

        // Head bias fold: b += S @ b1 (CPU: axpy over seed rows), off the
        // PRE-gemm frontier (the seed plane), like every bias site. Split
        // floor: per row the fold moves `sum_j |b_j| * sabs_in` of plane
        // slack plus `sum_j r_bj * |s_j| <= excess_sum * amax_in` of the
        // bias pairs' own uncovered residual into the bias accumulators.
        if bias_present {
            let bs = hs
                .bias
                .as_ref()
                .ok_or(Refusal::Unmappable("cached head bias missing"))?;
            if bs.plane.bytes.len() != n_y * 8 {
                return Err(Refusal::Unmappable("cached head bias length drift"));
            }
            let scale = self.add_plane(&bs.plane)?;
            let contrib = add_up(
                mul_up(bs.mass, frontier.sabs),
                mul_up(bs.excess, frontier.amax),
            );
            self.push_reduce(
                frontier.coeff,
                frontier.err,
                Some(scale),
                n_y,
                bs.berr,
                contrib,
            )?;
        }

        let w_excess = hs.w_excess;
        let weight_buf = self.add_plane(&hs.weight)?;
        // Split-floor propagation through the gemm: per output element,
        // `sum_j |w_j| * sabs_in + sum_j r_j * |s_j|` bounded by the max
        // column pair-mass norm and the plane excess sum times amax_in.
        let a_eff = hs.a_eff;
        let out_amax = mul_up(a_eff, frontier.amax);
        let out_sabs = add_up(
            mul_up(a_eff, frontier.sabs),
            mul_up(w_excess, frontier.amax),
        );
        let out_coeff = self.add_buffer(Init::Zeroed((n_h * self.rows) as u64 * 8))?;
        let out_err = self.add_buffer(Init::Zeroed((n_h * self.rows) as u64 * 4))?;
        let gamma = gamma_ds_dot(n_y + 2);
        let g_head = f32_up(compose_charge(gamma, rel))?;
        self.profile.push(format!(
            "head[n_y={n_y}] g_head={g_head:.3e} (gamma={gamma:.3e} wball={rel:.3e})"
        ));
        let mut uniforms = pack_words(&[
            u32_of(n_y, "head n_y")?,
            u32_of(n_h, "head n_h")?,
            u32_of(self.rows, "head rows")?,
        ]);
        uniforms.extend_from_slice(&g_head.to_le_bytes());
        self.steps.push(Step {
            label: "mr_gpu_backward_head_gemm",
            entry: "head_gemm",
            shader: shader_head(),
            uniforms,
            bindings: vec![
                LaneStepBinding {
                    buffer: weight_buf,
                    read_only: true,
                },
                LaneStepBinding {
                    buffer: frontier.coeff,
                    read_only: true,
                },
                LaneStepBinding {
                    buffer: out_coeff,
                    read_only: false,
                },
                LaneStepBinding {
                    buffer: out_err,
                    read_only: false,
                },
            ],
            workgroups: Self::workgroups_1d(n_h * self.rows)?,
        });
        Self::widen(&mut self.e_widen, 2 * n_y + 16);
        self.flush_ops_e += FLUSH_OPS_PER_TAP * n_y as f64;
        Ok(Frontier {
            coeff: out_coeff,
            err: out_err,
            width: n_h,
            amax: out_amax,
            sabs: out_sabs,
        })
    }

    fn apply_activation(&mut self, frontier: Frontier, n: usize) -> Result<Frontier, Refusal> {
        // Gate provenance: the fold-order relu list of `build_plan_full`
        // pairs the k-th Activation with its root-gate layer (the invariant
        // `plan_relu_layers_track_the_emitted_activations` pins); the f64
        // gates come from the SAME dom-override-else-root selection the CPU
        // arm and `gpu_seam::Builder::activation` use. They ride to the
        // device as ds SPLITS (normal-band residual <= 2^-48 relative,
        // charged in-kernel; the f32-subnormal band's absolute residue is
        // charged HOST-side into the split-floor channel below), so the
        // device certifies the CPU lane's OWN f64 relaxation — the M2
        // coefficient-overlap precondition. The plan's downcast Activation
        // payload is deliberately NOT used for values.
        let li = *self
            .relus
            .get(self.next_relu)
            .ok_or(Refusal::Unmappable("more Activations than relu records"))?;
        self.next_relu += 1;
        let rec = self
            .eng
            .root
            .layers
            .get(li)
            .ok_or(Refusal::Unmappable("relu layer index out of range"))?;
        let (alpha, s, c) = match self.dom.and_then(|d| d.layers.get(&li)) {
            Some(gv) => (&gv.alpha[..], &gv.s[..], &gv.c[..]),
            None => (&rec.alpha[..], &rec.s[..], &rec.c[..]),
        };
        if frontier.width != n || rec.n != n || alpha.len() != n || s.len() != n || c.len() != n {
            return Err(Refusal::Unmappable("activation width"));
        }
        // B5 (heuristic-only): the per-unstable |coefficient| collector reads
        // the PRE-transform frontier, exactly where the CPU arm collects it
        // (engine.rs:600-611 — `sum_r |L[unst[pos], r]|`). Emitted BEFORE the
        // gate transform for the same reason the CPU arm collects first; both
        // steps only READ `frontier.coeff`, so the order is documentation,
        // not a dependency. NO certified charge rides this plane: nothing
        // verdict-bearing reads `PassOut::coll` (see the kernel header).
        if self.collect_unst_abs && !rec.unst.is_empty() {
            self.push_collect(frontier, li, &rec.unst)?;
        }
        // B4: this layer's per-(row, neuron) gate overrides, compressed into
        // (neuron, row_start, row_end, quad) records. VERDICT-BEARING — see
        // `exception_records` and the FIX A1 charge fold below.
        let exc_records = match self.exc.and_then(|e| e.by_layer.get(&li)) {
            Some(list) if !list.is_empty() => {
                Some(exception_records(list, self.dir, n, self.rows, c)?)
            }
            _ => None,
        };
        // The direction-baked gate ds-pair quads (design §5 kernel 2):
        //   LOWER: (alpha, s, icept for v>=0 = 0, icept for v<0 = c)
        //   UPPER: (s, alpha, icept for v>=0 = c, icept for v<0 = 0)
        // — the same sign rule as the CPU arms (engine.rs gate/bias loops;
        // re-derivation of review item 4). Two vec4s per neuron: hi quad at
        // 2j, lo quad at 2j + 1.
        let mut gate = Vec::with_capacity(n * 8);
        // Split-floor stats (verify-fix): the gates are split HOST-side, so
        // their subnormal-band residual excess (beyond the in-kernel
        // `2 * DS_UNIT * |o|` cover — [`DS_GATE_COVER`]) is charged here.
        // Slope excess rides the E channel (elementwise map: max over BOTH
        // sides covers the sign selection incl. the kink flip, the same rule
        // as the kernel's `m`/`mi`); intercept mass + excess feed the
        // partial reduce below.
        let mut slope_norm = 0.0f64;
        let mut slope_excess = 0.0f64;
        let mut icept_mass = 0.0f64;
        let mut icept_excess = 0.0f64;
        for j in 0..n {
            let quad = match self.dir {
                LaneDir::Lower => [alpha[j], s[j], 0.0, c[j]],
                LaneDir::Upper => [s[j], alpha[j], c[j], 0.0],
            };
            let mut hi = [0.0f32; 4];
            let mut lo = [0.0f32; 4];
            let mut mass = [0.0f64; 4];
            let mut exc = [0.0f64; 4];
            for (slot, &v) in quad.iter().enumerate() {
                let (h, l) = ds_split_f64(v)?;
                hi[slot] = h;
                lo[slot] = l;
                mass[slot] = next_up(f64::from(h).abs() + f64::from(l).abs());
                exc[slot] = ds_split_excess(v, h, l, DS_GATE_COVER);
            }
            slope_norm = slope_norm.max(mass[0]).max(mass[1]);
            slope_excess = add_up(slope_excess, exc[0].max(exc[1]));
            icept_mass = add_up(icept_mass, mass[2].max(mass[3]));
            icept_excess = add_up(icept_excess, exc[2].max(exc[3]));
            gate.extend_from_slice(&hi);
            gate.extend_from_slice(&lo);
        }
        // ---------------------------------------------------------------
        // FIX A1 (SOUNDNESS, blocking for the B4 flip): fold the EXCEPTION
        // override magnitudes into this layer's HOST split-floor stats.
        //
        // The stats above are computed from the BASE gate quads only, and
        // they feed `a_eff` -> `Frontier::amax`/`sabs`, which multiply every
        // DOWNSTREAM split-excess charge (`w_excess * amax`,
        // `icept_mass * sabs + icept_excess * amax`). An override like the
        // batched-children pass's `(a2, s2, c2) = (1, 1, 0)` has slope mass
        // 1.0; on a layer whose base `slope_norm < 1`, the EXCEPTION ROW's
        // true coefficients then exceed the `amax` the host propagates, and
        // every downstream split-floor charge under-counts on exactly the
        // rows whose bounds get banked as child bounds (bab.rs:1612/1555).
        // The magnitude is split-floor-small (identically 0.0 in the normal
        // band) but this is the "charged-not-assumed-away" class that
        // refuted impl-v5, so it is CHARGED.
        //
        // The fold is the SAME per-quad reduction as the base loop, applied
        // to each record's direction-baked quad: `slope_norm` takes a MAX
        // (the plane map is elementwise — one neuron's slope per element),
        // the intercept channel ADDS (the reduce folds every neuron's
        // partial into the bias, and an exception row's partial can be as
        // large as `|c2|` where the base row's was `|c_j|`; adding both is
        // the conservative per-row bound `sum_j max(|c_j|, |c2_j|)`).
        // Excesses add, matching the base loop's own (loosening) convention.
        if let Some(records) = exc_records.as_ref() {
            fold_exception_charges(
                records,
                &mut slope_norm,
                &mut slope_excess,
                &mut icept_mass,
                &mut icept_excess,
            );
        }
        let gate_buf = self.add_buffer(Init::Data(bytes_of_f32s(&gate)))?;
        let total = n * self.rows;
        // FIX A5, stated: the widest plane decides the dispatch, so the pass
        // is admissible only while `width * R <= 65535 * 256`.
        dispatch_thread_cap(n, self.rows)?;
        let out_coeff = self.add_buffer(Init::Zeroed(total as u64 * 8))?;
        let out_err = self.add_buffer(Init::Zeroed(total as u64 * 4))?;
        let bias_partial = self.add_buffer(Init::Zeroed(total as u64 * 8))?;
        let partial_err = self.add_buffer(Init::Zeroed(total as u64 * 4))?;
        let n_exc = exc_records.as_ref().map_or(0, Vec::len);
        let uniforms = pack_words(&[
            u32_of(n, "gate neurons")?,
            u32_of(self.rows, "gate rows")?,
            u32_of(n_exc, "gate exception records")?,
            0,
        ]);
        let mut bindings = vec![
            LaneStepBinding {
                buffer: gate_buf,
                read_only: true,
            },
            LaneStepBinding {
                buffer: frontier.coeff,
                read_only: true,
            },
            LaneStepBinding {
                buffer: out_coeff,
                read_only: false,
            },
            LaneStepBinding {
                buffer: frontier.err,
                read_only: true,
            },
            LaneStepBinding {
                buffer: out_err,
                read_only: false,
            },
            LaneStepBinding {
                buffer: bias_partial,
                read_only: false,
            },
            LaneStepBinding {
                buffer: partial_err,
                read_only: false,
            },
        ];
        // With NO records the step is the UNCHANGED `gate_transform` entry,
        // byte-for-byte the pre-B4 dispatch (same label, same 7 bindings,
        // same uniform bytes — the third word is 0 either way): a pass
        // without exceptions is bit-identical to impl-v7 by construction.
        let (label, entry) = if n_exc == 0 {
            ("mr_gpu_backward_gate_transform", "gate_transform")
        } else {
            let bytes = exception_bytes(exc_records.as_ref().expect("n_exc > 0"));
            let exc_buf = self.add_buffer(Init::Data(bytes))?;
            bindings.push(LaneStepBinding {
                buffer: exc_buf,
                read_only: true,
            });
            ("mr_gpu_backward_gate_transform_exc", "gate_transform_exc")
        };
        self.steps.push(Step {
            label,
            entry,
            shader: shader_gate(),
            uniforms,
            bindings,
            workgroups: Self::workgroups_1d(total)?,
        });
        Self::widen(&mut self.e_widen, 16);
        self.flush_ops_e += FLUSH_OPS_PER_TAP;
        // Fold the intercept partials into the running bias. The partials
        // already carry the intercept (and their error plane the E*|c| +
        // DS_UNIT charges), so no scale and no berr here. Split floor: per
        // row the fold moves `sum_j |icept_j| * sabs_in + sum_j r_ij * |v_j|`
        // of uncovered slack into the bias accumulators.
        let contrib = add_up(
            mul_up(icept_mass, frontier.sabs),
            mul_up(icept_excess, frontier.amax),
        );
        self.push_reduce(bias_partial, partial_err, None, n, 0.0, contrib)?;
        let a_eff = norm_inflate(slope_norm, 1);
        Ok(Frontier {
            coeff: out_coeff,
            err: out_err,
            width: n,
            amax: mul_up(a_eff, frontier.amax),
            sabs: add_up(
                mul_up(a_eff, frontier.sabs),
                mul_up(slope_excess, frontier.amax),
            ),
        })
    }

    fn apply_conv(
        &mut self,
        frontier: Frontier,
        layer: &GpuCrownLayer,
    ) -> Result<Frontier, Refusal> {
        let GpuCrownLayer::Conv2d {
            weight_col,
            bias_expanded,
            out_channels,
            in_channels,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            out_h,
            out_w,
            in_h,
            in_w,
            cert_err: _,
        } = layer
        else {
            return Err(Refusal::Unmappable("not a conv layer"));
        };
        let n_out = out_channels * out_h * out_w;
        let n_in_t = in_channels * in_h * in_w;
        if frontier.width != n_out {
            return Err(Refusal::Unmappable("conv output width"));
        }
        if weight_col.len() != out_channels * in_channels * kernel_h * kernel_w {
            return Err(Refusal::Unmappable("conv weight length"));
        }
        if *stride_h == 0 || *stride_w == 0 {
            return Err(Refusal::Unmappable("conv zero stride"));
        }

        // PARAMETER provenance (the ds twin of the gate rule): the k-th
        // emitted Conv2d pairs with `build_plan_ext`'s k-th `conv_ops` entry —
        // the SAME traversal emitted both, so they cannot misalign; the shape
        // cross-check below is the fail-closed witness. Values ride as
        // exact-split ds pairs of the engine's f64 `wmat`/`bias`; the plan's
        // downcast payload is deliberately NOT used for values (it stays the
        // structural skeleton + the old seam's contract).
        let oi = *self
            .convs
            .get(self.next_conv)
            .ok_or(Refusal::Unmappable("more Conv2d layers than conv records"))?;
        self.next_conv += 1;
        let Some(TwinOp::Conv(cv)) = self.eng.net.ops.get(oi) else {
            return Err(Refusal::Unmappable("conv record is not a Conv op"));
        };
        if cv.kernel != (*out_channels, *in_channels, *kernel_h, *kernel_w)
            || cv.oshape != (*out_channels, *out_h, *out_w)
            || cv.ishape != (*in_channels, *in_h, *in_w)
            || cv.stride != (*stride_h, *stride_w)
            || (cv.pads.0, cv.pads.1) != (*pad_h, *pad_w)
            || cv.wmat.len() != weight_col.len()
            || cv.bias.len() != *out_channels
            || cv.bias_err.len() != *out_channels
        {
            return Err(Refusal::Unmappable("conv f64/plan shape drift"));
        }
        // B1: the conv statics come from the PlanStatic (same helper calls,
        // computed once — see apply_linear). Fold-order alignment is
        // fail-closed: the k-th emitted Conv2d must consume the k-th cached
        // ConvStatic, whose op_index must be the k-th provenance entry.
        let cs = self
            .statics
            .convs
            .get(self.next_conv - 1)
            .ok_or(Refusal::Unmappable(
                "more Conv2d layers than cached statics",
            ))?;
        if cs.op_index != oi {
            return Err(Refusal::Unmappable("cached conv provenance drift"));
        }
        if cs.weight.bytes.len() != cv.wmat.len() * 8 {
            return Err(Refusal::Unmappable("cached conv plane length drift"));
        }
        let rel = cs.rel;
        self.rho_star_ds = self.rho_star_ds.max(rel);

        // Conv bias fold: b += Lo @ bias (per-channel f64, expanded to n_out
        // ds pairs in the SAME channel-major order the plan expands), with the
        // CPU conv-bias arm's charges — |bias|*E in-kernel, (|L|+E)*berr, and
        // the fold envelope via the reduce's own g_reduce.
        if bias_expanded.is_some() {
            let p = out_h * out_w;
            let bs = cs
                .bias
                .as_ref()
                .ok_or(Refusal::Unmappable("cached conv bias missing"))?;
            if bs.plane.bytes.len() != n_out * 8 {
                return Err(Refusal::Unmappable("cached conv bias length drift"));
            }
            #[allow(clippy::cast_precision_loss)]
            let pf = p as f64;
            let contrib = add_up(
                mul_up(mul_up(bs.mass, pf), frontier.sabs),
                mul_up(mul_up(bs.excess, pf), frontier.amax),
            );
            let scale = self.add_plane(&bs.plane)?;
            self.push_reduce(
                frontier.coeff,
                frontier.err,
                Some(scale),
                n_out,
                bs.berr,
                contrib,
            )?;
        }

        // Conservative per-input-position tap bound (>= the real k_bwd).
        let k_bwd = out_channels * kernel_h * kernel_w;
        let gamma = gamma_ds_dot(k_bwd + 2);
        let g_src = f32_up(compose_charge(gamma, rel))?;
        self.profile.push(format!(
            "conv[k_bwd={k_bwd}] g_src={g_src:.3e} (gamma={gamma:.3e} wball={rel:.3e})"
        ));
        let w_excess = cs.w_excess;
        let weight_buf = self.add_plane(&cs.weight)?;
        // Split-floor propagation: every input position gathers each of its
        // channel's (co, kh, kw) weights at most once, so the per-ci full
        // kernel mass sum bounds the backward operator norm.
        let a_eff = cs.a_eff;
        let total = n_in_t * self.rows;
        let out_coeff = self.add_buffer(Init::Zeroed(total as u64 * 8))?;
        let out_err = self.add_buffer(Init::Zeroed(total as u64 * 4))?;
        let residual = self.add_buffer(Init::Zeroed(total as u64 * 4))?;
        let mut uniforms = pack_words(&[
            u32_of(*in_channels, "conv in_channels")?,
            u32_of(*in_h, "conv in_h")?,
            u32_of(*in_w, "conv in_w")?,
            u32_of(*out_channels, "conv out_channels")?,
            u32_of(*out_h, "conv out_h")?,
            u32_of(*out_w, "conv out_w")?,
            u32_of(*kernel_h, "conv kernel_h")?,
            u32_of(*kernel_w, "conv kernel_w")?,
            u32_of(*stride_h, "conv stride_h")?,
            u32_of(*stride_w, "conv stride_w")?,
            u32_of(*pad_h, "conv pad_h")?,
            u32_of(*pad_w, "conv pad_w")?,
            u32_of(self.rows, "conv rows")?,
            0,
            0,
        ]);
        uniforms.extend_from_slice(&g_src.to_le_bytes());
        self.steps.push(Step {
            label: "mr_gpu_backward_conv_backward",
            entry: "conv_backward",
            shader: shader_conv(),
            uniforms,
            bindings: vec![
                LaneStepBinding {
                    buffer: weight_buf,
                    read_only: true,
                },
                LaneStepBinding {
                    buffer: frontier.coeff,
                    read_only: true,
                },
                LaneStepBinding {
                    buffer: out_coeff,
                    read_only: false,
                },
                LaneStepBinding {
                    buffer: frontier.err,
                    read_only: true,
                },
                LaneStepBinding {
                    buffer: out_err,
                    read_only: false,
                },
                LaneStepBinding {
                    buffer: residual,
                    read_only: false,
                },
            ],
            workgroups: Self::workgroups_1d(total)?,
        });
        Self::widen(&mut self.e_widen, 4 * k_bwd + 16);
        self.flush_ops_e += FLUSH_OPS_PER_TAP * k_bwd as f64;
        Ok(Frontier {
            coeff: out_coeff,
            err: out_err,
            width: n_in_t,
            amax: mul_up(a_eff, frontier.amax),
            sabs: add_up(
                mul_up(a_eff, frontier.sabs),
                mul_up(w_excess, frontier.amax),
            ),
        })
    }

    fn apply_layer(
        &mut self,
        frontier: Frontier,
        layer: &GpuCrownLayer,
    ) -> Result<Frontier, Refusal> {
        match layer {
            GpuCrownLayer::Linear {
                weight: _,
                bias,
                out_features,
                in_features,
                cert_err: _,
            } => self.apply_linear(frontier, bias.is_some(), *out_features, *in_features),
            GpuCrownLayer::Activation { num_neurons, .. } => {
                self.apply_activation(frontier, *num_neurons)
            }
            GpuCrownLayer::Conv2d { .. } => self.apply_conv(frontier, layer),
            // Fail-closed: the plan builder never emits anything else for this
            // lane; an unknown layer must refuse, not be skipped.
            _ => Err(Refusal::Unmappable("unsupported plan layer")),
        }
    }

    fn apply_chain(
        &mut self,
        mut frontier: Frontier,
        layers: &[GpuCrownLayer],
    ) -> Result<Frontier, Refusal> {
        for layer in layers {
            frontier = self.apply_layer(frontier, layer)?;
        }
        Ok(frontier)
    }

    /// `A_z = A_x + A_y` at a residual common ancestor (the CPU `merge_into`
    /// with the ds unit).
    fn merge(&mut self, x: Frontier, y: Frontier) -> Result<Frontier, Refusal> {
        if x.width != y.width {
            return Err(Refusal::Unmappable("residual merge width"));
        }
        let total = x.width * self.rows;
        let out_coeff = self.add_buffer(Init::Zeroed(total as u64 * 8))?;
        let out_err = self.add_buffer(Init::Zeroed(total as u64 * 4))?;
        let uniforms = pack_words(&[u32_of(total, "merge total")?, 0, 0, 0]);
        self.steps.push(Step {
            label: "mr_gpu_backward_merge_add",
            entry: "merge_add",
            shader: shader_merge(),
            uniforms,
            bindings: vec![
                LaneStepBinding {
                    buffer: x.coeff,
                    read_only: true,
                },
                LaneStepBinding {
                    buffer: y.coeff,
                    read_only: true,
                },
                LaneStepBinding {
                    buffer: out_coeff,
                    read_only: false,
                },
                LaneStepBinding {
                    buffer: x.err,
                    read_only: true,
                },
                LaneStepBinding {
                    buffer: y.err,
                    read_only: true,
                },
                LaneStepBinding {
                    buffer: out_err,
                    read_only: false,
                },
            ],
            workgroups: Self::workgroups_1d(total)?,
        });
        Self::widen(&mut self.e_widen, 8);
        self.flush_ops_e += FLUSH_OPS_PER_TAP;
        Ok(Frontier {
            coeff: out_coeff,
            err: out_err,
            width: x.width,
            amax: norm_inflate(next_up(x.amax + y.amax), 1),
            sabs: add_up(x.sabs, y.sabs),
        })
    }

    fn apply_segment(
        &mut self,
        frontier: Frontier,
        segment: &GpuResnetSegment,
    ) -> Result<Frontier, Refusal> {
        match segment {
            GpuResnetSegment::Chain(layers) => self.apply_chain(frontier, layers),
            // out = z + F(z): A_z = A_out (identity, untouched buffers) +
            // backward_F(A_out). No buffer is ever written in place, so the
            // identity contribution needs no copy.
            GpuResnetSegment::Residual(branch) => {
                let through = self.apply_chain(frontier, branch)?;
                self.merge(through, frontier)
            }
            // out = F(z) + P(z): both branches read the SAME frontier.
            GpuResnetSegment::ResidualProj(f_branch, p_branch) => {
                let f_out = self.apply_chain(frontier, f_branch)?;
                let p_out = self.apply_chain(frontier, p_branch)?;
                self.merge(f_out, p_out)
            }
        }
    }
}

/// One B4 gate-override record: a (neuron, `[row_start, row_end)`) block and
/// the DIRECTION-BAKED ds-pair quad that replaces the base gate there.
///
/// `mass`/`excess` are the same per-slot split-floor statistics the base gate
/// loop computes ([`ds_split_excess`] against [`DS_GATE_COVER`]); FIX A1
/// folds them into the layer's host stats.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(super) struct ExcRecord {
    pub(super) neuron: usize,
    pub(super) row_start: usize,
    pub(super) row_end: usize,
    /// The SOURCE direction-baked f64 quad this record was built from — the
    /// run-length merge key (see [`ExcRecord::src_bits`]). Not uploaded: the
    /// device consumes `hi`/`lo`.
    pub(super) src: [f64; 4],
    pub(super) hi: [f32; 4],
    pub(super) lo: [f32; 4],
    pub(super) mass: [f64; 4],
    pub(super) excess: [f64; 4],
}

/// FIX A1 — fold the exception overrides into a relu layer's HOST
/// split-floor statistics. Pure, so the four channels are testable without a
/// plan or a device.
///
/// The reduction is the SAME one the base gate loop performs per neuron:
/// `slope_norm` is a MAX (the coefficient map is elementwise — one slope per
/// element, so the plane's operator norm is the largest slope any element
/// can apply, base or override), while the intercept channel ADDS (the bias
/// reduce folds EVERY neuron's intercept partial into the same accumulator,
/// so per row the bound is `sum_j max(|c_j|, |c2_j|)` and adding both terms
/// dominates it). Excesses add on both channels, matching the base loop's
/// own conservative convention.
pub(super) fn fold_exception_charges(
    records: &[ExcRecord],
    slope_norm: &mut f64,
    slope_excess: &mut f64,
    icept_mass: &mut f64,
    icept_excess: &mut f64,
) {
    for r in records {
        *slope_norm = slope_norm.max(r.mass[0]).max(r.mass[1]);
        *slope_excess = add_up(*slope_excess, r.excess[0].max(r.excess[1]));
        *icept_mass = add_up(*icept_mass, r.mass[2].max(r.mass[3]));
        *icept_excess = add_up(*icept_excess, r.excess[2].max(r.excess[3]));
    }
}

/// Compress a layer's per-row [`super::super::engine::Exc`] list into
/// (neuron, contiguous row block, quad) records, direction-baked exactly like
/// the base gate quads.
///
/// The batched-children pass writes `nf` consecutive rows per candidate block
/// with an IDENTICAL override (bab.rs:4114-4146), so this collapses to
/// `2 * k_trunk` records at the shipped preset. Fail-closed on everything the
/// kernel could not express: out-of-range neuron/row, a duplicated
/// `(neuron, row)` ELEMENT (two records matching one element would let the
/// LAST one win on the device while the CPU arm applies BOTH bias deltas),
/// and more records than [`MAX_EXC_RECORDS`].
///
/// FIX D6 (hardening session, impl-v9): the duplicate check is per
/// `(neuron, row)`, NOT per row. Two exceptions on the SAME pass row at
/// DIFFERENT neurons are legal and handled identically on both arms — the CPU
/// exception loop adds each `bpart` into `b[x.row]` independently
/// (engine.rs:761-787) and the kernel matches `j == neuron && row in [r0, r1)`
/// per thread, so each neuron's thread applies its own record
/// (`gate_transform.wgsl:212`). The pre-v9 check compared only the ROW of
/// ADJACENT entries in a list sorted by `(neuron, row)`, so a legal
/// multi-neuron row pair could refuse to CPU — fail-closed and inert at the
/// shipped `score_candidates` shape (one unique block per row), but it would
/// have silently inerted any future multi-neuron exception caller.
///
/// MERGE KEY (FIX D1, hardening session, impl-v9): run-length merging
/// compares the SOURCE f64 quads on BOTH sides ([`ExcRecord::src_bits`]).
/// The pre-v9 code compared the original quad's bits against the RECONSTRUCTED
/// ds-pair value (`f64::from(hi) + f64::from(lo)`), which are equal ONLY for
/// f32-exact quads — `recon(1/3) - 1/3 = 2.78e-16`. Consequence: for any
/// FRACTIONAL override (B7 fractional-alpha, lane-alpha, re-bake payloads) the
/// compression NEVER fired, the record count became `2 * k_trunk * nf` instead
/// of `2 * k_trunk`, and at `nf > 1` it crossed [`MAX_EXC_RECORDS`] into a
/// silent `Unmappable` -> CPU: a seam that reads as armed and is inert.
/// SOURCE bits rather than pair bits is the load-bearing choice — the per-slot
/// `mass`/`excess` the FIX A1 fold charges are functions of the SOURCE value
/// (`ds_split_excess(v, hi, lo, ..)`), so two distinct sources that happen to
/// split to the same pair carry DIFFERENT split-floor charges and must not
/// merge. Identical source bits imply identical pair AND identical charges.
///
/// DERIVATION OBLIGATION (i) — THE INTERCEPT SIGN SELECT. The CPU exception
/// arm (engine.rs:761-787) recomputes the element from the PRE-transform
/// original `v`: lower lane `l = vp*a2 + vn*s2` with bias part
/// `vn*(c2 - c[neuron])`, upper lane `l = vp*s2 + vn*a2` with `vp*(c2 - c)`.
/// The `- c[neuron]` is a DELTA because the CPU's bias loop already folded
/// the BASE intercept before the transform. On this lane the intercept
/// partial is produced by the gate kernel itself (`p = ds_mul(v, icept)`),
/// so the override must carry the ABSOLUTE `c2`, not the delta — the base
/// `c[neuron]` never entered the partial for these rows. Baking the same
/// sign rule as the base quad
/// (LOWER `(a2, s2, 0, c2)`, UPPER `(s2, a2, c2, 0)`) reproduces the CPU
/// element exactly: for `v < 0` the lower lane selects slope `s2` and
/// intercept `c2` giving `vn*s2` and `vn*c2`; for `v >= 0` it selects `a2`
/// and `0`. `c` is taken here only to CHECK that contract (an override that
/// does not change the intercept must still publish `c2`, so the record is
/// built from `x.c2` unconditionally) — see
/// `exception_quads_are_absolute_not_deltas`.
///
/// DERIVATION OBLIGATION (ii) — THE `E*mi` KINK TERM. The gate kernel charges
/// the upstream error carry against `mi = max(|icept+|, |icept-|)` and the
/// slope contraction against `m = max(|slope+|, |slope-|)`, i.e. the max over
/// BOTH sides, never the selected one: the STORED `v`'s sign can differ from
/// the exact one's across the kink (`|v_exact| <= E` there), and the true map
/// `v -> v*icept(sign v)` has global Lipschitz constant `mi` (design log,
/// "second re-derivation catch"). On an exception row those maxima must be
/// taken over the OVERRIDDEN quad — charging the base quad's `mi` while the
/// device applies `c2` would under-charge whenever `|c2| > |c|`, the
/// false-VERIFY direction. The kernel therefore substitutes the whole quad
/// (`ghi`/`glo`) before computing `m` and `mi`, so both derived maxima follow
/// the override automatically; `gate_transform_exc_charges_the_overridden_mi`
/// pins the WGSL, and the host mirror is the FIX A1 fold in
/// `apply_activation`.
pub(super) fn exception_records(
    list: &[super::super::engine::Exc],
    dir: LaneDir,
    n: usize,
    rows: usize,
    c: &[f64],
) -> Result<Vec<ExcRecord>, Refusal> {
    let _ = c; // documented above: the record is ABSOLUTE, never a delta
    let mut flat: Vec<(usize, usize, [f64; 4])> = Vec::with_capacity(list.len());
    for x in list {
        if x.neuron >= n || x.row >= rows {
            return Err(Refusal::Unmappable("exception outside the plane"));
        }
        let quad = match dir {
            LaneDir::Lower => [x.a2, x.s2, 0.0, x.c2],
            LaneDir::Upper => [x.s2, x.a2, x.c2, 0.0],
        };
        flat.push((x.neuron, x.row, quad));
    }
    flat.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    for w in flat.windows(2) {
        // FIX D6: the same ELEMENT twice, not merely the same row. The sort
        // is by `(neuron, row)`, so a duplicated element is always adjacent.
        if w[0].0 == w[1].0 && w[0].1 == w[1].1 {
            return Err(Refusal::Unmappable(
                "two exceptions on one (neuron, row) element",
            ));
        }
    }
    let key = |q: &[f64; 4]| q.map(f64::to_bits);
    let mut out: Vec<ExcRecord> = Vec::new();
    for (neuron, row, quad) in flat {
        if let Some(last) = out.last_mut() {
            // FIX D1: SOURCE bits on both sides (see the merge-key note above).
            if last.neuron == neuron && last.row_end == row && key(&quad) == last.src_bits() {
                last.row_end = row + 1;
                continue;
            }
        }
        let mut hi = [0.0f32; 4];
        let mut lo = [0.0f32; 4];
        let mut mass = [0.0f64; 4];
        let mut excess = [0.0f64; 4];
        for (slot, &v) in quad.iter().enumerate() {
            let (h, l) = ds_split_f64(v)?;
            hi[slot] = h;
            lo[slot] = l;
            mass[slot] = next_up(f64::from(h).abs() + f64::from(l).abs());
            excess[slot] = ds_split_excess(v, h, l, DS_GATE_COVER);
        }
        out.push(ExcRecord {
            neuron,
            row_start: row,
            row_end: row + 1,
            src: quad,
            hi,
            lo,
            mass,
            excess,
        });
        if out.len() > MAX_EXC_RECORDS {
            return Err(Refusal::Unmappable("too many exception records"));
        }
    }
    Ok(out)
}

impl ExcRecord {
    /// Bit key of the SOURCE quad (the run-length merge comparator, FIX D1):
    /// the f64 the record was built from, per slot, compared by bits so
    /// `-0.0` never merges with `0.0` and a NaN payload never merges with
    /// itself. This is the ONLY admissible key — see the merge-key note on
    /// [`exception_records`] for why the reconstructed pair value is not
    /// (it is inexact for fractional quads, and it does not determine the
    /// per-slot split-floor charges the FIX A1 fold reads).
    fn src_bits(&self) -> [u64; 4] {
        self.src.map(f64::to_bits)
    }

    /// Bit key of the RECONSTRUCTED ds-pair value per slot — what the device
    /// actually applies. Diagnostic only (the D1 pin prints the gap against
    /// [`Self::src_bits`]); never the merge key.
    pub(super) fn pair_value_bits(&self) -> [u64; 4] {
        let mut out = [0u64; 4];
        for (slot, o) in out.iter_mut().enumerate() {
            *o = (f64::from(self.hi[slot]) + f64::from(self.lo[slot])).to_bits();
        }
        out
    }
}

/// Encode records for `kernels/gate_transform.wgsl`'s exception table: THREE
/// `vec4<f32>` per record — `(neuron, row_start, row_end, 0)` bit-cast from
/// u32, then the hi quad, then the lo quad. One buffer (not three) keeps the
/// exception entry at 8 storage bindings, the wgpu default per-stage limit:
/// exceeding it fails the bind group ASYNC and the dispatch silently no-ops
/// (the banked ">8 bindings" trap).
///
/// # THE HEADER RIDES AS BIT-CAST SUBNORMAL f32 — IT DEPENDS ON DenormPreserve
/// # (FIX D7, hardening session, impl-v9)
///
/// The kernel declares `exc: array<vec4<f32>>` and `bitcast<u32>`s the header
/// lanes back (`gate_transform.wgsl:95`, `:208-211`), so every header word
/// with `0 < w < 2^23` — i.e. EVERY realistic `neuron` / `row_start` /
/// `row_end` — is an f32 SUBNORMAL bit pattern. On an adapter that flushed
/// subnormals ON LOAD, all three would read back as 0 and every record would
/// target `(neuron 0, rows [0, 0))`: NO override applied on the device while
/// the CPU arm applies it, i.e. device relaxation != CPU relaxation on exactly
/// the rows whose bounds get banked as child bounds. That is a silent
/// divergence, not a refusal.
///
/// It cannot happen on an admitted pass, and the reason is NOT local to this
/// encoding: [`super::authority::DeviceParityProof::qualify`] REFUSES unless
/// the device's RESOLVED DenormPreserve word channel is ON
/// (`authority.rs:81-86`), and that proof is the only way to mint the
/// [`VerdictAuthority`] every transaction requires. The coupling is stated
/// here and at the kernel binding because it is otherwise invisible at both
/// ends, and it is pinned by
/// `tests::exception_header_words_are_subnormals_that_depend_on_denorm_preserve`.
/// If the header is ever moved to a `array<u32>` binding (which would remove
/// the dependency), delete that pin in the same commit — do not weaken it.
pub(super) fn exception_bytes(records: &[ExcRecord]) -> Vec<u8> {
    let mut out = Vec::with_capacity(records.len() * 48);
    for r in records {
        #[allow(clippy::cast_possible_truncation)]
        for w in [r.neuron as u32, r.row_start as u32, r.row_end as u32, 0u32] {
            // The kernel reads this slot as `vec4<f32>` and `bitcast<u32>`s
            // it back; `f32::to_le_bytes` is `to_bits().to_le_bytes()`, so
            // writing the u32 words directly IS the same 16 bytes.
            out.extend_from_slice(&w.to_le_bytes());
        }
        for v in r.hi {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for v in r.lo {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out
}

fn bytes_of_f32s(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 4);
    for v in vals {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Exact-split a whole f64 WEIGHT slice into an interleaved ds-pair byte
/// plane (`hi, lo` per element) — the weight upload form — plus the up-sum
/// of the per-element split-residual EXCESS over the guaranteed in-kernel
/// cover ([`DS_SPLIT_COVER`]). Normal-band residuals are charged through
/// [`DS_SPLIT_REL`]; the returned excess (0.0 unless a parameter sits in
/// the f32-subnormal band) rides the split-floor channel.
pub(super) fn ds_pair_plane(vals: &[f64]) -> Result<(Vec<u8>, f64), Refusal> {
    let mut out = Vec::with_capacity(vals.len() * 8);
    let mut excess = 0.0f64;
    for &v in vals {
        let (hi, lo) = ds_split_f64(v)?;
        out.extend_from_slice(&hi.to_le_bytes());
        out.extend_from_slice(&lo.to_le_bytes());
        excess = add_up(excess, ds_split_excess(v, hi, lo, DS_SPLIT_COVER));
    }
    Ok((out, excess))
}

/// [`ds_pair_plane`] for a SCALE (bias) slice, additionally returning the
/// up-sum of the pair masses (`sum_j |hi_j| + |lo_j|`) — the per-row factor
/// that moves the reduced plane's split slack into the bias channel.
pub(super) fn ds_scale_plane(vals: &[f64]) -> Result<(Vec<u8>, f64, f64), Refusal> {
    let (bytes, excess) = ds_pair_plane(vals)?;
    let mut mass = 0.0f64;
    for chunk in bytes.as_chunks::<8>().0 {
        let hi = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let lo = f32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
        mass = add_up(mass, next_up(f64::from(hi).abs() + f64::from(lo).abs()));
    }
    Ok((bytes, mass, excess))
}

/// Max over output columns `k` of the up-sum of pair masses down column `k`
/// of the head weight (`w[j * n_h + k]`, row-major `(n_y, n_h)`): the
/// backward operator norm the split-floor channel propagates through.
pub(super) fn head_col_mass_max(w: &[f64], n_y: usize, n_h: usize) -> Result<f64, Refusal> {
    let mut cols = vec![0.0f64; n_h];
    for j in 0..n_y {
        for (k, col) in cols.iter_mut().enumerate() {
            let (hi, lo) = ds_split_f64(w[j * n_h + k])?;
            *col = next_up(*col + next_up(f64::from(hi).abs() + f64::from(lo).abs()));
        }
    }
    Ok(cols.iter().fold(0.0f64, |m, &c| m.max(c)))
}

/// Max over input channels `ci` of the up-sum of pair masses over ALL of
/// channel `ci`'s kernel taps (`w[((co * ic + ci)] * khw ..`): each input
/// position gathers each of its channel's `(co, kh, kw)` weights at most
/// once (the conv-backward tap structure), so this bounds the backward
/// operator norm per input element.
pub(super) fn conv_ci_mass_max(
    w: &[f64],
    oc: usize,
    ic: usize,
    khw: usize,
) -> Result<f64, Refusal> {
    let mut per_ci = vec![0.0f64; ic];
    for co in 0..oc {
        for (ci, acc) in per_ci.iter_mut().enumerate() {
            for t in 0..khw {
                let (hi, lo) = ds_split_f64(w[(co * ic + ci) * khw + t])?;
                *acc = next_up(*acc + next_up(f64::from(hi).abs() + f64::from(lo).abs()));
            }
        }
    }
    Ok(per_ci.iter().fold(0.0f64, |m, &c| m.max(c)))
}

/// EXACT split residual `|v - (hi + lo)|`, computed as
/// `(v - f64(hi)) - f64(lo)` — each subtraction is exact in f64 in EVERY
/// band: `fl32(x)` is 0 or within a factor of two of `x` (round-to-nearest
/// never leaves `[x/2, 2x]` for a nonzero result), so each difference is
/// exact by Sterbenz-or-zero; in the normal band the 29-low-order-bits
/// representability argument (see [`DS_SPLIT_REL`]) reaches the same
/// conclusion. No rounding anywhere: the returned value IS the residual.
pub(super) fn ds_split_residual(v: f64, hi: f32, lo: f32) -> f64 {
    ((v - f64::from(hi)) - f64::from(lo)).abs()
}

/// The split residual's EXCESS over the guaranteed in-kernel relative cover
/// `cover * (|hi| + |lo|)`, directed so cover + excess dominates the true
/// residual: the cover base is rounded DOWN (and clamped nonnegative for a
/// zero-mass pair) and the difference UP. Identically `0.0` whenever the
/// relative cover holds — every normal-band parameter — and `<= ~2^-149`
/// absolute per element in the f32-subnormal band.
pub(super) fn ds_split_excess(v: f64, hi: f32, lo: f32, cover: f64) -> f64 {
    let r = ds_split_residual(v, hi, lo);
    if r == 0.0 {
        return 0.0;
    }
    let covered = next_down(cover * (f64::from(hi).abs() + f64::from(lo).abs())).max(0.0);
    if r > covered {
        next_up(r - covered)
    } else {
        0.0
    }
}

/// Exact-split an f64 gate/weight parameter into a ds pair: `hi = fl32(v)`,
/// `lo = fl32(v - hi)`. Both subtractions used to FORM and to MEASURE the
/// pair are exact in f64 (see [`ds_split_residual`]), so the pair's residual
/// is exactly `(v - hi) - lo`. In the NORMAL band it obeys
/// `|v - hi - lo| <= 2^-48 |v|` and is charged through [`DS_SPLIT_REL`]
/// (weights/biases) or the gate transform's `2 * DS_UNIT` term (gates); in
/// the f32-SUBNORMAL band that relative claim FAILS (component rounding is
/// ABSOLUTE there) and the
/// uncovered part is charged exactly through [`ds_split_excess`] into the
/// split-floor channel.
pub(super) fn ds_split_f64(v: f64) -> Result<(f32, f32), Refusal> {
    if !v.is_finite() {
        return Err(Refusal::Unmappable("non-finite gate parameter"));
    }
    #[allow(clippy::cast_possible_truncation)]
    let hi = v as f32;
    if !hi.is_finite() {
        return Err(Refusal::Unmappable("gate parameter overflows f32"));
    }
    #[allow(clippy::cast_possible_truncation)]
    let lo = (v - f64::from(hi)) as f32;
    Ok((hi, lo))
}

/// The seed as an exact ds plane: `hi = seed as f32`, `lo = 0`. The downcast
/// residual and the seed's own certified error never reach the device — they
/// are already concretized into `eb` by [`gpu_seam::seed_penalty`] (which the
/// caller ran first, so every value here is finite and f32-representable).
/// Also returns `max |hi|` — the split-floor channel's `amax` seed.
fn seed_plane_bytes(seed: &Seed, n_y: usize, rows: usize) -> Result<(Vec<u8>, f64), Refusal> {
    let src = seed
        .s
        .as_slice()
        .ok_or(Refusal::Unmappable("seed is not standard layout"))?;
    if src.len() != n_y * rows {
        return Err(Refusal::Rows);
    }
    let mut out = Vec::with_capacity(n_y * rows * 8);
    let mut amax = 0.0f64;
    for &v in src {
        #[allow(clippy::cast_possible_truncation)]
        let hi = v as f32;
        if !hi.is_finite() {
            return Err(Refusal::Unmappable("seed overflows f32"));
        }
        amax = amax.max(f64::from(hi).abs());
        out.extend_from_slice(&hi.to_le_bytes());
        out.extend_from_slice(&0.0f32.to_le_bytes());
    }
    Ok((out, amax))
}

/// Assemble the whole pass as one drain-safe program from the pass-invariant
/// [`PlanStatic`] (plan skeleton + fold-order provenance + weight/bias ds
/// planes and charges) plus the per-pass dynamic part (`eng`/`dom` gates,
/// the seed). The statics were computed by the SAME helper functions this
/// builder used to call per pass, so the built program is bit-identical to
/// the pre-B1 uncached build (test-pinned).
#[allow(clippy::too_many_arguments)]
pub(super) fn build_program<'p>(
    statics: &'p PlanStatic,
    eng: &'p BackwardEngine<'_>,
    dom: Option<&'p DomainGates>,
    seed: &Seed,
    dir: LaneDir,
    shape: PassShape<'p>,
    n_y: usize,
    n_in: usize,
    rows: usize,
) -> Result<BuiltProgram<'p>, Refusal> {
    let plan: &Plan = &statics.ext.plan;
    let relus: &[usize] = &statics.ext.relus;
    let conv_ops: &[usize] = &statics.ext.conv_ops;
    let mut b = Builder::new(rows, dir, eng, dom, shape, statics, relus, conv_ops);
    let (seed_bytes, seed_amax) = seed_plane_bytes(seed, n_y, rows)?;
    let seed_coeff = b.add_buffer(Init::Data(seed_bytes))?;
    let seed_err = b.add_buffer(Init::Zeroed((n_y * rows) as u64 * 4))?;
    let mut frontier = Frontier {
        coeff: seed_coeff,
        err: seed_err,
        width: n_y,
        // The stored seed plane IS what the lane certifies against (its
        // downcast residual is charged into eb at functional level by
        // seed_penalty), so it carries no split slack.
        amax: seed_amax,
        sabs: 0.0,
    };
    match plan {
        Plan::Chain(layers) => frontier = b.apply_chain(frontier, layers)?,
        Plan::Segments(segments) => {
            for segment in segments {
                frontier = b.apply_segment(frontier, segment)?;
            }
        }
    }
    if !b.head_seen || frontier.width != n_in {
        return Err(Refusal::Unmappable("plan does not end at the input"));
    }
    // Fold-order completeness: every emitted Activation consumed exactly one
    // relu record, every Conv2d exactly one conv record (the same invariant
    // `retarget_plan` enforces for gates).
    if b.next_relu != relus.len() {
        return Err(Refusal::Unmappable("fewer Activations than relu records"));
    }
    if b.next_conv != conv_ops.len() {
        return Err(Refusal::Unmappable("fewer Conv2d layers than conv records"));
    }
    let flush_e = next_up(b.flush_ops_e * FLUSH_FLOOR);
    let flush_eb = next_up(b.flush_ops_eb * FLUSH_FLOOR);
    let split_e = frontier.sabs;
    let split_eb = b.split_eb;
    // FIX D2: the whole program is assembled, so the final arena total is
    // known; re-check it once here so the budget is enforced on the SUM even
    // if a future step is added outside `add_buffer`.
    arena_byte_cap(b.arena_bytes)?;
    #[allow(clippy::cast_precision_loss)]
    let arena_mb = b.arena_bytes as f64 / (1024.0 * 1024.0);
    let mut profile = b.profile;
    profile.push(format!(
        "egress e_widen={:.6} eb_widen={:.6} flush_e={:.3e} flush_eb={:.3e} \
         split_e={split_e:.3e} split_eb={split_eb:.3e} rho_star_ds={:.3e} \
         arena_mb={arena_mb:.1}",
        b.e_widen, b.eb_widen, flush_e, flush_eb, b.rho_star_ds
    ));
    let mut readbacks = vec![frontier.coeff, frontier.err, b.bias_acc, b.ebias_acc];
    let mut colls = Vec::with_capacity(b.colls.len());
    for (desc, buf) in &b.colls {
        colls.push(*desc);
        readbacks.push(*buf);
    }
    Ok(BuiltProgram {
        readbacks,
        colls,
        collect_unst_abs: shape.collect_unst_abs,
        amax: frontier.amax,
        buffers: b.buffers,
        steps: b.steps,
        e_widen: b.e_widen,
        eb_widen: b.eb_widen,
        flush_e,
        flush_eb,
        split_e,
        split_eb,
        rho_star_ds: b.rho_star_ds,
        profile,
        arena_bytes: b.arena_bytes,
        n_in,
        rows,
    })
}

// ---------------------------------------------------------------------------
// Dispatch + egress
// ---------------------------------------------------------------------------

fn f32s_from_bytes(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Dispatch a built program on the lane device (ONE submit, unconditional
/// drain inside the executor; `profile=true` is the caller-guaranteed
/// non-verdict-bearing one-step-per-submit mode) and return the four
/// readbacks as f32 streams plus the executor's timing telemetry.
pub(super) fn dispatch_program(
    gpu: &dyn MarginRowLaneGpu,
    built: &BuiltProgram<'_>,
    profile: bool,
    timing: &mut LaneProgramTiming,
) -> Result<Vec<Vec<f32>>, Refusal> {
    let buffers: Vec<LaneArenaBuffer<'_>> = built
        .buffers
        .iter()
        .map(|init| match init {
            Init::Data(bytes) => LaneArenaBuffer::Data(bytes),
            Init::DataRef(bytes) => LaneArenaBuffer::Data(bytes),
            Init::Zeroed(len) => LaneArenaBuffer::Zeroed(*len),
            Init::Resident(id) => LaneArenaBuffer::Resident(*id),
        })
        .collect();
    let steps: Vec<LaneStepSpec<'_>> = built
        .steps
        .iter()
        .map(|s| LaneStepSpec {
            label: s.label,
            entry: s.entry,
            shader_wgsl: s.shader,
            uniforms: &s.uniforms,
            bindings: &s.bindings,
            workgroups: s.workgroups,
        })
        .collect();
    let spec = LaneProgramSpec {
        label: "margin_row_gpu_backward_pass",
        buffers: &buffers,
        steps: &steps,
        readbacks: &built.readbacks,
        profile,
    };
    let outs = gpu
        .run_lane_program_timed(&spec, timing)
        .map_err(|_| Refusal::Device)?;
    if outs.len() != built.readbacks.len() {
        return Err(Refusal::Payload);
    }
    Ok(outs
        .into_iter()
        .map(|bytes| f32s_from_bytes(&bytes))
        .collect())
}

/// Host-side directed-rounding egress: readbacks -> certified `PassOut`.
///
/// Values (ds) convert EXACTLY; every error quantity is widened by its
/// composed f32 envelope, floored by the flush charge, and folded with the
/// seed penalty — each host op stepped UP (`next_up`), the
/// `add_nonnegative_f64_up` pattern of design §4.2.
pub(super) fn egress(
    built: &BuiltProgram<'_>,
    streams: Vec<Vec<f32>>,
    pen_seed: &[f64],
) -> Result<PassOut, Refusal> {
    if streams.len() != 4 + built.colls.len() {
        return Err(Refusal::Payload);
    }
    let mut it = streams.into_iter();
    let coeff = it.next().expect("len checked");
    let err = it.next().expect("len checked");
    let bias = it.next().expect("len checked");
    let ebias = it.next().expect("len checked");
    // B5: the heuristic-only collect readbacks, ds -> f64 exactly. NO
    // certified charge, no widening, no floor — nothing verdict-bearing
    // reads this map (see `Builder::push_collect`).
    let mut coll: Option<std::collections::BTreeMap<usize, Vec<f64>>> =
        built.collect_unst_abs.then(std::collections::BTreeMap::new);
    for (desc, stream) in built.colls.iter().zip(it) {
        if stream.len() != desc.positions * 2 {
            return Err(Refusal::Payload);
        }
        let mut sums = Vec::with_capacity(desc.positions);
        for p in 0..desc.positions {
            let (hi, lo) = (stream[2 * p], stream[2 * p + 1]);
            if !hi.is_finite() || !lo.is_finite() {
                return Err(Refusal::Payload);
            }
            sums.push(f64::from(hi) + f64::from(lo));
        }
        coll.as_mut()
            .ok_or(Refusal::Payload)?
            .insert(desc.layer, sums);
    }
    let (n_in, rows) = (built.n_in, built.rows);
    if coeff.len() != n_in * rows * 2
        || err.len() != n_in * rows
        || bias.len() != rows * 2
        || ebias.len() != rows
        || pen_seed.len() != rows
    {
        return Err(Refusal::Payload);
    }
    let mut a = Array2::<f64>::zeros((n_in, rows));
    let mut e = Array2::<f64>::zeros((n_in, rows));
    {
        let av = a.as_slice_mut().expect("standard layout");
        let ev = e.as_slice_mut().expect("standard layout");
        for i in 0..n_in * rows {
            let hi = coeff[2 * i];
            let lo = coeff[2 * i + 1];
            let err32 = err[i];
            if !hi.is_finite() || !lo.is_finite() || !err32.is_finite() || err32 < 0.0 {
                return Err(Refusal::Payload);
            }
            // Exact: both components are exact in f64 and the renormalized
            // pair fits ~49 significand bits (ds.rs `Ds::to_f64`).
            av[i] = f64::from(hi) + f64::from(lo);
            // Widened, flush-floored, then split-floored (the absolute
            // subnormal-split channel — 0.0 on normal-band parameters).
            ev[i] = next_up(
                next_up(next_up(f64::from(err32) * built.e_widen) + built.flush_e) + built.split_e,
            );
        }
    }
    let mut b = Vec::with_capacity(rows);
    let mut eb = Vec::with_capacity(rows);
    for r in 0..rows {
        let hi = bias[2 * r];
        let lo = bias[2 * r + 1];
        let eb32 = ebias[r];
        if !hi.is_finite() || !lo.is_finite() || !eb32.is_finite() || eb32 < 0.0 {
            return Err(Refusal::Payload);
        }
        b.push(f64::from(hi) + f64::from(lo));
        eb.push(next_up(
            next_up(
                next_up(next_up(f64::from(eb32) * built.eb_widen) + built.flush_eb)
                    + built.split_eb,
            ) + pen_seed[r],
        ));
    }
    let out = PassOut {
        a,
        e: Some(e),
        b,
        eb,
        coll,
        coll_rows: None,
        unst_rows: None,
    };
    // The lane's fail-closed NaN/Inf firewall (identical to the CPU pass's).
    if out.a.iter().any(|v| !v.is_finite())
        || out.b.iter().any(|v| !v.is_finite())
        || out
            .e
            .as_ref()
            .is_some_and(|e| e.iter().any(|v| !v.is_finite()))
        || out.eb.iter().any(|v| !v.is_finite())
    {
        return Err(Refusal::NonFinite);
    }
    Ok(out)
}

/// Host-observable stats of one pass (R9 telemetry; NEVER verdict input).
#[derive(Debug, Default, PartialEq)]
pub(crate) struct ExecStats {
    /// `Some(true)` = residency-cache hit, `Some(false)` = miss (built +
    /// stored), `None` = no cache offered (the per-pass rebuild path).
    pub(crate) plan_cache_hit: Option<bool>,
    /// Host program assembly (cache probe / PlanStatic build + build_program).
    pub(crate) build_ms: f64,
    /// Executor prepare phase (buffer creation + uploads + bind groups).
    pub(crate) upload_ms: f64,
    /// Executor submit -> completed drain.
    pub(crate) dispatch_ms: f64,
    /// Per-step (label, ms) — profile mode only.
    pub(crate) per_step_ms: Option<Vec<(&'static str, f64)>>,
    /// FIX D2 observable: per-pass arena bytes this program allocated
    /// (`BuiltProgram::arena_bytes`, budget [`MAX_PASS_ARENA_BYTES`]).
    pub(crate) arena_bytes: u64,
    /// FIX D3 observable: bytes LIVE in the executor's resident-plane
    /// registry after this pass (`MarginRowLaneGpu::resident_bytes`). The
    /// D2/D3 availability pair is watched together on the shadow line.
    pub(crate) resident_bytes: u64,
}

/// One certified pass on the lane device. Requires [`VerdictAuthority`]
/// (the type-enforced M1 gate); the exclusivity token and the host-side
/// deadline discipline of design §6 live here.
///
/// `cache`: the B1 plan-residency cache, OWNED BY THE CALLER (this session:
/// only the discard-only M2 shadow entry offers one; the M3-gated verdict
/// path passes `None` and rebuilds per pass — bit-for-bit the pre-B1
/// behavior). `profile`: the one-step-per-submit timing mode, structurally
/// unreachable from the verdict path (its caller passes `false`).
#[allow(clippy::too_many_arguments)]
pub(super) fn execute(
    authority: &VerdictAuthority,
    gpu: &std::sync::Arc<dyn MarginRowLaneGpu>,
    eng: &BackwardEngine<'_>,
    seed: &Seed,
    dom: Option<&DomainGates>,
    dir: LaneDir,
    shape: PassShape<'_>,
    ctx: &super::LaneCtx<'_>,
    plan_cache: Option<&PlanCache>,
    profile: bool,
    stats: &mut ExecStats,
) -> Result<PassOut, Refusal> {
    let _ = authority.parity_lanes(); // the capability is the point, not the value
    let net = eng.net;
    let rows = seed.s.ncols();

    let expired = |deadline: Option<Instant>| deadline.is_some_and(|d| Instant::now() >= d);
    if expired(ctx.deadline) {
        return Err(Refusal::Deadline);
    }

    // prepare: statics (cached or built) + per-dom gate admission + seed
    // penalty + program (all host-side, fallible). The plan builder's own
    // rho_star is deliberately NOT used for the floor guard: it includes the
    // u32 downcast ball of the f32 payload this lane replaces with ds pairs —
    // the lane floors against the fold+split ball it actually charges
    // (BuiltProgram::rho_star_ds).
    let build_started = Instant::now();
    let statics: std::sync::Arc<PlanStatic> = match plan_cache {
        Some(pc) => match pc.probe(eng) {
            Some(hit) => {
                // A hit skips build_plan_ext, so the per-dom gate admission
                // it would have re-run must run here (refusal parity).
                cache::check_gate_admission(eng, dom, &hit.ext.relus)?;
                stats.plan_cache_hit = Some(true);
                hit
            }
            None => {
                // Residency uploads happen ONLY on the cached path (a
                // per-pass build must not leak resident device buffers).
                // FIX D3: the built `PlanStatic` now OWNS its resident ids —
                // each plane releases its device buffer when the static is
                // dropped (cache.rs `ResidentPlane::drop`), so the registry
                // no longer grows once per BaB instance for the process's
                // whole lifetime.
                let built = std::sync::Arc::new(PlanStatic::build(eng, dom, Some(gpu))?);
                pc.store(std::sync::Arc::clone(&built));
                stats.plan_cache_hit = Some(false);
                built
            }
        },
        // No cache offered: per-pass rebuild, no residency (pre-B1 shape).
        None => std::sync::Arc::new(PlanStatic::build(eng, dom, None)?),
    };
    let pen_seed =
        gpu_seam::seed_penalty(seed, ctx.y_abs, net.n_y, rows).map_err(map_seam_refusal)?;
    let built = build_program(
        &statics, eng, dom, seed, dir, shape, net.n_y, net.n_in, rows,
    )?;
    stats.build_ms = build_started.elapsed().as_secs_f64() * 1e3;
    // R9 charge-profile telemetry, once per process: the per-charge breakdown
    // of the certified radius must be readable off any shadow/A-B run log.
    {
        static PROFILE_ONCE: std::sync::Once = std::sync::Once::new();
        PROFILE_ONCE.call_once(|| {
            eprintln!(
                "[gpu-backward] charge profile (first pass): {}",
                built.profile.join("; ")
            );
        });
    }

    // submit: last host-side deadline check strictly BEFORE submission; the
    // executor's prepare (upload) sits between this check and the submit,
    // and from submit to completed drain there is no fallible operation (R2;
    // in profile mode the R2 shape holds per submit).
    if expired(ctx.deadline) {
        return Err(Refusal::Deadline);
    }
    let mut timing = LaneProgramTiming::default();
    let streams = match dispatch_program(gpu.as_ref(), &built, profile, &mut timing) {
        Ok(streams) => streams,
        Err(refusal) => {
            // The executor has already driven the queue to idle (R2); a device
            // failure is terminal for the lane's channel (R3, lane-local).
            if matches!(refusal, Refusal::Device) {
                super::mark_channel_dead("L-lane device transaction failed");
            }
            return Err(refusal);
        }
    };
    stats.upload_ms = timing.upload_ms;
    stats.dispatch_ms = timing.dispatch_ms;
    // D2/D3 availability pair (R9): both must be readable off any shadow line.
    stats.arena_bytes = built.arena_bytes;
    stats.resident_bytes = gpu.resident_bytes();
    stats.per_step_ms = timing.per_step_ms.map(|per_step| {
        built
            .steps
            .iter()
            .map(|s| s.label)
            .zip(per_step)
            .collect::<Vec<_>>()
    });

    // publish: only now consult the deadline — expired means DISCARD.
    if expired(ctx.deadline) {
        return Err(Refusal::Deadline);
    }
    let pass = egress(&built, streams, &pen_seed)?;
    if !gpu_seam::error_floor_ok(&pass, eng.root.xabs.as_slice(), built.rho_star_ds) {
        return Err(Refusal::ErrorFloor);
    }
    // Fix A3: the realization probe consumes the PlanStatic's cached
    // midpoint forward (same decision procedure as gpu_seam::probe_guard;
    // the midpoint is pass-invariant, so one forward serves every pass).
    cache::probe_guard_cached(eng, seed, dom, dir, &pass, &statics.probe)?;
    Ok(pass)
}
