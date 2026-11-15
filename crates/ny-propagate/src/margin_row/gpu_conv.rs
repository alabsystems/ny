// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #mr-gpu-eval: the certified pass's CONV BACKWARD products on the GPU.
//!
//! # What moves, and why this op
//!
//! Every margin-row certified pass (`engine::BackwardEngine::run_collect_lane`
//! — the tree's main eval, the stacked main eval `domain_stack` batches, the
//! candidate-scoring pass, the y-refresh, the root eval, the backward-interm
//! prefix passes and the alpha ascent) spends its arithmetic in ONE op: the
//! conv arm's backward products. Per conv op the signed coefficient product
//! `lnew = W^T L` and the per-column error-lane product `en = |W|^T comb` each
//! cost `k_bwd = taps*co` multiply-adds per output entry (144-2304 on the
//! cifar100 / tinyimagenet resnets); every other op of the pass (ReLU gates,
//! beta shifts, exceptions, bias folds, merges) is O(1) per entry. The col-
//! scaling probe measured the pass as linear in `trunk work x columns` with the
//! error lane exactly half of it (docs/MARGIN_ROW_COLUMN_SCALING_2026-08-23.md,
//! B1 probe `outward_over_parity` 2.00-2.04). This module routes those two
//! products — and the row envelope's `R = 1` `|W|` product
//! (`errlane::env_conv_per_row`) — to the process-global SOUND f64 GEMM engine
//! (cuBLAS `Dgemm` on a CUDA build) as ONE GEMM per spatial tile:
//!
//! ```text
//!   C (ci x ip*R) = A (ci x taps*co) . B (taps*co x ip*R)
//!   A[ic, (ky*kw + kx)*co + oc] = wt[ic][ky][kx][oc]        (== ConvOp::wt, flat)
//!   B[(ky*kw + kx)*co + oc, isp*R + c] = src[oc*op + osp, c] if (ky,kx,osp) is a
//!                                        back tap of isp, else EXACT 0.0
//!   C[ic, isp*R + c] = dst[ic*ip + isp, c]                  (== dst's own layout)
//! ```
//!
//! # Soundness — the certificate is the CPU kernel's own, unchanged
//!
//! * SAME CONTRACTION. Row `(ic, isp, c)` of `C` is the sum, over every listed
//!   back tap of `isp` and every `oc`, of the same stored-f64 products the CPU
//!   kernel (`net::conv_backward_icgrain` / `conv_backward_blocked`) folds. The
//!   gathered `B` copies `src` bit-for-bit and `A` IS `ConvOp::wt` (or its
//!   elementwise `abs`, exactly the CPU's `w0.abs()`): NO f32 cast, no
//!   splitting, no new representation error anywhere.
//! * ORDER-INDEPENDENT ROUNDING. The engine slot's contract is IEEE-754
//!   binary64 round-to-nearest in ANY summation order (FMA and split-K trees
//!   included; `gemm_f64_with_deadline` splits only the output axes, never the
//!   contraction). For any evaluation tree of `n` nonzero products,
//!   `|fl(sum) - sum| <= gamma_n * sum|terms|` (Higham, Thm 3.1 / sec 3.1),
//!   independent of order. The absent taps are EXACT zeros: `0 * w` is `+-0`
//!   exactly for finite `w`, `fl(s + 0) = s`, `fma(0, w, s) = s`, so a zero
//!   leaf adds no rounding factor to any nonzero leaf's path and the rounding
//!   count stays `n_nz <= |back_taps[isp]| * co <= k_bwd`.
//! * THEREFORE the engine's existing charges cover the device result exactly
//!   as they cover the CPU kernel's (which is itself just one particular
//!   order): the signed product's rounding is inside
//!   `g = gamma(k_bwd + 2) + rel + gamma * rel` of `comb = E + g(|L| + E)`,
//!   and the non-negative `|W|` product's RN under-report (at most a
//!   `1 - gamma(n_nz)` factor) is inside `certify_up(., gamma(k_bwd + 8))`.
//!   No constant changes; published bounds differ from the dark lane only at
//!   the ulp level of a reordered f64 sum.
//! * UNDERFLOW. CUDA binary64 arithmetic keeps gradual underflow (FTZ/DAZ are
//!   f32-only), so the device's underflow behaviour is the CPU's; the lane
//!   treats both identically.
//! * FAIL-CLOSED. A transposed conv, a duplicate or out-of-range back tap, any
//!   non-finite operand, a CPU-only f64 scope, a missing / non-CUDA engine, any
//!   engine `Err` (lock wait or dispatch past the per-call budget included), a
//!   wrong output length or ANY non-finite output entry ⇒ this module returns
//!   `false` and the caller runs the unchanged CPU kernel, which overwrites
//!   every element of `dst` — bit-identical to the dark pass for that op. A
//!   device result is published only whole.
//! * BETA. The moved op is the LINEAR conv arm. Gates, exceptions and both
//!   beta kinds are host-side elementwise updates in the ReLU arm and are
//!   untouched, so the moved phase carries no beta logic to get wrong — unlike
//!   `gpu_backward`, which runs the WHOLE pass on device and refuses any
//!   domain with beta (memory: ny-THE-GPU-MAP-beta-blocks-the-phase-that-matters).
//!
//! # Why f64 and not f32 + gamma
//!
//! An f32 GEMM certified by an a-priori `gamma_k^f32 |A||B|` radius is sound
//! but VERDICT-BREAKING on this lane: the predecessor seam measured that radius
//! at 0.088-0.126 absolute on the cifar100 fold against closing margins of
//! ~1e-2..1e-1 (`gpu_backward` module docs, defect 1). An exact-slice (Ozaki)
//! f32 scheme needs O(10-30) slice GEMMs plus a host gather per slice. The GB10
//! measures cuBLAS `Dgemm` at 0.41 TF/s (memory ny-gpu-on-gb10) — roughly 15x
//! the CPU pass's effective f64 rate on these convs (2.4 ms/col for ~32 M
//! MACs/col on resnet_medium, 20 cores) — with the certificate unchanged.
//!
//! # RETIRED (ENGINE_PROGRAM R1 kill line + R2, 2026-09-29)
//!
//! Measured on the yaml route with zero `NY_*` (W7B, `w7b/w7b_v2.log`):
//! 38,552 per-op conv offloads at 3.74 ms each; cifar100 c2050 unsat 88.5 s ->
//! timeout and c1590 unsat 78 s -> unknown. That is program rule R1's kill
//! line ("batch before device": no per-op offload), and the route also puts
//! device products into certified values, which R2 forbids until §1.9's M9
//! preconditions hold. So the lane route is NEVER armed in any build
//! ([`armed`] is `false`): the typed key `margin_row.gpu_conv` is refused at
//! preset validation, its axis is `Unsafe` / `EnvOnly`, and an environment
//! request prints `[mr-gpu-conv] refused reason=retired(...)` at lane start
//! instead of arming. The code stays compiled and tested DARK behind the
//! test-only injected routes ([`GpuConvRoute::Engine`], `#bi-gpu`'s
//! test-admitted scope) as the reference for an M9 device certifier. With the
//! route [`GpuConvRoute::Off`] the engine calls `conv_apply_backward` exactly
//! as before — no counter, no allocation, no engine lookup.
//!
//! # Engagement observable (fixed schema, every lane)
//!
//! * lane start: `[mr-gpu-conv] armed route=sound_f64_engine ...` or
//!   `[mr-gpu-conv] dark` ([`announce`]);
//! * lane stop (census stop receipts, every stop path): `[mr-gpu-conv] done
//!   armed=<0|1> convs= gpu= ... verdict=<dark|VOID|engaged>` ([`stop_line`]).
//!   An armed lane with `gpu=0` prints `verdict=VOID`: it never engaged, which
//!   is not a negative.
//!
//! # Scoped routes (`#bi-gpu`)
//!
//! [`GpuConvRoute::Scoped`] wraps an inner route (`Global`, or a test
//! `Engine`) with its OWN counters ([`ScopedRoute`]): the same admission, the
//! same certificate and the same per-op CPU fallback as the inner route, but
//! every call is counted in the scope's tally instead of the lane counters
//! above, so a phase that routes only ITS passes (the backward-interm prefix
//! pairs, `super::bi_gpu`) can prove its own engagement without disturbing
//! the `[mr-gpu-conv]` receipt. Nothing constructs a scoped route unless
//! `margin_row.bi_gpu` is armed.

use std::borrow::Cow;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use ndarray::Array2;
use ny_core::{GemmEngine, NyError};
use rayon::prelude::*;

use super::census::elapsed_ns;
use super::net::ConvOp;

/// Smallest padded GEMM (`ci * ip * taps * co * R` multiply-adds) the GLOBAL
/// route dispatches. Below it one engine call's fixed cost (lock, launch,
/// synchronize: ~0.1-0.2 ms on the GB10) is comparable to the CPU kernel's
/// work. The R = 1 resnet_medium convs (~2.4 M) clear it; the 3-channel input
/// conv clears it from R = 3. A PERFORMANCE floor only: both routes are
/// certified by the same charges.
pub(crate) const GPU_CONV_MIN_MACS: u64 = 1 << 20;

/// Requested cap on one non-interruptible engine dispatch (MACs), passed
/// through `GemmEngine::deadline_dispatch_macs_hint`. The CUDA engine never
/// splits the contraction, so the certificate is unaffected by the value.
pub(crate) const GPU_CONV_DISPATCH_MACS: usize = 1 << 24;

/// Cap on the gathered `B` tile, in f64 entries (32 MiB). Tiles split the
/// spatial (output-column) axis only; every output entry stays ONE full
/// contraction.
pub(crate) const GPU_CONV_TILE_ELEMS: usize = 1 << 22;

/// Conservative CPU-equivalent rate used to size one call's wall budget: a
/// device call that has not finished (lock wait included) by the time the CPU
/// kernel would plausibly have finished is abandoned for the CPU kernel.
const FALLBACK_MACS_PER_SEC: f64 = 1.0e10;
/// Per-call budget floor (covers a cold lock wait behind another caller).
const CALL_BUDGET_FLOOR: Duration = Duration::from_millis(250);
/// Per-call budget ceiling.
const CALL_BUDGET_CAP: Duration = Duration::from_secs(30);
/// After a GLOBAL-route call misses its budget (the shared cuBLAS handle is
/// held by another caller — e.g. the internal verifier's sweep), every global
/// call refuses immediately for this long instead of each conv paying its own
/// budget in lock waits. Pure scheduling: a refusal runs the CPU kernel.
const BUSY_BACKOFF: Duration = Duration::from_secs(1);

/// Sentinel for an absent back tap in the dense tap table.
const NO_TAP: usize = usize::MAX;

/// Where a certified pass's conv backward products run.
#[derive(Clone, Default)]
pub(crate) enum GpuConvRoute {
    /// Dark: the historical CPU kernel, byte-identical.
    #[default]
    Off,
    /// Armed: the process-global sound-f64 engine (`crate::sound_f64_gemm`),
    /// admitted per call (CUDA provenance, MAC floor, not inside a CPU-only
    /// f64 scope).
    Global,
    /// Test / probe injection: THIS engine, admitted for every shape (no MAC
    /// floor, no provenance check). Same certificate, same fallback. Only the
    /// unit tests construct it today.
    #[cfg_attr(not(test), allow(dead_code))]
    Engine(Arc<dyn GemmEngine>),
    /// `#bi-gpu`: the inner route, counted in the scope's own tally (see the
    /// module docs, "Scoped routes"). Built only by [`ScopedRoute::new`], which
    /// never nests one scope inside another.
    Scoped(Arc<ScopedRoute>),
}

impl GpuConvRoute {
    /// True on the dark route (a scope over the dark route is dark too).
    #[inline]
    pub(crate) fn is_off(&self) -> bool {
        match self {
            Self::Off => true,
            Self::Scoped(scoped) => scoped.inner.is_off(),
            Self::Global | Self::Engine(_) => false,
        }
    }

    /// The lane's route for a resolved arming decision.
    #[must_use]
    pub(crate) fn from_armed(armed: bool) -> Self {
        if armed {
            Self::Global
        } else {
            Self::Off
        }
    }
}

/// Whether the lane route is armed: NEVER (retired; see the module docs,
/// "RETIRED"). A request is still observable through [`requested`].
#[must_use]
pub(crate) const fn armed() -> bool {
    false
}

/// Whether the RETIRED switch was requested (the environment is the only
/// route left: the typed key is refused at preset validation), latched once
/// per process. Observable only: it never arms anything.
pub(crate) fn requested() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(crate::preset_gates::margin_row_gpu_conv_enabled)
}

/// The route every [`super::engine::BackwardEngine`] starts with.
#[must_use]
pub(crate) fn lane_route() -> GpuConvRoute {
    GpuConvRoute::from_armed(armed())
}

/// Which of the conv arm's products a call is (telemetry only).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ConvLane {
    /// The signed coefficient product `W^T L`.
    Signed,
    /// The per-column error lane's `|W|^T comb`.
    Abs,
    /// The row envelope's `R = 1` `|W|^T comb`. Since 3739534a8 the row
    /// envelope contracts on the CPU (`errlane::env_conv_per_row` calls
    /// `envelope_error_conv_without_deadline`), so no production call offloads
    /// this lane; it stays a test-only tag of the offload contract.
    #[cfg(test)]
    EnvRow,
}

/// Why a call ran on the CPU instead (every one is counted).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Refusal {
    /// Below [`GPU_CONV_MIN_MACS`] (global route only).
    Small,
    /// Parity (measurement-grade) pass: the Python-parity arithmetic is kept.
    Parity,
    /// A transposed conv (its tap tables are not the plain-conv geometry this
    /// gather encodes).
    Transposed,
    /// Shape / layout / tap-table inconsistency, or a contraction the engine
    /// cannot dispatch in one piece.
    Shape,
    /// A non-finite operand (`0 * inf` would differ from the CPU's zero skip).
    NonFiniteIn,
    /// Inside an explicit CPU-only sound-f64 scope on this thread.
    CpuOnlyScope,
    /// No engine materialized yet (background initialization was started).
    NoEngine,
    /// The engine in the slot is not a CUDA engine (e.g. the CPU faer floor).
    Backend,
    /// The engine returned an error other than a deadline.
    Engine,
    /// The per-call wall budget expired (lock wait or dispatch).
    Deadline,
    /// Wrong output length or a non-finite output entry.
    BadOutput,
    /// Inside the [`BUSY_BACKOFF`] window after a global-route deadline miss.
    Backoff,
}

const REFUSALS: usize = 12;

impl Refusal {
    const ALL: [Refusal; REFUSALS] = [
        Refusal::Small,
        Refusal::Parity,
        Refusal::Transposed,
        Refusal::Shape,
        Refusal::NonFiniteIn,
        Refusal::CpuOnlyScope,
        Refusal::NoEngine,
        Refusal::Backend,
        Refusal::Engine,
        Refusal::Deadline,
        Refusal::BadOutput,
        Refusal::Backoff,
    ];

    fn index(self) -> usize {
        match self {
            Refusal::Small => 0,
            Refusal::Parity => 1,
            Refusal::Transposed => 2,
            Refusal::Shape => 3,
            Refusal::NonFiniteIn => 4,
            Refusal::CpuOnlyScope => 5,
            Refusal::NoEngine => 6,
            Refusal::Backend => 7,
            Refusal::Engine => 8,
            Refusal::Deadline => 9,
            Refusal::BadOutput => 10,
            Refusal::Backoff => 11,
        }
    }

    /// Stop-receipt field name.
    fn tag(self) -> &'static str {
        match self {
            Refusal::Small => "small",
            Refusal::Parity => "parity",
            Refusal::Transposed => "transposed",
            Refusal::Shape => "shape",
            Refusal::NonFiniteIn => "nonfinite_in",
            Refusal::CpuOnlyScope => "cpu_only",
            Refusal::NoEngine => "no_engine",
            Refusal::Backend => "backend_refused",
            Refusal::Engine => "engine_err",
            Refusal::Deadline => "deadline",
            Refusal::BadOutput => "bad_output",
            Refusal::Backoff => "backoff",
        }
    }
}

// ---------------------------------------------------------------------------
//  Counters (lane-scoped: reset by the census at lane start)
// ---------------------------------------------------------------------------

static CONVS: AtomicU64 = AtomicU64::new(0);
static GPU: AtomicU64 = AtomicU64::new(0);
static GPU_SIGNED: AtomicU64 = AtomicU64::new(0);
static GPU_ABS: AtomicU64 = AtomicU64::new(0);
static GPU_ENV: AtomicU64 = AtomicU64::new(0);
static GPU_MACS: AtomicU64 = AtomicU64::new(0);
static GPU_NS: AtomicU64 = AtomicU64::new(0);
static REFUSED: [AtomicU64; REFUSALS] = [const { AtomicU64::new(0) }; REFUSALS];
/// Provenance of the first engine that carried a call (process-level; the
/// slot never changes engines once materialized).
static BACKEND: OnceLock<&'static str> = OnceLock::new();
/// End of the current [`BUSY_BACKOFF`] window, in ns since [`clock_base`].
static BACKOFF_UNTIL_NS: AtomicU64 = AtomicU64::new(0);

/// Process-local monotonic origin for the backoff window.
fn clock_base() -> Instant {
    static BASE: OnceLock<Instant> = OnceLock::new();
    *BASE.get_or_init(Instant::now)
}

fn clock_ns() -> u64 {
    u64::try_from(clock_base().elapsed().as_nanos()).unwrap_or(u64::MAX)
}

// ---------------------------------------------------------------------------
//  Scoped counters (#bi-gpu)
// ---------------------------------------------------------------------------

/// One scope's call counters: the same fields as the lane counters above, but
/// owned by a [`ScopedRoute`] (so they are never reset by the lane census and
/// never mixed into the `[mr-gpu-conv]` receipt).
pub(crate) struct ConvTally {
    convs: AtomicU64,
    gpu: AtomicU64,
    gpu_signed: AtomicU64,
    gpu_abs: AtomicU64,
    gpu_env: AtomicU64,
    macs: AtomicU64,
    ns: AtomicU64,
    refused: [AtomicU64; REFUSALS],
}

impl ConvTally {
    fn new() -> Self {
        Self {
            convs: AtomicU64::new(0),
            gpu: AtomicU64::new(0),
            gpu_signed: AtomicU64::new(0),
            gpu_abs: AtomicU64::new(0),
            gpu_env: AtomicU64::new(0),
            macs: AtomicU64::new(0),
            ns: AtomicU64::new(0),
            refused: [const { AtomicU64::new(0) }; REFUSALS],
        }
    }

    /// A consistent-enough copy of every counter (each load is `Relaxed`; a
    /// scope's calls are issued by ONE pass at a time, so a snapshot taken
    /// between passes is exact).
    fn snapshot(&self) -> TallySnapshot {
        let mut refused = [0u64; REFUSALS];
        for (slot, counter) in refused.iter_mut().zip(self.refused.iter()) {
            *slot = counter.load(Relaxed);
        }
        TallySnapshot {
            convs: self.convs.load(Relaxed),
            gpu: self.gpu.load(Relaxed),
            gpu_signed: self.gpu_signed.load(Relaxed),
            gpu_abs: self.gpu_abs.load(Relaxed),
            gpu_env: self.gpu_env.load(Relaxed),
            macs: self.macs.load(Relaxed),
            ns: self.ns.load(Relaxed),
            refused,
        }
    }
}

/// Plain copy of a [`ConvTally`] (differenced around one pass pair by
/// `super::bi_gpu`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct TallySnapshot {
    /// Conv products offered to the route (device-carried + refused).
    pub(crate) convs: u64,
    /// Conv products whose result came from a validated device GEMM.
    pub(crate) gpu: u64,
    /// ... of which signed `W^T L`, per-column `|W|^T comb`, envelope `R = 1`.
    pub(crate) gpu_signed: u64,
    pub(crate) gpu_abs: u64,
    pub(crate) gpu_env: u64,
    /// Padded MACs dispatched to the device.
    pub(crate) macs: u64,
    /// Wall ns of the device-carried calls.
    pub(crate) ns: u64,
    /// Refused calls per [`Refusal`] (each ran the unchanged CPU kernel).
    refused: [u64; REFUSALS],
}

impl TallySnapshot {
    /// Conv products that fell back to the CPU kernel (every reason).
    #[must_use]
    pub(crate) fn fallback(&self) -> u64 {
        self.refused.iter().sum()
    }

    /// Refusals for one reason (the unit tests' per-reason probe).
    #[must_use]
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn refused(&self, reason: Refusal) -> u64 {
        self.refused[reason.index()]
    }

    /// Field-wise `self - earlier` (saturating; counters only grow).
    #[must_use]
    pub(crate) fn since(&self, earlier: &Self) -> Self {
        let mut refused = [0u64; REFUSALS];
        for (i, slot) in refused.iter_mut().enumerate() {
            *slot = self.refused[i].saturating_sub(earlier.refused[i]);
        }
        Self {
            convs: self.convs.saturating_sub(earlier.convs),
            gpu: self.gpu.saturating_sub(earlier.gpu),
            gpu_signed: self.gpu_signed.saturating_sub(earlier.gpu_signed),
            gpu_abs: self.gpu_abs.saturating_sub(earlier.gpu_abs),
            gpu_env: self.gpu_env.saturating_sub(earlier.gpu_env),
            macs: self.macs.saturating_sub(earlier.macs),
            ns: self.ns.saturating_sub(earlier.ns),
            refused,
        }
    }

    /// ` small=N parity=N ... backoff=N` — every refusal reason, fixed order
    /// (the `[mr-gpu-conv] done` receipt's field names).
    #[must_use]
    pub(crate) fn refusal_fields(&self) -> String {
        let mut fields = String::new();
        for r in Refusal::ALL {
            fields.push_str(&format!(" {}={}", r.tag(), self.refused[r.index()]));
        }
        fields
    }
}

/// `#bi-gpu`: an inner route plus the counters of every call made through it.
pub(crate) struct ScopedRoute {
    inner: GpuConvRoute,
    tally: ConvTally,
}

impl ScopedRoute {
    /// A fresh scope over `inner` (a scoped `inner` is unwrapped: scopes never
    /// nest, so every call is counted exactly once).
    #[must_use]
    pub(crate) fn new(inner: GpuConvRoute) -> Arc<Self> {
        let inner = match inner {
            GpuConvRoute::Scoped(scoped) => scoped.inner.clone(),
            other => other,
        };
        Arc::new(Self {
            inner,
            tally: ConvTally::new(),
        })
    }

    /// The scope's counters so far.
    #[must_use]
    pub(crate) fn tally(&self) -> TallySnapshot {
        self.tally.snapshot()
    }

    /// [`offload`] through the inner route, counted in THIS scope only.
    fn offload(
        &self,
        cv: &ConvOp,
        src: &Array2<f64>,
        dst: &mut Array2<f64>,
        abs_w: bool,
        lane: ConvLane,
        outward: bool,
    ) -> bool {
        let t = &self.tally;
        t.convs.fetch_add(1, Relaxed);
        match attempt(&self.inner, cv, src, dst, abs_w, outward, &t.ns) {
            Ok(macs) => {
                t.gpu.fetch_add(1, Relaxed);
                t.macs.fetch_add(macs, Relaxed);
                match lane {
                    ConvLane::Signed => t.gpu_signed.fetch_add(1, Relaxed),
                    ConvLane::Abs => t.gpu_abs.fetch_add(1, Relaxed),
                    #[cfg(test)]
                    ConvLane::EnvRow => t.gpu_env.fetch_add(1, Relaxed),
                };
                true
            }
            Err(refusal) => {
                t.refused[refusal.index()].fetch_add(1, Relaxed);
                false
            }
        }
    }
}

/// Provenance of the engine that carried the first device call in this
/// process (`none` before any call). Shared by every route.
#[must_use]
pub(crate) fn backend() -> &'static str {
    BACKEND.get().copied().unwrap_or("none")
}

/// Reset every lane counter (called by `census::reset` at lane start).
pub(crate) fn reset() {
    for a in [
        &CONVS,
        &GPU,
        &GPU_SIGNED,
        &GPU_ABS,
        &GPU_ENV,
        &GPU_MACS,
        &GPU_NS,
    ] {
        a.store(0, Relaxed);
    }
    for a in &REFUSED {
        a.store(0, Relaxed);
    }
}

/// Print (and return) the lane-start line. When armed, also start the sound
/// f64 engine's one-time background materialization (non-blocking), so the
/// first tree pass finds it ready; the root build takes seconds.
pub(crate) fn announce() -> String {
    let line = if armed() {
        crate::sound_f64_gemm::start_background_initialization();
        format!(
            "[mr-gpu-conv] armed route=sound_f64_engine backend_req=cuda \
             min_macs={GPU_CONV_MIN_MACS} dispatch_macs={GPU_CONV_DISPATCH_MACS} \
             tile_elems={GPU_CONV_TILE_ELEMS} source=gate"
        )
    } else {
        announce_line(requested())
    };
    eprintln!("{line}");
    line
}

/// The lane-start line of the (never armed) route: `[mr-gpu-conv] dark`, or,
/// when the retired switch was requested, the fixed-schema refusal.
#[must_use]
pub(crate) fn announce_line(requested: bool) -> String {
    if requested {
        "[mr-gpu-conv] refused reason=retired(r1_kill_c2050_c1590+r2_device_never_certifies) \
         route=cpu"
            .to_string()
    } else {
        "[mr-gpu-conv] dark".to_string()
    }
}

/// The `[mr-gpu-conv] done ...` stop receipt. Fixed schema: every field on
/// every line (`armed=0` lanes print zeros and `verdict=dark`).
pub(crate) fn stop_line() -> String {
    let is_armed = armed();
    let gpu = GPU.load(Relaxed);
    let refused: Vec<u64> = REFUSED.iter().map(|a| a.load(Relaxed)).collect();
    let fallback: u64 = refused.iter().sum();
    let mut fields = String::new();
    for r in Refusal::ALL {
        fields.push_str(&format!(" {}={}", r.tag(), refused[r.index()]));
    }
    let verdict = if !is_armed {
        "dark"
    } else if gpu == 0 {
        "VOID"
    } else {
        "engaged"
    };
    format!(
        "[mr-gpu-conv] done armed={} convs={} gpu={gpu} gpu_signed={} gpu_abs={} gpu_env={} \
         gpu_macs={} gpu_secs={:.3} fallback={fallback}{fields} backend={} verdict={verdict}",
        u8::from(is_armed),
        CONVS.load(Relaxed),
        GPU_SIGNED.load(Relaxed),
        GPU_ABS.load(Relaxed),
        GPU_ENV.load(Relaxed),
        GPU_MACS.load(Relaxed),
        GPU_NS.load(Relaxed) as f64 / 1e9,
        BACKEND.get().copied().unwrap_or("none"),
    )
}

/// Fields appended to the `[mr-census]` line.
pub(crate) fn census_fields() -> String {
    let fallback: u64 = REFUSED.iter().map(|a| a.load(Relaxed)).sum();
    format!(
        " gpuconv_armed={} gpuconv_gpu={} gpuconv_fallback={fallback}",
        u8::from(armed()),
        GPU.load(Relaxed),
    )
}

// ---------------------------------------------------------------------------
//  The offload
// ---------------------------------------------------------------------------

/// Padded GEMM work of one call: `ci * ip * taps * co * R` multiply-adds.
#[must_use]
pub(crate) fn conv_macs(cv: &ConvOp, r: usize) -> u64 {
    let (co, ci, kh, kw) = cv.kernel;
    let ip = cv.ishape.1 * cv.ishape.2;
    [ci, ip, kh * kw, co, r]
        .iter()
        .fold(1u64, |acc, &x| acc.saturating_mul(x as u64))
}

/// Wall budget for one call of `macs` work.
fn call_budget(macs: u64) -> Duration {
    let secs = macs as f64 / FALLBACK_MACS_PER_SEC;
    let proportional = if secs.is_finite() {
        Duration::from_secs_f64(secs.clamp(0.0, CALL_BUDGET_CAP.as_secs_f64()))
    } else {
        CALL_BUDGET_CAP
    };
    (CALL_BUDGET_FLOOR + proportional).min(CALL_BUDGET_CAP)
}

/// Try to compute `dst = conv_backward(cv, src, abs_w)` through `route`.
///
/// Returns `true` iff EVERY element of `dst` was written from a validated
/// device result. On `false` the caller MUST run `net::conv_apply_backward`
/// (which overwrites every element), so a partially written `dst` can never
/// be published. `outward = false` (a Parity pass) always refuses.
pub(crate) fn offload(
    route: &GpuConvRoute,
    cv: &ConvOp,
    src: &Array2<f64>,
    dst: &mut Array2<f64>,
    abs_w: bool,
    lane: ConvLane,
    outward: bool,
) -> bool {
    if route.is_off() {
        return false;
    }
    if let GpuConvRoute::Scoped(scoped) = route {
        return scoped.offload(cv, src, dst, abs_w, lane, outward);
    }
    CONVS.fetch_add(1, Relaxed);
    let outcome = attempt(route, cv, src, dst, abs_w, outward, &GPU_NS);
    match outcome {
        Ok(macs) => {
            GPU.fetch_add(1, Relaxed);
            GPU_MACS.fetch_add(macs, Relaxed);
            match lane {
                ConvLane::Signed => GPU_SIGNED.fetch_add(1, Relaxed),
                ConvLane::Abs => GPU_ABS.fetch_add(1, Relaxed),
                #[cfg(test)]
                ConvLane::EnvRow => GPU_ENV.fetch_add(1, Relaxed),
            };
            true
        }
        Err(refusal) => {
            REFUSED[refusal.index()].fetch_add(1, Relaxed);
            false
        }
    }
}

/// One routed call. `ns` receives the wall of a device-carried call (the lane
/// counter, or a scope's own).
fn attempt(
    route: &GpuConvRoute,
    cv: &ConvOp,
    src: &Array2<f64>,
    dst: &mut Array2<f64>,
    abs_w: bool,
    outward: bool,
    ns: &AtomicU64,
) -> Result<u64, Refusal> {
    if !outward {
        return Err(Refusal::Parity);
    }
    let macs = conv_macs(cv, src.ncols());
    let t0 = Instant::now();
    let deadline = t0 + call_budget(macs);
    let done = match route {
        GpuConvRoute::Off => return Err(Refusal::Shape),
        // `ScopedRoute::new` never nests scopes; fail closed if one ever did.
        GpuConvRoute::Scoped(_) => return Err(Refusal::Shape),
        GpuConvRoute::Engine(engine) => {
            conv_backward_on_engine(&**engine, cv, src, dst, abs_w, deadline)
        }
        GpuConvRoute::Global => {
            if macs < GPU_CONV_MIN_MACS {
                return Err(Refusal::Small);
            }
            if crate::sound_f64_gemm::cpu_only_f64_active() {
                return Err(Refusal::CpuOnlyScope);
            }
            if clock_ns() < BACKOFF_UNTIL_NS.load(Relaxed) {
                return Err(Refusal::Backoff);
            }
            let ran = crate::sound_f64_gemm::with_preinitialized_engine(|engine| {
                let backend = engine.backend_provenance();
                if !backend_admitted(backend) {
                    return Err(Refusal::Backend);
                }
                let _ = BACKEND.set(backend);
                conv_backward_on_engine(engine, cv, src, dst, abs_w, deadline)
            });
            let result = match ran {
                Some(result) => result,
                None => {
                    crate::sound_f64_gemm::start_background_initialization();
                    Err(Refusal::NoEngine)
                }
            };
            if result == Err(Refusal::Deadline) {
                let until = clock_ns()
                    .saturating_add(u64::try_from(BUSY_BACKOFF.as_nanos()).unwrap_or(u64::MAX));
                BACKOFF_UNTIL_NS.fetch_max(until, Relaxed);
            }
            result
        }
    };
    if done.is_ok() {
        ns.fetch_add(elapsed_ns(t0), Relaxed);
    }
    done
}

/// The global route admits only a CUDA engine: the CPU faer floor that fills
/// the same slot on a CUDA-less host would compete with the lane's own rayon
/// pool, and "gpu_conv" must mean the GPU. (`backend_provenance` is the
/// engine's truthful identity by the `GemmEngine` contract.)
#[must_use]
pub(crate) fn backend_admitted(backend: &str) -> bool {
    backend.starts_with("cuda")
}

/// Dense back-tap table `tab[isp * taps + ky * kw + kx] = osp` (or
/// [`NO_TAP`]). Refuses a tap outside the kernel / output, and a DUPLICATE
/// `(isp, ky, kx)` (the CPU kernel would add both; one GEMM row cannot).
pub(crate) fn tap_table(cv: &ConvOp) -> Result<Vec<usize>, Refusal> {
    let (_, _, kh, kw) = cv.kernel;
    let taps = kh * kw;
    let ip = cv.ishape.1 * cv.ishape.2;
    let op_ = cv.oshape.1 * cv.oshape.2;
    if cv.back_taps.len() != ip {
        return Err(Refusal::Shape);
    }
    let mut tab = vec![NO_TAP; ip * taps];
    for (isp, list) in cv.back_taps.iter().enumerate() {
        for &(ky, kx, osp) in list {
            if ky >= kh || kx >= kw || osp >= op_ {
                return Err(Refusal::Shape);
            }
            let slot = &mut tab[isp * taps + ky * kw + kx];
            if *slot != NO_TAP {
                return Err(Refusal::Shape);
            }
            *slot = osp;
        }
    }
    Ok(tab)
}

/// Gather `B` for spatial rows `i0..i1` (see the module docs), parallel over
/// the `taps * co` rows of `B`. Absent taps stay EXACT `0.0`.
#[allow(clippy::too_many_arguments)]
fn gather_tile(
    src: &[f64],
    tab: &[usize],
    co: usize,
    taps: usize,
    op_: usize,
    r: usize,
    i0: usize,
    i1: usize,
) -> Vec<f64> {
    let cols = (i1 - i0) * r;
    let mut b = vec![0.0f64; taps * co * cols];
    b.par_chunks_mut(cols).enumerate().for_each(|(j, brow)| {
        let t = j / co;
        let oc = j % co;
        for isp in i0..i1 {
            let osp = tab[isp * taps + t];
            if osp == NO_TAP {
                continue;
            }
            let s0 = (oc * op_ + osp) * r;
            let d0 = (isp - i0) * r;
            brow[d0..d0 + r].copy_from_slice(&src[s0..s0 + r]);
        }
    });
    b
}

/// The offload on an explicit engine: ONE `gemm_f64_with_deadline` per
/// spatial tile, every output entry one full-length contraction. Returns the
/// padded MACs dispatched, or the [`Refusal`] that sends the caller to the CPU
/// kernel. `dst` is written tile by tile; on `Err` the caller's CPU kernel
/// overwrites all of it (see [`offload`]).
pub(crate) fn conv_backward_on_engine(
    engine: &dyn GemmEngine,
    cv: &ConvOp,
    src: &Array2<f64>,
    dst: &mut Array2<f64>,
    abs_w: bool,
    deadline: Instant,
) -> Result<u64, Refusal> {
    if cv.transposed {
        return Err(Refusal::Transposed);
    }
    let (co, ci, kh, kw) = cv.kernel;
    let taps = kh * kw;
    let k = taps * co;
    let ip = cv.ishape.1 * cv.ishape.2;
    let op_ = cv.oshape.1 * cv.oshape.2;
    let r = src.ncols();
    if r == 0
        || k == 0
        || ci == 0
        || ip == 0
        || src.nrows() != co * op_
        || dst.nrows() != ci * ip
        || dst.ncols() != r
        || cv.wt.len() != ci * k
    {
        return Err(Refusal::Shape);
    }
    let Some(src_flat) = src.as_slice() else {
        return Err(Refusal::Shape);
    };
    if src_flat.iter().any(|v| !v.is_finite()) || cv.wt.iter().any(|v| !v.is_finite()) {
        return Err(Refusal::NonFiniteIn);
    }
    let tab = tap_table(cv)?;
    let cap = engine
        .deadline_dispatch_macs_hint(GPU_CONV_DISPATCH_MACS)
        .max(1);
    if k > cap {
        return Err(Refusal::Shape);
    }
    let a: Cow<'_, [f64]> = if abs_w {
        Cow::Owned(cv.wt.iter().map(|w| w.abs()).collect())
    } else {
        Cow::Borrowed(&cv.wt[..])
    };
    let Some(dst_flat) = dst.as_slice_mut() else {
        return Err(Refusal::Shape);
    };
    let tile_isp = (GPU_CONV_TILE_ELEMS / k.saturating_mul(r).max(1)).clamp(1, ip);
    let mut macs = 0u64;
    let mut i0 = 0usize;
    while i0 < ip {
        let i1 = (i0 + tile_isp).min(ip);
        let cols = (i1 - i0) * r;
        let b = gather_tile(src_flat, &tab, co, taps, op_, r, i0, i1);
        let c = engine
            .gemm_f64_with_deadline(ci, k, cols, &a, &b, deadline, cap)
            .map_err(|e| match e {
                NyError::DeadlineExceeded(_) => Refusal::Deadline,
                _ => Refusal::Engine,
            })?;
        if c.len() != ci * cols || c.iter().any(|v| !v.is_finite()) {
            return Err(Refusal::BadOutput);
        }
        for ic in 0..ci {
            let d0 = (ic * ip + i0) * r;
            dst_flat[d0..d0 + cols].copy_from_slice(&c[ic * cols..(ic + 1) * cols]);
        }
        macs = macs.saturating_add(((ci * k) as u64).saturating_mul(cols as u64));
        i0 = i1;
    }
    Ok(macs)
}
