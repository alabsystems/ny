// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #wave-beta-bab (ENGINE_PROGRAM §2 M6) — the margin-row tree as WAVES of
//! batched, UNTRUSTED beta-CROWN search with a STREAMING certify queue.
//!
//! # What
//!
//! `margin_row.tree_mode: wave` replaces the best-first frontier of
//! [`MarginRowBab::tree_loop_parallel`] (20 domains per step, each a dependent
//! chain of certified passes) with abc's wave shape on the lane's OUTER frame:
//!
//! 1. **One host pool, every class.** Each open tree class is a root domain in
//!    one LIFO pool (abc `bab_batch_size = total_num_or_spec`); a domain is a
//!    class row plus its split literals, its search multipliers (`beta`, one
//!    per literal) and its search slopes (`alpha`, over the universe of
//!    root-unstable trunk and head neurons, stored as `u16` fixed point and
//!    shared between siblings).
//! 2. **A wave** pops up to `margin_row.tree_wave_batch` domains and hands them
//!    to the injected [`SearchBackend`] (`backward_batch`: output-only
//!    beta-CROWN with the certified root boxes FROZEN and the splits PINNED,
//!    Adam on `(alpha, beta)`, keep-best). A domain whose search bound clears
//!    `-tau + delta` is CLAIMED (`tau = 4 gamma_f32(L) S`, `delta >= 0` raised
//!    while the certify queue refutes more than a fifth of its claims and
//!    lowered while it refutes under a twentieth); every other domain is SPLIT
//!    on the backend's `branch_scores` candidates (BaBSR intercept), several
//!    levels at once while the pool is thinner than a wave (`2^s` children).
//! 3. **The certify queue** runs on a PRIVATE rayon pool, concurrently with
//!    the next wave's search (double-buffered: wave `w`'s claims certify while
//!    wave `w + 1` searches, and are merged in claim order before wave `w + 2`
//!    pops — so the trajectory depends on neither the thread counts nor the
//!    timing). Each claim climbs the frozen-first LEAF LADDER, stopping at the
//!    first rung that certifies:
//!    * **frozen** — the lane's own certified domain evaluation on the ROOT
//!      y-pack (valid on every sub-region) with the domain's split gates, its
//!      clamped slope and multiplier hints: no network re-walk of the y rows;
//!    * **clip** — the same, with the y-box first re-concretised over the
//!      domain's split HALF-SPACES (`clip::halfspaces_for_splits`, the
//!      Clip-and-Verify necessary conditions; only when clip rows are retained
//!      and their frame check passed);
//!    * **y-pack** — the same on the domain's OWN certified y-pack (built from
//!      its split gates, cached by trunk path), i.e. today's per-domain
//!      evaluation;
//!    * otherwise **REFUTED**: the domain goes back to the pool and is SPLIT,
//!      never closed.
//! 4. **Closure.** A leaf closes only on `certify::margin_row::bound_evidence`
//!    / `empty_evidence` (M2's strict certified predicate on an Outward value,
//!    or a certified emptiness slack); its PATH is recorded in its class's
//!    [`CoverLedger`]. When the pool and the queue are empty, every class's
//!    ledger must PROVE its cover (`CoverLedger::prove`, the path-based gate)
//!    and the frame's `authorize_frame` (root / frozen-dual covers, G2, the
//!    class aggregation) must mint the token. Anything else is `Unknown`.
//!
//! The frozen-dual tier's hand-off needs no code here: on the outer frame the
//! tier runs first (`run_inner`) and every class it cannot close — its stall
//! — reaches the wave as a tree class.
//!
//! **Which search.** `wave` searches with the registered CPU f32 backend
//! (ny-bbe). `wave_cuda` searches with ONE composite backend,
//! `ny_core::search::FallbackBackend`: the registered DEVICE backend (ny-cuda's
//! wave kernels, bit for bit the CPU f32 search) with the CPU backend behind
//! it. The lane carries none of the fallback logic: the composite falls
//! through to the CPU search when the device's upload refuses (absent,
//! unavailable, failed self-test, lease), and switches the tree to it, once,
//! only on a DEVICE FAULT (`Refusal::Fault`, or a panic) — releasing the
//! device's buffers first and RE-RUNNING the faulted call on the CPU search, so
//! the switch changes speed, never the trajectory. Any other refusal (a
//! host-side validation refusal, the deadline) is that call's alone: the
//! device keeps the tree, the refused wave's fresh domains are claimed
//! unsearched (at the multipliers they carry) and, when its scoring refused,
//! its refuted domains split on the lane's own first-unsplit candidates;
//! certification decides every one of them. Only when every upload refuses
//! does the legacy tree run. The receipt names the EFFECTIVE backend(s) from
//! the composite's route (`FallbackBackend::route`): `backend=cuda` for a tree
//! searched on the device alone, `backend=cpu_f32 device_refused=<why>` after
//! an upload fall-through, and `backend=cuda->cpu_f32 switched_at_wave=<N>
//! switch_reason=<why>` after a mid-tree switch. The certify queue, the ladder
//! and the cover gate do not know which backend searched.
//!
//! # Trust (THE DEVICE NEVER CERTIFIES)
//!
//! The backend sees a plain-data copy of the lane IR and the certified boxes;
//! everything it returns is a HINT. Multipliers enter the certified pass only
//! through `beta::set_terms` / `beta::seed_with_head_terms` (finite and
//! positive, else dropped: weak duality makes every `beta >= 0` valid), slopes
//! only through `domain_alpha::write_alphas` / `head_gates_with` (clamped into
//! `[0, 1]`, `ms` re-derived: every such slope is a valid lower line),
//! branching candidates only after validation against the domain's own
//! literals (any split of a region covers it), and the search bound only
//! decides WHICH domains are certified. A search bug costs time; the published
//! verdict is the certified evaluator's and the cover gate's. The default
//! (`legacy`) never enters this module: the tree is bit-identical.
//!
//! # Observables
//!
//! Every armed outer frame prints exactly one
//! `[bbe] phase=wave armed=1 backend= ... mode= device_refused= device_switch=
//! switched_at_wave= switch_reason= first_refusal= wave-beta-bab/v1` summary
//! (also on a fallback, with `refused=`, `fallback=legacy`, `mode=` and
//! `device_refused=`); an armed run without it is VOID. A `wave_cuda` run
//! searched on the device ALONE only when `backend=` is exactly `cuda` (match
//! `backend=cuda ` with its trailing space: `backend=cuda->cpu_f32` starts with
//! the same bytes and did part of its search on the CPU). `first_refusal=`
//! names the first search call that refused (`backward_batch:<why>` or
//! `branch_scores:<why>`, a malformed output included); `search_refused=`
//! counts the waves with one. `wave-beta-bab/v1` is a contiguous literal for
//! binary pinning.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use ny_core::search::{
    BranchOut, DeviceLease, FallbackBackend, Refusal, ReluBox, SearchBackend, WaveBatch,
    WaveHandle, WaveOut, WaveProblem, WaveRoute, WaveSearchCfg, WaveSite, WaveSplit, WAVE_NONE,
};
use ny_core::{NyError, Result};
use rayon::prelude::*;

use super::{
    authorize_frame, build_pack, cover_fault_drop_next_leaf, ybox_crossing, HeadFix, Lru,
    MarginRowBab, MarginRowOutcome, RootEval, TrunkSplit, YPack,
};
use crate::certify::margin_row as cert_mr;
use crate::certify::{ClassCover, CoverLedger, FrameSplit, LeafClose, LeafEvidence, LeafHint};
use crate::margin_row::bounds::{
    compose_viay, head_gates_with, margin_seed, per_class_direct, MarginBatch, YBox,
};
use crate::margin_row::engine::{domain_gates, Collect, LaneDir};
use crate::margin_row::net::{TwinNet, TwinOp};
use crate::margin_row::root::RootGates;

// ---------------------------------------------------------------------------
//  The typed selector and its policy
// ---------------------------------------------------------------------------

/// Which tree engine the margin-row lane runs (`margin_row.tree_mode`).
///
/// The serde tokens are the typed preset key's values and the lever
/// declaration's `Enum` members ([`Self::TOKENS`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum TreeMode {
    /// Today's best-first tree (`legacy`, the default): bit-identical; this
    /// module is never entered.
    #[default]
    #[serde(rename = "legacy")]
    Legacy,
    /// The wave engine (M6) on the outer frame, searched by the registered
    /// CPU f32 backend (ny-bbe).
    #[serde(rename = "wave")]
    Wave,
    /// The wave engine (M6) on the outer frame, searched on the DEVICE
    /// (ny-cuda's `CudaSearchBackend`, bit for bit the CPU f32 search) when
    /// its factory is registered and its upload is admitted; a device that is
    /// absent, unavailable, fails its self-test or refuses the lease falls
    /// through to the CPU f32 backend (`backend=cpu_f32 device_refused=...`
    /// on the receipt), and only then to the legacy tree. The certify queue,
    /// the cover gate and every close are the same as `wave`'s.
    #[serde(rename = "wave_cuda")]
    WaveCuda,
}

impl TreeMode {
    /// The preset / environment vocabulary, in declaration order.
    pub const TOKENS: [&'static str; 3] = ["legacy", "wave", "wave_cuda"];

    /// Parse one canonical token (exact spelling).
    #[must_use]
    pub fn from_token(token: &str) -> Option<Self> {
        match token {
            "legacy" => Some(Self::Legacy),
            "wave" => Some(Self::Wave),
            "wave_cuda" => Some(Self::WaveCuda),
            _ => None,
        }
    }

    /// The canonical token.
    #[must_use]
    pub const fn token(self) -> &'static str {
        match self {
            Self::Legacy => "legacy",
            Self::Wave => "wave",
            Self::WaveCuda => "wave_cuda",
        }
    }

    /// The search backends this mode asks for, in order (the first whose
    /// upload is admitted searches the tree).
    #[must_use]
    pub const fn backends(self) -> &'static [WaveBackendKind] {
        match self {
            Self::Legacy => &[],
            Self::Wave => &[WaveBackendKind::CpuF32],
            Self::WaveCuda => &[WaveBackendKind::Cuda, WaveBackendKind::CpuF32],
        }
    }
}

/// Which registered wave-search backend (`TreeMode::backends`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaveBackendKind {
    /// ny-bbe's CPU f32 search (`set_wave_search_backend_factory`).
    CpuF32,
    /// ny-cuda's device search (`set_wave_cuda_search_backend_factory`).
    Cuda,
}

/// Declaration default of `margin_row.tree_wave_batch`: domains per wave.
pub const WAVE_BATCH_DEFAULT: usize = 256;

/// Reader clamp of `margin_row.tree_wave_batch`.
pub const WAVE_BATCH_MAX: usize = 16_384;

/// #router-r6 `rule.trial_width`: the lever whose declared grid span bounds a
/// wave-width trial (`[64, 1024]` today; the trial also stays inside
/// `[1, WAVE_BATCH_MAX]` and within a factor of 4 of the frozen width).
pub(crate) fn wave_width_lever() -> &'static str {
    ny_levers::decls::margin_row::MARGIN_ROW_TREE_WAVE_BATCH.name
}

/// Reader clamp of `margin_row.tree_wave_cert_threads`.
pub const WAVE_CERT_THREADS_MAX: usize = 1024;

/// Adam steps per domain per wave (abc's BaB `iteration` 8-10). A trusted
/// code constant, not a key: the program allows M6 three keys.
pub(crate) const WAVE_BETA_ITERS: usize = 8;

/// Adam step on the slopes (abc BaB `lr_alpha`).
pub(crate) const WAVE_LR_ALPHA: f64 = 0.1;

/// Adam step on the split multipliers (abc BaB `lr_beta`).
pub(crate) const WAVE_LR_BETA: f64 = 0.05;

/// Most split levels one domain takes in one wave (`2^3` children).
pub(crate) const WAVE_MAX_LEVELS: usize = 3;

/// Refutation rate above which the claim offset `delta` rises, and below which
/// it falls (the program's `refuted/claimed <= 20%` target).
const WAVE_REFUTE_HIGH: f64 = 0.20;
const WAVE_REFUTE_LOW: f64 = 0.05;

/// Byte ceiling of the whole wave engine (search lease + host pool) before the
/// headroom clamp.
const WAVE_MAX_BYTES: u64 = 4 << 30;

/// The frozen tree-engine policy of one lane invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WavePolicy {
    /// Legacy or wave.
    pub mode: TreeMode,
    /// Domains per wave (`margin_row.tree_wave_batch`).
    pub batch: usize,
    /// Private certify-pool threads (`margin_row.tree_wave_cert_threads`, 0 =
    /// half of the lane's rayon threads).
    pub cert_threads: usize,
}

impl Default for WavePolicy {
    fn default() -> Self {
        Self {
            mode: TreeMode::Legacy,
            batch: WAVE_BATCH_DEFAULT,
            cert_threads: 0,
        }
    }
}

impl WavePolicy {
    /// Resolve the three typed keys from the run's already-frozen,
    /// receipt-owning lever set and clamp them.
    ///
    /// This is the ONE authority on the scored route: the flight receipt
    /// serialises exactly this set (`FlightRecorder::materialize_levers`), and
    /// `MarginRowRuntimePolicy::from_levers` hands the lane this value, so the
    /// tree the lane runs and the tree the receipt names cannot disagree. (Before
    /// this, the lane read the bridge-written governed cells while the receipt
    /// read the frozen set; they agreed only when `preset_gates::apply` had run.)
    /// A missing or ill-typed entry is an error, never a silent fallback to a
    /// second authority.
    pub fn from_levers(levers: &ny_levers::LeverSet) -> std::result::Result<Self, String> {
        use ny_levers::decls::margin_row as d;

        let resolved = |decl: &'static ny_levers::LeverDecl| {
            levers
                .resolved(decl)
                .ok_or_else(|| format!("{} is missing from the frozen LeverSet", decl.name))
        };
        let platform_usize = |decl: &'static ny_levers::LeverDecl| {
            resolved(decl)?
                .value
                .as_u64()
                .and_then(|number| usize::try_from(number).ok())
                .ok_or_else(|| format!("{} is not a platform-sized unsigned integer", decl.name))
        };
        let mode = resolved(&d::MARGIN_ROW_TREE_MODE)?
            .value
            .as_str()
            .and_then(TreeMode::from_token)
            .ok_or_else(|| {
                format!(
                    "{} is not one of {:?}",
                    d::MARGIN_ROW_TREE_MODE.name,
                    TreeMode::TOKENS
                )
            })?;
        Ok(Self {
            mode,
            batch: platform_usize(&d::MARGIN_ROW_TREE_WAVE_BATCH)?,
            cert_threads: platform_usize(&d::MARGIN_ROW_TREE_WAVE_CERT_THREADS)?,
        }
        .clamped())
    }

    /// Compatibility resolver for callers WITHOUT a scored frozen lever set
    /// (`MarginRowRuntimePolicy::from_process_environment`, research
    /// subcommands): the three typed keys (environment, then the preset cell,
    /// then the declaration defaults), clamped. An unknown mode token is
    /// `legacy`. The scored lane never calls this: it receives
    /// [`Self::from_levers`]'s value.
    #[must_use]
    pub fn from_gates() -> Self {
        let mode = crate::preset_gates::margin_row_tree_mode()
            .as_deref()
            .and_then(TreeMode::from_token)
            .unwrap_or_default();
        Self {
            mode,
            batch: crate::preset_gates::margin_row_tree_wave_batch(),
            cert_threads: crate::preset_gates::margin_row_tree_wave_cert_threads(),
        }
        .clamped()
    }

    /// The reader clamps.
    #[must_use]
    pub fn clamped(self) -> Self {
        Self {
            mode: self.mode,
            batch: self.batch.clamp(1, WAVE_BATCH_MAX),
            cert_threads: self.cert_threads.min(WAVE_CERT_THREADS_MAX),
        }
    }

    /// The wave engine is selected (on the CPU or the device).
    #[must_use]
    pub fn armed(&self) -> bool {
        matches!(self.mode, TreeMode::Wave | TreeMode::WaveCuda)
    }
}

// ---------------------------------------------------------------------------
//  The injected backend
// ---------------------------------------------------------------------------

type BackendFactory = Box<dyn Fn() -> Option<Arc<dyn SearchBackend>> + Send + Sync>;

static FACTORY: OnceLock<BackendFactory> = OnceLock::new();
static CUDA_FACTORY: OnceLock<BackendFactory> = OnceLock::new();

/// Register the process's CPU wave-search backend factory (ny-cli, once, at
/// startup; `tree_mode: wave`, and `wave_cuda`'s fall-through). First install
/// wins. The closure runs at most once, on the first ARMED wave tree that asks
/// for it, so a dark lane constructs nothing.
pub fn set_wave_search_backend_factory<F>(factory: F)
where
    F: Fn() -> Option<Arc<dyn SearchBackend>> + Send + Sync + 'static,
{
    let _ = FACTORY.set(Box::new(factory));
}

/// Register the process's DEVICE wave-search backend factory (ny-cli with the
/// `cuda` feature, once, at startup; `tree_mode: wave_cuda`). First install
/// wins; the closure runs at most once, on the first `wave_cuda` tree. The
/// backend it builds touches no driver until its first upload.
pub fn set_wave_cuda_search_backend_factory<F>(factory: F)
where
    F: Fn() -> Option<Arc<dyn SearchBackend>> + Send + Sync + 'static,
{
    let _ = CUDA_FACTORY.set(Box::new(factory));
}

fn registered_backend(kind: WaveBackendKind) -> Option<Arc<dyn SearchBackend>> {
    static HANDLE: OnceLock<Arc<dyn SearchBackend>> = OnceLock::new();
    static CUDA_HANDLE: OnceLock<Arc<dyn SearchBackend>> = OnceLock::new();
    let (factory, handle) = match kind {
        WaveBackendKind::CpuF32 => (&FACTORY, &HANDLE),
        WaveBackendKind::Cuda => (&CUDA_FACTORY, &CUDA_HANDLE),
    };
    if let Some(backend) = handle.get() {
        return Some(Arc::clone(backend));
    }
    let built = factory.get().and_then(|factory| factory())?;
    Some(Arc::clone(handle.get_or_init(|| built)))
}

#[cfg(test)]
thread_local! {
    static TEST_BACKEND: std::cell::RefCell<Option<Arc<dyn SearchBackend>>> =
        const { std::cell::RefCell::new(None) };
    static TEST_DEVICE_BACKEND: std::cell::RefCell<Option<Arc<dyn SearchBackend>>> =
        const { std::cell::RefCell::new(None) };
    static TEST_HOOKS: std::cell::Cell<WaveHooks> =
        const { std::cell::Cell::new(WaveHooks::OFF) };
    static TEST_LEAVES: std::cell::RefCell<Vec<RecordedLeaf>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

thread_local! {
    /// The last `[bbe] phase=wave` summary this THREAD printed (the wave tree
    /// runs on the lane's calling thread). Telemetry only: nothing reads it
    /// but tests and receipts.
    static LAST_WAVE_SUMMARY: std::cell::RefCell<Option<String>> =
        const { std::cell::RefCell::new(None) };
}

/// Take the last `[bbe] phase=wave ... wave-beta-bab/v1` summary line printed
/// on this thread (`None` when no armed wave tree ran here since the last
/// take). Telemetry only — the receipt a caller can check that the wave
/// engine (and not its legacy fallback) ran.
pub fn take_last_wave_summary() -> Option<String> {
    LAST_WAVE_SUMMARY.with(|c| c.borrow_mut().take())
}

fn note_wave_summary(line: String) {
    eprintln!("{line}");
    LAST_WAVE_SUMMARY.with(|c| *c.borrow_mut() = Some(line));
}

/// The backend of `kind`: a test's thread-local injection, else the
/// registered one.
fn wave_backend(kind: WaveBackendKind) -> Option<Arc<dyn SearchBackend>> {
    #[cfg(test)]
    {
        let injected = match kind {
            WaveBackendKind::CpuF32 => TEST_BACKEND.with(|c| c.borrow().clone()),
            WaveBackendKind::Cuda => TEST_DEVICE_BACKEND.with(|c| c.borrow().clone()),
        };
        if let Some(b) = injected {
            return Some(b);
        }
    }
    registered_backend(kind)
}

/// TEST ONLY: [`with_test_backend`] with a DEVICE backend injected as well
/// (the `wave_cuda` mode's first choice).
#[cfg(test)]
pub(crate) fn with_test_backends<T>(
    cpu: Arc<dyn SearchBackend>,
    device: Arc<dyn SearchBackend>,
    hooks: WaveHooks,
    f: impl FnOnce() -> T,
) -> (T, Vec<RecordedLeaf>) {
    TEST_DEVICE_BACKEND.with(|c| *c.borrow_mut() = Some(device));
    let out = with_test_backend(cpu, hooks, f);
    TEST_DEVICE_BACKEND.with(|c| *c.borrow_mut() = None);
    out
}

/// TEST ONLY: run `f` with `backend` injected as this thread's wave backend
/// and `hooks` as its mutation hooks; returns `f`'s result and every leaf the
/// wave closed (in merge order).
#[cfg(test)]
pub(crate) fn with_test_backend<T>(
    backend: Arc<dyn SearchBackend>,
    hooks: WaveHooks,
    f: impl FnOnce() -> T,
) -> (T, Vec<RecordedLeaf>) {
    TEST_BACKEND.with(|c| *c.borrow_mut() = Some(backend));
    TEST_HOOKS.with(|c| c.set(hooks));
    TEST_LEAVES.with(|c| c.borrow_mut().clear());
    let out = f();
    TEST_BACKEND.with(|c| *c.borrow_mut() = None);
    TEST_HOOKS.with(|c| c.set(WaveHooks::OFF));
    let leaves = TEST_LEAVES.with(|c| std::mem::take(&mut *c.borrow_mut()));
    (out, leaves)
}

/// Mutation hooks (read only under `cfg(test)`; all off in production, where
/// every branch they guard is `cfg!(test) && ...` and folds away).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct WaveHooks {
    /// MUTANT: a claim whose search bound is positive closes on that search
    /// bound, uncertified.
    pub(crate) claims_are_closes: bool,
    /// MUTANT: the clip rung re-concretises over the OPPOSITE half-space of
    /// every split.
    pub(crate) flip_clip_side: bool,
    /// Every certification returns an error (an error must never close).
    pub(crate) certify_errors: bool,
}

impl WaveHooks {
    /// All hooks off.
    pub(crate) const OFF: Self = Self {
        claims_are_closes: false,
        flip_clip_side: false,
        certify_errors: false,
    };

    fn capture() -> Self {
        #[cfg(test)]
        {
            TEST_HOOKS.with(std::cell::Cell::get)
        }
        #[cfg(not(test))]
        {
            Self::OFF
        }
    }
}

/// TEST ONLY: one closed leaf of a wave tree, for containment checks.
#[cfg(test)]
#[derive(Debug, Clone)]
pub(crate) struct RecordedLeaf {
    /// Tree class.
    pub(crate) class: usize,
    /// Trunk splits.
    pub(crate) trunk: Vec<TrunkSplit>,
    /// Head fixes.
    pub(crate) heads: Vec<HeadFix>,
    /// Certified value (bound or emptiness slack).
    pub(crate) value: f64,
    /// Whether the leaf is an emptiness close.
    pub(crate) empty: bool,
    /// The ladder rung (`frozen`, `clip`, `ypack`, `mutant`).
    pub(crate) rung: &'static str,
}

// ---------------------------------------------------------------------------
//  The universe: the lane's own splittable neurons
// ---------------------------------------------------------------------------

/// One splittable neuron in the lane's own IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Site {
    /// Trunk ReLU `(layer, position in the layer's unstable list)`.
    Trunk {
        /// Trunk layer.
        li: usize,
        /// Unstable-list position.
        pos: usize,
    },
    /// Head neuron.
    Head {
        /// Head index.
        index: usize,
    },
}

/// The universe shared with the backend: every root-unstable trunk neuron
/// (certified box) and every head neuron unstable in the root y-box.
pub(crate) struct Universe {
    pub(crate) sites: Vec<Site>,
    search: Vec<WaveSite>,
    /// Per trunk layer: neuron index -> universe position ([`WAVE_NONE`]).
    trunk_pos: Vec<Vec<u32>>,
    /// Head neuron -> universe position ([`WAVE_NONE`]).
    head_pos: Vec<u32>,
    alpha0: Vec<f64>,
    head_op: usize,
}

impl Universe {
    pub(crate) fn build(
        net: &TwinNet,
        root: &RootGates,
        ly0: &[f64],
        uy0: &[f64],
    ) -> std::result::Result<Self, String> {
        let head_op = net.i_gemm1 + 1;
        if !matches!(net.ops.get(head_op), Some(TwinOp::Relu { .. })) {
            return Err("the head ReLU is not at i_gemm1 + 1".into());
        }
        if ly0.len() != net.n_y || uy0.len() != net.n_y {
            return Err("root y-box width differs from the head".into());
        }
        let mut sites = Vec::new();
        let mut search = Vec::new();
        let mut alpha0 = Vec::new();
        let mut trunk_pos = Vec::with_capacity(root.layers.len());
        for (li, layer) in root.layers.iter().enumerate() {
            let mut map = vec![WAVE_NONE; layer.n];
            for (pos, &idx) in layer.unst.iter().enumerate() {
                if idx >= layer.n || !(layer.l[idx] < 0.0 && layer.u[idx] > 0.0) {
                    continue;
                }
                map[idx] = u32::try_from(sites.len()).map_err(|_| "universe too large")?;
                sites.push(Site::Trunk { li, pos });
                search.push(WaveSite {
                    op: u32::try_from(layer.op).map_err(|_| "op index too large")?,
                    neuron: u32::try_from(idx).map_err(|_| "neuron index too large")?,
                });
                alpha0.push(layer.alpha.get(idx).copied().unwrap_or(f64::NAN));
            }
            trunk_pos.push(map);
        }
        let mut head_pos = vec![WAVE_NONE; net.n_y];
        for i in 0..net.n_y {
            if ly0[i] < 0.0 && uy0[i] > 0.0 {
                head_pos[i] = u32::try_from(sites.len()).map_err(|_| "universe too large")?;
                sites.push(Site::Head { index: i });
                search.push(WaveSite {
                    op: u32::try_from(head_op).map_err(|_| "op index too large")?,
                    neuron: u32::try_from(i).map_err(|_| "neuron index too large")?,
                });
                alpha0.push(
                    root.head_alpha
                        .as_ref()
                        .and_then(|h| h.get(i).copied())
                        .unwrap_or(f64::NAN),
                );
            }
        }
        if sites.is_empty() {
            return Err("no unstable neuron to split".into());
        }
        if u32::try_from(sites.len()).is_ok_and(|n| n == WAVE_NONE) {
            return Err("universe too large".into());
        }
        Ok(Self {
            sites,
            search,
            trunk_pos,
            head_pos,
            alpha0,
            head_op,
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.sites.len()
    }

    /// The search copy of this tree.
    fn problem(
        &self,
        net: &TwinNet,
        root: &RootGates,
        ly0: &[f64],
        uy0: &[f64],
        t: usize,
        classes: &[usize],
    ) -> std::result::Result<WaveProblem, String> {
        let mut boxes = super::super::root_search::reference_boxes(root);
        boxes.push(ReluBox {
            op: self.head_op,
            l: ly0.to_vec(),
            u: uy0.to_vec(),
        });
        let problem = WaveProblem {
            net: super::super::root_search::lower_twin_net(net)?,
            boxes,
            universe: self.search.clone(),
            alpha0: self.alpha0.clone(),
            lo: root.lo.clone(),
            hi: root.hi.clone(),
            target: t,
            classes: classes.to_vec(),
        };
        problem.validate()?;
        Ok(problem)
    }

    /// A domain's literals in the lane's IR: trunk splits and head fixes with
    /// their multipliers (as f64; sanitised by the certified installers).
    fn split_lists(
        &self,
        lits: &[WaveSplit],
        beta: &[f32],
    ) -> (Vec<TrunkSplit>, Vec<HeadFix>, Vec<f64>, Vec<f64>) {
        let (mut trunk, mut heads, mut tb, mut hb) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (k, lit) in lits.iter().enumerate() {
            let dir: i8 = if lit.active { 1 } else { -1 };
            let b = beta.get(k).map_or(0.0, |&b| f64::from(b));
            match self.sites[lit.pos as usize] {
                Site::Trunk { li, pos } => {
                    trunk.push((li, pos, dir));
                    tb.push(b);
                }
                Site::Head { index } => {
                    heads.push((index, dir));
                    hb.push(b);
                }
            }
        }
        (trunk, heads, tb, hb)
    }

    /// B2.3: a domain's slopes as a proof-bundle α table `(Relu op,
    /// neurons, slopes)` over the universe: the hint where it set one
    /// (finite), else the root's own slope; non-finite or out-of-range values
    /// are dropped (hint data only — the checker clamps whatever it reads).
    pub(crate) fn leaf_alpha_table(&self, hint: &[f64]) -> Vec<(u32, Vec<u32>, Vec<f64>)> {
        let mut by_op: std::collections::BTreeMap<u32, (Vec<u32>, Vec<f64>)> =
            std::collections::BTreeMap::new();
        for (p, site) in self.search.iter().enumerate() {
            let v = hint
                .get(p)
                .copied()
                .filter(|v| v.is_finite())
                .or_else(|| self.alpha0.get(p).copied())
                .filter(|v| v.is_finite() && (0.0..=1.0).contains(v));
            if let Some(v) = v {
                let e = by_op.entry(site.op).or_default();
                e.0.push(site.neuron);
                e.1.push(v);
            }
        }
        by_op
            .into_iter()
            .map(|(op, (idx, val))| (op, idx, val))
            .collect()
    }

    /// The head slopes a hint implies: the root's own where the hint is
    /// silent (NaN), else the hint (clamped by `head_gates_with`).
    fn head_alpha(&self, hint: &[f64], root: &RootGates) -> Vec<f64> {
        let mut out: Vec<f64> = match root.head_alpha.as_ref() {
            Some(h) if h.len() == self.head_pos.len() => h.clone(),
            _ => vec![f64::NAN; self.head_pos.len()],
        };
        for (i, &p) in self.head_pos.iter().enumerate() {
            if p != WAVE_NONE {
                let h = hint[p as usize];
                if h.is_finite() {
                    out[i] = h;
                }
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
//  Host pool entries and slope storage
// ---------------------------------------------------------------------------

/// `u16` code of "no slope hint" (the problem's own warm slope).
const ALPHA_UNSET: u16 = u16::MAX;
/// `u16` code of slope 1.
const ALPHA_ONE: f32 = 65534.0;

fn pack_alpha(a: f32) -> u16 {
    if a.is_finite() {
        (a.clamp(0.0, 1.0) * ALPHA_ONE).round() as u16
    } else {
        ALPHA_UNSET
    }
}

fn unpack_alpha(q: u16) -> f32 {
    if q == ALPHA_UNSET {
        f32::NAN
    } else {
        f32::from(q) / ALPHA_ONE
    }
}

/// One sub-domain of the wave tree.
#[derive(Clone)]
struct Dom {
    /// Row of the tree's class list.
    row: u32,
    /// Split literals, in split order.
    lits: Vec<WaveSplit>,
    /// Search multipliers, aligned with `lits`.
    beta: Vec<f32>,
    /// Search slopes (`u16` fixed point), shared with siblings; `None` = the
    /// problem's warm slopes.
    alpha: Option<Arc<[u16]>>,
    /// The certify queue refuted this domain: split it, never re-claim it.
    refuted: bool,
}

impl Dom {
    fn root(row: u32) -> Self {
        Self {
            row,
            lits: Vec::new(),
            beta: Vec::new(),
            alpha: None,
            refuted: false,
        }
    }

    fn alpha_f64(&self) -> Option<Vec<f64>> {
        self.alpha
            .as_ref()
            .map(|a| a.iter().map(|&q| f64::from(unpack_alpha(q))).collect())
    }

    /// Host bytes of this entry (the slopes counted in full, as if unshared).
    fn bytes(&self, stride: usize) -> u64 {
        (64 + 12 * self.lits.len() + 2 * stride) as u64
    }

    /// The `2^m` children over `cands` (fresh multipliers 0, the parent's
    /// slopes shared).
    fn children(&self, cands: &[u32]) -> Vec<Self> {
        let m = cands.len();
        (0..(1usize << m))
            .map(|mask| {
                let mut lits = self.lits.clone();
                let mut beta = self.beta.clone();
                for (b, &pos) in cands.iter().enumerate() {
                    lits.push(WaveSplit {
                        pos,
                        active: (mask >> b) & 1 == 1,
                    });
                    beta.push(0.0);
                }
                Self {
                    row: self.row,
                    lits,
                    beta,
                    alpha: self.alpha.clone(),
                    refuted: false,
                }
            })
            .collect()
    }
}

/// A wave packed for the backend.
struct Packed {
    rows: Vec<u32>,
    offsets: Vec<u32>,
    splits: Vec<WaveSplit>,
    beta: Vec<f32>,
    alpha: Vec<f32>,
    stride: usize,
}

impl Packed {
    fn of(doms: &[Dom], stride: usize) -> Self {
        let any_alpha = doms.iter().any(|d| d.alpha.is_some());
        let mut p = Self {
            rows: Vec::with_capacity(doms.len()),
            offsets: Vec::with_capacity(doms.len() + 1),
            splits: Vec::new(),
            beta: Vec::new(),
            alpha: Vec::new(),
            stride,
        };
        p.offsets.push(0);
        for d in doms {
            p.rows.push(d.row);
            p.splits.extend_from_slice(&d.lits);
            p.beta.extend_from_slice(&d.beta);
            p.offsets.push(p.splits.len() as u32);
            if any_alpha {
                match &d.alpha {
                    Some(a) => p.alpha.extend(a.iter().map(|&q| unpack_alpha(q))),
                    None => p.alpha.resize(p.alpha.len() + stride, f32::NAN),
                }
            }
        }
        p
    }

    fn batch(&self, deadline: Option<Instant>) -> WaveBatch<'_> {
        WaveBatch {
            count: self.rows.len(),
            rows: &self.rows,
            offsets: &self.offsets,
            splits: &self.splits,
            beta: &self.beta,
            alpha: &self.alpha,
            stride: self.stride,
            deadline,
        }
    }

    fn range(&self, k: usize) -> std::ops::Range<usize> {
        self.offsets[k] as usize..self.offsets[k + 1] as usize
    }
}

/// `gamma_k` of binary32 (`k u / (1 - k u)`, `u = 2^-24`).
fn gamma_f32(k: usize) -> f64 {
    let ku = k as f64 * f64::from(f32::EPSILON) / 2.0;
    if ku < 1.0 {
        ku / (1.0 - ku)
    } else {
        f64::INFINITY
    }
}

// ---------------------------------------------------------------------------
//  The certify ladder (TCB composition: existing certified primitives only)
// ---------------------------------------------------------------------------

/// The ladder rung that certified a leaf.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Rung {
    Frozen,
    Clip,
    YPack,
}

impl Rung {
    const fn token(self) -> &'static str {
        match self {
            Self::Frozen => "frozen",
            Self::Clip => "clip",
            Self::YPack => "ypack",
        }
    }
}

/// One rung's certified value.
enum LeafValue {
    /// The domain's certified margin lower bound (Outward).
    Bound(f64),
    /// The domain's region is certified empty with this slack.
    Empty(f64),
}

/// A claim's certification.
enum CertOutcome {
    Closed {
        evidence: LeafEvidence,
        rung: Rung,
        value: f64,
        empty: bool,
    },
    Refuted,
    Deadline,
}

/// Everything the certify queue reads (shared, read-only, across its pool).
struct CertCtx<'c> {
    universe: &'c Universe,
    root_pack: &'c Arc<YPack>,
    mbs: &'c [MarginBatch],
    cache: &'c Mutex<Lru>,
    hooks: WaveHooks,
}

/// Whether the clip rung may run on `trunk`: splits to cut with, retained
/// clip rows, and the fail-closed frame check passed.
fn clip_admissible(root: &RootGates, trunk: &[TrunkSplit]) -> bool {
    !trunk.is_empty()
        && crate::margin_row::clip::any_rows_retained(root)
        && crate::margin_row::clip::frame_ok(root)
}

/// Re-concretise `ybox` (the pack's own concretisation, or a copy of it)
/// over the half-spaces the trunk splits imply: exactly `clip_tighten_pack`'s
/// arithmetic, applied to a LOCAL box (the pack is shared). Shrink-only: every
/// delta is `>= 0` by construction, and a refused set leaves the box
/// untouched. `flip` is the side mutation pin (test only). Returns the number
/// of moved sides.
fn clip_ybox(
    root: &RootGates,
    pack: &YPack,
    trunk: &[TrunkSplit],
    ybox: &mut YBox,
    flip: bool,
) -> usize {
    let flipped: Vec<TrunkSplit>;
    let splits: &[TrunkSplit] = if cfg!(test) && flip {
        flipped = trunk.iter().map(|&(li, pos, d)| (li, pos, -d)).collect();
        &flipped
    } else {
        trunk
    };
    let hs = crate::margin_row::clip::halfspaces_for_splits(root, splits);
    if hs.is_empty() {
        return 0;
    }
    let (mid, rad) = (root.mid.as_slice(), root.rad.as_slice());
    let mut moved = 0usize;
    if let Some(dl) = crate::margin_row::clip::ybox_deltas(&pack.al.a, &hs, mid, rad, true) {
        if dl.len() == ybox.ly.len() {
            for (v, d) in ybox.ly.iter_mut().zip(&dl) {
                *v += *d;
                moved += usize::from(*d > 0.0);
            }
        }
    }
    if let Some(du) = crate::margin_row::clip::ybox_deltas(&pack.au.a, &hs, mid, rad, false) {
        if du.len() == ybox.uy.len() {
            for (v, d) in ybox.uy.iter_mut().zip(&du) {
                *v -= *d;
                moved += usize::from(*d > 0.0);
            }
        }
    }
    moved
}

/// TEST ONLY: the root y-box of `net` / `root` and its clip-rung
/// re-concretisation over `trunk` (optionally side-flipped), bypassing the
/// process-wide frame latch (the test checks the frame itself).
#[cfg(test)]
pub(crate) fn clip_ybox_for_test(
    net: &TwinNet,
    root: &RootGates,
    trunk: &[TrunkSplit],
    flip: bool,
) -> Result<(YBox, YBox, usize)> {
    let eng = crate::margin_row::engine::BackwardEngine::new(net, root);
    let pack = build_pack(&eng, None, &[])?;
    let base = YBox {
        ly: pack.ly0.clone(),
        uy: pack.uy0.clone(),
    };
    let mut clipped = base.clone();
    let moved = clip_ybox(root, &pack, trunk, &mut clipped, flip);
    Ok((base, clipped, moved))
}

impl MarginRowBab<'_> {
    /// One rung: the lane's certified domain evaluation (the MAIN pass of
    /// `eval_with_pack` — split gates, y-box, head gates, margin seed, one
    /// Outward pass, `max(direct, m1, via-y)` per column, min over columns)
    /// on `pack`, with the domain's hints installed through the certified
    /// installers. No ascent, no retirement: nothing here decides anything
    /// but this domain's certified value.
    ///
    /// `clipped` is the clip rung's y-box: `pack`'s own concretisation
    /// re-concretised over the domain's split half-spaces by [`clip_ybox`]
    /// (shrink-only, computed by the caller so a clip that moves nothing can
    /// skip the pass). `None` is `pack`'s own concretisation.
    #[allow(clippy::too_many_arguments)]
    fn wave_certify_on(
        &self,
        pack: &YPack,
        mb: &MarginBatch,
        trunk: &[TrunkSplit],
        heads: &[HeadFix],
        trunk_beta: &[f64],
        head_beta: &[f64],
        alpha: Option<&[f64]>,
        universe: &Universe,
        clipped: Option<YBox>,
    ) -> Result<LeafValue> {
        let root = self.eng.root;
        let mode = root.mode;
        let mut dom = domain_gates(root, trunk);
        if dom.infeasible {
            return Ok(LeafValue::Empty(dom.infeasible_slack));
        }
        let mut ybox = clipped.unwrap_or_else(|| YBox {
            ly: pack.ly0.clone(),
            uy: pack.uy0.clone(),
        });
        ybox.clamp(heads);
        if ybox.is_empty() {
            return Ok(LeafValue::Empty(ybox_crossing(&ybox)));
        }
        // Slope hints: head through `head_gates_with` (clamped, unstable-only),
        // trunk through `domain_alpha::write_alphas` (clamped, `ms` re-derived)
        // on the domain's movable slots only (never a split-pinned neuron).
        let head_alpha = alpha.map(|a| universe.head_alpha(a, root));
        let gates = head_gates_with(
            &ybox,
            mode,
            head_alpha.as_deref().or(root.head_alpha.as_deref()),
        );
        let ms = margin_seed(mb, &gates, &ybox, mode);
        if let Some(a) = alpha {
            let slots = crate::margin_row::domain_alpha::movable(root, &dom, trunk);
            if !slots.is_empty() {
                let mut vals = crate::margin_row::domain_alpha::read_alphas(root, &dom, &slots);
                for (v, slot) in vals.iter_mut().zip(&slots) {
                    let p = universe
                        .trunk_pos
                        .get(slot.li)
                        .and_then(|m| m.get(slot.idx))
                        .copied()
                        .unwrap_or(WAVE_NONE);
                    if p != WAVE_NONE {
                        let h = a[p as usize];
                        if h.is_finite() {
                            *v = h;
                        }
                    }
                }
                crate::margin_row::domain_alpha::write_alphas(root, &mut dom, &slots, &vals);
            }
        }
        // Multiplier hints (finite and > 0, else dropped, by the installers).
        crate::margin_row::beta::set_terms(&mut dom, root, trunk, trunk_beta);
        let shifted = if head_beta.iter().any(|&b| b.is_finite() && b > 0.0) {
            crate::margin_row::beta::seed_with_head_terms(
                &ms.seed,
                heads,
                head_beta,
                mode.outward(),
            )
        } else {
            None
        };
        let pass = self.eng.run_collect(
            shifted.as_ref().unwrap_or(&ms.seed),
            Some(&dom),
            LaneDir::Lower,
            None,
            Collect {
                unst_abs: false,
                rows: None,
                unst_rows: false,
            },
        )?;
        let nf = mb.nf();
        let per_j = per_class_direct(&self.eng, &pass, &ms, 0..nf);
        let m2v = compose_viay(
            &self.eng,
            mb,
            &gates,
            &pack.al,
            &pack.au,
            &pack.al_dots,
            &pack.au_dots,
            mode,
        );
        let mut b = f64::INFINITY;
        for r in 0..nf {
            let pj = per_j[r].max(ms.m1[r]).max(m2v[r]);
            if !(per_j[r].is_finite()
                && ms.m1[r].is_finite()
                && m2v[r].is_finite()
                && pj.is_finite())
            {
                return Err(NyError::NumericalInstability(
                    "margin_row wave: non-finite per-class leaf component".into(),
                ));
            }
            b = b.min(pj);
        }
        if !b.is_finite() {
            return Err(NyError::NumericalInstability(
                "margin_row wave: non-finite leaf bound".into(),
            ));
        }
        Ok(LeafValue::Bound(b))
    }

    /// The domain's own certified y-pack, cached by trunk path.
    fn wave_pack(&self, cache: &Mutex<Lru>, trunk: &[TrunkSplit]) -> Result<Arc<YPack>> {
        if let Some(pack) = cache.lock().ok().and_then(|mut c| c.get(trunk)) {
            return Ok(pack);
        }
        let dom = domain_gates(self.eng.root, trunk);
        let pack = Arc::new(build_pack(&self.eng, Some(&dom), trunk)?);
        if let Ok(mut c) = cache.lock() {
            c.put(trunk.to_vec(), Arc::clone(&pack));
        }
        Ok(pack)
    }

    /// The frozen -> clip -> y-pack ladder for one claim.
    fn wave_certify(&self, ctx: &CertCtx<'_>, dom: &Dom) -> Result<CertOutcome> {
        if cfg!(test) && ctx.hooks.certify_errors {
            return Err(NyError::NumericalInstability(
                "margin_row wave: injected certification error".into(),
            ));
        }
        let mode = self.eng.root.mode;
        let (trunk, heads, tb, hb) = ctx.universe.split_lists(&dom.lits, &dom.beta);
        let alpha = dom.alpha_f64();
        let mb = &ctx.mbs[dom.row as usize];
        for rung in [Rung::Frozen, Rung::Clip, Rung::YPack] {
            if self.deadline_hit() {
                return Ok(CertOutcome::Deadline);
            }
            let mut clipped: Option<YBox> = None;
            let pack = match rung {
                Rung::Frozen => Arc::clone(ctx.root_pack),
                Rung::Clip => {
                    if !clip_admissible(self.eng.root, &trunk) {
                        continue;
                    }
                    // The rung's y-box: the root pack's concretisation
                    // re-concretised over the split half-spaces. A clip that
                    // moves no side IS the frozen rung's evaluation bit for
                    // bit (same pack, gates and hints), which just refuted:
                    // skip the duplicate certified pass.
                    let mut ybox = YBox {
                        ly: ctx.root_pack.ly0.clone(),
                        uy: ctx.root_pack.uy0.clone(),
                    };
                    let moved = clip_ybox(
                        self.eng.root,
                        ctx.root_pack,
                        &trunk,
                        &mut ybox,
                        ctx.hooks.flip_clip_side,
                    );
                    if moved == 0 {
                        continue;
                    }
                    clipped = Some(ybox);
                    Arc::clone(ctx.root_pack)
                }
                Rung::YPack => {
                    if trunk.is_empty() {
                        // The root pack IS this domain's pack: rung 1 ran it.
                        continue;
                    }
                    self.wave_pack(ctx.cache, &trunk)?
                }
            };
            let value = self.wave_certify_on(
                &pack,
                mb,
                &trunk,
                &heads,
                &tb,
                &hb,
                alpha.as_deref(),
                ctx.universe,
                clipped,
            )?;
            let (evidence, v, empty) = match value {
                LeafValue::Bound(b) => (cert_mr::bound_evidence(mode, b), b, false),
                LeafValue::Empty(slack) => (cert_mr::empty_evidence(mode, slack), slack, true),
            };
            if let Some(evidence) = evidence {
                return Ok(CertOutcome::Closed {
                    evidence,
                    rung,
                    value: v,
                    empty,
                });
            }
        }
        Ok(CertOutcome::Refuted)
    }
}

// ---------------------------------------------------------------------------
//  The wave driver
// ---------------------------------------------------------------------------

/// What one wave's search decided.
#[derive(Default)]
struct WaveStep {
    /// Domains for the certify queue.
    claims: Vec<Dom>,
    /// Domains to split, each with its validated candidates.
    splits: Vec<(Dom, Vec<u32>)>,
    /// MUTANT (test only): claims closed on their search bound.
    mutant_closes: Vec<(Dom, f64)>,
    /// A refuted domain with no unsplit neuron left.
    exhausted: bool,
    /// The search refused on the deadline.
    deadline: bool,
    /// The search refused (not the deadline) or panicked: why.
    refused: Option<String>,
    /// The branch scoring refused (not the deadline) or panicked: why (the
    /// wave's splits then take the lane's own first-unsplit candidates).
    branch_refused: Option<String>,
    /// Domains searched, bound evaluations, seconds.
    searched: usize,
    passes: u64,
    search_s: f64,
    /// Candidates the lane had to pick itself.
    fallback_splits: usize,
    /// The largest `tau` seen.
    tau_max: f64,
}

/// Running totals of one wave tree (the summary line).
#[derive(Default)]
struct WaveTally {
    waves: usize,
    searched: usize,
    passes: u64,
    claims: usize,
    closed: usize,
    empty: usize,
    refuted: usize,
    rung: [usize; 3],
    children: usize,
    fallback_splits: usize,
    search_refused: usize,
    /// The first search call that refused, as `<method>:<why>` (one token).
    first_refusal: Option<String>,
    search_s: f64,
    certify_s: f64,
    tau_max: f64,
    min_margin: f64,
    peak_pool: usize,
}

/// Static per-tree search settings.
struct SearchPlan<'p> {
    backend: &'p dyn SearchBackend,
    handle: &'p WaveHandle,
    universe: &'p Universe,
    cfg: WaveSearchCfg,
    /// `tau = tau_rel * S`.
    tau_rel: f64,
    deadline: Option<Instant>,
    hooks: WaveHooks,
    mode: crate::margin_row::rounding::RoundMode,
}

/// Search one wave (on the driver thread, while the certify pool runs the
/// previous wave's claims): claims, splits with candidates, refusals.
fn search_wave(plan: &SearchPlan<'_>, batch: Vec<Dom>, delta: f64, levels: usize) -> WaveStep {
    let t0 = Instant::now();
    let stride = plan.universe.len();
    let mut step = WaveStep::default();
    let (fresh, refuted): (Vec<Dom>, Vec<Dom>) = batch.into_iter().partition(|d| !d.refuted);
    let mut to_split: Vec<Dom> = refuted;
    if !fresh.is_empty() {
        let packed = Packed::of(&fresh, stride);
        let mut out = WaveOut::default();
        let res = catch_unwind(AssertUnwindSafe(|| {
            plan.backend.backward_batch(
                plan.handle,
                &packed.batch(plan.deadline),
                &plan.cfg,
                &mut out,
            )
        }));
        match res {
            Ok(Ok(stats)) => {
                let stats = stats.into_hint().into_ingest();
                step.passes = stats.passes;
                let view = out.hint().into_ingest();
                // Shape gate: `well_formed` only checks the buffers against
                // the shape the BACKEND prepared; the multipliers are sliced
                // below by THIS wave's literal offsets, so a backend that
                // prepared another literal count must be refused here, not
                // panic the lane outside the `catch_unwind` above.
                if view.well_formed()
                    && view.count() == fresh.len()
                    && view.stride() == stride
                    && view.beta().len() == packed.splits.len()
                {
                    step.searched = fresh.len();
                    for (k, mut d) in fresh.into_iter().enumerate() {
                        // INGEST: multipliers finite and >= 0 (the certified
                        // installers drop anything else again), slopes only
                        // where the hint is finite.
                        let r = packed.range(k);
                        d.beta = view.beta()[r]
                            .iter()
                            .map(|&b| if b.is_finite() && b > 0.0 { b } else { 0.0 })
                            .collect();
                        let hint = view.alpha(k);
                        if hint.iter().any(|a| a.is_finite()) {
                            let prev = d.alpha.clone();
                            let packed_alpha: Vec<u16> = hint
                                .iter()
                                .enumerate()
                                .map(|(p, &a)| {
                                    if a.is_finite() {
                                        pack_alpha(a)
                                    } else {
                                        prev.as_ref().map_or(ALPHA_UNSET, |q| q[p])
                                    }
                                })
                                .collect();
                            d.alpha = Some(Arc::from(packed_alpha));
                        }
                        let lb = f64::from(view.lb(k));
                        let s = f64::from(view.scale(k));
                        // A non-finite or negative S gives no tolerance at
                        // all: such a domain claims only above `delta`.
                        let tau = if s.is_finite() && s >= 0.0 {
                            plan.tau_rel * s
                        } else {
                            0.0
                        };
                        step.tau_max = step.tau_max.max(tau);
                        if lb.is_finite() && lb > -tau + delta {
                            if cfg!(test) && plan.hooks.claims_are_closes {
                                if cert_mr::bound_evidence(plan.mode, lb).is_some() {
                                    step.mutant_closes.push((d, lb));
                                    continue;
                                }
                            }
                            step.claims.push(d);
                        } else {
                            to_split.push(d);
                        }
                    }
                } else {
                    step.refused = Some("malformed wave output".into());
                    step.claims = fresh;
                }
            }
            Ok(Err(Refusal::Deadline)) => {
                step.deadline = true;
                return step;
            }
            Ok(Err(r)) => {
                // A refusal is never a verdict: certification decides every
                // domain of this wave.
                step.refused = Some(r.to_string());
                step.claims = fresh;
            }
            Err(_) => {
                step.refused = Some("backward_batch panicked".into());
                step.claims = fresh;
            }
        }
    }
    if to_split.is_empty() {
        step.search_s = t0.elapsed().as_secs_f64();
        return step;
    }
    // Branching: the backend's candidates, validated against each domain's
    // own literals; the lane's first unsplit neuron when none survives.
    let packed = Packed::of(&to_split, stride);
    let mut br = BranchOut::default();
    let k = levels.clamp(1, WAVE_MAX_LEVELS);
    let scored = catch_unwind(AssertUnwindSafe(|| {
        plan.backend
            .branch_scores(plan.handle, &packed.batch(plan.deadline), k, &mut br)
    }));
    let view = match scored {
        Ok(Ok(_)) => {
            let v = br.hint().into_ingest();
            let ok = v.well_formed() && v.count() == to_split.len() && v.k() == k;
            if !ok {
                // Never silently: a malformed scoring is a refusal of it (the
                // splits then take the lane's own candidates).
                step.branch_refused = Some("malformed branch output".into());
            }
            ok.then_some(v)
        }
        Ok(Err(Refusal::Deadline)) => {
            step.deadline = true;
            return step;
        }
        Ok(Err(r)) => {
            step.branch_refused = Some(r.to_string());
            None
        }
        Err(_) => {
            step.branch_refused = Some("branch_scores panicked".into());
            None
        }
    };
    for (i, d) in to_split.into_iter().enumerate() {
        let mut cands: Vec<u32> = Vec::with_capacity(k);
        if let Some(v) = view.as_ref() {
            for &p in v.pos(i) {
                if p == WAVE_NONE || p as usize >= stride {
                    continue;
                }
                if d.lits.iter().any(|l| l.pos == p) || cands.contains(&p) {
                    continue;
                }
                cands.push(p);
                if cands.len() == k {
                    break;
                }
            }
        }
        if cands.is_empty() {
            if let Some(p) = (0..stride as u32).find(|p| d.lits.iter().all(|l| l.pos != *p)) {
                cands.push(p);
                step.fallback_splits += 1;
            }
        }
        if cands.is_empty() {
            // Every universe neuron is pinned: the region is linear up to the
            // frozen boxes. A fresh domain is certified; a refuted one cannot
            // be refined further.
            if d.refuted {
                step.exhausted = true;
            } else {
                step.claims.push(d);
            }
            continue;
        }
        step.splits.push((d, cands));
    }
    step.search_s = t0.elapsed().as_secs_f64();
    step
}

/// `why` as one receipt token: every character outside `[A-Za-z0-9._:/-]`
/// (a space, an `=`, a bracket, a comma) becomes `_`, and at most
/// [`RECEIPT_TOKEN_MAX`] characters survive, so a `key=value` reader of the
/// `[bbe] phase=wave` line never mis-splits a refusal reason.
fn one_token(why: &str) -> String {
    why.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | ':' | '/' | '-') {
                c
            } else {
                '_'
            }
        })
        .take(RECEIPT_TOKEN_MAX)
        .collect()
}

/// Longest refusal reason a receipt token carries.
const RECEIPT_TOKEN_MAX: usize = 160;

/// Bytes the wave engine may hold: [`WAVE_MAX_BYTES`], clamped to half of the
/// observable process headroom.
fn admitted_bytes() -> u64 {
    match crate::network::crown_memory::process_memory_headroom_bytes() {
        Some(headroom) => WAVE_MAX_BYTES.min(headroom / 2),
        None => WAVE_MAX_BYTES,
    }
}

impl MarginRowBab<'_> {
    /// The armed wave loop's refusal path: print the observable and run the
    /// legacy tree unchanged. A refusal is never a verdict.
    fn wave_fallback(
        &mut self,
        root_bound: f64,
        re: &RootEval,
        backend: &str,
        why: &str,
        device_refused: &str,
    ) -> Result<MarginRowOutcome> {
        let mode = self.cfg.wave.mode.token();
        // One token, like `device_refused=` / `device_switch=`: a refusal
        // reason carries spaces, `=` and brackets ("unsupported: upload_wave",
        // "memory: need N B > cap M B"), which a `key=value` reader of this
        // line would mis-split.
        let why = one_token(why);
        let device_refused = one_token(device_refused);
        note_wave_summary(format!(
            "[bbe] phase=wave armed=1 backend={backend} refused={why} fallback=legacy \
             mode={mode} device_refused={device_refused} wave-beta-bab/v1"
        ));
        if self.cfg.frontier > 1 {
            self.tree_loop_parallel(root_bound, re)
        } else {
            self.tree_loop(root_bound, re)
        }
    }

    /// Record one certified leaf in its class's cover ledger.
    ///
    /// B2.3 (proof bundle v2): AFTER the ledger counted the leaf, and only
    /// while the frame's hint ledger captures (the Proof profile), the leaf's
    /// hint — its literals in split order, the β it was certified under, its
    /// close and the ONE class its row discharges — goes into the FRAME's
    /// hint ledger, which streams it with the frame after the token exists.
    /// A leaf whose hint cannot be built marks its class `hint_lost`, so the
    /// class keeps its claim in the stream. Write-only either way.
    #[allow(clippy::too_many_arguments)]
    fn wave_leaf(
        &mut self,
        ledgers: &mut [CoverLedger],
        universe: &Universe,
        classes: &[usize],
        hint_lost: &mut [bool],
        dom: &Dom,
        evidence: LeafEvidence,
    ) {
        if cover_fault_drop_next_leaf() {
            return;
        }
        let (trunk, heads, _, _) = universe.split_lists(&dom.lits, &dom.beta);
        let path = cert_mr::leaf_path(&trunk, &heads, &self.cfg.initial_heads, None);
        let info = match &evidence {
            LeafEvidence::Bound(c) => Some((LeafClose::Bound, c.value())),
            LeafEvidence::Empty(c) => Some((LeafClose::Empty, c.value())),
            _ => None,
        };
        let row = dom.row as usize;
        ledgers[row].leaf(path, Some(evidence));
        if !self.ledger.hints.capturing() {
            return;
        }
        let splits: Option<Vec<FrameSplit>> = dom
            .lits
            .iter()
            .map(|lit| {
                let site = universe.sites.get(lit.pos as usize)?;
                Some(match *site {
                    Site::Trunk { li, pos } => FrameSplit::Trunk {
                        layer: u32::try_from(li).ok()?,
                        pos: u32::try_from(pos).ok()?,
                        positive: lit.active,
                    },
                    Site::Head { index } => FrameSplit::Head {
                        index: u32::try_from(index).ok()?,
                        positive: lit.active,
                    },
                })
            })
            .collect();
        match (splits, info, classes.get(row)) {
            (Some(splits), Some((close, margin)), Some(&class)) => {
                let beta: Vec<f64> = dom.beta.iter().map(|&b| f64::from(b)).collect();
                let hint = LeafHint {
                    splits,
                    beta,
                    close,
                    dis: vec![class],
                    margin,
                    lift: false,
                    empty_split: None,
                };
                // The slopes the certified pass ran under: the domain's
                // searched slope where it set one, else the root's.
                match dom.alpha_f64() {
                    Some(a) => {
                        let table = universe.leaf_alpha_table(&a);
                        self.ledger.hints.push_leaf_with_alpha(hint, table);
                    }
                    None => self.ledger.hints.push_leaf(hint),
                }
            }
            _ => {
                if let Some(lost) = hint_lost.get_mut(row) {
                    *lost = true;
                }
            }
        }
    }

    /// #wave-beta-bab: the wave tree on the outer frame (see the module docs).
    pub(super) fn tree_loop_wave(
        &mut self,
        root_bound: f64,
        re: &RootEval,
    ) -> Result<MarginRowOutcome> {
        let t0 = Instant::now();
        let policy = self.cfg.wave.clamped();
        let hooks = WaveHooks::capture();
        // The mode's search as ONE backend: `wave` is the CPU search;
        // `wave_cuda` is the device with the CPU search behind it
        // (`FallbackBackend`: the upload fall-through and the post-fault
        // switch live there, not here), or the CPU search alone when no device
        // backend is registered (`device_refused=no_backend`).
        let kinds = policy.mode.backends();
        let wants_device = kinds.contains(&WaveBackendKind::Cuda);
        let device = if wants_device {
            wave_backend(WaveBackendKind::Cuda)
        } else {
            None
        };
        let cpu = if kinds.contains(&WaveBackendKind::CpuF32) {
            wave_backend(WaveBackendKind::CpuF32)
        } else {
            None
        };
        let no_device = wants_device && device.is_none();
        let composite: Option<Arc<FallbackBackend>> =
            device.map(|d| Arc::new(FallbackBackend::new(d, cpu.clone())));
        let device_note = if no_device { "no_backend" } else { "none" };
        let backend: Arc<dyn SearchBackend> = match (&composite, cpu) {
            (Some(c), _) => Arc::clone(c) as Arc<dyn SearchBackend>,
            (None, Some(c)) => c,
            (None, None) => {
                return self.wave_fallback(root_bound, re, "none", "no_backend", device_note)
            }
        };
        let name = backend.name();
        let root = self.eng.root;
        let net = self.eng.net;
        let (ly0, uy0) = (re.pack.ly0.clone(), re.pack.uy0.clone());
        let universe = match Universe::build(net, root, &ly0, &uy0) {
            Ok(u) => u,
            Err(why) => return self.wave_fallback(root_bound, re, name, &why, device_note),
        };
        let classes = self.stats.tree_classes.clone();
        let problem = match universe.problem(net, root, &ly0, &uy0, self.mb.t, &classes) {
            Ok(p) => p,
            Err(why) => return self.wave_fallback(root_bound, re, name, &why, device_note),
        };
        let admitted = admitted_bytes();
        let lease = DeviceLease {
            bytes: admitted / 2,
        };
        // One upload; a refusal is never a verdict. On `wave_cuda` the
        // composite has already fallen through to the CPU search when the
        // device refused, so a refusal here means every backend refused: the
        // legacy tree runs, and the receipt names the last backend asked and
        // the device's own refusal.
        let handle = match catch_unwind(AssertUnwindSafe(|| backend.upload_wave(&problem, &lease)))
        {
            Ok(Ok(h)) if h.stride() == universe.len() => h,
            other => {
                let why = match other {
                    Ok(Ok(_)) => "stride_mismatch".to_string(),
                    Ok(Err(r)) => r.to_string(),
                    Err(_) => "upload_panicked".to_string(),
                };
                let (asked, device_refused) = match &composite {
                    Some(c) => {
                        let (primary, secondary) = c.names();
                        (
                            secondary.unwrap_or(primary),
                            c.primary_upload_refusal()
                                .unwrap_or_else(|| "none".to_string()),
                        )
                    }
                    None => (name, device_note.to_string()),
                };
                return self.wave_fallback(root_bound, re, asked, &why, &device_refused);
            }
        };
        drop(problem);
        // Which backend(s) searched this tree: the composite's route (a
        // single backend's is its own name).
        let route_of =
            |h: &WaveHandle| FallbackBackend::route(h).unwrap_or_else(|| WaveRoute::single(name));
        let device_refused = if no_device {
            device_note.to_string()
        } else {
            route_of(&handle)
                .primary_refused
                .map_or_else(|| "none".to_string(), |why| one_token(&why))
        };
        // The lane wave in which the composite switched (receipt only).
        let mut switched_at: Option<usize> = None;
        let available = rayon::current_num_threads().max(1);
        let cert_threads = if policy.cert_threads > 0 {
            policy.cert_threads
        } else {
            (available / 2).max(1)
        };
        let cert_pool = match rayon::ThreadPoolBuilder::new()
            .num_threads(cert_threads)
            .thread_name(|i| format!("ny-wave-cert-{i}"))
            .build()
        {
            Ok(p) => p,
            Err(e) => {
                return self.wave_fallback(
                    root_bound,
                    re,
                    name,
                    &format!("pool: {e}"),
                    &device_refused,
                )
            }
        };
        let mbs: Vec<MarginBatch> = (0..classes.len())
            .map(|r| self.mb.subset(&[r]))
            .collect::<Result<_>>()?;
        eprintln!(
            "[margin-row-root] tree_loop entry root_bound={root_bound:.6} \
             (this lane's own bound; NOT the multi-objective census)"
        );
        let stride = universe.len();
        let search_cfg = WaveSearchCfg {
            iters: WAVE_BETA_ITERS,
            lr_alpha: WAVE_LR_ALPHA,
            lr_beta: WAVE_LR_BETA,
            stop_above: crate::margin_row::beta::CLOSE_EPS,
            threads: available.saturating_sub(cert_threads).max(1),
        };
        let tau_rel = 4.0 * gamma_f32(2 * net.ops.len() + 64);
        let search_deadline = self.cfg.deadline;
        let pack_cache = Mutex::new(Lru {
            cap: self.cfg.lru_cap.max(1),
            entries: Vec::new(),
        });
        let ctx = CertCtx {
            universe: &universe,
            root_pack: &re.pack,
            mbs: &mbs,
            cache: &pack_cache,
            hooks,
        };
        let pool_cap = admitted / 2;
        // #router-r6 `rule.trial_width`: the wave width's AIMD machine (`None`
        // unless the acting optimizer armed the site — then every wave below
        // uses exactly `policy.batch`, today's width). It moves only inside
        // the knob's declared grid span and the reader clamp; a wave is the
        // unit, its boundary (after the merge) the safe point; the metric is
        // the tree's CERTIFIED closes per second; lease use is the host pool
        // against its own static budget.
        let mut width_trial =
            crate::trial_sites::declared_grid_span(wave_width_lever()).and_then(|(lo, hi)| {
                crate::trial_sites::width_trial(
                    crate::trial_sites::global(),
                    ny_core::trial_authority::TrialSite::WaveWidth,
                    policy.batch as u64,
                    lo.max(1),
                    hi.min(WAVE_BATCH_MAX as u64),
                )
            });
        let mut pool: Vec<Dom> = (0..classes.len() as u32).rev().map(Dom::root).collect();
        let mut pool_bytes: u64 = pool.iter().map(|d| d.bytes(stride)).sum();
        let mut pending: Vec<Dom> = Vec::new();
        let mut ledgers: Vec<CoverLedger> =
            (0..classes.len()).map(|_| CoverLedger::new()).collect();
        // B2.3: per class row, a leaf whose proof hint could not be built.
        let mut hint_lost: Vec<bool> = vec![false; classes.len()];
        let mut tally = WaveTally {
            min_margin: f64::INFINITY,
            ..WaveTally::default()
        };
        let mut delta = 0.0f64;
        let mut stop = String::from("queue_empty");
        // A certification error (or a lost / panicked certify job) fails the
        // frame closed AFTER the receipt is printed, so an armed run that
        // fails closed is distinguishable from one that never ran.
        let mut failure: Option<NyError> = None;
        self.stats.domains_created = classes.len();
        loop {
            if pool.is_empty() && pending.is_empty() {
                break;
            }
            if self.deadline_hit() {
                stop = "wallclock".into();
                break;
            }
            if self.stats.expansions >= self.cfg.max_expansions {
                stop = "max_expansions".into();
                break;
            }
            if pool_bytes > pool_cap.saturating_mul(2) {
                // Never evict (completeness): stop open instead.
                stop = "wave_memory".into();
                break;
            }
            // #router-r6: the live width (the trial's, else today's).
            let live_batch = crate::trial_sites::width_value(width_trial.as_ref(), policy.batch);
            // At the pool cap, deepen instead of widening: half-width waves,
            // one level per split.
            let at_cap = pool_bytes > pool_cap;
            let width = if at_cap {
                (live_batch / 2).max(1)
            } else {
                live_batch
            };
            let take = width.min(pool.len());
            let batch: Vec<Dom> = pool.split_off(pool.len() - take);
            for d in &batch {
                pool_bytes = pool_bytes.saturating_sub(d.bytes(stride));
            }
            let thin = pool.len() + pending.len() + take < live_batch;
            // Multi-level splits while the pool is thinner than one wave:
            // `floor(log2(batch / take))` levels, at most WAVE_MAX_LEVELS.
            let levels_wanted = if thin && !at_cap && take > 0 {
                let want = (live_batch / take).max(1);
                (usize::BITS - want.leading_zeros()).saturating_sub(1) as usize
            } else {
                1
            };
            let levels = levels_wanted.clamp(1, WAVE_MAX_LEVELS);
            // The streaming certify queue: the PREVIOUS wave's claims certify
            // on the private pool while THIS wave searches on this thread.
            // Only shared references cross into the spawned job; its results
            // come back through `cert_slot` and are merged in claim order.
            let this: &Self = &*self;
            let cert_slot: Mutex<Option<(Vec<Result<CertOutcome>>, f64)>> = Mutex::new(None);
            let (pend_ref, ctx_ref, slot_ref) = (&pending, &ctx, &cert_slot);
            let step = cert_pool.in_place_scope(|s| {
                if !pend_ref.is_empty() {
                    s.spawn(move |_| {
                        let tc = Instant::now();
                        let certified = catch_unwind(AssertUnwindSafe(|| {
                            pend_ref
                                .par_iter()
                                .map(|d| this.wave_certify(ctx_ref, d))
                                .collect::<Vec<Result<CertOutcome>>>()
                        }))
                        .unwrap_or_else(|_| {
                            vec![Err(NyError::NumericalInstability(
                                "margin_row wave: the certify queue panicked".into(),
                            ))]
                        });
                        if let Ok(mut slot) = slot_ref.lock() {
                            *slot = Some((certified, tc.elapsed().as_secs_f64()));
                        }
                    });
                }
                let plan = SearchPlan {
                    backend: &*backend,
                    handle: &handle,
                    universe: &universe,
                    cfg: search_cfg,
                    tau_rel,
                    deadline: search_deadline,
                    hooks,
                    mode: root.mode,
                };
                search_wave(&plan, batch, delta, levels)
            });
            let (certs, cert_secs) = cert_slot
                .into_inner()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .unwrap_or_default();
            if certs.len() != pending.len() {
                // A claim without its certification must never vanish from
                // the tree: fail closed (a panic, or a lost result).
                failure = Some(
                    certs
                        .into_iter()
                        .find_map(std::result::Result::err)
                        .unwrap_or_else(|| {
                            NyError::NumericalInstability(
                                "margin_row wave: the certify queue lost a claim".into(),
                            )
                        }),
                );
                stop = "fail_closed".into();
                break;
            }
            tally.waves += 1;
            tally.certify_s += cert_secs;
            tally.search_s += step.search_s;
            tally.searched += step.searched;
            tally.passes += step.passes;
            tally.fallback_splits += step.fallback_splits;
            tally.tau_max = tally.tau_max.max(step.tau_max);
            if step.refused.is_some() || step.branch_refused.is_some() {
                tally.search_refused += 1;
            }
            // Never swallowed: the first refusal of either search call is on
            // the receipt, by method.
            if tally.first_refusal.is_none() {
                tally.first_refusal = step
                    .refused
                    .as_deref()
                    .map(|why| format!("backward_batch:{}", one_token(why)))
                    .or_else(|| {
                        step.branch_refused
                            .as_deref()
                            .map(|why| format!("branch_scores:{}", one_token(why)))
                    });
            }
            // The composite switches inside the search call; the lane only
            // notes WHICH wave it happened in (receipt only).
            if switched_at.is_none() && composite.is_some() && route_of(&handle).switch.is_some() {
                switched_at = Some(tally.waves);
            }
            // Merge the previous wave's certifications, in claim order.
            let mut wave_closed = 0usize;
            let mut wave_refuted = 0usize;
            let mut cert_deadline = false;
            for (dom, res) in std::mem::take(&mut pending).into_iter().zip(certs) {
                // `Err` is never a close: the whole frame fails closed.
                let res = match res {
                    Ok(res) => res,
                    Err(e) => {
                        failure = Some(e);
                        break;
                    }
                };
                match res {
                    CertOutcome::Closed {
                        evidence,
                        rung,
                        value,
                        empty,
                    } => {
                        self.wave_leaf(
                            &mut ledgers,
                            &universe,
                            &classes,
                            &mut hint_lost,
                            &dom,
                            evidence,
                        );
                        #[cfg(test)]
                        record_leaf(&universe, &classes, &dom, value, empty, rung.token());
                        self.stats.closed += 1;
                        tally.closed += 1;
                        wave_closed += 1;
                        tally.rung[rung as usize] += 1;
                        if empty {
                            tally.empty += 1;
                        } else {
                            tally.min_margin = tally.min_margin.min(value);
                        }
                    }
                    CertOutcome::Refuted => {
                        tally.refuted += 1;
                        wave_refuted += 1;
                        let mut dom = dom;
                        dom.refuted = true;
                        pool_bytes = pool_bytes.saturating_add(dom.bytes(stride));
                        pool.push(dom);
                    }
                    CertOutcome::Deadline => cert_deadline = true,
                }
            }
            if failure.is_some() {
                stop = "fail_closed".into();
                break;
            }
            // MUTANT (test only): claims closed on their search bound.
            for (dom, lb) in step.mutant_closes {
                if let Some(evidence) = cert_mr::bound_evidence(root.mode, lb) {
                    self.wave_leaf(
                        &mut ledgers,
                        &universe,
                        &classes,
                        &mut hint_lost,
                        &dom,
                        evidence,
                    );
                    #[cfg(test)]
                    record_leaf(&universe, &classes, &dom, lb, false, "mutant");
                    self.stats.closed += 1;
                    tally.closed += 1;
                }
            }
            // The claim offset follows the certify queue's refutation rate.
            let judged = wave_closed + wave_refuted;
            if judged >= 8 {
                let rate = wave_refuted as f64 / judged as f64;
                if rate > WAVE_REFUTE_HIGH {
                    delta = (2.0 * delta).max(tally.tau_max.max(f64::MIN_POSITIVE));
                } else if rate < WAVE_REFUTE_LOW {
                    delta *= 0.5;
                    if delta < 0.25 * tally.tau_max {
                        delta = 0.0;
                    }
                }
            }
            tally.claims += step.claims.len();
            pending = step.claims;
            for (dom, cands) in step.splits {
                let kids = dom.children(&cands);
                self.stats.expansions += 1;
                self.stats.domains_created += kids.len();
                tally.children += kids.len();
                self.stats.max_depth = self.stats.max_depth.max(dom.lits.len() + cands.len());
                for kid in kids {
                    pool_bytes = pool_bytes.saturating_add(kid.bytes(stride));
                    pool.push(kid);
                }
            }
            tally.peak_pool = tally.peak_pool.max(pool.len() + pending.len());
            // #router-r1: write-only per-wave progress tap (FrontierKary:
            // expansions, creations, closes, open pool). Nothing reads it back;
            // one relaxed atomic load while the router is off.
            if ny_core::router::bus::enabled() {
                ny_core::router::bus::emit(ny_core::router::bus::ProgressSample::frontier(
                    ny_core::router::decl::ids::WAVE_BETA_BAB,
                    self.stats.expansions as u64,
                    self.stats.domains_created as u64,
                    self.stats.closed as u64,
                    (pool.len() + pending.len()) as u64,
                ));
            }
            // #router-r6 `rule.trial_width`: the wave boundary is this lane's
            // safe point; the machine sees the tree's certified closes and
            // decides the NEXT wave's width (a no-op when unarmed).
            if width_trial.is_some() {
                let (t_s, remaining_s) = crate::trial_sites::lane_clock(t0, self.cfg.deadline);
                crate::trial_sites::observe_width(
                    crate::trial_sites::global(),
                    &mut width_trial,
                    ny_core::router::trial::WidthObs {
                        units: tally.waves as u64,
                        closes: self.stats.closed as u64,
                        t_s,
                        remaining_s,
                        lease_use: (pool_cap > 0).then(|| pool_bytes as f64 / pool_cap as f64),
                    },
                );
            }
            if cert_deadline || step.deadline {
                stop = "wallclock_intra_expansion".into();
                break;
            }
            if step.exhausted {
                stop = "no_candidates".into();
                break;
            }
        }
        self.stats.stop.clone_from(&stop);
        self.stats.elapsed_secs = t0.elapsed().as_secs_f64();
        // The EFFECTIVE backend(s), from the composite's route: a tree the
        // device searched alone reads `backend=cuda`; a switched one
        // `backend=cuda->cpu_f32` (or `cuda->none` when the CPU search could
        // not take it over) with the wave and the device's reason.
        let route = route_of(&handle);
        let backends = route.backends();
        let (device_switch, switched_at_wave, switch_reason) = match &route.switch {
            None => ("none".to_string(), "none".to_string(), "none".to_string()),
            Some(sw) => {
                let why = one_token(&format!("{}:{}", sw.method, sw.reason));
                let at = switched_at.map_or_else(|| "unknown".to_string(), |w| w.to_string());
                let outcome = match &sw.to {
                    Ok(to) => format!("switched_to_{to}"),
                    Err(e) => format!("cpu_refused_{}", one_token(e)),
                };
                (format!("wave{at}:{why}:{outcome}"), at, why)
            }
        };
        let first_refusal = tally.first_refusal.as_deref().unwrap_or("none");
        note_wave_summary(format!(
            "[bbe] phase=wave armed=1 backend={backends} B={} cert_threads={cert_threads} \
             universe={stride} classes={} lease_mb={} waves={} searched={} passes={} claims={} \
             closed={} empty={} refuted={} rung_frozen={} rung_clip={} rung_ypack={} \
             splits={} children={} fallback_splits={} search_refused={} tau={:.3e} delta={:.3e} \
             peak_pool={} search_s={:.3} certify_s={:.3} min_margin={:.3e} stop={stop} \
             mode={} device_refused={device_refused} device_switch={device_switch} \
             switched_at_wave={switched_at_wave} switch_reason={switch_reason} \
             first_refusal={first_refusal} wave-beta-bab/v1",
            policy.batch,
            classes.len(),
            lease.bytes >> 20,
            tally.waves,
            tally.searched,
            tally.passes,
            tally.claims,
            tally.closed,
            tally.empty,
            tally.refuted,
            tally.rung[Rung::Frozen as usize],
            tally.rung[Rung::Clip as usize],
            tally.rung[Rung::YPack as usize],
            self.stats.expansions,
            tally.children,
            tally.fallback_splits,
            tally.search_refused,
            tally.tau_max,
            delta,
            tally.peak_pool,
            tally.search_s,
            tally.certify_s,
            tally.min_margin,
            policy.mode.token(),
        ));
        if let Some(e) = failure {
            return Err(e);
        }
        if stop != "queue_empty" {
            return Ok(MarginRowOutcome::Unknown {
                reason: stop,
                stats: Some(self.stats.clone()),
            });
        }
        // #publish-deadline, as the legacy loops.
        if self
            .cfg
            .effective_publish_deadline()
            .is_some_and(|dl| Instant::now() > dl)
        {
            self.stats.stop = "wallclock_before_verdict".into();
            return Ok(MarginRowOutcome::Unknown {
                reason: "wallclock_before_verdict".into(),
                stats: Some(self.stats.clone()),
            });
        }
        // U5 per class: the path-based cover gate on every class's leaves.
        let mut covers: Vec<ClassCover> = Vec::with_capacity(classes.len());
        for (r, ledger) in ledgers.iter().enumerate() {
            match ledger.prove(&[classes[r]]) {
                Ok(cover) => covers.push(ClassCover::Tree(cover)),
                Err(refusal) => {
                    eprintln!(
                        "[cover-gate] REFUSED scope=wave_class class={} leaves={} reason={refusal}",
                        classes[r],
                        ledger.recorded()
                    );
                    self.stats.ledger_ok = Some(false);
                    self.stats.stop = "unsat_refused".into();
                    return Ok(MarginRowOutcome::Unknown {
                        reason: format!("cover gate refused: {refusal}"),
                        stats: Some(self.stats.clone()),
                    });
                }
            }
        }
        // B2.3: the classes whose every leaf streamed with the frame (their
        // tree covers then claim nothing more in the bundle). Hint data only.
        if self.ledger.hints.capturing() {
            self.ledger.streamed_tree_classes = classes
                .iter()
                .zip(&hint_lost)
                .filter(|(_, lost)| !**lost)
                .map(|(c, _)| *c)
                .collect();
        }
        // U6-U7: the frame's own authorization over the pre-tree covers and
        // every class tree's cover.
        self.ledger.pre.extend(covers);
        match authorize_frame(
            &mut self.ledger,
            &self.stats.tree_classes,
            false,
            self.eng.net,
            self.eng.root,
            self.mb.t,
            re,
            &self.cfg,
        ) {
            Ok(token) => {
                self.stats.ledger_ok = Some(true);
                Ok(MarginRowOutcome::Unsat(self.stats.clone(), token))
            }
            Err(reason) => {
                self.stats.ledger_ok = Some(false);
                self.stats.stop = "unsat_refused".into();
                Ok(MarginRowOutcome::Unknown {
                    reason,
                    stats: Some(self.stats.clone()),
                })
            }
        }
    }
}

#[cfg(test)]
fn record_leaf(
    universe: &Universe,
    classes: &[usize],
    dom: &Dom,
    value: f64,
    empty: bool,
    rung: &'static str,
) {
    let (trunk, heads, _, _) = universe.split_lists(&dom.lits, &dom.beta);
    TEST_LEAVES.with(|c| {
        c.borrow_mut().push(RecordedLeaf {
            class: classes[dom.row as usize],
            trunk,
            heads,
            value,
            empty,
            rung,
        });
    });
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn wave_tree_mode_tokens_and_policy_clamps() {
        for (m, t) in [
            (TreeMode::Legacy, "legacy"),
            (TreeMode::Wave, "wave"),
            (TreeMode::WaveCuda, "wave_cuda"),
        ] {
            assert_eq!(m.token(), t);
            assert_eq!(TreeMode::from_token(t), Some(m));
        }
        assert_eq!(TreeMode::TOKENS, ["legacy", "wave", "wave_cuda"]);
        assert_eq!((TreeMode::Legacy.backends()).len(), 0);
        assert_eq!(TreeMode::Wave.backends(), [WaveBackendKind::CpuF32]);
        assert_eq!(
            TreeMode::WaveCuda.backends(),
            [WaveBackendKind::Cuda, WaveBackendKind::CpuF32],
            "wave_cuda tries the device first and falls through to the CPU search"
        );
        for bogus in [
            "Wave",
            "WAVE",
            " wave",
            "1",
            "true",
            "",
            "cuda",
            "wave-cuda",
            "WAVE_CUDA",
        ] {
            assert_eq!(TreeMode::from_token(bogus), None);
        }
        assert_eq!(TreeMode::default(), TreeMode::Legacy);
        let d = WavePolicy::default();
        assert!(!d.armed());
        assert_eq!(d.batch, WAVE_BATCH_DEFAULT);
        let c = WavePolicy {
            mode: TreeMode::Wave,
            batch: 0,
            cert_threads: usize::MAX,
        }
        .clamped();
        assert!(c.armed());
        assert!(WavePolicy {
            mode: TreeMode::WaveCuda,
            ..d
        }
        .armed());
        assert_eq!((c.batch, c.cert_threads), (1, WAVE_CERT_THREADS_MAX));
        assert_eq!(
            WavePolicy {
                batch: usize::MAX,
                ..d
            }
            .clamped()
            .batch,
            WAVE_BATCH_MAX
        );
    }

    #[test]
    fn wave_alpha_storage_round_trips_and_keeps_unset() {
        for a in [0.0f32, 1.0, 0.5, 0.123_456, 1e-7] {
            let q = pack_alpha(a);
            assert_ne!(q, ALPHA_UNSET);
            assert!((unpack_alpha(q) - a).abs() <= 1.0 / 65534.0);
        }
        assert_eq!(pack_alpha(7.0), pack_alpha(1.0), "clamped into [0, 1]");
        assert_eq!(pack_alpha(-3.0), 0);
        assert_eq!(pack_alpha(f32::NAN), ALPHA_UNSET);
        assert!(unpack_alpha(ALPHA_UNSET).is_nan());
        assert_eq!(unpack_alpha(pack_alpha(1.0)), 1.0);
    }

    #[test]
    fn wave_children_are_every_side_combination_of_distinct_sites() {
        let parent = Dom {
            row: 2,
            lits: vec![WaveSplit {
                pos: 9,
                active: true,
            }],
            beta: vec![0.5],
            alpha: Some(Arc::from(vec![1u16, 2, 3])),
            refuted: true,
        };
        let kids = parent.children(&[4, 7]);
        assert_eq!(kids.len(), 4);
        let mut seen: Vec<(bool, bool)> = kids
            .iter()
            .map(|k| {
                assert_eq!(k.row, 2);
                assert!(!k.refuted, "a child is a fresh domain");
                assert_eq!(k.lits[0], parent.lits[0], "the parent's path is a prefix");
                assert_eq!(k.beta, vec![0.5, 0.0, 0.0], "fresh splits start at beta 0");
                assert!(Arc::ptr_eq(
                    k.alpha.as_ref().expect("shared"),
                    parent.alpha.as_ref().expect("shared")
                ));
                assert_eq!((k.lits[1].pos, k.lits[2].pos), (4, 7));
                (k.lits[1].active, k.lits[2].active)
            })
            .collect();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), 4, "every side combination exactly once");
        assert_eq!(parent.children(&[]).len(), 1);
    }

    #[test]
    fn wave_packed_batch_matches_the_domains() {
        let a: Arc<[u16]> = Arc::from(vec![0u16, ALPHA_UNSET]);
        let doms = vec![
            Dom::root(0),
            Dom {
                row: 1,
                lits: vec![WaveSplit {
                    pos: 1,
                    active: false,
                }],
                beta: vec![0.25],
                alpha: Some(a),
                refuted: false,
            },
        ];
        let p = Packed::of(&doms, 2);
        let b = p.batch(None);
        assert!(b.validate(2, 2).is_ok());
        assert_eq!(b.offsets, &[0, 0, 1][..]);
        assert_eq!(b.alpha.len(), 4);
        assert!(b.alpha[0].is_nan() && b.alpha[1].is_nan(), "no hint: NaN");
        assert_eq!(b.alpha[2], 0.0);
        assert!(b.alpha[3].is_nan(), "unset slot: NaN");
        assert_eq!(p.range(1), 0..1);
        // No domain carries slopes: the batch passes none.
        assert_eq!((Packed::of(&doms[..1], 2).alpha).len(), 0);
    }

    #[test]
    fn wave_gamma_is_the_binary32_constant() {
        assert_eq!(gamma_f32(0), 0.0);
        let g = gamma_f32(100);
        assert!(g > 100.0 * 5.96e-8 && g < 100.0 * 5.97e-8);
        assert!(gamma_f32(usize::MAX).is_infinite());
    }

    /// #router-r6 `rule.trial_width` at the wave site: the trial's range is
    /// the knob's DECLARED grid span (`[64, 1024]`), inside the reader clamp,
    /// and around the declaration default it is exactly that span; an
    /// unarmed process builds no machine, so the wave's width is
    /// `policy.batch`, today's.
    #[test]
    fn router_r6_wave_width_trial_stays_inside_the_declared_span() {
        use ny_core::router::trial::WidthRange;
        let (lo, hi) =
            crate::trial_sites::declared_grid_span(wave_width_lever()).expect("a grid axis");
        assert_eq!((lo, hi), (64, 1024), "the declared grid span");
        assert!(hi <= WAVE_BATCH_MAX as u64 && lo >= 1);
        assert_eq!(
            WidthRange::around(WAVE_BATCH_DEFAULT as u64, 4, lo, hi),
            Some(WidthRange { min: 64, max: 1024 })
        );
        assert_eq!(
            WidthRange::around(4096, 4, lo, hi),
            None,
            "a width outside the span is never trialled"
        );
        assert!(
            crate::trial_sites::width_trial(
                crate::trial_sites::global(),
                ny_core::trial_authority::TrialSite::WaveWidth,
                WAVE_BATCH_DEFAULT as u64,
                lo,
                hi,
            )
            .is_none(),
            "unarmed: no machine"
        );
        assert_eq!(
            crate::trial_sites::width_value(None, WAVE_BATCH_DEFAULT),
            WAVE_BATCH_DEFAULT
        );
    }
}
