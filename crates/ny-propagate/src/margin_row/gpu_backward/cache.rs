// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! B1 — the plan/weight RESIDENCY cache of the certified GPU backward
//! (perf session, impl-v7; design `#B1` in the GPU throughput design).
//!
//! # What is cached, and why it is sound to cache it
//!
//! Everything in a [`PlanStatic`] is a pure function of the ENGINE'S
//! NETWORK PARAMETERS and the ROOT box — never of the per-pass domain
//! gates, the seed, or the direction:
//!
//! * the plan skeleton ([`gpu_seam::PlanExt`]): layer shapes, fold-order
//!   `relus`/`conv_ops` provenance lists (dom changes only the gate VALUES
//!   inside the plan's `Activation` payloads, which this lane deliberately
//!   never reads — the ds gate quads are rebuilt per pass from `eng`+`dom`);
//! * the ds-pair weight/bias byte planes ([`transaction::ds_pair_plane`] /
//!   [`transaction::ds_scale_plane`] over the engine's f64 parameters) and
//!   their charges (`w_excess`, masses, `rel` weight balls, the split-floor
//!   operator norms `a_eff`) — all computed ONCE by the same helper
//!   functions the per-pass path used, so a cache hit reproduces the
//!   uncached charges BIT-FOR-BIT (pinned in tests);
//! * fix A3: the realization probe's pass-invariant midpoint forward —
//!   `x = root.mid` never changes per pass, so `(y, pre)` from ONE exact
//!   f64 forward serve every pass ([`ProbeStatic`],
//!   [`probe_guard_cached`]).
//!
//! What is NOT cached (per-pass dynamic part): gate ds quads (dom-
//! dependent), the seed plane, zeroed intermediates (pooled device-side by
//! the executor's B1 arena pool), the seed penalty, and the per-dom GATE
//! ADMISSION — a cache hit re-runs [`gpu_seam::outward_gate_f32`] on the
//! dom-selected gates ([`check_gate_admission`]) so the hit path refuses
//! exactly where a fresh `build_plan_ext` would (fractional-alpha and
//! non-f32-exact-slope doms keep refusing; semantics-neutral by
//! construction).
//!
//! # Ownership / staleness (the structural argument)
//!
//! The design owner for a `PlanStatic` is the caller that owns the
//! `RootGates`/engine it was built from (post-B4 wiring: an
//! `Arc<PlanStatic>` carried by the BaB struct through `LaneCtx` — SEAM
//! STUB, see `mod.rs::run_pass_armed`). This session wires only the M2
//! SHADOW path to a cache (`mod.rs::shadow_plan_cache`), which is
//! discard-only by construction: a stale hit there can at worst produce a
//! false SHADOW TRIP (fail-closed — the channel latches dead), never a
//! wrong bound. The (M3-gated) verdict path passes NO cache and rebuilds
//! per pass, bit-for-bit today's behavior. Every probe re-validates a
//! fail-closed identity fingerprint ([`PlanCache::probe`]): engine/root
//! pointer identity, shape census, and bit-compared parameter sentinels —
//! a mismatch is a MISS (rebuild), never an error.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use ndarray::Array2;
use ny_core::{LaneResidentId, MarginRowLaneGpu};

use super::super::engine::{BackwardEngine, DomainGates, LaneDir, PassOut, Seed};
use super::super::gpu_seam::{self, PlanExt};
use super::super::net::TwinOp;
use super::transaction::{
    self, conv_ci_mass_max, ds_bias_abs, ds_pair_plane, ds_scale_plane, ds_weight_rel, f32_up,
    head_col_mass_max, map_seam_refusal, norm_inflate,
};
use super::Refusal;

/// One pass-invariant byte plane (ds-pair weights / expanded bias scales):
/// the bytes always exist host-side (the fallback and the no-GPU test
/// path); `resident` is the device-resident handle when the executor
/// accepted the one-time upload (B1 residency — the per-pass upload
/// disappears), else `None` and the plane rides `Init::DataRef` per pass.
///
/// # FIX D3 (hardening session, impl-v9): THE PLANE OWNS ITS RESIDENCY
///
/// impl-v7 had ONE process-scoped plan cache, so "resident until the executor
/// drops" was the same thing as "resident once". impl-v8 gave every
/// `MarginRowBab` its own cache (`bab.rs:532`, constructed per class at
/// `:1264`/`:1408`), which turned the executor's append-only registry into a
/// term that GREW ONCE PER BaB INSTANCE — each instance's first pass
/// re-uploading the entire weight/bias plane set, permanently, on a 121 GiB
/// SHARED CPU+GPU box.
///
/// The plane therefore keeps the executor handle that minted its id and
/// RELEASES it on drop. The lifetime is exactly right by construction: the id
/// dies with the [`PlanStatic`] that owns the plane, which dies with the last
/// `Arc` a pass is holding — so a release can never race a bound buffer.
/// VALUE-NEUTRAL: while a plane is bound its bytes are identical either way;
/// releasing only frees a plane nothing can reach any more.
pub(super) struct ResidentPlane {
    pub(super) bytes: Vec<u8>,
    pub(super) resident: Option<LaneResidentId>,
    /// The executor that minted `resident` (`None` whenever `resident` is
    /// `None`). Held so the release on drop cannot target a different device.
    gpu: Option<Arc<dyn MarginRowLaneGpu>>,
}

impl ResidentPlane {
    fn new(bytes: Vec<u8>, gpu: Option<&Arc<dyn MarginRowLaneGpu>>) -> Self {
        // Fail-closed fallback: an executor refusing residency (default
        // trait impl, upload error) just means per-pass Data uploads — the
        // pre-B1 behavior, same bytes.
        match gpu.and_then(|g| g.upload_resident(&bytes).ok().map(|id| (Arc::clone(g), id))) {
            Some((g, id)) => Self {
                bytes,
                resident: Some(id),
                gpu: Some(g),
            },
            None => Self {
                bytes,
                resident: None,
                gpu: None,
            },
        }
    }
}

impl Drop for ResidentPlane {
    fn drop(&mut self) {
        // Total and infallible (a Drop must never panic): the executor's
        // `release_resident` is a no-op for an unknown or already-released id.
        if let (Some(gpu), Some(id)) = (self.gpu.as_ref(), self.resident) {
            gpu.release_resident(id);
        }
    }
}

/// Static (pass-invariant) part of one bias fold site.
pub(super) struct BiasStatic {
    /// ds-pair scale plane (per-channel expanded for conv sites).
    pub(super) plane: ResidentPlane,
    /// Up-sum of the pair masses (per-CHANNEL for conv — the `p` expansion
    /// factor is applied at the use site, exactly as the uncached code did).
    pub(super) mass: f64,
    /// Up-sum of the per-entry split-residual excesses (same convention).
    pub(super) excess: f64,
    /// The absolute bias ball ([`ds_bias_abs`]), f32-up.
    pub(super) berr: f32,
}

/// Static part of the head gemm.
pub(super) struct HeadStatic {
    /// [`ds_weight_rel`] at `0.0` — the head's ds-split-only ball.
    pub(super) rel: f64,
    pub(super) weight: ResidentPlane,
    pub(super) w_excess: f64,
    /// `norm_inflate(head_col_mass_max(..), n_y)` — cached POST-inflation.
    pub(super) a_eff: f64,
    pub(super) bias: Option<BiasStatic>,
}

/// Static part of one conv layer, in fold order (`PlanExt::conv_ops`).
pub(super) struct ConvStatic {
    /// `net.ops` index — must equal the fold-order provenance entry.
    pub(super) op_index: usize,
    /// [`ds_weight_rel`] at `cv.weight_rel_err`.
    pub(super) rel: f64,
    pub(super) weight: ResidentPlane,
    pub(super) w_excess: f64,
    /// `norm_inflate(conv_ci_mass_max(..), k_bwd)` — cached POST-inflation.
    pub(super) a_eff: f64,
    pub(super) bias: Option<BiasStatic>,
    /// ABA sentinels: first/last f64 weight, bit-compared on cache probe.
    wmat_first: u64,
    wmat_last: u64,
}

/// Fix A3: the realization probe's pass-invariant midpoint forward.
pub(super) struct ProbeStatic {
    /// Head pre-activation at `x = root.mid` (`None` = the forward failed;
    /// the probe SKIPS, exactly as the per-pass probe skipped on a failed
    /// forward — a non-evaluation proves nothing).
    pub(super) y: Option<Vec<f64>>,
    /// Per relu OP index: the FULL pre-activation vector at the midpoint
    /// (the per-pass probe collected only the dom-fixed neurons; caching
    /// all of them makes the cache dom-independent — same forward, same
    /// values, superset collection).
    pub(super) pre: BTreeMap<usize, Vec<f64>>,
}

/// Fail-closed identity fingerprint of the engine a [`PlanStatic`] was
/// built from. Pointer identity + shape census + parameter sentinels: in
/// production (one net, one root per process-instance) this is exact; the
/// residual ABA window (a DIFFERENT net later at the SAME address with
/// identical shapes and sentinel bits) is documented shadow-only exposure —
/// the shadow discards, and its enclosure compare + probe guard fail closed.
#[derive(PartialEq, Eq)]
struct Ident {
    net_ptr: usize,
    root_ptr: usize,
    n_y: usize,
    n_in: usize,
    ops_len: usize,
    layers_len: usize,
    head_w_first: u64,
    head_w_last: u64,
    mid_first: u64,
    mid_last: u64,
}

impl Ident {
    fn of(eng: &BackwardEngine<'_>) -> Self {
        let (w1, _, _) = eng.net.gemm1();
        Self {
            net_ptr: std::ptr::from_ref(eng.net) as usize,
            root_ptr: std::ptr::from_ref(eng.root) as usize,
            n_y: eng.net.n_y,
            n_in: eng.net.n_in,
            ops_len: eng.net.ops.len(),
            layers_len: eng.root.layers.len(),
            head_w_first: w1.first().copied().unwrap_or(0.0).to_bits(),
            head_w_last: w1.last().copied().unwrap_or(0.0).to_bits(),
            mid_first: eng.root.mid.first().copied().unwrap_or(0.0).to_bits(),
            mid_last: eng.root.mid.last().copied().unwrap_or(0.0).to_bits(),
        }
    }
}

/// The cached pass-invariant half of the lane's program (B1 + fix A3).
pub(crate) struct PlanStatic {
    pub(super) ext: PlanExt,
    pub(super) head: HeadStatic,
    /// Fold-order conv statics, index-aligned with `ext.conv_ops`.
    pub(super) convs: Vec<ConvStatic>,
    pub(super) probe: ProbeStatic,
    ident: Ident,
}

impl PlanStatic {
    /// Build the static half — ALL the O(params) host work of a pass: the
    /// plan walk, the ds-pair splits, the split-floor operator norms, the
    /// probe forward. `gpu: Some(..)` additionally uploads the weight
    /// planes as resident device buffers — pass it ONLY when the result
    /// will be cached (a per-pass build must not leak resident buffers).
    ///
    /// Built WITH the current `dom`, so a miss-path build refuses exactly
    /// where today's per-pass `build_plan_ext` refused.
    pub(super) fn build(
        eng: &BackwardEngine<'_>,
        dom: Option<&DomainGates>,
        gpu: Option<&Arc<dyn MarginRowLaneGpu>>,
    ) -> Result<Self, Refusal> {
        let ext = gpu_seam::build_plan_ext(eng, dom).map_err(map_seam_refusal)?;
        let net = eng.net;

        // ---- head (the apply_linear statics, verbatim helper calls) ----
        let (w1, b1, (n_y, n_h)) = net.gemm1();
        let rel = ds_weight_rel(0.0)?;
        let (hb_bytes, hb_mass, hb_excess) = ds_scale_plane(b1)?;
        let head_bias = Some(BiasStatic {
            berr: f32_up(ds_bias_abs(b1, &[])?)?,
            plane: ResidentPlane::new(hb_bytes, gpu),
            mass: hb_mass,
            excess: hb_excess,
        });
        let (hw_bytes, hw_excess) = ds_pair_plane(w1)?;
        let head = HeadStatic {
            rel,
            a_eff: norm_inflate(head_col_mass_max(w1, n_y, n_h)?, n_y),
            weight: ResidentPlane::new(hw_bytes, gpu),
            w_excess: hw_excess,
            bias: head_bias,
        };

        // ---- convs (the apply_conv statics, fold order) ----
        let mut convs = Vec::with_capacity(ext.conv_ops.len());
        for &oi in &ext.conv_ops {
            let Some(TwinOp::Conv(cv)) = net.ops.get(oi) else {
                return Err(Refusal::Unmappable("conv record is not a Conv op"));
            };
            let rel = ds_weight_rel(cv.weight_rel_err)?;
            // Bias-presence predicate = the plan builder's own
            // (`gpu_seam::conv`): nothing to carry AND nothing to charge.
            let bias = if cv.bias.iter().all(|b| *b == 0.0) && cv.bias_err.iter().all(|e| *e == 0.0)
            {
                None
            } else {
                let p = cv.oshape.1 * cv.oshape.2;
                let mut expanded = Vec::with_capacity(cv.bias.len() * p * 8);
                let mut mass = 0.0f64;
                let mut excess = 0.0f64;
                for &b in &cv.bias {
                    let (hi, lo) = transaction::ds_split_f64(b)?;
                    mass = transaction::add_up(
                        mass,
                        super::super::rounding::next_up(f64::from(hi).abs() + f64::from(lo).abs()),
                    );
                    excess = transaction::add_up(
                        excess,
                        transaction::ds_split_excess(b, hi, lo, transaction::DS_SPLIT_COVER),
                    );
                    for _ in 0..p {
                        expanded.extend_from_slice(&hi.to_le_bytes());
                        expanded.extend_from_slice(&lo.to_le_bytes());
                    }
                }
                Some(BiasStatic {
                    berr: f32_up(ds_bias_abs(&cv.bias, &cv.bias_err)?)?,
                    plane: ResidentPlane::new(expanded, gpu),
                    mass,
                    excess,
                })
            };
            let (w_bytes, w_excess) = ds_pair_plane(&cv.wmat)?;
            let k_bwd = cv.kernel.0 * cv.kernel.2 * cv.kernel.3;
            convs.push(ConvStatic {
                op_index: oi,
                rel,
                a_eff: norm_inflate(
                    conv_ci_mass_max(
                        &cv.wmat,
                        cv.kernel.0,
                        cv.kernel.1,
                        cv.kernel.2 * cv.kernel.3,
                    )?,
                    k_bwd,
                ),
                weight: ResidentPlane::new(w_bytes, gpu),
                w_excess,
                bias,
                wmat_first: cv.wmat.first().copied().unwrap_or(0.0).to_bits(),
                wmat_last: cv.wmat.last().copied().unwrap_or(0.0).to_bits(),
            });
        }

        // ---- fix A3: the probe's midpoint forward, once ----
        let probe = build_probe_static(eng);

        Ok(Self {
            ext,
            head,
            convs,
            probe,
            ident: Ident::of(eng),
        })
    }

    /// Fail-closed identity check against a (possibly different) engine.
    fn matches(&self, eng: &BackwardEngine<'_>) -> bool {
        if self.ident != Ident::of(eng) {
            return false;
        }
        // Per-conv sentinels (the per-pass `apply_conv` shape cross-check
        // still runs on every layer regardless).
        self.convs
            .iter()
            .all(|cs| match eng.net.ops.get(cs.op_index) {
                Some(TwinOp::Conv(cv)) => {
                    cv.wmat.first().copied().unwrap_or(0.0).to_bits() == cs.wmat_first
                        && cv.wmat.last().copied().unwrap_or(0.0).to_bits() == cs.wmat_last
                }
                _ => false,
            })
    }
}

/// One exact f64 forward at `x = root.mid` with pre-activation collection
/// over EVERY trunk relu layer (superset of any dom's fixes — see
/// [`ProbeStatic::pre`]).
fn build_probe_static(eng: &BackwardEngine<'_>) -> ProbeStatic {
    let net = eng.net;
    let mut x = Array2::<f64>::zeros((net.n_in, 1));
    for i in 0..net.n_in.min(eng.root.mid.len()) {
        x[[i, 0]] = eng.root.mid[i];
    }
    let sel: BTreeMap<usize, Vec<usize>> = eng
        .root
        .layers
        .iter()
        .map(|rec| (rec.op, (0..rec.n).collect()))
        .collect();
    match net.forward_points(&x, &sel) {
        Ok((y, pre)) => ProbeStatic {
            y: Some((0..net.n_y).map(|j| y[[j, 0]]).collect()),
            pre: pre
                .into_iter()
                .map(|(op, vals)| (op, (0..vals.nrows()).map(|r| vals[[r, 0]]).collect()))
                .collect(),
        },
        // A forward failure is not evidence of unsoundness; every pass
        // skips the probe (the per-pass guard's own rule).
        Err(_) => ProbeStatic {
            y: None,
            pre: BTreeMap::new(),
        },
    }
}

/// The cached realization probe (fix A3): identical decision procedure to
/// `gpu_seam::probe_guard`, consuming the cached `(y, pre)` instead of
/// re-running the midpoint forward per pass. A published bound that the
/// realized midpoint value violates is unsound, full stop; a midpoint that
/// is not a member of the current domain proves nothing and SKIPS.
pub(super) fn probe_guard_cached(
    eng: &BackwardEngine<'_>,
    seed: &Seed,
    dom: Option<&DomainGates>,
    dir: LaneDir,
    out: &PassOut,
    probe: &ProbeStatic,
) -> Result<(), Refusal> {
    let Some(y) = probe.y.as_ref() else {
        return Ok(()); // forward failed at build: skip, never trip
    };
    let rows = seed.s.ncols();
    let fixes = gpu_seam::domain_fixes(eng, dom);
    for (op, list) in &fixes {
        let Some(vals) = probe.pre.get(op) else {
            return Ok(());
        };
        for (j, positive) in list {
            let Some(&v) = vals.get(*j) else {
                return Ok(());
            };
            if (*positive && v < 0.0) || (!*positive && v > 0.0) {
                // The probe point is outside this domain: not a counterexample.
                return Ok(());
            }
        }
    }
    let bound = match dir {
        LaneDir::Lower => eng.concretize_lower(out),
        LaneDir::Upper => eng.concretize_upper(out),
    };
    if bound.len() != rows {
        return Err(Refusal::Payload);
    }
    let ss = seed.s.as_slice().ok_or(Refusal::Payload)?;
    for r in 0..rows {
        let mut f = 0.0f64;
        for (j, yj) in y.iter().enumerate() {
            f += ss[j * rows + r] * yj;
        }
        if !f.is_finite() {
            return Ok(());
        }
        let tol = gpu_seam::PROBE_REL_SLACK * (1.0 + f.abs());
        let violated = match dir {
            LaneDir::Lower => bound[r] > f + tol,
            LaneDir::Upper => bound[r] < f - tol,
        };
        if violated {
            return Err(Refusal::Probe);
        }
    }
    Ok(())
}

/// Per-pass GATE ADMISSION on a cache hit: the cached plan skeleton is
/// dom-independent, but a fresh `build_plan_ext` would have re-run
/// [`gpu_seam::outward_gate_f32`] on THIS dom's gates (fractional-alpha
/// refusal, non-f32-exact repair) — so a hit re-runs exactly that check
/// over the fold-order relu list. Refusal-equivalence note: the dom-
/// INDEPENDENT refusals of the builder's activation arm (non-finite
/// `l`/`u`, `node_abs` overflow) were validated when the cached plan was
/// built and cannot differ per pass; only the refusal STRING of a
/// multi-defect pass could differ, never the admit/refuse decision.
pub(super) fn check_gate_admission(
    eng: &BackwardEngine<'_>,
    dom: Option<&DomainGates>,
    relus: &[usize],
) -> Result<(), Refusal> {
    for &li in relus {
        let rec = eng
            .root
            .layers
            .get(li)
            .ok_or(Refusal::Unmappable("relu layer index out of range"))?;
        let (alpha, s, c) = match dom.and_then(|d| d.layers.get(&li)) {
            Some(gv) => (&gv.alpha[..], &gv.s[..], &gv.c[..]),
            None => (&rec.alpha[..], &rec.s[..], &rec.c[..]),
        };
        if rec.n == 0
            || alpha.len() != rec.n
            || s.len() != rec.n
            || c.len() != rec.n
            || rec.l.len() != rec.n
            || rec.u.len() != rec.n
        {
            return Err(Refusal::Unmappable("relu gate width"));
        }
        gpu_seam::outward_gate_f32(alpha, s, c, &rec.l, &rec.u).map_err(map_seam_refusal)?;
    }
    Ok(())
}

/// One-slot residency cache with hit/miss telemetry. OWNED BY THE CALLER
/// (module docs: this session, the shadow entry; post-B4, the BaB struct).
#[derive(Default)]
pub(crate) struct PlanCache {
    slot: Mutex<Option<Arc<PlanStatic>>>,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl PlanCache {
    /// A validated hit, or `None` (= miss; the caller builds and
    /// [`Self::store`]s). A fingerprint mismatch is a MISS, never an error.
    pub(crate) fn probe(&self, eng: &BackwardEngine<'_>) -> Option<Arc<PlanStatic>> {
        let slot = self
            .slot
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match slot.as_ref() {
            Some(ps) if ps.matches(eng) => {
                self.hits.fetch_add(1, Ordering::Relaxed);
                Some(Arc::clone(ps))
            }
            _ => {
                self.misses.fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }

    /// Install (replacing any previous static — one live engine at a time).
    pub(crate) fn store(&self, ps: Arc<PlanStatic>) {
        *self
            .slot
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(ps);
    }

    /// (hits, misses) — R9 telemetry, test-pinned.
    pub(crate) fn counters(&self) -> (u64, u64) {
        (
            self.hits.load(Ordering::Relaxed),
            self.misses.load(Ordering::Relaxed),
        )
    }
}
