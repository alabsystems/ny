// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `#wave-beta-bab` (ENGINE_PROGRAM M6) tests: the margin-row wave tree on
//! INJECTED test backends (the real ny-bbe search is exercised end to end in
//! ny-cli, which may depend on both crates). A child of `margin_row::tests`
//! so it reuses the synthetic nets and oracles there.
//!
//! The backends are doubles: `ClaimAll` (every domain claims, candidates in
//! reverse universe order — the lane's certify queue decides everything),
//! `Garbage` (hostile bounds, multipliers, slopes and candidates: NaN, inf,
//! negative, out of range, repeated, already split), `Refuse` (the upload
//! refuses), `Panics` (the wave search panics) and `Tripwire` (counts every
//! call).
//!
//! Bit identity of the default:
//! * `wave_default_is_the_legacy_tree_bit_identical` (the backend is never
//!   consulted);
//! * `wave_refused_upload_runs_the_legacy_tree_bit_identical`.
//!
//! Soundness:
//! * `wave_leaves_hold_at_sampled_points` (containment of every certified
//!   leaf; the positive control that the wave proves through a real tree);
//! * `wave_garbage_backends_never_close_a_witnessed_sat_class` (fault
//!   injection);
//! * `wave_search_panics_and_refusals_never_make_a_verdict`.
//!
//! Mutation pins:
//! * `wave_claims_are_not_closes` (the claim-as-close mutant passes the cover
//!   gate on a SAT instance; only certification stops the real path);
//! * `wave_clip_side_is_pinned_on_both_sides` (the clip rung's half-space side,
//!   On and Off, against sampled truth);
//! * `wave_cover_gate_refuses_a_dropped_leaf`;
//! * `wave_certification_errors_never_close` (`Err` is never a close).
//!
//! Protocol:
//! * `wave_is_thread_count_independent` (certify-pool width 1 vs 4: identical
//!   trajectory and leaves).
//!
//! The `wave_cuda` mode's backend selection (the device first, then the CPU
//! search, then the legacy tree; the routing is `ny_core::search::
//! FallbackBackend`'s, the lane only composes it and prints its route):
//! * `wave_cuda_uses_the_device_backend_when_its_upload_is_admitted` (the CPU
//!   backend is never consulted; the same tree as `wave` on the same search);
//! * `wave_cuda_refused_device_falls_through_to_the_cpu_search` (a refused or
//!   missing device is named on the receipt and the CPU search runs the tree);
//! * `wave_mode_never_consults_the_device_backend`;
//! * `wave_cuda_with_every_backend_refusing_runs_the_legacy_tree_bit_identical`;
//! * `wave_cuda_device_fault_mid_tree_switches_to_the_cpu_search` (fault
//!   injection: a device that faults after its first wave search and latches;
//!   the faulted call re-runs on the CPU search, so the tree is the CPU tree,
//!   and the receipt names BOTH backends: `backend=<device>-><cpu>
//!   switched_at_wave= switch_reason=`);
//! * `wave_cuda_host_refusals_keep_the_device_and_reach_the_receipt` (a
//!   host-side refusal of either call, or a malformed scoring, is that call's
//!   alone: no switch, the CPU backend never consulted, `backend=` still the
//!   device's alone, and the refusal named on `first_refusal=`).

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use ndarray::Array2;
use ny_core::search::{
    BackendCaps, BranchOut, DeviceLease, Refusal, SearchBackend, SearchPrecision, Untrusted,
    WaveBatch, WaveHandle, WaveOut, WaveProblem, WaveSearchCfg, WaveStats, WAVE_NONE,
};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use super::{margins_at, sample_box, tiny_spec, tiny_targets, verdict_bits};
use crate::margin_row::bab::wave::{
    clip_ybox_for_test, take_last_wave_summary, with_test_backend, with_test_backends,
    RecordedLeaf, TreeMode, WaveHooks, WavePolicy,
};
use crate::margin_row::bab::{BabConfig, MarginRowBab, MarginRowOutcome};
use crate::margin_row::net::TwinNet;
use crate::margin_row::root::{ClipRows, RootGates};
use crate::margin_row::rounding::RoundMode;
use crate::margin_row::spec::{TwinOpSpec, TwinSpec};

// ---------------------------------------------------------------------------
//  Test backends
// ---------------------------------------------------------------------------

fn caps() -> BackendCaps {
    BackendCaps::Cpu {
        threads: 1,
        lanes: 1,
        precision: SearchPrecision::F64,
    }
}

fn domain_lits<'b>(batch: &WaveBatch<'b>, d: usize) -> &'b [ny_core::search::WaveSplit] {
    &batch.splits[batch.offsets[d] as usize..batch.offsets[d + 1] as usize]
}

/// Every domain claims; candidates are the unsplit universe positions in
/// REVERSE order (head neurons first).
#[derive(Default)]
struct ClaimAll {
    searches: AtomicUsize,
}

impl SearchBackend for ClaimAll {
    fn name(&self) -> &'static str {
        "claim_all"
    }

    fn caps(&self) -> BackendCaps {
        caps()
    }

    fn upload_wave(
        &self,
        problem: &WaveProblem,
        _lease: &DeviceLease,
    ) -> Result<WaveHandle, Refusal> {
        problem.validate().map_err(Refusal::Invalid)?;
        Ok(WaveHandle::new("claim_all", problem.universe.len(), ()))
    }

    fn backward_batch(
        &self,
        handle: &WaveHandle,
        batch: &WaveBatch<'_>,
        _cfg: &WaveSearchCfg,
        out: &mut WaveOut,
    ) -> Result<Untrusted<WaveStats>, Refusal> {
        self.searches.fetch_add(1, Ordering::Relaxed);
        out.prepare(batch.count, handle.stride(), batch.splits.len());
        let w = out.writer();
        for k in 0..batch.count {
            w.lb[k] = 1.0;
            w.scale[k] = 1.0;
            w.evals[k] = 1;
        }
        Ok(Untrusted::new(WaveStats {
            domains: batch.count as u64,
            ..WaveStats::default()
        }))
    }

    fn branch_scores(
        &self,
        handle: &WaveHandle,
        batch: &WaveBatch<'_>,
        k: usize,
        out: &mut BranchOut,
    ) -> Result<Untrusted<()>, Refusal> {
        out.prepare(batch.count, k);
        let w = out.writer();
        for d in 0..batch.count {
            let lits = domain_lits(batch, d);
            let mut c = 0usize;
            for p in (0..handle.stride() as u32).rev() {
                if c == k {
                    break;
                }
                if lits.iter().any(|l| l.pos == p) {
                    continue;
                }
                w.pos[d * k + c] = p;
                w.score[d * k + c] = 1.0;
                c += 1;
            }
        }
        Ok(Untrusted::new(()))
    }
}

/// Hostile outputs: `lb` +inf / huge / NaN / small; multipliers NaN,
/// negative, huge, inf; slopes NaN, negative, above 1; candidates padding,
/// out of range, repeated or already split.
struct Garbage {
    state: AtomicU64,
    /// Huge multipliers (they may make the certified pass fail closed).
    huge: bool,
}

impl Garbage {
    fn new(seed: u64, huge: bool) -> Self {
        Self {
            state: AtomicU64::new(seed),
            huge,
        }
    }

    fn next(&self) -> u64 {
        let s = self
            .state
            .fetch_add(0x9E37_79B9_7F4A_7C15, Ordering::Relaxed)
            .wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn pick<T: Copy>(&self, xs: &[T]) -> T {
        xs[(self.next() % xs.len() as u64) as usize]
    }
}

impl SearchBackend for Garbage {
    fn name(&self) -> &'static str {
        "garbage"
    }

    fn caps(&self) -> BackendCaps {
        caps()
    }

    fn upload_wave(
        &self,
        problem: &WaveProblem,
        _lease: &DeviceLease,
    ) -> Result<WaveHandle, Refusal> {
        Ok(WaveHandle::new("garbage", problem.universe.len(), ()))
    }

    fn backward_batch(
        &self,
        handle: &WaveHandle,
        batch: &WaveBatch<'_>,
        _cfg: &WaveSearchCfg,
        out: &mut WaveOut,
    ) -> Result<Untrusted<WaveStats>, Refusal> {
        out.prepare(batch.count, handle.stride(), batch.splits.len());
        let w = out.writer();
        for k in 0..batch.count {
            w.lb[k] = self.pick(&[f32::INFINITY, 1e30, f32::NAN, 0.5, -0.25, f32::MAX]);
            w.scale[k] = self.pick(&[f32::NAN, 1.0, 0.0, f32::INFINITY, 1e-30]);
        }
        let betas: &[f32] = if self.huge {
            &[f32::NAN, -5.0, 1e30, 0.25, f32::INFINITY, 3.0]
        } else {
            &[f32::NAN, -5.0, 0.25, 2.0, 0.0]
        };
        for b in w.beta.iter_mut() {
            *b = self.pick(betas);
        }
        for a in w.alpha.iter_mut() {
            *a = self.pick(&[f32::NAN, -1.0, 7.0, 0.5, 0.0, 1.0]);
        }
        Ok(Untrusted::new(WaveStats::default()))
    }

    fn branch_scores(
        &self,
        handle: &WaveHandle,
        batch: &WaveBatch<'_>,
        k: usize,
        out: &mut BranchOut,
    ) -> Result<Untrusted<()>, Refusal> {
        out.prepare(batch.count, k);
        let stride = handle.stride() as u64;
        let w = out.writer();
        for d in 0..batch.count {
            let own = domain_lits(batch, d).first().map_or(WAVE_NONE, |l| l.pos);
            for c in 0..k {
                w.pos[d * k + c] = match self.next() % 4 {
                    0 => WAVE_NONE,
                    1 => (stride + 3) as u32,
                    2 => own,
                    _ => (self.next() % stride.max(1)) as u32,
                };
                w.score[d * k + c] = self.pick(&[f32::NAN, -1.0, 2.0]);
            }
        }
        Ok(Untrusted::new(()))
    }
}

/// The upload refuses.
struct Refuse;

impl SearchBackend for Refuse {
    fn name(&self) -> &'static str {
        "refuse"
    }

    fn caps(&self) -> BackendCaps {
        caps()
    }

    fn upload_wave(
        &self,
        _problem: &WaveProblem,
        _lease: &DeviceLease,
    ) -> Result<WaveHandle, Refusal> {
        Err(Refusal::Unsupported("upload_wave"))
    }
}

/// The upload succeeds; the wave search panics and branching refuses.
struct Panics;

impl SearchBackend for Panics {
    fn name(&self) -> &'static str {
        "panics"
    }

    fn caps(&self) -> BackendCaps {
        caps()
    }

    fn upload_wave(
        &self,
        problem: &WaveProblem,
        _lease: &DeviceLease,
    ) -> Result<WaveHandle, Refusal> {
        Ok(WaveHandle::new("panics", problem.universe.len(), ()))
    }

    fn backward_batch(
        &self,
        _handle: &WaveHandle,
        _batch: &WaveBatch<'_>,
        _cfg: &WaveSearchCfg,
        _out: &mut WaveOut,
    ) -> Result<Untrusted<WaveStats>, Refusal> {
        panic!("wave search backend panicked (injected)");
    }

    fn branch_scores(
        &self,
        _handle: &WaveHandle,
        _batch: &WaveBatch<'_>,
        _k: usize,
        _out: &mut BranchOut,
    ) -> Result<Untrusted<()>, Refusal> {
        Err(Refusal::Unsupported("branch_scores"))
    }
}

/// Counts every call; never answers.
#[derive(Default)]
struct Tripwire {
    calls: AtomicUsize,
}

impl SearchBackend for Tripwire {
    fn name(&self) -> &'static str {
        "tripwire"
    }

    fn caps(&self) -> BackendCaps {
        self.calls.fetch_add(1, Ordering::Relaxed);
        caps()
    }

    fn upload_wave(
        &self,
        _problem: &WaveProblem,
        _lease: &DeviceLease,
    ) -> Result<WaveHandle, Refusal> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Err(Refusal::Unsupported("tripwire"))
    }
}

/// Another backend under a different name (the device double of
/// `wave_cuda`: same answers, a name the receipt can tell apart).
struct Renamed<B> {
    name: &'static str,
    inner: B,
}

impl<B: SearchBackend> SearchBackend for Renamed<B> {
    fn name(&self) -> &'static str {
        self.name
    }

    fn caps(&self) -> BackendCaps {
        self.inner.caps()
    }

    fn upload_wave(
        &self,
        problem: &WaveProblem,
        lease: &DeviceLease,
    ) -> Result<WaveHandle, Refusal> {
        self.inner.upload_wave(problem, lease)
    }

    fn backward_batch(
        &self,
        handle: &WaveHandle,
        batch: &WaveBatch<'_>,
        cfg: &WaveSearchCfg,
        out: &mut WaveOut,
    ) -> Result<Untrusted<WaveStats>, Refusal> {
        self.inner.backward_batch(handle, batch, cfg, out)
    }

    fn branch_scores(
        &self,
        handle: &WaveHandle,
        batch: &WaveBatch<'_>,
        k: usize,
        out: &mut BranchOut,
    ) -> Result<Untrusted<()>, Refusal> {
        self.inner.branch_scores(handle, batch, k, out)
    }
}

/// Uploads and answers its first `ok` wave searches like `inner`, then
/// refuses every later search and scoring (a device that faults mid-tree and
/// latches itself off).
struct FaultAfter<B> {
    name: &'static str,
    ok: usize,
    calls: AtomicUsize,
    inner: B,
}

impl<B: SearchBackend> SearchBackend for FaultAfter<B> {
    fn name(&self) -> &'static str {
        self.name
    }

    fn caps(&self) -> BackendCaps {
        self.inner.caps()
    }

    fn upload_wave(
        &self,
        problem: &WaveProblem,
        lease: &DeviceLease,
    ) -> Result<WaveHandle, Refusal> {
        self.inner.upload_wave(problem, lease)
    }

    fn backward_batch(
        &self,
        handle: &WaveHandle,
        batch: &WaveBatch<'_>,
        cfg: &WaveSearchCfg,
        out: &mut WaveOut,
    ) -> Result<Untrusted<WaveStats>, Refusal> {
        if self.calls.fetch_add(1, Ordering::Relaxed) >= self.ok {
            return Err(Refusal::Fault(
                "cuda wave search: device fault: injected".into(),
            ));
        }
        self.inner.backward_batch(handle, batch, cfg, out)
    }

    fn branch_scores(
        &self,
        handle: &WaveHandle,
        batch: &WaveBatch<'_>,
        k: usize,
        out: &mut BranchOut,
    ) -> Result<Untrusted<()>, Refusal> {
        if self.calls.load(Ordering::Relaxed) > self.ok {
            return Err(Refusal::Fault(
                "cuda wave search: device fault: injected".into(),
            ));
        }
        self.inner.branch_scores(handle, batch, k, out)
    }
}

/// Answers like `inner`, except for HOST-side refusals (`Refusal::Invalid`,
/// never a fault) of the listed calls, and a malformed first scoring when
/// `malformed_branch` (a device whose bounds check refused a slab: the
/// device stays usable).
struct HostRefuses<B> {
    name: &'static str,
    inner: B,
    search_refuses: &'static [usize],
    branch_refuses: &'static [usize],
    malformed_branch: bool,
    searches: AtomicUsize,
    branches: AtomicUsize,
}

impl<B> HostRefuses<B> {
    fn new(inner: B, search_refuses: &'static [usize], branch_refuses: &'static [usize]) -> Self {
        Self {
            name: "device_host",
            inner,
            search_refuses,
            branch_refuses,
            malformed_branch: false,
            searches: AtomicUsize::new(0),
            branches: AtomicUsize::new(0),
        }
    }
}

impl<B: SearchBackend> SearchBackend for HostRefuses<B> {
    fn name(&self) -> &'static str {
        self.name
    }

    fn caps(&self) -> BackendCaps {
        self.inner.caps()
    }

    fn upload_wave(
        &self,
        problem: &WaveProblem,
        lease: &DeviceLease,
    ) -> Result<WaveHandle, Refusal> {
        self.inner.upload_wave(problem, lease)
    }

    fn backward_batch(
        &self,
        handle: &WaveHandle,
        batch: &WaveBatch<'_>,
        cfg: &WaveSearchCfg,
        out: &mut WaveOut,
    ) -> Result<Untrusted<WaveStats>, Refusal> {
        if self
            .search_refuses
            .contains(&self.searches.fetch_add(1, Ordering::Relaxed))
        {
            return Err(Refusal::Invalid(
                "cuda wave search: host-side: injected".into(),
            ));
        }
        self.inner.backward_batch(handle, batch, cfg, out)
    }

    fn branch_scores(
        &self,
        handle: &WaveHandle,
        batch: &WaveBatch<'_>,
        k: usize,
        out: &mut BranchOut,
    ) -> Result<Untrusted<()>, Refusal> {
        let i = self.branches.fetch_add(1, Ordering::Relaxed);
        if self.branch_refuses.contains(&i) {
            return Err(Refusal::Invalid(
                "cuda wave search: host-side scoring: injected".into(),
            ));
        }
        if self.malformed_branch && i == 0 {
            out.prepare(batch.count + 1, k);
            return Ok(Untrusted::new(()));
        }
        self.inner.branch_scores(handle, batch, k, out)
    }
}

// ---------------------------------------------------------------------------
//  Fixtures
// ---------------------------------------------------------------------------

fn wave_cfg(cert_threads: usize) -> BabConfig {
    BabConfig {
        frontier: 1,
        frontier_contextual: 1,
        lru_cap: 64,
        max_expansions: 4000,
        deadline: Some(Instant::now() + Duration::from_secs(120)),
        wave: WavePolicy {
            mode: TreeMode::Wave,
            batch: 16,
            cert_threads,
        },
        ..BabConfig::default()
    }
}

fn legacy_cfg() -> BabConfig {
    BabConfig {
        frontier: 1,
        frontier_contextual: 1,
        lru_cap: 64,
        max_expansions: 400,
        ..BabConfig::default()
    }
}

/// A tiny instance the legacy lane proves only THROUGH a tree (so the wave
/// is entered), with the wave (on `ClaimAll`) proving it too. Panics VOID
/// when no seed qualifies.
struct Fixture {
    net: TwinNet,
    root: RootGates,
    t: usize,
    adv: Vec<usize>,
    lo: Vec<f64>,
    hi: Vec<f64>,
}

fn tree_fixture() -> Fixture {
    for seed in 0..64u64 {
        let mut rng = StdRng::seed_from_u64(seed);
        let spec = tiny_spec(&mut rng, 0.35);
        let Ok(net) = TwinNet::compile(&spec) else {
            continue;
        };
        for width in [0.01, 0.02, 0.04] {
            let lo = vec![0.01; 72];
            let hi: Vec<f64> = lo.iter().map(|l| l + width).collect();
            let (t, _worst, adv) = tiny_targets(&net, &lo, &hi);
            let Ok(root) = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None) else {
                continue;
            };
            let MarginRowOutcome::Unsat(stats, _) =
                MarginRowBab::run(&net, &root, t, &adv, legacy_cfg())
            else {
                continue;
            };
            if stats.expansions == 0 {
                continue;
            }
            let (wave, _) =
                with_test_backend(Arc::new(ClaimAll::default()), WaveHooks::default(), || {
                    MarginRowBab::run(&net, &root, t, &adv, wave_cfg(2))
                });
            if matches!(wave, MarginRowOutcome::Unsat(..)) {
                return Fixture {
                    net,
                    root,
                    t,
                    adv,
                    lo,
                    hi,
                };
            }
        }
    }
    panic!("no seeded fixture closes through a wave tree: the wave tests are VOID");
}

/// A tiny instance violated at its box midpoint (the orientation swap of
/// `lane_verdicts_unsat_and_fail_closed`): no sound lane may return `Unsat`.
fn sat_fixture(seed: u64) -> (TwinNet, RootGates, usize, Vec<usize>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let spec = tiny_spec(&mut rng, 0.35);
    let net = TwinNet::compile(&spec).expect("compile");
    let lo = vec![0.01; 72];
    let hi = vec![0.05; 72];
    let (t, worst, _) = tiny_targets(&net, &lo, &hi);
    // WITNESS the violation rather than infer it from the class ordering:
    // the property `y_worst - y_t > 0` fails at the box midpoint, so any
    // `Unsat` on this fixture is a false verdict.
    let mid = Array2::from_shape_fn((lo.len(), 1), |(i, _)| f64::midpoint(lo[i], hi[i]));
    let (y_mid, _) = net
        .forward_points(&mid, &std::collections::BTreeMap::new())
        .expect("forward");
    let m = margins_at(&net, &y_mid, worst, &[t])[0][0];
    assert!(
        m <= 0.0,
        "the SAT fixture (seed {seed}) is not violated at its midpoint: margin {m}"
    );
    let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
    (net, root, worst, vec![t])
}

/// Whether the last armed tree on this thread ran the WAVE engine (its
/// receipt, without a legacy fallback). Clears the receipt.
fn wave_engaged() -> bool {
    take_last_wave_summary()
        .is_some_and(|line| line.contains("wave-beta-bab/v1") && !line.contains("fallback=legacy"))
}

fn stats_key(out: &MarginRowOutcome) -> (bool, u64, usize, usize, usize, String) {
    let (verdict, rb, exp, stop) = verdict_bits(out);
    let (created, closed) = match out {
        MarginRowOutcome::Unsat(s, _) => (s.domains_created, s.closed),
        MarginRowOutcome::Unknown { stats, .. } => stats
            .as_ref()
            .map_or((0, 0), |s| (s.domains_created, s.closed)),
    };
    (verdict, rb, exp, created, closed, stop)
}

fn leaf_key(
    leaves: &[RecordedLeaf],
) -> Vec<(
    usize,
    Vec<(usize, usize, i8)>,
    Vec<(usize, i8)>,
    u64,
    &'static str,
)> {
    leaves
        .iter()
        .map(|l| {
            (
                l.class,
                l.trunk.clone(),
                l.heads.clone(),
                l.value.to_bits(),
                l.rung,
            )
        })
        .collect()
}

// ---------------------------------------------------------------------------
//  Bit identity of the default
// ---------------------------------------------------------------------------

#[test]
fn wave_default_is_the_legacy_tree_bit_identical() {
    assert_eq!(BabConfig::default().wave, WavePolicy::default());
    assert!(!BabConfig::default().wave.armed());
    assert_eq!(WavePolicy::default().mode, TreeMode::Legacy);
    let fx = tree_fixture();
    let tripwire = Arc::new(Tripwire::default());
    let plain = MarginRowBab::run(&fx.net, &fx.root, fx.t, &fx.adv, legacy_cfg());
    // An explicit legacy policy with odd knobs and a tripwire backend
    // injected: the knobs are inert and the backend is never consulted.
    let legacy = BabConfig {
        wave: WavePolicy {
            mode: TreeMode::Legacy,
            batch: 7,
            cert_threads: 3,
        },
        ..legacy_cfg()
    };
    let _ = take_last_wave_summary();
    let (dark, leaves) = with_test_backend(tripwire.clone(), WaveHooks::default(), || {
        MarginRowBab::run(&fx.net, &fx.root, fx.t, &fx.adv, legacy)
    });
    assert_eq!(
        take_last_wave_summary(),
        None,
        "a legacy tree printed a wave receipt"
    );
    assert_eq!(stats_key(&plain), stats_key(&dark));
    assert_eq!(
        tripwire.calls.load(Ordering::Relaxed),
        0,
        "legacy consulted a backend"
    );
    assert!(leaves.is_empty(), "legacy recorded wave leaves");
}

#[test]
fn wave_refused_upload_runs_the_legacy_tree_bit_identical() {
    let fx = tree_fixture();
    let plain = MarginRowBab::run(&fx.net, &fx.root, fx.t, &fx.adv, legacy_cfg());
    let armed = BabConfig {
        wave: WavePolicy {
            mode: TreeMode::Wave,
            batch: 16,
            cert_threads: 2,
        },
        ..legacy_cfg()
    };
    let (refused, leaves) = with_test_backend(Arc::new(Refuse), WaveHooks::default(), || {
        MarginRowBab::run(&fx.net, &fx.root, fx.t, &fx.adv, armed)
    });
    let line = take_last_wave_summary().expect("an armed tree always prints its receipt");
    assert!(
        line.contains("backend=refuse refused=") && line.contains("fallback=legacy"),
        "{line}"
    );
    assert_eq!(stats_key(&plain), stats_key(&refused));
    assert!(leaves.is_empty());
}

// ---------------------------------------------------------------------------
//  Soundness
// ---------------------------------------------------------------------------

#[test]
fn wave_leaves_hold_at_sampled_points() {
    let fx = tree_fixture();
    let (out, leaves) =
        with_test_backend(Arc::new(ClaimAll::default()), WaveHooks::default(), || {
            MarginRowBab::run(&fx.net, &fx.root, fx.t, &fx.adv, wave_cfg(2))
        });
    let line = take_last_wave_summary().expect("an armed tree always prints its receipt");
    assert!(
        line.contains("[bbe] phase=wave armed=1 backend=claim_all")
            && line.contains("wave-beta-bab/v1")
            && !line.contains("fallback=legacy"),
        "the wave engine did not run: {line}"
    );
    let MarginRowOutcome::Unsat(stats, token) = out else {
        panic!("the fixture's wave must prove");
    };
    assert!(token.publishable());
    assert_eq!(stats.ledger_ok, Some(true));
    assert!(
        !leaves.is_empty(),
        "a tree-closed wave recorded no leaf: VOID"
    );
    assert!(
        leaves
            .iter()
            .any(|l| !l.trunk.is_empty() || !l.heads.is_empty()),
        "no leaf below a class root: the tree was not exercised"
    );
    // Every certified value is a strict close.
    for l in &leaves {
        assert!(l.value > 0.0 && l.value.is_finite(), "{l:?}");
        assert_ne!(l.rung, "mutant");
    }
    // Containment: every sampled point inside a leaf's region has a margin
    // at least the leaf's certified bound; an emptiness leaf holds no point.
    let mut rng = StdRng::seed_from_u64(0x3a7e);
    let x = sample_box(&mut rng, &fx.root, 2000);
    let sel: std::collections::BTreeMap<usize, Vec<usize>> = fx
        .root
        .layers
        .iter()
        .map(|layer| (layer.op, layer.unst.clone()))
        .collect();
    let (y, pre) = fx.net.forward_points(&x, &sel).expect("forward");
    let mut inside_total = 0usize;
    for l in &leaves {
        let margins = margins_at(&fx.net, &y, fx.t, &[l.class]);
        for b in 0..x.ncols() {
            let inside = l.trunk.iter().all(|&(li, pos, dir)| {
                let z = pre[&fx.root.layers[li].op][[pos, b]];
                if dir > 0 {
                    z >= 0.0
                } else {
                    z <= 0.0
                }
            }) && l.heads.iter().all(|&(i, dir)| {
                if dir > 0 {
                    y[[i, b]] >= 0.0
                } else {
                    y[[i, b]] <= 0.0
                }
            });
            if !inside {
                continue;
            }
            inside_total += 1;
            assert!(
                !l.empty,
                "a certified-EMPTY leaf holds sampled point {b}: {l:?}"
            );
            let m = margins[0][b];
            assert!(
                m >= l.value - 1e-12 * (1.0 + m.abs()),
                "CONTAINMENT VIOLATION: point {b} in leaf {l:?} has margin {m}"
            );
        }
    }
    assert!(inside_total > 0, "no sampled point fell in any leaf: VOID");
    // The whole instance is robust at every sample (the verdict's claim).
    for (k, row) in margins_at(&fx.net, &y, fx.t, &fx.adv).iter().enumerate() {
        assert!(
            row.iter().all(|&m| m > 0.0),
            "class {k} violated at a sample"
        );
    }
    let _ = (&fx.lo, &fx.hi);
}

#[test]
fn wave_garbage_backends_never_close_a_witnessed_sat_class() {
    let mut runs = 0usize;
    let mut engaged = 0usize;
    for seed in [29u64, 31, 37] {
        let (net, root, t, adv) = sat_fixture(seed);
        for (tag, huge) in [("finite", false), ("huge", true)] {
            for bseed in 0..3u64 {
                let backend = Arc::new(Garbage::new(seed * 101 + bseed, huge));
                let cfg = BabConfig {
                    max_expansions: 300,
                    ..wave_cfg(2)
                };
                let _ = take_last_wave_summary();
                let (out, leaves) = with_test_backend(backend, WaveHooks::default(), || {
                    MarginRowBab::run(&net, &root, t, &adv, cfg)
                });
                assert!(
                    !matches!(out, MarginRowOutcome::Unsat(..)),
                    "MOAT VIOLATION: a {tag} garbage backend (seed {seed}/{bseed}) closed a \
                     witnessed SAT instance"
                );
                assert!(leaves.iter().all(|l| l.rung != "mutant"));
                engaged += usize::from(wave_engaged());
                runs += 1;
            }
        }
        // The honest-shaped claim-everything backend never closes it either.
        let (out, _) =
            with_test_backend(Arc::new(ClaimAll::default()), WaveHooks::default(), || {
                MarginRowBab::run(
                    &net,
                    &root,
                    t,
                    &adv,
                    BabConfig {
                        max_expansions: 300,
                        ..wave_cfg(2)
                    },
                )
            });
        assert!(!matches!(out, MarginRowOutcome::Unsat(..)));
    }
    assert_eq!(runs, 18);
    // A garbage run that fell back to the legacy tree tested the legacy tree,
    // not the wave's ingest gates: the moat is VOID unless the wave engine
    // itself met the garbage.
    assert!(
        engaged > 0,
        "no garbage run reached the wave engine (every run fell back): VOID"
    );
}

#[test]
fn wave_search_panics_and_refusals_never_make_a_verdict() {
    // A panicking search on a SAT instance: never Unsat.
    let (net, root, t, adv) = sat_fixture(29);
    let (out, _) = with_test_backend(Arc::new(Panics), WaveHooks::default(), || {
        MarginRowBab::run(
            &net,
            &root,
            t,
            &adv,
            BabConfig {
                max_expansions: 200,
                ..wave_cfg(1)
            },
        )
    });
    assert!(!matches!(out, MarginRowOutcome::Unsat(..)));
    // On the robust fixture a panicking search degrades to certify-everything
    // (lane-chosen splits): any verdict it reaches is the certifier's, and
    // every leaf it records is a strict certified close.
    let fx = tree_fixture();
    let (out, leaves) = with_test_backend(Arc::new(Panics), WaveHooks::default(), || {
        MarginRowBab::run(&fx.net, &fx.root, fx.t, &fx.adv, wave_cfg(2))
    });
    if let MarginRowOutcome::Unsat(stats, _) = &out {
        assert_eq!(stats.ledger_ok, Some(true));
    }
    assert!(leaves.iter().all(|l| l.value > 0.0 && l.rung != "mutant"));
}

// ---------------------------------------------------------------------------
//  Mutation pins
// ---------------------------------------------------------------------------

#[test]
fn wave_claims_are_not_closes() {
    let (net, root, t, adv) = sat_fixture(29);
    // Every domain claims with a positive search bound.
    let backend = || Arc::new(ClaimAll::default()) as Arc<dyn SearchBackend>;
    let cfg = || BabConfig {
        max_expansions: 300,
        ..wave_cfg(2)
    };
    // MUTANT: a positive search bound closes a claim uncertified. Its leaves
    // pass the path-based cover gate — the only barrier left is the G2
    // guard's own sampled claim check (or, off its samples, nothing).
    let mutant_hooks = WaveHooks {
        claims_are_closes: true,
        ..WaveHooks::default()
    };
    let (mutant, mutant_leaves) = with_test_backend(backend(), mutant_hooks, || {
        MarginRowBab::run(&net, &root, t, &adv, cfg())
    });
    assert!(
        mutant_leaves.iter().any(|l| l.rung == "mutant"),
        "the mutant never closed a claim: the pin is VOID"
    );
    let mutant_passed_the_cover = match &mutant {
        MarginRowOutcome::Unsat(..) => true,
        MarginRowOutcome::Unknown { reason, .. } => reason.contains("guard="),
    };
    assert!(
        mutant_passed_the_cover,
        "the claim-as-close mutant must get past certification and the cover gate on a SAT \
         instance (it did not, so the pin cannot see the mutation): {:?}",
        match &mutant {
            MarginRowOutcome::Unknown { reason, .. } => reason.clone(),
            MarginRowOutcome::Unsat(..) => "unsat".into(),
        }
    );
    // REAL: the same backend, the same budget: certification refutes every
    // claim on the violated region, and the tree never reaches the gate.
    let (real, real_leaves) = with_test_backend(backend(), WaveHooks::default(), || {
        MarginRowBab::run(&net, &root, t, &adv, cfg())
    });
    assert!(!matches!(real, MarginRowOutcome::Unsat(..)));
    assert!(real_leaves.iter().all(|l| l.rung != "mutant"));
    if let MarginRowOutcome::Unknown { reason, .. } = &real {
        assert!(
            !reason.contains("guard="),
            "the real path reached the gate: {reason}"
        );
    }
}

/// `x -> relu -> (relu(x) + x) -> flatten -> y`: layer 0's pre-activation is
/// the input itself, so its clip rows are EXACT (identity), and the residual
/// keeps `y` dependent on `x_j` on BOTH sides of relu 0's split — which is
/// what lets a side flip be seen on the Off side too.
fn clip_fixture(seed: u64) -> (TwinNet, RootGates) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut w =
        |n: usize, s: f64| -> Vec<f64> { (0..n).map(|_| rng.random_range(-s..s)).collect() };
    let (n_in, n_y, n_out) = (4usize, 3usize, 3usize);
    let spec = TwinSpec {
        n_in,
        ops: vec![
            TwinOpSpec::Relu { input: 0 },      // t1 = relu(x) (trunk relu 0)
            TwinOpSpec::Add { lhs: 1, rhs: 0 }, // t2 = relu(x) + x
            TwinOpSpec::Flatten { input: 2 },   // t3
            TwinOpSpec::Gemm {
                input: 3,
                weight: w(n_y * n_in, 1.0),
                bias: w(n_y, 0.3),
                bias_err: Vec::new(),
                weight_rel_err: 0.0,
                shape: (n_y, n_in),
            }, // t4 (y)
            TwinOpSpec::Relu { input: 4 },      // t5 (head)
            TwinOpSpec::Gemm {
                input: 5,
                weight: w(n_out * n_y, 1.0),
                bias: w(n_out, 0.3),
                bias_err: Vec::new(),
                weight_rel_err: 0.0,
                shape: (n_out, n_y),
            }, // t6
        ],
    };
    let net = TwinNet::compile(&spec).expect("compile");
    let lo = vec![-1.0; n_in];
    let hi = vec![1.0; n_in];
    let mut root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
    let layer = &mut root.layers[0];
    let k = layer.unst.len();
    let mut m = Array2::<f64>::zeros((k, n_in));
    for (r, &i) in layer.unst.iter().enumerate() {
        m[[r, i]] = 1.0;
    }
    layer.clip_rows = Some(ClipRows {
        m,
        d: Array2::<f64>::zeros((k, n_in)),
        sl_lo: vec![0.0; k],
        sl_up: vec![0.0; k],
        bm: vec![0.0; k],
        bd: vec![0.0; k],
    });
    (net, root)
}

#[test]
fn wave_clip_side_is_pinned_on_both_sides() {
    let mut seen_real_move = false;
    let mut flip_caught = [false; 2];
    for seed in 0..8u64 {
        let (net, root) = clip_fixture(seed);
        assert!(
            crate::margin_row::root::clip_rows_frame_holds(&root, 0.5),
            "the exact identity rows must pass the frame check"
        );
        let unst = root.layers[0].unst.clone();
        assert!(!unst.is_empty(), "x in [-1, 1] leaves relu 0 unstable");
        let mut rng = StdRng::seed_from_u64(seed + 100);
        let x = sample_box(&mut rng, &root, 1500);
        let (y, _) = net
            .forward_points(&x, &std::collections::BTreeMap::new())
            .expect("forward");
        for (pos, &idx) in unst.iter().enumerate() {
            for (s, dir) in [(0usize, 1i8), (1, -1)] {
                let trunk = [(0usize, pos, dir)];
                let (base, real, moved) =
                    clip_ybox_for_test(&net, &root, &trunk, false).expect("clip");
                let (_, flipped, _) = clip_ybox_for_test(&net, &root, &trunk, true).expect("clip");
                seen_real_move |= moved > 0;
                for i in 0..net.n_y {
                    assert!(
                        real.ly[i] >= base.ly[i] && real.uy[i] <= base.uy[i],
                        "shrink-only"
                    );
                }
                for b in 0..x.ncols() {
                    let xv = x[[idx, b]];
                    let inside = if dir > 0 { xv >= 0.0 } else { xv <= 0.0 };
                    if !inside {
                        continue;
                    }
                    for i in 0..net.n_y {
                        let v = y[[i, b]];
                        let tol = 1e-12 * (1.0 + v.abs());
                        assert!(
                            real.ly[i] <= v + tol && v <= real.uy[i] + tol,
                            "REAL CLIP RUNG LEFT THE REGION: seed {seed} pos {pos} dir {dir} \
                             y_{i} = {v} outside [{}, {}]",
                            real.ly[i],
                            real.uy[i]
                        );
                        if v < flipped.ly[i] - tol || v > flipped.uy[i] + tol {
                            flip_caught[s] = true;
                        }
                    }
                }
            }
        }
    }
    assert!(seen_real_move, "the clip rung never tightened: VOID");
    assert!(
        flip_caught[0],
        "a flipped ON-side half-space was never caught by sampled truth: VOID"
    );
    assert!(
        flip_caught[1],
        "a flipped OFF-side half-space was never caught by sampled truth: VOID"
    );
}

#[test]
fn wave_cover_gate_refuses_a_dropped_leaf() {
    let fx = tree_fixture();
    crate::margin_row::bab::inject_cover_fault_drop_next_leaf();
    let (out, leaves) =
        with_test_backend(Arc::new(ClaimAll::default()), WaveHooks::default(), || {
            MarginRowBab::run(&fx.net, &fx.root, fx.t, &fx.adv, wave_cfg(2))
        });
    assert!(!leaves.is_empty());
    match out {
        MarginRowOutcome::Unsat(..) => {
            panic!("MUTATION SURVIVED: a wave tree missing a leaf was authorized")
        }
        MarginRowOutcome::Unknown { reason, stats } => {
            assert!(reason.contains("cover gate refused"), "{reason}");
            assert_eq!(stats.and_then(|s| s.ledger_ok), Some(false));
        }
    }
}

#[test]
fn wave_certification_errors_never_close() {
    let fx = tree_fixture();
    let hooks = WaveHooks {
        certify_errors: true,
        ..WaveHooks::default()
    };
    let (out, leaves) = with_test_backend(Arc::new(ClaimAll::default()), hooks, || {
        MarginRowBab::run(&fx.net, &fx.root, fx.t, &fx.adv, wave_cfg(2))
    });
    assert!(leaves.is_empty(), "an erroring certifier closed a leaf");
    match out {
        MarginRowOutcome::Unsat(..) => panic!("an erroring certifier produced a verdict"),
        MarginRowOutcome::Unknown { reason, .. } => {
            assert!(reason.contains("fail-closed"), "{reason}");
        }
    }
}

// ---------------------------------------------------------------------------
//  Protocol
// ---------------------------------------------------------------------------

#[test]
fn wave_is_thread_count_independent() {
    let fx = tree_fixture();
    let run = |threads: usize| {
        with_test_backend(Arc::new(ClaimAll::default()), WaveHooks::default(), || {
            MarginRowBab::run(&fx.net, &fx.root, fx.t, &fx.adv, wave_cfg(threads))
        })
    };
    let (a, la) = run(1);
    let (b, lb) = run(4);
    assert_eq!(stats_key(&a), stats_key(&b));
    assert_eq!(leaf_key(&la), leaf_key(&lb));
    // And the SAT side: the same bounded trajectory at both widths.
    let (net, root, t, adv) = sat_fixture(31);
    let sat = |threads: usize| {
        with_test_backend(
            Arc::new(Garbage::new(5, false)),
            WaveHooks::default(),
            || {
                MarginRowBab::run(
                    &net,
                    &root,
                    t,
                    &adv,
                    BabConfig {
                        max_expansions: 120,
                        ..wave_cfg(threads)
                    },
                )
            },
        )
    };
    let (sa, sla) = sat(1);
    let (sb, slb) = sat(4);
    assert_eq!(stats_key(&sa), stats_key(&sb));
    assert_eq!(leaf_key(&sla), leaf_key(&slb));
}

// ---------------------------------------------------------------------------
//  `wave_cuda`: the device first, then the CPU search, then the legacy tree
// ---------------------------------------------------------------------------

fn cuda_cfg(cert_threads: usize) -> BabConfig {
    let base = wave_cfg(cert_threads);
    BabConfig {
        wave: WavePolicy {
            mode: TreeMode::WaveCuda,
            ..base.wave
        },
        ..base
    }
}

#[test]
fn wave_cuda_uses_the_device_backend_when_its_upload_is_admitted() {
    let fx = tree_fixture();
    let cpu = Arc::new(Tripwire::default());
    let device = Arc::new(Renamed {
        name: "device_claim_all",
        inner: ClaimAll::default(),
    });
    let (dev_out, dev_leaves) = with_test_backends(
        Arc::clone(&cpu) as Arc<dyn SearchBackend>,
        device as Arc<dyn SearchBackend>,
        WaveHooks::default(),
        || MarginRowBab::run(&fx.net, &fx.root, fx.t, &fx.adv, cuda_cfg(2)),
    );
    let line = take_last_wave_summary().expect("an armed tree always prints its receipt");
    assert!(
        line.contains("[bbe] phase=wave armed=1 backend=device_claim_all")
            && line.contains("mode=wave_cuda device_refused=none")
            && !line.contains("fallback=legacy"),
        "{line}"
    );
    assert_eq!(
        cpu.calls.load(Ordering::Relaxed),
        0,
        "the CPU backend was consulted although the device upload was admitted"
    );
    // The mode selects the backend only: the same search gives the same tree.
    let (cpu_out, cpu_leaves) =
        with_test_backend(Arc::new(ClaimAll::default()), WaveHooks::default(), || {
            MarginRowBab::run(&fx.net, &fx.root, fx.t, &fx.adv, wave_cfg(2))
        });
    assert_eq!(stats_key(&dev_out), stats_key(&cpu_out));
    assert_eq!(leaf_key(&dev_leaves), leaf_key(&cpu_leaves));
    assert!(
        !dev_leaves.is_empty(),
        "VOID: the device double closed no leaf"
    );
}

#[test]
fn wave_cuda_refused_device_falls_through_to_the_cpu_search() {
    let fx = tree_fixture();
    let (want, want_leaves) =
        with_test_backend(Arc::new(ClaimAll::default()), WaveHooks::default(), || {
            MarginRowBab::run(&fx.net, &fx.root, fx.t, &fx.adv, wave_cfg(2))
        });
    let _ = take_last_wave_summary();
    // A device whose upload refuses (unavailable, failed self-test, lease).
    let (out, leaves) = with_test_backends(
        Arc::new(ClaimAll::default()),
        Arc::new(Refuse),
        WaveHooks::default(),
        || MarginRowBab::run(&fx.net, &fx.root, fx.t, &fx.adv, cuda_cfg(2)),
    );
    let line = take_last_wave_summary().expect("receipt");
    assert!(
        line.contains("backend=claim_all")
            && line.contains("mode=wave_cuda device_refused=unsupported:_upload_wave")
            && !line.contains("fallback=legacy"),
        "{line}"
    );
    assert_eq!(stats_key(&out), stats_key(&want));
    assert_eq!(leaf_key(&leaves), leaf_key(&want_leaves));
    // No device registered at all.
    let (out, _) = with_test_backend(Arc::new(ClaimAll::default()), WaveHooks::default(), || {
        MarginRowBab::run(&fx.net, &fx.root, fx.t, &fx.adv, cuda_cfg(2))
    });
    let line = take_last_wave_summary().expect("receipt");
    assert!(
        line.contains("backend=claim_all") && line.contains("device_refused=no_backend"),
        "{line}"
    );
    assert_eq!(stats_key(&out), stats_key(&want));
}

#[test]
fn wave_mode_never_consults_the_device_backend() {
    let fx = tree_fixture();
    let device = Arc::new(Tripwire::default());
    let _ = with_test_backends(
        Arc::new(ClaimAll::default()),
        Arc::clone(&device) as Arc<dyn SearchBackend>,
        WaveHooks::default(),
        || MarginRowBab::run(&fx.net, &fx.root, fx.t, &fx.adv, wave_cfg(2)),
    );
    let line = take_last_wave_summary().expect("receipt");
    assert!(
        line.contains("backend=claim_all") && line.contains("mode=wave device_refused=none"),
        "{line}"
    );
    assert_eq!(device.calls.load(Ordering::Relaxed), 0);
}

#[test]
fn wave_cuda_with_every_backend_refusing_runs_the_legacy_tree_bit_identical() {
    let fx = tree_fixture();
    let plain = MarginRowBab::run(&fx.net, &fx.root, fx.t, &fx.adv, legacy_cfg());
    let armed = BabConfig {
        wave: WavePolicy {
            mode: TreeMode::WaveCuda,
            batch: 16,
            cert_threads: 2,
        },
        ..legacy_cfg()
    };
    let (refused, leaves) = with_test_backends(
        Arc::new(Refuse),
        Arc::new(Refuse),
        WaveHooks::default(),
        || MarginRowBab::run(&fx.net, &fx.root, fx.t, &fx.adv, armed),
    );
    let line = take_last_wave_summary().expect("receipt");
    // The refusal reason is ONE receipt token ("unsupported: upload_wave"
    // would end the value at its space).
    assert!(
        line.contains(
            "backend=refuse refused=unsupported:_upload_wave fallback=legacy mode=wave_cuda"
        ),
        "{line}"
    );
    assert_eq!(stats_key(&plain), stats_key(&refused));
    assert!(leaves.is_empty());
}

#[test]
fn wave_cuda_device_fault_mid_tree_switches_to_the_cpu_search() {
    let fx = tree_fixture();
    let (want, _) = with_test_backend(Arc::new(ClaimAll::default()), WaveHooks::default(), || {
        MarginRowBab::run(&fx.net, &fx.root, fx.t, &fx.adv, wave_cfg(2))
    });
    let _ = take_last_wave_summary();
    let cpu = Arc::new(ClaimAll::default());
    let device = Arc::new(FaultAfter {
        name: "device_faults",
        ok: 1,
        calls: AtomicUsize::new(0),
        inner: ClaimAll::default(),
    });
    let (out, leaves) = with_test_backends(
        Arc::clone(&cpu) as Arc<dyn SearchBackend>,
        Arc::clone(&device) as Arc<dyn SearchBackend>,
        WaveHooks::default(),
        || MarginRowBab::run(&fx.net, &fx.root, fx.t, &fx.adv, cuda_cfg(2)),
    );
    let line = take_last_wave_summary().expect("receipt");
    assert!(
        device.calls.load(Ordering::Relaxed) >= 2,
        "VOID: the tree ended before the injected fault ({line})"
    );
    // The receipt names the EFFECTIVE backends (the device, then the CPU
    // search: never `backend=device_faults ` alone, which would count a CPU
    // tree as device-engaged), the wave of the switch and the device's
    // reason.
    assert!(
        line.contains("backend=device_faults->claim_all ")
            && !line.contains("backend=device_faults ")
            && line.contains("device_refused=none device_switch=wave")
            && line.contains(":switched_to_claim_all")
            && line.contains("device_fault:_injected")
            && line.contains("switch_reason=backward_batch:")
            && !line.contains("switched_at_wave=none")
            && !line.contains("fallback=legacy"),
        "{line}"
    );
    let at: usize = line
        .split_whitespace()
        .find_map(|kv| kv.strip_prefix("switched_at_wave="))
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| panic!("switched_at_wave= is not a wave number: {line}"));
    assert!(at >= 2, "the device answered wave 1: {line}");
    assert!(
        cpu.searches.load(Ordering::Relaxed) > 0,
        "the CPU search never took over after the device fault"
    );
    // The faulted call RE-RUNS on the CPU search (which the device double
    // reproduces), so the switch changes speed, never the trajectory: the
    // whole tree is the CPU search's.
    let (want_again, want_leaves) =
        with_test_backend(Arc::new(ClaimAll::default()), WaveHooks::default(), || {
            MarginRowBab::run(&fx.net, &fx.root, fx.t, &fx.adv, wave_cfg(2))
        });
    let _ = take_last_wave_summary();
    assert_eq!(stats_key(&out), stats_key(&want_again));
    assert_eq!(leaf_key(&leaves), leaf_key(&want_leaves));
    // A search switch never changes what certification decides.
    if matches!(want, MarginRowOutcome::Unsat(..)) {
        let key = stats_key(&out);
        match out {
            MarginRowOutcome::Unsat(stats, token) => {
                assert!(token.publishable());
                assert_eq!(stats.ledger_ok, Some(true));
            }
            MarginRowOutcome::Unknown { .. } => {
                panic!("the switched tree lost a certified proof: {key:?}")
            }
        }
    }
}

/// Findings 2 and 4 of the M6 review: a HOST-side refusal (the device's bounds
/// check refusing a slab, a malformed scoring) is that call's refusal alone —
/// it never switches the tree to the CPU search (the CPU backend is a
/// tripwire here), the device keeps searching, `backend=` still names the
/// device ALONE, and the refusal is never swallowed: `first_refusal=` names
/// the call and its reason, `search_refused=` counts it.
#[test]
fn wave_cuda_host_refusals_keep_the_device_and_reach_the_receipt() {
    let fx = tree_fixture();
    let run = |device: Arc<HostRefuses<ClaimAll>>| {
        let cpu = Arc::new(Tripwire::default());
        let _ = take_last_wave_summary();
        let (out, leaves) = with_test_backends(
            Arc::clone(&cpu) as Arc<dyn SearchBackend>,
            Arc::clone(&device) as Arc<dyn SearchBackend>,
            WaveHooks::default(),
            || MarginRowBab::run(&fx.net, &fx.root, fx.t, &fx.adv, cuda_cfg(2)),
        );
        let line = take_last_wave_summary().expect("receipt");
        assert_eq!(
            cpu.calls.load(Ordering::Relaxed),
            0,
            "a host-side refusal consulted the CPU backend (a switch): {line}"
        );
        assert!(
            line.contains("[bbe] phase=wave armed=1 backend=device_host ")
                && line.contains(
                    "device_refused=none device_switch=none switched_at_wave=none \
                     switch_reason=none"
                )
                && !line.contains("fallback=legacy")
                && !line.contains("search_refused=0 "),
            "{line}"
        );
        assert!(leaves.iter().all(|l| l.value > 0.0 && l.rung != "mutant"));
        if let MarginRowOutcome::Unsat(stats, token) = &out {
            assert!(token.publishable());
            assert_eq!(stats.ledger_ok, Some(true));
        }
        line
    };
    // A refused wave search (the root wave: its fresh domain is claimed
    // unsearched; the device keeps the tree and searches the next wave).
    let device = Arc::new(HostRefuses::new(ClaimAll::default(), &[0], &[]));
    let line = run(Arc::clone(&device));
    assert!(
        device.searches.load(Ordering::Relaxed) >= 2,
        "VOID: the tree ended before the device searched again after its refusal: {line}"
    );
    assert!(
        line.contains(
            "first_refusal=backward_batch:invalid:_cuda_wave_search:_host-side:_injected "
        ),
        "{line}"
    );
    // A refused scoring (the splits take the lane's own candidates).
    let device = Arc::new(HostRefuses::new(ClaimAll::default(), &[], &[0]));
    let searches_before = device.searches.load(Ordering::Relaxed);
    let line = run(Arc::clone(&device));
    assert!(
        device.branches.load(Ordering::Relaxed) >= 1
            && device.searches.load(Ordering::Relaxed) >= searches_before + 2,
        "VOID: the tree never scored, or the device never searched after its refused \
         scoring: {line}"
    );
    assert!(
        line.contains(
            "first_refusal=branch_scores:invalid:_cuda_wave_search:_host-side_scoring:_injected "
        ),
        "{line}"
    );
    // A malformed scoring is a refusal too, never silently dropped.
    let mut malformed = HostRefuses::new(ClaimAll::default(), &[], &[]);
    malformed.malformed_branch = true;
    let device = Arc::new(malformed);
    let line = run(Arc::clone(&device));
    assert!(
        device.branches.load(Ordering::Relaxed) >= 1,
        "VOID: the tree never scored: {line}"
    );
    assert!(
        line.contains("first_refusal=branch_scores:malformed_branch_output "),
        "{line}"
    );
}

/// Fault injection on the SAT moat, through both new `wave_cuda` paths: a
/// GARBAGE device that faults after its first wave search, handing the rest
/// of the tree to a garbage CPU search (`switched_to_`), or to a CPU search
/// whose re-upload refuses (`cpu_refused_`: the device's buffers are already
/// released, so every later wave is refused and its fresh domains are claimed
/// unsearched). Neither path ever closes a witnessed SAT class, every leaf is
/// a certified close, and both paths are reached (else VOID).
#[test]
fn wave_cuda_device_faults_never_close_a_witnessed_sat_class() {
    let (mut switched, mut degenerate) = (0usize, 0usize);
    for seed in [29u64, 31, 37] {
        let (net, root, t, adv) = sat_fixture(seed);
        for cpu_refuses in [false, true] {
            let device = Arc::new(FaultAfter {
                name: "device_garbage",
                ok: 1,
                calls: AtomicUsize::new(0),
                inner: Garbage::new(seed * 7 + 1, false),
            });
            let cpu: Arc<dyn SearchBackend> = if cpu_refuses {
                Arc::new(Refuse)
            } else {
                Arc::new(Garbage::new(seed * 7 + 2, true))
            };
            let _ = take_last_wave_summary();
            let (out, leaves) = with_test_backends(
                cpu,
                Arc::clone(&device) as Arc<dyn SearchBackend>,
                WaveHooks::default(),
                || {
                    MarginRowBab::run(
                        &net,
                        &root,
                        t,
                        &adv,
                        BabConfig {
                            max_expansions: 300,
                            ..cuda_cfg(2)
                        },
                    )
                },
            );
            assert!(
                !matches!(out, MarginRowOutcome::Unsat(..)),
                "MOAT VIOLATION: a faulting garbage device (seed {seed}, cpu_refuses \
                 {cpu_refuses}) closed a witnessed SAT instance"
            );
            assert!(leaves.iter().all(|l| l.value > 0.0 && l.rung != "mutant"));
            let line = take_last_wave_summary().unwrap_or_default();
            let faulted = device.calls.load(Ordering::Relaxed) >= 2
                && line.contains("wave-beta-bab/v1")
                && !line.contains("fallback=legacy");
            switched += usize::from(faulted && line.contains(":switched_to_garbage"));
            degenerate += usize::from(faulted && line.contains(":cpu_refused_"));
        }
    }
    assert!(
        switched > 0 && degenerate > 0,
        "VOID: the device fault reached the wave engine's switch {switched} time(s) and \
         its refused re-upload {degenerate} time(s)"
    );
}

// ---------------------------------------------------------------------------
//  B2.3: the wave's leaves in the proof bundle
// ---------------------------------------------------------------------------

/// A recording sink the test keeps a handle to (the thread sink is boxed).
#[derive(Clone, Default)]
struct SharedRec(Arc<std::sync::Mutex<crate::certify::RecordingSink>>);

impl crate::certify::ProofSink for SharedRec {
    fn leaf(&mut self, leaf: &crate::certify::LeafRecord) {
        self.0.lock().expect("lock").leaf(leaf);
    }

    fn node(&mut self, node: &crate::certify::NodeRecord) {
        self.0.lock().expect("lock").node(node);
    }

    fn alpha(&mut self, alpha: &crate::certify::AlphaRecord) {
        self.0.lock().expect("lock").alpha(alpha);
    }

    fn sites(&mut self, sites: &crate::certify::SitesRecord) {
        self.0.lock().expect("lock").sites(sites);
    }

    fn fd_state(&mut self, state: &crate::certify::FdStateRecord) {
        self.0.lock().expect("lock").fd_state(state);
    }

    fn finish(&mut self, verdict: &crate::certify::VerdictRecord) {
        self.0.lock().expect("lock").finish(verdict);
    }
}

/// B2.3 (ENGINE_PROGRAM §1.7 B1): under hint capture every certified leaf of
/// the wave tree streams with its frame as a checkable node — its literals in
/// split order (the universe sites, translated by the frame's own site map),
/// its β, its close and the ONE class its row discharges — and the class's
/// tree cover no longer carries the opaque `wave` claim. Capture is
/// write-only: the tree, its leaves and the verdict are bit-identical with
/// capture off, which streams nothing.
#[test]
fn wave_leaves_stream_as_checkable_nodes_and_the_tree_is_bit_identical() {
    use crate::certify::{force_hint_capture_for_test, with_thread_sink, NodeClose};

    let fx = tree_fixture();
    let run = |capture: bool| {
        let sink = SharedRec::default();
        force_hint_capture_for_test(Some(capture));
        let ((out, _), leaves) =
            with_test_backend(Arc::new(ClaimAll::default()), WaveHooks::default(), || {
                with_thread_sink(Box::new(sink.clone()), || {
                    MarginRowBab::run(&fx.net, &fx.root, fx.t, &fx.adv, wave_cfg(2))
                })
            });
        force_hint_capture_for_test(None);
        let engaged = wave_engaged();
        let rec = sink.0.lock().expect("lock").clone();
        (out, leaves, rec, engaged)
    };
    let (off, off_leaves, off_rec, off_engaged) = run(false);
    let (on, on_leaves, rec, on_engaged) = run(true);
    assert!(
        off_engaged && on_engaged,
        "the wave engine did not run: VOID"
    );
    assert_eq!(stats_key(&off), stats_key(&on), "capture moved the tree");
    assert_eq!(leaf_key(&off_leaves), leaf_key(&on_leaves));
    assert!(off_rec.nodes.is_empty(), "capture off streams nothing");
    assert!(
        matches!(on, MarginRowOutcome::Unsat(..)),
        "the fixture proves"
    );
    assert!(
        !rec.nodes
            .iter()
            .any(|n| n.close == NodeClose::Opaque && n.src == "wave"),
        "a fully streamed wave class still claimed opaque"
    );
    let tree: Vec<&crate::certify::NodeRecord> =
        rec.nodes.iter().filter(|n| n.src == "tree").collect();
    assert_eq!(tree.len(), on_leaves.len(), "one node per certified leaf");
    for (n, l) in tree.iter().zip(&on_leaves) {
        assert_eq!(n.lits.len(), l.trunk.len() + l.heads.len());
        assert_eq!(n.beta.len(), n.lits.len());
        assert_eq!(n.margin.to_bits(), l.value.to_bits());
        if l.empty {
            assert_eq!(n.close, NodeClose::Empty);
            assert_eq!(n.dis, None);
        } else {
            assert_eq!(n.close, NodeClose::Bound);
            assert_eq!(n.dis.as_deref(), Some(&[l.class][..]));
        }
        let ops: Vec<u32> = fx.root.layers.iter().map(|g| g.op as u32).collect();
        let head_op = (fx.net.i_gemm1 + 1) as u32;
        assert!(n
            .lits
            .iter()
            .all(|lit| ops.contains(&lit.site.op) || lit.site.op == head_op));
    }
}
