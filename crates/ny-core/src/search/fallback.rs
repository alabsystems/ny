// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! [`FallbackBackend`] — M6's device-then-CPU WAVE search as ONE
//! [`SearchBackend`]: a PRIMARY backend (ny-cuda's device search) with a
//! SECONDARY behind it (ny-bbe's CPU f32 search, which the device reproduces bit
//! for bit). The lane (`margin_row.tree_mode: wave_cuda`) sees one backend and
//! one handle per tree; every fall-through and the post-fault switch happen
//! HERE, inside `upload_wave` / `backward_batch` / `branch_scores`, and the
//! lane reads which backend(s) actually searched from [`FallbackBackend::route`].
//!
//! # Routing
//!
//! * **Upload fall-through.** `upload_wave` asks the primary first. Any refusal
//!   of the primary's upload (absent, unavailable, failed self-test, lease) or
//!   a panic falls through to the secondary, which then searches the whole tree
//!   ([`WaveRoute::primary_refused`]). A refusal of both is the secondary's
//!   refusal, unchanged (the primary's is kept for the receipt,
//!   [`FallbackBackend::primary_upload_refusal`]).
//! * **Post-fault switch, once.** While the primary holds a tree, ONLY a device
//!   fault — [`Refusal::Fault`], or a panic escaping the primary — switches the
//!   tree to the secondary. Every other refusal (a host-side validation
//!   refusal, a malformed batch, the deadline) is that call's refusal alone,
//!   returned to the lane unchanged, and the primary keeps the tree. On a
//!   fault the primary's upload is RELEASED before the secondary's is built
//!   (never two search copies at once), the secondary uploads the retained
//!   problem, and the call that faulted is RE-RUN on the secondary, so the
//!   tree loses no search: the secondary reproduces the primary bit for bit,
//!   so a switch changes speed, never the trajectory. When the secondary's
//!   re-upload refuses too (or there is no secondary), every later call on the
//!   tree refuses with the fault (the lane then claims its fresh domains
//!   unsearched and certification decides every one of them).
//!
//! # Memory (the overlap, bounded)
//!
//! The re-upload needs the problem, so the tree RETAINS one host copy of it
//! ([`WaveProblem::approx_bytes`]) while the primary searches, when a secondary
//! exists. That copy is charged to the lease the lane granted: the primary is
//! admitted against `lease - retained`, and the secondary's re-upload against
//! `lease - retained` too (the copy is alive while it uploads and dropped right
//! after). So the composite holds at most `lease` bytes of search state plus
//! the retained copy, which is itself inside `lease`: never the primary's and
//! the secondary's copies at once. An upload-time fall-through retains nothing
//! and gives the secondary the whole lease (exactly the `wave` mode's upload).
//!
//! # Trust
//!
//! Routing only. Every value this backend returns is the primary's or the
//! secondary's own untrusted hint, which the lane ingests through its gate
//! exactly as it would the backend's; the certify queue and the cover gate
//! never learn which backend searched. Nothing here can make a value a bound.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use super::{
    BackendCaps, BranchOut, DeviceLease, Refusal, SearchBackend, Untrusted, WaveBatch, WaveHandle,
    WaveOut, WaveProblem, WaveSearchCfg, WaveStats,
};

/// The receipt name of "no backend" (a tree whose switch found no secondary).
pub const NO_BACKEND: &str = "none";

/// Which backend(s) searched one wave tree: the lane's receipt source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WaveRoute {
    /// The backend that searched the tree's first wave (the primary, or the
    /// secondary after an upload fall-through).
    pub first: &'static str,
    /// Why the primary's upload refused, when the tree ran on the secondary
    /// from its first wave (`None` when the primary took the tree, or for a
    /// single backend).
    pub primary_refused: Option<String>,
    /// The mid-tree switch, if a device fault caused one.
    pub switch: Option<WaveSwitch>,
}

/// A mid-tree switch (see the module docs).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WaveSwitch {
    /// The call that faulted (`backward_batch` or `branch_scores`).
    pub method: &'static str,
    /// The primary's fault (its refusal text, or the panic).
    pub reason: String,
    /// The secondary's name when its re-upload was admitted (it searched the
    /// rest of the tree, the faulted call included); else why it was not
    /// (every later call then refused).
    pub to: Result<&'static str, String>,
}

impl WaveRoute {
    /// A tree searched by `name` alone.
    #[must_use]
    pub fn single(name: &'static str) -> Self {
        Self {
            first: name,
            primary_refused: None,
            switch: None,
        }
    }

    /// The receipt's `backend=` value: the backend that searched, or
    /// `first->secondary` after a switch (`first->none` when the secondary
    /// could not take the tree). A tree searched on the device ALONE reads
    /// exactly `cuda`; a value that merely starts with it (`cuda->cpu_f32`)
    /// did part of its search elsewhere.
    #[must_use]
    pub fn backends(&self) -> String {
        match &self.switch {
            None => self.first.to_string(),
            Some(s) => format!(
                "{}->{}",
                self.first,
                s.to.as_ref().copied().unwrap_or(NO_BACKEND)
            ),
        }
    }
}

/// Where one tree's search runs now.
enum On {
    Primary(WaveHandle),
    Secondary(WaveHandle),
    /// Neither: the primary faulted and the secondary could not take over.
    Neither(String),
}

/// One tree's routing state (the composite's `WaveHandle` payload).
struct TreeState {
    on: On,
    /// The problem, retained while the primary searches and a secondary exists
    /// (the re-upload's source); dropped at the switch.
    retained: Option<WaveProblem>,
    /// The lease the lane granted this tree.
    lease: DeviceLease,
    route: WaveRoute,
}

struct Tree {
    stride: usize,
    state: Mutex<TreeState>,
}

/// A primary search backend with a secondary behind it (see the module docs).
pub struct FallbackBackend {
    primary: Arc<dyn SearchBackend>,
    secondary: Option<Arc<dyn SearchBackend>>,
    /// The first primary fault this composite saw (its `name()` / `caps()`
    /// then report the secondary). The primary latches itself anyway.
    faulted: OnceLock<String>,
    /// Why the primary's most recent upload refused (receipt only).
    primary_upload: Mutex<Option<String>>,
}

impl std::fmt::Debug for FallbackBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FallbackBackend")
            .field("primary", &self.primary.name())
            .field("secondary", &self.secondary.as_ref().map(|s| s.name()))
            .field("faulted", &self.faulted.get())
            .finish_non_exhaustive()
    }
}

/// A panic's message, when it has one.
fn panic_text(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|s| (*s).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "a non-string panic".to_string())
}

/// Run a backend call; a panic becomes a [`Refusal::Fault`] (a panic escaping
/// a backend leaves its device state unknown: treated as a fault).
fn guarded<T>(what: &str, f: impl FnOnce() -> Result<T, Refusal>) -> Result<T, Refusal> {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|payload| {
        Err(Refusal::Fault(format!(
            "{what} panicked: {}",
            panic_text(payload.as_ref())
        )))
    })
}

impl FallbackBackend {
    /// `primary` first, `secondary` (if any) behind it.
    #[must_use]
    pub fn new(primary: Arc<dyn SearchBackend>, secondary: Option<Arc<dyn SearchBackend>>) -> Self {
        Self {
            primary,
            secondary,
            faulted: OnceLock::new(),
            primary_upload: Mutex::new(None),
        }
    }

    /// The primary's and the secondary's names.
    #[must_use]
    pub fn names(&self) -> (&'static str, Option<&'static str>) {
        (
            self.primary.name(),
            self.secondary.as_ref().map(|s| s.name()),
        )
    }

    /// Why the primary's most recent `upload_wave` refused (`None` when it was
    /// admitted or never asked). Receipt only.
    #[must_use]
    pub fn primary_upload_refusal(&self) -> Option<String> {
        self.primary_upload
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// The first primary fault this composite saw, if any.
    #[must_use]
    pub fn primary_fault(&self) -> Option<&str> {
        self.faulted.get().map(String::as_str)
    }

    /// Which backend(s) searched the tree of `handle`, if this composite made
    /// it (`None` for any other backend's handle).
    #[must_use]
    pub fn route(handle: &WaveHandle) -> Option<WaveRoute> {
        let tree = handle.payload::<Tree>()?;
        let st = tree.state.lock().unwrap_or_else(PoisonError::into_inner);
        Some(st.route.clone())
    }

    fn tree<'h>(&self, handle: &'h WaveHandle) -> Result<&'h Tree, Refusal> {
        handle
            .payload::<Tree>()
            .ok_or_else(|| Refusal::Invalid(format!("foreign wave handle ({})", handle.backend())))
    }

    /// The post-fault switch (see the module docs): release the primary's
    /// upload, then re-upload the retained problem to the secondary.
    fn switch(&self, stride: usize, st: &mut TreeState, method: &'static str, reason: String) {
        let _ = self.faulted.set(reason.clone());
        // Release the primary's device state FIRST (the drop of its handle
        // frees its buffers): the two search copies never coexist.
        st.on = On::Neither(reason.clone());
        let retained = st.retained.take();
        let to = match (&self.secondary, retained) {
            (None, _) => Err("no secondary backend".to_string()),
            (Some(_), None) => Err("the problem was not retained".to_string()),
            (Some(secondary), Some(problem)) => {
                let lease = DeviceLease {
                    bytes: st.lease.bytes.saturating_sub(problem.approx_bytes()),
                };
                let upload = guarded("the secondary's upload", || {
                    secondary.upload_wave(&problem, &lease)
                });
                // The retained copy is released as soon as the re-upload is
                // decided.
                drop(problem);
                match upload {
                    Ok(h) if h.stride() == stride => {
                        st.on = On::Secondary(h);
                        Ok(secondary.name())
                    }
                    Ok(_) => Err("stride mismatch".to_string()),
                    Err(r) => Err(r.to_string()),
                }
            }
        };
        st.route.switch = Some(WaveSwitch { method, reason, to });
    }

    /// Run one search call on `handle`'s tree: on the backend that holds it,
    /// switching once (and re-running the call) on a primary fault. `run`'s
    /// flag asks for a CLEAN output first (the re-run after a fault: the
    /// primary may have written part of it).
    fn call<T>(
        &self,
        handle: &WaveHandle,
        method: &'static str,
        mut run: impl FnMut(&dyn SearchBackend, &WaveHandle, bool) -> Result<T, Refusal>,
    ) -> Result<T, Refusal> {
        let tree = self.tree(handle)?;
        let mut st = tree.state.lock().unwrap_or_else(PoisonError::into_inner);
        let fault = match &st.on {
            On::Primary(h) => match guarded(method, || run(&*self.primary, h, false)) {
                Err(Refusal::Fault(why)) => why,
                other => return other,
            },
            On::Secondary(h) => return run(&**self.secondary_or_refuse()?, h, false),
            On::Neither(why) => {
                return Err(Refusal::Fault(format!(
                    "no backend holds this tree after the primary's fault: {why}"
                )))
            }
        };
        self.switch(tree.stride, &mut st, method, fault);
        let On::Secondary(h) = &st.on else {
            let why = st
                .route
                .switch
                .as_ref()
                .map_or_else(String::new, |s| s.reason.clone());
            return Err(Refusal::Fault(format!(
                "the primary faulted and no backend took the tree over: {why}"
            )));
        };
        run(&**self.secondary_or_refuse()?, h, true)
    }

    fn secondary_or_refuse(&self) -> Result<&Arc<dyn SearchBackend>, Refusal> {
        self.secondary
            .as_ref()
            .ok_or_else(|| Refusal::Invalid("a secondary tree without a secondary backend".into()))
    }
}

impl SearchBackend for FallbackBackend {
    /// The primary's name until it faults, then the secondary's (`none` when
    /// there is none). A tree's own route is [`FallbackBackend::route`].
    fn name(&self) -> &'static str {
        match (self.faulted.get(), &self.secondary) {
            (None, _) => self.primary.name(),
            (Some(_), Some(s)) => s.name(),
            (Some(_), None) => NO_BACKEND,
        }
    }

    fn caps(&self) -> BackendCaps {
        match (self.faulted.get(), &self.secondary) {
            (Some(_), Some(s)) => s.caps(),
            _ => self.primary.caps(),
        }
    }

    fn upload_wave(
        &self,
        problem: &WaveProblem,
        lease: &DeviceLease,
    ) -> Result<WaveHandle, Refusal> {
        let stride = problem.universe.len();
        let retain = self.secondary.is_some();
        let retained_bytes = if retain { problem.approx_bytes() } else { 0 };
        let primary_lease = DeviceLease {
            bytes: lease.bytes.saturating_sub(retained_bytes),
        };
        let primary = guarded("the primary's upload", || {
            self.primary.upload_wave(problem, &primary_lease)
        });
        let why = match primary {
            Ok(h) if h.stride() == stride => {
                *self
                    .primary_upload
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner) = None;
                let state = TreeState {
                    on: On::Primary(h),
                    retained: retain.then(|| problem.clone()),
                    lease: *lease,
                    route: WaveRoute::single(self.primary.name()),
                };
                return Ok(WaveHandle::new(
                    self.primary.name(),
                    stride,
                    Tree {
                        stride,
                        state: Mutex::new(state),
                    },
                ));
            }
            Ok(_) => "stride mismatch".to_string(),
            Err(r) => r.to_string(),
        };
        *self
            .primary_upload
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = Some(why.clone());
        let Some(secondary) = &self.secondary else {
            return Err(Refusal::Invalid(format!(
                "{}: {why} (no secondary backend)",
                self.primary.name()
            )));
        };
        // The fall-through: the secondary takes the whole tree, with the
        // whole lease (nothing is retained).
        let h = guarded("the secondary's upload", || {
            secondary.upload_wave(problem, lease)
        })?;
        if h.stride() != stride {
            return Err(Refusal::Invalid(format!(
                "{}: stride mismatch",
                secondary.name()
            )));
        }
        let state = TreeState {
            on: On::Secondary(h),
            retained: None,
            lease: *lease,
            route: WaveRoute {
                first: secondary.name(),
                primary_refused: Some(why),
                switch: None,
            },
        };
        Ok(WaveHandle::new(
            secondary.name(),
            stride,
            Tree {
                stride,
                state: Mutex::new(state),
            },
        ))
    }

    fn backward_batch(
        &self,
        handle: &WaveHandle,
        batch: &WaveBatch<'_>,
        cfg: &WaveSearchCfg,
        out: &mut WaveOut,
    ) -> Result<Untrusted<WaveStats>, Refusal> {
        self.call(handle, "backward_batch", |b, h, fresh| {
            if fresh {
                *out = WaveOut::default();
            }
            b.backward_batch(h, batch, cfg, out)
        })
    }

    fn branch_scores(
        &self,
        handle: &WaveHandle,
        batch: &WaveBatch<'_>,
        k: usize,
        out: &mut BranchOut,
    ) -> Result<Untrusted<()>, Refusal> {
        self.call(handle, "branch_scores", |b, h, fresh| {
            if fresh {
                *out = BranchOut::default();
            }
            b.branch_scores(h, batch, k, out)
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::search::{ReluBox, SearchNet, SearchOp, WaveSite};

    fn problem() -> WaveProblem {
        WaveProblem {
            net: SearchNet {
                n_in: 2,
                ops: vec![
                    SearchOp::Gemm {
                        input: 0,
                        n_out: 2,
                        n_in: 2,
                        weight: vec![1.0, -1.0, 0.5, 2.0],
                        bias: vec![0.25, -0.5],
                    },
                    SearchOp::Relu {
                        input: 1,
                        head: false,
                    },
                ],
            },
            boxes: vec![ReluBox {
                op: 1,
                l: vec![-1.0, 0.5],
                u: vec![2.0, 3.0],
            }],
            universe: vec![WaveSite { op: 1, neuron: 0 }],
            alpha0: vec![0.5],
            lo: vec![-1.0, -1.0],
            hi: vec![1.0, 1.0],
            target: 0,
            classes: vec![1],
        }
    }

    const ROWS: [u32; 1] = [0];
    const OFFSETS: [u32; 2] = [0, 0];

    fn batch() -> WaveBatch<'static> {
        WaveBatch {
            count: 1,
            rows: &ROWS,
            offsets: &OFFSETS,
            splits: &[],
            beta: &[],
            alpha: &[],
            stride: 1,
            deadline: None,
        }
    }

    /// An upload that counts itself live until its handle is dropped.
    struct Live(Arc<AtomicUsize>);

    impl Drop for Live {
        fn drop(&mut self) {
            self.0.fetch_sub(1, Ordering::SeqCst);
        }
    }

    /// What a scripted call does instead of answering.
    #[derive(Clone)]
    enum Act {
        Answer,
        Refuse(Refusal),
        Panic,
    }

    /// A scripted backend: its upload answers `upload`; call `i` (both methods
    /// counted together) does `script[i]` (answer past the end), answering
    /// `lb = tag` and `pos = tag`.
    struct Scripted {
        name: &'static str,
        tag: f32,
        upload: Option<Refusal>,
        script: Vec<Act>,
        calls: AtomicUsize,
        uploads: AtomicUsize,
        /// Live uploads of EVERY scripted backend of a test.
        live: Arc<AtomicUsize>,
        /// `live` at each upload of this backend (before it counts itself).
        live_at_upload: Mutex<Vec<usize>>,
        leases: Mutex<Vec<u64>>,
        /// The output's domain count on entry of each call.
        entry_counts: Mutex<Vec<usize>>,
    }

    impl Scripted {
        fn new(name: &'static str, tag: f32, live: &Arc<AtomicUsize>) -> Self {
            Self {
                name,
                tag,
                upload: None,
                script: Vec::new(),
                calls: AtomicUsize::new(0),
                uploads: AtomicUsize::new(0),
                live: Arc::clone(live),
                live_at_upload: Mutex::new(Vec::new()),
                leases: Mutex::new(Vec::new()),
                entry_counts: Mutex::new(Vec::new()),
            }
        }

        fn act(&self) -> Result<(), Refusal> {
            let i = self.calls.fetch_add(1, Ordering::SeqCst);
            match self.script.get(i).cloned().unwrap_or(Act::Answer) {
                Act::Answer => Ok(()),
                Act::Refuse(r) => Err(r),
                Act::Panic => panic!("injected panic in {}", self.name),
            }
        }
    }

    impl SearchBackend for Scripted {
        fn name(&self) -> &'static str {
            self.name
        }

        fn caps(&self) -> BackendCaps {
            BackendCaps::Cpu {
                threads: 1,
                lanes: 1,
                precision: crate::search::SearchPrecision::F32,
            }
        }

        fn upload_wave(
            &self,
            problem: &WaveProblem,
            lease: &DeviceLease,
        ) -> Result<WaveHandle, Refusal> {
            self.uploads.fetch_add(1, Ordering::SeqCst);
            self.leases.lock().unwrap().push(lease.bytes);
            if let Some(r) = &self.upload {
                return Err(r.clone());
            }
            self.live_at_upload
                .lock()
                .unwrap()
                .push(self.live.load(Ordering::SeqCst));
            self.live.fetch_add(1, Ordering::SeqCst);
            Ok(WaveHandle::new(
                self.name,
                problem.universe.len(),
                Live(Arc::clone(&self.live)),
            ))
        }

        fn backward_batch(
            &self,
            handle: &WaveHandle,
            batch: &WaveBatch<'_>,
            _cfg: &WaveSearchCfg,
            out: &mut WaveOut,
        ) -> Result<Untrusted<WaveStats>, Refusal> {
            assert!(handle.payload::<Live>().is_some(), "a foreign handle");
            self.entry_counts
                .lock()
                .unwrap()
                .push(out.hint().into_ingest().count());
            // Write first, then act: a faulting primary leaves a dirty output.
            out.prepare(batch.count, handle.stride(), batch.splits.len());
            out.writer().lb[0] = self.tag;
            self.act()?;
            Ok(Untrusted::new(WaveStats::default()))
        }

        fn branch_scores(
            &self,
            handle: &WaveHandle,
            batch: &WaveBatch<'_>,
            k: usize,
            out: &mut BranchOut,
        ) -> Result<Untrusted<()>, Refusal> {
            assert!(handle.payload::<Live>().is_some(), "a foreign handle");
            out.prepare(batch.count, k);
            out.writer().pos[0] = self.tag as u32;
            self.act()?;
            Ok(Untrusted::new(()))
        }
    }

    const LEASE: DeviceLease = DeviceLease { bytes: 1 << 20 };

    fn search(b: &dyn SearchBackend, h: &WaveHandle) -> Result<f32, Refusal> {
        let mut out = WaveOut::default();
        // The stats are not what these tests read (the output's `lb` is).
        let _stats = b.backward_batch(h, &batch(), &WaveSearchCfg::default(), &mut out)?;
        Ok(out.hint().into_ingest().lb(0))
    }

    fn score(b: &dyn SearchBackend, h: &WaveHandle) -> Result<u32, Refusal> {
        let mut out = BranchOut::default();
        let _unit = b.branch_scores(h, &batch(), 1, &mut out)?;
        Ok(out.hint().into_ingest().pos(0)[0])
    }

    fn pair(
        primary: Scripted,
        secondary: Scripted,
    ) -> (FallbackBackend, Arc<Scripted>, Arc<Scripted>) {
        let (p, s) = (Arc::new(primary), Arc::new(secondary));
        let fb = FallbackBackend::new(
            Arc::clone(&p) as Arc<dyn SearchBackend>,
            Some(Arc::clone(&s) as Arc<dyn SearchBackend>),
        );
        (fb, p, s)
    }

    #[test]
    fn fallback_upload_falls_through_on_a_primary_refusal_and_keeps_its_reason() {
        let live = Arc::new(AtomicUsize::new(0));
        let mut dev = Scripted::new("dev", 1.0, &live);
        dev.upload = Some(Refusal::Memory { need: 9, cap: 3 });
        let (fb, p, s) = pair(dev, Scripted::new("cpu", 2.0, &live));
        let h = fb
            .upload_wave(&problem(), &LEASE)
            .expect("the secondary takes the tree");
        let route = FallbackBackend::route(&h).expect("a composite handle");
        assert_eq!(route.first, "cpu");
        assert_eq!(
            route.primary_refused.as_deref(),
            Some("memory: need 9 B > cap 3 B")
        );
        assert_eq!(route.switch, None);
        assert_eq!(route.backends(), "cpu");
        assert_eq!(fb.primary_upload_refusal(), route.primary_refused);
        // Nothing is retained on a fall-through: the secondary gets the whole
        // lease, exactly as a `wave` tree's upload.
        assert_eq!(*s.leases.lock().unwrap(), vec![LEASE.bytes]);
        assert_eq!(search(&fb, &h), Ok(2.0));
        assert_eq!(score(&fb, &h), Ok(2));
        assert_eq!(p.calls.load(Ordering::SeqCst), 0);
        // A refusal is not a fault: the composite still names the primary.
        assert_eq!(fb.name(), "dev");
        // Both refuse: the secondary's refusal, unchanged; the primary's kept.
        let live = Arc::new(AtomicUsize::new(0));
        let mut dev = Scripted::new("dev", 1.0, &live);
        dev.upload = Some(Refusal::Unsupported("upload_wave"));
        let mut cpu = Scripted::new("cpu", 2.0, &live);
        cpu.upload = Some(Refusal::Nothing("no ReLU to relax"));
        let (fb, _, _) = pair(dev, cpu);
        assert_eq!(
            fb.upload_wave(&problem(), &LEASE).map(|_| ()),
            Err(Refusal::Nothing("no ReLU to relax"))
        );
        assert_eq!(
            fb.primary_upload_refusal().as_deref(),
            Some("unsupported: upload_wave")
        );
    }

    #[test]
    fn fallback_host_refusals_stay_on_the_primary() {
        let live = Arc::new(AtomicUsize::new(0));
        let mut dev = Scripted::new("dev", 1.0, &live);
        dev.script = vec![
            Act::Refuse(Refusal::Invalid("a slab needs 10 elements of pin".into())),
            Act::Refuse(Refusal::Deadline),
            Act::Refuse(Refusal::Memory { need: 2, cap: 1 }),
            Act::Refuse(Refusal::Invalid("branch k out of range".into())),
        ];
        let (fb, p, s) = pair(dev, Scripted::new("cpu", 2.0, &live));
        let h = fb
            .upload_wave(&problem(), &LEASE)
            .expect("the primary takes the tree");
        // Every non-fault refusal is that call's alone, returned unchanged.
        assert_eq!(
            search(&fb, &h),
            Err(Refusal::Invalid("a slab needs 10 elements of pin".into()))
        );
        assert_eq!(search(&fb, &h), Err(Refusal::Deadline));
        assert_eq!(search(&fb, &h), Err(Refusal::Memory { need: 2, cap: 1 }));
        assert_eq!(
            score(&fb, &h),
            Err(Refusal::Invalid("branch k out of range".into()))
        );
        // ... and the primary keeps the tree.
        assert_eq!(search(&fb, &h), Ok(1.0));
        assert_eq!(score(&fb, &h), Ok(1));
        assert_eq!(p.calls.load(Ordering::SeqCst), 6);
        assert_eq!(
            s.uploads.load(Ordering::SeqCst),
            0,
            "no switch on a host refusal"
        );
        assert_eq!(FallbackBackend::route(&h), Some(WaveRoute::single("dev")));
        assert_eq!(fb.primary_fault(), None);
        assert_eq!(fb.name(), "dev");
    }

    #[test]
    fn fallback_device_fault_switches_once_releasing_the_primary_first() {
        let live = Arc::new(AtomicUsize::new(0));
        let mut dev = Scripted::new("dev", 1.0, &live);
        dev.script = vec![
            Act::Answer,
            Act::Refuse(Refusal::Fault(
                "launch wave_forward: ILLEGAL_ADDRESS".into(),
            )),
        ];
        let (fb, p, s) = pair(dev, Scripted::new("cpu", 2.0, &live));
        let prob = problem();
        let retained = prob.approx_bytes();
        let h = fb
            .upload_wave(&prob, &LEASE)
            .expect("the primary takes the tree");
        // The retained copy is charged to the lease the primary is admitted on.
        assert_eq!(*p.leases.lock().unwrap(), vec![LEASE.bytes - retained]);
        assert_eq!(search(&fb, &h), Ok(1.0));
        // The fault: the call is RE-RUN on the secondary (no search lost), on a
        // clean output, and the rest of the tree is the secondary's.
        assert_eq!(search(&fb, &h), Ok(2.0));
        assert_eq!(
            *s.entry_counts.lock().unwrap(),
            vec![0],
            "the re-run must start from a clean output, not the primary's partial one"
        );
        assert_eq!(
            *s.live_at_upload.lock().unwrap(),
            vec![0],
            "the primary's upload must be released BEFORE the secondary's is built"
        );
        assert_eq!(*s.leases.lock().unwrap(), vec![LEASE.bytes - retained]);
        assert_eq!(search(&fb, &h), Ok(2.0));
        assert_eq!(score(&fb, &h), Ok(2));
        assert_eq!(
            p.calls.load(Ordering::SeqCst),
            2,
            "the primary is never asked again"
        );
        assert_eq!(
            s.uploads.load(Ordering::SeqCst),
            1,
            "the switch happens once"
        );
        let route = FallbackBackend::route(&h).expect("composite");
        assert_eq!(route.first, "dev");
        assert_eq!(route.backends(), "dev->cpu");
        assert_eq!(
            route.switch,
            Some(WaveSwitch {
                method: "backward_batch",
                reason: "launch wave_forward: ILLEGAL_ADDRESS".into(),
                to: Ok("cpu"),
            })
        );
        assert_eq!(fb.name(), "cpu");
        drop(h);
        assert_eq!(live.load(Ordering::SeqCst), 0, "every upload released");
    }

    #[test]
    fn fallback_branch_fault_and_primary_panics_switch_too() {
        for act in [
            Act::Refuse(Refusal::Fault("download bpos: LAUNCH_FAILED".into())),
            Act::Panic,
        ] {
            let live = Arc::new(AtomicUsize::new(0));
            let mut dev = Scripted::new("dev", 1.0, &live);
            dev.script = vec![act];
            let (fb, _, s) = pair(dev, Scripted::new("cpu", 2.0, &live));
            let h = fb.upload_wave(&problem(), &LEASE).expect("primary");
            assert_eq!(
                score(&fb, &h),
                Ok(2),
                "the faulted scoring re-runs on the secondary"
            );
            let route = FallbackBackend::route(&h).expect("composite");
            let sw = route.switch.expect("switched");
            assert_eq!((sw.method, sw.to), ("branch_scores", Ok("cpu")));
            assert_eq!(s.uploads.load(Ordering::SeqCst), 1);
            assert_eq!(search(&fb, &h), Ok(2.0));
        }
    }

    #[test]
    fn fallback_refused_reupload_refuses_every_later_call_with_the_fault() {
        let live = Arc::new(AtomicUsize::new(0));
        let mut dev = Scripted::new("dev", 1.0, &live);
        dev.script = vec![Act::Refuse(Refusal::Fault("injected".into()))];
        let mut cpu = Scripted::new("cpu", 2.0, &live);
        cpu.upload = Some(Refusal::Memory { need: 5, cap: 4 });
        let (fb, p, _) = pair(dev, cpu);
        let h = fb.upload_wave(&problem(), &LEASE).expect("primary");
        assert!(matches!(search(&fb, &h), Err(Refusal::Fault(_))));
        assert_eq!(
            live.load(Ordering::SeqCst),
            0,
            "the primary's upload is released"
        );
        assert!(matches!(search(&fb, &h), Err(Refusal::Fault(_))));
        assert!(matches!(score(&fb, &h), Err(Refusal::Fault(_))));
        assert_eq!(p.calls.load(Ordering::SeqCst), 1);
        let route = FallbackBackend::route(&h).expect("composite");
        assert_eq!(route.backends(), "dev->none");
        assert_eq!(
            route.switch.expect("switched").to,
            Err("memory: need 5 B > cap 4 B".to_string())
        );
        // Without a secondary at all: the same, and nothing is retained.
        let live = Arc::new(AtomicUsize::new(0));
        let mut dev = Scripted::new("dev", 1.0, &live);
        dev.script = vec![Act::Refuse(Refusal::Fault("injected".into()))];
        let dev = Arc::new(dev);
        let fb = FallbackBackend::new(Arc::clone(&dev) as Arc<dyn SearchBackend>, None);
        let h = fb.upload_wave(&problem(), &LEASE).expect("primary");
        assert_eq!(*dev.leases.lock().unwrap(), vec![LEASE.bytes]);
        assert!(matches!(search(&fb, &h), Err(Refusal::Fault(_))));
        assert_eq!(
            FallbackBackend::route(&h).expect("composite").backends(),
            "dev->none"
        );
        assert_eq!(fb.name(), NO_BACKEND);
    }

    #[test]
    fn fallback_refuses_foreign_handles_and_names_single_routes() {
        let live = Arc::new(AtomicUsize::new(0));
        let (fb, _, _) = pair(
            Scripted::new("dev", 1.0, &live),
            Scripted::new("cpu", 2.0, &live),
        );
        let foreign = WaveHandle::new("cpu", 1, 7u32);
        assert!(matches!(search(&fb, &foreign), Err(Refusal::Invalid(_))));
        assert!(matches!(score(&fb, &foreign), Err(Refusal::Invalid(_))));
        assert_eq!(FallbackBackend::route(&foreign), None);
        assert_eq!(WaveRoute::single("cuda").backends(), "cuda");
        assert_eq!(fb.names(), ("dev", Some("cpu")));
    }
}
