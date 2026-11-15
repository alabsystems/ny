// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The device WAVE search's own tests (#wave-beta-bab, ENGINE_PROGRAM M6). The
//! bit-identity KATs against ny-bbe's CPU f32 wave search (the reference) live
//! in ny-cli (`cuda_wave_search_tests`), the only crate that sees both
//! backends; these pin what ny-cuda can pin alone.
//!
//! Without a device (every host, including CI and the Mac):
//! * `cuda_wave_refuses_malformed_problems_and_oversized_leases_before_the_device`;
//! * `cuda_wave_refuses_foreign_handles_without_the_device`;
//! * `cuda_wave_latched_unavailability_refuses_every_later_call` (fault
//!   injection into both latches: the backend's and the wave kernels' own);
//! * `cuda_wave_host_refusals_never_latch_and_device_faults_do` (the backend
//!   also serves the level search: only a driver error may switch it off);
//! * `cuda_wave_without_libcuda_refuses_latches_and_never_panics`;
//! * `cuda_wave_plan_charges_every_device_buffer` (lease accounting);
//! * `wave_buffer_table_is_the_one_source_of_every_buffer_list` (the one
//!   buffer table: ABI pointer names, layout, lookup, pointer fill);
//! * `cuda_wave_every_admitted_launch_fits_the_buffers_it_indexes`;
//! * `cuda_wave_run_refuses_a_launch_past_an_allocation` (fault injection:
//!   an executor holding one element too few is refused before any launch);
//! * `cuda_wave_run_slabs_cover_every_domain_once_in_order`;
//! * `cuda_wave_run_refuses_a_passed_deadline_before_any_launch`;
//! * `wave_kernel_abi_matches_the_rust_side` (THE ABI RATCHET: `WaveArgs`,
//!   `AdamConsts`, every shared `#define`, no atomics, no contraction);
//! * `wave_kernel_names_are_the_image_entry_points`;
//! * `wave_self_test_fixture_is_exact` (the oracle has ONE answer);
//! * `wave_lowering_mirrors_the_cpu_reference_values` (chords, warm slopes,
//!   the box, the transposes, the inverse gather's order) — with MUTATION
//!   pins on the inverse gather's order and the chord;
//! * `wave_adam_constants_are_the_reference_expressions`.
//!
//! On a host that ADVERTISES a device the device tests below MUST pass;
//! elsewhere each prints `SKIP=VOID` — it ran, and it proved nothing about
//! the kernels:
//! * `cuda_wave_device_self_test_passes_and_uploads`;
//! * `cuda_wave_is_exact_on_the_dyadic_fixture`;
//! * `cuda_wave_outputs_do_not_depend_on_the_slab_or_the_batch_cut` (R6);
//! * `cuda_wave_sanitizes_dirty_warm_hints`;
//! * `cuda_wave_refuses_late_and_malformed_batches_on_a_live_upload`;
//! * `cuda_wave_restores_the_callers_current_context`.

use std::time::{Duration, Instant};

use ny_core::search::{
    BranchOut, DeviceLease, Refusal, SearchBackend, WaveBatch, WaveHandle, WaveOut, WaveProblem,
    WaveSearchCfg, WaveSplit,
};

use super::kernels::{WAVE_KERNELS, WAVE_SEARCH_FATBIN};
use super::tests::{define, kernels_dir, with_search_device};
use super::wave_device::kernel_index;
use super::wave_plan::{
    self, blocks, kat_problem, kat_wave, launch_needs, literals_for, oracle_pass, oracle_scores,
    slab_buffers, slab_bytes, upload_buffers, upload_bytes, warm_alpha, warm_beta, AdamConsts,
    OracleDomain, WaveArgs, WaveBuf, WavePlan, WaveProg, ADAM_CONSTS_BYTES, ADAM_CONSTS_FIELDS,
    WAVE_ARGS_BYTES, WAVE_ARGS_FIELDS, WAVE_ARGS_POINTERS,
};
use super::wave_run::{self, ExecError, Kernel, Launch, RunErr, SlabHost, WaveExec};
use super::{CudaSearchBackend, CUDA_WAVE_UNAVAILABLE};

const BIG: DeviceLease = DeviceLease { bytes: 1 << 34 };

fn kat() -> WaveProblem {
    kat_problem(wave_run::KAT_SEED)
}

/// A wave's owned inputs.
struct Wave {
    rows: Vec<u32>,
    offsets: Vec<u32>,
    splits: Vec<WaveSplit>,
    beta: Vec<f32>,
    alpha: Vec<f32>,
    stride: usize,
}

impl Wave {
    fn kat(stride: usize, count: usize, seed: u64) -> Self {
        let (rows, offsets, splits, beta, alpha) = kat_wave(stride, count, seed);
        Self {
            rows,
            offsets,
            splits,
            beta,
            alpha,
            stride,
        }
    }

    fn batch(&self) -> WaveBatch<'_> {
        WaveBatch {
            count: self.rows.len(),
            rows: &self.rows,
            offsets: &self.offsets,
            splits: &self.splits,
            beta: &self.beta,
            alpha: &self.alpha,
            stride: self.stride,
            deadline: None,
        }
    }
}

// ---------------------------------------------------------------------------
//  No device needed
// ---------------------------------------------------------------------------

#[test]
fn cuda_wave_refuses_malformed_problems_and_oversized_leases_before_the_device() {
    let backend = CudaSearchBackend::new();
    let mut bad = kat();
    bad.lo.pop();
    assert!(matches!(
        backend.upload_wave(&bad, &BIG),
        Err(Refusal::Invalid(_))
    ));
    let mut crossed = kat();
    crossed.boxes[0].l[0] = 5.0;
    crossed.boxes[0].u[0] = -5.0;
    assert!(matches!(
        backend.upload_wave(&crossed, &BIG),
        Err(Refusal::Invalid(_))
    ));
    let mut linear = kat();
    linear
        .net
        .ops
        .retain(|op| !matches!(op, ny_core::search::SearchOp::Relu { .. }));
    assert!(
        matches!(backend.upload_wave(&linear, &BIG), Err(Refusal::Nothing(_))),
        "a net without a ReLU has nothing to search (the CPU search's refusal)"
    );
    let p = kat();
    let prog = WaveProg::lower(&p).expect("lower");
    let need =
        upload_bytes(&prog) + slab_bytes(prog.t, prog.stride, 1, literals_for(prog.stride, 1));
    match backend.upload_wave(&p, &DeviceLease { bytes: need - 1 }) {
        Err(Refusal::Memory { need: got, cap }) => {
            assert_eq!((got, cap), (need, need - 1), "the refusal names the need");
        }
        other => panic!("a lease one byte short must refuse with Memory: {other:?}"),
    }
    assert!(
        !backend.probed(),
        "a malformed problem or an oversized lease must refuse BEFORE the device is opened"
    );
}

#[test]
fn cuda_wave_refuses_foreign_handles_without_the_device() {
    let backend = CudaSearchBackend::new();
    let w = Wave::kat(3, 2, 1);
    let mut out = WaveOut::default();
    let mut br = BranchOut::default();
    for foreign in [
        WaveHandle::new("cpu_f32", 3, 7u32),
        WaveHandle::new("cuda", 3, 7u32),
    ] {
        assert!(
            matches!(
                backend.backward_batch(&foreign, &w.batch(), &WaveSearchCfg::default(), &mut out),
                Err(Refusal::Invalid(_))
            ),
            "a handle this backend did not make must refuse"
        );
        assert!(matches!(
            backend.branch_scores(&foreign, &w.batch(), 2, &mut br),
            Err(Refusal::Invalid(_))
        ));
    }
    assert!(!backend.probed());
}

#[test]
fn cuda_wave_latched_unavailability_refuses_every_later_call() {
    // Fault injection 1: a backend fault latches the wave search too.
    let backend = CudaSearchBackend::new();
    backend.latch("injected device fault");
    // A latched FAULT refuses as a fault, on the upload and on every call a
    // live tree could still make (so a composite may switch the tree away).
    match backend.upload_wave(&kat(), &BIG) {
        Err(Refusal::Fault(why)) => {
            assert!(why.starts_with(CUDA_WAVE_UNAVAILABLE), "{why}");
            assert!(why.contains("injected device fault"), "{why}");
        }
        other => panic!("a latched backend must refuse as a fault: {other:?}"),
    }
    assert!(matches!(
        backend.check_wave_alive(),
        Err(Refusal::Fault(why)) if why.contains("injected device fault")
    ));
    assert!(
        !backend.probed(),
        "a latched backend must not open the device"
    );
    // Fault injection 2: a failed wave self-test latches the WAVE search
    // only; the level search is not refused by it.
    let wave_only = CudaSearchBackend::new();
    let _ = wave_only
        .wave
        .set(Err("wave kernels: injected self-test failure".to_string()));
    assert!(
        wave_only.unavailable().is_none(),
        "the level search is unaffected"
    );
    assert!(wave_only
        .wave_unavailable()
        .is_some_and(|w| w.contains("injected self-test failure")));
    match wave_only.upload_wave(&kat(), &BIG) {
        Err(Refusal::Invalid(why)) => assert!(why.contains("injected self-test"), "{why}"),
        other => panic!("a latched wave search must refuse: {other:?}"),
    }
    assert!(!wave_only.probed());
}

#[test]
fn cuda_wave_host_refusals_never_latch_and_device_faults_do() {
    // The backend also serves the level search: a refusal that never touched
    // the device (the bounds check, a size mismatch, a poisoned lock) must
    // not switch the device off; a driver error must.
    let backend = CudaSearchBackend::new();
    for host in [
        "a slab of 3 domains / 9 literals needs 10 elements of pin, which holds 9",
        "the wave upload's buffers are poisoned by an earlier panic",
    ] {
        match backend.wave_refusal(RunErr::Invalid(host.to_string())) {
            Refusal::Invalid(why) => assert!(why.contains(host), "{why}"),
            other => panic!("{other:?}"),
        }
        assert!(
            backend.unavailable().is_none() && backend.wave_unavailable().is_none(),
            "a host-side refusal latched the backend"
        );
    }
    assert_eq!(backend.wave_refusal(RunErr::Deadline), Refusal::Deadline);
    assert!(backend.unavailable().is_none());
    match backend.wave_refusal(RunErr::Fault("launch wave_forward: ILLEGAL_ADDRESS".into())) {
        Refusal::Fault(why) => assert!(why.contains("device fault"), "{why}"),
        other => panic!("a driver error must refuse as a Fault: {other:?}"),
    }
    assert!(
        backend
            .unavailable()
            .is_some_and(|why| why.contains("ILLEGAL_ADDRESS")),
        "a device fault must latch the backend (both searches) off"
    );
    assert!(backend.wave_unavailable().is_some());
    // The executor's error kinds map onto the two.
    assert_eq!(
        RunErr::from(ExecError::Host("h".into())),
        RunErr::Invalid("h".into())
    );
    assert_eq!(
        RunErr::from(ExecError::Device("d".into())),
        RunErr::Fault("d".into())
    );
}

#[test]
fn cuda_wave_without_libcuda_refuses_latches_and_never_panics() {
    let forced = CudaSearchBackend {
        driver_override: Some(false),
        ..CudaSearchBackend::new()
    };
    let upload = || {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            forced.upload_wave(&kat(), &BIG)
        }))
    };
    match upload().expect("upload_wave PANICKED on a host without libcuda") {
        Err(Refusal::Invalid(why)) => assert!(why.contains("libcuda"), "{why}"),
        other => panic!("a driverless upload must refuse: {other:?}"),
    }
    assert!(forced.probed(), "the absence must be latched");
    assert!(forced
        .wave_unavailable()
        .is_some_and(|w| w.contains("libcuda")));
    assert!(matches!(upload(), Ok(Err(Refusal::Invalid(_)))));
}

/// Problems of several shapes for the pure tests: the dyadic fixture and
/// wider variants of it.
fn shapes() -> Vec<WaveProg> {
    let mut out = vec![WaveProg::lower(&kat()).expect("lower")];
    for seed in [3u64, 17, 99] {
        out.push(WaveProg::lower(&kat_problem(seed)).expect("lower"));
    }
    out
}

#[test]
fn cuda_wave_plan_charges_every_device_buffer() {
    for prog in shapes() {
        let upload = upload_bytes(&prog);
        let by_hand =
            4 * (prog.ops.len() + prog.fp.len() + prog.ip.len() + prog.classes.len()) as u64;
        assert_eq!(upload, by_hand, "the upload charges exactly its tables");
        let min_need = upload + slab_bytes(prog.t, prog.stride, 1, literals_for(prog.stride, 1));
        assert!(matches!(
            wave_plan::plan(&prog, min_need - 1),
            Err(Refusal::Memory { .. })
        ));
        let at_min = wave_plan::plan(&prog, min_need).expect("the minimum admits");
        assert_eq!(at_min.nd, 1);
        assert_eq!(
            at_min.total(),
            min_need,
            "the plan charges exactly the layout"
        );
        for cap in [min_need + 12_345, min_need * 7, 1u64 << 33] {
            let p = wave_plan::plan(&prog, cap).expect("admits");
            assert!(p.total() <= cap, "cap={cap}: {p:?}");
            assert_eq!(p.slab, slab_bytes(prog.t, prog.stride, p.nd, p.nl));
            assert_eq!(p.nl, literals_for(prog.stride, p.nd));
            assert!(p.nd >= 1 && p.nd <= wave_plan::budget_domains(&prog));
            assert!(p.nl >= prog.stride, "a single domain always fits a slab");
        }
        // Per slab element: 8 bytes of tensor block per float (coef + val),
        // 32 per universe position, 40 per literal, 36 + 128 per domain.
        let (t, s) = (prog.t as u64, prog.stride as u64);
        let (nd, nl) = (5u64, 77u64);
        let by_hand = nd * t * 8
            + nd * s * (3 * 4 + 2 * 8 + 4)
            + nl * (6 * 4 + 2 * 8)
            + nd * 4
            + nd * (6 * 4 + 2 * 4)
            + nd * 16 * 8;
        assert_eq!(slab_bytes(prog.t, prog.stride, 5, 77), by_hand);
    }
    // Every buffer the ABI names is in the layouts, and the reverse.
    let prog = WaveProg::lower(&kat()).expect("lower");
    let mut names: Vec<&str> = upload_buffers(&prog)
        .iter()
        .map(|b| b.0)
        .chain(slab_buffers(1, 1, 1, 1).iter().map(|b| b.0))
        .collect();
    names.sort_unstable();
    let mut abi: Vec<&str> = WAVE_ARGS_FIELDS[..WAVE_ARGS_POINTERS]
        .iter()
        .map(|f| f.0)
        .collect();
    abi.sort_unstable();
    assert_eq!(names, abi, "every pointer of WaveArgs is a charged buffer");
}

/// Finding 8 of the M6 review: the ~25 device buffers were listed by hand in
/// five places (the ABI struct, its field list, the pointer fill, the pointer
/// view, the allocator's struct and lookup macro). They now come from ONE
/// table (`wave_buffers!` in wave_plan.rs); this pins that every consumer reads
/// it: the ABI's pointer names, the planner's layout (names, element types,
/// order), the name lookup, and a pointer fill that lands each buffer in its
/// own slot.
#[test]
fn wave_buffer_table_is_the_one_source_of_every_buffer_list() {
    let names: Vec<&str> = WaveBuf::ALL.iter().map(|b| b.name()).collect();
    let mut unique = names.clone();
    unique.sort_unstable();
    unique.dedup();
    assert_eq!(unique.len(), WAVE_ARGS_POINTERS, "a buffer name repeats");
    assert_eq!(
        WAVE_ARGS_FIELDS[..WAVE_ARGS_POINTERS]
            .iter()
            .map(|f| f.0)
            .collect::<Vec<_>>(),
        names,
        "the ABI's pointer fields are the table, in order"
    );
    let prog = WaveProg::lower(&kat()).expect("lower");
    let layout: Vec<(&str, wave_plan::Elem)> = upload_buffers(&prog)
        .iter()
        .map(|b| (b.0, b.1))
        .chain(slab_buffers(3, 2, 5, 7).iter().map(|b| (b.0, b.1)))
        .collect();
    assert_eq!(
        layout,
        WaveBuf::ALL
            .iter()
            .map(|b| (b.name(), b.elem()))
            .collect::<Vec<_>>(),
        "the planner's layout is the table, in ABI order"
    );
    for b in WaveBuf::ALL {
        assert_eq!(WaveBuf::from_name(b.name()), Some(b));
    }
    assert_eq!(WaveBuf::from_name("bogus"), None);
    let args = WaveArgs::default().with_ptrs(|b| 1000 + b as u64);
    for (i, p) in args.pointers().iter().enumerate() {
        assert_eq!(*p, 1000 + i as u64, "{} landed in another slot", names[i]);
    }
}

#[test]
fn cuda_wave_every_admitted_launch_fits_the_buffers_it_indexes() {
    let mut checked = 0usize;
    for prog in shapes() {
        let min_need =
            upload_bytes(&prog) + slab_bytes(prog.t, prog.stride, 1, literals_for(prog.stride, 1));
        for cap in [min_need, min_need + 99_999, min_need * 9, 1u64 << 33] {
            let p = wave_plan::plan(&prog, cap).expect("admits");
            let layout = slab_buffers(prog.t, prog.stride, p.nd, p.nl);
            for nd in [1, p.nd / 2 + 1, p.nd] {
                for nl in [0, p.nl / 3, p.nl] {
                    for (name, need) in launch_needs(prog.t, prog.stride, nd, nl) {
                        let have = layout
                            .iter()
                            .find(|b| b.0 == name)
                            .map(|b| b.2.max(1))
                            .unwrap_or_else(|| panic!("{name} is indexed but not allocated"));
                        assert!(
                            need <= have,
                            "cap={cap} nd={nd} nl={nl}: {name} {need} > {have}"
                        );
                        checked += 1;
                    }
                }
            }
        }
    }
    assert!(checked > 1_000, "VOID: {checked}");
}

/// A host executor with NO kernels: it records what the launch sequence asks
/// for and answers every state read with "no domain active" (so a slab runs
/// exactly one backward walk and one concretisation).
#[derive(Default)]
struct Recorder {
    lens: Vec<(&'static str, usize)>,
    puts: Vec<SlabHost>,
    launches: Vec<Launch>,
    zeroed: Vec<usize>,
}

impl Recorder {
    fn holding(prog: &WaveProg, plan: &WavePlan) -> Self {
        let lens = upload_buffers(prog)
            .iter()
            .chain(slab_buffers(prog.t, prog.stride, plan.nd, plan.nl).iter())
            .map(|&(n, _, l)| (n, l.max(1)))
            .collect();
        Self {
            lens,
            ..Self::default()
        }
    }
}

impl WaveExec for Recorder {
    fn len_of(&self, name: &str) -> usize {
        self.lens.iter().find(|b| b.0 == name).map_or(0, |b| b.1)
    }
    fn put(&mut self, host: &SlabHost) -> Result<(), ExecError> {
        self.puts.push(host.clone());
        Ok(())
    }
    fn zero_coef(&mut self, n: usize) -> Result<(), ExecError> {
        self.zeroed.push(n);
        Ok(())
    }
    fn launch(&mut self, l: &Launch) -> Result<(), ExecError> {
        self.launches.push(*l);
        Ok(())
    }
    fn get_dom(&mut self, dom_u: &mut [u32], dom_f: &mut [f32]) -> Result<(), ExecError> {
        dom_u.fill(0);
        dom_f.fill(f32::NAN);
        Ok(())
    }
    fn get_best(&mut self, _alpha: &mut [f32], _beta: &mut [f32]) -> Result<(), ExecError> {
        Ok(())
    }
    fn get_branch(&mut self, _pos: &mut [u32], _score: &mut [f32]) -> Result<(), ExecError> {
        Ok(())
    }
}

#[test]
fn cuda_wave_run_refuses_a_launch_past_an_allocation() {
    let prog = WaveProg::lower(&kat()).expect("lower");
    let plan = wave_plan::plan(&prog, 1 << 30).expect("admits");
    let w = Wave::kat(prog.stride, 12, 5);
    let cfg = WaveSearchCfg::default();
    // The honest recorder runs.
    let mut ok = Recorder::holding(&prog, &plan);
    let mut out = WaveOut::default();
    out.prepare(12, prog.stride, w.splits.len());
    wave_run::run_search(&mut ok, &prog, &plan, &w.batch(), &cfg, &mut out.writer())
        .expect("the honest layout runs");
    assert!(!ok.launches.is_empty(), "VOID: nothing launched");
    // Fault injection: every buffer, one element short in turn.
    let mut refused = 0usize;
    for victim in 0..ok.lens.len() {
        let mut short = Recorder::holding(&prog, &plan);
        let name = short.lens[victim].0;
        let need = upload_buffers(&prog)
            .iter()
            .map(|b| (b.0, b.2))
            .chain(launch_needs(prog.t, prog.stride, 12, w.splits.len()))
            .find(|b| b.0 == name)
            .map_or(0, |b| b.1);
        if need == 0 {
            continue;
        }
        short.lens[victim].1 = need - 1;
        let mut out = WaveOut::default();
        out.prepare(12, prog.stride, w.splits.len());
        match wave_run::run_search(
            &mut short,
            &prog,
            &plan,
            &w.batch(),
            &cfg,
            &mut out.writer(),
        ) {
            // A HOST-side refusal: the device was never touched, so the
            // backend must not latch on it.
            Err(RunErr::Invalid(why)) => assert!(why.contains(short.lens[victim].0), "{why}"),
            other => panic!("{} one short must refuse: {other:?}", short.lens[victim].0),
        }
        assert!(
            short.launches.is_empty() && short.puts.is_empty(),
            "{}: a launch was issued before the bounds check refused",
            short.lens[victim].0
        );
        refused += 1;
    }
    assert!(refused >= 20, "VOID: only {refused} buffers were exercised");
}

#[test]
fn cuda_wave_run_slabs_cover_every_domain_once_in_order() {
    let prog = WaveProg::lower(&kat()).expect("lower");
    let w = Wave::kat(prog.stride, 23, 11);
    for (nd, nl) in [
        (1usize, prog.stride),
        (3, prog.stride),
        (4, prog.stride.max(5)),
        (64, 4096),
    ] {
        let plan = WavePlan {
            upload: upload_bytes(&prog),
            nd,
            nl,
            slab: slab_bytes(prog.t, prog.stride, nd, nl),
        };
        let mut x = Recorder::holding(&prog, &plan);
        let mut br = BranchOut::default();
        br.prepare(23, 3);
        wave_run::run_branch(&mut x, &prog, &plan, &w.batch(), 3, &mut br.writer()).expect("runs");
        let (mut rows, mut lits, mut alpha) = (Vec::new(), Vec::new(), Vec::new());
        for host in &x.puts {
            assert!(
                host.rows.len() <= nd && host.lit_pos.len() <= nl,
                "nd={nd} nl={nl}"
            );
            // Literal domains are slab-relative, in order.
            assert!(host.lit_dom.windows(2).all(|p| p[0] <= p[1]));
            assert!(host.lit_dom.iter().all(|&d| (d as usize) < host.rows.len()));
            rows.extend_from_slice(&host.rows);
            lits.extend_from_slice(&host.lit_pos);
            alpha.extend_from_slice(&host.alpha);
        }
        assert_eq!(rows, w.rows, "every domain exactly once, in order");
        assert_eq!(lits, w.splits.iter().map(|s| s.pos).collect::<Vec<_>>());
        let bits = |v: &[f32]| v.iter().map(|a| a.to_bits()).collect::<Vec<_>>();
        assert_eq!(
            bits(&alpha),
            bits(&w.alpha),
            "the warm slopes travel raw, in order"
        );
        let babsr = x
            .launches
            .iter()
            .filter(|l| l.kernel == Kernel::Babsr)
            .count();
        assert_eq!(babsr, x.puts.len(), "one scoring launch per slab");
    }
}

#[test]
fn cuda_wave_run_refuses_a_passed_deadline_before_any_launch() {
    let prog = WaveProg::lower(&kat()).expect("lower");
    let plan = wave_plan::plan(&prog, 1 << 30).expect("admits");
    let w = Wave::kat(prog.stride, 4, 2);
    let late = WaveBatch {
        deadline: Some(
            Instant::now()
                .checked_sub(Duration::from_millis(1))
                .unwrap_or_else(Instant::now),
        ),
        ..w.batch()
    };
    let mut x = Recorder::holding(&prog, &plan);
    let mut out = WaveOut::default();
    out.prepare(4, prog.stride, w.splits.len());
    assert_eq!(
        wave_run::run_search(
            &mut x,
            &prog,
            &plan,
            &late,
            &WaveSearchCfg::default(),
            &mut out.writer()
        ),
        Err(RunErr::Deadline)
    );
    assert_eq!(x.launches, [] as [Launch; 0]);
}

/// The fields of `struct <name> { ... };` in the kernel source: `(name,
/// kind)` with kind `p` pointer, `l` long long, `i` int, `d` double.
fn c_struct_fields(text: &str, name: &str) -> Vec<(String, char)> {
    let start = text
        .find(&format!("struct {name} {{"))
        .unwrap_or_else(|| panic!("the kernel source defines no struct {name}"));
    let body = &text[start..];
    let body = &body[body.find('{').expect("brace") + 1..body.find("};").expect("end")];
    body.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with("//"))
        .map(|l| {
            let l = l.trim_end_matches(';');
            let field = l.rsplit([' ', '*']).next().expect("field");
            let kind = if l.contains('*') {
                'p'
            } else if l.starts_with("long long") {
                'l'
            } else if l.starts_with("int") {
                'i'
            } else if l.starts_with("double") {
                'd'
            } else {
                '?'
            };
            (field.to_string(), kind)
        })
        .collect()
}

#[test]
fn wave_kernel_abi_matches_the_rust_side() {
    let text = std::fs::read_to_string(kernels_dir().join("wave_search.cu")).expect("source");
    let fields = c_struct_fields(&text, "WaveArgs");
    let want: Vec<(String, char)> = WAVE_ARGS_FIELDS
        .iter()
        .map(|&(n, k)| (n.to_string(), k))
        .collect();
    assert_eq!(
        fields, want,
        "struct WaveArgs differs from wave_plan.rs WAVE_ARGS_FIELDS"
    );
    assert!(WAVE_ARGS_FIELDS[..WAVE_ARGS_POINTERS]
        .iter()
        .all(|f| f.1 == 'p'));
    assert!(WAVE_ARGS_FIELDS[WAVE_ARGS_POINTERS..]
        .iter()
        .all(|f| f.1 != 'p'));
    assert!(text.contains(&format!(
        "static_assert(sizeof(WaveArgs) == {WAVE_ARGS_BYTES},"
    )));
    let adam = c_struct_fields(&text, "AdamConsts");
    assert_eq!(
        adam,
        ADAM_CONSTS_FIELDS
            .iter()
            .map(|&n| (n.to_string(), 'd'))
            .collect::<Vec<_>>()
    );
    assert!(text.contains(&format!(
        "static_assert(sizeof(AdamConsts) == {ADAM_CONSTS_BYTES},"
    )));
    // The Rust struct's own layout: pointers, then the 64-bit size, then ints.
    let a = WaveArgs::default();
    let base = std::ptr::addr_of!(a) as usize;
    assert_eq!(
        std::ptr::addr_of!(a.ptrs[WAVE_ARGS_POINTERS - 1]) as usize - base,
        8 * (WAVE_ARGS_POINTERS - 1)
    );
    assert_eq!(
        std::ptr::addr_of!(a.t) as usize - base,
        8 * WAVE_ARGS_POINTERS
    );
    assert_eq!(std::ptr::addr_of!(a.k) as usize - base, WAVE_ARGS_BYTES - 4);
    // Every shared constant.
    use wave_plan as w;
    let want: Vec<(&str, i64)> = vec![
        ("WAVE_THREADS", i64::from(w::WAVE_THREADS)),
        ("WAVE_BABSR_THREADS", i64::from(w::WAVE_BABSR_THREADS)),
        ("WAVE_MAX_K", w::WAVE_MAX_K as i64),
        ("OP_FIELDS", w::OP_FIELDS as i64),
        ("OP_CONV", i64::from(w::OP_CONV)),
        ("OP_AFFINE", i64::from(w::OP_AFFINE)),
        ("OP_RELU", i64::from(w::OP_RELU)),
        ("OP_ADD", i64::from(w::OP_ADD)),
        ("OP_COPY", i64::from(w::OP_COPY)),
        ("OP_GEMM", i64::from(w::OP_GEMM)),
        ("F_KIND", w::F_KIND as i64),
        ("F_OUT", w::F_OUT as i64),
        ("F_NOUT", w::F_NOUT as i64),
        ("F_IN", w::F_IN as i64),
        ("F_NIN", w::F_NIN as i64),
        ("F_RHS", w::F_RHS as i64),
        ("F_A", w::F_A as i64),
        ("F_B", w::F_B as i64),
        ("F_C", w::F_C as i64),
        ("F_W", w::F_W as i64),
        ("F_BIAS", w::F_BIAS as i64),
        ("F_G", w::F_G as i64),
        ("F_INV_START", w::F_INV_START as i64),
        ("F_INV_ENT", w::F_INV_ENT as i64),
        ("F_WT", w::F_WT as i64),
        ("F_SITE", w::F_SITE as i64),
        ("ST_ACTIVE", i64::from(w::ST_ACTIVE)),
        ("ST_GRAD", i64::from(w::ST_GRAD)),
        ("ST_IMPROVED", i64::from(w::ST_IMPROVED)),
        ("ST_EARLY", i64::from(w::ST_EARLY)),
        ("ST_HAS_BEST", i64::from(w::ST_HAS_BEST)),
        ("DOM_F", w::DOM_F as i64),
        ("DF_BIAS", w::DF_BIAS as i64),
        ("DF_BIAS_ABS", w::DF_BIAS_ABS as i64),
        ("DF_LB", w::DF_LB as i64),
        ("DF_SCALE", w::DF_SCALE as i64),
        ("DF_BEST_LB", w::DF_BEST_LB as i64),
        ("DF_BEST_SCALE", w::DF_BEST_SCALE as i64),
        ("DOM_U", w::DOM_U as i64),
        ("DU_STATE", w::DU_STATE as i64),
        ("DU_EVALS", w::DU_EVALS as i64),
        ("PHASE_DOMAINS", i64::from(w::PHASE_DOMAINS)),
        ("PHASE_LITERALS", i64::from(w::PHASE_LITERALS)),
        ("PHASE_SEED", i64::from(w::PHASE_SEED)),
    ];
    for (name, value) in want {
        assert_eq!(define(&text, name), Some(value), "#define {name}");
    }
    assert!(text.contains("#define WAVE_NONE 0xffffffffu"));
    // An atomic INTRINSIC (`atomicAdd`, `atomicCAS`, ...): the word in a
    // comment is fine, a call is not.
    let atomic_call = text
        .match_indices("atomic")
        .any(|(i, _)| text[i + 6..].starts_with(|c: char| c.is_ascii_uppercase()));
    assert!(
        !atomic_call
            && !text.contains("__fmaf")
            && !text.contains("fmaf(")
            && !text.contains("__fma_rn")
            && !text.contains(" fma("),
        "the wave kernels must not reduce with atomics nor contract multiply-adds"
    );
}

#[test]
fn wave_kernel_names_are_the_image_entry_points() {
    let text = std::fs::read_to_string(kernels_dir().join("wave_search.cu")).expect("source");
    for k in [
        Kernel::Init,
        Kernel::Backward,
        Kernel::Concretize,
        Kernel::Forward,
        Kernel::Adam,
        Kernel::Babsr,
    ] {
        assert_eq!(WAVE_KERNELS[kernel_index(k)], k.name());
        assert!(text.contains(&format!(
            "extern \"C\" __global__ void {}(const WaveArgs A",
            k.name()
        )));
        assert!(WAVE_SEARCH_FATBIN
            .windows(k.name().len())
            .any(|w| w == k.name().as_bytes()));
    }
}

/// The self-test's oracle domain `d` of a KAT wave (hints ingested).
fn oracle_domain(prog: &WaveProg, w: &Wave, d: usize) -> OracleDomain {
    let r = w.offsets[d] as usize..w.offsets[d + 1] as usize;
    OracleDomain {
        row: w.rows[d] as usize,
        lits: w.splits[r.clone()]
            .iter()
            .zip(&w.beta[r])
            .map(|(s, &b)| (s.pos, s.active, warm_beta(b)))
            .collect(),
        alpha: (0..prog.stride)
            .map(|p| warm_alpha(w.alpha[d * prog.stride + p], prog.fp[prog.sec.alpha0 + p]))
            .collect(),
    }
}

#[test]
fn wave_self_test_fixture_is_exact() {
    // The self-test demands the device's bounds bit for bit against the f64
    // oracle: a valid oracle only if every f32 partial sum of the fixture is
    // exact, i.e. the f32 sequential transcription gives the same bits.
    let prog = wave_run::kat_prog().expect("kat");
    let w = Wave::kat(prog.stride, wave_run::KAT_DOMAINS, wave_run::KAT_SEED);
    let (mut lits, mut sides, mut nonzero) = (0usize, [0usize; 2], 0usize);
    for d in 0..wave_run::KAT_DOMAINS {
        let dom = oracle_domain(&prog, &w, d);
        let a = oracle_pass::<f32>(&prog, &dom);
        let b = oracle_pass::<f64>(&prog, &dom);
        assert_eq!(f64::from(a.lb), b.lb, "domain {d} bound");
        assert_eq!(f64::from(a.scale), b.scale, "domain {d} S");
        for (x, y) in a.lam.iter().zip(&b.lam) {
            assert_eq!(f64::from(*x), *y, "domain {d} lam");
        }
        assert!(b.lb.is_finite() && b.scale > 0.0);
        nonzero += b.lam.iter().filter(|l| **l != 0.0).count();
        lits += dom.lits.len();
        for l in &dom.lits {
            sides[usize::from(l.1)] += 1;
        }
        let top = oracle_scores(&prog, &dom, &b.lam, 4);
        assert!(top.windows(2).all(|p| p[0].1 >= p[1].1));
    }
    assert!(prog.stride > 10 && lits > 4 && sides[0] > 0 && sides[1] > 0 && nonzero > 10);
    // Every op kind the kernels serve is in the fixture.
    let kinds: std::collections::BTreeSet<i32> = (0..prog.nops())
        .map(|k| prog.op(k)[wave_plan::F_KIND])
        .collect();
    assert_eq!(
        kinds.len(),
        6,
        "the fixture must exercise every op kind: {kinds:?}"
    );
}

#[test]
fn wave_lowering_mirrors_the_cpu_reference_values() {
    use wave_plan::site_neuron;
    // Chords and adaptive slopes (ny-bbe `SiteState::<f32>::new`).
    assert_eq!(site_neuron(-1.0, 3.0), (0.75, 0.75, 1.0));
    assert_eq!(site_neuron(-3.0, 1.0), (0.25, 0.75, 0.0));
    assert_eq!(
        site_neuron(1.0, 3.0),
        (0.0, 0.0, 1.0),
        "stable active: no chord"
    );
    assert_eq!(
        site_neuron(-3.0, -1.0),
        (0.0, 0.0, 0.0),
        "stable inactive: no chord"
    );
    // An infinite lower side: the chord degenerates to the intercept `u`, and
    // `u >= -l` fails, so the adaptive slope is 0.
    assert_eq!(site_neuron(f32::NEG_INFINITY, 2.0), (0.0, 2.0, 0.0));
    let (s, i, _) = site_neuron(-1.0, f32::INFINITY);
    assert_eq!((s, i), (0.0, f32::INFINITY));
    // MUTATION pin: a chord through the wrong end would not be the reference's.
    assert_ne!(site_neuron(-1.0, 3.0).0, 0.25);
    let p = kat();
    let prog = WaveProg::lower(&p).expect("lower");
    let s = &prog.sec;
    // Warm slopes: finite -> clamp, NaN -> adaptive.
    for (q, &a) in p.alpha0.iter().enumerate() {
        let got = prog.fp[s.alpha0 + q];
        let flat = prog.ip[s.uni_flat + q] as usize;
        let want = if a.is_finite() {
            a.clamp(0.0, 1.0) as f32
        } else {
            prog.fp[s.s_adapt + flat]
        };
        assert_eq!(got.to_bits(), want.to_bits(), "alpha0[{q}]");
    }
    for i in 0..p.net.n_in {
        assert_eq!(prog.fp[s.mid + i], f64::midpoint(p.lo[i], p.hi[i]) as f32);
        assert_eq!(prog.fp[s.rad + i], ((p.hi[i] - p.lo[i]) * 0.5) as f32);
    }
    // Every universe position maps to its own site neuron.
    for q in 0..prog.stride {
        let flat = prog.ip[s.uni_flat + q] as usize;
        assert_eq!(prog.ip[s.s_pos + flat] as usize, q);
    }
    // The conv inverse gather: every non-pad tap exactly once, per input in
    // (output position, tap) order — the reference's scatter order.
    let d = prog.op(0);
    assert_eq!(d[wave_plan::F_KIND], wave_plan::OP_CONV);
    let (taps, pp, nin) = (
        d[wave_plan::F_B] as usize,
        d[wave_plan::F_C] as usize,
        d[wave_plan::F_NIN] as usize,
    );
    let start =
        &prog.ip[d[wave_plan::F_INV_START] as usize..=d[wave_plan::F_INV_START] as usize + nin];
    let ent = &prog.ip[d[wave_plan::F_INV_ENT] as usize..];
    let gat = &prog.ip[d[wave_plan::F_G] as usize..d[wave_plan::F_G] as usize + taps * pp];
    let mut seen = 0usize;
    for g in 0..nin {
        let list: Vec<(u32, u32)> = (start[g]..start[g + 1])
            .map(|e| (ent[2 * e as usize], ent[2 * e as usize + 1]))
            .collect();
        assert!(
            list.windows(2).all(|w| w[0] < w[1]),
            "input {g}: not in (sp, t) order"
        );
        for &(sp, t) in &list {
            assert_eq!(gat[t as usize * pp + sp as usize] as usize, g);
        }
        seen += list.len();
    }
    assert_eq!(seen, gat.iter().filter(|&&g| g != wave_plan::NONE).count());
    // The dense transpose.
    let gemm = (0..prog.nops())
        .map(|k| prog.op(k))
        .find(|d| d[wave_plan::F_KIND] == wave_plan::OP_GEMM)
        .expect("a gemm");
    let (no, ni) = (gemm[wave_plan::F_A] as usize, gemm[wave_plan::F_B] as usize);
    for o in 0..no {
        for i in 0..ni {
            assert_eq!(
                prog.fp[gemm[wave_plan::F_W] as usize + o * ni + i].to_bits(),
                prog.fp[gemm[wave_plan::F_WT] as usize + i * no + o].to_bits()
            );
        }
    }
    assert!(prog.check().is_ok());
    // MUTATION pin: a lowering whose descriptor points past its table is
    // refused by the defence-in-depth check.
    let mut broken = prog.clone();
    broken.ops[wave_plan::F_W] = i32::try_from(broken.fp.len()).expect("fits");
    assert!(broken.check().is_err());
    let mut swapped = prog.clone();
    swapped.ops[wave_plan::F_IN] = swapped.ops[wave_plan::F_OUT];
    assert!(
        swapped.check().is_err(),
        "an op reading its own output must be refused"
    );
    // An input that starts before the output but runs into it (the backward's
    // gather writes and the bias chain's reads would race) is refused too.
    let mut overlapping = prog.clone();
    let (out0, nin0) = (
        overlapping.ops[wave_plan::F_OUT],
        overlapping.ops[wave_plan::F_NIN],
    );
    assert!(nin0 > 1 && out0 >= 1, "VOID: op 0 has no input to overlap");
    overlapping.ops[wave_plan::F_IN] = out0 - 1;
    match overlapping.check() {
        Err(Refusal::Invalid(why)) => assert!(
            why.contains("does not precede its output"),
            "refused for another reason, so the overlap itself is unpinned: {why}"
        ),
        other => panic!("an input that overlaps its op's output must be refused: {other:?}"),
    }
    let mut stray = prog.clone();
    let slot = stray.sec.s_pos
        + stray.ip[stray.sec.s_pos..stray.sec.s_pos + stray.sites]
            .iter()
            .position(|&q| q == wave_plan::NONE)
            .expect("a site neuron outside the universe");
    stray.ip[slot] = u32::try_from(stray.stride).expect("fits");
    assert!(
        stray.check().is_err(),
        "a site neuron mapped past the universe must be refused"
    );
}

#[test]
fn wave_adam_constants_are_the_reference_expressions() {
    // The reference's expressions are `ny_core::search`'s (ONE copy, read by
    // the CPU search too), evaluated at RUNTIME: a literal exponent may fold to
    // other bits than the runtime `powi` (`1 - 0.9^4`), so the expected values
    // are computed with a black-boxed exponent.
    let rt = |x: f64, t: i32| x.powi(std::hint::black_box(t));
    let k0 = AdamConsts::at(0, 0.1, 0.05);
    assert_eq!(k0.c1, 1.0 - rt(0.9, 1));
    assert_eq!(k0.c2, 1.0 - rt(0.999, 1));
    assert_eq!(
        (k0.lr_a, k0.lr_b),
        (0.1, 0.05),
        "no decay at the first step"
    );
    assert_eq!(k0.omb1, 1.0 - 0.9);
    let k3 = AdamConsts::at(3, 0.2, 0.3);
    assert_eq!(k3.c1, 1.0 - rt(0.9, 4));
    assert_eq!(k3.c2, 1.0 - rt(0.999, 4));
    assert_eq!((k3.c1, k3.c2), ny_core::search::wave_adam_bias(4));
    assert_eq!(k3.lr_a, 0.2 * rt(0.98, 3));
    assert_eq!(k3.lr_b, 0.3 * rt(0.98, 3));
    let dirty = AdamConsts::at(0, f64::NAN, -1.0);
    assert_eq!(
        (dirty.lr_a, dirty.lr_b),
        (0.1, 0.05),
        "the reference's defaults"
    );
    assert_eq!(blocks(0), 1);
    assert_eq!(blocks(257), 2);
    let args =
        WaveArgs::sized(&WaveProg::lower(&kat()).expect("lower"), 3, 5, true, 4).with_ptrs(|_| 1);
    assert!(args.pointers().iter().all(|&p| p == 1));
    assert_eq!((args.nd, args.nl, args.has_alpha, args.k), (3, 5, 1, 4));
}

// ---------------------------------------------------------------------------
//  Device tests (gated on an advertised device; SKIP is VOID)
// ---------------------------------------------------------------------------

/// Every output of a wave, bit for bit (NaN canonicalised; the sign of a zero
/// is not an output).
fn digest(out: &WaveOut) -> Vec<u32> {
    let v = out.hint().into_ingest();
    let canon = |x: f32| {
        if x.is_nan() {
            0x7FC0_0000
        } else if x == 0.0 {
            0
        } else {
            x.to_bits()
        }
    };
    let mut d = Vec::new();
    for k in 0..v.count() {
        d.extend([canon(v.lb(k)), canon(v.scale(k)), v.evals(k)]);
        d.extend(v.alpha(k).iter().map(|&a| canon(a)));
    }
    d.extend(v.beta().iter().map(|&b| canon(b)));
    d
}

fn search(
    backend: &CudaSearchBackend,
    h: &WaveHandle,
    b: &WaveBatch<'_>,
    cfg: &WaveSearchCfg,
) -> WaveOut {
    let mut out = WaveOut::default();
    let _ = backend
        .backward_batch(h, b, cfg, &mut out)
        .expect("backward_batch");
    out
}

/// A non-dyadic variant of the fixture (rounding everywhere, so an order that
/// moved with the geometry would show).
fn rough(seed: u64) -> WaveProblem {
    let mut p = kat_problem(seed);
    for (k, op) in p.net.ops.iter_mut().enumerate() {
        let f = |i: usize| 1.0 + ((i + k) % 7) as f64 * 1.1e-3;
        match op {
            ny_core::search::SearchOp::Conv(c) => {
                for (i, w) in c.weight.iter_mut().enumerate() {
                    *w *= f(i);
                }
            }
            ny_core::search::SearchOp::Gemm { weight, .. } => {
                for (i, w) in weight.iter_mut().enumerate() {
                    *w *= f(i);
                }
            }
            _ => {}
        }
    }
    p
}

#[test]
fn cuda_wave_device_self_test_passes_and_uploads() {
    with_search_device("cuda_wave_device_self_test_passes_and_uploads", |backend| {
        let p = kat();
        let h = backend
            .upload_wave(&p, &BIG)
            .expect("a device that opened must serve the wave search");
        assert_eq!(h.backend(), "cuda");
        assert_eq!(h.stride(), p.universe.len());
        assert!(backend.wave_unavailable().is_none());
        assert!(
            matches!(backend.wave.get(), Some(Ok(_))),
            "the self-test latched ok"
        );
    });
}

#[test]
fn cuda_wave_is_exact_on_the_dyadic_fixture() {
    with_search_device("cuda_wave_is_exact_on_the_dyadic_fixture", |backend| {
        let mut checked = 0usize;
        for seed in [wave_run::KAT_SEED, 3, 17] {
            let p = kat_problem(seed);
            let prog = WaveProg::lower(&p).expect("lower");
            let h = backend.upload_wave(&p, &BIG).expect("upload");
            let w = Wave::kat(prog.stride, 13, seed ^ 7);
            let cfg = WaveSearchCfg {
                iters: 0,
                stop_above: f64::INFINITY,
                ..WaveSearchCfg::default()
            };
            let out = search(backend, &h, &w.batch(), &cfg);
            let v = out.hint().into_ingest();
            let mut br = BranchOut::default();
            let _ = backend
                .branch_scores(&h, &w.batch(), 5, &mut br)
                .expect("branch");
            let bv = br.hint().into_ingest();
            for d in 0..13 {
                let dom = oracle_domain(&prog, &w, d);
                let want = oracle_pass::<f64>(&prog, &dom);
                assert_eq!(
                    (v.lb(d).to_bits(), v.scale(d).to_bits(), v.evals(d)),
                    ((want.lb as f32).to_bits(), (want.scale as f32).to_bits(), 1),
                    "seed {seed} domain {d}"
                );
                let top = oracle_scores(&prog, &dom, &want.lam, 5);
                for c in 0..5 {
                    let (wp, ws) = top.get(c).copied().unwrap_or((u32::MAX, f32::NAN));
                    assert_eq!(bv.pos(d)[c], wp, "seed {seed} domain {d} candidate {c}");
                    assert!(wave_run::same_value(bv.score(d)[c], ws));
                }
                checked += 1;
            }
        }
        assert!(checked >= 39, "VOID: {checked}");
    });
}

#[test]
fn cuda_wave_outputs_do_not_depend_on_the_slab_or_the_batch_cut() {
    with_search_device(
        "cuda_wave_outputs_do_not_depend_on_the_slab_or_the_batch_cut",
        |backend| {
            let p = rough(41);
            let prog = WaveProg::lower(&p).expect("lower");
            let w = Wave::kat(prog.stride, 29, 5);
            let cfg = WaveSearchCfg::default();
            let h = backend.upload_wave(&p, &BIG).expect("upload");
            let base = digest(&search(backend, &h, &w.batch(), &cfg));
            // Leases that admit slabs of 1, 2 and 5 domains.
            for nd in [1usize, 2, 5] {
                let lease = DeviceLease {
                    bytes: upload_bytes(&prog)
                        + slab_bytes(prog.t, prog.stride, nd, literals_for(prog.stride, nd)),
                };
                let small = backend.upload_wave(&p, &lease).expect("upload");
                assert_eq!(
                    base,
                    digest(&search(backend, &small, &w.batch(), &cfg)),
                    "slabs of {nd} domains changed the search"
                );
            }
            // The same domains in two calls.
            let cut = 11usize;
            let (lo, hi) = (w.offsets[cut] as usize, w.splits.len());
            let first = WaveBatch {
                count: cut,
                rows: &w.rows[..cut],
                offsets: &w.offsets[..=cut],
                splits: &w.splits[..lo],
                beta: &w.beta[..lo],
                alpha: &w.alpha[..cut * prog.stride],
                ..w.batch()
            };
            let rest_offsets: Vec<u32> = w.offsets[cut..]
                .iter()
                .map(|o| o - w.offsets[cut])
                .collect();
            let second = WaveBatch {
                count: 29 - cut,
                rows: &w.rows[cut..],
                offsets: &rest_offsets,
                splits: &w.splits[lo..hi],
                beta: &w.beta[lo..hi],
                alpha: &w.alpha[cut * prog.stride..],
                ..w.batch()
            };
            let (a, b) = (
                search(backend, &h, &first, &cfg),
                search(backend, &h, &second, &cfg),
            );
            let (va, vb, vt) = (a.hint().into_ingest(), b.hint().into_ingest(), {
                let all = search(backend, &h, &w.batch(), &cfg);
                all
            });
            let vt = vt.hint().into_ingest();
            for d in 0..29 {
                let (lb, alpha) = if d < cut {
                    (va.lb(d), va.alpha(d).to_vec())
                } else {
                    (vb.lb(d - cut), vb.alpha(d - cut).to_vec())
                };
                assert!(wave_run::same_value(lb, vt.lb(d)), "domain {d}");
                assert!(alpha
                    .iter()
                    .zip(vt.alpha(d))
                    .all(|(x, y)| wave_run::same_value(*x, *y)));
            }
            let stepped = (0..29).filter(|&d| vt.evals(d) > 1).count();
            assert!(stepped > 0, "VOID: no domain took an ascent step");
        },
    );
}

#[test]
fn cuda_wave_sanitizes_dirty_warm_hints() {
    with_search_device("cuda_wave_sanitizes_dirty_warm_hints", |backend| {
        let p = rough(43);
        let prog = WaveProg::lower(&p).expect("lower");
        let h = backend.upload_wave(&p, &BIG).expect("upload");
        let dirty = Wave::kat(prog.stride, 17, 9);
        // The same wave with every hint replaced by its ingested value.
        let mut clean = Wave::kat(prog.stride, 17, 9);
        for b in &mut clean.beta {
            *b = warm_beta(*b);
        }
        for d in 0..17 {
            for q in 0..prog.stride {
                let a = &mut clean.alpha[d * prog.stride + q];
                *a = warm_alpha(*a, prog.fp[prog.sec.alpha0 + q]);
            }
        }
        assert!(
            dirty.beta.iter().any(|b| !b.is_finite() || *b < 0.0)
                && dirty.alpha.iter().any(|a| !a.is_finite() || *a > 1.0),
            "VOID: the fixture carries no dirty hint"
        );
        let cfg = WaveSearchCfg::default();
        assert_eq!(
            digest(&search(backend, &h, &dirty.batch(), &cfg)),
            digest(&search(backend, &h, &clean.batch(), &cfg)),
            "a dirty hint must act as its ingested value"
        );
    });
}

#[test]
fn cuda_wave_refuses_late_and_malformed_batches_on_a_live_upload() {
    with_search_device(
        "cuda_wave_refuses_late_and_malformed_batches_on_a_live_upload",
        |backend| {
            let p = kat();
            let h = backend.upload_wave(&p, &BIG).expect("upload");
            let w = Wave::kat(p.universe.len(), 5, 3);
            let mut out = WaveOut::default();
            let cfg = WaveSearchCfg::default();
            let stats = backend
                .backward_batch(&h, &w.batch(), &cfg, &mut out)
                .expect("a valid batch searches")
                .into_hint()
                .into_ingest();
            assert_eq!(stats.domains, 5);
            assert!(stats.passes >= 5, "{stats:?}");
            let wrong_stride = WaveBatch {
                stride: w.stride + 1,
                ..w.batch()
            };
            assert!(matches!(
                backend.backward_batch(&h, &wrong_stride, &cfg, &mut out),
                Err(Refusal::Invalid(_))
            ));
            let late = WaveBatch {
                deadline: Some(
                    Instant::now()
                        .checked_sub(Duration::from_millis(1))
                        .unwrap_or_else(Instant::now),
                ),
                ..w.batch()
            };
            assert!(matches!(
                backend.backward_batch(&h, &late, &cfg, &mut out),
                Err(Refusal::Deadline)
            ));
            let mut br = BranchOut::default();
            for k in [0usize, 17] {
                assert!(matches!(
                    backend.branch_scores(&h, &w.batch(), k, &mut br),
                    Err(Refusal::Invalid(_))
                ));
            }
            let empty = WaveBatch {
                count: 0,
                rows: &[],
                offsets: &[0],
                splits: &[],
                beta: &[],
                alpha: &[],
                ..w.batch()
            };
            let stats = backend
                .backward_batch(&h, &empty, &cfg, &mut out)
                .expect("an empty wave is fine")
                .into_hint()
                .into_ingest();
            assert_eq!(stats.domains, 0);
            assert!(backend.unavailable().is_none() && backend.wave_unavailable().is_none());
        },
    );
}

#[test]
fn cuda_wave_restores_the_callers_current_context() {
    with_search_device(
        "cuda_wave_restores_the_callers_current_context",
        |backend| {
            let current = || cudarc::driver::result::ctx::get_current().ok().flatten();
            let p = kat();
            let w = Wave::kat(p.universe.len(), 4, 1);
            let before = current();
            let h = backend.upload_wave(&p, &BIG).expect("upload");
            let _ = search(backend, &h, &w.batch(), &WaveSearchCfg::default());
            assert_eq!(
                before,
                current(),
                "the wave search left its context current"
            );
            let primary = cudarc::driver::CudaContext::new(0).expect("primary context");
            let foreign = Some(primary.cu_ctx());
            assert_eq!(
                current(),
                foreign,
                "VOID: the primary context is not current"
            );
            let _ = search(backend, &h, &w.batch(), &WaveSearchCfg::default());
            let mut br = BranchOut::default();
            let _ = backend
                .branch_scores(&h, &w.batch(), 3, &mut br)
                .expect("branch");
            assert_eq!(
                current(),
                foreign,
                "a wave search did not restore the caller's context"
            );
            let fresh = CudaSearchBackend::new();
            let h2 = fresh
                .upload_wave(&p, &BIG)
                .expect("a first wave upload opens the device");
            let _ = search(&fresh, &h2, &w.batch(), &WaveSearchCfg::default());
            drop(h2);
            assert_eq!(
                current(),
                foreign,
                "opening a device for the wave search did not restore the caller's context"
            );
        },
    );
}
