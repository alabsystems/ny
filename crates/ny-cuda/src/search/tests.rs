// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! The device level search's own tests. The bit-identity KATs against the CPU
//! f32 search (the reference) live in ny-cli (`cuda_fd_search_tests`), the
//! only crate that sees both backends; these pin what ny-cuda can pin alone.
//!
//! Without a device (every host, including CI and the Mac):
//! * `cuda_search_construction_touches_no_driver`;
//! * `cuda_search_refuses_malformed_problems_and_oversized_leases_before_the_device`;
//! * `cuda_search_refuses_foreign_handles_without_the_device`;
//! * `cuda_search_latched_unavailability_refuses_every_later_call` (fault
//!   injection into the latch);
//! * `cuda_search_plan_charges_every_device_buffer` (lease accounting);
//! * `cuda_search_device_cap_keeps_the_reserve`;
//! * `cuda_search_without_libcuda_refuses_latches_and_never_panics` (fault
//!   injection of a driverless host on every host; the real path too where
//!   libcuda is genuinely absent);
//! * `cuda_search_oom_is_a_lease_refusal_and_any_other_device_error_latches`;
//! * `cuda_search_every_admitted_launch_fits_the_buffers_it_indexes`;
//! * `search_kernel_manifest_matches_the_committed_sources` (THE MANIFEST
//!   RATCHET);
//! * `search_kernel_source_constants_match_the_rust_side`;
//! * `search_host_math_mirrors_track_libm_and_the_documented_edges`;
//! * `search_self_test_fixture_is_exact`;
//! * `search_constants_round_like_the_cpu_search`.
//!
//! On a host that ADVERTISES a device (libcuda loadable, device count > 0) the
//! device tests below MUST pass; elsewhere each prints `SKIP=VOID` — it ran,
//! and it proved nothing about the kernels:
//! * `cuda_fd_device_self_test_passes_and_probes_caps`;
//! * `cuda_fd_warm_is_exact_on_dyadic_levels`;
//! * `cuda_fd_outputs_do_not_depend_on_the_slab_or_the_chunk_cut` (R6);
//! * `cuda_fd_sanitizes_dirty_multipliers_and_refuses_overflowing_nodes`;
//! * `cuda_fd_restores_the_callers_current_context`;
//! * `cuda_fd_refuses_late_malformed_and_deeper_chunks_on_a_live_upload`.

use std::time::{Duration, Instant};

use std::fmt::Write as _;

use ny_core::search::{
    BackendCaps, DeviceLease, FdHandle, FdSearchProblem, LevelChunk, LevelOut, Refusal,
    SearchBackend, FD_CLAMPED, FD_FAR, FD_FLOOR, FD_LOG_BUCKETS, FD_POINTS, FD_SUM_TILE, FD_WARM,
};
use sha2::{Digest, Sha256};

use super::kernels::{
    manifest_value, FD_SEARCH_FATBIN, KERNELS, REQUIRED_FLAGS, WAVE_KERNELS, WAVE_SEARCH_FATBIN,
};
use super::plan::{self, FD_NB};
use super::CudaSearchBackend;

const BIG: DeviceLease = DeviceLease { bytes: 1 << 34 };

pub(super) fn kernels_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels")
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hex = String::with_capacity(64);
    for b in Sha256::digest(bytes).iter() {
        let _ = write!(hex, "{b:02x}");
    }
    hex
}

fn small_problem() -> FdSearchProblem {
    plan::kat_problem(40, 4, 3)
}

fn chunk<'a>(depth: usize, sides: &'a [u8], l0: &'a [f32], l1: &'a [f32]) -> LevelChunk<'a> {
    LevelChunk {
        depth,
        count: sides.len() / depth,
        sides,
        l0,
        l1,
        deadline: None,
    }
}

// ---------------------------------------------------------------------------
//  No device needed
// ---------------------------------------------------------------------------

#[test]
fn cuda_search_construction_touches_no_driver() {
    let backend = CudaSearchBackend::new();
    assert_eq!(backend.name(), "cuda");
    assert!(!backend.probed(), "construction must not open the device");
    assert_eq!(
        backend.caps(),
        BackendCaps::Cuda {
            sm: 0,
            unified: false,
            mem_bytes: 0
        },
        "an unprobed backend reports zero caps"
    );
    assert!(backend.unavailable().is_none());
    // The root-alpha method (M5a) is not served by this backend.
    assert!(matches!(
        backend.root_joint_alpha(
            &ny_core::search::SearchNet {
                n_in: 1,
                ops: Vec::new()
            },
            &ny_core::search::RootSearchRequest {
                lo: vec![0.0],
                hi: vec![1.0],
                ref_boxes: Vec::new(),
                target: 0,
                adv: Vec::new(),
                iters: 0,
                lr: 0.0,
                deadline: None,
                threads: 1,
                mem_cap_bytes: 0,
            }
        ),
        Err(Refusal::Unsupported(_))
    ));
    assert!(!backend.probed());
}

#[test]
fn cuda_search_refuses_malformed_problems_and_oversized_leases_before_the_device() {
    let backend = CudaSearchBackend::new();
    let mut bad = small_problem();
    bad.c0 = f32::NAN;
    assert!(matches!(
        backend.upload_fd(&bad, &BIG),
        Err(Refusal::Invalid(_))
    ));
    let mut short = small_problem();
    short.a.pop();
    assert!(matches!(
        backend.upload_fd(&short, &BIG),
        Err(Refusal::Invalid(_))
    ));
    let p = small_problem();
    let need = plan::upload_bytes(p.n, p.depth) + plan::slab_bytes(FD_NB, FD_NB * p.depth);
    match backend.upload_fd(&p, &DeviceLease { bytes: need - 1 }) {
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
fn cuda_search_refuses_foreign_handles_without_the_device() {
    let backend = CudaSearchBackend::new();
    let (sides, l0, l1) = plan::kat_level(3, 2, 1);
    let ok = chunk(2, &sides, &l0, &l1);
    let mut out = LevelOut::default();
    for foreign in [FdHandle::new("cpu_f32", 7u32), FdHandle::new("cuda", 7u32)] {
        assert!(
            matches!(
                backend.fd_level(&foreign, &ok, &mut out),
                Err(Refusal::Invalid(_))
            ),
            "a handle this backend did not make must refuse"
        );
    }
    assert!(!backend.probed());
}

#[test]
fn cuda_search_latched_unavailability_refuses_every_later_call() {
    // Fault injection: latch the backend as a device fault would, then every
    // upload refuses at once — without opening the device.
    let backend = CudaSearchBackend::new();
    backend.latch("injected device fault");
    assert_eq!(
        backend.unavailable().as_deref(),
        Some("injected device fault")
    );
    // A LATCHED fault refuses as a fault (a composite may switch away on it).
    match backend.upload_fd(&small_problem(), &BIG) {
        Err(Refusal::Fault(why)) => assert!(why.contains("injected device fault"), "{why}"),
        other => panic!("a latched backend must refuse as a fault: {other:?}"),
    }
    assert!(
        !backend.probed(),
        "a latched backend must not open the device"
    );
    // The first latch wins (the reason is not overwritten).
    backend.latch("a second fault");
    assert_eq!(
        backend.unavailable().as_deref(),
        Some("injected device fault")
    );
}

#[test]
fn cuda_search_plan_charges_every_device_buffer() {
    for (n, depth) in [
        (1usize, 0usize),
        (1, 1),
        (300, 5),
        (11_418, 28),
        (3_448, 512),
    ] {
        let upload = plan::upload_bytes(n, depth);
        let by_hand = 4 * (depth * n) as u64
            + 4 * n as u64
            + 4 * n as u64
            + 32 * depth as u64
            + 4 * depth as u64
            + 4 * plan::FD_NK as u64;
        // `max(len, 1)` floors on empty buffers.
        assert!(
            upload >= by_hand && upload <= by_hand + 16,
            "n={n} depth={depth}"
        );
        let d = depth.max(1);
        let min_need = upload + plan::slab_bytes(FD_NB, FD_NB * d);
        assert!(matches!(
            plan::plan(n, depth, min_need - 1),
            Err(Refusal::Memory { .. })
        ));
        let at_min = plan::plan(n, depth, min_need).expect("the minimum admits");
        assert_eq!((at_min.slab_nodes, at_min.slab_nd), (FD_NB, FD_NB * d));
        assert_eq!(
            at_min.total(),
            min_need,
            "the plan charges exactly the layout"
        );
        for cap in [min_need + 12_345, min_need * 3, 1u64 << 33] {
            let p = plan::plan(n, depth, cap).expect("admits");
            assert!(p.total() <= cap, "n={n} depth={depth} cap={cap}: {p:?}");
            assert_eq!(p.slab, plan::slab_bytes(p.slab_nodes, p.slab_nd));
            assert!(p.slab_nodes >= FD_NB && p.slab_nodes <= plan::MAX_SLAB_NODES);
            assert!(p.slab_nd >= FD_NB * d);
            for dc in 1..=d {
                let k = plan::slab_nodes(n, dc, &p);
                assert!(
                    k >= 1 && k <= p.slab_nodes && k * dc <= p.slab_nd,
                    "n={n} depth={depth} cap={cap} dc={dc}: a launch of {k} exceeds {p:?}"
                );
                if cap == 1u64 << 33 {
                    assert_eq!(
                        k,
                        plan::budget_nodes(n, dc),
                        "an ample lease must not cap the launch (dc={dc})"
                    );
                }
            }
        }
    }
    // A deep, wide state (tiny: 11,418 generators, 512 rows) reserves what the
    // work budget can ask of it, not MAX_SLAB_NODES x 512 rows (~714 MB).
    let deep = plan::plan(11_418, 512, 1u64 << 33).expect("admits");
    assert!(
        deep.slab < 64 << 20,
        "the deep slab reserves {} B: {deep:?}",
        deep.slab
    );
    // Every slab buffer the kernels index is in the layout, per node.
    let names: Vec<&str> = plan::slab_buffers(1, 1).iter().map(|b| b.0).collect();
    for want in [
        "sides",
        "l0",
        "l1",
        "vec",
        "lm",
        "val",
        "abs",
        "flags",
        "slope",
        "eta",
        "evals",
        "items",
        "item_count",
        "out_clamped",
        "out_far",
        "out_next",
    ] {
        assert!(names.contains(&want), "{want} is not charged");
    }
}

#[test]
fn cuda_search_without_libcuda_refuses_latches_and_never_panics() {
    // cudarc PANICS on a driver entry point it cannot resolve; a host without
    // libcuda must get a latched Refusal from the first valid upload — and no
    // panic (which the lane would contain, but a refusal is the contract).
    let upload = |backend: &CudaSearchBackend| {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            backend.upload_fd(&small_problem(), &BIG)
        }))
    };
    let forced = CudaSearchBackend {
        driver_override: Some(false),
        ..CudaSearchBackend::new()
    };
    let mut hosts = vec![("forced", forced)];
    if !crate::CudaGemmEngine::runtime_driver_library_present() {
        // This host genuinely has no libcuda: the REAL probe path as well.
        hosts.push(("real", CudaSearchBackend::new()));
    }
    for (tag, backend) in &hosts {
        let result = upload(backend)
            .unwrap_or_else(|_| panic!("{tag}: upload_fd PANICKED on a host without libcuda"));
        match result {
            Err(Refusal::Invalid(why)) => assert!(why.contains("libcuda"), "{tag}: {why}"),
            other => panic!("{tag}: a driverless upload must refuse: {other:?}"),
        }
        assert!(backend.probed(), "{tag}: the absence must be latched");
        assert!(
            backend
                .unavailable()
                .is_some_and(|why| why.contains("libcuda")),
            "{tag}: {:?}",
            backend.unavailable()
        );
        // Every later call refuses at once, still without a panic.
        assert!(
            matches!(upload(backend), Ok(Err(Refusal::Invalid(_)))),
            "{tag}"
        );
        assert_eq!(
            backend.caps(),
            BackendCaps::Cuda {
                sm: 0,
                unified: false,
                mem_bytes: 0
            },
            "{tag}"
        );
    }
    eprintln!(
        "[cuda-search-test] driverless paths exercised: {:?}",
        hosts.iter().map(|h| h.0).collect::<Vec<_>>()
    );
}

#[test]
fn cuda_search_oom_is_a_lease_refusal_and_any_other_device_error_latches() {
    use super::device::{drv, DevError, DevKind};
    use cudarc::driver::result::DriverError;
    use cudarc::driver::sys::CUresult;
    let oom = DevError {
        kind: DevKind::Oom,
        what: "alloc: CUDA_ERROR_OUT_OF_MEMORY".to_string(),
    };
    let fault = DevError::fault("launch: CUDA_ERROR_ILLEGAL_ADDRESS");
    let host = DevError::host("a launch of 3 nodes at depth 2 needs 7 elements of slab buffer l0");
    if crate::CudaGemmEngine::runtime_driver_library_present() {
        // The classification of real driver errors (formatting one resolves
        // `cuGetErrorString`, which needs libcuda — as every real error has).
        let c_oom = drv("alloc")(DriverError(CUresult::CUDA_ERROR_OUT_OF_MEMORY));
        let c_fault = drv("launch")(DriverError(CUresult::CUDA_ERROR_ILLEGAL_ADDRESS));
        assert_eq!(c_oom.kind, DevKind::Oom, "{c_oom:?}");
        assert!(
            c_fault.kind == DevKind::Driver && c_fault.what.contains("ILLEGAL_ADDRESS"),
            "{c_fault:?}"
        );
    }
    let backend = CudaSearchBackend::new();
    assert_eq!(
        backend.refusal("fd", &oom, 10, 5),
        Refusal::Memory { need: 10, cap: 5 }
    );
    assert!(
        backend.unavailable().is_none(),
        "an out-of-memory is a lease refusal: it must NOT latch the backend off"
    );
    // A HOST-side refusal (the launch bounds check, a size conversion, a
    // poisoned lock) refuses that call only: the level and the wave search
    // share this backend, so it must never switch the device off.
    for search in ["fd", "wave"] {
        match backend.refusal(search, &host, 10, 5) {
            Refusal::Invalid(why) => assert!(why.contains("slab buffer l0"), "{why}"),
            other => panic!("a host-side refusal must refuse as Invalid: {other:?}"),
        }
        assert!(
            backend.unavailable().is_none() && backend.wave_unavailable().is_none(),
            "a host-side {search} refusal latched the backend off"
        );
    }
    match backend.refusal("fd", &fault, 10, 5) {
        Refusal::Fault(why) => assert!(why.contains("ILLEGAL_ADDRESS"), "{why}"),
        other => panic!("a device fault must refuse as a Fault: {other:?}"),
    }
    assert!(
        backend
            .unavailable()
            .is_some_and(|why| why.contains("ILLEGAL_ADDRESS")),
        "a device fault must latch the backend off"
    );
    match backend.upload_fd(&small_problem(), &BIG) {
        Err(Refusal::Fault(why)) => assert!(why.contains("ILLEGAL_ADDRESS"), "{why}"),
        other => panic!("a latched fault must refuse every later upload: {other:?}"),
    }
    assert!(!backend.probed(), "the refusals must not open the device");
}

#[test]
fn cuda_search_every_admitted_launch_fits_the_buffers_it_indexes() {
    // What the kernels index per launch (`launch_needs`) against what the
    // planner charges and the allocator allocates (`slab_buffers`), for every
    // launch the planner lets `fd_level` issue at every chunk depth.
    let mut checked = 0usize;
    for (n, depth) in [
        (1usize, 1usize),
        (40, 4),
        (300, 5),
        (777, 20),
        (11_418, 28),
        (3_448, 512),
        (11_418, 512),
    ] {
        let min_need = plan::upload_bytes(n, depth) + plan::slab_bytes(FD_NB, FD_NB * depth);
        for cap in [min_need, min_need + 99_999, min_need * 5, 1u64 << 33] {
            let p = plan::plan(n, depth, cap).expect("admits");
            let layout = plan::slab_buffers(p.slab_nodes, p.slab_nd);
            let have = |name: &str| {
                layout
                    .iter()
                    .find(|b| b.0 == name)
                    .map(|b| b.2.max(1))
                    .unwrap_or_else(|| panic!("{name} is indexed but not in the layout"))
            };
            for dc in 1..=depth {
                let k = plan::slab_nodes(n, dc, &p);
                for count in [1, k] {
                    for (name, need) in plan::launch_needs(count, dc) {
                        assert!(
                            need <= have(name),
                            "n={n} depth={depth} cap={cap} dc={dc} count={count}: {name} needs \
                             {need} > {}",
                            have(name)
                        );
                        checked += 1;
                    }
                }
            }
        }
    }
    // The need table names exactly the layout's buffers.
    let mut needs: Vec<&str> = plan::launch_needs(1, 1).iter().map(|b| b.0).collect();
    let mut names: Vec<&str> = plan::slab_buffers(1, 1).iter().map(|b| b.0).collect();
    needs.sort_unstable();
    names.sort_unstable();
    assert_eq!(needs, names);
    assert!(checked > 10_000, "VOID: {checked}");
}

#[test]
fn cuda_search_device_cap_keeps_the_reserve() {
    let gib = 1u64 << 30;
    assert_eq!(
        plan::device_cap(gib, 24 * gib),
        0,
        "1 GiB free is all reserve"
    );
    assert_eq!(
        plan::device_cap(20 * gib, 24 * gib),
        20 * gib - plan::DEVICE_RESERVE_BYTES
    );
    assert!(plan::device_cap(24 * gib, 24 * gib) <= 24 * gib / 20 * 17);
    assert_eq!(plan::device_cap(0, 0), 0);
}

#[test]
fn search_kernel_manifest_matches_the_committed_sources() {
    let dir = kernels_dir();
    let read = |name: &str| {
        std::fs::read(dir.join(name))
            .unwrap_or_else(|e| panic!("cannot read kernels/{name}: {e} (the ratchet must see it)"))
    };
    assert_eq!(manifest_value("format"), Some("2"), "manifest format");
    // Every image: (manifest key prefix, source, image, embedded bytes, entry points).
    let images: [(&str, &str, &str, &[u8], &[&str]); 2] = [
        (
            "",
            "fd_search.cu",
            "fd_search.fatbin",
            FD_SEARCH_FATBIN,
            &KERNELS,
        ),
        (
            "wave_",
            "wave_search.cu",
            "wave_search.fatbin",
            WAVE_SEARCH_FATBIN,
            &WAVE_KERNELS,
        ),
    ];
    let key = |prefix: &str, k: &str| manifest_value(&format!("{prefix}{k}"));
    for (prefix, want_src, want_img, embedded, kernels) in images {
        let source = key(prefix, "source").expect("manifest names the source");
        let image = key(prefix, "image").expect("manifest names the image");
        assert_eq!((source, image), (want_src, want_img));
        let src = read(source);
        let img = read(image);
        assert_eq!(
            key(prefix, "source_sha256"),
            Some(sha256_hex(&src).as_str()),
            "kernels/{source} changed without rebuilding the image: run \
             scripts/build_search_kernels.sh and commit the sources, images and manifest together"
        );
        assert_eq!(
            key(prefix, "image_sha256"),
            Some(sha256_hex(&img).as_str()),
            "kernels/{image} does not match the manifest: rebuild with \
             scripts/build_search_kernels.sh"
        );
        assert_eq!(
            embedded,
            img.as_slice(),
            "the embedded image differs from kernels/{image}"
        );
        let listed: Vec<&str> = key(prefix, "kernels")
            .expect("manifest lists the kernels")
            .split(',')
            .collect();
        assert_eq!(listed, kernels.to_vec(), "manifest kernel list of {image}");
        let text = String::from_utf8(src).expect("utf-8 source");
        for name in kernels {
            assert!(
                text.contains(&format!("{name}(")) && text.contains("extern \"C\" __global__"),
                "kernels/{source} does not define {name}"
            );
            assert!(
                img.windows(name.len()).any(|w| w == name.as_bytes()),
                "the image carries no entry point {name}"
            );
        }
    }
    let flags = manifest_value("flags").expect("manifest records the flags");
    for f in REQUIRED_FLAGS {
        assert!(
            flags.split_whitespace().any(|x| x == f),
            "flags lack {f}: {flags}"
        );
    }
    for arch in ["code=sm_86", "code=sm_121", "code=compute_80"] {
        assert!(flags.contains(arch), "flags lack {arch}: {flags}");
    }
    assert!(
        manifest_value("nvcc").is_some_and(|v| v.starts_with("Cuda compilation tools")),
        "manifest records the nvcc version"
    );
}

/// `#define NAME value` of the kernel source.
pub(super) fn define(text: &str, name: &str) -> Option<i64> {
    text.lines().find_map(|l| {
        let mut it = l.split_whitespace();
        (it.next() == Some("#define") && it.next() == Some(name))
            .then(|| it.next().and_then(|v| v.trim_end_matches('u').parse().ok()))
            .flatten()
    })
}

#[test]
fn search_kernel_source_constants_match_the_rust_side() {
    let text = std::fs::read_to_string(kernels_dir().join("fd_search.cu")).expect("source");
    let want: [(&str, i64); 11] = [
        ("FD_NB", FD_NB as i64),
        ("FD_THREADS", i64::from(plan::FD_THREADS)),
        ("FD_TILE", FD_SUM_TILE as i64),
        ("FD_BUCKETS", FD_LOG_BUCKETS as i64),
        ("FD_POINTS", FD_POINTS as i64),
        ("FD_COMPACT_THREADS", i64::from(plan::FD_COMPACT_THREADS)),
        ("FD_NVEC", plan::FD_NVEC as i64),
        ("FD_NK", plan::FD_NK as i64),
        ("PT_WARM", FD_WARM as i64),
        ("PT_CLAMPED", FD_CLAMPED as i64),
        ("PT_FAR", FD_FAR as i64),
    ];
    for (name, value) in want {
        assert_eq!(define(&text, name), Some(value), "#define {name}");
    }
    assert_eq!(define(&text, "PT_FLOOR"), Some(FD_FLOOR as i64));
    assert!(text.contains(&format!(
        "#define EVALS_REFUSED 0x{:08x}u",
        plan::EVALS_REFUSED
    )));
    assert!(
        !text.contains("atomicAdd") && !text.contains("__fmaf") && !text.contains("fmaf("),
        "the kernels must not reduce with atomics nor contract multiply-adds"
    );
}

#[test]
fn search_host_math_mirrors_track_libm_and_the_documented_edges() {
    let mut worst = 0.0f64;
    for k in 0..5_645u32 {
        let x = -87.0f32 + k as f32 * 0.031;
        let got = f64::from(plan::host_fast_exp(x));
        let want = f64::from(x).exp();
        let rel = (got - want).abs() / want;
        worst = worst.max(rel);
        assert!(rel <= 1e-6, "exp({x:e}) = {got:e} vs {want:e}");
    }
    let mut y = 1e-12f32;
    for _ in 0..80 {
        let got = f64::from(plan::host_fast_ln(y));
        let want = f64::from(y).ln();
        assert!((got - want).abs() <= 1e-5 + 4e-7 * want.abs(), "ln({y:e})");
        y *= 3.3;
    }
    assert_eq!(plan::host_fast_exp(0.0), 1.0);
    assert!(plan::host_fast_exp(f32::NAN).is_nan());
    assert_eq!(plan::host_fast_exp(f32::INFINITY), f32::INFINITY);
    assert_eq!(plan::host_fast_exp(f32::NEG_INFINITY), 0.0);
    assert_eq!(plan::host_fast_ln(f32::INFINITY), f32::INFINITY);
    assert!(plan::same_bits(f32::NAN, -f32::NAN));
    assert!(
        !plan::same_bits(0.0, -0.0),
        "same_bits is BITS (zeros are compared by callers)"
    );
    eprintln!("[cuda-search-math] worst exp rel {worst:e}");
}

#[test]
fn search_self_test_fixture_is_exact() {
    // The self-test demands the device's warm (value, S) bit for bit against
    // `exact_warm`: that is only a valid oracle if every f32 partial sum of the
    // fixture is exact, i.e. an f32 SEQUENTIAL evaluation gives the same bits.
    let p = plan::kat_problem(300, 5, 0x5E1F);
    let (sides, l0, l1) = plan::kat_level(11, 5, 0x5E1F);
    for k in 0..11 {
        let r = k * 5..(k + 1) * 5;
        let (v, s) = plan::exact_warm(&p, &sides[r.clone()], &l0[r.clone()], &l1[r.clone()]);
        // f32 sequential transcription (column order, no tiles).
        let d = 5;
        let mut rc: Vec<f32> = p.d.clone();
        for &e in &p.enc[..d] {
            rc[e as usize] = 0.0;
        }
        let (mut c0path, mut c0abs) = (p.c0, p.c0.abs());
        let (mut lag, mut lagabs, mut tsum) = (0.0f32, 0.0f32, 0.0f32);
        for i in 0..d {
            let (a, b) = (l0[r.start + i], l1[r.start + i]);
            let si = usize::from(sides[r.start + i]);
            let sign = if si == 0 { 1.0f32 } else { -1.0 };
            let rho = p.rho[i][si];
            let mult = rho + sign * (a - b);
            for (j, x) in rc.iter_mut().enumerate() {
                *x += mult * p.a[i * p.n + j];
            }
            c0path += p.c0s[i][si];
            c0abs += p.c0s[i][si].abs();
            lag += a * p.b0[i][si] + b * p.b1[i][si];
            lagabs += a * p.b0[i][si].abs() + b * p.b1[i][si].abs();
            tsum += (rho.abs() + a + b) * p.row_abs[i];
        }
        let sumabs: f32 = rc.iter().map(|x| x.abs()).sum();
        let v32 = c0path - sumabs - lag;
        let s32 = c0abs + sumabs + p.abs_d + tsum + lagabs;
        assert_eq!(
            (v32.to_bits(), s32.to_bits()),
            (v.to_bits(), s.to_bits()),
            "node {k}"
        );
        assert!(v.is_finite() && s.is_finite() && s > 0.0);
    }
    assert!(p.validate().is_ok());
    assert!(p.n > FD_SUM_TILE, "the fixture must span two column tiles");
}

#[test]
fn search_constants_round_like_the_cpu_search() {
    let k = plan::constants();
    assert_eq!(k[0].to_bits(), 0x3089_705F, "tol = (1e-9 f64) as f32");
    assert_eq!(k[1].to_bits(), 0x2B8C_BCCC, "floor_e = (1e-12 f64) as f32");
    assert_eq!(k[2].to_bits(), 0x0DA2_4260, "lw_min = (1e-30 f64) as f32");
    assert_eq!(k[3], 256.0);
    assert_eq!(k[4], 1e6);
    let u = f64::from(f32::EPSILON) / 2.0;
    assert_eq!(plan::rel_threshold(1), (4.0 * (u / (1.0 - u))) as f32);
    assert!(plan::rel_threshold(512) > plan::rel_threshold(8));
}

// ---------------------------------------------------------------------------
//  Device tests (gated on an advertised device; SKIP is VOID)
// ---------------------------------------------------------------------------

/// Run `body` on a backend whose device opened and passed its self-test when
/// this host advertises a device (then any refusal FAILS the test); print
/// `SKIP=VOID` otherwise.
pub(super) fn with_search_device(test: &str, body: impl FnOnce(&CudaSearchBackend)) {
    let _serial = crate::CUDA_HARDWARE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let library = crate::CudaGemmEngine::runtime_driver_library_present();
    let count = if library {
        crate::CudaGemmEngine::runtime_device_count()
    } else {
        None
    };
    match crate::cuda_test_admission(library, count) {
        crate::CudaTestAdmission::Capable => {
            let backend = CudaSearchBackend::new();
            if let Err(why) = backend.device() {
                panic!(
                    "{test}: this host advertises a CUDA device ({count:?}) but the search \
                     backend refused it: {why}"
                );
            }
            body(&backend);
        }
        unavailable => eprintln!(
            "[cuda-search-test] {test}: SKIP=VOID ({unavailable:?}, libcuda={library}, \
             devices={count:?}) — no CUDA device on this host; this run proved NOTHING about \
             the kernels"
        ),
    }
}

fn run(backend: &CudaSearchBackend, p: &FdSearchProblem, c: &LevelChunk<'_>) -> LevelOut {
    let h = backend.upload_fd(p, &BIG).expect("upload");
    let mut out = LevelOut::default();
    let _ = backend.fd_level(&h, c, &mut out).expect("fd_level");
    out
}

/// Every output of a level, bit for bit (NaN canonicalised; the sign of a
/// zero is not an output).
fn digest(out: &LevelOut) -> Vec<u32> {
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
        for pt in 0..FD_POINTS {
            d.push(canon(v.value(k, pt)));
            d.push(canon(v.scale(k, pt)));
        }
        for (a, b) in [v.clamped(k), v.far(k), v.next(k)] {
            d.extend(a.iter().chain(b).map(|&x| canon(x)));
        }
    }
    d
}

#[test]
fn cuda_fd_device_self_test_passes_and_probes_caps() {
    with_search_device(
        "cuda_fd_device_self_test_passes_and_probes_caps",
        |backend| {
            assert!(backend.probed() && backend.unavailable().is_none());
            match backend.caps() {
                BackendCaps::Cuda { sm, mem_bytes, .. } => {
                    assert!(sm >= 80, "sm {sm}");
                    assert!(mem_bytes > 0, "a probed device leases some memory");
                }
                other => panic!("{other:?}"),
            }
        },
    );
}

#[test]
fn cuda_fd_warm_is_exact_on_dyadic_levels() {
    with_search_device("cuda_fd_warm_is_exact_on_dyadic_levels", |backend| {
        let mut checked = 0usize;
        for (n, depth, count, seed) in [
            (300usize, 5usize, 37usize, 1u64),
            (257, 3, 9, 2),
            (40, 4, 8, 3),
        ] {
            let p = plan::kat_problem(n, depth, seed);
            for d in 1..=depth {
                let (sides, l0, l1) = plan::kat_level(count, d, seed ^ d as u64);
                let out = run(backend, &p, &chunk(d, &sides, &l0, &l1));
                let v = out.hint().into_ingest();
                for k in 0..count {
                    let r = k * d..(k + 1) * d;
                    let (want_v, want_s) =
                        plan::exact_warm(&p, &sides[r.clone()], &l0[r.clone()], &l1[r]);
                    assert_eq!(
                        (v.value(k, FD_WARM).to_bits(), v.scale(k, FD_WARM).to_bits()),
                        (want_v.to_bits(), want_s.to_bits()),
                        "n={n} d={d} node {k}"
                    );
                    checked += 1;
                }
            }
        }
        assert!(checked > 100, "VOID: {checked}");
    });
}

#[test]
fn cuda_fd_outputs_do_not_depend_on_the_slab_or_the_chunk_cut() {
    with_search_device(
        "cuda_fd_outputs_do_not_depend_on_the_slab_or_the_chunk_cut",
        |backend| {
            let (n, depth) = (777usize, 20usize);
            let count = 7 * FD_NB + 5;
            // Random-valued (not dyadic): rounding happens everywhere, so a
            // reduction whose order moved with the geometry would show.
            let mut p = plan::kat_problem(n, depth, 31);
            for (k, a) in p.a.iter_mut().enumerate() {
                *a *= 1.0 + (k % 7) as f32 * 1e-3;
            }
            p.row_abs = (0..depth)
                .map(|i| p.a[i * n..(i + 1) * n].iter().map(|v| v.abs()).sum())
                .collect();
            let (sides, mut l0, l1) = plan::kat_level(count, depth, 31);
            for (k, m) in l0.iter_mut().enumerate() {
                *m *= 1.0 + (k % 5) as f32 * 3e-4;
            }
            let base = digest(&run(backend, &p, &chunk(depth, &sides, &l0, &l1)));
            for slab in [1usize, 3, 8, 13, 64] {
                let small = CudaSearchBackend {
                    slab_override: Some(slab),
                    ..CudaSearchBackend::new()
                };
                assert_eq!(
                    base,
                    digest(&run(&small, &p, &chunk(depth, &sides, &l0, &l1))),
                    "slabs of {slab} nodes changed the search"
                );
            }
            let h = backend.upload_fd(&p, &BIG).expect("upload");
            let mut joined = Vec::new();
            let mut lo = 0usize;
            for cut in [1usize, 9, 30, count] {
                let r = lo * depth..cut * depth;
                let mut out = LevelOut::default();
                let _ = backend
                    .fd_level(
                        &h,
                        &chunk(depth, &sides[r.clone()], &l0[r.clone()], &l1[r]),
                        &mut out,
                    )
                    .expect("fd_level");
                joined.extend(digest(&out));
                lo = cut;
            }
            assert_eq!(base, joined, "cutting the level changed the search");
        },
    );
}

#[test]
fn cuda_fd_sanitizes_dirty_multipliers_and_refuses_overflowing_nodes() {
    with_search_device(
        "cuda_fd_sanitizes_dirty_multipliers_and_refuses_overflowing_nodes",
        |backend| {
            let depth = 5;
            let p = plan::kat_problem(90, depth, 51);
            let (sides, clean0, clean1) = plan::kat_level(2 * FD_NB, depth, 51);
            let (mut dirty0, mut dirty1) = (clean0.clone(), clean1.clone());
            let (mut zero0, mut zero1) = (clean0.clone(), clean1.clone());
            for (k, bad) in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -3.0, -0.0]
                .into_iter()
                .enumerate()
            {
                dirty0[k * depth + k % depth] = bad;
                zero0[k * depth + k % depth] = 0.0;
                dirty1[(k + 5) * depth] = bad;
                zero1[(k + 5) * depth] = 0.0;
            }
            let a = run(backend, &p, &chunk(depth, &sides, &dirty0, &dirty1));
            let b = run(backend, &p, &chunk(depth, &sides, &zero0, &zero1));
            assert_eq!(
                digest(&a),
                digest(&b),
                "a non-finite or negative multiplier must act as 0"
            );
            // A node whose multipliers overflow the value is REFUSED: no value,
            // and every proposal is its own (sanitised) point.
            // `(|rho| + a + b) * row_abs` overflows f32 on every row with a
            // nonzero coefficient, so the node's S is +inf.
            let (mut huge0, huge1) = (clean0, clean1);
            for m in &mut huge0[..depth] {
                *m = 3.0e38;
            }
            assert!(
                p.row_abs.iter().any(|&r| r >= 2.0),
                "VOID: no row overflows"
            );
            let h = run(backend, &p, &chunk(depth, &sides, &huge0, &huge1));
            let v = h.hint().into_ingest();
            assert!(
                (0..FD_POINTS).all(|pt| v.value(0, pt).is_nan()),
                "an overflowing node has no value"
            );
            for (x, y) in [v.clamped(0), v.far(0), v.next(0)] {
                assert_eq!(x, &huge0[..depth]);
                assert_eq!(y, &huge1[..depth]);
            }
            for k in 1..v.count() {
                assert!(v.value(k, FD_WARM).is_finite(), "node {k} was not refused");
                for (x, y) in [v.clamped(k), v.far(k), v.next(k)] {
                    assert!(x.iter().chain(y).all(|m| m.is_finite() && *m >= 0.0));
                }
            }
        },
    );
}

#[test]
fn cuda_fd_restores_the_callers_current_context() {
    with_search_device("cuda_fd_restores_the_callers_current_context", |backend| {
        let current = || cudarc::driver::result::ctx::get_current().ok().flatten();
        let p = small_problem();
        let (sides, l0, l1) = plan::kat_level(FD_NB + 1, 3, 4);
        // No context current on the caller's thread.
        let before = current();
        let out = run(backend, &p, &chunk(3, &sides, &l0, &l1));
        assert_eq!(
            before,
            current(),
            "the search left its own context current on the caller's thread"
        );
        assert!(out.hint().into_ingest().well_formed());
        // A FOREIGN context current — the production case: the verdict-feeding
        // engine's primary context — through a search on an opened device AND
        // through a first upload that opens a device (`cuCtxCreate` pushes the
        // new context onto this thread's stack).
        let primary = cudarc::driver::CudaContext::new(0).expect("primary context");
        let foreign = Some(primary.cu_ctx());
        assert_eq!(
            current(),
            foreign,
            "VOID: the primary context is not current"
        );
        let _ = run(backend, &p, &chunk(3, &sides, &l0, &l1));
        assert_eq!(
            current(),
            foreign,
            "a search did not restore the caller's context"
        );
        let fresh = CudaSearchBackend::new();
        let _ = run(&fresh, &p, &chunk(3, &sides, &l0, &l1));
        assert!(fresh.probed() && fresh.unavailable().is_none());
        assert_eq!(
            current(),
            foreign,
            "opening a device (and its first upload and search) did not restore the caller's \
             context"
        );
    });
}

#[test]
fn cuda_fd_refuses_late_malformed_and_deeper_chunks_on_a_live_upload() {
    with_search_device(
        "cuda_fd_refuses_late_malformed_and_deeper_chunks_on_a_live_upload",
        |backend| {
            let p = small_problem();
            let h = backend.upload_fd(&p, &BIG).expect("upload");
            assert_eq!(h.backend(), "cuda");
            let (sides, l0, l1) = plan::kat_level(5, 3, 41);
            let ok = chunk(3, &sides, &l0, &l1);
            let mut out = LevelOut::default();
            let stats = backend
                .fd_level(&h, &ok, &mut out)
                .expect("a valid chunk searches")
                .into_hint()
                .into_ingest();
            assert_eq!(stats.nodes, 5);
            assert!(stats.evals >= 5 && stats.refused == 0, "{stats:?}");
            assert!(matches!(
                backend.fd_level(&h, &LevelChunk { count: 4, ..ok }, &mut out),
                Err(Refusal::Invalid(_))
            ));
            let (s7, a7, b7) = plan::kat_level(2, p.depth + 1, 1);
            assert!(
                matches!(
                    backend.fd_level(&h, &chunk(p.depth + 1, &s7, &a7, &b7), &mut out),
                    Err(Refusal::Invalid(_))
                ),
                "a depth beyond the upload must refuse"
            );
            let late = LevelChunk {
                deadline: Some(
                    Instant::now()
                        .checked_sub(Duration::from_millis(1))
                        .unwrap_or_else(Instant::now),
                ),
                ..ok
            };
            assert!(matches!(
                backend.fd_level(&h, &late, &mut out),
                Err(Refusal::Deadline)
            ));
            let empty = LevelChunk {
                count: 0,
                sides: &[],
                l0: &[],
                l1: &[],
                ..ok
            };
            let stats = backend
                .fd_level(&h, &empty, &mut out)
                .expect("an empty chunk is fine")
                .into_hint()
                .into_ingest();
            assert_eq!(stats.nodes, 0);
            assert!(
                backend.unavailable().is_none(),
                "no refusal above is a fault"
            );
        },
    );
}
