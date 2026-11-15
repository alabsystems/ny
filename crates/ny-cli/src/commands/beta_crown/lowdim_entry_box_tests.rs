// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #m8-cgan-leaf-batch (dist_shift root cause, ENGINE_PROGRAM M8 brief 3.3):
//! the lowdim lane's entry box is the VNN-LIB loader's `f32` box, and
//! `VnnLibSpec::split_input_bounds_f32` widens EVERY endpoint outward by one
//! `f32` ulp (#2658, deliberate outward rounding, kept) and snaps a subnormal
//! endpoint outward to `0` / `+-f32::MIN_POSITIVE`. A declared point
//! `X_j == v` therefore arrives as `[next_down(v), next_up(v)]`: on
//! dist_shift `index2734` all 784 declared pixels looked free and the lane
//! counted `d_free=792 pinned=0` (W10B `LDB_d2734_dist_r1`). This pins that
//! the widening is ulp-scale (so no coordinate arrives genuinely wide, which
//! would be a spec-translation finding) and that the lane's narrow-pin
//! classification (`ny_propagate::lowdim::pin_kind`) takes every widened
//! point back as a pin while every declared interval stays free.

use ny_onnx::vnnlib::VnnLibSpec;
use ny_propagate::lowdim::{pin_counts, pin_kind, pin_tolerance, PinKind};

/// Spacing of `f32` values just above `|v|` (below it at `f32::MAX`).
fn f32_ulp(v: f32) -> f64 {
    let a = v.abs();
    if a == f32::MAX {
        return f64::from(a) - f64::from(f32::from_bits(a.to_bits() - 1));
    }
    f64::from(f32::from_bits(a.to_bits() + 1)) - f64::from(a)
}

/// The declared literals of dist_shift_2023 `index2734_delta0.13.vnnlib`
/// (pixels: exact `f32` values printed as f64 plus `0.0`/`1.0`), a binade
/// edge, and decimals that are NOT `f32` values.
const DECLARED_POINTS: [f64; 12] = [
    0.0,
    0.109_803_922_474_384_31,
    0.219_607_844_948_768_62,
    0.541_176_497_936_248_8,
    0.992_156_863_212_585_4,
    0.996_078_431_606_292_7,
    1.0,
    0.5,
    0.1,
    -52.088_510_761_260_984,
    1e-40,
    -3.0e-39,
];

/// The declared latent intervals of the same file.
const DECLARED_INTERVALS: [(f64, f64); 4] = [
    (-55.467_694_988_250_734, -52.088_510_761_260_984),
    (-76.635_776_157_379_15, -71.334_820_156_097_41),
    (-53.198_668_276_071_55, -52.765_359_128_713_605),
    (-59.586_454_544_067_38, -40.223_962_631_225_59),
];

#[test]
fn dist_shift_entry_box_widening_is_ulp_scale() {
    let mut spec = VnnLibSpec::new();
    spec.input_bounds = DECLARED_POINTS
        .iter()
        .map(|&v| (v, v))
        .chain(DECLARED_INTERVALS.iter().copied())
        .collect();
    let (lower, upper) = spec.split_input_bounds_f32();
    assert_eq!(
        lower.len(),
        DECLARED_POINTS.len() + DECLARED_INTERVALS.len()
    );

    for (j, &v) in DECLARED_POINTS.iter().enumerate() {
        let (lo, hi) = (f64::from(lower[j]), f64::from(upper[j]));
        // Outward (sound) and PRESENT: the widening is what made every
        // declared point look free to the historical `hi > lo` count.
        assert!(
            lo <= v && v <= hi,
            "X_{j} = {v:e}: [{lo:e}, {hi:e}] is not outward"
        );
        assert!(
            lo < hi,
            "X_{j} = {v:e}: the loader no longer widens a declared point"
        );
        // Ulp-scale: at most one f32 ulp of the rounded value on each side,
        // or the outward subnormal snap around zero.
        #[allow(clippy::cast_possible_truncation)]
        let v32 = v as f32;
        let bound = 2.0 * f32_ulp(v32).max(f64::from(f32::MIN_POSITIVE));
        assert!(
            hi - lo <= bound,
            "X_{j} = {v:e}: width {:e} exceeds two f32 ulps ({bound:e}); a declared point \
             arrived genuinely wide (spec-translation finding, not an M8 fix)",
            hi - lo
        );
        assert!(hi - lo <= pin_tolerance(lo, hi));
        assert_eq!(
            pin_kind(lo, hi),
            PinKind::Narrow,
            "X_{j} = {v:e}: [{lo:e}, {hi:e}] is not taken back as a pin"
        );
    }
    for (k, &(l, h)) in DECLARED_INTERVALS.iter().enumerate() {
        let j = DECLARED_POINTS.len() + k;
        let (lo, hi) = (f64::from(lower[j]), f64::from(upper[j]));
        assert!(lo <= l && h <= hi, "interval {k} is not outward");
        assert_eq!(
            pin_kind(lo, hi),
            PinKind::Free,
            "interval {k} was classified as a pin"
        );
    }

    // The full dist_shift shape: 784 declared pixels (k / 255 as the files
    // print them) and 8 latent intervals -> 784 pins, 8 free coordinates.
    let mut full = VnnLibSpec::new();
    full.input_bounds = (0..784)
        .map(|k| {
            #[allow(clippy::cast_precision_loss)]
            let v = f64::from((k % 256) as f32 / 255.0);
            let printed: f64 = format!("{v}").parse().expect("round-trips");
            (printed, printed)
        })
        .chain((0..8).map(|k| DECLARED_INTERVALS[k % DECLARED_INTERVALS.len()]))
        .collect();
    let (lo32, hi32) = full.split_input_bounds_f32();
    let lo: Vec<f64> = lo32.iter().map(|&v| f64::from(v)).collect();
    let hi: Vec<f64> = hi32.iter().map(|&v| f64::from(v)).collect();
    let historical_free = lo.iter().zip(&hi).filter(|(l, h)| h > l).count();
    assert_eq!(
        historical_free, 792,
        "the defect's own count (VOID if it moved)"
    );
    let counts = pin_counts(&lo, &hi);
    assert_eq!(counts.free, 8, "{counts:?}");
    assert_eq!(counts.exact + counts.narrow, 784, "{counts:?}");
}

// ------------------------------------------------ review(m8b): real-model probe

/// The dist_shift_2023 corpus: this checkout, or (from a git worktree, which
/// has no `benchmarks/`) the main checkout the worktree belongs to. A missing
/// corpus is VOID and FAILS the probe.
#[cfg(feature = "external-vnncomp")]
fn dist_shift_dir() -> std::path::PathBuf {
    const CORPUS: &str = "benchmarks/vnncomp2026/benchmarks/dist_shift_2023/2.0";
    let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let direct = workspace.join(CORPUS);
    if direct.is_dir() {
        return direct;
    }
    if let Ok(text) = std::fs::read_to_string(workspace.join(".git")) {
        if let Some(gitdir) = text.trim().strip_prefix("gitdir:") {
            let gitdir = std::path::PathBuf::from(gitdir.trim());
            if let Some(main_git) = gitdir
                .ancestors()
                .find(|path| path.file_name().is_some_and(|name| name == ".git"))
            {
                if let Some(main) = main_git.parent() {
                    let candidate = main.join(CORPUS);
                    if candidate.is_dir() {
                        return candidate;
                    }
                }
            }
        }
    }
    panic!(
        "VOID: the dist_shift_2023 corpus ({CORPUS}) is missing under {} and under the main \
         checkout",
        workspace.display()
    );
}

/// review(m8b) real-model probe (the builder's open gap: "never run on the
/// real mnist_concat model"). The W10B row `LDB_d2734_dist_r1` declined with
/// `d_free=792` and `layer_4 produces [1, 28, 28]`. Load the model and the
/// VNN-LIB box the way the sequential entry does (`load_onnx` ->
/// `to_propagate_network` -> `GraphNetwork::from_sequential`,
/// `split_input_bounds_f32`) and ask the ARMED admission whether the whole
/// 15-layer conversion lowers and folds to the 8 latent coordinates. Prints
/// the admission facts. It needs the downloaded dist_shift_2023 corpus, so it
/// belongs to the explicit `external-vnncomp` lane (absent from the default
/// lane, a hard failure when selected without the corpus; the corpus resolves
/// to the main tree from a worktree): `targo --unverified test -p ny-cli --bin
/// ny --features external-vnncomp dist_shift_real_model_admission_probe`.
#[cfg(feature = "external-vnncomp")]
#[test]
fn dist_shift_real_model_admission_probe() {
    use ny_propagate::lowdim::{admit_with, LowDimConfig};
    use ny_tensor::BoundedTensor;

    let dir = dist_shift_dir();
    let network = ny_onnx::load_onnx(dir.join("onnx/mnist_concat.onnx"))
        .expect("load mnist_concat")
        .to_propagate_network()
        .expect("sequential network");
    let graph = ny_propagate::GraphNetwork::from_sequential(&network).expect("graph");
    let spec = ny_onnx::vnnlib::load_vnnlib(dir.join("vnnlib/index2734_delta0.13.vnnlib"))
        .expect("load vnnlib");
    let (lower, upper) = spec.split_input_bounds_f32();
    assert_eq!(lower.len(), 792, "dist_shift input width");
    let armed = LowDimConfig {
        varying_dims: true,
        ..LowDimConfig::default()
    };
    let mut admitted = 0usize;
    for shape in [vec![1usize, 792], vec![792usize]] {
        let input = BoundedTensor::new(
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), lower.clone()).expect("lo"),
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), upper.clone()).expect("hi"),
        )
        .expect("input box");
        let unarmed = admit_with(&graph, &input, &LowDimConfig::default());
        assert!(unarmed.is_err(), "shape {shape:?}: admitted UNARMED");
        match admit_with(&graph, &input, &armed) {
            Ok(adm) => {
                eprintln!(
                    "[m8b-probe] shape={shape:?} admitted=1 input_dim={} sigmoid_ops={} \
                     output_width={}",
                    adm.input_dim(),
                    adm.sigmoid_ops(),
                    adm.output_width()
                );
                assert_eq!(adm.input_dim(), 8, "shape {shape:?}: free coordinates");
                admitted += 1;
            }
            Err(reason) => eprintln!("[m8b-probe] shape={shape:?} admitted=0 reason={reason}"),
        }
    }
    assert!(
        admitted > 0,
        "the armed lane admits the real mnist_concat under neither input shape (see the \
         [m8b-probe] lines)"
    );
}
