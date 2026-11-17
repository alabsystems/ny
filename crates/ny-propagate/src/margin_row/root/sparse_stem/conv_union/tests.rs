// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::margin_row::net::compile_conv;
use crate::margin_row::packed_root_engine::{test_support::Scalar, with_test_engine};
use num_rational::BigRational as Q;
use num_traits::{Signed, Zero};
use ny_core::packed_root_conv::{Engine, Limits, Output, Request};
use std::collections::BTreeSet;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

fn near_dense_source() -> (PackedRows<f32>, ConvOp) {
    let mut source = PackedRows {
        n_in: 100,
        p: 2,
        coordinates: Array2::from_shape_vec((2, 45), (0..90).collect()).unwrap(),
        m: Array2::zeros((2, 46)),
        d: Array2::zeros((2, 46)),
    };
    for row in 0..2 {
        for slot in 0..45 {
            source.m[[row, slot]] = (row * 45 + slot + 1) as f32 / 16.0;
            source.d[[row, slot]] = if (row + slot) % 2 == 0 { 1.0 } else { -1.0 }
                * (row * 50 + slot + 3) as f32
                / 32.0;
        }
    }
    source.m[[0, 45]] = -0.5;
    source.m[[1, 45]] = 1.25;
    source.d[[0, 45]] = -0.125;
    source.d[[1, 45]] = 0.125;
    let weights = (0..16)
        .flat_map(|oc| {
            let w = if oc % 2 == 0 { 1.0 } else { -1.0 } * 2f64.powi(oc - 8);
            [w, -2.0 * w]
        })
        .collect();
    (source, consumer(1, 16, 1, 2, 1, 2, 0, weights))
}

fn near_dense_oracle(engine: Arc<dyn Engine>) {
    let (source, c) = near_dense_source();
    let before = (source.m.clone(), source.d.clone());
    struct Borrowed {
        engine: Arc<dyn Engine>,
        m: usize,
        d: usize,
    }
    impl Engine for Borrowed {
        fn contract(&self, r: &Request<'_>, limits: Limits) -> Result<Option<Output>> {
            assert_eq!(r.m.as_ptr() as usize, self.m);
            assert_eq!(r.d.as_ptr() as usize, self.d);
            assert_eq!((r.source_stride, r.output_stride, r.m.len()), (46, 91, 92));
            self.engine.contract(r, limits)
        }
    }
    let engine = Arc::new(Borrowed {
        engine,
        m: source.m.as_ptr() as usize,
        d: source.d.as_ptr() as usize,
    });
    // The compressed plan fits, while the historical scalar map does not.
    assert_eq!(16 * 101 * 8, 12_928);
    assert_eq!(16 * 91 * 8 + 90 * 8 + 2 * 45 * 8, 13_088);
    assert_eq!(16 * 91 * 8 + 90 * 8 + 3 * 8 + 2 * 16, 12_424);
    with_test_engine(engine, || {
        let plan = source.plan_conv(&c, None).unwrap().unwrap();
        assert!(!plan.cpu_eligible);
        assert_eq!(plan.runs.len(), 2);
        let output = parity(&source, &c);
        assert_eq!(output.m.dim(), (16, 91));
        let (m, d) = output.materialize();
        for oc in 0..16 {
            let w = if oc % 2 == 0 { 1.0 } else { -1.0 } * 2f32.powi(oc as i32 - 8);
            for i in 0..=100 {
                let (em, ed) = if i < 90 {
                    let scale = if i < 45 { 1.0 } else { 2.0 };
                    let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                    let mag = (i + if i < 45 { 3 } else { 8 }) as f32 / 32.0;
                    (
                        w * if i < 45 { 1.0 } else { -2.0 } * (i + 1) as f32 / 16.0,
                        w.abs() * scale * sign * mag,
                    )
                } else if i == 100 {
                    (-3.0 * w, 0.125 * w.abs())
                } else {
                    (0.0, 0.0)
                };
                assert_eq!(m[[oc, i]].to_bits(), em.to_bits(), "M {oc}/{i}");
                assert_eq!(d[[oc, i]].to_bits(), ed.to_bits(), "D {oc}/{i}");
            }
        }
    });
    assert_bits(&source.m, &before.0);
    assert_bits(&source.d, &before.1);
}

#[test]
fn near_dense_gpu_admission_preserves_original_frame_and_exact_dyadics() {
    near_dense_oracle(Arc::new(Scalar::default()));
}

#[cfg(feature = "gpu-tests-cuda")]
#[test]
fn near_dense_cuda_admission_matches_independent_exact_bits() {
    let engine = ny_cuda::packed_root::PackedRootEngine::new(0)
        .expect("required near-dense CUDA qualification")
        .expect("required native CUDA architecture");
    near_dense_oracle(Arc::new(engine));
}

#[test]
fn near_dense_gpu_refusal_preserves_dense_fallback_and_errors() {
    let (source, c) = near_dense_source();
    assert!(!crate::margin_row::packed_root_engine::ready());
    assert!(source.contract_packed(&c, None).unwrap().is_none());
    for reply in [
        Reply::Decline,
        Reply::Deadline,
        Reply::Short,
        Reply::Nonfinite,
        Reply::ExpireThenDecline,
    ] {
        let engine = Arc::new(Scripted {
            reply,
            calls: AtomicUsize::new(0),
        });
        with_test_engine(engine.clone(), || {
            assert!(crate::margin_row::packed_root_engine::ready());
            {
                let _cpu = crate::sound_f64_gemm::CpuOnlyF64Guard::new();
                assert!(!crate::margin_row::packed_root_engine::ready());
                assert!(source.contract_packed(&c, None).unwrap().is_none());
            }
            assert_eq!(engine.calls.load(Ordering::SeqCst), 0);
            let deadline = matches!(reply, Reply::ExpireThenDecline)
                .then(|| Instant::now() + std::time::Duration::from_millis(200));
            let result = source.contract_packed(&c, deadline);
            match reply {
                Reply::Decline => assert!(result.unwrap().is_none()),
                Reply::Deadline | Reply::ExpireThenDecline => {
                    assert!(matches!(result, Err(NyError::DeadlineExceeded(_))))
                }
                Reply::Nonfinite => {
                    assert!(matches!(result, Err(NyError::NumericalInstability(_))))
                }
                Reply::Short => assert!(matches!(result, Err(NyError::InternalError(_)))),
            }
        });
        assert_eq!(engine.calls.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn packed_accelerator_matches_cpu_and_moves_both_lanes_across_consumers() {
    let source = padded_source();
    let before = (source.m.clone(), source.d.clone());
    let engine = Arc::new(Scalar::default());
    for c in [
        consumer(2, 2, 2, 3, 3, 3, 3, vec![-0.75; 36]),
        consumer(2, 3, 2, 3, 1, 1, 0, vec![0.5; 6]),
    ] {
        let cpu = source.contract_packed(&c, None).unwrap().unwrap();
        let output = with_test_engine(engine.clone(), || parity(&source, &c));
        assert_bits(&output.m, &cpu.m);
        assert_bits(&output.d, &cpu.d);
        assert_eq!(output.coordinates, cpu.coordinates);
        assert_eq!(
            *engine.pointers.lock().unwrap(),
            (output.m.as_ptr() as usize, output.d.as_ptr() as usize)
        );
    }
    assert_eq!(engine.calls.load(Ordering::SeqCst), 2);
    assert_bits(&source.m, &before.0);
    assert_bits(&source.d, &before.1);
}

#[derive(Clone, Copy)]
enum Reply {
    Decline,
    Deadline,
    Short,
    Nonfinite,
    ExpireThenDecline,
}
struct Scripted {
    reply: Reply,
    calls: AtomicUsize,
}
impl Engine for Scripted {
    fn contract(&self, r: &Request<'_>, limits: Limits) -> Result<Option<Output>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        match self.reply {
            Reply::Decline => Ok(None),
            Reply::Deadline => Err(NyError::DeadlineExceeded("injected engine deadline".into())),
            Reply::ExpireThenDecline => {
                let end = limits.deadline.unwrap();
                while Instant::now() < end {
                    std::thread::sleep(end.saturating_duration_since(Instant::now()));
                }
                Ok(None)
            }
            Reply::Short => Ok(Some(Output {
                m: vec![],
                d: vec![],
            })),
            Reply::Nonfinite => Ok(Some(Output {
                m: vec![f32::INFINITY; r.output_channels * r.output_spatial * r.output_stride],
                d: vec![0.0; r.output_channels * r.output_spatial * r.output_stride],
            })),
        }
    }
}

#[test]
fn packed_accelerator_decline_falls_back_but_failures_propagate() {
    let source = padded_source();
    let c = consumer(2, 2, 2, 3, 1, 1, 0, vec![0.5; 4]);
    for reply in [
        Reply::Decline,
        Reply::Deadline,
        Reply::Short,
        Reply::Nonfinite,
    ] {
        let engine = Arc::new(Scripted {
            reply,
            calls: AtomicUsize::new(0),
        });
        if matches!(reply, Reply::Decline) {
            with_test_engine(engine.clone(), || parity(&source, &c));
        } else {
            let result = with_test_engine(engine.clone(), || source.contract_packed(&c, None));
            match reply {
                Reply::Deadline => assert!(matches!(result, Err(NyError::DeadlineExceeded(_)))),
                Reply::Nonfinite => {
                    assert!(matches!(result, Err(NyError::NumericalInstability(_))))
                }
                _ => assert!(result.is_err()),
            }
        }
        assert_eq!(engine.calls.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn packed_accelerator_custody_and_admission_exclusions_never_dispatch() {
    let source = padded_source();
    let c = consumer(2, 2, 2, 3, 1, 1, 0, vec![0.5; 4]);
    let tiny = consumer(2, 2, 2, 3, 1, 1, 0, vec![2.0_f64.powi(-150); 4]);
    let engine = Arc::new(Scalar::default());
    with_test_engine(engine.clone(), || {
        {
            let _cpu = crate::sound_f64_gemm::CpuOnlyF64Guard::new();
            parity(&source, &c);
        }
        assert!(source.contract_packed(&c, Some(Instant::now())).is_err());
        assert!(source.contract_packed(&tiny, None).unwrap().is_none());
        let wide = PackedRows {
            n_in: source.n_in,
            p: source.p,
            coordinates: source.coordinates.clone(),
            m: source.m.mapv(f64::from),
            d: source.d.mapv(f64::from),
        };
        assert!(wide.contract_packed(&c, None).unwrap().is_none());
    });
    assert_eq!(engine.calls.load(Ordering::SeqCst), 0);
}

#[test]
fn packed_accelerator_expiry_on_decline_cannot_start_cpu_fallback() {
    let source = padded_source();
    let c = consumer(2, 2, 2, 3, 1, 1, 0, vec![0.5; 4]);
    let engine = Arc::new(Scripted {
        reply: Reply::ExpireThenDecline,
        calls: AtomicUsize::new(0),
    });
    let deadline = Instant::now() + std::time::Duration::from_millis(200);
    let result = with_test_engine(engine.clone(), || {
        source.contract_packed(&c, Some(deadline))
    });
    assert!(matches!(result, Err(NyError::DeadlineExceeded(_))));
    assert_eq!(engine.calls.load(Ordering::SeqCst), 1);
}

fn consumer(
    ci: usize,
    co: usize,
    ih: usize,
    iw: usize,
    kh: usize,
    kw: usize,
    pad: usize,
    weights: Vec<f64>,
) -> ConvOp {
    compile_conv(
        2,
        &weights,
        &vec![0.0; co],
        &vec![0.0; co],
        0.0,
        (co, ci, kh, kw),
        (1, 1),
        (pad, pad, pad, pad),
        (ci, ih, iw),
        (co, ih + 2 * pad + 1 - kh, iw + 2 * pad + 1 - kw),
    )
}

fn assert_bits(a: &Array2<f32>, b: &Array2<f32>) {
    assert_eq!(a.dim(), b.dim());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "coefficient {i}");
    }
}

fn parity(source: &PackedRows<f32>, c: &ConvOp) -> PackedRows<f32> {
    let mut m = Array2::<f32>::zeros((c.oshape.0 * c.oshape.1 * c.oshape.2, source.n_in + 1));
    let mut d = m.clone();
    source.contract(c, &mut m, &mut d, None).unwrap();
    let out = source
        .contract_packed(c, None)
        .unwrap()
        .expect("compact route admitted");
    let (pm, pd) = out.materialize();
    assert_bits(&pm, &m);
    assert_bits(&pd, &d);
    out
}

fn padded_source() -> PackedRows<f32> {
    let coordinates = Array2::from_shape_vec(
        (6, 3),
        vec![
            1,
            9,
            usize::MAX,
            9,
            17,
            25,
            17,
            25,
            usize::MAX,
            5,
            9,
            13,
            9,
            13,
            29,
            13,
            29,
            usize::MAX,
        ],
    )
    .unwrap();
    let mut m = Array2::<f32>::zeros((12, 4));
    let mut d = m.clone();
    for j in 0..12 {
        for slot in 0..3 {
            if coordinates[[j % 6, slot]] == usize::MAX {
                continue;
            }
            m[[j, slot]] = (j as f32 - slot as f32 - 4.0) * 0.125;
            d[[j, slot]] = (slot as f32 - j as f32 - 1.0) * 0.0625;
        }
        m[[j, 3]] = j as f32 * -0.25;
        d[[j, 3]] = j as f32 * 0.015625;
    }
    m[[0, 0]] = -0.0;
    d[[1, 0]] = f32::from_bits(1);
    m[[8, 1]] = -f32::from_bits(1);
    PackedRows {
        n_in: 257,
        p: 6,
        coordinates,
        m,
        d,
    }
}

#[test]
fn compact_conv_unions_preserve_padded_geometry_channels_and_source() {
    let source = padded_source();
    let before_m = source.m.clone();
    let before_d = source.d.clone();
    let c = consumer(
        2,
        2,
        2,
        3,
        3,
        3,
        3,
        (0..36)
            .map(|i| if i % 3 == 0 { -0.75 } else { 0.5 })
            .collect(),
    );
    let plan = source.plan_conv(&c, None).unwrap().unwrap();
    let (_, oh, ow) = c.oshape;
    let mut padded_outputs = 0;
    let mut widths = BTreeSet::new();
    for oy in 0..oh {
        for ox in 0..ow {
            let sp = oy * ow + ox;
            // Independent spec geometry, without reading c.gather or plan maps.
            let mut expected = BTreeSet::new();
            for ky in 0..3 {
                for kx in 0..3 {
                    let iy = oy as isize + ky - 3;
                    let ix = ox as isize + kx - 3;
                    if !(0..2).contains(&iy) || !(0..3).contains(&ix) {
                        continue;
                    }
                    for &i in source.coordinates.row(iy as usize * 3 + ix as usize).iter() {
                        if i != usize::MAX {
                            expected.insert(i);
                        }
                    }
                }
            }
            let actual: Vec<_> = plan
                .coordinates
                .row(sp)
                .iter()
                .copied()
                .take_while(|&i| i != usize::MAX)
                .collect();
            assert_eq!(actual, expected.into_iter().collect::<Vec<_>>());
            widths.insert(actual.len());
            padded_outputs += usize::from(actual.is_empty());
            assert!(plan
                .coordinates
                .row(sp)
                .iter()
                .skip(actual.len())
                .all(|&i| i == usize::MAX));
            for tap in 0..9 {
                let gi = c.gather[tap * oh * ow + sp];
                let bucket = sp * 9 + tap;
                let runs = &plan.runs[plan.offsets[bucket]..plan.offsets[bucket + 1]];
                let mut expanded = Vec::new();
                let mut previous_end = None;
                for run in runs {
                    assert!(run.len > 0);
                    if let Some(end) = previous_end {
                        assert!(end < run.dest, "runs are sorted, disjoint and maximal");
                    }
                    expanded.extend(run.dest..run.dest + run.len);
                    previous_end = Some(run.dest + run.len);
                }
                let expected_len = if gi == usize::MAX {
                    0
                } else {
                    source
                        .coordinates
                        .row(gi)
                        .iter()
                        .take_while(|&&i| i != usize::MAX)
                        .count()
                };
                assert_eq!(
                    expanded.len(),
                    expected_len,
                    "every valid source slot covered once"
                );
                for slot in 0..3 {
                    if gi == usize::MAX || source.coordinates[[gi, slot]] == usize::MAX {
                        assert!(expanded.get(slot).is_none());
                    } else {
                        assert_eq!(actual[expanded[slot]], source.coordinates[[gi, slot]]);
                    }
                }
            }
        }
    }
    assert!(padded_outputs > 0 && widths.len() >= 3);
    let output = parity(&source, &c);
    assert!(output.allocated_bytes() < output.m.nrows() * (source.n_in + 1) * 8);
    let next = consumer(
        2,
        2,
        oh,
        ow,
        3,
        3,
        1,
        (0..36)
            .map(|i| if i % 2 == 0 { -0.25 } else { 0.5 })
            .collect(),
    );
    let second = parity(&output, &next);
    assert_eq!(
        second.n_in, source.n_in,
        "original frame survives two compact Convs"
    );
    assert_bits(&source.m, &before_m);
    assert_bits(&source.d, &before_d);
    // A later consumer borrows exactly the same source after the first output
    // exists; this checks kernel borrowing, not root's future lifetime routing.
    let shortcut = consumer(2, 1, 2, 3, 1, 1, 0, vec![1.0, -1.0]);
    parity(&source, &shortcut);
    assert_bits(&source.m, &before_m);
    assert_bits(&source.d, &before_d);
}

#[test]
fn compact_conv_runs_cross_coordinate_and_channel_tiles_without_fma() {
    let slots = 530;
    let coordinates =
        Array2::from_shape_vec((2, slots), (0..slots).chain(8..slots + 8).collect()).unwrap();
    let mut source = PackedRows {
        n_in: 4096,
        p: 2,
        coordinates,
        m: Array2::<f32>::zeros((4, slots + 1)),
        d: Array2::<f32>::zeros((4, slots + 1)),
    };
    let eps = 2f32.powi(-23);
    source.m.row_mut(0).fill(-1.0);
    source.m.row_mut(1).fill(1.0 - eps);
    source.d.row_mut(0).fill(-0.25);
    source.d.row_mut(1).fill(0.125);
    // The second input channel has nonzero values but exactly zero weights.
    // Skipping those weights must not reset any preceding accumulation.
    source.m.row_mut(2).fill(7.0);
    source.m.row_mut(3).fill(-9.0);
    let weights: Vec<f64> = (0..17)
        .flat_map(|_| [1.0, f64::from(1.0 + eps), 0.0, -0.0])
        .collect();
    let c = consumer(2, 17, 1, 2, 1, 2, 0, weights);
    let plan = source.plan_conv(&c, None).unwrap().unwrap();
    assert_eq!(plan.runs.len(), 2);
    assert!(plan.runs.iter().all(|r| r.len == slots));
    let output = parity(&source, &c);
    assert_eq!(output.m.dim(), (17, 539));
    for oc in 0..17 {
        for i in [8, 255, 256, 511, 512, 529, 538] {
            assert_eq!(
                output.m[[oc, i]].to_bits(),
                0.0f32.to_bits(),
                "channel {oc}, column {i}"
            );
        }
    }
    let fused = (1.0 + eps).mul_add(1.0 - eps, -1.0);
    assert_ne!(
        fused.to_bits(),
        0.0f32.to_bits(),
        "witness must reject fusion"
    );

    // A bucket now has an early run entirely before a later window and a
    // second run crossing both 256 and 512. Nonconstant source values detect
    // forgetting to advance the source cursor over the skipped early run.
    source.coordinates = Array2::from_shape_vec(
        (2, slots),
        (0..16)
            .chain(200..714)
            .chain(16..200)
            .chain(714..1060)
            .collect(),
    )
    .unwrap();
    for row in 0..4 {
        for slot in 0..=slots {
            source.m[[row, slot]] = (row * 1000 + slot) as f32 * 0.0625;
            source.d[[row, slot]] = -(row as f32 + slot as f32) * 0.03125;
        }
    }
    let plan = source.plan_conv(&c, None).unwrap().unwrap();
    assert_eq!(plan.runs.len(), 4);
    assert_eq!(plan.runs[0].len, 16);
    assert_eq!(plan.runs[1].dest, 200);
    assert_eq!(plan.runs[1].len, 514);
    let output = parity(&source, &c);
    assert_eq!(output.m.dim(), (17, 1061));
}

#[test]
fn compact_conv_run_admission_charges_fragmented_metadata() {
    let coordinates = Array2::from_shape_vec(
        (2, 80),
        (0..160).step_by(2).chain((1..160).step_by(2)).collect(),
    )
    .unwrap();
    let old_bytes = 161 * 8 + 160 * size_of::<usize>() + 160 * size_of::<usize>();
    let run_bytes =
        161 * 8 + 160 * size_of::<usize>() + 160 * size_of::<ConvRun>() + 3 * size_of::<usize>();
    let mut source = PackedRows {
        n_in: (old_bytes + run_bytes) / 16 - 1,
        p: 2,
        coordinates,
        m: Array2::<f32>::zeros((2, 81)),
        d: Array2::<f32>::zeros((2, 81)),
    };
    let c = consumer(1, 1, 1, 2, 1, 2, 0, vec![1.0, -1.0]);
    // Old scalar maps would fit, but 160 singleton runs do not. A map-size
    // assumption that compression always wins would incorrectly admit here.
    assert!(old_bytes < (source.n_in + 1) * 8);
    assert!(run_bytes >= (source.n_in + 1) * 8);
    assert!(source.plan_conv(&c, None).unwrap().is_none());
    source.n_in = run_bytes / 8 + 1;
    let plan = source.plan_conv(&c, None).unwrap().unwrap();
    assert_eq!(plan.runs.len(), 160);
    assert!(plan.runs.iter().all(|r| r.len == 1));
    parity(&source, &c);
    // Empty coordinate maps still carry and accumulate physical bias once.
    source.coordinates = Array2::from_elem((2, 0), usize::MAX);
    source.m = Array2::from_shape_vec((2, 1), vec![2.0, 3.0]).unwrap();
    source.d = Array2::from_shape_vec((2, 1), vec![-0.5, 0.25]).unwrap();
    let plan = source.plan_conv(&c, None).unwrap().unwrap();
    assert!(plan.runs.is_empty());
    let output = parity(&source, &c);
    assert_eq!(output.m[[0, 0]], -1.0);
    assert_eq!(output.d[[0, 0]], -0.25);
}

#[test]
fn compact_conv_keeps_original_tap_order_signed_d_and_bias() {
    let mut source = PackedRows {
        n_in: 127,
        p: 3,
        coordinates: Array2::from_elem((3, 1), 11),
        m: Array2::<f32>::zeros((3, 2)),
        d: Array2::<f32>::zeros((3, 2)),
    };
    source.m[[0, 0]] = 16_777_216.0;
    source.m[[1, 0]] = 1.0;
    source.m[[2, 0]] = -16_777_216.0;
    source.d[[0, 0]] = -0.5;
    source.d[[1, 0]] = 0.125;
    source.d[[2, 0]] = -0.25;
    for j in 0..3 {
        source.m[[j, 1]] = (j + 1) as f32;
        source.d[[j, 1]] = -(j as f32) * 0.25;
    }
    let c = consumer(1, 1, 1, 3, 1, 3, 0, vec![1.0; 3]);
    let out = parity(&source, &c);
    assert_eq!(out.m[[0, 0]].to_bits(), 0.0f32.to_bits());
    assert_eq!(out.d[[0, 0]], -0.625);
    assert_eq!(out.m[[0, 1]], 6.0);
    assert_eq!(out.d[[0, 1]], -0.75);
    let regrouped = (source.m[[0, 0]] + source.m[[2, 0]]) + source.m[[1, 0]];
    assert_eq!(regrouped, 1.0, "witness must reject changed tap grouping");
    assert_ne!(out.m[[0, 0]].to_bits(), regrouped.to_bits());
    let negative = consumer(1, 1, 1, 3, 1, 3, 0, vec![-1.0; 3]);
    let neg = parity(&source, &negative);
    assert_eq!(neg.d[[0, 0]], -0.625, "D retains its sign under |w|");
    assert_eq!(neg.m[[0, 1]], -6.0);
}

#[test]
fn compact_conv_declines_invalid_geometry_weights_and_support_growth() {
    let source = padded_source();
    let c = consumer(2, 1, 2, 3, 1, 1, 0, vec![1.0, 1.0]);
    assert!(source.plan_conv(&c, None).unwrap().is_some());
    let mut bad = c.clone();
    bad.gather[6] = 0; // second channel aliases channel zero: cannot share maps.
    assert!(source.contract_packed(&bad, None).unwrap().is_none());
    for w in [
        f64::from_bits(1),
        f64::MIN_POSITIVE,
        2f64.powi(-150),
        f32::from_bits(1) as f64,
        f64::MAX,
        f64::INFINITY,
        f64::NAN,
    ] {
        let mut bad = c.clone();
        bad.wmat[0] = w;
        assert!(
            source.contract_packed(&bad, None).unwrap().is_none(),
            "{w:?}"
        );
    }
    let mut bad = c.clone();
    bad.wmat[0] = -0.0;
    assert!(source.contract_packed(&bad, None).unwrap().is_some());
    assert!(matches!(
        source.contract_packed(&c, Some(Instant::now() - std::time::Duration::from_secs(1))),
        Err(NyError::DeadlineExceeded(_))
    ));
    let mut invalid = padded_source();
    invalid.coordinates[[0, 2]] = 2;
    assert!(invalid.contract_packed(&c, None).unwrap().is_none());
    let mut invalid = padded_source();
    invalid.m[[0, 0]] = f32::INFINITY;
    assert!(matches!(
        invalid.contract_packed(&c, None),
        Err(NyError::NumericalInstability(_))
    ));
    let mut overflow = padded_source();
    overflow.m[[0, 0]] = f32::MAX;
    let huge = consumer(2, 1, 2, 3, 1, 1, 0, vec![2.0, 1.0]);
    assert!(matches!(
        overflow.contract_packed(&huge, None),
        Err(NyError::NumericalInstability(_))
    ));
    let dense = PackedRows {
        n_in: 3,
        p: 1,
        coordinates: Array2::from_shape_vec((1, 3), vec![0, 1, 2]).unwrap(),
        m: Array2::<f32>::zeros((1, 4)),
        d: Array2::<f32>::zeros((1, 4)),
    };
    let one = consumer(1, 1, 1, 1, 1, 1, 0, vec![1.0]);
    assert!(dense.contract_packed(&one, None).unwrap().is_none());
}

#[test]
fn compact_conv_original_frame_error_encloses_exact_signed_contraction() {
    fn q(value: f64) -> Q {
        Q::from_float(value).unwrap()
    }
    fn q32(value: f32) -> Q {
        q(ny_core::f32_to_f64_exact(value))
    }
    let source = padded_source();
    let c = consumer(
        2,
        2,
        2,
        3,
        3,
        3,
        3,
        (0..36)
            .map(|i| if i % 2 == 0 { 1.0000001 } else { -0.75000001 })
            .collect(),
    );
    let packed = source.contract_packed(&c, None).unwrap().unwrap();
    let (m, d) = packed.materialize();
    let mut h = vec![1.0; source.n_in];
    h[9] = 2f64.powi(20);
    h[17] = 2f64.powi(80);
    let error = source
        .conv_error(
            &c,
            &vec![0.0; source.m.nrows()],
            &h,
            next_up(h.iter().sum::<f64>() + 1.0),
            None,
        )
        .unwrap();
    let p_out = c.oshape.1 * c.oshape.2;
    let mut nonzero_defects = 0;
    for j in 0..m.nrows() {
        let (oc, sp) = (j / p_out, j % p_out);
        let mut exact_m = vec![Q::zero(); source.n_in + 1];
        let mut exact_d = exact_m.clone();
        // Exact real coefficients of ORIGINAL weights and stored source rows.
        // This uses neither the plan nor the f32 contraction/reduction helpers.
        for t in 0..c.k_fwd {
            let gi = c.gather[t * p_out + sp];
            if gi == usize::MAX {
                continue;
            }
            let w = q(c.wmat[oc * c.k_fwd + t]);
            let abs_w = w.abs();
            for (slot, &i) in source.coordinates.row(gi % source.p).iter().enumerate() {
                if i == usize::MAX {
                    break;
                }
                exact_m[i] += &w * q32(source.m[[gi, slot]]);
                exact_d[i] += &abs_w * q32(source.d[[gi, slot]]);
            }
            exact_m[source.n_in] += &w * q32(source.m[[gi, source.bias_column()]]);
            exact_d[source.n_in] += &abs_w * q32(source.d[[gi, source.bias_column()]]);
        }
        let mut defect = Q::zero();
        for i in 0..=source.n_in {
            let hi = if i == source.n_in { 1.0 } else { h[i] };
            defect += ((q32(m[[j, i]]) - &exact_m[i]).abs() + (q32(d[[j, i]]) - &exact_d[i]).abs())
                * q(hi);
        }
        // L1 weighted by the full original frame bounds BOTH signed walls
        // everywhere on |x_i| <= H_i, including negative input coordinates.
        assert!(error[j].is_finite());
        assert!(
            defect <= q(error[j]),
            "row {j} exact contraction debt exceeds certificate"
        );
        if !defect.is_zero() {
            nonzero_defects += 1;
        }
    }
    // Disabling the inherited numerical correction must be observable.
    assert!(
        nonzero_defects > 0,
        "zero-error mutant must fail the exact oracle"
    );
}
