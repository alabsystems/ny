// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
use super::*;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};

#[derive(Default)]
struct Cpu {
    calls: Mutex<Vec<(usize, usize, usize)>>,
    polls: AtomicUsize,
    refuse: AtomicUsize,
}
impl GemmEngine for Cpu {
    fn gemm_f32(&self, _: usize, _: usize, _: usize, _: &[f32], _: &[f32]) -> Result<Vec<f32>> {
        panic!("f64 only")
    }
    fn gemm_f64(&self, m: usize, k: usize, n: usize, a: &[f64], b: &[f64]) -> Result<Vec<f64>> {
        self.calls.lock().unwrap().push((m, k, n));
        let mut out = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                for p in 0..k {
                    out[i * n + j] += a[i * k + p] * b[p * n + j];
                }
            }
        }
        Ok(out)
    }
    fn poll_crown_backward_deadline(&self) -> Result<()> {
        let p = self.polls.fetch_add(1, Ordering::SeqCst) + 1;
        if p == self.refuse.load(Ordering::SeqCst) {
            Err(NyError::DeadlineExceeded("sparse test".into()))
        } else {
            Ok(())
        }
    }
}
fn geometry(stride: usize) -> Geometry {
    Geometry {
        oc: 3,
        ic: 2,
        kh: 3,
        kw: 3,
        sh: stride,
        sw: stride,
        ph: 1,
        pw: 1,
        oh: (7 + 2 - 3) / stride + 1,
        ow: (8 + 2 - 3) / stride + 1,
        in_h: 7,
        in_w: 8,
    }
}
fn fixture(g: Geometry, batch: usize) -> (Vec<f64>, Vec<f64>, Vec<f32>) {
    let spatial = g.oh * g.ow;
    let mut a = vec![-0.0; batch * g.oc * spatial];
    let mut e = vec![0.0; a.len()];
    for b in 0..batch {
        for (j, p) in [0, spatial / 2, spatial - 1].into_iter().enumerate() {
            for c in 0..g.oc {
                let i = b * g.oc * spatial + c * spatial + p;
                // One active spatial row has zero center but nonzero radius.
                a[i] = if j == 1 {
                    0.0
                } else {
                    (b as f64 + c as f64 - 3.0) / 8.0
                };
                e[i] = (c as f64 + 1.0) / 64.0;
            }
        }
    }
    let w = (0..g.oc * g.ic * g.kh * g.kw)
        .map(|i| (i as i32 % 7 - 3) as f32 / 16.0)
        .collect();
    (a, e, w)
}

#[test]
fn sparse_conv_exact_dyadic_interval_oracle_edges_overlap_stride_and_all16_rows() {
    for stride in [1, 2] {
        for batch in [1, 2, 16] {
            let g = geometry(stride);
            let (a, e, w) = fixture(g, batch);
            let cpu = Cpu::default();
            let (center, radius) = try_step(&cpu, batch, &a, &e, &w, g).unwrap().unwrap();
            let spatial = g.oh * g.ow;
            let ins = g.in_h * g.in_w;
            let n = g.ic * g.kh * g.kw;
            let mut truth = vec![0.0; batch * g.ic * ins];
            let mut uncertainty = truth.clone();
            // Independent direct adjoint interval computation. Bounded dyadic
            // products and reductions below are exact in binary64, with no tolerance.
            for b in 0..batch {
                for c in 0..g.oc {
                    for y in 0..g.oh {
                        for x in 0..g.ow {
                            let src = b * g.oc * spatial + c * spatial + y * g.ow + x;
                            for ic in 0..g.ic {
                                for ky in 0..g.kh {
                                    for kx in 0..g.kw {
                                        let iy = (y * g.sh + ky) as isize - g.ph as isize;
                                        let ix = (x * g.sw + kx) as isize - g.pw as isize;
                                        if iy < 0
                                            || ix < 0
                                            || iy >= g.in_h as isize
                                            || ix >= g.in_w as isize
                                        {
                                            continue;
                                        }
                                        let dst = b * g.ic * ins
                                            + ic * ins
                                            + iy as usize * g.in_w
                                            + ix as usize;
                                        let weight =
                                            f64::from(w[c * n + ic * g.kh * g.kw + ky * g.kw + kx]);
                                        truth[dst] += a[src] * weight;
                                        uncertainty[dst] += e[src] * weight.abs();
                                    }
                                }
                            }
                        }
                    }
                }
            }
            assert!(radius.contains(&0.0), "untouched sites stay exactly zero");
            for i in 0..center.len() {
                assert_eq!(center[i], truth[i], "center {i}");
                assert!(
                    center[i] - radius[i] <= truth[i] - uncertainty[i],
                    "lower {i}"
                );
                assert!(
                    center[i] + radius[i] >= truth[i] + uncertainty[i],
                    "upper {i}"
                );
            }
            let calls = cpu.calls.lock().unwrap();
            assert_eq!(calls.len(), 3);
            assert!(calls
                .iter()
                .all(|&(m, k, n0)| m == 3 * batch && k == g.oc && n0 == n));
            assert!(calls[0].0 < batch * spatial);
        }
    }
}

#[test]
fn sparse_conv_empty_support_has_no_dispatch_and_cannot_hide_bad_source() {
    let g = geometry(1);
    let cpu = Cpu::default();
    let len = g.oc * g.oh * g.ow;
    let a = vec![-0.0; len];
    let e = vec![0.0; len];
    let mut w = vec![1.0; g.oc * g.ic * g.kh * g.kw];
    let (out, err) = try_step(&cpu, 1, &a, &e, &w, g).unwrap().unwrap();
    assert!(out.iter().chain(&err).all(|v| exact_zero(*v)));
    assert!(cpu.calls.lock().unwrap().is_empty());
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        w[0] = bad;
        assert!(try_step(&cpu, 1, &a, &e, &w, g).is_err());
    }
    w[0] = 1.0;
    for bad in [f64::NAN, f64::INFINITY, -0.125] {
        let mut invalid = e.clone();
        invalid[0] = bad;
        assert!(try_step(&cpu, 1, &a, &invalid, &w, g).is_err());
    }
    assert!(cpu.calls.lock().unwrap().is_empty());
}

#[test]
fn sparse_conv_nonzero_subnormal_center_or_radius_never_becomes_empty_support() {
    let g = Geometry {
        oc: 1,
        ic: 1,
        kh: 1,
        kw: 1,
        sh: 1,
        sw: 1,
        ph: 0,
        pw: 0,
        oh: 1,
        ow: 8,
        in_h: 1,
        in_w: 8,
    };
    for radius_only in [false, true] {
        let cpu = Cpu::default();
        let mut a = vec![0.0; 8];
        let mut e = a.clone();
        if radius_only {
            e[3] = ETA64;
        } else {
            a[3] = ETA64;
        }
        let (out, err) = try_step(&cpu, 1, &a, &e, &[0.5], g).unwrap().unwrap();
        assert_eq!(out[3], 0.0);
        assert!(err[3] >= ETA64);
        assert_eq!(cpu.calls.lock().unwrap().as_slice(), &[(1, 1, 1); 3]);
        for i in [0, 1, 2, 4, 5, 6, 7] {
            assert!(exact_zero(out[i]) && exact_zero(err[i]));
        }
    }
}

#[test]
fn sparse_conv_dense_or_wide_frontiers_retain_original_route() {
    let g = geometry(1);
    let cpu = Cpu::default();
    let len = g.oc * g.oh * g.ow;
    let w = vec![1.0; g.oc * g.ic * g.kh * g.kw];
    assert!(try_step(&cpu, 1, &vec![1.0; len], &vec![0.0; len], &w, g)
        .unwrap()
        .is_none());
    assert!(
        try_step(&cpu, 17, &vec![0.0; 17 * len], &vec![0.0; 17 * len], &w, g)
            .unwrap()
            .is_none()
    );
    assert!(cpu.calls.lock().unwrap().is_empty());
}

#[test]
fn sparse_conv_every_cooperative_poll_refuses_without_a_completed_result() {
    let g = geometry(1);
    let (a, e, w) = fixture(g, 2);
    let baseline = Cpu::default();
    assert!(try_step(&baseline, 2, &a, &e, &w, g).unwrap().is_some());
    let polls = baseline.polls.load(Ordering::SeqCst);
    assert!(polls > 10);
    for fail in 1..=polls {
        let cpu = Cpu::default();
        cpu.refuse.store(fail, Ordering::SeqCst);
        assert!(try_step(&cpu, 2, &a, &e, &w, g).is_err(), "poll {fail}");
    }
}

#[test]
fn sparse_conv_activation_preserves_only_proven_exact_zeros_in_each_dual_branch() {
    let tiny = ETA64;
    let la = [-0.0, 0.0, 0.0, tiny];
    let ua = [0.0, -0.0, tiny, 0.0];
    let le = [0.0, 0.125, 0.0, 0.0];
    let ue = [0.0, 0.0, 0.0, 0.125];
    let (a, b, e, f) = activation_step_f64_branches(
        1,
        4,
        &la,
        &ua,
        &le,
        &ue,
        &[0.25; 4],
        &[0.5; 4],
        Some(&[0.75; 4]),
    )
    .unwrap();
    assert!(exact_zero(a[0]) && exact_zero(b[0]) && exact_zero(e[0]) && exact_zero(f[0]));
    assert!(e[1] > 0.0 && f[3] > 0.0 && e[3] >= tiny && f[2] >= tiny);
    // No NaN slope can be hidden by the exact-zero shortcut.
    assert!(activation_step_f64_branches(
        1,
        1,
        &[0.0],
        &[0.0],
        &[0.0],
        &[0.0],
        &[f32::NAN],
        &[0.5],
        None
    )
    .is_err());
}
