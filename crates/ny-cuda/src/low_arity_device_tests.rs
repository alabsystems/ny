// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Real production size/deadline entry, using exact source-valid relations.
use super::*;
use ny_core::{
    GpuCrownLayer, GpuCrownSeed, GpuResnetSegment, ResidentLocalL1LowerCutCarrier,
    ResidentLowerCutChannel, ResidentTwoLocalL1LowerCutCarrier,
};

fn ch(v: f32) -> ResidentLowerCutChannel {
    ResidentLowerCutChannel::try_new(v, 0.).unwrap()
}

#[test]
fn cuda_low_arity_local_l1_production_dispatch_encloses_pair_and_singleton_sources() {
    with_capable_cuda(|engine| {
        const N: usize = 4096;
        // The complete source exceeds the unchanged one-row resident MAC gate.
        // Every input/target coordinate is authored; only zero objective entries
        // are inactive. No duplicated rows or test-only gate bypass is used.
        let mut w = vec![0f32; N * N];
        for i in 4..N {
            w[i * N + i] = 1.;
        }
        // Independent dyadic means u,v,c,d fill all source columns. The first
        // pair is ((u+v)/2,(u-v)/2), and the singletons are c and d.
        for i in 0..N / 4 {
            w[i] = 2. / N as f32;
            w[N + i] = 2. / N as f32;
            w[N / 4 + i] = 2. / N as f32;
            w[N + N / 4 + i] = -2. / N as f32;
            w[2 * N + N / 2 + i] = 4. / N as f32;
            w[3 * N + 3 * N / 4 + i] = 4. / N as f32;
        }
        let segments = vec![GpuResnetSegment::Chain(vec![
            GpuCrownLayer::Activation {
                lower_slope: vec![0.; N],
                upper_slope: vec![0.5; N],
                lower_intercept: vec![0.; N],
                upper_intercept: vec![0.5; N],
                num_neurons: N,
            },
            GpuCrownLayer::Linear {
                weight: Arc::from(w),
                bias: None,
                out_features: N,
                in_features: N,
                cert_err: Default::default(),
            },
        ])];
        let lo = vec![-1.; N];
        let hi = vec![1.; N];
        let make_seed = |singleton_pair: bool, shift: f32| {
            let mut a = vec![0.; N];
            if singleton_pair {
                a[2] = -2.;
                a[3] = -2.;
            } else {
                a[0] = -2.;
                a[1] = -2.;
                a[2] = -2.;
            }
            GpuCrownSeed {
                lower_a: Arc::from(a.clone()),
                upper_a: Arc::from(a),
                lower_b: Arc::from(vec![4.25 + shift]),
                upper_b: Arc::from(vec![4.25 + shift]),
                num_specs: 1,
                current_dim: N,
            }
        };
        for shift in [0., -0.5] {
            for lens in [[2, 1], [1, 2], [1, 1]] {
                let end = Instant::now() + std::time::Duration::from_secs(30);
                let pair = lens == [1, 1];
                let seed = make_seed(pair, shift);
                let mut ids = [[0; 8]; 2];
                let mut pre = [[ch(0.); 8]; 2];
                let mut post = pre;
                if pair {
                    ids[0][0] = 2;
                    ids[1][0] = 3;
                } else if lens[0] == 2 {
                    ids[0][1] = 1;
                    ids[1][0] = 2;
                } else {
                    ids[0][0] = 2;
                    ids[1][1] = 1;
                }
                for g in 0..2 {
                    for i in 0..lens[g] {
                        pre[g][i] = ch(-1.);
                        post[g][i] = ch(2.);
                    }
                }
                // |(a+b)/2|+|(a-b)/2|<=1; each singleton |x|<=1.
                let c = ResidentTwoLocalL1LowerCutCarrier::try_new(
                    0,
                    N,
                    lens,
                    ids,
                    [1.; 2],
                    pre,
                    post,
                    ch(-2.),
                    end,
                )
                .unwrap();
                let before = cuda_deadline_f64_gemm_stats();
                let got = engine
                    .crown_backward_gpu_resnet_two_local_l1_lower_cut_with_deadline(
                        &segments, &seed, &lo, &hi, &c, end,
                    )
                    .unwrap();
                let after = cuda_deadline_f64_gemm_stats();
                assert!(
                    got > 0.24 + shift && got <= 0.25 + shift,
                    "lens={lens:?} got={got}"
                );
                assert!(after.calls > before.calls && after.dispatches > before.dispatches);
                eprintln!("low_arity_cuda_completed lens={lens:?} shift={shift} lower={got} calls={} dispatches={}",after.calls-before.calls,after.dispatches-before.dispatches);
            }
        }
        // Independently exercise the single-relation carrier for both new
        // arities. A singleton leaves the diamond's original triangles intact.
        for active in [1, 2] {
            let end = Instant::now() + std::time::Duration::from_secs(30);
            let seed = make_seed(false, 0.);
            let mut ids = [0; 8];
            let mut pre = [ch(0.); 8];
            let mut post = pre;
            if active == 1 {
                ids[0] = 2;
            } else {
                ids[1] = 1;
            }
            for i in 0..active {
                pre[i] = ch(-1.);
                post[i] = ch(2.);
            }
            let c = ResidentLocalL1LowerCutCarrier::try_new(
                0,
                N,
                active,
                ids,
                1.,
                pre,
                post,
                ch(-1.),
                end,
            )
            .unwrap();
            let before = cuda_deadline_f64_gemm_stats();
            let got = engine
                .crown_backward_gpu_resnet_local_l1_lower_cut_with_deadline(
                    &segments, &seed, &lo, &hi, &c, end,
                )
                .unwrap();
            let after = cuda_deadline_f64_gemm_stats();
            let ideal = if active == 1 { -0.75 } else { 0.25 };
            assert!(got > ideal - 0.01 && got <= ideal);
            assert!(after.calls > before.calls && after.dispatches > before.dispatches);
            eprintln!("low_arity_cuda_single_completed active={active} lower={got} calls={} dispatches={}",after.calls-before.calls,after.dispatches-before.dispatches);
        }
    });
}
