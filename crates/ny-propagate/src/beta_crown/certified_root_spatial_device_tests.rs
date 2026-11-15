// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Complete spatial source through the production CUDA entry and root owner.
use super::*;
use crate::layers::BatchNormLayer;

const C: usize = 64;
const H: usize = 32;
const D: usize = C * H * H;
const ACTIVE: usize = 16;

fn source(
    shift: f32,
    parameter_error: f32,
) -> (GraphNetwork, BoundedTensor, HashMap<String, BoundedTensor>) {
    // Every pair is identical, but distinct pairs have distinct authored
    // filters. All source pixels feed the later channel-pair cancellations.
    let mut k1 = ArrayD::zeros(IxDyn(&[C, 3, 3, 3]));
    for pair in 1..C / 2 {
        for ic in 0..3 {
            for y in 0..3 {
                for x in 0..3 {
                    let v = (((pair + ic * 3 + y * 2 + x) % 7) as f32 + 1.) / 256.;
                    k1[[2 * pair, ic, y, x]] = v;
                    k1[[2 * pair + 1, ic, y, x]] = v;
                }
            }
        }
    }
    k1[[0, 0, 1, 1]] = 1. / 32.;
    k1[[1, 0, 1, 1]] = 1. / 32.;
    let mut k2 = ArrayD::zeros(IxDyn(&[C, C, 3, 3]));
    for oc in 0..C - 1 {
        for pair in 0..C / 2 {
            for y in 0..3 {
                for x in 0..3 {
                    k2[[oc, 2 * pair, y, x]] = 1. / 64.;
                    k2[[oc, 2 * pair + 1, y, x]] = -1. / 64.;
                }
            }
        }
    }
    // The last target channel preserves original x[0,h,w] as a real tiny
    // objective term. It cannot be cropped or replaced by a constant.
    k2[[C - 1, 0, 1, 1]] = 32.;
    let mut bias = Array1::zeros(C);
    bias[C - 1] = -64.;
    let mut bn =
        BatchNormLayer::from_scale_bias(Array1::ones(C).into_dyn(), Array1::zeros(C).into_dyn())
            .unwrap();
    bn.scale_err = Array1::from_elem(C, parameter_error).into_dyn();
    let mut g = GraphNetwork::new();
    g.add_node(GraphNode::from_input(
        "conv1",
        Layer::Conv2d(
            Conv2dLayer::new(k1, Some(Array1::from_elem(C, 2.)), (1, 1), (1, 1)).unwrap(),
        ),
    ));
    g.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["conv1".into()],
    ));
    g.add_node(GraphNode::new(
        "conv2",
        Layer::Conv2d(Conv2dLayer::new(k2, Some(bias), (1, 1), (1, 1)).unwrap()),
        vec!["relu1".into()],
    ));
    g.add_node(GraphNode::new(
        "bn",
        Layer::BatchNorm(bn),
        vec!["conv2".into()],
    ));
    g.add_node(GraphNode::new(
        "relu2",
        Layer::ReLU(ReLULayer),
        vec!["bn".into()],
    ));
    g.add_node(GraphNode::new(
        "flat",
        Layer::Flatten(FlattenLayer::flatten_all()),
        vec!["relu2".into()],
    ));
    let mut weights = vec![0.; D];
    for value in &mut weights[..ACTIVE] {
        *value = -1.;
    }
    weights[(C - 1) * H * H + 100] = -EPS;
    g.add_node(GraphNode::new(
        "out",
        Layer::Linear(
            LinearLayer::new(
                Array2::from_shape_vec((1, D), weights).unwrap(),
                Some(arr1(&[0.25 + EPS + shift])),
            )
            .unwrap(),
        ),
        vec!["flat".into()],
    ));
    g.set_output("out");
    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[3, H, H]), -1.),
        ArrayD::from_elem(IxDyn(&[3, H, H]), 1.),
    )
    .unwrap();
    let bounds = g.collect_node_bounds(&input).unwrap();
    (g, input, bounds)
}

struct CountedCuda {
    inner: ny_cuda::CudaGemmEngine,
    calls: Mutex<Vec<(usize, usize)>>,
}
impl GemmEngine for CountedCuda {
    fn gemm_f32(&self, _: usize, _: usize, _: usize, _: &[f32], _: &[f32]) -> Result<Vec<f32>> {
        panic!("no CPU or ordinary GEMM retry")
    }
    fn as_gpu_crown_backward(&self) -> Option<&dyn GpuCrownBackward> {
        Some(self)
    }
}
impl GpuCrownBackward for CountedCuda {
    fn provides_sound_gpu_crown(&self) -> bool {
        self.inner.provides_sound_gpu_crown()
    }
    fn deadline_bounded_resnet_sound_max_rows(&self) -> usize {
        self.inner.deadline_bounded_resnet_sound_max_rows()
    }
    fn crown_backward_gpu(
        &self,
        _: &[GpuCrownLayer],
        _: &[f32],
        _: usize,
        _: &[f32],
        _: &[f32],
    ) -> Result<GpuCrownResult> {
        panic!("no ordinary CROWN retry")
    }
    fn crown_backward_gpu_resnet_sound_bounded_rows_with_deadline(
        &self,
        s: &[GpuResnetSegment],
        q: &GpuCrownSeed,
        l: &[f32],
        u: &[f32],
        f: &[Vec<f32>],
        n: &[Vec<f32>],
        end: Instant,
    ) -> Result<GpuCrownResult> {
        self.calls
            .lock()
            .unwrap()
            .push((q.num_specs, q.current_dim));
        self.inner
            .crown_backward_gpu_resnet_sound_bounded_rows_with_deadline(s, q, l, u, f, n, end)
    }
}

#[test]
fn spatial_feedback_cuda_complete_original_multibatch_bn_source_and_unsafe_shift() {
    let engine = CountedCuda {
        inner: ny_cuda::CudaGemmEngine::new().expect("explicit CUDA conformance requires CUDA"),
        calls: Mutex::new(Vec::new()),
    };
    for parameter_error in [0., 1. / 4194304.] {
        for shift in [0., -0.5] {
            let (g, input, bounds) = source(shift, parameter_error);
            let end = Instant::now() + Duration::from_secs(30);
            let limits = ExtractionLimits {
                max_nodes: 128,
                max_caller_bytes: 256 << 20,
            };
            // The production owner is called only after an ordinary complete root
            // result. This is that same stage, within the same query clock.
            let full = extract_charged_raw_bn_chain(
                &g,
                &input,
                "out",
                &bounds,
                &bounds,
                None,
                limits,
                end,
                &mut || true,
            )
            .unwrap();
            let a = Arc::<[f32]>::from(vec![1.]);
            let z = Arc::<[f32]>::from(vec![0.]);
            let seed = GpuCrownSeed {
                lower_a: a.clone(),
                upper_a: a,
                lower_b: z.clone(),
                upper_b: z,
                num_specs: 1,
                current_dim: 1,
            };
            let ordinary = engine
                .inner
                .crown_backward_gpu_resnet_sound_bounded_rows_with_deadline(
                    &full.segments,
                    &seed,
                    input.lower().as_slice().unwrap(),
                    input.upper().as_slice().unwrap(),
                    &full.frontier_abs,
                    &full.node_abs,
                    end,
                )
                .unwrap();
            let old = (ordinary.lower_bounds[0], ordinary.upper_bounds[0]);
            assert!(old.0 < 0.);
            drop(full);
            let mut cache = CachedLinearBounds::default();
            let mut coefficients = vec![0.; D];
            for value in &mut coefficients[..ACTIVE] {
                *value = -1.;
            }
            coefficients[(C - 1) * H * H + 100] = -EPS;
            cache.lower_a.insert(
                "relu2".into(),
                Array2::from_shape_vec((1, D), coefficients).unwrap(),
            );
            let defaults = BetaCrownConfig::default();
            let config = BetaCrownConfig {
                verification_artifact_authority: VerificationArtifactAuthority::VerdictOnly,
                certified_root_cut_raw_source: true,
                certified_root_cut_owner: QueryOwner::new(Some(end), true),
                ..defaults
            };
            let before = ny_cuda::cuda_deadline_f64_gemm_stats();
            engine.calls.lock().unwrap().clear();
            let mut out = [old];
            let result = try_tighten_with_spatial_advice(
                &config,
                &g,
                &input,
                &[vec![1.]],
                &[0.],
                &mut out,
                &bounds,
                None,
                Some(&engine),
                Some(end),
                Some(Advice {
                    cache: &cache,
                    active_indices: &[0],
                }),
            );
            assert_eq!(
                result,
                Some(0),
                "bounded API calls on refusal: {:?}",
                engine.calls.lock().unwrap()
            );
            let after = ny_cuda::cuda_deadline_f64_gemm_stats();
            let calls = engine.calls.lock().unwrap().clone();
            assert!(
                calls.len() >= 3,
                "must need multiple completed prefix batches and full replay: {calls:?}"
            );
            assert_eq!(calls.last(), Some(&(1, 1)));
            assert!(calls[..calls.len() - 1]
                .iter()
                .all(|(rows, width)| *rows <= 8 && *width == D));
            assert!(calls[..calls.len() - 1].iter().map(|v| v.0).sum::<usize>() >= ACTIVE);
            assert!(after.calls > before.calls && after.dispatches > before.dispatches);
            assert!(
                out[0].0 > 0.24 + shift && out[0].0 <= 0.25 + shift,
                "old={old:?} out={out:?}"
            );
            assert_eq!(out[0].1.to_bits(), old.1.to_bits());
            // Each pair cancels identically at every spatial tap and boundary.
            // Only the tiny original x term remains. Scale error widens its exact
            // real minimum by at most EPS*parameter_error, never to a safe shift.
            let truth =
                f64::from(0.25 + EPS + shift) - f64::from(EPS) * (1. + f64::from(parameter_error));
            assert!(f64::from(out[0].0) <= truth);
            eprintln!("spatial_feedback_cuda_completed shift={shift} bn_error={parameter_error} lower={} prefix_batches={} prefix_rows={} original_replays=1 gemm_calls={} dispatches={}",out[0].0,calls.len()-1,calls[..calls.len()-1].iter().map(|v|v.0).sum::<usize>(),after.calls-before.calls,after.dispatches-before.dispatches);
        }
    }
}
