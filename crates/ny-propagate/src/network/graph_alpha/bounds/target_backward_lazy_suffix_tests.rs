// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Real lazy-plan state and finite CPU-fallback controls; no environment writes.
use super::*;
use crate::{GraphNode, LinearLayer, ReLULayer};
use ndarray::{arr1, arr2};
use ny_core::{
    GemmEngine, GpuCrownBackward, GpuCrownLayer, GpuCrownResult, GpuCrownSeed, NaiveCpuGemmEngine,
};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

#[derive(Default)]
struct ReadyEngine {
    lookups: AtomicUsize,
}
impl GemmEngine for ReadyEngine {
    fn gemm_f32(&self, m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        NaiveCpuGemmEngine.gemm_f32(m, k, n, a, b)
    }
    fn as_gpu_crown_backward(&self) -> Option<&dyn GpuCrownBackward> {
        self.lookups.fetch_add(1, Ordering::SeqCst);
        Some(self)
    }
}
impl GpuCrownBackward for ReadyEngine {
    fn provides_sound_gpu_crown(&self) -> bool {
        true
    }
    fn honors_crown_backward_deadline(&self) -> bool {
        true
    }
    fn crown_backward_gpu(
        &self,
        _: &[GpuCrownLayer],
        _: &[f32],
        _: usize,
        _: &[f32],
        _: &[f32],
    ) -> Result<GpuCrownResult> {
        panic!("no resident dispatch is allowed in these short-authority controls")
    }
    fn crown_backward_gpu_seeded(
        &self,
        _: &[GpuCrownLayer],
        _: &GpuCrownSeed,
        _: &[f32],
        _: &[f32],
    ) -> Result<GpuCrownResult> {
        panic!("short-authority fast seeded suffix must refuse before dispatch")
    }
    fn crown_backward_gpu_seeded_sound(
        &self,
        _: &[GpuCrownLayer],
        _: &GpuCrownSeed,
        _: &[f32],
        _: &[f32],
    ) -> Result<GpuCrownResult> {
        panic!("short-authority sound seeded suffix must refuse before dispatch")
    }
}

fn fixture() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "pre",
        Layer::Linear(LinearLayer::new(arr2(&[[1., -1.], [-1., 1.]]), None).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["pre".into()],
    ));
    graph.add_node(GraphNode::new(
        "out",
        Layer::Linear(
            LinearLayer::new(arr2(&[[1., 1.], [1., -1.]]), Some(arr1(&[-0.25, 0.125]))).unwrap(),
        ),
        vec!["relu".into()],
    ));
    graph.set_output("out");
    let input =
        BoundedTensor::new(arr1(&[-1., -1.]).into_dyn(), arr1(&[1., 1.]).into_dyn()).unwrap();
    (graph, input)
}

#[test]
fn lazy_suffix_short_and_expired_authority_skip_actual_plan_and_lookup() {
    let (graph, input) = fixture();
    let ibp = graph.collect_node_bounds(&input).unwrap();
    let ancestors = graph.ancestors("out").unwrap();
    let crown = HashMap::new();
    let plan = LazyGpuSuffixPlan::new(&ancestors, &graph, &input, &crown, &ibp, None);
    let engine = ReadyEngine::default();
    let short = Instant::now() + Duration::from_secs(2);
    assert!(crate::sound_gpu_gate::gpu_suffix_declines_under_finite_authority(short));
    assert!(std::ptr::eq(
        plan.plan_for_finisher(Some(&engine), Some(short)),
        &raw const plan.empty
    ));
    assert!(
        plan.plan.get().is_none(),
        "real OnceLock must stay uninitialized"
    );
    assert_eq!(engine.lookups.load(Ordering::SeqCst), 0);

    let expired = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
    assert!(std::ptr::eq(
        plan.plan_for_finisher(Some(&engine), Some(expired)),
        &raw const plan.empty
    ));
    assert!(plan.plan.get().is_none());
    assert_eq!(engine.lookups.load(Ordering::SeqCst), 0);
    let relation = LinearBounds::identity(input.len());
    let contract = GraphTargetShapeContract::from_bounds("out", &ibp["out"]);
    let mut pending = CrownMergeAccumulator::new();
    pending.insert(
        NETWORK_INPUT.into(),
        CrownBounds::Dense(LinearBounds::identity(input.len())),
    );
    let error = try_finish_target_gpu_suffix_with_pending_input(
        &input,
        "out",
        GpuSuffixSeedRelation {
            bounds: &relation,
            node_output_box: None,
        },
        &plan.empty,
        Some(&engine),
        Some(expired),
        &contract,
        &mut pending,
    )
    .unwrap_err();
    assert!(
        error.is_deadline_exceeded(),
        "original finisher expiry must remain typed"
    );
    assert!(pending.contains_key(NETWORK_INPUT));
    assert_eq!(engine.lookups.load(Ordering::SeqCst), 0);

    // An unbounded, genuinely available route still builds the real nonempty
    // descriptors once. No global fallback factory or environment toggle is used.
    let built = plan.plan_for_finisher(Some(&engine), None);
    assert!(built.contains("pre") && built.contains("relu") && built.contains("out"));
    assert!(plan.plan.get().is_some());
    assert!(std::ptr::eq(
        built,
        plan.plan_for_finisher(Some(&engine), None)
    ));
    let lookups = engine.lookups.load(Ordering::SeqCst);
    let short = Instant::now() + Duration::from_secs(2);
    assert!(
        std::ptr::eq(
            plan.plan_for_finisher(Some(&engine), Some(short)),
            &raw const plan.empty
        ),
        "even an already warm plan must not bypass the existing finite refusal"
    );
    assert_eq!(engine.lookups.load(Ordering::SeqCst), lookups);
}

#[ntest::timeout(10000)]
#[test]
fn lazy_suffix_short_authority_preserves_real_cpu_fallback_bounds() {
    let (graph, input) = fixture();
    let ibp = graph.collect_node_bounds(&input).unwrap();
    let ancestors = graph.ancestors("out").unwrap();
    let crown = HashMap::new();
    let cold = LazyGpuSuffixPlan::new(&ancestors, &graph, &input, &crown, &ibp, None);
    let eager = LazyGpuSuffixPlan::new(&ancestors, &graph, &input, &crown, &ibp, None);
    assert!(eager.get_or_build().contains("out"));
    let engine = ReadyEngine::default();
    let end = Instant::now() + Duration::from_secs(2);
    let _scope = crate::sound_gpu_gate::GpuCrownDeadlineScope::set(Some(&engine), Some(end));
    let contract = GraphTargetShapeContract::from_bounds("out", &ibp["out"]);
    let run = |plan: &LazyGpuSuffixPlan<'_>| {
        graph
            .run_target_backward_pass(
                &input,
                "out",
                &crown,
                &ibp,
                None,
                Some(&engine),
                "lazy suffix finite fallback",
                Some(end),
                true,
                false,
                &ancestors,
                &contract,
                2,
                input.len(),
                false,
                plan,
                build_subset_seed(2, &[0, 1], None, Some(end)).unwrap(),
                None,
                None,
            )
            .unwrap()
            .expect("genuine nonconstant affine/ReLU input contribution")
    };
    let actual = run(&cold);
    let control = run(&eager);
    assert!(
        Instant::now() < end,
        "both tiny walks must finish inside their shared authority"
    );
    assert!(
        cold.plan.get().is_none(),
        "actual finite walk must never materialize its lazy plan"
    );
    assert!(
        eager.plan.get().is_some(),
        "control really supplied materialized descriptors"
    );
    assert!(
        engine.lookups.load(Ordering::SeqCst) > 0,
        "ready route is not a CPU-only false positive"
    );
    assert_eq!(
        actual
            .lower()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        control
            .lower()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        actual
            .upper()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        control
            .upper()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
    let lo = actual.lower();
    let hi = actual.upper();
    assert!(lo.iter().chain(hi.iter()).all(|v| v.is_finite()));
    // Authored corners/center: output0=|x-y|-1/4, output1=x-y+1/8.
    assert!(lo[[0]] <= -0.25 && hi[[0]] >= 1.75);
    assert!(lo[[1]] <= -1.875 && hi[[1]] >= 2.125);
}
