// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
use super::*;
use crate::layers::{BatchNormLayer, Layer, LinearLayer, ReLULayer};
use crate::network::GraphNode;
use ndarray::{arr1, arr2, Array1, Array2};
use ny_core::{GpuCrownLayer, GpuCrownResult, GpuResnetSegment, Result};
use ny_tensor::BoundedTensor;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

#[derive(Clone, Copy)]
enum Reply {
    Correct,
    WrongCount,
    Nan,
    Infinite,
    Inverted,
    Disjoint,
    Error,
    Late,
}

struct Engine {
    capacity: usize,
    sound: bool,
    calls: AtomicUsize,
    queries: AtomicUsize,
    reply: Reply,
    expected_coordinates: Option<Vec<usize>>,
    cooperative_deadline: std::sync::Mutex<Option<Instant>>,
}
impl Engine {
    fn new(reply: Reply) -> Self {
        Self {
            capacity: MAX_ROWS,
            sound: true,
            calls: AtomicUsize::new(0),
            queries: AtomicUsize::new(0),
            reply,
            expected_coordinates: None,
            cooperative_deadline: std::sync::Mutex::new(None),
        }
    }
}
impl GemmEngine for Engine {
    fn gemm_f32(&self, _: usize, _: usize, _: usize, _: &[f32], _: &[f32]) -> Result<Vec<f32>> {
        panic!("ordinary GEMM must never be used by selected bounded rows")
    }
    fn as_gpu_crown_backward(&self) -> Option<&dyn GpuCrownBackward> {
        self.queries.fetch_add(1, Ordering::Relaxed);
        Some(self)
    }
}
impl GpuCrownBackward for Engine {
    fn honors_crown_backward_deadline(&self) -> bool {
        true
    }
    fn set_crown_backward_deadline(&self, deadline: Option<Instant>) {
        *self.cooperative_deadline.lock().unwrap() = deadline;
    }
    fn provides_sound_gpu_crown(&self) -> bool {
        self.sound
    }
    fn deadline_bounded_resnet_sound_max_rows(&self) -> usize {
        self.capacity
    }
    fn crown_backward_gpu(
        &self,
        _: &[GpuCrownLayer],
        _: &[f32],
        _: usize,
        _: &[f32],
        _: &[f32],
    ) -> Result<GpuCrownResult> {
        panic!("ordinary GPU backward must never be used by selected bounded rows")
    }
    fn crown_backward_gpu_resnet_sound_bounded_rows_with_deadline(
        &self,
        segments: &[GpuResnetSegment],
        seed: &GpuCrownSeed,
        lower: &[f32],
        upper: &[f32],
        frontier: &[Vec<f32>],
        nodes: &[Vec<f32>],
        deadline: Instant,
    ) -> Result<GpuCrownResult> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        assert_eq!(self.calls.load(Ordering::Relaxed), 1, "no dispatched retry");
        assert!(Instant::now() < deadline);
        assert!(deadline.duration_since(Instant::now()) <= Duration::from_secs(2));
        assert_eq!((lower, upper), (&[-1.0][..], &[1.0][..]));
        assert!(seed.current_dim > seed.num_specs);
        assert_eq!(seed.lower_a, seed.upper_a);
        for (i, row) in seed.lower_a.chunks_exact(seed.current_dim).enumerate() {
            let coordinate = self.expected_coordinates.as_ref().map_or(i, |v| v[i]);
            for (j, &value) in row.iter().enumerate() {
                assert_eq!(value, if coordinate == j { 1.0 } else { 0.0 });
            }
        }
        assert!(seed
            .lower_b
            .iter()
            .chain(seed.upper_b.iter())
            .all(|&v| v == 0.0));
        let [GpuResnetSegment::Chain(layers)] = segments else {
            panic!("pure source chain")
        };
        let mut charged = 0;
        for layer in layers {
            match layer {
                GpuCrownLayer::Linear { cert_err, .. } | GpuCrownLayer::Conv2d { cert_err, .. }
                    if !cert_err.is_exact() =>
                {
                    charged += 1
                }
                _ => {}
            }
        }
        assert_eq!(charged, 1, "raw 1/3 BN error must reach the backend");
        assert_eq!(frontier.len(), 1);
        assert_eq!(nodes.len(), 1);
        let mut result = GpuCrownResult {
            // Original real source: (|x|-3/2)/3, x in [-1,1]. Its exact
            // minimum is -1/2, maximum -1/6. These are independent of CROWN.
            lower_bounds: vec![-0.5; seed.num_specs],
            upper_bounds: vec![(-1.0f32 / 6.0).next_up(); seed.num_specs],
        };
        match self.reply {
            Reply::Correct => {}
            Reply::WrongCount => {
                result.upper_bounds.pop();
            }
            Reply::Nan => result.lower_bounds[0] = f32::NAN,
            Reply::Infinite => result.upper_bounds[0] = f32::INFINITY,
            Reply::Inverted => result.lower_bounds[0] = 1.0,
            Reply::Disjoint => {
                result.lower_bounds[0] = 100.0;
                result.upper_bounds[0] = 101.0;
            }
            Reply::Error => return Err(NyError::UnsupportedOp("bounded refusal".into())),
            Reply::Late => {
                while Instant::now() < deadline {
                    std::hint::spin_loop();
                }
            }
        }
        Ok(result)
    }
}

pub(super) fn source(
    selected: usize,
) -> (GraphNetwork, BoundedTensor, HashMap<String, BoundedTensor>) {
    let width = selected + 1;
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "signed",
        Layer::Linear(LinearLayer::new(arr2(&[[1.], [-1.]]), None).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "pq",
        Layer::ReLU(ReLULayer),
        vec!["signed".into()],
    ));
    let mut bias = Array1::from_elem(width, -1.5);
    bias[selected] = 2.0; // A stable, genuinely unselected final coordinate.
    graph.add_node(GraphNode::new(
        "head",
        Layer::Linear(LinearLayer::new(Array2::ones((width, 2)), Some(bias)).unwrap()),
        vec!["pq".into()],
    ));
    let one = Array1::ones(width).into_dyn();
    let zero = Array1::zeros(width).into_dyn();
    let variance = Array1::from_elem(width, 9.0).into_dyn();
    graph.add_node(GraphNode::new(
        "bn",
        Layer::BatchNorm(BatchNormLayer::new(&one, &zero, &zero, &variance, 0.0).unwrap()),
        vec!["head".into()],
    ));
    graph.add_node(GraphNode::new(
        "act",
        Layer::ReLU(ReLULayer),
        vec!["bn".into()],
    ));
    graph.add_node(GraphNode::new(
        "out",
        Layer::Linear(LinearLayer::new(Array2::eye(width), None).unwrap()),
        vec!["act".into()],
    ));
    graph.set_output("out");
    let input = BoundedTensor::new(arr1(&[-1.]).into_dyn(), arr1(&[1.]).into_dyn()).unwrap();
    let bounds = graph.collect_node_bounds(&input).unwrap();
    (graph, input, bounds)
}

fn request<'a>(
    input: &'a BoundedTensor,
    bounds: &'a HashMap<String, BoundedTensor>,
    empty: &'a HashMap<String, BoundedTensor>,
    rows: &'a [usize],
    engine: &'a Engine,
    deadline: Instant,
) -> RowRefineRequest<'a> {
    RowRefineRequest {
        input,
        target: "bn",
        crown_bounds: empty,
        ibp_bounds: bounds,
        base: &bounds["bn"],
        engine: Some(engine),
        rows,
        per_node_deadline: Some(deadline),
        deadline_is_hard: true,
        collector_patches_override: false,
        label: "selected bounded rows test",
        cut_ctx: None,
        dense_budget_bytes: 32 << 20,
        enforce_dense_budget: true,
        patches_seed: false,
        shrink_only: true,
    }
}

fn bits(b: &BoundedTensor) -> (Vec<u32>, Vec<u32>) {
    (
        b.lower().iter().map(|v| v.to_bits()).collect(),
        b.upper().iter().map(|v| v.to_bits()).collect(),
    )
}

#[test]
fn selected_head_bounded_gpu_original_source_commits_one_and_eight_rows_with_bn_charge() {
    for rows in [1, 8] {
        let (graph, input, bounds) = source(rows);
        let before: HashMap<_, _> = bounds.iter().map(|(n, b)| (n, bits(b))).collect();
        let engine = Engine::new(Reply::Correct);
        let end = Instant::now() + Duration::from_secs(30);
        let mut transaction = super::super::run(
            &graph,
            &input,
            &bounds,
            &std::collections::HashSet::from(["bn".into()]),
            Some(&engine),
            end,
            32 << 20,
            true,
        )
        .unwrap();
        assert_eq!(transaction.selected, rows);
        assert_eq!(transaction.stage, "committed");
        assert_eq!(transaction.gpu.stage, "returned");
        assert!(transaction.gpu.workspace_bytes > transaction.seed_bytes);
        assert!(transaction.gpu.work > 0);
        let base = &bounds["bn"];
        let (published, count) = transaction.publish(base, base, Some(end));
        assert_eq!(count, rows);
        for i in 0..rows {
            assert!(f64::from(published.lower()[[i]]) <= -0.5);
            assert!(f64::from(published.upper()[[i]]) >= -1.0 / 6.0);
            assert!(published.upper()[[i]] < -0.16);
        }
        assert_eq!(
            published.lower()[[rows]].to_bits(),
            base.lower()[[rows]].to_bits()
        );
        assert_eq!(
            published.upper()[[rows]].to_bits(),
            base.upper()[[rows]].to_bits()
        );
        assert!(bounds.iter().all(|(n, b)| before[n] == bits(b)));
        assert_eq!(engine.calls.load(Ordering::Relaxed), 1);
    }
}

#[test]
fn selected_head_bounded_gpu_rejects_entire_bad_payload_and_never_retries_dispatch() {
    let (graph, input, bounds) = source(1);
    for reply in [
        Reply::WrongCount,
        Reply::Nan,
        Reply::Infinite,
        Reply::Inverted,
        Reply::Disjoint,
        Reply::Error,
    ] {
        let engine = Engine::new(reply);
        let mut transaction = super::super::run(
            &graph,
            &input,
            &bounds,
            &std::collections::HashSet::from(["bn".into()]),
            Some(&engine),
            Instant::now() + Duration::from_secs(30),
            32 << 20,
            true,
        )
        .unwrap();
        let base = &bounds["bn"];
        let (published, count) = transaction.publish(base, base, None);
        assert_eq!(count, 0);
        assert_eq!(published.lower(), base.lower());
        assert_eq!(published.upper(), base.upper());
        assert_eq!(engine.calls.load(Ordering::Relaxed), 1);
    }
}

#[test]
fn selected_head_bounded_gpu_collector_publishes_partial_provenance_and_dispatch_evidence() {
    use crate::network::graph_alpha::bounds::crown::CrownIbpCollectOptions;
    use crate::types::{BoundsProvenance, CrownIbpFallbackReason};
    let (graph, input, bounds) = source(8);
    let engine = Engine::new(Reply::Correct);
    let result = graph
        .collect_crown_ibp_bounds_dag_with_options(
            &input,
            CrownIbpCollectOptions {
                engine: Some(&engine),
                deadline: Some(Instant::now() + Duration::from_secs(30)),
                deadline_includes_ibp: true,
                precomputed_ibp: Some(bounds),
                original_root_head_rows: Some(super::super::OriginalRootHeadRows),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(engine.calls.load(Ordering::Relaxed), 1);
    assert_eq!(
        result.provenance["bn"],
        BoundsProvenance::ForwardFallback(CrownIbpFallbackReason::SelectedHeadRows)
    );
    let events: Vec<_> = result
        .fallback_events
        .iter()
        .filter(|event| event.reason == CrownIbpFallbackReason::SelectedHeadRows)
        .collect();
    assert_eq!(events.len(), 1);
    assert!(events[0].details.contains("completed=8"));
    assert!(events[0].details.contains("gpu_stage=returned"));
    assert!(events[0].details.contains("full_width_retry=0"));
    assert!(!result.all_requested_crown_targets_completed());
}

#[test]
fn selected_head_bounded_gpu_late_success_has_no_authority() {
    let (graph, input, bounds) = source(1);
    let empty = HashMap::new();
    let engine = Engine::new(Reply::Late);
    let req = request(
        &input,
        &bounds,
        &empty,
        &[0],
        &engine,
        Instant::now() + Duration::from_millis(100),
    );
    let mut observation = Observation::default();
    assert!(matches!(
        try_refine(&graph, &req, req.rows, &mut observation),
        Some(Attempt {
            outcome: RowRefineOutcome::Failed(NyError::DeadlineExceeded(_)),
            ..
        })
    ));
    assert_eq!(observation.stage, "deadline");
    assert_eq!(engine.calls.load(Ordering::Relaxed), 1);
}

#[test]
fn selected_head_bounded_gpu_smaller_capacity_preserves_ranked_prefix_and_canonical_rows() {
    let (graph, input, bounds) = source(8);
    let empty = HashMap::new();
    let mut engine = Engine::new(Reply::Correct);
    engine.capacity = 4;
    engine.expected_coordinates = Some(vec![4, 5, 6, 7]);
    let req = request(
        &input,
        &bounds,
        &empty,
        &[0, 1, 2, 3, 4, 5, 6, 7],
        &engine,
        Instant::now() + Duration::from_secs(1),
    );
    let attempt = try_refine(
        &graph,
        &req,
        &[7, 6, 5, 4, 3, 2, 1, 0],
        &mut Observation::default(),
    )
    .expect("four-row backend must serve the strongest ranked prefix");
    assert_eq!(attempt.selected, 4);
    assert_eq!(&attempt.rows[..4], &[4, 5, 6, 7]);
    let RowRefineOutcome::Refined { bounds: got, k, .. } = attempt.outcome else {
        panic!("refinement")
    };
    assert_eq!(k, 4);
    for i in 0..4 {
        assert_eq!(
            got.lower()[[i]].to_bits(),
            bounds["bn"].lower()[[i]].to_bits()
        );
        assert_eq!(
            got.upper()[[i]].to_bits(),
            bounds["bn"].upper()[[i]].to_bits()
        );
    }
    assert!(got.upper()[[4]] < -0.16);
    assert_eq!(engine.calls.load(Ordering::Relaxed), 1);
}

#[test]
fn selected_head_bounded_gpu_aggregate_quote_shrinks_before_allocating_or_dispatching() {
    let (graph, input, bounds) = source(8);
    let empty = HashMap::new();
    let engine = Engine::new(Reply::Correct);
    let deadline = Instant::now() + Duration::from_secs(1);
    let quote = quote_charged_raw_bn_chain(
        &graph,
        &input,
        "bn",
        &empty,
        &bounds,
        None,
        ExtractionLimits {
            max_nodes: 128,
            max_caller_bytes: 32 << 20,
        },
        deadline,
        &mut || true,
    )
    .unwrap();
    let four = bounded_fold_bytes(quote, 4).unwrap() + local_bytes(4, 9).unwrap().1;
    let eight = bounded_fold_bytes(quote, 8).unwrap() + local_bytes(8, 9).unwrap().1;
    assert!(four < eight);
    let mut req = request(
        &input,
        &bounds,
        &empty,
        &[0, 1, 2, 3, 4, 5, 6, 7],
        &engine,
        deadline,
    );
    req.dense_budget_bytes = four;
    let mut observation = Observation::default();
    let attempt =
        try_refine(&graph, &req, req.rows, &mut observation).expect("affordable smaller fold");
    assert_eq!(attempt.selected, 4);
    assert_eq!(observation.workspace_bytes, four);
    assert_eq!(engine.calls.load(Ordering::Relaxed), 1);
}

#[test]
fn selected_head_bounded_gpu_capacity_cpu_only_memory_and_bad_identity_refuse_before_dispatch() {
    let (graph, input, bounds) = source(1);
    let empty = HashMap::new();
    for (sound, capacity) in [(false, 8), (true, 0), (true, 9)] {
        let mut engine = Engine::new(Reply::Correct);
        engine.sound = sound;
        engine.capacity = capacity;
        assert!(backend(&engine, 1).is_none());
        assert_eq!(engine.calls.load(Ordering::Relaxed), 0);
    }
    let engine = Engine::new(Reply::Correct);
    for rows in [&[][..], &[0, 0][..], &[2][..], &[1, 0][..]] {
        let req = request(
            &input,
            &bounds,
            &empty,
            rows,
            &engine,
            Instant::now() + Duration::from_secs(1),
        );
        assert!(try_refine(&graph, &req, req.rows, &mut Observation::default()).is_none());
    }
    let mut req = request(
        &input,
        &bounds,
        &empty,
        &[0],
        &engine,
        Instant::now() + Duration::from_secs(1),
    );
    req.dense_budget_bytes = 1;
    assert!(try_refine(&graph, &req, req.rows, &mut Observation::default()).is_none());
    assert_eq!(engine.calls.load(Ordering::Relaxed), 0);
    let before = engine.queries.load(Ordering::Relaxed);
    let _guard = crate::sound_f64_gemm::CpuOnlyF64Guard::new();
    assert!(try_refine(&graph, &req, req.rows, &mut Observation::default()).is_none());
    assert_eq!(
        engine.queries.load(Ordering::Relaxed),
        before,
        "CPU-only does not inspect GPU factories or capabilities"
    );
}

#[test]
fn selected_head_bounded_gpu_aggregate_memory_and_work_are_checked_without_overflow() {
    assert!(fits_memory(
        1024,
        1024,
        ProcessMemoryEnvelope::Bounded {
            headroom_bytes: 2048
        }
    ));
    assert!(!fits_memory(
        1024,
        1024,
        ProcessMemoryEnvelope::Bounded {
            headroom_bytes: 2047
        }
    ));
    assert!(!fits_memory(1024, 1023, ProcessMemoryEnvelope::Unbounded));
    assert!(!fits_memory(
        MAX_BYTES + 1,
        usize::MAX,
        ProcessMemoryEnvelope::Unbounded
    ));
    assert!(!fits_memory(
        1,
        usize::MAX,
        ProcessMemoryEnvelope::Unavailable
    ));
    let (graph, input, bounds) = source(1);
    let q = quote_charged_raw_bn_chain(
        &graph,
        &input,
        "bn",
        &bounds,
        &bounds,
        None,
        ExtractionLimits {
            max_nodes: 128,
            max_caller_bytes: 32 << 20,
        },
        Instant::now() + Duration::from_secs(1),
        &mut || true,
    )
    .unwrap();
    assert!(bounded_fold_bytes(q, usize::MAX).is_none());
    assert!(bounded_fold_work(q, u64::MAX, 1).is_none());
    assert!(bounded_fold_work(q, 1, u64::MAX).is_none());
}
