// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `#cgan-child-zono` prepass pins: dark by default, shrink-only merges that
//! keep the current entry on a shape mismatch or a disjoint element, memory
//! admission that refuses instead of running, and parallel tranches that
//! publish maps bit-identical to the serial schedule.
//!
//! #m8-cgan-leaf-batch: the batch evaluator is the production default and
//! publishes maps and overrides bitwise identical to the serial evaluator;
//! an unobservable headroom keeps the serial reference; a batched tranche
//! reserves the row-task scratch once, else runs serially; the per-batch
//! line carries `engine= row_tasks= prepass_ms=` and the version marker.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use ndarray::{ArrayD, IxDyn};
use ny_levers::Source;
use ny_tensor::BoundedTensor;

use super::super::shared::MultiObjInputDomain;
use super::{
    run_child_zono_prepass, run_child_zono_prepass_scheduled, ChildZonoEvaluator,
    PRODUCTION_CHILD_ZONO_EVALUATOR,
};
use crate::beta_crown::config::BetaCrownConfig;
use crate::layers::{Layer, LinearLayer, SigmoidLayer};
use crate::zono_core::cgan::{cgan_child_zono_plan, CganChildZonoPlan};
use crate::zono_core::tests::{random_child, sample_points, small_cgan_graph, Rng};
use crate::zono_core::ZonoProgram;
use crate::{GraphNetwork, GraphNode};

fn armed_plan() -> CganChildZonoPlan {
    CganChildZonoPlan {
        max_generators: 16_384,
        max_secs: None,
        source: Source::Config,
    }
}

fn big_headroom() -> Option<u64> {
    Some(64 << 30)
}

fn tiny_headroom() -> Option<u64> {
    Some(1 << 20)
}

fn root_box() -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[5]), vec![-0.6, -0.4, 0.1, -0.9, 0.3]).expect("lo"),
        ArrayD::from_shape_vec(IxDyn(&[5]), vec![0.5, 0.35, 0.8, -0.2, 0.9]).expect("hi"),
    )
    .expect("root box")
}

fn domain(
    input: BoundedTensor,
    existing: Option<HashMap<String, BoundedTensor>>,
) -> MultiObjInputDomain {
    MultiObjInputDomain {
        input_bounds: Arc::new(input),
        obj_bounds: vec![(f32::NEG_INFINITY, f32::INFINITY)],
        linear_bounds: None,
        depth: 1,
        priority: 0.0,
        needs_bounding: true,
        node_bounds_override: existing.map(Arc::new),
        inherited_alpha_state: None,
    }
}

fn is_within(inner: &BoundedTensor, outer: &BoundedTensor) -> bool {
    inner.shape() == outer.shape()
        && inner
            .lower()
            .iter()
            .zip(outer.lower().iter())
            .all(|(i, o)| i >= o)
        && inner
            .upper()
            .iter()
            .zip(outer.upper().iter())
            .all(|(i, o)| i <= o)
}

fn bits(t: &BoundedTensor) -> (Vec<usize>, Vec<u32>, Vec<u32>) {
    (
        t.shape().to_vec(),
        t.lower().iter().map(|v| v.to_bits()).collect(),
        t.upper().iter().map(|v| v.to_bits()).collect(),
    )
}

#[test]
fn child_zono_is_dark_by_default() {
    let config = BetaCrownConfig::default();
    assert!(!config.alpha_config.cgan_child_zono_refresh.enabled);
    assert_eq!(
        config.alpha_config.cgan_child_zono_refresh.max_generators, 16_384,
        "the default budget must not bind (Z1: 1024 cost 83 leaves vs 15)"
    );
    assert!(cgan_child_zono_plan(&config).is_none());

    let graph = small_cgan_graph(71, 0);
    let root = root_box();
    let root_map = graph
        .collect_node_bounds_with_engine(&root, None)
        .expect("root map");
    let existing = Arc::new(root_map.clone());
    let mut domains = vec![
        domain(root.clone(), None),
        MultiObjInputDomain {
            node_bounds_override: Some(Arc::clone(&existing)),
            ..domain(root, None)
        },
    ];
    let result = run_child_zono_prepass(
        &config,
        &graph,
        Some(&root_map),
        None,
        &[0, 1],
        &mut domains,
    );
    assert!(result.is_none(), "a dark key must do no work");
    assert!(domains[0].node_bounds_override.is_none());
    assert!(Arc::ptr_eq(
        domains[1]
            .node_bounds_override
            .as_ref()
            .expect("override kept"),
        &existing
    ));

    let mut armed = BetaCrownConfig::default();
    armed.alpha_config.cgan_child_zono_refresh.enabled = true;
    let plan = cgan_child_zono_plan(&armed).expect("armed plan");
    assert_eq!(plan.source, Source::Config);
    assert_ne!(plan.source, Source::LegacyEnv);
    assert_eq!(plan.max_generators, 16_384);
}

#[test]
fn child_zono_merge_is_shrink_only() {
    let graph = small_cgan_graph(73, 0);
    let root = root_box();
    let root_map = graph
        .collect_node_bounds_with_engine(&root, None)
        .expect("root map");
    let program = ZonoProgram::compile_with_ibp_shapes(&graph, &root).expect("compile");
    let mut rng = Rng::new(74);
    let mut domains = Vec::new();
    for k in 0..6 {
        let child = random_child(&root, 1 + k, &mut rng);
        let existing = (k % 2 == 0).then(|| {
            graph
                .collect_node_bounds_with_engine(&child, None)
                .expect("child IBP map")
        });
        domains.push(domain(child, existing));
    }
    let before: Vec<Option<Arc<HashMap<String, BoundedTensor>>>> = domains
        .iter()
        .map(|d| d.node_bounds_override.clone())
        .collect();
    let indices: Vec<usize> = (0..domains.len()).collect();
    let batch = run_child_zono_prepass_scheduled(
        &armed_plan(),
        &graph,
        Some(&root_map),
        None,
        &indices,
        &mut domains,
        4,
        big_headroom,
        ChildZonoEvaluator::Serial,
    )
    .expect("armed prepass");
    assert_eq!(
        batch.tally().installed,
        domains.len(),
        "{:?}",
        batch.tally()
    );

    for (idx, dom) in domains.iter().enumerate() {
        let entries = batch.entries_for(idx).expect("installed entries");
        assert!(!entries.is_empty());
        if let Some(existing) = before[idx].as_ref() {
            let merged = dom.node_bounds_override.as_ref().expect("merged override");
            for (name, current) in existing.iter() {
                let candidate = merged.get(name).expect("no existing key may be dropped");
                assert!(
                    is_within(candidate, current),
                    "`{name}` widened by the merge"
                );
            }
        } else {
            assert!(dom.node_bounds_override.is_none());
        }
    }
    // The batched-rebound replacements never exceed the root entries.
    let overrides = batch
        .domain_overrides_for(&indices, Some(&root_map))
        .expect("replacements");
    for replacements in overrides.iter().flatten() {
        for (name, value) in replacements {
            assert!(
                is_within(value, &root_map[name]),
                "`{name}` wider than the root"
            );
        }
    }
    // And every published entry contains the sampled activations of its child.
    let steps = program.steps();
    for (idx, dom) in domains.iter().enumerate() {
        let entries = batch.entries_for(idx).expect("entries");
        let (points, rows) = sample_points(&dom.input_bounds, 2_000, idx as u64);
        let mut checked = 0usize;
        program
            .evaluate_points(&points, rows, |step, values| {
                let Some(s) = step else { return };
                let Some(entry) = entries.get(&steps[s].node) else {
                    return;
                };
                let lo: Vec<f32> = entry.lower().iter().copied().collect();
                let hi: Vec<f32> = entry.upper().iter().copied().collect();
                let n = lo.len();
                for r in 0..rows {
                    for j in 0..n {
                        let v = values[r * n + j];
                        assert!(
                            f64::from(lo[j]) <= v && v <= f64::from(hi[j]),
                            "domain {idx}: `{}`[{j}] = {v} outside [{}, {}]",
                            steps[s].node,
                            lo[j],
                            hi[j]
                        );
                        checked += 1;
                    }
                }
            })
            .expect("evaluate");
        assert!(checked > 0, "no published value was checked: VOID");
    }
}

#[test]
fn child_zono_disjoint_or_mismatch_keeps_current_entry() {
    let graph = small_cgan_graph(75, 0);
    let root = root_box();
    let root_map = graph
        .collect_node_bounds_with_engine(&root, None)
        .expect("root map");
    let mut rng = Rng::new(76);
    let child = random_child(&root, 3, &mut rng);
    let mut existing = graph
        .collect_node_bounds_with_engine(&child, None)
        .expect("child map");
    // `bn2` (first ReLU input): element 0 made DISJOINT from the true range.
    let bn2 = existing.get("bn2").expect("bn2 entry").clone();
    let mut lo = bn2.lower().clone();
    let mut hi = bn2.upper().clone();
    let anchor = hi.iter().next().copied().expect("element");
    *lo.iter_mut().next().expect("element") = anchor + 10.0;
    *hi.iter_mut().next().expect("element") = anchor + 11.0;
    let disjoint = BoundedTensor::new(lo, hi).expect("disjoint entry");
    existing.insert("bn2".to_string(), disjoint.clone());
    // `conv` (second ReLU input): replaced by a SHAPE-MISMATCHED entry.
    let conv = existing.get("conv").expect("conv entry").clone();
    let n = conv.len();
    let mismatched = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[n]), vec![-1.0e3; n]).expect("lo"),
        ArrayD::from_shape_vec(IxDyn(&[n]), vec![1.0e3; n]).expect("hi"),
    )
    .expect("mismatched entry");
    existing.insert("conv".to_string(), mismatched.clone());

    let mut domains = vec![domain(child, Some(existing))];
    let batch = run_child_zono_prepass_scheduled(
        &armed_plan(),
        &graph,
        Some(&root_map),
        None,
        &[0],
        &mut domains,
        1,
        big_headroom,
        ChildZonoEvaluator::Serial,
    )
    .expect("armed prepass");
    assert_eq!(batch.tally().installed, 1, "{:?}", batch.tally());
    let merged = domains[0].node_bounds_override.as_ref().expect("merged");
    // The mismatched entry is kept byte-for-byte.
    assert_eq!(bits(&merged["conv"]), bits(&mismatched));
    // The disjoint element is kept; the rest of the entry only shrinks.
    let m = &merged["bn2"];
    assert_eq!(
        m.lower().iter().next().copied().map(f32::to_bits),
        disjoint.lower().iter().next().copied().map(f32::to_bits)
    );
    assert_eq!(
        m.upper().iter().next().copied().map(f32::to_bits),
        disjoint.upper().iter().next().copied().map(f32::to_bits)
    );
    assert!(is_within(m, &disjoint));
    // The zonotope itself kept the looser (true) side, never the disjoint box.
    let published = &batch.entries_for(0).expect("entries")["bn2"];
    let first_hi = published.upper().iter().next().copied().expect("element");
    assert!(
        first_hi < anchor + 10.0,
        "the zonotope adopted a disjoint inherited element"
    );
}

#[test]
fn child_zono_parallel_bit_identical_to_serial() {
    let graph = small_cgan_graph(77, 1);
    let root = root_box();
    let root_map = graph
        .collect_node_bounds_with_engine(&root, None)
        .expect("root map");
    let mut rng = Rng::new(78);
    let mut serial = Vec::new();
    for k in 0..8 {
        let child = random_child(&root, k, &mut rng);
        let existing = (k % 3 == 0).then(|| {
            graph
                .collect_node_bounds_with_engine(&child, None)
                .expect("child map")
        });
        serial.push(domain(child, existing));
    }
    let mut parallel = serial.clone();
    let indices: Vec<usize> = (0..serial.len()).collect();
    let a = run_child_zono_prepass_scheduled(
        &armed_plan(),
        &graph,
        Some(&root_map),
        None,
        &indices,
        &mut serial,
        1,
        big_headroom,
        ChildZonoEvaluator::Serial,
    )
    .expect("serial prepass");
    let b = run_child_zono_prepass_scheduled(
        &armed_plan(),
        &graph,
        Some(&root_map),
        None,
        &indices,
        &mut parallel,
        4,
        big_headroom,
        ChildZonoEvaluator::Serial,
    )
    .expect("parallel prepass");
    assert_eq!(a.tally().width_max, 1);
    assert!(
        b.tally().width_max > 1,
        "the parallel arm never ran wide: VOID"
    );
    assert_eq!(a.tally().installed, serial.len());
    assert_eq!(a.tally().installed, b.tally().installed);
    for idx in indices {
        let (ea, eb) = (
            a.entries_for(idx).expect("serial entries"),
            b.entries_for(idx).expect("parallel entries"),
        );
        let mut keys: Vec<&String> = ea.keys().collect();
        keys.sort();
        let mut keys_b: Vec<&String> = eb.keys().collect();
        keys_b.sort();
        assert_eq!(keys, keys_b);
        for key in keys {
            assert_eq!(
                bits(&ea[key]),
                bits(&eb[key]),
                "domain {idx} `{key}` differs"
            );
        }
        match (
            &serial[idx].node_bounds_override,
            &parallel[idx].node_bounds_override,
        ) {
            (None, None) => {}
            (Some(x), Some(y)) => {
                for (key, value) in x.iter() {
                    assert_eq!(bits(value), bits(&y[key]), "override {idx} `{key}` differs");
                }
            }
            _ => panic!("domain {idx}: override presence differs between schedules"),
        }
    }
}

#[test]
fn child_zono_memory_admission_refuses_instead_of_running() {
    let graph = small_cgan_graph(79, 0);
    let root = root_box();
    let root_map = graph
        .collect_node_bounds_with_engine(&root, None)
        .expect("root map");
    let existing = Arc::new(root_map.clone());
    let mut domains = vec![
        domain(root.clone(), None),
        MultiObjInputDomain {
            node_bounds_override: Some(Arc::clone(&existing)),
            ..domain(root, None)
        },
    ];
    let batch = run_child_zono_prepass_scheduled(
        &armed_plan(),
        &graph,
        Some(&root_map),
        None,
        &[0, 1],
        &mut domains,
        4,
        tiny_headroom,
        ChildZonoEvaluator::Serial,
    )
    .expect("armed prepass");
    assert_eq!(batch.tally().refused_memory, 2, "{:?}", batch.tally());
    assert_eq!(batch.tally().installed, 0);
    assert!(batch.entries_for(0).is_none() && batch.entries_for(1).is_none());
    assert!(domains[0].node_bounds_override.is_none());
    assert!(Arc::ptr_eq(
        domains[1].node_bounds_override.as_ref().expect("kept"),
        &existing
    ));
}

#[test]
fn child_zono_program_refusal_leaves_domains_untouched() {
    let w = ndarray::Array2::from_shape_vec((3, 5), vec![0.1f32; 15]).expect("w");
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "fc",
        Layer::Linear(LinearLayer::new(w, None).expect("fc")),
    ));
    graph.add_node(GraphNode::new(
        "sig",
        Layer::Sigmoid(SigmoidLayer::new()),
        vec!["fc".to_string()],
    ));
    graph.set_output("sig");
    let mut domains = vec![domain(root_box(), None)];
    let result = run_child_zono_prepass_scheduled(
        &armed_plan(),
        &graph,
        None,
        None,
        &[0],
        &mut domains,
        4,
        big_headroom,
        ChildZonoEvaluator::Serial,
    );
    assert!(
        result.is_none(),
        "an unsupported graph must leave the batch on today's path"
    );
    assert!(domains[0].node_bounds_override.is_none());
}

// ------------------------------------------------ #m8-cgan-leaf-batch

/// Eight children of the root box, every third carrying its own IBP map as
/// an existing override (the merge path).
fn eight_children(graph: &GraphNetwork, seed: u64) -> Vec<MultiObjInputDomain> {
    let root = root_box();
    let mut rng = Rng::new(seed);
    (0..8)
        .map(|k| {
            let child = random_child(&root, k, &mut rng);
            let existing = (k % 3 == 0).then(|| {
                graph
                    .collect_node_bounds_with_engine(&child, None)
                    .expect("child map")
            });
            domain(child, existing)
        })
        .collect()
}

/// Both runs installed the same children with bitwise-identical entries and
/// overrides.
fn assert_same_publication(
    a: &super::ChildZonoBatch,
    b: &super::ChildZonoBatch,
    da: &[MultiObjInputDomain],
    db: &[MultiObjInputDomain],
) {
    assert_eq!(a.tally().installed, da.len(), "{:?}", a.tally());
    assert_eq!(a.tally().installed, b.tally().installed);
    for idx in 0..da.len() {
        let (ea, eb) = (
            a.entries_for(idx).expect("first entries"),
            b.entries_for(idx).expect("second entries"),
        );
        let mut keys: Vec<&String> = ea.keys().collect();
        keys.sort();
        let mut keys_b: Vec<&String> = eb.keys().collect();
        keys_b.sort();
        assert_eq!(keys, keys_b, "domain {idx}: different published nodes");
        for key in keys {
            assert_eq!(
                bits(&ea[key]),
                bits(&eb[key]),
                "domain {idx} `{key}` differs"
            );
        }
        match (&da[idx].node_bounds_override, &db[idx].node_bounds_override) {
            (None, None) => {}
            (Some(x), Some(y)) => {
                assert_eq!(x.len(), y.len());
                for (key, value) in x.iter() {
                    assert_eq!(bits(value), bits(&y[key]), "override {idx} `{key}` differs");
                }
            }
            _ => panic!("domain {idx}: override presence differs between evaluators"),
        }
    }
}

/// The batch evaluator (children concurrently, every row pass row-parallel)
/// publishes maps and overrides bitwise identical to the serial evaluator,
/// at tranche widths 1 and 4, and it demonstrably ran batched (VOID guard).
#[test]
fn child_zono_batch_evaluator_is_bit_identical_to_serial() {
    let graph = small_cgan_graph(81, 1);
    let root_map = graph
        .collect_node_bounds_with_engine(&root_box(), None)
        .expect("root map");
    for width in [1usize, 4] {
        let mut serial = eight_children(&graph, 82);
        let mut batched = serial.clone();
        let indices: Vec<usize> = (0..serial.len()).collect();
        let a = run_child_zono_prepass_scheduled(
            &armed_plan(),
            &graph,
            Some(&root_map),
            None,
            &indices,
            &mut serial,
            width,
            big_headroom,
            ChildZonoEvaluator::Serial,
        )
        .expect("serial prepass");
        let b = run_child_zono_prepass_scheduled(
            &armed_plan(),
            &graph,
            Some(&root_map),
            None,
            &indices,
            &mut batched,
            width,
            big_headroom,
            ChildZonoEvaluator::Batch,
        )
        .expect("batch prepass");
        assert_eq!(a.tally().engine, ChildZonoEvaluator::Serial);
        assert_eq!(a.tally().batch_tranches, 0);
        assert_eq!(a.tally().row_tasks, 0);
        assert_eq!(
            b.tally().engine,
            ChildZonoEvaluator::Batch,
            "the batch evaluator never ran: VOID"
        );
        assert_eq!(b.tally().batch_tranches, b.tally().tranches);
        assert!(b.tally().row_tasks > 0, "no row task ran: VOID");
        assert_eq!(a.tally().width_max, b.tally().width_max);
        assert_same_publication(&a, &b, &serial, &batched);
    }
}

fn no_headroom() -> Option<u64> {
    None
}

/// An unobservable headroom keeps today's schedule exactly (serial, one
/// child at a time, sequential rows) even under the batch evaluator.
#[test]
fn child_zono_unobserved_headroom_keeps_the_serial_reference() {
    let graph = small_cgan_graph(83, 0);
    let root_map = graph
        .collect_node_bounds_with_engine(&root_box(), None)
        .expect("root map");
    let mut serial = eight_children(&graph, 84);
    let mut batched = serial.clone();
    let indices: Vec<usize> = (0..serial.len()).collect();
    let a = run_child_zono_prepass_scheduled(
        &armed_plan(),
        &graph,
        Some(&root_map),
        None,
        &indices,
        &mut serial,
        4,
        no_headroom,
        ChildZonoEvaluator::Serial,
    )
    .expect("serial prepass");
    let b = run_child_zono_prepass_scheduled(
        &armed_plan(),
        &graph,
        Some(&root_map),
        None,
        &indices,
        &mut batched,
        4,
        no_headroom,
        ChildZonoEvaluator::Batch,
    )
    .expect("batch evaluator prepass");
    assert_eq!(b.tally().engine, ChildZonoEvaluator::Serial);
    assert_eq!(b.tally().batch_tranches, 0);
    assert_eq!(b.tally().row_tasks, 0);
    assert_eq!(b.tally().width_max, 1);
    assert_eq!(b.tally().tranches, serial.len());
    assert_same_publication(&a, &b, &serial, &batched);
}

static RESERVE_HEADROOM: AtomicU64 = AtomicU64::new(0);

fn reserve_headroom() -> Option<u64> {
    Some(RESERVE_HEADROOM.load(Ordering::SeqCst))
}

/// A batched tranche reserves the program's pool-wide row-task scratch once
/// on top of its children's estimates: with room for one child plus the
/// reserve every tranche runs batched (one child each); one byte short of
/// that, every tranche runs on the serial evaluator; the maps are identical
/// either way.
#[test]
fn child_zono_batched_tranche_reserves_the_row_task_scratch() {
    let graph = small_cgan_graph(85, 0);
    let root = root_box();
    let root_map = graph
        .collect_node_bounds_with_engine(&root, None)
        .expect("root map");
    let program = ZonoProgram::compile_with_ibp_shapes(&graph, &root).expect("compile");
    let plan = armed_plan();
    // Children WITHOUT overrides, every coordinate varying: one estimate.
    let mut rng = Rng::new(86);
    let children: Vec<MultiObjInputDomain> = (0..4)
        .map(|k| domain(random_child(&root, k, &mut rng), None))
        .collect();
    let unstable = program.unstable_upper_bounds(&[&root_map]);
    let estimate = program.estimate_peak_bytes(5, plan.max_generators, &unstable);
    let reserve = program.row_task_scratch_bytes(rayon::current_num_threads());
    assert!(
        reserve > 0,
        "the fixture's convolutions reserve nothing: VOID"
    );
    let indices: Vec<usize> = (0..children.len()).collect();

    RESERVE_HEADROOM.store(2 * (estimate + reserve), Ordering::SeqCst);
    let mut fits = children.clone();
    let a = run_child_zono_prepass_scheduled(
        &plan,
        &graph,
        Some(&root_map),
        None,
        &indices,
        &mut fits,
        4,
        reserve_headroom,
        ChildZonoEvaluator::Batch,
    )
    .expect("batched prepass");
    assert_eq!(
        a.tally().engine,
        ChildZonoEvaluator::Batch,
        "{:?}",
        a.tally()
    );
    assert_eq!(a.tally().batch_tranches, children.len());
    assert_eq!(
        a.tally().width_max,
        1,
        "the reserve must be counted once per tranche"
    );
    assert_eq!(a.tally().refused_memory, 0);

    RESERVE_HEADROOM.store(2 * (estimate + reserve - 1), Ordering::SeqCst);
    let mut short = children;
    let b = run_child_zono_prepass_scheduled(
        &plan,
        &graph,
        Some(&root_map),
        None,
        &indices,
        &mut short,
        4,
        reserve_headroom,
        ChildZonoEvaluator::Batch,
    )
    .expect("serial-fallback prepass");
    assert_eq!(
        b.tally().engine,
        ChildZonoEvaluator::Serial,
        "{:?}",
        b.tally()
    );
    assert_eq!(b.tally().batch_tranches, 0);
    assert_eq!(
        b.tally().refused_memory,
        0,
        "a child that fits alone must still run"
    );
    assert_same_publication(&a, &b, &fits, &short);
}

/// The production prepass runs the batch evaluator (whenever the headroom is
/// observable), and its per-batch line carries the new fields and the
/// version marker at the END of the fixed schema.
#[test]
fn child_zono_production_evaluator_is_batch_and_reports_its_line() {
    assert_eq!(PRODUCTION_CHILD_ZONO_EVALUATOR, ChildZonoEvaluator::Batch);
    let graph = small_cgan_graph(87, 1);
    let root_map = graph
        .collect_node_bounds_with_engine(&root_box(), None)
        .expect("root map");
    let mut domains = eight_children(&graph, 88);
    let indices: Vec<usize> = (0..domains.len()).collect();
    let mut armed = BetaCrownConfig::default();
    armed.alpha_config.cgan_child_zono_refresh.enabled = true;
    let plan = cgan_child_zono_plan(&armed).expect("armed plan");
    let batch = run_child_zono_prepass(
        &armed,
        &graph,
        Some(&root_map),
        None,
        &indices,
        &mut domains,
    )
    .expect("armed prepass");
    let tally = batch.tally();
    assert_eq!(tally.installed, domains.len(), "{tally:?}");
    if crate::network::crown_memory::process_memory_headroom_bytes().is_some() {
        assert_eq!(tally.engine, ChildZonoEvaluator::Batch, "{tally:?}");
        assert!(tally.batch_tranches > 0 && tally.row_tasks > 0, "{tally:?}");
    } else {
        assert_eq!(tally.engine, ChildZonoEvaluator::Serial, "{tally:?}");
    }
    let line = tally.render(1, &plan, None);
    assert!(
        line.starts_with("[cgan-child-zono] armed=1 children="),
        "{line}"
    );
    let tail = format!(
        " max_generators={} engine={} row_tasks={} prepass_ms=",
        plan.max_generators,
        tally.engine.token(),
        tally.row_tasks
    );
    assert!(line.contains(&tail), "{line}");
    assert!(line.ends_with(" m8-cgan-leaf-batch/v1"), "{line}");
    assert!(tally.prepass_nanos > 0, "{tally:?}");
}
