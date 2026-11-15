// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! M8a (`#m8-input-split-batch`): the lowdim lane's two typed keys
//! (`bab.branching.input_split.lowdim_batch_engine` and
//! `bab.branching.input_split.engine`) — their tokens are exactly the
//! declaration's, they are dark by default and arrive through the scored
//! run's own preset projection with `Source::Config` — and the REAL ny-bbe
//! CPU f32 backend, registered exactly as `main.rs` registers it, reaching
//! the armed lane: it batches, the lane certifies every close, and a planted
//! violation is never verified.

use std::sync::Arc;
use std::time::{Duration, Instant};

use ndarray::{Array2, ArrayD, IxDyn};
use ny_propagate::layers::{LinearLayer, ReLULayer};
use ny_propagate::lowdim::{
    self, InputSplitEngine, LowDimBatchEngine, LowDimConfig, LowDimOutcome, LowDimStop,
};
use ny_propagate::network::GraphNetwork;
use ny_propagate::{Layer, Network};
use ny_tensor::BoundedTensor;

use super::preset_delivered_config_value;

fn projected_levers(preset: &crate::preset::PresetConfig) -> ny_levers::LeverSet {
    let serialized = serde_json::to_value(preset).expect("preset serializes");
    let captured = ny_levers::RawLeverInputs::capture_with(ny_levers::all(), |_| None);
    ny_levers::LeverSet::resolve_layered(ny_levers::all(), &captured, |decl| {
        preset_delivered_config_value(&serialized, decl)
    })
    .expect("the projected preset must match its declarations")
}

fn declared_tokens(decl: &ny_levers::LeverDecl) -> Vec<&'static str> {
    match decl.kind {
        ny_levers::LeverKind::Enum(tokens) => tokens.to_vec(),
        other => panic!("{} must be an Enum lever, is {other:?}", decl.name),
    }
}

fn axis_tokens(name: &str) -> Vec<&'static str> {
    let axis = ny_levers::space::axes()
        .iter()
        .find(|a| a.name == name)
        .unwrap_or_else(|| panic!("{name} has a search-space axis"));
    match axis.domain {
        ny_levers::space::Domain::Enum(tokens) => tokens.to_vec(),
        ref other => panic!("{name}'s axis must be an Enum, is {other:?}"),
    }
}

#[test]
fn lowdim_batch_engine_tokens_are_exactly_the_declaration() {
    let decl = &ny_levers::decls::lowdim_lane::LOWDIM_BATCH_ENGINE;
    let ours: Vec<&str> = [LowDimBatchEngine::Legacy, LowDimBatchEngine::Bbe]
        .iter()
        .map(|e| e.token())
        .collect();
    assert_eq!(ours, declared_tokens(decl));
    assert_eq!(ours, axis_tokens(decl.name));
    for e in [LowDimBatchEngine::Legacy, LowDimBatchEngine::Bbe] {
        assert_eq!(LowDimBatchEngine::from_token(e.token()), Some(e));
        let yaml: LowDimBatchEngine = serde_yaml::from_str(e.token()).expect("serde token");
        assert_eq!(yaml, e);
    }
    assert_eq!(LowDimBatchEngine::from_token("cpu_f32"), None);
    assert_eq!(LowDimBatchEngine::default(), LowDimBatchEngine::Legacy);
    assert_eq!(
        decl.default,
        ny_levers::DefaultSpec::Text(LowDimBatchEngine::default().token())
    );
}

#[test]
fn input_split_engine_tokens_are_exactly_the_declaration() {
    let decl = &ny_levers::decls::lowdim_lane::INPUT_SPLIT_ENGINE;
    let ours: Vec<&str> = [InputSplitEngine::Legacy, InputSplitEngine::Bbe]
        .iter()
        .map(|e| e.token())
        .collect();
    assert_eq!(ours, declared_tokens(decl));
    assert_eq!(ours, axis_tokens(decl.name));
    for e in [InputSplitEngine::Legacy, InputSplitEngine::Bbe] {
        assert_eq!(InputSplitEngine::from_token(e.token()), Some(e));
        let yaml: InputSplitEngine = serde_yaml::from_str(e.token()).expect("serde token");
        assert_eq!(yaml, e);
    }
    assert_eq!(InputSplitEngine::from_token("graph"), None);
    assert_eq!(InputSplitEngine::default(), InputSplitEngine::Legacy);
    assert_eq!(
        decl.default,
        ny_levers::DefaultSpec::Text(InputSplitEngine::default().token())
    );
}

#[test]
fn lowdim_m8_keys_resolve_from_the_frozen_lever_set() {
    use crate::commands::beta_crown::{resolve_input_split_engine, resolve_lowdim_batch_engine};
    // Dark: absent keys resolve at the declaration default, and the lane
    // config is the historical one.
    let dark = projected_levers(&crate::preset::PresetConfig::default());
    for decl in [
        &ny_levers::decls::lowdim_lane::LOWDIM_BATCH_ENGINE,
        &ny_levers::decls::lowdim_lane::INPUT_SPLIT_ENGINE,
    ] {
        let r = dark.resolved(decl).expect("declared");
        assert_eq!(r.value.as_str(), Some("legacy"), "{}", decl.name);
        assert_eq!(r.source, ny_levers::Source::Default);
    }
    assert_eq!(
        resolve_lowdim_batch_engine(Some(&dark), None),
        LowDimBatchEngine::Legacy
    );
    assert_eq!(
        resolve_input_split_engine(Some(&dark), None, false),
        InputSplitEngine::Legacy
    );
    // Armed through the scored projection: Source::Config, never LegacyEnv,
    // and the resolvers deliver into BetaCrownConfig -> LowDimConfig.
    let mut preset = crate::preset::PresetConfig::default();
    preset.bab.branching.input_split.lowdim_batch_engine = Some(LowDimBatchEngine::Bbe);
    preset.bab.branching.input_split.engine = Some(InputSplitEngine::Bbe);
    let armed = projected_levers(&preset);
    for decl in [
        &ny_levers::decls::lowdim_lane::LOWDIM_BATCH_ENGINE,
        &ny_levers::decls::lowdim_lane::INPUT_SPLIT_ENGINE,
    ] {
        let r = armed.resolved(decl).expect("declared");
        assert_eq!(r.value.as_str(), Some("bbe"), "{}", decl.name);
        assert_eq!(r.source, ny_levers::Source::Config);
        assert_ne!(r.source, ny_levers::Source::LegacyEnv);
    }
    let config = ny_propagate::BetaCrownConfig {
        input_split_lowdim_batch_engine: resolve_lowdim_batch_engine(Some(&armed), None),
        input_split_engine: resolve_input_split_engine(Some(&armed), None, false),
        ..Default::default()
    };
    let lane = config.lowdim_config();
    assert_eq!(lane.batch_engine, LowDimBatchEngine::Bbe);
    assert_eq!(lane.engine, InputSplitEngine::Bbe);
    assert!(
        lowdim::fold_shared(&lane),
        "engine bbe implies the shared fold"
    );
    // Everything else stays historical.
    let historical = LowDimConfig::default();
    assert_eq!(lane.batch, historical.batch);
    assert_eq!(lane.split, historical.split);
    assert!(!lane.shared_worklist && !lane.clip && !lane.varying_dims);
    // The interactive route: the typed preset over an empty environment.
    assert_eq!(
        resolve_lowdim_batch_engine(None, Some(&preset)),
        LowDimBatchEngine::Bbe
    );
    assert_eq!(
        resolve_input_split_engine(None, Some(&preset), true),
        InputSplitEngine::Bbe
    );
}

/// Deterministic LCG in `[-1, 1)`.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

/// `3 -> 10 -> ReLU -> 10 -> ReLU -> 2` on `[-1, 1]^3` (f32 endpoints).
fn tiny_net() -> (GraphNetwork, BoundedTensor) {
    let mut rng = Lcg(2026);
    let mut net = Network::new();
    for (i, (o, n)) in [(10usize, 3usize), (10, 10), (2, 10)]
        .into_iter()
        .enumerate()
    {
        let w = Array2::from_shape_fn((o, n), |_| (rng.next() * 0.9) as f32);
        let b = ndarray::Array1::from_shape_fn(o, |_| (rng.next() * 0.2) as f32);
        net.add_layer(Layer::Linear(LinearLayer::new(w, Some(b)).expect("linear")));
        if i < 2 {
            net.add_layer(Layer::ReLU(ReLULayer::new()));
        }
    }
    let graph = GraphNetwork::from_sequential(&net).expect("converts");
    let lo = ArrayD::from_elem(IxDyn(&[1, 3]), -1.0f32);
    let hi = ArrayD::from_elem(IxDyn(&[1, 3]), 1.0f32);
    (graph, BoundedTensor::new(lo, hi).expect("box"))
}

/// Sampled `(min, max)` of `y0 - y1` over the box (the lane's own f64
/// lowering; corners included).
fn sampled_range(graph: &GraphNetwork, input: &BoundedTensor) -> (f64, f64) {
    let admission = lowdim::admit(graph, input).expect("admits");
    let mut rng = Lcg(7);
    let (mut mn, mut mx) = (f64::INFINITY, f64::NEG_INFINITY);
    for i in 0..6000 {
        let x: Vec<f64> = (0..3)
            .map(|k| {
                if i < 8 {
                    if (i >> k) & 1 == 1 {
                        1.0
                    } else {
                        -1.0
                    }
                } else {
                    rng.next()
                }
            })
            .collect();
        let y = lowdim::lowered_point_values(&admission, &x).expect("evaluates");
        let v = y[0] - y[1];
        mn = mn.min(v);
        mx = mx.max(v);
    }
    (mn, mx)
}

/// Register the process backend exactly as `main.rs` does (first install
/// wins; every test here installs the same one).
fn register_cpu_f32() {
    lowdim::set_forward_linear_search_backend_factory(|| {
        Some(Arc::new(ny_bbe::CpuF32Backend::new()) as Arc<dyn ny_core::search::SearchBackend>)
    });
}

fn armed() -> LowDimConfig {
    LowDimConfig {
        batch: 64,
        chunk: 8,
        batch_engine: LowDimBatchEngine::Bbe,
        ..LowDimConfig::default()
    }
}

fn field<'a>(line: &'a str, key: &str) -> &'a str {
    line.split_whitespace()
        .find_map(|t| t.strip_prefix(&format!("{key}=")))
        .unwrap_or_else(|| panic!("no {key}= in {line}"))
}

fn prove(thr: f64, cfg: &LowDimConfig) -> LowDimOutcome {
    let (graph, input) = tiny_net();
    lowdim::prove_objective_lower_bound(
        &graph,
        &input,
        &[1.0f32, -1.0],
        thr as f32,
        Instant::now() + Duration::from_secs(60),
        cfg,
    )
}

#[test]
fn registered_cpu_f32_backend_reaches_the_armed_lowdim_lane() {
    register_cpu_f32();
    assert!(lowdim::forward_linear_search_backend_registered());
    let (graph, input) = tiny_net();
    let (mn, mx) = sampled_range(&graph, &input);
    let _ = lowdim::take_last_bbe_summary();
    let outcome = prove(mn - 0.1 * (mx - mn), &armed());
    let line = lowdim::take_last_bbe_summary().expect("the armed lane printed its [bbe] line");
    assert!(
        line.starts_with("[bbe] phase=input_split lane=lowdim scheduler=single "),
        "{line}"
    );
    assert_eq!(field(&line, "backend"), "cpu_f32", "{line}");
    assert_eq!(field(&line, "probe"), "ok", "{line}");
    assert_eq!(field(&line, "fallback"), "none", "{line}");
    assert!(
        field(&line, "calls").parse::<u64>().expect("calls") >= 2,
        "{line}"
    );
    assert!(line.ends_with("m8-input-split-batch/v1"), "{line}");
    assert!(matches!(outcome, LowDimOutcome::Verified(_)), "{outcome:?}");
    // Dark: the same lane unarmed prints no [bbe] line.
    let _ = prove(mn - 0.1 * (mx - mn), &LowDimConfig::default());
    assert!(lowdim::take_last_bbe_summary().is_none());
}

#[test]
fn real_cpu_f32_lowdim_bbe_proves_a_tiny_net_and_certifies_every_close() {
    register_cpu_f32();
    let (graph, input) = tiny_net();
    let (mn, mx) = sampled_range(&graph, &input);
    let legacy = prove(
        mn - 0.05 * (mx - mn),
        &LowDimConfig {
            batch: 64,
            chunk: 8,
            ..LowDimConfig::default()
        },
    );
    let searched = prove(mn - 0.05 * (mx - mn), &armed());
    let line = lowdim::take_last_bbe_summary().expect("line");
    let (LowDimOutcome::Verified(l), LowDimOutcome::Verified(s)) = (&legacy, &searched) else {
        panic!("both must verify a true margin: legacy {legacy:?} searched {searched:?}");
    };
    assert_eq!(field(&line, "fallback"), "none", "{line}");
    assert!(
        field(&line, "batched_max").parse::<u64>().expect("n") > 1,
        "{line}"
    );
    assert!(s.bbe_certified > 0, "VOID: nothing certified ({line})");
    // Every close of the searched run is a claim the lane's f64 bound closed
    // (clip is off: there is no other way to close).
    assert_eq!(s.closed, s.bbe_certified, "{s:?}");
    assert!(s.bbe_claimed >= s.bbe_certified);
    assert!((s.closed_volume - 1.0).abs() < 1e-9, "{s:?}");
    assert!(l.domains > 0);
}

#[test]
fn real_cpu_f32_lowdim_bbe_never_verifies_a_sat_instance() {
    register_cpu_f32();
    let (graph, input) = tiny_net();
    let (mn, mx) = sampled_range(&graph, &input);
    for frac in [0.001, 0.02, 0.3] {
        let out = prove(mn + frac * (mx - mn), &armed());
        match out {
            LowDimOutcome::Unproved(_, stop) => assert_ne!(stop, LowDimStop::NonFinite, "{frac}"),
            other => panic!("SOUNDNESS: a sampled-violated threshold ({frac}): {other:?}"),
        }
        let line = lowdim::take_last_bbe_summary().expect("line");
        assert_eq!(field(&line, "fallback"), "none", "{line}");
    }
}
