// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! CPU point advice from the existing original-tower VJP templates.
//! Outputs and analytic adjoints steer a search only. They cannot authorize
//! SAT or UNSAT; the caller retains the original dual-source replay gate.

use super::*;
use ndarray::{ArrayView1, ArrayView2};
use ny_core::GpuCrownLayer;
use ny_propagate::PointVjpBatchPlan;
use ny_tensor::BoundedTensor;
use std::sync::atomic::{AtomicUsize, Ordering};

pub(super) struct DenseTwinAdvice {
    f: PointVjpBatchPlan,
    g: PointVjpBatchPlan,
    forward_pairs: AtomicUsize,
    analytic_vjps: AtomicUsize,
}

fn dense_template(plan: &PointVjpBatchPlan) -> bool {
    if plan.input_dim == 0 || plan.output_dim == 0 {
        return false;
    }
    let mut width = plan.output_dim;
    for (index, layer) in plan.layers_backward.iter().enumerate() {
        match layer {
            GpuCrownLayer::Linear {
                weight,
                bias,
                out_features,
                in_features,
                ..
            } => {
                if *out_features != width
                    || *in_features == 0
                    || out_features.checked_mul(*in_features) != Some(weight.len())
                    || bias
                        .as_ref()
                        .is_some_and(|bias| bias.len() != *out_features)
                    || !weight.iter().all(|x| x.is_finite())
                    || bias
                        .as_ref()
                        .is_some_and(|bias| !bias.iter().all(|x| x.is_finite()))
                {
                    return false;
                }
                width = *in_features;
            }
            GpuCrownLayer::Activation {
                lower_slope,
                upper_slope,
                lower_intercept,
                upper_intercept,
                num_neurons,
                ..
            } => {
                if *num_neurons != width {
                    return false;
                }
                if !plan.mask_positions.contains(&index)
                    && (lower_slope.len() != width
                        || lower_intercept.len() != width
                        || lower_slope != upper_slope
                        || lower_intercept != upper_intercept
                        || !lower_slope
                            .iter()
                            .chain(lower_intercept.iter())
                            .all(|x| x.is_finite()))
                {
                    return false;
                }
            }
            _ => return false,
        }
    }
    width == plan.input_dim
        && plan
            .mask_positions
            .iter()
            .enumerate()
            .all(|(slot, &index)| {
                !plan.mask_positions[..slot].contains(&index)
                    && matches!(
                        plan.layers_backward.get(index),
                        Some(GpuCrownLayer::Activation { .. })
                    )
            })
}

fn reverse(plan: &PointVjpBatchPlan, masks: &[Vec<f32>], mut seed: Vec<f32>) -> Option<Vec<f32>> {
    if seed.len() != plan.output_dim || masks.len() != plan.mask_positions.len() {
        return None;
    }
    for (index, layer) in plan.layers_backward.iter().enumerate() {
        match layer {
            GpuCrownLayer::Linear {
                weight,
                out_features,
                in_features,
                ..
            } => {
                if seed.len() != *out_features {
                    return None;
                }
                let matrix =
                    ArrayView2::from_shape((*out_features, *in_features), weight.as_ref()).ok()?;
                seed = matrix
                    .t()
                    .dot(&ArrayView1::from(&seed[..]))
                    .into_raw_vec_and_offset()
                    .0;
            }
            GpuCrownLayer::Activation {
                lower_slope,
                num_neurons,
                ..
            } => {
                if seed.len() != *num_neurons {
                    return None;
                }
                let slope = match plan
                    .mask_positions
                    .iter()
                    .position(|&position| position == index)
                {
                    Some(slot) => masks.get(slot)?,
                    None => lower_slope,
                };
                if slope.len() != seed.len() {
                    return None;
                }
                for (value, &slope) in seed.iter_mut().zip(slope.iter()) {
                    *value *= slope;
                }
            }
            _ => return None,
        }
    }
    (seed.len() == plan.input_dim && seed.iter().all(|x| x.is_finite())).then_some(seed)
}

// The race's shared_device=false fact must also cover setup. Reject every
// non-dense/affine node before shape discovery can enter a different backend.
fn cpu_dense_tower(graph: &GraphNetwork, stopped: impl Fn() -> bool) -> bool {
    use ny_propagate::Layer;
    !graph.node_names().is_empty()
        && graph.node_names().iter().all(|name| {
            !stopped()
                && graph.node(name).is_some_and(|node| {
                    node.inputs().len() == 1
                        && matches!(
                            node.layer(),
                            Layer::Linear(_)
                                | Layer::ReLU(_)
                                | Layer::Flatten(_)
                                | Layer::Reshape(_)
                                | Layer::AddConstant(_)
                                | Layer::SubConstant(_)
                                | Layer::MulConstant(_)
                                | Layer::DivConstant(_)
                        )
                })
        })
}

#[cfg(test)]
thread_local! {
    static POINT_PLAN_PROBES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static POINT_PAIR_RUNS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(super) fn test_reset_work() {
    POINT_PLAN_PROBES.with(|count| count.set(0));
    POINT_PAIR_RUNS.with(|count| count.set(0));
}

#[cfg(test)]
pub(super) fn test_work() -> (usize, usize) {
    (
        POINT_PLAN_PROBES.with(std::cell::Cell::get),
        POINT_PAIR_RUNS.with(std::cell::Cell::get),
    )
}

fn compile_cpu_point_plan(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    deadline: std::time::Instant,
) -> Option<PointVjpBatchPlan> {
    #[cfg(test)]
    POINT_PLAN_PROBES.with(|count| count.set(count.get() + 1));
    // On the admitted layer fragment, finite Linear IBP uses the existing
    // pollable CPU contraction; affine/shape/ReLU steps have no accelerator.
    graph.build_point_vjp_batch_plan_with_deadline(input, Some(deadline))
}

impl DenseTwinAdvice {
    pub(super) fn build(
        f: &GraphNetwork,
        g: &GraphNetwork,
        bounds: &[Bound],
        deadline: std::time::Instant,
    ) -> Option<Self> {
        Self::build_cancellable(f, g, bounds, deadline, None)
    }

    pub(super) fn build_cancellable(
        f: &GraphNetwork,
        g: &GraphNetwork,
        bounds: &[Bound],
        deadline: std::time::Instant,
        cancel: Option<&ny_propagate::PgdCancelToken>,
    ) -> Option<Self> {
        let stopped = || {
            std::time::Instant::now() >= deadline
                || cancel.is_some_and(ny_propagate::PgdCancelToken::is_cancelled)
        };
        if stopped() || !cpu_dense_tower(f, &stopped) || !cpu_dense_tower(g, &stopped) {
            return None;
        }
        let input = Verifier::bounds_to_tensor(bounds, None).ok()?;
        let f = compile_cpu_point_plan(f, &input, deadline)?;
        if stopped() {
            return None;
        }
        let g = compile_cpu_point_plan(g, &input, deadline)?;
        if stopped() {
            return None;
        }
        if f.input_dim != g.input_dim
            || f.output_dim != g.output_dim
            || !dense_template(&f)
            || !dense_template(&g)
        {
            return None;
        }
        Some(Self {
            f,
            g,
            forward_pairs: AtomicUsize::new(0),
            analytic_vjps: AtomicUsize::new(0),
        })
    }

    pub(super) fn forward(&self, point: &[f32]) -> Option<Vec<f32>> {
        if !point.iter().all(|x| x.is_finite()) {
            return None;
        }
        #[cfg(test)]
        POINT_PAIR_RUNS.with(|count| count.set(count.get() + 1));
        self.forward_pairs.fetch_add(1, Ordering::Relaxed);
        let f = self.f.forward_point(point).ok()?;
        let g = self.g.forward_point(point).ok()?;
        if f.len() != g.len() {
            return None;
        }
        let difference: Vec<_> = f.iter().zip(g.iter()).map(|(&f, &g)| f - g).collect();
        difference
            .iter()
            .all(|x| x.is_finite())
            .then_some(difference)
    }

    /// Adjoint of the stored affine/ReLU template's signed output difference.
    /// This is an analytic search direction, not a derivative theorem about
    /// the discontinuous original Float32 implementation.
    pub(super) fn signed_gradient(
        &self,
        point: &[f32],
        output: usize,
        sign: f32,
    ) -> Option<Vec<f32>> {
        if output >= self.f.output_dim
            || (sign != -1.0 && sign != 1.0)
            || !point.iter().all(|x| x.is_finite())
        {
            return None;
        }
        #[cfg(test)]
        POINT_PAIR_RUNS.with(|count| count.set(count.get() + 1));
        self.forward_pairs.fetch_add(1, Ordering::Relaxed);
        let (f_masks, f_output) = self.f.forward_point_masks(point).ok()?;
        let (g_masks, g_output) = self.g.forward_point_masks(point).ok()?;
        if !f_output
            .iter()
            .chain(g_output.iter())
            .all(|x| x.is_finite())
        {
            return None;
        }
        let mut f_seed = vec![0.0; self.f.output_dim];
        let mut g_seed = vec![0.0; self.g.output_dim];
        f_seed[output] = sign;
        g_seed[output] = -sign;
        let f = reverse(&self.f, &f_masks, f_seed)?;
        let g = reverse(&self.g, &g_masks, g_seed)?;
        let result: Vec<_> = f.iter().zip(g.iter()).map(|(&f, &g)| f + g).collect();
        if !result.iter().all(|x| x.is_finite()) {
            return None;
        }
        self.analytic_vjps.fetch_add(1, Ordering::Relaxed);
        Some(result)
    }

    pub(super) fn receipt(&self) {
        eprintln!("[relational-vjp] done forward_pairs={} analytic_vjps={} original_dual_gate_unchanged=1", self.forward_pairs.load(Ordering::Relaxed), self.analytic_vjps.load(Ordering::Relaxed));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};
    use ny_propagate::layers::{
        AddLayer, DivConstantLayer, LinearLayer, ReLULayer, SubConstantLayer,
    };
    use ny_propagate::{GraphNode, Layer};
    use std::time::{Duration, Instant};

    fn deadline() -> Instant {
        Instant::now() + Duration::from_secs(10)
    }

    fn scalar(a: f32, b: f32, c: f32, d: f32) -> GraphNetwork {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "first",
            Layer::Linear(LinearLayer::new(arr2(&[[a]]), Some(arr1(&[b]))).unwrap()),
        ));
        graph.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer),
            vec!["first".into()],
        ));
        graph.add_node(GraphNode::new(
            "output",
            Layer::Linear(LinearLayer::new(arr2(&[[c]]), Some(arr1(&[d]))).unwrap()),
            vec!["relu".into()],
        ));
        graph.set_output("output");
        graph
    }

    fn pair() -> (GraphNetwork, GraphNetwork) {
        let tower = |perturbed: bool| {
            let mut graph = GraphNetwork::new();
            graph.add_node(GraphNode::from_input(
                "shift",
                Layer::SubConstant(SubConstantLayer::new(arr1(&[0.25, -0.5]).into_dyn())),
            ));
            graph.add_node(GraphNode::new(
                "scale",
                Layer::DivConstant(DivConstantLayer::new(arr1(&[2.0, 4.0]).into_dyn())),
                vec!["shift".into()],
            ));
            let (w, b) = if perturbed {
                (
                    arr2(&[[2.125, 0.875], [-1.0625, 2.75]]),
                    arr1(&[-0.125, 0.25]),
                )
            } else {
                (arr2(&[[2.0, 1.0], [-1.0, 3.0]]), arr1(&[-0.25, 0.125]))
            };
            graph.add_node(GraphNode::new(
                "first",
                Layer::Linear(LinearLayer::new(w, Some(b)).unwrap()),
                vec!["scale".into()],
            ));
            graph.add_node(GraphNode::new(
                "relu",
                Layer::ReLU(ReLULayer),
                vec!["first".into()],
            ));
            let w = if perturbed {
                arr2(&[[0.875, 0.625], [-0.5, 0.9375], [0.25, -0.375]])
            } else {
                arr2(&[[1.0, 0.5], [-0.25, 1.0], [0.125, -0.5]])
            };
            graph.add_node(GraphNode::new(
                "output",
                Layer::Linear(LinearLayer::new(w, Some(arr1(&[0.125, -0.25, 0.375]))).unwrap()),
                vec!["relu".into()],
            ));
            graph.set_output("output");
            graph
        };
        (tower(false), tower(true))
    }

    #[test]
    fn prepared_forward_matches_both_original_graphs_with_constant_normalization() {
        let (f, g) = pair();
        let bounds = [Bound::new(-2.0, 2.0); 2];
        let advice = DenseTwinAdvice::build(&f, &g, &bounds, deadline()).expect("dense pair");
        for point in [[-1.5, -0.75], [0.25, 0.75], [1.5, -0.5], [1.5, 1.25]] {
            let actual = advice.forward(&point).unwrap();
            let f = forward_point_vec(&f, &point).unwrap();
            let g = forward_point_vec(&g, &point).unwrap();
            for ((&actual, &f), &g) in actual.iter().zip(f.iter()).zip(g.iter()) {
                assert!(
                    (actual - (f - g)).abs() <= 1e-5 * (1.0 + f.abs() + g.abs()),
                    "actual={actual} original_f={f} original_g={g}"
                );
            }
        }
        assert_eq!(advice.forward_pairs.load(Ordering::Relaxed), 4);
    }

    #[test]
    fn signed_adjoint_matches_hand_computed_piecewise_slopes() {
        let f = scalar(2.0, -1.0, 3.0, 0.125);
        let g = scalar(1.0, 1.0, -2.0, 0.375);
        let advice = DenseTwinAdvice::build(&f, &g, &[Bound::new(-2.0, 2.0)], deadline()).unwrap();
        for (point, slope) in [(-1.5, 0.0), (-0.5, 2.0), (0.25, 2.0), (1.0, 8.0)] {
            for sign in [-1.0, 1.0] {
                assert_eq!(
                    advice.signed_gradient(&[point], 0, sign).unwrap(),
                    vec![sign * slope]
                );
            }
        }
        assert_eq!(advice.analytic_vjps.load(Ordering::Relaxed), 8);
    }

    #[test]
    fn multi_output_adjoint_matches_independent_graph_finite_differences() {
        let (f, g) = pair();
        let advice =
            DenseTwinAdvice::build(&f, &g, &[Bound::new(-2.0, 2.0); 2], deadline()).unwrap();
        for point in [[-1.5, -0.75], [0.25, 0.75], [1.5, -0.5], [1.5, 1.25]] {
            for output in 0..3 {
                for sign in [-1.0, 1.0] {
                    let gradient = advice.signed_gradient(&point, output, sign).unwrap();
                    for axis in 0..2 {
                        let h = 1.0 / 1024.0;
                        let mut plus = point;
                        let mut minus = point;
                        plus[axis] += h;
                        minus[axis] -= h;
                        let fp = forward_point_vec(&f, &plus).unwrap()[output];
                        let gp = forward_point_vec(&g, &plus).unwrap()[output];
                        let fm = forward_point_vec(&f, &minus).unwrap()[output];
                        let gm = forward_point_vec(&g, &minus).unwrap()[output];
                        let numerical = sign * ((fp - gp) - (fm - gm)) / (2.0 * h);
                        assert!((gradient[axis]-numerical).abs()<=2e-3*(1.0+numerical.abs()),"point={point:?} output={output} sign={sign} axis={axis} analytic={} numerical={numerical}",gradient[axis]);
                    }
                }
            }
        }
        assert_eq!(advice.analytic_vjps.load(Ordering::Relaxed), 24);
    }

    #[test]
    fn malformed_or_nonfinite_advice_inputs_are_refused() {
        let f = scalar(2.0, -1.0, 3.0, 0.125);
        let g = scalar(1.0, 1.0, -2.0, 0.375);
        let advice = DenseTwinAdvice::build(&f, &g, &[Bound::new(-2.0, 2.0)], deadline()).unwrap();
        for point in [
            vec![],
            vec![1.0, 2.0],
            vec![f32::NAN],
            vec![f32::INFINITY],
            vec![f32::MAX],
        ] {
            assert!(advice.forward(&point).is_none(), "{point:?}");
            assert!(
                advice.signed_gradient(&point, 0, 1.0).is_none(),
                "{point:?}"
            );
        }
        for sign in [0.0, f32::NAN, f32::INFINITY] {
            assert!(advice.signed_gradient(&[1.0], 0, sign).is_none());
        }
        assert!(advice.signed_gradient(&[1.0], 1, 1.0).is_none());
        assert!(reverse(&advice.f, &[], vec![1.0]).is_none());
        assert!(reverse(&advice.f, &[vec![1.0]], vec![]).is_none());
    }

    #[test]
    fn unsupported_fan_in_preserves_legacy_point_search() {
        let mut f = scalar(0.0, 1.0, 1.0, 0.0);
        let g = scalar(0.0, 0.0, 0.0, 0.0);
        f.add_node(GraphNode::binary(
            "sum",
            Layer::Add(AddLayer),
            "output",
            "output",
        ));
        f.set_output("sum");
        let bounds = [Bound::new(-1.0, 1.0)];
        assert!(DenseTwinAdvice::build(&f, &g, &bounds, deadline()).is_none());
        let diff = build_difference_network(&f, &g).unwrap();
        let candidate = search_isomorphic_deviation(&diff, &bounds, 0.5, deadline())
            .expect("unchanged fallback finds the planted violation");
        assert!(forward_point_vec(&diff, &candidate).unwrap()[0].abs() > 0.5);
    }

    #[test]
    fn absolute_deadline_prevents_template_setup_and_search() {
        let f = scalar(0.0, 0.0, 0.0, 0.0);
        let g = scalar(0.0, 1.0, 1.0, 0.0);
        let bounds = [Bound::new(-1.0, 1.0)];
        assert!(DenseTwinAdvice::build(
            &f,
            &g,
            &bounds,
            Instant::now()
                .checked_sub(Duration::from_millis(1))
                .expect("test clock supports an expired deadline")
        )
        .is_none());
        let advice = DenseTwinAdvice::build(&f, &g, &bounds, deadline()).unwrap();
        let diff = build_difference_network(&f, &g).unwrap();
        assert!(search_isomorphic_deviation_with_advice(
            &diff,
            &bounds,
            0.5,
            Instant::now()
                .checked_sub(Duration::from_millis(1))
                .expect("test clock supports an expired deadline"),
            Some(&advice)
        )
        .is_none());
        assert_eq!(advice.forward_pairs.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn typed_policy_projection_is_opt_in() {
        let parse = |text| serde_yaml::from_str::<crate::preset::PresetConfig>(text).unwrap();
        assert!(!RelationalPresetKeys::default().relational_point_vjp);
        assert!(!BetaCrownConfig::default().relational_point_vjp);
        for text in ["{}\n", "attack:\n  relational_point_vjp: false\n"] {
            let preset = parse(text);
            assert!(!RelationalPresetKeys::from_preset(&preset).relational_point_vjp);
            let mut config = BetaCrownConfig::default();
            crate::preset::apply_preset(&mut config, &preset).unwrap();
            assert!(!config.relational_point_vjp);
        }
        let armed = parse("attack:\n  relational_point_vjp: true\n");
        assert!(RelationalPresetKeys::from_preset(&armed).relational_point_vjp);
        let mut config = BetaCrownConfig::default();
        crate::preset::apply_preset(&mut config, &armed).unwrap();
        assert!(config.relational_point_vjp);
        crate::preset::apply_preset(
            &mut config,
            &parse("attack:\n  relational_point_vjp: false\n"),
        )
        .unwrap();
        assert!(!config.relational_point_vjp);
        assert!(
            !RelationalPresetKeys::from_preset(&parse("attack:\n  pgd_vjp_batch: true\n"))
                .relational_point_vjp
        );
        for bad in [
            "attack:\n  relational_point_vjp: fast\n",
            "attack:\n  relational_point_vjp: 2\n",
            "attack:\n  relational_point_vpj: true\n",
        ] {
            assert!(serde_yaml::from_str::<crate::preset::PresetConfig>(bad).is_err());
        }
    }

    #[test]
    fn shared_policy_applicator_changes_only_its_owned_field() {
        let preset: crate::preset::PresetConfig =
            serde_yaml::from_str("attack:\n  relational_point_vjp: true\n  pgd_restarts: 7\n")
                .unwrap();
        let mut config = BetaCrownConfig::default();
        let before = serde_json::to_value(&config).unwrap();
        crate::preset::apply_relational_point_vjp_policy(&mut config.relational_point_vjp, &preset);
        assert!(config.relational_point_vjp);
        let mut after = serde_json::to_value(&config).unwrap();
        after["relational_point_vjp"] = serde_json::Value::Bool(false);
        assert_eq!(
            after, before,
            "early projection must not import unrelated solver/attack settings"
        );
    }

    #[test]
    fn typed_policy_is_enabled_only_in_qualified_isomorphic_preset() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
        let mut read = 0;
        let mut configured = Vec::new();
        for year in ["vnncomp24", "vnncomp25", "vnncomp26"] {
            for entry in fs::read_dir(root.join(year)).unwrap() {
                let path = entry.unwrap().path();
                if path.extension().and_then(|extension| extension.to_str()) != Some("yaml") {
                    continue;
                }
                let preset: crate::preset::PresetConfig =
                    serde_yaml::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
                read += 1;
                if preset.attack.relational_point_vjp.is_some() {
                    configured.push((
                        path.strip_prefix(&root).unwrap().to_path_buf(),
                        preset.attack.relational_point_vjp,
                    ));
                    assert_eq!(preset.attack.concurrent, Some(true));
                }
            }
        }
        assert!(
            read > 40,
            "positive control: shipped presets were read, count={read}"
        );
        assert_eq!(
            configured,
            vec![(
                Path::new("vnncomp26/isomorphic_acasxu_2026.yaml").to_path_buf(),
                Some(true)
            )]
        );
    }

    #[test]
    fn shipped_default_discovery_freezes_both_policies_without_importing_numerical_caps() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .unwrap();
        let configs = auto_derive_configs_dir(&[root.join("crates/ny-cli/src/commands")])
            .expect("actual config discovery");
        assert_eq!(configs.canonicalize().unwrap(), root.join("configs"));
        let path = resolve_preset_path(&configs, "isomorphic_acasxu_2026")
            .expect("actual category resolver");
        assert_eq!(
            path.canonicalize().unwrap(),
            root.join("configs/vnncomp26/isomorphic_acasxu_2026.yaml")
        );
        let snapshot = BetaCrownPresetSnapshot::load(&path);
        let loaded = snapshot.loaded().expect("valid shipped snapshot");
        let keys = RelationalPresetKeys::from_preset(loaded);
        assert!(keys.attack_concurrent && keys.relational_point_vjp);
        assert_eq!(keys.edge_wave_cap, None);
        assert_eq!(keys.paired_difference, None);
        assert_eq!(loaded.solver.scored_wall_clamp, Some(true));
        let baseline: crate::preset::PresetConfig =
            serde_yaml::from_str("solver:\n  scored_wall_clamp: true\n").unwrap();
        for initial in [BetaCrownConfig::default(), BetaCrownConfig::acas_xu()] {
            let mut previous = initial.clone();
            crate::preset::apply_preset(&mut previous, &baseline).unwrap();
            let mut adopted = initial;
            crate::preset::apply_preset(&mut adopted, loaded).unwrap();
            let mut expected = serde_json::to_value(previous).unwrap();
            expected["attack_concurrent"] = serde_json::Value::Bool(true);
            expected["relational_point_vjp"] = serde_json::Value::Bool(true);
            assert_eq!(serde_json::to_value(adopted).unwrap(), expected, "shipped policy must change only the two measured flags beside the historical clamp");
        }
    }

    #[test]
    fn planted_violation_is_found_through_prepared_point_execution() {
        let f = scalar(0.0, 0.0, 0.0, 0.0);
        let g = scalar(0.0, 1.0, 1.0, 0.0);
        let bounds = [Bound::new(-1.0, 1.0)];
        let advice = DenseTwinAdvice::build(&f, &g, &bounds, deadline()).unwrap();
        let diff = build_difference_network(&f, &g).unwrap();
        let candidate =
            search_isomorphic_deviation_with_advice(&diff, &bounds, 0.5, deadline(), Some(&advice))
                .expect("planted advice candidate");
        assert!(
            advice.forward_pairs.load(Ordering::Relaxed) > 0,
            "actual prepared execution must arrive"
        );
        let f = forward_point_vec(&f, &candidate).unwrap();
        let g = forward_point_vec(&g, &candidate).unwrap();
        assert!(
            (f[0] - g[0]).abs() > 0.5,
            "independent original graph replay"
        );
    }

    #[test]
    fn pre_cancelled_preparation_never_enters_shape_probe() {
        let (f, g) = pair();
        let token = ny_propagate::PgdCancelToken::new();
        token.cancel();
        POINT_PLAN_PROBES.with(|count| count.set(0));
        assert!(DenseTwinAdvice::build_cancellable(
            &f,
            &g,
            &[Bound::new(-2.0, 2.0); 2],
            deadline(),
            Some(&token)
        )
        .is_none());
        assert_eq!(POINT_PLAN_PROBES.with(std::cell::Cell::get), 0);
    }

    #[test]
    fn non_cpu_fragment_declines_before_any_shape_probe() {
        use ny_propagate::layers::Conv2dLayer;
        let mut convolution = GraphNetwork::new();
        let kernel = ArrayD::from_elem(IxDyn(&[1, 1, 1, 1]), 1.0);
        let layer = Conv2dLayer::with_input_shape(kernel, None, (1, 1), (0, 0), 1, 1).unwrap();
        convolution.add_node(GraphNode::from_input("conv", Layer::Conv2d(layer)));
        convolution.set_output("conv");
        let dense = scalar(1.0, 0.0, 1.0, 0.0);
        for (f, g) in [(&convolution, &dense), (&dense, &convolution)] {
            POINT_PLAN_PROBES.with(|count| count.set(0));
            assert!(DenseTwinAdvice::build(f, g, &[Bound::new(-1.0, 1.0)], deadline()).is_none());
            assert_eq!(POINT_PLAN_PROBES.with(std::cell::Cell::get), 0);
        }
    }

    #[test]
    fn cancelled_composed_search_does_not_execute_prepared_forwards_or_vjps() {
        let (f, g) = pair();
        let bounds = [Bound::new(-2.0, 2.0); 2];
        let token = ny_propagate::PgdCancelToken::new();
        POINT_PLAN_PROBES.with(|count| count.set(0));
        let advice =
            DenseTwinAdvice::build_cancellable(&f, &g, &bounds, deadline(), Some(&token)).unwrap();
        assert_eq!(POINT_PLAN_PROBES.with(std::cell::Cell::get), 2);
        token.cancel();
        let diff = build_difference_network(&f, &g).unwrap();
        assert!(search_isomorphic_deviation_with_advice_and_cancel(
            &diff,
            &bounds,
            0.05,
            deadline(),
            Some(&advice),
            Some(&token)
        )
        .is_none());
        assert_eq!(advice.forward_pairs.load(Ordering::Relaxed), 0);
        assert_eq!(advice.analytic_vjps.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn both_policies_reach_serial_and_scoped_routes_from_one_frozen_preset() {
        for concurrent in [false, true] {
            for prepared in [false, true] {
                let yaml = format!(
                    "attack:\n  concurrent: {concurrent}\n  relational_point_vjp: {prepared}\n"
                );
                let loaded: crate::preset::PresetConfig = serde_yaml::from_str(&yaml).unwrap();
                let keys = RelationalPresetKeys::from_preset(&loaded);
                let mut config = BetaCrownConfig::default();
                crate::preset::apply_preset(&mut config, &loaded).unwrap();
                assert_eq!(keys.attack_concurrent, concurrent);
                assert_eq!(keys.relational_point_vjp, prepared);
                assert_eq!(config.attack_concurrent, concurrent);
                assert_eq!(config.relational_point_vjp, prepared);
            }
        }
    }
}
