// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::{warm_start_alpha_config, BatchedScalarBounds};

#[test]
fn test_batched_scalar_bounds_field_roundtrip_4404() {
    let bounds = BatchedScalarBounds {
        lower_bounds: vec![-1.0_f32, 0.25],
        upper_bounds: vec![1.5_f32, 2.0],
        linear_bounds: vec![None, None],
    };

    assert_eq!(bounds.lower_bounds, vec![-1.0_f32, 0.25]);
    assert_eq!(bounds.upper_bounds, vec![1.5_f32, 2.0]);
    assert!(
        bounds.linear_bounds.iter().all(Option::is_none),
        "test carrier should preserve per-domain optional linear bounds"
    );
}

#[test]
fn child_warm_start_clears_root_only_cgan_collection_policy() {
    let mut config = crate::beta_crown::config::BetaCrownConfig::default();
    config.alpha_config.fix_interm_bounds = false;
    config.alpha_config.cgan_sparse_target_complete_root = true;
    config.alpha_config.cgan_complete_crown_ibp_root = true;
    config.input_split_alpha_iteration = 3;
    config.input_split_lr_alpha = 0.025;

    let child = warm_start_alpha_config(&config, None);
    assert!(
        child.fix_interm_bounds,
        "children must keep cheap intermediates"
    );
    assert!(
        !child.cgan_sparse_target_complete_root,
        "the root-only collector must not escape into BaB children"
    );
    assert!(
        !child.cgan_complete_crown_ibp_root,
        "the complete root-only collector must not escape into BaB children"
    );
    assert_eq!(child.iterations, 3);
    assert_eq!(child.learning_rate, 0.025);
}

/// `#cgan-child-interm-refine` regression suite.
///
/// The lever gives each BaB child its OWN intermediate-bound map — the complete
/// cGAN CROWN-IBP cascade run on the child's own box under a per-child wall
/// budget — instead of leaving it on the frozen root map. Four things have to
/// be true and each has its own test:
///
///   1. dark by default, and inert unless the ROOT armed the cascade (otherwise
///      a shipped preset changes behaviour by accident);
///   2. armed, the config really carries the cascade (otherwise dead
///      capability);
///   3. armed, a child really ENTERS the cascade — asserted on the collector's
///      OWN returned source, never inferred from the config it was handed
///      (otherwise every measurement of the lever is vacuous);
///   4. every published child bound still CONTAINS an independently computed
///      reachable envelope of the child box, on BOTH sides of the switch
///      (otherwise the lever is a false proof).
mod cgan_child_interm_refine_tests {
    use super::super::{
        child_interm_refine_alpha_config, install_child_interm_refine_overrides,
        ChildIntermRefineDomain,
    };
    use crate::layers::{Conv2dLayer, ConvTranspose2dLayer, ReLULayer};
    use crate::network::GraphNode;
    use crate::{GraphNetwork, Layer};
    use ndarray::{Array1, ArrayD, IxDyn};
    use ny_tensor::BoundedTensor;
    use std::collections::HashMap;
    use std::sync::Arc;

    fn armed_config(max_secs: f64) -> crate::beta_crown::config::BetaCrownConfig {
        let mut config = crate::beta_crown::config::BetaCrownConfig::default();
        config.alpha_config.cgan_complete_crown_ibp_root = true;
        config.alpha_config.cgan_child_interm_refine_max_secs = Some(max_secs);
        config
    }

    #[test]
    fn dark_by_default_and_inert_without_the_root_arm() {
        let default = crate::beta_crown::config::BetaCrownConfig::default();
        assert!(
            child_interm_refine_alpha_config(&default, None).is_none(),
            "with no child budget the prepass must not exist at all — the dark path \
             must be byte-identical to today"
        );

        let mut root_armed_only = crate::beta_crown::config::BetaCrownConfig::default();
        root_armed_only.alpha_config.cgan_complete_crown_ibp_root = true;
        assert!(
            child_interm_refine_alpha_config(&root_armed_only, None).is_none(),
            "arming the ROOT cascade alone must not start refining children"
        );

        let mut budget_only = crate::beta_crown::config::BetaCrownConfig::default();
        budget_only.alpha_config.cgan_child_interm_refine_max_secs = Some(120.0);
        assert!(
            child_interm_refine_alpha_config(&budget_only, None).is_none(),
            "a child must never enter a lane its root did not arm"
        );

        let mut nonpositive = crate::beta_crown::config::BetaCrownConfig::default();
        nonpositive.alpha_config.cgan_complete_crown_ibp_root = true;
        nonpositive.alpha_config.cgan_child_interm_refine_max_secs = Some(0.0);
        assert!(
            child_interm_refine_alpha_config(&nonpositive, None).is_none(),
            "a non-positive budget must leave the lane dark rather than arm it unbounded"
        );
    }

    #[test]
    fn armed_config_carries_the_cascade() {
        let refine = child_interm_refine_alpha_config(&armed_config(120.0), None)
            .expect("armed child refinement config");
        assert!(
            refine.cgan_complete_crown_ibp_root,
            "the prepass must request the complete cascade — this is the entire delta"
        );
        assert!(
            refine.fix_interm_bounds,
            "the cascade's own eligibility predicate requires the fixed-intermediate contract"
        );
        assert!(
            !refine.cgan_sparse_target_complete_root,
            "the OTHER typed cGAN collector must stay off"
        );
    }

    #[test]
    fn per_child_budget_only_ever_brings_the_deadline_nearer() {
        let mut config = armed_config(0.5);
        let ceiling = std::time::Instant::now() + std::time::Duration::from_secs(600);
        let refine = child_interm_refine_alpha_config(&config, Some(ceiling))
            .expect("armed child refinement config");
        let deadline = refine.deadline.expect("composed deadline");
        assert!(
            deadline < ceiling,
            "a 0.5 s per-child budget under a 600 s ceiling must compose a NEARER deadline"
        );

        // And a budget far beyond the ceiling must never extend it.
        config.alpha_config.cgan_child_interm_refine_max_secs = Some(10_000.0);
        let refine = child_interm_refine_alpha_config(&config, Some(ceiling))
            .expect("armed child refinement config");
        assert!(
            refine.deadline.expect("composed deadline") <= ceiling,
            "the per-child budget must never EXTEND the caller's ceiling"
        );
    }

    /// Sequential ConvTranspose2d -> ReLU -> Conv2d -> ReLU, the minimal shape
    /// the typed cGAN cascade admits: its eligibility predicate demands a
    /// sequential graph with both Conv2d and ConvTranspose2d layers.
    fn cgan_shaped_graph() -> GraphNetwork {
        let convt = ConvTranspose2dLayer::with_input_shape(
            ArrayD::from_shape_vec(IxDyn(&[1, 2, 1, 1]), vec![0.85_f32, -0.62])
                .expect("ConvTranspose kernel"),
            Some(Array1::from_vec(vec![0.07_f32, -0.11])),
            (1, 1),
            (0, 0),
            2,
            2,
        )
        .expect("ConvTranspose layer");
        let conv = Conv2dLayer::with_input_shape(
            ArrayD::from_shape_vec(IxDyn(&[2, 2, 1, 1]), vec![0.9_f32, -0.45, 0.33, 0.71])
                .expect("Conv kernel"),
            Some(Array1::from_vec(vec![-0.05_f32, 0.12])),
            (1, 1),
            (0, 0),
            2,
            2,
        )
        .expect("Conv layer");

        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("gen", Layer::ConvTranspose2d(convt)));
        graph.add_node(GraphNode::new(
            "gen_relu",
            Layer::ReLU(ReLULayer),
            vec!["gen".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "disc",
            Layer::Conv2d(conv),
            vec!["gen_relu".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "disc_relu",
            Layer::ReLU(ReLULayer),
            vec!["disc".to_string()],
        ));
        graph.set_output("disc_relu");
        graph
    }

    /// A BaB CHILD box: a strict sub-box of the root domain, which is the only
    /// place this lever ever runs.
    fn child_box() -> BoundedTensor {
        BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![-0.4_f32, -0.15, 0.05, -0.3])
                .expect("child lower"),
            ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![0.25_f32, 0.4, 0.5, 0.2])
                .expect("child upper"),
        )
        .expect("child box")
    }

    /// An independently computed INNER approximation of the child box's
    /// reachable set: the exact activation at each of a grid of concrete
    /// points, taken through the pointwise collector. A published bound that
    /// fails to contain this is unsound beyond any doubt.
    fn reachable_envelope(graph: &GraphNetwork) -> HashMap<String, (Vec<f32>, Vec<f32>)> {
        let domain = child_box();
        let lo: Vec<f32> = domain.lower().iter().copied().collect();
        let hi: Vec<f32> = domain.upper().iter().copied().collect();
        let steps = 6usize;
        let mut env: HashMap<String, (Vec<f32>, Vec<f32>)> = HashMap::new();
        let total = steps.pow(lo.len() as u32);
        for index in 0..total {
            let mut rest = index;
            let point: Vec<f32> = (0..lo.len())
                .map(|d| {
                    let step = rest % steps;
                    rest /= steps;
                    lo[d] + (step as f32 / (steps - 1) as f32) * (hi[d] - lo[d])
                })
                .collect();
            let pt = ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), point).expect("sample point");
            let degenerate = BoundedTensor::new(pt.clone(), pt).expect("degenerate box");
            let acts = graph
                .collect_node_activations_pointwise(&degenerate, None)
                .expect("pointwise activations");
            for (name, value) in acts {
                let v: Vec<f32> = value.lower().iter().copied().collect();
                match env.get_mut(&name) {
                    Some((mn, mx)) => {
                        for (k, x) in v.iter().enumerate() {
                            if *x < mn[k] {
                                mn[k] = *x;
                            }
                            if *x > mx[k] {
                                mx[k] = *x;
                            }
                        }
                    }
                    None => {
                        env.insert(name, (v.clone(), v));
                    }
                }
            }
        }
        env
    }

    fn assert_contains(
        bounds: &HashMap<String, BoundedTensor>,
        env: &HashMap<String, (Vec<f32>, Vec<f32>)>,
        label: &str,
    ) {
        let tol = 1e-4_f32;
        let mut checked = 0usize;
        for (name, bound) in bounds {
            let Some((mn, mx)) = env.get(name) else {
                continue;
            };
            let lower: Vec<f32> = bound.lower().iter().copied().collect();
            let upper: Vec<f32> = bound.upper().iter().copied().collect();
            if lower.len() != mn.len() {
                continue;
            }
            checked += 1;
            for k in 0..lower.len() {
                assert!(
                    lower[k] <= mn[k] + tol,
                    "{label}: node '{name}' row {k} lower bound {} EXCLUDES reachable min {} \
                     — a tighter bound that excludes a reachable point is a FALSE PROOF, \
                     not a tighter bound",
                    lower[k],
                    mn[k],
                );
                assert!(
                    upper[k] >= mx[k] - tol,
                    "{label}: node '{name}' row {k} upper bound {} EXCLUDES reachable max {}",
                    upper[k],
                    mx[k],
                );
            }
        }
        assert!(
            checked >= 3,
            "{label}: containment covered only {checked} nodes — the envelope join matched \
             nothing, so this assertion is a NO-OP rather than a pass"
        );
    }

    /// A domain carrier shaped like the two real ones: the installer is generic
    /// over exactly these two fields.
    struct FakeDomain {
        input_bounds: Arc<BoundedTensor>,
        node_bounds_override: Option<Arc<HashMap<String, BoundedTensor>>>,
    }

    impl ChildIntermRefineDomain for FakeDomain {
        fn refine_input_bounds(&self) -> &Arc<BoundedTensor> {
            &self.input_bounds
        }
        fn refine_override_slot(&mut self) -> &mut Option<Arc<HashMap<String, BoundedTensor>>> {
            &mut self.node_bounds_override
        }
    }

    /// (3) lane entry, asserted on the collector's own source via the
    /// installer's success tally, and (4) containment on both sides.
    #[test]
    fn armed_prepass_enters_the_cascade_and_contains_the_reachable_set() {
        let graph = cgan_shaped_graph();
        let env = reachable_envelope(&graph);
        let exec_order = graph.exec_order().expect("exec order");

        // The DARK side: what a child is bounded against today.
        let dark_alpha = {
            let mut cfg = crate::beta_crown::config::BetaCrownConfig::default();
            cfg.alpha_config.fix_interm_bounds = true;
            cfg.alpha_config
        };
        let (dark_bounds, dark_source) = graph
            .collect_alpha_reference_bounds_with_engine_and_source(
                &child_box(),
                &dark_alpha,
                None,
                exec_order,
            )
            .expect("dark child reference bounds");
        assert!(
            !dark_source.is_cgan_complete_crown_ibp(),
            "with the lever dark a child must NOT already be on the complete cascade; \
             if it were, this lever is a no-op and every measurement of it is vacuous"
        );

        // The ARMED side, through the production entry point.
        let refine = child_interm_refine_alpha_config(&armed_config(120.0), None)
            .expect("armed child refinement config");
        let mut domains = vec![FakeDomain {
            input_bounds: Arc::new(child_box()),
            node_bounds_override: None,
        }];
        let tally =
            install_child_interm_refine_overrides(&refine, &graph, None, &[0], &mut domains);
        assert_eq!(
            tally.installed, 1,
            "the armed prepass must ENTER the complete cGAN cascade and install its map; \
             a declined target here ({tally:?}) is a VOID measurement, not a negative one"
        );

        let armed_bounds = domains[0]
            .node_bounds_override
            .as_ref()
            .expect("installed refined map");
        assert_contains(&dark_bounds, &env, "dark child");
        assert_contains(armed_bounds, &env, "armed child");
    }

    /// A domain that already carries an override (complete clipping / fresh
    /// domain clip) must keep every entry of it, and the merge must be
    /// SHRINK-ONLY: the prepass may tighten a clip bound, never widen one.
    ///
    /// This is the common case on cgan2026, which arms both `clip.relaxed` and
    /// `input_split_fresh_domain_clip`; getting it wrong in the skip direction
    /// makes the lever a no-op, and getting it wrong in the overwrite direction
    /// discards a certified clip bound.
    #[test]
    fn prepass_merge_is_shrink_only_against_an_existing_override() {
        let graph = cgan_shaped_graph();
        let refine = child_interm_refine_alpha_config(&armed_config(120.0), None)
            .expect("armed child refinement config");

        // A deliberately NARROW stand-in clip bound on one node, plus a node
        // the cascade will not produce at all, so both merge directions are
        // exercised.
        let ibp = graph
            .collect_node_bounds(&child_box())
            .expect("IBP map for the child box");
        let mut clip: HashMap<String, BoundedTensor> = HashMap::new();
        let disc = ibp.get("disc").expect("disc node in the IBP map").clone();
        clip.insert("disc".to_string(), disc.clone());
        clip.insert("clip_only_sentinel".to_string(), disc);
        let existing = Arc::new(clip);

        let mut domains = vec![FakeDomain {
            input_bounds: Arc::new(child_box()),
            node_bounds_override: Some(Arc::clone(&existing)),
        }];
        let tally =
            install_child_interm_refine_overrides(&refine, &graph, None, &[0], &mut domains);
        assert_eq!(
            tally.merged_into_existing_override, 1,
            "the prepass must MERGE into an existing override, not skip the domain —              cgan arms the clip on most children, so skipping would make this a no-op"
        );
        assert_eq!(tally.installed, 1, "the merged map must be installed");

        let merged = domains[0]
            .node_bounds_override
            .as_ref()
            .expect("merged override");
        assert!(
            merged.contains_key("clip_only_sentinel"),
            "an entry only the clip had must SURVIVE the merge; dropping it would              discard a certified bound the domain was relying on"
        );
        for (name, before) in existing.iter() {
            let after = merged.get(name).expect("every clip entry retained");
            for (k, (b, a)) in before.lower().iter().zip(after.lower().iter()).enumerate() {
                assert!(
                    *a >= *b - 1e-5,
                    "node '{name}' row {k} lower {a} is BELOW the clip's {b}; the merge                      must be shrink-only"
                );
            }
            for (k, (b, a)) in before.upper().iter().zip(after.upper().iter()).enumerate() {
                assert!(
                    *a <= *b + 1e-5,
                    "node '{name}' row {k} upper {a} is ABOVE the clip's {b}; the merge                      must be shrink-only"
                );
            }
        }
    }
    /// The ROOT box the child splits out of: same shape, strictly wider on
    /// every coordinate.
    fn wider_root_box() -> BoundedTensor {
        BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![-1.0_f32, -1.0, -1.0, -1.0])
                .expect("root lower"),
            ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![1.0_f32, 1.0, 1.0, 1.0])
                .expect("root upper"),
        )
        .expect("root box")
    }

    fn interval_width_sum(bound: &BoundedTensor) -> f32 {
        bound
            .lower()
            .iter()
            .zip(bound.upper().iter())
            .map(|(l, u)| u - l)
            .sum()
    }

    /// Run the armed prepass on the child box with a given
    /// (REUSED, not reinvented) `cgan_narrow_target_max_rows` ceiling, with
    /// the frozen ROOT map preinstalled as the child's override — which is
    /// exactly the state a deferred cGAN child is in today.
    fn refine_child_with_ceiling(
        graph: &GraphNetwork,
        root_map: &Arc<HashMap<String, BoundedTensor>>,
        max_rows: usize,
    ) -> HashMap<String, BoundedTensor> {
        let mut config = armed_config(120.0);
        config.alpha_config.cgan_narrow_target_max_rows = Some(max_rows);
        let refine =
            child_interm_refine_alpha_config(&config, None).expect("armed child refinement config");
        let mut domains = vec![FakeDomain {
            input_bounds: Arc::new(child_box()),
            node_bounds_override: Some(Arc::clone(root_map)),
        }];
        let tally = install_child_interm_refine_overrides(&refine, graph, None, &[0], &mut domains);
        assert_eq!(
            tally.installed, 1,
            "the armed prepass must install on this fixture (ceiling {max_rows}); \
             a declined cascade here ({tally:?}) makes the tightening assertions VACUOUS"
        );
        domains[0]
            .node_bounds_override
            .as_ref()
            .expect("installed refined map")
            .as_ref()
            .clone()
    }

    /// (2) in the suite banner, plus the SCHEDULING direction, measured at the
    /// production seam in three separations, each of which fails on its own:
    ///
    ///   * refined-vs-frozen: the ceiling-4 child map is STRICTLY tighter than
    ///     the frozen root map on the narrow target downstream of the unstable
    ///     generator ReLU (the whole point of the lever);
    ///   * retained-vs-declined: a ceiling-0 arm — every target declined, the
    ///     cascade can only publish its certified seed — is strictly WORSE on
    ///     the narrow target, so the tightening is bought by the RETAINED
    ///     target, not by the seed re-anchoring alone;
    ///   * decline really declines: the 6-row wide pre-layer is byte-equal
    ///     between ceiling 4 and ceiling 0 (both decline it), and an
    ///     everything-retained arm tightens it — so the ceiling is what
    ///     withheld that work, in the direction the value law demands.
    ///
    /// The ceiling is `cgan_narrow_target_max_rows`, REUSED from
    /// #cgan-narrow-target-budget; this lane deliberately has no second row
    /// ceiling of its own.
    #[test]
    fn refined_child_is_strictly_tighter_on_the_narrow_target_and_the_ceiling_schedules_it() {
        let graph = mixed_wide_then_narrow_graph();

        // The generator pre-activation must actually be UNSTABLE on the child
        // box, or this fixture proves nothing about relaxation error.
        let child_ibp = graph
            .collect_node_bounds(&child_box())
            .expect("child IBP map");
        let gen_pre = child_ibp.get("gen").expect("gen node");
        assert!(
            gen_pre.lower().iter().any(|l| *l < 0.0) && gen_pre.upper().iter().any(|u| *u > 0.0),
            "fixture defect: the generator ReLU is STABLE on the child box, so there is \
             no intermediate relaxation error for the refinement to remove"
        );

        // The frozen ROOT map: the complete cascade run on the ROOT box, which
        // is what every deferred child is bounded against today.
        let mut root_config = armed_config(120.0);
        root_config.alpha_config.cgan_narrow_target_max_rows = Some(4);
        let root_refine = child_interm_refine_alpha_config(&root_config, None)
            .expect("armed config for the root collection");
        let exec_order = graph.exec_order().expect("exec order");
        let (root_map, root_source) = graph
            .collect_alpha_reference_bounds_with_engine_and_source(
                &wider_root_box(),
                &root_refine,
                None,
                exec_order,
            )
            .expect("root reference map");
        assert!(
            root_source.is_cgan_complete_crown_ibp(),
            "the root arm must come from the same cascade, or the comparison is not \
             refinement-vs-frozen but collector-vs-collector"
        );
        let root_map = Arc::new(root_map);

        // Arm A: ceiling 4 — the 3-row narrow target is RETAINED, the 6-row
        // wide target and the 8-row conv targets are DECLINED. This is the
        // shipped shape of the lever.
        let refined = refine_child_with_ceiling(&graph, &root_map, 4);
        // Arm B: ceiling 0 — every target declined.
        let declined = refine_child_with_ceiling(&graph, &root_map, 0);
        // Arm C: ceiling far above every row count — everything retained.
        let all_retained = refine_child_with_ceiling(&graph, &root_map, 1_000);

        // Tighter is only meaningful if still SOUND: the refined map must
        // contain an independently sampled reachable envelope of the child box.
        let env = reachable_envelope(&graph);
        assert_contains(&refined, &env, "refined child (ceiling 4)");

        let width = |m: &HashMap<String, BoundedTensor>, node: &str| {
            interval_width_sum(m.get(node).unwrap_or_else(|| panic!("{node} entry")))
        };

        let root_narrow = width(&root_map, "narrow");
        let refined_narrow = width(&refined, "narrow");
        let declined_narrow = width(&declined, "narrow");
        assert!(
            refined_narrow < root_narrow - 1e-4,
            "a refined child must be STRICTLY tighter than the frozen root map on the \
             narrow target downstream of the unstable ReLU (refined {refined_narrow} vs \
             root {root_narrow}); equality means the prepass changed nothing and the \
             lever is a costed no-op"
        );
        assert!(
            refined_narrow < declined_narrow - 1e-4,
            "the tightening must be bought by the RETAINED narrow target: with every \
             target declined the child keeps its seed ({declined_narrow}), and the \
             retained arm ({refined_narrow}) must beat it — otherwise the ceiling is \
             not scheduling the work this lane exists to schedule"
        );

        let refined_wide = width(&refined, "wide");
        let declined_wide = width(&declined, "wide");
        let all_retained_wide = width(&all_retained, "wide");
        assert!(
            (refined_wide - declined_wide).abs() <= 1e-6,
            "ceiling 4 must DECLINE the 6-row wide target exactly as ceiling 0 does \
             (ceil4 {refined_wide} vs ceil0 {declined_wide}); a difference means the \
             ceiling admitted work the value law says to withhold"
        );
        assert!(
            all_retained_wide < refined_wide - 1e-4,
            "the wide target must be TIGHTENABLE when retained ({all_retained_wide} vs \
             declined-at-ceiling-4 {refined_wide}) — otherwise the decline assertion \
             above is vacuously green on an inert fixture"
        );
    }

    /// Fail-closed on provenance: on a graph the cGAN cascade REFUSES (here a
    /// Linear-only chain, which fails the cascade's own eligibility predicate),
    /// the collector still returns an ordinary map — and the prepass must NOT
    /// install it. Installing an ordinary map would read as "the lever ran"
    /// while the cascade never did, which is the vacuous-null shape the
    /// provenance gate exists to refuse; the domain must keep its historical
    /// frozen path untouched instead.
    #[test]
    fn prepass_declines_when_the_cascade_refuses_and_leaves_the_domain_untouched() {
        use crate::layers::LinearLayer;
        use ndarray::Array2;
        let mut graph = GraphNetwork::new();
        let l0 = LinearLayer::new(
            Array2::from_shape_vec((2, 4), vec![0.9_f32, -0.4, 0.3, 0.7, -0.6, 0.5, 0.2, -0.8])
                .expect("weights"),
            Some(Array1::from_vec(vec![0.1_f32, -0.05])),
        )
        .expect("linear layer");
        graph.add_node(GraphNode::from_input("fc", Layer::Linear(l0)));
        graph.add_node(GraphNode::new(
            "fc_relu",
            Layer::ReLU(ReLULayer),
            vec!["fc".to_string()],
        ));
        graph.set_output("fc_relu");

        let refine = child_interm_refine_alpha_config(&armed_config(120.0), None)
            .expect("armed child refinement config");
        let flat_child = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[4]), vec![-0.4_f32, -0.15, 0.05, -0.3])
                .expect("child lower"),
            ArrayD::from_shape_vec(IxDyn(&[4]), vec![0.25_f32, 0.4, 0.5, 0.2])
                .expect("child upper"),
        )
        .expect("child box");
        let sentinel: Arc<HashMap<String, BoundedTensor>> = Arc::new(HashMap::new());
        let mut domains = vec![FakeDomain {
            input_bounds: Arc::new(flat_child),
            node_bounds_override: Some(Arc::clone(&sentinel)),
        }];
        let tally =
            install_child_interm_refine_overrides(&refine, &graph, None, &[0], &mut domains);
        assert_eq!(
            tally.installed, 0,
            "a refused cascade must never install a map ({tally:?}); an ordinary map \
             installed here would read as the lever running when it did not"
        );
        assert_eq!(
            tally.declined, 1,
            "the refusal must be tallied as a decline"
        );
        let after = domains[0]
            .node_bounds_override
            .as_ref()
            .expect("existing override retained");
        assert!(
            Arc::ptr_eq(after, &sentinel),
            "the domain must keep its EXACT historical override object — not a merged \
             or substituted one — when the cascade refuses"
        );
    }

    /// The containment checker itself must be shown ABLE TO FAIL: a
    /// deliberately-tightened-WRONG bound — one that excludes points the
    /// network provably reaches — must be rejected, not admired for its
    /// tightness. (The offline harness demonstrated this hazard concretely:
    /// halving a ReLU chord intercept produced a "tighter" root that excluded
    /// 534 reachable points.)
    #[test]
    #[should_panic(expected = "EXCLUDES reachable")]
    fn containment_check_rejects_a_deliberately_tightened_wrong_bound() {
        let graph = cgan_shaped_graph();
        let env = reachable_envelope(&graph);
        let (mn, mx) = env.get("disc").expect("disc envelope").clone();
        // A sliver hugging the reachable MAX: strictly inside the reachable
        // range, so its lower bound excludes the reachable minimum.
        let poisoned_lower: Vec<f32> = mn
            .iter()
            .zip(mx.iter())
            .map(|(a, b)| a + 0.75 * (b - a))
            .collect();
        let n = poisoned_lower.len();
        let mut poisoned = HashMap::new();
        poisoned.insert(
            "disc".to_string(),
            BoundedTensor::new(
                ArrayD::from_shape_vec(IxDyn(&[n]), poisoned_lower).expect("poisoned lower"),
                ArrayD::from_shape_vec(IxDyn(&[n]), mx).expect("poisoned upper"),
            )
            .expect("poisoned bound"),
        );
        assert_contains(&poisoned, &env, "poisoned");
    }
    /// The strict-tightening fixture: ConvTranspose -> ReLU -> Conv -> ReLU ->
    /// Flatten -> Linear(6, WIDE) -> ReLU -> Linear(3, NARROW) -> ReLU.
    ///
    /// Shaped after the real cGAN surface (ConvTranspose+Conv for cascade
    /// eligibility, Gemm targets at the end like the discriminator head), with
    /// the layer the refinement must tighten placed the way the offline
    /// measurement found value: the NARROW 3-row target sits DOWNSTREAM of the
    /// unstable generator ReLU, and the WIDE 6-row pre-layer sits ABOVE the
    /// ceiling=4 the test arms, so the reused `cgan_narrow_target_max_rows`
    /// ceiling separates them exactly as it does on the official rows.
    /// Mixed-sign weights keep every ReLU unstable on the child box.
    fn mixed_wide_then_narrow_graph() -> GraphNetwork {
        use crate::layers::{FlattenLayer, LinearLayer};
        use ndarray::Array2;
        let convt = ConvTranspose2dLayer::with_input_shape(
            ArrayD::from_shape_vec(IxDyn(&[1, 2, 1, 1]), vec![1.7_f32, -1.3])
                .expect("ConvTranspose kernel"),
            Some(Array1::from_vec(vec![0.15_f32, -0.2])),
            (1, 1),
            (0, 0),
            2,
            2,
        )
        .expect("ConvTranspose layer");
        let conv = Conv2dLayer::with_input_shape(
            ArrayD::from_shape_vec(IxDyn(&[2, 2, 1, 1]), vec![1.2_f32, -1.0, -0.8, 1.4])
                .expect("mid kernel"),
            Some(Array1::from_vec(vec![0.1_f32, -0.05])),
            (1, 1),
            (0, 0),
            2,
            2,
        )
        .expect("mid layer");
        let w_wide = Array2::from_shape_fn((6, 8), |(r, c)| {
            0.6 - 0.17 * (r as f32) + 0.43 * (c as f32) - 0.09 * ((r * c) as f32)
        });
        let wide = LinearLayer::new(
            w_wide,
            Some(Array1::from_shape_fn(6, |r| 0.05 * r as f32 - 0.2)),
        )
        .expect("wide layer");
        let w_narrow = Array2::from_shape_fn((3, 6), |(r, c)| {
            0.31 - 0.11 * (r as f32) + 0.07 * (c as f32) - 0.05 * ((r + c) as f32)
        });
        let narrow = LinearLayer::new(
            w_narrow,
            Some(Array1::from_shape_fn(3, |r| 0.1 - 0.04 * r as f32)),
        )
        .expect("narrow layer");
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("gen", Layer::ConvTranspose2d(convt)));
        graph.add_node(GraphNode::new(
            "gen_relu",
            Layer::ReLU(ReLULayer),
            vec!["gen".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "mid",
            Layer::Conv2d(conv),
            vec!["gen_relu".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "mid_relu",
            Layer::ReLU(ReLULayer),
            vec!["mid".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "flat",
            Layer::Flatten(FlattenLayer::new(0)),
            vec!["mid_relu".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "wide",
            Layer::Linear(wide),
            vec!["flat".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "wide_relu",
            Layer::ReLU(ReLULayer),
            vec!["wide".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "narrow",
            Layer::Linear(narrow),
            vec!["wide_relu".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "narrow_relu",
            Layer::ReLU(ReLULayer),
            vec!["narrow".to_string()],
        ));
        graph.set_output("narrow_relu");
        graph
    }

    /// `#cgan-child-seed` regression suite: the seeded, optionally parallel
    /// prepass and the fresh-clip map handoff.
    ///
    /// What each test pins, and why it would fail if the lever were wrong:
    ///   * seeded child maps stay SHRINK-ONLY against the inherited root map
    ///     and still CONTAIN a 1e4-point sampled reachable envelope of random
    ///     child boxes (a seed that excluded a reachable point is a false proof);
    ///   * a refused cascade leaves the domain's exact override object alone;
    ///   * no inherited map at all falls back to the unseeded cascade, with a
    ///     bit-identical map;
    ///   * parallel tranches publish maps BIT-IDENTICAL to the serial reference
    ///     (and the parallel arm demonstrably ran wider than 1);
    ///   * the planes the fresh clip builds with the domain's map contain the
    ///     sampled true spec outputs, as do the root-only (dark) planes;
    ///   * every key is dark by default and the dark plan is the historical one.
    mod child_seed_tests {
        use super::super::super::{
            compute_crown_or_ibp_bounds_with_node_bounds,
            install_child_interm_refine_overrides_scheduled, ChildRefineSchedule,
        };
        use super::*;
        use ndarray::Array2;

        const SAMPLES: usize = 10_000;

        fn next_unit(state: &mut u64) -> f32 {
            *state ^= *state << 13;
            *state ^= *state >> 7;
            *state ^= *state << 17;
            ((*state >> 11) as f64 / (1u64 << 53) as f64) as f32
        }

        /// Random sub-boxes of `wider_root_box()` ([-1, 1]^4), each side at
        /// least 0.1 wide: legitimate BaB children of that root.
        fn random_child_boxes(count: usize, seed: u64) -> Vec<BoundedTensor> {
            let mut state = seed | 1;
            (0..count)
                .map(|_| {
                    let mut lo = Vec::with_capacity(4);
                    let mut hi = Vec::with_capacity(4);
                    for _ in 0..4 {
                        let width = 0.1 + 0.9 * next_unit(&mut state);
                        let start = -1.0 + (2.0 - width) * next_unit(&mut state);
                        lo.push(start);
                        hi.push((start + width).min(1.0));
                    }
                    BoundedTensor::new(
                        ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), lo).expect("child lower"),
                        ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), hi).expect("child upper"),
                    )
                    .expect("child box")
                })
                .collect()
        }

        /// `samples` uniform points of `domain` plus every corner, flat.
        fn sample_points(domain: &BoundedTensor, samples: usize, seed: u64) -> Vec<Vec<f32>> {
            let lo: Vec<f32> = domain.lower().iter().copied().collect();
            let hi: Vec<f32> = domain.upper().iter().copied().collect();
            let mut points = Vec::with_capacity(samples + (1 << lo.len()));
            for corner in 0..(1usize << lo.len()) {
                points.push(
                    (0..lo.len())
                        .map(|d| if corner >> d & 1 == 1 { hi[d] } else { lo[d] })
                        .collect(),
                );
            }
            let mut state = seed | 1;
            for _ in 0..samples {
                points.push(
                    (0..lo.len())
                        .map(|d| {
                            (lo[d] + next_unit(&mut state) * (hi[d] - lo[d])).clamp(lo[d], hi[d])
                        })
                        .collect(),
                );
            }
            points
        }

        fn activations_at(graph: &GraphNetwork, point: &[f32]) -> HashMap<String, Vec<f32>> {
            let pt = ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), point.to_vec()).expect("point");
            let degenerate = BoundedTensor::new(pt.clone(), pt).expect("degenerate box");
            graph
                .collect_node_activations_pointwise(&degenerate, None)
                .expect("pointwise activations")
                .into_iter()
                .map(|(name, value)| (name, value.lower().iter().copied().collect()))
                .collect()
        }

        /// Min/max of every node's activation over the sampled points: an
        /// INNER approximation of the reachable set, in `assert_contains` form.
        fn sampled_envelope(
            graph: &GraphNetwork,
            domain: &BoundedTensor,
            seed: u64,
        ) -> HashMap<String, (Vec<f32>, Vec<f32>)> {
            let mut env: HashMap<String, (Vec<f32>, Vec<f32>)> = HashMap::new();
            for point in sample_points(domain, SAMPLES, seed) {
                for (name, values) in activations_at(graph, &point) {
                    match env.get_mut(&name) {
                        Some((mn, mx)) => {
                            for (k, x) in values.iter().enumerate() {
                                mn[k] = mn[k].min(*x);
                                mx[k] = mx[k].max(*x);
                            }
                        }
                        None => {
                            env.insert(name, (values.clone(), values));
                        }
                    }
                }
            }
            env
        }

        /// The frozen ROOT map: the complete cascade on the root box, exactly
        /// what every deferred child inherits today.
        fn root_map(graph: &GraphNetwork) -> HashMap<String, BoundedTensor> {
            let refine = child_interm_refine_alpha_config(&armed_config(120.0), None)
                .expect("armed config for the root collection");
            let exec_order = graph.exec_order().expect("exec order");
            let (map, source) = graph
                .collect_alpha_reference_bounds_with_engine_and_source(
                    &wider_root_box(),
                    &refine,
                    None,
                    exec_order,
                )
                .expect("root reference map");
            assert!(
                source.is_cgan_complete_crown_ibp(),
                "the root map must come from the complete cascade, as on cgan2026"
            );
            map
        }

        /// A schedule whose admission always grants `width` (the probe reports
        /// a 1 TiB headroom and the estimate is one byte), so a parallel arm
        /// really runs parallel and the comparison is not vacuous.
        fn schedule(seed_from_parent: bool, width: usize) -> ChildRefineSchedule {
            ChildRefineSchedule {
                seed_from_parent,
                requested_width: width,
                worker_estimate_bytes: 1,
                headroom_probe: || Some(1 << 40),
            }
        }

        fn armed_refine() -> crate::AlphaCrownConfig {
            child_interm_refine_alpha_config(&armed_config(120.0), None)
                .expect("armed child refinement config")
        }

        fn map_bits(map: &HashMap<String, BoundedTensor>) -> Vec<(String, Vec<(u32, u32)>)> {
            let mut rows: Vec<(String, Vec<(u32, u32)>)> = map
                .iter()
                .map(|(name, bound)| {
                    (
                        name.clone(),
                        bound
                            .lower()
                            .iter()
                            .zip(bound.upper().iter())
                            .map(|(lo, hi)| (lo.to_bits(), hi.to_bits()))
                            .collect(),
                    )
                })
                .collect();
            rows.sort_by(|a, b| a.0.cmp(&b.0));
            rows
        }

        #[test]
        fn child_seed_merge_is_shrink_only_and_contains_samples() {
            let graph = mixed_wide_then_narrow_graph();
            let root = root_map(&graph);
            let refine = armed_refine();
            let mut boxes = vec![child_box()];
            boxes.extend(random_child_boxes(3, 0x5eed_c0de));
            for (b, child) in boxes.iter().enumerate() {
                let mut domains = vec![FakeDomain {
                    input_bounds: Arc::new(child.clone()),
                    node_bounds_override: None,
                }];
                let tally = install_child_interm_refine_overrides_scheduled(
                    &refine,
                    &graph,
                    None,
                    Some(&root),
                    schedule(true, 1),
                    &[0],
                    &mut domains,
                );
                assert_eq!(
                    tally.installed, 1,
                    "box {b}: the seeded prepass must enter the cascade ({tally:?})"
                );
                assert_eq!(tally.seeded, 1, "box {b}: the cascade must run SEEDED");
                assert!(
                    tally.seed_nodes > 0,
                    "box {b}: the seed intersected no node — the lever did not engage \
                     and every assertion below would be vacuous ({tally:?})"
                );
                assert_eq!(
                    tally.seed_declined_nodes, 0,
                    "box {b}: two valid enclosures of one box can never be disjoint"
                );
                let installed = domains[0]
                    .node_bounds_override
                    .as_ref()
                    .expect("installed seeded map");

                // Shrink-only against the inherited root map on every node the
                // child baseline carries (the only nodes the seed may touch).
                let baseline = graph
                    .collect_forward_linear_bounds_dag_cached_for_typed_cgan(
                        child, &refine, None, None,
                    )
                    .expect("child forward-linear baseline");
                let mut compared = 0usize;
                for (name, root_bound) in &root {
                    let (Some(child_bound), true) =
                        (installed.get(name), baseline.contains_key(name))
                    else {
                        continue;
                    };
                    if child_bound.shape() != root_bound.shape() {
                        continue;
                    }
                    compared += 1;
                    for (k, (c, r)) in child_bound
                        .lower()
                        .iter()
                        .zip(root_bound.lower().iter())
                        .enumerate()
                    {
                        assert!(
                            *c >= *r,
                            "box {b}: node '{name}' row {k} lower {c} is BELOW the inherited \
                             root bound {r}; the seed must be shrink-only"
                        );
                    }
                    for (k, (c, r)) in child_bound
                        .upper()
                        .iter()
                        .zip(root_bound.upper().iter())
                        .enumerate()
                    {
                        assert!(
                            *c <= *r,
                            "box {b}: node '{name}' row {k} upper {c} is ABOVE the inherited \
                             root bound {r}; the seed must be shrink-only"
                        );
                    }
                }
                assert!(
                    compared >= 3,
                    "box {b}: shrink-only compared only {compared} nodes"
                );

                // Tighter is only meaningful if still SOUND.
                let env = sampled_envelope(&graph, child, 0xa11ce ^ b as u64);
                assert_contains(installed, &env, "seeded child");
            }
        }

        #[test]
        fn child_seed_declined_cascade_leaves_the_domain_untouched() {
            use crate::layers::LinearLayer;
            let mut graph = GraphNetwork::new();
            let fc = LinearLayer::new(
                Array2::from_shape_vec((2, 4), vec![0.9_f32, -0.4, 0.3, 0.7, -0.6, 0.5, 0.2, -0.8])
                    .expect("weights"),
                Some(Array1::from_vec(vec![0.1_f32, -0.05])),
            )
            .expect("linear layer");
            graph.add_node(GraphNode::from_input("fc", Layer::Linear(fc)));
            graph.add_node(GraphNode::new(
                "fc_relu",
                Layer::ReLU(ReLULayer),
                vec!["fc".to_string()],
            ));
            graph.set_output("fc_relu");
            let flat_child = BoundedTensor::new(
                ArrayD::from_shape_vec(IxDyn(&[4]), vec![-0.4_f32, -0.15, 0.05, -0.3])
                    .expect("child lower"),
                ArrayD::from_shape_vec(IxDyn(&[4]), vec![0.25_f32, 0.4, 0.5, 0.2])
                    .expect("child upper"),
            )
            .expect("child box");
            let inherited = graph.collect_node_bounds(&flat_child).expect("IBP map");
            let sentinel: Arc<HashMap<String, BoundedTensor>> = Arc::new(HashMap::new());
            let mut domains = vec![FakeDomain {
                input_bounds: Arc::new(flat_child),
                node_bounds_override: Some(Arc::clone(&sentinel)),
            }];
            let tally = install_child_interm_refine_overrides_scheduled(
                &armed_refine(),
                &graph,
                None,
                Some(&inherited),
                schedule(true, 1),
                &[0],
                &mut domains,
            );
            assert_eq!(
                tally.installed, 0,
                "a refused cascade installs nothing ({tally:?})"
            );
            assert_eq!(tally.declined, 1);
            assert_eq!(tally.seeded, 0);
            assert!(
                Arc::ptr_eq(
                    domains[0]
                        .node_bounds_override
                        .as_ref()
                        .expect("override retained"),
                    &sentinel
                ),
                "the domain must keep its EXACT historical override object"
            );
        }

        #[test]
        fn child_seed_without_any_inherited_map_is_the_unseeded_cascade() {
            let refine = armed_refine();
            let run = |seeded: bool| {
                // A fresh graph per arm, so neither can be served from the
                // other's collection cache.
                let graph = mixed_wide_then_narrow_graph();
                let mut domains = vec![FakeDomain {
                    input_bounds: Arc::new(child_box()),
                    node_bounds_override: None,
                }];
                let tally = install_child_interm_refine_overrides_scheduled(
                    &refine,
                    &graph,
                    None,
                    None,
                    schedule(seeded, 1),
                    &[0],
                    &mut domains,
                );
                let map = domains[0]
                    .node_bounds_override
                    .as_ref()
                    .expect("installed map")
                    .as_ref()
                    .clone();
                (tally, map)
            };
            let (seeded_tally, seeded_map) = run(true);
            let (plain_tally, plain_map) = run(false);
            assert_eq!(seeded_tally.seed_unavailable, 1, "{seeded_tally:?}");
            assert_eq!(seeded_tally.seeded, 0);
            assert_eq!(plain_tally.seed_unavailable, 0);
            assert_eq!(
                map_bits(&seeded_map),
                map_bits(&plain_map),
                "with nothing to seed from, the seeded arm must BE the unseeded cascade"
            );
        }

        #[test]
        fn parallel_child_refine_maps_bit_identical_to_serial() {
            let refine = armed_refine();
            let root = root_map(&mixed_wide_then_narrow_graph());
            let boxes = random_child_boxes(6, 0xbead_f00d);
            let ibp_graph = mixed_wide_then_narrow_graph();
            let overrides: Vec<Option<Arc<HashMap<String, BoundedTensor>>>> = boxes
                .iter()
                .enumerate()
                .map(|(i, child)| {
                    // Half the domains carry a valid child-local override (the
                    // child's own IBP map), as clipped cgan children do.
                    (i % 2 == 1).then(|| {
                        Arc::new(ibp_graph.collect_node_bounds(child).expect("child IBP map"))
                    })
                })
                .collect();
            for seeded in [false, true] {
                let run = |width: usize| {
                    let graph = mixed_wide_then_narrow_graph();
                    let mut domains: Vec<FakeDomain> = boxes
                        .iter()
                        .zip(&overrides)
                        .map(|(child, existing)| FakeDomain {
                            input_bounds: Arc::new(child.clone()),
                            node_bounds_override: existing.clone(),
                        })
                        .collect();
                    let indices: Vec<usize> = (0..domains.len()).collect();
                    let tally = install_child_interm_refine_overrides_scheduled(
                        &refine,
                        &graph,
                        None,
                        Some(&root),
                        schedule(seeded, width),
                        &indices,
                        &mut domains,
                    );
                    let maps: Vec<Vec<(String, Vec<(u32, u32)>)>> = domains
                        .iter()
                        .map(|d| map_bits(d.node_bounds_override.as_ref().expect("installed")))
                        .collect();
                    (tally, maps)
                };
                let (serial, serial_maps) = run(1);
                let (parallel, parallel_maps) = run(4);
                assert_eq!(serial.installed, 6, "seeded={seeded}: {serial:?}");
                assert_eq!(serial.max_width, 1);
                assert_eq!(
                    parallel.max_width, 4,
                    "seeded={seeded}: the parallel arm never ran wider than 1, so this \
                     comparison would be vacuous ({parallel:?})"
                );
                for (field, a, b) in [
                    ("installed", serial.installed, parallel.installed),
                    ("declined", serial.declined, parallel.declined),
                    (
                        "merged_into_existing_override",
                        serial.merged_into_existing_override,
                        parallel.merged_into_existing_override,
                    ),
                    ("seeded", serial.seeded, parallel.seeded),
                    ("seed_nodes", serial.seed_nodes, parallel.seed_nodes),
                    (
                        "seed_tightened_elems",
                        serial.seed_tightened_elems,
                        parallel.seed_tightened_elems,
                    ),
                    (
                        "unstable_rows_seeded",
                        serial.unstable_rows_seeded,
                        parallel.unstable_rows_seeded,
                    ),
                ] {
                    assert_eq!(a, b, "seeded={seeded}: tally field {field} differs");
                }
                if seeded {
                    assert_eq!(serial.seeded, 6, "every child must run seeded");
                }
                assert_eq!(
                    serial_maps, parallel_maps,
                    "seeded={seeded}: parallel tranches must publish maps BIT-IDENTICAL to \
                     the serial reference"
                );
            }
        }

        #[test]
        fn fresh_clip_with_domain_map_planes_contain_samples() {
            let graph = mixed_wide_then_narrow_graph();
            let root = root_map(&graph);
            let refine = armed_refine();
            let spec = Array2::from_shape_vec((2, 3), vec![1.0_f32, -1.0, 0.0, 0.0, 1.0, -1.0])
                .expect("spec");
            let mut boxes = vec![child_box()];
            boxes.extend(random_child_boxes(2, 0xc11f));
            let mut planes_checked = 0usize;
            for (b, child) in boxes.iter().enumerate() {
                // The refined map exactly as the deferred rebound retains it.
                let mut domains = vec![FakeDomain {
                    input_bounds: Arc::new(child.clone()),
                    node_bounds_override: None,
                }];
                let tally = install_child_interm_refine_overrides_scheduled(
                    &refine,
                    &graph,
                    None,
                    Some(&root),
                    schedule(true, 1),
                    &[0],
                    &mut domains,
                );
                assert_eq!(tally.installed, 1, "box {b}: {tally:?}");
                let domain_map = domains[0]
                    .node_bounds_override
                    .clone()
                    .expect("refined domain map");
                let points = sample_points(child, SAMPLES, 0x9a7e ^ b as u64);
                let truths: Vec<(Vec<f32>, Vec<f64>)> = points
                    .into_iter()
                    .map(|point| {
                        let acts = activations_at(&graph, &point);
                        let y = acts.get("narrow_relu").expect("output activation");
                        let s: Vec<f64> = spec
                            .rows()
                            .into_iter()
                            .map(|row| {
                                row.iter()
                                    .zip(y.iter())
                                    .map(|(c, v)| f64::from(*c) * f64::from(*v))
                                    .sum()
                            })
                            .collect();
                        (point, s)
                    })
                    .collect();
                for (label, child_map) in [
                    ("root-only planes (key dark)", None),
                    ("domain-map planes (key armed)", Some(domain_map.as_ref())),
                ] {
                    let (bounds, linear) = compute_crown_or_ibp_bounds_with_node_bounds(
                        &graph,
                        child,
                        &spec,
                        None,
                        Some(&root),
                        child_map,
                        None,
                        None,
                        None,
                        None,
                        true,
                    )
                    .expect("fresh-clip bounds");
                    planes_checked += usize::from(linear.is_some());
                    let tol = 1e-4_f64;
                    for (point, s) in &truths {
                        for (i, truth) in s.iter().enumerate() {
                            if let Some(linear) = linear.as_ref() {
                                let mut lower = f64::from(linear.lower_b()[i]);
                                let mut upper = f64::from(linear.upper_b()[i]);
                                for (j, x) in point.iter().enumerate() {
                                    let x = f64::from(*x);
                                    lower += f64::from(linear.lower_a()[[i, j]]) * x;
                                    upper += f64::from(linear.upper_a()[[i, j]]) * x;
                                    if let Some(err) = linear.lower_a_err() {
                                        lower -= f64::from(err[[i, j]]) * x.abs();
                                    }
                                    if let Some(err) = linear.upper_a_err() {
                                        upper += f64::from(err[[i, j]]) * x.abs();
                                    }
                                }
                                assert!(
                                    lower <= truth + tol && truth - tol <= upper,
                                    "box {b}, {label}: spec row {i} plane [{lower}, {upper}] \
                                     EXCLUDES the true value {truth} at {point:?} — a clip \
                                     built on these planes would discard a reachable point"
                                );
                            }
                            let lo = f64::from(bounds.lower().iter().nth(i).copied().expect("lo"));
                            let hi = f64::from(bounds.upper().iter().nth(i).copied().expect("hi"));
                            assert!(
                                lo <= truth + tol && truth - tol <= hi,
                                "box {b}, {label}: spec row {i} bound [{lo}, {hi}] EXCLUDES \
                                 the true value {truth}"
                            );
                        }
                    }
                }
            }
            assert!(
                planes_checked > 0,
                "the bounder returned no affine planes on any arm, so the plane-containment \
                 check above was VOID, not a pass"
            );
        }

        #[test]
        fn child_seed_keys_dark_by_default_leave_the_historical_schedule() {
            use super::super::super::super::child_seed::{
                any_child_seed_key_armed, child_refine_plan, retain_map_for_fresh_clip,
            };
            let default = crate::beta_crown::config::BetaCrownConfig::default();
            assert!(
                !default
                    .alpha_config
                    .cgan_child_interm_refine_seed_from_parent
            );
            assert_eq!(default.alpha_config.cgan_child_interm_refine_parallel, 0);
            assert!(!default.alpha_config.cgan_child_interm_refine_fresh_clip_map);
            assert!(!any_child_seed_key_armed(&default));
            assert!(
                child_refine_plan(&default, None, None).is_none(),
                "no per-child budget: no prepass, no schedule, no collector work"
            );
            assert!(!retain_map_for_fresh_clip(&default, true));

            // The historical prepass armed, the new keys dark: the plan must be
            // the serial, unseeded schedule and the rebound must still clear.
            let armed = armed_config(120.0);
            let (_refine, plan) =
                child_refine_plan(&armed, None, None).expect("armed prepass plan");
            assert!(!plan.seed_from_parent);
            assert_eq!(plan.requested_width, 1);
            assert!(!any_child_seed_key_armed(&armed));
            assert!(!retain_map_for_fresh_clip(&armed, true));

            // A key set WITHOUT the per-child budget must stay inert.
            let mut keys_only = crate::beta_crown::config::BetaCrownConfig::default();
            keys_only.alpha_config.cgan_complete_crown_ibp_root = true;
            keys_only
                .alpha_config
                .cgan_child_interm_refine_seed_from_parent = true;
            keys_only.alpha_config.cgan_child_interm_refine_parallel = 4;
            keys_only
                .alpha_config
                .cgan_child_interm_refine_fresh_clip_map = true;
            assert!(any_child_seed_key_armed(&keys_only));
            assert!(child_refine_plan(&keys_only, None, None).is_none());
            assert!(!retain_map_for_fresh_clip(&keys_only, false));
        }
    }
}
