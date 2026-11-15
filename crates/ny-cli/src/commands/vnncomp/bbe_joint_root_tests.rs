// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #bbe-joint-root (ENGINE_PROGRAM M5a) at the CLI boundary — the only crate
//! that sees BOTH sides of the search/certify split.
//!
//! * `root_search_is_dark_without_the_key_and_typed_with_it` — the typed key
//!   `margin_row.root_search` resolves to the bit-identical `ascent` without
//!   the key and arrives through the scored run's own preset projection
//!   (`preset_delivered_config_value`, NO environment) with `Source::Config`,
//!   all the way into the lane's frozen `RootCertifyOnceRuntimePolicy`.
//! * `real_cpu_f32_search_through_the_ingest_gate_is_certified_and_sound` —
//!   the REAL `ny_bbe::CpuF32Backend`, registered exactly as `main.rs` does,
//!   driven through the lane's public certify-once entry on small twin-wall
//!   nets: the search engages (`[bbe]` telemetry, no refusal, iterations run,
//!   every trunk layer and the head hinted), and whatever it proposes, the
//!   published root is never below the certified incumbent, stays below every
//!   sampled margin (negative on the falsified runner-up property), and the
//!   kept root's boxes and chords never move.

use std::sync::Arc;

use ndarray::Array2;

use ny_propagate::margin_row::root_certify_once::{maybe_certify_once_root, Kept};
use ny_propagate::margin_row::{
    BackwardIntermRuntimePolicy, MarginRowRuntimePolicy, RootAlphaRuntimePolicy,
    RootCertifyOnceRuntimePolicy, RootGates, RootSearch, RoundMode, TwinNet, TwinOpSpec, TwinSpec,
};

use super::preset_delivered_config_value;

fn projected_levers(preset: &crate::preset::PresetConfig) -> ny_levers::LeverSet {
    let serialized = serde_json::to_value(preset).expect("preset serializes");
    let captured = ny_levers::RawLeverInputs::capture_with(ny_levers::all(), |_| None);
    ny_levers::LeverSet::resolve_layered(ny_levers::all(), &captured, |decl| {
        preset_delivered_config_value(&serialized, decl)
    })
    .expect("the projected preset must match its declarations")
}

#[test]
fn root_search_is_dark_without_the_key_and_typed_with_it() {
    let decl = &ny_levers::decls::margin_row::MARGIN_ROW_ROOT_SEARCH;
    let dark = projected_levers(&crate::preset::PresetConfig::default());
    let resolved = dark.resolved(decl).expect("the lever is declared");
    assert_eq!(resolved.value.as_str(), Some(RootSearch::Ascent.token()));
    assert_eq!(resolved.source, ny_levers::Source::Default);
    assert_eq!(
        MarginRowRuntimePolicy::from_levers(&dark)
            .expect("policy")
            .root_certify_once
            .search,
        RootSearch::Ascent
    );

    let mut preset = crate::preset::PresetConfig::default();
    preset.margin_row.root_search = Some(RootSearch::JointF32);
    let typed = projected_levers(&preset);
    let resolved = typed.resolved(decl).expect("the lever is declared");
    assert_eq!(resolved.value.as_str(), Some(RootSearch::JointF32.token()));
    assert_eq!(resolved.source, ny_levers::Source::Config);
    assert_ne!(resolved.source, ny_levers::Source::LegacyEnv);
    // The yaml route reaches the lane's frozen policy with zero `NY_*` set.
    assert_eq!(
        MarginRowRuntimePolicy::from_levers(&typed)
            .expect("policy")
            .root_certify_once
            .search,
        RootSearch::JointF32
    );
}

// ---------------------------------------------------------------------------
//  The real backend through the real ingest gate
// ---------------------------------------------------------------------------

struct Lcg(u64);

impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }

    fn vec(&mut self, n: usize, scale: f64) -> Vec<f64> {
        (0..n).map(|_| scale * (2.0 * self.unit() - 1.0)).collect()
    }
}

/// The margin-row twin-wall shape: 2x6x6 input, stride-2 conv, a residual
/// block, two trunk ReLUs, a 5-wide head ReLU, 4 classes.
fn twin_spec(seed: u64, scale: f64) -> TwinSpec {
    let mut rng = Lcg(seed);
    let conv1 = TwinOpSpec::Conv {
        input: 0,
        weight: rng.vec(4 * 2 * 3 * 3, scale),
        bias: rng.vec(4, scale),
        bias_err: vec![0.0; 4],
        weight_rel_err: 1e-15,
        kernel: (4, 2, 3, 3),
        stride: (2, 2),
        pads: (1, 1, 1, 1),
        ishape: (2, 6, 6),
        oshape: (4, 3, 3),
    };
    let conv2 = TwinOpSpec::Conv {
        input: 2,
        weight: rng.vec(4 * 4 * 3 * 3, scale),
        bias: rng.vec(4, scale),
        bias_err: vec![1e-18; 4],
        weight_rel_err: 1e-15,
        kernel: (4, 4, 3, 3),
        stride: (1, 1),
        pads: (1, 1, 1, 1),
        ishape: (4, 3, 3),
        oshape: (4, 3, 3),
    };
    TwinSpec {
        n_in: 72,
        ops: vec![
            conv1,
            TwinOpSpec::Relu { input: 1 },
            conv2,
            TwinOpSpec::Add { lhs: 3, rhs: 2 },
            TwinOpSpec::Relu { input: 4 },
            TwinOpSpec::Flatten { input: 5 },
            TwinOpSpec::Gemm {
                input: 6,
                weight: rng.vec(5 * 36, scale),
                bias: rng.vec(5, scale),
                bias_err: Vec::new(),
                weight_rel_err: 0.0,
                shape: (5, 36),
            },
            TwinOpSpec::Relu { input: 7 },
            TwinOpSpec::Gemm {
                input: 8,
                weight: rng.vec(4 * 5, scale),
                bias: rng.vec(4, scale),
                bias_err: Vec::new(),
                weight_rel_err: 0.0,
                shape: (4, 5),
            },
        ],
    }
}

/// Class scores (last tensor) at the columns of `x`.
fn scores(net: &TwinNet, x: &Array2<f64>) -> Array2<f64> {
    let (y, _) = net
        .forward_points(x, &std::collections::BTreeMap::new())
        .expect("forward");
    let (w2, b2, (n_out, n_y)) = net.gemm2();
    Array2::from_shape_fn((n_out, x.ncols()), |(o, b)| {
        let mut s = b2[o];
        for k in 0..n_y {
            s += w2[o * n_y + k] * y[[k, b]].max(0.0);
        }
        s
    })
}

#[test]
fn real_cpu_f32_search_through_the_ingest_gate_is_certified_and_sound() {
    // Exactly the registration `main.rs` performs (first install wins; the test
    // binary never runs `main`, so this is the only one).
    ny_propagate::margin_row::set_root_search_backend_factory(|| {
        Some(Arc::new(ny_bbe::CpuF32Backend::with_threads(2))
            as Arc<dyn ny_core::search::SearchBackend>)
    });
    let policy = RootCertifyOnceRuntimePolicy {
        enabled: true,
        iters: 20,
        refresh: 0,
        secs: 600.0,
        search: RootSearch::JointF32,
    };
    let (mut runs, mut iterated, mut accepted, mut falsified) = (0usize, 0usize, 0usize, 0usize);
    for seed in 0..6u64 {
        let spec = twin_spec(0xBBE0_0100 + seed, 0.4 + 0.1 * (seed % 3) as f64);
        let net = TwinNet::compile(&spec).expect("compile");
        let half = 0.3 + 0.05 * (seed % 2) as f64;
        let mut rng = Lcg(seed ^ 0x77);
        let centre: Vec<f64> = (0..72).map(|_| rng.unit() - 0.5).collect();
        let lo: Vec<f64> = centre.iter().map(|c| c - half).collect();
        let hi: Vec<f64> = centre.iter().map(|c| c + half).collect();
        // Sampled points: centre, both corners, random corners and interior.
        let x = Array2::from_shape_fn((72, 96), |(i, b)| match b {
            0 => centre[i],
            1 => lo[i],
            2 => hi[i],
            b if b % 2 == 0 => {
                if rng.unit() < 0.5 {
                    lo[i]
                } else {
                    hi[i]
                }
            }
            _ => lo[i] + rng.unit() * (hi[i] - lo[i]),
        });
        let s = scores(&net, &x);
        let order: Vec<usize> = {
            let mut o: Vec<usize> = (0..4).collect();
            o.sort_by(|&a, &b| s[[b, 0]].total_cmp(&s[[a, 0]]));
            o
        };
        let (winner, runner_up) = (order[0], order[1]);
        for target in [winner, runner_up] {
            let adv: Vec<usize> = (0..4).filter(|&j| j != target).collect();
            let before = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
            let mut root =
                RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
            let report = maybe_certify_once_root(
                &net,
                &mut root,
                false,
                target,
                &adv,
                None,
                None,
                policy,
                RootAlphaRuntimePolicy::default(),
                BackwardIntermRuntimePolicy::default(),
            )
            .expect("an armed policy runs the phase");
            runs += 1;
            let ctx = format!("seed {seed} target {target}");
            let bbe = report
                .bbe
                .as_ref()
                .expect("joint_f32 carries [bbe] telemetry");
            if report.stop != "closed" {
                assert_eq!(bbe.backend, "cpu_f32", "{ctx}: the registered backend ran");
                assert!(bbe.refused.is_none(), "{ctx}: refused {:?}", bbe.refused);
                assert_eq!(bbe.hint_layers, 2, "{ctx}: both trunk layers hinted");
                assert!(bbe.hint_head, "{ctx}: the head hinted");
                assert!(bbe.iters <= policy.iters);
                iterated += usize::from(bbe.iters > 0);
            }
            assert!(
                report.published >= report.incumbent,
                "{ctx}: published {} below the incumbent {}",
                report.published,
                report.incumbent
            );
            let truth = (0..x.ncols())
                .flat_map(|b| adv.iter().map(move |&j| (b, j)))
                .map(|(b, j)| s[[target, b]] - s[[j, b]])
                .fold(f64::INFINITY, f64::min);
            assert!(
                report.published <= truth + 1e-9,
                "{ctx}: certified {} ABOVE the sampled margin {truth}",
                report.published
            );
            if target == runner_up {
                assert!(truth < 0.0, "{ctx}: the runner-up property is falsified");
                assert!(
                    report.published < 0.0,
                    "{ctx}: a FALSE property must never certify"
                );
                falsified += 1;
            }
            for (li, (old, new)) in before.layers.iter().zip(&root.layers).enumerate() {
                assert_eq!(old.l, new.l, "{ctx}: l moved at layer {li}");
                assert_eq!(old.u, new.u, "{ctx}: u moved at layer {li}");
                assert_eq!(old.s, new.s, "{ctx}: s moved at layer {li}");
                assert_eq!(old.c, new.c, "{ctx}: c moved at layer {li}");
                for j in 0..new.n {
                    assert!(
                        (0.0..=1.0).contains(&new.alpha[j]),
                        "{ctx}: slope out of range"
                    );
                    assert_eq!(new.ms[j], new.alpha[j].max(new.s[j]), "{ctx}: ms");
                }
            }
            accepted += usize::from(matches!(report.kept, Kept::Alpha | Kept::Refresh));
        }
    }
    eprintln!(
        "[bbe-joint-root-cli-test] runs={runs} iterated={iterated} accepted={accepted} \
         falsified={falsified}"
    );
    assert!(iterated > 0, "the real search never iterated: VOID");
    assert!(falsified > 0, "no falsified property was exercised: VOID");
}
